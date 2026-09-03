"""
Follow-up to experiments/reasoning_stack.py: re-score the SAME three frozen
arms under a quote-normalized grounding check, to isolate whether structured
reasoning helps once a benign scoring artifact is removed.

WHAT THE PRIOR RUN FOUND. At n=2000, arm0 (shipped) scored 35.2% unsupported,
arm1 41.4% (tie), arm2 42.4% (CI-separated, worse). But the failures were
overwhelmingly MECHANICAL, and a majority of the failing spans were not
over-claims at all: they were spans that ARE in the note, returned wrapped in
literal quote characters, which ra.verifier's exact-substring test
(`span not in desc`) marked unsupported. 55% of arm1's bad spans and 67% of
arm2's were that artifact, against 0% for the baseline. So the prior headline
charged arms 1 and 2 for a formatting habit, not for ungrounded reasoning.

WHAT THIS RUN CHANGES -- THE CHECK, NOT THE ARMS. The arms are frozen: their
prompts are untouched and their stances are REUSED VERBATIM from the prior
run's cache, so not a single new stance LLM call is made. The only thing that
moves is the span test in the mechanical layer, which now uses
scripts/eval_agent._is_grounded -- the project's already-vetted normalizer
(strip surrounding quote characters, then case-insensitive substring), the
same test scripts/eval_agent.py --validate already uses to report grounding.

This is a CORRECTION, not a relaxation. The inner text must still be a real
substring of the applicant's note. A genuine paraphrase -- different words,
not just quoting or casing -- still fails, exactly as before. Note the
normalizer is slightly broader than the artifact category the prior run
measured: it also forgives re-casing and smart quotes, so the count of spans
it newly accepts can legitimately exceed the prior "quote_wrapping_artifact"
tally. That count is reported, with examples, so the correction is auditable
rather than taken on trust.

APPLIED IDENTICALLY TO ALL THREE ARMS, baseline included. The baseline is the
built-in control: it had ~0 quote-wrap failures, so its normalized rate should
be essentially unchanged. If the baseline moves materially, the normalizer is
being lenient rather than corrective, and this run says so instead of hiding
it.

Everything else is held exactly as shipped: the policy-id membership check is
UNCHANGED (a cited id must still be in the set ra._retrieve_policy returns),
the LLM grounding auditor is UNCHANGED (byte-identical
ra.VERIFIER_SYSTEM_PROMPT / ra.VERIFIER_USER_TEMPLATE via
ra._call_verifier_llm), retrieval is the shipped in-process BM25, and the
model is the same at temperature 0.

LLM-CALL REUSE, AND WHY IT'S SOUND. Normalization can only ADD mechanical
passes -- a span passing the raw test still passes the normalized one -- so
any stance that reached the LLM layer before reaches it again with byte-
identical inputs. At temperature 0 that call would reproduce its own prior
answer, so the prior verdict is reused. A fresh LLM call is made ONLY for a
stance that newly passes the mechanical layer under normalization, i.e. one
the raw check never let through. Both counts are printed.

SCOPE. This does NOT change the shipping decision. The baseline still ships;
two-call reasoning still costs ~2.3x the latency of the baseline whatever
this re-scoring says about its grounding. This isolates the research question
and nothing else.

Does not modify agent/reconciler_agent.py, experiments/reasoning_stack.py, or
any arm prompt.
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
import reconciler_agent as ra  # noqa: E402
import llm_client as lc  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from eval_agent import (  # noqa: E402
    _QUOTE_CHARS, _is_grounded, wilson_ci, UNDERPOWERED_N,
    load_test_population, base_rate_sample,
)

K = 4
RETRIEVAL_CONFIG_PATH = ROOT / "config" / "retrieval.json"
CHECKPOINT_EVERY = 50
ARM_NAMES = ["arm0_baseline", "arm1_structured", "arm2_two_call"]

PRIOR_JSON = ROOT / "experiments" / "reasoning_comparison_n2000.json"
PRIOR_CACHE = ROOT / "experiments" / "reasoning_stance_cache_n2000.pkl"


# ---------------------------------------------------------------------------
# The normalized mechanical layer.
#
# Mirrors agent/reconciler_agent.py verifier() line for line -- same order
# (neutral short-circuit, then spans, then policy ids, then the LLM layer),
# same verdict/source vocabulary, same byte-identical verifier prompt -- with
# exactly ONE substitution: `span not in desc` becomes
# `not _is_grounded(span, desc)`.
#
# The safe-direction stance downgrade the shipped node applies is deliberately
# NOT reproduced: nothing here routes a decision, and the prior experiment
# recorded only the verdict too, so the two runs stay directly comparable.
# ---------------------------------------------------------------------------
def verify_normalized(stance_out: dict, desc: str, reuse_verdict: str = None) -> tuple[dict, bool]:
    """Returns (result, made_fresh_llm_call).

    reuse_verdict, when given, is this stance's verdict from the prior run's
    LLM layer -- reused rather than re-requested because the inputs are
    byte-identical and the model runs at temperature 0.
    """
    stance = stance_out["stance"]
    evidence = stance_out.get("stance_evidence") or []
    policy_ids = stance_out.get("stance_policy_ids") or []
    rationale = stance_out.get("stance_rationale", "")

    if stance == "neutral":
        return {"verifier_verdict": "skipped_neutral", "verifier_source": "skipped",
                "llm_reused": False}, False

    # Layer 1a -- spans. THE ONE CHANGED LINE.
    bad_span = next((span for span in evidence if not _is_grounded(span, desc)), None)
    if bad_span is not None:
        return {"verifier_verdict": "unsupported", "verifier_source": "mechanical",
                "mechanical_cause": "span", "llm_reused": False}, False

    # Layer 1b -- policy-id membership. UNCHANGED from shipped.
    retrieved_ids = {p["id"] for p in (ra._retrieve_policy(desc) if desc else [])}
    bad_policy = next((pid for pid in policy_ids if pid not in retrieved_ids), None)
    if bad_policy is not None:
        return {"verifier_verdict": "unsupported", "verifier_source": "mechanical",
                "mechanical_cause": "policy_id", "llm_reused": False}, False

    # Layer 2 -- the LLM grounding auditor. UNCHANGED from shipped.
    if reuse_verdict is not None:
        return {"verifier_verdict": reuse_verdict, "verifier_source": "llm",
                "llm_reused": True}, False

    evidence_block = "\n".join(f'- "{e}"' for e in evidence) or "(none quoted)"
    policy_block = "\n".join(
        f'[{pid}] {ra.POLICY_CORPUS_BY_ID.get(pid, "(policy text not found)")}'
        for pid in policy_ids
    ) or "(none cited)"
    out = ra._call_verifier_llm(
        ra.VERIFIER_SYSTEM_PROMPT,
        ra.VERIFIER_USER_TEMPLATE.format(
            stance=stance, evidence_block=evidence_block,
            policy_block=policy_block, rationale=rationale,
        ),
    )
    return {"verifier_verdict": out["verdict"], "verifier_source": "llm",
            "llm_reused": False}, True


def newly_accepted_spans(stance_out: dict, desc: str) -> list:
    """Spans the raw exact-substring check rejected but the normalizer
    accepts -- the audit trail for the correction."""
    return [
        span for span in (stance_out.get("stance_evidence") or [])
        if isinstance(span, str) and span not in desc and _is_grounded(span, desc)
    ]


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
def unsupported_stats(recs: list, verdict_key: str) -> dict:
    """unsupported / non-neutral, same denominator definition as the prior
    run's unsupported_stats and retrieval_stack's before it."""
    non_neutral = [r for r in recs if r["stance"] != "neutral"]
    n = len(non_neutral)
    k = sum(1 for r in non_neutral if r[verdict_key] == "unsupported")
    if n == 0:
        return {"n": 0, "k": 0, "rate": None, "ci_low": None, "ci_high": None, "underpowered": True}
    ci_low, ci_high = wilson_ci(k, n)
    return {"n": n, "k": k, "rate": k / n, "ci_low": ci_low, "ci_high": ci_high,
            "underpowered": n < UNDERPOWERED_N}


def llm_layer_stats(recs: list, verdict_key: str, source_key: str) -> dict:
    non_neutral = [r for r in recs if r["stance"] != "neutral"]
    reached = [r for r in non_neutral if r[source_key] == "llm"]
    n = len(reached)
    k = sum(1 for r in reached if r[verdict_key] == "unsupported")
    out = {"n_reached_llm": n, "k_llm_unsupported": k,
           "mech_pass_rate": n / len(non_neutral) if non_neutral else None}
    if n == 0:
        return {**out, "rate": None, "ci_low": None, "ci_high": None}
    ci_low, ci_high = wilson_ci(k, n)
    return {**out, "rate": k / n, "ci_low": ci_low, "ci_high": ci_high}


def cis_overlap(a: dict, b: dict) -> bool:
    return a["ci_low"] <= b["ci_high"] and b["ci_low"] <= a["ci_high"]


def arm_summary(recs: list) -> dict:
    non_neutral = [r for r in recs if r["stance"] != "neutral"]
    confident = [r for r in non_neutral if r["stance_confidence"] >= ra.CONF_THRESHOLD]
    verdicts = ["supported", "unsupported", "unclear", "skipped_neutral"]
    return {
        "n_notes": len(recs),
        "n_neutral": len(recs) - len(non_neutral),
        "neutral_rate": (len(recs) - len(non_neutral)) / len(recs) if recs else None,
        "n_non_neutral": len(non_neutral),
        "raw": {
            "unsupported": unsupported_stats(recs, "verifier_verdict_raw"),
            "llm_layer": llm_layer_stats(recs, "verifier_verdict_raw", "verifier_source_raw"),
            "n_confident_supported": sum(
                1 for r in confident if r["verifier_verdict_raw"] == "supported"),
            "verdict_breakdown": {
                v: sum(1 for r in recs if r["verifier_verdict_raw"] == v) for v in verdicts},
            "n_mechanical_fail": sum(1 for r in recs if r["verifier_source_raw"] == "mechanical"),
        },
        "normalized": {
            "unsupported": unsupported_stats(recs, "verifier_verdict_norm"),
            "llm_layer": llm_layer_stats(recs, "verifier_verdict_norm", "verifier_source_norm"),
            "n_confident_supported": sum(
                1 for r in confident if r["verifier_verdict_norm"] == "supported"),
            "verdict_breakdown": {
                v: sum(1 for r in recs if r["verifier_verdict_norm"] == v) for v in verdicts},
            "n_mechanical_fail": sum(1 for r in recs if r["verifier_source_norm"] == "mechanical"),
            "n_mechanical_fail_span": sum(
                1 for r in recs if r.get("mechanical_cause_norm") == "span"),
            "n_mechanical_fail_policy_id": sum(
                1 for r in recs if r.get("mechanical_cause_norm") == "policy_id"),
        },
        # The correction, quantified.
        "n_records_flipped_to_mech_pass": sum(
            1 for r in recs
            if r["verifier_source_raw"] == "mechanical" and r["verifier_source_norm"] != "mechanical"),
        # Split deliberately: a newly-accepted span on a NEUTRAL stance cannot
        # move the score at all (neutral short-circuits to skipped_neutral
        # before the span test runs). Counting those together with the
        # score-relevant ones inflates the apparent size of the correction,
        # and inflates it most for the arm that is most often neutral -- the
        # baseline. Only the non-neutral count belongs in a comparison.
        "n_spans_newly_accepted_scoring": sum(
            r["n_spans_newly_accepted"] for r in recs if r["stance"] != "neutral"),
        "n_spans_newly_accepted_all": sum(r["n_spans_newly_accepted"] for r in recs),
        # NOTE: counts LLM calls made by THIS process only. A resumed run
        # serves earlier calls from the verifier cache, so this undercounts
        # the calls the re-scoring actually required. The caching is sound --
        # the verdicts are real either way -- but for "how much new work did
        # this analysis need", read n_records_flipped_to_mech_pass instead,
        # which is resume-independent.
        "n_fresh_llm_calls_this_run": sum(1 for r in recs if r["fresh_llm_call"]),
    }


def parse_args():
    p = argparse.ArgumentParser(
        description="Re-score reasoning_stack's frozen arms under a quote-normalized "
                    "grounding check. Reuses prior stances; makes no new stance calls.")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--track", action="store_true",
                   help="log one MLflow run per arm to ./mlruns (off by default)")
    p.add_argument("--experiment", default="reasoning_ab_spanfix")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def print_report(summary: dict, n_notes: int, model: str, audit: dict) -> None:
    def pct(x):
        return "n/a" if x is None else f"{100 * x:.1f}%"

    def ci(s):
        return "n/a" if s["rate"] is None else f"[{100*s['ci_low']:.1f},{100*s['ci_high']:.1f}]"

    print("\n" + "=" * 112)
    print(f"QUOTE-NORMALIZED RE-SCORING -- OLD vs NEW  (n={n_notes} notes, arms frozen, "
          f"stances reused verbatim)")
    print("=" * 112)
    print(f"{'arm':17s} {'RAW unsup%':>11s} {'raw k/n':>11s} {'raw 95% CI':>15s}   "
          f"{'NORM unsup%':>11s} {'norm k/n':>11s} {'norm 95% CI':>15s} {'delta':>8s}")
    for name, s in summary.items():
        r, nz = s["raw"]["unsupported"], s["normalized"]["unsupported"]
        delta = (nz["rate"] - r["rate"]) if (r["rate"] is not None and nz["rate"] is not None) else None
        raw_kn = "{}/{}".format(r["k"], r["n"])
        norm_kn = "{}/{}".format(nz["k"], nz["n"])
        delta_str = "n/a" if delta is None else "{:+.1f}pp".format(100 * delta)
        print(f"{name:17s} {pct(r['rate']):>11s} {raw_kn:>11s} {ci(r):>15s}   "
              f"{pct(nz['rate']):>11s} {norm_kn:>11s} {ci(nz):>15s} {delta_str:>8s}")

    print("\n-- Sanity check: stances were reused, so these MUST be identical to the prior run --")
    ok = True
    for name, s in summary.items():
        prior = audit["prior_summary"][name]
        neutral_match = s["n_neutral"] == prior["n_neutral"]
        nn_match = s["n_non_neutral"] == prior["n_non_neutral"]
        raw_match = s["raw"]["unsupported"]["k"] == prior["k_unsupported"]
        ok = ok and neutral_match and nn_match and raw_match
        flag = "OK " if (neutral_match and nn_match and raw_match) else "BUG"
        print(f"  [{flag}] {name:17s} neutral {s['n_neutral']} vs {prior['n_neutral']}  |  "
              f"non-neutral {s['n_non_neutral']} vs {prior['n_non_neutral']}  |  "
              f"raw unsupported k {s['raw']['unsupported']['k']} vs {prior['k_unsupported']}")
    if not ok:
        print("\n  STOP -- the re-scored run disagrees with the prior run on something that "
              "reusing\n  stances cannot change. Something other than the grounding check "
              "differs. Not interpreting.")
        return

    print("\n-- What the correction actually did, per arm --")
    for name, s in summary.items():
        r, nz = s["raw"], s["normalized"]
        print(f"\n  {name}")
        print(f"    mechanical failures          {r['n_mechanical_fail']} raw  ->  "
              f"{nz['n_mechanical_fail']} normalized   "
              f"({s['n_records_flipped_to_mech_pass']} records flipped to a mechanical pass)")
        print(f"    mechanical pass rate         "
              f"{pct(r['llm_layer']['mech_pass_rate'])}  ->  {pct(nz['llm_layer']['mech_pass_rate'])}")
        print(f"    remaining mechanical fails   span={nz['n_mechanical_fail_span']}  "
              f"policy_id={nz['n_mechanical_fail_policy_id']}")
        print(f"    spans newly accepted         {s['n_spans_newly_accepted_scoring']} on "
              f"non-neutral stances (score-relevant); "
              f"{s['n_spans_newly_accepted_all']} incl. neutral (cannot move the score)")
        print(f"    new LLM verdicts required    {s['n_records_flipped_to_mech_pass']}  "
              f"({s['n_fresh_llm_calls_this_run']} called in this process, rest served from "
              f"cache/prior run at temperature 0)")
        print(f"    LLM-layer unsupported        {pct(r['llm_layer']['rate'])} "
              f"{ci(r['llm_layer'])} on n={r['llm_layer']['n_reached_llm']}  ->  "
              f"{pct(nz['llm_layer']['rate'])} {ci(nz['llm_layer'])} on "
              f"n={nz['llm_layer']['n_reached_llm']}")
        print(f"    confident AND supported      {r['n_confident_supported']}  ->  "
              f"{nz['n_confident_supported']}")
        print(f"    verdicts (normalized)        {nz['verdict_breakdown']}")

    # -------------------------------------------------------------------
    # The control: baseline had ~0 quote-wrap failures, so it should barely
    # move. If it moves a lot, the normalizer is leniency, not a fix.
    # -------------------------------------------------------------------
    print("\n" + "-" * 112)
    print("CONTROL -- did the fix behave like a correction or like leniency?")
    print("-" * 112)
    base = summary["arm0_baseline"]
    br, bn = base["raw"]["unsupported"], base["normalized"]["unsupported"]
    bdelta = bn["rate"] - br["rate"]
    print(f"  Baseline moved {pct(br['rate'])} -> {pct(bn['rate'])} ({100*bdelta:+.1f}pp) on "
          f"{base['n_spans_newly_accepted_scoring']} score-relevant newly-accepted spans.")
    if abs(bdelta) <= 0.02:
        print(f"  Essentially unchanged, as predicted -- the baseline had almost no quote-wrap "
              f"failures\n  to correct. That is the control passing: the normalizer is fixing a "
              f"specific scoring\n  artifact, not handing every arm a blanket discount.")
    else:
        print(f"  FLAG -- the baseline moved more than 2pp despite having had ~0 quote-wrap "
              f"failures.\n  That is not what a targeted correction should do. Treat the arm "
              f"comparisons below as\n  suspect and inspect the normalizer before believing them.")

    # -------------------------------------------------------------------
    # The research question, now answerable.
    # -------------------------------------------------------------------
    print("\n" + "-" * 112)
    print("RESEARCH QUESTION -- does structured reasoning help grounding once the artifact is gone?")
    print("-" * 112)
    print(f"  Normalized baseline: {pct(bn['rate'])} ({bn['k']}/{bn['n']}, 95% CI {ci(bn)}).")
    for name in ARM_NAMES[1:]:
        s = summary[name]
        u = s["normalized"]["unsupported"]
        print(f"\n  {name} vs arm0_baseline, both normalized:")
        if u["rate"] is None or bn["rate"] is None:
            print("    n/a -- no non-neutral stances.")
            continue
        if u["underpowered"] or bn["underpowered"]:
            print(f"    [UNDERPOWERED] n={u['n']} vs {bn['n']} -- not interpreted.")
            continue
        delta = u["rate"] - bn["rate"]
        if cis_overlap(u, bn):
            print(f"    TIE. {pct(u['rate'])} {ci(u)} vs baseline {pct(bn['rate'])} {ci(bn)} "
                  f"-- 95% CIs overlap.\n    Point estimate {100*delta:+.1f}pp, which is not a "
                  f"separation at this n and is reported as a tie.")
        elif delta < 0:
            print(f"    LOWER, CI-separated. {pct(u['rate'])} {ci(u)} vs baseline "
                  f"{pct(bn['rate'])} {ci(bn)} ({100*delta:+.1f}pp).\n    Structured reasoning "
                  f"improves grounding once the formatting artifact is removed.")
        else:
            print(f"    HIGHER, CI-separated. {pct(u['rate'])} {ci(u)} vs baseline "
                  f"{pct(bn['rate'])} {ci(bn)} ({100*delta:+.1f}pp).\n    Still worse on "
                  f"grounding even after the correction.")
        # Same abstention guard as the prior run -- the headline stays gameable
        # by going neutral, and the stances here are the prior run's, so the
        # neutral rates are the prior run's too.
        print(f"    Neutral rate {pct(s['neutral_rate'])} vs baseline {pct(base['neutral_rate'])} "
              f"(unchanged from the prior run -- stances reused).\n    Confident-and-supported "
              f"{s['normalized']['n_confident_supported']} vs "
              f"{base['normalized']['n_confident_supported']}.")

    # -------------------------------------------------------------------
    # The conditional the prior run flagged but could not test.
    # -------------------------------------------------------------------
    print("\n" + "-" * 112)
    print("THE 6.2%-vs-30% CONDITIONAL -- does it survive on comparable subsets?")
    print("-" * 112)
    print("  The prior run found arm1's LLM-layer unsupported rate was 6.2% vs the baseline's")
    print("  30.1%, but warned it conditioned on a post-treatment variable: only 62.4% of arm1's")
    print("  stances got past the raw mechanical layer, vs 92.7% of the baseline's, so the")
    print("  auditor was reading a cleaner self-selected subset. Normalization raises those pass")
    print("  rates, making the subsets more comparable. Here is the same comparison now:\n")
    print(f"  {'arm':17s} {'mech pass':>11s} {'LLM-layer unsup':>17s} {'95% CI':>16s} {'n':>7s}")
    for name, s in summary.items():
        ll = s["normalized"]["llm_layer"]
        print(f"  {name:17s} {pct(ll['mech_pass_rate']):>11s} {pct(ll['rate']):>17s} "
              f"{ci(ll):>16s} {ll['n_reached_llm']:>7d}")
    bll = base["normalized"]["llm_layer"]
    for name in ARM_NAMES[1:]:
        ll = summary[name]["normalized"]["llm_layer"]
        if ll["rate"] is None or bll["rate"] is None:
            continue
        verdict = "TIE (CIs overlap)" if cis_overlap(ll, bll) else (
            "SEPARATED, lower" if ll["rate"] < bll["rate"] else "SEPARATED, higher")
        print(f"\n  {name} vs baseline on the LLM layer: {verdict} "
              f"({pct(ll['rate'])} vs {pct(bll['rate'])}).")
    spread = max(abs(summary[n]["normalized"]["llm_layer"]["mech_pass_rate"]
                     - bll["mech_pass_rate"]) for n in ARM_NAMES)
    print(f"\n  Residual selection effect: mechanical pass rates still differ by up to "
          f"{100*spread:.1f}pp\n  across arms, so this remains a decomposition rather than a "
          f"clean head-to-head. It is\n  closer to comparable than the prior run's, not fully "
          f"comparable.")

    print("\n" + "-" * 112)
    print("SCOPE")
    print("-" * 112)
    print("  This does NOT change the shipping decision. The baseline still ships. Two-call")
    print("  reasoning still costs ~2.3x the baseline's latency (9.63s vs 4.27s per note) and")
    print("  ~2.1x its tokens, none of which this re-scoring touches -- it re-scores grounding")
    print("  only. What changed here is a measurement artifact, not the agent.")
    print(f"\n  Model: {model}. Arms frozen, prompts untouched, stances reused verbatim from")
    print(f"  {PRIOR_CACHE.name}. Grounding check: quote-normalized via "
          f"scripts/eval_agent._is_grounded.")
    print(f"  Policy-id check and LLM auditor prompt: unchanged from shipped.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    out_json = args.out or (ROOT / "experiments" / "reasoning_comparison_spanfix_n2000.json")
    verifier_cache_path = ROOT / "experiments" / "reasoning_spanfix_verifier_cache_n2000.pkl"

    print("=== reasoning_stack_spanfix.py ===\n")

    # -- Step 0 -------------------------------------------------------------
    print("Step 0: reuse -- load the prior run's frozen stances, make no new stance calls.")
    with open(RETRIEVAL_CONFIG_PATH) as f:
        retrieval_mode = json.load(f).get("mode", "inprocess")
    assert retrieval_mode == "inprocess", (
        f"config/retrieval.json mode is {retrieval_mode!r}, expected 'inprocess' -- the policy-id "
        f"membership check must see the same retrieval the prior run did.")
    print(f"  config/retrieval.json mode = {retrieval_mode!r}  (asserted)")

    for path in (PRIOR_JSON, PRIOR_CACHE):
        if not path.exists():
            raise SystemExit(f"missing prior-run artifact: {path}")
    with open(PRIOR_JSON) as f:
        prior = json.load(f)
    with open(PRIOR_CACHE, "rb") as f:
        stance_cache = pickle.load(f)
    print(f"  loaded {len(stance_cache)} (loan_id, arm) stances from {PRIOR_CACHE.name}")
    print(f"  loaded prior results from {PRIOR_JSON.name} "
          f"(n_scored={prior['n_scored']}, seed={prior['seed']})")

    config = lc.load_config()
    lc.configure_from_config()
    print(f"  reused from reconciler_agent: verifier's mechanical layer (mirrored), "
          f"_retrieve_policy, _call_verifier_llm,")
    print(f"    VERIFIER_SYSTEM_PROMPT, VERIFIER_USER_TEMPLATE, POLICY_CORPUS_BY_ID, "
          f"CONF_THRESHOLD={ra.CONF_THRESHOLD}")
    print(f"  reused from scripts/eval_agent: _QUOTE_CHARS={_QUOTE_CHARS!r}, _is_grounded, "
          f"wilson_ci, UNDERPOWERED_N")
    print(f"  model={config.get('model')}  temperature={config.get('temperature')}")
    print("  NOT touched: agent/reconciler_agent.py, experiments/reasoning_stack.py, "
          "any arm prompt.\n")

    # -- the frozen sample --------------------------------------------------
    seed = prior["seed"]
    sample = base_rate_sample(load_test_population(), prior["sample_n"], seed)
    desc_by_id = {int(i): (row["desc_clean"] or "").strip() for i, row in sample.iterrows()}
    prior_ids = {r["loan_id"] for r in prior["records"][ARM_NAMES[0]]}
    assert prior_ids <= set(desc_by_id), (
        "the prior run's loan ids are not a subset of base_rate_sample(n, seed) -- the frozen "
        "sample could not be reconstructed, so descs cannot be matched to stances.")
    print(f"Frozen sample reconstructed: {len(prior_ids)} loan ids, seed={seed}, "
          f"all present in the prior records.\n")

    prior_by_arm = {
        name: {r["loan_id"]: r for r in prior["records"][name]} for name in ARM_NAMES
    }

    # -- Step 1/2 -- re-verify ----------------------------------------------
    verifier_cache = {}
    if verifier_cache_path.exists():
        with open(verifier_cache_path, "rb") as f:
            verifier_cache = pickle.load(f)
        print(f"Loaded {len(verifier_cache)} cached normalized verdicts from "
              f"{verifier_cache_path.name}")

    records = {name: [] for name in ARM_NAMES}
    n_missing = 0
    n_fresh = n_reused_prior = n_from_cache = 0
    audit_examples = []
    t0 = time.perf_counter()

    print("Step 2: re-verifying all three arms under the normalized check ...")
    for name in ARM_NAMES:
        for loan_id in sorted(prior_by_arm[name]):
            key = (loan_id, name)
            if key not in stance_cache:
                n_missing += 1
                continue
            stance_out = stance_cache[key][0]
            desc = desc_by_id[loan_id]
            prior_rec = prior_by_arm[name][loan_id]

            # Reuse the prior LLM verdict when this stance already reached the
            # LLM layer -- byte-identical inputs at temperature 0.
            reuse = (prior_rec["verifier_verdict"]
                     if prior_rec["verifier_source"] == "llm" else None)

            if key in verifier_cache:
                result = verifier_cache[key]
                n_from_cache += 1
                fresh = False
            else:
                result, fresh = verify_normalized(stance_out, desc, reuse)
                verifier_cache[key] = result
                if fresh:
                    n_fresh += 1
                    if n_fresh % CHECKPOINT_EVERY == 0:
                        with open(verifier_cache_path, "wb") as f:
                            pickle.dump(verifier_cache, f)
                        print(f"  ... {n_fresh} fresh verifier calls "
                              f"({time.perf_counter() - t0:.0f}s elapsed)")
            if result.get("llm_reused"):
                n_reused_prior += 1

            new_spans = newly_accepted_spans(stance_out, desc)
            # Audit only what could actually change a verdict -- a neutral
            # stance never reaches the span test.
            if new_spans and stance_out["stance"] != "neutral" and len(audit_examples) < 20:
                audit_examples.append({
                    "arm": name, "loan_id": loan_id,
                    "span": new_spans[0][:200],
                    "raw_check": "rejected", "normalized_check": "accepted",
                })

            records[name].append({
                "loan_id": loan_id,
                "stance": stance_out["stance"],
                "stance_confidence": stance_out["stance_confidence"],
                "stance_source": stance_out["stance_source"],
                "n_evidence": len(stance_out.get("stance_evidence") or []),
                "n_policy_ids": len(stance_out.get("stance_policy_ids") or []),
                "verifier_verdict_raw": prior_rec["verifier_verdict"],
                "verifier_source_raw": prior_rec["verifier_source"],
                "verifier_verdict_norm": result["verifier_verdict"],
                "verifier_source_norm": result["verifier_source"],
                "mechanical_cause_norm": result.get("mechanical_cause"),
                "llm_verdict_reused": result.get("llm_reused", False),
                "fresh_llm_call": fresh,
                "n_spans_newly_accepted": len(new_spans),
            })

    with open(verifier_cache_path, "wb") as f:
        pickle.dump(verifier_cache, f)

    print(f"\nDone in {time.perf_counter() - t0:.0f}s.")
    print(f"  fresh verifier LLM calls this run : {n_fresh}")
    print(f"  prior LLM verdicts reused         : {n_reused_prior}")
    print(f"  results served from local cache   : {n_from_cache}")
    print(f"  stance LLM calls made             : 0  (arms frozen, stances reused verbatim)")
    if n_missing:
        print(f"  SKIPPED (missing from stance cache): {n_missing}")
    else:
        print(f"  missing from stance cache         : 0")

    # -- Step 3 -------------------------------------------------------------
    summary = {name: arm_summary(records[name]) for name in ARM_NAMES}
    audit = {
        "prior_summary": {
            name: {
                "n_neutral": prior["summary"][name]["funnel"]["n_neutral"],
                "n_non_neutral": prior["summary"][name]["funnel"]["n_non_neutral"],
                "k_unsupported": prior["summary"][name]["unsupported"]["k"],
            } for name in ARM_NAMES
        },
        "newly_accepted_span_examples": audit_examples,
        "n_missing_from_stance_cache": n_missing,
        "n_fresh_llm_calls": n_fresh,
        "n_prior_llm_verdicts_reused": n_reused_prior,
        "n_stance_llm_calls": 0,
    }

    # -- Step 4 -------------------------------------------------------------
    print_report(summary, len(records[ARM_NAMES[0]]), config.get("model"), audit)

    print("\n-- Audit sample: spans the raw check rejected and the normalizer accepts --")
    for ex in audit_examples[:8]:
        print(f"  [{ex['arm']}] loan {ex['loan_id']}: {ex['span']!r}")

    report = {
        "_status": "complete",
        "experiment": "quote-normalized re-scoring of reasoning_stack's frozen arms",
        "sample_n": prior["sample_n"],
        "n_scored": len(records[ARM_NAMES[0]]),
        "seed": seed,
        "held_fixed": {
            "arms": "frozen -- prompts untouched, stances reused verbatim from "
                    f"{PRIOR_CACHE.name}",
            "stance_llm_calls": 0,
            "retrieval": f"shipped in-process BM25 over ra.POLICY_CORPUS "
                         f"({len(ra.POLICY_CORPUS)} chunks), k={K}",
            "retrieval_mode": retrieval_mode,
            "policy_id_check": "unchanged from shipped",
            "llm_auditor": "unchanged from shipped (byte-identical VERIFIER prompts)",
            "model": config.get("model"),
            "temperature": config.get("temperature"),
            "conf_threshold": ra.CONF_THRESHOLD,
        },
        "varied": "the mechanical span test only: `span in desc` -> "
                  "eval_agent._is_grounded(span, desc)",
        "prior_run": str(PRIOR_JSON.relative_to(ROOT)),
        "audit": audit,
        "summary": summary,
        "records": records,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nSaved {out_json}")

    # -- MLflow -------------------------------------------------------------
    if args.track:
        from tracking import track_run
        import mlflow

        print(f"\nLogging {len(summary)} runs to MLflow experiment {args.experiment!r} ...")
        for name, s in summary.items():
            r, nz = s["raw"], s["normalized"]
            params = {
                "arm": name,
                "model": config.get("model"),
                "corpus": "shipped",
                "k": K,
                "n": s["n_notes"],
                "seed": seed,
                # The two facts that make this run interpretable at a glance.
                "stances": f"reused_verbatim_from_{PRIOR_CACHE.name}",
                "stance_llm_calls": 0,
                "grounding_check": "quote_normalized_eval_agent._is_grounded",
                "policy_id_check": "unchanged",
                "llm_auditor": "unchanged",
                "prior_run": PRIOR_JSON.name,
            }
            nan = float("nan")
            metrics = {
                "raw_unsupported_rate": r["unsupported"]["rate"] if r["unsupported"]["rate"] is not None else nan,
                "raw_unsupported_ci_low": r["unsupported"]["ci_low"] if r["unsupported"]["ci_low"] is not None else nan,
                "raw_unsupported_ci_high": r["unsupported"]["ci_high"] if r["unsupported"]["ci_high"] is not None else nan,
                "raw_unsupported_k": r["unsupported"]["k"],
                "raw_mech_pass_rate": r["llm_layer"]["mech_pass_rate"] if r["llm_layer"]["mech_pass_rate"] is not None else nan,
                "raw_llm_layer_unsupported_rate": r["llm_layer"]["rate"] if r["llm_layer"]["rate"] is not None else nan,
                "raw_n_confident_supported": r["n_confident_supported"],
                "raw_n_mechanical_fail": r["n_mechanical_fail"],
                "norm_unsupported_rate": nz["unsupported"]["rate"] if nz["unsupported"]["rate"] is not None else nan,
                "norm_unsupported_ci_low": nz["unsupported"]["ci_low"] if nz["unsupported"]["ci_low"] is not None else nan,
                "norm_unsupported_ci_high": nz["unsupported"]["ci_high"] if nz["unsupported"]["ci_high"] is not None else nan,
                "norm_unsupported_k": nz["unsupported"]["k"],
                "norm_mech_pass_rate": nz["llm_layer"]["mech_pass_rate"] if nz["llm_layer"]["mech_pass_rate"] is not None else nan,
                "norm_llm_layer_unsupported_rate": nz["llm_layer"]["rate"] if nz["llm_layer"]["rate"] is not None else nan,
                "norm_llm_layer_ci_low": nz["llm_layer"]["ci_low"] if nz["llm_layer"]["ci_low"] is not None else nan,
                "norm_llm_layer_ci_high": nz["llm_layer"]["ci_high"] if nz["llm_layer"]["ci_high"] is not None else nan,
                "norm_llm_layer_n_reached": nz["llm_layer"]["n_reached_llm"],
                "norm_n_confident_supported": nz["n_confident_supported"],
                "norm_n_mechanical_fail": nz["n_mechanical_fail"],
                "norm_n_mechanical_fail_span": nz["n_mechanical_fail_span"],
                "norm_n_mechanical_fail_policy_id": nz["n_mechanical_fail_policy_id"],
                "delta_unsupported_rate": (
                    nz["unsupported"]["rate"] - r["unsupported"]["rate"]
                    if None not in (nz["unsupported"]["rate"], r["unsupported"]["rate"]) else nan),
                "n_records_flipped_to_mech_pass": s["n_records_flipped_to_mech_pass"],
                "n_spans_newly_accepted_scoring": s["n_spans_newly_accepted_scoring"],
                "n_spans_newly_accepted_all": s["n_spans_newly_accepted_all"],
                "n_fresh_llm_calls_this_run": s["n_fresh_llm_calls_this_run"],
                "neutral_rate": s["neutral_rate"] if s["neutral_rate"] is not None else nan,
                "non_neutral_n": s["n_non_neutral"],
            }
            with track_run(experiment=args.experiment, run_name=f"spanfix_{name}", params=params):
                mlflow.log_metrics(metrics)
                mlflow.log_artifact(str(out_json))
            print(f"  logged run: spanfix_{name}")


if __name__ == "__main__":
    main()
