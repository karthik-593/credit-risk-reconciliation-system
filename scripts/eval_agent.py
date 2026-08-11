"""
Headline eval for the tabular/text reconciliation agent (agent/reconciler_agent.py),
run over the locked TEST split's desc-only population. Four modes, one flag:

  (default) STUB  -- deterministic, hash-keyed stub stances. Zero LLM tokens.
                      Exercises every route bucket + the underpowered guard.
  --validate      -- 40 REAL LLM calls only. Reports parse-success rate,
                      evidence-grounding rate, stance distribution. Gate
                      before spending tokens on the full run.
  --real          -- SAMPLE_N=2000 run with the real adapter (agent/llm_client.py).
                      Every stance cached by loan_id to
                      results/eval_stance_cache.pkl so re-runs are free.
                      Saves results/agent_eval.json.
  --full          -- the ENTIRE locked TEST population (21,616 rows), real
                      adapter, same cache (reused for the 2,000 already
                      scored, new calls only for the rest). Saves
                      results/agent_eval_fullpower.json -- does NOT overwrite
                      agent_eval.json.

Population: split_indices.pkl TEST indices ONLY -- never train (train was
tuning's; TEST is read here for evaluation, same as the tuning notebook's own
single test-touching cell). Rows come from feasibility_frame.pkl, the
desc-only population the tabular models were both built and scored on.
--validate/--real draw a base-rate-preserved random sample (seed 42) from
TEST; --full uses all of it, no sampling.

Does not modify agent/reconciler_agent.py. This script imports and calls its
existing node functions (tabular_score, text_stance, reconciler, _route,
auto_decision, human_review, explanation) in the same order build_graph()
wires them -- called directly rather than through the compiled graph so
--real can cache-check each loan_id's stance before hitting the LLM, which
the compiled StateGraph has no hook for.
"""
import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
import reconciler_agent as ra  # noqa: E402

FRAME_PATH = ROOT / "data" / "interim" / "feasibility_frame.pkl"
SPLIT_PATH = ROOT / "data" / "interim" / "split_indices.pkl"
CACHE_PATH = ROOT / "results" / "eval_stance_cache.pkl"
VERIFIER_CACHE_PATH = ROOT / "results" / "eval_verifier_cache.pkl"
# Output path is chosen in main() by mode: agent_eval.json (stub/real) or
# agent_eval_fullpower.json (--full) -- --full must never overwrite the other.

RAW_TABULAR_COLS = [
    "loan_amnt", "term", "int_rate", "grade", "sub_grade",
    "annual_inc", "dti", "emp_length", "home_ownership",
    "verification_status", "fico_range_low", "fico_range_high", "purpose",
]

SAMPLE_N = 2000
SEED = 42
VALIDATE_N = 40
REVIEW_BUDGET_PCT = 15.0
UNDERPOWERED_N = 30


# ---------------------------------------------------------------------------
# Stub LLM client -- deterministic, hash-keyed, zero tokens.
# ---------------------------------------------------------------------------
class StubLLMClient:
    """Deterministic stand-in for a real LLM: derives stance/confidence from
    an md5 hash of the prompt text, so re-runs are stable and the mix spans
    all three stances and both sides of CONF_THRESHOLD -- exercising every
    route bucket without spending a token. Not a narrative read of anything;
    do not mistake its output for signal."""

    STANCES = ["corroborates_risk", "mitigates_risk", "neutral"]

    def complete(self, system: str, user: str) -> str:
        digest = hashlib.md5(user.encode("utf-8")).hexdigest()
        n = int(digest, 16)
        stance = self.STANCES[n % 3]
        confidence = round(0.15 + ((n // 3) % 81) / 100, 2)  # spans ~0.15-0.95
        payload = {
            "stance": stance,
            "evidence_spans": [],
            "cited_policy_ids": [],
            "confidence": confidence,
            "rationale": "stub eval client -- synthetic, not a real narrative read",
        }
        return json.dumps(payload)


# ---------------------------------------------------------------------------
# Data loading / sampling
# ---------------------------------------------------------------------------
def load_test_population() -> pd.DataFrame:
    frame = pd.read_pickle(FRAME_PATH)
    with open(SPLIT_PATH, "rb") as f:
        split = pickle.load(f)
    test_idx = split["test_idx"]  # TEST ONLY. train_idx is never read in this script.
    return frame.loc[test_idx]


def base_rate_sample(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    y = frame["default"]
    base_rate = y.mean()
    n_pos = min(round(n * base_rate), int((y == 1).sum()))
    n_neg = min(n - n_pos, int((y == 0).sum()))
    pos = frame[y == 1].sample(n=n_pos, random_state=seed)
    neg = frame[y == 0].sample(n=n_neg, random_state=seed)
    return pd.concat([pos, neg]).sample(frac=1, random_state=seed)


# ---------------------------------------------------------------------------
# Running one loan through the same node sequence build_graph() wires:
# tabular_score -> text_stance -> verifier -> reconciler -> route -> ...
# ---------------------------------------------------------------------------
def run_one(row: pd.Series, loan_id, stance_cache: dict | None = None,
            verifier_cache: dict | None = None) -> dict:
    tabular_features = {c: row[c] for c in RAW_TABULAR_COLS}
    application = {"tabular_features": tabular_features, "desc_clean": row["desc_clean"]}
    state = {"application": application}

    state.update(ra.tabular_score(state))

    if stance_cache is not None and loan_id in stance_cache:
        stance_out = stance_cache[loan_id]
    else:
        stance_out = ra.text_stance(state)
        if stance_cache is not None:
            stance_cache[loan_id] = stance_out
    state.update(stance_out)

    # Verifier reads the (possibly cached) stance and may downgrade it --
    # cached separately from the stance itself since it's a distinct LLM
    # call (or no call at all, for neutral/mechanical-fail cases) keyed by
    # the same loan_id.
    if verifier_cache is not None and loan_id in verifier_cache:
        verifier_out = verifier_cache[loan_id]
    else:
        verifier_out = ra.verifier(state)
        if verifier_cache is not None:
            verifier_cache[loan_id] = verifier_out
    state.update(verifier_out)

    state.update(ra.reconciler(state))
    next_node = ra._route(state)
    state.update(ra.auto_decision(state) if next_node == "auto_decision" else ra.human_review(state))
    state.update(ra.explanation(state))
    return state


def tabular_alone_decision(p_default: float) -> str:
    return "decline" if p_default >= ra.HIGH_RISK else "approve"


# ---------------------------------------------------------------------------
# --validate: 40 real calls, LLM-quality gate only. No tabular scoring needed.
# ---------------------------------------------------------------------------
_QUOTE_CHARS = "\"'“”‘’"


def _is_grounded(span: str, desc: str) -> bool:
    """Case-insensitive substring check, with surrounding quote characters
    stripped from the span first -- some models wrap an otherwise-exact
    quote in its own "..." or lightly re-case the first letter, neither of
    which is a fabricated quote. A genuine paraphrase (different words, not
    just quoting/casing) still correctly fails this check."""
    normalized_span = span.strip().strip(_QUOTE_CHARS).strip()
    return normalized_span.lower() in desc.lower()


def run_validate(sample: pd.DataFrame) -> None:
    from llm_client import configure_from_config
    configure_from_config()

    print(f"=== VALIDATE: {len(sample)} real LLM calls (stance, + verifier where applicable) ===")

    source_counts = {"parsed": 0, "parse_error": 0, "api_error": 0, "empty": 0}
    stance_counts = {"corroborates_risk": 0, "mitigates_risk": 0, "neutral": 0}
    verifier_counts = {"supported": 0, "unsupported": 0, "unclear": 0, "skipped_neutral": 0}
    verifier_source_counts = {"mechanical": 0, "llm": 0, "skipped": 0}
    n_downgraded = 0
    n_evidence_provided = 0
    n_evidence_grounded = 0

    for loan_id, row in sample.iterrows():
        state = {"application": {"desc_clean": row["desc_clean"]}}
        out = ra.text_stance(state)
        source = out.get("stance_source", "parsed")
        source_counts[source] = source_counts.get(source, 0) + 1

        if source in ("api_error", "empty"):
            # Infrastructure failure, not a model verdict -- excluded from
            # stance distribution and grounding metrics below.
            continue

        stance_counts[out["stance"]] = stance_counts.get(out["stance"], 0) + 1

        # NOTE: text_stance() returns evidence under "stance_evidence", not
        # "evidence_spans" (that's the raw LLM JSON's key, before text_stance
        # renames it). An earlier version of this function read the wrong
        # key here and silently reported 0 evidence on every call, even
        # successful ones -- fixed.
        spans = out.get("stance_evidence") or []
        if spans:
            n_evidence_provided += 1
            if all(_is_grounded(span, row["desc_clean"]) for span in spans):
                n_evidence_grounded += 1

        state.update(out)
        verifier_out = ra.verifier(state)
        verifier_counts[verifier_out["verifier_verdict"]] = (
            verifier_counts.get(verifier_out["verifier_verdict"], 0) + 1
        )
        verifier_source_counts[verifier_out["verifier_source"]] = (
            verifier_source_counts.get(verifier_out["verifier_source"], 0) + 1
        )
        if "stance" in verifier_out:  # verifier only includes this key when it downgraded
            n_downgraded += 1

    n = len(sample)
    n_no_stance = source_counts["api_error"] + source_counts["empty"]
    n_scored = n - n_no_stance

    print("\nStance-source breakdown:")
    for src in ["parsed", "parse_error", "api_error", "empty"]:
        print(f"  {src:12s}: {source_counts[src]:3d} ({100 * source_counts[src] / n:.1f}%)")
    print(f"\nCalls that never produced a stance (api_error + empty, excluded below): "
          f"{n_no_stance}/{n} ({100 * n_no_stance / n:.1f}%)")

    if source_counts["api_error"] > 0:
        print("\n" + "!" * 78)
        print(f"WARNING: {source_counts['api_error']} api_error(s) in this run (network/rate-limit/")
        print("quota/timeout failures after retries). The numbers below are computed only over")
        print("the calls that actually produced a response and are NOT a clean read of model")
        print("behavior -- re-run once the underlying failure is resolved before trusting this")
        print("as a gate result.")
        print("!" * 78)

    if n_scored == 0:
        print("\nNo scored calls remain (every call was api_error/empty) -- nothing to report.")
        return

    print(f"\nStance distribution (of {n_scored} scored calls; api_error/empty excluded):")
    for s, c in stance_counts.items():
        print(f"  {s}: {c} ({100 * c / n_scored:.1f}%)")

    print(f"\nEvidence provided:  {n_evidence_provided}/{n_scored}")
    if n_evidence_provided:
        evidence_grounding_rate = n_evidence_grounded / n_evidence_provided
        print(f"Evidence grounding rate (of those with evidence): {evidence_grounding_rate:.2%}")
    else:
        print("Evidence grounding rate: n/a (no evidence provided by any scored call)")

    print(f"\nVerifier verdict (of {n_scored} scored calls):")
    for v, c in verifier_counts.items():
        print(f"  {v:16s}: {c} ({100 * c / n_scored:.1f}%)")
    print("Verifier source (where the verdict came from):")
    for s, c in verifier_source_counts.items():
        print(f"  {s:16s}: {c} ({100 * c / n_scored:.1f}%)")
    print(f"Downgraded to neutral by the verifier: {n_downgraded}/{n_scored} "
          f"({100 * n_downgraded / n_scored:.1f}%)")

    print("\nGate check: review the above before running --real.")


# ---------------------------------------------------------------------------
# stub / real: full run
# ---------------------------------------------------------------------------
def run_full(sample: pd.DataFrame, mode: str) -> pd.DataFrame:
    uses_real_adapter = mode in ("real", "full")
    stance_cache: dict = {}
    verifier_cache: dict = {}
    if uses_real_adapter:
        from llm_client import configure_from_config
        configure_from_config()
        if CACHE_PATH.exists():
            with open(CACHE_PATH, "rb") as f:
                stance_cache = pickle.load(f)
            print(f"Loaded {len(stance_cache)} cached stances from {CACHE_PATH}")
        if VERIFIER_CACHE_PATH.exists():
            with open(VERIFIER_CACHE_PATH, "rb") as f:
                verifier_cache = pickle.load(f)
            print(f"Loaded {len(verifier_cache)} cached verifier verdicts from {VERIFIER_CACHE_PATH}")
    else:
        ra.configure_llm_client(StubLLMClient())

    checkpoint_interval = 500 if len(sample) > 5000 else 100

    records = []
    n_stance_cache_hit = 0
    n_stance_new_calls = 0
    n_verifier_cache_hit = 0
    n_verifier_new_calls = 0
    for i, (loan_id, row) in enumerate(sample.iterrows()):
        stance_was_cached = uses_real_adapter and (loan_id in stance_cache)
        verifier_was_cached = uses_real_adapter and (loan_id in verifier_cache)
        state = run_one(
            row, loan_id,
            stance_cache=stance_cache if uses_real_adapter else None,
            verifier_cache=verifier_cache if uses_real_adapter else None,
        )
        if uses_real_adapter:
            if stance_was_cached:
                n_stance_cache_hit += 1
            else:
                n_stance_new_calls += 1
            if verifier_was_cached:
                n_verifier_cache_hit += 1
            else:
                n_verifier_new_calls += 1
        records.append({
            "loan_id": loan_id,
            "p_default": state["p_default"],
            "stance": state["stance"],
            "confidence": state["stance_confidence"],
            "route": state["route"],
            "agent_decision": state["decision"],
            "tabular_alone_decision": tabular_alone_decision(state["p_default"]),
            "y": int(row["default"]),
            "stance_source": state.get("stance_source", "parsed"),
            "verifier_verdict": state.get("verifier_verdict", "skipped_neutral"),
            "verifier_source": state.get("verifier_source", "skipped"),
        })
        if uses_real_adapter and (i + 1) % checkpoint_interval == 0:
            with open(CACHE_PATH, "wb") as f:
                pickle.dump(stance_cache, f)
            with open(VERIFIER_CACHE_PATH, "wb") as f:
                pickle.dump(verifier_cache, f)
            print(f"  ... {i + 1}/{len(sample)} (caches checkpointed; "
                  f"stance hits={n_stance_cache_hit} new={n_stance_new_calls}; "
                  f"verifier hits={n_verifier_cache_hit} new={n_verifier_new_calls})")

    if uses_real_adapter:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(stance_cache, f)
        with open(VERIFIER_CACHE_PATH, "wb") as f:
            pickle.dump(verifier_cache, f)
        print(f"\nStance cache:   {n_stance_cache_hit} hits, {n_stance_new_calls} new calls "
              f"({len(stance_cache)} cached total)")
        print(f"Verifier cache: {n_verifier_cache_hit} hits, {n_verifier_new_calls} new calls "
              f"({len(verifier_cache)} cached total)")

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------
def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial rate -- better-behaved than the
    normal approximation at small n or rates near 0/1, both of which show up
    in these buckets."""
    if n == 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = (z * ((phat * (1 - phat) / n + z * z / (4 * n * n)) ** 0.5)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def bucket_stats(df: pd.DataFrame, mask, label: str, underpowered_log: list) -> dict:
    n = int(mask.sum())
    if n < UNDERPOWERED_N:
        print(f"  [{label}] n={n}  <-- UNDERPOWERED (<{UNDERPOWERED_N}), not interpreted")
        underpowered_log.append({"section": "realized_rates", "bucket": label, "n": n})
        return {"n": n, "underpowered": True}
    k = int(df.loc[mask, "y"].sum())
    rate = k / n
    ci_low, ci_high = wilson_ci(k, n)
    print(f"  [{label}] n={n}  realized_default_rate={rate:.4f}  "
          f"95% CI=[{ci_low:.4f}, {ci_high:.4f}]")
    return {
        "n": n, "underpowered": False, "k": k, "realized_default_rate": rate,
        "ci_low": ci_low, "ci_high": ci_high,
    }


def section_route_distribution(df: pd.DataFrame) -> dict:
    print("\n=== 1. Route distribution ===")
    counts = df["route"].value_counts()
    result = {}
    for route in ["agree", "disagree", "low_conf"]:
        n = int(counts.get(route, 0))
        pct = 100 * n / len(df)
        print(f"  {route:10s}: {n:5d} ({pct:.2f}%)")
        result[route] = {"n": n, "pct": pct}
    disagree_rate = result["disagree"]["pct"]
    print(f"\n  DISAGREE RATE: {disagree_rate:.2f}% -- reported plainly; "
          f"near-zero is a finding, not a bug.")
    result["disagree_rate_pct"] = disagree_rate
    return result


def section_decisions_changed(df: pd.DataFrame) -> dict:
    print("\n=== 2. Decisions changed by text ===")
    changed_mask = df["agent_decision"] == "human_review"
    n_changed = int(changed_mask.sum())
    pct_changed = 100 * n_changed / len(df)
    print(f"  Changed (deferred to human review instead of an automatic call): "
          f"{n_changed} ({pct_changed:.2f}%)")

    approve_to_review = int(((df["tabular_alone_decision"] == "approve") & changed_mask).sum())
    decline_to_review = int(((df["tabular_alone_decision"] == "decline") & changed_mask).sum())
    print(f"    tabular said approve -> deferred: {approve_to_review} "
          f"({100 * approve_to_review / len(df):.2f}%)")
    print(f"    tabular said decline -> deferred: {decline_to_review} "
          f"({100 * decline_to_review / len(df):.2f}%)")
    return {
        "n_changed": n_changed,
        "pct_changed": pct_changed,
        "approve_to_review": approve_to_review,
        "decline_to_review": decline_to_review,
    }


def section_realized_rates(df: pd.DataFrame, is_stub: bool, underpowered_log: list) -> dict:
    print("\n=== 3. Realized default rate: flagged/mitigated vs clean ===")
    if is_stub:
        print("  NOTE: stub stances are hash-derived noise, uncorrelated with real narrative")
        print("  content. These numbers exercise the code path only -- not a finding.")

    clean_approve = (df["tabular_alone_decision"] == "approve") & (df["route"] == "agree")
    approve_flagged = (df["tabular_alone_decision"] == "approve") & (df["route"] == "disagree")
    clean_decline = (df["tabular_alone_decision"] == "decline") & (df["route"] == "agree")
    decline_mitigated = (df["tabular_alone_decision"] == "decline") & (df["route"] == "disagree")

    result = {
        "clean_approve": bucket_stats(df, clean_approve, "clean_approve", underpowered_log),
        "approve_but_flagged": bucket_stats(df, approve_flagged, "approve_but_flagged", underpowered_log),
        "clean_decline": bucket_stats(df, clean_decline, "clean_decline", underpowered_log),
        "decline_but_mitigated": bucket_stats(df, decline_mitigated, "decline_but_mitigated", underpowered_log),
    }

    def compare(a, b, op, expect_label):
        if a.get("underpowered") or b.get("underpowered"):
            print(f"  {expect_label}: skipped (one side underpowered)")
            return None

        # CI overlap check FIRST -- a point-estimate direction is not
        # reportable as a pass/fail when the intervals overlap; that's
        # exactly the mistake Build 5 made (DECISIONS.md).
        overlap = a["ci_low"] <= b["ci_high"] and b["ci_low"] <= a["ci_high"]
        if overlap:
            print(f"  {expect_label}: INCONCLUSIVE at this n "
                  f"(95% CIs overlap: [{a['ci_low']:.4f},{a['ci_high']:.4f}] vs "
                  f"[{b['ci_low']:.4f},{b['ci_high']:.4f}])")
            return "inconclusive"

        holds = (a["realized_default_rate"] > b["realized_default_rate"]) if op == ">" \
            else (a["realized_default_rate"] < b["realized_default_rate"])
        direction = "HOLDS" if holds else "DOES NOT HOLD (opposite direction)"
        print(f"  {expect_label}: SIGNIFICANT -- {direction} (95% CIs disjoint: "
              f"[{a['ci_low']:.4f},{a['ci_high']:.4f}] vs [{b['ci_low']:.4f},{b['ci_high']:.4f}])")
        return holds

    result["flagged_higher_than_clean_approve"] = compare(
        result["approve_but_flagged"], result["clean_approve"], ">",
        "approve_but_flagged > clean_approve (expected)"
    )
    result["mitigated_lower_than_clean_decline"] = compare(
        result["decline_but_mitigated"], result["clean_decline"], "<",
        "decline_but_mitigated < clean_decline (expected)"
    )
    return result


def section_calibration_fp_fn(df: pd.DataFrame, underpowered_log: list) -> dict:
    print("\n=== 4. Calibration lift, FP/FN deltas, review-rate budget ===")
    print("  Definitions: Brier = mean((p_default - y)^2). Calibration lift = "
          "Brier(full) - Brier(auto-decided subset); positive means the automated")
    print("  subset is better-calibrated than the full population (ambiguous cases filtered out).")

    def brier(sub: pd.DataFrame) -> float:
        return float(((sub["p_default"] - sub["y"]) ** 2).mean())

    brier_full = brier(df)
    # "Automated" = whatever the agent actually auto-decided, defined by the
    # DECISION, not the route label -- agree and low_conf both resolve to
    # auto_decision under the current _route() (DECISIONS.md Build 5/6;
    # low_conf no longer means "deferred"). Deriving this from route=="agree"
    # alone would silently exclude low_conf's now-automated rows.
    deferred_mask = df["agent_decision"] == "human_review"
    automated_mask = ~deferred_mask
    n_automated = int(automated_mask.sum())
    print(f"\n  Brier, full sample (n={len(df)}): {brier_full:.4f}")
    if n_automated >= UNDERPOWERED_N:
        brier_automated = brier(df[automated_mask])
        calibration_lift = brier_full - brier_automated
        print(f"  Brier, auto-decided subset (n={n_automated}): {brier_automated:.4f}")
        print(f"  Calibration lift: {calibration_lift:+.4f}")
    else:
        brier_automated = None
        calibration_lift = None
        print(f"  auto-decided subset UNDERPOWERED (n={n_automated}), skipping calibration lift")
        underpowered_log.append({"section": "calibration", "bucket": "automated_subset", "n": n_automated})

    fp_baseline_mask = (df["tabular_alone_decision"] == "decline") & (df["y"] == 0)
    fn_baseline_mask = (df["tabular_alone_decision"] == "approve") & (df["y"] == 1)
    fp_baseline = int(fp_baseline_mask.sum())
    fn_baseline = int(fn_baseline_mask.sum())

    fp_deferred = int((fp_baseline_mask & deferred_mask).sum())
    fn_deferred = int((fn_baseline_mask & deferred_mask).sum())
    fp_automated = fp_baseline - fp_deferred
    fn_automated = fn_baseline - fn_deferred

    print(f"\n  Tabular-alone baseline (everyone auto-decided): "
          f"FP={fp_baseline} ({100 * fp_baseline / len(df):.2f}%)  "
          f"FN={fn_baseline} ({100 * fn_baseline / len(df):.2f}%)")
    print(f"  Of those, deferred to human review instead of auto-executed: "
          f"FP={fp_deferred}  FN={fn_deferred}")
    print(f"  Remaining automated errors: FP={fp_automated}  FN={fn_automated}")

    review_rate = 100 * int(deferred_mask.sum()) / len(df)
    within_budget = review_rate <= REVIEW_BUDGET_PCT
    print(f"\n  Review rate: {review_rate:.2f}% vs ~{REVIEW_BUDGET_PCT:.0f}% budget "
          f"({'within' if within_budget else 'OVER'} budget)")

    return {
        "brier_full": brier_full,
        "brier_automated": brier_automated,
        "calibration_lift": calibration_lift,
        "fp_baseline": fp_baseline, "fn_baseline": fn_baseline,
        "fp_deferred": fp_deferred, "fn_deferred": fn_deferred,
        "fp_automated": fp_automated, "fn_automated": fn_automated,
        "review_rate_pct": review_rate,
        "review_budget_pct": REVIEW_BUDGET_PCT,
        "within_budget": within_budget,
    }


def section_fairness(records_df: pd.DataFrame, sample: pd.DataFrame, underpowered_log: list) -> dict:
    print("\n=== 5. Fairness slice (proxy groups: home_ownership, verification_status) ===")
    print("  LendingClub does not collect protected attributes; these are the closest")
    print("  available demographic-adjacent proxies, not a substitute for real ones.")

    joined = records_df.set_index("loan_id").join(sample[["home_ownership", "verification_status"]])
    result = {}
    for group_col in ["home_ownership", "verification_status"]:
        print(f"\n  -- {group_col} --")
        result[group_col] = {}
        for g in sorted(joined[group_col].dropna().astype(str).unique()):
            sub = joined[joined[group_col].astype(str) == g]
            n = len(sub)
            if n < UNDERPOWERED_N:
                print(f"    {g:20s} n={n}  <-- UNDERPOWERED, not interpreted")
                result[group_col][g] = {"n": n, "underpowered": True}
                underpowered_log.append({"section": "fairness", "bucket": f"{group_col}={g}", "n": n})
                continue
            tabular_approve_rate = 100 * (sub["tabular_alone_decision"] == "approve").mean()
            agent_approve_rate = 100 * (sub["agent_decision"] == "auto_approve").mean()
            default_rate = 100 * sub["y"].mean()
            shift = agent_approve_rate - tabular_approve_rate
            flag = "  <-- shifted >5pp by text" if abs(shift) > 5 else ""
            print(f"    {g:20s} n={n:4d}  tabular_approve={tabular_approve_rate:5.1f}%  "
                  f"agent_approve={agent_approve_rate:5.1f}%  shift={shift:+.1f}pp  "
                  f"default_rate={default_rate:5.1f}%{flag}")
            result[group_col][g] = {
                "n": n, "underpowered": False,
                "tabular_approve_rate_pct": tabular_approve_rate,
                "agent_approve_rate_pct": agent_approve_rate,
                "shift_pp": shift,
                "default_rate_pct": default_rate,
            }
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--validate", action="store_true", help="40 real LLM calls only; gate before --real.")
    mode_group.add_argument("--real", action="store_true", help="SAMPLE_N=2000 run with the real LLM adapter; caches by loan_id.")
    mode_group.add_argument("--full", action="store_true", help="ENTIRE locked TEST population, real adapter, same cache; saves to agent_eval_fullpower.json.")
    args = parser.parse_args()

    mode = "full" if args.full else ("real" if args.real else ("validate" if args.validate else "stub"))
    print(f"=== eval_agent.py -- mode={mode} ===\n")

    test_frame = load_test_population()
    test_rate = test_frame["default"].mean()
    print(f"TEST population: {len(test_frame):,} rows, default rate {test_rate:.4f}")

    if mode == "full":
        sample = test_frame
        sample_rate = test_rate
        print(f"Using the ENTIRE locked TEST population: {len(sample):,} rows. No sampling; "
              f"train/val are never read by this script.\n")
    else:
        n = VALIDATE_N if mode == "validate" else SAMPLE_N
        sample = base_rate_sample(test_frame, n, SEED)
        sample_rate = sample["default"].mean()
        check = "OK" if abs(sample_rate - 0.1526) < 0.02 else "CHECK"
        print(f"Sample: {len(sample):,} rows (seed={SEED}), default rate {sample_rate:.4f} "
              f"(target ~0.1526, {check})\n")

    if mode == "validate":
        run_validate(sample)
        return

    records_df = run_full(sample, mode)

    print("\n" + "=" * 78)
    print(f"AGENT EVAL REPORT (mode={mode}, n={len(records_df)})")
    print("=" * 78)
    if mode == "stub":
        print("STUB MODE: every stance below is hash-derived noise, not a real narrative")
        print("read. That means every number downstream of route/stance -- review rate,")
        print("FP/FN deltas, fairness shifts, all of it -- reflects the stub's synthetic")
        print("confidence distribution, not agent behavior. This run verifies the eval")
        print("mechanics and the underpowered guard only. Re-run with --real for findings.")

    # --- 0. stance_source breakdown FIRST. Failures are EXCLUDED from every
    # metric below (not folded into "neutral"), so a transient infra failure
    # can't silently masquerade as a real low-confidence verdict.
    n_total = len(records_df)
    source_counts = records_df["stance_source"].value_counts()
    print("\n=== 0. stance_source breakdown (failures excluded from all metrics below) ===")
    for src in ["parsed", "parse_error", "api_error", "empty"]:
        c = int(source_counts.get(src, 0))
        print(f"  {src:12s}: {c:6d} ({100 * c / n_total:.2f}%)")

    excluded_mask = records_df["stance_source"].isin(["api_error", "empty"])
    n_excluded = int(excluded_mask.sum())
    if n_excluded > 0:
        print("\n" + "!" * 78)
        print(f"WARNING: {n_excluded}/{n_total} rows never produced a stance (api_error/empty) "
              f"and are EXCLUDED from every section below -- not folded into neutral/low_conf.")
        print("This run is partially contaminated for COVERAGE purposes even though the")
        print("analyzed rows themselves are clean. Check results records for stance_source.")
        print("!" * 78)

    analyzed_df = records_df.loc[~excluded_mask].copy()
    n_analyzed = len(analyzed_df)
    print(f"\nAnalyzed rows (all sections below use this population): {n_analyzed}/{n_total} "
          f"({100 * n_analyzed / n_total:.2f}%)")

    verifier_counts = analyzed_df["verifier_verdict"].value_counts()
    print("\n=== 0b. Verifier verdict breakdown (grounding check on the stance, "
          "not a risk re-judgment) ===")
    for v in ["supported", "unsupported", "unclear", "skipped_neutral"]:
        c = int(verifier_counts.get(v, 0))
        print(f"  {v:16s}: {c:6d} ({100 * c / n_analyzed:.2f}%)")
    n_downgraded_total = int((analyzed_df["verifier_verdict"] == "unsupported").sum())
    print(f"Downgraded to neutral by the verifier: {n_downgraded_total}/{n_analyzed} "
          f"({100 * n_downgraded_total / n_analyzed:.2f}%) -- these rows' route/decision "
          f"below reflect the DOWNGRADED stance, not the original one.")

    underpowered_log: list = []
    report = {
        "mode": mode,
        "sample_n": n_total,
        "analyzed_n": n_analyzed,
        "excluded_n": n_excluded,
        "seed": SEED,
        "test_population_n": len(test_frame),
        "test_default_rate": float(test_rate),
        "sample_default_rate": float(sample_rate),
        "stance_source_breakdown": {
            src: int(source_counts.get(src, 0)) for src in ["parsed", "parse_error", "api_error", "empty"]
        },
        "verifier_verdict_breakdown": {
            v: int(verifier_counts.get(v, 0)) for v in ["supported", "unsupported", "unclear", "skipped_neutral"]
        },
        "route_distribution": section_route_distribution(analyzed_df),
        "decisions_changed": section_decisions_changed(analyzed_df),
        "realized_rates": section_realized_rates(analyzed_df, mode == "stub", underpowered_log),
        "calibration_fp_fn": section_calibration_fp_fn(analyzed_df, underpowered_log),
        "fairness": section_fairness(analyzed_df, sample, underpowered_log),
    }

    print("\n=== 6. Underpowered guard (buckets n<30, not interpreted) ===")
    if not underpowered_log:
        print("  none -- every reported bucket met the n>=30 threshold")
    else:
        for item in underpowered_log:
            print(f"  [{item['section']}] {item['bucket']}: n={item['n']}")
    report["underpowered_buckets"] = underpowered_log
    report["records"] = records_df.to_dict(orient="records")  # ALL rows, incl. excluded, for transparency

    out_path = ROOT / "results" / ("agent_eval_fullpower.json" if mode == "full" else "agent_eval.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
