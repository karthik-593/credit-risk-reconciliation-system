"""
Headline eval for the tabular/text reconciliation agent (agent/reconciler_agent.py),
run over the locked TEST split's desc-only population. Three modes, one flag:

  (default) STUB  -- deterministic, hash-keyed stub stances. Zero LLM tokens.
                      Exercises every route bucket + the underpowered guard.
  --validate      -- 40 REAL LLM calls only. Reports parse-success rate,
                      evidence-grounding rate, stance distribution. Gate
                      before spending tokens on the full run.
  --real          -- full SAMPLE_N run with the real adapter (agent/llm_client.py).
                      Every stance cached by loan_id to
                      results/eval_stance_cache.pkl so re-runs are free.

Population: split_indices.pkl TEST indices ONLY -- never train (train was
tuning's; TEST is read here, once, for evaluation, same as the tuning
notebook's own single test-touching cell). Rows come from
feasibility_frame.pkl, the desc-only population the tabular models were both
built and scored on. A base-rate-preserved random sample (SAMPLE_N, seed 42)
is drawn from TEST.

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
OUT_PATH = ROOT / "results" / "agent_eval.json"

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
# Running one loan through the same node sequence build_graph() wires.
# ---------------------------------------------------------------------------
def run_one(row: pd.Series, loan_id, cache: dict | None = None) -> dict:
    tabular_features = {c: row[c] for c in RAW_TABULAR_COLS}
    application = {"tabular_features": tabular_features, "desc_clean": row["desc_clean"]}
    state = {"application": application}

    state.update(ra.tabular_score(state))

    if cache is not None and loan_id in cache:
        stance_out = cache[loan_id]
    else:
        stance_out = ra.text_stance(state)
        if cache is not None:
            cache[loan_id] = stance_out
    state.update(stance_out)

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

    print(f"=== VALIDATE: {len(sample)} real LLM calls ===")

    source_counts = {"parsed": 0, "parse_error": 0, "api_error": 0, "empty": 0}
    stance_counts = {"corroborates_risk": 0, "mitigates_risk": 0, "neutral": 0}
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

    print("\nGate check: review the above before running --real.")


# ---------------------------------------------------------------------------
# stub / real: full run
# ---------------------------------------------------------------------------
def run_full(sample: pd.DataFrame, mode: str) -> pd.DataFrame:
    cache = {}
    if mode == "real":
        from llm_client import configure_from_config
        configure_from_config()
        if CACHE_PATH.exists():
            with open(CACHE_PATH, "rb") as f:
                cache = pickle.load(f)
            print(f"Loaded {len(cache)} cached stances from {CACHE_PATH}")
    else:
        ra.configure_llm_client(StubLLMClient())

    records = []
    for i, (loan_id, row) in enumerate(sample.iterrows()):
        state = run_one(row, loan_id, cache=cache if mode == "real" else None)
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
        })
        if mode == "real" and (i + 1) % 100 == 0:
            with open(CACHE_PATH, "wb") as f:
                pickle.dump(cache, f)
            print(f"  ... {i + 1}/{len(sample)} (cache checkpointed)")

    if mode == "real":
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(cache, f)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------
def bucket_stats(df: pd.DataFrame, mask, label: str, underpowered_log: list) -> dict:
    n = int(mask.sum())
    if n < UNDERPOWERED_N:
        print(f"  [{label}] n={n}  <-- UNDERPOWERED (<{UNDERPOWERED_N}), not interpreted")
        underpowered_log.append({"section": "realized_rates", "bucket": label, "n": n})
        return {"n": n, "underpowered": True}
    rate = float(df.loc[mask, "y"].mean())
    print(f"  [{label}] n={n}  realized_default_rate={rate:.4f}")
    return {"n": n, "underpowered": False, "realized_default_rate": rate}


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
        holds = (a["realized_default_rate"] > b["realized_default_rate"]) if op == ">" \
            else (a["realized_default_rate"] < b["realized_default_rate"])
        print(f"  {expect_label}: {'HOLDS' if holds else 'DOES NOT HOLD'}")
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
    agree_mask = df["route"] == "agree"
    n_agree = int(agree_mask.sum())
    print(f"\n  Brier, full sample (n={len(df)}): {brier_full:.4f}")
    if n_agree >= UNDERPOWERED_N:
        brier_agree = brier(df[agree_mask])
        calibration_lift = brier_full - brier_agree
        print(f"  Brier, auto-decided (agree) subset (n={n_agree}): {brier_agree:.4f}")
        print(f"  Calibration lift: {calibration_lift:+.4f}")
    else:
        brier_agree = None
        calibration_lift = None
        print(f"  agree subset UNDERPOWERED (n={n_agree}), skipping calibration lift")
        underpowered_log.append({"section": "calibration", "bucket": "agree_subset", "n": n_agree})

    fp_baseline_mask = (df["tabular_alone_decision"] == "decline") & (df["y"] == 0)
    fn_baseline_mask = (df["tabular_alone_decision"] == "approve") & (df["y"] == 1)
    fp_baseline = int(fp_baseline_mask.sum())
    fn_baseline = int(fn_baseline_mask.sum())

    deferred_mask = df["route"] != "agree"
    fp_deferred = int((fp_baseline_mask & deferred_mask).sum())
    fn_deferred = int((fn_baseline_mask & deferred_mask).sum())
    fp_automated = fp_baseline - fp_deferred
    fn_automated = fn_baseline - fn_deferred

    print(f"\n  Tabular-alone baseline (everyone auto-decided): "
          f"FP={fp_baseline} ({100 * fp_baseline / len(df):.2f}%)  "
          f"FN={fn_baseline} ({100 * fn_baseline / len(df):.2f}%)")
    print(f"  Of those, deferred to human review instead of auto-executed: "
          f"FP={fp_deferred}  FN={fn_deferred}")
    print(f"  Remaining automated (agree-route) errors: FP={fp_automated}  FN={fn_automated}")

    review_rate = 100 * int(deferred_mask.sum()) / len(df)
    within_budget = review_rate <= REVIEW_BUDGET_PCT
    print(f"\n  Review rate: {review_rate:.2f}% vs ~{REVIEW_BUDGET_PCT:.0f}% budget "
          f"({'within' if within_budget else 'OVER'} budget)")

    return {
        "brier_full": brier_full,
        "brier_agree_only": brier_agree,
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
    mode_group.add_argument("--real", action="store_true", help="Full run with the real LLM adapter; caches by loan_id.")
    args = parser.parse_args()

    mode = "real" if args.real else ("validate" if args.validate else "stub")
    print(f"=== eval_agent.py -- mode={mode} ===\n")

    test_frame = load_test_population()
    test_rate = test_frame["default"].mean()
    print(f"TEST population: {len(test_frame):,} rows, default rate {test_rate:.4f}")

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

    if mode == "real":
        n_api_error = int((records_df["stance_source"] == "api_error").sum())
        if n_api_error > 0:
            print("\n" + "!" * 78)
            print(f"WARNING: {n_api_error}/{len(records_df)} rows hit api_error (network/rate-limit/")
            print("quota/timeout after retries) and fell back to stance=neutral, confidence=0.0.")
            print("Those rows are NOT real model verdicts -- they inflate low_conf/disagree-rate")
            print("and contaminate every section below. Check results/agent_eval.json's per-record")
            print("stance_source field before trusting this report; re-run once the underlying")
            print("failure is resolved.")
            print("!" * 78)

    print("\n" + "=" * 78)
    print(f"AGENT EVAL REPORT (mode={mode}, n={len(records_df)})")
    print("=" * 78)
    if mode == "stub":
        print("STUB MODE: every stance below is hash-derived noise, not a real narrative")
        print("read. That means every number downstream of route/stance -- review rate,")
        print("FP/FN deltas, fairness shifts, all of it -- reflects the stub's synthetic")
        print("confidence distribution, not agent behavior. This run verifies the eval")
        print("mechanics and the underpowered guard only. Re-run with --real for findings.")

    underpowered_log: list = []
    report = {
        "mode": mode,
        "sample_n": len(records_df),
        "seed": SEED,
        "test_population_n": len(test_frame),
        "test_default_rate": float(test_rate),
        "sample_default_rate": float(sample_rate),
        "route_distribution": section_route_distribution(records_df),
        "decisions_changed": section_decisions_changed(records_df),
        "realized_rates": section_realized_rates(records_df, mode == "stub", underpowered_log),
        "calibration_fp_fn": section_calibration_fp_fn(records_df, underpowered_log),
        "fairness": section_fairness(records_df, sample, underpowered_log),
    }

    print("\n=== 6. Underpowered guard (buckets n<30, not interpreted) ===")
    if not underpowered_log:
        print("  none -- every reported bucket met the n>=30 threshold")
    else:
        for item in underpowered_log:
            print(f"  [{item['section']}] {item['bucket']}: n={item['n']}")
    report["underpowered_buckets"] = underpowered_log
    report["records"] = records_df.to_dict(orient="records")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()
