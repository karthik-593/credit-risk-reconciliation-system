"""
Ablation: FIXED pipeline (reconciler_agent.reconciler()) vs AGENTIC pipeline
(bounded_reconciler.bounded_reconciler()) -- same loans, same tabular score,
same stance, same (possibly verifier-downgraded) stance; the ONLY thing that
differs is the reconciliation step. Measures whether the bounded evidence-
gathering layer (Case C conflicts call the similar-loan tool once) actually
improves decisions, not just whether it changes them.

Does not modify reconciler_agent.py, bounded_reconciler.py,
similar_loan_tool.py, config/retrieval.json, or any eval script -- imports
and calls their real functions directly, unmodified.

SAMPLE: 5000 loans, base-rate-preserved, seed=42, from the locked TEST split
-- NOT the full 21,616 (stated explicitly, not silently, per instructions).
Why 5000 and not full:
  - results/eval_stance_cache.pkl covers the ENTIRE TEST population
    (verified: 21,616/21,616 loan_ids present) -- every stance at any sample
    size is free, zero new LLM calls, regardless of n.
  - results/eval_verifier_cache.pkl (2000 loans, from eval_agent.py --real)
    and results/eval_verifier_sample_cache.pkl (2500 loans, from
    eval_verifier.py) both used base_rate_sample(seed=42) -- verified live
    that they NEST inside a fresh n=5000 draw of the same function
    (sample_2000 subset-of sample_2500 subset-of sample_5000), so 2500 of
    the 5000 verifier verdicts are ALSO free. Only the remaining ~2500
    loans need a fresh verifier call (result cached separately to
    results/ablation_verifier_cache.pkl -- never written into either of the
    two existing shared cache files, avoiding any collision).
  - Going to the full 21,616 would need ~19,616 fresh verifier calls
    (hours, mostly idle LLM latency) to move a number that's already
    well-estimated at n=5000 -- not worth it for what this ablation asks.
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
import reconciler_agent as ra    # noqa: E402
import bounded_reconciler as br  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from eval_agent import load_test_population, base_rate_sample, wilson_ci, RAW_TABULAR_COLS  # noqa: E402

sys.path.insert(0, str(ROOT / "experiments"))
from tracking import track_run  # noqa: E402
import mlflow  # noqa: E402

STANCE_CACHE_PATH = ROOT / "results" / "eval_stance_cache.pkl"
VERIFIER_CACHE_2000_PATH = ROOT / "results" / "eval_verifier_cache.pkl"
VERIFIER_CACHE_2500_PATH = ROOT / "results" / "eval_verifier_sample_cache.pkl"
ABLATION_VERIFIER_CACHE_PATH = ROOT / "results" / "ablation_verifier_cache.pkl"
OUT_JSON = ROOT / "experiments" / "ablation_results.json"

SAMPLE_N = 5000
SEED = 42
UNDERPOWERED_N = 30


def run_one(row: pd.Series, loan_id: int, stance_cache: dict, verifier_cache: dict) -> dict:
    tabular_features = {c: row[c] for c in RAW_TABULAR_COLS}
    application = {"tabular_features": tabular_features, "desc_clean": row["desc_clean"]}
    state = {"application": application}
    state.update(ra.tabular_score(state))

    stance_out = stance_cache[loan_id]   # guaranteed present -- full-population cache
    state.update(stance_out)

    verifier_was_cached = loan_id in verifier_cache
    if verifier_was_cached:
        verifier_out = verifier_cache[loan_id]
    else:
        verifier_out = ra.verifier(state)
        verifier_cache[loan_id] = verifier_out
    state.update(verifier_out)

    # From here `state` holds the FINAL (possibly verifier-downgraded)
    # stance -- identical starting point fed to BOTH pipelines below.
    case = br.classify_case(state)

    # --- Pipeline A: FIXED (the existing, unmodified reconciler) ---
    fixed_route_out = ra.reconciler(state)
    fixed_state = {**state, **fixed_route_out}
    fixed_next = ra._route(fixed_state)
    fixed_decision = (
        ra.auto_decision(fixed_state) if fixed_next == "auto_decision" else ra.human_review(fixed_state)
    )["decision"]

    # --- Pipeline B: AGENTIC (bounded reconciler) ---
    agentic_out = br.bounded_reconciler(state)

    assert fixed_route_out["route"] == agentic_out["route"], (
        "case classification must be identical between pipelines -- they only "
        "diverge in what happens AFTER a Case C conflict is identified"
    )

    tabular_risky = state["p_default"] >= ra.HIGH_RISK
    tabular_alone_decision = "decline" if tabular_risky else "approve"

    return {
        "loan_id": loan_id,
        "y": int(row["default"]),
        "p_default": state["p_default"],
        "tabular_alone_decision": tabular_alone_decision,
        "stance": state["stance"],
        "stance_confidence": state["stance_confidence"],
        "verifier_verdict": state.get("verifier_verdict"),
        "verifier_was_cached": verifier_was_cached,
        "case": case,
        "route_fixed": fixed_route_out["route"],
        "decision_fixed": fixed_decision,
        "route_agentic": agentic_out["route"],
        "decision_agentic": agentic_out["decision"],
        "action_taken": agentic_out["action_taken"],
        "evidence_used": agentic_out["evidence_used"],
        "final_route_agentic": agentic_out["final_route"],
        "bounded_reason": agentic_out["bounded_reason"],
    }


def _rate(mask: pd.Series, n_total: int) -> dict:
    k = int(mask.sum())
    if n_total == 0:
        return {"n": 0, "k": 0, "rate": None, "ci_low": None, "ci_high": None}
    rate = k / n_total
    ci_low, ci_high = wilson_ci(k, n_total)
    return {"n": n_total, "k": k, "rate": rate, "ci_low": ci_low, "ci_high": ci_high}


def _ci_overlap(a: dict, b: dict) -> bool:
    return a["ci_low"] <= b["ci_high"] and b["ci_low"] <= a["ci_high"]


def main():
    print("=== ablation_agentic.py -- FIXED vs AGENTIC reconciler ===\n")

    print(f"Step 1: sample {SAMPLE_N} loans, base-rate-preserved, seed={SEED}, from locked TEST ...")
    test_frame = load_test_population()
    sample = base_rate_sample(test_frame, SAMPLE_N, SEED)
    print(f"  {len(sample)} loans, default rate {sample['default'].mean():.4f} "
          f"(TEST full population: {len(test_frame):,})\n")

    print("Step 2: load stance cache (must cover the full sample -- no new stance calls) ...")
    with open(STANCE_CACHE_PATH, "rb") as f:
        stance_cache = pickle.load(f)
    missing = [i for i in sample.index if int(i) not in stance_cache]
    assert not missing, f"{len(missing)} sampled loans have no cached stance -- expected full coverage"
    print(f"  {len(stance_cache):,} stances cached; all {len(sample)} sampled loans covered, 0 new stance calls.\n")

    print("Step 3: load + merge verifier caches (2000-loan + 2500-loan, both seed=42, nested) ...")
    verifier_cache: dict = {}
    for path in (VERIFIER_CACHE_2000_PATH, VERIFIER_CACHE_2500_PATH, ABLATION_VERIFIER_CACHE_PATH):
        if path.exists():
            with open(path, "rb") as f:
                loaded = pickle.load(f)
            verifier_cache.update(loaded)
            print(f"  merged {len(loaded):,} entries from {path.name} (running total: {len(verifier_cache):,})")
    n_already_cached = sum(1 for i in sample.index if int(i) in verifier_cache)
    print(f"  {n_already_cached}/{len(sample)} sampled loans already have a cached verifier verdict; "
          f"~{len(sample) - n_already_cached} new verifier calls expected (only for non-neutral stances).\n")

    print(f"Step 4: running {len(sample)} loans through both pipelines ...")
    t_start = time.perf_counter()
    records = []
    n_new_verifier_calls = 0
    for i, (raw_loan_id, row) in enumerate(sample.iterrows()):
        loan_id = int(raw_loan_id)
        was_cached_before = loan_id in verifier_cache
        rec = run_one(row, loan_id, stance_cache, verifier_cache)
        if not was_cached_before:
            n_new_verifier_calls += 1
        records.append(rec)
        if (i + 1) % 250 == 0:
            elapsed = time.perf_counter() - t_start
            print(f"  ... {i + 1}/{len(sample)}  ({n_new_verifier_calls} new verifier calls so far, "
                  f"{elapsed:.0f}s elapsed)")
            with open(ABLATION_VERIFIER_CACHE_PATH, "wb") as f:
                pickle.dump(verifier_cache, f)

    with open(ABLATION_VERIFIER_CACHE_PATH, "wb") as f:
        pickle.dump(verifier_cache, f)
    print(f"  done: {len(records)} loans, {n_new_verifier_calls} new verifier LLM calls "
          f"({time.perf_counter() - t_start:.0f}s total).\n")

    df = pd.DataFrame(records)
    n_total = len(df)

    # -----------------------------------------------------------------
    # 1. Review rate: FIXED vs AGENTIC
    # -----------------------------------------------------------------
    review_fixed = _rate(df["decision_fixed"] == "human_review", n_total)
    review_agentic = _rate(df["decision_agentic"] == "human_review", n_total)

    # -----------------------------------------------------------------
    # 2 & 3. Loans B auto-decided that A would have deferred -- the ONLY
    # subset where fixed and agentic can ever differ (Case A/B are
    # identical by construction/regression tests; Case C either matches A
    # (still human_review) or resolves to auto_decision). Confirmed below
    # with an assertion, not just asserted in prose.
    # -----------------------------------------------------------------
    differ_mask = df["decision_fixed"] != df["decision_agentic"]
    resolved_mask = (df["case"] == "C") & (df["final_route_agentic"] == "auto_decision")
    assert (differ_mask == resolved_mask).all(), \
        "fixed and agentic decisions differ ONLY where agentic auto-resolved a Case C conflict -- invariant broken"
    n_resolved = int(resolved_mask.sum())

    resolved_default_rate = _rate(df.loc[resolved_mask, "y"] == 1, n_resolved)

    correct_mask = (
        ((df["decision_agentic"] == "auto_decline") & (df["y"] == 1))
        | ((df["decision_agentic"] == "auto_approve") & (df["y"] == 0))
    ) & resolved_mask
    decision_match = _rate(correct_mask, n_resolved)
    naive_always_approve_baseline = 1 - sample["default"].mean()   # accuracy of "always guess no-default"

    # -----------------------------------------------------------------
    # 4. Similar-loan tool firing rate on Case C conflicts
    # -----------------------------------------------------------------
    case_c_mask = df["case"] == "C"
    n_case_c = int(case_c_mask.sum())
    sufficient_mask = case_c_mask & df["evidence_used"].notna()
    insufficient_mask = case_c_mask & df["evidence_used"].isna()
    sufficient_stats = _rate(sufficient_mask, n_case_c)

    # -----------------------------------------------------------------
    # Print report
    # -----------------------------------------------------------------
    print("=" * 90)
    print(f"ABLATION REPORT  (n={n_total} loans, seed={SEED})")
    print("=" * 90)

    print("\n-- 1. Review rate --")
    print(f"  FIXED:   {review_fixed['rate']:.2%}  [{review_fixed['ci_low']:.2%}, {review_fixed['ci_high']:.2%}]  "
          f"(k={review_fixed['k']}/{review_fixed['n']})")
    print(f"  AGENTIC: {review_agentic['rate']:.2%}  [{review_agentic['ci_low']:.2%}, {review_agentic['ci_high']:.2%}]  "
          f"(k={review_agentic['k']}/{review_agentic['n']})")
    overlap = _ci_overlap(review_fixed, review_agentic)
    if overlap:
        print("  95% CIs OVERLAP -- NOT a statistically significant reduction at this n.")
    else:
        direction = "LOWER" if review_agentic["rate"] < review_fixed["rate"] else "HIGHER"
        print(f"  95% CIs do NOT overlap -- agentic review rate is SIGNIFICANTLY {direction}.")

    print(f"\n-- 2. Of the {n_resolved} conflicts agentic resolved to accept-tabular "
          f"(that fixed would have sent to human review) --")
    if n_resolved < UNDERPOWERED_N:
        print(f"  UNDERPOWERED (n={n_resolved} < {UNDERPOWERED_N}) -- not interpreted.")
    else:
        print(f"  Actual default rate: {resolved_default_rate['rate']:.2%}  "
              f"[{resolved_default_rate['ci_low']:.2%}, {resolved_default_rate['ci_high']:.2%}]")
        print(f"  Agent's decision matches the actual outcome: {decision_match['rate']:.2%}  "
              f"[{decision_match['ci_low']:.2%}, {decision_match['ci_high']:.2%}]")
        print(f"  Naive 'always approve' baseline accuracy on the FULL sample: {naive_always_approve_baseline:.2%} "
              f"(reference point, not this subset's own baseline)")
        beats_naive = decision_match["ci_low"] > naive_always_approve_baseline
        beats_chance = decision_match["ci_low"] > 0.5
        print(f"  Beats naive always-approve baseline (CI lower bound > {naive_always_approve_baseline:.2%})? "
              f"{'YES' if beats_naive else 'NO'}")
        print(f"  Beats a coin flip (CI lower bound > 50%)? {'YES' if beats_chance else 'NO'}")

    print(f"\n-- 3. On the {n_resolved} loans where fixed and agentic differ --")
    print("  Fixed defers ALL of these to human review -- it commits to no approve/decline call here, "
          "so it has no accuracy to score against y.")
    if n_resolved >= UNDERPOWERED_N:
        print(f"  Agentic commits to a decision (identical to the tabular-alone call, by construction of "
              f"the 'evidence contradicts narrative -> accept tabular' branch) that matches the actual "
              f"outcome {decision_match['rate']:.2%} of the time (same figure as section 2 above).")
    else:
        print(f"  UNDERPOWERED (n={n_resolved} < {UNDERPOWERED_N}) -- not interpreted.")

    print(f"\n-- 4. Similar-loan tool firing rate on the {n_case_c} Case C conflicts --")
    print(f"  sufficient=True:  {sufficient_stats['rate']:.2%}  "
          f"[{sufficient_stats['ci_low']:.2%}, {sufficient_stats['ci_high']:.2%}]  (k={sufficient_stats['k']})")
    print(f"  sufficient=False: {1 - sufficient_stats['rate']:.2%}  (k={n_case_c - sufficient_stats['k']})")

    # -----------------------------------------------------------------
    # Honest verdict
    # -----------------------------------------------------------------
    print("\n" + "=" * 90)
    print("HONEST VERDICT")
    print("=" * 90)
    review_reduced_significantly = (not overlap) and review_agentic["rate"] < review_fixed["rate"]
    quality_ok = n_resolved >= UNDERPOWERED_N and decision_match["ci_low"] > naive_always_approve_baseline

    if n_resolved < UNDERPOWERED_N:
        verdict = (
            f"CHANGED LITTLE -- only {n_resolved} conflicts were resolved without human review at this "
            f"sample size, too few to say whether decision quality held up. The agentic layer barely "
            f"fired; no measurable benefit demonstrated at n={n_total}."
        )
    elif review_reduced_significantly and quality_ok:
        verdict = (
            "WIN -- the agentic layer reduced review load by a statistically significant margin "
            "WITHOUT a measurable drop in decision quality on the loans it resolved on its own "
            "(decision-match rate's CI lower bound clears the naive always-approve baseline)."
        )
    elif review_reduced_significantly and not quality_ok:
        verdict = (
            "TRADEOFF -- the agentic layer significantly reduced review load, but the decisions it made "
            "on the loans it resolved on its own do NOT clearly beat a naive always-approve baseline at "
            "this n. It is deferring less, not clearly deciding better."
        )
    else:
        verdict = (
            "CHANGED LITTLE -- review rate did not drop by a statistically significant margin at this "
            "sample size. The agentic layer adds complexity without a demonstrated benefit here."
        )
    print(verdict)

    # -----------------------------------------------------------------
    # MLflow
    # -----------------------------------------------------------------
    print("\nLogging to MLflow experiment 'ablation_agentic' (local ./mlruns) ...")
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "sample_n": n_total,
        "seed": SEED,
        "sample_default_rate": float(sample["default"].mean()),
        "n_new_verifier_calls": n_new_verifier_calls,
        "review_rate_fixed": review_fixed,
        "review_rate_agentic": review_agentic,
        "n_resolved_without_human": n_resolved,
        "resolved_conflict_default_rate": resolved_default_rate if n_resolved >= UNDERPOWERED_N else None,
        "decision_match_rate": decision_match if n_resolved >= UNDERPOWERED_N else None,
        "naive_always_approve_baseline": naive_always_approve_baseline,
        "n_case_c": n_case_c,
        "sufficient_rate": sufficient_stats,
        "verdict": verdict,
        "records": records,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2, default=str)

    with track_run(experiment="ablation_agentic", run_name="fixed", params={"system": "fixed", "n_notes": n_total}):
        mlflow.log_metrics({"review_rate": review_fixed["rate"]})
        mlflow.log_artifact(str(OUT_JSON))

    agentic_metrics = {"review_rate": review_agentic["rate"], "sufficient_rate": sufficient_stats["rate"] or 0.0}
    if n_resolved >= UNDERPOWERED_N:
        agentic_metrics["resolved_conflict_default_rate"] = resolved_default_rate["rate"]
        agentic_metrics["decision_match_rate"] = decision_match["rate"]
    with track_run(experiment="ablation_agentic", run_name="agentic", params={"system": "agentic", "n_notes": n_total}):
        mlflow.log_metrics(agentic_metrics)
        mlflow.log_artifact(str(OUT_JSON))

    print(f"Saved {OUT_JSON}")
    print("Logged 2 MLflow runs (fixed, agentic) under experiment 'ablation_agentic'.")


if __name__ == "__main__":
    main()
