"""
Fast, sample-based measurement of what the VERIFIER node does to the
reconciler's routing: does grounding-checking the word-reader's stance
shrink the disagree rate, and does the resulting review queue become a
CLEANER catch of the tabular model's errors (fewer disagreements, but more
precisely targeted) -- not the 21,616-loan full run, a validation-scale
measurement.

Read-only over already-computed artifacts:
  - results/agent_eval_fullpower.json  (p_default, tabular_alone_decision, y
    per loan, for the entire locked TEST population)
  - results/eval_stance_cache.pkl      (loan_id -> full stance detail:
    stance, evidence, cited policy ids, confidence, rationale)
  - data/interim/feasibility_frame.pkl (loan_id -> desc_clean)
  - data/interim/split_indices.pkl     (the TEST population definition)

Only the verifier's LLM calls are new (local Qwen, temperature 0) -- the
stance itself is always read from cache, never re-called. Reuses the real
agent.reconciler_agent.verifier()/reconciler()/_route()/auto_decision()/
human_review() functions directly (not a reimplementation), so this
measures the actual production logic. Does not modify
agent/reconciler_agent.py.

Uses its own verifier cache (results/eval_verifier_sample_cache.pkl),
separate from scripts/eval_agent.py's (results/eval_verifier_cache.pkl), so
this can run safely alongside a concurrent eval_agent.py run without either
one clobbering the other's cache file.
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
import reconciler_agent as ra  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_agent import wilson_ci, UNDERPOWERED_N  # reuse, don't duplicate

FULLPOWER_PATH = ROOT / "results" / "agent_eval_fullpower.json"
STANCE_CACHE_PATH = ROOT / "results" / "eval_stance_cache.pkl"
FRAME_PATH = ROOT / "data" / "interim" / "feasibility_frame.pkl"
SPLIT_PATH = ROOT / "data" / "interim" / "split_indices.pkl"
VERIFIER_SAMPLE_CACHE_PATH = ROOT / "results" / "eval_verifier_sample_cache.pkl"
OUT_JSON = ROOT / "results" / "eval_verifier.json"

SAMPLE_N = 2500
SEED = 42
REQUIRED_STANCE_FIELDS = {"stance", "stance_evidence", "stance_policy_ids", "stance_confidence", "stance_rationale"}


def _safe_load_pickle(path, retries=3, delay=1.0):
    """A concurrent eval_agent.py run may be checkpointing this exact file
    (re-saving the full stance cache periodically). Retry on read failure
    rather than risk reading mid-write."""
    last_exc = None
    for _ in range(retries):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as exc:
            last_exc = exc
            time.sleep(delay)
    raise last_exc


def check_cache_shape() -> dict:
    print("=" * 78)
    print("STEP 0: cache shape check")
    print("=" * 78)
    cache = _safe_load_pickle(STANCE_CACHE_PATH)
    print(f"results/eval_stance_cache.pkl: {len(cache):,} cached stances")

    sample_key = next(iter(cache))
    sample_val = cache[sample_key]
    print(f"\nExample entry (loan_id={sample_key}):")
    for k, v in sample_val.items():
        text = repr(v)
        if len(text) > 150:
            text = text[:150] + "...(truncated)"
        print(f"  {k}: {text}")

    missing = REQUIRED_STANCE_FIELDS - set(sample_val.keys())
    if missing:
        print(f"\nMISSING required fields: {sorted(missing)}.")
        print("Cannot run the verifier without a fresh stance call to backfill these -- "
              "stopping, need a different plan.")
        sys.exit(1)
    print(f"\nAll required fields present {sorted(REQUIRED_STANCE_FIELDS)} -- the verifier can "
          f"run entirely from cache; ONLY new verifier LLM calls are needed, zero stance re-calls.")
    return cache


def main():
    stance_cache = check_cache_shape()

    print("\n" + "=" * 78)
    print("Loading population + drawing sample")
    print("=" * 78)
    with open(FULLPOWER_PATH) as f:
        fullpower = json.load(f)
    fullpower_by_id = {r["loan_id"]: r for r in fullpower["records"]}

    with open(SPLIT_PATH, "rb") as f:
        split = pickle.load(f)
    test_idx = list(split["test_idx"])  # the locked TEST population, per the task's instruction

    frame = pd.read_pickle(FRAME_PATH)
    desc_by_id = frame["desc_clean"].to_dict()

    missing_from_fullpower = [lid for lid in test_idx if lid not in fullpower_by_id]
    if missing_from_fullpower:
        print(f"NOTE: {len(missing_from_fullpower)} TEST loan_ids missing from "
              f"agent_eval_fullpower.json's records -- excluded from the sampling pool.")
    pool_ids = [lid for lid in test_idx if lid in fullpower_by_id]
    y_by_id = {lid: fullpower_by_id[lid]["y"] for lid in pool_ids}
    pool = pd.Series(y_by_id)

    base_rate = pool.mean()
    n_pos = min(round(SAMPLE_N * base_rate), int((pool == 1).sum()))
    n_neg = min(SAMPLE_N - n_pos, int((pool == 0).sum()))
    pos_ids = pool[pool == 1].sample(n=n_pos, random_state=SEED).index.tolist()
    neg_ids = pool[pool == 0].sample(n=n_neg, random_state=SEED).index.tolist()
    sample_ids = pos_ids + neg_ids
    np.random.RandomState(SEED).shuffle(sample_ids)

    sample_rate = pool.loc[sample_ids].mean()
    print(f"TEST population (split_indices.pkl test_idx): {len(test_idx):,} loans")
    print(f"Sample: {len(sample_ids):,} loans (seed={SEED}), default rate {sample_rate:.4f} "
          f"(target ~0.1526)")

    print("\n" + "=" * 78)
    print("Running verifier (stance from cache only; new LLM calls where needed)")
    print("=" * 78)
    from llm_client import configure_from_config
    configure_from_config()

    verifier_cache: dict = {}
    if VERIFIER_SAMPLE_CACHE_PATH.exists():
        verifier_cache = _safe_load_pickle(VERIFIER_SAMPLE_CACHE_PATH)
        print(f"Loaded {len(verifier_cache)} cached verifier verdicts from a prior run of this script")

    records = []
    n_llm_calls = 0
    n_cache_hits = 0
    for i, loan_id in enumerate(sample_ids):
        fp_record = fullpower_by_id[loan_id]
        p_default = fp_record["p_default"]
        y = fp_record["y"]
        tabular_alone_decision = fp_record["tabular_alone_decision"]

        stance_detail = stance_cache.get(loan_id)
        desc_clean = desc_by_id.get(loan_id, "")
        if stance_detail is None:
            continue  # shouldn't happen -- fullpower covers all of TEST

        state = {
            "application": {"desc_clean": desc_clean},
            "stance": stance_detail["stance"],
            "stance_evidence": stance_detail.get("stance_evidence", []),
            "stance_policy_ids": stance_detail.get("stance_policy_ids", []),
            "stance_confidence": stance_detail.get("stance_confidence", 0.0),
            "stance_rationale": stance_detail.get("stance_rationale", ""),
        }

        # PRE-verifier route: what Build 7 already measured, recomputed on
        # THIS sample for a fair apples-to-apples baseline.
        pre_state = {**state, "p_default": p_default}
        pre_route = ra.reconciler(pre_state)["route"]

        if loan_id in verifier_cache:
            verifier_out = verifier_cache[loan_id]
            n_cache_hits += 1
        else:
            verifier_out = ra.verifier(state)
            verifier_cache[loan_id] = verifier_out
            if verifier_out["verifier_source"] == "llm":
                n_llm_calls += 1

        post_state = {**state, **verifier_out, "p_default": p_default}
        post_route = ra.reconciler(post_state)["route"]
        routed_state = {**post_state, "route": post_route}
        next_node = ra._route(routed_state)
        decision = (ra.auto_decision(routed_state) if next_node == "auto_decision"
                    else ra.human_review(routed_state))["decision"]

        records.append({
            "loan_id": loan_id,
            "p_default": p_default,
            "y": y,
            "tabular_alone_decision": tabular_alone_decision,
            "stance": stance_detail["stance"],
            "verifier_verdict": verifier_out["verifier_verdict"],
            "verifier_source": verifier_out["verifier_source"],
            "downgraded": "stance" in verifier_out,
            "pre_route": pre_route,
            "post_route": post_route,
            "agent_decision": decision,
        })

        if (i + 1) % 250 == 0:
            with open(VERIFIER_SAMPLE_CACHE_PATH, "wb") as f:
                pickle.dump(verifier_cache, f)
            print(f"  ... {i + 1}/{len(sample_ids)} ({n_llm_calls} new LLM calls so far)")

    with open(VERIFIER_SAMPLE_CACHE_PATH, "wb") as f:
        pickle.dump(verifier_cache, f)
    print(f"\nDone: {len(records)} loans processed, {n_llm_calls} new verifier LLM calls, "
          f"{n_cache_hits} cache hits from a prior run of this script.")

    df = pd.DataFrame(records)
    n = len(df)
    underpowered_log: list = []

    def report_rate(label, k, n_, section):
        if n_ < UNDERPOWERED_N:
            print(f"  [{label}] n={n_}  <-- UNDERPOWERED (<{UNDERPOWERED_N}), not interpreted")
            underpowered_log.append({"section": section, "bucket": label, "n": n_})
            return {"n": n_, "underpowered": True}
        rate = k / n_
        lo, hi = wilson_ci(k, n_)
        print(f"  [{label}] n={n_}  rate={rate:.4f}  95% CI=[{lo:.4f}, {hi:.4f}]")
        return {"n": n_, "underpowered": False, "k": k, "rate": rate, "ci_low": lo, "ci_high": hi}

    # === 1. Verifier verdict breakdown ===
    print("\n" + "=" * 78)
    print("1. Verifier verdict breakdown")
    print("=" * 78)
    verdict_counts = df["verifier_verdict"].value_counts()
    for v in ["supported", "unsupported", "unclear", "skipped_neutral"]:
        c = int(verdict_counts.get(v, 0))
        print(f"  {v:16s}: {c:5d} ({100 * c / n:.2f}%)")
    source_counts = df["verifier_source"].value_counts()
    print("  by source:")
    for s in ["mechanical", "llm", "skipped"]:
        c = int(source_counts.get(s, 0))
        print(f"    {s:14s}: {c:5d} ({100 * c / n:.2f}%)")
    n_downgraded = int(df["downgraded"].sum())
    print(f"  Downgraded to neutral: {n_downgraded}/{n} ({100 * n_downgraded / n:.2f}%)")

    # === 2. Route distribution + disagree rate, pre vs post ===
    print("\n" + "=" * 78)
    print("2. Route distribution: PRE- vs POST-verifier, disagree rate")
    print("=" * 78)
    print("PRE-verifier route distribution (this sample):")
    for r in ["agree", "disagree", "low_conf"]:
        c = int((df["pre_route"] == r).sum())
        print(f"  {r:10s}: {c:5d} ({100 * c / n:.2f}%)")
    print("POST-verifier route distribution (this sample):")
    for r in ["agree", "disagree", "low_conf"]:
        c = int((df["post_route"] == r).sum())
        print(f"  {r:10s}: {c:5d} ({100 * c / n:.2f}%)")

    pre_disagree_n = int((df["pre_route"] == "disagree").sum())
    post_disagree_n = int((df["post_route"] == "disagree").sum())
    print()
    pre_disagree_stats = report_rate("pre-verifier disagree rate (this sample)", pre_disagree_n, n, "disagree_rate")
    post_disagree_stats = report_rate("post-verifier disagree rate (this sample)", post_disagree_n, n, "disagree_rate")

    build7_disagree_rate_pct = fullpower["route_distribution"]["disagree_rate_pct"]
    print(f"\n  Build 7 (n=21,616) disagree rate for reference: {build7_disagree_rate_pct:.2f}%")
    if not pre_disagree_stats["underpowered"] and not post_disagree_stats["underpowered"]:
        pre_pct = pre_disagree_stats["rate"] * 100
        post_pct = post_disagree_stats["rate"] * 100
        shrink_pp = pre_pct - post_pct
        rel_shrink = 100 * shrink_pp / pre_pct if pre_pct else float("nan")
        print(f"  The verifier shrinks the disagree rate by {shrink_pp:.2f}pp on this sample "
              f"({pre_pct:.2f}% -> {post_pct:.2f}%), a {rel_shrink:.1f}% relative reduction.")

    # === 3. Decisions changed by the verifier ===
    print("\n" + "=" * 78)
    print("3. Decisions changed by the verifier (disagree -> low_conf via downgrade)")
    print("=" * 78)
    flipped_mask = (df["pre_route"] == "disagree") & (df["post_route"] == "low_conf")
    n_flipped = int(flipped_mask.sum())
    if pre_disagree_n > 0:
        flip_stats = report_rate("disagree -> low_conf (of pre-verifier disagree loans)",
                                  n_flipped, pre_disagree_n, "decisions_changed")
    else:
        print("  No pre-verifier disagree loans in this sample -- nothing to flip.")
        flip_stats = {"n": pre_disagree_n, "underpowered": True}

    # === 4. Catch rate on tabular errors: pre vs post ===
    print("\n" + "=" * 78)
    print("4. Catch rate on tabular errors: PRE vs POST verifier (this sample)")
    print("=" * 78)
    print("  Definitions (matches Build 7's code; NOTE the README mislabels these, see below):")
    print("  FALSE DECLINE  = tabular declined a loan that was actually fine")
    print("                   (tabular_alone_decision==decline & y==0)")
    print("  FALSE APPROVAL = tabular approved a loan that actually defaulted")
    print("                   (tabular_alone_decision==approve & y==1)")

    false_decline_mask = (df["tabular_alone_decision"] == "decline") & (df["y"] == 0)
    false_approval_mask = (df["tabular_alone_decision"] == "approve") & (df["y"] == 1)
    n_false_decline = int(false_decline_mask.sum())
    n_false_approval = int(false_approval_mask.sum())
    print(f"\n  False declines in sample: {n_false_decline}")
    print(f"  False approvals in sample: {n_false_approval}")

    cal = fullpower["calibration_fp_fn"]
    build7_false_decline_catch_pct = 100 * cal["fp_deferred"] / cal["fp_baseline"]
    build7_false_approval_catch_pct = 100 * cal["fn_deferred"] / cal["fn_baseline"]

    pre_fd_caught = int((false_decline_mask & (df["pre_route"] == "disagree")).sum())
    post_fd_caught = int((false_decline_mask & (df["post_route"] == "disagree")).sum())
    pre_fa_caught = int((false_approval_mask & (df["pre_route"] == "disagree")).sum())
    post_fa_caught = int((false_approval_mask & (df["post_route"] == "disagree")).sum())

    print("\n  False-decline catch rate -- this is the metric the README calls \"24%\" "
          "(mislabeled there as \"false approvals\"; it is actually false DECLINES,")
    print("  per the code's own fp_baseline_mask definition -- see the flag in DECISIONS.md Build 9):")
    pre_fd_stats = report_rate("PRE-verifier", pre_fd_caught, n_false_decline, "fd_catch")
    post_fd_stats = report_rate("POST-verifier", post_fd_caught, n_false_decline, "fd_catch")
    print(f"  Build 7 (n=21,616) reference: {build7_false_decline_catch_pct:.2f}%")

    print("\n  False-approval catch rate (Build 7's actual \"6%\" number):")
    pre_fa_stats = report_rate("PRE-verifier", pre_fa_caught, n_false_approval, "fa_catch")
    post_fa_stats = report_rate("POST-verifier", post_fa_caught, n_false_approval, "fa_catch")
    print(f"  Build 7 (n=21,616) reference: {build7_false_approval_catch_pct:.2f}%")

    print("\n  Precision of the disagree queue (of loans sent to review, % that were a real")
    print("  tabular error -- false decline OR false approval): does grounding make it CLEANER?")
    pre_disagree_mask = df["pre_route"] == "disagree"
    post_disagree_mask = df["post_route"] == "disagree"
    pre_precision_hits = int((pre_disagree_mask & (false_decline_mask | false_approval_mask)).sum())
    post_precision_hits = int((post_disagree_mask & (false_decline_mask | false_approval_mask)).sum())
    pre_precision_stats = report_rate("PRE-verifier queue precision", pre_precision_hits, pre_disagree_n, "precision")
    post_precision_stats = report_rate("POST-verifier queue precision", post_precision_hits, post_disagree_n, "precision")

    # === 5. Underpowered guard ===
    print("\n" + "=" * 78)
    print("5. Underpowered guard (buckets n<30, not interpreted)")
    print("=" * 78)
    if not underpowered_log:
        print("  none -- every reported bucket met the n>=30 threshold")
    else:
        for item in underpowered_log:
            print(f"  [{item['section']}] {item['bucket']}: n={item['n']}")

    output = {
        "sample_n": n,
        "sample_default_rate": float(sample_rate),
        "seed": SEED,
        "n_new_llm_calls": n_llm_calls,
        "n_cache_hits_this_run": n_cache_hits,
        "verifier_verdict_counts": {v: int(verdict_counts.get(v, 0)) for v in
                                     ["supported", "unsupported", "unclear", "skipped_neutral"]},
        "verifier_source_counts": {s: int(source_counts.get(s, 0)) for s in ["mechanical", "llm", "skipped"]},
        "n_downgraded": n_downgraded,
        "pre_verifier_disagree": pre_disagree_stats,
        "post_verifier_disagree": post_disagree_stats,
        "build7_disagree_rate_pct_reference": build7_disagree_rate_pct,
        "disagree_to_low_conf_flips": flip_stats,
        "false_decline_catch": {
            "pre": pre_fd_stats, "post": post_fd_stats,
            "build7_reference_pct": build7_false_decline_catch_pct,
            "note": "Build 7's README mislabels this 'false approvals'; it is false declines per the code.",
        },
        "false_approval_catch": {
            "pre": pre_fa_stats, "post": post_fa_stats,
            "build7_reference_pct": build7_false_approval_catch_pct,
        },
        "disagree_queue_precision": {"pre": pre_precision_stats, "post": post_precision_stats},
        "underpowered_buckets": underpowered_log,
        "records": records,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved {OUT_JSON}")


if __name__ == "__main__":
    main()
