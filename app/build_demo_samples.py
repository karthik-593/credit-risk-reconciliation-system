"""
Build-time only. Produces the two small JSON files the deployed Streamlit
app actually ships and reads: app/demo_samples.json and
app/policy_corpus.json. Run this locally (where data/, models/, and
results/ all exist); the deployed app never touches those directories or
imports reconciler_agent.py -- it only reads the two JSON files this script
writes.

Joins three sources that each hold part of the picture:
  - results/agent_eval_fullpower.json  ("records": p_default, route,
    decision, tabular_alone_decision, realized y -- but NOT evidence/
    rationale/policy citations, those were never in the per-record eval
    output)
  - results/eval_stance_cache.pkl      (loan_id -> full stance dict:
    evidence spans, cited policy ids, rationale)
  - data/interim/feasibility_frame.pkl (loan_id -> desc_clean, the actual
    borrower text)

Stratifies by (route, tabular_alone_decision) -- 6 combinations -- so the
shipped sample spans every headline scenario the app explains: agreement on
both an approve and a decline call, genuine disagreement in both directions,
and low-confidence/silent-text fallback on both sides.
"""
import json
import pickle
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "agent"))

EVAL_PATH = ROOT / "results" / "agent_eval_fullpower.json"
CACHE_PATH = ROOT / "results" / "eval_stance_cache.pkl"
FRAME_PATH = ROOT / "data" / "interim" / "feasibility_frame.pkl"

SAMPLES_PER_GROUP = 34  # 6 groups x 34 ~= 204, close to the requested ~200
SEED = 42
MIN_DESC_LEN = 15  # skip near-empty desc_clean where possible


def main():
    import reconciler_agent as ra  # read-only: just reads POLICY_CORPUS, no model load

    with open(EVAL_PATH) as f:
        eval_report = json.load(f)
    records = eval_report["records"]

    with open(CACHE_PATH, "rb") as f:
        stance_cache = pickle.load(f)

    frame = pd.read_pickle(FRAME_PATH)
    desc_by_loan_id = frame["desc_clean"].to_dict()

    joined = []
    for r in records:
        loan_id = r["loan_id"]
        desc = desc_by_loan_id.get(loan_id)
        stance_detail = stance_cache.get(loan_id)
        if desc is None or stance_detail is None:
            continue
        joined.append({
            "loan_id": loan_id,
            "desc_clean": desc,
            "p_default": r["p_default"],
            "tabular_alone_decision": r["tabular_alone_decision"],
            "stance": r["stance"],
            "confidence": r["confidence"],
            "route": r["route"],
            "agent_decision": r["agent_decision"],
            "y": r["y"],
            "stance_source": r["stance_source"],
            "stance_evidence": stance_detail.get("stance_evidence", []),
            "stance_policy_ids": stance_detail.get("stance_policy_ids", []),
            "stance_rationale": stance_detail.get("stance_rationale", ""),
        })

    print(f"Joined {len(joined)}/{len(records)} records (desc_clean + cached stance detail both found)")

    rng = random.Random(SEED)
    groups = {}
    for row in joined:
        key = (row["route"], row["tabular_alone_decision"])
        groups.setdefault(key, []).append(row)

    print("\nGroup sizes (route, tabular_alone_decision):")
    for key, rows in sorted(groups.items()):
        print(f"  {key}: {len(rows)}")

    selected = []
    for key, rows in sorted(groups.items()):
        rng.shuffle(rows)
        substantive = [row for row in rows if len(row["desc_clean"]) >= MIN_DESC_LEN]
        pool = substantive if len(substantive) >= SAMPLES_PER_GROUP else rows
        selected.extend(pool[:SAMPLES_PER_GROUP])

    rng.shuffle(selected)
    print(f"\nSelected {len(selected)} demo loans across {len(groups)} groups")

    # Give each sample a short, human label for the selectbox.
    for row in selected:
        snippet = row["desc_clean"][:60].strip()
        if len(row["desc_clean"]) > 60:
            snippet += "..."
        row["label"] = f"#{row['loan_id']} [{row['route']}] \"{snippet}\""

    with open(APP_DIR / "demo_samples.json", "w") as f:
        json.dump(selected, f, indent=2)
    print(f"Wrote {APP_DIR / 'demo_samples.json'} ({len(selected)} loans)")

    with open(APP_DIR / "policy_corpus.json", "w") as f:
        json.dump(ra.POLICY_CORPUS, f, indent=2)
    print(f"Wrote {APP_DIR / 'policy_corpus.json'} ({len(ra.POLICY_CORPUS)} policies)")

    # Headline stats, sourced directly from the real eval JSON -- not
    # hardcoded, so they can never silently drift from the actual result.
    rd = eval_report["route_distribution"]
    cal = eval_report["calibration_fp_fn"]
    fp_catch_pct = 100 * cal["fp_deferred"] / cal["fp_baseline"]
    fn_catch_pct = 100 * cal["fn_deferred"] / cal["fn_baseline"]
    headline = {
        "n_loans_evaluated": eval_report["analyzed_n"],
        "disagree_rate_pct": rd["disagree_rate_pct"],
        "review_rate_pct": cal["review_rate_pct"],
        "review_budget_pct": cal["review_budget_pct"],
        "within_budget": cal["within_budget"],
        "fp_catch_pct": fp_catch_pct,
        "fn_catch_pct": fn_catch_pct,
        "calibration_lift": cal["calibration_lift"],
    }
    with open(APP_DIR / "headline_stats.json", "w") as f:
        json.dump(headline, f, indent=2)
    print(f"Wrote {APP_DIR / 'headline_stats.json'}")
    print(json.dumps(headline, indent=2))


if __name__ == "__main__":
    main()
