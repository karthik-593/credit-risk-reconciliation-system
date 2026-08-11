"""
Reframes the reconciler agent's routing as a selective-prediction /
review-efficiency problem: if human review capacity is a fixed budget (not
free), how many of the REAL defaults get caught depending on which loans
are sent to review first?

Read-only over results/agent_eval_fullpower.json's cached records (21,616
real loans, already evaluated). Does not modify agent/reconciler_agent.py,
does not re-run the tabular model or the LLM -- purely a re-analysis of
already-computed p_default, route, and realized y.

Two orderings, compared honestly against a random-review floor:
  - TABULAR BASELINE : rank all loans by p_default descending.
  - AGENT ROUTING     : disagree first, then low_conf, then agree; within
                        each group, by p_default descending -- the order
                        the agent would actually hand loans to a reviewer
                        if review capacity were rationed instead of the
                        11.6% it happened to produce.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EVAL_PATH = ROOT / "results" / "agent_eval_fullpower.json"
OUT_JSON = ROOT / "results" / "review_efficiency.json"
OUT_FIG = ROOT / "results" / "review_efficiency.png"

ROUTE_PRIORITY = {"disagree": 0, "low_conf": 1, "agree": 2}
BUDGETS_TO_REPORT = [5, 10, 15, 20]


def load_records():
    with open(EVAL_PATH) as f:
        report = json.load(f)
    records = report["records"]
    n_before = len(records)
    records = [r for r in records if r.get("stance_source", "parsed") == "parsed"]
    if len(records) != n_before:
        print(f"Excluded {n_before - len(records)} non-'parsed' stance_source rows "
              f"(infrastructure failures, not real verdicts).")
    return records, report


def capture_curve(records_sorted):
    """Cumulative fraction of ALL real defaults captured, at every
    population-fraction boundary from 0 to 1 (one point per record)."""
    y = np.array([r["y"] for r in records_sorted], dtype=float)
    total_defaults = y.sum()
    cum = np.concatenate([[0.0], np.cumsum(y)])
    capture_frac = cum / total_defaults
    budget_frac = np.arange(len(cum)) / (len(cum) - 1)
    return budget_frac, capture_frac


def value_at_budget(budget_frac, capture_frac, target_pct):
    idx = int(round((target_pct / 100.0) * (len(budget_frac) - 1)))
    idx = min(max(idx, 0), len(budget_frac) - 1)
    return float(budget_frac[idx]), float(capture_frac[idx])


def curve_auc(budget_frac, capture_frac):
    return float(np.trapezoid(capture_frac, budget_frac))


def main():
    records, report = load_records()
    n = len(records)
    total_defaults = int(sum(r["y"] for r in records))
    base_rate = total_defaults / n
    print(f"Records analyzed: {n:,}")
    print(f"Total real defaults (y==1): {total_defaults:,} ({100 * base_rate:.2f}% base rate)\n")

    tabular_sorted = sorted(records, key=lambda r: -r["p_default"])
    agent_sorted = sorted(records, key=lambda r: (ROUTE_PRIORITY[r["route"]], -r["p_default"]))

    tb_x, tb_y = capture_curve(tabular_sorted)
    ag_x, ag_y = capture_curve(agent_sorted)

    tabular_auc = curve_auc(tb_x, tb_y)
    agent_auc = curve_auc(ag_x, ag_y)
    random_auc = 0.5
    theoretical_max_auc = 1 - base_rate / 2
    auc_gap = agent_auc - tabular_auc

    print("AUC of the review-efficiency curve (higher = catches real defaults sooner):")
    print(f"  Random floor:      {random_auc:.4f}")
    print(f"  Tabular baseline:  {tabular_auc:.4f}")
    print(f"  Agent routing:     {agent_auc:.4f}")
    print(f"  Theoretical max:   {theoretical_max_auc:.4f}  (given {100 * base_rate:.2f}% base rate; "
          f"no ordering can exceed this)")
    print(f"  Agent - Tabular AUC gap: {auc_gap:+.4f}\n")

    print(f"{'Budget':>8s} {'Tabular':>10s} {'Agent':>10s} {'Gap (Agent-Tabular)':>22s}")
    per_budget = {}
    for b in BUDGETS_TO_REPORT:
        _, tb_val = value_at_budget(tb_x, tb_y, b)
        _, ag_val = value_at_budget(ag_x, ag_y, b)
        gap = ag_val - tb_val
        print(f"{b:>7d}% {tb_val:>9.2%} {ag_val:>9.2%} {gap:>+21.2%}")
        per_budget[str(b)] = {"tabular_capture": tb_val, "agent_capture": ag_val, "gap": gap}
    print()

    # Actual agent review rate: disagree is first in the agent ordering and
    # its size IS the real review count, so "top X% of the agent ordering"
    # at X=disagree rate is exactly "all disagree-routed loans".
    n_disagree = sum(1 for r in records if r["route"] == "disagree")
    actual_review_rate_pct = 100 * n_disagree / n
    _, ag_val_actual = value_at_budget(ag_x, ag_y, actual_review_rate_pct)
    _, tb_val_actual = value_at_budget(tb_x, tb_y, actual_review_rate_pct)
    print(f"At the agent's ACTUAL review rate ({actual_review_rate_pct:.2f}%, "
          f"n={n_disagree:,} disagree-routed loans):")
    print(f"  Agent routing captures:    {ag_val_actual:.2%} of all real defaults")
    print(f"  Tabular baseline captures: {tb_val_actual:.2%} of all real defaults at the same budget")
    print(f"  Gap: {ag_val_actual - tb_val_actual:+.2%}\n")

    # --- Honest verdict ---
    if abs(auc_gap) < 0.01:
        verdict = ("COMPETITIVE with the tabular baseline -- the curves are close (AUC gap "
                   f"{auc_gap:+.4f}); the agent's text-aware queue does not clearly beat "
                   "ranking by score alone, and does not clearly lose to it either.")
    elif auc_gap > 0:
        verdict = (f"Marginally BETTER than tabular-alone by AUC ({auc_gap:+.4f}), though check "
                   "the per-budget gaps above -- an AUC edge can hide budgets where it's flat or worse.")
    else:
        verdict = (f"WORSE than tabular-alone by AUC ({auc_gap:+.4f}) -- at this operating point, "
                   "ranking by the tabular score alone front-loads real defaults faster than the "
                   "agent's disagree-first queue does.")
    print("HONEST VERDICT:", verdict, "\n")

    # Complementary single-point framing already established (Build 7) --
    # not recomputed, just carried forward from the same source JSON.
    cal = report["calibration_fp_fn"]
    fp_catch_pct = 100 * cal["fp_deferred"] / cal["fp_baseline"]
    fn_catch_pct = 100 * cal["fn_deferred"] / cal["fn_baseline"]
    print(f"Complementary single-point framing (Build 7, unchanged): disagreement routing "
          f"catches {fp_catch_pct:.0f}% of the tabular model's false approvals vs only "
          f"{fn_catch_pct:.0f}% of its false declines -- an asymmetric safety net, not a "
          f"general review-efficiency win (see curve above for that question).")

    # --- Plot: B/W-friendly (line style, not just colour) ---
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
        "font.size": 11,
    })
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(ag_x * 100, ag_y * 100, linestyle="-", color="#3b6ea5", linewidth=2.2,
            label="Agent routing (disagree → low_conf → agree)")
    ax.plot(tb_x * 100, tb_y * 100, linestyle="--", color="#c0453a", linewidth=2.2,
            label="Tabular baseline (rank by p(default))")
    ax.plot([0, 100], [0, 100], linestyle=":", color="#7a7a7a", linewidth=1.8,
            label="Random review (floor)")

    _, tb_15 = value_at_budget(tb_x, tb_y, 15)
    _, ag_15 = value_at_budget(ag_x, ag_y, 15)
    ax.axvline(15, color="#444444", linewidth=0.8, linestyle=(0, (1, 3)))
    ax.plot([15], [tb_15 * 100], marker="o", markersize=7, color="#c0453a", zorder=5)
    ax.plot([15], [ag_15 * 100], marker="o", markersize=7, color="#3b6ea5", zorder=5)
    label_y = min(max(tb_15, ag_15) * 100 + 6, 92)
    ax.annotate(f"15% budget: tabular {tb_15:.0%}, agent {ag_15:.0%}",
                xy=(15, label_y), fontsize=9, color="#333333")

    ax.set_xlabel("Review budget (% of all loans reviewed)")
    ax.set_ylabel("% of real defaults captured")
    ax.set_title("Review efficiency: agent routing vs. tabular score vs. random")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=150)
    print(f"\nSaved figure to {OUT_FIG}")

    output = {
        "n_records": n,
        "total_defaults": total_defaults,
        "base_rate": base_rate,
        "auc": {
            "random": random_auc,
            "tabular": tabular_auc,
            "agent": agent_auc,
            "theoretical_max": theoretical_max_auc,
            "agent_minus_tabular": auc_gap,
        },
        "per_budget_pct": per_budget,
        "actual_review_rate_pct": actual_review_rate_pct,
        "actual_review_n_disagree": n_disagree,
        "at_actual_review_rate": {
            "agent_capture": ag_val_actual,
            "tabular_capture": tb_val_actual,
            "gap": ag_val_actual - tb_val_actual,
        },
        "complementary_single_point_build7": {
            "fp_catch_pct": fp_catch_pct,
            "fn_catch_pct": fn_catch_pct,
        },
        "verdict": verdict,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved {OUT_JSON}")


if __name__ == "__main__":
    main()
