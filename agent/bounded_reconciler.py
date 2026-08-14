"""
Bounded reconciler -- a NEW, toggleable reconciliation node that EXTENDS
reconciler_agent.reconciler() with ONE bounded evidence-gathering step on
high-confidence conflicts (Case C), instead of routing every conflict
straight to human review. Cases A (agreement) and B (low-confidence/silent)
are UNCHANGED -- byte-for-byte the same classification and routing as the
current reconciler(), verified by regression tests in
test_bounded_reconciler.py.

Does NOT modify reconciler_agent.py, config/retrieval.json, or the default
graph -- this is an additive, opt-in alternative, not wired in anywhere.
Reuses (imports, never redefines) HIGH_RISK, CONF_THRESHOLD, auto_decision,
human_review, explanation from reconciler_agent, and find_similar_loans
from similar_loan_tool.py.

CASE CLASSIFICATION -- same precedence as reconciler_agent.reconciler()'s
if/elif chain: low-confidence-or-neutral is checked FIRST. This matters:
the spec's prose for Case A includes "(not risky AND neutral)", but Case B
is defined with an unconditional "OR stance == neutral" -- those overlap.
Checking Case B first (exactly like the current code) makes that overlap
resolve the same way the current system already does: ANY neutral stance
lands in Case B/low_conf regardless of tabular_risky, never in Case A. This
is what makes Case A/B byte-for-byte match the current agree/low_conf
buckets, which the spec requires and the regression tests verify.

HARD CONSTRAINTS:
  - ONE evidence action per case, executed once, no loop, no retry with
    different params -- _resolve_case_c() has exactly one, unconditional
    call to find_similar_loans() and no branch calls it twice.
  - The similar-loan tool is called AT MOST once per loan -- enforced by
    that single call site; verified in tests via a call-counting stub.
  - The empirical neighbor default_rate NEVER replaces or is treated as a
    new p_default -- it only ever feeds a binary accept-tabular/defer
    decision. Asserted below: bounded_reconciler() never returns a
    "p_default" key, i.e. never touches the tabular probability at all.
  - Case A/B behavior is IDENTICAL to reconciler_agent.reconciler() -- see
    classify_case()'s precedence above and the regression tests.

BUILD 12 FIX (DECISIONS.md Build 11 -> Build 12): Build 11's ablation showed the
original tiebreaker -- comparing the neighbor default_rate directly against
HIGH_RISK (0.1703, a deliberately LOW bar tuned for the ~15% population base
rate) -- decided WORSE than chance (40.4% match rate) on the conflicts it
auto-resolved. A neighbor rate of, say, 0.20 is technically ">= HIGH_RISK" but
is still a MINORITY outcome (80% of those neighbors repaid) -- it is weak
evidence, not strong evidence, and treating "above a low bar" as "contradicts
the narrative" was the bug. The fix: only auto-resolve when the neighbor rate is
FAR from the population base rate (STRONG_LOW/STRONG_HIGH bands below); anything
in between (WEAK) defers to human review, same as the fixed pipeline would.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
import reconciler_agent as ra   # noqa: E402 -- HIGH_RISK, CONF_THRESHOLD, auto_decision, human_review, explanation (reused, not redefined)
import similar_loan_tool as slt  # noqa: E402

# --------------------------------------------------------------------------
# Build 12 tiebreaker bands. base_rate is read from config/decision_threshold.json's
# val_default_rate (~0.1527) -- the SAME val-slice base rate HIGH_RISK itself was
# picked against (scripts/compute_decision_threshold.py), not a fresh hardcoded
# number. Reused, not redefined.
#
# STRONG_LOW_MULTIPLIER / STRONG_HIGH_MULTIPLIER (0.5x / 2.0x base_rate) define a
# margin around base_rate: a neighbor default_rate has to be at least HALF the
# base rate (unusually safe) or DOUBLE it (unusually risky) to count as strong
# evidence. Anything between is WEAK -- near-average, saying almost nothing about
# THIS loan specifically -- and the agent defers instead of acting on it.
# --------------------------------------------------------------------------
_DECISION_THRESHOLD_PATH = ROOT / "config" / "decision_threshold.json"


def _load_base_rate() -> float:
    with open(_DECISION_THRESHOLD_PATH) as f:
        return json.load(f)["val_default_rate"]


BASE_RATE = _load_base_rate()
STRONG_LOW_MULTIPLIER = 0.5
STRONG_HIGH_MULTIPLIER = 2.0
STRONG_LOW = BASE_RATE * STRONG_LOW_MULTIPLIER    # ~0.076 -- similar loans default far LESS than average
STRONG_HIGH = BASE_RATE * STRONG_HIGH_MULTIPLIER  # ~0.306 -- similar loans default far MORE than average


def classify_case(state: dict) -> Literal["A", "B", "C"]:
    """Same 3-way split, same precedence, as reconciler_agent.reconciler()."""
    tabular_risky = state["p_default"] >= ra.HIGH_RISK
    stance = state["stance"]
    conf = state["stance_confidence"]

    if conf < ra.CONF_THRESHOLD or stance == "neutral":
        return "B"
    if (tabular_risky and stance == "mitigates_risk") or ((not tabular_risky) and stance == "corroborates_risk"):
        return "C"
    return "A"


_ROUTE_BY_CASE = {"A": "agree", "B": "low_conf", "C": "disagree"}


def _resolve_case_c(state: dict) -> dict:
    """The ONE bounded evidence action for a high-confidence conflict.

    ACTION 2 (similar-loan retrieval) is the only implemented path.
    ACTION 1 (policy retrieval) and ACTION 3 (both) are named hooks for
    later -- deliberately not built out (spec: keep it extensible, don't
    over-build).

    Calls find_similar_loans() EXACTLY ONCE -- no loop, no re-call with
    different params, regardless of what it returns."""
    tabular_risky = state["p_default"] >= ra.HIGH_RISK
    stance = state["stance"]
    assert (tabular_risky and stance == "mitigates_risk") or ((not tabular_risky) and stance == "corroborates_risk"), \
        "_resolve_case_c called on a state that isn't actually a Case C conflict"

    tabular_features = state["application"]["tabular_features"]
    evidence = slt.find_similar_loans(tabular_features)   # <-- THE ONE evidence call for this case, unconditional

    if not evidence["sufficient"]:
        return {
            "action_taken": "similar_loan_retrieval",
            "evidence_used": None,
            "final_route": "human_review",
            "bounded_reason": (
                f"no comparable evidence -- fewer than {slt.MIN_NEIGHBORS} genuine neighbors in-band "
                f"(grade_band={evidence['grade_band']}, n_neighbors={evidence['n_neighbors']}); "
                f"deferring to human review."
            ),
        }

    default_rate = evidence["default_rate"]
    evidence_used = {
        "type": "similar_loan",
        "default_rate": default_rate,
        "n_neighbors": evidence["n_neighbors"],
    }
    conflict_label = "tabular_risky_vs_mitigates" if (tabular_risky and stance == "mitigates_risk") \
        else "tabular_safe_vs_corroborates"

    # Band the evidence relative to the population base rate (Build 12 fix --
    # see module docstring). Only an EXTREME rate counts as strong evidence;
    # anything near base_rate is WEAK and defers, it does not decide.
    if default_rate <= STRONG_LOW:
        band = "STRONG_LOW"
    elif default_rate >= STRONG_HIGH:
        band = "STRONG_HIGH"
    else:
        band = "WEAK"

    if band == "WEAK":
        return {
            "action_taken": "similar_loan_retrieval",
            "evidence_used": evidence_used,
            "final_route": "human_review",
            "bounded_reason": (
                f"conflict={conflict_label}; neighbor default_rate={default_rate:.3f} is WEAK evidence "
                f"(within [{STRONG_LOW:.3f}, {STRONG_HIGH:.3f}] around base_rate={BASE_RATE:.3f}, n="
                f"{evidence['n_neighbors']}) -- too close to average to overrule either side; deferring "
                f"to human review, same as the fixed pipeline would."
            ),
        }

    # tabular_risky_vs_mitigates: STRONG_LOW supports the narrative (similar
    # loans default far less than average -> the mitigation looks real).
    # tabular_safe_vs_corroborates: STRONG_HIGH supports the narrative
    # (similar loans default far more than average -> the risk concern looks
    # real). Tiebreaker only -- never a substitute probability.
    supports_narrative = (band == "STRONG_LOW") if conflict_label == "tabular_risky_vs_mitigates" \
        else (band == "STRONG_HIGH")

    if supports_narrative:
        return {
            "action_taken": "similar_loan_retrieval",
            "evidence_used": evidence_used,
            "final_route": "human_review",
            "bounded_reason": (
                f"conflict={conflict_label}; neighbor default_rate={default_rate:.3f} is {band} "
                f"(base_rate={BASE_RATE:.3f}, n={evidence['n_neighbors']}) and supports the narrative; "
                f"routing to human review with evidence."
            ),
        }
    return {
        "action_taken": "similar_loan_retrieval",
        "evidence_used": evidence_used,
        "final_route": "auto_decision",
        "bounded_reason": (
            f"conflict={conflict_label}; neighbor default_rate={default_rate:.3f} is {band} "
            f"(base_rate={BASE_RATE:.3f}, n={evidence['n_neighbors']}) and contradicts the narrative; "
            f"accepting the tabular decision."
        ),
    }


def bounded_reconciler(state: dict) -> dict:
    """Drop-in, toggleable alternative to reconciler_agent.reconciler() +
    _route() + auto_decision()/human_review() combined into one node.

    Case A/B: IDENTICAL behavior to the current reconciler -- no evidence
    gathering, accepts the tabular decision immediately.
    Case C: ONE bounded evidence action (see _resolve_case_c), then routes
    -- may resolve directly to auto_decision (evidence contradicts the
    narrative) instead of always deferring to human_review, which is the
    whole point of this extension over the current always-defer behavior.

    Additive keys only: route (same 3 values as before, for anything still
    reading it) + decision (existing key) + action_taken/evidence_used/
    final_route/bounded_reason (new)."""
    case = classify_case(state)
    route = _ROUTE_BY_CASE[case]

    if case in ("A", "B"):
        reason = (
            "agreement -- narrative confirms the tabular decision" if case == "A"
            else "low confidence or neutral narrative -- no opinion to weigh, deferring to tabular"
        )
        out = {
            "route": route,
            "action_taken": "accept_tabular",
            "evidence_used": None,
            "final_route": "auto_decision",
            "bounded_reason": reason,
        }
    else:
        out = {"route": route, **_resolve_case_c(state)}

    decision = (
        ra.auto_decision(state)["decision"] if out["final_route"] == "auto_decision"
        else ra.human_review(state)["decision"]
    )
    out["decision"] = decision

    assert "p_default" not in out, \
        "bounded_reconciler must never write p_default -- evidence is a tiebreaker, never a new probability"
    return out


def bounded_explanation(state: dict) -> dict:
    """Mirrors reconciler_agent.explanation() (reused, not duplicated) but
    threads the bounded reconciler's extra keys into the memo too."""
    base_memo = ra.explanation(state)["memo"]
    memo = (
        f"{base_memo}\n"
        f"Bounded action: {state.get('action_taken')}\n"
        f"Evidence used: {state.get('evidence_used')}\n"
        f"Final route: {state.get('final_route')}\n"
        f"Bounded reason: {state.get('bounded_reason')}"
    )
    return {"memo": memo}


if __name__ == "__main__":
    import pickle

    import pandas as pd

    sys.path.insert(0, str(ROOT / "scripts"))
    from eval_agent import RAW_TABULAR_COLS  # noqa: E402

    print("=== bounded_reconciler.py -- one example per case ===\n")

    frame = pd.read_pickle(ROOT / "data" / "interim" / "feasibility_frame.pkl")
    with open(ROOT / "data" / "interim" / "split_indices.pkl", "rb") as f:
        split = pickle.load(f)
    sample_loan_id = int(split["test_idx"][0])
    tabular_features = {c: frame.loc[sample_loan_id, c] for c in RAW_TABULAR_COLS}

    p_default, _shap_drivers = ra._score_tabular(tabular_features)
    tabular_risky = p_default >= ra.HIGH_RISK
    conflict_stance = "mitigates_risk" if tabular_risky else "corroborates_risk"
    agree_stance = "corroborates_risk" if tabular_risky else "mitigates_risk"

    print(f"Real loan_id={sample_loan_id}: p_default={p_default:.4f} "
          f"(tabular_risky={tabular_risky}, HIGH_RISK={ra.HIGH_RISK:.4f})\n")

    examples = [
        ("CASE A -- agreement", {
            "application": {"tabular_features": tabular_features, "desc_clean": ""},
            "p_default": p_default, "stance": agree_stance, "stance_confidence": 0.85,
        }),
        ("CASE B -- low confidence", {
            "application": {"tabular_features": tabular_features, "desc_clean": ""},
            "p_default": p_default, "stance": conflict_stance, "stance_confidence": 0.20,
        }),
        ("CASE C -- high-confidence conflict (real similar-loan lookup)", {
            "application": {"tabular_features": tabular_features, "desc_clean": ""},
            "p_default": p_default, "stance": conflict_stance, "stance_confidence": 0.85,
        }),
    ]

    for label, state in examples:
        case = classify_case(state)
        out = bounded_reconciler(state)
        print(f"{label}  (classify_case -> {case})")
        print(f"  stance={state['stance']!r}  confidence={state['stance_confidence']}")
        print(f"  route={out['route']!r}  action_taken={out['action_taken']!r}  "
              f"final_route={out['final_route']!r}  decision={out['decision']!r}")
        print(f"  evidence_used={out['evidence_used']}")
        print(f"  bounded_reason={out['bounded_reason']}")
        print()
