"""
Tests for agent/bounded_reconciler.py.

Script-style, matching this directory's other test files -- run directly:
    python agent/test_bounded_reconciler.py

Case A/B tests are REGRESSION tests against reconciler_agent.reconciler()/
_route()/auto_decision()/human_review() -- the bounded reconciler must
produce byte-for-byte the same route+decision as the current pipeline for
every non-conflict case. A stub that raises on any call is installed for
these so an accidental evidence-gathering call fails loudly rather than
quietly hitting the real (slow, data-backed) tool.

Case C tests use a call-counting STUB similar-loan tool (no real lookup,
no data files needed) for fast, deterministic, isolated behavior tests --
exactly the sufficient/insufficient, STRONG_LOW/STRONG_HIGH/WEAK shapes the
spec calls out. The WEAK test (test_case_c_weak_evidence_defers_to_human_review)
is the Build 11 -> Build 12 regression check: a 0.20 neighbor rate is what
broke Build 11 (it's >= the old HIGH_RISK=0.17 comparison but is still a
near-base-rate, weak signal) -- it must now defer, not auto-decide.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
import reconciler_agent as ra    # noqa: E402
import bounded_reconciler as br  # noqa: E402


def _old_pipeline_route_and_decision(state: dict) -> tuple[str, str]:
    """The CURRENT (unmodified) reconciler -> _route -> auto_decision/
    human_review pipeline, for regression comparison."""
    route = ra.reconciler(state)["route"]
    full_state = {**state, "route": route}
    next_node = ra._route(full_state)
    decision = (ra.auto_decision(full_state) if next_node == "auto_decision" else ra.human_review(full_state))["decision"]
    return route, decision


def base_state(p_default, stance, confidence, tabular_features=None):
    return {
        "application": {"tabular_features": tabular_features or {}, "desc_clean": "n/a"},
        "p_default": p_default,
        "stance": stance,
        "stance_confidence": confidence,
    }


def _forbidden_tool(*args, **kwargs):
    raise AssertionError("similar-loan tool must not be called for Case A/B")


def _counting_stub(result: dict, counter: dict):
    def _fn(features):
        counter["n"] = counter["n"] + 1
        return result
    return _fn


# ---------------------------------------------------------------------------
# Case A / B regression -- MUST match the current reconciler exactly.
# ---------------------------------------------------------------------------
REGRESSION_CASES = [
    # (name, p_default, stance, confidence)
    ("agree-risky-corroborates", 0.60, "corroborates_risk", 0.90),
    ("agree-safe-mitigates", 0.05, "mitigates_risk", 0.90),
    ("agree-safe-neutral-high-conf", 0.05, "neutral", 0.90),      # neutral -> low_conf regardless of confidence
    ("low-conf-below-threshold", 0.60, "mitigates_risk", 0.10),   # would be Case C direction, but conf too low
    ("low-conf-neutral-risky", 0.60, "neutral", 0.90),
    ("low-conf-neutral-safe", 0.05, "neutral", 0.30),
]


def test_case_a_b_match_current_reconciler_exactly():
    original = br.slt.find_similar_loans
    br.slt.find_similar_loans = _forbidden_tool
    try:
        for name, p, stance, conf in REGRESSION_CASES:
            state = base_state(p, stance, conf)
            old_route, old_decision = _old_pipeline_route_and_decision(state)
            case = br.classify_case(state)
            assert case in ("A", "B"), f"[{name}] expected case A/B, classify_case returned {case!r}"

            new = br.bounded_reconciler(state)
            assert new["route"] == old_route, f"[{name}] route mismatch: {new['route']!r} != {old_route!r}"
            assert new["decision"] == old_decision, \
                f"[{name}] decision mismatch: {new['decision']!r} != {old_decision!r}"
            assert new["action_taken"] == "accept_tabular", f"[{name}] expected no evidence action"
            assert new["evidence_used"] is None
            print(f"    [{name}] case={case} route={new['route']:9s} decision={new['decision']}")
    finally:
        br.slt.find_similar_loans = original


# ---------------------------------------------------------------------------
# Case C -- stubbed similar-loan tool, deterministic evidence.
# ---------------------------------------------------------------------------
def test_case_c_low_rate_supports_narrative_routes_to_human_review():
    state = base_state(p_default=0.60, stance="mitigates_risk", confidence=0.90,
                        tabular_features={"grade": "A"})
    assert br.classify_case(state) == "C"

    counter = {"n": 0}
    original = br.slt.find_similar_loans
    br.slt.find_similar_loans = _counting_stub(
        {"n_neighbors": 50, "default_rate": 0.02, "sufficient": True, "grade_band": "A"}, counter,
    )
    try:
        out = br.bounded_reconciler(state)
    finally:
        br.slt.find_similar_loans = original

    assert counter["n"] == 1, "similar-loan tool must be called exactly once"
    assert out["final_route"] == "human_review"
    assert out["decision"] == "human_review"
    assert out["action_taken"] == "similar_loan_retrieval"
    assert out["evidence_used"] == {"type": "similar_loan", "default_rate": 0.02, "n_neighbors": 50}


def test_case_c_high_rate_contradicts_narrative_accepts_tabular():
    state = base_state(p_default=0.60, stance="mitigates_risk", confidence=0.90,
                        tabular_features={"grade": "A"})
    assert br.classify_case(state) == "C"

    counter = {"n": 0}
    original = br.slt.find_similar_loans
    br.slt.find_similar_loans = _counting_stub(
        {"n_neighbors": 50, "default_rate": 0.50, "sufficient": True, "grade_band": "A"}, counter,
    )
    try:
        out = br.bounded_reconciler(state)
    finally:
        br.slt.find_similar_loans = original

    assert counter["n"] == 1
    assert out["final_route"] == "auto_decision"
    assert out["decision"] == "auto_decline"           # p_default=0.60 >= HIGH_RISK -> decline
    assert out["action_taken"] == "similar_loan_retrieval"
    assert out["evidence_used"]["default_rate"] == 0.50


def test_case_c_weak_evidence_defers_to_human_review():
    """Build 11 -> Build 12 regression check: a neighbor rate of 0.20 is near
    BASE_RATE (~0.153, within [STRONG_LOW~0.076, STRONG_HIGH~0.305]) -- WEAK
    evidence. Build 11's tiebreaker (comparing against HIGH_RISK=0.17) would
    have called this "contradicts the narrative" and auto-decided; it must
    now defer to human review instead, exactly like the fixed pipeline."""
    state = base_state(p_default=0.60, stance="mitigates_risk", confidence=0.90,
                        tabular_features={"grade": "A"})
    assert br.classify_case(state) == "C"
    assert br.STRONG_LOW < 0.20 < br.STRONG_HIGH, "test fixture assumption broken -- 0.20 must be WEAK"

    counter = {"n": 0}
    original = br.slt.find_similar_loans
    br.slt.find_similar_loans = _counting_stub(
        {"n_neighbors": 50, "default_rate": 0.20, "sufficient": True, "grade_band": "A"}, counter,
    )
    try:
        out = br.bounded_reconciler(state)
    finally:
        br.slt.find_similar_loans = original

    assert counter["n"] == 1, "similar-loan tool must be called exactly once"
    assert out["final_route"] == "human_review", \
        f"WEAK evidence (0.20) must defer to human review, got final_route={out['final_route']!r}"
    assert out["decision"] == "human_review"
    assert out["action_taken"] == "similar_loan_retrieval"
    assert out["evidence_used"]["default_rate"] == 0.20
    assert "WEAK" in out["bounded_reason"]


def test_case_c_insufficient_evidence_routes_to_human_review():
    state = base_state(p_default=0.05, stance="corroborates_risk", confidence=0.90,
                        tabular_features={"grade": "G"})
    assert br.classify_case(state) == "C"

    counter = {"n": 0}
    original = br.slt.find_similar_loans
    br.slt.find_similar_loans = _counting_stub(
        {"n_neighbors": 4, "default_rate": None, "sufficient": False, "grade_band": "G"}, counter,
    )
    try:
        out = br.bounded_reconciler(state)
    finally:
        br.slt.find_similar_loans = original

    assert counter["n"] == 1
    assert out["final_route"] == "human_review"
    assert out["decision"] == "human_review"
    assert out["evidence_used"] is None
    assert "no comparable evidence" in out["bounded_reason"]


def test_case_c_safe_vs_corroborates_high_rate_supports_narrative():
    """Mirror conflict direction: tabular=safe, narrative=corroborates_risk."""
    state = base_state(p_default=0.05, stance="corroborates_risk", confidence=0.90,
                        tabular_features={"grade": "A"})
    assert br.classify_case(state) == "C"

    counter = {"n": 0}
    original = br.slt.find_similar_loans
    br.slt.find_similar_loans = _counting_stub(
        {"n_neighbors": 50, "default_rate": 0.40, "sufficient": True, "grade_band": "A"}, counter,
    )
    try:
        out = br.bounded_reconciler(state)
    finally:
        br.slt.find_similar_loans = original

    assert counter["n"] == 1
    assert out["final_route"] == "human_review"
    assert out["decision"] == "human_review"


def test_case_c_safe_vs_corroborates_low_rate_accepts_tabular():
    state = base_state(p_default=0.05, stance="corroborates_risk", confidence=0.90,
                        tabular_features={"grade": "A"})
    assert br.classify_case(state) == "C"

    counter = {"n": 0}
    original = br.slt.find_similar_loans
    br.slt.find_similar_loans = _counting_stub(
        {"n_neighbors": 50, "default_rate": 0.03, "sufficient": True, "grade_band": "A"}, counter,
    )
    try:
        out = br.bounded_reconciler(state)
    finally:
        br.slt.find_similar_loans = original

    assert counter["n"] == 1
    assert out["final_route"] == "auto_decision"
    assert out["decision"] == "auto_approve"           # p_default=0.05 < HIGH_RISK -> approve


def test_bounded_explanation_threads_new_keys_into_memo():
    state = base_state(p_default=0.60, stance="mitigates_risk", confidence=0.90,
                        tabular_features={"grade": "A"})
    state.update({
        "shap_drivers": [], "stance_evidence": [], "stance_policy_ids": [],
        "stance_rationale": "", "verifier_verdict": "supported",
        "verifier_source": "llm", "verifier_reason": "",
    })
    counter = {"n": 0}
    original = br.slt.find_similar_loans
    br.slt.find_similar_loans = _counting_stub(
        {"n_neighbors": 50, "default_rate": 0.02, "sufficient": True, "grade_band": "A"}, counter,
    )
    try:
        state.update(br.bounded_reconciler(state))
        memo = br.bounded_explanation(state)["memo"]
    finally:
        br.slt.find_similar_loans = original

    for token in ("Bounded action:", "Evidence used:", "Final route:", "Bounded reason:"):
        assert token in memo, f"memo missing {token!r}"


if __name__ == "__main__":
    test_case_a_b_match_current_reconciler_exactly()
    print("PASS  Case A/B match the current reconciler exactly (regression, similar-loan tool never called)")

    test_case_c_low_rate_supports_narrative_routes_to_human_review()
    print("PASS  Case C, low neighbor rate (risky+mitigates) -> human_review with evidence")

    test_case_c_high_rate_contradicts_narrative_accepts_tabular()
    print("PASS  Case C, high neighbor rate (risky+mitigates) -> accept tabular (auto_decline)")

    test_case_c_weak_evidence_defers_to_human_review()
    print("PASS  Case C, WEAK neighbor rate (0.20, risky+mitigates) -> DEFER to human review "
          "(the Build 11 bug, now fixed)")

    test_case_c_insufficient_evidence_routes_to_human_review()
    print("PASS  Case C, insufficient evidence -> human_review, reason cites 'no comparable evidence'")

    test_case_c_safe_vs_corroborates_high_rate_supports_narrative()
    print("PASS  Case C, safe+corroborates + high neighbor rate -> human_review with evidence")

    test_case_c_safe_vs_corroborates_low_rate_accepts_tabular()
    print("PASS  Case C, safe+corroborates + low neighbor rate -> accept tabular (auto_approve)")

    test_bounded_explanation_threads_new_keys_into_memo()
    print("PASS  bounded_explanation threads action_taken/evidence_used/final_route/bounded_reason into the memo")

    print("\nAll bounded_reconciler tests passed.")
