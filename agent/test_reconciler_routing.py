"""
Smoke test for reconciler routing logic. Exercises reconciler() -> _route()
-> {auto_decision, human_review} -> explanation() directly with stubbed
(p_default, stance, confidence) triples, bypassing tabular_score/text_stance
entirely (no trained model or LLM client required to run this).
"""
from reconciler_agent import reconciler, _route, auto_decision, human_review, explanation

CASES = [
    {
        "name": "agree-risky",
        "state": {
            "p_default": 0.72,
            "shap_drivers": [("int_rate", 0.9), ("grade", 0.6)],
            "stance": "corroborates_risk",
            "stance_confidence": 0.80,
            "stance_evidence": ["I lost my job last month"],
            "stance_policy_ids": ["policy_2.2"],
            "stance_rationale": "Applicant discloses recent job loss with no stated re-employment.",
        },
        "expect_route": "agree",
        "expect_decision": "auto_decline",
    },
    {
        "name": "agree-safe",
        "state": {
            "p_default": 0.12,
            "shap_drivers": [("grade", -0.5), ("annual_inc", -0.3)],
            "stance": "mitigates_risk",
            "stance_confidence": 0.75,
            "stance_evidence": ["I have been at my job for 9 years"],
            "stance_policy_ids": ["policy_3.3"],
            "stance_rationale": "Applicant states long, stable tenure at current employer.",
        },
        "expect_route": "agree",
        "expect_decision": "auto_approve",
    },
    {
        "name": "disagree-risky-but-mitigates",
        "state": {
            "p_default": 0.68,
            "shap_drivers": [("int_rate", 0.8), ("dti", 0.4)],
            "stance": "mitigates_risk",
            "stance_confidence": 0.80,
            "stance_evidence": ["I've paid off two other loans early"],
            "stance_policy_ids": ["policy_3.3"],
            "stance_rationale": "Applicant cites a strong repayment track record despite high tabular risk.",
        },
        "expect_route": "disagree",
        "expect_decision": "human_review",
    },
    {
        "name": "disagree-safe-but-corroborates",
        "state": {
            "p_default": 0.15,
            "shap_drivers": [("grade", -0.4)],
            "stance": "corroborates_risk",
            "stance_confidence": 0.80,
            "stance_evidence": ["using this to avoid a collections action on my old card"],
            "stance_policy_ids": ["policy_5.1"],
            "stance_rationale": "Applicant discloses using proceeds to avoid a collections action, a strong risk flag.",
        },
        "expect_route": "disagree",
        "expect_decision": "human_review",
    },
    {
        "name": "low-confidence",
        "state": {
            "p_default": 0.55,
            "shap_drivers": [("int_rate", 0.5)],
            "stance": "corroborates_risk",
            "stance_confidence": 0.30,
            "stance_evidence": ["things have been tough"],
            "stance_policy_ids": ["policy_2.3"],
            "stance_rationale": "Vague statement of financial stress with no specific triggering event.",
        },
        "expect_route": "low_conf",
        # low_conf now takes the tabular decision (Fix 2, DECISIONS.md Build 5/6):
        # p_default=0.55 >= HIGH_RISK -> auto_decline, not human_review. Only a
        # genuine "disagree" route still goes to human_review.
        "expect_decision": "auto_decline",
    },
]


def run_case(case: dict) -> None:
    state = dict(case["state"])

    state.update(reconciler(state))
    assert state["route"] == case["expect_route"], (
        f"[{case['name']}] route: expected {case['expect_route']!r}, got {state['route']!r}"
    )

    next_node = _route(state)
    state.update(auto_decision(state) if next_node == "auto_decision" else human_review(state))
    assert state["decision"] == case["expect_decision"], (
        f"[{case['name']}] decision: expected {case['expect_decision']!r}, got {state['decision']!r}"
    )

    state.update(explanation(state))

    print(f"PASS  {case['name']:32s} route={state['route']:12s} decision={state['decision']}")


if __name__ == "__main__":
    for case in CASES:
        run_case(case)
    print(f"\nAll {len(CASES)} routing cases passed.")
