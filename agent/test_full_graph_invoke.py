"""
Integration test: a single real application flowing through the fully
compiled graph (app.invoke), exercising the real _score_tabular() against
the trained Model A file and the real _retrieve_policy() BM25 lookup, with
only the LLM call stubbed out (no live API key needed). Confirms state
actually threads node-to-node end to end, not just that each node works
in isolation -- now through five real nodes (tabular_score, text_stance,
verifier, reconciler, auto_decision/human_review), not four.

The stub client answers BOTH prompts it's asked (text_stance's and the
verifier's, distinguished by a marker in the system prompt) since one
LLMClient now serves two different call sites in the graph. Its evidence
span is a real substring of desc_clean and its cited policy is one that's
actually retrieved for it, so the verifier's mechanical layer passes and
its LLM layer is exercised too (the earlier version of this fixture had
evidence that didn't match desc_clean at all -- harmless before the
verifier existed, but the verifier would now correctly catch and downgrade
it, which is a fine thing to demonstrate but wasn't this test's point).
"""
import json

import reconciler_agent as ra


class TwoPurposeClient:
    """Serves both call sites the graph now makes: text_stance's stance
    JSON, and the verifier's grounding-verdict JSON. Distinguishes them by
    a marker unique to VERIFIER_SYSTEM_PROMPT."""

    def complete(self, system: str, user: str) -> str:
        if "grounding auditor" in system.lower():
            return json.dumps({
                "verdict": "supported",
                "reason": "The quoted tenure statement is the applicant's own words and matches "
                          "the cited policy on long employer tenure.",
            })
        return json.dumps({
            "stance": "mitigates_risk",
            "evidence_spans": ["I have been at my job for 9 years"],
            "cited_policy_ids": ["policy_3.3"],
            "confidence": 0.75,
            "rationale": "Applicant states long, stable employment tenure.",
        })


APPLICATION = {
    "application": {
        "tabular_features": {
            "loan_amnt": 12000, "term": "36 months", "int_rate": 14.5,
            "grade": "C", "sub_grade": "C3", "annual_inc": 55000, "dti": 22.1,
            "emp_length": "5 years", "home_ownership": "RENT",
            "verification_status": "Verified", "fico_range_low": 690,
            "fico_range_high": 694, "purpose": "debt_consolidation",
        },
        "desc_clean": "I want to consolidate my credit card debt into one lower payment. "
                      "I have been at my job for 9 years.",
    }
}


if __name__ == "__main__":
    ra.configure_llm_client(TwoPurposeClient())

    app = ra.build_graph()
    final_state = app.invoke(APPLICATION)

    assert isinstance(final_state.get("p_default"), float), \
        f"p_default missing or not a float: {final_state.get('p_default')!r}"
    assert final_state.get("stance") in ("corroborates_risk", "mitigates_risk", "neutral"), \
        f"stance missing or invalid: {final_state.get('stance')!r}"
    assert final_state.get("verifier_verdict") in \
        ("supported", "unsupported", "unclear", "skipped_neutral"), \
        f"verifier_verdict missing or invalid: {final_state.get('verifier_verdict')!r}"
    assert final_state.get("decision") in ("auto_approve", "auto_decline", "human_review"), \
        f"decision missing or invalid: {final_state.get('decision')!r}"
    assert isinstance(final_state.get("memo"), str) and final_state["memo"].strip(), \
        "memo missing or empty"

    # With a genuinely grounded stance (mechanical checks pass, LLM says
    # supported), the verifier must NOT have downgraded it.
    assert final_state["stance"] == "mitigates_risk", \
        f"expected the grounded stance to pass through unchanged, got {final_state['stance']!r}"
    assert final_state["verifier_verdict"] == "supported", \
        f"expected a supported verdict, got {final_state['verifier_verdict']!r}"

    ra._llm_client = None  # leave module state clean for any test run after this one

    print("PASS  full graph invoke")
    print(f"  p_default = {final_state['p_default']:.4f}")
    print(f"  stance    = {final_state['stance']} (conf {final_state['stance_confidence']:.2f})")
    print(f"  verifier  = {final_state['verifier_verdict']} (source: {final_state['verifier_source']})")
    print(f"  route     = {final_state['route']}")
    print(f"  decision  = {final_state['decision']}")
    print(f"  memo      =\n{final_state['memo']}")
