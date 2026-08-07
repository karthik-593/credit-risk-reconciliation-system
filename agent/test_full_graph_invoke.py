"""
Integration test: a single real application flowing through the fully
compiled graph (app.invoke), exercising the real _score_tabular() against
the trained Model A file and the real _retrieve_policy() BM25 lookup, with
only the LLM call stubbed out (no live API key needed). Confirms state
actually threads node-to-node end to end, not just that each node works
in isolation.
"""
import json

import reconciler_agent as ra


class FixedStanceClient:
    def complete(self, system: str, user: str) -> str:
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
        "desc_clean": "I want to consolidate my credit card debt into one lower payment.",
    }
}


if __name__ == "__main__":
    ra.configure_llm_client(FixedStanceClient())

    app = ra.build_graph()
    final_state = app.invoke(APPLICATION)

    assert isinstance(final_state.get("p_default"), float), \
        f"p_default missing or not a float: {final_state.get('p_default')!r}"
    assert final_state.get("stance") in ("corroborates_risk", "mitigates_risk", "neutral"), \
        f"stance missing or invalid: {final_state.get('stance')!r}"
    assert final_state.get("decision") in ("auto_approve", "auto_decline", "human_review"), \
        f"decision missing or invalid: {final_state.get('decision')!r}"
    assert isinstance(final_state.get("memo"), str) and final_state["memo"].strip(), \
        "memo missing or empty"

    ra._llm_client = None  # leave module state clean for any test run after this one

    print("PASS  full graph invoke")
    print(f"  p_default = {final_state['p_default']:.4f}")
    print(f"  stance    = {final_state['stance']} (conf {final_state['stance_confidence']:.2f})")
    print(f"  route     = {final_state['route']}")
    print(f"  decision  = {final_state['decision']}")
    print(f"  memo      =\n{final_state['memo']}")
