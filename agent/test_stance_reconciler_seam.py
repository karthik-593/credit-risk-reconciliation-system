"""
Integration test: the seam between text_stance's own JSON-parsing path and
the live reconciler(). Runs the real text_stance() node (real _retrieve_policy
BM25 lookup, real _call_llm_json parsing) against a stub LLMClient so no
model file or live API key is needed, then feeds its output straight into
reconciler() exactly as the graph would. p_default is fabricated per case
since this test targets the stance -> reconciler seam, not tabular scoring.
"""
import json

import reconciler_agent as ra


class FixedStanceClient:
    """Stub LLMClient returning a fixed, valid stance JSON string."""

    def __init__(self, stance: str, confidence: float):
        self._payload = json.dumps({
            "stance": stance,
            "evidence_spans": ["stub evidence span"],
            "cited_policy_ids": ["policy_1.1"],
            "confidence": confidence,
            "rationale": "stub rationale for integration test",
        })

    def complete(self, system: str, user: str) -> str:
        return self._payload


class BrokenClient:
    """Stub LLMClient returning unparseable text, to exercise the neutral fallback."""

    def complete(self, system: str, user: str) -> str:
        return "this is not json"


class RaisingClient:
    """Stub LLMClient whose complete() raises, simulating a network/rate-limit/
    quota/timeout failure after the adapter's own retries are exhausted --
    exercises the api_error path (distinct from a parse_error: the call never
    produced a response at all)."""

    def complete(self, system: str, user: str) -> str:
        raise RuntimeError("simulated: Gemini call failed after 5 attempts: 429 quota exceeded")


APPLICATION = {"application": {"desc_clean": "I want to consolidate my credit card debt."}}

CASES = [
    {
        "name": "mitigates+risky-tabular -> disagree",
        "client": FixedStanceClient("mitigates_risk", 0.8),
        "p_default": 0.85,
        "expect_stance": "mitigates_risk",
        "expect_route": "disagree",
        "expect_stance_source": "parsed",
    },
    {
        "name": "corroborates+safe-tabular -> disagree",
        "client": FixedStanceClient("corroborates_risk", 0.8),
        # 0.20 was "safe" under the old hardcoded HIGH_RISK=0.50; it is NOT
        # safe under the VAL-selected threshold (~0.17, DECISIONS.md Build
        # 5/6). 0.05 stays genuinely below the threshold either way.
        "p_default": 0.05,
        "expect_stance": "corroborates_risk",
        "expect_route": "disagree",
        "expect_stance_source": "parsed",
    },
    {
        "name": "corroborates+risky-tabular -> agree",
        "client": FixedStanceClient("corroborates_risk", 0.8),
        "p_default": 0.85,
        "expect_stance": "corroborates_risk",
        "expect_route": "agree",
        "expect_stance_source": "parsed",
    },
    {
        "name": "broken JSON -> neutral fallback -> low_conf",
        "client": BrokenClient(),
        "p_default": 0.50,
        "expect_stance": "neutral",
        "expect_route": "low_conf",
        "expect_stance_source": "parse_error",
    },
    {
        "name": "api call fails -> neutral fallback -> low_conf",
        "client": RaisingClient(),
        "p_default": 0.50,
        "expect_stance": "neutral",
        "expect_route": "low_conf",
        "expect_stance_source": "api_error",
    },
]


def run_case(case: dict) -> None:
    ra.configure_llm_client(case["client"])

    # Real text_stance() node: real _retrieve_policy() BM25 call, real
    # _call_llm_json() parsing path, deliberately blind to p_default.
    state = dict(APPLICATION)
    stance_out = ra.text_stance(state)
    assert stance_out["stance"] == case["expect_stance"], (
        f"[{case['name']}] stance: expected {case['expect_stance']!r}, got {stance_out['stance']!r}"
    )
    assert stance_out["stance_source"] == case["expect_stance_source"], (
        f"[{case['name']}] stance_source: expected {case['expect_stance_source']!r}, "
        f"got {stance_out['stance_source']!r}"
    )

    # Fabricate the tabular channel to isolate the stance -> reconciler seam.
    full_state = {**state, **stance_out, "p_default": case["p_default"]}
    route_out = ra.reconciler(full_state)
    assert route_out["route"] == case["expect_route"], (
        f"[{case['name']}] route: expected {case['expect_route']!r}, got {route_out['route']!r}"
    )

    print(f"PASS  {case['name']:42s} stance={stance_out['stance']:18s} "
          f"source={stance_out['stance_source']:12s} route={route_out['route']}")


if __name__ == "__main__":
    for case in CASES:
        run_case(case)
    ra._llm_client = None  # leave module state clean for any test run after this one
    print(f"\nAll {len(CASES)} stance-to-reconciler seam cases passed.")
