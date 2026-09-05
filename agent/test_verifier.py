"""
Tests for the verifier node -- the second agent that checks whether
text_stance's own quoted evidence and cited policy actually support the
stance it reached. Falsifiable only: never re-judges risk, never proposes a
different stance, only says whether the claim is grounded.

Cases (a)-(d) are exactly what was asked for; the fifth (LLM-failure
fallback) is an addition testing the explicit "do NOT downgrade on an
infra failure" rule, mirroring how the stance node's own failure paths are
tested elsewhere in this suite.
"""
import json

import reconciler_agent as ra

DESC = "I want to consolidate my credit card debt into one lower payment. I have been at my job for 9 years."
REAL_EVIDENCE = "I have been at my job for 9 years"
REAL_POLICY_ID = "policy_3.3"  # confirmed retrieved for DESC (long tenure)


class FixedVerifierClient:
    """Stub LLMClient returning a fixed, valid verifier verdict JSON."""

    def __init__(self, verdict: str, reason: str = "stub reason"):
        self._payload = json.dumps({"verdict": verdict, "reason": reason})

    def complete(self, system: str, user: str) -> str:
        return self._payload


class RaisingClient:
    """Simulates a network/rate-limit/timeout failure during the verifier's
    LLM call, after the adapter's own retries are exhausted."""

    def complete(self, system: str, user: str) -> str:
        raise RuntimeError("simulated: verifier LLM call failed after 5 attempts")


def base_stance_state(evidence, policy_ids, stance="mitigates_risk", confidence=0.8):
    return {
        "application": {"desc_clean": DESC},
        "stance": stance,
        "stance_evidence": evidence,
        "stance_policy_ids": policy_ids,
        "stance_confidence": confidence,
        "stance_rationale": "stub rationale",
    }


def case_a_evidence_not_substring():
    """(a) evidence not a substring -> unsupported -> downgraded -> low_conf."""
    state = base_stance_state(
        evidence=["this quote does not appear anywhere in the applicant's statement"],
        policy_ids=[REAL_POLICY_ID],
    )
    verifier_out = ra.verifier(state)
    assert verifier_out["verifier_verdict"] == "unsupported", verifier_out
    assert verifier_out["verifier_source"] == "mechanical", verifier_out
    assert verifier_out["stance"] == "neutral", verifier_out
    assert verifier_out["stance_confidence"] == 0.0, verifier_out

    full_state = {**state, **verifier_out, "p_default": 0.5}
    route_out = ra.reconciler(full_state)
    assert route_out["route"] == "low_conf", route_out
    print("PASS  (a) evidence not substring -> unsupported (mechanical) -> downgraded -> low_conf")


def case_b_policy_not_retrieved():
    """(b) cited policy id not in the retrieved set -> unsupported."""
    state = base_stance_state(
        evidence=[REAL_EVIDENCE],
        policy_ids=["policy_9.9"],  # never exists / never retrieved
    )
    verifier_out = ra.verifier(state)
    assert verifier_out["verifier_verdict"] == "unsupported", verifier_out
    assert verifier_out["verifier_source"] == "mechanical", verifier_out
    assert verifier_out["stance"] == "neutral", verifier_out
    assert verifier_out["stance_confidence"] == 0.0, verifier_out

    full_state = {**state, **verifier_out, "p_default": 0.5}
    route_out = ra.reconciler(full_state)
    assert route_out["route"] == "low_conf", route_out
    print("PASS  (b) uncited/unretrieved policy id -> unsupported (mechanical) -> downgraded -> low_conf")


def case_c_neutral_skipped():
    """(c) neutral stance -> skipped, untouched."""
    state = base_stance_state(evidence=[], policy_ids=[], stance="neutral", confidence=0.5)
    verifier_out = ra.verifier(state)
    assert verifier_out["verifier_verdict"] == "skipped_neutral", verifier_out
    assert verifier_out["verifier_source"] == "skipped", verifier_out
    assert "stance" not in verifier_out, "neutral stance must not be touched/rewritten"
    print("PASS  (c) neutral stance -> skipped_neutral, untouched")


def case_d_genuinely_supported():
    """(d) genuinely supported mitigates -> passes through unchanged -> can still disagree."""
    ra.configure_llm_client(FixedVerifierClient("supported", "Quote matches the applicant's own words and the cited policy."))
    state = base_stance_state(evidence=[REAL_EVIDENCE], policy_ids=[REAL_POLICY_ID])
    verifier_out = ra.verifier(state)
    assert verifier_out["verifier_verdict"] == "supported", verifier_out
    assert verifier_out["verifier_source"] == "llm", verifier_out
    assert "stance" not in verifier_out, "supported stance must pass through unchanged (not rewritten)"

    # risky tabular + mitigates (unchanged) -> disagree, exactly as it would
    # without the verifier in the pipeline at all.
    full_state = {**state, **verifier_out, "p_default": 0.85}
    route_out = ra.reconciler(full_state)
    assert route_out["route"] == "disagree", route_out
    print("PASS  (d) genuinely supported mitigates -> passes through unchanged -> still disagrees")


def case_e_llm_failure_does_not_downgrade():
    """(bonus) verifier LLM call fails -> 'unclear', NOT downgraded -- an
    infra failure is not evidence the stance is wrong."""
    ra.configure_llm_client(RaisingClient())
    state = base_stance_state(evidence=[REAL_EVIDENCE], policy_ids=[REAL_POLICY_ID])
    verifier_out = ra.verifier(state)
    assert verifier_out["verifier_verdict"] == "unclear", verifier_out
    assert verifier_out["verifier_source"] == "llm", verifier_out
    assert "stance" not in verifier_out, "an infra failure must not downgrade the stance"

    full_state = {**state, **verifier_out, "p_default": 0.85}
    route_out = ra.reconciler(full_state)
    assert route_out["route"] == "disagree", route_out  # unchanged from the original stance
    print("PASS  (e) verifier LLM call fails -> unclear, NOT downgraded -> original route unaffected")


# ---------------------------------------------------------------------------
# (f)-(h) -- regression cases for the grounding-check fix.
#
# The mechanical layer used to test `span not in desc`, an exact substring
# match. That downgraded genuinely grounded stances to neutral whenever the
# model wrapped its quote in quote characters or re-cased a word. These three
# cases pin the corrected behaviour AND its boundary: (f) and (g) must now
# pass the mechanical layer, (h) must still fail it. Without (h), a future
# "improvement" to the normalizer could start accepting paraphrase and no
# test would notice.
# ---------------------------------------------------------------------------
def case_f_quote_wrapped_evidence_passes_mechanical():
    """(f) a real quote wrapped in literal quote chars -> NOT mechanically
    unsupported; it reaches the LLM verifier like any grounded span."""
    ra.configure_llm_client(FixedVerifierClient("supported", "Quote matches the applicant's own words."))
    state = base_stance_state(evidence=[f'"{REAL_EVIDENCE}"'], policy_ids=[REAL_POLICY_ID])
    verifier_out = ra.verifier(state)
    assert verifier_out["verifier_source"] == "llm", (
        f"quote-wrapped evidence must clear the mechanical layer, got {verifier_out}")
    assert verifier_out["verifier_verdict"] == "supported", verifier_out
    assert "stance" not in verifier_out, "a grounded stance must not be downgraded"
    print("PASS  (f) quote-wrapped real evidence -> clears mechanical layer, reaches LLM verifier")


def case_g_casing_variant_passes_mechanical():
    """(g) a casing variant of a present span -> same."""
    ra.configure_llm_client(FixedVerifierClient("supported", "Quote matches the applicant's own words."))
    state = base_stance_state(evidence=[REAL_EVIDENCE.upper()], policy_ids=[REAL_POLICY_ID])
    verifier_out = ra.verifier(state)
    assert verifier_out["verifier_source"] == "llm", (
        f"a re-cased real quote must clear the mechanical layer, got {verifier_out}")
    assert verifier_out["verifier_verdict"] == "supported", verifier_out
    assert "stance" not in verifier_out, "a grounded stance must not be downgraded"
    print("PASS  (g) re-cased real evidence -> clears mechanical layer, reaches LLM verifier")


def case_h_paraphrase_still_unsupported():
    """(h) THE BOUNDARY. A near-miss paraphrase sharing most of its words with
    the statement, but not a substring even after quote/case normalization ->
    still mechanically unsupported. The fix must accept quoting and casing
    hygiene ONLY, never paraphrase."""
    # DESC says "I have been at my job for 9 years"; this says "9 years at my
    # job" -- same words, reordered. Word overlap is high; it is not a quote.
    paraphrase = "I have worked at my job for 9 years"
    assert paraphrase not in DESC, "test setup: the paraphrase must not be a literal substring"
    state = base_stance_state(evidence=[paraphrase], policy_ids=[REAL_POLICY_ID])
    verifier_out = ra.verifier(state)
    assert verifier_out["verifier_verdict"] == "unsupported", verifier_out
    assert verifier_out["verifier_source"] == "mechanical", verifier_out
    assert verifier_out["stance"] == "neutral", verifier_out
    assert verifier_out["stance_confidence"] == 0.0, verifier_out
    print("PASS  (h) near-miss paraphrase -> STILL unsupported (mechanical) -- boundary holds")


if __name__ == "__main__":
    case_a_evidence_not_substring()
    case_b_policy_not_retrieved()
    case_c_neutral_skipped()
    case_d_genuinely_supported()
    case_e_llm_failure_does_not_downgrade()
    case_f_quote_wrapped_evidence_passes_mechanical()
    case_g_casing_variant_passes_mechanical()
    case_h_paraphrase_still_unsupported()
    ra._llm_client = None  # leave module state clean for any test run after this one
    print("\nAll 8 verifier cases passed.")
