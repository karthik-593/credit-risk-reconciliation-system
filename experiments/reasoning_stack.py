"""
Reasoning A/B/C experiment: does structured / multi-step reasoning in the
note -> stance step lower the verifier's UNSUPPORTED rate?

Companion to experiments/retrieval_stack.py, and deliberately its mirror
image. That script held REASONING fixed (the shipped stance prompt) and
varied RETRIEVAL, and found a null: three retrievers, overlapping Wilson
CIs, reported as a tie. This script inverts the axis -- RETRIEVAL is pinned
to the shipped in-process BM25 over reconciler_agent.POLICY_CORPUS (19
chunks, k=4) for every arm, and only the REASONING differs:

  arm0_baseline    the shipped node, called as-is: ra.text_stance(state).
                   One LLM call, one prompt, straight to a stance.
  arm1_structured  ONE call, but the prompt forces an ordered chain:
                   extract facts from the note -> pick the retrieved rule ->
                   test that rule's conditions against ONLY the extracted
                   facts -> if the note doesn't establish them, return
                   neutral instead of a risk judgment.
  arm2_two_call    TWO calls. Call 1 sees the note and NO policy at all and
                   only extracts facts; call 2 sees those facts plus the
                   retrieved rules and applies them. The separation is the
                   point: call 1 cannot bend its fact extraction toward a
                   rule it hasn't been shown.

Every arm returns the exact contract ra.text_stance returns, and every arm
is scored by the SAME verifier -- ra.verifier(state), unmodified. The
verifier is the measuring instrument here; changing it would be a different
experiment, so it is held identical across all three arms and this script
never touches it.

WHY THE HEADLINE METRIC ALONE IS NOT ENOUGH. "unsupported / non-neutral" is
trivially gameable by abstaining: an arm that answers "neutral" to
everything has an undefined-to-zero unsupported rate and is useless. The
prompts in arms 1 and 2 explicitly push toward neutral when a rule's
conditions aren't established, so they are exactly the kind of change that
can win the headline by going quiet. So the full funnel is reported per arm
-- neutral rate, non-neutral n, confident non-neutral n, confident AND
supported n -- and the verdict below reads the headline THROUGH the neutral
rate. A lower unsupported rate bought entirely with a higher neutral rate is
reported as abstention, not as better reasoning.

Wilson 95% CIs throughout, same discipline as scripts/eval_agent.py and
retrieval_stack.py (DECISIONS.md): overlapping CIs across arms are a TIE,
reported as a tie, never rounded into a winner.

Does NOT modify agent/reconciler_agent.py, the verifier, or any config. It
imports the real stance node, the real verifier, the real prompts, the real
JSON-parsing/fallback contract and the real BM25 retrieval from
reconciler_agent -- the only NEW code here is two prompts (arms 1 and 2) and
the glue that maps their JSON onto the same output contract.

Cost metering: llm_client's view of the `requests` module is wrapped by a
proxy (_MeteredRequests) that reads the REAL prompt/completion token counts
out of Ollama's OpenAI-compatible `usage` field and the real wall-clock time
of each POST. Nothing is estimated from character counts, and no shipped
file is edited -- the same wrap-the-adapter trick agent/llm_client.py's own
__main__ block uses to capture raw text.
"""
import argparse
import json
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
import reconciler_agent as ra  # noqa: E402
import llm_client as lc  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from eval_agent import (  # noqa: E402
    load_test_population, base_rate_sample, wilson_ci, UNDERPOWERED_N,
)

K = 4                      # chunks retrieved per note -- the shipped default
RETRIEVAL_CONFIG_PATH = ROOT / "config" / "retrieval.json"
CHECKPOINT_EVERY = 50

ARM_NAMES = ["arm0_baseline", "arm1_structured", "arm2_two_call"]


# ---------------------------------------------------------------------------
# Cost metering -- real tokens, real latency, no shipped file touched.
#
# llm_client's adapters call requests.post() through their module-global
# `requests`. Rebinding lc.requests to this proxy means only llm_client's
# view of the module is metered; the real `requests` module is left alone,
# and every other attribute (requests.exceptions.ConnectionError etc.) is
# delegated straight through by __getattr__.
#
# Retries are counted, not hidden: each POST attempt adds its own latency
# and tokens, because a call that had to be retried genuinely cost that.
# ---------------------------------------------------------------------------
class _Meter:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.calls = 0
        self.seconds = 0.0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.usage_missing = 0

    def snapshot(self) -> dict:
        return {
            "calls": self.calls,
            "seconds": self.seconds,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "usage_missing": self.usage_missing,
        }


METER = _Meter()


class _MeteredRequests:
    """Stands in for llm_client's `requests` module: meters .post(), passes
    everything else through untouched."""

    def __init__(self, real_module):
        self._real = real_module

    def __getattr__(self, name):
        return getattr(self._real, name)

    def post(self, *args, **kwargs):
        t0 = time.perf_counter()
        resp = self._real.post(*args, **kwargs)
        METER.seconds += time.perf_counter() - t0
        METER.calls += 1
        try:
            usage = resp.json().get("usage") or {}
        except Exception:
            usage = {}
        if usage:
            METER.prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
            METER.completion_tokens += int(usage.get("completion_tokens", 0) or 0)
        else:
            # A provider that doesn't report usage (or an error body) --
            # counted so the report can say tokens are incomplete rather
            # than quietly reporting a too-low number as if it were exact.
            METER.usage_missing += 1
        return resp


def install_meter() -> None:
    lc.requests = _MeteredRequests(lc.requests)


# ---------------------------------------------------------------------------
# Shared output contract.
#
# _to_contract() is ra.text_stance()'s own mapping of raw LLM JSON onto the
# state keys (agent/reconciler_agent.py:458-477), lifted verbatim so arms 1
# and 2 produce byte-identical shapes to arm 0's. Arm 0 does not use it --
# it calls the shipped node, which does this itself.
# ---------------------------------------------------------------------------
def _to_contract(out: dict) -> dict:
    return {
        "stance": out.get("stance", "neutral"),
        "stance_evidence": out.get("evidence_spans", []),
        "stance_policy_ids": out.get("cited_policy_ids", []),
        "stance_confidence": float(out.get("confidence", 0.0)),
        "stance_rationale": out.get("rationale", ""),
        "stance_source": out.get("stance_source", "parsed"),
        "stance_error_detail": out.get("stance_error_detail", ""),
    }


def _policy_block(policy: list[dict]) -> str:
    """Same rendering ra.text_stance uses for its policy snippets."""
    return "\n".join(f'[{p["id"]}] {p["text"]}' for p in policy) or "(none)"


# ===========================================================================
# ARM 0 -- baseline. The shipped node, unchanged, called directly.
# Its unsupported rate is recomputed FRESH on this sample; the ~41% quoted in
# the README came from a different sample and is not carried over here.
# ===========================================================================
def arm0_baseline(desc: str, policy: list[dict]) -> dict:
    # `policy` is ignored on purpose: text_stance runs its own
    # ra._retrieve_policy(desc) internally. Same function, same deterministic
    # BM25, same input -> the same k=4 chunks the caller already computed.
    return ra.text_stance({"application": {"desc_clean": desc}})


# ===========================================================================
# ARM 1 -- structured, ONE call. Same retrieval, same output contract, new
# prompt that forces facts -> rule -> conditions -> stance in that order,
# with an explicit "conditions not established -> neutral" gate.
# extracted_facts/applicable_rule/conditions_met ride along in the JSON;
# ra._call_llm_json only validates `stance`, so they pass through harmlessly
# and are recorded for inspection.
# ===========================================================================
ARM1_SYSTEM_PROMPT = """You are an underwriting narrative analyst.

You are shown ONLY a loan applicant's own written statement and a set of
retrieved underwriting-policy snippets. You are NOT shown the model's risk
score, and you must NOT try to guess or output a risk probability.

Your single job: judge whether the applicant's statement CORROBORATES risk,
MITIGATES risk, or is NEUTRAL -- but you must reach that judgment by working
through the following steps IN ORDER, and you must not skip ahead.

STEP 1 -- EXTRACT FACTS. List the facts the statement literally asserts.
  A fact belongs in this list only if the applicant actually wrote it. Do not
  add anything implied, assumed, typical, or inferred from the loan purpose.
  If the statement asserts no financial facts at all, this list is empty.

STEP 2 -- SELECT A RULE. From the retrieved policy snippets ONLY, pick the
  single snippet whose conditions are closest to the STEP 1 facts. If no
  retrieved snippet is about anything the STEP 1 facts describe, select none.

STEP 3 -- TEST THE RULE'S CONDITIONS. State what conditions the selected
  snippet requires, and check them against the STEP 1 facts and NOTHING ELSE.
  A condition is met only if a STEP 1 fact establishes it. A condition that
  is merely plausible, likely, or usually true is NOT met.

STEP 4 -- DECIDE. If you selected no rule, or the STEP 3 conditions are not
  established by the STEP 1 facts, then stance MUST be "neutral" -- return
  neutral rather than a risk judgment you cannot ground. Only when the
  conditions ARE established may you return "corroborates_risk" or
  "mitigates_risk", and then the stance must be the direction that snippet
  assigns.

Rules:
- evidence_spans MUST be exact quotes copied character-for-character from the
  statement.
- cited_policy_ids MUST come only from the retrieved snippets provided.
- confidence is how CLEARLY the text supports your stance (0-1), NOT a
  probability of default.
- If the statement is empty, boilerplate, or uninformative: stance="neutral",
  confidence low.

Respond with ONLY this JSON, no prose:
{
  "extracted_facts": ["fact the statement literally asserts", ...],
  "applicable_rule": "policy_id or none",
  "rule_conditions": "what that policy requires, in one sentence",
  "conditions_met": true,
  "stance": "corroborates_risk | mitigates_risk | neutral",
  "evidence_spans": ["exact quote", ...],
  "cited_policy_ids": ["policy_id", ...],
  "confidence": 0.0,
  "rationale": "one sentence grounded in the cited policy"
}"""

# Same statement + snippets framing as the shipped prompt, written out here
# rather than importing ra.STANCE_USER_TEMPLATE, so this arm's prompt pair is
# entirely its own and editing one arm can never perturb another.
ARM1_USER_TEMPLATE = 'Applicant statement:\n"""{desc_clean}"""\n\nRetrieved policy snippets:\n{policy_block}'


def arm1_structured(desc: str, policy: list[dict]) -> dict:
    out = ra._call_llm_json(
        ARM1_SYSTEM_PROMPT,
        ARM1_USER_TEMPLATE.format(
            desc_clean=desc or "(empty)", policy_block=_policy_block(policy),
        ),
    )
    contract = _to_contract(out)
    contract["chain"] = {
        "extracted_facts": out.get("extracted_facts"),
        "applicable_rule": out.get("applicable_rule"),
        "conditions_met": out.get("conditions_met"),
    }
    return contract


# ===========================================================================
# ARM 2 -- TWO calls.
#
# Call 1 sees the statement and NO policy whatsoever, and only extracts what
# the statement says. That isolation is the whole hypothesis: a fact
# extractor that has never seen the rule cannot shade its extraction toward
# satisfying that rule. Its output is not stance JSON, so it goes through
# ra._llm_client.complete() + ra._strip_code_fence + json.loads directly --
# ra._call_llm_json would reject it for having no `stance` key.
#
# Call 2 sees the extracted facts plus the retrieved rules and applies them,
# through the normal ra._call_llm_json path.
#
# Failure handling mirrors ra._call_llm_json exactly: an infra failure on
# call 1 becomes ra._neutral_fallback("api_error", ...), an unparseable
# response becomes ra._neutral_fallback("parse_error", ...) -- so a broken
# call 1 shows up in stance_source as a failure, never as a genuine neutral.
# ===========================================================================
ARM2_EXTRACT_SYSTEM_PROMPT = """You are a fact extractor for an underwriting file.

You are shown ONLY a loan applicant's own written statement. You are shown no
policy, no rules, and no risk score. You must NOT judge risk, must NOT decide
whether anything is good or bad for the application, and must NOT recommend
anything.

Your single job: report what the statement literally asserts.

Rules:
- A fact belongs in your output only if the applicant actually wrote it. Do
  not add anything implied, assumed, typical, or inferred.
- For every fact, quote the exact span of the statement it came from, copied
  character-for-character.
- If the statement is empty, boilerplate, or asserts no financial facts,
  return empty lists.

Respond with ONLY this JSON, no prose:
{
  "facts": ["fact the statement literally asserts", ...],
  "verbatim_spans": ["exact quote from the statement", ...]
}"""

ARM2_EXTRACT_USER_TEMPLATE = 'Applicant statement:\n"""{desc_clean}"""'

ARM2_APPLY_SYSTEM_PROMPT = """You are an underwriting narrative analyst.

You are shown a list of facts already extracted from a loan applicant's
written statement, the exact quotes those facts came from, and a set of
retrieved underwriting-policy snippets. You are NOT shown the model's risk
score, and you must NOT try to guess or output a risk probability.

Your single job: apply the retrieved policy to the extracted facts, and
nothing else.

- Treat the extracted facts as the COMPLETE record of what the applicant
  said. Anything not in that list was not stated, and you must not reason
  from it -- not from what is plausible, likely, or usually true of such
  applicants.
- Pick the single retrieved snippet whose conditions the extracted facts come
  closest to. If no retrieved snippet is about anything the facts describe,
  select none.
- Check that snippet's conditions against the extracted facts. A condition is
  met only if an extracted fact establishes it.
- If you selected no snippet, or its conditions are not established by the
  extracted facts, then stance MUST be "neutral" -- return neutral rather
  than a risk judgment you cannot ground. Only when the conditions ARE
  established may you return "corroborates_risk" or "mitigates_risk", and
  then the stance must be the direction that snippet assigns.

Rules:
- evidence_spans MUST be chosen from the provided verbatim quotes, copied
  character-for-character.
- cited_policy_ids MUST come only from the retrieved snippets provided.
- confidence is how CLEARLY the facts support your stance (0-1), NOT a
  probability of default.
- If the fact list is empty: stance="neutral", confidence low.

Respond with ONLY this JSON, no prose:
{
  "applicable_rule": "policy_id or none",
  "rule_conditions": "what that policy requires, in one sentence",
  "conditions_met": true,
  "stance": "corroborates_risk | mitigates_risk | neutral",
  "evidence_spans": ["exact quote", ...],
  "cited_policy_ids": ["policy_id", ...],
  "confidence": 0.0,
  "rationale": "one sentence grounded in the cited policy"
}"""

ARM2_APPLY_USER_TEMPLATE = """Facts extracted from the applicant's statement:
{facts_block}

Exact quotes those facts came from:
{spans_block}

Retrieved policy snippets:
{policy_block}"""


def _extract_facts(desc: str) -> tuple[dict, dict]:
    """Call 1. Returns (parsed, None) on success, (None, fallback) on
    failure -- the fallback already in ra's neutral-fallback shape."""
    if ra._llm_client is None:
        return None, ra._neutral_fallback("empty", "no LLM client configured")
    try:
        raw = ra._llm_client.complete(
            ARM2_EXTRACT_SYSTEM_PROMPT,
            ARM2_EXTRACT_USER_TEMPLATE.format(desc_clean=desc or "(empty)"),
        )
    except Exception as exc:
        return None, ra._neutral_fallback("api_error", f"extract call: {type(exc).__name__}: {exc}")
    if not raw or not raw.strip():
        return None, ra._neutral_fallback("empty", "extract call returned an empty response")
    try:
        parsed = json.loads(ra._strip_code_fence(raw))
        if not isinstance(parsed, dict):
            raise ValueError(f"extract call returned {type(parsed).__name__}, not an object")
    except Exception as exc:
        return None, ra._neutral_fallback("parse_error", f"extract call: {type(exc).__name__}: {exc}")
    return parsed, None


def arm2_two_call(desc: str, policy: list[dict]) -> dict:
    facts_json, fallback = _extract_facts(desc)
    if fallback is not None:
        contract = _to_contract(fallback)
        contract["chain"] = {"extract_failed": True}
        return contract

    facts = facts_json.get("facts") or []
    spans = facts_json.get("verbatim_spans") or []
    facts_block = "\n".join(f"- {f}" for f in facts) or "(no facts asserted)"
    spans_block = "\n".join(f'- "{s}"' for s in spans) or "(none quoted)"

    out = ra._call_llm_json(
        ARM2_APPLY_SYSTEM_PROMPT,
        ARM2_APPLY_USER_TEMPLATE.format(
            facts_block=facts_block, spans_block=spans_block,
            policy_block=_policy_block(policy),
        ),
    )
    contract = _to_contract(out)
    contract["chain"] = {
        "extracted_facts": facts,
        "verbatim_spans": spans,
        "applicable_rule": out.get("applicable_rule"),
        "conditions_met": out.get("conditions_met"),
    }
    return contract


ARMS = {
    "arm0_baseline": arm0_baseline,
    "arm1_structured": arm1_structured,
    "arm2_two_call": arm2_two_call,
}


# ---------------------------------------------------------------------------
# Scoring -- the SAME verifier for every arm, called as the shipped node.
#
# ra.verifier() re-runs ra._retrieve_policy(desc) itself for its policy-id
# membership check; retrieval is deterministic BM25 over the same corpus, so
# it sees exactly the id set the arm was shown. Its return dict may also
# carry the safe-direction downgrade (stance -> "neutral", confidence -> 0.0)
# that reconciler() would act on; nothing is routed off it here, and the
# ARM's OWN pre-verifier stance is what the funnel below counts, so a
# downgrade can never be mistaken for the arm having abstained.
# ---------------------------------------------------------------------------
def score(desc: str, stance_out: dict) -> dict:
    state = {"application": {"desc_clean": desc}, **stance_out}
    return ra.verifier(state)


def _trim_chain(chain) -> dict:
    """Keep the chain's shape for auditing without carrying every extracted
    fact string into a 2000-note results file."""
    if not isinstance(chain, dict):
        return {}
    facts = chain.get("extracted_facts")
    return {
        "n_extracted_facts": len(facts) if isinstance(facts, list) else None,
        "applicable_rule": chain.get("applicable_rule"),
        "conditions_met": chain.get("conditions_met"),
        "extract_failed": chain.get("extract_failed", False),
    }


def parse_args():
    p = argparse.ArgumentParser(
        description="Reasoning A/B/C experiment: baseline vs structured vs two-call, "
                    "retrieval held fixed at the shipped in-process BM25.")
    p.add_argument("--n", type=int, default=2000,
                   help="sample size, base-rate-preserved from the locked TEST split")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=None, help="override the output json path")
    p.add_argument("--report-only", action="store_true",
                   help="skip the LLM loop; re-aggregate and re-print from the existing "
                        "results/checkpoint file (works on an in-progress run)")
    p.add_argument("--track", action="store_true",
                   help="log one MLflow run per arm to the local ./mlruns store (off by default)")
    p.add_argument("--experiment", default="reasoning_ab", help="MLflow experiment name, used with --track")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Aggregation -- the full funnel, not just the headline.
# ---------------------------------------------------------------------------
def unsupported_stats(recs: list) -> dict:
    """Identical to retrieval_stack.unsupported_stats: of the NON-NEUTRAL
    stances an arm produced, what fraction did the verifier mark
    unsupported, with a Wilson 95% CI."""
    non_neutral = [r for r in recs if r["stance"] != "neutral"]
    n = len(non_neutral)
    k = sum(1 for r in non_neutral if r["verifier_verdict"] == "unsupported")
    if n == 0:
        return {"n": 0, "k": 0, "rate": None, "ci_low": None, "ci_high": None, "underpowered": True}
    rate = k / n
    ci_low, ci_high = wilson_ci(k, n)
    return {"n": n, "k": k, "rate": rate, "ci_low": ci_low, "ci_high": ci_high,
            "underpowered": n < UNDERPOWERED_N}


def funnel(recs: list) -> dict:
    n_notes = len(recs)
    non_neutral = [r for r in recs if r["stance"] != "neutral"]
    confident = [r for r in non_neutral if r["stance_confidence"] >= ra.CONF_THRESHOLD]
    verdicts = ["supported", "unsupported", "unclear", "skipped_neutral"]
    sources = ["mechanical", "llm", "skipped"]

    def mean(key: str) -> float:
        return float(np.mean([r[key] for r in recs])) if recs else float("nan")

    return {
        "n_notes": n_notes,
        "n_neutral": n_notes - len(non_neutral),
        "neutral_rate": (n_notes - len(non_neutral)) / n_notes if n_notes else None,
        "n_non_neutral": len(non_neutral),
        "n_confident_non_neutral": len(confident),
        # What the task calls "what actually reaches a disagree route".
        "n_confident_supported": sum(1 for r in confident if r["verifier_verdict"] == "supported"),
        # The shipped graph is slightly wider than that: only "unsupported"
        # triggers the downgrade to neutral/0.0, so an "unclear" verdict
        # passes through and CAN still route a disagree. Both are reported so
        # the strict and the actual survivor counts are visible side by side.
        "n_confident_not_unsupported": sum(1 for r in confident if r["verifier_verdict"] != "unsupported"),
        "verdict_breakdown": {v: sum(1 for r in recs if r["verifier_verdict"] == v) for v in verdicts},
        "verifier_source_breakdown": {s: sum(1 for r in recs if r["verifier_source"] == s) for s in sources},
        # A spike here means the arm is fabricating spans or policy ids --
        # a different failure than an LLM judging a real quote ungrounded.
        "n_mechanical_unsupported": sum(
            1 for r in recs
            if r["verifier_verdict"] == "unsupported" and r["verifier_source"] == "mechanical"),
        "n_llm_unsupported": sum(
            1 for r in recs
            if r["verifier_verdict"] == "unsupported" and r["verifier_source"] == "llm"),
        "stance_source_breakdown": {
            s: sum(1 for r in recs if r["stance_source"] == s)
            for s in sorted({r["stance_source"] for r in recs})
        },
        "cost": {
            "mean_reasoning_calls": mean("reasoning_calls"),
            "mean_reasoning_tokens": mean("reasoning_tokens"),
            "mean_reasoning_s": mean("reasoning_s"),
            "mean_verifier_tokens": mean("verifier_tokens"),
            "mean_verifier_s": mean("verifier_s"),
            "mean_total_tokens": mean("reasoning_tokens") + mean("verifier_tokens"),
            "mean_total_s": mean("reasoning_s") + mean("verifier_s"),
            "n_notes_missing_usage": sum(1 for r in recs if r.get("usage_missing", 0)),
        },
    }


def llm_layer_stats(recs: list) -> dict:
    """Of the non-neutral stances that SURVIVED the mechanical layer and
    actually reached the LLM grounding auditor, what fraction did it call
    unsupported -- with a Wilson CI.

    This separates two failures the headline metric fuses: spans/ids that
    are malformed or fabricated (caught mechanically, for free) versus
    reasoning the auditor reads as ungrounded. An arm can be bad at one and
    good at the other.

    READ WITH CARE -- this conditions on a POST-TREATMENT variable. The arms
    do not send the same stances to the LLM layer: an arm with sloppier span
    formatting has more of its output filtered out mechanically first, so
    what reaches the auditor is a cleaner, self-selected subset. That makes
    this a decomposition to explain the headline, NOT an independent
    head-to-head an arm can be declared the winner of. mech_pass_rate is
    printed alongside so the size of that selection effect is visible.
    """
    non_neutral = [r for r in recs if r["stance"] != "neutral"]
    reached = [r for r in non_neutral if r["verifier_source"] == "llm"]
    n = len(reached)
    k = sum(1 for r in reached if r["verifier_verdict"] == "unsupported")
    out = {
        "n_reached_llm": n,
        "k_llm_unsupported": k,
        "mech_pass_rate": n / len(non_neutral) if non_neutral else None,
    }
    if n == 0:
        return {**out, "rate": None, "ci_low": None, "ci_high": None}
    ci_low, ci_high = wilson_ci(k, n)
    return {**out, "rate": k / n, "ci_low": ci_low, "ci_high": ci_high}


def span_failure_breakdown(cache: dict, desc_by_id: dict) -> dict:
    """WHY each arm's evidence spans failed the verifier's exact-substring
    check. Purely offline -- re-runs the same mechanical test the verifier
    already ran, over the persisted stance cache, and makes no LLM calls.

    This exists because "mechanical unsupported" is not one failure. Three
    of the categories below are the model genuinely over-claiming; one is
    not:

      paraphrase_or_invention   the span isn't in the note in any form. A
                                real grounding failure -- the model wrote a
                                quote the applicant never wrote.
      quoted_the_POLICY         the span is verbatim (or near-verbatim)
                                RETRIEVED POLICY TEXT, cited as if it were
                                the applicant's own words. The exact defect
                                DECISIONS.md records for llama3.1:8b, and
                                the reason the mechanical layer exists.
      quote_wrapping_artifact   the span IS in the note, but the model
                                wrapped it in literal quote characters, so
                                the substring test fails on punctuation the
                                model added. A PROMPT-FORMATTING artifact,
                                not a grounding failure.
      whitespace_artifact       matches the note once whitespace is
                                normalized. Also formatting, not grounding.

    The last two are a CONFOUND, and it is reported rather than corrected:
    the headline unsupported rate is computed from the unmodified verifier
    applied identically to every arm (no workaround, per the experiment's
    terms), so an arm whose prompt happens to induce quote-wrapping is
    charged for it. Fixing that means editing an arm's prompt, which is a
    separate run -- not something to slip into the one being measured. This
    breakdown is what lets a reader see how much of a gap is real.
    """
    cats = {name: {} for name in ARM_NAMES}
    n_bad_spans = {name: 0 for name in ARM_NAMES}
    for (loan_id, arm), entry in cache.items():
        if arm not in cats:
            continue
        stance_out = entry[0]
        if stance_out.get("stance") == "neutral":
            continue
        desc = desc_by_id.get(loan_id)
        if desc is None:
            continue
        policy_texts = [p["text"] for p in (ra._retrieve_policy(desc) if desc else [])]
        for span in (stance_out.get("stance_evidence") or []):
            if not isinstance(span, str) or span in desc:
                continue
            n_bad_spans[arm] += 1
            stripped = span.strip().strip('"').strip("'").strip()
            if stripped and stripped in desc:
                cat = "quote_wrapping_artifact"
            elif _norm_ws(span) and _norm_ws(span) in _norm_ws(desc):
                cat = "whitespace_artifact"
            elif any(_norm_ws(span) in _norm_ws(t) for t in policy_texts):
                cat = "quoted_the_POLICY_as_evidence"
            else:
                cat = "paraphrase_or_invention"
            cats[arm][cat] = cats[arm].get(cat, 0) + 1
    return {name: {"n_bad_spans": n_bad_spans[name], "by_category": cats[name]} for name in ARM_NAMES}


_WS_RE = re.compile(r"\s+")


def _norm_ws(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def cis_overlap(a: dict, b: dict) -> bool:
    return a["ci_low"] <= b["ci_high"] and b["ci_low"] <= a["ci_high"]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_report(summary: dict, sample_n: int, model: str) -> None:
    print("\n" + "=" * 112)
    print(f"REASONING A/B/C SUMMARY  (n={sample_n} real borrower notes, "
          f"retrieval held fixed: shipped in-process BM25, {len(ra.POLICY_CORPUS)} chunks, k={K})")
    print("=" * 112)
    print(f"{'arm':17s} {'unsupported%':>13s} {'k/n':>10s} {'95% CI':>16s} "
          f"{'neutral%':>9s} {'conf+supp':>10s} {'tok/note':>9s} {'sec/note':>9s}")
    for name, s in summary.items():
        u, f = s["unsupported"], s["funnel"]
        rate = f"{100 * u['rate']:.1f}%" if u["rate"] is not None else "n/a"
        kn = f"{u['k']}/{u['n']}"
        ci = f"[{100*u['ci_low']:.1f},{100*u['ci_high']:.1f}]" if u["rate"] is not None else "n/a"
        neutral = f"{100 * f['neutral_rate']:.1f}%" if f["neutral_rate"] is not None else "n/a"
        print(f"{name:17s} {rate:>13s} {kn:>10s} {ci:>16s} {neutral:>9s} "
              f"{f['n_confident_supported']:>10d} {f['cost']['mean_reasoning_tokens']:>9.0f} "
              f"{f['cost']['mean_reasoning_s']:>9.2f}")
    print("\n(tok/note and sec/note are the REASONING calls only -- the part that differs "
          "between arms.\n verifier cost is identical by construction and reported "
          "separately in the funnel below.)")

    print("\n-- Full funnel per arm --")
    for name, s in summary.items():
        f = s["funnel"]
        c = f["cost"]
        print(f"\n  {name}")
        print(f"    notes                        {f['n_notes']}")
        print(f"    neutral                      {f['n_neutral']}  "
              f"({100 * f['neutral_rate']:.1f}%)" if f["neutral_rate"] is not None else "")
        print(f"    non-neutral (denominator)    {f['n_non_neutral']}")
        print(f"    confident non-neutral        {f['n_confident_non_neutral']}  "
              f"(stance_confidence >= {ra.CONF_THRESHOLD})")
        print(f"    confident AND supported      {f['n_confident_supported']}")
        print(f"    confident AND not-unsupported{f['n_confident_not_unsupported']:>4d}  "
              f"(what the shipped graph actually lets route a disagree)")
        print(f"    verdicts                     {f['verdict_breakdown']}")
        print(f"    verifier source              {f['verifier_source_breakdown']}")
        print(f"    unsupported by source        mechanical={f['n_mechanical_unsupported']}  "
              f"llm={f['n_llm_unsupported']}")
        ll = s["llm_layer"]
        if ll["rate"] is not None:
            print(f"    of those reaching the LLM   {ll['k_llm_unsupported']}/{ll['n_reached_llm']} "
                  f"unsupported = {100*ll['rate']:.1f}%  "
                  f"[{100*ll['ci_low']:.1f},{100*ll['ci_high']:.1f}]  "
                  f"(mechanical pass rate {100*ll['mech_pass_rate']:.1f}%)")
        sf = s.get("span_failures")
        if sf is not None:
            print(f"    bad spans, by cause          {sf['n_bad_spans']} total  "
                  f"{sf['by_category'] or '{}'}")
        print(f"    stance_source                {f['stance_source_breakdown']}")
        print(f"    cost/note                    reasoning: {c['mean_reasoning_calls']:.2f} calls, "
              f"{c['mean_reasoning_tokens']:.0f} tok, {c['mean_reasoning_s']:.2f}s   |   "
              f"verifier: {c['mean_verifier_tokens']:.0f} tok, {c['mean_verifier_s']:.2f}s   |   "
              f"total: {c['mean_total_tokens']:.0f} tok, {c['mean_total_s']:.2f}s")
        if c["n_notes_missing_usage"]:
            print(f"    NOTE: {c['n_notes_missing_usage']} notes had >=1 call with no usage "
                  f"field -- token counts for this arm are a LOWER BOUND.")

    # -------------------------------------------------------------------
    # Verdict. Two questions, answered separately and in order:
    #   1. did the unsupported rate move at all (CI-separated)?
    #   2. if it moved down, was it earned or abstained?
    # -------------------------------------------------------------------
    print("\n" + "-" * 112)
    print("VERDICT")
    print("-" * 112)
    base = summary["arm0_baseline"]
    bu, bf = base["unsupported"], base["funnel"]
    if bu["rate"] is None:
        print("  arm0_baseline produced no non-neutral stances -- nothing to compare against.")
        return

    print(f"  Baseline (arm0, the shipped node) unsupported rate on THIS sample: "
          f"{100*bu['rate']:.1f}%  ({bu['k']}/{bu['n']}, 95% CI "
          f"[{100*bu['ci_low']:.1f},{100*bu['ci_high']:.1f}]).")
    if bu["underpowered"]:
        print(f"  [UNDERPOWERED] non-neutral n={bu['n']} < {UNDERPOWERED_N} -- not interpreted.")

    for name in ARM_NAMES[1:]:
        s = summary[name]
        u, f = s["unsupported"], s["funnel"]
        print(f"\n  {name} vs arm0_baseline:")
        if u["rate"] is None:
            print(f"    {name} produced NO non-neutral stances at all "
                  f"({100*f['neutral_rate']:.1f}% neutral). Its unsupported rate is undefined, "
                  f"not zero -- this arm answered 'neutral' to everything and made no "
                  f"groundable claim to verify. That is total abstention, not better reasoning.")
            continue
        if u["underpowered"] or bu["underpowered"]:
            print(f"    [UNDERPOWERED] non-neutral n={u['n']} (baseline {bu['n']}), "
                  f"< {UNDERPOWERED_N} -- not interpreted.")
            continue

        delta = u["rate"] - bu["rate"]
        overlap = cis_overlap(u, bu)
        if overlap:
            print(f"    TIE. {100*u['rate']:.1f}% [{100*u['ci_low']:.1f},{100*u['ci_high']:.1f}] "
                  f"vs baseline {100*bu['rate']:.1f}% [{100*bu['ci_low']:.1f},{100*bu['ci_high']:.1f}] "
                  f"-- 95% CIs overlap. Point estimate moved {delta*100:+.1f}pp, but that is not a "
                  f"separation at n={sample_n} and is reported as a tie, not a win.")
        elif delta < 0:
            print(f"    LOWER, CI-separated. {100*u['rate']:.1f}% "
                  f"[{100*u['ci_low']:.1f},{100*u['ci_high']:.1f}] vs baseline "
                  f"{100*bu['rate']:.1f}% [{100*bu['ci_low']:.1f},{100*bu['ci_high']:.1f}] "
                  f"({delta*100:+.1f}pp).")
        else:
            print(f"    HIGHER, CI-separated. {100*u['rate']:.1f}% "
                  f"[{100*u['ci_low']:.1f},{100*u['ci_high']:.1f}] vs baseline "
                  f"{100*bu['rate']:.1f}% ({delta*100:+.1f}pp) -- this arm made grounding WORSE.")

        # Earned or abstained? Asked of every arm whose rate moved down at
        # all, CI-separated or not -- a point-estimate "improvement" bought
        # by going quiet is worth naming even when it's a tie.
        neutral_delta = f["neutral_rate"] - bf["neutral_rate"]
        supp_delta = f["n_confident_supported"] - bf["n_confident_supported"]
        print(f"    Neutral rate {100*f['neutral_rate']:.1f}% vs baseline "
              f"{100*bf['neutral_rate']:.1f}% ({neutral_delta*100:+.1f}pp). "
              f"Confident-and-supported stances {f['n_confident_supported']} vs "
              f"{bf['n_confident_supported']} ({supp_delta:+d}).")
        if delta < 0 and supp_delta <= 0:
            print(f"    -> EARNED? NO. Whatever the unsupported rate did, this arm produced "
                  f"{'fewer' if supp_delta < 0 else 'no more'} well-grounded confident stances "
                  f"than the baseline. Any drop is the agent going quiet, not reasoning better.")
        elif delta < 0 and supp_delta > 0:
            print(f"    -> EARNED. Lower unsupported rate AND {supp_delta} more "
                  f"confident-and-supported stances -- more grounded output, not less output.")

        # Cost, stated whether or not there was a gain.
        c, bc = f["cost"], bf["cost"]
        lat_x = c["mean_reasoning_s"] / bc["mean_reasoning_s"] if bc["mean_reasoning_s"] else float("nan")
        tok_x = c["mean_reasoning_tokens"] / bc["mean_reasoning_tokens"] if bc["mean_reasoning_tokens"] else float("nan")
        cost_note = (f"    Cost: {tok_x:.2f}x baseline tokens, {lat_x:.2f}x baseline reasoning "
                     f"latency ({c['mean_reasoning_s']:.2f}s vs {bc['mean_reasoning_s']:.2f}s per note).")
        if overlap and lat_x > 1.2:
            cost_note += " Paid for, and bought no CI-separated gain."
        print(cost_note)

    # A confound worth stating in the verdict, not buried in the funnel: some
    # bad spans are the model over-claiming, and some are it wrapping a REAL
    # quote in quote characters. The headline charges an arm for both,
    # because the verifier is applied unmodified and identically to every arm
    # -- so the split has to be visible for the headline to be readable.
    if all("span_failures" in s for s in summary.values()):
        artifact_keys = ("quote_wrapping_artifact", "whitespace_artifact")
        rows = []
        for name, s in summary.items():
            by_cat = s["span_failures"]["by_category"]
            total = s["span_failures"]["n_bad_spans"]
            artifact = sum(by_cat.get(k, 0) for k in artifact_keys)
            rows.append((name, total, artifact, total - artifact, by_cat))
        if any(a for _, _, a, _, _ in rows):
            print("\n  CONFOUND -- not every mechanically-unsupported span is a grounding failure:")
            for name, total, artifact, real, by_cat in rows:
                pct = f"{100 * artifact / total:.0f}%" if total else "n/a"
                print(f"    {name:17s} {total:>3d} bad spans = {real:>3d} genuine over-claim "
                      f"+ {artifact:>3d} formatting artifact ({pct} artifact)")
            print("    A formatting artifact is a span that IS in the note but was returned "
                  "wrapped in\n    quote characters, so the exact-substring test failed on "
                  "punctuation the model added.\n    It is charged to the arm anyway: the verifier "
                  "is held unmodified and identical across\n    arms by the terms of this "
                  "experiment, and correcting it means editing an arm's prompt,\n    which is a "
                  "separate run. Read any arm's gap against its artifact share above.")

    print(f"\n  Model: {model} (identical across all three arms). Verifier: "
          f"ra.verifier(), unmodified and identical across all three arms. "
          f"Retrieval: shipped in-process BM25, k={K}, identical across all three arms.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    sample_n, seed = args.n, args.seed
    out_json = args.out or (ROOT / "experiments" / f"reasoning_comparison_n{sample_n}.json")
    cache_path = ROOT / "experiments" / f"reasoning_stance_cache_n{sample_n}.pkl"
    # --report-only READS the results/checkpoint file but must never write it:
    # a full run in another process is checkpointing into that exact path, and
    # writing it from here would race that writer and could hand it back a
    # truncated file. Partial reports go to their own path instead.
    write_json = (out_json.with_name(out_json.stem + "_partial.json")
                  if args.report_only else out_json)

    print("=== reasoning_stack.py ===\n")

    # -- Step 0 -------------------------------------------------------------
    print("Step 0: reuse, don't reimplement.")
    with open(RETRIEVAL_CONFIG_PATH) as f:
        retrieval_mode = json.load(f).get("mode", "inprocess")
    assert retrieval_mode == "inprocess", (
        f"config/retrieval.json mode is {retrieval_mode!r}, expected 'inprocess'. Every arm must "
        f"share the shipped in-process BM25 over ra.POLICY_CORPUS -- routing through the MCP "
        f"subprocess would add transport variance to an experiment that is supposed to vary "
        f"reasoning ONLY.")
    print(f"  config/retrieval.json mode = {retrieval_mode!r}  (asserted)")

    config = lc.load_config()
    lc.configure_from_config()
    install_meter()
    reused = [
        "text_stance", "verifier", "_retrieve_policy", "_call_llm_json", "_strip_code_fence",
        "_neutral_fallback", "STANCE_SYSTEM_PROMPT", "STANCE_USER_TEMPLATE",
        "POLICY_CORPUS", "POLICY_CORPUS_BY_ID", "CONF_THRESHOLD", "_llm_client",
    ]
    missing = [s for s in reused if not hasattr(ra, s)]
    assert not missing, f"reconciler_agent is missing expected symbols: {missing}"
    print(f"  reused from reconciler_agent: {', '.join(reused)}")
    print(f"    ra.POLICY_CORPUS: {len(ra.POLICY_CORPUS)} chunks | "
          f"ra.POLICY_CORPUS_BY_ID: {len(ra.POLICY_CORPUS_BY_ID)} ids | "
          f"ra.CONF_THRESHOLD = {ra.CONF_THRESHOLD}")
    print(f"    ra._llm_client: {ra._llm_client.__class__.__name__} "
          f"(provider={config.get('provider')}, model={config.get('model')}, "
          f"temperature={config.get('temperature')})")
    print("  reused from scripts/eval_agent: load_test_population, base_rate_sample, "
          "wilson_ci, UNDERPOWERED_N")
    print("  NOTE: ra.STANCE_* prompts are used by arm0 only, via ra.text_stance itself. "
          "Arms 1 and 2 carry their own prompts.")
    print("  NOT touched: agent/reconciler_agent.py, the verifier, config/*.\n")

    # -- Step 1 -------------------------------------------------------------
    print(f"Step 1: freeze the sample -- base_rate_sample from the locked TEST split, "
          f"n={sample_n}, seed={seed} ...")
    test_frame = load_test_population()
    sample = base_rate_sample(test_frame, sample_n, seed)
    print(f"  n={len(sample)}  default rate {sample['default'].mean():.4f}  "
          f"(TEST full population: {len(test_frame):,})")
    print(f"  retrieval is deterministic and self-run by each arm, so all three arms see the "
          f"identical top-{K} chunks per note.\n")

    records = {name: [] for name in ARM_NAMES}
    completed_loan_ids = set()
    if out_json.exists():
        with open(out_json) as f:
            prev = json.load(f)
        if prev.get("sample_n") == sample_n and prev.get("seed") == seed:
            records = {name: prev.get("records", {}).get(name, []) for name in ARM_NAMES}
            per_arm_ids = [{r["loan_id"] for r in records[name]} for name in ARM_NAMES]
            completed_loan_ids = set.intersection(*per_arm_ids) if all(per_arm_ids) else set()
            # Purge partial records for loans that didn't finish every arm --
            # the checkpoint only writes every CHECKPOINT_EVERY notes, so a
            # crash can land mid-note. The main loop reprocesses them.
            for name in ARM_NAMES:
                records[name] = [r for r in records[name] if r["loan_id"] in completed_loan_ids]
            print(f"Resuming from {out_json.name}: {len(completed_loan_ids)}/{len(sample)} notes "
                  f"already complete across all {len(ARM_NAMES)} arms.\n")

    if args.report_only:
        if not completed_loan_ids:
            raise SystemExit(f"--report-only: nothing complete in {out_json}")
        print(f"--report-only: aggregating the {len(completed_loan_ids)} completed notes "
              f"and skipping the LLM loop.\n")
    else:
        # -- Step 2 ---------------------------------------------------------
        cache = {}
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
            print(f"Loaded {len(cache)} cached (loan_id, arm) stance results from {cache_path.name}")

        n_hits = n_new = n_processed = 0
        print(f"Step 2: running {len(sample) - len(completed_loan_ids)} remaining notes through "
              f"{len(ARM_NAMES)} arms (reasoning + the SAME verifier per arm) ...")
        for raw_loan_id, row in sample.iterrows():
            # Native int: json.dump(default=str) on checkpoint would stringify
            # a numpy.int64 and it would never match again on resume.
            loan_id = int(raw_loan_id)
            if loan_id in completed_loan_ids:
                continue
            desc = (row["desc_clean"] or "").strip()
            policy = ra._retrieve_policy(desc) if desc else []
            retrieved_ids = [p["id"] for p in policy]

            for arm_name, arm_fn in ARMS.items():
                cache_key = (loan_id, arm_name)
                if cache_key in cache:
                    stance_out, reasoning_cost = cache[cache_key]
                    n_hits += 1
                else:
                    METER.reset()
                    stance_out = arm_fn(desc, policy)
                    reasoning_cost = METER.snapshot()
                    cache[cache_key] = (stance_out, reasoning_cost)
                    n_new += 1

                # The verifier is always run fresh, never cached -- it is the
                # measuring instrument and must be applied identically to
                # every arm on every note.
                chain = stance_out.get("chain")
                verifier_in = {k: v for k, v in stance_out.items() if k != "chain"}
                METER.reset()
                verifier_out = score(desc, verifier_in)
                verifier_cost = METER.snapshot()

                records[arm_name].append({
                    "loan_id": loan_id,
                    "retrieved_ids": retrieved_ids,
                    "stance": stance_out["stance"],
                    "stance_confidence": stance_out["stance_confidence"],
                    "stance_source": stance_out["stance_source"],
                    "n_evidence": len(stance_out.get("stance_evidence") or []),
                    "n_policy_ids": len(stance_out.get("stance_policy_ids") or []),
                    "verifier_verdict": verifier_out["verifier_verdict"],
                    "verifier_source": verifier_out["verifier_source"],
                    "reasoning_calls": reasoning_cost["calls"],
                    "reasoning_tokens": reasoning_cost["total_tokens"],
                    "reasoning_s": reasoning_cost["seconds"],
                    "verifier_tokens": verifier_cost["total_tokens"],
                    "verifier_s": verifier_cost["seconds"],
                    "usage_missing": reasoning_cost["usage_missing"] + verifier_cost["usage_missing"],
                    "chain": _trim_chain(chain),
                })

            n_processed += 1
            if n_processed % CHECKPOINT_EVERY == 0:
                done = len(completed_loan_ids) + n_processed
                print(f"  ... {done}/{len(sample)}  (cache: {n_hits} hits, {n_new} new this run)")
                out_json.parent.mkdir(parents=True, exist_ok=True)
                with open(out_json, "w") as f:
                    json.dump({"_status": "in_progress", "sample_n": sample_n, "seed": seed,
                               "records": records}, f, indent=2, default=str)
                with open(cache_path, "wb") as f:
                    pickle.dump(cache, f)

        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)
        print(f"\nDone: {len(sample)} notes x {len(ARM_NAMES)} arms. "
              f"Cache: {n_hits} hits, {n_new} new calls ({len(cache)} cached total).")

    # -- Step 3 -------------------------------------------------------------
    n_scored = len(records[ARM_NAMES[0]])
    summary = {
        name: {"unsupported": unsupported_stats(records[name]),
               "funnel": funnel(records[name]),
               "llm_layer": llm_layer_stats(records[name])}
        for name in ARM_NAMES
    }

    # Decompose the mechanical failures. Reads the persisted stance cache (no
    # LLM calls); skipped with a printed note if the cache isn't on disk, so
    # a --report-only pass without it still produces the rest of the report.
    scored_ids = {r["loan_id"] for r in records[ARM_NAMES[0]]}
    desc_by_id = {
        int(i): (row["desc_clean"] or "").strip()
        for i, row in sample.iterrows() if int(i) in scored_ids
    }
    spans = None
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            spans = span_failure_breakdown(pickle.load(f), desc_by_id)
        for name in ARM_NAMES:
            summary[name]["span_failures"] = spans[name]
    else:
        print(f"\n(no {cache_path.name} on disk -- skipping the span-failure breakdown)")

    # -- Step 4 -------------------------------------------------------------
    print_report(summary, n_scored, config.get("model"))

    report = {
        "_status": "complete" if not args.report_only else "report_only",
        "sample_n": sample_n,
        "n_scored": n_scored,
        "seed": seed,
        "held_fixed": {
            "retrieval": f"shipped in-process BM25 over ra.POLICY_CORPUS "
                         f"({len(ra.POLICY_CORPUS)} chunks), k={K}",
            "retrieval_mode": retrieval_mode,
            "verifier": "agent/reconciler_agent.py verifier(), unmodified",
            "model": config.get("model"),
            "temperature": config.get("temperature"),
            "conf_threshold": ra.CONF_THRESHOLD,
        },
        "varied": "the note -> stance reasoning only",
        "summary": summary,
        "records": records,
    }
    write_json.parent.mkdir(parents=True, exist_ok=True)
    with open(write_json, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nSaved {write_json}")

    # -- Optional MLflow tracking -------------------------------------------
    if args.track:
        from tracking import track_run
        import mlflow

        print(f"\nLogging {len(summary)} runs to MLflow experiment {args.experiment!r} "
              f"(local ./mlruns) ...")
        for name, s in summary.items():
            u, f = s["unsupported"], s["funnel"]
            c = f["cost"]
            params = {
                "arm": name,
                "model": config.get("model"),
                "corpus": "shipped",
                "k": K,
                "n": n_scored,
                "seed": seed,
            }
            metrics = {
                "unsupported_rate": u["rate"] if u["rate"] is not None else float("nan"),
                "unsupported_ci_low": u["ci_low"] if u["ci_low"] is not None else float("nan"),
                "unsupported_ci_high": u["ci_high"] if u["ci_high"] is not None else float("nan"),
                "unsupported_k": u["k"],
                "non_neutral_n": u["n"],
                "neutral_rate": f["neutral_rate"] if f["neutral_rate"] is not None else float("nan"),
                "n_confident_non_neutral": f["n_confident_non_neutral"],
                "n_confident_supported": f["n_confident_supported"],
                "n_confident_not_unsupported": f["n_confident_not_unsupported"],
                "n_mechanical_unsupported": f["n_mechanical_unsupported"],
                "n_llm_unsupported": f["n_llm_unsupported"],
                "mean_reasoning_calls": c["mean_reasoning_calls"],
                "mean_reasoning_tokens": c["mean_reasoning_tokens"],
                "mean_reasoning_s": c["mean_reasoning_s"],
                "mean_total_tokens": c["mean_total_tokens"],
                "mean_total_s": c["mean_total_s"],
                **{f"verdict_{v}": n for v, n in f["verdict_breakdown"].items()},
                **{f"verifier_source_{v}": n for v, n in f["verifier_source_breakdown"].items()},
            }
            # The two decompositions that explain the headline -- without
            # these, a run in MLflow shows an arm's unsupported rate with no
            # way to see that most of it was span formatting rather than
            # ungrounded reasoning. Same post-treatment caveat as
            # llm_layer_stats(): mech_pass_rate is logged alongside so the
            # selection effect stays visible in the tracked record too.
            ll = s["llm_layer"]
            metrics.update({
                "llm_layer_unsupported_rate": ll["rate"] if ll["rate"] is not None else float("nan"),
                "llm_layer_ci_low": ll["ci_low"] if ll["ci_low"] is not None else float("nan"),
                "llm_layer_ci_high": ll["ci_high"] if ll["ci_high"] is not None else float("nan"),
                "llm_layer_n_reached": ll["n_reached_llm"],
                "mech_pass_rate": ll["mech_pass_rate"] if ll["mech_pass_rate"] is not None else float("nan"),
            })
            sf = s.get("span_failures")
            if sf is not None:
                by_cat = sf["by_category"]
                artifact = (by_cat.get("quote_wrapping_artifact", 0)
                            + by_cat.get("whitespace_artifact", 0))
                metrics.update({
                    "bad_spans_total": sf["n_bad_spans"],
                    "bad_spans_formatting_artifact": artifact,
                    "bad_spans_genuine_overclaim": sf["n_bad_spans"] - artifact,
                    "bad_spans_quoted_policy": by_cat.get("quoted_the_POLICY_as_evidence", 0),
                    "bad_spans_artifact_share": artifact / sf["n_bad_spans"] if sf["n_bad_spans"] else 0.0,
                })
            with track_run(experiment=args.experiment, run_name=f"n{n_scored}_{name}", params=params):
                mlflow.log_metrics(metrics)
                mlflow.log_artifact(str(write_json))
            print(f"  logged run: n{n_scored}_{name}")


if __name__ == "__main__":
    main()
