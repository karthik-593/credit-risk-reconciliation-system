"""
Credit-Risk Reconciliation Agent â 4-node LangGraph skeleton.

Flow:
    START -> tabular_score -> text_stance -> reconciler
                                                |
                     agree/low-conf (text silent or agrees) -> auto_decision --\
                     disagree (text actively conflicts)      -> human_review ---> explanation -> END

Design invariants (do not break these â they are the whole point):
  1. text_stance NEVER sees p_default. It forms an INDEPENDENT read from the
     borrower narrative + retrieved policy only. The reconciler is the first
     node that sees both channels.
  2. The reconciler DECIDES with simple, defensible rules over (stance, confidence).
     It does not invent a "narrative risk probability" to weigh against the
     calibrated tabular one.
  3. Retrieval is a TOOL the stance node calls, not a separate "agent".

Fill in the three TODO stubs (model load, LLM call, retriever) and it runs.
"""

from __future__ import annotations
import json
import pickle
import re
from pathlib import Path
from typing import Literal, Protocol, TypedDict, Optional

import numpy as np
import pandas as pd
import shap
from rank_bm25 import BM25Okapi
from langgraph.graph import StateGraph, START, END


# --------------------------------------------------------------------------
# The text_stance prompt â the piece where a good prompt vs a lazy one is the
# whole difference. Independence + a strict output contract are baked in.
# --------------------------------------------------------------------------
STANCE_SYSTEM_PROMPT = """You are an underwriting narrative analyst.

You are shown ONLY a loan applicant's own written statement and a set of
retrieved underwriting-policy snippets. You are NOT shown the model's risk
score, and you must NOT try to guess or output a risk probability.

Your single job: judge whether the applicant's statement CORROBORATES risk,
MITIGATES risk, or is NEUTRAL, relative to a standard underwriting read â and
ground that judgment in the retrieved policy.

Rules:
- Base every judgment only on what the statement actually says. Do not invent
  facts the applicant did not state.
- evidence_spans MUST be exact quotes copied from the statement.
- cited_policy_ids MUST come only from the retrieved snippets provided.
- confidence is how CLEARLY the text supports your stance (0-1), NOT a
  probability of default.
- If the statement is empty, boilerplate, or uninformative: stance="neutral",
  confidence low.

Respond with ONLY this JSON, no prose:
{
  "stance": "corroborates_risk | mitigates_risk | neutral",
  "evidence_spans": ["exact quote", ...],
  "cited_policy_ids": ["policy_id", ...],
  "confidence": 0.0,
  "rationale": "one sentence grounded in the cited policy"
}"""

STANCE_USER_TEMPLATE = """Applicant statement:
\"\"\"{desc_clean}\"\"\"

Retrieved policy snippets:
{policy_block}"""


# --------------------------------------------------------------------------
# State â one object threaded through every node.
# --------------------------------------------------------------------------
class AppState(TypedDict, total=False):
    # inputs
    application: dict            # {"tabular_features": {...}, "desc_clean": str}
    # tabular channel
    p_default: float
    shap_drivers: list           # [(feature, contribution), ...]
    # narrative channel (independent of p_default)
    stance: Literal["corroborates_risk", "mitigates_risk", "neutral"]
    stance_evidence: list
    stance_policy_ids: list
    stance_confidence: float
    stance_rationale: str
    stance_source: Literal["parsed", "parse_error", "api_error", "empty"]
    stance_error_detail: str
    # reconciliation
    route: Literal["agree", "disagree", "low_conf"]
    decision: Literal["auto_approve", "auto_decline", "human_review"]
    memo: str


CONF_THRESHOLD = 0.55          # below this, narrative read is too weak to trust

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DECISION_THRESHOLD_PATH = _REPO_ROOT / "config" / "decision_threshold.json"


def _load_high_risk_threshold() -> float:
    """HIGH_RISK is chosen on the VAL slice only (F1-max for the default
    class, Youden's J fallback if degenerate) by
    scripts/compute_decision_threshold.py -- never hardcoded, never touches
    TEST. 0.50 was Model A's unexamined convenience default; at this
    dataset's ~15% base rate it almost never fired. See
    config/decision_threshold.json for the frozen value and how it was
    picked."""
    with open(_DECISION_THRESHOLD_PATH) as f:
        return json.load(f)["threshold"]


HIGH_RISK = _load_high_risk_threshold()   # tabular p_default cutoff for "risky"


# ==========================================================================
# Stub 1 â tabular scoring (Model A, from the reconciliation pipeline)
# ==========================================================================
_MODEL_PATH = _REPO_ROOT / "models" / "model_a_tuned_calibrated.pkl"

_EMP_LENGTH_MAP = {
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
    "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9,
    "10+ years": 10,
}
_CATEGORICAL_COLS = ["grade", "sub_grade", "home_ownership", "verification_status", "purpose"]

_model_bundle: Optional[dict] = None       # {"model", "features", "categories", "config"}
_shap_explainer = None


def _build_tabular_row(features: dict, feature_order: list[str], categories: dict) -> pd.DataFrame:
    """Encode one application's tabular_features dict exactly as Model A was
    trained: term -> numeric months, emp_length -> ordinal years, the five
    categorical columns cast against the SAME category vocabulary/codes used
    at training time (required for correct XGBoost categorical-split
    inference â re-deriving categories independently per call could assign
    different codes to the same string)."""
    row = {}
    for col in feature_order:
        if col == "term":
            raw = str(features.get("term", ""))
            digits = "".join(ch for ch in raw if ch.isdigit())
            row[col] = float(digits) if digits else np.nan
        elif col == "emp_length":
            row[col] = _EMP_LENGTH_MAP.get(features.get("emp_length"), np.nan)
        else:
            row[col] = features.get(col)

    df = pd.DataFrame([row], columns=feature_order)
    for c in _CATEGORICAL_COLS:
        df[c] = pd.Categorical(df[c], categories=categories[c])
    numeric_cols = [c for c in feature_order if c not in _CATEGORICAL_COLS]
    df[numeric_cols] = df[numeric_cols].astype("float32")
    return df


def _load_model_bundle() -> dict:
    global _model_bundle
    if _model_bundle is None:
        with open(_MODEL_PATH, "rb") as f:
            _model_bundle = pickle.load(f)
    return _model_bundle


def _score_tabular(features: dict) -> tuple[float, list]:
    bundle = _load_model_bundle()
    model, feature_order, categories = bundle["model"], bundle["features"], bundle["categories"]

    X = _build_tabular_row(features, feature_order, categories)
    p_default = float(model.predict_proba(X)[:, 1][0])

    global _shap_explainer
    if _shap_explainer is None:
        _shap_explainer = shap.TreeExplainer(model)
    sv = _shap_explainer.shap_values(X)
    sv = sv[-1] if isinstance(sv, list) else sv
    contributions = list(zip(feature_order, np.asarray(sv)[0].tolist()))
    top5 = sorted(contributions, key=lambda kv: abs(kv[1]), reverse=True)[:5]

    return p_default, top5


# ==========================================================================
# Stub 2 â policy retrieval (BM25 over a small synthetic corpus)
# ==========================================================================
POLICY_CORPUS = [
    {"id": "policy_1.1", "text": "Debt-to-income (DTI) ratio above 40% requires manual underwriting review regardless of grade."},
    {"id": "policy_1.2", "text": "DTI between 35% and 40% is acceptable for grades A-C without additional documentation."},
    {"id": "policy_1.3", "text": "A stated intent to consolidate multiple high-interest debts into a single lower-rate payment is a standard, low-risk loan purpose and does not itself indicate elevated risk."},
    {"id": "policy_2.1", "text": "Medical hardship disclosed in the applicant's statement (unexpected procedure, uninsured treatment, family illness) should be treated as a mitigating factor if consistent with the loan amount requested."},
    {"id": "policy_2.2", "text": "Job loss or reduced hours disclosed by the applicant is a corroborating risk factor unless the applicant also states verified re-employment or a stable alternative income source."},
    {"id": "policy_2.3", "text": "General statements of financial 'stress' or 'struggle' without a specific triggering event are neutral and should not be treated as corroborating or mitigating on their own."},
    {"id": "policy_3.1", "text": "Applicants reporting under 1 year at current employer combined with a stated recent job change should be flagged as a corroborating risk factor pending employment verification."},
    {"id": "policy_3.2", "text": "Self-employed or 1099 income disclosed in the applicant's own words, without a stated verification path, is a corroborating risk factor per underwriting guidance section 3."},
    {"id": "policy_3.3", "text": "A stated long tenure (5+ years) at a single employer is a mitigating factor for income stability."},
    {"id": "policy_4.1", "text": "Statements describing the loan proceeds funding a new business venture or startup costs are a corroborating risk factor; business income is not underwritten income."},
    {"id": "policy_4.2", "text": "Home improvement or repair purposes tied to a specific, itemized project are treated as neutral-to-mitigating, since they represent asset-preserving spend."},
    {"id": "policy_4.3", "text": "Vague or generic purpose statements ('need money', 'personal reasons') with no specific detail should be treated as neutral; do not infer risk from vagueness alone."},
    {"id": "policy_5.1", "text": "Any mention of using this loan to pay off or avoid a prior default, repossession, garnishment, or collections action is a strong corroborating risk factor."},
    {"id": "policy_5.2", "text": "Any mention of a co-signer, secondary applicant, or shared household income not reflected in the tabular application data should be flagged for manual verification; do not treat it as automatically mitigating."},
    {"id": "policy_6.1", "text": "Statements that closely match known solicitation or advertising boilerplate (e.g. promotional phrasing copied from marketing material) rather than a first-person account are a fraud-review flag, not a risk-mitigating narrative."},
    {"id": "policy_6.2", "text": "Internal inconsistency between the stated loan purpose and the stated use of funds within the same statement is a fraud-review flag."},
    {"id": "policy_6.3", "text": "A statement that is empty, purely templated, or contains no first-person financial detail carries no narrative signal and must be scored neutral with low confidence."},
    {"id": "policy_7.1", "text": "Retirement income disclosed as the primary repayment source should be treated as neutral unless the applicant also states its amount or stability relative to the requested payment."},
    {"id": "policy_7.2", "text": "Educational expenses (tuition, college costs) disclosed for the applicant or a dependent are treated as neutral; education borrowing is a standard, expected use of consumer credit."},
]

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_bm25_index: Optional[BM25Okapi] = None


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _get_bm25_index() -> BM25Okapi:
    global _bm25_index
    if _bm25_index is None:
        tokenized_corpus = [_tokenize(p["text"]) for p in POLICY_CORPUS]
        _bm25_index = BM25Okapi(tokenized_corpus)
    return _bm25_index


def _retrieve_policy(desc_clean: str, k: int = 4) -> list[dict]:
    if not desc_clean.strip():
        return []
    bm25 = _get_bm25_index()
    scores = bm25.get_scores(_tokenize(desc_clean))
    top_idx = np.argsort(scores)[::-1][:k]
    return [POLICY_CORPUS[i] for i in top_idx]


# ==========================================================================
# Stub 3 â LLM call (provider-agnostic; plug in a concrete client)
# ==========================================================================
class LLMClient(Protocol):
    """Minimal interface text_stance needs. Implement this against whichever
    provider SDK you choose and pass it to configure_llm_client(); the graph
    itself has no dependency on a specific vendor.

    Contract on failure: complete() should RAISE (network error, non-2xx
    after retries, timeout, quota/rate-limit exhaustion, etc.) rather than
    return None or an empty string to signal failure. _call_llm_json treats
    any exception raised here as stance_source="api_error" -- distinct from
    a call that returns text successfully but fails to parse as valid stance
    JSON (stance_source="parse_error"). Don't swallow the underlying error
    inside the adapter; let it propagate so its message ends up in
    stance_error_detail."""

    def complete(self, system: str, user: str) -> str:
        """Return the raw model response text for a single-turn completion."""
        ...


_llm_client: Optional[LLMClient] = None


def configure_llm_client(client: LLMClient) -> None:
    """Wire up the concrete LLM adapter used by text_stance. Call this once
    at process start-up with an adapter implementing LLMClient for whichever
    provider SDK you choose, e.g. configure_llm_client(MyLLMClient(...))."""
    global _llm_client
    _llm_client = client


_NEUTRAL_FALLBACK_BASE = {
    "stance": "neutral",
    "evidence_spans": [],
    "cited_policy_ids": [],
    "confidence": 0.0,
}

# Kept for backward compatibility with anything reading the old shape directly
# (e.g. rationale text comparisons) -- prefer checking stance_source instead,
# which distinguishes WHY this fallback fired.
_NEUTRAL_FALLBACK = {
    **_NEUTRAL_FALLBACK_BASE,
    "rationale": "No narrative read available; see stance_source for why.",
}

_FALLBACK_RATIONALE_BY_SOURCE = {
    "empty": "No LLM client configured or the model returned an empty response; no narrative read available.",
    "api_error": "The LLM call failed (network/API/rate-limit/timeout) after retries; no narrative read available.",
    "parse_error": "The model responded but its output did not parse as valid stance JSON; no narrative read available.",
}

_VALID_STANCES = {"corroborates_risk", "mitigates_risk", "neutral"}


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if not text.startswith("```"):
        return text
    lines = text.split("\n")
    if lines[0].strip("`").lower().startswith("json") or lines[0] == "```":
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _neutral_fallback(source: str, detail: str) -> dict:
    """Same safe neutral/0.0 result regardless of cause, but source+detail
    make WHY it fired visible to callers instead of looking identical to a
    genuine neutral verdict."""
    return {
        **_NEUTRAL_FALLBACK_BASE,
        "rationale": _FALLBACK_RATIONALE_BY_SOURCE[source],
        "stance_source": source,
        "stance_error_detail": detail,
    }


def _call_llm_json(system: str, user: str) -> dict:
    if _llm_client is None:
        return _neutral_fallback("empty", "no LLM client configured")

    try:
        raw = _llm_client.complete(system, user)
    except Exception as exc:
        # Anything raised while getting a response at all -- network error,
        # non-2xx after retries, timeout, quota exhaustion -- is an
        # infrastructure failure, not a model verdict.
        return _neutral_fallback("api_error", f"{type(exc).__name__}: {exc}")

    if not raw or not raw.strip():
        return _neutral_fallback("empty", "LLM returned an empty response")

    try:
        parsed = json.loads(_strip_code_fence(raw))
        if parsed.get("stance") not in _VALID_STANCES:
            raise ValueError(f"invalid stance: {parsed.get('stance')!r}")
    except Exception as exc:
        # The call succeeded -- this is a real response that failed to parse
        # as valid stance JSON, not an infrastructure failure.
        return _neutral_fallback("parse_error", f"{type(exc).__name__}: {exc}")

    parsed = dict(parsed)
    parsed.setdefault("stance_source", "parsed")
    return parsed


# --------------------------------------------------------------------------
# Nodes.
# --------------------------------------------------------------------------
def tabular_score(state: AppState) -> dict:
    p, drivers = _score_tabular(state["application"]["tabular_features"])
    return {"p_default": p, "shap_drivers": drivers}


def text_stance(state: AppState) -> dict:
    """Independent narrative read. Deliberately does NOT read state['p_default']."""
    desc = state["application"].get("desc_clean", "").strip()
    policy = _retrieve_policy(desc) if desc else []
    policy_block = "\n".join(f'[{p["id"]}] {p["text"]}' for p in policy) or "(none)"
    out = _call_llm_json(
        STANCE_SYSTEM_PROMPT,
        STANCE_USER_TEMPLATE.format(desc_clean=desc or "(empty)", policy_block=policy_block),
    )
    return {
        "stance": out.get("stance", "neutral"),
        "stance_evidence": out.get("evidence_spans", []),
        "stance_policy_ids": out.get("cited_policy_ids", []),
        "stance_confidence": float(out.get("confidence", 0.0)),
        "stance_rationale": out.get("rationale", ""),
        "stance_source": out.get("stance_source", "parsed"),
        "stance_error_detail": out.get("stance_error_detail", ""),
    }


def reconciler(state: AppState) -> dict:
    """First node to see BOTH channels. Rule-based, defensible."""
    tabular_risky = state["p_default"] >= HIGH_RISK
    stance = state["stance"]
    conf = state["stance_confidence"]

    if conf < CONF_THRESHOLD or stance == "neutral":
        route = "low_conf"
    elif tabular_risky and stance == "mitigates_risk":
        route = "disagree"          # numbers say risky, words push back
    elif (not tabular_risky) and stance == "corroborates_risk":
        route = "disagree"          # numbers say safe, words raise a flag
    else:
        route = "agree"
    return {"route": route}


def auto_decision(state: AppState) -> dict:
    decision = "auto_decline" if state["p_default"] >= HIGH_RISK else "auto_approve"
    return {"decision": decision}


def human_review(state: AppState) -> dict:
    return {"decision": "human_review"}


def explanation(state: AppState) -> dict:
    """Writes the audit memo for whatever was decided (both branches land here)."""
    memo = (
        f"Decision: {state['decision']}\n"
        f"Tabular p(default): {state['p_default']:.3f} "
        f"(drivers: {state.get('shap_drivers')})\n"
        f"Narrative stance: {state['stance']} (conf {state['stance_confidence']:.2f})\n"
        f"Evidence: {state.get('stance_evidence')}\n"
        f"Policy cited: {state.get('stance_policy_ids')}\n"
        f"Rationale: {state.get('stance_rationale')}"
    )
    # A disagreement memo can be LLM-written here for readability â but the
    # DECISION above was already made by rules, not by the LLM.
    return {"memo": memo}


def _route(state: AppState) -> str:
    """agree AND low_conf both take the tabular decision (auto_decision) --
    a neutral/low-confidence stance means the text channel is SILENT, not
    that it disagrees, so defer to tabular rather than manufacture a review.
    Only genuine disagree (a confident, opposing stance) goes to human_review.
    Changed from the original agree-only rule after Build 5's --real run
    showed low_conf routing to review was inflating review volume with
    silence, not disagreement -- see DECISIONS.md Build 5/6."""
    return "human_review" if state["route"] == "disagree" else "auto_decision"


# --------------------------------------------------------------------------
# Graph.
# --------------------------------------------------------------------------
def build_graph():
    g = StateGraph(AppState)
    g.add_node("tabular_score", tabular_score)
    g.add_node("text_stance", text_stance)
    g.add_node("reconciler", reconciler)
    g.add_node("auto_decision", auto_decision)
    g.add_node("human_review", human_review)
    g.add_node("explanation", explanation)

    g.add_edge(START, "tabular_score")
    g.add_edge("tabular_score", "text_stance")
    g.add_edge("text_stance", "reconciler")
    g.add_conditional_edges("reconciler", _route,
                            {"auto_decision": "auto_decision",
                             "human_review": "human_review"})
    g.add_edge("auto_decision", "explanation")
    g.add_edge("human_review", "explanation")
    g.add_edge("explanation", END)
    return g.compile()


if __name__ == "__main__":
    app = build_graph()
    # example = {"application": {"tabular_features": {...}, "desc_clean": "..."}}
    # print(app.invoke(example)["memo"])
