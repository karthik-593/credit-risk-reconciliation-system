"""
Streamlit demo of the credit-risk reconciliation agent.

DEFAULT MODE ("Explore real results") needs nothing but streamlit + pandas
and the two small JSON files shipped in this folder (demo_samples.json,
policy_corpus.json, headline_stats.json) -- no model, no pickle, no API key.
Those files are pre-built by app/build_demo_samples.py from the real
n=21,616 evaluation (results/agent_eval_fullpower.json), run once locally
where data/models/results all exist.

LIVE MODE (off by default) is best-effort: it tries to import the real
agent/reconciler_agent.py and agent/llm_client.py. On a minimal cloud
deployment those heavy imports (xgboost, shap, langgraph, rank_bm25) or a
missing/unreachable LLM provider will fail -- caught, and the app falls back
to a friendly message and stays on cached mode rather than crashing.

Does not modify agent/reconciler_agent.py.
"""
import json
import random
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent

STANCE_LABELS = {
    "corroborates_risk": "flags risk",
    "mitigates_risk": "eases concern",
    "neutral": "no clear signal",
}
DECISION_LABELS = {
    "auto_approve": "Approved automatically",
    "auto_decline": "Declined automatically",
    "human_review": "Sent to human review",
}
TABULAR_DECISION_LABELS = {"approve": "Approve", "decline": "Decline"}
ROUTE_BANNER = {
    "agree": ("success", "✅ The two readers agreed.",
              "The number reader's model score and the word reader's take on the borrower's "
              "own note pointed the same way, so the agent acted on it automatically."),
    "disagree": ("warning", "⚠️ They disagreed — sent to human review.",
                 "The model score and the note's narrative pointed opposite ways. Rather than "
                 "let either channel overrule the other, the agent defers to a human."),
    "low_conf": ("info", "\U0001f4ac The note said nothing useful — used the numbers.",
                 "The borrower's note was neutral, boilerplate, or the model wasn't confident "
                 "in its read of it. Silence isn't disagreement, so the agent falls back to the "
                 "tabular model's call instead of manufacturing a review."),
}


# ---------------------------------------------------------------------------
# Cached data loading -- the only files the deployed app needs.
# ---------------------------------------------------------------------------
@st.cache_data
def load_demo_samples() -> list[dict]:
    with open(APP_DIR / "demo_samples.json") as f:
        return json.load(f)


@st.cache_data
def load_policy_corpus() -> dict[str, str]:
    with open(APP_DIR / "policy_corpus.json") as f:
        policies = json.load(f)
    return {p["id"]: p["text"] for p in policies}


@st.cache_data
def load_headline_stats() -> dict:
    with open(APP_DIR / "headline_stats.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Live mode: best-effort import of the real agent. Never crashes the app.
# ---------------------------------------------------------------------------
def try_load_live_agent():
    """Returns (ra_module, configure_from_config_fn, error_message).
    error_message is None on success."""
    try:
        sys.path.insert(0, str(ROOT / "agent"))
        import reconciler_agent as ra  # noqa: PLC0415
        from llm_client import configure_from_config  # noqa: PLC0415
    except Exception as exc:  # missing xgboost/shap/langgraph/rank_bm25/etc.
        return None, None, f"Live mode needs the full agent environment, not installed here ({exc})."

    try:
        configure_from_config()
    except FileNotFoundError as exc:
        return None, None, f"No config/llm.json found ({exc})."
    except RuntimeError as exc:
        return None, None, f"LLM provider not reachable/configured ({exc})."
    except Exception as exc:
        return None, None, f"Could not configure the LLM client ({exc})."

    return ra, configure_from_config, None


# ---------------------------------------------------------------------------
# Shared display components
# ---------------------------------------------------------------------------
def render_two_readers(p_default: float, tabular_decision: str, stance: str,
                        confidence: float, rationale: str) -> None:
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("**\U0001f522 NUMBER READER** — tabular model")
            st.metric("p(default)", f"{p_default:.1%}")
            st.markdown(f"Its call: **{TABULAR_DECISION_LABELS.get(tabular_decision, tabular_decision)}**")
    with col2:
        with st.container(border=True):
            st.markdown("**\U0001f4dd WORD READER** — reads only the borrower's note")
            label = STANCE_LABELS.get(stance, stance)
            st.metric("Stance", label, help=f"raw label: {stance}")
            st.markdown(f"Confidence: **{confidence:.0%}**")
            if rationale:
                st.caption(rationale)


def render_route_banner(route: str) -> None:
    kind, headline, explanation = ROUTE_BANNER.get(
        route, ("info", route, "")
    )
    box = {"success": st.success, "warning": st.warning, "info": st.info}[kind]
    box(f"**{headline}**\n\n{explanation}")


def render_evidence(evidence: list) -> None:
    if not evidence:
        st.caption("No quoted evidence from the note (consistent with a neutral/low-signal read).")
        return
    st.markdown("**Quoted from the borrower's note:**")
    for span in evidence:
        st.markdown(f"> {span}")


def render_policy_citations(policy_ids: list, policy_corpus: dict) -> None:
    if not policy_ids:
        st.caption("No underwriting policy cited for this read.")
        return
    st.markdown("**Underwriting policy cited:**")
    for pid in policy_ids:
        text = policy_corpus.get(pid, "(policy text not found)")
        st.markdown(f"- `{pid}` — {text}")


def render_decision_and_outcome(agent_decision: str, y: int) -> None:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Agent's final decision**")
        st.markdown(f"### {DECISION_LABELS.get(agent_decision, agent_decision)}")
    with col2:
        st.markdown("**What actually happened** (historical, already known)")
        if y == 1:
            st.markdown("### ❌ Actually defaulted")
        else:
            st.markdown("### ✅ Actually repaid in full")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Credit Risk Reconciliation — Demo", layout="wide")

    st.markdown(
        "<span style='background:#000;color:#fff;padding:2px 8px;border-radius:4px;"
        "font-size:0.8em;font-weight:600;'>DEMO</span>",
        unsafe_allow_html=True,
    )
    st.title("Credit Risk Reconciliation Agent")
    st.caption(
        "A tabular model reads the numbers. An LLM reads the borrower's own free-text note, "
        "independently, without ever seeing the model's score. A rule-based reconciler decides "
        "what to do when they agree, disagree, or the note has nothing useful to say."
    )
    st.caption(
        "Evaluation used local Qwen 2.5; live mode may use a hosted model. "
        "All loans below are real, historical LendingClub loans from 2007–2013."
    )

    stats = load_headline_stats()
    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Loans evaluated", f"{stats['n_loans_evaluated']:,}")
    c2.metric("Disagree rate", f"{stats['disagree_rate_pct']:.1f}%")
    budget_note = "within" if stats["within_budget"] else "OVER"
    c3.metric("Review rate", f"{stats['review_rate_pct']:.1f}%",
              help=f"{budget_note} the ~{stats['review_budget_pct']:.0f}% budget")
    c4.metric("Bad-approval catch rate", f"{stats['fp_catch_pct']:.0f}%",
              help=f"vs {stats['fn_catch_pct']:.0f}% for bad declines — an asymmetry, "
                   f"not a symmetric safety net")
    st.caption(
        "Historical 2007–2013 loans. Whether disagreement genuinely improves decisions is "
        "still **inconclusive** at this sample size — reported honestly, not rounded into a claim. "
        "See DECISIONS.md in the repo for the full statistical record."
    )
    st.divider()

    live_mode = st.toggle("\U0001f7e2 Live mode (type your own borrower note)", value=False)

    if live_mode:
        run_live_mode()
    else:
        run_explore_mode()


def run_explore_mode():
    samples = load_demo_samples()
    policy_corpus = load_policy_corpus()

    routes_present = sorted({s["route"] for s in samples})
    selected_routes = st.multiselect("Filter by route", routes_present, default=routes_present)
    filtered = [s for s in samples if s["route"] in selected_routes] or samples

    if "picked_loan_id" not in st.session_state:
        st.session_state.picked_loan_id = filtered[0]["loan_id"]

    labels_by_id = {s["loan_id"]: s["label"] for s in filtered}
    ids_in_order = [s["loan_id"] for s in filtered]

    col_a, col_b = st.columns([4, 1])
    with col_a:
        if st.session_state.picked_loan_id not in ids_in_order:
            st.session_state.picked_loan_id = ids_in_order[0]
        picked = st.selectbox(
            "Pick a loan", options=ids_in_order,
            format_func=lambda i: labels_by_id[i],
            key="picked_loan_id",
        )
    with col_b:
        st.write("")
        st.write("")
        if st.button("\U0001f3b2 Surprise me"):
            st.session_state.picked_loan_id = random.choice(ids_in_order)
            st.rerun()

    row = next(s for s in filtered if s["loan_id"] == st.session_state.picked_loan_id)
    render_loan(row, policy_corpus)

    with st.expander(f"\U0001f4cb Browse all {len(filtered)} sample loans in this filter"):
        table = pd.DataFrame(filtered)[
            ["loan_id", "route", "stance", "p_default", "tabular_alone_decision", "agent_decision", "y"]
        ].rename(columns={
            "p_default": "p(default)", "tabular_alone_decision": "number reader's call",
            "agent_decision": "agent decision", "y": "actually defaulted",
        })
        table["p(default)"] = table["p(default)"].map(lambda p: f"{p:.1%}")
        table["actually defaulted"] = table["actually defaulted"].map({0: "no", 1: "yes"})
        st.dataframe(table, width="stretch", hide_index=True)


def render_loan(row: dict, policy_corpus: dict) -> None:
    st.subheader(f"Loan #{row['loan_id']}")
    with st.container(border=True):
        st.markdown("**Borrower's note:**")
        st.markdown(f"> {row['desc_clean']}")

    render_two_readers(
        row["p_default"], row["tabular_alone_decision"],
        row["stance"], row["confidence"], row["stance_rationale"],
    )

    render_route_banner(row["route"])

    col1, col2 = st.columns(2)
    with col1:
        render_evidence(row["stance_evidence"])
    with col2:
        render_policy_citations(row["stance_policy_ids"], policy_corpus)

    st.divider()
    render_decision_and_outcome(row["agent_decision"], row["y"])


def run_live_mode():
    st.subheader("Live mode")
    st.caption(
        "Runs the REAL narrative-reading step (text_stance: BM25 policy retrieval + the "
        "configured LLM) on text you type, then the real rule-based reconciler. There's no "
        "tabular loan application behind free-typed text, so you set a hypothetical model "
        "score yourself below to see how the two channels would reconcile."
    )

    if "live_agent" not in st.session_state:
        ra, _, err = try_load_live_agent()
        st.session_state.live_agent = ra
        st.session_state.live_agent_error = err

    if st.session_state.live_agent_error:
        st.warning(
            f"Live mode isn't available right now — {st.session_state.live_agent_error} "
            f"Showing cached demo results instead."
        )
        run_explore_mode()
        return

    ra = st.session_state.live_agent
    policy_corpus = load_policy_corpus()

    desc = st.text_area(
        "Borrower's note", height=120,
        placeholder="e.g. I'm consolidating three credit cards into one lower monthly payment. "
                    "I've been at my job for six years and have never missed a payment.",
    )
    hypothetical_p = st.slider(
        "Hypothetical p(default) from the tabular model", 0.0, 1.0, 0.15, 0.01,
        help=f"The real decision threshold is {ra.HIGH_RISK:.1%}, chosen on a validation slice.",
    )

    if st.button("Analyze", type="primary", disabled=not desc.strip()):
        with st.spinner("Reading the note..."):
            state = {"application": {"desc_clean": desc}}
            stance_out = ra.text_stance(state)

        if stance_out.get("stance_source") in ("api_error", "empty"):
            st.error(
                f"The model didn't respond (stance_source={stance_out['stance_source']!r}). "
                f"{stance_out.get('stance_error_detail', '')} Try again in a moment."
            )
            return

        full_state = {
            "p_default": hypothetical_p,
            "stance": stance_out["stance"],
            "stance_confidence": stance_out["stance_confidence"],
        }
        route_out = ra.reconciler(full_state)
        tabular_decision = "decline" if hypothetical_p >= ra.HIGH_RISK else "approve"
        decision = (
            ("auto_decline" if tabular_decision == "decline" else "auto_approve")
            if route_out["route"] != "disagree" else "human_review"
        )

        render_two_readers(
            hypothetical_p, tabular_decision,
            stance_out["stance"], stance_out["stance_confidence"], stance_out["stance_rationale"],
        )
        render_route_banner(route_out["route"])
        col1, col2 = st.columns(2)
        with col1:
            render_evidence(stance_out["stance_evidence"])
        with col2:
            render_policy_citations(stance_out["stance_policy_ids"], policy_corpus)
        st.divider()
        st.markdown("**Agent's final decision**")
        st.markdown(f"### {DECISION_LABELS.get(decision, decision)}")
        st.caption("No real-world outcome exists for text you typed yourself.")


if __name__ == "__main__":
    main()
