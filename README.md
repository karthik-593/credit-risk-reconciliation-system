# Credit-Risk Reconciliation System

**A tabular model reads the numbers. A language model reads the borrower's own note — independently, without ever seeing the model's score. A rule-based reconciler decides what to do when they agree, disagree, or the note says nothing useful.**

🔗 **[Live demo](https://credit-risk-reconciliation-system-ahfvuevc5vbvpq6r4uozjm.streamlit.app/)** — browse real 2007–2013 loans, see both readers side by side, and watch where they disagree. *(Explore mode uses cached results from the full evaluation; historical data.)*

---

## The question

Does a borrower's free-text loan description carry credit-risk signal **beyond** the tabular underwriting fields (grade, rate, income, DTI, FICO) already capture — or is any apparent lift just a selection artifact (who bothers to write a note, or how much they write)?

This project answers that honestly, then builds an agent on top of the answer, then measures whether the agent actually improves decisions. Every number below traces to a file in the repo — nothing is rounded up.

Built on the LendingClub accepted-loans dataset (2007–2018, ~2.2M loans), using the `desc` free-text field present on loans originated **2007–2013** (the field was discontinued industry-wide after 2015 — a stated limitation, not a hidden one).

---

## Headline results

**1 — The text carries a small but real signal.** Isolated with a 3-way test (presence, length, content), across 3 seeds:

| Comparison | Mean ΔPR-AUC |
|---|---|
| Presence of a note alone (`has_desc`) | −0.0002 |
| Word count alone (length) | +0.0014 |
| **Actual content (TF-IDF)** | **+0.0099** |
| Content beyond length | +0.0086 |

The content lift survives both controls, holds sign across all seeds, and the top terms read as genuine borrower language (`bills`, `business`, `college`, `retirement`) — not template artifacts. Modest (~3.6% relative over tabular-only), but real.

**2 — The agent, evaluated on 21,616 held-out loans, is honest about what it can and can't prove:**

- The two readers **disagree on 11.6%** of loans; those go to human review — **within a 15% review budget**.
- On ~70% of loans the note says nothing decision-relevant (short, plain 2007-era text) — so the word-reader correctly stays silent rather than inventing an opinion.
- **Whether disagreement produces *better* decisions is inconclusive** even at full sample — the effect leans the right direction but the confidence intervals overlap. Reported as inconclusive, not rounded into a claim.
- **One thing it clearly does:** disagreements catch **24% of the tabular model's wrongful declines** (good borrowers it would have rejected) vs only **6% of its bad approvals** (defaulters it let through) — it's much better at pulling back over-cautious declines than at catching missed bad loans. Stable across sample sizes.
- **Fair** (approval shifts ≤~3pp across groups) and **calibrated** (auto-decided subset slightly better than the whole).

The full statistical record — including three evaluation-harness bugs that were caught and corrected — is in [`DECISIONS.md`](DECISIONS.md).

---

## How it works

```
Loan (13 tabular fields + borrower note)
        │
        ├──────────────┐
        ▼              ▼
  Number reader    Word reader  ← reads only the note + retrieved policy;
  (XGBoost+SHAP)   (Qwen 2.5)      never sees the number reader's score
        │              │
        └──────┬───────┘
               ▼
          Reconciler  (rules over stance + confidence)
               │
     agree / silent → take the tabular decision
     disagree       → human review (with a cited risk memo)
```

- **Independence is the core design choice.** The word-reader forms its judgment before either channel sees the other, so a disagreement is real, not an echo.
- **Rules decide, the LLM explains.** The reconciler routes on plain, auditable rules; the LLM judges the narrative and must quote the borrower's exact words and cite retrieved policy — it never makes the final call.
- **RAG over lending policy.** The word-reader retrieves relevant policy chunks (BM25) and grounds its verdict in them, with citations that are substring-verified against the note.

---

## Technical summary

| Component | Choice |
|---|---|
| Tabular model | XGBoost (Optuna-tuned: depth 2, 427 trees), 13 origination-time features, SHAP drivers. Leakage-audited — no post-origination fields. No calibration layer (measured unnecessary: raw ECE 0.0029). |
| Text signal | TF-IDF (~1,500 features, min_df=5) — chosen over embeddings because the features are auditable words. |
| Word reader | Qwen 2.5 (7.6B) via Ollama, local, temperature 0, behind a provider-agnostic `LLMClient`. |
| Retrieval | BM25 top-k=4 over a 19-chunk synthetic policy corpus (embeddings = planned v2). |
| Orchestration | LangGraph — 4-node state machine with human-in-the-loop routing. |
| Decision threshold | 0.1703 (F1-max on a validation slice, never test). |
| Evaluation | Locked test split, Wilson 95% CIs, base-rate-preserved sampling, failure reason-codes. |

---

## Repository

```
scripts/       feasibility pipeline (01–06) + train_final.py + compute_decision_threshold.py
notebooks/     feasibility_story.ipynb (does text help) · tabular_tuning.ipynb (Optuna + calibration)
agent/         reconciler_agent.py (the 4-node graph) · llm_client.py · tests
config/        frozen decisions: tabular_best_params · decision_threshold · llm
app/           streamlit_demo.py (the live demo) + deploy notes
results/       feasibility_metrics.json · agent_eval_fullpower.json · cached stances
DECISIONS.md   the full honest record — every finding, every failure, every fix
```

## Run it

```bash
pip install -r requirements.txt
# Feasibility pipeline (place the LendingClub .csv.gz at data/raw/ first):
python scripts/01_explore_header.py   # ... through 06
# Reproduce the tuned model:
python scripts/train_final.py
# The demo (local, uses Ollama + Qwen 2.5):
streamlit run app/streamlit_demo.py
```

---

## Honest limitations

- **Vintage-bound.** Built on 2007–2013 loans; the `desc` field no longer exists on new loans. This is a methods demonstration on a historical corpus, not a deployable live product.
- **Synthetic policy corpus.** No real lender playbook is public; the 19 policy chunks are honestly constructed and aligned to the model's features.
- **The headline is inconclusive.** The text signal is real but small; whether it improves final decisions is not proven at this sample size. That is the finding, reported as-is.
