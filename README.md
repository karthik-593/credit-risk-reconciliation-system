# Credit-Risk Reconciliation System

**A strict tabular model rejects some good borrowers. An independent note-reader catches them back — it flags borrowers the model wrongly declined for human review, grounded in lending policy.**

**[Live demo](https://credit-risk-reconciliation-system-ahfvuevc5vbvpq6r4uozjm.streamlit.app/)** — historical 2007–2013 data, explore mode.

---

## The question

Does a borrower's free-text loan description carry credit-risk signal **beyond** the tabular underwriting fields (grade, rate, income, DTI, FICO) already capture — or is any apparent lift just a selection artifact (who bothers to write a note, or how much they write)? Built on the LendingClub accepted-loans dataset (2007–2018, ~2.2M loans), using the `desc` free-text field present on loans originated **2007–2013** — the field was discontinued industry-wide after 2015, so this is a methods study on historical data, stated plainly rather than implied.

Every number below traces to a file in this repo. Nothing is rounded up, and the parts that didn't work are reported alongside the parts that did — that's the point of the project, not an afterthought.

---

## What it does

Two readers judge the same loan independently — neither sees the other's output. A number-reader (XGBoost) scores the tabular fields. A word-reader (a local LLM) reads only the borrower's note and retrieved lending-policy snippets, and forms a stance — corroborates risk, mitigates risk, or says nothing (most notes say nothing). A verifier checks that the word-reader's citations are actually grounded in the note and the retrieved policy, downgrading unsupported claims before they reach the next step. A rule-based reconciler compares the two readers: agreement or silence takes the tabular decision automatically; a confident disagreement goes to human review with a memo. The reconciler decides on plain rules — the LLM never makes the final call.

---

## Honest headline results

| Question | Result |
|---|---|
| Does the note carry signal beyond tabular features? | **Yes, small but real** — content lift +0.0099 PR-AUC (3 seeds, holds sign) |
| Tabular model quality | PR-AUC 0.2803 on locked test (~1.8x the 0.153 base-rate floor; much higher would signal leakage) |
| What the agent catches | ~24% of the model's **wrongful declines** (good borrowers it would have rejected) vs. only ~6% of its **bad approvals** (defaulters it let through) |
| Does disagreement improve final decisions? | **Inconclusive** at n=21,616 — point estimate leans the right way, confidence intervals overlap |
| Does grounding-checking help? | The verifier cuts the disagree rate ~12%→7% (statistically significant) and downgrades ~41% of otherwise-confident, conflicting stances as unsupported |

- **The text signal is real but small.** Isolated with a 3-way test (presence-only, length-only, actual content), across 3 seeds: presence alone −0.0002 PR-AUC, length alone +0.0014, actual content +0.0099. The content lift survives both controls and holds sign across every seed — modest (~3.6% relative over tabular-only), but real, not noise.
- **The tabular model is deliberately not over-strong.** PR-AUC 0.2803 against a 0.1526 base rate (13 origination-time features only, leakage-audited, no post-origination columns). A near-perfect score on a problem this hard would be the tell that something leaked, not a win.
- **The agent's real value is rescuing wrongful declines, not catching missed defaulters.** Of the tabular model's own errors, disagreement-routing catches 24.05% of the loans it wrongly declined vs. 5.90% of the defaulters it wrongly approved — the note is much better at pushing back on over-caution than at raising a flag the numbers missed.
- **Whether any of this changes outcomes for the better is not proven.** At the full 21,616-loan locked test set — 10x the power of the first read — both headline comparisons (flagged-approves vs. clean-approves; mitigated-declines vs. clean-declines) still have overlapping 95% CIs. Point estimates lean the expected direction; that is reported as encouraging, not as evidence.
- **Fair** (approval-rate shifts of −2.6pp to −3.8pp across demographic-proxy groups, no group exceeds the 5pp flag) **and calibrated** (auto-decided subset Brier 0.1157 vs. 0.1213 for the full population — the automated cases are slightly better-calibrated, not worse).
- **Within budget.** Review rate lands at 11.57% of all loans against a ~15% assumed budget.

Full statistical record — every build, every bug, every honest verdict — in [`DECISIONS.md`](DECISIONS.md).

---

## What didn't work

This is the section most portfolio projects skip. Two extensions were built, measured honestly, and did not survive measurement — that is reported here as plainly as the parts that did work.

**As a pure review-triage queue, the agent loses to just ranking by tabular score.** Reframing routing as a review-budget problem (Build 8): if a reviewer can only look at a fixed % of loans, does the agent's disagree-first queue front-load real defaults better than simply ranking every loan by p_default? No — tabular-alone AUC 0.6627 vs. the agent's queue 0.6219 (gap −0.0408), and the agent loses at every tested budget (−2.6pp to −6.6pp). The `disagree` bucket mixes two different conflict directions by construction, so it isn't sorted by raw risk the way a pure ranking is — a different objective from catching wrongful declines, and an honest loss on this one.

**A bounded agentic layer cut review load 3.6x — and initially decided worse than chance.** `agent/bounded_reconciler.py` extends the reconciler: on a confident tabular-vs-narrative conflict, it looks up ~50 similar historical loans (grade-banded, nearest-neighbor on 8 standardized origination features, TRAIN-only, leakage-tested first) and may resolve the conflict to the tabular decision instead of automatically deferring to a human. Build 11 measured it against real outcomes: review rate dropped 8.76%→2.42%, but on the 317 conflicts it resolved on its own, its decisions matched the actual outcome only 40.4% of the time — **worse than a coin flip**. Diagnosis: the tiebreaker compared a neighbor default *rate* directly against the decision *threshold* (0.1703) — a deliberately low bar tuned for the whole population, so a 20%-of-neighbors-defaulted reading (a minority outcome) was being read as "strong evidence to decline." Build 12 fixed it to require the neighbor rate be genuinely extreme (≤0.5x or ≥2.0x the population base rate) before acting at all, deferring on everything in between. Re-measured: auto-resolved conflicts dropped from 317 to 107, and the decision-match rate moved from confidently-worse-than-chance to statistically indistinguishable from chance (47.7%, CI straddles 50%) — real progress — but it still does not beat simply deferring to a human. **The simpler original design was correct, and that was proven by measurement, not assumed.**

---

## The stack

| Component | Choice |
|---|---|
| Tabular model | XGBoost (Optuna-tuned: max_depth 2, 427 trees), 13 origination-time features, SHAP drivers. Leakage-audited — no post-origination fields. No calibration layer (measured, not assumed: isotonic calibration made test-set Brier and ECE slightly *worse*, 0.0029→0.0046 ECE — so it was left out). |
| Text signal | TF-IDF (~1,500 features, min_df=5) — chosen over embeddings because the features are auditable words. |
| Word reader | Qwen 2.5, 7.6B params, Q4_K_M (4-bit) quantized, via Ollama, local, temperature 0, behind a provider-agnostic `LLMClient` (Gemini adapter also implemented). |
| Verifier | A second, independent LLM pass that checks the word-reader's own quoted evidence and cited policy actually support its stance — mechanical substring/citation check first (free), LLM check second, safe-direction only (downgrades to neutral, never escalates). |
| Retrieval (shipped default) | BM25 top-k=4 over a 19-chunk synthetic policy corpus. An MCP-wrapped version of the same retrieval exists (`mcp/policy_server.py`), opt-in via `config/retrieval.json`, proven byte-identical to the in-process path by an equivalence test — off by default. |
| Orchestration | LangGraph — a 7-node state machine (tabular score → stance → verifier → reconciler → auto_decision/human_review → explanation) with human-in-the-loop routing. |
| Decision threshold | 0.1703 (F1-max on a validation slice, never test — not the arbitrary 0.50 default). |
| Evaluation | Locked test split (21,616 loans), Wilson 95% CIs everywhere, base-rate-preserved sampling, explicit failure reason-codes (an LLM timeout is never silently folded into "neutral"). |

A separate retrieval experiment explored a much larger, partly-real policy corpus — see below; it never became the shipped default.

---

## Engineering notes

The hardest part of this project was making sure the evaluation wasn't lying to me — every finding below was caught by re-reading raw rows or re-deriving a number independently, not by trusting a first pass.

- **Three evaluation-harness bugs, caught before they shaped a conclusion.** An early full run showed a 73.85% review rate against a ~15% budget — alarming, until diagnosis showed it was two compounding artifacts: (1) the routing logic sent *silent* (neutral/low-confidence) stances to human review alongside genuine disagreements, conflating "the note said nothing" with "the note pushed back"; (2) the decision threshold was still XGBoost's unexamined 0.50 default, which almost never fires at a ~15% base rate, so the decline side of every comparison was structurally empty. A third, smaller gap — reporting a point-estimate comparison as "does not hold" without its confidence interval — was caught and corrected to the "inconclusive at this n" standard used everywhere else in the project.
- **A plain-English label-swap**, caught by reading raw rows: the 24%/6% catch-rate figures were numerically correct but described backwards (called "false approvals" when they were actually wrongful declines, and vice versa). Numbers and code were right; only the English was inverted. Fixed in exactly the three places it appeared.
- **A model-selection rejection.** Llama 3.1 8B was the first local LLM tried for the word-reader; a validation run showed it citing the retrieved *policy text* as if it were the applicant's own words in 3 of 5 ungrounded cases — a real reasoning defect, not a data artifact. Switched to Qwen 2.5 7.6B, which didn't reproduce the failure.

---

## Retrieval experiment: a robust null

A separate line of work (`experiments/retrieval_stack.py`) A/B/C-tested three retrievers — BM25, a `bge-small-en-v1.5` bi-encoder, and bi-encoder + cross-encoder rerank — against the verifier's unsupported rate, across three sample sizes (n=200, 1,000, 2,000) and two policy corpora (a 96-chunk corpus of 31 real SEC-extracted lending-policy chunks + 65 synthetic ones; a 110-chunk version adding 14 chunks that reword the real rules into borrower-risk language). Result at every scale: **a tie**. The three retrievers' 95% CIs on unsupported rate overlap at n=200, n=1,000, and n=2,000, on both corpora. The reworded chunks fire more often on the dense retrievers than the verbatim-real ones did (up to 49.9% vs. 37.7%) but *less* often on BM25 — and none of it moved the unsupported rate. Read together: **the bottleneck is the model's reasoning over evidence, not which retriever finds the evidence.** Tracked in MLflow (`experiments/tracking.py`, local file store, never wired into the shipped agent).

---

## Honest limitations

- **Vintage-bound.** Built on 2007–2013 loans; the `desc` field no longer exists on new LendingClub originations. This is a methods demonstration on a historical corpus, not a deployable live product.
- **The shipped policy corpus is synthetic.** The 19-chunk corpus the live agent actually retrieves from is honestly constructed, aligned to the model's features — not verbatim lender text. A separate, larger corpus with 31 chunks extracted from the actual SEC-filed LendingClub/WebBank credit policy was built and A/B-tested (see above), but it was never wired in as the default; the shipped retrieval remains fully synthetic.
- **The headline is inconclusive.** The text signal is real but small; whether it improves final decisions is not proven, even at the full 21,616-loan population. That is the finding, reported as-is, not as a caveat to explain away.
- **The agentic extension didn't beat the simple baseline.** Two builds (11 and 12) tried to let the agent resolve some conflicts itself instead of always deferring to a human. The fixed version stopped making confidently bad calls, but never demonstrated it decides better than deferring. The simpler design ships; the more agentic one doesn't, and that's a measured result, not a design preference.

---

## Repository

```
scripts/       feasibility pipeline (01–06) + train_final.py + compute_decision_threshold.py
               + eval_agent.py, eval_verifier.py, review_efficiency.py, measure_perf.py
notebooks/     feasibility_story.ipynb (does text help) · tabular_tuning.ipynb (Optuna + calibration)
agent/         reconciler_agent.py (the shipped 5-node graph) · llm_client.py
               bounded_reconciler.py + similar_loan_tool.py (the agentic extension — not wired in)
               mcp_policy_client.py (opt-in MCP retrieval path) · tests for all of the above
mcp/           policy_server.py (MCP server wrapping the same in-process retrieval)
experiments/   retrieval_stack.py (BM25 vs bi-encoder vs rerank A/B/C) · ablation_agentic.py
               (fixed vs agentic pipeline) · tracking.py (local MLflow helper)
config/        frozen decisions: tabular_best_params · decision_threshold · llm · retrieval
               + the experimental policy_corpus_v2/v3.json (not the shipped default)
app/           streamlit_demo.py (the live demo) + deploy notes
results/       feasibility_metrics.json · agent_eval_fullpower.json · cached stances/verdicts
DECISIONS.md   the full honest record — every build, every finding, every failure, every fix
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
