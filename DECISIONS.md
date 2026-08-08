# DECISIONS — Loan Text-Signal Feasibility Test

A running log of what was tested, what came back, and what was decided. Written plainly,
including what didn't work. This file is the honest record — the same discipline that made
the KKBox repo defensible. Record results here as each prompt returns.

---

## Finding 0 — Data audit (Prompt 1) — COMPLETE

**Question:** Does usable borrower text and a real default label coexist in the same years?

**Result:**

`desc` fill rate by issue year:

| Year | Total | Non-empty desc | % Filled | Median len (chars) |
|---|---|---|---|---|
| 2007 | 603 | 588 | 97.5% | 152 |
| 2008 | 2,393 | 2,393 | 100.0% | 196 |
| 2009 | 5,281 | 5,025 | 95.2% | 290 |
| 2010 | 12,537 | 8,286 | 66.1% | 344 |
| 2011 | 21,721 | 12,723 | 58.6% | 264 |
| 2012 | 53,367 | 32,743 | 61.4% | 187 |
| 2013 | 134,814 | 48,730 | 36.2% | 132 |
| 2014 | 235,629 | 15,274 | 6.5% | 115 |
| 2015+ | — | ~0 | 0.0% | — |

Maturity (2007–2013): resolved (Fully Paid + Charged Off) = **227,957 / 230,716 (98.8%)**.
Full-set default rate in-window ≈ 15%.

**Decision:** Both gates pass. Text and outcomes coexist in **2007–2013**. Restrict the
entire test to that window. Proceed to the signal test.

**Caveats carried forward:**
- **Selection bias:** `desc` fill *decays* 97% → 36% across the window. Who writes a
  description is not random → must separate "has text" (Model B) from "text content"
  (Model C). This is why the test is 3-way, not 2-way.
- **Template cruft:** every `desc` wrapped as `Borrower added on MM/DD/YY > … <br>`.
  Strip before text features.

---

## Finding 1 — Fixed frame + template strip (Prompt A) — COMPLETE

**Question:** After restricting to 2007–2013 resolved rows with non-empty desc, how
many rows remain, and did the template strip leave real borrower text?

**Result:**
- Resolved frame (2007–2013, Fully Paid + Charged Off): 227,957 rows — matches Finding 0.
- Class balance, full resolved: 192,619 FP (84.50%) / 35,338 CO (15.50%).
- Non-empty desc (whitespace-stripped): 108,076 rows (47.4% of resolved). Reconciles
  with Finding 0's 110,488 in-window desc rows after subtracting unresolved loans +
  whitespace-only descs (~2,400).
- Class balance, desc subset: 84.74% / 15.26% — marginal default rate 0.24 pts below full.
- Template strip: 'Borrower added on MM/DD/YY >' prefix, <br>, and HTML removed;
  multi-entry descs concatenated with prose intact (per 5 before/after examples).
- Saved: feasibility_frame.pkl (108,076 × 20, incl. desc_clean).

**Decision:** Clean text remains; population fixed at 108,076. Proceed to Model A (Prompt B).

**Caveats carried forward:**
- Near-identical *marginal* default rate (15.26 vs 15.50) is NOT evidence that has_desc
  lacks a *conditional* selection effect — that is exactly what Prompt C measures. Not
  pre-judged here.
- has_desc is constant (=1) on this desc-only frame, so Model B necessarily runs on the
  full resolved set. The decisive gate compares C's lift (desc-only pop) against the
  has_desc lift (full pop) — keep the two populations straight when reading the verdict.
- Multi-entry template residue is the highest-risk cleaning failure; Prompt D's
  top-feature audit is the backstop.

---

## Finding 2 — Model A, tabular baseline + locked split (Prompt B) — COMPLETE

**Question:** What is the tabular-only default-prediction performance — the number to beat?

**Leakage audit:** 13 features, all origination-time — loan_amnt, term, int_rate, grade,
sub_grade, annual_inc, dti, emp_length, home_ownership, verification_status,
fico_range_low, fico_range_high, purpose. No paid_*, balance, last_fico_*, recoveries, or
status-derived fields. Note: int_rate/grade/sub_grade are LendingClub's own priced-in risk
assessment — known at origination (not leakage), but they make the baseline strong and
absorptive, which raises the bar for text. ROC-AUC 0.69 is consistent with a clean
no-leakage model; a near-1.0 would have been the alarm.

**Encoding:** term→numeric months; emp_length ordinal 0–10 with 4,114 (~3.8%) NaN left for
XGBoost native sparsity handling (not imputed); grade/sub_grade/home_ownership/
verification_status/purpose as category dtype via enable_categorical. Split: stratified
80/20, random_state=42 → train 86,460 / test 21,616, both 15.26%. Indices → split_indices.pkl.

**Result (held-out test, 21,616 rows):**
- PR-AUC 0.2769 (no-skill = base rate 0.1526; ~1.81×).
- ROC-AUC 0.6912.

**Decision:** Baseline recorded. Split discipline for downstream models:
- Model C (Prompt D) reuses split_indices.pkl EXACTLY — identical desc-only population.
- Model B (Prompt C) reuses the split POLICY only (stratified 80/20, rs=42) on its own
  full 227,957-row population — has_desc is constant on the desc-only frame, so B cannot
  run there. Two populations; do not conflate.
- Lock PR-AUC = sklearn average_precision_score and the XGBoost hyperparameters; reuse
  verbatim for A-full/B-full/C so the A→C delta reflects features only.

---

## Finding 3 — has_desc selection effect (Prompt C) — COMPLETE

**Question:** Does *merely having* a description predict default, independent of content?

**Result (full 2007–2013 resolved set, 227,957 rows; fresh stratified 80/20, rs=42;
Model A hyperparameters; PR-AUC = average_precision_score; train 182,365 / test 45,592,
both 15.50%):**
- A-full (13 tabular): PR-AUC 0.2809 / ROC-AUC 0.6882.
- B-full (+ has_desc): PR-AUC 0.2807 / ROC-AUC 0.6879.
- Delta B-full − A-full = **−0.0002 PR-AUC** → flat (negative is noise-level; read as zero).
- has_desc gain 6.83, ranks 13/14 — indistinguishable from the weakest tabular feature
  (fico_range_high 6.70), far below grade (181.12).

**SELECTION-EFFECT BAR = −0.0002 (flat).**

**Decision:** The presence of a desc carries ~zero conditional signal once grade/rate/etc.
are in the model — because that selection runs THROUGH grade/rate, which tabular already
holds. Gate for Prompt D effectively reduces to (C − A) > 0.

**Caveats carried forward — the bar is low but the burden is NOT:**
- A flat bar means the selection control absorbs no spurious lift. All defense against a
  FALSE pass now falls on Prompt D's top-feature audit, min_df floor, and 3-seed stability.
  A tiny artifactual lift would clear this gate — authenticity scrutiny goes UP, not down.
- has_desc is binary; it cannot see WITHIN-writer selection (e.g. desc length/verbosity as
  a borrower-trait proxy). TF-IDF L2-norm dampens raw length but not vocabulary richness.
  DECISION: INCLUDE A+len (tabular + desc_clean word count) as a second bar, folded into
  Prompt D on the identical desc-only rows/split/seeds. A+len becomes the stricter
  baseline: the content claim requires C to beat A+len, not just A. Length counted on
  desc_clean so template/multi-entry wrappers don't masquerade as verbosity.

---

## Finding 4 — Model C, text content + verdict (Prompt D) — COMPLETE

**Question:** Does the CONTENT of borrower text add signal beyond (a) tabular, (b) has_desc
presence, and (c) desc verbosity?

**Result (desc-only frame, 108,076 rows; 3 seeds 42/43/44, split reshuffled per seed;
Model A hyperparameters; PR-AUC = average_precision_score; TF-IDF max_features~1500,
English stopwords, min_df=5):**

| Model | PR-AUC | ROC-AUC |
|---|---|---|
| A (13 tabular) | 0.2753 ± 0.0043 | 0.6911 ± 0.0007 |
| A+len (+ word count) | 0.2767 ± 0.0048 | 0.6918 ± 0.0009 |
| C (+ TF-IDF ~1500) | 0.2853 ± 0.0043 | 0.6969 ± 0.0019 |

Deltas (PR-AUC), all seeds positive:
- Length channel (A+len − A): +0.0014 ± 0.0006
- Content lift (C − A): +0.0099 ± 0.0002 (per seed: 0.0097 / 0.0101 / 0.0100)
- Content beyond length (C − A+len): +0.0086 ± 0.0005
- has_desc selection bar (Finding 3): −0.0002 (flat)

Top-20 TF-IDF by gain (seed 42): appreciate, need, bills, business, college, matter,
retirement, combine, rate, thats, card, advertising, fix, problems, growing, things,
opening, proceeds, loans, pool. Zero template residue (no 'borrower'/'added'/'br'/dates).
Several tokens (business, college, pool, proceeds, opening) are purpose-adjacent — yet
`purpose` is already a tabular column, so this lift is WITHIN-purpose discrimination the
column cannot capture. Direct rebuttal to the README's "text is an echo of the fields" fear.

**Verdict: PASS.** Content lift positive across all 3 seeds, survives the presence bar
(flat) and the verbosity bar (+0.0086 beyond A+len), audit clean. The CONTENT of borrower
text carries genuine incremental signal.

**Honest caveats on the PASS — real, not large:**
- Effect is MODEST: +0.0099 PR-AUC, ~3.6% relative over A. The earlier "43× the bar"
  framing is DROPPED — the has_desc bar is statistically zero, so a ratio against it is
  meaningless. The real competing baseline was A+len (+0.0014), which content beat by
  +0.0086.
- The magnitude is itself data for PROJECT.md risk #2 (does text change the decision often
  enough to matter). Statistical PASS ≠ large effect; scope the project accordingly.
- The top-feature semantic story ('stress vocabulary', 'tone words') is interpretation,
  not a finding. The audit passed on its actual criterion — no residue, no leakage.
  Direction of each token unverified (gain shows use, not sign).

---

## Final call

**GO — build the project, framed honestly for what it is.** Gate zero passed: borrower
text carries real, seed-stable signal beyond tabular fields, beyond desc presence, and
beyond verbosity, with no template/leakage artifacts. The signal is genuine but modest
(+0.0099 PR-AUC), which pre-answers part of PROJECT.md risk #2 — measure "how often does
text change the decision" early, because a lift this size predicts it moves the margin,
not the mass.

**First day-one risk to retest at scale:** this entire finding lives on the desc-only
population (the 47% who wrote text) in 2007–2013. LendingClub STOPPED collecting `desc` in
2015. So the signal is not only vintage-bounded but built on an input feature that no
longer exists on new data — the "at scale" version of this test is not retestable on
post-2015 loans because the column is gone. That bounds what the full project can honestly
claim: it is a methods demonstration on a historical corpus (legitimate for the fintech
portfolio target), NOT a deployable live system. Build it as the former, say so plainly,
and the desc-only / stopped-collecting scope becomes a stated limitation rather than a
discovered embarrassment.

---
---

# BUILD PHASE — Reconciliation Agent (post-gate)

Gate passed, so PROJECT.md is un-shelved and the agentic layer is being built. Same honest
record, continued. Logged after each verified result.

## Build 0 — Architecture: what was cut and why — COMPLETE

**Question:** A "make it a true multi-agent system" proposal (5 agents, 8 NLP sub-tasks,
narrative-risk probability, evidence-retrieval loop) was on the table. Adopt it?

**Decision:** Partially. Kept the good additions; cut everything unfalsifiable. Final
design is **4 nodes**, not 5 inflated agents:
tabular_score → text_stance → reconciler → explanation.

Adopted from the proposal:
- Explanation as a separate node (clean split of decide vs. write-memo).
- Business metrics for eval (% decisions changed by text, human-review rate, calibration,
  FP/FN) alongside trajectory/faithfulness.
- Specify the reconciler logic explicitly (don't leave it vague).

Rejected — and why (this is the discipline that separates this from theatre):
- **8-task Narrative Agent** (fraud, contradiction, employment-consistency): LendingClub
  has NO ground truth for these. Unvalidatable outputs = vibes with a label. Do the one
  task the feasibility test validated: corroborate / mitigate / neutral vs tabular risk.
- **"Narrative Risk = 0.42, confidence 0.84":** an LLM asked to rate risk 0–1 emits an
  uncalibrated confabulation. The XGBoost probability is calibrated against realized
  default; do not weigh a real number against a made-up one.
- **Evidence-retrieval loop** ("retrieve more → re-run reconciliation"): a static
  application has no additional borrower evidence to fetch. Motion for the appearance of
  agency. Replaced with: low confidence → human review.
- **Calling every node an "agent":** a deterministic XGBoost scorer is not an agent.
  Reserve the claim for the reconciler's routing, which is the one place with real agency.
- **Tech maximalism** (LangSmith + BGE + spaCy + FAISS + reranker): each is a dependency
  to explain and a way to ship nothing. v1 stays minimal.

## Build 1 — Core invariants — COMPLETE

**Decision:** Three non-negotiables baked into the code, not just intended:
1. **Independence:** text_stance NEVER reads p_default. It forms its narrative read from
   desc + retrieved policy only. The reconciler is the FIRST node to see both channels.
   This is the agentic analog of the content-vs-selection separation that made Finding 4
   honest — if the text channel sees the score first, "agreement" is meaningless.
2. **Reconciler decides by RULES** over (stance, confidence), not a learned meta-classifier
   (no training data for the meta-decision) and not an LLM that invents the decision. An
   LLM may WRITE the disagreement memo; it does not MAKE the call.
3. **Retrieval is a tool** the stance node calls, not a separate "agent".

## Build 2 — Node implementation — COMPLETE (uncommitted)

In `agent/reconciler_agent.py`. Graph/state/routing unchanged from the agreed skeleton.

- **_score_tabular:** loads models/model_a.pkl; RE-DERIVES Model A's training-time
  categorical codes from feasibility_frame.pkl before encoding. Why: XGBoost categorical
  splits are code-based; an independent `.astype("category")` per call assigns codes by
  whichever strings appear in that call, so the same string could map to a different code
  than training used — silently corrupting every categorical split. Returns (p_default,
  top-5 SHAP drivers via TreeExplainer). Tested on a real sample — works.
- **_retrieve_policy:** 19-chunk synthetic underwriting corpus (DTI limits, hardship/
  medical, employment verification, fraud flags, purpose-specific), BM25 via rank_bm25,
  top-k=4. Tested — sensible matches.
- **_call_llm_json:** provider-agnostic LLMClient Protocol + configure_llm_client()
  injection (no vendor committed yet), strict JSON parse with code-fence tolerance, falls
  back to stance="neutral"/confidence=0.0 on any parse/validation failure or unconfigured
  client. Tested: good / fenced / broken / invalid-stance + unconfigured-error path.

## Build 3 — Tests — COMPLETE (uncommitted)

- **Routing smoke test** (test_reconciler_routing.py): reconciler → _route →
  auto_decision/human_review → explanation with stubbed states. 5/5:
  agree-risky→auto_decline, agree-safe→auto_approve, both disagree directions→human_review,
  low-confidence→human_review.
- **stance→reconciler seam** (test_stance_reconciler_seam.py): real text_stance (real BM25,
  real JSON parse) through real reconciler, stub LLMClient. 4/4:
  mitigates+0.85→disagree, corroborates+0.20→disagree, corroborates+0.85→agree,
  unparseable→neutral→low_conf.
- **Full graph invoke** (test_full_graph_invoke.py): one real application through
  build_graph().invoke(); real _score_tabular against models/model_a.pkl (p_default=0.1700,
  correct SHAP drivers), real _retrieve_policy, stub LLM. Final state threaded p_default,
  stance=mitigates_risk, route=agree, decision=auto_approve, non-empty memo end-to-end.

**Note carried forward:** the `disagree` route fires only in two corners (risky+mitigates,
safe+corroborates). Most applications won't hit them — watch the DISAGREE RATE in the eval.
A near-zero rate is PROJECT.md risk #2 surfacing (text rarely changes the decision), and it
is a finding to REPORT, not a bug to tune away.

## Build 4 — Tabular tuning + calibration test (measured, not assumed) — COMPLETE

**Question:** Does tuning Model A's hyperparameters move test performance, and does adding
an isotonic calibration layer improve probability quality?

**Split discipline:** notebooks/tabular_tuning.ipynb carved train_inner (~70%) / val (~15%)
/ calib (~15%) out of the locked TRAIN indices only (stratified, seed 42). Categories
re-derived once from the full feasibility_frame.pkl, never re-cast per split. The locked
TEST set was read in exactly one cell — the last one — with X_test/y_test undefined
everywhere above it (a real NameError guard, not a comment). 40-trial Optuna search
(TPE, seed 42) optimized val PR-AUC; best params frozen to config/tabular_best_params.json.

**Result (locked TEST, read once):**
- model_a (original): PR-AUC 0.2769 / ROC-AUC 0.6912.
- tuned: PR-AUC 0.2803 / ROC-AUC 0.6920. **Delta +0.0034** — small, as expected; tuning a
  13-feature tabular model has limited headroom.
- Isotonic calibration, fit on the ~13k-row calib slice, tested on locked TEST: Brier
  0.1213 → 0.1216, ECE 0.0029 → 0.0046. **Both got slightly worse.** Raw XGBoost was
  already well-calibrated out of the box (ECE 0.0029 is tight); isotonic on a calib slice
  this size added noise rather than correcting bias.

**Decision:** Remove the isotonic layer from scripts/train_final.py. models/
model_a_tuned_calibrated.pkl now holds `{model, features, categories, config}` only — no
calibrator key. Filename kept as-is (avoids touching the agent's load path later); the
calib split is still carved out for reproducibility of train_inner/val, just unused.
Removed because it was measured to not help on held-out data, not because calibration is
assumed unnecessary in general — the honest-caveat discipline applies to convenient
findings too, not just inconvenient ones.

**Caveat carried forward:** a ~13k-row calib slice may simply be too small for isotonic to
generalize on a model this already well-calibrated. If calibration is revisited, a larger
calib slice or a parametric approach (Platt scaling) would be the next thing to try — not
logged as a retry, since none is planned.

## Build 5 — first --real eval (flawed, superseded) — COMPLETE

**Question:** With the reconciler agent wired to the tuned tabular model and qwen2.5:latest
(both pre-flight checks clean — zero policy-text-as-evidence violations, textbook-correct
route/stance cross-tab on the 40-loan validate set) — what does the agent do on the full
2,000-loan locked-test sample?

**Raw numbers, as run:**
- Disagree rate: 5.10% (102/2000). Route split: agree 26.15% / disagree 5.10% / low_conf
  68.75%.
- Review rate: 73.85% (1,477/2,000) vs a ~15% assumed budget — ~5x over.
- Decisions changed by text: 1,474 tabular-approve → deferred; only 3 tabular-decline →
  deferred.
- Realized default: approve_but_flagged 14.71% (n=102) vs clean_approve 15.49% (n=523) —
  expected inequality (flagged > clean) did NOT hold. clean_decline / decline_but_mitigated:
  both n=0 (underpowered, tabular declined almost nobody).
- Calibration lift: −0.0018 (auto-decided subset very slightly *worse* calibrated than the
  full sample, not better).
- Fairness slice: approval rate shifted ~72–78pp downward, uniformly, across every
  home_ownership and verification_status group (tabular ~100% approve → agent ~21–29%
  approve in every group alike).

**DIAGNOSIS — why these numbers don't mean what they look like they mean:**

1. **73.85% "review rate" is mostly model SILENCE, not model DISAGREEMENT.** The old
   `_route()` sent both `disagree` (102 loans, genuine tabular-vs-text conflict) AND
   `low_conf` (1,375 loans — neutral stance or confidence < 0.55) to human_review. A neutral/
   low-confidence stance means the text channel had nothing to say, not that it pushed back
   on the tabular read. Routing silence to review inflates the review-rate number without
   reflecting any real disagreement — this is a routing-policy artifact, not an agent finding.
2. **The 0.50 threshold barely ever fires at a 15.5% base rate.** Tabular-alone baseline: FP=2,
   FN=304 out of 2,000 — almost nobody gets auto-declined, because a calibrated model trained
   on ~15% positives rarely pushes any single row's probability past 0.50. That's WHY
   `clean_decline`/`decline_but_mitigated` came back n=0 (underpowered) — the decline side of
   every comparison was structurally empty before the agent did anything. 0.50 was never
   tuned for this base rate; it was Model A's convenience default, carried forward unexamined.
3. **The n=102 `approve_but_flagged` vs `clean_approve` gap (14.71% vs 15.49%, 0.78pt) is well
   inside the ~±7pt 95% CI on a group this size** (Wilson: approve_but_flagged ≈
   [9.1%, 22.9%], clean_approve ≈ [12.6%, 18.8%] — substantially overlapping). "Does not hold"
   was the correct mechanical read of the point estimates, but the honest statistical read is
   **inconclusive at this n**, not a failed hypothesis. Reporting it as a flat "does not hold"
   without the interval was itself a small honesty gap in Build 5's report — flagged here so
   it isn't repeated.
4. **The fairness "shift" (~72–78pp, uniform across every group) is the review-rate artifact
   wearing a different hat.** Since ~74% of loans got deferred to human_review regardless of
   group, "agent_approve rate" cratered relative to "tabular_approve rate" identically
   everywhere — a mechanical consequence of (1) and (2), not a group-specific finding. Uniform
   magnitude across unrelated proxy groups is itself evidence it's a routing artifact, not a
   fairness effect.

**The finding that SURVIVES this diagnosis, and is real:** 68.75% of stances came back
`neutral` or below the confidence threshold. **The stance channel is low-signal on this
population** — most 2007–2013 `desc` text is short, boilerplate, or genuinely uninformative
about risk, and the agent (correctly) declines to manufacture a confident read out of it. This
is PROJECT.md risk #2 landing for real, not a bug: text rarely gives the agent something
worth acting on, and a routing policy that defers on silence should not be confused with a
routing policy that defers on disagreement. That distinction is exactly what Build 5's flaw
(1) collapsed, and what Fix 2 (next) restores.

**Decision:** Do not delete or overwrite this entry — it's the honest record of catching a
routing-policy bug and a missing-CI reporting gap before either shaped a real conclusion. Two
fixes follow (Build 6): (a) pick HIGH_RISK on the VAL slice instead of hardcoding 0.50, (b)
change `_route()` so low_conf defers to the tabular decision and only genuine `disagree`
triggers human review.

## Build 6 — corrected eval — COMPLETE

**Fix 1 (threshold, leakage-safe):** scripts/compute_decision_threshold.py picked HIGH_RISK
on the VAL slice of the TRAIN split only (train_inner/val/calib carved from
split_indices.pkl's train_idx, same split as notebooks/tabular_tuning.ipynb — TEST never
read). F1-max for the default class on 12,969 val rows (15.27% default rate): **threshold =
0.1703**, val F1 = 0.358, val decline rate = 35.4%. Non-degenerate, no Youden's J fallback
needed. Frozen to config/decision_threshold.json; `reconciler_agent.HIGH_RISK` now reads it
at import time instead of a hardcoded 0.50.

**Fix 2 (routing, scoped edit):** `_route()` changed from `agree→auto_decision,
else→human_review` to `disagree→human_review, else (agree OR low_conf)→auto_decision`.
low_conf now takes the tabular decision instead of triggering review — silence defers to the
calibrated model, only a confident, opposing stance earns a review. `reconciler()`'s own
stance/confidence logic, the graph, and the state schema are unchanged; verified via the same
11 existing tests (2 required fixture updates — one for the new HIGH_RISK boundary moving a
hardcoded p_default from "safe" to "risky", one for the new low_conf→auto_decline behavior —
both documented inline in the test files; test_full_graph_invoke.py needed no fixture change,
its real application's p_default naturally crossed the new threshold and the test's
assertions are membership-based, not exact-route).

**Re-run discipline:** scripts/eval_agent.py --real re-run from results/eval_stance_cache.pkl
— **zero new LLM calls** (2,000/2,000 cache hits; stances don't depend on threshold or
routing, only p_default/route/decision are recomputed). Caught and fixed a second bug while
reviewing the first pass: section 4 (calibration/FP-FN/review-rate) was still deriving
"deferred" from `route != "agree"`, which was correct under the OLD routing but silently
wrong under the NEW one (low_conf is no longer deferred). Fixed to derive from
`agent_decision == "human_review"` directly — the review-rate number now matches section 2's
"decisions changed" exactly (12.20% both), which is itself the check that the bug is gone.

**Report (n=2,000, same locked-TEST sample, seed 42):**

1. **stance_source:** 2,000/2,000 parsed. Zero api_error, zero parse_error, zero empty —
   clean run, cache reuse didn't introduce any staleness.
2. **Route:** agree 19.05% (381) / disagree **12.20%** (244) / low_conf 68.75% (1,375).
   Disagree rate is now real (was 5.10% under the contaminated Build 5 routing).
3. **Decisions changed by text:** 244 (12.20%) — 57 approve→deferred (2.85%), 187
   decline→deferred (9.35%). Matches the disagree rate exactly, as it must now that only
   `disagree` triggers review.
4. **Realized default, WITH Wilson 95% CIs (z=1.96):**
   | bucket | n | rate | 95% CI |
   |---|---|---|---|
   | clean_approve | 336 | 10.12% | [7.33%, 13.81%] |
   | approve_but_flagged | 57 | 8.77% | [3.81%, 18.95%] |
   | clean_decline | 45 | 22.22% | [12.54%, 36.27%] |
   | decline_but_mitigated | 187 | 25.13% | [19.46%, 31.81%] |

   Decline-side buckets are non-empty for the first time (real threshold). Both directional
   comparisons: **INCONCLUSIVE at this n** — CIs overlap substantially in both cases. This is
   the honest result, not a downgrade of Build 5's flaw (3): even with the threshold fixed, a
   sample this size doesn't have the power to resolve whether disagreement tracks realized
   risk, given the effect (if any) is plausibly on the same modest scale the feasibility phase
   found (~0.01 PR-AUC).
5. **Calibration + FP/FN + review budget:** Brier full 0.1241, auto-decided subset (n=1,756)
   0.1180, **calibration lift +0.0062** (small, positive, real this time — computed on the
   correct automated population). Tabular-alone baseline FP=561 (28.05%) / FN=131 (6.55%).
   Deferral catches these asymmetrically: **25.0% of FPs deferred (140/561) vs only 3.8% of
   FNs (5/131)** — disagreement-based review disproportionately catches bad-decline cases, not
   bad-approve cases. Remaining automated errors: FP=421, FN=126. **Review rate 12.20% vs
   ~15% budget — within budget** (was reported as 5x over in Build 5; that was entirely the
   routing bug).
6. **Fairness slice:** shifts collapsed from Build 5's ~72–78pp to **−2.2pp to −3.9pp** across
   every home_ownership/verification_status group — none exceed the 5pp flag threshold.
   Confirms Build 5 diagnosis (4): it really was the review-rate artifact, not a fairness
   effect.
7. **Underpowered flags:** only `home_ownership=OTHER` (n=2). No realized-rate bucket is
   underpowered anymore.

**Honest verdict:** The engineering problems are fixed and verifiably so — review rate now
matches its own definition, sits within budget, calibration lift is positive, the fairness
artifact is gone, decline-side buckets exist. But the **scientific question is still open,
not resolved in the agent's favor**: whether a confident text disagreement actually predicts
realized default better than tabular alone is INCONCLUSIVE at n=2,000 in both directions
tested. Combined with 68.75% of stances landing neutral/low-confidence, this is consistent
with — not a reversal of — the standing finding: the stance channel is low-signal on this
population (PROJECT.md risk #2). What changed is that the eval now measures this honestly
instead of through a routing bug that made the whole system look both more active and more
harmful than it is. More data (a larger TEST-adjacent eval population, or accepting a wider
CI) is the only way to move the two INCONCLUSIVE comparisons off dead center — not logged as
a planned next step, since none is committed yet.

## Pending

- Commit discipline: requirements.txt bump (langgraph, shap, rank_bm25, optuna,
  python-dotenv, requests — PINNED) in the SAME commit as the code that needs it.
