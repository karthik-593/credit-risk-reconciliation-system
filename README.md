# Credit Risk Reconciliation System

Does a borrower's free-text loan description carry credit-risk signal beyond
what's already captured by traditional underwriting features (grade, rate,
income, DTI, FICO, ...)? This project reconciles two views of the same
loan — the tabular underwriting view and the borrower's own narrative — and
measures whether the text adds anything real, or whether any apparent lift
is just a selection artifact (who bothers to write a description, or how
much they write).

Built on the LendingClub accepted-loans dataset (2007–2018, ~2.2M loans, 151
columns), using the `desc` free-text field present on loans originated
2007–2013.

## Approach

The core risk in text-augmented credit models is mistaking a **selection
effect** for genuine content signal — e.g. borrowers who bother writing a
description might just differ in risk, independent of what they wrote. This
project isolates that explicitly, in stages:

1. **Feasibility check** — confirm the `desc` field is actually usable
   (fill rate and length by vintage year) and only proceed on populations
   where it is.
2. **Leakage-safe tabular baseline (Model A)** — trained only on
   origination-time features. No payment history, balances, recoveries, or
   status-derived fields.
3. **Selection-bias control (has_desc)** — on the full resolved population
   (not just loans with text), test whether merely *having* a description
   predicts default, independent of its content.
4. **Length control (A+len)** — test whether word count alone (verbosity)
   explains any lift, before crediting the model for "understanding" text.
5. **Content model (C)** — TF-IDF over the cleaned description, added to
   the same tabular features, scored against A and A+len on the identical
   rows and train/test split so every delta is attributable to the feature
   set alone.
6. **Stability check** — steps 4–5 repeated across 3 random seeds
   (reshuffled splits) to confirm deltas aren't an artifact of one split.

## Key findings

| Comparison | Mean ΔPR-AUC (3 seeds) |
|---|---|
| has_desc selection bar (presence alone) | −0.0002 |
| Length channel (word count alone) | +0.0014 |
| Content lift (TF-IDF vs tabular-only) | +0.0099 |
| Content beyond length (TF-IDF vs tabular+word count) | +0.0086 |

Presence of a description carries no standalone signal (has_desc ranks
13th of 14 features by gain in its model). Word count buys almost nothing.
The TF-IDF content lift survives both controls, holds sign across all three
seeds, and the top contributing terms read as genuine borrower language
(financial purpose, tone, stress vocabulary) rather than template artifacts
from the data collection wrapper — so the lift is judged real, not noise.

## Repository structure

```
scripts/
  01_explore_header.py          # schema check on the raw file, no full load
  02_build_subset.py             # column-limited load + desc fill-rate/maturity checks
  03_build_feasibility_frame.py  # fixed modeling population + desc cleaning
  04_train_model_a.py            # tabular-only baseline, saves train/test split
  05_selection_bias_control.py   # has_desc control on the full resolved population
  06_train_text_models.py        # A vs A+len vs C, 3-seed stability check
data/                             # not tracked; see Setup
models/                           # not tracked; produced by the scripts
```

## Setup

```bash
pip install -r requirements.txt
```

Download the LendingClub accepted-loans file (`accepted_2007_to_2018Q4.csv.gz`)
and place it at `data/raw/accepted_2007_to_2018Q4.csv.gz`, then run the
scripts in order from the repository root:

```bash
python scripts/01_explore_header.py
python scripts/02_build_subset.py
python scripts/03_build_feasibility_frame.py
python scripts/04_train_model_a.py
python scripts/05_selection_bias_control.py
python scripts/06_train_text_models.py
```

Each script reads/writes intermediate artifacts under `data/interim/` and
`models/` so the pipeline can be resumed from any stage.
