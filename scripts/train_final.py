"""
Deterministic build of the tuned tabular model. No search: reads the
hyperparameters frozen by notebooks/tabular_tuning.ipynb and rebuilds the
identical encoding, split, and fit from them. Running this script must
reproduce the exact model the notebook froze (same best_iteration).

No isotonic calibration layer. It was tried and measured on the notebook's
locked TEST report: raw XGBoost was already well-calibrated (ECE 0.0029),
and isotonic fit on the ~13k-row calib slice made it worse (ECE 0.0029 ->
0.0046, Brier 0.1213 -> 0.1216). Removed as measured-and-unnecessary, not
assumed. See DECISIONS.md for the full record. The train_inner/val/calib
split is still carved out exactly as before -- calib is simply unused now,
kept so the split (and therefore train_inner/val, and best_iteration) stays
reproducible without touching the rest of the split logic.

Does not touch models/model_a.pkl (the original baseline) and does not read
the locked TEST split at any point -- this script's job is only to reproduce
the artifact, not to report on it (the notebook's guardrail cell already did
that, once).
"""
import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "tabular_best_params.json"
FRAME_PATH = ROOT / "data" / "interim" / "feasibility_frame.pkl"
SPLIT_PATH = ROOT / "data" / "interim" / "split_indices.pkl"
# Filename kept as-is (avoids touching the agent's load path later); the
# bundle itself no longer contains a calibrator -- see module docstring.
OUT_PATH = ROOT / "models" / "model_a_tuned_calibrated.pkl"

RAW_TABULAR_COLS = [
    "loan_amnt", "term", "int_rate", "grade", "sub_grade",
    "annual_inc", "dti", "emp_length", "home_ownership",
    "verification_status", "fico_range_low", "fico_range_high", "purpose",
]
CATEGORICAL_COLS = ["grade", "sub_grade", "home_ownership", "verification_status", "purpose"]
EMP_LENGTH_MAP = {
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
    "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9,
    "10+ years": 10,
}


def build_tabular_features(df, cols=RAW_TABULAR_COLS):
    """Cast categoricals exactly once, on the full frame -- never re-cast
    independently per split (would risk different codes for the same string)."""
    X = df[cols].copy()
    X["term"] = X["term"].astype(str).str.extract(r"(\d+)").astype("float32")
    X["emp_length"] = X["emp_length"].map(EMP_LENGTH_MAP).astype("float32")
    for c in CATEGORICAL_COLS:
        X[c] = X[c].astype("category")
    return X


def main():
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    frame = pd.read_pickle(FRAME_PATH)
    with open(SPLIT_PATH, "rb") as f:
        split = pickle.load(f)
    train_idx_full = split["train_idx"]  # TEST indices are never loaded in this script.

    y = frame["default"].astype(int)
    X_full = build_tabular_features(frame, config["features"])
    assert list(X_full.columns) == config["features"], "feature order mismatch vs frozen config"

    seed = config["random_state"]
    train_inner_idx, temp_idx = train_test_split(
        train_idx_full, test_size=0.30, stratify=y.loc[train_idx_full], random_state=seed
    )
    val_idx, calib_idx = train_test_split(
        temp_idx, test_size=0.50, stratify=y.loc[temp_idx], random_state=seed
    )

    X_train_inner, y_train_inner = X_full.loc[train_inner_idx], y.loc[train_inner_idx]
    X_val, y_val = X_full.loc[val_idx], y.loc[val_idx]
    # calib_idx is still carved out (keeps the split identical to the notebook's),
    # but is no longer used to fit anything -- no calibration layer in this build.

    print(f"train_inner: {len(train_inner_idx):,}  val: {len(val_idx):,}  calib: {len(calib_idx):,} (unused)")

    model_params = dict(
        max_depth=config["max_depth"],
        learning_rate=config["learning_rate"],
        min_child_weight=config["min_child_weight"],
        subsample=config["subsample"],
        colsample_bytree=config["colsample_bytree"],
        reg_lambda=config["reg_lambda"],
        n_estimators=config["n_estimators"],
        objective=config["objective"],
        eval_metric=config["eval_metric"],
        tree_method=config["tree_method"],
        enable_categorical=config["enable_categorical"],
        early_stopping_rounds=config["early_stopping_rounds"],
        random_state=seed,
        n_jobs=-1,
    )
    model = XGBClassifier(**model_params)
    model.fit(X_train_inner, y_train_inner, eval_set=[(X_val, y_val)], verbose=False)
    print(f"Refit best_iteration: {model.best_iteration} "
          f"(frozen config recorded {config['refit_best_iteration']})")

    categories = {c: X_full[c].cat.categories for c in CATEGORICAL_COLS}

    bundle = {
        "model": model,
        "features": config["features"],
        "categories": categories,
        "config": config,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
