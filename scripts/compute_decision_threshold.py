"""
Picks the decision threshold (HIGH_RISK) the reconciler agent's auto_decision
node uses, on the VALIDATION slice only -- never train_inner (used to fit the
tuned model) and never TEST (locked, reserved for evaluation). Reuses the
exact train_inner/val/calib split notebooks/tabular_tuning.ipynb and
scripts/train_final.py carve out of split_indices.pkl's TRAIN indices, so
"val" here is the same rows that never touched model fitting or early
stopping.

Maximizes F1 for the default (positive) class across threshold candidates
from the val precision-recall curve. Falls back to Youden's J (max TPR-FPR)
if the F1-argmax is degenerate (0% or 100% decline rate -- can happen at a
low base rate where F1 is maximized at a trivial threshold).

Writes config/decision_threshold.json. Never reads split_indices.pkl's
test_idx.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
FRAME_PATH = ROOT / "data" / "interim" / "feasibility_frame.pkl"
SPLIT_PATH = ROOT / "data" / "interim" / "split_indices.pkl"
MODEL_PATH = ROOT / "models" / "model_a_tuned_calibrated.pkl"
OUT_PATH = ROOT / "config" / "decision_threshold.json"

SEED = 42
CATEGORICAL_COLS = ["grade", "sub_grade", "home_ownership", "verification_status", "purpose"]
EMP_LENGTH_MAP = {
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
    "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9,
    "10+ years": 10,
}


def build_tabular_features(df, cols):
    X = df[cols].copy()
    X["term"] = X["term"].astype(str).str.extract(r"(\d+)").astype("float32")
    X["emp_length"] = X["emp_length"].map(EMP_LENGTH_MAP).astype("float32")
    for c in CATEGORICAL_COLS:
        X[c] = X[c].astype("category")
    return X


def main():
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    model, feature_order, categories = bundle["model"], bundle["features"], bundle["categories"]

    frame = pd.read_pickle(FRAME_PATH)
    with open(SPLIT_PATH, "rb") as f:
        split = pickle.load(f)
    train_idx_full = split["train_idx"]  # TEST indices are never read in this script.

    y = frame["default"].astype(int)
    X_full = build_tabular_features(frame, feature_order)
    for c in CATEGORICAL_COLS:
        X_full[c] = pd.Categorical(X_full[c], categories=categories[c])

    train_inner_idx, temp_idx = train_test_split(
        train_idx_full, test_size=0.30, stratify=y.loc[train_idx_full], random_state=SEED
    )
    val_idx, calib_idx = train_test_split(
        temp_idx, test_size=0.50, stratify=y.loc[temp_idx], random_state=SEED
    )

    X_val, y_val = X_full.loc[val_idx], y.loc[val_idx]
    p_val = model.predict_proba(X_val)[:, 1]

    precision, recall, pr_thresholds = precision_recall_curve(y_val, p_val)
    f1s = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    f1s_for_thresholds = f1s[:-1]  # precision_recall_curve returns len(thresholds) == len(precision)-1
    best_idx = int(np.argmax(f1s_for_thresholds))
    threshold = float(pr_thresholds[best_idx])
    best_f1 = float(f1s_for_thresholds[best_idx])
    decline_rate = float((p_val >= threshold).mean())
    method = "f1_max"

    degenerate = decline_rate <= 0.0 or decline_rate >= 1.0
    if degenerate:
        fpr, tpr, roc_thresholds = roc_curve(y_val, p_val)
        j = tpr - fpr
        best_idx2 = int(np.argmax(j))
        threshold = float(roc_thresholds[best_idx2])
        decline_rate = float((p_val >= threshold).mean())
        best_f1 = float(f1_score(y_val, (p_val >= threshold).astype(int)))
        method = "youden_j"

    config = {
        "threshold": threshold,
        "method": method,
        "val_f1_at_threshold": best_f1,
        "val_decline_rate": decline_rate,
        "val_rows": int(len(val_idx)),
        "val_default_rate": float(y_val.mean()),
        "random_state": SEED,
        "note": (
            "Selected on the VAL slice of the TRAIN split only (train_inner/val/calib "
            "carved from split_indices.pkl's train_idx, same split as "
            "notebooks/tabular_tuning.ipynb / scripts/train_final.py). "
            "split_indices.pkl's test_idx was never read by this script."
        ),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(config, f, indent=2)

    print(f"val rows: {len(val_idx)}  val default rate: {y_val.mean():.4f}")
    print(f"method: {method}")
    print(f"threshold: {threshold:.4f}")
    print(f"F1 at threshold: {best_f1:.4f}")
    print(f"decline rate at threshold: {decline_rate:.4f}")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
