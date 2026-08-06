"""
Model A: tabular-only default classifier. Uses only fields known at
loan origination (no post-origination leakage: no payment history,
balances, recoveries, or status-derived fields). Saves the train/test
split so later text-based models can be scored on the identical rows.
"""
import pickle
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score

FRAME_PATH = "data/interim/feasibility_frame.pkl"
SPLIT_PATH = "data/interim/split_indices.pkl"
MODEL_PATH = "models/model_a.pkl"
RANDOM_STATE = 42

RAW_TABULAR_COLS = [
    "loan_amnt", "term", "int_rate", "grade", "sub_grade",
    "annual_inc", "dti", "emp_length", "home_ownership",
    "verification_status", "fico_range_low", "fico_range_high", "purpose",
]

EMP_MAP = {
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
    "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9,
    "10+ years": 10,
}


def build_tabular_features(df, cols=RAW_TABULAR_COLS):
    X = df[cols].copy()
    X["term"] = X["term"].astype(str).str.extract(r"(\d+)").astype("float32")
    X["emp_length"] = X["emp_length"].map(EMP_MAP).astype("float32")
    for c in ["grade", "sub_grade", "home_ownership", "verification_status", "purpose"]:
        X[c] = X[c].astype("category")
    return X


def make_model(seed=RANDOM_STATE):
    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        enable_categorical=True,
        random_state=seed,
        n_jobs=-1,
    )


if __name__ == "__main__":
    df = pd.read_pickle(FRAME_PATH)

    X = build_tabular_features(df)
    y = df["default"].astype(int)

    print("=== Feature list (tabular-only, origination-time fields) ===")
    for i, c in enumerate(X.columns, 1):
        kind = "categorical" if str(X[c].dtype) == "category" else "numeric"
        print(f"{i:2d}. {c:22s} ({kind}, missing={X[c].isna().sum()})")

    idx_train, idx_test = train_test_split(
        df.index, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    with open(SPLIT_PATH, "wb") as f:
        pickle.dump({"train_idx": idx_train, "test_idx": idx_test, "random_state": RANDOM_STATE}, f)

    X_train, X_test = X.loc[idx_train], X.loc[idx_test]
    y_train, y_test = y.loc[idx_train], y.loc[idx_test]

    model = make_model()
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]

    pr_auc = average_precision_score(y_test, proba)
    roc_auc = roc_auc_score(y_test, proba)

    print(f"\nTest PR-AUC:  {pr_auc:.4f}  (base rate = {y_test.mean():.4f})")
    print(f"Test ROC-AUC: {roc_auc:.4f}")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "features": list(X.columns)}, f)
