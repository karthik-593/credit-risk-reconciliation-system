"""
Selection-bias control: does the mere presence of a `desc` field
(has_desc) carry predictive signal on its own? Runs on the FULL
2007-2013 resolved population (not the desc-only frame, where
has_desc would be constant) so the flag can actually vary. This
establishes the baseline that any text-content model must beat.
"""
import importlib.util
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_model_a = _load_module("train_model_a", "scripts/04_train_model_a.py")
build_tabular_features = _model_a.build_tabular_features
make_model = _model_a.make_model

SUBSET_PATH = "data/interim/subset_cache.pkl"
RANDOM_STATE = 42

if __name__ == "__main__":
    df = pd.read_pickle(SUBSET_PATH)

    if "issue_year" not in df.columns:
        df["issue_year"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce").dt.year

    mask_years = df["issue_year"].between(2007, 2013)
    mask_status = df["loan_status"].isin(["Fully Paid", "Charged Off"])
    df = df.loc[mask_years & mask_status].copy()
    df["default"] = (df["loan_status"] == "Charged Off").astype(int)

    desc_stripped = df["desc"].str.strip()
    df["has_desc"] = (desc_stripped.notna() & (desc_stripped != "")).astype(int)

    X_tabular = build_tabular_features(df)
    X_tabular_plus = X_tabular.copy()
    X_tabular_plus["has_desc"] = df["has_desc"].astype("float32")
    y = df["default"].astype(int)

    idx_train, idx_test = train_test_split(
        df.index, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    y_train, y_test = y.loc[idx_train], y.loc[idx_test]

    results = {}
    for name, X in [("A-full", X_tabular), ("B-full", X_tabular_plus)]:
        m = make_model()
        m.fit(X.loc[idx_train], y_train)
        proba = m.predict_proba(X.loc[idx_test])[:, 1]
        results[name] = {
            "pr_auc": average_precision_score(y_test, proba),
            "roc_auc": roc_auc_score(y_test, proba),
            "model": m,
        }

    print(f"A-full: PR-AUC={results['A-full']['pr_auc']:.4f}  ROC-AUC={results['A-full']['roc_auc']:.4f}")
    print(f"B-full: PR-AUC={results['B-full']['pr_auc']:.4f}  ROC-AUC={results['B-full']['roc_auc']:.4f}")

    delta = results["B-full"]["pr_auc"] - results["A-full"]["pr_auc"]
    print(f"\nSelection-effect bar (B-full - A-full) PR-AUC: {delta:+.4f}")

    gains = results["B-full"]["model"].get_booster().get_score(importance_type="gain")
    ranked = sorted(gains.items(), key=lambda x: x[1], reverse=True)
    rank = [f for f, _ in ranked].index("has_desc") + 1
    print(f"has_desc rank: {rank} of {len(ranked)} features")
