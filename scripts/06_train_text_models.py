"""
Isolates the incremental value of borrower free-text (`desc`) beyond
tabular features, controlling for two selection effects:

  - has_desc (Script 05): does merely *having* a description help?
  - word count (A+len, here): does verbosity alone explain the lift?

Three models per random seed, all scored on the identical rows:
  A     - 13 tabular features only
  A+len - 13 tabular features + word count of cleaned desc
  C     - 13 tabular features + TF-IDF over cleaned desc

Seed 42 reuses the exact split saved by Script 04 so results align
with Model A's reported numbers; seeds 43/44 reshuffle the split on
the same population to check stability of the deltas.
"""
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer

import importlib.util


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_model_a = _load_module("train_model_a", "scripts/04_train_model_a.py")
build_tabular_features = _model_a.build_tabular_features
make_model = _model_a.make_model
RAW_TABULAR_COLS = _model_a.RAW_TABULAR_COLS

FRAME_PATH = "data/interim/feasibility_frame.pkl"
SPLIT_PATH = "data/interim/split_indices.pkl"
SEEDS = [42, 43, 44]
CATEGORICAL_COLS = ["grade", "sub_grade", "home_ownership", "verification_status", "purpose"]


def run():
    df = pd.read_pickle(FRAME_PATH)
    y = df["default"].astype(int)
    word_count = df["desc_clean"].str.split().str.len().astype("float32")

    with open(SPLIT_PATH, "rb") as f:
        saved_split = pickle.load(f)

    results = {"A": [], "A+len": [], "C": []}
    seed42_gains = None

    for seed in SEEDS:
        if seed == 42:
            idx_train, idx_test = saved_split["train_idx"], saved_split["test_idx"]
        else:
            idx_train, idx_test = train_test_split(
                df.index, test_size=0.2, stratify=y, random_state=seed
            )
        y_train, y_test = y.loc[idx_train], y.loc[idx_test]

        X_tab = build_tabular_features(df)

        # Model A
        m = make_model(seed)
        m.fit(X_tab.loc[idx_train], y_train)
        proba = m.predict_proba(X_tab.loc[idx_test])[:, 1]
        results["A"].append((average_precision_score(y_test, proba), roc_auc_score(y_test, proba)))

        # Model A+len
        X_len = X_tab.copy()
        X_len["desc_word_count"] = word_count
        m = make_model(seed)
        m.fit(X_len.loc[idx_train], y_train)
        proba = m.predict_proba(X_len.loc[idx_test])[:, 1]
        results["A+len"].append((average_precision_score(y_test, proba), roc_auc_score(y_test, proba)))

        # Model C: TF-IDF fit on train only
        vec = TfidfVectorizer(max_features=1500, stop_words="english", min_df=5)
        tfidf_train = vec.fit_transform(df.loc[idx_train, "desc_clean"]).toarray().astype("float32")
        tfidf_test = vec.transform(df.loc[idx_test, "desc_clean"]).toarray().astype("float32")
        feat_names = [f"tfidf__{w}" for w in vec.get_feature_names_out()]

        Xtr_c = pd.concat([
            X_tab.loc[idx_train].reset_index(drop=True),
            pd.DataFrame(tfidf_train, columns=feat_names),
        ], axis=1)
        Xte_c = pd.concat([
            X_tab.loc[idx_test].reset_index(drop=True),
            pd.DataFrame(tfidf_test, columns=feat_names),
        ], axis=1)
        for c in CATEGORICAL_COLS:
            Xtr_c[c] = Xtr_c[c].astype("category")
            Xte_c[c] = Xte_c[c].astype("category")

        m = make_model(seed)
        m.fit(Xtr_c, y_train.reset_index(drop=True))
        proba = m.predict_proba(Xte_c)[:, 1]
        results["C"].append((average_precision_score(y_test, proba), roc_auc_score(y_test, proba)))

        if seed == 42:
            seed42_gains = m.get_booster().get_score(importance_type="gain")

    return results, seed42_gains


def summarize(results, seed42_gains):
    def stats(vals):
        arr = np.array(vals)
        return arr.mean(), arr.std()

    for name in ["A", "A+len", "C"]:
        pr_vals = [v[0] for v in results[name]]
        roc_vals = [v[1] for v in results[name]]
        pr_m, pr_s = stats(pr_vals)
        roc_m, roc_s = stats(roc_vals)
        print(f"{name:8s} PR-AUC:  {pr_m:.4f} +/- {pr_s:.4f}   per seed: {[round(x, 4) for x in pr_vals]}")
        print(f"{name:8s} ROC-AUC: {roc_m:.4f} +/- {roc_s:.4f}   per seed: {[round(x, 4) for x in roc_vals]}")

    delta_len = [results["A+len"][i][0] - results["A"][i][0] for i in range(len(SEEDS))]
    delta_content = [results["C"][i][0] - results["A"][i][0] for i in range(len(SEEDS))]
    delta_beyond_len = [results["C"][i][0] - results["A+len"][i][0] for i in range(len(SEEDS))]

    print(f"\nLength channel (A+len - A):     {np.mean(delta_len):+.4f} +/- {np.std(delta_len):.4f}")
    print(f"Content lift (C - A):           {np.mean(delta_content):+.4f} +/- {np.std(delta_content):.4f}")
    print(f"Content beyond length (C-A+len): {np.mean(delta_beyond_len):+.4f} +/- {np.std(delta_beyond_len):.4f}")

    tfidf_gains = {k: v for k, v in seed42_gains.items() if k.startswith("tfidf__")}
    top20 = sorted(tfidf_gains.items(), key=lambda x: x[1], reverse=True)[:20]
    print("\nTop 20 TF-IDF features by gain (seed 42):")
    for feat, gain in top20:
        print(f"  {feat.replace('tfidf__', ''):20s} gain={gain:8.2f}")


if __name__ == "__main__":
    results, seed42_gains = run()
    summarize(results, seed42_gains)
