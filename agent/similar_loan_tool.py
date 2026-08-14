"""
Similar-loan retrieval tool -- given a loan's origination features, finds
comparable historical TRAIN-set loans (bucket-then-distance: same grade
band, then nearest-neighbor on standardized numeric origination features)
and returns their observed default rate as tiebreaker evidence for CONFLICT
cases (tabular says risky, narrative says safer, or vice versa).

STANDALONE TOOL -- not wired into reconciler_agent.py, the graph, or eval.
Nothing here is imported BY reconciler_agent.py, and this module doesn't
call any of its node functions; it only reuses the tiny, stable pieces that
already define "the 13 origination features" and their encoding
(RAW_TABULAR_COLS from scripts/eval_agent.py, _EMP_LENGTH_MAP from
reconciler_agent.py) so this tool's notion of "origination feature" can
never silently drift from the tabular model's own.

THE LEAKAGE FENCE (non-negotiable, tested in test_similar_loan_tool.py
BEFORE anything here is trusted):
  - Row leakage: candidates are drawn ONLY from split_indices.pkl's
    train_idx. test_idx is never read as a candidate source by this module.
  - Column leakage: similarity distance uses ONLY 8 of the 13 origination
    features (DISTANCE_COLS below); grade buckets but never enters the
    distance; nothing post-origination (loan_status, default, issue_d, ...)
    is ever read for distance.
  - Outcome leakage: a neighbor's y (default) is read only because it's a
    TRAIN loan whose outcome is legitimately observed history. The QUERY
    loan's own y is never read by this tool -- callers pass features only.

SIMILARITY:
  1. Bucket candidates to the query's grade (widen to adjacent grades, via
     the natural A..G risk ordering, if the same-grade band is too small --
     mirrors real risk-segmentation policy, not a distance criterion).
  2. Standardize the 8 numeric features using TRAIN's mean/std (fit once,
     cached at first use -- test_idx never enters this fit).
  3. Within the band, keep only candidates within a DISTANCE CUTOFF: the
     95th percentile of "distance to one's own 50th-nearest in-band
     neighbor", computed once per band from TRAIN itself. That's what
     "genuine" neighbor means here -- close enough that a typical same-band
     loan would also consider it close. An unusual loan (sparse
     neighborhood) will have few or no candidates pass this and gets an
     honest insufficient-evidence answer, never a rate padded with
     dissimilar loans.
  4. Return up to K=50 of the closest genuine neighbors, closest first;
     default_rate is their mean y (fraction with default==1).

Cost note: index-building (standardization stats + per-band KDTrees +
cutoffs) runs once per process and is cached at module level -- one KDTree
50-NN self-query per grade band over TRAIN (86,460 rows across 7 bands),
a few seconds total, not a per-query cost.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from eval_agent import RAW_TABULAR_COLS  # noqa: E402 -- the 13 origination features, single source of truth

sys.path.insert(0, str(ROOT / "agent"))
import reconciler_agent as ra  # noqa: E402 -- reused ONLY for _EMP_LENGTH_MAP (ordinal encoding), read-only

FRAME_PATH = ROOT / "data" / "interim" / "feasibility_frame.pkl"
SPLIT_PATH = ROOT / "data" / "interim" / "split_indices.pkl"

# The 8 numeric features similarity distance is computed over -- a STRICT
# SUBSET of RAW_TABULAR_COLS. grade buckets (see _grade_band_for); sub_grade,
# home_ownership, verification_status, purpose are part of the 13
# origination features but are not used for distance at all.
DISTANCE_COLS = [
    "loan_amnt", "term", "int_rate", "annual_inc", "dti",
    "emp_length", "fico_range_low", "fico_range_high",
]

K = 50
MIN_NEIGHBORS = 25
CUTOFF_PERCENTILE = 95      # per-band distance cutoff = this percentile of TRAIN's own 50-NN self-distances
WIDEN_MIN_BAND = 200        # widen to adjacent grades if the same-grade band has fewer candidates than this

_GRADE_ORDER = ["A", "B", "C", "D", "E", "F", "G"]

_index: dict | None = None   # lazily built, cached -- see _build_index()/_get_index()


def _encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw origination columns -> the numeric encoding the distance metric
    uses: term as int months, emp_length as ordinal years (reusing
    reconciler_agent's own _EMP_LENGTH_MAP so this can't silently diverge
    from how the tabular model itself encodes the same field). Touches
    ONLY DISTANCE_COLS's underlying raw fields -- nothing post-origination."""
    out = pd.DataFrame(index=df.index)
    out["loan_amnt"] = df["loan_amnt"].astype("float64")
    out["term"] = df["term"].astype(str).str.extract(r"(\d+)")[0].astype("float64")
    out["int_rate"] = df["int_rate"].astype("float64")
    out["annual_inc"] = df["annual_inc"].astype("float64")
    out["dti"] = df["dti"].astype("float64")
    out["emp_length"] = df["emp_length"].map(ra._EMP_LENGTH_MAP).astype("float64")
    out["fico_range_low"] = df["fico_range_low"].astype("float64")
    out["fico_range_high"] = df["fico_range_high"].astype("float64")
    assert list(out.columns) == DISTANCE_COLS
    return out


def _build_index() -> dict:
    """Fit TRAIN-only standardization stats, per-grade-band KDTrees, and
    per-band distance cutoffs, once. NEVER reads test_idx as a candidate
    source (test_idx is not even loaded here beyond the split file itself)."""
    frame = pd.read_pickle(FRAME_PATH)
    with open(SPLIT_PATH, "rb") as f:
        split = pickle.load(f)
    train_idx = split["train_idx"]          # ONLY this -- test_idx is irrelevant to index-building
    train = frame.loc[train_idx]

    encoded = _encode_features(train)
    valid = encoded.dropna()                 # drops the ~3.9% with missing emp_length -- never imputed for candidates
    grades = train.loc[valid.index, "grade"].astype(str).to_numpy()
    y = train.loc[valid.index, "default"].astype(int).to_numpy()
    ids = valid.index.to_numpy()

    mean = valid.mean()
    std = valid.std().replace(0, 1.0)
    standardized = ((valid - mean) / std).to_numpy()

    band_data: dict[str, dict] = {}
    for g in sorted(set(grades)):
        mask = grades == g
        X = standardized[mask]
        if len(X) < 2:
            continue
        tree = cKDTree(X)
        k_query = min(K + 1, len(X))         # +1 -- a point's own nearest neighbor is itself, distance 0
        dists, _ = tree.query(X, k=k_query)
        self_knn_dist = dists[:, -1]         # distance to the farthest of each point's own up-to-50 neighbors
        cutoff = float(np.percentile(self_knn_dist, CUTOFF_PERCENTILE))
        band_data[g] = {
            "tree": tree,
            "ids": ids[mask],
            "y": y[mask],
            "cutoff": cutoff,
            "n": int(mask.sum()),
        }

    return {"mean": mean, "std": std, "bands": band_data}


def _get_index() -> dict:
    global _index
    if _index is None:
        _index = _build_index()
    return _index


def _grade_band_for(grade: str, bands: dict) -> list[str]:
    """Same grade if it already has enough candidates; otherwise widen
    outward (B -> A,B,C -> A,B,C,D, ...) via the natural A..G risk
    ordering -- adjacent grades are the closest real risk neighbors, so
    this mirrors actual risk-segmentation policy rather than any distance
    criterion."""
    if grade not in _GRADE_ORDER:
        return [grade]  # unknown grade string -- will simply find 0 candidates -> insufficient, not a crash
    idx = _GRADE_ORDER.index(grade)
    selected = {grade}
    total = bands.get(grade, {}).get("n", 0)
    radius = 1
    while total < WIDEN_MIN_BAND and radius <= len(_GRADE_ORDER):
        lo, hi = max(0, idx - radius), min(len(_GRADE_ORDER) - 1, idx + radius)
        widened = set(_GRADE_ORDER[lo:hi + 1])
        if widened == selected:
            break  # already covers the full grade range, can't widen further
        selected = widened
        total = sum(bands.get(g, {}).get("n", 0) for g in selected)
        radius += 1
    return sorted(selected, key=_GRADE_ORDER.index)


def _query_neighbors(features: dict) -> tuple[np.ndarray, np.ndarray, str]:
    """Returns (neighbor loan_ids, neighbor y values, grade_band_label) for
    up to K genuine (within-cutoff) neighbors, closest first. This is the
    ONLY place that touches the KDTree/candidate pool -- find_similar_loans()
    just summarizes this into the public contract. Exposed (not private-only
    in spirit) so tests can directly verify no returned loan_id is ever a
    TEST-set row."""
    idx = _get_index()
    bands = idx["bands"]

    grade = str(features.get("grade"))
    band_grades = _grade_band_for(grade, bands)
    grade_band_label = "+".join(band_grades)

    q_encoded = _encode_features(pd.DataFrame([features])).iloc[0]
    if q_encoded.isna().any():
        # A missing QUERY feature is imputed with TRAIN's mean (0 in
        # standardized space) so the tool still returns an answer for an
        # incomplete query -- documented, not silent. Candidates are NEVER
        # imputed this way (see _build_index's valid.dropna()).
        q_encoded = q_encoded.fillna(idx["mean"])
    q_std = ((q_encoded - idx["mean"]) / idx["std"]).to_numpy()

    all_dists, all_ids, all_ys = [], [], []
    for g in band_grades:
        b = bands.get(g)
        if b is None:
            continue
        d, i = b["tree"].query(q_std, k=min(K, b["n"]))
        d = np.atleast_1d(d)
        i = np.atleast_1d(i)
        keep = d <= b["cutoff"]
        all_dists.append(d[keep])
        all_ids.append(b["ids"][i[keep]])
        all_ys.append(b["y"][i[keep]])

    if not all_dists or sum(len(a) for a in all_dists) == 0:
        return np.array([], dtype=int), np.array([], dtype=int), grade_band_label

    dists = np.concatenate(all_dists)
    ids = np.concatenate(all_ids)
    ys = np.concatenate(all_ys)
    order = np.argsort(dists)[:K]
    return ids[order], ys[order], grade_band_label


def find_similar_loans(features: dict) -> dict:
    """features: a dict with (at least) the 13 origination columns, same
    shape as reconciler_agent's tabular_features (e.g. state["application"]
    ["tabular_features"]) -- see RAW_TABULAR_COLS. Returns:
        {"n_neighbors": int, "default_rate": float | None,
         "sufficient": bool, "grade_band": str}
    default_rate is the fraction of returned neighbors with y==1 (historical
    default). None / sufficient=False when fewer than MIN_NEIGHBORS genuine
    (within-cutoff) neighbors exist -- an unusual loan gets an honest
    insufficient-evidence answer, never a rate padded with dissimilar loans."""
    ids, ys, grade_band_label = _query_neighbors(features)
    n = len(ys)
    if n < MIN_NEIGHBORS:
        return {"n_neighbors": n, "default_rate": None, "sufficient": False, "grade_band": grade_band_label}
    return {
        "n_neighbors": n,
        "default_rate": float(ys.mean()),
        "sufficient": True,
        "grade_band": grade_band_label,
    }


if __name__ == "__main__":
    print("Building index (TRAIN-only standardization + per-grade-band KDTrees)...")
    _get_index()
    print(f"  {len(_index['bands'])} grade bands indexed.\n")

    frame = pd.read_pickle(FRAME_PATH)
    with open(SPLIT_PATH, "rb") as f:
        split = pickle.load(f)
    sample_loan_id = int(split["test_idx"][0])   # a real TEST loan's FEATURES only, never its outcome
    sample_features = {c: frame.loc[sample_loan_id, c] for c in RAW_TABULAR_COLS}

    print(f"Example query -- loan_id={sample_loan_id} (a TEST-set loan; only its features are used):")
    for k, v in sample_features.items():
        print(f"  {k:20s} = {v}")

    result = find_similar_loans(sample_features)
    print("\nResult:")
    for k, v in result.items():
        print(f"  {k:12s} = {v}")
