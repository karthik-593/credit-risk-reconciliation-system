"""
Tests for agent/similar_loan_tool.py. The leakage fence is tested FIRST and
must pass before this tool's output is trusted for anything -- see the
module docstring in similar_loan_tool.py for exactly what "leakage" means
here (row/column/outcome).

Script-style, matching this directory's other test files -- run directly:
    python agent/test_similar_loan_tool.py
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from eval_agent import RAW_TABULAR_COLS  # noqa: E402

sys.path.insert(0, str(ROOT / "agent"))
import similar_loan_tool as slt  # noqa: E402

FRAME_PATH = ROOT / "data" / "interim" / "feasibility_frame.pkl"
SPLIT_PATH = ROOT / "data" / "interim" / "split_indices.pkl"

_frame = pd.read_pickle(FRAME_PATH)
with open(SPLIT_PATH, "rb") as f:
    _split = pickle.load(f)
TRAIN_IDX = set(int(i) for i in _split["train_idx"])
TEST_IDX = set(int(i) for i in _split["test_idx"])


def _features_for(loan_id: int) -> dict:
    row = _frame.loc[loan_id]
    return {c: row[c] for c in RAW_TABULAR_COLS}


def test_candidate_pool_is_train_only():
    """Structural leakage guard: the ENTIRE candidate universe the tool can
    ever draw a neighbor from must be a subset of train_idx and disjoint
    from test_idx -- checked exhaustively (every band's full candidate
    pool), not sampled."""
    idx = slt._get_index()
    assert len(idx["bands"]) > 0, "index built no grade bands at all"
    for grade, band in idx["bands"].items():
        ids = set(int(i) for i in band["ids"])
        assert ids <= TRAIN_IDX, f"grade {grade}: candidate pool contains a non-train loan_id"
        assert ids.isdisjoint(TEST_IDX), f"grade {grade}: candidate pool contains a TEST loan_id"


def test_no_neighbor_ever_in_test_set():
    """Behavioral leakage guard: query with real TEST-set loans' FEATURES
    (never their outcome) and confirm every neighbor id actually returned
    is a genuine train_idx loan, never from test_idx."""
    rng = np.random.default_rng(42)
    sample_ids = rng.choice(list(TEST_IDX), size=30, replace=False)
    checked = 0
    for loan_id in sample_ids:
        features = _features_for(int(loan_id))
        ids, ys, _ = slt._query_neighbors(features)
        if len(ids) == 0:
            continue
        id_set = set(int(i) for i in ids)
        assert id_set <= TRAIN_IDX, f"query loan_id={loan_id}: a returned neighbor is not in train_idx"
        assert id_set.isdisjoint(TEST_IDX), f"query loan_id={loan_id}: a returned neighbor is a TEST loan"
        checked += 1
    assert checked > 0, "no query in the sample returned any neighbors -- test didn't actually exercise anything"


def test_distance_matrix_uses_only_origination_columns():
    """Column-leakage guard: the encoded feature matrix must only ever be
    built from origination-time columns (a subset of the 13), never a
    post-origination field like loan_status/default/issue_d/desc."""
    assert set(slt.DISTANCE_COLS) <= set(RAW_TABULAR_COLS), \
        "DISTANCE_COLS must be a subset of the 13 origination features"

    sample = _frame.head(20)
    encoded = slt._encode_features(sample)
    assert list(encoded.columns) == slt.DISTANCE_COLS

    forbidden = {"loan_status", "default", "issue_d", "issue_year", "desc", "desc_clean", "title"}
    assert forbidden.isdisjoint(encoded.columns), \
        f"post-origination column(s) leaked into the distance matrix: {forbidden & set(encoded.columns)}"


def test_insufficient_for_unusual_loan():
    """An extreme, out-of-distribution query (high rate + tiny income +
    short tenure combination that's rare within grade A) should not find
    25 genuine same-band neighbors."""
    features = {
        "loan_amnt": 40000.0, "term": " 60 months", "int_rate": 30.99,
        "grade": "A", "sub_grade": "A1", "annual_inc": 9000.0, "dti": 45.0,
        "emp_length": "< 1 year", "home_ownership": "RENT",
        "verification_status": "Not Verified", "fico_range_low": 660.0,
        "fico_range_high": 664.0, "purpose": "small_business",
    }
    result = slt.find_similar_loans(features)
    assert result["sufficient"] is False, f"expected an unusual loan to be insufficient, got {result}"
    assert result["default_rate"] is None
    assert result["n_neighbors"] < slt.MIN_NEIGHBORS


def test_sufficient_returns_valid_fraction():
    rng = np.random.default_rng(7)
    sample_ids = rng.choice(list(TEST_IDX), size=20, replace=False)
    saw_sufficient = False
    for loan_id in sample_ids:
        result = slt.find_similar_loans(_features_for(int(loan_id)))
        assert isinstance(result["n_neighbors"], int)
        if result["sufficient"]:
            saw_sufficient = True
            assert 0.0 <= result["default_rate"] <= 1.0, f"default_rate out of [0,1]: {result}"
            assert result["n_neighbors"] >= slt.MIN_NEIGHBORS
        else:
            assert result["default_rate"] is None
    assert saw_sufficient, "expected at least one sufficient result among 20 real TEST loans"


def test_grade_ordering_sanity():
    """Soft check: A-grade queries should show a lower observed neighbor
    default rate than D-grade queries -- confirms similarity is
    meaningfully tied to risk, not scrambled. Uses real TEST-set loans of
    each grade (features only)."""
    grade_a_ids = [i for i in TEST_IDX if _frame.loc[i, "grade"] == "A"][:15]
    grade_d_ids = [i for i in TEST_IDX if _frame.loc[i, "grade"] == "D"][:15]
    assert len(grade_a_ids) >= 5 and len(grade_d_ids) >= 5, "not enough A/D test loans to run this sanity check"

    def avg_rate(ids):
        rates = [r["default_rate"] for i in ids
                 if (r := slt.find_similar_loans(_features_for(i)))["sufficient"]]
        return (sum(rates) / len(rates)) if rates else None

    rate_a = avg_rate(grade_a_ids)
    rate_d = avg_rate(grade_d_ids)
    assert rate_a is not None and rate_d is not None, "not enough sufficient results to compare grades"
    print(f"    avg A-grade neighbor default_rate={rate_a:.3f}   avg D-grade={rate_d:.3f}")
    assert rate_a < rate_d, "grade encodes risk -- A should show a lower historical default rate than D"


if __name__ == "__main__":
    test_candidate_pool_is_train_only()
    print("PASS  candidate pool is train-only, disjoint from test_idx (structural, exhaustive)")

    test_no_neighbor_ever_in_test_set()
    print("PASS  no neighbor ever in test set (behavioral, 30 real TEST-loan queries)")

    test_distance_matrix_uses_only_origination_columns()
    print("PASS  distance matrix uses only origination columns")

    test_insufficient_for_unusual_loan()
    print("PASS  unusual out-of-distribution loan -> sufficient=False, default_rate=None")

    test_sufficient_returns_valid_fraction()
    print("PASS  sufficient results have a valid [0,1] default_rate, n_neighbors>=MIN_NEIGHBORS")

    test_grade_ordering_sanity()
    print("PASS  grade ordering sanity check (A-grade default rate < D-grade)")

    print("\nAll similar_loan_tool tests passed.")
