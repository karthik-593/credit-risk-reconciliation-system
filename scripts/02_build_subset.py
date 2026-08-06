"""
Load only the columns needed for the analysis, in chunks, and cache
the result. Also runs the desc fill-rate-by-year and loan maturity
checks that determine which vintages are usable for text modeling.
"""
import pandas as pd

RAW_PATH = "data/raw/accepted_2007_to_2018Q4.csv.gz"
CACHE_PATH = "data/interim/subset_cache.pkl"

USECOLS = [
    "desc", "title", "purpose", "loan_status", "issue_d",
    "loan_amnt", "term", "int_rate", "grade", "sub_grade",
    "annual_inc", "dti", "emp_length", "home_ownership",
    "verification_status", "fico_range_low", "fico_range_high",
]

DTYPE_MAP = {
    "desc": "string", "title": "string", "purpose": "category",
    "loan_status": "category", "issue_d": "string",
    "loan_amnt": "float32", "term": "category", "int_rate": "float32",
    "grade": "category", "sub_grade": "category", "annual_inc": "float32",
    "dti": "float32", "emp_length": "category", "home_ownership": "category",
    "verification_status": "category", "fico_range_low": "float32",
    "fico_range_high": "float32",
}


def build_subset():
    chunks = [
        chunk for chunk in pd.read_csv(
            RAW_PATH, compression="gzip", usecols=USECOLS,
            dtype=DTYPE_MAP, chunksize=200_000, low_memory=False,
        )
    ]
    df = pd.concat(chunks, ignore_index=True)
    df.to_pickle(CACHE_PATH)
    return df


def desc_fill_rate_by_year(df):
    df["issue_year"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce").dt.year
    desc_stripped = df["desc"].str.strip()
    is_nonempty = desc_stripped.notna() & (desc_stripped != "")
    df["_desc_nonempty"] = is_nonempty
    df["_desc_len"] = desc_stripped.str.len().where(is_nonempty)

    rows = []
    for year, g in df.groupby("issue_year", dropna=True):
        total = len(g)
        n_nonempty = g["_desc_nonempty"].sum()
        rows.append({
            "year": int(year),
            "total_loans": total,
            "desc_nonempty_count": int(n_nonempty),
            "desc_nonempty_pct": round(100 * n_nonempty / total, 2) if total else 0,
            "median_desc_len_chars": g.loc[g["_desc_nonempty"], "_desc_len"].median(),
        })
    return pd.DataFrame(rows).sort_values("year")


def maturity_check(df):
    print("=== loan_status value counts (full subset) ===")
    print(df["loan_status"].value_counts(dropna=False))

    sub = df[df["issue_year"].between(2007, 2013)]
    print("\n=== loan_status value counts (2007-2013 only) ===")
    print(sub["loan_status"].value_counts(dropna=False))

    resolved = sub["loan_status"].isin(["Fully Paid", "Charged Off"]).sum()
    print(f"\n2007-2013 total: {len(sub)}, resolved: {resolved} ({100 * resolved / len(sub):.2f}%)")


if __name__ == "__main__":
    df = build_subset()
    print(f"Subset shape: {df.shape}")

    print("\n=== desc fill rate by issue year ===")
    print(desc_fill_rate_by_year(df).to_string(index=False))

    print()
    maturity_check(df)
