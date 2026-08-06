"""
Build the fixed modeling population: 2007-2013 vintages, resolved
loans only (Fully Paid / Charged Off), restricted to rows with a
non-empty `desc` field so tabular- and text-based models are always
compared on the identical population. Also strips the LendingClub
template wrapper ("Borrower added on MM/DD/YY >", <br> tags) from
`desc` to leave the borrower's actual free text.
"""
import re
import pandas as pd

SUBSET_PATH = "data/interim/subset_cache.pkl"
OUT_PATH = "data/interim/feasibility_frame.pkl"

BORROWER_PREFIX_RE = re.compile(r"Borrower added on \d{2}/\d{2}/\d{2}\s*>\s*", re.IGNORECASE)
HTML_TAG_RE = re.compile(r"<[^>]+>")


def clean_desc(text: str) -> str:
    t = BORROWER_PREFIX_RE.sub(" ", text)
    t = HTML_TAG_RE.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def build_feasibility_frame():
    df = pd.read_pickle(SUBSET_PATH)

    if "issue_year" not in df.columns:
        df["issue_year"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce").dt.year

    mask_years = df["issue_year"].between(2007, 2013)
    mask_status = df["loan_status"].isin(["Fully Paid", "Charged Off"])
    df = df.loc[mask_years & mask_status].copy()

    df["default"] = (df["loan_status"] == "Charged Off").astype(int)

    desc_stripped = df["desc"].str.strip()
    is_nonempty = desc_stripped.notna() & (desc_stripped != "")
    df = df.loc[is_nonempty].copy()
    df["desc"] = df["desc"].str.strip()
    df["desc_clean"] = df["desc"].apply(clean_desc)

    df.to_pickle(OUT_PATH)
    return df


if __name__ == "__main__":
    df = build_feasibility_frame()
    print(f"Feasibility frame: {df.shape[0]} rows, {df.shape[1]} cols")
    print(df["default"].value_counts(normalize=True).rename("pct") * 100)
