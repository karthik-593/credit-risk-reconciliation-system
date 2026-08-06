"""
Read only the CSV header (nrows=0) to confirm which columns exist
before loading any row data. Avoids pulling ~2M rows into memory
just to check schema.
"""
import pandas as pd

RAW_PATH = "data/raw/accepted_2007_to_2018Q4.csv.gz"

df0 = pd.read_csv(RAW_PATH, compression="gzip", nrows=0, low_memory=False)
cols = list(df0.columns)

print(f"Total columns: {len(cols)}")
print("\nFull column list:")
for i, c in enumerate(cols):
    print(f"{i + 1:3d}. {c}")

check_cols = ["desc", "title", "emp_title", "purpose", "loan_status", "issue_d"]
fico_cols = [c for c in cols if "fico" in c.lower()]

print("\n--- Presence check ---")
for c in check_cols:
    print(f"{c}: {'PRESENT' if c in cols else 'MISSING'}")

print(f"\nFICO-related columns found: {fico_cols if fico_cols else 'NONE'}")
