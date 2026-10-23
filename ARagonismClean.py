Python 3.11.6 (tags/v3.11.6:8b6ee5b, Oct  2 2023, 14:57:12) [MSC v.1935 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
import pandas as pd
import numpy as np
from pathlib import Path

# =========================
# User settings
# =========================
INPUT_FILE = "tox21_ar_raw.xlsx"   # modify file name
SHEET_NAME = 0                     
THRESHOLD = 10.0  #modify it to 5, 10, 20

# 输出文件
OUTPUT_CLEAN = "tox21_ar_processed.xlsx"
OUTPUT_CONFLICT = "tox21_ar_conflicting_smiles.xlsx"


def load_file(file_path, sheet_name=0):
    """
    根据文件后缀读取 Excel 或 CSV
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
    elif suffix == ".csv":
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
    return df


def normalize_channel_outcome(x):
    """
    标准化 CHANNEL_OUTCOME
    """
    if pd.isna(x):
        return np.nan
    x = str(x).strip().lower()
    # 统一可能出现的写法
    if x in ["active", "active agonist"]:
        return "active"
    elif x in ["inactive", "inactive agonist"]:
        return "inactive"
    else:
        return x


def assign_label(row, threshold=10.0):
    """
    根据规则赋标签：
    inactive -> negative / 0
    active and AC50 < threshold -> positive / 1
    active and AC50 >= threshold -> negative / 0
    """
    outcome = row["CHANNEL_OUTCOME_std"]
    ac50 = row["AC50"]

    if outcome == "inactive":
        return pd.Series(["negative", 0])

    elif outcome == "active":
        if pd.isna(ac50):
            # 你说已经删掉 inactive无AC50；这里保留防御性处理
            return pd.Series([np.nan, np.nan])
        elif ac50 < threshold:
            return pd.Series(["positive", 1])
        else:
            return pd.Series(["negative", 0])

    return pd.Series([np.nan, np.nan])


def main():
    # =========================
    # 1. Read data
    # =========================
    df = load_file(INPUT_FILE, SHEET_NAME)

    print(f"Original rows: {len(df)}")

    # =========================
    # 2. Check required columns
    # =========================
    required_cols = ["SMILES", "AC50", "CHANNEL_OUTCOME"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}")

    # =========================
    # 3. Basic cleaning
    # =========================
    df = df.copy()

    # 去掉SMILES缺失
    df = df[df["SMILES"].notna()].copy()
    df["SMILES"] = df["SMILES"].astype(str).str.strip()

    # AC50转数值
    df["AC50"] = pd.to_numeric(df["AC50"], errors="coerce")

    # 标准化 CHANNEL_OUTCOME
    df["CHANNEL_OUTCOME_std"] = df["CHANNEL_OUTCOME"].apply(normalize_channel_outcome)

    # 仅保留 active / inactive
    df = df[df["CHANNEL_OUTCOME_std"].isin(["active", "inactive"])].copy()

    print(f"Rows after basic cleaning: {len(df)}")

    # =========================
    # 4. Assign class label
    # =========================
    df[["class_note", "label"]] = df.apply(assign_label, axis=1, threshold=THRESHOLD)

    # 去掉无法赋值的行
    df = df[df["label"].notna()].copy()
    df["label"] = df["label"].astype(int)

    print(f"Rows after labeling: {len(df)}")

    # =========================
    # 5. Detect conflicting SMILES
    #    same SMILES has both active and inactive
    # =========================
    outcome_per_smiles = df.groupby("SMILES")["CHANNEL_OUTCOME_std"].nunique()
    conflicting_smiles = outcome_per_smiles[outcome_per_smiles > 1].index.tolist()

    df_conflict = df[df["SMILES"].isin(conflicting_smiles)].copy()
    df_non_conflict = df[~df["SMILES"].isin(conflicting_smiles)].copy()

    print(f"Number of conflicting SMILES: {len(conflicting_smiles)}")
    print(f"Rows with conflicting SMILES: {len(df_conflict)}")
    print(f"Rows without conflict: {len(df_non_conflict)}")

    # =========================
    # 6. Deduplicate non-conflicting rows by SMILES
    #    Keep the row with minimum AC50
    # =========================
    # 对inactive记录也按最小AC50保留；如果某些inactive AC50为空，排序时放后
    df_non_conflict = df_non_conflict.sort_values(
        by=["SMILES", "AC50"],
        ascending=[True, True],
        na_position="last"
    )

    df_final = df_non_conflict.drop_duplicates(subset=["SMILES"], keep="first").copy()

    print(f"Final deduplicated rows: {len(df_final)}")

    # =========================
    # 7. Optional: add summary sheet
    # =========================
    summary = pd.DataFrame({
        "Metric": [
            "Original rows",
            "Rows after cleaning",
            "Rows after labeling",
            "Number of conflicting SMILES",
            "Rows with conflicting SMILES",
            "Rows without conflict",
...             "Final deduplicated rows",
...             "Positive count",
...             "Negative count",
...             "Threshold"
...         ],
...         "Value": [
...             len(load_file(INPUT_FILE, SHEET_NAME)),
...             len(df),
...             len(df),
...             len(conflicting_smiles),
...             len(df_conflict),
...             len(df_non_conflict),
...             len(df_final),
...             int((df_final["label"] == 1).sum()),
...             int((df_final["label"] == 0).sum()),
...             THRESHOLD
...         ]
...     })
... 
...     # =========================
...     # 8. Save results
...     # =========================
...     with pd.ExcelWriter(OUTPUT_CLEAN, engine="openpyxl") as writer:
...         df_final.to_excel(writer, sheet_name="processed_data", index=False)
...         summary.to_excel(writer, sheet_name="summary", index=False)
... 
...     with pd.ExcelWriter(OUTPUT_CONFLICT, engine="openpyxl") as writer:
...         df_conflict.to_excel(writer, sheet_name="conflicting_rows", index=False)
... 
...         if len(conflicting_smiles) > 0:
...             pd.DataFrame({"SMILES": conflicting_smiles}).to_excel(
...                 writer, sheet_name="conflicting_smiles_list", index=False
...             )
... 
...     print(f"\nProcessed data saved to: {OUTPUT_CLEAN}")
...     print(f"Conflict report saved to: {OUTPUT_CONFLICT}")
... 
... 
... if __name__ == "__main__":
