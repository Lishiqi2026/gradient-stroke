#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regenerate Table 4 from the clinical workbook using participants with paired
baseline (before) and post-intervention (after) lesion files.

The follow-up clinical workbook is not used because Table 4 contains baseline
demographic and clinical characteristics only.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

CLINICAL_XLSX = Path(r"F:\lishiqi\预处理数据\cortex\董分组.xlsx")
LESION_ROOT = Path(r"H:\fmridata\Lession\Bingzao")
OUTPUT_DIR = Path(r"F:\lishiqi\end_picture\Ending\Table 4")

GROUPS = ("Control", "TMS")
EXPECTED_N = {"Control": 18, "TMS": 23}

# Workbook columns by position: A ID, C sex, D age, E lesion side, G Group.
# Baseline clinical scores: H NIHSS, J ADL, L MMSE, N ESRS, O HAMD, Q VFT.
COL = {
    "id": 0,
    "sex": 2,
    "age": 3,
    "lesion_side": 4,
    "group": 6,
    "NIHSS": 7,
    "ADL": 9,
    "MMSE": 11,
    "ESRS": 13,
    "HAMD": 14,
    "VFT": 16,
}
MEASURES = ("NIHSS", "ADL", "MMSE", "ESRS", "HAMD", "VFT")


def normalize_id(value) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\.0$", "", str(value).strip())


def get_file_ids(folder: Path) -> set[str]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Expected lesion folder not found: {folder}")
    ids = set()
    for path in folder.rglob("*"):
        if not path.is_file():
            continue
        match = re.match(r"(\d+)-", path.name)
        if match:
            ids.add(match.group(1))
    return ids


def fmt_mean_sd(values: pd.Series) -> str:
    return f"{values.mean():.1f} $\\pm$ {values.std(ddof=1):.1f}"


def fmt_count_pct(count: int, denominator: int) -> str:
    return f"{count} ({100 * count / denominator:.1f}\\%)"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a paired-lesion-file Table 4.")
    parser.add_argument("--clinical-xlsx", type=Path, default=CLINICAL_XLSX)
    parser.add_argument("--lesion-root", type=Path, default=LESION_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    raw = pd.read_excel(args.clinical_xlsx)
    if raw.shape[1] < 17:
        raise ValueError(f"Expected at least 17 columns in clinical workbook; found {raw.shape[1]}.")

    data = raw.copy()
    id_column = data.columns[COL["id"]]
    data[id_column] = data.iloc[:, COL["id"]].map(normalize_id)
    data["_Group"] = data.iloc[:, COL["group"]].astype(str).str.strip()
    data = data[data["_Group"].isin(GROUPS)].copy()

    if data[id_column].duplicated().any():
        duplicates = data.loc[data[id_column].duplicated(False), id_column].tolist()
        raise ValueError(f"Duplicate participant IDs in workbook: {duplicates}")

    file_ids: dict[str, dict[str, set[str]]] = {}
    for group in GROUPS:
        before_ids = get_file_ids(args.lesion_root / group / "before")
        after_ids = get_file_ids(args.lesion_root / group / "after")
        file_ids[group] = {"before": before_ids, "after": after_ids}

    # Stop if a file is stored under a group inconsistent with the workbook.
    workbook_group = dict(zip(data[id_column], data["_Group"]))
    for folder_group in GROUPS:
        for timepoint in ("before", "after"):
            for subject_id in file_ids[folder_group][timepoint]:
                if subject_id in workbook_group and workbook_group[subject_id] != folder_group:
                    raise ValueError(
                        f"Group mismatch for {subject_id}: folder={folder_group}, "
                        f"workbook={workbook_group[subject_id]}"
                    )

    included_by_group = {
        group: file_ids[group]["before"] & file_ids[group]["after"]
        for group in GROUPS
    }

    # This guard protects against unnoticed changes in the file inventory.
    observed_counts = {
        group: len(included_by_group[group] & set(data.loc[data["_Group"] == group, id_column]))
        for group in GROUPS
    }
    if observed_counts != EXPECTED_N:
        raise ValueError(
            f"Paired lesion-file cohort changed. Expected {EXPECTED_N}, found {observed_counts}. "
            "Review the file inventory and update the expected counts only after verification."
        )

    selected_ids = set().union(*(included_by_group[group] for group in GROUPS))
    selected = data[data[id_column].isin(selected_ids)].copy()

    # Ensure each table variable is complete in the analysis cohort.
    requested_positions = [COL[k] for k in ("sex", "age", "lesion_side", *MEASURES)]
    missing = {}
    for position in requested_positions:
        column = data.columns[position]
        n_missing = int(selected.iloc[:, position].isna().sum())
        if n_missing:
            missing[str(column)] = n_missing
    if missing:
        raise ValueError(f"Missing values in Table 4 analysis data: {missing}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Audit each clinical record against the paired lesion-file requirement.
    audit_rows = []
    for _, row in data.iterrows():
        subject_id = row[id_column]
        group = row["_Group"]
        has_before = subject_id in file_ids[group]["before"]
        has_after = subject_id in file_ids[group]["after"]
        if has_before and has_after:
            status = "included: paired before/after lesion files"
        elif has_before:
            status = "excluded: after lesion file unavailable"
        elif has_after:
            status = "excluded: before lesion file unavailable"
        else:
            status = "excluded: before and after lesion files unavailable"
        audit_rows.append({
            "Subject_ID": subject_id,
            "Group": group,
            "Before_lesion_file": has_before,
            "After_lesion_file": has_after,
            "Table4_status": status,
        })
    audit = pd.DataFrame(audit_rows).sort_values(["Group", "Subject_ID"])
    audit_path = args.output_dir / "clinical_table4_inclusion_audit.csv"
    audit.to_csv(audit_path, index=False, encoding="utf-8-sig")

    # Build summary rows in manuscript display order.
    stats = []
    for group in GROUPS:
        group_data = selected[selected["_Group"] == group]
        n = len(group_data)
        stats.append({
            "Group": group,
            "n": n,
            "Age_mean": group_data.iloc[:, COL["age"]].mean(),
            "Age_sd": group_data.iloc[:, COL["age"]].std(ddof=1),
            "Male_n": int((group_data.iloc[:, COL["sex"]].astype(str).str.upper() == "M").sum()),
            "Female_n": int((group_data.iloc[:, COL["sex"]].astype(str).str.upper() == "F").sum()),
            "Right_n": int((group_data.iloc[:, COL["lesion_side"]].astype(str).str.strip() == "右").sum()),
            "Left_n": int((group_data.iloc[:, COL["lesion_side"]].astype(str).str.strip() == "左").sum()),
        })
        for measure in MEASURES:
            values = group_data.iloc[:, COL[measure]]
            stats[-1][f"{measure}_mean"] = values.mean()
            stats[-1][f"{measure}_sd"] = values.std(ddof=1)

    overall = {"Group": "Overall", "n": len(selected)}
    for key in ("age",):
        values = selected.iloc[:, COL[key]]
        overall["Age_mean"] = values.mean()
        overall["Age_sd"] = values.std(ddof=1)
    overall["Male_n"] = int((selected.iloc[:, COL["sex"]].astype(str).str.upper() == "M").sum())
    overall["Female_n"] = int((selected.iloc[:, COL["sex"]].astype(str).str.upper() == "F").sum())
    overall["Right_n"] = int((selected.iloc[:, COL["lesion_side"]].astype(str).str.strip() == "右").sum())
    overall["Left_n"] = int((selected.iloc[:, COL["lesion_side"]].astype(str).str.strip() == "左").sum())
    for measure in MEASURES:
        values = selected.iloc[:, COL[measure]]
        overall[f"{measure}_mean"] = values.mean()
        overall[f"{measure}_sd"] = values.std(ddof=1)
    stats.insert(0, overall)

    stats_df = pd.DataFrame(stats)
    results_path = args.output_dir / "clinical_table4_results.csv"
    stats_df.to_csv(results_path, index=False, encoding="utf-8-sig")

    overall_row, control, tms = stats
    n_all, n_c, n_t = overall_row["n"], control["n"], tms["n"]

    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        rf"\caption{{Baseline demographic and clinical characteristics of the {n_all} participants with paired lesion files}}",
        r"\label{tab:baseline-lesion-cohort}",
        r"\small",
        r"\begin{tabular}{@{}lccc@{}}",
        r"\toprule",
        rf"Characteristic & Overall ($n={n_all}$) & Control ($n={n_c}$) & TMS ($n={n_t}$) \\",
        r"\midrule",
        rf"Age, years & {fmt_mean_sd(selected.iloc[:, COL['age']])} & {fmt_mean_sd(selected[selected['_Group']=='Control'].iloc[:, COL['age']])} & {fmt_mean_sd(selected[selected['_Group']=='TMS'].iloc[:, COL['age']])} \\",
        r"Sex, $n$ (\%) & & & \\",
        rf"\quad Male & {fmt_count_pct(overall_row['Male_n'], n_all)} & {fmt_count_pct(control['Male_n'], n_c)} & {fmt_count_pct(tms['Male_n'], n_t)} \\",
        rf"\quad Female & {fmt_count_pct(overall_row['Female_n'], n_all)} & {fmt_count_pct(control['Female_n'], n_c)} & {fmt_count_pct(tms['Female_n'], n_t)} \\",
        r"Lesion side, $n$ (\%) & & & \\",
        rf"\quad Right & {fmt_count_pct(overall_row['Right_n'], n_all)} & {fmt_count_pct(control['Right_n'], n_c)} & {fmt_count_pct(tms['Right_n'], n_t)} \\",
        rf"\quad Left & {fmt_count_pct(overall_row['Left_n'], n_all)} & {fmt_count_pct(control['Left_n'], n_c)} & {fmt_count_pct(tms['Left_n'], n_t)} \\",
        r"\midrule",
        r"\textit{Clinical assessments at baseline} & & & \\",
    ]
    for measure in MEASURES:
        lines.append(
            rf"{measure} & {fmt_mean_sd(selected.iloc[:, COL[measure]])} "
            rf"& {fmt_mean_sd(selected[selected['_Group']=='Control'].iloc[:, COL[measure]])} "
            rf"& {fmt_mean_sd(selected[selected['_Group']=='TMS'].iloc[:, COL[measure]])} \\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\vspace{4pt}",
        r"\begin{minipage}{\linewidth}",
        r"\footnotesize\raggedright",
        r"Data are mean $\pm$ s.d. or $n$ (\%). Percentages are calculated within each group. "
        r"This table summarizes participants with both baseline and post-intervention lesion files "
        rf"(Control, $n={n_c}$; TMS, $n={n_t}$). All clinical scores shown are baseline measurements.",
        r"NIHSS, National Institutes of Health Stroke Scale; ADL, activities of daily living; "
        r"HAMD, Hamilton Depression Rating Scale; MMSE, Mini-Mental State Examination; "
        r"VFT, verbal fluency test; ESRS, Essen Stroke Risk Score.",
        r"\end{minipage}",
        r"\end{table}",
        "",
    ])
    tex_path = args.output_dir / "clinical_table4_table.tex"
    tex_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Analysis cohort: N={n_all}, Control={n_c}, TMS={n_t}")
    print(f"LaTeX: {tex_path}")
    print(f"CSV: {results_path}")
    print(f"Inclusion audit: {audit_path}")
    print("Follow-up clinical workbook is not used: Table 4 contains baseline variables only.")


if __name__ == "__main__":
    main()

