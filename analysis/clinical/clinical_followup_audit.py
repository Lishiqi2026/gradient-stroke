"""Read-only audit and exploratory complete-case analysis of supplied clinical workbooks.

Run with D:\\python\\python.exe (existing Python 3.11 with scipy) or another
Python environment with numpy, scipy and openpyxl. Outputs contain no patient
names or hospital numbers. The bundled Python originally lacked scipy.
Adjacent unlabelled scale columns are ASSUMED to be baseline then post-treatment;
the source only explicitly labels the third MMSE/VFT columns as 'one month'.
"""
from __future__ import annotations

import collections
import os
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import sys

BASE = Path(os.environ.get("GRADIENT_STROKE_FOLLOWUP_OUTPUT", "outputs/clinical_followup"))
BASE.mkdir(parents=True, exist_ok=True)
import numpy as np
import openpyxl
import scipy
from scipy import stats

FULL_PATH = Path(os.environ["GRADIENT_STROKE_FULL_FOLLOWUP_XLSX"])
FOLLOW_PATH = Path(os.environ["GRADIENT_STROKE_FOLLOWUP_XLSX"])
GROUP_MAP = {"对照组": "Control", "治疗组": "TMS", "Control": "Control", "TMS": "TMS"}
SCALES = {"NIHSS": (8, 9), "ADL": (10, 11), "MMSE": (12, 13), "HAMD": (16, 17), "VFT": (18, 19)}
FOLLOW_COLS = {"MMSE": 14, "VFT": 20}
REASONS = {
    "外地，联系不上": ("unreachable_out_of_area", "Unreachable while living outside the study area"),
    "支气管咯血": ("hemoptysis", "Hemoptysis"),
    "再发脑梗死": ("recurrent_cerebral_infarction", "Recurrent cerebral infarction"),
    "消化道出血停药": ("gastrointestinal_bleeding_medication_stopped", "Gastrointestinal bleeding with medication discontinued"),
    "因肩袖损伤手术": ("rotator_cuff_surgery", "Surgery for rotator cuff injury"),
    "摔倒骨折": ("fracture_after_fall", "Fracture following a fall"),
    "冠心病放支架": ("coronary_stent", "Coronary stent placement"),
}


def native(value):
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(name, value):
    (BASE / name).write_text(json.dumps(native(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(name, rows):
    rows = native(rows)
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (BASE / name).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def normalized_key(value):
    return str(value).strip() if value is not None else ""


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def structure(path):
    wb = openpyxl.load_workbook(path, data_only=False, read_only=False)
    sheets = []
    for ws in wb:
        columns = []
        for col in range(1, ws.max_column + 1):
            values = [ws.cell(row, col).value for row in range(2, ws.max_row + 1)]
            columns.append({
                "column": openpyxl.utils.get_column_letter(col), "column_number": col,
                "header": ws.cell(1, col).value,
                "nonempty": sum(x is not None for x in values),
                "numeric": sum(numeric(x) for x in values),
                "text": sum(isinstance(x, str) for x in values),
                "blank": sum(x is None for x in values),
            })
        sheets.append({"sheet": ws.title, "max_row": ws.max_row, "max_column": ws.max_column,
                       "data_rows": max(0, ws.max_row - 1), "columns": columns,
                       "merged_ranges": [str(x) for x in ws.merged_cells.ranges],
                       "hidden_rows": [r for r, d in ws.row_dimensions.items() if d.hidden],
                       "hidden_columns": [c for c, d in ws.column_dimensions.items() if d.hidden],
                       "comment_count": sum(cell.comment is not None for row in ws for cell in row),
                       "formula_count": sum(cell.data_type == "f" for row in ws for cell in row),
                       "date_cell_count": sum(isinstance(cell.value, (datetime.datetime, datetime.date)) for row in ws for cell in row)})
    return {"path": str(path), "sha256": file_hash(path), "sheets": sheets}


def describe(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return {"n": len(x), "mean": np.mean(x), "sd": np.std(x, ddof=1) if len(x) > 1 else None,
            "median": np.median(x), "q1": np.percentile(x, 25), "q3": np.percentile(x, 75),
            "min": np.min(x), "max": np.max(x)} if len(x) else {"n": 0}


def bh(rows, pkey="p", qkey="q"):
    valid = [i for i, row in enumerate(rows) if row.get(pkey) is not None]
    order = sorted(valid, key=lambda i: rows[i][pkey])
    n = len(order)
    current = 1.0
    for rank, i in reversed(list(enumerate(order, 1))):
        current = min(current, rows[i][pkey] * n / rank)
        rows[i][qkey] = current


def paired(y, x):
    y, x = np.asarray(y, dtype=float), np.asarray(x, dtype=float)
    ok = np.isfinite(y) & np.isfinite(x)
    difference = y[ok] - x[ok]
    n = len(difference)
    result = stats.ttest_1samp(difference, 0)
    sem = stats.sem(difference)
    ci = np.mean(difference) + np.array([-1, 1]) * stats.t.ppf(.975, n - 1) * sem
    out = {"n": n, "mean_change": np.mean(difference), "sd_change": np.std(difference, ddof=1),
           "ci_low": ci[0], "ci_high": ci[1], "t": result.statistic, "df": n - 1, "p": result.pvalue,
           "zero_change_n": np.sum(difference == 0)}
    if np.any(difference != 0):
        w = stats.wilcoxon(difference, zero_method="wilcox", correction=False, alternative="two-sided", method="auto")
        out.update({"wilcoxon_statistic": w.statistic, "wilcoxon_p": w.pvalue})
    else:
        out.update({"wilcoxon_statistic": 0., "wilcoxon_p": 1.})
    return out


def welch(tms, control):
    tms, control = np.asarray(tms, dtype=float), np.asarray(control, dtype=float)
    tms, control = tms[np.isfinite(tms)], control[np.isfinite(control)]
    nt, nc = len(tms), len(control)
    vt, vc = np.var(tms, ddof=1) / nt, np.var(control, ddof=1) / nc
    se = math.sqrt(vt + vc)
    df = (vt + vc) ** 2 / (vt ** 2 / (nt - 1) + vc ** 2 / (nc - 1))
    estimate = np.mean(tms) - np.mean(control)
    ci = estimate + np.array([-1, 1]) * stats.t.ppf(.975, df) * se
    tt = stats.ttest_ind(tms, control, equal_var=False)
    return {"n_TMS": nt, "n_Control": nc, "mean_TMS": np.mean(tms), "mean_Control": np.mean(control),
            "estimate_TMS_minus_Control": estimate, "ci_low": ci[0], "ci_high": ci[1],
            "t": tt.statistic, "df": df, "p": tt.pvalue}


def ancova(records, measure, outcome_time, adjustment_time, cohort):
    complete = [r for r in records if numeric(r[f"{measure}_{outcome_time}"]) and numeric(r[f"{measure}_{adjustment_time}"])]
    y = np.array([r[f"{measure}_{outcome_time}"] for r in complete], float)
    baseline = np.array([r[f"{measure}_{adjustment_time}"] for r in complete], float)
    treatment = np.array([r["group"] == "TMS" for r in complete], float)
    x = np.column_stack((np.ones(len(y)), treatment, baseline))
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ beta
    n, k = x.shape
    df = n - k
    xtxi = np.linalg.inv(x.T @ x)
    mse = residual @ residual / df
    covariance = mse * xtxi
    h = np.einsum("ij,jk,ik->i", x, xtxi, x)
    hc3_cov = xtxi @ (x.T @ ((residual ** 2 / (1 - h) ** 2)[:, None] * x)) @ xtxi
    se = math.sqrt(covariance[1, 1])
    se_hc3 = math.sqrt(hc3_cov[1, 1])
    critical_t = stats.t.ppf(.975, df)
    ci = beta[1] + np.array([-1, 1]) * critical_t * se
    ci_hc3 = beta[1] + np.array([-1, 1]) * critical_t * se_hc3
    return {"cohort": cohort, "measure": measure,
            "model": f"{outcome_time} ~ group + {adjustment_time}", "n": n,
            "n_Control": int(np.sum(treatment == 0)), "n_TMS": int(np.sum(treatment == 1)),
            "estimate_TMS_minus_Control": beta[1], "se": se, "ci_low": ci[0], "ci_high": ci[1],
            "t": beta[1] / se, "df": df, "p": 2 * stats.t.sf(abs(beta[1] / se), df),
            "baseline_coefficient": beta[2], "intercept": beta[0],
            "hc3_se": se_hc3, "hc3_ci_low": ci_hc3[0], "hc3_ci_high": ci_hc3[1],
            "hc3_t": beta[1] / se_hc3, "hc3_p": 2 * stats.t.sf(abs(beta[1] / se_hc3), df),
            "hc3_reference_distribution": f"Student t, residual df={df}",
            "r_squared": 1 - residual @ residual / np.sum((y - np.mean(y)) ** 2),
            "max_leverage": np.max(h)}


def fmt(x, digits=3):
    return f"{x:.{digits}g}"


def interaction_sensitivity(records, measure, outcome_time, cohort):
    y = np.array([r[f"{measure}_{outcome_time}"] for r in records], float)
    b = np.array([r[f"{measure}_baseline"] for r in records], float)
    g = np.array([r["group"] == "TMS" for r in records], float)
    centered = b - b.mean()
    x = np.column_stack((np.ones(len(y)), g, centered, g * centered))
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    resid = y - x @ beta
    inv = np.linalg.inv(x.T @ x)
    h = np.einsum("ij,jk,ik->i", x, inv, x)
    df = len(y) - 4
    conventional = (resid @ resid / df) * inv
    robust = inv @ (x.T @ ((resid ** 2 / (1 - h) ** 2)[:, None] * x)) @ inv
    out = {"cohort": cohort, "measure": measure, "model": f"{outcome_time} ~ group + centered_baseline + group:centered_baseline",
           "n": len(y), "df": df, "pooled_mean_baseline": b.mean(),
           "Control_baseline_slope": beta[2], "TMS_baseline_slope": beta[2] + beta[3],
           "overlapping_baseline_range_low": max(b[g == 0].min(), b[g == 1].min()),
           "overlapping_baseline_range_high": min(b[g == 0].max(), b[g == 1].max())}
    for name, index in [("group_at_mean_baseline", 1), ("interaction", 3)]:
        out[f"{name}_estimate"] = beta[index]
        for method, cov in [("ols", conventional), ("hc3", robust)]:
            se = math.sqrt(cov[index, index])
            ci = beta[index] + np.array([-1, 1]) * stats.t.ppf(.975, df) * se
            out[f"{name}_{method}_se"] = se
            out[f"{name}_{method}_ci_low"] = ci[0]
            out[f"{name}_{method}_ci_high"] = ci[1]
            out[f"{name}_{method}_p"] = 2 * stats.t.sf(abs(beta[index] / se), df)
    return out


def main():
    BASE.mkdir(parents=True, exist_ok=True)
    fs = openpyxl.load_workbook(FULL_PATH, data_only=True, read_only=True)["Sheet1"]
    ss = openpyxl.load_workbook(FOLLOW_PATH, data_only=True, read_only=True)["Sheet1"]
    full_rows = [tuple(row) for row in fs.iter_rows(min_row=2, values_only=True)]
    short_rows = [tuple(row) for row in ss.iter_rows(min_row=2, values_only=True)]
    ids = collections.Counter(normalized_key(r[0]) for r in full_rows)
    names = collections.Counter(normalized_key(r[1]) for r in full_rows)
    short_ids = collections.Counter(normalized_key(r[0]) for r in short_rows)
    short_names = collections.Counter(normalized_key(r[1]) for r in short_rows)
    assert len(full_rows) == 45 and len(short_rows) == 28
    assert all(v == 1 for v in ids.values()) and all(v == 1 for v in names.values())
    assert all(v == 1 for v in short_ids.values()) and all(v == 1 for v in short_names.values())
    full_index = {normalized_key(r[0]): i for i, r in enumerate(full_rows)}
    records = []
    for i, row in enumerate(full_rows):
        record = {"anonymous_id": f"P{i + 1:03d}", "source_excel_row": i + 2,
                  "group": GROUP_MAP[row[6]], "sex": row[2], "age": row[3], "ESRS_baseline": row[14]}
        for scale, (before, after) in SCALES.items():
            record[f"{scale}_baseline"] = row[before - 1]
            record[f"{scale}_post"] = row[after - 1]
        for scale, col in FOLLOW_COLS.items():
            value = row[col - 1]
            record[f"{scale}_followup"] = value if numeric(value) else None
        record["followup_available"] = numeric(record["MMSE_followup"]) and numeric(record["VFT_followup"])
        reason = row[13] if isinstance(row[13], str) else None
        assert reason is None or reason in REASONS
        record["unavailable_reason_code"] = REASONS[reason][0] if reason else None
        records.append(record)
    matches, conflicts = [], []
    for j, row in enumerate(short_rows):
        ident = normalized_key(row[0])
        assert ident in full_index
        i = full_index[ident]
        match = {"anonymous_id": records[i]["anonymous_id"], "full_excel_row": i + 2,
                 "follow_excel_row": j + 2, "same_identifier": True,
                 "same_name_internal_check": normalized_key(row[1]) == normalized_key(full_rows[i][1]),
                 "same_group": GROUP_MAP[row[2]] == records[i]["group"]}
        for field, col in [("MMSE_baseline", 4), ("MMSE_post", 5), ("MMSE_followup", 6),
                           ("VFT_baseline", 7), ("VFT_post", 8), ("VFT_followup", 9)]:
            match[f"same_{field}"] = row[col - 1] == records[i][field]
            if not match[f"same_{field}"]:
                conflicts.append({"anonymous_id": records[i]["anonymous_id"], "field": field,
                                  "full_value": records[i][field], "follow_value": row[col - 1]})
        matches.append(match)
    retained = [r for r in records if r["followup_available"]]
    missing = [r for r in records if not r["followup_available"]]
    matched_ids = {r["anonymous_id"] for r in matches}
    assert matched_ids == {r["anonymous_id"] for r in retained}
    assert all(r["MMSE_followup"] is None and r["VFT_followup"] is None for r in missing)
    assert all(r["unavailable_reason_code"] is not None for r in missing)
    assert not conflicts and all(all(v for k, v in m.items() if k.startswith("same_")) for m in matches)
    reconciliation = {"full_rows": len(records), "followup_rows": len(retained),
                      "duplicate_identifier_count_full": sum(v - 1 for v in ids.values()),
                      "duplicate_name_count_full": sum(v - 1 for v in names.values()),
                      "duplicate_identifier_count_follow": sum(v - 1 for v in short_ids.values()),
                      "duplicate_name_count_follow": sum(v - 1 for v in short_names.values()),
                      "all_followup_rows_match_full_by_both_identifier_and_name": True,
                      "clinical_values_compared": 28 * 6, "clinical_value_conflicts": conflicts,
                      "all_groups_agree": True, "numeric_followup_set_matches_short_workbook": True,
                      "matches": matches,
                      "mri_analysis_membership": "Unavailable; cannot identify manuscript 42/37 participants from these workbooks."}
    availability = []
    for group in ["All", "Control", "TMS"]:
        rr = [r for r in records if group == "All" or r["group"] == group]
        for scale in SCALES:
            for time in ["baseline", "post", "followup"]:
                field = f"{scale}_{time}"
                structurally_available = time != "followup" or scale in FOLLOW_COLS
                n = sum(numeric(r.get(field)) for r in rr)
                availability.append({"group": group, "measure": scale, "time": time,
                                     "cohort_denominator": len(rr), "numeric_n": n,
                                     "missing_n": len(rr) - n if structurally_available else None,
                                     "column_present": structurally_available,
                                     "interpretation": "available measurement" if structurally_available else "not supplied; do not call these patients lost to follow-up for this scale"})
    reason_rows = []
    for chinese, (code, english) in REASONS.items():
        reason_rows.append({"reason_code": code, "recorded_text": chinese, "english": english,
                           "Control": sum(r["group"] == "Control" and r["unavailable_reason_code"] == code for r in records),
                           "TMS": sum(r["group"] == "TMS" and r["unavailable_reason_code"] == code for r in records),
                           "All": sum(r["unavailable_reason_code"] == code for r in records)})
    descriptive = []
    for cohort, rr in [("all_45", records), ("followup_complete_28", retained), ("followup_unavailable_17", missing)]:
        for group in ["All", "Control", "TMS"]:
            gg = [r for r in rr if group == "All" or r["group"] == group]
            for scale in SCALES:
                for time in ["baseline", "post", "followup"]:
                    xx = [r.get(f"{scale}_{time}") for r in gg if numeric(r.get(f"{scale}_{time}"))]
                    if xx:
                        descriptive.append({"cohort": cohort, "group": group, "measure": scale, "time": time, **describe(xx)})
    paired_rows = []
    for cohort, rr, scales, comparisons in [
        ("all_45_audit_only", records, list(SCALES), [("baseline", "post")]),
        ("followup_complete_28", retained, ["MMSE", "VFT"], [("baseline", "post"), ("baseline", "followup"), ("post", "followup")])]:
        for group in ["Control", "TMS"]:
            gg = [r for r in rr if r["group"] == group]
            for before, after in comparisons:
                family = []
                for scale in scales:
                    row = {"cohort": cohort, "group": group, "measure": scale, "comparison": f"{after}_minus_{before}",
                           "interpretation": "positive is improvement" if scale not in ["NIHSS", "HAMD"] else "negative is improvement",
                           **paired([r[f"{scale}_{after}"] for r in gg], [r[f"{scale}_{before}"] for r in gg])}
                    family.append(row)
                bh(family)
                bh(family, pkey="wilcoxon_p", qkey="wilcoxon_q")
                paired_rows.extend(family)
    # Each within-group follow-up time contrast spans two outcomes x two groups.
    for comparison in ["post_minus_baseline", "followup_minus_baseline", "followup_minus_post"]:
        family = [r for r in paired_rows if r["cohort"] == "followup_complete_28" and r["comparison"] == comparison]
        bh(family)
        bh(family, pkey="wilcoxon_p", qkey="wilcoxon_q")
        for row in family:
            row["bh_family_size"] = len(family)
    between_rows = []
    for cohort, rr, scales, comparisons in [
        ("all_45_audit_only", records, list(SCALES), [("baseline", "post")]),
        ("followup_complete_28", retained, ["MMSE", "VFT"], [("baseline", "followup"), ("post", "followup")])]:
        for before, after in comparisons:
            family = []
            for scale in scales:
                sign = -1 if scale in ["NIHSS", "HAMD"] else 1
                tt = [sign * (r[f"{scale}_{after}"] - r[f"{scale}_{before}"]) for r in rr if r["group"] == "TMS"]
                cc = [sign * (r[f"{scale}_{after}"] - r[f"{scale}_{before}"]) for r in rr if r["group"] == "Control"]
                family.append({"cohort": cohort, "measure": scale, "comparison": f"improvement_{after}_versus_{before}", **welch(tt, cc)})
            bh(family)
            between_rows.extend(family)
    ancovas = []
    for cohort, rr, scales, outcome, adjust in [
        ("all_45_audit_only", records, list(SCALES), "post", "baseline"),
        ("followup_complete_28", retained, ["MMSE", "VFT"], "followup", "baseline"),
        ("followup_complete_28", retained, ["MMSE", "VFT"], "followup", "post")]:
        family = [ancova(rr, scale, outcome, adjust, cohort) for scale in scales]
        bh(family)
        bh(family, pkey="hc3_p", qkey="hc3_q")
        for row in family:
            row["bh_family_size"] = len(family)
        ancovas.extend(family)
    attrition_baseline = []
    for group in ["All", "Control", "TMS"]:
        good = [r for r in retained if group == "All" or r["group"] == group]
        bad = [r for r in missing if group == "All" or r["group"] == group]
        for measure, field in [("age", "age"), ("ESRS", "ESRS_baseline")] + [(s, f"{s}_baseline") for s in SCALES]:
            gx, bx = np.array([r[field] for r in good], float), np.array([r[field] for r in bad], float)
            pool = math.sqrt(((len(gx) - 1) * gx.var(ddof=1) + (len(bx) - 1) * bx.var(ddof=1)) / (len(gx) + len(bx) - 2))
            attrition_baseline.append({"group": group, "measure": measure, "retained_n": len(gx), "unavailable_n": len(bx),
                                       "retained_mean": gx.mean(), "retained_sd": gx.std(ddof=1),
                                       "unavailable_mean": bx.mean(), "unavailable_sd": bx.std(ddof=1),
                                       "difference_retained_minus_unavailable": gx.mean() - bx.mean(),
                                       "standardized_difference": (gx.mean() - bx.mean()) / pool})
    interactions = []
    for cohort, rr, scales, time in [("all_45_audit_only", records, list(SCALES), "post"),
                                     ("followup_complete_28", retained, ["MMSE", "VFT"], "followup")]:
        family = [interaction_sensitivity(rr, scale, time, cohort) for scale in scales]
        for effect in ["group_at_mean_baseline", "interaction"]:
            for method in ["ols", "hc3"]:
                bh(family, pkey=f"{effect}_{method}_p", qkey=f"{effect}_{method}_q")
        for r in family:
            r["bh_family_size"] = len(family)
        interactions.extend(family)
    table = np.array([[12, 8], [16, 9]])
    fisher = stats.fisher_exact(table, alternative="two-sided")
    attrition = {"Control": {"total": 20, "available": 12, "unavailable": 8, "available_percent": 60, "unavailable_percent": 40},
                 "TMS": {"total": 25, "available": 16, "unavailable": 9, "available_percent": 64, "unavailable_percent": 36},
                 "All": {"total": 45, "available": 28, "unavailable": 17, "available_percent": 100 * 28 / 45, "unavailable_percent": 100 * 17 / 45},
                 "Fisher_exact_table_Control_then_TMS_available_unavailable": table.tolist(),
                 "Fisher_exact_two_sided_p": fisher.pvalue, "Fisher_odds_ratio_available_Control_vs_TMS": fisher.statistic,
                 "missing_pattern": "All 17 with unavailable one-month MMSE also have unavailable one-month VFT. Both baseline and post-treatment scores are complete for all 45.",
                 "missing_mechanism": "Cannot establish MCAR/MAR/MNAR from the supplied spreadsheets. Recorded clinical events may be associated with outcome.",
                 "event_dates_available": False, "causal_relation_to_TMS_recorded": False,
                 "time_label": "one month", "anchor_to_end_of_treatment_explicitly_recorded": False,
                 "reasons": reason_rows}
    tms_adl = [r["ADL_post"] for r in records if r["group"] == "TMS"]
    published_mean = 77.29
    implied_removed = sum(tms_adl) - 24 * published_mean
    source_compatibility = {
        "published_TMS_n": 24,
        "published_TMS_ADL_post_mean_rounded": published_mean,
        "supplied_TMS_n": len(tms_adl),
        "supplied_TMS_ADL_post_sum": sum(tms_adl),
        "supplied_TMS_ADL_post_mean": float(np.mean(tms_adl)),
        "supplied_TMS_ADL_post_max": max(tms_adl),
        "required_excluded_ADL_post_to_match_published_mean": implied_removed,
        "required_excluded_ADL_post_interval_from_2decimal_rounding": [sum(tms_adl) - 24 * (published_mean + .005), sum(tms_adl) - 24 * (published_mean - .005)],
        "possible_by_excluding_one_supplied_TMS_participant": bool(min(tms_adl) <= implied_removed <= max(tms_adl)),
        "interpretation": "The manuscript's TMS ADL post mean for n=24 cannot arise by excluding one of the 25 supplied TMS records, even allowing two-decimal rounding. This demonstrates a source/result incompatibility; it does not identify its cause.",
    }
    issues = [
        "The first two scale columns are not explicitly labelled pre/post; analysis assumes baseline then post-treatment in manuscript order.",
        "The one-month time label has no dates or explicit anchor to completion of intervention.",
        "No supplied imaging inclusion/QC identifier permits reconstruction of the manuscript 42-person (18/24) or 37-person imaging subsets.",
        "The workbook does not provide one-month NIHSS, ADL or HAMD columns; these scales cannot be analysed longitudinally at one month.",
        "Reasons are recorded in the MMSE follow-up cells and are applied to the participant's unavailable follow-up, supported by identical VFT missingness; no event onset dates or treatment-causality adjudications are present.",
        "All 28 follow-up records are complete cases selected from the 45-person clinical cohort. Estimated effects are exploratory complete-case associations, not intention-to-treat estimates.",
        "The manuscript n=24 TMS ADL post mean (77.29) is incompatible with any n=24 subset of the supplied 25 TMS records; the proposed full-45 clinical analysis must be explicitly distinguished from the manuscript's 42-person imaging cohort.",
    ]
    results = {"assumptions": issues, "software": {"python": sys.version, "numpy": np.__version__, "openpyxl": openpyxl.__version__, "scipy": scipy.__version__},
               "reconciliation": reconciliation, "attrition": attrition, "availability": availability,
               "descriptive": descriptive, "paired_tests": paired_rows, "between_group_welch_tests": between_rows,
               "ancova": ancovas, "attrition_baseline_description": attrition_baseline,
               "published_42_clinical_source_compatibility": source_compatibility,
               "baseline_group_interaction_sensitivity": interactions}
    write_json("workbook_structure.json", [structure(FULL_PATH), structure(FOLLOW_PATH)])
    write_json("identity_reconciliation.json", reconciliation)
    write_json("analysis_results.json", results)
    write_csv("anonymous_clinical.csv", records)
    write_csv("availability.csv", availability)
    write_csv("attrition_reasons.csv", reason_rows)
    write_csv("descriptive_statistics.csv", descriptive)
    write_csv("paired_tests.csv", paired_rows)
    write_csv("between_group_tests.csv", between_rows)
    write_csv("ancova.csv", ancovas)
    write_csv("attrition_baseline.csv", attrition_baseline)
    write_csv("interaction_sensitivity.csv", interactions)
    lines = ["# 临床随访数据独立审计", "", "审计不修改原始工作簿。姓名与住院标识仅在内存中用于匹配，输出使用 P001–P045 匿名编号。", "", "## 数据结构与队列", "",
             "- 总临床表共45人：对照组20人、治疗组25人。NIHSS、ADL、MMSE、HAMD、VFT各有两列数值，45人均完整。ESRS只有一列。",
             "- 只有MMSE与VFT有标为“一个月”的第三列，两量表均为同一28人完整：对照组12人、治疗组16人。",
             "- 28人随访表可按住院标识与姓名分别唯一匹配总表；组别全部一致；168个MMSE/VFT数值全部一致；无重复或冲突。",
             "- 28人表示临床随访完整子集，不能称为影像队列。论文的42人（18/24）与37人分析队列无法由这两份临床表重建。",
             "- 第一、二列未标明治疗前/后，本次暂按论文顺序解释为baseline/post。请核实后将表头补齐。表内“一个月”没有日期，也未明确相对于干预结束还是入组。", "", "## 随访未获得的发生与原因", "",
             f"总计17/45（{100*17/45:.1f}%）没有两量表的一月随访：对照8/20（40.0%），治疗9/25（36.0%）。两组随访可用率Fisher精确检验P={fisher.pvalue:.3f}；该检验不能证明不存在选择偏倚。", "",
             "| 表内记录 | 对照 | 治疗 | 合计 |", "|---|---:|---:|---:|"]
    lines += [f"| {r['recorded_text']} | {r['Control']} | {r['TMS']} | {r['All']} |" for r in reason_rows]
    lines += ["", "上述记录支持‘随访未获得的已记录原因’。资料没有事件发生日期、严重程度、治疗暴露关系或因果判定，因此不得称其为TMS不良事件，也不得断言与TMS无关。外地联系不上与医疗事件应分别报告。", "", "## 45人完整临床数据的治疗前后结果", "", "此队列包含20名常规康复对照和25名常规康复+rTMS治疗者；它与论文影像42人队列是不同分析样本。下面的治疗前/后仍以未标注列的顺序假设为前提。", "", "| 量表 | 组别 | n | 假定治疗前均值±SD | 假定治疗后均值±SD | 有利方向改善均值±SD |", "|---|---|---:|---:|---:|---:|"]
    for scale in ["NIHSS", "ADL", "HAMD", "MMSE", "VFT"]:
        for group in ["Control", "TMS"]:
            rr = [r for r in records if r["group"] == group]
            pre = describe([r[f"{scale}_baseline"] for r in rr])
            post = describe([r[f"{scale}_post"] for r in rr])
            sign = -1 if scale in ["NIHSS", "HAMD"] else 1
            change = describe([sign*(r[f"{scale}_post"]-r[f"{scale}_baseline"]) for r in rr])
            lines.append(f"| {scale} | {group} | {len(rr)} | {pre['mean']:.2f}±{pre['sd']:.2f} | {post['mean']:.2f}±{post['sd']:.2f} | {change['mean']:.2f}±{change['sd']:.2f} |")
    lines += ["", "治疗后 = 组别 + 对应治疗前分数；五终点组成BH家族。", "", "| 终点 | β组别 | 95% CI | P | BH q | HC3 P | HC3 q |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in ancovas:
        if row["cohort"] == "all_45_audit_only":
            lines.append(f"| {row['measure']} | {row['estimate_TMS_minus_Control']:.3f} | {row['ci_low']:.3f} to {row['ci_high']:.3f} | {fmt(row['p'])} | {fmt(row['q'])} | {fmt(row['hc3_p'])} | {fmt(row['hc3_q'])} |")
    lines += ["", "## 28人完整病例的随访描述", "", "以下baseline与post均仅包含有一月随访的同一28人，不能把45人的治疗前均值与28人的随访均值当作配对变化。", "", "| 量表 | 组别 | n | 假定治疗前均值±SD | 假定治疗后均值±SD | 一月随访均值±SD |", "|---|---|---:|---:|---:|---:|"]
    for scale in ["MMSE", "VFT"]:
        for group in ["Control", "TMS"]:
            rr = [r for r in retained if r["group"] == group]
            dd = [describe([r[f"{scale}_{t}"] for r in rr]) for t in ["baseline", "post", "followup"]]
            lines.append(f"| {scale} | {group} | {len(rr)} | " + " | ".join(f"{d['mean']:.2f}±{d['sd']:.2f}" for d in dd) + " |")
    lines += ["", "## 基线调整后的组间随访比较", "", "模型为随访分数 = 截距 + 组别（TMS=1） + 对应基线分数。两终点组成一个BH校正家族。估计为TMS减Control，95% CI为常规OLS t区间（df=25）。", "", "| 终点 | β组别 | 95% CI | P | BH q | HC3 P | HC3 q |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in ancovas:
        if row["cohort"] == "followup_complete_28" and row["model"] == "followup ~ group + baseline":
            lines.append(f"| {row['measure']} | {row['estimate_TMS_minus_Control']:.3f} | {row['ci_low']:.3f} to {row['ci_high']:.3f} | {fmt(row['p'])} | {fmt(row['q'])} | {fmt(row['hc3_p'])} | {fmt(row['hc3_q'])} |")
    lines += ["", "HC3敏感性结果使用同一残差自由度的t参考分布。此完整病例分析不能恢复缺失者的临床结局，不能作为意向性治疗估计。", "", "## 干预结束后的变化与维持", "", "配对t检验报告随访减治疗后变化。MMSE与VFT×两个组构成四次比较的BH家族；Wilcoxon符号秩为辅助敏感性检验，结果见CSV，避免根据显著性选择检验。", "", "| 量表 | 组别 | n | 随访减治疗后 | 95% CI | P | BH q |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in paired_rows:
        if row["cohort"] == "followup_complete_28" and row["comparison"] == "followup_minus_post":
            lines.append(f"| {row['measure']} | {row['group']} | {row['n']} | {row['mean_change']:.3f} | {row['ci_low']:.3f} to {row['ci_high']:.3f} | {fmt(row['p'])} | {fmt(row['q'])} |")
    lines += ["", "随访≈治疗后并不能仅凭P>0.05证明‘效果维持’或等效。须呈现变化与区间，并以完整病例限制解释。", "", "治疗后分数调整模型分析随访组间差异，回答在相同治疗后分数条件下的一月结局差异。它与基线调整的总随访获益是不同问题，且调整治疗后分数可能调整掉部分干预关联。", "", "| 终点 | 调整post的β组别 | 95% CI | P | BH q | HC3 P |", "|---|---:|---:|---:|---:|---:|"]
    for row in ancovas:
        if row["cohort"] == "followup_complete_28" and row["model"] == "followup ~ group + post":
            lines.append(f"| {row['measure']} | {row['estimate_TMS_minus_Control']:.3f} | {row['ci_low']:.3f} to {row['ci_high']:.3f} | {fmt(row['p'])} | {fmt(row['q'])} | {fmt(row['hc3_p'])} |")
    lines += ["", "## 选择偏倚与模型敏感性", "",
              "有随访者与无随访者的基线NIHSS分别为9.07和7.06（标准化差异0.67），ADL分别为35.00和46.18（标准化差异−0.73）。各组分别比较见attrition_baseline.csv；这进一步限制完整病例的推广。",
              "另拟合探索性group×centered-baseline交互模型，结果见interaction_sensitivity.csv。在28人数据中，采用HC3时MMSE的组效应（在全组平均baseline=21.39）为2.50，95% CI −0.08至5.09，P=0.0567；VFT的对应组效应（平均baseline=7.96）为2.87，95% CI 2.06至3.69，P=1.49×10⁻⁷。这与MMSE稳健性不足、VFT关联较稳定的主要解释一致。两随访交互项的HC3 BH q为0.196和0.0657，不作确认性效应异质性结论。",
              "45人治疗前后数据的HAMD组别×baseline交互强（HC3 P=4.40×10⁻¹⁴，五终点BH q=2.20×10⁻¹³）；共同斜率HAMD ANCOVA仅作模型平均调整关联。交互模型在平均baseline时的HAMD组效应为−3.82，方向与简单ANCOVA一致。", "",
              "## 对论文修改的约束", "",
             "1. 在Participants、Clinical assessments、Results、Figure 3a图注和统计方法中明确分母，分开45人临床表、42人主影像队列与28人随访完整子集。",
             f"2. 原42人临床表存在来源冲突：25名治疗者的ADL post总分{sum(tms_adl)}，原稿24人均值77.29意味着排除者ADL post约{implied_removed:.2f}（考虑四舍五入仍为120上下），而原表最大值{max(tms_adl)}，因此原表任意排除1人都不能产生该均值。不能把原42人临床结果当作已复核；本次提供可复现45人临床统计，建议明确以完整临床45人为临床分析队列并替换原临床表及叙述，42人影像结果另列。若坚持42人临床分析，须提供相应源数据/ID匹配名单重新计算。",
             "3. 添加一月随访仅限MMSE和VFT，列出12/16与8/9缺失人数和分组原因。",
             "4. 在统计方法写明complete-case、无填补、ANCOVA两随访终点的BH家族、配对两量表×两组的四比较BH家族、基线调整模型和配对辅助检验，避免从未显著结果推断等效或无偏。",
             "5. 数据没有预设随访终点/缺失处理方案资料，随访分析应标为exploratory。",
             "6. 用户已确认实际干预4周、20次，引言2周应改为4周；‘一个月’的起算点仍需根据原始研究记录确认。", "",
             "## 可复现性", "", "可复现脚本为 clinical_followup_audit.py；所有CSV/JSON均由此脚本生成。软件版本见analysis_results.json。原始文件SHA-256见workbook_structure.json，原文件未改写。",
             "分组映射：对照组→Control、治疗组→TMS；缺失保留为None，不转成0、不做LOCF。值域仅检查明显不可能值，不对VFT未知版本或ADL未说明版本做自动更正。"]
    (BASE / "findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(native({"full_n": len(records), "followup_n": len(retained), "reconciliation_ok": True,
                             "attrition": attrition, "followup_ancova": [r for r in ancovas if r["cohort"] == "followup_complete_28"],
                             "followup_descriptive": [d for d in descriptive if d["cohort"] == "followup_complete_28" and d["measure"] in ["MMSE", "VFT"] and d["group"] != "All"]}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
