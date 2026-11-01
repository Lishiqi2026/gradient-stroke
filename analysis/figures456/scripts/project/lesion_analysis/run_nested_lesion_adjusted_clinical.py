from __future__ import annotations
import os as _portable_os
from pathlib import Path as _PortablePath

def _portable_path(kind, relative):
    project = _PortablePath(_portable_os.environ['PAPER_PROJECT_ROOT'])
    root = project if kind == 'project' else project.parent
    return str(root / relative)
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
ROOT = Path(_portable_path('project', 'lesion_analysis'))
INPUT_PATH = ROOT / 'lesion_adjusted_gradient_value' / 'tables' / 'analysis_cohort.csv'
OUTPUT_ROOT = ROOT / 'nested_lesion_adjusted_clinical'
TABLES = OUTPUT_ROOT / 'tables'
FIGURES = OUTPUT_ROOT / 'figures'
SEED = 20260804
GROUP_CONFLICT_ID = 'PRIVATE_SUBJECT_038'
MANUAL_REVIEW_FLAGS = {'PRIVATE_SUBJECT_040', 'PRIVATE_SUBJECT_006', 'PRIVATE_SUBJECT_038'}
OUTCOMES = {'NIHSS': {'pre': 'NIHSS_pre', 'post': 'NIHSS_post', 'favorable': 'negative'}, 'ADL': {'pre': 'ADL_pre', 'post': 'ADL_post', 'favorable': 'positive'}}
MODEL_TERMS = {'M1_clinical': ['baseline_z', 'group_tms', 'age_z', 'sex_male', 'esrs_z'], 'M2_volume_side': ['baseline_z', 'group_tms', 'age_z', 'sex_male', 'esrs_z', 'log_lesion_volume_z', 'laterality_z'], 'M3_topology': ['baseline_z', 'group_tms', 'age_z', 'sex_male', 'esrs_z', 'log_lesion_volume_z', 'laterality_z', 'brainstem_burden_z', 'deep_cortical_burden_z']}
MODEL_LABELS = {'M1_clinical': 'M1  Clinical', 'M2_volume_side': 'M2  + volume/side', 'M3_topology': 'M3  + topology'}

def zscore(values: pd.Series) -> pd.Series:
    numeric = values.astype(float)
    sd = float(numeric.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        raise RuntimeError(f'Cannot standardize {values.name}')
    return (numeric - float(numeric.mean())) / sd

def load_data() -> pd.DataFrame:
    data = pd.read_csv(INPUT_PATH, dtype={'subject_id': str})
    data['subject_id'] = data['subject_id'].astype(str)
    deep = np.expm1(data['log1p_deep_subcortical_gray_fraction_pct'].astype(float))
    cortex = np.expm1(data['log1p_cerebral_cortex_fraction_pct'].astype(float))
    data['deep_cortical_fraction_pct'] = deep + cortex
    data['log1p_deep_cortical_fraction_pct'] = np.log1p(data['deep_cortical_fraction_pct'])
    data['lesion_volume_ml'] = np.expm1(data['log1p_lesion_volume'].astype(float))
    data['brainstem_fraction_pct'] = np.expm1(data['log1p_brainstem_fraction_pct'].astype(float))
    data['deep_gray_fraction_pct'] = deep
    data['cortical_fraction_pct'] = cortex
    data['manual_review_flag'] = data['subject_id'].isin(MANUAL_REVIEW_FLAGS)
    if len(data) != 38 or data['subject_id'].nunique() != 38:
        raise RuntimeError(f'Unexpected clinical-lesion cohort: {len(data)}')
    if int(data['group_tms'].sum()) != 22:
        raise RuntimeError('Unexpected primary treatment assignment')
    return data.sort_values('subject_id').reset_index(drop=True)

def cohort_variants(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    primary = data.copy()
    exclude_conflict = data.loc[data['subject_id'] != GROUP_CONFLICT_ID].copy()
    reclassify = data.copy()
    reclassify.loc[reclassify['subject_id'] == GROUP_CONFLICT_ID, 'group_tms'] = 0
    reclassify.loc[reclassify['subject_id'] == GROUP_CONFLICT_ID, 'group'] = 'Control'
    exclude_qc = data.loc[~data['manual_review_flag']].copy()
    return {'primary_clinical_grouping': primary, 'exclude_group_conflict': exclude_conflict, 'reclassify_conflict_as_control': reclassify, 'exclude_manual_review_flags': exclude_qc}

def build_design(frame: pd.DataFrame, outcome: str, model_id: str) -> pd.DataFrame:
    design = pd.DataFrame({'baseline_z': zscore(frame[OUTCOMES[outcome]['pre']]), 'group_tms': frame['group_tms'].astype(float), 'age_z': zscore(frame['age']), 'sex_male': frame['sex_male'].astype(float), 'esrs_z': zscore(frame['esrs']), 'log_lesion_volume_z': zscore(frame['log1p_lesion_volume']), 'laterality_z': zscore(frame['mask_lateralization_index']), 'brainstem_burden_z': zscore(frame['log1p_brainstem_fraction_pct']), 'deep_cortical_burden_z': zscore(frame['log1p_deep_cortical_fraction_pct'])}, index=frame.index)
    return sm.add_constant(design[MODEL_TERMS[model_id]], has_constant='add')

def freedman_lane_group_test(y: np.ndarray, full_x: pd.DataFrame, n_permutations: int, rng: np.random.Generator) -> float:
    reduced_x = full_x.drop(columns='group_tms')
    reduced = sm.OLS(y, reduced_x).fit()
    observed = float(sm.OLS(y, full_x).fit().params['group_tms'])
    fitted = reduced.fittedvalues.to_numpy(float)
    residuals = reduced.resid.to_numpy(float)
    null = np.empty(n_permutations, dtype=float)
    for index in range(n_permutations):
        y_permuted = fitted + rng.permutation(residuals)
        null[index] = float(sm.OLS(y_permuted, full_x).fit().params['group_tms'])
    return float((1 + np.count_nonzero(np.abs(null) >= abs(observed))) / (n_permutations + 1))

def joint_block_p(result, terms: list[str]) -> float:
    if not terms:
        return np.nan
    restriction = np.zeros((len(terms), len(result.params)), dtype=float)
    names = list(result.params.index)
    for (row, term) in enumerate(terms):
        restriction[row, names.index(term)] = 1.0
    return float(result.wald_test(restriction, scalar=True).pvalue)

def max_vif(design: pd.DataFrame) -> float:
    matrix = design.to_numpy(float)
    values = []
    for (index, name) in enumerate(design.columns):
        if name == 'const':
            continue
        values.append(float(variance_inflation_factor(matrix, index)))
    return max(values)

def fit_models(cohorts: dict[str, pd.DataFrame], n_permutations: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    treatment_rows = []
    coefficient_rows = []
    counter = 0
    for (cohort_name, frame) in cohorts.items():
        for (outcome, outcome_info) in OUTCOMES.items():
            m1_beta = None
            for model_id in MODEL_TERMS:
                design = build_design(frame, outcome, model_id)
                y = frame[outcome_info['post']].to_numpy(float)
                result = sm.OLS(y, design).fit(cov_type='HC3')
                permutation_p = freedman_lane_group_test(y, design, n_permutations, np.random.default_rng(SEED + counter * 10000))
                counter += 1
                beta = float(result.params['group_tms'])
                if m1_beta is None:
                    m1_beta = beta
                confidence = result.conf_int().loc['group_tms']
                if model_id == 'M2_volume_side':
                    block_terms = ['log_lesion_volume_z', 'laterality_z']
                elif model_id == 'M3_topology':
                    block_terms = ['brainstem_burden_z', 'deep_cortical_burden_z']
                else:
                    block_terms = []
                reduced = sm.OLS(y, design.drop(columns='group_tms')).fit()
                rss_full = float(np.sum(result.resid ** 2))
                rss_reduced = float(np.sum(reduced.resid ** 2))
                partial_r2 = max((rss_reduced - rss_full) / rss_reduced, 0.0)
                treatment_rows.append({'cohort': cohort_name, 'outcome': outcome, 'model_id': model_id, 'model_label': MODEL_LABELS[model_id], 'n': int(result.nobs), 'control_n': int((frame['group_tms'] == 0).sum()), 'tms_n': int((frame['group_tms'] == 1).sum()), 'treatment_beta_raw_score': beta, 'ci_low_hc3': float(confidence.iloc[0]), 'ci_high_hc3': float(confidence.iloc[1]), 'se_hc3': float(result.bse['group_tms']), 'p_hc3': float(result.pvalues['group_tms']), 'p_freedman_lane': permutation_p, 'treatment_partial_r2': partial_r2, 'beta_change_from_M1': beta - m1_beta, 'absolute_beta_percent_change_from_M1': 100.0 * (abs(beta) - abs(m1_beta)) / abs(m1_beta), 'added_lesion_block': ';'.join(block_terms), 'added_lesion_block_wald_p_hc3': joint_block_p(result, block_terms), 'r_squared': float(result.rsquared), 'adjusted_r_squared': float(result.rsquared_adj), 'max_vif': max_vif(design), 'favorable_treatment_direction': outcome_info['favorable'], 'n_permutations': n_permutations})
                for term in result.params.index:
                    interval = result.conf_int().loc[term]
                    coefficient_rows.append({'cohort': cohort_name, 'outcome': outcome, 'model_id': model_id, 'term': term, 'beta': float(result.params[term]), 'se_hc3': float(result.bse[term]), 'ci_low_hc3': float(interval.iloc[0]), 'ci_high_hc3': float(interval.iloc[1]), 'p_hc3': float(result.pvalues[term])})
    return (pd.DataFrame(treatment_rows), pd.DataFrame(coefficient_rows))

def lesion_group_table(data: pd.DataFrame) -> pd.DataFrame:
    metrics = [('lesion_volume_ml', 'Lesion volume (ml)'), ('mask_lateralization_index', 'Mask lateralization index'), ('brainstem_fraction_pct', 'Brainstem fraction (%)'), ('deep_gray_fraction_pct', 'Deep-gray fraction (%)'), ('cortical_fraction_pct', 'Cortical fraction (%)'), ('deep_cortical_fraction_pct', 'Deep-gray + cortical fraction (%)')]
    rows = []
    for (column, label) in metrics:
        control = data.loc[data['group_tms'] == 0, column].astype(float)
        tms = data.loc[data['group_tms'] == 1, column].astype(float)
        pooled_sd = np.sqrt(((len(control) - 1) * control.var(ddof=1) + (len(tms) - 1) * tms.var(ddof=1)) / (len(control) + len(tms) - 2))
        rows.append({'metric': column, 'label': label, 'control_n': len(control), 'control_mean': float(control.mean()), 'control_sd': float(control.std(ddof=1)), 'control_median': float(control.median()), 'tms_n': len(tms), 'tms_mean': float(tms.mean()), 'tms_sd': float(tms.std(ddof=1)), 'tms_median': float(tms.median()), 'smd_tms_minus_control': float((tms.mean() - control.mean()) / pooled_sd) if pooled_sd > 0 else np.nan})
    return pd.DataFrame(rows)

def leave_one_out_primary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (outcome, outcome_info) in OUTCOMES.items():
        for model_id in MODEL_TERMS:
            for subject_id in data['subject_id']:
                frame = data.loc[data['subject_id'] != subject_id].copy()
                design = build_design(frame, outcome, model_id)
                result = sm.OLS(frame[outcome_info['post']].to_numpy(float), design).fit(cov_type='HC3')
                confidence = result.conf_int().loc['group_tms']
                rows.append({'outcome': outcome, 'model_id': model_id, 'excluded_subject_id': subject_id, 'excluded_group': data.loc[data['subject_id'] == subject_id, 'group'].iloc[0], 'excluded_manual_review_flag': bool(data.loc[data['subject_id'] == subject_id, 'manual_review_flag'].iloc[0]), 'treatment_beta_raw_score': float(result.params['group_tms']), 'ci_low_hc3': float(confidence.iloc[0]), 'ci_high_hc3': float(confidence.iloc[1]), 'p_hc3': float(result.pvalues['group_tms'])})
    return pd.DataFrame(rows)

def forest_plot(treatment: pd.DataFrame) -> None:
    primary = treatment.loc[treatment['cohort'] == 'primary_clinical_grouping'].copy()
    colors = {'M1_clinical': '#666666', 'M2_volume_side': '#4C78A8', 'M3_topology': '#B55D45'}
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'], 'font.size': 7, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    (figure, axes) = plt.subplots(2, 1, figsize=(183 / 25.4, 120 / 25.4))
    model_order = list(MODEL_TERMS)
    y = np.arange(len(model_order))[::-1]
    for (axis, outcome, panel) in zip(axes, OUTCOMES, ['a', 'b']):
        frame = primary.loc[primary['outcome'] == outcome].set_index('model_id')
        for (position, model_id) in zip(y, model_order):
            row = frame.loc[model_id]
            axis.plot([row['ci_low_hc3'], row['ci_high_hc3']], [position, position], color=colors[model_id], linewidth=1.4)
            axis.scatter(row['treatment_beta_raw_score'], position, s=28, color=colors[model_id], edgecolor='white', linewidth=0.5, zorder=3)
            axis.text(0.98, position, f"{row['treatment_beta_raw_score']:.2f} [{row['ci_low_hc3']:.2f}, {row['ci_high_hc3']:.2f}]", transform=axis.get_yaxis_transform(), ha='right', va='center', fontsize=5.6)
        axis.axvline(0, color='#555555', linestyle='--', linewidth=0.8)
        axis.set_yticks(y, [MODEL_LABELS[model] for model in model_order])
        axis.set_xlabel(f'Adjusted TMS - Control difference in post {outcome}')
        direction = 'negative favours TMS' if outcome == 'NIHSS' else 'positive favours TMS'
        axis.set_title(f'{outcome} ({direction})', loc='left', fontweight='bold')
        axis.text(-0.14, 1.04, panel, transform=axis.transAxes, fontsize=9, fontweight='bold')
    figure.subplots_adjust(left=0.22, right=0.985, bottom=0.11, top=0.94, hspace=0.72)
    for suffix in ('svg', 'pdf', 'png'):
        figure.savefig(FIGURES / f'nested_lesion_adjusted_treatment_coefficients.{suffix}', dpi=300 if suffix == 'png' else None)
    plt.close(figure)

def conflict_sensitivity_plot(treatment: pd.DataFrame) -> None:
    selected = treatment.loc[treatment['model_id'] == 'M3_topology'].copy()
    cohort_order = ['primary_clinical_grouping', 'exclude_group_conflict', 'reclassify_conflict_as_control', 'exclude_manual_review_flags']
    labels = ['Primary', 'Exclude conflict', 'Conflict -> Control', 'Exclude lesion QC']
    colors = ['#B55D45', '#4C78A8', '#8A6FA8', '#777777']
    (figure, axes) = plt.subplots(2, 1, figsize=(183 / 25.4, 120 / 25.4))
    y = np.arange(len(cohort_order))[::-1]
    for (axis, outcome, panel) in zip(axes, OUTCOMES, ['a', 'b']):
        frame = selected.loc[selected['outcome'] == outcome].set_index('cohort')
        for (position, cohort, color) in zip(y, cohort_order, colors):
            row = frame.loc[cohort]
            axis.plot([row['ci_low_hc3'], row['ci_high_hc3']], [position, position], color=color, linewidth=1.3)
            axis.scatter(row['treatment_beta_raw_score'], position, s=25, color=color, edgecolor='white', linewidth=0.45, zorder=3)
        axis.axvline(0, color='#555555', linestyle='--', linewidth=0.8)
        axis.set_yticks(y, labels)
        axis.set_xlabel(f'M3 adjusted TMS - Control difference in post {outcome}')
        axis.set_title(outcome, loc='left', fontweight='bold')
        axis.text(-0.14, 1.04, panel, transform=axis.transAxes, fontsize=9, fontweight='bold')
    figure.subplots_adjust(left=0.22, right=0.985, bottom=0.11, top=0.94, hspace=0.72)
    for suffix in ('svg', 'pdf', 'png'):
        figure.savefig(FIGURES / f'group_assignment_sensitivity.{suffix}', dpi=300 if suffix == 'png' else None)
    plt.close(figure)

def write_summary(treatment: pd.DataFrame, loo: pd.DataFrame) -> None:
    primary = treatment.loc[treatment['cohort'] == 'primary_clinical_grouping']

    def row(outcome: str, model_id: str) -> pd.Series:
        return primary.loc[(primary['outcome'] == outcome) & (primary['model_id'] == model_id)].iloc[0]
    nihss = [row('NIHSS', model) for model in MODEL_TERMS]
    adl = [row('ADL', model) for model in MODEL_TERMS]
    conflict = treatment.loc[treatment['model_id'] == 'M3_topology', ['cohort', 'outcome', 'treatment_beta_raw_score', 'ci_low_hc3', 'ci_high_hc3', 'p_hc3', 'p_freedman_lane']]
    conflict_header = '| ' + ' | '.join(conflict.columns) + ' |'
    conflict_separator = '| ' + ' | '.join(['---'] * len(conflict.columns)) + ' |'
    conflict_rows = []
    for record in conflict.itertuples(index=False, name=None):
        values = [f'{value:.4f}' if isinstance(value, (float, np.floating)) else str(value) for value in record]
        conflict_rows.append('| ' + ' | '.join(values) + ' |')
    conflict_markdown = '\n'.join([conflict_header, conflict_separator, *conflict_rows])
    text = f"# 病灶校正后的TMS相关临床差异\n\n## 方法\n\n- 共同队列38例：TMS 22例、Control 16例；NIHSS和ADL均完整。\n- M1：治疗后评分 ~ 治疗前评分 + Treatment + age + sex + ESRS。\n- M2：M1 + log病灶体积 + MNI mask侧化指数。\n- M3：M2 + 脑干受累 + 深部灰质/皮层联合受累。\n- Treatment编码为TMS=1、Control=0；NIHSS负系数和ADL正系数倾向TMS组结局更好。\n- 报告原始量表分数、HC3稳健置信区间及5,000次Freedman-Lane残差置换。\n\n## NIHSS\n\n- M1：TMS-Control={nihss[0]['treatment_beta_raw_score']:.3f}分，95% CI {nihss[0]['ci_low_hc3']:.3f}至{nihss[0]['ci_high_hc3']:.3f}，置换p={nihss[0]['p_freedman_lane']:.4f}。\n- M2：TMS-Control={nihss[1]['treatment_beta_raw_score']:.3f}分，95% CI {nihss[1]['ci_low_hc3']:.3f}至{nihss[1]['ci_high_hc3']:.3f}，相对M1绝对系数变化{nihss[1]['absolute_beta_percent_change_from_M1']:.1f}%。\n- M3：TMS-Control={nihss[2]['treatment_beta_raw_score']:.3f}分，95% CI {nihss[2]['ci_low_hc3']:.3f}至{nihss[2]['ci_high_hc3']:.3f}，相对M1绝对系数变化{nihss[2]['absolute_beta_percent_change_from_M1']:.1f}%，置换p={nihss[2]['p_freedman_lane']:.4f}。\n\n## ADL\n\n- M1：TMS-Control={adl[0]['treatment_beta_raw_score']:.3f}分，95% CI {adl[0]['ci_low_hc3']:.3f}至{adl[0]['ci_high_hc3']:.3f}，置换p={adl[0]['p_freedman_lane']:.4f}。\n- M2：TMS-Control={adl[1]['treatment_beta_raw_score']:.3f}分，95% CI {adl[1]['ci_low_hc3']:.3f}至{adl[1]['ci_high_hc3']:.3f}，相对M1绝对系数变化{adl[1]['absolute_beta_percent_change_from_M1']:.1f}%。\n- M3：TMS-Control={adl[2]['treatment_beta_raw_score']:.3f}分，95% CI {adl[2]['ci_low_hc3']:.3f}至{adl[2]['ci_high_hc3']:.3f}，相对M1绝对系数变化{adl[2]['absolute_beta_percent_change_from_M1']:.1f}%，置换p={adl[2]['p_freedman_lane']:.4f}。\n\n## 解释\n\n- M2新增的病灶体积/侧别块在NIHSS和ADL中的联合Wald p分别为{nihss[1]['added_lesion_block_wald_p_hc3']:.4f}和{adl[1]['added_lesion_block_wald_p_hc3']:.4f}，Treatment系数仅变化{nihss[1]['absolute_beta_percent_change_from_M1']:.1f}%和{adl[1]['absolute_beta_percent_change_from_M1']:.1f}%。这组数据不支持病灶体积和侧别解释掉主要组间差异。\n- M3加入拓扑后，Treatment优势没有衰减，反而增至NIHSS {nihss[2]['treatment_beta_raw_score']:.2f}分和ADL {adl[2]['treatment_beta_raw_score']:.2f}分。拓扑块对NIHSS的联合p={nihss[2]['added_lesion_block_wald_p_hc3']:.4f}，对ADL的联合p={adl[2]['added_lesion_block_wald_p_hc3']:.6f}。\n- M3最大VIF为NIHSS {nihss[2]['max_vif']:.2f}、ADL {adl[2]['max_vif']:.2f}。病灶拓扑与组别明显不平衡，因此M3应作为敏感性模型，主结论优先依据M1与M2的一致性。\n\n## 分组冲突与稳健性\n\n`PRIVATE_SUBJECT_038`在Raw-data目录属于Control，但当前临床表和所有病灶/梯度派生数据中属于TMS。主分析沿用临床/派生口径，并同时输出排除该例、将其改判Control以及排除3例病灶QC病例的M3敏感性结果。\n\n{conflict_markdown}\n\n逐例删除M3时，NIHSS的Treatment系数范围为{loo.loc[loo['outcome'].eq('NIHSS'), 'treatment_beta_raw_score'].min():.3f}至{loo.loc[loo['outcome'].eq('NIHSS'), 'treatment_beta_raw_score'].max():.3f}，ADL为{loo.loc[loo['outcome'].eq('ADL'), 'treatment_beta_raw_score'].min():.3f}至{loo.loc[loo['outcome'].eq('ADL'), 'treatment_beta_raw_score'].max():.3f}。完整结果保存在`tables/leave_one_out_primary.csv`。\n\n这些分析是病灶校正后的观察性组间差异，不应写成已证明的TMS因果效应。正式定稿前还必须核实`PRIVATE_SUBJECT_038`的真实分组，以及研究是否随机分组、Control是否为假刺激。\n"
    (OUTPUT_ROOT / 'RESULTS_ZH.md').write_text(text, encoding='utf-8')

def run(n_permutations: int) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    data = load_data()
    cohorts = cohort_variants(data)
    (treatment, coefficients) = fit_models(cohorts, n_permutations)
    lesion_groups = lesion_group_table(data)
    loo = leave_one_out_primary(data)
    data.to_csv(TABLES / 'analysis_cohort_38.csv', index=False, encoding='utf-8-sig')
    treatment.to_csv(TABLES / 'treatment_coefficient_comparison.csv', index=False, encoding='utf-8-sig')
    coefficients.to_csv(TABLES / 'all_model_coefficients.csv', index=False, encoding='utf-8-sig')
    lesion_groups.to_csv(TABLES / 'lesion_group_characteristics.csv', index=False, encoding='utf-8-sig')
    loo.to_csv(TABLES / 'leave_one_out_primary.csv', index=False, encoding='utf-8-sig')
    forest_plot(treatment)
    conflict_sensitivity_plot(treatment)
    write_summary(treatment, loo)
    manifest = {'analysis_date': '2026-08-04', 'primary_n': 38, 'primary_groups': {'TMS': 22, 'Control': 16}, 'outcomes': ['NIHSS_post', 'ADL_post'], 'models': MODEL_TERMS, 'topology_definition': 'brainstem log fraction and log1p(sum of deep-gray and cortical lesion fractions)', 'continuous_covariates': 'z-scored within each analysis cohort; outcome remains in raw score units', 'robust_se': 'HC3', 'permutation': f'Freedman-Lane residual permutation under the no-group reduced model; {n_permutations} permutations', 'group_conflict': {'subject_id': GROUP_CONFLICT_ID, 'raw_data_folder': 'Control', 'clinical_and_derivative_tables': 'TMS', 'primary_rule': 'use clinical and derivative grouping', 'sensitivities': ['exclude subject', 'reclassify as Control']}, 'interpretation': 'adjusted observational group differences, not proof of causal treatment effects', 'random_seed': SEED}
    (OUTPUT_ROOT / 'analysis_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(treatment.loc[treatment['cohort'] == 'primary_clinical_grouping', ['outcome', 'model_id', 'n', 'treatment_beta_raw_score', 'ci_low_hc3', 'ci_high_hc3', 'p_hc3', 'p_freedman_lane', 'absolute_beta_percent_change_from_M1', 'added_lesion_block_wald_p_hc3', 'max_vif']].to_string(index=False))

def main() -> None:
    parser = argparse.ArgumentParser(description='Compare nested clinical, lesion-volume and lesion-topology adjustment models.')
    parser.add_argument('--permutations', type=int, default=5000)
    arguments = parser.parse_args()
    if arguments.permutations < 1:
        raise ValueError('permutations must be positive')
    run(arguments.permutations)
if __name__ == '__main__':
    main()
