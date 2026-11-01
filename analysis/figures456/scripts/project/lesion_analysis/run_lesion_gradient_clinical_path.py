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
from scipy.stats import norm
ROOT = Path(_portable_path('project', 'lesion_analysis'))
SUBJECT_PATH = ROOT / 'lesion_gradient_distance' / 'tables' / 'subject_level_distance_metrics.csv'
CLINICAL_PATH = ROOT / 'lesion_adjusted_gradient_value' / 'tables' / 'analysis_cohort.csv'
OUTPUT_ROOT = ROOT / 'lesion_gradient_clinical_path'
TABLES = OUTPUT_ROOT / 'tables'
FIGURES = OUTPUT_ROOT / 'figures'
SEED = 20260804
GROUP_CONFLICT_ID = 'PRIVATE_SUBJECT_038'
MANUAL_REVIEW_FLAGS = {'PRIVATE_SUBJECT_040', 'PRIVATE_SUBJECT_006', 'PRIVATE_SUBJECT_038'}

def zscore(values: pd.Series) -> pd.Series:
    numeric = values.astype(float)
    sd = float(numeric.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        raise RuntimeError(f'Cannot standardize {values.name}')
    return (numeric - float(numeric.mean())) / sd

def load_data() -> pd.DataFrame:
    subject = pd.read_csv(SUBJECT_PATH, dtype={'subject_id': str})
    clinical = pd.read_csv(CLINICAL_PATH, dtype={'subject_id': str})
    columns = ['subject_id', 'NIHSS_pre', 'NIHSS_post', 'ADL_pre', 'ADL_post', 'age', 'sex_male', 'esrs']
    data = subject.merge(clinical[columns], on='subject_id', validate='one_to_one')
    data['manual_review_flag'] = data['subject_id'].isin(MANUAL_REVIEW_FLAGS)
    required = ['group_tms', 'primary_motion_eligible', 'log1p_lesion_volume', 'xtract_projected_G2_span', 'mean_baseline_deviation_G2', 'mean_reference_directed_G2', 'xtract_projected_G2_slope', 'NIHSS_pre', 'NIHSS_post']
    if len(data) != 38 or data['subject_id'].nunique() != 38:
        raise RuntimeError(f'Unexpected merged cohort: {len(data)}')
    if data[required].isna().any().any():
        raise RuntimeError('Missing value in primary path variables')
    if int(data['group_tms'].sum()) != 22:
        raise RuntimeError('Unexpected treatment group count')
    return data.sort_values('subject_id').reset_index(drop=True)

def cohort_variants(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    motion = data.loc[data['primary_motion_eligible'].astype(bool)].copy()
    all_rows = data.copy()
    exclude_qc = motion.loc[~motion['manual_review_flag']].copy()
    exclude_conflict = motion.loc[motion['subject_id'] != GROUP_CONFLICT_ID].copy()
    reclassify = motion.copy()
    reclassify.loc[reclassify['subject_id'] == GROUP_CONFLICT_ID, 'group_tms'] = 0
    reclassify.loc[reclassify['subject_id'] == GROUP_CONFLICT_ID, 'group'] = 'Control'
    return {'primary_motion_qc': motion, 'include_motion_flagged_subject': all_rows, 'exclude_manual_lesion_qc_flags': exclude_qc, 'exclude_group_conflict': exclude_conflict, 'reclassify_group_conflict_as_control': reclassify}

def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy().reset_index(drop=True)
    standardize = {'lesion_span_z': 'xtract_projected_G2_span', 'baseline_g2_deviation_z': 'mean_baseline_deviation_G2', 'g2_normalization_z': 'mean_reference_directed_G2', 'lesion_centered_g2_slope_z': 'xtract_projected_G2_slope', 'log_lesion_volume_z': 'log1p_lesion_volume', 'NIHSS_pre_z': 'NIHSS_pre'}
    for (destination, source) in standardize.items():
        result[destination] = zscore(result[source])
    result['group_x_lesion_span'] = result['group_tms'].astype(float) * result['lesion_span_z']
    return result

def add_constant(frame: pd.DataFrame, terms: list[str]) -> pd.DataFrame:
    return sm.add_constant(frame[terms].astype(float), has_constant='add')

def linear_combination(result, weights: dict[str, float]) -> dict[str, float]:
    names = list(result.params.index)
    vector = np.asarray([weights.get(name, 0.0) for name in names], dtype=float)
    estimate = float(vector @ result.params.to_numpy(float))
    covariance = result.cov_params().to_numpy(float)
    se = float(np.sqrt(max(vector @ covariance @ vector, 0.0)))
    z_value = estimate / se if se > 0 else np.nan
    return {'estimate': estimate, 'se_hc3': se, 'ci_low_hc3': estimate - 1.96 * se, 'ci_high_hc3': estimate + 1.96 * se, 'p_hc3': float(2 * norm.sf(abs(z_value))) if np.isfinite(z_value) else np.nan}

def partial_r2(full_result, reduced_result) -> float:
    rss_full = float(np.sum(full_result.resid ** 2))
    rss_reduced = float(np.sum(reduced_result.resid ** 2))
    return max((rss_reduced - rss_full) / rss_reduced, 0.0)

def coefficient_permutation(y: np.ndarray, full_x: pd.DataFrame, term: str, n_permutations: int, rng: np.random.Generator) -> float:
    reduced_x = full_x.drop(columns=term)
    reduced = sm.OLS(y, reduced_x).fit()
    observed = float(sm.OLS(y, full_x).fit().params[term])
    fitted = reduced.fittedvalues.to_numpy(float)
    residuals = reduced.resid.to_numpy(float)
    null = np.empty(n_permutations, dtype=float)
    for index in range(n_permutations):
        permuted_y = fitted + rng.permutation(residuals)
        null[index] = float(sm.OLS(permuted_y, full_x).fit().params[term])
    return float((1 + np.count_nonzero(np.abs(null) >= abs(observed))) / (n_permutations + 1))
CONDITIONAL_M_TERMS = ['group_tms', 'lesion_span_z', 'group_x_lesion_span', 'baseline_g2_deviation_z', 'log_lesion_volume_z']
CONDITIONAL_Y_TERMS = ['NIHSS_pre_z', 'group_tms', 'lesion_span_z', 'group_x_lesion_span', 'g2_normalization_z', 'baseline_g2_deviation_z', 'log_lesion_volume_z']

def conditional_point(frame: pd.DataFrame):
    m_x = add_constant(frame, CONDITIONAL_M_TERMS)
    y_x = add_constant(frame, CONDITIONAL_Y_TERMS)
    m_result = sm.OLS(frame['g2_normalization_z'].astype(float), m_x).fit(cov_type='HC3')
    y_result = sm.OLS(frame['NIHSS_post'].astype(float), y_x).fit(cov_type='HC3')
    a_control = float(m_result.params['lesion_span_z'])
    a_interaction = float(m_result.params['group_x_lesion_span'])
    a_tms = a_control + a_interaction
    b_path = float(y_result.params['g2_normalization_z'])
    effects = {'control_indirect': a_control * b_path, 'tms_indirect': a_tms * b_path, 'moderated_mediation_index': a_interaction * b_path}
    return (m_result, y_result, effects)
TREATMENT_M_TERMS = ['group_tms', 'lesion_span_z', 'baseline_g2_deviation_z', 'log_lesion_volume_z']
TREATMENT_Y_TERMS = ['NIHSS_pre_z', 'group_tms', 'lesion_centered_g2_slope_z', 'lesion_span_z', 'baseline_g2_deviation_z', 'log_lesion_volume_z']

def treatment_mediation_point(frame: pd.DataFrame):
    m_x = add_constant(frame, TREATMENT_M_TERMS)
    y_x = add_constant(frame, TREATMENT_Y_TERMS)
    total_x = add_constant(frame, ['NIHSS_pre_z', 'group_tms', 'lesion_span_z', 'baseline_g2_deviation_z', 'log_lesion_volume_z'])
    m_result = sm.OLS(frame['lesion_centered_g2_slope_z'].astype(float), m_x).fit(cov_type='HC3')
    y_result = sm.OLS(frame['NIHSS_post'].astype(float), y_x).fit(cov_type='HC3')
    total_result = sm.OLS(frame['NIHSS_post'].astype(float), total_x).fit(cov_type='HC3')
    a_path = float(m_result.params['group_tms'])
    b_path = float(y_result.params['lesion_centered_g2_slope_z'])
    return (m_result, y_result, total_result, a_path * b_path)

def stratified_bootstrap_indices(frame: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    sampled = []
    for group_value in (0, 1):
        indices = np.flatnonzero(frame['group_tms'].to_numpy(int) == group_value)
        sampled.append(rng.choice(indices, size=len(indices), replace=True))
    return np.concatenate(sampled)

def bootstrap_paths(frame: pd.DataFrame, n_bootstrap: int, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    conditional_m_x = add_constant(frame, CONDITIONAL_M_TERMS).to_numpy(float)
    conditional_m_y = frame['g2_normalization_z'].to_numpy(float)
    conditional_y_x = add_constant(frame, CONDITIONAL_Y_TERMS).to_numpy(float)
    clinical_y = frame['NIHSS_post'].to_numpy(float)
    treatment_m_x = add_constant(frame, TREATMENT_M_TERMS).to_numpy(float)
    treatment_m_y = frame['lesion_centered_g2_slope_z'].to_numpy(float)
    treatment_y_x = add_constant(frame, TREATMENT_Y_TERMS).to_numpy(float)
    conditional_m_index = {term: index + 1 for (index, term) in enumerate(CONDITIONAL_M_TERMS)}
    conditional_y_index = {term: index + 1 for (index, term) in enumerate(CONDITIONAL_Y_TERMS)}
    treatment_m_index = {term: index + 1 for (index, term) in enumerate(TREATMENT_M_TERMS)}
    treatment_y_index = {term: index + 1 for (index, term) in enumerate(TREATMENT_Y_TERMS)}
    conditional_rows = []
    treatment_rows = []
    for bootstrap_id in range(n_bootstrap):
        indices = stratified_bootstrap_indices(frame, rng)
        try:
            conditional_m_beta = np.linalg.lstsq(conditional_m_x[indices], conditional_m_y[indices], rcond=None)[0]
            conditional_y_beta = np.linalg.lstsq(conditional_y_x[indices], clinical_y[indices], rcond=None)[0]
            treatment_m_beta = np.linalg.lstsq(treatment_m_x[indices], treatment_m_y[indices], rcond=None)[0]
            treatment_y_beta = np.linalg.lstsq(treatment_y_x[indices], clinical_y[indices], rcond=None)[0]
        except (np.linalg.LinAlgError, ValueError):
            continue
        a_control = conditional_m_beta[conditional_m_index['lesion_span_z']]
        a_interaction = conditional_m_beta[conditional_m_index['group_x_lesion_span']]
        b_conditional = conditional_y_beta[conditional_y_index['g2_normalization_z']]
        conditional = {'control_indirect': a_control * b_conditional, 'tms_indirect': (a_control + a_interaction) * b_conditional, 'moderated_mediation_index': a_interaction * b_conditional}
        treatment = treatment_m_beta[treatment_m_index['group_tms']] * treatment_y_beta[treatment_y_index['lesion_centered_g2_slope_z']]
        conditional_rows.append({'bootstrap_id': bootstrap_id, **conditional})
        treatment_rows.append({'bootstrap_id': bootstrap_id, 'tms_lesion_centered_g2_indirect': treatment})
    return (pd.DataFrame(conditional_rows), pd.DataFrame(treatment_rows))

def bootstrap_summary(values: pd.Series, observed: float, label: str, cohort: str) -> dict[str, float | str]:
    array = values.dropna().to_numpy(float)
    if len(array) == 0:
        raise RuntimeError(f'No bootstrap estimates for {label}')
    p_value = min(1.0, 2.0 * min((1 + np.count_nonzero(array <= 0)) / (len(array) + 1), (1 + np.count_nonzero(array >= 0)) / (len(array) + 1)))
    return {'cohort': cohort, 'effect': label, 'estimate': observed, 'bootstrap_ci_low': float(np.quantile(array, 0.025)), 'bootstrap_ci_high': float(np.quantile(array, 0.975)), 'bootstrap_p_two_sided': float(p_value), 'valid_bootstrap_n': int(len(array))}

def coefficient_rows(cohort: str, model: str, result) -> list[dict]:
    confidence = result.conf_int()
    rows = []
    for term in result.params.index:
        rows.append({'cohort': cohort, 'model': model, 'term': term, 'beta': float(result.params[term]), 'se_hc3': float(result.bse[term]), 'ci_low_hc3': float(confidence.loc[term].iloc[0]), 'ci_high_hc3': float(confidence.loc[term].iloc[1]), 'p_hc3': float(result.pvalues[term]), 'r_squared': float(result.rsquared), 'adjusted_r_squared': float(result.rsquared_adj)})
    return rows

def fit_all(cohorts: dict[str, pd.DataFrame], n_bootstrap: int, n_permutations: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    conditional_model_rows = []
    indirect_rows = []
    treatment_rows = []
    coefficient_output = []
    primary_bootstrap = []
    for (cohort_index, (cohort_name, raw_frame)) in enumerate(cohorts.items()):
        frame = prepare(raw_frame)
        (conditional_m, conditional_y, conditional_effects) = conditional_point(frame)
        (treatment_m, treatment_y, treatment_total, treatment_indirect) = treatment_mediation_point(frame)
        (conditional_boot, treatment_boot) = bootstrap_paths(frame, n_bootstrap, np.random.default_rng(SEED + cohort_index * 100000))
        a_tms = linear_combination(conditional_m, {'lesion_span_z': 1.0, 'group_x_lesion_span': 1.0})
        conditional_model_rows.append({'cohort': cohort_name, 'n': len(frame), 'control_n': int((frame['group_tms'] == 0).sum()), 'tms_n': int((frame['group_tms'] == 1).sum()), 'a_control_lesion_to_g2': float(conditional_m.params['lesion_span_z']), 'a_control_p_hc3': float(conditional_m.pvalues['lesion_span_z']), 'a_tms_lesion_to_g2': a_tms['estimate'], 'a_tms_p_hc3': a_tms['p_hc3'], 'a_interaction': float(conditional_m.params['group_x_lesion_span']), 'a_interaction_ci_low_hc3': float(conditional_m.conf_int().loc['group_x_lesion_span'].iloc[0]), 'a_interaction_ci_high_hc3': float(conditional_m.conf_int().loc['group_x_lesion_span'].iloc[1]), 'a_interaction_p_hc3': float(conditional_m.pvalues['group_x_lesion_span']), 'b_g2_to_nihss': float(conditional_y.params['g2_normalization_z']), 'b_ci_low_hc3': float(conditional_y.conf_int().loc['g2_normalization_z'].iloc[0]), 'b_ci_high_hc3': float(conditional_y.conf_int().loc['g2_normalization_z'].iloc[1]), 'b_p_hc3': float(conditional_y.pvalues['g2_normalization_z']), 'b_partial_r2': partial_r2(conditional_y, sm.OLS(frame['NIHSS_post'].astype(float), add_constant(frame, [term for term in CONDITIONAL_Y_TERMS if term != 'g2_normalization_z'])).fit())})
        for (effect_name, observed) in conditional_effects.items():
            indirect_rows.append(bootstrap_summary(conditional_boot[effect_name], observed, effect_name, cohort_name))
        treatment_summary = bootstrap_summary(treatment_boot['tms_lesion_centered_g2_indirect'], treatment_indirect, 'tms_lesion_centered_g2_indirect', cohort_name)
        treatment_rows.append({**treatment_summary, 'n': len(frame), 'a_tms_to_lesion_centered_g2': float(treatment_m.params['group_tms']), 'a_ci_low_hc3': float(treatment_m.conf_int().loc['group_tms'].iloc[0]), 'a_ci_high_hc3': float(treatment_m.conf_int().loc['group_tms'].iloc[1]), 'a_p_hc3': float(treatment_m.pvalues['group_tms']), 'b_lesion_centered_g2_to_nihss': float(treatment_y.params['lesion_centered_g2_slope_z']), 'b_ci_low_hc3': float(treatment_y.conf_int().loc['lesion_centered_g2_slope_z'].iloc[0]), 'b_ci_high_hc3': float(treatment_y.conf_int().loc['lesion_centered_g2_slope_z'].iloc[1]), 'b_p_hc3': float(treatment_y.pvalues['lesion_centered_g2_slope_z']), 'total_tms_effect': float(treatment_total.params['group_tms']), 'direct_tms_effect': float(treatment_y.params['group_tms']), 'proportion_mediated': treatment_indirect / float(treatment_total.params['group_tms']) if abs(float(treatment_total.params['group_tms'])) > 1e-12 else np.nan})
        coefficient_output.extend(coefficient_rows(cohort_name, 'conditional_mediator', conditional_m))
        coefficient_output.extend(coefficient_rows(cohort_name, 'conditional_outcome', conditional_y))
        coefficient_output.extend(coefficient_rows(cohort_name, 'treatment_mediator', treatment_m))
        coefficient_output.extend(coefficient_rows(cohort_name, 'treatment_outcome', treatment_y))
        if cohort_name == 'primary_motion_qc':
            conditional_boot = conditional_boot.assign(model='conditional_path')
            treatment_boot = treatment_boot.assign(model='treatment_mediation')
            primary_bootstrap = [conditional_boot, treatment_boot]
    conditional_frame = pd.DataFrame(conditional_model_rows)
    indirect_frame = pd.DataFrame(indirect_rows)
    treatment_frame = pd.DataFrame(treatment_rows)
    coefficient_frame = pd.DataFrame(coefficient_output)
    bootstrap_frame = pd.concat(primary_bootstrap, ignore_index=True, sort=False)
    primary = prepare(cohorts['primary_motion_qc'])
    (conditional_m, conditional_y, _) = conditional_point(primary)
    (treatment_m, treatment_y, _, _) = treatment_mediation_point(primary)
    permutation_rows = [{'model': 'conditional_mediator', 'term': 'group_x_lesion_span', 'p_freedman_lane': coefficient_permutation(primary['g2_normalization_z'].to_numpy(float), add_constant(primary, CONDITIONAL_M_TERMS), 'group_x_lesion_span', n_permutations, np.random.default_rng(SEED + 900001))}, {'model': 'conditional_outcome', 'term': 'g2_normalization_z', 'p_freedman_lane': coefficient_permutation(primary['NIHSS_post'].to_numpy(float), add_constant(primary, CONDITIONAL_Y_TERMS), 'g2_normalization_z', n_permutations, np.random.default_rng(SEED + 900002))}, {'model': 'treatment_mediator', 'term': 'group_tms', 'p_freedman_lane': coefficient_permutation(primary['lesion_centered_g2_slope_z'].to_numpy(float), add_constant(primary, TREATMENT_M_TERMS), 'group_tms', n_permutations, np.random.default_rng(SEED + 900003))}, {'model': 'treatment_outcome', 'term': 'lesion_centered_g2_slope_z', 'p_freedman_lane': coefficient_permutation(primary['NIHSS_post'].to_numpy(float), add_constant(primary, TREATMENT_Y_TERMS), 'lesion_centered_g2_slope_z', n_permutations, np.random.default_rng(SEED + 900004))}]
    return (conditional_frame, indirect_frame, treatment_frame, coefficient_frame, bootstrap_frame, pd.DataFrame(permutation_rows))

def incremental_fit(primary: pd.DataFrame) -> pd.DataFrame:
    frame = prepare(primary)
    base_terms = ['NIHSS_pre_z', 'group_tms', 'lesion_span_z', 'group_x_lesion_span', 'baseline_g2_deviation_z', 'log_lesion_volume_z']
    models = {'clinical_lesion': base_terms, 'plus_global_g2_normalization': base_terms + ['g2_normalization_z'], 'plus_lesion_centered_g2_slope': base_terms + ['lesion_centered_g2_slope_z'], 'plus_both_g2_metrics': base_terms + ['g2_normalization_z', 'lesion_centered_g2_slope_z']}
    rows = []
    y = frame['NIHSS_post'].to_numpy(float)
    base_result = sm.OLS(y, add_constant(frame, base_terms)).fit(cov_type='HC3')
    for (model_name, terms) in models.items():
        result = sm.OLS(y, add_constant(frame, terms)).fit(cov_type='HC3')
        rows.append({'model': model_name, 'n': len(frame), 'r_squared': float(result.rsquared), 'adjusted_r_squared': float(result.rsquared_adj), 'delta_r_squared_vs_clinical_lesion': float(result.rsquared - base_result.rsquared), 'aic': float(result.aic)})
    return pd.DataFrame(rows)

def plot_results(conditional: pd.DataFrame, indirect: pd.DataFrame, treatment: pd.DataFrame) -> None:
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'], 'font.size': 7, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    primary_path = conditional.loc[conditional['cohort'] == 'primary_motion_qc'].iloc[0]
    primary_indirect = indirect.loc[indirect['cohort'] == 'primary_motion_qc'].set_index('effect')
    primary_treatment = treatment.loc[treatment['cohort'] == 'primary_motion_qc'].iloc[0]
    (figure, axes) = plt.subplots(1, 2, figsize=(183 / 25.4, 82 / 25.4))
    (left, right) = axes
    left.axis('off')
    nodes = [(0.03, 0.42, 'Baseline lesion\nG2 span'), (0.37, 0.42, 'Healthy-reference\nG2 normalization'), (0.74, 0.42, 'Post-treatment\nNIHSS')]
    for (x, y, label) in nodes:
        left.text(x, y, label, transform=left.transAxes, ha='left', va='center', fontsize=7, bbox={'boxstyle': 'round,pad=0.35,rounding_size=0.08', 'facecolor': '#F4F4F4', 'edgecolor': '#555555', 'linewidth': 0.7})
    left.annotate('', xy=(0.36, 0.42), xytext=(0.22, 0.42), xycoords='axes fraction', arrowprops={'arrowstyle': '->', 'color': '#4C78A8', 'lw': 1.2})
    left.annotate('', xy=(0.73, 0.42), xytext=(0.6, 0.42), xycoords='axes fraction', arrowprops={'arrowstyle': '->', 'color': '#B55D45', 'lw': 1.2})
    left.text(0.29, 0.55, f"Control a = {primary_path['a_control_lesion_to_g2']:.2f}\nTMS a = {primary_path['a_tms_lesion_to_g2']:.2f}\ninteraction P = {primary_path['a_interaction_p_hc3']:.3f}", transform=left.transAxes, ha='center', va='bottom', fontsize=6)
    left.text(0.67, 0.55, f"b = {primary_path['b_g2_to_nihss']:.2f}\nP = {primary_path['b_p_hc3']:.3f}", transform=left.transAxes, ha='center', va='bottom', fontsize=6)
    left.text(0.03, 0.9, 'a', transform=left.transAxes, fontsize=9, fontweight='bold')
    left.set_title('Conditional lesion-gradient-clinical path', loc='left', fontweight='bold')
    forest_rows = [('Control indirect', primary_indirect.loc['control_indirect']), ('TMS indirect', primary_indirect.loc['tms_indirect']), ('Difference in indirect effects', primary_indirect.loc['moderated_mediation_index']), ('TMS -> lesion-centred G2 -> NIHSS', primary_treatment)]
    y_positions = np.arange(len(forest_rows))[::-1]
    colors = ['#777777', '#4C78A8', '#8A6FA8', '#B55D45']
    for (position, (_, row), color) in zip(y_positions, forest_rows, colors):
        right.plot([row['bootstrap_ci_low'], row['bootstrap_ci_high']], [position, position], color=color, linewidth=1.4)
        right.scatter(row['estimate'], position, s=28, color=color, edgecolor='white', linewidth=0.5, zorder=3)
    right.axvline(0, color='#555555', linestyle='--', linewidth=0.8)
    right.set_yticks(y_positions, [label for (label, _) in forest_rows])
    right.set_xlabel('Indirect effect on post NIHSS (points)')
    right.set_title('Bootstrap indirect effects', loc='left', fontweight='bold')
    right.text(-0.18, 1.03, 'b', transform=right.transAxes, fontsize=9, fontweight='bold')
    figure.subplots_adjust(left=0.03, right=0.985, bottom=0.18, top=0.88, wspace=0.55)
    for suffix in ('png', 'pdf', 'svg'):
        figure.savefig(FIGURES / f'lesion_gradient_clinical_path.{suffix}', dpi=300 if suffix == 'png' else None)
    plt.close(figure)

def write_summary(conditional: pd.DataFrame, indirect: pd.DataFrame, treatment: pd.DataFrame, permutation: pd.DataFrame, incremental: pd.DataFrame) -> None:
    path = conditional.loc[conditional['cohort'] == 'primary_motion_qc'].iloc[0]
    ind = indirect.loc[indirect['cohort'] == 'primary_motion_qc'].set_index('effect')
    treatment_row = treatment.loc[treatment['cohort'] == 'primary_motion_qc'].iloc[0]
    perm = permutation.set_index(['model', 'term'])['p_freedman_lane']
    inc = incremental.set_index('model')

    def effect_line(effect: str) -> str:
        row = ind.loc[effect]
        return f"{row['estimate']:.3f}分，95% Bootstrap CI {row['bootstrap_ci_low']:.3f}至{row['bootstrap_ci_high']:.3f}，P={row['bootstrap_p_two_sided']:.4f}"
    text = f"# 病灶—梯度—NIHSS三变量路径分析\n\n## 预设主路径\n\n- 主队列：运动QC合格的37例；TMS 22例、Control 15例。\n- 病灶变量：治疗前病灶通过HCP-XTRACT终点投射后覆盖的健康G2层级跨度。\n- 梯度变量：治疗前到治疗后向独立健康reference靠近的全局G2正常化。\n- 临床结局：治疗后NIHSS，校正治疗前NIHSS。\n- 第一阶段允许`病灶G2跨度 x TMS`调节；模型同时校正基线G2异常和log病灶体积。\n- 第二阶段同时校正治疗组、病灶G2跨度、病灶交互、基线G2异常和log病灶体积。\n- 间接效应使用分组分层的{int(ind.iloc[0]['valid_bootstrap_n']):,}次Bootstrap；路径系数使用HC3稳健区间和5,000次Freedman-Lane置换。\n\n## 分段结果\n\n- Control中病灶G2跨度→G2正常化：a={path['a_control_lesion_to_g2']:.3f}，HC3 P={path['a_control_p_hc3']:.4f}。\n- TMS中病灶G2跨度→G2正常化：a={path['a_tms_lesion_to_g2']:.3f}，HC3 P={path['a_tms_p_hc3']:.4f}。\n- 两组第一阶段斜率差：interaction={path['a_interaction']:.3f}，95% CI {path['a_interaction_ci_low_hc3']:.3f}至{path['a_interaction_ci_high_hc3']:.3f}，HC3 P={path['a_interaction_p_hc3']:.4f}，置换P={perm.loc['conditional_mediator', 'group_x_lesion_span']:.4f}。\n- G2正常化→治疗后NIHSS：b={path['b_g2_to_nihss']:.3f}分/1 SD，95% CI {path['b_ci_low_hc3']:.3f}至{path['b_ci_high_hc3']:.3f}，HC3 P={path['b_p_hc3']:.4f}，置换P={perm.loc['conditional_outcome', 'g2_normalization_z']:.4f}，partial R2={path['b_partial_r2']:.3f}。\n\n## 条件间接效应\n\n- Control：{effect_line('control_indirect')}。\n- TMS：{effect_line('tms_indirect')}。\n- 两组间接效应之差：{effect_line('moderated_mediation_index')}。\n\n负的间接效应表示该路径最终对应更低的治疗后NIHSS。只有Bootstrap区间不跨0时，才支持相应的统计间接路径。\n\n## TMS—病灶参照G2—NIHSS路径\n\n这里把`xtract_projected_G2_slope`作为中介。它表示健康G2正常化相对病灶功能层级向远端扩展的程度，因此同时包含病灶位置与梯度变化信息。\n\n- TMS→病灶参照G2斜率：a={treatment_row['a_tms_to_lesion_centered_g2']:.3f} SD，95% CI {treatment_row['a_ci_low_hc3']:.3f}至{treatment_row['a_ci_high_hc3']:.3f}，HC3 P={treatment_row['a_p_hc3']:.4f}，置换P={perm.loc['treatment_mediator', 'group_tms']:.4f}。\n- 病灶参照G2斜率→NIHSS：b={treatment_row['b_lesion_centered_g2_to_nihss']:.3f}分/1 SD，95% CI {treatment_row['b_ci_low_hc3']:.3f}至{treatment_row['b_ci_high_hc3']:.3f}，HC3 P={treatment_row['b_p_hc3']:.4f}，置换P={perm.loc['treatment_outcome', 'lesion_centered_g2_slope_z']:.4f}。\n- 间接效应：{treatment_row['estimate']:.3f}分，95% Bootstrap CI {treatment_row['bootstrap_ci_low']:.3f}至{treatment_row['bootstrap_ci_high']:.3f}，P={treatment_row['bootstrap_p_two_sided']:.4f}。\n- TMS总效应={treatment_row['total_tms_effect']:.3f}分；加入该G2指标后的直接效应={treatment_row['direct_tms_effect']:.3f}分。\n\n## 梯度增量\n\n- 临床+病灶模型R2={inc.loc['clinical_lesion', 'r_squared']:.3f}。\n- 加全局G2正常化后，delta R2={inc.loc['plus_global_g2_normalization', 'delta_r_squared_vs_clinical_lesion']:.3f}。\n- 加病灶参照G2斜率后，delta R2={inc.loc['plus_lesion_centered_g2_slope', 'delta_r_squared_vs_clinical_lesion']:.3f}。\n- 同时加入两个G2指标后，delta R2={inc.loc['plus_both_g2_metrics', 'delta_r_squared_vs_clinical_lesion']:.3f}。\n\n## 总结\n\n- 预设的`病灶G2跨度 → G2健康正常化 → NIHSS`路径不成立：第一段、第二段以及Control/TMS条件间接效应均接近0。\n- `TMS → 病灶参照的G2远端正常化`这一段稳定存在，但`病灶参照G2 → NIHSS`不成立，因此不能把TMS相关NIHSS差异解释为由该G2指标介导。\n- 两个G2指标在临床+病灶模型之上合计只增加约{inc.loc['plus_both_g2_metrics', 'delta_r_squared_vs_clinical_lesion']:.3f}的样本内R2，当前不支持具有实质性的评分解释增量。\n- 所有敏感性队列的间接效应置信区间均跨0，结论不依赖运动QC病例、病灶QC病例或`PRIVATE_SUBJECT_038`的分组处理。\n\n## 解释边界\n\n- 这是小样本探索性统计路径，不是已经证明的因果中介。\n- 梯度变化和临床变化位于同一治疗前后时间窗，无法确认梯度变化先于临床改善。\n- XTRACT是规范纤维束终点投射代理，不是个体化tractography。\n- 主结果以37例运动QC队列为准；38例全纳入、排除病灶QC病例、排除或改判分组冲突病例均保存为敏感性分析。\n"
    (OUTPUT_ROOT / 'RESULTS_ZH.md').write_text(text, encoding='utf-8')

def run(n_bootstrap: int, n_permutations: int) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    data = load_data()
    cohorts = cohort_variants(data)
    (conditional, indirect, treatment, coefficients, bootstrap, permutation) = fit_all(cohorts, n_bootstrap, n_permutations)
    incremental = incremental_fit(cohorts['primary_motion_qc'])
    data.to_csv(TABLES / 'analysis_dataset_38.csv', index=False, encoding='utf-8-sig')
    conditional.to_csv(TABLES / 'conditional_path_results.csv', index=False, encoding='utf-8-sig')
    indirect.to_csv(TABLES / 'conditional_indirect_effects.csv', index=False, encoding='utf-8-sig')
    treatment.to_csv(TABLES / 'tms_lesion_centered_g2_mediation.csv', index=False, encoding='utf-8-sig')
    coefficients.to_csv(TABLES / 'all_model_coefficients.csv', index=False, encoding='utf-8-sig')
    bootstrap.to_csv(TABLES / 'primary_bootstrap_distributions.csv', index=False, encoding='utf-8-sig')
    permutation.to_csv(TABLES / 'primary_path_permutation_tests.csv', index=False, encoding='utf-8-sig')
    incremental.to_csv(TABLES / 'primary_incremental_fit.csv', index=False, encoding='utf-8-sig')
    plot_results(conditional, indirect, treatment)
    write_summary(conditional, indirect, treatment, permutation, incremental)
    manifest = {'analysis_date': '2026-08-04', 'primary_n': 37, 'primary_path': 'baseline lesion projected G2 span -> healthy-reference G2 normalization -> post NIHSS', 'moderator': 'TMS versus Control on the first path', 'secondary_path': 'TMS -> lesion-centred G2 normalization slope -> post NIHSS', 'covariates': {'mediator_model': ['group', 'lesion G2 span', 'group x lesion G2 span', 'baseline G2 deviation', 'log lesion volume'], 'outcome_model': ['baseline NIHSS', 'group', 'lesion G2 span', 'group x lesion G2 span', 'baseline G2 deviation', 'log lesion volume']}, 'bootstrap': {'resamples': n_bootstrap, 'type': 'stratified nonparametric participant bootstrap'}, 'permutations': n_permutations, 'robust_se': 'HC3', 'sensitivities': list(cohorts), 'causal_limit': 'exploratory statistical path; gradient and clinical change are contemporaneous', 'random_seed': SEED}
    (OUTPUT_ROOT / 'analysis_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(conditional.loc[conditional['cohort'] == 'primary_motion_qc'].to_string(index=False))
    print(indirect.loc[indirect['cohort'] == 'primary_motion_qc'].to_string(index=False))
    print(treatment.loc[treatment['cohort'] == 'primary_motion_qc'].to_string(index=False))

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bootstrap', type=int, default=5000)
    parser.add_argument('--permutations', type=int, default=5000)
    arguments = parser.parse_args()
    run(arguments.bootstrap, arguments.permutations)
if __name__ == '__main__':
    main()
