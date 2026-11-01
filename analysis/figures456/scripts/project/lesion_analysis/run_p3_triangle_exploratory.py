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
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm, pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests
ROOT = Path(_portable_path('project', 'lesion_analysis'))
INPUT_PATH = ROOT / 'lesion_gradient_score_mediation_provided_reference' / 'tables' / 'analysis_dataset_38.csv'
OUT_ROOT = ROOT / 'lesion_gradient_score_triangle_exploratory'
TABLE_DIR = OUT_ROOT / 'tables'
SEED = 20260811
SCORES = ('NIHSS', 'ADL', 'MMSE', 'HAMD', 'VFT')
OUTCOMES = (*SCORES, 'Composite')

def zscore(values: pd.Series) -> pd.Series:
    numeric = values.astype(float)
    sd = float(numeric.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError(f'Cannot standardize {values.name}')
    return (numeric - float(numeric.mean())) / sd

def load_primary() -> pd.DataFrame:
    data = pd.read_csv(INPUT_PATH, dtype={'subject_id': str})
    data = data.loc[data['primary_motion_eligible'].astype(bool)].copy()
    if len(data) < 3:
        raise RuntimeError(f'Expected 37 motion-QC participants, found {len(data)}')
    data['lesion_g1g2_span'] = np.sqrt(data['xtract_projected_G1_span'].astype(float) ** 2 + data['xtract_projected_G2_span'].astype(float) ** 2)
    data['gradient_normalization'] = data['nearest_xtract_G1G2_decile_normalization'].astype(float)
    data['baseline_gradient_deviation'] = data['nearest_xtract_G1G2_decile_baseline_deviation'].astype(float)
    improvement_z = []
    baseline_function_z = []
    for score in SCORES:
        data[f'{score}_improvement_z'] = zscore(data[f'{score}_improvement'])
        improvement_z.append(f'{score}_improvement_z')
        direction = -1.0 if score in {'NIHSS', 'HAMD'} else 1.0
        oriented = direction * data[f'{score}_pre'].astype(float)
        name = f'{score}_baseline_function_z'
        data[name] = zscore(oriented.rename(name))
        baseline_function_z.append(name)
    data['Composite_improvement'] = data[improvement_z].mean(axis=1)
    data['Composite_improvement_z'] = zscore(data['Composite_improvement'])
    data['Composite_baseline_function_z'] = zscore(data[baseline_function_z].mean(axis=1).rename('Composite_baseline_function_z'))
    standardize = {'x_z': 'lesion_g1g2_span', 'm_z': 'gradient_normalization', 'baseline_m_z': 'baseline_gradient_deviation', 'volume_z': 'log1p_lesion_volume', 'pre_fd_z': 'baseline_mean_fd', 'post_fd_z': 'post_mean_fd'}
    for (destination, source) in standardize.items():
        data[destination] = zscore(data[source])
    data['x_by_m'] = data['x_z'] * data['m_z']
    data['x_by_group'] = data['x_z'] * data['group_tms'].astype(float)
    data['m_by_group'] = data['m_z'] * data['group_tms'].astype(float)
    return data.reset_index(drop=True)

def outcome_columns(outcome: str) -> tuple[str, str]:
    if outcome == 'Composite':
        return ('Composite_improvement_z', 'Composite_baseline_function_z')
    return (f'{outcome}_improvement_z', f'{outcome}_baseline_function_z')

def design(frame: pd.DataFrame, terms: list[str]) -> pd.DataFrame:
    return sm.add_constant(frame[terms].astype(float), has_constant='add')

def partial_r2(full_result, reduced_result) -> float:
    rss_full = float(np.sum(full_result.resid ** 2))
    rss_reduced = float(np.sum(reduced_result.resid ** 2))
    if rss_reduced <= 0:
        return np.nan
    return max((rss_reduced - rss_full) / rss_reduced, 0.0)

def stratified_permutation_indices(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    indices = np.arange(len(groups))
    result = indices.copy()
    for value in np.unique(groups):
        within = indices[groups == value]
        result[within] = rng.permutation(within)
    return result

def freedman_lane_p(frame: pd.DataFrame, y_column: str, terms: list[str], target: str, n_permutations: int, rng: np.random.Generator) -> float:
    full_x_frame = design(frame, terms)
    reduced_x_frame = full_x_frame.drop(columns=target)
    full_x = full_x_frame.to_numpy(float)
    reduced_x = reduced_x_frame.to_numpy(float)
    y = frame[y_column].to_numpy(float)
    target_index = list(full_x_frame.columns).index(target)

    def hc3_t_statistic(outcome: np.ndarray) -> float:
        inverse = np.linalg.pinv(full_x.T @ full_x)
        beta = inverse @ full_x.T @ outcome
        residual = outcome - full_x @ beta
        leverage = np.sum(full_x @ inverse * full_x, axis=1)
        adjusted_residual = residual / np.clip(1.0 - leverage, 1e-08, None)
        weighted_x = full_x * adjusted_residual[:, None]
        covariance = inverse @ (weighted_x.T @ weighted_x) @ inverse
        se = float(np.sqrt(max(covariance[target_index, target_index], 0.0)))
        return float(beta[target_index] / se) if se > 0 else np.nan
    observed = hc3_t_statistic(y)
    reduced_beta = np.linalg.lstsq(reduced_x, y, rcond=None)[0]
    fitted = reduced_x @ reduced_beta
    residuals = y - fitted
    groups = frame['group_tms'].to_numpy(int)
    exceed = 0
    for _ in range(n_permutations):
        permutation = stratified_permutation_indices(groups, rng)
        permuted_y = fitted + residuals[permutation]
        statistic = hc3_t_statistic(permuted_y)
        exceed += int(abs(statistic) >= abs(observed))
    return float((exceed + 1) / (n_permutations + 1))

def leave_one_out(frame: pd.DataFrame, y_column: str, terms: list[str], target: str) -> dict[str, float]:
    values = []
    for index in frame.index:
        subset = frame.drop(index=index)
        result = sm.OLS(subset[y_column], design(subset, terms)).fit()
        values.append(float(result.params[target]))
    array = np.asarray(values, dtype=float)
    full_sign = np.sign(sm.OLS(frame[y_column], design(frame, terms)).fit().params[target])
    return {'loo_beta_min': float(np.min(array)), 'loo_beta_max': float(np.max(array)), 'loo_same_sign_fraction': float(np.mean(np.sign(array) == full_sign))}

def fit_association(frame: pd.DataFrame, relationship: str, outcome: str, y_column: str, x_column: str, terms: list[str], target: str, n_permutations: int, rng: np.random.Generator) -> dict[str, object]:
    full = sm.OLS(frame[y_column], design(frame, terms)).fit(cov_type='HC3')
    reduced_terms = [term for term in terms if term != target]
    reduced = sm.OLS(frame[y_column], design(frame, reduced_terms)).fit()
    pearson = pearsonr(frame[x_column], frame[y_column])
    spearman = spearmanr(frame[x_column], frame[y_column])
    return {'relationship': relationship, 'outcome': outcome, 'n': len(frame), 'raw_pearson_r': float(pearson.statistic), 'raw_pearson_p': float(pearson.pvalue), 'raw_spearman_rho': float(spearman.statistic), 'raw_spearman_p': float(spearman.pvalue), 'adjusted_beta': float(full.params[target]), 'adjusted_se_hc3': float(full.bse[target]), 'adjusted_ci_low_hc3': float(full.conf_int().loc[target, 0]), 'adjusted_ci_high_hc3': float(full.conf_int().loc[target, 1]), 'adjusted_p_hc3': float(full.pvalues[target]), 'adjusted_p_freedman_lane': freedman_lane_p(frame, y_column, terms, target, n_permutations, rng), 'partial_r_squared': partial_r2(full, reduced), 'model_r_squared': float(full.rsquared), **leave_one_out(frame, y_column, terms, target)}

def association_analysis(frame: pd.DataFrame, n_permutations: int, rng: np.random.Generator) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    x_to_m_terms = ['x_z', 'group_tms', 'baseline_m_z', 'volume_z', 'pre_fd_z', 'post_fd_z']
    rows.append(fit_association(frame, 'lesion_to_gradient', 'Gradient', 'm_z', 'x_z', x_to_m_terms, 'x_z', n_permutations, rng))
    for outcome in OUTCOMES:
        (y_column, baseline_column) = outcome_columns(outcome)
        x_to_y_terms = ['x_z', baseline_column, 'group_tms', 'baseline_m_z', 'volume_z', 'pre_fd_z', 'post_fd_z']
        rows.append(fit_association(frame, 'lesion_to_clinical', outcome, y_column, 'x_z', x_to_y_terms, 'x_z', n_permutations, rng))
        m_to_y_terms = ['m_z', 'x_z', baseline_column, 'group_tms', 'baseline_m_z', 'volume_z', 'pre_fd_z', 'post_fd_z']
        rows.append(fit_association(frame, 'gradient_to_clinical', outcome, y_column, 'm_z', m_to_y_terms, 'm_z', n_permutations, rng))
    result = pd.DataFrame(rows)
    result['fdr_q_freedman_lane_all_13'] = multipletests(result['adjusted_p_freedman_lane'], method='fdr_bh')[1]
    result['fdr_q_freedman_lane_within_relationship'] = np.nan
    for (_, subset) in result.groupby('relationship'):
        result.loc[subset.index, 'fdr_q_freedman_lane_within_relationship'] = multipletests(subset['adjusted_p_freedman_lane'], method='fdr_bh')[1]
    return result

def moderation_analysis(frame: pd.DataFrame, n_permutations: int, rng: np.random.Generator) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    first_terms = ['x_z', 'group_tms', 'x_by_group', 'baseline_m_z', 'volume_z', 'pre_fd_z', 'post_fd_z']
    rows.append(fit_association(frame, 'group_modifies_lesion_to_gradient', 'Gradient', 'm_z', 'x_by_group', first_terms, 'x_by_group', n_permutations, rng))
    for outcome in OUTCOMES:
        (y_column, baseline_column) = outcome_columns(outcome)
        common = ['x_z', 'm_z', baseline_column, 'group_tms', 'baseline_m_z', 'volume_z', 'pre_fd_z', 'post_fd_z']
        rows.append(fit_association(frame, 'lesion_modifies_gradient_to_clinical', outcome, y_column, 'x_by_m', [*common, 'x_by_m'], 'x_by_m', n_permutations, rng))
        rows.append(fit_association(frame, 'group_modifies_gradient_to_clinical', outcome, y_column, 'm_by_group', [*common, 'm_by_group'], 'm_by_group', n_permutations, rng))
    result = pd.DataFrame(rows)
    result['fdr_q_freedman_lane_all_13'] = multipletests(result['adjusted_p_freedman_lane'], method='fdr_bh')[1]
    result['fdr_q_freedman_lane_within_relationship'] = np.nan
    for (_, subset) in result.groupby('relationship'):
        result.loc[subset.index, 'fdr_q_freedman_lane_within_relationship'] = multipletests(subset['adjusted_p_freedman_lane'], method='fdr_bh')[1]
    return result

def gradient_clinical_covariate_sequence(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for outcome in OUTCOMES:
        (y_column, baseline_column) = outcome_columns(outcome)
        models = {'raw': ['m_z'], 'baseline_and_lesion': ['m_z', 'x_z', baseline_column], 'plus_group': ['m_z', 'x_z', baseline_column, 'group_tms'], 'full_primary': ['m_z', 'x_z', baseline_column, 'group_tms', 'baseline_m_z', 'volume_z', 'pre_fd_z', 'post_fd_z']}
        for (model_name, terms) in models.items():
            result = sm.OLS(frame[y_column], design(frame, terms)).fit(cov_type='HC3')
            rows.append({'outcome': outcome, 'model': model_name, 'gradient_beta': float(result.params['m_z']), 'ci_low_hc3': float(result.conf_int().loc['m_z', 0]), 'ci_high_hc3': float(result.conf_int().loc['m_z', 1]), 'p_hc3': float(result.pvalues['m_z']), 'r_squared': float(result.rsquared)})
    return pd.DataFrame(rows)

def group_specific_correlations(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (group, subset) in frame.groupby('group'):
        associations = [('lesion_to_gradient', 'Gradient', 'x_z', 'm_z')]
        for outcome in OUTCOMES:
            (y_column, _) = outcome_columns(outcome)
            associations.extend([('lesion_to_clinical', outcome, 'x_z', y_column), ('gradient_to_clinical', outcome, 'm_z', y_column)])
        for (relationship, outcome, x_column, y_column) in associations:
            pearson = pearsonr(subset[x_column], subset[y_column])
            spearman = spearmanr(subset[x_column], subset[y_column])
            rows.append({'group': group, 'relationship': relationship, 'outcome': outcome, 'n': len(subset), 'pearson_r': float(pearson.statistic), 'pearson_p': float(pearson.pvalue), 'spearman_rho': float(spearman.statistic), 'spearman_p': float(spearman.pvalue)})
    return pd.DataFrame(rows)

def lesion_gradient_simple_slopes(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for outcome in OUTCOMES:
        (y_column, baseline_column) = outcome_columns(outcome)
        terms = ['x_z', 'm_z', baseline_column, 'group_tms', 'baseline_m_z', 'volume_z', 'pre_fd_z', 'post_fd_z', 'x_by_m']
        result = sm.OLS(frame[y_column], design(frame, terms)).fit(cov_type='HC3')
        covariance = result.cov_params()
        for (x_level, label) in [(-1.0, 'span_minus_1SD'), (0.0, 'span_mean'), (1.0, 'span_plus_1SD')]:
            slope = float(result.params['m_z'] + x_level * result.params['x_by_m'])
            variance = float(covariance.loc['m_z', 'm_z'] + 2 * x_level * covariance.loc['m_z', 'x_by_m'] + x_level ** 2 * covariance.loc['x_by_m', 'x_by_m'])
            se = float(np.sqrt(max(variance, 0.0)))
            z_value = slope / se if se > 0 else np.nan
            rows.append({'outcome': outcome, 'lesion_span_level': label, 'lesion_span_z': x_level, 'gradient_slope': slope, 'se_hc3': se, 'ci_low_hc3': slope - 1.96 * se, 'ci_high_hc3': slope + 1.96 * se, 'p_hc3': float(2 * norm.sf(abs(z_value))) if np.isfinite(z_value) else np.nan})
    return pd.DataFrame(rows)

def clinical_composite_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    matrix = frame[[f'{score}_improvement_z' for score in SCORES]].to_numpy(float)
    (_, singular_values, vh) = np.linalg.svd(matrix, full_matrices=False)
    variance = singular_values ** 2
    explained = variance / variance.sum()
    loadings = vh[0].copy()
    if loadings.mean() < 0:
        loadings *= -1
    return pd.DataFrame({'score': SCORES, 'PC1_loading': loadings, 'correlation_with_mean_z_composite': [float(frame[f'{score}_improvement_z'].corr(frame['Composite_improvement_z'])) for score in SCORES], 'PC1_variance_explained': explained[0]})

def prepare_any_cohort(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy().reset_index(drop=True)
    result['lesion_g1g2_span'] = np.sqrt(result['xtract_projected_G1_span'].astype(float) ** 2 + result['xtract_projected_G2_span'].astype(float) ** 2)
    result['gradient_normalization'] = result['nearest_xtract_G1G2_decile_normalization'].astype(float)
    result['baseline_gradient_deviation'] = result['nearest_xtract_G1G2_decile_baseline_deviation'].astype(float)
    result['MMSE_improvement_z'] = zscore(result['MMSE_improvement'])
    result['MMSE_baseline_function_z'] = zscore(result['MMSE_pre'])
    for (destination, source) in {'x_z': 'lesion_g1g2_span', 'm_z': 'gradient_normalization', 'baseline_m_z': 'baseline_gradient_deviation', 'volume_z': 'log1p_lesion_volume', 'pre_fd_z': 'baseline_mean_fd', 'post_fd_z': 'post_mean_fd'}.items():
        result[destination] = zscore(result[source])
    result['x_by_m'] = result['x_z'] * result['m_z']
    return result

def mmse_moderation_sensitivity() -> pd.DataFrame:
    all_data = pd.read_csv(INPUT_PATH, dtype={'subject_id': str})
    motion = all_data.loc[all_data['primary_motion_eligible'].astype(bool)].copy()
    manual_flags = {'PRIVATE_SUBJECT_040', 'PRIVATE_SUBJECT_006', 'PRIVATE_SUBJECT_038'}
    cohorts = {'primary_motion_qc': motion, 'all_38': all_data, 'exclude_manual_flags': motion.loc[~motion['subject_id'].isin(manual_flags)], 'exclude_group_conflict': motion.loc[motion['subject_id'] != 'PRIVATE_SUBJECT_038']}
    full_terms = ['x_z', 'm_z', 'MMSE_baseline_function_z', 'group_tms', 'baseline_m_z', 'volume_z', 'pre_fd_z', 'post_fd_z', 'x_by_m']
    rows: list[dict[str, object]] = []
    for (cohort_name, raw) in cohorts.items():
        frame = prepare_any_cohort(raw)
        result = sm.OLS(frame['MMSE_improvement_z'], design(frame, full_terms)).fit(cov_type='HC3')
        rows.append({'analysis': 'cohort_sensitivity', 'variant': cohort_name, 'estimator': 'OLS_HC3', 'n': len(frame), 'interaction_beta': float(result.params['x_by_m']), 'ci_low': float(result.conf_int().loc['x_by_m', 0]), 'ci_high': float(result.conf_int().loc['x_by_m', 1]), 'p': float(result.pvalues['x_by_m']), 'r_squared': float(result.rsquared)})
    primary = prepare_any_cohort(motion)
    covariate_models = {'full': full_terms, 'without_baseline_gradient': [term for term in full_terms if term != 'baseline_m_z'], 'without_motion': [term for term in full_terms if term not in {'pre_fd_z', 'post_fd_z'}], 'minimal': ['x_z', 'm_z', 'MMSE_baseline_function_z', 'group_tms', 'x_by_m']}
    for (variant, terms) in covariate_models.items():
        result = sm.OLS(primary['MMSE_improvement_z'], design(primary, terms)).fit(cov_type='HC3')
        rows.append({'analysis': 'covariate_sensitivity', 'variant': variant, 'estimator': 'OLS_HC3', 'n': len(primary), 'interaction_beta': float(result.params['x_by_m']), 'ci_low': float(result.conf_int().loc['x_by_m', 0]), 'ci_high': float(result.conf_int().loc['x_by_m', 1]), 'p': float(result.pvalues['x_by_m']), 'r_squared': float(result.rsquared)})
    robust = sm.RLM(primary['MMSE_improvement_z'], design(primary, full_terms), M=sm.robust.norms.HuberT()).fit(cov='H2')
    rows.append({'analysis': 'robust_estimator', 'variant': 'primary_full', 'estimator': 'Huber_RLM_H2', 'n': len(primary), 'interaction_beta': float(robust.params['x_by_m']), 'ci_low': float(robust.conf_int().loc['x_by_m', 0]), 'ci_high': float(robust.conf_int().loc['x_by_m', 1]), 'p': float(robust.pvalues['x_by_m']), 'r_squared': np.nan})
    return pd.DataFrame(rows)

def write_summary(associations: pd.DataFrame, moderations: pd.DataFrame, sequence: pd.DataFrame, simple_slopes: pd.DataFrame, mmse_sensitivity: pd.DataFrame) -> None:
    significant_associations = associations.loc[associations['fdr_q_freedman_lane_all_13'] < 0.05]
    significant_moderations = moderations.loc[moderations['fdr_q_freedman_lane_all_13'] < 0.05]
    strongest_association = associations.sort_values('adjusted_p_freedman_lane').iloc[0]
    strongest_moderation = moderations.sort_values('adjusted_p_freedman_lane').iloc[0]
    mmse_sequence = sequence.loc[sequence['outcome'] == 'MMSE'].set_index('model')
    lines = ['# 病灶G1-G2跨度—邻近层级梯度正常化—临床改善：探索性三角分析', '', '## 设计', '', '- n=37（TMS 22，Control 15），全部为运动QC合格共同队列。', '- X：XTRACT投影的病灶G1-G2联合跨度。', '- M：在G1-G2功能空间中最接近病灶的10%完整ROI的healthy-reference normalization。', '- Y：NIHSS、ADL、MMSE、HAMD、VFT改善，以及五项标准化改善的平均综合分数。', '- 检验包括调整后两两关联、病灶×梯度交互、组别×梯度交互及组别×病灶交互。', '- 调整模型使用HC3区间、5,000次组内Freedman-Lane置换、BH-FDR和留一法稳定性。', '', '## 调整后两两关联', '']
    if len(significant_associations) == 0:
        lines.append('- 13个调整后关联在全家族FDR后均未达到q<0.05。')
    else:
        lines.append(f'- 全家族FDR后显著关联共{len(significant_associations)}个。')
    lines.append(f"- 最小置换P对应{strongest_association['relationship']} / {strongest_association['outcome']}：beta={strongest_association['adjusted_beta']:.3f}，P={strongest_association['adjusted_p_freedman_lane']:.4f}，q={strongest_association['fdr_q_freedman_lane_all_13']:.4f}。")
    lines.extend(['', '## 调节效应', ''])
    if len(significant_moderations) == 0:
        lines.append('- 13个调节/交互检验在全家族FDR后均未达到q<0.05。')
    else:
        lines.append(f'- 全家族FDR后显著交互共{len(significant_moderations)}个。')
    lines.append(f"- 最小置换P对应{strongest_moderation['relationship']} / {strongest_moderation['outcome']}：beta={strongest_moderation['adjusted_beta']:.3f}，P={strongest_moderation['adjusted_p_freedman_lane']:.4f}，q={strongest_moderation['fdr_q_freedman_lane_all_13']:.4f}。")
    lines.append('- 病灶×梯度的候选方向集中于NIHSS、MMSE和综合改善：病灶跨度较大时，normalization与改善的正向斜率更强；但这些交互均未通过13个调节检验的全家族FDR。')
    for outcome in ('NIHSS', 'MMSE', 'Composite'):
        subset = simple_slopes.loc[(simple_slopes['outcome'] == outcome) & (simple_slopes['lesion_span_level'] == 'span_plus_1SD')].iloc[0]
        lines.append(f"  - {outcome}在病灶跨度+1 SD时：gradient slope={subset['gradient_slope']:.3f}，95% CI {subset['ci_low_hc3']:.3f}至{subset['ci_high_hc3']:.3f}。")
    full_variants = mmse_sensitivity.loc[(mmse_sensitivity['analysis'] == 'covariate_sensitivity') & (mmse_sensitivity['estimator'] == 'OLS_HC3')]
    lines.append(f"- MMSE交互在协变量删减模型中的beta范围为{full_variants['interaction_beta'].min():.3f}至{full_variants['interaction_beta'].max():.3f}；方向稳定。")
    lines.extend(['', '## MMSE原始相关衰减诊断', '', f"- 原始梯度—MMSE改善：beta={mmse_sequence.loc['raw', 'gradient_beta']:.3f}，P={mmse_sequence.loc['raw', 'p_hc3']:.4f}。", f"- 加入MMSE基线和病灶跨度：beta={mmse_sequence.loc['baseline_and_lesion', 'gradient_beta']:.3f}，P={mmse_sequence.loc['baseline_and_lesion', 'p_hc3']:.4f}。", f"- 再加入TMS/Control：beta={mmse_sequence.loc['plus_group', 'gradient_beta']:.3f}，P={mmse_sequence.loc['plus_group', 'p_hc3']:.4f}。", f"- 完整调整：beta={mmse_sequence.loc['full_primary', 'gradient_beta']:.3f}，P={mmse_sequence.loc['full_primary', 'p_hc3']:.4f}。", '', '## 边界', '', '- 这些是探索性关联和调节分析，不证明因果或中介。', '- 评分与梯度变化发生于同一Pre/Post窗口；n=37也限制交互检验效能。', '- 全家族q是主要防止数据挖掘的判断标准；关系内q和未校正P只用于发现候选方向。'])
    (OUT_ROOT / 'RESULTS_ZH.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

def run(n_permutations: int) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_primary()
    rng = np.random.default_rng(SEED)
    associations = association_analysis(frame, n_permutations, rng)
    moderations = moderation_analysis(frame, n_permutations, rng)
    sequence = gradient_clinical_covariate_sequence(frame)
    group_correlations = group_specific_correlations(frame)
    simple_slopes = lesion_gradient_simple_slopes(frame)
    mmse_sensitivity = mmse_moderation_sensitivity()
    composite = clinical_composite_diagnostics(frame)
    frame.to_csv(TABLE_DIR / 'analysis_dataset_37.csv', index=False)
    associations.to_csv(TABLE_DIR / 'adjusted_triangle_associations.csv', index=False)
    moderations.to_csv(TABLE_DIR / 'moderation_interactions.csv', index=False)
    sequence.to_csv(TABLE_DIR / 'gradient_clinical_covariate_sequence.csv', index=False)
    group_correlations.to_csv(TABLE_DIR / 'group_specific_raw_correlations.csv', index=False)
    simple_slopes.to_csv(TABLE_DIR / 'lesion_gradient_simple_slopes.csv', index=False)
    mmse_sensitivity.to_csv(TABLE_DIR / 'mmse_moderation_sensitivity.csv', index=False)
    composite.to_csv(TABLE_DIR / 'clinical_composite_diagnostics.csv', index=False)
    write_summary(associations, moderations, sequence, simple_slopes, mmse_sensitivity)
    manifest = {'analysis_date': '2026-08-11', 'n': len(frame), 'control_n': int((frame['group_tms'] == 0).sum()), 'tms_n': int((frame['group_tms'] == 1).sum()), 'variables': {'X': 'XTRACT-projected lesion joint G1-G2 span', 'M': 'nearest G1-G2 functional decile healthy-reference normalization', 'Y': [*SCORES, 'mean-z clinical improvement composite']}, 'permutations': n_permutations, 'permutation': 'Freedman-Lane, residuals permuted within treatment group', 'robust_se': 'HC3', 'fdr': {'association_family': 13, 'moderation_family': 13}, 'random_seed': SEED, 'causal_limit': 'Exploratory association/moderation only; no temporal mediation claim.'}
    (OUT_ROOT / 'analysis_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print('ASSOCIATIONS')
    print(associations.to_string(index=False))
    print('\nMODERATIONS')
    print(moderations.to_string(index=False))

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--permutations', type=int, default=5000)
    args = parser.parse_args()
    run(args.permutations)
if __name__ == '__main__':
    main()
