from __future__ import annotations
import os as _portable_os
from pathlib import Path as _PortablePath

def _portable_path(kind, relative):
    project = _PortablePath(_portable_os.environ['PAPER_PROJECT_ROOT'])
    root = project if kind == 'project' else project.parent
    return str(root / relative)
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
ROOT = Path(_portable_path('project', 'lesion_analysis'))
SUBJECT_PATH = ROOT / 'lesion_gradient_distance' / 'figures' / 'nature' / 'lesion_final' / 'bcdf_recomputed_provided_reference' / 'recomputed_subject_metrics.csv'
CLINICAL_PATH = ROOT / 'lesion_matched_clinical_mediation' / 'tables' / 'analysis_cohort_38.csv'
OUT_ROOT = ROOT / 'lesion_gradient_score_mediation_provided_reference'
TABLE_DIR = OUT_ROOT / 'tables'
SEED = 20260811
SCORES = ('NIHSS', 'ADL', 'MMSE', 'HAMD', 'VFT')
LOWER_IS_BETTER = {'NIHSS', 'HAMD'}

@dataclass(frozen=True)
class PathSpec:
    path_id: str
    lesion_label: str
    lesion_variable: str
    gradient_label: str
    gradient_variable: str
    baseline_gradient_variable: str
    interpretation: str
PATHS = (PathSpec(path_id='P1_G2_span_global_G2', lesion_label='Projected lesion G2 span', lesion_variable='xtract_projected_G2_span', gradient_label='Whole-cortex G2 normalization', gradient_variable='mean_reference_directed_G2', baseline_gradient_variable='mean_baseline_deviation_G2', interpretation='Whether a lesion spanning more of the healthy G2 hierarchy is associated with clinical improvement through whole-cortex G2 normalization.'), PathSpec(path_id='P2_G2_position_G2_slope', lesion_label='Projected lesion G2 position', lesion_variable='xtract_projected_lesion_G2', gradient_label='Lesion-centred G2 distance slope', gradient_variable='xtract_projected_G2_slope', baseline_gradient_variable='mean_baseline_deviation_G2', interpretation='Whether lesion position along healthy G2 is associated with recovery through the anatomical-distance-adjusted G2 reorganization-distance slope.'), PathSpec(path_id='P3_G1G2_span_nearest_G1G2', lesion_label='Projected lesion G1-G2 span', lesion_variable='xtract_projected_G1G2_span', gradient_label='Nearest G1-G2 decile normalization', gradient_variable='nearest_xtract_G1G2_decile_normalization', baseline_gradient_variable='nearest_xtract_G1G2_decile_baseline_deviation', interpretation='Whether a lesion spanning more of joint healthy G1-G2 space is associated with recovery through normalization in the functionally nearest intact ROIs.'))

def zscore(values: pd.Series) -> pd.Series:
    numeric = values.astype(float)
    sd = float(numeric.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError(f'Cannot standardize {values.name}')
    return (numeric - float(numeric.mean())) / sd

def load_data() -> pd.DataFrame:
    subject = pd.read_csv(SUBJECT_PATH, dtype={'subject_id': str})
    clinical = pd.read_csv(CLINICAL_PATH, dtype={'subject_id': str})
    clinical_columns = ['subject_id', 'age', 'sex_male', 'esrs']
    for score in SCORES:
        clinical_columns.extend([f'{score}_pre', f'{score}_post'])
    data = subject.merge(clinical[clinical_columns], on='subject_id', how='inner', validate='one_to_one')
    if len(data) != 38 or data['subject_id'].nunique() != 38:
        raise RuntimeError(f'Expected 38 participants, found {len(data)}')
    data['xtract_projected_G1G2_span'] = np.sqrt(data['xtract_projected_G1_span'].astype(float) ** 2 + data['xtract_projected_G2_span'].astype(float) ** 2)
    for score in SCORES:
        pre = data[f'{score}_pre'].astype(float)
        post = data[f'{score}_post'].astype(float)
        if score in LOWER_IS_BETTER:
            data[f'{score}_improvement'] = pre - post
        else:
            data[f'{score}_improvement'] = post - pre
    required = ['group_tms', 'primary_motion_eligible', 'log1p_lesion_volume', 'baseline_mean_fd', 'post_mean_fd']
    for spec in PATHS:
        required.extend([spec.lesion_variable, spec.gradient_variable, spec.baseline_gradient_variable])
    for score in SCORES:
        required.extend([f'{score}_pre', f'{score}_post', f'{score}_improvement'])
    missing = data[required].isna().sum()
    if int(missing.sum()) != 0:
        raise RuntimeError(f'Missing primary values:\n{missing[missing > 0]}')
    if int(data['group_tms'].sum()) != 22:
        raise RuntimeError('Unexpected TMS count')
    return data.sort_values('subject_id').reset_index(drop=True)

def prepare_path(frame: pd.DataFrame, spec: PathSpec, score: str) -> pd.DataFrame:
    result = frame.copy().reset_index(drop=True)
    source_map = {'x_z': spec.lesion_variable, 'm_z': spec.gradient_variable, 'baseline_m_z': spec.baseline_gradient_variable, 'volume_z': 'log1p_lesion_volume', 'pre_fd_z': 'baseline_mean_fd', 'post_fd_z': 'post_mean_fd', 'score_pre_z': f'{score}_pre'}
    for (destination, source) in source_map.items():
        result[destination] = zscore(result[source])
    result['score_improvement'] = result[f'{score}_improvement'].astype(float)
    return result
M_TERMS = ['x_z', 'group_tms', 'baseline_m_z', 'volume_z', 'pre_fd_z', 'post_fd_z']
Y_TERMS = ['m_z', 'x_z', 'score_pre_z', 'group_tms', 'baseline_m_z', 'volume_z', 'pre_fd_z', 'post_fd_z']
TOTAL_TERMS = ['x_z', 'score_pre_z', 'group_tms', 'baseline_m_z', 'volume_z', 'pre_fd_z', 'post_fd_z']

def design(frame: pd.DataFrame, terms: list[str]) -> pd.DataFrame:
    return sm.add_constant(frame[terms].astype(float), has_constant='add')

def fit_point(frame: pd.DataFrame) -> dict[str, object]:
    m_result = sm.OLS(frame['m_z'].astype(float), design(frame, M_TERMS)).fit(cov_type='HC3')
    y_result = sm.OLS(frame['score_improvement'].astype(float), design(frame, Y_TERMS)).fit(cov_type='HC3')
    total_result = sm.OLS(frame['score_improvement'].astype(float), design(frame, TOTAL_TERMS)).fit(cov_type='HC3')
    a = float(m_result.params['x_z'])
    b = float(y_result.params['m_z'])
    return {'a': a, 'a_se_hc3': float(m_result.bse['x_z']), 'a_ci_low_hc3': float(m_result.conf_int().loc['x_z', 0]), 'a_ci_high_hc3': float(m_result.conf_int().loc['x_z', 1]), 'a_p_hc3': float(m_result.pvalues['x_z']), 'b': b, 'b_se_hc3': float(y_result.bse['m_z']), 'b_ci_low_hc3': float(y_result.conf_int().loc['m_z', 0]), 'b_ci_high_hc3': float(y_result.conf_int().loc['m_z', 1]), 'b_p_hc3': float(y_result.pvalues['m_z']), 'direct_effect': float(y_result.params['x_z']), 'direct_p_hc3': float(y_result.pvalues['x_z']), 'total_effect': float(total_result.params['x_z']), 'total_p_hc3': float(total_result.pvalues['x_z']), 'indirect_effect': a * b, 'mediator_r_squared': float(m_result.rsquared), 'outcome_r_squared': float(y_result.rsquared), 'outcome_delta_r_squared_for_m': float(y_result.rsquared - sm.OLS(frame['score_improvement'].astype(float), design(frame, [term for term in Y_TERMS if term != 'm_z'])).fit().rsquared)}

def stratified_indices(frame: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    pieces: list[np.ndarray] = []
    groups = frame['group_tms'].to_numpy(int)
    for value in (0, 1):
        indices = np.flatnonzero(groups == value)
        pieces.append(rng.choice(indices, size=len(indices), replace=True))
    return np.concatenate(pieces)

def fast_coef(y: np.ndarray, x: np.ndarray, coefficient_index: int) -> float:
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    return float(beta[coefficient_index])

def zscore_array(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    sd = float(np.std(values, ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError('Cannot standardize bootstrap sample')
    return (values - float(np.mean(values))) / sd

def bootstrap_indirect(raw_frame: pd.DataFrame, spec: PathSpec, score: str, n_bootstrap: int, rng: np.random.Generator) -> np.ndarray:
    estimates = np.full(n_bootstrap, np.nan, dtype=float)
    raw = {'x': raw_frame[spec.lesion_variable].to_numpy(float), 'm': raw_frame[spec.gradient_variable].to_numpy(float), 'baseline_m': raw_frame[spec.baseline_gradient_variable].to_numpy(float), 'volume': raw_frame['log1p_lesion_volume'].to_numpy(float), 'pre_fd': raw_frame['baseline_mean_fd'].to_numpy(float), 'post_fd': raw_frame['post_mean_fd'].to_numpy(float), 'score_pre': raw_frame[f'{score}_pre'].to_numpy(float), 'score_improvement': raw_frame[f'{score}_improvement'].to_numpy(float), 'group': raw_frame['group_tms'].to_numpy(float)}
    for bootstrap_id in range(n_bootstrap):
        indices = stratified_indices(raw_frame, rng)
        try:
            x_z = zscore_array(raw['x'][indices])
            m_z = zscore_array(raw['m'][indices])
            baseline_m_z = zscore_array(raw['baseline_m'][indices])
            volume_z = zscore_array(raw['volume'][indices])
            pre_fd_z = zscore_array(raw['pre_fd'][indices])
            post_fd_z = zscore_array(raw['post_fd'][indices])
            score_pre_z = zscore_array(raw['score_pre'][indices])
            group = raw['group'][indices]
            m_x = np.column_stack([np.ones(len(indices)), x_z, group, baseline_m_z, volume_z, pre_fd_z, post_fd_z])
            y_x = np.column_stack([np.ones(len(indices)), m_z, x_z, score_pre_z, group, baseline_m_z, volume_z, pre_fd_z, post_fd_z])
            a = fast_coef(m_z, m_x, 1)
            b = fast_coef(raw['score_improvement'][indices], y_x, 1)
            estimates[bootstrap_id] = a * b
        except (ValueError, np.linalg.LinAlgError):
            continue
    return estimates[np.isfinite(estimates)]

def bootstrap_summary(estimates: np.ndarray, point: float) -> dict[str, float | int]:
    if len(estimates) == 0:
        raise RuntimeError('No valid bootstrap estimates')
    nonpositive = (np.count_nonzero(estimates <= 0) + 1) / (len(estimates) + 1)
    nonnegative = (np.count_nonzero(estimates >= 0) + 1) / (len(estimates) + 1)
    return {'indirect_effect': float(point), 'bootstrap_ci_low': float(np.quantile(estimates, 0.025)), 'bootstrap_ci_high': float(np.quantile(estimates, 0.975)), 'bootstrap_p_two_sided': float(min(1.0, 2 * min(nonpositive, nonnegative))), 'valid_bootstrap_n': int(len(estimates))}

def run_models(cohorts: dict[str, pd.DataFrame], n_bootstrap: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    for (cohort_name, raw_frame) in cohorts.items():
        for spec in PATHS:
            for score in SCORES:
                frame = prepare_path(raw_frame, spec, score)
                point = fit_point(frame)
                estimates = bootstrap_indirect(raw_frame, spec, score, n_bootstrap, rng)
                boot = bootstrap_summary(estimates, float(point['indirect_effect']))
                rows.append({'cohort': cohort_name, 'path_id': spec.path_id, 'lesion_variable': spec.lesion_variable, 'gradient_variable': spec.gradient_variable, 'outcome': score, 'n': len(raw_frame), 'control_n': int((raw_frame['group_tms'] == 0).sum()), 'tms_n': int((raw_frame['group_tms'] == 1).sum()), **point, **boot})
                if cohort_name == 'primary_motion_qc':
                    bootstrap_rows.extend(({'path_id': spec.path_id, 'outcome': score, 'bootstrap_id': bootstrap_id, 'indirect_effect': value} for (bootstrap_id, value) in enumerate(estimates)))
    results = pd.DataFrame(rows)
    results['fdr_q_within_path'] = np.nan
    results['fdr_q_across_all_15'] = np.nan
    for (cohort_name, cohort_frame) in results.groupby('cohort'):
        cohort_indices = cohort_frame.index
        results.loc[cohort_indices, 'fdr_q_across_all_15'] = multipletests(cohort_frame['bootstrap_p_two_sided'].to_numpy(float), method='fdr_bh')[1]
        for (_, path_frame) in cohort_frame.groupby('path_id'):
            results.loc[path_frame.index, 'fdr_q_within_path'] = multipletests(path_frame['bootstrap_p_two_sided'].to_numpy(float), method='fdr_bh')[1]
    return (results, pd.DataFrame(bootstrap_rows))

def correlations(primary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in PATHS:
        x = primary[spec.lesion_variable].astype(float)
        m = primary[spec.gradient_variable].astype(float)
        rows.append({'path_id': spec.path_id, 'association': 'lesion_to_gradient_raw_pearson', 'outcome': 'gradient', 'r': float(x.corr(m))})
        for score in SCORES:
            y = primary[f'{score}_improvement'].astype(float)
            rows.extend([{'path_id': spec.path_id, 'association': 'lesion_to_improvement_raw_pearson', 'outcome': score, 'r': float(x.corr(y))}, {'path_id': spec.path_id, 'association': 'gradient_to_improvement_raw_pearson', 'outcome': score, 'r': float(m.corr(y))}])
    return pd.DataFrame(rows)

def mediator_diagnostics(primary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sensitivity_rows: list[dict[str, object]] = []
    vif_rows: list[dict[str, object]] = []
    model_terms = {'full_primary': M_TERMS, 'without_baseline_gradient': ['x_z', 'group_tms', 'volume_z', 'pre_fd_z', 'post_fd_z'], 'minimal_group_adjusted': ['x_z', 'group_tms']}
    for spec in PATHS:
        frame = prepare_path(primary, spec, 'NIHSS')
        for (model_name, terms) in model_terms.items():
            result = sm.OLS(frame['m_z'].astype(float), design(frame, terms)).fit(cov_type='HC3')
            sensitivity_rows.append({'path_id': spec.path_id, 'model': model_name, 'a': float(result.params['x_z']), 'ci_low_hc3': float(result.conf_int().loc['x_z', 0]), 'ci_high_hc3': float(result.conf_int().loc['x_z', 1]), 'p_hc3': float(result.pvalues['x_z']), 'r_squared': float(result.rsquared)})
        full_x = design(frame, M_TERMS)
        for (index, term) in enumerate(full_x.columns):
            vif_rows.append({'path_id': spec.path_id, 'term': term, 'vif': float(variance_inflation_factor(full_x.to_numpy(float), index))})
    return (pd.DataFrame(sensitivity_rows), pd.DataFrame(vif_rows))

def write_summary(results: pd.DataFrame, a_sensitivity: pd.DataFrame) -> None:
    primary = results.loc[results['cohort'] == 'primary_motion_qc'].copy()
    strongest = primary.sort_values('bootstrap_p_two_sided').iloc[0]
    significant = primary.loc[primary['fdr_q_across_all_15'] < 0.05]
    lines = ['# Lesion -> gradient -> clinical improvement 路径分析', '', '## 统一口径', '', '- 主队列为运动QC合格的37例（TMS 22，Control 15）。', '- 梯度指标全部来自所给gradient_calculation流程产生并对齐到固定healthy reference的新版结果。', '- NIHSS、HAMD用Pre-Post；ADL、MMSE、VFT用Post-Pre，因此所有改善值均为越大越好。', '- 中介模型控制组别、log病灶体积、相应基线梯度异常以及Pre/Post头动。', '- 结局模型另外控制相应评分基线；间接效应为a×b，使用组别分层Bootstrap。', '- 主推断对3条路径×5项评分的15个间接效应统一做BH-FDR。', '', '## 预设路径', '']
    for spec in PATHS:
        lines.append(f'- {spec.path_id}: {spec.lesion_label} -> {spec.gradient_label} -> 评分改善。')
    lines.extend(['', '## 主结果', ''])
    if len(significant) == 0:
        lines.append('- 15个间接效应中，没有任何一个在全家族FDR校正后达到q<0.05。')
    else:
        lines.append(f'- 全家族FDR后显著的间接路径共{len(significant)}条。')
    lines.append(f"- 未校正P值最小的路径为{strongest['path_id']} -> {strongest['outcome']}：间接效应={strongest['indirect_effect']:.4f}分，95% Bootstrap CI {strongest['bootstrap_ci_low']:.4f}至{strongest['bootstrap_ci_high']:.4f}，P={strongest['bootstrap_p_two_sided']:.4f}，全家族q={strongest['fdr_q_across_all_15']:.4f}。")
    p3_sensitivity = a_sensitivity.loc[a_sensitivity['path_id'] == 'P3_G1G2_span_nearest_G1G2'].set_index('model')
    p3_full = p3_sensitivity.loc['full_primary']
    p3_reduced = p3_sensitivity.loc['without_baseline_gradient']
    lines.append(f"- P3的病灶->梯度第一段在控制基线梯度偏离后很强（a={p3_full['a']:.3f}, P={p3_full['p_hc3']:.4g}），但去掉该基线协变量后减弱（a={p3_reduced['a']:.3f}, P={p3_reduced['p_hc3']:.4f}）；这提示存在基线调整/抑制效应，不能只按强相关解释。")
    lines.extend(['', '## 各路径分段系数', ''])
    for spec in PATHS:
        frame = primary.loc[primary['path_id'] == spec.path_id]
        first = frame.iloc[0]
        lines.append(f"- {spec.path_id}的病灶->梯度a路径：a={first['a']:.3f} SD/SD，95% HC3 CI {first['a_ci_low_hc3']:.3f}至{first['a_ci_high_hc3']:.3f}，P={first['a_p_hc3']:.4f}。")
        for (_, row) in frame.iterrows():
            lines.append(f"  - 梯度->{row['outcome']}改善：b={row['b']:.3f}分/SD，P={row['b_p_hc3']:.4f}；间接效应={row['indirect_effect']:.3f}，Bootstrap CI {row['bootstrap_ci_low']:.3f}至{row['bootstrap_ci_high']:.3f}，q15={row['fdr_q_across_all_15']:.3f}。")
    lines.extend(['', '## 解释边界', '', '- 病灶是在治疗前测量，但梯度变化和评分变化发生在同一Pre/Post时间窗，因此这是探索性统计中介，不证明时间先后或因果机制。', '- n=37对中介分析偏小，Bootstrap区间和留待外部验证的效应方向比单个P值更重要。', '- 病灶G2位置是坐标变量，不代表严重程度；其正负方向依赖已固定的healthy G2方向。'])
    (OUT_ROOT / 'RESULTS_ZH.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

def run(n_bootstrap: int) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    primary = data.loc[data['primary_motion_eligible'].astype(bool)].copy()
    cohorts = {'primary_motion_qc': primary, 'all_38_sensitivity': data.copy()}
    (results, bootstrap) = run_models(cohorts, n_bootstrap)
    (a_sensitivity, vif) = mediator_diagnostics(primary)
    data.to_csv(TABLE_DIR / 'analysis_dataset_38.csv', index=False)
    results.to_csv(TABLE_DIR / 'mediation_results.csv', index=False)
    bootstrap.to_csv(TABLE_DIR / 'primary_bootstrap_distributions.csv', index=False)
    correlations(primary).to_csv(TABLE_DIR / 'raw_correlations.csv', index=False)
    a_sensitivity.to_csv(TABLE_DIR / 'a_path_covariate_sensitivity.csv', index=False)
    vif.to_csv(TABLE_DIR / 'mediator_model_vif.csv', index=False)
    pd.DataFrame([{'path_id': spec.path_id, 'lesion_label': spec.lesion_label, 'lesion_variable': spec.lesion_variable, 'gradient_label': spec.gradient_label, 'gradient_variable': spec.gradient_variable, 'baseline_gradient_variable': spec.baseline_gradient_variable, 'interpretation': spec.interpretation} for spec in PATHS]).to_csv(TABLE_DIR / 'path_dictionary.csv', index=False)
    write_summary(results, a_sensitivity)
    manifest = {'analysis_date': '2026-08-11', 'primary_n': int(len(primary)), 'primary_control_n': int((primary['group_tms'] == 0).sum()), 'primary_tms_n': int((primary['group_tms'] == 1).sum()), 'healthy_reference_source': str(SUBJECT_PATH), 'clinical_source': str(CLINICAL_PATH), 'bootstrap_resamples': n_bootstrap, 'bootstrap_type': 'nonparametric participant bootstrap stratified by treatment group', 'fdr_primary_family': '3 paths x 5 outcomes = 15 indirect effects', 'score_improvement_orientation': {'NIHSS': 'pre - post', 'ADL': 'post - pre', 'MMSE': 'post - pre', 'HAMD': 'pre - post', 'VFT': 'post - pre'}, 'mediator_covariates': ['treatment group', 'log lesion volume', 'corresponding baseline gradient deviation', 'baseline mean FD', 'post mean FD'], 'outcome_additional_covariate': 'baseline clinical score', 'causal_limit': 'Exploratory statistical mediation only; gradient and clinical changes are contemporaneous.', 'random_seed': SEED}
    (OUT_ROOT / 'analysis_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(results.loc[results['cohort'] == 'primary_motion_qc'].to_string(index=False))

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bootstrap', type=int, default=5000)
    args = parser.parse_args()
    run(args.bootstrap)
if __name__ == '__main__':
    main()
