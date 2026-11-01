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
from statsmodels.stats.outliers_influence import variance_inflation_factor
ROOT = Path(_portable_path('project', 'lesion_analysis'))
DISTANCE_ROOT = ROOT / 'lesion_gradient_distance' / 'tables'
CLINICAL_ROOT = ROOT / 'lesion_adjusted_gradient_value' / 'tables'
OUTPUT_ROOT = ROOT / 'lesion_tms_moderation'
TABLES = OUTPUT_ROOT / 'tables'
FIGURES = OUTPUT_ROOT / 'figures'
SUBJECT_METRICS = DISTANCE_ROOT / 'subject_level_distance_metrics.csv'
ROI_METRICS = DISTANCE_ROOT / 'roi_level_distance_reorganization.csv'
CLINICAL = CLINICAL_ROOT / 'analysis_cohort.csv'
PRIMARY_MODIFIER = 'xtract_projected_G1_span'
SEED = 20260804
MANUAL_REVIEW_FLAGS = {'PRIVATE_SUBJECT_040', 'PRIVATE_SUBJECT_006', 'PRIVATE_SUBJECT_038'}

def zscore(values: pd.Series) -> pd.Series:
    numeric = values.astype(float)
    sd = float(numeric.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        raise RuntimeError(f'Cannot standardize {values.name}')
    return (numeric - float(numeric.mean())) / sd

def robust_range(values: pd.Series) -> float:
    array = values.to_numpy(float)
    return float(np.quantile(array, 0.95) - np.quantile(array, 0.05))

def load_analysis_data() -> pd.DataFrame:
    subjects = pd.read_csv(SUBJECT_METRICS, dtype={'subject_id': str})
    roi = pd.read_csv(ROI_METRICS, dtype={'subject_id': str})
    clinical = pd.read_csv(CLINICAL, dtype={'subject_id': str})
    intact = roi.loc[roi['intact_roi'].astype(bool)].copy()
    gradient_summary = intact.groupby('subject_id', sort=False).agg(intact_roi_n=('roi', 'size'), G1_range_pre=('G1_pre_z', robust_range), G1_range_post=('G1_post_z', robust_range), G2_range_pre=('G2_pre_z', robust_range), G2_range_post=('G2_post_z', robust_range)).reset_index()
    clinical_columns = ['subject_id', 'NIHSS_pre', 'NIHSS_post', 'ADL_pre', 'ADL_post', 'age', 'sex_male', 'esrs', 'manual_review_flag']
    data = subjects.merge(clinical[clinical_columns], on='subject_id', validate='one_to_one').merge(gradient_summary, on='subject_id', validate='one_to_one')
    data['manual_review_flag'] = data['subject_id'].isin(MANUAL_REVIEW_FLAGS)
    standardize = [PRIMARY_MODIFIER, 'xtract_projected_G2_span', 'xtract_projected_lesion_G1', 'log1p_lesion_volume', 'G1_range_pre', 'G2_range_pre', 'mean_baseline_deviation_G1', 'NIHSS_pre', 'ADL_pre', 'baseline_mean_fd', 'post_mean_fd', 'age', 'esrs']
    for column in standardize:
        data[f'{column}_z'] = zscore(data[column])
    if len(data) != 38 or data['subject_id'].nunique() != 38:
        raise RuntimeError(f'Unexpected analysis cohort: {len(data)} rows')
    required = [PRIMARY_MODIFIER, 'G1_range_pre', 'G1_range_post', 'NIHSS_pre', 'NIHSS_post', 'group_tms', 'log1p_lesion_volume']
    if data[required].isna().any().any():
        raise RuntimeError('Missing primary moderation data')
    return data.sort_values('subject_id').reset_index(drop=True)

def model_design(frame: pd.DataFrame, baseline: str, modifier: str, extra_covariates: list[str]) -> pd.DataFrame:
    design = pd.DataFrame({'baseline_z': frame[f'{baseline}_z'].to_numpy(float), 'group_tms': frame['group_tms'].to_numpy(float), 'lesion_gradient_z': frame[f'{modifier}_z'].to_numpy(float), 'group_x_lesion_gradient': frame['group_tms'].to_numpy(float) * frame[f'{modifier}_z'].to_numpy(float), 'log_lesion_volume_z': frame['log1p_lesion_volume_z'].to_numpy(float)}, index=frame.index)
    for covariate in extra_covariates:
        if covariate == 'sex_male':
            design[covariate] = frame[covariate].to_numpy(float)
        else:
            design[covariate] = frame[f'{covariate}_z'].to_numpy(float)
    return sm.add_constant(design, has_constant='add')

def linear_combination(result, weights: dict[str, float]) -> dict[str, float]:
    names = list(result.params.index)
    vector = np.asarray([weights.get(name, 0.0) for name in names], dtype=float)
    estimate = float(vector @ result.params.to_numpy(float))
    covariance = result.cov_params().to_numpy(float)
    se = float(np.sqrt(max(vector @ covariance @ vector, 0.0)))
    z_value = estimate / se if se > 0 else np.nan
    return {'estimate': estimate, 'se_hc3': se, 'ci_low_hc3': estimate - 1.96 * se, 'ci_high_hc3': estimate + 1.96 * se, 'p_hc3': float(2 * norm.sf(abs(z_value))) if np.isfinite(z_value) else np.nan}

def freedman_lane_interaction(y: np.ndarray, full_x: pd.DataFrame, n_permutations: int, rng: np.random.Generator) -> float:
    reduced_x = full_x.drop(columns='group_x_lesion_gradient')
    reduced = sm.OLS(y, reduced_x).fit()
    observed = float(sm.OLS(y, full_x).fit().params['group_x_lesion_gradient'])
    null = np.empty(n_permutations, dtype=float)
    fitted = reduced.fittedvalues.to_numpy(float)
    residuals = reduced.resid.to_numpy(float)
    for index in range(n_permutations):
        y_permuted = fitted + rng.permutation(residuals)
        null[index] = float(sm.OLS(y_permuted, full_x).fit().params['group_x_lesion_gradient'])
    return float((1 + np.count_nonzero(np.abs(null) >= abs(observed))) / (n_permutations + 1))

def fit_specification(data: pd.DataFrame, specification: dict, n_permutations: int, seed_offset: int) -> tuple[dict, list[dict], list[dict]]:
    keep = specification['keep'](data)
    frame = data.loc[keep].copy()
    outcome = specification['outcome']
    baseline = specification['baseline']
    modifier = specification['modifier']
    extra_covariates = specification.get('extra_covariates', [])
    full_x = model_design(frame, baseline, modifier, extra_covariates)
    y = frame[outcome].to_numpy(float)
    result = sm.OLS(y, full_x).fit(cov_type='HC3')
    reduced = sm.OLS(y, full_x.drop(columns='group_x_lesion_gradient')).fit()
    interaction = 'group_x_lesion_gradient'
    confidence = result.conf_int().loc[interaction]
    rss_full = float(np.sum(result.resid ** 2))
    rss_reduced = float(np.sum(reduced.resid ** 2))
    partial_r2 = max((rss_reduced - rss_full) / rss_reduced, 0.0)
    permutation_p = freedman_lane_interaction(y, full_x, n_permutations, np.random.default_rng(SEED + seed_offset))
    vif_rows = []
    numeric_x = full_x.to_numpy(float)
    for (index, name) in enumerate(full_x.columns):
        if name == 'const':
            continue
        vif_rows.append({'model_id': specification['model_id'], 'term': name, 'vif': float(variance_inflation_factor(numeric_x, index))})
    row = {'model_id': specification['model_id'], 'analysis_role': specification['role'], 'outcome': outcome, 'baseline': baseline, 'modifier': modifier, 'cohort': specification['cohort'], 'n': int(result.nobs), 'control_n': int((frame['group_tms'] == 0).sum()), 'tms_n': int((frame['group_tms'] == 1).sum()), 'interaction_beta': float(result.params[interaction]), 'interaction_se_hc3': float(result.bse[interaction]), 'interaction_ci_low_hc3': float(confidence.iloc[0]), 'interaction_ci_high_hc3': float(confidence.iloc[1]), 'interaction_p_hc3': float(result.pvalues[interaction]), 'interaction_p_freedman_lane': permutation_p, 'interaction_partial_r2': partial_r2, 'r_squared': float(result.rsquared), 'adjusted_r_squared': float(result.rsquared_adj), 'condition_number': float(result.condition_number), 'max_vif': float(max((item['vif'] for item in vif_rows))), 'extra_covariates': ';'.join(extra_covariates), 'n_permutations': n_permutations}
    slope_rows = []
    for (group_name, group_value) in (('Control', 0.0), ('TMS', 1.0)):
        weights = {'lesion_gradient_z': 1.0}
        if group_value == 1.0:
            weights[interaction] = 1.0
        estimate = linear_combination(result, weights)
        slope_rows.append({'model_id': specification['model_id'], 'group': group_name, 'outcome_change_per_1sd_modifier': estimate['estimate'], **{key: value for (key, value) in estimate.items() if key != 'estimate'}})
    contrast_rows = []
    for modifier_z in (-1.0, 0.0, 1.0):
        estimate = linear_combination(result, {'group_tms': 1.0, interaction: modifier_z})
        contrast_rows.append({'model_id': specification['model_id'], 'modifier_z': modifier_z, 'tms_minus_control': estimate['estimate'], **{key: value for (key, value) in estimate.items() if key != 'estimate'}})
    return (row, slope_rows + vif_rows, contrast_rows)

def specifications() -> list[dict]:
    all_rows = lambda data: np.ones(len(data), dtype=bool)
    motion = lambda data: data['primary_motion_eligible'].astype(bool).to_numpy()
    lesion_qc = lambda data: ~data['manual_review_flag'].astype(bool).to_numpy()
    motion_lesion_qc = lambda data: (data['primary_motion_eligible'].astype(bool) & ~data['manual_review_flag'].astype(bool)).to_numpy()
    return [{'model_id': 'gradient_G1_primary', 'role': 'primary_gradient', 'outcome': 'G1_range_post', 'baseline': 'G1_range_pre', 'modifier': PRIMARY_MODIFIER, 'cohort': 'primary_motion_qc', 'keep': motion}, {'model_id': 'clinical_NIHSS_primary', 'role': 'primary_clinical', 'outcome': 'NIHSS_post', 'baseline': 'NIHSS_pre', 'modifier': PRIMARY_MODIFIER, 'cohort': 'all_38', 'keep': all_rows}, {'model_id': 'gradient_G1_all38', 'role': 'sensitivity', 'outcome': 'G1_range_post', 'baseline': 'G1_range_pre', 'modifier': PRIMARY_MODIFIER, 'cohort': 'all_38', 'keep': all_rows}, {'model_id': 'gradient_G1_lesion_qc', 'role': 'sensitivity', 'outcome': 'G1_range_post', 'baseline': 'G1_range_pre', 'modifier': PRIMARY_MODIFIER, 'cohort': 'motion_and_lesion_qc', 'keep': motion_lesion_qc}, {'model_id': 'gradient_G1_motion_adjusted', 'role': 'sensitivity', 'outcome': 'G1_range_post', 'baseline': 'G1_range_pre', 'modifier': PRIMARY_MODIFIER, 'cohort': 'all_38', 'keep': all_rows, 'extra_covariates': ['baseline_mean_fd', 'post_mean_fd']}, {'model_id': 'clinical_NIHSS_lesion_qc', 'role': 'sensitivity', 'outcome': 'NIHSS_post', 'baseline': 'NIHSS_pre', 'modifier': PRIMARY_MODIFIER, 'cohort': 'exclude_lesion_qc', 'keep': lesion_qc}, {'model_id': 'clinical_NIHSS_adjusted', 'role': 'sensitivity', 'outcome': 'NIHSS_post', 'baseline': 'NIHSS_pre', 'modifier': PRIMARY_MODIFIER, 'cohort': 'all_38', 'keep': all_rows, 'extra_covariates': ['age', 'sex_male', 'esrs']}, {'model_id': 'gradient_G1_normalization_supplementary', 'role': 'supplementary', 'outcome': 'mean_reference_directed_G1', 'baseline': 'mean_baseline_deviation_G1', 'modifier': PRIMARY_MODIFIER, 'cohort': 'primary_motion_qc', 'keep': motion}, {'model_id': 'gradient_G2_supplementary', 'role': 'supplementary', 'outcome': 'G2_range_post', 'baseline': 'G2_range_pre', 'modifier': 'xtract_projected_G2_span', 'cohort': 'primary_motion_qc', 'keep': motion}, {'model_id': 'clinical_ADL_supplementary', 'role': 'supplementary', 'outcome': 'ADL_post', 'baseline': 'ADL_pre', 'modifier': PRIMARY_MODIFIER, 'cohort': 'all_38', 'keep': all_rows}]

def modifier_overlap(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for modifier in [PRIMARY_MODIFIER, 'xtract_projected_lesion_G1']:
        group_ranges = {}
        for (group_name, group_value) in (('Control', 0), ('TMS', 1)):
            values = data.loc[data['group_tms'] == group_value, modifier].astype(float)
            group_ranges[group_name] = (float(values.min()), float(values.max()))
            rows.append({'modifier': modifier, 'group': group_name, 'n': len(values), 'mean': float(values.mean()), 'sd': float(values.std(ddof=1)), 'min': float(values.min()), 'median': float(values.median()), 'max': float(values.max()), 'unique_n': int(values.nunique())})
        common_low = max(group_ranges['Control'][0], group_ranges['TMS'][0])
        common_high = min(group_ranges['Control'][1], group_ranges['TMS'][1])
        for row in rows:
            if row['modifier'] == modifier:
                row['common_support_low'] = common_low
                row['common_support_high'] = common_high
    return pd.DataFrame(rows)

def leave_one_out_primary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for specification in specifications()[:2]:
        primary = data.loc[specification['keep'](data)].copy()
        for subject_id in primary['subject_id']:
            frame = primary.loc[primary['subject_id'] != subject_id].copy()
            full_x = model_design(frame, specification['baseline'], specification['modifier'], specification.get('extra_covariates', []))
            result = sm.OLS(frame[specification['outcome']].to_numpy(float), full_x).fit(cov_type='HC3')
            term = 'group_x_lesion_gradient'
            confidence = result.conf_int().loc[term]
            rows.append({'model_id': specification['model_id'], 'excluded_subject_id': subject_id, 'excluded_group': primary.loc[primary['subject_id'] == subject_id, 'group'].iloc[0], 'excluded_manual_review_flag': bool(primary.loc[primary['subject_id'] == subject_id, 'manual_review_flag'].iloc[0]), 'n': int(result.nobs), 'interaction_beta': float(result.params[term]), 'interaction_ci_low_hc3': float(confidence.iloc[0]), 'interaction_ci_high_hc3': float(confidence.iloc[1]), 'interaction_p_hc3': float(result.pvalues[term])})
    return pd.DataFrame(rows)

def adjusted_plot_data(data: pd.DataFrame, specification: dict) -> tuple[pd.DataFrame, object]:
    frame = data.loc[specification['keep'](data)].copy()
    full_x = model_design(frame, specification['baseline'], specification['modifier'], specification.get('extra_covariates', []))
    result = sm.OLS(frame[specification['outcome']].to_numpy(float), full_x).fit(cov_type='HC3')
    nuisance = [name for name in full_x.columns if name not in {'const', 'group_tms', 'lesion_gradient_z', 'group_x_lesion_gradient'}]
    adjusted = frame[specification['outcome']].to_numpy(float).copy()
    for name in nuisance:
        adjusted -= result.params[name] * full_x[name].to_numpy(float)
    frame['adjusted_outcome'] = adjusted
    return (frame, result)

def make_figure(data: pd.DataFrame, model_results: pd.DataFrame) -> None:
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'], 'font.size': 7, 'axes.titlesize': 8, 'axes.labelsize': 7, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    (figure, axes) = plt.subplots(1, 2, figsize=(183 / 25.4, 78 / 25.4))
    chosen = [specifications()[0], specifications()[1]]
    colors = {'Control': '#777777', 'TMS': '#3F78A8'}
    markers = {'Control': 'o', 'TMS': 's'}
    for (axis, spec, panel) in zip(axes, chosen, ['a', 'b']):
        (frame, result) = adjusted_plot_data(data, spec)
        modifier = spec['modifier']
        modifier_mean = float(data[modifier].mean())
        modifier_sd = float(data[modifier].std(ddof=1))
        x_grid = np.linspace(float(frame[modifier].min()), float(frame[modifier].max()), 100)
        z_grid = (x_grid - modifier_mean) / modifier_sd
        for (group_name, group_value) in (('Control', 0), ('TMS', 1)):
            subset = frame.loc[frame['group_tms'] == group_value]
            axis.scatter(subset[modifier], subset['adjusted_outcome'], s=20, marker=markers[group_name], color=colors[group_name], edgecolor='white', linewidth=0.45, alpha=0.86, label=group_name)
            prediction = result.params['const'] + result.params['group_tms'] * group_value + result.params['lesion_gradient_z'] * z_grid + result.params['group_x_lesion_gradient'] * group_value * z_grid
            axis.plot(x_grid, prediction, color=colors[group_name], linewidth=1.35)
        result_row = model_results.loc[model_results['model_id'] == spec['model_id']].iloc[0]
        axis.text(0.02, 0.98, f"interaction beta={result_row['interaction_beta']:.3f} [{result_row['interaction_ci_low_hc3']:.3f}, {result_row['interaction_ci_high_hc3']:.3f}]\nP(HC3)={result_row['interaction_p_hc3']:.3f}; P(perm)={result_row['interaction_p_freedman_lane']:.3f}", transform=axis.transAxes, ha='left', va='top', fontsize=5.8)
        axis.set_xlabel('Projected lesion G1 span')
        if spec['outcome'] == 'G1_range_post':
            axis.set_ylabel('Post G1 robust range (covariate-adjusted)')
            axis.set_title('Gradient-range moderation', loc='left', fontweight='bold')
        else:
            axis.set_ylabel('Post NIHSS (covariate-adjusted)')
            axis.set_title('Clinical-outcome moderation', loc='left', fontweight='bold')
        axis.text(-0.12, 1.04, panel, transform=axis.transAxes, fontsize=9, fontweight='bold')
    axes[0].legend(loc='lower right', fontsize=6)
    figure.subplots_adjust(left=0.09, right=0.985, top=0.91, bottom=0.18, wspace=0.34)
    for suffix in ('svg', 'pdf', 'png'):
        figure.savefig(FIGURES / f'lesion_G1_span_tms_moderation.{suffix}', dpi=300 if suffix == 'png' else None)
    plt.close(figure)

def write_summary(results: pd.DataFrame, slopes: pd.DataFrame, contrasts: pd.DataFrame, overlap: pd.DataFrame, leave_one_out: pd.DataFrame) -> None:
    gradient = results.loc[results['model_id'] == 'gradient_G1_primary'].iloc[0]
    clinical = results.loc[results['model_id'] == 'clinical_NIHSS_primary'].iloc[0]
    clinical_qc = results.loc[results['model_id'] == 'clinical_NIHSS_lesion_qc'].iloc[0]
    clinical_adjusted = results.loc[results['model_id'] == 'clinical_NIHSS_adjusted'].iloc[0]
    normalization = results.loc[results['model_id'] == 'gradient_G1_normalization_supplementary'].iloc[0]
    gradient_g2 = results.loc[results['model_id'] == 'gradient_G2_supplementary'].iloc[0]
    clinical_adl = results.loc[results['model_id'] == 'clinical_ADL_supplementary'].iloc[0]
    overlap_row = overlap.loc[(overlap['modifier'] == PRIMARY_MODIFIER) & (overlap['group'] == 'TMS')].iloc[0]
    loo_gradient = leave_one_out.loc[leave_one_out['model_id'] == 'gradient_G1_primary']
    loo_clinical = leave_one_out.loc[leave_one_out['model_id'] == 'clinical_NIHSS_primary']

    def interpretation(row: pd.Series) -> str:
        if row['interaction_p_freedman_lane'] < 0.05:
            return '置换检验支持交互'
        return '未获得稳定交互证据'
    text = f"# Lesion对TMS梯度与临床效应的调节分析\n\n## 预设方案\n\n- 主病灶指标：`xtract_projected_G1_span`，即基线病灶通过HCP-XTRACT规范纤维束终点投影后覆盖的健康reference G1范围。\n- 主梯度终点：治疗后皮层G1稳健范围（P95-P5），校正治疗前范围；主要运动QC队列n={int(gradient['n'])}。\n- 主临床终点：治疗后NIHSS，校正治疗前NIHSS；完整队列n={int(clinical['n'])}。\n- 两个模型均校正治疗组、病灶指标主效应和log病灶体积，核心检验为`TMS x projected G1 span`。\n- 推断：HC3稳健标准误及5,000次Freedman-Lane残差置换；不进行正式moderated mediation。\n\n## 主结果\n\n- G1范围模型：交互beta={gradient['interaction_beta']:.4f}，95% CI {gradient['interaction_ci_low_hc3']:.4f}至{gradient['interaction_ci_high_hc3']:.4f}，HC3 p={gradient['interaction_p_hc3']:.4f}，置换p={gradient['interaction_p_freedman_lane']:.4f}，partial R2={gradient['interaction_partial_r2']:.3f}；{interpretation(gradient)}。\n- NIHSS模型：交互beta={clinical['interaction_beta']:.4f}分/1 SD，95% CI {clinical['interaction_ci_low_hc3']:.4f}至{clinical['interaction_ci_high_hc3']:.4f}，HC3 p={clinical['interaction_p_hc3']:.4f}，置换p={clinical['interaction_p_freedman_lane']:.4f}，partial R2={clinical['interaction_partial_r2']:.3f}；{interpretation(clinical)}。\n\nNIHSS交互beta为负表示随着病灶投影G1跨度增加，TMS相对Control的治疗后NIHSS更低；beta为正则表示相对结局更差。G1范围交互只表示TMS相关的range变化不同，不能在没有健康reference方向判定时自动称为“正常化”。\n\n## 敏感性与补充结果\n\n- 排除3例病灶QC病例后，NIHSS交互beta={clinical_qc['interaction_beta']:.4f}，HC3 p={clinical_qc['interaction_p_hc3']:.4f}，置换p={clinical_qc['interaction_p_freedman_lane']:.4f}；两种推断不完全一致，只能视为敏感性提示。\n- 额外校正年龄、性别和ESRS后，NIHSS交互beta={clinical_adjusted['interaction_beta']:.4f}，HC3 p={clinical_adjusted['interaction_p_hc3']:.4f}，置换p={clinical_adjusted['interaction_p_freedman_lane']:.4f}；该模型参数更多且不是预设主模型，不能替代主结果。\n- 健康reference G1正常化补充模型交互beta={normalization['interaction_beta']:.4f}，置换p={normalization['interaction_p_freedman_lane']:.4f}；未支持病灶G1跨度调节TMS相关G1正常化。\n- G2范围补充模型置换p={gradient_g2['interaction_p_freedman_lane']:.4f}；ADL补充模型置换p={clinical_adl['interaction_p_freedman_lane']:.4f}，均无稳定交互证据。\n\n## 可估计性与限制\n\n- 直接`direct_lesion_G1`不可用于调节分析：TMS 0/22可计算，Control仅5/16可计算。\n- XTRACT投影G1跨度覆盖38/38例；共同支持区间为{overlap_row['common_support_low']:.4f}至{overlap_row['common_support_high']:.4f}。\n- XTRACT结果是规范纤维束终点投影代理，不是个体化tractography或完整disconnectome。\n- 样本仅38例，交互效应功效有限；敏感性模型只用于判断方向和稳定性，不能据其中最显著者改写主假设。\n- 逐例删除时，G1交互beta范围为{loo_gradient['interaction_beta'].min():.4f}至{loo_gradient['interaction_beta'].max():.4f}；NIHSS交互beta范围为{loo_clinical['interaction_beta'].min():.4f}至{loo_clinical['interaction_beta'].max():.4f}，其中{int((loo_clinical['interaction_beta'] < 0).sum())}/{len(loo_clinical)}次保持负方向。\n- 当前结果可以写成lesion topology与梯度/临床效应的调节证据或无证据，不能声称已经证明`lesion -> gradient -> clinical`中介链。\n\n## 输出\n\n- `tables/analysis_dataset.csv`：38例建模数据。\n- `tables/moderation_model_results.csv`：主分析、敏感性及补充模型。\n- `tables/group_specific_slopes.csv`：Control和TMS组内病灶指标斜率。\n- `tables/vif_diagnostics.csv`：各模型共线性诊断。\n- `tables/conditional_treatment_contrasts.csv`：病灶指标-1/0/+1 SD处的TMS-Control条件差异。\n- `tables/modifier_overlap.csv`：组内分布与共同支持。\n- `tables/leave_one_out_primary.csv`：两条主模型的逐例删除稳定性。\n- `figures/lesion_G1_span_tms_moderation.*`：交互可视化。\n"
    (OUTPUT_ROOT / 'RESULTS_ZH.md').write_text(text, encoding='utf-8')

def run(n_permutations: int) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    data = load_analysis_data()
    model_rows = []
    diagnostics = []
    contrasts = []
    for (index, specification) in enumerate(specifications()):
        (row, slope_vif_rows, contrast_rows) = fit_specification(data, specification, n_permutations, index * 10000)
        model_rows.append(row)
        diagnostics.extend(slope_vif_rows)
        contrasts.extend(contrast_rows)
    results = pd.DataFrame(model_rows)
    diagnostic_table = pd.DataFrame(diagnostics)
    group_slopes = diagnostic_table.loc[diagnostic_table['group'].notna()].dropna(axis=1, how='all')
    vif_diagnostics = diagnostic_table.loc[diagnostic_table['term'].notna()].dropna(axis=1, how='all')
    contrast_table = pd.DataFrame(contrasts)
    overlap = modifier_overlap(data)
    leave_one_out = leave_one_out_primary(data)
    data.to_csv(TABLES / 'analysis_dataset.csv', index=False, encoding='utf-8-sig')
    results.to_csv(TABLES / 'moderation_model_results.csv', index=False, encoding='utf-8-sig')
    group_slopes.to_csv(TABLES / 'group_specific_slopes.csv', index=False, encoding='utf-8-sig')
    vif_diagnostics.to_csv(TABLES / 'vif_diagnostics.csv', index=False, encoding='utf-8-sig')
    contrast_table.to_csv(TABLES / 'conditional_treatment_contrasts.csv', index=False, encoding='utf-8-sig')
    overlap.to_csv(TABLES / 'modifier_overlap.csv', index=False, encoding='utf-8-sig')
    leave_one_out.to_csv(TABLES / 'leave_one_out_primary.csv', index=False, encoding='utf-8-sig')
    make_figure(data, results)
    write_summary(results, group_slopes, contrast_table, overlap, leave_one_out)
    manifest = {'analysis_date': '2026-08-04', 'primary_modifier': PRIMARY_MODIFIER, 'primary_gradient_model': 'G1_range_post ~ G1_range_pre + TMS + projected_G1_span + TMS*projected_G1_span + log1p_lesion_volume', 'primary_clinical_model': 'NIHSS_post ~ NIHSS_pre + TMS + projected_G1_span + TMS*projected_G1_span + log1p_lesion_volume', 'continuous_scaling': 'baseline, modifier and log lesion volume z-scored across the 38-person common cohort', 'robust_se': 'HC3', 'permutation': f'Freedman-Lane residual permutation under the no-interaction reduced model; {n_permutations} permutations', 'primary_gradient_n': int(results.loc[results['model_id'] == 'gradient_G1_primary', 'n'].iloc[0]), 'primary_clinical_n': int(results.loc[results['model_id'] == 'clinical_NIHSS_primary', 'n'].iloc[0]), 'direct_lesion_G1_not_estimable': 'TMS 0/22; Control 5/16', 'causal_limit': 'observational moderation evidence only; no moderated mediation claim', 'random_seed': SEED}
    (OUTPUT_ROOT / 'analysis_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(results.to_string(index=False))

def main() -> None:
    parser = argparse.ArgumentParser(description='Test whether projected lesion G1 span moderates TMS-related gradient and NIHSS outcomes.')
    parser.add_argument('--permutations', type=int, default=5000)
    arguments = parser.parse_args()
    if arguments.permutations < 0:
        raise ValueError('permutations must be non-negative')
    run(arguments.permutations)
if __name__ == '__main__':
    main()
