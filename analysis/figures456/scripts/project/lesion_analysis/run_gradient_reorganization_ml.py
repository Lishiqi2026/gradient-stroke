from __future__ import annotations
import os as _portable_os
from pathlib import Path as _PortablePath

def _portable_path(kind, relative):
    project = _PortablePath(_portable_os.environ['PAPER_PROJECT_ROOT'])
    root = project if kind == 'project' else project.parent
    return str(root / relative)
import argparse
import json
import sys
import warnings
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed
from scipy.linalg import orthogonal_procrustes
from statsmodels.stats.multitest import multipletests
PROJECT_ROOT = Path(_portable_path('project', ''))
ANALYSIS_ROOT = PROJECT_ROOT / 'lesion_analysis'
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))
import run_lesion_adjusted_gradient_value as base
OUTPUT_ROOT = ANALYSIS_ROOT / 'lesion_adjusted_gradient_reorganization'
TABLES = OUTPUT_ROOT / 'tables'
FIGURES = OUTPUT_ROOT / 'figures'
SPLITS = OUTPUT_ROOT / 'splits'
BLOCK_LABELS = {'clinical': 'Clinical', 'clinical_lesion': 'Clinical + lesion', 'clinical_gradient': 'Clinical + reorganization', 'combined': 'Clinical + lesion + reorganization'}
BLOCK_COLORS = {'clinical': '#777777', 'clinical_lesion': '#D07A3B', 'clinical_gradient': '#4C78A8', 'combined': '#248F78'}
REORGANIZATION_SUMMARIES = ('cortical_normalized_reorganization', 'subcortical_normalized_reorganization')

def ensure_dirs() -> None:
    for path in (TABLES, FIGURES, SPLITS):
        path.mkdir(parents=True, exist_ok=True)

def gradient_file(root: Path, group: str, session: str, subject_id: str) -> Path:
    matches = sorted((root / group / session).glob(f'{subject_id}-*.npy'))
    if len(matches) != 1:
        raise RuntimeError(f'Expected one gradient file for {root.name}/{group}/{session}/{subject_id}, found {len(matches)}')
    return matches[0]

def aligned_reorganization(before: np.ndarray, after: np.ndarray, expected_regions: int, subject_id: str, modality: str) -> tuple[np.ndarray, dict[str, float]]:
    before = np.asarray(before, dtype=float)
    after = np.asarray(after, dtype=float)
    if before.shape != (expected_regions, 5) or after.shape != (expected_regions, 5):
        raise RuntimeError(f'Unexpected {modality} gradient shape for {subject_id}: {before.shape}, {after.shape}')
    if not np.all(np.isfinite(before)) or not np.all(np.isfinite(after)):
        raise RuntimeError(f'Non-finite {modality} gradients for {subject_id}')
    before_centered = before - before.mean(axis=0, keepdims=True)
    after_centered = after - after.mean(axis=0, keepdims=True)
    (rotation, _) = orthogonal_procrustes(after_centered, before_centered)
    after_aligned = after_centered @ rotation
    residual = after_aligned - before_centered
    baseline_dispersion = float(np.sqrt(np.mean(before_centered ** 2)))
    if baseline_dispersion <= 0:
        raise RuntimeError(f'Zero {modality} baseline gradient dispersion for {subject_id}')
    normalized = residual / baseline_dispersion
    metrics = {f'{modality}_baseline_dispersion': baseline_dispersion, f'{modality}_reorganization_rms': float(np.sqrt(np.mean(residual ** 2))), f'{modality}_normalized_reorganization': float(np.sqrt(np.mean(normalized ** 2))), f'{modality}_alignment_fit_correlation': float(np.corrcoef(before_centered.ravel(), after_aligned.ravel())[0, 1])}
    return (normalized[:, :2].reshape(-1), metrics)

def load_reorganization_inputs() -> base.PreparedData:
    baseline = base.load_inputs()
    cortical_rows = []
    subcortical_rows = []
    metric_rows = []
    for row in baseline.table.itertuples():
        cortical_before = np.load(gradient_file(base.CORTICAL_ROOT, row.group, 'before', row.subject_id))
        cortical_after = np.load(gradient_file(base.CORTICAL_ROOT, row.group, 'after', row.subject_id))
        subcortical_before = np.load(gradient_file(base.SUBCORTICAL_ROOT, row.group, 'before', row.subject_id))
        subcortical_after = np.load(gradient_file(base.SUBCORTICAL_ROOT, row.group, 'after', row.subject_id))
        (cortical, cortical_metrics) = aligned_reorganization(cortical_before, cortical_after, 400, row.subject_id, 'cortical')
        (subcortical, subcortical_metrics) = aligned_reorganization(subcortical_before, subcortical_after, 54, row.subject_id, 'subcortical')
        cortical_rows.append(cortical)
        subcortical_rows.append(subcortical)
        metric_rows.append({'subject_id': row.subject_id, **cortical_metrics, **subcortical_metrics})
    table = baseline.table.merge(pd.DataFrame(metric_rows), on='subject_id', validate='one_to_one')
    if table[list(REORGANIZATION_SUMMARIES)].isna().any().any():
        raise RuntimeError('Missing reorganization summary metrics')
    return base.PreparedData(table=table, cortical=np.vstack(cortical_rows), subcortical=np.vstack(subcortical_rows))

def fit_reorganization_hc3(prepared: base.PreparedData, cohort: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = prepared.table
    base_columns = ['baseline_outcome_z', 'group_tms', 'age_z', 'sex_male', 'esrs_z', *[f'{feature}_z' for feature in base.LESION_FEATURES]]
    coefficient_rows = []
    joint_rows = []
    for outcome in base.OUTCOMES:
        direction = -1.0 if outcome == 'NIHSS' else 1.0
        baseline_outcome = direction * table[f'{outcome}_pre'].to_numpy(float)
        post_outcome = direction * table[f'{outcome}_post'].to_numpy(float)
        mean = float(np.mean(baseline_outcome))
        sd = float(np.std(baseline_outcome, ddof=1))
        model_data = pd.DataFrame({'baseline_outcome_z': (baseline_outcome - mean) / sd, 'group_tms': table['group_tms'].to_numpy(float), 'age_z': base.zscore(table['age']), 'sex_male': table['sex_male'].to_numpy(float), 'esrs_z': base.zscore(table['esrs'])})
        for feature in base.LESION_FEATURES:
            model_data[f'{feature}_z'] = base.zscore(table[feature])
        for feature in REORGANIZATION_SUMMARIES:
            model_data[f'{feature}_z'] = base.zscore(table[feature])
        y = (post_outcome - mean) / sd
        base_x = sm.add_constant(model_data[base_columns], has_constant='add')
        full_columns = base_columns + [f'{name}_z' for name in REORGANIZATION_SUMMARIES]
        full_x = sm.add_constant(model_data[full_columns], has_constant='add')
        base_result = sm.OLS(y, base_x).fit(cov_type='HC3')
        full_result = sm.OLS(y, full_x).fit(cov_type='HC3')
        restrictions = np.zeros((len(REORGANIZATION_SUMMARIES), len(full_result.params)))
        parameter_names = list(full_result.params.index)
        for (row_index, feature) in enumerate(REORGANIZATION_SUMMARIES):
            restrictions[row_index, parameter_names.index(f'{feature}_z')] = 1.0
        wald = full_result.wald_test(restrictions, use_f=True, scalar=True)
        partial_r2 = max(0.0, 1.0 - float(np.sum(full_result.resid ** 2)) / float(np.sum(base_result.resid ** 2)))
        joint_rows.append({'cohort': cohort, 'outcome': outcome, 'n': int(full_result.nobs), 'test': 'cortical and subcortical normalized reorganization added jointly', 'wald_f': float(wald.statistic), 'df_num': len(REORGANIZATION_SUMMARIES), 'df_denom': float(full_result.df_resid), 'p_value_hc3': float(wald.pvalue), 'partial_r2_in_sample': partial_r2, 'base_r_squared': float(base_result.rsquared), 'full_r_squared': float(full_result.rsquared)})
        confidence = full_result.conf_int()
        for feature in REORGANIZATION_SUMMARIES:
            term = f'{feature}_z'
            coefficient_rows.append({'cohort': cohort, 'outcome': outcome, 'n': int(full_result.nobs), 'term': feature, 'standardized_beta': float(full_result.params[term]), 'ci_low': float(confidence.loc[term, 0]), 'ci_high': float(confidence.loc[term, 1]), 'robust_se_hc3': float(full_result.bse[term]), 'p_value': float(full_result.pvalues[term])})
    coefficients = pd.DataFrame(coefficient_rows)
    joint = pd.DataFrame(joint_rows)
    coefficients['fdr_q'] = np.nan
    joint['fdr_q'] = np.nan
    for cohort_name in coefficients['cohort'].unique():
        selected = coefficients['cohort'] == cohort_name
        coefficients.loc[selected, 'fdr_q'] = multipletests(coefficients.loc[selected, 'p_value'], method='fdr_bh')[1]
        selected_joint = joint['cohort'] == cohort_name
        joint.loc[selected_joint, 'fdr_q'] = multipletests(joint.loc[selected_joint, 'p_value_hc3'], method='fdr_bh')[1]
    return (coefficients, joint)

def relabel_increments(increments: pd.DataFrame) -> pd.DataFrame:
    result = increments.copy()
    result['contrast'] = result['contrast'].str.replace('gradient', 'gradient reorganization', regex=False)
    return result

def plot_results(performance: pd.DataFrame, increments: pd.DataFrame) -> None:
    primary = performance.loc[performance['cohort'] == 'primary_full']
    (fig, axes) = plt.subplots(2, 2, figsize=(7.6, 5.4), facecolor='white', gridspec_kw={'width_ratios': (1.0, 1.08)})
    rng = np.random.default_rng(base.SEED)
    for (row_index, outcome) in enumerate(base.OUTCOMES):
        outcome_frame = primary.loc[primary['outcome'] == outcome]
        for (position, block) in enumerate(base.BLOCKS):
            values = outcome_frame.loc[outcome_frame['block'] == block, 'post_mae'].to_numpy(float)
            jitter = rng.uniform(-0.07, 0.07, len(values))
            axes[row_index, 0].scatter(position + jitter, values, s=9, color=BLOCK_COLORS[block], alpha=0.45, edgecolor='none')
            mean = float(np.mean(values))
            (low, high) = np.percentile(values, [2.5, 97.5])
            axes[row_index, 0].errorbar(position, mean, yerr=[[mean - low], [high - mean]], fmt='o', markersize=4.5, color=BLOCK_COLORS[block], capsize=2, linewidth=1)
        axes[row_index, 0].set_xticks(range(len(base.BLOCKS)), ['Clinical', '+ lesion', '+ reorganization', 'Combined'], rotation=12, ha='right')
        axes[row_index, 0].set_ylabel(f'{outcome} post-score MAE')
        axes[row_index, 0].set_title(f'{outcome}: repeated nested CV')
        contrast_names = ('lesion beyond clinical', 'gradient reorganization beyond clinical', 'gradient reorganization beyond clinical + lesion')
        increment_frame = increments.loc[(increments['cohort'] == 'primary_full') & (increments['outcome'] == outcome)]
        for (position, contrast) in enumerate(contrast_names):
            row = increment_frame.loc[increment_frame['contrast'] == contrast].iloc[0]
            axes[row_index, 1].errorbar(position, row['mean_mae_reduction'], yerr=[[row['mean_mae_reduction'] - row['repeat_interval_low']], [row['repeat_interval_high'] - row['mean_mae_reduction']]], fmt='o', markersize=5, color=('#D07A3B', '#4C78A8', '#248F78')[position], capsize=2, linewidth=1)
        axes[row_index, 1].axhline(0, color='#777777', linewidth=0.7, linestyle='--')
        axes[row_index, 1].set_xticks(range(3), ['Lesion\n| clinical', 'Reorganization\n| clinical', 'Reorganization\n| clinical + lesion'])
        axes[row_index, 1].set_ylabel('MAE reduction\n(positive = better)')
        axes[row_index, 1].set_title(f'{outcome}: incremental association')
    for (label, axis) in zip(('A', 'B', 'C', 'D'), axes.ravel()):
        axis.text(-0.14, 1.07, label, transform=axis.transAxes, fontsize=10, fontweight='bold', va='top')
        axis.spines[['top', 'right']].set_visible(False)
        axis.tick_params(width=0.6, length=2.5)
    fig.suptitle('Lesion-adjusted association of gradient reorganization with recovery', fontsize=10, y=0.995)
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.105, top=0.9, hspace=0.72, wspace=0.52)
    for (suffix, kwargs) in (('png', {'dpi': 300}), ('pdf', {})):
        fig.savefig(FIGURES / f'gradient_reorganization_recovery_ml.{suffix}', bbox_inches='tight', facecolor='white', **kwargs)
    plt.close(fig)

def write_summary(summary: pd.DataFrame, increments: pd.DataFrame, permutation: pd.DataFrame, joint: pd.DataFrame, coefficients: pd.DataFrame) -> None:
    lines = ['# 病灶校正后的梯度重组与临床恢复 ML 分析', '', '## 这项 ML 在回答什么', '', '本分析检验治疗前后梯度重组模式是否能够在临床和基线病灶信息之外解释治疗后 NIHSS/ADL。由于输入包含治疗后影像，它是同期恢复标志物分析，不是治疗前预测模型，也不能用于提前决定谁接受 TMS。', '', '- 38 例共同队列；敏感性分析排除 3 个病灶人工复核病例。', '- 每例治疗后五维梯度先正交 Procrustes 对齐到治疗前五维梯度。', '- 皮层 400 ROI 和皮层下 54 ROI 的 G1/G2 对齐残差除以各自基线五维离散度，形成 908 维重组模式。', '- 重组模式的标准化和 PCA 均仅在训练折内拟合；主模型为重复嵌套交叉验证 Ridge 回归。', '', '## 主要性能', '', '| 结局 | 模型 | post MAE | post R2 | 改善量 R2 |', '|---|---|---:|---:|---:|']
    primary = summary.loc[summary['cohort'] == 'primary_full']
    for outcome in base.OUTCOMES:
        for block in base.BLOCKS:
            lookup = primary.loc[(primary['outcome'] == outcome) & (primary['block'] == block)].set_index('metric')['mean']
            lines.append(f"| {outcome} | {BLOCK_LABELS[block]} | {lookup['post_mae']:.3f} | {lookup['post_r2']:.3f} | {lookup['change_r2']:.3f} |")
    lines.extend(['', '## 重组模式在临床和病灶之上的增量', ''])
    primary_contrast = 'gradient reorganization beyond clinical + lesion'
    for outcome in base.OUTCOMES:
        row = increments.loc[(increments['cohort'] == 'primary_full') & (increments['outcome'] == outcome) & (increments['contrast'] == primary_contrast)].iloc[0]
        direction = '降低' if row['mean_mae_reduction'] > 0 else '增加'
        lines.append(f"- {outcome}：加入梯度重组后平均{direction} MAE {abs(row['mean_mae_reduction']):.3f}；重复区间 {row['repeat_interval_low']:.3f} 到 {row['repeat_interval_high']:.3f}；{row['fraction_repeats_improved'] * 100:.1f}% 重复划分改善。")
    if not permutation.empty:
        lines.extend(['', '## 条件重组块置换检验', ''])
        for row in permutation.itertuples():
            lines.append(f'- {row.outcome}：实际 MAE reduction={row.observed_mae_reduction:.3f}，组内重组模式置换单侧 P={row.one_sided_p_value:.3f}。')
    lines.extend(['', '## 低维解释性模型', ''])
    for row in joint.loc[joint['cohort'] == 'primary_full'].itertuples():
        lines.append(f'- {row.outcome}：皮层和皮层下全局标准化重组量联合 HC3 检验 P={row.p_value_hc3:.3f}，FDR q={row.fdr_q:.3f}，样本内 partial R2={row.partial_r2_in_sample:.3f}。')
    primary_cortical = coefficients.loc[(coefficients['cohort'] == 'primary_full') & (coefficients['outcome'] == 'NIHSS') & (coefficients['term'] == 'cortical_normalized_reorganization')].iloc[0]
    sensitivity_cortical = coefficients.loc[(coefficients['cohort'] == 'exclude_manual_review_flags') & (coefficients['outcome'] == 'NIHSS') & (coefficients['term'] == 'cortical_normalized_reorganization')].iloc[0]
    sensitivity_joint = joint.loc[(joint['cohort'] == 'exclude_manual_review_flags') & (joint['outcome'] == 'NIHSS')].iloc[0]
    lines.extend(['', '### NIHSS 方向与敏感性', '', f"- 全队列皮层全局重组量 beta={primary_cortical['standardized_beta']:.3f}，95% CI {primary_cortical['ci_low']:.3f} 到 {primary_cortical['ci_high']:.3f}，P={primary_cortical['p_value']:.3f}，四项系数 FDR q={primary_cortical['fdr_q']:.3f}。", f"- 排除 3 个病灶 QC 病例后 beta={sensitivity_cortical['standardized_beta']:.3f}，95% CI {sensitivity_cortical['ci_low']:.3f} 到 {sensitivity_cortical['ci_high']:.3f}，FDR q={sensitivity_cortical['fdr_q']:.3f}；两项重组量联合检验 q={sensitivity_joint['fdr_q']:.3f}。", '- 模型已把 NIHSS 方向统一为数值越高表示恢复越好，因此负 beta 表示重组越大者恢复越差。该结果支持“过度/失调性重组”的探索性解释，而不是重组越大必然越好。'])
    lines.extend(['', '## 解释边界', '', '- 交叉验证只降低样本内过拟合风险，不会把治疗后影像变成治疗前预测指标。', '- 若重组模式有增量，可将其描述为恢复相关影像标志物；若无增量，则当前数据不支持它解释个体恢复差异。', '- 任何阳性结果仍需独立队列、统一扫描时间点和预注册特征定义复现。'])
    (OUTPUT_ROOT / 'RESULTS_ZH.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

def save_splits(name: str, prepared: base.PreparedData, splits: list[dict]) -> None:
    ids = prepared.table['subject_id'].to_numpy()
    records = [{'cohort': name, 'repeat': int(split['repeat']), 'fold': int(split['fold']), 'train_subject_ids': ids[split['train']].tolist(), 'test_subject_ids': ids[split['test']].tolist()} for split in splits]
    (SPLITS / f'nested_cv_splits_{name}.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Test lesion-adjusted associations between gradient reorganization and recovery.')
    parser.add_argument('--repeats', type=int, default=50)
    parser.add_argument('--permutations', type=int, default=100)
    parser.add_argument('--jobs', type=int, default=-1)
    parser.add_argument('--quick', action='store_true')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    if args.quick:
        args.repeats = 1
        args.permutations = min(args.permutations, 2)
    if args.repeats < 1 or args.permutations < 0:
        raise ValueError('repeats must be positive and permutations must be non-negative')
    ensure_dirs()
    primary = load_reorganization_inputs()
    sensitivity = base.subset_prepared(primary, ~primary.table['subject_id'].isin(base.MANUAL_REVIEW_FLAGS).to_numpy())
    cohorts = {'primary_full': primary, 'exclude_manual_review_flags': sensitivity}
    cohort_blocks = {'primary_full': base.BLOCKS, 'exclude_manual_review_flags': ('clinical_lesion', 'combined')}
    split_sets = {name: base.make_outer_splits(prepared.table['group_tms'].to_numpy(int), args.repeats, seed_offset=20000 + 1000 * index) for (index, (name, prepared)) in enumerate(cohorts.items())}
    for (name, prepared) in cohorts.items():
        save_splits(name, prepared, split_sets[name])
    export_columns = ['subject_id', 'group', 'manual_review_flag', 'NIHSS_pre', 'NIHSS_post', 'ADL_pre', 'ADL_post', *base.LESION_FEATURES, 'cortical_baseline_dispersion', 'cortical_reorganization_rms', 'cortical_normalized_reorganization', 'cortical_alignment_fit_correlation', 'subcortical_baseline_dispersion', 'subcortical_reorganization_rms', 'subcortical_normalized_reorganization', 'subcortical_alignment_fit_correlation']
    primary.table[export_columns].to_csv(TABLES / 'reorganization_cohort.csv', index=False, encoding='utf-8-sig')
    jobs_to_run = [(cohort_name, prepared, outcome, block, split_sets[cohort_name]) for (cohort_name, prepared) in cohorts.items() for outcome in base.OUTCOMES for block in cohort_blocks[cohort_name]]
    results = Parallel(n_jobs=args.jobs, verbose=10)((delayed(base.run_job)(*job) for job in jobs_to_run))
    predictions = pd.concat([result[0] for result in results], ignore_index=True)
    tuning = pd.concat([result[1] for result in results], ignore_index=True)
    performance = base.repeat_performance(predictions)
    summary = base.summarize_performance(performance)
    increments = relabel_increments(base.incremental_mae(performance))
    coefficient_frames = []
    joint_frames = []
    for (cohort_name, prepared) in cohorts.items():
        (coefficients, joint) = fit_reorganization_hc3(prepared, cohort_name)
        coefficient_frames.append(coefficients)
        joint_frames.append(joint)
    coefficients = pd.concat(coefficient_frames, ignore_index=True)
    joint = pd.concat(joint_frames, ignore_index=True)
    (permutation, permutation_null) = base.run_permutation_tests(primary, predictions, split_sets['primary_full'], args.permutations, args.jobs)
    if not permutation.empty:
        permutation['test'] = permutation['test'].str.replace('gradient block', 'gradient reorganization block', regex=False)
        permutation['permutation_scheme'] = 'aligned cortical and subcortical reorganization rows jointly shuffled within treatment group'
    predictions.to_csv(TABLES / 'nested_cv_predictions.csv', index=False, encoding='utf-8-sig')
    tuning.to_csv(TABLES / 'outer_fold_tuning.csv', index=False, encoding='utf-8-sig')
    performance.to_csv(TABLES / 'repeat_level_performance.csv', index=False, encoding='utf-8-sig')
    summary.to_csv(TABLES / 'performance_summary.csv', index=False, encoding='utf-8-sig')
    increments.to_csv(TABLES / 'incremental_mae_summary.csv', index=False, encoding='utf-8-sig')
    base.aggregate_oof(predictions).to_csv(TABLES / 'participant_mean_oof_predictions.csv', index=False, encoding='utf-8-sig')
    coefficients.to_csv(TABLES / 'hc3_reorganization_coefficients.csv', index=False, encoding='utf-8-sig')
    joint.to_csv(TABLES / 'hc3_reorganization_joint_tests.csv', index=False, encoding='utf-8-sig')
    if not permutation.empty:
        permutation.to_csv(TABLES / 'reorganization_block_permutation.csv', index=False, encoding='utf-8-sig')
        permutation_null.to_csv(TABLES / 'reorganization_block_permutation_null.csv', index=False, encoding='utf-8-sig')
    plot_results(performance, increments)
    write_summary(summary, increments, permutation, joint, coefficients)
    manifest = {'analysis_date': '2026-08-03', 'primary_cohort_n': int(len(primary.table)), 'primary_group_counts': primary.table['group'].value_counts().sort_index().to_dict(), 'sensitivity_cohort_n': int(len(sensitivity.table)), 'sensitivity_excluded_subjects': sorted(base.MANUAL_REVIEW_FLAGS), 'outcomes': ['NIHSS_post', 'ADL_post'], 'clinical_block': {outcome: base.clinical_columns(outcome) for outcome in base.OUTCOMES}, 'lesion_block': list(base.LESION_FEATURES), 'reorganization_block': {'timing': 'post-treatment minus baseline after within-subject 5D orthogonal Procrustes alignment', 'normalization': 'aligned residual divided by baseline five-dimensional RMS dispersion', 'raw_dimensions': {'cortical_G1G2': 800, 'subcortical_G1G2': 108}, 'foldwise_pca_candidates': [list(config) for config in base.PCA_CONFIGS]}, 'model': 'Ridge regression', 'outer_cv': f'stratified 5-fold repeated {args.repeats} times', 'inner_cv': 'stratified 4-fold tuning of Ridge alpha and reorganization PCA dimensions', 'reorganization_block_permutations': args.permutations, 'primary_contrast': 'clinical+lesion versus clinical+lesion+gradient reorganization', 'leakage_control': 'scaling, PCA, regularization tuning, and fitting occur only within training folds', 'interpretation': 'cross-validated concurrent recovery-marker association; not prospective baseline prediction', 'post_treatment_imaging_used': True, 'external_validation': False, 'random_seed': base.SEED}
    (OUTPUT_ROOT / 'analysis_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Primary cohort: {len(primary.table)}')
    print(f'Sensitivity cohort: {len(sensitivity.table)}')
    print(f'Prediction rows: {len(predictions)}')
    print(f'Output: {OUTPUT_ROOT}')
if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=FutureWarning)
        main()
