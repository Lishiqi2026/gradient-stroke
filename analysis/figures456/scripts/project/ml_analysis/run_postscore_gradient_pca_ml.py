from __future__ import annotations
import os as _portable_os
from pathlib import Path as _PortablePath

def _portable_path(kind, relative):
    project = _PortablePath(_portable_os.environ['PAPER_PROJECT_ROOT'])
    root = project if kind == 'project' else project.parent
    return str(root / relative)
import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
os.environ.setdefault('JOBLIB_TEMP_FOLDER', _portable_path('project', '.cache/joblib'))
Path(os.environ['JOBLIB_TEMP_FOLDER']).mkdir(parents=True, exist_ok=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
try:
    from ml_analysis.run_clinical_gradient_ml import CORTICAL_ROOT, OUTPUT_ROOT as CHANGE_SCORE_OUTPUT_ROOT, SEED, SUBCORTICAL_ROOT, find_baseline_arrays, load_dataset
except ModuleNotFoundError:
    from run_clinical_gradient_ml import CORTICAL_ROOT, OUTPUT_ROOT as CHANGE_SCORE_OUTPUT_ROOT, SEED, SUBCORTICAL_ROOT, find_baseline_arrays, load_dataset
OUTPUT_ROOT = CHANGE_SCORE_OUTPUT_ROOT.parent / 'postscore_gradient_pca'
OUTCOMES = ('NIHSS', 'ADL')
BASELINE_COLUMNS = ['NIHSS_pre', 'ADL_pre', 'HAMD_pre', 'MMSE_pre', 'VFT_pre']
BLOCK_LABELS = {'baseline_score': 'Baseline score', 'clinical_compact': 'Compact clinical', 'clinical_expanded': 'Expanded clinical', 'clinical_expanded_cortical_pca': 'Expanded clinical + cortical PCA', 'clinical_expanded_all_gradient_pca': 'Expanded clinical + cortical/subcortical PCA'}
COLORS = {'baseline_score': '#7F7F7F', 'clinical_compact': '#2B6F8A', 'clinical_expanded': '#4C78A8', 'clinical_expanded_cortical_pca': '#D07A3B', 'clinical_expanded_all_gradient_pca': '#248F78'}

@dataclass(frozen=True)
class BlockSpec:
    clinical_columns: tuple[str, ...]
    pca_configs: tuple[tuple[int, int], ...]

@dataclass
class PreparedData:
    table: pd.DataFrame
    cortical: np.ndarray | None
    subcortical: np.ndarray | None

def block_spec(outcome: str, block: str) -> BlockSpec:
    outcome_pre = f'{outcome}_pre'
    compact = (outcome_pre, 'age', 'sex_male', 'group_tms', 'esrs')
    expanded = tuple(BASELINE_COLUMNS + ['age', 'sex_male', 'group_tms', 'esrs'])
    if block == 'baseline_score':
        return BlockSpec((outcome_pre,), ((0, 0),))
    if block == 'clinical_compact':
        return BlockSpec(compact, ((0, 0),))
    if block == 'clinical_expanded':
        return BlockSpec(expanded, ((0, 0),))
    if block == 'clinical_expanded_cortical_pca':
        return BlockSpec(expanded, ((2, 0), (5, 0), (10, 0)))
    if block == 'clinical_expanded_all_gradient_pca':
        return BlockSpec(expanded, ((2, 2), (5, 2), (5, 5), (10, 5)))
    raise KeyError(block)

def load_inputs() -> tuple[PreparedData, PreparedData, dict]:
    (full, source_metadata) = load_dataset()
    full = full.sort_values('subject_id').reset_index(drop=True)
    common = full[full['has_baseline_gradient']].copy().sort_values('subject_id').reset_index(drop=True)
    cortical_paths = find_baseline_arrays(CORTICAL_ROOT)
    subcortical_paths = find_baseline_arrays(SUBCORTICAL_ROOT)
    cortical_rows = []
    subcortical_rows = []
    for subject_id in common['subject_id']:
        cortical = np.asarray(np.load(cortical_paths[subject_id])[:, :2], dtype=float)
        subcortical = np.asarray(np.load(subcortical_paths[subject_id])[:, :2], dtype=float)
        if cortical.shape != (400, 2) or subcortical.shape != (54, 2):
            raise ValueError(f'Unexpected gradient shape for {subject_id}: {cortical.shape}, {subcortical.shape}')
        if not np.all(np.isfinite(cortical)) or not np.all(np.isfinite(subcortical)):
            raise ValueError(f'Non-finite baseline gradient values for {subject_id}')
        cortical_rows.append(cortical.reshape(-1))
        subcortical_rows.append(subcortical.reshape(-1))
    clinical_export_columns = ['subject_id', 'group', 'group_tms', 'age', 'sex', 'sex_male', 'esrs', *BASELINE_COLUMNS, 'NIHSS_post', 'ADL_post', 'has_baseline_gradient']
    export_table = full[clinical_export_columns].copy()
    metadata = {'clinical_n': int(len(full)), 'gradient_common_n': int(len(common)), 'clinical_group_counts': full['group'].value_counts().sort_index().to_dict(), 'gradient_group_counts': common['group'].value_counts().sort_index().to_dict(), 'clinical_without_gradient': source_metadata['clinical_without_gradient'], 'outcomes': {'NIHSS': 'NIHSS_post', 'ADL': 'ADL_post'}, 'lesion_features_used': False, 'post_treatment_imaging_used': False, 'raw_gradient_dimensions': {'cortical_G1G2': 800, 'subcortical_G1G2': 108}, 'export_table_columns': clinical_export_columns}
    return (PreparedData(full, None, None), PreparedData(common, np.vstack(cortical_rows), np.vstack(subcortical_rows)), {'metadata': metadata, 'export_table': export_table})

def make_outer_splits(groups: np.ndarray, repeats: int) -> list[dict]:
    records = []
    dummy = np.zeros(len(groups))
    for repeat in range(repeats):
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + repeat)
        for (fold, (train, test)) in enumerate(splitter.split(dummy, groups)):
            records.append({'repeat': repeat, 'fold': fold, 'train': train, 'test': test})
    return records

def make_inner_splits(groups: np.ndarray, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(groups)), groups))

def transform_parts(clinical_train: np.ndarray, clinical_test: np.ndarray, cortical_train: np.ndarray | None, cortical_test: np.ndarray | None, subcortical_train: np.ndarray | None, subcortical_test: np.ndarray | None, cortical_components: int, subcortical_components: int) -> tuple[np.ndarray, np.ndarray]:
    clinical_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
    train_parts = [clinical_pipe.fit_transform(clinical_train)]
    test_parts = [clinical_pipe.transform(clinical_test)]
    if cortical_components:
        if cortical_train is None or cortical_test is None:
            raise ValueError('Cortical PCA requested without cortical data')
        cortical_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scale', StandardScaler()), ('pca', PCA(n_components=cortical_components, svd_solver='full'))])
        train_parts.append(cortical_pipe.fit_transform(cortical_train))
        test_parts.append(cortical_pipe.transform(cortical_test))
    if subcortical_components:
        if subcortical_train is None or subcortical_test is None:
            raise ValueError('Subcortical PCA requested without subcortical data')
        subcortical_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scale', StandardScaler()), ('pca', PCA(n_components=subcortical_components, svd_solver='full'))])
        train_parts.append(subcortical_pipe.fit_transform(subcortical_train))
        test_parts.append(subcortical_pipe.transform(subcortical_test))
    return (np.column_stack(train_parts), np.column_stack(test_parts))

def model_candidates(algorithm: str) -> list[dict]:
    if algorithm == 'ridge':
        return [{'alpha': float(alpha)} for alpha in np.logspace(3, -3, 7)]
    if algorithm == 'elasticnet':
        candidates = []
        for alpha in np.logspace(1, -2, 4):
            for l1_ratio in (0.5, 0.1, 0.9):
                candidates.append({'alpha': float(alpha), 'l1_ratio': float(l1_ratio)})
        return candidates
    raise KeyError(algorithm)

def make_model(algorithm: str, parameters: dict):
    if algorithm == 'ridge':
        return Ridge(alpha=parameters['alpha'], max_iter=100000)
    if algorithm == 'elasticnet':
        return ElasticNet(alpha=parameters['alpha'], l1_ratio=parameters['l1_ratio'], max_iter=50000, tol=0.0001, random_state=SEED)
    raise KeyError(algorithm)

def tune_and_predict(prepared: PreparedData, outcome: str, block: str, algorithm: str, split: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    spec = block_spec(outcome, block)
    table = prepared.table
    clinical = table[list(spec.clinical_columns)].to_numpy(dtype=float)
    target = table[f'{outcome}_post'].to_numpy(dtype=float)
    groups = table['group_tms'].to_numpy(dtype=int)
    (outer_train, outer_test) = (split['train'], split['test'])
    inner_splits = make_inner_splits(groups[outer_train], SEED + 10000 + split['repeat'] * 10 + split['fold'])
    candidates = model_candidates(algorithm)
    best_score = np.inf
    best_config = None
    best_parameters = None
    for (cortical_components, subcortical_components) in spec.pca_configs:
        transformed_inner = []
        for (inner_train_local, inner_valid_local) in inner_splits:
            inner_train = outer_train[inner_train_local]
            inner_valid = outer_train[inner_valid_local]
            transformed_inner.append((*transform_parts(clinical[inner_train], clinical[inner_valid], None if prepared.cortical is None else prepared.cortical[inner_train], None if prepared.cortical is None else prepared.cortical[inner_valid], None if prepared.subcortical is None else prepared.subcortical[inner_train], None if prepared.subcortical is None else prepared.subcortical[inner_valid], cortical_components, subcortical_components), target[inner_train], target[inner_valid]))
        for parameters in candidates:
            fold_errors = []
            for (x_train, x_valid, y_train, y_valid) in transformed_inner:
                model = make_model(algorithm, parameters)
                model.fit(x_train, y_train)
                fold_errors.append(mean_absolute_error(y_valid, model.predict(x_valid)))
            score = float(np.mean(fold_errors))
            if score < best_score - 1e-12:
                best_score = score
                best_config = (cortical_components, subcortical_components)
                best_parameters = parameters
    if best_config is None or best_parameters is None:
        raise RuntimeError('No model candidate was selected')
    (x_train, x_test) = transform_parts(clinical[outer_train], clinical[outer_test], None if prepared.cortical is None else prepared.cortical[outer_train], None if prepared.cortical is None else prepared.cortical[outer_test], None if prepared.subcortical is None else prepared.subcortical[outer_train], None if prepared.subcortical is None else prepared.subcortical[outer_test], best_config[0], best_config[1])
    final_model = make_model(algorithm, best_parameters)
    final_model.fit(x_train, target[outer_train])
    prediction = np.asarray(final_model.predict(x_test), dtype=float).reshape(-1)
    tuning = {'repeat': split['repeat'], 'fold': split['fold'], 'inner_mae': best_score, 'cortical_pca_components': best_config[0], 'subcortical_pca_components': best_config[1], 'model_parameters': json.dumps(best_parameters, sort_keys=True)}
    return (outer_test, prediction, tuning)

def finite_correlation(function, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 3 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return float(function(y_true, y_pred).statistic)

def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if np.std(y_pred) > 0:
        (slope, intercept) = np.polyfit(y_pred, y_true, 1)
    else:
        (slope, intercept) = (np.nan, np.nan)
    return {'mae': float(mean_absolute_error(y_true, y_pred)), 'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))), 'r2': float(r2_score(y_true, y_pred)), 'pearson_r': finite_correlation(pearsonr, y_true, y_pred), 'spearman_rho': finite_correlation(spearmanr, y_true, y_pred), 'calibration_intercept': float(intercept), 'calibration_slope': float(slope)}

def improvement_values(outcome: str, baseline: np.ndarray, post: np.ndarray) -> np.ndarray:
    if outcome == 'NIHSS':
        return baseline - post
    if outcome == 'ADL':
        return post - baseline
    raise KeyError(outcome)

def run_nested_cv(prepared: PreparedData, cohort: str, outcome: str, block: str, algorithm: str, splits: list[dict], jobs: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print(f'{cohort} | {algorithm} | {outcome} | {block}', flush=True)
    results = Parallel(n_jobs=jobs, prefer='processes', max_nbytes='100K')((delayed(tune_and_predict)(prepared, outcome, block, algorithm, split) for split in splits))
    table = prepared.table
    target = table[f'{outcome}_post'].to_numpy(dtype=float)
    baseline = table[f'{outcome}_pre'].to_numpy(dtype=float)
    subject_ids = table['subject_id'].to_numpy()
    prediction_rows = []
    tuning_rows = []
    for (split, (test, prediction, tuning)) in zip(splits, results):
        observed_improvement = improvement_values(outcome, baseline[test], target[test])
        predicted_improvement = improvement_values(outcome, baseline[test], prediction)
        for (local, row_index) in enumerate(test):
            prediction_rows.append({'cohort': cohort, 'algorithm': algorithm, 'outcome': outcome, 'block': block, 'repeat': split['repeat'], 'fold': split['fold'], 'subject_id': subject_ids[row_index], 'group': table.iloc[row_index]['group'], 'observed_post_score': target[row_index], 'predicted_post_score': prediction[local], 'observed_improvement': observed_improvement[local], 'predicted_improvement': predicted_improvement[local]})
        tuning_rows.append({'cohort': cohort, 'algorithm': algorithm, 'outcome': outcome, 'block': block, **tuning})
    predictions = pd.DataFrame(prediction_rows)
    repeat_rows = []
    for (repeat, frame) in predictions.groupby('repeat'):
        post_metrics = metrics(frame['observed_post_score'].to_numpy(), frame['predicted_post_score'].to_numpy())
        improvement_metrics = metrics(frame['observed_improvement'].to_numpy(), frame['predicted_improvement'].to_numpy())
        repeat_rows.append({'cohort': cohort, 'algorithm': algorithm, 'outcome': outcome, 'block': block, 'repeat': repeat, **{f'post_{key}': value for (key, value) in post_metrics.items()}, **{f'improvement_{key}': value for (key, value) in improvement_metrics.items()}})
    return (predictions, pd.DataFrame(tuning_rows), pd.DataFrame(repeat_rows))

def aggregate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    keys = ['cohort', 'algorithm', 'outcome', 'block', 'subject_id', 'group']
    return predictions.groupby(keys, as_index=False).agg(observed_post_score=('observed_post_score', 'first'), predicted_post_score=('predicted_post_score', 'mean'), predicted_post_sd=('predicted_post_score', 'std'), observed_improvement=('observed_improvement', 'first'), predicted_improvement=('predicted_improvement', 'mean'), predicted_improvement_sd=('predicted_improvement', 'std'), n_predictions=('predicted_post_score', 'size'))

def summarize_repeats(repeat_metrics: pd.DataFrame) -> pd.DataFrame:
    keys = ['cohort', 'algorithm', 'outcome', 'block']
    metric_columns = [column for column in repeat_metrics if column.startswith(('post_', 'improvement_'))]
    rows = []
    for (values, frame) in repeat_metrics.groupby(keys, sort=False):
        row = {**dict(zip(keys, values)), 'n_repeats': int(frame['repeat'].nunique())}
        for column in metric_columns:
            observed = frame[column].dropna().to_numpy(dtype=float)
            row[f'{column}_mean'] = float(np.mean(observed))
            row[f'{column}_median'] = float(np.median(observed))
            row[f'{column}_p025'] = float(np.quantile(observed, 0.025))
            row[f'{column}_p975'] = float(np.quantile(observed, 0.975))
        rows.append(row)
    return pd.DataFrame(rows)

def incremental_summary(repeat_metrics: pd.DataFrame) -> pd.DataFrame:
    frame = repeat_metrics[(repeat_metrics['cohort'] == 'gradient_common') & (repeat_metrics['algorithm'] == 'ridge')]
    pivot = frame.pivot_table(index=['outcome', 'repeat'], columns='block', values='post_mae')
    rows = []
    for outcome in OUTCOMES:
        outcome_frame = pivot.loc[outcome]
        for reference in ('clinical_compact', 'clinical_expanded'):
            for block in ('clinical_expanded_cortical_pca', 'clinical_expanded_all_gradient_pca'):
                difference = outcome_frame[reference] - outcome_frame[block]
                rows.append({'outcome': outcome, 'reference': reference, 'comparison_block': block, 'positive_means_pca_improves_mae': True, 'mean_mae_reduction': float(difference.mean()), 'median_mae_reduction': float(difference.median()), 'repeat_interval_low': float(difference.quantile(0.025)), 'repeat_interval_high': float(difference.quantile(0.975)), 'fraction_repeats_improved': float(np.mean(difference > 0)), 'n_repeats': len(difference)})
    return pd.DataFrame(rows)

def style_axis(ax: plt.Axes) -> None:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', color='#D9D9D9', linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)

def plot_model_comparison(repeat_metrics: pd.DataFrame, output: Path) -> None:
    frame = repeat_metrics[(repeat_metrics['cohort'] == 'gradient_common') & (repeat_metrics['algorithm'] == 'ridge')]
    blocks = tuple(BLOCK_LABELS)
    rng = np.random.default_rng(SEED)
    (fig, axes) = plt.subplots(1, 2, figsize=(11.3, 4.4), constrained_layout=True)
    for (ax, outcome) in zip(axes, OUTCOMES):
        values = [frame[(frame['outcome'] == outcome) & (frame['block'] == block)]['post_mae'].to_numpy() for block in blocks]
        boxes = ax.boxplot(values, positions=np.arange(len(blocks)), widths=0.56, patch_artist=True, showfliers=False)
        for (patch, block) in zip(boxes['boxes'], blocks):
            patch.set_facecolor(COLORS[block])
            patch.set_alpha(0.72)
            patch.set_edgecolor('#333333')
        for (index, (values_block, block)) in enumerate(zip(values, blocks)):
            ax.scatter(index + rng.uniform(-0.11, 0.11, len(values_block)), values_block, s=10, color=COLORS[block], alpha=0.36, linewidth=0)
        ax.set_xticks(np.arange(len(blocks)), [BLOCK_LABELS[block] for block in blocks], rotation=25, ha='right')
        ax.set_ylabel(f'MAE ({outcome} post-score points)')
        ax.set_title(outcome)
        style_axis(ax)
    fig.suptitle('Baseline-adjusted post-score prediction: repeated nested CV', fontsize=12, fontweight='bold')
    for suffix in ('png', 'pdf'):
        fig.savefig(output.with_suffix(f'.{suffix}'), dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_observed_predicted(aggregated: pd.DataFrame, output: Path) -> None:
    blocks = ('clinical_expanded', 'clinical_expanded_all_gradient_pca')
    frame = aggregated[(aggregated['cohort'] == 'gradient_common') & (aggregated['algorithm'] == 'ridge') & aggregated['block'].isin(blocks)]
    (fig, axes) = plt.subplots(2, 2, figsize=(9.0, 7.6), constrained_layout=True)
    group_colors = {'Control': '#4C78A8', 'TMS': '#D55E00'}
    for (row, outcome) in enumerate(OUTCOMES):
        for (column, block) in enumerate(blocks):
            ax = axes[row, column]
            part = frame[(frame['outcome'] == outcome) & (frame['block'] == block)]
            for (group, group_frame) in part.groupby('group'):
                ax.scatter(group_frame['observed_post_score'], group_frame['predicted_post_score'], s=28, color=group_colors[group], label=group, alpha=0.85, edgecolor='white', linewidth=0.5)
            low = min(part['observed_post_score'].min(), part['predicted_post_score'].min())
            high = max(part['observed_post_score'].max(), part['predicted_post_score'].max())
            margin = max((high - low) * 0.06, 0.5)
            ax.plot([low - margin, high + margin], [low - margin, high + margin], color='#333333', linestyle='--', linewidth=0.9)
            result = metrics(part['observed_post_score'].to_numpy(), part['predicted_post_score'].to_numpy())
            ax.text(0.04, 0.96, f"MAE = {result['mae']:.2f}\n$R^2$ = {result['r2']:.2f}\nr = {result['pearson_r']:.2f}", transform=ax.transAxes, va='top', fontsize=8.5)
            ax.set_xlabel(f'Observed {outcome} post score')
            ax.set_ylabel('Mean out-of-fold prediction')
            if row == 0:
                ax.set_title(BLOCK_LABELS[block])
            style_axis(ax)
    (handles, labels) = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(handles, labels, frameon=False, loc='lower right')
    fig.suptitle('Post-score predictions: expanded clinical vs gradient-PCA model', fontsize=12, fontweight='bold')
    for suffix in ('png', 'pdf'):
        fig.savefig(output.with_suffix(f'.{suffix}'), dpi=300, bbox_inches='tight')
    plt.close(fig)

def save_splits(table: pd.DataFrame, cohort: str, splits: list[dict], output: Path) -> None:
    ids = table['subject_id'].to_numpy()
    records = []
    for split in splits:
        records.append({'cohort': cohort, 'repeat': int(split['repeat']), 'fold': int(split['fold']), 'train_subject_ids': ids[split['train']].tolist(), 'test_subject_ids': ids[split['test']].tolist()})
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Predict post-treatment scores using clinical data and fold-wise gradient PCA.')
    parser.add_argument('--repeats', type=int, default=50)
    parser.add_argument('--jobs', type=int, default=-1)
    parser.add_argument('--quick', action='store_true')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    if args.quick:
        args.repeats = 1
    if args.repeats < 1:
        raise ValueError('repeats must be positive')
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (clinical_all, gradient_common, loaded) = load_inputs()
    loaded['export_table'].to_csv(OUTPUT_ROOT / 'postscore_clinical_table.csv', index=False, encoding='utf-8-sig')
    cohorts = {'clinical_all': clinical_all, 'gradient_common': gradient_common}
    splits = {name: make_outer_splits(prepared.table['group_tms'].to_numpy(dtype=int), args.repeats) for (name, prepared) in cohorts.items()}
    for (name, prepared) in cohorts.items():
        save_splits(prepared.table, name, splits[name], OUTPUT_ROOT / f'nested_cv_splits_{name}.json')
    jobs_to_run = []
    for outcome in OUTCOMES:
        for block in ('baseline_score', 'clinical_compact', 'clinical_expanded'):
            jobs_to_run.append(('clinical_all', 'ridge', outcome, block))
        for block in BLOCK_LABELS:
            jobs_to_run.append(('gradient_common', 'ridge', outcome, block))
        for block in ('clinical_expanded', 'clinical_expanded_all_gradient_pca'):
            jobs_to_run.append(('gradient_common', 'elasticnet', outcome, block))
    all_predictions = []
    all_tuning = []
    all_repeat_metrics = []
    for (index, (cohort, algorithm, outcome, block)) in enumerate(jobs_to_run, start=1):
        print(f'[{index}/{len(jobs_to_run)}]', end=' ', flush=True)
        (predictions, tuning, repeat_metrics) = run_nested_cv(cohorts[cohort], cohort, outcome, block, algorithm, splits[cohort], args.jobs)
        all_predictions.append(predictions)
        all_tuning.append(tuning)
        all_repeat_metrics.append(repeat_metrics)
    predictions = pd.concat(all_predictions, ignore_index=True)
    tuning = pd.concat(all_tuning, ignore_index=True)
    repeat_metrics = pd.concat(all_repeat_metrics, ignore_index=True)
    aggregated = aggregate_predictions(predictions)
    summary = summarize_repeats(repeat_metrics)
    incremental = incremental_summary(repeat_metrics)
    predictions.to_csv(OUTPUT_ROOT / 'repeated_oof_predictions.csv', index=False, encoding='utf-8-sig')
    aggregated.to_csv(OUTPUT_ROOT / 'participant_mean_oof_predictions.csv', index=False, encoding='utf-8-sig')
    tuning.to_csv(OUTPUT_ROOT / 'outer_fold_tuning.csv', index=False, encoding='utf-8-sig')
    repeat_metrics.to_csv(OUTPUT_ROOT / 'repeat_level_performance.csv', index=False, encoding='utf-8-sig')
    summary.to_csv(OUTPUT_ROOT / 'repeat_performance_summary.csv', index=False, encoding='utf-8-sig')
    incremental.to_csv(OUTPUT_ROOT / 'incremental_mae_summary.csv', index=False, encoding='utf-8-sig')
    plot_model_comparison(repeat_metrics, OUTPUT_ROOT / 'figure_postscore_model_comparison')
    plot_observed_predicted(aggregated, OUTPUT_ROOT / 'figure_postscore_observed_predicted')
    manifest = {**loaded['metadata'], 'analysis_date': '2026-08-03', 'primary_task': 'baseline-adjusted prediction of post-treatment NIHSS and ADL scores', 'sensitivity_task': 'derived improvement predictions are included in output tables', 'outer_cv': f'stratified 5-fold repeated {args.repeats} times', 'inner_cv': 'stratified 4-fold tuning within each outer training fold', 'pca_rule': 'scaling and PCA fitted separately within every inner/outer training fold', 'pca_candidate_components': {'cortical_only': [[2, 0], [5, 0], [10, 0]], 'cortical_subcortical': [[2, 2], [5, 2], [5, 5], [10, 5]]}, 'algorithms': ['ridge', 'elasticnet'], 'random_seed': SEED, 'limitations': ['Internal validation only; no independent external cohort.', 'Treatment group conditions outcome prediction but does not estimate individual causal treatment benefit.', 'PCA components are fold-specific and should not be interpreted as stable anatomical biomarkers.', 'No lesion-derived variable is included.']}
    (OUTPUT_ROOT / 'analysis_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Completed. Outputs: {OUTPUT_ROOT}', flush=True)
if __name__ == '__main__':
    main()
