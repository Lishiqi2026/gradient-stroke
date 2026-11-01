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
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
os.environ.setdefault('JOBLIB_TEMP_FOLDER', _portable_path('project', '.cache/joblib'))
Path(os.environ['JOBLIB_TEMP_FOLDER']).mkdir(parents=True, exist_ok=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import pearsonr, spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
ROOT = Path(_portable_path('project', ''))
CLINICAL_PATH = Path(_portable_path('work', '董分组.xlsx'))
CORTICAL_ROOT = ROOT / '皮层' / '(400，5)'
SUBCORTICAL_ROOT = ROOT / '皮层下' / '(54,5)'
OUTPUT_ROOT = ROOT / 'ml_analysis' / 'outputs' / 'no_lesion'
SEED = 20260803
PRIMARY_OUTCOMES = ('NIHSS', 'ADL')
ALL_OUTCOMES = ('NIHSS', 'ADL', 'HAMD', 'MMSE', 'VFT')
OUTCOME_COLUMNS = {'NIHSS': ('NIHSS_pre', 'NIHSS_post', -1), 'ADL': ('ADL_pre', 'ADL_post', 1), 'HAMD': ('HAMD_pre', 'HAMD_post', -1), 'MMSE': ('MMSE_pre', 'MMSE_post', 1), 'VFT': ('VFT_pre', 'VFT_post', 1)}
BLOCK_LABELS = {'baseline_score': 'Baseline score', 'clinical': 'Clinical', 'clinical_cortical_gradient': 'Clinical + cortical gradient', 'clinical_all_gradient': 'Clinical + cortical/subcortical gradient'}

@dataclass(frozen=True)
class RunSpec:
    cohort: str
    algorithm: str
    outcomes: tuple[str, ...]
    blocks: tuple[str, ...]

def subject_id_from_path(path: Path) -> str:
    match = re.match('(\\d+)', path.name)
    if not match:
        raise ValueError(f'Cannot parse subject ID from {path}')
    return match.group(1)

def find_baseline_arrays(root: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for group in ('Control', 'TMS'):
        for path in sorted((root / group / 'before').glob('*.npy')):
            subject_id = subject_id_from_path(path)
            if subject_id in paths:
                raise ValueError(f'Duplicate baseline array for subject {subject_id}')
            paths[subject_id] = path
    return paths

def gradient_summaries(array: np.ndarray, prefix: str) -> dict[str, float]:
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError(f'Expected a two-dimensional array with >=2 gradients; got {array.shape}')
    output: dict[str, float] = {}
    for gradient_index in range(2):
        values = np.asarray(array[:, gradient_index], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f'Non-finite values in {prefix} G{gradient_index + 1}')
        (q05, q95) = np.quantile(values, [0.05, 0.95])
        name = f'{prefix}_g{gradient_index + 1}'
        output[f'{name}_mean'] = float(np.mean(values))
        output[f'{name}_sd'] = float(np.std(values, ddof=1))
        output[f'{name}_robust_range'] = float(q95 - q05)
        output[f'{name}_mean_abs'] = float(np.mean(np.abs(values)))
    g1 = np.asarray(array[:, 0], dtype=float)
    g2 = np.asarray(array[:, 1], dtype=float)
    output[f'{prefix}_g1_g2_corr'] = float(np.corrcoef(g1, g2)[0, 1])
    return output

def load_dataset() -> tuple[pd.DataFrame, dict]:
    clinical = pd.read_excel(CLINICAL_PATH)
    clinical = clinical.rename(columns={'病历号': 'subject_id', '性别': 'sex', '年龄': 'age', 'Group': 'group', 'ESRS': 'esrs'})
    clinical = clinical.drop(columns=['姓名', '左右侧', '部位'], errors='ignore')
    clinical['subject_id'] = clinical['subject_id'].astype(str)
    clinical['sex_male'] = (clinical['sex'].str.upper() == 'M').astype(int)
    clinical['group_tms'] = (clinical['group'].str.upper() == 'TMS').astype(int)
    for (outcome, (pre, post, direction)) in OUTCOME_COLUMNS.items():
        if direction == -1:
            clinical[f'{outcome}_improvement'] = clinical[pre] - clinical[post]
        else:
            clinical[f'{outcome}_improvement'] = clinical[post] - clinical[pre]
    cortical_paths = find_baseline_arrays(CORTICAL_ROOT)
    subcortical_paths = find_baseline_arrays(SUBCORTICAL_ROOT)
    common_gradient_ids = sorted(set(cortical_paths) & set(subcortical_paths))
    if set(cortical_paths) != set(subcortical_paths):
        raise ValueError('Cortical and subcortical baseline subject sets do not match')
    gradient_rows = []
    for subject_id in common_gradient_ids:
        cortical = np.load(cortical_paths[subject_id])
        subcortical = np.load(subcortical_paths[subject_id])
        if cortical.shape != (400, 5):
            raise ValueError(f'Unexpected cortical shape for {subject_id}: {cortical.shape}')
        if subcortical.shape != (54, 5):
            raise ValueError(f'Unexpected subcortical shape for {subject_id}: {subcortical.shape}')
        row = {'subject_id': subject_id}
        row.update(gradient_summaries(cortical, 'cortical'))
        row.update(gradient_summaries(subcortical, 'subcortical'))
        gradient_rows.append(row)
    gradients = pd.DataFrame(gradient_rows)
    data = clinical.merge(gradients, how='left', on='subject_id', validate='one_to_one')
    data['has_baseline_gradient'] = data['subject_id'].isin(common_gradient_ids)
    clinical_ids = set(data['subject_id'])
    unmatched_gradient = sorted(set(common_gradient_ids) - clinical_ids)
    if unmatched_gradient:
        raise ValueError(f'Gradient arrays without clinical rows: {unmatched_gradient}')
    metadata = {'clinical_n': int(len(data)), 'gradient_n': int(data['has_baseline_gradient'].sum()), 'clinical_group_counts': data['group'].value_counts().sort_index().to_dict(), 'gradient_group_counts': data.loc[data['has_baseline_gradient'], 'group'].value_counts().sort_index().to_dict(), 'clinical_without_gradient': sorted(data.loc[~data['has_baseline_gradient'], 'subject_id'].tolist()), 'gradient_feature_definition': {'gradients_used': ['G1', 'G2'], 'modalities': ['cortical_400_roi', 'subcortical_54_roi'], 'summaries_per_gradient': ['mean', 'sd', '5th_to_95th_percentile_range', 'mean_absolute_value'], 'cross_gradient_summary': 'within-modality Pearson correlation between G1 and G2', 'network_labels_used': False}, 'lesion_features_used': False, 'exported_table_excludes_name_and_lesion_description_fields': True}
    return (data, metadata)

def feature_columns(data: pd.DataFrame, outcome: str, block: str) -> list[str]:
    baseline = OUTCOME_COLUMNS[outcome][0]
    clinical = [baseline, 'age', 'sex_male', 'group_tms', 'esrs']
    cortical = sorted((column for column in data if column.startswith('cortical_')))
    subcortical = sorted((column for column in data if column.startswith('subcortical_')))
    if block == 'baseline_score':
        return [baseline]
    if block == 'clinical':
        return clinical
    if block == 'clinical_cortical_gradient':
        return clinical + cortical
    if block == 'clinical_all_gradient':
        return clinical + cortical + subcortical
    raise KeyError(block)

def make_outer_splits(groups: np.ndarray, repeats: int, seed: int=SEED) -> list[dict]:
    split_records = []
    dummy = np.zeros(len(groups))
    for repeat in range(repeats):
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + repeat)
        for (fold, (train, test)) in enumerate(splitter.split(dummy, groups)):
            split_records.append({'repeat': repeat, 'fold': fold, 'train': train, 'test': test})
    return split_records

def make_inner_splits(groups: np.ndarray, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    dummy = np.zeros(len(groups))
    return list(splitter.split(dummy, groups))

def build_search(algorithm: str, inner_splits: Iterable, n_features: int, jobs: int) -> GridSearchCV:
    if algorithm == 'ridge':
        estimator = Ridge(max_iter=100000)
        grid = {'model__alpha': np.logspace(-4, 4, 17)}
    elif algorithm == 'elasticnet':
        estimator = ElasticNet(max_iter=50000, tol=0.0001, random_state=SEED)
        grid = {'model__alpha': np.logspace(-3, 1, 9), 'model__l1_ratio': [0.1, 0.5, 0.9]}
    elif algorithm == 'pls':
        estimator = PLSRegression(scale=False, max_iter=1000)
        grid = {'model__n_components': list(range(1, min(6, n_features) + 1))}
    else:
        raise KeyError(algorithm)
    pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scale', StandardScaler()), ('model', estimator)])
    return GridSearchCV(pipeline, grid, scoring='neg_mean_absolute_error', cv=list(inner_splits), n_jobs=jobs, refit=True, error_score='raise')

def model_coefficients(search: GridSearchCV, feature_names: list[str]) -> list[tuple[str, float]]:
    model = search.best_estimator_.named_steps['model']
    coefficients = np.asarray(model.coef_, dtype=float).reshape(-1)
    if coefficients.size != len(feature_names):
        raise ValueError(f'Coefficient count {coefficients.size} != feature count {len(feature_names)}')
    return list(zip(feature_names, coefficients))

def finite_correlation(function, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 3 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return float(function(y_true, y_pred).statistic)

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if np.std(y_pred) > 0:
        (slope, intercept) = np.polyfit(y_pred, y_true, 1)
    else:
        (slope, intercept) = (np.nan, np.nan)
    return {'mae': float(mean_absolute_error(y_true, y_pred)), 'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))), 'r2': float(r2_score(y_true, y_pred)), 'pearson_r': finite_correlation(pearsonr, y_true, y_pred), 'spearman_rho': finite_correlation(spearmanr, y_true, y_pred), 'calibration_intercept': float(intercept), 'calibration_slope': float(slope)}

def run_nested_cv(data: pd.DataFrame, cohort: str, outcome: str, block: str, algorithm: str, splits: list[dict], jobs: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    columns = feature_columns(data, outcome, block)
    x = data[columns].to_numpy(dtype=float)
    y = data[f'{outcome}_improvement'].to_numpy(dtype=float)
    groups = data['group_tms'].to_numpy(dtype=int)
    subject_ids = data['subject_id'].to_numpy()

    def fit_one_split(split: dict) -> tuple[list[dict], list[dict], dict]:
        repeat = split['repeat']
        fold = split['fold']
        train = split['train']
        test = split['test']
        inner = make_inner_splits(groups[train], SEED + 10000 + repeat * 10 + fold)
        search = build_search(algorithm, inner, x.shape[1], 1)
        search.fit(x[train], y[train])
        predicted = np.asarray(search.predict(x[test]), dtype=float).reshape(-1)
        local_predictions = []
        for (row_index, prediction) in zip(test, predicted):
            local_predictions.append({'cohort': cohort, 'algorithm': algorithm, 'outcome': outcome, 'block': block, 'repeat': repeat, 'fold': fold, 'subject_id': subject_ids[row_index], 'group': data.iloc[row_index]['group'], 'observed_improvement': y[row_index], 'predicted_improvement': prediction})
        local_coefficients = []
        for (feature, coefficient) in model_coefficients(search, columns):
            local_coefficients.append({'cohort': cohort, 'algorithm': algorithm, 'outcome': outcome, 'block': block, 'repeat': repeat, 'fold': fold, 'feature': feature, 'standardized_coefficient': coefficient})
        tuning = {'cohort': cohort, 'algorithm': algorithm, 'outcome': outcome, 'block': block, 'repeat': repeat, 'fold': fold, 'best_inner_mae': -float(search.best_score_), 'best_params': json.dumps(search.best_params_, sort_keys=True)}
        return (local_predictions, local_coefficients, tuning)
    split_results = Parallel(n_jobs=jobs, prefer='processes')((delayed(fit_one_split)(split) for split in splits))
    prediction_rows = [row for result in split_results for row in result[0]]
    coefficient_rows = [row for result in split_results for row in result[1]]
    tuning_rows = [result[2] for result in split_results]
    predictions = pd.DataFrame(prediction_rows)
    repeat_rows = []
    for (repeat, frame) in predictions.groupby('repeat'):
        metrics = calculate_metrics(frame['observed_improvement'], frame['predicted_improvement'])
        repeat_rows.append({'cohort': cohort, 'algorithm': algorithm, 'outcome': outcome, 'block': block, 'repeat': repeat, **metrics})
    return (predictions, pd.DataFrame(coefficient_rows), pd.DataFrame(tuning_rows).merge(pd.DataFrame(repeat_rows), on=['cohort', 'algorithm', 'outcome', 'block', 'repeat'], how='left'))

def summarize_coefficients(coefficients: pd.DataFrame) -> pd.DataFrame:
    if coefficients.empty:
        return coefficients
    rows = []
    keys = ['cohort', 'algorithm', 'outcome', 'block', 'feature']
    for (values, frame) in coefficients.groupby(keys, sort=False):
        coef = frame['standardized_coefficient'].to_numpy(dtype=float)
        positive = float(np.mean(coef > 0))
        negative = float(np.mean(coef < 0))
        rows.append({**dict(zip(keys, values)), 'n_outer_fits': len(coef), 'coefficient_mean': float(np.mean(coef)), 'coefficient_median': float(np.median(coef)), 'coefficient_sd': float(np.std(coef, ddof=1)), 'coefficient_p05': float(np.quantile(coef, 0.05)), 'coefficient_p95': float(np.quantile(coef, 0.95)), 'positive_fraction': positive, 'negative_fraction': negative, 'sign_consistency': max(positive, negative), 'selection_frequency': float(np.mean(np.abs(coef) > 1e-12))})
    return pd.DataFrame(rows)

def aggregate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    keys = ['cohort', 'algorithm', 'outcome', 'block', 'subject_id', 'group']
    return predictions.groupby(keys, as_index=False).agg(observed_improvement=('observed_improvement', 'first'), predicted_improvement=('predicted_improvement', 'mean'), prediction_sd=('predicted_improvement', 'std'), n_predictions=('predicted_improvement', 'size'))

def bootstrap_performance(aggregated: pd.DataFrame, bootstrap: int=2000) -> pd.DataFrame:
    rows = []
    keys = ['cohort', 'algorithm', 'outcome', 'block']
    rng = np.random.default_rng(SEED)
    metric_names = list(calculate_metrics(np.arange(5), np.arange(5)))
    for (values, frame) in aggregated.groupby(keys, sort=False):
        y = frame['observed_improvement'].to_numpy(dtype=float)
        pred = frame['predicted_improvement'].to_numpy(dtype=float)
        observed = calculate_metrics(y, pred)
        distributions = {metric: [] for metric in metric_names}
        for _ in range(bootstrap):
            indices = rng.integers(0, len(frame), len(frame))
            metrics = calculate_metrics(y[indices], pred[indices])
            for (metric, value) in metrics.items():
                if np.isfinite(value):
                    distributions[metric].append(value)
        for metric in metric_names:
            values_boot = distributions[metric]
            (low, high) = np.quantile(values_boot, [0.025, 0.975]) if values_boot else (np.nan, np.nan)
            rows.append({**dict(zip(keys, values)), 'n': len(frame), 'metric': metric, 'estimate': observed[metric], 'bootstrap_ci_low': float(low), 'bootstrap_ci_high': float(high), 'bootstrap_resamples': bootstrap})
    return pd.DataFrame(rows)

def paired_incremental_summary(repeat_metrics: pd.DataFrame) -> pd.DataFrame:
    frame = repeat_metrics[(repeat_metrics['cohort'] == 'gradient_common') & (repeat_metrics['algorithm'] == 'ridge')]
    pivot = frame.pivot_table(index=['outcome', 'repeat'], columns='block', values='mae')
    rows = []
    if 'clinical' not in pivot:
        return pd.DataFrame()
    for outcome in ALL_OUTCOMES:
        if outcome not in pivot.index.get_level_values('outcome'):
            continue
        outcome_frame = pivot.loc[outcome]
        for block in ('clinical_cortical_gradient', 'clinical_all_gradient'):
            if block not in outcome_frame:
                continue
            difference = outcome_frame['clinical'] - outcome_frame[block]
            rows.append({'outcome': outcome, 'comparison': f'clinical_vs_{block}', 'positive_means_gradient_improves_mae': True, 'mean_mae_reduction': float(difference.mean()), 'median_mae_reduction': float(difference.median()), 'repeat_interval_low': float(difference.quantile(0.025)), 'repeat_interval_high': float(difference.quantile(0.975)), 'fraction_repeats_improved': float(np.mean(difference > 0)), 'n_repeats': len(difference)})
    return pd.DataFrame(rows)

def summarize_repeat_performance(repeat_metrics: pd.DataFrame) -> pd.DataFrame:
    keys = ['cohort', 'algorithm', 'outcome', 'block']
    metrics = ['mae', 'rmse', 'r2', 'pearson_r', 'spearman_rho', 'calibration_intercept', 'calibration_slope']
    rows = []
    for (values, frame) in repeat_metrics.groupby(keys, sort=False):
        row = {**dict(zip(keys, values)), 'n_repeats': int(frame['repeat'].nunique())}
        for metric in metrics:
            observed = frame[metric].dropna().to_numpy(dtype=float)
            row[f'{metric}_mean'] = float(np.mean(observed))
            row[f'{metric}_median'] = float(np.median(observed))
            row[f'{metric}_p025'] = float(np.quantile(observed, 0.025))
            row[f'{metric}_p975'] = float(np.quantile(observed, 0.975))
        rows.append(row)
    return pd.DataFrame(rows)

def block_permutation_test(data: pd.DataFrame, outcome: str, permutations: int, jobs: int) -> dict:
    clinical_columns = feature_columns(data, outcome, 'clinical')
    combined_columns = feature_columns(data, outcome, 'clinical_all_gradient')
    gradient_columns = [column for column in combined_columns if column not in clinical_columns]
    x_clinical = data[clinical_columns].to_numpy(dtype=float)
    x_combined = data[combined_columns].to_numpy(dtype=float)
    y = data[f'{outcome}_improvement'].to_numpy(dtype=float)
    groups = data['group_tms'].to_numpy(dtype=int)
    splits = make_outer_splits(groups, repeats=1)

    def predict(x: np.ndarray, columns: list[str], seed_offset: int) -> np.ndarray:

        def fit_one_split(split: dict) -> tuple[np.ndarray, np.ndarray]:
            (train, test) = (split['train'], split['test'])
            inner = make_inner_splits(groups[train], SEED + seed_offset + split['fold'])
            search = build_search('ridge', inner, len(columns), 1)
            search.fit(x[train], y[train])
            return (test, np.asarray(search.predict(x[test])).reshape(-1))
        result = np.full(len(y), np.nan)
        split_predictions = Parallel(n_jobs=jobs, prefer='processes')((delayed(fit_one_split)(split) for split in splits))
        for (test, prediction) in split_predictions:
            result[test] = prediction
        return result
    clinical_prediction = predict(x_clinical, clinical_columns, 30000)
    combined_prediction = predict(x_combined, combined_columns, 40000)
    clinical_mae = mean_absolute_error(y, clinical_prediction)
    combined_mae = mean_absolute_error(y, combined_prediction)
    observed_delta = clinical_mae - combined_mae
    rng = np.random.default_rng(SEED + ALL_OUTCOMES.index(outcome))
    gradient_indices = [combined_columns.index(column) for column in gradient_columns]
    null_deltas = []
    for permutation in range(permutations):
        permuted = x_combined.copy()
        permuted[:, gradient_indices] = permuted[rng.permutation(len(data))][:, gradient_indices]
        permuted_prediction = predict(permuted, combined_columns, 50000 + permutation * 10)
        null_deltas.append(float(clinical_mae - mean_absolute_error(y, permuted_prediction)))
    null = np.asarray(null_deltas)
    p_value = float((1 + np.count_nonzero(null >= observed_delta)) / (permutations + 1))
    return {'outcome': outcome, 'test': 'conditional_gradient_block_permutation', 'clinical_mae': float(clinical_mae), 'combined_mae': float(combined_mae), 'observed_mae_reduction': float(observed_delta), 'null_mean_mae_reduction': float(np.mean(null)), 'null_p95_mae_reduction': float(np.quantile(null, 0.95)), 'one_sided_p_value': p_value, 'permutations': permutations, 'outer_cv': 'one fixed stratified 5-fold split', 'inner_cv': 'stratified 4-fold tuning within each outer training fold'}

def style_axis(ax: plt.Axes) -> None:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', color='#D9D9D9', linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)

def plot_model_comparison(repeat_metrics: pd.DataFrame, output: Path) -> None:
    frame = repeat_metrics[(repeat_metrics['cohort'] == 'gradient_common') & (repeat_metrics['algorithm'] == 'ridge') & repeat_metrics['outcome'].isin(PRIMARY_OUTCOMES)]
    blocks = tuple(BLOCK_LABELS)
    colors = ['#7F7F7F', '#2B6F8A', '#D07A3B', '#248F78']
    (fig, axes) = plt.subplots(1, 2, figsize=(10.4, 4.1), constrained_layout=True)
    rng = np.random.default_rng(SEED)
    for (ax, outcome) in zip(axes, PRIMARY_OUTCOMES):
        values = [frame[(frame['outcome'] == outcome) & (frame['block'] == block)]['mae'].to_numpy() for block in blocks]
        boxes = ax.boxplot(values, positions=np.arange(len(blocks)), widths=0.55, patch_artist=True, showfliers=False)
        for (patch, color) in zip(boxes['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.72)
            patch.set_edgecolor('#333333')
        for (index, (array, color)) in enumerate(zip(values, colors)):
            jitter = rng.uniform(-0.11, 0.11, len(array))
            ax.scatter(index + jitter, array, s=11, color=color, alpha=0.38, linewidth=0)
        ax.set_xticks(np.arange(len(blocks)), [BLOCK_LABELS[block] for block in blocks], rotation=24, ha='right')
        ax.set_ylabel(f'MAE ({outcome} improvement points)')
        ax.set_title(outcome)
        style_axis(ax)
    fig.suptitle('Repeated nested-CV performance: ridge regression', fontsize=12, fontweight='bold')
    for suffix in ('png', 'pdf'):
        fig.savefig(output.with_suffix(f'.{suffix}'), dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_observed_predicted(aggregated: pd.DataFrame, output: Path) -> None:
    frame = aggregated[(aggregated['cohort'] == 'gradient_common') & (aggregated['algorithm'] == 'ridge') & aggregated['block'].isin(['clinical', 'clinical_all_gradient']) & aggregated['outcome'].isin(PRIMARY_OUTCOMES)]
    blocks = ('clinical', 'clinical_all_gradient')
    (fig, axes) = plt.subplots(2, 2, figsize=(9.0, 7.6), constrained_layout=True)
    colors = {'Control': '#4C78A8', 'TMS': '#D55E00'}
    for (row, outcome) in enumerate(PRIMARY_OUTCOMES):
        for (column, block) in enumerate(blocks):
            ax = axes[row, column]
            part = frame[(frame['outcome'] == outcome) & (frame['block'] == block)]
            for (group, group_frame) in part.groupby('group'):
                ax.scatter(group_frame['observed_improvement'], group_frame['predicted_improvement'], s=28, color=colors.get(group, '#555555'), label=group, alpha=0.85, edgecolor='white', linewidth=0.5)
            low = min(part['observed_improvement'].min(), part['predicted_improvement'].min())
            high = max(part['observed_improvement'].max(), part['predicted_improvement'].max())
            margin = max((high - low) * 0.06, 0.5)
            ax.plot([low - margin, high + margin], [low - margin, high + margin], color='#333333', linestyle='--', linewidth=0.9)
            metrics = calculate_metrics(part['observed_improvement'], part['predicted_improvement'])
            ax.text(0.04, 0.96, f"MAE = {metrics['mae']:.2f}\n$R^2$ = {metrics['r2']:.2f}\nr = {metrics['pearson_r']:.2f}", transform=ax.transAxes, va='top', fontsize=8.5)
            ax.set_xlabel(f'Observed {outcome} improvement')
            ax.set_ylabel('Mean out-of-fold prediction')
            if row == 0:
                ax.set_title(BLOCK_LABELS[block])
            style_axis(ax)
    (handles, labels) = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(handles, labels, frameon=False, loc='lower right')
    fig.suptitle('Out-of-fold predictions: clinical vs baseline-gradient model', fontsize=12, fontweight='bold')
    for suffix in ('png', 'pdf'):
        fig.savefig(output.with_suffix(f'.{suffix}'), dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_coefficient_stability(coefficients: pd.DataFrame, output: Path) -> None:
    frame = coefficients[(coefficients['cohort'] == 'gradient_common') & (coefficients['algorithm'] == 'ridge') & (coefficients['block'] == 'clinical_all_gradient') & coefficients['outcome'].isin(PRIMARY_OUTCOMES)].copy()
    (fig, axes) = plt.subplots(1, 2, figsize=(10.4, 5.5), constrained_layout=True)
    for (ax, outcome) in zip(axes, PRIMARY_OUTCOMES):
        part = frame[frame['outcome'] == outcome]
        ranking = part.groupby('feature')['standardized_coefficient'].median().abs().sort_values(ascending=False).head(12).index
        summaries = []
        for feature in ranking:
            values = part.loc[part['feature'] == feature, 'standardized_coefficient'].to_numpy()
            summaries.append((feature, np.median(values), np.quantile(values, 0.05), np.quantile(values, 0.95)))
        summaries.sort(key=lambda item: item[1])
        y_pos = np.arange(len(summaries))
        medians = np.asarray([item[1] for item in summaries])
        lows = np.asarray([item[2] for item in summaries])
        highs = np.asarray([item[3] for item in summaries])
        ax.errorbar(medians, y_pos, xerr=np.vstack([medians - lows, highs - medians]), fmt='o', color='#2B6F8A', ecolor='#8FA9B5', capsize=2, markersize=4)
        ax.axvline(0, color='#555555', linewidth=0.8)
        ax.set_yticks(y_pos, [item[0] for item in summaries], fontsize=8)
        ax.set_xlabel('Standardized ridge coefficient (median, 5th-95th percentile)')
        ax.set_title(outcome)
        style_axis(ax)
        ax.grid(axis='x', color='#D9D9D9', linewidth=0.6, alpha=0.7)
        ax.grid(axis='y', visible=False)
    fig.suptitle('Outer-fold coefficient stability', fontsize=12, fontweight='bold')
    for suffix in ('png', 'pdf'):
        fig.savefig(output.with_suffix(f'.{suffix}'), dpi=300, bbox_inches='tight')
    plt.close(fig)

def save_split_manifest(data: pd.DataFrame, cohort: str, splits: list[dict], output: Path) -> None:
    subject_ids = data['subject_id'].to_numpy()
    records = []
    for split in splits:
        records.append({'cohort': cohort, 'repeat': int(split['repeat']), 'fold': int(split['fold']), 'train_subject_ids': subject_ids[split['train']].tolist(), 'test_subject_ids': subject_ids[split['test']].tolist()})
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run leakage-controlled clinical and baseline-gradient ML without lesion features.')
    parser.add_argument('--repeats', type=int, default=50, help='Number of repeated outer 5-fold CV runs')
    parser.add_argument('--permutations', type=int, default=200, help='Gradient-block permutations for each primary outcome')
    parser.add_argument('--bootstrap', type=int, default=2000, help='Participant bootstrap resamples for performance CIs')
    parser.add_argument('--jobs', type=int, default=-1, help='Parallel jobs used by each inner grid search')
    parser.add_argument('--quick', action='store_true', help='Validation run: 2 repeats, no permutations, 100 bootstraps')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    if args.quick:
        args.repeats = 1
        args.permutations = 0
        args.bootstrap = 100
    if args.repeats < 1 or args.permutations < 0 or args.bootstrap < 1:
        raise ValueError('repeats and bootstrap must be positive; permutations must be non-negative')
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (data, metadata) = load_dataset()
    data.to_csv(OUTPUT_ROOT / 'clinical_gradient_feature_table.csv', index=False, encoding='utf-8-sig')
    clinical_all = data.copy().reset_index(drop=True)
    gradient_common = data[data['has_baseline_gradient']].copy().reset_index(drop=True)
    cohorts = {'clinical_all': clinical_all, 'gradient_common': gradient_common}
    cohort_splits = {name: make_outer_splits(frame['group_tms'].to_numpy(dtype=int), args.repeats) for (name, frame) in cohorts.items()}
    for (name, frame) in cohorts.items():
        save_split_manifest(frame, name, cohort_splits[name], OUTPUT_ROOT / f'nested_cv_splits_{name}.json')
    if args.quick:
        specs = [RunSpec('clinical_all', 'ridge', PRIMARY_OUTCOMES, ('baseline_score', 'clinical')), RunSpec('gradient_common', 'ridge', PRIMARY_OUTCOMES, tuple(BLOCK_LABELS)), RunSpec('gradient_common', 'elasticnet', PRIMARY_OUTCOMES, ('clinical', 'clinical_all_gradient')), RunSpec('gradient_common', 'pls', PRIMARY_OUTCOMES, ('clinical', 'clinical_all_gradient'))]
    else:
        specs = [RunSpec('clinical_all', 'ridge', ALL_OUTCOMES, ('baseline_score', 'clinical')), RunSpec('gradient_common', 'ridge', ALL_OUTCOMES, tuple(BLOCK_LABELS)), RunSpec('gradient_common', 'elasticnet', PRIMARY_OUTCOMES, ('clinical', 'clinical_all_gradient')), RunSpec('gradient_common', 'pls', PRIMARY_OUTCOMES, ('clinical', 'clinical_all_gradient'))]
    all_predictions = []
    all_coefficients = []
    all_metrics = []
    total_jobs = sum((len(spec.outcomes) * len(spec.blocks) for spec in specs))
    completed = 0
    for spec in specs:
        cohort_data = cohorts[spec.cohort]
        for outcome in spec.outcomes:
            for block in spec.blocks:
                completed += 1
                print(f'[{completed}/{total_jobs}] {spec.cohort} | {spec.algorithm} | {outcome} | {block}', flush=True)
                (predictions, coefficients, metrics) = run_nested_cv(cohort_data, spec.cohort, outcome, block, spec.algorithm, cohort_splits[spec.cohort], args.jobs)
                all_predictions.append(predictions)
                all_coefficients.append(coefficients)
                all_metrics.append(metrics)
    predictions = pd.concat(all_predictions, ignore_index=True)
    coefficients = pd.concat(all_coefficients, ignore_index=True)
    tuning_and_metrics = pd.concat(all_metrics, ignore_index=True)
    repeat_metric_columns = ['cohort', 'algorithm', 'outcome', 'block', 'repeat', 'mae', 'rmse', 'r2', 'pearson_r', 'spearman_rho', 'calibration_intercept', 'calibration_slope']
    repeat_metrics = tuning_and_metrics[repeat_metric_columns].drop_duplicates()
    repeat_summary = summarize_repeat_performance(repeat_metrics)
    aggregated = aggregate_predictions(predictions)
    performance = bootstrap_performance(aggregated, args.bootstrap)
    coefficient_summary = summarize_coefficients(coefficients)
    incremental = paired_incremental_summary(repeat_metrics)
    predictions.to_csv(OUTPUT_ROOT / 'repeated_oof_predictions.csv', index=False, encoding='utf-8-sig')
    aggregated.to_csv(OUTPUT_ROOT / 'participant_mean_oof_predictions.csv', index=False, encoding='utf-8-sig')
    repeat_metrics.to_csv(OUTPUT_ROOT / 'repeat_level_performance.csv', index=False, encoding='utf-8-sig')
    repeat_summary.to_csv(OUTPUT_ROOT / 'repeat_performance_summary.csv', index=False, encoding='utf-8-sig')
    performance.to_csv(OUTPUT_ROOT / 'aggregate_performance_bootstrap_ci.csv', index=False, encoding='utf-8-sig')
    coefficients.to_csv(OUTPUT_ROOT / 'outer_fold_coefficients.csv', index=False, encoding='utf-8-sig')
    coefficient_summary.to_csv(OUTPUT_ROOT / 'coefficient_stability_summary.csv', index=False, encoding='utf-8-sig')
    tuning_and_metrics.drop(columns=[column for column in repeat_metric_columns[5:] if column in tuning_and_metrics]).to_csv(OUTPUT_ROOT / 'outer_fold_tuning.csv', index=False, encoding='utf-8-sig')
    incremental.to_csv(OUTPUT_ROOT / 'incremental_mae_summary.csv', index=False, encoding='utf-8-sig')
    permutation_rows = []
    if args.permutations:
        for outcome in PRIMARY_OUTCOMES:
            print(f'Permutation test | {outcome} | n={args.permutations}', flush=True)
            permutation_rows.append(block_permutation_test(gradient_common, outcome, args.permutations, args.jobs))
    pd.DataFrame(permutation_rows).to_csv(OUTPUT_ROOT / 'primary_gradient_block_permutation.csv', index=False, encoding='utf-8-sig')
    plot_model_comparison(repeat_metrics, OUTPUT_ROOT / 'figure_primary_model_comparison')
    plot_observed_predicted(aggregated, OUTPUT_ROOT / 'figure_primary_observed_predicted')
    plot_coefficient_stability(coefficients, OUTPUT_ROOT / 'figure_primary_coefficient_stability')
    metadata.update({'analysis_date': '2026-08-03', 'analysis_scope': 'clinical and baseline cortical/subcortical gradients only', 'outcome_direction': 'positive values always indicate improvement', 'outcomes': {'NIHSS': 'pre - post', 'ADL': 'post - pre', 'HAMD': 'pre - post', 'MMSE': 'post - pre', 'VFT': 'post - pre'}, 'primary_outcomes': list(PRIMARY_OUTCOMES), 'outer_cv': f'stratified 5-fold repeated {args.repeats} times', 'inner_cv': 'stratified 4-fold grid search', 'selection_metric': 'mean absolute error', 'algorithms': ['ridge', 'elasticnet', 'partial_least_squares'], 'permutations': args.permutations, 'bootstrap_resamples': args.bootstrap, 'random_seed': SEED, 'important_limitations': ['Internal validation only; no independent external cohort.', 'Network-specific gradient summaries were not used because the authoritative local 7-network label file is unavailable.', 'Treatment group is a predictor of observed outcome, not an estimate of individual causal treatment benefit.', 'No lesion-derived variable is included in this analysis.']})
    (OUTPUT_ROOT / 'analysis_manifest.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Completed. Outputs: {OUTPUT_ROOT}', flush=True)
if __name__ == '__main__':
    main()
