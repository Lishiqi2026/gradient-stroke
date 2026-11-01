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
ROOT = Path(_portable_path('project', ''))
OUTPUT_ROOT = ROOT / 'ml_analysis' / 'outputs' / 'clinical_indicator'
os.environ.setdefault('JOBLIB_TEMP_FOLDER', str(OUTPUT_ROOT / '.joblib_temp'))
Path(os.environ['JOBLIB_TEMP_FOLDER']).mkdir(parents=True, exist_ok=True)
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import pearsonr, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from run_clinical_gradient_ml import CORTICAL_ROOT, SEED, find_baseline_arrays, load_dataset
LABEL_PATH = ROOT / 'fc_analysis' / 'tables' / 'schaefer400_roi_network_labels.csv'
OUTCOMES = ('NIHSS', 'ADL')
NETWORK_ORDER = ('Visual', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Control', 'Default')
METRICS = ('dispersion', 'segregation', 'between_disp')
BLOCK_LABELS = {'clinical_dispersion': 'Clinical + Dispersion', 'clinical_segregation': 'Clinical + Segregation', 'clinical_between_disp': 'Clinical + betweenDisp', 'clinical_g1_all': 'Clinical + G1 indicators', 'clinical_g2_all': 'Clinical + G2 indicators', 'clinical_all_indicators': 'Clinical + G1/G2 indicators'}

@dataclass
class PreparedData:
    table: pd.DataFrame
    indicator_columns: tuple[str, ...]

def compute_metrics_for_gradient(values: np.ndarray, network_indices: dict[str, np.ndarray]) -> dict[str, float]:
    """Port of indicator.py: per-network dispersion, betweenDisp and segregation."""
    centroids: dict[str, float] = {}
    dispersion: dict[str, float] = {}
    for network in NETWORK_ORDER:
        network_values = values[network_indices[network]]
        centroids[network] = float(np.nanmean(network_values))
        dispersion[network] = float(np.nanstd(network_values, ddof=1))
    output: dict[str, float] = {}
    for network in NETWORK_ORDER:
        other_distances = [abs(centroids[network] - centroids[other]) for other in NETWORK_ORDER if other != network]
        between_disp = float(np.nanmean(other_distances))
        segregation = float(between_disp / (dispersion[network] + 1e-12))
        output[f'{network}_dispersion'] = dispersion[network]
        output[f'{network}_between_disp'] = between_disp
        output[f'{network}_segregation'] = segregation
    return output

def load_prepared_data() -> tuple[PreparedData, dict]:
    (data, source_metadata) = load_dataset()
    data = data[data['has_baseline_gradient']].copy().sort_values('subject_id').reset_index(drop=True)
    labels = pd.read_csv(LABEL_PATH)
    required = {'roi', 'network'}
    if not required.issubset(labels.columns):
        raise ValueError(f'Missing label columns: {required - set(labels.columns)}')
    labels = labels.sort_values('roi').reset_index(drop=True)
    if labels['roi'].tolist() != list(range(1, 401)):
        raise ValueError('Schaefer label rows are not a complete ordered ROI 1..400 mapping')
    if set(labels['network']) != set(NETWORK_ORDER):
        raise ValueError(f"Unexpected network labels: {sorted(set(labels['network']))}")
    network_indices = {network: np.flatnonzero(labels['network'].to_numpy() == network) for network in NETWORK_ORDER}
    cortical_paths = find_baseline_arrays(CORTICAL_ROOT)
    rows = []
    for subject_id in data['subject_id']:
        array = np.asarray(np.load(cortical_paths[subject_id]), dtype=float)
        if array.shape != (400, 5):
            raise ValueError(f'Unexpected cortical gradient shape for {subject_id}: {array.shape}')
        if not np.all(np.isfinite(array[:, :2])):
            raise ValueError(f'Non-finite G1/G2 values for {subject_id}')
        row = {'subject_id': subject_id}
        for (gradient_index, gradient_name) in enumerate(('g1', 'g2')):
            metrics = compute_metrics_for_gradient(array[:, gradient_index], network_indices)
            row.update({f'{gradient_name}_{name}': value for (name, value) in metrics.items()})
        rows.append(row)
    indicator_table = pd.DataFrame(rows)
    indicator_columns = tuple((column for column in indicator_table if column != 'subject_id'))
    data = data.merge(indicator_table, on='subject_id', validate='one_to_one')
    if data[list(indicator_columns)].isna().any().any():
        raise ValueError('Indicator table contains missing values')
    metadata = {'analysis_date': '2026-08-05', 'n': int(len(data)), 'group_counts': data['group'].value_counts().sort_index().to_dict(), 'outcomes': {outcome: f'{outcome}_post' for outcome in OUTCOMES}, 'clinical_base': ['outcome-specific baseline score', 'age', 'sex', 'ESRS', 'treatment group'], 'candidate_blocks': BLOCK_LABELS, 'indicator_source_formula': _portable_path('project', 'code/indicator.py'), 'atlas_labels': str(LABEL_PATH), 'indicator_definition': {'gradients': ['G1', 'G2'], 'networks': list(NETWORK_ORDER), 'dispersion': 'sample SD of ROI gradient values within network (ddof=1)', 'between_disp': 'mean absolute centroid difference to the other six networks', 'segregation': 'between_disp / (dispersion + 1e-12)', 'features_total': len(indicator_columns)}, 'source_metadata': source_metadata, 'lesion_features_used': False, 'post_treatment_imaging_used': False}
    return (PreparedData(data, indicator_columns), metadata)

def clinical_columns(outcome: str) -> list[str]:
    return [f'{outcome}_pre', 'age', 'sex_male', 'group_tms', 'esrs']

def block_columns(prepared: PreparedData, outcome: str, block: str) -> list[str]:
    clinical = clinical_columns(outcome)
    indicators = list(prepared.indicator_columns)
    if block == 'clinical_reference':
        return clinical
    if block == 'clinical_dispersion':
        selected = [column for column in indicators if column.endswith('_dispersion')]
    elif block == 'clinical_segregation':
        selected = [column for column in indicators if column.endswith('_segregation')]
    elif block == 'clinical_between_disp':
        selected = [column for column in indicators if column.endswith('_between_disp')]
    elif block == 'clinical_g1_all':
        selected = [column for column in indicators if column.startswith('g1_')]
    elif block == 'clinical_g2_all':
        selected = [column for column in indicators if column.startswith('g2_')]
    elif block == 'clinical_all_indicators':
        selected = indicators
    else:
        raise KeyError(block)
    return clinical + selected

def make_splits(groups: np.ndarray, repeats: int) -> list[dict]:
    records = []
    dummy = np.zeros(len(groups))
    for repeat in range(repeats):
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + repeat)
        for (fold, (train, test)) in enumerate(splitter.split(dummy, groups)):
            records.append({'repeat': repeat, 'fold': fold, 'train': train, 'test': test})
    return records

def model_candidates(algorithm: str) -> list[dict]:
    if algorithm == 'ridge':
        return [{'alpha': float(alpha)} for alpha in np.logspace(3, -3, 7)]
    if algorithm == 'elasticnet':
        return [{'alpha': float(alpha), 'l1_ratio': float(l1_ratio)} for alpha in np.logspace(1, -2, 4) for l1_ratio in (0.1, 0.5, 0.9)]
    raise KeyError(algorithm)

def make_model(algorithm: str, parameters: dict):
    if algorithm == 'ridge':
        return Ridge(alpha=parameters['alpha'], max_iter=100000)
    if algorithm == 'elasticnet':
        return ElasticNet(alpha=parameters['alpha'], l1_ratio=parameters['l1_ratio'], max_iter=50000, tol=0.0001, random_state=SEED)
    raise KeyError(algorithm)

def make_pipeline(model):
    return Pipeline([('imputer', SimpleImputer(strategy='median')), ('scale', StandardScaler()), ('model', model)])

def tune_predict(prepared: PreparedData, outcome: str, block: str, algorithm: str, split: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    table = prepared.table
    x = table[block_columns(prepared, outcome, block)].to_numpy(dtype=float)
    y = table[f'{outcome}_post'].to_numpy(dtype=float)
    groups = table['group_tms'].to_numpy(dtype=int)
    (outer_train, outer_test) = (split['train'], split['test'])
    inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED + 10000 + split['repeat'] * 10 + split['fold'])
    inner_splits = list(inner.split(np.zeros(len(outer_train)), groups[outer_train]))
    best_score = np.inf
    best_parameters = None
    for parameters in model_candidates(algorithm):
        errors = []
        for (train_local, valid_local) in inner_splits:
            train = outer_train[train_local]
            valid = outer_train[valid_local]
            pipeline = make_pipeline(make_model(algorithm, parameters))
            pipeline.fit(x[train], y[train])
            errors.append(mean_absolute_error(y[valid], pipeline.predict(x[valid])))
        score = float(np.mean(errors))
        if score < best_score - 1e-12:
            best_score = score
            best_parameters = parameters
    if best_parameters is None:
        raise RuntimeError('No tuning candidate selected')
    final_pipeline = make_pipeline(make_model(algorithm, best_parameters))
    final_pipeline.fit(x[outer_train], y[outer_train])
    prediction = np.asarray(final_pipeline.predict(x[outer_test]), dtype=float)
    tuning = {'repeat': split['repeat'], 'fold': split['fold'], 'inner_mae': best_score, 'parameters': json.dumps(best_parameters, sort_keys=True), 'n_features': x.shape[1]}
    return (outer_test, prediction, tuning)

def safe_correlation(function, observed: np.ndarray, predicted: np.ndarray) -> float:
    if np.std(observed) == 0 or np.std(predicted) == 0:
        return float('nan')
    return float(function(observed, predicted)[0])

def metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {'mae': float(mean_absolute_error(observed, predicted)), 'rmse': float(np.sqrt(mean_squared_error(observed, predicted))), 'r2': float(r2_score(observed, predicted)), 'pearson_r': safe_correlation(pearsonr, observed, predicted), 'spearman_rho': safe_correlation(spearmanr, observed, predicted)}

def improvement(outcome: str, baseline: np.ndarray, post: np.ndarray) -> np.ndarray:
    return baseline - post if outcome == 'NIHSS' else post - baseline

def run_block(prepared: PreparedData, outcome: str, block: str, algorithm: str, splits: list[dict], jobs: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print(f'{algorithm} | {outcome} | {block}', flush=True)
    fitted = Parallel(n_jobs=jobs, prefer='processes', max_nbytes='100K')((delayed(tune_predict)(prepared, outcome, block, algorithm, split) for split in splits))
    table = prepared.table
    y = table[f'{outcome}_post'].to_numpy(dtype=float)
    baseline = table[f'{outcome}_pre'].to_numpy(dtype=float)
    ids = table['subject_id'].to_numpy()
    prediction_rows = []
    tuning_rows = []
    for (split, (test, prediction, tuning)) in zip(splits, fitted):
        for (local, row_index) in enumerate(test):
            prediction_rows.append({'algorithm': algorithm, 'outcome': outcome, 'block': block, 'repeat': split['repeat'], 'fold': split['fold'], 'subject_id': ids[row_index], 'group': table.iloc[row_index]['group'], 'observed_post': y[row_index], 'predicted_post': prediction[local], 'observed_improvement': improvement(outcome, baseline[row_index], y[row_index]), 'predicted_improvement': improvement(outcome, baseline[row_index], prediction[local])})
        tuning_rows.append({'algorithm': algorithm, 'outcome': outcome, 'block': block, **tuning})
    predictions = pd.DataFrame(prediction_rows)
    repeat_rows = []
    for (repeat, frame) in predictions.groupby('repeat'):
        post_metrics = metrics(frame['observed_post'].to_numpy(), frame['predicted_post'].to_numpy())
        improvement_metrics = metrics(frame['observed_improvement'].to_numpy(), frame['predicted_improvement'].to_numpy())
        repeat_rows.append({'algorithm': algorithm, 'outcome': outcome, 'block': block, 'repeat': repeat, **{f'post_{key}': value for (key, value) in post_metrics.items()}, **{f'improvement_{key}': value for (key, value) in improvement_metrics.items()}})
    return (predictions, pd.DataFrame(tuning_rows), pd.DataFrame(repeat_rows))

def summarize(repeat_metrics: pd.DataFrame) -> pd.DataFrame:
    keys = ['algorithm', 'outcome', 'block']
    metric_columns = [column for column in repeat_metrics if column.startswith(('post_', 'improvement_'))]
    rows = []
    for (values, frame) in repeat_metrics.groupby(keys, sort=False):
        row = {**dict(zip(keys, values)), 'n_repeats': int(frame['repeat'].nunique())}
        for column in metric_columns:
            values_array = frame[column].dropna().to_numpy(dtype=float)
            row[f'{column}_mean'] = float(np.mean(values_array))
            row[f'{column}_median'] = float(np.median(values_array))
            row[f'{column}_p025'] = float(np.quantile(values_array, 0.025))
            row[f'{column}_p975'] = float(np.quantile(values_array, 0.975))
        rows.append(row)
    return pd.DataFrame(rows)

def incremental_summary(repeat_metrics: pd.DataFrame) -> pd.DataFrame:
    pivot = repeat_metrics.pivot_table(index=['algorithm', 'outcome', 'repeat'], columns='block', values='post_mae')
    rows = []
    for algorithm in sorted(repeat_metrics['algorithm'].unique()):
        for outcome in OUTCOMES:
            frame = pivot.loc[algorithm, outcome]
            for block in BLOCK_LABELS:
                difference = frame['clinical_reference'] - frame[block]
                rows.append({'algorithm': algorithm, 'outcome': outcome, 'block': block, 'positive_means_indicator_improves_mae': True, 'mean_mae_reduction': float(difference.mean()), 'median_mae_reduction': float(difference.median()), 'repeat_interval_low': float(difference.quantile(0.025)), 'repeat_interval_high': float(difference.quantile(0.975)), 'fraction_repeats_improved': float(np.mean(difference > 0)), 'n_repeats': int(len(difference))})
    return pd.DataFrame(rows)

def aggregate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    keys = ['algorithm', 'outcome', 'block', 'subject_id', 'group']
    return predictions.groupby(keys, as_index=False).agg(observed_post=('observed_post', 'first'), predicted_post=('predicted_post', 'mean'), predicted_post_sd=('predicted_post', 'std'), observed_improvement=('observed_improvement', 'first'), predicted_improvement=('predicted_improvement', 'mean'), predicted_improvement_sd=('predicted_improvement', 'std'), n_predictions=('predicted_post', 'size'))

def participant_bootstrap_summary(aggregated: pd.DataFrame, draws: int=10000, seed: int=20260805) -> pd.DataFrame:
    """Paired subject bootstrap of ensemble-OOF MAE differences versus clinical reference."""
    rng = np.random.default_rng(seed)
    rows = []
    for algorithm in sorted(aggregated['algorithm'].unique()):
        for outcome in OUTCOMES:
            subset = aggregated[(aggregated['algorithm'] == algorithm) & (aggregated['outcome'] == outcome)]
            reference = subset[subset['block'] == 'clinical_reference'].sort_values('subject_id')
            for block in BLOCK_LABELS:
                comparison = subset[subset['block'] == block].sort_values('subject_id')
                if reference['subject_id'].tolist() != comparison['subject_id'].tolist():
                    raise ValueError(f'Subject mismatch for {algorithm}, {outcome}, {block}')
                observed = reference['observed_post'].to_numpy(dtype=float)
                reference_prediction = reference['predicted_post'].to_numpy(dtype=float)
                comparison_prediction = comparison['predicted_post'].to_numpy(dtype=float)
                subject_differences = np.abs(observed - reference_prediction) - np.abs(observed - comparison_prediction)
                sample_indices = rng.integers(0, len(observed), size=(draws, len(observed)), endpoint=False)
                bootstrap_differences = subject_differences[sample_indices].mean(axis=1)
                rows.append({'algorithm': algorithm, 'outcome': outcome, 'block': block, 'positive_means_indicator_improves_mae': True, 'ensemble_reference_mae': float(np.mean(np.abs(observed - reference_prediction))), 'ensemble_indicator_mae': float(np.mean(np.abs(observed - comparison_prediction))), 'ensemble_mae_reduction': float(subject_differences.mean()), 'bootstrap_ci_low': float(np.quantile(bootstrap_differences, 0.025)), 'bootstrap_ci_high': float(np.quantile(bootstrap_differences, 0.975)), 'bootstrap_probability_improved': float(np.mean(bootstrap_differences > 0)), 'subjects_with_lower_absolute_error': int(np.sum(subject_differences > 0)), 'n_subjects': int(len(observed)), 'bootstrap_draws': draws})
    return pd.DataFrame(rows)

def save_splits(prepared: PreparedData, splits: list[dict]) -> None:
    ids = prepared.table['subject_id'].to_numpy()
    records = [{'repeat': split['repeat'], 'fold': split['fold'], 'train_subject_ids': ids[split['train']].tolist(), 'test_subject_ids': ids[split['test']].tolist()} for split in splits]
    (OUTPUT_ROOT / 'nested_cv_splits.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Predict post-treatment scores with clinical covariates included in every indicator block.')
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
    (prepared, manifest) = load_prepared_data()
    feature_columns = ['subject_id', 'group', 'group_tms', 'age', 'sex_male', 'esrs', 'NIHSS_pre', 'ADL_pre', 'NIHSS_post', 'ADL_post', *prepared.indicator_columns]
    prepared.table[feature_columns].to_csv(OUTPUT_ROOT / 'clinical_indicator_feature_table.csv', index=False, encoding='utf-8-sig')
    splits = make_splits(prepared.table['group_tms'].to_numpy(dtype=int), args.repeats)
    save_splits(prepared, splits)
    jobs_to_run = []
    for algorithm in ('ridge', 'elasticnet'):
        for outcome in OUTCOMES:
            jobs_to_run.append((algorithm, outcome, 'clinical_reference'))
            jobs_to_run.extend(((algorithm, outcome, block) for block in BLOCK_LABELS))
    predictions_all = []
    tuning_all = []
    repeat_all = []
    for (index, (algorithm, outcome, block)) in enumerate(jobs_to_run, start=1):
        print(f'[{index}/{len(jobs_to_run)}]', end=' ', flush=True)
        (predictions, tuning, repeat_metrics) = run_block(prepared, outcome, block, algorithm, splits, args.jobs)
        predictions_all.append(predictions)
        tuning_all.append(tuning)
        repeat_all.append(repeat_metrics)
    predictions = pd.concat(predictions_all, ignore_index=True)
    tuning = pd.concat(tuning_all, ignore_index=True)
    repeat_metrics = pd.concat(repeat_all, ignore_index=True)
    performance = summarize(repeat_metrics)
    incremental = incremental_summary(repeat_metrics)
    aggregated = aggregate_predictions(predictions)
    bootstrap = participant_bootstrap_summary(aggregated)
    predictions.to_csv(OUTPUT_ROOT / 'repeated_oof_predictions.csv', index=False, encoding='utf-8-sig')
    aggregated.to_csv(OUTPUT_ROOT / 'participant_mean_oof_predictions.csv', index=False, encoding='utf-8-sig')
    tuning.to_csv(OUTPUT_ROOT / 'outer_fold_tuning.csv', index=False, encoding='utf-8-sig')
    repeat_metrics.to_csv(OUTPUT_ROOT / 'repeat_level_performance.csv', index=False, encoding='utf-8-sig')
    performance.to_csv(OUTPUT_ROOT / 'repeat_performance_summary.csv', index=False, encoding='utf-8-sig')
    incremental.to_csv(OUTPUT_ROOT / 'incremental_mae_summary.csv', index=False, encoding='utf-8-sig')
    bootstrap.to_csv(OUTPUT_ROOT / 'participant_bootstrap_mae.csv', index=False, encoding='utf-8-sig')
    manifest.update({'outer_cv': f'treatment-stratified 5-fold repeated {args.repeats} times', 'inner_cv': 'treatment-stratified 4-fold tuning inside each outer training fold', 'algorithms': ['ridge', 'elasticnet'], 'random_seed': SEED, 'clinical_reference_policy': 'computed only as an internal paired reference; excluded from candidate indicator blocks', 'participant_bootstrap': "10,000 paired subject resamples of MAE differences using each subject's mean repeated-OOF prediction", 'limitations': ['Internal validation only; no independent external cohort.', 'All six candidate blocks include the same compact clinical base.', 'Treatment group conditions prognosis and does not estimate causal individual treatment benefit.', 'Indicator block comparisons are exploratory and involve multiple candidate blocks.']})
    (OUTPUT_ROOT / 'analysis_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Completed. Outputs: {OUTPUT_ROOT}', flush=True)
if __name__ == '__main__':
    main()
