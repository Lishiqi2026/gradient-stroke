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
OUTPUT_ROOT = ROOT / 'ml_analysis' / 'outputs' / 'five_outcome_lesion_informed'
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
from run_clinical_gradient_ml import CORTICAL_ROOT, SEED, find_baseline_arrays
from run_clinical_indicator_ml import LABEL_PATH, NETWORK_ORDER, compute_metrics_for_gradient
CLINICAL_PATH = Path(_portable_path('work', '董分组.xlsx'))
FOLLOWUP_PATH = Path(_portable_path('work', '随访数据.xlsx'))
LESION_FINAL = ROOT / 'lesion_analysis' / 'lesion_gradient_distance' / 'figures' / 'nature' / 'lesion_final' / 'bcdf_recomputed_provided_reference'
SUBJECT_METRICS_PATH = LESION_FINAL / 'recomputed_subject_metrics.csv'
ROI_METRICS_PATH = LESION_FINAL / 'recomputed_roi_metrics.csv'
OUTCOMES = ('NIHSS', 'ADL', 'MMSE', 'HAMD', 'VFT')
LOWER_IS_BETTER = {'NIHSS': True, 'ADL': False, 'MMSE': False, 'HAMD': True, 'VFT': False}
BLOCK_LABELS = {'clinical_g2_vis_smn': 'Clinical + targeted G2 VIS/SMN', 'clinical_g2_network_all': 'Clinical + all-network G2', 'clinical_between_disp': 'Clinical + G1/G2 betweenDisp', 'clinical_healthy_deviation': 'Clinical + baseline healthy deviation', 'clinical_lesion_topology': 'Clinical + lesion topology', 'clinical_g2_lesion_curated': 'Clinical + curated G2/lesion', 'clinical_all_baseline': 'Clinical + all baseline imaging'}
FORBIDDEN_FEATURE_TOKENS = ('post_', 'normalization', 'reference_directed', 'reorganization', '_slope', 'model_r_squared', 'delta_mean_fd')

@dataclass
class PreparedData:
    table: pd.DataFrame
    block_feature_columns: dict[str, tuple[str, ...]]

def load_clinical() -> pd.DataFrame:
    table = pd.read_excel(CLINICAL_PATH)
    table = table.rename(columns={'病历号': 'subject_id', '性别': 'sex', '年龄': 'age', 'Group': 'group', 'ESRS': 'esrs'})
    table['subject_id'] = table['subject_id'].astype(str)
    table['sex_male'] = (table['sex'].astype(str).str.upper() == 'M').astype(int)
    table['group_tms'] = (table['group'].astype(str).str.upper() == 'TMS').astype(int)
    required = ['subject_id', 'group', 'age', 'sex_male', 'group_tms', 'esrs', *[f'{outcome}_{time}' for outcome in OUTCOMES for time in ('pre', 'post')]]
    if table[required].isna().any().any():
        raise ValueError('Clinical table has missing required values')
    return table[required].copy()

def network_indices() -> dict[str, np.ndarray]:
    labels = pd.read_csv(LABEL_PATH).sort_values('roi').reset_index(drop=True)
    if labels['roi'].tolist() != list(range(1, 401)):
        raise ValueError('Schaefer labels are not ordered ROI 1..400')
    if set(labels['network']) != set(NETWORK_ORDER):
        raise ValueError('Schaefer network labels do not match the expected seven networks')
    return {network: np.flatnonzero(labels['network'].to_numpy() == network) for network in NETWORK_ORDER}

def build_gradient_features(subject_ids: list[str]) -> pd.DataFrame:
    indices = network_indices()
    paths = find_baseline_arrays(CORTICAL_ROOT)
    rows = []
    for subject_id in subject_ids:
        array = np.asarray(np.load(paths[subject_id]), dtype=float)
        if array.shape != (400, 5) or not np.isfinite(array[:, :2]).all():
            raise ValueError(f'Invalid cortical gradient array for {subject_id}: {array.shape}')
        row: dict[str, float | str] = {'subject_id': subject_id}
        for (gradient_index, gradient_name) in enumerate(('g1', 'g2')):
            values = array[:, gradient_index]
            metrics = compute_metrics_for_gradient(values, indices)
            row.update({f'{gradient_name}_{name}': value for (name, value) in metrics.items()})
            for network in NETWORK_ORDER:
                row[f'{gradient_name}_{network}_centroid'] = float(np.mean(values[indices[network]]))
        rows.append(row)
    return pd.DataFrame(rows)

def build_network_deviation_features(roi_metrics: pd.DataFrame) -> pd.DataFrame:
    selected = roi_metrics[['subject_id', 'network', 'baseline_deviation_G2', 'baseline_deviation_G1G2']].copy()
    selected['subject_id'] = selected['subject_id'].astype(str)
    grouped = selected.groupby(['subject_id', 'network'], as_index=False).agg(baseline_deviation_G2=('baseline_deviation_G2', 'mean'), baseline_deviation_G1G2=('baseline_deviation_G1G2', 'mean'))
    rows = []
    for (subject_id, frame) in grouped.groupby('subject_id'):
        by_network = frame.set_index('network')
        row: dict[str, float | str] = {'subject_id': subject_id}
        for network in NETWORK_ORDER:
            row[f'g2_{network}_baseline_deviation'] = float(by_network.loc[network, 'baseline_deviation_G2'])
            row[f'g1g2_{network}_baseline_deviation'] = float(by_network.loc[network, 'baseline_deviation_G1G2'])
        rows.append(row)
    return pd.DataFrame(rows)

def define_feature_blocks(table: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    targeted_networks = ('Visual', 'SomMot')
    targeted_g2 = []
    for network in targeted_networks:
        targeted_g2.extend([f'g2_{network}_centroid', f'g2_{network}_dispersion', f'g2_{network}_between_disp', f'g2_{network}_segregation', f'g2_{network}_baseline_deviation', f'g1g2_{network}_baseline_deviation'])
    all_network_g2 = [column for column in table if column.startswith('g2_') and any((column.endswith(suffix) for suffix in ('_centroid', '_dispersion', '_between_disp', '_segregation', '_baseline_deviation')))]
    between_disp = [column for column in table if column.startswith(('g1_', 'g2_')) and column.endswith('_between_disp')]
    healthy_deviation = ['mean_baseline_deviation_G1', 'mean_baseline_deviation_G2', 'mean_baseline_deviation_G1G2', 'mean_baseline_deviation_5D', 'nearest_anatomical_decile_baseline_deviation_G1G2', 'nearest_xtract_G1G2_decile_baseline_deviation', *[column for column in table if column.startswith(('g2_', 'g1g2_')) and column.endswith('_baseline_deviation')]]
    lesion_topology = ['log1p_lesion_volume', 'mask_lateralization_index', 'cortical_overlap_voxels', 'cortical_overlap_roi_n', 'xtract_projectable_tract_damage_fraction', 'xtract_projected_effective_roi_n', 'xtract_projected_max_roi_weight', 'xtract_projected_lesion_G1', 'xtract_projected_lesion_G2', 'xtract_projected_G1_span', 'xtract_projected_G2_span']
    curated_lesion = ['log1p_lesion_volume', 'mask_lateralization_index', 'xtract_projectable_tract_damage_fraction', 'xtract_projected_effective_roi_n', 'xtract_projected_lesion_G2', 'xtract_projected_G2_span', 'mean_baseline_deviation_G2', 'mean_baseline_deviation_G1G2', 'nearest_xtract_G1G2_decile_baseline_deviation']
    extracted_gradient = [column for column in table if column.startswith(('g1_', 'g2_', 'g1g2_'))]
    global_deviation = ['mean_baseline_deviation_G1', 'mean_baseline_deviation_G2', 'mean_baseline_deviation_G1G2', 'mean_baseline_deviation_5D', 'nearest_anatomical_decile_baseline_deviation_G1G2', 'nearest_xtract_G1G2_decile_baseline_deviation']
    all_baseline = list(dict.fromkeys([*extracted_gradient, *global_deviation, *lesion_topology]))
    blocks = {'clinical_g2_vis_smn': tuple(targeted_g2), 'clinical_g2_network_all': tuple(sorted(all_network_g2)), 'clinical_between_disp': tuple(sorted(between_disp)), 'clinical_healthy_deviation': tuple(dict.fromkeys(healthy_deviation)), 'clinical_lesion_topology': tuple(lesion_topology), 'clinical_g2_lesion_curated': tuple(dict.fromkeys([*targeted_g2, *curated_lesion])), 'clinical_all_baseline': tuple(sorted(all_baseline))}
    for (block, columns) in blocks.items():
        if not columns:
            raise ValueError(f'No imaging features selected for {block}')
        missing = sorted(set(columns) - set(table.columns))
        if missing:
            raise ValueError(f'Missing features for {block}: {missing}')
        forbidden = [column for column in columns if any((token in column.lower() for token in FORBIDDEN_FEATURE_TOKENS))]
        if forbidden:
            raise ValueError(f'Post-treatment or derived-outcome features in {block}: {forbidden}')
    return blocks

def followup_metadata(eligible_ids: set[str]) -> dict:
    followup = pd.read_excel(FOLLOWUP_PATH)
    followup['subject_id'] = followup.iloc[:, 0].astype(str)
    numeric_columns = list(followup.columns)[3:9]
    return {'rows': int(len(followup)), 'eligible_lesion_qc_overlap': int(followup['subject_id'].isin(eligible_ids).sum()), 'content': 'MMSE and VFT baseline, immediate-post values duplicating the main clinical table, plus one-month MMSE and VFT; no one-month NIHSS, ADL or HAMD', 'numeric_column_positions': [str(column) for column in numeric_columns], 'used_in_primary_five_outcome_models': False}

def load_prepared_data() -> tuple[PreparedData, dict]:
    clinical = load_clinical()
    subject_metrics = pd.read_csv(SUBJECT_METRICS_PATH)
    subject_metrics['subject_id'] = subject_metrics['subject_id'].astype(str)
    subject_metrics = subject_metrics[subject_metrics['primary_motion_eligible'].astype(bool)].copy()
    if 'xtract_projection_eligible' in subject_metrics:
        subject_metrics = subject_metrics.loc[subject_metrics['xtract_projection_eligible'].astype(bool)].copy()
    allowed_subject_columns = ['subject_id', 'group', 'group_tms', 'primary_motion_eligible', 'baseline_mean_fd', 'baseline_retained_frames', 'lesion_volume_ml_recomputed', 'log1p_lesion_volume', 'mask_lateralization_index', 'cortical_overlap_voxels', 'cortical_overlap_roi_n', 'mean_baseline_deviation_G1', 'mean_baseline_deviation_G2', 'mean_baseline_deviation_G1G2', 'mean_baseline_deviation_5D', 'nearest_anatomical_decile_baseline_deviation_G1G2', 'xtract_projectable_tract_damage_fraction', 'xtract_projected_effective_roi_n', 'xtract_projected_max_roi_weight', 'xtract_projected_lesion_G1', 'xtract_projected_lesion_G2', 'xtract_projected_G1_span', 'xtract_projected_G2_span', 'nearest_xtract_G1G2_decile_baseline_deviation']
    subject_metrics = subject_metrics[allowed_subject_columns]
    ids = sorted(set(clinical['subject_id']) & set(subject_metrics['subject_id']))
    gradient_features = build_gradient_features(ids)
    roi_metrics = pd.read_csv(ROI_METRICS_PATH)
    network_deviation = build_network_deviation_features(roi_metrics)
    table = clinical[clinical['subject_id'].isin(ids)].copy()
    table = table.merge(subject_metrics.drop(columns=['group', 'group_tms']), on='subject_id', validate='one_to_one')
    table = table.merge(gradient_features, on='subject_id', validate='one_to_one')
    table = table.merge(network_deviation, on='subject_id', validate='one_to_one')
    table = table.sort_values('subject_id').reset_index(drop=True)
    blocks = define_feature_blocks(table)
    all_features = sorted({column for columns in blocks.values() for column in columns})
    if table[all_features].isna().any().any() or not np.isfinite(table[all_features].to_numpy(dtype=float)).all():
        raise ValueError('Baseline imaging feature table contains missing or non-finite values')
    metadata = {'analysis_date': '2026-08-11', 'primary_n': int(len(table)), 'group_counts': table['group'].value_counts().sort_index().to_dict(), 'motion_qc': 'primary_motion_eligible from lesion_final; 37/38 lesion subjects retained', 'outcomes': {outcome: f'{outcome}_post' for outcome in OUTCOMES}, 'clinical_base_for_every_block': ['outcome-specific baseline score', 'age', 'sex', 'ESRS', 'treatment group'], 'clinical_only_block_included': False, 'candidate_blocks': BLOCK_LABELS, 'imaging_feature_count_by_block': {block: len(columns) for (block, columns) in blocks.items()}, 'feature_sources': {'raw_baseline_cortical_gradients': str(CORTICAL_ROOT), 'schaefer_network_labels': str(LABEL_PATH), 'baseline_lesion_and_healthy_deviation': str(SUBJECT_METRICS_PATH), 'network_baseline_deviation': str(ROI_METRICS_PATH)}, 'excluded_as_leakage': ['all post-treatment gradients and post-treatment motion', 'normalization and reorganization metrics', 'distance-normalization slopes and their model statistics', 'the 67-ROI FDR mask derived from whole-cohort post-treatment normalization'], 'hypothesis_basis': 'G2 and VIS/SomMot emphasis follows the supplied lesion_final group-level mechanism results; this is same-cohort hypothesis generation, not independent validation', 'followup_table': followup_metadata(set(ids))}
    return (PreparedData(table=table, block_feature_columns=blocks), metadata)

def clinical_columns(outcome: str) -> list[str]:
    return [f'{outcome}_pre', 'age', 'sex_male', 'group_tms', 'esrs']

def model_columns(prepared: PreparedData, outcome: str, block: str) -> list[str]:
    if block not in prepared.block_feature_columns:
        raise KeyError(block)
    return [*clinical_columns(outcome), *prepared.block_feature_columns[block]]

def make_outer_splits(groups: np.ndarray, repeats: int) -> list[dict]:
    records = []
    for repeat in range(repeats):
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + repeat)
        for (fold, (train, test)) in enumerate(splitter.split(np.zeros(len(groups)), groups)):
            records.append({'repeat': repeat, 'fold': fold, 'train': train, 'test': test})
    return records

def model_candidates(algorithm: str) -> list[dict]:
    if algorithm == 'ridge':
        return [{'alpha': float(alpha)} for alpha in np.logspace(4, -4, 9)]
    if algorithm == 'elasticnet':
        return [{'alpha': float(alpha), 'l1_ratio': float(l1_ratio)} for alpha in np.logspace(1, -3, 5) for l1_ratio in (0.1, 0.5, 0.9)]
    raise KeyError(algorithm)

def make_model(algorithm: str, parameters: dict):
    if algorithm == 'ridge':
        return Ridge(alpha=parameters['alpha'], max_iter=100000)
    if algorithm == 'elasticnet':
        return ElasticNet(alpha=parameters['alpha'], l1_ratio=parameters['l1_ratio'], max_iter=100000, tol=0.0001, random_state=SEED)
    raise KeyError(algorithm)

def make_pipeline(algorithm: str, parameters: dict) -> Pipeline:
    return Pipeline([('imputer', SimpleImputer(strategy='median')), ('scale', StandardScaler()), ('model', make_model(algorithm, parameters))])

def tune_predict(prepared: PreparedData, outcome: str, block: str, algorithm: str, split: dict) -> tuple[np.ndarray, np.ndarray, dict, list[dict]]:
    columns = model_columns(prepared, outcome, block)
    x = prepared.table[columns].to_numpy(dtype=float)
    y = prepared.table[f'{outcome}_post'].to_numpy(dtype=float)
    groups = prepared.table['group_tms'].to_numpy(dtype=int)
    (outer_train, outer_test) = (split['train'], split['test'])
    inner_splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED + 10000 + split['repeat'] * 10 + split['fold'])
    inner_splits = list(inner_splitter.split(np.zeros(len(outer_train)), groups[outer_train]))
    best_score = np.inf
    best_parameters = None
    for parameters in model_candidates(algorithm):
        errors = []
        for (train_local, valid_local) in inner_splits:
            train = outer_train[train_local]
            valid = outer_train[valid_local]
            pipeline = make_pipeline(algorithm, parameters)
            pipeline.fit(x[train], y[train])
            errors.append(mean_absolute_error(y[valid], pipeline.predict(x[valid])))
        score = float(np.mean(errors))
        if score < best_score - 1e-12:
            best_score = score
            best_parameters = parameters
    if best_parameters is None:
        raise RuntimeError('No tuning candidate selected')
    final_pipeline = make_pipeline(algorithm, best_parameters)
    final_pipeline.fit(x[outer_train], y[outer_train])
    prediction = np.asarray(final_pipeline.predict(x[outer_test]), dtype=float)
    coefficients = np.asarray(final_pipeline.named_steps['model'].coef_, dtype=float).reshape(-1)
    coefficient_rows = [{'feature': feature, 'standardized_coefficient': float(value)} for (feature, value) in zip(columns, coefficients)]
    tuning = {'repeat': split['repeat'], 'fold': split['fold'], 'inner_mae': best_score, 'parameters': json.dumps(best_parameters, sort_keys=True), 'n_features': len(columns)}
    return (outer_test, prediction, tuning, coefficient_rows)

def safe_correlation(function, observed: np.ndarray, predicted: np.ndarray) -> float:
    if np.std(observed) == 0 or np.std(predicted) == 0:
        return float('nan')
    return float(function(observed, predicted)[0])

def metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {'mae': float(mean_absolute_error(observed, predicted)), 'rmse': float(np.sqrt(mean_squared_error(observed, predicted))), 'r2': float(r2_score(observed, predicted)), 'pearson_r': safe_correlation(pearsonr, observed, predicted), 'spearman_rho': safe_correlation(spearmanr, observed, predicted)}

def improvement_values(outcome: str, baseline: np.ndarray, post: np.ndarray) -> np.ndarray:
    return baseline - post if LOWER_IS_BETTER[outcome] else post - baseline

def run_condition(prepared: PreparedData, outcome: str, block: str, algorithm: str, splits: list[dict], jobs: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print(f'{algorithm} | {outcome} | {block}', flush=True)
    fitted = Parallel(n_jobs=jobs, prefer='processes', max_nbytes='100K')((delayed(tune_predict)(prepared, outcome, block, algorithm, split) for split in splits))
    table = prepared.table
    target = table[f'{outcome}_post'].to_numpy(dtype=float)
    baseline = table[f'{outcome}_pre'].to_numpy(dtype=float)
    ids = table['subject_id'].to_numpy()
    prediction_rows = []
    tuning_rows = []
    coefficient_rows = []
    for (split, (test, prediction, tuning, coefficients)) in zip(splits, fitted):
        observed_improvement = improvement_values(outcome, baseline[test], target[test])
        predicted_improvement = improvement_values(outcome, baseline[test], prediction)
        for (local, row_index) in enumerate(test):
            prediction_rows.append({'algorithm': algorithm, 'outcome': outcome, 'block': block, 'repeat': split['repeat'], 'fold': split['fold'], 'subject_id': ids[row_index], 'group': table.iloc[row_index]['group'], 'observed_post': target[row_index], 'predicted_post': prediction[local], 'observed_improvement': observed_improvement[local], 'predicted_improvement': predicted_improvement[local]})
        tuning_rows.append({'algorithm': algorithm, 'outcome': outcome, 'block': block, **tuning})
        coefficient_rows.extend(({'algorithm': algorithm, 'outcome': outcome, 'block': block, 'repeat': split['repeat'], 'fold': split['fold'], **row} for row in coefficients))
    predictions = pd.DataFrame(prediction_rows)
    repeat_rows = []
    for (repeat, frame) in predictions.groupby('repeat'):
        post_metrics = metrics(frame['observed_post'].to_numpy(), frame['predicted_post'].to_numpy())
        improvement_metrics = metrics(frame['observed_improvement'].to_numpy(), frame['predicted_improvement'].to_numpy())
        repeat_rows.append({'algorithm': algorithm, 'outcome': outcome, 'block': block, 'repeat': repeat, **{f'post_{key}': value for (key, value) in post_metrics.items()}, **{f'improvement_{key}': value for (key, value) in improvement_metrics.items()}})
    return (predictions, pd.DataFrame(tuning_rows), pd.DataFrame(repeat_rows), pd.DataFrame(coefficient_rows))

def summarize_repeats(repeat_metrics: pd.DataFrame) -> pd.DataFrame:
    keys = ['algorithm', 'outcome', 'block']
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
    output = pd.DataFrame(rows)
    output['post_mae_rank'] = output.groupby(['algorithm', 'outcome'])['post_mae_mean'].rank(method='min', ascending=True)
    return output.sort_values(['algorithm', 'outcome', 'post_mae_rank']).reset_index(drop=True)

def paired_best_summary(repeat_metrics: pd.DataFrame, performance: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for algorithm in sorted(performance['algorithm'].unique()):
        for outcome in OUTCOMES:
            subset = performance[(performance['algorithm'] == algorithm) & (performance['outcome'] == outcome)].sort_values('post_mae_mean')
            best = subset.iloc[0]['block']
            repeat_subset = repeat_metrics[(repeat_metrics['algorithm'] == algorithm) & (repeat_metrics['outcome'] == outcome)].pivot(index='repeat', columns='block', values='post_mae')
            for other in subset['block'].iloc[1:]:
                difference = repeat_subset[other] - repeat_subset[best]
                rows.append({'algorithm': algorithm, 'outcome': outcome, 'selected_best_block': best, 'comparison_block': other, 'positive_means_selected_best_has_lower_mae': True, 'mean_mae_advantage': float(difference.mean()), 'repeat_interval_low': float(difference.quantile(0.025)), 'repeat_interval_high': float(difference.quantile(0.975)), 'fraction_repeats_selected_best_wins': float(np.mean(difference > 0))})
    return pd.DataFrame(rows)

def aggregate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    keys = ['algorithm', 'outcome', 'block', 'subject_id', 'group']
    return predictions.groupby(keys, as_index=False).agg(observed_post=('observed_post', 'first'), predicted_post=('predicted_post', 'mean'), predicted_post_sd=('predicted_post', 'std'), observed_improvement=('observed_improvement', 'first'), predicted_improvement=('predicted_improvement', 'mean'), predicted_improvement_sd=('predicted_improvement', 'std'), n_predictions=('predicted_post', 'size'))

def coefficient_stability(coefficients: pd.DataFrame) -> pd.DataFrame:
    keys = ['algorithm', 'outcome', 'block', 'feature']
    rows = []
    for (values, frame) in coefficients.groupby(keys, sort=False):
        coef = frame['standardized_coefficient'].to_numpy(dtype=float)
        nonzero = np.abs(coef) > 1e-12
        rows.append({**dict(zip(keys, values)), 'mean_coefficient': float(np.mean(coef)), 'mean_absolute_coefficient': float(np.mean(np.abs(coef))), 'median_coefficient': float(np.median(coef)), 'nonzero_fraction': float(np.mean(nonzero)), 'positive_fraction': float(np.mean(coef > 0)), 'sign_consistency': float(max(np.mean(coef >= 0), np.mean(coef <= 0))), 'n_outer_folds': int(len(coef))})
    return pd.DataFrame(rows).sort_values(['algorithm', 'outcome', 'block', 'mean_absolute_coefficient'], ascending=[True, True, True, False])

def save_splits(prepared: PreparedData, splits: list[dict]) -> None:
    ids = prepared.table['subject_id'].to_numpy()
    records = [{'repeat': split['repeat'], 'fold': split['fold'], 'train_subject_ids': ids[split['train']].tolist(), 'test_subject_ids': ids[split['test']].tolist()} for split in splits]
    (OUTPUT_ROOT / 'nested_cv_splits.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Five post-treatment outcome models; every candidate block includes clinical features.')
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
    export_columns = ['subject_id', 'group', 'group_tms', 'age', 'sex_male', 'esrs', *[f'{outcome}_{time}' for outcome in OUTCOMES for time in ('pre', 'post')], *sorted({column for columns in prepared.block_feature_columns.values() for column in columns})]
    prepared.table[export_columns].to_csv(OUTPUT_ROOT / 'five_outcome_baseline_feature_table.csv', index=False, encoding='utf-8-sig')
    (OUTPUT_ROOT / 'feature_block_definitions.json').write_text(json.dumps(prepared.block_feature_columns, ensure_ascii=False, indent=2), encoding='utf-8')
    splits = make_outer_splits(prepared.table['group_tms'].to_numpy(dtype=int), args.repeats)
    save_splits(prepared, splits)
    jobs_to_run = [(algorithm, outcome, block) for algorithm in ('ridge', 'elasticnet') for outcome in OUTCOMES for block in BLOCK_LABELS]
    all_predictions = []
    all_tuning = []
    all_repeat_metrics = []
    all_coefficients = []
    for (index, (algorithm, outcome, block)) in enumerate(jobs_to_run, start=1):
        print(f'[{index}/{len(jobs_to_run)}]', end=' ', flush=True)
        (predictions, tuning, repeat_metrics, coefficients) = run_condition(prepared, outcome, block, algorithm, splits, args.jobs)
        all_predictions.append(predictions)
        all_tuning.append(tuning)
        all_repeat_metrics.append(repeat_metrics)
        all_coefficients.append(coefficients)
    predictions = pd.concat(all_predictions, ignore_index=True)
    tuning = pd.concat(all_tuning, ignore_index=True)
    repeat_metrics = pd.concat(all_repeat_metrics, ignore_index=True)
    coefficients = pd.concat(all_coefficients, ignore_index=True)
    performance = summarize_repeats(repeat_metrics)
    best_comparisons = paired_best_summary(repeat_metrics, performance)
    aggregated = aggregate_predictions(predictions)
    coefficient_summary = coefficient_stability(coefficients)
    predictions.to_csv(OUTPUT_ROOT / 'repeated_oof_predictions.csv', index=False, encoding='utf-8-sig')
    aggregated.to_csv(OUTPUT_ROOT / 'participant_mean_oof_predictions.csv', index=False, encoding='utf-8-sig')
    tuning.to_csv(OUTPUT_ROOT / 'outer_fold_tuning.csv', index=False, encoding='utf-8-sig')
    repeat_metrics.to_csv(OUTPUT_ROOT / 'repeat_level_performance.csv', index=False, encoding='utf-8-sig')
    performance.to_csv(OUTPUT_ROOT / 'repeat_performance_summary.csv', index=False, encoding='utf-8-sig')
    best_comparisons.to_csv(OUTPUT_ROOT / 'selected_best_pairwise_summary.csv', index=False, encoding='utf-8-sig')
    coefficients.to_csv(OUTPUT_ROOT / 'outer_fold_coefficients.csv', index=False, encoding='utf-8-sig')
    coefficient_summary.to_csv(OUTPUT_ROOT / 'coefficient_stability_summary.csv', index=False, encoding='utf-8-sig')
    manifest.update({'outer_cv': f'treatment-stratified 5-fold repeated {args.repeats} times', 'inner_cv': 'treatment-stratified 4-fold tuning inside each outer training fold', 'algorithms': ['ridge', 'elasticnet'], 'random_seed': SEED, 'limitations': ['Internal validation only; no independent external cohort.', 'All seven candidate blocks include the same compact clinical base; no clinical-only block was run.', 'Feature-family exploration and best-block ranking are multiple exploratory comparisons.', 'G2/VIS/SomMot hypotheses were informed by post-treatment imaging results from an overlapping cohort.', 'Treatment group conditions prognosis and does not estimate causal individual treatment benefit.']})
    (OUTPUT_ROOT / 'analysis_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Completed. Outputs: {OUTPUT_ROOT}', flush=True)
if __name__ == '__main__':
    main()
