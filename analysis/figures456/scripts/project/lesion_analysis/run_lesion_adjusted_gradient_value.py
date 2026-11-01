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
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
os.environ.setdefault('JOBLIB_TEMP_FOLDER', _portable_path('project', '.cache/joblib'))
Path(os.environ['JOBLIB_TEMP_FOLDER']).mkdir(parents=True, exist_ok=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
PROJECT_ROOT = Path(_portable_path('project', ''))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from ml_analysis.run_clinical_gradient_ml import CORTICAL_ROOT, SEED, SUBCORTICAL_ROOT, find_baseline_arrays, load_dataset
ANALYSIS_ROOT = PROJECT_ROOT / 'lesion_analysis'
LESION_TABLE = ANALYSIS_ROOT / 'results_abcd' / 'tables' / 'D2_subcortical_gradient_subject_metrics.csv'
OUTPUT_ROOT = ANALYSIS_ROOT / 'lesion_adjusted_gradient_value'
TABLES = OUTPUT_ROOT / 'tables'
FIGURES = OUTPUT_ROOT / 'figures'
SPLITS = OUTPUT_ROOT / 'splits'
OUTCOMES = ('NIHSS', 'ADL')
BLOCKS = ('clinical', 'clinical_lesion', 'clinical_gradient', 'combined')
BLOCK_LABELS = {'clinical': 'Clinical', 'clinical_lesion': 'Clinical + lesion', 'clinical_gradient': 'Clinical + gradient', 'combined': 'Clinical + lesion + gradient'}
BLOCK_COLORS = {'clinical': '#777777', 'clinical_lesion': '#D07A3B', 'clinical_gradient': '#4C78A8', 'combined': '#248F78'}
LESION_FEATURES = ('log1p_lesion_volume', 'mask_lateralization_index', 'log1p_brainstem_fraction_pct', 'log1p_deep_subcortical_gray_fraction_pct', 'log1p_cerebral_white_matter_fraction_pct', 'log1p_cerebral_cortex_fraction_pct')
GRADIENT_SUMMARY_FEATURES = ('cortical_G1_sd', 'cortical_G2_sd', 'subcortical_G1_sd', 'subcortical_G2_sd')
MANUAL_REVIEW_FLAGS = {'PRIVATE_SUBJECT_040', 'PRIVATE_SUBJECT_006', 'PRIVATE_SUBJECT_038'}
PCA_CONFIGS = ((2, 2), (5, 2), (5, 5), (10, 5))
RIDGE_ALPHAS = tuple((float(value) for value in np.logspace(3, -3, 7)))

@dataclass
class PreparedData:
    table: pd.DataFrame
    cortical: np.ndarray
    subcortical: np.ndarray

def ensure_dirs() -> None:
    for path in (TABLES, FIGURES, SPLITS):
        path.mkdir(parents=True, exist_ok=True)

def subset_prepared(prepared: PreparedData, keep: np.ndarray) -> PreparedData:
    indices = np.flatnonzero(keep)
    return PreparedData(table=prepared.table.iloc[indices].reset_index(drop=True), cortical=prepared.cortical[indices], subcortical=prepared.subcortical[indices])

def load_inputs() -> PreparedData:
    (clinical, _) = load_dataset()
    clinical = clinical.copy()
    clinical['subject_id'] = clinical['subject_id'].astype(str)
    lesion = pd.read_csv(LESION_TABLE, dtype={'subject_id': str})
    lesion_columns = ['subject_id', 'log1p_lesion_volume', 'mask_lateralization_index', 'brainstem_fraction_pct', 'deep_subcortical_gray_fraction_pct', 'cerebral_white_matter_fraction_pct', 'cerebral_cortex_fraction_pct', 'outside_mni_brain_fraction_pct']
    table = clinical.merge(lesion[lesion_columns], on='subject_id', how='inner', validate='one_to_one')
    table = table.sort_values('subject_id').reset_index(drop=True)
    table['manual_review_flag'] = table['subject_id'].isin(MANUAL_REVIEW_FLAGS)
    for compartment in ('brainstem', 'deep_subcortical_gray', 'cerebral_white_matter', 'cerebral_cortex'):
        table[f'log1p_{compartment}_fraction_pct'] = np.log1p(table[f'{compartment}_fraction_pct'].astype(float))
    cortical_paths = find_baseline_arrays(CORTICAL_ROOT)
    subcortical_paths = find_baseline_arrays(SUBCORTICAL_ROOT)
    cortical_rows = []
    subcortical_rows = []
    summary_rows = []
    for subject_id in table['subject_id']:
        cortical = np.asarray(np.load(cortical_paths[subject_id])[:, :2], dtype=float)
        subcortical = np.asarray(np.load(subcortical_paths[subject_id])[:, :2], dtype=float)
        if cortical.shape != (400, 2) or subcortical.shape != (54, 2):
            raise RuntimeError(f'Unexpected baseline gradient shape for {subject_id}: {cortical.shape}, {subcortical.shape}')
        if not np.all(np.isfinite(cortical)) or not np.all(np.isfinite(subcortical)):
            raise RuntimeError(f'Non-finite baseline gradient values for {subject_id}')
        cortical_rows.append(cortical.reshape(-1))
        subcortical_rows.append(subcortical.reshape(-1))
        summary_rows.append({'subject_id': subject_id, 'cortical_G1_sd': float(np.std(cortical[:, 0], ddof=1)), 'cortical_G2_sd': float(np.std(cortical[:, 1], ddof=1)), 'subcortical_G1_sd': float(np.std(subcortical[:, 0], ddof=1)), 'subcortical_G2_sd': float(np.std(subcortical[:, 1], ddof=1))})
    table = table.merge(pd.DataFrame(summary_rows), on='subject_id', validate='one_to_one')
    required = ['age', 'sex_male', 'group_tms', 'esrs', 'NIHSS_pre', 'NIHSS_post', 'ADL_pre', 'ADL_post', *LESION_FEATURES, *GRADIENT_SUMMARY_FEATURES]
    if len(table) != 38 or table['subject_id'].nunique() != 38:
        raise RuntimeError(f'Expected the 38-subject lesion-gradient cohort, found {len(table)}')
    if table[required].isna().any().any():
        missing = table.loc[table[required].isna().any(axis=1), ['subject_id', *required]]
        raise RuntimeError(f'Missing required values:\n{missing}')
    return PreparedData(table=table, cortical=np.vstack(cortical_rows), subcortical=np.vstack(subcortical_rows))

def clinical_columns(outcome: str) -> list[str]:
    return [f'{outcome}_pre', 'age', 'sex_male', 'group_tms', 'esrs']

def block_uses_lesion(block: str) -> bool:
    return block in {'clinical_lesion', 'combined'}

def block_uses_gradient(block: str) -> bool:
    return block in {'clinical_gradient', 'combined'}

def pca_configs(block: str) -> tuple[tuple[int, int], ...]:
    return PCA_CONFIGS if block_uses_gradient(block) else ((0, 0),)

def make_outer_splits(groups: np.ndarray, repeats: int, seed_offset: int=0) -> list[dict]:
    rows = []
    for repeat in range(repeats):
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + seed_offset + repeat)
        for (fold, (train, test)) in enumerate(splitter.split(np.zeros(len(groups)), groups)):
            rows.append({'repeat': repeat, 'fold': fold, 'train': train, 'test': test})
    return rows

def make_inner_splits(groups: np.ndarray, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(groups)), groups))

def scaled_part(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pipe = Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
    return (pipe.fit_transform(train), pipe.transform(test))

def pca_part(train: np.ndarray, test: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    pipe = Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler()), ('pca', PCA(n_components=n_components, svd_solver='full'))])
    return (pipe.fit_transform(train), pipe.transform(test))

def transform_block(prepared: PreparedData, outcome: str, block: str, train: np.ndarray, test: np.ndarray, cortical_components: int, subcortical_components: int) -> tuple[np.ndarray, np.ndarray]:
    table = prepared.table
    clinical = table[clinical_columns(outcome)].to_numpy(float)
    (train_parts, test_parts) = ([], [])
    (train_part, test_part) = scaled_part(clinical[train], clinical[test])
    train_parts.append(train_part)
    test_parts.append(test_part)
    if block_uses_lesion(block):
        lesion = table[list(LESION_FEATURES)].to_numpy(float)
        (train_part, test_part) = scaled_part(lesion[train], lesion[test])
        train_parts.append(train_part)
        test_parts.append(test_part)
    if block_uses_gradient(block):
        (train_part, test_part) = pca_part(prepared.cortical[train], prepared.cortical[test], cortical_components)
        train_parts.append(train_part)
        test_parts.append(test_part)
        (train_part, test_part) = pca_part(prepared.subcortical[train], prepared.subcortical[test], subcortical_components)
        train_parts.append(train_part)
        test_parts.append(test_part)
    return (np.column_stack(train_parts), np.column_stack(test_parts))

def tune_outer_fold(prepared: PreparedData, outcome: str, block: str, split: dict) -> tuple[np.ndarray, dict]:
    target = prepared.table[f'{outcome}_post'].to_numpy(float)
    groups = prepared.table['group_tms'].to_numpy(int)
    outer_train = split['train']
    outer_test = split['test']
    inner_splits = make_inner_splits(groups[outer_train], SEED + 10000 + split['repeat'] * 10 + split['fold'])
    candidates = []
    for (cortical_components, subcortical_components) in pca_configs(block):
        transformed_folds = []
        for (inner_train_local, inner_test_local) in inner_splits:
            inner_train = outer_train[inner_train_local]
            inner_test = outer_train[inner_test_local]
            (x_train, x_test) = transform_block(prepared, outcome, block, inner_train, inner_test, cortical_components, subcortical_components)
            transformed_folds.append((inner_train, inner_test, x_train, x_test))
        for alpha in RIDGE_ALPHAS:
            fold_mae = []
            for (inner_train, inner_test, x_train, x_test) in transformed_folds:
                model = Ridge(alpha=alpha, max_iter=100000)
                model.fit(x_train, target[inner_train])
                fold_mae.append(mean_absolute_error(target[inner_test], model.predict(x_test)))
            candidates.append({'cortical_components': cortical_components, 'subcortical_components': subcortical_components, 'alpha': alpha, 'inner_mae': float(np.mean(fold_mae))})
    best = min(candidates, key=lambda row: (row['inner_mae'], row['cortical_components'] + row['subcortical_components'], -row['alpha']))
    (x_train, x_test) = transform_block(prepared, outcome, block, outer_train, outer_test, best['cortical_components'], best['subcortical_components'])
    model = Ridge(alpha=best['alpha'], max_iter=100000)
    model.fit(x_train, target[outer_train])
    return (np.asarray(model.predict(x_test), dtype=float), best)

def run_job(cohort: str, prepared: PreparedData, outcome: str, block: str, splits: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows = []
    tuning_rows = []
    table = prepared.table
    for split in splits:
        (prediction, best) = tune_outer_fold(prepared, outcome, block, split)
        for (index, value) in zip(split['test'], prediction):
            prediction_rows.append({'cohort': cohort, 'outcome': outcome, 'block': block, 'repeat': split['repeat'], 'fold': split['fold'], 'subject_id': table.loc[index, 'subject_id'], 'group': table.loc[index, 'group'], 'observed_pre': float(table.loc[index, f'{outcome}_pre']), 'observed_post': float(table.loc[index, f'{outcome}_post']), 'predicted_post': float(value)})
        tuning_rows.append({'cohort': cohort, 'outcome': outcome, 'block': block, 'repeat': split['repeat'], 'fold': split['fold'], **best})
    return (pd.DataFrame(prediction_rows), pd.DataFrame(tuning_rows))

def metric_values(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    correlation = float(pearsonr(observed, predicted).statistic) if np.std(predicted) > 0 else np.nan
    return {'mae': float(mean_absolute_error(observed, predicted)), 'rmse': float(np.sqrt(mean_squared_error(observed, predicted))), 'r2': float(r2_score(observed, predicted)), 'pearson_r': correlation}

def repeat_performance(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (keys, subset) in predictions.groupby(['cohort', 'outcome', 'block', 'repeat'], sort=False):
        (cohort, outcome, block, repeat) = keys
        post = metric_values(subset['observed_post'].to_numpy(float), subset['predicted_post'].to_numpy(float))
        if outcome == 'NIHSS':
            observed_change = subset['observed_pre'] - subset['observed_post']
            predicted_change = subset['observed_pre'] - subset['predicted_post']
        else:
            observed_change = subset['observed_post'] - subset['observed_pre']
            predicted_change = subset['predicted_post'] - subset['observed_pre']
        change = metric_values(observed_change.to_numpy(float), predicted_change.to_numpy(float))
        rows.append({'cohort': cohort, 'outcome': outcome, 'block': block, 'repeat': int(repeat), 'n': len(subset), **{f'post_{key}': value for (key, value) in post.items()}, **{f'change_{key}': value for (key, value) in change.items()}})
    return pd.DataFrame(rows)

def summarize_performance(performance: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_columns = ['post_mae', 'post_rmse', 'post_r2', 'post_pearson_r', 'change_mae', 'change_rmse', 'change_r2', 'change_pearson_r']
    for (keys, subset) in performance.groupby(['cohort', 'outcome', 'block'], sort=False):
        (cohort, outcome, block) = keys
        for metric in metric_columns:
            values = subset[metric].dropna().to_numpy(float)
            rows.append({'cohort': cohort, 'outcome': outcome, 'block': block, 'metric': metric, 'n_repeats': len(values), 'mean': float(np.mean(values)), 'median': float(np.median(values)), 'sd': float(np.std(values, ddof=1)), 'repeat_interval_low': float(np.percentile(values, 2.5)), 'repeat_interval_high': float(np.percentile(values, 97.5))})
    return pd.DataFrame(rows)

def incremental_mae(performance: pd.DataFrame) -> pd.DataFrame:
    comparisons = (('clinical', 'clinical_lesion', 'lesion beyond clinical', False), ('clinical', 'clinical_gradient', 'gradient beyond clinical', False), ('clinical_lesion', 'combined', 'gradient beyond clinical + lesion', True), ('clinical_gradient', 'combined', 'lesion beyond clinical + gradient', False))
    rows = []
    for cohort in performance['cohort'].unique():
        cohort_frame = performance.loc[performance['cohort'] == cohort]
        for outcome in OUTCOMES:
            outcome_frame = cohort_frame.loc[cohort_frame['outcome'] == outcome]
            available = set(outcome_frame['block'])
            for (reference, comparison, label, primary) in comparisons:
                if reference not in available or comparison not in available:
                    continue
                reference_values = outcome_frame.loc[outcome_frame['block'] == reference, ['repeat', 'post_mae']].set_index('repeat')['post_mae']
                comparison_values = outcome_frame.loc[outcome_frame['block'] == comparison, ['repeat', 'post_mae']].set_index('repeat')['post_mae']
                delta = (reference_values - comparison_values).dropna().to_numpy(float)
                rows.append({'cohort': cohort, 'outcome': outcome, 'contrast': label, 'reference_block': reference, 'comparison_block': comparison, 'primary_contrast': primary, 'positive_means_comparison_improves': True, 'mean_mae_reduction': float(np.mean(delta)), 'median_mae_reduction': float(np.median(delta)), 'repeat_interval_low': float(np.percentile(delta, 2.5)), 'repeat_interval_high': float(np.percentile(delta, 97.5)), 'fraction_repeats_improved': float(np.mean(delta > 0)), 'n_repeats': len(delta)})
    return pd.DataFrame(rows)

def zscore(values: pd.Series | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    sd = float(np.std(array, ddof=1))
    if not np.isfinite(sd) or sd == 0:
        raise RuntimeError('Cannot standardize a constant variable')
    return (array - float(np.mean(array))) / sd

def fit_hc3_models(prepared: PreparedData, cohort: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = prepared.table
    base_columns = ['baseline_outcome_z', 'group_tms', 'age_z', 'sex_male', 'esrs_z', *[f'{feature}_z' for feature in LESION_FEATURES]]
    rows = []
    joint_rows = []
    for outcome in OUTCOMES:
        direction = -1.0 if outcome == 'NIHSS' else 1.0
        baseline = direction * table[f'{outcome}_pre'].to_numpy(float)
        post = direction * table[f'{outcome}_post'].to_numpy(float)
        baseline_mean = float(np.mean(baseline))
        baseline_sd = float(np.std(baseline, ddof=1))
        model_data = pd.DataFrame({'baseline_outcome_z': (baseline - baseline_mean) / baseline_sd, 'group_tms': table['group_tms'].to_numpy(float), 'age_z': zscore(table['age']), 'sex_male': table['sex_male'].to_numpy(float), 'esrs_z': zscore(table['esrs'])})
        for feature in LESION_FEATURES:
            model_data[f'{feature}_z'] = zscore(table[feature])
        for feature in GRADIENT_SUMMARY_FEATURES:
            model_data[f'{feature}_z'] = zscore(table[feature])
        y = (post - baseline_mean) / baseline_sd
        base_x = sm.add_constant(model_data[base_columns], has_constant='add')
        full_columns = base_columns + [f'{feature}_z' for feature in GRADIENT_SUMMARY_FEATURES]
        full_x = sm.add_constant(model_data[full_columns], has_constant='add')
        base_result = sm.OLS(y, base_x).fit(cov_type='HC3')
        full_result = sm.OLS(y, full_x).fit(cov_type='HC3')
        restrictions = np.zeros((len(GRADIENT_SUMMARY_FEATURES), len(full_result.params)))
        parameter_names = list(full_result.params.index)
        for (row_index, feature) in enumerate(GRADIENT_SUMMARY_FEATURES):
            restrictions[row_index, parameter_names.index(f'{feature}_z')] = 1.0
        wald = full_result.wald_test(restrictions, use_f=True, scalar=True)
        partial_r2 = max(0.0, 1.0 - float(np.sum(full_result.resid ** 2)) / float(np.sum(base_result.resid ** 2)))
        joint_rows.append({'cohort': cohort, 'outcome': outcome, 'n': int(full_result.nobs), 'test': 'four baseline gradient dispersion summaries added jointly', 'wald_f': float(wald.statistic), 'df_num': len(GRADIENT_SUMMARY_FEATURES), 'df_denom': float(full_result.df_resid), 'p_value_hc3': float(wald.pvalue), 'partial_r2_in_sample': partial_r2, 'base_r_squared': float(base_result.rsquared), 'full_r_squared': float(full_result.rsquared)})
        confidence = full_result.conf_int()
        for feature in GRADIENT_SUMMARY_FEATURES:
            term = f'{feature}_z'
            rows.append({'cohort': cohort, 'outcome': outcome, 'n': int(full_result.nobs), 'term': feature, 'standardized_beta': float(full_result.params[term]), 'ci_low': float(confidence.loc[term, 0]), 'ci_high': float(confidence.loc[term, 1]), 'robust_se_hc3': float(full_result.bse[term]), 'p_value': float(full_result.pvalues[term])})
    coefficients = pd.DataFrame(rows)
    coefficients['fdr_q'] = np.nan
    for cohort_name in coefficients['cohort'].unique():
        selected = coefficients['cohort'] == cohort_name
        coefficients.loc[selected, 'fdr_q'] = multipletests(coefficients.loc[selected, 'p_value'], method='fdr_bh')[1]
    joint = pd.DataFrame(joint_rows)
    joint['fdr_q'] = np.nan
    for cohort_name in joint['cohort'].unique():
        selected = joint['cohort'] == cohort_name
        joint.loc[selected, 'fdr_q'] = multipletests(joint.loc[selected, 'p_value_hc3'], method='fdr_bh')[1]
    return (coefficients, joint)

def permute_gradients_within_group(prepared: PreparedData, seed: int) -> PreparedData:
    rng = np.random.default_rng(seed)
    assignment = np.arange(len(prepared.table))
    groups = prepared.table['group_tms'].to_numpy(int)
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        assignment[indices] = rng.permutation(indices)
    return PreparedData(table=prepared.table, cortical=prepared.cortical[assignment], subcortical=prepared.subcortical[assignment])

def permutation_one(prepared: PreparedData, outcome: str, splits: list[dict], permutation: int) -> float:
    permuted = permute_gradients_within_group(prepared, SEED + 500000 + permutation)
    (predictions, _) = run_job('permuted', permuted, outcome, 'combined', splits)
    return float(mean_absolute_error(predictions['observed_post'], predictions['predicted_post']))

def run_permutation_tests(prepared: PreparedData, predictions: pd.DataFrame, splits: list[dict], permutations: int, jobs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if permutations <= 0:
        return (pd.DataFrame(), pd.DataFrame())
    fixed_splits = [split for split in splits if split['repeat'] == 0]
    summary_rows = []
    null_rows = []
    for outcome in OUTCOMES:
        actual = predictions.loc[(predictions['cohort'] == 'primary_full') & (predictions['outcome'] == outcome) & (predictions['repeat'] == 0)]
        base = actual.loc[actual['block'] == 'clinical_lesion']
        combined = actual.loc[actual['block'] == 'combined']
        base_mae = float(mean_absolute_error(base['observed_post'], base['predicted_post']))
        combined_mae = float(mean_absolute_error(combined['observed_post'], combined['predicted_post']))
        observed_reduction = base_mae - combined_mae
        null_mae = Parallel(n_jobs=jobs, verbose=5)((delayed(permutation_one)(prepared, outcome, fixed_splits, permutation) for permutation in range(permutations)))
        null_reduction = base_mae - np.asarray(null_mae, dtype=float)
        p_value = (1.0 + float(np.count_nonzero(null_reduction >= observed_reduction))) / (permutations + 1.0)
        summary_rows.append({'outcome': outcome, 'test': 'gradient block permutation conditional on clinical and lesion block', 'permutation_scheme': 'baseline cortical and subcortical gradient rows jointly shuffled within treatment group', 'clinical_lesion_mae': base_mae, 'observed_combined_mae': combined_mae, 'observed_mae_reduction': observed_reduction, 'null_mean_mae_reduction': float(np.mean(null_reduction)), 'null_p95_mae_reduction': float(np.percentile(null_reduction, 95)), 'one_sided_p_value': p_value, 'permutations': permutations, 'outer_cv': 'first fixed stratified 5-fold repeat', 'inner_cv': 'stratified 4-fold tuning repeated under every permutation'})
        for (permutation, (mae, reduction)) in enumerate(zip(null_mae, null_reduction)):
            null_rows.append({'outcome': outcome, 'permutation': permutation, 'permuted_combined_mae': mae, 'mae_reduction_vs_clinical_lesion': reduction})
    return (pd.DataFrame(summary_rows), pd.DataFrame(null_rows))

def aggregate_oof(predictions: pd.DataFrame) -> pd.DataFrame:
    return predictions.groupby(['cohort', 'outcome', 'block', 'subject_id', 'group'], as_index=False).agg(observed_pre=('observed_pre', 'first'), observed_post=('observed_post', 'first'), predicted_post=('predicted_post', 'mean'), prediction_sd=('predicted_post', 'std'))

def plot_results(performance: pd.DataFrame, increments: pd.DataFrame) -> None:
    primary = performance.loc[performance['cohort'] == 'primary_full']
    (fig, axes) = plt.subplots(2, 2, figsize=(7.6, 5.4), facecolor='white', gridspec_kw={'width_ratios': (1.0, 1.08)})
    rng = np.random.default_rng(SEED)
    for (row_index, outcome) in enumerate(OUTCOMES):
        outcome_frame = primary.loc[primary['outcome'] == outcome]
        for (position, block) in enumerate(BLOCKS):
            values = outcome_frame.loc[outcome_frame['block'] == block, 'post_mae'].to_numpy(float)
            jitter = rng.uniform(-0.07, 0.07, len(values))
            axes[row_index, 0].scatter(position + jitter, values, s=9, color=BLOCK_COLORS[block], alpha=0.45, edgecolor='none')
            mean = float(np.mean(values))
            (low, high) = np.percentile(values, [2.5, 97.5])
            axes[row_index, 0].errorbar(position, mean, yerr=[[mean - low], [high - mean]], fmt='o', markersize=4.5, color=BLOCK_COLORS[block], capsize=2, linewidth=1)
        axes[row_index, 0].set_xticks(range(len(BLOCKS)), ['Clinical', '+ lesion', '+ gradient', 'Combined'], rotation=12, ha='right')
        axes[row_index, 0].set_ylabel(f'{outcome} post-score MAE')
        axes[row_index, 0].set_title(f'{outcome}: repeated nested CV')
        increment_frame = increments.loc[(increments['cohort'] == 'primary_full') & (increments['outcome'] == outcome)]
        contrasts = ('lesion beyond clinical', 'gradient beyond clinical', 'gradient beyond clinical + lesion')
        for (position, contrast) in enumerate(contrasts):
            row = increment_frame.loc[increment_frame['contrast'] == contrast].iloc[0]
            axes[row_index, 1].errorbar(position, row['mean_mae_reduction'], yerr=[[row['mean_mae_reduction'] - row['repeat_interval_low']], [row['repeat_interval_high'] - row['mean_mae_reduction']]], fmt='o', markersize=5, color=('#D07A3B', '#4C78A8', '#248F78')[position], capsize=2, linewidth=1)
        axes[row_index, 1].axhline(0, color='#777777', linewidth=0.7, linestyle='--')
        axes[row_index, 1].set_xticks(range(3), ['Lesion\n| clinical', 'Gradient\n| clinical', 'Gradient\n| clinical + lesion'])
        axes[row_index, 1].set_ylabel('MAE reduction\n(positive = better)')
        axes[row_index, 1].set_title(f'{outcome}: incremental value')
    for (label, axis) in zip(('A', 'B', 'C', 'D'), axes.ravel()):
        axis.text(-0.14, 1.07, label, transform=axis.transAxes, fontsize=10, fontweight='bold', va='top')
        axis.spines[['top', 'right']].set_visible(False)
        axis.tick_params(width=0.6, length=2.5)
    fig.suptitle('Lesion-adjusted incremental value of baseline gradients', fontsize=10, y=0.995)
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.105, top=0.9, hspace=0.72, wspace=0.52)
    for (suffix, kwargs) in (('png', {'dpi': 300}), ('pdf', {})):
        fig.savefig(FIGURES / f'lesion_adjusted_gradient_value.{suffix}', bbox_inches='tight', facecolor='white', **kwargs)
    plt.close(fig)

def write_summary(summary: pd.DataFrame, increments: pd.DataFrame, permutation: pd.DataFrame, joint: pd.DataFrame) -> None:
    lines = ['# 病灶校正后梯度增量价值分析', '', '## 设计', '', '- 主分析为 38 例同时具有基线临床、基线病灶、皮层及皮层下梯度的共同队列。', '- 主结局分别为治疗后 NIHSS 和 ADL；临床基准包含对应基线量表、年龄、性别、ESRS 和治疗组。', '- 病灶块仅使用治疗前病灶：log1p 总体积、侧化，以及脑干、深部灰质、脑白质和皮层区室比例。', '- 梯度块为皮层 400 ROI 与皮层下 54 ROI 的基线 G1/G2；标准化和 PCA 均严格在训练折内拟合。', '- 正的 MAE reduction 表示增加该信息后预测改善；重复区间描述划分稳定性，不作为独立重复样本的置信区间。', '', '## 重复嵌套交叉验证', '', '| 结局 | 模型 | post MAE | post R2 | 改善量 R2 |', '|---|---|---:|---:|---:|']
    primary_summary = summary.loc[summary['cohort'] == 'primary_full']
    for outcome in OUTCOMES:
        for block in BLOCKS:
            lookup = primary_summary.loc[(primary_summary['outcome'] == outcome) & (primary_summary['block'] == block)].set_index('metric')['mean']
            lines.append(f"| {outcome} | {BLOCK_LABELS[block]} | {lookup['post_mae']:.3f} | {lookup['post_r2']:.3f} | {lookup['change_r2']:.3f} |")
    lines.extend(['', '## 梯度在病灶之上的主增量', ''])
    for outcome in OUTCOMES:
        row = increments.loc[(increments['cohort'] == 'primary_full') & (increments['outcome'] == outcome) & increments['primary_contrast']].iloc[0]
        direction = '降低' if row['mean_mae_reduction'] > 0 else '增加'
        lines.append(f"- {outcome}：联合模型相对临床+病灶模型平均{direction} MAE {abs(row['mean_mae_reduction']):.3f}；重复区间 {row['repeat_interval_low']:.3f} 到 {row['repeat_interval_high']:.3f}；{row['fraction_repeats_improved'] * 100:.1f}% 的重复划分中改善。")
    if not permutation.empty:
        lines.extend(['', '## 条件梯度块置换检验', ''])
        for row in permutation.itertuples():
            lines.append(f'- {row.outcome}：实际 MAE reduction={row.observed_mae_reduction:.3f}，组内梯度置换单侧 P={row.one_sided_p_value:.3f}。')
    lines.extend(['', '## 全样本 HC3 解释性模型', ''])
    for row in joint.loc[joint['cohort'] == 'primary_full'].itertuples():
        lines.append(f'- {row.outcome}：4 个预设基线梯度离散度指标在临床和病灶之后的联合检验 P={row.p_value_hc3:.3f}，FDR q={row.fdr_q:.3f}，样本内 partial R2={row.partial_r2_in_sample:.3f}。')
    lines.extend(['', '## 解释边界', '', '- 这是内部验证的预后分析，不是独立外部验证，也不是个体 TMS 因果获益模型。', '- 梯度若没有改善 MAE，应报告为当前样本未支持独立个体预测增量；不能据此否定其群体或机制意义。', '- 梯度若改善 MAE，也仍需置换检验、QC 排除敏感性和外部队列复现后才能称为可靠预测标志物。'])
    (OUTPUT_ROOT / 'RESULTS_ZH.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

def save_split_manifest(name: str, prepared: PreparedData, splits: list[dict]) -> None:
    subject_ids = prepared.table['subject_id'].to_numpy()
    records = []
    for split in splits:
        records.append({'cohort': name, 'repeat': int(split['repeat']), 'fold': int(split['fold']), 'train_subject_ids': subject_ids[split['train']].tolist(), 'test_subject_ids': subject_ids[split['test']].tolist()})
    (SPLITS / f'nested_cv_splits_{name}.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Test baseline-gradient value beyond clinical and baseline-lesion information.')
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
    primary = load_inputs()
    sensitivity = subset_prepared(primary, ~primary.table['subject_id'].isin(MANUAL_REVIEW_FLAGS).to_numpy())
    cohorts = {'primary_full': primary, 'exclude_manual_review_flags': sensitivity}
    cohort_blocks = {'primary_full': BLOCKS, 'exclude_manual_review_flags': ('clinical_lesion', 'combined')}
    split_sets = {name: make_outer_splits(prepared.table['group_tms'].to_numpy(int), args.repeats, seed_offset=1000 * index) for (index, (name, prepared)) in enumerate(cohorts.items())}
    for (name, prepared) in cohorts.items():
        save_split_manifest(name, prepared, split_sets[name])
    export_columns = ['subject_id', 'group', 'manual_review_flag', 'outside_mni_brain_fraction_pct', 'NIHSS_pre', 'NIHSS_post', 'ADL_pre', 'ADL_post', 'age', 'sex_male', 'group_tms', 'esrs', *LESION_FEATURES, *GRADIENT_SUMMARY_FEATURES]
    primary.table[export_columns].to_csv(TABLES / 'analysis_cohort.csv', index=False, encoding='utf-8-sig')
    jobs_to_run = []
    for (cohort_name, prepared) in cohorts.items():
        for outcome in OUTCOMES:
            for block in cohort_blocks[cohort_name]:
                jobs_to_run.append((cohort_name, prepared, outcome, block, split_sets[cohort_name]))
    results = Parallel(n_jobs=args.jobs, verbose=10)((delayed(run_job)(*job) for job in jobs_to_run))
    predictions = pd.concat([result[0] for result in results], ignore_index=True)
    tuning = pd.concat([result[1] for result in results], ignore_index=True)
    performance = repeat_performance(predictions)
    summary = summarize_performance(performance)
    increments = incremental_mae(performance)
    coefficients = []
    joint_tests = []
    for (cohort_name, prepared) in cohorts.items():
        (cohort_coefficients, cohort_joint) = fit_hc3_models(prepared, cohort_name)
        coefficients.append(cohort_coefficients)
        joint_tests.append(cohort_joint)
    coefficients_frame = pd.concat(coefficients, ignore_index=True)
    joint_frame = pd.concat(joint_tests, ignore_index=True)
    (permutation, permutation_null) = run_permutation_tests(primary, predictions, split_sets['primary_full'], args.permutations, args.jobs)
    predictions.to_csv(TABLES / 'nested_cv_predictions.csv', index=False, encoding='utf-8-sig')
    tuning.to_csv(TABLES / 'outer_fold_tuning.csv', index=False, encoding='utf-8-sig')
    performance.to_csv(TABLES / 'repeat_level_performance.csv', index=False, encoding='utf-8-sig')
    summary.to_csv(TABLES / 'performance_summary.csv', index=False, encoding='utf-8-sig')
    increments.to_csv(TABLES / 'incremental_mae_summary.csv', index=False, encoding='utf-8-sig')
    aggregate_oof(predictions).to_csv(TABLES / 'participant_mean_oof_predictions.csv', index=False, encoding='utf-8-sig')
    coefficients_frame.to_csv(TABLES / 'hc3_gradient_summary_coefficients.csv', index=False, encoding='utf-8-sig')
    joint_frame.to_csv(TABLES / 'hc3_gradient_block_joint_tests.csv', index=False, encoding='utf-8-sig')
    if not permutation.empty:
        permutation.to_csv(TABLES / 'gradient_block_permutation.csv', index=False, encoding='utf-8-sig')
        permutation_null.to_csv(TABLES / 'gradient_block_permutation_null.csv', index=False, encoding='utf-8-sig')
    plot_results(performance, increments)
    write_summary(summary, increments, permutation, joint_frame)
    manifest = {'analysis_date': '2026-08-03', 'primary_cohort_n': int(len(primary.table)), 'primary_group_counts': primary.table['group'].value_counts().sort_index().to_dict(), 'sensitivity_cohort_n': int(len(sensitivity.table)), 'sensitivity_excluded_subjects': sorted(MANUAL_REVIEW_FLAGS), 'outcomes': ['NIHSS_post', 'ADL_post'], 'clinical_block': {outcome: clinical_columns(outcome) for outcome in OUTCOMES}, 'lesion_block': list(LESION_FEATURES), 'lesion_timing': 'baseline only', 'gradient_block': {'timing': 'baseline only', 'raw_dimensions': {'cortical_G1G2': 800, 'subcortical_G1G2': 108}, 'foldwise_pca_candidates': [list(config) for config in PCA_CONFIGS]}, 'model': 'Ridge regression', 'outer_cv': f'stratified 5-fold repeated {args.repeats} times', 'inner_cv': 'stratified 4-fold tuning of Ridge alpha and gradient PCA dimensions', 'ridge_alpha_candidates': list(RIDGE_ALPHAS), 'primary_incremental_contrast': 'combined minus clinical+lesion, expressed as clinical+lesion MAE minus combined MAE', 'gradient_block_permutations': args.permutations, 'permutation_scheme': 'joint cortical/subcortical gradient row shuffle within observed treatment group', 'leakage_control': 'imputation, scaling, PCA, regularization tuning, and fitting occur only within training folds', 'interpretation': 'prognostic prediction conditional on observed treatment group; not an individualized causal treatment effect', 'external_validation': False, 'random_seed': SEED}
    (OUTPUT_ROOT / 'analysis_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Primary cohort: {len(primary.table)}')
    print(f'Sensitivity cohort: {len(sensitivity.table)}')
    print(f'Prediction rows: {len(predictions)}')
    print(f'Output: {OUTPUT_ROOT}')
if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=FutureWarning)
        main()
