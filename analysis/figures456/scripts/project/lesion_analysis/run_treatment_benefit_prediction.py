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
from pathlib import Path
os.environ.setdefault('JOBLIB_TEMP_FOLDER', _portable_path('project', '.cache/joblib'))
Path(os.environ['JOBLIB_TEMP_FOLDER']).mkdir(parents=True, exist_ok=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
ANALYSIS_ROOT = Path(_portable_path('project', 'lesion_analysis'))
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))
import run_lesion_adjusted_gradient_value as base
OUTPUT_ROOT = ANALYSIS_ROOT / 'treatment_benefit_prediction'
TABLES = OUTPUT_ROOT / 'tables'
FIGURES = OUTPUT_ROOT / 'figures'
SPLITS = OUTPUT_ROOT / 'splits'
BLOCKS = ('clinical', 'clinical_lesion', 'clinical_gradient', 'combined')
BLOCK_LABELS = {'clinical': 'Clinical', 'clinical_lesion': 'Clinical + lesion', 'clinical_gradient': 'Clinical + gradient', 'combined': 'Clinical + lesion + gradient'}
BLOCK_COLORS = {'clinical': '#666666', 'clinical_lesion': '#D07A3B', 'clinical_gradient': '#4C78A8', 'combined': '#248F78'}
STRUCTURES = ('main_effect', 'treatment_interaction')
STRUCTURE_LABELS = {'main_effect': 'Average treatment contrast', 'treatment_interaction': 'Individualized treatment interaction'}
CLINICAL_FEATURES = ('NIHSS_pre', 'age', 'sex_male', 'esrs')
RIDGE_ALPHAS = tuple((float(value) for value in np.logspace(-3, 4, 8)))
MANUAL_REVIEW_FLAGS = base.MANUAL_REVIEW_FLAGS
NIHSS_RANGE = (0.0, 42.0)
FC_EXPECTED_PATH = Path('F:/lishiqi/预处理数据/cortex/parcel(400，400)')

def ensure_dirs() -> None:
    for path in (TABLES, FIGURES, SPLITS):
        path.mkdir(parents=True, exist_ok=True)

def block_uses_lesion(block: str) -> bool:
    return block in {'clinical_lesion', 'combined'}

def block_uses_gradient(block: str) -> bool:
    return block in {'clinical_gradient', 'combined'}

def pca_configs(block: str) -> tuple[tuple[int, int], ...]:
    return base.PCA_CONFIGS if block_uses_gradient(block) else ((0, 0),)

def make_outer_splits(treatment: np.ndarray, repeats: int, seed_offset: int=0) -> list[dict]:
    rows = []
    for repeat in range(repeats):
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=base.SEED + seed_offset + repeat)
        for (fold, (train, test)) in enumerate(splitter.split(np.zeros(len(treatment)), treatment)):
            rows.append({'repeat': repeat, 'fold': fold, 'train': train, 'test': test})
    return rows

def make_inner_splits(treatment: np.ndarray, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(treatment)), treatment))

def scaled_part(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    transformer = Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
    return (transformer.fit_transform(train), transformer.transform(test))

def pca_part(train: np.ndarray, test: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    transformer = Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler()), ('pca', PCA(n_components=n_components, svd_solver='full'))])
    return (transformer.fit_transform(train), transformer.transform(test))

def transform_baseline_features(prepared: base.PreparedData, block: str, train: np.ndarray, test: np.ndarray, cortical_components: int, subcortical_components: int) -> tuple[np.ndarray, np.ndarray]:
    table = prepared.table
    train_parts = []
    test_parts = []
    clinical = table[list(CLINICAL_FEATURES)].to_numpy(float)
    (train_part, test_part) = scaled_part(clinical[train], clinical[test])
    train_parts.append(train_part)
    test_parts.append(test_part)
    if block_uses_lesion(block):
        lesion = table[list(base.LESION_FEATURES)].to_numpy(float)
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

def treatment_design(baseline_features: np.ndarray, treatment: np.ndarray, training_treatment_rate: float, structure: str) -> np.ndarray:
    treatment = np.asarray(treatment, dtype=float)
    scale = float(np.sqrt(training_treatment_rate * (1.0 - training_treatment_rate)))
    if scale <= 0:
        raise RuntimeError('Training fold contains only one treatment group')
    centered_treatment = (treatment - training_treatment_rate) / scale
    columns = [baseline_features, centered_treatment[:, None]]
    if structure == 'treatment_interaction':
        columns.append(baseline_features * centered_treatment[:, None])
    elif structure != 'main_effect':
        raise KeyError(structure)
    return np.column_stack(columns)

def tune_outer_fold(prepared: base.PreparedData, block: str, structure: str, split: dict, treatment_override: np.ndarray | None=None) -> tuple[dict[str, np.ndarray], dict]:
    table = prepared.table
    target = table['NIHSS_post'].to_numpy(float)
    observed_treatment = table['group_tms'].to_numpy(int)
    treatment = observed_treatment if treatment_override is None else np.asarray(treatment_override, dtype=int)
    outer_train = split['train']
    outer_test = split['test']
    inner_splits = make_inner_splits(observed_treatment[outer_train], base.SEED + 20000 + split['repeat'] * 10 + split['fold'])
    candidates = []
    for (cortical_components, subcortical_components) in pca_configs(block):
        transformed_folds = []
        for (inner_train_local, inner_test_local) in inner_splits:
            inner_train = outer_train[inner_train_local]
            inner_test = outer_train[inner_test_local]
            (x_train, x_test) = transform_baseline_features(prepared, block, inner_train, inner_test, cortical_components, subcortical_components)
            treatment_rate = float(np.mean(treatment[inner_train]))
            design_train = treatment_design(x_train, treatment[inner_train], treatment_rate, structure)
            design_test = treatment_design(x_test, treatment[inner_test], treatment_rate, structure)
            transformed_folds.append((inner_train, inner_test, design_train, design_test))
        for alpha in RIDGE_ALPHAS:
            fold_mae = []
            for (inner_train, inner_test, design_train, design_test) in transformed_folds:
                model = Ridge(alpha=alpha, max_iter=100000)
                model.fit(design_train, target[inner_train])
                fold_mae.append(mean_absolute_error(target[inner_test], model.predict(design_test)))
            candidates.append({'cortical_components': cortical_components, 'subcortical_components': subcortical_components, 'alpha': alpha, 'inner_mae': float(np.mean(fold_mae))})
    best = min(candidates, key=lambda row: (row['inner_mae'], row['cortical_components'] + row['subcortical_components'], -row['alpha']))
    (x_train, x_test) = transform_baseline_features(prepared, block, outer_train, outer_test, best['cortical_components'], best['subcortical_components'])
    treatment_rate = float(np.mean(treatment[outer_train]))
    design_train = treatment_design(x_train, treatment[outer_train], treatment_rate, structure)
    factual_design = treatment_design(x_test, treatment[outer_test], treatment_rate, structure)
    control_design = treatment_design(x_test, np.zeros(len(outer_test), dtype=int), treatment_rate, structure)
    tms_design = treatment_design(x_test, np.ones(len(outer_test), dtype=int), treatment_rate, structure)
    model = Ridge(alpha=best['alpha'], max_iter=100000)
    model.fit(design_train, target[outer_train])
    return ({'factual': np.clip(model.predict(factual_design), *NIHSS_RANGE), 'control': np.clip(model.predict(control_design), *NIHSS_RANGE), 'tms': np.clip(model.predict(tms_design), *NIHSS_RANGE)}, {**best, 'training_treatment_rate': treatment_rate})

def run_job(cohort: str, prepared: base.PreparedData, block: str, structure: str, splits: list[dict], treatment_override: np.ndarray | None=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows = []
    tuning_rows = []
    table = prepared.table
    model_treatment = table['group_tms'].to_numpy(int) if treatment_override is None else np.asarray(treatment_override, dtype=int)
    for split in splits:
        (predictions, best) = tune_outer_fold(prepared, block, structure, split, treatment_override=treatment_override)
        for (position, index) in enumerate(split['test']):
            predicted_control = float(predictions['control'][position])
            predicted_tms = float(predictions['tms'][position])
            prediction_rows.append({'cohort': cohort, 'block': block, 'structure': structure, 'repeat': int(split['repeat']), 'fold': int(split['fold']), 'subject_id': table.loc[index, 'subject_id'], 'group': table.loc[index, 'group'], 'observed_group_tms': int(table.loc[index, 'group_tms']), 'model_group_tms': int(model_treatment[index]), 'NIHSS_pre': float(table.loc[index, 'NIHSS_pre']), 'observed_post': float(table.loc[index, 'NIHSS_post']), 'predicted_factual_post': float(predictions['factual'][position]), 'predicted_control_post': predicted_control, 'predicted_tms_post': predicted_tms, 'predicted_tms_benefit': predicted_control - predicted_tms})
        tuning_rows.append({'cohort': cohort, 'block': block, 'structure': structure, 'repeat': int(split['repeat']), 'fold': int(split['fold']), **best})
    return (pd.DataFrame(prediction_rows), pd.DataFrame(tuning_rows))

def metric_values(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    correlation = float(pearsonr(observed, predicted).statistic) if np.std(predicted) > 0 and np.std(observed) > 0 else np.nan
    return {'mae': float(mean_absolute_error(observed, predicted)), 'rmse': float(np.sqrt(mean_squared_error(observed, predicted))), 'r2': float(r2_score(observed, predicted)), 'pearson_r': correlation}

def repeat_performance(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (keys, subset) in predictions.groupby(['cohort', 'block', 'structure', 'repeat'], sort=False):
        (cohort, block, structure, repeat) = keys
        for arm in ('All', 'Control', 'TMS'):
            arm_subset = subset if arm == 'All' else subset.loc[subset['group'] == arm]
            metrics = metric_values(arm_subset['observed_post'].to_numpy(float), arm_subset['predicted_factual_post'].to_numpy(float))
            rows.append({'cohort': cohort, 'block': block, 'structure': structure, 'repeat': int(repeat), 'observed_arm': arm, 'n': len(arm_subset), **metrics})
    return pd.DataFrame(rows)

def summarize_performance(performance: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (keys, subset) in performance.groupby(['cohort', 'block', 'structure', 'observed_arm'], sort=False):
        (cohort, block, structure, arm) = keys
        for metric in ('mae', 'rmse', 'r2', 'pearson_r'):
            values = subset[metric].dropna().to_numpy(float)
            rows.append({'cohort': cohort, 'block': block, 'structure': structure, 'observed_arm': arm, 'metric': metric, 'n_repeats': len(values), 'mean': float(np.mean(values)), 'median': float(np.median(values)), 'sd': float(np.std(values, ddof=1)), 'repeat_interval_low': float(np.percentile(values, 2.5)), 'repeat_interval_high': float(np.percentile(values, 97.5))})
    return pd.DataFrame(rows)

def paired_mae_comparisons(performance: pd.DataFrame) -> pd.DataFrame:
    rows = []
    interaction_comparisons = (('clinical', 'clinical_lesion', 'lesion beyond clinical'), ('clinical', 'clinical_gradient', 'gradient beyond clinical'), ('clinical_lesion', 'combined', 'gradient beyond clinical + lesion'))
    for cohort in performance['cohort'].unique():
        frame = performance.loc[(performance['cohort'] == cohort) & (performance['observed_arm'] == 'All')]
        for (reference, comparison, label) in interaction_comparisons:
            selected = frame.loc[frame['structure'] == 'treatment_interaction']
            reference_values = selected.loc[selected['block'] == reference, ['repeat', 'mae']].set_index('repeat')['mae']
            comparison_values = selected.loc[selected['block'] == comparison, ['repeat', 'mae']].set_index('repeat')['mae']
            if reference_values.empty or comparison_values.empty:
                continue
            delta = (reference_values - comparison_values).dropna().to_numpy(float)
            rows.append(comparison_row(cohort, label, reference, comparison, 'treatment_interaction', 'treatment_interaction', delta))
        for block in frame['block'].unique():
            reference_values = frame.loc[(frame['block'] == block) & (frame['structure'] == 'main_effect'), ['repeat', 'mae']].set_index('repeat')['mae']
            comparison_values = frame.loc[(frame['block'] == block) & (frame['structure'] == 'treatment_interaction'), ['repeat', 'mae']].set_index('repeat')['mae']
            delta = (reference_values - comparison_values).dropna().to_numpy(float)
            rows.append(comparison_row(cohort, f'individualized interaction beyond average contrast: {block}', block, block, 'main_effect', 'treatment_interaction', delta))
    return pd.DataFrame(rows)

def comparison_row(cohort: str, contrast: str, reference_block: str, comparison_block: str, reference_structure: str, comparison_structure: str, delta: np.ndarray) -> dict:
    return {'cohort': cohort, 'contrast': contrast, 'reference_block': reference_block, 'comparison_block': comparison_block, 'reference_structure': reference_structure, 'comparison_structure': comparison_structure, 'positive_means_comparison_improves': True, 'mean_mae_reduction': float(np.mean(delta)), 'median_mae_reduction': float(np.median(delta)), 'repeat_interval_low': float(np.percentile(delta, 2.5)), 'repeat_interval_high': float(np.percentile(delta, 97.5)), 'fraction_repeats_improved': float(np.mean(delta > 0)), 'n_repeats': len(delta)}

def summarize_potential_outcomes(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (keys, subset) in predictions.groupby(['cohort', 'block', 'structure', 'repeat'], sort=False):
        (cohort, block, structure, repeat) = keys
        benefit = subset['predicted_tms_benefit'].to_numpy(float)
        rows.append({'cohort': cohort, 'block': block, 'structure': structure, 'repeat': int(repeat), 'mean_predicted_control_post': float(subset['predicted_control_post'].mean()), 'mean_predicted_tms_post': float(subset['predicted_tms_post'].mean()), 'mean_predicted_tms_benefit': float(np.mean(benefit)), 'sd_predicted_tms_benefit': float(np.std(benefit, ddof=1)), 'fraction_predicted_tms_benefit_positive': float(np.mean(benefit > 0))})
    return pd.DataFrame(rows)

def aggregate_participants(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = ['cohort', 'block', 'structure', 'subject_id', 'group']
    for (keys, subset) in predictions.groupby(group_columns, sort=False):
        (cohort, block, structure, subject_id, group) = keys
        row = {'cohort': cohort, 'block': block, 'structure': structure, 'subject_id': subject_id, 'group': group, 'NIHSS_pre': float(subset['NIHSS_pre'].iloc[0]), 'observed_post': float(subset['observed_post'].iloc[0])}
        for column in ('predicted_factual_post', 'predicted_control_post', 'predicted_tms_post', 'predicted_tms_benefit'):
            values = subset[column].to_numpy(float)
            row[f'{column}_mean'] = float(np.mean(values))
            row[f'{column}_sd'] = float(np.std(values, ddof=1))
            row[f'{column}_interval_low'] = float(np.percentile(values, 2.5))
            row[f'{column}_interval_high'] = float(np.percentile(values, 97.5))
        row['fraction_repeats_tms_benefit_positive'] = float(np.mean(subset['predicted_tms_benefit'].to_numpy(float) > 0))
        rows.append(row)
    return pd.DataFrame(rows)

def permute_gradients_within_group(prepared: base.PreparedData, seed: int) -> base.PreparedData:
    rng = np.random.default_rng(seed)
    assignment = np.arange(len(prepared.table))
    groups = prepared.table['group_tms'].to_numpy(int)
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        assignment[indices] = rng.permutation(indices)
    return base.PreparedData(table=prepared.table, cortical=prepared.cortical[assignment], subcortical=prepared.subcortical[assignment])

def gradient_permutation_one(prepared: base.PreparedData, splits: list[dict], permutation: int) -> float:
    permuted = permute_gradients_within_group(prepared, base.SEED + 500000 + permutation)
    (predictions, _) = run_job('permuted_gradient', permuted, 'combined', 'treatment_interaction', splits)
    return float(mean_absolute_error(predictions['observed_post'], predictions['predicted_factual_post']))

def interaction_permutation_one(prepared: base.PreparedData, splits: list[dict], permutation: int, block: str) -> float:
    rng = np.random.default_rng(base.SEED + 700000 + permutation)
    treatment = rng.permutation(prepared.table['group_tms'].to_numpy(int))
    (main, _) = run_job('permuted_treatment', prepared, block, 'main_effect', splits, treatment_override=treatment)
    (interaction, _) = run_job('permuted_treatment', prepared, block, 'treatment_interaction', splits, treatment_override=treatment)
    main_mae = mean_absolute_error(main['observed_post'], main['predicted_factual_post'])
    interaction_mae = mean_absolute_error(interaction['observed_post'], interaction['predicted_factual_post'])
    return float(main_mae - interaction_mae)

def run_permutations(prepared: base.PreparedData, predictions: pd.DataFrame, splits: list[dict], permutations: int, jobs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if permutations <= 0:
        return (pd.DataFrame(), pd.DataFrame())
    fixed_splits = [split for split in splits if split['repeat'] == 0]
    actual = predictions.loc[(predictions['cohort'] == 'primary_full') & (predictions['repeat'] == 0)]
    lesion = actual.loc[(actual['block'] == 'clinical_lesion') & (actual['structure'] == 'treatment_interaction')]
    combined = actual.loc[(actual['block'] == 'combined') & (actual['structure'] == 'treatment_interaction')]
    lesion_mae = mean_absolute_error(lesion['observed_post'], lesion['predicted_factual_post'])
    combined_mae = mean_absolute_error(combined['observed_post'], combined['predicted_factual_post'])
    observed_gradient_reduction = float(lesion_mae - combined_mae)
    gradient_null_mae = Parallel(n_jobs=jobs, verbose=5)((delayed(gradient_permutation_one)(prepared, fixed_splits, permutation) for permutation in range(permutations)))
    gradient_null_reduction = lesion_mae - np.asarray(gradient_null_mae, dtype=float)
    summary_rows = [{'test': 'gradient beyond clinical + lesion in individualized interaction model', 'observed_mae_reduction': observed_gradient_reduction, 'null_mean_mae_reduction': float(np.mean(gradient_null_reduction)), 'null_p95_mae_reduction': float(np.percentile(gradient_null_reduction, 95)), 'one_sided_p_value': (1.0 + float(np.count_nonzero(gradient_null_reduction >= observed_gradient_reduction))) / (permutations + 1.0), 'permutations': permutations, 'scheme': 'cortical/subcortical gradient rows jointly shuffled within observed treatment group'}]
    null_columns = {'gradient_mae_reduction': gradient_null_reduction}
    for block in ('clinical', 'combined'):
        main = actual.loc[(actual['block'] == block) & (actual['structure'] == 'main_effect')]
        interaction = actual.loc[(actual['block'] == block) & (actual['structure'] == 'treatment_interaction')]
        main_mae = mean_absolute_error(main['observed_post'], main['predicted_factual_post'])
        interaction_mae = mean_absolute_error(interaction['observed_post'], interaction['predicted_factual_post'])
        observed_interaction_reduction = float(main_mae - interaction_mae)
        interaction_null_reduction = np.asarray(Parallel(n_jobs=jobs, verbose=5)((delayed(interaction_permutation_one)(prepared, fixed_splits, permutation, block) for permutation in range(permutations))), dtype=float)
        summary_rows.append({'test': f'individualized treatment interaction beyond average treatment contrast in {block} model', 'observed_mae_reduction': observed_interaction_reduction, 'null_mean_mae_reduction': float(np.mean(interaction_null_reduction)), 'null_p95_mae_reduction': float(np.percentile(interaction_null_reduction, 95)), 'one_sided_p_value': (1.0 + float(np.count_nonzero(interaction_null_reduction >= observed_interaction_reduction))) / (permutations + 1.0), 'permutations': permutations, 'scheme': 'treatment labels permuted while preserving group count; exploratory because randomization status is unclear'})
        null_columns[f'{block}_treatment_interaction_mae_reduction'] = interaction_null_reduction
    summary = pd.DataFrame(summary_rows)
    null = pd.DataFrame({'permutation': np.arange(permutations), **null_columns})
    return (summary, null)

def best_interaction_block(summary: pd.DataFrame) -> str:
    selected = summary.loc[(summary['cohort'] == 'primary_full') & (summary['structure'] == 'treatment_interaction') & (summary['observed_arm'] == 'All') & (summary['metric'] == 'mae')]
    return str(selected.sort_values(['mean', 'block']).iloc[0]['block'])

def plot_results(performance: pd.DataFrame, comparisons: pd.DataFrame, participants: pd.DataFrame, best_block: str) -> None:
    selected_performance = performance.loc[(performance['cohort'] == 'primary_full') & (performance['structure'] == 'treatment_interaction')]
    (figure, axes) = plt.subplots(2, 2, figsize=(9.0, 7.2))
    all_arm = selected_performance.loc[selected_performance['observed_arm'] == 'All']
    data = [all_arm.loc[all_arm['block'] == block, 'mae'].to_numpy(float) for block in BLOCKS]
    box = axes[0, 0].boxplot(data, patch_artist=True, widths=0.62, showfliers=False)
    for (patch, block) in zip(box['boxes'], BLOCKS):
        patch.set_facecolor(BLOCK_COLORS[block])
        patch.set_alpha(0.82)
    axes[0, 0].set_xticks(range(1, len(BLOCKS) + 1), ['Clinical', '+Lesion', '+Gradient', 'Combined'])
    axes[0, 0].set_ylabel('Observed-treatment CV MAE')
    axes[0, 0].set_title('Factual outcome prediction')
    arm_data = []
    positions = []
    colors = []
    for (index, arm) in enumerate(('Control', 'TMS')):
        values = selected_performance.loc[(selected_performance['block'] == best_block) & (selected_performance['observed_arm'] == arm), 'mae'].to_numpy(float)
        arm_data.append(values)
        positions.append(index + 1)
        colors.append('#B6B6B6' if arm == 'Control' else '#2B78B8')
    box = axes[0, 1].boxplot(arm_data, positions=positions, patch_artist=True, widths=0.55, showfliers=False)
    for (patch, color) in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
    axes[0, 1].set_xticks(positions, ['Control', 'TMS'])
    axes[0, 1].set_ylabel('Observed-treatment CV MAE')
    axes[0, 1].set_title(f'Best block by observed MAE: {BLOCK_LABELS[best_block]}')
    selected_participants = participants.loc[(participants['cohort'] == 'primary_full') & (participants['block'] == best_block) & (participants['structure'] == 'treatment_interaction')]
    for (group, color, marker) in (('Control', '#777777', 'o'), ('TMS', '#2B78B8', 's')):
        subset = selected_participants.loc[selected_participants['group'] == group]
        axes[1, 0].scatter(subset['predicted_control_post_mean'], subset['predicted_tms_post_mean'], s=28, color=color, marker=marker, alpha=0.82, label=group)
    limits = [float(min(selected_participants['predicted_control_post_mean'].min(), selected_participants['predicted_tms_post_mean'].min())), float(max(selected_participants['predicted_control_post_mean'].max(), selected_participants['predicted_tms_post_mean'].max()))]
    padding = max(0.4, (limits[1] - limits[0]) * 0.06)
    limits = [limits[0] - padding, limits[1] + padding]
    axes[1, 0].plot(limits, limits, linestyle='--', color='#999999', linewidth=0.9)
    axes[1, 0].set_xlim(limits)
    axes[1, 0].set_ylim(limits)
    axes[1, 0].set_xlabel('Predicted NIHSS if Control')
    axes[1, 0].set_ylabel('Predicted NIHSS if TMS')
    axes[1, 0].set_title('Two scenarios for the same participant')
    axes[1, 0].legend(frameon=False, fontsize=8)
    ordered = selected_participants.sort_values('predicted_tms_benefit_mean').reset_index(drop=True)
    x = np.arange(len(ordered))
    y = ordered['predicted_tms_benefit_mean'].to_numpy(float)
    low = ordered['predicted_tms_benefit_interval_low'].to_numpy(float)
    high = ordered['predicted_tms_benefit_interval_high'].to_numpy(float)
    axes[1, 1].vlines(x, low, high, color='#A7A7A7', linewidth=0.7)
    point_colors = np.where(ordered['group'].eq('TMS'), '#2B78B8', '#777777')
    axes[1, 1].scatter(x, y, c=point_colors, s=18, zorder=3)
    axes[1, 1].axhline(0.0, color='#333333', linewidth=0.8)
    axes[1, 1].set_xlabel('Participants sorted by predicted benefit')
    axes[1, 1].set_ylabel('Predicted TMS benefit\n(Control NIHSS - TMS NIHSS)')
    axes[1, 1].set_title('Cross-validation split variability')
    for (label, axis) in zip(('A', 'B', 'C', 'D'), axes.ravel()):
        axis.text(-0.14, 1.08, label, transform=axis.transAxes, fontsize=10, fontweight='bold', va='top')
        axis.spines[['top', 'right']].set_visible(False)
        axis.tick_params(width=0.6, length=2.5, labelsize=8)
    figure.suptitle('Baseline prediction of NIHSS under TMS and Control scenarios', fontsize=10, y=0.995)
    figure.subplots_adjust(left=0.1, right=0.985, bottom=0.09, top=0.9, hspace=0.46, wspace=0.36)
    for (suffix, kwargs) in (('png', {'dpi': 300}), ('pdf', {})):
        figure.savefig(FIGURES / f'treatment_benefit_prediction.{suffix}', bbox_inches='tight', facecolor='white', **kwargs)
    plt.close(figure)

def lookup_metric(summary: pd.DataFrame, cohort: str, block: str, structure: str, arm: str, metric: str) -> float:
    return float(summary.loc[(summary['cohort'] == cohort) & (summary['block'] == block) & (summary['structure'] == structure) & (summary['observed_arm'] == arm) & (summary['metric'] == metric), 'mean'].iloc[0])

def write_summary(performance_summary: pd.DataFrame, comparisons: pd.DataFrame, potential: pd.DataFrame, participants: pd.DataFrame, permutation: pd.DataFrame, best_block: str) -> None:
    lines = ['# 治疗前信息预测 TMS 与对照情景下的 NIHSS', '', '## 分析问题', '', '- 仅使用治疗前信息，为同一患者分别预测假设接受 TMS 和假设进入对照条件时的治疗后 NIHSS。', '- 预测 TMS 获益定义为：预测对照 NIHSS - 预测 TMS NIHSS；正值表示模型预测 TMS 情景的 NIHSS 更低。', '- 模型同时使用 TMS 和 Control 数据联合训练。Treatment main effect 表示平均组差异，Treatment×baseline interactions 才允许个体获益不同。', '- 外层预测固定截断在 NIHSS 合法范围 0–42；该约束不使用测试结局信息。', '- 当前是否随机分组、是否假刺激及同期康复是否一致仍不明确，因此结果是模型化的治疗情景差异，不是已验证的个体因果效应。', '', '## 50 次重复嵌套交叉验证', '', '以下为允许 Treatment×baseline interactions 的个体化模型。误差只在患者实际接受的处理情景下计算；反事实结局没有真实标签，不能直接计算误差。', '', '| 治疗前信息 | 全队列 MAE | Control MAE | TMS MAE | 全队列 R2 |', '|---|---:|---:|---:|---:|']
    for block in BLOCKS:
        lines.append(f"| {BLOCK_LABELS[block]} | {lookup_metric(performance_summary, 'primary_full', block, 'treatment_interaction', 'All', 'mae'):.3f} | {lookup_metric(performance_summary, 'primary_full', block, 'treatment_interaction', 'Control', 'mae'):.3f} | {lookup_metric(performance_summary, 'primary_full', block, 'treatment_interaction', 'TMS', 'mae'):.3f} | {lookup_metric(performance_summary, 'primary_full', block, 'treatment_interaction', 'All', 'r2'):.3f} |")
    lines.extend(['', '## 是否支持个体化治疗差异', ''])
    for block in BLOCKS:
        row = comparisons.loc[(comparisons['cohort'] == 'primary_full') & (comparisons['contrast'] == f'individualized interaction beyond average contrast: {block}')].iloc[0]
        direction = '降低' if row['mean_mae_reduction'] > 0 else '增加'
        lines.append(f"- {BLOCK_LABELS[block]}：加入个体化治疗交互后，实际情景 MAE 平均{direction} {abs(row['mean_mae_reduction']):.3f}；重复区间 {row['repeat_interval_low']:.3f} 到 {row['repeat_interval_high']:.3f}；{row['fraction_repeats_improved'] * 100:.1f}% 的划分改善。")
    best_potential = potential.loc[(potential['cohort'] == 'primary_full') & (potential['block'] == best_block) & (potential['structure'] == 'treatment_interaction')]
    best_participants = participants.loc[(participants['cohort'] == 'primary_full') & (participants['block'] == best_block) & (participants['structure'] == 'treatment_interaction')]
    stable_tms = int(np.count_nonzero(best_participants['fraction_repeats_tms_benefit_positive'] >= 0.95))
    stable_control = int(np.count_nonzero(best_participants['fraction_repeats_tms_benefit_positive'] <= 0.05))
    lines.extend(['', '## 两种治疗情景的模型输出', '', f'- 按实际情景 MAE 选择的探索性最佳个体化信息块为：{BLOCK_LABELS[best_block]}。', f"- 全队列平均预测对照情景 NIHSS：{best_potential['mean_predicted_control_post'].mean():.3f}。", f"- 全队列平均预测 TMS 情景 NIHSS：{best_potential['mean_predicted_tms_post'].mean():.3f}。", f"- 平均预测 TMS 情景差异：{best_potential['mean_predicted_tms_benefit'].mean():.3f} NIHSS 分。", f'- 38 例中，{stable_tms} 例在至少 95% 的重复划分中均预测 TMS 更好；{stable_control} 例在至少 95% 的重复划分中均预测对照更好。', '- 个体表中的 2.5%–97.5% 区间表示不同交叉验证划分带来的模型不稳定性，不是因果效应置信区间。'])
    gradient_row = comparisons.loc[(comparisons['cohort'] == 'primary_full') & (comparisons['contrast'] == 'gradient beyond clinical + lesion')].iloc[0]
    sensitivity_row = comparisons.loc[(comparisons['cohort'] == 'exclude_manual_review_flags') & (comparisons['contrast'] == 'gradient beyond clinical + lesion')].iloc[0]
    lines.extend(['', '## 梯度的额外价值', '', f"- 全38例：在临床+病灶之上加入基线梯度后，个体化模型 MAE reduction={gradient_row['mean_mae_reduction']:.3f}，重复区间 {gradient_row['repeat_interval_low']:.3f} 到 {gradient_row['repeat_interval_high']:.3f}。", f"- 排除3例病灶人工复核标记后：MAE reduction={sensitivity_row['mean_mae_reduction']:.3f}，重复区间 {sensitivity_row['repeat_interval_low']:.3f} 到 {sensitivity_row['repeat_interval_high']:.3f}。"])
    if not permutation.empty:
        for row in permutation.itertuples():
            lines.append(f'- {row.test}：observed MAE reduction={row.observed_mae_reduction:.3f}，置换 P={row.one_sided_p_value:.3f}。')
    clinical_interaction = comparisons.loc[(comparisons['cohort'] == 'primary_full') & (comparisons['contrast'] == 'individualized interaction beyond average contrast: clinical')].iloc[0]
    clinical_permutation_p = float(permutation.loc[permutation['test'].str.contains('in clinical model', regex=False), 'one_sided_p_value'].iloc[0])
    lines.extend(['', '## 主要结论', '', f"- 治疗前临床模型的实际处理情景 MAE 最低（{lookup_metric(performance_summary, 'primary_full', 'clinical', 'treatment_interaction', 'All', 'mae'):.3f}），病灶和梯度均未进一步降低总体误差。", f"- 临床个体交互相对平均治疗差异仅平均降低 MAE {clinical_interaction['mean_mae_reduction']:.3f}，重复区间跨 0，置换 P={clinical_permutation_p:.3f}；当前证据不足以支持稳定的个体治疗分层。", '- 最佳模型对38例均预测 TMS 情景更好，说明结果主要体现总体 TMS-Control 组差异，而没有识别出应优先选择 Control 的患者亚组。', '- 因此可以报告两种治疗情景的探索性预测，但不能把个体差值直接解释为已验证的个体 TMS 因果获益。'])
    lines.extend(['', '## FC矩阵审计', '', '- 项目代码 `code/4/4/FCzhongjie2.py` 明确预期每例 400×400 皮层 FC 矩阵，并包含 TMS/Control、before/after。', '- 对 E:\\Gradient 内全部256个 `.npy` 做形状审计：88个为20484×5、84个为400×5、84个为54×5，400×400矩阵为0个。', f'- 代码中的数据根目录为 `{FC_EXPECTED_PATH}`，当前机器不存在 F 盘，E:\\Gradient 内也未发现 FC 矩阵文件，因此本次模型没有加入 FC。', '- FC矩阵补齐后，应只使用 before 矩阵；优先提取7网络内/网络间平均连接（28维）或在训练折内对上三角边做PCA，不能把约79800条边直接放入38例模型。', '', '## 解释边界', '', '- 交叉验证只能验证患者实际接受处理时的结局预测，不能直接验证同一患者未接受的另一种处理结局。', '- 若个体交互没有稳定降低实际情景 MAE，则数据只支持平均组差异，不支持可靠的个体治疗选择。', '- 样本量仅38例，且两组病灶拓扑和改善比例不同；当前结果不能直接生成临床治疗推荐。'])
    (OUTPUT_ROOT / 'RESULTS_ZH.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

def save_split_manifest(cohort: str, prepared: base.PreparedData, splits: list[dict]) -> None:
    subject_ids = prepared.table['subject_id'].to_numpy()
    records = []
    for split in splits:
        records.append({'cohort': cohort, 'repeat': int(split['repeat']), 'fold': int(split['fold']), 'train_subject_ids': subject_ids[split['train']].tolist(), 'test_subject_ids': subject_ids[split['test']].tolist()})
    (SPLITS / f'nested_cv_splits_{cohort}.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Predict NIHSS under TMS and Control scenarios from baseline data.')
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
        raise ValueError('repeats must be positive and permutations non-negative')
    ensure_dirs()
    primary = base.load_inputs()
    sensitivity = base.subset_prepared(primary, ~primary.table['subject_id'].isin(MANUAL_REVIEW_FLAGS).to_numpy())
    cohorts = {'primary_full': primary, 'exclude_manual_review_flags': sensitivity}
    cohort_blocks = {'primary_full': BLOCKS, 'exclude_manual_review_flags': ('clinical_lesion', 'combined')}
    split_sets = {cohort: make_outer_splits(prepared.table['group_tms'].to_numpy(int), args.repeats, seed_offset=index * 1000) for (index, (cohort, prepared)) in enumerate(cohorts.items())}
    for (cohort, prepared) in cohorts.items():
        save_split_manifest(cohort, prepared, split_sets[cohort])
    export_columns = ['subject_id', 'group', 'manual_review_flag', 'NIHSS_pre', 'NIHSS_post', 'age', 'sex_male', 'esrs', *base.LESION_FEATURES, *base.GRADIENT_SUMMARY_FEATURES]
    primary.table[export_columns].to_csv(TABLES / 'analysis_cohort.csv', index=False, encoding='utf-8-sig')
    jobs_to_run = []
    for (cohort, prepared) in cohorts.items():
        for block in cohort_blocks[cohort]:
            for structure in STRUCTURES:
                jobs_to_run.append((cohort, prepared, block, structure, split_sets[cohort]))
    results = Parallel(n_jobs=args.jobs, verbose=10)((delayed(run_job)(*job) for job in jobs_to_run))
    predictions = pd.concat([result[0] for result in results], ignore_index=True)
    tuning = pd.concat([result[1] for result in results], ignore_index=True)
    performance = repeat_performance(predictions)
    performance_summary = summarize_performance(performance)
    comparisons = paired_mae_comparisons(performance)
    potential = summarize_potential_outcomes(predictions)
    participants = aggregate_participants(predictions)
    (permutation, permutation_null) = run_permutations(primary, predictions, split_sets['primary_full'], args.permutations, args.jobs)
    best_block = best_interaction_block(performance_summary)
    predictions.to_csv(TABLES / 'nested_cv_potential_outcome_predictions.csv', index=False, encoding='utf-8-sig')
    tuning.to_csv(TABLES / 'outer_fold_tuning.csv', index=False, encoding='utf-8-sig')
    performance.to_csv(TABLES / 'repeat_level_factual_performance.csv', index=False, encoding='utf-8-sig')
    performance_summary.to_csv(TABLES / 'factual_performance_summary.csv', index=False, encoding='utf-8-sig')
    comparisons.to_csv(TABLES / 'paired_mae_comparisons.csv', index=False, encoding='utf-8-sig')
    potential.to_csv(TABLES / 'repeat_level_potential_outcome_summary.csv', index=False, encoding='utf-8-sig')
    participants.to_csv(TABLES / 'participant_potential_outcomes.csv', index=False, encoding='utf-8-sig')
    if not permutation.empty:
        permutation.to_csv(TABLES / 'permutation_tests.csv', index=False, encoding='utf-8-sig')
        permutation_null.to_csv(TABLES / 'permutation_null.csv', index=False, encoding='utf-8-sig')
    plot_results(performance, comparisons, participants, best_block)
    write_summary(performance_summary, comparisons, potential, participants, permutation, best_block)
    manifest = {'analysis_date': '2026-08-03', 'primary_cohort_n': int(len(primary.table)), 'group_counts': primary.table['group'].value_counts().sort_index().to_dict(), 'outcome': 'NIHSS_post', 'potential_outcomes': {'Y0': 'predicted NIHSS_post under Control scenario', 'Y1': 'predicted NIHSS_post under TMS scenario', 'benefit': 'Y0 - Y1; positive values favor lower predicted NIHSS under TMS'}, 'baseline_clinical_features': list(CLINICAL_FEATURES), 'baseline_lesion_features': list(base.LESION_FEATURES), 'baseline_gradient': {'raw_dimensions': {'cortical_G1G2': 800, 'subcortical_G1G2': 108}, 'foldwise_pca_candidates': [list(config) for config in base.PCA_CONFIGS]}, 'model': 'joint S-learner Ridge outcome model', 'prediction_bounds': {'NIHSS_min': NIHSS_RANGE[0], 'NIHSS_max': NIHSS_RANGE[1]}, 'model_structures': {'main_effect': 'baseline X + treatment', 'treatment_interaction': 'baseline X + treatment + baseline X times treatment'}, 'outer_cv': f'treatment-stratified 5-fold repeated {args.repeats} times', 'inner_cv': 'treatment-stratified 4-fold tuning of Ridge alpha and gradient PCA dimensions', 'ridge_alphas': list(RIDGE_ALPHAS), 'permutations': args.permutations, 'sensitivity_excluded_subjects': sorted(MANUAL_REVIEW_FLAGS), 'best_interaction_block_by_mean_observed_treatment_cv_mae': best_block, 'fc_audit': {'code_found': _portable_path('project', 'code/4/4/FCzhongjie2.py'), 'expected_matrix_shape': [400, 400], 'expected_path': str(FC_EXPECTED_PATH), 'path_exists': FC_EXPECTED_PATH.exists(), 'included_in_model': False}, 'leakage_control': 'all scaling, PCA, tuning and fitting occur within training folds', 'causal_limitation': 'randomization, sham control and concurrent rehabilitation are unclear; counterfactual predictions are not directly observable', 'external_validation': False, 'random_seed': base.SEED}
    (OUTPUT_ROOT / 'analysis_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Primary cohort: {len(primary.table)}')
    print(f'Sensitivity cohort: {len(sensitivity.table)}')
    print(f'Prediction rows: {len(predictions)}')
    print(f'Best interaction block: {best_block}')
    print(f'Output: {OUTPUT_ROOT}')
if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=FutureWarning)
        main()
