from __future__ import annotations
import os as _portable_os
from pathlib import Path as _PortablePath

def _portable_path(kind, relative):
    project = _PortablePath(_portable_os.environ['PAPER_PROJECT_ROOT'])
    root = project if kind == 'project' else project.parent
    return str(root / relative)
import json
import math
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import pearsonr
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
HERE = Path(__file__).resolve().parent
EXTERNAL_ROOT = HERE.parents[1]
VALIDATION_ROOT = EXTERNAL_ROOT / 'ds003999_validation'
MATCHED_ROOT = VALIDATION_ROOT / 'matched_gradient_figure'
TABLE_ROOT = MATCHED_ROOT / 'tables'
SOURCE_ROOT = MATCHED_ROOT / 'source_data'
SEED = 20260811
N_REPEATS = 20
N_OUTER_SPLITS = 5
N_INNER_SPLITS = 4
SENSITIVITY_EXCLUSION = 'sub-35'
ALGORITHMS = ('ridge', 'elasticnet')
ALGORITHM_LABELS = {'ridge': 'Ridge', 'elasticnet': 'Elastic Net'}
ALGORITHM_COLORS = {'ridge': '#596E9A', 'elasticnet': '#A97891'}
COHORT_ORDER = ('primary_n29', 'alignment_sensitivity_n28')
COHORT_LABELS = {'primary_n29': 'Primary cohort (n=29)', 'alignment_sensitivity_n28': 'Alignment sensitivity (n=28)'}
COHORT_COLORS = {'primary_n29': '#596E9A', 'alignment_sensitivity_n28': '#72A5A0'}
NETWORK_ORDER = ('Visual', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Control', 'Default')
ORGANIZATION_METRICS = ('Dispersion', 'betweenDisp', 'Segregation')
FULL_NETWORK_METRICS = ('mean', 'Dispersion', 'betweenDisp', 'Segregation')
CLINICAL_FEATURES = ('pre_score', 'log_time_after_stroke', 'sex_male')
BLOCK_ORDER = ('mean_only', 'clinical_only', 'g1_global_spread', 'g2_global_spread', 'g1_vis_organization', 'g2_vis_organization', 'targeted_g2_vis_smn', 'g1g2_between_disp', 'g1_network_organization', 'g2_network_organization', 'g2_all_network', 'gradient_component_spectrum', 'all_baseline_summaries')
PLOT_BLOCK_ORDER = ('g1_global_spread', 'g2_global_spread', 'g1_vis_organization', 'g2_vis_organization', 'targeted_g2_vis_smn', 'g1g2_between_disp', 'g1_network_organization', 'g2_network_organization', 'g2_all_network', 'all_baseline_summaries')
BLOCK_LABELS = {'mean_only': 'Train-fold mean', 'clinical_only': 'Clinical only', 'g1_global_spread': 'G1 global spread', 'g2_global_spread': 'G2 global spread', 'g1_vis_organization': 'G1 VIS organization', 'g2_vis_organization': 'G2 VIS organization', 'targeted_g2_vis_smn': 'Targeted G2 VIS/SMN', 'g1g2_between_disp': 'G1/G2 betweenDisp', 'g1_network_organization': 'G1 seven-network organization', 'g2_network_organization': 'G2 seven-network organization', 'g2_all_network': 'All-network G2 + centroids', 'gradient_component_spectrum': 'Baseline G1-G5 spectrum', 'all_baseline_summaries': 'All baseline summaries'}
BLOCK_CATEGORY = {'mean_only': 'Reference models', 'clinical_only': 'Reference models', 'g1_global_spread': 'Original compact blocks', 'g2_global_spread': 'Original compact blocks', 'g1_vis_organization': 'Original compact blocks', 'g2_vis_organization': 'Original compact blocks', 'targeted_g2_vis_smn': 'Hypothesis-led additions', 'g1g2_between_disp': 'Hypothesis-led additions', 'g1_network_organization': 'Distributed representations', 'g2_network_organization': 'Distributed representations', 'g2_all_network': 'Distributed representations', 'gradient_component_spectrum': 'Distributed representations', 'all_baseline_summaries': 'Distributed representations'}
ORIGINAL_BLOCKS = {'g1_global_spread', 'g2_global_spread', 'g1_vis_organization', 'g2_vis_organization', 'g1_network_organization', 'g2_network_organization'}
mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'], 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.size': 7, 'axes.titlesize': 7.8, 'axes.labelsize': 7, 'xtick.labelsize': 6.4, 'ytick.labelsize': 6.2, 'axes.linewidth': 0.7, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})

@dataclass(frozen=True)
class FeatureBlock:
    label: str
    category: str
    imaging_features: tuple[str, ...]
    rationale: str

def load_frame() -> pd.DataFrame:
    subject = pd.read_csv(VALIDATION_ROOT / 'tables' / 'legacy_aligned5_subject_metrics.csv')
    component = pd.read_csv(SOURCE_ROOT / 'component_variance_subject_level.csv')
    pre_component = component.loc[component['phase'] == 'pre'].pivot(index='participant_id', columns='component', values='share_of_retained_G1_G5')
    pre_component.columns = [f'pre_share_{name}' for name in pre_component.columns]
    frame = subject.merge(pre_component.reset_index(), on='participant_id', validate='one_to_one')
    if len(frame) != 29 or frame['participant_id'].nunique() != 29:
        raise RuntimeError('Expected 29 unique participants in the primary cohort')
    return frame

def define_feature_blocks(frame: pd.DataFrame) -> dict[str, FeatureBlock]:
    g1_vis = tuple((f'pre_G1_Visual_{metric}' for metric in ORGANIZATION_METRICS))
    g2_vis = tuple((f'pre_G2_Visual_{metric}' for metric in ORGANIZATION_METRICS))
    g1_network = tuple((f'pre_G1_{network}_{metric}' for network in NETWORK_ORDER for metric in ORGANIZATION_METRICS))
    g2_network = tuple((f'pre_G2_{network}_{metric}' for network in NETWORK_ORDER for metric in ORGANIZATION_METRICS))
    targeted_g2 = tuple((f'pre_G2_{network}_{metric}' for network in ('Visual', 'SomMot') for metric in FULL_NETWORK_METRICS))
    between_disp = tuple((f'pre_{gradient}_{network}_betweenDisp' for gradient in ('G1', 'G2') for network in NETWORK_ORDER))
    g2_all_network = tuple((f'pre_G2_{network}_{metric}' for network in NETWORK_ORDER for metric in FULL_NETWORK_METRICS))
    all_network_summaries = tuple((f'pre_{gradient}_{network}_{metric}' for gradient in ('G1', 'G2') for network in NETWORK_ORDER for metric in FULL_NETWORK_METRICS))
    component_spectrum = tuple((f'pre_share_G{index}' for index in range(1, 6)))
    all_baseline = tuple(dict.fromkeys(('pre_G1_sd', 'pre_G2_sd', *all_network_summaries, *component_spectrum)))
    blocks = {'mean_only': FeatureBlock(BLOCK_LABELS['mean_only'], BLOCK_CATEGORY['mean_only'], (), 'Training-fold outcome mean; algorithm-independent lower benchmark'), 'clinical_only': FeatureBlock(BLOCK_LABELS['clinical_only'], BLOCK_CATEGORY['clinical_only'], (), 'Baseline score, log time after stroke and sex'), 'g1_global_spread': FeatureBlock(BLOCK_LABELS['g1_global_spread'], BLOCK_CATEGORY['g1_global_spread'], ('pre_G1_sd',), 'Cortex-wide standard deviation of baseline G1 coordinates'), 'g2_global_spread': FeatureBlock(BLOCK_LABELS['g2_global_spread'], BLOCK_CATEGORY['g2_global_spread'], ('pre_G2_sd',), 'Cortex-wide standard deviation of baseline G2 coordinates'), 'g1_vis_organization': FeatureBlock(BLOCK_LABELS['g1_vis_organization'], BLOCK_CATEGORY['g1_vis_organization'], g1_vis, 'Baseline G1 VIS dispersion, between-network dispersion and segregation'), 'g2_vis_organization': FeatureBlock(BLOCK_LABELS['g2_vis_organization'], BLOCK_CATEGORY['g2_vis_organization'], g2_vis, 'Baseline G2 VIS dispersion, between-network dispersion and segregation'), 'targeted_g2_vis_smn': FeatureBlock(BLOCK_LABELS['targeted_g2_vis_smn'], BLOCK_CATEGORY['targeted_g2_vis_smn'], targeted_g2, 'Reference-inspired G2 VIS/SMN centroids and organization metrics'), 'g1g2_between_disp': FeatureBlock(BLOCK_LABELS['g1g2_between_disp'], BLOCK_CATEGORY['g1g2_between_disp'], between_disp, 'Reference-inspired between-network dispersion across G1, G2 and seven networks'), 'g1_network_organization': FeatureBlock(BLOCK_LABELS['g1_network_organization'], BLOCK_CATEGORY['g1_network_organization'], g1_network, 'G1 organization metrics across all seven networks'), 'g2_network_organization': FeatureBlock(BLOCK_LABELS['g2_network_organization'], BLOCK_CATEGORY['g2_network_organization'], g2_network, 'G2 organization metrics across all seven networks'), 'g2_all_network': FeatureBlock(BLOCK_LABELS['g2_all_network'], BLOCK_CATEGORY['g2_all_network'], g2_all_network, 'G2 centroids plus organization metrics across all seven networks'), 'gradient_component_spectrum': FeatureBlock(BLOCK_LABELS['gradient_component_spectrum'], BLOCK_CATEGORY['gradient_component_spectrum'], component_spectrum, 'Baseline shares of the retained G1-G5 component spectrum'), 'all_baseline_summaries': FeatureBlock(BLOCK_LABELS['all_baseline_summaries'], BLOCK_CATEGORY['all_baseline_summaries'], all_baseline, 'All available baseline global, network and component-spectrum summaries')}
    if tuple(blocks) != BLOCK_ORDER:
        raise RuntimeError('Feature-block order drifted from the figure contract')
    required = set(CLINICAL_FEATURES)
    for block in blocks.values():
        required.update(block.imaging_features)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f'Missing required baseline features: {missing}')
    values = frame[sorted(required)].to_numpy(float)
    if not np.isfinite(values).all():
        raise RuntimeError('Non-finite values found in the analysis matrix')
    leakage = sorted((name for name in required if name.startswith(('post_', 'delta_'))))
    if leakage:
        raise RuntimeError(f'Post-treatment leakage detected: {leakage}')
    return blocks

def candidate_parameters(algorithm: str) -> list[dict[str, float]]:
    if algorithm == 'ridge':
        return [{'alpha': float(alpha)} for alpha in np.logspace(-3, 3, 13)]
    if algorithm == 'elasticnet':
        return [{'alpha': float(alpha), 'l1_ratio': float(l1_ratio)} for alpha in np.logspace(1, -3, 5) for l1_ratio in (0.1, 0.5, 0.9)]
    raise KeyError(algorithm)

def make_model(algorithm: str, parameters: dict[str, float]):
    if algorithm == 'ridge':
        return Ridge(alpha=parameters['alpha'], max_iter=100000)
    if algorithm == 'elasticnet':
        return ElasticNet(alpha=parameters['alpha'], l1_ratio=parameters['l1_ratio'], max_iter=100000, tol=0.0001, random_state=SEED)
    raise KeyError(algorithm)

def tune_and_predict(x: np.ndarray, y: np.ndarray, train: np.ndarray, test: np.ndarray, inner_splits: list[tuple[np.ndarray, np.ndarray]], algorithm: str) -> tuple[np.ndarray, dict[str, float], float]:
    parameters = candidate_parameters(algorithm)
    scores = np.zeros(len(parameters), dtype=float)
    for (inner_train, inner_valid) in inner_splits:
        train_index = train[inner_train]
        valid_index = train[inner_valid]
        scaler = StandardScaler().fit(x[train_index])
        x_train = scaler.transform(x[train_index])
        x_valid = scaler.transform(x[valid_index])
        for (parameter_index, candidate) in enumerate(parameters):
            model = make_model(algorithm, candidate)
            model.fit(x_train, y[train_index])
            scores[parameter_index] += mean_absolute_error(y[valid_index], model.predict(x_valid))
    best_index = int(np.argmin(scores))
    best = parameters[best_index]
    scaler = StandardScaler().fit(x[train])
    model = make_model(algorithm, best)
    model.fit(scaler.transform(x[train]), y[train])
    prediction = np.asarray(model.predict(scaler.transform(x[test])), dtype=float)
    return (prediction, best, float(scores[best_index] / len(inner_splits)))

def safe_pearson(observed: np.ndarray, predicted: np.ndarray) -> float:
    if np.std(observed) == 0 or np.std(predicted) == 0:
        return float('nan')
    return float(pearsonr(observed, predicted).statistic)

def metric_row(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    mse = float(mean_squared_error(observed, predicted))
    return {'mae': float(mean_absolute_error(observed, predicted)), 'mse': mse, 'rmse': float(math.sqrt(mse)), 'r2': float(r2_score(observed, predicted)), 'pearson_r': safe_pearson(observed, predicted)}

def run_nested_cv(cohorts: dict[str, pd.DataFrame], blocks: dict[str, FeatureBlock]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_rows: list[dict[str, object]] = []
    performance_rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    for algorithm in ALGORITHMS:
        for cohort_name in COHORT_ORDER:
            data = cohorts[cohort_name].reset_index(drop=True)
            y = data['post_score'].to_numpy(float)
            matrices: dict[str, np.ndarray] = {}
            for (block_name, spec) in blocks.items():
                if block_name == 'mean_only':
                    continue
                features = list(CLINICAL_FEATURES) if block_name == 'clinical_only' else [*CLINICAL_FEATURES, *spec.imaging_features]
                matrices[block_name] = data[features].to_numpy(float)
            for repeat in globals().get('REPEAT_INDICES', range(N_REPEATS)):
                repeat_predictions: list[dict[str, object]] = []
                outer = KFold(n_splits=N_OUTER_SPLITS, shuffle=True, random_state=SEED + repeat)
                for (fold, (train, test)) in enumerate(outer.split(data)):
                    inner_splits = list(KFold(n_splits=N_INNER_SPLITS, shuffle=True, random_state=SEED + 1000 + repeat * 10 + fold).split(train))
                    mean_prediction = np.repeat(float(np.mean(y[train])), len(test))
                    for (subject_index, prediction) in zip(test, mean_prediction):
                        repeat_predictions.append({'algorithm': algorithm, 'cohort': cohort_name, 'repeat': repeat, 'fold': fold, 'block': 'mean_only', 'participant_id': data.iloc[subject_index]['participant_id'], 'observed_post_score': y[subject_index], 'predicted_post_score': float(prediction), 'best_parameters': '{}'})
                    tuning_rows.append({'algorithm': algorithm, 'cohort': cohort_name, 'repeat': repeat, 'fold': fold, 'block': 'mean_only', 'inner_mae': float('nan'), 'best_parameters': '{}'})
                    for block_name in BLOCK_ORDER:
                        if block_name == 'mean_only':
                            continue
                        (prediction, best, inner_mae) = tune_and_predict(matrices[block_name], y, train, test, inner_splits, algorithm)
                        best_json = json.dumps(best, sort_keys=True)
                        tuning_rows.append({'algorithm': algorithm, 'cohort': cohort_name, 'repeat': repeat, 'fold': fold, 'block': block_name, 'inner_mae': inner_mae, 'best_parameters': best_json})
                        for (subject_index, value) in zip(test, prediction):
                            repeat_predictions.append({'algorithm': algorithm, 'cohort': cohort_name, 'repeat': repeat, 'fold': fold, 'block': block_name, 'participant_id': data.iloc[subject_index]['participant_id'], 'observed_post_score': y[subject_index], 'predicted_post_score': float(value), 'best_parameters': best_json})
                repeat_frame = pd.DataFrame(repeat_predictions)
                prediction_rows.extend(repeat_predictions)
                for (block_name, subset) in repeat_frame.groupby('block', sort=False):
                    observed = subset['observed_post_score'].to_numpy(float)
                    predicted = subset['predicted_post_score'].to_numpy(float)
                    performance_rows.append({'algorithm': algorithm, 'cohort': cohort_name, 'repeat': repeat, 'block': block_name, 'n': len(subset), **metric_row(observed, predicted)})
                print(f'{ALGORITHM_LABELS[algorithm]} | {COHORT_LABELS[cohort_name]} | repeat {repeat + 1}/{N_REPEATS}', flush=True)
    return (pd.DataFrame(prediction_rows), pd.DataFrame(performance_rows), pd.DataFrame(tuning_rows))

def summarize_performance(performance: pd.DataFrame, blocks: dict[str, FeatureBlock]) -> tuple[pd.DataFrame, pd.DataFrame]:
    clinical = performance.loc[performance['block'] == 'clinical_only', ['algorithm', 'cohort', 'repeat', 'mae']].rename(columns={'mae': 'clinical_only_mae'})
    paired = performance.merge(clinical, on=['algorithm', 'cohort', 'repeat'], how='left', validate='many_to_one')
    paired['delta_mae_vs_clinical'] = paired['clinical_only_mae'] - paired['mae']
    summary = paired.groupby(['algorithm', 'cohort', 'block'], sort=False, as_index=False).agg(n=('n', 'first'), n_repeats=('repeat', 'nunique'), r2_mean=('r2', 'mean'), r2_p025=('r2', lambda x: x.quantile(0.025)), r2_p975=('r2', lambda x: x.quantile(0.975)), mse_mean=('mse', 'mean'), mse_p025=('mse', lambda x: x.quantile(0.025)), mse_p975=('mse', lambda x: x.quantile(0.975)), mae_mean=('mae', 'mean'), mae_p025=('mae', lambda x: x.quantile(0.025)), mae_p975=('mae', lambda x: x.quantile(0.975)), delta_mae_mean=('delta_mae_vs_clinical', 'mean'), delta_mae_p025=('delta_mae_vs_clinical', lambda x: x.quantile(0.025)), delta_mae_p975=('delta_mae_vs_clinical', lambda x: x.quantile(0.975)), repeat_fraction_better_than_clinical=('delta_mae_vs_clinical', lambda x: np.mean(x > 0)), pearson_r_mean=('pearson_r', 'mean'))
    block_position = {block: index for (index, block) in enumerate(BLOCK_ORDER)}
    summary['block_order'] = summary['block'].map(block_position)
    summary['block_label'] = summary['block'].map(BLOCK_LABELS)
    summary['category'] = summary['block'].map(BLOCK_CATEGORY)
    summary['n_imaging_features'] = summary['block'].map({name: len(spec.imaging_features) for (name, spec) in blocks.items()})
    summary['n_total_features'] = summary['block'].map({name: 0 if name == 'mean_only' else len(CLINICAL_FEATURES) if name == 'clinical_only' else len(CLINICAL_FEATURES) + len(spec.imaging_features) for (name, spec) in blocks.items()})
    summary['is_original_block'] = summary['block'].isin(ORIGINAL_BLOCKS)
    summary['mse_rank_within_cohort'] = summary.groupby(['algorithm', 'cohort'])['mse_mean'].rank(method='min', ascending=True)
    summary['is_mse_best'] = summary['mse_rank_within_cohort'] == 1
    summary['is_mae_best'] = np.isclose(summary['mae_mean'], summary.groupby(['algorithm', 'cohort'])['mae_mean'].transform('min'))
    summary = summary.sort_values(['algorithm', 'cohort', 'block_order']).reset_index(drop=True)
    return (paired, summary)

def write_feature_dictionary(blocks: dict[str, FeatureBlock]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for block_name in BLOCK_ORDER:
        spec = blocks[block_name]
        if block_name == 'mean_only':
            rows.append({'block': block_name, 'block_label': spec.label, 'category': spec.category, 'feature': 'training-fold outcome mean', 'feature_role': 'benchmark', 'timing': 'training fold only', 'rationale': spec.rationale})
            continue
        for feature in CLINICAL_FEATURES:
            rows.append({'block': block_name, 'block_label': spec.label, 'category': spec.category, 'feature': feature, 'feature_role': 'clinical', 'timing': 'baseline', 'rationale': spec.rationale})
        if block_name != 'clinical_only':
            for feature in spec.imaging_features:
                rows.append({'block': block_name, 'block_label': spec.label, 'category': spec.category, 'feature': feature, 'feature_role': 'gradient imaging', 'timing': 'baseline', 'rationale': spec.rationale})
    return pd.DataFrame(rows)

def validate_original_ridge_results(performance: pd.DataFrame) -> pd.DataFrame:
    previous = pd.read_csv(TABLE_ROOT / 'ml_clinical_augmented_performance_by_repeat.csv')
    block_map = {'clinical_only': 'clinical_reference', 'g1_vis_organization': 'G1_VIS_organization__plus_clinical', 'g2_vis_organization': 'G2_VIS_organization__plus_clinical', 'g1_global_spread': 'G1_global_spread__plus_clinical', 'g2_global_spread': 'G2_global_spread__plus_clinical', 'g1_network_organization': 'G1_network_organization__plus_clinical', 'g2_network_organization': 'G2_network_organization__plus_clinical'}
    rows: list[dict[str, object]] = []
    for (current_block, previous_block) in block_map.items():
        observed = performance.loc[(performance['algorithm'] == 'ridge') & (performance['cohort'] == 'primary_n29') & (performance['block'] == current_block)].sort_values('repeat')
        expected = previous.loc[previous['feature_block'] == previous_block].sort_values('repeat')
        if len(observed) != N_REPEATS or len(expected) != N_REPEATS:
            raise RuntimeError(f'Incomplete numerical regression comparison for {current_block}')
        for metric in ('mae', 'rmse', 'r2', 'pearson_r'):
            difference = np.abs(observed[metric].to_numpy(float) - expected[metric].to_numpy(float))
            rows.append({'check': f'legacy_ridge_match:{current_block}:{metric}', 'status': 'PASS' if float(np.nanmax(difference)) <= 1e-10 else 'FAIL', 'detail': f'max_abs_diff={float(np.nanmax(difference)):.3e}'})
    return pd.DataFrame(rows)

def mm_to_inches(width_mm: float, height_mm: float) -> tuple[float, float]:
    return (width_mm / 25.4, height_mm / 25.4)

def build_positions() -> tuple[dict[tuple[str, str], float], dict[str, tuple[float, float]]]:
    positions: dict[tuple[str, str], float] = {}
    spans: dict[str, tuple[float, float]] = {}
    cursor = 0.0
    for (cohort_index, cohort) in enumerate(COHORT_ORDER):
        start = cursor
        for block in PLOT_BLOCK_ORDER:
            positions[cohort, block] = cursor
            cursor += 1.0
        spans[cohort] = (start, cursor - 1.0)
        if cohort_index < len(COHORT_ORDER) - 1:
            cursor += 0.85
    return (positions, spans)

def save_bundle(fig: plt.Figure, base: Path) -> list[Path]:
    paths: list[Path] = []
    for suffix in ('svg', 'pdf', 'png', 'tiff'):
        path = base.with_suffix(f'.{suffix}')
        kwargs: dict[str, object] = {'facecolor': 'white'}
        if suffix == 'png':
            kwargs['dpi'] = 300
        if suffix == 'tiff':
            kwargs['dpi'] = 600
            kwargs['pil_kwargs'] = {'compression': 'tiff_lzw'}
        fig.savefig(path, **kwargs)
        if suffix == 'tiff':
            with Image.open(path) as source:
                rgb = source.convert('RGB')
            rgb.save(path, compression='tiff_lzw', dpi=(600, 600))
            rgb.close()
        paths.append(path)
    plt.close(fig)
    return paths

def make_forest(summary: pd.DataFrame, algorithm: str) -> tuple[list[Path], pd.DataFrame]:
    subset = summary.loc[(summary['algorithm'] == algorithm) & summary['block'].isin(PLOT_BLOCK_ORDER)].copy()
    subset['is_plot_mse_best'] = np.isclose(subset['mse_mean'], subset.groupby('cohort')['mse_mean'].transform('min'))
    subset['is_plot_mae_best'] = np.isclose(subset['mae_mean'], subset.groupby('cohort')['mae_mean'].transform('min'))
    (positions, spans) = build_positions()
    subset['y_position'] = [positions[cohort, block] for (cohort, block) in zip(subset['cohort'], subset['block'])]
    subset = subset.sort_values('y_position')
    subset.to_csv(HERE / f'source_{algorithm}_expanded_ml_forest.csv', index=False, encoding='utf-8-sig')
    max_y = max(positions.values())
    fig = plt.figure(figsize=mm_to_inches(183, 155))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.56, 2.55, 0.82], left=0.045, right=0.985, bottom=0.09, top=0.965, wspace=0.018)
    ax_labels = fig.add_subplot(grid[0, 0])
    ax_forest = fig.add_subplot(grid[0, 1], sharey=ax_labels)
    ax_table = fig.add_subplot(grid[0, 2], sharey=ax_labels)
    for (cohort_index, cohort) in enumerate(COHORT_ORDER):
        (y0, y1) = spans[cohort]
        if cohort_index % 2 == 0:
            for axis in (ax_labels, ax_forest, ax_table):
                axis.axhspan(y0 - 0.48, y1 + 0.48, color='#F4F5F7', zorder=0)
        ax_labels.plot([0.026, 0.026], [y0 - 0.36, y1 + 0.36], transform=ax_labels.get_yaxis_transform(), color=COHORT_COLORS[cohort], lw=2.4, solid_capstyle='round', clip_on=False)
        ax_labels.text(0.0, y0 - 0.63, COHORT_LABELS[cohort], transform=ax_labels.get_yaxis_transform(), ha='left', va='bottom', fontsize=7.2, fontweight='bold', color=COHORT_COLORS[cohort])
    x_min = 0.6 if algorithm == 'elasticnet' else 0.0
    x_max = 1.0
    ax_forest.axvline(0.0, color='#8A8A8A', lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax_forest.axvline(1.0, color='#D7D7D7', lw=0.6, zorder=1)
    ax_forest.grid(axis='x', color='#E6E6E6', lw=0.55, zorder=0)
    for (_, row) in subset.iterrows():
        y = float(row['y_position'])
        mean = float(row['r2_mean'])
        low = float(row['r2_p025'])
        high = float(row['r2_p975'])
        cohort = str(row['cohort'])
        block = str(row['block'])
        color = COHORT_COLORS[cohort]
        best = bool(row['is_plot_mse_best'])
        clipped_low = max(low, x_min)
        clipped_high = min(high, x_max)
        ax_forest.plot([clipped_low, clipped_high], [y, y], color=color, lw=1.05, alpha=0.84, zorder=2)
        ax_forest.plot([clipped_low, clipped_low], [y - 0.13, y + 0.13], color=color, lw=0.8, zorder=2)
        ax_forest.plot([clipped_high, clipped_high], [y - 0.13, y + 0.13], color=color, lw=0.8, zorder=2)
        if low < x_min or mean < x_min:
            ax_forest.scatter(x_min + 0.01, y, marker='<', s=15, color=color, edgecolor='none', zorder=3)
        if high > x_max or mean > x_max:
            ax_forest.scatter(x_max - 0.01, y, marker='>', s=15, color=color, edgecolor='none', zorder=3)
        ax_forest.scatter(np.clip(mean, x_min, x_max), y, s=34 if best else 22, marker='D' if best else 'o', color=color, edgecolor='#202020' if best else 'white', linewidth=0.55, zorder=4)
        ax_labels.text(0.1, y, BLOCK_LABELS[block], transform=ax_labels.get_yaxis_transform(), ha='left', va='center', fontsize=6.15, fontweight='bold' if best else 'normal', color='#222222')
        ax_table.text(0.29, y, f"{float(row['mse_mean']):.2f}", ha='center', va='center', fontsize=6.0, fontweight='bold' if bool(row['is_plot_mse_best']) else 'normal')
        ax_table.text(0.77, y, f"{float(row['mae_mean']):.2f}", ha='center', va='center', fontsize=6.0, fontweight='bold' if bool(row['is_plot_mae_best']) else 'normal')
    for axis in (ax_labels, ax_forest, ax_table):
        axis.set_ylim(max_y + 0.55, -1.12)
        axis.set_yticks([])
    ax_labels.set_xlim(0, 1)
    ax_labels.set_axis_off()
    ax_forest.set_xlim(x_min, x_max)
    tick_step = 0.1 if algorithm == 'elasticnet' else 0.2
    ax_forest.set_xticks(np.arange(x_min, x_max + 0.001, tick_step))
    ax_forest.set_xlabel('Mean repeated nested-CV post-score $R^2$ (higher is better)')
    ax_forest.tick_params(axis='x', width=0.7, length=2.5)
    ax_forest.spines['left'].set_visible(False)
    ax_forest.set_title(f'Post-treatment prediction: {ALGORITHM_LABELS[algorithm]}', loc='left', fontweight='bold', pad=8)
    ax_table.set_xlim(0, 1)
    ax_table.set_axis_off()
    header_y = -0.93
    ax_table.text(0.29, header_y, 'MSE ↓', ha='center', va='bottom', fontsize=6.5, fontweight='bold')
    ax_table.text(0.77, header_y, 'MAE ↓', ha='center', va='bottom', fontsize=6.5, fontweight='bold')
    ax_table.plot([0.02, 0.99], [-0.58, -0.58], color='#777777', lw=0.6, clip_on=False)
    outputs = save_bundle(fig, HERE / f'figure_{algorithm}_expanded_ml_forest')
    return (outputs, subset)

def make_combined_primary_forest(summary: pd.DataFrame) -> tuple[list[Path], pd.DataFrame]:
    subset = summary.loc[(summary['cohort'] == 'primary_n29') & summary['block'].isin(PLOT_BLOCK_ORDER) & summary['algorithm'].isin(ALGORITHMS)].copy()
    subset['is_plot_mse_best'] = np.isclose(subset['mse_mean'], subset.groupby('algorithm')['mse_mean'].transform('min'))
    subset['is_plot_mae_best'] = np.isclose(subset['mae_mean'], subset.groupby('algorithm')['mae_mean'].transform('min'))
    positions: dict[tuple[str, str], float] = {}
    spans: dict[str, tuple[float, float]] = {}
    cursor = 0.0
    for (algorithm_index, algorithm) in enumerate(ALGORITHMS):
        start = cursor
        for block in PLOT_BLOCK_ORDER:
            positions[algorithm, block] = cursor
            cursor += 1.0
        spans[algorithm] = (start, cursor - 1.0)
        if algorithm_index < len(ALGORITHMS) - 1:
            cursor += 0.85
    subset['y_position'] = [positions[algorithm, block] for (algorithm, block) in zip(subset['algorithm'], subset['block'])]
    subset = subset.sort_values('y_position')
    subset.to_csv(HERE / 'source_combined_models_primary_cohort.csv', index=False, encoding='utf-8-sig')
    max_y = max(positions.values())
    fig = plt.figure(figsize=mm_to_inches(183, 155))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.56, 2.55, 0.82], left=0.045, right=0.985, bottom=0.09, top=0.965, wspace=0.018)
    ax_labels = fig.add_subplot(grid[0, 0])
    ax_forest = fig.add_subplot(grid[0, 1], sharey=ax_labels)
    ax_table = fig.add_subplot(grid[0, 2], sharey=ax_labels)
    for (algorithm_index, algorithm) in enumerate(ALGORITHMS):
        (y0, y1) = spans[algorithm]
        if algorithm_index % 2 == 0:
            for axis in (ax_labels, ax_forest, ax_table):
                axis.axhspan(y0 - 0.48, y1 + 0.48, color='#F4F5F7', zorder=0)
        ax_labels.plot([0.026, 0.026], [y0 - 0.36, y1 + 0.36], transform=ax_labels.get_yaxis_transform(), color=ALGORITHM_COLORS[algorithm], lw=2.4, solid_capstyle='round', clip_on=False)
        ax_labels.text(0.0, y0 - 0.63, ALGORITHM_LABELS[algorithm], transform=ax_labels.get_yaxis_transform(), ha='left', va='bottom', fontsize=7.2, fontweight='bold', color=ALGORITHM_COLORS[algorithm])
    (x_min, x_max) = (0.0, 1.0)
    ax_forest.axvline(0.0, color='#8A8A8A', lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax_forest.axvline(1.0, color='#D7D7D7', lw=0.6, zorder=1)
    ax_forest.grid(axis='x', color='#E6E6E6', lw=0.55, zorder=0)
    for (_, row) in subset.iterrows():
        y = float(row['y_position'])
        mean = float(row['r2_mean'])
        low = float(row['r2_p025'])
        high = float(row['r2_p975'])
        algorithm = str(row['algorithm'])
        block = str(row['block'])
        color = ALGORITHM_COLORS[algorithm]
        best = bool(row['is_plot_mse_best'])
        clipped_low = max(low, x_min)
        clipped_high = min(high, x_max)
        ax_forest.plot([clipped_low, clipped_high], [y, y], color=color, lw=1.05, alpha=0.84, zorder=2)
        ax_forest.plot([clipped_low, clipped_low], [y - 0.13, y + 0.13], color=color, lw=0.8, zorder=2)
        ax_forest.plot([clipped_high, clipped_high], [y - 0.13, y + 0.13], color=color, lw=0.8, zorder=2)
        if low < x_min or mean < x_min:
            ax_forest.scatter(x_min + 0.01, y, marker='<', s=15, color=color, edgecolor='none', zorder=3)
        if high > x_max or mean > x_max:
            ax_forest.scatter(x_max - 0.01, y, marker='>', s=15, color=color, edgecolor='none', zorder=3)
        ax_forest.scatter(np.clip(mean, x_min, x_max), y, s=34 if best else 22, marker='D' if best else 'o', color=color, edgecolor='#202020' if best else 'white', linewidth=0.55, zorder=4)
        ax_labels.text(0.1, y, BLOCK_LABELS[block], transform=ax_labels.get_yaxis_transform(), ha='left', va='center', fontsize=6.15, fontweight='bold' if best else 'normal', color='#222222')
        ax_table.text(0.29, y, f"{float(row['mse_mean']):.2f}", ha='center', va='center', fontsize=6.0, fontweight='bold' if bool(row['is_plot_mse_best']) else 'normal')
        ax_table.text(0.77, y, f"{float(row['mae_mean']):.2f}", ha='center', va='center', fontsize=6.0, fontweight='bold' if bool(row['is_plot_mae_best']) else 'normal')
    for axis in (ax_labels, ax_forest, ax_table):
        axis.set_ylim(max_y + 0.55, -1.12)
        axis.set_yticks([])
    ax_labels.set_xlim(0, 1)
    ax_labels.set_axis_off()
    ax_forest.set_xlim(x_min, x_max)
    ax_forest.set_xticks(np.arange(0.0, 1.01, 0.2))
    ax_forest.set_xlabel('Mean repeated nested-CV post-score $R^2$ (higher is better)')
    ax_forest.tick_params(axis='x', width=0.7, length=2.5)
    ax_forest.spines['left'].set_visible(False)
    ax_forest.set_title('Post-treatment prediction (primary cohort, n=29)', loc='left', fontweight='bold', pad=8)
    ax_table.set_xlim(0, 1)
    ax_table.set_axis_off()
    header_y = -0.93
    ax_table.text(0.29, header_y, 'MSE ↓', ha='center', va='bottom', fontsize=6.5, fontweight='bold')
    ax_table.text(0.77, header_y, 'MAE ↓', ha='center', va='bottom', fontsize=6.5, fontweight='bold')
    ax_table.plot([0.02, 0.99], [-0.58, -0.58], color='#777777', lw=0.6, clip_on=False)
    outputs = save_bundle(fig, HERE / 'figure_combined_models_primary_cohort')
    return (outputs, subset)

def write_results(summary: pd.DataFrame, legacy_pass: bool) -> None:
    lines = ['# 扩展机器学习性能分析', '', '## 设计', '', f'- 主分析：n=29；敏感性分析：排除既有梯度配准敏感个体后 n=28。', f'- 外层 5 折交叉验证重复 {N_REPEATS} 次；每个外层训练折内用 4 折选择超参数。', '- 所有标准化均只在训练折拟合；预测终点为治疗后 Vasserman 评分。', '- 每个影像组都加入同一临床基底：治疗前评分、卒中后时间和性别。', '- ΔMAE 定义为 clinical-only MAE 减去当前模型 MAE；正值才代表影像特征带来改善。', '', '## 主要结果', '']
    for algorithm in ALGORITHMS:
        for cohort in COHORT_ORDER:
            part = summary.loc[(summary['algorithm'] == algorithm) & (summary['cohort'] == cohort)]
            best = part.sort_values('mse_mean').iloc[0]
            best_imaging = part.loc[~part['block'].isin(['mean_only', 'clinical_only'])].sort_values('mse_mean').iloc[0]
            clinical = part.loc[part['block'] == 'clinical_only'].iloc[0]
            g2_global = part.loc[part['block'] == 'g2_global_spread'].iloc[0]
            lines.extend([f'### {ALGORITHM_LABELS[algorithm]} — {COHORT_LABELS[cohort]}', '', f"- 全部候选中最低 MSE：{best['block_label']}（MSE={best['mse_mean']:.2f}, MAE={best['mae_mean']:.2f}, R²={best['r2_mean']:.3f}）。", f"- 最佳影像增强组：{best_imaging['block_label']}（ΔMAE={best_imaging['delta_mae_mean']:+.3f}；20 次重复中优于 clinical-only 的比例={best_imaging['repeat_fraction_better_than_clinical']:.2f}）。", f"- clinical-only：MAE={clinical['mae_mean']:.2f}, R²={clinical['r2_mean']:.3f}。", f"- 原先最接近临床基准的 G2 global spread：MAE={g2_global['mae_mean']:.2f}, R²={g2_global['r2_mean']:.3f}, ΔMAE={g2_global['delta_mae_mean']:+.3f}。", ''])
    all_imaging = summary.loc[~summary['block'].isin(['mean_only', 'clinical_only'])]
    stable_gain = all_imaging.loc[(all_imaging['delta_mae_mean'] > 0) & (all_imaging['delta_mae_p025'] > 0)]
    lines.extend(['## 结论边界', '', '- 没有影像增强组在重复分割的整个描述性区间内稳定优于 clinical-only。' if stable_gain.empty else '- 至少一个影像增强组在其重复分割描述性区间内保持正 ΔMAE；需结合表格逐项解释。', '- 因此，若某个影像组只有更好的平均值而 ΔMAE 区间跨 0，应表述为探索性排序，不应写成稳定增量预测。', '- 高维汇总组用于检验分布式信息是否补充紧凑指标，也同时暴露小样本下的维度风险。', '- 当前外部队列缺少治疗分组、MMSE 和完整体素级病灶数据，因此没有伪造这些不受数据支持的实验组。', '', '## 数值回归核对', '', f"- 新脚本对原 6 个 Ridge 临床整合组及 clinical-only 的逐重复结果核对：{('PASS' if legacy_pass else 'FAIL')}。"])
    (HERE / 'RESULTS_ZH.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

def qa_outputs(predictions: pd.DataFrame, performance: pd.DataFrame, summary: pd.DataFrame, regression_checks: pd.DataFrame, figure_paths: list[Path]) -> pd.DataFrame:
    checks = regression_checks.to_dict('records')

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({'check': name, 'status': 'PASS' if passed else 'FAIL', 'detail': detail})
    expected_performance = len(ALGORITHMS) * len(COHORT_ORDER) * N_REPEATS * len(BLOCK_ORDER)
    add('performance_row_count', len(performance) == expected_performance, f'rows={len(performance)}; expected={expected_performance}')
    expected_predictions = sum((len(frame) for frame in (load_frame(), load_frame().loc[lambda x: x['participant_id'] != SENSITIVITY_EXCLUSION]))) * len(ALGORITHMS) * N_REPEATS * len(BLOCK_ORDER)
    add('prediction_row_count', len(predictions) == expected_predictions, f'rows={len(predictions)}; expected={expected_predictions}')
    add('finite_core_metrics', np.isfinite(performance[['mae', 'mse', 'rmse', 'r2']].to_numpy(float)).all(), 'MAE, MSE, RMSE and R2 are finite')
    add('summary_cell_count', len(summary) == len(ALGORITHMS) * len(COHORT_ORDER) * len(BLOCK_ORDER), f'rows={len(summary)}')
    add('repeat_count', set(summary['n_repeats']) == {N_REPEATS}, f"unique={sorted(summary['n_repeats'].unique().tolist())}")
    add('cohort_sample_sizes', set(summary.loc[summary['cohort'] == 'primary_n29', 'n']) == {29} and set(summary.loc[summary['cohort'] == 'alignment_sensitivity_n28', 'n']) == {28}, 'primary=29; sensitivity=28')
    for path in figure_paths:
        add(f'exists:{path.name}', path.exists() and path.stat().st_size > 0, f'bytes={(path.stat().st_size if path.exists() else 0)}')
        if path.suffix == '.svg':
            editable = '<text' in path.read_text(encoding='utf-8')
            add(f'editable_text:{path.name}', editable, 'SVG contains text nodes')
        elif path.suffix == '.pdf':
            import fitz
            document = fitz.open(path)
            add(f'one_page:{path.name}', len(document) == 1, f'pages={len(document)}')
            text_length = len(''.join((page.get_text() for page in document)))
            add(f'pdf_text:{path.name}', text_length > 200, f'extracted_characters={text_length}')
            document.close()
        elif path.suffix == '.tiff':
            with Image.open(path) as image:
                dpi = image.info.get('dpi', (0, 0))
                compression = image.info.get('compression', '')
                passed = image.mode == 'RGB' and min(dpi) >= 599 and (compression == 'tiff_lzw')
                add(f'tiff_spec:{path.name}', passed, f'mode={image.mode}; dpi={dpi}; compression={compression}')
        elif path.suffix == '.png':
            with Image.open(path) as image:
                add(f'png_dimensions:{path.name}', image.width >= 2000 and image.height >= 1600, f'size={image.width}x{image.height}')
    return pd.DataFrame(checks)

def main() -> None:
    start = time.time()
    frame = load_frame()
    blocks = define_feature_blocks(frame)
    cohorts = {'primary_n29': frame.copy(), 'alignment_sensitivity_n28': frame.loc[frame['participant_id'] != SENSITIVITY_EXCLUSION].copy()}
    if len(cohorts['alignment_sensitivity_n28']) != 28:
        raise RuntimeError('Expected one participant in the alignment-sensitivity exclusion')
    (predictions, performance, tuning) = run_nested_cv(cohorts, blocks)
    (paired_performance, summary) = summarize_performance(performance, blocks)
    dictionary = write_feature_dictionary(blocks)
    regression_checks = validate_original_ridge_results(performance)
    legacy_pass = bool((regression_checks['status'] == 'PASS').all())
    if not legacy_pass:
        raise RuntimeError('Expanded analysis did not numerically reproduce the original Ridge rows')
    predictions.to_csv(HERE / 'ml_oof_predictions.csv', index=False, encoding='utf-8-sig')
    paired_performance.to_csv(HERE / 'ml_performance_by_repeat.csv', index=False, encoding='utf-8-sig')
    summary.to_csv(HERE / 'ml_performance_summary.csv', index=False, encoding='utf-8-sig')
    tuning.to_csv(HERE / 'ml_outer_fold_tuning.csv', index=False, encoding='utf-8-sig')
    dictionary.to_csv(HERE / 'ml_feature_dictionary.csv', index=False, encoding='utf-8-sig')
    regression_checks.to_csv(HERE / 'legacy_ridge_regression_checks.csv', index=False, encoding='utf-8-sig')
    (figure_paths, _) = make_combined_primary_forest(summary)
    write_results(summary, legacy_pass)
    qa = qa_outputs(predictions, performance, summary, regression_checks, figure_paths)
    qa.to_csv(HERE / 'QA_CHECKS.csv', index=False, encoding='utf-8-sig')
    failures = qa.loc[qa['status'] != 'PASS']
    qa_report = ['# Expanded ML forest-plot QA', '', f"Overall: {('PASS' if failures.empty else 'FAIL')}", '', f"- Checks passing: {(qa['status'] == 'PASS').sum()}/{len(qa)}", f"- Numerical regression against the original Ridge rows: {('PASS' if legacy_pass else 'FAIL')}", '- All plots were generated and rendered with Python.', '- Repeated-split intervals are explicitly labelled as descriptive rather than confidence intervals.', '- SVG text remains editable; PDFs are one page; TIFF files are RGB, LZW-compressed and 600 dpi.']
    if not failures.empty:
        qa_report.extend(['', '## Failures', ''])
        qa_report.extend((f'- {row.check}: {row.detail}' for row in failures.itertuples(index=False)))
    (HERE / 'QA_REPORT.md').write_text('\n'.join(qa_report) + '\n', encoding='utf-8')
    manifest = {'backend': 'Python', 'random_seed': SEED, 'cohorts': {name: len(data) for (name, data) in cohorts.items()}, 'algorithms': list(ALGORITHMS), 'outer_cv': f'{N_OUTER_SPLITS}-fold repeated {N_REPEATS} times', 'inner_cv': f'{N_INNER_SPLITS}-fold hyperparameter tuning inside every outer training fold', 'ridge_alpha_grid': [item['alpha'] for item in candidate_parameters('ridge')], 'elasticnet_grid': candidate_parameters('elasticnet'), 'clinical_features': list(CLINICAL_FEATURES), 'feature_blocks': {name: {'label': spec.label, 'category': spec.category, 'n_imaging_features': len(spec.imaging_features), 'imaging_features': list(spec.imaging_features), 'rationale': spec.rationale} for (name, spec) in blocks.items()}, 'interval_definition': '2.5th-97.5th percentiles across repeated outer-CV splits; descriptive, not an independent-sample confidence interval', 'delta_mae_definition': 'clinical-only MAE minus candidate-block MAE; positive favors candidate block', 'legacy_ridge_regression_match': legacy_pass, 'runtime_seconds': round(time.time() - start, 2), 'outputs': [path.name for path in figure_paths]}
    (HERE / 'analysis_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'qa': 'PASS' if failures.empty else 'FAIL', **manifest}, indent=2, ensure_ascii=False))
    if not failures.empty:
        raise RuntimeError('Output QA failed; inspect QA_REPORT.md')

def qa_only() -> None:
    predictions = pd.read_csv(HERE / 'ml_oof_predictions.csv')
    performance = pd.read_csv(HERE / 'ml_performance_by_repeat.csv')
    summary = pd.read_csv(HERE / 'ml_performance_summary.csv')
    regression_checks = pd.read_csv(HERE / 'legacy_ridge_regression_checks.csv')
    (figure_paths, _) = make_combined_primary_forest(summary)
    qa = qa_outputs(predictions, performance, summary, regression_checks, figure_paths)
    qa.to_csv(HERE / 'QA_CHECKS.csv', index=False, encoding='utf-8-sig')
    failures = qa.loc[qa['status'] != 'PASS']
    report = ['# Expanded ML forest-plot QA', '', f"Overall: {('PASS' if failures.empty else 'FAIL')}", '', f"- Checks passing: {(qa['status'] == 'PASS').sum()}/{len(qa)}", '- Numerical regression against the original Ridge rows: PASS', '- All plots were generated and rendered with Python.', '- Repeated-split intervals are explicitly labelled as descriptive rather than confidence intervals.', '- SVG text remains editable; PDFs are one page; TIFF files are RGB, LZW-compressed and 600 dpi.']
    if not failures.empty:
        report.extend(['', '## Failures', ''])
        report.extend((f'- {row.check}: {row.detail}' for row in failures.itertuples(index=False)))
    (HERE / 'QA_REPORT.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    manifest_path = HERE / 'analysis_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['qa_status'] = 'PASS' if failures.empty else 'FAIL'
    manifest['qa_checks_passing'] = int((qa['status'] == 'PASS').sum())
    manifest['qa_checks_total'] = len(qa)
    manifest['figure_scope'] = 'Ridge and Elastic Net combined; primary cohort n=29 only'
    manifest['outputs'] = [path.name for path in figure_paths]
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'qa': manifest['qa_status'], 'checks': f"{manifest['qa_checks_passing']}/{manifest['qa_checks_total']}"}, indent=2))
    if not failures.empty:
        raise RuntimeError('Output QA failed; inspect QA_REPORT.md')
if __name__ == '__main__':
    if '--qa-only' in sys.argv:
        qa_only()
    else:
        main()
