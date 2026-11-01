from __future__ import annotations
import os as _portable_os
from pathlib import Path as _PortablePath

def _portable_path(kind, relative):
    project = _PortablePath(_portable_os.environ['PAPER_PROJECT_ROOT'])
    root = project if kind == 'project' else project.parent
    return str(root / relative)
import json
from pathlib import Path
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
ROOT = Path(_portable_path('work', ''))
ANALYSIS = ROOT / 'Gradient' / 'lesion_analysis'
LESION_RESULTS = ANALYSIS / 'results_abcd'
OUT = ANALYSIS / 'ml_results'
FIGURES = OUT / 'figures'
TABLES = OUT / 'tables'
SPLITS = OUT / 'splits'
CORTICAL_ROOT = ROOT / 'Gradient' / '皮层' / '(400，5)'
SUBCORTICAL_ROOT = ROOT / 'Gradient' / '皮层下' / '(54,5)'
LABELS_PATH = ANALYSIS / 'resources' / 'Schaefer2018_400Parcels_7Networks_order.txt'
RANDOM_STATE = 20260803
N_SPLITS = 5
N_REPEATS = 20
NETWORK_ORDER = ('VIS', 'SMN', 'DAN', 'VAN', 'LIM', 'CON', 'DMN')
NETWORK_MAP = {'Vis': 'VIS', 'SomMot': 'SMN', 'DorsAttn': 'DAN', 'SalVentAttn': 'VAN', 'Limbic': 'LIM', 'Cont': 'CON', 'Default': 'DMN'}
BLOCK_ORDER = ('clinical', 'clinical_lesion', 'clinical_gradient', 'combined')
BLOCK_LABELS = {'clinical': 'Clinical', 'clinical_lesion': 'Clinical + lesion', 'clinical_gradient': 'Clinical + gradient', 'combined': 'Combined'}
BLOCK_COLORS = {'clinical': '#777777', 'clinical_lesion': '#D98C3F', 'clinical_gradient': '#4C78A8', 'combined': '#2A9D8F'}
MODEL_ORDER = ('ridge', 'elastic_net', 'pls')
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans', 'sans-serif'], 'font.size': 7, 'axes.titlesize': 8, 'axes.labelsize': 7, 'axes.linewidth': 0.7, 'axes.spines.right': False, 'axes.spines.top': False, 'legend.frameon': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})

def ensure_dirs() -> None:
    for path in (FIGURES, TABLES, SPLITS):
        path.mkdir(parents=True, exist_ok=True)

def save_figure(fig: plt.Figure, stem: str) -> None:
    for (suffix, kwargs) in (('png', {'dpi': 300}), ('svg', {}), ('pdf', {}), ('tiff', {'dpi': 600})):
        fig.savefig(FIGURES / f'{stem}.{suffix}', bbox_inches='tight', facecolor='white', **kwargs)

def gradient_file(root: Path, group: str, session: str, subject: str) -> Path:
    matches = sorted((root / group / session).glob(f'{subject}-*.npy'))
    if len(matches) != 1:
        raise RuntimeError(f'Expected one gradient file for {root.name}/{group}/{session}/{subject}, found {len(matches)}')
    return matches[0]

def roi_networks() -> np.ndarray:
    networks = []
    for line in LABELS_PATH.read_text(encoding='utf-8').splitlines():
        fields = line.split()
        if len(fields) < 2:
            continue
        networks.append(NETWORK_MAP[fields[1].split('_')[2]])
    array = np.asarray(networks)
    if array.shape != (400,):
        raise RuntimeError(f'Expected 400 ROI network labels, found {array.shape}')
    return array

def build_feature_table() -> tuple[pd.DataFrame, dict[str, list[str]], pd.DataFrame]:
    subject_metrics = pd.read_csv(LESION_RESULTS / 'tables' / 'D_gradient_clinical_subject_metrics.csv', dtype={'subject_id': str})
    lesion_metrics = pd.read_csv(LESION_RESULTS / 'tables' / 'B_participant_lesion_metrics.csv', dtype={'subject_id': str})
    audit = pd.read_csv(ANALYSIS / 'mni_registration' / 'registration_audit_final.csv', dtype={'subject_id': str})
    network_burden = pd.read_csv(LESION_RESULTS / 'tables' / 'B_seven_network_burden.csv', dtype={'subject_id': str})
    network_wide = network_burden.pivot(index='subject_id', columns='network', values='network_damage_pct')
    network_wide.columns = [f'lesion_{network}_pct' for network in network_wide.columns]
    network_wide = network_wide.reset_index()
    lesion_columns = ['subject_id', 'baseline_lesion_volume_ml', 'schaefer_cortical_fraction_pct']
    audit_columns = ['subject_id', 'mni_centroid_x', 'mni_centroid_y', 'mni_centroid_z', 'manual_audit_status']
    data = subject_metrics.merge(lesion_metrics[lesion_columns], on='subject_id', validate='one_to_one')
    data = data.merge(audit[audit_columns], on='subject_id', validate='one_to_one')
    data = data.merge(network_wide, on='subject_id', validate='one_to_one')
    data['log1p_lesion_volume_ml'] = np.log1p(data['baseline_lesion_volume_ml'].astype(float))
    data['age'] = data['年龄'].astype(float)
    roi_network = roi_networks()
    gradient_rows = []
    for row in data.itertuples():
        cortical = np.load(gradient_file(CORTICAL_ROOT, row.group, 'before', row.subject_id))
        subcortical = np.load(gradient_file(SUBCORTICAL_ROOT, row.group, 'before', row.subject_id))
        if cortical.shape != (400, 5) or subcortical.shape != (54, 5):
            raise RuntimeError(f'Unexpected baseline gradient shape for {row.subject_id}')
        values: dict[str, object] = {'subject_id': row.subject_id}
        for network in NETWORK_ORDER:
            mask = roi_network == network
            for component in (1, 2):
                values[f'cortical_{network}_G{component}_mean'] = float(np.mean(cortical[mask, component - 1]))
        for component in (1, 2):
            values[f'cortical_G{component}_sd'] = float(np.std(cortical[:, component - 1], ddof=1))
            values[f'subcortical_G{component}_mean'] = float(np.mean(subcortical[:, component - 1]))
            values[f'subcortical_G{component}_sd'] = float(np.std(subcortical[:, component - 1], ddof=1))
        gradient_rows.append(values)
    gradient_frame = pd.DataFrame(gradient_rows)
    data = data.merge(gradient_frame, on='subject_id', validate='one_to_one')
    clinical_features = ['group_tms', 'age', 'sex_male', 'NIHSS_pre', 'ADL_pre', 'MMSE_pre', 'HAMD_pre', 'VFT_pre']
    lesion_features = ['log1p_lesion_volume_ml', 'schaefer_cortical_fraction_pct', 'mask_lateralization_index', 'mni_centroid_x', 'mni_centroid_y', 'mni_centroid_z'] + [f'lesion_{network}_pct' for network in NETWORK_ORDER]
    gradient_features = [column for column in gradient_frame.columns if column != 'subject_id']
    candidate_features = clinical_features + lesion_features + gradient_features
    constant = [column for column in candidate_features if data[column].nunique(dropna=False) <= 1]
    lesion_features = [column for column in lesion_features if column not in constant]
    gradient_features = [column for column in gradient_features if column not in constant]
    blocks = {'clinical': clinical_features, 'clinical_lesion': clinical_features + lesion_features, 'clinical_gradient': clinical_features + gradient_features, 'combined': clinical_features + lesion_features + gradient_features}
    if len(data) != 38 or data['subject_id'].nunique() != 38:
        raise RuntimeError('The fair multimodal comparison requires the 38-subject common cohort')
    if data[blocks['combined'] + ['clinical_composite_change']].isna().any().any():
        raise RuntimeError('Missing value in the common-cohort ML table')
    dictionary_rows = []
    descriptions = {'group_tms': 'Observed treatment group (TMS=1, Control=0); not an individualized causal treatment effect', 'age': 'Age at baseline', 'sex_male': 'Sex indicator (male=1)', 'NIHSS_pre': 'Baseline NIHSS', 'ADL_pre': 'Baseline ADL', 'MMSE_pre': 'Baseline MMSE', 'HAMD_pre': 'Baseline HAMD', 'VFT_pre': 'Baseline VFT', 'log1p_lesion_volume_ml': 'log(1 + baseline lesion volume in ml)', 'schaefer_cortical_fraction_pct': 'Percent of lesion voxels intersecting Schaefer cortical parcels', 'mask_lateralization_index': 'MNI mask right-minus-left voxel fraction', 'mni_centroid_x': 'Baseline lesion centroid MNI x coordinate', 'mni_centroid_y': 'Baseline lesion centroid MNI y coordinate', 'mni_centroid_z': 'Baseline lesion centroid MNI z coordinate'}
    for feature in blocks['combined']:
        if feature in clinical_features:
            source_block = 'clinical'
        elif feature in lesion_features:
            source_block = 'lesion'
        else:
            source_block = 'gradient'
        if feature.startswith('lesion_') and feature.endswith('_pct'):
            description = 'Percent direct baseline lesion damage to the named Schaefer network'
        elif feature.startswith('cortical_'):
            description = 'Baseline cortical gradient summary'
        elif feature.startswith('subcortical_'):
            description = 'Baseline 54-region subcortical gradient summary; atlas identity remains to be documented'
        else:
            description = descriptions.get(feature, feature)
        dictionary_rows.append({'feature': feature, 'source_block': source_block, 'description': description, 'included_in_primary_common_cohort': True, 'timing': 'baseline'})
    for feature in constant:
        dictionary_rows.append({'feature': feature, 'source_block': 'lesion', 'description': 'Excluded because the feature was constant in the 38-subject common cohort', 'included_in_primary_common_cohort': False, 'timing': 'baseline'})
    dictionary = pd.DataFrame(dictionary_rows)
    return (data.sort_values('subject_id').reset_index(drop=True), blocks, dictionary)

def model_grid(name: str, n_features: int, seed: int) -> tuple[Pipeline, dict[str, list[object]]]:
    if name == 'ridge':
        model = Ridge()
        grid: dict[str, list[object]] = {'model__alpha': np.logspace(-3, 3, 13).tolist()}
    elif name == 'elastic_net':
        model = ElasticNet(max_iter=30000, random_state=seed)
        grid = {'model__alpha': np.logspace(-3, 1, 7).tolist(), 'model__l1_ratio': [0.1, 0.5, 0.9]}
    elif name == 'pls':
        model = PLSRegression(scale=False, max_iter=1000)
        grid = {'model__n_components': list(range(1, min(5, n_features) + 1))}
    else:
        raise ValueError(name)
    return (Pipeline([('scale', StandardScaler()), ('model', model)]), grid)

def extract_coefficients(search: GridSearchCV, features: list[str]) -> np.ndarray:
    fitted = search.best_estimator_.named_steps['model']
    coefficients = np.asarray(fitted.coef_, dtype=float).reshape(-1)
    if coefficients.size != len(features):
        raise RuntimeError(f'Coefficient size mismatch: {coefficients.size} vs {len(features)}')
    return coefficients

def run_nested_cv(data: pd.DataFrame, blocks: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    y = data['clinical_composite_change'].to_numpy(float)
    stratify = data['group_tms'].to_numpy(int)
    splitter = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
    split_rows = list(splitter.split(np.zeros(len(data)), stratify))
    prediction_rows = []
    coefficient_rows = []
    split_manifest = []
    for (split_index, (train, test)) in enumerate(split_rows):
        repeat = split_index // N_SPLITS
        fold = split_index % N_SPLITS
        seed = RANDOM_STATE + split_index
        inner = KFold(n_splits=4, shuffle=True, random_state=seed)
        split_manifest.append({'split_index': split_index, 'repeat': repeat, 'fold': fold, 'train_subject_ids': data.loc[train, 'subject_id'].tolist(), 'test_subject_ids': data.loc[test, 'subject_id'].tolist()})
        for block in BLOCK_ORDER:
            features = blocks[block]
            x_train = data.loc[train, features].to_numpy(float)
            x_test = data.loc[test, features].to_numpy(float)
            for model_name in MODEL_ORDER:
                (pipeline, grid) = model_grid(model_name, len(features), seed)
                search = GridSearchCV(pipeline, grid, scoring='neg_mean_absolute_error', cv=inner, n_jobs=1, refit=True, error_score='raise')
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    search.fit(x_train, y[train])
                prediction = np.asarray(search.predict(x_test), dtype=float).reshape(-1)
                best_params = json.dumps(search.best_params_, sort_keys=True)
                for (index, predicted) in zip(test, prediction):
                    prediction_rows.append({'subject_id': data.loc[index, 'subject_id'], 'group': data.loc[index, 'group'], 'repeat': repeat, 'fold': fold, 'split_index': split_index, 'model': model_name, 'feature_block': block, 'observed_composite_improvement': y[index], 'predicted_composite_improvement': float(predicted), 'best_params': best_params})
                coefficients = extract_coefficients(search, features)
                for (feature, coefficient) in zip(features, coefficients):
                    coefficient_rows.append({'repeat': repeat, 'fold': fold, 'split_index': split_index, 'model': model_name, 'feature_block': block, 'feature': feature, 'standardized_coefficient': float(coefficient), 'best_params': best_params})
        if fold == N_SPLITS - 1:
            print(f'Completed nested-CV repeat {repeat + 1}/{N_REPEATS}', flush=True)
    predictions = pd.DataFrame(prediction_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    performance_rows = []
    for ((model_name, block, repeat), subset) in predictions.groupby(['model', 'feature_block', 'repeat'], sort=False):
        observed = subset['observed_composite_improvement'].to_numpy(float)
        predicted = subset['predicted_composite_improvement'].to_numpy(float)
        if len(subset) != len(data) or subset['subject_id'].nunique() != len(data):
            raise RuntimeError(f'Incomplete out-of-fold prediction set for {model_name}/{block}/repeat {repeat}')
        correlation = float(pearsonr(observed, predicted).statistic) if np.std(predicted) > 0 else np.nan
        calibration_slope = float(np.polyfit(predicted, observed, 1)[0]) if np.std(predicted) > 0 else np.nan
        performance_rows.append({'model': model_name, 'feature_block': block, 'repeat': repeat, 'n': len(subset), 'mae': float(mean_absolute_error(observed, predicted)), 'rmse': float(np.sqrt(mean_squared_error(observed, predicted))), 'r2': float(r2_score(observed, predicted)), 'pearson_r': correlation, 'calibration_slope': calibration_slope})
    return (predictions, coefficients, pd.DataFrame(performance_rows), split_manifest)

def summarize_performance(performance: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = ('mae', 'rmse', 'r2', 'pearson_r', 'calibration_slope')
    for ((model_name, block), subset) in performance.groupby(['model', 'feature_block'], sort=False):
        for metric in metrics:
            values = subset[metric].dropna().to_numpy(float)
            rows.append({'model': model_name, 'feature_block': block, 'metric': metric, 'n_repeats': len(values), 'mean': float(np.mean(values)), 'median': float(np.median(values)), 'sd': float(np.std(values, ddof=1)), 'repeat_interval_low': float(np.percentile(values, 2.5)), 'repeat_interval_high': float(np.percentile(values, 97.5))})
    return pd.DataFrame(rows)

def aggregate_oof(predictions: pd.DataFrame, model: str, block: str) -> pd.DataFrame:
    subset = predictions.loc[(predictions['model'] == model) & (predictions['feature_block'] == block)]
    return subset.groupby(['subject_id', 'group'], as_index=False).agg(observed=('observed_composite_improvement', 'first'), predicted=('predicted_composite_improvement', 'mean'), prediction_sd=('predicted_composite_improvement', 'std'))

def plot_main_figure(performance: pd.DataFrame, predictions: pd.DataFrame, coefficients: pd.DataFrame) -> None:
    (fig, axes) = plt.subplots(2, 2, figsize=(7.2, 5.7), facecolor='white')
    rng = np.random.default_rng(RANDOM_STATE)
    ridge = performance.loc[performance['model'] == 'ridge']
    for (position, block) in enumerate(BLOCK_ORDER):
        values = ridge.loc[ridge['feature_block'] == block, 'mae'].to_numpy(float)
        jitter = rng.uniform(-0.08, 0.08, len(values))
        axes[0, 0].scatter(position + jitter, values, s=10, color=BLOCK_COLORS[block], alpha=0.55, edgecolor='none')
        mean = float(np.mean(values))
        (low, high) = np.percentile(values, [2.5, 97.5])
        axes[0, 0].errorbar(position, mean, yerr=[[mean - low], [high - mean]], fmt='o', markersize=5, color=BLOCK_COLORS[block], capsize=2, linewidth=1.1)
    axes[0, 0].set_xticks(range(len(BLOCK_ORDER)), ['Clinical', '+ lesion', '+ gradient', 'Combined'], rotation=18, ha='right')
    axes[0, 0].set_ylabel('Nested-CV MAE')
    axes[0, 0].set_title('Ridge performance by feature block')
    oof = aggregate_oof(predictions, 'ridge', 'combined')
    group_colors = {'Control': '#4C78A8', 'TMS': '#D65F45'}
    for (group, subset) in oof.groupby('group'):
        axes[0, 1].scatter(subset['observed'], subset['predicted'], s=18, color=group_colors[group], alpha=0.78, edgecolor='white', linewidth=0.35, label=group)
    limits = [float(min(oof['observed'].min(), oof['predicted'].min())), float(max(oof['observed'].max(), oof['predicted'].max()))]
    padding = (limits[1] - limits[0]) * 0.06
    limits = [limits[0] - padding, limits[1] + padding]
    axes[0, 1].plot(limits, limits, color='#555555', linewidth=0.8, linestyle='--')
    axes[0, 1].set_xlim(limits)
    axes[0, 1].set_ylim(limits)
    axes[0, 1].set_xlabel('Observed composite improvement')
    axes[0, 1].set_ylabel('Mean repeated OOF prediction')
    axes[0, 1].set_title('Combined ridge: observed vs predicted')
    axes[0, 1].legend(fontsize=6, loc='upper left')
    mae = mean_absolute_error(oof['observed'], oof['predicted'])
    r2 = r2_score(oof['observed'], oof['predicted'])
    correlation = pearsonr(oof['observed'], oof['predicted']).statistic
    axes[0, 1].text(0.97, 0.04, f'MAE={mae:.2f}\nR2={r2:.2f}\nr={correlation:.2f}', transform=axes[0, 1].transAxes, ha='right', va='bottom', fontsize=6)
    clinical_mae = ridge.loc[ridge['feature_block'] == 'clinical', ['repeat', 'mae']].set_index('repeat')['mae']
    additions = ('clinical_lesion', 'clinical_gradient', 'combined')
    for (position, block) in enumerate(additions):
        block_mae = ridge.loc[ridge['feature_block'] == block, ['repeat', 'mae']].set_index('repeat')['mae']
        delta = (block_mae - clinical_mae).sort_index().to_numpy(float)
        jitter = rng.uniform(-0.08, 0.08, len(delta))
        axes[1, 0].scatter(position + jitter, delta, s=10, color=BLOCK_COLORS[block], alpha=0.55, edgecolor='none')
        mean = float(np.mean(delta))
        (low, high) = np.percentile(delta, [2.5, 97.5])
        axes[1, 0].errorbar(position, mean, yerr=[[mean - low], [high - mean]], fmt='o', markersize=5, color=BLOCK_COLORS[block], capsize=2, linewidth=1.1)
    axes[1, 0].axhline(0, color='#777777', linewidth=0.7, linestyle='--')
    axes[1, 0].set_xticks(range(len(additions)), ['+ lesion', '+ gradient', 'Combined'])
    axes[1, 0].set_ylabel('Delta MAE vs clinical\n(negative = improvement)')
    axes[1, 0].set_title('Incremental predictive value')
    combined_ridge = coefficients.loc[(coefficients['model'] == 'ridge') & (coefficients['feature_block'] == 'combined')]
    stability = combined_ridge.groupby('feature')['standardized_coefficient'].agg(mean='mean', mean_abs=lambda x: float(np.mean(np.abs(x))), low=lambda x: float(np.percentile(x, 2.5)), high=lambda x: float(np.percentile(x, 97.5))).sort_values('mean_abs', ascending=False).head(10).sort_values('mean')
    y = np.arange(len(stability))
    axes[1, 1].axvline(0, color='#777777', linewidth=0.7, linestyle='--')
    axes[1, 1].errorbar(stability['mean'], y, xerr=np.vstack([stability['mean'] - stability['low'], stability['high'] - stability['mean']]), fmt='o', markersize=4, color='#2A9D8F', ecolor='#2A9D8F', capsize=2, linewidth=1)
    labels = [name.replace('cortical_', 'ctx ').replace('subcortical_', 'subctx ').replace('_mean', ' mean').replace('_sd', ' SD').replace('_', ' ') for name in stability.index]
    axes[1, 1].set_yticks(y, labels, fontsize=5.8)
    axes[1, 1].set_xlabel('Standardized coefficient\n(2.5th-97.5th percentile across outer folds)')
    axes[1, 1].set_title('Combined ridge coefficient stability')
    for (label, ax) in zip(('A', 'B', 'C', 'D'), axes.ravel()):
        ax.text(-0.16, 1.08, label, transform=ax.transAxes, fontsize=11, fontweight='bold', va='top')
        ax.tick_params(width=0.6, length=2.5)
    fig.suptitle('Internally validated multimodal prediction of clinical improvement', fontsize=9, y=0.995)
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.96), h_pad=2.0, w_pad=2.0)
    save_figure(fig, 'ML_multimodal_prediction')
    plt.close(fig)

def main() -> None:
    ensure_dirs()
    (data, blocks, dictionary) = build_feature_table()
    dictionary.to_csv(TABLES / 'ML_feature_dictionary.csv', index=False, encoding='utf-8-sig')
    data[['subject_id', 'group', 'clinical_composite_change', 'manual_audit_status'] + blocks['combined']].to_csv(TABLES / 'ML_common_cohort_features.csv', index=False, encoding='utf-8-sig')
    (predictions, coefficients, performance, split_manifest) = run_nested_cv(data, blocks)
    predictions.to_csv(TABLES / 'ML_nested_cv_predictions.csv', index=False, encoding='utf-8-sig')
    coefficients.to_csv(TABLES / 'ML_outer_fold_coefficients.csv', index=False, encoding='utf-8-sig')
    performance.to_csv(TABLES / 'ML_performance_by_repeat.csv', index=False, encoding='utf-8-sig')
    summary = summarize_performance(performance)
    summary.to_csv(TABLES / 'ML_performance_summary.csv', index=False, encoding='utf-8-sig')
    aggregate_oof(predictions, 'ridge', 'combined').to_csv(TABLES / 'ML_combined_ridge_aggregated_oof.csv', index=False, encoding='utf-8-sig')
    (SPLITS / 'nested_cv_splits.json').write_text(json.dumps(split_manifest, ensure_ascii=True, indent=2), encoding='utf-8')
    plot_main_figure(performance, predictions, coefficients)
    manifest = {'analysis_date': '2026-08-03', 'cohort': '38-subject complete lesion + cortical gradient + subcortical gradient + clinical intersection', 'target': 'mean direction-oriented clinical change across NIHSS, ADL, MMSE, HAMD, and VFT, standardized by baseline-score SD', 'outer_cv': f'{N_REPEATS} repeats of {N_SPLITS}-fold stratified CV by observed treatment group', 'inner_cv': '4-fold shuffled CV minimizing MAE', 'models': list(MODEL_ORDER), 'feature_blocks': {key: value for (key, value) in blocks.items()}, 'leakage_control': 'all feature scaling, model fitting, and hyperparameter selection occur inside outer-training data', 'interpretation': 'prognostic prediction conditional on observed group; not individualized causal TMS benefit', 'external_validation': False}
    (OUT / 'ml_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'ML cohort: {len(data)}')
    print(f'Feature counts: { {block: len(features) for (block, features) in blocks.items()}}')
    print(f'Prediction rows: {len(predictions)}')
    print(f'Output: {OUT}')
if __name__ == '__main__':
    main()
