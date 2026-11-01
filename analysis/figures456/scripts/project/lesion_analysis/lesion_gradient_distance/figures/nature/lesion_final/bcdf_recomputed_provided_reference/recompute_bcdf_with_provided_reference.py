"""Recompute panels b, c, d and f with the supplied gradient pipeline reference.

The supplied/precomputed five-dimensional aligned gradients are used for every
Pre/Post subject. XTRACT cortical projection weights and anatomical distances are
reference-independent and are retained from the lesion-distance analysis, while
lesion positions and distances in G1/G2 space are recomputed from the new fixed
healthy reference.
"""
from __future__ import annotations
import os as _portable_os
from pathlib import Path as _PortablePath

def _portable_path(kind, relative):
    project = _PortablePath(_portable_os.environ['PAPER_PROJECT_ROOT'])
    root = project if kind == 'project' else project.parent
    return str(root / relative)
import itertools
import json
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import t
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
OUT_DIR = Path(__file__).resolve().parent
LESION_DISTANCE_DIR = OUT_DIR.parents[3]
ANALYSIS_ROOT = OUT_DIR.parents[4]
PROVIDED_DIR = LESION_DISTANCE_DIR / 'provided_code_g1_g2_recalculation'
ALIGNED_DIR = PROVIDED_DIR / 'aligned_gradients_z'
REFERENCE_PATH = PROVIDED_DIR / 'healthy_reference_provided_code_aligned_z_400x5.npy'
OLD_ROI_PATH = LESION_DISTANCE_DIR / 'tables' / 'roi_level_distance_reorganization.csv'
OLD_SUBJECT_PATH = LESION_DISTANCE_DIR / 'tables' / 'subject_level_distance_metrics.csv'
OLD_GROUP_MODEL_PATH = LESION_DISTANCE_DIR / 'tables' / 'group_models_reference_directed_reorganization.csv'
MATCHED_PAIRS_PATH = ANALYSIS_ROOT / 'lesion_matched_clinical_mediation' / 'tables' / 'optimal_lesion_matched_pairs.csv'
SEED = 20260803
N_PERMUTATIONS = 5000
CONTROL = '#3F73A8'
TMS = '#D6534A'
BLACK = '#222222'
NEUTRAL = '#666666'
ZERO = '#A8A8A8'
mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'], 'font.size': 7, 'axes.linewidth': 0.7, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False, 'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})

def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(['true', '1', 'yes'])

def zscore_vector(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    sd = float(np.std(values, ddof=0))
    if not np.isfinite(sd) or sd <= 0:
        return np.zeros_like(values)
    return (values - float(np.mean(values))) / sd

def mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    mean = float(np.mean(values))
    if len(values) < 2:
        return (mean, np.nan, np.nan)
    se = float(np.std(values, ddof=1) / np.sqrt(len(values)))
    critical = float(t.ppf(0.975, df=len(values) - 1))
    return (mean, mean - critical * se, mean + critical * se)

def p_text(value: float) -> str:
    if value < 0.001:
        return '<0.001'
    return f'{value:.3f}'

def fit_roi_model(roi_table: pd.DataFrame, outcome: str, gradient_distance: str | None) -> dict[str, float]:
    columns = [outcome, 'anatomical_distance_min_mm']
    if gradient_distance is not None:
        columns.append(gradient_distance)
    model_data = roi_table.loc[as_bool(roi_table['intact_roi']), columns].dropna().copy()
    if len(model_data) < 10:
        return {'n_roi': len(model_data), 'r_squared': np.nan, 'anatomical_slope': np.nan, 'anatomical_slope_se_hc3': np.nan, 'gradient_slope': np.nan, 'gradient_slope_se_hc3': np.nan}
    model_data['log_anatomical_distance'] = np.log1p(model_data['anatomical_distance_min_mm'].to_numpy(float))
    predictors = pd.DataFrame({'anatomical_distance_z': zscore_vector(model_data['log_anatomical_distance'].to_numpy(float))}, index=model_data.index)
    if gradient_distance is not None:
        predictors['gradient_distance_z'] = zscore_vector(model_data[gradient_distance].to_numpy(float))
    design = sm.add_constant(predictors, has_constant='add')
    result = sm.OLS(model_data[outcome].to_numpy(float), design).fit(cov_type='HC3')
    output = {'n_roi': int(result.nobs), 'r_squared': float(result.rsquared), 'anatomical_slope': float(result.params['anatomical_distance_z']), 'anatomical_slope_se_hc3': float(result.bse['anatomical_distance_z'])}
    if gradient_distance is not None:
        output.update({'gradient_slope': float(result.params['gradient_distance_z']), 'gradient_slope_se_hc3': float(result.bse['gradient_distance_z'])})
    return output

def nearest_decile_mean(table: pd.DataFrame, distance: str, outcome: str) -> float:
    valid = table.loc[as_bool(table['intact_roi']), [distance, outcome]].dropna().copy()
    if len(valid) < 20:
        return np.nan
    rank = valid[distance].rank(method='first')
    decile = pd.qcut(rank, 10, labels=False)
    return float(valid.loc[decile == 0, outcome].mean())

def recompute_metrics() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    reference = np.asarray(np.load(REFERENCE_PATH), dtype=float)
    if reference.shape != (400, 5):
        raise ValueError(f'Unexpected reference shape: {reference.shape}')
    old_roi = pd.read_csv(OLD_ROI_PATH, dtype={'subject_id': str})
    old_subject = pd.read_csv(OLD_SUBJECT_PATH, dtype={'subject_id': str})
    old_subject = old_subject.set_index('subject_id', drop=False)
    roi_rows: list[pd.DataFrame] = []
    subject_rows: list[dict] = []
    g2_long_check = pd.read_csv(PROVIDED_DIR / 'provided_code_roi_normalization_long.csv', dtype={'subject_id': str})
    g2_lookup = g2_long_check.set_index(['subject_id', 'roi'])['normalization_G2']
    max_g2_difference = 0.0
    for subject_id in sorted(old_roi['subject_id'].unique(), key=lambda x: int(x)):
        frame = old_roi.loc[old_roi['subject_id'] == subject_id].sort_values('roi').copy()
        if frame['roi'].tolist() != list(range(1, 401)):
            raise ValueError(f'Incomplete ROI order for {subject_id}')
        pre = np.asarray(np.load(ALIGNED_DIR / f'{subject_id}_pre_aligned_z_400x5.npy'), dtype=float)
        post = np.asarray(np.load(ALIGNED_DIR / f'{subject_id}_post_aligned_z_400x5.npy'), dtype=float)
        if pre.shape != (400, 5) or post.shape != (400, 5):
            raise ValueError(f'Unexpected aligned shape for {subject_id}')
        pre_dev_g1 = np.abs(pre[:, 0] - reference[:, 0])
        post_dev_g1 = np.abs(post[:, 0] - reference[:, 0])
        pre_dev_g2 = np.abs(pre[:, 1] - reference[:, 1])
        post_dev_g2 = np.abs(post[:, 1] - reference[:, 1])
        pre_dev_g1g2 = np.linalg.norm(pre[:, :2] - reference[:, :2], axis=1)
        post_dev_g1g2 = np.linalg.norm(post[:, :2] - reference[:, :2], axis=1)
        pre_dev_5d = np.linalg.norm(pre - reference, axis=1)
        post_dev_5d = np.linalg.norm(post - reference, axis=1)
        normalization_g1 = pre_dev_g1 - post_dev_g1
        normalization_g2 = pre_dev_g2 - post_dev_g2
        normalization_g1g2 = pre_dev_g1g2 - post_dev_g1g2
        normalization_5d = pre_dev_5d - post_dev_5d
        reorganization_g1g2 = np.linalg.norm(post[:, :2] - pre[:, :2], axis=1)
        provided_values = np.array([g2_lookup.loc[subject_id, roi] for roi in range(1, 401)], dtype=float)
        max_g2_difference = max(max_g2_difference, float(np.max(np.abs(provided_values - normalization_g2))))
        weights = frame['xtract_projected_roi_weight'].to_numpy(float)
        if np.isfinite(weights).all() and float(weights.sum()) > 0:
            weights = weights / weights.sum()
            lesion_gradient = np.average(reference, axis=0, weights=weights)
        else:
            weights = np.zeros(400)
            lesion_gradient = np.full(5, np.nan)
        xtract_g1_distance = np.abs(reference[:, 0] - lesion_gradient[0])
        xtract_g2_distance = np.abs(reference[:, 1] - lesion_gradient[1])
        xtract_g1g2_distance = np.linalg.norm(reference[:, :2] - lesion_gradient[:2], axis=1)
        frame['G1_reference_z'] = reference[:, 0]
        frame['G2_reference_z'] = reference[:, 1]
        frame['G1_pre_z'] = pre[:, 0]
        frame['G2_pre_z'] = pre[:, 1]
        frame['G1_post_z'] = post[:, 0]
        frame['G2_post_z'] = post[:, 1]
        frame['reference_directed_G1'] = normalization_g1
        frame['reference_directed_G2'] = normalization_g2
        frame['reference_directed_G1G2'] = normalization_g1g2
        frame['reference_directed_5D'] = normalization_5d
        frame['reorganization_magnitude_G1G2'] = reorganization_g1g2
        frame['xtract_projected_G1_distance'] = xtract_g1_distance
        frame['xtract_projected_G2_distance'] = xtract_g2_distance
        frame['xtract_projected_G1G2_distance'] = xtract_g1g2_distance
        frame['baseline_deviation_G1'] = pre_dev_g1
        frame['baseline_deviation_G2'] = pre_dev_g2
        frame['baseline_deviation_G1G2'] = pre_dev_g1g2
        frame['baseline_deviation_5D'] = pre_dev_5d
        roi_rows.append(frame)
        intact = as_bool(frame['intact_roi']).to_numpy()
        anatomy_model = fit_roi_model(frame, 'reference_directed_G1G2', gradient_distance=None)
        old_row = old_subject.loc[subject_id].to_dict()
        old_row.update({'subject_id': subject_id, 'mean_reference_directed_G1': float(normalization_g1[intact].mean()), 'mean_reference_directed_G2': float(normalization_g2[intact].mean()), 'mean_reference_directed_G1G2': float(normalization_g1g2[intact].mean()), 'mean_reference_directed_5D': float(normalization_5d[intact].mean()), 'mean_reorganization_magnitude_G1G2': float(reorganization_g1g2[intact].mean()), 'mean_baseline_deviation_G1': float(pre_dev_g1[intact].mean()), 'mean_baseline_deviation_G2': float(pre_dev_g2[intact].mean()), 'mean_baseline_deviation_G1G2': float(pre_dev_g1g2[intact].mean()), 'mean_baseline_deviation_5D': float(pre_dev_5d[intact].mean()), 'nearest_anatomical_decile_baseline_deviation_G1G2': nearest_decile_mean(frame, 'anatomical_distance_min_mm', 'baseline_deviation_G1G2'), 'nearest_anatomical_decile_reference_directed_G1G2': nearest_decile_mean(frame, 'anatomical_distance_min_mm', 'reference_directed_G1G2'), 'anatomical_G1G2_slope': anatomy_model['anatomical_slope'], 'anatomical_G1G2_slope_se_hc3': anatomy_model['anatomical_slope_se_hc3'], 'anatomical_G1G2_model_r_squared': anatomy_model['r_squared'], 'xtract_projected_lesion_G1': float(lesion_gradient[0]), 'xtract_projected_lesion_G2': float(lesion_gradient[1]), 'nearest_xtract_G1G2_decile_normalization': nearest_decile_mean(frame, 'xtract_projected_G1G2_distance', 'reference_directed_G1G2'), 'nearest_xtract_G1G2_decile_baseline_deviation': nearest_decile_mean(frame, 'xtract_projected_G1G2_distance', 'baseline_deviation_G1G2')})
        for (space, outcome, distance) in [('G1', 'reference_directed_G1', 'xtract_projected_G1_distance'), ('G2', 'reference_directed_G2', 'xtract_projected_G2_distance'), ('G1G2', 'reference_directed_G1G2', 'xtract_projected_G1G2_distance')]:
            result = fit_roi_model(frame, outcome, distance)
            old_row[f'xtract_projected_{space}_slope'] = result['gradient_slope']
            old_row[f'xtract_projected_{space}_slope_se_hc3'] = result['gradient_slope_se_hc3']
            old_row[f'xtract_projected_{space}_model_r_squared'] = result['r_squared']
        subject_rows.append(old_row)
    roi_long = pd.concat(roi_rows, ignore_index=True)
    subject_metrics = pd.DataFrame(subject_rows)
    subject_metrics['subject_id'] = subject_metrics['subject_id'].astype(str)
    qc = {'subject_n': int(subject_metrics['subject_id'].nunique()), 'roi_rows': int(len(roi_long)), 'max_abs_G2_normalization_difference_vs_provided_long_csv': max_g2_difference}
    return (roi_long, subject_metrics, qc)

def freedman_lane_group_test(table: pd.DataFrame, outcome: str, baseline_deviation: str, extra_covariates: list[str], n_permutations: int, rng: np.random.Generator) -> dict[str, float]:
    covariates = ['log1p_lesion_volume', 'mask_lateralization_index', baseline_deviation, 'baseline_mean_fd', 'post_mean_fd', *extra_covariates]
    data = table[[outcome, 'group_tms', *covariates]].dropna().copy()
    y = data[outcome].to_numpy(float)
    reduced_x = sm.add_constant(data[covariates].astype(float), has_constant='add')
    full_x = sm.add_constant(data[['group_tms', *covariates]].astype(float), has_constant='add')
    reduced = sm.OLS(y, reduced_x).fit()
    observed = sm.OLS(y, full_x).fit(cov_type='HC3')
    observed_beta = float(observed.params['group_tms'])
    matrix = full_x.to_numpy(float)
    group_index = list(full_x.columns).index('group_tms')
    beta_projection = np.linalg.pinv(matrix)[group_index]
    fitted = reduced.fittedvalues.to_numpy(float)
    residual = reduced.resid.to_numpy(float)
    permuted = np.empty(n_permutations, dtype=float)
    for index in range(n_permutations):
        y_perm = fitted + residual[rng.permutation(len(residual))]
        permuted[index] = float(beta_projection @ y_perm)
    p_value = float((1 + np.count_nonzero(np.abs(permuted) >= abs(observed_beta))) / (n_permutations + 1))
    confidence = observed.conf_int().loc['group_tms']
    return {'outcome': outcome, 'baseline_deviation_covariate': baseline_deviation, 'extra_covariates': ';'.join(extra_covariates), 'n': int(observed.nobs), 'group_tms_beta': observed_beta, 'ci_low_hc3': float(confidence.iloc[0]), 'ci_high_hc3': float(confidence.iloc[1]), 'se_hc3': float(observed.bse['group_tms']), 'p_hc3': float(observed.pvalues['group_tms']), 'p_freedman_lane': p_value, 'n_permutations': n_permutations, 'r_squared': float(observed.rsquared)}

def run_group_models(subject_metrics: pd.DataFrame) -> pd.DataFrame:
    specs = [('global_normalization_control', 'mean_reference_directed_G1', 'mean_baseline_deviation_G1', []), ('global_normalization_control', 'mean_reference_directed_G2', 'mean_baseline_deviation_G2', []), ('global_normalization_control', 'mean_reference_directed_G1G2', 'mean_baseline_deviation_G1G2', []), ('global_normalization_control', 'mean_reference_directed_5D', 'mean_baseline_deviation_5D', []), ('global_normalization_control', 'mean_reorganization_magnitude_G1G2', 'mean_baseline_deviation_G1G2', []), ('global_normalization_control', 'nearest_anatomical_decile_reference_directed_G1G2', 'nearest_anatomical_decile_baseline_deviation_G1G2', []), ('global_normalization_control', 'anatomical_G1G2_slope', 'mean_baseline_deviation_G1G2', []), ('xtract_lesion_gradient', 'xtract_projected_G1_slope', 'mean_baseline_deviation_G1', ['xtract_projected_lesion_G1', 'xtract_projectable_tract_damage_fraction', 'xtract_projected_effective_roi_n']), ('xtract_lesion_gradient', 'xtract_projected_G2_slope', 'mean_baseline_deviation_G2', ['xtract_projected_lesion_G2', 'xtract_projectable_tract_damage_fraction', 'xtract_projected_effective_roi_n']), ('xtract_lesion_gradient', 'xtract_projected_G1G2_slope', 'mean_baseline_deviation_G1G2', ['xtract_projected_lesion_G1', 'xtract_projected_lesion_G2', 'xtract_projectable_tract_damage_fraction', 'xtract_projected_effective_roi_n']), ('xtract_lesion_gradient', 'nearest_xtract_G1G2_decile_normalization', 'nearest_xtract_G1G2_decile_baseline_deviation', ['xtract_projected_lesion_G1', 'xtract_projected_lesion_G2', 'xtract_projectable_tract_damage_fraction', 'xtract_projected_effective_roi_n'])]
    cohorts = {'primary_motion_qc': subject_metrics.loc[as_bool(subject_metrics['primary_motion_eligible'])], 'all_available_sensitivity': subject_metrics}
    rng = np.random.default_rng(SEED)
    rows: list[dict] = []
    for (cohort_name, cohort) in cohorts.items():
        for (family, outcome, baseline, extra) in specs:
            rows.append({'cohort': cohort_name, 'family': family, **freedman_lane_group_test(cohort, outcome, baseline, extra, N_PERMUTATIONS, rng)})
    models = pd.DataFrame(rows)
    models['fdr_q_hc3'] = np.nan
    models['fdr_q_freedman_lane'] = np.nan
    for ((cohort, family), indices) in models.groupby(['cohort', 'family']).groups.items():
        indices = list(indices)
        models.loc[indices, 'fdr_q_hc3'] = multipletests(models.loc[indices, 'p_hc3'], method='fdr_bh')[1]
        models.loc[indices, 'fdr_q_freedman_lane'] = multipletests(models.loc[indices, 'p_freedman_lane'], method='fdr_bh')[1]
    return models

def residualized_distance_bins(roi: pd.DataFrame, eligible_subjects: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict] = []
    use = roi.loc[roi['subject_id'].astype(str).isin(eligible_subjects) & as_bool(roi['intact_roi'])].copy()
    for (subject_id, frame) in use.groupby('subject_id', sort=True):
        frame = frame.dropna(subset=['reference_directed_G2', 'xtract_projected_G2_distance', 'anatomical_distance_min_mm']).copy()
        y = frame['reference_directed_G2'].to_numpy(float)
        anatomical = frame['anatomical_distance_min_mm'].to_numpy(float)
        design = np.column_stack([np.ones(len(frame)), anatomical])
        fitted = design @ np.linalg.lstsq(design, y, rcond=None)[0]
        frame['G2_normalization_anatomical_adjusted'] = y - fitted + np.mean(y)
        frame['distance_decile'] = pd.qcut(frame['xtract_projected_G2_distance'].rank(method='first'), q=10, labels=False).astype(int) + 1
        group = str(frame['group'].iloc[0])
        for (decile, bin_frame) in frame.groupby('distance_decile'):
            records.append({'subject_id': str(subject_id), 'group': group, 'distance_decile': int(decile), 'mean_G2_normalization_anatomical_adjusted': float(bin_frame['G2_normalization_anatomical_adjusted'].mean()), 'mean_xtract_G2_distance': float(bin_frame['xtract_projected_G2_distance'].mean()), 'roi_n': int(len(bin_frame))})
    subject_bins = pd.DataFrame.from_records(records)
    summary_rows: list[dict] = []
    for ((group, decile), frame) in subject_bins.groupby(['group', 'distance_decile']):
        (mean, low, high) = mean_ci(frame['mean_G2_normalization_anatomical_adjusted'].to_numpy(float))
        summary_rows.append({'group': group, 'distance_decile': int(decile), 'mean': mean, 'ci_low': low, 'ci_high': high, 'n_subjects': int(frame['subject_id'].nunique())})
    return (subject_bins, pd.DataFrame(summary_rows))

def exact_paired_signflip(differences: np.ndarray) -> dict[str, float]:
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    n = len(differences)
    observed = float(np.mean(differences))
    null = np.fromiter((float(np.mean(differences * np.asarray(signs))) for signs in itertools.product((-1.0, 1.0), repeat=n)), dtype=float, count=2 ** n)
    p_value = float(np.mean(np.abs(null) >= abs(observed) - 1e-15))
    sd = float(np.std(differences, ddof=1))
    se = sd / np.sqrt(n)
    critical = float(t.ppf(0.975, df=n - 1))
    return {'n_pairs': n, 'mean_tms_minus_control': observed, 'sd_pair_difference': sd, 'se_pair_difference': se, 'ci_low_t': observed - critical * se, 'ci_high_t': observed + critical * se, 'exact_signflip_p': p_value, 'exact_permutations': 2 ** n}

def run_matched_tests(subject_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = pd.read_csv(MATCHED_PAIRS_PATH, dtype={'control_subject_id': str, 'tms_subject_id': str})
    metric_lookup = subject_metrics.set_index('subject_id')
    motion_ids = set(subject_metrics.loc[as_bool(subject_metrics['primary_motion_eligible']), 'subject_id'].astype(str))
    motion_pairs = pairs['control_subject_id'].isin(motion_ids) & pairs['tms_subject_id'].isin(motion_ids)
    sets = {'all_pairs': pairs, 'tight_pairs': pairs.loc[pairs['standardized_rms_distance'] < 1.0], 'motion_qc_pairs': pairs.loc[motion_pairs], 'motion_qc_tight_pairs': pairs.loc[motion_pairs & (pairs['standardized_rms_distance'] < 1.0)]}
    outcomes = ['xtract_projected_G1_slope', 'xtract_projected_G2_slope', 'xtract_projected_G1G2_slope', 'nearest_xtract_G1G2_decile_normalization']
    test_rows: list[dict] = []
    pair_rows: list[dict] = []
    for (set_name, set_table) in sets.items():
        for outcome in outcomes:
            differences = []
            for pair in set_table.itertuples():
                control = float(metric_lookup.loc[pair.control_subject_id, outcome])
                tms_value = float(metric_lookup.loc[pair.tms_subject_id, outcome])
                difference = tms_value - control
                differences.append(difference)
                pair_rows.append({'match_set': set_name, 'pair_id': int(pair.pair_id), 'control_subject_id': pair.control_subject_id, 'tms_subject_id': pair.tms_subject_id, 'standardized_rms_distance': float(pair.standardized_rms_distance), 'outcome': outcome, 'control_value': control, 'tms_value': tms_value, 'tms_minus_control': difference})
            test_rows.append({'match_set': set_name, 'outcome': outcome, **exact_paired_signflip(np.asarray(differences))})
    tests = pd.DataFrame(test_rows)
    tests['fdr_q_within_match_set'] = np.nan
    for (set_name, indices) in tests.groupby('match_set').groups.items():
        indices = list(indices)
        tests.loc[indices, 'fdr_q_within_match_set'] = multipletests(tests.loc[indices, 'exact_signflip_p'], method='fdr_bh')[1]
    return (tests, pd.DataFrame(pair_rows))

def violin_points(ax: plt.Axes, values: np.ndarray, position: float, color: str, rng: np.random.Generator, width: float=0.28) -> None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    violin = ax.violinplot(values, positions=[position], widths=width * 1.7, showmeans=False, showmedians=False, showextrema=False, points=300)
    for body in violin['bodies']:
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.18)
        body.set_linewidth(0.6)
    jitter = rng.uniform(-width * 0.35, width * 0.35, size=len(values))
    ax.scatter(position + jitter, values, s=13, color=color, edgecolor='white', linewidth=0.35, alpha=0.88, zorder=3)
    (mean, low, high) = mean_ci(values)
    ax.plot([position, position], [low, high], color=BLACK, lw=0.8, zorder=5)
    ax.plot([position - 0.1, position + 0.1], [mean, mean], color=BLACK, lw=1.6, zorder=6)

def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.13, 1.04, label, transform=ax.transAxes, fontsize=9, fontweight='bold', ha='left', va='bottom')

def plot_panel_b(ax: plt.Axes, primary: pd.DataFrame, models: pd.DataFrame, rng: np.random.Generator, panel_label: bool) -> None:
    specs = [('mean_reference_directed_G2', 'G2'), ('mean_reference_directed_G1G2', 'G1–G2')]
    for (metric_index, (metric, _)) in enumerate(specs):
        for (group, offset, color) in [('Control', -0.17, CONTROL), ('TMS', 0.17, TMS)]:
            violin_points(ax, primary.loc[primary['group'] == group, metric].to_numpy(float), metric_index + offset, color, rng, width=0.26)
    ax.axhline(0, color=ZERO, lw=0.7, zorder=0)
    ax.set_xticks([0, 1], [label for (_, label) in specs])
    ax.set_ylabel('Healthy-reference normalization')
    ax.set_title('Greater whole-cortex normalization in TMS', loc='left', fontweight='bold', pad=3)
    values = primary[[metric for (metric, _) in specs]].to_numpy(float)
    (ymin, ymax) = (float(np.nanmin(values)), float(np.nanmax(values)))
    span = ymax - ymin
    ax.set_ylim(ymin - 0.08 * span, ymax + 0.3 * span)
    model_lookup = models.set_index('outcome')
    for (index, (metric, _)) in enumerate(specs):
        row = model_lookup.loc[metric]
        ax.text(index, ymax + 0.13 * span, f"adjusted beta={row['group_tms_beta']:.3f}\nq={p_text(row['fdr_q_freedman_lane'])}", ha='center', va='bottom', fontsize=5.7)
    if panel_label:
        add_panel_label(ax, 'b')

def plot_panel_c(ax: plt.Axes, summary: pd.DataFrame, panel_label: bool) -> None:
    for (group, color) in [('Control', CONTROL), ('TMS', TMS)]:
        frame = summary.loc[summary['group'] == group].sort_values('distance_decile')
        x = frame['distance_decile'].to_numpy(float)
        mean = frame['mean'].to_numpy(float)
        low = frame['ci_low'].to_numpy(float)
        high = frame['ci_high'].to_numpy(float)
        ax.fill_between(x, low, high, color=color, alpha=0.16, linewidth=0)
        ax.plot(x, mean, color=color, lw=1.65, marker='o', ms=3.1, label=group)
    ax.axhline(0, color=ZERO, lw=0.7)
    ax.set_xticks([1, 3, 5, 7, 10])
    ax.set_xlabel('G2 gradient-distance decile\nlesion-similar  →  lesion-dissimilar')
    ax.set_ylabel('Adjusted G2 normalization')
    ax.set_title('G2 normalization across lesion-related functional distance', loc='left', fontweight='bold', pad=3)
    ax.legend(loc='upper left', ncol=2, fontsize=5.8, handlelength=1.4)
    if panel_label:
        add_panel_label(ax, 'c')

def plot_panel_d(ax: plt.Axes, primary: pd.DataFrame, models: pd.DataFrame, rng: np.random.Generator, panel_label: bool) -> None:
    metric = 'xtract_projected_G2_slope'
    for (position, (group, color)) in enumerate([('Control', CONTROL), ('TMS', TMS)]):
        violin_points(ax, primary.loc[primary['group'] == group, metric].to_numpy(float), position, color, rng, width=0.28)
    ax.axhline(0, color=ZERO, lw=0.7)
    ax.set_xticks([0, 1], ['Control', 'TMS'])
    ax.set_ylabel('Individual G2 normalization–distance slope')
    ax.set_title('Individual G2 distance–normalization slopes', loc='left', fontweight='bold', pad=3)
    row = models.set_index('outcome').loc[metric]
    ax.text(0.5, 0.97, f"adjusted beta={row['group_tms_beta']:.3f}\npermutation p={p_text(row['p_freedman_lane'])}; q={p_text(row['fdr_q_freedman_lane'])}", transform=ax.transAxes, ha='center', va='top', fontsize=5.5)
    if panel_label:
        add_panel_label(ax, 'd')

def plot_panel_f(ax: plt.Axes, matched_pairs: pd.DataFrame, matched_tests: pd.DataFrame, panel_label: bool) -> None:
    pairs = matched_pairs.loc[(matched_pairs['match_set'] == 'motion_qc_pairs') & (matched_pairs['outcome'] == 'nearest_xtract_G1G2_decile_normalization')].sort_values('pair_id')
    rng = np.random.default_rng(20260822)
    control_position = 0.2
    tms_position = 0.8
    control_x: list[float] = []
    tms_x: list[float] = []
    for pair in pairs.itertuples():
        jitter = float(rng.uniform(-0.035, 0.035))
        control_x.append(control_position + jitter)
        tms_x.append(tms_position + jitter)
        ax.plot([control_position + jitter, tms_position + jitter], [pair.control_value, pair.tms_value], color='#A9ADB2', lw=0.82, alpha=0.52, zorder=1)
    ax.scatter(control_x, pairs['control_value'], color=CONTROL, s=30, edgecolor='none', lw=0.0, alpha=0.94, zorder=3)
    ax.scatter(tms_x, pairs['tms_value'], color=TMS, s=30, edgecolor='none', lw=0.0, alpha=0.94, zorder=3)
    ax.axhline(0, color=ZERO, lw=0.7)
    ax.set_xlim(-0.2, 1.2)
    ax.set_xticks([control_position, tms_position], ['Control', 'TMS'])
    ax.set_ylabel('G1–G2 normalization\n(lesion-similar intact regions)')
    row = matched_tests.loc[(matched_tests['match_set'] == 'motion_qc_pairs') & (matched_tests['outcome'] == 'nearest_xtract_G1G2_decile_normalization')].iloc[0]
    ax.set_title(f'Greater G1–G2 normalization after lesion matching ({len(pairs)} pairs)', loc='left', fontweight='bold', pad=3)
    values = np.concatenate([pairs['control_value'].to_numpy(float), pairs['tms_value'].to_numpy(float)])
    data_min = float(np.min(values))
    data_max = float(np.max(values))
    span = data_max - data_min
    bracket = data_max + 0.105 * span
    cap = bracket - 0.025 * span
    ax.plot([0.12, 0.12, 0.88, 0.88], [cap, bracket, bracket, cap], color=BLACK, lw=0.78, clip_on=False, zorder=5)
    q_value = float(row['fdr_q_within_match_set'])
    stars = '***' if q_value < 0.001 else '**' if q_value < 0.01 else '*' if q_value < 0.05 else 'ns'
    ax.text(0.5, bracket + 0.025 * span, stars, ha='center', va='bottom', fontsize=8.0, fontweight='bold', color=BLACK)
    ax.set_ylim(data_min - 0.055 * span, data_max + 0.245 * span)
    if panel_label:
        add_panel_label(ax, 'f')

def save_bundle(fig: plt.Figure, stem: str) -> None:
    base = OUT_DIR / stem
    tiff_path = base.with_suffix('.tiff')
    fig.savefig(base.with_suffix('.pdf'), bbox_inches='tight', pad_inches=0.04)
    fig.savefig(base.with_suffix('.svg'), bbox_inches='tight', pad_inches=0.04)
    fig.savefig(base.with_suffix('.png'), dpi=600, bbox_inches='tight', pad_inches=0.04)
    fig.savefig(tiff_path, dpi=600, bbox_inches='tight', pad_inches=0.04, pil_kwargs={'compression': 'tiff_lzw'})
    with Image.open(tiff_path) as image:
        rgb = image.convert('RGB')
    rgb.save(tiff_path, compression='tiff_lzw', dpi=(600, 600))
    plt.close(fig)

def make_figures(subject_metrics: pd.DataFrame, models: pd.DataFrame, bin_summary: pd.DataFrame, matched_tests: pd.DataFrame, matched_pairs: pd.DataFrame) -> None:
    primary = subject_metrics.loc[as_bool(subject_metrics['primary_motion_eligible'])].copy()
    primary_models = models.loc[models['cohort'] == 'primary_motion_qc'].copy()
    rng = np.random.default_rng(20260811)
    (fig, axes) = plt.subplots(2, 2, figsize=(7.2, 5.05))
    plot_panel_b(axes[0, 0], primary, primary_models, rng, panel_label=True)
    plot_panel_c(axes[0, 1], bin_summary, panel_label=True)
    plot_panel_d(axes[1, 0], primary, primary_models, rng, panel_label=True)
    plot_panel_f(axes[1, 1], matched_pairs, matched_tests, panel_label=True)
    fig.text(0.5, 0.006, 'All Pre/Post gradients recomputed and aligned with the supplied fixed healthy reference', ha='center', va='bottom', fontsize=5.7, color=NEUTRAL)
    fig.tight_layout(rect=[0.015, 0.025, 0.995, 0.995], h_pad=1.8, w_pad=2.0)
    save_bundle(fig, 'panels_b_c_d_f_recomputed_provided_reference')
    standalone_specs = [('b', plot_panel_b), ('c', plot_panel_c), ('d', plot_panel_d), ('f', plot_panel_f)]
    for (label, function) in standalone_specs:
        (fig, ax) = plt.subplots(figsize=(3.65, 3.05))
        if label == 'b':
            function(ax, primary, primary_models, np.random.default_rng(20260811), False)
        elif label == 'c':
            function(ax, bin_summary, False)
        elif label == 'd':
            function(ax, primary, primary_models, np.random.default_rng(20260812), False)
        else:
            function(ax, matched_pairs, matched_tests, False)
        fig.tight_layout(pad=0.7)
        save_bundle(fig, f'panel_{label}_recomputed_provided_reference')

def compare_old_new(subject_metrics: pd.DataFrame, models: pd.DataFrame) -> pd.DataFrame:
    old_subject = pd.read_csv(OLD_SUBJECT_PATH, dtype={'subject_id': str})
    merged = old_subject.merge(subject_metrics, on='subject_id', suffixes=('_old', '_new'), validate='one_to_one')
    metrics = ['mean_reference_directed_G2', 'mean_reference_directed_G1G2', 'xtract_projected_G2_slope', 'nearest_xtract_G1G2_decile_normalization']
    rows = []
    for metric in metrics:
        x = merged[f'{metric}_old'].to_numpy(float)
        y = merged[f'{metric}_new'].to_numpy(float)
        rows.append({'level': 'subject_metric', 'metric': metric, 'old_value': float(np.mean(x)), 'new_value': float(np.mean(y)), 'old_new_correlation': float(np.corrcoef(x, y)[0, 1])})
    old_models = pd.read_csv(OLD_GROUP_MODEL_PATH)
    old_primary = old_models.loc[old_models['cohort'] == 'primary_motion_qc'].set_index('outcome')
    new_primary = models.loc[models['cohort'] == 'primary_motion_qc'].set_index('outcome')
    for metric in ['mean_reference_directed_G2', 'mean_reference_directed_G1G2', 'xtract_projected_G2_slope', 'nearest_xtract_G1G2_decile_normalization']:
        rows.append({'level': 'adjusted_group_model', 'metric': metric, 'old_value': float(old_primary.loc[metric, 'group_tms_beta']), 'new_value': float(new_primary.loc[metric, 'group_tms_beta']), 'old_new_correlation': np.nan})
    return pd.DataFrame(rows)

def write_notes(qc: dict[str, float], subject_metrics: pd.DataFrame, models: pd.DataFrame, matched_tests: pd.DataFrame) -> None:
    primary = subject_metrics.loc[as_bool(subject_metrics['primary_motion_eligible'])]
    model_lookup = models.loc[models['cohort'] == 'primary_motion_qc'].set_index('outcome')
    matched = matched_tests.loc[(matched_tests['match_set'] == 'motion_qc_pairs') & (matched_tests['outcome'] == 'nearest_xtract_G1G2_decile_normalization')].iloc[0]
    summary = {'reference': 'supplied gradient_calculation pipeline fixed healthy 400x5 z reference', 'primary_cohort': {group: int(frame['subject_id'].nunique()) for (group, frame) in primary.groupby('group')}, 'panel_b_G2': model_lookup.loc['mean_reference_directed_G2'].to_dict(), 'panel_b_G1G2': model_lookup.loc['mean_reference_directed_G1G2'].to_dict(), 'panel_d_G2_slope': model_lookup.loc['xtract_projected_G2_slope'].to_dict(), 'panel_f_motion_qc_pairs': matched.to_dict(), 'qc': qc}
    (OUT_DIR / 'analysis_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding='utf-8')
    b_g2 = model_lookup.loc['mean_reference_directed_G2']
    b_g1g2 = model_lookup.loc['mean_reference_directed_G1G2']
    d = model_lookup.loc['xtract_projected_G2_slope']
    notes = f"# 使用所给梯度代码健康参考重新计算 b/c/d/f\n\n## 统一计算口径\n\n所有患者 Pre/Post 五维梯度均来自所给梯度代码产生并对齐到同一个固定健康参考的结果。\n每个 ROI 的 normalization 定义为：\n\n`normalization = distance(Pre, healthy) - distance(Post, healthy)`\n\n正值表示 Post 比 Pre 更接近健康参考。XTRACT投影权重和解剖距离不依赖梯度参考，\n因此保留；病灶的G1/G2位置及所有gradient distance均使用新参考重新计算。\n\n## 主要结果\n\n- b / 全皮层G2：调整后TMS beta={b_g2['group_tms_beta']:.4f}，\n  置换p={b_g2['p_freedman_lane']:.4f}，家族内q={b_g2['fdr_q_freedman_lane']:.4f}。\n- b / 全皮层G1–G2：beta={b_g1g2['group_tms_beta']:.4f}，\n  p={b_g1g2['p_freedman_lane']:.4f}，q={b_g1g2['fdr_q_freedman_lane']:.4f}。\n- d / G2距离斜率：beta={d['group_tms_beta']:.4f}，\n  p={d['p_freedman_lane']:.4f}，q={d['fdr_q_freedman_lane']:.4f}。\n- f / 运动质控10对最近G1–G2十分位：平均TMS-Control差=\n  {matched['mean_tms_minus_control']:.4f}，精确符号翻转p=\n  {matched['exact_signflip_p']:.4f}，四结局组内q=\n  {matched['fdr_q_within_match_set']:.4f}。\n\n## 推断说明\n\n- b、d的调整模型控制log病灶体积、侧化、相应baseline偏离及Pre/Post头动；\n  d还控制投影病灶G2位置、可投影束比例和有效皮层终点数。\n- c为描述性分箱曲线，正式推断对应d的受试者层面斜率。\n- f的灰线连接匹配的不同患者，不是同一患者的Pre/Post。\n- 这些仍是观察性调整后组间比较，不等同于无混杂因果效应。\n"
    (OUT_DIR / 'RESULTS_ZH.md').write_text(notes, encoding='utf-8')

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (roi_long, subject_metrics, qc) = recompute_metrics()
    models = run_group_models(subject_metrics)
    primary_ids = set(subject_metrics.loc[as_bool(subject_metrics['primary_motion_eligible']), 'subject_id'].astype(str))
    (subject_bins, bin_summary) = residualized_distance_bins(roi_long, primary_ids)
    (matched_tests, matched_pairs) = run_matched_tests(subject_metrics)
    comparison = compare_old_new(subject_metrics, models)
    roi_long.to_csv(OUT_DIR / 'recomputed_roi_metrics.csv', index=False)
    subject_metrics.to_csv(OUT_DIR / 'recomputed_subject_metrics.csv', index=False)
    models.to_csv(OUT_DIR / 'recomputed_group_models.csv', index=False)
    subject_bins.to_csv(OUT_DIR / 'panel_c_subject_distance_bins.csv', index=False)
    bin_summary.to_csv(OUT_DIR / 'panel_c_group_distance_bin_summary.csv', index=False)
    matched_tests.to_csv(OUT_DIR / 'recomputed_matched_tests.csv', index=False)
    matched_pairs.to_csv(OUT_DIR / 'recomputed_matched_pair_values.csv', index=False)
    comparison.to_csv(OUT_DIR / 'old_vs_new_reference_comparison.csv', index=False)
    make_figures(subject_metrics, models, bin_summary, matched_tests, matched_pairs)
    write_notes(qc, subject_metrics, models, matched_tests)
    print('QC')
    print(json.dumps(qc, indent=2))
    print('\nPrimary panel models')
    print(models.loc[(models['cohort'] == 'primary_motion_qc') & models['outcome'].isin(['mean_reference_directed_G2', 'mean_reference_directed_G1G2', 'xtract_projected_G2_slope', 'nearest_xtract_G1G2_decile_normalization'])][['outcome', 'n', 'group_tms_beta', 'ci_low_hc3', 'ci_high_hc3', 'p_freedman_lane', 'fdr_q_freedman_lane']].to_string(index=False))
    print('\nMatched panel f')
    print(matched_tests.loc[(matched_tests['match_set'] == 'motion_qc_pairs') & (matched_tests['outcome'] == 'nearest_xtract_G1G2_decile_normalization')].to_string(index=False))
    print(f'\nOutputs: {OUT_DIR}')
if __name__ == '__main__':
    main()
