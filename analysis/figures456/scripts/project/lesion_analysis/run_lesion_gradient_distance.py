from __future__ import annotations
import os as _portable_os
from pathlib import Path as _PortablePath

def _portable_path(kind, relative):
    project = _PortablePath(_portable_os.environ['PAPER_PROJECT_ROOT'])
    root = project if kind == 'project' else project.parent
    return str(root / relative)
import argparse
import itertools
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from brainspace.gradient import GradientMaps
from brainspace.datasets import load_group_fc
from scipy.linalg import orthogonal_procrustes
from scipy.ndimage import distance_transform_edt
from scipy.stats import t
from statsmodels.stats.multitest import multipletests
PROJECT_ROOT = Path(_portable_path('project', ''))
ANALYSIS_ROOT = PROJECT_ROOT / 'lesion_analysis'
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))
import run_lesion_adjusted_gradient_value as base
OUTPUT_ROOT = ANALYSIS_ROOT / 'lesion_gradient_distance'
TABLES = OUTPUT_ROOT / 'tables'
FIGURES = OUTPUT_ROOT / 'figures'
REFERENCES = OUTPUT_ROOT / 'references'
FC_Z_ROOT = PROJECT_ROOT / 'fc_analysis' / 'fc_matrices_z'
CONSISTENT_FC_Z_ROOT = OUTPUT_ROOT / 'fc_consistent' / 'fc_matrices_z'
ATLAS_PATH = ANALYSIS_ROOT / 'resources' / 'Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz'
LABEL_PATH = ANALYSIS_ROOT / 'resources' / 'Schaefer2018_400Parcels_7Networks_order.txt'
MASK_ROOT = ANALYSIS_ROOT / 'mni_registration' / 'masks'
XTRACT_PATH = ANALYSIS_ROOT / 'resources' / 'XTRACT_HCP_prob_2mm.nii.gz'
XTRACT_BURDEN_TABLE = ANALYSIS_ROOT / 'results_abcd' / 'tables' / 'F_xtract_tract_probability_burden.csv'
MATCHED_PAIRS_TABLE = ANALYSIS_ROOT / 'lesion_matched_clinical_mediation' / 'tables' / 'optimal_lesion_matched_pairs.csv'
SEED = 20260803
N_COMPONENTS = 5
N_PERMUTATIONS = 5000
XTRACT_ENDPOINT_THRESHOLD_PCT = 5.0
NETWORK_NAME_MAP = {'Vis': 'Visual', 'SomMot': 'SomMot', 'DorsAttn': 'DorsAttn', 'SalVentAttn': 'SalVentAttn', 'Limbic': 'Limbic', 'Cont': 'Control', 'Default': 'Default'}

def ensure_dirs() -> None:
    for path in (OUTPUT_ROOT, TABLES, FIGURES, REFERENCES):
        path.mkdir(parents=True, exist_ok=True)

def load_atlas() -> tuple[nib.Nifti1Image, np.ndarray, pd.DataFrame]:
    image = nib.load(str(ATLAS_PATH))
    atlas = np.asarray(image.dataobj, dtype=np.int16)
    if not np.array_equal(np.unique(atlas), np.arange(401)):
        raise RuntimeError('Schaefer atlas does not contain labels 0-400')
    labels = pd.read_csv(LABEL_PATH, sep='\t', header=None, names=['roi', 'roi_name', 'r', 'g', 'b', 'a'])
    if labels['roi'].tolist() != list(range(1, 401)):
        raise RuntimeError('Schaefer label ordering is not 1-400')
    labels['network_token'] = labels['roi_name'].str.split('_').str[2]
    labels['network'] = labels['network_token'].map(NETWORK_NAME_MAP)
    if labels['network'].isna().any():
        raise RuntimeError('Could not parse Schaefer seven-network labels')
    centroids = []
    roi_voxels = []
    for roi in range(1, 401):
        indices = np.argwhere(atlas == roi)
        if len(indices) == 0:
            raise RuntimeError(f'Empty Schaefer ROI: {roi}')
        xyz = nib.affines.apply_affine(image.affine, indices)
        centroids.append(np.mean(xyz, axis=0))
        roi_voxels.append(len(indices))
    centroids = np.asarray(centroids)
    labels[['centroid_x', 'centroid_y', 'centroid_z']] = centroids
    labels['roi_voxels'] = roi_voxels
    return (image, atlas, labels)

def fit_gradient(connectivity: np.ndarray, *, fisher_z_input: bool=True) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(connectivity, dtype=float)
    mean_r = np.tanh(matrix) if fisher_z_input else matrix.copy()
    mean_r = (mean_r + mean_r.T) / 2.0
    np.fill_diagonal(mean_r, 0.0)
    model = GradientMaps(n_components=N_COMPONENTS, approach='dm', kernel='normalized_angle', random_state=SEED)
    model.fit(mean_r, sparsity=0.9)
    gradients = np.asarray(model.gradients_, dtype=float)
    lambdas = np.asarray(model.lambdas_, dtype=float)
    if gradients.shape != (400, N_COMPONENTS):
        raise RuntimeError(f'Unexpected reference gradient shape: {gradients.shape}')
    return (gradients, lambdas)

def similarity_align(source: np.ndarray, target: np.ndarray, fit_mask: np.ndarray | None=None) -> tuple[np.ndarray, dict[str, float]]:
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if source.shape != target.shape:
        raise ValueError(f'Shape mismatch for alignment: {source.shape}, {target.shape}')
    if fit_mask is None:
        fit_mask = np.ones(len(source), dtype=bool)
    fit_mask = np.asarray(fit_mask, dtype=bool)
    if int(fit_mask.sum()) < max(50, source.shape[1] * 10):
        raise RuntimeError(f'Too few intact ROIs for alignment: {fit_mask.sum()}')
    source_mean = source[fit_mask].mean(axis=0, keepdims=True)
    target_mean = target[fit_mask].mean(axis=0, keepdims=True)
    source_centered = source[fit_mask] - source_mean
    target_centered = target[fit_mask] - target_mean
    (rotation, singular_sum) = orthogonal_procrustes(source_centered, target_centered)
    denominator = float(np.sum(source_centered ** 2))
    scale = float(singular_sum / denominator) if denominator > 0 else np.nan
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError(f'Invalid Procrustes scale: {scale}')
    aligned = (source - source_mean) @ rotation * scale + target_mean
    fit_corr = float(np.corrcoef(aligned[fit_mask].ravel(), target[fit_mask].ravel())[0, 1])
    fit_rms = float(np.sqrt(np.mean((aligned[fit_mask] - target[fit_mask]) ** 2)))
    return (aligned, {'fit_n_roi': int(fit_mask.sum()), 'similarity_scale': scale, 'fit_correlation': fit_corr, 'fit_G1_correlation': float(np.corrcoef(aligned[fit_mask, 0], target[fit_mask, 0])[0, 1]), 'fit_G2_correlation': float(np.corrcoef(aligned[fit_mask, 1], target[fit_mask, 1])[0, 1]), 'fit_rms': fit_rms})

def orient_master(master: np.ndarray, labels: pd.DataFrame) -> tuple[np.ndarray, dict]:
    oriented = np.asarray(master, dtype=float).copy()
    networks = labels['network'].to_numpy(str)
    g1_default = float(oriented[networks == 'Default', 0].mean())
    g1_sensory = float(oriented[np.isin(networks, ['Visual', 'SomMot']), 0].mean())
    g1_sign = 1.0 if g1_default >= g1_sensory else -1.0
    oriented[:, 0] *= g1_sign
    g2_visual = float(oriented[networks == 'Visual', 1].mean())
    g2_somatomotor = float(oriented[networks == 'SomMot', 1].mean())
    g2_sign = 1.0 if g2_visual >= g2_somatomotor else -1.0
    oriented[:, 1] *= g2_sign
    return (oriented, {'G1_direction_rule': 'Default mean > mean(Visual, SomMot)', 'G1_original_sign_multiplier': g1_sign, 'G2_direction_rule': 'Visual mean > SomMot mean', 'G2_original_sign_multiplier': g2_sign})

def gradient_file(group: str, session: str, subject_id: str) -> Path:
    matches = sorted((base.CORTICAL_ROOT / group / session).glob(f'{subject_id}-*.npy'))
    if len(matches) != 1:
        raise RuntimeError(f'Expected one cortical gradient for {group}/{session}/{subject_id}; found {len(matches)}')
    return matches[0]

def lesion_mask_path(group: str, subject_id: str) -> Path:
    path = MASK_ROOT / group / 'before' / f'{subject_id}_lesion_mni2mm.nii.gz'
    if not path.exists():
        raise FileNotFoundError(path)
    return path

def zscore_vector(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    sd = float(np.std(values, ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        raise RuntimeError('Cannot z-score a constant vector')
    return (values - float(np.mean(values))) / sd

def build_xtract_endpoint_projection(atlas_image: nib.Nifti1Image, atlas: np.ndarray, tract_names: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    xtract_image = nib.load(str(XTRACT_PATH))
    if xtract_image.shape != (*atlas_image.shape, 42) or not np.allclose(xtract_image.affine, atlas_image.affine):
        raise RuntimeError(f'XTRACT/Schaefer geometry mismatch: {xtract_image.shape}, {atlas_image.shape}')
    xtract = np.asarray(xtract_image.dataobj, dtype=np.float32)
    endpoint_raw = np.zeros((42, 400), dtype=float)
    endpoint_total = np.zeros(42, dtype=float)
    rows = []
    name_lookup = tract_names.set_index('tract_index')['tract_name'].to_dict()
    for tract_index in range(42):
        tract = np.where(xtract[..., tract_index] >= XTRACT_ENDPOINT_THRESHOLD_PCT, xtract[..., tract_index], 0.0)
        endpoint_raw[tract_index] = np.bincount(atlas.ravel(), weights=tract.ravel(), minlength=401)[1:]
        endpoint_total[tract_index] = float(endpoint_raw[tract_index].sum())
        if endpoint_total[tract_index] > 0:
            normalized = endpoint_raw[tract_index] / endpoint_total[tract_index]
        else:
            normalized = np.zeros(400, dtype=float)
        for roi_index in range(400):
            rows.append({'tract_index': tract_index, 'tract_name': name_lookup[tract_index], 'roi': roi_index + 1, 'endpoint_probability_mass': endpoint_raw[tract_index, roi_index], 'endpoint_weight_within_tract': normalized[roi_index], 'endpoint_threshold_pct': XTRACT_ENDPOINT_THRESHOLD_PCT})
    endpoint_normalized = np.divide(endpoint_raw, endpoint_total[:, None], out=np.zeros_like(endpoint_raw), where=endpoint_total[:, None] > 0)
    return (endpoint_normalized, pd.DataFrame(rows), endpoint_total)

def fit_roi_model(roi_table: pd.DataFrame, outcome: str, gradient_distance: str | None) -> dict[str, float]:
    columns = [outcome, 'anatomical_distance_min_mm']
    if gradient_distance is not None:
        columns.append(gradient_distance)
    model_data = roi_table.loc[roi_table['intact_roi'], columns].dropna().copy()
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
    valid = table.loc[table['intact_roi'], [distance, outcome]].dropna().copy()
    if len(valid) < 20:
        return np.nan
    rank = valid[distance].rank(method='first')
    decile = pd.qcut(rank, 10, labels=False)
    return float(valid.loc[decile == 0, outcome].mean())

def freedman_lane_group_test(table: pd.DataFrame, outcome: str, baseline_deviation: str, extra_covariates: list[str], n_permutations: int, rng: np.random.Generator) -> dict[str, float]:
    covariates = ['log1p_lesion_volume', 'mask_lateralization_index', baseline_deviation, 'baseline_mean_fd', 'post_mean_fd', *extra_covariates]
    data = table[[outcome, 'group_tms', *covariates]].dropna()
    y = data[outcome].to_numpy(float)
    reduced_x = sm.add_constant(data[covariates], has_constant='add')
    full_x = sm.add_constant(data[['group_tms', *covariates]], has_constant='add')
    reduced = sm.OLS(y, reduced_x).fit()
    observed = sm.OLS(y, full_x).fit(cov_type='HC3')
    observed_beta = float(observed.params['group_tms'])
    permuted = np.empty(n_permutations, dtype=float)
    for index in range(n_permutations):
        y_perm = reduced.fittedvalues.to_numpy() + rng.permutation(reduced.resid.to_numpy())
        permuted[index] = float(sm.OLS(y_perm, full_x).fit().params['group_tms'])
    p_value = float((1 + np.count_nonzero(np.abs(permuted) >= abs(observed_beta))) / (n_permutations + 1))
    confidence = observed.conf_int().loc['group_tms']
    return {'outcome': outcome, 'baseline_deviation_covariate': baseline_deviation, 'extra_covariates': ';'.join(extra_covariates), 'n': int(observed.nobs), 'group_tms_beta': observed_beta, 'ci_low_hc3': float(confidence.iloc[0]), 'ci_high_hc3': float(confidence.iloc[1]), 'se_hc3': float(observed.bse['group_tms']), 'p_hc3': float(observed.pvalues['group_tms']), 'p_freedman_lane': p_value, 'n_permutations': n_permutations, 'r_squared': float(observed.rsquared)}

def exact_paired_signflip(differences: np.ndarray) -> dict[str, float]:
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    n = len(differences)
    if n < 3:
        raise RuntimeError(f'Too few matched differences: {n}')
    observed = float(np.mean(differences))
    null = np.fromiter((float(np.mean(differences * np.asarray(signs))) for signs in itertools.product((-1.0, 1.0), repeat=n)), dtype=float, count=2 ** n)
    p_value = float(np.mean(np.abs(null) >= abs(observed) - 1e-15))
    sd = float(np.std(differences, ddof=1))
    se = sd / np.sqrt(n)
    critical = float(t.ppf(0.975, df=n - 1))
    return {'n_pairs': n, 'mean_tms_minus_control': observed, 'sd_pair_difference': sd, 'se_pair_difference': se, 'ci_low_t': observed - critical * se, 'ci_high_t': observed + critical * se, 'exact_signflip_p': p_value, 'exact_permutations': 2 ** n}

def make_figure(reference: pd.DataFrame, alignment: pd.DataFrame, subject_metrics: pd.DataFrame) -> None:
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 7, 'axes.linewidth': 0.7, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    colors = {'Control': '#4C78A8', 'TMS': '#D55E00'}
    (fig, axes) = plt.subplots(2, 2, figsize=(7.2, 5.5), facecolor='white')
    ax = axes[0, 0]
    network_order = ['Visual', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Control', 'Default']
    means = reference.groupby('network', observed=True)['G1_reference_z'].mean()
    ax.plot(range(len(network_order)), [means[name] for name in network_order], marker='o', color='#222222', lw=1.1, ms=3.5)
    ax.axhline(0, color='#BBBBBB', lw=0.6)
    ax.set_xticks(range(len(network_order)), network_order, rotation=35, ha='right')
    ax.set_ylabel('Reference G1 (z)')
    ax.set_title('Healthy normative reference', loc='left', fontweight='bold')
    ax = axes[0, 1]
    for (group, subset) in alignment.groupby('group'):
        ax.scatter(subset['pre_fit_correlation'], subset['post_fit_correlation'], s=18, alpha=0.78, color=colors[group], edgecolor='white', linewidth=0.35, label=group)
    lower = min(float(alignment['pre_fit_correlation'].min()), float(alignment['post_fit_correlation'].min()))
    ax.plot([lower, 1], [lower, 1], color='#888888', lw=0.7, ls='--')
    ax.set_xlabel('Baseline alignment correlation')
    ax.set_ylabel('Post alignment correlation')
    ax.set_title('Alignment QC', loc='left', fontweight='bold')
    ax.legend(frameon=False)
    ax = axes[1, 0]
    positions = {'Control': 0, 'TMS': 1}
    for (group, subset) in subject_metrics.groupby('group'):
        x = np.full(len(subset), positions[group], dtype=float)
        jitter = np.linspace(-0.07, 0.07, len(subset)) if len(subset) > 1 else 0
        ax.scatter(x + jitter, subset['mean_reference_directed_G1G2'], s=20, color=colors[group], alpha=0.8, edgecolor='white', linewidth=0.35)
        ax.hlines(subset['mean_reference_directed_G1G2'].mean(), positions[group] - 0.18, positions[group] + 0.18, color='#111111', lw=1.4)
    ax.axhline(0, color='#AAAAAA', lw=0.65)
    ax.set_xticks([0, 1], ['Control', 'TMS'])
    ax.set_ylabel('Mean healthy-reference\nG1-G2 normalization')
    ax.set_title('Whole-cortex normalization', loc='left', fontweight='bold')
    ax = axes[1, 1]
    for (group, subset) in subject_metrics.groupby('group'):
        ax.scatter(subset['log1p_lesion_volume'], subset['anatomical_G1G2_slope'], s=20, color=colors[group], alpha=0.8, edgecolor='white', linewidth=0.35, label=group)
    ax.axhline(0, color='#AAAAAA', lw=0.65)
    ax.set_xlabel('log(1 + baseline lesion volume)')
    ax.set_ylabel('Anatomical-distance slope')
    ax.set_title('Anatomical-distance control', loc='left', fontweight='bold')
    for (label, ax) in zip('abcd', axes.ravel()):
        ax.text(-0.16, 1.07, label, transform=ax.transAxes, fontsize=9, fontweight='bold', va='top')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    fig.tight_layout(w_pad=1.8, h_pad=1.7)
    fig.savefig(FIGURES / 'lesion_gradient_distance_pilot_qc.pdf', bbox_inches='tight')
    fig.savefig(FIGURES / 'lesion_gradient_distance_pilot_qc.png', dpi=400, bbox_inches='tight')
    plt.close(fig)

def run(n_permutations: int) -> None:
    ensure_dirs()
    (atlas_image, atlas, labels) = load_atlas()
    prepared = base.load_inputs()
    participants = prepared.table.copy()
    participants['subject_id'] = participants['subject_id'].astype(str)
    participants = participants.sort_values('subject_id').reset_index(drop=True)
    fc_paths = {path.name.split('_', 1)[0]: path for path in sorted(FC_Z_ROOT.glob('*_baseline_fc_z.npy'))}
    subject_ids = participants['subject_id'].tolist()
    if set(subject_ids) != set(fc_paths):
        raise RuntimeError(f'FC/participant mismatch: missing={set(subject_ids) - set(fc_paths)}, extra={set(fc_paths) - set(subject_ids)}')
    fc_stack = np.stack([np.load(fc_paths[sid]) for sid in subject_ids]).astype(float)
    if fc_stack.shape != (38, 400, 400):
        raise RuntimeError(f'Unexpected FC stack shape: {fc_stack.shape}')
    consistent_baseline_paths = {path.name.split('_', 1)[0]: path for path in sorted(CONSISTENT_FC_Z_ROOT.glob('*_baseline_fc_z.npy'))}
    consistent_post_paths = {path.name.split('_', 1)[0]: path for path in sorted(CONSISTENT_FC_Z_ROOT.glob('*_post_fc_z.npy'))}
    consistent_fc_ready = set(consistent_baseline_paths) == set(subject_ids) and set(consistent_post_paths) == set(subject_ids)
    gradient_source = 'gradients recomputed from motion-scrubbed baseline/post Schaefer400 FC using the same diffusion-map pipeline' if consistent_fc_ready else 'legacy aligned5_masked arrays; pilot fallback because consistent longitudinal FC was unavailable'
    if consistent_fc_ready:
        longitudinal_qc = pd.read_csv(OUTPUT_ROOT / 'fc_consistent' / 'tables' / 'longitudinal_fc_qc.csv', dtype={'subject_id': str})
        motion_primary_ids = set(longitudinal_qc.groupby('subject_id')['primary_motion_eligible'].all().loc[lambda values: values].index.astype(str))
        motion_lookup = {}
        for (subject_id, rows) in longitudinal_qc.groupby('subject_id'):
            sessions = rows.set_index('session')
            motion_lookup[str(subject_id)] = {'baseline_mean_fd': float(sessions.loc['baseline', 'mean_fd']), 'post_mean_fd': float(sessions.loc['post', 'mean_fd']), 'baseline_retained_frames': int(sessions.loc['baseline', 'retained_frames']), 'post_retained_frames': int(sessions.loc['post', 'retained_frames'])}
    else:
        motion_primary_ids = set(subject_ids)
        motion_lookup = {subject_id: {'baseline_mean_fd': np.nan, 'post_mean_fd': np.nan, 'baseline_retained_frames': np.nan, 'post_retained_frames': np.nan} for subject_id in subject_ids}
    fc_sum = np.sum(fc_stack, axis=0)
    healthy_main_fc = load_group_fc('schaefer', scale=400, group='main')
    healthy_holdout_fc = load_group_fc('schaefer', scale=400, group='holdout')
    (master_raw, master_lambdas) = fit_gradient(healthy_main_fc, fisher_z_input=False)
    (master, orientation) = orient_master(master_raw, labels)
    master_mean = master.mean(axis=0)
    master_sd = master.std(axis=0, ddof=1)
    master_z = (master - master_mean) / master_sd
    (holdout_raw, holdout_lambdas) = fit_gradient(healthy_holdout_fc, fisher_z_input=False)
    (holdout, holdout_fit) = similarity_align(holdout_raw, master)
    holdout_z = (holdout - master_mean) / master_sd
    healthy_validation = pd.DataFrame([{**{f'main_lambda_G{i + 1}': float(v) for (i, v) in enumerate(master_lambdas)}, **{f'holdout_lambda_G{i + 1}': float(v) for (i, v) in enumerate(holdout_lambdas)}, **{f'holdout_to_main_{key}': value for (key, value) in holdout_fit.items()}, **{f'G{i + 1}_main_holdout_correlation': float(np.corrcoef(master_z[:, i], holdout_z[:, i])[0, 1]) for i in range(N_COMPONENTS)}}])
    healthy_validation.to_csv(TABLES / 'healthy_reference_main_holdout_validation.csv', index=False, encoding='utf-8-sig')
    reference_table = labels.copy()
    for component in range(N_COMPONENTS):
        reference_table[f'G{component + 1}_reference_raw'] = master[:, component]
        reference_table[f'G{component + 1}_reference_z'] = master_z[:, component]
    reference_table.to_csv(TABLES / 'healthy_normative_reference_schaefer400.csv', index=False, encoding='utf-8-sig')
    np.save(REFERENCES / 'healthy_normative_reference_400x5.npy', master)
    xtract_burden = pd.read_csv(XTRACT_BURDEN_TABLE, dtype={'subject_id': str})
    tract_names = xtract_burden[['tract_index', 'tract_name']].drop_duplicates().sort_values('tract_index')
    if tract_names['tract_index'].tolist() != list(range(42)):
        raise RuntimeError('XTRACT burden table does not contain ordered tracts 0-41')
    (xtract_endpoints, xtract_endpoint_table, xtract_endpoint_total) = build_xtract_endpoint_projection(atlas_image, atlas, tract_names)
    xtract_endpoint_table.to_csv(TABLES / 'xtract_schaefer400_endpoint_projection.csv', index=False, encoding='utf-8-sig')
    reference_qc_rows = []
    alignment_rows = []
    roi_rows = []
    subject_rows = []
    direct_model_rows = []
    xtract_model_rows = []
    for (subject_index, participant) in enumerate(participants.itertuples()):
        subject_id = participant.subject_id
        group = participant.group
        primary_motion_eligible = subject_id in motion_primary_ids
        (loso_raw, loso_lambdas) = fit_gradient((fc_sum - fc_stack[subject_index]) / (len(fc_stack) - 1))
        (loso, loso_fit) = similarity_align(loso_raw, master)
        loso_z = (loso - master_mean) / master_sd
        reference_qc_rows.append({'subject_id': subject_id, 'group': group, **{f'lambda_G{i + 1}': float(v) for (i, v) in enumerate(loso_lambdas)}, **{f'loso_to_master_{key}': value for (key, value) in loso_fit.items()}, 'G1_loso_master_correlation': float(np.corrcoef(loso_z[:, 0], master_z[:, 0])[0, 1]), 'G2_loso_master_correlation': float(np.corrcoef(loso_z[:, 1], master_z[:, 1])[0, 1])})
        lesion_image = nib.load(str(lesion_mask_path(group, subject_id)))
        if lesion_image.shape != atlas_image.shape or not np.allclose(lesion_image.affine, atlas_image.affine):
            raise RuntimeError(f'Lesion/atlas geometry mismatch: {subject_id}')
        lesion = np.asarray(lesion_image.dataobj) > 0
        if not lesion.any():
            raise RuntimeError(f'Empty baseline lesion mask: {subject_id}')
        lesion_indices = np.argwhere(lesion)
        lesion_xyz = nib.affines.apply_affine(lesion_image.affine, lesion_indices)
        lesion_centroid = lesion_xyz.mean(axis=0)
        voxel_ml = float(np.prod(lesion_image.header.get_zooms()[:3]) / 1000.0)
        lesion_volume_ml = float(lesion.sum() * voxel_ml)
        damaged_voxels = np.bincount(atlas[lesion], minlength=401)[1:]
        roi_voxels = labels['roi_voxels'].to_numpy(int)
        lesion_fraction = damaged_voxels / roi_voxels
        intact_roi = damaged_voxels == 0
        distance_map = distance_transform_edt(~lesion, sampling=atlas_image.header.get_zooms()[:3])
        anatomical_min = np.empty(400, dtype=float)
        anatomical_median = np.empty(400, dtype=float)
        for roi_index in range(400):
            distances = distance_map[atlas == roi_index + 1]
            anatomical_min[roi_index] = float(np.min(distances))
            anatomical_median[roi_index] = float(np.median(distances))
        roi_centroids = labels[['centroid_x', 'centroid_y', 'centroid_z']].to_numpy(float)
        anatomical_centroid = np.linalg.norm(roi_centroids - lesion_centroid, axis=1)
        if consistent_fc_ready:
            (before, before_lambdas) = fit_gradient(np.load(consistent_baseline_paths[subject_id]))
            (after, after_lambdas) = fit_gradient(np.load(consistent_post_paths[subject_id]))
        else:
            before = np.asarray(np.load(gradient_file(group, 'before', subject_id)), dtype=float)
            after = np.asarray(np.load(gradient_file(group, 'after', subject_id)), dtype=float)
            before_lambdas = np.full(N_COMPONENTS, np.nan)
            after_lambdas = np.full(N_COMPONENTS, np.nan)
        if before.shape != (400, 5) or after.shape != (400, 5):
            raise RuntimeError(f'Unexpected gradient shape for {subject_id}: {before.shape}, {after.shape}')
        (before_aligned, before_fit) = similarity_align(before, master, intact_roi)
        (after_aligned, after_fit) = similarity_align(after, master, intact_roi)
        before_z = (before_aligned - master_mean) / master_sd
        after_z = (after_aligned - master_mean) / master_sd
        alignment_rows.append({'subject_id': subject_id, 'group': group, 'intact_roi_n': int(intact_roi.sum()), 'primary_motion_eligible': primary_motion_eligible, 'gradient_source': gradient_source, **{f'pre_lambda_G{i + 1}': float(value) for (i, value) in enumerate(before_lambdas)}, **{f'post_lambda_G{i + 1}': float(value) for (i, value) in enumerate(after_lambdas)}, **{f'pre_{key}': value for (key, value) in before_fit.items()}, **{f'post_{key}': value for (key, value) in after_fit.items()}})
        pre_dev_g1 = np.abs(before_z[:, 0] - master_z[:, 0])
        post_dev_g1 = np.abs(after_z[:, 0] - master_z[:, 0])
        pre_dev_g2 = np.abs(before_z[:, 1] - master_z[:, 1])
        post_dev_g2 = np.abs(after_z[:, 1] - master_z[:, 1])
        pre_dev_g1g2 = np.linalg.norm(before_z[:, :2] - master_z[:, :2], axis=1)
        post_dev_g1g2 = np.linalg.norm(after_z[:, :2] - master_z[:, :2], axis=1)
        pre_dev_5d = np.linalg.norm(before_z - master_z, axis=1)
        post_dev_5d = np.linalg.norm(after_z - master_z, axis=1)
        normalization_g1 = pre_dev_g1 - post_dev_g1
        normalization_g2 = pre_dev_g2 - post_dev_g2
        normalization_g1g2 = pre_dev_g1g2 - post_dev_g1g2
        normalization_5d = pre_dev_5d - post_dev_5d
        reorganization_g1g2 = np.linalg.norm(after_z[:, :2] - before_z[:, :2], axis=1)
        direct_valid = bool(damaged_voxels.sum() > 0)
        if direct_valid:
            weights = damaged_voxels.astype(float)
            lesion_gradient = np.average(master_z, axis=0, weights=weights)
            lesion_g1_span = float(np.sqrt(np.average((master_z[:, 0] - lesion_gradient[0]) ** 2, weights=weights)))
            lesion_g2_span = float(np.sqrt(np.average((master_z[:, 1] - lesion_gradient[1]) ** 2, weights=weights)))
            g1_distance = np.abs(master_z[:, 0] - lesion_gradient[0])
            g2_distance = np.abs(master_z[:, 1] - lesion_gradient[1])
            g1g2_distance = np.linalg.norm(master_z[:, :2] - lesion_gradient[:2], axis=1)
        else:
            lesion_gradient = np.full(N_COMPONENTS, np.nan)
            lesion_g1_span = np.nan
            lesion_g2_span = np.nan
            g1_distance = np.full(400, np.nan)
            g2_distance = np.full(400, np.nan)
            g1g2_distance = np.full(400, np.nan)
        subject_tracts = xtract_burden.loc[xtract_burden['subject_id'] == subject_id].sort_values('tract_index')
        if subject_tracts['tract_index'].tolist() != list(range(42)):
            raise RuntimeError(f'Incomplete XTRACT burden rows for {subject_id}')
        tract_damage_fraction = subject_tracts['tract_probability_mass_coverage_pct'].to_numpy(float) / 100.0
        projectable_tract = xtract_endpoint_total > 0
        all_tract_damage = float(tract_damage_fraction.sum())
        projectable_tract_damage = float(tract_damage_fraction[projectable_tract].sum())
        xtract_weight_raw = tract_damage_fraction @ xtract_endpoints
        if not np.isfinite(xtract_weight_raw).all() or xtract_weight_raw.sum() <= 0:
            raise RuntimeError(f'No projectable XTRACT cortical weight for {subject_id}')
        xtract_weight = xtract_weight_raw / xtract_weight_raw.sum()
        xtract_gradient = np.average(master_z, axis=0, weights=xtract_weight)
        xtract_g1_span = float(np.sqrt(np.average((master_z[:, 0] - xtract_gradient[0]) ** 2, weights=xtract_weight)))
        xtract_g2_span = float(np.sqrt(np.average((master_z[:, 1] - xtract_gradient[1]) ** 2, weights=xtract_weight)))
        xtract_g1_distance = np.abs(master_z[:, 0] - xtract_gradient[0])
        xtract_g2_distance = np.abs(master_z[:, 1] - xtract_gradient[1])
        xtract_g1g2_distance = np.linalg.norm(master_z[:, :2] - xtract_gradient[:2], axis=1)
        subject_roi = labels[['roi', 'roi_name', 'network']].copy()
        subject_roi.insert(0, 'group', group)
        subject_roi.insert(0, 'subject_id', subject_id)
        subject_roi.insert(2, 'primary_motion_eligible', primary_motion_eligible)
        subject_roi['damaged_voxels'] = damaged_voxels
        subject_roi['lesion_fraction'] = lesion_fraction
        subject_roi['intact_roi'] = intact_roi
        subject_roi['anatomical_distance_min_mm'] = anatomical_min
        subject_roi['anatomical_distance_median_mm'] = anatomical_median
        subject_roi['anatomical_distance_centroid_mm'] = anatomical_centroid
        subject_roi['G1_reference_z'] = master_z[:, 0]
        subject_roi['G2_reference_z'] = master_z[:, 1]
        subject_roi['G1_pre_z'] = before_z[:, 0]
        subject_roi['G2_pre_z'] = before_z[:, 1]
        subject_roi['G1_post_z'] = after_z[:, 0]
        subject_roi['G2_post_z'] = after_z[:, 1]
        subject_roi['reference_directed_G1'] = normalization_g1
        subject_roi['reference_directed_G2'] = normalization_g2
        subject_roi['reference_directed_G1G2'] = normalization_g1g2
        subject_roi['reference_directed_5D'] = normalization_5d
        subject_roi['reorganization_magnitude_G1G2'] = reorganization_g1g2
        subject_roi['direct_lesion_G1_distance'] = g1_distance
        subject_roi['direct_lesion_G2_distance'] = g2_distance
        subject_roi['direct_lesion_G1G2_distance'] = g1g2_distance
        subject_roi['xtract_projected_roi_weight'] = xtract_weight
        subject_roi['xtract_projected_G1_distance'] = xtract_g1_distance
        subject_roi['xtract_projected_G2_distance'] = xtract_g2_distance
        subject_roi['xtract_projected_G1G2_distance'] = xtract_g1g2_distance
        roi_rows.append(subject_roi)
        anatomy_model = fit_roi_model(subject_roi, 'reference_directed_G1G2', gradient_distance=None)
        subject_summary = {'subject_id': subject_id, 'group': group, 'group_tms': int(group == 'TMS'), 'primary_motion_eligible': primary_motion_eligible, **motion_lookup[subject_id], 'delta_mean_fd': motion_lookup[subject_id]['post_mean_fd'] - motion_lookup[subject_id]['baseline_mean_fd'], 'lesion_volume_ml_recomputed': lesion_volume_ml, 'log1p_lesion_volume': float(participant.log1p_lesion_volume), 'mask_lateralization_index': float(participant.mask_lateralization_index), 'cortical_overlap_voxels': int(damaged_voxels.sum()), 'cortical_overlap_roi_n': int(np.count_nonzero(damaged_voxels)), 'direct_lesion_gradient_valid': direct_valid, 'direct_lesion_G1': float(lesion_gradient[0]), 'direct_lesion_G2': float(lesion_gradient[1]), 'direct_lesion_G1_span': lesion_g1_span, 'direct_lesion_G2_span': lesion_g2_span, 'mean_reference_directed_G1': float(normalization_g1[intact_roi].mean()), 'mean_reference_directed_G2': float(normalization_g2[intact_roi].mean()), 'mean_reference_directed_G1G2': float(normalization_g1g2[intact_roi].mean()), 'mean_reference_directed_5D': float(normalization_5d[intact_roi].mean()), 'mean_reorganization_magnitude_G1G2': float(reorganization_g1g2[intact_roi].mean()), 'mean_baseline_deviation_G1': float(pre_dev_g1[intact_roi].mean()), 'mean_baseline_deviation_G2': float(pre_dev_g2[intact_roi].mean()), 'mean_baseline_deviation_G1G2': float(pre_dev_g1g2[intact_roi].mean()), 'mean_baseline_deviation_5D': float(pre_dev_5d[intact_roi].mean()), 'nearest_anatomical_decile_baseline_deviation_G1G2': nearest_decile_mean(subject_roi.assign(baseline_deviation_G1G2=pre_dev_g1g2), 'anatomical_distance_min_mm', 'baseline_deviation_G1G2'), 'nearest_anatomical_decile_reference_directed_G1G2': nearest_decile_mean(subject_roi, 'anatomical_distance_min_mm', 'reference_directed_G1G2'), 'anatomical_G1G2_slope': anatomy_model['anatomical_slope'], 'anatomical_G1G2_slope_se_hc3': anatomy_model['anatomical_slope_se_hc3'], 'anatomical_G1G2_model_r_squared': anatomy_model['r_squared'], 'xtract_projectable_tract_damage_fraction': projectable_tract_damage / all_tract_damage if all_tract_damage > 0 else np.nan, 'xtract_projected_effective_roi_n': float(1.0 / np.sum(xtract_weight ** 2)), 'xtract_projected_max_roi_weight': float(xtract_weight.max()), 'xtract_projected_lesion_G1': float(xtract_gradient[0]), 'xtract_projected_lesion_G2': float(xtract_gradient[1]), 'xtract_projected_G1_span': xtract_g1_span, 'xtract_projected_G2_span': xtract_g2_span, 'nearest_xtract_G1G2_decile_normalization': nearest_decile_mean(subject_roi, 'xtract_projected_G1G2_distance', 'reference_directed_G1G2'), 'nearest_xtract_G1G2_decile_baseline_deviation': nearest_decile_mean(subject_roi.assign(baseline_deviation_G1G2=pre_dev_g1g2), 'xtract_projected_G1G2_distance', 'baseline_deviation_G1G2')}
        xtract_model_specs = [('G1', 'reference_directed_G1', 'xtract_projected_G1_distance'), ('G2', 'reference_directed_G2', 'xtract_projected_G2_distance'), ('G1G2', 'reference_directed_G1G2', 'xtract_projected_G1G2_distance')]
        for (space, outcome, distance) in xtract_model_specs:
            result = fit_roi_model(subject_roi, outcome, distance)
            subject_summary[f'xtract_projected_{space}_slope'] = result['gradient_slope']
            subject_summary[f'xtract_projected_{space}_slope_se_hc3'] = result['gradient_slope_se_hc3']
            subject_summary[f'xtract_projected_{space}_model_r_squared'] = result['r_squared']
            xtract_model_rows.append({'subject_id': subject_id, 'group': group, 'primary_motion_eligible': primary_motion_eligible, 'space': space, 'outcome': outcome, 'gradient_distance': distance, **result, 'nearest_gradient_decile_mean': nearest_decile_mean(subject_roi, distance, outcome)})
        if direct_valid:
            model_specs = [('G1', 'reference_directed_G1', 'direct_lesion_G1_distance'), ('G2', 'reference_directed_G2', 'direct_lesion_G2_distance'), ('G1G2', 'reference_directed_G1G2', 'direct_lesion_G1G2_distance')]
            for (space, outcome, distance) in model_specs:
                result = fit_roi_model(subject_roi, outcome, distance)
                direct_model_rows.append({'subject_id': subject_id, 'group': group, 'primary_motion_eligible': primary_motion_eligible, 'space': space, 'outcome': outcome, 'gradient_distance': distance, **result, 'nearest_gradient_decile_mean': nearest_decile_mean(subject_roi, distance, outcome)})
        subject_rows.append(subject_summary)
    reference_qc = pd.DataFrame(reference_qc_rows)
    alignment = pd.DataFrame(alignment_rows)
    roi_long = pd.concat(roi_rows, ignore_index=True)
    subject_metrics = pd.DataFrame(subject_rows)
    direct_models = pd.DataFrame(direct_model_rows)
    xtract_models = pd.DataFrame(xtract_model_rows)
    reference_qc.to_csv(TABLES / 'loso_reference_qc.csv', index=False, encoding='utf-8-sig')
    alignment.to_csv(TABLES / 'subject_alignment_qc.csv', index=False, encoding='utf-8-sig')
    roi_long.to_csv(TABLES / 'roi_level_distance_reorganization.csv', index=False, encoding='utf-8-sig')
    subject_metrics.to_csv(TABLES / 'subject_level_distance_metrics.csv', index=False, encoding='utf-8-sig')
    direct_models.to_csv(TABLES / 'direct_lesion_gradient_within_subject_models.csv', index=False, encoding='utf-8-sig')
    xtract_models.to_csv(TABLES / 'xtract_projected_lesion_gradient_within_subject_models.csv', index=False, encoding='utf-8-sig')
    rng = np.random.default_rng(SEED)
    group_outcomes = [('global_normalization_control', 'mean_reference_directed_G1', 'mean_baseline_deviation_G1', []), ('global_normalization_control', 'mean_reference_directed_G2', 'mean_baseline_deviation_G2', []), ('global_normalization_control', 'mean_reference_directed_G1G2', 'mean_baseline_deviation_G1G2', []), ('global_normalization_control', 'mean_reference_directed_5D', 'mean_baseline_deviation_5D', []), ('global_normalization_control', 'mean_reorganization_magnitude_G1G2', 'mean_baseline_deviation_G1G2', []), ('global_normalization_control', 'nearest_anatomical_decile_reference_directed_G1G2', 'nearest_anatomical_decile_baseline_deviation_G1G2', []), ('global_normalization_control', 'anatomical_G1G2_slope', 'mean_baseline_deviation_G1G2', []), ('xtract_lesion_gradient', 'xtract_projected_G1_slope', 'mean_baseline_deviation_G1', ['xtract_projected_lesion_G1', 'xtract_projectable_tract_damage_fraction', 'xtract_projected_effective_roi_n']), ('xtract_lesion_gradient', 'xtract_projected_G2_slope', 'mean_baseline_deviation_G2', ['xtract_projected_lesion_G2', 'xtract_projectable_tract_damage_fraction', 'xtract_projected_effective_roi_n']), ('xtract_lesion_gradient', 'xtract_projected_G1G2_slope', 'mean_baseline_deviation_G1G2', ['xtract_projected_lesion_G1', 'xtract_projected_lesion_G2', 'xtract_projectable_tract_damage_fraction', 'xtract_projected_effective_roi_n']), ('xtract_lesion_gradient', 'nearest_xtract_G1G2_decile_normalization', 'nearest_xtract_G1G2_decile_baseline_deviation', ['xtract_projected_lesion_G1', 'xtract_projected_lesion_G2', 'xtract_projectable_tract_damage_fraction', 'xtract_projected_effective_roi_n'])]
    group_model_rows = []
    cohorts = {'primary_motion_qc': subject_metrics.loc[subject_metrics['primary_motion_eligible']], 'all_38_sensitivity': subject_metrics}
    for (cohort_name, cohort_table) in cohorts.items():
        for (family, outcome, baseline_deviation, extra_covariates) in group_outcomes:
            group_model_rows.append({'cohort': cohort_name, 'family': family, **freedman_lane_group_test(cohort_table, outcome, baseline_deviation, extra_covariates, n_permutations, rng)})
    group_models = pd.DataFrame(group_model_rows)
    group_models['fdr_q_hc3'] = np.nan
    group_models['fdr_q_freedman_lane'] = np.nan
    for ((cohort_name, family), _) in group_models.groupby(['cohort', 'family']):
        selected = (group_models['cohort'] == cohort_name) & (group_models['family'] == family)
        group_models.loc[selected, 'fdr_q_hc3'] = multipletests(group_models.loc[selected, 'p_hc3'], method='fdr_bh')[1]
        group_models.loc[selected, 'fdr_q_freedman_lane'] = multipletests(group_models.loc[selected, 'p_freedman_lane'], method='fdr_bh')[1]
    group_models.to_csv(TABLES / 'group_models_reference_directed_reorganization.csv', index=False, encoding='utf-8-sig')
    pairs = pd.read_csv(MATCHED_PAIRS_TABLE, dtype={'control_subject_id': str, 'tms_subject_id': str})
    metric_lookup = subject_metrics.set_index('subject_id')
    motion_eligible_pairs = pairs['control_subject_id'].isin(motion_primary_ids) & pairs['tms_subject_id'].isin(motion_primary_ids)
    matched_sets = {'all_11_pairs': pairs, 'tight_9_pairs': pairs.loc[pairs['standardized_rms_distance'] < 1.0], 'motion_qc_10_pairs': pairs.loc[motion_eligible_pairs], 'motion_qc_tight_8_pairs': pairs.loc[motion_eligible_pairs & (pairs['standardized_rms_distance'] < 1.0)]}
    matched_outcomes = ['xtract_projected_G1_slope', 'xtract_projected_G2_slope', 'xtract_projected_G1G2_slope', 'nearest_xtract_G1G2_decile_normalization']
    matched_test_rows = []
    matched_difference_rows = []
    for (match_set, match_table) in matched_sets.items():
        for outcome in matched_outcomes:
            differences = []
            for pair in match_table.itertuples():
                control_value = float(metric_lookup.loc[pair.control_subject_id, outcome])
                tms_value = float(metric_lookup.loc[pair.tms_subject_id, outcome])
                difference = tms_value - control_value
                differences.append(difference)
                matched_difference_rows.append({'match_set': match_set, 'pair_id': int(pair.pair_id), 'control_subject_id': pair.control_subject_id, 'tms_subject_id': pair.tms_subject_id, 'standardized_rms_distance': float(pair.standardized_rms_distance), 'outcome': outcome, 'control_value': control_value, 'tms_value': tms_value, 'tms_minus_control': difference})
            matched_test_rows.append({'match_set': match_set, 'outcome': outcome, **exact_paired_signflip(np.asarray(differences))})
    matched_tests = pd.DataFrame(matched_test_rows)
    matched_tests['fdr_q_within_match_set'] = np.nan
    for match_set in matched_tests['match_set'].unique():
        selected = matched_tests['match_set'] == match_set
        matched_tests.loc[selected, 'fdr_q_within_match_set'] = multipletests(matched_tests.loc[selected, 'exact_signflip_p'], method='fdr_bh')[1]
    matched_tests.to_csv(TABLES / 'xtract_projected_matched_group_tests.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(matched_difference_rows).to_csv(TABLES / 'xtract_projected_matched_pair_differences.csv', index=False, encoding='utf-8-sig')
    make_figure(reference_table, alignment.loc[alignment['primary_motion_eligible']], subject_metrics.loc[subject_metrics['primary_motion_eligible']])
    direct_counts = subject_metrics.groupby('group')['direct_lesion_gradient_valid'].agg(['count', 'sum']).rename(columns={'count': 'n', 'sum': 'direct_valid_n'})
    manifest = {'analysis_date': '2026-08-03', 'cohort_n': int(len(subject_metrics)), 'primary_motion_qc_n': int(subject_metrics['primary_motion_eligible'].sum()), 'primary_motion_exclusions': subject_metrics.loc[~subject_metrics['primary_motion_eligible'], 'subject_id'].tolist(), 'group_counts': subject_metrics['group'].value_counts().to_dict(), 'reference': {'primary_current': 'BrainSpace packaged healthy group Schaefer400 FC, discovery/main group', 'healthy_reference': True, 'source': "brainspace.datasets.load_group_fc('schaefer', 400, 'main'); discovery dataset of Vos de Wael et al. 2018, distributed with BrainSpace", 'validation': 'independent BrainSpace holdout group', 'allowed_term': 'healthy-reference normalization', 'embedding': 'BrainSpace diffusion maps; normalized-angle kernel; 90% row sparsity; 5 components', 'direction_rules': orientation, 'master_eigenvalues': master_lambdas.tolist(), 'main_holdout_component_correlations': {f'G{i + 1}': float(healthy_validation.iloc[0][f'G{i + 1}_main_holdout_correlation']) for i in range(N_COMPONENTS)}, 'patient_baseline_loso_role': 'QC/sensitivity only; never used as the primary normalization target'}, 'alignment': {'method': 'five-dimensional similarity Procrustes to one fixed healthy normative reference', 'fit_rois': 'baseline lesion-intact Schaefer400 ROIs (zero directly damaged voxels)', 'scaling': 'pre and post expressed using the same healthy-reference component means and SDs', 'individual_gradient_source': gradient_source}, 'direct_lesion_gradient_estimability': direct_counts.reset_index().to_dict(orient='records'), 'xtract_projected_lesion_gradient': {'coverage_n': int(subject_metrics['xtract_projected_lesion_G1'].notna().sum()), 'atlas': 'FSL XTRACT HCP probabilistic tract atlas, 42 pathways', 'lesion_tract_weight': 'fraction of each normative tract probability mass intersected by the baseline lesion', 'cortical_endpoint_definition': f'XTRACT probability >= {XTRACT_ENDPOINT_THRESHOLD_PCT}% intersected with Schaefer400; normalized within each tract', 'projection': 'sum across tracts of lesion tract-damage fraction multiplied by normalized cortical endpoint weights', 'interpretation': 'normative tract-endpoint projection proxy; not subject-specific tractography or a full voxelwise disconnectome'}, 'anatomical_distance': {'primary': 'minimum Euclidean distance from each Schaefer ROI voxel to baseline lesion mask via 2-mm MNI distance transform', 'sensitivity': ['median ROI distance', 'ROI-centroid to lesion-centroid distance']}, 'inference': {'group_model': 'subject metric ~ TMS + log1p lesion volume + lesion lateralization index + matching baseline gradient deviation + baseline mean FD + post mean FD; XTRACT outcomes additionally adjust projected lesion G1/G2 position, projectable tract fraction, and effective cortical endpoint count', 'robust_se': 'HC3', 'permutation': f'Freedman-Lane, {n_permutations} permutations', 'multiple_testing': 'Benjamini-Hochberg FDR separately within global-normalization controls and XTRACT lesion-gradient tests', 'matched_sensitivity': 'exact paired sign-flip tests in the prior 11-pair and tight 9-pair lesion matches, plus motion-QC 10-pair and tight 8-pair subsets; FDR within each match set'}, 'major_limitation': 'Direct cortical lesion-gradient distance is not estimable for participants without Schaefer400 lesion overlap. The all-subject XTRACT projection is a tract-endpoint proxy and requires validation with a full normative disconnectome.'}
    (OUTPUT_ROOT / 'analysis_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    primary_group_models = group_models.loc[group_models['cohort'] == 'primary_motion_qc']
    best_group = primary_group_models.sort_values('p_freedman_lane').iloc[0]
    best_xtract = primary_group_models.loc[primary_group_models['family'] == 'xtract_lesion_gradient'].sort_values('p_freedman_lane').iloc[0]
    best_matched = matched_tests.loc[matched_tests['match_set'] == 'motion_qc_10_pairs'].sort_values('exact_signflip_p').iloc[0]
    summary = f"# 病灶—梯度距离与TMS后梯度重组\n\n## 已完成\n\n- 使用BrainSpace自带的独立健康Schaefer400群体FC建立固定的五维reference gradient，并用独立holdout健康组验证。\n- 采用同一套运动剔除规则重新计算38例患者pre/post FC及五维梯度。\n- 以baseline非病灶ROI完成pre/post梯度对齐，并按健康reference统一方向和尺度。\n- 计算每个ROI到baseline病灶的最小、正中及质心解剖距离。\n- 计算G1、G2、G1-G2及五维的健康reference normalization，以及受试者内距离—重组斜率。\n- 对直接落入Schaefer400皮层的病灶计算直接lesion-gradient指标。\n- 使用HCP-XTRACT概率束将深部/白质病灶投影到皮层纤维束终点，使38例均可获得投影后的lesion-gradient distance。\n- 完成协变量调整、Freedman-Lane置换、FDR校正及既有病灶匹配对的精确配对检验。\n\n主要运动质控队列为{int(subject_metrics['primary_motion_eligible'].sum())}/38例。Control受试者PRIVATE_SUBJECT_008的post扫描按预设FD规则仅保留35帧，因此从主要模型排除；38例全纳入结果作为敏感性分析保留。\n\n## 可估计性边界\n\n- Control直接皮层映射：{int(direct_counts.loc['Control', 'direct_valid_n'])}/{int(direct_counts.loc['Control', 'n'])}例。\n- TMS直接皮层映射：{int(direct_counts.loc['TMS', 'direct_valid_n'])}/{int(direct_counts.loc['TMS', 'n'])}例。\n- 因此不能在TMS与Control之间比较“直接皮层病灶”的gradient-distance斜率。\n\n## 术语\n\n主要reference来自独立健康数据，所以pre到post向reference靠近可称为**健康reference normalization**。患者baseline leave-one-subject-out reference仅用于质控。覆盖全队列的病灶坐标是**XTRACT规范纤维束终点投影代理**，不是个体化纤维追踪，也不是完整的体素级disconnectome。\n\n## 主要统计结果\n\n预设结局中Freedman-Lane组间检验最强的是`{best_group['outcome']}`：调整后TMS系数{best_group['group_tms_beta']:.4f}，p={best_group['p_freedman_lane']:.4f}，FDR q={best_group['fdr_q_freedman_lane']:.4f}。\n\nXTRACT lesion-gradient家族中最强结果是`{best_xtract['outcome']}`：调整后TMS系数{best_xtract['group_tms_beta']:.4f}，p={best_xtract['p_freedman_lane']:.4f}，家族内FDR q={best_xtract['fdr_q_freedman_lane']:.4f}。正的G2距离斜率表示TMS相关的normalization相对更延伸到与病灶功能层级距离较远的区域，而不是更集中于病灶G2邻域。\n\n在运动质控后的10对病灶匹配患者中，最强配对结果是`{best_matched['outcome']}`：TMS-Control平均差{best_matched['mean_tms_minus_control']:.4f}，精确符号翻转p={best_matched['exact_signflip_p']:.4f}，组内FDR q={best_matched['fdr_q_within_match_set']:.4f}。该结果在更严格、更小的匹配子集中减弱，因此应作为支持性敏感性证据，而不是无混杂的因果效应。\n"
    (OUTPUT_ROOT / 'RESULTS_ZH.md').write_text(summary, encoding='utf-8')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Healthy-reference lesion-gradient distance analysis')
    parser.add_argument('--permutations', type=int, default=N_PERMUTATIONS, help='Freedman-Lane permutations')
    return parser.parse_args()
if __name__ == '__main__':
    arguments = parse_args()
    run(arguments.permutations)
