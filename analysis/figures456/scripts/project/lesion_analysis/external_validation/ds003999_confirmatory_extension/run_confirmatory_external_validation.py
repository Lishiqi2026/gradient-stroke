from __future__ import annotations
'Locked external validation of the main-cohort gradient phenotypes.\n\nThe analysis follows ``FROZEN_VALIDATION_PROTOCOL.md``.  It deliberately keeps\nexternal-discovered ROIs out of confirmatory tests and uses no post-treatment\nimaging variable in prognostic or predictive models.\n'
import os as _portable_os
from pathlib import Path as _PortablePath

def _portable_path(kind, relative):
    project = _PortablePath(_portable_os.environ['PAPER_PROJECT_ROOT'])
    root = project if kind == 'project' else project.parent
    return str(root / relative)
import hashlib
import json
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from brainspace.datasets import load_group_fc
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation
from scipy.stats import chi2_contingency, pearsonr, spearmanr, t as t_dist, ttest_ind
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
HERE = Path(__file__).resolve().parent
VALIDATION_ROOT = HERE.parent / 'ds003999_validation'
ANALYSIS_ROOT = Path(_portable_path('project', 'lesion_analysis'))
PROJECT_ROOT = Path(_portable_path('project', ''))
for module_path in (VALIDATION_ROOT, ANALYSIS_ROOT):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))
import run_external_validation_analysis as legacy
from audit_healthy_reference_gradient_calculation import legacy_style_adapted_embedding
SEED = 20260812
N_PERM = 20000
N_BOOT = 10000
N_SPIN = 5000
TABLES = HERE / 'tables'
SOURCE_DATA = HERE / 'source_data'
REFERENCES = HERE / 'references'
GRADIENTS = HERE / 'gradients_primary'
QA = HERE / 'qa'
PARTICIPANTS = PROJECT_ROOT / 'new_dataset' / 'participants.tsv'
EXTERNAL_FC = PROJECT_ROOT / 'new_dataset' / '(400,400)' / '(400,400)'
DISCOVERY = ANALYSIS_ROOT / 'lesion_gradient_distance' / 'figures' / 'nature' / 'lesion_final' / 'provided_code_G1_G2_surface_source_data.csv'
MAIN_FC = ANALYSIS_ROOT / 'lesion_gradient_distance' / 'fc_consistent' / 'fc_matrices_z'
MAIN_SUBJECTS = ANALYSIS_ROOT / 'lesion_gradient_distance' / 'provided_code_g1_g2_recalculation' / 'provided_code_subject_level_global_metrics.csv'
MAIN_MOTION = ANALYSIS_ROOT / 'lesion_gradient_distance' / 'fc_consistent' / 'tables' / 'longitudinal_fc_qc.csv'
ESTABLISHED_REFERENCE = VALIDATION_ROOT / 'references' / 'healthy_reference_400x5.npy'
NETWORK_ENDPOINTS = (('DorsAttn', 'betweenDisp'), ('Default', 'betweenDisp'), ('Control', 'Segregation'))

def ensure_dirs() -> None:
    for path in (TABLES, SOURCE_DATA, REFERENCES, GRADIENTS / 'pre', GRADIENTS / 'post', QA):
        path.mkdir(parents=True, exist_ok=True)

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda : handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def fit_provided(fc_r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    (gradients, lambdas, _) = legacy_style_adapted_embedding(fc_r)
    return (gradients, lambdas)

def orthogonal_align(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    (rotation, _) = orthogonal_procrustes(np.asarray(source, float), np.asarray(target, float))
    aligned = np.asarray(source, float) @ rotation
    return (aligned, rotation, {'fit_r': float(np.corrcoef(aligned.ravel(), target.ravel())[0, 1]), 'fit_G1_r': float(np.corrcoef(aligned[:, 0], target[:, 0])[0, 1]), 'fit_G2_r': float(np.corrcoef(aligned[:, 1], target[:, 1])[0, 1]), 'fit_rms': float(np.sqrt(np.mean((aligned - target) ** 2)))})

def match_components(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    """Apply only a component permutation and sign flip to source."""
    source = np.asarray(source, float)
    target = np.asarray(target, float)
    correlations = np.corrcoef(source.T, target.T)[:source.shape[1], source.shape[1]:]
    (source_index, target_index) = linear_sum_assignment(-np.abs(correlations))
    order = np.empty(source.shape[1], dtype=int)
    signs = np.empty(source.shape[1], dtype=float)
    matched_r = np.empty(source.shape[1], dtype=float)
    for (src, tgt) in zip(source_index, target_index):
        order[tgt] = src
        signs[tgt] = 1.0 if correlations[src, tgt] >= 0 else -1.0
        matched_r[tgt] = abs(correlations[src, tgt])
    matched = source[:, order] * signs
    return (matched, {'component_source_order_for_target': ';'.join(map(str, order.tolist())), 'component_signs': ';'.join((f'{value:+.0f}' for value in signs)), 'component_abs_r': ';'.join((f'{value:.6f}' for value in matched_r)), 'minimum_component_abs_r': float(matched_r.min())})

def reference_scale(reference: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(reference, axis=0)
    sd = np.std(reference, axis=0, ddof=1)
    return (mean, sd, (reference - mean) / sd)

def baseline_locked_alignment(pre_raw: np.ndarray, post_raw: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    (post_matched, component_qc) = match_components(post_raw, pre_raw)
    (pre_aligned, rotation, pre_qc) = orthogonal_align(pre_raw, reference)
    post_aligned = post_matched @ rotation
    post_fit = {'post_fit_r': float(np.corrcoef(post_aligned.ravel(), reference.ravel())[0, 1]), 'post_fit_G1_r': float(np.corrcoef(post_aligned[:, 0], reference[:, 0])[0, 1]), 'post_fit_G2_r': float(np.corrcoef(post_aligned[:, 1], reference[:, 1])[0, 1])}
    return (pre_aligned, post_aligned, {**component_qc, **{f'pre_{key}': value for (key, value) in pre_qc.items()}, **post_fit, 'pre_scale': 1.0, 'post_scale': 1.0, 'translation': 0.0})

def separate_orthogonal_alignment(pre_raw: np.ndarray, post_raw: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    (pre_aligned, _, pre_qc) = orthogonal_align(pre_raw, reference)
    (post_aligned, _, post_qc) = orthogonal_align(post_raw, reference)
    return (pre_aligned, post_aligned, {**{f'pre_{key}': value for (key, value) in pre_qc.items()}, **{f'post_{key}': value for (key, value) in post_qc.items()}, 'pre_scale': 1.0, 'post_scale': 1.0, 'translation': 0.0})

def joint_pair_alignment(pre_raw: np.ndarray, post_raw: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    (post_matched, component_qc) = match_components(post_raw, pre_raw)
    stacked = np.vstack([pre_raw, post_matched])
    target = np.vstack([reference, reference])
    (_, rotation, joint_qc) = orthogonal_align(stacked, target)
    pre_aligned = pre_raw @ rotation
    post_aligned = post_matched @ rotation
    return (pre_aligned, post_aligned, {**component_qc, **{f'joint_{key}': value for (key, value) in joint_qc.items()}, 'pre_scale': 1.0, 'post_scale': 1.0, 'translation': 0.0})

def separate_similarity_alignment(pre_raw: np.ndarray, post_raw: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    reference_sd = np.std(reference, axis=0, ddof=1)
    (pre, post, mask, pre_qc, post_qc) = legacy.robust_align_pair(pre_raw, post_raw, reference, reference_sd)
    return (pre, post, {**{f'pre_{key}': value for (key, value) in pre_qc.items()}, **{f'post_{key}': value for (key, value) in post_qc.items()}, 'fit_roi_n': int(mask.sum())})
ALIGNERS = {'separate_orthogonal': separate_orthogonal_alignment, 'baseline_locked_orthogonal': baseline_locked_alignment, 'joint_pair_orthogonal': joint_pair_alignment, 'separate_similarity_robust90': separate_similarity_alignment}

def normalization(pre: np.ndarray, post: np.ndarray, reference: np.ndarray) -> dict[str, np.ndarray]:
    (mean, sd, ref_z) = reference_scale(reference)
    pre_z = (pre - mean) / sd
    post_z = (post - mean) / sd
    pre_dev_g1 = np.abs(pre_z[:, 0] - ref_z[:, 0])
    post_dev_g1 = np.abs(post_z[:, 0] - ref_z[:, 0])
    pre_dev_g2 = np.abs(pre_z[:, 1] - ref_z[:, 1])
    post_dev_g2 = np.abs(post_z[:, 1] - ref_z[:, 1])
    pre_dev_g1g2 = np.linalg.norm(pre_z[:, :2] - ref_z[:, :2], axis=1)
    post_dev_g1g2 = np.linalg.norm(post_z[:, :2] - ref_z[:, :2], axis=1)
    return {'pre_z': pre_z, 'post_z': post_z, 'reference_z': ref_z, 'pre_dev_G1': pre_dev_g1, 'pre_dev_G2': pre_dev_g2, 'pre_dev_G1G2': pre_dev_g1g2, 'normalization_G1': pre_dev_g1 - post_dev_g1, 'normalization_G2': pre_dev_g2 - post_dev_g2, 'normalization_G1G2': pre_dev_g1g2 - post_dev_g1g2}

def fixed_scores(metrics: dict[str, np.ndarray], weights: np.ndarray, fdr: np.ndarray, networks: np.ndarray) -> dict[str, float]:
    signed = np.where(fdr, weights, 0.0)
    denominator = float(np.sum(np.abs(signed)))
    positive = fdr & (weights > 0)
    negative = fdr & (weights < 0)
    vis_smn_positive = positive & np.isin(networks, ['Visual', 'SomMot'])
    norm = metrics['normalization_G2']
    pre_dev = metrics['pre_dev_G2']
    return {'G2_fixed67_betaweighted_normalization': float(np.sum(signed * norm) / denominator), 'G2_fixed67_signweighted_normalization': float(np.mean(np.sign(weights[fdr]) * norm[fdr])), 'G2_positive59_mean_normalization': float(np.mean(norm[positive])), 'G2_VIS_SomMot_positive53_mean_normalization': float(np.mean(norm[vis_smn_positive])), 'baseline_G2_fixed67_absweighted_deviation': float(np.sum(np.abs(signed) * pre_dev) / denominator), 'mean_G1_normalization': float(np.mean(metrics['normalization_G1'])), 'mean_G2_normalization': float(np.mean(metrics['normalization_G2'])), 'mean_G1G2_normalization': float(np.mean(metrics['normalization_G1G2']))}

def signflip_p(values: np.ndarray, *, alternative: str, seed: int=0) -> float:
    values = np.asarray(values, float)
    observed = float(np.mean(values))
    rng = np.random.default_rng(SEED + seed)
    exceed = 0
    batch = 2000
    completed = 0
    while completed < N_PERM:
        size = min(batch, N_PERM - completed)
        null = np.mean(values * rng.choice((-1.0, 1.0), size=(size, len(values))), axis=1)
        if alternative == 'greater':
            exceed += int(np.sum(null >= observed))
        else:
            exceed += int(np.sum(np.abs(null) >= abs(observed)))
        completed += size
    return float((exceed + 1) / (N_PERM + 1))

def bootstrap_mean_dz(values: np.ndarray, *, seed: int) -> dict[str, float]:
    values = np.asarray(values, float)
    rng = np.random.default_rng(SEED + seed)
    indices = rng.integers(0, len(values), size=(N_BOOT, len(values)))
    samples = values[indices]
    means = samples.mean(axis=1)
    sds = samples.std(axis=1, ddof=1)
    dzs = np.divide(means, sds, out=np.full_like(means, np.nan), where=sds > 0)
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    return {'n': len(values), 'mean': mean, 'sd': sd, 'mean_ci_low_boot': float(np.percentile(means, 2.5)), 'mean_ci_high_boot': float(np.percentile(means, 97.5)), 'dz': mean / sd, 'dz_ci_low_boot': float(np.nanpercentile(dzs, 2.5)), 'dz_ci_high_boot': float(np.nanpercentile(dzs, 97.5)), 'p_signflip_two_sided': signflip_p(values, alternative='two-sided', seed=seed + 101), 'p_signflip_greater': signflip_p(values, alternative='greater', seed=seed + 202)}

def hedges_g(control: np.ndarray, tms: np.ndarray) -> float:
    control = np.asarray(control, float)
    tms = np.asarray(tms, float)
    pooled = math.sqrt(((len(control) - 1) * np.var(control, ddof=1) + (len(tms) - 1) * np.var(tms, ddof=1)) / (len(control) + len(tms) - 2))
    d = (float(np.mean(tms)) - float(np.mean(control))) / pooled
    correction = 1 - 3 / (4 * (len(control) + len(tms)) - 9)
    return float(correction * d)

def bootstrap_hedges_g(control: np.ndarray, tms: np.ndarray, *, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(SEED + seed)
    output = np.empty(N_BOOT)
    for index in range(N_BOOT):
        c = control[rng.integers(0, len(control), len(control))]
        t = tms[rng.integers(0, len(tms), len(tms))]
        output[index] = hedges_g(c, t)
    return (float(np.nanpercentile(output, 2.5)), float(np.nanpercentile(output, 97.5)))

def process_external(participants: pd.DataFrame, labels: pd.DataFrame, discovery: pd.DataFrame, references: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    weights = discovery['adjusted_TMS_beta_G2'].to_numpy(float)
    fdr = discovery['G2_FDR_significant'].astype(bool).to_numpy()
    networks = labels['network'].to_numpy(str)
    subject_rows: list[dict[str, object]] = []
    roi_rows: list[dict[str, object]] = []
    network_rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []
    for (participant_index, participant) in enumerate(participants.itertuples(index=False)):
        subject_id = str(participant.participant_id)
        pre_fc = np.load(legacy.locate_array(EXTERNAL_FC, 'pre', subject_id))
        post_fc = np.load(legacy.locate_array(EXTERNAL_FC, 'post', subject_id))
        (pre_raw, pre_lambdas) = fit_provided(pre_fc)
        (post_raw, post_lambdas) = fit_provided(post_fc)
        for (reference_name, reference) in references.items():
            methods = ALIGNERS if reference_name == 'healthy_main' else {'separate_orthogonal': separate_orthogonal_alignment}
            for (alignment, aligner) in methods.items():
                (pre, post, qc) = aligner(pre_raw, post_raw, reference)
                metrics = normalization(pre, post, reference)
                row = {'participant_id': subject_id, 'reference': reference_name, 'alignment': alignment, **fixed_scores(metrics, weights, fdr, networks)}
                subject_rows.append(row)
                qc_rows.append({'participant_id': subject_id, 'reference': reference_name, 'alignment': alignment, **{f'pre_lambda_G{i + 1}': float(value) for (i, value) in enumerate(pre_lambdas)}, **{f'post_lambda_G{i + 1}': float(value) for (i, value) in enumerate(post_lambdas)}, **qc})
                if reference_name == 'healthy_main' and alignment == 'separate_orthogonal':
                    np.save(GRADIENTS / 'pre' / f'{subject_id}_pre_reference_locked_z_400x5.npy', metrics['pre_z'])
                    np.save(GRADIENTS / 'post' / f'{subject_id}_post_reference_locked_z_400x5.npy', metrics['post_z'])
                    for roi_index in range(400):
                        roi_rows.append({'participant_id': subject_id, 'roi': roi_index + 1, 'roi_name': labels.iloc[roi_index]['roi_name'], 'network': networks[roi_index], 'G1_pre_z': metrics['pre_z'][roi_index, 0], 'G1_post_z': metrics['post_z'][roi_index, 0], 'G2_pre_z': metrics['pre_z'][roi_index, 1], 'G2_post_z': metrics['post_z'][roi_index, 1], 'G1_normalization': metrics['normalization_G1'][roi_index], 'G2_normalization': metrics['normalization_G2'][roi_index], 'discovery_adjusted_TMS_beta_G2': weights[roi_index], 'discovery_G2_FDR': bool(fdr[roi_index])})
                    for component in (1, 2):
                        pre_network = legacy.component_network_metrics(metrics['pre_z'][:, component - 1], labels)
                        post_network = legacy.component_network_metrics(metrics['post_z'][:, component - 1], labels)
                        for (network, metric_name) in NETWORK_ENDPOINTS:
                            if component != 2:
                                continue
                            network_rows.append({'participant_id': subject_id, 'endpoint': f'G{component}_{network}_{metric_name}', 'gradient': f'G{component}', 'network': network, 'metric': metric_name, 'pre': pre_network[network][metric_name], 'post': post_network[network][metric_name], 'delta_post_minus_pre': post_network[network][metric_name] - pre_network[network][metric_name]})
        print(f'external gradient rerun {participant_index + 1:02d}/{len(participants)}: {subject_id}', flush=True)
    return (pd.DataFrame(subject_rows), pd.DataFrame(roi_rows), pd.DataFrame(network_rows), pd.DataFrame(qc_rows))

def process_main(labels: pd.DataFrame, discovery: pd.DataFrame, reference: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    main_subjects = pd.read_csv(MAIN_SUBJECTS, dtype={'subject_id': str})
    motion = pd.read_csv(MAIN_MOTION, dtype={'subject_id': str})
    eligible = set(motion.groupby('subject_id')['primary_motion_eligible'].all().loc[lambda values: values].index.astype(str))
    main_subjects = main_subjects.loc[main_subjects['subject_id'].isin(eligible)].copy()
    weights = discovery['adjusted_TMS_beta_G2'].to_numpy(float)
    fdr = discovery['G2_FDR_significant'].astype(bool).to_numpy()
    networks = labels['network'].to_numpy(str)
    subject_rows: list[dict[str, object]] = []
    network_rows: list[dict[str, object]] = []
    for row in main_subjects.itertuples(index=False):
        subject_id = str(row.subject_id)
        pre_fc_z = np.load(MAIN_FC / f'{subject_id}_baseline_fc_z.npy')
        post_fc_z = np.load(MAIN_FC / f'{subject_id}_post_fc_z.npy')
        (pre_raw, _) = fit_provided(np.tanh(pre_fc_z))
        (post_raw, _) = fit_provided(np.tanh(post_fc_z))
        (pre, post, _) = separate_orthogonal_alignment(pre_raw, post_raw, reference)
        metrics = normalization(pre, post, reference)
        subject_rows.append({'subject_id': subject_id, 'group': row.group, 'group_tms': int(row.group == 'TMS'), **fixed_scores(metrics, weights, fdr, networks)})
        pre_network = legacy.component_network_metrics(metrics['pre_z'][:, 1], labels)
        post_network = legacy.component_network_metrics(metrics['post_z'][:, 1], labels)
        for (network, metric_name) in NETWORK_ENDPOINTS:
            network_rows.append({'subject_id': subject_id, 'group': row.group, 'endpoint': f'G2_{network}_{metric_name}', 'network': network, 'metric': metric_name, 'delta_post_minus_pre': post_network[network][metric_name] - pre_network[network][metric_name]})
    return (pd.DataFrame(subject_rows), pd.DataFrame(network_rows))

def fixed_signature_inference(external_subjects: pd.DataFrame, main_subjects: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    endpoint = 'G2_fixed67_betaweighted_normalization'
    main_control = main_subjects.loc[main_subjects['group'] == 'Control', endpoint].to_numpy(float)
    main_tms = main_subjects.loc[main_subjects['group'] == 'TMS', endpoint].to_numpy(float)
    discovery_g = hedges_g(main_control, main_tms)
    discovery_ci = bootstrap_hedges_g(main_control, main_tms, seed=120)
    direction = 1.0 if discovery_g >= 0 else -1.0
    sesoi = 0.5 * abs(discovery_g)
    discovery_row = pd.DataFrame([{'endpoint': endpoint, 'control_n': len(main_control), 'tms_n': len(main_tms), 'control_mean': float(main_control.mean()), 'tms_mean': float(main_tms.mean()), 'TMS_minus_Control_mean': float(main_tms.mean() - main_control.mean()), 'hedges_g': discovery_g, 'hedges_g_ci_low_boot': discovery_ci[0], 'hedges_g_ci_high_boot': discovery_ci[1], 'discovery_direction': int(direction), 'SESOI_external_dz': sesoi}])
    rows: list[dict[str, object]] = []
    endpoints = ('G2_fixed67_betaweighted_normalization', 'G2_fixed67_signweighted_normalization', 'G2_positive59_mean_normalization', 'G2_VIS_SomMot_positive53_mean_normalization')
    for ((reference, alignment), frame) in external_subjects.groupby(['reference', 'alignment'], sort=False):
        for (endpoint_index, endpoint_name) in enumerate(endpoints):
            if endpoint_name != endpoints[0] and (not (reference == 'healthy_main' and alignment == 'separate_orthogonal')):
                continue
            for (subset, selected) in (('all29', frame), ('exclude_sub-35', frame.loc[frame['participant_id'] != 'sub-35'])):
                if subset != 'all29' and (not (reference == 'healthy_main' and alignment == 'separate_orthogonal')):
                    continue
                oriented = direction * selected[endpoint_name].to_numpy(float)
                summary = bootstrap_mean_dz(oriented, seed=300 + endpoint_index * 20 + len(rows))
                if summary['dz_ci_low_boot'] > 0 and summary['p_signflip_greater'] < 0.05 and (summary['dz'] >= sesoi):
                    decision = 'support'
                elif summary['dz_ci_high_boot'] < sesoi:
                    decision = 'not_reproduced'
                else:
                    decision = 'inconclusive'
                rows.append({'endpoint': endpoint_name, 'reference': reference, 'alignment': alignment, 'subset': subset, 'discovery_direction_multiplier': int(direction), 'SESOI_dz': sesoi, **summary, 'raw_mean_before_direction_orientation': float(selected[endpoint_name].mean()), 'decision': decision})
    return (discovery_row, pd.DataFrame(rows))

def random_rotation(rng: np.random.Generator) -> np.ndarray:
    return Rotation.random(random_state=rng).as_matrix()

def bijective_spin(x: np.ndarray, y: np.ndarray, centroids: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    observed = float(pearsonr(x, y).statistic)
    left = centroids[:200]
    right = centroids[200:]
    reflection = np.diag([-1.0, 1.0, 1.0])
    rng = np.random.default_rng(SEED + 880)
    null = np.empty(N_SPIN)
    for index in range(N_SPIN):
        left_rotation = random_rotation(rng)
        right_rotation = reflection @ left_rotation @ reflection
        permuted = np.empty(400)
        for (offset, points, rotation) in ((0, left, left_rotation), (200, right, right_rotation)):
            rotated = points @ rotation.T
            cost = np.sum((rotated[:, None, :] - points[None, :, :]) ** 2, axis=2)
            (source_index, target_index) = linear_sum_assignment(cost)
            permuted[offset + target_index] = x[offset + source_index]
        null[index] = pearsonr(permuted, y).statistic
    p = float((1 + np.sum(np.abs(null) >= abs(observed))) / (N_SPIN + 1))
    return ({'n_roi': 400, 'pearson_r': observed, 'spearman_rho': float(spearmanr(x, y).statistic), 'spin_p_two_sided': p, 'n_spin': N_SPIN, 'spin_assignment': 'hemisphere-preserving bijective Hungarian'}, null)

def max_t_signflip(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, float)
    observed = matrix.mean(axis=0) / (matrix.std(axis=0, ddof=1) / math.sqrt(len(matrix)))
    rng = np.random.default_rng(SEED + 1900)
    exceed = np.zeros(matrix.shape[1], dtype=int)
    batch = 1000
    completed = 0
    while completed < N_PERM:
        size = min(batch, N_PERM - completed)
        signed = matrix[None, :, :] * rng.choice((-1.0, 1.0), size=(size, len(matrix), 1))
        means = signed.mean(axis=1)
        sds = signed.std(axis=1, ddof=1)
        t_null = means / (sds / math.sqrt(len(matrix)))
        maxima = np.max(np.abs(t_null), axis=1)
        exceed += np.sum(maxima[:, None] >= np.abs(observed)[None, :], axis=0)
        completed += size
    return (observed, (exceed + 1) / (N_PERM + 1))

def network_inference(main_network: pd.DataFrame, external_network: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    discovery_rows: list[dict[str, object]] = []
    weights: dict[str, float] = {}
    pooled_sd: dict[str, float] = {}
    for (endpoint, frame) in main_network.groupby('endpoint', sort=False):
        control = frame.loc[frame['group'] == 'Control', 'delta_post_minus_pre'].to_numpy(float)
        tms = frame.loc[frame['group'] == 'TMS', 'delta_post_minus_pre'].to_numpy(float)
        g = hedges_g(control, tms)
        ci = bootstrap_hedges_g(control, tms, seed=2100 + len(discovery_rows))
        weights[endpoint] = g
        pooled_sd[endpoint] = math.sqrt(((len(control) - 1) * control.var(ddof=1) + (len(tms) - 1) * tms.var(ddof=1)) / (len(control) + len(tms) - 2))
        discovery_rows.append({'endpoint': endpoint, 'control_n': len(control), 'tms_n': len(tms), 'control_mean_delta': float(control.mean()), 'tms_mean_delta': float(tms.mean()), 'TMS_minus_Control_delta': float(tms.mean() - control.mean()), 'hedges_g': g, 'hedges_g_ci_low_boot': ci[0], 'hedges_g_ci_high_boot': ci[1], 'direction_multiplier': 1 if g >= 0 else -1, 'SESOI_dz': 0.5 * abs(g)})
    discovery_table = pd.DataFrame(discovery_rows)
    wide = external_network.pivot(index='participant_id', columns='endpoint', values='delta_post_minus_pre')
    order = discovery_table['endpoint'].tolist()
    oriented_matrix = np.column_stack([np.sign(weights[endpoint]) * wide[endpoint].to_numpy(float) for endpoint in order])
    (t_observed, p_max_t) = max_t_signflip(oriented_matrix)
    external_rows: list[dict[str, object]] = []
    for (index, endpoint) in enumerate(order):
        values = oriented_matrix[:, index]
        summary = bootstrap_mean_dz(values, seed=2300 + index)
        sesoi = 0.5 * abs(weights[endpoint])
        if summary['dz_ci_low_boot'] > 0 and p_max_t[index] < 0.05 and (summary['dz'] >= sesoi):
            decision = 'support'
        elif summary['dz_ci_high_boot'] < sesoi:
            decision = 'not_reproduced'
        else:
            decision = 'inconclusive'
        external_rows.append({'endpoint': endpoint, 'discovery_hedges_g': weights[endpoint], 'discovery_direction_multiplier': int(np.sign(weights[endpoint])), 'SESOI_dz': sesoi, **summary, 'maxT_t_observed': float(t_observed[index]), 'p_maxT_FWER_two_sided': float(p_max_t[index]), 'decision': decision})
    scaled = np.column_stack([wide[endpoint].to_numpy(float) / wide[endpoint].std(ddof=1) for endpoint in order])
    main_weights = np.array([weights[endpoint] for endpoint in order])
    composite = scaled @ main_weights / np.sum(np.abs(main_weights))
    composite_summary = bootstrap_mean_dz(composite, seed=2450)
    external_rows.append({'endpoint': 'discovery_weighted_three_network_composite', 'discovery_hedges_g': np.nan, 'discovery_direction_multiplier': 1, 'SESOI_dz': float(0.5 * np.average(np.abs(main_weights), weights=np.abs(main_weights))), **composite_summary, 'maxT_t_observed': np.nan, 'p_maxT_FWER_two_sided': np.nan, 'decision': 'support' if composite_summary['dz_ci_low_boot'] > 0 and composite_summary['p_signflip_greater'] < 0.05 else 'inconclusive'})
    return (discovery_table, pd.DataFrame(external_rows))

def standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    return (values - values.mean()) / values.std(ddof=1)

def prognostic_models(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = standardize(data['post_score'].to_numpy(float))
    predictor = standardize(data['baseline_G2_fixed67_absweighted_deviation'].to_numpy(float))
    covariates = np.column_stack([standardize(data['pre_score'].to_numpy(float)), standardize(data['log_time_after_stroke'].to_numpy(float)), data['sex_male'].to_numpy(float)])
    reduced = np.column_stack([np.ones(len(data)), covariates])
    full = np.column_stack([np.ones(len(data)), predictor, covariates])
    model = sm.OLS(y, full).fit(cov_type='HC3')
    beta = float(model.params[1])
    t_observed = float(model.tvalues[1])
    reduced_model = sm.OLS(y, reduced).fit()
    rng = np.random.default_rng(SEED + 2600)
    exceed = 0
    for _ in range(N_PERM):
        permuted_y = reduced_model.fittedvalues + reduced_model.resid[rng.permutation(len(data))]
        permuted = sm.OLS(permuted_y, full).fit(cov_type='HC3')
        exceed += int(abs(float(permuted.tvalues[1])) >= abs(t_observed))
    permutation_p = float((exceed + 1) / (N_PERM + 1))
    boot = np.empty(N_BOOT)
    for index in range(N_BOOT):
        sample = rng.integers(0, len(data), len(data))
        try:
            boot[index] = sm.OLS(y[sample], full[sample]).fit(cov_type='HC3').params[1]
        except Exception:
            boot[index] = np.nan
    df_resid = len(data) - full.shape[1]
    partial_r2 = float(t_observed ** 2 / (t_observed ** 2 + df_resid))
    confidence = model.conf_int()[1]
    result_rows = [{'model': 'post_score_ANCOVA', 'outcome': 'z(post Vasserman); higher = worse residual outcome', 'predictor': 'z(baseline fixed67 abs-weighted G2 deviation)', 'covariates': 'z(pre Vasserman), z(log1p days), sex_male', 'n': len(data), 'standardized_beta': beta, 'hc3_se': float(model.bse[1]), 'hc3_ci_low': float(confidence[0]), 'hc3_ci_high': float(confidence[1]), 'bootstrap_ci_low': float(np.nanpercentile(boot, 2.5)), 'bootstrap_ci_high': float(np.nanpercentile(boot, 97.5)), 'hc3_p_two_sided': float(model.pvalues[1]), 'studentized_freedman_lane_p_two_sided': permutation_p, 'partial_r_squared': partial_r2, 'model_r_squared': float(sm.OLS(y, full).fit().rsquared), 'interpretation': 'positive beta means larger baseline deviation predicts a worse post score'}]
    improvement = standardize(data['improvement'].to_numpy(float))
    improvement_model = sm.OLS(improvement, full).fit(cov_type='HC3')
    ci_change = improvement_model.conf_int()[1]
    result_rows.append({'model': 'improvement_sensitivity', 'outcome': 'z(pre - post Vasserman); higher = more improvement', 'predictor': 'z(baseline fixed67 abs-weighted G2 deviation)', 'covariates': 'z(pre Vasserman), z(log1p days), sex_male', 'n': len(data), 'standardized_beta': float(improvement_model.params[1]), 'hc3_se': float(improvement_model.bse[1]), 'hc3_ci_low': float(ci_change[0]), 'hc3_ci_high': float(ci_change[1]), 'bootstrap_ci_low': np.nan, 'bootstrap_ci_high': np.nan, 'hc3_p_two_sided': float(improvement_model.pvalues[1]), 'studentized_freedman_lane_p_two_sided': np.nan, 'partial_r_squared': float(improvement_model.tvalues[1] ** 2 / (improvement_model.tvalues[1] ** 2 + df_resid)), 'model_r_squared': float(sm.OLS(improvement, full).fit().rsquared), 'interpretation': 'positive beta means larger baseline deviation predicts more improvement'})
    predictor_residual = sm.OLS(predictor, reduced).fit().resid
    outcome_residual = sm.OLS(y, reduced).fit().resid
    plot_data = data[['participant_id']].copy()
    plot_data['adjusted_baseline_G2_deviation_residual'] = predictor_residual
    plot_data['adjusted_post_score_residual'] = outcome_residual
    return (pd.DataFrame(result_rows), plot_data)

def nested_loocv(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    clinical = ['pre_score', 'log_time_after_stroke', 'sex_male']
    blocks = {'clinical_only': clinical, 'clinical_plus_fixed_baseline_G2': clinical + ['baseline_G2_fixed67_absweighted_deviation']}
    alphas = np.logspace(-4, 4, 33)
    y = data['post_score'].to_numpy(float)
    rows: list[dict[str, object]] = []
    loo = LeaveOneOut()
    for (fold, (train, test)) in enumerate(loo.split(data)):
        rows.append({'participant_id': data.iloc[test[0]]['participant_id'], 'fold': fold, 'model': 'training_mean', 'observed_post_score': float(y[test[0]]), 'predicted_post_score': float(y[train].mean()), 'best_alpha': np.nan})
        for (block, features) in blocks.items():
            x = data[features].to_numpy(float)
            best_alpha = None
            best_mae = np.inf
            inner = LeaveOneOut()
            for alpha in alphas:
                predictions = np.empty(len(train))
                x_train = x[train]
                y_train = y[train]
                for (inner_train, inner_test) in inner.split(x_train):
                    pipeline = Pipeline([('scale', StandardScaler()), ('ridge', Ridge(alpha=float(alpha)))])
                    pipeline.fit(x_train[inner_train], y_train[inner_train])
                    predictions[inner_test[0]] = pipeline.predict(x_train[inner_test])[0]
                mae = mean_absolute_error(y_train, predictions)
                if mae < best_mae:
                    best_mae = float(mae)
                    best_alpha = float(alpha)
            final = Pipeline([('scale', StandardScaler()), ('ridge', Ridge(alpha=float(best_alpha)))])
            final.fit(x[train], y[train])
            prediction = float(final.predict(x[test])[0])
            rows.append({'participant_id': data.iloc[test[0]]['participant_id'], 'fold': fold, 'model': block, 'observed_post_score': float(y[test[0]]), 'predicted_post_score': prediction, 'best_alpha': best_alpha})
        print(f'nested LOOCV fold {fold + 1:02d}/{len(data)}', flush=True)
    predictions = pd.DataFrame(rows)
    performance_rows: list[dict[str, object]] = []
    denominator = float(np.sum((y - y.mean()) ** 2))
    for (model_name, frame) in predictions.groupby('model', sort=False):
        observed = frame['observed_post_score'].to_numpy(float)
        predicted = frame['predicted_post_score'].to_numpy(float)
        performance_rows.append({'model': model_name, 'n': len(frame), 'MAE': float(mean_absolute_error(observed, predicted)), 'RMSE': float(math.sqrt(mean_squared_error(observed, predicted))), 'CV_R2': float(1 - np.sum((observed - predicted) ** 2) / denominator), 'pearson_r_descriptive': float(pearsonr(observed, predicted).statistic)})
    performance = pd.DataFrame(performance_rows)
    wide = predictions.pivot(index='participant_id', columns='model', values=['observed_post_score', 'predicted_post_score'])
    observed = wide['observed_post_score']['clinical_only'].to_numpy(float)
    clinical_error = np.abs(observed - wide['predicted_post_score']['clinical_only'].to_numpy(float))
    gradient_error = np.abs(observed - wide['predicted_post_score']['clinical_plus_fixed_baseline_G2'].to_numpy(float))
    paired_delta = clinical_error - gradient_error
    boot = bootstrap_mean_dz(paired_delta, seed=2900)
    increment = pd.DataFrame([{'comparison': 'clinical_only minus clinical_plus_fixed_baseline_G2', 'positive_delta_MAE_means_gradient_improves_prediction': True, 'delta_MAE': float(clinical_error.mean() - gradient_error.mean()), 'paired_delta_error_ci_low_boot': boot['mean_ci_low_boot'], 'paired_delta_error_ci_high_boot': boot['mean_ci_high_boot'], 'paired_error_dz': boot['dz'], 'p_paired_signflip_greater': boot['p_signflip_greater'], 'fraction_subjects_lower_error_with_gradient': float(np.mean(paired_delta > 0)), 'decision': 'support' if boot['mean_ci_low_boot'] > 0 and boot['p_signflip_greater'] < 0.05 else 'no_demonstrated_increment'}])
    paired = pd.DataFrame({'participant_id': wide.index, 'observed_post_score': observed, 'clinical_only_absolute_error': clinical_error, 'clinical_plus_gradient_absolute_error': gradient_error, 'clinical_minus_gradient_absolute_error': paired_delta})
    return (predictions, performance, pd.concat([increment, paired], ignore_index=True))

def bootstrap_smd(group_a: np.ndarray, group_b: np.ndarray, seed: int) -> tuple[float, float, float]:

    def smd(a: np.ndarray, b: np.ndarray) -> float:
        pooled = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
        return float((a.mean() - b.mean()) / pooled)
    observed = smd(group_a, group_b)
    rng = np.random.default_rng(SEED + seed)
    boot = np.empty(N_BOOT)
    for index in range(N_BOOT):
        boot[index] = smd(group_a[rng.integers(0, len(group_a), len(group_a))], group_b[rng.integers(0, len(group_b), len(group_b))])
    return (observed, float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5)))

def complete_case_audit(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    lesion_fields = ['hemisphere', 'size', 'Broca', 'Wernike', 'Basal', 'Thalamus']
    complete = np.ones(len(data), dtype=bool)
    for field in lesion_fields:
        complete &= data[field].notna().to_numpy() & (data[field].astype(str).str.lower().to_numpy() != 'n/a')
    data = data.copy()
    data['lesion_metadata_complete'] = complete
    rows: list[dict[str, object]] = []
    variables = (('pre_score', 'Baseline Vasserman'), ('post_score', 'Post Vasserman'), ('improvement', 'Vasserman improvement'), ('time_after_stroke', 'Days after stroke'), ('baseline_G2_fixed67_absweighted_deviation', 'Baseline fixed G2 deviation'))
    for (index, (variable, label)) in enumerate(variables):
        group_a = data.loc[complete, variable].to_numpy(float)
        group_b = data.loc[~complete, variable].to_numpy(float)
        (effect, low, high) = bootstrap_smd(group_a, group_b, 3200 + index)
        welch = ttest_ind(group_a, group_b, equal_var=False)
        rows.append({'variable': variable, 'label': label, 'effect_type': 'SMD complete13 minus incomplete16', 'complete_n': len(group_a), 'incomplete_n': len(group_b), 'complete_mean': float(group_a.mean()), 'incomplete_mean': float(group_b.mean()), 'effect': effect, 'ci_low_boot': low, 'ci_high_boot': high, 'p_welch_two_sided': float(welch.pvalue)})
    sex = pd.crosstab(data['lesion_metadata_complete'], data['sex'])
    aphasia = pd.crosstab(data['lesion_metadata_complete'], data['type'])
    sex_test = chi2_contingency(sex, correction=False)
    aphasia_test = chi2_contingency(aphasia, correction=False)
    categorical = pd.DataFrame([{'variable': 'sex', 'complete_n': int(complete.sum()), 'incomplete_n': int((~complete).sum()), 'chi_squared': float(sex_test.statistic), 'degrees_freedom': int(sex_test.dof), 'p_chi_squared': float(sex_test.pvalue), 'cramers_v': float(math.sqrt(sex_test.statistic / (len(data) * min(sex.shape[0] - 1, sex.shape[1] - 1))))}, {'variable': 'aphasia_type', 'complete_n': int(complete.sum()), 'incomplete_n': int((~complete).sum()), 'chi_squared': float(aphasia_test.statistic), 'degrees_freedom': int(aphasia_test.dof), 'p_chi_squared': float(aphasia_test.pvalue), 'cramers_v': float(math.sqrt(aphasia_test.statistic / (len(data) * min(aphasia.shape[0] - 1, aphasia.shape[1] - 1))))}])
    continuous = pd.DataFrame(rows)
    continuous['p_holm_five_continuous_audit'] = multipletests(continuous['p_welch_two_sided'], method='holm')[1]
    return (continuous, categorical)

def main() -> None:
    ensure_dirs()
    labels = legacy.load_labels()
    participants = pd.read_csv(PARTICIPANTS, sep='\t')
    participants['improvement'] = participants['pre_score'] - participants['post_score']
    participants['sex_male'] = (participants['sex'] == 'M').astype(int)
    participants['log_time_after_stroke'] = np.log1p(participants['time_after_stroke'].astype(float))
    discovery = pd.read_csv(DISCOVERY)
    if int(discovery['G2_FDR_significant'].astype(bool).sum()) != 67:
        raise RuntimeError('Frozen discovery G2 mask is not 67 parcels')
    healthy_main_fc = load_group_fc('schaefer', scale=400, group='main')
    healthy_holdout_fc = load_group_fc('schaefer', scale=400, group='holdout')
    (healthy_main_raw, healthy_main_lambdas) = fit_provided(healthy_main_fc)
    established = np.load(ESTABLISHED_REFERENCE)
    (healthy_main, _, main_reference_qc) = orthogonal_align(healthy_main_raw, established)
    (healthy_holdout_raw, healthy_holdout_lambdas) = fit_provided(healthy_holdout_fc)
    (healthy_holdout, _, holdout_reference_qc) = orthogonal_align(healthy_holdout_raw, healthy_main)
    references = {'healthy_main': healthy_main, 'healthy_holdout': healthy_holdout}
    np.save(REFERENCES / 'healthy_main_reference_locked_400x5.npy', healthy_main)
    np.save(REFERENCES / 'healthy_holdout_reference_locked_400x5.npy', healthy_holdout)
    (_, _, healthy_main_z) = reference_scale(healthy_main)
    reference_table = labels.copy()
    for component in range(5):
        reference_table[f'G{component + 1}_healthy_main_z'] = healthy_main_z[:, component]
    reference_table.to_csv(REFERENCES / 'healthy_main_reference_schaefer400.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame([{**{f'healthy_main_{key}': value for (key, value) in main_reference_qc.items()}, **{f'holdout_to_main_{key}': value for (key, value) in holdout_reference_qc.items()}, **{f'healthy_main_lambda_G{i + 1}': float(value) for (i, value) in enumerate(healthy_main_lambdas)}, **{f'healthy_holdout_lambda_G{i + 1}': float(value) for (i, value) in enumerate(healthy_holdout_lambdas)}, **{f'G{i + 1}_healthy_main_holdout_r': float(np.corrcoef(healthy_main[:, i], healthy_holdout[:, i])[0, 1]) for i in range(5)}}]).to_csv(TABLES / 'healthy_reference_audit.csv', index=False, encoding='utf-8-sig')
    (external_subjects, external_roi, external_network, alignment_qc) = process_external(participants, labels, discovery, references)
    external_subjects.to_csv(TABLES / 'external_subject_metrics_all_alignments.csv', index=False, encoding='utf-8-sig')
    external_roi.to_csv(TABLES / 'external_primary_roi_normalization_long.csv', index=False, encoding='utf-8-sig')
    external_network.to_csv(TABLES / 'external_primary_locked_network_metrics_long.csv', index=False, encoding='utf-8-sig')
    alignment_qc.to_csv(QA / 'gradient_alignment_qc.csv', index=False, encoding='utf-8-sig')
    (main_subjects, main_network) = process_main(labels, discovery, healthy_main)
    main_subjects.to_csv(TABLES / 'main_unified_subject_fixed_signature.csv', index=False, encoding='utf-8-sig')
    main_network.to_csv(TABLES / 'main_unified_locked_network_metrics_long.csv', index=False, encoding='utf-8-sig')
    (discovery_signature, signature_tests) = fixed_signature_inference(external_subjects, main_subjects)
    discovery_signature.to_csv(TABLES / 'frozen_signature_discovery_effect_and_SESOI.csv', index=False, encoding='utf-8-sig')
    signature_tests.to_csv(TABLES / 'external_fixed_G2_signature_tests.csv', index=False, encoding='utf-8-sig')
    primary_roi_map = external_roi.groupby(['roi', 'roi_name', 'network'], as_index=False).agg(external_mean_G1_normalization=('G1_normalization', 'mean'), external_mean_G2_normalization=('G2_normalization', 'mean'), external_sd_G2_normalization=('G2_normalization', 'std')).merge(discovery[['roi', 'adjusted_TMS_beta_G1', 'adjusted_TMS_beta_G2', 'fdr_q_G2', 'G2_FDR_significant']], on='roi', validate='one_to_one')
    centroids = legacy.schaefer_sphere_centroids()
    (spin_result, spin_null) = bijective_spin(primary_roi_map['adjusted_TMS_beta_G2'].to_numpy(float), primary_roi_map['external_mean_G2_normalization'].to_numpy(float), centroids)
    network_design = pd.get_dummies(primary_roi_map['network'], dtype=float).to_numpy(float)
    residual_discovery = sm.OLS(primary_roi_map['adjusted_TMS_beta_G2'], network_design).fit().resid
    residual_external = sm.OLS(primary_roi_map['external_mean_G2_normalization'], network_design).fit().resid
    spin_result['network_adjusted_pearson_r'] = float(pearsonr(residual_discovery, residual_external).statistic)
    primary_signature = signature_tests.loc[(signature_tests['endpoint'] == 'G2_fixed67_betaweighted_normalization') & (signature_tests['reference'] == 'healthy_main') & (signature_tests['alignment'] == 'separate_orthogonal') & (signature_tests['subset'] == 'all29')].iloc[0]
    spin_result['gatekeeping_primary_signature_passed'] = bool(primary_signature['decision'] == 'support')
    spin_result['spatial_replication_decision'] = 'support' if spin_result['gatekeeping_primary_signature_passed'] and spin_result['pearson_r'] > 0 and (spin_result['spin_p_two_sided'] < 0.05) else 'not_supported_under_gatekeeping'
    primary_roi_map.to_csv(TABLES / 'external_primary_mean_G1_G2_maps.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame([spin_result]).to_csv(TABLES / 'external_G2_spatial_replication_bijective_spin.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame({'spin': np.arange(N_SPIN), 'null_pearson_r': spin_null}).to_csv(TABLES / 'external_G2_bijective_spin_null.csv', index=False, encoding='utf-8-sig')
    (main_network_effects, external_network_tests) = network_inference(main_network, external_network)
    main_network_effects.to_csv(TABLES / 'main_locked_network_effects_and_SESOI.csv', index=False, encoding='utf-8-sig')
    external_network_tests.to_csv(TABLES / 'external_locked_network_replication_tests.csv', index=False, encoding='utf-8-sig')
    primary_subject = external_subjects.loc[(external_subjects['reference'] == 'healthy_main') & (external_subjects['alignment'] == 'separate_orthogonal')].drop(columns=['reference', 'alignment'])
    analysis_data = participants.merge(primary_subject, on='participant_id', validate='one_to_one')
    analysis_data.to_csv(TABLES / 'external_primary_subject_analysis_data.csv', index=False, encoding='utf-8-sig')
    clinical_summary = bootstrap_mean_dz(analysis_data['improvement'].to_numpy(float), seed=2550)
    clinical_change = pd.DataFrame([{'endpoint': 'Vasserman improvement = pre - post', 'n': len(analysis_data), 'pre_mean': float(analysis_data['pre_score'].mean()), 'pre_sd': float(analysis_data['pre_score'].std(ddof=1)), 'post_mean': float(analysis_data['post_score'].mean()), 'post_sd': float(analysis_data['post_score'].std(ddof=1)), **clinical_summary, 'positive_means_improvement': True, 'scope': 'general rehabilitation change; no treatment-specific inference'}])
    clinical_change.to_csv(TABLES / 'external_clinical_change.csv', index=False, encoding='utf-8-sig')
    analysis_data[['participant_id', 'pre_score', 'post_score', 'improvement']].to_csv(SOURCE_DATA / 'panel_a_clinical_change.csv', index=False, encoding='utf-8-sig')
    (prognostic, prognostic_plot) = prognostic_models(analysis_data)
    prognostic.to_csv(TABLES / 'external_baseline_G2_prognostic_models.csv', index=False, encoding='utf-8-sig')
    prognostic_plot.to_csv(SOURCE_DATA / 'panel_d_adjusted_prognostic_scatter.csv', index=False, encoding='utf-8-sig')
    (predictions, performance, increment_and_paired) = nested_loocv(analysis_data)
    predictions.to_csv(TABLES / 'external_baseline_only_nested_LOOCV_predictions.csv', index=False, encoding='utf-8-sig')
    performance.to_csv(TABLES / 'external_baseline_only_nested_LOOCV_performance.csv', index=False, encoding='utf-8-sig')
    increment_and_paired.to_csv(TABLES / 'external_baseline_only_prediction_increment.csv', index=False, encoding='utf-8-sig')
    (complete_continuous, complete_categorical) = complete_case_audit(analysis_data)
    complete_continuous.to_csv(TABLES / 'lesion_metadata_complete_case_continuous_audit.csv', index=False, encoding='utf-8-sig')
    complete_categorical.to_csv(TABLES / 'lesion_metadata_complete_case_categorical_audit.csv', index=False, encoding='utf-8-sig')
    signature_tests.to_csv(SOURCE_DATA / 'panel_a_fixed_G2_signature_alignment_sensitivity.csv', index=False, encoding='utf-8-sig')
    primary_roi_map.to_csv(SOURCE_DATA / 'panel_b_spatial_replication.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame({'spin': np.arange(N_SPIN), 'null_pearson_r': spin_null}).to_csv(SOURCE_DATA / 'panel_b_spin_null.csv', index=False, encoding='utf-8-sig')
    external_network_tests.to_csv(SOURCE_DATA / 'panel_c_locked_network_endpoints.csv', index=False, encoding='utf-8-sig')
    performance.to_csv(SOURCE_DATA / 'panel_e_prediction_performance.csv', index=False, encoding='utf-8-sig')
    increment_and_paired.to_csv(SOURCE_DATA / 'panel_e_paired_prediction_errors.csv', index=False, encoding='utf-8-sig')
    complete_continuous.to_csv(SOURCE_DATA / 'panel_f_complete_case_audit.csv', index=False, encoding='utf-8-sig')
    manifest = {'analysis_date': '2026-08-12', 'scope': 'general rehabilitation replication; not TMS-specific', 'n_external': len(analysis_data), 'n_main_primary_motion_qc': len(main_subjects), 'gradient_pipeline': 'Fisher-z -> strongest positive 10% -> cosine affinity -> diffusion maps', 'primary_alignment': 'pre and post separately aligned to the same fixed reference using orthogonal rotation only; no scale or translation', 'sensitivity_alignments': ['baseline_locked_orthogonal', 'joint_pair_orthogonal', 'separate_similarity_robust90', 'healthy_holdout'], 'healthy_reference': 'BrainSpace packaged independent healthy Schaefer400 main group FC', 'fixed_G2_ROI_count': int(discovery['G2_FDR_significant'].astype(bool).sum()), 'fixed_positive_ROI_count': int((discovery['G2_FDR_significant'].astype(bool) & (discovery['adjusted_TMS_beta_G2'] > 0)).sum()), 'fixed_VIS_SomMot_positive_ROI_count': int((discovery['G2_FDR_significant'].astype(bool) & (discovery['adjusted_TMS_beta_G2'] > 0) & discovery['network'].isin(['Visual', 'SomMot'])).sum()), 'primary_endpoint': 'G2_fixed67_betaweighted_normalization', 'SESOI_rule': '0.5 * abs(main unified Hedges g)', 'network_endpoints': [f'G2_{network}_{metric}' for (network, metric) in NETWORK_ENDPOINTS], 'prognostic_predictor': 'baseline_G2_fixed67_absweighted_deviation', 'prediction': 'outer LOOCV; inner LOOCV ridge; baseline-only variables', 'permutations': N_PERM, 'bootstrap_resamples': N_BOOT, 'spins': N_SPIN, 'input_hashes': {'frozen_discovery_map': sha256(DISCOVERY), 'participants': sha256(PARTICIPANTS), 'frozen_protocol': sha256(HERE / 'FROZEN_VALIDATION_PROTOCOL.md')}, 'constructs_untestable': ['TMS versus Control effect', 'lesion-gradient distance', 'lesion G1-G2 span', 'lesion-gradient-MMSE moderation or mediation', 'disconnectome mechanism']}
    (HERE / 'analysis_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print('confirmatory external validation complete', flush=True)
if __name__ == '__main__':
    main()
