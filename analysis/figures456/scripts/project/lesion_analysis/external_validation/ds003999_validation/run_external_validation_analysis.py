from __future__ import annotations
import os as _portable_os
from pathlib import Path as _PortablePath

def _portable_path(kind, relative):
    project = _PortablePath(_portable_os.environ['PAPER_PROJECT_ROOT'])
    root = project if kind == 'project' else project.parent
    return str(root / relative)
import json
import math
import re
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from brainspace.datasets import load_group_fc, load_parcellation
from brainspace.gradient import GradientMaps
from neuromaps.datasets import fetch_fslr
from scipy.linalg import orthogonal_procrustes
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.stats import pearsonr, spearmanr, t as t_dist, ttest_1samp, ttest_rel
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
SEED = 20260811
N_COMPONENTS = 5
N_PERM = 5000
N_SPIN = 5000
N_ML_REPEATS = 20
N_ML_SPLITS = 5
DATA_ROOT = Path(_portable_path('project', 'new_dataset'))
HERE = Path(__file__).resolve().parent
TABLES = HERE / 'tables'
REFERENCES = HERE / 'references'
GRADIENTS = HERE / 'recomputed_gradients'
QC = HERE / 'qa'
FC_ROOT = DATA_ROOT / '(400,400)' / '(400,400)'
LEGACY_ROOT = DATA_ROOT / '(400,5)' / '(400,5)'
ROI_ORDER = LEGACY_ROOT / 'QC' / 'Schaefer400_ROI_order.csv'
PARTICIPANTS = DATA_ROOT / 'participants.tsv'
DISCOVERY_SOURCE = Path(_portable_path('project', 'lesion_analysis')) / 'lesion_gradient_distance' / 'figures' / 'nature' / 'lesion_final' / 'provided_code_G1_G2_surface_source_data.csv'
NETWORK_MAP = {'Vis': 'Visual', 'SomMot': 'SomMot', 'DorsAttn': 'DorsAttn', 'SalVentAttn': 'SalVentAttn', 'Limbic': 'Limbic', 'Cont': 'Control', 'Default': 'Default'}
NETWORK_ORDER = ('Visual', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Control', 'Default')

def ensure_dirs() -> None:
    for path in (TABLES, REFERENCES, GRADIENTS / 'pre', GRADIENTS / 'post', QC):
        path.mkdir(parents=True, exist_ok=True)

def parse_network(name: str) -> str:
    token = name.split('_')[2]
    if token not in NETWORK_MAP:
        raise RuntimeError(f'Unknown Schaefer network token: {token}')
    return NETWORK_MAP[token]

def load_labels() -> pd.DataFrame:
    labels = pd.read_csv(ROI_ORDER)
    labels = labels.rename(columns={'roi_index_1based': 'roi'})
    if labels['roi'].tolist() != list(range(1, 401)):
        raise RuntimeError('ROI order is not 1..400')
    labels['network'] = labels['roi_name'].map(parse_network)
    labels['hemisphere'] = labels['roi_name'].str.split('_').str[1]
    return labels

def fit_gradient(connectivity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(connectivity, dtype=float)
    if matrix.shape != (400, 400):
        raise RuntimeError(f'Unexpected FC shape: {matrix.shape}')
    if not np.isfinite(matrix).all():
        raise RuntimeError('FC contains non-finite values')
    if np.nanmin(matrix) < -1.000001 or np.nanmax(matrix) > 1.000001:
        raise RuntimeError('Supplied FC is not a correlation matrix in [-1, 1]')
    matrix = (matrix + matrix.T) / 2.0
    np.fill_diagonal(matrix, 0.0)
    model = GradientMaps(n_components=N_COMPONENTS, approach='dm', kernel='normalized_angle', random_state=SEED)
    model.fit(matrix, sparsity=0.9)
    gradients = np.asarray(model.gradients_, dtype=float)
    lambdas = np.asarray(model.lambdas_, dtype=float)
    if gradients.shape != (400, N_COMPONENTS):
        raise RuntimeError(f'Unexpected gradient shape: {gradients.shape}')
    return (gradients, lambdas)

def similarity_align(source: np.ndarray, target: np.ndarray, fit_mask: np.ndarray | None=None) -> tuple[np.ndarray, dict[str, float]]:
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if source.shape != target.shape:
        raise RuntimeError(f'Alignment shape mismatch: {source.shape} vs {target.shape}')
    if fit_mask is None:
        fit_mask = np.ones(source.shape[0], dtype=bool)
    fit_mask = np.asarray(fit_mask, dtype=bool)
    source_mean = source[fit_mask].mean(axis=0, keepdims=True)
    target_mean = target[fit_mask].mean(axis=0, keepdims=True)
    source_centered = source[fit_mask] - source_mean
    target_centered = target[fit_mask] - target_mean
    (rotation, singular_sum) = orthogonal_procrustes(source_centered, target_centered)
    denominator = float(np.sum(source_centered ** 2))
    scale = float(singular_sum / denominator)
    aligned = (source - source_mean) @ rotation * scale + target_mean
    return (aligned, {'fit_n_roi': int(fit_mask.sum()), 'scale': scale, 'fit_r': float(np.corrcoef(aligned[fit_mask].ravel(), target[fit_mask].ravel())[0, 1]), 'fit_rms': float(np.sqrt(np.mean((aligned[fit_mask] - target[fit_mask]) ** 2))), 'fit_G1_r': float(np.corrcoef(aligned[fit_mask, 0], target[fit_mask, 0])[0, 1]), 'fit_G2_r': float(np.corrcoef(aligned[fit_mask, 1], target[fit_mask, 1])[0, 1])})

def orient_master(master: np.ndarray, labels: pd.DataFrame) -> tuple[np.ndarray, dict[str, object]]:
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
    return (oriented, {'G1_rule': 'Default mean > mean(Visual, SomMot)', 'G1_sign_multiplier': g1_sign, 'G2_rule': 'Visual mean > SomMot mean', 'G2_sign_multiplier': g2_sign})

def locate_array(root: Path, session: str, subject_id: str) -> Path:
    matches = sorted((root / session).glob(f'*{subject_id}_ses-{session}_*.npy'))
    if len(matches) != 1:
        raise RuntimeError(f'Expected one {session} array for {subject_id}; found {len(matches)}')
    return matches[0]

def robust_align_pair(pre_raw: np.ndarray, post_raw: np.ndarray, master: np.ndarray, master_sd: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float], dict[str, float]]:
    (pre_initial, _) = similarity_align(pre_raw, master)
    residual = np.linalg.norm((pre_initial - master) / master_sd, axis=1)
    keep_n = int(round(0.9 * len(residual)))
    keep_index = np.argsort(residual)[:keep_n]
    robust_mask = np.zeros(len(residual), dtype=bool)
    robust_mask[keep_index] = True
    (pre, pre_qc) = similarity_align(pre_raw, master, robust_mask)
    (post, post_qc) = similarity_align(post_raw, master, robust_mask)
    return (pre, post, robust_mask, pre_qc, post_qc)

def normalization_maps(pre: np.ndarray, post: np.ndarray, master: np.ndarray, master_mean: np.ndarray, master_sd: np.ndarray) -> dict[str, np.ndarray]:
    pre_z = (pre - master_mean) / master_sd
    post_z = (post - master_mean) / master_sd
    ref_z = (master - master_mean) / master_sd
    pre_g1 = np.abs(pre_z[:, 0] - ref_z[:, 0])
    post_g1 = np.abs(post_z[:, 0] - ref_z[:, 0])
    pre_g2 = np.abs(pre_z[:, 1] - ref_z[:, 1])
    post_g2 = np.abs(post_z[:, 1] - ref_z[:, 1])
    pre_g1g2 = np.linalg.norm(pre_z[:, :2] - ref_z[:, :2], axis=1)
    post_g1g2 = np.linalg.norm(post_z[:, :2] - ref_z[:, :2], axis=1)
    return {'pre_z': pre_z, 'post_z': post_z, 'pre_dev_G1': pre_g1, 'post_dev_G1': post_g1, 'pre_dev_G2': pre_g2, 'post_dev_G2': post_g2, 'pre_dev_G1G2': pre_g1g2, 'post_dev_G1G2': post_g1g2, 'norm_G1': pre_g1 - post_g1, 'norm_G2': pre_g2 - post_g2, 'norm_G1G2': pre_g1g2 - post_g1g2}

def component_network_metrics(values: np.ndarray, labels: pd.DataFrame) -> dict[str, dict[str, float]]:
    centroids: dict[str, float] = {}
    dispersion: dict[str, float] = {}
    network_array = labels['network'].to_numpy(str)
    for network in NETWORK_ORDER:
        selected = np.asarray(values)[network_array == network]
        centroids[network] = float(np.mean(selected))
        dispersion[network] = float(np.std(selected, ddof=1))
    output: dict[str, dict[str, float]] = {}
    for network in NETWORK_ORDER:
        differences = [abs(centroids[network] - centroids[other]) for other in NETWORK_ORDER if other != network]
        between = float(np.mean(differences))
        output[network] = {'mean': centroids[network], 'Dispersion': dispersion[network], 'betweenDisp': between, 'Segregation': between / dispersion[network]}
    return output

def signflip_p(values: np.ndarray, n_perm: int=N_PERM, alternative: str='two-sided') -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(SEED + len(values) * 17)
    observed = float(np.mean(values))
    null = np.mean(values[None, :] * rng.choice((-1.0, 1.0), size=(n_perm, len(values))), axis=1)
    if alternative == 'greater':
        return float((1 + np.sum(null >= observed)) / (n_perm + 1))
    return float((1 + np.sum(np.abs(null) >= abs(observed))) / (n_perm + 1))

def one_sample_summary(values: np.ndarray, endpoint: str) -> dict[str, float | str]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    se = sd / math.sqrt(n)
    critical = float(t_dist.ppf(0.975, n - 1))
    result = ttest_1samp(values, 0.0)
    return {'endpoint': endpoint, 'n': n, 'mean': mean, 'sd': sd, 'ci_low': mean - critical * se, 'ci_high': mean + critical * se, 't': float(result.statistic), 'p_parametric_two_sided': float(result.pvalue), 'p_signflip_two_sided': signflip_p(values), 'p_signflip_greater': signflip_p(values, alternative='greater')}

def standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    sd = float(np.std(values, ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        raise RuntimeError('Cannot standardize a constant vector')
    return (values - float(np.mean(values))) / sd

def hc3_fit(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    inv = np.linalg.pinv(x.T @ x)
    beta = inv @ x.T @ y
    residual = y - x @ beta
    leverage = np.sum(x @ inv * x, axis=1)
    adjusted = residual / np.clip(1.0 - leverage, 1e-08, None)
    meat = x.T @ ((adjusted ** 2)[:, None] * x)
    covariance = inv @ meat @ inv
    se = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    t_values = beta / se
    return (beta, se, t_values)

def freedman_lane_p(y: np.ndarray, predictor: np.ndarray, covariates: np.ndarray, n_perm: int=N_PERM, seed_offset: int=0) -> tuple[float, float, float, float, float, float]:
    y = np.asarray(y, dtype=float)
    predictor = np.asarray(predictor, dtype=float)
    covariates = np.asarray(covariates, dtype=float)
    reduced = np.column_stack([np.ones(len(y)), covariates])
    full = np.column_stack([np.ones(len(y)), predictor, covariates])
    (beta, se, t_values) = hc3_fit(y, full)
    df = len(y) - full.shape[1]
    p_hc3 = float(2 * t_dist.sf(abs(t_values[1]), df))
    reduced_beta = np.linalg.pinv(reduced) @ y
    fitted = reduced @ reduced_beta
    residual = y - fitted
    rng = np.random.default_rng(SEED + seed_offset)
    exceed = 0
    for _ in range(n_perm):
        permuted_y = fitted + residual[rng.permutation(len(residual))]
        (_, _, permuted_t) = hc3_fit(permuted_y, full)
        exceed += int(abs(permuted_t[1]) >= abs(t_values[1]))
    p_perm = float((exceed + 1) / (n_perm + 1))
    ci_low = float(beta[1] - t_dist.ppf(0.975, df) * se[1])
    ci_high = float(beta[1] + t_dist.ppf(0.975, df) * se[1])
    partial_r2 = float(t_values[1] ** 2 / (t_values[1] ** 2 + df))
    return (float(beta[1]), float(se[1]), ci_low, ci_high, p_hc3, p_perm, partial_r2)

def _surface_coords(surface_path: Path) -> np.ndarray:
    image = nib.load(str(surface_path))
    return np.asarray(image.darrays[0].data, dtype=float)

def schaefer_sphere_centroids() -> np.ndarray:
    surfaces = fetch_fslr()
    lh = _surface_coords(Path(surfaces['sphere'].L))
    rh = _surface_coords(Path(surfaces['sphere'].R))
    atlas = np.asarray(load_parcellation('schaefer', 400, join=True))
    if atlas.size != len(lh) + len(rh):
        raise RuntimeError('fsLR/Schaefer surface mismatch')
    centroids = np.full((400, 3), np.nan)
    for roi in range(400):
        if roi < 200:
            coordinates = lh[atlas[:len(lh)] == roi + 1]
        else:
            coordinates = rh[atlas[len(lh):] == roi + 1]
        centroid = coordinates.mean(axis=0)
        centroids[roi] = centroid / np.linalg.norm(centroid)
    return centroids

def random_rotation(rng: np.random.Generator) -> np.ndarray:
    (u1, u2, u3) = rng.random(3)
    quaternion = np.array([np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2), np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2), np.sqrt(u1) * np.sin(2.0 * np.pi * u3), np.sqrt(u1) * np.cos(2.0 * np.pi * u3)])
    return Rotation.from_quat(quaternion).as_matrix()

def spin_correlation(x: np.ndarray, y: np.ndarray, centroids: np.ndarray) -> tuple[float, float, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    observed = float(pearsonr(x, y).statistic)
    left = centroids[:200]
    right = centroids[200:]
    left_tree = cKDTree(left)
    right_tree = cKDTree(right)
    reflection = np.diag([-1.0, 1.0, 1.0])
    rng = np.random.default_rng(SEED + 991)
    null = np.empty(N_SPIN, dtype=float)
    for index in range(N_SPIN):
        rotation_left = random_rotation(rng)
        rotation_right = reflection @ rotation_left @ reflection
        left_mapping = left_tree.query(left @ rotation_left.T, k=1)[1]
        right_mapping = right_tree.query(right @ rotation_right.T, k=1)[1]
        permuted = np.concatenate([x[left_mapping], x[200 + right_mapping]])
        null[index] = pearsonr(permuted, y).statistic
    p = float((1 + np.sum(np.abs(null) >= abs(observed))) / (N_SPIN + 1))
    return (observed, p, null)

def nested_cv_feature_blocks(data: pd.DataFrame, feature_blocks: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows: list[dict[str, object]] = []
    performance_rows: list[dict[str, object]] = []
    alphas = np.logspace(-3, 3, 13)
    y = data['post_score'].to_numpy(float)
    for repeat in range(N_ML_REPEATS):
        outer = KFold(n_splits=N_ML_SPLITS, shuffle=True, random_state=SEED + repeat)
        for (fold, (train, test)) in enumerate(outer.split(data)):
            inner = KFold(n_splits=4, shuffle=True, random_state=SEED + 1000 + repeat * 10 + fold)
            for (block, features) in feature_blocks.items():
                pipeline = Pipeline([('scale', StandardScaler()), ('ridge', Ridge())])
                search = GridSearchCV(pipeline, {'ridge__alpha': alphas}, scoring='neg_mean_absolute_error', cv=inner, n_jobs=1)
                x = data[features].to_numpy(float)
                search.fit(x[train], y[train])
                predicted = search.predict(x[test])
                for (subject_index, prediction) in zip(test, predicted):
                    prediction_rows.append({'repeat': repeat, 'fold': fold, 'feature_block': block, 'subject_id': data.iloc[subject_index]['participant_id'], 'observed_post_score': y[subject_index], 'predicted_post_score': float(prediction), 'best_alpha': float(search.best_params_['ridge__alpha'])})
        repeat_predictions = pd.DataFrame([row for row in prediction_rows if row['repeat'] == repeat])
        for (block, subset) in repeat_predictions.groupby('feature_block'):
            observed = subset['observed_post_score'].to_numpy(float)
            predicted = subset['predicted_post_score'].to_numpy(float)
            performance_rows.append({'repeat': repeat, 'feature_block': block, 'n': len(subset), 'mae': float(mean_absolute_error(observed, predicted)), 'rmse': float(np.sqrt(mean_squared_error(observed, predicted))), 'r2': float(r2_score(observed, predicted)), 'pearson_r': float(pearsonr(observed, predicted).statistic)})
        print(f'nested CV repeat {repeat + 1}/{N_ML_REPEATS}', flush=True)
    return (pd.DataFrame(prediction_rows), pd.DataFrame(performance_rows))

def main() -> None:
    ensure_dirs()
    labels = load_labels()
    participants = pd.read_csv(PARTICIPANTS, sep='\t')
    participants['improvement'] = participants['pre_score'] - participants['post_score']
    participants['sex_male'] = (participants['sex'] == 'M').astype(int)
    participants['log_time_after_stroke'] = np.log1p(participants['time_after_stroke'].astype(float))
    if len(participants) != 29 or participants['participant_id'].nunique() != 29:
        raise RuntimeError('Expected 29 unique participants')
    discovery = pd.read_csv(DISCOVERY_SOURCE)
    if discovery['roi'].tolist() != labels['roi'].tolist():
        raise RuntimeError('Discovery and external ROI indices differ')
    if discovery['roi_name'].tolist() != labels['roi_name'].tolist():
        raise RuntimeError('Discovery and external ROI names differ')
    discovery_beta = discovery['adjusted_TMS_beta_G2'].to_numpy(float)
    discovery_fdr = discovery['G2_FDR_significant'].astype(bool).to_numpy()
    discovery_weights = np.where(discovery_fdr, discovery_beta, 0.0)
    if int(discovery_fdr.sum()) != 67:
        raise RuntimeError('Expected 67 fixed discovery G2 ROIs')
    healthy_main_fc = load_group_fc('schaefer', scale=400, group='main')
    healthy_holdout_fc = load_group_fc('schaefer', scale=400, group='holdout')
    (master_raw, master_lambdas) = fit_gradient(healthy_main_fc)
    (master, orientation) = orient_master(master_raw, labels)
    master_mean = master.mean(axis=0)
    master_sd = master.std(axis=0, ddof=1)
    master_z = (master - master_mean) / master_sd
    (holdout_raw, holdout_lambdas) = fit_gradient(healthy_holdout_fc)
    (holdout, holdout_qc) = similarity_align(holdout_raw, master)
    holdout_z = (holdout - master_mean) / master_sd
    reference = labels.copy()
    for component in range(N_COMPONENTS):
        reference[f'G{component + 1}_reference_raw'] = master[:, component]
        reference[f'G{component + 1}_reference_z'] = master_z[:, component]
    reference.to_csv(REFERENCES / 'healthy_reference_schaefer400.csv', index=False, encoding='utf-8-sig')
    np.save(REFERENCES / 'healthy_reference_400x5.npy', master)
    pd.DataFrame([{**orientation, **{f'main_lambda_G{i + 1}': float(value) for (i, value) in enumerate(master_lambdas)}, **{f'holdout_lambda_G{i + 1}': float(value) for (i, value) in enumerate(holdout_lambdas)}, **{f'holdout_{key}': value for (key, value) in holdout_qc.items()}, **{f'G{i + 1}_main_holdout_r': float(np.corrcoef(master_z[:, i], holdout_z[:, i])[0, 1]) for i in range(N_COMPONENTS)}}]).to_csv(TABLES / 'healthy_reference_validation.csv', index=False, encoding='utf-8-sig')
    subject_rows: list[dict[str, object]] = []
    roi_rows: list[dict[str, object]] = []
    network_rows: list[dict[str, object]] = []
    alignment_rows: list[dict[str, object]] = []
    for participant in participants.itertuples(index=False):
        subject_id = participant.participant_id
        pre_fc = np.load(locate_array(FC_ROOT, 'pre', subject_id))
        post_fc = np.load(locate_array(FC_ROOT, 'post', subject_id))
        (pre_raw, pre_lambdas) = fit_gradient(pre_fc)
        (post_raw, post_lambdas) = fit_gradient(post_fc)
        (pre, post, robust_mask, pre_qc, post_qc) = robust_align_pair(pre_raw, post_raw, master, master_sd)
        (pre_all, pre_all_qc) = similarity_align(pre_raw, master)
        (post_all, post_all_qc) = similarity_align(post_raw, master)
        metrics = normalization_maps(pre, post, master, master_mean, master_sd)
        metrics_all = normalization_maps(pre_all, post_all, master, master_mean, master_sd)
        np.save(GRADIENTS / 'pre' / f'{subject_id}_pre_healthy_aligned_400x5.npy', pre)
        np.save(GRADIENTS / 'post' / f'{subject_id}_post_healthy_aligned_400x5.npy', post)
        legacy_pre = np.load(locate_array(LEGACY_ROOT, 'pre', subject_id))
        legacy_post = np.load(locate_array(LEGACY_ROOT, 'post', subject_id))
        legacy_pre_g1 = float(np.corrcoef(legacy_pre[:, 0], metrics['pre_z'][:, 0])[0, 1])
        legacy_pre_g2 = float(np.corrcoef(legacy_pre[:, 1], metrics['pre_z'][:, 1])[0, 1])
        legacy_post_g1 = float(np.corrcoef(legacy_post[:, 0], metrics['post_z'][:, 0])[0, 1])
        legacy_post_g2 = float(np.corrcoef(legacy_post[:, 1], metrics['post_z'][:, 1])[0, 1])
        weight_denominator = float(np.sum(np.abs(discovery_weights)))
        betaweighted_signature = float(np.sum(discovery_weights * metrics['norm_G2']) / weight_denominator)
        signweighted_signature = float(np.mean(np.sign(discovery_weights[discovery_fdr]) * metrics['norm_G2'][discovery_fdr]))
        baseline_signature_deviation = float(np.sum(np.abs(discovery_weights) * metrics['pre_dev_G2']) / weight_denominator)
        row = {'participant_id': subject_id, 'mean_baseline_deviation_G1': float(np.mean(metrics['pre_dev_G1'])), 'mean_baseline_deviation_G2': float(np.mean(metrics['pre_dev_G2'])), 'mean_baseline_deviation_G1G2': float(np.mean(metrics['pre_dev_G1G2'])), 'mean_normalization_G1': float(np.mean(metrics['norm_G1'])), 'mean_normalization_G2': float(np.mean(metrics['norm_G2'])), 'mean_normalization_G1G2': float(np.mean(metrics['norm_G1G2'])), 'allroi_mean_normalization_G1': float(np.mean(metrics_all['norm_G1'])), 'allroi_mean_normalization_G2': float(np.mean(metrics_all['norm_G2'])), 'allroi_mean_normalization_G1G2': float(np.mean(metrics_all['norm_G1G2'])), 'G2_FDR_betaweighted_signature': betaweighted_signature, 'G2_FDR_signweighted_signature': signweighted_signature, 'baseline_G2_FDR_weighted_deviation': baseline_signature_deviation, 'G2_positive59_mean_normalization': float(np.mean(metrics['norm_G2'][discovery_fdr & (discovery_beta > 0)])), 'G2_negative8_mean_normalization': float(np.mean(metrics['norm_G2'][discovery_fdr & (discovery_beta < 0)])), 'legacy_pre_G1_r': legacy_pre_g1, 'legacy_pre_G2_r': legacy_pre_g2, 'legacy_post_G1_r': legacy_post_g1, 'legacy_post_G2_r': legacy_post_g2, 'robust_fit_roi_n': int(robust_mask.sum())}
        subject_rows.append(row)
        alignment_rows.append({'participant_id': subject_id, **{f'pre_lambda_G{i + 1}': float(value) for (i, value) in enumerate(pre_lambdas)}, **{f'post_lambda_G{i + 1}': float(value) for (i, value) in enumerate(post_lambdas)}, **{f'pre_robust_{key}': value for (key, value) in pre_qc.items()}, **{f'post_robust_{key}': value for (key, value) in post_qc.items()}, **{f'pre_allroi_{key}': value for (key, value) in pre_all_qc.items()}, **{f'post_allroi_{key}': value for (key, value) in post_all_qc.items()}, 'excluded_alignment_rois': ';'.join((str(i + 1) for i in np.where(~robust_mask)[0]))})
        for roi_index in range(400):
            roi_rows.append({'participant_id': subject_id, 'roi': roi_index + 1, 'roi_name': labels.iloc[roi_index]['roi_name'], 'network': labels.iloc[roi_index]['network'], 'G1_pre_z': metrics['pre_z'][roi_index, 0], 'G1_post_z': metrics['post_z'][roi_index, 0], 'G2_pre_z': metrics['pre_z'][roi_index, 1], 'G2_post_z': metrics['post_z'][roi_index, 1], 'G1_normalization': metrics['norm_G1'][roi_index], 'G2_normalization': metrics['norm_G2'][roi_index], 'G1G2_normalization': metrics['norm_G1G2'][roi_index], 'discovery_adjusted_TMS_beta_G2': discovery_beta[roi_index], 'discovery_G2_FDR': bool(discovery_fdr[roi_index])})
        for component in (1, 2):
            pre_component = component_network_metrics(metrics['pre_z'][:, component - 1], labels)
            post_component = component_network_metrics(metrics['post_z'][:, component - 1], labels)
            for network in NETWORK_ORDER:
                for metric_name in ('mean', 'Dispersion', 'betweenDisp', 'Segregation'):
                    network_rows.append({'participant_id': subject_id, 'gradient': f'G{component}', 'network': network, 'metric': metric_name, 'pre': pre_component[network][metric_name], 'post': post_component[network][metric_name], 'delta_post_minus_pre': post_component[network][metric_name] - pre_component[network][metric_name]})
        print(f'recomputed gradients: {subject_id}', flush=True)
    subject_metrics = participants.merge(pd.DataFrame(subject_rows), on='participant_id', validate='one_to_one')
    roi_metrics = pd.DataFrame(roi_rows)
    network_metrics = pd.DataFrame(network_rows)
    alignment = pd.DataFrame(alignment_rows)
    subject_metrics.to_csv(TABLES / 'subject_level_external_validation_metrics.csv', index=False, encoding='utf-8-sig')
    roi_metrics.to_csv(TABLES / 'subject_roi_normalization_long.csv', index=False, encoding='utf-8-sig')
    network_metrics.to_csv(TABLES / 'subject_network_gradient_metrics_long.csv', index=False, encoding='utf-8-sig')
    alignment.to_csv(QC / 'gradient_alignment_qc.csv', index=False, encoding='utf-8-sig')
    clinical_paired = ttest_rel(subject_metrics['pre_score'], subject_metrics['post_score'])
    clinical_summary = pd.DataFrame([{'n': len(subject_metrics), 'pre_mean': subject_metrics['pre_score'].mean(), 'pre_sd': subject_metrics['pre_score'].std(ddof=1), 'post_mean': subject_metrics['post_score'].mean(), 'post_sd': subject_metrics['post_score'].std(ddof=1), 'improvement_definition': 'pre_score - post_score; positive indicates lower post-treatment impairment', 'improvement_mean': subject_metrics['improvement'].mean(), 'improvement_sd': subject_metrics['improvement'].std(ddof=1), 'paired_t': clinical_paired.statistic, 'paired_p_two_sided': clinical_paired.pvalue, 'improvement_signflip_p_two_sided': signflip_p(subject_metrics['improvement'].to_numpy(float))}])
    clinical_summary.to_csv(TABLES / 'clinical_change_summary.csv', index=False, encoding='utf-8-sig')
    global_rows = []
    for endpoint in ('mean_normalization_G1', 'mean_normalization_G2', 'mean_normalization_G1G2'):
        global_rows.append(one_sample_summary(subject_metrics[endpoint].to_numpy(float), endpoint))
    global_tests = pd.DataFrame(global_rows)
    global_tests['fdr_q_three_endpoints'] = multipletests(global_tests['p_signflip_two_sided'], method='fdr_bh')[1]
    global_tests.to_csv(TABLES / 'global_normalization_tests.csv', index=False, encoding='utf-8-sig')
    signature_tests = pd.DataFrame([one_sample_summary(subject_metrics[column].to_numpy(float), column) for column in ('G2_FDR_betaweighted_signature', 'G2_FDR_signweighted_signature', 'G2_positive59_mean_normalization', 'G2_negative8_mean_normalization')])
    signature_tests['fdr_q_four_signature_endpoints'] = multipletests(signature_tests['p_signflip_two_sided'], method='fdr_bh')[1]
    signature_tests.to_csv(TABLES / 'fixed_discovery_signature_tests.csv', index=False, encoding='utf-8-sig')
    mean_map = roi_metrics.groupby(['roi', 'roi_name', 'network'], as_index=False).agg(external_mean_G1_normalization=('G1_normalization', 'mean'), external_mean_G2_normalization=('G2_normalization', 'mean'), external_mean_G1G2_normalization=('G1G2_normalization', 'mean'), external_sd_G2_normalization=('G2_normalization', 'std')).merge(discovery[['roi', 'adjusted_TMS_beta_G1', 'adjusted_TMS_beta_G2', 'fdr_q_G2', 'G2_FDR_significant']], on='roi', validate='one_to_one')
    centroids = schaefer_sphere_centroids()
    (spatial_r, spatial_p, spatial_null) = spin_correlation(mean_map['adjusted_TMS_beta_G2'].to_numpy(float), mean_map['external_mean_G2_normalization'].to_numpy(float), centroids)
    spatial_spearman = float(spearmanr(mean_map['adjusted_TMS_beta_G2'], mean_map['external_mean_G2_normalization']).statistic)
    network_design = pd.get_dummies(mean_map['network'], drop_first=False, dtype=float)
    residual_discovery = sm.OLS(mean_map['adjusted_TMS_beta_G2'].to_numpy(float), network_design.to_numpy(float)).fit().resid
    residual_external = sm.OLS(mean_map['external_mean_G2_normalization'].to_numpy(float), network_design.to_numpy(float)).fit().resid
    network_adjusted_r = float(pearsonr(residual_discovery, residual_external).statistic)
    mean_map.to_csv(TABLES / 'external_mean_normalization_maps.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame({'permutation': np.arange(N_SPIN), 'null_pearson_r': spatial_null}).to_csv(TABLES / 'spatial_spin_null_G2.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame([{'n_roi': 400, 'pearson_r': spatial_r, 'spearman_rho': spatial_spearman, 'spin_p_two_sided': spatial_p, 'network_adjusted_pearson_r': network_adjusted_r, 'spin_method': '5000 hemispherically coupled fsLR sphere rotations with nearest-parcel reassignment'}]).to_csv(TABLES / 'spatial_replication_G2.csv', index=False, encoding='utf-8-sig')
    y = standardize(subject_metrics['improvement'].to_numpy(float))
    signature = standardize(subject_metrics['G2_FDR_betaweighted_signature'].to_numpy(float))
    covariates = np.column_stack([standardize(subject_metrics['pre_score'].to_numpy(float)), standardize(subject_metrics['log_time_after_stroke'].to_numpy(float)), subject_metrics['sex_male'].to_numpy(float)])
    (beta, se, ci_low, ci_high, p_hc3, p_perm, partial_r2) = freedman_lane_p(y, signature, covariates, seed_offset=201)
    full_x = np.column_stack([np.ones(len(y)), signature, covariates])
    model = sm.OLS(y, full_x).fit(cov_type='HC3')
    cov_only = np.column_stack([np.ones(len(y)), covariates])
    adjusted_x = sm.OLS(signature, cov_only).fit().resid
    adjusted_y = sm.OLS(y, cov_only).fit().resid
    subject_metrics['adjusted_G2_signature_residual'] = adjusted_x
    subject_metrics['adjusted_recovery_residual'] = adjusted_y
    subject_metrics.to_csv(TABLES / 'subject_level_external_validation_metrics.csv', index=False, encoding='utf-8-sig')
    clinical_model = pd.DataFrame([{'outcome': 'z(pre_score - post_score); positive = greater recovery', 'predictor': 'z(fixed discovery 67-ROI beta-weighted G2 normalization signature)', 'covariates': 'z(pre_score), z(log1p(time_after_stroke)), sex_male', 'n': len(y), 'standardized_beta': beta, 'hc3_se': se, 'hc3_ci_low': ci_low, 'hc3_ci_high': ci_high, 'hc3_p_two_sided': p_hc3, 'studentized_freedman_lane_p_two_sided': p_perm, 'partial_r_squared': partial_r2, 'model_r_squared': model.rsquared}])
    clinical_model.to_csv(TABLES / 'clinical_G2_signature_model.csv', index=False, encoding='utf-8-sig')
    sensitivity_rows = []
    sensitivity_specs: list[tuple[str, pd.DataFrame, np.ndarray]] = []
    sensitivity_specs.append(('full_primary', subject_metrics, covariates))
    no35 = subject_metrics.loc[subject_metrics['participant_id'] != 'sub-35'].copy()
    no35_cov = np.column_stack([standardize(no35['pre_score']), standardize(no35['log_time_after_stroke']), no35['sex_male'].to_numpy(float)])
    sensitivity_specs.append(('exclude_sub-35', no35, no35_cov))
    for (name, frame, covs) in sensitivity_specs:
        result = freedman_lane_p(standardize(frame['improvement']), standardize(frame['G2_FDR_betaweighted_signature']), covs, n_perm=5000, seed_offset=300 + len(frame))
        sensitivity_rows.append({'model': name, 'n': len(frame), 'standardized_beta': result[0], 'hc3_se': result[1], 'hc3_ci_low': result[2], 'hc3_ci_high': result[3], 'hc3_p': result[4], 'permutation_p': result[5], 'partial_r_squared': result[6]})
    lesion_subset = subject_metrics.dropna(subset=['size', 'Broca', 'Wernike', 'Basal', 'Thalamus']).copy()
    ordinal_maps = {'size': {'S': 0, 'M': 1, 'H': 2}, 'Broca': {'N': 0, 'P': 1, 'Y': 2}, 'Wernike': {'N': 0, 'P': 1, 'Y': 2}, 'Basal': {'N': 0, 'P': 1, 'Y': 2}, 'Thalamus': {'N': 0, 'P': 1, 'Y': 2}}
    for (index, (name, mapping)) in enumerate(ordinal_maps.items()):
        lesion_value = lesion_subset[name].map(mapping).to_numpy(float)
        lesion_covariates = np.column_stack([standardize(lesion_subset['pre_score']), standardize(lesion_value)])
        result = freedman_lane_p(standardize(lesion_subset['improvement']), standardize(lesion_subset['G2_FDR_betaweighted_signature']), lesion_covariates, n_perm=3000, seed_offset=400 + index)
        sensitivity_rows.append({'model': f'lesion_subset_adjust_{name}', 'n': len(lesion_subset), 'standardized_beta': result[0], 'hc3_se': result[1], 'hc3_ci_low': result[2], 'hc3_ci_high': result[3], 'hc3_p': result[4], 'permutation_p': result[5], 'partial_r_squared': result[6]})
    sensitivity = pd.DataFrame(sensitivity_rows)
    lesion_mask = sensitivity['model'].str.startswith('lesion_subset')
    sensitivity.loc[lesion_mask, 'fdr_q_five_lesion_models'] = multipletests(sensitivity.loc[lesion_mask, 'permutation_p'], method='fdr_bh')[1]
    sensitivity.to_csv(TABLES / 'clinical_G2_signature_sensitivity_models.csv', index=False, encoding='utf-8-sig')
    network_test_rows = []
    for ((gradient, network, metric_name), subset) in network_metrics.groupby(['gradient', 'network', 'metric']):
        summary = one_sample_summary(subset['delta_post_minus_pre'].to_numpy(float), f'{gradient}_{network}_{metric_name}')
        summary.update({'gradient': gradient, 'network': network, 'metric': metric_name})
        network_test_rows.append(summary)
    network_tests = pd.DataFrame(network_test_rows)
    network_tests['fdr_q_56_tests'] = multipletests(network_tests['p_signflip_two_sided'], method='fdr_bh')[1]
    network_tests.to_csv(TABLES / 'exploratory_network_metric_tests.csv', index=False, encoding='utf-8-sig')
    wide_network = network_metrics.pivot_table(index='participant_id', columns=['gradient', 'network', 'metric'], values='pre')
    wide_network.columns = [f'pre_{g}_{n}_{m}' for (g, n, m) in wide_network.columns]
    wide_network = wide_network.reset_index()
    ml_data = subject_metrics.merge(wide_network, on='participant_id', validate='one_to_one')
    ml_data['pre_G2_Visual_SomMot_separation'] = np.abs(ml_data['pre_G2_Visual_mean'] - ml_data['pre_G2_SomMot_mean'])
    clinical_features = ['pre_score', 'log_time_after_stroke', 'sex_male']
    between_features = [f'pre_G{gradient}_{network}_betweenDisp' for gradient in (1, 2) for network in NETWORK_ORDER]
    targeted_features = [f'pre_G2_{network}_{metric}' for network in ('Visual', 'SomMot') for metric in ('mean', 'Dispersion', 'betweenDisp', 'Segregation')] + ['pre_G2_Visual_SomMot_separation']
    healthy_features = ['mean_baseline_deviation_G1', 'mean_baseline_deviation_G2', 'mean_baseline_deviation_G1G2', 'baseline_G2_FDR_weighted_deviation']
    feature_blocks = {'clinical': clinical_features, 'clinical_plus_G1G2_betweenDisp': clinical_features + between_features, 'clinical_plus_targeted_G2_VIS_SomMot': clinical_features + targeted_features, 'clinical_plus_healthy_deviation': clinical_features + healthy_features, 'clinical_plus_all_baseline_gradient': clinical_features + list(dict.fromkeys(between_features + targeted_features + healthy_features))}
    dictionary_rows = []
    for (block, features) in feature_blocks.items():
        for feature in features:
            dictionary_rows.append({'feature_block': block, 'feature': feature, 'timing': 'baseline', 'outcome': 'post_score', 'baseline_score_always_included': True})
    pd.DataFrame(dictionary_rows).to_csv(TABLES / 'ml_feature_dictionary.csv', index=False, encoding='utf-8-sig')
    (ml_predictions, ml_performance) = nested_cv_feature_blocks(ml_data, feature_blocks)
    ml_predictions.to_csv(TABLES / 'ml_nested_cv_predictions.csv', index=False, encoding='utf-8-sig')
    ml_performance.to_csv(TABLES / 'ml_nested_cv_performance_by_repeat.csv', index=False, encoding='utf-8-sig')
    ml_summary_rows = []
    baseline_mae = ml_performance.loc[ml_performance['feature_block'] == 'clinical'].set_index('repeat')['mae']
    for (block, subset) in ml_performance.groupby('feature_block'):
        values = subset['mae'].to_numpy(float)
        aligned = subset.set_index('repeat')['mae']
        delta = baseline_mae.loc[aligned.index].to_numpy(float) - aligned.to_numpy(float)
        ml_summary_rows.append({'feature_block': block, 'n_repeats': len(values), 'mean_mae': np.mean(values), 'sd_mae': np.std(values, ddof=1), 'mae_percentile_2.5': np.percentile(values, 2.5), 'mae_percentile_97.5': np.percentile(values, 97.5), 'mean_delta_mae_clinical_minus_block': np.mean(delta), 'positive_delta_means_lower_error_than_clinical': True, 'descriptive_repeat_fraction_better_than_clinical': np.mean(delta > 0)})
    pd.DataFrame(ml_summary_rows).to_csv(TABLES / 'ml_nested_cv_performance_summary.csv', index=False, encoding='utf-8-sig')
    manifest = {'dataset': 'OpenNeuro ds003999 v1.0.2 local derivative matrices', 'n': 29, 'treatment_design': 'single longitudinal stroke rehabilitation cohort; no TMS/control groups', 'fc_input': 'subject/session Schaefer400 Pearson correlation matrices; diagonal=1; no Fisher-z inversion', 'healthy_reference': 'BrainSpace packaged Schaefer400 healthy main group FC', 'healthy_reference_validation': 'BrainSpace packaged independent healthy holdout group FC', 'gradient_pipeline': 'diffusion maps, normalized-angle kernel, 90% sparsity, five components', 'alignment_primary': 'five-dimensional similarity Procrustes using baseline-derived 90% lowest-residual ROIs; identical ROI mask for pre and post', 'alignment_limitation': 'voxelwise lesion masks unavailable for most participants, so the robust mask is not a verified intact-tissue mask', 'normalization': 'absolute healthy-reference deviation at pre minus absolute deviation at post; positive values are closer to healthy at post', 'primary_discovery_signature': 'fixed 67-ROI G2 FDR map from discovery adjusted TMS-versus-control model; beta-weighted projection', 'clinical_model': 'standardized Vasserman improvement adjusted for baseline score, log time after stroke, and sex; HC3 plus studentized Freedman-Lane permutation', 'spatial_test': 'hemispherically coupled 5000-spin test on fsLR sphere centroids', 'ml': '20 repeats of nested five-fold CV with inner four-fold alpha tuning; all imaging predictors are baseline-only; external feature-family test, not frozen-model transport', 'motion_limitation': 'no FD/QC covariate was supplied with the local FC matrices', 'random_seed': SEED}
    (HERE / 'analysis_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print('analysis complete', flush=True)
if __name__ == '__main__':
    main()
