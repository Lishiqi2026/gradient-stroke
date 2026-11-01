from __future__ import annotations
import os as _portable_os
from pathlib import Path as _PortablePath

def _portable_path(kind, relative):
    project = _PortablePath(_portable_os.environ['PAPER_PROJECT_ROOT'])
    root = project if kind == 'project' else project.parent
    return str(root / relative)
import itertools
import json
import math
import re
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import linear_sum_assignment
from scipy.stats import t as student_t
from statsmodels.stats.multitest import multipletests
ROOT = Path(_portable_path('work', ''))
HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results_abcd' / 'tables'
GRADIENT_RESULTS = HERE / 'lesion_adjusted_gradient_value' / 'tables'
OUT = HERE / 'lesion_matched_clinical_mediation'
TABLES = OUT / 'tables'
SEED = 20260803
N_BOOT = 1000
N_PERM = 5000
TIGHT_MATCH_CALIPER = 1.25
OUTCOMES = ('NIHSS', 'ADL', 'MMSE', 'HAMD', 'VFT')
PRIMARY_OUTCOMES = ('NIHSS', 'ADL')
LOWER_IS_BETTER = {'NIHSS', 'HAMD'}

def ensure_dirs() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)

def normalize_subject_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace('\\.0$', '', regex=True)

def normalize_location(value: object) -> str:
    text = re.sub('\\s+', '', str(value).strip())
    text = text.replace(',', '、').replace('，', '、')
    return text

def coarse_location(value: object) -> str:
    text = normalize_location(value)
    if any((word in text for word in ('脑桥', '桥脑', '延髓', '桥臂'))):
        return 'brainstem_any'
    if any((word in text for word in ('额', '顶', '颞', '枕', '岛叶', '胼胝体', '半卵圆中心'))):
        return 'lobar_or_mixed'
    return 'deep_periventricular'

def normalized_side(value: object) -> str:
    text = str(value).strip()
    if '左' in text:
        return 'left'
    if '右' in text:
        return 'right'
    return 'other'

def load_clinical() -> pd.DataFrame:
    candidates = []
    for workbook in ROOT.glob('*.xlsx'):
        frame = pd.read_excel(workbook, sheet_name=0)
        if {'Group', 'NIHSS_pre', 'NIHSS_post', '部位', '左右侧'}.issubset(frame.columns):
            candidates.append((workbook, frame))
    if len(candidates) != 1:
        raise RuntimeError(f'Expected one clinical workbook; found {len(candidates)}')
    (_, frame) = candidates[0]
    frame = frame.rename(columns={frame.columns[0]: 'subject_id', '年龄': 'age', '性别': 'sex_text', '左右侧': 'clinical_side_text', '部位': 'location_text', 'Group': 'group', 'ESRS': 'esrs'})
    frame['subject_id'] = normalize_subject_id(frame['subject_id'])
    frame['group_tms'] = (frame['group'].astype(str).str.upper() == 'TMS').astype(int)
    frame['sex_male'] = frame['sex_text'].astype(str).str.contains('男').astype(int)
    frame['location_normalized'] = frame['location_text'].map(normalize_location)
    frame['coarse_location'] = frame['location_text'].map(coarse_location)
    frame['clinical_side'] = frame['clinical_side_text'].map(normalized_side)
    return frame

def oriented_values(frame: pd.DataFrame, outcome: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    before = frame[f'{outcome}_pre'].to_numpy(float)
    after = frame[f'{outcome}_post'].to_numpy(float)
    sign = -1.0 if outcome in LOWER_IS_BETTER else 1.0
    oriented_before = sign * before
    oriented_after = sign * after
    improvement = oriented_after - oriented_before
    return (oriented_before, oriented_after, improvement)

def ols_coefficient(y: np.ndarray, x: np.ndarray, index: int) -> float:
    return float(np.linalg.lstsq(x, y, rcond=None)[0][index])

def exact_group_permutation_ancova(before: np.ndarray, after: np.ndarray, group: np.ndarray) -> tuple[float, float]:
    x = np.column_stack([np.ones(len(group)), before, group])
    observed = ols_coefficient(after, x, 2)
    n_tms = int(group.sum())
    values = []
    for selected in itertools.combinations(range(len(group)), n_tms):
        permuted = np.zeros(len(group), dtype=float)
        permuted[list(selected)] = 1.0
        px = np.column_stack([np.ones(len(group)), before, permuted])
        values.append(ols_coefficient(after, px, 2))
    values = np.asarray(values)
    p_value = float(np.count_nonzero(np.abs(values) >= abs(observed) - 1e-12) / len(values))
    return (observed, p_value)

def bootstrap_two_group_difference(values: np.ndarray, group: np.ndarray, rng: np.random.Generator, n_boot: int=N_BOOT) -> tuple[float, float]:
    control = values[group == 0]
    tms = values[group == 1]
    boot = np.empty(n_boot)
    for i in range(n_boot):
        boot[i] = rng.choice(tms, len(tms), replace=True).mean() - rng.choice(control, len(control), replace=True).mean()
    return (float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975)))

def location_distributions(clinical: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    exact = clinical.groupby(['location_normalized', 'location_text', 'group'], dropna=False).size().rename('n').reset_index()
    coarse = clinical.groupby(['coarse_location', 'clinical_side', 'group'], dropna=False).size().rename('n').reset_index()
    return (exact, coarse)

def exact_location_subgroup_analysis(clinical: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    subgroup = clinical.loc[clinical['location_normalized'] == '基底节、侧脑室旁'].copy()
    rng = np.random.default_rng(SEED + 100)
    rows = []
    for outcome in OUTCOMES:
        (before, after, improvement) = oriented_values(subgroup, outcome)
        group = subgroup['group_tms'].to_numpy(int)
        (coefficient, permutation_p) = exact_group_permutation_ancova(before, after, group)
        model = sm.OLS(after, np.column_stack([np.ones(len(group)), before, group])).fit(cov_type='HC3')
        (low, high) = bootstrap_two_group_difference(improvement, group, rng)
        rows.append({'outcome': outcome, 'control_n': int((group == 0).sum()), 'tms_n': int((group == 1).sum()), 'control_pre_mean': float(subgroup.loc[group == 0, f'{outcome}_pre'].mean()), 'control_post_mean': float(subgroup.loc[group == 0, f'{outcome}_post'].mean()), 'tms_pre_mean': float(subgroup.loc[group == 1, f'{outcome}_pre'].mean()), 'tms_post_mean': float(subgroup.loc[group == 1, f'{outcome}_post'].mean()), 'control_improvement_mean': float(improvement[group == 0].mean()), 'tms_improvement_mean': float(improvement[group == 1].mean()), 'improvement_difference_tms_minus_control': float(improvement[group == 1].mean() - improvement[group == 0].mean()), 'improvement_difference_boot_ci_low': low, 'improvement_difference_boot_ci_high': high, 'baseline_adjusted_group_coefficient': coefficient, 'baseline_adjusted_hc3_ci_low': float(model.conf_int()[2, 0]), 'baseline_adjusted_hc3_ci_high': float(model.conf_int()[2, 1]), 'baseline_adjusted_hc3_p': float(model.pvalues[2]), 'baseline_adjusted_exact_permutation_p': permutation_p})
    results = pd.DataFrame(rows)
    results['permutation_fdr_q'] = multipletests(results['baseline_adjusted_exact_permutation_p'], method='fdr_bh')[1]
    keep = ['subject_id', 'group', 'clinical_side', 'location_text', *[column for outcome in OUTCOMES for column in (f'{outcome}_pre', f'{outcome}_post')]]
    return (subgroup[keep].copy(), results)

def load_joint_cohort(clinical: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, float]]:
    lesion = pd.read_csv(RESULTS / 'B_participant_lesion_metrics.csv', dtype={'subject_id': str})
    compartment = pd.read_csv(RESULTS / 'B2_anatomical_compartment_metrics.csv', dtype={'subject_id': str})
    spatial = pd.read_csv(RESULTS / 'E_spatial_lesion_phenotypes.csv', dtype={'subject_id': str})
    topology = pd.read_csv(RESULTS / 'C2_lesion_topology_pca_subject_scores.csv', dtype={'subject_id': str})
    xtract_long = pd.read_csv(RESULTS / 'F_xtract_system_probability_burden.csv', dtype={'subject_id': str})
    xtract = xtract_long.pivot(index='subject_id', columns='system', values='expected_overlap_ml').reset_index()
    xtract.columns = ['subject_id' if column == 'subject_id' else f'xtract_{column}_ml' for column in xtract.columns]
    cortical = pd.read_csv(RESULTS / 'D_gradient_clinical_subject_metrics.csv', dtype={'subject_id': str})
    subcortical = pd.read_csv(RESULTS / 'D2_subcortical_gradient_subject_metrics.csv', dtype={'subject_id': str})
    baseline_gradient = pd.read_csv(GRADIENT_RESULTS / 'analysis_cohort.csv', dtype={'subject_id': str})
    clinical_columns = ['subject_id', 'group', 'group_tms', 'age', 'sex_male', 'esrs', 'clinical_side', 'location_text', 'location_normalized', 'coarse_location', *[column for outcome in OUTCOMES for column in (f'{outcome}_pre', f'{outcome}_post')]]
    frame = clinical[clinical_columns].copy()
    frame = frame.merge(lesion[['subject_id', 'log1p_lesion_volume', 'mask_lateralization_index', 'registration_manual_status']], on='subject_id', how='inner')
    frame = frame.merge(compartment[['subject_id', 'outside_mni_brain_fraction_pct', 'brainstem_fraction_pct', 'deep_subcortical_gray_fraction_pct', 'cerebral_white_matter_fraction_pct', 'cerebral_cortex_fraction_pct']], on='subject_id', how='left')
    frame = frame.merge(spatial[['subject_id', 'absolute_centroid_x_mm', 'centroid_y_mm', 'centroid_z_mm', 'maximum_extent_mm']], on='subject_id', how='left')
    frame = frame.merge(topology.drop(columns='group'), on='subject_id', how='left')
    frame = frame.merge(xtract, on='subject_id', how='left')
    frame = frame.merge(cortical[['subject_id', 'g1_reorganization_rms', 'g2_reorganization_rms']], on='subject_id', how='left')
    frame = frame.merge(subcortical[['subject_id', 'subcortical_5d_baseline_dispersion', 'subcortical_5d_normalized_reorganization']], on='subject_id', how='left')
    frame = frame.merge(baseline_gradient[['subject_id', 'cortical_G1_sd']], on='subject_id', how='left')
    for column in ('brainstem_fraction_pct', 'deep_subcortical_gray_fraction_pct', 'cerebral_white_matter_fraction_pct', 'cerebral_cortex_fraction_pct'):
        frame[f'log1p_{column}'] = np.log1p(frame[column].astype(float))
    for system in ('motor_projection', 'thalamic_sensory_radiations', 'middle_cerebellar_peduncle'):
        column = f'xtract_{system}_ml'
        frame[f'log1p_{column}'] = np.log1p(frame[column].fillna(0).astype(float))
    match_features = ['log1p_lesion_volume', 'mask_lateralization_index', 'log1p_brainstem_fraction_pct', 'log1p_deep_subcortical_gray_fraction_pct', 'log1p_cerebral_white_matter_fraction_pct', 'log1p_cerebral_cortex_fraction_pct', 'absolute_centroid_x_mm', 'centroid_y_mm', 'centroid_z_mm', 'maximum_extent_mm', 'log1p_xtract_motor_projection_ml', 'log1p_xtract_thalamic_sensory_radiations_ml', 'log1p_xtract_middle_cerebellar_peduncle_ml', 'age', 'sex_male', 'esrs']
    weights = {column: 1.0 for column in match_features}
    weights.update({'age': 0.35, 'sex_male': 0.25, 'esrs': 0.35})
    if frame[match_features].isna().any().any():
        missing = frame[match_features].isna().sum()
        raise RuntimeError(f'Missing matching features: {missing[missing > 0].to_dict()}')
    return (frame, match_features, weights)

def standardized_features(frame: pd.DataFrame, features: list[str], weights: dict[str, float]) -> np.ndarray:
    values = frame[features].to_numpy(float)
    scale = values.std(axis=0, ddof=1)
    scale[scale < 1e-12] = 1.0
    z = (values - values.mean(axis=0)) / scale
    weight_vector = np.sqrt(np.asarray([weights[column] for column in features], dtype=float))
    return z * weight_vector

def optimal_pairs(frame: pd.DataFrame, features: list[str], weights: dict[str, float]) -> pd.DataFrame:
    z = standardized_features(frame, features, weights)
    zframe = pd.DataFrame(z, index=frame.index, columns=features)
    rows = []
    pair_number = 0
    for ((location, side), stratum) in frame.groupby(['coarse_location', 'clinical_side'], sort=True):
        control = stratum.loc[stratum['group_tms'] == 0]
        tms = stratum.loc[stratum['group_tms'] == 1]
        if control.empty or tms.empty:
            continue
        cz = zframe.loc[control.index].to_numpy(float)
        tz = zframe.loc[tms.index].to_numpy(float)
        distances = np.sqrt(((cz[:, None, :] - tz[None, :, :]) ** 2).mean(axis=2))
        (c_index, t_index) = linear_sum_assignment(distances)
        for (ci, ti) in zip(c_index, t_index):
            pair_number += 1
            rows.append({'pair_id': pair_number, 'coarse_location': location, 'clinical_side': side, 'control_subject_id': control.iloc[ci]['subject_id'], 'tms_subject_id': tms.iloc[ti]['subject_id'], 'standardized_rms_distance': float(distances[ci, ti])})
    return pd.DataFrame(rows)

def matched_subject_frame(frame: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    records = []
    indexed = frame.set_index('subject_id', drop=False)
    for pair in pairs.itertuples(index=False):
        for subject_id in (pair.control_subject_id, pair.tms_subject_id):
            record = indexed.loc[str(subject_id)].to_dict()
            record['pair_id'] = int(pair.pair_id)
            record['match_distance'] = float(pair.standardized_rms_distance)
            records.append(record)
    return pd.DataFrame(records)

def standardized_mean_difference(control: np.ndarray, tms: np.ndarray) -> float:
    pooled = math.sqrt((np.var(control, ddof=1) + np.var(tms, ddof=1)) / 2.0)
    if pooled < 1e-12:
        return 0.0
    return float((np.mean(tms) - np.mean(control)) / pooled)

def balance_table(full: pd.DataFrame, matched: pd.DataFrame, features: list[str], label: str) -> pd.DataFrame:
    rows = []
    for feature in features:
        before_control = full.loc[full.group_tms == 0, feature].to_numpy(float)
        before_tms = full.loc[full.group_tms == 1, feature].to_numpy(float)
        after_control = matched.loc[matched.group_tms == 0, feature].to_numpy(float)
        after_tms = matched.loc[matched.group_tms == 1, feature].to_numpy(float)
        rows.append({'match_set': label, 'feature': feature, 'smd_before': standardized_mean_difference(before_control, before_tms), 'smd_after': standardized_mean_difference(after_control, after_tms)})
    return pd.DataFrame(rows)

def pair_fixed_design(frame: pd.DataFrame, before: np.ndarray, group: np.ndarray, extra: np.ndarray | None=None) -> np.ndarray:
    pair_dummies = pd.get_dummies(frame['pair_id'].astype(str), drop_first=True, dtype=float).to_numpy()
    columns = [np.ones(len(frame)), before, group]
    if extra is not None:
        if extra.ndim == 1:
            extra = extra[:, None]
        columns.extend([extra[:, column] for column in range(extra.shape[1])])
    columns.extend([pair_dummies[:, column] for column in range(pair_dummies.shape[1])])
    return np.column_stack(columns)

def bootstrap_pair_ancova(frame: pd.DataFrame, outcome: str, rng: np.random.Generator, n_boot: int=N_BOOT) -> tuple[float, float]:
    pair_ids = frame['pair_id'].drop_duplicates().to_numpy()
    coefficients = []
    for _ in range(n_boot):
        sampled = rng.choice(pair_ids, len(pair_ids), replace=True)
        blocks = []
        for (new_id, old_id) in enumerate(sampled):
            block = frame.loc[frame.pair_id == old_id].copy()
            block['pair_id'] = new_id
            blocks.append(block)
        boot = pd.concat(blocks, ignore_index=True)
        (before, after, _) = oriented_values(boot, outcome)
        group = boot['group_tms'].to_numpy(float)
        x = pair_fixed_design(boot, before, group)
        coefficients.append(ols_coefficient(after, x, 2))
    return (float(np.quantile(coefficients, 0.025)), float(np.quantile(coefficients, 0.975)))

def paired_permutation_ancova(frame: pd.DataFrame, outcome: str, rng: np.random.Generator, n_perm: int=N_PERM) -> tuple[float, float]:
    (before, after, _) = oriented_values(frame, outcome)
    observed_group = frame['group_tms'].to_numpy(float)
    dummies = pair_dummies(frame)
    observed_x = np.column_stack([np.ones(len(frame)), before, observed_group, dummies])
    observed = ols_coefficient(after, observed_x, 2)
    pair_ids = frame['pair_id'].drop_duplicates().to_numpy()
    possible = 2 ** len(pair_ids)
    if possible <= n_perm:
        assignments = itertools.product((0, 1), repeat=len(pair_ids))
    else:
        assignments = (rng.integers(0, 2, len(pair_ids)) for _ in range(n_perm))
    null = []
    for assignment in assignments:
        group = observed_group.copy()
        for (swap, pair_id) in zip(assignment, pair_ids):
            if swap:
                mask = frame['pair_id'].to_numpy() == pair_id
                group[mask] = 1.0 - group[mask]
        x = np.column_stack([np.ones(len(frame)), before, group, dummies])
        null.append(ols_coefficient(after, x, 2))
    null = np.asarray(null)
    p_value = float((1 + np.count_nonzero(np.abs(null) >= abs(observed) - 1e-12)) / (len(null) + 1))
    return (observed, p_value)

def paired_sign_flip_p(differences: np.ndarray, rng: np.random.Generator) -> float:
    n = len(differences)
    possible = 2 ** n
    observed = abs(float(np.mean(differences)))
    if possible <= N_PERM:
        signs = itertools.product((-1.0, 1.0), repeat=n)
    else:
        signs = (rng.choice((-1.0, 1.0), size=n) for _ in range(N_PERM))
    null = np.asarray([abs(float(np.mean(differences * np.asarray(sign)))) for sign in signs])
    return float((1 + np.count_nonzero(null >= observed - 1e-12)) / (len(null) + 1))

def matched_clinical_analysis(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + (1000 if label == 'matched_all' else 2000))
    rows = []
    for outcome in OUTCOMES:
        paired_differences = []
        for (_, pair) in frame.groupby('pair_id'):
            (_, _, improvement) = oriented_values(pair, outcome)
            group = pair['group_tms'].to_numpy(int)
            paired_differences.append(float(improvement[group == 1][0] - improvement[group == 0][0]))
        paired_differences = np.asarray(paired_differences)
        mean = float(paired_differences.mean())
        sem = float(paired_differences.std(ddof=1) / math.sqrt(len(paired_differences)))
        critical = float(student_t.ppf(0.975, len(paired_differences) - 1))
        (coefficient, permutation_p) = paired_permutation_ancova(frame, outcome, rng)
        (x_before, x_after, _) = oriented_values(frame, outcome)
        x_group = frame['group_tms'].to_numpy(float)
        model = sm.OLS(x_after, pair_fixed_design(frame, x_before, x_group)).fit(cov_type='HC3')
        (boot_low, boot_high) = bootstrap_pair_ancova(frame, outcome, rng)
        rows.append({'match_set': label, 'outcome': outcome, 'n_pairs': int(frame.pair_id.nunique()), 'paired_improvement_difference_mean': mean, 'paired_improvement_ci_low': mean - critical * sem, 'paired_improvement_ci_high': mean + critical * sem, 'paired_improvement_sign_flip_p': paired_sign_flip_p(paired_differences, rng), 'pair_fe_baseline_adjusted_group_coefficient': coefficient, 'pair_fe_boot_ci_low': boot_low, 'pair_fe_boot_ci_high': boot_high, 'pair_fe_hc3_p': float(model.pvalues[2]), 'pair_fe_within_pair_permutation_p': permutation_p})
    results = pd.DataFrame(rows)
    results['pair_fe_permutation_fdr_q'] = multipletests(results['pair_fe_within_pair_permutation_p'], method='fdr_bh')[1]
    return results

def zscore(values: np.ndarray) -> np.ndarray:
    sd = float(np.std(values, ddof=1))
    if sd < 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - float(np.mean(values))) / sd

def pair_dummies(frame: pd.DataFrame) -> np.ndarray:
    return pd.get_dummies(frame['pair_id'].astype(str), drop_first=True, dtype=float).to_numpy()

def mediation_coefficients_matched(frame: pd.DataFrame, outcome: str, mediator: str, baseline_mediator: str) -> dict[str, float]:
    mediator_z = zscore(frame[mediator].to_numpy(float))
    baseline_m_z = zscore(frame[baseline_mediator].to_numpy(float))
    (before, after, _) = oriented_values(frame, outcome)
    scale = float(np.std(before, ddof=1))
    before_z = (before - float(np.mean(before))) / scale
    after_z = (after - float(np.mean(before))) / scale
    group = frame['group_tms'].to_numpy(float)
    dummies = pair_dummies(frame)
    xa = np.column_stack([np.ones(len(frame)), group, baseline_m_z, dummies])
    fit_a = sm.OLS(mediator_z, xa).fit(cov_type='HC3')
    a = float(fit_a.params[1])
    xb = np.column_stack([np.ones(len(frame)), before_z, group, mediator_z, baseline_m_z, dummies])
    fit_b = sm.OLS(after_z, xb).fit(cov_type='HC3')
    b = float(fit_b.params[3])
    direct = float(fit_b.params[2])
    xc = np.column_stack([np.ones(len(frame)), before_z, group, baseline_m_z, dummies])
    fit_c = sm.OLS(after_z, xc).fit(cov_type='HC3')
    total = float(fit_c.params[2])
    return {'path_a': a, 'path_a_hc3_p': float(fit_a.pvalues[1]), 'path_b': b, 'path_b_hc3_p': float(fit_b.pvalues[3]), 'direct_effect': direct, 'total_effect': total, 'indirect_effect': a * b}

def mediation_product_matched_fast(frame: pd.DataFrame, outcome: str, mediator: str, baseline_mediator: str) -> float:
    mediator_z = zscore(frame[mediator].to_numpy(float))
    baseline_m_z = zscore(frame[baseline_mediator].to_numpy(float))
    (before, after, _) = oriented_values(frame, outcome)
    scale = float(np.std(before, ddof=1))
    before_z = (before - float(np.mean(before))) / scale
    after_z = (after - float(np.mean(before))) / scale
    group = frame['group_tms'].to_numpy(float)
    dummies = pair_dummies(frame)
    xa = np.column_stack([np.ones(len(frame)), group, baseline_m_z, dummies])
    xb = np.column_stack([np.ones(len(frame)), before_z, group, mediator_z, baseline_m_z, dummies])
    a = ols_coefficient(mediator_z, xa, 1)
    b = ols_coefficient(after_z, xb, 3)
    return a * b

def bootstrap_matched_mediation(frame: pd.DataFrame, outcome: str, mediator: str, baseline_mediator: str, rng: np.random.Generator) -> tuple[float, float, float]:
    pair_ids = frame['pair_id'].drop_duplicates().to_numpy()
    values = []
    for _ in range(N_BOOT):
        sampled = rng.choice(pair_ids, len(pair_ids), replace=True)
        blocks = []
        for (new_id, old_id) in enumerate(sampled):
            block = frame.loc[frame.pair_id == old_id].copy()
            block['pair_id'] = new_id
            blocks.append(block)
        boot = pd.concat(blocks, ignore_index=True)
        try:
            values.append(mediation_product_matched_fast(boot, outcome, mediator, baseline_mediator))
        except (np.linalg.LinAlgError, ValueError):
            continue
    values = np.asarray(values)
    (low, high) = np.quantile(values, [0.025, 0.975])
    p_value = 2.0 * min((1 + np.count_nonzero(values <= 0)) / (len(values) + 1), (1 + np.count_nonzero(values >= 0)) / (len(values) + 1))
    return (float(low), float(high), float(min(p_value, 1.0)))

def matched_mediation_analysis(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    mediators = {'cortical_g1_reorganization_rms': ('g1_reorganization_rms', 'cortical_G1_sd'), 'subcortical_5d_normalized_reorganization': ('subcortical_5d_normalized_reorganization', 'subcortical_5d_baseline_dispersion')}
    rng = np.random.default_rng(SEED + (3000 if label == 'matched_all' else 4000))
    rows = []
    for outcome in PRIMARY_OUTCOMES:
        for (mediator_label, (mediator, baseline_mediator)) in mediators.items():
            estimates = mediation_coefficients_matched(frame, outcome, mediator, baseline_mediator)
            (low, high, bootstrap_p) = bootstrap_matched_mediation(frame, outcome, mediator, baseline_mediator, rng)
            rows.append({'analysis': label, 'outcome': outcome, 'mediator': mediator_label, 'n': len(frame), 'n_pairs': int(frame.pair_id.nunique()), **estimates, 'indirect_boot_ci_low': low, 'indirect_boot_ci_high': high, 'indirect_bootstrap_p': bootstrap_p})
    results = pd.DataFrame(rows)
    results['indirect_fdr_q'] = multipletests(results['indirect_bootstrap_p'], method='fdr_bh')[1]
    return results

def mediation_coefficients_full(frame: pd.DataFrame, outcome: str, mediator: str, baseline_mediator: str) -> dict[str, float]:
    data = frame.copy()
    mediator_z = zscore(data[mediator].to_numpy(float))
    baseline_m_z = zscore(data[baseline_mediator].to_numpy(float))
    (before, after, _) = oriented_values(data, outcome)
    outcome_scale = float(np.std(before, ddof=1))
    before_z = (before - float(np.mean(before))) / outcome_scale
    after_z = (after - float(np.mean(before))) / outcome_scale
    group = data['group_tms'].to_numpy(float)
    covariates = np.column_stack([zscore(data['age'].to_numpy(float)), data['sex_male'].to_numpy(float), zscore(data['esrs'].to_numpy(float)), zscore(data['log1p_lesion_volume'].to_numpy(float)), zscore(data['mask_lateralization_index'].to_numpy(float)), zscore(data['lesion_topology_pc1'].to_numpy(float)), zscore(data['lesion_topology_pc2'].to_numpy(float))])
    xa = np.column_stack([np.ones(len(data)), group, baseline_m_z, covariates])
    fit_a = sm.OLS(mediator_z, xa).fit(cov_type='HC3')
    a = float(fit_a.params[1])
    xb = np.column_stack([np.ones(len(data)), before_z, group, mediator_z, baseline_m_z, covariates])
    fit_b = sm.OLS(after_z, xb).fit(cov_type='HC3')
    b = float(fit_b.params[3])
    xc = np.column_stack([np.ones(len(data)), before_z, group, baseline_m_z, covariates])
    fit_c = sm.OLS(after_z, xc).fit(cov_type='HC3')
    return {'path_a': a, 'path_a_hc3_p': float(fit_a.pvalues[1]), 'path_b': b, 'path_b_hc3_p': float(fit_b.pvalues[3]), 'direct_effect': float(fit_b.params[2]), 'total_effect': float(fit_c.params[2]), 'indirect_effect': a * b}

def mediation_product_full_fast(frame: pd.DataFrame, outcome: str, mediator: str, baseline_mediator: str) -> float:
    mediator_z = zscore(frame[mediator].to_numpy(float))
    baseline_m_z = zscore(frame[baseline_mediator].to_numpy(float))
    (before, after, _) = oriented_values(frame, outcome)
    outcome_scale = float(np.std(before, ddof=1))
    before_z = (before - float(np.mean(before))) / outcome_scale
    after_z = (after - float(np.mean(before))) / outcome_scale
    group = frame['group_tms'].to_numpy(float)
    covariates = np.column_stack([zscore(frame['age'].to_numpy(float)), frame['sex_male'].to_numpy(float), zscore(frame['esrs'].to_numpy(float)), zscore(frame['log1p_lesion_volume'].to_numpy(float)), zscore(frame['mask_lateralization_index'].to_numpy(float)), zscore(frame['lesion_topology_pc1'].to_numpy(float)), zscore(frame['lesion_topology_pc2'].to_numpy(float))])
    xa = np.column_stack([np.ones(len(frame)), group, baseline_m_z, covariates])
    xb = np.column_stack([np.ones(len(frame)), before_z, group, mediator_z, baseline_m_z, covariates])
    a = ols_coefficient(mediator_z, xa, 1)
    b = ols_coefficient(after_z, xb, 3)
    return a * b

def bootstrap_full_mediation(frame: pd.DataFrame, outcome: str, mediator: str, baseline_mediator: str, rng: np.random.Generator) -> tuple[float, float, float]:
    group_indices = {group: frame.index[frame.group_tms == group].to_numpy() for group in (0, 1)}
    values = []
    for _ in range(N_BOOT):
        indices = np.concatenate([rng.choice(index, len(index), replace=True) for index in group_indices.values()])
        boot = frame.loc[indices].reset_index(drop=True)
        try:
            values.append(mediation_product_full_fast(boot, outcome, mediator, baseline_mediator))
        except (np.linalg.LinAlgError, ValueError):
            continue
    values = np.asarray(values)
    (low, high) = np.quantile(values, [0.025, 0.975])
    p_value = 2.0 * min((1 + np.count_nonzero(values <= 0)) / (len(values) + 1), (1 + np.count_nonzero(values >= 0)) / (len(values) + 1))
    return (float(low), float(high), float(min(p_value, 1.0)))

def full_mediation_analysis(frame: pd.DataFrame) -> pd.DataFrame:
    mediators = {'cortical_g1_reorganization_rms': ('g1_reorganization_rms', 'cortical_G1_sd'), 'subcortical_5d_normalized_reorganization': ('subcortical_5d_normalized_reorganization', 'subcortical_5d_baseline_dispersion')}
    rng = np.random.default_rng(SEED + 5000)
    rows = []
    for outcome in PRIMARY_OUTCOMES:
        for (mediator_label, (mediator, baseline_mediator)) in mediators.items():
            estimates = mediation_coefficients_full(frame, outcome, mediator, baseline_mediator)
            (low, high, bootstrap_p) = bootstrap_full_mediation(frame, outcome, mediator, baseline_mediator, rng)
            rows.append({'analysis': 'full_cohort_topology_adjusted', 'outcome': outcome, 'mediator': mediator_label, 'n': len(frame), 'n_pairs': np.nan, **estimates, 'indirect_boot_ci_low': low, 'indirect_boot_ci_high': high, 'indirect_bootstrap_p': bootstrap_p})
    results = pd.DataFrame(rows)
    results['indirect_fdr_q'] = multipletests(results['indirect_bootstrap_p'], method='fdr_bh')[1]
    return results

def fmt(value: float, digits: int=3) -> str:
    if pd.isna(value):
        return 'NA'
    return f'{value:.{digits}f}'

def write_summary(clinical: pd.DataFrame, exact_results: pd.DataFrame, pairs: pd.DataFrame, balance: pd.DataFrame, clinical_results: pd.DataFrame, mediation: pd.DataFrame) -> None:
    exact_n = int((clinical.location_normalized == '基底节、侧脑室旁').sum())
    lines = ['# 病灶相似患者的临床改善与梯度中介', '', '## 分析设计', '', f'- 主临床表共 {len(clinical)} 例；文字部位‘基底节、侧脑室旁’亚组为 {exact_n} 例。', f'- 定量病灶共同队列生成 {len(pairs)} 对最优匹配；紧匹配阈值为标准化 RMS 距离 ≤ {TIGHT_MATCH_CALIPER}。', '- 匹配先严格限定临床病灶大类和左右侧，再根据体积、侧化、区室比例、质心、空间范围、三个规范纤维系统负荷及低权重人口学变量最小化距离。', '- 主要临床结局为 NIHSS 和 ADL；MMSE、HAMD、VFT 为次要结局。正系数均表示 TMS 组改善更多。', '- 中介仅预设皮层 G1 全局重组 RMS 与皮层下五维归一化重组；同时报告匹配分析与38例全队列拓扑校正敏感性分析。', '', '## 完全相同文字部位亚组', '', '| 结局 | Control改善 | TMS改善 | 差值 | 基线校正系数 | 置换P | FDR q |', '|---|---:|---:|---:|---:|---:|---:|']
    for row in exact_results.itertuples():
        lines.append(f'| {row.outcome} | {fmt(row.control_improvement_mean)} | {fmt(row.tms_improvement_mean)} | {fmt(row.improvement_difference_tms_minus_control)} | {fmt(row.baseline_adjusted_group_coefficient)} | {fmt(row.baseline_adjusted_exact_permutation_p)} | {fmt(row.permutation_fdr_q)} |')
    all_balance = balance.loc[balance.match_set == 'matched_all']
    tight_pairs = pairs.loc[pairs.standardized_rms_distance <= TIGHT_MATCH_CALIPER]
    tight_balance = balance.loc[balance.match_set == 'matched_tight']
    lines.extend(['', '## 定量病灶匹配', '', f'- 全部匹配：{len(pairs)} 对；紧匹配：{len(tight_pairs)} 对。', f'- 匹配前最大绝对SMD为 {all_balance.smd_before.abs().max():.3f}；全部匹配后为 {all_balance.smd_after.abs().max():.3f}。', f'- 紧匹配后最大绝对SMD为 {tight_balance.smd_after.abs().max():.3f}。' if not tight_balance.empty else '- 紧匹配样本不足，未计算平衡性。', '- 匹配后的最大绝对SMD仍高于0.20，说明病灶和人口学残余不平衡未完全消除；匹配结果应作为支持性敏感性分析，而不是无混杂的因果估计。', '', '| 匹配集 | 结局 | 对数 | 改善差值 | 95% CI | 基线校正系数 | Bootstrap CI | 置换P | FDR q |', '|---|---|---:|---:|---|---:|---|---:|---:|'])
    for row in clinical_results.itertuples():
        lines.append(f'| {row.match_set} | {row.outcome} | {row.n_pairs} | {fmt(row.paired_improvement_difference_mean)} | {fmt(row.paired_improvement_ci_low)} to {fmt(row.paired_improvement_ci_high)} | {fmt(row.pair_fe_baseline_adjusted_group_coefficient)} | {fmt(row.pair_fe_boot_ci_low)} to {fmt(row.pair_fe_boot_ci_high)} | {fmt(row.pair_fe_within_pair_permutation_p)} | {fmt(row.pair_fe_permutation_fdr_q)} |')
    lines.extend(['', '## 探索性梯度中介', '', '间接效应以基线临床评分标准差为单位；Bootstrap区间跨0表示没有稳定中介证据。', '', '| 分析 | 结局 | 中介 | a | b | 间接效应 | 95% CI | P | FDR q |', '|---|---|---|---:|---:|---:|---|---:|---:|'])
    for row in mediation.itertuples():
        lines.append(f'| {row.analysis} | {row.outcome} | {row.mediator} | {fmt(row.path_a)} | {fmt(row.path_b)} | {fmt(row.indirect_effect)} | {fmt(row.indirect_boot_ci_low)} to {fmt(row.indirect_boot_ci_high)} | {fmt(row.indirect_bootstrap_p)} | {fmt(row.indirect_fdr_q)} |')
    lines.extend(['', '## 解释边界', '', '- 不能用‘TMS组组内显著而Control组不显著’推断组间治疗差异；本分析直接检验两组改善差和基线校正组系数。', '- 当前是否随机、是否假刺激以及同期康复是否一致仍不明确，因此组系数是病灶可比条件下的调整后组间关联，不自动等于因果TMS效应。', '- 梯度和临床结局在同一前后时间窗测量，间接效应只能称为探索性统计中介，不能单独证明机制。', '- 文字病灶部位粗糙且存在大量单例；定量mask匹配是主证据，完全相同文字亚组仅用于透明敏感性分析。', '- 定量匹配仍存在皮层受累比例、病灶范围、年龄或ESRS的残余不平衡；需要随机化和完整治疗方案信息才能把组差异升级为治疗效应。'])
    (OUT / 'RESULTS_ZH.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

def main() -> None:
    ensure_dirs()
    clinical = load_clinical()
    (exact_distribution, coarse_distribution) = location_distributions(clinical)
    exact_distribution.to_csv(TABLES / 'location_exact_distribution.csv', index=False, encoding='utf-8-sig')
    coarse_distribution.to_csv(TABLES / 'location_coarse_distribution.csv', index=False, encoding='utf-8-sig')
    (exact_subjects, exact_results) = exact_location_subgroup_analysis(clinical)
    exact_subjects.to_csv(TABLES / 'exact_basal_ganglia_periventricular_subjects.csv', index=False, encoding='utf-8-sig')
    exact_results.to_csv(TABLES / 'exact_basal_ganglia_periventricular_clinical_results.csv', index=False, encoding='utf-8-sig')
    (joint, match_features, match_weights) = load_joint_cohort(clinical)
    pairs = optimal_pairs(joint, match_features, match_weights)
    pairs.to_csv(TABLES / 'optimal_lesion_matched_pairs.csv', index=False, encoding='utf-8-sig')
    all_matched = matched_subject_frame(joint, pairs)
    tight_pairs = pairs.loc[pairs.standardized_rms_distance <= TIGHT_MATCH_CALIPER].copy()
    tight_matched = matched_subject_frame(joint, tight_pairs) if len(tight_pairs) else pd.DataFrame()
    balance_frames = [balance_table(joint, all_matched, match_features, 'matched_all')]
    clinical_frames = [matched_clinical_analysis(all_matched, 'matched_all')]
    mediation_frames = [matched_mediation_analysis(all_matched, 'matched_all')]
    if len(tight_pairs) >= 4:
        balance_frames.append(balance_table(joint, tight_matched, match_features, 'matched_tight'))
        clinical_frames.append(matched_clinical_analysis(tight_matched, 'matched_tight'))
        mediation_frames.append(matched_mediation_analysis(tight_matched, 'matched_tight'))
    balance = pd.concat(balance_frames, ignore_index=True)
    clinical_results = pd.concat(clinical_frames, ignore_index=True)
    mediation_frames.append(full_mediation_analysis(joint))
    mediation = pd.concat(mediation_frames, ignore_index=True)
    balance.to_csv(TABLES / 'matching_balance.csv', index=False, encoding='utf-8-sig')
    clinical_results.to_csv(TABLES / 'matched_clinical_results.csv', index=False, encoding='utf-8-sig')
    mediation.to_csv(TABLES / 'gradient_mediation_results.csv', index=False, encoding='utf-8-sig')
    joint.to_csv(TABLES / 'analysis_cohort_38.csv', index=False, encoding='utf-8-sig')
    manifest = {'analysis_date': '2026-08-03', 'clinical_n': int(len(clinical)), 'quantitative_lesion_gradient_n': int(len(joint)), 'exact_location_subgroup': '基底节、侧脑室旁', 'exact_location_subgroup_n': int((clinical.location_normalized == '基底节、侧脑室旁').sum()), 'matching': {'exact_strata': ['coarse clinical lesion location', 'clinical lesion side'], 'distance': 'weighted standardized RMS with optimal linear assignment', 'features': match_features, 'feature_weights': match_weights, 'all_pairs_n': int(len(pairs)), 'tight_caliper': TIGHT_MATCH_CALIPER, 'tight_pairs_n': int(len(tight_pairs))}, 'clinical_outcomes': list(OUTCOMES), 'primary_outcomes': list(PRIMARY_OUTCOMES), 'mediators': ['cortical G1 reorganization RMS', 'subcortical 5D normalized reorganization'], 'bootstrap_resamples': N_BOOT, 'paired_permutation_max': N_PERM, 'interpretation': 'adjusted observational group comparison and exploratory statistical mediation', 'random_seed': SEED}
    (OUT / 'analysis_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    write_summary(clinical, exact_results, pairs, balance, clinical_results, mediation)
    print(f'Clinical cohort: {len(clinical)}')
    print(f'Quantitative lesion-gradient cohort: {len(joint)}')
    print(f"Exact location subgroup: {(clinical.location_normalized == '基底节、侧脑室旁').sum()}")
    print(f'Matched pairs: {len(pairs)}; tight pairs: {len(tight_pairs)}')
    print(f'Output: {OUT}')
if __name__ == '__main__':
    main()
