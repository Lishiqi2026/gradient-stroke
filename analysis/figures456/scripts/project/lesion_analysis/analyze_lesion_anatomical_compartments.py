from __future__ import annotations
import os as _portable_os
from pathlib import Path as _PortablePath

def _portable_path(kind, relative):
    project = _PortablePath(_portable_os.environ['PAPER_PROJECT_ROOT'])
    root = project if kind == 'project' else project.parent
    return str(root / relative)
import json
import re
from pathlib import Path
from xml.etree import ElementTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from scipy.stats import fisher_exact, mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
ROOT = Path(_portable_path('project', ''))
ANALYSIS = ROOT / 'lesion_analysis'
REG = ANALYSIS / 'mni_registration'
RESOURCES = ANALYSIS / 'resources'
RESULTS = ANALYSIS / 'results_abcd'
TABLES = RESULTS / 'tables'
FIGURES = RESULTS / 'figures'
NIFTI = RESULTS / 'nifti'
SUBCORTICAL_ROOT = ROOT / '皮层下' / '(54,5)'
GROUPS = ('Control', 'TMS')
GROUP_COLORS = {'Control': '#4C78A8', 'TMS': '#D65F45'}
COMPARTMENTS = ('brainstem', 'deep_subcortical_gray', 'cerebral_white_matter', 'cerebral_cortex', 'cerebellum', 'atlas_unlabeled')
COMPARTMENT_LABELS = {'brainstem': 'Brainstem', 'deep_subcortical_gray': 'Deep/subcortical gray', 'cerebral_white_matter': 'Cerebral white matter', 'cerebral_cortex': 'Cerebral cortex', 'cerebellum': 'Cerebellum', 'atlas_unlabeled': 'Atlas-unlabeled'}
COMPARTMENT_COLORS = {'brainstem': '#457B9D', 'deep_subcortical_gray': '#6A4C93', 'cerebral_white_matter': '#2A9D8F', 'cerebral_cortex': '#E76F51', 'cerebellum': '#E9C46A', 'atlas_unlabeled': '#B8B8B8'}
OUTCOMES = {'NIHSS': ('NIHSS_pre', 'NIHSS_post', -1.0), 'ADL': ('ADL_pre', 'ADL_post', 1.0), 'MMSE': ('MMSE_pre', 'MMSE_post', 1.0), 'HAMD': ('HAMD_pre', 'HAMD_post', -1.0), 'VFT': ('VFT_pre', 'VFT_post', 1.0)}
HO_VALUES = {'cerebral_white_matter': {1, 12}, 'cerebral_cortex': {2, 13}, 'brainstem': {8}, 'deep_subcortical_gray': {4, 5, 6, 7, 9, 10, 11, 15, 16, 17, 18, 19, 20, 21}}
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans', 'sans-serif'], 'font.size': 7, 'axes.titlesize': 8, 'axes.labelsize': 7, 'axes.linewidth': 0.7, 'axes.spines.right': False, 'axes.spines.top': False, 'legend.frameon': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})

def ensure_output_dirs() -> None:
    for path in (TABLES, FIGURES, NIFTI):
        path.mkdir(parents=True, exist_ok=True)

def save_figure(fig: plt.Figure, stem: str) -> None:
    for (suffix, kwargs) in (('png', {'dpi': 300}), ('svg', {}), ('pdf', {}), ('tiff', {'dpi': 600})):
        fig.savefig(FIGURES / f'{stem}.{suffix}', bbox_inches='tight', facecolor='white', **kwargs)

def zscore(values: pd.Series | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    sd = float(np.nanstd(array, ddof=1))
    if not np.isfinite(sd) or sd == 0:
        return np.zeros_like(array)
    return (array - float(np.nanmean(array))) / sd

def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta for x minus y."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float((np.sum(x[:, None] > y[None, :]) - np.sum(x[:, None] < y[None, :])) / (len(x) * len(y)))

def fit_hc3(y: np.ndarray, predictors: pd.DataFrame):
    design = sm.add_constant(predictors.astype(float), has_constant='add')
    return sm.OLS(np.asarray(y, dtype=float), design, missing='raise').fit(cov_type='HC3')

def residualize(values: np.ndarray, covariates: pd.DataFrame) -> np.ndarray:
    design = sm.add_constant(covariates.astype(float), has_constant='add')
    return np.asarray(sm.OLS(np.asarray(values, dtype=float), design, missing='raise').fit().resid)

def load_reference_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    lesion = pd.read_csv(TABLES / 'B_participant_lesion_metrics.csv', dtype={'subject_id': str})
    clinical = pd.read_csv(TABLES / 'D_gradient_clinical_subject_metrics.csv', dtype={'subject_id': str})
    if len(lesion) != 38 or len(clinical) != 38:
        raise RuntimeError(f'Expected 38 subjects, found lesion={len(lesion)}, clinical={len(clinical)}')
    if set(lesion['subject_id']) != set(clinical['subject_id']):
        raise RuntimeError('Lesion and clinical subject sets differ')
    return (lesion, clinical)

def load_atlases() -> tuple[nib.Nifti1Image, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    reference = nib.load(str(RESOURCES / 'MNI152_T1_2mm.nii.gz'))
    paths = (RESOURCES / 'HarvardOxford-sub-maxprob-thr25-2mm.nii.gz', RESOURCES / 'Cerebellum-MNIfnirt-maxprob-thr25-2mm.nii.gz', RESOURCES / 'JHU-ICBM-labels-2mm.nii.gz', RESOURCES / 'MNI152_T1_2mm_brain_mask.nii.gz')
    arrays = []
    for path in paths:
        image = nib.load(str(path))
        if image.shape != reference.shape or not np.allclose(image.affine, reference.affine):
            raise RuntimeError(f'Atlas geometry mismatch: {path.name}')
        arrays.append(np.asarray(image.dataobj, dtype=np.int16))
    return (reference, arrays[0], arrays[1], arrays[2], arrays[3] > 0)

def build_compartment_map(ho_sub: np.ndarray, cerebellum: np.ndarray, brain_mask: np.ndarray) -> tuple[np.ndarray, dict]:
    label_map = np.zeros(ho_sub.shape, dtype=np.uint8)
    masks = {'brainstem': np.isin(ho_sub, list(HO_VALUES['brainstem'])), 'deep_subcortical_gray': np.isin(ho_sub, list(HO_VALUES['deep_subcortical_gray'])), 'cerebral_white_matter': np.isin(ho_sub, list(HO_VALUES['cerebral_white_matter'])), 'cerebral_cortex': np.isin(ho_sub, list(HO_VALUES['cerebral_cortex'])), 'cerebellum': cerebellum > 0}
    overlaps_before_priority = {}
    occupied = np.zeros(ho_sub.shape, dtype=bool)
    for (value, compartment) in enumerate(COMPARTMENTS[:-1], start=1):
        candidate = masks[compartment]
        overlaps_before_priority[compartment] = int(np.count_nonzero(candidate & occupied))
        accepted = candidate & ~occupied
        label_map[accepted] = value
        occupied |= accepted
    atlas_unlabeled_in_brain = brain_mask & ~occupied
    metadata = {'label_values': {str(index): name for (index, name) in enumerate(COMPARTMENTS[:-1], start=1)}, 'priority_order': list(COMPARTMENTS[:-1]), 'overlap_voxels_removed_by_priority': overlaps_before_priority, 'atlas_unlabeled_in_brain_voxels': int(atlas_unlabeled_in_brain.sum()), 'note': 'Atlas-unlabeled lesion burden is calculated per lesion as every voxel not assigned to labels 1-5; it can include voxels outside the MNI brain mask, which are also reported separately as QC.'}
    return (label_map, metadata)

def parse_xml_labels(path: Path, offset: int) -> dict[int, str]:
    root = ElementTree.parse(path).getroot()
    return {int(label.attrib['index']) + offset: (label.text or '').strip() for label in root.findall('./data/label')}

def baseline_mask_path(group: str, subject_id: str) -> Path:
    path = REG / 'masks' / group / 'before' / f'{subject_id}_lesion_mni2mm.nii.gz'
    if not path.exists():
        raise FileNotFoundError(path)
    return path

def extract_anatomical_metrics(lesion_reference: pd.DataFrame, reference: nib.Nifti1Image, compartment_map: np.ndarray, jhu: np.ndarray, brain_mask: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    jhu_labels = parse_xml_labels(RESOURCES / 'JHU-labels.xml', offset=0)
    voxel_ml = abs(float(np.linalg.det(reference.affine[:3, :3]))) / 1000.0
    participant_rows = []
    tract_rows = []
    tract_sizes = np.bincount(jhu.ravel(), minlength=max(jhu_labels) + 1)
    for row in lesion_reference.itertuples():
        image = nib.load(str(baseline_mask_path(row.group, row.subject_id)))
        if image.shape != reference.shape or not np.allclose(image.affine, reference.affine):
            raise RuntimeError(f'Lesion geometry mismatch: {row.subject_id}')
        lesion = np.asarray(image.dataobj) > 0.5
        lesion_voxels = int(lesion.sum())
        if lesion_voxels == 0:
            raise RuntimeError(f'Empty baseline lesion: {row.subject_id}')
        values: dict[str, object] = {'subject_id': row.subject_id, 'group': row.group, 'lesion_voxels': lesion_voxels, 'baseline_lesion_volume_ml': lesion_voxels * voxel_ml, 'outside_mni_brain_voxels': int(np.count_nonzero(lesion & ~brain_mask)), 'outside_mni_brain_fraction_pct': float(np.count_nonzero(lesion & ~brain_mask) * 100.0 / lesion_voxels)}
        assigned = 0
        for (label_value, compartment) in enumerate(COMPARTMENTS[:-1], start=1):
            count = int(np.count_nonzero(lesion & (compartment_map == label_value)))
            assigned += count
            values[f'{compartment}_voxels'] = count
            values[f'{compartment}_volume_ml'] = count * voxel_ml
            values[f'{compartment}_fraction_pct'] = count * 100.0 / lesion_voxels
            values[f'{compartment}_involved'] = int(count > 0)
        unlabeled = lesion_voxels - assigned
        values['atlas_unlabeled_voxels'] = unlabeled
        values['atlas_unlabeled_volume_ml'] = unlabeled * voxel_ml
        values['atlas_unlabeled_fraction_pct'] = unlabeled * 100.0 / lesion_voxels
        values['atlas_unlabeled_involved'] = int(unlabeled > 0)
        values['compartment_fraction_sum_pct'] = sum((float(values[f'{name}_fraction_pct']) for name in COMPARTMENTS))
        participant_rows.append(values)
        counts = np.bincount(jhu[lesion], minlength=max(jhu_labels) + 1)
        for (tract_value, tract_name) in sorted(jhu_labels.items()):
            if tract_value == 0:
                continue
            count = int(counts[tract_value])
            tract_rows.append({'subject_id': row.subject_id, 'group': row.group, 'tract_value': tract_value, 'tract_name': tract_name, 'damaged_voxels': count, 'damaged_volume_ml': count * voxel_ml, 'lesion_fraction_pct': count * 100.0 / lesion_voxels, 'tract_damage_pct': count * 100.0 / int(tract_sizes[tract_value]), 'tract_involved': int(count > 0)})
    participant = pd.DataFrame(participant_rows)
    tracts = pd.DataFrame(tract_rows)
    if not np.allclose(participant['compartment_fraction_sum_pct'], 100.0):
        raise RuntimeError('Anatomical compartment fractions do not sum to 100%')
    if not np.allclose(participant.set_index('subject_id')['baseline_lesion_volume_ml'].sort_index(), lesion_reference.set_index('subject_id')['baseline_lesion_volume_ml'].sort_index()):
        raise RuntimeError('Recomputed lesion volumes differ from the frozen A/B/C/D table')
    return (participant, tracts)

def compartment_prevalence(participant: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for compartment in COMPARTMENTS:
        for group in ('All', *GROUPS):
            subset = participant if group == 'All' else participant.loc[participant['group'] == group]
            involved = subset[f'{compartment}_involved'].astype(int)
            fractions = subset[f'{compartment}_fraction_pct'].astype(float)
            rows.append({'compartment': compartment, 'compartment_label': COMPARTMENT_LABELS[compartment], 'group': group, 'n': len(subset), 'involved_n': int(involved.sum()), 'prevalence_pct': float(involved.mean() * 100), 'median_lesion_fraction_pct': float(fractions.median()), 'iqr_low': float(fractions.quantile(0.25)), 'iqr_high': float(fractions.quantile(0.75))})
    return pd.DataFrame(rows)

def compartment_group_comparisons(participant: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for compartment in COMPARTMENTS:
        control = participant.loc[participant['group'] == 'Control', f'{compartment}_fraction_pct'].to_numpy(float)
        tms = participant.loc[participant['group'] == 'TMS', f'{compartment}_fraction_pct'].to_numpy(float)
        control_present = int(np.count_nonzero(control > 0))
        tms_present = int(np.count_nonzero(tms > 0))
        presence_table = [[tms_present, len(tms) - tms_present], [control_present, len(control) - control_present]]
        (_, presence_p) = fisher_exact(presence_table, alternative='two-sided')
        rows.append({'compartment': compartment, 'compartment_label': COMPARTMENT_LABELS[compartment], 'control_n': len(control), 'tms_n': len(tms), 'control_involved_n': control_present, 'tms_involved_n': tms_present, 'control_median_fraction_pct': float(np.median(control)), 'tms_median_fraction_pct': float(np.median(tms)), 'cliffs_delta_tms_minus_control': cliffs_delta(tms, control), 'mannwhitney_p': float(mannwhitneyu(tms, control, alternative='two-sided').pvalue), 'fisher_presence_p': float(presence_p)})
    table = pd.DataFrame(rows)
    table['mannwhitney_fdr_q'] = multipletests(table['mannwhitney_p'], method='fdr_bh')[1]
    table['fisher_presence_fdr_q'] = multipletests(table['fisher_presence_p'], method='fdr_bh')[1]
    return table

def summarize_tracts(tracts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (tract_name, tract) in tracts.groupby('tract_name', sort=True):
        for group in ('All', *GROUPS):
            subset = tract if group == 'All' else tract.loc[tract['group'] == group]
            rows.append({'tract_name': tract_name, 'group': group, 'n': len(subset), 'involved_n': int(subset['tract_involved'].sum()), 'prevalence_pct': float(subset['tract_involved'].mean() * 100), 'median_lesion_fraction_pct': float(subset['lesion_fraction_pct'].median()), 'median_tract_damage_pct': float(subset['tract_damage_pct'].median()), 'maximum_tract_damage_pct': float(subset['tract_damage_pct'].max())})
    return pd.DataFrame(rows)

def plot_anatomical_burden(participant: pd.DataFrame, tract_summary: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(7.2, 5.5))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.05, 1), width_ratios=(1.45, 1.1), hspace=0.48, wspace=0.4)
    ax_stack = fig.add_subplot(grid[0, :])
    ax_dist = fig.add_subplot(grid[1, 0])
    ax_tract = fig.add_subplot(grid[1, 1])
    ordered_frames = []
    for group in GROUPS:
        subset = participant.loc[participant['group'] == group].copy()
        subset = subset.sort_values(['brainstem_fraction_pct', 'cerebral_white_matter_fraction_pct', 'deep_subcortical_gray_fraction_pct'], ascending=False)
        ordered_frames.append(subset)
    ordered = pd.concat(ordered_frames, ignore_index=True)
    x = np.arange(len(ordered))
    bottom = np.zeros(len(ordered))
    for compartment in COMPARTMENTS:
        values = ordered[f'{compartment}_fraction_pct'].to_numpy(float)
        ax_stack.bar(x, values, bottom=bottom, width=0.86, color=COMPARTMENT_COLORS[compartment], label=COMPARTMENT_LABELS[compartment])
        bottom += values
    boundary = int(np.count_nonzero(ordered['group'] == 'Control')) - 0.5
    ax_stack.axvline(boundary, color='#333333', linewidth=0.7)
    ax_stack.text((boundary - 0.5) / 2, 104, 'Control', ha='center', va='bottom', fontweight='bold')
    ax_stack.text((boundary + len(ordered) - 0.5) / 2, 104, 'TMS', ha='center', va='bottom', fontweight='bold')
    ax_stack.set_ylim(0, 110)
    ax_stack.set_xlim(-0.6, len(ordered) - 0.4)
    ax_stack.set_ylabel('Fraction of lesion (%)')
    ax_stack.set_xlabel('Participants (ordered within group)')
    ax_stack.set_xticks([])
    ax_stack.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.16), fontsize=6.2, handlelength=1.1, columnspacing=1.4)
    ax_stack.text(-0.06, 1.08, 'a', transform=ax_stack.transAxes, fontsize=10, fontweight='bold', va='top')
    ax_stack.set_title('Individual anatomical composition of baseline lesions', pad=14)
    positions = np.arange(len(COMPARTMENTS))
    rng = np.random.default_rng(20260803)
    offsets = {'Control': -0.16, 'TMS': 0.16}
    for group in GROUPS:
        subset = participant.loc[participant['group'] == group]
        for (index, compartment) in enumerate(COMPARTMENTS):
            values = subset[f'{compartment}_fraction_pct'].to_numpy(float)
            jitter = rng.uniform(-0.055, 0.055, len(values))
            ax_dist.scatter(np.full(len(values), index + offsets[group]) + jitter, values, s=12, alpha=0.65, color=GROUP_COLORS[group], edgecolor='white', linewidth=0.25, label=group if index == 0 else None)
            median = float(np.median(values))
            (q1, q3) = np.quantile(values, [0.25, 0.75])
            ax_dist.plot([index + offsets[group]] * 2, [q1, q3], color=GROUP_COLORS[group], linewidth=2.3, solid_capstyle='round')
            ax_dist.plot([index + offsets[group] - 0.07, index + offsets[group] + 0.07], [median, median], color='#222222', linewidth=0.75)
    ax_dist.set_xticks(positions, ['Brainstem', 'Deep gray', 'White matter', 'Cortex', 'Cerebellum', 'Unlabeled'], rotation=28, ha='right')
    ax_dist.set_ylabel('Fraction of lesion (%)')
    ax_dist.set_title('Compartment burden by treatment group')
    ax_dist.text(-0.1, 1.08, 'b', transform=ax_dist.transAxes, fontsize=10, fontweight='bold', va='top')
    cohort = tract_summary.loc[tract_summary['group'] == 'All'].nlargest(8, ['involved_n', 'maximum_tract_damage_pct'])
    names = cohort['tract_name'].tolist()[::-1]
    short_names = {'Pontine crossing tract (a part of MCP)': 'Pontine crossing tract', 'Middle cerebellar peduncle': 'Middle cerebellar peduncle', 'Superior corona radiata R': 'Superior corona radiata (R)', 'Posterior limb of internal capsule R': 'Posterior limb, internal capsule (R)', 'Anterior limb of internal capsule R': 'Anterior limb, internal capsule (R)', 'External capsule R': 'External capsule (R)', 'Corticospinal tract R': 'Corticospinal tract (R)', 'Superior fronto-occipital fasciculus (could be a part of anterior internal capsule) R': 'SFO fasciculus (R)'}
    y = np.arange(len(names))
    for group in GROUPS:
        lookup = tract_summary.loc[(tract_summary['group'] == group) & tract_summary['tract_name'].isin(names)].set_index('tract_name')
        values = np.array([lookup.loc[name, 'prevalence_pct'] for name in names], dtype=float)
        ax_tract.scatter(values, y + offsets[group] * 0.7, s=20, color=GROUP_COLORS[group], label=group, zorder=3)
    ax_tract.set_yticks(y, [short_names.get(name, name) for name in names])
    ax_tract.set_xlim(0, 100)
    ax_tract.set_xlabel('Participants with tract overlap (%)')
    ax_tract.set_title('Most frequently involved JHU tracts')
    ax_tract.grid(axis='x', color='#E5E5E5', linewidth=0.5)
    ax_tract.legend(loc='lower right', fontsize=6.2)
    ax_tract.text(-0.13, 1.08, 'c', transform=ax_tract.transAxes, fontsize=10, fontweight='bold', va='top')
    fig.suptitle('Lesions preferentially involve deep and white-matter anatomy', fontsize=10, fontweight='bold', y=0.995)
    save_figure(fig, 'B2_anatomical_burden_distribution')
    plt.close(fig)

def gradient_path(group: str, session: str, subject_id: str) -> Path:
    matches = sorted((SUBCORTICAL_ROOT / group / session).glob(f'{subject_id}-*.npy'))
    if len(matches) != 1:
        raise RuntimeError(f'Expected one subcortical gradient file for {group}/{session}/{subject_id}, found {len(matches)}')
    return matches[0]

def subcortical_gradient_metrics(participant: pd.DataFrame, clinical: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in participant.itertuples():
        before = np.load(gradient_path(row.group, 'before', row.subject_id)).astype(float)
        after = np.load(gradient_path(row.group, 'after', row.subject_id)).astype(float)
        if before.shape != (54, 5) or after.shape != (54, 5):
            raise RuntimeError(f'Unexpected subcortical gradient shape for {row.subject_id}: {before.shape}, {after.shape}')
        if not np.all(np.isfinite(before)) or not np.all(np.isfinite(after)):
            raise RuntimeError(f'Non-finite subcortical gradients for {row.subject_id}')
        before_centered = before - before.mean(axis=0, keepdims=True)
        after_centered = after - after.mean(axis=0, keepdims=True)
        (rotation, _) = orthogonal_procrustes(after_centered, before_centered)
        after_aligned = after_centered @ rotation
        residual = after_aligned - before_centered
        baseline_dispersion = float(np.sqrt(np.mean(before_centered ** 2)))
        aligned_rms = float(np.sqrt(np.mean(residual ** 2)))
        rows.append({'subject_id': row.subject_id, 'group': row.group, 'subcortical_5d_baseline_dispersion': baseline_dispersion, 'subcortical_5d_reorganization_rms': aligned_rms, 'subcortical_5d_normalized_reorganization': aligned_rms / baseline_dispersion, 'subcortical_g1_aligned_reorganization_rms': float(np.sqrt(np.mean(residual[:, 0] ** 2))), 'subcortical_g2_aligned_reorganization_rms': float(np.sqrt(np.mean(residual[:, 1] ** 2))), 'subcortical_alignment_fit_correlation': float(np.corrcoef(before_centered.ravel(), after_aligned.ravel())[0, 1]), 'subcortical_g1_raw_before_after_correlation': float(np.corrcoef(before_centered[:, 0], after_centered[:, 0])[0, 1])})
    metrics = pd.DataFrame(rows)
    merge_columns = ['subject_id', 'group_tms', 'age_z', 'sex_male', 'mask_lateralization_index', 'nihss_baseline_z', 'clinical_composite_baseline', 'clinical_composite_post']
    for (pre_column, post_column, _) in OUTCOMES.values():
        merge_columns.extend((pre_column, post_column))
    metrics = metrics.merge(clinical[merge_columns], on='subject_id', validate='one_to_one')
    lesion_columns = ['subject_id', 'baseline_lesion_volume_ml', 'outside_mni_brain_fraction_pct']
    lesion_columns.extend((f'{compartment}_fraction_pct' for compartment in COMPARTMENTS))
    lesion_columns.extend((f'{compartment}_involved' for compartment in COMPARTMENTS))
    metrics = metrics.merge(participant[lesion_columns], on='subject_id', validate='one_to_one')
    metrics['log1p_lesion_volume'] = np.log1p(metrics['baseline_lesion_volume_ml'])
    metrics['lesion_burden_z'] = zscore(metrics['log1p_lesion_volume'])
    metrics['subcortical_reorganization_z'] = zscore(metrics['subcortical_5d_normalized_reorganization'])
    return metrics

def variance_inflation_for_term(predictors: pd.DataFrame, term: str) -> float:
    design = sm.add_constant(predictors.astype(float), has_constant='add')
    index = list(design.columns).index(term)
    return float(variance_inflation_factor(design.to_numpy(float), index))

def subcortical_models(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]]]:
    outcome = metrics['subcortical_reorganization_z'].to_numpy(float)
    core_covariates = pd.DataFrame({'group_tms': metrics['group_tms'], 'age_z': metrics['age_z'], 'sex_male': metrics['sex_male'], 'mask_lateralization_index': metrics['mask_lateralization_index'], 'nihss_baseline_z': metrics['nihss_baseline_z']})
    global_predictors = core_covariates.copy()
    global_predictors.insert(0, 'lesion_burden_z', metrics['lesion_burden_z'])
    global_result = fit_hc3(outcome, global_predictors)
    clinical_predictors = pd.DataFrame({'clinical_composite_baseline': metrics['clinical_composite_baseline'], 'group_tms': metrics['group_tms'], 'subcortical_reorganization_z': metrics['subcortical_reorganization_z'], 'lesion_burden_z': metrics['lesion_burden_z'], 'age_z': metrics['age_z'], 'sex_male': metrics['sex_male'], 'mask_lateralization_index': metrics['mask_lateralization_index']})
    clinical_result = fit_hc3(metrics['clinical_composite_post'].to_numpy(float), clinical_predictors)
    global_rows = []
    for (outcome_name, result, term) in (('subcortical_5d_normalized_reorganization', global_result, 'lesion_burden_z'), ('clinical_composite_post', clinical_result, 'subcortical_reorganization_z')):
        global_rows.append({'outcome': outcome_name, 'n': int(result.nobs), 'term': term, 'standardized_beta': float(result.params[term]), 'ci_low': float(result.conf_int().loc[term, 0]), 'ci_high': float(result.conf_int().loc[term, 1]), 'robust_se_hc3': float(result.bse[term]), 'p_value': float(result.pvalues[term]), 'r_squared': float(result.rsquared)})
    global_table = pd.DataFrame(global_rows)
    global_table['fdr_q'] = multipletests(global_table['p_value'], method='fdr_bh')[1]
    compartment_rows = []
    for compartment in COMPARTMENTS[:-1]:
        involved_n = int(metrics[f'{compartment}_involved'].sum())
        predictor_name = f'{compartment}_fraction_z'
        if involved_n < 5:
            compartment_rows.append({'compartment': compartment, 'compartment_label': COMPARTMENT_LABELS[compartment], 'n': len(metrics), 'involved_n': involved_n, 'term': predictor_name, 'estimable': False, 'standardized_beta': np.nan, 'ci_low': np.nan, 'ci_high': np.nan, 'robust_se_hc3': np.nan, 'p_value': np.nan, 'r_squared': np.nan, 'vif': np.nan, 'estimability_note': 'fewer than five participants had direct overlap'})
            continue
        fraction_z = zscore(np.log1p(metrics[f'{compartment}_fraction_pct']))
        predictors = core_covariates.copy()
        predictors.insert(0, 'lesion_burden_z', metrics['lesion_burden_z'])
        predictors.insert(1, predictor_name, fraction_z)
        result = fit_hc3(outcome, predictors)
        vif = variance_inflation_for_term(predictors, predictor_name)
        compartment_rows.append({'compartment': compartment, 'compartment_label': COMPARTMENT_LABELS[compartment], 'n': int(result.nobs), 'involved_n': involved_n, 'term': predictor_name, 'estimable': True, 'standardized_beta': float(result.params[predictor_name]), 'ci_low': float(result.conf_int().loc[predictor_name, 0]), 'ci_high': float(result.conf_int().loc[predictor_name, 1]), 'robust_se_hc3': float(result.bse[predictor_name]), 'p_value': float(result.pvalues[predictor_name]), 'r_squared': float(result.rsquared), 'vif': vif, 'estimability_note': 'exploratory one-compartment-at-a-time model adjusted for total lesion volume'})
    compartment_table = pd.DataFrame(compartment_rows)
    compartment_table['fdr_q'] = np.nan
    estimable = compartment_table['estimable'] & compartment_table['p_value'].notna()
    if estimable.any():
        compartment_table.loc[estimable, 'fdr_q'] = multipletests(compartment_table.loc[estimable, 'p_value'], method='fdr_bh')[1]
    partial = {'lesion_to_subcortical': (residualize(metrics['lesion_burden_z'].to_numpy(float), core_covariates), residualize(outcome, core_covariates)), 'subcortical_to_clinical': (residualize(metrics['subcortical_reorganization_z'].to_numpy(float), clinical_predictors.drop(columns='subcortical_reorganization_z')), residualize(metrics['clinical_composite_post'].to_numpy(float), clinical_predictors.drop(columns='subcortical_reorganization_z')))}
    return (global_table, compartment_table, partial)

def subcortical_sensitivity_models(metrics: pd.DataFrame) -> pd.DataFrame:
    manual_review_flags = {'PRIVATE_SUBJECT_040', 'PRIVATE_SUBJECT_006', 'PRIVATE_SUBJECT_038'}
    specifications = (('primary_full', np.ones(len(metrics), dtype=bool), 'subcortical_5d_normalized_reorganization', False), ('exclude_large_lesion_PRIVATE_SUBJECT_040', metrics['subject_id'].ne('PRIVATE_SUBJECT_040').to_numpy(), 'subcortical_5d_normalized_reorganization', False), ('exclude_all_manual_review_flags', ~metrics['subject_id'].isin(manual_review_flags).to_numpy(), 'subcortical_5d_normalized_reorganization', False), ('exclude_outside_brain_over_1pct', metrics['outside_mni_brain_fraction_pct'].le(1).to_numpy(), 'subcortical_5d_normalized_reorganization', False), ('raw_rms_adjusted_for_baseline_dispersion', np.ones(len(metrics), dtype=bool), 'subcortical_5d_reorganization_rms', True))
    rows = []
    for (analysis, keep, outcome_column, adjust_dispersion) in specifications:
        subset = metrics.loc[keep].copy()
        predictors = pd.DataFrame({'lesion_burden_z': zscore(subset['log1p_lesion_volume']), 'group_tms': subset['group_tms'].to_numpy(float), 'age_z': zscore(subset['age_z']), 'sex_male': subset['sex_male'].to_numpy(float), 'mask_lateralization_index': subset['mask_lateralization_index'].to_numpy(float), 'nihss_baseline_z': zscore(subset['nihss_baseline_z'])})
        if adjust_dispersion:
            predictors['baseline_gradient_dispersion_z'] = zscore(subset['subcortical_5d_baseline_dispersion'])
        result = fit_hc3(zscore(subset[outcome_column]), predictors)
        term = 'lesion_burden_z'
        rows.append({'analysis': analysis, 'outcome': outcome_column, 'n': int(result.nobs), 'excluded_subjects': ';'.join(sorted(set(metrics['subject_id']) - set(subset['subject_id']))), 'term': term, 'standardized_beta': float(result.params[term]), 'ci_low': float(result.conf_int().loc[term, 0]), 'ci_high': float(result.conf_int().loc[term, 1]), 'robust_se_hc3': float(result.bse[term]), 'p_value': float(result.pvalues[term]), 'r_squared': float(result.rsquared)})
    return pd.DataFrame(rows)

def subcortical_group_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    control = metrics.loc[metrics['group'] == 'Control', 'subcortical_5d_normalized_reorganization'].to_numpy(float)
    tms = metrics.loc[metrics['group'] == 'TMS', 'subcortical_5d_normalized_reorganization'].to_numpy(float)
    return pd.DataFrame([{'metric': 'subcortical_5d_normalized_reorganization', 'control_n': len(control), 'tms_n': len(tms), 'control_median': float(np.median(control)), 'tms_median': float(np.median(tms)), 'cliffs_delta_tms_minus_control': cliffs_delta(tms, control), 'mannwhitney_p': float(mannwhitneyu(tms, control, alternative='two-sided').pvalue)}])

def topology_adjusted_clinical_models(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    topology_features = ('brainstem', 'deep_subcortical_gray', 'cerebral_white_matter', 'cerebral_cortex')
    feature_matrix = np.column_stack([np.log1p(metrics[f'{compartment}_fraction_pct'].to_numpy(float)) for compartment in topology_features])
    standardized = StandardScaler().fit_transform(feature_matrix)
    pca = PCA(n_components=2)
    scores = pca.fit_transform(standardized)
    metrics_with_scores = metrics[['subject_id', 'group']].copy()
    metrics_with_scores['lesion_topology_pc1'] = scores[:, 0]
    metrics_with_scores['lesion_topology_pc2'] = scores[:, 1]
    loadings = pd.DataFrame({'compartment': topology_features, 'compartment_label': [COMPARTMENT_LABELS[name] for name in topology_features], 'pc1_weight': pca.components_[0], 'pc2_weight': pca.components_[1]})
    explained = pd.DataFrame({'component': ['PC1', 'PC2'], 'explained_variance_ratio': pca.explained_variance_ratio_, 'cumulative_explained_variance_ratio': np.cumsum(pca.explained_variance_ratio_)})
    rows = []
    for (outcome, (pre_column, post_column, direction)) in OUTCOMES.items():
        baseline = direction * metrics[pre_column].to_numpy(float)
        post = direction * metrics[post_column].to_numpy(float)
        baseline_mean = float(np.mean(baseline))
        baseline_sd = float(np.std(baseline, ddof=1))
        baseline_z = (baseline - baseline_mean) / baseline_sd
        post_z = (post - baseline_mean) / baseline_sd
        common = pd.DataFrame({'baseline_outcome_z': baseline_z, 'group_tms': metrics['group_tms'].to_numpy(float), 'lesion_burden_z': metrics['lesion_burden_z'].to_numpy(float), 'age_z': metrics['age_z'].to_numpy(float), 'sex_male': metrics['sex_male'].to_numpy(float), 'mask_lateralization_index': metrics['mask_lateralization_index'].to_numpy(float)})
        specifications = {'volume_adjusted': common, 'volume_and_topology_adjusted': common.assign(lesion_topology_pc1=scores[:, 0], lesion_topology_pc2=scores[:, 1])}
        for (model_name, predictors) in specifications.items():
            result = fit_hc3(post_z, predictors)
            term = 'group_tms'
            rows.append({'outcome': outcome, 'model': model_name, 'n': int(result.nobs), 'term': term, 'standardized_beta': float(result.params[term]), 'ci_low': float(result.conf_int().loc[term, 0]), 'ci_high': float(result.conf_int().loc[term, 1]), 'robust_se_hc3': float(result.bse[term]), 'p_value': float(result.pvalues[term]), 'r_squared': float(result.rsquared)})
    table = pd.DataFrame(rows)
    table['fdr_q'] = np.nan
    for model_name in table['model'].unique():
        selected = table['model'] == model_name
        table.loc[selected, 'fdr_q'] = multipletests(table.loc[selected, 'p_value'], method='fdr_bh')[1]
    return (table, metrics_with_scores, pd.concat([loadings.assign(record_type='loading'), explained.assign(record_type='variance')], ignore_index=True, sort=False))

def plot_topology_adjusted_clinical_models(table: pd.DataFrame) -> None:
    (fig, ax) = plt.subplots(figsize=(6.2, 2.85))
    outcomes = list(OUTCOMES)
    y = np.arange(len(outcomes))[::-1]
    model_styles = {'volume_adjusted': ('Volume-adjusted', '#777777', -0.1), 'volume_and_topology_adjusted': ('Volume + topology-adjusted', '#D65F45', 0.1)}
    ax.axvline(0, color='#999999', linewidth=0.7)
    for (model_name, (label, color, offset)) in model_styles.items():
        subset = table.loc[table['model'] == model_name].set_index('outcome').loc[outcomes]
        estimate = subset['standardized_beta'].to_numpy(float)
        low = subset['ci_low'].to_numpy(float)
        high = subset['ci_high'].to_numpy(float)
        ax.errorbar(estimate, y + offset, xerr=np.vstack((estimate - low, high - estimate)), fmt='o', color=color, ecolor=color, alpha=0.9, markersize=4.2, capsize=2, linewidth=0.85, label=label)
    ax.set_yticks(y, outcomes)
    ax.set_xlabel('TMS coefficient (95% CI), standardized outcome units')
    ax.set_ylabel('Post-intervention outcome\n(higher = better; baseline-adjusted)')
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=6.2)
    fig.suptitle('Treatment estimate after lesion-topology adjustment', fontsize=9, fontweight='bold', y=0.96)
    fig.subplots_adjust(left=0.13, right=0.64, bottom=0.2, top=0.84)
    save_figure(fig, 'C2_topology_adjusted_clinical_models')
    plt.close(fig)

def plot_adjusted_scatter(ax: plt.Axes, x: np.ndarray, y: np.ndarray, groups: np.ndarray, row: pd.Series, xlabel: str, ylabel: str) -> None:
    for group in GROUPS:
        mask = groups == group
        ax.scatter(x[mask], y[mask], s=19, alpha=0.72, color=GROUP_COLORS[group], edgecolor='white', linewidth=0.35, label=group)
    fit = np.polyfit(x, y, 1)
    line_x = np.linspace(float(x.min()), float(x.max()), 100)
    ax.plot(line_x, np.polyval(fit, line_x), color='#333333', linewidth=1.0)
    ax.axhline(0, color='#BDBDBD', linewidth=0.5)
    ax.axvline(0, color='#BDBDBD', linewidth=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    p_text = '<0.001' if row.p_value < 0.001 else f'={row.p_value:.3f}'
    ax.text(0.03, 0.96, f'beta={row.standardized_beta:.2f}, p{p_text}', transform=ax.transAxes, ha='left', va='top', fontsize=6.5)

def plot_subcortical_models(metrics: pd.DataFrame, global_table: pd.DataFrame, compartment_table: pd.DataFrame, partial: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    (fig, axes) = plt.subplots(1, 3, figsize=(7.2, 2.65), gridspec_kw={'width_ratios': (1, 1.05, 1)})
    groups = metrics['group'].to_numpy()
    global_lookup = global_table.set_index('outcome')
    plot_adjusted_scatter(axes[0], *partial['lesion_to_subcortical'], groups, global_lookup.loc['subcortical_5d_normalized_reorganization'], 'Adjusted total lesion burden', 'Adjusted subcortical reorganization')
    axes[0].legend(loc='lower right', fontsize=6.2)
    axes[0].set_title('Total lesion burden')
    axes[0].text(-0.18, 1.08, 'a', transform=axes[0].transAxes, fontsize=10, fontweight='bold', va='top')
    forest = compartment_table.loc[compartment_table['estimable']].copy().iloc[::-1]
    y = np.arange(len(forest))
    axes[1].axvline(0, color='#888888', linewidth=0.7)
    axes[1].errorbar(forest['standardized_beta'], y, xerr=np.vstack((forest['standardized_beta'] - forest['ci_low'], forest['ci_high'] - forest['standardized_beta'])), fmt='o', color='#5B5B5B', ecolor='#8A8A8A', markersize=4, capsize=2, linewidth=0.8)
    axes[1].set_yticks(y, forest['compartment_label'])
    axes[1].set_xlim(-0.6, 2.1)
    axes[1].set_xlabel('Standardized beta (95% CI)')
    axes[1].set_title('Topography beyond total volume')
    for (ypos, row) in enumerate(forest.itertuples()):
        q_text = 'NA' if not np.isfinite(row.fdr_q) else '<0.001' if row.fdr_q < 0.001 else f'={row.fdr_q:.3f}'
        axes[1].text(1.38, ypos, f'q{q_text}; n={row.involved_n}', transform=axes[1].transData, ha='left', va='center', fontsize=5.5, color='#555555')
    axes[1].text(-0.18, 1.08, 'b', transform=axes[1].transAxes, fontsize=10, fontweight='bold', va='top')
    plot_adjusted_scatter(axes[2], *partial['subcortical_to_clinical'], groups, global_lookup.loc['clinical_composite_post'], 'Adjusted subcortical reorganization', 'Adjusted post clinical composite')
    axes[2].set_title('Clinical relevance')
    axes[2].text(-0.18, 1.08, 'c', transform=axes[2].transAxes, fontsize=10, fontweight='bold', va='top')
    fig.suptitle('Anatomical lesion burden and subcortical gradient reorganization', fontsize=9.5, fontweight='bold', y=1.02)
    fig.tight_layout(w_pad=1.4)
    save_figure(fig, 'D2_lesion_subcortical_gradient_associations')
    plt.close(fig)

def main() -> None:
    ensure_output_dirs()
    (lesion_reference, clinical) = load_reference_tables()
    (reference, ho_sub, cerebellum, jhu, brain_mask) = load_atlases()
    (compartment_map, compartment_metadata) = build_compartment_map(ho_sub, cerebellum, brain_mask)
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(compartment_map, reference.affine, header), NIFTI / 'B2_anatomical_compartment_labels.nii.gz')
    (participant, tracts) = extract_anatomical_metrics(lesion_reference, reference, compartment_map, jhu, brain_mask)
    prevalence = compartment_prevalence(participant)
    comparisons = compartment_group_comparisons(participant)
    tract_summary = summarize_tracts(tracts)
    participant.to_csv(TABLES / 'B2_anatomical_compartment_metrics.csv', index=False, encoding='utf-8-sig')
    prevalence.to_csv(TABLES / 'B2_anatomical_compartment_prevalence.csv', index=False, encoding='utf-8-sig')
    comparisons.to_csv(TABLES / 'B2_anatomical_group_comparisons.csv', index=False, encoding='utf-8-sig')
    tracts.to_csv(TABLES / 'B2_jhu_tract_burden.csv', index=False, encoding='utf-8-sig')
    tract_summary.to_csv(TABLES / 'B2_jhu_tract_prevalence.csv', index=False, encoding='utf-8-sig')
    plot_anatomical_burden(participant, tract_summary)
    gradient_metrics = subcortical_gradient_metrics(participant, clinical)
    (global_models, compartment_models, partial) = subcortical_models(gradient_metrics)
    sensitivity_models = subcortical_sensitivity_models(gradient_metrics)
    gradient_group_comparison = subcortical_group_comparison(gradient_metrics)
    (topology_clinical, topology_scores, topology_pca) = topology_adjusted_clinical_models(gradient_metrics)
    gradient_metrics.to_csv(TABLES / 'D2_subcortical_gradient_subject_metrics.csv', index=False, encoding='utf-8-sig')
    global_models.to_csv(TABLES / 'D2_subcortical_gradient_global_models.csv', index=False, encoding='utf-8-sig')
    compartment_models.to_csv(TABLES / 'D2_compartment_subcortical_models.csv', index=False, encoding='utf-8-sig')
    sensitivity_models.to_csv(TABLES / 'D2_subcortical_gradient_sensitivity_models.csv', index=False, encoding='utf-8-sig')
    gradient_group_comparison.to_csv(TABLES / 'D2_subcortical_gradient_group_comparison.csv', index=False, encoding='utf-8-sig')
    topology_clinical.to_csv(TABLES / 'C2_topology_adjusted_clinical_models.csv', index=False, encoding='utf-8-sig')
    topology_scores.to_csv(TABLES / 'C2_lesion_topology_pca_subject_scores.csv', index=False, encoding='utf-8-sig')
    topology_pca.to_csv(TABLES / 'C2_lesion_topology_pca_definition.csv', index=False, encoding='utf-8-sig')
    plot_topology_adjusted_clinical_models(topology_clinical)
    plot_subcortical_models(gradient_metrics, global_models, compartment_models, partial)
    all_prevalence = prevalence.loc[prevalence['group'] == 'All'].set_index('compartment')
    manifest = {'analysis_date': '2026-08-03', 'n': int(len(participant)), 'group_n': participant['group'].value_counts().sort_index().to_dict(), 'compartment_definition': {'source': 'Harvard-Oxford subcortical max-probability atlas (25% threshold), supplemented by FNIRT cerebellar atlas', 'exclusive_priority': list(COMPARTMENTS[:-1]), 'deep_subcortical_gray_includes': ['thalamus', 'caudate', 'putamen', 'pallidum', 'hippocampus', 'amygdala', 'accumbens'], 'atlas_unlabeled': 'all lesion voxels not assigned to the five named atlas compartments', 'map_metadata': compartment_metadata}, 'white_matter_tract_analysis': 'JHU ICBM-DTI-81 label atlas; descriptive direct overlap only', 'subcortical_gradient': {'input': 'paired 54 x 5 arrays', 'node_identity_available': False, 'primary_metric': 'five-dimensional follow-up-to-baseline orthogonal Procrustes residual RMS divided by baseline five-dimensional RMS dispersion', 'reason_for_global_only': 'the 54-node anatomical identity/order is undocumented'}, 'compartment_prevalence_n': {name: int(all_prevalence.loc[name, 'involved_n']) for name in COMPARTMENTS}, 'outside_mni_brain_qc': {'maximum_fraction_pct': float(participant['outside_mni_brain_fraction_pct'].max()), 'subjects_over_1pct': participant.loc[participant['outside_mni_brain_fraction_pct'] > 1, 'subject_id'].tolist()}, 'statistical_models': {'group_comparison': 'Mann-Whitney U for lesion fraction and Fisher exact test for presence; BH FDR by family', 'lesion_to_subcortical': 'HC3 OLS adjusted for group, age, sex, mask lateralization, and baseline NIHSS', 'compartment_to_subcortical': 'one compartment at a time, adjusted for total lesion volume and the same covariates; fewer than five involved participants is non-estimable', 'subcortical_to_clinical': 'HC3 OLS adjusted for baseline composite, group, total lesion volume, age, sex, and mask lateralization', 'sensitivity': 'excludes the largest lesion, all manual review flags, or outside-brain QC outlier; an alternative raw-RMS model adjusts for baseline gradient dispersion', 'topology_adjusted_clinical': 'baseline-adjusted HC3 OLS with group, total lesion volume, age, sex, mask lateralization, and two PCA scores derived from log1p brainstem, deep-gray, white-matter, and cortical lesion fractions'}}
    (RESULTS / 'anatomical_extension_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Participants: {len(participant)}')
    print('Compartment prevalence:')
    print(all_prevalence[['involved_n', 'prevalence_pct', 'median_lesion_fraction_pct']].to_string())
    print('\nGroup comparisons:')
    print(comparisons[['compartment', 'cliffs_delta_tms_minus_control', 'mannwhitney_p', 'mannwhitney_fdr_q']].to_string(index=False))
    print('\nSubcortical global models:')
    print(global_models.to_string(index=False))
    print('\nCompartment-subcortical models:')
    print(compartment_models.to_string(index=False))
    print('\nSubcortical sensitivity models:')
    print(sensitivity_models.to_string(index=False))
    print('\nTopology-adjusted clinical group effects:')
    print(topology_clinical.to_string(index=False))
if __name__ == '__main__':
    main()
