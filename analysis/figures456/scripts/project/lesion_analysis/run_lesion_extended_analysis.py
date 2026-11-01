from __future__ import annotations
import os as _portable_os
from pathlib import Path as _PortablePath

def _portable_path(kind, relative):
    project = _PortablePath(_portable_os.environ['PAPER_PROJECT_ROOT'])
    root = project if kind == 'project' else project.parent
    return str(root / relative)
import json
from pathlib import Path
from xml.etree import ElementTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import chi2_contingency, mannwhitneyu
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
ROOT = Path(_portable_path('project', ''))
ANALYSIS = ROOT / 'lesion_analysis'
REG = ANALYSIS / 'mni_registration'
RESOURCES = ANALYSIS / 'resources'
RESULTS = ANALYSIS / 'results_abcd'
TABLES = RESULTS / 'tables'
FIGURES = RESULTS / 'figures'
GROUPS = ('Control', 'TMS')
GROUP_COLORS = {'Control': '#4C78A8', 'TMS': '#D65F45'}
OUTCOMES = {'NIHSS': ('NIHSS_pre', 'NIHSS_post', -1.0), 'ADL': ('ADL_pre', 'ADL_post', 1.0), 'MMSE': ('MMSE_pre', 'MMSE_post', 1.0), 'HAMD': ('HAMD_pre', 'HAMD_post', -1.0), 'VFT': ('VFT_pre', 'VFT_post', 1.0)}
COMPARTMENTS = ('brainstem', 'deep_subcortical_gray', 'cerebral_white_matter', 'cerebral_cortex', 'cerebellum', 'atlas_unlabeled')
COMPARTMENT_LABELS = {'brainstem': 'Brainstem', 'deep_subcortical_gray': 'Deep/subcortical gray', 'cerebral_white_matter': 'Cerebral white matter', 'cerebral_cortex': 'Cerebral cortex', 'cerebellum': 'Cerebellum', 'atlas_unlabeled': 'Atlas-unlabeled'}
COMPARTMENT_COLORS = {'brainstem': '#457B9D', 'deep_subcortical_gray': '#6A4C93', 'cerebral_white_matter': '#2A9D8F', 'cerebral_cortex': '#E76F51', 'cerebellum': '#E9C46A', 'atlas_unlabeled': '#B8B8B8'}
XTRACT_SYSTEMS = {'motor_projection': [13, 14], 'dorsal_language_attention': [1, 2, 15, 16, 30, 31, 32, 33, 34, 35], 'ventral_association': [21, 22, 23, 24, 26, 27, 38, 39, 40, 41], 'limbic': [7, 8, 9, 10, 11, 12, 19, 20], 'thalamic_sensory_radiations': [3, 4, 5, 6, 28, 29, 36, 37], 'commissural': [0, 17, 18], 'middle_cerebellar_peduncle': [25]}
SYSTEM_LABELS = {'motor_projection': 'Corticospinal projection', 'dorsal_language_attention': 'Dorsal language/attention', 'ventral_association': 'Ventral association', 'limbic': 'Limbic', 'thalamic_sensory_radiations': 'Thalamic/sensory radiations', 'commissural': 'Commissural', 'middle_cerebellar_peduncle': 'Middle cerebellar peduncle'}
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans', 'sans-serif'], 'font.size': 7, 'axes.titlesize': 8, 'axes.labelsize': 7, 'axes.linewidth': 0.7, 'axes.spines.right': False, 'axes.spines.top': False, 'legend.frameon': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})

def save_figure(fig: plt.Figure, stem: str) -> None:
    for (suffix, kwargs) in (('png', {'dpi': 300}), ('svg', {}), ('pdf', {}), ('tiff', {'dpi': 600})):
        fig.savefig(FIGURES / f'{stem}.{suffix}', bbox_inches='tight', facecolor='white', **kwargs)

def zscore(values: pd.Series | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    sd = float(np.nanstd(array, ddof=1))
    return (array - float(np.nanmean(array))) / sd if np.isfinite(sd) and sd > 0 else np.zeros_like(array)

def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float((np.sum(x[:, None] > y[None, :]) - np.sum(x[:, None] < y[None, :])) / (len(x) * len(y)))

def fit_hc3(y: np.ndarray, predictors: pd.DataFrame):
    design = sm.add_constant(predictors.astype(float), has_constant='add')
    return sm.OLS(np.asarray(y, dtype=float), design, missing='raise').fit(cov_type='HC3')

def baseline_mask(group: str, subject_id: str) -> Path:
    return REG / 'masks' / group / 'before' / f'{subject_id}_lesion_mni2mm.nii.gz'

def post_mask(group: str, subject_id: str) -> Path:
    return REG / 'masks' / group / 'after' / f'{subject_id}_lesion_mni2mm.nii.gz'

def load_analysis_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    anatomy = pd.read_csv(TABLES / 'B2_anatomical_compartment_metrics.csv', dtype={'subject_id': str})
    lesion_qc = pd.read_csv(TABLES / 'B_participant_lesion_metrics.csv', dtype={'subject_id': str})
    clinical = pd.read_csv(TABLES / 'D2_subcortical_gradient_subject_metrics.csv', dtype={'subject_id': str})
    selected = lesion_qc[['subject_id', 'mask_derived_side', 'registration_manual_status']]
    anatomy = anatomy.merge(selected, on='subject_id', how='left', validate='one_to_one')
    if len(anatomy) != 38 or len(clinical) != 38 or set(anatomy['subject_id']) != set(clinical['subject_id']):
        raise RuntimeError('Expected the frozen 38-subject common cohort')
    return (anatomy, clinical)

def extract_spatial_phenotypes(anatomy: pd.DataFrame) -> pd.DataFrame:
    rows = []
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    for row in anatomy.itertuples():
        image = nib.load(str(baseline_mask(row.group, row.subject_id)))
        mask = np.asarray(image.dataobj) > 0.5
        indices = np.argwhere(mask)
        coordinates = nib.affines.apply_affine(image.affine, indices)
        (labeled, component_n) = ndimage.label(mask, structure=structure)
        component_sizes = np.bincount(labeled[mask], minlength=component_n + 1)[1:]
        lower = coordinates.min(axis=0)
        upper = coordinates.max(axis=0)
        span = upper - lower + np.abs(np.diag(image.affine)[:3])
        bbox_volume_ml = float(np.prod(span) / 1000.0)
        lesion_volume_ml = float(mask.sum() * abs(np.linalg.det(image.affine[:3, :3])) / 1000.0)
        fractions = {name: float(getattr(row, f'{name}_fraction_pct')) for name in COMPARTMENTS}
        dominant = max(fractions, key=fractions.get)
        rows.append({'subject_id': row.subject_id, 'group': row.group, 'lesion_volume_ml': lesion_volume_ml, 'centroid_x_mm': float(coordinates[:, 0].mean()), 'centroid_y_mm': float(coordinates[:, 1].mean()), 'centroid_z_mm': float(coordinates[:, 2].mean()), 'absolute_centroid_x_mm': float(abs(coordinates[:, 0].mean())), 'extent_x_mm': float(span[0]), 'extent_y_mm': float(span[1]), 'extent_z_mm': float(span[2]), 'maximum_extent_mm': float(span.max()), 'component_count_26': int(component_n), 'largest_component_fraction_pct': float(component_sizes.max() * 100.0 / mask.sum()), 'bounding_box_fill_pct': float(lesion_volume_ml * 100.0 / bbox_volume_ml), 'dominant_compartment': dominant, 'dominant_compartment_label': COMPARTMENT_LABELS[dominant], 'mask_derived_side': row.mask_derived_side, 'registration_manual_status': row.registration_manual_status})
    return pd.DataFrame(rows)

def continuous_group_comparisons(data: pd.DataFrame, metrics: dict[str, str]) -> pd.DataFrame:
    rows = []
    for (metric, label) in metrics.items():
        control = data.loc[data['group'] == 'Control', metric].dropna().to_numpy(float)
        tms = data.loc[data['group'] == 'TMS', metric].dropna().to_numpy(float)
        rows.append({'metric': metric, 'metric_label': label, 'control_n': len(control), 'tms_n': len(tms), 'control_median': float(np.median(control)), 'tms_median': float(np.median(tms)), 'cliffs_delta_tms_minus_control': cliffs_delta(tms, control), 'mannwhitney_p': float(mannwhitneyu(tms, control, alternative='two-sided').pvalue)})
    table = pd.DataFrame(rows)
    table['fdr_q'] = multipletests(table['mannwhitney_p'], method='fdr_bh')[1]
    return table

def permutation_contingency(data: pd.DataFrame, category: str, permutations: int=20000) -> tuple[pd.DataFrame, dict]:
    table = pd.crosstab(data['group'], data[category]).reindex(index=GROUPS, fill_value=0)
    observed = float(chi2_contingency(table, correction=False)[0])
    rng = np.random.default_rng(20260803)
    groups = data['group'].eq('TMS').to_numpy(np.uint8)
    (category_codes, _) = pd.factorize(data[category], sort=True)
    category_n = table.shape[1]
    totals = np.bincount(category_codes, minlength=category_n)
    row_totals = np.array([np.count_nonzero(groups == 0), np.count_nonzero(groups == 1)], dtype=float)
    expected = row_totals[:, None] * totals[None, :] / len(data)

    def statistic_for(group_labels: np.ndarray) -> float:
        tms_counts = np.bincount(category_codes[group_labels == 1], minlength=category_n)
        counts = np.vstack((totals - tms_counts, tms_counts))
        return float(np.sum((counts - expected) ** 2 / expected))
    extreme = 0
    for _ in range(permutations):
        statistic = statistic_for(rng.permutation(groups))
        extreme += int(statistic >= observed - 1e-12)
    result = {'category': category, 'chi_square': observed, 'permutations': permutations, 'permutation_p': (extreme + 1) / (permutations + 1)}
    long_table = table.reset_index().melt(id_vars='group', var_name='category', value_name='n')
    long_table['category_variable'] = category
    return (long_table, result)

def plot_spatial_phenotypes(spatial: pd.DataFrame, comparisons: pd.DataFrame) -> None:
    (fig, axes) = plt.subplots(2, 2, figsize=(7.2, 5.0))
    sizes = 14 + 12 * np.log1p(spatial['lesion_volume_ml'].to_numpy(float))
    for group in GROUPS:
        selected = spatial['group'] == group
        axes[0, 0].scatter(spatial.loc[selected, 'centroid_x_mm'], spatial.loc[selected, 'centroid_y_mm'], s=sizes[selected], color=GROUP_COLORS[group], alpha=0.72, edgecolor='white', linewidth=0.35, label=group)
        axes[0, 1].scatter(spatial.loc[selected, 'centroid_y_mm'], spatial.loc[selected, 'centroid_z_mm'], s=sizes[selected], color=GROUP_COLORS[group], alpha=0.72, edgecolor='white', linewidth=0.35, label=group)
    axes[0, 0].axvline(0, color='#999999', linewidth=0.6)
    axes[0, 0].set_xlabel('MNI x (mm)')
    axes[0, 0].set_ylabel('MNI y (mm)')
    axes[0, 0].set_title('Axial distribution of lesion centroids')
    axes[0, 0].legend(fontsize=6.2)
    axes[0, 1].set_xlabel('MNI y (mm)')
    axes[0, 1].set_ylabel('MNI z (mm)')
    axes[0, 1].set_title('Sagittal distribution of lesion centroids')
    phenotype = pd.crosstab(spatial['group'], spatial['dominant_compartment'], normalize='index').reindex(index=GROUPS, fill_value=0)
    phenotype = phenotype.reindex(columns=[name for name in COMPARTMENTS if name in phenotype.columns], fill_value=0)
    bottom = np.zeros(len(GROUPS))
    for compartment in phenotype.columns:
        values = phenotype[compartment].to_numpy(float) * 100
        axes[1, 0].bar(np.arange(len(GROUPS)), values, bottom=bottom, color=COMPARTMENT_COLORS[compartment], label=COMPARTMENT_LABELS[compartment])
        bottom += values
    axes[1, 0].set_xticks(np.arange(len(GROUPS)), GROUPS)
    axes[1, 0].set_ylim(0, 100)
    axes[1, 0].set_ylabel('Participants (%)')
    axes[1, 0].set_title('Dominant lesion phenotype')
    axes[1, 0].legend(fontsize=5.7, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.16))
    rng = np.random.default_rng(20260803)
    metric = 'largest_component_fraction_pct'
    for (index, group) in enumerate(GROUPS):
        values = spatial.loc[spatial['group'] == group, metric].to_numpy(float)
        axes[1, 1].scatter(np.full(len(values), index) + rng.uniform(-0.08, 0.08, len(values)), values, s=18, color=GROUP_COLORS[group], alpha=0.7, edgecolor='white', linewidth=0.3)
        axes[1, 1].plot([index - 0.12, index + 0.12], [np.median(values)] * 2, color='#222222', linewidth=1.0)
    model = comparisons.set_index('metric').loc[metric]
    axes[1, 1].set_xticks(np.arange(len(GROUPS)), GROUPS)
    axes[1, 1].set_ylabel('Largest connected component (%)')
    axes[1, 1].set_title(f'Lesion contiguity (q={model.fdr_q:.3f})')
    for (label, ax) in zip('abcd', axes.ravel()):
        ax.text(-0.13, 1.08, label, transform=ax.transAxes, fontsize=10, fontweight='bold', va='top')
    fig.suptitle('Spatial and morphological phenotypes of baseline lesions', fontsize=10, fontweight='bold', y=1.0)
    fig.tight_layout(h_pad=2.4, w_pad=1.5)
    save_figure(fig, 'E_spatial_lesion_phenotypes')
    plt.close(fig)

def xtract_labels() -> dict[int, str]:
    root = ElementTree.parse(RESOURCES / 'XTRACT.xml').getroot()
    labels = {int(node.attrib['index']): (node.text or '').strip() for node in root.findall('./data/label')}
    if len(labels) != 42 or set(labels) != set(range(42)):
        raise RuntimeError('Unexpected XTRACT label definition')
    return labels

def extract_xtract_metrics(anatomy: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    atlas_image = nib.load(str(RESOURCES / 'XTRACT_HCP_prob_2mm.nii.gz'))
    atlas = np.asarray(atlas_image.dataobj, dtype=np.float32)
    labels = xtract_labels()
    reference = nib.load(str(baseline_mask(anatomy.iloc[0]['group'], anatomy.iloc[0]['subject_id'])))
    if atlas.shape != reference.shape + (42,) or not np.allclose(atlas_image.affine, reference.affine):
        raise RuntimeError(f'XTRACT geometry mismatch: {atlas.shape}, {reference.shape}')
    atlas = np.clip(atlas, 0, 100)
    probability_mass = atlas.reshape(-1, 42).sum(axis=0) / 100.0
    voxel_ml = abs(float(np.linalg.det(reference.affine[:3, :3]))) / 1000.0
    system_maps = {system: np.max(atlas[..., indices], axis=3) for (system, indices) in XTRACT_SYSTEMS.items()}
    system_mass = {system: float(values.sum() / 100.0) for (system, values) in system_maps.items()}
    tract_rows = []
    system_rows = []
    for row in anatomy.itertuples():
        image = nib.load(str(baseline_mask(row.group, row.subject_id)))
        lesion = np.asarray(image.dataobj) > 0.5
        lesion_values = atlas[lesion, :]
        mass = lesion_values.sum(axis=0) / 100.0
        for (index, tract_name) in labels.items():
            tract_rows.append({'subject_id': row.subject_id, 'group': row.group, 'tract_index': index, 'tract_name': tract_name, 'expected_overlap_ml': float(mass[index] * voxel_ml), 'tract_probability_mass_coverage_pct': float(mass[index] * 100.0 / probability_mass[index]), 'mean_tract_probability_within_lesion_pct': float(lesion_values[:, index].mean()), 'overlap_at_probability_5pct': int(np.any(lesion_values[:, index] >= 5.0))})
        for (system, probability) in system_maps.items():
            values = probability[lesion]
            overlap_mass = float(values.sum() / 100.0)
            system_rows.append({'subject_id': row.subject_id, 'group': row.group, 'system': system, 'system_label': SYSTEM_LABELS[system], 'expected_overlap_ml': overlap_mass * voxel_ml, 'system_probability_mass_coverage_pct': overlap_mass * 100.0 / system_mass[system], 'mean_system_probability_within_lesion_pct': float(values.mean()), 'overlap_at_probability_5pct': int(np.any(values >= 5.0))})
    return (pd.DataFrame(tract_rows), pd.DataFrame(system_rows), system_maps)

def xtract_group_comparisons(table: pd.DataFrame, unit: str, label_column: str) -> pd.DataFrame:
    rows = []
    for (value, subset) in table.groupby(unit, sort=False):
        control = subset.loc[subset['group'] == 'Control', 'expected_overlap_ml'].to_numpy(float)
        tms = subset.loc[subset['group'] == 'TMS', 'expected_overlap_ml'].to_numpy(float)
        rows.append({unit: value, label_column: subset[label_column].iloc[0], 'control_involved_n': int(subset.loc[subset['group'] == 'Control', 'overlap_at_probability_5pct'].sum()), 'tms_involved_n': int(subset.loc[subset['group'] == 'TMS', 'overlap_at_probability_5pct'].sum()), 'control_median_expected_overlap_ml': float(np.median(control)), 'tms_median_expected_overlap_ml': float(np.median(tms)), 'cliffs_delta_tms_minus_control': cliffs_delta(tms, control), 'mannwhitney_p': float(mannwhitneyu(tms, control, alternative='two-sided').pvalue)})
    result = pd.DataFrame(rows)
    result['fdr_q'] = multipletests(result['mannwhitney_p'], method='fdr_bh')[1]
    return result

def xtract_clinical_models(system_table: pd.DataFrame, clinical: pd.DataFrame) -> pd.DataFrame:
    wide = system_table.pivot(index='subject_id', columns='system', values='expected_overlap_ml')
    presence = system_table.pivot(index='subject_id', columns='system', values='overlap_at_probability_5pct')
    data = clinical.copy().set_index('subject_id')
    rows = []
    outcome_specs: list[tuple[str, np.ndarray, np.ndarray, str]] = []
    composite_baseline = zscore(data['clinical_composite_baseline'])
    composite_post = zscore(data['clinical_composite_post'])
    outcome_specs.append(('clinical_composite', composite_baseline, composite_post, 'primary_composite'))
    for (outcome, (pre_column, post_column, direction)) in OUTCOMES.items():
        baseline = direction * data[pre_column].to_numpy(float)
        post = direction * data[post_column].to_numpy(float)
        mean = float(np.mean(baseline))
        sd = float(np.std(baseline, ddof=1))
        outcome_specs.append((outcome, (baseline - mean) / sd, (post - mean) / sd, 'secondary_outcomes'))
    for system in XTRACT_SYSTEMS:
        burden = np.log1p(wide.loc[data.index, system].to_numpy(float))
        burden_z = zscore(burden)
        control_n = int(presence.loc[data.index[data['group'] == 'Control'], system].sum())
        tms_n = int(presence.loc[data.index[data['group'] == 'TMS'], system].sum())
        interaction_estimable = control_n >= 3 and tms_n >= 3 and (np.std(burden[data['group'] == 'Control']) > 0) and (np.std(burden[data['group'] == 'TMS']) > 0)
        for (outcome, baseline_z, post_z, family) in outcome_specs:
            predictors = pd.DataFrame({'baseline_outcome_z': baseline_z, 'group_tms': data['group_tms'].to_numpy(float), 'lesion_burden_z': zscore(data['log1p_lesion_volume']), 'xtract_system_burden_z': burden_z, 'group_x_system_burden': data['group_tms'].to_numpy(float) * burden_z, 'age_z': data['age_z'].to_numpy(float), 'sex_male': data['sex_male'].to_numpy(float), 'mask_lateralization_index': data['mask_lateralization_index'].to_numpy(float)})
            model = fit_hc3(post_z, predictors)
            rows.append({'system': system, 'system_label': SYSTEM_LABELS[system], 'outcome': outcome, 'multiple_testing_family': family, 'n': int(model.nobs), 'control_involved_n': control_n, 'tms_involved_n': tms_n, 'interaction_estimable': interaction_estimable, 'system_main_beta': float(model.params['xtract_system_burden_z']), 'system_main_ci_low': float(model.conf_int().loc['xtract_system_burden_z', 0]), 'system_main_ci_high': float(model.conf_int().loc['xtract_system_burden_z', 1]), 'system_main_p': float(model.pvalues['xtract_system_burden_z']), 'group_interaction_beta': float(model.params['group_x_system_burden']) if interaction_estimable else np.nan, 'group_interaction_ci_low': float(model.conf_int().loc['group_x_system_burden', 0]) if interaction_estimable else np.nan, 'group_interaction_ci_high': float(model.conf_int().loc['group_x_system_burden', 1]) if interaction_estimable else np.nan, 'group_interaction_p': float(model.pvalues['group_x_system_burden']) if interaction_estimable else np.nan, 'r_squared': float(model.rsquared)})
    result = pd.DataFrame(rows)
    result['system_main_fdr_q'] = np.nan
    result['group_interaction_fdr_q'] = np.nan
    for family in result['multiple_testing_family'].unique():
        selected = result['multiple_testing_family'] == family
        result.loc[selected, 'system_main_fdr_q'] = multipletests(result.loc[selected, 'system_main_p'], method='fdr_bh')[1]
        estimable = selected & result['group_interaction_p'].notna()
        if estimable.any():
            result.loc[estimable, 'group_interaction_fdr_q'] = multipletests(result.loc[estimable, 'group_interaction_p'], method='fdr_bh')[1]
    return result

def plot_xtract_results(tract_table: pd.DataFrame, system_table: pd.DataFrame, system_comparisons: pd.DataFrame, clinical_models: pd.DataFrame) -> None:
    (fig, axes) = plt.subplots(1, 3, figsize=(7.2, 3.1), gridspec_kw={'width_ratios': (1.1, 1.15, 1.05)})
    prevalence = tract_table.groupby('tract_name', as_index=False).agg(involved_n=('overlap_at_probability_5pct', 'sum'), median_overlap=('expected_overlap_ml', 'median')).nlargest(10, ['involved_n', 'median_overlap'])
    names = prevalence['tract_name'].tolist()[::-1]
    short = {'Cingulum subsection: Dorsal L': 'Dorsal cingulum (L)', 'Cingulum subsection: Dorsal R': 'Dorsal cingulum (R)', 'Cingulum subsection: Peri-genual L': 'Perigenual cingulum (L)', 'Cingulum subsection: Peri-genual R': 'Perigenual cingulum (R)', 'Cingulum subsection: Temporal L': 'Temporal cingulum (L)', 'Cingulum subsection: Temporal R': 'Temporal cingulum (R)', 'Inferior Fronto-Occipital Fasciculus L': 'IFOF (L)', 'Inferior Fronto-Occipital Fasciculus R': 'IFOF (R)', 'Middle Cerebellar Peduncle': 'Middle cerebellar peduncle'}
    y = np.arange(len(names))
    offsets = {'Control': -0.12, 'TMS': 0.12}
    for group in GROUPS:
        grouped = tract_table.loc[tract_table['group'] == group].groupby('tract_name')['overlap_at_probability_5pct'].mean() * 100
        axes[0].scatter([grouped.get(name, 0) for name in names], y + offsets[group], s=20, color=GROUP_COLORS[group], label=group)
    axes[0].set_yticks(y, [short.get(name, name) for name in names])
    axes[0].set_xlabel('Participants with >=5% tract probability overlap (%)')
    axes[0].set_title('Most commonly intersected XTRACT pathways')
    axes[0].grid(axis='x', color='#E5E5E5', linewidth=0.5)
    axes[0].legend(fontsize=6.2)
    order = list(XTRACT_SYSTEMS)[::-1]
    forest = system_comparisons.set_index('system').loc[order]
    y2 = np.arange(len(order))
    axes[1].axvline(0, color='#888888', linewidth=0.7)
    axes[1].scatter(forest['cliffs_delta_tms_minus_control'], y2, s=23, color='#555555')
    for (ypos, row) in enumerate(forest.itertuples()):
        axes[1].text(0.98, ypos, f'q={row.fdr_q:.3f}', transform=axes[1].get_yaxis_transform(), ha='right', va='center', fontsize=5.8)
    axes[1].set_xlim(-1.0, 1.0)
    axes[1].set_yticks(y2, [SYSTEM_LABELS[name] for name in order])
    axes[1].set_xlabel("Cliff's delta (TMS - Control)")
    axes[1].set_title('Probability-weighted system burden')
    primary = clinical_models.loc[clinical_models['outcome'] == 'clinical_composite'].set_index('system').loc[order]
    estimable = primary['interaction_estimable'].to_numpy(bool)
    axes[2].axvline(0, color='#888888', linewidth=0.7)
    axes[2].errorbar(primary.loc[estimable, 'group_interaction_beta'], y2[estimable], xerr=np.vstack((primary.loc[estimable, 'group_interaction_beta'] - primary.loc[estimable, 'group_interaction_ci_low'], primary.loc[estimable, 'group_interaction_ci_high'] - primary.loc[estimable, 'group_interaction_beta'])), fmt='o', color='#D65F45', ecolor='#A0A0A0', markersize=4, capsize=2, linewidth=0.8)
    for (ypos, row) in enumerate(primary.itertuples()):
        text = 'not estimable' if not row.interaction_estimable else f'q={row.group_interaction_fdr_q:.3f}'
        axes[2].text(0.98, ypos, text, transform=axes[2].get_yaxis_transform(), ha='right', va='center', fontsize=5.6)
    axes[2].set_yticks(y2, [])
    axes[2].set_xlabel('Group x tract-system burden beta (95% CI)')
    axes[2].set_title('Clinical composite moderation')
    for (label, ax) in zip('abc', axes):
        ax.text(-0.18, 1.08, label, transform=ax.transAxes, fontsize=10, fontweight='bold', va='top')
    fig.suptitle('Normative probabilistic tract involvement', fontsize=10, fontweight='bold', y=1.01)
    fig.tight_layout(w_pad=1.4)
    save_figure(fig, 'F_xtract_probability_weighted_burden')
    plt.close(fig)

def extract_longitudinal_metrics(anatomy: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in anatomy.itertuples():
        before_image = nib.load(str(baseline_mask(row.group, row.subject_id)))
        before = np.asarray(before_image.dataobj) > 0.5
        path = post_mask(row.group, row.subject_id)
        values: dict[str, object] = {'subject_id': row.subject_id, 'group': row.group, 'has_post_mask': int(path.exists()), 'baseline_volume_ml': float(before.sum() * abs(np.linalg.det(before_image.affine[:3, :3])) / 1000.0)}
        if not path.exists():
            values.update({'post_mask_nonempty': 0, 'primary_longitudinal_included': 0, 'post_volume_ml': np.nan, 'absolute_volume_change_ml': np.nan, 'symmetric_volume_change_pct': np.nan, 'dice': np.nan, 'baseline_persistence_pct': np.nan, 'post_new_fraction_pct': np.nan, 'centroid_displacement_mm': np.nan})
            rows.append(values)
            continue
        after_image = nib.load(str(path))
        after = np.asarray(after_image.dataobj) > 0.5
        if after_image.shape != before_image.shape or not np.allclose(after_image.affine, before_image.affine):
            raise RuntimeError(f'Longitudinal geometry mismatch: {row.subject_id}')
        voxel_ml = abs(float(np.linalg.det(before_image.affine[:3, :3]))) / 1000.0
        pre_volume = float(before.sum() * voxel_ml)
        post_volume = float(after.sum() * voxel_ml)
        intersection = int(np.count_nonzero(before & after))
        post_nonempty = bool(after.any())
        if post_nonempty:
            before_centroid = nib.affines.apply_affine(before_image.affine, np.argwhere(before)).mean(axis=0)
            after_centroid = nib.affines.apply_affine(after_image.affine, np.argwhere(after)).mean(axis=0)
            displacement = float(np.linalg.norm(after_centroid - before_centroid))
        else:
            displacement = np.nan
        values.update({'post_mask_nonempty': int(post_nonempty), 'primary_longitudinal_included': int(post_nonempty and row.subject_id != 'PRIVATE_SUBJECT_009'), 'post_volume_ml': post_volume, 'absolute_volume_change_ml': post_volume - pre_volume, 'symmetric_volume_change_pct': 200.0 * (post_volume - pre_volume) / (post_volume + pre_volume), 'dice': 2.0 * intersection / (before.sum() + after.sum()), 'baseline_persistence_pct': intersection * 100.0 / before.sum(), 'post_new_fraction_pct': (after.sum() - intersection) * 100.0 / after.sum() if post_nonempty else np.nan, 'centroid_displacement_mm': displacement})
        rows.append(values)
    return pd.DataFrame(rows)

def longitudinal_group_comparisons(longitudinal: pd.DataFrame) -> pd.DataFrame:
    metrics = {'absolute_volume_change_ml': 'Absolute volume change (ml)', 'symmetric_volume_change_pct': 'Symmetric volume change (%)', 'dice': 'Baseline-follow-up Dice', 'baseline_persistence_pct': 'Baseline lesion persistence (%)', 'post_new_fraction_pct': 'Follow-up new-lesion fraction (%)', 'centroid_displacement_mm': 'Centroid displacement (mm)'}
    primary = longitudinal.loc[longitudinal['primary_longitudinal_included'] == 1]
    return continuous_group_comparisons(primary, metrics)

def plot_longitudinal(longitudinal: pd.DataFrame, comparisons: pd.DataFrame) -> None:
    primary = longitudinal.loc[longitudinal['primary_longitudinal_included'] == 1].copy()
    (fig, axes) = plt.subplots(1, 3, figsize=(7.2, 2.85))
    x_positions = {'Control': (0, 1), 'TMS': (2.4, 3.4)}
    for group in GROUPS:
        subset = primary.loc[primary['group'] == group]
        (x0, x1) = x_positions[group]
        for row in subset.itertuples():
            axes[0].plot([x0, x1], [row.baseline_volume_ml, row.post_volume_ml], color=GROUP_COLORS[group], alpha=0.35, linewidth=0.8)
        axes[0].scatter(np.full(len(subset), x0), subset['baseline_volume_ml'], s=13, color=GROUP_COLORS[group], alpha=0.72)
        axes[0].scatter(np.full(len(subset), x1), subset['post_volume_ml'], s=13, color=GROUP_COLORS[group], alpha=0.72)
    axes[0].set_yscale('log')
    axes[0].set_xticks([0, 1, 2.4, 3.4], ['Baseline', 'Follow-up', 'Baseline', 'Follow-up'], rotation=24, ha='right')
    axes[0].text(0.5, 1.02, 'Control', transform=axes[0].get_xaxis_transform(), ha='center', fontweight='bold')
    axes[0].text(2.9, 1.02, 'TMS', transform=axes[0].get_xaxis_transform(), ha='center', fontweight='bold')
    axes[0].set_ylabel('Lesion volume (ml, log scale)')
    axes[0].set_title('Paired lesion volume')
    rng = np.random.default_rng(20260803)
    for (ax, metric, ylabel) in ((axes[1], 'symmetric_volume_change_pct', 'Symmetric volume change (%)'), (axes[2], 'dice', 'Baseline-follow-up Dice')):
        for (index, group) in enumerate(GROUPS):
            values = primary.loc[primary['group'] == group, metric].to_numpy(float)
            ax.scatter(np.full(len(values), index) + rng.uniform(-0.07, 0.07, len(values)), values, s=18, color=GROUP_COLORS[group], alpha=0.7, edgecolor='white', linewidth=0.3)
            ax.plot([index - 0.12, index + 0.12], [np.median(values)] * 2, color='#222222', linewidth=1.0)
        q = comparisons.set_index('metric').loc[metric, 'fdr_q']
        ax.set_xticks(np.arange(len(GROUPS)), GROUPS)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{ylabel} (q={q:.3f})')
    for (label, ax) in zip('abc', axes):
        ax.text(-0.18, 1.08, label, transform=ax.transAxes, fontsize=10, fontweight='bold', va='top')
    fig.suptitle('Longitudinal lesion-mask evolution', fontsize=9.5, fontweight='bold', y=1.02)
    fig.tight_layout(w_pad=1.4)
    save_figure(fig, 'G_longitudinal_lesion_evolution')
    plt.close(fig)

def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    (anatomy, clinical) = load_analysis_data()
    spatial = extract_spatial_phenotypes(anatomy)
    spatial_metrics = {'absolute_centroid_x_mm': 'Absolute centroid distance from midline (mm)', 'centroid_y_mm': 'Centroid anterior-posterior coordinate (mm)', 'centroid_z_mm': 'Centroid inferior-superior coordinate (mm)', 'maximum_extent_mm': 'Maximum lesion extent (mm)', 'component_count_26': '26-connected component count', 'largest_component_fraction_pct': 'Largest connected component fraction (%)', 'bounding_box_fill_pct': 'Bounding-box fill (%)'}
    spatial_comparisons = continuous_group_comparisons(spatial, spatial_metrics)
    (phenotype_counts, phenotype_test) = permutation_contingency(spatial, 'dominant_compartment')
    (laterality_counts, laterality_test) = permutation_contingency(spatial, 'mask_derived_side')
    spatial.to_csv(TABLES / 'E_spatial_lesion_phenotypes.csv', index=False, encoding='utf-8-sig')
    spatial_comparisons.to_csv(TABLES / 'E_spatial_group_comparisons.csv', index=False, encoding='utf-8-sig')
    pd.concat([phenotype_counts, laterality_counts], ignore_index=True).to_csv(TABLES / 'E_categorical_phenotype_counts.csv', index=False, encoding='utf-8-sig')
    (TABLES / 'E_categorical_permutation_tests.json').write_text(json.dumps({'dominant_compartment': phenotype_test, 'laterality': laterality_test}, indent=2), encoding='utf-8')
    plot_spatial_phenotypes(spatial, spatial_comparisons)
    (tract_table, system_table, _) = extract_xtract_metrics(anatomy)
    tract_comparisons = xtract_group_comparisons(tract_table, 'tract_index', 'tract_name')
    system_comparisons = xtract_group_comparisons(system_table, 'system', 'system_label')
    clinical_models = xtract_clinical_models(system_table, clinical)
    tract_table.to_csv(TABLES / 'F_xtract_tract_probability_burden.csv', index=False, encoding='utf-8-sig')
    system_table.to_csv(TABLES / 'F_xtract_system_probability_burden.csv', index=False, encoding='utf-8-sig')
    tract_comparisons.to_csv(TABLES / 'F_xtract_tract_group_comparisons.csv', index=False, encoding='utf-8-sig')
    system_comparisons.to_csv(TABLES / 'F_xtract_system_group_comparisons.csv', index=False, encoding='utf-8-sig')
    clinical_models.to_csv(TABLES / 'F_xtract_system_clinical_models.csv', index=False, encoding='utf-8-sig')
    plot_xtract_results(tract_table, system_table, system_comparisons, clinical_models)
    longitudinal = extract_longitudinal_metrics(anatomy)
    longitudinal_comparisons = longitudinal_group_comparisons(longitudinal)
    longitudinal.to_csv(TABLES / 'G_longitudinal_lesion_metrics.csv', index=False, encoding='utf-8-sig')
    longitudinal_comparisons.to_csv(TABLES / 'G_longitudinal_group_comparisons.csv', index=False, encoding='utf-8-sig')
    plot_longitudinal(longitudinal, longitudinal_comparisons)
    manifest = {'analysis_date': '2026-08-03', 'cohort_n': int(len(anatomy)), 'spatial_analysis': {'features': list(spatial_metrics), 'dominant_phenotype_definition': 'compartment containing the largest fraction of each baseline lesion', 'categorical_group_test': '20,000-label-permutation chi-square test'}, 'xtract_analysis': {'source': 'FSL XTRACT HCP probabilistic tract atlas, 42 pathways', 'resampling': '1 mm to MNI152 2 mm using FSL flirt, trilinear interpolation', 'burden': 'probability-weighted expected overlap volume; >=5% probability used only for prevalence/estimability', 'interpretation': 'normative tract involvement proxy, not subject-specific tractography or a full disconnectome', 'systems': {name: {'label': SYSTEM_LABELS[name], 'tract_indices': indices} for (name, indices) in XTRACT_SYSTEMS.items()}}, 'longitudinal_analysis': {'post_mask_available_n': int(longitudinal['has_post_mask'].sum()), 'primary_nonempty_paired_n': int(longitudinal['primary_longitudinal_included'].sum()), 'missing_post_subject': longitudinal.loc[longitudinal['has_post_mask'] == 0, 'subject_id'].tolist(), 'empty_post_subject': longitudinal.loc[(longitudinal['has_post_mask'] == 1) & (longitudinal['post_mask_nonempty'] == 0), 'subject_id'].tolist(), 'interpretation': 'secondary mask-evolution analysis; timing and delineation effects prevent causal treatment interpretation'}}
    (RESULTS / 'extended_lesion_analysis_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Spatial group comparisons:')
    print(spatial_comparisons.to_string(index=False))
    print('\nCategorical tests:')
    print(json.dumps({'dominant_compartment': phenotype_test, 'laterality': laterality_test}, indent=2))
    print('\nXTRACT system group comparisons:')
    print(system_comparisons.to_string(index=False))
    print('\nPrimary XTRACT clinical models:')
    print(clinical_models.loc[clinical_models['outcome'] == 'clinical_composite'].to_string(index=False))
    print('\nLongitudinal group comparisons:')
    print(longitudinal_comparisons.to_string(index=False))
if __name__ == '__main__':
    main()
