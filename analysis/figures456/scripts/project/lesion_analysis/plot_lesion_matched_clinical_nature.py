from __future__ import annotations
import os as _portable_os
from pathlib import Path as _PortablePath

def _portable_path(kind, relative):
    project = _PortablePath(_portable_os.environ['PAPER_PROJECT_ROOT'])
    root = project if kind == 'project' else project.parent
    return str(root / relative)
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import nibabel as nib
import numpy as np
import pandas as pd
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.size'] = 6.4
plt.rcParams['axes.titlesize'] = 7.0
plt.rcParams['axes.labelsize'] = 6.7
plt.rcParams['axes.linewidth'] = 0.65
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['legend.frameon'] = False
plt.rcParams['xtick.labelsize'] = 6.0
plt.rcParams['ytick.labelsize'] = 6.0
plt.rcParams['xtick.major.width'] = 0.55
plt.rcParams['ytick.major.width'] = 0.55
plt.rcParams['xtick.major.size'] = 2.5
plt.rcParams['ytick.major.size'] = 2.5
HERE = Path(__file__).resolve().parent
ANALYSIS = HERE / 'lesion_matched_clinical_mediation'
TABLES = ANALYSIS / 'tables'
FIGURES = ANALYSIS / 'figures'
SOURCE = ANALYSIS / 'figure_source_data'
MASKS = HERE / 'mni_registration' / 'masks'
TEMPLATE = HERE / 'resources' / 'MNI152_T1_2mm_brain.nii.gz'
WIDTH_MM = 183.0
HEIGHT_MM = 170.0
FIGSIZE = (WIDTH_MM / 25.4, HEIGHT_MM / 25.4)
SLICE_Z_MM = (-30, -14, 2, 18)
TIGHT_CALIPER = 1.25
RNG = np.random.default_rng(20260803)
CONTROL = '#484878'
CONTROL_SOFT = '#B4C0E4'
TMS = '#D24B40'
TMS_SOFT = '#F1C4BE'
NEUTRAL = '#8A8A8A'
NEUTRAL_LIGHT = '#D8D8D8'
INK = '#272727'
MATCH_ALL = '#5B8FD6'
MATCH_TIGHT = '#D24B40'
PALE_BAND = '#F5F5F5'
FEATURE_LABELS = {'log1p_lesion_volume': 'Lesion volume', 'mask_lateralization_index': 'Laterality index', 'log1p_brainstem_fraction_pct': 'Brainstem', 'log1p_deep_subcortical_gray_fraction_pct': 'Deep grey matter', 'log1p_cerebral_white_matter_fraction_pct': 'Cerebral white matter', 'log1p_cerebral_cortex_fraction_pct': 'Cerebral cortex', 'absolute_centroid_x_mm': 'Centroid |x|', 'centroid_y_mm': 'Centroid y', 'centroid_z_mm': 'Centroid z', 'maximum_extent_mm': 'Maximum extent', 'log1p_xtract_motor_projection_ml': 'Corticospinal projection', 'log1p_xtract_thalamic_sensory_radiations_ml': 'Thalamic radiations', 'log1p_xtract_middle_cerebellar_peduncle_ml': 'Cerebellar peduncle', 'age': 'Age', 'sex_male': 'Sex', 'esrs': 'ESRS'}
FEATURE_ORDER = ['log1p_lesion_volume', 'mask_lateralization_index', 'log1p_brainstem_fraction_pct', 'log1p_deep_subcortical_gray_fraction_pct', 'log1p_cerebral_white_matter_fraction_pct', 'log1p_cerebral_cortex_fraction_pct', 'absolute_centroid_x_mm', 'centroid_y_mm', 'centroid_z_mm', 'maximum_extent_mm', 'log1p_xtract_motor_projection_ml', 'log1p_xtract_thalamic_sensory_radiations_ml', 'log1p_xtract_middle_cerebellar_peduncle_ml', 'age', 'sex_male', 'esrs']

def ensure_dirs() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)

def read_inputs() -> dict[str, pd.DataFrame]:
    return {'pairs': pd.read_csv(TABLES / 'optimal_lesion_matched_pairs.csv', dtype=str), 'balance': pd.read_csv(TABLES / 'matching_balance.csv'), 'exact': pd.read_csv(TABLES / 'exact_basal_ganglia_periventricular_subjects.csv', dtype={'subject_id': str}), 'exact_results': pd.read_csv(TABLES / 'exact_basal_ganglia_periventricular_clinical_results.csv'), 'matched_results': pd.read_csv(TABLES / 'matched_clinical_results.csv'), 'cohort': pd.read_csv(TABLES / 'analysis_cohort_38.csv', dtype={'subject_id': str})}

def lesion_path(group: str, subject_id: str) -> Path:
    path = MASKS / group / 'before' / f'{subject_id}_lesion_mni2mm.nii.gz'
    if not path.exists():
        raise FileNotFoundError(path)
    return path

def build_overlap_maps(pairs: pd.DataFrame) -> tuple[nib.Nifti1Image, nib.Nifti1Image]:
    control_ids = pairs['control_subject_id'].astype(str).tolist()
    tms_ids = pairs['tms_subject_id'].astype(str).tolist()
    reference = nib.load(lesion_path('Control', control_ids[0]))

    def proportion(group: str, subject_ids: list[str]) -> np.ndarray:
        accum = np.zeros(reference.shape, dtype=np.float32)
        for subject_id in subject_ids:
            image = nib.load(lesion_path(group, subject_id))
            if image.shape != reference.shape or not np.allclose(image.affine, reference.affine):
                raise RuntimeError(f'MNI-grid mismatch for {group} {subject_id}')
            accum += image.get_fdata(dtype=np.float32) > 0.5
        return accum / len(subject_ids)
    control = nib.Nifti1Image(proportion('Control', control_ids), reference.affine, reference.header)
    tms = nib.Nifti1Image(proportion('TMS', tms_ids), reference.affine, reference.header)
    nib.save(control, SOURCE / 'matched_overlap_control_proportion.nii.gz')
    nib.save(tms, SOURCE / 'matched_overlap_tms_proportion.nii.gz')
    return (control, tms)

def transparent_cmap(color: str) -> LinearSegmentedColormap:
    rgb = tuple((int(color[index:index + 2], 16) / 255 for index in (1, 3, 5)))
    return LinearSegmentedColormap.from_list('transparent_signal', [(rgb[0], rgb[1], rgb[2], 0.0), (rgb[0], rgb[1], rgb[2], 0.96)])

def slice_index(image: nib.Nifti1Image, z_mm: float) -> int:
    voxel = nib.affines.apply_affine(np.linalg.inv(image.affine), [0.0, 0.0, z_mm])
    return int(np.clip(round(float(voxel[2])), 0, image.shape[2] - 1))

def prepare_anatomy() -> tuple[nib.Nifti1Image, np.ndarray]:
    image = nib.load(TEMPLATE)
    data = image.get_fdata(dtype=np.float32)
    nonzero = data[data > 0]
    (low, high) = np.percentile(nonzero, [1, 99.5])
    data = np.clip((data - low) / (high - low), 0, 1)
    return (image, data)

def plot_overlap_plate(fig: plt.Figure, spec, control_image: nib.Nifti1Image, tms_image: nib.Nifti1Image) -> list[plt.Axes]:
    (anatomy_image, anatomy) = prepare_anatomy()
    control = control_image.get_fdata(dtype=np.float32)
    tms = tms_image.get_fdata(dtype=np.float32)
    vmax = max(float(control.max()), float(tms.max()), 1 / 11)
    inner = spec.subgridspec(2, 4, hspace=0.035, wspace=0.025)
    axes = []
    for (row, (label, overlay, color)) in enumerate((('Control', control, CONTROL), ('TMS', tms, TMS))):
        for (column, z_mm) in enumerate(SLICE_Z_MM):
            ax = fig.add_subplot(inner[row, column])
            index = slice_index(anatomy_image, z_mm)
            anatomical_slice = np.rot90(anatomy[:, :, index])
            overlay_slice = np.rot90(overlay[:, :, index])
            ax.set_facecolor('black')
            ax.imshow(anatomical_slice, cmap='gray', vmin=0, vmax=0.92, interpolation='nearest')
            masked = np.ma.masked_where(overlay_slice <= 0, overlay_slice)
            ax.imshow(masked, cmap=transparent_cmap(color), vmin=0, vmax=vmax, interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if column == 0:
                ax.text(-0.055, 0.5, f'{label}\n(n=11)', transform=ax.transAxes, color=color, ha='right', va='center', fontsize=6.2, fontweight='bold', clip_on=False)
            if row == 0:
                ax.text(0.5, 1.04, f'z={z_mm} mm', transform=ax.transAxes, ha='center', va='bottom', fontsize=5.8, color=INK)
            axes.append(ax)
    return axes

def plot_balance(ax: plt.Axes, balance: pd.DataFrame) -> None:
    pivot = balance.pivot(index='feature', columns='match_set', values='smd_after')
    before = balance.drop_duplicates('feature').set_index('feature')['smd_before']
    y = np.arange(len(FEATURE_ORDER))[::-1]
    ax.axvspan(0, 0.2, color='#EDF4FB', zorder=0)
    ax.axvline(0.2, color=NEUTRAL, lw=0.8, ls='--', zorder=1)
    for separator in (2.5, 8.5, 11.5):
        ax.axhline(separator, color='#E7E7E7', lw=0.55, zorder=0)
    ax.scatter(before.reindex(FEATURE_ORDER).abs(), y + 0.18, color=NEUTRAL, marker='x', s=16, linewidth=0.8, label='Before', zorder=4)
    ax.scatter(pivot['matched_all'].reindex(FEATURE_ORDER).abs(), y, facecolor='white', edgecolor=MATCH_ALL, marker='o', s=19, linewidth=0.9, label='11 pairs', zorder=5)
    ax.scatter(pivot['matched_tight'].reindex(FEATURE_ORDER).abs(), y - 0.18, facecolor=MATCH_TIGHT, edgecolor=MATCH_TIGHT, marker='D', s=13, linewidth=0.6, label='9 tight pairs', zorder=6)
    ax.set_yticks(y)
    ax.set_yticklabels([FEATURE_LABELS[feature] for feature in FEATURE_ORDER])
    ax.set_xlabel('Absolute standardized mean difference')
    ax.set_xlim(-0.02, max(1.0, float(before.abs().max()) + 0.08))
    ax.set_ylim(-0.7, len(y) - 0.3)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.015), ncol=3, fontsize=5.6, handletextpad=0.35, columnspacing=0.8)
    ax.text(0.205, len(y) - 0.5, 'target', fontsize=5.2, color=NEUTRAL, ha='left', va='top')
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)

def improvements(frame: pd.DataFrame, outcome: str) -> np.ndarray:
    if outcome in {'NIHSS', 'HAMD'}:
        return frame[f'{outcome}_pre'].to_numpy(float) - frame[f'{outcome}_post'].to_numpy(float)
    return frame[f'{outcome}_post'].to_numpy(float) - frame[f'{outcome}_pre'].to_numpy(float)

def plot_exact_improvement(ax: plt.Axes, exact: pd.DataFrame, exact_results: pd.DataFrame, outcome: str) -> None:
    result = exact_results.loc[exact_results.outcome == outcome].iloc[0]
    for (position, (group, color)) in enumerate((('Control', CONTROL), ('TMS', TMS))):
        subset = exact.loc[exact.group == group]
        values = improvements(subset, outcome)
        jitter = RNG.uniform(-0.065, 0.065, len(values))
        ax.scatter(np.full(len(values), position) + jitter, values, s=25, facecolor=color, edgecolor='white', linewidth=0.55, alpha=0.92, zorder=4)
        mean = float(np.mean(values))
        ax.plot([position - 0.16, position + 0.16], [mean, mean], color=INK, lw=1.2, zorder=5)
    ax.axhline(0, color=NEUTRAL_LIGHT, lw=0.75, zorder=0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Control\n(n=5)', 'TMS\n(n=5)'])
    ax.set_xlim(-0.42, 1.42)
    ax.set_ylabel(f'{outcome} improvement (points)')
    ax.set_title(outcome, pad=3, fontweight='bold')
    ax.text(0.03, 0.985, f'Δ={result.improvement_difference_tms_minus_control:.1f}; q={result.permutation_fdr_q:.3f}', transform=ax.transAxes, ha='left', va='top', fontsize=5.5, color=INK)

def matched_ids(pairs: pd.DataFrame, tight: bool) -> list[str]:
    selected = pairs.loc[pairs.standardized_rms_distance.astype(float) <= TIGHT_CALIPER] if tight else pairs
    return selected.control_subject_id.astype(str).tolist() + selected.tms_subject_id.astype(str).tolist()

def standardised_effect_source(matched_results: pd.DataFrame, cohort: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (match_set, tight) in (('matched_all', False), ('matched_tight', True)):
        identifiers = matched_ids(pairs, tight)
        subset = cohort.loc[cohort.subject_id.astype(str).isin(identifiers)]
        for outcome in ('NIHSS', 'ADL', 'MMSE', 'HAMD', 'VFT'):
            result = matched_results.loc[(matched_results.match_set == match_set) & (matched_results.outcome == outcome)].iloc[0]
            scale = float(subset[f'{outcome}_pre'].std(ddof=1))
            rows.append({'match_set': match_set, 'outcome': outcome, 'n_pairs': int(result.n_pairs), 'standardized_adjusted_effect': result.pair_fe_baseline_adjusted_group_coefficient / scale, 'standardized_ci_low': result.pair_fe_boot_ci_low / scale, 'standardized_ci_high': result.pair_fe_boot_ci_high / scale, 'permutation_fdr_q': result.pair_fe_permutation_fdr_q, 'raw_adjusted_effect': result.pair_fe_baseline_adjusted_group_coefficient, 'raw_ci_low': result.pair_fe_boot_ci_low, 'raw_ci_high': result.pair_fe_boot_ci_high, 'baseline_sd': scale})
    return pd.DataFrame(rows)

def plot_effect_forest(ax: plt.Axes, effects: pd.DataFrame) -> None:
    outcomes = ['NIHSS', 'ADL', 'MMSE', 'HAMD', 'VFT']
    y = np.arange(len(outcomes))[::-1]
    offsets = {'matched_all': 0.13, 'matched_tight': -0.13}
    styles = {'matched_all': (MATCH_ALL, 'o', '11 pairs'), 'matched_tight': (MATCH_TIGHT, 'D', '9 tight pairs')}
    for (index, yi) in enumerate(y):
        if index % 2 == 0:
            ax.axhspan(yi - 0.42, yi + 0.42, color=PALE_BAND, zorder=0)
    for (match_set, (color, marker, label)) in styles.items():
        part = effects.loc[effects.match_set == match_set].set_index('outcome').reindex(outcomes)
        position = y + offsets[match_set]
        for (yi, estimate, low, high) in zip(position, part.standardized_adjusted_effect, part.standardized_ci_low, part.standardized_ci_high):
            ax.plot([low, high], [yi, yi], color=color, lw=1.05, zorder=3)
            ax.scatter(estimate, yi, marker=marker, s=22 if marker == 'o' else 17, facecolor='white' if marker == 'o' else color, edgecolor=color, linewidth=0.9, zorder=4)
    ax.axvline(0, color=NEUTRAL, ls='--', lw=0.8, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(outcomes)
    ax.set_ylim(-0.55, len(outcomes) - 0.45)
    low = float(effects.standardized_ci_low.min())
    high = float(effects.standardized_ci_high.max())
    margin = 0.1 * (high - low)
    ax.set_xlim(min(-0.1, low - margin), high + margin)
    ax.set_xlabel('Baseline-adjusted TMS–Control difference\n(baseline SD units; positive favours TMS)')
    handles = [Line2D([0], [0], marker=marker, color=color, markerfacecolor='white' if marker == 'o' else color, markeredgecolor=color, lw=1, label=label) for (color, marker, label) in styles.values()]
    ax.legend(handles=handles, loc='lower right', fontsize=5.7, handlelength=1.4, borderaxespad=0.2)

def panel_label(fig: plt.Figure, x: float, y: float, label: str, color: str=INK) -> None:
    fig.text(x, y, label, ha='left', va='top', fontsize=8.2, fontweight='bold', color=color)

def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix('.svg'), facecolor='white')
    fig.savefig(stem.with_suffix('.pdf'), facecolor='white')
    fig.savefig(stem.with_suffix('.tiff'), dpi=600, facecolor='white')
    fig.savefig(stem.with_suffix('.png'), dpi=300, facecolor='white')

def write_contract_and_legend(effects: pd.DataFrame) -> None:
    contract = f'# Figure contract\n\n- Core conclusion: Among patients with identical or quantitatively similar baseline lesions, the TMS group showed greater multidomain clinical improvement, while residual matching imbalance limits causal interpretation.\n- Figure archetype: Asymmetric mixed-modality clinical composite.\n- Target/output: Nature-family double-column figure; {WIDTH_MM:.0f} × {HEIGHT_MM:.0f} mm.\n- Backend: Python/matplotlib only.\n- Panel a: Fixed-coordinate axial MNI slices of lesion-overlap proportions in the 11 matched Control and TMS participants.\n- Panel b: Absolute standardized mean differences before matching, after all-pair matching and after tight matching.\n- Panel c: Individual NIHSS and ADL improvements in the exact text-defined basal-ganglia/periventricular subgroup (5 vs 5).\n- Panel d: Standardized baseline-adjusted group differences across five outcomes in 11-pair and 9-pair analyses.\n- Hero evidence: Patient-level primary-outcome improvements and adjusted effect estimates.\n- Validation evidence: Agreement between all-pair and tight-pair estimates.\n- Reviewer-risk control: The balance panel exposes residual imbalance rather than implying exchangeability.\n- Image integrity: Fixed z coordinates {list(SLICE_Z_MM)} mm; nearest display interpolation; no smoothing; overlap is the fraction of 11 matched participants; global percentile scaling of the anatomical template only.\n'
    (ANALYSIS / 'FIGURE_CONTRACT.md').write_text(contract, encoding='utf-8')
    legend = '# Figure legend\n\n**Lesion-matched comparison of multidomain clinical improvement.** **a,** Baseline lesion-overlap proportions for the 11 optimally matched Control participants and 11 TMS participants, displayed at four prespecified axial MNI coordinates. Warmer or darker colour within each group denotes overlap in a larger fraction of participants; masks were not smoothed. **b,** Covariate balance before matching, after optimal matching of all available common-support strata (11 pairs), and after applying the prespecified tight distance threshold (9 pairs). Points show absolute standardized mean differences; the dashed line marks 0.20. **c,** Individual NIHSS and ADL improvements among patients sharing the exact clinical location description “basal ganglia and periventricular region” (Control, n=5; TMS, n=5). Horizontal lines denote group means; Δ is the TMS-minus-Control improvement difference and q is the Benjamini–Hochberg-adjusted exact permutation value from the baseline-adjusted model. **d,** Baseline-adjusted TMS-minus-Control differences in the lesion-matched cohorts. Estimates and bootstrap 95% confidence intervals are expressed in baseline-score standard-deviation units; positive values favour TMS. Matching was exact on coarse lesion class and clinical lesion side and minimized a weighted distance across lesion volume, laterality, anatomical compartment burden, spatial coordinates, selected normative tract burdens and low-weight demographic covariates. Residual imbalance remained after matching, so estimates represent adjusted observational group differences rather than unconfounded causal treatment effects.\n'
    (ANALYSIS / 'FIGURE_LEGEND.md').write_text(legend, encoding='utf-8')
    statistics = {'panel_a': {'n': '11 Control and 11 TMS participants in the all-pair matched sample', 'summary': 'voxelwise participant overlap proportion', 'smoothing': 'none', 'slice_coordinates_mni_z_mm': list(SLICE_Z_MM)}, 'panel_b': {'metric': 'absolute standardized mean difference', 'target_threshold': 0.2, 'matching_sets': ['before', '11 pairs', '9 tight pairs']}, 'panel_c': {'n': '5 Control and 5 TMS', 'center': 'group mean', 'test': 'exact label permutation of baseline-adjusted group coefficient', 'multiple_testing': 'Benjamini-Hochberg across five outcomes'}, 'panel_d': {'n': '11 pairs and 9 tight pairs', 'estimate': 'pair-fixed-effect baseline-adjusted group coefficient', 'interval': '1000 matched-pair bootstrap resamples', 'standardization': 'division by the matched-sample baseline-score SD', 'test': 'within-pair label permutation; BH FDR across five outcomes'}, 'effect_source_rows': int(len(effects))}
    (SOURCE / 'statistics_definition.json').write_text(json.dumps(statistics, ensure_ascii=False, indent=2), encoding='utf-8')

def main() -> None:
    ensure_dirs()
    data = read_inputs()
    data['pairs']['standardized_rms_distance'] = data['pairs']['standardized_rms_distance'].astype(float)
    (control_overlap, tms_overlap) = build_overlap_maps(data['pairs'])
    effects = standardised_effect_source(data['matched_results'], data['cohort'], data['pairs'])
    exact_source_rows = []
    for outcome in ('NIHSS', 'ADL'):
        frame = data['exact'][['subject_id', 'group']].copy()
        frame['outcome'] = outcome
        frame['improvement'] = improvements(data['exact'], outcome)
        exact_source_rows.append(frame)
    exact_source = pd.concat(exact_source_rows, ignore_index=True)
    exact_source.to_csv(SOURCE / 'panel_c_exact_location_improvements.csv', index=False)
    data['balance'].to_csv(SOURCE / 'panel_b_matching_balance.csv', index=False)
    effects.to_csv(SOURCE / 'panel_d_standardized_effects.csv', index=False)
    data['pairs'].to_csv(SOURCE / 'panel_a_matched_pair_index.csv', index=False)
    fig = plt.figure(figsize=FIGSIZE, facecolor='white')
    grid = fig.add_gridspec(3, 12, height_ratios=[1.42, 1.03, 1.48], left=0.14, right=0.985, bottom=0.075, top=0.955, hspace=0.52, wspace=0.78)
    overlap_axes = plot_overlap_plate(fig, grid[0, :], control_overlap, tms_overlap)
    ax_balance = fig.add_subplot(grid[1:, :5])
    ax_nihss = fig.add_subplot(grid[1, 5:8])
    ax_adl = fig.add_subplot(grid[1, 8:11])
    ax_forest = fig.add_subplot(grid[2, 5:12])
    plot_balance(ax_balance, data['balance'])
    plot_exact_improvement(ax_nihss, data['exact'], data['exact_results'], 'NIHSS')
    plot_exact_improvement(ax_adl, data['exact'], data['exact_results'], 'ADL')
    plot_effect_forest(ax_forest, effects)
    fig.text(0.14, 0.974, 'Matched lesion topography', ha='left', va='bottom', fontsize=7.1, fontweight='bold')
    fig.text(0.985, 0.974, 'Colour intensity indicates participant overlap', ha='right', va='bottom', fontsize=5.6, color=NEUTRAL)
    fig.text(0.14, 0.643, 'Covariate balance', ha='left', va='bottom', fontsize=7.1, fontweight='bold')
    fig.text(0.478, 0.643, 'Exact basal ganglia/periventricular subgroup', ha='left', va='bottom', fontsize=7.1, fontweight='bold')
    fig.text(0.478, 0.35, 'Adjusted multidomain effects', ha='left', va='bottom', fontsize=7.1, fontweight='bold')
    panel_label(fig, 0.018, 0.965, 'a')
    panel_label(fig, 0.018, 0.616, 'b')
    panel_label(fig, 0.438, 0.616, 'c')
    panel_label(fig, 0.438, 0.331, 'd')
    stem = FIGURES / 'lesion_matched_clinical_improvement_nature'
    save_figure(fig, stem)
    plt.close(fig)
    write_contract_and_legend(effects)
    print(f'Figure: {stem}')
    print(f'Size: {WIDTH_MM:.0f} x {HEIGHT_MM:.0f} mm')
    print(f'Source data: {SOURCE}')
if __name__ == '__main__':
    main()
