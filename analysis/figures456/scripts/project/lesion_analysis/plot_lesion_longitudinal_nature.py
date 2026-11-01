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
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
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
TABLES = HERE / 'results_abcd' / 'tables'
MASKS = HERE / 'mni_registration' / 'masks'
TEMPLATE = HERE / 'resources' / 'MNI152_T1_2mm_brain.nii.gz'
OUT = HERE / 'lesion_longitudinal_figure'
FIGURES = OUT / 'figures'
SOURCE = OUT / 'source_data'
WIDTH_MM = 183.0
HEIGHT_MM = 155.0
SLICE_Z_MM = (-48, -32, -16, 0, 16, 32, 48)
RNG = np.random.default_rng(20260803)
CONTROL = '#484878'
CONTROL_PRE = '#B4C0E4'
TMS = '#D24B40'
TMS_PRE = '#F1C4BE'
PAIR_GREY = '#9A9A9A'
INK = '#272727'
NEUTRAL = '#777777'
NEUTRAL_LIGHT = '#D8D8D8'
OVERLAP_CMAP = LinearSegmentedColormap.from_list('lesion_overlap', ['#FFF2CC', '#E69F00', '#B2182B'])
SPATIAL_ROWS = (('TMS', 'before', 'TMS — Pre', TMS), ('TMS', 'after', 'TMS — Post', TMS), ('Control', 'before', 'Control — Pre', CONTROL), ('Control', 'after', 'Control — Post', CONTROL))

def ensure_dirs() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)

def bh_adjust(p_values: list[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0, 1)
    return result

def read_longitudinal_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_rows = pd.read_csv(TABLES / 'G_longitudinal_lesion_metrics.csv', dtype={'subject_id': str})
    comparisons = pd.read_csv(TABLES / 'G_longitudinal_group_comparisons.csv')
    if len(all_rows) != 38:
        raise RuntimeError(f'Expected 38 participants, found {len(all_rows)}')
    included = all_rows.loc[all_rows['primary_longitudinal_included'].eq(1)].copy()
    counts = included.groupby('group').size().to_dict()
    if counts != {'Control': 15, 'TMS': 21}:
        raise RuntimeError(f'Unexpected paired longitudinal cohort: {counts}')
    if included[['baseline_volume_ml', 'post_volume_ml']].isna().any().any():
        raise RuntimeError('Included paired cohort contains missing lesion volumes')
    if (included[['baseline_volume_ml', 'post_volume_ml']] <= 0).any().any():
        raise RuntimeError('Included paired cohort contains empty lesion masks')
    audit = all_rows[['subject_id', 'group', 'has_post_mask', 'post_mask_nonempty', 'primary_longitudinal_included', 'baseline_volume_ml', 'post_volume_ml']].copy()
    audit['exclusion_reason'] = np.select([audit['has_post_mask'].eq(0), audit['has_post_mask'].eq(1) & audit['post_mask_nonempty'].eq(0)], ['missing follow-up lesion mask', 'empty follow-up lesion mask; QC exclusion'], default='included')
    return (included, comparisons, audit)

def mask_path(group: str, time: str, subject_id: str) -> Path:
    path = MASKS / group / time / f'{subject_id}_lesion_mni2mm.nii.gz'
    if not path.exists():
        raise FileNotFoundError(path)
    return path

def validate_and_build_overlap_maps(included: pd.DataFrame) -> tuple[dict[tuple[str, str], nib.Nifti1Image], pd.DataFrame]:
    reference = nib.load(TEMPLATE)
    voxel_ml = abs(float(np.linalg.det(reference.affine[:3, :3]))) / 1000.0
    accumulators = {(group, time): np.zeros(reference.shape, dtype=np.float32) for (group, time, _, _) in SPATIAL_ROWS}
    long_rows: list[dict[str, object]] = []
    for row in included.sort_values(['group', 'subject_id']).itertuples():
        derived: dict[str, float] = {}
        for (time, column) in (('before', 'baseline_volume_ml'), ('after', 'post_volume_ml')):
            image = nib.load(str(mask_path(row.group, time, row.subject_id)))
            if image.shape != reference.shape or not np.allclose(image.affine, reference.affine):
                raise RuntimeError(f'MNI-grid mismatch for {row.group} {row.subject_id} {time}')
            mask = np.asarray(image.dataobj) > 0.5
            if not np.any(mask):
                raise RuntimeError(f'Empty included mask for {row.group} {row.subject_id} {time}')
            volume = float(mask.sum() * voxel_ml)
            expected = float(getattr(row, column))
            if not np.isclose(volume, expected, rtol=0, atol=1e-06):
                raise RuntimeError(f'Volume mismatch for {row.subject_id} {time}: mask={volume}, table={expected}')
            derived[time] = volume
            accumulators[row.group, time] += mask
        long_rows.extend([{'subject_id': row.subject_id, 'group': row.group, 'time': 'Pre', 'lesion_volume_ml': derived['before']}, {'subject_id': row.subject_id, 'group': row.group, 'time': 'Post', 'lesion_volume_ml': derived['after']}])
    maps: dict[tuple[str, str], nib.Nifti1Image] = {}
    for (group, time, _, _) in SPATIAL_ROWS:
        n = int((included['group'] == group).sum())
        data = accumulators[group, time] / n
        image = nib.Nifti1Image(data, reference.affine, reference.header)
        maps[group, time] = image
        nib.save(image, SOURCE / f"overlap_{group.lower()}_{('pre' if time == 'before' else 'post')}_proportion.nii.gz")
    return (maps, pd.DataFrame(long_rows))

def calculate_within_group_statistics(included: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    p_values: list[float] = []
    for group in ('Control', 'TMS'):
        subset = included.loc[included['group'] == group].sort_values('subject_id')
        pre = subset['baseline_volume_ml'].to_numpy(float)
        post = subset['post_volume_ml'].to_numpy(float)
        test = wilcoxon(pre, post, alternative='two-sided', zero_method='wilcox', correction=False, method='approx')
        p_values.append(float(test.pvalue))
        rows.append({'group': group, 'n': len(subset), 'pre_median_ml': float(np.median(pre)), 'pre_q1_ml': float(np.quantile(pre, 0.25)), 'pre_q3_ml': float(np.quantile(pre, 0.75)), 'post_median_ml': float(np.median(post)), 'post_q1_ml': float(np.quantile(post, 0.25)), 'post_q3_ml': float(np.quantile(post, 0.75)), 'median_change_ml': float(np.median(post - pre)), 'wilcoxon_statistic': float(test.statistic), 'wilcoxon_p': float(test.pvalue)})
    q_values = bh_adjust(p_values)
    for (row, q_value) in zip(rows, q_values):
        row['wilcoxon_fdr_q'] = float(q_value)
    return pd.DataFrame(rows)

def prepare_anatomy() -> tuple[nib.Nifti1Image, np.ndarray]:
    image = nib.load(TEMPLATE)
    data = image.get_fdata(dtype=np.float32)
    nonzero = data[data > 0]
    (low, high) = np.percentile(nonzero, [1, 99.5])
    return (image, np.clip((data - low) / (high - low), 0, 1))

def slice_index(image: nib.Nifti1Image, z_mm: float) -> int:
    voxel = nib.affines.apply_affine(np.linalg.inv(image.affine), [0.0, 0.0, z_mm])
    return int(np.clip(round(float(voxel[2])), 0, image.shape[2] - 1))

def plot_spatial_plate(fig: plt.Figure, spec, maps: dict[tuple[str, str], nib.Nifti1Image], counts: dict[str, int]) -> float:
    (anatomy_image, anatomy) = prepare_anatomy()
    arrays = {key: image.get_fdata(dtype=np.float32) for (key, image) in maps.items()}
    vmax = max((float(array.max()) for array in arrays.values()))
    inner = spec.subgridspec(5, len(SLICE_Z_MM), height_ratios=[1, 1, 1, 1, 0.12], hspace=0.045, wspace=0.025)
    for (row_index, (group, time, label, label_color)) in enumerate(SPATIAL_ROWS):
        overlay = arrays[group, time]
        for (column_index, z_mm) in enumerate(SLICE_Z_MM):
            ax = fig.add_subplot(inner[row_index, column_index])
            index = slice_index(anatomy_image, z_mm)
            anatomical_slice = np.rot90(anatomy[:, :, index])
            overlay_slice = np.rot90(overlay[:, :, index])
            ax.set_facecolor('black')
            ax.imshow(anatomical_slice, cmap='gray', vmin=0, vmax=0.92, interpolation='nearest')
            ax.imshow(np.ma.masked_where(overlay_slice <= 0, overlay_slice), cmap=OVERLAP_CMAP, norm=Normalize(0, vmax), interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if column_index == 0:
                ax.text(-0.055, 0.5, f'{label}\n(n={counts[group]})', transform=ax.transAxes, ha='right', va='center', fontsize=6.0, fontweight='bold', color=label_color, clip_on=False)
            if row_index == 0:
                ax.text(0.5, 1.035, f'z={z_mm} mm', transform=ax.transAxes, ha='center', va='bottom', fontsize=5.6, color=INK)
    colorbar_ax = fig.add_subplot(inner[4, 4:7])
    colorbar = fig.colorbar(ScalarMappable(norm=Normalize(0, vmax), cmap=OVERLAP_CMAP), cax=colorbar_ax, orientation='horizontal')
    ticks = [0, vmax / 2, vmax]
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels([f'{100 * value:.0f}' for value in ticks])
    colorbar.ax.tick_params(labelsize=5.2, width=0.5, length=2, pad=1)
    colorbar.outline.set_linewidth(0.5)
    colorbar.set_label('Participants with lesion at voxel (%)', fontsize=5.5, labelpad=1.5)
    return vmax

def plot_volume_panel(ax: plt.Axes, included: pd.DataFrame, within_stats: pd.DataFrame, between_q: float) -> None:
    positions = {('Control', 'Pre'): 0.0, ('Control', 'Post'): 1.0, ('TMS', 'Pre'): 3.0, ('TMS', 'Post'): 4.0}
    colors = {('Control', 'Pre'): CONTROL_PRE, ('Control', 'Post'): CONTROL, ('TMS', 'Pre'): TMS_PRE, ('TMS', 'Post'): TMS}
    values_by_position: dict[tuple[str, str], np.ndarray] = {}
    for group in ('Control', 'TMS'):
        subset = included.loc[included['group'] == group].sort_values('subject_id')
        pre = subset['baseline_volume_ml'].to_numpy(float)
        post = subset['post_volume_ml'].to_numpy(float)
        values_by_position[group, 'Pre'] = pre
        values_by_position[group, 'Post'] = post
        for (pre_value, post_value) in zip(pre, post):
            jitter = float(RNG.uniform(-0.055, 0.055))
            ax.plot([positions[group, 'Pre'] + jitter, positions[group, 'Post'] + jitter], [pre_value, post_value], color=PAIR_GREY, lw=0.55, alpha=0.55, zorder=1)
            ax.scatter(positions[group, 'Pre'] + jitter, pre_value, s=14, facecolor=colors[group, 'Pre'], edgecolor='white', linewidth=0.45, zorder=4)
            ax.scatter(positions[group, 'Post'] + jitter, post_value, s=14, facecolor=colors[group, 'Post'], edgecolor='white', linewidth=0.45, zorder=4)
    ordered_keys = [('Control', 'Pre'), ('Control', 'Post'), ('TMS', 'Pre'), ('TMS', 'Post')]
    boxplot = ax.boxplot([values_by_position[key] for key in ordered_keys], positions=[positions[key] for key in ordered_keys], widths=0.48, patch_artist=True, showfliers=False, whis=1.5, medianprops={'color': INK, 'linewidth': 1.15}, whiskerprops={'color': NEUTRAL, 'linewidth': 0.75}, capprops={'color': NEUTRAL, 'linewidth': 0.75}, boxprops={'color': INK, 'linewidth': 0.65}, zorder=2)
    for (box, key) in zip(boxplot['boxes'], ordered_keys):
        box.set_facecolor(colors[key])
        box.set_alpha(0.68)
    ax.set_yscale('log')
    ax.set_ylim(0.12, 270)
    major_ticks = [0.2, 1, 5, 20, 100]
    ax.yaxis.set_major_locator(FixedLocator(major_ticks))
    ax.yaxis.set_major_formatter(FixedFormatter(['0.2', '1', '5', '20', '100']))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel('Lesion volume (ml; log scale)')
    ax.set_xticks([0, 1, 3, 4])
    ax.set_xticklabels(['Pre', 'Post', 'Pre', 'Post'])
    ax.set_xlim(-0.65, 4.65)
    ax.axvline(2.0, color='#E7E7E7', lw=0.7, zorder=0)
    ax.grid(False)
    group_specs = (('Control', 0.5, CONTROL), ('TMS', 3.5, TMS))
    for (group, center, color) in group_specs:
        n = int((included['group'] == group).sum())
        ax.text(center, -0.18, f'{group} (n={n})', transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=6.2, fontweight='bold', color=color, clip_on=False)
        stats_row = within_stats.loc[within_stats['group'] == group].iloc[0]
        maximum = max(float(values_by_position[group, 'Pre'].max()), float(values_by_position[group, 'Post'].max()))
        bracket_y = maximum * 1.12
        cap_y = bracket_y / 1.08
        (left, right) = (center - 0.5, center + 0.5)
        ax.plot([left, left, right, right], [cap_y, bracket_y, bracket_y, cap_y], color=INK, lw=0.65, clip_on=False, zorder=5)
        ax.text(center, bracket_y * 1.08, f'q={stats_row.wilcoxon_fdr_q:.3f}', ha='center', va='bottom', fontsize=5.8, color=INK)
    ax.text(0.5, 1.035, f'Between-group absolute volume change: q={between_q:.3f}', transform=ax.transAxes, ha='center', va='bottom', fontsize=5.7, color=NEUTRAL)

def panel_label(fig: plt.Figure, x: float, y: float, label: str) -> None:
    fig.text(x, y, label, ha='left', va='top', fontsize=8.2, fontweight='bold', color=INK)

def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix('.svg'), facecolor='white')
    fig.savefig(stem.with_suffix('.pdf'), facecolor='white')
    fig.savefig(stem.with_suffix('.tiff'), dpi=600, facecolor='white')
    fig.savefig(stem.with_suffix('.png'), dpi=300, facecolor='white')

def write_documentation(included: pd.DataFrame, within_stats: pd.DataFrame, between_row: pd.Series, overlap_vmax: float) -> None:
    counts = included.groupby('group').size().to_dict()
    contract = f"# Figure contract\n\n- Core conclusion: Paired longitudinal lesion masks show the spatial distribution and within-patient volume evolution of lesions in the TMS and Control groups; both groups decrease in volume, without evidence of a between-group difference in absolute change.\n- Figure archetype: Image plate + quantification.\n- Target/output: Nature-family double-column figure; {WIDTH_MM:.0f} x {HEIGHT_MM:.0f} mm.\n- Backend: Python/matplotlib only.\n- Panel a: Four directly labelled rows of overlap proportions in the paired longitudinal cohort: TMS pre, TMS post, Control pre and Control post.\n- Panel b: Raw lesion volumes, box plots and within-patient connections for Control (n={counts['Control']}) and TMS (n={counts['TMS']}).\n- Hero evidence: Spatial overlap at seven prespecified axial MNI coordinates.\n- Validation evidence: Individual paired lesion-volume trajectories.\n- Statistics: Two-sided paired Wilcoxon signed-rank tests with BH correction across the two within-group comparisons; between-group absolute volume change uses the existing two-sided Mann-Whitney comparison with BH correction across six longitudinal lesion metrics.\n- Image integrity: No smoothing; nearest display interpolation; common overlap-proportion colour scale with maximum {100 * overlap_vmax:.1f}%; same participants at pre and post within each group.\n- Reviewer risk: One Control participant lacks a follow-up lesion mask and one TMS participant has an empty follow-up native lesion mask with a prespecified QC exclusion; neither participant contributes to either time point in the paired longitudinal figure.\n"
    (OUT / 'FIGURE_CONTRACT.md').write_text(contract, encoding='utf-8')
    control_stats = within_stats.loc[within_stats.group == 'Control'].iloc[0]
    tms_stats = within_stats.loc[within_stats.group == 'TMS'].iloc[0]
    legend = f"# Figure legend\n\n**Longitudinal lesion topography and volume.** **a,** Lesion-overlap proportions at seven prespecified axial MNI coordinates for the paired longitudinal cohort, displayed in the order TMS pre, TMS post, Control pre and Control post. Each pre/post row within a group contains the same participants (TMS, n={counts['TMS']}; Control, n={counts['Control']}). Colour denotes the percentage of participants whose lesion includes the voxel. Registered masks were not smoothed and were displayed using nearest-neighbour interpolation. **b,** Individual pre-to-post lesion-volume trajectories. Grey lines connect repeated measurements from the same participant; points show raw volumes and boxes show the median and interquartile range, with whiskers extending to 1.5 times the interquartile range. The y axis is logarithmic and labelled in millilitres. Within-group q values are from two-sided paired Wilcoxon signed-rank tests with Benjamini-Hochberg correction across the Control and TMS comparisons (Control q={control_stats.wilcoxon_fdr_q:.3f}; TMS q={tms_stats.wilcoxon_fdr_q:.3f}). The between-group q value compares absolute volume change using a two-sided Mann-Whitney test with Benjamini-Hochberg correction across six longitudinal lesion metrics (q={float(between_row.fdr_q):.3f}). One Control participant without a follow-up mask and one TMS participant with an empty follow-up native lesion mask and a prespecified longitudinal-QC exclusion were excluded from both time points. Volume declined within each group, but absolute change did not differ between groups.\n"
    (OUT / 'FIGURE_LEGEND.md').write_text(legend, encoding='utf-8')
    statistics = {'paired_cohort': {'Control': counts['Control'], 'TMS': counts['TMS']}, 'panel_a': {'summary': 'voxelwise participant overlap proportion', 'slice_coordinates_mni_z_mm': list(SLICE_Z_MM), 'smoothing': 'none', 'display_interpolation': 'nearest', 'common_overlap_vmax_proportion': overlap_vmax}, 'panel_b': {'center': 'median', 'box': 'interquartile range', 'whiskers': '1.5 times interquartile range', 'raw_points': True, 'paired_lines': True, 'within_group_test': 'two-sided paired Wilcoxon signed-rank, normal approximation; BH across two groups', 'between_group_test': 'two-sided Mann-Whitney on absolute volume change; BH across six longitudinal lesion metrics', 'between_group_absolute_change_fdr_q': float(between_row.fdr_q)}}
    (SOURCE / 'statistics_definition.json').write_text(json.dumps(statistics, ensure_ascii=False, indent=2), encoding='utf-8')

def main() -> None:
    ensure_dirs()
    (included, comparisons, audit) = read_longitudinal_data()
    (maps, volume_long) = validate_and_build_overlap_maps(included)
    within_stats = calculate_within_group_statistics(included)
    between_row = comparisons.loc[comparisons['metric'] == 'absolute_volume_change_ml'].iloc[0]
    audit.to_csv(SOURCE / 'longitudinal_cohort_audit.csv', index=False)
    volume_long.to_csv(SOURCE / 'panel_b_paired_lesion_volumes.csv', index=False)
    within_stats.to_csv(SOURCE / 'panel_b_within_group_statistics.csv', index=False)
    comparisons.to_csv(SOURCE / 'longitudinal_group_comparisons.csv', index=False)
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), facecolor='white')
    grid = fig.add_gridspec(2, 12, height_ratios=[2.18, 1.0], left=0.105, right=0.985, bottom=0.105, top=0.95, hspace=0.3, wspace=0.45)
    counts = included.groupby('group').size().astype(int).to_dict()
    overlap_vmax = plot_spatial_plate(fig, grid[0, :], maps, counts)
    ax_volume = fig.add_subplot(grid[1, 2:10])
    plot_volume_panel(ax_volume, included, within_stats, float(between_row.fdr_q))
    fig.text(0.105, 0.976, 'Paired longitudinal lesion topography', ha='left', va='bottom', fontsize=7.1, fontweight='bold', color=INK)
    volume_box = ax_volume.get_position()
    fig.text(volume_box.x0, volume_box.y1 + 0.035, 'Paired lesion-volume change', ha='left', va='bottom', fontsize=7.1, fontweight='bold', color=INK)
    panel_label(fig, 0.018, 0.965, 'a')
    panel_label(fig, volume_box.x0 - 0.05, volume_box.y1 + 0.042, 'b')
    stem = FIGURES / 'lesion_longitudinal_topography_volume_nature'
    save_figure(fig, stem)
    plt.close(fig)
    write_documentation(included, within_stats, between_row, overlap_vmax)
    print(f'Figure: {stem}')
    print(f'Size: {WIDTH_MM:.0f} x {HEIGHT_MM:.0f} mm')
    print(f"Paired cohort: Control n={counts['Control']}; TMS n={counts['TMS']}")
    print(f'Maximum overlap: {100 * overlap_vmax:.1f}%')
if __name__ == '__main__':
    main()
