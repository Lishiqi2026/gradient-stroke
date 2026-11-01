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
from matplotlib.colors import Normalize
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import plot_lesion_longitudinal_nature as base
HERE = Path(__file__).resolve().parent
OUT = HERE / 'lesion_longitudinal_outlier_excluded_figure'
FIGURES = OUT / 'figures'
SOURCE = OUT / 'source_data'
EXCLUDED_SUBJECT = 'PRIVATE_SUBJECT_040'

def ensure_dirs() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)

def save_bundle(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix('.svg'), facecolor='white')
    fig.savefig(stem.with_suffix('.pdf'), facecolor='white')
    fig.savefig(stem.with_suffix('.tiff'), dpi=600, facecolor='white')
    fig.savefig(stem.with_suffix('.png'), dpi=300, facecolor='white')
    plt.close(fig)

def select_sensitivity_cohort(included: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    excluded = included.loc[included['subject_id'] == EXCLUDED_SUBJECT].copy()
    if len(excluded) != 1:
        raise RuntimeError(f'Expected one excluded participant, found {len(excluded)}')
    row = excluded.iloc[0]
    if row['group'] != 'Control' or not np.isclose(row['baseline_volume_ml'], float('__PRIVATE_FIG456_BASELINE_VOLUME_ML__')):
        raise RuntimeError('The requested approximately 170-ml Control lesion was not identified')
    excluded['exclusion_reason'] = 'post hoc sensitivity exclusion requested for approximately 170-ml baseline lesion'
    selected = included.loc[included['subject_id'] != EXCLUDED_SUBJECT].copy()
    counts = selected.groupby('group').size().to_dict()
    if counts != {'Control': 14, 'TMS': 21}:
        raise RuntimeError(f'Unexpected sensitivity cohort: {counts}')
    return (selected, excluded)

def recompute_between_group_statistics(selected: pd.DataFrame) -> pd.DataFrame:
    metrics = (('absolute_volume_change_ml', 'Absolute volume change (ml)'), ('symmetric_volume_change_pct', 'Symmetric volume change (%)'), ('dice', 'Baseline-follow-up Dice'), ('baseline_persistence_pct', 'Baseline lesion persistence (%)'), ('post_new_fraction_pct', 'Follow-up new-lesion fraction (%)'), ('centroid_displacement_mm', 'Centroid displacement (mm)'))
    rows: list[dict[str, object]] = []
    p_values: list[float] = []
    control = selected.loc[selected['group'] == 'Control']
    tms = selected.loc[selected['group'] == 'TMS']
    for (metric, label) in metrics:
        control_values = control[metric].to_numpy(float)
        tms_values = tms[metric].to_numpy(float)
        test = mannwhitneyu(tms_values, control_values, alternative='two-sided')
        p_values.append(float(test.pvalue))
        pairwise = np.sign(tms_values[:, None] - control_values[None, :])
        rows.append({'metric': metric, 'metric_label': label, 'control_n': len(control_values), 'tms_n': len(tms_values), 'control_median': float(np.median(control_values)), 'tms_median': float(np.median(tms_values)), 'cliffs_delta_tms_minus_control': float(pairwise.mean()), 'mannwhitney_p': float(test.pvalue)})
    q_values = base.bh_adjust(p_values)
    for (row, q_value) in zip(rows, q_values):
        row['fdr_q'] = float(q_value)
    return pd.DataFrame(rows)

def plot_spatial_plate_white(fig: plt.Figure, spec, maps, counts: dict[str, int]) -> float:
    (anatomy_image, anatomy) = base.prepare_anatomy()
    arrays = {key: image.get_fdata(dtype=np.float32) for (key, image) in maps.items()}
    vmax = max((float(array.max()) for array in arrays.values()))
    inner = spec.subgridspec(5, len(base.SLICE_Z_MM), height_ratios=[1, 1, 1, 1, 0.12], hspace=0.045, wspace=0.025)
    anatomy_cmap = plt.get_cmap('gray_r').copy()
    anatomy_cmap.set_bad('white')
    for (row_index, (group, time, label, label_color)) in enumerate(base.SPATIAL_ROWS):
        overlay = arrays[group, time]
        for (column_index, z_mm) in enumerate(base.SLICE_Z_MM):
            ax = fig.add_subplot(inner[row_index, column_index])
            index = base.slice_index(anatomy_image, z_mm)
            anatomical_slice = np.rot90(anatomy[:, :, index])
            overlay_slice = np.rot90(overlay[:, :, index])
            ax.set_facecolor('white')
            ax.imshow(np.ma.masked_where(anatomical_slice <= 0, anatomical_slice), cmap=anatomy_cmap, vmin=0, vmax=1.15, interpolation='nearest')
            ax.imshow(np.ma.masked_where(overlay_slice <= 0, overlay_slice), cmap=base.OVERLAP_CMAP, norm=Normalize(0, vmax), interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if column_index == 0:
                ax.text(-0.055, 0.5, f'{label}\n(n={counts[group]})', transform=ax.transAxes, ha='right', va='center', fontsize=6.0, fontweight='bold', color=label_color, clip_on=False)
            if row_index == 0:
                ax.text(0.5, 1.035, f'z={z_mm} mm', transform=ax.transAxes, ha='center', va='bottom', fontsize=5.6, color=base.INK)
    colorbar_ax = fig.add_subplot(inner[4, 4:7])
    colorbar = fig.colorbar(ScalarMappable(norm=Normalize(0, vmax), cmap=base.OVERLAP_CMAP), cax=colorbar_ax, orientation='horizontal')
    ticks = [0, vmax / 2, vmax]
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels([f'{100 * value:.0f}' for value in ticks])
    colorbar.ax.tick_params(labelsize=5.2, width=0.5, length=2, pad=1)
    colorbar.outline.set_linewidth(0.5)
    colorbar.set_label('Participants with lesion at voxel (%)', fontsize=5.5, labelpad=1.5)
    return vmax

def make_overlap_figure(selected: pd.DataFrame, maps) -> float:
    (width_mm, height_mm) = (183.0, 100.0)
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), facecolor='white')
    grid = fig.add_gridspec(1, 1, left=0.105, right=0.985, bottom=0.075, top=0.89)
    counts = selected.groupby('group').size().astype(int).to_dict()
    vmax = plot_spatial_plate_white(fig, grid[0, 0], maps, counts)
    fig.text(0.105, 0.965, 'Paired longitudinal lesion topography', ha='left', va='bottom', fontsize=7.1, fontweight='bold', color=base.INK)
    base.panel_label(fig, 0.018, 0.95, 'a')
    save_bundle(fig, FIGURES / 'panel_a_lesion_overlap_outlier_excluded_nature')
    return vmax

def plot_one_group(ax: plt.Axes, subset: pd.DataFrame, group: str, q_value: float, pre_color: str, post_color: str, group_color: str, ylim: tuple[float, float], yticks: list[float], seed: int, log_scale: bool) -> None:
    subset = subset.sort_values('subject_id')
    pre = subset['baseline_volume_ml'].to_numpy(float)
    post = subset['post_volume_ml'].to_numpy(float)
    rng = np.random.default_rng(seed)
    for (pre_value, post_value) in zip(pre, post):
        jitter = float(rng.uniform(-0.045, 0.045))
        ax.plot([0 + jitter, 1 + jitter], [pre_value, post_value], color=base.PAIR_GREY, lw=0.6, alpha=0.58, zorder=1)
        ax.scatter(0 + jitter, pre_value, s=19, facecolor=pre_color, edgecolor='white', linewidth=0.5, zorder=4)
        ax.scatter(1 + jitter, post_value, s=19, facecolor=post_color, edgecolor='white', linewidth=0.5, zorder=4)
    boxplot = ax.boxplot([pre, post], positions=[0, 1], widths=0.46, patch_artist=True, showfliers=False, whis=1.5, medianprops={'color': base.INK, 'linewidth': 1.15}, whiskerprops={'color': base.NEUTRAL, 'linewidth': 0.75}, capprops={'color': base.NEUTRAL, 'linewidth': 0.75}, boxprops={'color': base.INK, 'linewidth': 0.65}, zorder=2)
    for (box, color) in zip(boxplot['boxes'], (pre_color, post_color)):
        box.set_facecolor(color)
        box.set_alpha(0.68)
    ax.set_xlim(-0.55, 1.55)
    if log_scale:
        ax.set_yscale('log')
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Pre', 'Post'])
    ax.set_title(f'{group} (n={len(subset)})', fontsize=6.8, fontweight='bold', color=group_color, pad=5)
    ax.grid(False)
    bracket_y = max(float(pre.max()), float(post.max())) * 1.08
    cap_y = bracket_y - 0.025 * (ylim[1] - ylim[0])
    text_y = bracket_y + 0.025 * (ylim[1] - ylim[0])
    ax.plot([0, 0, 1, 1], [cap_y, bracket_y, bracket_y, cap_y], color=base.INK, lw=0.65, clip_on=False, zorder=5)
    ax.text(0.5, text_y, f'q={q_value:.3f}', ha='center', va='bottom', fontsize=5.8, color=base.INK)

def make_boxplot_figure(selected: pd.DataFrame, within_stats: pd.DataFrame, between_stats: pd.DataFrame) -> None:
    (width_mm, height_mm) = (120.0, 82.0)
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), facecolor='white')
    grid = fig.add_gridspec(1, 2, left=0.125, right=0.985, bottom=0.185, top=0.755, wspace=0.42)
    ax_control = fig.add_subplot(grid[0, 0])
    ax_tms = fig.add_subplot(grid[0, 1])
    control_q = float(within_stats.loc[within_stats['group'] == 'Control', 'wilcoxon_fdr_q'].iloc[0])
    tms_q = float(within_stats.loc[within_stats['group'] == 'TMS', 'wilcoxon_fdr_q'].iloc[0])
    plot_one_group(ax_control, selected.loc[selected['group'] == 'Control'], 'Control', control_q, base.CONTROL_PRE, base.CONTROL, base.CONTROL, (0.1, 60), [0.1, 0.5, 1, 5, 20, 50], 20260803, True)
    plot_one_group(ax_tms, selected.loc[selected['group'] == 'TMS'], 'TMS', tms_q, base.TMS_PRE, base.TMS, base.TMS, (0.15, 13), [0.2, 0.5, 1, 2, 5, 10], 20260804, True)
    ax_control.set_ylim(0.1, 60)
    ax_control.yaxis.set_major_locator(FixedLocator([0.1, 0.5, 1, 5, 20, 50]))
    ax_control.yaxis.set_major_formatter(FixedFormatter(['0.1', '0.5', '1', '5', '20', '50']))
    ax_control.yaxis.set_minor_formatter(NullFormatter())
    ax_tms.yaxis.set_major_locator(FixedLocator([0.2, 0.5, 1, 2, 5, 10]))
    ax_tms.yaxis.set_major_formatter(FixedFormatter(['0.2', '0.5', '1', '2', '5', '10']))
    ax_tms.yaxis.set_minor_formatter(NullFormatter())
    ax_control.set_ylabel('Lesion volume (ml; log scale)')
    ax_tms.set_ylabel('Lesion volume (ml; log scale)')
    between_q = float(between_stats.loc[between_stats['metric'] == 'absolute_volume_change_ml', 'fdr_q'].iloc[0])
    fig.text(0.125, 0.94, 'Paired lesion-volume change', ha='left', va='bottom', fontsize=7.1, fontweight='bold', color=base.INK)
    fig.text(0.985, 0.94, f'Independent log y-axis ranges; between-group change q={between_q:.3f}', ha='right', va='bottom', fontsize=5.5, color=base.NEUTRAL)
    base.panel_label(fig, 0.025, 0.93, 'b')
    save_bundle(fig, FIGURES / 'panel_b_lesion_volume_separate_log_axes_outlier_excluded_nature')

def write_documentation(selected: pd.DataFrame, excluded: pd.DataFrame, within_stats: pd.DataFrame, between_stats: pd.DataFrame, overlap_vmax: float) -> None:
    counts = selected.groupby('group').size().astype(int).to_dict()
    control_q = float(within_stats.loc[within_stats.group == 'Control', 'wilcoxon_fdr_q'].iloc[0])
    tms_q = float(within_stats.loc[within_stats.group == 'TMS', 'wilcoxon_fdr_q'].iloc[0])
    between_q = float(between_stats.loc[between_stats.metric == 'absolute_volume_change_ml', 'fdr_q'].iloc[0])
    legend = f"# Figure legends\n\n## Spatial overlap\n\n**Longitudinal lesion topography after exclusion of the extreme-volume Control participant.** Lesion-overlap proportions at seven prespecified axial MNI coordinates are shown for TMS pre, TMS post, Control pre and Control post. The same participants contribute at both time points within each group (TMS, n={counts['TMS']}; Control, n={counts['Control']}). One Control participant with a baseline lesion volume of {float(excluded.baseline_volume_ml.iloc[0]):.3f} ml was excluded from both pre and post maps in this post hoc sensitivity visualization. Colour denotes the percentage of participants whose lesion includes the voxel; masks were not smoothed and were displayed using nearest-neighbour interpolation on a common 0-{100 * overlap_vmax:.1f}% scale. The zero-valued region outside the anatomical template was masked to white, following a white-background statistical-map display convention.\n\n## Paired lesion volume\n\n**Paired lesion volume after exclusion of the extreme-volume Control participant.** Grey lines connect repeated measures from the same participant; points show raw volumes and boxes show medians and interquartile ranges, with whiskers extending to 1.5 times the interquartile range. Both axes are logarithmic displays labelled in raw millilitres, and Control and TMS use explicitly different y-axis ranges. No statistical test used log-transformed volumes. Within-group q values are from two-sided paired Wilcoxon signed-rank tests with Benjamini-Hochberg correction across the two groups (Control q={control_q:.3f}; TMS q={tms_q:.3f}). The between-group q value compares absolute volume change using a two-sided Mann-Whitney test with Benjamini-Hochberg correction across six longitudinal lesion metrics (q={between_q:.3f}). The exclusion is post hoc; this figure should be labelled as a sensitivity analysis rather than replacing the full paired cohort without a clinical or technical exclusion rationale.\n"
    (OUT / 'FIGURE_LEGENDS.md').write_text(legend, encoding='utf-8')
    contract = f"# Figure contract\n\n- Core conclusion: After post hoc exclusion of the approximately 170-ml Control lesion, both groups retain within-group volume decreases, while between-group absolute volume change remains non-significant.\n- Backend: Python/matplotlib only.\n- Spatial figure: White-background anatomical image plate, 183 x 100 mm; four rows on one overlap scale.\n- Volume figure: Clinical paired comparison, 120 x 82 mm; two logarithmic display axes with explicitly independent ranges and raw-millilitre tick labels.\n- Cohort: Control n={counts['Control']}; TMS n={counts['TMS']}.\n- Reviewer risk: The excluded participant was not removed by the frozen longitudinal QC workflow. The exclusion is post hoc and must be disclosed; separate y-axis ranges prevent direct visual comparison of absolute vertical distances between groups.\n"
    (OUT / 'FIGURE_CONTRACT.md').write_text(contract, encoding='utf-8')
    definitions = {'excluded_subject_id': EXCLUDED_SUBJECT, 'excluded_group': 'Control', 'excluded_baseline_volume_ml': float(excluded.baseline_volume_ml.iloc[0]), 'excluded_post_volume_ml': float(excluded.post_volume_ml.iloc[0]), 'exclusion_status': 'post hoc sensitivity exclusion requested by analyst', 'paired_cohort': counts, 'overlap_vmax_proportion': overlap_vmax, 'volume_axis_scale': 'logarithmic display; raw-millilitre data and labels', 'volume_axis_ranges_ml': {'Control': [0.1, 60], 'TMS': [0.15, 13]}, 'within_group_fdr_q': {'Control': control_q, 'TMS': tms_q}, 'between_group_absolute_change_fdr_q': between_q}
    (SOURCE / 'statistics_definition.json').write_text(json.dumps(definitions, ensure_ascii=False, indent=2), encoding='utf-8')

def main() -> None:
    ensure_dirs()
    (included, _, _) = base.read_longitudinal_data()
    (selected, excluded) = select_sensitivity_cohort(included)
    base.SOURCE = SOURCE
    (maps, volume_long) = base.validate_and_build_overlap_maps(selected)
    within_stats = base.calculate_within_group_statistics(selected)
    between_stats = recompute_between_group_statistics(selected)
    excluded.to_csv(SOURCE / 'excluded_participant.csv', index=False)
    volume_long.to_csv(SOURCE / 'paired_lesion_volumes_outlier_excluded.csv', index=False)
    within_stats.to_csv(SOURCE / 'within_group_statistics_outlier_excluded.csv', index=False)
    between_stats.to_csv(SOURCE / 'between_group_statistics_outlier_excluded.csv', index=False)
    overlap_vmax = make_overlap_figure(selected, maps)
    make_boxplot_figure(selected, within_stats, between_stats)
    write_documentation(selected, excluded, within_stats, between_stats, overlap_vmax)
    print('Outlier-excluded sensitivity figures complete')
    print('Excluded sensitivity participant:', EXCLUDED_SUBJECT)
    print('Cohort sizes:', selected.groupby('group').size().to_dict())
    print(f'Common overlap maximum: {100 * overlap_vmax:.1f}%')
if __name__ == '__main__':
    main()
