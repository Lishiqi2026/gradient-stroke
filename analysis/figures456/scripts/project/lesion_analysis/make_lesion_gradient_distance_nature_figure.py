from __future__ import annotations
import os as _portable_os
from pathlib import Path as _PortablePath

def _portable_path(kind, relative):
    project = _PortablePath(_portable_os.environ['PAPER_PROJECT_ROOT'])
    root = project if kind == 'project' else project.parent
    return str(root / relative)
import tempfile
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from brainspace.datasets import load_parcellation
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.lines import Line2D
from neuromaps.datasets import fetch_fslr
from scipy import stats
from surfplot import Plot
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.size': 6.5, 'axes.labelsize': 6.5, 'axes.titlesize': 7.2, 'xtick.labelsize': 6.1, 'ytick.labelsize': 6.1, 'axes.linewidth': 0.7, 'axes.spines.top': False, 'axes.spines.right': False, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7, 'legend.frameon': False, 'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'axes.unicode_minus': False})
ROOT = Path(__file__).resolve().parent
TABLE_DIR = ROOT / 'lesion_gradient_distance' / 'tables'
OUT_DIR = ROOT / 'lesion_gradient_distance' / 'figures' / 'nature'
SOURCE_DIR = OUT_DIR / 'source_data'
CONTROL = '#5B8DB8'
CONTROL_DARK = '#315F87'
TMS = '#E18439'
TMS_DARK = '#A95316'
NEUTRAL = '#777777'
LIGHT = '#D8D8D8'
BLACK = '#242424'
ZERO = '#B8B8B8'
G2_CMAP = LinearSegmentedColormap.from_list('g2_reference', ['#2D4F8B', '#79A8D8', '#F7F7F7', '#E7A66C', '#A34B24'])
DELTA_CMAP = LinearSegmentedColormap.from_list('normalization_delta', ['#315F87', '#A7C6DF', '#F8F8F8', '#F1B47D', '#A95316'])
METRIC_LABELS = {'mean_reference_directed_G2': 'Whole-cortex G2', 'mean_reference_directed_G1G2': 'Whole-cortex G1-G2', 'nearest_anatomical_decile_reference_directed_G1G2': 'Anatomically nearest 10%\nG1-G2', 'xtract_projected_G2_slope': 'XTRACT G2-distance slope', 'nearest_xtract_G1G2_decile_normalization': 'XTRACT-nearest 10%\nG1-G2'}
MATCH_LABELS = {'all_11_pairs': 'All pairs (11)', 'motion_qc_10_pairs': 'Motion QC (10)', 'tight_9_pairs': 'Tight match (9)', 'motion_qc_tight_8_pairs': 'Motion QC + tight (8)'}

def add_panel_label(ax: mpl.axes.Axes, label: str, x: float=-0.12, y: float=1.04) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=8.5, fontweight='bold', ha='left', va='bottom', color=BLACK)

def mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    mean = float(np.mean(values))
    if values.size < 2:
        return (mean, np.nan, np.nan)
    half = float(stats.t.ppf(0.975, values.size - 1) * stats.sem(values))
    return (mean, mean - half, mean + half)

def _crop_white(image: np.ndarray, threshold: float=0.995) -> np.ndarray:
    rgb = image[..., :3] if image.shape[-1] == 4 else image
    mask = np.any(rgb < threshold, axis=-1)
    if not np.any(mask):
        return image
    rows = np.flatnonzero(np.any(mask, axis=1))
    cols = np.flatnonzero(np.any(mask, axis=0))
    pad = 5
    (r0, r1) = (max(0, rows[0] - pad), min(image.shape[0], rows[-1] + pad + 1))
    (c0, c1) = (max(0, cols[0] - pad), min(image.shape[1], cols[-1] + pad + 1))
    return np.ascontiguousarray(image[r0:r1, c0:c1])

def render_surface(values: np.ndarray, cmap, color_range: tuple[float, float], path: Path) -> np.ndarray:
    surfaces = fetch_fslr()
    (lh_surf, rh_surf) = surfaces['veryinflated']
    atlas = load_parcellation('schaefer', 400, join=True)
    vertex_values = np.zeros_like(atlas, dtype=float)
    for label in range(1, 401):
        vertex_values[atlas == label] = values[label - 1]
    plot = Plot(lh_surf, rh_surf, views=['lateral', 'medial'], zoom=1.55, brightness=0.78, size=(1900, 620))
    plot.add_layer(vertex_values, cmap=cmap, color_range=list(color_range), cbar=False)
    plot.add_layer(atlas, cmap='gray', as_outline=True, cbar=False)
    surface_figure = plot.build()
    surface_figure.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0, facecolor='white')
    plt.close(surface_figure)
    return _crop_white(mpimg.imread(path))

def residualized_distance_bins(roi: pd.DataFrame, eligible_subjects: set[str]) -> pd.DataFrame:
    records: list[dict] = []
    use = roi.loc[roi['subject_id'].astype(str).isin(eligible_subjects) & roi['intact_roi'].astype(bool)].copy()
    for (subject_id, frame) in use.groupby('subject_id', sort=True):
        frame = frame.dropna(subset=['reference_directed_G2', 'xtract_projected_G2_distance', 'anatomical_distance_min_mm']).copy()
        y = frame['reference_directed_G2'].to_numpy(float)
        anatomical = frame['anatomical_distance_min_mm'].to_numpy(float)
        design = np.column_stack([np.ones(len(frame)), anatomical])
        fitted = design @ np.linalg.lstsq(design, y, rcond=None)[0]
        frame['G2_normalization_anatomical_adjusted'] = y - fitted + np.mean(y)
        frame['distance_decile'] = pd.qcut(frame['xtract_projected_G2_distance'], q=10, labels=False, duplicates='drop').astype(int) + 1
        group = str(frame['group'].iloc[0])
        for (decile, bin_frame) in frame.groupby('distance_decile'):
            records.append({'subject_id': str(subject_id), 'group': group, 'distance_decile': int(decile), 'mean_G2_normalization_anatomical_adjusted': float(bin_frame['G2_normalization_anatomical_adjusted'].mean()), 'mean_xtract_G2_distance': float(bin_frame['xtract_projected_G2_distance'].mean()), 'roi_n': int(len(bin_frame))})
    return pd.DataFrame.from_records(records)

def grouped_bin_summary(subject_bins: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for ((group, decile), frame) in subject_bins.groupby(['group', 'distance_decile']):
        (mean, low, high) = mean_ci(frame['mean_G2_normalization_anatomical_adjusted'].to_numpy())
        rows.append({'group': group, 'distance_decile': int(decile), 'mean': mean, 'ci_low': low, 'ci_high': high, 'n_subjects': int(frame['subject_id'].nunique())})
    return pd.DataFrame(rows)

def violin_points(ax: mpl.axes.Axes, values: np.ndarray, position: float, color: str, rng: np.random.Generator, width: float=0.28) -> None:
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

def p_text(value: float) -> str:
    if value < 0.001:
        return '<0.001'
    return f'{value:.3f}'

def load_data() -> dict[str, pd.DataFrame]:
    return {'roi': pd.read_csv(TABLE_DIR / 'roi_level_distance_reorganization.csv', dtype={'subject_id': str}), 'subject': pd.read_csv(TABLE_DIR / 'subject_level_distance_metrics.csv', dtype={'subject_id': str}), 'models': pd.read_csv(TABLE_DIR / 'group_models_reference_directed_reorganization.csv'), 'matched': pd.read_csv(TABLE_DIR / 'xtract_projected_matched_group_tests.csv'), 'pair_values': pd.read_csv(TABLE_DIR / 'xtract_projected_matched_pair_differences.csv', dtype={'control_subject_id': str, 'tms_subject_id': str}), 'reference': pd.read_csv(TABLE_DIR / 'healthy_normative_reference_schaefer400.csv')}

def prepare_source_data(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    subject = data['subject'].copy()
    primary = subject.loc[subject['primary_motion_eligible'].astype(bool)].copy()
    eligible = set(primary['subject_id'].astype(str))
    surface = data['reference'][['roi', 'roi_name', 'network', 'G2_reference_z']].copy()
    roi_primary = data['roi'].loc[data['roi']['subject_id'].astype(str).isin(eligible)].copy()
    surface_group = roi_primary.groupby(['roi', 'group'], as_index=False)['reference_directed_G2'].mean().pivot(index='roi', columns='group', values='reference_directed_G2')
    surface['Control_mean_G2_normalization'] = surface['roi'].map(surface_group['Control'])
    surface['TMS_mean_G2_normalization'] = surface['roi'].map(surface_group['TMS'])
    surface['TMS_minus_Control_G2_normalization'] = surface['TMS_mean_G2_normalization'] - surface['Control_mean_G2_normalization']
    subject_bins = residualized_distance_bins(data['roi'], eligible)
    bin_summary = grouped_bin_summary(subject_bins)
    primary_models = data['models'].loc[data['models']['cohort'] == 'primary_motion_qc'].copy()
    selected_outcomes = list(METRIC_LABELS)
    forest = primary_models.loc[primary_models['outcome'].isin(selected_outcomes)].copy()
    forest['display_label'] = forest['outcome'].map(METRIC_LABELS)
    matched_pairs = data['pair_values'].loc[(data['pair_values']['match_set'] == 'motion_qc_10_pairs') & (data['pair_values']['outcome'] == 'nearest_xtract_G1G2_decile_normalization')].copy()
    sensitivity = data['matched'].loc[data['matched']['outcome'] == 'nearest_xtract_G1G2_decile_normalization'].copy()
    sensitivity['display_label'] = sensitivity['match_set'].map(MATCH_LABELS)
    exports = {'panel_a_surface_source_data.csv': surface, 'panel_b_d_subject_metrics.csv': primary, 'panel_c_subject_distance_bins.csv': subject_bins, 'panel_c_group_distance_bin_summary.csv': bin_summary, 'panel_e_adjusted_effects.csv': forest, 'panel_f_motion_qc_matched_pairs.csv': matched_pairs, 'panel_g_match_sensitivity.csv': sensitivity}
    for (filename, frame) in exports.items():
        frame.to_csv(SOURCE_DIR / filename, index=False)
    return {'surface': surface, 'primary': primary, 'subject_bins': subject_bins, 'bin_summary': bin_summary, 'forest': forest, 'matched_pairs': matched_pairs, 'sensitivity': sensitivity}

def make_figure(source: dict[str, pd.DataFrame]) -> plt.Figure:
    rng = np.random.default_rng(20260804)
    fig = plt.figure(figsize=(7.2, 8.1), constrained_layout=False)
    grid = fig.add_gridspec(4, 12, height_ratios=[1.28, 1.15, 1.25, 1.2], left=0.075, right=0.985, bottom=0.065, top=0.978, hspace=0.63, wspace=1.9)
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, :5])
    ax_c = fig.add_subplot(grid[1, 5:])
    ax_d = fig.add_subplot(grid[2, :4])
    ax_e = fig.add_subplot(grid[2, 5:])
    ax_f = fig.add_subplot(grid[3, :6])
    ax_g = fig.add_subplot(grid[3, 6:])
    ax_a.set_axis_off()
    add_panel_label(ax_a, 'a', x=-0.035, y=1.0)
    surface = source['surface'].sort_values('roi')
    g2_values = surface['G2_reference_z'].to_numpy(float)
    delta_values = surface['TMS_minus_Control_G2_normalization'].to_numpy(float)
    delta_limit = float(np.nanpercentile(np.abs(delta_values), 98))
    delta_limit = max(delta_limit, 0.02)
    with tempfile.TemporaryDirectory(prefix='lesion_gradient_surface_') as tmp:
        ref_image = render_surface(g2_values, G2_CMAP, (float(np.nanmin(g2_values)), float(np.nanmax(g2_values))), Path(tmp) / 'reference.png')
        delta_image = render_surface(delta_values, DELTA_CMAP, (-delta_limit, delta_limit), Path(tmp) / 'delta.png')
    ref_ax = ax_a.inset_axes([0.0, 0.12, 0.48, 0.75])
    delta_ax = ax_a.inset_axes([0.52, 0.12, 0.48, 0.75])
    for (target, image) in ((ref_ax, ref_image), (delta_ax, delta_image)):
        target.imshow(image)
        target.set_axis_off()
    ax_a.text(0.24, 0.97, 'Healthy normative G2', ha='center', va='top', fontsize=7.2, fontweight='bold')
    ax_a.text(0.76, 0.97, 'TMS - Control G2 normalization (descriptive)', ha='center', va='top', fontsize=7.2, fontweight='bold')
    ref_cax = ax_a.inset_axes([0.135, 0.04, 0.21, 0.035])
    delta_cax = ax_a.inset_axes([0.655, 0.04, 0.21, 0.035])
    cb1 = fig.colorbar(mpl.cm.ScalarMappable(norm=Normalize(float(np.nanmin(g2_values)), float(np.nanmax(g2_values))), cmap=G2_CMAP), cax=ref_cax, orientation='horizontal')
    cb2 = fig.colorbar(mpl.cm.ScalarMappable(norm=Normalize(-delta_limit, delta_limit), cmap=DELTA_CMAP), cax=delta_cax, orientation='horizontal')
    cb1.set_label('Reference G2 (z)', fontsize=5.8, labelpad=1.0)
    cb2.set_label('Mean normalization difference', fontsize=5.8, labelpad=1.0)
    for colorbar in (cb1, cb2):
        colorbar.ax.tick_params(labelsize=5.3, length=1.5, pad=1)
        colorbar.outline.set_linewidth(0.45)
    primary = source['primary']
    global_specs = [('mean_reference_directed_G2', 'G2'), ('mean_reference_directed_G1G2', 'G1-G2')]
    for (metric_index, (metric, _)) in enumerate(global_specs):
        for (group, offset, color) in (('Control', -0.17, CONTROL), ('TMS', 0.17, TMS)):
            values = primary.loc[primary['group'] == group, metric].to_numpy(float)
            violin_points(ax_b, values, metric_index + offset, color, rng, width=0.26)
    ax_b.axhline(0, color=ZERO, lw=0.7, zorder=0)
    ax_b.set_xticks([0, 1], [label for (_, label) in global_specs])
    ax_b.set_ylabel('Healthy-reference normalization')
    ax_b.set_title('Whole-cortex normalization', loc='left', fontweight='bold', pad=3)
    add_panel_label(ax_b, 'b')
    models = source['forest'].set_index('outcome')
    ymax = float(np.nanmax(primary[[metric for (metric, _) in global_specs]].to_numpy()))
    ymin = float(np.nanmin(primary[[metric for (metric, _) in global_specs]].to_numpy()))
    span = ymax - ymin
    ax_b.set_ylim(ymin - 0.08 * span, ymax + 0.28 * span)
    for (i, (metric, _)) in enumerate(global_specs):
        row = models.loc[metric]
        ax_b.text(i, ymax + 0.12 * span, f"adjusted beta={row['group_tms_beta']:.3f}\nq={p_text(row['fdr_q_freedman_lane'])}", ha='center', va='bottom', fontsize=5.6)
    summary = source['bin_summary']
    for (group, color) in (('Control', CONTROL), ('TMS', TMS)):
        frame = summary.loc[summary['group'] == group].sort_values('distance_decile')
        x = frame['distance_decile'].to_numpy(float)
        mean = frame['mean'].to_numpy(float)
        low = frame['ci_low'].to_numpy(float)
        high = frame['ci_high'].to_numpy(float)
        ax_c.fill_between(x, low, high, color=color, alpha=0.16, linewidth=0)
        ax_c.plot(x, mean, color=color, lw=1.65, marker='o', ms=3.2, label=group)
    ax_c.axhline(0, color=ZERO, lw=0.7)
    ax_c.set_xticks([1, 3, 5, 7, 10])
    ax_c.set_xlabel('XTRACT-projected G2-distance decile\nnear lesion hierarchy  ->  far')
    ax_c.set_ylabel('G2 normalization\n(adjusted for anatomical distance)')
    ax_c.set_title('Distance-dependent gradient reorganization', loc='left', fontweight='bold', pad=3)
    ax_c.legend(loc='upper left', ncol=2, handlelength=1.4, columnspacing=1.0, fontsize=5.8)
    add_panel_label(ax_c, 'c', x=-0.09)
    slope_metric = 'xtract_projected_G2_slope'
    for (position, (group, color)) in enumerate((('Control', CONTROL), ('TMS', TMS))):
        violin_points(ax_d, primary.loc[primary['group'] == group, slope_metric].to_numpy(float), position, color, rng, width=0.28)
    ax_d.axhline(0, color=ZERO, lw=0.7)
    ax_d.set_xticks([0, 1], ['Control', 'TMS'])
    ax_d.set_ylabel('Within-subject G2-distance slope')
    ax_d.set_title('Subject-level G2 slopes', loc='left', fontweight='bold', pad=3)
    row = models.loc[slope_metric]
    ax_d.text(0.5, 0.97, f"adjusted beta={row['group_tms_beta']:.3f}\npermutation p={p_text(row['p_freedman_lane'])}; q={p_text(row['fdr_q_freedman_lane'])}", transform=ax_d.transAxes, ha='center', va='top', fontsize=5.4)
    add_panel_label(ax_d, 'd', x=-0.16)
    forest_order = ['mean_reference_directed_G2', 'mean_reference_directed_G1G2', 'nearest_anatomical_decile_reference_directed_G1G2', 'xtract_projected_G2_slope', 'nearest_xtract_G1G2_decile_normalization']
    forest = source['forest'].set_index('outcome').loc[forest_order].reset_index()
    y = np.arange(len(forest))[::-1]
    effect_colors = [TMS_DARK, TMS_DARK, TMS_DARK, '#7C5AA6', '#7C5AA6']
    for (yi, (_, effect), color) in zip(y, forest.iterrows(), effect_colors):
        ax_e.plot([effect['ci_low_hc3'], effect['ci_high_hc3']], [yi, yi], color=color, lw=1.15)
        ax_e.scatter(effect['group_tms_beta'], yi, s=20, color=color, edgecolor='white', lw=0.35, zorder=3)
    ax_e.axvline(0, color=ZERO, lw=0.8, ls='--')
    ax_e.set_yticks(y, forest['display_label'])
    ax_e.set_xlabel('Adjusted TMS effect (beta, HC3 95% CI)')
    ax_e.set_title('Prespecified adjusted effects', loc='left', fontweight='bold', pad=3)
    xmin = float(forest['ci_low_hc3'].min())
    xmax = float(forest['ci_high_hc3'].max())
    xspan = xmax - xmin
    ax_e.set_xlim(xmin - 0.1 * xspan, xmax + 0.35 * xspan)
    for (yi, (_, effect)) in zip(y, forest.iterrows()):
        ax_e.text(xmax + 0.05 * xspan, yi, f"q={p_text(effect['fdr_q_freedman_lane'])}", va='center', ha='left', fontsize=5.4, color=BLACK if effect['fdr_q_freedman_lane'] < 0.05 else NEUTRAL)
    add_panel_label(ax_e, 'e', x=-0.12)
    matched_pairs = source['matched_pairs'].sort_values('pair_id')
    for (_, pair) in matched_pairs.iterrows():
        ax_f.plot([0, 1], [pair['control_value'], pair['tms_value']], color='#B8B8B8', lw=0.8, alpha=0.8, zorder=1)
    ax_f.scatter(np.zeros(len(matched_pairs)), matched_pairs['control_value'], color=CONTROL, s=20, edgecolor='white', lw=0.35, zorder=3)
    ax_f.scatter(np.ones(len(matched_pairs)), matched_pairs['tms_value'], color=TMS, s=20, edgecolor='white', lw=0.35, zorder=3)
    control_mean = float(matched_pairs['control_value'].mean())
    tms_mean = float(matched_pairs['tms_value'].mean())
    ax_f.plot([0, 1], [control_mean, tms_mean], color=BLACK, lw=1.8, zorder=4)
    ax_f.scatter([0, 1], [control_mean, tms_mean], marker='D', s=25, color=BLACK, zorder=5)
    ax_f.axhline(0, color=ZERO, lw=0.7)
    ax_f.set_xlim(-0.25, 1.25)
    ax_f.set_xticks([0, 1], ['Control', 'TMS'])
    ax_f.set_ylabel('Nearest XTRACT G1-G2 decile\nnormalization')
    ax_f.set_title('Lesion-matched comparison', loc='left', fontweight='bold', pad=3)
    matched_summary = source['sensitivity'].set_index('match_set').loc['motion_qc_10_pairs']
    ax_f.text(0.5, 0.98, f"10 pairs; mean difference={matched_summary['mean_tms_minus_control']:.3f}\nexact p={p_text(matched_summary['exact_signflip_p'])}; q={p_text(matched_summary['fdr_q_within_match_set'])}", transform=ax_f.transAxes, ha='center', va='top', fontsize=5.5)
    add_panel_label(ax_f, 'f', x=-0.11)
    match_order = ['all_11_pairs', 'motion_qc_10_pairs', 'tight_9_pairs', 'motion_qc_tight_8_pairs']
    sensitivity = source['sensitivity'].set_index('match_set').loc[match_order].reset_index()
    y_match = np.arange(len(sensitivity))[::-1]
    for (yi, (_, effect)) in zip(y_match, sensitivity.iterrows()):
        significant = effect['fdr_q_within_match_set'] < 0.05
        color = TMS_DARK if significant else '#A98C78'
        ax_g.plot([effect['ci_low_t'], effect['ci_high_t']], [yi, yi], color=color, lw=1.2)
        ax_g.scatter(effect['mean_tms_minus_control'], yi, s=22, color=color, edgecolor='white', lw=0.35, zorder=3)
    ax_g.axvline(0, color=ZERO, lw=0.8, ls='--')
    ax_g.set_yticks(y_match, sensitivity['display_label'])
    ax_g.set_xlabel('Paired TMS-Control difference (95% CI)')
    ax_g.set_title('Matching robustness', loc='left', fontweight='bold', pad=3)
    xlow = float(sensitivity['ci_low_t'].min())
    xhigh = float(sensitivity['ci_high_t'].max())
    xrange = xhigh - xlow
    ax_g.set_xlim(min(-0.02, xlow - 0.08 * xrange), xhigh + 0.33 * xrange)
    for (yi, (_, effect)) in zip(y_match, sensitivity.iterrows()):
        ax_g.text(xhigh + 0.04 * xrange, yi, f"q={p_text(effect['fdr_q_within_match_set'])}", va='center', fontsize=5.4, color=BLACK if effect['fdr_q_within_match_set'] < 0.05 else NEUTRAL)
    add_panel_label(ax_g, 'g', x=-0.13)
    handles = [Line2D([0], [0], marker='o', color='none', markerfacecolor=CONTROL, markeredgecolor='white', markersize=5, label='Control (n=15)'), Line2D([0], [0], marker='o', color='none', markerfacecolor=TMS, markeredgecolor='white', markersize=5, label='TMS (n=22)')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.8, 0.997), ncol=2, fontsize=6.0, handletextpad=0.4, columnspacing=1.1)
    return fig

def write_legend(source: dict[str, pd.DataFrame]) -> None:
    models = source['forest'].set_index('outcome')
    match = source['sensitivity'].set_index('match_set').loc['motion_qc_10_pairs']
    legend = f"# Figure legend\n\n**Lesion-gradient distance and TMS-associated gradient normalization.**\n**a,** Healthy normative G2 mapped to Schaefer400/fsLR and the descriptive parcel-wise difference in mean G2 healthy-reference normalization between TMS and Control. The latter is a spatial visualization rather than an ROI-wise statistical map. Direct cortical lesion-gradient mapping was available for 5/16 Control and 0/22 TMS participants; therefore all-subject lesion-gradient analyses use a normative XTRACT tract-endpoint projection proxy. **b,** Whole-cortex G2 and joint G1-G2 healthy-reference normalization in the primary motion-QC cohort (Control, n=15; TMS, n=22). Points are participants; horizontal bars and vertical intervals denote the mean and 95% CI. Annotated effects are covariate-adjusted TMS coefficients; G2: beta={models.loc['mean_reference_directed_G2', 'group_tms_beta']:.3f}, permutation p={models.loc['mean_reference_directed_G2', 'p_freedman_lane']:.4f}, FDR q={models.loc['mean_reference_directed_G2', 'fdr_q_freedman_lane']:.4f}; G1-G2: beta={models.loc['mean_reference_directed_G1G2', 'group_tms_beta']:.3f}, p={models.loc['mean_reference_directed_G1G2', 'p_freedman_lane']:.4f}, q={models.loc['mean_reference_directed_G1G2', 'fdr_q_freedman_lane']:.4f}. **c,** Mean G2 normalization across within-participant deciles of XTRACT-projected G2 distance after residualizing minimum anatomical distance. Lines and ribbons show group means and 95% CIs across participants; binning is descriptive and inference is performed on participant-level slopes. **d,** Participant-level G2-distance slopes controlling anatomical distance. The adjusted TMS coefficient was {models.loc['xtract_projected_G2_slope', 'group_tms_beta']:.3f} (permutation p={models.loc['xtract_projected_G2_slope', 'p_freedman_lane']:.4f}, within-family FDR q={models.loc['xtract_projected_G2_slope', 'fdr_q_freedman_lane']:.4f}). A positive TMS coefficient indicates that normalization extends relatively farther in G2 space rather than remaining concentrated near the lesion-related G2 neighbourhood. **e,** Prespecified covariate-adjusted TMS effects with HC3 95% CIs. q values are Benjamini-Hochberg FDR values within the global-normalization or XTRACT lesion-gradient family. Models adjust for log lesion volume, lesion lateralization, matching baseline gradient deviation, baseline mean framewise displacement and post mean framewise displacement; XTRACT outcomes additionally adjust projected lesion G1/G2 position, projectable tract fraction and effective cortical endpoint count as applicable. p values were obtained from 5,000 Freedman-Lane permutations. **f,** Motion-QC lesion-matched pairs for normalization in the nearest XTRACT-projected G1-G2 decile. Grey lines join matched Control and TMS participants; diamonds show group means. The mean paired difference was {match['mean_tms_minus_control']:.3f} (exact sign-flip p={match['exact_signflip_p']:.4f}, within-set FDR q={match['fdr_q_within_match_set']:.4f}). **g,** The same paired effect across nested match definitions. Intervals are t-based 95% CIs; q values are FDR-corrected within each four-outcome match set. These observational estimates support an association with TMS but are not unconfounded causal effects.\n"
    (OUT_DIR / 'FIGURE_LEGEND.md').write_text(legend, encoding='utf-8')

def export_standalone_surface_panels(source: dict[str, pd.DataFrame]) -> None:
    panel_dir = OUT_DIR / 'panel_a_standalone'
    panel_dir.mkdir(parents=True, exist_ok=True)
    surface = source['surface'].sort_values('roi')
    g2_values = surface['G2_reference_z'].to_numpy(float)
    delta_values = surface['TMS_minus_Control_G2_normalization'].to_numpy(float)
    delta_limit = float(np.nanpercentile(np.abs(delta_values), 98))
    delta_limit = max(delta_limit, 0.02)
    panel_specs = [{'values': g2_values, 'cmap': G2_CMAP, 'limits': (float(np.nanmin(g2_values)), float(np.nanmax(g2_values))), 'title': 'Healthy normative G2', 'colorbar_label': 'Reference G2 (z)', 'filename': 'panel_a1_healthy_normative_G2_surface'}, {'values': delta_values, 'cmap': DELTA_CMAP, 'limits': (-delta_limit, delta_limit), 'title': 'TMS - Control G2 normalization (descriptive)', 'colorbar_label': 'Mean normalization difference', 'filename': 'panel_a2_TMS_minus_Control_G2_normalization_surface'}]
    surfaces = fetch_fslr()
    (lh_surf, rh_surf) = surfaces['veryinflated']
    atlas = load_parcellation('schaefer', 400, join=True)
    medial_wall = np.full_like(atlas, np.nan, dtype=float)
    medial_wall[atlas == 0] = 1.0
    medial_cmap = ListedColormap(['#D1D1D1'])
    outline_cmap = ListedColormap(['#606060'])
    for spec in panel_specs:
        vertex_values = np.full_like(atlas, np.nan, dtype=float)
        for label in range(1, 401):
            vertex_values[atlas == label] = spec['values'][label - 1]
        plot = Plot(lh_surf, rh_surf, views=['lateral', 'medial'], zoom=1.68, brightness=0.65, size=(3600, 2600), background=(1, 1, 1))
        plot.add_layer(medial_wall, cmap=medial_cmap, color_range=(0, 1), cbar=False)
        plot.add_layer(vertex_values, cmap=spec['cmap'], color_range=spec['limits'], cbar=False)
        plot.add_layer(atlas, cmap=outline_cmap, as_outline=True, cbar=False)
        fig = plot.build()
        fig.patch.set_facecolor('white')
        fig.set_size_inches(4.1, 2.7, forward=True)
        brain_axes = fig.axes[:4]
        if len(brain_axes) >= 4:
            positions = [[0.035, 0.525, 0.445, 0.355], [0.52, 0.525, 0.445, 0.355], [0.035, 0.165, 0.445, 0.355], [0.52, 0.165, 0.445, 0.355]]
            for (axis, position) in zip(brain_axes, positions):
                axis.set_position(position)
        fig.suptitle(spec['title'], x=0.5, y=0.965, ha='center', va='top', fontsize=8.3, fontweight='bold', fontname='Arial')
        color_axis = fig.add_axes([0.275, 0.105, 0.45, 0.028])
        colorbar = fig.colorbar(mpl.cm.ScalarMappable(norm=Normalize(spec['limits'][0], spec['limits'][1]), cmap=spec['cmap']), cax=color_axis, orientation='horizontal')
        colorbar.set_label(spec['colorbar_label'], fontsize=6.5, labelpad=1.5)
        colorbar.ax.tick_params(labelsize=5.8, length=1.8, width=0.5, pad=1.0)
        colorbar.outline.set_linewidth(0.5)
        base = panel_dir / spec['filename']
        fig.savefig(base.with_suffix('.pdf'), dpi=600, facecolor='white', bbox_inches=None, pad_inches=0)
        fig.savefig(base.with_suffix('.svg'), dpi=600, facecolor='white', bbox_inches=None, pad_inches=0)
        fig.savefig(base.with_suffix('.png'), dpi=600, facecolor='white', bbox_inches=None, pad_inches=0)
        plt.close(fig)

def export_standalone_panel_d(source: dict[str, pd.DataFrame]) -> None:
    panel_dir = OUT_DIR / 'panel_d_standalone'
    panel_dir.mkdir(parents=True, exist_ok=True)
    primary = source['primary']
    slope_metric = 'xtract_projected_G2_slope'
    model = source['forest'].set_index('outcome').loc[slope_metric]
    rng = np.random.default_rng(20260804)
    (fig, ax) = plt.subplots(figsize=(3.35, 2.7))
    for (position, (group, color)) in enumerate((('Control', CONTROL), ('TMS', TMS))):
        violin_points(ax, primary.loc[primary['group'] == group, slope_metric].to_numpy(float), position, color, rng, width=0.28)
    values = primary[slope_metric].to_numpy(float)
    value_min = float(np.nanmin(values))
    value_max = float(np.nanmax(values))
    span = value_max - value_min
    ax.set_ylim(value_min - 0.1 * span, value_max + 0.32 * span)
    ax.axhline(0, color=ZERO, lw=0.8, zorder=0)
    ax.set_xticks([0, 1], ['Control\n(n=15)', 'TMS\n(n=22)'])
    ax.set_ylabel('Within-subject G2-distance slope')
    ax.set_title('Subject-level G2 slopes', loc='left', fontweight='bold', pad=5)
    ax.text(0.5, 0.965, f"adjusted beta={model['group_tms_beta']:.3f}; permutation p={model['p_freedman_lane']:.3f}; FDR q={model['fdr_q_freedman_lane']:.3f}", transform=ax.transAxes, ha='center', va='top', fontsize=6.0, color=BLACK)
    ax.tick_params(axis='x', length=0, pad=4)
    fig.subplots_adjust(left=0.22, right=0.97, bottom=0.22, top=0.88)
    base = panel_dir / 'subject_level_G2_slopes_no_panel_label'
    fig.savefig(base.with_suffix('.pdf'), dpi=600, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(base.with_suffix('.svg'), dpi=600, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(base.with_suffix('.png'), dpi=600, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)

def export_standalone_panel_f(source: dict[str, pd.DataFrame]) -> None:
    panel_dir = OUT_DIR / 'panel_f_standalone'
    panel_dir.mkdir(parents=True, exist_ok=True)
    matched_pairs = source['matched_pairs'].sort_values('pair_id')
    matched_summary = source['sensitivity'].set_index('match_set').loc['motion_qc_10_pairs']
    (fig, ax) = plt.subplots(figsize=(3.55, 2.75))
    for (_, pair) in matched_pairs.iterrows():
        ax.plot([0, 1], [pair['control_value'], pair['tms_value']], color='#B8B8B8', lw=0.9, alpha=0.85, zorder=1)
    ax.scatter(np.zeros(len(matched_pairs)), matched_pairs['control_value'], color=CONTROL, s=27, edgecolor='white', linewidth=0.45, zorder=3)
    ax.scatter(np.ones(len(matched_pairs)), matched_pairs['tms_value'], color=TMS, s=27, edgecolor='white', linewidth=0.45, zorder=3)
    control_mean = float(matched_pairs['control_value'].mean())
    tms_mean = float(matched_pairs['tms_value'].mean())
    ax.plot([0, 1], [control_mean, tms_mean], color=BLACK, lw=2.0, zorder=4)
    ax.scatter([0, 1], [control_mean, tms_mean], marker='D', s=32, color=BLACK, zorder=5)
    values = np.concatenate([matched_pairs['control_value'].to_numpy(float), matched_pairs['tms_value'].to_numpy(float)])
    value_min = float(np.nanmin(values))
    value_max = float(np.nanmax(values))
    span = value_max - value_min
    ax.set_ylim(value_min - 0.1 * span, value_max + 0.25 * span)
    ax.axhline(0, color=ZERO, lw=0.8, zorder=0)
    ax.set_xlim(-0.22, 1.22)
    ax.set_xticks([0, 1], ['Control\n(n=10)', 'TMS\n(n=10)'])
    ax.set_ylabel('Nearest XTRACT G1-G2 decile\nnormalization')
    ax.set_title('Lesion-matched comparison', loc='left', fontweight='bold', pad=5)
    ax.text(0.5, 0.965, f"mean paired difference={matched_summary['mean_tms_minus_control']:.3f}; exact p={matched_summary['exact_signflip_p']:.3f}; FDR q={matched_summary['fdr_q_within_match_set']:.3f}", transform=ax.transAxes, ha='center', va='top', fontsize=5.9, color=BLACK)
    ax.tick_params(axis='x', length=0, pad=4)
    fig.subplots_adjust(left=0.23, right=0.97, bottom=0.22, top=0.88)
    base = panel_dir / 'lesion_matched_comparison_no_panel_label'
    fig.savefig(base.with_suffix('.pdf'), dpi=600, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(base.with_suffix('.svg'), dpi=600, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(base.with_suffix('.png'), dpi=600, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    source = prepare_source_data(data)
    export_standalone_surface_panels(source)
    export_standalone_panel_d(source)
    export_standalone_panel_f(source)
    figure = make_figure(source)
    base = OUT_DIR / 'lesion_gradient_distance_main_figure'
    figure.savefig(base.with_suffix('.svg'), bbox_inches='tight', pad_inches=0.02)
    figure.savefig(base.with_suffix('.pdf'), bbox_inches='tight', pad_inches=0.02)
    figure.savefig(base.with_suffix('.tiff'), dpi=600, bbox_inches='tight', pad_inches=0.02)
    figure.savefig(base.with_suffix('.png'), dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close(figure)
    write_legend(source)
    print(f'Saved figure bundle to: {OUT_DIR}')
if __name__ == '__main__':
    main()
