from __future__ import annotations
import os as _portable_os
from pathlib import Path as _PortablePath

def _portable_path(kind, relative):
    project = _PortablePath(_portable_os.environ['PAPER_PROJECT_ROOT'])
    root = project if kind == 'project' else project.parent
    return str(root / relative)
import math
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
from matplotlib.patches import Rectangle
from neuromaps.datasets import fetch_fslr
from PIL import Image
from scipy.stats import linregress, pearsonr, spearmanr, t as student_t
from surfplot import Plot
import make_main_tms_vs_external_adjusted_beta_surfaces as base
import run_G2_positive_spintest_main_vs_cross_cohort as g2_spin
HERE = Path(__file__).resolve().parent
OUTPUT = HERE / 'output'
SOURCE = HERE / 'source_data'
QA = HERE / 'qa'
FORAI = HERE.parent / 'forai'
INPUT = SOURCE / 'main_internal_and_cross_cohort_G2_beta_maps.csv'
STEM = OUTPUT / 'toprow_G2_beta_maps_separate_colorbars_spatial_correlation'
SPIN_STATS = SOURCE / 'toprow_G2_beta_map_spatial_correlation.csv'
SPIN_NULL = SOURCE / 'toprow_G2_beta_map_spin_null.csv'
INK = '#252525'
MUTED = '#6A6A6A'
POINT = '#738999'
LINE = '#A95316'
INTERNAL_CMAP = LinearSegmentedColormap.from_list('refined_ocean_terracotta', [(0.0, '#244A5A'), (0.22, '#4F7E8D'), (0.43, '#B8CFD2'), (0.5, '#F7F3EA'), (0.57, '#EBC6AD'), (0.78, '#C9785B'), (1.0, '#8F3F32')])
CROSS_CMAP = LinearSegmentedColormap.from_list('refined_aubergine_moss', [(0.0, '#49345E'), (0.22, '#78618C'), (0.43, '#C9BDD5'), (0.5, '#F7F3EA'), (0.57, '#D8DEB3'), (0.78, '#93A55F'), (1.0, '#556934')])

def add_surface(fig: plt.Figure, rect: list[float], image_path: Path) -> None:
    ax = fig.add_axes(rect)
    ax.imshow(mpimg.imread(image_path))
    ax.set_axis_off()

def map_specific_limit(values: np.ndarray) -> float:
    """Round each map's maximum absolute beta outward to 0.01."""
    return math.ceil(float(np.nanmax(np.abs(values))) * 100.0) / 100.0

def render_surface_hires(values: np.ndarray, stem: str, limit: float, cmap: mpl.colors.Colormap) -> Path:
    """Render a large, sharp 1 x 4 Schaefer-400 surface row without a colorbar."""
    cache = QA / 'surface_cache'
    cache.mkdir(parents=True, exist_ok=True)
    output = cache / f'{stem}.png'
    if output.exists():
        return output
    surfaces = fetch_fslr()
    (lh_surf, rh_surf) = surfaces['veryinflated']
    atlas = np.asarray(load_parcellation('schaefer', 400, join=True))
    vertex = np.full(atlas.shape, np.nan, dtype=float)
    for roi in range(1, 401):
        vertex[atlas == roi] = float(values[roi - 1])
    medial_wall = np.full(atlas.shape, np.nan, dtype=float)
    medial_wall[atlas == 0] = 1.0
    plot = Plot(lh_surf, rh_surf, views=['lateral', 'medial'], layout='row', zoom=1.45, brightness=0.72, size=(5200, 1050), background=(1, 1, 1))
    plot.add_layer(medial_wall, cmap=ListedColormap(['#CECECE']), color_range=(0, 1), cbar=False)
    plot.add_layer(vertex, cmap=cmap, color_range=(-limit, limit), cbar=False)
    plot.add_layer(atlas, cmap=ListedColormap(['#555555']), as_outline=True, cbar=False)
    surface_figure = plot.build()
    surface_figure.patch.set_facecolor('white')
    surface_figure.savefig(output, dpi=500, bbox_inches='tight', pad_inches=0.0, facecolor='white')
    plt.close(surface_figure)
    return output

def compute_spatial_statistics(source: pd.DataFrame) -> pd.DataFrame:
    internal = source['adjusted_TMS_beta_G2'].to_numpy(float)
    cross = source['main_tms_minus_external_adjusted_beta'].to_numpy(float)
    centroids = base.confirm.legacy.schaefer_sphere_centroids()
    (spin_result, spin_null) = base.confirm.bijective_spin(internal, cross, centroids)
    result = pd.DataFrame([{**spin_result, 'comparison': 'within-study adjusted G2 beta vs cross-cohort adjusted G2 beta', 'cross_cohort_causal_interpretation': False}])
    result.to_csv(SPIN_STATS, index=False)
    pd.DataFrame({'spin': np.arange(len(spin_null)), 'null_pearson_r': spin_null}).to_csv(SPIN_NULL, index=False)
    return result

def make_figure(source: pd.DataFrame, stats: pd.Series) -> plt.Figure:
    internal = source['adjusted_TMS_beta_G2'].to_numpy(float)
    cross = source['main_tms_minus_external_adjusted_beta'].to_numpy(float)
    internal_limit = map_specific_limit(internal)
    cross_limit = map_specific_limit(cross)
    internal_surface = render_surface_hires(internal, 'toprow_internal_all400_hires_refined_ocean_terracotta_row1x4', internal_limit, INTERNAL_CMAP)
    cross_surface = render_surface_hires(cross, 'toprow_cross_all400_hires_refined_aubergine_moss_row1x4', cross_limit, CROSS_CMAP)
    fig = plt.figure(figsize=(183 / 25.4, 150 / 25.4), facecolor='white')
    fig.text(0.02, 0.975, 'G2-normalization beta-map correspondence and VIS enrichment', fontsize=9.2, fontweight='bold', ha='left', va='top', color=INK)
    fig.text(0.018, 0.925, 'a', fontsize=9.5, fontweight='bold', va='top', color=INK)
    fig.text(0.265, 0.928, 'Main TMS vs Main Control', fontsize=7.5, fontweight='bold', ha='center', va='top', color=INK)
    fig.text(0.265, 0.903, 'within-study; fully adjusted', fontsize=5.4, ha='center', va='top', color=MUTED)
    fig.text(0.505, 0.925, 'b', fontsize=9.5, fontweight='bold', va='top', color=INK)
    fig.text(0.745, 0.928, 'Main TMS vs External', fontsize=7.5, fontweight='bold', ha='center', va='top', color=INK)
    fig.text(0.745, 0.903, 'cross-cohort; baseline-deviation adjusted', fontsize=5.4, ha='center', va='top', color=MUTED)
    add_surface(fig, [0.025, 0.66, 0.465, 0.205], internal_surface)
    add_surface(fig, [0.51, 0.66, 0.465, 0.205], cross_surface)
    cax_left = fig.add_axes([0.095, 0.625, 0.33, 0.014])
    cbar_left = fig.colorbar(mpl.cm.ScalarMappable(norm=Normalize(-internal_limit, internal_limit), cmap=INTERNAL_CMAP), cax=cax_left, orientation='horizontal')
    cbar_left.set_ticks([-internal_limit, 0, internal_limit])
    cbar_left.set_label('Within-study adjusted TMS beta', fontsize=5.7, labelpad=0.8)
    cbar_left.ax.tick_params(labelsize=5.1, length=1.5, width=0.45, pad=0.8)
    cbar_left.outline.set_linewidth(0.45)
    cax_right = fig.add_axes([0.575, 0.625, 0.33, 0.014])
    cbar_right = fig.colorbar(mpl.cm.ScalarMappable(norm=Normalize(-cross_limit, cross_limit), cmap=CROSS_CMAP), cax=cax_right, orientation='horizontal')
    cbar_right.set_ticks([-cross_limit, 0, cross_limit])
    cbar_right.set_label('Cross-cohort adjusted group beta', fontsize=5.7, labelpad=0.8)
    cbar_right.ax.tick_params(labelsize=5.1, length=1.5, width=0.45, pad=0.8)
    cbar_right.outline.set_linewidth(0.45)
    ax = fig.add_axes([0.055, 0.105, 0.43, 0.4])
    ax.set_box_aspect(1)
    common_limit = 0.62
    ax.add_patch(Rectangle((0, 0), common_limit, common_limit, facecolor='#F8EEEE', edgecolor='none', alpha=0.62, zorder=0))
    ax.add_patch(Rectangle((-common_limit, -common_limit), common_limit, common_limit, facecolor='#EEF3F8', edgecolor='none', alpha=0.72, zorder=0))
    both_positive = (internal > 0) & (cross > 0)
    both_negative = (internal < 0) & (cross < 0)
    point_colors = np.full(len(internal), '#BFC4C8', dtype=object)
    point_colors[both_negative] = '#2D6EA6'
    point_colors[both_positive] = '#B43C4A'
    ax.scatter(internal, cross, s=13, color=point_colors, alpha=0.68, edgecolor='white', linewidth=0.22, rasterized=False, zorder=3)
    ax.plot([-common_limit, common_limit], [-common_limit, common_limit], color='#A9A9A9', ls=(0, (2.2, 2.2)), lw=0.65, zorder=1)
    fit = linregress(internal, cross)
    x_line = np.linspace(internal.min(), internal.max(), 240)
    y_line = fit.intercept + fit.slope * x_line
    residuals = cross - (fit.intercept + fit.slope * internal)
    residual_se = np.sqrt(np.sum(residuals ** 2) / (len(internal) - 2))
    centered_sum_squares = np.sum((internal - internal.mean()) ** 2)
    critical = student_t.ppf(0.975, len(internal) - 2)
    confidence = critical * residual_se * np.sqrt(1 / len(internal) + (x_line - internal.mean()) ** 2 / centered_sum_squares)
    ax.fill_between(x_line, y_line - confidence, y_line + confidence, color='#353535', alpha=0.1, linewidth=0, zorder=2)
    ax.plot(x_line, y_line, color='#353535', lw=1.15, zorder=4)
    ax.axhline(0, color='#8E8E8E', lw=0.55, zorder=1)
    ax.axvline(0, color='#8E8E8E', lw=0.55, zorder=1)
    ax.set_xlim(-common_limit, common_limit)
    ax.set_ylim(-common_limit, common_limit)
    ticks = [-0.6, -0.3, 0, 0.3, 0.6]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel('Within-study adjusted TMS beta')
    ax.set_ylabel('Cross-cohort adjusted group beta')
    ax.set_title('Spatial correspondence across 400 ROIs', loc='left', fontsize=7.3, fontweight='bold', pad=5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=5.6, length=2.6, width=0.6)
    ax.text(-0.11, 1.1, 'c', transform=ax.transAxes, fontsize=9.5, fontweight='bold', va='top', color=INK)
    ax.text(0.78, 0.86, 'both positive', transform=ax.transAxes, fontsize=4.9, color='#A63845', ha='center')
    ax.text(0.28, 0.08, 'both negative', transform=ax.transAxes, fontsize=4.9, color='#245F93', ha='center')
    p_spin = float(stats['spin_p_two_sided'])
    spin_text = 'spin p < 0.001' if p_spin < 0.001 else f'spin p = {p_spin:.3f}'
    same_sign_n = int(np.sum(both_positive | both_negative))
    ax.text(0.03, 0.97, f"r = {float(stats['pearson_r']):.3f}\n{spin_text}\nsame sign: {same_sign_n}/400", transform=ax.transAxes, ha='left', va='top', fontsize=5.7, color=MUTED, linespacing=1.35, bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.9, 'pad': 1.6})
    network_stats_path = SOURCE / 'G2_positive_spintest_main_vs_cross_cohort_statistics.csv'
    network_null_path = SOURCE / 'G2_positive_spintest_main_vs_cross_cohort_nulls.npz'
    if not network_stats_path.exists() or not network_null_path.exists():
        raise FileNotFoundError('Run run_G2_positive_spintest_main_vs_cross_cohort.py before composing the figure.')
    network_stats = pd.read_csv(network_stats_path)
    with np.load(network_null_path) as network_nulls:
        within_null = network_nulls['within'].copy()
        cross_null = network_nulls['cross'].copy()
    within_stats = network_stats.loc[network_stats['contrast'].eq('Within-study')].reset_index(drop=True)
    cross_stats = network_stats.loc[network_stats['contrast'].eq('Cross-cohort')].reset_index(drop=True)
    ax_network = fig.add_axes([0.555, 0.165, 0.425, 0.255])
    g2_spin.draw_paired_spin_panel(ax_network, within_stats, within_null, cross_stats, cross_null, q_note_y=-0.25, method_note_y=-0.35)
    ax_network.text(-0.11, 1.18, 'd', transform=ax_network.transAxes, fontsize=9.5, fontweight='bold', va='top', color=INK)
    fig.text(0.5, 0.016, 'Cross-cohort: no FDR-significant ROIs; network enrichment is not defined. Within-study spin results are shown.', fontsize=4.8, color=MUTED, ha='center', va='bottom')
    return fig

def export(fig: plt.Figure) -> list[Path]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FORAI.mkdir(parents=True, exist_ok=True)
    outputs = [STEM.with_suffix(ext) for ext in ('.pdf', '.svg', '.png', '.tiff')]
    fig.savefig(outputs[0], dpi=600)
    fig.savefig(outputs[1], dpi=600)
    fig.savefig(outputs[2], dpi=400)
    fig.savefig(outputs[3], dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)
    for path in outputs:
        (FORAI / path.name).write_bytes(path.read_bytes())
    return outputs

def main() -> None:
    base.apply_style()
    source = pd.read_csv(INPUT).sort_values('roi').reset_index(drop=True)
    if len(source) != 400 or not np.array_equal(source['roi'].to_numpy(), np.arange(1, 401)):
        raise ValueError('Expected ROIs 1-400 in canonical order.')
    source[['roi', 'roi_name', 'network', 'adjusted_TMS_beta_G2', 'main_tms_minus_external_adjusted_beta']].to_csv(SOURCE / 'toprow_G2_beta_map_spatial_source_data.csv', index=False)
    if SPIN_STATS.exists() and SPIN_NULL.exists():
        stats = pd.read_csv(SPIN_STATS).iloc[0]
    else:
        stats = compute_spatial_statistics(source).iloc[0]
    outputs = export(make_figure(source, stats))
    print(f"Pearson r={float(stats['pearson_r']):.6f}; Spearman rho={float(stats['spearman_rho']):.6f}; spin p={float(stats['spin_p_two_sided']):.6f}")
    for output in outputs:
        print(output)
if __name__ == '__main__':
    main()
