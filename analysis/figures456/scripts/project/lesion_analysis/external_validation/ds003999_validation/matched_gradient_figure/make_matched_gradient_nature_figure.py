from __future__ import annotations
import os as _portable_os
from pathlib import Path as _PortablePath

def _portable_path(kind, relative):
    project = _PortablePath(_portable_os.environ['PAPER_PROJECT_ROOT'])
    root = project if kind == 'project' else project.parent
    return str(root / relative)
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'
import matplotlib as mpl
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from brainspace.datasets import load_parcellation
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from neuromaps.datasets import fetch_fslr
from PIL import Image
from scipy.stats import gaussian_kde, t as t_dist
from surfplot import Plot
HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))
import run_external_validation_analysis as base
FIGURES = HERE / 'figures' / 'nature'
SURFACE_CACHE = HERE / 'qa' / 'surface_cache'
TABLES = HERE / 'tables'
SOURCE = HERE / 'source_data'
BLUE = '#2166AC'
BLUE_LIGHT = '#92C5DE'
RED = '#B2182B'
RED_LIGHT = '#F4A582'
GOLD = '#D9A400'
CHARCOAL = '#272727'
MID = '#767676'
LIGHT = '#D8D8D8'
POST = '#3775BA'
VIOLET = '#7C6CCF'
GAIN = '#3C7D68'
RADAR_NEGATIVE = '#877CB3'
RADAR_POSITIVE = '#B76E79'
RADAR_SPOKE = '#D0D0D0'
RADAR_ZERO = '#5A535B'
NETWORK_ORDER = list(base.NETWORK_ORDER)
NETWORK_ABBR = {'Visual': 'VIS', 'SomMot': 'SMN', 'DorsAttn': 'DAN', 'SalVentAttn': 'VAN', 'Limbic': 'LIM', 'Control': 'CON', 'Default': 'DMN'}
METRIC_LABEL = {'Dispersion': 'Dispersion', 'betweenDisp': 'Between-net dispersion', 'Segregation': 'Segregation'}
ML_ORDER = ['G1_VIS_organization', 'G2_VIS_organization', 'G1_global_spread', 'G2_global_spread', 'G1_network_organization', 'G2_network_organization']
ML_LABELS = {'G1_VIS_organization': 'G1 VIS organization', 'G2_VIS_organization': 'G2 VIS organization', 'G1_global_spread': 'G1 global spread', 'G2_global_spread': 'G2 global spread', 'G1_network_organization': 'G1 seven-network organization', 'G2_network_organization': 'G2 seven-network organization'}
T_CMAP = LinearSegmentedColormap.from_list('nature_t', ['#053061', '#2166AC', '#92C5DE', '#F7F7F7', '#F4A582', '#B2182B', '#67001F'], N=256)

def apply_style(font_size: float=8.0) -> None:
    plt.rcParams.update({'font.size': font_size, 'axes.labelsize': font_size, 'axes.titlesize': font_size + 0.5, 'xtick.labelsize': font_size - 0.5, 'ytick.labelsize': font_size - 0.5, 'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.unicode_minus': False})

def save_all(fig: plt.Figure, stem: str, dpi: int=600) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    base_path = FIGURES / stem
    fig.savefig(base_path.with_suffix('.svg'), bbox_inches='tight', pad_inches=0.035)
    fig.savefig(base_path.with_suffix('.pdf'), bbox_inches='tight', pad_inches=0.035)
    fig.savefig(base_path.with_suffix('.png'), dpi=300, bbox_inches='tight', pad_inches=0.035)
    tiff_path = base_path.with_suffix('.tiff')
    fig.savefig(tiff_path, dpi=dpi, bbox_inches='tight', pad_inches=0.035, pil_kwargs={'compression': 'tiff_lzw'})
    with Image.open(tiff_path) as image:
        rgb = image.convert('RGB')
        rgb.save(tiff_path, dpi=(dpi, dpi), compression='tiff_lzw')
    plt.close(fig)

def add_panel_label(ax: plt.Axes, label: str, x: float=-0.08, y: float=1.04) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=11, fontweight='bold', ha='left', va='bottom', color=CHARCOAL)

def roi_to_vertices(values: np.ndarray) -> np.ndarray:
    atlas = np.asarray(load_parcellation('schaefer', 400, join=True))
    output = np.zeros_like(atlas, dtype=float)
    for roi in range(1, 401):
        output[atlas == roi] = values[roi - 1]
    return output

def render_surface(roi_values: np.ndarray, cache_name: str, cmap, color_range: tuple[float, float], fdr_mask: np.ndarray | None=None, mask_zeros: bool=False) -> Path:
    SURFACE_CACHE.mkdir(parents=True, exist_ok=True)
    out = SURFACE_CACHE / f'{cache_name}.png'
    surfaces = fetch_fslr()
    (lh_surf, rh_surf) = surfaces['veryinflated']
    atlas = np.asarray(load_parcellation('schaefer', 400, join=True))
    medial_wall = np.full_like(atlas, np.nan, dtype=float)
    medial_wall[atlas == 0] = 1.0
    plot = Plot(lh_surf, rh_surf, views=['lateral', 'medial'], size=(6400, 4200), zoom=1.7, background=(1, 1, 1), brightness=1.0 if mask_zeros else 0.72)
    if mask_zeros:
        cortex_base = np.full_like(atlas, np.nan, dtype=float)
        cortex_base[atlas > 0] = 1.0
        plot.add_layer(cortex_base, cmap=ListedColormap(['#FFFFFF']), color_range=(0, 1), cbar=False)
    vertices = roi_to_vertices(roi_values)
    plot.add_layer(medial_wall, cmap=ListedColormap(['#D1D1D1']), color_range=(0, 1), cbar=False)
    plot.add_layer(vertices, cmap=cmap, color_range=list(color_range), cbar=False, zero_transparent=mask_zeros)
    plot.add_layer(atlas, cmap=ListedColormap(['#606060']), as_outline=True, cbar=False)
    if fdr_mask is not None and np.any(fdr_mask):
        fdr_vertices = roi_to_vertices(np.asarray(fdr_mask, dtype=float))
        plot.add_layer(fdr_vertices, cmap=ListedColormap([GOLD]), as_outline=True, cbar=False, alpha=1.0, zero_transparent=True)
    built = plot.build()
    built.patch.set_facecolor('white')
    built.set_size_inches(12.0, 8.0, forward=True)
    brain_axes = built.axes[:4]
    if len(brain_axes) >= 4:
        positions = [[0.025, 0.505, 0.465, 0.455], [0.51, 0.505, 0.465, 0.455], [0.025, 0.035, 0.465, 0.455], [0.51, 0.035, 0.465, 0.455]]
        for (axis, position) in zip(brain_axes, positions):
            axis.set_position(position)
    built.savefig(out, dpi=400, bbox_inches=None, pad_inches=0.0, facecolor='white')
    plt.close(built)
    return out

def surface_axis(ax: plt.Axes, image_path: Path, title: str, vlim: float, note: str | None=None, panel_label: str | None=None, colorbar_label: str='paired t (post − pre)') -> None:
    ax.imshow(mpimg.imread(image_path))
    ax.set_axis_off()
    ax.set_title(title, loc='left', fontweight='bold', pad=0.5)
    if panel_label:
        add_panel_label(ax, panel_label, x=-0.085, y=1.01)
    cax = inset_axes(ax, width='44%', height='4.2%', loc='lower center', bbox_to_anchor=(0, -0.115, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)
    cb = ax.figure.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=T_CMAP), cax=cax, orientation='horizontal')
    cb.set_ticks([-vlim, 0, vlim])
    cb.set_label(colorbar_label, labelpad=1.5)
    cb.ax.xaxis.set_label_position('top')
    cb.ax.tick_params(length=2, pad=1)
    cb.outline.set_linewidth(0.5)
    if note:
        ax.text(0.5, -0.235, note, transform=ax.transAxes, ha='center', va='top', fontsize=6.7, color=MID)

def load_gradients() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    participants = pd.read_csv(base.PARTICIPANTS, sep='\t')
    labels = base.load_labels()
    (pre, post) = ([], [])
    for sid in participants['participant_id']:
        pre.append(np.load(base.locate_array(base.LEGACY_ROOT, 'pre', sid)))
        post.append(np.load(base.locate_array(base.LEGACY_ROOT, 'post', sid)))
    return (participants, labels, np.stack(pre), np.stack(post))

def make_t_surfaces(roi_stats: pd.DataFrame) -> tuple[dict[str, Path], float]:
    tmax = float(np.ceil(10 * roi_stats['t'].abs().max()) / 10)
    images: dict[str, Path] = {}
    for gradient in ('G1', 'G2'):
        frame = roi_stats.query('gradient == @gradient').sort_values('roi')
        values = np.where(frame['nominal_p_lt_0_05'], frame['t'], 0.0)
        images[gradient] = render_surface(values, f'{gradient}_paired_t_nominal_reference_masked_surface_hd', T_CMAP, (-tmax, tmax), mask_zeros=True)
    return (images, tmax)

def make_clinical_beta_surfaces(models: pd.DataFrame) -> tuple[dict[str, Path], float]:
    significant = models.loc[models['hc3_p_two_sided'] < 0.05]
    beta_limit = float(np.ceil(significant['standardized_beta'].abs().max() * 10) / 10)
    images: dict[str, Path] = {}
    for gradient in ('G1', 'G2'):
        frame = models.query('gradient == @gradient').sort_values('roi')
        values = np.where(frame['hc3_p_two_sided'] < 0.05, frame['standardized_beta'], 0.0)
        images[gradient] = render_surface(values, f'{gradient}_clinical_beta_nominal_p005_reference_style', T_CMAP, (-beta_limit, beta_limit), mask_zeros=True)
    return (images, beta_limit)

def make_mean_surfaces(pre: np.ndarray, post: np.ndarray) -> dict[str, tuple[Path, Path, float]]:
    mean_values: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for (gidx, gradient) in enumerate(('G1', 'G2')):
        mean_values[gradient] = (np.mean(pre[:, :, gidx], axis=0), np.mean(post[:, :, gidx], axis=0))
    limit = float(max((np.max(np.abs(values)) for pair in mean_values.values() for values in pair)))
    outputs: dict[str, tuple[Path, Path, float]] = {}
    for (gradient, (pre_mean, post_mean)) in mean_values.items():
        pre_img = render_surface(pre_mean, f'{gradient}_pre_mean', T_CMAP, (-limit, limit))
        post_img = render_surface(post_mean, f'{gradient}_post_mean', T_CMAP, (-limit, limit))
        outputs[gradient] = (pre_img, post_img, limit)
    return outputs

def plot_radar(ax: plt.Axes, counts: pd.DataFrame, gradient: str, show_legend: bool=False) -> None:
    frame = counts.query('gradient == @gradient').set_index('network').loc[NETWORK_ORDER]
    positive = frame['nominal_positive_n'].to_numpy(float)
    negative = frame['nominal_negative_n'].to_numpy(float)
    angles = np.linspace(0, 2 * np.pi, len(NETWORK_ORDER), endpoint=False)
    closed_angles = np.r_[angles, angles[0]]
    count_max = max(1.0, float(max(positive.max(), negative.max())))
    zero_radius = count_max * 0.5
    positive_r = zero_radius + positive / count_max * (count_max - zero_radius)
    negative_r = zero_radius + negative / count_max * (count_max - zero_radius)
    ax.set_ylim(0, count_max)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.grid(False)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.spines['polar'].set_color('#050505')
    ax.spines['polar'].set_linewidth(1.6)
    for angle in angles:
        ax.plot([angle, angle], [0, count_max], color=RADAR_SPOKE, lw=0.75, linestyle=(0, (3, 2)), zorder=0)
    theta = np.linspace(0, 2 * np.pi, 720)
    ax.plot(theta, np.full_like(theta, zero_radius), color=RADAR_ZERO, lw=1.05, linestyle=(0, (4, 2.5)), zorder=1)
    ax.fill(closed_angles, np.r_[negative_r, negative_r[0]], color=RADAR_NEGATIVE, alpha=0.16, zorder=2)
    ax.fill(closed_angles, np.r_[positive_r, positive_r[0]], color=RADAR_POSITIVE, alpha=0.16, zorder=3)
    ax.plot(closed_angles, np.r_[negative_r, negative_r[0]], color=RADAR_NEGATIVE, lw=1.75, marker='o', ms=3.4, markeredgewidth=0, zorder=4)
    ax.plot(closed_angles, np.r_[positive_r, positive_r[0]], color=RADAR_POSITIVE, lw=1.75, marker='o', ms=3.4, markeredgewidth=0, zorder=5)
    label_radius = 0.515
    for (angle, network) in zip(angles, NETWORK_ORDER):
        x = 0.5 + label_radius * np.sin(angle)
        y = 0.5 + label_radius * np.cos(angle)
        ha = 'left' if x > 0.56 else 'right' if x < 0.44 else 'center'
        va = 'bottom' if y > 0.56 else 'top' if y < 0.44 else 'center'
        ax.text(x, y, NETWORK_ABBR[network], transform=ax.transAxes, ha=ha, va=va, fontsize=6.2, color=CHARCOAL, clip_on=False)
    for (raw, displayed, color) in ((negative, negative_r, RADAR_NEGATIVE), (positive, positive_r, RADAR_POSITIVE)):
        if np.any(raw > 0):
            idx = int(np.argmax(raw))
            radius = float(displayed[idx])
            text_radius = radius - count_max * 0.08 if radius >= count_max * 0.96 else radius + count_max * 0.06
            ax.text(angles[idx], text_radius, str(int(raw[idx])), ha='center', va='top' if radius >= count_max * 0.96 else 'bottom', fontsize=5.8, fontweight='bold', color=color, clip_on=False, zorder=8)
    ax.set_title(gradient, y=1.1, fontweight='bold', fontsize=7.0)
    if show_legend:
        ax.legend(handles=[Patch(facecolor=RADAR_NEGATIVE, edgecolor='#333333', linewidth=0.35, label='negative t'), Patch(facecolor=RADAR_POSITIVE, edgecolor='#333333', linewidth=0.35, label='positive t')], loc='upper center', bbox_to_anchor=(1.05, -0.15), ncol=2, fontsize=6.3, handlelength=1.5)

def short_roi_name(name: str) -> str:
    return name.replace('7Networks_', '').replace('Default', 'DMN').replace('DorsAttn', 'DAN').replace('SomMot', 'SMN')

def plot_clinical_roi_forest(ax: plt.Axes, models: pd.DataFrame) -> None:
    top = models.loc[~((models['gradient'] == 'G2') & (models['roi'] == 172))].sort_values('hc3_p_two_sided').head(8).copy().iloc[::-1]
    y = np.arange(len(top))
    beta = top['standardized_beta'].to_numpy(float)
    lo = top['hc3_ci_low'].to_numpy(float)
    hi = top['hc3_ci_high'].to_numpy(float)
    colors = [VIOLET if g == 'G2' else POST for g in top['gradient']]
    for (yi, b, lower, upper, color) in zip(y, beta, lo, hi, colors):
        ax.errorbar(b, yi, xerr=[[b - lower], [upper - b]], fmt='none', ecolor=color, elinewidth=1.2, capsize=2)
    ax.scatter(beta, y, c=colors, s=25, edgecolor='white', linewidth=0.5, zorder=3)
    ax.axvline(0, color=LIGHT, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f'{g}  {short_roi_name(n)}' for (g, n) in zip(top['gradient'], top['roi_name'])], fontsize=5.7)
    ax.set_xlabel('Standardized β [HC3 95% CI]')
    ax.set_title('Additional exploratory clinical ROIs', loc='left', fontweight='bold')
    ax.text(1.0, 1.01, 'next-ranked regions; selected f ROI excluded; all q>0.05', transform=ax.transAxes, ha='right', fontsize=5.4, color=MID)
    add_panel_label(ax, 'g', x=-0.22)

def plot_roi_association_landscape(ax_top: plt.Axes, ax_bottom: plt.Axes, models: pd.DataFrame, add_label: bool=True) -> None:
    network_rank = {network: idx for (idx, network) in enumerate(NETWORK_ORDER)}
    ordered = models.copy()
    ordered['network_rank'] = ordered['network'].map(network_rank)
    ordered['hemisphere_rank'] = ordered['hemisphere'].map({'LH': 0, 'RH': 1})
    ordered = ordered.sort_values(['network_rank', 'hemisphere_rank', 'roi'])
    first = ordered.query("gradient == 'G1'")
    network_sequence = first['network'].to_numpy()
    boundaries = np.flatnonzero(network_sequence[1:] != network_sequence[:-1]) + 0.5
    centers = []
    for network in NETWORK_ORDER:
        idx = np.flatnonzero(network_sequence == network)
        centers.append(float(idx.mean()))
    for (ax, gradient) in ((ax_top, 'G1'), (ax_bottom, 'G2')):
        frame = ordered.query('gradient == @gradient').reset_index(drop=True)
        x = np.arange(len(frame))
        beta = frame['standardized_beta'].to_numpy(float)
        nominal = frame['hc3_p_two_sided'].to_numpy(float) < 0.05
        ax.scatter(x[~nominal], beta[~nominal], s=4.2, color='#C8C9CB', alpha=0.72, edgecolor='none', zorder=1)
        positive = nominal & (beta > 0)
        negative = nominal & (beta < 0)
        ax.scatter(x[positive], beta[positive], s=9, color=RED, edgecolor='white', linewidth=0.25, zorder=3)
        ax.scatter(x[negative], beta[negative], s=9, color=BLUE, edgecolor='white', linewidth=0.25, zorder=3)
        ax.axhline(0, color=MID, lw=0.55, zorder=0)
        for boundary in boundaries:
            ax.axvline(boundary, color=LIGHT, lw=0.45, zorder=0)
        ax.set_xlim(-5, 404)
        ax.set_ylim(-0.72, 0.72)
        ax.set_ylabel(f'{gradient}  β', labelpad=2)
        ax.tick_params(axis='y', labelsize=5.5, length=2)
        ax.spines['bottom'].set_visible(False)
        if gradient == 'G1':
            ax.set_xticks([])
            ax.set_title('Whole-cortex clinical-association landscape', loc='left', fontweight='bold', pad=2)
            ax.text(0.99, 0.91, 'red/blue: HC3 p<0.05; grey: other ROIs', transform=ax.transAxes, ha='right', va='top', fontsize=5.2, color=MID, bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.82, 'pad': 0.6})
            if add_label:
                add_panel_label(ax, 'g', x=-0.15, y=1.18)
        else:
            ax.set_xticks(centers)
            ax.set_xticklabels([NETWORK_ABBR[n] for n in NETWORK_ORDER], fontsize=5.4)
            ax.tick_params(axis='x', length=0, pad=1)
            ax.set_xlabel('Schaefer400 ROIs grouped by network', labelpad=2)

def plot_external_ml(ax: plt.Axes, summary: pd.DataFrame, add_label: bool=True) -> None:
    frame = summary.set_index('feature_family').loc[ML_ORDER].copy().iloc[::-1]
    y = np.arange(len(frame))
    means = frame['plus_clinical_mean_mae'].to_numpy(float)
    combined_lo = frame['plus_clinical_mae_percentile_2.5'].to_numpy(float)
    combined_hi = frame['plus_clinical_mae_percentile_97.5'].to_numpy(float)
    clinical_reference = float(frame['clinical_only_mean_mae'].iloc[0])
    best_family = frame['plus_clinical_mean_mae'].idxmin()
    for (family, yi, mean, lower, upper) in zip(frame.index, y, means, combined_lo, combined_hi):
        color = GAIN if family == best_family else POST
        ax.errorbar(mean, yi, xerr=[[mean - lower], [upper - mean]], fmt='o', ms=4.2, color=color, ecolor=color, capsize=1.8, lw=1.0, zorder=2)
        ax.text(upper + 0.08, yi, f'{mean:.2f}', ha='left', va='center', fontsize=5.0, color=color, fontweight='bold' if family == best_family else 'normal')
    ax.axvline(clinical_reference, color=MID, lw=0.9, ls='--', zorder=0)
    ax.text(clinical_reference + 0.05, len(frame) - 0.62, f'clinical-only reference\nMAE {clinical_reference:.2f}', ha='left', va='top', fontsize=4.8, color=MID, bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.84, 'pad': 0.5})
    ax.set_yticks(y)
    ax.set_yticklabels([ML_LABELS[x] for x in frame.index], fontsize=5.7)
    ax.set_ylim(-1.2, len(frame) - 0.35)
    ax.set_xlim(max(0.0, min(float(combined_lo.min()), clinical_reference) - 0.25), float(combined_hi.max()) + 0.55)
    ax.grid(axis='x', color='#ECECEC', lw=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel('Repeated nested-CV MAE (lower is better)')
    ax.set_title('Clinical-integrated external prediction', loc='left', fontweight='bold', pad=4)
    ax.text(1.0, 0.01, 'all feature groups include pre-score, time and sex; n=29\nwhiskers: repeat 2.5-97.5%; 20 x nested 5-fold CV', transform=ax.transAxes, ha='right', va='bottom', fontsize=5.0, color=MID)
    if add_label:
        add_panel_label(ax, 'h', x=-0.22)

def make_ml_r2_standalone(summary: pd.DataFrame) -> None:
    frame = summary.set_index('feature_family').loc[ML_ORDER].copy().iloc[::-1]
    y = np.arange(len(frame))
    means = frame['plus_clinical_mean_r2'].to_numpy(float)
    lower = frame['plus_clinical_r2_percentile_2.5'].to_numpy(float)
    upper = frame['plus_clinical_r2_percentile_97.5'].to_numpy(float)
    clinical_reference = float(frame['clinical_only_mean_r2'].iloc[0])
    best_family = frame['plus_clinical_mean_r2'].idxmax()
    (fig, ax) = plt.subplots(figsize=(6.4, 4.1))
    for (family, yi, mean, lo, hi) in zip(frame.index, y, means, lower, upper):
        color = GAIN if family == best_family else POST
        ax.errorbar(mean, yi, xerr=[[mean - lo], [hi - mean]], fmt='o', ms=5.0, color=color, ecolor=color, capsize=2.2, lw=1.2)
        ax.text(hi + 0.004, yi, f'{mean:.3f}', ha='left', va='center', fontsize=6.2, color=color, fontweight='bold' if family == best_family else 'normal')
    ax.axvline(clinical_reference, color=MID, lw=1.0, ls='--')
    ax.text(clinical_reference - 0.004, len(frame) - 0.5, f'clinical-only reference\nR² {clinical_reference:.3f}', ha='right', va='top', fontsize=6.2, color=MID, bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.88, 'pad': 0.7})
    ax.set_yticks(y)
    ax.set_yticklabels([ML_LABELS[x] for x in frame.index], fontsize=7.0)
    ax.set_ylim(-1.05, len(frame) - 0.3)
    ax.set_xlim(min(0.5, float(lower.min()) - 0.02), max(0.95, float(upper.max()) + 0.03))
    ax.grid(axis='x', color='#ECECEC', lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel('Repeated nested-CV R² (higher is better)')
    ax.set_title('Clinical-integrated external prediction: explained variance', loc='left', fontweight='bold')
    ax.text(1.0, 0.01, 'all feature groups include pre-score, time and sex; n=29\nwhiskers: repeat 2.5-97.5%; 20 x nested 5-fold CV', transform=ax.transAxes, ha='right', va='bottom', fontsize=6.0, color=MID)
    fig.subplots_adjust(left=0.27, right=0.96, top=0.9, bottom=0.18)
    save_all(fig, 'ml_clinical_integrated_R2_performance')

def plot_metric_heatmap(ax: plt.Axes, effects: pd.DataFrame, add_label: bool=True, colorbar_ax: plt.Axes | None=None) -> None:
    row_keys = [(g, m) for g in ('G1', 'G2') for m in ('Dispersion', 'betweenDisp', 'Segregation')]
    matrix = np.zeros((len(row_keys), len(NETWORK_ORDER)))
    pmat = np.ones_like(matrix)
    for (i, (gradient, metric)) in enumerate(row_keys):
        frame = effects.query('gradient == @gradient and metric == @metric').set_index('network')
        for (j, network) in enumerate(NETWORK_ORDER):
            matrix[i, j] = frame.loc[network, 'paired_cohen_dz']
            pmat[i, j] = frame.loc[network, 'p_signflip_two_sided']
    limit = max(0.55, float(np.ceil(np.max(np.abs(matrix)) * 10) / 10))
    im = ax.imshow(matrix, cmap=T_CMAP, norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), aspect='auto')
    ax.set_xticks(range(len(NETWORK_ORDER)))
    ax.set_xticklabels([NETWORK_ABBR[n] for n in NETWORK_ORDER], rotation=35, ha='right')
    ax.set_yticks(range(len(row_keys)))
    ax.set_yticklabels([f'{g}  {METRIC_LABEL[m]}' for (g, m) in row_keys], fontsize=6.6)
    ax.tick_params(length=0)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if pmat[i, j] < 0.05:
                ax.text(j, i, '*', ha='center', va='center', color='black', fontsize=9, fontweight='bold')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.axhline(2.5, color='white', lw=2.0)
    ax.set_title('Seven-network organization', loc='left', fontweight='bold')
    cax = colorbar_ax
    if cax is None:
        cax = inset_axes(ax, width='55%', height='5%', loc='lower center', bbox_to_anchor=(0, -0.2, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
    cb = ax.figure.colorbar(im, cax=cax, orientation='horizontal')
    cb.set_label('paired Cohen $d_z$ (post − pre)', labelpad=1)
    cb.ax.tick_params(length=2, pad=1)
    cb.outline.set_linewidth(0.5)
    if add_label:
        add_panel_label(ax, 'd', x=-0.18, y=1.03)

def plot_component_variance(ax: plt.Axes, long: pd.DataFrame, tests: pd.DataFrame, add_label: bool=True) -> None:
    comps = [f'G{k}' for k in range(1, 6)]
    x = np.arange(5)
    for (k, comp) in enumerate(comps):
        pre_vals = long.query("phase == 'pre' and component == @comp")['share_of_retained_G1_G5'].to_numpy(float)
        post_vals = long.query("phase == 'post' and component == @comp")['share_of_retained_G1_G5'].to_numpy(float)
        for (a, b) in zip(pre_vals, post_vals):
            ax.plot([x[k] - 0.11, x[k] + 0.11], [a, b], color=LIGHT, lw=0.45, alpha=0.55, zorder=1)
        for (vals, xpos, color, label) in ((pre_vals, x[k] - 0.11, CHARCOAL, 'Pre'), (post_vals, x[k] + 0.11, POST, 'Post')):
            mean = np.mean(vals)
            ci = t_dist.ppf(0.975, len(vals) - 1) * np.std(vals, ddof=1) / np.sqrt(len(vals))
            ax.errorbar(xpos, mean, yerr=ci, fmt='o', ms=4, color=color, ecolor=color, elinewidth=1.1, capsize=2, zorder=3, label=label if k == 0 else None)
    for (k, row) in tests.set_index('component').loc[comps].iterrows():
        if row['fdr_q_5_components'] < 0.05:
            idx = comps.index(k)
            ytop = long.query('component == @k')['share_of_retained_G1_G5'].max()
            ax.text(idx, ytop + 0.025, 'q<0.05', ha='center', fontsize=5.8)
    ax.set_xticks(x)
    ax.set_xticklabels(comps)
    ax.set_ylabel('Share of retained G1–G5 eigenvalues')
    ax.set_title('Gradient component variance', loc='left', fontweight='bold')
    ax.legend(loc='upper right', ncol=2, fontsize=6.5, handletextpad=0.3, columnspacing=0.8)
    ax.tick_params(length=3)
    if add_label:
        add_panel_label(ax, 'e')

def plot_clinical(ax: plt.Axes, subjects: pd.DataFrame, selected: pd.Series, add_label: bool=True) -> None:
    x = subjects['adjusted_G2_delta_residual'].to_numpy(float)
    y = subjects['adjusted_improvement_residual'].to_numpy(float)
    ax.scatter(x, y, s=24, color=VIOLET, edgecolor='white', linewidth=0.55, alpha=0.88, zorder=3)
    xx = np.linspace(x.min() - 0.08, x.max() + 0.08, 100)
    ax.plot(xx, selected['standardized_beta'] * xx, color=CHARCOAL, lw=1.5, zorder=2)
    ax.axhline(0, color=LIGHT, lw=0.7, zorder=0)
    ax.axvline(0, color=LIGHT, lw=0.7, zorder=0)
    ax.set_xlabel('Adjusted G2 change\n(LH Default PFC 7)')
    ax.set_ylabel('Adjusted Vasserman improvement')
    ax.set_title('Exploratory clinical bridge', loc='left', fontweight='bold')
    ax.text(0.03, 0.97, f"β={selected['standardized_beta']:.2f}  [{selected['hc3_ci_low']:.2f}, {selected['hc3_ci_high']:.2f}]\nwhole-cortex q={selected['fdr_q_400']:.3f}\ntargeted permutation p={selected['studentized_freedman_lane_p_two_sided']:.4f}", transform=ax.transAxes, ha='left', va='top', fontsize=6.4, color=CHARCOAL)
    ax.text(0.98, 0.03, 'data-selected ROI', transform=ax.transAxes, ha='right', va='bottom', fontsize=6.1, color=RED)
    if add_label:
        add_panel_label(ax, 'f')

def make_main(t_images: dict[str, Path], tmax: float, roi_stats: pd.DataFrame, counts: pd.DataFrame, effects: pd.DataFrame, component_long: pd.DataFrame, component_tests: pd.DataFrame, clinical_subjects: pd.DataFrame, selected: pd.Series, clinical_models: pd.DataFrame, ml_summary: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(7.2, 10.5), facecolor='white')
    gs = fig.add_gridspec(4, 2, height_ratios=[1.25, 1.0, 1.0, 1.12], hspace=0.76, wspace=0.5)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    g1 = roi_stats.query("gradient == 'G1'")
    g2 = roi_stats.query("gradient == 'G2'")
    surface_axis(ax_a, t_images['G1'], 'G1 longitudinal ROI effects', tmax, note=f"{int(g1['nominal_p_lt_0_05'].sum())} ROIs at two-sided paired p<0.05 (uncorrected)", panel_label='a')
    surface_axis(ax_b, t_images['G2'], 'G2 longitudinal ROI effects', tmax, note=f"{int(g2['nominal_p_lt_0_05'].sum())} ROIs at two-sided paired p<0.05 (uncorrected)", panel_label='b')
    radar_grid = gs[1, 0].subgridspec(1, 2, wspace=0.88)
    ax_c1 = fig.add_subplot(radar_grid[0, 0], projection='polar')
    ax_c2 = fig.add_subplot(radar_grid[0, 1], projection='polar')
    plot_radar(ax_c1, counts, 'G1', show_legend=True)
    plot_radar(ax_c2, counts, 'G2', show_legend=False)
    add_panel_label(ax_c1, 'c', x=-0.4, y=1.4)
    ax_c1.text(-0.27, 1.4, 'Nominal ROI counts by network', transform=ax_c1.transAxes, fontweight='bold', fontsize=8.5)
    ax_d = fig.add_subplot(gs[1, 1])
    plot_metric_heatmap(ax_d, effects, add_label=True)
    ax_e = fig.add_subplot(gs[2, 0])
    plot_component_variance(ax_e, component_long, component_tests, add_label=True)
    ax_f = fig.add_subplot(gs[2, 1])
    plot_clinical(ax_f, clinical_subjects, selected, add_label=True)
    landscape_grid = gs[3, 0].subgridspec(2, 1, hspace=0.13)
    ax_g1 = fig.add_subplot(landscape_grid[0, 0])
    ax_g2 = fig.add_subplot(landscape_grid[1, 0], sharex=ax_g1)
    plot_roi_association_landscape(ax_g1, ax_g2, clinical_models)
    ax_h = fig.add_subplot(gs[3, 1])
    plot_external_ml(ax_h, ml_summary)
    fig.suptitle('Independent longitudinal gradient analysis in ds003999', x=0.07, y=0.995, ha='left', fontsize=11, fontweight='bold')
    fig.text(0.07, 0.972, 'All panels use E:/paper/lesion_refresh_20261028/work/Gradient/new_dataset (n=29); single-arm cohort with no TMS/control contrast', ha='left', va='top', fontsize=7.2, color=MID)
    fig.text(0.1, 0.018, 'Voxelwise lesion masks are unavailable; lesion–gradient distance and lesion-informed models cannot be tested in this cohort.', fontsize=5.8, color=MID)
    fig.subplots_adjust(top=0.92, bottom=0.075, left=0.1, right=0.97)
    save_all(fig, 'matched_gradient_validation_main')

def make_standalone_surfaces(t_images: dict[str, Path], tmax: float, roi_stats: pd.DataFrame) -> None:
    for gradient in ('G1', 'G2'):
        frame = roi_stats.query('gradient == @gradient')
        (fig, ax) = plt.subplots(figsize=(7.2, 2.75))
        if gradient == 'G1':
            fdr = frame.query('fdr_q_lt_0_05').iloc[0]
            note = f"Colored: two-sided paired p<0.05; gold outline: FDR ROI {fdr['roi_name']} (q={fdr['fdr_q_400']:.3f})"
        else:
            note = f"Colored: two-sided paired p<0.05; no ROI survived BH-FDR (minimum q={frame['fdr_q_400'].min():.3f})"
        surface_axis(ax, t_images[gradient], f'{gradient}: longitudinal paired ROI effects (n=29)', tmax, note=note)
        fig.subplots_adjust(top=0.88, bottom=0.25, left=0.02, right=0.98)
        save_all(fig, f'{gradient}_paired_t_surface_standalone')

def make_mean_surface_figures(mean_images: dict[str, tuple[Path, Path, float]]) -> None:
    for (gradient, (pre_img, post_img, limit)) in mean_images.items():
        (fig, axes) = plt.subplots(2, 1, figsize=(7.2, 4.5))
        for (ax, image_path, phase) in zip(axes, (pre_img, post_img), ('Pre', 'Post')):
            ax.imshow(mpimg.imread(image_path))
            ax.set_axis_off()
            ax.set_title(phase, loc='left', fontweight='bold', pad=0)
        cax = fig.add_axes([0.34, 0.035, 0.32, 0.018])
        cb = fig.colorbar(mpl.cm.ScalarMappable(norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), cmap=T_CMAP), cax=cax, orientation='horizontal')
        cb.set_label(f'Mean raw {gradient} coordinate', labelpad=1)
        cb.set_ticks([-limit, 0, limit])
        cb.ax.tick_params(length=2, pad=1)
        cb.outline.set_linewidth(0.5)
        fig.suptitle(f'{gradient}: group-mean pre/post gradient surfaces', x=0.08, y=0.99, ha='left', fontsize=10, fontweight='bold')
        save_all(fig, f'{gradient}_mean_pre_post_surfaces')

def make_radar_standalone(counts: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(6.2, 3.25))
    gs = fig.add_gridspec(1, 2, wspace=0.42)
    axes = [fig.add_subplot(gs[0, i], projection='polar') for i in range(2)]
    plot_radar(axes[0], counts, 'G1', show_legend=True)
    plot_radar(axes[1], counts, 'G2', show_legend=False)
    fig.suptitle('Nominally significant longitudinal ROIs by network', x=0.08, y=0.985, ha='left', fontweight='bold', fontsize=10)
    fig.text(0.08, 0.92, 'Two-sided paired p<0.05; counts are descriptive unless FDR is stated', fontsize=7, color=MID)
    fig.subplots_adjust(top=0.82, bottom=0.16, left=0.07, right=0.97)
    save_all(fig, 'network_nominal_roi_count_radar')

def make_heatmap_standalone(effects: pd.DataFrame) -> None:
    (fig, ax) = plt.subplots(figsize=(7.2, 3.45))
    plot_metric_heatmap(ax, effects, add_label=False)
    fig.subplots_adjust(left=0.23, right=0.98, top=0.9, bottom=0.26)
    save_all(fig, 'network_metric_effect_size_heatmap')

def make_component_standalone(component_long: pd.DataFrame, component_tests: pd.DataFrame) -> None:
    (fig, ax) = plt.subplots(figsize=(5.0, 3.4))
    plot_component_variance(ax, component_long, component_tests, add_label=False)
    fig.subplots_adjust(left=0.14, right=0.97, top=0.9, bottom=0.16)
    save_all(fig, 'component_variance_pre_post')

def make_clinical_standalone(subjects: pd.DataFrame, selected: pd.Series) -> None:
    (fig, ax) = plt.subplots(figsize=(4.3, 3.65))
    plot_clinical(ax, subjects, selected, add_label=False)
    fig.subplots_adjust(left=0.18, right=0.96, top=0.91, bottom=0.18)
    save_all(fig, 'exploratory_G2_clinical_partial_regression')

def make_ridge_standalone(centroids: pd.DataFrame) -> None:
    (fig, axes) = plt.subplots(1, 2, figsize=(7.2, 4.6), sharey=True)
    for (ax, gradient) in zip(axes, ('G1', 'G2')):
        frame_g = centroids.query('gradient == @gradient')
        (low, high) = np.nanpercentile(np.r_[frame_g['pre'], frame_g['post']], [0.5, 99.5])
        pad = 0.08 * (high - low)
        xgrid = np.linspace(low - pad, high + pad, 300)
        for (row, network) in enumerate(NETWORK_ORDER[::-1]):
            frame = frame_g.query('network == @network')
            for (values, color, alpha) in ((frame['pre'].to_numpy(float), CHARCOAL, 0.12), (frame['post'].to_numpy(float), POST, 0.18)):
                density = gaussian_kde(values)(xgrid)
                density = density / density.max() * 0.72
                ax.fill_between(xgrid, row, row + density, color=color, alpha=alpha, linewidth=0)
                ax.plot(xgrid, row + density, color=color, lw=1.0)
            ax.axhline(row, color=LIGHT, lw=0.45, zorder=0)
        ax.set_yticks(range(7))
        ax.set_yticklabels([NETWORK_ABBR[n] for n in NETWORK_ORDER[::-1]])
        ax.set_xlabel(f'Raw {gradient} network centroid')
        ax.set_title(gradient, loc='left', fontweight='bold')
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
        ax.tick_params(axis='y', length=0)
    axes[0].legend(handles=[Line2D([0], [0], color=CHARCOAL, lw=1.5, label='Pre'), Line2D([0], [0], color=POST, lw=1.5, label='Post')], loc='lower left', bbox_to_anchor=(0, -0.29), ncol=2)
    fig.suptitle('Subject-level seven-network gradient distributions', x=0.08, y=0.99, ha='left', fontweight='bold')
    fig.text(0.08, 0.945, 'Density curves describe 29 subject-level network centroids per phase', fontsize=7, color=MID)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.88, bottom=0.19, wspace=0.25)
    save_all(fig, 'network_gradient_distribution_ridgelines')

def main() -> None:
    apply_style()
    roi_stats = pd.read_csv(TABLES / 'raw_aligned5_roi_paired_tests.csv')
    counts = pd.read_csv(SOURCE / 'network_nominal_roi_counts.csv')
    effects = pd.read_csv(TABLES / 'network_metric_effect_sizes.csv')
    component_long = pd.read_csv(SOURCE / 'component_variance_subject_level.csv')
    component_tests = pd.read_csv(TABLES / 'component_variance_paired_tests.csv')
    clinical_subjects = pd.read_csv(SOURCE / 'selected_G2_clinical_partial_regression_data.csv')
    selected = pd.read_csv(TABLES / 'selected_exploratory_G2_clinical_model.csv').iloc[0]
    clinical_models = pd.read_csv(TABLES / 'clinical_roi_models_G1_G2.csv')
    ml_summary = pd.read_csv(TABLES / 'ml_clinical_augmented_performance_summary.csv')
    centroids = pd.read_csv(SOURCE / 'network_centroid_pre_post_long.csv')
    (_, _, pre, post) = load_gradients()
    (t_images, tmax) = make_t_surfaces(roi_stats)
    mean_images = make_mean_surfaces(pre, post)
    make_main(t_images, tmax, roi_stats, counts, effects, component_long, component_tests, clinical_subjects, selected, clinical_models, ml_summary)
    make_standalone_surfaces(t_images, tmax, roi_stats)
    make_mean_surface_figures(mean_images)
    make_radar_standalone(counts)
    make_heatmap_standalone(effects)
    make_component_standalone(component_long, component_tests)
    make_clinical_standalone(clinical_subjects, selected)
    make_ridge_standalone(centroids)
    make_ml_r2_standalone(ml_summary)
    print(f'Figures written to {FIGURES}', flush=True)
if __name__ == '__main__':
    main()
