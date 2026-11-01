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
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import plotting
import regenerate_clinical_association_forai as landscape
import regenerate_clinical_association_network_glass_brains_forai as glass
OUTPUT_PDF = glass.OUTPUT_DIR / 'g_whole_cortex_clinical_association_with_network_glass_brains.pdf'

def apply_style() -> None:
    mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'], 'font.size': 7.0, 'axes.labelsize': 7.5, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.3, 'axes.linewidth': 0.7, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': landscape.INK, 'axes.labelcolor': landscape.INK, 'xtick.color': landscape.INK, 'ytick.color': landscape.INK, 'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'axes.unicode_minus': False, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})

def add_scatter_axis(fig: plt.Figure, rect: list[float], ordered: pd.DataFrame, gradient: str) -> plt.Axes:
    ax = fig.add_axes(rect)
    frame = ordered.loc[ordered['gradient'].eq(gradient)].reset_index(drop=True)
    x = np.empty(len(frame), dtype=float)
    for (network_index, network) in enumerate(glass.NETWORK_ORDER):
        indices = np.flatnonzero(frame['network'].eq(network).to_numpy())
        if not len(indices):
            raise ValueError(f'No ROIs found for network: {network}')
        x[indices] = network_index + (np.arange(len(indices), dtype=float) + 0.5) / len(indices)
    beta = frame['standardized_beta'].to_numpy(float)
    nominal = frame['hc3_p_two_sided'].to_numpy(float) < 0.05
    positive = nominal & (beta > 0.0)
    negative = nominal & (beta < 0.0)
    ax.scatter(x[~nominal], beta[~nominal], s=8.5, color=landscape.GREY, alpha=0.68, edgecolor='none', zorder=1)
    ax.scatter(x[positive], beta[positive], s=25.0, color=glass.RED, edgecolor='white', linewidth=0.35, zorder=3)
    ax.scatter(x[negative], beta[negative], s=25.0, color=glass.BLUE, edgecolor='white', linewidth=0.35, zorder=3)
    ax.axhline(0.0, color=landscape.MID, linewidth=0.6, zorder=0)
    for boundary in range(1, len(glass.NETWORK_ORDER)):
        ax.axvline(boundary, color=landscape.SEPARATOR, linewidth=0.48, zorder=0)
    ax.set_xlim(0.0, float(len(glass.NETWORK_ORDER)))
    ax.set_ylim(-0.68, 0.68)
    ax.set_yticks([-0.5, 0.0, 0.5])
    ax.set_ylabel(f'{gradient}  β', labelpad=3)
    ax.set_xticks([])
    ax.tick_params(axis='y', length=2.4, width=0.7, pad=2)
    ax.spines['bottom'].set_visible(False)
    return ax

def add_glass_row(fig: plt.Figure, models: pd.DataFrame, centroids: dict[int, np.ndarray], gradient: str, plot_left: float, plot_width: float, y: float, height: float, show_network_labels: bool) -> None:
    figure_edges = np.linspace(plot_left, plot_left + plot_width, len(glass.NETWORK_ORDER) + 1)
    for (index, network) in enumerate(glass.NETWORK_ORDER):
        cell_left = float(figure_edges[index])
        cell_right = float(figure_edges[index + 1])
        cell_width = cell_right - cell_left
        pad = min(0.004, cell_width * 0.055)
        axis = fig.add_axes([cell_left + pad, y, cell_width - 2.0 * pad, height], zorder=3)
        (coordinates, colors, n_positive, n_negative) = glass.marker_data(models, centroids, gradient, network)
        display = plotting.plot_glass_brain(None, figure=fig, axes=axis, display_mode='z', colorbar=False, annotate=False, draw_cross=False, black_bg=False, alpha=0.34)
        if coordinates:
            display.add_markers(coordinates, marker_color=colors, marker_size=28, edgecolors='white', linewidths=0.5, alpha=0.98, zorder=10)
        else:
            fig.text((cell_left + cell_right) / 2.0, y + 0.005, 'none', ha='center', va='bottom', fontsize=4.8, color='#9A9A9A')
        if show_network_labels:
            fig.text((cell_left + cell_right) / 2.0, y - 0.012, glass.NETWORK_ABBR[network], ha='center', va='top', fontsize=6.2, color=landscape.INK)

def make_figure(models: pd.DataFrame, centroids: dict[int, np.ndarray]) -> plt.Figure:
    (ordered, _, _) = landscape.order_models(models)
    (width_mm, height_mm) = (145.0, 108.0)
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), facecolor='white')
    plot_left = 0.082
    plot_right = 0.988
    plot_width = plot_right - plot_left
    add_scatter_axis(fig, [plot_left, 0.725, plot_width, 0.165], ordered, 'G1')
    add_glass_row(fig, models, centroids, 'G1', plot_left, plot_width, y=0.548, height=0.135, show_network_labels=False)
    add_scatter_axis(fig, [plot_left, 0.33, plot_width, 0.165], ordered, 'G2')
    add_glass_row(fig, models, centroids, 'G2', plot_left, plot_width, y=0.153, height=0.135, show_network_labels=True)
    fig.text(plot_left, 0.972, 'Whole-cortex clinical-association landscape', ha='left', va='top', fontsize=10.0, fontweight='bold', color=landscape.INK)
    fig.text(plot_left, 0.918, 'Grey: other ROIs; red/blue: positive/negative HC3 p<0.05 dots and ROI-centroid spheres; n=29', ha='left', va='top', fontsize=6.3, color=landscape.MID)
    fig.text(0.53, 0.055, 'Schaefer400 ROIs grouped by network', ha='center', va='bottom', fontsize=7.1, color=landscape.INK)
    return fig

def main() -> None:
    apply_style()
    if not glass.SOURCE_CSV.is_file():
        raise FileNotFoundError(f'Source data not found: {glass.SOURCE_CSV}')
    if not glass.ATLAS_NIFTI.is_file():
        raise FileNotFoundError(f'Schaefer atlas not found: {glass.ATLAS_NIFTI}')
    models = pd.read_csv(glass.SOURCE_CSV)
    atlas_img = nib.load(glass.ATLAS_NIFTI)
    atlas_data = np.asarray(atlas_img.dataobj).astype(np.int16)
    glass.validate_inputs(models, atlas_data)
    centroids = glass.roi_centroids_mm(atlas_img, atlas_data)
    glass.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = make_figure(models, centroids)
    fig.savefig(OUTPUT_PDF, format='pdf', bbox_inches='tight', pad_inches=0.035, facecolor='white', metadata={'Creator': 'Python / Matplotlib / Nilearn', 'Title': 'Whole-cortex clinical-association landscape with glass brains', 'Subject': 'G1 and G2 HC3 ROI associations and network-resolved centroid locations'})
    qa_png = Path(tempfile.gettempdir()) / 'g_clinical_association_with_glass_brains_qa.png'
    fig.savefig(qa_png, format='png', dpi=300, bbox_inches='tight', pad_inches=0.035, facecolor='white')
    plt.close(fig)
    print(f'Saved: {OUTPUT_PDF}')
    print(f'QA preview: {qa_png}')
if __name__ == '__main__':
    main()
