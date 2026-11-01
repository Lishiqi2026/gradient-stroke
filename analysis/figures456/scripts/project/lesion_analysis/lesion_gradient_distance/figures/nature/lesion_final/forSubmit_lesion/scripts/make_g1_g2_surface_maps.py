from __future__ import annotations
import os as _portable_os
from pathlib import Path as _PortablePath

def _portable_path(kind, relative):
    project = _PortablePath(_portable_os.environ['PAPER_PROJECT_ROOT'])
    root = project if kind == 'project' else project.parent
    return str(root / relative)
from pathlib import Path
import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from brainspace.datasets import load_parcellation
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
from neuromaps.datasets import fetch_fslr
import numpy as np
import pandas as pd
from surfplot import Plot
from figure_common import QA, SOURCE, save_submission_figure
G1_CMAP = LinearSegmentedColormap.from_list('g1_reference', ['#315F87', '#A7C6DF', '#F8F8F8', '#E8A07A', '#A84B3C'])
G2_CMAP = LinearSegmentedColormap.from_list('g2_reference', ['#B48A2C', '#E7D39B', '#F8F8F8', '#B7ADD2', '#5E4B8B'])
EFFECT_CMAP = LinearSegmentedColormap.from_list('normalization', ['#315F87', '#A7C6DF', '#F8F8F8', '#F1B47D', '#A95316'])

def robust_limit(values: np.ndarray, minimum: float=0.02) -> float:
    finite = np.abs(np.asarray(values, float)[np.isfinite(values)])
    return max(float(np.percentile(finite, 98)), minimum)

def render_surface(values: np.ndarray, name: str, cmap, limit: float) -> Path:
    cache = QA / 'surface_cache'
    cache.mkdir(parents=True, exist_ok=True)
    output = cache / f'{name}.png'
    surfaces = fetch_fslr()
    (lh_surf, rh_surf) = surfaces['veryinflated']
    atlas = np.asarray(load_parcellation('schaefer', 400, join=True))
    vertex = np.full(atlas.shape, np.nan, dtype=float)
    for roi in range(1, 401):
        vertex[atlas == roi] = values[roi - 1]
    medial_wall = np.full(atlas.shape, np.nan, dtype=float)
    medial_wall[atlas == 0] = 1.0
    plot = Plot(lh_surf, rh_surf, views=['lateral', 'medial'], layout='grid', zoom=1.64, brightness=0.68, size=(3000, 1650), background=(1, 1, 1))
    plot.add_layer(medial_wall, cmap=ListedColormap(['#D1D1D1']), color_range=(0, 1), cbar=False)
    plot.add_layer(vertex, cmap=cmap, color_range=(-limit, limit), cbar=False)
    plot.add_layer(atlas, cmap=ListedColormap(['#666666']), as_outline=True, cbar=False)
    figure = plot.build()
    figure.patch.set_facecolor('white')
    figure.savefig(output, dpi=400, bbox_inches='tight', pad_inches=0.0, facecolor='white')
    plt.close(figure)
    return output

def add_surface_panel(axis: plt.Axes, image_path: Path, title: str, values: np.ndarray, cmap, limit: float, colorbar_label: str) -> None:
    axis.set_axis_off()
    image_axis = axis.inset_axes([0.0, 0.15, 1.0, 0.76])
    image_axis.imshow(mpimg.imread(image_path))
    image_axis.set_axis_off()
    axis.text(0.5, 0.985, title, transform=axis.transAxes, ha='center', va='top', fontsize=7.2, fontweight='bold')
    color_axis = axis.inset_axes([0.24, 0.055, 0.52, 0.028])
    colorbar = plt.colorbar(mpl.cm.ScalarMappable(norm=Normalize(-limit, limit), cmap=cmap), cax=color_axis, orientation='horizontal')
    colorbar.set_ticks([-limit, 0, limit])
    colorbar.ax.tick_params(labelsize=5.3, length=1.6, width=0.45, pad=1)
    colorbar.outline.set_linewidth(0.45)
    colorbar.set_label(colorbar_label, fontsize=5.5, labelpad=1.0)

def main() -> None:
    source = pd.read_csv(SOURCE / 'G1_G2_surface_source_data.csv').sort_values('roi')
    g1_reference = source['G1_reference_z'].to_numpy(float)
    g2_reference = source['G2_reference_z'].to_numpy(float)
    g1_effect = source['adjusted_TMS_beta_G1'].to_numpy(float)
    g2_effect = source['G2_adjusted_beta_FDR_masked'].to_numpy(float)
    g1_ref_limit = 1.5
    g2_ref_limit = 2.0
    g1_effect_limit = robust_limit(g1_effect)
    g2_effect_limit = robust_limit(source['adjusted_TMS_beta_G2'].to_numpy(float))
    specifications = [(g1_reference, 'g1_reference', G1_CMAP, g1_ref_limit, 'Healthy-reference G1 hierarchy', 'Reference G1 (z)'), (g1_effect, 'g1_adjusted_effect', EFFECT_CMAP, g1_effect_limit, 'TMS-related G1 normalization (no FDR-significant ROIs)', 'Adjusted TMS beta'), (g2_reference, 'g2_reference', G2_CMAP, g2_ref_limit, 'Healthy-reference G2 hierarchy', 'Reference G2 (z)'), (g2_effect, 'g2_adjusted_effect_fdr', EFFECT_CMAP, g2_effect_limit, 'TMS-related G2 normalization (67 FDR-significant ROIs)', 'Adjusted TMS beta')]
    rendered = [render_surface(values, name, cmap, limit) for (values, name, cmap, limit, _, _) in specifications]
    (fig, axes) = plt.subplots(2, 2, figsize=(7.2, 5.25))
    for (axis, image_path, specification) in zip(axes.ravel(), rendered, specifications):
        (values, _, cmap, limit, title, label) = specification
        add_surface_panel(axis, image_path, title, values, cmap, limit, label)
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.02, top=0.985, wspace=0.05, hspace=0.1)
    save_submission_figure(fig, 'G1_G2_reference_and_adjusted_TMS_effect')
if __name__ == '__main__':
    main()
