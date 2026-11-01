from __future__ import annotations
import os as _portable_os
from pathlib import Path as _PortablePath

def _portable_path(kind, relative):
    project = _PortablePath(_portable_os.environ['PAPER_PROJECT_ROOT'])
    root = project if kind == 'project' else project.parent
    return str(root / relative)
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from PIL import Image
from brainspace.datasets import load_parcellation
from neuromaps.datasets import fetch_fslr
from surfplot import Plot
SCRIPT_DIR = Path(__file__).resolve().parent.parent / 'scripts'
sys.path.insert(0, str(SCRIPT_DIR))
from figure_common import SOURCE
from make_g1_g2_surface_maps import robust_limit
TMS_EFFECT_CMAP = LinearSegmentedColormap.from_list('g2_tms_effect_blue_red', ['#2166AC', '#67A9CF', '#D1E5F0', '#FFFFFF', '#FDDBC7', '#EF8A62', '#B2182B'])
G2_STANDALONE_CMAP = LinearSegmentedColormap.from_list('g2_normative_yellow_purple', ['#C58F16', '#E3BE5B', '#F4E5B5', '#FFFFFF', '#DDD1EA', '#9B7BC0', '#5B3A8A'])

def render_surface_crisp(values, name: str, cmap, limit: float):
    """Render directly through VTK at high pixel resolution to avoid a soft rescale."""
    cache = Path(__file__).resolve().parent.parent / 'qa' / 'surface_cache'
    cache.mkdir(parents=True, exist_ok=True)
    output = cache / f'{name}.png'
    surfaces = fetch_fslr()
    (lh_surf, rh_surf) = surfaces['veryinflated']
    atlas = np.asarray(load_parcellation('schaefer', 400, join=True))
    vertex = np.full(atlas.shape, np.nan, dtype=float)
    for roi in range(1, 401):
        vertex[atlas == roi] = values[roi - 1]
    cortical_mask = np.where(atlas > 0, 1.0, np.nan)
    plot = Plot(lh_surf, rh_surf, views=['lateral', 'medial'], layout='grid', zoom=1.66, brightness=1.0, size=(2800, 2016), background=(1, 1, 1))
    plot.add_layer(cortical_mask, cmap=ListedColormap(['#FFFFFF']), color_range=(0, 1), cbar=False)
    plot.add_layer(vertex, cmap=cmap, color_range=(-limit, limit), cbar=False)
    plot.add_layer(atlas, cmap=ListedColormap(['#525252']), as_outline=True, cbar=False)
    rendered = plot.render()
    try:
        rendered.set_background('white')
    except Exception:
        pass
    rendered.screenshot(str(output), transparent_bg=False)
    try:
        rendered.close()
    except Exception:
        pass
    return output

def add_surface_panel_crisp(axis, image_path, title, cmap, limit, colorbar_label):
    axis.set_axis_off()
    image_axis = axis.inset_axes([0.0, 0.15, 1.0, 0.76])
    image_axis.imshow(mpimg.imread(image_path), interpolation='none', resample=False)
    image_axis.set_axis_off()
    axis.text(0.5, 0.985, title, transform=axis.transAxes, ha='center', va='top', fontsize=7.2, fontweight='bold')
    color_axis = axis.inset_axes([0.24, 0.055, 0.52, 0.028])
    colorbar = plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(-limit, limit), cmap=cmap), cax=color_axis, orientation='horizontal')
    colorbar.set_ticks([-limit, 0, limit])
    colorbar.ax.tick_params(labelsize=5.3, length=1.6, width=0.45, pad=1)
    colorbar.outline.set_linewidth(0.45)
    colorbar.set_label(colorbar_label, fontsize=5.5, labelpad=1.0)

def save_standalone_crisp(fig, output_stem: str) -> None:
    base = Path(__file__).resolve().parent / output_stem
    options = {'bbox_inches': 'tight', 'pad_inches': 0.035, 'facecolor': 'white'}
    fig.savefig(base.with_suffix('.pdf'), dpi=600, **options)
    fig.savefig(base.with_suffix('.svg'), dpi=600, **options)
    fig.savefig(base.with_suffix('.png'), dpi=600, **options)
    fig.savefig(base.with_suffix('.tiff'), dpi=600, pil_kwargs={'compression': 'tiff_lzw'}, **options)
    plt.close(fig)
    for suffix in ('.png', '.tiff'):
        path = base.with_suffix(suffix)
        with Image.open(path) as source:
            rgb = Image.new('RGB', source.size, 'white')
            if source.mode == 'RGBA':
                rgb.paste(source, mask=source.getchannel('A'))
            else:
                rgb.paste(source.convert('RGB'))
            if suffix == '.tiff':
                rgb.save(path, dpi=(600, 600), compression='tiff_lzw')
            else:
                rgb.save(path, dpi=(600, 600))

def save_standalone(values, cache_name: str, cmap, limit: float, title: str, colorbar_label: str, output_stem: str) -> None:
    surface = render_surface_crisp(values, cache_name, cmap, limit)
    (width_mm, height_mm) = (120.0, 68.0)
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), facecolor='white')
    axis = fig.add_axes([0.015, 0.015, 0.97, 0.97])
    add_surface_panel_crisp(axis, surface, title, cmap, limit, colorbar_label)
    save_standalone_crisp(fig, output_stem)

def main() -> None:
    source = pd.read_csv(SOURCE / 'G1_G2_surface_source_data.csv').sort_values('roi')
    g2_reference = source['G2_reference_z'].to_numpy(float)
    save_standalone(values=g2_reference, cache_name='standalone_healthy_normative_g2', cmap=G2_STANDALONE_CMAP, limit=2.0, title='Healthy normative G2', colorbar_label='Reference G2 (z)', output_stem='healthy_normative_G2')
    g2_effect = source['G2_adjusted_beta_FDR_masked'].to_numpy(float)
    effect_limit = robust_limit(source['adjusted_TMS_beta_G2'].to_numpy(float))
    save_standalone(values=g2_effect, cache_name='standalone_g2_normalized_tms_effect_fdr', cmap=TMS_EFFECT_CMAP, limit=effect_limit, title=f'Adjusted TMS effect on G2 normalization: FDR q<0.05 ({int(source.G2_FDR_significant.sum())} ROIs)', colorbar_label='Adjusted TMS beta', output_stem='G2_normalized_TMS_effect')
if __name__ == '__main__':
    main()
