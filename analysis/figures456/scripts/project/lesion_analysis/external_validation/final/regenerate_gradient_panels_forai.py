from __future__ import annotations
import os as _portable_os
from pathlib import Path as _PortablePath

def _portable_path(kind, relative):
    project = _PortablePath(_portable_os.environ['PAPER_PROJECT_ROOT'])
    root = project if kind == 'project' else project.parent
    return str(root / relative)
import sys
import tempfile
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
HERE = Path(__file__).resolve().parent
SOURCE_DIR = HERE.parent / 'ds003999_validation' / 'matched_gradient_figure'
OUTPUT_DIR = HERE / 'forai'
QA_PREVIEW_DIR = OUTPUT_DIR / '_qa_preview'
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))
import make_matched_gradient_nature_figure as source
GRADIENT_CMAPS = {'G1': mpl.colormaps['PiYG'], 'G2': mpl.colormaps['PuOr']}

def save_pdf(fig: plt.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    QA_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f'{stem}.pdf', format='pdf', bbox_inches='tight', pad_inches=0.04, metadata={'Creator': 'Python / Matplotlib'})
    fig.savefig(QA_PREVIEW_DIR / f'{stem}.png', format='png', dpi=160, bbox_inches='tight', pad_inches=0.04)
    plt.close(fig)

def add_colorbar(fig: plt.Figure, cax: plt.Axes, gradient: str, limit: float, label: str) -> None:
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit), cmap=GRADIENT_CMAPS[gradient]), cax=cax, orientation='horizontal')
    cb.set_ticks([-limit, 0.0, limit])
    cb.set_label(label, labelpad=3)
    cb.ax.tick_params(length=2.5, pad=1.5)
    cb.outline.set_linewidth(0.6)

def render_mean_surfaces(pre: np.ndarray, post: np.ndarray) -> dict[str, tuple[Path, Path, float]]:
    outputs: dict[str, tuple[Path, Path, float]] = {}
    for (gradient_index, gradient) in enumerate(('G1', 'G2')):
        pre_mean = np.mean(pre[:, :, gradient_index], axis=0)
        post_mean = np.mean(post[:, :, gradient_index], axis=0)
        limit = float(max(np.max(np.abs(pre_mean)), np.max(np.abs(post_mean))))
        pre_image = source.render_surface(pre_mean, f'{gradient}_pre_mean_forai', GRADIENT_CMAPS[gradient], (-limit, limit))
        post_image = source.render_surface(post_mean, f'{gradient}_post_mean_forai', GRADIENT_CMAPS[gradient], (-limit, limit))
        outputs[gradient] = (pre_image, post_image, limit)
    return outputs

def render_longitudinal_surfaces(roi_stats: pd.DataFrame) -> dict[str, tuple[Path, float]]:
    outputs: dict[str, tuple[Path, float]] = {}
    for gradient in ('G1', 'G2'):
        frame = roi_stats.query('gradient == @gradient').sort_values('roi')
        values = np.where(frame['nominal_p_lt_0_05'], frame['t'], 0.0)
        limit = float(np.ceil(np.max(np.abs(values)) * 10.0) / 10.0)
        image = source.render_surface(values, f'{gradient}_paired_t_forai', GRADIENT_CMAPS[gradient], (-limit, limit), mask_zeros=True)
        outputs[gradient] = (image, limit)
    return outputs

def make_pre_post_pdfs(mean_images: dict[str, tuple[Path, Path, float]]) -> None:
    specifications = (('j', 'G1', 'Pre', 0), ('k', 'G1', 'Post', 1), ('l', 'G2', 'Pre', 0), ('m', 'G2', 'Post', 1))
    for (letter, gradient, phase, image_index) in specifications:
        (pre_image, post_image, limit) = mean_images[gradient]
        image_path = (pre_image, post_image)[image_index]
        fig = plt.figure(figsize=(7.2, 4.8))
        grid = fig.add_gridspec(3, 3, height_ratios=[18, 1.0, 0.9], width_ratios=[1.0, 3.2, 1.0], left=0.025, right=0.975, top=0.93, bottom=0.07, hspace=0.28)
        ax = fig.add_subplot(grid[0, :])
        ax.imshow(plt.imread(image_path), interpolation='none')
        ax.set_axis_off()
        ax.set_title(f'{gradient} group mean — {phase}', loc='left', fontweight='bold', pad=2)
        add_colorbar(fig, fig.add_subplot(grid[1, 1]), gradient, limit, 'Mean raw gradient coordinate')
        note_ax = fig.add_subplot(grid[2, :])
        note_ax.set_axis_off()
        note_ax.text(0.5, 0.5, f'Group mean across 29 paired participants; {gradient} pre and post share one color scale', ha='center', va='center', fontsize=7.0, color=source.MID)
        save_pdf(fig, f'{letter}_{gradient}_{phase.lower()}_gradient_surface')

def make_longitudinal_pdfs(roi_stats: pd.DataFrame, t_images: dict[str, tuple[Path, float]]) -> None:
    for (letter, gradient) in (('a', 'G1'), ('b', 'G2')):
        frame = roi_stats.query('gradient == @gradient')
        (image_path, limit) = t_images[gradient]
        fig = plt.figure(figsize=(7.2, 5.0))
        grid = fig.add_gridspec(3, 3, height_ratios=[18, 1.0, 1.0], width_ratios=[1.0, 3.2, 1.0], left=0.025, right=0.975, top=0.93, bottom=0.07, hspace=0.28)
        ax = fig.add_subplot(grid[0, :])
        ax.imshow(plt.imread(image_path), interpolation='none')
        ax.set_axis_off()
        ax.set_title(f'{gradient} longitudinal ROI effects', loc='left', fontweight='bold', pad=2)
        add_colorbar(fig, fig.add_subplot(grid[1, 1]), gradient, limit, 'Paired t (post − pre)')
        note_ax = fig.add_subplot(grid[2, :])
        note_ax.set_axis_off()
        note_ax.text(0.5, 0.5, f"Colored: two-sided paired p<0.05 (uncorrected); white: non-significant; {int(frame['nominal_p_lt_0_05'].sum())} colored ROIs; n=29", ha='center', va='center', fontsize=7.0, color=source.MID)
        save_pdf(fig, f'{letter}_{gradient}_longitudinal_ROI_effects')

def main() -> None:
    source.apply_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    roi_stats = pd.read_csv(source.TABLES / 'raw_aligned5_roi_paired_tests.csv')
    (_, _, pre, post) = source.load_gradients()
    with tempfile.TemporaryDirectory(prefix='gradient_surface_') as cache_dir:
        source.SURFACE_CACHE = Path(cache_dir)
        mean_images = render_mean_surfaces(pre, post)
        t_images = render_longitudinal_surfaces(roi_stats)
        make_pre_post_pdfs(mean_images)
        make_longitudinal_pdfs(roi_stats, t_images)
    print('Generated PDFs:')
    for path in sorted(OUTPUT_DIR.glob('*.pdf')):
        print(path.name)
    print('Shared mean ranges:')
    for (gradient, (_, _, limit)) in mean_images.items():
        print(f'{gradient}: {-limit:.6g} to {limit:.6g} ({GRADIENT_CMAPS[gradient].name})')
    print('Longitudinal t ranges:')
    for (gradient, (_, limit)) in t_images.items():
        print(f'{gradient}: {-limit:.6g} to {limit:.6g} ({GRADIENT_CMAPS[gradient].name})')
if __name__ == '__main__':
    main()
