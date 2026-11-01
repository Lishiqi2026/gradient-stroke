from __future__ import annotations
import os as _portable_os
from pathlib import Path as _PortablePath

def _portable_path(kind, relative):
    project = _PortablePath(_portable_os.environ['PAPER_PROJECT_ROOT'])
    root = project if kind == 'project' else project.parent
    return str(root / relative)
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import plot_lesion_longitudinal_nature as base
HERE = Path(__file__).resolve().parent
OUT = HERE / 'lesion_longitudinal_figure'
FIGURES = OUT / 'figures'

def save_bundle(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix('.svg'), facecolor='white')
    fig.savefig(stem.with_suffix('.pdf'), facecolor='white')
    fig.savefig(stem.with_suffix('.tiff'), dpi=600, facecolor='white')
    fig.savefig(stem.with_suffix('.png'), dpi=300, facecolor='white')
    plt.close(fig)

def make_overlap_figure(included, maps) -> None:
    (width_mm, height_mm) = (183.0, 100.0)
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), facecolor='white')
    grid = fig.add_gridspec(1, 1, left=0.105, right=0.985, bottom=0.075, top=0.89)
    counts = included.groupby('group').size().astype(int).to_dict()
    base.plot_spatial_plate(fig, grid[0, 0], maps, counts)
    fig.text(0.105, 0.965, 'Paired longitudinal lesion topography', ha='left', va='bottom', fontsize=7.1, fontweight='bold', color=base.INK)
    base.panel_label(fig, 0.018, 0.95, 'a')
    save_bundle(fig, FIGURES / 'panel_a_lesion_overlap_four_conditions_nature')

def make_boxplot_figure(included, comparisons) -> None:
    (width_mm, height_mm) = (89.0, 82.0)
    within_stats = base.calculate_within_group_statistics(included)
    between_row = comparisons.loc[comparisons['metric'] == 'absolute_volume_change_ml'].iloc[0]
    base.RNG = np.random.default_rng(20260803)
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), facecolor='white')
    grid = fig.add_gridspec(1, 1, left=0.205, right=0.985, bottom=0.19, top=0.785)
    ax = fig.add_subplot(grid[0, 0])
    base.plot_volume_panel(ax, included, within_stats, float(between_row.fdr_q))
    for label in ax.texts:
        if label.get_text().startswith('Between-group absolute volume change'):
            label.set_text(f'Between-group change: q={float(between_row.fdr_q):.3f}')
            label.set_fontsize(5.4)
    fig.text(0.205, 0.94, 'Paired lesion-volume change', ha='left', va='bottom', fontsize=7.0, fontweight='bold', color=base.INK)
    base.panel_label(fig, 0.025, 0.93, 'b')
    save_bundle(fig, FIGURES / 'panel_b_lesion_volume_paired_boxplot_nature')

def make_boxplot_linear_figure(included, comparisons) -> None:
    (width_mm, height_mm) = (89.0, 82.0)
    within_stats = base.calculate_within_group_statistics(included)
    between_row = comparisons.loc[comparisons['metric'] == 'absolute_volume_change_ml'].iloc[0]
    base.RNG = np.random.default_rng(20260803)
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), facecolor='white')
    grid = fig.add_gridspec(1, 1, left=0.205, right=0.985, bottom=0.19, top=0.785)
    ax = fig.add_subplot(grid[0, 0])
    base.plot_volume_panel(ax, included, within_stats, float(between_row.fdr_q))
    ax.set_yscale('linear')
    ax.set_ylim(0, 225)
    ax.set_yticks([0, 50, 100, 150, 200])
    ax.set_yticklabels(['0', '50', '100', '150', '200'])
    ax.set_ylabel('Lesion volume (ml)')
    for label in ax.texts:
        if label.get_text().startswith('Between-group absolute volume change'):
            label.set_text(f'Between-group change: q={float(between_row.fdr_q):.3f}')
            label.set_fontsize(5.4)
    fig.text(0.205, 0.94, 'Paired lesion-volume change', ha='left', va='bottom', fontsize=7.0, fontweight='bold', color=base.INK)
    base.panel_label(fig, 0.025, 0.93, 'b')
    save_bundle(fig, FIGURES / 'panel_b_lesion_volume_paired_boxplot_linear_nature')

def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    (included, comparisons, _) = base.read_longitudinal_data()
    (maps, _) = base.validate_and_build_overlap_maps(included)
    make_overlap_figure(included, maps)
    make_boxplot_figure(included, comparisons)
    make_boxplot_linear_figure(included, comparisons)
    print('Separate overlap and boxplot export complete')
    print('Overlap: 183 x 100 mm')
    print('Boxplot: 89 x 82 mm')
    print('Linear-scale boxplot: 89 x 82 mm')
if __name__ == '__main__':
    main()
