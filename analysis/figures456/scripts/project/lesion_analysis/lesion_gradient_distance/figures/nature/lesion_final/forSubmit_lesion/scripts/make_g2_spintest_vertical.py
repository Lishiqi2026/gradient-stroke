from __future__ import annotations
'Vertical spin-test violin panels for a narrow manuscript column.\n\nThe saved spin-test statistics and null distributions are reused verbatim; this\nscript changes only the layout and styling.\n'
import os as _portable_os
from pathlib import Path as _PortablePath

def _portable_path(kind, relative):
    project = _PortablePath(_portable_os.environ['PAPER_PROJECT_ROOT'])
    root = project if kind == 'project' else project.parent
    return str(root / relative)
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from figure_common import INK, MID, save_submission_figure
PACKAGE = Path(__file__).resolve().parents[1]
FIGURES = PACKAGE / 'figures'
STATS_PATH = FIGURES / 'G2_TMS_effect_spintest_statistics.csv'
NULL_PATH = FIGURES / 'G2_TMS_effect_spintest_null_distributions.npz'
NETWORK_ORDER = ['Visual', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Control', 'Default']
NETWORK_ABBR = ['VIS', 'SMN', 'DAN', 'VAN', 'LIM', 'CON', 'DMN']
POSITIVE = '#B2182B'
NEGATIVE = '#2166AC'

def draw_direction_panel(axis: plt.Axes, stats: pd.DataFrame, null: np.ndarray, color: str, title: str, *, show_xlabels: bool) -> None:
    stats = stats.set_index('network').loc[NETWORK_ORDER].reset_index()
    positions = np.arange(len(NETWORK_ORDER), dtype=float)
    violins = axis.violinplot([null[:, index] for index in range(len(NETWORK_ORDER))], positions=positions, widths=0.68, showmeans=False, showmedians=False, showextrema=False, bw_method=0.18)
    for body in violins['bodies']:
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.24)
        body.set_linewidth(0.7)
    observed = stats['actual_enrichment_ratio'].to_numpy(float)
    for (index, value) in enumerate(observed):
        axis.plot([index - 0.25, index + 0.25], [value, value], color=color, linewidth=1.85, solid_capstyle='round', zorder=5)
        q_value = float(stats.iloc[index]['spin_fdr_q'])
        if q_value < 0.05:
            axis.text(index, value + 0.032, f'*  q={q_value:.3f}', ha='center', va='bottom', fontsize=5.4, fontweight='bold', color=color)
    axis.set_xlim(-0.52, len(NETWORK_ORDER) - 0.48)
    axis.set_ylim(0.0, 0.86)
    axis.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
    axis.set_xticks(positions)
    if show_xlabels:
        axis.set_xticklabels(NETWORK_ABBR)
        axis.set_xlabel('Functional network', labelpad=2)
    else:
        axis.set_xticklabels([])
        axis.tick_params(axis='x', length=0)
    axis.set_title(title, loc='left', color=color, fontweight='bold', fontsize=6.8, pad=3)
    axis.tick_params(axis='both', pad=1.5)
    axis.grid(False)

def main() -> None:
    stats = pd.read_csv(STATS_PATH)
    with np.load(NULL_PATH, allow_pickle=False) as archive:
        positive_null = archive['positive']
        negative_null = archive['negative']
        n_permutations = int(np.asarray(archive['n_permutations']).item())
    positive_stats = stats.query("direction == 'positive'").copy()
    negative_stats = stats.query("direction == 'negative'").copy()
    (width_mm, height_mm) = (66.0, 70.0)
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), facecolor='white')
    grid = fig.add_gridspec(2, 1, left=0.2, right=0.985, bottom=0.15, top=0.84, hspace=0.35)
    axes = [fig.add_subplot(grid[row, 0]) for row in range(2)]
    draw_direction_panel(axes[0], positive_stats, positive_null, POSITIVE, 'Positive G2 effect (59 ROIs)', show_xlabels=False)
    draw_direction_panel(axes[1], negative_stats, negative_null, NEGATIVE, 'Negative G2 effect (8 ROIs)', show_xlabels=True)
    fig.text(0.2, 0.965, 'Network enrichment of significant G2 effects', ha='left', va='top', fontsize=7.7, fontweight='bold', color=INK)
    fig.text(0.2, 0.91, f'{n_permutations:,} rotations; violin, spatial null; bar, observed', ha='left', va='top', fontsize=5.2, color=MID)
    fig.text(0.045, 0.5, 'Network enrichment ratio', ha='center', va='center', rotation=90, fontsize=6.5, color=INK)
    save_submission_figure(fig, 'G2_TMS_effect_spintest_two_panel_vertical')
if __name__ == '__main__':
    main()
