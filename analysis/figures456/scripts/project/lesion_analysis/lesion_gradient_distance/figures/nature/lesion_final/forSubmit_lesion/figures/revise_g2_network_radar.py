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
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
SCRIPT_DIR = Path(__file__).resolve().parent.parent / 'scripts'
sys.path.insert(0, str(SCRIPT_DIR))
from figure_common import INK, SOURCE, save_submission_figure
NETWORK_ORDER = ['Visual', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Control', 'Default']
NEGATIVE = '#2166AC'
POSITIVE = '#B2182B'
SPOKE = '#D0D0D0'
ZERO = '#5A535B'

def map_radius(values: np.ndarray, maximum: float) -> np.ndarray:
    """Map zero to the reference circle and counts into the outer half."""
    base_radius = maximum * 0.5
    return base_radius + values / maximum * (maximum - base_radius)

def main() -> None:
    counts = pd.read_csv(SOURCE / 'G2_FDR_network_counts.csv').set_index('network').loc[NETWORK_ORDER]
    negative = counts['negative_FDR_ROI_count'].to_numpy(float)
    positive = counts['positive_FDR_ROI_count'].to_numpy(float)
    maximum = float(max(negative.max(), positive.max()))
    angles = np.linspace(0, 2 * np.pi, len(counts), endpoint=False)
    closed_angles = np.r_[angles, angles[0]]
    neg_radius = map_radius(negative, maximum)
    pos_radius = map_radius(positive, maximum)
    (width_mm, height_mm) = (100.0, 94.0)
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), facecolor='white')
    axis = fig.add_subplot(111, projection='polar')
    axis.set_theta_offset(np.pi / 2)
    axis.set_theta_direction(-1)
    axis.set_ylim(0, maximum)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.grid(False)
    axis.spines['polar'].set_color('#050505')
    axis.spines['polar'].set_linewidth(1.45)
    base_radius = maximum * 0.5
    for angle in angles:
        axis.plot([angle, angle], [base_radius, maximum], color=SPOKE, lw=0.7, linestyle=(0, (3, 2)), zorder=0)
    theta = np.linspace(0, 2 * np.pi, 720)
    axis.fill(theta, np.full_like(theta, base_radius), color='white', zorder=0)
    axis.plot(theta, np.full_like(theta, base_radius), color=ZERO, lw=0.9, linestyle=(0, (4, 2.5)), zorder=1)
    for (radius, color, zorder) in ((neg_radius, NEGATIVE, 3), (pos_radius, POSITIVE, 4)):
        closed_radius = np.r_[radius, radius[0]]
        axis.fill(closed_angles, closed_radius, color=color, alpha=0.16, zorder=zorder)
        axis.plot(closed_angles, closed_radius, color=color, lw=1.65, marker='o', markersize=3.7, markerfacecolor=color, markeredgecolor='white', markeredgewidth=0.45, zorder=zorder + 2)
    labels = counts['network_abbreviation'].tolist()
    for (angle, label) in zip(angles, labels):
        x = 0.5 + 0.515 * np.sin(angle)
        y = 0.5 + 0.515 * np.cos(angle)
        ha = 'left' if x > 0.56 else 'right' if x < 0.44 else 'center'
        va = 'bottom' if y > 0.56 else 'top' if y < 0.44 else 'center'
        axis.text(x, y, label, transform=axis.transAxes, ha=ha, va=va, fontsize=10.0, fontweight='bold', color=INK)
    for (raw, radius, color) in ((negative, neg_radius, NEGATIVE), (positive, pos_radius, POSITIVE)):
        index = int(np.argmax(raw))
        if raw[index] <= 0:
            continue
        text_radius = radius[index] - maximum * 0.075 if radius[index] >= maximum * 0.96 else radius[index] + maximum * 0.09
        axis.text(angles[index], text_radius, str(int(raw[index])), ha='center', va='top' if radius[index] >= maximum * 0.96 else 'bottom', fontsize=10.0, fontweight='bold', color=color, zorder=8)
    legend_handles = [Line2D([0], [0], color=NEGATIVE, lw=1.65, marker='o', markersize=3.8, markeredgecolor='white', markeredgewidth=0.4, label='Negative adjusted TMS beta'), Line2D([0], [0], color=POSITIVE, lw=1.65, marker='o', markersize=3.8, markeredgecolor='white', markeredgewidth=0.4, label='Positive adjusted TMS beta')]
    fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, 0.018), ncol=2, frameon=False, fontsize=7.0, handlelength=1.8, columnspacing=1.5)
    fig.subplots_adjust(left=0.055, right=0.945, bottom=0.145, top=0.985)
    save_submission_figure(fig, 'G2_FDR_network_distribution')
if __name__ == '__main__':
    main()
