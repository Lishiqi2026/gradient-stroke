from __future__ import annotations
import os as _portable_os
from pathlib import Path as _PortablePath

def _portable_path(kind, relative):
    project = _PortablePath(_portable_os.environ['PAPER_PROJECT_ROOT'])
    root = project if kind == 'project' else project.parent
    return str(root / relative)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from figure_common import INK, SOURCE, save_submission_figure
NETWORK_ORDER = ['Visual', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Control', 'Default']
NEGATIVE = '#877CB3'
POSITIVE = '#B76E79'
SPOKE = '#D0D0D0'
ZERO = '#5A535B'

def map_radius(values: np.ndarray, maximum: float) -> np.ndarray:
    zero_radius = maximum * 0.5
    return zero_radius + values / maximum * (maximum - zero_radius)

def main() -> None:
    counts = pd.read_csv(SOURCE / 'G2_FDR_network_counts.csv').set_index('network').loc[NETWORK_ORDER]
    negative = counts['negative_FDR_ROI_count'].to_numpy(float)
    positive = counts['positive_FDR_ROI_count'].to_numpy(float)
    maximum = float(max(negative.max(), positive.max()))
    angles = np.linspace(0, 2 * np.pi, len(counts), endpoint=False)
    closed = np.r_[angles, angles[0]]
    neg_radius = map_radius(negative, maximum)
    pos_radius = map_radius(positive, maximum)
    fig = plt.figure(figsize=(4.35, 4.25))
    axis = fig.add_subplot(111, projection='polar')
    axis.set_theta_offset(np.pi / 2)
    axis.set_theta_direction(-1)
    axis.set_ylim(0, maximum)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.grid(False)
    axis.spines['polar'].set_color('#050505')
    axis.spines['polar'].set_linewidth(1.45)
    for angle in angles:
        axis.plot([angle, angle], [0, maximum], color=SPOKE, lw=0.7, linestyle=(0, (3, 2)), zorder=0)
    theta = np.linspace(0, 2 * np.pi, 720)
    axis.plot(theta, np.full_like(theta, maximum * 0.5), color=ZERO, lw=1.0, linestyle=(0, (4, 2.5)), zorder=1)
    for (radius, color, zorder) in ((neg_radius, NEGATIVE, 3), (pos_radius, POSITIVE, 4)):
        closed_radius = np.r_[radius, radius[0]]
        axis.fill(closed, closed_radius, color=color, alpha=0.16, zorder=zorder)
        axis.plot(closed, closed_radius, color=color, lw=1.65, marker='o', markersize=3.5, markeredgewidth=0, zorder=zorder + 2)
    labels = counts['network_abbreviation'].tolist()
    for (angle, label) in zip(angles, labels):
        x = 0.5 + 0.515 * np.sin(angle)
        y = 0.5 + 0.515 * np.cos(angle)
        ha = 'left' if x > 0.56 else 'right' if x < 0.44 else 'center'
        va = 'bottom' if y > 0.56 else 'top' if y < 0.44 else 'center'
        axis.text(x, y, label, transform=axis.transAxes, ha=ha, va=va, fontsize=9.6, color=INK)
    for (raw, radius, color) in ((negative, neg_radius, NEGATIVE), (positive, pos_radius, POSITIVE)):
        index = int(np.argmax(raw))
        if raw[index] > 0:
            text_radius = radius[index] - maximum * 0.055 if radius[index] >= maximum * 0.96 else radius[index] + maximum * 0.045
            axis.text(angles[index], text_radius, str(int(raw[index])), ha='center', va='top' if radius[index] >= maximum * 0.96 else 'bottom', fontsize=8.8, fontweight='bold', color=color)
    fig.text(0.07, 0.985, 'Network distribution of significant G2 effects', ha='left', va='top', fontsize=8.2, fontweight='bold', color=INK)
    fig.text(0.07, 0.94, 'Positive effects cluster primarily in VIS and SMN', ha='left', va='top', fontsize=5.5, color='#747474')
    fig.subplots_adjust(left=0.07, right=0.93, bottom=0.06, top=0.86)
    save_submission_figure(fig, 'G2_FDR_network_distribution')
if __name__ == '__main__':
    main()
