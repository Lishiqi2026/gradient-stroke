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
from figure_common import INK, MID, SOURCE, save_submission_figure
from make_lesion_longitudinal_volume import CONTROL_POST_POINT, CONTROL_PRE_POINT, PAIR_LINE, TMS_POST_POINT, TMS_PRE_POINT, prepare_plot_data
RNG = np.random.default_rng(20260822)

def draw_volume_group_raw(axis: plt.Axes, data: pd.DataFrame, stats: pd.Series, group: str) -> None:
    frame = data.query('group == @group').pivot(index='subject_id', columns='time', values='lesion_volume_ml').reindex(columns=['Pre', 'Post']).dropna().sort_index()
    pre = frame['Pre'].to_numpy(float)
    post = frame['Post'].to_numpy(float)
    if group == 'Control':
        pre_color = CONTROL_PRE_POINT
        post_color = CONTROL_POST_POINT
        lower = -5.0
        upper = 75.0
        ticks = [0, 15, 30, 45, 60, 75]
    else:
        pre_color = TMS_PRE_POINT
        post_color = TMS_POST_POINT
        upper = 20.0
        lower = -upper / 15.0
        ticks = [0, 5, 10, 15, 20]
    for (before, after) in zip(pre, post):
        jitter = float(RNG.uniform(-0.035, 0.035))
        axis.plot([jitter, 1 + jitter], [before, after], color=PAIR_LINE, lw=0.82, alpha=0.52, zorder=1)
        axis.scatter(jitter, before, s=30, facecolor=pre_color, edgecolor='none', linewidth=0.0, alpha=0.94, zorder=3)
        axis.scatter(1 + jitter, after, s=30, facecolor=post_color, edgecolor='none', linewidth=0.0, alpha=0.94, zorder=3)
    axis.set_yticks(ticks)
    axis.set_ylim(lower, upper)
    axis.set_xticks([0, 1])
    axis.set_xticklabels(['Pre', 'Post'])
    axis.set_xlim(-0.38, 1.38)
    axis.set_ylabel('Lesion volume (ml)')
    axis.set_title(f'{group} (n={len(frame)})', color=post_color, fontweight='bold', fontsize=8.0, pad=8)
    axis.grid(False)
    maximum = max(float(pre.max()), float(post.max()))
    bracket = maximum + 0.17 * upper
    cap = bracket - 0.025 * upper
    q_value = float(stats['wilcoxon_fdr_q'])
    stars = '***' if q_value < 0.001 else '**' if q_value < 0.01 else '*' if q_value < 0.05 else 'ns'
    axis.plot([-0.1, -0.1, 1.1, 1.1], [cap, bracket, bracket, cap], color=INK, lw=0.78, clip_on=False, zorder=5)
    axis.text(0.5, bracket + 0.035 * upper, stars, ha='center', va='bottom', fontsize=8.0, fontweight='bold', color=INK)

def main() -> None:
    volume = pd.read_csv(SOURCE / 'lesion_volume_subjects.csv', dtype={'subject_id': str})
    (selected, statistics) = prepare_plot_data(volume)
    (width_mm, height_mm) = (120.0, 76.0)
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), facecolor='white')
    grid = fig.add_gridspec(1, 2, left=0.125, right=0.985, bottom=0.165, top=0.805, wspace=0.46)
    for (column, group) in enumerate(('Control', 'TMS')):
        stats = statistics.query('group == @group').iloc[0]
        draw_volume_group_raw(fig.add_subplot(grid[0, column]), selected, stats, group)
    fig.text(0.125, 0.955, 'Paired longitudinal lesion-volume change', ha='left', va='top', fontsize=8.2, fontweight='bold', color=INK)
    fig.text(0.985, 0.955, 'Independent linear y-axis ranges', ha='right', va='top', fontsize=5.7, color=MID)
    save_submission_figure(fig, 'lesion_longitudinal_volume_raw_ml')
if __name__ == '__main__':
    main()
