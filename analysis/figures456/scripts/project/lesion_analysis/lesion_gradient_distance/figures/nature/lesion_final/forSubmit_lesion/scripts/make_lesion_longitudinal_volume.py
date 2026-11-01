from __future__ import annotations
import os as _portable_os
from pathlib import Path as _PortablePath

def _portable_path(kind, relative):
    project = _PortablePath(_portable_os.environ['PAPER_PROJECT_ROOT'])
    root = project if kind == 'project' else project.parent
    return str(root / relative)
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter, NullLocator
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from figure_common import INK, MID, SOURCE, save_submission_figure
EXCLUDED_SUBJECT = 'PRIVATE_SUBJECT_040'
RNG = np.random.default_rng(20260822)
CONTROL_PRE_POINT = '#A8C2DD'
CONTROL_POST_POINT = '#3F73A8'
TMS_PRE_POINT = '#F2B8B0'
TMS_POST_POINT = '#D6534A'
PAIR_LINE = '#A9ADB2'

def bh_adjust(p_values: list[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0, 1)
    return result

def prepare_plot_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = data.loc[~data['subject_id'].eq(EXCLUDED_SUBJECT)].copy()
    rows: list[dict[str, object]] = []
    p_values: list[float] = []
    for group in ('Control', 'TMS'):
        frame = selected.query('group == @group').pivot(index='subject_id', columns='time', values='lesion_volume_ml').reindex(columns=['Pre', 'Post']).dropna().sort_index()
        test = wilcoxon(frame['Pre'].to_numpy(float), frame['Post'].to_numpy(float), alternative='two-sided', zero_method='wilcox', correction=False, method='approx')
        p_values.append(float(test.pvalue))
        rows.append({'group': group, 'n': len(frame), 'pre_median_ml': float(frame['Pre'].median()), 'post_median_ml': float(frame['Post'].median()), 'median_change_ml': float(np.median(frame['Post'] - frame['Pre'])), 'wilcoxon_statistic': float(test.statistic), 'wilcoxon_p': float(test.pvalue)})
    for (row, q_value) in zip(rows, bh_adjust(p_values)):
        row['wilcoxon_fdr_q'] = float(q_value)
    return (selected, pd.DataFrame(rows))

def draw_volume_group(axis: plt.Axes, data: pd.DataFrame, stats: pd.Series, group: str) -> None:
    frame = data.query('group == @group').pivot(index='subject_id', columns='time', values='lesion_volume_ml').reindex(columns=['Pre', 'Post']).dropna().sort_index()
    pre = frame['Pre'].to_numpy(float)
    post = frame['Post'].to_numpy(float)
    if group == 'Control':
        pre_color = CONTROL_PRE_POINT
        post_color = CONTROL_POST_POINT
    else:
        pre_color = TMS_PRE_POINT
        post_color = TMS_POST_POINT
    for (before, after) in zip(pre, post):
        jitter = float(RNG.uniform(-0.035, 0.035))
        axis.plot([jitter, 1 + jitter], [before, after], color=PAIR_LINE, lw=0.82, alpha=0.52, zorder=1)
        axis.scatter(jitter, before, s=30, facecolor=pre_color, edgecolor='none', linewidth=0.0, alpha=0.94, zorder=3)
        axis.scatter(1 + jitter, after, s=30, facecolor=post_color, edgecolor='none', linewidth=0.0, alpha=0.94, zorder=3)
    axis.set_yscale('log')
    if group == 'Control':
        axis.set_ylim(0.12, 160)
        ticks = [0.2, 1, 5, 25, 125]
        labels = ['0.2', '1', '5', '25', '125']
    else:
        axis.set_ylim(0.18, 35)
        ticks = [0.2, 1, 5, 25]
        labels = ['0.2', '1', '5', '25']
    axis.yaxis.set_major_locator(FixedLocator(ticks))
    axis.yaxis.set_major_formatter(FixedFormatter(labels))
    axis.yaxis.set_minor_locator(NullLocator())
    axis.yaxis.set_minor_formatter(NullFormatter())
    axis.set_xticks([0, 1])
    axis.set_xticklabels(['Pre', 'Post'])
    axis.set_xlim(-0.38, 1.38)
    axis.set_ylabel('Lesion volume (ml; log scale)')
    axis.set_title(f'{group} (n={len(frame)})', color=post_color, fontweight='bold', fontsize=8.0, pad=8)
    axis.grid(False)
    maximum = max(float(pre.max()), float(post.max()))
    bracket = maximum * 1.65
    cap = bracket / 1.08
    q_value = float(stats['wilcoxon_fdr_q'])
    stars = '***' if q_value < 0.001 else '**' if q_value < 0.01 else '*' if q_value < 0.05 else 'ns'
    axis.plot([-0.1, -0.1, 1.1, 1.1], [cap, bracket, bracket, cap], color=INK, lw=0.78, clip_on=False, zorder=5)
    axis.text(0.5, bracket * 1.06, stars, ha='center', va='bottom', fontsize=8.0, fontweight='bold', color=INK)

def main() -> None:
    volume = pd.read_csv(SOURCE / 'lesion_volume_subjects.csv', dtype={'subject_id': str})
    (selected, statistics) = prepare_plot_data(volume)
    selected.to_csv(SOURCE / 'lesion_volume_subjects_plotted.csv', index=False, encoding='utf-8-sig')
    statistics.to_csv(SOURCE / 'lesion_volume_statistics_plotted.csv', index=False, encoding='utf-8-sig')
    (width_mm, height_mm) = (120.0, 76.0)
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), facecolor='white')
    grid = fig.add_gridspec(1, 2, left=0.125, right=0.985, bottom=0.165, top=0.805, wspace=0.46)
    for (column, group) in enumerate(('Control', 'TMS')):
        stats = statistics.query('group == @group').iloc[0]
        draw_volume_group(fig.add_subplot(grid[0, column]), selected, stats, group)
    fig.text(0.125, 0.955, 'Paired longitudinal lesion-volume change', ha='left', va='top', fontsize=8.2, fontweight='bold', color=INK)
    fig.text(0.985, 0.955, 'Independent logarithmic y-axis ranges', ha='right', va='top', fontsize=5.7, color=MID)
    save_submission_figure(fig, 'lesion_longitudinal_volume')
if __name__ == '__main__':
    main()
