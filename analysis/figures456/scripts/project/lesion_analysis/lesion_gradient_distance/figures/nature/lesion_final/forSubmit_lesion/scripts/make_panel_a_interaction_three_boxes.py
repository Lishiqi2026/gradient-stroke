from __future__ import annotations
import os as _portable_os
from pathlib import Path as _PortablePath

def _portable_path(kind, relative):
    project = _PortablePath(_portable_os.environ['PAPER_PROJECT_ROOT'])
    root = project if kind == 'project' else project.parent
    return str(root / relative)
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from figure_common import MID, SOURCE, save_submission_figure
from make_panel_a_interaction_three_versions import HIGH, LOW, MEAN, MODEL_TERMS, ZERO, load_analysis, statistics_text, style_axis

def bootstrap_slopes(data: pd.DataFrame, repeats: int=5000) -> pd.DataFrame:
    design = sm.add_constant(data[MODEL_TERMS].astype(float), has_constant='add')
    names = list(design.columns)
    matrix = design.to_numpy(float)
    outcome = data['MMSE_improvement_z'].to_numpy(float)
    group = data['group_tms'].to_numpy(int)
    control = np.flatnonzero(group == 0)
    tms = np.flatnonzero(group == 1)
    gradient_index = names.index('m_z')
    interaction_index = names.index('x_by_m')
    rng = np.random.default_rng(20260822)
    output = np.empty((repeats, 3), dtype=float)
    levels = np.asarray([-1.0, 0.0, 1.0])
    for bootstrap_id in range(repeats):
        sampled = np.concatenate([rng.choice(control, len(control), replace=True), rng.choice(tms, len(tms), replace=True)])
        coefficients = np.linalg.lstsq(matrix[sampled], outcome[sampled], rcond=None)[0]
        output[bootstrap_id] = coefficients[gradient_index] + levels * coefficients[interaction_index]
    return pd.DataFrame({'bootstrap_id': np.repeat(np.arange(1, repeats + 1), 3), 'lesion_span_z': np.tile(levels, repeats), 'span_label': np.tile(['Low span (−1 SD)', 'Mean span', 'High span (+1 SD)'], repeats), 'conditional_gradient_MMSE_slope': output.reshape(-1)})

def main() -> None:
    (data, _, moderation, slopes) = load_analysis()
    bootstrap = bootstrap_slopes(data)
    slope_lookup = slopes.set_index('lesion_span_level')
    specifications = [('Low span (−1 SD)', 'span_minus_1SD', LOW), ('Mean span', 'span_mean', MEAN), ('High span (+1 SD)', 'span_plus_1SD', HIGH)]
    positions = [0.0, 0.72, 1.44]
    box_rows: list[dict[str, object]] = []
    for (label, key, _) in specifications:
        values = bootstrap.loc[bootstrap['span_label'].eq(label), 'conditional_gradient_MMSE_slope'].to_numpy(float)
        box_rows.append({'label': label, 'key': key, 'med': float(np.quantile(values, 0.5)), 'q1': float(np.quantile(values, 0.25)), 'q3': float(np.quantile(values, 0.75)), 'whislo': float(np.quantile(values, 0.025)), 'whishi': float(np.quantile(values, 0.975)), 'fliers': []})
    (fig, axis) = plt.subplots(figsize=(102 / 25.4, 78 / 25.4), facecolor='white')
    for (position, box, (_, key, color)) in zip(positions, box_rows, specifications):
        axis.bxp([box], positions=[position], widths=0.4, patch_artist=True, showfliers=False, manage_ticks=False, boxprops={'facecolor': mpl.colors.to_rgba(color, 0.24), 'edgecolor': color, 'linewidth': 1.0}, medianprops={'color': color, 'linewidth': 1.35}, whiskerprops={'color': color, 'linewidth': 0.9}, capprops={'color': color, 'linewidth': 0.9})
        row = slope_lookup.loc[key]
        beta = float(row['gradient_slope'])
        axis.scatter(position, beta, marker='D', s=31, facecolor='white', edgecolor=color, linewidth=1.0, zorder=5)
        axis.text(position, float(box['whishi']) + 0.08, f"β={beta:.2f}\nHC3 P={float(row['p_hc3']):.3f}", ha='center', va='bottom', fontsize=5.8, color=color, linespacing=1.1)
    axis.axhline(0, color=ZERO, linewidth=0.75, linestyle=(0, (3, 2)))
    axis.set_xlim(-0.5, 1.94)
    axis.set_xticks(positions)
    axis.set_xticklabels(['Low span\n(−1 SD)', 'Mean span', 'High span\n(+1 SD)'])
    axis.set_ylim(min((float(box['whislo']) for box in box_rows)) - 0.14, max((float(box['whishi']) for box in box_rows)) + 0.35)
    axis.set_ylabel('Conditional normalization–MMSE slope (β)')
    axis.set_title('Conditional gradient–MMSE slopes across lesion span', loc='left', fontweight='bold', pad=5)
    axis.text(0.01, 0.985, statistics_text(moderation) + '\nn=37; exploratory', transform=axis.transAxes, ha='left', va='top', fontsize=5.7, color=MID, linespacing=1.25)
    axis.text(0.99, 0.02, 'Boxes: bootstrap IQR; whiskers: 95% percentile interval; diamonds: model β', transform=axis.transAxes, ha='right', va='bottom', fontsize=5.3, color=MID)
    style_axis(axis)
    fig.subplots_adjust(left=0.19, right=0.97, bottom=0.19, top=0.84)
    save_submission_figure(fig, 'panel_a_interaction_three_bootstrap_boxes')
    bootstrap.to_csv(SOURCE / 'panel_a_interaction_three_boxes_bootstrap_slopes.csv', index=False, encoding='utf-8-sig')
if __name__ == '__main__':
    main()
