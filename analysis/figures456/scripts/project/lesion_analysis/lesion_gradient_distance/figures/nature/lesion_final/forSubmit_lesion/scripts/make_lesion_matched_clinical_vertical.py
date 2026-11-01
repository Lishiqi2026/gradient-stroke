from __future__ import annotations
'Compact right-column layout for the lesion-matched clinical comparison.\n\nFigure contract\n---------------\nCore conclusion: In an exact lesion-location subgroup, clinical improvement is\nlarger in TMS than Control for both NIHSS and ADL.\nArchetype: quantitative clinical strip for insertion beside lesion panels.\nBackend/output: Python; editable PDF/SVG plus 600-dpi PNG/TIFF.\nReviewer risk: n=5 per group and the permutation/FDR definition must remain visible.\n'
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
CONTROL = '#3F73A8'
TMS = '#D6534A'
GROUP_POSITIONS = {'Control': 0.0, 'TMS': 0.56}
RNG = np.random.default_rng(20260822)

def significance_stars(q_value: float) -> str:
    if q_value < 0.001:
        return '***'
    if q_value < 0.01:
        return '**'
    if q_value < 0.05:
        return '*'
    return 'ns'

def draw_panel(axis: plt.Axes, data: pd.DataFrame, statistics: pd.Series, outcome: str, *, show_xlabels: bool) -> None:
    frame = data.query('outcome == @outcome')
    for (group, color) in (('Control', CONTROL), ('TMS', TMS)):
        position = GROUP_POSITIONS[group]
        values = frame.query('group == @group')['improvement'].to_numpy(float)
        jitter = RNG.normal(0, 0.014, len(values))
        axis.scatter(position + jitter, values, s=21, facecolor=color, edgecolor='white', linewidth=0.4, alpha=0.92, zorder=3)
        mean = float(values.mean())
        axis.plot([position - 0.09, position + 0.09], [mean, mean], color=INK, linewidth=1.1, zorder=4)
    difference = float(statistics['improvement_difference_tms_minus_control'])
    q_value = float(statistics['permutation_fdr_q'])
    star = significance_stars(q_value)
    if outcome == 'NIHSS':
        axis.set_ylim(-0.2, 4.9)
        axis.set_yticks([0, 1, 2, 3, 4])
        (bracket_y, cap_y, text_y) = (4.45, 4.32, 4.55)
    else:
        axis.set_ylim(-2.0, 52.0)
        axis.set_yticks([0, 10, 20, 30, 40, 50])
        (bracket_y, cap_y, text_y) = (48.0, 46.6, 49.0)
    left = GROUP_POSITIONS['Control']
    right = GROUP_POSITIONS['TMS']
    axis.plot([left, left, right, right], [cap_y, bracket_y, bracket_y, cap_y], color=INK, linewidth=0.75, clip_on=False, zorder=5)
    axis.text((left + right) / 2, text_y, star, ha='center', va='bottom', fontsize=7.4, fontweight='bold', color=INK)
    axis.text(0.0, 1.035, outcome, transform=axis.transAxes, ha='left', va='bottom', fontsize=7.2, fontweight='bold', color=INK)
    axis.text(1.0, 1.035, f'TMS-Control difference = {difference:.1f}', transform=axis.transAxes, ha='right', va='bottom', fontsize=5.4, color=MID)
    axis.axhline(0, color='#D6D6D6', linewidth=0.6, zorder=0)
    axis.set_xlim(-0.17, 0.73)
    axis.set_xticks([left, right])
    if show_xlabels:
        axis.set_xticklabels(['Control\n(n=5)', 'TMS\n(n=5)'])
    else:
        axis.set_xticklabels([])
        axis.tick_params(axis='x', length=0)
    axis.tick_params(axis='both', pad=1.5)
    axis.grid(False)

def main() -> None:
    data = pd.read_csv(SOURCE / 'lesion_matched_clinical_improvements.csv', dtype={'subject_id': str})
    statistics = pd.read_csv(SOURCE / 'lesion_matched_clinical_statistics.csv')
    (width_mm, height_mm) = (64.0, 70.0)
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), facecolor='white')
    grid = fig.add_gridspec(2, 1, left=0.2, right=0.985, bottom=0.17, top=0.845, hspace=0.39)
    axes = [fig.add_subplot(grid[row, 0]) for row in range(2)]
    draw_panel(axes[0], data, statistics.query("outcome == 'NIHSS'").iloc[0], 'NIHSS', show_xlabels=False)
    draw_panel(axes[1], data, statistics.query("outcome == 'ADL'").iloc[0], 'ADL', show_xlabels=True)
    fig.text(0.2, 0.965, 'Clinical improvement in a lesion-matched subgroup', ha='left', va='top', fontsize=7.7, fontweight='bold', color=INK)
    fig.text(0.2, 0.914, 'Exact basal ganglia/periventricular lesions; n=5 per group', ha='left', va='top', fontsize=5.3, color=MID)
    fig.text(0.045, 0.515, 'Clinical improvement (points)', ha='center', va='center', rotation=90, fontsize=6.5, color=INK)
    fig.text(0.985, 0.035, '* permutation FDR q=0.026', ha='right', va='bottom', fontsize=5.1, color=MID)
    save_submission_figure(fig, 'lesion_matched_clinical_improvement_vertical')
if __name__ == '__main__':
    main()
