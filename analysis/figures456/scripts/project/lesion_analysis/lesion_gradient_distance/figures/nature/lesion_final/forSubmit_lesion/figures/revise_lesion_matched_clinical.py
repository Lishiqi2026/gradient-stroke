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
import numpy as np
import pandas as pd
SCRIPT_DIR = Path(__file__).resolve().parent.parent / 'scripts'
sys.path.insert(0, str(SCRIPT_DIR))
from figure_common import INK, SOURCE, save_submission_figure
RNG = np.random.default_rng(20260821)
CONTROL_DARK = '#3F73A8'
TMS_DARK = '#D6534A'

def draw_outcome(axis: plt.Axes, data: pd.DataFrame, statistics: pd.Series, outcome: str) -> None:
    frame = data.query('outcome == @outcome')
    for (position, (group, color)) in enumerate((('Control', CONTROL_DARK), ('TMS', TMS_DARK))):
        values = frame.query('group == @group')['improvement'].to_numpy(float)
        x = position + RNG.normal(0, 0.022, len(values))
        axis.scatter(x, values, s=25, color=color, alpha=0.94, edgecolor='white', linewidth=0.42, zorder=3)
        mean = float(values.mean())
        axis.plot([position - 0.16, position + 0.16], [mean, mean], color=INK, lw=1.15, zorder=4)
    difference = float(statistics['improvement_difference_tms_minus_control'])
    q_value = float(statistics['permutation_fdr_q'])
    axis.text(0.03, 0.97, f'difference={difference:.1f}; q={q_value:.3f}', transform=axis.transAxes, ha='left', va='top', fontsize=6.2)
    axis.axhline(0, color='#D6D6D6', lw=0.65, zorder=0)
    axis.set_xticks([0, 1])
    axis.set_xticklabels(['Control\n(n=5)', 'TMS\n(n=5)'])
    axis.set_xlim(-0.38, 1.38)
    axis.set_title(outcome, fontweight='bold', pad=5)
    axis.set_ylabel(f'{outcome} improvement (points)', labelpad=3)
    axis.tick_params(axis='both', pad=2)
    if outcome == 'NIHSS':
        axis.set_ylim(-0.2, 4.25)
    else:
        axis.set_ylim(-2, 47)

def main() -> None:
    data = pd.read_csv(SOURCE / 'lesion_matched_clinical_improvements.csv', dtype={'subject_id': str})
    statistics = pd.read_csv(SOURCE / 'lesion_matched_clinical_statistics.csv')
    (width_mm, height_mm) = (120.0, 68.0)
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), facecolor='white')
    grid = fig.add_gridspec(1, 2, left=0.115, right=0.985, bottom=0.205, top=0.79, wspace=0.34)
    axes = [fig.add_subplot(grid[0, column]) for column in range(2)]
    draw_outcome(axes[0], data, statistics.query("outcome == 'NIHSS'").iloc[0], 'NIHSS')
    draw_outcome(axes[1], data, statistics.query("outcome == 'ADL'").iloc[0], 'ADL')
    fig.text(0.115, 0.955, 'Exact basal ganglia/periventricular lesion subgroup', ha='left', va='top', fontsize=8.2, fontweight='bold', color=INK)
    save_submission_figure(fig, 'lesion_matched_clinical_improvement')
if __name__ == '__main__':
    main()
