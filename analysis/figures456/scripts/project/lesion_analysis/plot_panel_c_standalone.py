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
import pandas as pd
import plot_lesion_matched_clinical_nature as base
HERE = Path(__file__).resolve().parent
ANALYSIS = HERE / 'lesion_matched_clinical_mediation'
TABLES = ANALYSIS / 'tables'
FIGURES = ANALYSIS / 'figures'
SOURCE = ANALYSIS / 'figure_source_data'
WIDTH_MM = 89.0
HEIGHT_MM = 65.0

def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)
    exact = pd.read_csv(TABLES / 'exact_basal_ganglia_periventricular_subjects.csv', dtype={'subject_id': str})
    results = pd.read_csv(TABLES / 'exact_basal_ganglia_periventricular_clinical_results.csv')
    statistics = results.loc[results['outcome'].isin(['NIHSS', 'ADL'])].copy()
    statistics.to_csv(SOURCE / 'panel_c_exact_location_statistics.csv', index=False)
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), facecolor='white')
    grid = fig.add_gridspec(1, 2, left=0.145, right=0.985, bottom=0.235, top=0.745, wspace=0.46)
    ax_nihss = fig.add_subplot(grid[0, 0])
    ax_adl = fig.add_subplot(grid[0, 1])
    base.plot_exact_improvement(ax_nihss, exact, results, 'NIHSS')
    base.plot_exact_improvement(ax_adl, exact, results, 'ADL')
    fig.text(0.145, 0.935, 'Exact basal ganglia/periventricular subgroup', ha='left', va='top', fontsize=7.0, fontweight='bold', color=base.INK)
    base.panel_label(fig, 0.025, 0.94, 'c')
    stem = FIGURES / 'panel_c_exact_location_improvement'
    fig.savefig(stem.with_suffix('.svg'), facecolor='white')
    fig.savefig(stem.with_suffix('.pdf'), facecolor='white')
    fig.savefig(stem.with_suffix('.tiff'), dpi=600, facecolor='white')
    fig.savefig(stem.with_suffix('.png'), dpi=300, facecolor='white')
    plt.close(fig)
    print(f'Standalone panel c: {stem}')
    print(f'Size: {WIDTH_MM:.0f} x {HEIGHT_MM:.0f} mm')
if __name__ == '__main__':
    main()
