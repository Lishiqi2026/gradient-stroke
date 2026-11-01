from __future__ import annotations
import os as _portable_os
from pathlib import Path as _PortablePath

def _portable_path(kind, relative):
    project = _PortablePath(_portable_os.environ['PAPER_PROJECT_ROOT'])
    root = project if kind == 'project' else project.parent
    return str(root / relative)
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
ROOT = Path(_portable_path('project', 'ml_analysis'))
RESULT_ROOT = ROOT / 'outputs' / 'five_outcome_lesion_informed'
FIGURE_ROOT = RESULT_ROOT / 'figures_nature' / 'modelwise_five_tasks_forest'
OUTCOMES = ('NIHSS', 'ADL', 'MMSE', 'HAMD', 'VFT')
ALGORITHMS = ('ridge', 'elasticnet')
ALGORITHM_LABELS = {'ridge': 'Ridge', 'elasticnet': 'Elastic Net'}
ALGORITHM_COLORS = {'ridge': '#565681', 'elasticnet': '#C77C98'}
OUTCOME_COLORS = {'NIHSS': '#596E9A', 'ADL': '#72A5A0', 'MMSE': '#D1955B', 'HAMD': '#A97891', 'VFT': '#899A67'}
BLOCK_LABELS = {'clinical_between_disp': 'G1/G2 betweenDisp', 'clinical_g2_vis_smn': 'Targeted G2 VIS/SMN', 'clinical_g2_network_all': 'All-network G2', 'clinical_healthy_deviation': 'Healthy deviation', 'clinical_lesion_topology': 'Lesion topology', 'clinical_g2_lesion_curated': 'Curated G2/lesion', 'clinical_all_baseline': 'All baseline imaging'}
mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'], 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.size': 7, 'axes.titlesize': 7.5, 'axes.labelsize': 7, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'axes.linewidth': 0.7, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})

def mm_to_inches(width_mm: float, height_mm: float) -> tuple[float, float]:
    return (width_mm / 25.4, height_mm / 25.4)

def summarize_repeats(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()
    frame['post_mse'] = frame['post_rmse'] ** 2
    summary = frame.groupby(['algorithm', 'outcome', 'block'], as_index=False).agg(n_repeats=('repeat', 'nunique'), r2_mean=('post_r2', 'mean'), r2_p025=('post_r2', lambda x: x.quantile(0.025)), r2_p975=('post_r2', lambda x: x.quantile(0.975)), mse_mean=('post_mse', 'mean'), mse_p025=('post_mse', lambda x: x.quantile(0.025)), mse_p975=('post_mse', lambda x: x.quantile(0.975)), mae_mean=('post_mae', 'mean'), mae_p025=('post_mae', lambda x: x.quantile(0.025)), mae_p975=('post_mae', lambda x: x.quantile(0.975)))
    summary['mse_rank_within_task'] = summary.groupby(['algorithm', 'outcome'])['mse_mean'].rank(method='min', ascending=True)
    summary['is_task_best'] = summary['mse_rank_within_task'] == 1
    summary['block_label'] = summary['block'].map(BLOCK_LABELS)
    return summary

def build_positions(block_order: list[str]) -> tuple[dict[tuple[str, str], float], dict[str, tuple[float, float]]]:
    positions: dict[tuple[str, str], float] = {}
    spans: dict[str, tuple[float, float]] = {}
    cursor = 0.0
    for (outcome_index, outcome) in enumerate(OUTCOMES):
        start = cursor
        for block in block_order:
            positions[outcome, block] = cursor
            cursor += 1.0
        end = cursor - 1.0
        spans[outcome] = (start, end)
        if outcome_index < len(OUTCOMES) - 1:
            cursor += 0.75
    return (positions, spans)

def save_bundle(fig: plt.Figure, base: Path) -> list[Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for suffix in ('svg', 'pdf', 'png', 'tiff'):
        path = base.with_suffix(f'.{suffix}')
        kwargs: dict[str, object] = {'facecolor': 'white'}
        if suffix in ('png', 'tiff'):
            kwargs['dpi'] = 600
        if suffix == 'tiff':
            kwargs['pil_kwargs'] = {'compression': 'tiff_lzw'}
        fig.savefig(path, **kwargs)
        paths.append(path)
    plt.close(fig)
    return paths

def make_forest(summary: pd.DataFrame, algorithm: str) -> tuple[list[Path], pd.DataFrame]:
    subset = summary[summary['algorithm'] == algorithm].copy()
    outcome_mse_min = subset.groupby('outcome')['mse_mean'].transform('min')
    outcome_mae_min = subset.groupby('outcome')['mae_mean'].transform('min')
    subset['is_mse_best'] = np.isclose(subset['mse_mean'], outcome_mse_min)
    subset['is_mae_best'] = np.isclose(subset['mae_mean'], outcome_mae_min)
    mean_rank = subset.groupby('block')['mse_rank_within_task'].mean().sort_values()
    block_order = mean_rank.index.tolist()
    subset['cross_task_mean_mse_rank'] = subset['block'].map(mean_rank)
    (positions, spans) = build_positions(block_order)
    subset['y_position'] = [positions[o, b] for (o, b) in zip(subset['outcome'], subset['block'])]
    subset = subset.sort_values('y_position')
    subset.to_csv(FIGURE_ROOT / f'source_{algorithm}_five_tasks_forest.csv', index=False, encoding='utf-8-sig')
    max_y = max(positions.values())
    fig = plt.figure(figsize=mm_to_inches(183, 205))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.42, 2.55, 0.82], left=0.055, right=0.975, bottom=0.075, top=0.965, wspace=0.02)
    ax_labels = fig.add_subplot(grid[0, 0])
    ax_forest = fig.add_subplot(grid[0, 1], sharey=ax_labels)
    ax_table = fig.add_subplot(grid[0, 2], sharey=ax_labels)
    for (index, outcome) in enumerate(OUTCOMES):
        (y0, y1) = spans[outcome]
        if index % 2 == 0:
            for axis in (ax_labels, ax_forest, ax_table):
                axis.axhspan(y0 - 0.48, y1 + 0.48, color='#F4F5F7', zorder=0)
        ax_labels.plot([0.025, 0.025], [y0 - 0.38, y1 + 0.38], transform=ax_labels.get_yaxis_transform(), color=OUTCOME_COLORS[outcome], lw=2.4, solid_capstyle='round', clip_on=False)
    (x_min, x_max) = (-0.5, 1.0)
    ax_forest.axvline(0.0, color='#8A8A8A', lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax_forest.axvline(1.0, color='#D7D7D7', lw=0.6, zorder=1)
    ax_forest.grid(axis='x', color='#E6E6E6', lw=0.55, zorder=0)
    for (_, row) in subset.iterrows():
        y = float(row['y_position'])
        mean = float(row['r2_mean'])
        low = float(row['r2_p025'])
        high = float(row['r2_p975'])
        outcome = str(row['outcome'])
        best = bool(row['is_task_best'])
        clipped_low = max(low, x_min)
        clipped_high = min(high, x_max)
        ax_forest.plot([clipped_low, clipped_high], [y, y], color=OUTCOME_COLORS[outcome], lw=1.1, alpha=0.82, zorder=2)
        ax_forest.plot([clipped_low, clipped_low], [y - 0.13, y + 0.13], color=OUTCOME_COLORS[outcome], lw=0.8, zorder=2)
        ax_forest.plot([clipped_high, clipped_high], [y - 0.13, y + 0.13], color=OUTCOME_COLORS[outcome], lw=0.8, zorder=2)
        if low < x_min:
            ax_forest.scatter(x_min + 0.01, y, marker='<', s=15, color=OUTCOME_COLORS[outcome], edgecolor='none', zorder=3)
        ax_forest.scatter(np.clip(mean, x_min, x_max), y, s=34 if best else 22, marker='D' if best else 'o', color=OUTCOME_COLORS[outcome], edgecolor='#202020' if best else 'white', linewidth=0.55, zorder=4)
        label_weight = 'bold' if best else 'normal'
        ax_labels.text(0.1, y, BLOCK_LABELS[str(row['block'])], transform=ax_labels.get_yaxis_transform(), ha='left', va='center', fontsize=6.2, fontweight=label_weight, color='#222222')
        ax_table.text(0.29, y, f"{float(row['mse_mean']):.2f}", ha='center', va='center', fontsize=6.2, fontweight='bold' if bool(row['is_mse_best']) else 'normal')
        ax_table.text(0.77, y, f"{float(row['mae_mean']):.2f}", ha='center', va='center', fontsize=6.2, fontweight='bold' if bool(row['is_mae_best']) else 'normal')
    for outcome in OUTCOMES:
        (y0, _) = spans[outcome]
        ax_labels.text(0.0, y0 - 0.58, outcome, transform=ax_labels.get_yaxis_transform(), ha='left', va='bottom', fontsize=7.2, fontweight='bold', color=OUTCOME_COLORS[outcome])
    for axis in (ax_labels, ax_forest, ax_table):
        axis.set_ylim(max_y + 0.55, -1.1)
        axis.set_yticks([])
    ax_labels.set_xlim(0, 1)
    ax_labels.set_axis_off()
    ax_forest.set_xlim(x_min, x_max)
    ax_forest.set_xticks(np.arange(-0.5, 1.01, 0.25))
    ax_forest.set_xlabel('Mean repeated-CV post-score $R^2$ (higher is better)')
    ax_forest.tick_params(axis='x', width=0.7, length=2.5)
    ax_forest.spines['left'].set_visible(False)
    ax_forest.set_title('Mean $R^2$ and split-to-split range', loc='left', fontweight='bold', pad=8)
    ax_table.set_xlim(0, 1)
    ax_table.set_axis_off()
    ax_table.text(0.29, -0.92, 'MSE ↓', ha='center', va='bottom', fontsize=7, fontweight='bold')
    ax_table.text(0.77, -0.92, 'MAE ↓', ha='center', va='bottom', fontsize=7, fontweight='bold')
    ax_table.plot([0.04, 0.96], [-0.58, -0.58], color='#777777', lw=0.6, clip_on=False)
    paths = save_bundle(fig, FIGURE_ROOT / f'figure_{algorithm}_five_tasks_forest')
    return (paths, subset)

def write_contract() -> Path:
    contract = '# Five-task forest-plot contract\n\nCore conclusion: Within each fixed algorithm, compact clinical-plus-imaging feature sets can be compared across all five outcomes by jointly displaying predictive performance and its repeated-split stability.\n\nFigure archetype: Quantitative grid expressed as a grouped forest plot with an aligned metric table.\n\nBackend: Python only.\n\nOutput: Two independent Nature double-column figures, 183 x 205 mm; editable SVG/PDF, 600-dpi TIFF, PNG preview and CSV source data.\n\nEvidence map:\n\n- Hero evidence: mean repeated-CV post-score R2 with 2.5th-97.5th percentile split variability for all 35 task-feature combinations.\n- Supporting evidence: aligned raw MSE and MAE table; downward arrows indicate that lower values are better.\n- Visual emphasis: diamonds and bold feature labels identify the lowest-MSE feature set within each task. The minimum MSE and minimum MAE value in each task are independently bolded.\n\nStatistics: n = 37; treatment-stratified 5-fold outer CV repeated 50 times; treatment-stratified 4-fold inner tuning. MSE is RMSE squared within each repeat before aggregation.\n\nReviewer risks: intervals are split-to-split variability rather than independent-sample confidence intervals; intervals below R2 = -0.50 are visually truncated and marked by arrows; best feature sets are post-hoc exploratory selections; no clinical-only comparator is included.\n'
    path = FIGURE_ROOT / 'FIGURE_CONTRACT.md'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contract, encoding='utf-8')
    return path

def main() -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    repeat_level = pd.read_csv(RESULT_ROOT / 'repeat_level_performance.csv')
    summary = summarize_repeats(repeat_level)
    outputs: list[Path] = []
    for algorithm in ALGORITHMS:
        (paths, _) = make_forest(summary, algorithm)
        outputs.extend(paths)
    contract = write_contract()
    manifest = {'backend': 'Python', 'expected_figure_files': len(outputs), 'all_files_exist': all((path.exists() for path in outputs)), 'all_files_nonempty': all((path.stat().st_size > 0 for path in outputs)), 'all_svg_have_editable_text': all(('<text' in path.read_text(encoding='utf-8') for path in outputs if path.suffix == '.svg')), 'n_algorithms': 2, 'n_rows_per_figure': 35, 'n_repeats_per_cell': sorted(summary['n_repeats'].unique().tolist()), 'forest_metric': 'post-score R2 mean with 2.5th-97.5th repeated-split percentiles', 'right_table_metrics': ['raw MSE', 'raw MAE'], 'contract': str(contract), 'outputs': [str(path) for path in outputs]}
    qa_path = FIGURE_ROOT / 'figure_forest_qa_manifest.json'
    qa_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))
if __name__ == '__main__':
    main()
