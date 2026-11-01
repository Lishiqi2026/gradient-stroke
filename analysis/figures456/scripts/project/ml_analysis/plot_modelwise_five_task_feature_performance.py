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
from matplotlib.lines import Line2D
ROOT = Path(_portable_path('project', 'ml_analysis'))
RESULT_ROOT = ROOT / 'outputs' / 'five_outcome_lesion_informed'
FIGURE_ROOT = RESULT_ROOT / 'figures_nature' / 'modelwise_five_tasks'
OUTCOMES = ('NIHSS', 'ADL', 'MMSE', 'HAMD', 'VFT')
ALGORITHMS = ('ridge', 'elasticnet')
ALGORITHM_LABELS = {'ridge': 'Ridge', 'elasticnet': 'Elastic Net'}
ALGORITHM_COLORS = {'ridge': '#565681', 'elasticnet': '#C77C98'}
BLOCK_LABELS = {'clinical_between_disp': 'G1/G2 betweenDisp', 'clinical_g2_vis_smn': 'Targeted G2 VIS/SMN', 'clinical_g2_network_all': 'All-network G2', 'clinical_healthy_deviation': 'Healthy deviation', 'clinical_lesion_topology': 'Lesion topology', 'clinical_g2_lesion_curated': 'Curated G2/lesion', 'clinical_all_baseline': 'All baseline imaging'}
BLOCK_SHORT_LABELS = {'clinical_between_disp': 'BtwDisp', 'clinical_g2_vis_smn': 'VIS/SMN', 'clinical_g2_network_all': 'All G2', 'clinical_healthy_deviation': 'Healthy', 'clinical_lesion_topology': 'Lesion', 'clinical_g2_lesion_curated': 'Curated', 'clinical_all_baseline': 'All img.'}
OUTCOME_COLORS = {'NIHSS': '#596E9A', 'ADL': '#77A6A1', 'MMSE': '#D19A66', 'HAMD': '#A77791', 'VFT': '#8E9B6E'}
OUTCOME_MARKERS = {'NIHSS': 'o', 'ADL': 's', 'MMSE': '^', 'HAMD': 'D', 'VFT': 'P'}
mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'], 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.size': 7, 'axes.titlesize': 8, 'axes.labelsize': 7, 'xtick.labelsize': 6.3, 'ytick.labelsize': 6.3, 'axes.linewidth': 0.7, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False, 'legend.fontsize': 6.2})

def mm_to_inches(width_mm: float, height_mm: float) -> tuple[float, float]:
    return (width_mm / 25.4, height_mm / 25.4)

def panel_label(ax: plt.Axes, label: str, x: float=-0.13, y: float=1.03) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=8, fontweight='bold', ha='left', va='bottom')

def text_color(cmap, norm, value: float) -> str:
    (r, g, b, _) = cmap(norm(value))
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return 'white' if luminance < 0.52 else '#202020'

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

def build_source_data(repeat_level: pd.DataFrame) -> pd.DataFrame:
    data = repeat_level.copy()
    data['post_mse'] = data['post_rmse'] ** 2
    summary = data.groupby(['algorithm', 'outcome', 'block'], as_index=False).agg(n_repeats=('repeat', 'nunique'), mse_mean=('post_mse', 'mean'), mse_median=('post_mse', 'median'), mse_p025=('post_mse', lambda x: x.quantile(0.025)), mse_p975=('post_mse', lambda x: x.quantile(0.975)), rmse_mean=('post_rmse', 'mean'), mae_mean=('post_mae', 'mean'), r2_mean=('post_r2', 'mean'), r2_p025=('post_r2', lambda x: x.quantile(0.025)), r2_p975=('post_r2', lambda x: x.quantile(0.975)))
    summary['mse_rank_within_task'] = summary.groupby(['algorithm', 'outcome'])['mse_mean'].rank(method='min', ascending=True)
    best_mse = summary.groupby(['algorithm', 'outcome'])['mse_mean'].transform('min')
    summary['relative_mse_to_task_best'] = summary['mse_mean'] / best_mse
    summary['block_label'] = summary['block'].map(BLOCK_LABELS)
    return summary

def heatmap(ax: plt.Axes, matrix: pd.DataFrame, annotations: pd.DataFrame, cmap, norm, title: str, fmt: str) -> mpl.image.AxesImage:
    values = matrix.to_numpy(dtype=float)
    labels = annotations.to_numpy(dtype=float)
    image = ax.imshow(values, cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            ax.text(col, row, format(labels[row, col], fmt), ha='center', va='center', fontsize=5.5, color=text_color(cmap, norm, values[row, col]))
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels([BLOCK_SHORT_LABELS[x] for x in matrix.columns], ha='center', linespacing=0.95)
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(matrix.index)
    ax.tick_params(axis='both', length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, loc='left', fontweight='bold', pad=4)
    return image

def make_model_figure(source: pd.DataFrame, algorithm: str) -> tuple[list[Path], pd.DataFrame]:
    subset = source[source['algorithm'] == algorithm].copy()
    order = subset.groupby('block')['mse_rank_within_task'].mean().sort_values().index.tolist()
    subset['mean_cross_task_rank'] = subset['block'].map(subset.groupby('block')['mse_rank_within_task'].mean())
    subset = subset.sort_values(['outcome', 'mse_rank_within_task'])
    source_path = FIGURE_ROOT / f'source_{algorithm}_five_tasks_all_features.csv'
    subset.to_csv(source_path, index=False, encoding='utf-8-sig')
    relative = subset.pivot(index='outcome', columns='block', values='relative_mse_to_task_best').loc[list(OUTCOMES), order]
    raw_mse = subset.pivot(index='outcome', columns='block', values='mse_mean').loc[list(OUTCOMES), order]
    r2 = subset.pivot(index='outcome', columns='block', values='r2_mean').loc[list(OUTCOMES), order]
    fig = plt.figure(figsize=mm_to_inches(183, 130))
    grid = fig.add_gridspec(4, 2, height_ratios=[1.0, 0.09, 0.86, 0.13], left=0.175, right=0.975, bottom=0.055, top=0.9, hspace=0.72, wspace=0.34)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    cax_a = fig.add_subplot(grid[1, 0])
    cax_b = fig.add_subplot(grid[1, 1])
    ax_c = fig.add_subplot(grid[2, :])
    ax_note = fig.add_subplot(grid[3, :])
    mse_cmap = mpl.colormaps['Blues_r']
    mse_max = max(1.35, float(np.nanpercentile(relative.to_numpy(), 95)))
    mse_norm = mpl.colors.Normalize(vmin=1.0, vmax=mse_max)
    image_a = heatmap(ax_a, relative, raw_mse, mse_cmap, mse_norm, 'Prediction error across tasks', '.2f')
    cb_a = fig.colorbar(image_a, cax=cax_a, orientation='horizontal')
    cb_a.set_label('MSE relative to task-best model (lower is better)')
    cb_a.ax.tick_params(length=2, width=0.6, labelsize=6)
    panel_label(ax_a, 'a', x=-0.13)
    r2_cmap = mpl.colormaps['RdBu']
    r2_min = min(-0.25, float(np.nanpercentile(r2.to_numpy(), 5)))
    r2_norm = mpl.colors.TwoSlopeNorm(vmin=r2_min, vcenter=0.0, vmax=1.0)
    image_b = heatmap(ax_b, r2, r2, r2_cmap, r2_norm, 'Explained variance across tasks', '.2f')
    cb_b = fig.colorbar(image_b, cax=cax_b, orientation='horizontal')
    cb_b.set_label('Mean repeated-CV post-score $R^2$ (higher is better)')
    cb_b.ax.tick_params(length=2, width=0.6, labelsize=6)
    panel_label(ax_b, 'b', x=-0.13)
    y_positions = np.arange(len(order))[::-1]
    rng = np.random.default_rng(20260811)
    for (y, block) in zip(y_positions, order):
        block_data = subset[subset['block'] == block].set_index('outcome').loc[list(OUTCOMES)]
        ranks = block_data['mse_rank_within_task'].to_numpy(dtype=float)
        ax_c.plot([ranks.min(), ranks.max()], [y, y], color='#D0D0D0', lw=1.2, zorder=1)
        for outcome in OUTCOMES:
            rank = float(block_data.loc[outcome, 'mse_rank_within_task'])
            ax_c.scatter(rank, y + rng.uniform(-0.07, 0.07), s=22, marker=OUTCOME_MARKERS[outcome], color=OUTCOME_COLORS[outcome], edgecolor='white', linewidth=0.45, zorder=2)
        mean_rank = float(ranks.mean())
        ax_c.scatter(mean_rank, y, s=42, marker='D', color='#262626', edgecolor='white', linewidth=0.5, zorder=3)
        ax_c.text(7.14, y, f'mean {mean_rank:.1f}', ha='left', va='center', fontsize=6, color='#4A4A4A')
    ax_c.set_yticks(y_positions)
    ax_c.set_yticklabels([BLOCK_LABELS[x] for x in order])
    ax_c.set_xlim(0.7, 7.75)
    ax_c.set_xticks(range(1, 8))
    ax_c.set_xlabel('MSE rank within each task (1 = lowest error)')
    ax_c.set_title('Cross-task consistency', loc='left', fontweight='bold', pad=4)
    ax_c.spines['left'].set_visible(False)
    ax_c.tick_params(axis='y', length=0)
    ax_c.tick_params(axis='x', length=2.5, width=0.7)
    panel_label(ax_c, 'c', x=-0.07)
    legend_items = [Line2D([0], [0], marker=OUTCOME_MARKERS[outcome], color='none', markerfacecolor=OUTCOME_COLORS[outcome], markeredgecolor='white', markersize=5, label=outcome) for outcome in OUTCOMES]
    legend_items.append(Line2D([0], [0], marker='D', color='none', markerfacecolor='#262626', markeredgecolor='white', markersize=5, label='Mean rank'))
    ax_c.legend(handles=legend_items, loc='lower right', bbox_to_anchor=(1.0, 1.02), ncol=6, columnspacing=0.8, handletextpad=0.25)
    fig.suptitle(f'{ALGORITHM_LABELS[algorithm]}: five post-treatment tasks across all clinical-plus-imaging feature sets', fontsize=9, fontweight='bold', color=ALGORITHM_COLORS[algorithm])
    ax_note.set_axis_off()
    ax_note.text(0.0, 0.65, 'Mean over 50 repeated stratified outer-CV partitions (n = 37). Panel a: numbers = raw MSE; colour = within-task relative MSE.\nAll models include baseline score, age, sex, ESRS and treatment group. No clinical-only comparator.', transform=ax_note.transAxes, fontsize=5.8, color='#4A4A4A', va='top')
    output_paths = save_bundle(fig, FIGURE_ROOT / f'figure_{algorithm}_five_tasks_all_features')
    return (output_paths, subset)

def write_contract() -> Path:
    text = '# Model-wise five-task figure contract\n\nCore conclusion: For each fixed regression algorithm, the figure identifies which clinical-plus-imaging feature sets maintain low prediction error and useful explained variance across all five post-treatment outcomes.\n\nFigure archetype: Quantitative grid.\n\nTarget journal/output: Nature-family double-column; editable SVG/PDF, 600-dpi TIFF, PNG preview, and CSV source data.\n\nBackend: Python only.\n\nFinal size: Two independent figures, each 183 x 130 mm.\n\nPanel map:\n\n- a: Five-task by seven-feature-set MSE grid. Numbers are raw mean MSE; colour is MSE relative to the best feature set within each task.\n- b: Matched mean repeated-CV post-score R2 grid.\n- c: MSE ranks for all five tasks and their cross-task mean rank.\n\nEvidence hierarchy:\n\n- Hero evidence: task-normalized MSE grid with raw MSE annotations.\n- Validation evidence: R2 grid.\n- Robustness evidence: cross-task MSE-rank distribution.\n\nStatistics: n = 37; treatment-stratified 5-fold outer CV repeated 50 times; treatment-stratified 4-fold inner tuning. MSE is computed per repeat as RMSE squared and then averaged.\n\nReviewer risks: Raw MSE is not directly comparable between outcomes with different score scales; colour therefore uses within-task relative MSE. Intervals reflect repeated split variability, not external-cohort uncertainty. All models contain clinical covariates and there is intentionally no clinical-only comparator.\n'
    path = FIGURE_ROOT / 'FIGURE_CONTRACT.md'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    return path

def main() -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    repeat_level = pd.read_csv(RESULT_ROOT / 'repeat_level_performance.csv')
    source = build_source_data(repeat_level)
    all_outputs: list[Path] = []
    for algorithm in ALGORITHMS:
        (paths, _) = make_model_figure(source, algorithm)
        all_outputs.extend(paths)
    contract_path = write_contract()
    checks = {'backend': 'Python', 'expected_figure_files': len(all_outputs), 'all_files_exist': all((path.exists() for path in all_outputs)), 'all_files_nonempty': all((path.stat().st_size > 0 for path in all_outputs)), 'all_svg_have_editable_text': all(('<text' in path.read_text(encoding='utf-8') for path in all_outputs if path.suffix == '.svg')), 'n_algorithms': len(ALGORITHMS), 'n_outcomes_per_algorithm': len(OUTCOMES), 'n_feature_sets_per_algorithm': len(BLOCK_LABELS), 'n_repeats_per_cell': sorted(source['n_repeats'].unique().tolist()), 'contract': str(contract_path), 'outputs': [str(path) for path in all_outputs]}
    manifest_path = FIGURE_ROOT / 'figure_modelwise_qa_manifest.json'
    manifest_path.write_text(json.dumps(checks, indent=2), encoding='utf-8')
    print(json.dumps(checks, indent=2))
if __name__ == '__main__':
    main()
