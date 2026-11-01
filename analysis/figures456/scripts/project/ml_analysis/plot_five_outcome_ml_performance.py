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
from sklearn.metrics import mean_absolute_error, r2_score
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.size'] = 7
plt.rcParams['axes.titlesize'] = 8
plt.rcParams['axes.labelsize'] = 7
plt.rcParams['xtick.labelsize'] = 6.5
plt.rcParams['ytick.labelsize'] = 6.5
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.linewidth'] = 0.7
plt.rcParams['legend.frameon'] = False
plt.rcParams['legend.fontsize'] = 6.5
plt.rcParams['lines.linewidth'] = 1.0
ROOT = Path(_portable_path('project', 'ml_analysis'))
PRIMARY_ROOT = ROOT / 'outputs' / 'five_outcome_lesion_informed'
FOLLOWUP_ROOT = ROOT / 'outputs' / 'one_month_followup_sensitivity'
FIGURE_ROOT = PRIMARY_ROOT / 'figures_nature'
OUTCOMES = ('NIHSS', 'ADL', 'MMSE', 'HAMD', 'VFT')
ALGORITHMS = ('ridge', 'elasticnet')
ALGORITHM_LABELS = {'ridge': 'Ridge', 'elasticnet': 'Elastic Net'}
ALGORITHM_COLORS = {'ridge': '#484878', 'elasticnet': '#D28EA5'}
ALGORITHM_MARKERS = {'ridge': 'o', 'elasticnet': 's'}
GROUP_COLORS = {'Control': '#7884B4', 'TMS': '#E28E2C'}
BLOCK_LABELS = {'clinical_between_disp': 'G1/G2 betweenDisp', 'clinical_g2_vis_smn': 'Targeted G2 VIS/SMN', 'clinical_g2_network_all': 'All-network G2', 'clinical_healthy_deviation': 'Healthy deviation', 'clinical_lesion_topology': 'Lesion topology', 'clinical_g2_lesion_curated': 'Curated G2/lesion', 'clinical_all_baseline': 'All baseline imaging'}

def mm_to_inches(width_mm: float, height_mm: float) -> tuple[float, float]:
    return (width_mm / 25.4, height_mm / 25.4)

def panel_label(ax: plt.Axes, label: str, x: float=-0.12, y: float=1.03) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=8, fontweight='bold', ha='left', va='bottom')

def style_axis(ax: plt.Axes) -> None:
    ax.spines['left'].set_linewidth(0.7)
    ax.spines['bottom'].set_linewidth(0.7)
    ax.tick_params(width=0.7, length=2.5)
    ax.grid(False)

def save_bundle(fig: plt.Figure, base: Path, dpi: int=600) -> list[str]:
    base.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in ('svg', 'pdf', 'png', 'tiff'):
        path = base.with_suffix(f'.{suffix}')
        kwargs = {'facecolor': 'white'}
        if suffix in ('png', 'tiff'):
            kwargs['dpi'] = dpi
        if suffix == 'tiff':
            kwargs['pil_kwargs'] = {'compression': 'tiff_lzw'}
        fig.savefig(path, **kwargs)
        outputs.append(str(path))
    plt.close(fig)
    return outputs

def text_color_for_value(cmap, norm, value: float) -> str:
    (r, g, b, _) = cmap(norm(value))
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return 'white' if luminance < 0.52 else '#222222'

def best_row(performance: pd.DataFrame, outcome: str) -> pd.Series:
    return performance[performance['outcome'] == outcome].sort_values('post_mae_mean').iloc[0]

def add_heatmap(ax: plt.Axes, performance: pd.DataFrame, algorithm: str, block_order: list[str], norm: mpl.colors.Normalize, cmap) -> mpl.image.AxesImage:
    subset = performance[performance['algorithm'] == algorithm]
    pivot = subset.pivot(index='outcome', columns='block', values='post_r2_mean').loc[list(OUTCOMES), block_order]
    matrix = pivot.to_numpy(dtype=float)
    image = ax.imshow(matrix, cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            ax.text(column, row, f'{value:.2f}', ha='center', va='center', fontsize=5.6, color=text_color_for_value(cmap, norm, value))
    ax.set_xticks(range(len(block_order)))
    ax.set_xticklabels([BLOCK_LABELS[block] for block in block_order], rotation=36, ha='right')
    ax.set_yticks(range(len(OUTCOMES)))
    ax.set_yticklabels(OUTCOMES)
    ax.tick_params(axis='both', length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(ALGORITHM_LABELS[algorithm], loc='left', fontweight='bold', pad=4)
    return image

def make_overview(performance: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    mean_ranks = performance.groupby('block', as_index=False).agg(mean_rank=('post_mae_rank', 'mean'), median_rank=('post_mae_rank', 'median'))
    block_order = mean_ranks.sort_values('mean_rank')['block'].tolist()
    source = performance[['algorithm', 'outcome', 'block', 'post_mae_mean', 'post_mae_p025', 'post_mae_p975', 'post_r2_mean', 'post_pearson_r_mean', 'improvement_r2_mean', 'post_mae_rank']].copy()
    source['block_label'] = source['block'].map(BLOCK_LABELS)
    source.to_csv(FIGURE_ROOT / 'source_overview_performance.csv', index=False, encoding='utf-8-sig')
    fig = plt.figure(figsize=mm_to_inches(183, 125), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, height_ratios=[1.0, 0.78, 0.07], hspace=0.1, wspace=0.12)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, :])
    ax_note = fig.add_subplot(grid[2, :])
    cmap = mpl.colormaps['RdBu']
    norm = mpl.colors.TwoSlopeNorm(vmin=-0.25, vcenter=0.0, vmax=1.0)
    image = add_heatmap(ax_a, performance, 'ridge', block_order, norm, cmap)
    add_heatmap(ax_b, performance, 'elasticnet', block_order, norm, cmap)
    panel_label(ax_a, 'a', x=-0.15)
    panel_label(ax_b, 'b', x=-0.15)
    colorbar = fig.colorbar(image, ax=[ax_a, ax_b], orientation='horizontal', shrink=0.42, pad=0.02)
    colorbar.set_label('Mean repeated-CV post-score $R^2$')
    colorbar.ax.tick_params(length=2, width=0.6, labelsize=6)
    y = np.arange(len(block_order))[::-1]
    rng = np.random.default_rng(20260811)
    for (yi, block) in zip(y, block_order):
        block_values = performance[performance['block'] == block]
        for (algorithm, offset) in (('ridge', 0.08), ('elasticnet', -0.08)):
            values = block_values[block_values['algorithm'] == algorithm]['post_mae_rank'].to_numpy(dtype=float)
            jitter = rng.uniform(-0.045, 0.045, len(values))
            ax_c.scatter(values, np.full(len(values), yi + offset) + jitter, s=15, marker=ALGORITHM_MARKERS[algorithm], color=ALGORITHM_COLORS[algorithm], edgecolor='white', linewidth=0.35, alpha=0.7, zorder=2)
        values_all = block_values['post_mae_rank'].to_numpy(dtype=float)
        ax_c.plot([np.min(values_all), np.max(values_all)], [yi, yi], color='#C8C8C8', linewidth=1.1, zorder=1)
        ax_c.scatter(np.mean(values_all), yi, s=35, color='#272727', edgecolor='white', linewidth=0.45, zorder=3)
        ax_c.text(7.12, yi, f'mean {np.mean(values_all):.1f}', va='center', ha='left', fontsize=6, color='#4D4D4D')
    ax_c.set_yticks(y)
    ax_c.set_yticklabels([BLOCK_LABELS[block] for block in block_order])
    ax_c.set_xlim(0.7, 7.75)
    ax_c.set_xticks(range(1, 8))
    ax_c.set_xlabel('MAE rank within each outcome and algorithm (1 = best)')
    ax_c.set_title('Cross-task robustness', loc='left', fontweight='bold', pad=4)
    style_axis(ax_c)
    ax_c.spines['left'].set_visible(False)
    ax_c.tick_params(axis='y', length=0)
    legend = [Line2D([0], [0], marker='o', color='none', markerfacecolor=ALGORITHM_COLORS['ridge'], markeredgecolor='white', markersize=5, label='Ridge'), Line2D([0], [0], marker='s', color='none', markerfacecolor=ALGORITHM_COLORS['elasticnet'], markeredgecolor='white', markersize=5, label='Elastic Net'), Line2D([0], [0], marker='o', color='none', markerfacecolor='#272727', markeredgecolor='white', markersize=5, label='Mean rank')]
    ax_c.legend(handles=legend, loc='lower right', bbox_to_anchor=(1.0, 1.02), ncol=3, columnspacing=1.1, handletextpad=0.35)
    panel_label(ax_c, 'c', x=-0.075)
    fig.suptitle('Post-treatment prediction across five outcomes', fontsize=9, fontweight='bold')
    ax_note.set_axis_off()
    ax_note.text(0.5, 0.35, 'All candidate models include the same outcome-specific clinical base; n=37; outer 5-fold CV repeated 50 times.', ha='center', va='center', fontsize=6, color='#4D4D4D', transform=ax_note.transAxes)
    outputs = save_bundle(fig, FIGURE_ROOT / 'figure_overview_five_outcomes')
    return (outputs, source)

def draw_performance_forest(ax: plt.Axes, performance: pd.DataFrame, outcome: str) -> list[str]:
    subset = performance[performance['outcome'] == outcome].copy()
    order = subset.groupby('block')['post_mae_mean'].mean().sort_values(ascending=True).index.tolist()
    y = np.arange(len(order))[::-1]
    for (algorithm, offset) in (('ridge', 0.13), ('elasticnet', -0.13)):
        values = subset[subset['algorithm'] == algorithm].set_index('block').loc[order]
        mean = values['post_mae_mean'].to_numpy(dtype=float)
        lower = values['post_mae_p025'].to_numpy(dtype=float)
        upper = values['post_mae_p975'].to_numpy(dtype=float)
        ax.errorbar(mean, y + offset, xerr=np.vstack([mean - lower, upper - mean]), fmt=ALGORITHM_MARKERS[algorithm], markersize=4.2, color=ALGORITHM_COLORS[algorithm], ecolor=ALGORITHM_COLORS[algorithm], elinewidth=1.0, capsize=2.0, capthick=0.8, markeredgecolor='white', markeredgewidth=0.45, label=ALGORITHM_LABELS[algorithm], zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([BLOCK_LABELS[block] for block in order])
    ax.set_xlabel(f'Post-treatment {outcome} MAE')
    ax.set_title('Repeated nested-CV performance', loc='left', fontweight='bold', pad=4)
    ax.legend(loc='lower right', ncol=2, columnspacing=0.9, handletextpad=0.35)
    style_axis(ax)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    return order

def draw_scatter(ax: plt.Axes, performance: pd.DataFrame, aggregated: pd.DataFrame, outcome: str) -> dict:
    selected = best_row(performance, outcome)
    algorithm = selected['algorithm']
    block = selected['block']
    frame = aggregated[(aggregated['algorithm'] == algorithm) & (aggregated['outcome'] == outcome) & (aggregated['block'] == block)].copy()
    observed = frame['observed_post'].to_numpy(dtype=float)
    predicted = frame['predicted_post'].to_numpy(dtype=float)
    mae = mean_absolute_error(observed, predicted)
    r2 = r2_score(observed, predicted)
    limits = [min(observed.min(), predicted.min()), max(observed.max(), predicted.max())]
    margin = max(0.5, 0.07 * (limits[1] - limits[0]))
    limits = [limits[0] - margin, limits[1] + margin]
    ax.plot(limits, limits, linestyle='--', color='#8A8A8A', linewidth=0.8, zorder=1)
    for group in ('Control', 'TMS'):
        group_frame = frame[frame['group'] == group]
        ax.scatter(group_frame['observed_post'], group_frame['predicted_post'], s=22, color=GROUP_COLORS[group], edgecolor='white', linewidth=0.5, alpha=0.9, label=group, zorder=2)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Observed post score')
    ax.set_ylabel('Mean repeated-OOF prediction')
    ax.set_title('Best observed combination', loc='left', fontweight='bold', pad=4)
    ax.text(0.03, 0.97, f'{ALGORITHM_LABELS[algorithm]}\n{BLOCK_LABELS[block]}\nMAE={mae:.2f}; $R^2$={r2:.2f}; n={len(frame)}', transform=ax.transAxes, ha='left', va='top', fontsize=6.2, linespacing=1.25)
    ax.legend(loc='lower right', handletextpad=0.3, borderpad=0.2)
    style_axis(ax)
    return {'algorithm': algorithm, 'block': block, 'ensemble_mae': float(mae), 'ensemble_r2': float(r2), 'n': int(len(frame))}

def make_task_figure(performance: pd.DataFrame, aggregated: pd.DataFrame, outcome: str) -> tuple[list[str], dict]:
    source_performance = performance[performance['outcome'] == outcome].copy()
    source_performance['block_label'] = source_performance['block'].map(BLOCK_LABELS)
    source_performance.to_csv(FIGURE_ROOT / f'source_{outcome.lower()}_performance.csv', index=False, encoding='utf-8-sig')
    selected = best_row(performance, outcome)
    source_scatter = aggregated[(aggregated['outcome'] == outcome) & (aggregated['algorithm'] == selected['algorithm']) & (aggregated['block'] == selected['block'])].copy()
    source_scatter.to_csv(FIGURE_ROOT / f'source_{outcome.lower()}_best_oof.csv', index=False, encoding='utf-8-sig')
    fig = plt.figure(figsize=mm_to_inches(183, 80), constrained_layout=True)
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.075], hspace=0.06)
    grid = outer[0].subgridspec(1, 2, width_ratios=[1.45, 0.82], wspace=0.12)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_note = fig.add_subplot(outer[1])
    draw_performance_forest(ax_a, performance, outcome)
    scatter_metrics = draw_scatter(ax_b, performance, aggregated, outcome)
    panel_label(ax_a, 'a', x=-0.13)
    panel_label(ax_b, 'b', x=-0.16)
    fig.suptitle(f'{outcome}: post-treatment prediction', fontsize=9, fontweight='bold')
    ax_note.set_axis_off()
    ax_note.text(0.5, 0.35, 'Whiskers: 2.5–97.5 percentile across 50 repeated outer splits (not an independent-sample CI). Best model selected post hoc.', ha='center', va='center', fontsize=5.7, color='#4D4D4D', transform=ax_note.transAxes)
    outputs = save_bundle(fig, FIGURE_ROOT / f'figure_task_{outcome.lower()}')
    return (outputs, scatter_metrics)

def make_followup(performance: pd.DataFrame, aggregated: pd.DataFrame) -> tuple[list[str], list[dict]]:
    followup_figure_root = FIGURE_ROOT / 'followup'
    followup_figure_root.mkdir(parents=True, exist_ok=True)
    source = performance.copy()
    source['block_label'] = source['block'].map(BLOCK_LABELS)
    source.to_csv(followup_figure_root / 'source_one_month_performance.csv', index=False, encoding='utf-8-sig')
    fig = plt.figure(figsize=mm_to_inches(183, 125), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, width_ratios=[1.45, 0.82], height_ratios=[1.0, 1.0, 0.07], hspace=0.18, wspace=0.12)
    scatter_results = []
    labels = (('a', 'b'), ('c', 'd'))
    for (row, outcome) in enumerate(('MMSE', 'VFT')):
        ax_perf = fig.add_subplot(grid[row, 0])
        ax_scatter = fig.add_subplot(grid[row, 1])
        draw_performance_forest(ax_perf, performance, outcome)
        scatter_results.append(draw_scatter(ax_scatter, performance, aggregated, outcome))
        ax_perf.set_title(f'{outcome}: repeated nested-CV MAE', loc='left', fontweight='bold', pad=4)
        ax_scatter.set_title(f'{outcome}: best model', loc='left', fontweight='bold', pad=4)
        panel_label(ax_perf, labels[row][0], x=-0.13)
        panel_label(ax_scatter, labels[row][1], x=-0.16)
        selected = best_row(performance, outcome)
        selected_scatter = aggregated[(aggregated['outcome'] == outcome) & (aggregated['algorithm'] == selected['algorithm']) & (aggregated['block'] == selected['block'])]
        selected_scatter.to_csv(followup_figure_root / f'source_one_month_{outcome.lower()}_best_oof.csv', index=False, encoding='utf-8-sig')
    ax_note = fig.add_subplot(grid[2, :])
    fig.suptitle('Exploratory one-month prediction (n=23)', fontsize=9, fontweight='bold')
    ax_note.set_axis_off()
    ax_note.text(0.5, 0.35, 'Whiskers: 2.5–97.5 percentile across 50 repeated outer splits; small-sample sensitivity analysis; best model selected post hoc.', ha='center', va='center', fontsize=5.7, color='#4D4D4D', transform=ax_note.transAxes)
    outputs = save_bundle(fig, followup_figure_root / 'figure_one_month_followup')
    return (outputs, scatter_results)

def make_contact_sheet(png_paths: list[Path]) -> str:
    columns = 2
    rows = int(np.ceil(len(png_paths) / columns))
    (fig, axes) = plt.subplots(rows, columns, figsize=(14.4, 4.8 * rows), squeeze=False)
    for (ax, path) in zip(axes.ravel(), png_paths):
        ax.imshow(plt.imread(path))
        ax.set_title(path.stem, fontsize=10, loc='left')
        ax.set_axis_off()
    for ax in axes.ravel()[len(png_paths):]:
        ax.set_axis_off()
    fig.tight_layout(pad=1.0)
    output = FIGURE_ROOT / 'qa_contact_sheet.png'
    fig.savefig(output, dpi=200, facecolor='white')
    plt.close(fig)
    return str(output)

def qa_exports(paths: list[str]) -> dict:
    records = []
    for path_text in paths:
        path = Path(path_text)
        record = {'path': str(path), 'exists': path.exists(), 'bytes': path.stat().st_size if path.exists() else 0}
        if path.suffix.lower() == '.svg' and path.exists():
            svg_text = path.read_text(encoding='utf-8')
            record['editable_text_nodes'] = '<text' in svg_text
        records.append(record)
    return {'all_files_exist': all((record['exists'] for record in records)), 'all_files_nonempty': all((record['bytes'] > 0 for record in records)), 'all_svg_have_text_nodes': all((record.get('editable_text_nodes', True) for record in records)), 'files': records}

def main() -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    performance = pd.read_csv(PRIMARY_ROOT / 'repeat_performance_summary.csv')
    aggregated = pd.read_csv(PRIMARY_ROOT / 'participant_mean_oof_predictions.csv')
    followup_performance = pd.read_csv(FOLLOWUP_ROOT / 'repeat_performance_summary.csv')
    followup_aggregated = pd.read_csv(FOLLOWUP_ROOT / 'participant_mean_oof_predictions.csv')
    all_outputs = []
    (overview_outputs, _) = make_overview(performance)
    all_outputs.extend(overview_outputs)
    best_models = {}
    for outcome in OUTCOMES:
        (outputs, scatter_metrics) = make_task_figure(performance, aggregated, outcome)
        all_outputs.extend(outputs)
        best_models[outcome] = scatter_metrics
    (followup_outputs, followup_best) = make_followup(followup_performance, followup_aggregated)
    all_outputs.extend(followup_outputs)
    png_paths = [FIGURE_ROOT / 'figure_overview_five_outcomes.png', *[FIGURE_ROOT / f'figure_task_{outcome.lower()}.png' for outcome in OUTCOMES], FIGURE_ROOT / 'followup' / 'figure_one_month_followup.png']
    contact_sheet = make_contact_sheet(png_paths)
    qa = qa_exports(all_outputs)
    qa['contact_sheet'] = contact_sheet
    qa['backend'] = 'Python/matplotlib only'
    qa['primary_n'] = 37
    qa['followup_n'] = 23
    qa['outer_cv'] = 'treatment-stratified 5-fold repeated 50 times'
    qa['interval_definition'] = '2.5th to 97.5th percentile across repeated split-level metrics'
    qa['best_models_from_mean_repeat_mae'] = best_models
    qa['followup_best_models'] = followup_best
    (FIGURE_ROOT / 'figure_qa_manifest.json').write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'outputs': len(all_outputs), 'qa': {k: qa[k] for k in ('all_files_exist', 'all_files_nonempty', 'all_svg_have_text_nodes')}, 'contact_sheet': contact_sheet}, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
