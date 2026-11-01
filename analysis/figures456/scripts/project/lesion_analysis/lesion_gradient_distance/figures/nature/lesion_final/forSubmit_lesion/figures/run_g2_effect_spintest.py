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
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
SCRIPT_DIR = Path(__file__).resolve().parent.parent / 'scripts'
sys.path.insert(0, str(SCRIPT_DIR))
from figure_common import INK, MID, SOURCE, save_submission_figure
COORD_FILE = Path(_portable_path('work', 'download/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_1mm.Centroid_RAS.csv'))
NETWORK_ORDER = ['Visual', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Control', 'Default']
NETWORK_ABBR = ['VIS', 'SMN', 'DAN', 'VAN', 'LIM', 'CON', 'DMN']
NEGATIVE = '#2166AC'
POSITIVE = '#B2182B'
N_PERMUTATIONS = 10000
SEED = 20260822

def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0, 1)
    return result

def load_inputs() -> tuple[pd.DataFrame, np.ndarray]:
    source = pd.read_csv(SOURCE / 'G1_G2_surface_source_data.csv').sort_values('roi')
    coords_frame = pd.read_csv(COORD_FILE).sort_values('ROI Label')
    if len(source) != 400 or len(coords_frame) != 400:
        raise ValueError('Expected exactly 400 cortical ROIs in both source and coordinate files')
    if not np.array_equal(source['roi'].to_numpy(int), coords_frame['ROI Label'].to_numpy(int)):
        raise ValueError('ROI ordering differs between the effect source and coordinate file')
    coords = coords_frame[['R', 'A', 'S']].to_numpy(float)
    norms = np.linalg.norm(coords, axis=1, keepdims=True)
    coords = coords / norms
    return (source, coords)

def build_null_labels(coords: np.ndarray, network_labels: np.ndarray, n_permutations: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    null_labels = np.empty((n_permutations, len(coords)), dtype=np.int8)
    for permutation in range(n_permutations):
        rotated = Rotation.random(random_state=rng).apply(coords)
        nearest = np.argmin(cdist(rotated, coords, metric='sqeuclidean'), axis=1)
        null_labels[permutation] = network_labels[nearest]
    return null_labels

def test_direction(selected: np.ndarray, null_labels: np.ndarray, network_labels: np.ndarray, direction: str) -> tuple[pd.DataFrame, np.ndarray]:
    n_selected = int(selected.sum())
    if n_selected == 0:
        raise ValueError(f'No {direction} FDR-significant ROIs')
    actual = np.array([np.mean(network_labels[selected] == index) for index in range(len(NETWORK_ORDER))], dtype=float)
    null = np.empty((null_labels.shape[0], len(NETWORK_ORDER)), dtype=float)
    selected_null = null_labels[:, selected]
    for index in range(len(NETWORK_ORDER)):
        null[:, index] = np.mean(selected_null == index, axis=1)
    p_values = (1.0 + np.sum(null >= actual[None, :], axis=0)) / (null.shape[0] + 1.0)
    q_values = bh_adjust(p_values)
    rows = []
    for (index, network) in enumerate(NETWORK_ORDER):
        rows.append({'direction': direction, 'network': network, 'network_abbreviation': NETWORK_ABBR[index], 'n_selected_rois': n_selected, 'actual_count': int(np.sum(network_labels[selected] == index)), 'actual_enrichment_ratio': actual[index], 'null_mean': float(null[:, index].mean()), 'null_sd': float(null[:, index].std(ddof=1)), 'spin_p_upper': float(p_values[index]), 'spin_fdr_q': float(q_values[index]), 'n_permutations': null.shape[0]})
    return (pd.DataFrame(rows), null)

def draw_combined(axis, positive_stats: pd.DataFrame, positive_null: np.ndarray, negative_stats: pd.DataFrame, negative_null: np.ndarray) -> None:
    centers = np.arange(len(NETWORK_ORDER), dtype=float)
    offset = 0.18
    specifications = [(positive_stats, positive_null, POSITIVE, centers - offset), (negative_stats, negative_null, NEGATIVE, centers + offset)]
    for (stats, null, color, positions) in specifications:
        violins = axis.violinplot([null[:, index] for index in range(len(NETWORK_ORDER))], positions=positions, widths=0.32, showmeans=False, showmedians=False, showextrema=False, bw_method=0.18)
        for body in violins['bodies']:
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.25)
            body.set_linewidth(0.75)
        actual = stats['actual_enrichment_ratio'].to_numpy(float)
        for (index, (position, value)) in enumerate(zip(positions, actual)):
            axis.plot([position - 0.14, position + 0.14], [value, value], color=color, lw=2.0, solid_capstyle='round', zorder=5)
            q_value = float(stats.iloc[index]['spin_fdr_q'])
            if q_value < 0.05:
                axis.text(position, value + 0.026, f'*\nq={q_value:.3f}', ha='center', va='bottom', fontsize=5.8, fontweight='bold', color=color)
    axis.set_xticks(centers)
    axis.set_xticklabels(NETWORK_ABBR)
    axis.set_xlim(-0.58, len(NETWORK_ORDER) - 0.42)
    axis.set_ylim(0, 0.86)
    axis.set_xlabel('Functional network')
    axis.set_ylabel('Network enrichment ratio')
    axis.grid(False)
    axis.tick_params(axis='both', pad=2)
    legend = [Line2D([0], [0], color=POSITIVE, lw=5, alpha=0.45, label=f'Positive effect null (n={int(positive_stats.n_selected_rois.iloc[0])} ROIs)'), Line2D([0], [0], color=NEGATIVE, lw=5, alpha=0.45, label=f'Negative effect null (n={int(negative_stats.n_selected_rois.iloc[0])} ROIs)'), Line2D([0], [0], color=INK, lw=2, label='Observed enrichment')]
    axis.legend(handles=legend, loc='lower center', bbox_to_anchor=(0.5, 1.015), frameon=False, ncol=3, fontsize=5.8, handlelength=1.6, columnspacing=1.3)

def draw_direction_panel(axis, stats: pd.DataFrame, null: np.ndarray, color: str, title: str) -> None:
    positions = np.arange(len(NETWORK_ORDER), dtype=float)
    violins = axis.violinplot([null[:, index] for index in range(len(NETWORK_ORDER))], positions=positions, widths=0.7, showmeans=False, showmedians=False, showextrema=False, bw_method=0.18)
    for body in violins['bodies']:
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.24)
        body.set_linewidth(0.75)
    observed = stats['actual_enrichment_ratio'].to_numpy(float)
    for (index, value) in enumerate(observed):
        axis.plot([index - 0.28, index + 0.28], [value, value], color=color, lw=2.2, solid_capstyle='round', zorder=5)
        q_value = float(stats.iloc[index]['spin_fdr_q'])
        if q_value < 0.05:
            axis.text(index, value + 0.026, f'*\nq={q_value:.3f}', ha='center', va='bottom', fontsize=5.8, fontweight='bold', color=color)
    axis.set_xticks(positions)
    axis.set_xticklabels(NETWORK_ABBR)
    axis.set_xlim(-0.55, len(NETWORK_ORDER) - 0.45)
    axis.set_ylim(0, 0.86)
    axis.set_title(title, color=color, fontweight='bold', pad=5)
    axis.set_xlabel('Functional network')
    axis.grid(False)
    axis.tick_params(axis='both', pad=2)

def draw_interval_summary(axis, positive_stats: pd.DataFrame, positive_null: np.ndarray, negative_stats: pd.DataFrame, negative_null: np.ndarray) -> None:
    centers = np.arange(len(NETWORK_ORDER))[::-1].astype(float)
    offset = 0.14
    specifications = [(positive_stats, positive_null, POSITIVE, centers + offset, 'Positive effect'), (negative_stats, negative_null, NEGATIVE, centers - offset, 'Negative effect')]
    for (stats, null, color, positions, label) in specifications:
        lower = np.quantile(null, 0.025, axis=0)
        median = np.quantile(null, 0.5, axis=0)
        upper = np.quantile(null, 0.975, axis=0)
        observed = stats['actual_enrichment_ratio'].to_numpy(float)
        axis.hlines(positions, lower, upper, color=color, lw=2.2, alpha=0.3, zorder=1)
        axis.scatter(median, positions, s=23, facecolor='white', edgecolor=color, linewidth=0.9, zorder=2)
        axis.scatter(observed, positions, s=34, marker='D', facecolor=color, edgecolor='white', linewidth=0.45, zorder=4, label=label)
        for (index, row) in stats.reset_index(drop=True).iterrows():
            q_value = float(row['spin_fdr_q'])
            if q_value < 0.05:
                axis.text(observed[index] + 0.025, positions[index], f'q={q_value:.3f}', ha='left', va='center', fontsize=6.1, fontweight='bold', color=color)
    axis.set_yticks(centers)
    axis.set_yticklabels(NETWORK_ABBR)
    axis.set_xlim(-0.015, 0.86)
    axis.set_ylim(-0.55, len(NETWORK_ORDER) - 0.45)
    axis.set_xlabel('Network enrichment ratio')
    axis.set_ylabel('Functional network')
    axis.grid(axis='x', color='#E3E3E3', lw=0.55, zorder=0)
    axis.tick_params(axis='both', pad=2)
    direction_legend = axis.legend(loc='lower right', frameon=False, fontsize=6.0, ncol=2, handletextpad=0.4, columnspacing=1.2)
    axis.add_artist(direction_legend)
    null_legend = [Line2D([0], [0], color=MID, lw=2.2, alpha=0.5, label='95% spatial-null interval'), Line2D([0], [0], marker='o', linestyle='none', markerfacecolor='white', markeredgecolor=MID, markersize=4.0, label='Null median'), Line2D([0], [0], marker='D', linestyle='none', markerfacecolor=INK, markeredgecolor='white', markersize=4.5, label='Observed')]
    axis.legend(handles=null_legend, loc='lower center', bbox_to_anchor=(0.5, 1.015), frameon=False, fontsize=5.8, ncol=3, handlelength=1.8, columnspacing=1.2)

def main() -> None:
    (source, coords) = load_inputs()
    network_labels = pd.Categorical(source['network'], categories=NETWORK_ORDER, ordered=True).codes.astype(np.int8)
    if np.any(network_labels < 0):
        raise ValueError('At least one ROI has an unknown network label')
    significant = source['G2_FDR_significant'].astype(bool).to_numpy()
    beta = source['adjusted_TMS_beta_G2'].to_numpy(float)
    positive = significant & (beta > 0)
    negative = significant & (beta < 0)
    null_labels = build_null_labels(coords, network_labels, N_PERMUTATIONS, SEED)
    (positive_stats, positive_null) = test_direction(positive, null_labels, network_labels, 'positive')
    (negative_stats, negative_null) = test_direction(negative, null_labels, network_labels, 'negative')
    statistics = pd.concat([positive_stats, negative_stats], ignore_index=True)
    statistics.to_csv(Path(__file__).resolve().parent / 'G2_TMS_effect_spintest_statistics.csv', index=False, encoding='utf-8-sig')
    np.savez_compressed(Path(__file__).resolve().parent / 'G2_TMS_effect_spintest_null_distributions.npz', positive=positive_null, negative=negative_null, network_order=np.asarray(NETWORK_ORDER), n_permutations=N_PERMUTATIONS, seed=SEED)
    (width_mm, height_mm) = (132.0, 82.0)
    (violin_fig, violin_axis) = plt.subplots(figsize=(width_mm / 25.4, height_mm / 25.4))
    draw_combined(violin_axis, positive_stats, positive_null, negative_stats, negative_null)
    violin_fig.text(0.105, 0.965, 'Spin-test network enrichment of FDR-significant G2 effects', ha='left', va='top', fontsize=8.2, fontweight='bold', color=INK)
    violin_fig.text(0.105, 0.905, '10,000 rotations; horizontal bars, observed; violins, spatial null', ha='left', va='top', fontsize=5.7, color=MID)
    violin_fig.subplots_adjust(left=0.105, right=0.985, bottom=0.17, top=0.76)
    save_submission_figure(violin_fig, 'G2_TMS_effect_spintest')
    (summary_width_mm, summary_height_mm) = (125.0, 82.0)
    (summary_fig, summary_axis) = plt.subplots(figsize=(summary_width_mm / 25.4, summary_height_mm / 25.4))
    draw_interval_summary(summary_axis, positive_stats, positive_null, negative_stats, negative_null)
    summary_fig.text(0.13, 0.965, 'Spin-test network enrichment of FDR-significant G2 effects', ha='left', va='top', fontsize=8.2, fontweight='bold', color=INK)
    summary_fig.text(0.13, 0.91, '10,000 rotations; positive and negative effects tested separately', ha='left', va='top', fontsize=5.7, color=MID)
    summary_fig.subplots_adjust(left=0.13, right=0.985, bottom=0.16, top=0.8)
    save_submission_figure(summary_fig, 'G2_TMS_effect_spintest_interval')
    (panel_width_mm, panel_height_mm) = (174.0, 76.0)
    (panel_fig, panel_axes) = plt.subplots(1, 2, figsize=(panel_width_mm / 25.4, panel_height_mm / 25.4), sharey=True)
    draw_direction_panel(panel_axes[0], positive_stats, positive_null, POSITIVE, f'Positive G2 effect (n={int(positive_stats.n_selected_rois.iloc[0])} ROIs)')
    draw_direction_panel(panel_axes[1], negative_stats, negative_null, NEGATIVE, f'Negative G2 effect (n={int(negative_stats.n_selected_rois.iloc[0])} ROIs)')
    panel_axes[0].set_ylabel('Network enrichment ratio')
    panel_axes[1].tick_params(axis='y', which='both', left=False, labelleft=False)
    panel_fig.text(0.08, 0.965, 'Spin-test network enrichment of FDR-significant G2 effects', ha='left', va='top', fontsize=8.2, fontweight='bold', color=INK)
    panel_fig.text(0.985, 0.965, '10,000 rotations; horizontal bars, observed; violins, spatial null', ha='right', va='top', fontsize=5.7, color=MID)
    panel_fig.subplots_adjust(left=0.08, right=0.985, bottom=0.18, top=0.78, wspace=0.24)
    save_submission_figure(panel_fig, 'G2_TMS_effect_spintest_two_panel')
if __name__ == '__main__':
    main()
