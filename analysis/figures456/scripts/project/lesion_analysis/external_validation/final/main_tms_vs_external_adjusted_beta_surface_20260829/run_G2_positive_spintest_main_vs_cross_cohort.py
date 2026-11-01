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
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
HERE = Path(__file__).resolve().parent
OUTPUT = HERE / 'output'
SOURCE = HERE / 'source_data'
FORAI = HERE.parent / 'forai'
INPUT = SOURCE / 'main_internal_and_cross_cohort_G2_beta_maps.csv'
COORD_FILE = Path(_portable_path('work', 'download/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_1mm.Centroid_RAS.csv'))
NETWORK_ORDER = ['Visual', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Control', 'Default']
NETWORK_ABBR = ['VIS', 'SMN', 'DAN', 'VAN', 'LIM', 'CON', 'DMN']
WITHIN_COLOR = '#6D5A8D'
CROSS_COLOR = '#4B9272'
INK = '#252525'
MUTED = '#6A6A6A'
N_PERMUTATIONS = 10000
SEED = 20260822

def apply_style() -> None:
    mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], 'font.size': 6.5, 'axes.linewidth': 0.65, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'axes.unicode_minus': False, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})

def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0, 1)
    return result

def load_inputs() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    source = pd.read_csv(INPUT).sort_values('roi').reset_index(drop=True)
    coords_frame = pd.read_csv(COORD_FILE).sort_values('ROI Label').reset_index(drop=True)
    if len(source) != 400 or len(coords_frame) != 400:
        raise ValueError('Expected exactly 400 cortical ROIs.')
    if not np.array_equal(source['roi'].to_numpy(int), coords_frame['ROI Label'].to_numpy(int)):
        raise ValueError('ROI ordering differs between the beta map and coordinate table.')
    coords = coords_frame[['R', 'A', 'S']].to_numpy(float)
    coords /= np.linalg.norm(coords, axis=1, keepdims=True)
    network_labels = pd.Categorical(source['network'], categories=NETWORK_ORDER, ordered=True).codes.astype(np.int8)
    if np.any(network_labels < 0):
        raise ValueError('Unknown network label in ROI source table.')
    return (source, coords, network_labels)

def build_null_labels(coords: np.ndarray) -> np.ndarray:
    """Match the supplied reference spin test using exact nearest centroids."""
    rng = np.random.default_rng(SEED)
    tree = cKDTree(coords)
    null_indices = np.empty((N_PERMUTATIONS, len(coords)), dtype=np.int16)
    for permutation in range(N_PERMUTATIONS):
        rotated = Rotation.random(random_state=rng).apply(coords)
        null_indices[permutation] = tree.query(rotated, k=1)[1]
        if (permutation + 1) % 1000 == 0:
            print(f'network spins {permutation + 1:05d}/{N_PERMUTATIONS}', flush=True)
    return null_indices

def test_positive_set(selected: np.ndarray, null_indices: np.ndarray, network_labels: np.ndarray, contrast: str) -> tuple[pd.DataFrame, np.ndarray]:
    selected = np.asarray(selected, dtype=bool)
    n_selected = int(selected.sum())
    if n_selected == 0:
        rows = [{'contrast': contrast, 'direction': 'positive', 'network': network, 'network_abbreviation': NETWORK_ABBR[i], 'n_selected_rois': 0, 'actual_count': 0, 'actual_enrichment_ratio': np.nan, 'null_mean': np.nan, 'null_sd': np.nan, 'spin_p_upper': np.nan, 'spin_fdr_q': np.nan, 'n_permutations': 0, 'seed': SEED, 'status': 'not_testable_no_FDR_significant_ROIs'} for (i, network) in enumerate(NETWORK_ORDER)]
        return (pd.DataFrame(rows), np.full((N_PERMUTATIONS, len(NETWORK_ORDER)), np.nan))
    actual = np.array([np.mean(network_labels[selected] == index) for index in range(len(NETWORK_ORDER))], dtype=float)
    selected_null_labels = network_labels[null_indices[:, selected]]
    null = np.stack([np.mean(selected_null_labels == index, axis=1) for index in range(len(NETWORK_ORDER))], axis=1)
    p_values = (1.0 + np.sum(null >= actual[None, :], axis=0)) / (N_PERMUTATIONS + 1.0)
    q_values = bh_adjust(p_values)
    rows: list[dict[str, object]] = []
    for (index, network) in enumerate(NETWORK_ORDER):
        rows.append({'contrast': contrast, 'direction': 'positive', 'network': network, 'network_abbreviation': NETWORK_ABBR[index], 'n_selected_rois': n_selected, 'actual_count': int(np.sum(network_labels[selected] == index)), 'actual_enrichment_ratio': float(actual[index]), 'null_mean': float(null[:, index].mean()), 'null_sd': float(null[:, index].std(ddof=1)), 'spin_p_upper': float(p_values[index]), 'spin_fdr_q': float(q_values[index]), 'n_permutations': N_PERMUTATIONS, 'seed': SEED})
    return (pd.DataFrame(rows), null)

def draw_paired_spin_panel(axis: plt.Axes, within_stats: pd.DataFrame, within_null: np.ndarray, cross_stats: pd.DataFrame, cross_null: np.ndarray, q_note_y: float=-0.39, method_note_y: float=-0.54) -> None:
    centers = np.arange(len(NETWORK_ORDER), dtype=float)
    offset = 0.17
    axis.axvspan(-0.48, 0.48, color='#F2F0E8', alpha=0.72, lw=0, zorder=0)
    specifications = [(within_stats, within_null, WITHIN_COLOR, centers - offset), (cross_stats, cross_null, CROSS_COLOR, centers + offset)]
    for (stats, null, color, positions) in specifications:
        if int(stats['n_selected_rois'].iloc[0]) == 0:
            axis.text(0.98, 0.98, 'Cross-cohort: 0 FDR-significant ROIs; enrichment not defined', transform=axis.transAxes, ha='right', va='top', fontsize=5.2, color=color)
            continue
        violins = axis.violinplot([null[:, index] for index in range(len(NETWORK_ORDER))], positions=positions, widths=0.3, showmeans=False, showmedians=False, showextrema=False, bw_method=0.18)
        for body in violins['bodies']:
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.2)
            body.set_linewidth(0.65)
        observed = stats['actual_enrichment_ratio'].to_numpy(float)
        for (index, (position, value)) in enumerate(zip(positions, observed)):
            axis.plot([position - 0.13, position + 0.13], [value, value], color=color, lw=2.1, solid_capstyle='round', zorder=5)
            q_value = float(stats.iloc[index]['spin_fdr_q'])
            if q_value < 0.05:
                axis.text(position, min(value + 0.055, 1.075), '*', ha='center', va='bottom', fontsize=8.0, fontweight='bold', color=color, zorder=7)
    axis.set_xticks(centers)
    axis.set_xticklabels(NETWORK_ABBR)
    axis.set_xlim(-0.55, len(NETWORK_ORDER) - 0.45)
    axis.set_ylim(0, 1.12)
    axis.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axis.set_xlabel('Functional network')
    axis.set_ylabel('Network enrichment ratio')
    axis.set_title('Spin-test enrichment of positive G2 effects', loc='left', fontsize=7.1, fontweight='bold', pad=13)
    axis.grid(False)
    axis.tick_params(axis='both', pad=1.8, labelsize=5.5)
    handles = [Line2D([0], [0], color=WITHIN_COLOR, lw=5, alpha=0.55, label=f'Within-study ({int(within_stats.n_selected_rois.iloc[0])} ROIs)'), Line2D([0], [0], color=CROSS_COLOR, lw=5, alpha=0.55, label=f'Cross-cohort ({int(cross_stats.n_selected_rois.iloc[0])} ROIs)')]
    axis.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 1.01), frameon=False, ncol=2, fontsize=5.0, handlelength=1.35, handletextpad=0.45, columnspacing=0.9)
    within_vis_q = float(within_stats.loc[within_stats['network_abbreviation'].eq('VIS'), 'spin_fdr_q'].iloc[0])
    cross_vis_q = float(cross_stats.loc[cross_stats['network_abbreviation'].eq('VIS'), 'spin_fdr_q'].iloc[0])
    axis.text(0.42, q_note_y, f'within q={within_vis_q:.3f}', transform=axis.transAxes, ha='right', va='top', fontsize=4.8, color=WITHIN_COLOR, fontweight='bold', clip_on=False)
    axis.text(0.44, q_note_y, f'cross q={cross_vis_q:.3f}' if np.isfinite(cross_vis_q) else 'cross: not testable (0 ROIs)', transform=axis.transAxes, ha='left', va='top', fontsize=4.8, color=CROSS_COLOR, fontweight='bold', clip_on=False)
    axis.text(1.0, method_note_y, '10,000 rotations; violins, spatial null; bars, observed; q values are spin-FDR', transform=axis.transAxes, ha='right', va='top', fontsize=4.2, color=MUTED, clip_on=False)

def prepare_spin_results() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    (source, coords, network_labels) = load_inputs()
    null_indices = build_null_labels(coords)
    within_selected = (source['fdr_q_G2'].to_numpy(float) < 0.05) & (source['adjusted_TMS_beta_G2'].to_numpy(float) > 0)
    cross_selected = (source['cross_cohort_fdr_q_400'].to_numpy(float) < 0.05) & (source['main_tms_minus_external_adjusted_beta'].to_numpy(float) > 0)
    (within_stats, within_null) = test_positive_set(within_selected, null_indices, network_labels, 'Within-study')
    (cross_stats, cross_null) = test_positive_set(cross_selected, null_indices, network_labels, 'Cross-cohort')
    statistics = pd.concat([within_stats, cross_stats], ignore_index=True)
    statistics.to_csv(SOURCE / 'G2_positive_spintest_main_vs_cross_cohort_statistics.csv', index=False, encoding='utf-8-sig')
    np.savez_compressed(SOURCE / 'G2_positive_spintest_main_vs_cross_cohort_nulls.npz', within=within_null, cross=cross_null, network_order=np.asarray(NETWORK_ORDER), n_permutations=N_PERMUTATIONS, seed=SEED)
    return (statistics, within_null, cross_null)

def export_standalone(statistics: pd.DataFrame, within_null: np.ndarray, cross_null: np.ndarray) -> list[Path]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FORAI.mkdir(parents=True, exist_ok=True)
    within_stats = statistics.loc[statistics['contrast'].eq('Within-study')].reset_index(drop=True)
    cross_stats = statistics.loc[statistics['contrast'].eq('Cross-cohort')].reset_index(drop=True)
    (fig, axis) = plt.subplots(figsize=(105 / 25.4, 75 / 25.4))
    draw_paired_spin_panel(axis, within_stats, within_null, cross_stats, cross_null)
    fig.subplots_adjust(left=0.15, right=0.985, bottom=0.31, top=0.78)
    stem = OUTPUT / 'G2_positive_spintest_main_vs_cross_cohort'
    outputs = [stem.with_suffix(ext) for ext in ('.pdf', '.svg', '.png', '.tiff')]
    fig.savefig(outputs[0], dpi=600)
    fig.savefig(outputs[1], dpi=600)
    fig.savefig(outputs[2], dpi=400)
    fig.savefig(outputs[3], dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)
    for path in outputs:
        (FORAI / path.name).write_bytes(path.read_bytes())
    return outputs

def main() -> None:
    apply_style()
    (statistics, within_null, cross_null) = prepare_spin_results()
    outputs = export_standalone(statistics, within_null, cross_null)
    print(statistics.to_string(index=False))
    for output in outputs:
        print(output)
if __name__ == '__main__':
    main()
