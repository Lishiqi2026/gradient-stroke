from __future__ import annotations
'Recompute and audit the healthy Schaefer-400 gradient reference.\n\nThis script performs two calculations from the same packaged healthy group FC:\n\n1. An exact rerun of the pipeline used by ``run_lesion_gradient_distance.py``.\n2. A parcel-level adaptation of the core steps in\n   ``E:/paper/lesion_refresh_20261028/work/Gradient/code/gradient_calculation/allnpy.py``.\n\nThe legacy source code is vertex-wise and expects an external vertex-wise HCP\nembedding that is not present in the project.  Its result below is therefore\nexplicitly labelled "adapted" and is aligned in five dimensions to the fixed\nSchaefer-400 healthy reference solely to make G1/G2 comparable.\n'
import os as _portable_os
from pathlib import Path as _PortablePath

def _portable_path(kind, relative):
    project = _PortablePath(_portable_os.environ['PAPER_PROJECT_ROOT'])
    root = project if kind == 'project' else project.parent
    return str(root / relative)
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from brainspace.datasets import load_group_fc, load_parcellation
from brainspace.gradient import GradientMaps
from brainspace.gradient.kernels import compute_affinity
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
from neuromaps.datasets import fetch_fslr
from surfplot import Plot
ROOT = Path(_portable_path('project', 'lesion_analysis'))
ANALYSIS = ROOT / 'lesion_gradient_distance'
TABLE_DIR = ANALYSIS / 'tables'
REFERENCE_DIR = ANALYSIS / 'references'
OUT_DIR = ANALYSIS / 'healthy_reference_gradient_calculation_audit'
FIGURE_DIR = OUT_DIR / 'figures'
sys.path.insert(0, str(ROOT))
import run_lesion_gradient_distance as current
mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'], 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'font.size': 7})
G1_CMAP = LinearSegmentedColormap.from_list('g1_reference', ['#355F8A', '#F7F7F3', '#B5483A'])
G2_CMAP = LinearSegmentedColormap.from_list('g2_reference', ['#B58B2A', '#F7F7F3', '#5C4A8A'])

def zscore_columns(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mean = values.mean(axis=0, keepdims=True)
    sd = values.std(axis=0, ddof=1, keepdims=True)
    if np.any(~np.isfinite(sd)) or np.any(sd <= 0):
        raise RuntimeError('Cannot z-score constant/non-finite gradient component')
    return (values - mean) / sd

def legacy_style_adapted_embedding(fc_r: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Adapt the numerical core of gradient_calculation/allnpy.py to 400 parcels."""
    matrix = np.asarray(fc_r, dtype=float).copy()
    matrix = (matrix + matrix.T) / 2.0
    with np.errstate(divide='ignore', invalid='ignore'):
        fisher_z = np.arctanh(matrix)
    thresholds = np.percentile(fisher_z, 90, axis=0)
    sparse_profiles = np.where(fisher_z > thresholds[:, None], fisher_z, 0.0)
    sparse_profiles[sparse_profiles < 0] = 0.0
    np.fill_diagonal(sparse_profiles, 0.0)
    sparse_profiles = np.nan_to_num(sparse_profiles, nan=0.0, posinf=0.0, neginf=0.0)
    affinity = compute_affinity(sparse_profiles, kernel='cosine', sparsity=0.9, pre_sparsify=True, non_negative=True)
    model = GradientMaps(n_components=current.N_COMPONENTS, approach='dm', kernel='cosine', random_state=0)
    model.fit(affinity)
    return (np.asarray(model.gradients_, dtype=float), np.asarray(model.lambdas_, dtype=float), affinity)

def correlation_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.corrcoef(left.T, right.T)[:left.shape[1], left.shape[1]:]

def render_surface(values: np.ndarray, title: str, label: str, cmap, stem: Path) -> None:
    surfaces = fetch_fslr()
    (lh_surf, rh_surf) = surfaces['veryinflated']
    atlas = load_parcellation('schaefer', 400, join=True)
    vertex_values = np.full_like(atlas, np.nan, dtype=float)
    for roi in range(1, 401):
        vertex_values[atlas == roi] = values[roi - 1]
    medial_wall = np.full_like(atlas, np.nan, dtype=float)
    medial_wall[atlas == 0] = 1.0
    limit = float(np.nanmax(np.abs(values)))
    plot = Plot(lh_surf, rh_surf, views=['lateral', 'medial'], zoom=1.68, brightness=0.65, size=(3600, 2600), background=(1, 1, 1))
    plot.add_layer(medial_wall, cmap=ListedColormap(['#D1D1D1']), color_range=(0, 1), cbar=False)
    plot.add_layer(vertex_values, cmap=cmap, color_range=(-limit, limit), cbar=False)
    plot.add_layer(atlas, cmap=ListedColormap(['#606060']), as_outline=True, cbar=False)
    figure = plot.build()
    figure.patch.set_facecolor('white')
    figure.set_size_inches(4.1, 2.7, forward=True)
    brain_axes = figure.axes[:4]
    if len(brain_axes) >= 4:
        positions = [[0.035, 0.525, 0.445, 0.355], [0.52, 0.525, 0.445, 0.355], [0.035, 0.165, 0.445, 0.355], [0.52, 0.165, 0.445, 0.355]]
        for (axis, position) in zip(brain_axes, positions):
            axis.set_position(position)
    figure.suptitle(title, x=0.5, y=0.965, fontsize=8.3, fontweight='bold', fontname='Arial')
    color_axis = figure.add_axes([0.275, 0.105, 0.45, 0.028])
    colorbar = figure.colorbar(mpl.cm.ScalarMappable(norm=Normalize(-limit, limit), cmap=cmap), cax=color_axis, orientation='horizontal')
    colorbar.set_label(label, fontsize=6.5, labelpad=1.5)
    colorbar.ax.tick_params(labelsize=5.8, length=1.8, width=0.5, pad=1.0)
    colorbar.outline.set_linewidth(0.5)
    for (suffix, options) in {'.pdf': {}, '.svg': {}, '.png': {'dpi': 600}}.items():
        figure.savefig(stem.with_suffix(suffix), facecolor='white', bbox_inches='tight', pad_inches=0.02, **options)
    plt.close(figure)

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(TABLE_DIR / 'healthy_normative_reference_schaefer400.csv')
    saved = np.load(REFERENCE_DIR / 'healthy_normative_reference_400x5.npy')
    healthy_fc = load_group_fc('schaefer', scale=400, group='main')
    (exact_raw, exact_lambdas) = current.fit_gradient(healthy_fc, fisher_z_input=False)
    (exact_oriented, direction_rules) = current.orient_master(exact_raw, labels)
    exact_z = zscore_columns(exact_oriented)
    (legacy_raw, legacy_lambdas, legacy_affinity) = legacy_style_adapted_embedding(healthy_fc)
    (legacy_direct_oriented, legacy_direction_rules) = current.orient_master(legacy_raw, labels)
    (legacy_aligned, legacy_fit) = current.similarity_align(legacy_raw, exact_oriented)
    legacy_aligned_z = (legacy_aligned - exact_oriented.mean(axis=0)) / exact_oriented.std(axis=0, ddof=1)
    exact_difference = exact_oriented - saved
    raw_corr = correlation_matrix(legacy_raw, exact_oriented)
    aligned_corr = correlation_matrix(legacy_aligned_z, exact_z)
    source = labels[['roi', 'roi_name', 'network']].copy()
    for component in range(current.N_COMPONENTS):
        number = component + 1
        source[f'saved_G{number}_raw'] = saved[:, component]
        source[f'exact_rerun_G{number}_raw'] = exact_oriented[:, component]
        source[f'exact_rerun_G{number}_z'] = exact_z[:, component]
        source[f'legacy_adapted_direct_G{number}_raw'] = legacy_direct_oriented[:, component]
        source[f'legacy_adapted_aligned_G{number}_z'] = legacy_aligned_z[:, component]
    source.to_csv(OUT_DIR / 'healthy_reference_G1_G2_source_data.csv', index=False)
    rows: list[dict] = []
    for component in range(current.N_COMPONENTS):
        rows.append({'pipeline': 'exact_current_pipeline_rerun', 'component': f'G{component + 1}', 'eigenvalue': float(exact_lambdas[component]), 'correlation_with_saved_reference': float(np.corrcoef(exact_oriented[:, component], saved[:, component])[0, 1]), 'rmse_vs_saved_reference': float(np.sqrt(np.mean(exact_difference[:, component] ** 2))), 'max_abs_difference_vs_saved_reference': float(np.max(np.abs(exact_difference[:, component])))})
        rows.append({'pipeline': 'legacy_gradient_calculation_adapted_and_5D_aligned', 'component': f'G{component + 1}', 'eigenvalue': float(legacy_lambdas[component]), 'correlation_with_saved_reference': float(aligned_corr[component, component]), 'rmse_vs_saved_reference': float(np.sqrt(np.mean((legacy_aligned_z[:, component] - exact_z[:, component]) ** 2))), 'max_abs_difference_vs_saved_reference': float(np.max(np.abs(legacy_aligned_z[:, component] - exact_z[:, component])))})
    pd.DataFrame(rows).to_csv(OUT_DIR / 'healthy_reference_recalculation_qc.csv', index=False)
    pd.DataFrame(raw_corr, index=[f'legacy_raw_G{i + 1}' for i in range(current.N_COMPONENTS)], columns=[f'current_G{i + 1}' for i in range(current.N_COMPONENTS)]).to_csv(OUT_DIR / 'legacy_raw_vs_current_component_correlations.csv')
    pd.DataFrame(aligned_corr, index=[f'legacy_aligned_G{i + 1}' for i in range(current.N_COMPONENTS)], columns=[f'current_G{i + 1}' for i in range(current.N_COMPONENTS)]).to_csv(OUT_DIR / 'legacy_aligned_vs_current_component_correlations.csv')
    np.save(OUT_DIR / 'healthy_reference_exact_rerun_400x5.npy', exact_oriented)
    np.save(OUT_DIR / 'healthy_reference_legacy_adapted_raw_400x5.npy', legacy_raw)
    np.save(OUT_DIR / 'healthy_reference_legacy_adapted_aligned_z_400x5.npy', legacy_aligned_z)
    np.save(OUT_DIR / 'healthy_reference_legacy_adapted_affinity_400x400.npy', legacy_affinity)
    audit = {'healthy_fc': 'BrainSpace packaged healthy Schaefer400 group FC, main/discovery group', 'healthy_fc_shape': list(healthy_fc.shape), 'exact_rerun_matches_saved_max_abs_difference': float(np.max(np.abs(exact_difference))), 'current_direction_rules': direction_rules, 'legacy_adaptation': {'source_logic': 'gradient_calculation/allnpy.py core adapted from vertex level to Schaefer400', 'not_exact_reason': 'external vertex-wise HCP embedding and original vertex FC inputs are absent', 'alignment_to_current': legacy_fit, 'direct_orientation_rules': legacy_direction_rules}}
    (OUT_DIR / 'audit_manifest.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    render_surface(exact_z[:, 0], 'Healthy normative G1 (exact reference rerun)', 'Reference G1 (z)', G1_CMAP, FIGURE_DIR / 'healthy_normative_G1_exact_rerun_surface')
    render_surface(exact_z[:, 1], 'Healthy normative G2 (exact reference rerun)', 'Reference G2 (z)', G2_CMAP, FIGURE_DIR / 'healthy_normative_G2_exact_rerun_surface')
    render_surface(legacy_aligned_z[:, 0], 'Healthy G1 (legacy pipeline adapted; 5D aligned)', 'Adapted G1 (z)', G1_CMAP, FIGURE_DIR / 'healthy_G1_legacy_adapted_aligned_surface')
    render_surface(legacy_aligned_z[:, 1], 'Healthy G2 (legacy pipeline adapted; 5D aligned)', 'Adapted G2 (z)', G2_CMAP, FIGURE_DIR / 'healthy_G2_legacy_adapted_aligned_surface')
    print(f'Saved audit to: {OUT_DIR}')
    print(f'Exact rerun max abs difference: {np.max(np.abs(exact_difference)):.12g}')
    print('Legacy adapted aligned correlations: ' + ', '.join((f'G{i + 1}={aligned_corr[i, i]:.4f}' for i in range(current.N_COMPONENTS))))
if __name__ == '__main__':
    main()
