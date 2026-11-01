from __future__ import annotations
import os as _portable_os
from pathlib import Path as _PortablePath

def _portable_path(kind, relative):
    project = _PortablePath(_portable_os.environ['PAPER_PROJECT_ROOT'])
    root = project if kind == 'project' else project.parent
    return str(root / relative)
import json
import math
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from brainspace.datasets import load_parcellation
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
from neuromaps.datasets import fetch_fslr
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from surfplot import Plot
HERE = Path(__file__).resolve().parent
FINAL = HERE.parent
OUTPUT = HERE / 'output'
SOURCE = HERE / 'source_data'
QA = HERE / 'qa'
FORAI = FINAL / 'forai'
ANALYSIS = Path(_portable_path('project', 'lesion_analysis'))
CONFIRMATORY = ANALYSIS / 'external_validation' / 'ds003999_confirmatory_extension'
MAIN_PACKAGE = ANALYSIS / 'lesion_gradient_distance' / 'figures' / 'nature' / 'lesion_final' / 'forSubmit_lesion'
MAIN_SOURCE = MAIN_PACKAGE / 'source_data' / 'G1_G2_surface_source_data.csv'
MAIN_UNIFIED = CONFIRMATORY / 'tables' / 'main_unified_subject_fixed_signature.csv'
EXTERNAL_ROI = CONFIRMATORY / 'tables' / 'external_primary_roi_normalization_long.csv'
REFERENCE_TABLE = CONFIRMATORY / 'references' / 'healthy_main_reference_schaefer400.csv'
REFERENCE_ARRAY = CONFIRMATORY / 'references' / 'healthy_main_reference_locked_400x5.npy'
MAIN_FC = ANALYSIS / 'lesion_gradient_distance' / 'fc_consistent' / 'fc_matrices_z'
sys.path.insert(0, str(CONFIRMATORY))
import run_confirmatory_external_validation as confirm
N_PERM = 5000
SEED = 20260829
INK = '#252525'
MUTED = '#666666'
EFFECT_CMAP = LinearSegmentedColormap.from_list('normalization_beta', ['#315F87', '#A7C6DF', '#F8F8F8', '#F1B47D', '#A95316'])

def apply_style() -> None:
    mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], 'font.size': 6.5, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'axes.unicode_minus': False})

def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    parsed = series.astype('string').str.strip().str.lower().map({'true': True, 'false': False, '1': True, '0': False, 'yes': True, 'no': False})
    if parsed.isna().any():
        bad = sorted(series.loc[parsed.isna()].astype(str).unique().tolist())
        raise ValueError(f'Unrecognized boolean values: {bad}')
    return parsed.astype(bool)

def build_main_tms_roi() -> pd.DataFrame:
    subject_table = pd.read_csv(MAIN_UNIFIED, dtype={'subject_id': str})
    subject_ids = subject_table.loc[subject_table['group'].eq('TMS'), 'subject_id'].astype(str).tolist()
    reference = np.load(REFERENCE_ARRAY)
    labels = pd.read_csv(REFERENCE_TABLE).sort_values('roi')
    rows: list[pd.DataFrame] = []
    for (index, subject_id) in enumerate(subject_ids, start=1):
        pre_fc_z = np.load(MAIN_FC / f'{subject_id}_baseline_fc_z.npy')
        post_fc_z = np.load(MAIN_FC / f'{subject_id}_post_fc_z.npy')
        (pre_raw, _) = confirm.fit_provided(np.tanh(pre_fc_z))
        (post_raw, _) = confirm.fit_provided(np.tanh(post_fc_z))
        (pre, post, _) = confirm.separate_orthogonal_alignment(pre_raw, post_raw, reference)
        metrics = confirm.normalization(pre, post, reference)
        rows.append(pd.DataFrame({'subject_id': subject_id, 'roi': np.arange(1, 401), 'roi_name': labels['roi_name'].to_numpy(str), 'network': labels['network'].to_numpy(str), 'group_main_tms': 1, 'baseline_deviation_G2': metrics['pre_dev_G2'], 'normalization_G2': metrics['normalization_G2']}))
        print(f'main TMS unified ROI data {index:02d}/{len(subject_ids)}: {subject_id}', flush=True)
    return pd.concat(rows, ignore_index=True)

def build_external_roi() -> pd.DataFrame:
    frame = pd.read_csv(EXTERNAL_ROI, dtype={'participant_id': str})
    reference = pd.read_csv(REFERENCE_TABLE).sort_values('roi')
    reference_lookup = reference.set_index('roi')['G2_healthy_main_z']
    frame['baseline_deviation_G2'] = np.abs(frame['G2_pre_z'].to_numpy(float) - frame['roi'].map(reference_lookup).to_numpy(float))
    return frame.rename(columns={'participant_id': 'subject_id'}).assign(group_main_tms=0)[['subject_id', 'roi', 'roi_name', 'network', 'group_main_tms', 'baseline_deviation_G2', 'G2_normalization']].rename(columns={'G2_normalization': 'normalization_G2'})

def fit_cross_cohort(main_tms: pd.DataFrame, external: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([main_tms, external], ignore_index=True)
    rows: list[dict[str, object]] = []
    for (roi, frame) in combined.groupby('roi', sort=True):
        data = frame[['normalization_G2', 'group_main_tms', 'baseline_deviation_G2']].dropna()
        y = data['normalization_G2'].to_numpy(float)
        reduced_x = sm.add_constant(data[['baseline_deviation_G2']].astype(float), has_constant='add')
        full_x = sm.add_constant(data[['group_main_tms', 'baseline_deviation_G2']].astype(float), has_constant='add')
        reduced = sm.OLS(y, reduced_x).fit()
        observed = sm.OLS(y, full_x).fit(cov_type='HC3')
        observed_beta = float(observed.params['group_main_tms'])
        rng = np.random.default_rng(SEED + int(roi) * 101)
        indices = np.vstack([rng.permutation(len(y)) for _ in range(N_PERM)])
        permuted_y = reduced.fittedvalues.to_numpy()[None, :] + reduced.resid.to_numpy()[indices]
        group_weight = np.linalg.pinv(full_x.to_numpy(float))[1]
        permuted_beta = permuted_y @ group_weight
        p_value = float((1 + np.count_nonzero(np.abs(permuted_beta) >= abs(observed_beta))) / (N_PERM + 1))
        confidence = observed.conf_int().loc['group_main_tms']
        first = frame.iloc[0]
        rows.append({'roi': int(roi), 'roi_name': str(first['roi_name']), 'network': str(first['network']), 'main_tms_n': int((data['group_main_tms'] == 1).sum()), 'external_n': int((data['group_main_tms'] == 0).sum()), 'main_tms_mean_normalization': float(data.loc[data['group_main_tms'] == 1, 'normalization_G2'].mean()), 'external_mean_normalization': float(data.loc[data['group_main_tms'] == 0, 'normalization_G2'].mean()), 'main_tms_minus_external_adjusted_beta': observed_beta, 'cross_cohort_ci_low_hc3': float(confidence.iloc[0]), 'cross_cohort_ci_high_hc3': float(confidence.iloc[1]), 'cross_cohort_se_hc3': float(observed.bse['group_main_tms']), 'cross_cohort_p_hc3': float(observed.pvalues['group_main_tms']), 'cross_cohort_p_freedman_lane': p_value})
        if int(roi) % 40 == 0:
            print(f'cross-cohort ROI models {int(roi):03d}/400', flush=True)
    output = pd.DataFrame(rows).sort_values('roi')
    output['cross_cohort_fdr_q_400'] = multipletests(output['cross_cohort_p_freedman_lane'], method='fdr_bh')[1]
    return output

def prepare_source_data() -> tuple[pd.DataFrame, dict[str, object]]:
    for path in (MAIN_SOURCE, MAIN_UNIFIED, EXTERNAL_ROI, REFERENCE_TABLE, REFERENCE_ARRAY, MAIN_FC):
        if not path.exists():
            raise FileNotFoundError(path)
    SOURCE.mkdir(parents=True, exist_ok=True)
    main_source = pd.read_csv(MAIN_SOURCE).sort_values('roi')
    main_tms = build_main_tms_roi()
    external = build_external_roi()
    cross = fit_cross_cohort(main_tms, external)
    source = main_source[['roi', 'roi_name', 'network', 'adjusted_TMS_beta_G2', 'permutation_p_G2', 'fdr_q_G2', 'G2_FDR_significant']].merge(cross, on=['roi', 'roi_name', 'network'], validate='one_to_one')
    source['main_internal_FDR_masked_beta'] = np.where(as_bool(source['G2_FDR_significant']), source['adjusted_TMS_beta_G2'], 0.0)
    source['cross_cohort_FDR_significant'] = source['cross_cohort_fdr_q_400'] < 0.05
    source['cross_cohort_FDR_masked_beta'] = np.where(source['cross_cohort_FDR_significant'], source['main_tms_minus_external_adjusted_beta'], 0.0)
    source.to_csv(SOURCE / 'main_internal_and_cross_cohort_G2_beta_maps.csv', index=False, encoding='utf-8-sig')
    main_beta = source['adjusted_TMS_beta_G2'].to_numpy(float)
    cross_beta = source['main_tms_minus_external_adjusted_beta'].to_numpy(float)
    summary = {'main_control_n': 15, 'main_tms_n': int(main_tms['subject_id'].nunique()), 'external_n': int(external['subject_id'].nunique()), 'main_internal_fdr_roi_n': int(as_bool(source['G2_FDR_significant']).sum()), 'cross_cohort_nominal_roi_n': int((source['cross_cohort_p_freedman_lane'] < 0.05).sum()), 'cross_cohort_fdr_roi_n': int(source['cross_cohort_FDR_significant'].sum()), 'cross_cohort_min_p': float(source['cross_cohort_p_freedman_lane'].min()), 'cross_cohort_min_q': float(source['cross_cohort_fdr_q_400'].min()), 'map_pearson_r': float(np.corrcoef(main_beta, cross_beta)[0, 1]), 'n_permutations_per_roi': N_PERM, 'cross_cohort_adjustment': 'baseline G2 deviation only', 'cross_cohort_interpretation': 'exploratory cohort contrast; not a causal TMS effect'}
    (SOURCE / 'analysis_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return (source, summary)

def nice_limit(*arrays: np.ndarray) -> float:
    maximum = max((float(np.nanmax(np.abs(np.asarray(array, float)))) for array in arrays))
    return math.ceil(maximum / 0.05) * 0.05

def render_surface(values: np.ndarray, stem: str, limit: float, significant_mask: np.ndarray | None=None) -> Path:
    cache = QA / 'surface_cache'
    cache.mkdir(parents=True, exist_ok=True)
    output = cache / f'{stem}.png'
    surfaces = fetch_fslr()
    (lh_surf, rh_surf) = surfaces['veryinflated']
    atlas = np.asarray(load_parcellation('schaefer', 400, join=True))
    vertex = np.full(atlas.shape, np.nan, dtype=float)
    for roi in range(1, 401):
        vertex[atlas == roi] = float(values[roi - 1])
    medial_wall = np.full(atlas.shape, np.nan, dtype=float)
    medial_wall[atlas == 0] = 1.0
    significant_vertex = np.full(atlas.shape, np.nan, dtype=float)
    if significant_mask is not None:
        for roi in np.flatnonzero(np.asarray(significant_mask, bool)) + 1:
            significant_vertex[atlas == roi] = 1.0
    plot = Plot(lh_surf, rh_surf, views=['lateral', 'medial'], layout='grid', zoom=1.62, brightness=0.68, size=(2400, 1320), background=(1, 1, 1))
    plot.add_layer(medial_wall, cmap=ListedColormap(['#D1D1D1']), color_range=(0, 1), cbar=False)
    plot.add_layer(vertex, cmap=EFFECT_CMAP, color_range=(-limit, limit), cbar=False)
    plot.add_layer(atlas, cmap=ListedColormap(['#666666']), as_outline=True, cbar=False)
    if significant_mask is not None:
        plot.add_layer(significant_vertex, cmap=ListedColormap(['#8E1B1B']), color_range=(0, 1), as_outline=True, cbar=False)
    figure = plot.build()
    figure.patch.set_facecolor('white')
    figure.savefig(output, dpi=320, bbox_inches='tight', pad_inches=0.0, facecolor='white')
    plt.close(figure)
    return output

def add_surface(figure: plt.Figure, rect: list[float], path: Path) -> None:
    axis = figure.add_axes(rect)
    axis.imshow(mpimg.imread(path))
    axis.set_axis_off()

def make_figure(source: pd.DataFrame, summary: dict[str, object]) -> plt.Figure:
    internal = source['adjusted_TMS_beta_G2'].to_numpy(float)
    cross = source['main_tms_minus_external_adjusted_beta'].to_numpy(float)
    internal_masked = source['main_internal_FDR_masked_beta'].to_numpy(float)
    cross_masked = source['cross_cohort_FDR_masked_beta'].to_numpy(float)
    limit = nice_limit(internal, cross)
    paths = {'internal_all': render_surface(internal, 'internal_all400', limit), 'cross_all': render_surface(cross, 'cross_all400', limit), 'internal_fdr': render_surface(internal_masked, 'internal_fdr', limit, as_bool(source['G2_FDR_significant']).to_numpy()), 'cross_fdr': render_surface(cross_masked, 'cross_fdr', limit, source['cross_cohort_FDR_significant'].to_numpy(bool))}
    figure = plt.figure(figsize=(183 / 25.4, 124 / 25.4), facecolor='white')
    figure.text(0.018, 0.982, 'G2-normalization beta maps', fontsize=9.0, fontweight='bold', ha='left', va='top', color=INK)
    figure.text(0.018, 0.946, 'Within-study treatment contrast versus exploratory cross-cohort contrast', fontsize=6.2, ha='left', va='top', color=MUTED)
    figure.text(0.275, 0.902, 'Main TMS vs Main Control', ha='center', fontsize=7.5, fontweight='bold', color=INK)
    figure.text(0.275, 0.872, 'within-study; fully adjusted', ha='center', fontsize=5.3, color=MUTED)
    figure.text(0.735, 0.902, 'Main TMS vs External', ha='center', fontsize=7.5, fontweight='bold', color=INK)
    figure.text(0.735, 0.872, 'cross-cohort; baseline-deviation adjusted', ha='center', fontsize=5.3, color=MUTED)
    figure.text(0.018, 0.683, 'All 400 ROIs\n(unthresholded)', ha='left', va='center', fontsize=5.6, fontweight='bold', color=INK)
    figure.text(0.018, 0.335, 'FDR q < 0.05', ha='left', va='center', fontsize=5.6, fontweight='bold', color=INK)
    add_surface(figure, [0.105, 0.545, 0.36, 0.29], paths['internal_all'])
    add_surface(figure, [0.535, 0.545, 0.36, 0.29], paths['cross_all'])
    add_surface(figure, [0.105, 0.195, 0.36, 0.29], paths['internal_fdr'])
    add_surface(figure, [0.535, 0.195, 0.36, 0.29], paths['cross_fdr'])
    figure.text(0.275, 0.504, f"{summary['main_internal_fdr_roi_n']} / 400 FDR-significant ROIs", ha='center', fontsize=5.2, color=INK)
    figure.text(0.735, 0.504, f"{summary['cross_cohort_fdr_roi_n']} / 400 FDR-significant ROIs: LH Vis 3; RH Vis 1", ha='center', fontsize=5.2, color=INK)
    cax = figure.add_axes([0.24, 0.132, 0.52, 0.022])
    colorbar = plt.colorbar(mpl.cm.ScalarMappable(norm=Normalize(-limit, limit), cmap=EFFECT_CMAP), cax=cax, orientation='horizontal')
    colorbar.set_ticks([-limit, 0, limit])
    colorbar.set_label('Adjusted group beta for G2 normalization', fontsize=5.8, labelpad=1.0)
    colorbar.ax.tick_params(labelsize=5.2, length=1.7, width=0.45, pad=1.0)
    colorbar.outline.set_linewidth(0.45)
    figure.text(0.018, 0.075, 'Internal model: lesion volume, lesion laterality, baseline G2 deviation, Pre FD and Post FD; HC3 OLS + 5,000 Freedman-Lane permutations.', fontsize=4.7, color=MUTED, ha='left')
    figure.text(0.018, 0.045, 'Cross-cohort model: baseline G2 deviation only; group is confounded with cohort/site and the beta is not a causal TMS effect. BH-FDR is applied separately within each 400-ROI family.', fontsize=4.7, color=MUTED, ha='left')
    figure.text(0.982, 0.075, f"n = {summary['main_tms_n']} TMS / {summary['main_control_n']} Control", fontsize=4.7, color=MUTED, ha='right')
    figure.text(0.982, 0.045, f"n = {summary['main_tms_n']} Main TMS / {summary['external_n']} External", fontsize=4.7, color=MUTED, ha='right')
    return figure

def export(figure: plt.Figure) -> list[Path]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FORAI.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT / 'main_internal_vs_main_tms_external_adjusted_G2_beta_surfaces'
    outputs = [stem.with_suffix(suffix) for suffix in ('.svg', '.pdf', '.png', '.tiff')]
    figure.savefig(outputs[0], bbox_inches='tight', pad_inches=0.02)
    figure.savefig(outputs[1], bbox_inches='tight', pad_inches=0.02)
    figure.savefig(outputs[2], dpi=400, bbox_inches='tight', pad_inches=0.02)
    figure.savefig(outputs[3], dpi=600, bbox_inches='tight', pad_inches=0.02, pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(figure)
    for path in outputs:
        (FORAI / path.name).write_bytes(path.read_bytes())
    return outputs

def write_qa(source: pd.DataFrame, summary: dict[str, object], outputs: list[Path]) -> None:
    QA.mkdir(parents=True, exist_ok=True)
    checks = {'all_outputs_exist': all((path.exists() and path.stat().st_size > 0 for path in outputs)), 'roi_n': int(len(source)), 'roi_order_is_1_to_400': bool(np.array_equal(source['roi'].to_numpy(), np.arange(1, 401))), 'main_internal_fdr_roi_n': summary['main_internal_fdr_roi_n'], 'cross_cohort_fdr_roi_n': summary['cross_cohort_fdr_roi_n'], 'shared_colour_scale': True, 'spatial_smoothing_or_interpolation': False, 'cross_cohort_causal_label_used': False}
    (QA / 'qa_metrics.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    report = '\n'.join(['# QA report', '', '- Backend: Python only.', '- The two columns and both thresholding rows share the same symmetric colour scale.', '- Parcel values are projected directly to Schaefer400; no smoothing or interpolation is applied.', f"- Internal FDR-significant ROI count: {summary['main_internal_fdr_roi_n']}.", f"- Cross-cohort FDR-significant ROI count: {summary['cross_cohort_fdr_roi_n']}.", '- The cross-cohort contrast is explicitly labelled exploratory and non-causal.', '- SVG/PDF text is editable; cortical renders are embedded raster plates.'])
    (QA / 'QA_REPORT.md').write_text(report, encoding='utf-8')

def main() -> None:
    apply_style()
    (source, summary) = prepare_source_data()
    figure = make_figure(source, summary)
    outputs = export(figure)
    write_qa(source, summary, outputs)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for output in outputs:
        print(output)
if __name__ == '__main__':
    main()
