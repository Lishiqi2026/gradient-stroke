from __future__ import annotations
'Recompute longitudinal G1/G2 with the provided gradient-code logic.\n\nThe numerical embedding follows the core of\n``E:/paper/lesion_refresh_20261028/work/Gradient/code/gradient_calculation/allnpy.py`` at Schaefer-400 resolution:\nFisher z -> strongest 10% positive profiles -> cosine affinity -> diffusion map.\nThe resulting five-dimensional embeddings are aligned, using baseline\nlesion-intact parcels, to one fixed healthy reference calculated with the same\nadapted embedding.  The healthy axes are fixed once by five-dimensional\nalignment to the established healthy reference so G1/G2 signs and rotations do\nnot change across participants or sessions.\n'
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
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from brainspace.datasets import load_group_fc, load_parcellation
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
from neuromaps.datasets import fetch_fslr
from statsmodels.stats.multitest import multipletests
from surfplot import Plot
ROOT = Path(_portable_path('project', 'lesion_analysis'))
ANALYSIS = ROOT / 'lesion_gradient_distance'
TABLE_DIR = ANALYSIS / 'tables'
FC_DIR = ANALYSIS / 'fc_consistent' / 'fc_matrices_z'
DATA_DIR = ANALYSIS / 'provided_code_g1_g2_recalculation'
GRADIENT_DIR = DATA_DIR / 'aligned_gradients_z'
FIGURE_DIR = ANALYSIS / 'figures' / 'nature' / 'provided_code_g1_g2_surfaces'
sys.path.insert(0, str(ROOT))
import run_lesion_gradient_distance as current
from audit_healthy_reference_gradient_calculation import legacy_style_adapted_embedding, zscore_columns
SEED = 20260810
N_PERMUTATIONS = 5000
mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'], 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.size': 7, 'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'axes.unicode_minus': False})
G1_CMAP = LinearSegmentedColormap.from_list('g1_reference', ['#315F87', '#A7C6DF', '#F8F8F8', '#E8A07A', '#A84B3C'])
G2_CMAP = LinearSegmentedColormap.from_list('g2_reference', ['#B48A2C', '#E7D39B', '#F8F8F8', '#B7ADD2', '#5E4B8B'])
NORMALIZATION_CMAP = LinearSegmentedColormap.from_list('normalization', ['#315F87', '#A7C6DF', '#F8F8F8', '#F1B47D', '#A95316'])

def robust_symmetric_limit(*arrays: np.ndarray, minimum: float=0.02) -> float:
    values = np.concatenate([np.ravel(np.asarray(array, dtype=float)) for array in arrays])
    values = values[np.isfinite(values)]
    return max(float(np.percentile(np.abs(values), 98)), minimum)

def render_surface_hd(values: np.ndarray, *, title: str, colorbar_label: str, cmap, limit: float, stem: Path) -> None:
    surfaces = fetch_fslr()
    (lh_surf, rh_surf) = surfaces['veryinflated']
    atlas = load_parcellation('schaefer', 400, join=True)
    vertex_values = np.full_like(atlas, np.nan, dtype=float)
    for roi in range(1, 401):
        vertex_values[atlas == roi] = values[roi - 1]
    medial_wall = np.full_like(atlas, np.nan, dtype=float)
    medial_wall[atlas == 0] = 1.0
    plot = Plot(lh_surf, rh_surf, views=['lateral', 'medial'], zoom=1.7, brightness=0.66, size=(4800, 3200), background=(1, 1, 1))
    plot.add_layer(medial_wall, cmap=ListedColormap(['#D1D1D1']), color_range=(0, 1), cbar=False)
    plot.add_layer(vertex_values, cmap=cmap, color_range=(-limit, limit), cbar=False)
    plot.add_layer(atlas, cmap=ListedColormap(['#606060']), as_outline=True, cbar=False)
    figure = plot.build()
    figure.patch.set_facecolor('white')
    figure.set_size_inches(6.4, 4.15, forward=True)
    brain_axes = figure.axes[:4]
    if len(brain_axes) >= 4:
        positions = [[0.03, 0.515, 0.455, 0.37], [0.515, 0.515, 0.455, 0.37], [0.03, 0.145, 0.455, 0.37], [0.515, 0.145, 0.455, 0.37]]
        for (axis, position) in zip(brain_axes, positions):
            axis.set_position(position)
    figure.suptitle(title, x=0.5, y=0.968, ha='center', va='top', fontsize=10.0, fontweight='bold', fontname='Arial')
    color_axis = figure.add_axes([0.285, 0.075, 0.43, 0.026])
    colorbar = figure.colorbar(mpl.cm.ScalarMappable(norm=Normalize(-limit, limit), cmap=cmap), cax=color_axis, orientation='horizontal')
    colorbar.set_label(colorbar_label, fontsize=7.5, labelpad=1.8)
    colorbar.ax.tick_params(labelsize=6.7, length=2.0, width=0.55, pad=1.2)
    colorbar.outline.set_linewidth(0.55)
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(stem.with_suffix('.pdf'), bbox_inches='tight', pad_inches=0.02)
    figure.savefig(stem.with_suffix('.svg'), bbox_inches='tight', pad_inches=0.02)
    figure.savefig(stem.with_suffix('.png'), dpi=600, bbox_inches='tight', pad_inches=0.02)
    figure.savefig(stem.with_suffix('.tiff'), dpi=600, bbox_inches='tight', pad_inches=0.02, pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(figure)

def fit_roi_group_model(frame: pd.DataFrame, *, outcome: str, baseline_deviation: str, roi: int) -> dict[str, float]:
    covariates = ['log1p_lesion_volume', 'mask_lateralization_index', baseline_deviation, 'baseline_mean_fd', 'post_mean_fd']
    data = frame.loc[frame['intact_roi'].astype(bool), [outcome, 'group_tms', *covariates]].dropna()
    y = data[outcome].to_numpy(float)
    reduced_x = sm.add_constant(data[covariates].astype(float), has_constant='add')
    full_x = sm.add_constant(data[['group_tms', *covariates]].astype(float), has_constant='add')
    reduced = sm.OLS(y, reduced_x).fit()
    observed = sm.OLS(y, full_x).fit(cov_type='HC3')
    rng = np.random.default_rng(SEED + roi * 101 + (0 if 'G1' in outcome else 1000000))
    permutation_indices = np.vstack([rng.permutation(len(y)) for _ in range(N_PERMUTATIONS)])
    permuted_y = reduced.fittedvalues.to_numpy()[None, :] + reduced.resid.to_numpy()[permutation_indices]
    group_weight = np.linalg.pinv(full_x.to_numpy(float))[1]
    permuted_beta = permuted_y @ group_weight
    observed_beta = float(observed.params['group_tms'])
    permutation_p = float((1 + np.count_nonzero(np.abs(permuted_beta) >= abs(observed_beta))) / (N_PERMUTATIONS + 1))
    confidence = observed.conf_int().loc['group_tms']
    return {'roi': roi, 'outcome': outcome, 'n': int(observed.nobs), 'control_n': int((data['group_tms'] == 0).sum()), 'tms_n': int((data['group_tms'] == 1).sum()), 'adjusted_group_tms_beta': observed_beta, 'ci_low_hc3': float(confidence.iloc[0]), 'ci_high_hc3': float(confidence.iloc[1]), 'se_hc3': float(observed.bse['group_tms']), 'p_hc3': float(observed.pvalues['group_tms']), 'p_freedman_lane': permutation_p, 'r_squared': float(observed.rsquared)}

def make_contact_sheet(paths: list[Path], *, nrows: int, ncols: int, stem: Path, figsize: tuple[float, float]) -> None:
    (figure, axes) = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    for (axis, path) in zip(axes.ravel(), paths):
        axis.imshow(mpimg.imread(path))
        axis.set_axis_off()
    for axis in axes.ravel()[len(paths):]:
        axis.set_axis_off()
    figure.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.995, wspace=0.025, hspace=0.025)
    figure.savefig(stem.with_suffix('.pdf'), bbox_inches='tight', pad_inches=0.02)
    figure.savefig(stem.with_suffix('.svg'), bbox_inches='tight', pad_inches=0.02)
    figure.savefig(stem.with_suffix('.png'), dpi=400, bbox_inches='tight', pad_inches=0.02)
    plt.close(figure)

def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    GRADIENT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(TABLE_DIR / 'healthy_normative_reference_schaefer400.csv')
    subject = pd.read_csv(TABLE_DIR / 'subject_level_distance_metrics.csv', dtype={'subject_id': str})
    old_roi = pd.read_csv(TABLE_DIR / 'roi_level_distance_reorganization.csv', dtype={'subject_id': str})
    intact_lookup = {str(subject_id): group.sort_values('roi')['intact_roi'].astype(bool).to_numpy() for (subject_id, group) in old_roi.groupby('subject_id')}
    healthy_fc = load_group_fc('schaefer', scale=400, group='main')
    (healthy_legacy_raw, healthy_lambdas, _) = legacy_style_adapted_embedding(healthy_fc)
    official_reference = np.load(ANALYSIS / 'references' / 'healthy_normative_reference_400x5.npy')
    (healthy_master, healthy_alignment) = current.similarity_align(healthy_legacy_raw, official_reference)
    master_mean = healthy_master.mean(axis=0)
    master_sd = healthy_master.std(axis=0, ddof=1)
    healthy_master_z = (healthy_master - master_mean) / master_sd
    subject_rows: list[pd.DataFrame] = []
    qc_rows: list[dict] = []
    for row in subject.itertuples(index=False):
        subject_id = str(row.subject_id)
        group = str(row.group)
        intact = intact_lookup[subject_id]
        baseline_z_fc = np.load(FC_DIR / f'{subject_id}_baseline_fc_z.npy')
        post_z_fc = np.load(FC_DIR / f'{subject_id}_post_fc_z.npy')
        (before_raw, before_lambdas, _) = legacy_style_adapted_embedding(np.tanh(baseline_z_fc))
        (after_raw, after_lambdas, _) = legacy_style_adapted_embedding(np.tanh(post_z_fc))
        (before_aligned, before_fit) = current.similarity_align(before_raw, healthy_master, intact)
        (after_aligned, after_fit) = current.similarity_align(after_raw, healthy_master, intact)
        before_z = (before_aligned - master_mean) / master_sd
        after_z = (after_aligned - master_mean) / master_sd
        np.save(GRADIENT_DIR / f'{subject_id}_pre_aligned_z_400x5.npy', before_z)
        np.save(GRADIENT_DIR / f'{subject_id}_post_aligned_z_400x5.npy', after_z)
        pre_dev_g1 = np.abs(before_z[:, 0] - healthy_master_z[:, 0])
        post_dev_g1 = np.abs(after_z[:, 0] - healthy_master_z[:, 0])
        pre_dev_g2 = np.abs(before_z[:, 1] - healthy_master_z[:, 1])
        post_dev_g2 = np.abs(after_z[:, 1] - healthy_master_z[:, 1])
        roi_frame = labels[['roi', 'roi_name', 'network']].copy()
        roi_frame.insert(0, 'group', group)
        roi_frame.insert(0, 'subject_id', subject_id)
        roi_frame['group_tms'] = int(group == 'TMS')
        roi_frame['primary_motion_eligible'] = bool(row.primary_motion_eligible)
        roi_frame['intact_roi'] = intact
        roi_frame['log1p_lesion_volume'] = float(row.log1p_lesion_volume)
        roi_frame['mask_lateralization_index'] = float(row.mask_lateralization_index)
        roi_frame['baseline_mean_fd'] = float(row.baseline_mean_fd)
        roi_frame['post_mean_fd'] = float(row.post_mean_fd)
        roi_frame['G1_reference_z'] = healthy_master_z[:, 0]
        roi_frame['G2_reference_z'] = healthy_master_z[:, 1]
        roi_frame['G1_pre_z'] = before_z[:, 0]
        roi_frame['G1_post_z'] = after_z[:, 0]
        roi_frame['G2_pre_z'] = before_z[:, 1]
        roi_frame['G2_post_z'] = after_z[:, 1]
        roi_frame['baseline_deviation_G1'] = pre_dev_g1
        roi_frame['baseline_deviation_G2'] = pre_dev_g2
        roi_frame['normalization_G1'] = pre_dev_g1 - post_dev_g1
        roi_frame['normalization_G2'] = pre_dev_g2 - post_dev_g2
        subject_rows.append(roi_frame)
        qc_rows.append({'subject_id': subject_id, 'group': group, 'primary_motion_eligible': bool(row.primary_motion_eligible), 'intact_roi_n': int(intact.sum()), **{f'pre_{key}': value for (key, value) in before_fit.items()}, **{f'post_{key}': value for (key, value) in after_fit.items()}, **{f'pre_lambda_G{i + 1}': float(value) for (i, value) in enumerate(before_lambdas)}, **{f'post_lambda_G{i + 1}': float(value) for (i, value) in enumerate(after_lambdas)}})
    roi_long = pd.concat(subject_rows, ignore_index=True)
    roi_long.to_csv(DATA_DIR / 'provided_code_roi_normalization_long.csv', index=False)
    pd.DataFrame(qc_rows).to_csv(DATA_DIR / 'subject_alignment_qc.csv', index=False)
    np.save(DATA_DIR / 'healthy_reference_provided_code_aligned_400x5.npy', healthy_master)
    np.save(DATA_DIR / 'healthy_reference_provided_code_aligned_z_400x5.npy', healthy_master_z)
    primary = roi_long.loc[roi_long['primary_motion_eligible'].astype(bool)].copy()
    map_rows: list[dict] = []
    model_rows: list[dict] = []
    for (roi, roi_frame) in primary.groupby('roi', sort=True):
        intact_frame = roi_frame.loc[roi_frame['intact_roi'].astype(bool)]
        base = {'roi': int(roi), 'roi_name': str(roi_frame['roi_name'].iloc[0]), 'network': str(roi_frame['network'].iloc[0]), 'control_n_intact': int((intact_frame['group'] == 'Control').sum()), 'tms_n_intact': int((intact_frame['group'] == 'TMS').sum())}
        for gradient in ('G1', 'G2'):
            control_mean = float(intact_frame.loc[intact_frame['group'] == 'Control', f'normalization_{gradient}'].mean())
            tms_mean = float(intact_frame.loc[intact_frame['group'] == 'TMS', f'normalization_{gradient}'].mean())
            base[f'Control_mean_{gradient}_normalization'] = control_mean
            base[f'TMS_mean_{gradient}_normalization'] = tms_mean
            base[f'TMS_minus_Control_{gradient}_normalization'] = tms_mean - control_mean
            model_rows.append(fit_roi_group_model(roi_frame, outcome=f'normalization_{gradient}', baseline_deviation=f'baseline_deviation_{gradient}', roi=int(roi)))
        map_rows.append(base)
    maps = pd.DataFrame(map_rows).sort_values('roi')
    models = pd.DataFrame(model_rows)
    models['gradient'] = models['outcome'].str.extract('(G[12])')
    models['fdr_q_within_gradient'] = np.nan
    for gradient in ('G1', 'G2'):
        selected = models['gradient'] == gradient
        models.loc[selected, 'fdr_q_within_gradient'] = multipletests(models.loc[selected, 'p_freedman_lane'], method='fdr_bh')[1]
    models['fdr_q_across_G1_G2'] = multipletests(models['p_freedman_lane'], method='fdr_bh')[1]
    model_wide = models.pivot(index='roi', columns='gradient')
    for gradient in ('G1', 'G2'):
        maps[f'adjusted_TMS_beta_{gradient}'] = maps['roi'].map(model_wide['adjusted_group_tms_beta'][gradient])
        maps[f'permutation_p_{gradient}'] = maps['roi'].map(model_wide['p_freedman_lane'][gradient])
        maps[f'fdr_q_{gradient}'] = maps['roi'].map(model_wide['fdr_q_within_gradient'][gradient])
    maps['G1_reference_z'] = healthy_master_z[:, 0]
    maps['G2_reference_z'] = healthy_master_z[:, 1]
    maps.to_csv(DATA_DIR / 'provided_code_G1_G2_surface_source_data.csv', index=False)
    models.to_csv(DATA_DIR / 'provided_code_roi_adjusted_group_models.csv', index=False)
    figure_stems: dict[str, Path] = {}
    for (gradient, cmap) in (('G1', G1_CMAP), ('G2', G2_CMAP)):
        reference = maps[f'{gradient}_reference_z'].to_numpy(float)
        control = maps[f'Control_mean_{gradient}_normalization'].to_numpy(float)
        tms = maps[f'TMS_mean_{gradient}_normalization'].to_numpy(float)
        difference = maps[f'TMS_minus_Control_{gradient}_normalization'].to_numpy(float)
        adjusted = maps[f'adjusted_TMS_beta_{gradient}'].to_numpy(float)
        reference_limit = robust_symmetric_limit(reference)
        group_limit = robust_symmetric_limit(control, tms)
        effect_limit = robust_symmetric_limit(difference, adjusted)
        specifications = [('reference', reference, f'Healthy normative {gradient} (provided pipeline)', f'Reference {gradient} (z)', cmap, reference_limit), ('control_mean', control, f'Control mean {gradient} normalization', 'Mean normalization', NORMALIZATION_CMAP, group_limit), ('tms_mean', tms, f'TMS mean {gradient} normalization', 'Mean normalization', NORMALIZATION_CMAP, group_limit), ('unadjusted_difference', difference, f'TMS - Control {gradient} normalization (unadjusted)', 'Mean normalization difference', NORMALIZATION_CMAP, effect_limit), ('adjusted_beta', adjusted, f'Adjusted TMS effect on {gradient} normalization', 'Adjusted TMS beta', NORMALIZATION_CMAP, effect_limit)]
        for (key, values, title, colorbar_label, plot_cmap, limit) in specifications:
            stem = FIGURE_DIR / f'{gradient}_{key}_surface_hd'
            render_surface_hd(values, title=title, colorbar_label=colorbar_label, cmap=plot_cmap, limit=limit, stem=stem)
            figure_stems[f'{gradient}_{key}'] = stem
    main_paths = [figure_stems['G1_reference'].with_suffix('.png'), figure_stems['G1_adjusted_beta'].with_suffix('.png'), figure_stems['G2_reference'].with_suffix('.png'), figure_stems['G2_adjusted_beta'].with_suffix('.png')]
    make_contact_sheet(main_paths, nrows=2, ncols=2, stem=FIGURE_DIR / 'main_candidate_reference_and_adjusted_effect_G1_G2', figsize=(12.8, 8.3))
    full_paths = []
    for gradient in ('G1', 'G2'):
        for key in ('reference', 'control_mean', 'tms_mean', 'unadjusted_difference', 'adjusted_beta'):
            full_paths.append(figure_stems[f'{gradient}_{key}'].with_suffix('.png'))
    make_contact_sheet(full_paths, nrows=2, ncols=5, stem=FIGURE_DIR / 'all_available_G1_G2_surface_maps_contact_sheet', figsize=(25.0, 8.3))
    manifest = {'gradient_embedding': 'provided gradient_calculation/allnpy.py core adapted to Schaefer400', 'healthy_reference': 'BrainSpace packaged healthy main/discovery Schaefer400 group FC', 'healthy_axis_fixing': 'five-dimensional similarity alignment to the established fixed healthy reference', 'subject_alignment': 'five-dimensional similarity alignment using baseline lesion-intact ROIs', 'normalization': 'abs(pre_z - healthy_z) - abs(post_z - healthy_z)', 'primary_cohort': {'Control': int(primary.loc[primary['group'] == 'Control', 'subject_id'].nunique()), 'TMS': int(primary.loc[primary['group'] == 'TMS', 'subject_id'].nunique())}, 'adjusted_roi_model': 'normalization ~ TMS + log1p lesion volume + lesion lateralization + ROI baseline deviation + baseline mean FD + post mean FD', 'inference': f'HC3 CI; Freedman-Lane {N_PERMUTATIONS} permutations; BH-FDR within G1/G2 and across 800 tests', 'continuous_effect_map_warning': 'adjusted beta surfaces are effect-size maps, not thresholded significance maps', 'healthy_lambdas': healthy_lambdas.tolist(), 'healthy_alignment': healthy_alignment}
    (DATA_DIR / 'analysis_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    summary = []
    for gradient in ('G1', 'G2'):
        selected = models.loc[models['gradient'] == gradient]
        summary.append({'gradient': gradient, 'min_permutation_p': float(selected['p_freedman_lane'].min()), 'min_fdr_q_within_gradient': float(selected['fdr_q_within_gradient'].min()), 'fdr_significant_roi_n': int((selected['fdr_q_within_gradient'] < 0.05).sum()), 'mean_unadjusted_TMS_minus_Control': float(maps[f'TMS_minus_Control_{gradient}_normalization'].mean()), 'mean_adjusted_TMS_beta': float(maps[f'adjusted_TMS_beta_{gradient}'].mean())})
    pd.DataFrame(summary).to_csv(DATA_DIR / 'surface_map_summary.csv', index=False)
    print(pd.DataFrame(summary).to_string(index=False))
    print(f'Saved figures: {FIGURE_DIR}')
    print(f'Saved source data: {DATA_DIR}')
if __name__ == '__main__':
    main()
