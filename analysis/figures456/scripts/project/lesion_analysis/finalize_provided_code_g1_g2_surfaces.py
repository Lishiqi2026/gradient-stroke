from __future__ import annotations
'Post-process the provided-code G1/G2 recalculation without recomputing gradients.'
import os as _portable_os
from pathlib import Path as _PortablePath

def _portable_path(kind, relative):
    project = _PortablePath(_portable_os.environ['PAPER_PROJECT_ROOT'])
    root = project if kind == 'project' else project.parent
    return str(root / relative)
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
ROOT = Path(_portable_path('project', 'lesion_analysis'))
ANALYSIS = ROOT / 'lesion_gradient_distance'
DATA_DIR = ANALYSIS / 'provided_code_g1_g2_recalculation'
FIGURE_DIR = ANALYSIS / 'figures' / 'nature' / 'provided_code_g1_g2_surfaces'
sys.path.insert(0, str(ROOT))
import run_lesion_gradient_distance as current
from make_provided_code_g1_g2_normalization_surfaces import NORMALIZATION_CMAP, make_contact_sheet, render_surface_hd, robust_symmetric_limit

def main() -> None:
    roi = pd.read_csv(DATA_DIR / 'provided_code_roi_normalization_long.csv', dtype={'subject_id': str})
    maps = pd.read_csv(DATA_DIR / 'provided_code_G1_G2_surface_source_data.csv')
    models = pd.read_csv(DATA_DIR / 'provided_code_roi_adjusted_group_models.csv')
    primary = roi.loc[roi['primary_motion_eligible'].astype(bool) & roi['intact_roi'].astype(bool)].copy()
    subject_rows = []
    for (subject_id, frame) in primary.groupby('subject_id', sort=True):
        subject_rows.append({'subject_id': str(subject_id), 'group': str(frame['group'].iloc[0]), 'group_tms': int(frame['group_tms'].iloc[0]), 'log1p_lesion_volume': float(frame['log1p_lesion_volume'].iloc[0]), 'mask_lateralization_index': float(frame['mask_lateralization_index'].iloc[0]), 'baseline_mean_fd': float(frame['baseline_mean_fd'].iloc[0]), 'post_mean_fd': float(frame['post_mean_fd'].iloc[0]), 'intact_roi_n': int(len(frame)), 'mean_normalization_G1': float(frame['normalization_G1'].mean()), 'mean_normalization_G2': float(frame['normalization_G2'].mean()), 'mean_baseline_deviation_G1': float(frame['baseline_deviation_G1'].mean()), 'mean_baseline_deviation_G2': float(frame['baseline_deviation_G2'].mean())})
    subject = pd.DataFrame(subject_rows)
    subject.to_csv(DATA_DIR / 'provided_code_subject_level_global_metrics.csv', index=False)
    rng = np.random.default_rng(20260810)
    global_rows = []
    for gradient in ('G1', 'G2'):
        global_rows.append(current.freedman_lane_group_test(subject, outcome=f'mean_normalization_{gradient}', baseline_deviation=f'mean_baseline_deviation_{gradient}', extra_covariates=[], n_permutations=5000, rng=rng))
    global_models = pd.DataFrame(global_rows)
    global_models['fdr_q_two_gradients'] = multipletests(global_models['p_freedman_lane'], method='fdr_bh')[1]
    global_models.to_csv(DATA_DIR / 'provided_code_global_adjusted_group_models.csv', index=False)
    alignment_qc = pd.read_csv(DATA_DIR / 'subject_alignment_qc.csv', dtype={'subject_id': str})
    alignment_columns = ['pre_fit_G1_correlation', 'pre_fit_G2_correlation', 'post_fit_G1_correlation', 'post_fit_G2_correlation']
    alignment_qc['minimum_G1_G2_alignment_correlation'] = alignment_qc[alignment_columns].min(axis=1)
    sensitivity_data = subject.merge(alignment_qc[['subject_id', 'minimum_G1_G2_alignment_correlation']], on='subject_id', how='inner', validate='one_to_one')
    sensitivity_rows = []
    sensitivity_rng = np.random.default_rng(20260810)
    for (set_name, threshold) in (('all_primary_motion_qc', -np.inf), ('minimum_alignment_correlation_ge_0.30', 0.3), ('minimum_alignment_correlation_ge_0.40', 0.4)):
        selected_data = sensitivity_data.loc[sensitivity_data['minimum_G1_G2_alignment_correlation'] >= threshold].copy()
        for gradient in ('G1', 'G2'):
            result = current.freedman_lane_group_test(selected_data, outcome=f'mean_normalization_{gradient}', baseline_deviation=f'mean_baseline_deviation_{gradient}', extra_covariates=[], n_permutations=5000, rng=sensitivity_rng)
            sensitivity_rows.append({'set': set_name, 'minimum_alignment_correlation_threshold': np.nan if not np.isfinite(threshold) else threshold, 'gradient': gradient, 'control_n': int((selected_data['group'] == 'Control').sum()), 'tms_n': int((selected_data['group'] == 'TMS').sum()), **result})
    pd.DataFrame(sensitivity_rows).to_csv(DATA_DIR / 'provided_code_alignment_quality_sensitivity.csv', index=False)
    g1_beta = maps['adjusted_TMS_beta_G1'].to_numpy(float)
    g1_stem = FIGURE_DIR / 'G1_adjusted_beta_no_FDR_ROIs_surface_hd'
    render_surface_hd(g1_beta, title='Adjusted TMS effect on G1 normalization (no ROI survives FDR)', colorbar_label='Adjusted TMS beta (continuous effect-size map)', cmap=NORMALIZATION_CMAP, limit=robust_symmetric_limit(g1_beta), stem=g1_stem)
    g2_models = models.loc[models['gradient'] == 'G2'].set_index('roi')
    g2_q = maps['roi'].map(g2_models['fdr_q_within_gradient']).to_numpy(float)
    g2_beta = maps['adjusted_TMS_beta_G2'].to_numpy(float)
    g2_significant = g2_q < 0.05
    maps['G2_FDR_significant'] = g2_significant
    maps['G2_adjusted_beta_FDR_masked'] = np.where(g2_significant, g2_beta, 0.0)
    maps.to_csv(DATA_DIR / 'provided_code_G1_G2_surface_source_data.csv', index=False)
    effect_limit = robust_symmetric_limit(g2_beta)
    masked_stem = FIGURE_DIR / 'G2_adjusted_beta_FDR_q05_masked_surface_hd'
    render_surface_hd(maps['G2_adjusted_beta_FDR_masked'].to_numpy(float), title=f'Adjusted TMS effect on G2 normalization: FDR q < 0.05 ({g2_significant.sum()} ROIs)', colorbar_label='Adjusted TMS beta (non-significant ROIs shown as white)', cmap=NORMALIZATION_CMAP, limit=effect_limit, stem=masked_stem)
    make_contact_sheet([FIGURE_DIR / 'G1_reference_surface_hd.png', g1_stem.with_suffix('.png'), FIGURE_DIR / 'G2_reference_surface_hd.png', masked_stem.with_suffix('.png')], nrows=2, ncols=2, stem=FIGURE_DIR / 'main_candidate_with_G2_FDR_mask', figsize=(12.8, 8.3))
    print(global_models.to_string(index=False))
    print(f'G2 FDR-significant ROIs: {int(g2_significant.sum())}')
if __name__ == '__main__':
    main()
