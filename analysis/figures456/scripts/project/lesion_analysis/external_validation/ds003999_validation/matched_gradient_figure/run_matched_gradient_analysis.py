from __future__ import annotations
import os as _portable_os
from pathlib import Path as _PortablePath

def _portable_path(kind, relative):
    project = _PortablePath(_portable_os.environ['PAPER_PROJECT_ROOT'])
    root = project if kind == 'project' else project.parent
    return str(root / relative)
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as t_dist
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests
HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))
import run_external_validation_analysis as base
TABLES = HERE / 'tables'
SOURCE = HERE / 'source_data'
QA = HERE / 'qa'
SEED = 20260811
N_PERM_SELECTED = 10000

def ensure_dirs() -> None:
    for path in (TABLES, SOURCE, QA):
        path.mkdir(parents=True, exist_ok=True)

def load_paired_gradients(participants: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    (pre, post) = ([], [])
    for sid in participants['participant_id']:
        pre_path = base.locate_array(base.LEGACY_ROOT, 'pre', sid)
        post_path = base.locate_array(base.LEGACY_ROOT, 'post', sid)
        pre_arr = np.asarray(np.load(pre_path), dtype=float)
        post_arr = np.asarray(np.load(post_path), dtype=float)
        if pre_arr.shape != (400, 5) or post_arr.shape != (400, 5):
            raise RuntimeError(f'Unexpected aligned5 shape for {sid}: {pre_arr.shape}, {post_arr.shape}')
        pre.append(pre_arr)
        post.append(post_arr)
    return (np.stack(pre), np.stack(post))

def roi_paired_tests(pre: np.ndarray, post: np.ndarray, labels: pd.DataFrame, participant_ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    delta_rows: list[dict[str, object]] = []
    n = len(participant_ids)
    for (gidx, gname) in enumerate(('G1', 'G2')):
        delta = post[:, :, gidx] - pre[:, :, gidx]
        (t_values, p_values) = ttest_rel(post[:, :, gidx], pre[:, :, gidx], axis=0)
        q_values = multipletests(p_values, method='fdr_bh')[1]
        means = np.mean(delta, axis=0)
        sds = np.std(delta, axis=0, ddof=1)
        sems = sds / np.sqrt(n)
        crit = t_dist.ppf(0.975, n - 1)
        for roi0 in range(400):
            meta = labels.iloc[roi0]
            rows.append({'gradient': gname, 'roi': roi0 + 1, 'roi_name': meta['roi_name'], 'network': meta['network'], 'hemisphere': meta['hemisphere'], 'n': n, 'mean_delta_post_minus_pre': means[roi0], 'sd_delta': sds[roi0], 'ci_low': means[roi0] - crit * sems[roi0], 'ci_high': means[roi0] + crit * sems[roi0], 'paired_cohen_dz': means[roi0] / sds[roi0] if sds[roi0] > 0 else np.nan, 't': t_values[roi0], 'p_two_sided': p_values[roi0], 'fdr_q_400': q_values[roi0], 'nominal_p_lt_0_05': bool(p_values[roi0] < 0.05), 'fdr_q_lt_0_05': bool(q_values[roi0] < 0.05)})
        for (sidx, sid) in enumerate(participant_ids):
            for roi0 in range(400):
                delta_rows.append({'participant_id': sid, 'gradient': gname, 'roi': roi0 + 1, 'delta_post_minus_pre': delta[sidx, roi0]})
    return (pd.DataFrame(rows), pd.DataFrame(delta_rows))

def roi_sensitivity_without_sub35(pre: np.ndarray, post: np.ndarray, labels: pd.DataFrame, participant_ids: list[str]) -> pd.DataFrame:
    keep = np.array([sid != 'sub-35' for sid in participant_ids], dtype=bool)
    rows: list[dict[str, object]] = []
    for (gidx, gname) in enumerate(('G1', 'G2')):
        (t_values, p_values) = ttest_rel(post[keep, :, gidx], pre[keep, :, gidx], axis=0)
        q_values = multipletests(p_values, method='fdr_bh')[1]
        for roi0 in range(400):
            rows.append({'gradient': gname, 'roi': roi0 + 1, 'roi_name': labels.iloc[roi0]['roi_name'], 'n': int(keep.sum()), 't': t_values[roi0], 'p_two_sided': p_values[roi0], 'fdr_q_400': q_values[roi0]})
    return pd.DataFrame(rows)

def make_network_counts(roi_stats: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for ((gradient, network), frame) in roi_stats.groupby(['gradient', 'network'], sort=False):
        nominal = frame['nominal_p_lt_0_05']
        fdr = frame['fdr_q_lt_0_05']
        rows.append({'gradient': gradient, 'network': network, 'nominal_negative_n': int((nominal & (frame['t'] < 0)).sum()), 'nominal_positive_n': int((nominal & (frame['t'] > 0)).sum()), 'fdr_negative_n': int((fdr & (frame['t'] < 0)).sum()), 'fdr_positive_n': int((fdr & (frame['t'] > 0)).sum()), 'network_roi_n': int(len(frame))})
    return pd.DataFrame(rows)

def prepare_network_metric_effects() -> pd.DataFrame:
    src = PARENT / 'tables' / 'legacy_aligned5_exact_network_tests.csv'
    frame = pd.read_csv(src)
    frame['paired_cohen_dz'] = frame['t'] / np.sqrt(frame['n'])
    frame['dz_ci_low_approx'] = frame['ci_low'] / frame['sd']
    frame['dz_ci_high_approx'] = frame['ci_high'] / frame['sd']
    frame['nominal_p_lt_0_05'] = frame['p_signflip_two_sided'] < 0.05
    frame['fdr_q_lt_0_05'] = frame['fdr_q_42_exact_indicator_tests'] < 0.05
    return frame

def prepare_component_variance() -> tuple[pd.DataFrame, pd.DataFrame]:
    qc_path = PARENT / 'qa' / 'provided_code_gradient_alignment_qc.csv'
    qc = pd.read_csv(qc_path)
    rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []
    for phase in ('pre', 'post'):
        eigen = qc[[f'{phase}_lambda_G{k}' for k in range(1, 6)]].to_numpy(float)
        shares = eigen / eigen.sum(axis=1, keepdims=True)
        for (sidx, sid) in enumerate(qc['participant_id']):
            for k in range(1, 6):
                rows.append({'participant_id': sid, 'phase': phase, 'component': f'G{k}', 'eigenvalue': eigen[sidx, k - 1], 'share_of_retained_G1_G5': shares[sidx, k - 1]})
    long = pd.DataFrame(rows)
    for k in range(1, 6):
        pre_vals = long.query("phase == 'pre' and component == @comp", local_dict={'comp': f'G{k}'})['share_of_retained_G1_G5'].to_numpy(float)
        post_vals = long.query("phase == 'post' and component == @comp", local_dict={'comp': f'G{k}'})['share_of_retained_G1_G5'].to_numpy(float)
        (t_value, p_value) = ttest_rel(post_vals, pre_vals)
        delta = post_vals - pre_vals
        test_rows.append({'component': f'G{k}', 'n': len(delta), 'pre_mean': float(np.mean(pre_vals)), 'post_mean': float(np.mean(post_vals)), 'mean_delta': float(np.mean(delta)), 'paired_cohen_dz': float(np.mean(delta) / np.std(delta, ddof=1)), 't': float(t_value), 'p_two_sided': float(p_value)})
    tests = pd.DataFrame(test_rows)
    tests['fdr_q_5_components'] = multipletests(tests['p_two_sided'], method='fdr_bh')[1]
    return (long, tests)

def clinical_roi_models(pre: np.ndarray, post: np.ndarray, participants: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = base.standardize(participants['improvement'].to_numpy(float))
    covariates = np.column_stack([base.standardize(participants['pre_score'].to_numpy(float)), base.standardize(np.log1p(participants['time_after_stroke'].to_numpy(float))), base.standardize((participants['sex'] == 'M').to_numpy(float))])
    n = len(y)
    df = n - (2 + covariates.shape[1])
    model_rows: list[dict[str, object]] = []
    for (gidx, gname) in enumerate(('G1', 'G2')):
        delta = post[:, :, gidx] - pre[:, :, gidx]
        pvals = []
        cached = []
        for roi0 in range(400):
            predictor = base.standardize(delta[:, roi0])
            full = np.column_stack([np.ones(n), predictor, covariates])
            (beta, se, t_values) = base.hc3_fit(y, full)
            p_value = float(2 * t_dist.sf(abs(t_values[1]), df))
            partial_r2 = float(t_values[1] ** 2 / (t_values[1] ** 2 + df))
            cached.append((predictor, beta[1], se[1], t_values[1], p_value, partial_r2))
            pvals.append(p_value)
        qvals = multipletests(pvals, method='fdr_bh')[1]
        crit = t_dist.ppf(0.975, df)
        for (roi0, (_, beta_value, se_value, t_value, p_value, partial_r2)) in enumerate(cached):
            model_rows.append({'gradient': gname, 'roi': roi0 + 1, 'roi_name': labels.iloc[roi0]['roi_name'], 'network': labels.iloc[roi0]['network'], 'hemisphere': labels.iloc[roi0]['hemisphere'], 'n': n, 'standardized_beta': beta_value, 'hc3_se': se_value, 'hc3_ci_low': beta_value - crit * se_value, 'hc3_ci_high': beta_value + crit * se_value, 'hc3_t': t_value, 'hc3_p_two_sided': p_value, 'fdr_q_400': qvals[roi0], 'partial_r_squared': partial_r2})
    models = pd.DataFrame(model_rows)
    selected = models.query("gradient == 'G2'").sort_values('hc3_p_two_sided').iloc[0].copy()
    roi0 = int(selected['roi']) - 1
    predictor = base.standardize(post[:, roi0, 1] - pre[:, roi0, 1])
    perm_result = base.freedman_lane_p(y, predictor, covariates, n_perm=N_PERM_SELECTED, seed_offset=172)
    selected_summary = pd.DataFrame([{**selected.to_dict(), 'studentized_freedman_lane_p_two_sided': perm_result[5], 'selection_note': 'Lowest HC3 p among 400 G2 ROIs in this cohort; exploratory, not independent validation', 'outcome': 'standardized Vasserman improvement (pre minus post)', 'predictor': 'standardized raw G2 change (post minus pre)', 'covariates': 'standardized baseline score; log1p(time after stroke); sex'}])
    reduced = np.column_stack([np.ones(n), covariates])
    y_resid = y - reduced @ (np.linalg.pinv(reduced) @ y)
    x_resid = predictor - reduced @ (np.linalg.pinv(reduced) @ predictor)
    subject_plot = participants[['participant_id', 'improvement', 'pre_score', 'post_score']].copy()
    subject_plot['selected_gradient'] = 'G2'
    subject_plot['selected_roi'] = roi0 + 1
    subject_plot['selected_roi_name'] = labels.iloc[roi0]['roi_name']
    subject_plot['raw_G2_delta_post_minus_pre'] = post[:, roi0, 1] - pre[:, roi0, 1]
    subject_plot['standardized_G2_delta'] = predictor
    subject_plot['adjusted_G2_delta_residual'] = x_resid
    subject_plot['adjusted_improvement_residual'] = y_resid
    return (models, selected_summary, subject_plot)

def make_network_centroid_source() -> pd.DataFrame:
    path = PARENT / 'tables' / 'legacy_aligned5_network_metrics_long.csv'
    frame = pd.read_csv(path)
    return frame.loc[frame['metric'] == 'mean'].copy()

def main() -> None:
    ensure_dirs()
    labels = base.load_labels()
    participants = pd.read_csv(base.PARTICIPANTS, sep='\t')
    participants['improvement'] = participants['pre_score'] - participants['post_score']
    (pre, post) = load_paired_gradients(participants)
    (roi_stats, roi_delta_long) = roi_paired_tests(pre, post, labels, participants['participant_id'].tolist())
    roi_sensitivity = roi_sensitivity_without_sub35(pre, post, labels, participants['participant_id'].tolist())
    network_counts = make_network_counts(roi_stats)
    network_effects = prepare_network_metric_effects()
    (component_long, component_tests) = prepare_component_variance()
    (clinical_models, selected_clinical, selected_subjects) = clinical_roi_models(pre, post, participants, labels)
    network_centroids = make_network_centroid_source()
    roi_stats.to_csv(TABLES / 'raw_aligned5_roi_paired_tests.csv', index=False, encoding='utf-8-sig')
    roi_sensitivity.to_csv(TABLES / 'raw_aligned5_roi_paired_tests_without_sub35.csv', index=False, encoding='utf-8-sig')
    network_effects.to_csv(TABLES / 'network_metric_effect_sizes.csv', index=False, encoding='utf-8-sig')
    component_tests.to_csv(TABLES / 'component_variance_paired_tests.csv', index=False, encoding='utf-8-sig')
    clinical_models.to_csv(TABLES / 'clinical_roi_models_G1_G2.csv', index=False, encoding='utf-8-sig')
    selected_clinical.to_csv(TABLES / 'selected_exploratory_G2_clinical_model.csv', index=False, encoding='utf-8-sig')
    roi_delta_long.to_csv(SOURCE / 'roi_delta_subject_level.csv', index=False, encoding='utf-8-sig')
    network_counts.to_csv(SOURCE / 'network_nominal_roi_counts.csv', index=False, encoding='utf-8-sig')
    network_centroids.to_csv(SOURCE / 'network_centroid_pre_post_long.csv', index=False, encoding='utf-8-sig')
    component_long.to_csv(SOURCE / 'component_variance_subject_level.csv', index=False, encoding='utf-8-sig')
    selected_subjects.to_csv(SOURCE / 'selected_G2_clinical_partial_regression_data.csv', index=False, encoding='utf-8-sig')
    labels.to_csv(SOURCE / 'Schaefer400_roi_labels.csv', index=False, encoding='utf-8-sig')
    summary = {'n': int(len(participants)), 'roi_results': {g: {'nominal_p_lt_0_05': int((f['p_two_sided'] < 0.05).sum()), 'fdr_q_lt_0_05': int((f['fdr_q_400'] < 0.05).sum()), 'minimum_p': float(f['p_two_sided'].min()), 'minimum_q': float(f['fdr_q_400'].min())} for (g, f) in roi_stats.groupby('gradient')}, 'network_metric_results': {'tests': int(len(network_effects)), 'nominal_p_lt_0_05': int(network_effects['nominal_p_lt_0_05'].sum()), 'fdr_q_lt_0_05': int(network_effects['fdr_q_lt_0_05'].sum())}, 'selected_exploratory_G2': selected_clinical.iloc[0].to_dict(), 'code_provenance': {'surface': _portable_path('project', 'code/2/2/a.py'), 'component_variance': _portable_path('project', 'code/2/2/b.py'), 'network_distribution': _portable_path('project', 'code/2/2/c.py'), 'roi_paired_test_and_network_counts': _portable_path('project', 'code/2/2/d.py'), 'network_metrics': [_portable_path('project', 'code/2/2/e.py'), _portable_path('project', 'code/indicator.py')]}}
    (HERE / 'analysis_manifest.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str), flush=True)
if __name__ == '__main__':
    main()
