from __future__ import annotations
import os as _portable_os
from pathlib import Path as _PortablePath

def _portable_path(kind, relative):
    project = _PortablePath(_portable_os.environ['PAPER_PROJECT_ROOT'])
    root = project if kind == 'project' else project.parent
    return str(root / relative)
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import run_external_validation_analysis as base
TABLES = HERE / 'tables'

def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    labels = base.load_labels()
    participants = pd.read_csv(base.PARTICIPANTS, sep='\t')
    participants['improvement'] = participants['pre_score'] - participants['post_score']
    participants['sex_male'] = (participants['sex'] == 'M').astype(int)
    participants['log_time_after_stroke'] = np.log1p(participants['time_after_stroke'].astype(float))
    rows: list[dict[str, object]] = []
    subject_rows: list[dict[str, object]] = []
    for participant in participants.itertuples(index=False):
        subject_id = participant.participant_id
        pre = np.asarray(np.load(base.locate_array(base.LEGACY_ROOT, 'pre', subject_id)), dtype=float)
        post = np.asarray(np.load(base.locate_array(base.LEGACY_ROOT, 'post', subject_id)), dtype=float)
        if pre.shape != (400, 5) or post.shape != (400, 5):
            raise RuntimeError(f'Unexpected aligned5 shape for {subject_id}')
        subject_values: dict[str, object] = {'participant_id': subject_id, 'pre_G1_sd': float(np.std(pre[:, 0], ddof=1)), 'post_G1_sd': float(np.std(post[:, 0], ddof=1)), 'delta_G1_sd': float(np.std(post[:, 0], ddof=1) - np.std(pre[:, 0], ddof=1)), 'pre_G2_sd': float(np.std(pre[:, 1], ddof=1)), 'post_G2_sd': float(np.std(post[:, 1], ddof=1)), 'delta_G2_sd': float(np.std(post[:, 1], ddof=1) - np.std(pre[:, 1], ddof=1))}
        for component in (1, 2):
            pre_metrics = base.component_network_metrics(pre[:, component - 1], labels)
            post_metrics = base.component_network_metrics(post[:, component - 1], labels)
            for network in base.NETWORK_ORDER:
                for metric in ('mean', 'Dispersion', 'betweenDisp', 'Segregation'):
                    pre_value = pre_metrics[network][metric]
                    post_value = post_metrics[network][metric]
                    subject_values[f'pre_G{component}_{network}_{metric}'] = pre_value
                    subject_values[f'post_G{component}_{network}_{metric}'] = post_value
                    subject_values[f'delta_G{component}_{network}_{metric}'] = post_value - pre_value
                    rows.append({'participant_id': subject_id, 'gradient': f'G{component}', 'network': network, 'metric': metric, 'pre': pre_value, 'post': post_value, 'delta_post_minus_pre': post_value - pre_value})
        subject_rows.append(subject_values)
    long = pd.DataFrame(rows)
    subject = participants.merge(pd.DataFrame(subject_rows), on='participant_id', validate='one_to_one')
    long.to_csv(TABLES / 'legacy_aligned5_network_metrics_long.csv', index=False, encoding='utf-8-sig')
    subject.to_csv(TABLES / 'legacy_aligned5_subject_metrics.csv', index=False, encoding='utf-8-sig')
    test_rows = []
    exact_metrics = long.loc[long['metric'].isin(['Dispersion', 'betweenDisp', 'Segregation'])]
    for ((gradient, network, metric), frame) in exact_metrics.groupby(['gradient', 'network', 'metric']):
        result = base.one_sample_summary(frame['delta_post_minus_pre'].to_numpy(float), f'{gradient}_{network}_{metric}')
        result.update({'gradient': gradient, 'network': network, 'metric': metric})
        test_rows.append(result)
    tests = pd.DataFrame(test_rows)
    tests['fdr_q_42_exact_indicator_tests'] = multipletests(tests['p_signflip_two_sided'], method='fdr_bh')[1]
    tests.to_csv(TABLES / 'legacy_aligned5_exact_network_tests.csv', index=False, encoding='utf-8-sig')
    global_tests = pd.DataFrame([base.one_sample_summary(subject['delta_G1_sd'].to_numpy(float), 'delta_G1_sd'), base.one_sample_summary(subject['delta_G2_sd'].to_numpy(float), 'delta_G2_sd')])
    global_tests['fdr_q_two_gradients'] = multipletests(global_tests['p_signflip_two_sided'], method='fdr_bh')[1]
    global_tests.to_csv(TABLES / 'legacy_aligned5_global_gradient_sd_tests.csv', index=False, encoding='utf-8-sig')
    clinical = ['pre_score', 'log_time_after_stroke', 'sex_male']
    between = [f'pre_G{component}_{network}_betweenDisp' for component in (1, 2) for network in base.NETWORK_ORDER]
    targeted = [f'pre_G2_{network}_{metric}' for network in ('Visual', 'SomMot') for metric in ('mean', 'Dispersion', 'betweenDisp', 'Segregation')]
    subject['pre_G2_Visual_SomMot_separation'] = np.abs(subject['pre_G2_Visual_mean'] - subject['pre_G2_SomMot_mean'])
    targeted.append('pre_G2_Visual_SomMot_separation')
    all_network_g2 = [f'pre_G2_{network}_{metric}' for network in base.NETWORK_ORDER for metric in ('mean', 'Dispersion', 'betweenDisp', 'Segregation')]
    all_network_g1g2 = [f'pre_G{component}_{network}_{metric}' for component in (1, 2) for network in base.NETWORK_ORDER for metric in ('mean', 'Dispersion', 'betweenDisp', 'Segregation')]
    blocks = {'clinical': clinical, 'clinical_plus_G1G2_betweenDisp': clinical + between, 'clinical_plus_targeted_G2_VIS_SomMot': clinical + targeted, 'clinical_plus_all_network_G2': clinical + all_network_g2, 'clinical_plus_all_network_G1G2': clinical + all_network_g1g2}
    (predictions, performance) = base.nested_cv_feature_blocks(subject, blocks)
    predictions.to_csv(TABLES / 'legacy_aligned5_ml_predictions.csv', index=False, encoding='utf-8-sig')
    performance.to_csv(TABLES / 'legacy_aligned5_ml_performance_by_repeat.csv', index=False, encoding='utf-8-sig')
    baseline_mae = performance.loc[performance['feature_block'] == 'clinical'].set_index('repeat')['mae']
    summary_rows = []
    for (block, frame) in performance.groupby('feature_block'):
        values = frame['mae'].to_numpy(float)
        aligned = frame.set_index('repeat')['mae']
        delta = baseline_mae.loc[aligned.index].to_numpy(float) - aligned.to_numpy(float)
        summary_rows.append({'feature_block': block, 'n_repeats': len(values), 'mean_mae': float(np.mean(values)), 'sd_mae': float(np.std(values, ddof=1)), 'mae_percentile_2.5': float(np.percentile(values, 2.5)), 'mae_percentile_97.5': float(np.percentile(values, 97.5)), 'mean_delta_mae_clinical_minus_block': float(np.mean(delta)), 'descriptive_repeat_fraction_better_than_clinical': float(np.mean(delta > 0))})
    pd.DataFrame(summary_rows).to_csv(TABLES / 'legacy_aligned5_ml_performance_summary.csv', index=False, encoding='utf-8-sig')
    print('legacy aligned5 validation complete', flush=True)
if __name__ == '__main__':
    main()
