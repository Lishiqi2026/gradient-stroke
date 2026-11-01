from __future__ import annotations
import os as _portable_os
from pathlib import Path as _PortablePath

def _portable_path(kind, relative):
    project = _PortablePath(_portable_os.environ['PAPER_PROJECT_ROOT'])
    root = project if kind == 'project' else project.parent
    return str(root / relative)
import argparse
import json
import os
from pathlib import Path
OUTPUT_ROOT = Path(_portable_path('project', 'ml_analysis/outputs/one_month_followup_sensitivity'))
os.environ.setdefault('JOBLIB_TEMP_FOLDER', str(OUTPUT_ROOT / '.joblib_temp'))
Path(os.environ['JOBLIB_TEMP_FOLDER']).mkdir(parents=True, exist_ok=True)
import numpy as np
import pandas as pd
import run_five_outcome_lesion_informed_ml as primary
OUTCOMES = ('MMSE', 'VFT')

def load_followup_prepared() -> tuple[primary.PreparedData, dict]:
    (prepared, primary_metadata) = primary.load_prepared_data()
    followup = pd.read_excel(primary.FOLLOWUP_PATH)
    if followup.shape[1] < 9:
        raise ValueError(f'Unexpected follow-up table shape: {followup.shape}')
    followup_table = pd.DataFrame({'subject_id': followup.iloc[:, 0].astype(str), 'MMSE_pre_followup': pd.to_numeric(followup.iloc[:, 3]), 'MMSE_immediate_followup': pd.to_numeric(followup.iloc[:, 4]), 'MMSE_one_month': pd.to_numeric(followup.iloc[:, 5]), 'VFT_pre_followup': pd.to_numeric(followup.iloc[:, 6]), 'VFT_immediate_followup': pd.to_numeric(followup.iloc[:, 7]), 'VFT_one_month': pd.to_numeric(followup.iloc[:, 8])})
    table = prepared.table.merge(followup_table, on='subject_id', how='inner', validate='one_to_one')
    for outcome in OUTCOMES:
        if not np.array_equal(table[f'{outcome}_pre'].to_numpy(dtype=float), table[f'{outcome}_pre_followup'].to_numpy(dtype=float)):
            raise ValueError(f'{outcome} baseline mismatch between clinical and follow-up tables')
        if not np.array_equal(table[f'{outcome}_post'].to_numpy(dtype=float), table[f'{outcome}_immediate_followup'].to_numpy(dtype=float)):
            raise ValueError(f'{outcome} immediate-post mismatch between clinical and follow-up tables')
        table[f'{outcome}_immediate_post'] = table[f'{outcome}_post']
        table[f'{outcome}_post'] = table[f'{outcome}_one_month']
    table = table.sort_values('subject_id').reset_index(drop=True)
    metadata = {'analysis_date': '2026-08-11', 'task': 'secondary prediction of one-month MMSE and VFT', 'n': int(len(table)), 'group_counts': table['group'].value_counts().sort_index().to_dict(), 'clinical_only_block_included': False, 'candidate_blocks': primary.BLOCK_LABELS, 'source_followup': str(primary.FOLLOWUP_PATH), 'cross_table_validation': 'baseline and immediate-post MMSE/VFT exactly match the main clinical table', 'primary_feature_metadata': primary_metadata, 'limitations': ['Only 23 motion-QC lesion subjects have one-month MMSE/VFT; results are highly exploratory.', 'No one-month NIHSS, ADL or HAMD are available in the follow-up workbook.', 'Every candidate block includes the same compact clinical base; no clinical-only block is run.']}
    return (primary.PreparedData(table, prepared.block_feature_columns), metadata)

def save_splits(prepared: primary.PreparedData, splits: list[dict]) -> None:
    ids = prepared.table['subject_id'].to_numpy()
    records = [{'repeat': split['repeat'], 'fold': split['fold'], 'train_subject_ids': ids[split['train']].tolist(), 'test_subject_ids': ids[split['test']].tolist()} for split in splits]
    (OUTPUT_ROOT / 'nested_cv_splits.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='One-month MMSE/VFT sensitivity analysis.')
    parser.add_argument('--repeats', type=int, default=50)
    parser.add_argument('--jobs', type=int, default=-1)
    parser.add_argument('--quick', action='store_true')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    if args.quick:
        args.repeats = 1
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (prepared, manifest) = load_followup_prepared()
    splits = primary.make_outer_splits(prepared.table['group_tms'].to_numpy(dtype=int), args.repeats)
    save_splits(prepared, splits)
    jobs_to_run = [(algorithm, outcome, block) for algorithm in ('ridge', 'elasticnet') for outcome in OUTCOMES for block in primary.BLOCK_LABELS]
    predictions_all = []
    tuning_all = []
    repeat_all = []
    coefficients_all = []
    for (index, (algorithm, outcome, block)) in enumerate(jobs_to_run, start=1):
        print(f'[{index}/{len(jobs_to_run)}]', end=' ', flush=True)
        (predictions, tuning, repeat_metrics, coefficients) = primary.run_condition(prepared, outcome, block, algorithm, splits, args.jobs)
        predictions_all.append(predictions)
        tuning_all.append(tuning)
        repeat_all.append(repeat_metrics)
        coefficients_all.append(coefficients)
    predictions = pd.concat(predictions_all, ignore_index=True)
    tuning = pd.concat(tuning_all, ignore_index=True)
    repeat_metrics = pd.concat(repeat_all, ignore_index=True)
    coefficients = pd.concat(coefficients_all, ignore_index=True)
    performance = primary.summarize_repeats(repeat_metrics)
    aggregated = primary.aggregate_predictions(predictions)
    coefficient_summary = primary.coefficient_stability(coefficients)
    predictions.to_csv(OUTPUT_ROOT / 'repeated_oof_predictions.csv', index=False, encoding='utf-8-sig')
    aggregated.to_csv(OUTPUT_ROOT / 'participant_mean_oof_predictions.csv', index=False, encoding='utf-8-sig')
    tuning.to_csv(OUTPUT_ROOT / 'outer_fold_tuning.csv', index=False, encoding='utf-8-sig')
    repeat_metrics.to_csv(OUTPUT_ROOT / 'repeat_level_performance.csv', index=False, encoding='utf-8-sig')
    performance.to_csv(OUTPUT_ROOT / 'repeat_performance_summary.csv', index=False, encoding='utf-8-sig')
    coefficients.to_csv(OUTPUT_ROOT / 'outer_fold_coefficients.csv', index=False, encoding='utf-8-sig')
    coefficient_summary.to_csv(OUTPUT_ROOT / 'coefficient_stability_summary.csv', index=False, encoding='utf-8-sig')
    manifest.update({'outer_cv': f'treatment-stratified 5-fold repeated {args.repeats} times', 'inner_cv': 'treatment-stratified 4-fold tuning inside each outer training fold', 'algorithms': ['ridge', 'elasticnet'], 'random_seed': primary.SEED})
    (OUTPUT_ROOT / 'analysis_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Completed. Outputs: {OUTPUT_ROOT}', flush=True)
if __name__ == '__main__':
    main()
