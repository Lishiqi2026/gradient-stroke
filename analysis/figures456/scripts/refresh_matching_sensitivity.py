from pathlib import Path
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];AN=ROOT/'work/Gradient/lesion_analysis';BCDF=AN/'lesion_gradient_distance/figures/nature/lesion_final/bcdf_recomputed_provided_reference'
sys.path.insert(0,str(BCDF));import recompute_bcdf_with_provided_reference as original
subject=pd.read_csv(BCDF/'recomputed_subject_metrics.csv',dtype={'subject_id':str});pairs_dir=AN/'lesion_matched_clinical_mediation/tables'
pairs=pd.read_csv(pairs_dir/'optimal_pairs_all_baseline.csv',dtype={'control_subject_id':str,'tms_subject_id':str})
valid=set(subject.loc[subject.xtract_projection_eligible.astype(bool),'subject_id']);available=pairs[pairs.control_subject_id.isin(valid)&pairs.tms_subject_id.isin(valid)]
available.to_csv(pairs_dir/'computable_pairs_all_available.csv',index=False);original.MATCHED_PAIRS_PATH=pairs_dir/'computable_pairs_all_available.csv'
tests,values=original.run_matched_tests(subject);tests.to_csv(BCDF/'recomputed_matched_tests.csv',index=False);values.to_csv(BCDF/'recomputed_matched_pair_values.csv',index=False)
print('MATCHED SENSITIVITY UPDATED',tests.groupby('match_set').n_pairs.first().to_dict(),flush=True)
