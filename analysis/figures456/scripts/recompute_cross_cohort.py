from pathlib import Path
import os,sys,json
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
RUN=Path(__file__).resolve().parents[1];WORK=RUN/'work';ROOT=WORK/'Gradient/lesion_analysis'
HERE=ROOT/'external_validation/final/main_tms_vs_external_adjusted_beta_surface_20260829'
import pandas as pd
sys.path.insert(0,str(HERE))
import make_main_tms_vs_external_adjusted_beta_surfaces as m
clinical=pd.read_csv(WORK/'clinical_corrected.csv',dtype={'subject_id':str})
qc=pd.read_csv(ROOT/'lesion_gradient_distance/fc_consistent/tables/longitudinal_fc_qc.csv',dtype={'subject_id':str})
eligible=qc.groupby('subject_id').primary_motion_eligible.all()
table=clinical.loc[clinical.subject_id.isin(eligible.index[eligible])].copy()
assert 'PRIVATE_SUBJECT_008' not in set(table.subject_id)
assert table.loc[table.subject_id.eq('PRIVATE_SUBJECT_038'),'group'].tolist()==['Control']
out=RUN/'cross_cohort_recomputed';out.mkdir(exist_ok=True)
table.to_csv(out/'functional_cohort.csv',index=False)
m.MAIN_UNIFIED=out/'functional_cohort.csv'
main=m.build_main_tms_roi();external=m.build_external_roi()
main.to_csv(out/'main_tms_roi.csv',index=False);external.to_csv(out/'external_roi.csv',index=False)
cross=m.fit_cross_cohort(main,external);cross.to_csv(out/'cross_cohort_roi_models.csv',index=False)
(out/'cohort_definition.json').write_text(json.dumps(dict(group_counts=table.group.value_counts().to_dict(),uses_voxelwise_lesion_mask=False,exclusion='PRIVATE_SUBJECT_008',group_override={'PRIVATE_SUBJECT_038':'Control'},baseline='Same supplied all-ROI orthogonal alignment to locked reference'),indent=2),encoding='utf-8')
print('CROSS COHORT COMPLETE',table.group.value_counts().to_dict(),flush=True)
