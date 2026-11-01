import os
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
os.environ['JOBLIB_TEMP_FOLDER']=str(BASE/'work/.joblib_temp')
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='1'
import sys,json,time,hashlib
import pandas as pd
ROOT=BASE/'work/Gradient/ml_analysis'
sys.path.insert(0,str(ROOT))
import run_five_outcome_lesion_informed_ml as m

out=ROOT/'outputs/five_outcome_lesion_informed'
while not all((BASE/'work/Gradient/lesion_analysis/lesion_gradient_distance/figures/nature/lesion_final/bcdf_recomputed_provided_reference'/name).exists() for name in ['recomputed_roi_metrics.csv','recomputed_subject_metrics.csv']):
    print('WAITING new lesion features',flush=True);time.sleep(20)
m.load_clinical=lambda:pd.read_csv(BASE/'work/clinical_corrected.csv',dtype={'subject_id':str})
m.followup_metadata=lambda ids:{'used_in_primary_five_outcome_models':False}
prepared,manifest=m.load_prepared_data();data=prepared.table
assert 'PRIVATE_SUBJECT_008' not in set(data.subject_id)
assert data.loc[data.subject_id.eq('PRIVATE_SUBJECT_038'),'group'].tolist()==['Control']
out.mkdir(parents=True,exist_ok=True)
data.to_csv(out/'five_outcome_baseline_feature_table.csv',index=False)
(out/'feature_block_definitions.json').write_text(json.dumps(prepared.block_feature_columns,indent=2),encoding='utf-8')
manifest.update(analysis_date='2026-10-28',motion_qc='At least 100 retained frames at each session',excluded_subjects=['PRIVATE_SUBJECT_008'],group_overrides={'PRIVATE_SUBJECT_038':'Control'})
(out/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
splits=m.make_outer_splits(data.group_tms.to_numpy(int),50)
m.OUTPUT_ROOT=out
m.save_splits(prepared,splits)
signature=hashlib.sha256((data.to_csv(index=False)+json.dumps(prepared.block_feature_columns,sort_keys=True)+str(m.SEED)+'50'+Path(m.__file__).read_text(encoding='utf-8')).encode()).hexdigest()[:16]
check=out/('checkpoints_'+signature);check.mkdir(exist_ok=True)
all_parts=[[],[],[],[]]
conditions=[(a,o,b) for a in ('elasticnet','ridge') for o in m.OUTCOMES for b in m.BLOCK_LABELS]
branch=None
if '--algorithm' in sys.argv:
    branch=sys.argv[sys.argv.index('--algorithm')+1]
    if branch not in ['elasticnet','ridge']:raise ValueError('Unknown algorithm')
    conditions=[condition for condition in conditions if condition[0]==branch]
for i,(algorithm,outcome,block) in enumerate(conditions,1):
    tag=f'{algorithm}_{outcome}_{block}'
    files=[check/f'{tag}_{suffix}.csv' for suffix in ['predictions','tuning','metrics','coefficients']]
    start=time.time()
    if all(p.exists() for p in files):
        parts=[pd.read_csv(p,dtype={'subject_id':str}) for p in files]
    else:
        parts=m.run_condition(prepared,outcome,block,algorithm,splits,8)
        for part,f in zip(parts,files):
            temporary=f.with_suffix(f.suffix+'.tmp');part.to_csv(temporary,index=False);temporary.replace(f)
    for dest,part in zip(all_parts,parts):dest.append(part)
    print(f'COMPLETE {i}/70 {tag} {time.time()-start:.1f}s',flush=True)
if branch:
    print('ALGORITHM BRANCH COMPLETE',branch,flush=True)
    sys.exit(0)
predictions,tuning,repeats,coefficients=[pd.concat(x,ignore_index=True) for x in all_parts]
performance=m.summarize_repeats(repeats)
outputs={
'repeated_oof_predictions.csv':predictions,
'outer_fold_tuning.csv':tuning,
'repeat_level_performance.csv':repeats,
'outer_fold_coefficients.csv':coefficients,
'repeat_performance_summary.csv':performance,
'selected_best_pairwise_summary.csv':m.paired_best_summary(repeats,performance),
'participant_mean_oof_predictions.csv':m.aggregate_predictions(predictions),
'coefficient_stability_summary.csv':m.coefficient_stability(coefficients)}
for name,table in outputs.items():table.to_csv(out/name,index=False,encoding='utf-8-sig')
manifest=json.loads((out/'analysis_manifest.json').read_text(encoding='utf-8'))
manifest.update({'correction':'PRIVATE_SUBJECT_038: TMS -> Control','group_counts':data.group.value_counts().to_dict(),'recomputed':True})
(out/'analysis_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print('ALL ML COMPLETE',flush=True)
