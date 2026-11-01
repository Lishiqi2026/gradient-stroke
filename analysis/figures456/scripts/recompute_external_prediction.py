from pathlib import Path
import os,sys,time,hashlib
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
RUN=Path(__file__).resolve().parents[1];WORK=RUN/'work'
HERE=WORK/'Gradient/lesion_analysis/external_validation/final/folder-forai'
os.environ['JOBLIB_TEMP_FOLDER']=str(WORK/'.joblib_temp')
import pandas as pd
sys.path.insert(0,str(HERE))
module_path=HERE/'run_expanded_ml_forest.py'
source=module_path.read_text(encoding='utf-8')
source=source.replace('for repeat in range(N_REPEATS):',"for repeat in globals().get('REPEAT_INDICES', range(N_REPEATS)):")
module_path.write_text(source,encoding='utf-8')
import run_expanded_ml_forest as m
frame=m.load_frame();blocks=m.define_feature_blocks(frame)
cohorts={'primary_n29':frame.copy(),'alignment_sensitivity_n28':frame.loc[frame.participant_id.ne('sub-35')].copy()}
signature=hashlib.sha256((frame.to_csv(index=False)+str(m.SEED)+str(m.N_REPEATS)+Path(m.__file__).read_text(encoding='utf-8')).encode()).hexdigest()[:16]
check=RUN/('external_prediction_checkpoints_'+signature);check.mkdir(exist_ok=True)
parts=[[],[],[]]
algorithms=tuple(m.ALGORITHMS);cohort_order=tuple(m.COHORT_ORDER)
for algorithm in algorithms:
    for cohort in cohort_order:
        for repeat in range(m.N_REPEATS):
            files=[check/f'{algorithm}_{cohort}_{repeat}_{kind}.csv' for kind in ['predictions','performance','tuning']]
            if all(f.exists() for f in files):
                result=[pd.read_csv(f) for f in files]
            else:
                m.ALGORITHMS=(algorithm,);m.COHORT_ORDER=(cohort,);m.REPEAT_INDICES=(repeat,)
                result=m.run_nested_cv(cohorts,blocks)
                for d,f in zip(result,files):d.to_csv(f,index=False)
            for dest,d in zip(parts,result):dest.append(d)
pred,performance,tuning=[pd.concat(p,ignore_index=True) for p in parts]
paired,summary=m.summarize_performance(performance,blocks)
for name,d in [('ml_oof_predictions.csv',pred),('ml_performance_by_repeat.csv',paired),('ml_performance_summary.csv',summary),('ml_outer_fold_tuning.csv',tuning),('ml_feature_dictionary.csv',m.write_feature_dictionary(blocks))]:d.to_csv(HERE/name,index=False)
print('EXTERNAL PREDICTION RECOMPUTED',len(pred),len(performance),flush=True)
