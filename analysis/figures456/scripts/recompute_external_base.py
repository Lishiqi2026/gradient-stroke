from pathlib import Path
import os,sys
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
RUN=Path(__file__).resolve().parents[1];WORK=RUN/'work';ROOT=WORK/'Gradient/lesion_analysis'
os.environ['JOBLIB_TEMP_FOLDER']=str(WORK/'.joblib_temp')
import numpy as np,pandas as pd
VALID=ROOT/'external_validation/ds003999_validation'
sys.path[:0]=[str(ROOT),str(VALID),str(VALID/'matched_gradient_figure'),str(ROOT/'external_validation/ds003999_confirmatory_extension')]
import run_legacy_aligned5_validation as legacy
import run_external_validation_analysis as base
import run_confirmatory_external_validation as confirm
import run_matched_gradient_analysis as matched
if '--resume-fc' not in sys.argv:legacy.main()
participants=pd.read_csv(base.PARTICIPANTS,sep='\t');labels=base.load_labels()
reference=np.load(confirm.REFERENCES/'healthy_main_reference_locked_400x5.npy')
confirm.TABLES.mkdir(parents=True,exist_ok=True)
qc=[];roi=[]
for participant in participants.itertuples():
    sid=participant.participant_id;record=dict(participant_id=sid);raws=[]
    for phase in ['pre','post']:
        fc=np.load(base.locate_array(confirm.EXTERNAL_FC,phase,sid))
        raw,eigen=confirm.fit_provided(fc);raws.append(raw)
        record.update({f'{phase}_lambda_G{i+1}':float(value) for i,value in enumerate(eigen)})
    pre,post,_=confirm.separate_orthogonal_alignment(*raws,reference)
    metrics=confirm.normalization(pre,post,reference)
    frame=labels[['roi','roi_name','network']].copy();frame['participant_id']=sid
    for g in ['G1','G2']:
        i=int(g[-1])-1
        frame[g+'_pre_z']=metrics['pre_z'][:,i];frame[g+'_post_z']=metrics['post_z'][:,i];frame[g+'_normalization']=metrics['normalization_'+g]
    roi.append(frame);qc.append(record)
    print('EXTERNAL RAW FC',sid,flush=True)
(VALID/'qa').mkdir(exist_ok=True)
pd.DataFrame(qc).to_csv(VALID/'qa/provided_code_gradient_alignment_qc.csv',index=False)
pd.concat(roi,ignore_index=True).to_csv(confirm.TABLES/'external_primary_roi_normalization_long.csv',index=False)
matched.main()
print('EXTERNAL BASE STATISTICS COMPLETE',flush=True)
