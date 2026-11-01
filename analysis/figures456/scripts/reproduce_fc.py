from pathlib import Path
import os,sys,json
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
RUN=Path(__file__).resolve().parents[1];WORK=RUN/'work';ROOT=WORK/'Gradient/lesion_analysis'
import numpy as np,pandas as pd
sys.path.insert(0,str(ROOT))
import build_longitudinal_fc_for_lesion_gradient as original
subjects=pd.read_csv(WORK/'subjects.csv',dtype={'subject_id':str});out=ROOT/'lesion_gradient_distance/fc_consistent'
for name in ['tables','fc_matrices_r','fc_matrices_z']:(out/name).mkdir(parents=True,exist_ok=True)
rows=[]
for row in subjects.itertuples():
    for phase in ['baseline','post']:
        ts=np.loadtxt(out/'timeseries_fsl'/f'{row.subject_id}_{phase}_schaefer400.tsv')
        fd=np.loadtxt(out/'fd'/f'{row.subject_id}_{phase}_PowerFD.txt').reshape(-1)
        if len(fd)!=len(ts) or ts.shape[1]!=400:raise ValueError('Input time-series/FD mismatch')
        keep,high=original.scrub_keep_mask(fd)
        if keep.sum()>=original.ABSOLUTE_MINIMUM_FRAMES:
            r,z=original.correlation_matrices(ts[keep]);np.save(out/'fc_matrices_r'/f'{row.subject_id}_{phase}_fc_r.npy',r);np.save(out/'fc_matrices_z'/f'{row.subject_id}_{phase}_fc_z.npy',z)
        rows.append(dict(subject_id=row.subject_id,group=row.group,session=phase,mean_fd=float(fd.mean()),median_fd=float(np.median(fd)),max_fd=float(fd.max()),high_motion_frames=int(high.sum()),scrubbed_frames=int((~keep).sum()),retained_frames=int(keep.sum()),retained_fraction=float(keep.mean()),primary_motion_eligible=bool(keep.sum()>=100)))
pd.DataFrame(rows).to_csv(out/'tables/longitudinal_fc_qc.csv',index=False)
print('FC REPRODUCED',len(subjects),'subjects',flush=True)
