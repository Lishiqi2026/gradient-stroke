from pathlib import Path
import os,sys
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
RUN=Path(__file__).resolve().parents[1];ROOT=RUN/'work/Gradient/lesion_analysis'
HERE=ROOT/'lesion_gradient_distance/figures/nature/lesion_final/bcdf_recomputed_provided_reference'
sys.path.insert(0,str(HERE))
import recompute_bcdf_with_provided_reference as m
roi,subjects,qc=m.recompute_metrics()
roi.to_csv(HERE/'recomputed_roi_metrics.csv',index=False);subjects.to_csv(HERE/'recomputed_subject_metrics.csv',index=False)
print('CORE FEATURES RECOMPUTED',len(subjects),flush=True)
