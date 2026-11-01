from pathlib import Path
import sys
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];VALID=ROOT/'work/Gradient/lesion_analysis/external_validation/ds003999_validation'
sys.path.insert(0,str(VALID))
import run_external_validation_analysis as original
data=pd.read_csv(original.PARTICIPANTS,sep='\t');delta=data.pre_score-data.post_score;test=original.ttest_rel(data.pre_score,data.post_score)
out=VALID/'matched_gradient_figure/source_data';out.mkdir(parents=True,exist_ok=True)
data[['participant_id','pre_score','post_score']].to_csv(out/'clinical_score_pre_post.csv',index=False)
summary=pd.DataFrame([dict(n=len(data),pre_mean=data.pre_score.mean(),pre_sd=data.pre_score.std(ddof=1),post_mean=data.post_score.mean(),post_sd=data.post_score.std(ddof=1),improvement_definition='pre_score - post_score; positive indicates lower post-treatment impairment',improvement_mean=delta.mean(),improvement_sd=delta.std(ddof=1),paired_t=float(test.statistic),paired_p_two_sided=float(test.pvalue),improvement_signflip_p_two_sided=original.signflip_p(delta.to_numpy(float)))])
(VALID/'tables').mkdir(exist_ok=True);summary.to_csv(VALID/'tables/clinical_change_summary.csv',index=False)
print('EXTERNAL CLINICAL SOURCES RECOMPUTED',len(data),flush=True)
