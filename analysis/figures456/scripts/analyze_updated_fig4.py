from pathlib import Path
import os,sys,json,time
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
RUN=Path(__file__).resolve().parents[1];WORK=RUN/'work';PROJECT=WORK/'Gradient';ROOT=PROJECT/'lesion_analysis'
os.environ['JOBLIB_TEMP_FOLDER']=str(WORK/'.joblib_temp')
import numpy as np,pandas as pd
from statsmodels.stats.multitest import multipletests
DIST=ROOT/'lesion_gradient_distance';BCDF=DIST/'figures/nature/lesion_final/bcdf_recomputed_provided_reference';SUB=DIST/'figures/nature/lesion_final/forSubmit_lesion';SOURCE=SUB/'source_data'
for p in [SOURCE,BCDF,ROOT/'lesion_matched_clinical_mediation/tables',ROOT/'lesion_gradient_score_triangle_exploratory/tables']:p.mkdir(parents=True,exist_ok=True)
sys.path[:0]=[str(ROOT),str(PROJECT/'ml_analysis'),str(BCDF)]
if '--wait' in sys.argv:
    while not (DIST/'tables/subject_level_distance_metrics.csv').exists():time.sleep(15)

# Replace hard-coded sample labels; statistical families and tests are unchanged.
for p in [BCDF/'recompute_bcdf_with_provided_reference.py',ROOT/'run_p3_triangle_exploratory.py']:
    text=p.read_text(encoding='utf-8')
    for a,b in [('all_11_pairs','all_pairs'),('tight_9_pairs','tight_pairs'),('motion_qc_10_pairs','motion_qc_pairs'),('motion_qc_tight_8_pairs','motion_qc_tight_pairs'),('all_38_sensitivity','all_available_sensitivity')]:text=text.replace(a,b)
    text=text.replace('if len(data) != 37:','if len(data) < 3:')
    p.write_text(text,encoding='utf-8')
import recompute_bcdf_with_provided_reference as b
import run_lesion_matched_clinical_mediation as matching
import make_provided_code_g1_g2_normalization_surfaces as roi_model
import run_p3_triangle_exploratory as triangle

clinical=pd.read_csv(WORK/'clinical_corrected.csv',dtype={'subject_id':str})
clinical['location_normalized']=clinical.location_text.map(matching.normalize_location)
clinical['coarse_location']=clinical.location_text.map(matching.coarse_location)
clinical['clinical_side']=clinical.clinical_side_text.map(matching.normalized_side)
assert 'PRIVATE_SUBJECT_008' not in set(clinical.subject_id)
assert clinical.loc[clinical.subject_id.eq('PRIVATE_SUBJECT_038'),'group'].tolist()==['Control']
exact,exact_stats=matching.exact_location_subgroup_analysis(clinical)
exact.to_csv(SOURCE/'exact_clinical_subgroup_subjects.csv',index=False)
exact_stats.to_csv(SOURCE/'lesion_matched_clinical_statistics.csv',index=False)
exact_long=[]
for outcome in ['NIHSS','ADL']:
    for row in exact.itertuples():
        pre=getattr(row,outcome+'_pre');post=getattr(row,outcome+'_post')
        exact_long.append(dict(subject_id=row.subject_id,group=row.group,outcome=outcome,pre=pre,post=post,improvement=pre-post if outcome=='NIHSS' else post-pre))
pd.DataFrame(exact_long).to_csv(SOURCE/'lesion_matched_clinical_improvements.csv',index=False)

print('RECOMPUTE individual distance/normalization metrics',flush=True)
roi,subject,qc=b.recompute_metrics()
assert 'PRIVATE_SUBJECT_008' not in set(subject.subject_id)
roi.to_csv(BCDF/'recomputed_roi_metrics.csv',index=False)
subject.to_csv(BCDF/'recomputed_subject_metrics.csv',index=False)
primary=subject.loc[subject.primary_motion_eligible.astype(bool)].copy()
print('PRIMARY GROUPS',primary.groupby('group').size().to_dict(),flush=True)
models=b.run_group_models(subject);models.to_csv(BCDF/'recomputed_group_models.csv',index=False)
projected=primary.loc[primary.xtract_projection_eligible.astype(bool)].copy()
bins,bin_summary=b.residualized_distance_bins(roi,set(projected.subject_id));bins.to_csv(BCDF/'panel_c_subject_distance_bins.csv',index=False);bin_summary.to_csv(BCDF/'panel_c_group_distance_bin_summary.csv',index=False)

res=ROOT/'results_abcd/tables'
lesion=pd.read_csv(res/'B_participant_lesion_metrics.csv',dtype={'subject_id':str})
anatomy=pd.read_csv(res/'B2_anatomical_compartment_metrics.csv',dtype={'subject_id':str})
spatial=pd.read_csv(res/'E_spatial_lesion_phenotypes.csv',dtype={'subject_id':str})
systems=pd.read_csv(res/'F_xtract_system_probability_burden.csv',dtype={'subject_id':str})
joint=clinical.merge(lesion[['subject_id','log1p_lesion_volume','mask_lateralization_index']],on='subject_id',validate='one_to_one')
cols=['outside_mni_brain_fraction_pct','brainstem_fraction_pct','deep_subcortical_gray_fraction_pct','cerebral_white_matter_fraction_pct','cerebral_cortex_fraction_pct']
joint=joint.merge(anatomy[['subject_id',*cols]],on='subject_id',validate='one_to_one')
joint=joint.merge(spatial[['subject_id','absolute_centroid_x_mm','centroid_y_mm','centroid_z_mm','maximum_extent_mm']],on='subject_id',validate='one_to_one')
for col in cols[1:]:joint['log1p_'+col]=np.log1p(joint[col])
syswide=systems.pivot(index='subject_id',columns='system',values='expected_overlap_ml')
for name in ['motor_projection','thalamic_sensory_radiations','middle_cerebellar_peduncle']:
    joint['log1p_xtract_'+name+'_ml']=np.log1p(joint.subject_id.map(syswide[name]).fillna(0))
features=['log1p_lesion_volume','mask_lateralization_index',*['log1p_'+c for c in cols[1:]],'absolute_centroid_x_mm','centroid_y_mm','centroid_z_mm','maximum_extent_mm',*['log1p_xtract_'+name+'_ml' for name in ['motor_projection','thalamic_sensory_radiations','middle_cerebellar_peduncle']],'age','sex_male','esrs']
weights={c:1. for c in features};weights.update(age=.35,sex_male=.25,esrs=.35)
pairs=matching.optimal_pairs(joint,features,weights)
pairdir=ROOT/'lesion_matched_clinical_mediation/tables'
joint.to_csv(pairdir/'analysis_cohort_all_baseline.csv',index=False)
pairs.to_csv(pairdir/'optimal_pairs_all_baseline.csv',index=False)
eligible=pairs.control_subject_id.isin(projected.subject_id)&pairs.tms_subject_id.isin(projected.subject_id)
active=pairs.loc[eligible].copy();active.to_csv(pairdir/'motion_eligible_pairs.csv',index=False)
available_projected=subject.loc[subject.xtract_projection_eligible.astype(bool)]
computable=pairs.loc[pairs.control_subject_id.isin(available_projected.subject_id)&pairs.tms_subject_id.isin(available_projected.subject_id)].copy()
computable.to_csv(pairdir/'computable_pairs_all_available.csv',index=False)
b.MATCHED_PAIRS_PATH=pairdir/'computable_pairs_all_available.csv'
matched_tests,matched_values=b.run_matched_tests(subject)
matched_tests.to_csv(BCDF/'recomputed_matched_tests.csv',index=False);matched_values.to_csv(BCDF/'recomputed_matched_pair_values.csv',index=False)
print('MATCHED',len(pairs),'initial',len(active),'eligible',flush=True)

print('ROI models: original HC3 + 5000 Freedman-Lane permutations',flush=True)
raw=pd.read_csv(DIST/'provided_code_g1_g2_recalculation/provided_code_roi_normalization_long.csv',dtype={'subject_id':str})
raw=raw.loc[raw.primary_motion_eligible.astype(bool)].copy()
maprows=[];modelrows=[]
ref=np.load(b.REFERENCE_PATH)
for n,frame in raw.groupby('roi',sort=True):
    intact=frame.loc[frame.intact_roi.astype(bool)]
    rec=dict(roi=int(n),roi_name=frame.roi_name.iloc[0],network=frame.network.iloc[0],control_n_intact=int(intact.group.eq('Control').sum()),tms_n_intact=int(intact.group.eq('TMS').sum()))
    for g in ['G1','G2']:
        for group in ['Control','TMS']:rec[f'{group}_mean_{g}_normalization']=float(intact.loc[intact.group.eq(group),'normalization_'+g].mean())
        rec[f'TMS_minus_Control_{g}_normalization']=rec[f'TMS_mean_{g}_normalization']-rec[f'Control_mean_{g}_normalization']
        fit=roi_model.fit_roi_group_model(frame,outcome='normalization_'+g,baseline_deviation='baseline_deviation_'+g,roi=int(n));fit['gradient']=g;modelrows.append(fit)
        rec[f'{g}_reference_z']=float(ref[int(n)-1,int(g[-1])-1])
    maprows.append(rec)
    if n%50==0:print('ROI',n,flush=True)
maps=pd.DataFrame(maprows);fits=pd.DataFrame(modelrows);fits['fdr_q_within_gradient']=np.nan
for g in ['G1','G2']:
    h=fits.gradient.eq(g);fits.loc[h,'fdr_q_within_gradient']=multipletests(fits.loc[h,'p_freedman_lane'],method='fdr_bh')[1];lookup=fits.loc[h].set_index('roi')
    for dst,src in [('adjusted_TMS_beta_','adjusted_group_tms_beta'),('permutation_p_','p_freedman_lane'),('fdr_q_','fdr_q_within_gradient')]:maps[dst+g]=maps.roi.map(lookup[src])
maps['G2_FDR_significant']=maps.fdr_q_G2.lt(.05);maps['G2_adjusted_beta_FDR_masked']=np.where(maps.G2_FDR_significant,maps.adjusted_TMS_beta_G2,0)
maps.to_csv(SOURCE/'G1_G2_surface_source_data.csv',index=False);fits.to_csv(SOURCE/'ROI_adjusted_models.csv',index=False)
print('G2 FDR ROIs',int(maps.G2_FDR_significant.sum()),flush=True)

sys.path.insert(0,str(SUB/'figures'))
import run_g2_effect_spintest as network_spin
_,coordinates=network_spin.load_inputs()
network_codes=np.array([network_spin.NETWORK_ORDER.index(n) for n in maps.network])
null_labels=network_spin.build_null_labels(coordinates,network_codes,network_spin.N_PERMUTATIONS,network_spin.SEED)
spin_tables=[];spin_nulls={}
for direction,sign in [('positive',1),('negative',-1)]:
    selected=maps.G2_FDR_significant.to_numpy(bool)&(maps.adjusted_TMS_beta_G2.to_numpy(float)*sign>0)
    if selected.any():
        stats,null=network_spin.test_direction(selected,null_labels,network_codes,direction)
        stats['status']='tested'
    else:
        stats=pd.DataFrame([dict(direction=direction,network=n,network_abbreviation=abbr,n_selected_rois=0,actual_count=0,actual_enrichment_ratio=np.nan,null_mean=np.nan,null_sd=np.nan,spin_p_upper=np.nan,spin_fdr_q=np.nan,n_permutations=network_spin.N_PERMUTATIONS,status='not_defined_no_selected_ROIs') for n,abbr in zip(network_spin.NETWORK_ORDER,network_spin.NETWORK_ABBR)])
        null=np.full((network_spin.N_PERMUTATIONS,7),np.nan)
    spin_tables.append(stats);spin_nulls[direction]=null
pd.concat(spin_tables,ignore_index=True).to_csv(SOURCE/'G2_network_spin_statistics.csv',index=False)
np.savez_compressed(SOURCE/'G2_network_spin_nulls.npz',**spin_nulls)

print('MMSE interaction: original 13-test families',flush=True)
data=projected.merge(clinical[['subject_id',*[f'{o}_{phase}' for o in matching.OUTCOMES for phase in ['pre','post']]]],on='subject_id',validate='one_to_one')
for outcome in matching.OUTCOMES:data[outcome+'_improvement']=(data[outcome+'_pre']-data[outcome+'_post']) if outcome in ['NIHSS','HAMD'] else (data[outcome+'_post']-data[outcome+'_pre'])
tabledir=ROOT/'lesion_gradient_score_triangle_exploratory/tables'
inputfile=tabledir/'analysis_dataset_unstandardized.csv';data.to_csv(inputfile,index=False);triangle.INPUT_PATH=inputfile
frame=triangle.load_primary();rng=np.random.default_rng(triangle.SEED)
associations=triangle.association_analysis(frame,5000,rng);interactions=triangle.moderation_analysis(frame,5000,rng);slopes=triangle.lesion_gradient_simple_slopes(frame)
for name,d in [('analysis_dataset_current.csv',frame),('adjusted_triangle_associations.csv',associations),('moderation_interactions.csv',interactions),('lesion_gradient_simple_slopes.csv',slopes)]:d.to_csv(tabledir/name,index=False)
frame.to_csv(tabledir/'analysis_dataset_37.csv',index=False) # filename retained solely for original plotting entry
(RUN/'analysis_summary.json').write_text(json.dumps(dict(primary_n=len(primary),primary_groups=primary.group.value_counts().to_dict(),available_functional_n=len(subject),projection_primary_n=len(projected),projection_primary_groups=projected.group.value_counts().to_dict(),lesion_pairs_n=int(pd.read_csv(DIST/'tables/lesion_volume_subjects.csv').subject_id.nunique()),clinical_subgroup_groups=exact.group.value_counts().to_dict(),initial_matched_pairs=len(pairs),final_matched_pairs=len(active),G2_FDR_ROIs=int(maps.G2_FDR_significant.sum()),excluded_subject='PRIVATE_SUBJECT_008',group_overrides={'PRIVATE_SUBJECT_038':'Control'},matching_sex_encoding='M/F normalized to 1/0; legacy matching script incorrectly coded M/F as all zero'),indent=2),encoding='utf-8')
print('FIG4 ANALYSIS COMPLETE',flush=True)
