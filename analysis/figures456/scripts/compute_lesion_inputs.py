from pathlib import Path
import os,sys,json,time
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
RUN=Path(__file__).resolve().parents[1];WORK=RUN/'work';PROJECT=WORK/'Gradient';ROOT=PROJECT/'lesion_analysis'
os.environ['JOBLIB_TEMP_FOLDER']=str(WORK/'.joblib_temp')
import numpy as np,pandas as pd,nibabel as nib
from scipy import ndimage
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from brainspace.datasets import load_group_fc
sys.path[:0]=[str(ROOT),str(PROJECT/'ml_analysis')]
import run_lesion_gradient_distance as distance
import analyze_lesion_anatomical_compartments as anatomy
import run_lesion_extended_analysis as extended
import audit_healthy_reference_gradient_calculation as embedding

DIST=ROOT/'lesion_gradient_distance';TABLES=DIST/'tables';PROVIDED=DIST/'provided_code_g1_g2_recalculation'
for p in [TABLES,PROVIDED,PROVIDED/'aligned_gradients_z',ROOT/'results_abcd/tables']:p.mkdir(parents=True,exist_ok=True)
subjects=pd.read_csv(WORK/'subjects.csv',dtype={'subject_id':str})
if '--wait' in sys.argv:
    while True:
        exits=list((RUN/'registration_status').glob('*.exit'))
        if len(exits)==len(subjects):break
        print('WAITING registration',len(exits),'/',len(subjects),flush=True);time.sleep(20)
errors={p.stem:p.read_text().strip() for p in (RUN/'registration_status').glob('*.exit') if p.read_text().strip()!='0'}
if errors:raise RuntimeError(f'Registration failures need resolution: {errors}')

atlas_image,atlas,labels=distance.load_atlas()
resource=ROOT/'resources'
brain=np.asarray(nib.load(resource/'MNI152_T1_2mm_brain_mask.nii.gz').dataobj)>.5
ref,ho,cereb,jhu,_=anatomy.load_atlases();compartment_map,definition=anatomy.build_compartment_map(ho,cereb,brain)
labels.to_csv(TABLES/'atlas_labels.csv',index=False)
clinical=pd.read_csv(WORK/'clinical_corrected.csv',dtype={'subject_id':str})
qc=pd.read_csv(DIST/'fc_consistent/tables/longitudinal_fc_qc.csv',dtype={'subject_id':str})
eligible=qc.groupby('subject_id').primary_motion_eligible.all();eligible_ids=set(eligible.index[eligible])
fc_ready_ids={sid for sid in subjects.subject_id if all((DIST/'fc_consistent/fc_matrices_z'/f'{sid}_{phase}_fc_z.npy').exists() for phase in ['baseline','post'])}
assert 'PRIVATE_SUBJECT_008' not in eligible_ids

review_path=WORK/'registration_review.csv'
review=pd.read_csv(review_path,dtype={'subject_id':str}).set_index('subject_id') if review_path.exists() else None
lesion_rows=[];long_rows=[];audit=[]
for row in subjects.itertuples():
    sid=row.subject_id;images={};masks={}
    for phase in ['before','after']:
        f=ROOT/'mni_registration/masks'/row.group/phase/f'{sid}_lesion_mni2mm.nii.gz'
        im=nib.load(f);m=np.asarray(im.dataobj)>.5
        assert im.shape==atlas.shape and np.allclose(im.affine,atlas_image.affine)
        if not m.any():raise RuntimeError(f'Empty new standardized mask {sid}/{phase}; native-positive mask needs registration review')
        images[phase]=im;masks[phase]=m
        native=next((WORK/'native_lesions'/row.group/phase).glob(f'{sid}-*.nii*'))
        nim=nib.load(native);nmask=np.asarray(nim.dataobj)>.5
        audit.append(dict(subject_id=sid,group=row.group,phase=phase,native_voxels=int(nmask.sum()),native_volume_ml=float(nmask.sum()*abs(np.linalg.det(nim.affine[:3,:3]))/1000),mni_voxels=int(m.sum()),mni_volume_ml=float(m.sum()*.008)))
        long_rows.append(dict(subject_id=sid,group=row.group,time='Pre' if phase=='before' else 'Post',lesion_volume_ml=float(m.sum()*.008)))
    m=masks['before'];xyz=nib.affines.apply_affine(atlas_image.affine,np.argwhere(m));n=int(m.sum());left=float((xyz[:,0]<0).sum()/n);right=float((xyz[:,0]>0).sum()/n)
    damage=np.bincount(atlas[m],minlength=401)[1:]
    lesion_rows.append(dict(subject_id=sid,group=row.group,baseline_lesion_volume_ml=n*.008,log1p_lesion_volume=float(np.log1p(n*.008)),mask_lateralization_index=right-left,mask_derived_side='left' if left>right else 'right',registration_manual_status=str(review.loc[sid,'status']) if review is not None else 'review_record_not_supplied',schaefer_cortical_overlap_ml=float(damage.sum()*.008),schaefer_cortical_fraction_pct=float(damage.sum()*100/n)))
lesion=pd.DataFrame(lesion_rows)
lesion.to_csv(ROOT/'results_abcd/tables/B_participant_lesion_metrics.csv',index=False)
pd.DataFrame(long_rows).to_csv(TABLES/'lesion_volume_subjects.csv',index=False)
pd.DataFrame(audit).to_csv(TABLES/'native_vs_mni_lesion_audit.csv',index=False)

print('EXTRACT anatomy and XTRACT',flush=True)
anatomical,tracts=anatomy.extract_anatomical_metrics(lesion,ref,compartment_map,jhu,brain)
anatomical=anatomical.merge(lesion[['subject_id','mask_derived_side','registration_manual_status']],on='subject_id',validate='one_to_one')
anatomical.to_csv(ROOT/'results_abcd/tables/B2_anatomical_compartment_metrics.csv',index=False)
spatial=extended.extract_spatial_phenotypes(anatomical);spatial.to_csv(ROOT/'results_abcd/tables/E_spatial_lesion_phenotypes.csv',index=False)
xtract,systems,maps=extended.extract_xtract_metrics(anatomical);del maps
xtract.to_csv(ROOT/'results_abcd/tables/F_xtract_tract_probability_burden.csv',index=False)
systems.to_csv(ROOT/'results_abcd/tables/F_xtract_system_probability_burden.csv',index=False)
endpoints,endpoint_table,endpoint_total=distance.build_xtract_endpoint_projection(atlas_image,atlas,xtract[['tract_index','tract_name']].drop_duplicates())
endpoint_table.to_csv(TABLES/'xtract_schaefer400_endpoint_projection.csv',index=False)
np.save(TABLES/'xtract_endpoint_weights.npy',endpoints)

topology_names=['brainstem','deep_subcortical_gray','cerebral_white_matter','cerebral_cortex']
x=np.column_stack([np.log1p(anatomical[c+'_fraction_pct']) for c in topology_names]);scale=StandardScaler();pca=PCA(n_components=2);scores=pca.fit_transform(scale.fit_transform(x))
topology=anatomical[['subject_id','group']].copy();topology['lesion_topology_pc1']=scores[:,0];topology['lesion_topology_pc2']=scores[:,1];topology.to_csv(ROOT/'results_abcd/tables/C2_lesion_topology_pca_subject_scores.csv',index=False)
(TABLES/'topology_pca_definition.json').write_text(json.dumps(dict(features=topology_names,mean=scale.mean_.tolist(),scale=scale.scale_.tolist(),components=pca.components_.tolist(),explained_variance_ratio=pca.explained_variance_ratio_.tolist()),indent=2),encoding='utf-8')

print('FIXED HEALTHY REFERENCE and new-mask gradient alignment',flush=True)
official=np.load(DIST/'references/healthy_normative_reference_400x5.npy')
healthy_fc_file=DIST/'references/healthy_main_fc_400x400.npy'
healthy_fc=np.load(healthy_fc_file) if healthy_fc_file.exists() else load_group_fc('schaefer',scale=400,group='main')
raw,_,_=embedding.legacy_style_adapted_embedding(healthy_fc)
master,_=distance.similarity_align(raw,official);mean=master.mean(axis=0);sd=master.std(axis=0,ddof=1);reference=(master-mean)/sd
old_reference=np.load(PROVIDED/'healthy_reference_provided_code_aligned_z_400x5.npy')
assert np.max(np.abs(reference-old_reference))<1e-8
np.save(PROVIDED/'healthy_reference_provided_code_aligned_z_400x5.npy',reference)
np.save(PROVIDED/'healthy_reference_provided_code_aligned_400x5.npy',master)
legacy_ref=(official-official.mean(axis=0))/official.std(axis=0,ddof=1)
roi_parts=[];provided_parts=[];summary=[];alignment=[]
meta=lesion.set_index('subject_id')
for row in subjects.itertuples():
    sid=row.subject_id
    if sid not in fc_ready_ids:continue
    m=np.asarray(nib.load(ROOT/'mni_registration/masks'/row.group/'before'/f'{sid}_lesion_mni2mm.nii.gz').dataobj)>.5
    damage=np.bincount(atlas[m],minlength=401)[1:];intact=damage==0
    q=qc.loc[qc.subject_id.eq(sid)].set_index('session')
    info=dict(subject_id=sid,group=row.group,group_tms=int(row.group=='TMS'),primary_motion_eligible=bool(sid in eligible_ids),baseline_mean_fd=float(q.loc['baseline','mean_fd']),post_mean_fd=float(q.loc['post','mean_fd']),baseline_retained_frames=int(q.loc['baseline','retained_frames']),post_retained_frames=int(q.loc['post','retained_frames']),delta_mean_fd=float(q.loc['post','mean_fd']-q.loc['baseline','mean_fd']),lesion_volume_ml_recomputed=float(m.sum()*.008),log1p_lesion_volume=float(meta.loc[sid,'log1p_lesion_volume']),mask_lateralization_index=float(meta.loc[sid,'mask_lateralization_index']),cortical_overlap_voxels=int(damage.sum()),cortical_overlap_roi_n=int((damage>0).sum()))
    burden=xtract.loc[xtract.subject_id.eq(sid)].sort_values('tract_index').tract_probability_mass_coverage_pct.to_numpy(float)/100
    w=burden@endpoints
    projectable=bool(np.isfinite(w).all() and w.sum()>0)
    info['xtract_projection_eligible']=projectable
    info['xtract_projection_reason']='available' if projectable else 'no_cortical_endpoint_projection'
    if projectable:
        w=w/w.sum();location=np.average(legacy_ref,axis=0,weights=w)
        spans=[float(np.sqrt(np.average((legacy_ref[:,i]-location[i])**2,weights=w))) for i in [0,1]]
        effective=float(1/(w*w).sum());maximum=float(w.max())
    else:
        w=np.zeros(400);spans=[np.nan,np.nan];effective=np.nan;maximum=np.nan
        print('NO XTRACT CORTICAL PROJECTION',sid,'retained for global metrics',flush=True)
    info.update(xtract_projectable_tract_damage_fraction=float(burden[endpoint_total>0].sum()/burden.sum()) if burden.sum()>0 else 0.,xtract_projected_effective_roi_n=effective,xtract_projected_max_roi_weight=maximum,xtract_projected_G1_span=spans[0],xtract_projected_G2_span=spans[1])
    before=after=None
    for session,tag in [('baseline','pre'),('post','post')]:
        fc=np.load(DIST/'fc_consistent/fc_matrices_z'/f'{sid}_{session}_fc_z.npy')
        raw,lambdas,_=embedding.legacy_style_adapted_embedding(np.tanh(fc));aligned,fit=distance.similarity_align(raw,master,intact);z=(aligned-mean)/sd
        np.save(PROVIDED/'aligned_gradients_z'/f'{sid}_{tag}_aligned_z_400x5.npy',z)
        if tag=='pre':before=z
        else:after=z
        alignment.append(dict(subject_id=sid,phase=tag,intact_roi_n=int(intact.sum()),**fit))
    distmap=ndimage.distance_transform_edt(~m,sampling=atlas_image.header.get_zooms()[:3])
    frame=labels[['roi','roi_name','network','centroid_x','centroid_y','centroid_z']].copy()
    for key,value in info.items():frame[key]=value
    frame['intact_roi']=intact;frame['directly_damaged_voxels']=damage;frame['xtract_projected_roi_weight']=w
    frame['anatomical_distance_min_mm']=ndimage.minimum(distmap,labels=atlas,index=np.arange(1,401))
    frame['anatomical_distance_median_mm']=ndimage.median(distmap,labels=atlas,index=np.arange(1,401))
    for g in ['G1','G2']:
        i=int(g[-1])-1;frame[f'baseline_deviation_{g}']=np.abs(before[:,i]-reference[:,i]);frame[f'normalization_{g}']=np.abs(before[:,i]-reference[:,i])-np.abs(after[:,i]-reference[:,i])
    roi_parts.append(frame);provided_parts.append(frame.copy());summary.append(info)
    print('SUBJECT METRICS',sid,'intact',int(intact.sum()),flush=True)
pd.concat(roi_parts,ignore_index=True).to_csv(TABLES/'roi_level_distance_reorganization.csv',index=False)
pd.concat(provided_parts,ignore_index=True).to_csv(PROVIDED/'provided_code_roi_normalization_long.csv',index=False)
pd.DataFrame(summary).to_csv(TABLES/'subject_level_distance_metrics.csv',index=False)
pd.DataFrame(alignment).to_csv(PROVIDED/'subject_alignment_qc.csv',index=False)
print('LESION INPUTS COMPLETE',len(summary),flush=True)
