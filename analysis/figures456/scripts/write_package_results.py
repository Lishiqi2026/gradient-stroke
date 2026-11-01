from pathlib import Path
import os,json,shutil,hashlib,csv
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT/'work';PROJECT=WORK/'Gradient';AN=PROJECT/'lesion_analysis';DIST=AN/'lesion_gradient_distance';BCDF=DIST/'figures/nature/lesion_final/bcdf_recomputed_provided_reference'
RESULT=ROOT/'results';RESULT.mkdir(exist_ok=True)
subjects=pd.read_csv(WORK/'subjects.csv',dtype={'subject_id':str});clinical=pd.read_csv(WORK/'clinical_corrected.csv',dtype={'subject_id':str})
qc=pd.read_csv(DIST/'fc_consistent/tables/longitudinal_fc_qc.csv',dtype={'subject_id':str});both=qc.groupby('subject_id').primary_motion_eligible.all();metrics=pd.read_csv(BCDF/'recomputed_subject_metrics.csv',dtype={'subject_id':str});projection=metrics.set_index('subject_id').xtract_projection_eligible
flow=clinical[['subject_id','group']].copy();flow['paired_lesion_available']=flow.subject_id.isin(subjects.subject_id);flow['both_sessions_pass_motion_QC']=flow.subject_id.map(both).fillna(False);flow['cortical_projection_available']=flow.subject_id.map(projection).fillna(False)
flow['global_normalization_included']=flow.both_sessions_pass_motion_QC;flow['distance_and_interaction_included']=flow.global_normalization_included&flow.cortical_projection_available;flow['primary_ML_included']=flow.distance_and_interaction_included
for phase in ['baseline','post']:
    lookup=qc[qc.session.eq(phase)].set_index('subject_id').retained_frames;flow[phase+'_retained_frames']=flow.subject_id.map(lookup)
source=DIST/'figures/nature/lesion_final/forSubmit_lesion/source_data'
if (source/'exact_clinical_subgroup_subjects.csv').exists():exact=pd.read_csv(source/'exact_clinical_subgroup_subjects.csv',dtype={'subject_id':str});flow['clinical_subgroup_included']=flow.subject_id.isin(exact.subject_id)
pairs=AN/'lesion_matched_clinical_mediation/tables/motion_eligible_pairs.csv'
if pairs.exists():
    pairs=pd.read_csv(pairs,dtype={'control_subject_id':str,'tms_subject_id':str});flow['matched_pair_included']=flow.subject_id.isin(set(pairs.control_subject_id)|set(pairs.tms_subject_id))
flow['exclusion_reason']=np.where(~flow.paired_lesion_available,'no_supplied_paired_new_lesion',np.where(~flow.both_sessions_pass_motion_QC,'fewer_than_100_retained_frames_in_at_least_one_session',np.where(~flow.cortical_projection_available,'no_XTRACT_cortical_endpoint_projection_for_distance_features','')))
assert 'PRIVATE_SUBJECT_008' not in set(flow.subject_id);assert flow.loc[flow.subject_id.eq('PRIVATE_SUBJECT_038'),'group'].tolist()==['Control'];flow.to_csv(RESULT/'participant_inclusion_by_analysis.csv',index=False)
if (BCDF/'recomputed_group_models.csv').exists():
    models=pd.read_csv(BCDF/'recomputed_group_models.csv');models=models[models.cohort.eq('primary_motion_qc')&models.outcome.isin(['mean_reference_directed_G2','mean_reference_directed_G1G2','xtract_projected_G2_slope'])];models.to_csv(RESULT/'Fig4_group_effects.csv',index=False)
    lines=['Updated Fig. 4 statistics (same original model and FDR families).']
    for row in models.itertuples():lines.append(f'{row.outcome}: n={row.n}, adjusted beta={row.group_tms_beta:.6g}, 95% HC3 CI [{row.ci_low_hc3:.6g}, {row.ci_high_hc3:.6g}], permutation P={row.p_freedman_lane:.6g}, q={row.fdr_q_freedman_lane:.6g}.')
    (RESULT/'Fig4_results_text.txt').write_text('\n'.join(lines),encoding='utf-8')
if (BCDF/'recomputed_matched_tests.csv').exists():
    matched=pd.read_csv(BCDF/'recomputed_matched_tests.csv');matched=matched.loc[matched.match_set.eq('motion_qc_pairs')]
    matched.to_csv(RESULT/'Fig4f_exact_paired_tests.csv',index=False)
    values=pd.read_csv(BCDF/'recomputed_matched_pair_values.csv',dtype={'control_subject_id':str,'tms_subject_id':str})
    values=values.loc[values.match_set.eq('motion_qc_pairs')&values.outcome.eq('nearest_xtract_G1G2_decile_normalization')].sort_values('pair_id')
    values.to_csv(RESULT/'Fig4f_paired_differences.csv',index=False)
triangle=AN/'lesion_gradient_score_triangle_exploratory/tables'
if (triangle/'moderation_interactions.csv').exists():
    interactions=pd.read_csv(triangle/'moderation_interactions.csv')
    mmse=interactions.loc[interactions.outcome.eq('MMSE')&interactions.relationship.eq('lesion_modifies_gradient_to_clinical')]
    mmse.to_csv(RESULT/'Fig4g_MMSE_interaction.csv',index=False)
    slopes=pd.read_csv(triangle/'lesion_gradient_simple_slopes.csv');slopes.loc[slopes.outcome.eq('MMSE')].to_csv(RESULT/'Fig4g_slopes_at_mean_plusminus_SD.csv',index=False)
    if (source/'MMSE_plotted_group_slopes.csv').exists():shutil.copy2(source/'MMSE_plotted_group_slopes.csv',RESULT/'Fig4g_plotted_group_slopes.csv')
if (BCDF/'recomputed_group_models.csv').exists() and (RESULT/'Fig4f_exact_paired_tests.csv').exists():
    text=[]
    for metric,panel in [('mean_reference_directed_G2','Fig. 4d G2'),('mean_reference_directed_G1G2','Fig. 4d G1–G2'),('xtract_projected_G2_slope','Fig. 4e 斜率组间差异')]:
        row=models.loc[models.outcome.eq(metric)].iloc[0]
        text.append(f'{panel}：n={row["n"]}，调整后 β={row.group_tms_beta:.6f}，95% HC3 CI [{row.ci_low_hc3:.6f}, {row.ci_high_hc3:.6f}]，置换 P={row.p_freedman_lane:.6f}，q={row.fdr_q_freedman_lane:.6f}。')
    row=matched.loc[matched.outcome.eq('nearest_xtract_G1G2_decile_normalization')].iloc[0]
    text.append(f'Fig. 4f：最终 {int(row.n_pairs)} 对；TMS−Control 平均配对差异={row.mean_tms_minus_control:.6f}，95% t CI [{row.ci_low_t:.6f}, {row.ci_high_t:.6f}]，精确符号翻转 P={row.exact_signflip_p:.8f}，q={row.fdr_q_within_match_set:.6f}，共 {int(row.exact_permutations)} 种符号组合。逐对数值见 Fig4f_paired_differences.csv。')
    if (RESULT/'Fig4g_MMSE_interaction.csv').exists():
        row=mmse.iloc[0];text.append(f'Fig. 4g：n={int(row["n"])}，病灶跨度×归一化交互 β={row.adjusted_beta:.6f}，95% HC3 CI [{row.adjusted_ci_low_hc3:.6f}, {row.adjusted_ci_high_hc3:.6f}]，置换 P={row.adjusted_p_freedman_lane:.6f}，13 项检验的 FDR q={row.fdr_q_freedman_lane_all_13:.6f}；本次交互作用未达到显著性。')
    if (RESULT/'Fig4g_plotted_group_slopes.csv').exists():
        groups=pd.read_csv(RESULT/'Fig4g_plotted_group_slopes.csv')
        text.append('Fig. 4g 图中各组中位病灶跨度处的模型斜率：'+'；'.join(f'{r.lesion_span_group}，n={r.n}，跨度 z={r.median_lesion_span_z:.4f}，β={r.conditional_gradient_slope:.6f}' for r in groups.itertuples())+'。这些显示斜率与另表中 ±1 SD 处的简单斜率定义不同。')
    if (source/'G1_G2_surface_source_data.csv').exists():
        maps=pd.read_csv(source/'G1_G2_surface_source_data.csv');significant=maps.G2_FDR_significant.astype(bool)
        text.append(f'Fig. 4c：G2 FDR 显著 ROI 共 {int(significant.sum())} 个，其中正向 {int((significant&maps.adjusted_TMS_beta_G2.gt(0)).sum())} 个、负向 {int((significant&maps.adjusted_TMS_beta_G2.lt(0)).sum())} 个；网络旋转检验完整 P/q 见重画数据中的 G2_network_spin_statistics.csv。')
    (RESULT/'Fig4_latest_results_CN.txt').write_text('\n\n'.join(text),encoding='utf-8')
config=ROOT/'package_config.json';figure=int(os.environ['PAPER_FIGURE']) if os.environ.get('PAPER_FIGURE') else (json.loads(config.read_text())['figure'] if config.exists() else 0)
inputs=ROOT/'input_data/work';cache=RESULT/'replay_data';cache.mkdir(exist_ok=True)
for f in WORK.rglob('*'):
    if not f.is_file() or f.suffix not in ['.csv','.tsv','.json','.npy','.npz','.txt']:continue
    rel=f.relative_to(WORK)
    if '.joblib_temp' in f.parts or any('checkpoints' in part for part in f.parts) or (inputs/rel).exists():continue
    # Result tables and aligned gradients are enough to replay plots; no opaque model binaries.
    target=cache/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,target)
for dirname in ['cross_cohort_recomputed']:
    if (ROOT/dirname).exists():shutil.copytree(ROOT/dirname,RESULT/dirname,dirs_exist_ok=True)
report=dict(figure=figure,available_functional_sensitivity_n=len(metrics),clinical_input_n=len(flow),paired_lesion_n=int(flow.paired_lesion_available.sum()),global_normalization_n=int(flow.global_normalization_included.sum()),distance_interaction_ML_n=int(flow.distance_and_interaction_included.sum()),excluded_subjects=['PRIVATE_SUBJECT_008'],group_overrides={'PRIVATE_SUBJECT_038':'Control'},missing_projection_subjects=flow.loc[flow.global_normalization_included&~flow.cortical_projection_available,'subject_id'].tolist(),motion_excluded=flow.loc[flow.paired_lesion_available&~flow.both_sessions_pass_motion_QC,'subject_id'].tolist(),scope='Reproduction starts from supplied processed ROI time series, motion traces, registered lesions, baseline gradients and external processed FC/gradients. Raw scanner preprocessing is outside this package.')
(RESULT/'analysis_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
entries=[]
for f in ROOT.rglob('*'):
    if not f.is_file() or f.name=='FILE_MANIFEST.csv' or 'work'==f.relative_to(ROOT).parts[0] or '__pycache__' in f.parts:continue
    entries.append(dict(path=f.relative_to(ROOT).as_posix(),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
with (ROOT/'FILE_MANIFEST.csv').open('w',encoding='utf-8-sig',newline='') as out:
    writer=csv.DictWriter(out,fieldnames=['path','bytes','sha256']);writer.writeheader();writer.writerows(entries)
print('RESULTS AND MANIFEST WRITTEN',report,flush=True)
