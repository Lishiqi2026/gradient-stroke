from pathlib import Path
import os,sys,json
os.environ['MPLBACKEND']='Agg'
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
RUN=Path(__file__).resolve().parents[1];WORK=RUN/'work';ROOT=WORK/'Gradient/lesion_analysis'
DIST=ROOT/'lesion_gradient_distance';FINAL=DIST/'figures/nature/lesion_final';BCDF=FINAL/'bcdf_recomputed_provided_reference';SUB=FINAL/'forSubmit_lesion';SOURCE=SUB/'source_data'
OUT=RUN/'rendered/figure4';OUT.mkdir(parents=True,exist_ok=True)
sys.path[:0]=[str(ROOT),str(BCDF),str(SUB/'scripts'),str(SUB/'figures')]
import numpy as np,pandas as pd,matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,Normalize
from nilearn import plotting
from statsmodels.stats.multitest import multipletests
import recompute_bcdf_with_provided_reference as b
import figure_common as common
import make_lesion_longitudinal_volume as volumes
import make_panel_a_interaction_three_versions as interaction
import make_standalone_g2_surfaces as surface
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
BLUE='#3F73A8';RED='#D6534A';COLORS={'Control':BLUE,'TMS':RED}
def save(fig,name):
    if name=='panel_a_interaction_version2_faceted_tertiles':
        fig.set_size_inches(145/25.4,60/25.4)
        fig.subplots_adjust(top=.70,bottom=.20,wspace=.20)
        for text in list(fig.texts)+[t for ax in fig.axes for t in ax.texts]:
            if text.get_fontsize()<7:text.set_fontsize(7)
    # Standalone plots intentionally contain no panel letters.
    for text in list(fig.texts)+[t for ax in fig.axes for t in ax.texts]:
        if text.get_text().strip() in [f'{x}{dot}' for x in 'abcdefg' for dot in ['', '.']]:text.remove()
        elif text.get_text().startswith('Lesion span modifies'):
            text.set_text('Lesion span and normalization–MMSE association')
        elif 'association strengthens' in text.get_text():
            text.set_text('; '.join(text.get_text().split(';')[:3]))
    fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight',pad_inches=.04,dpi=400)
    fig.savefig(OUT/f'{name}.png',bbox_inches='tight',pad_inches=.04,dpi=220)
    plt.close(fig)
def stars(q):return '***' if q<.001 else '**' if q<.01 else '*' if q<.05 else 'ns'
def bracket(ax,x1,x2,y,h,label):
    ax.plot([x1,x1,x2,x2],[y-h,y,y,y-h],color='#666666',lw=.7)
    ax.text((x1+x2)/2,y+h*.15,label,ha='center',va='bottom',fontsize=9)

print('RENDER longitudinal volume and overlap',flush=True)
long=pd.read_csv(DIST/'tables/lesion_volume_subjects.csv',dtype={'subject_id':str});long.to_csv(SOURCE/'lesion_volume_subjects.csv',index=False)
# Use all supplied valid masks; retain the previous ID exclusion as a separate sensitivity table.
volumes.EXCLUDED_SUBJECT='PRIVATE_SUBJECT_008';selected,statistics=volumes.prepare_plot_data(long)
statistics.to_csv(SOURCE/'lesion_volume_statistics_all_valid.csv',index=False)
volumes.EXCLUDED_SUBJECT='PRIVATE_SUBJECT_040';_,sensitivity=volumes.prepare_plot_data(long)
sensitivity.to_csv(SOURCE/'lesion_volume_statistics_legacy_outlier_exclusion.csv',index=False)
fig,axes=plt.subplots(1,2,figsize=(5.5,2.9));rng=np.random.default_rng(20260822)
for ax,group in zip(axes,['Control','TMS']):
    wide=selected[selected.group.eq(group)].pivot(index='subject_id',columns='time',values='lesion_volume_ml');color=COLORS[group]
    for row in wide.itertuples():
        jitter=rng.uniform(-.035,.035);ax.plot([jitter,1+jitter],[row.Pre,row.Post],color='#ACAFB4',alpha=.55,lw=.7)
        ax.scatter([jitter,1+jitter],[row.Pre,row.Post],c=[color,color],alpha=.75,s=20,zorder=3,edgecolors='none')
    maximum=float(wide[['Pre','Post']].to_numpy().max());height=max(maximum,.01)
    bracket(ax,-.08,1.08,height*1.12,height*.035,stars(float(statistics.loc[statistics.group.eq(group),'wilcoxon_fdr_q'].iloc[0])))
    ax.set(xticks=[0,1],xticklabels=['Pre','Post'],xlim=(-.35,1.35),ylim=(-height*.04,height*1.3),ylabel='Lesion volume (mL)',title=f'{group} (n={len(wide)})');ax.title.set_color(color)
fig.tight_layout(w_pad=2);save(fig,'lesion_volume_pre_post')
import plot_lesion_longitudinal_nature as overlap
overlap.SOURCE=SOURCE
wide=selected.pivot(index=['subject_id','group'],columns='time',values='lesion_volume_ml').reset_index().rename(columns={'Pre':'baseline_volume_ml','Post':'post_volume_ml'})
maps,plotted=overlap.validate_and_build_overlap_maps(wide);plotted.to_csv(SOURCE/'lesion_overlap_subjects.csv',index=False)
fig=plt.figure(figsize=(4.5,1.0))
for i,(group,phase) in enumerate([('Control','before'),('Control','after'),('TMS','before'),('TMS','after')]):
    ax=fig.add_axes([.005+i*.25,.24,.24,.55]);cmap=LinearSegmentedColormap.from_list('lesion'+str(i),['#F4F6F9',COLORS[group]])
    plotting.plot_glass_brain(maps[(group,phase)],display_mode='xz',axes=ax,figure=fig,colorbar=False,cmap=cmap,vmin=0,vmax=.4,threshold=1e-6,plot_abs=False,annotate=False)
    ax.set_title(f'{group} '+('Pre' if phase=='before' else 'Post')+f' (n={int(wide.group.eq(group).sum())})',fontsize=6.5)
    color_axis=fig.add_axes([.035+i*.25,.10,.18,.035]);bar=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,.4),cmap=cmap),cax=color_axis,orientation='horizontal');bar.set_ticks([0,.2,.4]);bar.ax.tick_params(labelsize=6,length=1,pad=1);bar.outline.set_linewidth(.3)
save(fig,'lesion_overlap_maps')

print('RENDER clinical subgroup',flush=True)
data=pd.read_csv(SOURCE/'lesion_matched_clinical_improvements.csv');tests=pd.read_csv(SOURCE/'lesion_matched_clinical_statistics.csv')
fig,axes=plt.subplots(2,1,figsize=(3.6,4.1));rng=np.random.default_rng(20260822)
for ax,outcome in zip(axes,['NIHSS','ADL']):
    frame=data[data.outcome.eq(outcome)]
    for i,group in enumerate(['Control','TMS']):
        values=frame.loc[frame.group.eq(group),'improvement'].to_numpy();ax.scatter(i+rng.normal(0,.035,len(values)),values,s=24,color=COLORS[group],edgecolor='white',linewidth=.4)
        ax.plot([i-.12,i+.12],[values.mean()]*2,color='#555555',lw=1)
    lo=min(0,float(frame.improvement.min()));hi=float(frame.improvement.max());span=max(hi-lo,1);row=tests[tests.outcome.eq(outcome)].iloc[0]
    bracket(ax,0,1,hi+.15*span,.035*span,stars(row.permutation_fdr_q));ax.set(xlim=(-.35,1.35),ylim=(lo-.05*span,hi+.35*span),title=outcome,ylabel='Clinical improvement (points)',xticks=[0,1],xticklabels=[f'{g}\n(n={int(frame.group.eq(g).sum())})' for g in ['Control','TMS']]);ax.axhline(0,color='#BBBBBB',lw=.6)
fig.tight_layout(h_pad=1.4);save(fig,'clinical_subgroup_improvement')

print('RENDER whole-cortex and distance panels',flush=True)
subject=pd.read_csv(BCDF/'recomputed_subject_metrics.csv');subject=subject.loc[subject.primary_motion_eligible.astype(bool)];models=pd.read_csv(BCDF/'recomputed_group_models.csv');models=models[models.cohort.eq('primary_motion_qc')]
def group_violin(metrics,names,title,ylabel,name):
    fig,axes=plt.subplots(1,len(metrics),figsize=(1.9*len(metrics),2.5),squeeze=False)
    for ax,metric,label in zip(axes[0],metrics,names):
        available=subject.dropna(subset=[metric])
        for i,g in enumerate(['Control','TMS']):b.violin_points(ax,available.loc[available.group.eq(g),metric].to_numpy(),i,COLORS[g],np.random.default_rng(20260822+i),width=.6)
        vals=available[metric].to_numpy();lo=float(vals.min());hi=float(vals.max());span=max(hi-lo,.01);row=models[models.outcome.eq(metric)].iloc[0]
        bracket(ax,0,1,hi+.12*span,.035*span,stars(row.fdr_q_freedman_lane));ax.set(ylim=(lo-.07*span,hi+.35*span),xticks=[0,1],xticklabels=[f'{g}\n(n={int(available.group.eq(g).sum())})' for g in ['Control','TMS']],ylabel=ylabel,title=label);ax.axhline(0,color='#BBBBBB',lw=.6)
    fig.suptitle(title,fontsize=9,fontweight='bold');fig.tight_layout();save(fig,name)
group_violin(['mean_reference_directed_G2','mean_reference_directed_G1G2'],['G2','G1–G2'],'Whole-cortex normalization','Healthy-reference normalization','whole_cortex_normalization')
group_violin(['xtract_projected_G2_slope'],['G2 distance–normalization slope'],'Individual G2 slopes','Within-subject G2 slope','individual_distance_slopes')
summary=pd.read_csv(BCDF/'panel_c_group_distance_bin_summary.csv');fig,ax=plt.subplots(figsize=(2.5,2.3));b.plot_panel_c(ax,summary,False);ax.set_title('',loc='left');ax.set_title('G2 normalization across\nlesion-related functional distance',fontsize=8);fig.tight_layout();save(fig,'normalization_by_distance')
pairs=pd.read_csv(BCDF/'recomputed_matched_pair_values.csv');tests=pd.read_csv(BCDF/'recomputed_matched_tests.csv');fig,ax=plt.subplots(figsize=(2.3,2.2));b.plot_panel_f(ax,pairs,tests,False);ax.set_title('',loc='left');ax.set_title('Lesion-matched comparison',fontsize=8)
count=int(tests.loc[tests.match_set.eq('motion_qc_pairs'),'n_pairs'].iloc[0]) if 'n_pairs' in tests else pairs.loc[pairs.match_set.eq('motion_qc_pairs'),'pair_id'].nunique();ax.set_title(f'Lesion-matched comparison ({count} pairs)',fontsize=8);fig.tight_layout();save(fig,'lesion_matched_comparison')

print('RENDER ROI maps and network enrichment',flush=True)
source=pd.read_csv(SOURCE/'G1_G2_surface_source_data.csv');order=['Visual','SomMot','DorsAttn','SalVentAttn','Limbic','Control','Default'];abbr=['VIS','SMN','DAN','VAN','LIM','CON','DMN'];counts=[]
for network in order:
    d=source[source.network.eq(network)];s=d.G2_FDR_significant.astype(bool)
    counts.append(dict(network=network,network_abbreviation=abbr[order.index(network)],negative_FDR_ROI_count=int((s&d.adjusted_TMS_beta_G2.lt(0)).sum()),positive_FDR_ROI_count=int((s&d.adjusted_TMS_beta_G2.gt(0)).sum())))
pd.DataFrame(counts).to_csv(SOURCE/'G2_FDR_network_counts.csv',index=False)
from render_legacy_network_radar import render_network_radar
render_network_radar(SOURCE,save)
stats=pd.read_csv(SOURCE/'G2_network_spin_statistics.csv');nulls=np.load(SOURCE/'G2_network_spin_nulls.npz');positive=stats[stats.direction.eq('positive')];fig,ax=plt.subplots(figsize=(2.35,2.0))
if positive.n_selected_rois.iloc[0]>0:
    violin=ax.violinplot([nulls['positive'][:,i] for i in range(7)],positions=range(7),showextrema=False,widths=.8)
    for body in violin['bodies']:body.set_facecolor('#DDA7AE');body.set_alpha(.5);body.set_edgecolor('none')
    for i,row in enumerate(positive.itertuples()):
        ax.plot([i-.28,i+.28],[row.actual_enrichment_ratio]*2,color='#B2182B',lw=1.3)
        if row.spin_fdr_q<.05:ax.text(i,max(row.actual_enrichment_ratio,float(np.quantile(nulls['positive'][:,i],.99)))+.025,f'q={row.spin_fdr_q:.3g}',ha='center',fontsize=6,color='#B2182B')
else:ax.text(.5,.5,'No positive FDR-significant ROIs',ha='center',transform=ax.transAxes)
ax.set(xticks=range(7),xticklabels=abbr,ylabel='Network enrichment ratio',title='Spin test of positive G2 effect');fig.tight_layout();save(fig,'network_spin_test')
surface.save_standalone_crisp=lambda fig,stem:save(fig,'healthy_reference_G2' if stem=='healthy_normative_G2' else 'adjusted_TMS_effect_G2')
if '--reuse-surfaces' in sys.argv:
    render_original=surface.render_surface_crisp
    def render_cached(values,name,cmap,limit):
        path=SUB/'qa/surface_cache'/f'{name}.png'
        return path if path.exists() else render_original(values,name,cmap,limit)
    surface.render_surface_crisp=render_cached
surface.main()
from matplotlib.colors import Normalize
fig=plt.figure(figsize=(3.45,1.8))
for j,(stem,title,cmap,limit,label) in enumerate([('standalone_healthy_normative_g2','Healthy reference G2',surface.G2_STANDALONE_CMAP,2.,'Reference G2 (z)'),('standalone_g2_normalized_tms_effect_fdr',f'TMS G2 effect ({int(source.G2_FDR_significant.sum())} ROIs)',surface.TMS_EFFECT_CMAP,surface.robust_limit(source.adjusted_TMS_beta_G2.to_numpy()),'Adjusted TMS beta')]):
    ax=fig.add_axes([.01+j*.5,.22,.48,.64]);ax.imshow(plt.imread(SUB/'qa/surface_cache'/f'{stem}.png'));ax.set_axis_off();ax.set_title(title,fontsize=6.8,pad=3)
    cax=fig.add_axes([.045+j*.5,.11,.41,.025]);cb=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(-limit,limit),cmap=cmap),cax=cax,orientation='horizontal');cb.set_ticks([-limit,0,limit]);cb.ax.tick_params(labelsize=6,length=1.5,pad=1);cb.set_label(label,fontsize=6,labelpad=1);cb.outline.set_linewidth(.4)
save(fig,'G2_reference_and_adjusted_effect')
print('RENDER MMSE interaction',flush=True)
common.save_submission_figure=lambda fig,stem,**kw:save(fig,stem)
interaction.save_submission_figure=common.save_submission_figure
data,model,moderation,slopes=interaction.load_analysis();adjusted=interaction.adjusted_outcome(data,model)
model_lines=interaction.version_2_faceted_tertiles(data,adjusted,model,moderation)
model_lines.to_csv(SOURCE/'MMSE_conditional_model_lines.csv',index=False)
points=data.copy();points['covariate_adjusted_MMSE_improvement_z']=adjusted
points['display_span_tertile']=pd.qcut(points.x_z,3,labels=['Low','Intermediate','High'])
points.to_csv(SOURCE/'MMSE_interaction_plotted_subjects.csv',index=False)
records=[]
for level,part in points.groupby('display_span_tertile',observed=True):
    span=float(part.x_z.median());records.append(dict(lesion_span_group=str(level),n=len(part),median_lesion_span_z=span,conditional_gradient_slope=float(model.params['m_z']+span*model.params['x_by_m'])))
pd.DataFrame(records).to_csv(SOURCE/'MMSE_plotted_group_slopes.csv',index=False)
print('FIG4 PANELS COMPLETE',flush=True)
