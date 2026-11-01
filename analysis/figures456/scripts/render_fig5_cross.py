from pathlib import Path
import os,sys,json,textwrap,shutil,hashlib
os.environ['MPLBACKEND']='Agg'
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
RUN=Path(__file__).resolve().parents[1];ROOT=RUN/'work/Gradient/lesion_analysis'
HERE=ROOT/'external_validation/final/main_tms_vs_external_adjusted_beta_surface_20260829'
OUT=RUN/'rendered/figure5';OUT.mkdir(parents=True,exist_ok=True)
import numpy as np,pandas as pd,matplotlib.pyplot as plt,matplotlib.image as mpimg
from matplotlib.colors import Normalize
sys.path.insert(0,str(HERE))
import make_main_tms_vs_external_adjusted_beta_surfaces as base
import make_toprow_separate_colorbars_spatial_correlation as top
import run_G2_positive_spintest_main_vs_cross_cohort as spin
base.apply_style();base.SOURCE.mkdir(exist_ok=True)
internal=pd.read_csv(base.MAIN_SOURCE).sort_values('roi');cross=pd.read_csv(RUN/'cross_cohort_recomputed/cross_cohort_roi_models.csv')
cols=['roi','roi_name','network','adjusted_TMS_beta_G2','permutation_p_G2','fdr_q_G2','G2_FDR_significant']
source=internal[cols].merge(cross,on=['roi','roi_name','network'],validate='one_to_one')
source['main_internal_FDR_masked_beta']=np.where(source.G2_FDR_significant,source.adjusted_TMS_beta_G2,0)
source['cross_cohort_FDR_significant']=source.cross_cohort_fdr_q_400.lt(.05)
source['cross_cohort_FDR_masked_beta']=np.where(source.cross_cohort_FDR_significant,source.main_tms_minus_external_adjusted_beta,0)
source.to_csv(base.SOURCE/'main_internal_and_cross_cohort_G2_beta_maps.csv',index=False)
stats=top.compute_spatial_statistics(source).iloc[0]
spin_stats,within_null,cross_null=spin.prepare_spin_results()
def save(fig,name):
    for t in list(fig.texts)+[t for ax in fig.axes for t in ax.texts]:
        if t.get_text().strip() in ['a','b','c','d','e','f','a.','b.','c.','d.','e.','f.']:t.remove()
    fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight',pad_inches=.04,dpi=400);fig.savefig(OUT/f'{name}.png',dpi=220,bbox_inches='tight',pad_inches=.04);plt.close(fig)
fig=plt.figure(figsize=(4.6,4))
for j,(column,title,cmap,label) in enumerate([('adjusted_TMS_beta_G2','Main TMS vs Main Control',top.INTERNAL_CMAP,'Within-study adjusted TMS beta'),('main_tms_minus_external_adjusted_beta','Main TMS vs External',top.CROSS_CMAP,'Cross-cohort adjusted group beta')]):
    values=source[column].to_numpy();limit=top.map_specific_limit(values)
    # Distinct cache names prevent reuse of an older beta-map image.
    digest=hashlib.sha256(values.tobytes()+str(limit).encode()).hexdigest()[:12]
    path=top.render_surface_hires(values,'refreshed_'+digest+'_'+column,limit,cmap)
    y=.55-j*.48;ax=fig.add_axes([.015,y+.04,.97,.30]);ax.imshow(mpimg.imread(path));ax.set_axis_off();ax.set_title(title,fontsize=9,fontweight='bold',pad=4)
    cax=fig.add_axes([.12,y,.76,.017]);cb=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(-limit,limit),cmap=cmap),cax=cax,orientation='horizontal');cb.set_ticks([-limit,0,limit]);cb.ax.tick_params(labelsize=6,length=2);cb.set_label(label,fontsize=7,labelpad=1);cb.outline.set_linewidth(.4)
save(fig,'within_and_cross_cohort_beta_maps')
internal_values=source.adjusted_TMS_beta_G2.to_numpy();cross_values=source.main_tms_minus_external_adjusted_beta.to_numpy()
fig,ax=plt.subplots(figsize=(3.3,4));ax.set_box_aspect(1)
limit=max(.62,float(np.max(np.abs(np.r_[internal_values,cross_values])))*1.12)
positive=(internal_values>0)&(cross_values>0);negative=(internal_values<0)&(cross_values<0)
colors=np.full(400,'#BFC4C8',dtype=object);colors[positive]='#B43C4A';colors[negative]='#2D6EA6'
ax.scatter(internal_values,cross_values,s=13,c=colors,alpha=.68,edgecolor='white',linewidth=.22,zorder=3)
ax.plot([-limit,limit],[-limit,limit],color='#A9A9A9',ls=(0,(2.2,2.2)),lw=.65)
fit=top.linregress(internal_values,cross_values);grid=np.linspace(internal_values.min(),internal_values.max(),240);line=fit.intercept+fit.slope*grid
residuals=cross_values-(fit.intercept+fit.slope*internal_values);se=np.sqrt(np.sum(residuals**2)/(len(internal_values)-2));critical=top.student_t.ppf(.975,len(internal_values)-2)
ci=critical*se*np.sqrt(1/len(internal_values)+(grid-internal_values.mean())**2/np.sum((internal_values-internal_values.mean())**2))
ax.fill_between(grid,line-ci,line+ci,color='#353535',alpha=.10,lw=0);ax.plot(grid,line,color='#353535',lw=1.15)
ax.axhline(0,color='#8E8E8E',lw=.55);ax.axvline(0,color='#8E8E8E',lw=.55)
ax.set(xlim=(-limit,limit),ylim=(-limit,limit),xlabel='Within-study adjusted TMS beta',ylabel='Cross-cohort adjusted group beta',title='Spatial correspondence across 400 ROIs')
p_spin=float(stats.spin_p_two_sided);spin_text='spin p < 0.001' if p_spin<.001 else f'spin p = {p_spin:.3f}'
ax.text(.03,.97,f'r = {float(stats.pearson_r):.3f}\n{spin_text}\nsame sign: {int((positive|negative).sum())}/400',transform=ax.transAxes,ha='left',va='top',fontsize=6,bbox=dict(facecolor='white',edgecolor='none',alpha=.9,pad=1.6))
ax.text(.78,.86,'both positive',transform=ax.transAxes,fontsize=5.5,color='#A63845',ha='center');ax.text(.28,.08,'both negative',transform=ax.transAxes,fontsize=5.5,color='#245F93',ha='center')
fig.tight_layout();save(fig,'beta_map_spatial_correspondence')
fig,ax=plt.subplots(figsize=(4.1,3.5));spin.draw_paired_spin_panel(ax,spin_stats[spin_stats.contrast.eq('Within-study')].reset_index(drop=True),within_null,spin_stats[spin_stats.contrast.eq('Cross-cohort')].reset_index(drop=True),cross_null);fig.subplots_adjust(left=.15,right=.985,bottom=.31,top=.78);save(fig,'within_and_cross_cohort_network_spin')
mapping={'o_clinical_score_pre_post':'clinical_score_pre_post','e_gradient_component_variance_blue_rose':'gradient_component_variance','j_G1_pre_gradient_surface':'G1_pre_surface','k_G1_post_gradient_surface':'G1_post_surface','l_G2_pre_gradient_surface':'G2_pre_surface','m_G2_post_gradient_surface':'G2_post_surface','a_G1_longitudinal_ROI_effects':'G1_longitudinal_effect','b_G2_longitudinal_ROI_effects':'G2_longitudinal_effect','d_seven_network_organization':'network_organization','f_exploratory_clinical_bridge':'clinical_gradient_association','g_whole_cortex_clinical_association':'cortical_clinical_associations','g_whole_cortex_clinical_association_network_glass_brains':'clinical_associations_by_network'}
for old,new in mapping.items():shutil.copy2(HERE.parent/'forai'/f'{old}.pdf',OUT/f'{new}.pdf')
print('FIG5 PANELS COMPLETE',flush=True)
