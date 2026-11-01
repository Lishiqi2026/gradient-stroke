from pathlib import Path
import os,sys
os.environ['MPLBACKEND']='Agg'
RUN=Path(__file__).resolve().parents[1];PROJECT=RUN/'work/Gradient';OUT=RUN/'rendered/figure6';OUT.mkdir(parents=True,exist_ok=True)
import numpy as np,pandas as pd,matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0,str(PROJECT/'ml_analysis'))
import plot_modelwise_five_task_forest as primary
ext=PROJECT/'lesion_analysis/external_validation/final/folder-forai';sys.path.insert(0,str(ext));import run_expanded_ml_forest as external
summary=primary.summarize_repeats(pd.read_csv(primary.RESULT_ROOT/'repeat_level_performance.csv'));a=summary[summary.algorithm.eq('elasticnet')].copy()
order=a.groupby('block').mse_rank_within_task.mean().sort_values().index.tolist();a['order']=a.block.map({n:i for i,n in enumerate(order)});a=a.sort_values(['outcome','order']);a.to_csv(OUT/'primary_forest_plot_data.csv',index=False)
b=pd.read_csv(ext/'ml_performance_summary.csv');b=b[b.cohort.eq('primary_n29')&b.block.isin(external.PLOT_BLOCK_ORDER)].copy();b['order']=b.block.map({n:i for i,n in enumerate(external.PLOT_BLOCK_ORDER)});b.to_csv(OUT/'external_forest_plot_data.csv',index=False)
n=pd.read_csv(primary.RESULT_ROOT/'five_outcome_baseline_feature_table.csv').subject_id.nunique()
def save(fig,name):
    fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight',pad_inches=.04);fig.savefig(OUT/f'{name}.png',bbox_inches='tight',pad_inches=.04,dpi=220);plt.close(fig)
def forest(frame,groups,column,colors,title,name):
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':7,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(4.0,7.6));lab=fig.add_axes([.015,.07,.48,.86]);ax=fig.add_axes([.50,.07,.29,.86]);table=fig.add_axes([.80,.07,.19,.86]);cursor=0;positions=[]
    step=1 if column=='outcome' else 1.75
    for gi,g in enumerate(groups):
        rows=frame[frame[column].eq(g)].sort_values('order');start=cursor;end=start+(len(rows)-1)*step;color=colors[g]
        if gi%2==0:
            for plot in [lab,ax,table]:plot.axhspan(start-.45,end+.45,color='#F3F4F6',zorder=0)
        lab.text(.02,start-.73,{'ridge':'Ridge','elasticnet':'Elastic Net'}.get(g,g),color=color,fontsize=8,fontweight='bold',va='center')
        lab.plot([.015,.015],[start-.25,end+.25],color=color,lw=1.7)
        for row in rows.itertuples():
            y=cursor;best=bool(np.isclose(row.mse_mean,rows.mse_mean.min()));label=getattr(row,'block_label',row.block)
            lab.text(.045,y,label,va='center',fontsize=6.5,fontweight='bold' if best else 'normal')
            lo=max(-.5,row.r2_p025);hi=min(1,row.r2_p975);mean=max(-.5,min(1,row.r2_mean));ax.plot([lo,hi],[y,y],color=color,lw=1.1);ax.scatter(mean,y,marker='D' if best else 'o',s=22 if best else 12,c=color,edgecolors='#303030' if best else 'none',linewidth=.5,zorder=3)
            if row.r2_p025<-.5:ax.annotate('',xy=(-.5,y),xytext=(-.36,y),arrowprops=dict(arrowstyle='->',lw=.6,color=color))
            table.text(.24,y,f'{row.mse_mean:.2f}',ha='center',va='center',fontsize=6.3,fontweight='bold' if best else 'normal');table.text(.78,y,f'{row.mae_mean:.2f}',ha='center',va='center',fontsize=6.3,fontweight='bold' if np.isclose(row.mae_mean,rows.mae_mean.min()) else 'normal');cursor+=step
        cursor+=1.25
    for plot in [lab,ax,table]:plot.set_ylim(cursor-.7,-1.8);plot.set_yticks([])
    lab.set_xlim(0,1);lab.set_axis_off();table.set_xlim(0,1);table.set_axis_off();table.text(.24,-1.2,'MSE ↓',ha='center',fontsize=7,fontweight='bold');table.text(.78,-1.2,'MAE ↓',ha='center',fontsize=7,fontweight='bold')
    ax.set_xlim(-.5,1);ax.set_xticks([-.5,0,.5,1]);ax.axvline(0,color='#B1B1B1',lw=.6,ls='--');ax.spines['left'].set_visible(False);ax.set_xlabel('Post-score $R^2$');fig.suptitle(title,fontsize=10,fontweight='bold',y=.98);save(fig,name)
forest(a,list(primary.OUTCOMES),'outcome',primary.OUTCOME_COLORS,f'Primary cohort (n={n})','primary_cohort_prediction')
forest(b,['ridge','elasticnet'],'algorithm',external.ALGORITHM_COLORS,'External cohort (n=29)','external_cohort_prediction')
fig,ax=plt.subplots(figsize=(5,.4));ax.set_axis_off();ax.legend(handles=[Line2D([0],[0],color='#777777',lw=1,marker='o',markersize=3,label='Mean and 2.5–97.5% repeat percentiles'),Line2D([0],[0],color='#777777',marker='D',markersize=4,lw=0,label='Lowest mean MSE')],loc='center',ncol=2,frameon=False,fontsize=7);save(fig,'prediction_legend')
print('FIG6 PANELS COMPLETE',n,flush=True)
