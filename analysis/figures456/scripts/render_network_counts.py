from pathlib import Path
import os
os.environ['MPLBACKEND']='Agg'
import numpy as np,pandas as pd,matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'rendered/figure5';OUT.mkdir(parents=True,exist_ok=True)
p=ROOT/'work/Gradient/lesion_analysis/external_validation/ds003999_validation/matched_gradient_figure/source_data/network_nominal_roi_counts.csv';data=pd.read_csv(p)
order=['Visual','SomMot','DorsAttn','SalVentAttn','Limbic','Control','Default'];abbr=['VIS','SMN','DAN','VAN','LIM','CON','DMN']
for g,negative,positive in [('G1','#B56591','#6A9A43'),('G2','#C48141','#7C67A1')]:
    d=data[data.gradient.eq(g)].set_index('network').reindex(order);fig,ax=plt.subplots(figsize=(2.5,2.5));y=np.arange(7)
    ax.barh(y,-d.nominal_negative_n,color=negative,height=.7);ax.barh(y,d.nominal_positive_n,color=positive,height=.7);ax.set_yticks(y,abbr);ax.invert_yaxis();ax.axvline(0,color='#B0B0B0',lw=.6);ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0);ax.tick_params(labelsize=8);ax.set_xlabel('Nominally significant ROIs',fontsize=8);ax.set_title('Negative t     Positive t',fontsize=8)
    fig.tight_layout();fig.savefig(OUT/f'{g}_network_ROI_counts.pdf',bbox_inches='tight',pad_inches=.03);fig.savefig(OUT/f'{g}_network_ROI_counts.png',bbox_inches='tight',pad_inches=.03,dpi=200);plt.close(fig)
print('NETWORK COUNT PANELS COMPLETE',flush=True)
