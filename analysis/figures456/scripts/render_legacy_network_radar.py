"""Render current G2 network counts with the original radar plotting routine."""
from pathlib import Path
import sys

RUN=Path(__file__).resolve().parent
if RUN.name=='scripts':RUN=RUN.parent
REL=Path('lesion_analysis/lesion_gradient_distance/figures/nature/lesion_final/forSubmit_lesion')

def render_network_radar(source,save):
    project=RUN/'scripts/project'
    if not project.exists():project=RUN/'work/Gradient'
    module_root=project/REL
    for directory in [module_root/'scripts',module_root/'figures']:
        sys.path.insert(0,str(directory))
    import revise_g2_network_radar as radar
    radar.SOURCE=Path(source)
    radar.save_submission_figure=lambda fig,stem:save(fig,'network_distribution')
    radar.main()

def main():
    import os
    os.environ['MPLBACKEND']='Agg'
    source=RUN/'work/Gradient'/REL/'source_data'
    if not (source/'G2_FDR_network_counts.csv').exists():
        source=RUN/'results/replay_data/Gradient'/REL/'source_data'
    output=RUN/'rendered/figure4';output.mkdir(parents=True,exist_ok=True)
    def save(fig,name):
        import matplotlib.pyplot as plt
        fig.savefig(output/(name+'.pdf'),bbox_inches='tight',pad_inches=.04,dpi=400)
        fig.savefig(output/(name+'.png'),bbox_inches='tight',pad_inches=.04,dpi=220)
        plt.close(fig)
    render_network_radar(source,save)
    print('ORIGINAL RADAR STYLE RENDERED WITH CURRENT COUNTS',flush=True)

if __name__=='__main__':main()
