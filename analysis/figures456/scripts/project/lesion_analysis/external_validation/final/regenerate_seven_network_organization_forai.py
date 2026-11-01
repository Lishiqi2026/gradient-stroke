from __future__ import annotations
import os as _portable_os
from pathlib import Path as _PortablePath

def _portable_path(kind, relative):
    project = _PortablePath(_portable_os.environ['PAPER_PROJECT_ROOT'])
    root = project if kind == 'project' else project.parent
    return str(root / relative)
import tempfile
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
HERE = Path(__file__).resolve().parent
SOURCE_DATA = HERE.parent / 'ds003999_validation' / 'matched_gradient_figure' / 'tables' / 'network_metric_effect_sizes.csv'
OUTPUT_DIR = HERE / 'forai'
OUTPUT_PDF = OUTPUT_DIR / 'd_seven_network_organization.pdf'
NETWORK_ORDER = ['Visual', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Control', 'Default']
NETWORK_LABELS = {'Visual': 'VIS', 'SomMot': 'SMN', 'DorsAttn': 'DAN', 'SalVentAttn': 'VAN', 'Limbic': 'LIM', 'Control': 'CON', 'Default': 'DMN'}
METRIC_STYLE = {'Dispersion': {'label': 'Disp.', 'color': '#3F6F8F', 'offset': -0.19}, 'Segregation': {'label': 'Segr.', 'color': '#A8BCC6', 'offset': 0.0}, 'betweenDisp': {'label': 'Inter-net', 'color': '#B76E79', 'offset': 0.19}}
METRICS = list(METRIC_STYLE)
ALPHA = 0.05
THRESHOLD = -np.log10(ALPHA)
CHARCOAL = '#272727'
MID = '#767676'

def apply_style() -> None:
    mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'], 'font.size': 8.3, 'axes.labelsize': 8.5, 'axes.titlesize': 9.5, 'xtick.labelsize': 8.0, 'ytick.labelsize': 8.5, 'axes.linewidth': 0.75, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False, 'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'axes.unicode_minus': False, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})

def plot_gradient(ax: plt.Axes, effects: pd.DataFrame, gradient: str, x_max: float, show_ylabels: bool) -> None:
    y_base = np.arange(len(NETWORK_ORDER), dtype=float)
    for metric in METRICS:
        style = METRIC_STYLE[metric]
        frame = effects.loc[effects['gradient'].eq(gradient) & effects['metric'].eq(metric)].set_index('network').reindex(NETWORK_ORDER)
        if frame['p_signflip_two_sided'].isna().any():
            raise ValueError(f'Incomplete p values for {gradient}, {metric}')
        p_values = frame['p_signflip_two_sided'].to_numpy(float)
        neglogp = -np.log10(np.clip(p_values, np.finfo(float).tiny, 1.0))
        ypos = y_base + float(style['offset'])
        ax.barh(ypos, neglogp, height=0.17, color=str(style['color']), edgecolor='none', zorder=2)
        for (x_value, y_value, p_value) in zip(neglogp, ypos, p_values):
            if p_value < ALPHA:
                ax.text(x_value + 0.045, y_value, '*', ha='left', va='center', fontsize=10.0, fontweight='bold', color=CHARCOAL, zorder=4)
    ax.axvline(THRESHOLD, color=CHARCOAL, linestyle=(0, (3, 2)), linewidth=0.8, zorder=1)
    ax.set_xlim(0.0, x_max)
    ax.set_ylim(-0.46, len(NETWORK_ORDER) - 0.54)
    ax.set_yticks(y_base, [NETWORK_LABELS[name] for name in NETWORK_ORDER])
    ax.tick_params(axis='y', length=0, pad=3, labelleft=show_ylabels)
    ax.tick_params(axis='x', length=2.5, width=0.75, pad=2)
    ax.spines['left'].set_color(CHARCOAL)
    ax.spines['bottom'].set_color(CHARCOAL)
    ax.spines['left'].set_linewidth(0.75)
    ax.spines['bottom'].set_linewidth(0.75)
    ax.set_title(gradient, fontweight='bold', pad=3)

def make_figure(effects: pd.DataFrame) -> plt.Figure:
    all_p = effects['p_signflip_two_sided'].to_numpy(float)
    max_neglogp = float(np.max(-np.log10(np.clip(all_p, np.finfo(float).tiny, 1.0))))
    x_max = max(2.25, np.ceil((max_neglogp + 0.08) * 20) / 20)
    (fig, axes) = plt.subplots(2, 1, figsize=(3.25, 3.85), sharex=True, sharey=True)
    plot_gradient(axes[0], effects, 'G1', x_max=x_max, show_ylabels=True)
    plot_gradient(axes[1], effects, 'G2', x_max=x_max, show_ylabels=True)
    axes[0].invert_yaxis()
    axes[0].tick_params(axis='x', labelbottom=False)
    handles = [Patch(facecolor=str(METRIC_STYLE[metric]['color']), edgecolor='none', label=str(METRIC_STYLE[metric]['label'])) for metric in METRICS]
    fig.suptitle('Seven-network organization', x=0.17, y=0.987, ha='left', fontsize=10.5, fontweight='bold', color=CHARCOAL)
    fig.text(0.17, 0.947, 'Two-sided sign-flip p; n=29; dashed line, p=0.05', ha='left', va='top', fontsize=7.4, color=MID)
    fig.supxlabel('−log10(p)', x=0.58, y=0.078, fontsize=8.5, color=CHARCOAL)
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.58, 0.008), ncol=3, fontsize=7.4, handlelength=1.1, handleheight=0.8, handletextpad=0.35, columnspacing=0.9, borderaxespad=0)
    fig.subplots_adjust(left=0.17, right=0.975, top=0.85, bottom=0.18, hspace=0.27)
    return fig

def main() -> None:
    apply_style()
    if not SOURCE_DATA.is_file():
        raise FileNotFoundError(f'Source data not found: {SOURCE_DATA}')
    effects = pd.read_csv(SOURCE_DATA)
    required = {'gradient', 'network', 'metric', 'p_signflip_two_sided'}
    missing = required.difference(effects.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = make_figure(effects)
    fig.savefig(OUTPUT_PDF, format='pdf', bbox_inches='tight', pad_inches=0.04, metadata={'Creator': 'Python / Matplotlib', 'Title': 'Seven-network organization'})
    qa_png = Path(tempfile.gettempdir()) / 'd_seven_network_organization_forai_qa.png'
    fig.savefig(qa_png, dpi=300, bbox_inches='tight', pad_inches=0.04)
    plt.close(fig)
    print(f'Saved: {OUTPUT_PDF}')
    print(f'QA preview: {qa_png}')
    print(f'Rows: {len(effects)}')
    print(f"Nominal p<0.05: {int((effects['p_signflip_two_sided'] < 0.05).sum())}")
    print(f"FDR q<0.05: {int((effects['fdr_q_42_exact_indicator_tests'] < 0.05).sum())}")
if __name__ == '__main__':
    main()
