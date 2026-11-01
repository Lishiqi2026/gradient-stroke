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
HERE = Path(__file__).resolve().parent
SOURCE_CSV = HERE.parent / 'ds003999_validation' / 'matched_gradient_figure' / 'tables' / 'clinical_roi_models_G1_G2.csv'
OUTPUT_DIR = HERE / 'forai'
OUTPUT_PDF = OUTPUT_DIR / 'g_whole_cortex_clinical_association.pdf'
NETWORK_ORDER = ['Visual', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Control', 'Default']
NETWORK_ABBR = {'Visual': 'VIS', 'SomMot': 'SMN', 'DorsAttn': 'DAN', 'SalVentAttn': 'VAN', 'Limbic': 'LIM', 'Control': 'CON', 'Default': 'DMN'}
RED = '#B2182B'
BLUE = '#2166AC'
GREY = '#C8C9CB'
MID = '#767676'
SEPARATOR = '#D8D8D8'
INK = '#272727'

def apply_style() -> None:
    mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'], 'font.size': 7.0, 'axes.labelsize': 8.0, 'axes.titlesize': 9.0, 'xtick.labelsize': 7.0, 'ytick.labelsize': 6.8, 'axes.linewidth': 0.75, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': INK, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK, 'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'axes.unicode_minus': False, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})

def order_models(models: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, list[float]]:
    network_rank = {network: index for (index, network) in enumerate(NETWORK_ORDER)}
    ordered = models.copy()
    ordered['network_rank'] = ordered['network'].map(network_rank)
    ordered['hemisphere_rank'] = ordered['hemisphere'].map({'LH': 0, 'RH': 1})
    ordered = ordered.sort_values(['gradient', 'network_rank', 'hemisphere_rank', 'roi'])
    first = ordered.loc[ordered['gradient'].eq('G1')].reset_index(drop=True)
    sequence = first['network'].to_numpy()
    boundaries = np.flatnonzero(sequence[1:] != sequence[:-1]) + 0.5
    centers: list[float] = []
    for network in NETWORK_ORDER:
        indices = np.flatnonzero(sequence == network)
        if not len(indices):
            raise ValueError(f'No ROIs found for network: {network}')
        centers.append(float(indices.mean()))
    return (ordered, boundaries, centers)

def draw_landscape(models: pd.DataFrame) -> plt.Figure:
    required = {'gradient', 'roi', 'network', 'hemisphere', 'standardized_beta', 'hc3_p_two_sided'}
    missing = required.difference(models.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    (ordered, boundaries, centers) = order_models(models)
    (width_mm, height_mm) = (120.0, 78.0)
    (fig, axes) = plt.subplots(2, 1, figsize=(width_mm / 25.4, height_mm / 25.4), sharex=True, gridspec_kw={'hspace': 0.1}, facecolor='white')
    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.18, top=0.83)
    for (index, (ax, gradient)) in enumerate(zip(axes, ('G1', 'G2'))):
        frame = ordered.loc[ordered['gradient'].eq(gradient)].reset_index(drop=True)
        x = np.arange(len(frame), dtype=float)
        beta = frame['standardized_beta'].to_numpy(float)
        nominal = frame['hc3_p_two_sided'].to_numpy(float) < 0.05
        positive = nominal & (beta > 0.0)
        negative = nominal & (beta < 0.0)
        ax.scatter(x[~nominal], beta[~nominal], s=10.5, color=GREY, alpha=0.7, edgecolor='none', zorder=1)
        ax.scatter(x[positive], beta[positive], s=23, color=RED, edgecolor='white', linewidth=0.35, zorder=3)
        ax.scatter(x[negative], beta[negative], s=23, color=BLUE, edgecolor='white', linewidth=0.35, zorder=3)
        ax.axhline(0.0, color=MID, linewidth=0.65, zorder=0)
        for boundary in boundaries:
            ax.axvline(boundary, color=SEPARATOR, linewidth=0.55, zorder=0)
        ax.set_xlim(-5.0, 404.0)
        ax.set_ylim(-0.68, 0.68)
        ax.set_yticks([-0.5, 0.0, 0.5])
        ax.set_ylabel(f'{gradient}  β', labelpad=4)
        ax.tick_params(axis='y', length=2.7, width=0.75, pad=2)
        ax.spines['bottom'].set_visible(False)
        if index == 0:
            ax.set_xticks(centers, [''] * len(centers))
            ax.tick_params(axis='x', bottom=True, length=3.0, width=0.75, pad=0)
            ax.text(0.99, 0.9, 'red/blue: HC3 p<0.05; grey: other ROIs', transform=ax.transAxes, ha='right', va='top', fontsize=6.0, color=MID, bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.86, 'pad': 0.6})
        else:
            ax.set_xticks(centers, [NETWORK_ABBR[network] for network in NETWORK_ORDER])
            ax.tick_params(axis='x', length=0, pad=2)
            ax.set_xlabel('Schaefer400 ROIs grouped by network', labelpad=3)
    fig.text(0.13, 0.965, 'Whole-cortex clinical-association landscape', ha='left', va='top', fontsize=9.5, fontweight='bold', color=INK)
    return fig

def main() -> None:
    apply_style()
    if not SOURCE_CSV.is_file():
        raise FileNotFoundError(f'Source data not found: {SOURCE_CSV}')
    models = pd.read_csv(SOURCE_CSV)
    fig = draw_landscape(models)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PDF, format='pdf', bbox_inches='tight', pad_inches=0.035, facecolor='white', metadata={'Creator': 'Python / Matplotlib', 'Title': 'Whole-cortex clinical-association landscape', 'Subject': 'HC3 clinical associations across 400 Schaefer ROIs for G1 and G2'})
    qa_png = Path(tempfile.gettempdir()) / 'g_whole_cortex_clinical_association_forai_qa.png'
    fig.savefig(qa_png, format='png', dpi=300, bbox_inches='tight', pad_inches=0.035, facecolor='white')
    plt.close(fig)
    print(f'Saved: {OUTPUT_PDF}')
    print(f'QA preview: {qa_png}')
    for gradient in ('G1', 'G2'):
        frame = models.loc[models['gradient'].eq(gradient)]
        nominal = frame['hc3_p_two_sided'].lt(0.05)
        positive = int((nominal & frame['standardized_beta'].gt(0)).sum())
        negative = int((nominal & frame['standardized_beta'].lt(0)).sum())
        print(f'{gradient}: positive={positive}; negative={negative}; total={positive + negative}')
if __name__ == '__main__':
    main()
