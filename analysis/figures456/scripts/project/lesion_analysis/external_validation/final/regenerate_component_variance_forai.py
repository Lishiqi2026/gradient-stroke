from __future__ import annotations
import os as _portable_os
from pathlib import Path as _PortablePath

def _portable_path(kind, relative):
    project = _PortablePath(_portable_os.environ['PAPER_PROJECT_ROOT'])
    root = project if kind == 'project' else project.parent
    return str(root / relative)
import argparse
import tempfile
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
HERE = Path(__file__).resolve().parent
SOURCE_DATA = HERE.parent / 'ds003999_validation' / 'matched_gradient_figure' / 'source_data' / 'component_variance_subject_level.csv'
OUTPUT_DIR = HERE / 'forai'
OUTPUT_PDF = OUTPUT_DIR / 'e_gradient_component_variance.pdf'
PRE_COLOR = '#FA8072'
POST_COLOR = '#FED8B0'
COORDINATED_PRE_COLOR = '#3F6F8F'
COORDINATED_POST_COLOR = '#B76E79'
AXIS_COLOR = '#222222'
PALETTES = {'reference': (PRE_COLOR, POST_COLOR), 'coordinated': (COORDINATED_PRE_COLOR, COORDINATED_POST_COLOR)}

def apply_style() -> None:
    mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'], 'font.size': 11.5, 'axes.labelsize': 14.0, 'axes.titlesize': 14.0, 'xtick.labelsize': 12.5, 'ytick.labelsize': 12.5, 'legend.fontsize': 12.0, 'axes.linewidth': 0.9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': AXIS_COLOR, 'axes.labelcolor': AXIS_COLOR, 'xtick.color': AXIS_COLOR, 'ytick.color': AXIS_COLOR, 'xtick.direction': 'out', 'ytick.direction': 'out', 'xtick.major.size': 3.5, 'ytick.major.size': 3.5, 'xtick.major.width': 0.9, 'ytick.major.width': 0.9, 'legend.frameon': False, 'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'axes.unicode_minus': False, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})

def ordered_components(frame: pd.DataFrame) -> list[str]:
    components = frame['component'].dropna().astype(str).unique().tolist()
    return sorted(components, key=lambda value: int(value.removeprefix('G')))

def summarize(frame: pd.DataFrame, phase: str, components: list[str]) -> tuple[np.ndarray, np.ndarray]:
    grouped = frame.loc[frame['phase'].eq(phase)].groupby('component', observed=True)['share_of_retained_G1_G5'].agg(['mean', 'std']).reindex(components)
    if grouped.isna().any().any():
        raise ValueError(f'Missing or insufficient values for phase={phase!r}')
    return (grouped['mean'].to_numpy(float), grouped['std'].to_numpy(float))

def make_figure(frame: pd.DataFrame, pre_color: str=PRE_COLOR, post_color: str=POST_COLOR) -> plt.Figure:
    components = ordered_components(frame)
    (pre_mean, pre_sd) = summarize(frame, 'pre', components)
    (post_mean, post_sd) = summarize(frame, 'post', components)
    x = np.arange(1, len(components) + 1, dtype=float)
    offset = 0.1
    (fig, ax) = plt.subplots(figsize=(4.0, 3.5))
    fig.subplots_adjust(left=0.235, right=0.955, bottom=0.19, top=0.865)
    for (xpos, mean, sd, color) in ((x - offset, pre_mean, pre_sd, pre_color), (x + offset, post_mean, post_sd, post_color)):
        ax.errorbar(xpos, mean, yerr=sd, fmt='D', linestyle='none', markersize=5.2, markerfacecolor=color, markeredgecolor=color, markeredgewidth=0.8, color=color, ecolor=color, elinewidth=1.15, capsize=0, zorder=3)
    ax.set_xlim(0.55, len(components) + 0.45)
    ax.set_ylim(0.0, 0.65)
    ax.set_xticks(x, [str(index) for index in range(1, len(components) + 1)])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6])
    ax.set_xlabel('Component', labelpad=5)
    ax.set_ylabel('Explained (λ / Σλ)', labelpad=6)
    ax.set_title('External validation', fontweight='bold', pad=8)
    ax.grid(False)
    handles = [Line2D([0], [0], marker='D', linestyle='none', markerfacecolor=color, markeredgecolor=color, markersize=5.5, label=label) for (label, color) in (('Pre', pre_color), ('Post', post_color))]
    ax.legend(handles=handles, loc='upper right', borderaxespad=0.25, handlelength=0.8, handletextpad=0.35, labelspacing=0.3)
    return fig

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Regenerate the external-validation component plot.')
    parser.add_argument('--palette', choices=sorted(PALETTES), default='reference', help='Color version to export (default: reference).')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    apply_style()
    if not SOURCE_DATA.is_file():
        raise FileNotFoundError(f'Source data not found: {SOURCE_DATA}')
    frame = pd.read_csv(SOURCE_DATA)
    required = {'participant_id', 'phase', 'component', 'share_of_retained_G1_G5'}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (pre_color, post_color) = PALETTES[args.palette]
    output_pdf = OUTPUT_PDF if args.palette == 'reference' else OUTPUT_DIR / 'e_gradient_component_variance_blue_rose.pdf'
    fig = make_figure(frame, pre_color=pre_color, post_color=post_color)
    fig.savefig(output_pdf, format='pdf', bbox_inches='tight', pad_inches=0.04, metadata={'Creator': 'Python / Matplotlib', 'Title': 'External validation gradient component variance', 'Subject': 'Mean plus or minus standard deviation across 29 paired participants'})
    qa_png = Path(tempfile.gettempdir()) / f'e_gradient_component_variance_{args.palette}_qa.png'
    fig.savefig(qa_png, dpi=300, bbox_inches='tight', pad_inches=0.04)
    plt.close(fig)
    print(f'Saved: {output_pdf}')
    print(f'Palette: Pre={pre_color}; Post={post_color}')
    print(f'QA preview: {qa_png}')
    print(f"Participants: {frame['participant_id'].nunique()}")
    print(f"Components: {', '.join(ordered_components(frame))}")
if __name__ == '__main__':
    main()
