from __future__ import annotations
import os as _portable_os
from pathlib import Path as _PortablePath

def _portable_path(kind, relative):
    project = _PortablePath(_portable_os.environ['PAPER_PROJECT_ROOT'])
    root = project if kind == 'project' else project.parent
    return str(root / relative)
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
PACKAGE = Path(__file__).resolve().parents[1]
SOURCE = PACKAGE / 'source_data'
FIGURES = PACKAGE / 'figures'
QA = PACKAGE / 'qa'
CONTROL = '#4C4B7F'
CONTROL_PRE = '#B8C5E5'
TMS = '#D64F45'
TMS_PRE = '#F2C4BE'
INK = '#282828'
MID = '#747474'
PAIR = '#A4A4A4'
GOLD = '#D59A28'
mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'], 'font.size': 7.0, 'axes.labelsize': 7.0, 'axes.titlesize': 8.0, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'axes.linewidth': 0.65, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'figure.facecolor': 'white', 'savefig.facecolor': 'white'})

def save_submission_figure(fig: plt.Figure, stem: str, *, close: bool=True) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    base = FIGURES / stem
    options = {'bbox_inches': 'tight', 'pad_inches': 0.035, 'facecolor': 'white'}
    fig.savefig(base.with_suffix('.pdf'), **options)
    fig.savefig(base.with_suffix('.svg'), **options)
    fig.savefig(base.with_suffix('.png'), dpi=600, **options)
    fig.savefig(base.with_suffix('.tiff'), dpi=600, pil_kwargs={'compression': 'tiff_lzw'}, **options)
    if close:
        plt.close(fig)
    for suffix in ('.png', '.tiff'):
        path = base.with_suffix(suffix)
        with Image.open(path) as source:
            rgb = Image.new('RGB', source.size, 'white')
            if source.mode == 'RGBA':
                rgb.paste(source, mask=source.getchannel('A'))
            else:
                rgb.paste(source.convert('RGB'))
            if suffix == '.tiff':
                rgb.save(path, dpi=(600, 600), compression='tiff_lzw')
            else:
                rgb.save(path, dpi=(600, 600))
