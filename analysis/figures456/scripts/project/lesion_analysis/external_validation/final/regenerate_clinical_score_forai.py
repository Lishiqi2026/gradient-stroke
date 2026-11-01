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
SOURCE_CSV = HERE.parent / 'ds003999_validation' / 'matched_gradient_figure' / 'source_data' / 'clinical_score_pre_post.csv'
SUMMARY_CSV = HERE.parent / 'ds003999_validation' / 'tables' / 'clinical_change_summary.csv'
OUTPUT_DIR = HERE / 'forai'
OUTPUT_PDF = OUTPUT_DIR / 'o_clinical_score_pre_post.pdf'
PRE_COLOR = '#3F6F8F'
POST_COLOR = '#B76E79'
PAIR_LINE = '#A9ADB2'
INK = '#282828'

def apply_style() -> None:
    mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'], 'font.size': 7.0, 'axes.labelsize': 7.0, 'axes.titlesize': 8.0, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'axes.linewidth': 0.65, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': INK, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK, 'xtick.direction': 'out', 'ytick.direction': 'out', 'xtick.major.size': 3.0, 'ytick.major.size': 3.0, 'xtick.major.width': 0.65, 'ytick.major.width': 0.65, 'legend.frameon': False, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'axes.unicode_minus': False})

def significance_stars(p_value: float) -> str:
    if p_value < 0.001:
        return '***'
    if p_value < 0.01:
        return '**'
    if p_value < 0.05:
        return '*'
    return 'ns'

def make_figure(data: pd.DataFrame, summary: pd.Series) -> plt.Figure:
    required = {'participant_id', 'pre_score', 'post_score'}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    frame = data.dropna(subset=['pre_score', 'post_score']).sort_values('participant_id')
    pre = frame['pre_score'].to_numpy(float)
    post = frame['post_score'].to_numpy(float)
    n = len(frame)
    if n != int(summary['n']):
        raise ValueError(f"Participant mismatch: source n={n}, summary n={int(summary['n'])}")
    (width_mm, height_mm) = (100.0, 76.0)
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), facecolor='white')
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.25, right=0.85, bottom=0.165, top=0.79)
    rng = np.random.default_rng(20260824)
    for (before, after) in zip(pre, post):
        jitter = float(rng.uniform(-0.035, 0.035))
        ax.plot([jitter, 1.0 + jitter], [before, after], color=PAIR_LINE, lw=0.82, alpha=0.52, zorder=1)
        ax.scatter(jitter, before, s=30, facecolor=PRE_COLOR, edgecolor=PRE_COLOR, linewidth=0.0, alpha=0.94, zorder=3)
        ax.scatter(1.0 + jitter, after, s=30, facecolor=POST_COLOR, edgecolor=POST_COLOR, linewidth=0.0, alpha=0.96, zorder=3)
    ax.set_xlim(-0.38, 1.38)
    ax.set_ylim(0.0, 64.0)
    ax.set_xticks([0.0, 1.0], ['Pre', 'Post'])
    ax.set_yticks(np.arange(0.0, 61.0, 10.0))
    ax.set_ylabel('Vasserman score')
    ax.set_title(f'External validation (n={n})', color=INK, fontweight='bold', fontsize=8.0, pad=8)
    ax.grid(False)
    bracket = 60.0
    cap = 58.4
    ax.plot([-0.1, -0.1, 1.1, 1.1], [cap, bracket, bracket, cap], color=INK, lw=0.78, clip_on=False, zorder=5)
    permutation_p = float(summary['improvement_signflip_p_two_sided'])
    ax.text(0.5, 61.0, significance_stars(permutation_p), ha='center', va='bottom', fontsize=8.0, fontweight='bold', color=INK)
    fig.text(0.16, 0.955, 'Longitudinal clinical-score change', ha='left', va='top', fontsize=8.2, fontweight='bold', color=INK)
    return fig

def main() -> None:
    apply_style()
    if not SOURCE_CSV.is_file():
        raise FileNotFoundError(f'Source data not found: {SOURCE_CSV}')
    if not SUMMARY_CSV.is_file():
        raise FileNotFoundError(f'Summary data not found: {SUMMARY_CSV}')
    data = pd.read_csv(SOURCE_CSV)
    summary = pd.read_csv(SUMMARY_CSV).iloc[0]
    fig = make_figure(data, summary)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PDF, format='pdf', bbox_inches='tight', pad_inches=0.035, facecolor='white', metadata={'Creator': 'Python / Matplotlib', 'Title': 'Longitudinal clinical-score change', 'Subject': 'Raw paired Vasserman scores in 29 external-validation participants'})
    qa_png = Path(tempfile.gettempdir()) / 'o_clinical_score_pre_post_forai_qa.png'
    fig.savefig(qa_png, format='png', dpi=300, bbox_inches='tight', pad_inches=0.035, facecolor='white')
    plt.close(fig)
    print(f'Saved: {OUTPUT_PDF}')
    print(f'QA preview: {qa_png}')
    print(f'Palette: Pre={PRE_COLOR}; Post={POST_COLOR}; paired line={PAIR_LINE}')
    print(f"n={int(summary['n'])}; paired t={summary['paired_t']:.6g}; permutation p={summary['improvement_signflip_p_two_sided']:.6g}")
if __name__ == '__main__':
    main()
