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
import numpy as np
import pandas as pd
from PIL import Image
SOURCE_ROOT = Path(_portable_path('project', 'lesion_analysis/lesion_gradient_distance/figures/nature/lesion_final/bcdf_recomputed_provided_reference'))
OUTPUT_STEM = Path(__file__).resolve().parent / 'panel_d_recomputed_provided_reference'
SUBJECT_PATH = SOURCE_ROOT / 'recomputed_subject_metrics.csv'
MODEL_PATH = SOURCE_ROOT / 'recomputed_group_models.csv'
CONTROL_DARK = '#3F73A8'
CONTROL_LIGHT = '#A8C2DD'
TMS_DARK = '#D6534A'
TMS_LIGHT = '#F2B8B0'
INK = '#252525'
MID = '#6F6F6F'
ZERO = '#A8A8A8'
RNG = np.random.default_rng(20260822)
mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'], 'font.size': 7.0, 'axes.labelsize': 7.0, 'axes.titlesize': 8.0, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'axes.linewidth': 0.7, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'figure.facecolor': 'white', 'savefig.facecolor': 'white'})

def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(['true', '1', 'yes'])

def significance_stars(q_value: float) -> str:
    if q_value < 0.001:
        return '***'
    if q_value < 0.01:
        return '**'
    if q_value < 0.05:
        return '*'
    return 'ns'

def draw_violin_box_points(axis, values, position: float, dark: str, light: str) -> None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    violin = axis.violinplot(values, positions=[position], widths=0.76, showmeans=False, showmedians=False, showextrema=False, points=400, bw_method='scott')
    for body in violin['bodies']:
        body.set_facecolor(light)
        body.set_edgecolor(dark)
        body.set_linewidth(0.55)
        body.set_alpha(0.68)
    box = axis.boxplot(values, positions=[position], widths=0.23, patch_artist=True, showfliers=False, manage_ticks=False, whis=1.5, boxprops={'facecolor': 'white', 'edgecolor': dark, 'linewidth': 0.85}, medianprops={'color': dark, 'linewidth': 1.2}, whiskerprops={'color': dark, 'linewidth': 0.8}, capprops={'color': dark, 'linewidth': 0.8})
    for patch in box['boxes']:
        patch.set_alpha(0.92)
    jitter = RNG.uniform(-0.13, 0.13, size=len(values))
    axis.scatter(position + jitter, values, s=17, facecolor='white', edgecolor=dark, linewidth=0.7, alpha=0.68, zorder=5)

def save_figure(fig) -> None:
    options = {'bbox_inches': 'tight', 'pad_inches': 0.04, 'facecolor': 'white'}
    fig.savefig(OUTPUT_STEM.with_suffix('.pdf'), **options)
    fig.savefig(OUTPUT_STEM.with_suffix('.svg'), **options)
    fig.savefig(OUTPUT_STEM.with_suffix('.png'), dpi=600, **options)
    fig.savefig(OUTPUT_STEM.with_suffix('.tiff'), dpi=600, pil_kwargs={'compression': 'tiff_lzw'}, **options)
    plt.close(fig)
    for suffix in ('.png', '.tiff'):
        path = OUTPUT_STEM.with_suffix(suffix)
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

def main() -> None:
    subjects = pd.read_csv(SUBJECT_PATH, dtype={'subject_id': str})
    primary = subjects.loc[as_bool(subjects['primary_motion_eligible'])].copy()
    models = pd.read_csv(MODEL_PATH)
    metric = 'xtract_projected_G2_slope'
    model = models.loc[models['cohort'].eq('primary_motion_qc')].drop_duplicates('outcome').set_index('outcome').loc[metric]
    (width_mm, height_mm) = (72.0, 78.0)
    (fig, axis) = plt.subplots(figsize=(width_mm / 25.4, height_mm / 25.4), facecolor='white')
    fig.subplots_adjust(left=0.22, right=0.985, bottom=0.17, top=0.75)
    for (position, (group, dark, light)) in enumerate([('Control', CONTROL_DARK, CONTROL_LIGHT), ('TMS', TMS_DARK, TMS_LIGHT)]):
        values = primary.loc[primary['group'].eq(group), metric].to_numpy(float)
        draw_violin_box_points(axis, values, float(position), dark, light)
    q_value = float(model['fdr_q_freedman_lane'])
    bracket_y = 0.148
    cap_y = 0.141
    axis.plot([0.0, 0.0, 1.0, 1.0], [cap_y, bracket_y, bracket_y, cap_y], color=INK, linewidth=0.9, clip_on=False, zorder=8)
    axis.text(0.5, 0.153, significance_stars(q_value), ha='center', va='bottom', fontsize=9.0, fontweight='bold', color=INK)
    n_control = int(primary['group'].eq('Control').sum())
    n_tms = int(primary['group'].eq('TMS').sum())
    axis.axhline(0, color=ZERO, linewidth=0.7, zorder=0)
    axis.set_xlim(-0.58, 1.58)
    axis.set_ylim(-0.24, 0.17)
    axis.set_xticks([0, 1])
    axis.set_xticklabels([f'Control\n(n={n_control})', f'TMS\n(n={n_tms})'])
    axis.set_ylabel('Individual G2 normalization–distance slope')
    axis.grid(False)
    fig.text(0.22, 0.95, 'Individual G2 distance–normalization slopes', ha='left', va='top', fontsize=8.5, fontweight='bold', color=INK)
    fig.text(0.22, 0.89, 'Permutation-based inference; FDR corrected', ha='left', va='top', fontsize=5.6, color=MID)
    save_figure(fig)
if __name__ == '__main__':
    main()
