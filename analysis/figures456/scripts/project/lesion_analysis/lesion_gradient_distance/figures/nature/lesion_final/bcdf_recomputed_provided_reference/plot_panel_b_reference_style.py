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
ROOT = Path(__file__).resolve().parent
SUBJECT_PATH = ROOT / 'recomputed_subject_metrics.csv'
MODEL_PATH = ROOT / 'recomputed_group_models.csv'
OUTPUT_STEM = ROOT / 'panel_b_recomputed_provided_reference'
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

def draw_violin_box_points(axis: plt.Axes, values: np.ndarray, position: float, dark: str, light: str) -> None:
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

def add_significance(axis: plt.Axes, q_value: float) -> None:
    bracket_y = 0.305
    cap_y = 0.293
    axis.plot([0.0, 0.0, 1.0, 1.0], [cap_y, bracket_y, bracket_y, cap_y], color=INK, linewidth=0.9, clip_on=False, zorder=8)
    axis.text(0.5, 0.316, significance_stars(q_value), ha='center', va='bottom', fontsize=9.0, fontweight='bold', color=INK)

def save_figure(fig: plt.Figure) -> None:
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
    model_lookup = models.loc[models['cohort'].eq('primary_motion_qc')].drop_duplicates('outcome').set_index('outcome')
    specs = [('mean_reference_directed_G2', 'G2'), ('mean_reference_directed_G1G2', 'G1–G2')]
    source_rows: list[pd.DataFrame] = []
    (width_mm, height_mm) = (120.0, 78.0)
    (fig, axes) = plt.subplots(1, 2, figsize=(width_mm / 25.4, height_mm / 25.4), sharey=True, facecolor='white')
    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.17, top=0.79, wspace=0.2)
    for (axis, (metric, title)) in zip(axes, specs):
        for (position, (group, dark, light)) in enumerate([('Control', CONTROL_DARK, CONTROL_LIGHT), ('TMS', TMS_DARK, TMS_LIGHT)]):
            values = primary.loc[primary['group'].eq(group), metric].to_numpy(float)
            draw_violin_box_points(axis, values, float(position), dark, light)
            source_rows.append(pd.DataFrame({'subject_id': primary.loc[primary['group'].eq(group), 'subject_id'].to_numpy(), 'group': group, 'metric': metric, 'value': values}))
        row = model_lookup.loc[metric]
        add_significance(axis, float(row['fdr_q_freedman_lane']))
        axis.axhline(0, color=ZERO, linewidth=0.7, zorder=0)
        axis.set_xlim(-0.58, 1.58)
        axis.set_ylim(-0.33, 0.39)
        axis.set_xticks([0, 1])
        axis.set_xticklabels([f"Control\n(n={(primary['group'] == 'Control').sum()})", f"TMS\n(n={(primary['group'] == 'TMS').sum()})"])
        axis.set_title(title, fontweight='bold', pad=5)
        axis.grid(False)
    axes[0].set_ylabel('Healthy-reference normalization')
    axes[1].spines['left'].set_visible(False)
    axes[1].tick_params(axis='y', left=False, labelleft=False)
    fig.text(0.13, 0.95, 'Whole-cortex normalization', ha='left', va='top', fontsize=8.5, fontweight='bold', color=INK)
    fig.text(0.985, 0.95, 'Positive values indicate movement toward the healthy reference', ha='right', va='top', fontsize=5.6, color=MID)
    pd.concat(source_rows, ignore_index=True).to_csv(ROOT / 'panel_b_recomputed_provided_reference_source_data.csv', index=False, encoding='utf-8-sig')
    model_lookup.loc[[metric for (metric, _) in specs]].reset_index().to_csv(ROOT / 'panel_b_recomputed_provided_reference_statistics.csv', index=False, encoding='utf-8-sig')
    save_figure(fig)
if __name__ == '__main__':
    main()
