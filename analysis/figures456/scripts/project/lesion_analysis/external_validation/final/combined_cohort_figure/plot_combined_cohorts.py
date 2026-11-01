"""Redraw the supplied cohort forest plots from their saved numeric summaries."""
import os as _portable_os
from pathlib import Path as _PortablePath

def _portable_path(kind, relative):
    project = _PortablePath(_portable_os.environ['PAPER_PROJECT_ROOT'])
    root = project if kind == 'project' else project.parent
    return str(root / relative)
from pathlib import Path
import json
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import fitz
from PIL import Image
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
PRIMARY = ROOT / 'ml_analysis/outputs/five_outcome_lesion_informed/figures_nature/modelwise_five_tasks_forest/source_elasticnet_five_tasks_forest.csv'
EXTERNAL = OUT.parent / 'folder-forai/source_combined_models_primary_cohort.csv'
BASE = OUT / 'figure_primary_external_cohorts'
OUTCOME_COLORS = {'NIHSS': '#596E9A', 'ADL': '#72A5A0', 'MMSE': '#D1955B', 'HAMD': '#A97891', 'VFT': '#899A67'}
ALGORITHM_COLORS = {'ridge': '#596E9A', 'elasticnet': '#A97891'}
CONTRACT = 'Core purpose: Display predictive performance and split variability in the primary five-outcome cohort alongside the external Vasserman cohort.\nArchetype: Quantitative grid, two cohort panels with aligned forest plots and metric tables.\nStyle: Original outcome and algorithm colors, original pale grey group bands, colored group rules, round points and dark-edged best-model diamonds.\nRevision: Larger typography; panel subtitles and footer commentary removed from the artwork; shared legend centered. Statistical details remain in this accompanying note.\nBackend: Python only. Size: 183 x 210 mm; editable PDF/SVG, 600 dpi RGB TIFF, PNG preview.\nPanel a: Primary cohort n=37, Elastic Net, five outcomes and seven feature blocks per outcome; 50 repeats of 5-fold outer CV, 4-fold inner tuning.\nPanel b: External cohort n=29, Vasserman outcome, Ridge and Elastic Net, ten displayed feature blocks per algorithm; 20 repeats of 5-fold outer CV, 4-fold inner tuning.\nPoints: Mean post-score R2. Bars: 2.5th-97.5th repeat percentiles, descriptive split variability, not confidence intervals. Diamonds: Lowest mean MSE among displayed blocks within each outcome/algorithm. MSE and MAE minima independently bolded.\nSame R2 axis in both panels (-0.5 to 1); left arrows explicitly mark intervals extending below -0.5. Source tables retain full endpoints.\nAll 55 supplied rows retained, no model fitting, no changes to metric values. Each cohort retains its own feature definitions and CV design.\nInterpretation limits: Different outcomes, feature sets and repeat counts prevent direct raw-error comparisons between cohorts. External data were used for within-cohort nested CV, not evaluation of a frozen primary-cohort model. No clinical-only comparator is displayed; incremental benefit cannot be established from this figure. Best-block selection is exploratory.\n'

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'FIGURE_NOTES.txt').write_text(CONTRACT, encoding='utf-8')
    a = pd.read_csv(PRIMARY).sort_values('y_position')
    b = pd.read_csv(EXTERNAL).sort_values('y_position')
    assert len(a) == 35 and len(b) == 20
    assert set(a.algorithm) == {'elasticnet'}
    assert set(b.cohort) == {'primary_n29'} and set(b.n) == {29}
    assert set(a.n_repeats) == {50} and set(b.n_repeats) == {20}
    a.to_csv(OUT / 'source_primary.csv', index=False)
    b.to_csv(OUT / 'source_external.csv', index=False)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'], 'font.size': 7.2, 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.linewidth': 0.55, 'xtick.labelsize': 7.2, 'axes.spines.top': False, 'axes.spines.right': False})
    fig = plt.figure(figsize=(183 / 25.4, 210 / 25.4), facecolor='white')
    text_artists = []

    def txt(ax, x, y, s, **kw):
        artist = ax.text(x, y, s, **kw)
        text_artists.append(artist)
        return artist
    for (idx, (df, group_col, groups, palette)) in enumerate([(a, 'outcome', ['NIHSS', 'ADL', 'MMSE', 'HAMD', 'VFT'], OUTCOME_COLORS), (b, 'algorithm', ['ridge', 'elasticnet'], ALGORITHM_COLORS)]):
        left = 0.03 if idx == 0 else 0.525
        width = 0.45
        label_w = 0.2
        table_w = 0.099
        forest_w = width - label_w - table_w - 0.01
        lab = fig.add_axes([left, 0.095, label_w, 0.815])
        forest = fig.add_axes([left + label_w, 0.095, forest_w, 0.815])
        table = fig.add_axes([left + label_w + forest_w + 0.006, 0.095, table_w, 0.815])
        axes = (lab, forest, table)
        row_step = 1.0 if idx == 0 else 1.84
        cursor = 1.15
        plotted = []
        for (gi, group) in enumerate(groups):
            color = palette[group]
            rows = df.loc[df[group_col] == group]
            start = cursor
            end = start + (len(rows) - 1) * row_step
            if gi % 2 == 0:
                for ax in axes:
                    ax.axhspan(start - 0.48, end + 0.48, color='#F4F5F7', zorder=0)
            lab.plot([0.018, 0.018], [start - 0.36, end + 0.36], color=color, lw=2.0, solid_capstyle='round', clip_on=False)
            label = {'ridge': 'Ridge', 'elasticnet': 'Elastic Net'}.get(group, group)
            txt(lab, 0.015, start - 0.71, label, fontsize=8.2, fontweight='bold', va='center', color=color)
            for (j, (_, r)) in enumerate(rows.iterrows()):
                y = start + j * row_step
                best = np.isclose(r.mse_mean, rows.mse_mean.min())
                mae_best = np.isclose(r.mae_mean, rows.mae_mean.min())
                label = r.block_label.replace('seven-network', '7-network')
                label = textwrap.fill(label, width=24 if idx == 0 else 26, break_long_words=False)
                txt(lab, 0.065, y, label, fontsize=7.2, va='center', fontweight='bold' if best else 'normal', linespacing=1.05, color='#222222')
                (low, high) = (float(r.r2_p025), float(r.r2_p975))
                forest.plot([max(-0.5, low), min(1, high)], [y, y], color=color, lw=1.05, alpha=0.84, zorder=2)
                for val in [low, high]:
                    if -0.5 <= val <= 1:
                        forest.plot([val, val], [y - 0.13, y + 0.13], color=color, lw=0.65)
                if low < -0.5:
                    forest.plot(-0.475, y, marker='<', color=color, ms=3.1)
                forest.plot(r.r2_mean, y, marker='D' if best else 'o', ms=4.4 if best else 3.7, color=color, mec='#202020' if best else 'white', mew=0.55, zorder=3)
                for (x, key, is_best) in [(0.24, 'mse_mean', best), (0.79, 'mae_mean', mae_best)]:
                    txt(table, x, y, f'{r[key]:.2f}', ha='center', va='center', fontsize=7.2, fontweight='bold' if is_best else 'normal')
                plotted.append((group, r.block, y))
            cursor = end + 2.0 if idx == 0 else end + 3.2
        ymax = cursor - 1.35 if idx == 0 else cursor - 2.55
        for ax in axes:
            ax.set_ylim(ymax, -0.3)
            ax.set_yticks([])
        for ax in (lab, table):
            ax.set_xlim(0, 1)
            ax.set_axis_off()
        forest.set_xlim(-0.5, 1.02)
        forest.set_xticks([-0.5, 0, 0.5, 1])
        forest.set_xlabel('Mean post-score $R^2$', fontsize=7.6, labelpad=4)
        forest.spines['left'].set_visible(False)
        forest.tick_params(length=2.5, width=0.55)
        forest.set_axisbelow(True)
        forest.grid(axis='x', color='#E6E6E6', lw=0.55)
        forest.axvline(0, color='#8A8A8A', lw=0.8, ls=(0, (3, 2)))
        for (ax, x, title) in [(lab, 0.025, 'Feature set'), (forest, 0.5, 'Mean $R^2$ and range'), (table, 0.24, 'MSE'), (table, 0.79, 'MAE')]:
            txt(ax, x, 1.02, title, transform=ax.transAxes, ha='left' if ax is lab else 'center', fontsize=7.3, fontweight='bold')
        heading = 'Primary cohort (n = 37)' if idx == 0 else 'External cohort (n = 29)'
        fig.text(left - 0.022, 0.965, 'a' if idx == 0 else 'b', fontsize=11, fontweight='bold')
        fig.text(left + 0.01, 0.965, heading, fontsize=10, fontweight='bold')
    fig.legend(handles=[Line2D([], [], marker='o', color='#68717A', lw=0.8, markersize=3, label='Mean and 2.5–97.5% split range'), Line2D([], [], marker='D', color='#68717A', lw=0, markersize=3.5, label='Lowest MSE within group')], loc='lower center', bbox_to_anchor=(0.5, 0.012), ncol=2, frameon=False, fontsize=7.6, columnspacing=2.2)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = [t.get_text() for t in text_artists if not fig.bbox.contains(*t.get_window_extent(renderer).get_points()[0]) or not fig.bbox.contains(*t.get_window_extent(renderer).get_points()[1])]
    assert not outside, outside
    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(BASE.with_suffix('.' + ext), dpi=300, facecolor='white')
    fig.canvas.draw()
    fig.savefig(BASE.with_suffix('.tiff'), dpi=600, facecolor='white', pil_kwargs={'compression': 'tiff_lzw'})
    with Image.open(BASE.with_suffix('.tiff')) as im:
        rgb = im.convert('RGB')
    rgb.save(BASE.with_suffix('.tiff'), compression='tiff_lzw', dpi=(600, 600))
    plt.close(fig)
    with fitz.open(BASE.with_suffix('.pdf')) as doc:
        assert len(doc) == 1
        page = doc[0]
        text = page.get_text()
        assert all((word in text for word in ['Primary cohort', 'External cohort', 'NIHSS', 'ADL', 'MMSE', 'HAMD', 'VFT']))
        assert not any((word in text for word in ['Vasserman', 'repeats', 'Intervals describe', 'Both cohorts']))
        page.get_pixmap(matrix=fitz.Matrix(1.8, 1.8), alpha=False).save(OUT / 'preview.png')
        size = [round(page.rect.width * 25.4 / 72, 2), round(page.rect.height * 25.4 / 72, 2)]
    assert '<text' in BASE.with_suffix('.svg').read_text(encoding='utf-8')
    qa = {'status': 'PASS', 'rows_primary': len(a), 'rows_external': len(b), 'size_mm': size, 'source_values_preserved': True, 'editable_pdf_svg': True, 'text_outside_canvas': outside, 'truncated_intervals_marked': int((a.r2_p025 < -0.5).sum() + (b.r2_p025 < -0.5).sum()), 'backend': 'Python', 'tiff_dpi': 600, 'outcome_colors': OUTCOME_COLORS, 'algorithm_colors': ALGORITHM_COLORS}
    (OUT / 'QA.json').write_text(json.dumps(qa, indent=2), encoding='utf-8')
    print(json.dumps(qa))
if __name__ == '__main__':
    main()
