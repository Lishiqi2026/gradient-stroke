from __future__ import annotations
import os as _portable_os
from pathlib import Path as _PortablePath

def _portable_path(kind, relative):
    project = _PortablePath(_portable_os.environ['PAPER_PROJECT_ROOT'])
    root = project if kind == 'project' else project.parent
    return str(root / relative)
import shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from PIL import Image
import make_matched_gradient_nature_figure as mainfig
HERE = Path(__file__).resolve().parent
FINAL = HERE.parent.parent / 'final'

def save_final(fig: plt.Figure, stem: str, dpi: int=600) -> None:
    FINAL.mkdir(parents=True, exist_ok=True)
    base = FINAL / stem
    fig.savefig(base.with_suffix('.svg'), bbox_inches='tight', pad_inches=0.04)
    fig.savefig(base.with_suffix('.pdf'), bbox_inches='tight', pad_inches=0.04)
    fig.savefig(base.with_suffix('.png'), dpi=600, bbox_inches='tight', pad_inches=0.04)
    tiff = base.with_suffix('.tiff')
    fig.savefig(tiff, dpi=dpi, bbox_inches='tight', pad_inches=0.04, pil_kwargs={'compression': 'tiff_lzw'})
    with Image.open(tiff) as image:
        image.convert('RGB').save(tiff, dpi=(dpi, dpi), compression='tiff_lzw')
    plt.close(fig)

def panel_a_b(roi_stats: pd.DataFrame, t_images: dict[str, Path], tmax: float) -> None:
    for (letter, gradient) in (('a', 'G1'), ('b', 'G2')):
        frame = roi_stats.query('gradient == @gradient')
        fig = plt.figure(figsize=(7.2, 5.0))
        grid = fig.add_gridspec(3, 3, height_ratios=[18, 1.0, 1.0], width_ratios=[1.0, 3.2, 1.0], left=0.025, right=0.975, top=0.93, bottom=0.07, hspace=0.28)
        ax = fig.add_subplot(grid[0, :])
        ax.imshow(plt.imread(t_images[gradient]), interpolation='none')
        ax.set_axis_off()
        ax.set_title(f'{gradient} longitudinal ROI effects', loc='left', fontweight='bold', pad=2)
        cax = fig.add_subplot(grid[1, 1])
        cb = fig.colorbar(mpl.cm.ScalarMappable(norm=TwoSlopeNorm(vmin=-tmax, vcenter=0, vmax=tmax), cmap=mainfig.T_CMAP), cax=cax, orientation='horizontal')
        cb.set_ticks([-tmax, 0, tmax])
        cb.set_label('Paired t (post − pre)', labelpad=2)
        cb.ax.tick_params(length=2.5, pad=1.5)
        cb.outline.set_linewidth(0.6)
        note_ax = fig.add_subplot(grid[2, :])
        note_ax.set_axis_off()
        note_ax.text(0.5, 0.5, f"Colored: two-sided paired p<0.05 (uncorrected); white: non-significant; {int(frame['nominal_p_lt_0_05'].sum())} colored ROIs; n=29", ha='center', va='center', fontsize=7.0, color=mainfig.MID)
        save_final(fig, f'{letter}_{gradient}_longitudinal_ROI_effects')

def add_fixed_colorbar(fig: plt.Figure, cax: plt.Axes, limit: float, label: str, orientation: str='horizontal') -> None:
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), cmap=mainfig.T_CMAP), cax=cax, orientation=orientation)
    cb.set_ticks([-limit, 0, limit])
    cb.set_label(label, labelpad=3)
    cb.ax.tick_params(length=2.5, pad=1.5)
    cb.outline.set_linewidth(0.6)

def panel_pre_post_surfaces(mean_images: dict[str, tuple[Path, Path, float]]) -> None:
    specifications = [('j', 'G1', 'Pre', 0), ('k', 'G1', 'Post', 1), ('l', 'G2', 'Pre', 0), ('m', 'G2', 'Post', 1)]
    for (letter, gradient, phase, index) in specifications:
        (pre_image, post_image, limit) = mean_images[gradient]
        image_path = (pre_image, post_image)[index]
        fig = plt.figure(figsize=(7.2, 4.8))
        grid = fig.add_gridspec(3, 3, height_ratios=[18, 1.0, 0.9], width_ratios=[1.0, 3.2, 1.0], left=0.025, right=0.975, top=0.93, bottom=0.07, hspace=0.28)
        ax = fig.add_subplot(grid[0, :])
        ax.imshow(plt.imread(image_path), interpolation='none')
        ax.set_axis_off()
        ax.set_title(f'{gradient} group mean — {phase}', loc='left', fontweight='bold', pad=2)
        add_fixed_colorbar(fig, fig.add_subplot(grid[1, 1]), limit, 'Mean raw gradient coordinate')
        note_ax = fig.add_subplot(grid[2, :])
        note_ax.set_axis_off()
        note_ax.text(0.5, 0.5, 'Group mean across 29 paired participants; G1 and G2 use one common color scale', ha='center', va='center', fontsize=7.0, color=mainfig.MID)
        save_final(fig, f'{letter}_{gradient}_{phase.lower()}_gradient_surface')

def panel_six_surface_plate(mean_images: dict[str, tuple[Path, Path, float]], t_images: dict[str, Path], tmax: float) -> None:
    limit = mean_images['G1'][2]
    fig = plt.figure(figsize=(7.2, 8.9))
    grid = fig.add_gridspec(3, 3, width_ratios=[1, 1, 0.055], height_ratios=[1, 1, 1], left=0.025, right=0.92, top=0.95, bottom=0.065, wspace=0.055, hspace=0.12)
    rows = [('Pre', {gradient: mean_images[gradient][0] for gradient in ('G1', 'G2')}), ('Post', {gradient: mean_images[gradient][1] for gradient in ('G1', 'G2')}), ('Post − pre', t_images)]
    for (row_index, (phase, images)) in enumerate(rows):
        for (column_index, gradient) in enumerate(('G1', 'G2')):
            ax = fig.add_subplot(grid[row_index, column_index])
            ax.imshow(plt.imread(images[gradient]), interpolation='none')
            ax.set_axis_off()
            ax.set_title(f'{gradient} — {phase}', loc='left', fontsize=8.5, fontweight='bold', pad=1)
    add_fixed_colorbar(fig, fig.add_subplot(grid[0:2, 2]), limit, 'Mean raw gradient coordinate', orientation='vertical')
    add_fixed_colorbar(fig, fig.add_subplot(grid[2, 2]), tmax, 'Paired t (post − pre)', orientation='vertical')
    fig.suptitle('Longitudinal G1 and G2 cortical gradients', x=0.025, y=0.992, ha='left', fontsize=10, fontweight='bold')
    fig.text(0.025, 0.022, 'Change maps: colored ROIs have two-sided paired p<0.05 (uncorrected); non-significant ROIs are white; n=29.', fontsize=6.8, color=mainfig.MID)
    save_final(fig, 'n_G1_G2_pre_post_change_six_surface_plate')

def panel_clinical_score(participants: pd.DataFrame, clinical_summary: pd.Series) -> None:
    (fig, ax) = plt.subplots(figsize=(4.2, 3.8))
    rng = np.random.default_rng(20260811)
    for row in participants.itertuples(index=False):
        jitter = float(rng.uniform(-0.035, 0.035))
        ax.plot([jitter, 1 + jitter], [row.pre_score, row.post_score], color='#B8BABD', lw=0.75, alpha=0.65, zorder=1)
        ax.scatter(jitter, row.pre_score, s=18, facecolor='white', edgecolor=mainfig.CHARCOAL, lw=0.75, zorder=2)
        ax.scatter(1 + jitter, row.post_score, s=18, facecolor=mainfig.POST, edgecolor='white', lw=0.45, zorder=2)
    means = participants[['pre_score', 'post_score']].mean().to_numpy(float)
    sems = participants[['pre_score', 'post_score']].sem().to_numpy(float)
    ci95 = sems * mainfig.t_dist.ppf(0.975, len(participants) - 1)
    for (x, mean, ci, face) in zip((0, 1), means, ci95, (mainfig.CHARCOAL, mainfig.POST)):
        ax.errorbar(x, mean, yerr=ci, fmt='D', ms=5.2, mfc=face, mec='white', mew=0.55, color=face, capsize=3, lw=1.25, zorder=4)
    ax.set_xticks([0, 1], ['Pre', 'Post'])
    ax.set_xlim(-0.32, 1.32)
    ax.set_ylabel('Vasserman score')
    ax.set_title('Clinical score reduction', loc='left', fontweight='bold')
    ax.text(0.98, 0.97, f"n={len(participants)}\nmean pre − post={clinical_summary['improvement_mean']:.2f}\npaired t={clinical_summary['paired_t']:.2f}\n$p_{{perm}}$={clinical_summary['improvement_signflip_p_two_sided']:.4f}", transform=ax.transAxes, ha='right', va='top', fontsize=7.0)
    ax.text(0.02, 0.02, 'Lower scores indicate less impairment', transform=ax.transAxes, ha='left', va='bottom', fontsize=6.6, color=mainfig.MID)
    ax.tick_params(axis='x', length=0)
    fig.subplots_adjust(left=0.17, right=0.96, top=0.9, bottom=0.15)
    save_final(fig, 'o_clinical_score_pre_post')

def panel_c(counts: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(6.4, 3.45))
    gs = fig.add_gridspec(1, 2, wspace=0.45)
    axes = [fig.add_subplot(gs[0, index], projection='polar') for index in range(2)]
    mainfig.plot_radar(axes[0], counts, 'G1', show_legend=True)
    mainfig.plot_radar(axes[1], counts, 'G2', show_legend=False)
    fig.suptitle('Nominal longitudinal ROI counts by network', x=0.08, y=0.985, ha='left', fontweight='bold', fontsize=10)
    fig.text(0.08, 0.925, 'Two-sided paired p<0.05 (uncorrected); counts are descriptive', fontsize=7, color=mainfig.MID)
    fig.subplots_adjust(top=0.82, bottom=0.16, left=0.07, right=0.97)
    save_final(fig, 'c_network_nominal_ROI_counts')

def panel_d(effects: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(7.2, 3.75))
    grid = fig.add_gridspec(2, 3, height_ratios=[15, 1.0], width_ratios=[1.0, 3.7, 1.0], left=0.23, right=0.98, top=0.91, bottom=0.12, hspace=0.38)
    ax = fig.add_subplot(grid[0, :])
    cax = fig.add_subplot(grid[1, 1])
    mainfig.plot_metric_heatmap(ax, effects, add_label=False, colorbar_ax=cax)
    save_final(fig, 'd_seven_network_organization')

def panel_e(component_long: pd.DataFrame, component_tests: pd.DataFrame) -> None:
    (fig, ax) = plt.subplots(figsize=(6.5, 3.7))
    mainfig.plot_component_variance(ax, component_long, component_tests, add_label=False)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.9, bottom=0.18)
    save_final(fig, 'e_gradient_component_variance')

def panel_f(clinical_subjects: pd.DataFrame, selected: pd.Series) -> None:
    (fig, ax) = plt.subplots(figsize=(5.7, 4.45))
    mainfig.plot_clinical(ax, clinical_subjects, selected, add_label=False)
    fig.subplots_adjust(left=0.15, right=0.97, top=0.9, bottom=0.17)
    save_final(fig, 'f_exploratory_clinical_bridge')

def panel_g(clinical_models: pd.DataFrame) -> None:
    (fig, axes) = plt.subplots(2, 1, figsize=(7.2, 4.2), sharex=True, gridspec_kw={'hspace': 0.13})
    mainfig.plot_roi_association_landscape(axes[0], axes[1], clinical_models, add_label=False)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.9, bottom=0.16)
    save_final(fig, 'g_whole_cortex_clinical_association')

def panel_h(ml_summary: pd.DataFrame) -> None:
    (fig, ax) = plt.subplots(figsize=(6.6, 4.2))
    mainfig.plot_external_ml(ax, ml_summary, add_label=False)
    for label in ax.get_yticklabels():
        label.set_fontsize(7.0)
    fig.subplots_adjust(left=0.3, right=0.97, top=0.9, bottom=0.18)
    save_final(fig, 'h_ML_MAE_performance')

def copy_r2_panel() -> None:
    source_stem = mainfig.FIGURES / 'ml_clinical_integrated_R2_performance'
    target_stem = FINAL / 'i_ML_R2_performance'
    for suffix in ('.svg', '.pdf', '.png', '.tiff'):
        shutil.copy2(source_stem.with_suffix(suffix), target_stem.with_suffix(suffix))

def main() -> None:
    mainfig.apply_style()
    FINAL.mkdir(parents=True, exist_ok=True)
    roi_stats = pd.read_csv(mainfig.TABLES / 'raw_aligned5_roi_paired_tests.csv')
    counts = pd.read_csv(mainfig.SOURCE / 'network_nominal_roi_counts.csv')
    effects = pd.read_csv(mainfig.TABLES / 'network_metric_effect_sizes.csv')
    component_long = pd.read_csv(mainfig.SOURCE / 'component_variance_subject_level.csv')
    component_tests = pd.read_csv(mainfig.TABLES / 'component_variance_paired_tests.csv')
    clinical_subjects = pd.read_csv(mainfig.SOURCE / 'selected_G2_clinical_partial_regression_data.csv')
    selected = pd.read_csv(mainfig.TABLES / 'selected_exploratory_G2_clinical_model.csv').iloc[0]
    clinical_models = pd.read_csv(mainfig.TABLES / 'clinical_roi_models_G1_G2.csv')
    ml_summary = pd.read_csv(mainfig.TABLES / 'ml_clinical_augmented_performance_summary.csv')
    (participants, _, pre, post) = mainfig.load_gradients()
    clinical_summary = pd.read_csv(mainfig.PARENT / 'tables' / 'clinical_change_summary.csv').iloc[0]
    (t_images, tmax) = mainfig.make_t_surfaces(roi_stats)
    mean_images = mainfig.make_mean_surfaces(pre, post)
    panel_a_b(roi_stats, t_images, tmax)
    panel_pre_post_surfaces(mean_images)
    panel_six_surface_plate(mean_images, t_images, tmax)
    panel_clinical_score(participants, clinical_summary)
    participants[['participant_id', 'pre_score', 'post_score']].to_csv(mainfig.SOURCE / 'clinical_score_pre_post.csv', index=False, encoding='utf-8-sig')
    panel_c(counts)
    panel_d(effects)
    panel_e(component_long, component_tests)
    panel_f(clinical_subjects, selected)
    panel_g(clinical_models)
    panel_h(ml_summary)
    copy_r2_panel()
    print(f'Standalone panels written to {FINAL}', flush=True)
if __name__ == '__main__':
    main()
