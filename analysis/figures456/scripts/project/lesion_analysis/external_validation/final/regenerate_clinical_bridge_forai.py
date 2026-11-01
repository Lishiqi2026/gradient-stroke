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
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from nilearn.plotting import plot_glass_brain
HERE = Path(__file__).resolve().parent
SOURCE_DIR = HERE.parent / 'ds003999_validation' / 'matched_gradient_figure'
SUBJECT_DATA = SOURCE_DIR / 'source_data' / 'selected_G2_clinical_partial_regression_data.csv'
MODEL_DATA = SOURCE_DIR / 'tables' / 'selected_exploratory_G2_clinical_model.csv'
OUTPUT_DIR = HERE / 'forai'
OUTPUT_PDF = OUTPUT_DIR / 'f_exploratory_clinical_bridge.pdf'
SCHAEFER400_ATLAS = HERE.parent.parent / 'resources' / 'Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz'
ROI_COLOR = '#7D6CA5'
ROI_LOCATOR_COLOR = '#A77B16'
CHARCOAL = '#272727'
MID = '#767676'
LIGHT = '#D8D8D8'

def apply_style() -> None:
    mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'], 'font.size': 8.2, 'axes.labelsize': 8.7, 'axes.titlesize': 10.5, 'xtick.labelsize': 8.0, 'ytick.labelsize': 8.0, 'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False, 'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'axes.unicode_minus': False, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})

def add_roi_glass_brain_inset(fig: plt.Figure, roi: int, roi_name: str) -> None:
    if not SCHAEFER400_ATLAS.is_file():
        raise FileNotFoundError(f'Required Schaefer400 atlas not found: {SCHAEFER400_ATLAS}')
    atlas_img = nib.load(SCHAEFER400_ATLAS)
    atlas_data = np.asarray(atlas_img.dataobj)
    roi_mask = atlas_data == roi
    if int(roi_mask.sum()) == 0:
        raise ValueError(f'ROI {roi} is not present in the Schaefer400 atlas')
    roi_centroid_mm = nib.affines.apply_affine(atlas_img.affine, np.argwhere(roi_mask)).mean(axis=0)
    roi_img = nib.Nifti1Image(roi_mask.astype(np.float32), atlas_img.affine)
    box_ax = fig.add_axes([0.655, 0.205, 0.3, 0.27], zorder=5)
    box_ax.set_facecolor('none')
    box_ax.set_xticks([])
    box_ax.set_yticks([])
    for spine in box_ax.spines.values():
        spine.set_color('#CFCFCF')
        spine.set_linewidth(0.65)
    box_ax.text(0.05, 0.92, f'ROI {roi}', ha='left', va='top', fontsize=6.8, fontweight='bold', color=CHARCOAL, transform=box_ax.transAxes)
    short_name = roi_name.replace('7Networks_LH_', 'LH ').replace('_', ' ')
    box_ax.text(0.5, 0.055, short_name, ha='center', va='bottom', fontsize=6.3, color=CHARCOAL, transform=box_ax.transAxes)
    glass_ax = fig.add_axes([0.668, 0.265, 0.278, 0.165], zorder=6)
    glass_ax.set_axis_off()
    roi_cmap = LinearSegmentedColormap.from_list('selected_roi', ['#F1E7C4', ROI_LOCATOR_COLOR])
    display = plot_glass_brain(roi_img, display_mode='ortho', figure=fig, axes=glass_ax, threshold=0.1, plot_abs=True, cmap=roi_cmap, colorbar=False, annotate=False, black_bg=False, alpha=0.52, resampling_interpolation='nearest')
    display.add_markers([roi_centroid_mm], marker_color=ROI_LOCATOR_COLOR, marker_size=28, edgecolors='white', linewidths=0.55, zorder=10)
    for cut_ax in display.axes.values():
        for image in cut_ax.ax.images:
            image.set_rasterized(True)

def make_figure(subjects: pd.DataFrame, selected: pd.Series) -> plt.Figure:
    x = subjects['adjusted_G2_delta_residual'].to_numpy(float)
    y = subjects['adjusted_improvement_residual'].to_numpy(float)
    beta = float(selected['standardized_beta'])
    ci_low = float(selected['hc3_ci_low'])
    ci_high = float(selected['hc3_ci_high'])
    (fig, ax) = plt.subplots(figsize=(4.7, 3.6))
    fig.subplots_adjust(left=0.16, right=0.97, top=0.89, bottom=0.19)
    x_pad = 0.045 * (x.max() - x.min())
    y_pad = 0.055 * (y.max() - y.min())
    x_limits = (float(x.min() - x_pad), float(x.max() + x_pad))
    y_limits = (float(y.min() - y_pad), float(y.max() + y_pad))
    xx = np.linspace(x_limits[0], x_limits[1], 200)
    low_line = ci_low * xx
    high_line = ci_high * xx
    ax.fill_between(xx, np.minimum(low_line, high_line), np.maximum(low_line, high_line), color=ROI_COLOR, alpha=0.1, linewidth=0, zorder=1)
    ax.plot(xx, beta * xx, color=CHARCOAL, linewidth=1.45, zorder=2)
    ax.scatter(x, y, s=43, color=ROI_COLOR, edgecolor='white', linewidth=0.7, alpha=0.92, zorder=3)
    ax.axhline(0, color=LIGHT, linewidth=0.7, zorder=0)
    ax.axvline(0, color=LIGHT, linewidth=0.7, zorder=0)
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_box_aspect(1)
    ax.set_xlabel('Adjusted G2 change\n(LH Default PFC 7)', labelpad=4)
    ax.set_ylabel('Adjusted Vasserman improvement', labelpad=5)
    ax.set_title('Association between G2 change\nand clinical improvement', loc='left', fontweight='bold', pad=6)
    ax.tick_params(length=3, width=0.8, pad=2)
    ax.text(0.035, 0.965, f"β={beta:.2f} [{ci_low:.2f}, {ci_high:.2f}]\npermutation p={float(selected['studentized_freedman_lane_p_two_sided']):.4f}; q={float(selected['fdr_q_400']):.3f}\nn={int(selected['n'])}; exploratory, data-selected ROI", transform=ax.transAxes, ha='left', va='top', fontsize=7.0, color=CHARCOAL)
    add_roi_glass_brain_inset(fig, roi=int(selected['roi']), roi_name=str(selected['roi_name']))
    return fig

def main() -> None:
    apply_style()
    for path in (SUBJECT_DATA, MODEL_DATA):
        if not path.is_file():
            raise FileNotFoundError(f'Required source data not found: {path}')
    subjects = pd.read_csv(SUBJECT_DATA)
    selected = pd.read_csv(MODEL_DATA).iloc[0]
    required_subject_columns = {'adjusted_G2_delta_residual', 'adjusted_improvement_residual'}
    missing = required_subject_columns.difference(subjects.columns)
    if missing:
        raise ValueError(f'Missing required subject columns: {sorted(missing)}')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = make_figure(subjects, selected)
    fig.savefig(OUTPUT_PDF, format='pdf', bbox_inches='tight', pad_inches=0.04, metadata={'Creator': 'Python / Matplotlib / Nilearn', 'Title': 'Association between G2 change and clinical improvement'})
    qa_png = Path(tempfile.gettempdir()) / 'f_exploratory_clinical_bridge_forai_qa.png'
    fig.savefig(qa_png, dpi=300, bbox_inches='tight', pad_inches=0.04)
    plt.close(fig)
    print(f'Saved: {OUTPUT_PDF}')
    print(f'QA preview: {qa_png}')
    print(f'Participants: {len(subjects)}')
    print(f"Selected ROI: {int(selected['roi'])} ({selected['roi_name']})")
if __name__ == '__main__':
    main()
