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
from matplotlib.patches import Patch
from nilearn import plotting
HERE = Path(__file__).resolve().parent
SOURCE_CSV = HERE.parent / 'ds003999_validation' / 'matched_gradient_figure' / 'tables' / 'clinical_roi_models_G1_G2.csv'
ATLAS_NIFTI = HERE.parent.parent / 'resources' / 'Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz'
OUTPUT_DIR = HERE / 'forai'
OUTPUT_PDF = OUTPUT_DIR / 'g_whole_cortex_clinical_association_network_glass_brains.pdf'
NETWORK_ORDER = ['Visual', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Control', 'Default']
NETWORK_ABBR = {'Visual': 'VIS', 'SomMot': 'SMN', 'DorsAttn': 'DAN', 'SalVentAttn': 'VAN', 'Limbic': 'LIM', 'Control': 'CON', 'Default': 'DMN'}
RED = '#B2182B'
BLUE = '#2166AC'
INK = '#282828'
MID = '#737373'

def apply_style() -> None:
    mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'], 'font.size': 7.0, 'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})

def validate_inputs(models: pd.DataFrame, atlas_data: np.ndarray) -> None:
    required = {'gradient', 'network', 'roi', 'standardized_beta', 'hc3_p_two_sided'}
    missing = required.difference(models.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    labels = np.unique(atlas_data.astype(int))
    expected = np.arange(0, 401)
    if not np.array_equal(labels, expected):
        raise ValueError('Expected atlas labels 0 through 400 exactly')

def roi_centroids_mm(atlas_img: nib.Nifti1Image, atlas_data: np.ndarray) -> dict[int, np.ndarray]:
    centroids: dict[int, np.ndarray] = {}
    for roi in range(1, 401):
        voxel_coordinates = np.argwhere(atlas_data == roi)
        if not len(voxel_coordinates):
            raise ValueError(f'ROI {roi} is absent from the atlas')
        centroids[roi] = nib.affines.apply_affine(atlas_img.affine, voxel_coordinates).mean(axis=0)
    return centroids

def marker_data(frame: pd.DataFrame, centroids: dict[int, np.ndarray], gradient: str, network: str) -> tuple[list[np.ndarray], list[str], int, int]:
    selected = frame.loc[frame['gradient'].eq(gradient) & frame['network'].eq(network) & frame['hc3_p_two_sided'].lt(0.05)]
    coordinates = [centroids[int(roi)] for roi in selected['roi']]
    colors = [RED if float(beta) > 0 else BLUE for beta in selected['standardized_beta']]
    positive_count = int((selected['standardized_beta'] > 0).sum())
    negative_count = int((selected['standardized_beta'] < 0).sum())
    return (coordinates, colors, positive_count, negative_count)

def make_figure(models: pd.DataFrame, centroids: dict[int, np.ndarray]) -> tuple[plt.Figure, dict[tuple[str, str], tuple[int, int]]]:
    (width_mm, height_mm) = (183.0, 88.0)
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), facecolor='white')
    fig.text(0.045, 0.965, 'Network-resolved clinical associations', ha='left', va='top', fontsize=10.0, fontweight='bold', color=INK)
    fig.text(0.045, 0.895, 'Nilearn sagittal and axial glass-brain views; spheres mark HC3 p<0.05 Schaefer-400 ROI centroids', ha='left', va='top', fontsize=6.3, color=MID)
    grid_left = 0.09
    grid_right = 0.995
    column_width = (grid_right - grid_left) / len(NETWORK_ORDER)
    row_y = {'G1': 0.515, 'G2': 0.215}
    row_height = 0.225
    counts: dict[tuple[str, str], tuple[int, int]] = {}
    for (column, network) in enumerate(NETWORK_ORDER):
        x0 = grid_left + column * column_width
        x_center = x0 + column_width / 2.0
        fig.text(x_center, 0.795, NETWORK_ABBR[network], ha='center', va='center', fontsize=7.0, fontweight='bold', color=INK)
        for (gradient, y0) in row_y.items():
            (coordinates, colors, n_positive, n_negative) = marker_data(models, centroids, gradient, network)
            counts[gradient, network] = (n_positive, n_negative)
            axis = fig.add_axes([x0 + 0.002, y0, column_width - 0.004, row_height])
            display = plotting.plot_glass_brain(None, figure=fig, axes=axis, display_mode='xz', colorbar=False, annotate=False, draw_cross=False, black_bg=False, alpha=0.34)
            if coordinates:
                display.add_markers(coordinates, marker_color=colors, marker_size=26, edgecolors='white', linewidths=0.55, alpha=0.96, zorder=10)
            if n_positive + n_negative == 0:
                fig.text(x_center, y0 + 0.015, 'none', ha='center', va='bottom', fontsize=5.2, color='#9A9A9A')
    for (gradient, y0) in row_y.items():
        fig.text(0.05, y0 + row_height / 2.0, gradient, ha='center', va='center', fontsize=8.2, fontweight='bold', color=INK)
    handles = [Patch(facecolor=RED, edgecolor='none', label='Positive standardized β'), Patch(facecolor=BLUE, edgecolor='none', label='Negative standardized β')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.53, 0.05), ncol=2, frameon=False, fontsize=6.2, handlelength=1.0, handleheight=0.8, columnspacing=1.8)
    return (fig, counts)

def main() -> None:
    apply_style()
    if not SOURCE_CSV.is_file():
        raise FileNotFoundError(f'Source data not found: {SOURCE_CSV}')
    if not ATLAS_NIFTI.is_file():
        raise FileNotFoundError(f'Schaefer atlas not found: {ATLAS_NIFTI}')
    models = pd.read_csv(SOURCE_CSV)
    atlas_img = nib.load(ATLAS_NIFTI)
    atlas_data = np.asarray(atlas_img.dataobj).astype(np.int16)
    validate_inputs(models, atlas_data)
    centroids = roi_centroids_mm(atlas_img, atlas_data)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (fig, counts) = make_figure(models, centroids)
    fig.savefig(OUTPUT_PDF, format='pdf', bbox_inches='tight', pad_inches=0.035, facecolor='white', metadata={'Creator': 'Python / Matplotlib / Nilearn', 'Title': 'Network-resolved clinical associations', 'Subject': 'Fourteen G1 and G2 network-resolved glass-brain maps'})
    qa_png = Path(tempfile.gettempdir()) / 'g_clinical_association_network_glass_brains_qa.png'
    fig.savefig(qa_png, format='png', dpi=300, bbox_inches='tight', pad_inches=0.035, facecolor='white')
    plt.close(fig)
    print(f'Saved: {OUTPUT_PDF}')
    print(f'QA preview: {qa_png}')
    for gradient in ('G1', 'G2'):
        summary = ', '.join((f'{NETWORK_ABBR[network]} +{counts[gradient, network][0]}/-{counts[gradient, network][1]}' for network in NETWORK_ORDER))
        print(f'{gradient}: {summary}')
if __name__ == '__main__':
    main()
