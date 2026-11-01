"""Render baseline MNI lesion-density maps separately for Control and TMS."""
from __future__ import annotations
import os as _portable_os
from pathlib import Path as _PortablePath

def _portable_path(kind, relative):
    project = _PortablePath(_portable_os.environ['PAPER_PROJECT_ROOT'])
    root = project if kind == 'project' else project.parent
    return str(root / relative)
import io
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import nibabel as nib
from nilearn import plotting
import numpy as np
import pandas as pd
from scipy.ndimage import center_of_mass
ROOT = Path(_portable_path('project', 'lesion_analysis'))
MASK_ROOT = ROOT / 'mni_registration' / 'masks'
MNI_PATH = ROOT / 'resources' / 'MNI152_T1_2mm.nii.gz'
FROZEN_NIFTI = ROOT / 'results_abcd' / 'nifti'
OUTPUT = ROOT / 'grouped_density_maps'
NIFTI_OUTPUT = OUTPUT / 'nifti'
GROUPS = ('Control', 'TMS')
EXPECTED_N = {'Control': 16, 'TMS': 22}
CUT_COORDS_Z = (-30, -15, 0, 15, 30, 45, 55)
CMAP = plt.get_cmap('OrRd')
matplotlib.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'], 'font.size': 7, 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.linewidth': 0.8})

def ensure_dirs() -> None:
    NIFTI_OUTPUT.mkdir(parents=True, exist_ok=True)

def save_formats(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUTPUT / f'{stem}.svg', bbox_inches='tight', facecolor='white')
    fig.savefig(OUTPUT / f'{stem}.pdf', bbox_inches='tight', facecolor='white')
    fig.savefig(OUTPUT / f'{stem}.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(OUTPUT / f'{stem}.tiff', dpi=600, bbox_inches='tight', facecolor='white', pil_kwargs={'compression': 'tiff_lzw'})

def load_density_maps(reference: nib.Nifti1Image) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, list[str]]]:
    counts: dict[str, np.ndarray] = {}
    percentages: dict[str, np.ndarray] = {}
    subject_ids: dict[str, list[str]] = {}
    for group in GROUPS:
        paths = sorted((MASK_ROOT / group / 'before').glob('*_lesion_mni2mm.nii.gz'))
        if len(paths) != EXPECTED_N[group]:
            raise RuntimeError(f'Expected {EXPECTED_N[group]} baseline {group} masks, found {len(paths)}')
        count = np.zeros(reference.shape, dtype=np.float32)
        ids = []
        for path in paths:
            image = nib.load(str(path))
            if image.shape != reference.shape or not np.allclose(image.affine, reference.affine):
                raise RuntimeError(f'MNI geometry mismatch: {path}')
            mask = np.asarray(image.dataobj) > 0.5
            if not mask.any():
                raise RuntimeError(f'Empty baseline lesion mask: {path}')
            count += mask.astype(np.float32)
            ids.append(path.name.split('_lesion_mni2mm')[0])
        percentage = count * (100.0 / len(paths))
        counts[group] = count
        percentages[group] = percentage
        subject_ids[group] = ids
        frozen = nib.load(str(FROZEN_NIFTI / f'baseline_overlap_{group.lower()}_count.nii.gz'))
        if frozen.shape != reference.shape or not np.allclose(frozen.affine, reference.affine):
            raise RuntimeError(f'Frozen overlap geometry mismatch for {group}')
        if not np.array_equal(count, np.asarray(frozen.dataobj, dtype=np.float32)):
            raise RuntimeError(f'Recomputed density does not match frozen A/B/C/D count map for {group}')
    return (counts, percentages, subject_ids)

def save_niftis(reference: nib.Nifti1Image, counts: dict[str, np.ndarray], percentages: dict[str, np.ndarray]) -> None:
    header = reference.header.copy()
    for group in GROUPS:
        group_key = group.lower()
        nib.save(nib.Nifti1Image(counts[group], reference.affine, header), str(NIFTI_OUTPUT / f'baseline_lesion_density_{group_key}_count.nii.gz'))
        nib.save(nib.Nifti1Image(percentages[group], reference.affine, header), str(NIFTI_OUTPUT / f'baseline_lesion_density_{group_key}_percent.nii.gz'))

def render_row(data: np.ndarray, reference: nib.Nifti1Image, background: nib.Nifti1Image, vmax: float) -> np.ndarray:
    stat_image = nib.Nifti1Image(data.astype(np.float32), reference.affine, reference.header)
    fig = plt.figure(figsize=(12.2, 2.25), facecolor='white')
    display = plotting.plot_stat_map(stat_image, bg_img=background, threshold=1e-06, cmap=CMAP, vmin=0, vmax=vmax, symmetric_cbar=False, display_mode='z', cut_coords=list(CUT_COORDS_Z), draw_cross=False, colorbar=False, black_bg=False, annotate=False, radiological=False, figure=fig, title=None)
    display.annotate(left_right=True, positions=False, size=7)
    for (coordinate, cut_axis) in display.axes.items():
        cut_axis.ax.set_title(f'z={int(round(float(coordinate)))} mm', fontsize=7, pad=2)
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buffer.seek(0)
    return mpimg.imread(buffer)

def plot_grouped_axial(percentages: dict[str, np.ndarray], reference: nib.Nifti1Image, background: nib.Nifti1Image, vmax: float) -> None:
    rendered = {group: render_row(percentages[group], reference, background, vmax) for group in GROUPS}
    (fig, axes) = plt.subplots(2, 1, figsize=(8.1, 4.15), facecolor='white')
    for (axis, group) in zip(axes, GROUPS):
        axis.imshow(rendered[group], aspect='auto')
        axis.axis('off')
        axis.text(-0.012, 0.5, f'{group}\n(n={EXPECTED_N[group]})', transform=axis.transAxes, ha='right', va='center', fontsize=8, fontweight='bold', clip_on=False)
    fig.subplots_adjust(left=0.1, right=0.89, top=0.88, bottom=0.07, hspace=0.13)
    colorbar_axis = fig.add_axes((0.92, 0.2, 0.018, 0.58))
    scalar = ScalarMappable(norm=Normalize(vmin=0, vmax=vmax), cmap=CMAP)
    scalar.set_array([])
    colorbar = fig.colorbar(scalar, cax=colorbar_axis)
    colorbar.set_label('Participants with lesion (%)', fontsize=7)
    colorbar.ax.tick_params(labelsize=6, width=0.6, length=2.5)
    fig.suptitle('Baseline lesion density by treatment group', fontsize=10, fontweight='bold', y=0.965)
    fig.text(0.5, 0.018, 'MNI152 2 mm; neurological orientation; common color scale', ha='center', fontsize=6.5, color='#444444')
    save_formats(fig, 'baseline_lesion_density_grouped_axial')
    plt.close(fig)

def plot_individual_group(group: str, data: np.ndarray, reference: nib.Nifti1Image, background: nib.Nifti1Image, vmax: float) -> None:
    stat_image = nib.Nifti1Image(data.astype(np.float32), reference.affine, reference.header)
    fig = plt.figure(figsize=(8.1, 2.5), facecolor='white')
    display = plotting.plot_stat_map(stat_image, bg_img=background, threshold=1e-06, cmap=CMAP, vmin=0, vmax=vmax, symmetric_cbar=False, display_mode='z', cut_coords=list(CUT_COORDS_Z), draw_cross=False, colorbar=True, cbar_tick_format='%.0f', black_bg=False, annotate=False, radiological=False, figure=fig, title=f'{group} baseline lesion density (n={EXPECTED_N[group]})')
    display.annotate(left_right=True, positions=False, size=7)
    for (coordinate, cut_axis) in display.axes.items():
        cut_axis.ax.set_title(f'z={int(round(float(coordinate)))} mm', fontsize=7, pad=2)
    display._cbar.set_label('Participants with lesion (%)', fontsize=7)
    save_formats(fig, f'baseline_lesion_density_{group.lower()}_axial')
    plt.close(fig)

def pooled_center(percentages: dict[str, np.ndarray], affine: np.ndarray) -> tuple[float, float, float]:
    pooled = percentages['Control'] + percentages['TMS']
    voxel_center = center_of_mass(pooled)
    return tuple((float(value) for value in nib.affines.apply_affine(affine, voxel_center)))

def render_ortho(data: np.ndarray, reference: nib.Nifti1Image, background: nib.Nifti1Image, vmax: float, coordinates: tuple[float, float, float]) -> np.ndarray:
    stat_image = nib.Nifti1Image(data.astype(np.float32), reference.affine, reference.header)
    fig = plt.figure(figsize=(5.8, 2.15), facecolor='white')
    plotting.plot_stat_map(stat_image, bg_img=background, threshold=1e-06, cmap=CMAP, vmin=0, vmax=vmax, symmetric_cbar=False, display_mode='ortho', cut_coords=coordinates, draw_cross=False, colorbar=False, black_bg=False, annotate=True, radiological=False, figure=fig, title=None)
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buffer.seek(0)
    return mpimg.imread(buffer)

def plot_grouped_ortho(percentages: dict[str, np.ndarray], reference: nib.Nifti1Image, background: nib.Nifti1Image, vmax: float) -> tuple[float, float, float]:
    coordinates = pooled_center(percentages, reference.affine)
    rendered = {group: render_ortho(percentages[group], reference, background, vmax, coordinates) for group in GROUPS}
    (fig, axes) = plt.subplots(2, 1, figsize=(5.2, 4.05), facecolor='white')
    for (axis, group) in zip(axes, GROUPS):
        axis.imshow(rendered[group], aspect='auto')
        axis.axis('off')
        axis.text(-0.02, 0.5, f'{group}\n(n={EXPECTED_N[group]})', transform=axis.transAxes, ha='right', va='center', fontsize=8, fontweight='bold', clip_on=False)
    fig.subplots_adjust(left=0.15, right=0.87, top=0.88, bottom=0.08, hspace=0.1)
    colorbar_axis = fig.add_axes((0.91, 0.2, 0.025, 0.58))
    scalar = ScalarMappable(norm=Normalize(vmin=0, vmax=vmax), cmap=CMAP)
    scalar.set_array([])
    colorbar = fig.colorbar(scalar, cax=colorbar_axis)
    colorbar.set_label('Participants with lesion (%)', fontsize=7)
    colorbar.ax.tick_params(labelsize=6, width=0.6, length=2.5)
    fig.suptitle('Baseline lesion density at the pooled center', fontsize=10, fontweight='bold', y=0.965)
    fig.text(0.5, 0.025, f'Common cut coordinates: x={coordinates[0]:.1f}, y={coordinates[1]:.1f}, z={coordinates[2]:.1f} mm', ha='center', fontsize=6.5, color='#444444')
    save_formats(fig, 'baseline_lesion_density_grouped_ortho')
    plt.close(fig)
    return coordinates

def write_summary(counts: dict[str, np.ndarray], percentages: dict[str, np.ndarray], subject_ids: dict[str, list[str]], reference: nib.Nifti1Image, vmax: float, coordinates: tuple[float, float, float]) -> None:
    voxel_ml = abs(float(np.linalg.det(reference.affine[:3, :3]))) / 1000.0
    rows = []
    for group in GROUPS:
        peak_voxel = np.unravel_index(np.argmax(percentages[group]), percentages[group].shape)
        peak_mni = nib.affines.apply_affine(reference.affine, peak_voxel)
        rows.append({'group': group, 'n': EXPECTED_N[group], 'maximum_overlap_count': int(np.max(counts[group])), 'maximum_overlap_pct': float(np.max(percentages[group])), 'union_volume_ml': float(np.count_nonzero(counts[group]) * voxel_ml), 'peak_mni_x': float(peak_mni[0]), 'peak_mni_y': float(peak_mni[1]), 'peak_mni_z': float(peak_mni[2]), 'shared_display_vmax_pct': vmax})
    pd.DataFrame(rows).to_csv(OUTPUT / 'grouped_density_summary.csv', index=False, encoding='utf-8-sig')
    manifest = {'analysis_date': '2026-08-03', 'input': 'final baseline lesion masks in MNI152 2 mm space', 'groups': {group: subject_ids[group] for group in GROUPS}, 'normalization': 'voxelwise lesion count divided by the number of participants in each group, multiplied by 100', 'axial_cut_coordinates_mni_z_mm': list(CUT_COORDS_Z), 'orthogonal_common_cut_coordinates_mni_mm': list(coordinates), 'shared_vmax_pct': vmax, 'orientation': 'neurological; radiological=False; L/R annotations generated by Nilearn', 'post_treatment_masks_used': False, 'validation': 'recomputed group count maps exactly matched the frozen A/B/C/D count NIfTI files', 'source_plotting_pattern': 'adapted from the provided Nilearn lesion-density workflow'}
    (OUTPUT / 'plot_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

def main() -> None:
    ensure_dirs()
    reference = nib.load(str(MNI_PATH))
    background = reference
    (counts, percentages, subject_ids) = load_density_maps(reference)
    save_niftis(reference, counts, percentages)
    vmax = float(max((np.max(percentages[group]) for group in GROUPS)))
    plot_grouped_axial(percentages, reference, background, vmax)
    for group in GROUPS:
        plot_individual_group(group, percentages[group], reference, background, vmax)
    coordinates = plot_grouped_ortho(percentages, reference, background, vmax)
    write_summary(counts, percentages, subject_ids, reference, vmax, coordinates)
    print(f"Control n={EXPECTED_N['Control']}; TMS n={EXPECTED_N['TMS']}")
    print(f'Shared vmax={vmax:.3f}%')
    print(f'Output: {OUTPUT}')
if __name__ == '__main__':
    main()
