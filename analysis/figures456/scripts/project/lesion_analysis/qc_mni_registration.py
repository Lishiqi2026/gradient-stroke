from __future__ import annotations
import os as _portable_os
from pathlib import Path as _PortablePath

def _portable_path(kind, relative):
    project = _PortablePath(_portable_os.environ['PAPER_PROJECT_ROOT'])
    root = project if kind == 'project' else project.parent
    return str(root / relative)
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
ROOT = Path(_portable_path('work', ''))
ANALYSIS = ROOT / 'Gradient' / 'lesion_analysis'
REG = ANALYSIS / 'mni_registration'
RESOURCES = ANALYSIS / 'resources'
MNI_PATH = RESOURCES / 'MNI152_T1_2mm.nii.gz'
MNI_MASK_PATH = RESOURCES / 'MNI152_T1_2mm_brain_mask.nii.gz'

def load_data(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    image = nib.load(str(path))
    return (image, np.asarray(image.dataobj, dtype=np.float32))

def robust_scale(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = data[mask & np.isfinite(data)]
    if values.size == 0:
        return np.zeros_like(data)
    (low, high) = np.percentile(values, [1, 99])
    if high <= low:
        return np.zeros_like(data)
    return np.clip((data - low) / (high - low), 0, 1)

def dice(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    denom = int(a.sum() + b.sum())
    return float(2 * np.count_nonzero(a & b) / denom) if denom else np.nan

def masked_corr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    x = a[mask & np.isfinite(a) & np.isfinite(b)].astype(float)
    y = b[mask & np.isfinite(a) & np.isfinite(b)].astype(float)
    if x.size < 10 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])

def centroid_voxel(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.asarray(mask.shape, dtype=float) / 2
    return coords.mean(axis=0)

def show_plane(ax: plt.Axes, background: np.ndarray, axis: int, index: int, overlays: list[tuple[np.ndarray, str, float]], contours: list[tuple[np.ndarray, str]], title: str) -> None:

    def plane(arr: np.ndarray) -> np.ndarray:
        if axis == 0:
            return arr[index, :, :].T
        if axis == 1:
            return arr[:, index, :].T
        return arr[:, :, index].T
    ax.imshow(plane(background), cmap='gray', origin='lower', interpolation='nearest')
    for (overlay, color, alpha) in overlays:
        layer = np.ma.masked_where(plane(overlay) <= 0, plane(overlay))
        cmap = matplotlib.colors.ListedColormap([color])
        ax.imshow(layer, cmap=cmap, origin='lower', interpolation='nearest', alpha=alpha)
    for (contour, color) in contours:
        values = plane(contour)
        if np.any(values > 0) and np.any(values <= 0):
            ax.contour(values, levels=[0.5], colors=[color], linewidths=0.7)
    ax.set_title(title, fontsize=8)
    ax.set_axis_off()

def native_volume_ml(group: str, session: str, subject: str) -> float:
    matches = list((ROOT / 'Bingzao' / group / session).glob(f'{subject}-*.nii'))
    if not matches:
        return np.nan
    (image, data) = load_data(matches[0])
    voxel_volume = float(abs(np.linalg.det(image.affine[:3, :3])))
    return float(np.count_nonzero(data > 0.5) * voxel_volume / 1000)

def qc_subject(group: str, subject: str, mni: np.ndarray, mni_mask: np.ndarray) -> dict:
    work = REG / 'work' / group / subject
    before_path = REG / 'masks' / group / 'before' / f'{subject}_lesion_mni2mm.nii.gz'
    after_path = REG / 'masks' / group / 'after' / f'{subject}_lesion_mni2mm.nii.gz'
    t1_path = work / 't1_in_mni.nii.gz'
    brain_path = work / 't1_brain_mask_in_mni.nii.gz'
    jac_path = work / 't1_to_mni_jacobian.nii.gz'
    (_, t1) = load_data(t1_path)
    (_, warped_brain) = load_data(brain_path)
    (_, before) = load_data(before_path)
    after = load_data(after_path)[1] if after_path.exists() else np.zeros_like(before)
    (_, jac) = load_data(jac_path)
    ref_mask = mni_mask > 0.5
    brain_mask = warped_brain > 0.5
    before_mask = before > 0.5
    after_mask = after > 0.5
    common = ref_mask & brain_mask
    scaled_t1 = robust_scale(t1, brain_mask)
    scaled_mni = robust_scale(mni, ref_mask)
    center = np.rint(centroid_voxel(before_mask)).astype(int)
    center = np.clip(center, [0, 0, 0], np.asarray(before.shape) - 1)
    axes = [0, 1, 2]
    labels = ['Sagittal', 'Coronal', 'Axial']
    (fig, plot_axes) = plt.subplots(2, 3, figsize=(7.2, 4.5), facecolor='white')
    for (col, (axis, label)) in enumerate(zip(axes, labels)):
        show_plane(plot_axes[0, col], scaled_t1, axis, int(center[axis]), overlays=[], contours=[(ref_mask, '#E69F00'), (brain_mask, '#00A6A6')], title=f'{label}: T1 in MNI')
        show_plane(plot_axes[1, col], scaled_mni, axis, int(center[axis]), overlays=[(before_mask, '#D62728', 0.7), (after_mask, '#00A6A6', 0.45)], contours=[], title=f'{label}: lesion overlay')
    brain_dice = dice(ref_mask, brain_mask)
    template_corr = masked_corr(scaled_mni, scaled_t1, common)
    finite_jac = jac[np.isfinite(jac) & ref_mask]
    (jac_p01, jac_p99) = np.percentile(finite_jac, [1, 99]) if finite_jac.size else (np.nan, np.nan)
    fig.suptitle(f'{group} | subject {subject} | brain Dice={brain_dice:.3f} | template r={template_corr:.3f}\norange=MNI boundary, cyan=warped brain/post lesion, red=baseline lesion', fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    qc_dir = REG / 'qc' / group
    qc_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(qc_dir / f'{subject}_registration_qc.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    voxel_ml = 8.0 / 1000
    return {'group': group, 'subject_id': subject, 'brain_mask_dice': brain_dice, 'template_intensity_correlation': template_corr, 'jacobian_p01': float(jac_p01), 'jacobian_p99': float(jac_p99), 'jacobian_nonpositive_fraction': float(np.mean(finite_jac <= 0)) if finite_jac.size else np.nan, 'baseline_native_volume_ml': native_volume_ml(group, 'before', subject), 'baseline_mni_volume_ml': float(before_mask.sum() * voxel_ml), 'post_native_volume_ml': native_volume_ml(group, 'after', subject), 'post_mni_volume_ml': float(after_mask.sum() * voxel_ml) if after_path.exists() else np.nan, 'baseline_centroid_mni_x': float(nib.affines.apply_affine(nib.load(str(before_path)).affine, center)[0]), 'baseline_centroid_mni_y': float(nib.affines.apply_affine(nib.load(str(before_path)).affine, center)[1]), 'baseline_centroid_mni_z': float(nib.affines.apply_affine(nib.load(str(before_path)).affine, center)[2]), 'has_post_mask': bool(after_path.exists()), 'status': 'PASS' if brain_dice >= 0.7 and template_corr >= 0.45 else 'REVIEW'}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', help='Process one subject ID; default scans completed subjects')
    args = parser.parse_args()
    (_, mni) = load_data(MNI_PATH)
    (_, mni_mask) = load_data(MNI_MASK_PATH)
    rows = []
    for group in ('Control', 'TMS'):
        work_root = REG / 'work' / group
        if not work_root.exists():
            continue
        for work in sorted((path for path in work_root.iterdir() if path.is_dir())):
            if args.subject and work.name != args.subject:
                continue
            required = [work / '.done', work / 't1_in_mni.nii.gz', work / 't1_brain_mask_in_mni.nii.gz', work / 't1_to_mni_jacobian.nii.gz', REG / 'masks' / group / 'before' / f'{work.name}_lesion_mni2mm.nii.gz']
            if all((path.exists() for path in required)):
                rows.append(qc_subject(group, work.name, mni, mni_mask))
    summary = REG / 'registration_qc_summary.csv'
    if rows:
        with summary.open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f'QC subjects: {len(rows)}')
    print(f'Summary: {summary}')
    for row in rows:
        print(row['group'], row['subject_id'], row['status'], f"Dice={row['brain_mask_dice']:.3f}", f"r={row['template_intensity_correlation']:.3f}")
if __name__ == '__main__':
    main()
