from __future__ import annotations
import os as _portable_os
from pathlib import Path as _PortablePath

def _portable_path(kind, relative):
    project = _PortablePath(_portable_os.environ['PAPER_PROJECT_ROOT'])
    root = project if kind == 'project' else project.parent
    return str(root / relative)
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import openpyxl
ROOT = Path(_portable_path('work', ''))
ANALYSIS = ROOT / 'Gradient' / 'lesion_analysis'
REG = ANALYSIS / 'mni_registration'
CLINICAL_PATH = ROOT / '董分组.xlsx'

def load_data(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    image = nib.load(str(path))
    return (image, np.asarray(image.dataobj, dtype=np.float32))

def robust_scale(data: np.ndarray, mask: np.ndarray | None=None) -> np.ndarray:
    valid = np.isfinite(data)
    if mask is not None:
        valid &= mask
    values = data[valid]
    if values.size == 0:
        return np.zeros_like(data, dtype=np.float32)
    (low, high) = np.percentile(values, [1, 99])
    if high <= low:
        return np.zeros_like(data, dtype=np.float32)
    return np.clip((data - low) / (high - low), 0, 1).astype(np.float32)

def normalized_mutual_information(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    x = a[mask & np.isfinite(a) & np.isfinite(b)]
    y = b[mask & np.isfinite(a) & np.isfinite(b)]
    if x.size < 100:
        return np.nan
    (hist, _, _) = np.histogram2d(x, y, bins=64, range=((0, 1), (0, 1)))
    pxy = hist / hist.sum()
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    nz = pxy > 0
    denom = px[:, None] * py[None, :]
    mi = float(np.sum(pxy[nz] * np.log(pxy[nz] / denom[nz])))
    hx = float(-np.sum(px[px > 0] * np.log(px[px > 0])))
    hy = float(-np.sum(py[py > 0] * np.log(py[py > 0])))
    return float(mi / np.sqrt(hx * hy)) if hx > 0 and hy > 0 else np.nan

def clinical_rows() -> dict[str, dict[str, object]]:
    workbook = openpyxl.load_workbook(CLINICAL_PATH, data_only=True, read_only=True)
    sheet = workbook[workbook.sheetnames[0]]
    rows = list(sheet.iter_rows(values_only=True))
    headers = [str(value) for value in rows[0]]
    result = {}
    for values in rows[1:]:
        if values[0] is None:
            continue
        result[str(values[0])] = dict(zip(headers, values))
    workbook.close()
    return result

def read_qc_summary() -> list[dict[str, str]]:
    with (REG / 'registration_qc_summary.csv').open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))

def native_mask(group: str, session: str, subject: str) -> Path | None:
    matches = sorted((ROOT / 'Bingzao' / group / session).glob(f'{subject}-*.nii'))
    return matches[0] if matches else None

def laterality_metrics(group: str, subject: str) -> dict[str, object]:
    path = REG / 'masks' / group / 'before' / f'{subject}_lesion_mni2mm.nii.gz'
    (image, data) = load_data(path)
    coords = np.argwhere(data > 0.5)
    xyz = nib.affines.apply_affine(image.affine, coords)
    left = float(np.mean(xyz[:, 0] < -1))
    right = float(np.mean(xyz[:, 0] > 1))
    midline = float(1 - left - right)
    if left >= 0.8:
        computed = 'left'
    elif right >= 0.8:
        computed = 'right'
    elif left >= 0.1 and right >= 0.1:
        computed = 'bilateral'
    else:
        computed = 'midline'
    return {'mni_centroid_x': float(np.mean(xyz[:, 0])), 'mni_centroid_y': float(np.mean(xyz[:, 1])), 'mni_centroid_z': float(np.mean(xyz[:, 2])), 'mni_left_fraction': left, 'mni_right_fraction': right, 'mni_midline_fraction': midline, 'mask_derived_side': computed}

def laterality_status(clinical_side: str, location: str, metrics: dict[str, object]) -> str:
    expected = {'左': 'left', '右': 'right'}.get(clinical_side, 'unknown')
    computed = str(metrics['mask_derived_side'])
    centroid_x = float(metrics['mni_centroid_x'])
    is_brainstem = any((term in location for term in ('脑桥', '桥脑', '延髓', '中脑')))
    is_described_bilateral = any((term in location for term in ('双', '两侧', '双侧')))
    if is_described_bilateral:
        return 'BILATERAL_DESCRIPTION'
    if is_brainstem and abs(centroid_x) <= 6:
        return 'MIDLINE_BRAINSTEM_CAUTION'
    if expected in ('left', 'right') and computed != expected:
        return 'CLINICAL_MASK_DISCREPANCY'
    return 'MATCH'

def plane(data: np.ndarray, axis: int, index: int) -> np.ndarray:
    if axis == 0:
        return data[index, :, :].T
    if axis == 1:
        return data[:, index, :].T
    return data[:, :, index].T

def proxy_plane(proxy: object, axis: int, index: int) -> np.ndarray:
    if axis == 0:
        values = proxy[index, :, :]
    elif axis == 1:
        values = proxy[:, index, :]
    else:
        values = proxy[:, :, index]
    return np.asarray(values, dtype=np.float32).T

def checkerboard(a: np.ndarray, b: np.ndarray, tile: int=12) -> np.ndarray:
    (yy, xx) = np.indices(a.shape)
    choose_b = (yy // tile + xx // tile) % 2 == 1
    return np.where(choose_b, b, a)

def qc_t2_to_t1(group: str, subject: str) -> dict[str, object]:
    work = REG / 'work' / group / subject
    out_dir = REG / 'qc_t2_to_t1' / group
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_path = out_dir / f'{subject}_t2_to_t1_qc.png'
    cache_path = out_dir / f'{subject}_t2_to_t1_metrics.json'
    if figure_path.exists() and cache_path.exists():
        with cache_path.open(encoding='utf-8') as stream:
            return json.load(stream)
    t1_path = native_mask(group, 'before', subject)
    if t1_path is None:
        raise FileNotFoundError(f'Missing native baseline lesion for {group}/{subject}')
    source_stem = t1_path.stem
    source_t1 = ROOT / 'T1w' / f'{source_stem}.nii'
    t1_image = nib.load(str(source_t1))
    t1_brain_image = nib.load(str(work / 't1_brain_mask.nii.gz'))
    (_, lesion_t1) = load_data(work / 'lesion_before_in_t1.nii.gz')
    center = np.rint(np.argwhere(lesion_t1 > 0.5).mean(axis=0)).astype(int)
    center = np.clip(center, [0, 0, 0], np.asarray(t1_image.shape) - 1)
    sample = (slice(None, None, 4), slice(None, None, 4), slice(None, None, 2))
    t1_sample = np.asarray(t1_image.dataobj[sample], dtype=np.float32)
    brain_sample = np.asarray(t1_brain_image.dataobj[sample], dtype=np.float32) > 0.5
    t1_sample_scaled = robust_scale(t1_sample, brain_sample)
    session_data: dict[str, dict[str, object] | None] = {}
    for session in ('before', 'after'):
        image_path = work / f't2_{session}_in_t1.nii.gz'
        matrix_path = work / f't2_{session}_to_t1.mat'
        if not image_path.exists() or not matrix_path.exists():
            session_data[session] = None
            continue
        t2_image = nib.load(str(image_path))
        t2_sample = np.asarray(t2_image.dataobj[sample], dtype=np.float32)
        t2_support = np.isfinite(t2_sample) & (t2_sample != 0)
        overlap = brain_sample & t2_support
        t2_sample_scaled = robust_scale(t2_sample, overlap)
        matrix = np.loadtxt(matrix_path)
        session_data[session] = {'image': t2_image, 'nmi': normalized_mutual_information(t1_sample_scaled, t2_sample_scaled, overlap), 'determinant': float(np.linalg.det(matrix[:3, :3])), 'reflection': bool(np.linalg.det(matrix[:3, :3]) < 0)}
    axes = [0, 1, 2]
    names = ['Sagittal', 'Coronal', 'Axial']
    (fig, plot_axes) = plt.subplots(2, 3, figsize=(7.2, 4.7), facecolor='white')
    for (row, session) in enumerate(('before', 'after')):
        values = session_data[session]
        for (col, (axis, name)) in enumerate(zip(axes, names)):
            ax = plot_axes[row, col]
            t1_slice_raw = proxy_plane(t1_image.dataobj, axis, int(center[axis]))
            brain_slice = proxy_plane(t1_brain_image.dataobj, axis, int(center[axis])) > 0.5
            t1_slice = robust_scale(t1_slice_raw, brain_slice)
            if values is None:
                ax.imshow(t1_slice, cmap='gray', origin='lower', interpolation='nearest')
                ax.text(0.5, 0.5, 'Post T2 unavailable', transform=ax.transAxes, ha='center', va='center', fontsize=8, color='#A23B3B', bbox={'facecolor': 'white', 'alpha': 0.8, 'edgecolor': 'none'})
            else:
                t2_slice_raw = proxy_plane(values['image'].dataobj, axis, int(center[axis]))
                t2_slice = robust_scale(t2_slice_raw, (t2_slice_raw != 0) & brain_slice)
                ax.imshow(checkerboard(t1_slice, t2_slice), cmap='gray', origin='lower', interpolation='nearest')
                lesion_slice = plane(lesion_t1 > 0.5, axis, int(center[axis]))
                if np.any(lesion_slice) and np.any(~lesion_slice):
                    ax.contour(lesion_slice, levels=[0.5], colors=['#D62728'], linewidths=0.8)
            ax.set_title(f'{session.capitalize()} T2 to T1 | {name}', fontsize=8)
            ax.set_axis_off()
    before = session_data['before']
    after = session_data['after']
    before_text = 'missing' if before is None else f"NMI={before['nmi']:.3f}, det={before['determinant']:.3f}"
    after_text = 'missing' if after is None else f"NMI={after['nmi']:.3f}, det={after['determinant']:.3f}"
    fig.suptitle(f'{group} | subject {subject} | multimodal rigid-registration QC\nbefore: {before_text} | after: {after_text} | red=baseline lesion', fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(figure_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    metrics = {'t2_before_to_t1_nmi': np.nan if before is None else before['nmi'], 't2_before_to_t1_determinant': np.nan if before is None else before['determinant'], 't2_before_reflection_detected': 'NA' if before is None else before['reflection'], 't2_after_to_t1_nmi': np.nan if after is None else after['nmi'], 't2_after_to_t1_determinant': np.nan if after is None else after['determinant'], 't2_after_reflection_detected': 'NA' if after is None else after['reflection']}
    with cache_path.open('w', encoding='utf-8') as stream:
        json.dump(metrics, stream, ensure_ascii=True, indent=2)
    return metrics

def manual_flags(subject: str, laterality: str, post_native_volume: float, has_post: bool) -> tuple[str, str, str]:
    reasons = []
    if subject == 'PRIVATE_SUBJECT_040':
        reasons.append('large hemispheric lesion; inspect nonlinear boundary fit')
    if laterality == 'CLINICAL_MASK_DISCREPANCY':
        reasons.append('clinical side conflicts with source and MNI mask')
    if subject == 'PRIVATE_SUBJECT_038':
        reasons.append('bilateral clinical description but lesion volume is left-dominant')
    if has_post and np.isfinite(post_native_volume) and (post_native_volume == 0):
        reasons.append('native post-treatment lesion mask is empty')
    if not has_post:
        reasons.append('post-treatment lesion mask unavailable')
    manual = 'REVIEW' if any((s in subject for s in ('PRIVATE_SUBJECT_040', 'PRIVATE_SUBJECT_006', 'PRIVATE_SUBJECT_038'))) else 'PASS'
    baseline_use = 'INCLUDE_WITH_FLAG' if manual == 'REVIEW' else 'INCLUDE'
    longitudinal_use = 'EXCLUDE' if not has_post or post_native_volume == 0 else baseline_use
    return (manual, baseline_use, longitudinal_use + (f" | {'; '.join(reasons)}" if reasons else ''))

def parse_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan

def main() -> None:
    clinical = clinical_rows()
    audit_rows = []
    t2_rows = []
    for row in read_qc_summary():
        group = row['group']
        subject = row['subject_id']
        clinical_row = clinical.get(subject, {})
        clinical_side = str(clinical_row.get('左右侧', ''))
        location = str(clinical_row.get('部位', ''))
        lateral = laterality_metrics(group, subject)
        side_status = laterality_status(clinical_side, location, lateral)
        t2_metrics = qc_t2_to_t1(group, subject)
        has_post = row['has_post_mask'].lower() == 'true'
        post_native = parse_float(row['post_native_volume_ml'])
        (manual, baseline_use, longitudinal_and_reason) = manual_flags(subject, side_status, post_native, has_post)
        (longitudinal_use, _, reason) = longitudinal_and_reason.partition(' | ')
        before_native = native_mask(group, 'before', subject)
        audit = {**row, 'clinical_lesion_side': clinical_side, 'clinical_location': location, 'native_orientation': '' if before_native is None else ''.join(nib.aff2axcodes(nib.load(str(before_native)).affine)), **lateral, 'laterality_status': side_status, **t2_metrics, 'automatic_registration_qc': row['status'], 'manual_audit_status': manual, 'baseline_analysis_status': baseline_use, 'longitudinal_mask_analysis_status': longitudinal_use, 'audit_reason': reason}
        audit_rows.append(audit)
        t2_rows.append({'group': group, 'subject_id': subject, **t2_metrics})
    audit_path = REG / 'registration_audit_final.csv'
    with audit_path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)
    t2_path = REG / 't2_to_t1_qc_metrics.csv'
    with t2_path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(t2_rows[0]))
        writer.writeheader()
        writer.writerows(t2_rows)
    print(f'Audit subjects: {len(audit_rows)}')
    print(f"Manual review: {sum((row['manual_audit_status'] == 'REVIEW' for row in audit_rows))}")
    print(f"Reflection detected: {sum((str(row['t2_before_reflection_detected']) == 'True' or str(row['t2_after_reflection_detected']) == 'True' for row in audit_rows))}")
    print(f'Final audit: {audit_path}')
    print(f'T2-to-T1 metrics: {t2_path}')
if __name__ == '__main__':
    main()
