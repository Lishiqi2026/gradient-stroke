from __future__ import annotations
import os as _portable_os
from pathlib import Path as _PortablePath

def _portable_path(kind, relative):
    project = _PortablePath(_portable_os.environ['PAPER_PROJECT_ROOT'])
    root = project if kind == 'project' else project.parent
    return str(root / relative)
import argparse
import json
import sys
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
PROJECT_ROOT = Path(_portable_path('project', ''))
ANALYSIS_ROOT = PROJECT_ROOT / 'lesion_analysis'
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))
import run_lesion_adjusted_gradient_value as base
FMRI_ROOT = Path(_portable_path('work', 'GretnaFunNIfTI(预处理后的fmri)'))
ATLAS_PATH = ANALYSIS_ROOT / 'resources' / 'Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz'
BASELINE_TIMESERIES = PROJECT_ROOT / 'fc_analysis' / 'timeseries_fsl'
OUTPUT_ROOT = ANALYSIS_ROOT / 'lesion_gradient_distance' / 'fc_consistent'
TABLES = OUTPUT_ROOT / 'tables'
TIMESERIES = OUTPUT_ROOT / 'timeseries_fsl'
FC_R = OUTPUT_ROOT / 'fc_matrices_r'
FC_Z = OUTPUT_ROOT / 'fc_matrices_z'
FINAL_FMRI_NAME = 'bcNGSdswranrest.nii'
FD_THRESHOLD = 0.5
MIN_RETAINED_FRAMES = 100
ABSOLUTE_MINIMUM_FRAMES = 30

def ensure_dirs() -> None:
    for path in (OUTPUT_ROOT, TABLES, TIMESERIES, FC_R, FC_Z):
        path.mkdir(parents=True, exist_ok=True)

def scrub_keep_mask(fd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    high_motion = np.asarray(fd > FD_THRESHOLD, dtype=bool)
    scrubbed = high_motion.copy()
    for time_index in np.flatnonzero(high_motion):
        scrubbed[max(0, time_index - 1):min(len(fd), time_index + 3)] = True
    return (~scrubbed, high_motion)

def correlation_matrices(timeseries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if timeseries.ndim != 2 or timeseries.shape[1] != 400:
        raise RuntimeError(f'Expected T x 400 time series, got {timeseries.shape}')
    if not np.all(np.isfinite(timeseries)):
        raise RuntimeError('Non-finite ROI time series')
    if np.any(np.std(timeseries, axis=0, ddof=1) <= 0):
        raise RuntimeError('At least one ROI has a constant time series')
    correlation = np.corrcoef(timeseries, rowvar=False)
    correlation = np.clip(correlation, -0.999999, 0.999999)
    fisher_z = np.arctanh(correlation)
    np.fill_diagonal(correlation, 0.0)
    np.fill_diagonal(fisher_z, 0.0)
    return (correlation.astype(np.float32), fisher_z.astype(np.float32))

def prepare_manifest() -> pd.DataFrame:
    ensure_dirs()
    participants = base.load_inputs().table[['subject_id', 'group']].copy()
    participants['subject_id'] = participants['subject_id'].astype(str)
    atlas_image = nib.load(str(ATLAS_PATH))
    rows = []
    for participant in participants.itertuples():
        for (session, visit_suffix) in (('baseline', '1'), ('post', '2')):
            visits = sorted(FMRI_ROOT.glob(f'{participant.subject_id}-*-{visit_suffix}'))
            visits = [visit for visit in visits if (visit / FINAL_FMRI_NAME).exists()]
            if len(visits) != 1:
                raise RuntimeError(f'Expected one {session} fMRI for {participant.subject_id}, found {visits}')
            visit = visits[0]
            image = nib.load(str(visit / FINAL_FMRI_NAME))
            if image.shape != (91, 109, 91, 235):
                raise RuntimeError(f'Unexpected fMRI shape for {visit.name}: {image.shape}')
            if image.shape[:3] != atlas_image.shape or not np.allclose(image.affine, atlas_image.affine):
                raise RuntimeError(f'Atlas/fMRI grid mismatch for {visit.name}')
            fd_path = visit / 'PowerFD.txt'
            if not fd_path.exists():
                raise FileNotFoundError(fd_path)
            fd = np.loadtxt(fd_path).reshape(-1)
            if len(fd) != 235 or not np.all(np.isfinite(fd)):
                raise RuntimeError(f'Invalid PowerFD for {visit.name}: {fd.shape}')
            if session == 'baseline':
                timeseries_path = BASELINE_TIMESERIES / f'{participant.subject_id}_baseline_schaefer400.tsv'
            else:
                timeseries_path = TIMESERIES / f'{participant.subject_id}_post_schaefer400.tsv'
            rows.append({'subject_id': participant.subject_id, 'group': participant.group, 'session': session, 'visit': visit.name, 'fmri_path': str(visit / FINAL_FMRI_NAME), 'power_fd_path': str(fd_path), 'timeseries_path': str(timeseries_path)})
    manifest = pd.DataFrame(rows).sort_values(['subject_id', 'session'])
    if len(manifest) != 76:
        raise RuntimeError(f'Expected 76 longitudinal scans, found {len(manifest)}')
    manifest.to_csv(TABLES / 'longitudinal_fmri_manifest.csv', index=False, encoding='utf-8-sig')
    post = manifest.loc[manifest['session'] == 'post', ['visit', 'subject_id']]
    (OUTPUT_ROOT / 'post_visits.tsv').write_text(''.join((f'{row.visit}\t{row.subject_id}\n' for row in post.itertuples())), encoding='utf-8')
    return manifest

def extract_post_timeseries_native() -> None:
    """Extract ROI means natively; numerically equivalent to FSL fslmeants --label."""
    ensure_dirs()
    manifest_path = TABLES / 'longitudinal_fmri_manifest.csv'
    if not manifest_path.exists():
        prepare_manifest()
    manifest = pd.read_csv(manifest_path, dtype={'subject_id': str})
    post = manifest.loc[manifest['session'] == 'post'].copy()
    atlas = np.asarray(nib.load(str(ATLAS_PATH)).dataobj, dtype=np.int16)
    mask = atlas > 0
    labels = atlas[mask]
    counts = np.bincount(labels, minlength=401)[1:]
    if np.any(counts == 0):
        raise RuntimeError('At least one Schaefer ROI is empty')
    for (index, row) in enumerate(post.itertuples(), start=1):
        output = Path(row.timeseries_path)
        if output.exists():
            try:
                existing = np.loadtxt(output)
                if existing.shape == (235, 400) and np.all(np.isfinite(existing)):
                    print(f'[skip {index:02d}/38] {row.subject_id}', flush=True)
                    continue
            except Exception:
                pass
        image = nib.load(row.fmri_path)
        proxy = image.dataobj
        timeseries = np.empty((235, 400), dtype=np.float64)
        for time_index in range(235):
            volume = np.asarray(proxy[..., time_index])
            sums = np.bincount(labels, weights=volume[mask], minlength=401)[1:]
            timeseries[time_index] = sums / counts
        if not np.all(np.isfinite(timeseries)):
            raise RuntimeError(f'Non-finite extracted time series: {row.subject_id}')
        np.savetxt(output, timeseries, fmt='%.10f', delimiter='\t')
        print(f'[done {index:02d}/38] {row.subject_id}', flush=True)

def build_fc() -> None:
    ensure_dirs()
    manifest_path = TABLES / 'longitudinal_fmri_manifest.csv'
    if not manifest_path.exists():
        prepare_manifest()
    manifest = pd.read_csv(manifest_path, dtype={'subject_id': str})
    qc_rows = []
    for row in manifest.itertuples():
        timeseries_path = Path(row.timeseries_path)
        if not timeseries_path.exists():
            raise FileNotFoundError(f'Missing time series {timeseries_path}. Extract post scans with extract_post_timeseries_fsl.sh first.')
        timeseries = np.loadtxt(timeseries_path)
        if timeseries.shape != (235, 400):
            raise RuntimeError(f'Unexpected time-series shape for {row.subject_id}/{row.session}: {timeseries.shape}')
        fd = np.loadtxt(row.power_fd_path).reshape(-1)
        (keep, high_motion) = scrub_keep_mask(fd)
        if int(keep.sum()) < ABSOLUTE_MINIMUM_FRAMES:
            raise RuntimeError(f'Fewer than {ABSOLUTE_MINIMUM_FRAMES} retained frames for {row.subject_id}/{row.session}: {keep.sum()}')
        (correlation, fisher_z) = correlation_matrices(timeseries[keep])
        np.save(FC_R / f'{row.subject_id}_{row.session}_fc_r.npy', correlation)
        np.save(FC_Z / f'{row.subject_id}_{row.session}_fc_z.npy', fisher_z)
        qc_rows.append({'subject_id': row.subject_id, 'group': row.group, 'session': row.session, 'visit': row.visit, 'mean_fd': float(fd.mean()), 'median_fd': float(np.median(fd)), 'max_fd': float(fd.max()), 'high_motion_frames': int(high_motion.sum()), 'scrubbed_frames': int((~keep).sum()), 'retained_frames': int(keep.sum()), 'retained_fraction': float(keep.mean()), 'primary_motion_eligible': bool(keep.sum() >= MIN_RETAINED_FRAMES), 'global_mean_fc_r': float(correlation[np.triu_indices_from(correlation, k=1)].mean())})
    qc = pd.DataFrame(qc_rows)
    qc.to_csv(TABLES / 'longitudinal_fc_qc.csv', index=False, encoding='utf-8-sig')
    manifest = {'cohort_n': int(qc['subject_id'].nunique()), 'scan_n': int(len(qc)), 'sessions': qc['session'].value_counts().to_dict(), 'timeseries': 'FSL fslmeants with Schaefer400 --label', 'motion': {'fd_threshold': FD_THRESHOLD, 'scrubbing': 'high-FD frame, one preceding frame, and two following frames', 'minimum_retained_frames': MIN_RETAINED_FRAMES, 'absolute_minimum_for_sensitivity_matrix': ABSOLUTE_MINIMUM_FRAMES}}
    (OUTPUT_ROOT / 'fc_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Prepare matched baseline/post Schaefer400 FC for lesion-gradient analysis')
    parser.add_argument('stage', choices=('prepare', 'extract', 'build'), help='Prepare manifest, extract post ROI time series, or build FC matrices')
    return parser.parse_args()
if __name__ == '__main__':
    args = parse_args()
    if args.stage == 'prepare':
        prepare_manifest()
    elif args.stage == 'extract':
        extract_post_timeseries_native()
    else:
        build_fc()
