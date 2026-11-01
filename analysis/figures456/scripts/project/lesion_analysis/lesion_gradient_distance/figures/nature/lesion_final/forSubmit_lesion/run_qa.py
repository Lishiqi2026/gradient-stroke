from __future__ import annotations
import os as _portable_os
from pathlib import Path as _PortablePath

def _portable_path(kind, relative):
    project = _PortablePath(_portable_os.environ['PAPER_PROJECT_ROOT'])
    root = project if kind == 'project' else project.parent
    return str(root / relative)
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
HERE = Path(__file__).resolve().parent
FIGURES = HERE / 'figures'
SOURCE = HERE / 'source_data'
QA = HERE / 'qa'
STEMS = ['lesion_longitudinal_volume', 'lesion_matched_clinical_improvement', 'G2_FDR_network_distribution', 'G1_G2_reference_and_adjusted_TMS_effect']

def main() -> None:
    QA.mkdir(parents=True, exist_ok=True)
    checks: dict[str, object] = {}
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        PdfReader = None
    for stem in STEMS:
        for suffix in ('pdf', 'svg', 'png', 'tiff'):
            path = FIGURES / f'{stem}.{suffix}'
            checks[f'{stem}_{suffix}_exists'] = path.exists()
            checks[f'{stem}_{suffix}_bytes'] = path.stat().st_size if path.exists() else 0
        if PdfReader is not None:
            checks[f'{stem}_pdf_one_page'] = len(PdfReader(str(FIGURES / f'{stem}.pdf')).pages) == 1
        for suffix in ('png', 'tiff'):
            with Image.open(FIGURES / f'{stem}.{suffix}') as image:
                checks[f'{stem}_{suffix}_mode'] = image.mode
                checks[f'{stem}_{suffix}_size'] = list(image.size)
                checks[f'{stem}_{suffix}_dpi'] = [float(value) for value in image.info.get('dpi', (0, 0))]
    volume = pd.read_csv(SOURCE / 'lesion_volume_subjects_plotted.csv')
    volume_stats = pd.read_csv(SOURCE / 'lesion_volume_statistics_plotted.csv')
    matched = pd.read_csv(SOURCE / 'lesion_matched_clinical_improvements.csv')
    matched_stats = pd.read_csv(SOURCE / 'lesion_matched_clinical_statistics.csv')
    radar = pd.read_csv(SOURCE / 'G2_FDR_network_counts.csv')
    surfaces = pd.read_csv(SOURCE / 'G1_G2_surface_source_data.csv')
    checks['lesion_volume_35_subjects'] = volume['subject_id'].nunique() == 35
    checks['lesion_volume_control_n14'] = volume.query("group == 'Control'")['subject_id'].nunique() == 14
    checks['lesion_volume_tms_n21'] = volume.query("group == 'TMS'")['subject_id'].nunique() == 21
    checks['lesion_volume_extreme_control_excluded'] = PRIVATE_SUBJECT_040 not in set(volume['subject_id'].astype(str))
    checks['lesion_volume_q_values'] = set(volume_stats['wilcoxon_fdr_q'].round(3)) == {0.03, 0.004}
    checks['matched_two_outcomes'] = set(matched['outcome']) == {'NIHSS', 'ADL'}
    checks['matched_each_group_n5'] = all(matched.groupby(['outcome', 'group'])['subject_id'].nunique().eq(5))
    checks['matched_q_0_026'] = matched_stats['permutation_fdr_q'].round(3).eq(0.026).all()
    checks['radar_67_total'] = int(radar['total_FDR_ROI_count'].sum()) == 67
    checks['radar_59_positive'] = int(radar['positive_FDR_ROI_count'].sum()) == 59
    checks['radar_8_negative'] = int(radar['negative_FDR_ROI_count'].sum()) == 8
    checks['surface_400_roi'] = len(surfaces) == 400
    checks['surface_g2_fdr_67'] = int(surfaces['G2_FDR_significant'].astype(bool).sum()) == 67
    checks = {key: bool(value) if isinstance(value, np.bool_) else value for (key, value) in checks.items()}
    boolean_checks = [key for (key, value) in checks.items() if isinstance(value, bool)]
    checks['overall_pass'] = all((checks[key] for key in boolean_checks))
    (QA / 'qa_report.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    lines = ['# QA report', ''] + [f'- {key}: {value}' for (key, value) in checks.items()]
    (QA / 'QA_REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(checks, indent=2))
    if not checks['overall_pass']:
        raise SystemExit('QA failed')
if __name__ == '__main__':
    main()
