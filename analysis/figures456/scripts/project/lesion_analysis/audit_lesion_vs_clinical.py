from __future__ import annotations
import os as _portable_os
from pathlib import Path as _PortablePath

def _portable_path(kind, relative):
    project = _PortablePath(_portable_os.environ['PAPER_PROJECT_ROOT'])
    root = project if kind == 'project' else project.parent
    return str(root / relative)
import json
from pathlib import Path
from xml.etree import ElementTree
import nibabel as nib
import numpy as np
import openpyxl
import pandas as pd
ROOT = Path(_portable_path('work', ''))
ANALYSIS = ROOT / 'Gradient' / 'lesion_analysis'
REG = ANALYSIS / 'mni_registration'
RESOURCES = ANALYSIS / 'resources'
CLINICAL_PATH = ROOT / '董分组.xlsx'
ATLAS_SPECS = {'HO_subcortical': (RESOURCES / 'HarvardOxford-sub-maxprob-thr25-2mm.nii.gz', RESOURCES / 'HarvardOxford-Subcortical.xml', 1), 'HO_cortical': (RESOURCES / 'HarvardOxford-cort-maxprob-thr25-2mm.nii.gz', RESOURCES / 'HarvardOxford-Cortical.xml', 1), 'JHU_white_matter': (RESOURCES / 'JHU-ICBM-labels-2mm.nii.gz', RESOURCES / 'JHU-labels.xml', 0), 'Cerebellum': (RESOURCES / 'Cerebellum-MNIfnirt-maxprob-thr25-2mm.nii.gz', RESOURCES / 'Cerebellum_MNIfnirt.xml', 1)}
TERM_PATTERNS = {'基底节': ('caudate', 'putamen', 'pallidum', 'accumbens', 'internal capsule', 'external capsule'), '侧脑室旁': ('cerebral white matter', 'corona radiata', 'lateral ventricle', 'corpus callosum', 'fornix'), '丘脑': ('thalamus', 'thalamic radiation'), '脑桥': ('brain-stem', 'pontine', 'cerebellar peduncle', 'corticospinal tract', 'medial lemniscus'), '延髓': ('brain-stem', 'corticospinal tract', 'medial lemniscus', 'inferior cerebellar peduncle'), '额叶': ('frontal', 'precentral', 'anterior corona radiata', 'genu of corpus callosum'), '顶叶': ('parietal', 'postcentral', 'angular', 'supramarginal', 'precuneous', 'superior longitudinal', 'superior corona radiata'), '颞叶': ('temporal', 'hippocampus', 'amygdala', 'sagittal stratum', 'uncinate', 'inferior longitudinal'), '枕叶': ('occipital', 'cuneal', 'lingual', 'optic radiation', 'posterior thalamic radiation'), '半卵圆中心': ('cerebral white matter', 'corona radiata', 'superior longitudinal fasciculus'), '放射冠': ('corona radiata',), '小脑': ('cerebell', 'crus', 'vermis'), '海马': ('hippocampus',)}

def clinical_table() -> pd.DataFrame:
    workbook = openpyxl.load_workbook(CLINICAL_PATH, data_only=True, read_only=True)
    sheet = workbook[workbook.sheetnames[0]]
    rows = list(sheet.iter_rows(values_only=True))
    workbook.close()
    frame = pd.DataFrame(rows[1:], columns=[str(value) for value in rows[0]])
    frame = frame.loc[frame['病历号'].notna()].copy()
    frame['subject_id'] = frame['病历号'].astype(int).astype(str)
    return frame

def label_map(xml_path: Path, value_offset: int) -> dict[int, str]:
    root = ElementTree.parse(xml_path).getroot()
    result = {}
    for label in root.findall('./data/label'):
        value = int(label.attrib['index']) + value_offset
        result[value] = (label.text or '').strip()
    return result

def load_atlases() -> dict[str, tuple[np.ndarray, dict[int, str]]]:
    reference = nib.load(str(RESOURCES / 'MNI152_T1_2mm.nii.gz'))
    atlases = {}
    for (name, (image_path, xml_path, offset)) in ATLAS_SPECS.items():
        image = nib.load(str(image_path))
        if image.shape != reference.shape or not np.allclose(image.affine, reference.affine):
            raise RuntimeError(f'Atlas geometry mismatch: {name}')
        atlases[name] = (np.asarray(image.dataobj, dtype=np.int16), label_map(xml_path, offset))
    return atlases

def clinical_terms(location: str) -> list[str]:
    terms = []
    for term in ('基底节', '侧脑室旁', '丘脑', '脑桥', '桥脑', '延髓', '半卵圆中心', '放射冠', '小脑', '海马'):
        if term in location:
            normalized = '脑桥' if term == '桥脑' else term
            if normalized not in terms:
                terms.append(normalized)
    for (character, term) in (('额', '额叶'), ('顶', '顶叶'), ('颞', '颞叶'), ('枕', '枕叶')):
        if character in location and term not in terms:
            terms.append(term)
    return terms

def atlas_overlap(mask: np.ndarray, atlas: np.ndarray, labels: dict[int, str]) -> list[tuple[str, int, float]]:
    values = atlas[mask]
    counts = np.bincount(values, minlength=int(atlas.max()) + 1)
    rows = []
    lesion_voxels = int(mask.sum())
    for (value, count) in enumerate(counts):
        if value == 0 or count == 0 or value not in labels:
            continue
        rows.append((labels[value], int(count), float(count * 100.0 / lesion_voxels)))
    return sorted(rows, key=lambda row: row[1], reverse=True)

def compact_top(overlaps: list[tuple[str, int, float]], limit: int=4) -> str:
    if not overlaps:
        return 'none'
    return '; '.join((f'{name} ({percent:.1f}% lesion)' for (name, _, percent) in overlaps[:limit]))

def match_terms(terms: list[str], all_regions: list[str]) -> tuple[list[str], list[str], dict[str, list[str]]]:
    supported = []
    unsupported = []
    evidence = {}
    for term in terms:
        patterns = TERM_PATTERNS[term]
        matches = sorted({region for region in all_regions if any((pattern in region.lower() for pattern in patterns))})
        evidence[term] = matches
        if matches:
            supported.append(term)
        else:
            unsupported.append(term)
    return (supported, unsupported, evidence)

def location_status(terms: list[str], supported: list[str]) -> str:
    if not terms:
        return 'UNMAPPED_DESCRIPTION'
    if len(supported) == len(terms):
        return 'CONSISTENT_COARSE'
    if supported:
        return 'PARTIAL_SUPPORT'
    return 'NO_ATLAS_SUPPORT'

def main() -> None:
    clinical = clinical_table().set_index('subject_id')
    registration = pd.read_csv(REG / 'registration_audit_final.csv', dtype={'subject_id': str}).set_index('subject_id')
    atlases = load_atlases()
    rows = []
    for group in ('Control', 'TMS'):
        for path in sorted((REG / 'masks' / group / 'before').glob('*_lesion_mni2mm.nii.gz')):
            subject = path.name.split('_')[0]
            image = nib.load(str(path))
            mask = np.asarray(image.dataobj) > 0.5
            clinical_row = clinical.loc[subject]
            registration_row = registration.loc[subject]
            location = str(clinical_row['部位'])
            terms = clinical_terms(location)
            atlas_results = {}
            all_regions = []
            for (atlas_name, (atlas, labels)) in atlases.items():
                overlaps = atlas_overlap(mask, atlas, labels)
                atlas_results[atlas_name] = overlaps
                all_regions.extend((name for (name, _, _) in overlaps))
            (supported, unsupported, evidence) = match_terms(terms, all_regions)
            loc_status = location_status(terms, supported)
            side_status = str(registration_row['laterality_status'])
            if side_status == 'CLINICAL_MASK_DISCREPANCY':
                overall = 'REVIEW_CLEAR_SIDE_CONFLICT'
            elif side_status in ('BILATERAL_DESCRIPTION', 'MIDLINE_BRAINSTEM_CAUTION'):
                overall = 'REVIEW_SIDE_SEMANTICS'
            elif loc_status in ('PARTIAL_SUPPORT', 'NO_ATLAS_SUPPORT', 'UNMAPPED_DESCRIPTION'):
                overall = 'REVIEW_LOCATION_OR_ATLAS_LIMIT'
            else:
                overall = 'COARSELY_CONSISTENT'
            evidence_text = '; '.join((f"{term}: {(', '.join(names[:3]) if names else 'no atlas evidence')}" for (term, names) in evidence.items()))
            rows.append({'group': group, 'subject_id': subject, 'clinical_side': clinical_row['左右侧'], 'clinical_location': location, 'mask_derived_side': registration_row['mask_derived_side'], 'mni_centroid_x': registration_row['mni_centroid_x'], 'mni_centroid_y': registration_row['mni_centroid_y'], 'mni_centroid_z': registration_row['mni_centroid_z'], 'side_comparison': side_status, 'clinical_location_terms': ';'.join(terms), 'supported_terms': ';'.join(supported), 'unsupported_terms': ';'.join(unsupported), 'location_comparison': loc_status, 'term_evidence': evidence_text, 'top_HO_subcortical': compact_top(atlas_results['HO_subcortical']), 'top_HO_cortical': compact_top(atlas_results['HO_cortical']), 'top_JHU_white_matter': compact_top(atlas_results['JHU_white_matter']), 'top_Cerebellum': compact_top(atlas_results['Cerebellum']), 'overall_comparison': overall})
    result = pd.DataFrame(rows)
    output = REG / 'lesion_vs_clinical_audit.csv'
    result.to_csv(output, index=False, encoding='utf-8-sig')
    summary = {'n': len(result), 'side_comparison': result['side_comparison'].value_counts().to_dict(), 'location_comparison': result['location_comparison'].value_counts().to_dict(), 'overall_comparison': result['overall_comparison'].value_counts().to_dict()}
    (REG / 'lesion_vs_clinical_audit_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'Audit: {output}')
if __name__ == '__main__':
    main()
