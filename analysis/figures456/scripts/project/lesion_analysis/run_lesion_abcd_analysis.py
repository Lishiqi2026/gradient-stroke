from __future__ import annotations
import os as _portable_os
from pathlib import Path as _PortablePath

def _portable_path(kind, relative):
    project = _PortablePath(_portable_os.environ['PAPER_PROJECT_ROOT'])
    root = project if kind == 'project' else project.parent
    return str(root / relative)
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import nibabel as nib
import numpy as np
import openpyxl
import pandas as pd
from scipy.stats import mannwhitneyu
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
ROOT = Path(_portable_path('work', ''))
ANALYSIS = ROOT / 'Gradient' / 'lesion_analysis'
REG = ANALYSIS / 'mni_registration'
RESOURCES = ANALYSIS / 'resources'
OUT = ANALYSIS / 'results_abcd'
FIGURES = OUT / 'figures'
TABLES = OUT / 'tables'
NIFTI = OUT / 'nifti'
CLINICAL_PATH = ROOT / '董分组.xlsx'
GRADIENT_ROOT = ROOT / 'Gradient' / '皮层' / '(400，5)'
ATLAS_PATH = RESOURCES / 'Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz'
ATLAS_LABELS_PATH = RESOURCES / 'Schaefer2018_400Parcels_7Networks_order.txt'
MNI_PATH = RESOURCES / 'MNI152_T1_2mm.nii.gz'
GROUPS = ('Control', 'TMS')
GROUP_COLORS = {'Control': '#4C78A8', 'TMS': '#D65F45'}
NETWORK_ORDER = ('VIS', 'SMN', 'DAN', 'VAN', 'LIM', 'CON', 'DMN')
NETWORK_MAP = {'Vis': 'VIS', 'SomMot': 'SMN', 'DorsAttn': 'DAN', 'SalVentAttn': 'VAN', 'Limbic': 'LIM', 'Cont': 'CON', 'Default': 'DMN'}
OUTCOMES = {'NIHSS': ('NIHSS_pre', 'NIHSS_post', -1.0), 'ADL': ('ADL_pre', 'ADL_post', 1.0), 'MMSE': ('MMSE_pre', 'MMSE_post', 1.0), 'HAMD': ('HAMD_pre', 'HAMD_post', -1.0), 'VFT': ('VFT_pre', 'VFT_post', 1.0)}
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans', 'sans-serif'], 'font.size': 7, 'axes.titlesize': 8, 'axes.labelsize': 7, 'axes.linewidth': 0.7, 'axes.spines.right': False, 'axes.spines.top': False, 'legend.frameon': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})

def ensure_output_dirs() -> None:
    for path in (FIGURES, TABLES, NIFTI):
        path.mkdir(parents=True, exist_ok=True)

def save_figure(fig: plt.Figure, stem: str) -> None:
    for (suffix, kwargs) in (('png', {'dpi': 300}), ('svg', {}), ('pdf', {}), ('tiff', {'dpi': 600})):
        fig.savefig(FIGURES / f'{stem}.{suffix}', bbox_inches='tight', facecolor='white', **kwargs)

def zscore(values: pd.Series | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    sd = float(np.nanstd(array, ddof=1))
    return (array - float(np.nanmean(array))) / sd if sd > 0 else np.zeros_like(array)

def load_clinical() -> pd.DataFrame:
    workbook = openpyxl.load_workbook(CLINICAL_PATH, data_only=True, read_only=True)
    sheet = workbook[workbook.sheetnames[0]]
    rows = list(sheet.iter_rows(values_only=True))
    workbook.close()
    frame = pd.DataFrame(rows[1:], columns=[str(value) for value in rows[0]])
    frame = frame.loc[frame['病历号'].notna()].copy()
    frame['subject_id'] = frame['病历号'].astype(int).astype(str)
    frame['sex_male'] = frame['性别'].astype(str).str.upper().eq('M').astype(int)
    return frame

def load_atlas_labels() -> pd.DataFrame:
    rows = []
    for line in ATLAS_LABELS_PATH.read_text(encoding='utf-8').splitlines():
        fields = line.split()
        if len(fields) < 2:
            continue
        label = int(fields[0])
        name = fields[1]
        token = name.split('_')[2]
        rows.append({'roi': label, 'roi_name': name, 'network': NETWORK_MAP[token]})
    labels = pd.DataFrame(rows)
    if len(labels) != 400 or set(labels['network']) != set(NETWORK_ORDER):
        raise RuntimeError('Unexpected Schaefer-400 label table')
    return labels

def list_masks() -> list[tuple[str, str, Path]]:
    rows = []
    for group in GROUPS:
        directory = REG / 'masks' / group / 'before'
        for path in sorted(directory.glob('*_lesion_mni2mm.nii.gz')):
            rows.append((group, path.name.split('_')[0], path))
    if len(rows) != 38:
        raise RuntimeError(f'Expected 38 baseline masks, found {len(rows)}')
    return rows

def extract_lesion_metrics(masks: list[tuple[str, str, Path]], labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, list[np.ndarray]], nib.Nifti1Image]:
    atlas_image = nib.load(str(ATLAS_PATH))
    atlas = np.asarray(atlas_image.dataobj, dtype=np.int16)
    roi_sizes = np.bincount(atlas.ravel(), minlength=401).astype(float)
    network_rois = {network: labels.loc[labels['network'] == network, 'roi'].to_numpy(dtype=int) for network in NETWORK_ORDER}
    audit = pd.read_csv(REG / 'registration_audit_final.csv', dtype={'subject_id': str})
    audit = audit.set_index('subject_id')
    participant_rows = []
    parcel_rows = []
    network_rows = []
    arrays: dict[str, list[np.ndarray]] = {'All': [], 'Control': [], 'TMS': []}
    reference_image = None
    for (group, subject, path) in masks:
        image = nib.load(str(path))
        mask = np.asarray(image.dataobj) > 0.5
        if image.shape != atlas.shape or not np.allclose(image.affine, atlas_image.affine):
            raise RuntimeError(f'Atlas geometry mismatch for {subject}')
        if not np.any(mask):
            raise RuntimeError(f'Empty baseline MNI mask for {subject}')
        reference_image = image
        arrays['All'].append(mask)
        arrays[group].append(mask)
        overlap_counts = np.bincount(atlas[mask], minlength=401).astype(float)
        roi_damage_pct = np.divide(overlap_counts[1:401] * 100.0, roi_sizes[1:401], out=np.zeros(400, dtype=float), where=roi_sizes[1:401] > 0)
        voxel_ml = abs(float(np.linalg.det(image.affine[:3, :3]))) / 1000.0
        lesion_voxels = int(mask.sum())
        cortical_voxels = int(overlap_counts[1:401].sum())
        audit_row = audit.loc[subject]
        participant_rows.append({'subject_id': subject, 'group': group, 'baseline_lesion_volume_ml': lesion_voxels * voxel_ml, 'log1p_lesion_volume': np.log1p(lesion_voxels * voxel_ml), 'schaefer_cortical_overlap_ml': cortical_voxels * voxel_ml, 'schaefer_cortical_fraction_pct': cortical_voxels * 100.0 / lesion_voxels, 'mask_lateralization_index': float(audit_row['mni_right_fraction'] - audit_row['mni_left_fraction']), 'mask_derived_side': audit_row['mask_derived_side'], 'laterality_status': audit_row['laterality_status'], 'registration_manual_status': audit_row['manual_audit_status']})
        for (roi_index, row) in labels.iterrows():
            roi = int(row['roi'])
            parcel_rows.append({'subject_id': subject, 'group': group, 'roi': roi, 'roi_name': row['roi_name'], 'network': row['network'], 'damaged_voxels': int(overlap_counts[roi]), 'roi_voxels': int(roi_sizes[roi]), 'roi_damage_pct': float(roi_damage_pct[roi_index])})
        for (network, rois) in network_rois.items():
            damaged = float(overlap_counts[rois].sum())
            total = float(roi_sizes[rois].sum())
            network_rows.append({'subject_id': subject, 'group': group, 'network': network, 'damaged_voxels': int(damaged), 'network_voxels': int(total), 'network_damage_pct': damaged * 100.0 / total, 'network_involved': int(damaged > 0)})
    if reference_image is None:
        raise RuntimeError('No reference lesion image')
    return (pd.DataFrame(participant_rows), pd.DataFrame(parcel_rows), pd.DataFrame(network_rows), arrays, reference_image)

def save_overlap_maps(arrays: dict[str, list[np.ndarray]], reference: nib.Nifti1Image) -> dict[str, np.ndarray]:

    def save_nifti(data: np.ndarray, dtype: np.dtype, path: Path) -> None:
        header = reference.header.copy()
        header.set_data_dtype(dtype)
        nib.save(nib.Nifti1Image(data.astype(dtype), reference.affine, header), path)
    proportions = {}
    summary = []
    for (label, masks) in arrays.items():
        count = np.sum(np.stack(masks, axis=0), axis=0).astype(np.int16)
        proportion = count.astype(np.float32) / len(masks)
        slug = label.lower()
        save_nifti(count, np.int16, NIFTI / f'baseline_overlap_{slug}_count.nii.gz')
        save_nifti(proportion, np.float32, NIFTI / f'baseline_overlap_{slug}_proportion.nii.gz')
        save_nifti(count > 0, np.uint8, NIFTI / f'baseline_coverage_{slug}_binary.nii.gz')
        proportions[label] = proportion
        voxel_ml = abs(float(np.linalg.det(reference.affine[:3, :3]))) / 1000.0
        summary.append({'group': label, 'n': len(masks), 'maximum_overlap_count': int(count.max()), 'maximum_overlap_pct': float(proportion.max() * 100), 'union_volume_ml': float(np.count_nonzero(count) * voxel_ml)})
    pd.DataFrame(summary).to_csv(TABLES / 'A_overlap_summary.csv', index=False, encoding='utf-8-sig')
    return proportions

def robust_scale(data: np.ndarray) -> np.ndarray:
    valid = np.isfinite(data) & (data > 0)
    (low, high) = np.percentile(data[valid], [1, 99])
    return np.clip((data - low) / (high - low), 0, 1)

def plot_a_overlap(proportions: dict[str, np.ndarray], arrays: dict[str, list[np.ndarray]]) -> None:
    mni_image = nib.load(str(MNI_PATH))
    mni = robust_scale(np.asarray(mni_image.dataobj, dtype=np.float32))
    z_slices = (-48, -32, -16, 0, 16, 32, 48)
    z_indices = [int(round((z - mni_image.affine[2, 3]) / mni_image.affine[2, 2])) for z in z_slices]
    vmax = max((float(values.max() * 100) for values in proportions.values()))
    cmap = LinearSegmentedColormap.from_list('lesion', ['#F4E8C1', '#E69F00', '#B2182B'])
    (fig, axes) = plt.subplots(3, len(z_slices), figsize=(7.2, 4.05), facecolor='white')
    image_artist = None
    for (row, label) in enumerate(('All', 'Control', 'TMS')):
        values = proportions[label] * 100
        for (col, (z, index)) in enumerate(zip(z_slices, z_indices)):
            ax = axes[row, col]
            background = mni[:, :, index].T
            overlay = values[:, :, index].T
            ax.imshow(background, cmap='gray', origin='lower', interpolation='nearest')
            masked = np.ma.masked_where(overlay <= 0, overlay)
            image_artist = ax.imshow(masked, cmap=cmap, vmin=0, vmax=vmax, origin='lower', interpolation='nearest', alpha=0.82)
            ax.set_xlim(7, 84)
            ax.set_ylim(9, 101)
            if row == 0:
                ax.set_title(f'z={z} mm', fontsize=7)
            if col == 0:
                ax.text(-0.11, 0.5, f'{label}\nn={len(arrays[label])}', transform=ax.transAxes, ha='right', va='center', fontsize=7, fontweight='bold')
            if row == 2:
                ax.text(0.02, 0.02, 'R', transform=ax.transAxes, color='white', fontsize=6, fontweight='bold')
                ax.text(0.98, 0.02, 'L', transform=ax.transAxes, color='white', fontsize=6, fontweight='bold', ha='right')
            ax.set_axis_off()
    fig.subplots_adjust(left=0.075, right=0.9, top=0.88, bottom=0.05, wspace=0.03, hspace=0.06)
    color_ax = fig.add_axes((0.92, 0.18, 0.014, 0.6))
    colorbar = fig.colorbar(image_artist, cax=color_ax)
    colorbar.set_label('Participants with lesion (%)', fontsize=7)
    colorbar.ax.tick_params(labelsize=6, width=0.6, length=2)
    fig.text(0.01, 0.96, 'A', fontsize=12, fontweight='bold', va='top')
    fig.suptitle('Baseline lesion overlap in MNI152 2 mm space', fontsize=9, y=0.97)
    save_figure(fig, 'A_baseline_lesion_overlap')
    plt.close(fig)

def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    comparisons = np.subtract.outer(a, b)
    return float((np.count_nonzero(comparisons > 0) - np.count_nonzero(comparisons < 0)) / comparisons.size)

def burden_group_statistics(participants: pd.DataFrame, networks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    variables = [('lesion_volume_ml', participants, 'baseline_lesion_volume_ml', 'all'), ('cortical_fraction_pct', participants, 'schaefer_cortical_fraction_pct', 'all')]
    for (name, frame, column, family) in variables:
        control = frame.loc[frame['group'] == 'Control', column].to_numpy(float)
        tms = frame.loc[frame['group'] == 'TMS', column].to_numpy(float)
        test = mannwhitneyu(tms, control, alternative='two-sided')
        rows.append({'metric': name, 'network': 'NA', 'family': family, 'control_n': len(control), 'tms_n': len(tms), 'control_median': float(np.median(control)), 'tms_median': float(np.median(tms)), 'cliffs_delta_tms_minus_control': cliffs_delta(tms, control), 'p_value': float(test.pvalue)})
    for network in NETWORK_ORDER:
        subset = networks.loc[networks['network'] == network]
        control = subset.loc[subset['group'] == 'Control', 'network_damage_pct'].to_numpy(float)
        tms = subset.loc[subset['group'] == 'TMS', 'network_damage_pct'].to_numpy(float)
        test = mannwhitneyu(tms, control, alternative='two-sided')
        rows.append({'metric': 'network_damage_pct', 'network': network, 'family': 'seven_networks', 'control_n': len(control), 'tms_n': len(tms), 'control_median': float(np.median(control)), 'tms_median': float(np.median(tms)), 'cliffs_delta_tms_minus_control': cliffs_delta(tms, control), 'p_value': float(test.pvalue)})
    result = pd.DataFrame(rows)
    result['fdr_q'] = result['p_value']
    network_mask = result['family'] == 'seven_networks'
    result.loc[network_mask, 'fdr_q'] = multipletests(result.loc[network_mask, 'p_value'], method='fdr_bh')[1]
    return result

def grouped_violin(ax: plt.Axes, frame: pd.DataFrame, column: str, transform, ylabel: str, ticks=None) -> None:
    rng = np.random.default_rng(20260803)
    for (position, group) in enumerate(GROUPS, start=1):
        values = transform(frame.loc[frame['group'] == group, column].to_numpy(float))
        parts = ax.violinplot(values, positions=[position], widths=0.72, showextrema=False)
        for body in parts['bodies']:
            body.set_facecolor(GROUP_COLORS[group])
            body.set_edgecolor('none')
            body.set_alpha(0.25)
        (q1, median, q3) = np.percentile(values, [25, 50, 75])
        ax.vlines(position, q1, q3, color=GROUP_COLORS[group], linewidth=2.2)
        ax.scatter(position, median, s=18, color='white', edgecolor=GROUP_COLORS[group], linewidth=0.9, zorder=4)
        jitter = rng.uniform(-0.12, 0.12, len(values))
        ax.scatter(position + jitter, values, s=10, color=GROUP_COLORS[group], alpha=0.7, edgecolor='white', linewidth=0.25, zorder=3)
    ax.set_xticks((1, 2), GROUPS)
    ax.set_ylabel(ylabel)
    if ticks is not None:
        (actual, labels) = ticks
        ax.set_yticks(transform(np.asarray(actual, dtype=float)), labels)
    ax.spines['left'].set_bounds(*ax.get_ylim())

def plot_b_burden(participants: pd.DataFrame, networks: pd.DataFrame) -> None:
    (fig, axes) = plt.subplots(1, 3, figsize=(7.2, 3.15), gridspec_kw={'width_ratios': (0.82, 0.82, 1.75)}, facecolor='white')
    grouped_violin(axes[0], participants, 'baseline_lesion_volume_ml', np.log1p, 'Baseline lesion volume (ml)', ticks=([0, 1, 5, 10, 50, 100, 200], ['0', '1', '5', '10', '50', '100', '200']))
    axes[0].set_title('Total lesion burden')
    grouped_violin(axes[1], participants, 'schaefer_cortical_fraction_pct', lambda x: x, 'Lesion within Schaefer cortex (%)')
    axes[1].set_title('Cortical involvement')
    rng = np.random.default_rng(20260803)
    for (index, network) in enumerate(NETWORK_ORDER):
        for (offset, group) in ((-0.16, 'Control'), (0.16, 'TMS')):
            values = networks.loc[(networks['network'] == network) & (networks['group'] == group), 'network_damage_pct'].to_numpy(float)
            shown = np.log1p(values)
            x = index + offset + rng.uniform(-0.055, 0.055, len(values))
            axes[2].scatter(x, shown, s=8, color=GROUP_COLORS[group], alpha=0.55, edgecolor='none')
            median = float(np.median(shown))
            (q1, q3) = np.percentile(shown, [25, 75])
            axes[2].vlines(index + offset, q1, q3, color=GROUP_COLORS[group], linewidth=1.5)
            axes[2].scatter(index + offset, median, s=20, color='white', edgecolor=GROUP_COLORS[group], linewidth=0.9, zorder=4)
    actual_ticks = np.asarray([0, 0.5, 1, 5, 10])
    axes[2].set_yticks(np.log1p(actual_ticks), ['0', '0.5', '1', '5', '10'])
    axes[2].set_xticks(range(len(NETWORK_ORDER)), NETWORK_ORDER)
    axes[2].set_ylabel('Network damaged (%)')
    axes[2].set_title('Seven-network lesion burden')
    axes[2].scatter([], [], color=GROUP_COLORS['Control'], s=18, label='Control')
    axes[2].scatter([], [], color=GROUP_COLORS['TMS'], s=18, label='TMS')
    axes[2].legend(loc='upper right', fontsize=6, ncol=2, handletextpad=0.3, columnspacing=0.8)
    for ax in axes:
        ax.tick_params(width=0.6, length=2.5)
    fig.text(0.01, 0.98, 'B', fontsize=12, fontweight='bold', va='top')
    fig.suptitle('Baseline lesion-burden distributions', fontsize=9, y=0.98)
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.91), w_pad=1.6)
    save_figure(fig, 'B_lesion_burden_distributions')
    plt.close(fig)

def merge_clinical(participants: pd.DataFrame, clinical: pd.DataFrame) -> pd.DataFrame:
    merged = participants.merge(clinical, on='subject_id', how='left', validate='one_to_one')
    if merged['Group'].isna().any():
        raise RuntimeError('Missing clinical row for a lesion participant')
    conflicts = merged.loc[merged['group'] != merged['Group'].astype(str), ['subject_id', 'group', 'Group']]
    if not conflicts.empty:
        raise RuntimeError(f'Group conflict in lesion cohort:\n{conflicts}')
    merged['group_tms'] = merged['group'].eq('TMS').astype(int)
    merged['age_z'] = zscore(merged['年龄'])
    merged['lesion_burden_z'] = zscore(merged['log1p_lesion_volume'])
    return merged

def fit_hc3(y: np.ndarray, predictors: pd.DataFrame) -> object:
    model = sm.OLS(np.asarray(y, dtype=float), sm.add_constant(predictors.astype(float), has_constant='add'), missing='raise')
    return model.fit(cov_type='HC3')

def clinical_model_table(data: pd.DataFrame, analysis: str) -> pd.DataFrame:
    rows = []
    for (outcome, (pre_column, post_column, direction)) in OUTCOMES.items():
        baseline = direction * data[pre_column].astype(float).to_numpy()
        post = direction * data[post_column].astype(float).to_numpy()
        baseline_sd = float(np.std(baseline, ddof=1))
        baseline_mean = float(np.mean(baseline))
        y = (post - baseline_mean) / baseline_sd
        predictors = pd.DataFrame({'baseline_outcome': (baseline - baseline_mean) / baseline_sd, 'group_tms': data['group_tms'].to_numpy(float), 'lesion_burden_z': zscore(data['log1p_lesion_volume']), 'group_x_lesion': data['group_tms'].to_numpy(float) * zscore(data['log1p_lesion_volume']), 'age_z': zscore(data['年龄']), 'sex_male': data['sex_male'].to_numpy(float), 'mask_lateralization_index': data['mask_lateralization_index'].to_numpy(float)})
        result = fit_hc3(y, predictors)
        for term in result.params.index:
            t_value = float(result.params[term] / result.bse[term])
            partial_r2 = t_value ** 2 / (t_value ** 2 + result.df_resid)
            rows.append({'analysis': analysis, 'outcome': outcome, 'n': int(result.nobs), 'term': term, 'standardized_beta': float(result.params[term]), 'ci_low': float(result.conf_int().loc[term, 0]), 'ci_high': float(result.conf_int().loc[term, 1]), 'robust_se_hc3': float(result.bse[term]), 'p_value': float(result.pvalues[term]), 'partial_r2': float(partial_r2), 'r_squared': float(result.rsquared)})
    table = pd.DataFrame(rows)
    table['fdr_q'] = np.nan
    for term in ('group_tms', 'lesion_burden_z', 'group_x_lesion'):
        mask = table['term'] == term
        table.loc[mask, 'fdr_q'] = multipletests(table.loc[mask, 'p_value'], method='fdr_bh')[1]
    return table

def plot_c_models(table: pd.DataFrame) -> None:
    terms = (('group_tms', 'Treatment group\n(TMS - Control)', '#D65F45'), ('lesion_burden_z', 'Log lesion burden', '#4C78A8'), ('group_x_lesion', 'Group x lesion', '#2A9D8F'))
    outcomes = list(OUTCOMES)
    selected = table.loc[(table['analysis'] == 'primary') & table['term'].isin([term[0] for term in terms])]
    maximum = float(np.nanmax(np.abs(selected[['ci_low', 'ci_high']].to_numpy())))
    limit = max(0.55, maximum * 1.12)
    (fig, axes) = plt.subplots(1, 3, figsize=(7.2, 3.15), sharey=True, facecolor='white')
    for (ax, (term, title, color)) in zip(axes, terms):
        subset = selected.loc[selected['term'] == term].set_index('outcome').loc[outcomes].reset_index()
        y = np.arange(len(outcomes))[::-1]
        errors = np.vstack([subset['standardized_beta'] - subset['ci_low'], subset['ci_high'] - subset['standardized_beta']])
        ax.axvline(0, color='#8C8C8C', linewidth=0.7, linestyle='--')
        for (row, ypos) in zip(subset.itertuples(), y):
            significant = np.isfinite(row.fdr_q) and row.fdr_q < 0.05
            ax.errorbar(row.standardized_beta, ypos, xerr=[[row.standardized_beta - row.ci_low], [row.ci_high - row.standardized_beta]], fmt='o', markersize=4.8, markerfacecolor=color if significant else 'white', markeredgecolor=color, markeredgewidth=0.9, ecolor=color, elinewidth=1.0, capsize=2)
            q_text = '<0.001' if row.fdr_q < 0.001 else f'={row.fdr_q:.3f}'
            ax.text(limit * 0.96, ypos, f'q{q_text}', ha='right', va='center', fontsize=5.7, color='#555555')
        ax.set_xlim(-limit, limit)
        ax.set_title(title)
        ax.set_xlabel('Standardized beta (95% CI)')
        ax.set_yticks(y, outcomes)
        ax.tick_params(width=0.6, length=2.5)
    axes[0].set_ylabel('Post-intervention outcome\n(higher = better; adjusted for baseline)')
    fig.text(0.01, 0.98, 'C', fontsize=12, fontweight='bold', va='top')
    fig.suptitle('Baseline-adjusted lesion moderation models', fontsize=9, y=0.98)
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.9), w_pad=1.2)
    save_figure(fig, 'C_baseline_adjusted_models')
    plt.close(fig)

def gradient_file(group: str, session: str, subject: str) -> Path:
    matches = sorted((GRADIENT_ROOT / group / session).glob(f'{subject}-*.npy'))
    if len(matches) != 1:
        raise RuntimeError(f'Expected one gradient file for {group}/{session}/{subject}, found {len(matches)}')
    return matches[0]

def gradient_subject_metrics(data: pd.DataFrame, labels: pd.DataFrame, network_table: pd.DataFrame) -> pd.DataFrame:
    roi_networks = labels.sort_values('roi')['network'].to_numpy()
    network_wide = network_table.pivot(index='subject_id', columns='network', values='network_damage_pct')
    rows = []
    for row in data.itertuples():
        before = np.load(gradient_file(row.group, 'before', row.subject_id))
        after = np.load(gradient_file(row.group, 'after', row.subject_id))
        if before.shape != (400, 5) or after.shape != (400, 5):
            raise RuntimeError(f'Unexpected gradient shape for {row.subject_id}')
        values = {'subject_id': row.subject_id, 'group': row.group, 'g1_reorganization_rms': float(np.sqrt(np.mean((after[:, 0] - before[:, 0]) ** 2))), 'g2_reorganization_rms': float(np.sqrt(np.mean((after[:, 1] - before[:, 1]) ** 2)))}
        for network in NETWORK_ORDER:
            roi_mask = roi_networks == network
            values[f'{network}_g1_before'] = float(np.mean(before[roi_mask, 0]))
            values[f'{network}_g1_after'] = float(np.mean(after[roi_mask, 0]))
            values[f'{network}_g1_change'] = values[f'{network}_g1_after'] - values[f'{network}_g1_before']
            values[f'{network}_lesion_pct'] = float(network_wide.loc[row.subject_id, network])
        rows.append(values)
    metrics = pd.DataFrame(rows)
    columns = ['subject_id', 'group_tms', '年龄', 'sex_male', 'mask_lateralization_index', 'log1p_lesion_volume']
    for (outcome, (pre, post, _)) in OUTCOMES.items():
        columns.extend((pre, post))
    metrics = metrics.merge(data[columns], on='subject_id', validate='one_to_one')
    baseline_components = []
    post_components = []
    for (_, (pre_column, post_column, direction)) in OUTCOMES.items():
        baseline = direction * metrics[pre_column].astype(float).to_numpy()
        post = direction * metrics[post_column].astype(float).to_numpy()
        mean = float(np.mean(baseline))
        sd = float(np.std(baseline, ddof=1))
        baseline_components.append((baseline - mean) / sd)
        post_components.append((post - mean) / sd)
    metrics['clinical_composite_baseline'] = np.mean(np.vstack(baseline_components), axis=0)
    metrics['clinical_composite_post'] = np.mean(np.vstack(post_components), axis=0)
    metrics['clinical_composite_change'] = metrics['clinical_composite_post'] - metrics['clinical_composite_baseline']
    metrics['age_z'] = zscore(metrics['年龄'])
    metrics['nihss_baseline_z'] = zscore(metrics['NIHSS_pre'])
    metrics['lesion_burden_z'] = zscore(metrics['log1p_lesion_volume'])
    metrics['g1_reorganization_z'] = zscore(metrics['g1_reorganization_rms'])
    return metrics

def gradient_network_models(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for network in NETWORK_ORDER:
        baseline = metrics[f'{network}_g1_before'].to_numpy(float)
        post = metrics[f'{network}_g1_after'].to_numpy(float)
        mean = float(np.mean(baseline))
        sd = float(np.std(baseline, ddof=1))
        burden = metrics[f'{network}_lesion_pct'].to_numpy(float)
        involved_n = int(np.count_nonzero(burden > 0))
        if involved_n == 0:
            rows.append({'network': network, 'n': len(metrics), 'involved_n': 0, 'term': 'network_lesion_burden_z', 'estimable': False, 'standardized_beta': np.nan, 'ci_low': np.nan, 'ci_high': np.nan, 'robust_se_hc3': np.nan, 'p_value': np.nan, 'r_squared': np.nan, 'estimability_note': 'no participant had direct lesion overlap with this network'})
            continue
        burden_z = zscore(np.log1p(burden))
        predictors = pd.DataFrame({'baseline_network_g1': (baseline - mean) / sd, 'group_tms': metrics['group_tms'].to_numpy(float), 'network_lesion_burden_z': burden_z, 'age_z': metrics['age_z'].to_numpy(float), 'sex_male': metrics['sex_male'].to_numpy(float), 'mask_lateralization_index': metrics['mask_lateralization_index'].to_numpy(float), 'nihss_baseline_z': metrics['nihss_baseline_z'].to_numpy(float)})
        result = fit_hc3((post - mean) / sd, predictors)
        term = 'network_lesion_burden_z'
        rows.append({'network': network, 'n': int(result.nobs), 'involved_n': involved_n, 'term': term, 'estimable': True, 'standardized_beta': float(result.params[term]), 'ci_low': float(result.conf_int().loc[term, 0]), 'ci_high': float(result.conf_int().loc[term, 1]), 'robust_se_hc3': float(result.bse[term]), 'p_value': float(result.pvalues[term]), 'r_squared': float(result.rsquared), 'estimability_note': 'sparse direct cortical overlap; interpret as exploratory'})
    table = pd.DataFrame(rows)
    table['fdr_q'] = np.nan
    estimable = table['estimable'] & table['p_value'].notna()
    table.loc[estimable, 'fdr_q'] = multipletests(table.loc[estimable, 'p_value'], method='fdr_bh')[1]
    return table

def residualize(values: np.ndarray, covariates: pd.DataFrame) -> np.ndarray:
    result = sm.OLS(np.asarray(values, dtype=float), sm.add_constant(covariates.astype(float), has_constant='add')).fit()
    return np.asarray(result.resid)

def global_models(metrics: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]]]:
    rows = []
    partial = {}
    specifications = {'G1_reorganization': (metrics['g1_reorganization_z'].to_numpy(float), pd.DataFrame({'group_tms': metrics['group_tms'], 'age_z': metrics['age_z'], 'sex_male': metrics['sex_male'], 'mask_lateralization_index': metrics['mask_lateralization_index'], 'nihss_baseline_z': metrics['nihss_baseline_z']})), 'clinical_composite_post': (metrics['clinical_composite_post'].to_numpy(float), pd.DataFrame({'clinical_composite_baseline': metrics['clinical_composite_baseline'], 'group_tms': metrics['group_tms'], 'age_z': metrics['age_z'], 'sex_male': metrics['sex_male'], 'mask_lateralization_index': metrics['mask_lateralization_index']}))}
    x = metrics['lesion_burden_z'].to_numpy(float)
    for (outcome, (y, covariates)) in specifications.items():
        predictors = covariates.copy()
        predictors.insert(0, 'lesion_burden_z', x)
        result = fit_hc3(y, predictors)
        term = 'lesion_burden_z'
        rows.append({'outcome': outcome, 'n': int(result.nobs), 'term': term, 'standardized_beta': float(result.params[term]), 'ci_low': float(result.conf_int().loc[term, 0]), 'ci_high': float(result.conf_int().loc[term, 1]), 'robust_se_hc3': float(result.bse[term]), 'p_value': float(result.pvalues[term]), 'r_squared': float(result.rsquared)})
        partial[outcome] = (residualize(x, covariates), residualize(y, covariates))
    table = pd.DataFrame(rows)
    table['fdr_q'] = multipletests(table['p_value'], method='fdr_bh')[1]
    return (table, partial)

def plot_partial(ax: plt.Axes, x: np.ndarray, y: np.ndarray, groups: np.ndarray, model_row: pd.Series, ylabel: str) -> None:
    for group in GROUPS:
        mask = groups == group
        ax.scatter(x[mask], y[mask], s=17, color=GROUP_COLORS[group], alpha=0.75, edgecolor='white', linewidth=0.35, label=group)
    fit = np.polyfit(x, y, 1)
    line_x = np.linspace(float(x.min()), float(x.max()), 100)
    ax.plot(line_x, np.polyval(fit, line_x), color='#333333', linewidth=1.0)
    ax.axhline(0, color='#B0B0B0', linewidth=0.5)
    ax.axvline(0, color='#B0B0B0', linewidth=0.5)
    ax.set_xlabel('Adjusted log lesion burden')
    ax.set_ylabel(ylabel)
    ax.text(0.04, 0.96, f"beta={model_row['standardized_beta']:.2f}\n95% CI {model_row['ci_low']:.2f}, {model_row['ci_high']:.2f}\np={model_row['p_value']:.3f}", transform=ax.transAxes, ha='left', va='top', fontsize=6)
    ax.tick_params(width=0.6, length=2.5)

def plot_d_gradient(network_models: pd.DataFrame, global_table: pd.DataFrame, partial: dict[str, tuple[np.ndarray, np.ndarray]], metrics: pd.DataFrame) -> None:
    (fig, axes) = plt.subplots(1, 3, figsize=(7.2, 3.3), gridspec_kw={'width_ratios': (1.15, 1, 1)}, facecolor='white')
    y = np.arange(len(NETWORK_ORDER))[::-1]
    axes[0].axvline(0, color='#8C8C8C', linewidth=0.7, linestyle='--')
    subset = network_models.set_index('network').loc[list(NETWORK_ORDER)].reset_index()
    estimable = subset['estimable'].astype(bool).to_numpy()
    axes[0].errorbar(subset.loc[estimable, 'standardized_beta'], y[estimable], xerr=np.vstack([subset.loc[estimable, 'standardized_beta'] - subset.loc[estimable, 'ci_low'], subset.loc[estimable, 'ci_high'] - subset.loc[estimable, 'standardized_beta']]), fmt='o', markersize=4.0, color='#4C78A8', ecolor='#4C78A8', elinewidth=0.9, capsize=1.8)
    for (row, ypos) in zip(subset.itertuples(), y):
        if not row.estimable:
            axes[0].text(0, ypos, 'not estimable', ha='center', va='center', fontsize=5.5, color='#777777')
    network_tick_labels = [f'{network} (n+={int(count)})' for (network, count) in zip(subset['network'], subset['involved_n'])]
    axes[0].set_yticks(y, network_tick_labels)
    axes[0].set_xlabel('Standardized beta (95% CI)')
    axes[0].set_title('Network burden -> post G1\nn+ = direct cortical overlap', fontsize=7.3)
    axes[0].tick_params(width=0.6, length=2.5)
    groups = metrics['group'].to_numpy()
    gradient_row = global_table.set_index('outcome').loc['G1_reorganization']
    plot_partial(axes[1], *partial['G1_reorganization'], groups, gradient_row, 'Adjusted G1 reorganization (RMS)')
    axes[1].set_title('Global gradient reorganization')
    clinical_row = global_table.set_index('outcome').loc['clinical_composite_post']
    plot_partial(axes[2], *partial['clinical_composite_post'], groups, clinical_row, 'Adjusted post clinical composite')
    axes[2].set_title('Multidomain clinical recovery')
    axes[2].legend(fontsize=5.8, loc='lower right')
    fig.text(0.01, 0.98, 'D', fontsize=12, fontweight='bold', va='top')
    fig.suptitle('Lesion burden, gradient reorganization, and clinical recovery', fontsize=9, y=0.98)
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.9), w_pad=1.4)
    save_figure(fig, 'D_lesion_gradient_clinical_associations')
    plt.close(fig)

def main() -> None:
    ensure_output_dirs()
    clinical = load_clinical()
    labels = load_atlas_labels()
    masks = list_masks()
    (participants, parcels, networks, arrays, reference) = extract_lesion_metrics(masks, labels)
    participants.to_csv(TABLES / 'B_participant_lesion_metrics.csv', index=False, encoding='utf-8-sig')
    parcels.to_csv(TABLES / 'B_schaefer400_parcel_burden.csv', index=False, encoding='utf-8-sig')
    networks.to_csv(TABLES / 'B_seven_network_burden.csv', index=False, encoding='utf-8-sig')
    proportions = save_overlap_maps(arrays, reference)
    plot_a_overlap(proportions, arrays)
    burden_stats = burden_group_statistics(participants, networks)
    burden_stats.to_csv(TABLES / 'B_group_comparisons.csv', index=False, encoding='utf-8-sig')
    plot_b_burden(participants, networks)
    analysis_data = merge_clinical(participants, clinical)
    primary = clinical_model_table(analysis_data, 'primary')
    sensitivity_large = clinical_model_table(analysis_data.loc[analysis_data['subject_id'] != 'PRIVATE_SUBJECT_040'].copy(), 'exclude_large_lesion_PRIVATE_SUBJECT_040')
    flagged = {'PRIVATE_SUBJECT_040', 'PRIVATE_SUBJECT_006', 'PRIVATE_SUBJECT_038'}
    sensitivity_flags = clinical_model_table(analysis_data.loc[~analysis_data['subject_id'].isin(flagged)].copy(), 'exclude_manual_review_flags')
    clinical_models = pd.concat([primary, sensitivity_large, sensitivity_flags], ignore_index=True)
    clinical_models.to_csv(TABLES / 'C_clinical_models.csv', index=False, encoding='utf-8-sig')
    plot_c_models(clinical_models)
    gradient_metrics = gradient_subject_metrics(analysis_data, labels, networks)
    gradient_metrics.to_csv(TABLES / 'D_gradient_clinical_subject_metrics.csv', index=False, encoding='utf-8-sig')
    network_models = gradient_network_models(gradient_metrics)
    network_models.to_csv(TABLES / 'D_network_gradient_models.csv', index=False, encoding='utf-8-sig')
    (global_table, partial) = global_models(gradient_metrics)
    global_table.to_csv(TABLES / 'D_global_lesion_associations.csv', index=False, encoding='utf-8-sig')
    plot_d_gradient(network_models, global_table, partial, gradient_metrics)
    manifest = {'analysis_date': '2026-08-03', 'baseline_lesion_n': int(len(participants)), 'group_n': participants.groupby('group').size().astype(int).to_dict(), 'gradient_intersection_n': int(len(gradient_metrics)), 'atlas': 'Schaefer2018 400 parcels, 7 networks, FSL MNI152 2 mm', 'primary_lesion_metric': 'log1p baseline lesion volume in ml', 'network_burden': "percent of each Schaefer network's voxels intersected by the baseline lesion", 'clinical_model': 'oriented post outcome ~ oriented baseline + group + log lesion volume + group x volume + age + sex + mask lateralization; HC3 SE', 'manual_review_flags': ['PRIVATE_SUBJECT_040', 'PRIVATE_SUBJECT_006', 'PRIVATE_SUBJECT_038'], 'target_network_status': 'not prespecified because the TMS target is unavailable; all seven networks are shown', 'network_model_estimability': 'TMS had no direct Schaefer cortical overlap; group-by-network-burden interactions were not estimated. VIS and LIM had no overlap in either group.', 'post_mask_policy': 'post-treatment lesion masks are QC/sensitivity data and are not treatment predictors'}
    (OUT / 'analysis_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Participants: {len(participants)} | Control={sum(participants.group == 'Control')} | TMS={sum(participants.group == 'TMS')}")
    print(f'Gradient intersection: {len(gradient_metrics)}')
    print(f'Figures: {FIGURES}')
    print(f'Tables: {TABLES}')
    print(f'NIfTI maps: {NIFTI}')
if __name__ == '__main__':
    main()
