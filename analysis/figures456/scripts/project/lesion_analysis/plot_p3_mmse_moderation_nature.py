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
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from PIL import Image
ROOT = Path(_portable_path('project', 'lesion_analysis'))
ANALYSIS_ROOT = ROOT / 'lesion_gradient_score_triangle_exploratory'
TABLE_DIR = ANALYSIS_ROOT / 'tables'
FIGURE_DIR = ANALYSIS_ROOT / 'figures' / 'nature'
SOURCE_DIR = FIGURE_DIR / 'source_data'
DATA_PATH = TABLE_DIR / 'analysis_dataset_37.csv'
MODERATION_PATH = TABLE_DIR / 'moderation_interactions.csv'
SLOPES_PATH = TABLE_DIR / 'lesion_gradient_simple_slopes.csv'
SENSITIVITY_PATH = TABLE_DIR / 'mmse_moderation_sensitivity.csv'
MM_PER_INCH = 25.4
FIG_WIDTH_MM = 183
FIG_HEIGHT_MM = 126
COLORS = {'low': '#3B6FB6', 'mean': '#6F6F6F', 'high': '#C96A3D', 'dark': '#272727', 'mid': '#737373', 'light': '#D8D8D8', 'band_blue': '#EDF3FA', 'band_orange': '#FBF1EC', 'band_grey': '#F2F2F2'}
mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'], 'font.size': 7, 'axes.titlesize': 7.5, 'axes.labelsize': 7, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'axes.spines.right': False, 'axes.spines.top': False, 'axes.linewidth': 0.7, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7, 'xtick.major.size': 3, 'ytick.major.size': 3, 'legend.frameon': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'savefig.facecolor': 'white', 'figure.facecolor': 'white'})
MODEL_TERMS = ['x_z', 'm_z', 'MMSE_baseline_function_z', 'group_tms', 'baseline_m_z', 'volume_z', 'pre_fd_z', 'post_fd_z', 'x_by_m']

def design(frame: pd.DataFrame) -> pd.DataFrame:
    return sm.add_constant(frame[MODEL_TERMS].astype(float), has_constant='add')

def load_and_fit() -> tuple[pd.DataFrame, object]:
    data = pd.read_csv(DATA_PATH, dtype={'subject_id': str})
    if len(data) != 37:
        raise RuntimeError(f'Expected n=37, found {len(data)}')
    result = sm.OLS(data['MMSE_improvement_z'], design(data)).fit(cov_type='HC3')
    return (data, result)

def covariate_adjusted_outcome(data: pd.DataFrame, result) -> np.ndarray:
    mean_group = float(data['group_tms'].mean())
    structural = float(result.params['const']) + float(result.params['group_tms']) * mean_group + float(result.params['x_z']) * data['x_z'].to_numpy(float) + float(result.params['m_z']) * data['m_z'].to_numpy(float) + float(result.params['x_by_m']) * data['x_by_m'].to_numpy(float)
    return structural + result.resid.to_numpy(float)

def marginal_prediction_lines(data: pd.DataFrame, result) -> pd.DataFrame:
    mean_group = float(data['group_tms'].mean())
    grid = np.linspace(float(data['m_z'].quantile(0.03)), float(data['m_z'].quantile(0.97)), 160)
    rows: list[dict[str, object]] = []
    covariance = result.cov_params().to_numpy(float)
    names = list(result.params.index)
    for (lesion_z, label, color_key) in [(-1.0, 'Low span (-1 SD)', 'low'), (0.0, 'Mean span', 'mean'), (1.0, 'High span (+1 SD)', 'high')]:
        for gradient_z in grid:
            values = {'const': 1.0, 'x_z': lesion_z, 'm_z': gradient_z, 'MMSE_baseline_function_z': 0.0, 'group_tms': mean_group, 'baseline_m_z': 0.0, 'volume_z': 0.0, 'pre_fd_z': 0.0, 'post_fd_z': 0.0, 'x_by_m': lesion_z * gradient_z}
            vector = np.asarray([values[name] for name in names], dtype=float)
            prediction = float(vector @ result.params.to_numpy(float))
            se = float(np.sqrt(max(vector @ covariance @ vector, 0.0)))
            rows.append({'lesion_span_z': lesion_z, 'span_label': label, 'color_key': color_key, 'gradient_normalization_z': gradient_z, 'predicted_MMSE_improvement_z': prediction, 'ci_low_hc3': prediction - 1.96 * se, 'ci_high_hc3': prediction + 1.96 * se})
    return pd.DataFrame(rows)

def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.1, 1.035, label, transform=ax.transAxes, fontsize=8, fontweight='bold', va='bottom', ha='left')

def clean_axis(ax: plt.Axes) -> None:
    ax.spines['left'].set_color('#4A4A4A')
    ax.spines['bottom'].set_color('#4A4A4A')
    ax.tick_params(colors='#333333')
    ax.xaxis.label.set_color('#222222')
    ax.yaxis.label.set_color('#222222')

def plot_interaction(ax: plt.Axes, data: pd.DataFrame, adjusted_y: np.ndarray, predictions: pd.DataFrame, moderation: pd.Series, with_label: bool) -> None:
    norm = mpl.colors.Normalize(vmin=-1.6, vmax=1.6)
    cmap = mpl.colors.LinearSegmentedColormap.from_list('span_map', [COLORS['low'], '#D4D4D4', COLORS['high']])
    scatter = ax.scatter(data['m_z'], adjusted_y, c=data['x_z'], cmap=cmap, norm=norm, s=23, alpha=0.78, edgecolor='white', linewidth=0.45, zorder=4)
    for color_key in ('low', 'mean', 'high'):
        frame = predictions.loc[predictions['color_key'] == color_key]
        x = frame['gradient_normalization_z'].to_numpy(float)
        y = frame['predicted_MMSE_improvement_z'].to_numpy(float)
        lo = frame['ci_low_hc3'].to_numpy(float)
        hi = frame['ci_high_hc3'].to_numpy(float)
        color = COLORS[color_key]
        ax.fill_between(x, lo, hi, color=color, alpha=0.085, linewidth=0, zorder=1)
        ax.plot(x, y, color=color, lw=1.55, zorder=3)
    ax.axhline(0, color='#A0A0A0', lw=0.65, ls=(0, (2.5, 2.5)), zorder=0)
    ax.set_xlabel('Nearest G1-G2 normalization (z)')
    ax.set_ylabel('Covariate-adjusted MMSE improvement (z)')
    ax.set_title('Lesion span conditions the gradient-MMSE association', loc='left', fontweight='bold', pad=4)
    ax.text(0.025, 0.975, f"Interaction beta={float(moderation['adjusted_beta']):.3f}\nstudentized permutation P={float(moderation['adjusted_p_freedman_lane']):.3f}; FDR q={float(moderation['fdr_q_freedman_lane_all_13']):.3f}\nn=37; exploratory", transform=ax.transAxes, va='top', ha='left', fontsize=6.4, color='#333333', linespacing=1.25)
    handles = [mpl.lines.Line2D([0], [0], color=COLORS['low'], lw=1.6, label='Low span (-1 SD)'), mpl.lines.Line2D([0], [0], color=COLORS['mean'], lw=1.6, label='Mean span'), mpl.lines.Line2D([0], [0], color=COLORS['high'], lw=1.6, label='High span (+1 SD)')]
    ax.legend(handles=handles, loc='lower right', fontsize=6.1, handlelength=1.8, labelspacing=0.35, borderaxespad=0.3)
    cax = ax.inset_axes([0.035, 0.055, 0.3, 0.022])
    cbar = ax.figure.colorbar(scatter, cax=cax, orientation='horizontal')
    cbar.set_label('Observed lesion span (z)', fontsize=5.8, labelpad=1.5)
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.tick_params(labelsize=5.8, width=0.5, length=2)
    cbar.outline.set_linewidth(0.5)
    clean_axis(ax)
    if with_label:
        panel_label(ax, 'a')

def plot_simple_slopes(ax: plt.Axes, slopes: pd.DataFrame, with_label: bool) -> None:
    order = ['span_minus_1SD', 'span_mean', 'span_plus_1SD']
    labels = ['Low span\n(-1 SD)', 'Mean span', 'High span\n(+1 SD)']
    color_keys = ['low', 'mean', 'high']
    frame = slopes.loc[(slopes['outcome'] == 'MMSE') & slopes['lesion_span_level'].isin(order)].set_index('lesion_span_level').loc[order]
    y = np.arange(len(order))[::-1]
    for (yi, (_, row), color_key) in zip(y, frame.iterrows(), color_keys):
        color = COLORS[color_key]
        ax.plot([row['ci_low_hc3'], row['ci_high_hc3']], [yi, yi], color=color, lw=1.45, solid_capstyle='round')
        ax.scatter(row['gradient_slope'], yi, s=28, color=color, edgecolor='white', linewidth=0.45, zorder=3)
        ax.text(1.52, yi, f"P={float(row['p_hc3']):.3f}", va='center', ha='left', fontsize=5.8, color='#4A4A4A')
    ax.axvline(0, color='#8A8A8A', lw=0.7, ls=(0, (3, 2)))
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(-0.55, 1.88)
    ax.set_xlabel('Gradient-MMSE slope (beta)')
    ax.set_title('Conditional simple slopes', loc='left', fontweight='bold', pad=4)
    ax.text(0.02, -0.31, 'Points: beta; lines: 95% HC3 CI', transform=ax.transAxes, fontsize=5.7, color='#555555', va='top')
    clean_axis(ax)
    if with_label:
        panel_label(ax, 'b')

def sensitivity_display(sensitivity: pd.DataFrame) -> pd.DataFrame:
    specifications = [('cohort_sensitivity', 'primary_motion_qc', 'Primary QC (n=37)', 'cohort'), ('cohort_sensitivity', 'all_38', 'All participants (n=38)', 'cohort'), ('cohort_sensitivity', 'exclude_manual_flags', 'Exclude QC flags (n=34)', 'cohort'), ('cohort_sensitivity', 'exclude_group_conflict', 'Exclude group conflict (n=36)', 'cohort'), ('covariate_sensitivity', 'without_baseline_gradient', 'No baseline gradient', 'model'), ('covariate_sensitivity', 'without_motion', 'No motion covariates', 'model'), ('covariate_sensitivity', 'minimal', 'Minimal adjustment', 'model'), ('robust_estimator', 'primary_full', 'Huber robust model', 'estimator')]
    rows = []
    for (analysis, variant, label, family) in specifications:
        row = sensitivity.loc[(sensitivity['analysis'] == analysis) & (sensitivity['variant'] == variant)].iloc[0]
        rows.append({**row.to_dict(), 'display_label': label, 'family': family})
    return pd.DataFrame(rows)

def plot_sensitivity(ax: plt.Axes, sensitivity: pd.DataFrame, with_label: bool) -> None:
    frame = sensitivity_display(sensitivity)
    y = np.arange(len(frame))[::-1]
    family_colors = {'cohort': COLORS['low'], 'model': COLORS['mean'], 'estimator': COLORS['high']}
    ax.axhspan(3.5, 7.5, color=COLORS['band_blue'], zorder=0)
    ax.axhspan(0.5, 3.5, color=COLORS['band_grey'], zorder=0)
    ax.axhspan(-0.5, 0.5, color=COLORS['band_orange'], zorder=0)
    for (yi, (_, row)) in zip(y, frame.iterrows()):
        color = family_colors[row['family']]
        ax.plot([row['ci_low'], row['ci_high']], [yi, yi], color=color, lw=1.25)
        ax.scatter(row['interaction_beta'], yi, color=color, s=22, edgecolor='white', linewidth=0.4, zorder=3)
    ax.axvline(0, color='#7A7A7A', lw=0.7, ls=(0, (3, 2)))
    ax.set_yticks(y)
    ax.set_yticklabels(frame['display_label'])
    ax.set_xlim(-0.12, 0.72)
    ax.set_xlabel('Interaction beta')
    ax.set_title('Sensitivity of the MMSE interaction', loc='left', fontweight='bold', pad=13)
    ax.text(0.0, 1.012, 'Overall interaction family: FDR q=0.271', transform=ax.transAxes, ha='left', va='bottom', fontsize=5.8, color='#555555')
    clean_axis(ax)
    if with_label:
        panel_label(ax, 'c')

def save_bundle(fig: plt.Figure, base_name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    base = FIGURE_DIR / base_name
    fig.savefig(base.with_suffix('.pdf'), bbox_inches='tight', pad_inches=0.04)
    fig.savefig(base.with_suffix('.svg'), bbox_inches='tight', pad_inches=0.04)
    fig.savefig(base.with_suffix('.png'), dpi=600, bbox_inches='tight', pad_inches=0.04)
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=600, bbox_inches='tight', pad_inches=0.04)
    buffer.seek(0)
    Image.open(buffer).convert('RGB').save(base.with_suffix('.tiff'), format='TIFF', compression='tiff_lzw', dpi=(600, 600))

def make_composite(data: pd.DataFrame, adjusted_y: np.ndarray, predictions: pd.DataFrame, moderation: pd.Series, slopes: pd.DataFrame, sensitivity: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(FIG_WIDTH_MM / MM_PER_INCH, FIG_HEIGHT_MM / MM_PER_INCH))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.47, 1.0], height_ratios=[0.88, 1.18], left=0.075, right=0.985, bottom=0.115, top=0.94, wspace=0.6, hspace=0.58)
    ax_a = fig.add_subplot(grid[:, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 1])
    plot_interaction(ax_a, data, adjusted_y, predictions, moderation, with_label=True)
    plot_simple_slopes(ax_b, slopes, with_label=True)
    plot_sensitivity(ax_c, sensitivity, with_label=True)
    save_bundle(fig, 'lesion_span_gradient_MMSE_moderation_nature')
    plt.close(fig)

def make_standalones(data: pd.DataFrame, adjusted_y: np.ndarray, predictions: pd.DataFrame, moderation: pd.Series, slopes: pd.DataFrame, sensitivity: pd.DataFrame) -> None:
    specs = [('panel_a_interaction_standalone', (92 / MM_PER_INCH, 78 / MM_PER_INCH), lambda ax: plot_interaction(ax, data, adjusted_y, predictions, moderation, with_label=False)), ('panel_b_simple_slopes_standalone', (89 / MM_PER_INCH, 69 / MM_PER_INCH), lambda ax: plot_simple_slopes(ax, slopes, with_label=False)), ('panel_c_sensitivity_standalone', (104 / MM_PER_INCH, 91 / MM_PER_INCH), lambda ax: plot_sensitivity(ax, sensitivity, with_label=False))]
    for (name, size, plotter) in specs:
        (fig, ax) = plt.subplots(figsize=size)
        plotter(ax)
        fig.subplots_adjust(left=0.2 if 'panel_c' in name else 0.18, right=0.96, bottom=0.2, top=0.9)
        save_bundle(fig, name)
        plt.close(fig)

def export_source_data(data: pd.DataFrame, adjusted_y: np.ndarray, predictions: pd.DataFrame, moderation: pd.Series, slopes: pd.DataFrame, sensitivity: pd.DataFrame) -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    points = data[['subject_id', 'group', 'lesion_g1g2_span', 'x_z', 'gradient_normalization', 'm_z', 'MMSE_pre', 'MMSE_post', 'MMSE_improvement', 'MMSE_improvement_z']].copy()
    points['covariate_adjusted_MMSE_improvement_z'] = adjusted_y
    points.to_csv(SOURCE_DIR / 'panel_a_adjusted_points.csv', index=False)
    predictions.to_csv(SOURCE_DIR / 'panel_a_prediction_lines.csv', index=False)
    slopes.loc[slopes['outcome'] == 'MMSE'].to_csv(SOURCE_DIR / 'panel_b_simple_slopes.csv', index=False)
    sensitivity_display(sensitivity).to_csv(SOURCE_DIR / 'panel_c_sensitivity.csv', index=False)
    moderation.to_frame().T.to_csv(SOURCE_DIR / 'interaction_statistics.csv', index=False)

def write_notes() -> None:
    notes = "# Figure notes: lesion span x gradient normalization on MMSE improvement\n\n## Core conclusion\n\nExploratory evidence suggests that lesion-related G1-G2 span may condition the association between healthy-reference gradient normalization and MMSE improvement. The interaction is nominally supported but does not survive family-wide FDR correction.\n\n## Panels\n\n- a: Covariate-adjusted subject-level MMSE improvement and marginal HC3 prediction lines at lesion-span z = -1, 0, and +1. Point colour encodes each participant's observed lesion span.\n- b: Conditional gradient-MMSE slopes at low, mean, and high lesion span. Lines are 95% HC3 confidence intervals.\n- c: Cohort, covariate, and robust-estimator sensitivity analyses for the interaction coefficient.\n\n## Statistics\n\n- Biological replicates: 37 stroke participants (22 TMS, 15 Control).\n- Primary model: MMSE improvement z ~ lesion span z + normalization z + their interaction + baseline MMSE + treatment group + baseline gradient deviation + log lesion volume + pre/post mean FD.\n- Interaction: beta = 0.333; HC3 P = 0.043; studentized within-group Freedman-Lane P = 0.026; FDR q across 13 interaction tests = 0.271.\n- This is an exploratory moderation result, not a causal or FDR-confirmed effect.\n\n## Image integrity\n\n- Quantitative vector figure; no microscopy, image enhancement, cropping, or pseudo-colour manipulation.\n- All plotted values are traceable to the CSV files in source_data.\n"
    (FIGURE_DIR / 'FIGURE_NOTES.md').write_text(notes, encoding='utf-8')

def main() -> None:
    (data, model) = load_and_fit()
    adjusted_y = covariate_adjusted_outcome(data, model)
    predictions = marginal_prediction_lines(data, model)
    moderation_table = pd.read_csv(MODERATION_PATH)
    moderation = moderation_table.loc[(moderation_table['relationship'] == 'lesion_modifies_gradient_to_clinical') & (moderation_table['outcome'] == 'MMSE')].iloc[0]
    slopes = pd.read_csv(SLOPES_PATH)
    sensitivity = pd.read_csv(SENSITIVITY_PATH)
    make_composite(data, adjusted_y, predictions, moderation, slopes, sensitivity)
    make_standalones(data, adjusted_y, predictions, moderation, slopes, sensitivity)
    export_source_data(data, adjusted_y, predictions, moderation, slopes, sensitivity)
    write_notes()
    manifest = {'backend': 'Python/matplotlib', 'figure_size_mm': [FIG_WIDTH_MM, FIG_HEIGHT_MM], 'n': 37, 'outputs': ['lesion_span_gradient_MMSE_moderation_nature', 'panel_a_interaction_standalone', 'panel_b_simple_slopes_standalone', 'panel_c_sensitivity_standalone'], 'formats': ['pdf', 'svg', 'png_600dpi', 'tiff_RGB_600dpi'], 'claim_status': 'exploratory; interaction not family-wide FDR significant'}
    (FIGURE_DIR / 'figure_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(FIGURE_DIR)
if __name__ == '__main__':
    main()
