from __future__ import annotations
import os as _portable_os
from pathlib import Path as _PortablePath

def _portable_path(kind, relative):
    project = _PortablePath(_portable_os.environ['PAPER_PROJECT_ROOT'])
    root = project if kind == 'project' else project.parent
    return str(root / relative)
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from figure_common import INK, MID, SOURCE, save_submission_figure
WORKSPACE = next((parent for parent in Path(__file__).resolve().parents if parent.name == 'lesion_analysis'))
TABLE_DIR = WORKSPACE / 'lesion_gradient_score_triangle_exploratory' / 'tables'
DATA_PATH = TABLE_DIR / 'analysis_dataset_37.csv'
MODERATION_PATH = TABLE_DIR / 'moderation_interactions.csv'
SLOPES_PATH = TABLE_DIR / 'lesion_gradient_simple_slopes.csv'
LOW = '#3F73A8'
MEAN = '#747474'
HIGH = '#D6534A'
PURPLE = '#745A9B'
PURPLE_LIGHT = '#E4DDEC'
POINT_GREY = '#A6ABB1'
ZERO = '#A5A5A5'
MODEL_TERMS = ['x_z', 'm_z', 'MMSE_baseline_function_z', 'group_tms', 'baseline_m_z', 'volume_z', 'pre_fd_z', 'post_fd_z', 'x_by_m']

def load_analysis() -> tuple[pd.DataFrame, object, pd.Series, pd.DataFrame]:
    data = pd.read_csv(DATA_PATH, dtype={'subject_id': str})
    design = sm.add_constant(data[MODEL_TERMS].astype(float), has_constant='add')
    result = sm.OLS(data['MMSE_improvement_z'], design).fit(cov_type='HC3')
    moderation_table = pd.read_csv(MODERATION_PATH)
    moderation = moderation_table.loc[moderation_table['relationship'].eq('lesion_modifies_gradient_to_clinical') & moderation_table['outcome'].eq('MMSE')].iloc[0]
    slopes = pd.read_csv(SLOPES_PATH)
    slopes = slopes.loc[slopes['outcome'].eq('MMSE')].copy()
    return (data, result, moderation, slopes)

def adjusted_outcome(data: pd.DataFrame, result) -> np.ndarray:
    structural = float(result.params['const']) + float(result.params['group_tms']) * float(data['group_tms'].mean()) + float(result.params['x_z']) * data['x_z'].to_numpy(float) + float(result.params['m_z']) * data['m_z'].to_numpy(float) + float(result.params['x_by_m']) * data['x_by_m'].to_numpy(float)
    return structural + result.resid.to_numpy(float)

def prediction_at_span(data: pd.DataFrame, result, lesion_span_z: float, gradient_grid: np.ndarray) -> pd.DataFrame:
    names = list(result.params.index)
    covariance = result.cov_params().to_numpy(float)
    group_mean = float(data['group_tms'].mean())
    rows: list[dict[str, float]] = []
    for gradient_z in gradient_grid:
        values = {'const': 1.0, 'x_z': lesion_span_z, 'm_z': float(gradient_z), 'MMSE_baseline_function_z': 0.0, 'group_tms': group_mean, 'baseline_m_z': 0.0, 'volume_z': 0.0, 'pre_fd_z': 0.0, 'post_fd_z': 0.0, 'x_by_m': lesion_span_z * float(gradient_z)}
        vector = np.asarray([values[name] for name in names], dtype=float)
        prediction = float(vector @ result.params.to_numpy(float))
        se = float(np.sqrt(max(vector @ covariance @ vector, 0.0)))
        rows.append({'lesion_span_z': lesion_span_z, 'gradient_normalization_z': float(gradient_z), 'predicted_MMSE_improvement_z': prediction, 'ci_low_hc3': prediction - 1.96 * se, 'ci_high_hc3': prediction + 1.96 * se})
    return pd.DataFrame(rows)

def conditional_slope_curve(data: pd.DataFrame, result) -> pd.DataFrame:
    span_grid = np.linspace(float(data['x_z'].min()) - 0.05, float(data['x_z'].max()) + 0.05, 300)
    covariance = result.cov_params()
    beta_gradient = float(result.params['m_z'])
    beta_interaction = float(result.params['x_by_m'])
    var_gradient = float(covariance.loc['m_z', 'm_z'])
    var_interaction = float(covariance.loc['x_by_m', 'x_by_m'])
    covariance_terms = float(covariance.loc['m_z', 'x_by_m'])
    slope = beta_gradient + beta_interaction * span_grid
    se = np.sqrt(np.maximum(var_gradient + span_grid ** 2 * var_interaction + 2.0 * span_grid * covariance_terms, 0.0))
    return pd.DataFrame({'lesion_span_z': span_grid, 'conditional_gradient_MMSE_slope': slope, 'ci_low_hc3': slope - 1.96 * se, 'ci_high_hc3': slope + 1.96 * se})

def style_axis(axis: plt.Axes) -> None:
    axis.spines['left'].set_color('#454545')
    axis.spines['bottom'].set_color('#454545')
    axis.tick_params(colors='#333333')
    axis.grid(False)

def statistics_text(moderation: pd.Series) -> str:
    return f"Interaction β={float(moderation['adjusted_beta']):.3f}; permutation P={float(moderation['adjusted_p_freedman_lane']):.3f}; FDR q={float(moderation['fdr_q_freedman_lane_all_13']):.3f}"

def version_1_direct_slopes(data: pd.DataFrame, adjusted_y: np.ndarray, result, moderation: pd.Series, slopes: pd.DataFrame) -> pd.DataFrame:
    gradient_grid = np.linspace(float(data['m_z'].quantile(0.03)), float(data['m_z'].quantile(0.97)), 180)
    slope_lookup = slopes.set_index('lesion_span_level')
    specifications = [(-1.0, 'Low span (−1 SD)', 'span_minus_1SD', LOW), (0.0, 'Mean span', 'span_mean', MEAN), (1.0, 'High span (+1 SD)', 'span_plus_1SD', HIGH)]
    (fig, axis) = plt.subplots(figsize=(108 / 25.4, 82 / 25.4), facecolor='white')
    axis.scatter(data['m_z'], adjusted_y, s=25, facecolor=POINT_GREY, edgecolor='white', linewidth=0.45, alpha=0.67, zorder=4)
    prediction_frames: list[pd.DataFrame] = []
    for (lesion_z, label, slope_key, color) in specifications:
        frame = prediction_at_span(data, result, lesion_z, gradient_grid)
        frame['display_label'] = label
        prediction_frames.append(frame)
        x = frame['gradient_normalization_z'].to_numpy(float)
        y = frame['predicted_MMSE_improvement_z'].to_numpy(float)
        low = frame['ci_low_hc3'].to_numpy(float)
        high = frame['ci_high_hc3'].to_numpy(float)
        axis.fill_between(x, low, high, color=color, alpha=0.075, linewidth=0, zorder=1)
        axis.plot(x, y, color=color, linewidth=1.75, zorder=3)
        slope_value = float(slope_lookup.loc[slope_key, 'gradient_slope'])
        axis.text(float(x[-1]) + 0.08, float(y[-1]), f'{label}\nβ={slope_value:.2f}', color=color, fontsize=6.0, ha='left', va='center', linespacing=1.05)
    axis.axhline(0, color=ZERO, linewidth=0.7, linestyle=(0, (3, 2)), zorder=0)
    axis.set_xlim(float(gradient_grid.min()) - 0.12, float(gradient_grid.max()) + 0.8)
    y_values = np.concatenate([adjusted_y, np.concatenate([f['ci_low_hc3'] for f in prediction_frames]), np.concatenate([f['ci_high_hc3'] for f in prediction_frames])])
    axis.set_ylim(float(np.nanmin(y_values)) - 0.15, float(np.nanmax(y_values)) + 0.18)
    axis.set_xlabel('Nearest G1–G2 normalization (z)')
    axis.set_ylabel('Covariate-adjusted MMSE improvement (z)')
    axis.set_title('Normalization relates more strongly to MMSE improvement\nwhen lesion span is high', loc='left', fontweight='bold', pad=5)
    axis.text(0.01, 0.985, statistics_text(moderation) + '\nn=37; exploratory', transform=axis.transAxes, ha='left', va='top', fontsize=5.7, color=MID, linespacing=1.25)
    axis.text(0.99, 0.02, 'Grey points: adjusted participant observations', transform=axis.transAxes, ha='right', va='bottom', fontsize=5.4, color=MID)
    style_axis(axis)
    fig.subplots_adjust(left=0.17, right=0.94, bottom=0.17, top=0.83)
    save_submission_figure(fig, 'panel_a_interaction_version1_direct_slopes')
    return pd.concat(prediction_frames, ignore_index=True)

def version_2_faceted_tertiles(data: pd.DataFrame, adjusted_y: np.ndarray, result, moderation: pd.Series) -> pd.DataFrame:
    display = data.copy()
    display['adjusted_MMSE_z'] = adjusted_y
    display['span_tertile'] = pd.qcut(display['x_z'], q=3, labels=['Low', 'Intermediate', 'High'])
    gradient_grid = np.linspace(float(data['m_z'].quantile(0.03)), float(data['m_z'].quantile(0.97)), 160)
    colors = {'Low': LOW, 'Intermediate': MEAN, 'High': HIGH}
    (fig, axes) = plt.subplots(1, 3, figsize=(180 / 25.4, 75 / 25.4), sharex=True, sharey=True, facecolor='white')
    prediction_frames: list[pd.DataFrame] = []
    y_candidates = [adjusted_y]
    for (axis, level) in zip(axes, ['Low', 'Intermediate', 'High']):
        subset = display.loc[display['span_tertile'].astype(str).eq(level)].copy()
        representative_span = float(subset['x_z'].median())
        prediction = prediction_at_span(data, result, representative_span, gradient_grid)
        prediction['span_tertile'] = level
        prediction['representative_span_z'] = representative_span
        prediction_frames.append(prediction)
        color = colors[level]
        axis.fill_between(prediction['gradient_normalization_z'], prediction['ci_low_hc3'], prediction['ci_high_hc3'], color=color, alpha=0.12, linewidth=0)
        axis.plot(prediction['gradient_normalization_z'], prediction['predicted_MMSE_improvement_z'], color=color, linewidth=1.8)
        axis.scatter(subset['m_z'], subset['adjusted_MMSE_z'], s=28, facecolor=color, edgecolor='white', linewidth=0.5, alpha=0.76, zorder=4)
        conditional_beta = float(result.params['m_z'] + representative_span * result.params['x_by_m'])
        axis.axhline(0, color=ZERO, linewidth=0.65, linestyle=(0, (3, 2)), zorder=0)
        axis.set_title(f'{level} lesion span\n(n={len(subset)}; median z={representative_span:+.2f})', fontweight='bold', color=color, pad=5)
        axis.text(0.05, 0.94, f'Model slope β={conditional_beta:.2f}', transform=axis.transAxes, ha='left', va='top', fontsize=5.8, color=color)
        style_axis(axis)
        y_candidates.extend([prediction['ci_low_hc3'].to_numpy(float), prediction['ci_high_hc3'].to_numpy(float)])
    y_all = np.concatenate(y_candidates)
    for axis in axes:
        axis.set_ylim(float(np.nanmin(y_all)) - 0.12, float(np.nanmax(y_all)) + 0.12)
        axis.set_xlabel('G1–G2 normalization (z)')
    axes[0].set_ylabel('Adjusted MMSE improvement (z)')
    fig.suptitle('Lesion span modifies the normalization–MMSE relationship', x=0.08, y=0.97, ha='left', fontsize=8.5, fontweight='bold', color=INK)
    fig.text(0.08, 0.905, statistics_text(moderation) + '; association strengthens from low to high lesion span; tertiles are for display only', ha='left', va='top', fontsize=5.7, color=MID)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.18, top=0.77, wspace=0.17)
    save_submission_figure(fig, 'panel_a_interaction_version2_faceted_tertiles')
    return pd.concat(prediction_frames, ignore_index=True)

def version_3_conditional_effect(data: pd.DataFrame, result, moderation: pd.Series, slopes: pd.DataFrame) -> pd.DataFrame:
    curve = conditional_slope_curve(data, result)
    x = curve['lesion_span_z'].to_numpy(float)
    y = curve['conditional_gradient_MMSE_slope'].to_numpy(float)
    low = curve['ci_low_hc3'].to_numpy(float)
    high = curve['ci_high_hc3'].to_numpy(float)
    (fig, axis) = plt.subplots(figsize=(108 / 25.4, 82 / 25.4), facecolor='white')
    positive = low > 0
    if np.any(positive):
        threshold = float(x[np.flatnonzero(positive)[0]])
        axis.axvspan(threshold, float(x.max()), color=HIGH, alpha=0.065, linewidth=0)
        axis.text(threshold + 0.04, float(np.nanmax(high)) - 0.06, 'Pointwise 95% HC3 CI above 0', color=HIGH, fontsize=5.5, ha='left', va='top')
    axis.fill_between(x, low, high, color=PURPLE_LIGHT, alpha=0.82, linewidth=0)
    axis.plot(x, y, color=PURPLE, linewidth=1.9)
    axis.axhline(0, color=ZERO, linewidth=0.75, linestyle=(0, (3, 2)))
    order = ['span_minus_1SD', 'span_mean', 'span_plus_1SD']
    labels = ['−1 SD', 'Mean', '+1 SD']
    colors = [LOW, MEAN, HIGH]
    slope_lookup = slopes.set_index('lesion_span_level')
    for (span_z, key, label, color) in zip([-1.0, 0.0, 1.0], order, labels, colors):
        row = slope_lookup.loc[key]
        axis.errorbar(span_z, float(row['gradient_slope']), yerr=np.asarray([[float(row['gradient_slope'] - row['ci_low_hc3'])], [float(row['ci_high_hc3'] - row['gradient_slope'])]]), fmt='o', markersize=5.0, markerfacecolor=color, markeredgecolor='white', markeredgewidth=0.5, ecolor=color, elinewidth=1.1, capsize=2.2, zorder=5)
        axis.text(span_z, float(row['ci_high_hc3']) + 0.055, label, color=color, fontsize=5.7, ha='center', va='bottom')
    axis.scatter(data['x_z'], np.full(len(data), 0.025), marker='|', s=40, color='#8F949A', alpha=0.52, transform=axis.get_xaxis_transform(), clip_on=False)
    axis.text(0.99, 0.035, 'Observed lesion-span distribution', transform=axis.transAxes, ha='right', va='bottom', fontsize=5.4, color=MID)
    axis.set_xlabel('Lesion G1–G2 span (z)')
    axis.set_ylabel('Conditional normalization–MMSE slope (β)')
    axis.set_title('The clinical relevance of normalization increases with lesion span', loc='left', fontweight='bold', pad=5)
    axis.text(0.01, 0.985, statistics_text(moderation) + '\nn=37; exploratory', transform=axis.transAxes, ha='left', va='top', fontsize=5.7, color=MID, linespacing=1.25)
    axis.set_xlim(float(x.min()), float(x.max()))
    axis.set_ylim(float(np.nanmin(low)) - 0.12, float(np.nanmax(high)) + 0.16)
    style_axis(axis)
    fig.subplots_adjust(left=0.18, right=0.96, bottom=0.17, top=0.84)
    save_submission_figure(fig, 'panel_a_interaction_version3_conditional_effect')
    return curve

def main() -> None:
    (data, result, moderation, slopes) = load_analysis()
    adjusted_y = adjusted_outcome(data, result)
    direct_predictions = version_1_direct_slopes(data, adjusted_y, result, moderation, slopes)
    faceted_predictions = version_2_faceted_tertiles(data, adjusted_y, result, moderation)
    conditional_curve = version_3_conditional_effect(data, result, moderation, slopes)
    points = data[['subject_id', 'group', 'x_z', 'm_z', 'MMSE_improvement_z', 'lesion_g1g2_span', 'gradient_normalization']].copy()
    points['covariate_adjusted_MMSE_improvement_z'] = adjusted_y
    points.to_csv(SOURCE / 'panel_a_interaction_versions_subject_points.csv', index=False, encoding='utf-8-sig')
    direct_predictions.to_csv(SOURCE / 'panel_a_interaction_version1_model_lines.csv', index=False, encoding='utf-8-sig')
    faceted_predictions.to_csv(SOURCE / 'panel_a_interaction_version2_model_lines.csv', index=False, encoding='utf-8-sig')
    conditional_curve.to_csv(SOURCE / 'panel_a_interaction_version3_conditional_slope.csv', index=False, encoding='utf-8-sig')
    moderation.to_frame().T.to_csv(SOURCE / 'panel_a_interaction_statistics.csv', index=False, encoding='utf-8-sig')
if __name__ == '__main__':
    main()
