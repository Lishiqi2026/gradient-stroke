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
import warnings
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
ANALYSIS_ROOT = Path(_portable_path('project', 'lesion_analysis'))
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))
import run_gradient_reorganization_ml as reorganization
import run_lesion_adjusted_gradient_value as base
OUTPUT_ROOT = ANALYSIS_ROOT / 'responder_classification'
TABLES = OUTPUT_ROOT / 'tables'
FIGURES = OUTPUT_ROOT / 'figures'
SPLITS = OUTPUT_ROOT / 'splits'
MODES = ('baseline_prediction', 'reorganization_marker')
MODE_LABELS = {'baseline_prediction': 'Baseline prediction', 'reorganization_marker': 'Post-treatment reorganization marker'}
BLOCK_LABELS = {'clinical': 'Clinical', 'clinical_lesion': 'Clinical + lesion', 'clinical_gradient': 'Clinical + gradient', 'combined': 'Clinical + lesion + gradient'}
BLOCK_COLORS = {'clinical': '#777777', 'clinical_lesion': '#D07A3B', 'clinical_gradient': '#4C78A8', 'combined': '#248F78'}
BENCHMARK_COLUMNS = {'treatment_group_only': ['group_tms'], 'clinical_without_group': ['NIHSS_pre', 'age', 'sex_male', 'esrs']}
LOGISTIC_C = tuple((float(value) for value in np.logspace(-3, 3, 7)))
RESPONDER_THRESHOLD = 4.0

def ensure_dirs() -> None:
    for path in (TABLES, FIGURES, SPLITS):
        path.mkdir(parents=True, exist_ok=True)

def add_labels(prepared: base.PreparedData) -> base.PreparedData:
    table = prepared.table.copy()
    table['NIHSS_improvement'] = table['NIHSS_pre'] - table['NIHSS_post']
    table['NIHSS_responder'] = (table['NIHSS_improvement'] >= RESPONDER_THRESHOLD).astype(int)
    table['ADL_improvement'] = table['ADL_post'] - table['ADL_pre']
    table['ADL_responder_10'] = (table['ADL_improvement'] >= 10).astype(int)
    return base.PreparedData(table=table, cortical=prepared.cortical, subcortical=prepared.subcortical)

def load_modes() -> dict[str, base.PreparedData]:
    baseline = add_labels(base.load_inputs())
    changed = add_labels(reorganization.load_reorganization_inputs())
    if baseline.table['subject_id'].tolist() != changed.table['subject_id'].tolist():
        raise RuntimeError('Baseline and reorganization cohorts differ')
    if not np.array_equal(baseline.table['NIHSS_responder'], changed.table['NIHSS_responder']):
        raise RuntimeError('Responder labels differ across modes')
    counts = baseline.table['NIHSS_responder'].value_counts()
    if counts.min() < 5:
        raise RuntimeError(f'Responder classes cannot support five-fold CV: {counts.to_dict()}')
    return {'baseline_prediction': baseline, 'reorganization_marker': changed}

def make_outer_splits(labels: np.ndarray, repeats: int, seed_offset: int) -> list[dict]:
    rows = []
    for repeat in range(repeats):
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=base.SEED + seed_offset + repeat)
        for (fold, (train, test)) in enumerate(splitter.split(np.zeros(len(labels)), labels)):
            rows.append({'repeat': repeat, 'fold': fold, 'train': train, 'test': test})
    return rows

def make_inner_splits(labels: np.ndarray, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(labels)), labels))

def make_classifier(c_value: float) -> LogisticRegression:
    return LogisticRegression(C=c_value, penalty='l2', solver='liblinear', max_iter=10000, random_state=base.SEED)

def tune_outer_fold(prepared: base.PreparedData, block: str, split: dict) -> tuple[np.ndarray, dict]:
    labels = prepared.table['NIHSS_responder'].to_numpy(int)
    outer_train = split['train']
    outer_test = split['test']
    inner_splits = make_inner_splits(labels[outer_train], base.SEED + 30000 + split['repeat'] * 10 + split['fold'])
    candidates = []
    for (cortical_components, subcortical_components) in base.pca_configs(block):
        transformed_folds = []
        for (inner_train_local, inner_test_local) in inner_splits:
            inner_train = outer_train[inner_train_local]
            inner_test = outer_train[inner_test_local]
            (x_train, x_test) = base.transform_block(prepared, 'NIHSS', block, inner_train, inner_test, cortical_components, subcortical_components)
            transformed_folds.append((inner_train, inner_test, x_train, x_test))
        for c_value in LOGISTIC_C:
            fold_auc = []
            for (inner_train, inner_test, x_train, x_test) in transformed_folds:
                model = make_classifier(c_value)
                model.fit(x_train, labels[inner_train])
                probability = model.predict_proba(x_test)[:, 1]
                fold_auc.append(roc_auc_score(labels[inner_test], probability))
            candidates.append({'cortical_components': cortical_components, 'subcortical_components': subcortical_components, 'logistic_c': c_value, 'inner_roc_auc': float(np.mean(fold_auc))})
    best = max(candidates, key=lambda row: (row['inner_roc_auc'], -(row['cortical_components'] + row['subcortical_components']), -row['logistic_c']))
    (x_train, x_test) = base.transform_block(prepared, 'NIHSS', block, outer_train, outer_test, best['cortical_components'], best['subcortical_components'])
    model = make_classifier(best['logistic_c'])
    model.fit(x_train, labels[outer_train])
    return (model.predict_proba(x_test)[:, 1], best)

def tune_simple_outer_fold(prepared: base.PreparedData, columns: list[str], split: dict) -> tuple[np.ndarray, dict]:
    labels = prepared.table['NIHSS_responder'].to_numpy(int)
    features = prepared.table[columns].to_numpy(float)
    outer_train = split['train']
    outer_test = split['test']
    inner_splits = make_inner_splits(labels[outer_train], base.SEED + 60000 + split['repeat'] * 10 + split['fold'])
    transformed_folds = []
    for (inner_train_local, inner_test_local) in inner_splits:
        inner_train = outer_train[inner_train_local]
        inner_test = outer_train[inner_test_local]
        (x_train, x_test) = base.scaled_part(features[inner_train], features[inner_test])
        transformed_folds.append((inner_train, inner_test, x_train, x_test))
    candidates = []
    for c_value in LOGISTIC_C:
        fold_auc = []
        for (inner_train, inner_test, x_train, x_test) in transformed_folds:
            model = make_classifier(c_value)
            model.fit(x_train, labels[inner_train])
            fold_auc.append(roc_auc_score(labels[inner_test], model.predict_proba(x_test)[:, 1]))
        candidates.append({'logistic_c': c_value, 'inner_roc_auc': float(np.mean(fold_auc))})
    best = max(candidates, key=lambda row: (row['inner_roc_auc'], -row['logistic_c']))
    (x_train, x_test) = base.scaled_part(features[outer_train], features[outer_test])
    model = make_classifier(best['logistic_c'])
    model.fit(x_train, labels[outer_train])
    return (model.predict_proba(x_test)[:, 1], best)

def run_benchmark_job(prepared: base.PreparedData, block: str, columns: list[str], splits: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows = []
    tuning_rows = []
    table = prepared.table
    for split in splits:
        (probabilities, best) = tune_simple_outer_fold(prepared, columns, split)
        for (index, probability) in zip(split['test'], probabilities):
            prediction_rows.append({'mode': 'baseline_prediction', 'cohort': 'primary_full', 'block': block, 'repeat': split['repeat'], 'fold': split['fold'], 'subject_id': table.loc[index, 'subject_id'], 'group': table.loc[index, 'group'], 'NIHSS_pre': float(table.loc[index, 'NIHSS_pre']), 'NIHSS_post': float(table.loc[index, 'NIHSS_post']), 'NIHSS_improvement': float(table.loc[index, 'NIHSS_improvement']), 'observed_responder': int(table.loc[index, 'NIHSS_responder']), 'predicted_probability': float(probability)})
        tuning_rows.append({'mode': 'baseline_prediction', 'cohort': 'primary_full', 'block': block, 'repeat': split['repeat'], 'fold': split['fold'], 'columns': ';'.join(columns), **best})
    return (pd.DataFrame(prediction_rows), pd.DataFrame(tuning_rows))

def run_job(mode: str, cohort: str, prepared: base.PreparedData, block: str, splits: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows = []
    tuning_rows = []
    table = prepared.table
    for split in splits:
        (probabilities, best) = tune_outer_fold(prepared, block, split)
        for (index, probability) in zip(split['test'], probabilities):
            prediction_rows.append({'mode': mode, 'cohort': cohort, 'block': block, 'repeat': split['repeat'], 'fold': split['fold'], 'subject_id': table.loc[index, 'subject_id'], 'group': table.loc[index, 'group'], 'NIHSS_pre': float(table.loc[index, 'NIHSS_pre']), 'NIHSS_post': float(table.loc[index, 'NIHSS_post']), 'NIHSS_improvement': float(table.loc[index, 'NIHSS_improvement']), 'observed_responder': int(table.loc[index, 'NIHSS_responder']), 'predicted_probability': float(probability)})
        tuning_rows.append({'mode': mode, 'cohort': cohort, 'block': block, 'repeat': split['repeat'], 'fold': split['fold'], **best})
    return (pd.DataFrame(prediction_rows), pd.DataFrame(tuning_rows))

def classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = (probabilities >= 0.5).astype(int)
    (tn, fp, fn, tp) = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {'roc_auc': float(roc_auc_score(labels, probabilities)), 'pr_auc': float(average_precision_score(labels, probabilities)), 'balanced_accuracy': float(balanced_accuracy_score(labels, predictions)), 'accuracy': float(accuracy_score(labels, predictions)), 'sensitivity': float(tp / (tp + fn)) if tp + fn else np.nan, 'specificity': float(tn / (tn + fp)) if tn + fp else np.nan, 'brier_score': float(brier_score_loss(labels, probabilities))}

def repeat_performance(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (keys, subset) in predictions.groupby(['mode', 'cohort', 'block', 'repeat'], sort=False):
        (mode, cohort, block, repeat) = keys
        metrics = classification_metrics(subset['observed_responder'].to_numpy(int), subset['predicted_probability'].to_numpy(float))
        rows.append({'mode': mode, 'cohort': cohort, 'block': block, 'repeat': int(repeat), 'n': len(subset), **metrics})
    return pd.DataFrame(rows)

def summarize_performance(performance: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = ('roc_auc', 'pr_auc', 'balanced_accuracy', 'accuracy', 'sensitivity', 'specificity', 'brier_score')
    for (keys, subset) in performance.groupby(['mode', 'cohort', 'block'], sort=False):
        (mode, cohort, block) = keys
        for metric in metrics:
            values = subset[metric].dropna().to_numpy(float)
            rows.append({'mode': mode, 'cohort': cohort, 'block': block, 'metric': metric, 'n_repeats': len(values), 'mean': float(np.mean(values)), 'median': float(np.median(values)), 'sd': float(np.std(values, ddof=1)), 'repeat_interval_low': float(np.percentile(values, 2.5)), 'repeat_interval_high': float(np.percentile(values, 97.5))})
    return pd.DataFrame(rows)

def incremental_summary(performance: pd.DataFrame) -> pd.DataFrame:
    comparisons = (('clinical', 'clinical_lesion', 'lesion beyond clinical'), ('clinical', 'clinical_gradient', 'gradient beyond clinical'), ('clinical_lesion', 'combined', 'gradient beyond clinical + lesion'))
    rows = []
    for ((mode, cohort), frame) in performance.groupby(['mode', 'cohort'], sort=False):
        available = set(frame['block'])
        for (reference, comparison, label) in comparisons:
            if reference not in available or comparison not in available:
                continue
            for metric in ('roc_auc', 'balanced_accuracy'):
                reference_values = frame.loc[frame['block'] == reference, ['repeat', metric]].set_index('repeat')[metric]
                comparison_values = frame.loc[frame['block'] == comparison, ['repeat', metric]].set_index('repeat')[metric]
                delta = (comparison_values - reference_values).dropna().to_numpy(float)
                rows.append({'mode': mode, 'cohort': cohort, 'contrast': label, 'reference_block': reference, 'comparison_block': comparison, 'metric': metric, 'positive_means_comparison_improves': True, 'mean_increment': float(np.mean(delta)), 'median_increment': float(np.median(delta)), 'repeat_interval_low': float(np.percentile(delta, 2.5)), 'repeat_interval_high': float(np.percentile(delta, 97.5)), 'fraction_repeats_improved': float(np.mean(delta > 0)), 'n_repeats': len(delta)})
    return pd.DataFrame(rows)

def aggregate_oof(predictions: pd.DataFrame) -> pd.DataFrame:
    return predictions.groupby(['mode', 'cohort', 'block', 'subject_id', 'group'], as_index=False).agg(NIHSS_pre=('NIHSS_pre', 'first'), NIHSS_post=('NIHSS_post', 'first'), NIHSS_improvement=('NIHSS_improvement', 'first'), observed_responder=('observed_responder', 'first'), predicted_probability=('predicted_probability', 'mean'), probability_sd=('predicted_probability', 'std'))

def permute_gradient_within_group(prepared: base.PreparedData, seed: int) -> base.PreparedData:
    rng = np.random.default_rng(seed)
    assignment = np.arange(len(prepared.table))
    groups = prepared.table['group_tms'].to_numpy(int)
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        assignment[indices] = rng.permutation(indices)
    return base.PreparedData(table=prepared.table, cortical=prepared.cortical[assignment], subcortical=prepared.subcortical[assignment])

def permutation_one(mode: str, prepared: base.PreparedData, splits: list[dict], permutation: int) -> float:
    permuted = permute_gradient_within_group(prepared, base.SEED + 700000 + 10000 * MODES.index(mode) + permutation)
    (predictions, _) = run_job(mode, 'permuted', permuted, 'combined', splits)
    labels = predictions['observed_responder'].to_numpy(int)
    probabilities = predictions['predicted_probability'].to_numpy(float)
    return float(roc_auc_score(labels, probabilities))

def run_permutations(modes: dict[str, base.PreparedData], predictions: pd.DataFrame, split_sets: dict[str, list[dict]], permutations: int, jobs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if permutations <= 0:
        return (pd.DataFrame(), pd.DataFrame())
    summary_rows = []
    null_rows = []
    for (mode, prepared) in modes.items():
        fixed_splits = [split for split in split_sets[mode] if split['repeat'] == 0]
        actual = predictions.loc[(predictions['mode'] == mode) & (predictions['cohort'] == 'primary_full') & (predictions['repeat'] == 0)]
        reference = actual.loc[actual['block'] == 'clinical_lesion']
        combined = actual.loc[actual['block'] == 'combined']
        reference_auc = float(roc_auc_score(reference['observed_responder'], reference['predicted_probability']))
        combined_auc = float(roc_auc_score(combined['observed_responder'], combined['predicted_probability']))
        observed_increment = combined_auc - reference_auc
        null_auc = Parallel(n_jobs=jobs, verbose=5)((delayed(permutation_one)(mode, prepared, fixed_splits, permutation) for permutation in range(permutations)))
        null_increment = np.asarray(null_auc, dtype=float) - reference_auc
        p_value = (1.0 + float(np.count_nonzero(null_increment >= observed_increment))) / (permutations + 1.0)
        summary_rows.append({'mode': mode, 'test': 'conditional gradient-block permutation for ROC AUC increment', 'permutation_scheme': 'cortical/subcortical gradient rows jointly shuffled within observed treatment group', 'clinical_lesion_roc_auc': reference_auc, 'combined_roc_auc': combined_auc, 'observed_auc_increment': observed_increment, 'null_mean_auc_increment': float(np.mean(null_increment)), 'null_p95_auc_increment': float(np.percentile(null_increment, 95)), 'one_sided_p_value': p_value, 'permutations': permutations, 'outer_cv': 'first fixed responder-stratified five-fold repeat', 'inner_cv': 'four-fold tuning repeated under every permutation'})
        for (permutation, (auc, increment)) in enumerate(zip(null_auc, null_increment)):
            null_rows.append({'mode': mode, 'permutation': permutation, 'permuted_combined_roc_auc': auc, 'auc_increment_vs_clinical_lesion': increment})
    return (pd.DataFrame(summary_rows), pd.DataFrame(null_rows))

def save_splits(mode: str, prepared: base.PreparedData, splits: list[dict]) -> None:
    ids = prepared.table['subject_id'].to_numpy()
    labels = prepared.table['NIHSS_responder'].to_numpy(int)
    records = [{'mode': mode, 'repeat': int(split['repeat']), 'fold': int(split['fold']), 'train_subject_ids': ids[split['train']].tolist(), 'test_subject_ids': ids[split['test']].tolist(), 'train_responder_n': int(labels[split['train']].sum()), 'test_responder_n': int(labels[split['test']].sum())} for split in splits]
    (SPLITS / f'nested_cv_splits_{mode}.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')

def plot_results(performance: pd.DataFrame, increments: pd.DataFrame) -> None:
    (fig, axes) = plt.subplots(2, 2, figsize=(7.6, 5.4), facecolor='white')
    rng = np.random.default_rng(base.SEED)
    for (row_index, mode) in enumerate(MODES):
        frame = performance.loc[(performance['mode'] == mode) & (performance['cohort'] == 'primary_full')]
        for (position, block) in enumerate(base.BLOCKS):
            values = frame.loc[frame['block'] == block, 'roc_auc'].to_numpy(float)
            axes[row_index, 0].scatter(position + rng.uniform(-0.07, 0.07, len(values)), values, s=9, color=BLOCK_COLORS[block], alpha=0.45, edgecolor='none')
            mean = float(np.mean(values))
            (low, high) = np.percentile(values, [2.5, 97.5])
            axes[row_index, 0].errorbar(position, mean, yerr=[[mean - low], [high - mean]], fmt='o', color=BLOCK_COLORS[block], markersize=4.5, capsize=2)
        axes[row_index, 0].axhline(0.5, color='#888888', linestyle='--', linewidth=0.7)
        axes[row_index, 0].set_xticks(range(4), ['Clinical', '+ lesion', '+ gradient', 'Combined'], rotation=12, ha='right')
        axes[row_index, 0].set_ylabel('ROC AUC')
        axes[row_index, 0].set_title(MODE_LABELS[mode])
        increment_frame = increments.loc[(increments['mode'] == mode) & (increments['cohort'] == 'primary_full') & (increments['metric'] == 'roc_auc')]
        contrasts = ('lesion beyond clinical', 'gradient beyond clinical', 'gradient beyond clinical + lesion')
        for (position, contrast) in enumerate(contrasts):
            row = increment_frame.loc[increment_frame['contrast'] == contrast].iloc[0]
            axes[row_index, 1].errorbar(position, row['mean_increment'], yerr=[[row['mean_increment'] - row['repeat_interval_low']], [row['repeat_interval_high'] - row['mean_increment']]], fmt='o', color=('#D07A3B', '#4C78A8', '#248F78')[position], markersize=5, capsize=2)
        axes[row_index, 1].axhline(0, color='#888888', linestyle='--', linewidth=0.7)
        axes[row_index, 1].set_xticks(range(3), ['Lesion\n| clinical', 'Gradient\n| clinical', 'Gradient\n| clinical + lesion'])
        axes[row_index, 1].set_ylabel('ROC AUC increment\n(positive = better)')
        axes[row_index, 1].set_title('Incremental classification value')
    for (label, axis) in zip(('A', 'B', 'C', 'D'), axes.ravel()):
        axis.text(-0.14, 1.07, label, transform=axis.transAxes, fontsize=10, fontweight='bold', va='top')
        axis.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Classification of clinically significant NIHSS improvement', fontsize=10, y=0.995)
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.105, top=0.9, hspace=0.72, wspace=0.48)
    for (suffix, kwargs) in (('png', {'dpi': 300}), ('pdf', {})):
        fig.savefig(FIGURES / f'NIHSS_responder_classification.{suffix}', bbox_inches='tight', facecolor='white', **kwargs)
    plt.close(fig)

def write_summary(modes: dict[str, base.PreparedData], summary: pd.DataFrame, increments: pd.DataFrame, permutation: pd.DataFrame) -> None:
    table = modes['baseline_prediction'].table
    responder_n = int(table['NIHSS_responder'].sum())
    nonresponder_n = int(len(table) - responder_n)
    lines = ['# NIHSS 有效改善分类 ML', '', '## 分类定义', '', f'- Responder：NIHSS 治疗后相对治疗前下降至少 {RESPONDER_THRESHOLD:.0f} 分。', f'- 38 例中 responder {responder_n} 例，non-responder {nonresponder_n} 例。', f"- Control responder {int(table.loc[table['group'] == 'Control', 'NIHSS_responder'].sum())}/16；TMS responder {int(table.loc[table['group'] == 'TMS', 'NIHSS_responder'].sum())}/22。", '- ADL 改善至少 10 分时为 37/38 responder，任何改善时为 38/38，因此 ADL 不具备可训练的二分类结局。', '', '## 两种分类用途', '', '- Baseline prediction：只使用治疗前临床、病灶和基线梯度，可解释为内部验证的治疗前预测。', '- Reorganization marker：使用治疗后的梯度重组，只能解释为同期恢复标志物，不能用于治疗前决策。', '', '## 50次重复嵌套交叉验证', '', '| 用途 | 模型 | ROC AUC | 平衡准确率 | 敏感度 | 特异度 |', '|---|---|---:|---:|---:|---:|']
    primary = summary.loc[summary['cohort'] == 'primary_full']
    for mode in MODES:
        for block in base.BLOCKS:
            lookup = primary.loc[(primary['mode'] == mode) & (primary['block'] == block)].set_index('metric')['mean']
            lines.append(f"| {MODE_LABELS[mode]} | {BLOCK_LABELS[block]} | {lookup['roc_auc']:.3f} | {lookup['balanced_accuracy']:.3f} | {lookup['sensitivity']:.3f} | {lookup['specificity']:.3f} |")
    lines.extend(['', '## 梯度在临床和病灶之上的增量', ''])
    for mode in MODES:
        row = increments.loc[(increments['mode'] == mode) & (increments['cohort'] == 'primary_full') & (increments['contrast'] == 'gradient beyond clinical + lesion') & (increments['metric'] == 'roc_auc')].iloc[0]
        perm = permutation.loc[permutation['mode'] == mode].iloc[0] if not permutation.empty else None
        text = f"- {MODE_LABELS[mode]}：加入梯度后平均 ROC AUC 改变量 {row['mean_increment']:+.3f}，重复区间 {row['repeat_interval_low']:+.3f} 到 {row['repeat_interval_high']:+.3f}，{row['fraction_repeats_improved'] * 100:.1f}% 重复改善。"
        if perm is not None:
            text += f" 条件置换 P={perm['one_sided_p_value']:.3f}。"
        lines.append(text)
    lines.extend(['', '## 治疗组依赖审计', ''])
    benchmark_labels = {'treatment_group_only': '仅治疗组', 'clinical_without_group': '临床但去掉治疗组'}
    for (block, label) in benchmark_labels.items():
        lookup = primary.loc[(primary['mode'] == 'baseline_prediction') & (primary['block'] == block)].set_index('metric')['mean']
        lines.append(f"- {label}：ROC AUC={lookup['roc_auc']:.3f}，平衡准确率={lookup['balanced_accuracy']:.3f}。")
    lines.append('- 若仅治疗组的性能接近完整临床模型，说明分类器主要利用组间 responder 比例差异，不能解释为发现了可泛化的个体影像标志物。')
    lines.extend(['', '## 解释限制', '', '- 样本仅38例且治疗组与responder高度相关，分类器可能主要识别治疗组，而不是稳定的个体生物标志物。', '- ROC AUC、敏感度和特异度均来自内部重复交叉验证，不是独立验证性能。', '- 不能根据本模型直接进行临床治疗选择。'])
    (OUTPUT_ROOT / 'RESULTS_ZH.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Classify clinically significant NIHSS improvement.')
    parser.add_argument('--repeats', type=int, default=50)
    parser.add_argument('--permutations', type=int, default=100)
    parser.add_argument('--jobs', type=int, default=-1)
    parser.add_argument('--quick', action='store_true')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    if args.quick:
        args.repeats = 1
        args.permutations = min(args.permutations, 2)
    if args.repeats < 1 or args.permutations < 0:
        raise ValueError('repeats must be positive and permutations non-negative')
    ensure_dirs()
    modes = load_modes()
    sensitivity_modes = {mode: base.subset_prepared(prepared, ~prepared.table['subject_id'].isin(base.MANUAL_REVIEW_FLAGS).to_numpy()) for (mode, prepared) in modes.items()}
    split_sets = {mode: make_outer_splits(prepared.table['NIHSS_responder'].to_numpy(int), args.repeats, seed_offset=40000 + 1000 * index) for (index, (mode, prepared)) in enumerate(modes.items())}
    sensitivity_splits = {mode: make_outer_splits(prepared.table['NIHSS_responder'].to_numpy(int), args.repeats, seed_offset=50000 + 1000 * index) for (index, (mode, prepared)) in enumerate(sensitivity_modes.items())}
    for (mode, prepared) in modes.items():
        save_splits(mode, prepared, split_sets[mode])
    modes['baseline_prediction'].table[['subject_id', 'group', 'manual_review_flag', 'NIHSS_pre', 'NIHSS_post', 'NIHSS_improvement', 'NIHSS_responder', 'ADL_pre', 'ADL_post', 'ADL_improvement', 'ADL_responder_10']].to_csv(TABLES / 'responder_labels.csv', index=False, encoding='utf-8-sig')
    jobs_to_run = []
    for (mode, prepared) in modes.items():
        for block in base.BLOCKS:
            jobs_to_run.append((mode, 'primary_full', prepared, block, split_sets[mode]))
        for block in ('clinical_lesion', 'combined'):
            jobs_to_run.append((mode, 'exclude_manual_review_flags', sensitivity_modes[mode], block, sensitivity_splits[mode]))
    results = Parallel(n_jobs=args.jobs, verbose=10)((delayed(run_job)(*job) for job in jobs_to_run))
    benchmark_results = Parallel(n_jobs=args.jobs, verbose=5)((delayed(run_benchmark_job)(modes['baseline_prediction'], block, columns, split_sets['baseline_prediction']) for (block, columns) in BENCHMARK_COLUMNS.items()))
    all_results = results + benchmark_results
    predictions = pd.concat([result[0] for result in all_results], ignore_index=True)
    tuning = pd.concat([result[1] for result in all_results], ignore_index=True)
    performance = repeat_performance(predictions)
    summary = summarize_performance(performance)
    increments = incremental_summary(performance)
    (permutation, permutation_null) = run_permutations(modes, predictions, split_sets, args.permutations, args.jobs)
    predictions.to_csv(TABLES / 'nested_cv_predictions.csv', index=False, encoding='utf-8-sig')
    tuning.to_csv(TABLES / 'outer_fold_tuning.csv', index=False, encoding='utf-8-sig')
    performance.to_csv(TABLES / 'repeat_level_performance.csv', index=False, encoding='utf-8-sig')
    summary.to_csv(TABLES / 'performance_summary.csv', index=False, encoding='utf-8-sig')
    increments.to_csv(TABLES / 'incremental_performance_summary.csv', index=False, encoding='utf-8-sig')
    aggregate_oof(predictions).to_csv(TABLES / 'participant_mean_oof_probabilities.csv', index=False, encoding='utf-8-sig')
    if not permutation.empty:
        permutation.to_csv(TABLES / 'gradient_block_permutation.csv', index=False, encoding='utf-8-sig')
        permutation_null.to_csv(TABLES / 'gradient_block_permutation_null.csv', index=False, encoding='utf-8-sig')
    plot_results(performance, increments)
    write_summary(modes, summary, increments, permutation)
    labels = modes['baseline_prediction'].table
    manifest = {'analysis_date': '2026-08-03', 'primary_cohort_n': int(len(labels)), 'responder_definition': 'NIHSS_pre - NIHSS_post >= 4 points', 'responder_n': int(labels['NIHSS_responder'].sum()), 'nonresponder_n': int((1 - labels['NIHSS_responder']).sum()), 'group_by_responder': pd.crosstab(labels['group'], labels['NIHSS_responder']).to_dict(), 'ADL_10_point_responder_n': int(labels['ADL_responder_10'].sum()), 'ADL_classification_status': 'not modeled because 37/38 meet the 10-point threshold', 'modes': {'baseline_prediction': 'baseline clinical + baseline lesion + baseline gradient', 'reorganization_marker': 'baseline clinical + baseline lesion + post-treatment aligned gradient reorganization'}, 'model': 'L2-regularized logistic regression', 'outer_cv': f'responder-stratified five-fold repeated {args.repeats} times', 'inner_cv': 'responder-stratified four-fold tuning of logistic C and PCA dimensions by ROC AUC', 'probability_threshold': 0.5, 'gradient_pca_candidates': [list(config) for config in base.PCA_CONFIGS], 'permutations_per_mode': args.permutations, 'primary_incremental_contrast': 'clinical+lesion versus clinical+lesion+gradient', 'group_dependence_benchmarks': BENCHMARK_COLUMNS, 'external_validation': False, 'limitations': ['Small single cohort.', 'Treatment group is strongly associated with responder status.', 'Reorganization-marker mode uses post-treatment imaging and is not prospective prediction.'], 'threshold_sources': [{'scale': 'NIHSS', 'threshold': 'at least 4-point improvement', 'doi': '10.1177/1941874415619964'}, {'scale': 'Barthel Index', 'threshold': '10-point clinically meaningful difference', 'url': 'https://www.bmj.com/content/366/bmj.l5101'}], 'random_seed': base.SEED}
    (OUTPUT_ROOT / 'analysis_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Responder/non-responder: {manifest['responder_n']}/{manifest['nonresponder_n']}")
    print(f'Prediction rows: {len(predictions)}')
    print(f'Output: {OUTPUT_ROOT}')
if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=FutureWarning)
        main()
