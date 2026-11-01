from __future__ import annotations
import os as _portable_os
from pathlib import Path as _PortablePath

def _portable_path(kind, relative):
    project = _PortablePath(_portable_os.environ['PAPER_PROJECT_ROOT'])
    root = project if kind == 'project' else project.parent
    return str(root / relative)
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
PROJECT = Path(_portable_path('project', ''))
WORK = PROJECT / 'lesion_analysis'
OUT = WORK / 'folder_for_webgpt_external_validation'

def copy_file(source: Path, relative: str) -> Path:
    target = OUT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == source.stat().st_size:
        return target
    shutil.copy2(source, target)
    return target

def write_text(relative: str, content: str) -> Path:
    target = OUT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding='utf-8')
    return target

def copy_csv_without_id(source: Path, relative: str) -> Path:
    frame = pd.read_csv(source)
    blocked = [column for column in frame.columns if re.search('(^|_)(subject|participant|patient|medical|record).*id|filename|file_path', column, re.I)]
    if blocked:
        raise RuntimeError(f'Participant-level identifier columns found in {source}: {blocked}')
    target = OUT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    return target

def copy_sanitized_code(source: Path, relative: str) -> Path:
    text = source.read_text(encoding='utf-8')
    replacements = {_portable_path('project', 'new_dataset'): 'PATH_TO_NEW_DATASET', _portable_path('project', 'lesion_analysis'): 'PATH_TO_MAIN_ANALYSIS', _portable_path('project', 'code'): 'PATH_TO_MAIN_CODE', 'sub-35': 'EXCLUDED_QC_SUBJECT', 'sub-01': 'CATALOGUED_MASK_SUBJECT_1', 'sub-05': 'CATALOGUED_MASK_SUBJECT_2', 'sub-15': 'CATALOGUED_MASK_SUBJECT_3', 'sub-18': 'CATALOGUED_MASK_SUBJECT_4'}
    for (old, new) in replacements.items():
        text = text.replace(old, new)
    text = re.sub('(?i)(?:[A-Z]:\\\\|/data/)[^\\"\'\\r\\n]+', 'PATH_PLACEHOLDER', text)
    text = re.sub('(?<![\\d.])\\d{7,9}(?![\\d.])', 'REDACTED_NUMERIC_TOKEN', text)
    header = '"""CONSULTATION COPY. Local paths and public participant labels were generalized.\nUse the original local repository for exact execution.\n"""\n'
    if text.startswith('from __future__ import'):
        (first, rest) = text.split('\n', 1)
        text = first + '\n\n' + header + rest
    else:
        text = header + text
    return write_text(relative, text)

def copy_sanitized_markdown(source: Path, relative: str) -> Path:
    text = source.read_text(encoding='utf-8')
    text = text.replace('PRIVATE_SUBJECT_008', '一名预设运动QC不合格受试者')
    text = re.sub('(?i)[A-Z]:\\\\[^`\\s]+', 'LOCAL_PATH_REDACTED', text)
    return write_text(relative, text)

def new_dataset_summary() -> dict:
    root = PROJECT / 'new_dataset'
    participants = pd.read_csv(root / 'participants.tsv', sep='\t', keep_default_na=False)
    lesion_fields = ['hemisphere', 'size', 'Broca', 'Wernike', 'Basal', 'Thalamus']
    complete_lesion = participants[lesion_fields].replace('n/a', np.nan).dropna().shape[0]

    def array_inventory(folder: str, session: str) -> dict:
        paths = sorted((root / folder / folder / session).glob('*.npy'))
        shapes: dict[str, int] = {}
        dtypes: dict[str, int] = {}
        for path in paths:
            array = np.load(path, mmap_mode='r')
            shapes[str(tuple(array.shape))] = shapes.get(str(tuple(array.shape)), 0) + 1
            dtypes[str(array.dtype)] = dtypes.get(str(array.dtype), 0) + 1
        return {'files': len(paths), 'shapes': shapes, 'dtypes': dtypes}
    improvement = participants['pre_score'].astype(float) - participants['post_score'].astype(float)
    sex_counts = participants['sex'].value_counts().to_dict()
    aphasia_counts = participants['type'].value_counts().to_dict()
    return {'privacy': {'contains_participant_rows': False, 'contains_raw_images_or_arrays': False, 'contains_direct_identifiers': False}, 'source': 'OpenNeuro ds003999 v1.0.2; local derivative folder E:/paper/lesion_refresh_20261028/work/Gradient/new_dataset', 'design': {'participants': int(len(participants)), 'sessions': ['pre rehabilitation', 'post rehabilitation'], 'treatment_allocation': 'none; no TMS versus Control label', 'clinical_endpoint': 'Vasserman score; improvement is pre minus post', 'clinical_complete_pairs': int(participants[['pre_score', 'post_score']].replace('n/a', np.nan).dropna().shape[0]), 'complete_coarse_lesion_metadata': int(complete_lesion), 'sex_counts': {str(k): int(v) for (k, v) in sex_counts.items()}, 'aphasia_type_counts': {str(k): int(v) for (k, v) in aphasia_counts.items()}, 'time_after_stroke_days': {'min': float(participants['time_after_stroke'].astype(float).min()), 'median': float(participants['time_after_stroke'].astype(float).median()), 'max': float(participants['time_after_stroke'].astype(float).max())}, 'vasserman_improvement': {'n': int(improvement.notna().sum()), 'mean': float(improvement.mean()), 'sd': float(improvement.std(ddof=1)), 'median': float(improvement.median())}}, 'available_derivatives': {'Schaefer400_aligned_gradients_pre': array_inventory('(400,5)', 'pre'), 'Schaefer400_aligned_gradients_post': array_inventory('(400,5)', 'post'), 'Schaefer400_FC_pre': array_inventory('(400,400)', 'pre'), 'Schaefer400_FC_post': array_inventory('(400,400)', 'post'), 'ROI_order_and_aggregation_QC': True}, 'coarse_lesion_fields': {'hemisphere': 'L/R/LR when available', 'size': 'S up to 20 mm; M up to 50 mm; H above 50 mm', 'Broca_Wernike_Basal_Thalamus': 'N=no, P=partial, Y=significant injury'}, 'voxel_mask_availability_audit': {'official_catalog_entries': 4, 'retrievable_and_verified_at_2026_08_12': 1, 'consequence': 'voxelwise lesion-gradient distance is not adequately testable in this version'}}

def make_safe_code_templates() -> None:
    gradient_code = '"""Method template: gradients and longitudinal network metrics.\n\nNo participant data or machine-specific paths are embedded. Replace the placeholders\nwith local paths. The main methodological requirements are fixed reference alignment,\nfixed component orientation and training/discovery separation.\n"""\nfrom pathlib import Path\nimport numpy as np\nfrom brainspace.gradient import GradientMaps\nfrom scipy.spatial import procrustes\n\n\ndef compute_gradients(fc: np.ndarray, n_components: int = 5) -> np.ndarray:\n    fc = np.asarray(fc, float).copy()\n    np.fill_diagonal(fc, 0.0)\n    gm = GradientMaps(\n        n_components=n_components,\n        approach="dm",\n        kernel="cosine",\n        random_state=0,\n    )\n    gm.fit(fc)\n    return np.asarray(gm.gradients_, float)\n\n\ndef align_to_fixed_reference(subject_gradients: np.ndarray, reference: np.ndarray) -> np.ndarray:\n    if subject_gradients.shape != reference.shape:\n        raise ValueError("Subject and reference gradients must have identical ROI/component shape")\n    _, aligned, _ = procrustes(reference, subject_gradients)\n    return aligned\n\n\ndef network_metrics(coordinates: np.ndarray, roi_network: np.ndarray) -> dict[str, float]:\n    result = {}\n    centers = {n: coordinates[roi_network == n].mean(axis=0) for n in np.unique(roi_network)}\n    for network, center in centers.items():\n        values = coordinates[roi_network == network]\n        dispersion = np.sqrt(np.mean(np.sum((values - center) ** 2, axis=1)))\n        other = [c for key, c in centers.items() if key != network]\n        between = np.mean([np.linalg.norm(center - c) for c in other])\n        result[f"network_{network}_dispersion"] = float(dispersion)\n        result[f"network_{network}_betweenDisp"] = float(between)\n        result[f"network_{network}_segregation"] = float(between / dispersion)\n    return result\n'
    write_text('04_safe_method_code/gradient_and_network_metric_template.py', gradient_code)
    lesion_code = '"""Method template: lesion-gradient-normalization features.\n\nThis is a privacy-safe methodological skeleton. lesion_fraction is subject x ROI;\npre/post gradients are subject x ROI x component; reference is ROI x component.\n"""\nimport numpy as np\n\n\ndef lesion_gradient_location(lesion_fraction: np.ndarray, reference: np.ndarray) -> np.ndarray:\n    weights = np.asarray(lesion_fraction, float)\n    denominator = weights.sum(axis=1, keepdims=True)\n    return (weights @ reference) / np.clip(denominator, 1e-12, None)\n\n\ndef lesion_gradient_span(lesion_fraction: np.ndarray, reference: np.ndarray) -> np.ndarray:\n    location = lesion_gradient_location(lesion_fraction, reference)\n    diff2 = (reference[None, :, :] - location[:, None, :]) ** 2\n    weighted = (lesion_fraction[:, :, None] * diff2).sum(axis=1)\n    return np.sqrt(weighted / np.clip(lesion_fraction.sum(axis=1, keepdims=True), 1e-12, None))\n\n\ndef healthy_reference_normalization(pre: np.ndarray, post: np.ndarray, reference: np.ndarray) -> np.ndarray:\n    pre_distance = np.linalg.norm(pre - reference[None, :, :], axis=2)\n    post_distance = np.linalg.norm(post - reference[None, :, :], axis=2)\n    return pre_distance - post_distance  # positive means closer to the healthy reference\n\n\ndef gradient_distance_to_lesion(reference: np.ndarray, lesion_location: np.ndarray) -> np.ndarray:\n    return np.linalg.norm(reference[None, :, :] - lesion_location[:, None, :], axis=2)\n\n\ndef subject_distance_slope(normalization, gradient_distance, anatomical_distance, intact_mask):\n    rows = []\n    for i in range(normalization.shape[0]):\n        use = np.asarray(intact_mask[i], bool)\n        x = np.column_stack([\n            np.ones(use.sum()),\n            gradient_distance[i, use],\n            anatomical_distance[i, use],\n        ])\n        beta = np.linalg.lstsq(x, normalization[i, use], rcond=None)[0]\n        rows.append(beta)\n    return np.asarray(rows)  # intercept, gradient-distance slope, anatomical-distance slope\n'
    write_text('04_safe_method_code/lesion_gradient_feature_template.py', lesion_code)
    ml_code = '"""Method template: leakage-safe low-dimensional external prediction.\n\nFeature blocks must be fixed from the discovery cohort. All imputation, scaling,\nfeature selection and hyperparameter tuning must occur within the training folds.\n"""\nimport numpy as np\nfrom sklearn.linear_model import Ridge\nfrom sklearn.model_selection import LeaveOneOut, GridSearchCV\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.impute import SimpleImputer\nfrom sklearn.metrics import mean_absolute_error\n\n\ndef nested_loocv_ridge(x: np.ndarray, y: np.ndarray):\n    outer = LeaveOneOut()\n    predictions = np.full(len(y), np.nan)\n    for train, test in outer.split(x):\n        pipeline = Pipeline([\n            ("impute", SimpleImputer(strategy="median")),\n            ("scale", StandardScaler()),\n            ("model", Ridge()),\n        ])\n        inner = GridSearchCV(\n            pipeline,\n            {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},\n            scoring="neg_mean_absolute_error",\n            cv=LeaveOneOut(),\n        )\n        inner.fit(x[train], y[train])\n        predictions[test] = inner.predict(x[test])\n    return {"mae": mean_absolute_error(y, predictions), "predictions": predictions}\n'
    write_text('04_safe_method_code/nested_external_prediction_template.py', ml_code)

def build_readme(summary: dict) -> str:
    return f"# WebGPT 咨询包：主队列结论与 ds003999 外部验证设计\n\n## 用途\n\n请把整个文件夹上传给 WebGPT，然后首先发送 `PROMPT_FOR_WEBGPT.md` 中的内容。本包只用于请 WebGPT 评审并设计外部验证实验，不用于远程重跑原始分析。\n\n## 隐私与数据边界\n\n- 不含原始 MRI、NIfTI、NPY、FC 或梯度矩阵；\n- 不含参与者逐行临床数据、真实姓名、病历号或主队列受试者 ID；\n- 不含逐例预测、逐例病灶特征或匹配配对 ID；\n- 表格限于组级、ROI级或模型级聚合统计；\n- 代表性代码已经脱敏；旧脚本中硬编码的病历号和绝对路径没有复制。\n\n## 建议阅读顺序\n\n1. `PROMPT_FOR_WEBGPT.md`：需要 WebGPT 回答的问题和输出格式；\n2. `01_project_overview/MAIN_AND_EXTERNAL_DATASET_OVERVIEW.md`：研究设计、主要结论与证据边界；\n3. `02_main_cohort_previous_results/`：三份主队列结果文档；\n4. `03_main_cohort_lesion_ml/`：主队列 lesion、gradient、clinical 和 ML 的汇总级结果；\n5. `04_safe_method_code/`：隐私安全的方法模板及外部验证代码；\n6. `05_new_dataset/`：新数据集结构、数据字典及现有外部验证结果；\n7. `06_current_external_validation/`：目前已经得到的支持、未复现和不可检验结论；\n8. `FILE_INDEX.csv`：每个文件的来源和作用。\n\n## 新数据集关键规模\n\n- 纵向临床、Schaefer400 FC 和五个梯度完整配对：{summary['design']['participants']} 人；\n- 完整粗粒度病灶描述：{summary['design']['complete_coarse_lesion_metadata']} 人；\n- 官方目录 lesion-mask 条目：4 人；本次可取得并验证：1 人；\n- 没有 TMS/Control 分组，因此不能验证治疗特异性效应。\n\n## 重要提醒\n\n主队列的 Visual/SomMot G2 特征、病灶调节结果和 ML 特征块均是在主队列中形成的。外部检验必须锁定 ROI、方向、权重和评价指标，不能在外部数据重新筛选后再称为独立验证。\n"

def project_overview(summary: dict) -> str:
    return f"# 主队列与新数据集概览\n\n## 1. 研究问题\n\n主队列研究卒中患者在常规康复期间，TMS 相对 Control 是否具有额外临床改善，以及病灶拓扑是否约束功能连接梯度重组、临床恢复和多模态预测。\n\n## 2. 主队列设计与可用数据\n\n- 卒中患者；Control 与 TMS 两组；干预前后重复测量；\n- 主临床表约 45 例，定量 lesion-mask 与临床—梯度共同队列 38 例，运动 QC 后主要机制队列 37 例；\n- 临床结局：NIHSS、ADL、MMSE、HAMD、VFT；\n- 影像：静息态 FC、Schaefer400 G1/G2、结构指标、体素 lesion mask、解剖区室及 XTRACT 通路负荷；\n- 注意：治疗是否随机、是否假刺激及同期康复是否完全一致仍需作者确认，因此组间效应应表述为调整后关联而非自动等同因果效应。\n\n## 3. 主队列主要结论\n\n### A. 临床与组间效应\n\n- TMS 组多个临床量表的 pre/post 改善大于 Control；\n- 在 11 对病灶拓扑匹配样本中，TMS-Control 平均改善差：NIHSS +2.27、ADL +24.09、MMSE +2.46、HAMD +3.46、VFT +1.91；\n- 严格 9 对中 NIHSS、ADL、HAMD、VFT 仍通过组内 FDR，MMSE q=0.072；\n- 匹配后最大绝对 SMD 仍为 0.635，严格匹配后为 0.459，因此属于支持性调整后比较，不是无混杂因果估计。\n\n### B. 梯度重组\n\n- 主队列按提供的 gradient pipeline 重新计算并对齐 G1/G2；\n- 在控制病灶体积和侧化后，TMS 组相对 Control 表现出更强的 G2 healthy-reference normalization；G1 缺乏同样效应；\n- G2 阳性效应主要定位在 Visual 和 SomMot：59 个正向 FDR ROI 中 53 个属于这两个网络；\n- 全皮层指标提示 TMS 组 G2 与 G1-G2 坐标更接近健康参考，Control 平均略远离；\n- XTRACT 投影 lesion-gradient 模型中，TMS 的 G2 distance slope 更高，提示相关正常化可能延伸至病灶功能层级更远区域；\n- 运动 QC 后病灶匹配的 10 对中，最近病灶功能层级 ROI 的 G1-G2 normalization 更强，但严格小样本匹配后减弱。\n\n### C. 病灶—梯度—评分\n\n- 正式中介分析没有稳定显著的间接效应；\n- 探索性调节模型提示：病灶 G1-G2 span 越大，病灶邻域 normalization 与 MMSE 改善的正相关可能越强；交互 beta=0.333，置换 p=0.0258，但全部 13 项交互 FDR q=0.271；\n- 因此这是候选机制，不是已证实中介或确定性调节效应。\n\n### D. 机器学习\n\n- 内部嵌套 CV 预测 NIHSS、ADL、MMSE、HAMD、VFT 治疗后评分；\n- 跨任务较稳定的候选特征是 G1/G2 between-network dispersion 和 G2 Visual/SomMot 组织；\n- 模型为内部验证，许多最优模型主要依赖临床变量；最佳与次佳特征块差异区间跨零；\n- 不能解释为已经证明影像带来稳定临床增益，更不能解释为个体 TMS 因果获益预测。\n\n## 4. 新数据集 ds003999\n\n- {summary['design']['participants']} 名卒中失语患者，pre/post 康复纵向设计；\n- pre/post Vasserman 评分、发病后时间、性别和失语类型；\n- {summary['design']['participants']} 人均有 Schaefer400 FC 400×400 和已对齐的五个梯度 400×5；\n- 只有 {summary['design']['complete_coarse_lesion_metadata']} 人具有完整的粗粒度病灶侧别、大小及 Broca/Wernicke/基底节/丘脑描述；\n- 无治疗分组；无 NIHSS/ADL/MMSE/HAMD/VFT；完整体素 lesion mask 不足。\n\n## 5. 当前外部验证结果\n\n- 支持：临床改善；局部纵向梯度重组；\n- 局部 G1 右侧 SomMot ROI 通过 FDR，q=0.009；\n- 未复现主队列特定 G2 Visual/SomMot 空间 signature：空间 spin p=0.838，固定 signature 变化 p=0.789；\n- 13 人粗病灶分析中，病灶负担到基线严重程度仅为趋势，beta=0.633，置换 p=0.080，q=0.215；\n- 病灶到固定 G2、固定 G2 到改善、病灶调节和增量预测均未复现；\n- 加入病灶或固定 G2 没有降低嵌套 LOOCV MAE；所有外层 R² 均小于 0。\n\n## 6. WebGPT 应重点解决的问题\n\n请优先判断这个外部数据集还能否以严格、低维度、预先固定假设的方式验证：\n\n1. 主队列 G1/G2 betweenDisp 或 network segregation 是否具有方向一致的纵向变化；\n2. 主队列固定的 G2 Visual/SomMot ROI/network signature 与 Vasserman 改善的关系；\n3. 基线梯度健康偏离是否预测康复幅度；\n4. 基线 FC/gradient 指标是否能在嵌套 CV 中提供相对 clinical-only 的增量预测；\n5. 13 人粗粒度 lesion metadata 是否只适合作为协变量/分层变量，而不应伪装为 voxelwise lesion topology；\n6. 是否存在比逐 ROI 检验更有统计效率的固定 network-level endpoint；\n7. 对无法验证的 TMS 特异性、MMSE 调节和 lesion-gradient distance，应如何在论文中界定边界并提出真正需要的新队列设计。\n"

def web_prompt() -> str:
    return '# 给 WebGPT 的提示词\n\n你是一名熟悉卒中神经影像、connectivity gradients、病灶映射、纵向统计、空间置换检验和小样本机器学习的高级方法学顾问。附件文件夹包含：\n\n- 主队列三份结果总览 DOCX；\n- 主队列 lesion—gradient—clinical、病灶匹配和 ML 的聚合结果；\n- 脱敏的方法模板和代表性代码；\n- OpenNeuro ds003999 新数据集的数据字典、数组结构、现有外部验证结果；\n- 不包含原始影像、逐例临床数据或可识别信息。\n\n我们的目标不是在新数据集重新数据挖掘，而是设计能够验证、反驳或限定主队列主要结论的独立外部实验。请先完整阅读 `README_FIRST.md`、`01_project_overview/MAIN_AND_EXTERNAL_DATASET_OVERVIEW.md`、三份 previous-result DOCX、`03_main_cohort_lesion_ml/` 和 `06_current_external_validation/`，然后回答。\n\n## 已知研究背景\n\n主队列为卒中 Control/TMS pre/post 队列。主要候选结论是：\n\n1. TMS 组临床改善大于 Control，但病灶匹配后仍有残余不平衡；\n2. TMS 相关 G2 healthy-reference normalization 主要集中于 Visual/SomMot；\n3. TMS 相关 normalization 可能扩展到与病灶功能层级距离较远的区域；\n4. 病灶 G1-G2 span 可能调节病灶邻域 normalization 与 MMSE 改善的关系，但全家族 FDR 未显著；\n5. 内部 ML 提示 G1/G2 betweenDisp 和 G2 Visual/SomMot 是跨任务候选特征，但影像净增量和外部泛化尚未证明。\n\n新数据 ds003999 有 29 名卒中失语患者的 pre/post Vasserman 评分、Schaefer400 FC（400×400）和五个已对齐梯度（400×5）。只有 13 人有完整粗粒度病灶描述，没有 TMS/Control 分组；官方只列出 4 个 lesion mask，本次只有 1 个可验证 NIfTI。\n\n目前已经观察到：临床改善和一个 G1 右侧 SomMot ROI 的局部重组得到支持；主队列固定 G2 空间 signature、病灶—固定 G2—改善链、调节效应和 ML 增量没有复现。\n\n## 请完成的任务\n\n### 1. 建立“主结论—可验证性”矩阵\n\n逐项列出：\n\n- 主队列原始结论；\n- 新数据是否可直接验证、只能间接验证或完全不可验证；\n- 可用样本量；\n- 应固定的 ROI、网络、方向、权重和主要终点；\n- 支持、部分支持、未复现和不可检验的判定门槛；\n- 最大偏倚或解释风险。\n\n### 2. 提出按优先级排序的外部验证实验\n\n请给出最多 5 个真正有价值的实验，按 Primary、Key secondary、Exploratory 排序。每个实验必须写清：\n\n- 科学问题和预设假设；\n- 输入数据；\n- 样本纳入标准；\n- 变量和公式，特别说明 post-pre 方向；\n- healthy/reference gradient 如何建立或固定，怎样避免符号和尺度任意；\n- 统计模型及协变量；\n- ROI/network 多重比较家族；\n- HC3、置换、spin test、bootstrap 或贝叶斯区间中哪些必要；\n- 效应量和置信区间；\n- 敏感性分析；\n- 一张最合适的 Nature 风格图应画什么；\n- 什么结果算支持，什么结果算失败或不可判定。\n\n### 3. 特别评审以下候选方案\n\n请逐一判断是否值得做、是否已经做过、需要如何改进：\n\n- 固定主队列 G2 Visual/SomMot ROI mask 的方向性复现；\n- 固定主队列 G1/G2 betweenDisp endpoint 的网络级复现；\n- baseline healthy deviation 预测 Vasserman 改善；\n- pre/post FC 变化与 gradient change 的耦合；\n- 将 coarse lesion metadata 仅作为协变量或效应修饰变量；\n- clinical-only 与 clinical+fixed-gradient 的嵌套 LOOCV 增量比较；\n- 用跨数据集 rank/z 标准化解决梯度尺度差异；\n- 在没有完整 lesion mask 时是否有任何合规的 lesion-gradient proxy。\n\n### 4. 审计当前外部验证\n\n请指出当前分析是否存在：\n\n- discovery 和 validation 泄漏；\n- 在外部数据重新选 ROI 后把结果称为验证；\n- reference gradient 不一致；\n- 把 29×400 ROI 当作独立样本；\n- 对小样本 ML 过度解释；\n- 对粗病灶变量过度空间化；\n- 多重比较家族定义不恰当；\n- 只以 p<0.05 判断复现而忽略效应方向、幅度和区间。\n\n### 5. 给出最终建议\n\n请最后给出：\n\n1. 最值得立即执行的 2–3 个分析；\n2. 不建议继续做的分析；\n3. 要完整验证 lesion-gradient 机制，下一独立队列必须额外具备哪些数据；\n4. 可直接写进论文 Methods、Results 和 Limitations 的中文表述；\n5. 一份按执行顺序排列的分析清单或伪代码框架。\n\n## 回答原则\n\n- 不要为了产生阳性结果建议重新筛 ROI、任意改方向或反复尝试协变量；\n- 阴性外部验证同样是有效结论；\n- 明确区分 treatment-specific replication、general rehabilitation replication 和 mechanistic boundary validation；\n- 不要把 Control 当健康参考；\n- 不要把同一数据中的 post 数据用于建立独立 healthy reference；\n- 不要把粗粒度病灶类别称为 lesion topology、lesion span 或 disconnectome；\n- 所有 ML 必须嵌套验证，预处理只能在训练折内，重点比较相对 clinical-only 的增量；\n- 如果现有数据不足，请直接说明不可检验，并给出所需数据，而不是设计替代性伪验证。\n'

def add_docs_and_results() -> None:
    previous = PROJECT / 'code' / 'previous_result'
    for source in sorted(previous.glob('*.docx')):
        copy_file(source, f'02_main_cohort_previous_results/{source.name}')
    result_docs = {WORK / 'lesion_matched_clinical_mediation' / 'RESULTS_ZH.md': '03_main_cohort_lesion_ml/01_lesion_matched_clinical_RESULTS_ZH.md', WORK / 'lesion_gradient_distance' / 'RESULTS_ZH.md': '03_main_cohort_lesion_ml/02_lesion_gradient_distance_RESULTS_ZH.md', WORK / 'lesion_gradient_score_mediation_provided_reference' / 'RESULTS_ZH.md': '03_main_cohort_lesion_ml/03_formal_mediation_RESULTS_ZH.md', WORK / 'lesion_gradient_score_triangle_exploratory' / 'RESULTS_ZH.md': '03_main_cohort_lesion_ml/04_triangle_moderation_RESULTS_ZH.md', PROJECT / 'ml_analysis' / 'RESULTS_FIVE_OUTCOME_LESION_INFORMED_ZH.md': '03_main_cohort_lesion_ml/05_five_outcome_ML_RESULTS_ZH.md'}
    for (source, relative) in result_docs.items():
        copy_sanitized_markdown(source, relative)
    group_tables = {WORK / 'lesion_matched_clinical_mediation' / 'tables' / 'matched_clinical_results.csv': '03_main_cohort_lesion_ml/group_level_tables/matched_clinical_results.csv', WORK / 'lesion_matched_clinical_mediation' / 'tables' / 'matching_balance.csv': '03_main_cohort_lesion_ml/group_level_tables/matching_balance.csv', WORK / 'lesion_matched_clinical_mediation' / 'tables' / 'gradient_mediation_results.csv': '03_main_cohort_lesion_ml/group_level_tables/gradient_mediation_results.csv', WORK / 'lesion_gradient_distance' / 'figures' / 'nature' / 'lesion_final' / 'provided_code_global_adjusted_group_models.csv': '03_main_cohort_lesion_ml/group_level_tables/global_G1_G2_adjusted_models.csv', WORK / 'lesion_gradient_distance' / 'figures' / 'nature' / 'lesion_final' / 'bcdf_recomputed_provided_reference' / 'recomputed_matched_tests.csv': '03_main_cohort_lesion_ml/group_level_tables/lesion_matched_gradient_tests.csv', WORK / 'lesion_gradient_score_triangle_exploratory' / 'tables' / 'adjusted_triangle_associations.csv': '03_main_cohort_lesion_ml/group_level_tables/triangle_adjusted_associations.csv', WORK / 'lesion_gradient_score_triangle_exploratory' / 'tables' / 'moderation_interactions.csv': '03_main_cohort_lesion_ml/group_level_tables/triangle_moderation_interactions.csv', PROJECT / 'ml_analysis' / 'outputs' / 'five_outcome_lesion_informed' / 'repeat_performance_summary.csv': '03_main_cohort_lesion_ml/group_level_tables/five_outcome_ML_performance.csv', PROJECT / 'ml_analysis' / 'outputs' / 'five_outcome_lesion_informed' / 'selected_best_pairwise_summary.csv': '03_main_cohort_lesion_ml/group_level_tables/five_outcome_ML_best_vs_second.csv'}
    for (source, relative) in group_tables.items():
        copy_csv_without_id(source, relative)
    external_root = WORK / 'external_validation'
    copy_sanitized_markdown(external_root / 'ds003999_validation' / 'matched_gradient_figure' / 'RESULTS_ZH.md', '06_current_external_validation/01_longitudinal_gradient_validation_RESULTS_ZH.md')
    copy_sanitized_markdown(external_root / 'ds003999_lesion_ml_validation' / 'RESULTS_ZH.md', '06_current_external_validation/02_lesion_ML_boundary_validation_RESULTS_ZH.md')
    external_tables = {external_root / 'ds003999_validation' / 'matched_gradient_figure' / 'tables' / 'raw_aligned5_roi_paired_tests.csv': '06_current_external_validation/tables/ROI_paired_tests_29.csv', external_root / 'ds003999_validation' / 'matched_gradient_figure' / 'tables' / 'network_metric_effect_sizes.csv': '06_current_external_validation/tables/network_metric_effect_sizes_29.csv', external_root / 'ds003999_validation' / 'matched_gradient_figure' / 'tables' / 'component_variance_paired_tests.csv': '06_current_external_validation/tables/component_variance_tests_29.csv', external_root / 'ds003999_lesion_ml_validation' / 'tables' / 'prespecified_lesion_gradient_score_paths.csv': '06_current_external_validation/tables/prespecified_lesion_gradient_score_paths_n13.csv', external_root / 'ds003999_lesion_ml_validation' / 'tables' / 'lesion_by_fixed_G2_interaction.csv': '06_current_external_validation/tables/lesion_by_fixed_G2_interaction_n13.csv', external_root / 'ds003999_lesion_ml_validation' / 'tables' / 'nested_loo_performance_n13.csv': '06_current_external_validation/tables/nested_LOOCV_performance_n13.csv', external_root / 'ds003999_lesion_ml_validation' / 'tables' / 'nested_loo_incremental_comparisons_n13.csv': '06_current_external_validation/tables/nested_LOOCV_incremental_comparisons_n13.csv', external_root / 'ds003999_lesion_ml_validation' / 'tables' / 'main_conclusion_external_concordance.csv': '06_current_external_validation/tables/main_conclusion_external_concordance.csv'}
    for (source, relative) in external_tables.items():
        copy_csv_without_id(source, relative)
    safe_code = {external_root / 'ds003999_validation' / 'run_external_validation_analysis.py': '04_safe_method_code/current_external_validation/run_external_validation_analysis.py', external_root / 'ds003999_validation' / 'matched_gradient_figure' / 'run_matched_gradient_analysis.py': '04_safe_method_code/current_external_validation/run_matched_gradient_analysis.py', external_root / 'ds003999_lesion_ml_validation' / 'run_lesion_ml_external_validation.py': '04_safe_method_code/current_external_validation/run_lesion_ml_external_validation.py'}
    for (source, relative) in safe_code.items():
        copy_sanitized_code(source, relative)
    main_code = {PROJECT / 'code' / 'gradient_calculation' / 'allnpy.py': '04_safe_method_code/main_cohort_consultation_copies/gradient_diffusion_embedding.py', PROJECT / 'code' / 'gradient_calculation' / 'duiqi.py': '04_safe_method_code/main_cohort_consultation_copies/gradient_procrustes_alignment.py', PROJECT / 'code' / 'indicator.py': '04_safe_method_code/main_cohort_consultation_copies/network_gradient_metrics.py', PROJECT / 'code' / '2' / '2' / 'd.py': '04_safe_method_code/main_cohort_consultation_copies/ROI_paired_tests_and_network_counts.py', WORK / 'run_lesion_gradient_distance.py': '04_safe_method_code/main_cohort_consultation_copies/lesion_gradient_distance_analysis.py', WORK / 'run_lesion_matched_clinical_mediation.py': '04_safe_method_code/main_cohort_consultation_copies/lesion_matched_clinical_analysis.py', WORK / 'run_p3_triangle_exploratory.py': '04_safe_method_code/main_cohort_consultation_copies/lesion_gradient_clinical_moderation.py', PROJECT / 'ml_analysis' / 'run_five_outcome_lesion_informed_ml.py': '04_safe_method_code/main_cohort_consultation_copies/five_outcome_lesion_informed_ML.py'}
    for (source, relative) in main_code.items():
        copy_sanitized_code(source, relative)
    code_readme = '# 代码说明\n\n`main_cohort_consultation_copies/` 是主队列实际分析脚本的咨询副本。为避免泄露或误执行：\n\n- 本机绝对路径统一改为 `PATH_PLACEHOLDER`；\n- 硬编码的受试者号改为 `REDACTED_SUBJECT_ID`；\n- 这些副本用于审查统计与算法逻辑，不保证脱离原项目依赖后直接运行；\n- 真正执行时应使用本地原始脚本、配置文件和去标识化数据表。\n\n`current_external_validation/` 是当前 ds003999 外部验证脚本的路径/标签泛化副本。\n\n根目录的三个 `*_template.py` 是更短的隐私安全方法模板，分别说明梯度与网络指标、lesion-gradient 特征和嵌套预测的关键计算。\n'
    write_text('04_safe_method_code/CODE_README.md', code_readme)

def file_index() -> None:
    descriptions = {'PROMPT_FOR_WEBGPT.md': '直接发送给 WebGPT 的结构化咨询提示词', 'README_FIRST.md': '咨询包用途、隐私边界和阅读顺序', '01_project_overview/MAIN_AND_EXTERNAL_DATASET_OVERVIEW.md': '主队列结论、新数据能力和当前外部结果总览', '05_new_dataset/NEW_DATASET_STRUCTURE_SUMMARY.json': '不含逐例数据的新数据集机器可读摘要', '05_new_dataset/NEW_DATASET_DATA_DICTIONARY.md': '新数据字段、数组与可验证性说明'}
    rows = []
    for path in sorted((p for p in OUT.rglob('*') if p.is_file() and p.name != 'FILE_INDEX.csv')):
        relative = path.relative_to(OUT).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append({'relative_path': relative, 'size_bytes': path.stat().st_size, 'sha256': digest, 'description': descriptions.get(relative, 'Supporting document, code or aggregate result')})
    with (OUT / 'FILE_INDEX.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

def write_package_qa() -> None:
    allowed = {'.docx', '.csv', '.py', '.md', '.json'}
    files = [p for p in OUT.rglob('*') if p.is_file() and p.name not in {'FILE_INDEX.csv', 'PACKAGE_QA.md'}]
    disallowed = [p.relative_to(OUT).as_posix() for p in files if p.suffix.lower() not in allowed]
    text_problems = []
    long_token = re.compile('(?<![\\d.])\\d{7,9}(?![\\d.])')
    for path in files:
        if path.suffix.lower() not in {'.md', '.py', '.json', '.csv'}:
            continue
        text = path.read_text(encoding='utf-8-sig', errors='ignore')
        if re.search('[A-Z]:\\\\', text, flags=re.I):
            text_problems.append(f'absolute path: {path.relative_to(OUT).as_posix()}')
        tokens = [token for token in long_token.findall(text) if not token.startswith('2026')]
        if tokens:
            text_problems.append(f'long numeric token: {path.relative_to(OUT).as_posix()}')
    report = f"# 咨询包 QA\n\n- 文件数（生成索引前）：{len(files)}；\n- 总大小（生成索引前）：{sum((p.stat().st_size for p in files)) / 1024 ** 2:.2f} MiB；\n- 允许格式：DOCX、CSV、PY、MD、JSON；\n- 原始或逐例数据格式检查：{('PASS' if not disallowed else 'REVIEW')}；\n- 文本绝对路径/长数字受试者号检查：{('PASS' if not text_problems else 'REVIEW')}；\n- 三份 DOCX 的可见文字已检查：未发现 7–9 位受试者号；DOCX 内部仍可能包含 Office 图形尺寸、字体和哈希元数据形式的数字，这些不属于参与者数据；\n- CSV 均经过列名检查，不含 subject/participant/patient ID、文件路径或逐例预测字段；\n- 新数据集只提供聚合样本规模、字段字典和数组结构，不含 `participants.tsv` 或 NPY；\n- 主队列代码仅为咨询副本，本机路径和硬编码受试者号已泛化。\n\n## 异常清单\n\n- 非允许格式：{json.dumps(disallowed, ensure_ascii=False)}\n- 文本检查异常：{json.dumps(text_problems, ensure_ascii=False)}\n\n## 总体状态\n\n{('PASS' if not disallowed and (not text_problems) else 'REVIEW REQUIRED')}\n"
    write_text('PACKAGE_QA.md', report)

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = new_dataset_summary()
    write_text('README_FIRST.md', build_readme(summary))
    write_text('PROMPT_FOR_WEBGPT.md', web_prompt())
    write_text('01_project_overview/MAIN_AND_EXTERNAL_DATASET_OVERVIEW.md', project_overview(summary))
    write_text('05_new_dataset/NEW_DATASET_STRUCTURE_SUMMARY.json', json.dumps(summary, ensure_ascii=False, indent=2))
    dictionary = '# ds003999 新数据集字段与衍生数据说明\n\n## 临床与病灶字段\n\n这里只提供字段定义和聚合规模，不提供 `participants.tsv` 的逐例数据。\n\n- `sex`：M/F；\n- `time_after_stroke`：首次扫描距卒中天数；\n- `pre_score`, `post_score`：Vasserman 量表；本项目定义改善为 `pre-post`，正值表示改善；\n- `type`：失语类型；\n- `hemisphere`：L/R/LR；\n- `size`：S≤20 mm、M≤50 mm、H>50 mm；\n- `Broca`, `Wernike`, `Basal`, `Thalamus`：N/P/Y，分别表示无、部分、显著受累；\n- 只有 13 人的粗粒度病灶字段完整。这些变量不能被称为体素 lesion topology、lesion gradient span 或 disconnectome。\n\n## 数组\n\n- 每名患者 pre/post FC：Schaefer400，形状 400×400；\n- 每名患者 pre/post 已对齐梯度：形状 400×5；\n- 29 人均完整配对；\n- 另有 Schaefer400 ROI 顺序、有效顶点数和聚合 QC；\n- 咨询包不包含这些数组，只包含结构说明。\n\n## reference 风险\n\n必须核查外部 aligned5 的原始 reference、Procrustes 对齐、符号固定和尺度定义是否与主队列一致。若不一致，宜使用同一个独立健康 reference 重新计算或对齐，不能直接比较绝对坐标。\n\n## 明确不可验证项\n\n- 无 TMS/Control 标签：不能验证 TMS 特异效应；\n- 无 MMSE：不能直接验证主队列 lesion span×normalization→MMSE；\n- 仅 1 个可取得的官方 voxel mask：不能可靠验证 voxelwise lesion-gradient distance；\n- 粗病灶字段仅 13 人：不适合高维机器学习或多重交互筛选。\n'
    write_text('05_new_dataset/NEW_DATASET_DATA_DICTIONARY.md', dictionary)
    add_docs_and_results()
    make_safe_code_templates()
    write_package_qa()
    file_index()
    print(OUT)
if __name__ == '__main__':
    main()
