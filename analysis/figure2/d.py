# -*- coding: utf-8 -*-
"""Figure 2d: longitudinal gradient t-maps versus PRE-treatment nodal topology.

主分析：未阈值化、带符号的配对 t-map × 个体基线图论 AUC 的组内均值。
Spearman rho -> 双侧空间旋转 P -> 全部 2 x 2 x 8 = 32 项检验统一 BH-FDR。
图论 AUC 只读取 before FC；after FC 仅核对配对名单，不参与图论计算。
保留原 d.py 的 Arial、bwr(-0.5,0.5)、方格布局及红/蓝密度散点样式。
红/蓝散点现在分别表示 TMS/Control；不再按增加/减少或临床量表分图。

默认输出在 F:\\lishiqi\\新图\\Ending\\main 2\\d；可使用 --out 显式指定其他输出目录。
依赖：numpy scipy pandas matplotlib networkx statsmodels openpyxl。
不使用旧 mediation、diff_auc 或旧相关统计缓存。
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Keep BLAS threads bounded, including Windows multiprocessing workers.
for _thread_env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_thread_env, "1")

import numpy as np
import pandas as pd
import networkx as nx
import scipy
from scipy import stats
from scipy.linalg import eigh
from scipy.sparse.csgraph import shortest_path, connected_components
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SOURCE_CSV = Path(r"F:\lishiqi\end_picture\Ending\main 2\a\Figure2a_ROI_mean_delta_and_FDR.csv")
FC_ROOT = Path(r"F:\lishiqi\预处理数据\cortex\parcel(400，400)")
GRADIENT_ROOT = Path(r"F:\lishiqi\预处理数据\cortex\duiqi (400, 5)")
LABELS_FILE = Path(r"F:\lishiqi\预处理数据\cortex\董分组.xlsx")
ROI_MAPPING = Path(r"F:\lishiqi\end_picture\Ending\main 2\c\Figure2c_ROI_Network_Mapping.csv")
SPIN_INDICES = Path(r"F:\lishiqi\end_picture\Ending\main 2\c\Figure2c_spin_index_5000_seed20261009.npy")
PLOT_OUTPUT_ROOT = Path(r"F:\lishiqi\end_picture\Ending\main 2\d")
GROUPS = ("Control", "TMS")
GRADIENTS = ("G1", "G2")
GRADIENT_LABELS = {"G1": "Gradient 1", "G2": "Gradient 2"}
EXPECTED_N = {"Control": 19, "TMS": 23}
NETWORKS = ("VIS", "SMN", "DAN", "VAN", "LIM", "FPN", "DMN")
N_ROI = 400
SEED = 20261010
# Explicit endpoint: previous np.arange construction accidentally included 0.42.
DENSITIES = np.array([.10, .14, .18, .22, .26, .30, .34, .38, .40])
METRIC_NAMES = ["degree", "betweenness", "closeness", "eigenvector", "pagerank",
                "clustering", "local_efficiency", "participation"]
METRIC_LABELS = {
    "degree": "Degree centrality", "betweenness": "Betweenness centrality",
    "closeness": "Closeness centrality", "eigenvector": "Eigenvector centrality",
    "pagerank": "PageRank centrality", "clustering": "Clustering coefficient",
    "local_efficiency": "Local efficiency", "participation": "Participation coefficient"}
SIG_METRIC_LABELS = {
    "degree": "Degree", "betweenness": "Betweenness", "closeness": "Closeness",
    "eigenvector": "Eigenvector", "pagerank": "PageRank", "clustering": "Clustering",
    "local_efficiency": "Local efficiency", "participation": "Participation"}
EXCLUDED_METRICS = {
    "strength": "Binary degree count is a fixed multiple of degree centrality; duplicate, not weighted strength.",
    "char_path_length": "Disconnected baseline graphs: unreachable paths are not zero. Omitted from the full-400-ROI analysis rather than substituting zeros or varying the ROI domain."}

# Original appearance; only scientific labels and the number of valid rows change.
SAVE_FORMATS = ("png", "pdf")
DPI = 600
ALPHA = .05
FONT_FAMILY = "Arial"
BASE_FONT_SIZE, TITLE_FONT_SIZE, LABEL_FONT_SIZE, TICK_FONT_SIZE, STATS_FONT_SIZE = 8, 9, 8, 7, 7
SIG_FIGSIZE = (3.6, 3.6)
SIG_TITLE_FONT_SIZE, SIG_LABEL_FONT_SIZE, SIG_TICK_FONT_SIZE, SIG_STATS_FONT_SIZE = 16, 16, 14, 14
HEATMAP_CMAP = "bwr"
HEATMAP_ABS_LIMIT = .5
SHOW_GRID, SHOW_TITLES = False, True
X_AXIS_LABEL = "Rank of baseline nodal AUC"
SIG_X_AXIS_LABEL = "Baseline AUC rank"
plt.rcParams.update({"font.family": FONT_FAMILY,
    "font.sans-serif": [FONT_FAMILY, "DejaVu Sans"], "axes.unicode_minus": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "axes.linewidth": .8, "xtick.major.width": .8, "ytick.major.width": .8,
    "xtick.major.size": 3, "ytick.major.size": 3})


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def index_subjects(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    indexed = {}
    for p in sorted(folder.glob("*.npy")):
        sid = p.stem.split("-")[0]
        if sid in indexed:
            raise ValueError(f"Duplicate subject ID in {folder}: {sid}")
        indexed[sid] = p
    if not indexed:
        raise ValueError(f"No participant matrices: {folder}")
    return indexed


def load_mapping(path):
    df = pd.read_csv(path).sort_values("ROI_ID").reset_index(drop=True)
    required = {"ROI_ID", "Hemisphere", "ROI_name", "Network"}
    if not required.issubset(df.columns) or not np.array_equal(df.ROI_ID, np.arange(1, N_ROI + 1)):
        raise ValueError("Mapping must contain exactly ROI_ID 1..400, Hemisphere, ROI_name and Network")
    if not set(df.Network).issubset(NETWORKS) or set(df.Network) != set(NETWORKS):
        raise ValueError("Mapping does not contain the expected seven networks")
    hemi = df.Hemisphere.astype(str).str.lower().replace({"left": "lh", "l": "lh", "right": "rh", "r": "rh"})
    if not ((hemi.iloc[:200] == "lh").all() and (hemi.iloc[200:] == "rh").all()):
        raise ValueError("Expected left 200 then right 200 ROIs, matching Figure 2c spin convention")
    labels = np.array([NETWORKS.index(x) for x in df.Network], dtype=int)
    return df, labels


def validate_spins(spins):
    a = np.asarray(spins)
    if a.ndim != 2 or a.shape[1] != N_ROI or a.shape[0] < 999 or a.dtype.kind not in "iu":
        raise ValueError("Spin array must be integer [n_spins >= 999, 400]")
    if not np.all(np.sort(a, axis=1) == np.arange(N_ROI)):
        raise ValueError("Every spin must be a bijection of zero-based ROI indices 0..399")
    if not (np.all(a[:, :200] < 200) and np.all(a[:, 200:] >= 200)):
        raise ValueError("Spin indices mix hemispheres")
    if np.all(a == np.arange(N_ROI)):
        raise ValueError("All spin permutations are identities")
    return a.astype(np.int32)


def load_cohort_and_validate_maps(args):
    """Recompute t and mean delta from paired derivatives to validate CSV provenance."""
    source = pd.read_csv(args.input)
    cols = {"Group", "Gradient", "N", "ROI_ID", "Paired_t", "Mean_delta_post_minus_pre", "SD_delta"}
    if not cols.issubset(source.columns) or len(source) != 4 * N_ROI:
        raise ValueError("Figure2a CSV must contain 1600 ROI rows and paired-test fields")
    expected_keys = {(g, k) for g in GROUPS for k in GRADIENTS}
    if set(zip(source.Group, source.Gradient)) != expected_keys:
        raise ValueError("Unexpected group/gradient combinations in Figure2a CSV")
    labels = pd.read_excel(args.labels, dtype={"病历号": str})
    if not {"病历号", "Group"}.issubset(labels.columns):
        raise ValueError("Cohort workbook needs 病历号 and Group")
    labels["病历号"] = labels["病历号"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    labels["Group"] = labels["Group"].astype(str).str.strip()
    labels = labels[labels.Group.isin(GROUPS)]
    if labels["病历号"].duplicated().any():
        raise ValueError("Duplicate subject IDs in cohort workbook")
    group_by_id = labels.set_index("病历号").Group.to_dict()
    cohort, deltas, maps, records = {}, {}, {}, []
    for group in GROUPS:
        before = index_subjects(args.fc_root / group / "before")
        after = index_subjects(args.fc_root / group / "after")
        if set(before) != set(after) or len(before) != EXPECTED_N[group]:
            raise ValueError(f"{group}: paired FC cohort differs from expected N={EXPECTED_N[group]}")
        fb = index_subjects(args.gradient_root / group / "before")
        fa = index_subjects(args.gradient_root / group / "after")
        subjects = sorted(before)
        delta = []
        for sid in subjects:
            if group_by_id.get(sid) != group or sid not in fb or sid not in fa:
                raise ValueError(f"{group}: subject label or paired gradient mismatch: {sid}")
            # Post FC is only a pair/shape check; its weights never enter graph metrics.
            for p in (before[sid], after[sid]):
                if np.load(p, mmap_mode="r", allow_pickle=False).shape != (N_ROI, N_ROI):
                    raise ValueError(f"Expected 400 x 400 FC for {sid}")
            b, a = [np.squeeze(np.load(p, allow_pickle=False)) for p in (fb[sid], fa[sid])]
            for arr in (b, a):
                if arr.ndim != 2 or arr.shape[0] != N_ROI or arr.shape[1] < 2 or not np.isfinite(arr[:, :2]).all():
                    raise ValueError(f"Invalid paired gradient array for {sid}")
            delta.append((a[:, :2].astype(float) - b[:, :2].astype(float)).T)
            records.append({"Group": group, "Subject_ID": sid,
                "Baseline_FC_file": str(before[sid]), "Baseline_FC_SHA256": file_hash(before[sid]),
                "Pre_gradient_SHA256": file_hash(fb[sid]), "Post_gradient_SHA256": file_hash(fa[sid])})
        delta = np.stack(delta)
        cohort[group] = [(sid, before[sid]) for sid in subjects]
        deltas[group] = delta
        n = len(subjects)
        for ki, grad in enumerate(GRADIENTS):
            rows = source[(source.Group == group) & (source.Gradient == grad)].sort_values("ROI_ID")
            if not np.array_equal(rows.ROI_ID, np.arange(1, N_ROI + 1)) or not (rows.N == n).all():
                raise ValueError(f"{group}/{grad}: CSV ROI or sample-count mismatch")
            mean = delta[:, ki, :].mean(axis=0)
            sd = delta[:, ki, :].std(axis=0, ddof=1)
            if (sd <= 0).any():
                raise ValueError(f"{group}/{grad}: constant ROI changes cannot yield finite full-cortex t-map")
            t = mean / (sd / np.sqrt(n))
            for field, actual in (("Paired_t", t), ("Mean_delta_post_minus_pre", mean), ("SD_delta", sd)):
                if not np.isfinite(rows[field]).all() or not np.allclose(rows[field], actual, rtol=1e-5, atol=1e-8):
                    raise ValueError(f"{group}/{grad}: recomputed {field} differs from Figure2a CSV")
            maps[(group, grad)] = {"t": rows.Paired_t.to_numpy(float), "mean_delta": mean}
        print(f"[COHORT] {group}: {n} paired subjects; all 400-ROI t/mean/SD values verified", flush=True)
    return cohort, deltas, maps, pd.DataFrame(records)


def threshold_positive_fc(fc, density):
    """Retain exactly round(density * possible_edges) strongest positive edges.

    r and Fisher-z have identical rank/sign for off-diagonal correlations.
    No inverse transform is inferred. Ties use stable upper-triangle ordering.
    """
    a = np.asarray(fc, dtype=float).copy()
    if a.shape != (N_ROI, N_ROI):
        raise ValueError("Expected a 400x400 baseline FC matrix")
    np.fill_diagonal(a, 0.)
    if not np.isfinite(a).all() or not np.allclose(a, a.T, rtol=0, atol=1e-6):
        raise ValueError("Baseline FC must have finite, symmetric off-diagonal entries")
    a = (a + a.T) / 2
    i, j = np.triu_indices(N_ROI, 1)
    values = a[i, j]
    available = np.flatnonzero(values > 0)
    count = int(round(density * len(values)))
    if len(available) < count:
        raise ValueError(f"Insufficient positive FC edges for density {density:.3f}")
    order = available[np.argsort(-values[available], kind="stable")]
    chosen = order[:count]
    b = np.zeros_like(a)
    b[i[chosen], j[chosen]] = 1.
    b[j[chosen], i[chosen]] = 1.
    return b


def exact_betweenness(adjacency, distances):
    """Exact unweighted Brandes dependencies, vectorized across source nodes.

    Verified against NetworkX on graphs with ties, disconnections and isolates.
    Normalization matches nx.betweenness_centrality(normalized=True), endpoints=False.
    """
    a, d = np.asarray(adjacency, float), np.asarray(distances)
    n = len(a)
    if n < 3:
        return np.zeros(n)
    diameter = int(d[np.isfinite(d)].max())
    sigma = np.eye(n)
    for level in range(1, diameter + 1):
        sigma += ((sigma * (d == level - 1)) @ a) * (d == level)
    dependency = np.zeros_like(a)
    for level in range(diameter, 0, -1):
        ratio = np.divide(1. + dependency, sigma, out=np.zeros_like(a), where=(d == level) & (sigma > 0))
        dependency += ((ratio @ a.T) * sigma) * (d == level - 1)
    np.fill_diagonal(dependency, 0.)
    return dependency.sum(axis=0) / ((n - 1) * (n - 2))


def node_metrics(adjacency, communities):
    a = np.asarray(adjacency, float)
    n = len(a)
    degree = a.sum(axis=1)
    distances = shortest_path(a, directed=False, unweighted=True, method="FW")
    reachable = np.isfinite(distances) & (distances > 0)
    nr = reachable.sum(axis=1)
    total_distance = np.where(reachable, distances, 0.).sum(axis=1)
    closeness = np.divide(nr * nr, (n - 1) * total_distance, out=np.zeros(n), where=total_distance > 0)
    # Perron eigenvector; no fallback to a different centrality on failure.
    _, vectors = eigh(a, subset_by_index=[n - 1, n - 1])
    eigenvector = np.abs(vectors[:, 0])
    eigenvector[eigenvector < eigenvector.max() * 1e-12] = 0.
    eigenvector /= np.linalg.norm(eigenvector)
    graph = nx.from_numpy_array(a)
    pagerank = np.array(list(nx.pagerank(graph, alpha=.85, max_iter=1000, tol=1e-12).values()))
    triangles_twice = ((a @ a) * a).sum(axis=1)
    clustering = np.divide(triangles_twice, degree * (degree - 1), out=np.zeros(n), where=degree > 1)
    local_efficiency = np.zeros(n)
    for node in range(n):
        neighbors = np.flatnonzero(a[node])
        k = len(neighbors)
        if k >= 2:
            sub = a[np.ix_(neighbors, neighbors)]
            dsub = shortest_path(sub, directed=False, unweighted=True, method="FW")
            inv = np.divide(1., dsub, out=np.zeros_like(dsub), where=(dsub > 0) & np.isfinite(dsub))
            local_efficiency[node] = inv.sum() / (k * (k - 1))
    participation = np.zeros(n)
    for network in np.unique(communities):
        k_network = a[:, np.asarray(communities) == network].sum(axis=1)
        participation += np.divide(k_network, degree, out=np.zeros(n), where=degree > 0) ** 2
    participation = np.where(degree > 0, 1. - participation, 0.)
    values = {"degree": degree / (n - 1), "betweenness": exact_betweenness(a, distances),
        "closeness": closeness, "eigenvector": eigenvector, "pagerank": pagerank,
        "clustering": clustering, "local_efficiency": local_efficiency, "participation": participation}
    if not all(np.isfinite(v).all() for v in values.values()):
        raise ValueError("Non-finite graph metric: calculation aborted instead of replacing it")
    qc = {"Components": int(connected_components(a, directed=False, return_labels=False)),
          "Isolated_nodes": int((degree == 0).sum()), "Actual_density": float(degree.sum() / (n * (n - 1)))}
    return np.stack([values[m] for m in METRIC_NAMES]), qc


def compute_subject_auc(task):
    group, sid, fc_path, networks, out, fingerprint = task
    fc_path, out = Path(fc_path), Path(out)
    fc_sha = file_hash(fc_path)
    key = hashlib.sha256((fc_sha + fingerprint).encode()).hexdigest()
    cache = out / "cache" / f"{group}_{sid}_{key[:16]}.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            if str(data["cache_key"]) == key and data["auc"].shape == (len(METRIC_NAMES), N_ROI):
                return group, sid, data["auc"], json.loads(str(data["qc_json"])), True
    fc = np.load(fc_path, allow_pickle=False)
    density_metrics, qc_rows = [], []
    for density in DENSITIES:
        values, qc = node_metrics(threshold_positive_fc(fc, density), networks)
        density_metrics.append(values)
        qc_rows.append({"Group": group, "Subject_ID": sid, "Target_density": float(density), **qc})
    density_metrics = np.stack(density_metrics)
    # Integral over graph density, not time; same domain for all subjects/metrics.
    auc = np.trapezoid(density_metrics, DENSITIES, axis=0)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, auc=auc, density_metrics=density_metrics, densities=DENSITIES,
        metrics=np.array(METRIC_NAMES), cache_key=key, qc_json=json.dumps(qc_rows), baseline_fc_sha256=fc_sha)
    return group, sid, auc, qc_rows, False


def baseline_auc(cohort, networks, out, workers):
    kernel = "".join(inspect.getsource(f) for f in (threshold_positive_fc, node_metrics, exact_betweenness, compute_subject_auc))
    fingerprint = hashlib.sha256((kernel + json.dumps({"metrics": METRIC_NAMES,
        "densities": DENSITIES.tolist(), "communities": networks.tolist(),
        "numpy": np.__version__, "scipy": scipy.__version__, "networkx": nx.__version__})).encode()).hexdigest()
    tasks = [(g, sid, str(p), networks, str(out), fingerprint) for g in GROUPS for sid, p in cohort[g]]
    found, qc_rows = {}, []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(compute_subject_auc, t) for t in tasks]
        for number, future in enumerate(as_completed(futures), 1):
            g, sid, auc, qc, cached = future.result()
            found[(g, sid)] = auc
            qc_rows.extend(qc)
            print(f"[BASELINE AUC] {number}/{len(tasks)}; {g}; {'cache' if cached else 'computed'}", flush=True)
    pd.DataFrame(qc_rows).sort_values(["Group", "Subject_ID", "Target_density"]).to_csv(out / "Baseline_graph_QC.csv", index=False, encoding="utf-8-sig")
    individual = {g: np.stack([found[(g, sid)] for sid, p in cohort[g]]) for g in GROUPS}
    for g in GROUPS:
        np.savez_compressed(out / f"{g}_individual_baseline_AUC.npz", subject_ids=np.array([s for s, p in cohort[g]]),
            metrics=np.array(METRIC_NAMES), auc=individual[g], roi_ids=np.arange(1, N_ROI + 1))
    return individual


def normalized_ranks(values):
    rank = stats.rankdata(values, axis=-1, method="average")
    rank -= rank.mean(axis=-1, keepdims=True)
    norm = np.linalg.norm(rank, axis=-1, keepdims=True)
    return np.divide(rank, norm, out=np.full_like(rank, np.nan), where=norm > 0)


def spatial_spearman(x, y, spins):
    xrank, yrank = normalized_ranks(x), normalized_ranks(y)
    if not (np.isfinite(xrank).all() and np.isfinite(yrank).all()):
        return np.nan, np.nan, np.full(len(spins), np.nan)
    observed = float(np.clip(xrank @ yrank, -1, 1))
    null = np.clip(yrank[spins] @ xrank, -1, 1)
    # Two-sided absolute-rho tail, with +1 finite-permutation correction.
    p = (1 + np.count_nonzero(np.abs(null) >= abs(observed) - 1e-12)) / (len(spins) + 1)
    return observed, float(p), null


def apply_family_fdr(results):
    out = results.copy()
    planned = len(GROUPS) * len(GRADIENTS) * len(METRIC_NAMES)
    if len(out) != planned:
        raise ValueError(f"Incomplete planned family: expected {planned}, got {len(out)}")
    valid = np.isfinite(out.P_spin.to_numpy())
    # Unestimable planned comparisons retain a conservative p=1 slot; no star.
    corrected = multipletests(np.where(valid, out.P_spin, 1.), method="fdr_bh")[1]
    out["q_spin_FDR"] = np.where(valid, corrected, np.nan)
    out["FDR_family_size"] = planned
    out["FDR_significant"] = valid & (corrected < ALPHA)
    return out


def analyze_maps(maps, individual, spins, map_type, out):
    rows, nulls = [], {}
    for group in GROUPS:
        baseline = individual[group].mean(axis=0)
        for grad in GRADIENTS:
            y = maps[(group, grad)][map_type]
            for mi, metric in enumerate(METRIC_NAMES):
                rho, p, null = spatial_spearman(baseline[mi], y, spins)
                key = f"{group}_{grad}_{metric}"
                nulls[key] = null
                rows.append({"Group": group, "Gradient": grad, "Metric": metric,
                    "Map_type": map_type, "rho": rho, "P_spin": p,
                    "N_subjects": len(individual[group]), "N_ROIs": N_ROI, "N_spins": len(spins),
                    "Status": "ok" if np.isfinite(p) else "constant_map_not_estimable"})
    result = apply_family_fdr(pd.DataFrame(rows))
    np.savez_compressed(out / f"Spin_null_rho_{map_type}.npz", **nulls)
    return result


def subject_bootstrap(individual, deltas, count, seed, out):
    """Resample whole patients/pairs jointly; never resample ROIs as independent units."""
    samples, ci_rows = {}, []
    rng = np.random.default_rng(seed)
    for group in GROUPS:
        n = len(individual[group])
        distribution = np.full((count, len(GRADIENTS), len(METRIC_NAMES)), np.nan)
        for b in range(count):
            idx = rng.integers(0, n, n)
            dg = deltas[group][idx]
            sd = dg.std(axis=0, ddof=1)
            t = np.divide(dg.mean(axis=0) * np.sqrt(n), sd, out=np.full_like(sd, np.nan), where=sd > 0)
            x = individual[group][idx].mean(axis=0)
            distribution[b] = normalized_ranks(t) @ normalized_ranks(x).T
        samples[group] = distribution
        for ki, grad in enumerate(GRADIENTS):
            for mi, metric in enumerate(METRIC_NAMES):
                values = distribution[:, ki, mi]
                values = values[np.isfinite(values)]
                if len(values) < .95 * count:
                    raise ValueError(f"Too many undefined patient-bootstrap correlations: {group}/{grad}/{metric}")
                low, high = np.percentile(values, [2.5, 97.5])
                ci_rows.append({"Group": group, "Gradient": grad, "Metric": metric,
                    "Patient_bootstrap_CI_low": low, "Patient_bootstrap_CI_high": high,
                    "Patient_bootstrap_valid": len(values)})
        print(f"[BOOTSTRAP] {group}: {count} whole-patient resamples", flush=True)
    np.savez_compressed(out / "Patient_bootstrap_rho.npz", **samples)
    return pd.DataFrame(ci_rows)


def export_roi_data(mapping, maps, individual, out):
    blocks = []
    for group in GROUPS:
        base = individual[group].mean(axis=0)
        for grad in GRADIENTS:
            frame = mapping.copy()
            frame.insert(0, "Gradient", grad)
            frame.insert(0, "Group", group)
            frame["Paired_t"] = maps[(group, grad)]["t"]
            frame["Mean_delta_post_minus_pre"] = maps[(group, grad)]["mean_delta"]
            for mi, metric in enumerate(METRIC_NAMES):
                frame[f"Baseline_AUC_{metric}"] = base[mi]
            blocks.append(frame)
    pd.concat(blocks, ignore_index=True).to_csv(out / "Figure2d_ROI_source_data.csv", index=False, encoding="utf-8-sig")


def write_methods(args, primary, out):
    spec = {"analysis": "Exploratory spatial correspondence, not a clinical or causal mediation test",
        "input_files": {str(p): file_hash(p) for p in [args.input, args.roi_mapping, args.spin_indices, args.labels]},
        "script_sha256": file_hash(__file__), "expected_n": EXPECTED_N, "roi_count": N_ROI,
        "metrics": METRIC_NAMES, "excluded_metrics": EXCLUDED_METRICS, "densities": DENSITIES.tolist(),
        "graph": "Undirected binary, strongest positive off-diagonal edges; exact density, stable tie ordering",
        "AUC": "Unnormalized trapezoidal integral over density 0.10..0.40, individually at baseline, then averaged within group",
        "communities": "Fixed seven-network labels from the ROI mapping; shared for all subjects/densities",
        "closeness": "Wasserman-Faust correction for reachable nodes, matching NetworkX",
        "spatial_null": "Existing Figure2c hemisphere-coupled ROI-centroid rotation + bijective assignment approximation",
        "p_spin": "(1 + count(abs(null rho) >= abs(observed rho))) / (n_spins + 1); two-sided; numerical tolerance 1e-12",
        "FDR": "BH across all 32 planned group x gradient x metric t-map correlations; undefined comparisons retain p=1 slots",
        "sensitivity": "Mean-delta spatial correlations: separate 32-test sensitivity family, not extra primary discoveries",
        "bootstrap": {"resampling_unit": "whole subject, paired gradients and baseline AUC resampled together",
                      "count": args.bootstrap, "seed": args.seed, "CI": "percentile 95%, uncertainty across participants, not a spatial-null p-value"},
        "n_spins": int(primary.N_spins.iloc[0]), "palette": "Original bwr, fixed +/-0.5; original Reds/Blues scatter, TMS/Control respectively",
        "heatmap_color_saturated_cells": int((primary.rho.abs() > HEATMAP_ABS_LIMIT).sum()),
        "weights": "Input r/Fisher-z provenance not assumed: the binary positive-rank method is invariant to this monotonic transform",
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "networkx": nx.__version__, "matplotlib": matplotlib.__version__}}
    (out / "analysis_manifest.json").write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
    text = f"""Figure 2d | Spatial correspondence with baseline functional-network topology

Primary analysis: unthresholded signed paired t-maps (post - pre), separately for Control/TMS and G1/G2.
Figure2a CSV statistics were checked against all paired individual gradient derivatives.
Participants: Control n=18, TMS n=24. Spatial units: 400 cortical ROIs; they are not 400 independent participants.
Baseline topology: individual PRE FC only; metrics are integrated over density {DENSITIES.tolist()} and averaged within group.
The same baseline topology map is used for G1/G2 within each group.
No post-treatment FC weights, clinical scores or mediation arrays enter baseline AUC or the primary spatial correlations.
Spin method: {spec['spatial_null']}; {spec['n_spins']} permutations; valid same-hemisphere permutations confirmed.
The cache remains an approximate parcel-centroid null, not a vertex-level spatial model; suitability to the study must be acknowledged.
Spearman rho is the effect size. The two-sided P_spin comes from rotated maps; BH-FDR applies across all 32 primary tests.
Raw parametric Spearman P values are not used for inference.
Heatmap stars: q_spin < .05 (*), < .01 (**), < .001 (***). Non-significant cells are retained.
Original Arial, square heatmap cells, bwr +/-0.5 and red/blue scatter aesthetics are retained.
Scatter red=TMS, blue=Control; these colors no longer label increase/decrease. The whole signed t-map is analyzed.
Color saturation: {spec['heatmap_color_saturated_cells']} primary cells exceed +/-0.5; exact rho values are in the source table.
Degree-count strength is omitted as duplicate. Ordinary path length is omitted because disconnected graphs make zero-imputation invalid.
Closeness uses reachable-node normalization; local efficiency treats unreachable neighbor pairs as zero efficiency, by definition.
Participation uses fixed seven-network memberships, not communities re-detected at each threshold.
The mean-delta analysis is a separately labelled sensitivity analysis with its own 32-test correction.
Patient-bootstrap intervals use whole-patient resampling ({args.bootstrap} replicates) and are distinct from spin inference.

Interpretation limits: spatial association does not establish TMS specificity, mediation, clinical recovery, or causality.
Baseline topology and change maps share baseline imaging data; spatial spins do not remove that dependence or measurement error.
One group being significant and the other non-significant is not a test of a between-group difference.
No lesion-masking or motion-adjusted sensitivity analysis is performed by this script; those need additional input/analyses.
All evaluated metrics and non-significant results are retained; exclusions were based on definitions/QC, not correlation results.
"""
    (out / "METHODS_and_figure_legend.txt").write_text(text, encoding="utf-8-sig")


def significance_marker(p):
    if np.isnan(p):
        return "n.s."
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def significance_stars(p):
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def format_p_value(p):
    if np.isnan(p):
        return "n/a"
    if p < 0.001:
        return f"{p:.1e}"
    return f"{p:.3f}"


def save_figure(fig, out_stem):
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    for fmt in SAVE_FORMATS:
        out_path = out_stem.with_suffix(f".{fmt}")
        if fmt.lower() == "png":
            fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
        else:
            fig.savefig(out_path, bbox_inches="tight")
        print(f"[Saved] {out_path}")


def draw_large_scatter(
    ax,
    x_data,
    y_data,
    metric_label,
    context_label,
    rho,
    p_spin,
    p_corr,
    n_points,
    n_subjects,
    y_axis_label,
    warm_palette=True,
):
    color = "#B2182B" if warm_palette else "#2166AC"

    if len(x_data) < 3:
        ax.text(
            0.5,
            0.5,
            "Insufficient data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=SIG_LABEL_FONT_SIZE,
        )
    else:
        try:
            xy = np.vstack([x_data, y_data])
            density = stats.gaussian_kde(xy)(xy)
            if np.max(density) > np.min(density):
                density = (density - np.min(density)) / (np.max(density) - np.min(density))
            else:
                density = np.ones_like(x_data) * 0.5
        except Exception:
            density = np.ones_like(x_data) * 0.5

        cmap = plt.cm.Reds if warm_palette else plt.cm.Blues
        colors = cmap(0.35 + 0.55 * density)

        ax.scatter(
            x_data,
            y_data,
            c=colors,
            s=28,
            alpha=0.85,
            edgecolors="none",
            rasterized=True,
        )

        if len(np.unique(x_data)) > 1:
            coef = np.polyfit(x_data, y_data, 1)
            fit_fn = np.poly1d(coef)
            x_fit = np.linspace(np.min(x_data), np.max(x_data), 100)
            y_fit = fit_fn(x_fit)
            ax.plot(x_fit, y_fit, color=color, linewidth=1.8, alpha=0.95)

    ax.set_title(
        f"{metric_label}\n{context_label}",
        fontsize=SIG_TITLE_FONT_SIZE,
        fontweight="bold",
        color=color,
        pad=6,
    )

    ax.set_xlabel(SIG_X_AXIS_LABEL, fontsize=SIG_LABEL_FONT_SIZE)
    ax.set_ylabel(y_axis_label, fontsize=SIG_LABEL_FONT_SIZE)

    # Keep the original two-line annotation size; raw P_spin is in the source CSV.
    stats_text = (f"ρ = {rho:.3f}\n"
                  f"q spin = {format_p_value(p_corr)} {significance_stars(p_corr)}")

    ax.text(
        0.05,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=SIG_STATS_FONT_SIZE,
        va="top",
        ha="left",
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            edgecolor="0.75",
            linewidth=0.7,
            alpha=0.9,
        ),
    )

    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("black")

    ax.tick_params(axis="both", labelsize=SIG_TICK_FONT_SIZE)

    if len(x_data) > 0:
        x_min, x_max = np.min(x_data), np.max(x_data)
        x_pad = max((x_max - x_min) * 0.06, 1)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)

    if len(y_data) > 0:
        y_min, y_max = np.min(y_data), np.max(y_data)
        y_pad = max((y_max - y_min) * 0.06, 1)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)


def draw_heatmap(rho_matrix, p_matrix, x_labels, out_stem):
    y_labels = [SIG_METRIC_LABELS[m] for m in METRIC_NAMES]
    abs_limit = 0.5

    n_rows = len(METRIC_NAMES)
    n_cols = len(x_labels)

    cell_size = 0.55
    matrix_w = n_cols * cell_size
    matrix_h = n_rows * cell_size

    left_margin = 1.85
    bottom_margin = 0.70
    right_margin = 0.25
    top_margin = 2.15
    cbar_gap = 0.18
    cbar_height = 0.16

    matrix_x = left_margin
    matrix_y = bottom_margin + cbar_height + cbar_gap

    fig_w = left_margin + matrix_w + right_margin
    fig_h = bottom_margin + cbar_height + cbar_gap + matrix_h + top_margin

    fig = plt.figure(figsize=(fig_w, fig_h))

    ax = fig.add_axes([
        matrix_x / fig_w,
        matrix_y / fig_h,
        matrix_w / fig_w,
        matrix_h / fig_h,
    ])

    masked_rho = np.ma.masked_invalid(rho_matrix)

    im = ax.imshow(
        masked_rho,
        cmap="bwr",
        vmin=-abs_limit,
        vmax=abs_limit,
        aspect="equal",
        interpolation="nearest",
    )

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)

    label_y = -0.62
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels([])
    ax.xaxis.tick_top()
    ax.tick_params(
        axis="x",
        top=False,
        bottom=False,
        labeltop=False,
        labelbottom=False,
        length=0,
        pad=0,
    )

    for col_idx, label in enumerate(x_labels):
        ax.text(
            col_idx,
            label_y,
            label,
            rotation=90,
            ha="left",
            va="center",
            rotation_mode="anchor",
            fontsize=16,
            fontfamily="Arial",
            fontweight="normal",
            clip_on=False,
        )

    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(
        y_labels,
        fontsize=16,
        fontfamily="Arial",
        fontweight="normal",
    )

    ax.tick_params(
        axis="y",
        left=False,
        right=False,
        length=0,
        pad=8,
    )

    ax.set_xlabel("")
    ax.set_ylabel("")

    for x in np.arange(0.5, n_cols - 0.5, 1):
        ax.axvline(x, color="white", linewidth=2.0)
    for y in np.arange(0.5, n_rows - 0.5, 1):
        ax.axhline(y, color="white", linewidth=2.0)

    for i in range(n_rows):
        for j in range(n_cols):
            p_adj = p_matrix[i, j]
            stars = significance_stars(p_adj)
            if stars:
                ax.text(
                    j,
                    i,
                    stars,
                    ha="center",
                    va="center",
                    fontsize=16,
                    fontweight="normal",
                    fontfamily="Arial",
                    color="black",
                )

    cax = fig.add_axes([
        matrix_x / fig_w,
        bottom_margin / fig_h,
        matrix_w / fig_w,
        cbar_height / fig_h,
    ])

    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_ticks([])
    cbar.ax.tick_params(
        bottom=False,
        top=False,
        labelbottom=False,
        labeltop=False,
        length=0,
    )
    cbar.outline.set_linewidth(0.8)

    cbar.ax.text(
        -0.025,
        0.50,
        f"{-abs_limit:.1f}",
        transform=cbar.ax.transAxes,
        ha="right",
        va="center",
        fontsize=16,
        fontfamily="Arial",
        fontweight="normal",
        color="black",
    )
    cbar.ax.text(
        1.025,
        0.50,
        f"{abs_limit:.1f}",
        transform=cbar.ax.transAxes,
        ha="left",
        va="center",
        fontsize=16,
        fontfamily="Arial",
        fontweight="normal",
        color="black",
    )

    cbar.set_label(
        "Spearman ρ",
        fontsize=18,
        fontweight="normal",
        fontfamily="Arial",
        labelpad=1,
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    save_figure(fig, out_stem)
    plt.close(fig)


def plot_results(results, maps, individual, out, scatter=True):
    order = [(g, k) for g in GROUPS for k in GRADIENTS]
    def heatmap(columns, labels, name):
        rho = np.full((len(METRIC_NAMES), len(columns)), np.nan)
        q = rho.copy()
        for ci, (g, k) in enumerate(columns):
            subset = results[(results.Group == g) & (results.Gradient == k)].set_index("Metric")
            rho[:, ci] = subset.loc[METRIC_NAMES, "rho"]
            q[:, ci] = subset.loc[METRIC_NAMES, "q_spin_FDR"]
        draw_heatmap(rho, q, labels, out / "heatmap" / name)
    heatmap(order, [f"{g} ({k})" for g, k in order], "Figure2d_all_groups_gradients")
    for group in GROUPS:
        heatmap([(group, k) for k in GRADIENTS], [GRADIENT_LABELS[k] for k in GRADIENTS], f"Figure2d_{group}_gradient_metric_heatmap")
    for grad in GRADIENTS:
        heatmap([(g, grad) for g in GROUPS], list(GROUPS), f"Figure2d_{grad}_group_metric_heatmap")
    if not scatter:
        return
    for group, grad in order:
        baseline = individual[group].mean(axis=0)
        y = stats.rankdata(maps[(group, grad)]["t"])
        subset = results[(results.Group == group) & (results.Gradient == grad)].set_index("Metric")
        for mi, metric in enumerate(METRIC_NAMES):
            row = subset.loc[metric]
            fig, ax = plt.subplots(figsize=SIG_FIGSIZE)
            draw_large_scatter(ax, stats.rankdata(baseline[mi]), y,
                SIG_METRIC_LABELS[metric], f"{group} ({grad})", row.rho,
                row.P_spin, row.q_spin_FDR, N_ROI, len(individual[group]), "Paired t rank", warm_palette=group == "TMS")
            fig.tight_layout()
            save_figure(fig, out / "scatter" / f"Figure2d_{group}_{grad}_{metric}")
            plt.close(fig)
    print("[PLOTS] All cells and all 32 scatter plots exported; no significance-based selection", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=SOURCE_CSV)
    p.add_argument("--fc-root", type=Path, default=FC_ROOT)
    p.add_argument("--gradient-root", type=Path, default=GRADIENT_ROOT)
    p.add_argument("--labels", type=Path, default=LABELS_FILE)
    p.add_argument("--roi-mapping", type=Path, default=ROI_MAPPING)
    p.add_argument("--spin-indices", type=Path, default=SPIN_INDICES)
    p.add_argument("--out", type=Path, default=PLOT_OUTPUT_ROOT)
    p.add_argument("--workers", type=int, default=2, help="Parallel baseline-AUC subjects; cached after first run")
    p.add_argument("--bootstrap", type=int, default=1000, help="Whole-patient bootstrap replicates; 0 disables")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--no-scatter", action="store_true", help="Export all statistics and heatmaps, omit individual scatter images")
    p.add_argument("--validate-only", action="store_true", help="Validate cohort/maps/spin inputs, without recomputing baseline AUC")
    return p.parse_args()


def main():
    args = parse_args()
    if args.workers < 1 or args.bootstrap < 0 or (0 < args.bootstrap < 200):
        raise ValueError("workers >= 1; bootstrap must be 0 or at least 200")
    args.out.mkdir(parents=True, exist_ok=True)
    mapping, networks = load_mapping(args.roi_mapping)
    spins = validate_spins(np.load(args.spin_indices, allow_pickle=False))
    cohort, deltas, maps, participants = load_cohort_and_validate_maps(args)
    participants.to_csv(args.out / "Cohort_manifest.csv", index=False, encoding="utf-8-sig")
    mapping.to_csv(args.out / "ROI_mapping_used.csv", index=False, encoding="utf-8-sig")
    print(f"[INPUTS] {len(spins)} valid spin permutations; cohort and t-maps verified", flush=True)
    if args.validate_only:
        print("[DONE] Input validation only. No association results computed.", flush=True)
        return
    individual = baseline_auc(cohort, networks, args.out, args.workers)
    export_roi_data(mapping, maps, individual, args.out)
    primary = analyze_maps(maps, individual, spins, "t", args.out)
    sensitivity = analyze_maps(maps, individual, spins, "mean_delta", args.out)
    if args.bootstrap:
        ci = subject_bootstrap(individual, deltas, args.bootstrap, args.seed, args.out)
        primary = primary.merge(ci, on=["Group", "Gradient", "Metric"], how="left", validate="one_to_one")
    primary.to_csv(args.out / "Figure2d_primary_tmap_spatial_correlations.csv", index=False, encoding="utf-8-sig")
    sensitivity.to_csv(args.out / "Figure2d_sensitivity_mean_delta.csv", index=False, encoding="utf-8-sig")
    write_methods(args, primary, args.out)
    plot_results(primary, maps, individual, args.out, scatter=not args.no_scatter)
    print(f"[DONE] {len(primary)} primary tests; {int(primary.FDR_significant.sum())} q_spin < .05; {args.out}", flush=True)


if __name__ == "__main__":
    main()
