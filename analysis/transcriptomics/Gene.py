# -*- coding: utf-8 -*-
import os
import re
import glob
import warnings
from pathlib import Path
import inspect

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import ttest_ind, norm, pearsonr
from statsmodels.stats.multitest import multipletests
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

import nibabel as nib
from nilearn import datasets
import abagen

try:
    import gseapy as gp
    HAS_GSEAPY = True
except ImportError:
    HAS_GSEAPY = False


# =========================
# Compatibility patches
# =========================
if not hasattr(pd.DataFrame, "append"):
    def _append(self, other, ignore_index=False, **kwargs):
        return pd.concat([self, other], ignore_index=ignore_index)
    pd.DataFrame.append = _append

if "inplace" not in pd.DataFrame.set_axis.__code__.co_varnames:
    _orig_set_axis = pd.DataFrame.set_axis

    def _set_axis(self, labels, axis=0, inplace=None):
        out = _orig_set_axis(self, labels, axis=axis)
        if inplace:
            self.__dict__.update(out.__dict__)
            return None
        return out
    pd.DataFrame.set_axis = _set_axis


# =========================
# Global config：组间变化量分析（仅 Gradient 1）
# =========================
OUT_DIR = Path(r"F:\lishiqi\组间比较\基因\gene")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_ROOT = Path(r"F:\lishiqi\预处理数据\cortex\顶点级duiqi转换（400，5）")

GROUP_DIRS = {
    "Control": {
        "after_dir": str(DATA_ROOT / "Control" / "after"),
        "before_dir": str(DATA_ROOT / "Control" / "before"),
    },
    "TMS": {
        "after_dir": str(DATA_ROOT / "TMS" / "after"),
        "before_dir": str(DATA_ROOT / "TMS" / "before"),
    },
}

# 统计方向：TMS (Post - Pre) - Control (Post - Pre)
PREFIX = "TMS_change_vs_Control_change_G1"

LH_LABEL = r"F:\lishiqi\新图\Figure 7\lh.Schaefer2018_400Parcels_7Networks_order.label.gii"
RH_LABEL = r"F:\lishiqi\新图\Figure 7\rh.Schaefer2018_400Parcels_7Networks_order.label.gii"

# 你现在没有这些外部文件，所以默认跳过这几块分析
GENESET_GMT = ""          # 若以后要做 GSEA，再填 .gmt 路径
HANSEN_RECEPTOR_CSV = ""  # 若以后有 receptor csv，再填路径
CELLTYPE_PROFILE_CSV = "" # 若以后有 cell-type csv，再填路径
NEUROSYNTH_TERMS_CSV = "" # 若以后有 Neurosynth csv，再填路径

# 你当前也没有行为表，所以行为 PLS 默认跳过
RUN_BEHAVIOR_PLS = False
BEHAVIOR_CSV = ""  # 若以后有行为表，再填路径；至少含 Fundata,before,after,before1,after1,behav_group

GRAD_IDX = 0  # 仅 Gradient 1（Python 列索引 0）
N_PERM = 5000
N_BOOT_GENE = 1000
RANDOM_STATE = 0


# =========================
# Behavioral matrix（当前默认不用）
# =========================
def build_behavior_df():
    """
    若后续你要恢复行为 PLS：
    需要提供一个 csv，至少包含：
    Fundata, before, after, before1, after1, behav_group
    """
    if not BEHAVIOR_CSV or not os.path.isfile(BEHAVIOR_CSV):
        raise FileNotFoundError(
            "[BEHAV] 未提供行为表。若要运行行为 PLS，请设置 BEHAVIOR_CSV。"
        )

    behav = pd.read_csv(BEHAVIOR_CSV)
    need_cols = ["Fundata", "before", "after", "before1", "after1", "behav_group"]
    miss = [c for c in need_cols if c not in behav.columns]
    if miss:
        raise ValueError(f"[BEHAV] 行为表缺少列: {miss}")

    behav["Fundata"] = behav["Fundata"].astype(str).str.strip().str.lower()
    behav["delta"] = behav["after"] - behav["before"]
    behav["delta1"] = behav["after1"] - behav["before1"]
    return behav


# =========================
# Utils
# =========================
def subject_id_from_path(fpath: str) -> str:
    """
    原代码是匹配 sub\\d+。
    你现在的文件名形如：
    PRIVATE_SUBJECT_022-PRIVATE_SUBJECT_046-1_aligned5_masked.npy
    这里为了后续兼容，优先取：
    1) sub\\d+
    2) 第一个数字块（如 PRIVATE_SUBJECT_022）
    """
    base = os.path.basename(fpath)

    m = re.search(r"(sub\d+)", base, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower()

    m2 = re.match(r"(\d+)-", base)
    if m2:
        return m2.group(1).lower()

    stem = Path(base).stem
    return stem.lower()


def coerce_index_to_int(idx: pd.Index) -> pd.Index:
    try:
        return idx.astype(int)
    except Exception:
        def grab_int(name):
            m = re.search(r"(\d+)$", str(name))
            return int(m.group(1)) if m else None

        vals = [grab_int(x) for x in idx]
        if any(v is None for v in vals):
            raise ValueError("AHBA parcel ID parse failed")
        return pd.Index(vals, name="parcel")


def _zscore_2d(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, float)
    mu = np.nanmean(A, axis=0, keepdims=True)
    sd = np.nanstd(A, axis=0, ddof=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (A - mu) / sd


def _make_spin(n_perm, random_state=0, verbose=False):
    from brainspace.null_models import SpinPermutations
    sig = inspect.signature(SpinPermutations.__init__).parameters
    kwargs = {}
    for k in ("n_rot", "n_perm", "n_rotate", "n_rotations", "n_rep"):
        if k in sig:
            kwargs[k] = n_perm
            break
    if "random_state" in sig:
        kwargs["random_state"] = random_state
    if "verbose" in sig:
        kwargs["verbose"] = verbose
    return SpinPermutations(**kwargs) if kwargs else SpinPermutations(n_perm)


# =========================
# AHBA
# =========================
def load_ahba_schaefer400():
    assert os.path.isfile(LH_LABEL), f"{LH_LABEL}"
    assert os.path.isfile(RH_LABEL), f"{RH_LABEL}"

    atlas = (LH_LABEL, RH_LABEL)

    expr = abagen.get_expression_data(
        atlas=atlas,
        lr_mirror='bidirectional',
        probe_selection='diff_stability',
        sample_norm='srs',
        gene_norm='srs',
        agg_metric='mean',
        ibf_threshold=0.5,
        missing=None
    )
    expr.index = coerce_index_to_int(expr.index)
    expr = expr.loc[(expr.index >= 1) & (expr.index <= 400)].copy()
    expr = expr[~expr.index.duplicated(keep='first')].sort_index()

    nan_parcel = np.isnan(expr.values).any(axis=1)
    good_parcels = expr.index[~nan_parcel].astype(int).tolist()
    bad_parcels = expr.index[nan_parcel].astype(int).tolist()

    if bad_parcels:
        print(f"[AHBA] NaN parcels: {len(bad_parcels)} {bad_parcels[:10]}{'...' if len(bad_parcels) > 10 else ''}")

    expr = expr.loc[good_parcels]
    print(f"[AHBA] valid parcels = {len(good_parcels)} / 400")
    return expr


def save_ahba_outputs(expr, out_dir, prefix):
    expr_path = Path(out_dir, f"{prefix}_ahba_expr_schaefer400.csv")
    expr.to_csv(expr_path)
    Path(out_dir, f"{prefix}_ahba_valid_parcels.txt").write_text(
        "\n".join(map(str, expr.index.astype(int).tolist())),
        encoding="utf-8"
    )
    print(f"[AHBA] saved: {expr_path}")


# =========================
# Parcellation and imaging
# =========================
def load_parcellation_labels():
    lh = nib.load(LH_LABEL).darrays[0].data.astype(int)
    rh = nib.load(RH_LABEL).darrays[0].data.astype(int)
    return lh, rh


def build_vertices_per_parcel(lh_lab, rh_lab):
    verts_l = {p: np.where(lh_lab == p)[0] for p in np.unique(lh_lab) if p > 0}
    verts_r = {p: np.where(rh_lab == p)[0] for p in np.unique(rh_lab) if p > 0}

    vpp = {p: [] for p in range(1, 401)}
    offset = len(lh_lab)

    for p, idx in verts_l.items():
        vpp[p].extend(idx.tolist())

    for p, idx in verts_r.items():
        vpp[p].extend((idx + offset).tolist())

    for p in range(1, 401):
        vpp.setdefault(p, [])

    return vpp, len(lh_lab) + len(rh_lab)


def list_npy(dirpath):
    return sorted(glob.glob(os.path.join(dirpath, "*.npy")))


def subject_g1_parcel_mean(fpath, grad_idx, parcellation=None, n_vertices_total=None):
    arr = np.load(fpath)
    if arr.ndim != 2 or arr.shape[1] <= grad_idx:
        raise ValueError(f"{os.path.basename(fpath)} shape invalid: {arr.shape}")

    X = arr[:, grad_idx]

    if arr.shape[0] == 400:
        return X.astype(float)
    else:
        if parcellation is None or n_vertices_total is None:
            raise RuntimeError("parcellation is required for vertex-wise input")
        if X.shape[0] != n_vertices_total:
            raise ValueError(f"{os.path.basename(fpath)} vertex number mismatch: {X.shape[0]} vs {n_vertices_total}")

        g1_parcel = np.full(400, np.nan, dtype=float)
        for p, vidx in parcellation.items():
            if not vidx:
                continue
            g1_parcel[p - 1] = np.nanmean(X[vidx])
        return g1_parcel


def load_group_parcel_matrix(dirpath, grad_idx, parcellation=None, n_vertices_total=None):
    files = list_npy(dirpath)
    if not files:
        raise RuntimeError(f"no .npy files found in {dirpath}")

    mats = []
    kept = []

    for f in files:
        try:
            g1 = subject_g1_parcel_mean(f, grad_idx, parcellation, n_vertices_total)
            if g1.shape[0] != 400:
                raise ValueError("parcel number is not 400")
            mats.append(g1.astype(float))
            kept.append(f)
        except Exception as e:
            print(f"[SKIP] {os.path.basename(f)}: {e}")

    if not mats:
        raise RuntimeError(f"no valid files in {dirpath}")

    M = np.vstack(mats)
    print(f"[LOAD] {dirpath}: kept={len(kept)}/{len(files)}; shape={M.shape}")
    return M, kept


def build_subject_file_map(dirpath):
    """Map subject IDs to files and reject ambiguous duplicate IDs."""
    files = list_npy(dirpath)
    if not files:
        raise RuntimeError(f"no .npy files found in {dirpath}")

    file_map = {}
    for fpath in files:
        sid = subject_id_from_path(fpath)
        if sid in file_map:
            raise RuntimeError(
                f"duplicate subject ID '{sid}' in {dirpath}:\n"
                f"  {file_map[sid]}\n"
                f"  {fpath}"
            )
        file_map[sid] = fpath
    return file_map


def load_paired_change_matrix(group_name, grad_idx, parcellation=None, n_vertices_total=None):
    """
    Match Pre/Post files within one group by subject ID, then calculate
    one 400-parcel change vector per subject: Post - Pre.
    """
    cfg = GROUP_DIRS[group_name]
    before_map = build_subject_file_map(cfg["before_dir"])
    after_map = build_subject_file_map(cfg["after_dir"])

    common_ids = sorted(set(before_map) & set(after_map))
    only_before = sorted(set(before_map) - set(after_map))
    only_after = sorted(set(after_map) - set(before_map))

    if only_before:
        print(f"[WARN] {group_name}: only Pre, excluded n={len(only_before)}: {only_before[:10]}")
    if only_after:
        print(f"[WARN] {group_name}: only Post, excluded n={len(only_after)}: {only_after[:10]}")
    if not common_ids:
        raise RuntimeError(f"no matched Pre/Post subjects for {group_name}")

    changes = []
    kept_ids = []
    for sid in common_ids:
        try:
            before = subject_g1_parcel_mean(
                before_map[sid], grad_idx, parcellation, n_vertices_total
            )
            after = subject_g1_parcel_mean(
                after_map[sid], grad_idx, parcellation, n_vertices_total
            )
            changes.append(after.astype(float) - before.astype(float))
            kept_ids.append(sid)
        except Exception as e:
            print(f"[SKIP] {group_name} subject {sid}: {e}")

    if not changes:
        raise RuntimeError(f"no valid matched Pre/Post subjects for {group_name}")

    change_mat = np.vstack(changes)
    print(
        f"[CHANGE] {group_name}: matched/kept={len(kept_ids)}/{len(common_ids)}; "
        f"shape={change_mat.shape}; definition=Post-Pre"
    )
    return change_mat, kept_ids


# =========================
# Group diff
# =========================
def t_to_z(tvals, df):
    p = 2 * norm.sf(np.abs(tvals))
    z = np.sign(tvals) * norm.isf(p / 2.0)
    return z, p


def group_diff_to_z(group1_mat, group2_mat):
    n_parcels = group1_mat.shape[1]
    t = np.zeros(n_parcels, dtype=float)
    p = np.ones(n_parcels, dtype=float)

    for i in range(n_parcels):
        a = group1_mat[:, i]
        b = group2_mat[:, i]
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]

        if len(a) < 2 or len(b) < 2:
            t[i] = 0.0
            p[i] = 1.0
            continue

        tt, pp = ttest_ind(a, b, equal_var=False, nan_policy='omit')
        t[i] = tt
        p[i] = pp

    z = np.sign(t) * norm.isf(p / 2.0)
    return t, z, p


# =========================
# PLS: z-map vs AHBA
# =========================
def run_pls_onecomp(z_vec, expr_df):
    parcels_expr = expr_df.index.astype(int).tolist()
    parcels_use = np.array(parcels_expr, dtype=int)

    X = z_vec[parcels_use - 1].reshape(-1, 1)
    Y = expr_df.values

    Xs = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
    Ys = StandardScaler(with_mean=True, with_std=True).fit_transform(Y)

    pls = PLSRegression(n_components=1)
    pls.fit(Xs, Ys)

    svals = np.linalg.svd(Xs.T @ Ys, full_matrices=False, compute_uv=False)
    s1 = float(svals[0])

    gene_weights = pls.y_weights_.ravel()
    brain_scores = pls.x_scores_.ravel()

    genes = expr_df.columns.tolist()
    order = np.argsort(-gene_weights)
    ranking = pd.DataFrame({
        "gene": np.array(genes)[order],
        "weight": gene_weights[order]
    })

    return {
        "X_used": X.squeeze(),
        "Y_used": Y,
        "Xs": Xs.squeeze(),
        "Ys": Ys,
        "pls": pls,
        "s1": s1,
        "gene_weights": gene_weights,
        "brain_scores": brain_scores,
        "ranking": ranking
    }


def bootstrap_pls_gene_significance(z_vec, expr_df, n_boot=1000, random_state=0):
    rng = np.random.RandomState(random_state)

    parcels = expr_df.index.astype(int).to_numpy()
    X = z_vec[parcels - 1].reshape(-1, 1)
    Y = expr_df.values

    Xs = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
    Ys = StandardScaler(with_mean=True, with_std=True).fit_transform(Y)

    pls = PLSRegression(n_components=1)
    pls.fit(Xs, Ys)
    w_obs = pls.y_weights_.ravel()

    n_parcels, n_genes = Y.shape
    boot_w = np.full((n_boot, n_genes), np.nan, dtype=float)

    for b in range(n_boot):
        idx = rng.randint(0, n_parcels, size=n_parcels)
        Xb = Xs[idx]
        Yb = Ys[idx]
        try:
            pls_b = PLSRegression(n_components=1)
            pls_b.fit(Xb, Yb)
            boot_w[b, :] = pls_b.y_weights_.ravel()
        except Exception as e:
            print(f"[BOOT] iteration {b} failed: {e}")

    se = np.nanstd(boot_w, axis=0, ddof=1)
    se[se == 0] = np.nan

    z_boot = w_obs / se
    p_boot = 2 * norm.sf(np.abs(z_boot))

    valid = np.isfinite(p_boot)
    q_boot = np.full_like(p_boot, np.nan, dtype=float)
    if valid.any():
        _, q_tmp, _, _ = multipletests(p_boot[valid], alpha=0.05, method="fdr_bh")
        q_boot[valid] = q_tmp

    genes = expr_df.columns.to_numpy()
    df = pd.DataFrame({
        "gene": genes,
        "weight": w_obs,
        "z_boot": z_boot,
        "p_boot": p_boot,
        "q_boot": q_boot
    })
    df["signif_fdr05"] = df["q_boot"] < 0.05
    return df


# =========================
# Enrichment
# =========================
def run_gene_enrichment(pls_gene_df, out_dir, prefix, geneset_gmt=None):
    if geneset_gmt is None:
        geneset_gmt = GENESET_GMT

    if not geneset_gmt:
        print("[GSEA] skipped: GENESET_GMT is empty")
        return

    if not os.path.isfile(geneset_gmt):
        print(f"[GSEA] skipped: file not found -> {geneset_gmt}")
        return

    if not HAS_GSEAPY:
        print("[GSEA] skipped: gseapy not installed")
        return

    rnk = pls_gene_df[["gene", "weight"]].copy()
    rnk_path = Path(out_dir, f"{prefix}_PLS1_prerank.rnk")
    rnk.to_csv(rnk_path, sep="\t", index=False, header=False)

    outdir_gsea = Path(out_dir, f"{prefix}_GSEA")
    outdir_gsea.mkdir(parents=True, exist_ok=True)

    print(f"[GSEA] running prerank with {geneset_gmt}")
    enr = gp.prerank(
        rnk=str(rnk_path),
        gene_sets=geneset_gmt,
        outdir=str(outdir_gsea),
        permutation_num=1000,
        seed=RANDOM_STATE,
        threads=4,
        verbose=True
    )

    try:
        res = enr.res2d
        res_path = Path(out_dir, f"{prefix}_GSEA_results.csv")
        res.to_csv(res_path)
        print(f"[GSEA] saved: {res_path}")
    except Exception as e:
        print(f"[GSEA] failed to save results: {e}")


# =========================
# Receptor / cell-type / Neurosynth
# =========================
def correlate_brain_map_with_feature_table(brain_map, feature_df, parcels_col=None, feature_label="feature"):
    if parcels_col is not None and parcels_col in feature_df.columns:
        df = feature_df.set_index(parcels_col).copy()
    else:
        df = feature_df.copy()

    try:
        parcels = df.index.to_numpy().astype(int)
    except Exception as e:
        raise ValueError(f"feature_df index cannot be converted to parcel ID: {e}")

    names = []
    rs = []
    ps = []
    ns = []

    for col in df.columns:
        col_vals = pd.to_numeric(df[col], errors="coerce").to_numpy()
        vals_brain = brain_map[parcels - 1]
        mask = np.isfinite(vals_brain) & np.isfinite(col_vals)
        n = int(mask.sum())

        if n < 5:
            r = np.nan
            p = np.nan
        else:
            r, p = pearsonr(vals_brain[mask], col_vals[mask])

        names.append(col)
        rs.append(r)
        ps.append(p)
        ns.append(n)

    pvals = np.array(ps, dtype=float)
    qvals = np.full_like(pvals, np.nan, dtype=float)
    valid = np.isfinite(pvals)
    if valid.any():
        _, q_tmp, _, _ = multipletests(pvals[valid], alpha=0.05, method="fdr_bh")
        qvals[valid] = q_tmp

    out_df = pd.DataFrame({
        feature_label: names,
        "r": rs,
        "p": pvals,
        "q": qvals,
        "n_parcels": ns
    })
    out_df["signif_fdr05"] = out_df["q"] < 0.05
    return out_df


def load_hansen_receptors():
    f = HANSEN_RECEPTOR_CSV
    if not os.path.isfile(f):
        raise FileNotFoundError(f"[HANSEN] file not found: {f}")

    df = pd.read_csv(f)
    print(f"[HANSEN] raw shape: {df.shape}")

    if "receptor" not in df.columns:
        raise ValueError("[HANSEN] CSV missing 'receptor' column")

    df = df.set_index("receptor")
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.T

    def _parse_parcel(name):
        m = re.search(r"(\d+)$", str(name))
        if not m:
            raise ValueError(f"[HANSEN] parcel parse failed: {name}")
        return int(m.group(1))

    df.index = [_parse_parcel(ix) for ix in df.index]
    df.index.name = "parcel"
    df = df.sort_index()

    print(f"[HANSEN] processed shape: {df.shape}")
    return df


def load_celltype_profile():
    f = CELLTYPE_PROFILE_CSV
    if not os.path.isfile(f):
        raise FileNotFoundError(f"[CELLTYPE] file not found: {f}")

    df = pd.read_csv(f)
    print(f"[CELLTYPE] raw shape: {df.shape}")

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all")
    print(f"[CELLTYPE] after drop empty columns: {df.shape}")

    n_ct, _ = df.shape
    df.index = [f"celltype_{i+1}" for i in range(n_ct)]
    df = df.T
    df.index = np.arange(1, df.shape[0] + 1)

    print(f"[CELLTYPE] processed shape: {df.shape}")
    return df


# =========================
# Spin permutation
# =========================
def build_vertex_map_from_parcel_values(parcel_values, lh_lab, rh_lab):
    nl, nr = len(lh_lab), len(rh_lab)
    v = np.full(nl + nr, np.nan, dtype=float)

    # The two GIFTI label files both use local labels 1..200.  AHBA/Schaefer
    # parcel values use global order: LH 1..200 followed by RH 201..400.
    left_labels = sorted(int(x) for x in np.unique(lh_lab) if x > 0)
    right_labels = sorted(int(x) for x in np.unique(rh_lab) if x > 0)
    if len(left_labels) + len(right_labels) != len(parcel_values):
        raise ValueError(
            "Parcellation/value size mismatch: "
            f"LH={len(left_labels)}, RH={len(right_labels)}, values={len(parcel_values)}"
        )

    for global_idx, local_label in enumerate(left_labels):
        val = parcel_values[global_idx]
        if np.isfinite(val):
            v[np.where(lh_lab == local_label)[0]] = val

    right_offset = len(left_labels)
    for local_idx, local_label in enumerate(right_labels):
        val = parcel_values[right_offset + local_idx]
        if np.isfinite(val):
            v[np.where(rh_lab == local_label)[0] + nl] = val

    return v


def spin_permutation_pval(z_vec, expr_df, lh_lab, rh_lab, n_perm=5000, random_state=0):
    fs5 = datasets.fetch_surf_fsaverage("fsaverage5")
    coords_l = nib.load(fs5.sphere_left).darrays[0].data
    coords_r = nib.load(fs5.sphere_right).darrays[0].data

    left_labels = sorted(int(x) for x in np.unique(lh_lab) if x > 0)
    right_labels = sorted(int(x) for x in np.unique(rh_lab) if x > 0)
    if len(left_labels) + len(right_labels) != 400:
        raise ValueError(
            f"Expected 400 parcels, got LH={len(left_labels)} and RH={len(right_labels)}"
        )

    def parcel_centroids(coords, labels, ordered_labels):
        centroids = np.vstack([np.mean(coords[labels == lab], axis=0) for lab in ordered_labels])
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        if np.any(norms == 0) or not np.isfinite(norms).all():
            raise RuntimeError("Invalid parcel centroid encountered during spin setup")
        return centroids / norms

    # Spin the 200 parcel centroids in each hemisphere.  This avoids rotating
    # medial-wall NaNs into cortical parcels and preserves the LH/RH ordering
    # used by the Schaefer-400 AHBA expression matrix.
    centroid_l = parcel_centroids(coords_l, lh_lab, left_labels)
    centroid_r = parcel_centroids(coords_r, rh_lab, right_labels)
    n_left = len(left_labels)
    values_l = np.asarray(z_vec[:n_left], dtype=float)
    values_r = np.asarray(z_vec[n_left:n_left + len(right_labels)], dtype=float)

    rot = _make_spin(n_perm=n_perm, random_state=random_state, verbose=False)
    rot.fit(centroid_l, centroid_r)

    perm = rot.randomize(values_l, values_r)
    if isinstance(perm, tuple):
        permL, permR = perm
    else:
        raise RuntimeError("Unexpected return from SpinPermutations.randomize")

    if permL.shape[0] != n_perm and permL.shape[1] == n_perm:
        permL = permL.T
        permR = permR.T

    parcels_use = expr_df.index.astype(int).to_numpy()
    Y = expr_df.values
    y_mu = np.nanmean(Y, axis=0)
    y_sd = np.nanstd(Y, axis=0, ddof=1)
    y_sd[(~np.isfinite(y_sd)) | (y_sd == 0)] = 1.0
    Ys = (Y - y_mu) / y_sd

    surr_s1 = np.full(n_perm, np.nan, dtype=float)
    for k in range(n_perm):
        z_perm = np.concatenate([permL[k], permR[k]], axis=0)
        Xp = z_perm[parcels_use - 1].reshape(-1, 1)
        xp_sd = np.nanstd(Xp, axis=0, ddof=1)
        if (not np.isfinite(xp_sd).all()) or np.any(xp_sd == 0):
            continue
        Xps = (Xp - np.nanmean(Xp, axis=0)) / xp_sd

        if not (np.isfinite(Xps).all() and np.isfinite(Ys).all()):
            continue

        surr_s1[k] = float(np.linalg.svd(Xps.T @ Ys, full_matrices=False, compute_uv=False)[0])

    valid_surr = surr_s1[np.isfinite(surr_s1)]
    if valid_surr.size == 0:
        raise RuntimeError("All spin permutations were invalid; check hemisphere parcel labels")
    if valid_surr.size < n_perm:
        print(f"[SPIN] valid permutations={valid_surr.size}/{n_perm}")
    return valid_surr


# =========================
# Behavioral PLS（当前默认跳过）
# =========================
def pls_svd_covariance(X: np.ndarray, Y: np.ndarray, n_components=None):
    Xs = _zscore_2d(X)
    Ys = _zscore_2d(Y)

    C = (Xs.T @ Ys) / (Xs.shape[0] - 1)
    U, S, Vt = np.linalg.svd(C, full_matrices=False)
    V = Vt.T

    if n_components is None:
        n_components = min(U.shape[1], V.shape[1])
    n_components = int(n_components)

    U = U[:, :n_components]
    V = V[:, :n_components]
    S = S[:n_components]

    T = Xs @ U
    UY = Ys @ V
    return {"U": U, "V": V, "S": S, "T": T, "UY": UY, "Xs": Xs, "Ys": Ys}


def pls_permutation_test(X: np.ndarray, Y: np.ndarray, n_perm=5000, n_components=None, random_state=0):
    rng = np.random.RandomState(random_state)
    base = pls_svd_covariance(X, Y, n_components=n_components)
    S_obs = base["S"].copy()
    K = len(S_obs)

    S_null = np.zeros((n_perm, K), dtype=float)
    for i in range(n_perm):
        idx = rng.permutation(Y.shape[0])
        res = pls_svd_covariance(X, Y[idx], n_components=K)
        S_null[i, :] = res["S"]

    pvals = np.array([(np.sum(S_null[:, k] >= S_obs[k]) + 1) / (n_perm + 1) for k in range(K)], dtype=float)
    return S_obs, S_null, pvals


def pls_bootstrap_saliences(X: np.ndarray, Y: np.ndarray, U_ref: np.ndarray, V_ref: np.ndarray,
                            n_boot=1000, random_state=0):
    rng = np.random.RandomState(random_state)
    n_subj = X.shape[0]
    K = U_ref.shape[1]
    n_parc = U_ref.shape[0]
    n_behv = V_ref.shape[0]

    Ub = np.zeros((n_boot, n_parc, K), dtype=float)
    Vb = np.zeros((n_boot, n_behv, K), dtype=float)

    for b in range(n_boot):
        samp = rng.randint(0, n_subj, size=n_subj)
        res = pls_svd_covariance(X[samp], Y[samp], n_components=K)
        U_b = res["U"]
        V_b = res["V"]

        for k in range(K):
            r = np.corrcoef(U_b[:, k], U_ref[:, k])[0, 1]
            if np.isfinite(r) and r < 0:
                U_b[:, k] *= -1
                V_b[:, k] *= -1

        Ub[b] = U_b
        Vb[b] = V_b

    U_se = np.std(Ub, axis=0, ddof=1)
    V_se = np.std(Vb, axis=0, ddof=1)
    U_se[U_se == 0] = np.nan
    V_se[V_se == 0] = np.nan
    return Ub, Vb, U_se, V_se


def salience_stats_from_se(W: np.ndarray, SE: np.ndarray, alpha=0.05, method="fdr_bh"):
    Z = W / SE
    P = 2 * norm.sf(np.abs(Z))

    Q = np.full_like(P, np.nan, dtype=float)
    for k in range(P.shape[1]):
        valid = np.isfinite(P[:, k])
        if valid.any():
            _, qk, _, _ = multipletests(P[valid, k], alpha=alpha, method=method)
            Q[valid, k] = qk
    return Z, P, Q


def corr_with_composite(scores: np.ndarray, data: np.ndarray, var_names, comp_prefix="LC"):
    out = []
    K = scores.shape[1]

    for k in range(K):
        sk = scores[:, k]
        for j in range(data.shape[1]):
            x = data[:, j]
            m = np.isfinite(sk) & np.isfinite(x)
            if m.sum() < 5:
                r, p = np.nan, np.nan
            else:
                r, p = pearsonr(sk[m], x[m])
            out.append({
                "var": var_names[j],
                "component": f"{comp_prefix}{k+1}",
                "r": r,
                "p": p,
                "n": int(m.sum())
            })

    df = pd.DataFrame(out)
    df["q"] = np.nan

    for comp in df["component"].unique():
        idx = df["component"] == comp
        pvals = df.loc[idx, "p"].to_numpy(dtype=float)
        valid = np.isfinite(pvals)
        if valid.any():
            _, qv, _, _ = multipletests(pvals[valid], alpha=0.05, method="fdr_bh")
            sub_idx = df.index[idx][valid]
            df.loc[sub_idx, "q"] = qv

    return df


def run_behavior_pls_pipeline(cn_mat, cn_files, ad_mat, ad_files, out_dir, prefix,
                              behav_cols=("before", "after", "before1", "after1"),
                              n_perm=5000, n_boot=1000, random_state=0,
                              group_labels=("after", "before")):
    behav = build_behavior_df()
    behav_path = Path(out_dir, f"{prefix}_behavior_raw.csv")
    behav.to_csv(behav_path, index=False)

    cn_sub = [subject_id_from_path(f) for f in cn_files]
    ad_sub = [subject_id_from_path(f) for f in ad_files]

    cn_df = pd.DataFrame({"Fundata": cn_sub, "img_group": group_labels[0]})
    ad_df = pd.DataFrame({"Fundata": ad_sub, "img_group": group_labels[1]})

    X = np.vstack([cn_mat, ad_mat])
    img_meta = pd.concat([cn_df, ad_df], ignore_index=True)
    img_meta["Fundata"] = img_meta["Fundata"].astype(str).str.lower()
    img_meta["img_row"] = np.arange(len(img_meta))

    ok = img_meta["Fundata"].notna() & (img_meta["Fundata"] != "none")
    X = X[ok.values]
    img_meta = img_meta.loc[ok].reset_index(drop=True)
    img_meta["img_row"] = np.arange(len(img_meta))

    merged = img_meta.merge(behav, on="Fundata", how="inner")
    if merged.shape[0] < 6:
        raise RuntimeError(f"[BEHAV-PLS] matched subjects too few: {merged.shape[0]}")

    idx = merged["img_row"].to_numpy()
    X_use = X[idx, :]
    Y_use = merged.loc[:, list(behav_cols)].to_numpy(dtype=float)

    merged_path = Path(out_dir, f"{prefix}_behaviorPLS_matched_subjects.csv")
    merged.to_csv(merged_path, index=False)

    K = min(Y_use.shape[1], 10)
    pls = pls_svd_covariance(X_use, Y_use, n_components=K)

    S_obs, S_null, p_perm = pls_permutation_test(
        X_use, Y_use, n_perm=n_perm, n_components=K, random_state=random_state
    )

    perm_df = pd.DataFrame({
        "component": [f"LC{i+1}" for i in range(K)],
        "singular_value_S": S_obs,
        "p_perm": p_perm
    })
    perm_df["signif_p05"] = perm_df["p_perm"] < 0.05
    perm_df.to_csv(Path(out_dir, f"{prefix}_behaviorPLS_permutation.csv"), index=False)
    np.save(Path(out_dir, f"{prefix}_behaviorPLS_perm_null_S.npy"), S_null)

    Ub, Vb, U_se, V_se = pls_bootstrap_saliences(
        X_use, Y_use,
        U_ref=pls["U"],
        V_ref=pls["V"],
        n_boot=n_boot,
        random_state=random_state
    )
    U_Z, U_P, U_Q = salience_stats_from_se(pls["U"], U_se, method="fdr_bh")
    V_Z, V_P, V_Q = salience_stats_from_se(pls["V"], V_se, method="fdr_bh")

    brain_cols = [f"LC{i+1}" for i in range(K)]

    brain_sal = pd.DataFrame(pls["U"], columns=brain_cols)
    brain_sal.insert(0, "parcel", np.arange(1, pls["U"].shape[0] + 1))
    for i, c in enumerate(brain_cols):
        brain_sal[f"{c}_Z"] = U_Z[:, i]
        brain_sal[f"{c}_p"] = U_P[:, i]
        brain_sal[f"{c}_q"] = U_Q[:, i]
    brain_sal.to_csv(Path(out_dir, f"{prefix}_behaviorPLS_SFC_saliences.csv"), index=False)

    beh_names = list(behav_cols)
    beh_sal = pd.DataFrame(pls["V"], columns=brain_cols)
    beh_sal.insert(0, "behavior", beh_names)
    for i, c in enumerate(brain_cols):
        beh_sal[f"{c}_Z"] = V_Z[:, i]
        beh_sal[f"{c}_p"] = V_P[:, i]
        beh_sal[f"{c}_q"] = V_Q[:, i]
    beh_sal.to_csv(Path(out_dir, f"{prefix}_behaviorPLS_behavioral_saliences.csv"), index=False)

    score_df = merged[["Fundata", "img_group", "behav_group"]].copy()
    for i, c in enumerate(brain_cols):
        score_df[f"{c}_SFC_score"] = pls["T"][:, i]
        score_df[f"{c}_BEH_score"] = pls["UY"][:, i]
    score_df.to_csv(Path(out_dir, f"{prefix}_behaviorPLS_composite_scores.csv"), index=False)

    parcel_names = [f"parcel_{i}" for i in range(1, X_use.shape[1] + 1)]
    df_corr_brain = corr_with_composite(pls["T"], X_use, parcel_names, comp_prefix="LC")
    df_corr_brain.to_csv(Path(out_dir, f"{prefix}_behaviorPLS_explain_corr_SFC.csv"), index=False)

    df_corr_beh = corr_with_composite(pls["UY"], Y_use, beh_names, comp_prefix="LC")
    df_corr_beh.to_csv(Path(out_dir, f"{prefix}_behaviorPLS_explain_corr_behavior.csv"), index=False)

    print(f"[BEHAV-PLS] matched N={X_use.shape[0]}, parcels={X_use.shape[1]}, behaviors={Y_use.shape[1]}, K={K}")
    print(f"[BEHAV-PLS] permutation p-values: {p_perm}")
    return perm_df


# =========================
# One-group pipeline
# =========================
def plot_top_absZ_gt3_gene_expression_heatmap(expr_df, pls_gene_stats, z_map, out_dir, prefix,
                                              z_thresh=3.0, top_n=10, use_half_brain=True):

    need_cols = {"gene", "z_boot"}
    if not need_cols.issubset(pls_gene_stats.columns):
        raise ValueError(f"pls_gene_stats must contain columns: {need_cols}")

    sig_df = pls_gene_stats.loc[np.abs(pls_gene_stats["z_boot"]) > z_thresh, ["gene", "z_boot"]].copy()
    if sig_df.empty:
        print(f"[HEATMAP] {prefix}: no genes passed |Z| > {z_thresh}, skip.")
        return None

    sig_df["abs_z_boot"] = np.abs(sig_df["z_boot"])
    sig_df = sig_df.sort_values("abs_z_boot", ascending=False)

    selected_genes = []
    for g in sig_df["gene"].astype(str).tolist():
        if g in expr_df.columns and g not in selected_genes:
            selected_genes.append(g)
        if len(selected_genes) >= top_n:
            break

    if len(selected_genes) == 0:
        print(f"[HEATMAP] {prefix}: selected genes not found in expr_df.")
        return None

    parcels = expr_df.index.astype(int).to_numpy()

    if use_half_brain:
        half_mask = parcels <= 200
        if half_mask.sum() >= 5:
            parcels_use = parcels[half_mask]
        else:
            n_half = max(len(parcels) // 2, 1)
            parcels_use = parcels[:n_half]
    else:
        parcels_use = parcels

    heat_df = expr_df.loc[parcels_use, selected_genes].copy()

    parcel_z = z_map[parcels_use - 1]
    order_rows = np.argsort(parcel_z)
    heat_df = heat_df.iloc[order_rows, :]

    arr = heat_df.to_numpy(dtype=float)
    mu = np.nanmean(arr, axis=0, keepdims=True)
    sd = np.nanstd(arr, axis=0, ddof=1, keepdims=True)
    sd[sd == 0] = 1.0
    arr_z = (arr - mu) / sd

    heat_df_z = pd.DataFrame(arr_z, index=heat_df.index, columns=heat_df.columns)

    heat_csv = Path(out_dir, f"{prefix}_top{len(selected_genes)}_absZgt{int(z_thresh)}_halfbrain_expression_matrix.csv")
    heat_df_z.to_csv(heat_csv)

    n_rows, n_cols = heat_df_z.shape
    fig_w = max(6, 0.6 * n_cols + 2)
    fig_h = max(5, min(12, 0.06 * n_rows + 2.5))

    plt.figure(figsize=(fig_w, fig_h), dpi=300)
    im = plt.imshow(
        heat_df_z.values,
        aspect='auto',
        interpolation='nearest',
        cmap='bwr',
        vmin=-2.5,
        vmax=2.5
    )

    plt.title(f"{prefix}: top {len(selected_genes)} genes (|Z| > {z_thresh})", fontsize=11)
    plt.xlabel("Genes", fontsize=9)
    plt.ylabel("Half-brain parcels", fontsize=9)

    plt.xticks(np.arange(n_cols), heat_df_z.columns, rotation=90, fontsize=8)

    if n_rows <= 25:
        yticks = np.arange(n_rows)
        yticklabels = heat_df_z.index.astype(int).tolist()
    else:
        step = max(n_rows // 10, 1)
        yticks = np.arange(0, n_rows, step)
        yticklabels = [int(heat_df_z.index[i]) for i in yticks]

    plt.yticks(yticks, yticklabels, fontsize=7)

    cbar = plt.colorbar(im)
    cbar.set_label("Expression (gene-wise z-score)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    plt.tight_layout()
    out_png = Path(out_dir, f"{prefix}_top{len(selected_genes)}_absZgt{int(z_thresh)}_halfbrain_expression_heatmap.png")
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

    print(f"[HEATMAP] saved: {out_png}")
    print(f"[HEATMAP] matrix saved: {heat_csv}")
    print(f"[HEATMAP] selected genes: {selected_genes}")

    return out_png


def run_between_groups(expr, lh_lab, rh_lab, parcellation, n_vertices_total):
    """
    Build a Gradient-1 between-group change map using the required contrast:
        subject change = Post - Pre
        group contrast = TMS change - Control change

    The downstream AHBA/PLS1/bootstrap/spin logic is retained from the
    within-group version.
    """
    prefix = PREFIX
    grad_idx = GRAD_IDX
    grad_label = f"G{grad_idx + 1}"

    print("\n" + "=" * 90)
    print(f"[BETWEEN GROUPS] {prefix} | {grad_label}")
    print("[CONTRAST] TMS (Post - Pre) - Control (Post - Pre)")
    print("=" * 90)

    save_ahba_outputs(expr, OUT_DIR, prefix)
    parcels_use = expr.index.astype(int).tolist()

    print(f"[1/6] Match subjects and calculate Post - Pre ({grad_label}) ...")
    control_change, control_ids = load_paired_change_matrix(
        "Control", grad_idx, parcellation, n_vertices_total
    )
    tms_change, tms_ids = load_paired_change_matrix(
        "TMS", grad_idx, parcellation, n_vertices_total
    )
    print(f"[INFO] Control change N={len(control_ids)}; TMS change N={len(tms_ids)}")

    # Save the subject-level changes used by the statistical test.
    np.save(Path(OUT_DIR, f"{prefix}_Control_subject_changes.npy"), control_change)
    np.save(Path(OUT_DIR, f"{prefix}_TMS_subject_changes.npy"), tms_change)
    pd.DataFrame({"subject_id": control_ids}).to_csv(
        Path(OUT_DIR, f"{prefix}_Control_subject_ids.csv"), index=False
    )
    pd.DataFrame({"subject_id": tms_ids}).to_csv(
        Path(OUT_DIR, f"{prefix}_TMS_subject_ids.csv"), index=False
    )

    print("[2/6] Between-group Welch t: TMS change vs Control change -> signed z ...")
    # group_diff_to_z returns group1 - group2; therefore z > 0 means
    # TMS (Post - Pre) > Control (Post - Pre).
    t_map, z_map, p_map = group_diff_to_z(tms_change, control_change)

    df_all = pd.DataFrame({
        "parcel": np.arange(1, 401),
        "t": t_map,
        "z": z_map,
        "p": p_map,
        "contrast": "TMS(Post-Pre) - Control(Post-Pre)",
    })
    df_all.to_csv(Path(OUT_DIR, f"{prefix}_allparcels.csv"), index=False)

    df_use = df_all.iloc[np.array(parcels_use) - 1].copy()
    df_use.to_csv(Path(OUT_DIR, f"{prefix}_validparcels.csv"), index=False)

    print("[3/6] PLS 1-component: between-group z-map vs AHBA genes ...")
    pls_res = run_pls_onecomp(z_vec=z_map, expr_df=expr)
    z_use = z_map[np.array(parcels_use) - 1]

    pls_res["ranking"].to_csv(
        Path(OUT_DIR, f"{prefix}_PLS1_gene_ranking_desc.csv"), index=False
    )

    pls_brain_df = pd.DataFrame({
        "parcel": parcels_use,
        "pls1_brain_score": pls_res["brain_scores"],
        "z_map": z_use,
    })
    pls_brain_df.to_csv(
        Path(OUT_DIR, f"{prefix}_PLS1_brain_scores_validparcels.csv"), index=False
    )

    brain_scores_full = np.full(400, np.nan, dtype=float)
    brain_scores_full[np.array(parcels_use) - 1] = pls_res["brain_scores"]

    print("[4/6] Bootstrap PLS1 ...")
    pls_gene_stats = bootstrap_pls_gene_significance(
        z_vec=z_map,
        expr_df=expr,
        n_boot=N_BOOT_GENE,
        random_state=RANDOM_STATE,
    )
    pls_gene_stats_path = Path(OUT_DIR, f"{prefix}_PLS1_gene_bootstrap_stats.csv")
    pls_gene_stats.to_csv(pls_gene_stats_path, index=False)

    pls_gene_stats["is_absZ_gt3"] = np.abs(pls_gene_stats["z_boot"]) > 3
    genes_absZ_gt3 = pls_gene_stats.loc[
        pls_gene_stats["is_absZ_gt3"], "gene"
    ].astype(str).tolist()

    absz_txt_path = Path(OUT_DIR, f"{prefix}_genes_absZgt3.txt")
    absz_txt_path.write_text("\n".join(genes_absZ_gt3), encoding="utf-8")
    print(f"[GENE] {prefix}: |Z| > 3 genes n={len(genes_absZ_gt3)}")
    print(f"[GENE] Metascape-ready list: {absz_txt_path}")

    plot_top_absZ_gt3_gene_expression_heatmap(
        expr_df=expr,
        pls_gene_stats=pls_gene_stats,
        z_map=z_map,
        out_dir=OUT_DIR,
        prefix=prefix,
        z_thresh=3.0,
        top_n=10,
        use_half_brain=True,
    )

    print("[5/6] Optional modules ...")
    if GENESET_GMT:
        run_gene_enrichment(pls_gene_df=pls_gene_stats, out_dir=OUT_DIR, prefix=prefix)
    else:
        print("[GSEA] skipped: GENESET_GMT is empty")

    if HANSEN_RECEPTOR_CSV and os.path.isfile(HANSEN_RECEPTOR_CSV):
        rec_df = load_hansen_receptors()
        res_rec = correlate_brain_map_with_feature_table(
            brain_map=brain_scores_full,
            feature_df=rec_df,
            parcels_col=None,
            feature_label="receptor",
        )
        res_rec_path = Path(OUT_DIR, f"{prefix}_assoc_Hansen_receptors.csv")
        res_rec.to_csv(res_rec_path, index=False)
        print(f"[HANSEN] saved: {res_rec_path}")
    else:
        print("[HANSEN] skipped: no receptor csv")

    if CELLTYPE_PROFILE_CSV and os.path.isfile(CELLTYPE_PROFILE_CSV):
        cell_df = load_celltype_profile()
        res_cell = correlate_brain_map_with_feature_table(
            brain_map=brain_scores_full,
            feature_df=cell_df,
            parcels_col=None,
            feature_label="celltype",
        )
        res_cell_path = Path(OUT_DIR, f"{prefix}_assoc_celltypes.csv")
        res_cell.to_csv(res_cell_path, index=False)
        print(f"[CELLTYPE] saved: {res_cell_path}")
    else:
        print("[CELLTYPE] skipped: no cell-type csv")

    if NEUROSYNTH_TERMS_CSV and os.path.isfile(NEUROSYNTH_TERMS_CSV):
        neu_df = pd.read_csv(NEUROSYNTH_TERMS_CSV)
        res_neu = correlate_brain_map_with_feature_table(
            brain_map=brain_scores_full,
            feature_df=neu_df,
            parcels_col="parcel" if "parcel" in neu_df.columns else None,
            feature_label="term",
        )
        res_neu_path = Path(OUT_DIR, f"{prefix}_assoc_Neurosynth_terms.csv")
        res_neu.to_csv(res_neu_path, index=False)
        print(f"[NEUROSYNTH] saved: {res_neu_path}")
    else:
        print("[NEUROSYNTH] skipped: no Neurosynth csv")

    print("[6/6] Spin permutation ...")
    surr = spin_permutation_pval(
        z_map, expr, lh_lab, rh_lab,
        n_perm=N_PERM,
        random_state=RANDOM_STATE,
    )
    s_obs = pls_res["s1"]
    p_spin = (np.sum(surr >= s_obs) + 1) / (len(surr) + 1)

    np.save(Path(OUT_DIR, f"{prefix}_spin_s1_null.npy"), surr)
    with open(Path(OUT_DIR, f"{prefix}_spin_result.txt"), "w", encoding="utf-8") as f:
        f.write(f"Observed s1 = {s_obs:.6f}\n")
        f.write(f"Spin p-value (right-tailed) = {p_spin:.6g}\n")
        f.write(f"n_perm = {len(surr)}\n")
        f.write("Contrast = TMS(Post-Pre) - Control(Post-Pre)\n")

    r, p = pearsonr(pls_res["brain_scores"], z_use)
    print(f"[CORR] PLS1 brain_score vs z_map (valid parcels): r={r:.3f}, p={p:.3g}")

    if RUN_BEHAVIOR_PLS:
        print(
            "[BEHAV-PLS] skipped: this between-group change pipeline does not "
            "define a compatible behavioral contrast."
        )
    else:
        print("[BEHAV-PLS] skipped: no behavior csv")

    return {
        "prefix": prefix,
        "n_control": len(control_ids),
        "n_tms": len(tms_ids),
        "n_absZ_gt3": len(genes_absZ_gt3),
        "txt_path": str(absz_txt_path),
    }


# =========================
# Main
# =========================
def main():
    print("[GLOBAL 1/3] Load AHBA Schaefer-400 once ...")
    expr = load_ahba_schaefer400()

    print("[GLOBAL 2/3] Load parcellation labels ...")
    lh_lab, rh_lab = load_parcellation_labels()
    parcellation, n_vertices_total = build_vertices_per_parcel(lh_lab, rh_lab)

    print("[GLOBAL 3/3] Run Gradient-1 between-group change analysis ...")
    result = run_between_groups(
        expr=expr,
        lh_lab=lh_lab,
        rh_lab=rh_lab,
        parcellation=parcellation,
        n_vertices_total=n_vertices_total,
    )

    print("\n" + "=" * 90)
    print("SUMMARY: TMS(Post-Pre) - Control(Post-Pre), Gradient 1")
    print("=" * 90)
    print(f"Control matched N = {result['n_control']}")
    print(f"TMS matched N = {result['n_tms']}")
    print(f"|Z| > 3 genes n = {result['n_absZ_gt3']}")
    print(f"Metascape-ready file = {result['txt_path']}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
