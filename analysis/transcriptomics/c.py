# -*- coding: utf-8 -*-
"""
Plot cortical PLS1 score brain map on Schaefer400 surface.

Input files expected in RESULT_DIR:
    TMS_change_vs_Control_change_G1_ahba_expr_schaefer400.csv
    TMS_change_vs_Control_change_G1_validparcels.csv
    TMS_change_vs_Control_change_G1_PLS1_gene_bootstrap_stats.csv

Output:
    1) parcel-level PLS1 score csv
    2) cortical surface brain maps PNG/PDF
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from surfplot import Plot
from neuromaps.datasets import fetch_fslr
from brainspace.datasets import load_parcellation
import nibabel as nib
# =========================================================
# Font settings
# =========================================================
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"


# =========================================================
# Paths
# =========================================================
RESULT_DIR = Path(r"F:\lishiqi\组间比较\基因\gene")
OUT_DIR = Path(r"F:\lishiqi\组间比较\基因\c")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Schaefer400 annot 文件
ANNOT_DIR = Path(r"F:\lishiqi\预处理数据\subcortex")
LH_LABEL = Path(r"F:\lishiqi\新图\Figure 7\lh.Schaefer2018_400Parcels_7Networks_order.label.gii")
RH_LABEL = Path(r"F:\lishiqi\新图\Figure 7\rh.Schaefer2018_400Parcels_7Networks_order.label.gii")

PREFIXES = [
    "TMS_change_vs_Control_change_G1",
]

N_PARCELS = 400
N_PER_HEMI = 200

WEIGHT_COL = "weight"

# 固定对称色条范围，保持与原组内图相同的视觉标尺
USE_COMMON_SCALE = True

# 是否画 Schaefer parcel 边界
DRAW_CONTOURS = True

# =========================================================
# fsLR surface + Schaefer400 atlas
# 用这个来画出和你之前代码一样的脑区边界
# =========================================================
SURFACES = fetch_fslr()
LH_SURF, RH_SURF = SURFACES["veryinflated"]

ATLAS_32K = load_parcellation("schaefer", 400, join=True)

# 是否显示色条
SHOW_COLORBAR = True

# 2 x 2 论文总图中的统一单面板规格（约 90 x 83 mm）
PANEL_SIZE = (3.55, 2.75)
OUTPUT_DPI = 600

# 是否显示标题
ADD_TITLE = False

TITLE_MAP = {
    "TMS_change_vs_Control_change_G1": "TMS change vs Control change (G1)",
}

# =========================================================
# Colorbar font settings
# =========================================================
COLORBAR_LABEL_FONTSIZE = 8.0
COLORBAR_TICK_FONTSIZE = 6.8
COLORBAR_FONT = "Arial"

# 右侧竖向色条位置：[left, bottom, width, height]
# 宽度由 0.022 增至 0.032，使色条与脑图的视觉重量更协调。
COLORBAR_POS = [0.845, 0.18, 0.032, 0.64]
# =========================================================
# Colormap: negative = orange, positive = purple
# 和你截图里类似
# =========================================================
PLS1_CMAP = LinearSegmentedColormap.from_list(
    "orange_white_purple",
    [
        "#B65A00",
        "#E58B1A",
        "#F6C66D",
        "#F7F7F7",
        "#B5ABD8",
        "#7B5AA6",
        "#3F176D",
    ],
    N=256
)
MEDIAL_WALL_CMAP = ListedColormap(["#B8B8B8"])   # medial wall 灰色
OUTLINE_CMAP = ListedColormap(["#8A8A8A"])       # Schaefer 边界灰色

# =========================================================
# Basic functions
# =========================================================
def zscore_cols(arr):
    arr = np.asarray(arr, dtype=float)
    mu = np.nanmean(arr, axis=0, keepdims=True)
    sd = np.nanstd(arr, axis=0, ddof=1, keepdims=True)
    sd[~np.isfinite(sd)] = 1.0
    sd[sd == 0] = 1.0
    return (arr - mu) / sd


def zscore_1d(x):
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x, ddof=1)
    if (not np.isfinite(sd)) or sd == 0:
        sd = 1.0
    return (x - mu) / sd


def clean_name(x):
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)


def is_medial_or_unknown(name):
    name = name.lower()
    return (
        "medial" in name
        or "unknown" in name
        or "background" in name
    )


# =========================================================
# Step 1. Compute parcel-level PLS1 score
# =========================================================
def compute_pls1_score(prefix):
    expr_path = RESULT_DIR / f"{prefix}_ahba_expr_schaefer400.csv"
    valid_path = RESULT_DIR / f"{prefix}_validparcels.csv"
    gene_path = RESULT_DIR / f"{prefix}_PLS1_gene_bootstrap_stats.csv"

    for p in [expr_path, valid_path, gene_path]:
        if not p.exists():
            raise FileNotFoundError(f"[ERROR] File not found: {p}")

    # AHBA expression matrix: rows = parcels, columns = genes
    expr = pd.read_csv(expr_path, index_col=0)
    expr.index = expr.index.astype(int)

    # valid parcels
    valid = pd.read_csv(valid_path)
    if "parcel" not in valid.columns:
        raise ValueError(f"[ERROR] validparcels file has no 'parcel' column: {valid_path}")

    valid["parcel"] = valid["parcel"].astype(int)
    valid = valid.sort_values("parcel").reset_index(drop=True)

    # PLS1 gene weights
    gene_df = pd.read_csv(gene_path)
    required_cols = ["gene", WEIGHT_COL]
    missing = [c for c in required_cols if c not in gene_df.columns]
    if missing:
        raise ValueError(
            f"[ERROR] Missing columns in {gene_path.name}: {missing}\n"
            f"Available columns: {list(gene_df.columns)}"
        )

    gene_df["gene"] = gene_df["gene"].astype(str)
    gene_df[WEIGHT_COL] = pd.to_numeric(gene_df[WEIGHT_COL], errors="coerce")
    gene_df = gene_df.dropna(subset=[WEIGHT_COL]).copy()

    common_genes = [g for g in expr.columns if g in set(gene_df["gene"])]

    if len(common_genes) == 0:
        raise RuntimeError(f"[ERROR] No common genes between expression matrix and gene weights: {prefix}")

    # Keep the same parcel order as validparcels
    parcels = valid["parcel"].to_numpy(dtype=int)

    missing_parcels = [p for p in parcels if p not in expr.index]
    if missing_parcels:
        raise ValueError(
            f"[ERROR] Some valid parcels are not found in expression matrix: {missing_parcels[:10]}"
        )

    expr_use = expr.loc[parcels, common_genes].copy()

    weight_map = gene_df.set_index("gene")[WEIGHT_COL]
    w = weight_map.loc[common_genes].to_numpy(dtype=float)

    # Core PLS1 score calculation
    Y = expr_use.to_numpy(dtype=float)
    Ys = zscore_cols(Y)

    region_score = Ys @ w
    region_score = zscore_1d(region_score)

    # Full Schaefer400 vector
    full_score = np.full(N_PARCELS, np.nan, dtype=float)
    for parcel, score in zip(parcels, region_score):
        if 1 <= parcel <= N_PARCELS:
            full_score[parcel - 1] = score

    out_csv = OUT_DIR / f"{prefix}_PLS1_score_schaefer400.csv"
    out_df = pd.DataFrame({
        "parcel": np.arange(1, N_PARCELS + 1),
        "PLS1_score": full_score
    })
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"[OK] Saved parcel PLS1 score: {out_csv}")
    print(f"[INFO] {prefix}: common genes = {len(common_genes)}")
    print(f"[INFO] {prefix}: score range = {np.nanmin(full_score):.3f} to {np.nanmax(full_score):.3f}")

    return full_score


# =========================================================
# Step 2. Choose fsaverage mesh according to annot vertex number
# =========================================================
def get_fsaverage_by_vertex_count(n_vertices):
    if n_vertices == 10242:
        mesh = "fsaverage5"
    elif n_vertices == 40962:
        mesh = "fsaverage6"
    elif n_vertices == 163842:
        mesh = "fsaverage"
    else:
        raise ValueError(
            f"[ERROR] Unknown surface vertex number: {n_vertices}\n"
            f"Common values: fsaverage5=10242, fsaverage6=40962, fsaverage=163842"
        )

    print(f"[INFO] Using nilearn {mesh}, vertices per hemisphere = {n_vertices}")
    return datasets.fetch_surf_fsaverage(mesh=mesh)

def score400_to_32k(score400):
    """
    把 Schaefer400 parcel-level PLS1 score 映射到 fsLR 32k 顶点。
    score400: shape = (400,)
    ATLAS_32K: Schaefer400 fsLR atlas, label = 1-400
    """
    score400 = np.asarray(score400, dtype=float)

    if score400.shape[0] != 400:
        raise ValueError(f"[ERROR] score400 length should be 400, got {score400.shape[0]}")

    value_32k = np.full_like(ATLAS_32K, np.nan, dtype=float)

    for i in range(400):
        val = score400[i]

        # 缺失值显示成 0，也就是色条中间的白色附近
        if not np.isfinite(val):
            val = 0.0

        value_32k[ATLAS_32K == (i + 1)] = val

    return value_32k
# =========================================================
# Step 3. Map 400 parcel scores to surface vertices
# =========================================================
def parcel_score_to_vertices(full_score):
    if not LH_LABEL.exists():
        raise FileNotFoundError(f"[ERROR] LH label.gii not found: {LH_LABEL}")
    if not RH_LABEL.exists():
        raise FileNotFoundError(f"[ERROR] RH label.gii not found: {RH_LABEL}")

    # 读取 label.gii
    lh_img = nib.load(str(LH_LABEL))
    rh_img = nib.load(str(RH_LABEL))

    lh_labels = np.asarray(lh_img.darrays[0].data).astype(int)
    rh_labels = np.asarray(rh_img.darrays[0].data).astype(int)

    lh_vertex_data = np.full(lh_labels.shape, np.nan, dtype=float)
    rh_vertex_data = np.full(rh_labels.shape, np.nan, dtype=float)

    # 左半球 label，一般是 1-200，0 是 medial wall / background
    lh_unique = np.unique(lh_labels)
    lh_unique = lh_unique[lh_unique > 0]
    lh_unique = np.sort(lh_unique)

    # 右半球 label，可能是 1-200，也可能是 201-400
    rh_unique = np.unique(rh_labels)
    rh_unique = rh_unique[rh_unique > 0]
    rh_unique = np.sort(rh_unique)

    print(f"[INFO] LH unique labels: {lh_unique.min()} to {lh_unique.max()}, n={len(lh_unique)}")
    print(f"[INFO] RH unique labels: {rh_unique.min()} to {rh_unique.max()}, n={len(rh_unique)}")

    if len(lh_unique) != N_PER_HEMI:
        warnings.warn(
            f"[WARNING] LH parcel number = {len(lh_unique)}, expected {N_PER_HEMI}"
        )
    if len(rh_unique) != N_PER_HEMI:
        warnings.warn(
            f"[WARNING] RH parcel number = {len(rh_unique)}, expected {N_PER_HEMI}"
        )

    # =========================
    # LH: 映射到 parcel 1-200
    # =========================
    for lab in lh_unique:
        # 常见情况：左半球 label.gii 编号为 1-200
        parcel = int(lab)

        if 1 <= parcel <= 200:
            lh_vertex_data[lh_labels == lab] = full_score[parcel - 1]
        elif 1 <= parcel <= 400:
            lh_vertex_data[lh_labels == lab] = full_score[parcel - 1]

    # =========================
    # RH: 映射到 parcel 201-400
    # =========================
    # 情况 A：右半球 label.gii 编号是 1-200
    # 那么需要 +200，变成 201-400
    if rh_unique.max() <= 200:
        for lab in rh_unique:
            parcel = 200 + int(lab)
            if 201 <= parcel <= 400:
                rh_vertex_data[rh_labels == lab] = full_score[parcel - 1]

    # 情况 B：右半球 label.gii 编号本身就是 201-400
    else:
        for lab in rh_unique:
            parcel = int(lab)
            if 201 <= parcel <= 400:
                rh_vertex_data[rh_labels == lab] = full_score[parcel - 1]

    return lh_vertex_data, rh_vertex_data, lh_labels, rh_labels


# =========================================================
# Step 4. Plot 4-view brain map
# =========================================================
def plot_pls1_surface(prefix, full_score, vmin, vmax):
    """
    用 surfplot + fsLR32k + Schaefer400 outline 画 PLS1 score 脑图。
    medial wall 显示为灰色，Schaefer parcel 边界显示为灰色。
    """

    # 400 parcel score -> fsLR 32k vertex
    value_32k = score400_to_32k(full_score)

    # =====================================================
    # medial wall 灰色层
    # ATLAS_32K == 0 通常是 medial wall / background
    # 只让 medial wall 有值，其余地方是 NaN
    # =====================================================
    medial_wall = np.full_like(ATLAS_32K, np.nan, dtype=float)
    medial_wall[ATLAS_32K == 0] = 1.0

    # =====================================================
    # 建图
    # brightness 不要太高，否则脑图太平；
    # 0.55-0.70 比较接近参考图
    # =====================================================
    if SHOW_COLORBAR:
        p = Plot(
            LH_SURF,
            RH_SURF,
            views=["lateral", "medial"],
            zoom=1.60,
            brightness=0.65,
            size=(355, 275),
            background=(1, 1, 1)
        )
    else:
        p = Plot(
            LH_SURF,
            RH_SURF,
            views=["lateral", "medial"],
            zoom=1.65,
            brightness=0.65,
            size=(355, 275),
            background=(1, 1, 1)
        )

    # =====================================================
    # 第 0 层：medial wall 灰色
    # 这一步只给 atlas=0 的区域上灰色
    # =====================================================
    p.add_layer(
        medial_wall,
        cmap=MEDIAL_WALL_CMAP,
        color_range=(0, 1),
        cbar=False
    )

    # =====================================================
    # 第 1 层：PLS1 score 连续色图
    # 只覆盖 Schaefer400 parcel 区域
    # =====================================================
    p.add_layer(
        value_32k,
        cmap=PLS1_CMAP,
        color_range=(vmin, vmax),
        cbar=False
    )

    # =====================================================
    # 第 2 层：Schaefer400 灰色边界线
    # =====================================================
    if DRAW_CONTOURS:
        p.add_layer(
            ATLAS_32K,
            cmap=OUTLINE_CMAP,
            as_outline=True,
            cbar=False
        )

    fig = p.build()
    fig.set_size_inches(*PANEL_SIZE, forward=True)
    # 给右侧竖向色条留出固定空间，同时尽量放大四个脑表面视图。
    fig.subplots_adjust(left=0.01, right=0.82, top=0.99, bottom=0.01,
                        wspace=0.02, hspace=0.02)

    # =====================================================
    # 标题，可选
    # =====================================================
    if ADD_TITLE:
        fig.suptitle(
            TITLE_MAP.get(prefix, prefix),
            fontsize=22,
            fontname="Arial",
            fontweight="bold",
            y=0.98
        )

    # =====================================================
    # 手动 colorbar
    # =====================================================
    if SHOW_COLORBAR:
        cax = fig.add_axes(COLORBAR_POS)

        norm = Normalize(vmin=vmin, vmax=vmax)
        sm = ScalarMappable(norm=norm, cmap=PLS1_CMAP)
        sm.set_array([])

        cb = fig.colorbar(sm, cax=cax, orientation="vertical")

        cb.set_ticks([vmin, vmax])
        cb.set_ticklabels([f"{vmin:.2f}", f"{vmax:.2f}"])

        # 色条标题：使用轴坐标精确控制与色条的距离。
        # 上下端点刻度位于右侧，标题放在中部，因此两者不会重叠。
        cb.set_label("")
        cb.ax.text(
            1.55,
            0.50,
            "PLS1 score",
            transform=cb.ax.transAxes,
            fontsize=COLORBAR_LABEL_FONTSIZE,
            fontname=COLORBAR_FONT,
            fontweight="normal",
            rotation=270,
            ha="center",
            va="center",
        )
        # 色条标题和刻度均放在右侧；色条整体左移以保留完整文字空间。
        cb.ax.yaxis.set_label_position("right")
        cb.ax.yaxis.set_ticks_position("right")

        # 色条刻度：-2.74 / 2.74
        cb.ax.tick_params(
            axis="y",
            labelsize=COLORBAR_TICK_FONTSIZE,
            width=0.8,
            length=3,
            pad=2
        )

        for tick in cb.ax.get_yticklabels():
            tick.set_fontname(COLORBAR_FONT)
            tick.set_fontsize(COLORBAR_TICK_FONTSIZE)
            tick.set_fontweight("normal")

        cb.outline.set_linewidth(0.8)

    out_png = OUT_DIR / f"{prefix}_PLS1_score_brainmap_surfplot_grayMedial.png"
    out_pdf = OUT_DIR / f"{prefix}_PLS1_score_brainmap_surfplot_grayMedial.pdf"
    out_svg = OUT_DIR / f"{prefix}_PLS1_score_brainmap_surfplot_grayMedial.svg"

    # 固定画布尺寸，方便与另外三个面板严格等分排版。
    fig.savefig(out_svg, facecolor="white")
    fig.savefig(out_png, dpi=OUTPUT_DPI, facecolor="white")
    fig.savefig(out_pdf, facecolor="white")
    plt.close(fig)

    print(f"[OK] Saved brain map: {out_svg}")
    print(f"[OK] Saved brain map: {out_png}")
    print(f"[OK] Saved brain map: {out_pdf}")


# =========================================================
# Main
# =========================================================
def main():
    print("==================================================")
    print("[INFO] RESULT_DIR =", RESULT_DIR)
    print("[INFO] OUT_DIR    =", OUT_DIR)
    print("[INFO] LH_LABEL   =", LH_LABEL)
    print("[INFO] RH_LABEL   =", RH_LABEL)
    print("==================================================")

    all_scores = {}

    for prefix in PREFIXES:
        print(f"\n[RUN] Computing PLS1 score: {prefix}")
        full_score = compute_pls1_score(prefix)
        all_scores[prefix] = full_score

    if USE_COMMON_SCALE:
        # =====================================================
        # 保留原图固定的 -1.80 到 1.80 色条范围
        # =====================================================
        vmin, vmax = -1.80, 1.80

        print(f"\n[INFO] Common fixed color scale: {vmin:.3f} to {vmax:.3f}")

    else:
        vmin, vmax = None, None
        print("\n[INFO] Each group uses its own color scale.")

    for prefix, full_score in all_scores.items():
        print(f"\n[RUN] Plotting brain map: {prefix}")

        if USE_COMMON_SCALE:
            plot_vmin, plot_vmax = vmin, vmax
        else:
            plot_vmin = np.nanmin(full_score)
            plot_vmax = np.nanmax(full_score)

        plot_pls1_surface(prefix, full_score, plot_vmin, plot_vmax)

    print("\n[DONE] All PLS1 score brain maps finished.")


if __name__ == "__main__":
    main()
