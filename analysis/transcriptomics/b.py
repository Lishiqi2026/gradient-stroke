#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, linregress

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# =========================
# 路径配置：组间 Gradient 1 结果
# =========================
RESULT_DIR = Path(r"F:\lishiqi\组间比较\基因\gene")
OUT_DIR = Path(r"F:\lishiqi\组间比较\基因\b")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PREFIXES = [
    "TMS_change_vs_Control_change_G1",
]

# 参考图配色：橙色散点 + 紫色拟合线
COLOR_MAP = {
    "TMS_change_vs_Control_change_G1": "#F2B75E",
}
REGRESSION_COLOR = "#7063B7"
# =========================
# 字体大小设置
# =========================
PANEL_SIZE = (3.55, 2.75)
OUTPUT_DPI = 600
AX_LABEL_FONTSIZE = 8.5   # 最终四联图中的坐标轴标题
TICK_FONTSIZE = 7.5       # x轴、y轴刻度数字
ANNOT_FONTSIZE = 7.5      # 图里面 r 和 p_spin 的图注

def zscore_cols(arr):
    arr = np.asarray(arr, float)
    mu = np.nanmean(arr, axis=0, keepdims=True)
    sd = np.nanstd(arr, axis=0, ddof=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (arr - mu) / sd


def zscore_1d(x):
    x = np.asarray(x, float)
    mu = np.nanmean(x)
    sd = np.nanstd(x, ddof=1)
    if sd == 0 or not np.isfinite(sd):
        sd = 1.0
    return (x - mu) / sd


def parse_spin_p(txt_path):
    p_spin = np.nan
    if not txt_path.exists():
        return p_spin

    for line in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "Spin p-value" in line:
            m = re.search(r"=\s*([\-+eE0-9\.]+)", line)
            if m:
                p_spin = float(m.group(1))
                break
    return p_spin


def fmt_p(p):
    if not np.isfinite(p):
        return "NA"
    if p < 0.0001:
        return "< 0.0001"
    return f"= {p:.4f}"


def make_one(prefix):
    expr_path = RESULT_DIR / f"{prefix}_ahba_expr_schaefer400.csv"
    valid_path = RESULT_DIR / f"{prefix}_validparcels.csv"
    gene_path = RESULT_DIR / f"{prefix}_PLS1_gene_bootstrap_stats.csv"
    spin_path = RESULT_DIR / f"{prefix}_spin_result.txt"

    # --------- 检查文件是否存在 ---------
    required_files = [expr_path, valid_path, gene_path, spin_path]
    missing = [str(p) for p in required_files if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"[ERROR] Missing files for prefix: {prefix}\n" + "\n".join(missing)
        )

    expr = pd.read_csv(expr_path, index_col=0)
    expr.index = expr.index.astype(int)

    valid = pd.read_csv(valid_path)
    y_col = "t" if "t" in valid.columns else "z"
    valid["parcel"] = valid["parcel"].astype(int)
    valid = valid.sort_values("parcel").reset_index(drop=True)

    gene_df = pd.read_csv(gene_path)
    gene_df["gene"] = gene_df["gene"].astype(str)
    gene_df["weight"] = pd.to_numeric(gene_df["weight"], errors="coerce")
    gene_df = gene_df.dropna(subset=["weight"])

    common_genes = [g for g in expr.columns if g in set(gene_df["gene"])]
    if len(common_genes) == 0:
        raise RuntimeError(f"[ERROR] No common genes found for {prefix}")

    expr_use = expr.loc[valid["parcel"].to_numpy(), common_genes].copy()

    weight_map = gene_df.set_index("gene")["weight"]
    w = weight_map.loc[common_genes].to_numpy(dtype=float)

    Y = expr_use.to_numpy(dtype=float)
    Ys = zscore_cols(Y)
    region_score = Ys @ w
    region_score = zscore_1d(region_score)

    y = pd.to_numeric(valid[y_col], errors="coerce").to_numpy(dtype=float)

    mask = np.isfinite(region_score) & np.isfinite(y)
    x = region_score[mask]
    y = y[mask]

    if len(x) < 3:
        raise RuntimeError(f"[ERROR] Too few valid points for correlation: {prefix}")

    r, _ = pearsonr(x, y)
    fit = linregress(x, y)
    p_spin = parse_spin_p(spin_path)

    point_color = COLOR_MAP.get(prefix, "#D8D465")

    fig, ax = plt.subplots(figsize=PANEL_SIZE, dpi=300)

    ax.scatter(
        x, y,
        s=18,
        color=point_color,
        alpha=0.82,
        edgecolors="white",
        linewidths=0.25
    )

    xx = np.linspace(x.min() - 0.1 * np.ptp(x), x.max() + 0.1 * np.ptp(x), 300)
    yy = fit.intercept + fit.slope * xx
    ax.plot(xx, yy, color=REGRESSION_COLOR, lw=1.35)

    ax.axhline(0, color="black", lw=0.65, ls=(0, (3, 3)))
    ax.axvline(0, color="black", lw=0.65, ls=(0, (3, 3)))

    ax.set_xlabel("PLS1 score", fontsize=AX_LABEL_FONTSIZE)
    ax.set_ylabel(
        "TMS-Control change t-value" if y_col == "t" else "TMS-Control change z-value",
        fontsize=AX_LABEL_FONTSIZE
    )

    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE, width=0.8, length=3.5)
    for s in ax.spines.values():
        s.set_linewidth(0.8)

    ax.text(
        0.57, 0.04,
        rf"$r_{{({len(x) - 2})}}={r:.2f}$" + "\n" + rf"$p_{{spin}}$ {fmt_p(p_spin)}",
        transform=ax.transAxes,
        fontsize=ANNOT_FONTSIZE,
        ha="left",
        va="bottom",
        linespacing=0.90
    )

    # 固定面板边界，避免 bbox_inches='tight' 使四张图导出尺寸不一致。
    fig.subplots_adjust(left=0.17, right=0.98, top=0.97, bottom=0.20)

    out_png = OUT_DIR / f"{prefix}_scatter_only.png"
    out_pdf = OUT_DIR / f"{prefix}_scatter_only.pdf"
    out_svg = OUT_DIR / f"{prefix}_scatter_only.svg"
    fig.savefig(out_svg, facecolor="white")
    fig.savefig(out_png, dpi=OUTPUT_DPI, facecolor="white")
    fig.savefig(out_pdf, facecolor="white")
    plt.close(fig)

    print(f"[OK] {out_svg}")
    print(f"[OK] {out_png}")
    print(f"[OK] {out_pdf}")


def main():
    for prefix in PREFIXES:
        make_one(prefix)


if __name__ == "__main__":
    main()
