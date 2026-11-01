# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow
import matplotlib as mpl

# =========================
# 字体设置：Arial
# =========================
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

# 2 x 2 论文总图中的统一单面板规格（约 90 x 83 mm）
PANEL_SIZE = (3.55, 2.75)
OUTPUT_DPI = 600
TITLE_FONTSIZE = 8.5
HEADER_FONTSIZE = 7.6
BODY_FONTSIZE = 7.4
PFDR_FONTSIZE = 6.5
ELLIPSIS_FONTSIZE = 10.0

# =========================
# 路径配置
# =========================
RESULT_DIR = Path(r"F:\lishiqi\组间比较\基因\gene")
OUT_DIR = Path(r"F:\lishiqi\组间比较\基因\a")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PREFIXES = [
    "TMS_change_vs_Control_change_G1",
]

TOP_POS = 4
TOP_NEG = 4

WEIGHT_COL = "weight"
PVAL_COL = "q_boot"
SIG_THRESH = 0.05
DISPLAY_COL = "weight_z"


def zscore_1d(x):
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x, ddof=1)
    if (not np.isfinite(sd)) or sd == 0:
        sd = 1.0
    return (x - mu) / sd


def blend(c1, c2, t):
    c1 = np.array(c1, dtype=float)
    c2 = np.array(c2, dtype=float)
    return tuple((1 - t) * c1 + t * c2)


def lighten(color, amount):
    white = np.array([1.0, 1.0, 1.0])
    color = np.array(color, dtype=float)
    return tuple((1 - amount) * color + amount * white)


def p_lighten_factor(p):
    if not np.isfinite(p):
        return 0.50

    p = float(np.clip(p, 1e-300, 1.0))
    p_lo, p_hi = 1e-4, 0.05

    lp = -np.log10(p)
    lp_lo = -np.log10(p_hi)
    lp_hi = -np.log10(p_lo)

    if lp_hi == lp_lo:
        s = 1.0
    else:
        s = (lp - lp_lo) / (lp_hi - lp_lo)
    s = float(np.clip(s, 0.0, 1.0))

    return 0.08 + (1.0 - s) * 0.42


def color_by_rank_and_sign(rank_idx, total_n, sign="pos", pval=np.nan):
    total_n = max(int(total_n), 1)
    t = rank_idx / max(total_n - 1, 1)

    if sign == "pos":
        c_start = (0.83, 0.05, 0.15)
        c_mid   = (0.95, 0.35, 0.20)
        c_end   = (0.98, 0.67, 0.44)
    else:
        c_start = (0.00, 0.33, 0.74)
        c_mid   = (0.12, 0.60, 0.88)
        c_end   = (0.63, 0.85, 0.95)

    if t <= 0.5:
        base = blend(c_start, c_mid, t / 0.5 if 0.5 != 0 else 0)
    else:
        base = blend(c_mid, c_end, (t - 0.5) / 0.5)

    base = lighten(base, p_lighten_factor(pval))
    return base


def load_one(prefix):
    gene_path = RESULT_DIR / f"{prefix}_PLS1_gene_bootstrap_stats.csv"
    if not gene_path.exists():
        raise FileNotFoundError(f"[ERROR] File not found: {gene_path}")

    df = pd.read_csv(gene_path)

    required_cols = ["gene", WEIGHT_COL, PVAL_COL]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"[ERROR] Missing columns in {gene_path.name}: {missing_cols}\n"
            f"Available columns: {list(df.columns)}"
        )

    df["gene"] = df["gene"].astype(str)
    df[WEIGHT_COL] = pd.to_numeric(df[WEIGHT_COL], errors="coerce")
    df[PVAL_COL] = pd.to_numeric(df[PVAL_COL], errors="coerce")
    df = df.dropna(subset=[WEIGHT_COL]).copy()

    df[DISPLAY_COL] = zscore_1d(df[WEIGHT_COL].to_numpy())

    sig = df[df[PVAL_COL] < SIG_THRESH].copy()
    work = sig if len(sig) >= (TOP_POS + TOP_NEG) else df.copy()

    pos = (
        work[work[DISPLAY_COL] > 0]
        .sort_values(DISPLAY_COL, ascending=False)
        .head(TOP_POS)
        .copy()
    )

    neg = (
        work[work[DISPLAY_COL] < 0]
        .sort_values(DISPLAY_COL, ascending=True)
        .head(TOP_NEG)
        .copy()
    )

    pos_mid = (
        work[work[DISPLAY_COL] > 0]
        .sort_values(DISPLAY_COL, ascending=True)
        .head(1)
        .copy()
    )

    neg_mid = (
        work[work[DISPLAY_COL] < 0]
        .sort_values(DISPLAY_COL, ascending=False)
        .head(1)
        .copy()
    )

    return pos, pos_mid, neg_mid, neg


def draw_one(prefix):
    pos, pos_mid, neg_mid, neg = load_one(prefix)

    fig, ax = plt.subplots(figsize=PANEL_SIZE, dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    title_y = 0.965
    top_line_y = 0.91
    header_y = 0.855
    header_bottom_y = 0.82

    ax.text(0.54, title_y, "Ranked PLS1 gene loadings", ha="center", va="top",
            fontsize=TITLE_FONTSIZE)
    ax.plot([0.18, 0.95], [top_line_y, top_line_y], color="black", lw=0.75)
    ax.text(0.21, header_y, "Gene", ha="left", va="center", fontsize=HEADER_FONTSIZE)
    ax.text(0.90, header_y, "Loading", ha="right", va="center", fontsize=HEADER_FONTSIZE)
    ax.plot([0.18, 0.95], [header_bottom_y, header_bottom_y], color="black", lw=0.75)

    top_gap = 0.04
    row_step = 0.07

    pos_y = [
        header_bottom_y - top_gap,
        header_bottom_y - top_gap - row_step,
        header_bottom_y - top_gap - 2 * row_step,
        header_bottom_y - top_gap - 3 * row_step,
    ]

    red_ellipsis_y = pos_y[3] - 0.06
    pos_mid_y = red_ellipsis_y - 0.06
    neg_mid_y = pos_mid_y - 0.07
    blue_ellipsis_y = neg_mid_y - 0.06

    neg4_bottom_y = blue_ellipsis_y - 0.06 - 0.07 - 0.07 - 0.07
    neg_y = [
        neg4_bottom_y,
        neg4_bottom_y + 0.07,
        neg4_bottom_y + 0.07 * 2,
        neg4_bottom_y + 0.07 * 3,
    ]

    bottom_line_y = neg4_bottom_y - 0.04

    red_tip = header_bottom_y
    red_tail = pos_mid_y
    blue_tip = bottom_line_y
    blue_tail = neg_mid_y

    red_arrow = FancyArrow(
        0.08, red_tail, 0, red_tip - red_tail,
        width=0.05, head_width=0.10, head_length=0.06,
        length_includes_head=True,
        color=(0.96, 0.55, 0.55), alpha=0.88
    )
    blue_arrow = FancyArrow(
        0.08, blue_tail, 0, blue_tip - blue_tail,
        width=0.05, head_width=0.10, head_length=0.06,
        length_includes_head=True,
        color=(0.50, 0.70, 0.93), alpha=0.88
    )
    ax.add_patch(red_arrow)
    ax.add_patch(blue_arrow)

    ax.text(0.01, (red_tip + red_tail) / 2, "PFDR<0.05",
            rotation=90, ha="center", va="center", fontsize=PFDR_FONTSIZE)
    ax.text(0.01, (blue_tip + blue_tail) / 2, "PFDR<0.05",
            rotation=90, ha="center", va="center", fontsize=PFDR_FONTSIZE)

    for i, (y, (_, row)) in enumerate(zip(pos_y, pos.iterrows())):
        c = color_by_rank_and_sign(i, len(pos), sign="pos", pval=row[PVAL_COL])
        ax.text(0.20, y, row["gene"], ha="left", va="center",
                fontsize=BODY_FONTSIZE, fontstyle="italic", color=c)
        ax.text(0.90, y, f'{row[DISPLAY_COL]:.2f}', ha="right", va="center",
                fontsize=BODY_FONTSIZE, color=c)

    # 这里改成稳定显示的竖向省略号
    ax.text(0.36, red_ellipsis_y, r"$\vdots$", ha="center", va="center",
            fontsize=ELLIPSIS_FONTSIZE, color=(0.70, 0.12, 0.10))
    ax.text(0.83, red_ellipsis_y, r"$\vdots$", ha="center", va="center",
            fontsize=ELLIPSIS_FONTSIZE, color=(0.70, 0.12, 0.10))

    if len(pos_mid) > 0:
        row = pos_mid.iloc[0]
        c = color_by_rank_and_sign(len(pos), len(pos) + 1, sign="pos", pval=row[PVAL_COL])
        ax.text(0.20, pos_mid_y, row["gene"], ha="left", va="center",
                fontsize=BODY_FONTSIZE, fontstyle="italic", color=c)
        ax.text(0.90, pos_mid_y, f'{row[DISPLAY_COL]:.2f}', ha="right", va="center",
                fontsize=BODY_FONTSIZE, color=c)

    if len(neg_mid) > 0:
        row = neg_mid.iloc[0]
        c = color_by_rank_and_sign(len(neg), len(neg) + 1, sign="neg", pval=row[PVAL_COL])
        ax.text(0.20, neg_mid_y, row["gene"], ha="left", va="center",
                fontsize=BODY_FONTSIZE, fontstyle="italic", color=c)
        ax.text(0.90, neg_mid_y, f'{row[DISPLAY_COL]:.2f}', ha="right", va="center",
                fontsize=BODY_FONTSIZE, color=c)

    # 这里改成稳定显示的竖向省略号
    ax.text(0.36, blue_ellipsis_y, r"$\vdots$", ha="center", va="center",
            fontsize=ELLIPSIS_FONTSIZE, color=(0.05, 0.35, 0.80))
    ax.text(0.83, blue_ellipsis_y, r"$\vdots$", ha="center", va="center",
            fontsize=ELLIPSIS_FONTSIZE, color=(0.05, 0.35, 0.80))

    for i, (y, (_, row)) in enumerate(zip(neg_y, neg.iterrows())):
        c = color_by_rank_and_sign(i, len(neg), sign="neg", pval=row[PVAL_COL])
        ax.text(0.20, y, row["gene"], ha="left", va="center",
                fontsize=BODY_FONTSIZE, fontstyle="italic", color=c)
        ax.text(0.90, y, f'{row[DISPLAY_COL]:.2f}', ha="right", va="center",
                fontsize=BODY_FONTSIZE, color=c)

    ax.plot([0.18, 0.95], [bottom_line_y, bottom_line_y], color="black", lw=0.75)
    fig.subplots_adjust(left=0.03, right=0.98, top=0.98, bottom=0.03)

    out_png = OUT_DIR / f"{prefix}_ranked_loadings_only_weightZ_v2layout.png"
    out_pdf = OUT_DIR / f"{prefix}_ranked_loadings_only_weightZ_v2layout.pdf"
    out_svg = OUT_DIR / f"{prefix}_ranked_loadings_only_weightZ_v2layout.svg"

    # 不使用 tight 裁切，保证四个独立面板导出的画布尺寸完全一致。
    fig.savefig(out_png, dpi=OUTPUT_DPI, facecolor="white")
    fig.savefig(out_pdf, facecolor="white")
    fig.savefig(out_svg, facecolor="white")
    plt.close(fig)

    print(f"[OK] Saved: {out_png}")
    print(f"[OK] Saved: {out_pdf}")
    print(f"[OK] Saved: {out_svg}")


def main():
    print("==================================================")
    print("[INFO] RESULT_DIR =", RESULT_DIR)
    print("[INFO] OUT_DIR    =", OUT_DIR)
    print("[INFO] PREFIXES   =", PREFIXES)
    print("==================================================")

    for prefix in PREFIXES:
        print(f"[RUN] {prefix}")
        draw_one(prefix)

    print("[DONE] All figures finished.")


if __name__ == "__main__":
    main()
