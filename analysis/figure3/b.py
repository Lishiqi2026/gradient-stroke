#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Standalone plotting; preserve the original non-interactive backend.
import matplotlib as mpl
mpl.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D
from scipy.stats import ttest_ind


# =========================================================
# Font / style  —— 与原代码一致
# =========================================================
font_sizes = {
    "global": 7.0,
    "legend_item": 7.0,
    "axis_label": 7.0,
    "tick_label": 7.0,
    "title": 7.0,
}

mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": font_sizes["global"],
    "axes.titlesize": font_sizes["title"],
    "axes.labelsize": font_sizes["axis_label"],
    "xtick.labelsize": font_sizes["tick_label"],
    "ytick.labelsize": font_sizes["tick_label"],
    "legend.fontsize": font_sizes["legend_item"],
    "figure.titlesize": font_sizes["title"],
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


# =========================================================
# Paths / settings
# =========================================================
BASE = Path(r"F:\lishiqi\预处理数据\cortex\（10，）")

GROUPS = ["Control", "TMS"]
PLOT_FIRST_K = 10

OUT_DIR_SINGLE = Path(r"F:\lishiqi\组间比较\b")
OUT_DIR_COMBINED = Path(r"F:\lishiqi\组间比较\b")


# =========================================================
# Colors
# =========================================================
GROUP_COLORS = {
    "Control": {
        "before": "#C98599",
        "after":  "#E5B5BF",
    },
    "TMS": {
        "before": "#5E8CA5",
        "after":  "#A9C6D4",
    },
    "Between": {
        "before": "#C98599",
        "after":  "#5E8CA5",
    },
}


def get_legend_handles(group_name):
    before_color = GROUP_COLORS[group_name]["before"]
    after_color  = GROUP_COLORS[group_name]["after"]

    return [
        Line2D(
            [0], [0],
            marker='o', linestyle='none',
            markerfacecolor=before_color,
            markeredgecolor=before_color,
            markeredgewidth=1.0,
            markersize=7.0,
            label='Control'
        ),
        Line2D(
            [0], [0],
            marker='D', linestyle='none',
            markerfacecolor=after_color,
            markeredgecolor=after_color,
            markeredgewidth=1.0,
            markersize=7.0,
            label='TMS'
        ),
    ]


COMBINED_LEGEND_HANDLES = [
    Line2D([0], [0], marker='D', linestyle='none',
           markerfacecolor=GROUP_COLORS["Control"]["before"],
           markeredgecolor=GROUP_COLORS["Control"]["before"],
           markersize=6.0, label='Control Pre'),
    Line2D([0], [0], marker='D', linestyle='none',
           markerfacecolor=GROUP_COLORS["Control"]["after"],
           markeredgecolor=GROUP_COLORS["Control"]["after"],
           markersize=6.0, label='Control Post'),
    Line2D([0], [0], marker='D', linestyle='none',
           markerfacecolor=GROUP_COLORS["TMS"]["before"],
           markeredgecolor=GROUP_COLORS["TMS"]["before"],
           markersize=6.0, label='TMS Pre'),
    Line2D([0], [0], marker='D', linestyle='none',
           markerfacecolor=GROUP_COLORS["TMS"]["after"],
           markeredgecolor=GROUP_COLORS["TMS"]["after"],
           markersize=6.0, label='TMS Post'),
]

# =========================================================
# Layout control —— 与原代码一致
# =========================================================
BOTTOM_LAYOUT = {
    "xtick_area_in": 0.20,
    "legend_h_in": 0.18,
    "xlabel_h_in": 0.12,
    "gap_tick_legend_in": 0.04,
    "gap_legend_xlabel_in": 0.05,
    "bottom_blank_in": 0.02,
}

# =========================================================
# Significance settings —— 与原代码一致
# =========================================================
SIG_ALPHA = 0.05
SIG_STAR_FONTSIZE = 16
SIG_Y_PAD_FRAC = 0.02
SIG_TOP_MARGIN_FRAC = 0.04


# =========================================================
# Helpers
# =========================================================
def load_explained_stack(group_dir: Path):
    """
    新目录结构：
    E:\\newtreat\\（10，）\\Control\\before\\*_explained.npy
    E:\\newtreat\\（10，）\\Control\\after\\*_explained.npy
    E:\\newtreat\\（10，）\\TMS\\before\\*_explained.npy
    E:\\newtreat\\（10，）\\TMS\\after\\*_explained.npy
    """
    if not group_dir.exists():
        raise FileNotFoundError(f"Missing folder: {group_dir}")

    files = [p for p in group_dir.glob("*.npy") if "GROUP_" not in p.name]
    files = sorted(files)

    if len(files) == 0:
        raise FileNotFoundError(f"No per-file *_explained.npy found in: {group_dir}")

    E = np.stack([np.load(p) for p in files], axis=0)
    return E, len(files)


def _subject_key(path: Path):
    return path.stem.split("-")[0].split("_")[0]


def load_paired_change(group_name):
    """Match Pre/Post within a group by subject ID and return Post - Pre."""
    before_dir = BASE / group_name / "before"
    after_dir = BASE / group_name / "after"
    before_files = sorted(p for p in before_dir.glob("*.npy") if "GROUP_" not in p.name)
    after_files = sorted(p for p in after_dir.glob("*.npy") if "GROUP_" not in p.name)
    before_map = {_subject_key(p): p for p in before_files}
    after_map = {_subject_key(p): p for p in after_files}
    common = sorted(set(before_map) & set(after_map))
    if not common:
        raise RuntimeError(f"No matched Pre/Post subjects for {group_name}")
    changes = [np.load(after_map[sid]) - np.load(before_map[sid]) for sid in common]
    print(f"[INFO] {group_name}: matched Pre/Post subjects = {len(common)}")
    return np.stack(changes, axis=0), common


def save_figure_multi(fig, out_base: Path, png_dpi=600, tight=True, pad_inches=0.025):
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_base.with_suffix(".png")), dpi=png_dpi, bbox_inches=("tight" if tight else None), pad_inches=pad_inches)
    fig.savefig(str(out_base.with_suffix(".pdf")), bbox_inches=("tight" if tight else None), pad_inches=pad_inches)
    fig.savefig(str(out_base.with_suffix(".svg")), bbox_inches=("tight" if tight else None), pad_inches=pad_inches)
    print(
        f"Saved:\n"
        f"  {out_base.with_suffix('.png')}\n"
        f"  {out_base.with_suffix('.pdf')}\n"
        f"  {out_base.with_suffix('.svg')}"
    )


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", width=0.8, length=3)
    ax.grid(False)


def compute_mean_std(E, k):
    k = min(k, E.shape[1])
    mean = np.nanmean(E[:, :k], axis=0)
    std = np.nanstd(E[:, :k], axis=0, ddof=1) if E.shape[0] > 1 else np.zeros(k)
    return mean, std


def compute_component_pvalues(E_before, E_after, k):
    k = min(k, E_before.shape[1], E_after.shape[1])
    pvals = np.full(k, np.nan, dtype=float)

    for i in range(k):
        b = np.asarray(E_before[:, i], dtype=float)
        a = np.asarray(E_after[:, i], dtype=float)

        b = b[np.isfinite(b)]
        a = a[np.isfinite(a)]
        if len(b) >= 2 and len(a) >= 2:
            _, p = ttest_ind(b, a, equal_var=False, nan_policy="omit")
            pvals[i] = p

    return pvals


def add_significance_stars(ax, E_before, E_after, k, alpha=SIG_ALPHA):
    k = min(k, E_before.shape[1], E_after.shape[1])

    mean_b, std_b = compute_mean_std(E_before, k)
    mean_a, std_a = compute_mean_std(E_after, k)
    pvals = compute_component_pvalues(E_before, E_after, k)

    comps = np.arange(1, k + 1)

    y0, y1 = ax.get_ylim()
    yr = y1 - y0
    tops = np.maximum(mean_b + std_b, mean_a + std_a)

    for x, top, p in zip(comps, tops, pvals):
        if np.isfinite(p) and p < alpha:
            y_star = min(top + SIG_Y_PAD_FRAC * yr, y1 - SIG_TOP_MARGIN_FRAC * yr)
            ax.text(
                x, y_star, "*",
                ha="center",
                va="bottom",
                fontsize=SIG_STAR_FONTSIZE,
                color="black",
                clip_on=False,
                zorder=5,
            )


def point_error_plot(ax, E_before, E_after, k, group_name):
    k = min(k, E_before.shape[1], E_after.shape[1])
    before_color = GROUP_COLORS[group_name]["before"]
    after_color = GROUP_COLORS[group_name]["after"]
    comps = np.arange(1, k + 1)

    mean_b, std_b = compute_mean_std(E_before, k)
    mean_a, std_a = compute_mean_std(E_after, k)

    ax.axhline(0.0, color="#B9BDC1", linestyle=":", linewidth=0.9, zorder=1)

    offset = 0.12
    x_before = comps - offset
    x_after = comps + offset

    ax.errorbar(
        x_before, mean_b, yerr=std_b,
        fmt='o', markersize=7.0,
        elinewidth=1.55, capsize=3.4,
        color=before_color,
        markerfacecolor=before_color,
        markeredgecolor=before_color,
        markeredgewidth=0.55,
        alpha=0.98,
        linestyle='none',
        zorder=3,
    )

    ax.errorbar(
        x_after, mean_a, yerr=std_a,
        fmt='D', markersize=7.0,
        elinewidth=1.55, capsize=3.4,
        color=after_color,
        markerfacecolor=after_color,
        markeredgecolor=after_color,
        markeredgewidth=0.55,
        alpha=0.98,
        linestyle='none',
        zorder=3,
    )

    ax.set_xticks(comps)
    ax.set_xticklabels([str(i) for i in comps], fontsize=font_sizes["tick_label"])
    ax.tick_params(axis="y", labelsize=font_sizes["tick_label"])
    style_axes(ax)


def get_bottom_layout_params(fig_h):
    xtick_area_in = BOTTOM_LAYOUT["xtick_area_in"]
    legend_h_in = BOTTOM_LAYOUT["legend_h_in"]
    xlabel_h_in = BOTTOM_LAYOUT["xlabel_h_in"]
    gap_tick_legend_in = BOTTOM_LAYOUT["gap_tick_legend_in"]
    gap_legend_xlabel_in = BOTTOM_LAYOUT["gap_legend_xlabel_in"]
    bottom_blank_in = BOTTOM_LAYOUT["bottom_blank_in"]

    xlabel_y_frac = (bottom_blank_in + xlabel_h_in / 2.0) / fig_h

    legend_y_frac = (
        bottom_blank_in
        + xlabel_h_in
        + gap_legend_xlabel_in
        + legend_h_in / 2.0
    ) / fig_h

    bottom_frac = (
        bottom_blank_in
        + xlabel_h_in
        + gap_legend_xlabel_in
        + legend_h_in
        + gap_tick_legend_in
        + xtick_area_in
    ) / fig_h

    return {
        "legend_y_frac": legend_y_frac,
        "xlabel_y_frac": xlabel_y_frac,
        "bottom_frac": bottom_frac,
    }


# =========================================================
# Single plot
# =========================================================
def save_single_plot(group_name, E_b, E_a, out_base: Path, y_max):
    # Match the supplied compact landscape reference while preserving the data calculations.
    fig, ax = plt.subplots(figsize=(16.5, 7), dpi=192)
    k = min(PLOT_FIRST_K, E_b.shape[1], E_a.shape[1])
    comps = np.arange(1, k + 1)
    mean_b, std_b = compute_mean_std(E_b, k)
    mean_a, std_a = compute_mean_std(E_a, k)
    offset = 0.12

    ax.axhline(0.0, color="#B9BDC1", linestyle=":", linewidth=2.5, zorder=1)
    ax.errorbar(
        comps - offset, mean_b, yerr=std_b,
        fmt="o", markersize=19, elinewidth=2.7, capsize=7,
        capthick=2.7, color=GROUP_COLORS["Between"]["before"],
        markerfacecolor=GROUP_COLORS["Between"]["before"],
        markeredgecolor=GROUP_COLORS["Between"]["before"],
        linestyle="none", zorder=3, label="Control",
    )
    ax.errorbar(
        comps + offset, mean_a, yerr=std_a,
        fmt="D", markersize=19, elinewidth=2.7, capsize=7,
        capthick=2.7, color=GROUP_COLORS["Between"]["after"],
        markerfacecolor=GROUP_COLORS["Between"]["after"],
        markeredgecolor=GROUP_COLORS["Between"]["after"],
        linestyle="none", zorder=3, label="TMS",
    )

    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(-0.4, 0.4)
    ax.set_xticks(comps)
    ax.set_yticks(np.arange(-0.4, 0.401, 0.2))
    ax.set_xlabel("Component", fontsize=38, labelpad=12)
    ax.set_ylabel("Δ Explained variance", fontsize=38, labelpad=18)
    ax.tick_params(axis="both", labelsize=34, width=2.5, length=9, pad=5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(2.5)
    ax.spines["bottom"].set_linewidth(2.5)

    # Use marker-only legend handles, as in the reference (no error-bar glyphs).
    legend_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none",
            markerfacecolor=GROUP_COLORS["Between"]["before"],
            markeredgecolor=GROUP_COLORS["Between"]["before"],
            markersize=19, label="Control",
        ),
        Line2D(
            [0], [0], marker="D", linestyle="none",
            markerfacecolor=GROUP_COLORS["Between"]["after"],
            markeredgecolor=GROUP_COLORS["Between"]["after"],
            markersize=19, label="TMS",
        ),
    ]
    ax.legend(
        handles=legend_handles, loc="upper right", bbox_to_anchor=(1.0, 0.99),
        frameon=False, ncol=2, fontsize=36, handletextpad=0.5, columnspacing=1.2,
        borderaxespad=0.3,
    )
    # The reference reserves more whitespace above the data and places the
    # legend near the upper-right of the plotting area.
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.22, top=0.90)
    save_figure_multi(fig, out_base, png_dpi=192, tight=False, pad_inches=0)
    plt.close(fig)

# =========================================================
# Combined 1x2 plot
# =========================================================
def save_combined_1x2_plot(data_dict, out_base: Path, y_max):
    FIG_W = 4.0
    FIG_H = 3.5
    DPI = 600

    rc = {
        "font.family": "Arial",
        "font.size": font_sizes["global"],
        "axes.titlesize": font_sizes["title"],
        "axes.labelsize": font_sizes["axis_label"],
        "xtick.labelsize": font_sizes["tick_label"],
        "ytick.labelsize": font_sizes["tick_label"],
        "legend.fontsize": font_sizes["legend_item"],
        "figure.titlesize": font_sizes["title"],
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }

    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H), dpi=DPI, sharey=True)

        layout = get_bottom_layout_params(FIG_H)

        fig.subplots_adjust(
            left=0.13,
            right=0.985,
            top=0.80,
            bottom=layout["bottom_frac"],
            wspace=0.38
        )

        for ax, group_name in zip(axes, GROUPS):
            E_b, E_a = data_dict[group_name]
            point_error_plot(ax, E_b, E_a, PLOT_FIRST_K, group_name)

            ax.set_title(f"{group_name}", pad=4, fontsize=font_sizes["title"])
            ax.set_xlim(0.5, min(PLOT_FIRST_K, E_b.shape[1], E_a.shape[1]) + 0.5)
            ax.set_ylim(0, y_max)
            ax.set_xlabel("")

            if ax is axes[0]:
                ax.set_ylabel("Explained (λ / Σλ)", fontsize=font_sizes["axis_label"])
            else:
                ax.set_ylabel("")

            ax.tick_params(
                axis="x",
                labelsize=font_sizes["tick_label"],
                pad=1,
                width=0.8,
                length=3
            )
            ax.tick_params(
                axis="y",
                labelsize=font_sizes["tick_label"],
                pad=1,
                width=0.8,
                length=3
            )

        fig.canvas.draw()

        left_bbox = axes[0].get_position()
        right_bbox = axes[-1].get_position()
        x_center = (left_bbox.x0 + right_bbox.x1) / 2.0

        fig.legend(
            handles=COMBINED_LEGEND_HANDLES,
            loc="center",
            bbox_to_anchor=(x_center, layout["legend_y_frac"]),
            bbox_transform=fig.transFigure,
            ncol=2,
            frameon=False,
            fontsize=font_sizes["legend_item"],
            handlelength=1.2,
            columnspacing=1.0,
            handletextpad=0.4,
            borderaxespad=0.0
        )

        fig.text(
            x_center,
            layout["xlabel_y_frac"],
            "Component",
            ha="center",
            va="center",
            fontsize=font_sizes["axis_label"]
        )

        save_figure_multi(fig, out_base)
        plt.close(fig)


# =========================================================
# Main
# =========================================================
def main():
    control_change, control_ids = load_paired_change("Control")
    tms_change, tms_ids = load_paired_change("TMS")

    # Preserve mean +/- SD and the existing independent change-score data.
    y_max = 0.5
    out_base = OUT_DIR_SINGLE / "variance_explained_change_Control_vs_TMS"
    save_single_plot("Between", control_change, tms_change, out_base, y_max)


if __name__ == "__main__":
    main()
