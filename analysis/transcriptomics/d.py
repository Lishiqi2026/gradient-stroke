# -*- coding: utf-8 -*-
"""Metascape Top-20 enrichment bubble plot for the G1 between-group analysis.

Comparison represented by the uploaded gene list:
    TMS(Post - Pre) - Control(Post - Pre), Gradient 1

The values below are the 20 representative clusters in Table 3 of Metascape
report ``tycrmox7q``.  They are embedded so the figure remains reproducible
after the temporary Metascape report expires.
"""

from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd


# =========================
# Journal-style global setup
# =========================
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.0,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)

OUT_DIR = Path(r"F:\lishiqi\组间比较\基因\d")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PREFIX = "TMS_change_vs_Control_change_G1"

# 2 x 2 论文总图中的统一单面板规格（约 90 x 83 mm）
FIGSIZE = (3.55, 2.75)
OUTPUT_DPI = 600

# GO/Pathway, source, description, Count, Percentage, Log10(P), Log10(q)
METASCAPE_TOP20 = [
    ("GO:0034330", "GO Biological Processes", "cell junction organization", 54, 5.85, -15.08, -10.76),
    ("GO:0007267", "GO Biological Processes", "cell-cell signaling", 63, 6.83, -13.24, -9.39),
    ("GO:0048666", "GO Biological Processes", "neuron development", 60, 6.50, -13.04, -9.31),
    ("GO:0003013", "GO Biological Processes", "circulatory system process", 48, 5.20, -12.89, -9.27),
    ("hsa05200", "KEGG Pathway", "Pathways in cancer", 49, 5.31, -11.24, -7.83),
    ("GO:0141124", "GO Biological Processes", "intracellular signaling cassette", 58, 6.28, -10.67, -7.39),
    ("R-HSA-372790", "Reactome Gene Sets", "Signaling by GPCR", 57, 6.18, -10.60, -7.36),
    ("GO:0040008", "GO Biological Processes", "regulation of growth", 45, 4.88, -10.54, -7.35),
    ("GO:0016477", "GO Biological Processes", "cell migration", 58, 6.28, -9.98, -6.89),
    ("GO:0043269", "GO Biological Processes", "regulation of monoatomic ion transport", 38, 4.12, -9.84, -6.77),
    ("GO:0055080", "GO Biological Processes", "monoatomic cation homeostasis", 42, 4.55, -9.72, -6.69),
    ("WP5083", "WikiPathways", "Neuroinflammation and glutamatergic signaling", 20, 2.17, -8.08, -5.34),
    ("GO:0043408", "GO Biological Processes", "regulation of MAPK cascade", 47, 5.09, -7.90, -5.21),
    ("GO:0050804", "GO Biological Processes", "modulation of chemical synaptic transmission", 40, 4.33, -7.89, -5.21),
    ("GO:0030029", "GO Biological Processes", "actin filament-based process", 45, 4.88, -7.82, -5.17),
    ("R-HSA-382551", "Reactome Gene Sets", "Transport of small molecules", 51, 5.53, -7.70, -5.07),
    ("WP2431", "WikiPathways", "Spinal cord injury", 18, 1.95, -7.67, -5.06),
    ("GO:0007169", "GO Biological Processes", "cell surface receptor protein tyrosine kinase signaling pathway", 34, 3.68, -7.65, -5.05),
    ("WP5087", "WikiPathways", "Pleural mesothelioma", 37, 4.01, -7.64, -5.05),
    ("GO:0060322", "GO Biological Processes", "head development", 42, 4.55, -7.58, -5.00),
]

COLUMNS = [
    "Term",
    "Category",
    "Description",
    "Count",
    "Percent",
    "Log10(P)",
    "Log10(q)",
]


def bubble_area(count):
    """Map gene count to scatter area in points squared."""
    count = np.asarray(count, dtype=float)
    # 当前面板中相邻通路的中心间距约为 7.3 pt；将最大气泡直径
    # 控制在约 6.3 pt 以内，避免相邻行的圆圈发生重叠。
    # 图例复用同一映射，因此仍能准确表达 Count 的相对大小。
    return 10.0 + (count - 15.0) * 0.6


def wrap_label(label, width=100):
    return textwrap.fill(str(label), width=width, break_long_words=False)


def draw_top20():
    df = pd.DataFrame(METASCAPE_TOP20, columns=COLUMNS)
    df = df.sort_values("Log10(P)", ascending=True).head(20).reset_index(drop=True)

    # Keep an auditable machine-readable copy of the exact values plotted.
    data_path = OUT_DIR / f"{PREFIX}_Metascape_Top20_source.csv"
    df.to_csv(data_path, index=False, encoding="utf-8-sig")

    counts = df["Count"].to_numpy(dtype=float)
    y = np.arange(len(df))

    # Reversed, high-luminance paper-style scale requested by the user.
    # The hues follow the supplied reference: coral red, apricot orange,
    # cream yellow, mint green, cyan blue and lavender purple.
    # Low-to-high significance: red/orange -> yellow -> green/teal -> blue/purple.
    cmap = LinearSegmentedColormap.from_list(
        "reference_rank",
        [
            "#A74C5B", "#BD5561", "#D36A68", "#D95B4E",
            "#ED8F68", "#F3A976", "#F7C187", "#F8D699",
            "#F9F4C1", "#E9F0B6", "#C0DFB2", "#8FC9B2",
            "#6B97C0", "#7A6BAA",
        ],
        N=256,
    )
    # The reference uses a smooth ordinal color progression down the ranked
    # pathway list. Log10(P) remains quantitatively encoded by x position.
    color_rank = np.linspace(1.0, 0.0, len(df))
    norm = Normalize(vmin=0.0, vmax=1.0)

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=300)
    # 长通路名称需要更宽的左侧标签区；右侧数值轴仍保持论文式窄纵列。
    fig.subplots_adjust(left=0.63, right=0.97, top=0.84, bottom=0.14)

    ax.scatter(
        df["Log10(P)"],
        y,
        s=bubble_area(counts),
        c=color_rank,
        cmap=cmap,
        norm=norm,
        alpha=1.0,
        edgecolors="white",
        linewidths=0.35,
        zorder=3,
    )

    ax.set_yticks(y)
    ax.set_yticklabels([wrap_label(v) for v in df["Description"]], fontsize=5.3)
    ax.invert_yaxis()

    # Match the supplied reference: increasingly negative LogP values run rightward.
    ax.set_xlim(-7.0, -16.0)
    ax.set_xticks([-8, -10, -12, -14, -16])
    ax.set_xlabel("Log10(P)", fontsize=7.0, fontweight="bold", labelpad=2)
    fig.text(
        0.52,
        0.935,
        "Top-20 Enriched Pathways for Between-group\nG1 Change-related Genes",
        fontsize=7.8,
        fontweight="bold",
        ha="center",
        va="center",
        linespacing=1.05,
    )

    ax.grid(axis="y", linestyle=(0, (1.2, 2.0)), color="#BDBDBD", linewidth=0.45, zorder=0)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", direction="out", width=0.8, length=3.0, labelsize=5.5)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_counts = [20, 40, 60]
    handles = [
        plt.Line2D(
            [],
            [],
            linestyle="",
            marker="o",
            markersize=np.sqrt(bubble_area(value)),
            markerfacecolor="#737373",
            markeredgecolor="white",
            markeredgewidth=0.35,
            alpha=0.92,
            label=str(value),
        )
        for value in legend_counts
    ]
    legend = ax.legend(
        handles=handles,
        title="Count",
        loc="lower right",
        frameon=True,
        borderpad=0.40,
        labelspacing=0.35,
        handletextpad=0.45,
        fontsize=5.5,
        title_fontsize=6.0,
    )
    legend.get_frame().set_edgecolor("#8C8C8C")
    legend.get_frame().set_linewidth(0.6)
    legend.get_frame().set_facecolor("white")

    base = OUT_DIR / f"{PREFIX}_Metascape_Top20_reference_style"
    for suffix in ("png", "pdf", "svg"):
        # 固定画布，保证与另外三个面板导出为同一物理尺寸。
        save_kwargs = {"facecolor": "white"}
        if suffix == "png":
            save_kwargs["dpi"] = OUTPUT_DPI
        fig.savefig(base.with_suffix(f".{suffix}"), **save_kwargs)
        print(f"[OK] Saved: {base.with_suffix(f'.{suffix}')}")

    plt.close(fig)
    print(f"[OK] Saved: {data_path}")


def main():
    print("[INFO] Comparison: TMS(Post-Pre) - Control(Post-Pre), Gradient 1")
    print("[INFO] Metascape report: tycrmox7q")
    print("[INFO] OUT_DIR =", OUT_DIR)
    draw_top20()
    print("[DONE] Figure c finished.")


if __name__ == "__main__":
    main()
