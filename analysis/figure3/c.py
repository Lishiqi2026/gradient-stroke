#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Standalone ridgeline plotting.
from pathlib import Path

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba

from nilearn.surface import SurfaceImage
from nilearn.datasets import fetch_atlas_surf_destrieux, load_fsaverage
from nibabel.freesurfer.io import read_annot


# =========================================================
# Font / style
# =========================================================
font_sizes = {
    "global": 12.0,
    "legend_item": 16.0,
    "axis_label": 18.0,
    "tick_label": 18.0,
    "title": 18.0,
    "star": 18.0,
}

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = font_sizes["global"]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"


# =========================================================
# Paths
# =========================================================
OUTPUT_DIR = r"F:\lishiqi\组间比较\c"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_DIR = Path(r"D:\Codex_files\Control_TMS_main_figure\source_data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

base_dir = r"F:\lishiqi\预处理数据\cortex\组水平（18715）"
grad_paths = {
    "Control_post": os.path.join(base_dir, "Control_post.npy"),   # After
    "Control_pre":  os.path.join(base_dir, "Control_pre.npy"),    # Before
    "TMS_post":     os.path.join(base_dir, "TMS_post.npy"),
    "TMS_pre":      os.path.join(base_dir, "TMS_pre.npy"),
}

lh_annot_path = r"F:\lishiqi\预处理数据\cortex\lh.Schaefer2018_400Parcels_7Networks_order.annot"
rh_annot_path = r"F:\lishiqi\预处理数据\cortex\rh.Schaefer2018_400Parcels_7Networks_order.annot"

PAIRS = [
    ("TMS_change", "Control_change"),
]

GRADIENTS = {
    "G1": 0,
    "G2": 1,
}

GRADIENT_X_RANGES = {
    "G1": (-0.004, 0.006),
    "G2": (-0.002, 0.004),
}

NET_ORDER = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]

NET_ABBR = {
    "Vis": "VIS",
    "SomMot": "SMN",
    "DorsAttn": "DAN",
    "SalVentAttn": "VAN",
    "Limbic": "LIM",
    "Cont": "CON",
    "Default": "DMN",
}

# Harmonize with the unchanged ROI bars without reusing their exact colours.
# G1: rose / fresh green; G2: apricot / lavender. Keep a local group legend.
GRADIENT_RIDGE_COLORS = {
    "G1": {"before": "#CF88B5", "after": "#9BC46A"},
    "G2": {"before": "#DFAC75", "after": "#A494CC"},
}
FILL_ALPHA = 0.50      # Airy translucent fills; solid outlines stay readable.
LEGEND_ALPHA = FILL_ALPHA
RIDGE_LINEWIDTH = 1.05
BASELINE_COLOR = "#ADB3B8"
LEGEND_MARKER_SIZE = 9.0  # square legend block size (pt)


# =========================================================
# Helpers
# =========================================================
def get_network_from_label(label_name):
    if not isinstance(label_name, str):
        return "Unknown"
    for net in NET_ORDER:
        if net in label_name:
            return net
    return "Unknown"


def load_annot_labels(annot_path):
    labels, ctab, names = read_annot(annot_path)
    names = [n.decode() if isinstance(n, bytes) else n for n in names]

    val_to_name = {}
    max_val = np.max(labels)

    if max_val < len(names):
        val_to_name = {i: n for i, n in enumerate(names)}
    elif ctab is not None and ctab.shape[1] >= 5:
        for i, name in enumerate(names):
            label_id = ctab[i, 4]
            val_to_name[label_id] = name

    vertex_nets = []
    for val in labels:
        name = val_to_name.get(val, "Unknown")
        net_name = get_network_from_label(name)
        vertex_nets.append(net_name)

    return np.array(vertex_nets)


def load_gradient(grad_path):
    return np.load(grad_path)


def prepare_grouplevel_plot_data():
    missing = [k for k, p in grad_paths.items() if not os.path.exists(p)]
    if len(missing) > 0:
        raise FileNotFoundError(
            "These gradient files are missing:\n"
            + "\n".join([f"  {k}: {grad_paths[k]}" for k in missing])
        )

    print("[INFO] Setting up fsaverage5 mask...")
    fsavg_mesh = load_fsaverage(mesh="fsaverage5")
    destrieux = fetch_atlas_surf_destrieux()

    destrieux_atlas = SurfaceImage(
        mesh=fsavg_mesh["inflated"],
        data={"left": destrieux.map_left, "right": destrieux.map_right}
    )

    excluded_labels = [
        destrieux.labels.index("Unknown"),
        destrieux.labels.index("Medial_wall")
    ]
    is_excluded_l = np.isin(destrieux_atlas.data.parts["left"], excluded_labels)
    is_excluded_r = np.isin(destrieux_atlas.data.parts["right"], excluded_labels)
    rows_to_zero = np.concatenate([is_excluded_l, is_excluded_r])

    print("[INFO] Loading Schaefer 7-network labels...")
    lh_nets = load_annot_labels(lh_annot_path)
    rh_nets = load_annot_labels(rh_annot_path)
    full_nets = np.concatenate([lh_nets, rh_nets])[~rows_to_zero]

    print("[INFO] Loading group-level gradients...")
    emb_dict = {}
    for gname, gpath in grad_paths.items():
        print(f"[INFO] Loading gradient: {gpath}")
        emb_dict[gname] = load_gradient(gpath)

    # Between-group intervention comparison: each group contributes Post - Pre.
    emb_dict = {
        "Control_change": emb_dict["Control_post"] - emb_dict["Control_pre"],
        "TMS_change": emb_dict["TMS_post"] - emb_dict["TMS_pre"],
    }

    rows = []
    for i in range(len(full_nets)):
        net = full_nets[i]
        if net not in NET_ORDER:
            continue

        for gname, emb in emb_dict.items():
            g1 = emb[i, 0]
            g2 = emb[i, 1]

            if np.isfinite(g1):
                rows.append({
                    "Network": net,
                    "Group": gname,
                    "Metric": "G1",
                    "Value": g1
                })
            if np.isfinite(g2):
                rows.append({
                    "Network": net,
                    "Group": gname,
                    "Metric": "G2",
                    "Value": g2
                })

    plot_data = pd.DataFrame(rows)
    plot_data["Network"] = pd.Categorical(
        plot_data["Network"],
        categories=NET_ORDER,
        ordered=True
    )
    return plot_data


def _gaussian_kde_1d(values, x_grid, bandwidth=None):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return np.zeros_like(x_grid, dtype=float)

    n = v.size
    std = np.std(v, ddof=1)

    if bandwidth is None:
        bw = 1.06 * std * (n ** (-1 / 5))
        if not np.isfinite(bw) or bw <= 0:
            bw = (np.nanmax(v) - np.nanmin(v) + 1e-6) / 30.0
    else:
        bw = float(bandwidth)

    diff = (x_grid[:, None] - v[None, :]) / bw
    dens = np.exp(-0.5 * diff * diff).sum(axis=1) / (n * bw * np.sqrt(2 * np.pi))
    return dens


# =========================================================
# Plot ridge only
# =========================================================
def prepare_ridge_payload(gname, plot_data):
    xmin,xmax=GRADIENT_X_RANGES[gname]
    grid=np.linspace(xmin,xmax,256)
    payload={"grid":grid,"Control":[],"TMS":[]}
    for net in NET_ORDER:
        dens={}
        for group in ("Control","TMS"):
            vals=plot_data[(plot_data["Metric"]==gname)&(plot_data["Group"]==group+"_change")&(plot_data["Network"]==net)]["Value"].to_numpy()
            dens[group]=_gaussian_kde_1d(vals,grid)
        dmax=max(dens["Control"].max(),dens["TMS"].max())
        for group in dens:
            payload[group].append(dens[group]/dmax*.60 if dmax>0 else dens[group])
    return {k:np.asarray(v) for k,v in payload.items()}


def plot_ridge_for_pair_gradient(pair_key,gname,after_g,before_g,plot_data,out_dir):
    payload=prepare_ridge_payload(gname,plot_data)
    colors=GRADIENT_RIDGE_COLORS[gname]

    # Match the supplied reference images while retaining the same KDE data,
    # network order, colors, and group comparisons.
    fig = plt.figure(figsize=(4.12, 3.16), dpi=100)
    ax = fig.add_axes([0.155, 0.25, 0.75, 0.73])
    grid = payload["grid"]
    for i in range(len(NET_ORDER)):
        base = len(NET_ORDER) - 1 - i
        for group, color in (("Control", colors["before"]), ("TMS", colors["after"])):
            values = payload[group][i]
            ax.fill_between(grid, base, base + values, color=color, alpha=FILL_ALPHA, lw=0)
            ax.plot(grid, base + values, color=color, lw=RIDGE_LINEWIDTH)
        ax.hlines(base, grid[0], grid[-1], color="#B8BEC2", lw=0.9, zorder=0)

    xmin, xmax = GRADIENT_X_RANGES[gname]
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.05, 6.7)
    ax.set_yticks(np.arange(len(NET_ORDER) - 1, -1, -1), [NET_ABBR[net] for net in NET_ORDER])
    ax.set_xticks([xmin, 0.001, xmax], [f"{v:.3f}" for v in (xmin, 0.001, xmax)])
    ax.tick_params(axis="y", labelsize=18, length=0, pad=4)
    ax.tick_params(axis="x", labelsize=18, width=1.0, length=4, pad=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.set_xlabel("")
    ax.set_ylabel("")

    legend_handles = [
        matplotlib.patches.Patch(facecolor=colors["before"], edgecolor=colors["before"], alpha=FILL_ALPHA, label="Control"),
        matplotlib.patches.Patch(facecolor=colors["after"], edgecolor=colors["after"], alpha=FILL_ALPHA, label="TMS"),
    ]
    fig.legend(
        handles=legend_handles, loc="center", bbox_to_anchor=(0.465, 0.055),
        ncol=2, frameon=False, fontsize=19, handlelength=0.6,
        handleheight=0.65, handletextpad=0.25, columnspacing=0.6,
    )

    os.makedirs(out_dir, exist_ok=True)
    out_base = os.path.join(out_dir, f"ridge_only_{pair_key}_{gname}")
    fig.savefig(out_base + ".png", dpi=600, facecolor="white")
    fig.savefig(out_base + ".pdf", facecolor="white")
    fig.savefig(out_base + ".svg", facecolor="white")
    print(f"[SAVE] {out_base}.png/.pdf/.svg")
    np.savez_compressed(DATA_DIR / f"ridge_{gname}_density.npz", **payload)
    plt.close(fig)

# =========================================================
# Main
# =========================================================
def main():
    print("[STEP 1] Build group-level ridgeline table...")
    plot_data = prepare_grouplevel_plot_data()
    plot_data.to_csv(DATA_DIR / "ridge_vertex_changes.csv",index=False)

    print("[STEP 2] Draw ridge-only figures...")
    for gname in GRADIENTS.keys():
        for after_g, before_g in PAIRS:
            pair_key = f"{after_g}_vs_{before_g}"
            plot_ridge_for_pair_gradient(
                pair_key=pair_key,
                gname=gname,
                after_g=after_g,
                before_g=before_g,
                plot_data=plot_data,
                out_dir=OUTPUT_DIR
            )

    print("All done.")
    print(f"[SAVE] output dir: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
