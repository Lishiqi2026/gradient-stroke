#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Standalone network-metric plotting.
from pathlib import Path

import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

try:
    from scipy.stats import ttest_ind
except Exception as e:
    raise ImportError(
        "scipy is required for paired t-test.\n"
        "Install with: pip install scipy\n"
        f"Original error: {e}"
    )


# =========================================================
# Font / style
# =========================================================
font_sizes = {
    # Keep the colour legend one point larger than the other annotations.
    "global": 7.0,
    "legend_item": 7.0,
    "axis_label": 7.0,
    "tick_label": 7.0,
    "title": 7.0,
    "star": 7.0,
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
BASE_DATA_DIR = r"F:\lishiqi\预处理数据\cortex\duiqi（18715，5）"
OUTPUT_DIR = r"F:\lishiqi\组间比较\e"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_DIR = Path(r"D:\Codex_files\Control_TMS_main_figure\source_data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

LABELS_PATH = r"F:\lishiqi\结果目录\皮层×皮层\梯度\7net.npy"

DATA_DIRS = {
    "Control_after": os.path.join(BASE_DATA_DIR, "Control", "after"),
    "Control_before": os.path.join(BASE_DATA_DIR, "Control", "before"),
    "TMS_after": os.path.join(BASE_DATA_DIR, "TMS", "after"),
    "TMS_before": os.path.join(BASE_DATA_DIR, "TMS", "before"),
}

PAIRS = [
    ("Control_after", "Control_before"),
    ("TMS_after", "TMS_before"),
]

GRADIENTS = {
    "G1": 0,
    "G2": 1,
}

METRICS = ["Dispersion", "Segregation", "betweenDisp"]

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

NET_ID_TO_ABBR = {
    1: "VIS",
    2: "SMN",
    3: "DAN",
    4: "VAN",
    5: "LIM",
    6: "CON",
    7: "DMN",
}

ORDER_IDS = [1, 2, 3, 4, 5, 6, 7]

ALPHA_1 = 0.05
ALPHA_2 = 0.01

# Related hue families, distinct from both the unchanged ROI bars and ridges.
# These are metric categories, not the group/direction codes in adjacent panels.
GRADIENT_METRIC_COLORS = {
    "G1": {
        "Dispersion": "#72B7A6",
        "Segregation": "#B5C884",
        "betweenDisp": "#D598B2",
    },
    "G2": {
        "Dispersion": "#9296C9",
        "Segregation": "#DBC58E",
        "betweenDisp": "#D99B86",
    },
}
BAR_ALPHA = 0.96  # Luminous pastel colours without washed-out tiny bars.

# =========================================================
# Helpers
# =========================================================
def load_gradient(arr, idx):
    """
    Extract gradient G(idx+1) from a .npy array.
    Output shape must be (18715,)
    """
    if arr.ndim == 1:
        if idx != 0:
            raise ValueError("This file is 1D (single gradient). Cannot extract G2.")
        if arr.shape[0] != 18715:
            raise ValueError(f"1D array length {arr.shape[0]} != 18715")
        return arr.astype(float)

    if arr.ndim == 2:
        if arr.shape[0] == 18715:
            if idx >= arr.shape[1]:
                raise ValueError(f"Array has only {arr.shape[1]} gradients, cannot get idx={idx}")
            return arr[:, idx].astype(float)
        elif arr.shape[1] == 18715:
            if idx >= arr.shape[0]:
                raise ValueError(f"Array has only {arr.shape[0]} gradients, cannot get idx={idx}")
            return arr[idx, :].astype(float)
        else:
            raise ValueError(f"Unexpected 2D shape {arr.shape}, cannot locate 18715 dimension.")

    raise ValueError(f"Unsupported array ndim {arr.ndim}")


def get_subject_id(fname):
    """
    针对你当前文件名：
    PRIVATE_SUBJECT_022-PRIVATE_SUBJECT_046-1_aligned5_masked.npy -> PRIVATE_SUBJECT_022
    PRIVATE_SUBJECT_022-PRIVATE_SUBJECT_047-2_aligned5_masked.npy -> PRIVATE_SUBJECT_022
    """
    base = os.path.splitext(os.path.basename(fname))[0]

    # 先取第一个 '-' 前面的主被试ID
    sid = base.split("-")[0].strip()

    return sid


def compute_metrics_for_g(g, net_idx):
    centroids = {}
    dispersion = {}

    for n in range(1, 8):
        vox = net_idx[n]
        if vox.size == 0:
            centroids[n] = np.nan
            dispersion[n] = np.nan
        else:
            vals = g[vox]
            centroids[n] = np.nanmean(vals)
            dispersion[n] = np.nanstd(vals, ddof=1)

    between_disp = {}
    for n in range(1, 8):
        if np.isnan(centroids[n]):
            between_disp[n] = np.nan
            continue
        diffs = [abs(centroids[n] - centroids[m]) for m in range(1, 8) if m != n]
        between_disp[n] = np.nanmean(diffs) if diffs else np.nan

    eps = 1e-12
    segregation = {}
    for n in range(1, 8):
        if not np.isfinite(between_disp[n]) or not np.isfinite(dispersion[n]):
            segregation[n] = np.nan
        else:
            segregation[n] = between_disp[n] / (dispersion[n] + eps)

    out = {}
    for n in range(1, 8):
        out[n] = {
            "Dispersion": dispersion[n],
            "betweenDisp": between_disp[n],
            "Segregation": segregation[n],
        }
    return out


def load_network_labels():
    """Load the saved vertex labels, or rebuild them from local Schaefer annotations."""
    if os.path.exists(LABELS_PATH):
        labels = load_network_labels()
    else:
        from nibabel.freesurfer.io import read_annot
        from nilearn.datasets import fetch_atlas_surf_destrieux

        print(f"[INFO] Missing {LABELS_PATH}; rebuilding network labels from local Schaefer annotations...")
        network_codes = {
            "Vis": 1,
            "SomMot": 2,
            "DorsAttn": 3,
            "SalVentAttn": 4,
            "Limbic": 5,
            "Cont": 6,
            "Default": 7,
        }

        hemi_codes = []
        for annot_path in (
            r"F:\lishiqi\预处理数据\cortex\lh.Schaefer2018_400Parcels_7Networks_order.annot",
            r"F:\lishiqi\预处理数据\cortex\rh.Schaefer2018_400Parcels_7Networks_order.annot",
        ):
            annot_labels, ctab, names = read_annot(annot_path)
            names = [n.decode("utf-8") if isinstance(n, (bytes, bytearray)) else str(n) for n in names]
            if int(np.max(annot_labels)) < len(names):
                value_to_name = {i: name for i, name in enumerate(names)}
            elif ctab is not None and ctab.shape[1] >= 5:
                value_to_name = {int(ctab[i, 4]): name for i, name in enumerate(names)}
            else:
                raise RuntimeError(f"Cannot map annotation labels from {annot_path}")

            codes = np.zeros(annot_labels.shape, dtype=np.int8)
            for value in np.unique(annot_labels):
                label_name = value_to_name.get(int(value), "")
                for network, code in network_codes.items():
                    if network in label_name:
                        codes[annot_labels == value] = code
                        break
            hemi_codes.append(codes)

        full_labels = np.concatenate(hemi_codes)
        destrieux = fetch_atlas_surf_destrieux()
        excluded_ids = [
            destrieux.labels.index("Unknown"),
            destrieux.labels.index("Medial_wall"),
        ]
        excluded = np.concatenate([
            np.isin(destrieux.map_left, excluded_ids),
            np.isin(destrieux.map_right, excluded_ids),
        ])
        if excluded.shape != full_labels.shape:
            raise ValueError(
                f"Annotation/mask size mismatch: {full_labels.shape} vs {excluded.shape}"
            )
        labels = full_labels[~excluded]

    if labels.shape != (18715,):
        raise ValueError(f"labels shape {labels.shape} != (18715,)")
    if not np.all(np.isin(labels, [0, 1, 2, 3, 4, 5, 6, 7])):
        raise ValueError("Network labels must contain only values 0 through 7")
    print("[INFO] Network labels ready:", dict(zip(*np.unique(labels, return_counts=True))))
    return labels

def build_subject_metric_table():
    labels = load_network_labels()
    if labels.shape != (18715,):
        raise ValueError(f"labels shape {labels.shape} != (18715,)")

    valid_mask = np.isin(labels, [1, 2, 3, 4, 5, 6, 7])
    net_idx = {n: np.where((labels == n) & valid_mask)[0] for n in range(1, 8)}

    all_rows_long = []
    errors = []

    for group, d in DATA_DIRS.items():
        if not os.path.exists(d):
            raise FileNotFoundError(f"Directory not found: {d}")

        files = sorted([f for f in os.listdir(d) if f.endswith(".npy")])
        if not files:
            raise RuntimeError(f"No .npy files found in {d}")

        for fname in files:
            fpath = os.path.join(d, fname)
            sid = get_subject_id(fname)

            try:
                arr = np.load(fpath)
            except Exception as e:
                errors.append({
                    "group": group,
                    "subject_file": fname,
                    "error": f"load npy failed: {e}"
                })
                continue

            for gname, gidx in GRADIENTS.items():
                try:
                    g = load_gradient(arr, gidx)
                    m = compute_metrics_for_g(g, net_idx)

                    for n in range(1, 8):
                        all_rows_long.append({
                            "group": group,
                            "subject_id": sid,
                            "subject_file": fname,
                            "gradient": gname,
                            "network": n,
                            "network_name": NET_ID_TO_ABBR[n],
                            "Dispersion": m[n]["Dispersion"],
                            "Segregation": m[n]["Segregation"],
                            "betweenDisp": m[n]["betweenDisp"],
                        })

                except Exception as e:
                    errors.append({
                        "group": group,
                        "subject_file": fname,
                        "gradient": gname,
                        "error": str(e)
                    })

    df_long = pd.DataFrame(all_rows_long)
    df_err = pd.DataFrame(errors)

    df_long.to_csv(
        str(DATA_DIR / "all_groups_metrics_long_G1G2_for_bar.csv"),
        index=False,
        encoding="utf-8-sig"
    )
    if len(df_err) > 0:
        df_err.to_csv(
            str(DATA_DIR / "read_errors_G1G2_for_bar.csv"),
            index=False,
            encoding="utf-8-sig"
        )

    return df_long, df_err


def compute_between_group_results(df_long):
    pair_plot_data_by_g = {}
    results_rows = []

    for gname in GRADIENTS.keys():
        df_g = df_long[df_long["gradient"] == gname].copy()
        if df_g.empty:
            print(f"[WARN] No data for {gname}, skip.")
            continue

        pair_plot_data = {}

        key = "TMS_change_vs_Control_change"
        t_map = {(n, m): np.nan for n in ORDER_IDS for m in METRICS}
        p_map = {(n, m): np.nan for n in ORDER_IDS for m in METRICS}
        n_map = {(n, m): (0, 0) for n in ORDER_IDS for m in METRICS}

        for n in ORDER_IDS:
            changes = {}
            for group in ("Control", "TMS"):
                after = df_g[(df_g["group"] == f"{group}_after") & (df_g["network"] == n)][["subject_id"] + METRICS]
                before = df_g[(df_g["group"] == f"{group}_before") & (df_g["network"] == n)][["subject_id"] + METRICS]
                merged = pd.merge(after, before, on="subject_id", suffixes=("_after", "_before"))
                changes[group] = {
                    m: pd.to_numeric(merged[f"{m}_after"], errors="coerce").to_numpy(dtype=float)
                       - pd.to_numeric(merged[f"{m}_before"], errors="coerce").to_numpy(dtype=float)
                    for m in METRICS
                }

            for m in METRICS:
                control = changes["Control"][m]
                tms = changes["TMS"][m]
                control = control[np.isfinite(control)]
                tms = tms[np.isfinite(tms)]
                if control.size < 2 or tms.size < 2:
                    t_val, p_val = np.nan, np.nan
                else:
                    t_val, p_val = ttest_ind(tms, control, equal_var=False, nan_policy="omit")

                t_map[(n, m)] = t_val
                p_map[(n, m)] = p_val
                n_map[(n, m)] = (int(control.size), int(tms.size))
                results_rows.append({
                    "gradient": gname, "comparison": key,
                    "network": n, "network_name": NET_ID_TO_ABBR[n], "metric": m,
                    "n_control": int(control.size), "n_tms": int(tms.size),
                    "t_tms_minus_control": t_val, "p_two_sided": p_val,
                    "neglog10p": (-np.log10(max(p_val, 1e-300))) if np.isfinite(p_val) else np.nan,
                    "sig_p005": bool(np.isfinite(p_val) and p_val < ALPHA_1),
                    "sig_p001": bool(np.isfinite(p_val) and p_val < ALPHA_2),
                })

        pair_plot_data[key] = {
            "t": {m: np.array([t_map[(n, m)] for n in ORDER_IDS]) for m in METRICS},
            "p": {m: np.array([p_map[(n, m)] for n in ORDER_IDS]) for m in METRICS},
            "n": {m: np.array([n_map[(n, m)] for n in ORDER_IDS], dtype=object) for m in METRICS},
            "after": "TMS_change", "before": "Control_change"
        }
        print(f"[INFO] {gname} | {key}: Welch independent-samples tests completed")

        pair_plot_data_by_g[gname] = pair_plot_data

    df_res_all = pd.DataFrame(results_rows)
    df_res_all.to_csv(
        str(DATA_DIR / "between_group_change_results_long_G1G2_for_bar.csv"),
        index=False,
        encoding="utf-8-sig"
    )
    return pair_plot_data_by_g, df_res_all


# =========================================================
# Plot bar only
# =========================================================
def add_sig_text_barh(ax, vals, pvals, y_positions, x_pad):
    for xv, pv, yy in zip(vals, pvals, y_positions):
        if not np.isfinite(xv) or not np.isfinite(pv):
            continue

        if pv < ALPHA_2:
            star = "**"
        elif pv < ALPHA_1:
            star = "*"
        else:
            star = None

        if star is not None:
            ax.text(
                xv + x_pad, yy, star,
                ha="left", va="center",
                fontsize=font_sizes["star"],
                fontweight="bold",
                color="black"
            )


def plot_bar_for_pair_gradient(pair_key,gname,pair_plot_data,out_dir):
    # Reference style: three horizontal metric bars per network, a dashed
    # significance threshold, and a centered three-item legend below the axis.
    colors = GRADIENT_METRIC_COLORS[gname]
    metric_order = ["Dispersion", "Segregation", "betweenDisp"]
    metric_labels = {
        "Dispersion": "Disp.",
        "Segregation": "Segr.",
        "betweenDisp": "Inter-net",
    }
    offsets = {"Dispersion": 0.25, "Segregation": 0.0, "betweenDisp": -0.25}
    network_y = np.arange(len(NET_ORDER), dtype=float)
    values = {}
    pvals = {}
    for metric in metric_order:
        p = np.asarray(pair_plot_data["p"][metric], dtype=float)
        pvals[metric] = p
        values[metric] = -np.log10(np.clip(p, 1e-300, 1.0))
        values[metric] = np.nan_to_num(values[metric], nan=0.0, posinf=300.0, neginf=0.0)

    max_value = max((float(np.max(v)) for v in values.values()), default=0.0)
    x_max = max(2.05, max_value + 0.38)

    fig = plt.figure(figsize=(8.85, 8.7), dpi=300)
    ax = fig.add_axes([0.22, 0.25, 0.61, 0.64])
    for metric in metric_order:
        y = network_y + offsets[metric]
        bars = ax.barh(
            y, values[metric], height=0.22,
            color=colors[metric], alpha=BAR_ALPHA, edgecolor="none",
            label=metric_labels[metric], zorder=2,
        )
        p = pvals[metric]
        for bar, p_value in zip(bars, p):
            if not np.isfinite(p_value) or p_value >= ALPHA_1:
                continue
            star = "**" if p_value < ALPHA_2 else "*"
            ax.text(
                bar.get_width() + 0.045,
                bar.get_y() + bar.get_height() / 2,
                star, ha="left", va="center", fontsize=27,
                fontweight="bold", color="black", clip_on=False,
            )

    threshold = -np.log10(ALPHA_1)
    ax.axvline(threshold, color="black", linestyle="--", linewidth=2.4, zorder=1)
    ax.set_xlim(0, x_max)
    ax.set_xticks([0, 1, 2])
    ax.set_yticks(network_y, [NET_ABBR[net] for net in NET_ORDER])
    ax.set_ylim(len(NET_ORDER) - 0.5, -0.5)
    ax.tick_params(axis="y", labelsize=36, length=0, pad=12)
    ax.tick_params(axis="x", labelsize=31, width=2.2, length=10, pad=4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(2.3)
    ax.spines["bottom"].set_linewidth(2.3)
    ax.set_xlabel("")
    ax.set_ylabel("")

    handles = [
        matplotlib.patches.Patch(facecolor=colors[m], edgecolor="none", label=metric_labels[m])
        for m in metric_order
    ]
    fig.legend(
        handles=handles, loc="center", bbox_to_anchor=(0.502, 0.117),
        ncol=3, frameon=False, fontsize=36, handlelength=0.55,
        handleheight=0.78, handletextpad=0.25, columnspacing=0.14,
        borderaxespad=0.0,
    )

    os.makedirs(out_dir, exist_ok=True)
    out_base = os.path.join(out_dir, f"bar_only_{pair_key}_{gname}")
    fig.savefig(out_base + ".png", dpi=600, facecolor="white", bbox_inches="tight", pad_inches=0.10)
    fig.savefig(out_base + ".pdf", facecolor="white", bbox_inches="tight", pad_inches=0.10)
    fig.savefig(out_base + ".svg", facecolor="white", bbox_inches="tight", pad_inches=0.10)
    print(f"[SAVE] {out_base}.png/.pdf/.svg")
    plt.close(fig)

# =========================================================
# Main
# =========================================================
def main():
    print("[STEP 1] Build subject-level metric table...")
    df_long, df_err = build_subject_metric_table()

    print("[STEP 2] Compute Control-vs-TMS change-score results...")
    pair_plot_data_by_g, df_res = compute_between_group_results(df_long)


    print("[STEP 3] Draw bar-only figures...")
    for gname in GRADIENTS.keys():
        if gname not in pair_plot_data_by_g:
            continue

        pair_key = "TMS_change_vs_Control_change"
        plot_bar_for_pair_gradient(
            pair_key=pair_key,
            gname=gname,
            pair_plot_data=pair_plot_data_by_g[gname][pair_key],
            out_dir=OUTPUT_DIR
        )

    print("All done.")
    print(f"[SAVE] output dir: {OUTPUT_DIR}")
    if len(df_err) > 0:
        print("[NOTE] Some files had read/gradient-extraction errors. Check:")
        print(str(DATA_DIR / "read_errors_G1G2_for_bar.csv"))


if __name__ == "__main__":
    main()
