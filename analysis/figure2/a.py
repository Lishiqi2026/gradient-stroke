#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure 2a: within-group longitudinal cortical gradient changes.

Colors encode mean post-minus-pre change. Black parcel outlines encode
Benjamini-Hochberg FDR q < 0.05 within each group-by-gradient map.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, ListedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests
from surfplot import Plot
from neuromaps.datasets import fetch_fslr
from brainspace.datasets import load_parcellation

# -----------------------------
# Paths and analysis settings
# -----------------------------
LABELS_FILE = Path(r"F:\lishiqi\预处理数据\cortex\董分组.xlsx")
# The 400 x 400 matrices define the 19/23 complete-pair cohort. Gradient
# coordinates are read from the corresponding precomputed, aligned 400 x 5
# derivatives so G1/G2 retain the same cross-scan orientation.
CONNECTIVITY_ROOT = Path(r"F:\lishiqi\预处理数据\cortex\parcel(400，400)")
FEATURE_ROOT = Path(r"F:\lishiqi\预处理数据\cortex\duiqi (400, 5)")
OUTPUT_DIR = Path(r"F:\lishiqi\end_picture\Ending\main 2\a")

EXPECTED_N = {"Control": 19, "TMS": 23}
GROUP_ORDER = ["Control", "TMS"]
FEATURE_DIMS = [(0, "G1"), (1, "G2")]
N_ROI = 400
FDR_ALPHA = 0.05

# Higher-contrast, color-vision-friendly diverging palette: blue = decrease,
# warm coral = increase. One shared scale is used across all four maps.
DELTA_CMAP = LinearSegmentedColormap.from_list(
    "nature_blue_neutral_coral",
    ["#244F78", "#6F98B8", "#F4F1EA", "#D77E70", "#A93F45"],
    N=256,
)
ROI_OUTLINE_CMAP = ListedColormap(["#000000"])
SIG_MARK_CMAP = ListedColormap(["#E0A326", "#E0A326"])
FONT_FAMILY = "Arial"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [FONT_FAMILY, "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.unicode_minus": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})


def _find_subject_file(folder: Path, subject_id: str):
    """Find a file whose first hyphen-delimited token exactly matches the ID."""
    hits = [p for p in folder.glob("*.npy") if p.stem.split("-")[0] == subject_id]
    if len(hits) > 1:
        raise RuntimeError(f"Multiple files found for {subject_id} in {folder}: {[p.name for p in hits]}")
    return hits[0] if hits else None


def _read_cohort():
    if not LABELS_FILE.exists():
        raise FileNotFoundError(f"Participant label workbook not found: {LABELS_FILE}")
    labels = pd.read_excel(LABELS_FILE, dtype={"病历号": str})
    required = {"病历号", "Group"}
    missing_columns = required.difference(labels.columns)
    if missing_columns:
        raise ValueError(f"The label workbook is missing columns: {sorted(missing_columns)}")
    labels = labels.loc[:, ["病历号", "姓名", "Group"]].copy()
    labels["病历号"] = labels["病历号"].astype(str).str.replace(r"\.0$", "", regex=True)
    labels["Group"] = labels["Group"].astype(str).str.strip()
    labels = labels[labels["Group"].isin(GROUP_ORDER)].copy()
    if labels["病历号"].duplicated().any():
        dup = labels.loc[labels["病历号"].duplicated(keep=False), "病历号"].tolist()
        raise ValueError(f"Duplicate participant IDs in workbook: {dup}")
    return labels


def _files_by_subject(folder: Path):
    """Index source/derivative files by the first hyphen-delimited ID."""
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing data directory: {folder}")
    indexed = {}
    for path in folder.glob("*.npy"):
        subject_id = path.stem.split("-")[0]
        if subject_id in indexed:
            raise RuntimeError(f"Duplicate files for {subject_id} in {folder}")
        indexed[subject_id] = path
    return indexed


def _load_paired_features(labels, group, feature_dim):
    # Define the analysis cohort from complete 400 x 400 source-matrix pairs.
    matrix_before = _files_by_subject(CONNECTIVITY_ROOT / group / "before")
    matrix_after = _files_by_subject(CONNECTIVITY_ROOT / group / "after")
    if set(matrix_before) != set(matrix_after):
        raise RuntimeError(
            f"Before/after 400x400 subject IDs differ in {group}: "
            f"before-only={sorted(set(matrix_before)-set(matrix_after))}, "
            f"after-only={sorted(set(matrix_after)-set(matrix_before))}"
        )
    expected_n = EXPECTED_N[group]
    if len(matrix_before) != expected_n:
        raise RuntimeError(f"Expected {expected_n} complete matrix pairs for {group}, found {len(matrix_before)}")
    for sid, path in {**matrix_before, **matrix_after}.items():
        shape = np.load(path, mmap_mode="r").shape
        if shape != (N_ROI, N_ROI):
            raise ValueError(f"{path.name}: expected a {N_ROI}x{N_ROI} source matrix, found {shape}")
    label_rows = labels.set_index("病历号")
    bad_ids = [sid for sid in matrix_before if sid not in label_rows.index]
    wrong_group = [sid for sid in matrix_before if sid in label_rows.index and label_rows.loc[sid, "Group"] != group]
    if bad_ids or wrong_group:
        raise RuntimeError(f"Cohort-label mismatch in {group}: IDs absent from workbook={bad_ids}; wrong group={wrong_group}")

    before_dir = FEATURE_ROOT / group / "before"
    after_dir = FEATURE_ROOT / group / "after"
    feature_before = _files_by_subject(before_dir)
    feature_after = _files_by_subject(after_dir)
    if not set(matrix_before).issubset(feature_before) or not set(matrix_after).issubset(feature_after):
        missing_b = sorted(set(matrix_before) - set(feature_before))
        missing_a = sorted(set(matrix_after) - set(feature_after))
        raise FileNotFoundError(f"Aligned 400x5 gradient derivatives missing for {group}: before={missing_b}, after={missing_a}")

    before, after, included, missing = [], [], [], []
    for subject_id in sorted(matrix_before):
        bpath = feature_before[subject_id]
        apath = feature_after[subject_id]
        xb = np.squeeze(np.load(bpath))
        xa = np.squeeze(np.load(apath))
        for phase, arr in [("before", xb), ("after", xa)]:
            if arr.ndim != 2 or arr.shape[0] != N_ROI or arr.shape[1] <= feature_dim:
                raise ValueError(f"{subject_id} {phase}: expected shape (400, >= {feature_dim+1}), found {arr.shape}")
        before.append(xb[:, feature_dim].astype(float, copy=False))
        after.append(xa[:, feature_dim].astype(float, copy=False))
        included.append(subject_id)

    if not before:
        raise RuntimeError(f"No complete paired gradient files found for {group}.")
    print(f"[INFO] {group}: matched {len(included)} complete matrix/gradient pairs.")
    return included, np.stack(before), np.stack(after), missing


def _roi_to_surface(values, atlas):
    if len(values) != N_ROI:
        raise ValueError(f"Expected {N_ROI} ROI values, found {len(values)}")
    out = np.full(atlas.shape, np.nan, dtype=float)
    for roi_idx, value in enumerate(values, start=1):
        out[atlas == roi_idx] = value
    return out


def _save_single_surface(values, q_values, title, output_path, atlas, surf_l, surf_r, color_limit):
    """Save one group-by-gradient map in the four-view format used by c.py."""
    value_surface = _roi_to_surface(values, atlas)
    sig_roi = np.isfinite(q_values) & (q_values < FDR_ALPHA)
    sig_surface = _roi_to_surface(np.where(sig_roi, 1.0, np.nan), atlas)

    plot = Plot(
        surf_l,
        surf_r,
        views=["lateral", "medial"],
        size=(1250, 900),
        zoom=1.80,
        brightness=0.65,
        background=(1, 1, 1),
    )
    plot.add_layer(
        value_surface,
        cmap=DELTA_CMAP,
        color_range=(-color_limit, color_limit),
        cbar=False,
        zero_transparent=False,
    )
    # Every Schaefer parcel boundary is rendered in black.
    plot.add_layer(atlas, cmap=ROI_OUTLINE_CMAP, as_outline=True, cbar=False, alpha=1)
    if np.any(sig_roi):
        # Gold outlines separately identify parcels passing BH-FDR.
        plot.add_layer(
            sig_surface,
            cmap=SIG_MARK_CMAP,
            color_range=(0, 1),
            as_outline=True,
            cbar=False,
            alpha=1,
        )

    fig = plot.build(colorbar=False, scale=(2, 2))
    fig.patch.set_facecolor("white")
    # Add narrow gutters between the four rendered surface views without cropping them.
    image_artist = fig.axes[0].images[0]
    rendered = np.asarray(image_artist.get_array()).copy()
    image_h, image_w = rendered.shape[:2]
    mid_y, mid_x = image_h // 2, image_w // 2
    top_left = rendered[:mid_y, :mid_x]
    top_right = rendered[:mid_y, mid_x:]
    bottom_left = rendered[mid_y:, :mid_x]
    bottom_right = rendered[mid_y:, mid_x:]
    gap_x = max(24, int(image_w * 0.025))
    gap_y = max(24, int(image_h * 0.025))
    padded_shape = (image_h + gap_y, image_w + gap_x) + rendered.shape[2:]
    separated = np.empty(padded_shape, dtype=rendered.dtype)
    if rendered.ndim == 3 and rendered.shape[2] == 4:
        white = 1.0 if np.issubdtype(rendered.dtype, np.floating) else 255
        separated[...] = (white, white, white, 0)
    else:
        separated[...] = 1.0 if np.issubdtype(rendered.dtype, np.floating) else 255
    separated[:mid_y, :mid_x] = top_left
    separated[:mid_y, mid_x + gap_x:] = top_right
    separated[mid_y + gap_y:, :mid_x] = bottom_left
    separated[mid_y + gap_y:, mid_x + gap_x:] = bottom_right
    image_artist.set_data(separated)

    fig.suptitle(
        title,
        x=0.50, y=0.975, fontsize=56, fontname=FONT_FAMILY,
        fontweight="bold", ha="center", va="top",
    )

    sm = ScalarMappable(norm=Normalize(vmin=-color_limit, vmax=color_limit), cmap=DELTA_CMAP)
    sm.set_array([])
    # c.py reference dimensions: 35% of canvas width and a lower placement.
    cax = fig.add_axes([0.325, 0.065, 0.350, 0.045])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    ticks = np.array([-color_limit, 0.0, color_limit])
    cb.set_ticks(ticks)
    cb.ax.set_xticklabels([f"{-color_limit * 1000:.1f}", "0", f"{color_limit * 1000:.1f}"])
    cb.ax.tick_params(labelsize=42, length=6, width=1.4, pad=5)
    for label in cb.ax.get_xticklabels():
        label.set_fontname(FONT_FAMILY)
        label.set_fontsize(42)

    if np.any(sig_roi):
        legend = Line2D([0], [0], marker="s", linestyle="none", markersize=7,
                        markerfacecolor="white", markeredgecolor="#E0A326",
                        label="ROI: BH-FDR q < 0.05")
        fig.legend(handles=[legend], loc="lower center", bbox_to_anchor=(0.5, 0.018),
                   frameon=False, prop={"family": FONT_FAMILY, "size": 10})

    # Use c.py's fixed 13.5 x 10 inch canvas at 600 dpi (8100 x 6000 px).
    fig.set_size_inches(13.5, 10.0, forward=True)
    fig.savefig(output_path, dpi=600, facecolor="white", format="png")
    fig.savefig(output_path.with_suffix(".pdf"), facecolor="white", format="pdf")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = _read_cohort()
    surfaces = fetch_fslr()
    surf_l, surf_r = surfaces["veryinflated"]
    atlas = load_parcellation("schaefer", 400, join=True)

    results = {}
    rows = []
    participants = {}
    for group in GROUP_ORDER:
        participants[group] = {}
        for feature_dim, gradient in FEATURE_DIMS:
            ids, before, after, missing = _load_paired_features(labels, group, feature_dim)
            participants[group][gradient] = ids
            delta = after - before
            mean_delta = np.nanmean(delta, axis=0)
            sd_delta = np.nanstd(delta, axis=0, ddof=1)
            t_values, p_values = ttest_rel(after, before, axis=0, nan_policy="omit")
            q_values = np.full(N_ROI, np.nan, dtype=float)
            valid = np.isfinite(p_values)
            if valid.any():
                q_values[valid] = multipletests(p_values[valid], method="fdr_bh")[1]

            results[(group, gradient)] = {
                "mean_delta": mean_delta,
                "sd_delta": sd_delta,
                "t": t_values,
                "p": p_values,
                "q": q_values,
                "n": len(ids),
            }
            for roi in range(N_ROI):
                rows.append({
                    "Group": group,
                    "Gradient": gradient,
                    "N": len(ids),
                    "ROI_ID": roi + 1,
                    "Mean_delta_post_minus_pre": mean_delta[roi],
                    "SD_delta": sd_delta[roi],
                    "Paired_t": t_values[roi],
                    "P_value": p_values[roi],
                    "BH_FDR_q": q_values[roi],
                    "FDR_significant": bool(np.isfinite(q_values[roi]) and q_values[roi] < FDR_ALPHA),
                })
            print(f"[INFO] {group} {gradient}: n={len(ids)}, FDR-significant ROIs={(q_values < FDR_ALPHA).sum()}")

    all_values = np.concatenate([v["mean_delta"] for v in results.values()])
    color_limit = float(np.nanmax(np.abs(all_values)))
    if not np.isfinite(color_limit) or color_limit <= 0:
        raise ValueError("No non-zero finite mean gradient changes were available to plot.")

    for group in GROUP_ORDER:
        for _, gradient in FEATURE_DIMS:
            res = results[(group, gradient)]
            title = f"{group} (GD{gradient[-1]})"
            png_path = OUTPUT_DIR / f"Figure2a_{group}_{gradient}.png"
            _save_single_surface(
                res["mean_delta"], res["q"], title, png_path,
                atlas, surf_l, surf_r, color_limit
            )
            print(f"[SAVE] {png_path}")
            print(f"[SAVE] {png_path.with_suffix('.pdf')}")

    csv_path = OUTPUT_DIR / "Figure2a_ROI_mean_delta_and_FDR.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {csv_path}")


if __name__ == "__main__":
    main()
