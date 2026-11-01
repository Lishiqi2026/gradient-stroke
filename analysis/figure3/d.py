#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import glob
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from nibabel.freesurfer.io import read_annot
from matplotlib.patches import Rectangle
# ============================================================
# 0) 全局设置
# ============================================================
# 与参考水平柱状图一致：GD1 与 GD2 分别使用一组柔和的端点色。
BAR_COLORS = {
    0: {"negative": "#B76E99", "positive": "#7FA65A"},
    1: {"negative": "#C58A52", "positive": "#7D6CA5"},
}

INK = "#272727"
ZERO = "#A8A8A8"

ALPHA_A = 0.05
FEATURE_DIMS = [0, 1]     # 保持原代码：只分析前两个维度

# 现在只有两组
GROUP_ORDER = ["Control", "TMS"]

# 输出目录
OUT_DIR = r"F:\lishiqi\组间比较\d"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "axes.unicode_minus": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

# ============================================================
# 1) 输入路径
# ============================================================
BASE_DIR = r"F:\lishiqi\预处理数据\cortex\duiqi (400, 5)"

BEFORE_DIR_CONTROL = os.path.join(BASE_DIR, "Control", "before")
AFTER_DIR_CONTROL  = os.path.join(BASE_DIR, "Control", "after")

BEFORE_DIR_TMS = os.path.join(BASE_DIR, "TMS", "before")
AFTER_DIR_TMS  = os.path.join(BASE_DIR, "TMS", "after")

# annot 路径：沿用你前面给过的路径
LH_ANNOT = r"F:\lishiqi\预处理数据\cortex\lh.Schaefer2018_400Parcels_7Networks_order.annot"
RH_ANNOT = r"F:\lishiqi\预处理数据\cortex\rh.Schaefer2018_400Parcels_7Networks_order.annot"

GROUPS = {
    "Control": {"before_dir": BEFORE_DIR_CONTROL, "after_dir": AFTER_DIR_CONTROL},
    "TMS": {"before_dir": BEFORE_DIR_TMS, "after_dir": AFTER_DIR_TMS},
}

# ============================================================
# 2) 网络标签
# ============================================================
NET_ABBR_UPPER = {
    "Visual": "VIS",
    "SomMot": "SMN",
    "DAN": "DAN",
    "VAN/SM": "VAN",
    "Limbic": "LIM",
    "FPN": "CON",
    "DMN": "DMN",
}
LABEL_FONT = 11
TICK_FONT = 10
LEGEND_FONT = 11
DIRECTION_TITLE_FONT = 9.5

# Horizontal ROI-count bars use the ridgeline panel's typography so that
# panels with the same physical size have the same visual text weight.
HORIZONTAL_GLOBAL_FONT = 15
HORIZONTAL_TICK_FONT = 17
HORIZONTAL_AXIS_LABEL_FONT = 15
HORIZONTAL_DIRECTION_TITLE_FONT = 12.5
# ============================================================
# 3) 读取 annot -> 400 ROI 对应 7 networks
# ============================================================
def load_roi_networks_from_annot(lh_annot_path, rh_annot_path):
    net_order = ['Visual', 'SomMot', 'DAN', 'VAN/SM', 'Limbic', 'FPN', 'DMN']
    name2idx = {n: i for i, n in enumerate(net_order)}

    short2full = {
        'Vis': 'Visual',
        'SomMot': 'SomMot',
        'DorsAttn': 'DAN',
        'SalVentAttn': 'VAN/SM',
        'SalVentAttnA': 'VAN/SM',
        'SalVentAttnB': 'VAN/SM',
        'Limbic': 'Limbic',
        'Cont': 'FPN',
        'Default': 'DMN',
    }

    def parse_one_hemi(annot_path, hemi_expected="LH"):
        labels, ctab, names = read_annot(annot_path)
        names = [n.decode("utf-8") if isinstance(n, (bytes, bytearray)) else str(n) for n in names]

        # ``read_annot`` returns the color-table names in the atlas parcel
        # order.  For the Schaefer400 annotation this is the order that the
        # 200 ROI values for one hemisphere use (the other hemisphere is
        # concatenated below).  Keep that order; the trailing parcel number
        # is only a within-network label and must not be used as a global
        # sorting key.
        items = []
        for nm in names:
            low = nm.lower()
            if "unknown" in low or "medial" in low or "wall" in low:
                continue

            parts = nm.split("_")
            if len(parts) < 4:
                continue

            hemi = parts[1]
            net_short = parts[2]
            try:
                idx_num = int(parts[-1])
            except Exception:
                continue

            if hemi != hemi_expected:
                continue

            net_full = short2full.get(net_short, net_short)
            net_idx = name2idx.get(net_full, -1)
            items.append((idx_num, net_idx, nm))

        if len(items) != 200:
            raise RuntimeError(
                f"[ERROR] Expected 200 {hemi_expected} Schaefer parcels, got {len(items)}"
            )

        # Do not sort by ``idx_num`` here.  Values in the ROI arrays are
        # indexed by the atlas/color-table order, whereas ``idx_num`` resets
        # to 1 inside every network (Vis_1, SomMot_1, DAN_1, ...).
        return np.array([it[1] for it in items], dtype=int)

    lh_net = parse_one_hemi(lh_annot_path, "LH")
    rh_net = parse_one_hemi(rh_annot_path, "RH")
    roi_net_idx = np.concatenate([lh_net, rh_net], axis=0)

    if roi_net_idx.shape[0] != 400:
        raise RuntimeError(f"[ERROR] ROI mapping length mismatch: got {roi_net_idx.shape[0]}, expected 400")

    return roi_net_idx, net_order

# ============================================================
# 4) 一些辅助函数（保持原样）
# ============================================================
def choose_rmax(v):
    v = int(v)
    if v <= 4:
        return 4
    elif v <= 6:
        return 6
    elif v <= 8:
        return 8
    elif v <= 10:
        return 10
    elif v <= 12:
        return 12
    else:
        return int(np.ceil(v / 2.0) * 2)

def map_counts_to_radius(vals, rmax, zero_frac=0.5):
    vals = np.asarray(vals, dtype=float)
    vals = np.clip(vals, 0, None)

    r0 = rmax * zero_frac
    return r0 + (vals / float(rmax)) * (rmax - r0)

def annotate_one_max_each_series(ax, angles, neg_raw, pos_raw, neg_disp, pos_disp,
                                 rmax, neg_color, pos_color, fontsize=10):
    neg_raw = np.asarray(neg_raw, dtype=float)
    pos_raw = np.asarray(pos_raw, dtype=float)
    neg_disp = np.asarray(neg_disp, dtype=float)
    pos_disp = np.asarray(pos_disp, dtype=float)

    entries = []

    if np.any(neg_raw > 0):
        i_neg = int(np.argmax(neg_raw))
        entries.append(("neg", i_neg, neg_raw[i_neg], neg_disp[i_neg], neg_color))

    if np.any(pos_raw > 0):
        i_pos = int(np.argmax(pos_raw))
        entries.append(("pos", i_pos, pos_raw[i_pos], pos_disp[i_pos], pos_color))

    if not entries:
        return

    same_angle = False
    if len(entries) == 2 and entries[0][1] == entries[1][1]:
        same_angle = True

    for series, idx, raw_val, disp_val, color in entries:
        ang = angles[idx]

        # 不再用很大的 offset points，而是在极坐标半径上轻微移动
        if disp_val >= rmax * 0.88:
            # 非常靠近外圈时，数字放到线内侧
            rr = disp_val - 0.10 * rmax
        else:
            # 其余情况，数字放在线外侧一点点
            rr = disp_val + 0.075 * rmax

        # 防止数字太靠外或太靠中心
        rr = np.clip(rr, rmax * 0.56, rmax * 0.90)

        # 如果两个最大值在同一个网络/结构，左右轻微错开
        if same_angle:
            if series == "neg":
                ang = ang - 0.055
            else:
                ang = ang + 0.055

        ax.text(
            ang, rr, f"{int(round(raw_val))}",
            color=color,
            fontsize=fontsize,
            fontweight="bold",
            fontfamily="Arial",
            ha="center",
            va="center",
            clip_on=False,
            zorder=60
        )

# ============================================================
# 5) 新的被试文件配对读取
#    适配你现在的目录结构：
#    F:\lishiqi\预处理数据\顶点级duiqi转换（400，5）\Control\before\*.npy
#    F:\lishiqi\预处理数据\顶点级duiqi转换（400，5）\Control\after\*.npy
#    F:\lishiqi\预处理数据\顶点级duiqi转换（400，5）\TMS\before\*.npy
#    F:\lishiqi\预处理数据\顶点级duiqi转换（400，5）\TMS\after\*.npy
#
#    单个被试文件应为：
#    - (400,)
#    - (400, 5)
# ============================================================
def get_path_a_colors(group_tag):
    return BAR_COLORS[0]["negative"], BAR_COLORS[0]["positive"]

def _strip_known_suffix(name_no_ext):
    """
    去掉文件名中已知的公共后缀。
    例如：
    PRIVATE_SUBJECT_022-PRIVATE_SUBJECT_046-1_aligned5_masked
    -> PRIVATE_SUBJECT_022-PRIVATE_SUBJECT_046-1
    """
    suffixes = [
        "_aligned5_masked",
        "_aligned_masked",
        "_masked",
    ]
    out = name_no_ext
    changed = True
    while changed:
        changed = False
        for suf in suffixes:
            if out.endswith(suf):
                out = out[:-len(suf)]
                changed = True
    return out

def _canonical_subject_key(filepath):
    """
    按文件名最前面的那一段作为被试 ID 来配对。

    例如：
    before: PRIVATE_SUBJECT_022-PRIVATE_SUBJECT_046-1_aligned5_masked.npy
    after : PRIVATE_SUBJECT_022-PRIVATE_SUBJECT_047-2_aligned5_masked.npy
    都映射成：PRIVATE_SUBJECT_022
    """
    base = os.path.basename(filepath)
    name_no_ext = os.path.splitext(base)[0]
    s = _strip_known_suffix(name_no_ext)

    # 先按 - 或 _ 分割，取第一段作为 subject key
    parts = re.split(r"[-_]", s)
    if len(parts) == 0 or parts[0] == "":
        raise RuntimeError(f"[ERROR] Cannot parse subject key from filename: {filepath}")

    return parts[0]

def _collect_npy_files(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.npy")))
    if len(files) == 0:
        raise RuntimeError(f"[ERROR] No .npy files found in: {folder}")
    return files

def _build_subject_map(files):
    mp = {}
    for fp in files:
        key = _canonical_subject_key(fp)
        if key in mp:
            raise RuntimeError(
                f"[ERROR] Duplicate subject key detected: {key}\n"
                f"  file1: {mp[key]}\n"
                f"  file2: {fp}\n"
                f"Please check filename pattern."
            )
        mp[key] = fp
    return mp

def load_roi_features_for_subjects(before_dir, after_dir, feature_dim=0, R_expected=400):
    before_files = _collect_npy_files(before_dir)
    after_files  = _collect_npy_files(after_dir)

    before_map = _build_subject_map(before_files)
    after_map  = _build_subject_map(after_files)

    common_keys = sorted(set(before_map.keys()) & set(after_map.keys()))
    only_before = sorted(set(before_map.keys()) - set(after_map.keys()))
    only_after  = sorted(set(after_map.keys()) - set(before_map.keys()))

    if len(only_before) > 0:
        print(f"[WARN] Subjects only in BEFORE ({len(only_before)}): {only_before[:10]}")
    if len(only_after) > 0:
        print(f"[WARN] Subjects only in AFTER ({len(only_after)}): {only_after[:10]}")

    if len(common_keys) == 0:
        raise RuntimeError(
            f"[ERROR] No matched subjects between:\n"
            f"  BEFORE: {before_dir}\n"
            f"  AFTER : {after_dir}\n"
            f"Please check filename pattern."
        )

    X_before, X_after = [], []

    def pick_feat(x, fp):
        x = np.squeeze(x)

        if x.ndim == 2:
            if x.shape[0] != R_expected:
                raise ValueError(f"[ERROR] {fp}: expected ({R_expected}, k), got {x.shape}")
            if feature_dim < 0 or feature_dim >= x.shape[1]:
                raise ValueError(f"[ERROR] {fp}: feature_dim={feature_dim} out of range for shape {x.shape}")
            return x[:, feature_dim]

        elif x.ndim == 1:
            if x.shape[0] != R_expected:
                raise ValueError(f"[ERROR] {fp}: expected ({R_expected},), got {x.shape}")
            return x

        else:
            raise ValueError(f"[ERROR] {fp}: unsupported shape {x.shape}")

    for sub_key in common_keys:
        bpath = before_map[sub_key]
        apath = after_map[sub_key]

        xb = np.load(bpath)
        xa = np.load(apath)

        X_before.append(pick_feat(xb, bpath).astype(np.float32, copy=False))
        X_after.append(pick_feat(xa, apath).astype(np.float32, copy=False))

    X_before = np.stack(X_before, axis=0)
    X_after  = np.stack(X_after, axis=0)

    print(f"[INFO] Matched subjects: N={len(common_keys)}")
    print(f"[INFO] Loaded ROI: N={X_before.shape[0]}, R={X_before.shape[1]}, GD={feature_dim + 1}")
    return X_before, X_after, common_keys

# ============================================================
# 6) 配对 t 检验 + 网络计数（保持原样）
# ============================================================
def pathA_between_group_ttest_rois(control_change, tms_change):
    """Welch test of TMS versus Control intervention changes at every ROI."""
    _, p_vals = ttest_ind(
        tms_change, control_change, axis=0, equal_var=False, nan_policy="omit"
    )
    difference_mean = np.nanmean(tms_change, axis=0) - np.nanmean(control_change, axis=0)
    return p_vals.astype(np.float32), difference_mean.astype(np.float32)

def count_rois_per_network(sig_mask, roi_sign, roi_net_idx, n_net):
    sig_mask = np.asarray(sig_mask, dtype=bool)
    roi_sign = np.asarray(roi_sign, dtype=float)

    pos = sig_mask & (roi_sign > 0)
    neg = sig_mask & (roi_sign < 0)

    pos_counts = np.zeros(n_net, dtype=int)
    neg_counts = np.zeros(n_net, dtype=int)

    for i in range(n_net):
        pos_counts[i] = int(np.sum(pos & (roi_net_idx == i)))
        neg_counts[i] = int(np.sum(neg & (roi_net_idx == i)))

    return neg_counts, pos_counts

# ============================================================
# 7) 雷达图样式（保持原样）
# ============================================================
def radar_axes_style_like_reference(ax, net_order, rmax, zero_frac=0.5):
    n = len(net_order)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    angles_closed = np.concatenate([angles, angles[:1]])

    # 从正上方开始，顺时针
    ax.set_theta_offset(np.pi / 2.0)
    ax.set_theta_direction(-1)

    # 外圈上限
    ax.set_ylim(0, rmax)

    # 先保留角度位置，但不使用 matplotlib 自动文字标签
    labels = [NET_ABBR_UPPER.get(x, str(x)) for x in net_order]
    ax.set_xticks(angles)
    ax.set_xticklabels([])
    ax.tick_params(axis="x", length=0, pad=0)

    # 手动放置 7 个网络名称：距离外圈更统一，不会有的近、有的远
    label_r = 0.500   # 越大，网络名称离雷达图越远；建议 0.57–0.60

    for ang, lab in zip(angles, labels):
        # 这里用 axes 坐标，不用 polar 的 tick label
        x = 0.5 + label_r * np.sin(ang)
        y = 0.5 + label_r * np.cos(ang)

        if x > 0.56:
            ha = "left"
        elif x < 0.44:
            ha = "right"
        else:
            ha = "center"

        if y > 0.56:
            va = "bottom"
        elif y < 0.44:
            va = "top"
        else:
            va = "center"

        ax.text(
            x, y, lab,
            transform=ax.transAxes,
            fontsize=LABEL_FONT,
            color="0.2",
            fontfamily="Arial",
            fontweight="normal",
            ha=ha,
            va=va,
            clip_on=False
        )

    # 不显示径向刻度
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.grid(False)

    # 外圈
    ax.spines["polar"].set_color("black")
    ax.spines["polar"].set_linewidth(2.2)

    # 灰色径向虚线
    for ang in angles:
        ax.plot(
            [ang, ang], [0, rmax],
            linestyle="--",
            linewidth=0.9,
            color="0.83",
            zorder=0
        )

    # 0 基线虚线圆：半径 1/2
    r0 = rmax * zero_frac
    theta_dense = np.linspace(0, 2 * np.pi, 512)
    ax.plot(
        theta_dense,
        np.full_like(theta_dense, r0),
        linestyle="--",
        linewidth=1.5,
        color="0.25",
        zorder=0
    )

    return angles, angles_closed, r0

def plot_single_radar_pathA_only(net_order, A_neg_cnt, A_pos_cnt, max_count,
                                 out_dir, feature_dim, group_tag=None,
                                 save_svg=True, show_legend=True,
                                 show_max_text=True, file_suffix=""):
    gd_show = feature_dim + 1
    neg_color, pos_color = get_path_a_colors(group_tag)

    # 原始计数
    neg_raw = np.asarray(A_neg_cnt, dtype=float)
    pos_raw = np.asarray(A_pos_cnt, dtype=float)

    # 统一外圈上限
    rmax = choose_rmax(max_count)

    # 0 基线在半径一半
    zero_frac = 0.5
    neg_disp = map_counts_to_radius(neg_raw, rmax=rmax, zero_frac=zero_frac)
    pos_disp = map_counts_to_radius(pos_raw, rmax=rmax, zero_frac=zero_frac)

    FIG_W = 4.0
    FIG_H = 3.5
    DPI = 600

    rc = {
        "font.family": "Arial",
        "axes.unicode_minus": False,
        "font.size": LABEL_FONT,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }

    with plt.rc_context(rc):
        fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)

        gs = fig.add_gridspec(
            nrows=2,
            ncols=1,
            height_ratios=[3.1, 0.2],
            hspace=0.0
        )

        ax = fig.add_subplot(gs[0], polar=True)
        legend_ax = fig.add_subplot(gs[1])
        legend_ax.axis("off")

        fig.subplots_adjust(left=0.01, right=0.99, top=0.995, bottom=0.0)

        # 画布不变，只放大雷达图本身
        # 原始 0.72；这里用 0.78，比原图大，但不容易裁切字体
        pos = ax.get_position()
        scale = 0.78

        new_w = pos.width * scale
        new_h = pos.height * scale
        new_x = pos.x0 + (pos.width - new_w) / 2
        new_y = pos.y0 + (pos.height - new_h) / 2 - 0.010

        ax.set_position([new_x, new_y, new_w, new_h])
        legend_ax.axis("off")

        angles, angles_closed, r0 = radar_axes_style_like_reference(
            ax=ax,
            net_order=net_order,
            rmax=rmax,
            zero_frac=zero_frac
        )

        neg_closed = np.concatenate([neg_disp, neg_disp[:1]])
        pos_closed = np.concatenate([pos_disp, pos_disp[:1]])

        # 半透明填充
        ax.fill(
            angles_closed, neg_closed,
            color=neg_color,
            alpha=0.18,
            zorder=1
        )
        ax.fill(
            angles_closed, pos_closed,
            color=pos_color,
            alpha=0.18,
            zorder=2
        )

        # 曲线 + 圆点
        ax.plot(
            angles_closed, neg_closed,
            color=neg_color,
            linewidth=2.0,
            marker="o",
            markersize=3.8,
            zorder=3,
            label="Path A (-)"
        )
        ax.plot(
            angles_closed, pos_closed,
            color=pos_color,
            linewidth=2.0,
            marker="o",
            markersize=3.8,
            zorder=4,
            label="Path A (+)"
        )

        # A(-)/A(+) 各标 1 个最大值（可选）
        if show_max_text:
            annotate_one_max_each_series(
                ax=ax,
                angles=angles,
                neg_raw=neg_raw,
                pos_raw=pos_raw,
                neg_disp=neg_disp,
                pos_disp=pos_disp,
                rmax=rmax,
                neg_color=neg_color,
                pos_color=pos_color,
                fontsize=TICK_FONT
            )

        # 图例
        if show_legend:
            legend_handles = [
                Rectangle(
                    (0, 0), 1, 1,
                    facecolor=neg_color,
                    edgecolor="black",
                    linewidth=0.5
                ),
                Rectangle(
                    (0, 0), 1, 1,
                    facecolor=pos_color,
                    edgecolor="black",
                    linewidth=0.5
                ),
            ]
            legend_labels = ["Path A (-)", "Path A (+)"]

            legend_ax.legend(
                legend_handles, legend_labels,
                loc="center",
                bbox_to_anchor=(0.5, 0.27),
                ncol=2,
                frameon=False,
                fontsize=LEGEND_FONT,
                handlelength=1.2,
                columnspacing=0.8,
                handletextpad=0.4,
                borderaxespad=0.0
            )

        os.makedirs(out_dir, exist_ok=True)
        tag = f"_G{group_tag}" if group_tag is not None else ""

        out_png = os.path.join(out_dir, f"single_d_GD{gd_show}{tag}{file_suffix}.png")
        out_pdf = os.path.join(out_dir, f"single_d_GD{gd_show}{tag}{file_suffix}.pdf")

        fig.savefig(out_png, dpi=DPI)
        fig.savefig(out_pdf)

        if save_svg:
            out_svg = os.path.join(out_dir, f"single_d_GD{gd_show}{tag}{file_suffix}.svg")
            fig.savefig(out_svg)

        plt.close(fig)
        print(f"[SAVE] {out_png}")


# ============================================================
# 7b) 双向水平柱状图
#     统计逻辑仍为 TMS(Post-Pre) - Control(Post-Pre)，这里只改变表现形式。
# ============================================================
def plot_single_horizontal_counts(net_order, A_neg_cnt, A_pos_cnt, max_count,
                                  out_dir, feature_dim, group_tag=None,
                                  save_svg=True, show_max_text=True,
                                  file_suffix=""):
    gd_show = feature_dim + 1
    colors = BAR_COLORS[feature_dim]

    negative = np.asarray(A_neg_cnt, dtype=int)
    positive = np.asarray(A_pos_cnt, dtype=int)
    y = np.arange(len(net_order))

    # 保持原图画布尺寸和分辨率完全不变。
    FIG_W = 4.0
    FIG_H = 3.5
    DPI = 600

    axis_max = max(4, choose_rmax(max_count))
    if axis_max <= 4:
        tick_step = 1
    elif axis_max <= 12:
        tick_step = 2
    else:
        tick_step = max(2, int(np.ceil(axis_max / 4.0)))

    ticks = np.arange(-axis_max, axis_max + 0.5 * tick_step, tick_step)

    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": HORIZONTAL_GLOBAL_FONT,
        "xtick.labelsize": HORIZONTAL_TICK_FONT,
        "ytick.labelsize": HORIZONTAL_TICK_FONT,
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }

    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)

        ax.barh(
            y, -negative,
            height=0.56,
            color=colors["negative"],
            edgecolor="white",
            linewidth=0.6,
            zorder=2,
        )
        ax.barh(
            y, positive,
            height=0.56,
            color=colors["positive"],
            edgecolor="white",
            linewidth=0.6,
            zorder=2,
        )
        ax.axvline(0.0, color=ZERO, linewidth=1.0, zorder=1)

        # 使用绝对值显示左右两侧的 ROI 数量，与参考图一致。
        ax.set_xlim(-(axis_max + 0.55), axis_max + 0.55)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(abs(int(v))) for v in ticks])
        ax.set_yticks(y)
        ax.set_yticklabels(
            [NET_ABBR_UPPER.get(name, str(name)) for name in net_order],
            fontsize=HORIZONTAL_TICK_FONT,
        )
        ax.invert_yaxis()

        # Keep the network names clearly outside the negative bar-end labels.
        # The largest negative count is close to the left x-limit, so the
        # default 5-point tick padding lets its number collide with labels
        # such as ``SMN``.  Extra padding plus a slightly wider left margin
        # preserves the data area while keeping both text elements readable.
        ax.tick_params(axis="y", length=0, pad=7.72)  # Keep network labels fixed after shifting axes left.
        ax.tick_params(axis="x", length=3.0, width=0.9, pad=3)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color(INK)
        ax.spines["bottom"].set_linewidth(0.9)

        ax.text(
            0.25, 1.025, "ΔTMS < ΔControl",
            transform=ax.transAxes,
            ha="center", va="bottom",
            fontsize=HORIZONTAL_DIRECTION_TITLE_FONT,
            color=colors["negative"],
            fontweight="semibold",
            clip_on=False,
        )
        ax.text(
            0.75, 1.025, "ΔTMS > ΔControl",
            transform=ax.transAxes,
            ha="center", va="bottom",
            fontsize=HORIZONTAL_DIRECTION_TITLE_FONT,
            color=colors["positive"],
            fontweight="semibold",
            clip_on=False,
        )

        ax.set_xlabel("Number of ROIs with nominal P < 0.05", fontsize=HORIZONTAL_AXIS_LABEL_FONT)

        # 保留原代码“带最大值数字”和“无数字”两种输出。
        if show_max_text:
            if np.any(negative > 0):
                idx = int(np.argmax(negative))
                ax.text(
                    -negative[idx] - 0.10, idx, str(int(negative[idx])),
                    ha="right", va="center", fontsize=HORIZONTAL_TICK_FONT,
                    color=colors["negative"], fontweight="bold",
                    clip_on=False,
                )
            if np.any(positive > 0):
                idx = int(np.argmax(positive))
                ax.text(
                    positive[idx] + 0.10, idx, str(int(positive[idx])),
                    ha="left", va="center", fontsize=HORIZONTAL_TICK_FONT,
                    color=colors["positive"], fontweight="bold",
                    clip_on=False,
                )

        fig.subplots_adjust(left=0.19, right=0.92, top=0.885, bottom=0.195)

        os.makedirs(out_dir, exist_ok=True)
        tag = f"_G{group_tag}" if group_tag is not None else ""
        stem = f"single_d_GD{gd_show}{tag}{file_suffix}"

        out_png = os.path.join(out_dir, f"{stem}.png")
        out_pdf = os.path.join(out_dir, f"{stem}.pdf")

        # 不使用 bbox_inches='tight'，保证输出仍为 4.0 x 3.5 inch。
        fig.savefig(out_png, dpi=DPI)
        fig.savefig(out_pdf)

        if save_svg:
            out_svg = os.path.join(out_dir, f"{stem}.svg")
            fig.savefig(out_svg)

        plt.close(fig)
        print(f"[SAVE] {out_png}")


# ============================================================
# 8) main：输出组间比较的双向水平柱状图
# ============================================================
if __name__ == "__main__":
    print("[STEP 1] Load ROI network mapping from annot...")
    roi_net_idx, net_order = load_roi_networks_from_annot(LH_ANNOT, RH_ANNOT)

    for feat in FEATURE_DIMS:
        gd_show = feat + 1
        print(f"\n========== RUN Control vs TMS change | GD{gd_show} ==========")

        group_changes = {}
        for grp in GROUP_ORDER:
            cfg = GROUPS[grp]
            Xb, Xa, matched_subjects = load_roi_features_for_subjects(
                cfg["before_dir"], cfg["after_dir"], feature_dim=feat, R_expected=400
            )
            group_changes[grp] = Xa - Xb

        pA, difference_mean = pathA_between_group_ttest_rois(
            group_changes["Control"], group_changes["TMS"]
        )
        sigA = pA < ALPHA_A
        signA = np.sign(difference_mean)
        A_neg, A_pos = count_rois_per_network(
            sigA, signA, roi_net_idx, n_net=len(net_order)
        )
        print("  TMS change < Control change:", dict(zip(net_order, A_neg)))
        print("  TMS change > Control change:", dict(zip(net_order, A_pos)))

        max_count = int(np.max(np.concatenate([A_neg, A_pos])))
        max_count = max(1, max_count)

        plot_single_horizontal_counts(
            net_order=net_order, A_neg_cnt=A_neg, A_pos_cnt=A_pos,
            max_count=max_count, out_dir=OUT_DIR, feature_dim=feat,
            group_tag="Between", save_svg=True,
            show_max_text=True, file_suffix=""
        )
        plot_single_horizontal_counts(
            net_order=net_order, A_neg_cnt=A_neg, A_pos_cnt=A_pos,
            max_count=max_count, out_dir=OUT_DIR, feature_dim=feat,
            group_tag="Between", save_svg=True,
            show_max_text=False, file_suffix="_nonum"
        )

    print("\n[DONE] Between-group horizontal bar figures finished.")
