import os
import tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import gridspec
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from surfplot import Plot
from neuromaps.datasets import fetch_fslr
from brainspace.datasets import load_parcellation


# =============================
# Color range for each panel
# grad_col: 0 = Gradient 1, 1 = Gradient 2
# =============================
PANEL_RANGES = {
    ("Control", 0): (-0.0010, 0.0010),
    ("Control", 1): (-0.0030, 0.0030),
    ("TMS", 0):     (-0.0030, 0.0050),
    ("TMS", 1):     (-0.0010, 0.0010),
}

# =============================
# Group-specific colormaps
# Control: purple-white-orange
# TMS: blue-white-red
# =============================
RIGHT_CMAP_COLORS = [
    "#390D58",
    "#5F3892",
    "#877CB3",
    "#B6B0D5",
    "#DBDCEB",
    "#F8F5F4",
    "#FDDFB4",
    "#FCB866",
    "#DF8316",
    "#B75D05",
    "#8A3E06",
]

RIGHT_CMAP = LinearSegmentedColormap.from_list(
    "purple_white_orange",
    RIGHT_CMAP_COLORS,
    N=256
)

TVAL_CMAP_COLORS = [
    "#053061",
    "#2166AC",
    "#4393C3",
    "#92C5DE",
    "#D1E5F0",
    "#F7F7F7",
    "#FDDBC7",
    "#F4A582",
    "#D6604D",
    "#B2182B",
    "#67001F",
]

TVAL_CMAP = LinearSegmentedColormap.from_list(
    "blue_white_red",
    TVAL_CMAP_COLORS,
    N=256
)

# Match the supplied reference brain maps: Gradient 1 uses green-white-pink,
# while Gradient 2 uses orange-white-purple. Only the colour mapping changes;
# the data range, layout, typography and rendering pipeline remain unchanged.
GRADIENT1_CMAP = LinearSegmentedColormap.from_list(
    "gradient1_green_white_pink",
    ["#3A7D44", "#87B96A", "#DCEBC9", "#FFFFFF", "#F4D6E2", "#D786A7", "#A1125B"],
    N=257,
)

GRADIENT2_CMAP = LinearSegmentedColormap.from_list(
    "gradient2_orange_white_purple",
    ["#8A4A13", "#C47A27", "#F0C985", "#FFFDF7", "#D9D4E7", "#9A8ABF", "#3D226F"],
    N=257,
)

GRADIENT_CMAPS = {0: GRADIENT1_CMAP, 1: GRADIENT2_CMAP}

GROUP_CMAPS = {
    "Control": RIGHT_CMAP,
    "TMS": TVAL_CMAP,
}

mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["font.sans-serif"] = ["Arial"]
mpl.rcParams["font.size"] = 15
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"

# =============================
# Unified figure style
# 画布不变，字体统一 15 pt
# =============================
FIG_W = 4.0
FIG_H = 3.5
DPI = 600

# 标题行稍微加高，避免标题压脑图
TITLE_ROW_H = 0.30

# 色条区域加高，防止 Gradient 1 value 被裁掉
CBAR_AREA_H = 0.42

# 底部保留足够空间，不能太小
BOTTOM_BLANK_H = 0.20

TITLE_FONTSIZE = 16
CBAR_TICK_FONTSIZE = 16
CBAR_LABEL_FONTSIZE = 14

# 色条不要太粗，宽度略加一点
CBAR_WIDTH = "42%"
CBAR_HEIGHT = "28%"
CBAR_YSHIFT = 0.18

SAVE_TIGHT = False   # 关键：最终 PDF/SVG/PNG 不再自动裁剪

# =============================
# 数据路径
# =============================
DATA_DIR = r"F:\lishiqi\预处理数据\cortex\组水平（400）"
OUT_DIR = r"F:\lishiqi\组间比较\a"
os.makedirs(OUT_DIR, exist_ok=True)

GROUP_FILES = {
    "Control": {
        "pre": os.path.join(DATA_DIR, "Control_pre.npy"),
        "post": os.path.join(DATA_DIR, "Control_post.npy"),
    },
    "TMS": {
        "pre": os.path.join(DATA_DIR, "TMS_pre.npy"),
        "post": os.path.join(DATA_DIR, "TMS_post.npy"),
    }
}


# =============================
# surfplot 资源
# =============================
surfaces = fetch_fslr()
lh_surf, rh_surf = surfaces["veryinflated"]
atlas_32k = load_parcellation("schaefer", 400, join=True)


def _roi400_to_32k(roi_values):
    value_32k = np.zeros_like(atlas_32k, dtype=float)
    for lab in range(1, 401):
        value_32k[atlas_32k == lab] = roi_values[lab - 1]
    return value_32k


def _nonwhite_bbox(img, eps=1e-6):
    rgb = img[..., :3] if (img.ndim == 3 and img.shape[-1] == 4) else img
    mask = np.any(rgb < (1 - eps), axis=-1)
    if not np.any(mask):
        return None
    rows = np.where(np.any(mask, axis=1))[0]
    cols = np.where(np.any(mask, axis=0))[0]
    return (rows[0], rows[-1], cols[0], cols[-1])


def _union_bbox(bboxes):
    bboxes = [b for b in bboxes if b is not None]
    if not bboxes:
        return None
    r0 = min(b[0] for b in bboxes)
    r1 = max(b[1] for b in bboxes)
    c0 = min(b[2] for b in bboxes)
    c1 = max(b[3] for b in bboxes)
    return (r0, r1, c0, c1)


def _crop_pair_to_union(img1, img2, eps=1e-6):
    b1 = _nonwhite_bbox(img1, eps=eps)
    b2 = _nonwhite_bbox(img2, eps=eps)
    ub = _union_bbox([b1, b2])
    if ub is None:
        return img1, img2
    r0, r1, c0, c1 = ub
    return (
        np.ascontiguousarray(img1[r0:r1 + 1, c0:c1 + 1]),
        np.ascontiguousarray(img2[r0:r1 + 1, c0:c1 + 1]),
    )


def _split_grid_2x2(img, eps=1e-6):
    rgb = img[..., :3] if (img.ndim == 3 and img.shape[-1] == 4) else img
    h, w = rgb.shape[:2]
    mask = np.any(rgb < (1 - eps), axis=-1)

    # Fallback: hard split
    if not np.any(mask):
        rsplit, csplit = h // 2, w // 2
        return img[:rsplit, :csplit], img[:rsplit, csplit:], img[rsplit:, :csplit], img[rsplit:, csplit:]

    rows = np.where(np.any(mask, axis=1))[0]
    gaps = np.where(np.diff(rows) > 1)[0]
    rsplit = int(
        (rows[gaps[np.argmax(np.diff(rows)[gaps])]] + rows[gaps[np.argmax(np.diff(rows)[gaps])] + 1]) // 2
    ) if len(gaps) > 0 else h // 2

    cols = np.where(np.any(mask, axis=0))[0]
    gaps = np.where(np.diff(cols) > 1)[0]
    csplit = int(
        (cols[gaps[np.argmax(np.diff(cols)[gaps])]] + cols[gaps[np.argmax(np.diff(cols)[gaps])] + 1]) // 2
    ) if len(gaps) > 0 else w // 2

    rsplit = int(np.clip(rsplit, 1, h - 1))
    csplit = int(np.clip(csplit, 1, w - 1))

    TL = img[:rsplit, :csplit]
    TR = img[:rsplit, csplit:]
    BL = img[rsplit:, :csplit]
    BR = img[rsplit:, csplit:]
    return TL, TR, BL, BR


def _pick_rightlat_rightmed(img):
    TL, TR, BL, BR = _split_grid_2x2(img)
    # 和原代码一致：取 TR = right lateral, BR = right medial
    return TR, BR


def _load_surface_image_from_npy(npy_path, grad_col, temp_tag, cmap, vmin, vmax):
    """
    这里只替换原代码里 draw_cortex 的那一步，
    但输出仍然是一整张 2x2 grid 图，后面继续走原代码流程。
    """
    arr = np.load(npy_path)

    if arr.ndim == 1:
        cortex_vals = arr[:400].astype(float)
    elif arr.ndim == 2:
        cortex_vals = arr[:400, grad_col].astype(float)
    else:
        raise ValueError(f"Unsupported array shape: {arr.shape}")

    value_32k = _roi400_to_32k(cortex_vals)

    # 关键：这里一次性生成 lateral + medial 的整张 grid
    p = Plot(
        lh_surf,
        rh_surf,
        views=["lateral", "medial"],
        zoom=1.7,
        brightness=0.8,
        size=(2000, 1400)
    )

    p.add_layer(
        value_32k,
        cmap=cmap,
        color_range=[vmin, vmax],
        cbar=False
    )

    # 尽量接近原图的灰色 outline
    p.add_layer(
        atlas_32k,
        cmap="gray",
        as_outline=True,
        cbar=False
    )

    fig = p.build()

    temp_png = os.path.join(
        tempfile.gettempdir(),
        f"temp_{temp_tag}_{os.path.splitext(os.path.basename(npy_path))[0]}_g{grad_col+1}.png"
    )

    fig.savefig(temp_png, dpi=300, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)

    img = mpimg.imread(temp_png)

    # Convert alpha channel to white background
    if img.ndim == 3 and img.shape[-1] == 4:
        rgb = img[..., :3]
        a = img[..., 3:4]
        img = rgb * a + 1.0 * (1 - a)

    try:
        os.remove(temp_png)
    except Exception:
        pass

    return img


def _load_gradient_vector(npy_path, grad_col):
    """Load one group-level gradient as a 400-ROI vector."""
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"Missing file: {npy_path}")
    arr = np.load(npy_path)
    if arr.ndim == 1:
        if grad_col != 0:
            raise ValueError(f"Cannot extract Gradient {grad_col + 1} from 1D file: {npy_path}")
        return arr[:400].astype(float)
    if arr.ndim == 2:
        return arr[:400, grad_col].astype(float)
    raise ValueError(f"Unsupported array shape: {arr.shape}")


def _save_change_vector(change, out_path):
    """Save a temporary ROI vector for the unchanged surface-rendering pipeline."""
    np.save(out_path, np.asarray(change, dtype=float))
    return out_path


def make_between_group_delta(
    grad_col=0,
    fig_width=FIG_W,
    fig_height=FIG_H,
    title_row_height=TITLE_ROW_H,
    cbar_area_height=CBAR_AREA_H,
    bottom_blank_height=BOTTOM_BLANK_H,
    dpi=DPI,
    out_png=None,
    add_cbar=True,
    cbar_width=CBAR_WIDTH,
    cbar_height=CBAR_HEIGHT,
    cbar_label="Between-group difference in gradient change",
    cbar_ticksize=CBAR_TICK_FONTSIZE,
    cbar_labelsize=CBAR_LABEL_FONTSIZE,
    map_title="TMS − Control (ΔG)",
    title_fontname="Arial",
    title_fontsize=TITLE_FONTSIZE,
):
    """Plot (TMS post - TMS pre) - (Control post - Control pre) by ROI."""
    control_change = (
        _load_gradient_vector(GROUP_FILES["Control"]["post"], grad_col)
        - _load_gradient_vector(GROUP_FILES["Control"]["pre"], grad_col)
    )
    tms_change = (
        _load_gradient_vector(GROUP_FILES["TMS"]["post"], grad_col)
        - _load_gradient_vector(GROUP_FILES["TMS"]["pre"], grad_col)
    )
    delta_change = tms_change - control_change

    finite = delta_change[np.isfinite(delta_change)]
    abs_max = float(np.max(np.abs(finite))) if finite.size else 0.0
    if abs_max == 0:
        abs_max = 1.0
    vmin, vmax = -abs_max, abs_max
    group_cmap = GRADIENT_CMAPS[grad_col]

    # The inputs are group-level ROI means, so this map is a descriptive
    # difference-in-change map; it does not provide subject-level inference.
    temp_delta_path = os.path.join(
        tempfile.gettempdir(), f"TMS_minus_Control_delta_g{grad_col + 1}.npy"
    )
    np.save(temp_delta_path, delta_change)
    try:
        delta_img = _load_surface_image_from_npy(
            temp_delta_path,
            grad_col=grad_col,
            temp_tag="TMS_minus_Control_delta",
            cmap=group_cmap,
            vmin=vmin,
            vmax=vmax,
        )
    finally:
        try:
            os.remove(temp_delta_path)
        except OSError:
            pass

    # Keep all four views from surfplot's 2x2 output: left/right lateral and medial.
    # Add narrow white gutters between views, matching the supplied reference script.
    rendered = np.asarray(delta_img).copy()
    image_h, image_w = rendered.shape[:2]
    mid_y, mid_x = image_h // 2, image_w // 2
    top_left = rendered[:mid_y, :mid_x]
    top_right = rendered[:mid_y, mid_x:]
    bottom_left = rendered[mid_y:, :mid_x]
    bottom_right = rendered[mid_y:, mid_x:]
    gap_x = max(12, int(image_w * 0.025))
    gap_y = max(12, int(image_h * 0.025))
    separated = np.ones((image_h + gap_y, image_w + gap_x) + rendered.shape[2:], dtype=rendered.dtype)
    separated[:mid_y, :mid_x] = top_left
    separated[:mid_y, mid_x + gap_x:] = top_right
    separated[mid_y + gap_y:, :mid_x] = bottom_left
    separated[mid_y + gap_y:, mid_x + gap_x:] = bottom_right
    delta_img = separated

    fixed_height = title_row_height + bottom_blank_height
    if add_cbar:
        fixed_height += cbar_area_height
    brain_row_height = fig_height - fixed_height
    if brain_row_height <= 0:
        raise ValueError("fig_height is too small for the current layout settings.")

    print(
        f"[COLOR RANGE] TMS − Control difference in change | Gradient {grad_col + 1}: "
        f"vmin={vmin:.6f}, vmax={vmax:.6f}"
    )

    fig = plt.figure(figsize=(fig_width, fig_height))
    bottom_frac = bottom_blank_height / fig_height
    height_ratios = [title_row_height, brain_row_height]
    if add_cbar:
        height_ratios.append(cbar_area_height)

    gs = gridspec.GridSpec(
        len(height_ratios), 1,
        figure=fig,
        left=0.0, right=1.0, top=1.0, bottom=bottom_frac,
        wspace=0.0, hspace=0.0,
        height_ratios=height_ratios,
    )

    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    ax_title.text(
        0.5, 0.5, map_title,
        ha="center", va="center",
        fontsize=title_fontsize,
        fontname=title_fontname,
    )

    ax_map = fig.add_subplot(gs[1, 0])
    ax_map.axis("off")
    ax_map.imshow(delta_img, interpolation="nearest")

    if add_cbar:
        ax_cbar_container = fig.add_subplot(gs[2, :])
        ax_cbar_container.axis("off")
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=group_cmap)
        sm.set_array([])

        cax = inset_axes(
            ax_cbar_container,
            width=cbar_width,
            height=cbar_height,
            loc="center",
            bbox_to_anchor=(0, CBAR_YSHIFT, 1, 1),
            bbox_transform=ax_cbar_container.transAxes,
            borderpad=0,
        )
        cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cbar.set_ticks([])
        cbar.outline.set_linewidth(1.2)
        cbar.set_label(
            cbar_label,
            fontsize=cbar_labelsize,
            labelpad=3,
            fontname="Arial",
        )
        cax.text(
            -0.04, 0.5, f"{vmin:.4f}",
            transform=cax.transAxes,
            ha="right", va="center",
            fontsize=cbar_ticksize,
            fontname="Arial",
        )
        cax.text(
            1.04, 0.5, f"{vmax:.4f}",
            transform=cax.transAxes,
            ha="left", va="center",
            fontsize=cbar_ticksize,
            fontname="Arial",
        )

    if out_png is None:
        out_png = os.path.join(
            OUT_DIR, f"TMS_minus_Control_grad{grad_col + 1}_deltaG.png"
        )
    base, _ = os.path.splitext(out_png)

    fig.savefig(
        out_png,
        dpi=dpi,
        facecolor="white",
        edgecolor="white",
        transparent=False,
        bbox_inches=None,
        pad_inches=0,
    )
    fig.savefig(
        base + ".pdf",
        facecolor="white",
        edgecolor="white",
        bbox_inches=None,
        pad_inches=0,
    )
    fig.savefig(
        base + ".svg",
        facecolor="white",
        edgecolor="white",
        bbox_inches=None,
        pad_inches=0,
    )
    plt.close(fig)

    print("Saved:", out_png)
    print("Saved:", base + ".pdf")
    print("Saved:", base + ".svg")


if __name__ == "__main__":
    grad_configs = [
        (0, "grad1"),
        (1, "grad2"),
    ]

    for grad_col, grad_tag in grad_configs:
        out_png = os.path.join(
            OUT_DIR, f"TMS_minus_Control_{grad_tag}_deltaG.png"
        )

        make_between_group_delta(
            grad_col=grad_col,
            fig_width=FIG_W,
            fig_height=FIG_H,
            title_row_height=TITLE_ROW_H,
            cbar_area_height=CBAR_AREA_H,
            bottom_blank_height=BOTTOM_BLANK_H,
            dpi=DPI,
            out_png=out_png,
            cbar_label=f"ΔΔG{grad_col + 1} (TMS − Control)",
            cbar_width=CBAR_WIDTH,
            cbar_height=CBAR_HEIGHT,
            cbar_ticksize=CBAR_TICK_FONTSIZE,
            cbar_labelsize=CBAR_LABEL_FONTSIZE,
            map_title="TMS − Control (ΔG)",
            title_fontname="Arial",
            title_fontsize=TITLE_FONTSIZE,
        )
