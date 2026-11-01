import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

# ==========================================
# 1. 原始数据录入
# ==========================================
# Patient-level records are deliberately not embedded in the public code.
# Supply a private CSV with the columns below plus Group (Control/TMS).
clinical_csv = os.environ.get("GRADIENT_STROKE_CLINICAL_CSV")
if not clinical_csv:
    raise RuntimeError("Set GRADIENT_STROKE_CLINICAL_CSV to a private clinical CSV; see docs/figure23_table4_notes.md.")
cols = ['ID', 'NIHSS_pre', 'NIHSS_post', 'ADL_pre', 'ADL_post',
        'MMSE_pre', 'MMSE_post', 'HAMD_pre', 'HAMD_post', 'VFT_pre', 'VFT_post']
df_input = pd.read_csv(clinical_csv, dtype={'ID': str})
missing = set(cols + ['Group']).difference(df_input.columns)
if missing:
    raise ValueError(f"Clinical CSV is missing columns: {sorted(missing)}")
if df_input['ID'].duplicated().any():
    raise ValueError("Clinical CSV has duplicate participant IDs")
if not df_input['Group'].isin(['Control', 'TMS']).all():
    raise ValueError("Group must contain only Control or TMS")
df_control = df_input.loc[df_input['Group'].eq('Control'), cols + ['Group']].copy()
df_tms = df_input.loc[df_input['Group'].eq('TMS'), cols + ['Group']].copy()
df_base = pd.concat([df_control, df_tms], ignore_index=True)

# ==========================================
# 2. 读取 Excel 并融合随访数据 (M1)
# ==========================================
excel_path = os.environ.get("GRADIENT_STROKE_FOLLOWUP_XLSX")
if not excel_path:
    raise RuntimeError("Set GRADIENT_STROKE_FOLLOWUP_XLSX to the private follow-up workbook.")

df_excel = pd.read_excel(excel_path, header=None, skiprows=1, usecols=[0, 5, 8], dtype={0: str})
df_excel.columns = ['ID', 'MMSE_M1', 'VFT_M1']

df_base['ID'] = df_base['ID'].astype(str).str.strip()
df_excel['ID'] = df_excel['ID'].astype(str).str.strip()
df_merged = pd.merge(df_base, df_excel, on='ID', how='left')

# ==========================================
# 3. 绘图配置与工具函数：第二张图风格
# ==========================================
from matplotlib.lines import Line2D
from matplotlib.text import Text

groups = ['Control', 'TMS']
scales = ['NIHSS', 'ADL', 'HAMD', 'MMSE', 'VFT']

mpl.rcParams.update({
    "font.family": "Arial",
    "font.sans-serif": ["Arial"],
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.linewidth": 1.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})

# Nature-style final panel scale: approximately 183 mm wide.
# Keep all physical elements compact together so that the reduced canvas does
# not leave oversized labels, axes or markers dominating the paper figure.
FS_TITLE = 7.5
FS_TICK = 8.5
FS_LABEL = 10.2
FS_STAR = 7.3
FS_LEGEND = 9.0
FS_SUPTITLE = 7.5
AXIS_LW = 0.8
SIG_LW = 0.9
POINT_SIZE = 18
POINT_EDGE_LW = 0.45

# 第二张图类似的蓝色/粉色点
TIME_COLORS = {
    "Pre": "#6FA8DC",
    "Post": "#D6A5C9",
    "1-month follow-up": "#4DAF4A"
}

SHOW_NS = False  # False = 不显示 ns；True = 显示 ns

# ==========================================
# 新增：TMS vs Control 改善量比较
# ==========================================
SHOW_BETWEEN_DELTA = True

# 正向定义：delta 越大 = 改善越明显
# NIHSS / HAMD 是分数下降代表改善，所以用 pre - post
# ADL / MMSE / VFT 是分数上升代表改善，所以用 post - pre
LOWER_BETTER = {"NIHSS", "HAMD"}

# 如果想把 ns 也标出来，改成 True
SHOW_BETWEEN_NS = False

# 组间比较方法：Welch 独立样本 t 检验
# 如果你最初统计分析用的是 Mann-Whitney，可以把下面函数里那一行换成 mannwhitneyu
BETWEEN_LABEL_PREFIX = "Δ"


def enforce_arial(fig):
    """Ensure all figure text uses Arial before export."""
    for text in fig.findobj(match=Text):
        text.set_fontfamily("Arial")
        text.set_fontname("Arial")


def get_delta_values(df, scale, target_suffix="post"):
    """
    返回每个被试的改善量。
    delta > 0 表示改善。
    target_suffix:
        "post" 表示 Post - Pre 或 Pre - Post
        "M1"   表示 M1 - Pre 或 Pre - M1
    """
    pre_col = f"{scale}_pre"
    target_col = f"{scale}_{target_suffix}"

    if target_col not in df.columns:
        return np.array([])

    tmp = df[[pre_col, target_col]].apply(pd.to_numeric, errors="coerce").dropna()

    if len(tmp) == 0:
        return np.array([])

    if scale in LOWER_BETTER:
        delta = tmp[pre_col] - tmp[target_col]
    else:
        delta = tmp[target_col] - tmp[pre_col]

    return delta.to_numpy(dtype=float)


def compare_delta_between_groups(df_merged, scale, target_suffix="post"):
    """
    比较 TMS 改善量是否大于 Control 改善量。
    这里使用双侧 Welch t-test。
    """
    delta_control = get_delta_values(
        df_merged[df_merged["Group"] == "Control"],
        scale,
        target_suffix
    )

    delta_tms = get_delta_values(
        df_merged[df_merged["Group"] == "TMS"],
        scale,
        target_suffix
    )

    delta_control = delta_control[np.isfinite(delta_control)]
    delta_tms = delta_tms[np.isfinite(delta_tms)]

    if len(delta_control) <= 2 or len(delta_tms) <= 2:
        return None

    stat, p = stats.ttest_ind(
        delta_tms,
        delta_control,
        equal_var=False,
        nan_policy="omit"
    )

    mean_diff = np.nanmean(delta_tms) - np.nanmean(delta_control)

    return {
        "p": p,
        "star": get_star(p),
        "mean_diff": mean_diff,
        "n_control": len(delta_control),
        "n_tms": len(delta_tms)
    }

def get_star(p):
    if pd.isna(p):
        return 'ns'
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'ns'


def draw_sig_bar(ax, x1, x2, y, h, text, fontsize=FS_STAR):
    ax.plot(
        [x1, x1, x2, x2],
        [y, y + h, y + h, y],
        lw=SIG_LW,
        color='black',
        clip_on=False
    )
    ax.text(
        (x1 + x2) / 2,
        y + h,
        text,
        ha='center',
        va='bottom',
        color='black',
        fontsize=fontsize,
        fontweight='bold'
    )


# ==========================================
# 4. 绘图：每个量表一个小图，Control/TMS 放在同一张图里
# ==========================================
fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(7.2, 2.45))
fig.subplots_adjust(left=0.095, right=0.995, bottom=0.25, top=0.80, wspace=0.42)

legend_used = set()

for col_idx, scale in enumerate(scales):
    ax = axes[col_idx]

    if scale in ['MMSE', 'VFT']:
        time_cols = [f'{scale}_pre', f'{scale}_post', f'{scale}_M1']
        time_labels = ['Pre', 'Post', '1-month follow-up']
        offsets = np.array([-0.18, 0.00, 0.18])
    else:
        time_cols = [f'{scale}_pre', f'{scale}_post']
        time_labels = ['Pre', 'Post']
        offsets = np.array([-0.13, 0.13])

    all_values = []
    sig_infos = []

    for group_idx, group in enumerate(groups):
        group_data = df_merged[df_merged['Group'] == group]
        plot_data = group_data[time_cols].apply(pd.to_numeric, errors='coerce').dropna(how='all')

        x_positions = group_idx + offsets

        # 个体配对线 + 彩色散点
        for _, row in plot_data.iterrows():
            y_vals = row.to_numpy(dtype=float)
            valid_mask = ~np.isnan(y_vals)

            if valid_mask.sum() >= 2:
                ax.plot(
                    x_positions[valid_mask],
                    y_vals[valid_mask],
                    color='black',
                    alpha=0.35,
                    linewidth=0.55,
                    zorder=1
                )

            for i, label in enumerate(time_labels):
                if not np.isnan(y_vals[i]):
                    ax.scatter(
                        x_positions[i],
                        y_vals[i],
                        s=POINT_SIZE,
                        color=TIME_COLORS[label],
                        edgecolor='black',
                        linewidth=POINT_EDGE_LW,
                        alpha=0.95,
                        zorder=3
                    )
                    legend_used.add(label)

        vals = plot_data.to_numpy(dtype=float).ravel()
        all_values.extend(vals[~np.isnan(vals)].tolist())

        # Pre vs Post
        valid_post = plot_data[[time_cols[0], time_cols[1]]].dropna()
        if len(valid_post) > 2:
            _, p_post = stats.ttest_rel(valid_post.iloc[:, 0], valid_post.iloc[:, 1])
            sig_infos.append({
                "x1": x_positions[0],
                "x2": x_positions[1],
                "p": p_post,
                "level": 0
            })

        # Pre vs M1，仅 MMSE/VFT
        if len(time_cols) == 3:
            valid_m1 = plot_data[[time_cols[0], time_cols[2]]].dropna()
            if len(valid_m1) > 2:
                _, p_m1 = stats.ttest_rel(valid_m1.iloc[:, 0], valid_m1.iloc[:, 1])
                sig_infos.append({
                    "x1": x_positions[0],
                    "x2": x_positions[2],
                    "p": p_m1,
                    "level": 1
                })

    if len(all_values) == 0:
        continue

    overall_min_y = np.nanmin(all_values)
    overall_max_y = np.nanmax(all_values)
    y_range = overall_max_y - overall_min_y
    if y_range == 0:
        y_range = 1

    # 显著性括号
    bar_h = y_range * 0.06
    level_step = y_range * 0.22
    y_base = overall_max_y + y_range * 0.08
    max_level = 0

    for info in sig_infos:
        star = get_star(info["p"])
        if SHOW_NS or star != "ns":
            y = y_base + info["level"] * level_step
            draw_sig_bar(ax, info["x1"], info["x2"], y, bar_h, star)
            max_level = max(max_level, info["level"])

    # ==========================================
    # 新增：TMS vs Control 改善量比较
    # 在最上方画一条跨 Control 和 TMS 的括号
    # ==========================================
    if SHOW_BETWEEN_DELTA:
        delta_res = compare_delta_between_groups(
            df_merged=df_merged,
            scale=scale,
            target_suffix="post"
        )

        if delta_res is not None:
            star_delta = delta_res["star"]

            # 只有 TMS 改善量大于 Control，并且显著时才标
            # 如果你想不管方向都标显著性，可以去掉 delta_res["mean_diff"] > 0 这个条件
            if delta_res["mean_diff"] > 0 and (SHOW_BETWEEN_NS or star_delta != "ns"):

                # 放在已有组内显著性括号的上方
                between_level = max_level + 1
                y_between = y_base + between_level * level_step

                # 文字尽量短，避免和图重叠
                # Δ 表示 improvement/change score 的组间比较
                label_delta = f"{BETWEEN_LABEL_PREFIX} {star_delta}"

                draw_sig_bar(
                    ax,
                    0,      # Control 组中心
                    1,      # TMS 组中心
                    y_between,
                    bar_h,
                    label_delta,
                    fontsize=FS_STAR
                )

                max_level = between_level
    # 坐标轴
    ax.set_xlim(-0.45, 1.45)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(groups, fontsize=FS_TICK, fontweight='bold')
    ax.tick_params(axis='y', labelsize=FS_TICK, width=AXIS_LW, length=4.5)
    ax.tick_params(axis='x', width=AXIS_LW, length=4.5)

    bottom_y = max(0, overall_min_y - y_range * 0.12)
    top_y = y_base + (max_level + 1) * level_step + y_range * 0.18
    ax.set_ylim(bottom=bottom_y, top=top_y)

    ax.set_title(scale, fontsize=FS_TITLE, fontweight='bold', pad=3)

    if col_idx == 0:
        ax.set_ylabel("Clinical score", fontsize=FS_LABEL, fontweight='bold')

    # 第二张图类似：只保留左轴和下轴
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(AXIS_LW)
    ax.spines['bottom'].set_linewidth(AXIS_LW)


# 图例
legend_order = ["Pre", "Post", "1-month follow-up"]
handles = [
    Line2D(
        [0], [0],
        marker='o',
        color='black',
        markerfacecolor=TIME_COLORS[label],
        markeredgecolor='black',
        markersize=4,
        linewidth=0.8,
        label=label
    )
    for label in legend_order if label in legend_used
]

fig.legend(
    handles=handles,
    loc='lower center',
    bbox_to_anchor=(0.5, 0.055),
    ncol=len(handles),
    frameon=False,
    fontsize=FS_LEGEND,
    handlelength=1.0,
    handletextpad=0.35,
    columnspacing=0.9,
    borderaxespad=0.0
)

# ==========================================
# 5. 保存结果
# ==========================================
output_dir = os.environ.get("GRADIENT_STROKE_FIG3A_OUTPUT", os.path.join("outputs", "figure3", "g"))
os.makedirs(output_dir, exist_ok=True)

enforce_arial(fig)

save_base = os.path.join(output_dir, "clinical_scores_second_style")
save_paths = {
    "png": f"{save_base}.png",
    "svg": f"{save_base}.svg",
    "pdf": f"{save_base}.pdf",
    "tiff": f"{save_base}.tiff",
}

fig.savefig(save_paths["png"], dpi=600, bbox_inches='tight', pad_inches=0.02)
fig.savefig(save_paths["svg"], bbox_inches='tight', pad_inches=0.02)
fig.savefig(save_paths["pdf"], bbox_inches='tight', pad_inches=0.02)
fig.savefig(save_paths["tiff"], dpi=600, bbox_inches='tight', pad_inches=0.02)
plt.close(fig)

print("Figure saved:")
for path in save_paths.values():
    print(path)
