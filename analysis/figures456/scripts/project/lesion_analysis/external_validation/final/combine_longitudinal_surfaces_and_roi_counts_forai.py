from __future__ import annotations
import os as _portable_os
from pathlib import Path as _PortablePath

def _portable_path(kind, relative):
    project = _PortablePath(_portable_os.environ['PAPER_PROJECT_ROOT'])
    root = project if kind == 'project' else project.parent
    return str(root / relative)
import argparse
import sys
import tempfile
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, TwoSlopeNorm
from PIL import Image
HERE = Path(__file__).resolve().parent
SOURCE_DIR = HERE.parent / 'ds003999_validation' / 'matched_gradient_figure'
OUTPUT_DIR = HERE / 'forai'
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))
import make_matched_gradient_nature_figure as source
NETWORK_ORDER = ['Visual', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Control', 'Default']
NETWORK_LABELS = {'Visual': 'VIS', 'SomMot': 'SMN', 'DorsAttn': 'DAN', 'SalVentAttn': 'VAN', 'Limbic': 'LIM', 'Control': 'CON', 'Default': 'DMN'}
SURFACE_CMAPS = {'G1': mpl.colormaps['PiYG'], 'G2': mpl.colormaps['PuOr']}
COUNT_COLORS = {'G1': {'negative': '#B76E99', 'positive': '#7FA65A'}, 'G2': {'negative': '#C58A52', 'positive': '#7D6CA5'}}
INK = '#272727'
MID = '#767676'
ZERO = '#A8A8A8'

def apply_style() -> None:
    mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'], 'font.size': 7.0, 'axes.labelsize': 7.0, 'axes.titlesize': 8.0, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'axes.linewidth': 0.65, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False, 'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'axes.unicode_minus': False, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})

def lightened_cmap(cmap, white_fraction: float=0.62) -> LinearSegmentedColormap:
    colors = cmap(np.linspace(0.0, 1.0, 256))
    colors[:, :3] = colors[:, :3] * (1.0 - white_fraction) + white_fraction
    return LinearSegmentedColormap.from_list(f'{cmap.name}_faint', colors, N=256)

def render_highlight_plate(all_t_values: np.ndarray, significant_mask: np.ndarray, gradient: str, limit: float, output_dir: Path) -> Path:
    output = output_dir / f'{gradient}_paired_t_highlight_plate.png'
    surfaces = source.fetch_fslr()
    (lh_surf, rh_surf) = surfaces['veryinflated']
    atlas = np.asarray(source.load_parcellation('schaefer', 400, join=True))
    medial_wall = np.full_like(atlas, np.nan, dtype=float)
    medial_wall[atlas == 0] = 1.0
    all_vertices = source.roi_to_vertices(all_t_values)
    all_vertices[atlas == 0] = np.nan
    significant_values = np.where(significant_mask, all_t_values, 0.0)
    significant_vertices = source.roi_to_vertices(significant_values)
    significant_vertices[atlas == 0] = np.nan
    outline_vertices = source.roi_to_vertices(significant_mask.astype(float))
    outline_vertices[atlas == 0] = np.nan
    plot = source.Plot(lh_surf, rh_surf, views=['lateral', 'medial'], size=(6400, 4200), zoom=1.7, background=(1, 1, 1), brightness=0.82)
    plot.add_layer(medial_wall, cmap=ListedColormap(['#D1D1D1']), color_range=(0, 1), cbar=False)
    plot.add_layer(all_vertices, cmap=lightened_cmap(SURFACE_CMAPS[gradient]), color_range=(-limit, limit), cbar=False)
    plot.add_layer(significant_vertices, cmap=SURFACE_CMAPS[gradient], color_range=(-limit, limit), cbar=False, zero_transparent=True)
    plot.add_layer(atlas, cmap=ListedColormap(['#9A9A9A']), as_outline=True, cbar=False)
    plot.add_layer(outline_vertices, cmap=ListedColormap(['#303030']), as_outline=True, cbar=False, alpha=1.0, zero_transparent=True)
    built = plot.build()
    built.patch.set_facecolor('white')
    built.set_size_inches(12.0, 8.0, forward=True)
    built.savefig(output, dpi=400, bbox_inches=None, pad_inches=0.0, facecolor='white')
    plt.close(built)
    return output

def render_surface_strip(roi_values: np.ndarray, gradient: str, limit: float, output_dir: Path, surface_mode: str, significant_mask: np.ndarray) -> Path:
    output = output_dir / f'{gradient}_paired_t_surface_strip.png'
    if surface_mode == 'highlight':
        plate_path = render_highlight_plate(roi_values, significant_mask, gradient, limit, output_dir)
    else:
        source.SURFACE_CACHE = output_dir
        plate_path = source.render_surface(roi_values, f'{gradient}_paired_t_surface_plate_for_combination', SURFACE_CMAPS[gradient], (-limit, limit), mask_zeros=surface_mode == 'nominal')

    def trim_white(tile: Image.Image) -> Image.Image:
        rgb = np.asarray(tile.convert('RGB'))
        mask = np.any(rgb < 248, axis=2)
        (rows, columns) = np.where(mask)
        if not len(rows):
            return tile.convert('RGB')
        pad_x = max(12, int(tile.width * 0.025))
        pad_y = max(12, int(tile.height * 0.035))
        left = max(0, int(columns.min()) - pad_x)
        right = min(tile.width, int(columns.max()) + pad_x + 1)
        top = max(0, int(rows.min()) - pad_y)
        bottom = min(tile.height, int(rows.max()) + pad_y + 1)
        return tile.crop((left, top, right, bottom)).convert('RGB')
    with Image.open(plate_path) as plate_source:
        plate = plate_source.convert('RGB')
        (width, height) = plate.size
        quadrants = [plate.crop((0, 0, width // 2, height // 2)), plate.crop((width // 2, 0, width, height // 2)), plate.crop((0, height // 2, width // 2, height)), plate.crop((width // 2, height // 2, width, height))]
        tiles = [trim_white(tile) for tile in quadrants]
    target_height = 760
    resized: list[Image.Image] = []
    for tile in tiles:
        target_width = max(1, int(round(tile.width * target_height / tile.height)))
        resized.append(tile.resize((target_width, target_height), Image.Resampling.LANCZOS))
    gap_x = 55
    gap_y = 38
    column_widths = [max(resized[0].width, resized[2].width), max(resized[1].width, resized[3].width)]
    canvas = Image.new('RGB', (sum(column_widths) + gap_x, target_height * 2 + gap_y), 'white')
    for (index, tile) in enumerate(resized):
        (row, column) = divmod(index, 2)
        x_origin = 0 if column == 0 else column_widths[0] + gap_x
        x = x_origin + (column_widths[column] - tile.width) // 2
        y = row * (target_height + gap_y)
        canvas.paste(tile, (x, y))
    canvas.save(output, dpi=(350, 350))
    return output

def prepare_surface_strips(roi_stats: pd.DataFrame, cache_dir: Path, surface_mode: str) -> dict[str, tuple[Path, float, int]]:
    outputs: dict[str, tuple[Path, float, int]] = {}
    for gradient in ('G1', 'G2'):
        frame = roi_stats.query('gradient == @gradient').sort_values('roi')
        significant = frame['nominal_p_lt_0_05'].to_numpy(bool)
        all_t_values = frame['t'].to_numpy(float)
        values = np.where(significant, all_t_values, 0.0) if surface_mode == 'nominal' else all_t_values
        limit = float(np.ceil(np.max(np.abs(values)) * 10.0) / 10.0)
        image = render_surface_strip(values, gradient, limit, cache_dir, surface_mode=surface_mode, significant_mask=significant)
        outputs[gradient] = (image, limit, int(significant.sum()))
    return outputs

def add_surface_panel(fig: plt.Figure, slot, gradient: str, panel_label: str, image_path: Path, limit: float, significant_n: int, surface_mode: str) -> None:
    subgrid = slot.subgridspec(2, 3, height_ratios=[16.5, 1.0], width_ratios=[0.55, 3.8, 0.55], hspace=0.05)
    ax = fig.add_subplot(subgrid[0, :])
    ax.imshow(plt.imread(image_path), interpolation='none')
    ax.set_axis_off()
    if surface_mode == 'all':
        title = f'{gradient} paired t map · all 400 ROIs'
    elif surface_mode == 'highlight':
        title = f'{gradient} paired t map · p < 0.05 ROIs highlighted'
    else:
        title = f'{gradient} longitudinal ROI effects · {significant_n} ROIs'
    ax.set_title(title, loc='left', fontweight='bold', pad=1.5)
    ax.text(-0.025, 1.06, panel_label, transform=ax.transAxes, ha='left', va='bottom', fontsize=9.0, fontweight='bold', color=INK)
    cax = fig.add_subplot(subgrid[1, 1])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit), cmap=SURFACE_CMAPS[gradient]), cax=cax, orientation='horizontal')
    cb.set_ticks([-limit, 0.0, limit])
    cb.set_label('Paired t (post − pre)', labelpad=1.2)
    cb.ax.tick_params(length=2.2, width=0.6, pad=1.0)
    cb.outline.set_linewidth(0.55)

def add_count_panel(ax: plt.Axes, counts: pd.DataFrame, gradient: str, panel_label: str | None) -> None:
    frame = counts.loc[counts['gradient'].eq(gradient)].set_index('network').reindex(NETWORK_ORDER)
    negative = frame['nominal_negative_n'].to_numpy(int)
    positive = frame['nominal_positive_n'].to_numpy(int)
    y = np.arange(len(NETWORK_ORDER))
    colors = COUNT_COLORS[gradient]
    ax.barh(y, -negative, height=0.56, color=colors['negative'], edgecolor='white', linewidth=0.4, zorder=2)
    ax.barh(y, positive, height=0.56, color=colors['positive'], edgecolor='white', linewidth=0.4, zorder=2)
    ax.axvline(0.0, color=ZERO, linewidth=0.7, zorder=1)
    ax.set_xlim(-5.1, 5.1)
    ax.set_xticks([-4, -2, 0, 2, 4], ['4', '2', '0', '2', '4'])
    ax.set_yticks(y, [NETWORK_LABELS[name] for name in NETWORK_ORDER])
    ax.invert_yaxis()
    ax.tick_params(axis='y', length=0, pad=2.5)
    ax.tick_params(axis='x', length=2.2, width=0.65, pad=1.5)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color(INK)
    ax.spines['bottom'].set_linewidth(0.65)
    ax.set_title(f'{gradient} network counts', loc='left', fontweight='bold', pad=12)
    ax.text(0.25, 1.005, 'negative t', transform=ax.transAxes, ha='center', va='bottom', fontsize=6.0, color=colors['negative'], fontweight='bold')
    ax.text(0.75, 1.005, 'positive t', transform=ax.transAxes, ha='center', va='bottom', fontsize=6.0, color=colors['positive'], fontweight='bold')
    if gradient == 'G2':
        ax.set_xlabel('Nominal ROI count', labelpad=2.0)
    if panel_label:
        ax.text(-0.2, 1.1, panel_label, transform=ax.transAxes, ha='left', va='bottom', fontsize=9.0, fontweight='bold', color=INK)

def make_combined_figure(roi_stats: pd.DataFrame, counts: pd.DataFrame, surface_strips: dict[str, tuple[Path, float, int]], surface_mode: str) -> plt.Figure:
    (width_mm, height_mm) = (183.0, 132.0)
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), facecolor='white')
    grid = fig.add_gridspec(2, 2, width_ratios=[1.55, 1.0], height_ratios=[1.0, 1.0], left=0.055, right=0.985, top=0.825, bottom=0.12, wspace=0.16, hspace=0.37)
    for (row, (gradient, panel_label)) in enumerate((('G1', 'a'), ('G2', 'b'))):
        (image, limit, significant_n) = surface_strips[gradient]
        add_surface_panel(fig, grid[row, 0], gradient, panel_label, image, limit, significant_n, surface_mode)
        add_count_panel(fig.add_subplot(grid[row, 1]), counts, gradient, 'c' if row == 0 else None)
    fig.text(0.055, 0.965, 'Longitudinal cortical-gradient effects and network distribution', ha='left', va='top', fontsize=10.0, fontweight='bold', color=INK)
    fig.text(0.055, 0.915, 'Whole-cortex paired t maps with nominal effects highlighted and quantified by network' if surface_mode == 'highlight' else 'Whole-cortex paired t maps with direction-specific nominal ROI counts in the external-validation cohort' if surface_mode == 'all' else 'Spatial localization and direction-specific nominal ROI counts in the external-validation cohort', ha='left', va='top', fontsize=6.7, color=MID)
    fig.text(0.055, 0.025, 'Surfaces show all ROI t values in faint color; saturated, dark-outlined ROIs and bars indicate paired, two-sided p<0.05 (uncorrected); n=29.' if surface_mode == 'highlight' else 'Cortical surfaces show paired t values for all 400 ROIs; bars count paired, two-sided p<0.05 ROIs (uncorrected); descriptive counts; n=29.' if surface_mode == 'all' else 'Colored cortex and bars: paired, two-sided p<0.05 (uncorrected); non-significant ROIs are white; descriptive counts; n=29.', ha='left', va='bottom', fontsize=6.2, color=MID)
    return fig

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Combine cortical paired-t surfaces with nominal network ROI counts.')
    parser.add_argument('--surface-mode', choices=('nominal', 'all', 'highlight'), default='nominal', help='Show nominal ROIs, all t values, or all t values with nominal ROIs highlighted.')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    apply_style()
    roi_stats = pd.read_csv(SOURCE_DIR / 'tables' / 'raw_aligned5_roi_paired_tests.csv')
    counts = pd.read_csv(SOURCE_DIR / 'source_data' / 'network_nominal_roi_counts.csv')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.surface_mode == 'all':
        output_pdf = OUTPUT_DIR / 'abc_G1_G2_longitudinal_ROI_effects_network_counts_all_t.pdf'
    elif args.surface_mode == 'highlight':
        output_pdf = OUTPUT_DIR / 'abc_G1_G2_longitudinal_ROI_effects_network_counts_highlighted.pdf'
    else:
        output_pdf = OUTPUT_DIR / 'abc_G1_G2_longitudinal_ROI_effects_network_counts.pdf'
    with tempfile.TemporaryDirectory(prefix='combined_gradient_surface_') as cache:
        surface_strips = prepare_surface_strips(roi_stats, Path(cache), args.surface_mode)
        fig = make_combined_figure(roi_stats, counts, surface_strips, args.surface_mode)
        fig.savefig(output_pdf, format='pdf', bbox_inches='tight', pad_inches=0.035, facecolor='white', metadata={'Creator': 'Python / Matplotlib', 'Title': 'Longitudinal cortical-gradient effects and network distribution', 'Subject': 'Nominal longitudinal ROI effects in 29 paired participants'})
        qa_png = Path(tempfile.gettempdir()) / f'abc_longitudinal_roi_effects_network_counts_{args.surface_mode}_qa.png'
        fig.savefig(qa_png, format='png', dpi=300, bbox_inches='tight', pad_inches=0.035, facecolor='white')
        plt.close(fig)
    print(f'Saved: {output_pdf}')
    print(f'QA preview: {qa_png}')
    print(f'Surface mode: {args.surface_mode}')
    for gradient in ('G1', 'G2'):
        (_, limit, significant_n) = surface_strips[gradient]
        subset = counts.loc[counts['gradient'].eq(gradient)]
        negative_n = int(subset['nominal_negative_n'].sum())
        positive_n = int(subset['nominal_positive_n'].sum())
        print(f'{gradient}: surface range=±{limit:.1f}; significant={significant_n}; negative={negative_n}; positive={positive_n}')
if __name__ == '__main__':
    main()
