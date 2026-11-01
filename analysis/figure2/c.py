#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure 2b (legacy file/output names Figure2c): spatial-spin network enrichment based *exactly* on Figure 2a.

Input = ROI CSV produced by a.py, NOT the original mediation code.
Each group x gradient has 400 paired-test ROI rows with raw P and BH-FDR q.

Selection for exploratory network enrichment:
  increase = (P_value < .05) & (Mean_delta_post_minus_pre > 0)
  decrease = (P_value < .05) & (Mean_delta_post_minus_pre < 0)
The ROI-level P threshold is nominal and is not corrected across 400 ROIs.

For each selected mask, hemisphere-coupled random sphere rotations are applied
on 400 Schaefer parcel spherical centroids, with one-to-one Hungarian matching
within hemisphere. Network enrichment is tested with a one-sided upper-tail
spin P-value and BH-FDR across the complete 2 x 2 x 2 x 7 = 56 test family.

This is an ROI-centroid spin approximation. It conserves the number of selected
ROIs and avoids duplicate labels but is not identical to vertex-level spin tests.
Validate the chosen spatial null in the publication Methods.

Eight independent plots and audit tables are written to D:\\Codex_files\\Figure2c_eight_panels.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation
from statsmodels.stats.multitest import multipletests

# ========== 路径：已经与 a.py 的输出衔接 ==========
SOURCE_CSV = Path(r'F:\lishiqi\end_picture\Ending\main 2\a\Figure2a_ROI_mean_delta_and_FDR.csv')
OUTPUT_DIR = Path(r'F:\lishiqi\end_picture\Ending\main 2\c')
LH_ANNOT = Path(r'F:\lishiqi\预处理数据\cortex\lh.Schaefer2018_400Parcels_7Networks_order.annot')
RH_ANNOT = Path(r'F:\lishiqi\预处理数据\cortex\rh.Schaefer2018_400Parcels_7Networks_order.annot')

GROUPS = ('Control', 'TMS')
GRADIENTS = ('G1', 'G2')
DIRECTIONS = ('Increase', 'Decrease')
NETWORKS = ('VIS', 'SMN', 'DAN', 'VAN', 'LIM', 'FPN', 'DMN')
EXPECTED_N = {'Control': 19, 'TMS': 23}  # 与 a.py 完全相同
N_ROI = 400
FDR_ALPHA = 0.05  # Used to validate Figure 2a FDR flags and spin-test q values
ROI_P_ALPHA = 0.05  # Nominal paired-test P threshold for ROI selection
N_SPINS = 5000
SEED = 20261009

NETWORK_COLORS = ['#FDC894', '#E27C87', '#70A8CE', '#7EC8A2',
                  '#F79C7F', '#BFE5BB', '#FEEAB0']
INCREASE_COLOR = '#BB615D'
DECREASE_COLOR = '#3F769C'
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 11,
    'axes.spines.right': False, 'axes.spines.top': False,
    'axes.linewidth': .8, 'figure.facecolor': 'white',
    'savefig.facecolor': 'white', 'svg.fonttype': 'none',
    'pdf.fonttype': 42, 'axes.unicode_minus': False,
})


def read_figure2a_csv(path: Path) -> tuple[pd.DataFrame, dict]:
    """Check exact ROI set, subject Ns, q values and sign used in Figure 2a."""
    if not path.is_file():
        raise FileNotFoundError(
            f'找不到 Fig. 2a 结果文件：{path}\n请先运行你上传的 a.py。'
        )
    df = pd.read_csv(path, encoding='utf-8-sig')
    required = {'Group','Gradient','N','ROI_ID',
                'Mean_delta_post_minus_pre','P_value','BH_FDR_q','FDR_significant'}
    if not required.issubset(df.columns):
        raise ValueError('Fig. 2a CSV 缺少字段: '+repr(sorted(required-set(df.columns))))
    if len(df) != 1600:
        raise ValueError(f'Fig. 2a 必须包含 2组×2梯度×400ROI=1600 行，实际 {len(df)} 行')
    for key in ('N','ROI_ID','Mean_delta_post_minus_pre','P_value','BH_FDR_q'):
        df[key] = pd.to_numeric(df[key], errors='coerce')
    if not np.isfinite(df['Mean_delta_post_minus_pre']).all():
        raise ValueError('Fig. 2a 的 mean ΔG 存在缺失或无穷值，请先核对数据')
    result = {}
    for group in GROUPS:
        for grad in GRADIENTS:
            x = df[(df.Group==group)&(df.Gradient==grad)].sort_values('ROI_ID').copy()
            if len(x) != N_ROI or not np.array_equal(x.ROI_ID.to_numpy(),np.arange(1,401)):
                raise ValueError(f'{group} {grad}: ROI_ID 必须严格是 1..400，不能缺失或重复')
            if x.N.nunique()!=1 or int(x.N.iloc[0])!=EXPECTED_N[group]:
                raise ValueError(f'{group} {grad}: 被试数与 a.py 不一致，预期 {EXPECTED_N[group]}')
            p=x.P_value.to_numpy(float)
            q=x.BH_FDR_q.to_numpy(float)
            delta=x.Mean_delta_post_minus_pre.to_numpy(float)
            valid_p=np.isfinite(p)
            valid_q=np.isfinite(q)
            if not valid_p.all() or ((p[valid_p] < 0)|(p[valid_p]>1)).any():
                raise ValueError(f'{group} {grad}: 原始 P 值缺失或超出 0..1')
            if ((q[valid_q] < 0)|(q[valid_q]>1)).any():
                raise ValueError(f'{group} {grad}: q 值超出 0..1')
            fdr_selected=valid_q&(q<FDR_ALPHA)
            check=x.FDR_significant.astype(str).str.lower().str.strip().isin(('true','1')).to_numpy()
            if not np.array_equal(fdr_selected,check):
                raise ValueError(f'{group} {grad}: FDR_significant 与 q<0.05 不一致')
            # Use nominal paired-test P<.05 for the ROI masks, as requested.
            # These ROI selections are exploratory because 400 ROI tests are not
            # corrected at this selection stage; the spatial-spin family is
            # corrected separately below.
            chosen=valid_p&(p<ROI_P_ALPHA)
            plus=chosen&(delta>0)
            minus=chosen&(delta<0)
            if np.count_nonzero(chosen & ~(plus|minus)):
                warnings.warn(f'{group} {grad}: P<0.05 且 ΔG=0，未纳入任何方向')
            result[group,grad,'Increase']=plus
            result[group,grad,'Decrease']=minus
            print(f'[INPUT] {group:7s} {grad}: N={EXPECTED_N[group]}, nominal P<0.05 ROI={chosen.sum()}, '
                  f'Increase={plus.sum()}, Decrease={minus.sum()}')
    if len(result)!=8:
        raise AssertionError('Unexpected group-gradient-direction contrasts')
    return df,result


def schaefer_networks(lh_annot: Path, rh_annot: Path) -> pd.DataFrame:
    """Schaefer 7-network ROI table sorted LH 1..200 then RH 1..200."""
    try:
        from nibabel.freesurfer.io import read_annot
    except ImportError as exc:
        raise ImportError('需要安装 nibabel：pip install nibabel') from exc
    names_map={
        'Vis':'VIS', 'SomMot':'SMN', 'DorsAttn':'DAN',
        'SalVentAttn':'VAN', 'Limbic':'LIM', 'Cont':'FPN',
        'Default':'DMN'
    }
    rows=[]
    for hemi,path in [('LH',lh_annot),('RH',rh_annot)]:
        if not path.is_file():
            raise FileNotFoundError(f'找不到 Schaefer 注释图谱：{path}')
        _,_,names=read_annot(str(path))
        records=[]
        for name in names:
            nm=name.decode('utf-8') if isinstance(name,(bytes,bytearray)) else str(name)
            if any(s in nm.lower() for s in ('unknown','medial','wall')):
                continue
            parts=nm.split('_')
            if len(parts)<4 or parts[1]!=hemi:
                continue
            # The trailing parcel number is local to each network and can repeat
            # across networks. Keep the FreeSurfer color-table order, which is the
            # Schaefer *_order.annot parcel order used by ROI_ID 1..200 per hemi.
            name_part=parts[2]
            if name_part not in names_map:
                raise ValueError(f'无法识别的网络 {name_part}，ROI {nm}')
            records.append((nm,names_map[name_part]))
        if len(records)!=200:
            raise ValueError(f'{hemi} 图谱应对应200个ROI，实际 {len(records)}')
        if len({r[0] for r in records})!=200:
            raise ValueError(f'{hemi} 图谱包含重复的 ROI 名称')
        for idx,(full_name,network) in enumerate(records):
            roi=idx+1+(0 if hemi=='LH' else 200)
            rows.append({'ROI_ID':roi,'Hemisphere':hemi,
                         'ROI_name':full_name,'Network':network})
    lookup=pd.DataFrame(rows)
    assert len(lookup)==400
    return lookup


def centroid_spheres():
    """Compute 400 normalized spherical centroids using the *same* fsLR atlas as a(5)."""
    try:
        import nibabel as nib
        from neuromaps.datasets import fetch_fslr
        from brainspace.datasets import load_parcellation
    except ImportError as exc:
        raise ImportError('请安装 nibabel neuromaps brainspace') from exc
    fs=fetch_fslr(density='32k')
    if 'sphere' not in fs:
        raise RuntimeError('neuromaps fsLR 下载缺少 sphere 表面')
    sphere_left,sphere_right=fs['sphere']
    coords=[]
    for path in (sphere_left,sphere_right):
        # A GIFTI surface contains both POINTSET coordinates and TRIANGLE indices.
        # agg_data() without an intent returns both arrays as a tuple; select only
        # the pointset coordinates before converting to a NumPy array.
        surface=nib.load(str(path))
        pointset=surface.agg_data('NIFTI_INTENT_POINTSET')
        if isinstance(pointset,tuple):
            if len(pointset)!=1:
                raise ValueError(f'Sphere 表面应只有一个 POINTSET 数组：{path}')
            pointset=pointset[0]
        verts=np.asarray(pointset,dtype=float)
        if verts.ndim!=2 or verts.shape[1]!=3 or not np.isfinite(verts).all():
            raise ValueError(f'Sphere 顶点坐标格式不正确：{path} {verts.shape}')
        coords.append(verts)
    atlas=np.asarray(load_parcellation('schaefer',400,join=True)).ravel()
    n_l,n_r=map(len,coords)
    if len(atlas)!=n_l+n_r:
        raise ValueError('fsLR 32k 球面与 Schaefer400 顶点数不匹配')
    if np.any((atlas[:n_l]>200)&(atlas[:n_l]!=0)) or np.any((atlas[n_l:]<=200)&(atlas[n_l:]!=0)):
        raise ValueError('Schaefer 标签没有按左1-200/右201-400排列，请核查ROI编号')
    out=np.empty((400,3),dtype=float)
    for i in range(400):
        hemi=0 if i<200 else 1
        labels=atlas[:n_l] if hemi==0 else atlas[n_l:]
        sample=coords[hemi][labels==i+1]
        if len(sample)==0:
            raise ValueError(f'fsLR parcellation 中 ROI {i+1} 没有顶点')
        c=sample.mean(axis=0)
        norm=np.linalg.norm(c)
        if not np.isfinite(norm) or norm<=1e-8:
            raise ValueError(f'ROI {i+1} 球面重心无效')
        out[i]=c/norm
    return out


def generate_spins(points: np.ndarray, count: int, seed: int, cache_file: Path) -> np.ndarray:
    """Hemisphere-paired rotation + one-to-one parcel assignment; mask[permutation]."""
    if cache_file.is_file():
        spins=np.load(cache_file,allow_pickle=False)
        if spins.shape==(count,400):
            print(f'[CACHE] 使用已有 {count} 次旋转：{cache_file}')
            return spins
        raise ValueError('Spin 缓存形状错误。请删除该缓存后重新生成。')
    if points.shape!=(400,3):
        raise ValueError('Expected 400 unit spherical centroids')
    rg=np.random.default_rng(seed)
    reflection=np.diag([-1.,1.,1.])
    out=np.empty((count,400),dtype=np.int16)
    for k in range(count):
        rot=Rotation.random(random_state=rg).as_matrix()
        for hemi in (0,1):
            idx=slice(hemi*200,(hemi+1)*200)
            points_h=points[idx]
            rr=rot if hemi==0 else reflection@rot@reflection
            rotated=points_h@rr.T
            # Hungarian matching: every ROI is used exactly once per hemisphere.
            cost=1-np.clip(rotated@points_h.T,-1,1)
            source,destination=linear_sum_assignment(cost)
            out[k,hemi*200+destination]=hemi*200+source
        if (k+1)%500==0 or k+1==count:
            print(f'[SPIN] {k+1}/{count}')
    if not np.all(np.sort(out,axis=1)==np.arange(400)):
        raise AssertionError('Spin indices are not bijective')
    np.save(cache_file,out)
    print(f'[SAVE] {cache_file}')
    return out


def analyze(selected_masks: dict, lookup: pd.DataFrame,
            spins: np.ndarray | None) -> tuple[pd.DataFrame, dict]:
    """56 tests. Missing masks do not receive artificial p-values."""
    networks=lookup.sort_values('ROI_ID').Network.to_numpy()
    network_masks=np.column_stack([networks==nm for nm in NETWORKS]).astype(np.int16)
    total_in_network=network_masks.sum(axis=0)
    rows=[]
    null_distributions={}
    for group in GROUPS:
        for grad in GRADIENTS:
            for direction in DIRECTIONS:
                mask=selected_masks[group,grad,direction].astype(np.int8)
                n_sel=int(mask.sum())
                if n_sel and spins is None:
                    raise ValueError('非空 ROI 集合需要实际的球面旋转 Spin')
                obs_counts=mask@network_masks
                obs_fractions=obs_counts/n_sel if n_sel else np.full(7,np.nan)
                if n_sel:
                    # Every null map has exactly n_sel selected parcels.
                    rotated_masks=mask[spins]
                    if not np.all(rotated_masks.sum(axis=1)==n_sel):
                        raise AssertionError('Rotated ROI counts changed')
                    null_counts=rotated_masks@network_masks
                    null_fractions=null_counts/n_sel
                for ni,network in enumerate(NETWORKS):
                    key=group,grad,direction,network
                    if n_sel:
                        null=null_fractions[:,ni]
                        observed=float(obs_fractions[ni])
                        p=(1+int(np.count_nonzero(null>=observed-1e-12)))/(len(null)+1)
                        q025,q975=np.quantile(null,[.025,.975])
                        null_mean=float(null.mean())
                        # Intuitive fraction of selected parcels / spatial-null expectation.
                        enrichment=observed/null_mean if null_mean>0 else np.nan
                        null_distributions[key]=null.copy()
                    else:
                        p=q025=q975=null_mean=enrichment=np.nan
                        observed=np.nan
                    rows.append({
                        'Group':group,'Gradient':grad,'Direction':direction,
                        'Network':network,'TotalROI_in_network':int(total_in_network[ni]),
                        'N_selected_ROI':n_sel,'N_selected_in_network':int(obs_counts[ni]),
                        'Observed_fraction':observed,'Null_mean_fraction':null_mean,
                        'Enrichment_ratio_vs_null':enrichment,
                        'Null_fraction_2_5pct':q025,'Null_fraction_97_5pct':q975,
                        'P_spin_one_sided':p,
                        'Status':'Tested' if n_sel else 'Not tested: no nominal P<0.05 ROIs'
                    })
    results=pd.DataFrame(rows)
    assert len(results)==56
    # Explicit full family (56 combinations); empty sets count as p=1 in
    # adjustment only, then the q output remains NA (not tested).
    p_family=results['P_spin_one_sided'].fillna(1.0).to_numpy(float)
    results['Q_spin_BH_FDR56']=multipletests(p_family,method='fdr_bh')[1]
    results.loc[results['N_selected_ROI']==0,'Q_spin_BH_FDR56']=np.nan
    results['Spin_FDR_significant']=results['Q_spin_BH_FDR56']<.05
    results['Spin_nominal_significant']=results['P_spin_one_sided']<.05
    return results,null_distributions


def save_figure(results: pd.DataFrame, nulls: dict, directory: Path) -> None:
    """Save eight independent violin plots: group x gradient x direction.

    Plot styling only: the selected ROIs, spatial-spin null distributions,
    observed fractions, and nominal P values are computed upstream unchanged.
    """
    observed_color = '#5B5B5B'
    for group in GROUPS:
        for grad in GRADIENTS:
            for direction in DIRECTIONS:
                sub = results[(results.Group == group) &
                              (results.Gradient == grad) &
                              (results.Direction == direction)].set_index('Network')
                n_sel = int(sub.N_selected_ROI.iloc[0])
                fig, ax = plt.subplots(figsize=(9.2, 6.0))
                if n_sel:
                    for i, network in enumerate(NETWORKS):
                        row = sub.loc[network]
                        dist = nulls[(group, grad, direction, network)]
                        if np.nanstd(dist) > 1e-8 and np.unique(dist).size >= 3:
                            body = ax.violinplot(
                                [dist], positions=[i], widths=.72,
                                showmeans=False, showextrema=False,
                                showmedians=False)['bodies'][0]
                            body.set_facecolor(NETWORK_COLORS[i])
                            body.set_edgecolor('none')
                            body.set_alpha(1.0)
                        else:
                            ax.plot([i-.26, i+.26], [np.mean(dist)]*2,
                                    lw=3.0, color=NETWORK_COLORS[i], alpha=1.0)
                        observed = float(row.Observed_fraction)
                        ax.plot([i-.28, i+.28], [observed]*2,
                                color=observed_color, lw=4.4,
                                solid_capstyle='round', zorder=4)
                        p_spin = float(row.P_spin_one_sided)
                        if np.isfinite(p_spin) and p_spin < .05:
                            # Keep the nominal spatial-spin result visible without
                            # changing the inferential threshold or FDR results.
                            top = max(observed, float(np.max(dist)))
                            y = min(top+.035, 1.055)
                            ax.text(i, y, f'*\np={p_spin:.3g}', ha='center',
                                    va='bottom', fontsize=21, fontweight='bold',
                                    fontfamily='Arial', color='black', zorder=5)
                else:
                    ax.text(.5, .5, 'No ROIs at nominal paired-test P < 0.05',
                            transform=ax.transAxes, ha='center', va='center',
                            fontsize=15, color='#777777', fontfamily='Arial')
                ax.set_xticks(np.arange(7))
                ax.set_xticklabels(NETWORKS, fontsize=24, fontweight='bold',
                                   fontfamily='Arial')
                ax.set_xlim(-.65, 6.65)
                # Keep the full fraction scale; observed proportions can reach 1.
                ax.set_ylim(0, 1.15)
                ax.set_yticks(np.arange(0, 1.01, .2))
                ax.tick_params(axis='both', labelsize=22, width=0, length=0, pad=7)
                for tick in ax.get_yticklabels():
                    tick.set_fontfamily('Arial')
                    tick.set_fontweight('bold')
                ax.set_ylabel('Fraction of selected ROIs', fontsize=29,
                              fontweight='bold', fontfamily='Arial', labelpad=16)
                display_group = 'TMS' if group == 'TMS' else group
                display_gradient = f'GD{grad[-1]}'
                ax.set_title(f'{display_group} ({display_gradient}) - {direction}', fontsize=29,
                             fontweight='bold', fontfamily='Arial', pad=14)
                ax.spines['left'].set_linewidth(3.0)
                ax.spines['bottom'].set_linewidth(3.0)
                ax.grid(axis='y', color='#E2E2E2', lw=1.0, zorder=0)
                ax.set_axisbelow(True)
                fig.tight_layout(pad=0.7)
                base = directory / (
                    f'Figure2c_{group}_{grad}_{direction}_nominalP_spin_network_enrichment')
                for ext in ('png', 'pdf', 'svg'):
                    path = base.with_suffix('.'+ext)
                    opts = {'dpi': 400, 'bbox_inches': 'tight'} if ext == 'png' else {'bbox_inches': 'tight'}
                    fig.savefig(path, **opts)
                    print('[SAVE]', path)
                plt.close(fig)

def save_roi_list(masks: dict, lookup: pd.DataFrame, directory: Path) -> None:
    rows=[]
    lkp=lookup.set_index('ROI_ID')
    for group in GROUPS:
        for grad in GRADIENTS:
            for direction in DIRECTIONS:
                arr=masks[group,grad,direction]
                for roi in np.flatnonzero(arr)+1:
                    x=lkp.loc[roi]
                    rows.append({'Group':group,'Gradient':grad,'Direction':direction,
                                 'ROI_ID':roi,'Network':x.Network,'ROI_name':x.ROI_name})
    if rows:
        df=pd.DataFrame(rows)
    else:
        df=pd.DataFrame(columns=['Group','Gradient','Direction','ROI_ID','Network','ROI_name'])
    path=directory/'Figure2c_Selected_nominalP_ROI_Names.csv'
    df.to_csv(path,index=False,encoding='utf-8-sig')
    print('[SAVE]',path)


def write_notes(output: Path, args, no_selected: bool) -> None:
    text=(
        'Figure 2c methods and audit\n'
        '--------------------------\n'
        'Figure 2a input: '+str(args.input)+'\n'
        'ROI selection: within-group two-sided paired-test raw P < 0.05, uncorrected across 400 ROIs.\n'
        'Increase: P<0.05 and mean post-pre ΔG > 0; Decrease: P<0.05 and ΔG < 0.\n'
        'Eight standalone figures: Control/TMS x G1/G2 x Increase/Decrease.\n'
        'For each direction, network fraction = N(selected parcels in network)/N(selected parcels).\n'
        'Spatial null: LH/RH paired-random spherical rotation of ROI centroids;\n'
        'nearest 1-to-1 Hungarian matching, no duplicate parcels per rotation.\n'
        f'Permutations: {args.spins}; random seed: {args.seed}.\n'
        'Spin p = (1 + # rotated fractions >= observed)/(1 + #rotations).\n'
        'Spin P values one-sided for over-enrichment only.\n'
        'BH-FDR correction is across ALL 56 planned tests (2 groups x 2 gradients x 2 directions x 7 networks).\n'
        'Figure stars: uncorrected one-sided Spin P < 0.05; nominal exploratory findings.\n'
        'BH-FDR q values are retained in the results table; no corrected significance is implied by figure stars.\n'
        'When an ROI set is empty, no enrichment statistic is tested or inferred.\n'
        'ROI-centroid spin is an approximation to vertex-level rotation; report explicitly.\n'
        'No mediation inference and no direct between-group inference are supported.\n'
        'ROI masks use nominal P<0.05 and are exploratory; they are not ROI-level FDR-significant findings.\n'
        f'Any nominal P<0.05 ROI selected: {not no_selected}.\n'
    )
    (output/'Figure2c_METHODS_and_checks.txt').write_text(text,encoding='utf-8-sig')


def main() -> None:
    p=argparse.ArgumentParser(description='Fig2c from nominal P<0.05 Fig2a ROI masks + ROI-centroid spatial spins')
    p.add_argument('--input',type=Path,default=SOURCE_CSV)
    p.add_argument('--out',type=Path,default=OUTPUT_DIR)
    p.add_argument('--lh-annot',type=Path,default=LH_ANNOT)
    p.add_argument('--rh-annot',type=Path,default=RH_ANNOT)
    p.add_argument('--spins',type=int,default=N_SPINS)
    p.add_argument('--seed',type=int,default=SEED)
    # Only for runtime tests / users with exact matched ROI metadata and centroids.
    p.add_argument('--lookup-csv',type=Path,default=None,
                   help='Optional verified ROI_ID/Network/Hemisphere/ROI_name table')
    p.add_argument('--sphere-csv',type=Path,default=None,
                   help='Optional verified ROI_ID/x/y/z fsLR spherical centroids')
    args=p.parse_args()
    if args.spins < 999:
        raise ValueError('请使用至少 999 次 spin，正式图建议 5000-10000 次')
    args.out.mkdir(parents=True,exist_ok=True)
    source,masks=read_figure2a_csv(args.input)
    if args.lookup_csv:
        lookup=pd.read_csv(args.lookup_csv).sort_values('ROI_ID')
        if not {'ROI_ID','Network','ROI_name','Hemisphere'}.issubset(lookup):
            raise ValueError('ROI lookup CSV must include ROI_ID, Network, ROI_name, Hemisphere')
        if len(lookup)!=400 or not np.array_equal(lookup.ROI_ID.to_numpy(),np.arange(1,401)):
            raise ValueError('ROI lookup does not match Figure 2a 400-ROI ordering')
        if not set(lookup.Network).issubset(set(NETWORKS)):
            raise ValueError('Invalid networks in lookup')
    else:
        lookup=schaefer_networks(args.lh_annot,args.rh_annot)
    lookup.to_csv(args.out/'Figure2c_ROI_Network_Mapping.csv',index=False,encoding='utf-8-sig')
    save_roi_list(masks,lookup,args.out)
    any_roi=any(x.any() for x in masks.values())
    spins=None
    if any_roi:
        if args.sphere_csv:
            pts_df=pd.read_csv(args.sphere_csv).sort_values('ROI_ID')
            if not {'ROI_ID','x','y','z'}.issubset(pts_df):
                raise ValueError('Sphere CSV must have ROI_ID,x,y,z')
            if len(pts_df)!=400 or not np.array_equal(pts_df.ROI_ID.to_numpy(),np.arange(1,401)):
                raise ValueError('Sphere ROI order must be exactly 1..400')
            coords=pts_df[['x','y','z']].to_numpy(float)
            norms=np.linalg.norm(coords,axis=1)
            if not np.isfinite(norms).all() or (norms<1e-10).any():
                raise ValueError('Sphere coordinates missing or zero norm')
            coords/=norms[:,None]
        else:
            coords=centroid_spheres()
        cache=args.out/f'Figure2c_spin_index_{args.spins}_seed{args.seed}.npy'
        spins=generate_spins(coords,args.spins,args.seed,cache)
    else:
        print('[INFO] Fig2a 没有任何原始 P<0.05 ROI，所有方向均不进行 Spin-test。')
    results,nulls=analyze(masks,lookup,spins)
    table=args.out/'Figure2c_Spin_Enrichment_56Tests.csv'
    results.to_csv(table,index=False,encoding='utf-8-sig')
    print('[SAVE]',table)
    save_figure(results,nulls,args.out)
    write_notes(args.out,args,not any_roi)
    print('[DONE] 输出目录：',args.out)


if __name__=='__main__':
    main()
