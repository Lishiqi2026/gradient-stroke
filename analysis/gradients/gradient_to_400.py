from pathlib import Path
import nibabel as nib
import numpy as np
from nilearn.datasets import fetch_atlas_surf_destrieux
from brainspace.utils.parcellation import reduce_by_labels

BASE = Path(r'F:\lishiqi\预处理数据\cortex')
INPUT = BASE / 'duiqi（18715，5）'
OUTPUT = BASE / 'duiqi (400, 5)'

def labels_and_mask():
    atlas = fetch_atlas_surf_destrieux()
    excluded = [atlas.labels.index('Unknown'), atlas.labels.index('Medial_wall')]
    mask = np.concatenate([~np.isin(atlas.map_left, excluded),
                           ~np.isin(atlas.map_right, excluded)])
    lh = np.asarray(nib.load(str(BASE / 'lh.Schaefer2018_400Parcels_7Networks_order.label.gii')).darrays[0].data)
    rh = np.asarray(nib.load(str(BASE / 'rh.Schaefer2018_400Parcels_7Networks_order.label.gii')).darrays[0].data)
    if lh.shape != (10242,) or rh.shape != (10242,) or mask.shape != (20484,):
        raise ValueError('图谱或掩膜顶点数不正确')
    labels = np.concatenate([lh, np.where(rh > 0, rh + 200, 0)]).astype(int)
    if not np.array_equal(np.unique(labels[mask & (labels > 0)]), np.arange(1, 401)):
        raise ValueError('有效顶点中未找到完整的 400 个脑区标签')
    return labels, mask

def main():
    labels, mask = labels_and_mask()
    compact_labels = labels[mask]
    valid = compact_labels > 0
    if len(compact_labels) != 18715:
        raise ValueError('掩膜保留的顶点数不是 18715')
    paths = sorted(INPUT.rglob('*.npy'))
    if not paths:
        raise FileNotFoundError(f'没有找到输入文件：{INPUT}')
    for path in paths:
        target = OUTPUT / path.relative_to(INPUT)
        if target.exists():
            print(f'跳过已存在文件：{target}')
            continue
        try:
            x = np.load(path, allow_pickle=False)
            if x.shape != (18715, 5) or not np.isfinite(x[valid]).all():
                raise ValueError('维度不为 (18715, 5) 或有效顶点包含非有限值')
            result = reduce_by_labels(x[valid], compact_labels[valid], target_labels=np.arange(1, 401), red_op='mean', axis=1)
            if result.shape != (400, 5) or not np.isfinite(result).all():
                raise ValueError('输出无效')
            target.parent.mkdir(parents=True, exist_ok=True)
            np.save(target, result)
            print(f'已保存：{target}')
        except Exception as exc:
            print(f'失败：{path}：{exc}')

if __name__ == '__main__':
    main()
