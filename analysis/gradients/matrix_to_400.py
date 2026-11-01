from pathlib import Path
import nibabel as nib
import numpy as np
from nilearn.datasets import fetch_atlas_surf_destrieux
from brainspace.utils.parcellation import reduce_by_labels

BASE = Path(r'F:\lishiqi\预处理数据\cortex')
INPUT = BASE / 'Vertex（20484，20484）'
OUTPUT = BASE / 'parcel(400，400)'

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

def parcellate(matrix, labels, mask):
    valid = mask & (labels > 0)
    indices = np.flatnonzero(valid)
    valid_labels = labels[indices]
    result = np.empty((400, 400), dtype=np.float64)
    # 每次仅读取一个脑区的行，避免复制完整的 20484 × 20484 矩阵。
    for parcel in range(1, 401):
        rows = np.flatnonzero(valid & (labels == parcel))
        block = np.asarray(matrix[rows, :])[:, indices]
        if not np.isfinite(block).all():
            raise ValueError(f'脑区 {parcel} 的连接值包含 NaN 或无穷值')
        column_means = reduce_by_labels(block, valid_labels, target_labels=np.arange(1, 401), red_op='mean', axis=0)
        result[parcel - 1] = column_means.mean(axis=0)
    return result

def main():
    labels, mask = labels_and_mask()
    paths = sorted(INPUT.rglob('*.npy'))
    if not paths:
        raise FileNotFoundError(f'没有找到输入文件：{INPUT}')
    for path in paths:
        target = OUTPUT / path.relative_to(INPUT)
        if target.exists():
            print(f'跳过已存在文件：{target}')
            continue
        try:
            matrix = np.load(path, mmap_mode='r', allow_pickle=False)
            if matrix.shape != (20484, 20484):
                raise ValueError(f'维度不正确：{matrix.shape}')
            result = parcellate(matrix, labels, mask)
            if not np.isfinite(result).all():
                raise ValueError('输出含非有限值')
            target.parent.mkdir(parents=True, exist_ok=True)
            np.save(target, result)
            print(f'已保存：{target}')
        except Exception as exc:
            print(f'失败：{path}：{exc}')

if __name__ == '__main__':
    main()
