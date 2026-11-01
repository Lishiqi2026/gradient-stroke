import os
from pathlib import Path
import numpy as np
from brainspace.gradient.kernels import compute_affinity
from sklearn.metrics import pairwise_distances
from nilearn.datasets import fetch_atlas_surf_destrieux
#from brainspace.gradient import utils as embed
from mapalign import embed
from brainspace.gradient import GradientMaps

in_dir = Path(r"F:\lishiqi\预处理数据\cortex\Vertex（20484，20484）")
out_dir = Path(r"F:\lishiqi\预处理数据\cortex\Vertex (18715, 300)")
out_dir.mkdir(parents=True, exist_ok=True)

destrieux = fetch_atlas_surf_destrieux()
excluded_labels = [destrieux.labels.index("Unknown"),
                   destrieux.labels.index("Medial_wall")]

is_excluded_l = np.isin(destrieux.map_left, excluded_labels)
is_excluded_r = np.isin(destrieux.map_right, excluded_labels)
rows_to_zero = np.concatenate([is_excluded_l.ravel(), is_excluded_r.ravel()]).astype(bool)
cols_to_zero = rows_to_zero

n_vertices = rows_to_zero.size
remaining_idx = np.where(~rows_to_zero)[0]

files = sorted(in_dir.glob("*.npy"))
print(f"{len(files)} {out_dir}")

for f in files:
    out_path = out_dir / f.name

    if out_path.exists():
        print(f"Skip existing: {f.name}")
        continue
    else:
        print(f"{f.name} ")

    try:
        FC = np.load(f)
        if FC.shape[0] != FC.shape[1]:
            print(f" {f.name}: shape={FC.shape}")
            continue

        FCmapz = np.arctanh(FC)

        Thre = np.percentile(FCmapz, 90, axis=0)
        for vv in range(0, FCmapz.shape[1]):
            FCmapz[vv, :] = np.where(FCmapz[vv, :] > Thre[vv], FCmapz[vv, :], 0)

        FCmapz[FCmapz < 0] = 0
        np.fill_diagonal(FCmapz, 0)

        print("1 FCmapz shape:", FCmapz.shape)
        FCmapz = np.delete(FCmapz, rows_to_zero, axis=0)
        print("2 FCmapz shape:", FCmapz.shape)
        FCmapz = np.delete(FCmapz, cols_to_zero, axis=1)
        print("3 FCmapz shape:", FCmapz.shape)
        FCmapz = np.nan_to_num(FCmapz, nan=0, posinf=0, neginf=0)
        print("4 FCmapz shape:", FCmapz.shape)

        # 亲和矩阵仅在内存中用于计算梯度，不写入磁盘
        A = compute_affinity(
            FCmapz,
            kernel='cosine',
            sparsity=0.9,
            pre_sparsify=True,
            non_negative=True
        )

        gm = GradientMaps(n_components=300, approach='dm', kernel='cosine', random_state=0)
        gm.fit(A)
        emb_ = gm.gradients_
        emb = emb_

        np.save(out_path, emb)
        print(f"Saved gradient: {f.name}  {out_path}")

    except Exception as e:
        print(f" {f.name}: {e}")

print("")
