import os
import numpy as np
from scipy.spatial import procrustes

input_root = r"F:\lishiqi\预处理数据\cortex\Vertex (18715, 300)"
output_root = r"F:\lishiqi\预处理数据\cortex\duiqi（18715，5）"
template_path = r"F:\lishiqi\预处理数据\cortex\Template.npy"

template = np.load(template_path)
print(f"[INFO] Template shape: {template.shape}")

for group in ("Control", "TMS"):
    for timepoint in ("after", "before"):
        input_dir = os.path.join(input_root, group, timepoint)
        output_dir = os.path.join(output_root, group, timepoint)
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n[DIR] Processing: {input_dir}")
        print(f"[DIR] Output to : {output_dir}")

        for fname in sorted(os.listdir(input_dir)):
            if not fname.endswith(".npy"):
                continue

            in_path = os.path.join(input_dir, fname)
            out_path = os.path.join(output_dir, fname)
            X = np.load(in_path)

            if X.ndim != 2 or X.shape[0] != template.shape[0]:
                print(f"[SKIP] {fname}: shape {X.shape} vs template {template.shape}")
                continue
            if X.shape[1] < 5:
                print(f"[SKIP] {fname}: only {X.shape[1]} components (<5)")
                continue

            X5 = X[:, :5].astype(np.float64, copy=False)
            T5 = template[:, :5].astype(np.float64, copy=False)
            _, X5_aligned, _ = procrustes(T5, X5)

            np.save(out_path, X5_aligned.astype(np.float32))
            print(f"[OK] saved {out_path}, shape={X5_aligned.shape}")

print("\n[DONE] All 4 directories processed.")
