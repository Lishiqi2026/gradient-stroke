import os
import numpy as np
from neuromaps.transforms import mni152_to_fsaverage

input_root = r"H:\newtreat\GretnaFunNIfTI"
output_dir = r"F:\lishiqi\预处理数据\cortex\Vertex（20484，20484）"
input_filename = "bcNGSdswranrest.nii"

os.makedirs(output_dir, exist_ok=True)

subject_dirs = sorted(
    (path for path in os.scandir(input_root) if path.is_dir()),
    key=lambda path: path.name
)

for subject in subject_dirs:
    try:
        input_path = os.path.join(subject.path, input_filename)
        output_path = os.path.join(output_dir, f"{subject.name}.npy")

        if not os.path.isfile(input_path):
            print(f"Skip, input file not found: {input_path}")
            continue

        if os.path.exists(output_path):
            print(f"Skip existing: {output_path}")
            continue

        print(f"Processing: {input_path}")

        surf_data_l, surf_data_r = mni152_to_fsaverage(
            img=input_path,
            fsavg_density="10k",
            method="linear",
        )

        lh_data = np.array([item.data for item in surf_data_l.darrays]).T
        rh_data = np.array([item.data for item in surf_data_r.darrays]).T

        brain = np.vstack((lh_data, rh_data))
        brain = np.nan_to_num(brain)
        fc_matrix = np.corrcoef(brain)

        np.save(output_path, fc_matrix)
        print(f"Saved: {output_path}")

    except Exception as exc:
        print(f"Failed for {subject.name}: {exc}")

print("Finished")
