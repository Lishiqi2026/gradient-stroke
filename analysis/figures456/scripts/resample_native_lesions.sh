#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
: "${FSLDIR:?Set FSLDIR to an installed FSL directory before running this optional native-mask replay}"
export FSLOUTPUTTYPE=NIFTI_GZ
export PATH="$FSLDIR/share/fsl/bin:$FSLDIR/bin:$PATH"
INPUT="$ROOT/input_data/work"
OUTPUT="$ROOT/reapplied_mni"
REFERENCE="$INPUT/Gradient/lesion_analysis/resources/MNI152_T1_2mm.nii.gz"
for command in applywarp fslmaths; do command -v "$command" >/dev/null; done
tail -n +2 "$INPUT/subjects.csv" | while IFS=, read -r sid group before after; do
    sid="${sid//$'\r'/}";group="${group//$'\r'/}";before="${before//$'\r'/}";after="${after//$'\r'/}"
    for phase in before after; do
        stem="$before";[[ "$phase" == "after" ]] && stem="$after"
        mkdir -p "$OUTPUT/$group/$phase"
        output="$OUTPUT/$group/$phase/${sid}_lesion_mni2mm.nii.gz"
        applywarp --in="$INPUT/native_lesions/$group/$phase/$stem.nii.gz" --ref="$REFERENCE" --warp="$INPUT/registration_transforms/$sid/t1_to_mni_warp.nii.gz" --premat="$INPUT/registration_transforms/$sid/t2_${phase}_to_t1.mat" --out="$output" --interp=nn
        fslmaths "$output" -thr 0.5 -bin "$output"
        echo "REAPPLIED $sid $phase"
    done
done
