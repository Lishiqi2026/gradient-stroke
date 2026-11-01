# Figures 4–6: entry points and input requirements

This directory consolidates the byte-identical source trees from the three final figure packages. It contains 99 Python scripts, one optional Shell script, and the top-level Python runner. It is a code-only snapshot; no participant data, cached predictions, images, atlas cache, replay results, original private README, or private cohort configuration is included.

## Public source versus private execution

Participant-specific cohort-policy identifiers are replaced with private placeholders during repository preparation. Do not run public placeholders against the original identifiable data. Use the repository's private restoration process to create an ignored runtime copy and restore the cohort policy with authorized inputs. The original inclusion/exclusion conditions and group corrections must be retained. The old sensitivity helper also contains a private expected lesion-volume validation value; its placeholder must be restored before that helper is used. Supply the non-public `LOCAL_ONLY/fig456_private_values.json` to the restoration utility through `--private-values`, in addition to its participant-identifier mapping. That private JSON must remain outside the public repository and archive.

`package_config.json` now contains only a default figure number (4). The `--figure` argument selects 4, 5 or 6 and propagates to assembly and reporting through `PAPER_FIGURE`. Use separate runtime copies for final output snapshots where practical, since the original scripts reuse work and summary paths.

## Commands (in an authorized, restored runtime copy)

```bash
python -m pip install -r requirements.txt
python run.py --figure 4 --stage full
python run.py --figure 5 --stage full
python run.py --figure 6 --stage full
```

`--stage core` (or `prepare`) derives core features only and does not refresh all inferential statistics. `--stage plots` needs the authorized saved results/replay inputs, which are not distributed here. It is not a demonstration that can run from an empty repository. Full runs include the original repeated nested cross-validation and may take substantial time. `PDF_PYTHON` optionally selects a PDF-assembly Python executable. VTK/OpenGL is needed for cortical rendering.

## Manuscript analysis routes

| Scope | Scripts called by the runner |
|---|---|
| Shared connectivity and lesion features | `reproduce_fc.py`, `compute_lesion_inputs.py` |
| Figure 4 statistics | `analyze_updated_fig4.py` |
| Figure 4 rendering | `render_fig4.py` and `render_legacy_network_radar.py` |
| Figure 5 statistics | `analyze_updated_fig4.py`, `recompute_external_base.py`, `recompute_external_clinical.py`, `recompute_cross_cohort.py` |
| Figure 5 rendering | `render_external_base.py`, `render_fig5_cross.py`, `render_network_counts.py` |
| Figure 6 feature preparation | `derive_core_features.py`, `recompute_external_base.py`, `recompute_external_clinical.py` |
| Figure 6 prediction | `recompute_main_prediction.py`, `recompute_external_prediction.py` |
| Figure 6 rendering | `render_fig6.py` |
| Assembly and result export | `assemble_delivery.py`, `write_package_results.py` |

All names in the table are under `scripts/`. They import modules in `scripts/project/lesion_analysis/` and `scripts/project/ml_analysis/`; preserve the relative directory structure. The runner overlays these modules into its private `work/Gradient/` runtime directory and sets `PAPER_PROJECT_ROOT` accordingly. Some historical auxiliary modules contain old exploratory narrative text and sensitivity outputs. Their standalone `main()` functions are not all manuscript analyses. The final runner and wrappers above define the intended routes.

## Required input organization

Inputs are placed under `input_data/work/` in the authorized runtime copy. This inventory records file headers and code expectations, not participant records. It is not a complete substitute for the original data dictionary.

- `subjects.csv`: `subject_id, group, pre_stem, post_stem`.
- `clinical_corrected.csv`: `subject_id, sex, age, clinical_side_text, location_text, group, NIHSS_pre, NIHSS_post, ADL_pre, ADL_post, MMSE_pre, MMSE_post, esrs, HAMD_pre, HAMD_post, VFT_pre, VFT_post, sex_male, group_tms`.
- Optional `registration_review.csv`: `subject_id, group, status, note`.
- Paired parcel time series: `Gradient/lesion_analysis/lesion_gradient_distance/fc_consistent/timeseries_fsl/<subject_id>_<baseline|post>_schaefer400.tsv`; 400 columns.
- Matched Power FD vectors: the same analysis branch under `fc_consistent/fd/<subject_id>_<baseline|post>_PowerFD.txt`; length must equal the time-series row count.
- Registered lesion masks: `Gradient/lesion_analysis/mni_registration/masks/<Control|TMS>/<before|after>/<subject_id>_lesion_mni2mm.nii.gz`; the mask grid and affine must match the atlas.
- Native lesion masks: `native_lesions/<Control|TMS>/<before|after>/`, with filenames identified by the subject manifest.
- Atlas, Schaefer labels, anatomical compartment and XTRACT endpoint resources are required under the runtime resource paths used by the scripts. Their files and redistribution permissions are not supplied by this code-only package.
- Main-cohort baseline gradients and the fixed healthy-reference matrices are required in the locations used by the corresponding feature modules.
- External metadata: `Gradient/new_dataset/participants.tsv`; header fields are `participant_id, sex, time_after_stroke, pre_score, post_score, hemisphere, type, size, Broca, Wernike, Basal, Thalamus` (tab-delimited).
- External processed connectivity: `Gradient/new_dataset/(400,400)/(400,400)/`; processed gradient files: `Gradient/new_dataset/(400,5)/(400,5)/`, including the ROI-order metadata in its `QC/` directory.
- External healthy references and input layouts are specified in `scripts/project/lesion_analysis/external_validation/ds003999_validation/` and `ds003999_confirmatory_extension/`.

External public data originate from OpenNeuro ds003999, version 1.0.2 (https://doi.org/10.18112/openneuro.ds003999.v1.0.2). The processed derivatives expected by these scripts are additional inputs; downloading raw OpenNeuro data alone does not recreate them.

## Scope and environment limits

The runner begins with processed ROI time series, FD traces, registered lesions, baseline gradients and external processed FC/gradients. Raw-scanner GRETNA preprocessing, estimation of native-to-MNI registration transforms, and the original preparation of the corrected clinical/subject manifests are not reproduced by this entry point. `scripts/resample_native_lesions.sh` only reapplies supplied transforms; it requires FSL and `FSLDIR` and does not estimate registration.

The original package specifies Python 3.10+ and pins major scientific packages. `openpyxl` and `PyPDF2` are additionally imported by included auxiliary modules; they were added to requirements without invented version pins. Installation and end-to-end execution have not been tested for this public copy. Static Python syntax checking is the validation performed during preparation.

## Publication-copy changes

- Added `--figure` and propagated figure selection to assembly and result writing; no statistical algorithms changed.
- Removed a cached print of one individual's lesion volumes; the excluded participant remains defined by the restored cohort policy.
- Replaced a hardcoded individual-volume validation constant with a private restoration placeholder, preserving its validation condition.
- Changed that legacy helper's cached cohort-size print to derive counts from its selected input rows.
- Added the two previously undeclared auxiliary imports to requirements.
- Private identifier redaction and runtime restoration are applied by the repository preparation process; original scripts remain untouched.
