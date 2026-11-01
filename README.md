# Gradient Stroke — manuscript analysis code (preparation snapshot)

Source snapshot for "Selective Reorganization of the Visual–Somatomotor Cortical Gradient Associated with Repetitive Transcranial Magnetic Stimulation after Stroke".
No public repository URL, DOI, software license or tested release has yet been assigned.

## Scope

This repository contains custom analysis and plotting code, not patient imaging or clinical datasets. The three primary vertex-gradient scripts cover only one stage.
See [analysis map](docs/analysis_map.md), [Figure 2–3 and clinical notes](docs/figure23_table4_notes.md), and [Figure 4–6 notes](docs/figures456_notes.md).

- Primary vertex FC, embedding, fixed-template construction, alignment and parcellation.
- Figure 2 longitudinal regional, enrichment and graph-topology analyses.
- Figure 3 clinical and between-group gradient analyses.
- Clinical ANCOVA/HC3/interaction and follow-up calculations, plus Table 4.
- Shared lesion, external-validation and prediction analyses for Figures 4–6.
- Supplementary spatial transcriptomic scripts.

Historical helpers are retained where needed to preserve imports. The final wrapper entrypoints, not every auxiliary script, define the manuscript workflow. Inclusion of a legacy module is not a claim that all of its analyses were reported.

## Preparing private inputs

Hospital identifiers in source constants and examples have been replaced with opaque PRIVATE_SUBJECT_### keys. One individual numeric QC constant is also a private placeholder. Cohort exclusions and group-override logic are retained. Directly running these scripts against the original patient data can select the wrong cohort: prepare a private copy first.

    python prepare_local.py --subject-keys ..\LOCAL_ONLY\subject_keys.json --private-values ..\LOCAL_ONLY\fig456_private_values.json
    $env:GRADIENT_STROKE_CLINICAL_CSV = (Resolve-Path ..\LOCAL_ONLY\clinical_scores.csv).Path

The LOCAL_ONLY files above are author-side inputs, deliberately absent from this repository/archive. Readers need authorized study inputs and cohort-key configuration to reproduce the hospital analyses. Do not publish the private configuration, clinical CSV, resulting local_work directory or participant-level outputs.

prepare_local.py performs no analysis. It restores private constants into an ignored working copy, checks Python syntax and refuses to overwrite a non-empty output directory. Its optional --text-replacements private_paths.json argument accepts exact source-text replacements for input/output paths. Inspect all path assignments in the prepared copy.

## Installation and execution

Use an isolated Python environment. requirements.txt refers to versions recorded by the existing Figure 4–6 package plus dependencies imported by other analyses. Some added dependencies are unpinned; this is not a verified cross-platform environment lock.

Primary-gradient and Figure 2–3 scripts retain author-machine path defaults. Configure paths in the private copy before execution. Do not run analyses against original output locations just to test installation.

After providing the Figure 4–6 input tree described in its notes:

    python local_work/analysis/figures456/run.py --figure 4 --stage full
    python local_work/analysis/figures456/run.py --figure 5 --stage full
    python local_work/analysis/figures456/run.py --figure 6 --stage full

These sequential examples can be computationally expensive. The plots stage requires cached private results and does not recompute statistics. The three original figure packages shared identical script trees, so the code is included once.

## Inputs and reproduction limits

- Primary patient data follow the manuscript Data availability statement.
- External cohort: OpenNeuro ds003999, version 1.0.2, <https://doi.org/10.18112/openneuro.ds003999.v1.0.2>.
- BrainSpace healthy reference, Schaefer/Destrieux atlases, XTRACT resources and AHBA inputs must be obtained through their original providers under their terms.
- Executed GRETNA batches, complete original registration-estimation workflows, all supporting atlases and complete input manifests are not bundled. Existing Figure 4–6 entrypoints start from prebuilt inputs; transform replay does not reproduce registration estimation.
- The primary group alignment template and the healthy reference for the separate parcel-level lesion pipeline are different resources.
- Primary clinical/gradient/lesion/functional/projection samples are 45/42/41/39/37 for their respective analyses. The external cohort is 29. Panel-specific subgroups differ.

Preparation changes clinical input handling and private-constant storage, not the intended statistical models. Syntax, redaction and preparation-utility checks do not establish end-to-end numerical reproduction; this package has not rerun the scientific pipeline.

## Before public release

1. Test the prepared copy with authorized inputs against manuscript outputs.
2. Supply missing preprocessing/input-generation documentation where available.
3. Select a license for code the authors can license; retain third-party notices.
4. Create the actual GitHub repository and replace pending metadata with its verified URL.
5. Archive the manuscript-matched release, for example with Zenodo, and cite its real DOI.

No fabricated repository URL or DOI is supplied.

