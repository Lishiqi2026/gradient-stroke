# Gradient and preprocessing notes

These are the supplied primary-cohort custom scripts, not a replacement for GRETNA, SPM, MATLAB, BrainSpace or neuromaps. Those tools are not bundled.

1. Supply preprocessed 4D volumes to juzhen18.py for 10k fsaverage projection and vertex FC.
2. Run allnpy.py with its original affinity and embedding sequence.
3. Arrange raw gradient files into the group/timepoint layout expected by Template.py and duiqi.py using the private study manifest. The first two scripts do not automatically create those four directories.
4. Template.py constructs the primary group reference and refuses to overwrite an existing template. It is separate from the healthy-reference lesion workflow.
5. Run duiqi.py, then relevant matrix/gradient parcellation after configuring inputs.

Recorded preprocessing parameters from manuscript/user materials:

- Primary: TR2s, first5volumes removed, EPI-template normalization to2mm, 6mm smoothing, linear detrending, WM/CSF and Friston24 regression, no global-signal regression, 0.01–0.10Hz bandpass.
- External: TR3s, first5volumes removed, normalization to2mm, 4mm smoothing, linear detrending, WM/CSF and Friston24 regression, no global-signal regression, 0.01–0.08Hz bandpass.
- These notes are not executed GRETNA batch configurations. Initial preprocessing and downstream lesion-workflow FD censoring must not be conflated.
- External gradients/alignment followed the same method logic; primary scripts alone do not establish identical external implementation parameters.

Patient scans, workbooks, masks, individual gradients, hospital IDs and provider atlas caches are not part of the public source archive.

