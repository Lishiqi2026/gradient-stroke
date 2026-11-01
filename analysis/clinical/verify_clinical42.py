"""Audit only: replicate clinical models for the current paired-imaging manifest.

Reads a private source workbook and the 42-person analysis manifest.
The public adaptation omits a private plotting-code literal-data cross-check;
all clinical models and statistical calculations are retained.
Never executes the plotting script and never modifies source inputs.
Outputs conventional OLS and HC3 inference using residual-df Student t.
"""
from pathlib import Path
import os
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import scipy
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

sys.stdout.reconfigure(encoding="utf-8")
OUT = Path(os.environ.get("GRADIENT_STROKE_CLINICAL42_OUTPUT", "outputs/clinical42"))
OUT.mkdir(parents=True, exist_ok=True)
WORKBOOK = Path(os.environ["GRADIENT_STROKE_CLINICAL_XLSX"])
MANIFEST = Path(os.environ["GRADIENT_STROKE_COHORT_MANIFEST"])
SCALES = ["NIHSS", "ADL", "HAMD", "MMSE", "VFT"]


def coefficient(model, index, name):
    robust = model.get_robustcov_results(cov_type="HC3", use_t=True)
    conventional_ci = np.asarray(model.conf_int())[index]
    robust_ci = np.asarray(robust.conf_int())[index]
    return {
        f"{name}_estimate": float(model.params[index]),
        f"{name}_ols_se": float(model.bse[index]),
        f"{name}_ols_ci_low": float(conventional_ci[0]),
        f"{name}_ols_ci_high": float(conventional_ci[1]),
        f"{name}_ols_p": float(model.pvalues[index]),
        f"{name}_hc3_se": float(robust.bse[index]),
        f"{name}_hc3_ci_low": float(robust_ci[0]),
        f"{name}_hc3_ci_high": float(robust_ci[1]),
        f"{name}_hc3_p": float(robust.pvalues[index]),
    }


def independent_check(y, x, model):
    """Check library estimates/HC3 using direct matrix calculation."""
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    resid = y - x @ beta
    inv = np.linalg.inv(x.T @ x)
    h = np.einsum("ij,jk,ik->i", x, inv, x)
    hc3_cov = inv @ (x.T @ ((resid**2 / (1 - h)**2)[:, None] * x)) @ inv
    robust = model.get_robustcov_results(cov_type="HC3", use_t=True)
    np.testing.assert_allclose(model.params, beta, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(robust.cov_params(), hc3_cov, rtol=1e-8, atol=1e-11)
    independent_ci = beta[:, None] + np.array([-1, 1]) * stats.t.ppf(.975, model.df_resid) * np.sqrt(np.diag(hc3_cov))[:, None]
    np.testing.assert_allclose(robust.conf_int(), independent_ci, rtol=1e-8, atol=1e-10)


def main():
    workbook = pd.read_excel(WORKBOOK, dtype={"病历号": str})
    workbook["ID"] = workbook["病历号"].astype(str).str.strip()
    assert workbook.ID.is_unique and len(workbook) == 45
    raw = workbook.set_index("ID")
    columns = ["Group"] + [f"{scale}_{time}" for scale in SCALES for time in ("pre", "post")]
    manifest = pd.read_csv(MANIFEST, dtype={"Subject_ID": str})
    assert manifest.Subject_ID.is_unique and len(manifest) == 42
    data = raw.loc[manifest.Subject_ID].copy()
    np.testing.assert_array_equal(data.Group.to_numpy(), manifest.Group.to_numpy())
    assert data.Group.value_counts().to_dict() == {"TMS": 23, "Control": 19}
    assert not data[columns].isna().any().any()
    g = (data.Group == "TMS").to_numpy(float)
    ancova_rows, interaction_rows, descriptives = [], [], []
    for scale in SCALES:
        baseline = data[f"{scale}_pre"].to_numpy(float)
        y = data[f"{scale}_post"].to_numpy(float)
        x = np.column_stack((np.ones(42), g, baseline))
        model = sm.OLS(y, x).fit()
        independent_check(y, x, model)
        row = {"outcome": scale, "n": 42, "n_Control": 19, "n_TMS": 23,
               "residual_df": int(model.df_resid), "model": "post ~ group + baseline",
               "intercept": float(model.params[0]), "baseline_slope": float(model.params[2]),
               "pooled_mean_baseline": float(baseline.mean())}
        row.update(coefficient(model, 1, "group"))
        ancova_rows.append(row)
        centered = baseline - baseline.mean()
        xi = np.column_stack((np.ones(42), g, centered, g * centered))
        interaction = sm.OLS(y, xi).fit()
        independent_check(y, xi, interaction)
        rowi = {"outcome": scale, "n": 42, "n_Control": 19, "n_TMS": 23,
                "residual_df": int(interaction.df_resid),
                "model": "post ~ group + centered_baseline + group:centered_baseline",
                "pooled_mean_baseline": float(baseline.mean()),
                "Control_baseline_slope": float(interaction.params[2]),
                "TMS_baseline_slope": float(interaction.params[2] + interaction.params[3]),
                "overlap_baseline_low": float(max(baseline[g == 0].min(), baseline[g == 1].min())),
                "overlap_baseline_high": float(min(baseline[g == 0].max(), baseline[g == 1].max()))}
        rowi.update(coefficient(interaction, 1, "group_at_mean_baseline"))
        rowi.update(coefficient(interaction, 3, "interaction"))
        interaction_rows.append(rowi)
        for group, gv in (("Control", 0), ("TMS", 1)):
            pre, post = baseline[g == gv], y[g == gv]
            improvement = (pre - post) if scale in ("NIHSS", "HAMD") else (post - pre)
            descriptives.append({"outcome": scale, "group": group, "n": len(pre),
                                 "pre_mean": float(pre.mean()), "pre_sd": float(pre.std(ddof=1)),
                                 "post_mean": float(post.mean()), "post_sd": float(post.std(ddof=1)),
                                 "improvement_mean": float(improvement.mean()),
                                 "improvement_sd": float(improvement.std(ddof=1))})
    for rows, effects in ((ancova_rows, ["group"]), (interaction_rows, ["group_at_mean_baseline", "interaction"])):
        for effect in effects:
            for method in ("ols", "hc3"):
                q = multipletests([r[f"{effect}_{method}_p"] for r in rows], method="fdr_bh")[1]
                for row, value in zip(rows, q):
                    row[f"{effect}_{method}_q_BH5"] = float(value)
    pd.DataFrame(ancova_rows).to_csv(OUT / "clinical42_ancova_verified.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(interaction_rows).to_csv(OUT / "clinical42_interaction_verified.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(descriptives).to_csv(OUT / "clinical42_descriptives_verified.csv", index=False, encoding="utf-8-sig")
    provenance = {
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (WORKBOOK, MANIFEST)},
        "n": 42, "n_Control": 19, "n_TMS": 23,
        "excluded_from_clinical45": sorted(set(raw.index) - set(data.index)),
        "manifest_vs_workbook_groups": "All 42 match exactly",
        "independent_matrix_check": "All coefficients, HC3 covariances and t-based CIs agree",
        "reference_distribution": "Student t, residual df=39 for ANCOVA and 38 for interaction",
        "BH_families": "Separate five-outcome families for each model, effect and inference method",
        "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
    }
    (OUT / "clinical42_verification_provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ancova": ancova_rows, "HAMD_interaction": interaction_rows[2], "provenance": provenance}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
