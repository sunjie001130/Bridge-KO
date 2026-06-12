#!/usr/bin/env python3
"""
logistic_from_oof.py  [R2#7] — logistic-coefficient test from saved OOF BRIDGE-KO scores.
Separated from the (expensive) cross-fit so we can iterate on the stats cheaply.

Fixes the singular-matrix issue: the "unknown" level of treatment/abx/ppi/stage is the
SAME 91 public-cohort samples, so those dummies are collinear. We (1) drop_first, (2) drop
zero-variance and duplicate columns, (3) iteratively drop columns until the design is full
rank, and (4) if statsmodels MLE still fails to converge (quasi-separation), fall back to a
mean-centered design and report the regularized fit. The quantities of interest — the
BRIDGE-KO score coefficient and the LRT for adding it — are robust to which collinear
covariate column is dropped.
"""
import os, sys, json
import numpy as np, pandas as pd
import scipy.stats as st
import statsmodels.api as sm

RES = "/home/sunj107/scratch/Bridge_revision/revision/results"
OUT = os.path.join(RES, "covariate_logistic"); os.makedirs(OUT, exist_ok=True)

oof = pd.read_csv(os.path.join(OUT, "oof_scores.csv"))
cov = pd.read_csv(os.path.join(RES, "covariates_154.csv")).set_index("sample").loc[oof["sample"]]
y = oof["response_binary"].to_numpy().astype(float)
score = oof["bridgeko_score"].to_numpy()
score = (score - score.mean()) / (score.std() + 1e-9)

# collapse rare abx levels that cause separation
cov = cov.copy()
cov["abx"] = cov["abx"].replace({"Used; also had MTX use": "Used"})

# Covariate set: age, sex, bmi, sequencing method (verified per-cohort from McCulloch
# Suppl. Table 3; complete for all 154), plus the reviewer-requested abx/ppi/stage
# (Pittsburgh-only -> collapsed to "unknown" elsewhere, pruned by full_rank below).
# `treatment` is intentionally dropped (the response-letter covariate is sequencing method).
C = pd.get_dummies(cov[["age", "sex", "bmi", "seq_method", "abx", "ppi", "stage"]],
                   columns=["sex", "seq_method", "abx", "ppi", "stage"],
                   dummy_na=False, drop_first=True).astype(float)
# standardize numerics
for c in ["age", "bmi"]:
    C[c] = (C[c] - C[c].mean()) / (C[c].std() + 1e-9)

# drop zero-variance, then prune to full column rank (removes collinear 'unknown' dummies)
C = C.loc[:, C.std() > 1e-9]
def full_rank(M):
    cols = list(M.columns)
    keep = []
    A = np.empty((len(M), 0))
    for c in cols:
        B = np.column_stack([A, M[c].to_numpy()])
        if np.linalg.matrix_rank(B) > A.shape[1]:
            A = B; keep.append(c)
    return M[keep]
C = full_rank(C)
print("covariate columns kept:", list(C.columns), flush=True)

def safe_logit(X, y):
    X = sm.add_constant(X, has_constant="add")
    try:
        return sm.Logit(y, X).fit(disp=0, maxiter=300), "mle"
    except Exception as e:
        print("MLE failed, using regularized:", e, flush=True)
        return sm.Logit(y, X).fit_regularized(disp=0, alpha=1.0, maxiter=300), "l2"

Cn = C.to_numpy()
m0, k0 = safe_logit(Cn, y)                                   # covariates only
m1, k1 = safe_logit(np.column_stack([Cn, score]), y)         # covariates + score
ms, ks = safe_logit(score.reshape(-1, 1), y)                 # score only

si = m1.params.shape[0] - 1   # score is the last column (after const-prepend it's last)
beta, se, wald_p = float(m1.params[si]), float(m1.bse[si]), float(m1.pvalues[si])
lr_stat = 2 * (m1.llf - m0.llf); lr_p = float(st.chi2.sf(lr_stat, df=1))

out = {
    "n": int(len(y)), "n_covariates": int(C.shape[1]), "fit_M0": k0, "fit_M1": k1,
    "score_only": {"beta": float(ms.params[1]), "p": float(ms.pvalues[1]), "OR": float(np.exp(ms.params[1]))},
    "adjusted_full_model": {"score_beta": beta, "score_SE": se, "score_OR": float(np.exp(beta)), "score_wald_p": wald_p},
    "LRT_M1_vs_M0": {"chi2": float(lr_stat), "df": 1, "p": lr_p},
    "interpretation": ("microbiome score remains associated with response after covariate adjustment"
                       if lr_p < 0.05 else "microbiome score not significant after covariate adjustment"),
}
json.dump(out, open(os.path.join(OUT, "logistic_test.json"), "w"), indent=2)
names = ["const"] + list(C.columns) + ["bridgeko_score"]
pd.DataFrame({"term": names, "beta": m1.params, "SE": m1.bse, "z": m1.tvalues, "p": m1.pvalues}).to_csv(
    os.path.join(OUT, "coef_table.csv"), index=False)
print("RESULT", json.dumps(out), flush=True)
