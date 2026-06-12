#!/usr/bin/env python3
"""
covariate_logistic_test.py  [Phase G fix: R2#7 — statistically correct version]

Question: does the BRIDGE-KO microbiome signal remain associated with response AFTER
adjusting for available clinical covariates?

Method (avoids the in-sample leakage of the earlier stacker):
  1. Cross-fit: 5-fold CV over all 154 target samples (repeated over `--n-seeds` seeds).
     In each fold, train the full BRIDGE-KO DNN (paired original+knockoff, FDR-select,
     retrain) on the training folds and predict the held-out fold -> an OUT-OF-FOLD
     BRIDGE-KO score for every sample. Average across seeds.
  2. Logistic-coefficient test on the 154 samples:
       M0: response ~ covariates                         (reduced)
       M1: response ~ covariates + bridgeko_score        (full)
     Report the bridgeko_score coefficient (log-OR), its Wald p-value, and a likelihood-
     ratio test (chi-square, 1 df) of M1 vs M0. Also fit response ~ bridgeko_score alone.
  Covariates: age, sex, bmi, treatment, abx, ppi, stage (one-hot; "unknown" kept).

Inputs: fulltarget/full_big_equi_knockoffs_pca25.csv, fulltarget/full_meta.csv,
        covariates_154.csv.
Output: revision/results/covariate_logistic/{oof_scores.csv, logistic_test.json, coef_table.csv}
"""
import os, sys, argparse, json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(__file__))
from run_bridgeko import (set_global_seed, build_x3d_from_knockoff,
                          write_original_only_tsv_from_knockoff, DNN, FDR_control)
import scipy.stats as st
import statsmodels.api as sm


def fit_predict_fold(Xtr_df, ytr, Xte_df, run_dir, epochs=20, bs=30):
    cols = list(Xtr_df.columns[: Xtr_df.shape[1] // 2])
    x3d, _, _ = build_x3d_from_knockoff(Xtr_df, keep_cols=cols)
    p = x3d.shape[1]; coeff = 0.05 * np.sqrt(2.0 * np.log(max(p, 1)) / max(x3d.shape[0], 1))
    os.makedirs(run_dir, exist_ok=True)
    mf = DNN(epochs, bs, output_layer_activation="sigmoid").build_DNN(p, 1, coeff)
    mf.compile(loss="binary_crossentropy", optimizer="adam")
    mf.fit(x3d, ytr, epochs=epochs, batch_size=bs, verbose=0,
           callbacks=[DNN.Job_finish_Callback(run_dir, p)])
    tmp = os.path.join(run_dir, "train_orig.csv")
    write_original_only_tsv_from_knockoff(Xtr_df, tmp)
    sel = [f for f, _ in FDR_control().controlFilter(tmp, run_dir, offset=1, q=0.05)]
    if not sel:
        sel = cols  # fall back to all features if FDR selects none in a fold
    x3d_s, _, _ = build_x3d_from_knockoff(Xtr_df, keep_cols=sel)
    ps = x3d_s.shape[1]; cs = 0.05 * np.sqrt(2.0 * np.log(max(ps, 1)) / max(x3d_s.shape[0], 1))
    ms = DNN(epochs, bs, output_layer_activation="sigmoid").build_DNN(ps, 1, cs)
    ms.compile(loss="binary_crossentropy", optimizer="adam")
    ms.fit(x3d_s, ytr, epochs=epochs, batch_size=bs, verbose=0)
    x3d_te, _, _ = build_x3d_from_knockoff(Xte_df, keep_cols=sel)
    return ms.predict(x3d_te, verbose=0).reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined", required=True)   # full_big_equi_knockoffs_pca25.csv
    ap.add_argument("--meta", required=True)        # full_meta.csv (sample, response_binary)
    ap.add_argument("--covariates", required=True)  # covariates_154.csv
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    Xdf = pd.read_csv(args.combined)
    meta = pd.read_csv(args.meta)
    y = meta["response_binary"].to_numpy().astype(int)
    samples = meta["sample"].tolist()

    # cross-fit out-of-fold BRIDGE-KO scores, averaged over seeds
    oof = np.zeros(len(y)); counts = np.zeros(len(y))
    for seed in range(args.n_seeds):
        set_global_seed(1000 + seed)
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=1000 + seed)
        for k, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
            pred = fit_predict_fold(Xdf.iloc[tr], y[tr].astype(float), Xdf.iloc[te],
                                    os.path.join(args.outdir, f"s{seed}_f{k}"))
            oof[te] += pred; counts[te] += 1
            print(f"seed {seed} fold {k}: n_test={len(te)}", flush=True)
    oof /= np.maximum(counts, 1)
    pd.DataFrame({"sample": samples, "response_binary": y, "bridgeko_score": oof}).to_csv(
        os.path.join(args.outdir, "oof_scores.csv"), index=False)

    # ---- build covariate design ----
    cov = pd.read_csv(args.covariates).set_index("sample").loc[samples]
    cov_cols = ["age", "sex", "bmi", "treatment", "abx", "ppi", "stage"]
    C = pd.get_dummies(cov[cov_cols], columns=["sex", "treatment", "abx", "ppi", "stage"],
                       dummy_na=False, drop_first=True).astype(float)
    # standardize numeric + the score
    score = (oof - oof.mean()) / (oof.std() + 1e-9)
    Cz = (C - C.mean()) / (C.std().replace(0, 1))
    Cz = Cz.loc[:, Cz.std() > 0]  # drop degenerate dummy columns

    yv = y.astype(float)
    # M0: covariates only ; M1: covariates + score
    X0 = sm.add_constant(Cz.to_numpy())
    X1 = sm.add_constant(np.column_stack([Cz.to_numpy(), score]))
    m0 = sm.Logit(yv, X0).fit(disp=0, maxiter=200)
    m1 = sm.Logit(yv, X1).fit(disp=0, maxiter=200)
    # score-only model
    Xs = sm.add_constant(score.reshape(-1, 1))
    ms = sm.Logit(yv, Xs).fit(disp=0, maxiter=200)

    # likelihood-ratio test M1 vs M0 (1 df = the score term)
    lr_stat = 2 * (m1.llf - m0.llf)
    lr_p = st.chi2.sf(lr_stat, df=1)
    score_idx = X1.shape[1] - 1  # last column
    beta = float(m1.params[score_idx]); se = float(m1.bse[score_idx])
    wald_p = float(m1.pvalues[score_idx])

    out = {
        "n": int(len(yv)),
        "score_only": {"beta": float(ms.params[1]), "p": float(ms.pvalues[1]),
                       "OR": float(np.exp(ms.params[1]))},
        "adjusted_full_model": {
            "score_beta": beta, "score_SE": se, "score_OR": float(np.exp(beta)),
            "score_wald_p": wald_p},
        "LRT_M1_vs_M0": {"chi2": float(lr_stat), "df": 1, "p": float(lr_p)},
        "interpretation": ("BRIDGE-KO score remains associated with response after covariate "
                           "adjustment" if lr_p < 0.05 else
                           "BRIDGE-KO score not significant after covariate adjustment"),
    }
    json.dump(out, open(os.path.join(args.outdir, "logistic_test.json"), "w"), indent=2)
    # full coefficient table of M1
    names = ["const"] + list(Cz.columns) + ["bridgeko_score"]
    pd.DataFrame({"term": names, "beta": m1.params, "SE": m1.bse,
                  "z": m1.tvalues, "p": m1.pvalues}).to_csv(
        os.path.join(args.outdir, "coef_table.csv"), index=False)
    print("RESULT", json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
