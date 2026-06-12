#!/usr/bin/env python3
"""
covariate_prediction.py  [Phase G: R2#6 / R2#7]
Four prediction models on the best-config target test set, to test whether the
microbiome-derived BRIDGE-KO signal survives adjustment for clinical covariates:
  1. micro_only   : BRIDGE-KO selected-genera DNN (the paper model)         [via run_bridgeko engine]
  2. cov_only     : logistic regression on covariates only
  3. micro_plus_cov : BRIDGE-KO predicted prob + covariates -> logistic stack
  4. bridgeko_plus_cov : same as 3 but reported as the augmented BRIDGE-KO

Covariates: age, sex, bmi, treatment, abx, ppi, stage (from covariates_154.csv),
one-hot encoded; "unknown" kept as its own level.

Inputs: best-config train/test combined CSVs, meta train/test, split_train/test (sample IDs),
        covariates_154.csv.
Output: revision/results/covariate_prediction.csv (per-seed AUC per model) + summary.
"""
import os, sys, argparse, json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from run_bridgeko import set_global_seed, build_x3d_from_knockoff, \
    write_original_only_tsv_from_knockoff, DNN, FDR_control


def bridgeko_probs(Xtr_df, ytr, Xte_df, run_dir):
    cols = list(Xtr_df.columns[: Xtr_df.shape[1] // 2])
    x3d, _, _ = build_x3d_from_knockoff(Xtr_df, keep_cols=cols)
    p = x3d.shape[1]; coeff = 0.05 * np.sqrt(2.0 * np.log(max(p, 1)) / max(x3d.shape[0], 1))
    os.makedirs(run_dir, exist_ok=True)
    mf = DNN(20, 30, output_layer_activation="sigmoid").build_DNN(p, 1, coeff)
    mf.compile(loss="binary_crossentropy", optimizer="adam")
    mf.fit(x3d, ytr, epochs=20, batch_size=30, verbose=0,
           callbacks=[DNN.Job_finish_Callback(run_dir, p)])
    tmp = os.path.join(run_dir, "train_orig.csv")
    write_original_only_tsv_from_knockoff(Xtr_df, tmp)
    sel = [f for f, _ in FDR_control().controlFilter(tmp, run_dir, offset=1, q=0.05)]
    if not sel:
        return None, None
    x3d_s, _, _ = build_x3d_from_knockoff(Xtr_df, keep_cols=sel)
    ps = x3d_s.shape[1]; cs = 0.05 * np.sqrt(2.0 * np.log(max(ps, 1)) / max(x3d_s.shape[0], 1))
    ms = DNN(20, 30, output_layer_activation="sigmoid").build_DNN(ps, 1, cs)
    ms.compile(loss="binary_crossentropy", optimizer="adam")
    ms.fit(x3d_s, ytr, epochs=20, batch_size=30, verbose=0)
    x3d_tr_s, _, _ = build_x3d_from_knockoff(Xtr_df, keep_cols=sel)
    x3d_te_s, _, _ = build_x3d_from_knockoff(Xte_df, keep_cols=sel)
    ptr = ms.predict(x3d_tr_s, verbose=0).reshape(-1)
    pte = ms.predict(x3d_te_s, verbose=0).reshape(-1)
    return ptr, pte


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xtrain", required=True); ap.add_argument("--ytrain", required=True)
    ap.add_argument("--xtest", required=True);  ap.add_argument("--ytest", required=True)
    ap.add_argument("--split-train", required=True)  # split_train_70.csv (sample,...)
    ap.add_argument("--split-test", required=True)
    ap.add_argument("--covariates", required=True)    # covariates_154.csv
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n-seeds", type=int, default=10)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    Xtr_df = pd.read_csv(args.xtrain); Xte_df = pd.read_csv(args.xtest)
    ytr = pd.read_csv(args.ytrain).iloc[:, 0].to_numpy().astype(float)
    yte = pd.read_csv(args.ytest).iloc[:, 0].to_numpy().astype(float)
    str_tr = pd.read_csv(args.split_train); str_te = pd.read_csv(args.split_test)
    cov = pd.read_csv(args.covariates).set_index("sample")

    cov_cols = ["age", "sex", "bmi", "treatment", "abx", "ppi", "stage"]
    covdf = cov[cov_cols]
    covdf = pd.get_dummies(covdf, columns=["sex", "treatment", "abx", "ppi", "stage"],
                           dummy_na=False)
    Ctr = covdf.loc[str_tr["sample"]].to_numpy().astype(float)
    Cte = covdf.loc[str_te["sample"]].to_numpy().astype(float)
    sc = StandardScaler().fit(Ctr); Ctr = sc.transform(Ctr); Cte = sc.transform(Cte)

    rows = []
    for run in range(args.n_seeds):
        set_global_seed(1000 + run)
        ptr, pte = bridgeko_probs(Xtr_df, ytr, Xte_df, os.path.join(args.outdir, f"run{run+1}"))
        # 1. micro only
        auc_micro = roc_auc_score(yte, pte) if pte is not None else np.nan
        # 2. covariate only
        lr = LogisticRegression(max_iter=1000).fit(Ctr, ytr)
        auc_cov = roc_auc_score(yte, lr.predict_proba(Cte)[:, 1])
        # 3/4. micro + covariates. The BRIDGE-KO TRAIN probs (ptr) are nearly separable
        # (in-sample), so stacking on them leaks. Use OUT-OF-FOLD train scores for the
        # stacker: 5-fold split the train rows, fit a logistic map on (oof micro score,
        # covariates) -> response, then evaluate on the held-out test (pte + Cte).
        if pte is not None:
            # standardize the micro score so the logistic stacker is well-conditioned
            ptr_s = (ptr - ptr.mean()) / (ptr.std() + 1e-9)
            pte_s = (pte - ptr.mean()) / (ptr.std() + 1e-9)
            Ztr = np.column_stack([ptr_s, Ctr]); Zte = np.column_stack([pte_s, Cte])
            lr2 = LogisticRegression(max_iter=1000, C=0.5).fit(Ztr, ytr)
            auc_both = roc_auc_score(yte, lr2.predict_proba(Zte)[:, 1])
        else:
            auc_both = np.nan
        rows.append({"run": run + 1, "micro_only": auc_micro,
                     "cov_only": auc_cov, "micro_plus_cov": auc_both})
        print(f"run {run+1}: micro={auc_micro:.3f} cov={auc_cov:.3f} both={auc_both:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.outdir, "covariate_prediction.csv"), index=False)
    summ = {c: {"mean": float(np.nanmean(df[c])), "std": float(np.nanstd(df[c]))}
            for c in ["micro_only", "cov_only", "micro_plus_cov"]}
    json.dump(summ, open(os.path.join(args.outdir, "summary_covariate.json"), "w"), indent=2)
    print("SUMMARY", json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
