#!/usr/bin/env python3
"""
stability_auc.py  [Phase E: R1#6 / R2#4 / R2#6]
Two stability analyses for the best-config BRIDGE-KO AUC, reusing the run_bridgeko engine:

  --mode bootstrap : stratified balanced bootstrap of the TARGET test set by city x R/NR.
      Train once per seed on the fixed 70% (as in the paper); for each of B bootstrap
      resamples of the 30% test set (balanced across city x response strata), recompute
      AUC -> distribution + 95% CI. Repeated across the 10 trained seeds.

  --mode permutation : outcome-label permutation. Permute the TRAIN response labels,
      run the full select+retrain+AUC pipeline, build a null AUC distribution -> empirical
      p-value vs the observed mean AUC.

Inputs: best-config train/test combined CSVs + meta + covariates_154.csv (for strata).
Outputs: revision/results/stability_<mode>.csv (+ summary json)
"""
import os, sys, argparse, json, random
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from run_bridgeko import (set_global_seed, build_x3d_from_knockoff,
                          write_original_only_tsv_from_knockoff, DNN, FDR_control)


def train_select_retrain(Xtr_df, ytr, run_dir, epochs=20, bs=30):
    """Returns a trained model on selected features + the selected feature list."""
    cols = list(Xtr_df.columns[: Xtr_df.shape[1] // 2])
    x3d, _, _ = build_x3d_from_knockoff(Xtr_df, keep_cols=cols)
    p = x3d.shape[1]
    coeff = 0.05 * np.sqrt(2.0 * np.log(max(p, 1)) / max(x3d.shape[0], 1))
    os.makedirs(run_dir, exist_ok=True)
    full = DNN(num_epochs=epochs, batch_size=bs, output_layer_activation="sigmoid")
    mf = full.build_DNN(p, 1, coeff); mf.compile(loss="binary_crossentropy", optimizer="adam")
    mf.fit(x3d, ytr, epochs=epochs, batch_size=bs, verbose=0,
           callbacks=[DNN.Job_finish_Callback(run_dir, p)])
    tmp = os.path.join(run_dir, "train_orig.csv")
    write_original_only_tsv_from_knockoff(Xtr_df, tmp)
    sel = [f for f, _ in FDR_control().controlFilter(tmp, run_dir, offset=1, q=0.05)]
    if not sel:
        return None, []
    x3d_s, _, _ = build_x3d_from_knockoff(Xtr_df, keep_cols=sel)
    ps = x3d_s.shape[1]
    cs = 0.05 * np.sqrt(2.0 * np.log(max(ps, 1)) / max(x3d_s.shape[0], 1))
    ms = DNN(num_epochs=epochs, batch_size=bs, output_layer_activation="sigmoid")
    model = ms.build_DNN(ps, 1, cs); model.compile(loss="binary_crossentropy", optimizer="adam")
    model.fit(x3d_s, ytr, epochs=epochs, batch_size=bs, verbose=0)
    return model, sel


def strata_bootstrap_idx(strata, rng):
    """Balanced resample: equal draw per stratum, total = n."""
    levels = pd.unique(strata)
    per = max(1, len(strata) // len(levels))
    idx = []
    for lv in levels:
        pool = np.where(strata == lv)[0]
        idx.extend(rng.choice(pool, per, replace=True))
    return np.array(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["bootstrap", "permutation"], required=True)
    ap.add_argument("--xtrain", required=True); ap.add_argument("--ytrain", required=True)
    ap.add_argument("--xtest", required=True);  ap.add_argument("--ytest", required=True)
    ap.add_argument("--covariates", required=True)   # covariates_154.csv (for test strata)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-perm", type=int, default=200)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    Xtr_df = pd.read_csv(args.xtrain); Xte_df = pd.read_csv(args.xtest)
    ytr = pd.read_csv(args.ytrain).iloc[:, 0].to_numpy().astype(float)
    yte = pd.read_csv(args.ytest).iloc[:, 0].to_numpy().astype(float)

    if args.mode == "bootstrap":
        # need test-set strata aligned to Xte rows; covariates file is in target order,
        # but test rows are a 70/30 subset. We approximate strata by response only if
        # city alignment is unavailable for the test subset; here we use response x (city
        # if provided alongside). For now build strata from yte + an optional city file.
        # covariates here = split_test_30.csv (sample, city, response_binary) aligned to
        # the test rows in order; build city x response strata.
        split_te = pd.read_csv(args.covariates)
        assert len(split_te) == len(yte), "test strata file must align to test rows"
        strata_full = (split_te["city"].astype(str) + "|" +
                       split_te["response_binary"].astype(int).astype(str)).to_numpy()
        rows = []
        for run in range(args.n_seeds):
            set_global_seed(1000 + run)
            model, sel = train_select_retrain(Xtr_df, ytr, os.path.join(args.outdir, f"run{run+1}"))
            if model is None:
                continue
            x3d_te, _, _ = build_x3d_from_knockoff(Xte_df, keep_cols=sel)
            yprob = model.predict(x3d_te, verbose=0).reshape(-1)
            rng = np.random.default_rng(1000 + run)
            strata = strata_full  # city x response strata
            for b in range(args.n_boot):
                bi = strata_bootstrap_idx(strata, rng)
                if len(np.unique(yte[bi])) < 2:
                    continue
                rows.append({"run": run + 1, "boot": b,
                             "auc": roc_auc_score(yte[bi], yprob[bi])})
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(args.outdir, "stability_bootstrap.csv"), index=False)
        a = df["auc"].to_numpy()
        summ = {"mean": float(np.mean(a)), "ci_lo": float(np.percentile(a, 2.5)),
                "ci_hi": float(np.percentile(a, 97.5)), "n": int(len(a))}
        json.dump(summ, open(os.path.join(args.outdir, "summary_bootstrap.json"), "w"), indent=2)
        print("BOOTSTRAP", json.dumps(summ), flush=True)

    else:  # permutation
        # observed
        obs = []
        for run in range(args.n_seeds):
            set_global_seed(1000 + run)
            model, sel = train_select_retrain(Xtr_df, ytr, os.path.join(args.outdir, f"obs_run{run+1}"))
            if model is None:
                continue
            x3d_te, _, _ = build_x3d_from_knockoff(Xte_df, keep_cols=sel)
            obs.append(roc_auc_score(yte, model.predict(x3d_te, verbose=0).reshape(-1)))
        obs_mean = float(np.mean(obs))
        # null: permute train labels
        null = []
        for p in range(args.n_perm):
            set_global_seed(5000 + p)
            yperm = np.random.permutation(ytr)
            model, sel = train_select_retrain(Xtr_df, yperm, os.path.join(args.outdir, f"perm{p}"))
            if model is None:
                null.append(0.5); continue
            x3d_te, _, _ = build_x3d_from_knockoff(Xte_df, keep_cols=sel)
            null.append(roc_auc_score(yte, model.predict(x3d_te, verbose=0).reshape(-1)))
            print(f"perm {p+1}/{args.n_perm} auc={null[-1]:.3f}", flush=True)
        null = np.array(null)
        pval = (1 + np.sum(null >= obs_mean)) / (len(null) + 1)
        pd.DataFrame({"perm": range(len(null)), "auc": null}).to_csv(
            os.path.join(args.outdir, "stability_permutation.csv"), index=False)
        summ = {"obs_mean_auc": obs_mean, "null_mean": float(np.mean(null)),
                "emp_p": float(pval), "n_perm": int(len(null))}
        json.dump(summ, open(os.path.join(args.outdir, "summary_permutation.json"), "w"), indent=2)
        print("PERMUTATION", json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
