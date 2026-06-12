#!/usr/bin/env python3
"""
tune_bg_reg.py -- tune the background-regularization penalty (lambda_BG, margin)
for the BRIDGE-KO contrastive predictor and report its effect on the headline
arm (PCA, ComBat-both, top-25% source), exactly as the revised Methods (ver6,
R1 #3) promise:

  "lambda_BG, and the margin m when applicable, are selected using the training
   data only. The training set is further divided into an internal training
   subset and validation subset, candidate values are evaluated over a
   predefined grid, and the selected value maximizes validation AUC ... The
   held-out test set is reserved only for final model evaluation."

Protocol, per seed (10 seeds):
  1. Split the 70%-train ROWS into internal-train / internal-val (80/20,
     stratified on response, seeded).
  2. For each (lambda_BG, margin) on the grid: run the full BRIDGE-KO pipeline
     (train full -> FDR select -> retrain) on internal-train, score VALIDATION
     AUC on internal-val.
  3. Pick (lambda_BG*, margin*) = argmax val AUC (tie-break -> smaller lambda_BG,
     then smaller margin, for parsimony).
  4. Refit on the FULL 70%-train at (lambda_BG*, margin*); evaluate on the
     held-out 30% TEST. Also evaluate lambda_BG=0 on the same split as the
     baseline reference.

Outputs (revision/results/bg_reg/):
  tuning_grid.csv   -- per-seed validation AUC for every grid point
  auc_bg_tuned.csv  -- per-seed: chosen lambda_BG*/margin*, tuned test AUC,
                       baseline (lambda_BG=0) test AUC, n_selected
  summary_bg.json   -- mean/std test AUC for tuned vs baseline

The combined CSVs hold the original features in the first p columns and the
matched knockoffs in the next p columns; splitting "samples" is just splitting
rows of that frame (the column pairing is preserved).
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_bridgeko import set_global_seed, load_labels, fit_select_eval  # noqa: E402

# Predefined grid (see Methods). lambda_BG=0 is always included as the baseline.
LAMBDA_GRID = [0.0, 0.01, 0.1, 0.5, 1.0]
MARGIN_GRID = [0.0, 0.05]


def tune_one_seed(run, seed, Xtr_df, ytr, Xte_df, yte, seed_dir,
                  epochs, batch_size, val_frac=0.2):
    """Return (grid_rows, best, tuned_auc, tuned_nsel, base_auc, base_nsel)."""
    set_global_seed(seed)
    n = Xtr_df.shape[0]
    idx = np.arange(n)
    # Stratified internal train/val split on the 70%-train rows.
    strat = ytr if len(np.unique(ytr)) > 1 else None
    tr_idx, val_idx = train_test_split(
        idx, test_size=val_frac, random_state=seed, stratify=strat)
    Xtr_in = Xtr_df.iloc[tr_idx].reset_index(drop=True)
    ytr_in = ytr[tr_idx]
    Xval = Xtr_df.iloc[val_idx].reset_index(drop=True)
    yval = ytr[val_idx]

    grid_rows = []
    for lam in LAMBDA_GRID:
        for m in MARGIN_GRID:
            # margin only matters when the penalty is active; collapse the
            # redundant (lambda_BG=0, m>0) duplicates to a single baseline point.
            if lam == 0.0 and m != MARGIN_GRID[0]:
                continue
            set_global_seed(seed)  # same init per grid point -> fair comparison
            cfg_dir = os.path.join(seed_dir, f"val_lam{lam}_m{m}")
            auc, n_sel, _ = fit_select_eval(
                run, Xtr_in, ytr_in, Xval, yval, cfg_dir,
                epochs, batch_size, lambda_bg=lam, margin=m)
            grid_rows.append({"run": run + 1, "seed": seed, "lambda_bg": lam,
                              "margin": m, "val_auc": auc, "val_n_selected": n_sel})
            print(f"[tune] run{run+1} seed={seed} lam={lam} m={m} "
                  f"val_auc={auc} n_sel={n_sel}", flush=True)

    # Choose best by validation AUC; tie-break toward parsimony (small lam, m).
    valid = [r for r in grid_rows if not np.isnan(r["val_auc"])]
    best = max(valid, key=lambda r: (r["val_auc"], -r["lambda_bg"], -r["margin"]))
    lam_star, m_star = best["lambda_bg"], best["margin"]

    # Refit on FULL 70%-train at the chosen config; evaluate on held-out TEST.
    set_global_seed(seed)
    tuned_auc, tuned_nsel, _ = fit_select_eval(
        run, Xtr_df, ytr, Xte_df, yte, os.path.join(seed_dir, "final_tuned"),
        epochs, batch_size, lambda_bg=lam_star, margin=m_star)

    # Baseline (lambda_BG=0) on the same full-train/test for a paired comparison.
    set_global_seed(seed)
    base_auc, base_nsel, _ = fit_select_eval(
        run, Xtr_df, ytr, Xte_df, yte, os.path.join(seed_dir, "final_base"),
        epochs, batch_size, lambda_bg=0.0, margin=0.0)

    return grid_rows, best, tuned_auc, tuned_nsel, base_auc, base_nsel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xtrain", required=True)
    ap.add_argument("--ytrain", required=True)
    ap.add_argument("--xtest", required=True)
    ap.add_argument("--ytest", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=30)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    Xtr_df = pd.read_csv(args.xtrain)
    ytr = load_labels(args.ytrain)
    Xte_df = pd.read_csv(args.xtest)
    yte = load_labels(args.ytest)

    all_grid, per_seed = [], []
    for run in range(args.n_seeds):
        seed = args.seed_base + run
        seed_dir = os.path.join(args.outdir, f"run{run+1}")
        grid_rows, best, tuned_auc, tuned_nsel, base_auc, base_nsel = tune_one_seed(
            run, seed, Xtr_df, ytr, Xte_df, yte, seed_dir,
            args.epochs, args.batch_size)
        all_grid.extend(grid_rows)
        per_seed.append({"run": run + 1, "seed": seed,
                         "lambda_bg_star": best["lambda_bg"], "margin_star": best["margin"],
                         "val_auc_star": best["val_auc"],
                         "tuned_test_auc": tuned_auc, "tuned_n_selected": tuned_nsel,
                         "baseline_test_auc": base_auc, "baseline_n_selected": base_nsel})
        print(f"[tune] run{run+1} seed={seed} BEST lam={best['lambda_bg']} "
              f"m={best['margin']} | tuned_test={tuned_auc} base_test={base_auc}", flush=True)

    pd.DataFrame(all_grid).to_csv(os.path.join(args.outdir, "tuning_grid.csv"), index=False)
    ps = pd.DataFrame(per_seed)
    ps.to_csv(os.path.join(args.outdir, "auc_bg_tuned.csv"), index=False)

    def _ms(col):
        v = ps[col].dropna().to_numpy()
        return (float(np.mean(v)), float(np.std(v))) if len(v) else (None, None)

    tuned_mean, tuned_std = _ms("tuned_test_auc")
    base_mean, base_std = _ms("baseline_test_auc")
    summary = {
        "arm": "PCA, ComBat-both, top-25% (headline)",
        "n_seeds": args.n_seeds,
        "lambda_grid": LAMBDA_GRID, "margin_grid": MARGIN_GRID,
        "tuned_mean_auc": tuned_mean, "tuned_std_auc": tuned_std,
        "baseline_mean_auc": base_mean, "baseline_std_auc": base_std,
        "lambda_bg_star_counts": ps["lambda_bg_star"].value_counts().to_dict(),
        "margin_star_counts": ps["margin_star"].value_counts().to_dict(),
    }
    with open(os.path.join(args.outdir, "summary_bg.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
