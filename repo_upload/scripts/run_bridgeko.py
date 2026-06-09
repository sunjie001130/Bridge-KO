#!/usr/bin/env python3
"""
run_bridgeko.py — parameterized BRIDGE-KO DNN + knockoff-FDR + AUC runner.

Refactor of Bridge-KO/LLM_AUC.ipynb into a CLI so every revision experiment
reuses one engine. Steps per seed (unchanged from the notebook):
  1. train full paired original+knockoff DNN  -> per-epoch feature-importance
  2. knockoff FDR selection (DL/FDR/FDR_control.py)
  3. retrain DNN on selected features only
  4. AUC on held-out test set

Inputs are the "combined" CSVs: first p columns = original features, next p
columns = matched knockoff features (as produced by the knockoff_*.Rmd scripts).
Y files: single column of 0/1 response labels (header row present).

Usage:
  python run_bridgeko.py \
      --xtrain train_big_equi_knockoffs_70_25_pca.csv \
      --ytrain meta_train_70_pca.csv \
      --xtest  test_big_equi_knockoffs_30_25_pca.csv \
      --ytest  meta_test_30_pca.csv \
      --outdir results/equi_25_pca --n-seeds 10 --tag equi_pca_25
"""
import os, sys, random, argparse, json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Make the vendored DL package importable (code/DL/...). This file lives in
# <repo>/revision/scripts/, so the repo root is two levels up.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "code"))

import tensorflow as tf  # noqa: E402
from DL.DNN.DNN import DNN  # noqa: E402
from DL.FDR.FDR_control import FDR_control  # noqa: E402


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_labels(csv_path, label_col=None):
    df = pd.read_csv(csv_path)
    s = df.iloc[:, 0] if label_col is None else df[label_col]
    return s.to_numpy().astype(float).reshape(-1)


def build_x3d_from_knockoff(df, keep_cols=None):
    """Split a combined original+knockoff frame into a (n, p, 2) tensor."""
    n, m = df.shape
    p = m // 2
    orig_cols = list(df.columns[:p])
    if keep_cols is None:
        sel_idx = list(range(p)); used_cols = orig_cols
    else:
        pos = {c: i for i, c in enumerate(orig_cols)}
        used_cols = [c for c in keep_cols if c in pos]
        sel_idx = [pos[c] for c in used_cols]
    X_orig = df.iloc[:, sel_idx].to_numpy()
    X_knkf = df.iloc[:, p + np.array(sel_idx)].to_numpy()
    x3d = np.zeros((X_orig.shape[0], len(sel_idx), 2), dtype=float)
    x3d[:, :, 0] = X_orig
    x3d[:, :, 1] = X_knkf
    return x3d, used_cols, sel_idx


def write_original_only_tsv_from_knockoff(df, out_path):
    """FDR_control.controlFilter reads X with sep='\\t' to recover feature names."""
    p = df.shape[1] // 2
    df.iloc[:, :p].to_csv(out_path, index=False, sep="\t")


def run_one_seed(run, seed, Xtr_df, ytr, Xte_df, yte, run_dir, epochs, batch_size):
    set_global_seed(seed)
    train_orig_cols = list(Xtr_df.columns[: Xtr_df.shape[1] // 2])

    # Step 1: full DNN (produces per-epoch feature-importance files in run_dir)
    x3d_train_full, _, _ = build_x3d_from_knockoff(Xtr_df, keep_cols=train_orig_cols)
    p_full = x3d_train_full.shape[1]
    coeff = 0.05 * np.sqrt(2.0 * np.log(max(p_full, 1)) / max(x3d_train_full.shape[0], 1))

    dnn_full = DNN(num_epochs=epochs, batch_size=batch_size, output_layer_activation="sigmoid")
    model_full = dnn_full.build_DNN(p_full, n_outputs=1, coeff=coeff)
    model_full.compile(loss="binary_crossentropy", optimizer="adam", metrics=["AUC"])
    os.makedirs(run_dir, exist_ok=True)
    cb = DNN.Job_finish_Callback(run_dir, p_full)
    model_full.fit(x3d_train_full, ytr, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[cb])

    # Step 2: FDR selection
    tmp_orig = os.path.join(run_dir, "train_orig.csv")
    write_original_only_tsv_from_knockoff(Xtr_df, tmp_orig)
    fdr = FDR_control()
    selected = fdr.controlFilter(tmp_orig, run_dir, offset=1, q=0.05)
    selected_features = [f for f, stat in selected]
    n_sel = len(selected_features)

    pd.DataFrame({"feature": selected_features}).to_csv(
        os.path.join(run_dir, f"selected_features_run{run+1}.csv"), index=False)

    if n_sel == 0:
        return np.nan, 0

    # Step 3+4: retrain on selected, AUC on test
    x3d_tr_sel, _, _ = build_x3d_from_knockoff(Xtr_df, keep_cols=selected_features)
    x3d_te_sel, _, _ = build_x3d_from_knockoff(Xte_df, keep_cols=selected_features)
    p_sel = x3d_tr_sel.shape[1]
    coeff_sel = 0.05 * np.sqrt(2.0 * np.log(max(p_sel, 1)) / max(x3d_tr_sel.shape[0], 1))
    dnn_sel = DNN(num_epochs=epochs, batch_size=batch_size, output_layer_activation="sigmoid")
    model_sel = dnn_sel.build_DNN(p_sel, n_outputs=1, coeff=coeff_sel)
    model_sel.compile(loss="binary_crossentropy", optimizer="adam", metrics=["AUC"])
    model_sel.fit(x3d_tr_sel, ytr, epochs=epochs, batch_size=batch_size, verbose=0)
    yprob = model_sel.predict(x3d_te_sel, verbose=0).reshape(-1)
    return roc_auc_score(yte, yprob), n_sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xtrain", required=True)
    ap.add_argument("--ytrain", required=True)
    ap.add_argument("--xtest", required=True)
    ap.add_argument("--ytest", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tag", default="run")
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

    aucs, nsel = [], []
    for run in range(args.n_seeds):
        seed = args.seed_base + run
        run_dir = os.path.join(args.outdir, f"run{run+1}")
        auc, n = run_one_seed(run, seed, Xtr_df, ytr, Xte_df, yte,
                              run_dir, args.epochs, args.batch_size)
        aucs.append(auc); nsel.append(n)
        print(f"[{args.tag}] run {run+1}/{args.n_seeds} seed={seed} n_sel={n} auc={auc}", flush=True)

    out = pd.DataFrame({"tag": args.tag, "run": list(range(1, args.n_seeds + 1)),
                        "seed": [args.seed_base + r for r in range(args.n_seeds)],
                        "n_selected": nsel, "auc": aucs})
    out_csv = os.path.join(args.outdir, f"auc_{args.tag}.csv")
    out.to_csv(out_csv, index=False)
    valid = [a for a in aucs if not np.isnan(a)]
    summary = {"tag": args.tag, "n_valid": len(valid),
               "mean_auc": float(np.mean(valid)) if valid else None,
               "std_auc": float(np.std(valid)) if valid else None,
               "mean_n_selected": float(np.mean([n for n in nsel if n > 0])) if any(nsel) else 0.0}
    with open(os.path.join(args.outdir, f"summary_{args.tag}.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
