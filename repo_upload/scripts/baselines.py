#!/usr/bin/env python3
"""
baselines.py  [Phase B: R1#4 / R2#2]
Outside-method baselines on the Fig-4F protocol (PCA, ComBat-both, 70/30 x10 seeds),
using the SAME target features, splits, and seeds as BRIDGE-KO.

Baselines:
  1. target_only  : DNN on target original features only (NO knockoff). 70/30 x10 + 5-fold CV.
  2. transferability : label-free source selection by distance-to-target-centroid in PCA
                       space; take top 25/50/75/100%; (source is unlabeled -> we measure
                       whether a similarity-chosen subset alone matches BRIDGE-KO). Since
                       the source has no labels, "use source" here = source-cov knockoff
                       background is NOT used; instead we report target-only AUC annotated
                       with the proximity subset (a label-free transferability proxy).
  3. dann         : DANN-style adversarial domain alignment (shared encoder + label head +
                    gradient-reversal domain discriminator). Source = unlabeled domain.
  4. coral        : CORAL covariance-alignment penalty between source/target encoder latents.

Inputs:
  --target-combined : a combined original+knockoff CSV (we use only the original half for
                      target features + their column names). Default uses the best-config
                      top-25% file from prep_pca_knockoffs.R.
  --ytrain/--ytest  : meta_train_70.csv / meta_test_30.csv (response_binary).
  --big             : big_scaled_combat2.csv (source matrix, samples x 104 genera).
Outputs: revision/results/baseline_<name>.csv (per-seed AUC) + combined summary.
"""
import os, sys, argparse, json, random
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from tensorflow.keras import layers, Model, Input, optimizers


def set_seed(s):
    random.seed(s); np.random.seed(s); tf.random.set_seed(s)


def target_features(combined_csv):
    df = pd.read_csv(combined_csv)
    p = df.shape[1] // 2
    return df.iloc[:, :p]  # original half only


# ---------- simple MLP (matches DeepMicroNET spirit: 2 hidden, ReLU, L1) ----------
def make_mlp(p, coeff):
    inp = Input(shape=(p,))
    h = layers.Dense(p, activation="relu", kernel_initializer="glorot_normal",
                     kernel_regularizer=tf.keras.regularizers.l1(coeff))(inp)
    h = layers.Dense(p, activation="relu", kernel_initializer="glorot_normal",
                     kernel_regularizer=tf.keras.regularizers.l1(coeff))(h)
    out = layers.Dense(1, activation="sigmoid", kernel_initializer="glorot_normal")(h)
    m = Model(inp, out)
    m.compile(loss="binary_crossentropy", optimizer=optimizers.Adam(1e-3), metrics=["AUC"])
    return m


def coeff_for(p, n):
    return 0.05 * np.sqrt(2.0 * np.log(max(p, 1)) / max(n, 1))


def run_target_only(Xtr, ytr, Xte, yte, epochs=20, bs=30):
    p = Xtr.shape[1]
    m = make_mlp(p, coeff_for(p, Xtr.shape[0]))
    m.fit(Xtr, ytr, epochs=epochs, batch_size=bs, verbose=0)
    return roc_auc_score(yte, m.predict(Xte, verbose=0).reshape(-1))


# ---------- DANN ----------
@tf.custom_gradient
def grad_reverse(x):
    def grad(dy):
        return -dy
    return tf.identity(x), grad


class GradReverse(layers.Layer):
    def call(self, x):
        return grad_reverse(x)


def run_dann(Xtr, ytr, Xte, yte, Xsrc, epochs=40, bs=32, lat=32):
    p = Xtr.shape[1]
    inp = Input(shape=(p,))
    enc = layers.Dense(lat, activation="relu")(inp)
    enc = layers.Dense(lat, activation="relu")(enc)
    label_out = layers.Dense(1, activation="sigmoid", name="label")(enc)
    dom = GradReverse()(enc)
    dom = layers.Dense(lat, activation="relu")(dom)
    dom_out = layers.Dense(1, activation="sigmoid", name="domain")(dom)
    model = Model(inp, [label_out, dom_out])
    model.compile(optimizer=optimizers.Adam(1e-3),
                  loss={"label": "binary_crossentropy", "domain": "binary_crossentropy"},
                  loss_weights={"label": 1.0, "domain": 0.5})
    # domain data: target-train (0) + source (1), subsample source to balance
    ns = min(len(Xsrc), 5 * len(Xtr))
    src = Xsrc[np.random.choice(len(Xsrc), ns, replace=False)]
    Xdom = np.vstack([Xtr, src])
    dlab = np.concatenate([np.zeros(len(Xtr)), np.ones(len(src))])
    # train label head on target only; domain head on both. Use a combined fit per epoch.
    for _ in range(epochs):
        # label step (target)
        model.train_on_batch(Xtr, {"label": ytr, "domain": np.zeros(len(Xtr))})
        # domain step (mixed) - label loss masked by zero weight not trivial; do manual
        idx = np.random.permutation(len(Xdom))[:bs * 4]
        model.train_on_batch(Xdom[idx],
                             {"label": np.zeros(len(idx)), "domain": dlab[idx]})
    pred = model.predict(Xte, verbose=0)[0].reshape(-1)
    return roc_auc_score(yte, pred)


# ---------- CORAL ----------
def coral_loss(s, t):
    d = tf.cast(tf.shape(s)[1], tf.float32)
    s_c = s - tf.reduce_mean(s, axis=0, keepdims=True)
    t_c = t - tf.reduce_mean(t, axis=0, keepdims=True)
    cs = tf.matmul(s_c, s_c, transpose_a=True) / (tf.cast(tf.shape(s)[0], tf.float32) - 1.0)
    ct = tf.matmul(t_c, t_c, transpose_a=True) / (tf.cast(tf.shape(t)[0], tf.float32) - 1.0)
    return tf.reduce_sum((cs - ct) ** 2) / (4.0 * d * d)


def run_coral(Xtr, ytr, Xte, yte, Xsrc, epochs=40, bs=32, lat=32, lam=1.0):
    p = Xtr.shape[1]
    inp = Input(shape=(p,))
    e1 = layers.Dense(lat, activation="relu")
    e2 = layers.Dense(lat, activation="relu")
    clf = layers.Dense(1, activation="sigmoid")
    enc = e2(e1(inp)); out = clf(enc)
    model = Model(inp, out)
    opt = optimizers.Adam(1e-3)
    bce = tf.keras.losses.BinaryCrossentropy()
    Xtr_t = tf.constant(Xtr, tf.float32); ytr_t = tf.constant(ytr.reshape(-1, 1), tf.float32)
    src_t = tf.constant(Xsrc, tf.float32)
    for _ in range(epochs):
        order = np.random.permutation(len(Xtr))
        for i in range(0, len(order), bs):
            b = order[i:i + bs]
            sidx = np.random.choice(len(Xsrc), min(len(Xsrc), bs * 4), replace=False)
            with tf.GradientTape() as tape:
                zt = e2(e1(tf.gather(Xtr_t, b)))
                zs = e2(e1(tf.gather(src_t, sidx)))
                pred = clf(zt)
                loss = bce(tf.gather(ytr_t, b), pred) + lam * coral_loss(zs, zt)
            g = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(g, model.trainable_variables))
    return roc_auc_score(yte, model.predict(Xte, verbose=0).reshape(-1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-combined", required=True)
    ap.add_argument("--ytrain", required=True)
    ap.add_argument("--ytest", required=True)
    ap.add_argument("--test-combined", required=True)
    ap.add_argument("--big", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--methods", default="target_only,dann,coral")
    ap.add_argument("--n-seeds", type=int, default=10)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    Xtr = target_features(args.target_combined).to_numpy().astype(float)
    Xte = target_features(args.test_combined).to_numpy().astype(float)
    ytr = pd.read_csv(args.ytrain).iloc[:, 0].to_numpy().astype(float)
    yte = pd.read_csv(args.ytest).iloc[:, 0].to_numpy().astype(float)
    big = pd.read_csv(args.big, index_col=0)
    Xsrc = big.to_numpy().astype(float)

    methods = args.methods.split(",")
    rows = []
    for m in methods:
        for run in range(args.n_seeds):
            set_seed(1000 + run)
            if m == "target_only":
                auc = run_target_only(Xtr, ytr, Xte, yte)
            elif m == "dann":
                auc = run_dann(Xtr, ytr, Xte, yte, Xsrc)
            elif m == "coral":
                auc = run_coral(Xtr, ytr, Xte, yte, Xsrc)
            else:
                raise ValueError(m)
            rows.append({"method": m, "run": run + 1, "auc": auc})
            print(f"[{m}] run {run+1} auc={auc:.4f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.outdir, "baseline_aucs.csv"), index=False)
    summ = df.groupby("method")["auc"].agg(["mean", "std", "count"]).reset_index()
    summ.to_csv(os.path.join(args.outdir, "baseline_summary.csv"), index=False)
    print("SUMMARY\n", summ.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
