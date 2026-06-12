#!/usr/bin/env python3
"""
bg_reg.py -- background-regularization penalty (L_BG) for the BRIDGE-KO
contrastive predictor.

Implements the training-time term formalized in the revised Methods (ver6,
reviewer R1 #3):

    L_BG = (1/M) * sum_j max(0, |a~_j| - |a_j| + m),     m >= 0
    L    = L_pred + lambda_BG * L_BG + lambda_L1 * L_L1

where a_j / a~_j are the per-feature contributions of the original / knockoff
channels, "derived from the trained DNN weight matrices". We use exactly the
quantity the importance callback in DL/DNN/DNN.py builds as `h0`:

    h0[:, c] = local1.kernel[:, c, :] @ local2.kernel[:, :, :]      (c=0 orig, c=1 knockoff)

For this network local1.kernel has shape (p, 2, 1) and local2.kernel (p, 1, 1),
so the per-feature contributions reduce to the elementwise products:

    a_j    = local1.kernel[j, 0, 0] * local2.kernel[j, 0, 0]   (original channel)
    a~_j   = local1.kernel[j, 1, 0] * local2.kernel[j, 0, 0]   (knockoff channel)

These are products of trainable kernels, hence differentiable, so L_BG enters
the optimization objective and contributes gradients alongside the compiled
prediction loss and the L1 kernel penalty.

The DNN/DL package itself is intentionally left untouched: when lambda_bg <= 0
this function is a no-op, so the default pipeline is byte-identical to the
existing one and all cached AUCs reproduce.
"""
import tensorflow as tf


def _locally_connected_layers(model):
    """Return the two LocallyConnected1D layers (input-pairing layers) in order.

    Mirrors the layer access in DL/DNN/DNN.py's Job_finish_Callback, which reads
    model.layers[1] and model.layers[2]. We locate them by type to stay robust
    to any wrapping, but assert exactly two are present.
    """
    locs = [l for l in model.layers if type(l).__name__ == "LocallyConnected1D"]
    if len(locs) != 2:
        raise ValueError(
            f"expected 2 LocallyConnected1D layers, found {len(locs)}; "
            "bg_reg assumes the DeepMicroNET paired-input architecture")
    return locs[0], locs[1]


def add_background_regularization(model, lambda_bg=0.0, margin=0.0):
    """Attach the L_BG penalty to a built BRIDGE-KO DNN via model.add_loss().

    Parameters
    ----------
    model : keras.Model
        A model built by DL.DNN.DNN.build_DNN (already compiled or not).
    lambda_bg : float
        Strength of the background-regularization term. lambda_bg <= 0 is a
        no-op (returns the model unchanged) -> default pipeline reproduces.
    margin : float
        Margin m >= 0 in the hinge. m=0 penalizes only when the knockoff
        contribution matches/exceeds the original; m>0 demands the original
        exceed the knockoff by at least m.

    Returns
    -------
    model : the same model, with the penalty added when lambda_bg > 0.
    """
    if lambda_bg is None or lambda_bg <= 0.0:
        return model  # no-op: byte-identical to the un-regularized pipeline

    if margin < 0.0:
        raise ValueError(f"margin must be >= 0, got {margin}")

    local1, local2 = _locally_connected_layers(model)
    lam = tf.constant(float(lambda_bg), dtype=tf.float32)
    m = tf.constant(float(margin), dtype=tf.float32)

    def _l_bg():
        # Per-feature contributions h0[:, 0] (original) and h0[:, 1] (knockoff),
        # exactly as the importance callback forms them (DNN.py:95).
        k1 = local1.kernel  # (p, 2, 1)
        k2 = local2.kernel  # (p, 1, 1)
        a_orig = k1[:, 0, 0] * k2[:, 0, 0]   # (p,)
        a_knkf = k1[:, 1, 0] * k2[:, 0, 0]   # (p,)
        hinge = tf.nn.relu(tf.abs(a_knkf) - tf.abs(a_orig) + m)
        return lam * tf.reduce_mean(hinge)

    # Pass a callable so the penalty re-evaluates from current kernel values
    # each step (keeps gradients flowing to local1/local2 kernels).
    model.add_loss(_l_bg)
    return model


def l_bg_value(model, margin=0.0):
    """Compute the *unweighted* L_BG = mean(relu(|a~|-|a|+m)) for logging/sanity.

    Returns a python float from the model's current weights. Used by the
    gradient-sanity check to confirm the term decreases over epochs.
    """
    local1, local2 = _locally_connected_layers(model)
    k1 = local1.kernel
    k2 = local2.kernel
    a_orig = k1[:, 0, 0] * k2[:, 0, 0]
    a_knkf = k1[:, 1, 0] * k2[:, 0, 0]
    hinge = tf.nn.relu(tf.abs(a_knkf) - tf.abs(a_orig) + float(margin))
    return float(tf.reduce_mean(hinge).numpy())
