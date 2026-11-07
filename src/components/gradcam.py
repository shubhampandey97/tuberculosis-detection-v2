"""Grad-CAM for the TB classifiers (Keras 3).

Target: the pre-sigmoid logit z of the final Dense(1, sigmoid) layer, not the probability.
Gradients of a saturated sigmoid (P near 0 or 1) vanish, which washes out heatmaps for
confident predictions; the logit does not saturate.

Two maps come from one backward pass (signed CAM = sum_c A_c * mean_spatial(dz/dA_c)):
  * TB-evidence map        = ReLU(+CAM)  same target for every image
  * predicted-class map    = ReLU(+CAM) if predicted TB, ReLU(-CAM) if predicted Normal
Layers: one documented last-conv layer per model, see model_builder.LAST_CONV_LAYER.
"""
import keras
import numpy as np
import tensorflow as tf
from matplotlib import colormaps

from src.components.model_builder import LAST_CONV_LAYER  # noqa: F401  (also registers CaffePreprocess)

EMPTY_EPS = 1e-8  # a map whose max is <= this is "empty" (all evidence negative / zero)
_GRAD_MODELS = {}


def _grad_model(model, layer_names):
    key = (id(model), tuple(layer_names))
    if key not in _GRAD_MODELS:
        head = model.layers[-1]  # Dense(1, sigmoid)
        outputs = [model.get_layer(n).output for n in layer_names] + [head.input, model.output]
        _GRAD_MODELS[key] = keras.Model(model.inputs, outputs)  # Keras 3: no extra list around inputs
    return _GRAD_MODELS[key]


def gradcam_signed(images, model, layer_names, head_weights=None):
    """Signed CAMs for a batch of raw 0-255 RGB images.

    Returns ({layer: (B, h, w) signed CAM}, logits (B,), probs (B,)).
    head_weights=(W, b) replaces the final Dense weights (used for the randomised-head sanity check).
    """
    head = model.layers[-1]
    W, b = head_weights if head_weights is not None else (head.kernel, head.bias)
    x = tf.convert_to_tensor(images, tf.float32)
    with tf.GradientTape() as tape:
        *convs, feats, out = _grad_model(model, layer_names)(x, training=False)
        z = tf.matmul(feats, tf.cast(W, feats.dtype))[:, 0] + tf.cast(b, feats.dtype)[0]
        target = tf.reduce_sum(z)  # must be computed inside the tape
    grads = tape.gradient(target, convs)  # samples are independent -> per-sample grads
    cams = {}
    for name, a, g in zip(layer_names, convs, grads):
        weights = tf.reduce_mean(g, axis=(1, 2))  # (B, C) channel importance
        cams[name] = tf.einsum("bhwc,bc->bhw", a, weights).numpy()
    return cams, z.numpy(), np.asarray(out).ravel()


def to_heatmap(cam, sign=1.0):
    """ReLU(sign * CAM) scaled to [0, 1] per image. Returns (heatmaps, empty_flags).

    NaN guard: maps whose max <= EMPTY_EPS are returned as zeros and flagged empty.
    """
    sign = np.asarray(sign, dtype=np.float32).reshape(-1, 1, 1) if np.ndim(sign) else sign
    hm = np.maximum(cam * sign, 0.0)
    peak = hm.reshape(len(hm), -1).max(axis=1)
    empty = peak <= EMPTY_EPS
    scale = np.where(empty, 1.0, peak).reshape(-1, 1, 1)
    hm = np.where(empty.reshape(-1, 1, 1), 0.0, hm / scale)
    return hm.astype(np.float32), empty


def upsample(heatmaps, size):
    """Bilinear resize of (B, h, w) heatmaps to (B, *size), clipped at 0."""
    up = tf.image.resize(np.asarray(heatmaps)[..., None], size, method="bilinear")[..., 0]
    return np.maximum(up.numpy(), 0.0)


def overlay(img, heatmap, alpha=0.4):
    """RGB uint8 overlay of a [0,1] heatmap (any size) on a 0-255 RGB image. No plotting side effects."""
    img = np.asarray(img, dtype=np.float32)
    hm = upsample(heatmap[None], img.shape[:2])[0]
    colour = colormaps["jet"](np.clip(hm, 0, 1))[..., :3] * 255.0
    return np.clip((1 - alpha) * img + alpha * colour, 0, 255).astype(np.uint8)


def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    """Backward-compatible helper: TB-evidence heatmap in [0, 1] for the first image of img_array."""
    cams, _, _ = gradcam_signed(img_array, model, [last_conv_layer_name])
    hm, _ = to_heatmap(cams[last_conv_layer_name])
    return hm[0]
