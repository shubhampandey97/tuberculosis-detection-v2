"""Single-image inference shared by the Streamlit app (and the FastAPI endpoint planned in goal 8).

Same preprocessing as training (data_loader.load_image): RGB, plain bilinear resize to the
configured size, raw 0-255 floats. Model-specific preprocessing lives inside the model.
"""
import json
import os

import numpy as np

from src.utils.config_loader import load_config

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PRIMARY_MODEL = "resnet"  # chosen on val AUC (docs/audit.md, goal 4)
THRESHOLDS_PATH = os.path.join(ROOT, "artifacts", "clinical", "thresholds.json")


def _config():
    return load_config()


def img_size():
    return tuple(_config()["training"]["img_size"])


def load_model(name=PRIMARY_MODEL):
    import keras
    import src.components.model_builder  # noqa: F401  registers CaffePreprocess before loading

    path = os.path.join(ROOT, _config()["models_dir"], f"{name}_model.keras")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Train it with: python -m src.pipeline.train_pipeline")
    return keras.models.load_model(path)


def load_threshold(name=PRIMARY_MODEL):
    """Val-tuned threshold (highest threshold with val sensitivity >= 0.95), from goal 4."""
    if not os.path.exists(THRESHOLDS_PATH):
        raise FileNotFoundError(f"{THRESHOLDS_PATH} not found. Run: python -m src.pipeline.evaluate_clinical")
    with open(THRESHOLDS_PATH) as f:
        return float(json.load(f)["models"][name]["threshold"])


def preprocess_bytes(raw, size=None):
    """Encoded JPEG/PNG bytes -> (1, H, W, 3) float32 in 0-255, bit-identical to training.

    Uses the same TensorFlow decoder as data_loader.load_image. PIL decodes JPEGs slightly
    differently (different IDCT), which moved test scores by up to 0.024, so the model input
    must come from this path. Grayscale is expanded to RGB and alpha is dropped (channels=3).
    """
    import tensorflow as tf
    from src.components.data_loader import decode_and_resize

    return decode_and_resize(tf.constant(raw), size or img_size()).numpy()[None]


def preprocess(pil_image, size=None):
    """Fallback for an already-decoded PIL image (any mode) -> (1, H, W, 3) float32 in 0-255.

    Not bit-identical to training for JPEGs (see preprocess_bytes); prefer preprocess_bytes.
    """
    import tensorflow as tf

    rgb = np.asarray(pil_image.convert("RGB"), dtype=np.float32)
    return tf.image.resize(rgb, size or img_size()).numpy()[None]


def predict(model, x):
    """Uncalibrated model score P(TB) for a preprocessed batch of one image."""
    return float(model.predict(x, verbose=0).ravel()[0])


def explain(model, x, score, threshold, name=PRIMARY_MODEL):
    """Predicted-class Grad-CAM overlay (uint8 RGB) and whether the map was empty."""
    from src.components.gradcam import gradcam_signed, overlay, to_heatmap
    from src.components.model_builder import LAST_CONV_LAYER

    layer = LAST_CONV_LAYER[name]
    cams, _, _ = gradcam_signed(x, model, [layer])
    heatmap, empty = to_heatmap(cams[layer], 1.0 if score >= threshold else -1.0)
    return overlay(x[0], heatmap[0]), bool(empty[0])
