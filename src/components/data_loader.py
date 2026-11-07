"""tf.data pipelines built from data/splits.csv.

Images come out as float32 RGB in the raw 0-255 range. Model-specific preprocessing lives
inside each model (see model_builder.py), so the same input works for training, evaluation,
Grad-CAM and the app.
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

AUTOTUNE = tf.data.AUTOTUNE


def read_split(config, split):
    df = pd.read_csv(config["splits_path"])
    part = df[df["split"] == split].reset_index(drop=True)
    if part.empty:
        raise ValueError(f"No rows for split '{split}' in {config['splits_path']}")
    return part


def load_image(path, img_size):
    """Read a JPEG/PNG file, decode as 3-channel RGB and resize to img_size. Returns float32 in 0-255."""
    return decode_and_resize(tf.io.read_file(path), img_size)


def decode_and_resize(encoded, img_size):
    """Encoded JPEG/PNG bytes -> RGB float32 in 0-255 at img_size (shared by training and the app)."""
    img = tf.io.decode_image(encoded, channels=3, expand_animations=False)
    img = tf.image.resize(img, img_size)  # plain resize: non-square images are stretched
    return tf.cast(img, tf.float32)


def build_augmentation(config):
    aug, seed = config["augmentation"], config["seed"]
    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(aug["rotation"], seed=seed),
        tf.keras.layers.RandomZoom(aug["zoom"], seed=seed),
    ], name="augmentation")


# ---------------------------------------------------------------- goal 10: label-independent shortcut augmentation
MARKER_TOKENS = ["L", "R", "PA", "AP", "LT", "RT", "CXR", "ERECT", "PA\nERECT", "AP\nSUPINE"]


def _random_marker(rng):
    """A synthetic marker / text / handwriting patch: (gray uint8 patch, bool mask of drawn pixels)."""
    from PIL import Image, ImageDraw, ImageFont
    kind = rng.choice(["token", "date", "handwriting"], p=[0.55, 0.2, 0.25])
    if kind == "handwriting":
        w, h = int(rng.integers(40, 90)), int(rng.integers(12, 26))
        canvas = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(canvas)
        pts = [(0, h / 2)]
        for _ in range(int(rng.integers(6, 14))):
            pts.append((min(w - 1, pts[-1][0] + rng.uniform(3, 9)), rng.uniform(1, h - 1)))
        draw.line(pts, fill=255, width=int(rng.integers(1, 3)))
        mask = np.asarray(canvas) > 0
        value = 255 if rng.random() < 0.3 else 15   # handwriting is mostly dark ink
        patch = np.where(mask, value, 0).astype(np.uint8)
        return patch, mask
    text = (str(rng.choice(MARKER_TOKENS)) if kind == "token"
            else f"{rng.integers(1, 29):02d}/{rng.integers(1, 13):02d}/{rng.integers(10, 25):02d}")
    font = ImageFont.load_default(size=int(rng.integers(9, 22)))
    tmp = ImageDraw.Draw(Image.new("L", (1, 1)))
    left, top, right, bottom = tmp.multiline_textbbox((0, 0), text, font=font)
    pad = int(rng.integers(1, 4))
    w, h = right - left + 2 * pad, bottom - top + 2 * pad
    style = rng.choice(["white", "black", "boxed"])
    canvas = Image.new("L", (w, h), 255 if style == "boxed" else 0)
    ImageDraw.Draw(canvas).multiline_text((pad - left, pad - top), text, font=font,
                                          fill=0 if style == "boxed" else 255)
    arr = np.asarray(canvas)
    if style == "boxed":
        return arr.astype(np.uint8), np.ones_like(arr, dtype=bool)
    mask = arr > 64
    return np.where(mask, 255 if style == "white" else 10, 0).astype(np.uint8), mask


def add_markers(gray, rng, n_items):
    """Paste n_items random markers anywhere in a uint8 grayscale image (returns a new array)."""
    out = gray.copy()
    H, W = out.shape
    for _ in range(n_items):
        patch, mask = _random_marker(rng)
        ph, pw = min(patch.shape[0], H), min(patch.shape[1], W)
        patch, mask = patch[:ph, :pw], mask[:ph, :pw]
        y, x = int(rng.integers(0, H - ph + 1)), int(rng.integers(0, W - pw + 1))
        region = out[y:y + ph, x:x + pw]
        region[mask] = patch[mask]
    return out


def shortcut_augment(image, rng, params, force_markers=False):
    """Label-independent augmentation (goal 10). `image`: HxWx3 uint8 gray-as-RGB. Returns (uint8 image, info).
    The function never sees the label, so every probability is the same for both classes."""
    import io

    from PIL import Image
    gray = image[..., 0]
    info = {"markers": 0, "jpeg_quality": None}
    if force_markers or rng.random() < params["marker_p"]:
        info["markers"] = int(rng.integers(1, 4))
        gray = add_markers(gray, rng, info["markers"])
    if rng.random() < params["jpeg_p"]:
        q = int(rng.integers(params["jpeg_quality"][0], params["jpeg_quality"][1] + 1))
        buf = io.BytesIO()
        Image.fromarray(np.repeat(gray[..., None], 3, axis=2)).save(buf, format="JPEG", quality=q, subsampling=2)
        gray = np.asarray(Image.open(buf))[..., 0]
        info["jpeg_quality"] = q
    f = gray.astype(np.float32) / 255.0
    f = f + rng.uniform(-params["brightness"], params["brightness"])
    m = f.mean()
    f = (f - m) * rng.uniform(*params["contrast"]) + m
    f = np.clip(f, 0.0, 1.0) ** rng.uniform(*params["gamma"])
    gray = np.clip(np.round(f * 255.0), 0, 255).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2), info


def _tf_shortcut_map(params, seed):
    rng = np.random.default_rng(seed)  # one stream for the whole run (sequential map -> reproducible)

    def apply(x):
        img = np.clip(np.round(x), 0, 255).astype(np.uint8)
        return shortcut_augment(img, rng, params)[0]

    def tf_fn(x):
        y = tf.numpy_function(apply, [x], tf.uint8)
        y.set_shape(x.shape)
        return tf.cast(y, tf.float32)
    return tf_fn


def make_dataset(config, split=None, df=None, training=False):
    """Batched dataset of (image, label) for one split, or for the rows of a given df."""
    if df is None:
        df = read_split(config, split)
    img_size = tuple(config["training"]["img_size"])
    batch_size = config["training"]["batch_size"]

    ds = tf.data.Dataset.from_tensor_slices(
        (df["path"].values, df["label"].values.astype("float32")))
    if training:
        ds = ds.shuffle(len(df), seed=config["seed"], reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: (load_image(p, img_size), y), num_parallel_calls=AUTOTUNE)
    shortcut = config.get("augmentation", {}).get("shortcut")
    if training and shortcut:
        aug = _tf_shortcut_map(shortcut, config["seed"])
        ds = ds.map(lambda x, y: (aug(x), y))  # image only: the label never reaches the augmentation
    ds = ds.batch(batch_size)
    if training:
        augment = build_augmentation(config)
        # Sequential map: seeded random layers stay reproducible.
        ds = ds.map(lambda x, y: (augment(x, training=True), y))
    return ds.prefetch(AUTOTUNE)


def class_weights(train_df):
    """Balanced class weights from the training labels: {0: w_normal, 1: w_tb}."""
    labels = train_df["label"].values
    classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    return {int(c): float(w) for c, w in zip(classes, weights)}
