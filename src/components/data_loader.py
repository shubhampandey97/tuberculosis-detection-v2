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
    """Decode a JPEG/PNG as 3-channel RGB and resize to img_size. Returns float32 in 0-255."""
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    img = tf.image.resize(img, img_size)  # plain resize: non-square images are stretched
    return tf.cast(img, tf.float32)


def build_augmentation(config):
    aug, seed = config["augmentation"], config["seed"]
    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(aug["rotation"], seed=seed),
        tf.keras.layers.RandomZoom(aug["zoom"], seed=seed),
    ], name="augmentation")


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
