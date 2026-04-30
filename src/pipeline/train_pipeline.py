import os

# Enable XLA (optional)
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2"

import tensorflow as tf

from src.components.data_loader import load_datasets, apply_augmentation
from src.components.model_builder import build_model

DATA_PATH = "data/Dataset of Tuberculosis Chest X-rays Images"

def train_model(model_name):

    train_ds, val_ds = load_datasets(DATA_PATH)
    train_ds = apply_augmentation(train_ds)

    model = build_model(model_name)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10
    )

    os.makedirs("models", exist_ok=True)
    model.save(f"models/tensorflow/{model_name}_model.h5")

    return history


if __name__ == "__main__":
    train_model("efficientnet")
    # train_model("vgg")
    # train_model("resnet")