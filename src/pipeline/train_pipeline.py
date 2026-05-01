import os

# Enable XLA (optional)
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2"

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf

from src.components.data_loader import load_datasets, apply_augmentation
from src.components.model_builder import build_model

DATA_PATH = "data/Dataset of Tuberculosis Chest X-rays Images"

def train_model(model_name):

    train_ds, val_ds = load_datasets(DATA_PATH)
    train_ds = apply_augmentation(train_ds)

    # Extract labels from dataset
    labels = np.concatenate([y.numpy() for x, y in train_ds], axis=0)

    # Compute class weights
    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )

    class_weights = dict(enumerate(class_weights_array))

    print("Class Weights:", class_weights)

    model = build_model(model_name)

    # history = model.fit(
    #     train_ds,
    #     validation_data=val_ds,
    #     epochs=10
    # )

    # Handle class imbalance
    # class_weights = {
    #     0: 514/87,   # Normal class (minority)
    #     1: 1.0       # TB class (majority)
    # }

    class_weights = {
    0: 3.0,
    1: 1.0
    }

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        class_weight=class_weights
    )

    os.makedirs("models/tensorflow", exist_ok=True)
    model.save(f"models/tensorflow/{model_name}_model.h5")

    return history


if __name__ == "__main__":
    train_model("efficientnet")
    # train_model("vgg")
    # train_model("resnet")