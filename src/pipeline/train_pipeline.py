import os
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from src.components.data_loader import load_datasets, apply_augmentation
from src.components.model_builder import build_model
from src.utils.config_loader import load_config

# Load config
config = load_config()

DATA_PATH = config["data_path"]
IMG_SIZE = tuple(config["training"]["img_size"])
EPOCHS = config["training"]["epochs"]
MODEL_PATH = config["model"]["save_path"]

def train_model(model_name):

    train_ds, val_ds = load_datasets(DATA_PATH)
    train_ds = apply_augmentation(train_ds)

    # 🔥 Compute class weights dynamically
    labels = np.concatenate([y.numpy() for x, y in train_ds], axis=0)

    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )

    class_weights = dict(enumerate(class_weights_array))
    print("Class Weights:", class_weights)

    model = build_model(model_name)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weights
    )

    # Save model (no hardcoding)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)

    return history


# if __name__ == "__main__":
#     train_model(config["model"]["name"])

if __name__ == "__main__":

    models = ["vgg", "resnet", "efficientnet"]

    for m in models:
        print(f"\n🚀 Training {m.upper()}...\n")

        config["model"]["name"] = m
        config["model"]["save_path"] = f"models/tensorflow/{m}_model.h5"

        train_model(m)