import tensorflow as tf
import numpy as np
import os
import json

from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

from src.utils.config_loader import load_config

config = load_config()

DATA_PATH = config["data_path"]
MODEL_PATH = config["model"]["save_path"]
METRICS_PATH = config["artifacts"]["metrics_path"]

IMG_SIZE = tuple(config["training"]["img_size"])
BATCH_SIZE = config["training"]["batch_size"]

def load_test_data():
    test_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_PATH,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    return test_ds

def evaluate_model():

    print(f"\nEvaluating: {MODEL_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH)
    test_ds = load_test_data()

    y_true, y_pred, y_prob = [], [], []

    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        preds = (probs > 0.5).astype(int)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.flatten())
        y_prob.extend(probs.flatten())

    # Metrics
    report = classification_report(y_true, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    print("\nConfusion Matrix:")
    print(cm)

    print("\nROC-AUC Score:")
    print(roc_auc)

    # Save metrics (no hardcoding)
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)

    results = {
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nMetrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    evaluate_model()