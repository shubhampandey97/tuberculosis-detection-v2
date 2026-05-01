import tensorflow as tf
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
from src.utils.config_loader import load_config

# ---------------- CONFIG ----------------
config = load_config()

DATA_PATH = config["data_path"]
MODEL_PATH = config["model"]["save_path"]
METRICS_PATH = config["artifacts"]["metrics_path"]
PLOTS_DIR = config["artifacts"]["plots_dir"]

IMG_SIZE = tuple(config["training"]["img_size"])
BATCH_SIZE = config["training"]["batch_size"]

# ---------------- DATA ----------------
def load_test_data():
    return tf.keras.utils.image_dataset_from_directory(
        DATA_PATH,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

# ---------------- EVALUATION ----------------
def evaluate_model(model_path):

    print(f"\nEvaluating: {model_path}")

    model = tf.keras.models.load_model(model_path)
    test_ds = load_test_data()

    y_true, y_pred, y_prob = [], [], []

    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        preds = (probs > 0.5).astype(int)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.flatten())
        y_prob.extend(probs.flatten())

    report_dict = classification_report(y_true, y_pred, output_dict=True)
    report_text = classification_report(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)

    print(report_text)
    print(cm)
    print("ROC-AUC:", roc_auc)

    # 🔥 Save per-model artifacts
    model_name = os.path.basename(model_path).split(".")[0]

    model_metrics_path = f"artifacts/metrics/{model_name}.json"
    model_plot_dir = f"artifacts/plots/{model_name}"

    os.makedirs(os.path.dirname(model_metrics_path), exist_ok=True)
    os.makedirs(model_plot_dir, exist_ok=True)

    results = {
        "model": model_name,
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict
    }

    with open(model_metrics_path, "w") as f:
        json.dump(results, f, indent=4)

    # Confusion Matrix Plot
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.savefig(f"{model_plot_dir}/confusion_matrix.png")
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.legend()
    plt.savefig(f"{model_plot_dir}/roc_curve.png")
    plt.close()

    print(f"✅ Saved artifacts for {model_name}")

# ---------------- MAIN ----------------
if __name__ == "__main__":
    model_paths = [
        "models/tensorflow/resnet_model.h5",
        "models/tensorflow/vgg_model.h5",
        "models/tensorflow/efficientnet_model.h5"
    ]

    for path in model_paths:
        if os.path.exists(path):
            evaluate_model(path)
        else:
            print(f"⚠️ Skipping missing model: {path}")