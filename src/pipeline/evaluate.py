"""Evaluate trained models.

    python -m src.pipeline.evaluate

For every model in config.yaml:
  * per-image predictions for VAL and TEST -> artifacts/predictions/{model}_{split}.csv
  * metrics on TEST only (threshold 0.5)    -> artifacts/metrics/{model}.json + plots
  * model selection uses VAL AUC only; test numbers are reported but never used to pick.
"""
import json
import os

import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

import src.components.model_builder  # noqa: F401  registers CaffePreprocess for loading
from src.components.data_loader import make_dataset, read_split
from src.pipeline.train_pipeline import model_path
from src.utils.config_loader import load_config

CLASS_NAMES = ["Normal", "TB"]
THRESHOLD = 0.5  # fixed for now; tuned on VAL in goal 4


def predict_split(model, config, split):
    df = read_split(config, split)
    probs = model.predict(make_dataset(config, df=df), verbose=0).ravel()
    return df.assign(prob=probs)[["path", "label", "prob", "width", "height"]]


def test_metrics(name, test_pred, out_metrics, out_plots):
    y, p = test_pred["label"].values, test_pred["prob"].values
    pred = (p >= THRESHOLD).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    auc = roc_auc_score(y, p)
    results = {
        "model": name, "split": "test", "n": int(len(y)), "threshold": THRESHOLD,
        "roc_auc": float(auc), "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y, pred, labels=[0, 1], target_names=CLASS_NAMES, output_dict=True, zero_division=0),
    }
    with open(os.path.join(out_metrics, f"{name}.json"), "w") as f:
        json.dump(results, f, indent=4)

    plot_dir = os.path.join(out_plots, name)
    os.makedirs(plot_dir, exist_ok=True)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{name} - test confusion matrix")
    plt.savefig(os.path.join(plot_dir, "confusion_matrix.png"), bbox_inches="tight")
    plt.close()

    fpr, tpr, _ = roc_curve(y, p)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate (sensitivity)")
    plt.title(f"{name} - test ROC")
    plt.legend()
    plt.savefig(os.path.join(plot_dir, "roc_curve.png"), bbox_inches="tight")
    plt.close()
    return results


def main():
    config = load_config()
    art = config["artifacts"]
    for d in (art["metrics_dir"], art["plots_dir"], art["predictions_dir"]):
        os.makedirs(d, exist_ok=True)

    summary = []
    for name in config["models"]:
        path = model_path(config, name)
        if not os.path.exists(path):
            print(f"Skipping {name}: {path} not found")
            continue
        model = keras.models.load_model(path)
        preds = {}
        for split in ("val", "test"):
            preds[split] = predict_split(model, config, split)
            preds[split].to_csv(os.path.join(art["predictions_dir"], f"{name}_{split}.csv"), index=False)
        res = test_metrics(name, preds["test"], art["metrics_dir"], art["plots_dir"])
        val_auc = roc_auc_score(preds["val"]["label"], preds["val"]["prob"])
        summary.append({"model": name, "val_auc": float(val_auc), "test_auc": res["roc_auc"]})
        print(f"{name}: val AUC={val_auc:.4f}  test AUC={res['roc_auc']:.4f}  test CM={res['confusion_matrix']}")

    if summary:
        best = max(summary, key=lambda r: r["val_auc"])  # selection on VAL only
        out = {"selected_by": "val_auc", "best_model": best["model"], "models": summary}
        with open(os.path.join(art["metrics_dir"], "summary.json"), "w") as f:
            json.dump(out, f, indent=4)
        print(f"Best model by val AUC: {best['model']}")


if __name__ == "__main__":
    main()
