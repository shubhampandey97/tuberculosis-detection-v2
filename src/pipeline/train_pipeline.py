"""Train the TB classifiers from data/splits.csv.

    python -m src.pipeline.train_pipeline                    # all models in config.yaml
    python -m src.pipeline.train_pipeline --models vgg       # a subset
Smoke runs (tiny subset, 1 epoch, separate output folder) go through src.pipeline.smoke_test.
"""
import argparse
import json
import os
import time

from src.utils.config_loader import load_config
from src.utils.seed import set_global_seed


def model_path(config, model_name):
    return os.path.join(config["models_dir"], f"{model_name}_model.keras")


def train_model(model_name, config, train_df=None, val_df=None, epochs=None):
    """Train one model and save the best checkpoint (by val AUC). Returns (model, history dict)."""
    import keras
    from src.components.data_loader import class_weights, make_dataset, read_split
    from src.components.model_builder import build_model

    t = config["training"]
    train_df = read_split(config, "train") if train_df is None else train_df
    val_df = read_split(config, "val") if val_df is None else val_df
    epochs = t["epochs"] if epochs is None else epochs

    train_ds = make_dataset(config, df=train_df, training=True)
    val_ds = make_dataset(config, df=val_df)
    weights = class_weights(train_df)
    print(f"[{model_name}] train={len(train_df)} val={len(val_df)} class_weights={weights}")

    save_path = model_path(config, model_name)  # one file per model (fixes audit §5.2)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=t["patience"],
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(save_path, monitor="val_auc", mode="max",
                                        save_best_only=True, verbose=1),
    ]

    model = build_model(model_name, img_size=t["img_size"], learning_rate=t["learning_rate"])
    start = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                        class_weight=weights, callbacks=callbacks, verbose=2)
    seconds = time.time() - start

    record = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    record.update(model=model_name, saved_to=save_path, class_weights=weights,
                  n_train=len(train_df), n_val=len(val_df), epochs_run=len(history.epoch),
                  train_seconds=round(seconds, 1))
    os.makedirs(config["artifacts"]["history_dir"], exist_ok=True)
    with open(os.path.join(config["artifacts"]["history_dir"], f"{model_name}.json"), "w") as f:
        json.dump(record, f, indent=2)
    return model, record


def main():
    config = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=config["models"])
    args = parser.parse_args()

    set_global_seed(config["seed"])
    for name in args.models:
        print(f"\n=== Training {name} ===")
        _, record = train_model(name, config)
        print(f"[{name}] best val_auc={max(record['val_auc']):.4f} "
              f"epochs={record['epochs_run']} time={record['train_seconds']}s -> {record['saved_to']}")


if __name__ == "__main__":
    main()
