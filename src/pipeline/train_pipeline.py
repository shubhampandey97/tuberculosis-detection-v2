"""Train the TB classifiers from data/splits.csv.

    python -m src.pipeline.train_pipeline                    # all models in config.yaml
    python -m src.pipeline.train_pipeline --models vgg       # a subset
Smoke runs (tiny subset, 1 epoch, separate output folder) go through src.pipeline.smoke_test.
"""
import argparse
import json
import os
import time

import numpy as np

from src.utils.config_loader import load_config
from src.utils.seed import set_global_seed


def model_path(config, model_name):
    return os.path.join(config["models_dir"], f"{model_name}_model.keras")


def make_model(model_name, config):
    """A new model (default / goal 10 Arm 1) or, for Arm 2, the Arm 1 model with its top layers unfrozen."""
    import keras
    from src.components.model_builder import build_model, set_finetune
    t, opts = config["training"], config.get("model_options", {})
    if t.get("finetune"):
        init = os.path.join(t["init_from"], f"{model_name}_model.keras")
        if not os.path.exists(init):
            raise FileNotFoundError(f"Arm 2 needs the Arm 1 model first: {init}")
        model = keras.models.load_model(init)
        return set_finetune(model, t["finetune_from"][model_name], t["learning_rate"])
    return build_model(model_name, img_size=t["img_size"], learning_rate=t["learning_rate"],
                       crop_fraction=opts.get("crop_fraction"), standardize=opts.get("standardize", False))


def epoch_timer():
    """Keras callback recording wall-clock seconds per epoch (train + validation) in `.seconds`."""
    import keras

    class EpochTimer(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.seconds = []

        def on_epoch_begin(self, epoch, logs=None):
            self._t0 = time.time()

        def on_epoch_end(self, epoch, logs=None):
            self.seconds.append(round(time.time() - self._t0, 1))
    return EpochTimer()


def train_model(model_name, config, train_df=None, val_df=None, epochs=None):
    """Train one model and save the best checkpoint (by val AUC). Returns (model, history dict)."""
    import keras
    from src.components.data_loader import class_weights, make_dataset, read_split

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

    epoch_times = epoch_timer()
    callbacks.append(epoch_times)
    model = make_model(model_name, config)
    start = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                        class_weight=weights, callbacks=callbacks, verbose=2)
    seconds = time.time() - start

    record = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    record.update(model=model_name, saved_to=save_path, class_weights=weights,
                  n_train=len(train_df), n_val=len(val_df), epochs_run=len(history.epoch),
                  train_seconds=round(seconds, 1), epoch_seconds=epoch_times.seconds,
                  trainable_params=int(sum(np.prod(w.shape) for w in model.trainable_weights)),
                  tag=config.get("tag", "default"))
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
