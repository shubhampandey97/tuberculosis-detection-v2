"""1-epoch smoke test on a small stratified subset, for each model.

    python -m src.pipeline.smoke_test [--models vgg resnet efficientnet] [--no-determinism]

Writes only to models/smoke/ (never touches real models). Checks per model:
  * CaffePreprocess matches keras preprocess_input (vgg/resnet)
  * predictions are finite and within [0, 1]
  * the saved .keras model reloads and gives the same output
and reports timings used to estimate full-training time.
"""
import argparse
import copy
import json
import os
import time

N_TRAIN, N_VAL, N_CHECK = 200, 100, 32


def smoke_dir_for(config):
    """models/smoke for the default config (unchanged), models/smoke_<tag> for others (e.g. v10_arm1)."""
    tag = config.get("tag", "default")
    return "models/smoke" if tag == "default" else f"models/smoke_{tag}"


def stratified_sample(df, n, seed):
    from sklearn.model_selection import train_test_split
    sample, _ = train_test_split(df, train_size=n, stratify=df["label"], random_state=seed)
    return sample.reset_index(drop=True)


def main():
    from src.utils.config_loader import load_config
    from src.utils import seed as seed_mod

    config = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=config["models"])
    parser.add_argument("--no-determinism", action="store_true")
    parser.add_argument("--epochs", type=int, default=1,
                        help="2+ gives a warm (no tracing overhead) epoch for the full-training estimate")
    args = parser.parse_args()
    smoke_dir = smoke_dir_for(config)

    seed_mod.USE_OP_DETERMINISM = not args.no_determinism
    seed_mod.set_global_seed(config["seed"])

    import keras
    import numpy as np
    from src.components.data_loader import make_dataset, read_split
    from src.components.model_builder import CaffePreprocess, NEEDS_CAFFE
    from src.pipeline.train_pipeline import model_path, train_model

    cfg = copy.deepcopy(config)
    cfg["models_dir"] = smoke_dir
    cfg["artifacts"]["history_dir"] = os.path.join(smoke_dir, "history")
    if cfg["training"].get("finetune"):  # Arm 2 smoke starts from the Arm 1 *smoke* model
        cfg["training"]["init_from"] = "models/smoke_" + os.path.basename(
            cfg["training"]["init_from"]).replace("tensorflow_", "")
    n_train_full = len(read_split(cfg, "train"))
    n_val_full = len(read_split(cfg, "val"))

    train_df = stratified_sample(read_split(cfg, "train"), N_TRAIN, cfg["seed"])
    val_df = stratified_sample(read_split(cfg, "val"), N_VAL, cfg["seed"])
    check_df = read_split(cfg, "test").head(N_CHECK)
    check_x = np.concatenate([x for x, _ in make_dataset(cfg, df=check_df)])
    print(f"Smoke data: train={len(train_df)} (TB {int(train_df.label.sum())}), "
          f"val={len(val_df)} (TB {int(val_df.label.sum())}), check={len(check_df)}; "
          f"input range [{check_x.min():.1f}, {check_x.max():.1f}]; "
          f"op determinism={'on' if seed_mod.USE_OP_DETERMINISM else 'off'}")

    report = {}
    for name in args.models:
        print(f"\n=== Smoke: {name} ===")
        r = {}
        if name in NEEDS_CAFFE:
            app = {"vgg": keras.applications.vgg16, "resnet": keras.applications.resnet50}[name]
            r["preprocess_max_diff"] = float(np.abs(
                CaffePreprocess()(check_x).numpy() - app.preprocess_input(check_x.copy())).max())

        t0 = time.time()
        model, hist = train_model(name, cfg, train_df=train_df, val_df=val_df, epochs=args.epochs)
        r["fit_total_s"] = round(time.time() - t0, 1)
        r["epoch_seconds"] = hist["epoch_seconds"]
        r["val_auc_last_epoch"] = round(hist["val_auc"][-1], 4)
        r["trainable_params"] = hist["trainable_params"]

        # Warm inference throughput over the smoke training images (forward pass only).
        train_eval = make_dataset(cfg, df=train_df)
        model.predict(train_eval.take(1), verbose=0)
        t0 = time.time()
        model.predict(train_eval, verbose=0)
        fwd = (time.time() - t0) / len(train_df)
        r["s_per_image"] = round(fwd, 4)
        # Full-epoch estimate from the last (warm) epoch: training part scaled to the full train split,
        # validation part (forward only) scaled to the full val split.
        last = hist["epoch_seconds"][-1]
        train_per_img = max(last - fwd * len(val_df), 0) / len(train_df)
        r["est_full_epoch_min"] = round((train_per_img * n_train_full + fwd * n_val_full) / 60, 2)
        r["estimate_from_warm_epoch"] = len(hist["epoch_seconds"]) > 1

        probs = model.predict(check_x, verbose=0).ravel()
        r["pred_min"], r["pred_max"] = float(probs.min()), float(probs.max())
        r["pred_in_0_1"] = bool(np.all(np.isfinite(probs)) and probs.min() >= 0 and probs.max() <= 1)

        reloaded = keras.models.load_model(model_path(cfg, name))
        r["reload_max_diff"] = float(np.abs(reloaded.predict(check_x, verbose=0).ravel() - probs).max())
        r["reload_same"] = bool(np.allclose(reloaded.predict(check_x, verbose=0).ravel(), probs, atol=1e-5))
        report[name] = r
        print(json.dumps(r, indent=2))

    tag = "nodet" if args.no_determinism else "det"
    with open(os.path.join(smoke_dir, f"smoke_report_{tag}.json"), "w") as f:
        json.dump(report, f, indent=2)
    failed = [n for n, r in report.items() if not (r["pred_in_0_1"] and r["reload_same"]
              and r.get("preprocess_max_diff", 0) < 1e-3)]
    print("\nSMOKE TEST", "FAILED: " + ", ".join(failed) if failed else "PASSED")


if __name__ == "__main__":
    main()
