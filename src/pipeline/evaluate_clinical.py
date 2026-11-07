"""Goal 4: screening-oriented evaluation from the saved per-image predictions.

    python -m src.pipeline.evaluate_clinical                  # everything (masking reuses cached predictions)
    python -m src.pipeline.evaluate_clinical --skip-masking   # statistics only, no model runs
    python -m src.pipeline.evaluate_clinical --rerun-masking  # force re-prediction of masked test images

Reads artifacts/predictions/{model}_{val,test}.csv (no retraining). Positive class = TB (label 1);
an image is called TB when prob >= threshold (same rule as evaluate.py).

  1. Threshold per model, tuned on VAL: highest threshold with val sensitivity >= 0.95; applied unchanged to TEST.
  2. TEST metrics at 0.5 and at the tuned threshold: sensitivity, specificity, PPV, NPV, F1, ROC-AUC,
     with 95% stratified bootstrap CIs (2,000 resamples within each class, seed 42, percentile method).
     Resamples where PPV/NPV is undefined (0/0) are dropped from that CI and counted.
  3. Calibration on TEST: 10-bin reliability curve and Brier score.
  4. Subgroups on TEST TB images: non-square vs 256x256 (sensitivity, mean/median P(TB)).
  5. Shared errors: which false positives / negatives are common to the models.
  6. Masking (runs the models on TEST): outer border masked at 12% and 20% of each edge, and a
     centre-mask control (central 170x170 hidden, border visible). Fill = the image's own mean grey.
All outputs go to artifacts/clinical/.
"""
import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.utils.config_loader import load_config

N_BOOT = 2000
SEED = 42
TARGET_SENSITIVITY = 0.95
N_BINS = 10
OUT_DIR = "artifacts/clinical"
NAMES = {"vgg": "VGG16", "resnet": "ResNet50", "efficientnet": "EfficientNetB0"}
METRICS = ["sensitivity", "specificity", "ppv", "npv", "f1"]
# variant -> (region to fill, fraction of each edge). centre170 hides the complement of border12.
MASKS = {"border12": ("border", 0.12), "border20": ("border", 0.20), "centre170": ("centre", 0.12)}
MASK_LABELS = {"none": "unmasked", "border12": "border 12% masked", "border20": "border 20% masked",
               "centre170": "centre 170x170 masked (lungs hidden)"}


# ---------------------------------------------------------------- statistics
def stratified_boot_indices(y, n_boot, seed):
    """(n_boot, n) indices resampled with replacement within each class (class sizes fixed)."""
    rng = np.random.default_rng(seed)
    neg, pos = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    return np.hstack([rng.choice(neg, (n_boot, len(neg))), rng.choice(pos, (n_boot, len(pos)))])


def rates(tp, fp, tn, fn):
    tp, fp, tn, fn = (np.asarray(v, dtype=float) for v in (tp, fp, tn, fn))
    with np.errstate(invalid="ignore", divide="ignore"):  # 0/0 -> nan = undefined
        return {"sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp),
                "ppv": tp / (tp + fp), "npv": tn / (tn + fn), "f1": 2 * tp / (2 * tp + fp + fn)}


def counts(y, p, t):
    """Confusion counts; works on 1-D arrays or (n_boot, n) matrices."""
    pred, pos = p >= t, y == 1
    return ((pred & pos).sum(-1), (pred & ~pos).sum(-1), (~pred & ~pos).sum(-1), (~pred & pos).sum(-1))


def ci(point, boot):
    boot = np.asarray(boot, dtype=float)
    ok = boot[np.isfinite(boot)]
    lo, hi = (np.percentile(ok, [2.5, 97.5]) if len(ok) else (np.nan, np.nan))
    return {"value": None if not np.isfinite(point) else float(point),
            "ci_low": float(lo), "ci_high": float(hi), "n_undefined": int(len(boot) - len(ok))}


def metrics_at(y, p, t, idx):
    tp, fp, tn, fn = counts(y, p, t)
    point = rates(tp, fp, tn, fn)
    boot = rates(*counts(y[idx], p[idx], t))
    out = {m: ci(point[m], boot[m]) for m in METRICS}
    out["counts"] = {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}
    out["threshold"] = float(t)
    return out


def auc_ci(y, p, idx):
    return ci(roc_auc_score(y, p), [roc_auc_score(y[i], p[i]) for i in idx])


def brier_ci(y, p, idx):
    return ci(np.mean((p - y) ** 2), np.mean((p[idx] - y[idx]) ** 2, axis=1))


def tune_threshold(y, p, target=TARGET_SENSITIVITY):
    """Highest threshold t such that sensitivity(p >= t) >= target."""
    pos = np.sort(p[y == 1])[::-1]
    k = math.ceil(round(target * len(pos), 9))  # TB cases that must be caught
    t = float(pos[k - 1])
    tp, fp, tn, fn = counts(y, p, t)
    r = rates(tp, fp, tn, fn)
    assert r["sensitivity"] >= target
    return {"threshold": t, "k_required": k, "val_sensitivity": float(r["sensitivity"]),
            "val_specificity": float(r["specificity"]), "val_counts":
            {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}}


def calibration_bins(y, p, n_bins=N_BINS):
    b = np.clip(np.digitize(p, np.linspace(0, 1, n_bins + 1)[1:-1]), 0, n_bins - 1)
    rows = []
    for i in range(n_bins):
        m = b == i
        rows.append({"bin": f"{i / n_bins:.1f}-{(i + 1) / n_bins:.1f}", "n": int(m.sum()),
                     "mean_pred": float(p[m].mean()) if m.any() else np.nan,
                     "frac_tb": float(y[m].mean()) if m.any() else np.nan})
    return rows


# ---------------------------------------------------------------- masking (runs the models)
def mask_fn(img_size, kind, frac):
    import tensorflow as tf
    h, w = img_size
    ph, pw = int(round(frac * h)), int(round(frac * w))
    rows, cols = tf.range(h), tf.range(w)
    inner = ((rows >= ph) & (rows < h - ph))[:, None] & ((cols >= pw) & (cols < w - pw))[None, :]
    region = tf.logical_not(inner) if kind == "border" else inner

    def apply(img):
        return tf.where(region[..., None], tf.reduce_mean(img), img)  # image's own mean grey
    return apply, (h - 2 * ph, w - 2 * pw)


def masked_dataset(paths, config, variant):
    import tensorflow as tf
    from src.components.data_loader import load_image
    img_size = tuple(config["training"]["img_size"])
    ds = tf.data.Dataset.from_tensor_slices(paths)
    if variant == "none":
        ds = ds.map(lambda p: load_image(p, img_size))
    else:
        apply, _ = mask_fn(img_size, *MASKS[variant])
        ds = ds.map(lambda p: apply(load_image(p, img_size)))
    return ds.batch(config["training"]["batch_size"]).prefetch(tf.data.AUTOTUNE)


def run_masking(config, test, models, out_dir, rerun):
    """Returns {variant: {model: probs}}; caches CSVs in out_dir/predictions_masked/."""
    import time
    cache = os.path.join(out_dir, "predictions_masked")
    os.makedirs(cache, exist_ok=True)

    def path_of(v, m):
        return os.path.join(cache, f"{m}_{v}_test.csv")

    todo = [m for m in models if rerun or not all(os.path.exists(path_of(v, m)) for v in MASKS)]

    if todo:
        import keras
        import src.components.model_builder  # noqa: F401  registers CaffePreprocess
        from src.pipeline.train_pipeline import model_path
        for m in todo:
            t0 = time.time()
            model = keras.models.load_model(model_path(config, m))
            # Sanity: unmasked re-prediction of 32 images must match the saved predictions.
            head = test.head(32)
            again = model.predict(masked_dataset(head["path"].values, config, "none"), verbose=0).ravel()
            diff = float(np.abs(again - head[f"p_{m}"].values).max())
            assert diff < 1e-4, f"{m}: re-prediction differs from saved predictions by {diff}"
            for v in MASKS:
                probs = model.predict(masked_dataset(test["path"].values, config, v), verbose=0).ravel()
                test[["path", "label", "width", "height"]].assign(prob=probs).to_csv(path_of(v, m), index=False)
            print(f"[masking] {m}: done in {time.time() - t0:.0f}s (unmasked sanity diff {diff:.1e})")

    out = {}
    for v in MASKS:
        out[v] = {}
        for m in models:
            d = pd.read_csv(path_of(v, m))
            assert (d["path"].values == test["path"].values).all()
            out[v][m] = d["prob"].values
    return out


def save_mask_examples(test, config, out_dir):
    import tensorflow as tf
    from src.components.data_loader import load_image
    img_size = tuple(config["training"]["img_size"])
    picks = pd.concat([test[test.label == 0].head(2), test[test.label == 1].head(2)])
    cols = ["none"] + list(MASKS)
    fig, axes = plt.subplots(len(picks), len(cols), figsize=(2.6 * len(cols), 2.6 * len(picks)))
    for r, (_, row) in enumerate(picks.iterrows()):
        img = load_image(tf.constant(row["path"]), img_size)
        for c, v in enumerate(cols):
            shown = img if v == "none" else mask_fn(img_size, *MASKS[v])[0](img)
            ax = axes[r, c]
            ax.imshow(np.clip(shown.numpy(), 0, 255).astype("uint8"))
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(MASK_LABELS[v].replace(" (lungs hidden)", "\n(lungs hidden)"), fontsize=8)
            if c == 0:
                ax.set_ylabel(f"{'TB' if row.label else 'Normal'}\n{os.path.basename(row['path'])}", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "masked_examples.png"), dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- reporting helpers
def t3(x):
    """Truncate (never round up) to 3 decimals for display."""
    return "n/a" if x is None or not np.isfinite(x) else f"{math.floor(x * 1000) / 1000:.3f}"


def fmt(d):
    s = f"{t3(d['value'])} ({t3(d['ci_low'])}–{t3(d['ci_high'])})"
    return s + (f" [{d['n_undefined']} undef.]" if d.get("n_undefined") else "")


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


# ---------------------------------------------------------------- main
def configure(config):
    """Output and prediction folders from the config (defaults = the goal 4 paths, unchanged).
    For models with the goal 10 internal crop, add the S2 mask: the centre of the *cropped* field."""
    global OUT_DIR, PRED_DIR
    OUT_DIR = config["artifacts"].get("clinical_dir", "artifacts/clinical")
    PRED_DIR = config["artifacts"].get("predictions_dir", "artifacts/predictions")
    frac = config.get("model_options", {}).get("crop_fraction")
    if frac:
        from src.components.model_builder import crop_pixels
        size = config["training"]["img_size"][0]
        c = crop_pixels(size, frac)                       # 27 px removed by the model's crop
        inner = crop_pixels(size - 2 * c, 0.12)           # goal 4's 12% border, applied to the cropped field
        MASKS["centre_in_crop"] = ("centre", (c + inner) / size)   # input pixels 47-176 hidden at 224
        MASK_LABELS["centre_in_crop"] = "centre of the cropped field masked (S2 primary)"
        MASK_LABELS["centre170"] = "centre 170x170 masked (goal 4 version; hides the whole cropped field)"


PRED_DIR = "artifacts/predictions"


def load_predictions(models, split):
    base = None
    for m in models:
        d = pd.read_csv(os.path.join(PRED_DIR, f"{m}_{split}.csv"))
        if base is None:
            base = d[["path", "label", "width", "height"]].copy()
        assert (d["path"].values == base["path"].values).all(), f"{m}_{split}: row order differs"
        base[f"p_{m}"] = d["prob"].values
    base["nonsquare"] = (base["width"] != 256) | (base["height"] != 256)
    return base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-masking", action="store_true")
    parser.add_argument("--rerun-masking", action="store_true")
    args = parser.parse_args()

    config = load_config()
    configure(config)
    models = config["models"]
    os.makedirs(OUT_DIR, exist_ok=True)
    val, test = load_predictions(models, "val"), load_predictions(models, "test")
    y = test["label"].values
    idx = stratified_boot_indices(y, N_BOOT, SEED)  # shared by all models -> comparable CIs
    md = []

    # 1. thresholds on VAL
    thresholds = {m: tune_threshold(val["label"].values, val[f"p_{m}"].values) for m in models}
    with open(os.path.join(OUT_DIR, "thresholds.json"), "w") as f:
        json.dump({"rule": f"highest threshold with VAL sensitivity >= {TARGET_SENSITIVITY}",
                   "decision": "prob >= threshold -> TB", "models": thresholds}, f, indent=2)
    md.append("### Thresholds (chosen on VAL)\n" + md_table(
        ["model", "tuned threshold", "val sensitivity", "val specificity", "val [TN,FP / FN,TP]"],
        [[NAMES[m], f"{thresholds[m]['threshold']:.4g}", t3(thresholds[m]["val_sensitivity"]),
          t3(thresholds[m]["val_specificity"]),
          "[{tn},{fp} / {fn},{tp}]".format(**thresholds[m]["val_counts"])] for m in models]))

    # 2. TEST screening metrics
    results = {}
    for m in models:
        p = test[f"p_{m}"].values
        results[m] = {"roc_auc": auc_ci(y, p, idx), "brier": brier_ci(y, p, idx),
                      "at_0.5": metrics_at(y, p, 0.5, idx),
                      "at_tuned": metrics_at(y, p, thresholds[m]["threshold"], idx)}
    meta = {"split": "test", "n": int(len(y)), "n_normal": int((y == 0).sum()), "n_tb": int(y.sum()),
            "bootstrap": {"n_resamples": N_BOOT, "seed": SEED, "type": "stratified by class",
                          "ci": "95% percentile"}, "positive_class": "TB"}
    with open(os.path.join(OUT_DIR, "metrics_test.json"), "w") as f:
        json.dump({"meta": meta, "models": results}, f, indent=2)

    flat = []
    for m in models:
        for op in ("at_0.5", "at_tuned"):
            r = results[m][op]
            row = {"model": m, "operating_point": op, "threshold": r["threshold"], **r["counts"]}
            for k in METRICS:
                row.update({k: r[k]["value"], f"{k}_ci_low": r[k]["ci_low"],
                            f"{k}_ci_high": r[k]["ci_high"], f"{k}_n_undefined": r[k]["n_undefined"]})
            for k in ("roc_auc", "brier"):
                row.update({k: results[m][k]["value"], f"{k}_ci_low": results[m][k]["ci_low"],
                            f"{k}_ci_high": results[m][k]["ci_high"]})
            flat.append(row)
    pd.DataFrame(flat).to_csv(os.path.join(OUT_DIR, "metrics_test.csv"), index=False)

    for op, title in (("at_tuned", "TEST at the val-tuned threshold (headline)"), ("at_0.5", "TEST at threshold 0.5")):
        md.append(f"### {title}\n" + md_table(
            ["model", "threshold", "sensitivity", "specificity", "PPV", "NPV", "F1", "[TN,FP / FN,TP]"],
            [[NAMES[m], f"{results[m][op]['threshold']:.4g}"] + [fmt(results[m][op][k]) for k in METRICS]
             + ["[{tn},{fp} / {fn},{tp}]".format(**results[m][op]["counts"])] for m in models]))
    md.append("### Threshold-free (TEST)\n" + md_table(
        ["model", "ROC-AUC", "Brier score"],
        [[NAMES[m], fmt(results[m]["roc_auc"]), fmt(results[m]["brier"])] for m in models]))

    # 3. calibration
    cal_rows = []
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect calibration")
    for m in models:
        bins = calibration_bins(y, test[f"p_{m}"].values)
        cal_rows += [{"model": m, **b} for b in bins]
        b = pd.DataFrame(bins).dropna()
        ax1.plot(b["mean_pred"], b["frac_tb"], "o-",
                 label=f"{NAMES[m]} (Brier {t3(results[m]['brier']['value'])})")
        ax2.hist(test[f"p_{m}"], bins=np.linspace(0, 1, N_BINS + 1), histtype="step", lw=1.5, label=NAMES[m])
    ax1.set(xlabel="Mean predicted P(TB)", ylabel="Observed fraction TB", title="Reliability curve (test, 10 bins)")
    ax1.legend(fontsize=8)
    ax2.set(xlabel="Predicted P(TB)", ylabel="Images", title="Prediction histogram (test)", yscale="log")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "calibration.png"), dpi=120)
    plt.close(fig)
    cal = pd.DataFrame(cal_rows)
    cal.to_csv(os.path.join(OUT_DIR, "calibration_bins.csv"), index=False)
    md.append("### Calibration bins (TEST): n / mean predicted / observed TB fraction\n" + md_table(
        ["bin"] + [NAMES[m] for m in models],
        [[b] + [("{n} / {mp} / {fr}".format(n=r.n, mp=t3(r.mean_pred), fr=t3(r.frac_tb)) if r.n else "0")
                for r in (cal[(cal.bin == b) & (cal.model == m)].iloc[0] for m in models)]
         for b in cal["bin"].unique()]))

    # 4. subgroups (TB images only)
    sub_rows = []
    tb = test[test.label == 1]
    for m in models:
        for name, g in (("non-square", tb[tb.nonsquare]), ("256x256", tb[~tb.nonsquare])):
            p = g[f"p_{m}"].values
            g_idx = np.random.default_rng(SEED).choice(len(p), (N_BOOT, len(p)))
            row = {"model": m, "subgroup": name, "n_tb": len(p), "mean_p_tb": float(p.mean()),
                   "median_p_tb": float(np.median(p))}
            for op, t in (("0.5", 0.5), ("tuned", thresholds[m]["threshold"])):
                c = ci((p >= t).mean(), (p[g_idx] >= t).mean(axis=1))
                row.update({f"sens_{op}": c["value"], f"sens_{op}_ci_low": c["ci_low"],
                            f"sens_{op}_ci_high": c["ci_high"], f"missed_{op}": int((p < t).sum())})
            sub_rows.append(row)
    sub = pd.DataFrame(sub_rows)
    sub.to_csv(os.path.join(OUT_DIR, "subgroups.csv"), index=False)
    md.append("### Subgroups (TEST TB images)\n" + md_table(
        ["model", "subgroup", "n", "mean P(TB)", "median P(TB)", "sensitivity @0.5", "sensitivity @tuned"],
        [[NAMES[r.model], r.subgroup, r.n_tb, t3(r.mean_p_tb), t3(r.median_p_tb),
          f"{t3(r['sens_0.5'])} ({t3(r['sens_0.5_ci_low'])}–{t3(r['sens_0.5_ci_high'])}), missed {r['missed_0.5']}",
          f"{t3(r.sens_tuned)} ({t3(r.sens_tuned_ci_low)}–{t3(r.sens_tuned_ci_high)}), missed {r.missed_tuned}"]
         for _, r in sub.iterrows()]))

    # 5. shared errors
    err = test[["path", "label", "width", "height", "nonsquare"] + [f"p_{m}" for m in models]].copy()
    err["file"] = err["path"].map(os.path.basename)
    shared = {}
    for op in ("0.5", "tuned"):
        cols = []
        for m in models:
            t = 0.5 if op == "0.5" else thresholds[m]["threshold"]
            err[f"wrong_{op}_{m}"] = (err[f"p_{m}"] >= t).astype(int) != err["label"]
            cols.append(f"wrong_{op}_{m}")
        err[f"n_models_wrong_{op}"] = err[cols].sum(axis=1)
        for kind, lab in (("false_positive", 0), ("false_negative", 1)):
            e = err[err.label == lab]
            nw = e[f"n_models_wrong_{op}"]
            shared[f"{kind}@{op}"] = {
                "per_model": {m: int(e[f"wrong_{op}_{m}"].sum()) for m in models},
                "union": int((nw >= 1).sum()), "all_3": int((nw == 3).sum()),
                "exactly_2": int((nw == 2).sum()), "exactly_1": int((nw == 1).sum()),
                "all_3_files": sorted(e[nw == 3]["file"].tolist()),
                "all_3_nonsquare": int(e[nw == 3]["nonsquare"].sum()),
            }
    any_wrong = err[(err["n_models_wrong_0.5"] > 0) | (err["n_models_wrong_tuned"] > 0)]
    any_wrong.drop(columns="path").sort_values(["label", "n_models_wrong_tuned", "file"],
                                               ascending=[True, False, True]) \
        .to_csv(os.path.join(OUT_DIR, "shared_errors.csv"), index=False)
    with open(os.path.join(OUT_DIR, "shared_errors.json"), "w") as f:
        json.dump(shared, f, indent=2)
    md.append("### Shared errors (TEST)\n" + md_table(
        ["error @ threshold"] + [NAMES[m] for m in models] + ["union", "all 3", "exactly 2", "exactly 1",
                                                              "all-3 non-square"],
        [[k] + [v["per_model"][m] for m in models] + [v["union"], v["all_3"], v["exactly_2"], v["exactly_1"],
                                                       v["all_3_nonsquare"]] for k, v in shared.items()]))
    for k, v in shared.items():
        md.append(f"- {k}, wrong in all 3 models ({v['all_3']}): " + (", ".join(v["all_3_files"]) or "none"))

    # 6. masking
    if not args.skip_masking:
        masked = run_masking(config, test, models, OUT_DIR, args.rerun_masking)
        save_mask_examples(test, config, OUT_DIR)
        mask_rows, mask_json = [], {}
        for v in ["none"] + list(MASKS):
            mask_json[v] = {}
            for m in models:
                p = test[f"p_{m}"].values if v == "none" else masked[v][m]
                r = {"roc_auc": auc_ci(y, p, idx), "at_0.5": metrics_at(y, p, 0.5, idx),
                     "at_tuned": metrics_at(y, p, thresholds[m]["threshold"], idx),
                     "mean_p_normal": float(p[y == 0].mean()), "mean_p_tb": float(p[y == 1].mean())}
                mask_json[v][m] = r
                mask_rows.append({"variant": v, "model": m, "auc": r["roc_auc"]["value"],
                                  "auc_ci_low": r["roc_auc"]["ci_low"], "auc_ci_high": r["roc_auc"]["ci_high"],
                                  **{f"{k}_{op}": r[f"at_{op}"][k]["value"] for op in ("0.5", "tuned")
                                     for k in ("sensitivity", "specificity")},
                                  "mean_p_normal": r["mean_p_normal"], "mean_p_tb": r["mean_p_tb"]})
        _, centre = mask_fn(tuple(config["training"]["img_size"]), *MASKS["border12"])
        with open(os.path.join(OUT_DIR, "masking.json"), "w") as f:
            json.dump({"fill": "image's own mean grey (mean over all pixels, after 224x224 resize)",
                       "variants": {v: MASK_LABELS[v] for v in ["none"] + list(MASKS)},
                       "visible_centre_border12": centre, "threshold_note": "tuned thresholds from unmasked VAL",
                       "results": mask_json}, f, indent=2)
        pd.DataFrame(mask_rows).to_csv(os.path.join(OUT_DIR, "masking.csv"), index=False)
        md.append("### Masking (TEST; tuned thresholds from unmasked VAL)\n" + md_table(
            ["model", "input", "ROC-AUC (95% CI)", "sens @0.5", "spec @0.5", "sens @tuned", "spec @tuned",
             "mean P(TB) Normal / TB"],
            [[NAMES[m], MASK_LABELS[v], fmt(mask_json[v][m]["roc_auc"])]
             + [t3(mask_json[v][m][f"at_{op}"][k]["value"]) for op in ("0.5", "tuned")
                for k in ("sensitivity", "specificity")]
             + [f"{t3(mask_json[v][m]['mean_p_normal'])} / {t3(mask_json[v][m]['mean_p_tb'])}"]
             for m in models for v in ["none"] + list(MASKS)]))

    md.insert(0, f"Generated by src/pipeline/evaluate_clinical.py. TEST n={meta['n']} "
                 f"({meta['n_normal']} Normal, {meta['n_tb']} TB). 95% CIs: stratified bootstrap, {N_BOOT} "
                 f"resamples, seed {SEED}. Values truncated (not rounded) to 3 decimals.")
    with open(os.path.join(OUT_DIR, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(md) + "\n")
    print("\n\n".join(md))


if __name__ == "__main__":
    main()
