"""Goal 5: Grad-CAM attention analysis on the TEST split (ResNet50 primary, VGG16).

    python -m src.pipeline.gradcam_analysis [--sanity-only]

Steps
  1. Sanity check on 32 test images: logits/probabilities match the saved predictions, no NaN,
     heatmaps normalise to 1, and a randomised-head check (final Dense weights replaced by random
     ones; the heatmaps should change -> low correlation with the original maps).
  2. For every test image: TB-evidence map (+logit, primary) and predicted-class map (secondary,
     decision at the val-tuned threshold). Fraction of Grad-CAM energy in the outer 12% border
     vs the centre, plus the top / bottom / left / right strips. ResNet is also measured at
     conv4_block6_out (14x14) to check the 7x7 resolution caveat.
  3. Comparisons: Normal vs TB, correct vs wrong (Mann-Whitney U, two-sided).
  4. Galleries (predicted-class maps): shared misses + others (23), 10 correct Normal, 10 correct TB.
Outputs: artifacts/gradcam/
"""
import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy.stats import mannwhitneyu

from src.utils.config_loader import load_config

OUT_DIR = "artifacts/gradcam"
SEED = 42
BORDER_FRAC = 0.12
BATCH = 16
MODELS = ["resnet", "vgg"]  # primary first
NAMES = {"resnet": "ResNet50", "vgg": "VGG16"}
LAYERS = {"resnet": ["conv5_block3_out", "conv4_block6_out"], "vgg": ["block5_conv3"]}
PRIMARY_LAYER = {"resnet": "conv5_block3_out", "vgg": "block5_conv3"}
SHARED_MISSES = ["TB.1141.jpg", "TB.1892.jpg", "TB.2145.jpg", "TB.2324.jpg", "TB.321.jpg", "TB.441.jpg",
                 "TB.244.jpg"]
SHARED_FP = ["others (23).jpg"]
REGIONS = ["border", "top", "bottom", "left", "right"]


# ---------------------------------------------------------------- geometry
def region_masks(size, frac=BORDER_FRAC):
    h, w = size
    p = int(round(frac * h))
    rows, cols = np.arange(h)[:, None], np.arange(w)[None, :]
    top, bottom = np.broadcast_to(rows < p, (h, w)), np.broadcast_to(rows >= h - p, (h, w))
    mid = (rows >= p) & (rows < h - p)
    left, right = mid & (cols < p), mid & (cols >= w - p)  # corners belong to top/bottom
    return {"top": top, "bottom": bottom, "left": left, "right": right,
            "border": top | bottom | left | right}, p


def fractions(hm224, masks):
    """Energy fractions per region for (B, H, W) heatmaps; NaN for empty maps."""
    total = hm224.reshape(len(hm224), -1).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {r: np.where(total > 0, (hm224 * m).reshape(len(hm224), -1).sum(axis=1) / total, np.nan)
                for r, m in masks.items()}


# ---------------------------------------------------------------- data
def load_images(paths, img_size):
    import tensorflow as tf
    from src.components.data_loader import load_image
    return np.stack([load_image(tf.constant(p), img_size).numpy() for p in paths])


def batches(n, size=BATCH):
    for i in range(0, n, size):
        yield slice(i, min(i + size, n))


# ---------------------------------------------------------------- sanity check
def sanity(model, name, test, thr, img_size, masks):
    from src.components.gradcam import gradcam_signed, to_heatmap, upsample
    head = test.head(32)
    x = load_images(head["path"].values, img_size)
    layer = PRIMARY_LAYER[name]
    cams, z, prob = gradcam_signed(x, model, [layer])
    sig = 1 / (1 + np.exp(-z))
    tb, empty = to_heatmap(cams[layer])
    fr = fractions(upsample(tb, img_size), masks)
    ok = ~empty
    strips = fr["top"] + fr["bottom"] + fr["left"] + fr["right"]
    geometry_ok = bool(np.all((fr["border"][ok] >= 0) & (fr["border"][ok] <= 1))
                       and np.allclose(strips[ok], fr["border"][ok]))
    rng = np.random.default_rng(SEED)
    W = model.layers[-1].kernel.numpy()
    W_rand = rng.normal(0, W.std(), W.shape).astype(np.float32)
    cams_r, _, _ = gradcam_signed(x, model, [layer], head_weights=(W_rand, np.zeros(1, np.float32)))
    tb_r, empty_r = to_heatmap(cams_r[layer])
    corr = [np.corrcoef(a.ravel(), b.ravel())[0, 1] for a, b, e1, e2 in zip(tb, tb_r, empty, empty_r)
            if not (e1 or e2) and a.std() > 0 and b.std() > 0]
    res = {
        "model": name, "layer": layer, "n": len(head),
        "max_abs_prob_diff_vs_saved": float(np.abs(prob - head["prob"].values).max()),
        "max_abs_sigmoid_logit_vs_output": float(np.abs(sig - prob).max()),
        "any_nan": bool(np.isnan(tb).any() or np.isnan(z).any()),
        "empty_tb_maps": int(empty.sum()),
        "heatmap_range_ok": bool(tb.min() >= 0 and tb.max() <= 1),
        "region_fractions_ok": geometry_ok,
        "random_head": {"n_pairs": len(corr), "mean_pearson": float(np.mean(corr)),
                        "median_pearson": float(np.median(corr)), "min": float(np.min(corr)),
                        "max": float(np.max(corr)), "empty_random_maps": int(empty_r.sum())},
    }
    assert res["max_abs_prob_diff_vs_saved"] < 1e-4 and res["max_abs_sigmoid_logit_vs_output"] < 1e-4
    assert not res["any_nan"] and res["heatmap_range_ok"] and res["region_fractions_ok"]
    return res


# ---------------------------------------------------------------- full pass
def analyse_model(model, name, test, thr, img_size, masks):
    from src.components.gradcam import gradcam_signed, to_heatmap, upsample
    store = {}
    pred_tb = (test["prob"].values >= thr)
    sign = np.where(pred_tb, 1.0, -1.0)
    for sl in batches(len(test)):
        x = load_images(test["path"].values[sl], img_size)
        cams, z, prob = gradcam_signed(x, model, LAYERS[name])
        for layer in LAYERS[name]:
            for kind, s in (("tb", 1.0), ("pred", sign[sl])):
                hm, empty = to_heatmap(cams[layer], s)
                store.setdefault(f"{layer}__{kind}", []).append(hm)
                store.setdefault(f"{layer}__{kind}__empty", []).append(empty)
        store.setdefault("logit", []).append(z)
        store.setdefault("prob", []).append(prob)
    store = {k: np.concatenate(v) for k, v in store.items()}
    assert np.abs(store["prob"] - test["prob"].values).max() < 1e-4

    df = test[["path", "label", "width", "height", "prob"]].copy()
    df["file"] = df["path"].map(os.path.basename)
    df["pred"] = pred_tb.astype(int)
    df["correct"] = df["pred"] == df["label"]
    df["logit"] = store["logit"]
    for layer in LAYERS[name]:
        for kind in ("tb", "pred"):
            hm224 = upsample(store[f"{layer}__{kind}"], img_size)
            for r, v in fractions(hm224, masks).items():
                df[f"{layer}__{kind}__{r}"] = v
            df[f"{layer}__{kind}__empty"] = store[f"{layer}__{kind}__empty"]
    np.savez_compressed(os.path.join(OUT_DIR, f"heatmaps_{name}.npz"),
                        paths=test["path"].values, **{k: v for k, v in store.items() if "__" in k})
    return df


def describe(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        return {"n": 0}
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return {"n": int(len(v)), "mean": float(v.mean()), "median": float(med), "q1": float(q1), "q3": float(q3)}


def compare(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return None
    return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)


def summarise(dfs, uniform):
    out = {"uniform_reference": uniform, "groups": {}}
    for name, df in dfs.items():
        for layer in LAYERS[name]:
            for kind in ("tb", "pred"):
                key = f"{name}/{layer}/{kind}"

                def col(r, layer=layer, kind=kind):
                    return f"{layer}__{kind}__{r}"

                groups = {"all": df, "Normal": df[df.label == 0], "TB": df[df.label == 1],
                          "correct": df[df.correct], "wrong": df[~df.correct],
                          "TB correct": df[(df.label == 1) & df.correct],
                          "TB missed (FN)": df[(df.label == 1) & ~df.correct],
                          "Normal false positive": df[(df.label == 0) & ~df.correct]}
                g = {gname: {r: describe(gdf[col(r)]) for r in REGIONS} | {"n_empty": int(gdf[col("empty")].sum())}
                     for gname, gdf in groups.items()}
                g["p_normal_vs_tb_border"] = compare(groups["Normal"][col("border")], groups["TB"][col("border")])
                g["p_correct_vs_wrong_border"] = compare(groups["correct"][col("border")], groups["wrong"][col("border")])
                g["p_tb_correct_vs_missed_border"] = compare(groups["TB correct"][col("border")],
                                                             groups["TB missed (FN)"][col("border")])
                out["groups"][key] = g
    return out


# ---------------------------------------------------------------- figures
def boxplot(dfs, path):
    fig, axes = plt.subplots(1, len(dfs), figsize=(6 * len(dfs), 4.5), squeeze=False)
    for ax, (name, df) in zip(axes[0], dfs.items()):
        col = f"{PRIMARY_LAYER[name]}__tb__border"
        groups = [("Normal\ncorrect", (df.label == 0) & df.correct), ("Normal\nFP", (df.label == 0) & ~df.correct),
                  ("TB\ncorrect", (df.label == 1) & df.correct), ("TB\nmissed", (df.label == 1) & ~df.correct)]
        data = [df.loc[m, col].dropna().values for _, m in groups]
        ax.boxplot([d if len(d) else [np.nan] for d in data], showfliers=True)
        ax.set_xticks(range(1, len(groups) + 1))
        ax.set_xticklabels([f"{g}\n(n={len(d)})" for (g, _), d in zip(groups, data)], fontsize=8)
        ax.axhline(UNIFORM["border"], ls="--", c="grey", lw=0.8, label="uniform (area share)")
        ax.set(title=f"{NAMES[name]} ({PRIMARY_LAYER[name]}), TB-evidence map",
               ylabel="Grad-CAM energy in outer 12% border", ylim=(0, 1))
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def gallery(rows, dfs, heat, img_size, title, path, border_px):
    from src.components.gradcam import overlay
    imgs = load_images(rows["path"].values, img_size)
    fig, axes = plt.subplots(len(rows), 3, figsize=(9, 3.1 * len(rows)), squeeze=False)
    for i, (_, r) in enumerate(rows.iterrows()):
        axes[i, 0].imshow(imgs[i].astype(np.uint8))
        axes[i, 0].set_title(f"{r['file']} ({'TB' if r['label'] else 'Normal'}, {r['width']}x{r['height']})", fontsize=8)
        for j, name in enumerate(MODELS, start=1):
            d = dfs[name].set_index("path").loc[r["path"]]
            hm = heat[name][r["path"]]
            ax = axes[i, j]
            ax.imshow(overlay(imgs[i], hm))
            ax.add_patch(Rectangle((border_px, border_px), img_size[1] - 2 * border_px, img_size[0] - 2 * border_px,
                                   fill=False, ls="--", lw=0.8, ec="white"))
            verdict = "TB" if d["pred"] else "Normal"
            ok = "correct" if d["correct"] else "WRONG"
            ax.set_title(f"{NAMES[name]}: P(TB)={d['prob']:.3f} -> {verdict} ({ok})\n"
                         f"border energy {d[PRIMARY_LAYER[name] + '__pred__border']:.2f}", fontsize=8)
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig_h = 3.1 * len(rows)
    fig.suptitle(title + "\n(predicted-class Grad-CAM; dashed box = inner edge of the 12% border)",
                 fontsize=10, y=1 - 0.15 / fig_h)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.75 / fig_h))
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------- main
UNIFORM = {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity-only", action="store_true")
    parser.add_argument("--galleries-only", action="store_true",
                        help="redraw galleries from saved attention CSVs and heatmaps (no model runs)")
    args = parser.parse_args()
    if args.galleries_only:
        config = load_config()
        img_size = tuple(config["training"]["img_size"])
        _, border_px = region_masks(img_size)
        dfs = {}
        for name in MODELS:
            df = pd.read_csv(os.path.join(OUT_DIR, f"attention_{name}.csv"))
            df.insert(0, "path", pd.read_csv(f"artifacts/predictions/{name}_test.csv")["path"].values)
            dfs[name] = df
        make_galleries(dfs, img_size, border_px)
        return

    import keras
    import src.components.model_builder  # noqa: F401  registers CaffePreprocess before loading
    from src.pipeline.train_pipeline import model_path

    config = load_config()
    img_size = tuple(config["training"]["img_size"])
    os.makedirs(OUT_DIR, exist_ok=True)
    masks, border_px = region_masks(img_size)
    UNIFORM.update({r: float(m.mean()) for r, m in masks.items()})
    thresholds = json.load(open("artifacts/clinical/thresholds.json"))["models"]

    models, tests, sanity_res = {}, {}, {}
    for name in MODELS:
        models[name] = keras.models.load_model(model_path(config, name))
        tests[name] = pd.read_csv(f"artifacts/predictions/{name}_test.csv")
        sanity_res[name] = sanity(models[name], name, tests[name], thresholds[name]["threshold"], img_size, masks)
        print(json.dumps(sanity_res[name], indent=2))
    with open(os.path.join(OUT_DIR, "sanity.json"), "w") as f:
        json.dump(sanity_res, f, indent=2)
    if args.sanity_only:
        return

    dfs = {}
    for name in MODELS:
        t0 = time.time()
        dfs[name] = analyse_model(models[name], name, tests[name], thresholds[name]["threshold"], img_size, masks)
        dfs[name].drop(columns="path").to_csv(os.path.join(OUT_DIR, f"attention_{name}.csv"), index=False)
        print(f"[{name}] Grad-CAM on {len(dfs[name])} test images in {time.time() - t0:.0f}s")

    summary = summarise(dfs, UNIFORM)
    summary["thresholds"] = {n: thresholds[n]["threshold"] for n in MODELS}
    summary["layers"] = LAYERS
    with open(os.path.join(OUT_DIR, "attention_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    boxplot(dfs, os.path.join(OUT_DIR, "attention_boxplot.png"))

    base, both_ok = make_galleries(dfs, img_size, border_px)

    # what the shared misses have in common: simple image statistics vs reference groups
    from PIL import Image
    stats = base.copy()
    stats["kb"] = stats["path"].map(lambda p: os.path.getsize(p) / 1024)
    arr = [np.asarray(Image.open(p).convert("L"), dtype=np.float32) for p in stats["path"]]
    stats["brightness"] = [a.mean() for a in arr]
    stats["contrast"] = [a.std() for a in arr]
    for name in MODELS:
        stats[f"p_{name}"] = dfs[name]["prob"].values
        stats[f"border_tb_{name}"] = dfs[name][f"{PRIMARY_LAYER[name]}__tb__border"].values
    stats["group"] = np.select(
        [stats.file.isin(SHARED_MISSES), stats.file.isin(SHARED_FP), both_ok & (stats.label == 1),
         both_ok & (stats.label == 0)], ["shared miss", "shared FP", "TB correct (both)", "Normal correct (both)"],
        "other")
    stats.drop(columns="path").to_csv(os.path.join(OUT_DIR, "image_stats.csv"), index=False)
    cols = ["kb", "brightness", "contrast", "p_resnet", "p_vgg", "border_tb_resnet", "border_tb_vgg"]
    grp = stats[stats.group != "other"].groupby("group")[cols].median().round(3)
    grp["n"] = stats[stats.group != "other"].groupby("group").size()
    grp.to_csv(os.path.join(OUT_DIR, "shared_misses_vs_groups.csv"))
    print(grp.to_string())
    print(stats[stats.group.isin(["shared miss", "shared FP"])][["file", "width", "height"] + cols].round(3).to_string())


def make_galleries(dfs, img_size, border_px):
    """Predicted-class maps at the primary layer. Returns (base table, both-correct mask)."""
    heat = {}
    for name in MODELS:
        z = np.load(os.path.join(OUT_DIR, f"heatmaps_{name}.npz"), allow_pickle=True)
        heat[name] = dict(zip(z["paths"], z[f"{PRIMARY_LAYER[name]}__pred"]))
    base = dfs["resnet"][["path", "file", "label", "width", "height"]].copy()
    both_ok = dfs["resnet"]["correct"].values & dfs["vgg"]["correct"].values
    rng = np.random.default_rng(SEED)
    def pick(mask):
        return base[mask].iloc[np.sort(rng.choice(int(mask.sum()), 10, replace=False))]

    errors = base[base.file.isin(SHARED_MISSES + SHARED_FP)]
    errors = errors.set_index("file").loc[SHARED_MISSES + SHARED_FP].reset_index()
    gallery(errors, dfs, heat, img_size, "Shared errors: 7 TB missed by all 3 models + the shared false positive",
            os.path.join(OUT_DIR, "gallery_errors.png"), border_px)
    gallery(pick(both_ok & (base.label == 0).values), dfs, heat, img_size,
            "10 random Normal images correct for both models (seed 42)",
            os.path.join(OUT_DIR, "gallery_correct_normal.png"), border_px)
    gallery(pick(both_ok & (base.label == 1).values), dfs, heat, img_size,
            "10 random TB images correct for both models (seed 42)",
            os.path.join(OUT_DIR, "gallery_correct_tb.png"), border_px)
    return base, both_ok


if __name__ == "__main__":
    main()
