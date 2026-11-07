"""Goal 10: judge the pre-registered criteria S1-S4 (docs/audit.md, goal 10 pre-registration, 15:40).

    python -m src.pipeline.goal10_compare [--stage s4a|compare]

Reads outputs that already exist (no retraining, no new thresholds):
  goal 9 reference   artifacts/external/predictions/{m}_png.csv, artifacts/external/gradcam/attention_external.csv
  goal 10 arms       artifacts/v10_arm{1,2}/{clinical,external}/...
s4a  shortcut test on what the backbone sees (crop + standardisation) for all 2,982 Kaggle images
s4a_posthoc  POST HOC: which features the S4a tree / LR use
compare  S1 (paired dAUC), S2 (masking.json), S3 (lung energy, paired Wilcoxon), S4b, secondary specificity
Outputs: artifacts/v10_comparison/ (aggregate numbers only; no per-image NLM data).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from src.pipeline import evaluate_clinical as ec
from src.utils.config_loader import load_config

OUT = "artifacts/v10_comparison"
ARMS = {"arm1": "src/config/config_v10_arm1.yaml", "arm2": "src/config/config_v10_arm2.yaml"}
MODELS = ["resnet", "efficientnet"]
SITES = ["shenzhen", "montgomery"]
PRIMARY = ("arm2", "resnet")
G9_PRED = "artifacts/external/predictions"
G9_CAM = "artifacts/external/gradcam/attention_external.csv"


# ---------------------------------------------------------------- S4a
INTENSITY = ["dark_frac", "bright_frac", "p5", "p50", "p95", "skew", "kurtosis"]
ALL_FEATURES = ["width", "height", "aspect"] + INTENSITY


def processed_features(splits, size, crop, std):
    """S4a features of what the backbone sees (crop + standardisation), on a 64x64 thumbnail."""
    import tensorflow as tf
    from PIL import Image
    from scipy.stats import kurtosis, skew

    from src.components.data_loader import load_image
    rows = []
    for start in range(0, len(splits), 64):
        chunk = splits.iloc[start:start + 64]
        x = np.stack([load_image(tf.constant(p), size).numpy() for p in chunk.path])
        seen = std(crop(x)).numpy().mean(axis=-1)            # gray (R = G = B for these images)
        for img, r in zip(seen, chunk.itertuples()):
            t = np.asarray(Image.fromarray(img.astype(np.float32), mode="F").resize((64, 64)), dtype=np.float32)
            v = t.ravel()
            rows.append({"path": r.path, "label": r.label, "width": r.width, "height": r.height,
                         "aspect": r.width / r.height, "dark_frac": float((v < 10).mean()),
                         "bright_frac": float((v > 245).mean()), "p5": float(np.percentile(v, 5)),
                         "p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)),
                         "skew": float(skew(v)), "kurtosis": float(kurtosis(v)),
                         "mean_check": float(v.mean()), "sd_check": float(v.std())})
    return pd.DataFrame(rows)


def stage_s4a_posthoc():
    """POST HOC (added after the goal 10 results): which processed-input statistics carry the source signal.
    The S4a depth-3 tree and logistic regression refit on all 2,982 images; tree splits, impurity importances,
    standardised LR coefficients, and class medians of each feature. Descriptive only, not a criterion."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier, export_text

    from src.components.model_builder import CentralCrop, PerImageStandardize

    cfg = load_config(ARMS["arm2"])
    size = tuple(cfg["training"]["img_size"])
    df = processed_features(pd.read_csv(cfg["splits_path"]), size,
                            CentralCrop(cfg["model_options"]["crop_fraction"]), PerImageStandardize())
    X, y = df[ALL_FEATURES].astype(float).values, df.label.values
    tree = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X, y)
    t = tree.tree_
    path = tree.decision_path(X).toarray().astype(bool)            # images reaching each node
    nodes = [{"node": i, "feature": ALL_FEATURES[t.feature[i]], "threshold": float(t.threshold[i]),
              "n": int(path[:, i].sum()), "n_normal": int((y[path[:, i]] == 0).sum())}
             for i in range(t.node_count) if t.children_left[i] != -1]
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(max_iter=1000).fit(sc.transform(X), y)
    med = df.groupby("label")[ALL_FEATURES].median().T.rename(columns={0: "median_normal", 1: "median_tb"})
    tab = pd.DataFrame({"tree_importance": tree.feature_importances_, "lr_coef_standardised": lr.coef_[0]},
                       index=ALL_FEATURES).join(med)
    out = {"label": "POST HOC (not pre-registered)", "n": len(df), "tree_text": export_text(tree, feature_names=ALL_FEATURES),
           "tree_splits": nodes, "features": tab.reset_index(names="feature").to_dict("records"),
           "note": "fitted on all images (no CV) to describe the signal; CV AUCs are in s4a.json"}
    with open(os.path.join(OUT, "s4a_posthoc.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(out["tree_text"])
    print(tab.sort_values("tree_importance", ascending=False).to_string(float_format=lambda v: f"{v:.4f}"))


def stage_s4a():
    """data_audit.shortcut_test protocol (5-fold CV, LR + depth-3 tree, seed 42) on processed inputs."""
    import keras
    import tensorflow as tf
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    from src.components.data_loader import load_image
    from src.components.model_builder import CentralCrop, PerImageStandardize

    cfg = load_config(ARMS["arm2"])
    size = tuple(cfg["training"]["img_size"])
    opts = cfg["model_options"]
    crop, std = CentralCrop(opts["crop_fraction"]), PerImageStandardize()
    splits = pd.read_csv(cfg["splits_path"])

    # Both models (and both arms) share the same deterministic crop + standardisation layers. Check that on
    # 32 images through each saved model's own layers before computing the features once.
    x32 = np.stack([load_image(tf.constant(p), size).numpy() for p in splits.path[:32]])
    ref = std(crop(x32)).numpy()
    same = {}
    for arm, path in ARMS.items():
        c = load_config(path)
        for m in MODELS:
            model = keras.models.load_model(os.path.join(c["models_dir"], f"{m}_model.keras"))
            probe = keras.Model(model.inputs, model.get_layer("per_image_standardize").output)
            same[f"{arm}/{m}"] = float(np.abs(probe.predict(x32, verbose=0) - ref).max())
    print("[s4a] max |model layers - reference| on 32 images:", same)

    df = processed_features(splits, size, crop, std)
    feature_sets = {
        "width + height + aspect": ["width", "height", "aspect"],
        "intensity (processed input)": INTENSITY,
        "all features (S4a)": ALL_FEATURES,
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    res = []
    y = df.label.values
    for name, cols in feature_sets.items():
        X = df[cols].astype(float).values
        for model_name, model in {
            "logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
            "decision tree (depth 3)": DecisionTreeClassifier(max_depth=3, random_state=42),
        }.items():
            auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
            res.append({"features": name, "model": model_name, "roc_auc_mean": float(auc.mean()),
                        "roc_auc_std": float(auc.std())})
    res = pd.DataFrame(res)
    os.makedirs(OUT, exist_ok=True)
    res.to_csv(os.path.join(OUT, "s4a_shortcut_test.csv"), index=False)
    allf = res[res.features == "all features (S4a)"]
    out = {"n_images": len(df), "n_normal": int((y == 0).sum()), "n_tb": int(y.sum()),
           "layer_identity_max_diff": same,
           "thumbnail_mean_median": float(df.mean_check.median()), "thumbnail_sd_median": float(df.sd_check.median()),
           "results": res.to_dict("records"),
           "all_feature_auc_max": float(allf.roc_auc_mean.max()),
           "rule": "all-feature AUC <= 0.75 (higher of LR and tree, as the raw audit headline 0.918 was the tree)",
           "supported": bool(allf.roc_auc_mean.max() <= 0.75)}
    with open(os.path.join(OUT, "s4a.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(res.to_string(index=False))
    print(json.dumps({k: out[k] for k in ("all_feature_auc_max", "supported")}))


# ---------------------------------------------------------------- S1, S2, S3, S4b, secondary
def paired_delta_auc(y, p_new, p_old):
    from sklearn.metrics import roc_auc_score
    idx = ec.stratified_boot_indices(y, ec.N_BOOT, ec.SEED)
    point = roc_auc_score(y, p_new) - roc_auc_score(y, p_old)
    boot = [roc_auc_score(y[i], p_new[i]) - roc_auc_score(y[i], p_old[i]) for i in idx]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"auc_new": float(roc_auc_score(y, p_new)), "auc_old": float(roc_auc_score(y, p_old)),
            "delta": float(point), "ci_low": float(lo), "ci_high": float(hi)}


def stage_compare():
    from scipy.stats import wilcoxon

    g9cam = pd.read_csv(G9_CAM)
    g9cam = g9cam[(g9cam.site == "montgomery") & (g9cam.model == "resnet")].set_index("file")
    out = {}
    for arm, path in ARMS.items():
        cfg = load_config(path)
        art = cfg["artifacts"]
        ext = os.path.join(art["external_dir"], "evaluation", "metrics_external.json")
        ext = json.load(open(ext))
        masking = json.load(open(os.path.join(art["clinical_dir"], "masking.json")))["results"]
        internal = json.load(open(os.path.join(art["clinical_dir"], "metrics_test.json")))["models"]
        cam = pd.read_csv(os.path.join(art["external_dir"], "gradcam", "attention_external.csv"))
        ceiling = json.load(open(os.path.join(art["external_dir"], "gradcam", "h6_resolution_ceiling.json")))
        for m in MODELS:
            key = f"{arm}/{m}"
            r = {"threshold": ext["thresholds"][m], "internal_test_auc": internal[m]["roc_auc"]}
            new = pd.read_csv(os.path.join(art["external_dir"], "predictions", f"{m}_png.csv"))
            old = pd.read_csv(os.path.join(G9_PRED, f"{m}_png.csv"))
            both = new.merge(old[["site", "file", "prob"]], on=["site", "file"], suffixes=("", "_g9"))
            assert len(both) == len(new) == len(old), f"{key}: external image sets differ"

            # S1: paired dAUC vs goal 9 same architecture, PNG
            s1 = {}
            for site in SITES:
                g = both[both.site == site]
                s1[site] = paired_delta_auc(g.label.values, g.prob.values, g.prob_g9.values)
                s1[site]["auc_ci"] = ext["results"][f"{m}/png/{site}"]["roc_auc"]
            s1["supported"] = all(s1[s]["ci_low"] > 0 for s in SITES)
            s1["secondary_auc_ge_0.80"] = all(s1[s]["auc_new"] >= 0.80 for s in SITES)
            r["S1"] = s1

            # S2: lungs-hidden (centre of the cropped field); goal 4 version reported, trivially affected
            r["S2"] = {"centre_in_crop_auc": masking["centre_in_crop"][m]["roc_auc"],
                       "centre170_auc_trivially_affected": masking["centre170"][m]["roc_auc"],
                       "unmasked_auc": masking["none"][m]["roc_auc"],
                       "mean_p_centre_in_crop": [masking["centre_in_crop"][m]["mean_p_normal"],
                                                 masking["centre_in_crop"][m]["mean_p_tb"]],
                       "supported": masking["centre_in_crop"][m]["roc_auc"]["value"] <= 0.75}

            # S3: Montgomery TB-evidence map, median(lung energy - lung area) > 0; paired Wilcoxon vs goal 9 ResNet50
            g = cam[(cam.site == "montgomery") & (cam.model == m)].set_index("file")
            d = (g.tb_lung_energy - g.lung_area).dropna()
            paired = pd.concat([g.tb_lung_energy.rename("new"), g9cam.tb_lung_energy.rename("g9")], axis=1).dropna()
            w = wilcoxon(paired.new, paired.g9)
            r["S3"] = {"n": int(len(d)), "empty_maps": int(g.tb_empty.sum()),
                       "median_lung_energy": float(g.tb_lung_energy.median()),
                       "median_lung_area": float(g.lung_area.median()),
                       "median_energy_minus_area": float(d.median()),
                       "share_images_energy_gt_area": float((d > 0).mean()),
                       "vs_goal9_resnet": {"n_pairs": int(len(paired)),
                                           "median_paired_diff": float((paired.new - paired.g9).median()),
                                           "goal9_median_lung_energy": float(paired.g9.median()),
                                           "wilcoxon_p": float(w.pvalue)},
                       "resolution_ceiling_median": ceiling["models"][m]["median_ceiling"],
                       "supported": bool(d.median() > 0)}

            # S4b: within-site gap vs between-site shift of Normal scores, PNG
            mean = {s: {lab: float(new[(new.site == s) & (new.label == lab)].prob.mean()) for lab in (0, 1)}
                    for s in SITES}
            gap = {s: mean[s][1] - mean[s][0] for s in SITES}
            shift = abs(mean["shenzhen"][0] - mean["montgomery"][0])
            r["S4b"] = {"g": gap, "d": shift, "mean_p": mean, "supported": bool(min(gap.values()) > shift)}

            # Secondary (not a criterion): external specificity / sensitivity at the new val-tuned threshold
            r["secondary"] = {s: {k: ext["results"][f"{m}/png/{s}"]["at_tuned"][k] for k in
                                  ("specificity", "sensitivity")} for s in SITES}
            out[key] = r

    # goal 9 reference values recomputed from the saved files (sanity check of the S4b and S1 inputs)
    ref = {}
    for m in MODELS:
        old = pd.read_csv(os.path.join(G9_PRED, f"{m}_png.csv"))
        mean = {s: {lab: float(old[(old.site == s) & (old.label == lab)].prob.mean()) for lab in (0, 1)} for s in SITES}
        ref[m] = {"g": {s: mean[s][1] - mean[s][0] for s in SITES},
                  "d": abs(mean["shenzhen"][0] - mean["montgomery"][0])}
    s4a_path = os.path.join(OUT, "s4a.json")
    result = {"primary": "/".join(PRIMARY), "goal9_reference_S4b": ref, "models": out,
              "S4a": json.load(open(s4a_path)) if os.path.exists(s4a_path) else None}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "criteria.json"), "w") as f:
        json.dump(result, f, indent=2, default=bool)
    print(json.dumps(result, indent=1, default=bool))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["s4a", "s4a_posthoc", "compare"], required=True)
    args = parser.parse_args()
    {"s4a": stage_s4a, "s4a_posthoc": stage_s4a_posthoc, "compare": stage_compare}[args.stage]()


if __name__ == "__main__":
    main()
