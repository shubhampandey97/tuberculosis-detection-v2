"""Data audit for an image classification dataset (run BEFORE EDA and training).

Expects one sub-folder per class:
    data_dir/
        Normal Chest X-rays/  *.jpg
        TB Chest X-rays/      *.jpg

Run from the repo root:
    python -m src.pipeline.data_audit
    python -m src.pipeline.data_audit --data-dir "data/Some Other Dataset" --out reports/data_audit

What it checks (each answers "can I trust this data?"):
    1. Inventory        files per class, extensions, file sizes
    2. Integrity        corrupt / unreadable files, colour modes (RGB, L, RGBA)
    3. Metadata         width, height, aspect ratio, DPI per class
    4. Exact duplicates byte-identical files (MD5 hash), within and across classes
    5. Near-duplicates  visually identical files (perceptual dHash + Hamming distance)
    6. Pixel stats      brightness / contrast per class on small grayscale thumbnails
    7. Shortcut test    can a simple model predict the label from metadata ALONE?
                        If yes, the classes differ in how they were collected, and a CNN
                        may learn that instead of the disease.

Outputs (in --out):
    inventory.csv        one row per file with every measured property
    exact_duplicates.csv groups of byte-identical files
    near_duplicates.csv  pairs of visually near-identical files
    report.md            human-readable summary

Only needs: numpy, pandas, pillow, scikit-learn (no extra installs).
"""
import argparse
import hashlib
import os
from itertools import combinations

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
THUMB = 64          # thumbnail size for pixel statistics
HASH_SIZE = 8       # dHash grid -> 64-bit hash
NEAR_DUP_BITS = 4   # Hamming distance <= this counts as a near-duplicate (0 = identical look)


# ----------------------------------------------------------------------------- helpers
def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash(gray_img):
    """Difference hash: compares neighbouring pixels of a 9x8 thumbnail -> 64 bools.
    Robust to resizing, recompression and small brightness changes."""
    small = np.asarray(gray_img.resize((HASH_SIZE + 1, HASH_SIZE), Image.LANCZOS), dtype=np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return np.packbits(bits)  # 8 bytes


def inspect_file(path, label_name):
    row = {"path": path.replace("\\", "/"), "class": label_name,
           "ext": os.path.splitext(path)[1].lower(), "file_kb": os.path.getsize(path) / 1024,
           "readable": True, "error": ""}
    try:
        with Image.open(path) as im:
            im.verify()  # detects truncated / corrupt files
        with Image.open(path) as im:
            row.update(width=im.width, height=im.height, mode=im.mode, format=im.format,
                       dpi=str(im.info.get("dpi", "")))
            gray = im.convert("L")
            row["dhash"] = dhash(gray)
            thumb = np.asarray(gray.resize((THUMB, THUMB)), dtype=np.float32)
            row.update(mean_px=thumb.mean(), std_px=thumb.std(),
                       dark_frac=(thumb < 10).mean(), bright_frac=(thumb > 245).mean())
    except Exception as e:  # noqa: BLE001 - we want to record any failure
        row.update(readable=False, error=f"{type(e).__name__}: {e}")
    row["md5"] = md5(path)
    return row


# ----------------------------------------------------------------------------- checks
def build_inventory(data_dir):
    classes = sorted(d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)))
    if len(classes) < 2:
        raise ValueError(f"Expected one sub-folder per class in {data_dir}, found {classes}")
    rows, skipped = [], []
    for label, cls in enumerate(classes):          # alphabetical = same order as Keras
        folder = os.path.join(data_dir, cls)
        for name in sorted(os.listdir(folder)):
            path = os.path.join(folder, name)
            if os.path.splitext(name)[1].lower() not in IMAGE_EXTS:
                skipped.append(path)
                continue
            row = inspect_file(path, cls)
            row["label"] = label
            rows.append(row)
    df = pd.DataFrame(rows)
    df["aspect"] = df["width"] / df["height"]
    df["size_wxh"] = df["width"].astype("Int64").astype(str) + "x" + df["height"].astype("Int64").astype(str)
    return df, classes, skipped


def exact_duplicates(df):
    groups = df[df.duplicated("md5", keep=False)].sort_values(["md5", "path"])
    if groups.empty:
        return groups.assign(group=[]), 0
    groups = groups.assign(group=groups.groupby("md5").ngroup())
    cross = groups.groupby("md5")["class"].nunique().gt(1).sum()
    return groups[["group", "md5", "class", "path"]], int(cross)


def near_duplicates(df):
    ok = df[df["readable"]].reset_index(drop=True)
    hashes = np.stack(ok["dhash"].values)                     # (n, 8) uint8
    bits = np.unpackbits(hashes, axis=1).astype(bool)          # (n, 64)
    pairs = []
    for i in range(len(ok) - 1):                               # row-by-row keeps memory small
        dist = (bits[i + 1:] != bits[i]).sum(axis=1)
        for j in np.nonzero(dist <= NEAR_DUP_BITS)[0]:
            a, b = ok.iloc[i], ok.iloc[i + 1 + j]
            if a["md5"] == b["md5"]:
                continue                                       # already an exact duplicate
            pairs.append({"path_a": a["path"], "class_a": a["class"],
                          "path_b": b["path"], "class_b": b["class"],
                          "hamming": int(dist[j])})
    return pd.DataFrame(pairs, columns=["path_a", "class_a", "path_b", "class_b", "hamming"])


def shortcut_test(df):
    """Cross-validated ROC-AUC of simple models that only see metadata, never the image content.
    ~0.5 = no shortcut. Close to 1.0 = the label leaks through how files were collected."""
    ok = df[df["readable"]]
    y = ok["label"].values
    if len(np.unique(y)) != 2:
        return None  # only defined here for binary problems
    feature_sets = {
        "width + height + aspect": ["width", "height", "aspect"],
        "file size (KB)": ["file_kb"],
        "brightness/contrast": ["mean_px", "std_px", "dark_frac", "bright_frac"],
        "all metadata": ["width", "height", "aspect", "file_kb",
                         "mean_px", "std_px", "dark_frac", "bright_frac"],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    for name, cols in feature_sets.items():
        X = ok[cols].astype(float).values
        for model_name, model in {
            "logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
            "decision tree (depth 3)": DecisionTreeClassifier(max_depth=3, random_state=42),
        }.items():
            auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
            results.append({"features": name, "model": model_name,
                            "roc_auc_mean": auc.mean(), "roc_auc_std": auc.std()})
    return pd.DataFrame(results)


# ----------------------------------------------------------------------------- report
def md_table(df, index=True, floatfmt=".3f"):
    try:
        return df.to_markdown(index=index, floatfmt=floatfmt)  # needs the 'tabulate' package
    except ImportError:
        return "```\n" + df.to_string(index=index) + "\n```"


def write_report(out_dir, data_dir, df, classes, skipped, dups, cross, near, shortcut):
    ok = df[df["readable"]]
    lines = [f"# Data audit: `{data_dir}`", ""]

    lines += ["## 1. Inventory", ""]
    counts = df.groupby("class").agg(files=("path", "size"), mean_kb=("file_kb", "mean"))
    counts["share_%"] = 100 * counts["files"] / counts["files"].sum()
    lines += [md_table(counts.round(1)), "",
              f"Label mapping (alphabetical, same as Keras): "
              + ", ".join(f"`{c}` = {i}" for i, c in enumerate(classes)), "",
              f"Extensions: {df['ext'].value_counts().to_dict()}",
              f"Non-image files skipped: {len(skipped)}", ""]

    lines += ["## 2. Integrity", "",
              f"Unreadable / corrupt files: **{(~df['readable']).sum()}**", ""]
    if (~df["readable"]).any():
        lines += [md_table(df.loc[~df["readable"], ["path", "error"]], index=False), ""]
    lines += ["Colour modes per class:", "",
              md_table(pd.crosstab(ok["class"], ok["mode"])), ""]

    lines += ["## 3. Image size and metadata", "",
              "Top sizes per class (width x height):", ""]
    top = ok.groupby("class")["size_wxh"].value_counts().groupby(level=0).head(5)
    lines += [md_table(top.rename("count").to_frame()), "",
              "Number of distinct sizes per class: "
              + str(ok.groupby("class")["size_wxh"].nunique().to_dict()), "",
              "Aspect ratio (width/height) per class:", "",
              md_table(ok.groupby("class")["aspect"].describe()[["min", "50%", "max"]]), "",
              "DPI values per class:", "",
              md_table(pd.crosstab(ok["dpi"], ok["class"])), ""]

    lines += ["## 4. Exact duplicates (MD5)", "",
              f"Files in duplicate groups: **{len(dups)}** "
              f"({dups['group'].nunique() if len(dups) else 0} groups). "
              f"Groups spanning more than one class (label conflicts): **{cross}**.", "",
              "Duplicates must be removed before splitting, or one copy can land in train and the "
              "other in test.", ""]

    lines += ["## 5. Near-duplicates (perceptual hash)", "",
              f"Pairs with Hamming distance <= {NEAR_DUP_BITS} (not byte-identical): **{len(near)}**. "
              f"Cross-class pairs: **{int((near['class_a'] != near['class_b']).sum()) if len(near) else 0}**.",
              "", "These are candidates only: open a few pairs and look before removing anything.", ""]

    lines += ["## 6. Pixel statistics (64x64 grayscale)", "",
              md_table(ok.groupby("class")[["mean_px", "std_px", "dark_frac", "bright_frac"]].mean()), ""]

    lines += ["## 7. Shortcut test (metadata only, 5-fold CV ROC-AUC)", ""]
    if shortcut is None:
        lines += ["Skipped (only implemented for 2 classes).", ""]
    else:
        lines += [md_table(shortcut, index=False), "",
                  "How to read this: ~0.5 means metadata says nothing about the label. "
                  "Values near 1.0 mean the classes can be told apart **without looking at the "
                  "anatomy**, so a CNN might learn the same shortcut. Investigate before trusting "
                  "any high accuracy.", ""]

    path = os.path.join(out_dir, "report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=None, help="dataset root (default: data_path in config.yaml)")
    parser.add_argument("--out", default="reports/data_audit", help="output folder")
    args = parser.parse_args()

    data_dir = args.data_dir
    if data_dir is None:
        from src.utils.config_loader import load_config
        data_dir = load_config()["data_path"]
    os.makedirs(args.out, exist_ok=True)

    print(f"Auditing {data_dir} ...")
    df, classes, skipped = build_inventory(data_dir)
    dups, cross = exact_duplicates(df)
    near = near_duplicates(df)
    shortcut = shortcut_test(df)

    df.drop(columns="dhash").to_csv(os.path.join(args.out, "inventory.csv"), index=False)
    dups.to_csv(os.path.join(args.out, "exact_duplicates.csv"), index=False)
    near.to_csv(os.path.join(args.out, "near_duplicates.csv"), index=False)
    report = write_report(args.out, data_dir, df, classes, skipped, dups, cross, near, shortcut)

    print(f"Files: {len(df)}  unreadable: {(~df['readable']).sum()}  "
          f"exact-duplicate files: {len(dups)}  near-duplicate pairs: {len(near)}")
    if shortcut is not None:
        best = shortcut.loc[shortcut["roc_auc_mean"].idxmax()]
        print(f"Shortcut test: best metadata-only ROC-AUC = {best['roc_auc_mean']:.3f} "
              f"({best['features']}, {best['model']})")
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
