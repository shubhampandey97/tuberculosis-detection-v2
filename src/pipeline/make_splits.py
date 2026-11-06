"""Goal 1: de-duplicate the dataset and create a stratified, seeded 70/15/15 split.

Run from the repo root:
    python -m src.pipeline.make_splits

Outputs (no image files are deleted or moved):
    data/splits.csv      path, label, split, width, height   (one row per kept image)
    data/duplicates.csv  dropped_path, kept_path, md5        (byte-identical copies left out)

Labels: 0 = Normal, 1 = TB (alphabetical folder order, same as Keras).
Stratification: label x "not 256x256" flag. All Normal images are 256x256; 124 TB images are
not (e.g. 256x221), a possible shortcut (docs/audit.md §5.3), so they are spread evenly across
train/val/test for the subgroup analysis in goal 4.
"""
import hashlib
import os
import re
import sys

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from src.utils.config_loader import load_config

SEED = 42
VAL_FRAC = 0.15
TEST_FRAC = 0.15
CLASS_DIRS = {"Normal Chest X-rays": 0, "TB Chest X-rays": 1}
STANDARD_SIZE = (256, 256)  # (width, height) of every Normal image
TOLERANCE = 0.01  # allowed deviation for split fractions and class shares

SPLITS_CSV = "data/splits.csv"
DUPLICATES_CSV = "data/duplicates.csv"


def file_number(path):
    """Numeric part of 'TB.703.jpg' or 'others (76).jpg', used to pick which copy to keep."""
    nums = re.findall(r"\d+", os.path.basename(path))
    return int(nums[-1]) if nums else sys.maxsize


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def index_images(data_dir):
    found = sorted(d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)))
    if found != sorted(CLASS_DIRS):
        raise ValueError(f"Expected class folders {sorted(CLASS_DIRS)}, found {found}")

    rows = []
    for class_dir, label in CLASS_DIRS.items():
        folder = os.path.join(data_dir, class_dir)
        for name in os.listdir(folder):
            path = os.path.join(folder, name).replace("\\", "/")
            with Image.open(path) as im:  # reads the header only
                width, height = im.size
            rows.append({"path": path, "label": label, "width": width,
                         "height": height, "md5": md5(path)})
    return pd.DataFrame(rows)


def drop_duplicates(df):
    df = df.assign(num=df["path"].map(file_number)).sort_values(["md5", "num", "path"])

    cross_class = df.groupby("md5")["label"].nunique()
    cross_class = cross_class[cross_class > 1].index
    if len(cross_class):
        raise ValueError(f"{len(cross_class)} duplicate groups span both classes: {list(cross_class)}")

    is_dup = df.duplicated("md5", keep="first")
    kept_by_md5 = df[~is_dup].set_index("md5")["path"]
    dropped = df[is_dup]
    dup_report = pd.DataFrame({
        "dropped_path": dropped["path"].values,
        "kept_path": dropped["md5"].map(kept_by_md5).values,
        "md5": dropped["md5"].values,
    }).sort_values("dropped_path")

    return df[~is_dup].drop(columns="num"), dup_report


def is_nonstandard(df):
    return (df["width"] != STANDARD_SIZE[0]) | (df["height"] != STANDARD_SIZE[1])


def stratified_split(df):
    strata = df["label"].astype(str) + "_" + is_nonstandard(df).astype(int).astype(str)

    train, rest = train_test_split(df, test_size=VAL_FRAC + TEST_FRAC,
                                   stratify=strata, random_state=SEED)
    val, test = train_test_split(rest, test_size=TEST_FRAC / (VAL_FRAC + TEST_FRAC),
                                 stratify=strata.loc[rest.index], random_state=SEED)
    return pd.concat([train.assign(split="train"), val.assign(split="val"),
                      test.assign(split="test")])


def check(df, n_files, dup_report):
    targets = {"train": 1 - VAL_FRAC - TEST_FRAC, "val": VAL_FRAC, "test": TEST_FRAC}
    overall_tb = df["label"].mean()

    assert len(df) + len(dup_report) == n_files, "some files are unaccounted for"
    assert df["path"].is_unique, "duplicate paths in split"
    assert df["md5"].is_unique, "identical images remain in the split"
    assert df.groupby("md5")["split"].nunique().max() == 1, "an image hash appears in two splits"
    for split, frac in targets.items():
        part = df[df["split"] == split]
        assert abs(len(part) / len(df) - frac) <= TOLERANCE, f"{split} fraction off target"
        assert abs(part["label"].mean() - overall_tb) <= TOLERANCE, f"{split} class share off"


def report(df, dup_report, n_files):
    print(f"Files found: {n_files}   duplicates dropped: {len(dup_report)}   kept: {len(df)}")
    print("Dropped per class:", dup_report["dropped_path"].str.contains("/TB Chest").map(
        {True: "TB", False: "Normal"}).value_counts().to_dict())

    df = df.assign(cls=df["label"].map({0: "Normal", 1: "TB"}), odd=is_nonstandard(df))
    table = pd.crosstab(df["split"], df["cls"], margins=True, margins_name="total")
    table = table.reindex(["train", "val", "test", "total"])
    table["TB %"] = (100 * table["TB"] / table["total"]).round(1)
    table["share %"] = (100 * table["total"] / len(df)).round(1)
    odd = pd.crosstab(df["split"], df["cls"].where(df["odd"], "std"), margins=True,
                      margins_name="total").reindex(table.index)
    for cls in ["Normal", "TB"]:
        table[f"{cls} not 256x256"] = odd[cls] if cls in odd else 0
    print("\nClass counts per split:")
    print(table.to_string())


def main():
    data_dir = load_config()["data_path"]
    df = index_images(data_dir)
    n_files = len(df)

    df, dup_report = drop_duplicates(df)
    df = stratified_split(df)
    check(df, n_files, dup_report)

    order = {"train": 0, "val": 1, "test": 2}
    out = df.sort_values(["split", "label", "path"], key=lambda s: s.map(order) if s.name == "split" else s)
    out[["path", "label", "split", "width", "height"]].to_csv(SPLITS_CSV, index=False)
    dup_report.to_csv(DUPLICATES_CSV, index=False)

    report(df, dup_report, n_files)
    print(f"\nSaved {SPLITS_CSV} and {DUPLICATES_CSV}")


if __name__ == "__main__":
    main()
