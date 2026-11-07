"""Goal 9: external validation on the NLM Shenzhen and Montgomery chest X-ray sets.

    python -m src.pipeline.external_validation --stage sample     # 10 images per site, examples + histograms
    python -m src.pipeline.external_validation --stage prepare    # inventory + convert all images (cached)
    (later stages: overlap, evaluate, gradcam)

Data (not committed): data/external/ChinaSet_AllFiles/ChinaSet_AllFiles/ and
data/external/Montgomery/MontgomerySet/. Converted 256x256 images: data/external/prepared/.
Outputs: artifacts/external/.

Preprocessing (see docs/audit.md, goal 9):
  1. decode to 8-bit grayscale: palette/RGB images are checked to be truly gray; 16-bit images are
     scaled by their real bit depth (2**bits - 1), never by a blind /256;
  2. antialiased (Lanczos) downscale to 256x256, stretching non-square images like the training set;
  3. saved as lossless PNG (primary) and as JPEG at the training set's quality (sensitivity variant);
  4. model input then goes through the unchanged data_loader.decode_and_resize (224x224, raw 0-255).
"""
import argparse
import glob
import io
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image, JpegImagePlugin  # noqa: E402

from src.utils.config_loader import load_config  # noqa: E402

EXT_DIR = "data/external"
PREP_DIR = os.path.join(EXT_DIR, "prepared")
OUT_DIR = "artifacts/external"
SITES = {
    "shenzhen": {"root": os.path.join(EXT_DIR, "ChinaSet_AllFiles", "ChinaSet_AllFiles"),
                 "expected": {0: 326, 1: 336}},
    "montgomery": {"root": os.path.join(EXT_DIR, "Montgomery", "MontgomerySet"),
                   "expected": {0: 80, 1: 58}},
}
PREP_SIZE = (256, 256)
JUNK = ("__MACOSX", ".DS_Store", "Thumbs.db")
SEED = 42

# IJG standard luminance quantisation table (quality 50), used to estimate JPEG quality.
STD_LUMA = np.array([16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
                     14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113,
                     92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99])


# ---------------------------------------------------------------- inventory and readings
def is_junk(path):
    parts = path.replace("\\", "/").split("/")
    return any(p in JUNK or p.startswith("._") for p in parts)


def list_images(site):
    root = SITES[site]["root"]
    files = sorted(f for f in glob.glob(os.path.join(root, "CXR_png", "*.png")) if not is_junk(f))
    rows = []
    for f in files:
        m = re.search(r"_(\d{4})_([01])\.png$", f)
        if not m:
            raise ValueError(f"Unexpected file name: {f}")
        rows.append({"site": site, "file": os.path.basename(f), "path": f.replace("\\", "/"),
                     "label": int(m.group(2))})
    return pd.DataFrame(rows)


SHENZHEN_RE = re.compile(r"^\s*(male|female)\s*,?\s*(\d+)\s*(?:yrs?|years?)\s*$", re.I)
MONT_SEX_RE = re.compile(r"Patient'?s Sex:\s*([MF])\b", re.I)
MONT_AGE_RE = re.compile(r"Patient'?s Age:\s*(\d{1,3})Y\b", re.I)


def parse_reading(site, text):
    """Returns (sex, age, findings, ok). Strict: anything not matching exactly is left unknown."""
    lines = [ln.strip() for ln in text.replace("\r", "").split("\n") if ln.strip()]
    if site == "shenzhen":
        m = SHENZHEN_RE.match(lines[0]) if lines else None
        if not m:
            return None, None, " | ".join(lines), False
        sex = "M" if m.group(1).lower() == "male" else "F"
        return sex, int(m.group(2)), " | ".join(lines[1:]), True
    sex_m, age_m = MONT_SEX_RE.search(text), MONT_AGE_RE.search(text)
    findings = " | ".join(ln for ln in lines if not MONT_SEX_RE.search(ln) and not MONT_AGE_RE.search(ln))
    sex = sex_m.group(1).upper() if sex_m else None
    age = int(age_m.group(1)) if age_m else None
    return sex, age, findings, bool(sex_m and age_m)


# Manual corrections for readings the strict parser rejects, applied only where the meaning is unambiguous
# (approved 2026-11-07). "male 42" (no unit) and Montgomery sex "O" stay unknown.
READING_CORRECTIONS = {
    ("shenzhen", "CHNCXR_0044_0.png"): ("F", 32.0, "typo 'femal 32yrs' -> F, 32"),
    ("shenzhen", "CHNCXR_0045_0.png"): ("F", 60.0, "typo 'femal 60yrs' -> F, 60"),
    ("shenzhen", "CHNCXR_0096_0.png"): ("F", 69.0, "typo 'femal 69yrs' -> F, 69"),
    ("shenzhen", "CHNCXR_0115_0.png"): ("M", 16 / 12, "'male 16month' -> M, 1.33 years"),
    ("shenzhen", "CHNCXR_0201_0.png"): ("M", 16 / 12, "'male 16month' -> M, 1.33 years"),
    ("shenzhen", "CHNCXR_0202_0.png"): ("F", 64 / 365.25, "'female 64days' -> F, 0.18 years"),
    ("shenzhen", "CHNCXR_0647_1.png"): ("M", 38.0, "'Male , 38yrs,' (extra commas) -> M, 38"),
}


def add_readings(df):
    out = []
    for _, r in df.iterrows():
        txt_path = os.path.join(SITES[r.site]["root"], "ClinicalReadings", r.file.replace(".png", ".txt"))
        if not os.path.exists(txt_path):
            out.append((None, None, "", False, "missing reading file"))
            continue
        with open(txt_path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        sex, age, findings, ok = parse_reading(r.site, text)
        raw = "" if ok else text.strip().replace("\n", " / ")[:200]
        correction = ""
        if not ok and (r.site, r.file) in READING_CORRECTIONS:
            sex, age, correction = READING_CORRECTIONS[(r.site, r.file)]
            findings = " | ".join(ln.strip() for ln in text.replace("\r", "").split("\n")[1:] if ln.strip())
            ok = True
        out.append((sex, age, findings, ok, raw, correction))
    df = df.copy()
    df[["sex", "age", "findings", "reading_ok", "reading_raw_if_unparsed", "reading_correction"]] = \
        pd.DataFrame(out, index=df.index)
    return df


def stage_readings():
    """Re-parse the clinical readings (with corrections) into inventory.csv without re-reading images."""
    inv = pd.read_csv(os.path.join(OUT_DIR, "inventory.csv"))
    cols = ["sex", "age", "findings", "reading_ok", "reading_raw_if_unparsed", "reading_correction", "age_band"]
    fresh = pd.concat([add_readings(list_images(s)) for s in SITES])
    fresh["age_band"] = fresh["age"].map(age_band)
    inv = inv.drop(columns=[c for c in cols if c in inv]).merge(fresh[["site", "file"] + cols], on=["site", "file"])
    inv.to_csv(os.path.join(OUT_DIR, "inventory.csv"), index=False)
    unparsed = inv[~inv["reading_ok"].astype(bool)][["site", "file", "label", "reading_raw_if_unparsed"]]
    unparsed.to_csv(os.path.join(OUT_DIR, "readings_unparsed.csv"), index=False)
    corrected = inv[inv["reading_correction"].fillna("") != ""][["site", "file", "label", "reading_raw_if_unparsed",
                                                                 "reading_correction"]]
    corrected.to_csv(os.path.join(OUT_DIR, "readings_corrected.csv"), index=False)
    for site, g in inv.groupby("site"):
        print(f"{site}: parsed {int(g.reading_ok.astype(bool).sum())}/{len(g)}, corrected "
              f"{int((g.reading_correction.fillna('') != '').sum())}, sex {g.sex.fillna('unknown').value_counts().to_dict()}, "
              f"age bands {g.age_band.value_counts().to_dict()}")
    print("still unknown:\n" + unparsed.to_string(index=False))


def age_band(age):
    if age is None or pd.isna(age):
        return "unknown"
    return "<18" if age < 18 else "18-39" if age < 40 else "40-59" if age < 60 else ">=60"


# ---------------------------------------------------------------- preprocessing
def to_gray8(im, bits=None):
    """Decode any PIL image to 8-bit grayscale (numpy uint8) + a dict describing what was done."""
    info = {"mode": im.mode}
    if im.mode == "P":
        pal = np.array(im.getpalette()[:768]).reshape(-1, 3)
        used = np.unique(np.asarray(im))
        info["palette_gray"] = bool(np.all(pal[used, 0] == pal[used, 1]) and np.all(pal[used, 1] == pal[used, 2]))
        im = im.convert("RGB")
    if im.mode in ("RGBA", "LA"):
        im = im.convert(im.mode[:-1])
    if im.mode == "RGB":
        a = np.asarray(im)
        gray_ch = bool(np.array_equal(a[..., 0], a[..., 1]) and np.array_equal(a[..., 1], a[..., 2]))
        info["rgb_channels_equal"] = gray_ch
        g = a[..., 0] if gray_ch else np.asarray(im.convert("L"))  # ITU-R 601 luma if not gray
        info["bits"] = 8
        return g.astype(np.uint8), info
    if im.mode == "L":
        info["bits"] = 8
        return np.asarray(im, dtype=np.uint8), info
    if im.mode in ("I;16", "I;16B", "I;16L", "I"):
        a = np.asarray(im, dtype=np.float64)
        bits = bits or max(8, int(np.ceil(np.log2(a.max() + 1))))  # real bit depth, not a blind /256
        info["bits"] = bits
        return np.clip(np.round(a / (2 ** bits - 1) * 255), 0, 255).astype(np.uint8), info
    raise ValueError(f"Unsupported image mode {im.mode}")


def estimate_jpeg_quality(path):
    """IJG quality whose scaled standard luminance table best matches the file's table."""
    with Image.open(path) as im:
        q = np.array(im.quantization[0])
        sampling = JpegImagePlugin.get_sampling(im)
    best, best_err = None, np.inf
    for quality in range(1, 101):
        scale = 5000 / quality if quality < 50 else 200 - 2 * quality
        table = np.clip(np.floor((STD_LUMA * scale + 50) / 100), 1, 255)
        err = np.abs(np.sort(table) - np.sort(q)).sum()  # order-independent (zigzag vs natural)
        if err < best_err:
            best, best_err = quality, err
    return best, int(best_err), sampling


def prepare_image(path, jpeg_quality, jpeg_subsampling, bits=None):
    """Original -> 256x256 gray -> (png_bytes, jpeg_bytes, info)."""
    with Image.open(path) as im:
        im.load()
        gray, info = to_gray8(im, bits)
    info.update({"width": gray.shape[1], "height": gray.shape[0], "min": int(gray.min()), "max": int(gray.max())})
    small = Image.fromarray(gray, "L").resize(PREP_SIZE, Image.LANCZOS)  # antialiased; stretches non-square
    png = io.BytesIO()
    small.save(png, format="PNG")
    jpg = io.BytesIO()  # training JPEGs are 3-component, so save RGB at the training quality/subsampling
    small.convert("RGB").save(jpg, format="JPEG", quality=jpeg_quality, subsampling=jpeg_subsampling)
    return png.getvalue(), jpg.getvalue(), info


def prepared_paths(site, file):
    stem = file[:-4]
    return (os.path.join(PREP_DIR, "png", site, stem + ".png"), os.path.join(PREP_DIR, "jpg", site, stem + ".jpg"))


def training_jpeg_profile(n_per_class=100):
    """Estimated JPEG quality / subsampling of the training images, per class."""
    splits = pd.read_csv(load_config()["splits_path"])
    out = {}
    for label, name in ((0, "Normal"), (1, "TB")):
        paths = splits[splits.label == label]["path"].sample(n_per_class, random_state=SEED)
        est = [estimate_jpeg_quality(p) for p in paths]
        qs = pd.Series([e[0] for e in est])
        out[name] = {"n": n_per_class, "quality_median": float(qs.median()), "quality_min": int(qs.min()),
                     "quality_max": int(qs.max()), "quality_counts": qs.value_counts().head(5).to_dict(),
                     "max_table_error": int(max(e[1] for e in est)),
                     "subsampling": pd.Series([e[2] for e in est]).value_counts().to_dict()}
    return out


# ---------------------------------------------------------------- model-input view (unchanged training path)
def model_input(encoded_bytes):
    import tensorflow as tf
    from src.components.data_loader import decode_and_resize
    size = tuple(load_config()["training"]["img_size"])
    return decode_and_resize(tf.constant(encoded_bytes), size).numpy()


# ---------------------------------------------------------------- sample stage
def stage_sample(n_per_site=10):
    os.makedirs(OUT_DIR, exist_ok=True)
    profile = training_jpeg_profile()
    print("Training JPEG profile:", json.dumps(profile, indent=1))
    # Use the quality/subsampling shared by the training classes; if they differ, report and use the TB value
    # (the larger class) - decided on the full run.
    q = int(round(np.median([profile["Normal"]["quality_median"], profile["TB"]["quality_median"]])))
    sub_counts = profile["TB"]["subsampling"]  # PIL codes: 0 = 4:4:4, 1 = 4:2:2, 2 = 4:2:0
    jpeg_sub = int(max(sub_counts, key=sub_counts.get))
    rows = []
    samples = {}
    for site in SITES:
        df = add_readings(list_images(site))
        pick = pd.concat([df[df.label == 0].sample(n_per_site // 2, random_state=SEED),
                          df[df.label == 1].sample(n_per_site - n_per_site // 2, random_state=SEED)])
        samples[site] = []
        for _, r in pick.iterrows():
            png, jpg, info = prepare_image(r.path, q, jpeg_sub)
            p_png, p_jpg = prepared_paths(site, r.file)
            for p, b in ((p_png, png), (p_jpg, jpg)):
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p, "wb") as f:
                    f.write(b)
            rows.append({"site": site, "file": r.file, "label": r.label, "sex": r.sex, "age": r.age,
                         **info, "png_kb": len(png) / 1024, "jpg_kb": len(jpg) / 1024})
            samples[site].append((r, png, jpg))
    info_df = pd.DataFrame(rows)
    info_df.to_csv(os.path.join(OUT_DIR, "sample_preprocessing.csv"), index=False)
    print(info_df.drop(columns=["file"]).to_string())

    # training reference images (from the train split)
    splits = pd.read_csv(load_config()["splits_path"])
    train = splits[splits.split == "train"]
    ref = {name: train[train.label == lab].sample(200, random_state=SEED)["path"].tolist()
           for lab, name in ((0, "Normal"), (1, "TB"))}

    # --- examples figure: training references + originals / prepared / model input
    fig, axes = plt.subplots(3, 6, figsize=(15, 8.4))
    for j, (name, paths) in enumerate([("Normal", ref["Normal"][:3]), ("TB", ref["TB"][:3])]):
        for k, p in enumerate(paths):
            ax = axes[0, j * 3 + k]
            ax.imshow(Image.open(p).convert("L"), cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"training {name}\n{os.path.basename(p)} ({Image.open(p).size[0]}x{Image.open(p).size[1]})",
                         fontsize=7)
    row = 1
    for site in SITES:
        chosen = [s for s in samples[site] if s[0].label == 0][:1] + [s for s in samples[site] if s[0].label == 1][:1]
        for k, (r, png, jpg) in enumerate(chosen):
            with Image.open(r.path) as im:
                orig = im.convert("L")
                ow, oh = orig.size
                orig.thumbnail((512, 512))
            cols = [(orig, f"{site} original {ow}x{oh}"), (Image.open(io.BytesIO(png)), "prepared PNG 256x256"),
                    (model_input(png).astype(np.uint8), "model input 224x224")]
            for c, (img, title) in enumerate(cols):
                ax = axes[row, k * 3 + c]
                ax.imshow(img, cmap="gray", vmin=0, vmax=255)
                lab = "TB" if r.label else "Normal"
                ax.set_title(f"{title}\n{r.file} ({lab})", fontsize=7)
        row += 1
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        if not ax.images:
            ax.axis("off")
    fig.suptitle("Preprocessing check: training images (row 1) vs external images (original -> 256 PNG -> model input)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(OUT_DIR, "preprocessing_examples.png"), dpi=100)
    plt.close(fig)

    # --- intensity histograms on the model input (what the network sees)
    def gray_values(encoded_list):
        return np.concatenate([model_input(b)[..., 0].ravel() for b in encoded_list])

    groups = {
        "training Normal (200)": gray_values([open(p, "rb").read() for p in ref["Normal"]]),
        "training TB (200)": gray_values([open(p, "rb").read() for p in ref["TB"]]),
    }
    for site in SITES:
        groups[f"{site} PNG ({len(samples[site])})"] = gray_values([s[1] for s in samples[site]])
        groups[f"{site} JPEG ({len(samples[site])})"] = gray_values([s[2] for s in samples[site]])
    stats = {k: {"mean": float(v.mean()), "std": float(v.std()), "p1": float(np.percentile(v, 1)),
                 "p99": float(np.percentile(v, 99))} for k, v in groups.items()}
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for k, v in groups.items():
        ax.hist(v, bins=64, range=(0, 255), density=True, histtype="step", lw=1.6 if "training" in k else 1.0,
                ls="-" if "PNG" in k or "training" in k else ":", label=f"{k}: mean {v.mean():.0f}, sd {v.std():.0f}")
    ax.set(xlabel="pixel value (model input, 0-255)", ylabel="density", title="Intensity histograms of model input")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "preprocessing_histograms.png"), dpi=110)
    plt.close(fig)

    with open(os.path.join(OUT_DIR, "sample_summary.json"), "w") as f:
        json.dump({"training_jpeg_profile": profile, "jpeg_quality_used": q, "jpeg_subsampling_used": jpeg_sub,
                   "intensity_stats": stats}, f, indent=2, default=str)
    print(json.dumps(stats, indent=1))


# ---------------------------------------------------------------- data audit of the external sets
AUDIT_DIR = os.path.join(OUT_DIR, "data_audit")
SIM_CACHE = os.path.join(PREP_DIR, "sim_vectors_original.npz")
METADATA_FLAG_AUC = 0.80  # all-metadata AUC at or above this = the site's labels are guessable from metadata


def external_inventory(site):
    """data_audit.inspect_file on every image of a site; labels from the _0/_1 suffix, not folders."""
    from src.pipeline import data_audit as da
    rows = []
    for _, r in list_images(site).iterrows():
        row = da.inspect_file(r.path, "TB" if r.label else "Normal")
        row.update(site=site, file=r.file, label=r.label)
        rows.append(row)
    df = pd.DataFrame(rows)
    df["aspect"] = df["width"] / df["height"]
    return df


def training_audit_reference():
    """Training-set numbers from the original data audit (reports/data_audit/)."""
    inv = pd.read_csv("reports/data_audit/inventory.csv")
    pix = inv.groupby("class")[["mean_px", "std_px", "dark_frac", "bright_frac", "file_kb"]].median()
    return pix


def stage_audit():
    from src.pipeline import data_audit as da
    os.makedirs(AUDIT_DIR, exist_ok=True)
    sim, lines = {}, ["# Data audit of the external sets", "",
                      "Same functions as `src/pipeline/data_audit.py` (unchanged): `inspect_file` (64x64 thumbnail "
                      "pixel statistics, 32x32 z-normalised similarity vector, MD5), `exact_duplicates`, "
                      f"`near_duplicates` (thumbnail correlation >= {da.NEAR_DUP_CORR}) and `shortcut_test` "
                      "(5-fold stratified CV ROC-AUC, seed 42). Labels come from the `_0`/`_1` file suffix. "
                      "File size = original PNG.", ""]
    shortcut_rows, pixel_rows = [], []
    for site in SITES:
        df = external_inventory(site)
        sim[site] = (df["file"].values, np.stack(df["sim_vec"].values))
        df.drop(columns="sim_vec").to_csv(os.path.join(AUDIT_DIR, f"inventory_{site}.csv"), index=False)
        exact, cross = da.exact_duplicates(df)
        near = da.near_duplicates(df)
        exact.to_csv(os.path.join(AUDIT_DIR, f"exact_duplicates_{site}.csv"), index=False)
        near.to_csv(os.path.join(AUDIT_DIR, f"near_duplicates_{site}.csv"), index=False)
        sc = da.shortcut_test(df).assign(site=site)
        shortcut_rows.append(sc)
        for cls, g in df.groupby("class"):
            pixel_rows.append({"site": site, "class": cls, "n": len(g),
                               **g[["mean_px", "std_px", "dark_frac", "bright_frac", "file_kb"]].median().to_dict()})
        n_exact_groups = exact["group"].nunique() if len(exact) else 0
        lines += [f"## {site.capitalize()} ({len(df)} images, unreadable: {(~df['readable']).sum()})", "",
                  f"- Exact duplicate groups (MD5): **{n_exact_groups}** (cross-class: {cross}).",
                  f"- Near-duplicate pairs (thumbnail corr >= {da.NEAR_DUP_CORR}): **{len(near)}**"
                  + (f", cross-class: {(near.class_a != near.class_b).sum()}" if len(near) else "") + ".", ""]
    np.savez_compressed(SIM_CACHE, **{f"{s}_files": v[0] for s, v in sim.items()},
                        **{f"{s}_vecs": v[1] for s, v in sim.items()})

    sc = pd.concat(shortcut_rows)
    sc.to_csv(os.path.join(AUDIT_DIR, "shortcut_test.csv"), index=False)
    pix = pd.DataFrame(pixel_rows)
    ref = training_audit_reference().reset_index().assign(site="training (Kaggle)")
    ref["n"] = pd.read_csv("reports/data_audit/inventory.csv").groupby("class").size().values
    pix = pd.concat([ref, pix], ignore_index=True)[["site", "class", "n", "mean_px", "std_px", "dark_frac",
                                                    "bright_frac", "file_kb"]]
    pix.to_csv(os.path.join(AUDIT_DIR, "pixel_stats.csv"), index=False)

    # Training-set values from reports/data_audit/report.md section 7 (logistic regression, tree).
    train_sc = {"width + height + aspect": (0.5249, 0.5249), "file size (KB)": (0.7843, 0.8607),
                "brightness/contrast": (0.7443, 0.8028), "all metadata": (0.9107, 0.9183)}
    table = sc.pivot_table(index="features", columns=["site", "model"], values="roc_auc_mean")
    rows = []
    for feat in train_sc:
        r = {"features": feat, "training LR": train_sc[feat][0], "training tree": train_sc[feat][1]}
        for site in SITES:
            r[f"{site} LR"] = table.loc[feat, (site, "logistic regression")]
            r[f"{site} tree"] = table.loc[feat, (site, "decision tree (depth 3)")]
        rows.append(r)
    comp = pd.DataFrame(rows).set_index("features")
    comp.to_csv(os.path.join(AUDIT_DIR, "shortcut_vs_training.csv"))
    lines += ["## Metadata-only shortcut test (5-fold CV ROC-AUC), next to the training data audit", "",
              da.md_table(comp), "",
              f"Flag rule: all-metadata AUC >= {METADATA_FLAG_AUC} (either model) = the site's labels can be "
              "guessed from acquisition metadata, which limits it as an external test.", ""]
    for site in SITES:
        best = max(comp.loc["all metadata", f"{site} LR"], comp.loc["all metadata", f"{site} tree"])
        verdict = "**FLAGGED**" if best >= METADATA_FLAG_AUC else "not flagged"
        lines.append(f"- {site}: best all-metadata AUC {best:.3f} -> {verdict}")
    lines += ["", "## Pixel statistics per class (medians; 64x64 grayscale thumbnails of the original image)", "",
              da.md_table(pix.set_index(["site", "class"])), ""]
    with open(os.path.join(AUDIT_DIR, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# ---------------------------------------------------------------- prepare stage (all images)
def jpeg_settings():
    """Quality/subsampling for the JPEG variant, from the training files (cached)."""
    path = os.path.join(OUT_DIR, "jpeg_profile.json")
    if not os.path.exists(path):
        prof = training_jpeg_profile()
        q = int(round(np.median([prof["Normal"]["quality_median"], prof["TB"]["quality_median"]])))
        sub_counts = prof["TB"]["subsampling"]
        with open(path, "w") as f:
            json.dump({"profile": prof, "quality": q, "subsampling": int(max(sub_counts, key=sub_counts.get))},
                      f, indent=2, default=str)
    with open(path) as f:
        d = json.load(f)
    return d["quality"], d["subsampling"]


def stage_prepare(force=False):
    os.makedirs(OUT_DIR, exist_ok=True)
    q, sub = jpeg_settings()
    rows = []
    for site in SITES:
        df = add_readings(list_images(site))
        counts = df["label"].value_counts().to_dict()
        if counts != SITES[site]["expected"]:
            raise ValueError(f"{site}: label counts {counts} != expected {SITES[site]['expected']}")
        for _, r in df.iterrows():
            p_png, p_jpg = prepared_paths(site, r.file)
            png, jpg, info = prepare_image(r.path, q, sub)
            if force or not (os.path.exists(p_png) and os.path.exists(p_jpg)):
                for p, b in ((p_png, png), (p_jpg, jpg)):
                    os.makedirs(os.path.dirname(p), exist_ok=True)
                    with open(p, "wb") as f:
                        f.write(b)
            rows.append({**r.to_dict(), **info, "orig_kb": os.path.getsize(r.path) / 1024,
                         "png_path": p_png.replace("\\", "/"), "jpg_path": p_jpg.replace("\\", "/"),
                         "jpg_kb": len(jpg) / 1024, "age_band": age_band(r.age)})
        print(f"[prepare] {site}: {len(df)} images")
    inv = pd.DataFrame(rows)
    inv.to_csv(os.path.join(OUT_DIR, "inventory.csv"), index=False)
    unparsed = inv[~inv["reading_ok"].astype(bool)][["site", "file", "label", "reading_raw_if_unparsed"]]
    unparsed.to_csv(os.path.join(OUT_DIR, "readings_unparsed.csv"), index=False)

    summary = {"jpeg_variant": {"quality": q, "subsampling": sub}}
    for site, g in inv.groupby("site"):
        summary[site] = {
            "n": len(g), "labels": {int(k): int(v) for k, v in g["label"].value_counts().items()},
            "expected": SITES[site]["expected"], "modes": g["mode"].value_counts().to_dict(),
            "palette_not_gray": int((g.get("palette_gray") == False).sum()),  # noqa: E712
            "rgb_not_gray": int((g.get("rgb_channels_equal") == False).sum()),  # noqa: E712
            "bit_depths": g["bits"].value_counts().to_dict(),
            "pixel_max_below_255": int((g["max"] < 255).sum()), "pixel_min_above_0": int((g["min"] > 0).sum()),
            "width": [int(g.width.min()), int(g.width.max())], "height": [int(g.height.min()), int(g.height.max())],
            "aspect_w_over_h": [round(float((g.width / g.height).min()), 3), round(float((g.width / g.height).max()), 3)],
            "readings_parsed": int(g["reading_ok"].astype(bool).sum()), "readings_unparsed": int((~g["reading_ok"].astype(bool)).sum()),
            "sex": g["sex"].fillna("unknown").value_counts().to_dict(),
            "age_band": g["age_band"].value_counts().to_dict(),
            "age_range": [float(g.age.min()), float(g.age.max())] if g.age.notna().any() else None,
        }
    with open(os.path.join(OUT_DIR, "inventory_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary, indent=1, default=str))


# ---------------------------------------------------------------- overlap with the training dataset
OVERLAP_DIR = os.path.join(OUT_DIR, "overlap")
AUTO_EXCLUDE_CORR = 0.95


def training_sim_vectors():
    from src.pipeline import data_audit as da
    splits = pd.read_csv(load_config()["splits_path"])
    vecs = np.stack([da.sim_vector(Image.open(p).convert("L")) for p in splits["path"]])
    return splits, vecs


def stage_overlap():
    os.makedirs(OVERLAP_DIR, exist_ok=True)
    if not os.path.exists(SIM_CACHE):
        raise FileNotFoundError("Run --stage audit first (it caches the external similarity vectors).")
    cache = np.load(SIM_CACHE, allow_pickle=True)
    splits, train_vecs = training_sim_vectors()
    inv = pd.read_csv(os.path.join(OUT_DIR, "inventory.csv"))
    rows = []
    for site in SITES:
        files, ext_vecs = cache[f"{site}_files"], cache[f"{site}_vecs"]
        corr = ext_vecs @ train_vecs.T / ext_vecs.shape[1]          # Pearson correlations (n_ext, 2982)
        order = np.argsort(-corr, axis=1)
        for i, f in enumerate(files):
            j, j2 = order[i, 0], order[i, 1]
            m = splits.iloc[j]
            rows.append({"site": site, "file": f, "best_corr": float(corr[i, j]), "match_path": m.path,
                         "match_label": int(m.label), "match_split": m.split,
                         "second_corr": float(corr[i, j2])})
    best = pd.DataFrame(rows).merge(inv[["site", "file", "label", "png_path"]], on=["site", "file"])
    best.to_csv(os.path.join(OVERLAP_DIR, "best_matches.csv"), index=False)

    # distribution
    thresholds = [0.80, 0.90, 0.95, 0.99]
    dist = {}
    for site, g in best.groupby("site"):
        c = g["best_corr"]
        dist[site] = {"n": len(g), "median": float(c.median()), "p95": float(c.quantile(0.95)), "max": float(c.max()),
                      **{f">={t}": int((c >= t).sum()) for t in thresholds},
                      "candidates>=0.95_by_label": {f"external {lab} -> training {ml}": int(n) for (lab, ml), n in
                                                    g[g.best_corr >= 0.95].groupby(["label", "match_label"]).size().items()}}
    with open(os.path.join(OVERLAP_DIR, "distribution.json"), "w") as f:
        json.dump(dist, f, indent=2)
    print(json.dumps(dist, indent=1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (site, g) in zip(axes, best.groupby("site")):
        ax.hist(g["best_corr"], bins=np.linspace(min(0.5, g.best_corr.min()), 1.0, 51), color="steelblue")
        for t, ls in ((0.90, ":"), (0.95, "--"), (0.99, "-")):
            ax.axvline(t, color="red", ls=ls, lw=0.9, label=f"{t}: {int((g.best_corr >= t).sum())} images")
        ax.set(yscale="log", xlabel="best thumbnail correlation with any training-dataset image",
               ylabel="external images", title=f"{site} (n={len(g)})")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OVERLAP_DIR, "best_corr_histogram.png"), dpi=110)
    plt.close(fig)

    # pairs to inspect: top 10 overall + every pair >= 0.95
    to_check = pd.concat([best.nlargest(10, "best_corr"), best[best.best_corr >= AUTO_EXCLUDE_CORR]]) \
        .drop_duplicates(["site", "file"]).sort_values("best_corr", ascending=False)
    to_check.to_csv(os.path.join(OVERLAP_DIR, "pairs_to_inspect.csv"), index=False)
    per_page = 10
    for page in range(int(np.ceil(len(to_check) / per_page))):
        chunk = to_check.iloc[page * per_page:(page + 1) * per_page]
        fig, axes = plt.subplots(len(chunk), 2, figsize=(6, 3 * len(chunk)), squeeze=False)
        for k, (_, r) in enumerate(chunk.iterrows()):
            axes[k, 0].imshow(Image.open(r.png_path), cmap="gray", vmin=0, vmax=255)
            axes[k, 0].set_title(f"{r.site} {r.file} ({'TB' if r.label else 'Normal'})", fontsize=8)
            axes[k, 1].imshow(Image.open(r.match_path).convert("L"), cmap="gray", vmin=0, vmax=255)
            axes[k, 1].set_title(f"training {os.path.basename(r.match_path)} ({'TB' if r.match_label else 'Normal'}, "
                                 f"{r.match_split})\ncorr {r.best_corr:.4f}", fontsize=8)
            for ax in axes[k]:
                ax.set_xticks([])
                ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(os.path.join(OVERLAP_DIR, f"pairs_page{page + 1}.png"), dpi=90)
        plt.close(fig)
    print(f"{len(to_check)} pairs to inspect -> {OVERLAP_DIR}/pairs_page*.png")


# ---------------------------------------------------------------- crop-robust overlap search
CROP_MARGINS = (0.0, 0.05, 0.10)  # per edge; all 81 combinations of (top, bottom, left, right)


def crop_vectors(img256):
    """sim_vector of every crop of a 256x256 grayscale PIL image (crops are re-stretched to 32x32)."""
    from itertools import product

    from src.pipeline import data_audit as da
    w, h = img256.size
    combos = list(product(CROP_MARGINS, repeat=4))
    vecs = [da.sim_vector(img256.crop((int(lf * w), int(t * h), w - int(rt * w), h - int(b * h))))
            for t, b, lf, rt in combos]
    return combos, np.stack(vecs)


def stage_cropsearch():
    from src.pipeline import data_audit as da
    os.makedirs(OVERLAP_DIR, exist_ok=True)
    splits, train_vecs = training_sim_vectors()
    inv = pd.read_csv(os.path.join(OUT_DIR, "inventory.csv"))

    # Positive control: a cropped, re-saved copy of an external image must be found by the search.
    ctrl = inv.sample(20, random_state=SEED)
    ctrl_rows = []
    for _, r in ctrl.iterrows():
        with Image.open(r.path) as im:
            g = Image.fromarray(to_gray8(im)[0])
        w, h = g.size
        copy = g.crop((int(.06 * w), int(.08 * h), w - int(.06 * w), h - int(.07 * h))).resize(PREP_SIZE, Image.LANCZOS)
        buf = io.BytesIO()
        copy.convert("RGB").save(buf, format="JPEG", quality=75)
        copy_vec = da.sim_vector(Image.open(buf).convert("L"))
        _, vecs = crop_vectors(Image.open(r.png_path).convert("L"))
        c = vecs @ copy_vec / vecs.shape[1]
        ctrl_rows.append({"file": r.file, "no_crop_corr": float(c[0]), "crop_search_corr": float(c.max())})
    ctrl_df = pd.DataFrame(ctrl_rows)

    rows = []
    for _, r in inv.iterrows():
        combos, vecs = crop_vectors(Image.open(r.png_path).convert("L"))
        corr = vecs @ train_vecs.T / vecs.shape[1]                     # (81 crops, 2982)
        ci, tj = np.unravel_index(np.argmax(corr), corr.shape)
        m = splits.iloc[tj]
        rows.append({"site": r.site, "file": r.file, "label": r.label, "png_path": r.png_path,
                     "no_crop_best": float(corr[0].max()), "crop_search_best": float(corr[ci, tj]),
                     "best_crop_tblr": combos[ci], "match_path": m.path, "match_label": int(m.label),
                     "match_split": m.split})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OVERLAP_DIR, "crop_search.csv"), index=False)
    summary = {"crops_per_image": len(CROP_MARGINS) ** 4, "margins_per_edge": CROP_MARGINS,
               "positive_control": {"n": len(ctrl_df), "no_crop_corr_median": float(ctrl_df.no_crop_corr.median()),
                                    "no_crop_corr_min": float(ctrl_df.no_crop_corr.min()),
                                    "crop_search_corr_median": float(ctrl_df.crop_search_corr.median()),
                                    "crop_search_corr_min": float(ctrl_df.crop_search_corr.min())}}
    for site, g in res.groupby("site"):
        c = g["crop_search_best"]
        summary[site] = {"median": float(c.median()), "p95": float(c.quantile(.95)), "max": float(c.max()),
                         **{f">={t}": int((c >= t).sum()) for t in (0.90, 0.95, 0.99)},
                         "median_gain_vs_no_crop": float((c - g.no_crop_best).median())}
    with open(os.path.join(OVERLAP_DIR, "crop_search_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=1))

    top = res.nlargest(10, "crop_search_best")
    fig, axes = plt.subplots(len(top), 2, figsize=(6, 3 * len(top)), squeeze=False)
    for k, (_, r) in enumerate(top.iterrows()):
        axes[k, 0].imshow(Image.open(r.png_path), cmap="gray", vmin=0, vmax=255)
        axes[k, 0].set_title(f"{r.site} {r.file} ({'TB' if r.label else 'Normal'})\ncrop t/b/l/r {r.best_crop_tblr}",
                             fontsize=8)
        axes[k, 1].imshow(Image.open(r.match_path).convert("L"), cmap="gray", vmin=0, vmax=255)
        axes[k, 1].set_title(f"training {os.path.basename(r.match_path)} ({'TB' if r.match_label else 'Normal'})\n"
                             f"corr {r.crop_search_best:.4f}", fontsize=8)
        for ax in axes[k]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(OVERLAP_DIR, "crop_search_top10.png"), dpi=90)
    plt.close(fig)


# ---------------------------------------------------------------- evaluation (frozen models, val-tuned thresholds)
EVAL_DIR = os.path.join(OUT_DIR, "evaluation")
PRED_DIR = os.path.join(OUT_DIR, "predictions")
MODELS = ["resnet", "vgg", "efficientnet"]  # primary first
NAMES = {"resnet": "ResNet50", "vgg": "VGG16", "efficientnet": "EfficientNetB0"}
SMALL_GROUP = 30   # fewer images than this in a class -> flagged as small
MIN_AUC_CLASS = 10  # AUC only reported for a subgroup with at least this many of each class


def predict_paths(model, config, paths, labels):
    """Same tf.data path as training/evaluation (data_loader.make_dataset -> load_image -> decode_and_resize)."""
    from src.components.data_loader import make_dataset
    df = pd.DataFrame({"path": list(paths), "label": list(labels)})
    return model.predict(make_dataset(config, df=df), verbose=0).ravel()


def load_or_predict(config, inv):
    """Predictions for every model x site x variant (cached CSVs). Includes the parity check."""
    from src.components import inference
    os.makedirs(PRED_DIR, exist_ok=True)
    preds = {}
    for m in MODELS:
        needed = [os.path.join(PRED_DIR, f"{m}_{v}.csv") for v in ("png", "jpg")]
        if all(os.path.exists(p) for p in needed):
            for v, p in zip(("png", "jpg"), needed):
                preds[(m, v)] = pd.read_csv(p)
            continue
        model = inference.load_model(m)
        # Parity (CLAUDE.md rule): this path must reproduce the saved internal test predictions.
        saved = pd.read_csv(os.path.join(INTERNAL_PRED_DIR, f"{m}_test.csv")).head(32)  # same model's evaluate.py output
        diff = np.abs(predict_paths(model, config, saved.path, saved.label) - saved.prob.values).max()
        assert diff < 1e-4, f"{m}: parity check failed ({diff})"
        print(f"[evaluate] {m}: parity with saved internal test predictions OK (max diff {diff:.1e})")
        for v, p in zip(("png", "jpg"), needed):
            prob = predict_paths(model, config, inv[f"{v}_path"], inv["label"])
            out = inv[["site", "file", "label", "sex", "age", "age_band"]].assign(prob=prob)
            out.to_csv(p, index=False)
            preds[(m, v)] = out
    return preds


def site_frames(df):
    """Per-site views plus the pooled secondary view."""
    return {"shenzhen": df[df.site == "shenzhen"], "montgomery": df[df.site == "montgomery"], "pooled": df}


def evaluate_block(y, p, thr, n_boot, seed):
    from src.pipeline import evaluate_clinical as ec
    idx = ec.stratified_boot_indices(y, n_boot, seed)
    return {"n": int(len(y)), "n_normal": int((y == 0).sum()), "n_tb": int(y.sum()),
            "roc_auc": ec.auc_ci(y, p, idx), "brier": ec.brier_ci(y, p, idx),
            "at_tuned": ec.metrics_at(y, p, thr, idx), "at_0.5": ec.metrics_at(y, p, 0.5, idx),
            "mean_p_normal": float(p[y == 0].mean()), "mean_p_tb": float(p[y == 1].mean())}, idx


def subgroup_rows(df, m, thr, n_boot, seed):
    from src.pipeline import evaluate_clinical as ec
    rows = []
    for site in ("shenzhen", "montgomery"):
        s = df[df.site == site]
        for var, col in (("sex", "sex"), ("age band", "age_band")):
            for grp, g in s.groupby(s[col].fillna("unknown")):
                y, p = g.label.values, g.prob.values
                n0, n1 = int((y == 0).sum()), int((y == 1).sum())
                rng = np.random.default_rng(seed)
                row = {"model": m, "site": site, "variable": var, "group": grp, "n_normal": n0, "n_tb": n1,
                       "small": min(n0, n1) < SMALL_GROUP}
                for name, cls, pred in (("sensitivity", 1, lambda q: q >= thr), ("specificity", 0, lambda q: q < thr)):
                    q = p[y == cls]
                    if len(q):
                        boot = pred(q[rng.integers(0, len(q), (n_boot, len(q)))]).mean(axis=1)
                        row.update({name: float(pred(q).mean()), f"{name}_ci_low": float(np.percentile(boot, 2.5)),
                                    f"{name}_ci_high": float(np.percentile(boot, 97.5))})
                if n0 >= MIN_AUC_CLASS and n1 >= MIN_AUC_CLASS:
                    a = ec.auc_ci(y, p, ec.stratified_boot_indices(y, n_boot, seed))
                    row.update({"auc": a["value"], "auc_ci_low": a["ci_low"], "auc_ci_high": a["ci_high"]})
                rows.append(row)
    return rows


def stage_evaluate():
    from sklearn.metrics import roc_auc_score

    from src.pipeline import evaluate_clinical as ec
    os.makedirs(EVAL_DIR, exist_ok=True)
    config = load_config()
    inv = pd.read_csv(os.path.join(OUT_DIR, "inventory.csv"))
    thresholds = {m: v["threshold"] for m, v in
                  json.load(open(os.path.join(CLINICAL_DIR, "thresholds.json")))["models"].items()}
    internal = json.load(open(os.path.join(CLINICAL_DIR, "metrics_test.json")))["models"]
    preds = load_or_predict(config, inv)

    # Overlap exclusions: visually confirmed (main) and automatic >= 0.95 (robustness). Both empty here.
    confirmed = set()
    for f in ("visual_verdicts.csv", "crop_search_visual_verdicts.csv"):
        v = pd.read_csv(os.path.join(OVERLAP_DIR, f))
        confirmed |= set(zip(v.site[v.confirmed_overlap], v.file[v.confirmed_overlap]))
    cs = pd.read_csv(os.path.join(OVERLAP_DIR, "crop_search.csv"))
    auto = set(zip(cs.site[cs.crop_search_best >= AUTO_EXCLUDE_CORR], cs.file[cs.crop_search_best >= AUTO_EXCLUDE_CORR]))
    exclusions = {"confirmed_overlaps": sorted(confirmed), "auto_excluded_>=0.95": sorted(auto)}

    results, cal_rows, sub_rows, oracle_rows, jpg_rows = {}, [], [], [], []
    for m in MODELS:
        for v in ("png", "jpg"):
            df = preds[(m, v)]
            for site, g in site_frames(df).items():
                y, p = g.label.values, g.prob.values
                block, idx = evaluate_block(y, p, thresholds[m], ec.N_BOOT, ec.SEED)
                results[f"{m}/{v}/{site}"] = block
                if v == "png":
                    cal_rows += [{"model": m, "site": site, **b} for b in ec.calibration_bins(y, p)]
                    o = ec.tune_threshold(y, p, ec.TARGET_SENSITIVITY)
                    oracle_rows.append({"model": m, "site": site, "val_tuned_threshold": thresholds[m],
                                        "oracle_threshold": o["threshold"], "oracle_sensitivity": o["val_sensitivity"],
                                        "oracle_specificity": o["val_specificity"],
                                        "specificity_at_val_tuned": block["at_tuned"]["specificity"]["value"],
                                        "sensitivity_at_val_tuned": block["at_tuned"]["sensitivity"]["value"]})
                    pj = preds[(m, "jpg")].set_index(["site", "file"]).loc[list(zip(g.site, g.file)), "prob"].values
                    diffs = [roc_auc_score(y[i], pj[i]) - roc_auc_score(y[i], p[i]) for i in idx]
                    jpg_rows.append({"model": m, "site": site, "auc_png": roc_auc_score(y, p),
                                     "auc_jpg": roc_auc_score(y, pj), "diff_jpg_minus_png": roc_auc_score(y, pj) - roc_auc_score(y, p),
                                     "diff_ci_low": float(np.percentile(diffs, 2.5)),
                                     "diff_ci_high": float(np.percentile(diffs, 97.5))})
        sub_rows += subgroup_rows(preds[(m, "png")], m, thresholds[m], ec.N_BOOT, ec.SEED)

    with open(os.path.join(EVAL_DIR, "metrics_external.json"), "w") as f:
        json.dump({"thresholds": thresholds, "exclusions": exclusions, "bootstrap": {"n": ec.N_BOOT, "seed": ec.SEED},
                   "results": results}, f, indent=2)
    pd.DataFrame(cal_rows).to_csv(os.path.join(EVAL_DIR, "calibration_bins.csv"), index=False)
    sub = pd.DataFrame(sub_rows)
    sub.to_csv(os.path.join(EVAL_DIR, "subgroups.csv"), index=False)
    oracle = pd.DataFrame(oracle_rows)
    oracle.to_csv(os.path.join(EVAL_DIR, "oracle_thresholds.csv"), index=False)
    jpg = pd.DataFrame(jpg_rows)
    jpg.to_csv(os.path.join(EVAL_DIR, "png_vs_jpeg_auc.csv"), index=False)

    # ----- hypotheses (rules fixed in docs/audit.md before predictions existed)
    hyp = {"H1": {}, "H4": {}}
    for m in MODELS:
        for site in ("shenzhen", "montgomery"):
            r = results[f"{m}/png/{site}"]["roc_auc"]
            hyp["H1"][f"{m}/{site}"] = {"external_auc": r["value"], "ci_high": r["ci_high"],
                                        "internal_auc": internal[m]["roc_auc"]["value"],
                                        "supported": r["ci_high"] < internal[m]["roc_auc"]["value"]}
            j = jpg[(jpg.model == m) & (jpg.site == site)].iloc[0]
            hyp["H4"][f"{m}/{site}"] = {"diff": j.diff_jpg_minus_png, "ci": [j.diff_ci_low, j.diff_ci_high],
                                        "supported": bool(j.diff_jpg_minus_png > 0)}
    spec = {s: results[f"resnet/png/{s}"]["at_tuned"]["specificity"]["value"] for s in ("shenzhen", "montgomery")}
    auc = {s: results[f"resnet/png/{s}"]["roc_auc"]["value"] for s in ("shenzhen", "montgomery")}
    hyp["H2"] = {"montgomery_resnet_specificity": spec["montgomery"], "supported": spec["montgomery"] < 0.80}
    hyp["H3"] = {"testable": bool(confirmed), "note": "no confirmed overlaps (whole-image and crop search)"}
    hyp["H5"] = {"resnet_specificity": spec, "resnet_auc": auc,
                 "supported": all(spec[s] < 0.80 and auc[s] - spec[s] > 0 for s in spec)}
    with open(os.path.join(EVAL_DIR, "hypotheses.json"), "w") as f:
        json.dump(hyp, f, indent=2, default=bool)

    # ----- summary tables (values truncated, never rounded up)
    t3, fmt = ec.t3, ec.fmt
    lines = ["# External validation results (frozen models, val-tuned thresholds unchanged)", "",
             f"Thresholds: {', '.join(f'{NAMES[m]} {thresholds[m]:.4f}' for m in MODELS)}. 95% CIs: stratified "
             f"bootstrap, {ec.N_BOOT} resamples, seed {ec.SEED}. Values truncated to 3 decimals. Primary variant: "
             f"lossless PNG. Overlap exclusions: confirmed {len(confirmed)}, automatic (>= 0.95) {len(auto)} -> "
             "results with and without exclusions are identical.", "",
             "## Main table (PNG, at the val-tuned threshold)", "",
             "| model | data | n (N/TB) | ROC-AUC | sensitivity | specificity | PPV | NPV | F1 | Brier |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for m in MODELS:
        it = internal[m]
        lines.append(f"| {NAMES[m]} | internal test | 448 (77/371) | {fmt(it['roc_auc'])} | "
                     + " | ".join(fmt(it["at_tuned"][k]) for k in ec.METRICS) + f" | {fmt(it['brier'])} |")
        for site in ("shenzhen", "montgomery", "pooled"):
            r = results[f"{m}/png/{site}"]
            label = site if site != "pooled" else "pooled (secondary)"
            lines.append(f"| {NAMES[m]} | {label} | {r['n']} ({r['n_normal']}/{r['n_tb']}) | {fmt(r['roc_auc'])} | "
                         + " | ".join(fmt(r["at_tuned"][k]) for k in ec.METRICS) + f" | {fmt(r['brier'])} |")
    lines += ["", "## At threshold 0.5 (for comparison)", "",
              "| model | data | sensitivity | specificity | mean P(TB) Normal / TB |", "|---|---|---|---|---|"]
    for m in MODELS:
        for site in ("shenzhen", "montgomery"):
            r = results[f"{m}/png/{site}"]
            lines.append(f"| {NAMES[m]} | {site} | {fmt(r['at_0.5']['sensitivity'])} | {fmt(r['at_0.5']['specificity'])}"
                         f" | {t3(r['mean_p_normal'])} / {t3(r['mean_p_tb'])} |")
    lines += ["", "## PNG vs JPEG (quality 75, 4:2:0) ROC-AUC", "",
              "| model | site | AUC PNG | AUC JPEG | JPEG - PNG (95% CI, paired) |", "|---|---|---|---|---|"]
    for _, j in jpg.iterrows():
        lines.append(f"| {NAMES[j.model]} | {j.site} | {t3(j.auc_png)} | {t3(j.auc_jpg)} | "
                     f"{j.diff_jpg_minus_png:+.4f} ({j.diff_ci_low:+.4f} to {j.diff_ci_high:+.4f}) |")
    lines += ["", "## Oracle analysis (NOT a result: threshold chosen on the external data itself)", "",
              "| model | site | val-tuned threshold | sens / spec at val-tuned | oracle threshold for 95% sens | "
              "spec at oracle |", "|---|---|---|---|---|---|"]
    for _, o in oracle.iterrows():
        lines.append(f"| {NAMES[o.model]} | {o.site} | {o.val_tuned_threshold:.4f} | {t3(o.sensitivity_at_val_tuned)} / "
                     f"{t3(o.specificity_at_val_tuned)} | {o.oracle_threshold:.4f} | {t3(o.oracle_specificity)} |")
    lines += ["", "## Subgroups (ResNet50, PNG, val-tuned threshold; * = fewer than 30 images in a class)", "",
              "| site | variable | group | n (N/TB) | sensitivity | specificity | AUC |", "|---|---|---|---|---|---|---|"]
    for _, s in sub[sub.model == "resnet"].iterrows():
        def ci_txt(k):
            return "n/a" if pd.isna(s.get(k)) else f"{t3(s[k])} ({t3(s[k + '_ci_low'])}–{t3(s[k + '_ci_high'])})"
        lines.append(f"| {s.site} | {s.variable} | {s.group}{'*' if s.small else ''} | {s.n_normal + s.n_tb} "
                     f"({s.n_normal}/{s.n_tb}) | {ci_txt('sensitivity')} | {ci_txt('specificity')} | {ci_txt('auc')} |")
    lines += ["", "## Pre-registered hypotheses", "", "```", json.dumps(hyp, indent=1, default=bool), "```"]
    with open(os.path.join(EVAL_DIR, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# ---------------------------------------------------------------- Grad-CAM vs real lung masks (Montgomery)
GRADCAM_DIR = os.path.join(OUT_DIR, "gradcam")
CAM_MODELS = ["resnet", "vgg"]
MASK_ROOT = os.path.join(SITES["montgomery"]["root"], "ManualMask")


def load_lung_mask(file):
    """Left + right ManualMask combined, at the original image size (bool)."""
    masks = [np.asarray(Image.open(os.path.join(MASK_ROOT, side, file)).convert("L")) > 0
             for side in ("leftMask", "rightMask")]
    return masks[0] | masks[1]


def mask_to_model_grid(mask, size=None):
    """Same geometry as the image path (original -> 256x256 -> model size), nearest-neighbour so it stays binary."""
    import tensorflow as tf
    size = size or tuple(load_config()["training"]["img_size"])
    m256 = Image.fromarray(mask.astype(np.uint8) * 255).resize(PREP_SIZE, Image.NEAREST)
    m = tf.image.resize(np.asarray(m256)[..., None], size, method="nearest").numpy()[..., 0]
    return m > 127


def mask_alignment_figure(inv, path):
    mont = inv[inv.site == "montgomery"]
    picks = pd.concat([mont[mont.label == 0].head(2), mont[mont.label == 1].head(2)])
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.2))
    for k, (_, r) in enumerate(picks.iterrows()):
        mask = load_lung_mask(r.file)
        with Image.open(r.path) as im:
            orig = im.convert("L")
            small_orig = orig.resize((orig.width // 8, orig.height // 8))
        small_mask = Image.fromarray(mask.astype(np.uint8) * 255).resize(small_orig.size, Image.NEAREST)
        x = model_input(open(r.png_path, "rb").read())[..., 0]
        grid = mask_to_model_grid(mask)
        for row, (img, msk, title) in enumerate(((small_orig, np.asarray(small_mask) > 127, f"original {orig.size}"),
                                                (x, grid, "model input 224x224"))):
            ax = axes[row, k]
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            ax.contour(msk, levels=[0.5], colors="lime", linewidths=1.2)
            ax.set_title(f"{r.file} ({'TB' if r.label else 'Normal'})\n{title}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Lung mask (green contour) on the original and on the model input: geometry check", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=100)
    plt.close(fig)


def heatmaps_in_input_frame(maps, size, crop_fraction=None):
    """(B, h, w) Grad-CAM maps -> (B, H, W) in the 224x224 *input* frame.
    Without an internal crop this is the plain upsample (goal 5/9 behaviour). With the goal 10 CentralCrop,
    the maps describe the cropped field, so they are upsampled to the crop and placed in its position (zero
    outside), which keeps them aligned with the lung masks and the border geometry."""
    from src.components.gradcam import upsample
    from src.components.model_builder import crop_pixels
    if not crop_fraction:
        return upsample(maps, size)
    ch, cw = crop_pixels(size[0], crop_fraction), crop_pixels(size[1], crop_fraction)
    inner = upsample(maps, (size[0] - 2 * ch, size[1] - 2 * cw))
    out = np.zeros((len(maps),) + tuple(size), dtype=np.float32)
    out[:, ch:size[0] - ch, cw:size[1] - cw] = inner
    return out


def stage_gradcam():
    from scipy.stats import wilcoxon

    from src.components import inference
    from src.components.data_loader import load_image
    from src.components.gradcam import gradcam_signed, overlay, to_heatmap
    from src.components.model_builder import LAST_CONV_LAYER
    from src.pipeline.gradcam_analysis import fractions, region_masks
    import tensorflow as tf

    os.makedirs(GRADCAM_DIR, exist_ok=True)
    config = load_config()
    size = tuple(config["training"]["img_size"])
    inv = pd.read_csv(os.path.join(OUT_DIR, "inventory.csv"))
    thresholds = {m: v["threshold"] for m, v in
                  json.load(open(os.path.join(CLINICAL_DIR, "thresholds.json")))["models"].items()}
    border_masks, border_px = region_masks(size)
    centre_box = ~border_masks["border"]
    mask_alignment_figure(inv, os.path.join(GRADCAM_DIR, "mask_alignment.png"))

    lung = {r.file: mask_to_model_grid(load_lung_mask(r.file), size) for _, r in inv[inv.site == "montgomery"].iterrows()}
    heat_store, rows = {}, []
    for m in CAM_MODELS:
        model = inference.load_model(m)
        layer = LAST_CONV_LAYER[m]
        crop = next((lyr.fraction for lyr in model.layers if lyr.name == "central_crop"), None)  # goal 10
        saved = pd.read_csv(os.path.join(PRED_DIR, f"{m}_png.csv")).set_index(["site", "file"])
        for start in range(0, len(inv), 16):
            chunk = inv.iloc[start:start + 16]
            x = np.stack([load_image(tf.constant(p), size).numpy() for p in chunk.png_path])
            cams, _, prob = gradcam_signed(x, model, [layer])
            ref = saved.loc[list(zip(chunk.site, chunk.file)), "prob"].values
            assert np.abs(prob - ref).max() < 1e-4, f"{m}: Grad-CAM forward pass differs from saved predictions"
            pred = prob >= thresholds[m]
            tb_map, tb_empty = to_heatmap(cams[layer])
            pr_map, pr_empty = to_heatmap(cams[layer], np.where(pred, 1.0, -1.0))
            tb224 = heatmaps_in_input_frame(tb_map, size, crop)
            pr224 = heatmaps_in_input_frame(pr_map, size, crop)
            fr_tb, fr_pr = fractions(tb224, border_masks), fractions(pr224, border_masks)
            for i, (_, r) in enumerate(chunk.iterrows()):
                row = {"model": m, "site": r.site, "file": r.file, "label": int(r.label), "prob": float(prob[i]),
                       "pred": int(pred[i]), "correct": bool(pred[i] == r.label),
                       "tb_border": fr_tb["border"][i], "pred_border": fr_pr["border"][i],
                       "tb_empty": bool(tb_empty[i]), "pred_empty": bool(pr_empty[i])}
                if r.site == "montgomery":
                    lm = lung[r.file]
                    area = float(lm.mean())
                    for kind, hm, empty in (("tb", tb224[i], tb_empty[i]), ("pred", pr224[i], pr_empty[i])):
                        inside = float((hm * lm).sum() / hm.sum()) if not empty else np.nan
                        row[f"{kind}_lung_energy"] = inside
                    row.update(lung_area=area, lung_area_in_centre_box=float((lm & centre_box).sum() / lm.sum()))
                rows.append(row)
                heat_store[(m, r.site, r.file)] = pr224[i]  # input-frame map (identical overlay without a crop)
        print(f"[gradcam] {m}: done")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(GRADCAM_DIR, "attention_external.csv"), index=False)

    # ----- H6 and lung-mask analysis (Montgomery)
    mont = df[df.site == "montgomery"].copy()
    mont["tb_minus_area"] = mont.tb_lung_energy - mont.lung_area
    mont["pred_minus_area"] = mont.pred_lung_energy - mont.lung_area
    h6, lung_rows = {}, []
    for m in CAM_MODELS:
        g = mont[mont.model == m]
        d = g["tb_minus_area"].dropna()
        h6[m] = {"n": int(len(d)), "empty_tb_maps": int(g.tb_empty.sum()),
                 "median_lung_energy": float(g.tb_lung_energy.median()), "median_lung_area": float(g.lung_area.median()),
                 "median_diff": float(d.median()), "wilcoxon_p": float(wilcoxon(d).pvalue),
                 "supported": bool(d.median() <= 0)}
        groups = {"all": g, "Normal": g[g.label == 0], "TB": g[g.label == 1], "correct": g[g.correct],
                  "wrong": g[~g.correct]}
        for name, gg in groups.items():
            for kind in ("tb", "pred"):
                v = gg[f"{kind}_lung_energy"].dropna()
                lung_rows.append({"model": m, "map": "TB-evidence" if kind == "tb" else "predicted-class",
                                  "group": name, "n": len(v),
                                  "median_lung_energy": float(v.median()) if len(v) else np.nan,
                                  "median_lung_area": float(gg.lung_area.median()) if len(gg) else np.nan,
                                  "median_energy_minus_area": float(gg[f"{kind}_minus_area"].median()) if len(v) else np.nan,
                                  "share_of_images_energy_gt_area": float((gg[f"{kind}_minus_area"] > 0).mean()) if len(v) else np.nan})
    h6["supported_overall"] = all(h6[m]["supported"] for m in CAM_MODELS)
    h6["lung_area_inside_goal4_centre_box"] = {"median": float(mont.lung_area_in_centre_box.median()),
                                               "min": float(mont.lung_area_in_centre_box.min())}
    lung_tab = pd.DataFrame(lung_rows)
    lung_tab.to_csv(os.path.join(GRADCAM_DIR, "lung_mask_attention.csv"), index=False)

    # ----- border energy: external vs internal test (TB-evidence map, same layer, same 12% border)
    border_rows = []
    for m in CAM_MODELS:
        internal_path = f"artifacts/gradcam/attention_{m}.csv"  # goal 5 (default models only)
        internal = pd.read_csv(internal_path) if CONFIG_TAG == "default" and os.path.exists(internal_path) else None
        col = f"{LAST_CONV_LAYER[m]}__tb__border"
        for lab, name in ((0, "Normal"), (1, "TB")):
            if internal is not None:
                border_rows.append({"model": m, "data": "internal test", "class": name,
                                    "median_tb_border": float(internal.loc[internal.label == lab, col].median())})
            for site in ("shenzhen", "montgomery"):
                g = df[(df.model == m) & (df.site == site) & (df.label == lab)]
                border_rows.append({"model": m, "data": site, "class": name,
                                    "median_tb_border": float(g.tb_border.median())})
    border = pd.DataFrame(border_rows)
    border.to_csv(os.path.join(GRADCAM_DIR, "border_energy_vs_internal.csv"), index=False)
    with open(os.path.join(GRADCAM_DIR, "h6.json"), "w") as f:
        json.dump({"uniform_border_share": float(border_masks["border"].mean()), "H6": h6}, f, indent=2)

    # ----- galleries (ResNet50, predicted-class maps, val-tuned threshold)
    res = df[df.model == "resnet"].merge(inv[["site", "file", "png_path"]], on=["site", "file"])
    for site in ("shenzhen", "montgomery"):
        s = res[res.site == site]
        cats = [("TB missed (FN)", s[(s.label == 1) & ~s.correct].nsmallest(6, "prob")),
                ("Normal flagged TB (FP)", s[(s.label == 0) & ~s.correct].nlargest(6, "prob")),
                ("TB correct (TP)", s[(s.label == 1) & s.correct].nlargest(6, "prob")),
                ("Normal correct (TN)", s[(s.label == 0) & s.correct].nsmallest(6, "prob"))]
        fig, axes = plt.subplots(4, 6, figsize=(16, 11.5))
        for row_i, (cat, g) in enumerate(cats):
            n_total = {"TB missed (FN)": ((s.label == 1) & ~s.correct).sum(),
                       "Normal flagged TB (FP)": ((s.label == 0) & ~s.correct).sum(),
                       "TB correct (TP)": ((s.label == 1) & s.correct).sum(),
                       "Normal correct (TN)": ((s.label == 0) & s.correct).sum()}[cat]
            axes[row_i, 0].set_ylabel(f"{cat}\n{n_total} images in total", fontsize=9)
            for col_i in range(6):
                ax = axes[row_i, col_i]
                ax.set_xticks([])
                ax.set_yticks([])
                if col_i >= len(g):
                    if col_i == 0:
                        ax.text(0.5, 0.5, "none", ha="center", va="center", fontsize=10)
                    continue
                r = g.iloc[col_i]
                x = model_input(open(r.png_path, "rb").read())
                ax.imshow(overlay(x, heat_store[("resnet", site, r.file)]))
                if site == "montgomery":
                    ax.contour(lung[r.file], levels=[0.5], colors="white", linewidths=0.8)
                ax.add_patch(plt.Rectangle((border_px, border_px), size[1] - 2 * border_px, size[0] - 2 * border_px,
                                           fill=False, ls="--", lw=0.6, ec="white"))
                ax.set_title(f"{r.file}\nP(TB)={r.prob:.3f}", fontsize=7)
        fig.suptitle(f"{site.capitalize()}: ResNet50 predicted-class Grad-CAM at the val-tuned threshold "
                     f"({thresholds['resnet']:.4f}). Dashed box = 12% border"
                     + ("; white contour = true lung mask" if site == "montgomery" else ""), fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(os.path.join(GRADCAM_DIR, f"gallery_{site}.png"), dpi=90)
        plt.close(fig)

    print(json.dumps({"H6": h6}, indent=1))
    print(lung_tab.to_string(index=False))
    print(border.pivot_table(index=["model", "class"], columns="data", values="median_tb_border").round(3).to_string())


def resolution_ceiling(mask224, grid, crop_fraction=None):
    """Lung-energy share of a 'perfect' map at Grad-CAM resolution: the mask averaged onto a grid x grid map
    (the share of lung in each cell), then upsampled exactly like the heatmaps (gradcam.upsample)."""
    import tensorflow as tf

    from src.components.gradcam import upsample
    from src.components.model_builder import crop_pixels
    h, w = mask224.shape
    field = mask224
    if crop_fraction:  # the model sees only the cropped field, stretched back to h x w
        c = crop_pixels(h, crop_fraction)
        field = tf.image.resize(mask224[c:h - c, c:w - c].astype(np.uint8)[..., None] * 255, (h, w),
                                method="nearest").numpy()[..., 0] > 127
    coarse = field.reshape(grid, h // grid, grid, w // grid).mean(axis=(1, 3)).astype(np.float32)
    up = heatmaps_in_input_frame(coarse[None], (h, w), crop_fraction)[0] if crop_fraction \
        else upsample(coarse[None], (h, w))[0]
    return float((up * mask224).sum() / up.sum())


CAM_GRID = {"resnet": 7, "vgg": 14, "efficientnet": 7}  # spatial size of LAST_CONV_LAYER at 224x224


def stage_ceiling():
    """Robustness check for H6 / S3 (not a hypothesis): best achievable lung-energy share at Grad-CAM resolution."""
    inv = pd.read_csv(os.path.join(OUT_DIR, "inventory.csv"))
    config = load_config()
    size = tuple(config["training"]["img_size"])
    crop = config.get("model_options", {}).get("crop_fraction")
    observed = json.load(open(os.path.join(GRADCAM_DIR, "h6.json")))["H6"]
    rows = []
    for _, r in inv[inv.site == "montgomery"].iterrows():
        m = mask_to_model_grid(load_lung_mask(r.file), size)
        row = {"file": r.file, "label": r.label, "lung_area": float(m.mean())}
        for g in sorted({CAM_GRID[k] for k in CAM_MODELS}):
            row[f"ceiling_{g}x{g}"] = resolution_ceiling(m, g, crop)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(GRADCAM_DIR, "h6_resolution_ceiling.csv"), index=False)
    out = {"method": "mask averaged onto the Grad-CAM grid (of the cropped field, if the model crops), upsampled "
                     "like the heatmaps, lung-energy share computed as for the heatmaps",
           "crop_fraction": crop, "median_lung_area": float(df.lung_area.median()),
           "models": {k: {"grid": CAM_GRID[k], "median_ceiling": float(df[f"ceiling_{CAM_GRID[k]}x{CAM_GRID[k]}"].median()),
                          "observed_median": observed[k]["median_lung_energy"]} for k in CAM_MODELS}}
    for k, legacy in (("resnet", "resnet_conv5_7x7"), ("vgg", "vgg_block5_14x14")):  # goal 9 key names
        if k in out["models"]:
            out[legacy] = {"median_ceiling": out["models"][k]["median_ceiling"],
                           "observed_median": out["models"][k]["observed_median"]}
    with open(os.path.join(GRADCAM_DIR, "h6_resolution_ceiling.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=1))


# ---------------------------------------------------------------- shareable outputs (NLM data-use terms)
PUBLIC_PRED_DIR = os.path.join(OUT_DIR, "predictions_public")
PUBLIC_COLUMNS = ["site", "file", "label", "prob"]  # no age, sex or findings


def stage_public():
    """Per-image predictions without clinical readings, for the repository. Full files stay local."""
    os.makedirs(PUBLIC_PRED_DIR, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(PRED_DIR, "*.csv"))):
        out = os.path.join(PUBLIC_PRED_DIR, os.path.basename(f))
        pd.read_csv(f)[PUBLIC_COLUMNS].to_csv(out, index=False)
        print(f"[public] {out}")


def stage_h6plot():
    """Image-free figure: per-image lung-energy share on Montgomery vs lung area and resolution ceiling."""
    df = pd.read_csv(os.path.join(GRADCAM_DIR, "attention_external.csv"))
    df = df[df.site == "montgomery"]
    ceiling = json.load(open(os.path.join(GRADCAM_DIR, "h6_resolution_ceiling.json")))
    if "models" in ceiling:
        ceil = {k: v["median_ceiling"] for k, v in ceiling["models"].items()}
    else:  # goal 9 file format
        ceil = {"resnet": ceiling["resnet_conv5_7x7"]["median_ceiling"],
                "vgg": ceiling["vgg_block5_14x14"]["median_ceiling"]}
    grid = {k: f"{CAM_GRID[k]}×{CAM_GRID[k]}" for k in CAM_MODELS}
    area = ceiling["median_lung_area"]
    ink, muted, grid_col = "#0b0b0b", "#52514e", "#e4e3df"
    colours = {0: "#2a78d6", 1: "#eb6834"}   # reference palette, categorical slots 1-2
    markers = {0: "o", 1: "^"}               # secondary encoding: identity is not colour-only
    rng = np.random.default_rng(SEED)

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for i, m in enumerate(CAM_MODELS):
        for lab, dx in ((0, -0.17), (1, 0.17)):
            v = df[(df.model == m) & (df.label == lab)]["tb_lung_energy"].dropna().values
            x = i + dx + rng.uniform(-0.07, 0.07, len(v))
            ax.scatter(x, v, s=30, marker=markers[lab], color=colours[lab], alpha=0.75, linewidths=0.6,
                       edgecolors="#fcfcfb", zorder=3, label=("Normal" if lab == 0 else "TB") if i == 0 else None)
            med = float(np.median(v))
            ax.plot([i + dx - 0.12, i + dx + 0.12], [med, med], color=ink, lw=2, zorder=4)
        ax.plot([i - 0.38, i + 0.38], [ceil[m], ceil[m]], color=muted, lw=1.2, ls=(0, (4, 3)), zorder=2)
        ax.text(i + 0.40, ceil[m], f"resolution ceiling {ceil[m]:.3f}\n(perfect lung map at {grid[m]})",
                va="center", ha="left", fontsize=8, color=muted)
    ax.axhline(area, color=muted, lw=1.2, ls=(0, (1.5, 2)), zorder=1)
    ax.text(1.62, area, f"lung-area share {area:.3f}\n(uniform attention)", va="center", ha="left", fontsize=8,
            color=muted)
    from src.components.model_builder import LAST_CONV_LAYER
    medians = {m: float(df[df.model == m]["tb_lung_energy"].median()) for m in CAM_MODELS}
    ax.set_xticks(list(range(len(CAM_MODELS))))
    ax.set_xticklabels([f"{NAMES[m]} ({LAST_CONV_LAYER[m]})\nall-image median {medians[m]:.3f}" for m in CAM_MODELS],
                       fontsize=9, color=ink)
    ax.set_xlim(-0.55, 2.35)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of Grad-CAM energy inside the true lung masks", fontsize=9, color=ink)
    ax.yaxis.grid(True, color=grid_col, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(grid_col)
    ax.tick_params(colors=muted, labelsize=8)
    ax.legend(loc="upper left", frameon=False, fontsize=9, title="Montgomery image (black bar = median)",
              title_fontsize=8)
    title = ("Montgomery (n = 138): models look outside the true lung fields" if CONFIG_TAG == "default"
             else f"Montgomery (n = 138): Grad-CAM energy inside the true lung fields ({CONFIG_TAG})")
    ax.set_title(title, fontsize=11, color=ink, loc="left")
    fig.tight_layout()
    out = os.path.join(GRADCAM_DIR, "h6_lung_energy.png")
    fig.savefig(out, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[h6plot] {out}")


CONFIG_TAG = "default"
CLINICAL_DIR = "artifacts/clinical"
INTERNAL_PRED_DIR = "artifacts/predictions"


def configure(config):
    """Model-dependent outputs go to the config's external_dir (default: artifacts/external, unchanged).
    Shared inputs (inventory, prepared images, overlap, data audit) always stay in artifacts/external."""
    global CONFIG_TAG, CLINICAL_DIR, EVAL_DIR, PRED_DIR, GRADCAM_DIR, PUBLIC_PRED_DIR, MODELS, CAM_MODELS
    global INTERNAL_PRED_DIR
    CONFIG_TAG = config.get("tag", "default")
    INTERNAL_PRED_DIR = config["artifacts"].get("predictions_dir", "artifacts/predictions")
    out = config["artifacts"].get("external_dir", OUT_DIR)
    CLINICAL_DIR = config["artifacts"].get("clinical_dir", "artifacts/clinical")
    EVAL_DIR, PRED_DIR = os.path.join(out, "evaluation"), os.path.join(out, "predictions")
    GRADCAM_DIR, PUBLIC_PRED_DIR = os.path.join(out, "gradcam"), os.path.join(out, "predictions_public")
    MODELS = [m for m in ("resnet", "vgg", "efficientnet") if m in config["models"]]  # primary first
    CAM_MODELS = ["resnet", "vgg"] if CONFIG_TAG == "default" else list(MODELS)


def main():
    configure(load_config())
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["sample", "audit", "prepare", "overlap", "readings", "cropsearch",
                                            "evaluate", "gradcam", "ceiling", "public", "h6plot"], required=True)
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()
    if args.stage == "sample":
        stage_sample(args.n)
    elif args.stage == "audit":
        stage_audit()
    elif args.stage == "prepare":
        stage_prepare()
    elif args.stage == "overlap":
        stage_overlap()
    elif args.stage == "readings":
        stage_readings()
    elif args.stage == "cropsearch":
        stage_cropsearch()
    elif args.stage == "evaluate":
        stage_evaluate()
    elif args.stage == "gradcam":
        stage_gradcam()
    elif args.stage == "ceiling":
        stage_ceiling()
    elif args.stage == "public":
        stage_public()
    elif args.stage == "h6plot":
        stage_h6plot()


if __name__ == "__main__":
    main()
