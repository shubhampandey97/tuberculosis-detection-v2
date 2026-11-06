# Code audit: tuberculosis-detection-v2

This audit was done by reading the code before any changes, plus read-only checks on the dataset. Line numbers refer to the code **as it was at the time of the audit** (commit `050effd`). See the **Status** section at the end for what has changed since.

> **Correction (goal 1):** The first version of this audit said that 124 TB images were 96×96. That came from misreading the JPEG DPI field in the `file` output. Pillow shows that those 124 TB images are **not square** (e.g. 256×221), while every Normal image is 256×256. Section §5.3 has been updated.

---

## §1 Project map

```
config.yaml ─► config_loader.py ─► train_pipeline.py ─► data_loader.py (load + /255 + augment)
                                         │               model_builder.py (frozen base + head)
                                         ▼
                                  models/tensorflow/*.h5  (gitignored)
                                         │
                ┌────────────────────────┼─────────────────────────┐
                ▼                        ▼                         ▼
           evaluate.py           gradcam_pipeline.py        app/streamlit_app.py
     (own loader, no /255)       (gradcam.py, VGG only)    (gradcam.py, VGG only)
                ▼
  artifacts/metrics/*.json, artifacts/plots/*/
```

| File | Role |
|---|---|
| [src/config/config.yaml](../src/config/config.yaml) | Data path, model name and save path (models are switched by commenting blocks in and out), epochs=10, batch_size=32, img_size, artifact paths |
| [src/utils/config_loader.py](../src/utils/config_loader.py) | Loads the YAML relative to `src/`. The only path in the project that doesn't depend on the current directory |
| [src/components/data_loader.py](../src/components/data_loader.py) | 80/20 split with `image_dataset_from_directory`, scaling ×1/255, augmentation (flip, rotate, zoom) |
| [src/components/model_builder.py](../src/components/model_builder.py) | Builds VGG16, ResNet50 or EfficientNetB0 with a frozen base plus a GAP → Dropout → sigmoid head |
| [src/components/gradcam.py](../src/components/gradcam.py) | Loads images, builds Grad-CAM heatmaps, shows the overlay with `plt.show()` |
| [src/pipeline/train_pipeline.py](../src/pipeline/train_pipeline.py) | Computes class weights, trains all three models in a loop, saves `.h5` files |
| [src/pipeline/evaluate.py](../src/pipeline/evaluate.py) | Reloads the validation split, predicts at threshold 0.5, writes JSON metrics and PNG plots |
| [src/pipeline/gradcam_pipeline.py](../src/pipeline/gradcam_pipeline.py) | Demo: Grad-CAM on `TB.1.jpg` with VGG |
| [src/pipeline/test_data_pipeline.py](../src/pipeline/test_data_pipeline.py) | Smoke script that prints one batch's shape. Not a pytest test |
| `torch_dataset.py`, `torch_model.py`, `torch_train.py` | A parallel PyTorch path that no other file uses (§5.7) |
| [app/streamlit_app.py](../app/streamlit_app.py) | Upload → VGG prediction → Grad-CAM overlay |
| [Dockerfile](../Dockerfile) | CUDA base image, TF 2.16.1. The container command runs training only |
| `artifacts/` | Metrics JSON and confusion-matrix and ROC plots for all three models. Tracked in git |
| `notebooks/`, `logs/` | **Empty** folders. The README advertises an EDA notebook that doesn't exist |
| `models/`, `data/` | Present on disk but gitignored |

---

## §2 Data flow

**Dataset:** `data/Dataset of Tuberculosis Chest X-rays Images/`, path set in [config.yaml:1](../src/config/config.yaml#L1).
- 3,008 three-channel JPEGs.
- `Normal Chest X-rays/`: 514 files named `others (N).jpg`.
- `TB Chest X-rays/`: 2,494 files named `TB.N.jpg`.
- TB is 83% of the images, so the majority class is the positive class.

**Labels:** Keras assigns labels in alphanumeric folder order, so **Normal = 0 and TB = 1**. Training ([data_loader.py:8-24](../src/components/data_loader.py#L8-L24)) and evaluation ([evaluate.py:24-31](../src/pipeline/evaluate.py#L24-L31)) use the same folder and the same function, so the labels match. The sigmoid output is P(TB). In the old metrics, class "0" has 87 images (14% of 601), which fits the Normal share.

**Split (old):**
- `validation_split=0.2, seed=42` gives 2,407 training and 601 validation images.
- The split is random, not stratified: Normal is 17.1% of the data but 14.5% of validation.
- Evaluation reuses validation as the test set (§5.3).

**Preprocessing (old):**

| Stage | What happens to pixels | Where |
|---|---|---|
| Training (all 3 models) | ×1/255 → [0,1], then augmentation | [data_loader.py:26-29](../src/components/data_loader.py#L26-L29), [:36-44](../src/components/data_loader.py#L36-L44) |
| Evaluation | **Raw 0–255**, no scaling | [evaluate.py:23-31](../src/pipeline/evaluate.py#L23-L31) |
| Grad-CAM script | **Raw 0–255** | [gradcam.py:8-12](../src/components/gradcam.py#L8-L12) |
| Streamlit app | **Raw 0–255** | [streamlit_app.py:57-61](../app/streamlit_app.py#L57-L61) |
| What each pretrained model expects | VGG16 and ResNet50: `preprocess_input` (caffe-style BGR, mean-subtracted, 0–255 scale). EfficientNetB0: raw 0–255 (rescales internally) | not applied anywhere |

---

## §3 Models and training

[model_builder.py:8-37](../src/components/model_builder.py#L8-L37) builds each model the same way:
- An ImageNet base with `include_top=False` at 224×224×3.
- **Every base layer frozen** (lines 22-23), with no fine-tuning stage.
- Head: GlobalAveragePooling2D → Dropout(0.5) → Dense(1, sigmoid).
- Adam with learning rate 1e-4, binary cross-entropy, accuracy metric.

**Hyperparameters:**
- 10 epochs ([config.yaml:16](../src/config/config.yaml#L16)).
- Batch size 16 in training ([data_loader.py:4](../src/components/data_loader.py#L4), hard-coded) but 32 in evaluation (the config value). The config `batch_size` is ignored during training.
- No callbacks: no early stopping, no checkpointing, no learning-rate schedule.

**Class weights** ([train_pipeline.py:24-32](../src/pipeline/train_pipeline.py#L24-L32)) use `compute_class_weight('balanced')` on the training labels. With 427 Normal and 1,980 TB this works out to about **{0: 2.82, 1: 0.61}**. These figures are calculated from the counts; training logs weren't saved.

---

## §4 Evaluation

[evaluate.py:34-95](../src/pipeline/evaluate.py#L34-L95):
1. Loads each `.h5` file and iterates over the 601-image validation split.
2. Uses `probs > 0.5` as the prediction.
3. Computes `classification_report` ("0"/"1" keys, no class names), `confusion_matrix` and `roc_auc_score` on P(TB).
4. Writes `artifacts/metrics/{model}.json`, a confusion-matrix PNG and a ROC PNG (no axis labels, no chance line).

Old results (**invalid**: they come from the broken pipeline and the leaky split). Positive class = TB:

| Model | ROC-AUC | Confusion matrix [[TN,FP],[FN,TP]] | What it's doing |
|---|---|---|---|
| VGG16 | 0.985 | [[87,0],[63,451]] | Sensitivity 0.877, specificity 1.00. Misses 63 of 514 TB cases |
| ResNet50 | 0.303 | [[0,87],[0,514]] | Predicts **TB for every image** |
| EfficientNetB0 | 0.148 | [[76,11],[510,4]] | Predicts **Normal for almost every image** |

---

## §5 Problems found

### §5.1 Why ResNet and EfficientNet score below chance
The labels are consistent (§2), so label order is not the cause. The cause is preprocessing:
- **Training and evaluation see different pixel scales.** Training uses [0,1] ([data_loader.py:26-29](../src/components/data_loader.py#L26-L29)); evaluation uses 0–255 ([evaluate.py:23-31](../src/pipeline/evaluate.py#L23-L31)).
- **No model-specific preprocessing.** [model_builder.py](../src/components/model_builder.py) never calls `preprocess_input`.
- **EfficientNetB0** rescales inputs internally, so it expects 0–255. During training it effectively saw near-black images (about 0–0.004), so its frozen features carried little signal. At evaluation, real 0–255 inputs pushed predictions to Normal and inverted the ranking (AUC 0.15).
- **ResNet50** expects BGR, mean-subtracted 0–255 input. It was trained on one wrong scale and evaluated on another, so its output saturated at TB.
- **Why VGG survived (hypothesis):** VGG has ReLU layers and no BatchNorm, so multiplying the input by 255 mostly scales the features rather than reordering them. Ranking survives (AUC 0.985), but the 0.5 threshold shifts, which fits the one-sided errors. VGG's numbers come from the same broken pipeline, so they need re-running too.

### §5.2 Other bugs
| Issue | Location |
|---|---|
| **The training loop saves every model to `vgg_model.h5`.** `MODEL_PATH` is read once at import (line 16); the loop changes `config` (lines 61-62) but never `MODEL_PATH`. The existing `.h5` files' sizes (59/95/17 MB) match each architecture, so they come from earlier runs where the config was edited by hand | [train_pipeline.py:16](../src/pipeline/train_pipeline.py#L16), [:46](../src/pipeline/train_pipeline.py#L46), [:61-62](../src/pipeline/train_pipeline.py#L61-L62) |
| Computing class weights runs a full pass through the *augmented* training set | [train_pipeline.py:24](../src/pipeline/train_pipeline.py#L24) |
| Grad-CAM divides by `reduce_max` and returns NaN when the heatmap is all ≤0 | [gradcam.py:34](../src/components/gradcam.py#L34) |
| Grad-CAM wraps `[model.inputs]` in an extra list, which Keras 3 may flag with a structure warning | [gradcam.py:18](../src/components/gradcam.py#L18) |
| The two Grad-CAM entry points use different VGG layers (`block5_conv2` vs `block5_conv3`) | [gradcam_pipeline.py:10-11](../src/pipeline/gradcam_pipeline.py#L10-L11) vs [streamlit_app.py:19](../app/streamlit_app.py#L19) |
| The app has no `.convert("RGB")`, so a grayscale or RGBA upload gives the wrong input shape and crashes | [streamlit_app.py:51-59](../app/streamlit_app.py#L51-L59) |
| The app shows the uncalibrated sigmoid as "Confidence" and has no "not for clinical use" disclaimer | [streamlit_app.py:63-75](../app/streamlit_app.py#L63-L75) |
| `display_gradcam` calls `plt.show()`, so nothing appears in headless runs. The app duplicates this overlay code | [gradcam.py:52-54](../src/components/gradcam.py#L52-L54), [streamlit_app.py:83-89](../app/streamlit_app.py#L83-L89) |
| `torch_train` saves to `models/torch/`, which is never created, so it crashes after training | [torch_train.py:44](../src/pipeline/torch_train.py#L44) |

### §5.3 Data leakage and validity risks
1. **No independent test set.** Evaluation rebuilds the exact validation split used in `model.fit`. The reported metrics and the choice of VGG are both based on that one split. See [evaluate.py:24-31](../src/pipeline/evaluate.py#L24-L31) and [train_pipeline.py:39](../src/pipeline/train_pipeline.py#L39).
2. **26 pairs of byte-identical images (52 files), all within the same class** (5 Normal, 21 TB). A random split can put one copy in training and the other in test. Examples: Normal `others (76–80).jpg` = `others (509–513).jpg`; TB `TB.703` = `TB.745`; `TB.1000` = `TB.2040`. Only exact copies were checked, not near-duplicates.
3. **No patient IDs**, so a patient-level split isn't possible. This is a stated limitation.
4. **Possible shortcut from image source** *(corrected)*. All 514 Normal images are 256×256. Of the TB images (after removing duplicates), 2,349 are 256×256 and **124 are non-square** (widths 234–256, heights 185–240; e.g. `TB.100.jpg` is 256×221). These 124 also carry 96-dpi JPEG metadata, while the others report density 1×1. Resizing to 224×224 stretches them vertically, and a model could learn "stretched means TB". The different filename schemes also suggest the two classes come from different sources.
5. The old split is not stratified (14.5% Normal in validation vs 17.1% overall).
6. The PyTorch path validates on the **full training set** ([torch_dataset.py:22-30](../src/components/torch_dataset.py#L22-L30)).
7. **Strong source confounding (found by the data audit, `notebooks/01_data_audit.ipynb`).** Simple models that never see the anatomy can predict the label from metadata alone (5-fold CV ROC-AUC):

   | Metadata only | Logistic regression | Decision tree (depth 3) |
   |---|---|---|
   | Width / height / aspect | 0.525 | 0.525 |
   | File size (KB) | 0.784 | **0.861** |
   | Brightness / contrast | 0.744 | 0.803 |
   | All metadata | 0.911 | **0.918** |

   Visual inspection agrees. All Normal images share one look: the same small "L" marker in the top-right corner, similar exposure and vertical streak artefacts. TB images vary: "R" markers in different positions, handwritten text and dates at the bottom, and varied exposure and noise. The two classes most likely come from different sources, so a CNN could learn markers, annotations or compression instead of TB. Some of the brightness/contrast difference may be real pathology, so the effects can't be fully separated. **Consequence:** internal test scores are an upper bound and may partly measure the source, not the disease. Goal 4 tests this with border masking, subgroup analysis and external validation; goal 5 checks Grad-CAM for attention on markers.

### §5.4 Hard-coded values and paths (all relative to the current directory)
- `IMG_SIZE` and `BATCH_SIZE` are hard-coded in [data_loader.py:3-4](../src/components/data_loader.py#L3-L4), [model_builder.py:6](../src/components/model_builder.py#L6) and [gradcam.py:6](../src/components/gradcam.py#L6).
- Save paths: [train_pipeline.py:56,62](../src/pipeline/train_pipeline.py#L56-L62); [evaluate.py:64-65, 102-106](../src/pipeline/evaluate.py#L64-L65).
- Demo paths and layer: [gradcam_pipeline.py:4-5,11](../src/pipeline/gradcam_pipeline.py#L4-L11); [test_data_pipeline.py:3](../src/pipeline/test_data_pipeline.py#L3).
- PyTorch: [torch_train.py:9, 24, 44](../src/pipeline/torch_train.py#L9) (path, epochs, save directory).
- App: [streamlit_app.py:18-20](../app/streamlit_app.py#L18-L20).
- Config keys that nothing reads: `metrics_path` and `plots_dir` ([config.yaml:21-22](../src/config/config.yaml#L21-L22); loaded in evaluate.py:16-17 but never used), `model.name` (ignored by the main block), `training.batch_size` (ignored in training), and `IMG_SIZE` in [train_pipeline.py:14](../src/pipeline/train_pipeline.py#L14).

### §5.5 Missing seeds
Only the split seed is set ([data_loader.py:12,21](../src/components/data_loader.py#L12); [evaluate.py:28](../src/pipeline/evaluate.py#L28)). Nothing sets `tf.random`, NumPy, Python `random`, `PYTHONHASHSEED` or op determinism. The augmentation layers ([data_loader.py:37-39](../src/components/data_loader.py#L37-L39)) and Dropout have no seed. The PyTorch path has no `torch.manual_seed`.

### §5.6 What breaks on a fresh machine
- **`requirements.txt` doesn't include TensorFlow** (or torch), and only `numpy<2` is pinned.
- `data/` and `models/` are gitignored, there are no dataset download steps and there is no DVC. A fresh clone can't train, evaluate or run the app.
- The [.gitignore:19](../.gitignore#L19) re-include rule doesn't work: inline `#` comments aren't allowed, and a folder inside an ignored directory can't be re-included.
- README:
  - The code block opened at [README.md:82](../README.md#L82) is never closed.
  - The clone URL is a placeholder (`your-username`).
  - It tells users to install torch with cu121 (line 95).
  - The architecture section lists notebooks and logs that are empty.
- Dockerfile:
  - **`libgl1` is missing**, so `import cv2` (opencv-python) fails.
  - The `-base` CUDA image has no cuDNN, and plain `tensorflow==2.16.1` doesn't bundle CUDA, so it runs on CPU despite the "GPU" comment.
  - The container command runs the buggy training loop ([Dockerfile:35](../Dockerfile#L35)).
  - No Streamlit entry point or `EXPOSE`.
  - No `.dockerignore`, so `COPY . .` copies `.git`, the models and the dataset.
- No `__init__.py` files: imports only work with `python -m` from the repo root.
- `st.image(width="stretch")` needs a recent Streamlit, which isn't pinned.

### §5.7 Unused or dead files
- **PyTorch path:** [torch_dataset.py](../src/components/torch_dataset.py), [torch_model.py](../src/components/torch_model.py), [torch_train.py](../src/pipeline/torch_train.py).
  - No other file references them, and they have no `__pycache__` entries, so they were most likely never run.
  - They're also broken: no split, no ImageNet normalization, no class weights, a missing save directory, and torch isn't in the requirements.
- [test_data_pipeline.py](../src/pipeline/test_data_pipeline.py) is a print-only smoke script.
- `gradcam.display_gradcam` is only used by the demo script.
- Commented-out code blocks in [train_pipeline.py:51-52](../src/pipeline/train_pipeline.py#L51-L52), [evaluate.py:98-99](../src/pipeline/evaluate.py#L98-L99) and [config.yaml:7-13](../src/config/config.yaml#L7-L13).

### §5.8 Honesty issues in the README
- [README.md:20-26](../README.md#L20-L26) reports only VGG. It doesn't say that ResNet and EfficientNet failed, or that the "test" set was the validation set.
- [README.md:29-30](../README.md#L29-L30) says class weights "significantly improved recall and ROC-AUC", but nothing in `artifacts/` supports this.
- [README.md:14](../README.md#L14) says "Fully containerized", but the container can't run the app.

---

## §6 Gaps against the CLAUDE.md goals (at audit time)

| Area | Exists | Missing |
|---|---|---|
| Preprocessing fix and re-run | Root cause found (§5.1) | Per-model preprocessing used the same way everywhere, save-path fix, fresh results |
| Split and duplicates | Seeded 80/20 random split | Stratified train/val/test split, global seeds, de-duplication, shortcut check |
| Screening evaluation | ROC-AUC, confusion matrix, per-class precision/recall | Clearly named sensitivity/specificity/PPV/NPV, bootstrap CIs, calibration, threshold tuned on val |
| Grad-CAM | Single-image heatmap, app overlay | Gallery of false negatives, one layer per model, NaN guard |
| Engineering | — | pytest tests, CI, MLflow, DVC |
| Serving and docs | Dockerfile (training only, broken) | FastAPI endpoint, working image, MODEL_CARD.md, honest README table |

---

## Status

### Goal 1 — DATA: done (2026-11-06)
- **New:** [src/pipeline/make_splits.py](../src/pipeline/make_splits.py). Run with `python -m src.pipeline.make_splits`. It's deterministic (seed 42) and reproduces a byte-identical CSV on every run.
- **`.gitignore` (fixes §5.6):** `data/` is now `data/*` with `!data/splits.csv` and `!data/duplicates.csv`, so the split files are tracked but the images aren't. The `!data/sample/` re-include now works; its inline comment was moved to its own line.
- **Outputs (under `data/`, tracked in git):**
  - `data/splits.csv` with columns `path, label, split, width, height`.
  - `data/duplicates.csv` listing each dropped file and the copy that was kept.
  - No image files were deleted or moved.
- **Duplicates:** 26 byte-identical copies left out (5 Normal, 21 TB). All pairs are within one class. The lower-numbered file in each pair is kept. 3,008 files → **2,982 images**.
- **Split:** 70/15/15, stratified by label × "not 256×256" flag, so the non-square TB images are spread evenly for the subgroup analysis in goal 4.

| split | Normal | TB | total | TB % | TB not 256×256 |
|---|---|---|---|---|---|
| train | 356 | 1,731 | 2,087 | 82.9 | 87 |
| val | 76 | 371 | 447 | 83.0 | 19 |
| test | 77 | 371 | 448 | 82.8 | 18 |
| total | 509 | 2,473 | 2,982 | 82.9 | 124 |

- **Built-in checks** (the script stops with an error if any fail):
  - All 3,008 files are accounted for.
  - Paths and file hashes are unique.
  - No hash appears in two splits.
  - Split fractions and the TB share are within 1 percentage point of target.
  - A separate re-hash of `splits.csv` gave 2,982 unique hashes.
- **Not done yet:** near-duplicate detection (perceptual hashing). Exact copies only.
- **Near-duplicates:** checked with thumbnail correlation ≥ 0.99 in the data audit: N candidate pairs, M confirmed by visual check (removed / kept because …).
- **Limitation:** no patient IDs, so images from the same patient could still land in different splits.

### Data audit: done (2026-11-07)
- **New:** [src/pipeline/data_audit.py](../src/pipeline/data_audit.py) (`python -m src.pipeline.data_audit`) and [notebooks/01_data_audit.ipynb](../notebooks/01_data_audit.ipynb) (same checks, with plots). Outputs go to `reports/data_audit/`.
- **Confirmed:** 3,008 images, 0 corrupt, all RGB; 26 exact-duplicate groups (52 files), none cross-class; 124 non-square images, all TB, all with 96-dpi metadata.
- **New finding:** metadata-only ROC-AUC 0.918 (file size alone 0.861) → strong source confounding. See §5.3 item 7.
- **Method note:** a 64-bit perceptual hash (dHash) was tried for near-duplicates and rejected. It flagged 22,138 pairs, mostly different patients, because chest X-rays share the same layout. It was replaced with correlation of 32×32 thumbnails (≥ 0.99).
- **Goal 4 additions:** (a) metrics on test images with borders masked, (b) the non-square TB subgroup vs the rest, (c) external validation on a dataset where both classes come from the same hospital (e.g. Shenzhen or Montgomery).

### Goal 2 — PIPELINE FIX: done, smoke-tested (2026-11-07). Full training not run yet (goal 3).
- **Data** ([data_loader.py](../src/components/data_loader.py)):
  - A tf.data pipeline from `data/splits.csv` (fixes §5.3 item 1: there is now a held-out test split).
  - Images are decoded as RGB, resized to 224×224, and kept in **raw 0–255**. The `/255` step is removed (§5.1).
  - Class weights are computed from the training rows of `splits.csv`: {0: 2.931, 1: 0.603}.
- **Decision — resize:** a plain 224×224 resize is kept on purpose. Non-square TB images are stretched, and goal 4 measures whether that acts as a shortcut.
- **Augmentation** (training only, seeded): `RandomRotation(0.028)` (about ±10°) and `RandomZoom(0.1)`. `RandomFlip("horizontal")` is removed, because a flipped chest X-ray puts the heart on the wrong side. The old rotation of 0.1 meant ±36°.
- **Models** ([model_builder.py](../src/components/model_builder.py)):
  - Preprocessing is now inside each model, so a saved model takes raw 0–255 RGB everywhere (§5.1).
  - VGG16 and ResNet50 use a registered `CaffePreprocess` layer, identical to Keras `preprocess_input` (max difference 0.0). EfficientNetB0 has no extra layer.
  - The base is built with `input_tensor=`, so the graph stays flat. `LAST_CONV_LAYER` records the Grad-CAM layer for each model (goal 5).
  - The base is frozen; only the head is trainable (513, 2,049 and 1,281 parameters).
  - Metrics: accuracy and AUC.
- **Decision — format:** models are saved as `models/tensorflow/{name}_model.keras` (native Keras 3). A `Lambda(preprocess_input)` layer would not reload safely. The old `.h5` files are untouched; the app and the Grad-CAM script still use the old VGG `.h5` until goals 5 and 6.
- **Training** ([train_pipeline.py](../src/pipeline/train_pipeline.py)):
  - One save path per model (fixes §5.2).
  - Epoch cap 20. `EarlyStopping(val_auc, patience 4, restore_best_weights)` and `ModelCheckpoint(best val_auc)`.
  - History is saved to `artifacts/history/{name}.json`.
  - Batch size, image size, epochs, learning rate, patience and augmentation settings are read from [config.yaml](../src/config/config.yaml) (fixes §5.4 for training).
- **Seeds** ([seed.py](../src/utils/seed.py)): `PYTHONHASHSEED`, `keras.utils.set_random_seed(42)` and `enable_op_determinism()` (fixes §5.5). **Determinism kept:** in the smoke test it cost 0.93–1.08× the run time without it (limit was 2×), and the results were identical both ways.
- **Evaluation** ([evaluate.py](../src/pipeline/evaluate.py)):
  - Metrics and plots on the **test split only**, at threshold 0.5.
  - Per-image predictions for val and test go to `artifacts/predictions/{model}_{split}.csv` (`path, label, prob, width, height`).
  - The best model is chosen by **val AUC only** (`artifacts/metrics/summary.json`); test results are reported for all three.
- **Legacy:** old metrics and plots moved to [artifacts/legacy/](../artifacts/legacy/README.md), marked invalid.
- **Smoke test** ([smoke_test.py](../src/pipeline/smoke_test.py)): 1 epoch on 200 training and 100 val images (stratified), writing only to `models/smoke/`. **Passed for all 3 models:**

| model | preprocess vs Keras | predictions (32 test images) | reload max diff | 1-epoch fit | s/image (inference) |
|---|---|---|---|---|---|
| VGG16 | 0.0 | 0.002–0.882 | 0.0 | 42.3 s | 0.133 |
| ResNet50 | 0.0 | 0.285–0.916 | 0.0 | 28.8 s | 0.061 |
| EfficientNetB0 | n/a | 0.453–0.714 | 0.0 | 20.8 s | 0.033 |

  The smoke val AUCs (0.46–0.63) come from 1 epoch on 200 images. They only show the pipeline runs; they are **not** results. `evaluate.py` was also dry-run on the smoke models, with outputs written to a temporary folder.
- **Expected full training time (CPU):** about 5.6, 2.6 and 1.4 min per epoch (VGG, ResNet, EfficientNet). That's about 1.2–2.5 h in total if early stopping ends at 8–15 epochs, and about 3.2 h at the 20-epoch cap. Evaluation adds a few minutes.

### Goals 3–8: not started
