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

### Goal 2 — PIPELINE FIX: done (2026-11-07). Smoke-tested here; full training in goal 3.
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

### Goal 3 — RETRAIN: done (commit `5cd84e3`)
- All 3 models were retrained with the goal 2 pipeline (`python -m src.pipeline.train_pipeline`, log in `logs/train_full.log`, UTF-16) and evaluated with `python -m src.pipeline.evaluate`. Models: `models/tensorflow/{vgg,resnet,efficientnet}_model.keras`. Per-image predictions: `artifacts/predictions/{model}_{val,test}.csv`.
- **Epochs:** none of the models stopped early. All three ran the full 20-epoch cap, and the saved weights are the best val-AUC epoch:

| model | epochs run | best epoch (saved) | best val AUC | training time |
|---|---|---|---|---|
| VGG16 | 20 | 20 | 0.9977 | 6,238.7 s (≈104 min) |
| ResNet50 | 20 | 19 | 0.9989 | 2,776.9 s (≈46 min) |
| EfficientNetB0 | 20 | 19 | 0.9981 | 1,434.1 s (≈24 min) |

  Total ≈ 2.9 h (estimate was ≈ 3.2 h at the cap). VGG's val AUC was still rising at epoch 20 (0.99739 → 0.99775), so it may be slightly under-trained. ResNet and EfficientNet had flattened out.
- **Test results** (448 images: 77 Normal, 371 TB; threshold 0.5; positive class = TB). Sensitivity and specificity are worked out from the confusion matrices, truncated (not rounded) to 3 decimals:

| model | test ROC-AUC | confusion matrix [[TN,FP],[FN,TP]] | sensitivity | specificity |
|---|---|---|---|---|
| VGG16 | 0.9917 | [[76,1],[41,330]] | 330/371 = 0.889 | 76/77 = 0.987 |
| ResNet50 | 0.9983 | [[76,1],[17,354]] | 354/371 = 0.954 | 76/77 = 0.987 |
| EfficientNetB0 | 0.9964 | [[76,1],[39,332]] | 332/371 = 0.894 | 76/77 = 0.987 |

- **Model selection:** ResNet50 is best by **val** AUC (0.9989 vs 0.9981 EfficientNet, 0.9978 VGG; `artifacts/metrics/summary.json`). Test results were not used to choose.
- **Before/after:** the legacy AUCs (VGG 0.985, ResNet 0.303, EfficientNet 0.148; `artifacts/legacy/`) came from the broken preprocessing and a leaky, different split. They show that the §5.1 bug is fixed; they are **not** a fair performance comparison.
- **Caveats:**
  - At 0.5, all models miss many TB cases (17–41 of 371), below the ≥ 95% sensitivity screening target for VGG and EfficientNet. Goal 4 tunes the threshold on val.
  - Specificity rests on only 77 Normal test images.
  - Because of the source confounding (§5.3 item 7), these internal scores are an **upper bound**.

### Goal 4 — EVALUATION: done (2026-11-07)

**Primary model = ResNet50, chosen on val AUC (0.9989). Headline metrics = TEST at the val-tuned threshold.** Results at 0.5 and for the other two models are shown for comparison only. Test results were never used to choose a model or a threshold.

- **Script:** [src/pipeline/evaluate_clinical.py](../src/pipeline/evaluate_clinical.py) (`python -m src.pipeline.evaluate_clinical`; `--skip-masking` runs the statistics only, in about 17 s). It reads `artifacts/predictions/` (no retraining) and writes everything to `artifacts/clinical/`:
  - `thresholds.json`
  - `metrics_test.json` and `metrics_test.csv`
  - `calibration.png` and `calibration_bins.csv`
  - `subgroups.csv`
  - `shared_errors.csv` and `shared_errors.json`
  - `masking.json`, `masking.csv`, `masked_examples.png` and `predictions_masked/`
  - `summary.md` (all tables)
  - Run log: `logs/evaluate_clinical.log`
- **Method:**
  - Positive class = TB; an image is called TB when P(TB) ≥ threshold. Test has 448 images (77 Normal, 371 TB).
  - 95% CIs: stratified bootstrap (resampled within each class), 2,000 resamples, seed 42, percentile method. All models share the same resamples.
  - All values are **truncated, not rounded,** to 3 decimals.
  - **Undefined PPV/NPV resamples: 0** for every model and threshold. No resample lacked predicted positives or negatives.

**Thresholds (chosen on VAL):** highest threshold giving val sensitivity ≥ 0.95, i.e. at least 353 of 371 TB caught.

| model | tuned threshold | val sensitivity | val specificity |
|---|---|---|---|
| VGG16 | 0.2357 | 0.951 (353/371) | 1.000 (76/76) |
| **ResNet50** | 0.6142 | 0.951 (353/371) | 1.000 (76/76) |
| EfficientNetB0 | 0.3257 | 0.951 (353/371) | 1.000 (76/76) |

**Headline — TEST at the val-tuned threshold** (95% bootstrap CI):

| model | sensitivity | specificity | PPV | NPV | F1 | [TN,FP / FN,TP] |
|---|---|---|---|---|---|---|
| **ResNet50** | **0.935 (0.908–0.959)** | **1.000 (1.000–1.000)\*** | 1.000 (1.000–1.000)\* | 0.762 (0.693–0.836) | 0.966 (0.951–0.979) | [77,0 / 24,347] |
| VGG16 | 0.929 (0.902–0.954) | 0.987 (0.961–1.000) | 0.997 (0.991–1.000) | 0.745 (0.675–0.819) | 0.962 (0.947–0.976) | [76,1 / 26,345] |
| EfficientNetB0 | 0.946 (0.921–0.967) | 0.987 (0.961–1.000) | 0.997 (0.991–1.000) | 0.791 (0.723–0.863) | 0.970 (0.957–0.982) | [76,1 / 20,351] |

\* With 0 errors the bootstrap can't vary, so its interval collapses to 1.000–1.000. Exact Clopper–Pearson 95% lower bounds: specificity 77/77 → **0.953**; PPV 347/347 → **0.989**.

**For comparison — TEST at threshold 0.5:**

| model | sensitivity | specificity | PPV | NPV | F1 | [TN,FP / FN,TP] |
|---|---|---|---|---|---|---|
| ResNet50 | 0.954 (0.932–0.975) | 0.987 (0.961–1.000) | 0.997 (0.991–1.000) | 0.817 (0.749–0.888) | 0.975 (0.963–0.986) | [76,1 / 17,354] |
| VGG16 | 0.889 (0.857–0.919) | 0.987 (0.961–1.000) | 0.996 (0.990–1.000) | 0.649 (0.589–0.719) | 0.940 (0.921–0.957) | [76,1 / 41,330] |
| EfficientNetB0 | 0.894 (0.862–0.924) | 0.987 (0.961–1.000) | 0.996 (0.990–1.000) | 0.660 (0.596–0.733) | 0.943 (0.924–0.960) | [76,1 / 39,332] |

**Threshold-free (TEST):**

| model | ROC-AUC | Brier score |
|---|---|---|
| ResNet50 | 0.998 (0.996–0.999) | 0.026 (0.018–0.035) |
| VGG16 | 0.991 (0.982–0.998) | 0.072 (0.054–0.091) |
| EfficientNetB0 | 0.996 (0.991–0.999) | 0.071 (0.058–0.086) |

**Findings:**
1. **The ≥ 95% sensitivity target was met on val but not on test.** Test sensitivity at the tuned threshold is 0.929–0.946 for all three models; the CIs include 0.95 for ResNet and EfficientNet. The rule chooses the *highest* threshold that just reaches 0.95 on val (18 misses allowed), so it has no safety margin. For ResNet the tuned threshold (0.614) is even **above 0.5**: on test it trades sensitivity 0.954 → 0.935 for specificity 0.987 → 1.000. Switching to 0.5 because it looks better on test would use test data to tune, so the result stands as reported. **Open decision (before goal 6):** use a margin chosen on val, e.g. target 0.97 on val, or the threshold whose val sensitivity lower CI bound is ≥ 0.95.
2. **PPV and NPV reflect this dataset's 83% TB prevalence,** not a screening population. With realistic TB prevalence (well under 1–5%), PPV would be far lower and NPV higher. PPV/NPV here must not be read as clinical values. Specificity rests on 77 Normal images.
3. **Calibration** ([calibration.png](../artifacts/clinical/calibration.png)):
   - All three models **under-predict P(TB)** in the middle range: the reliability curves lie above the diagonal. For example, every VGG image with P(TB) in 0.1–0.9 was actually TB.
   - VGG's lowest bin (P < 0.1) holds 91 images with mean P(TB) 0.018, but 16.4% of them are TB: confidently wrong misses.
   - ResNet is the best calibrated (Brier 0.026).
   - A likely cause is the class weights (Normal weighted ≈ 2.9×), which push probabilities down. This is a hypothesis, not tested. The probabilities shouldn't be shown as "confidence" without recalibration (goal 6).
4. **Non-square subgroup** (test TB images only; 18 non-square vs 353 at 256×256):

| model | subgroup | n | mean P(TB) | sensitivity @0.5 | sensitivity @tuned |
|---|---|---|---|---|---|
| ResNet50 | non-square | 18 | 0.942 | 1.000 (exact 95% lower bound 0.814), 0 missed | 1.000, 0 missed |
| ResNet50 | 256×256 | 353 | 0.916 | 0.951 (0.929–0.974), 17 missed | 0.932 (0.903–0.957), 24 missed |
| VGG16 | non-square | 18 | 0.876 | 0.888 (0.722–1.000), 2 missed | 0.888 (0.722–1.000), 2 missed |
| VGG16 | 256×256 | 353 | 0.860 | 0.889 (0.855–0.920), 39 missed | 0.932 (0.903–0.957), 24 missed |
| EfficientNetB0 | non-square | 18 | 0.861 | 0.888 (0.722–1.000), 2 missed | 0.944 (0.833–1.000), 1 missed |
| EfficientNetB0 | 256×256 | 353 | 0.800 | 0.895 (0.861–0.926), 37 missed | 0.946 (0.920–0.968), 19 missed |

   Mean P(TB) is slightly higher for non-square images in all three models (+0.016 to +0.061). That's consistent with a small stretch effect, but with n = 18 the CIs overlap completely. **No evidence that the stretch drives detection, and not enough data to rule it out.** The image-size-only baseline was already measured in the data audit: width/height/aspect AUC 0.525 (§5.3 item 7).
5. **Shared errors** (`shared_errors.csv` lists every image any model got wrong, with each model's P(TB)):

| error @ threshold | VGG16 | ResNet50 | EfficientNetB0 | any model | all 3 | exactly 2 | only 1 |
|---|---|---|---|---|---|---|---|
| false positive @0.5 | 1 | 1 | 1 | 1 | 1 | 0 | 0 |
| false negative @0.5 | 41 | 17 | 39 | 62 | 7 | 21 | 34 |
| false positive @tuned | 1 | 0 | 1 | 1 | 0 | 1 | 0 |
| false negative @tuned | 26 | 24 | 20 | 43 | 6 | 15 | 22 |

   - The only false positive is the same Normal image for every model that makes one: `others (23).jpg`.
   - TB missed by all 3 at the tuned threshold: `TB.1141`, `TB.1892`, `TB.2145`, `TB.2324`, `TB.321`, `TB.441`. At 0.5 the same six plus `TB.244`.
   - None of the shared misses is a non-square image.
   - Most misses are model-specific (22 of 43 by only one model at tuned). These seven images are first in line for the goal 5 Grad-CAM gallery.
6. **Masking test** (runs the models; 5 min 54 s on CPU):
   - Variants: border 12% (27 px, 170×170 centre visible), border 20% (45 px, 134×134 visible), and a centre-mask control (central 170×170 hidden, 27-px border visible). The fill is the image's own mean grey, after the 224×224 resize. Examples: [masked_examples.png](../artifacts/clinical/masked_examples.png).
   - Thresholds are the unmasked val-tuned ones.
   - Sanity check: re-predicting unmasked images matched the saved predictions to ≤ 3e-8.

| model | input | ROC-AUC (95% CI) | sens @0.5 | spec @0.5 | sens @tuned | spec @tuned | mean P(TB) Normal / TB |
|---|---|---|---|---|---|---|---|
| ResNet50 | unmasked | 0.998 (0.996–0.999) | 0.954 | 0.987 | 0.935 | 1.000 | 0.051 / 0.917 |
| ResNet50 | border 12% | 0.970 (0.951–0.986) | 1.000 | 0.025 | 1.000 | 0.103 | 0.811 / 0.975 |
| ResNet50 | border 20% | 0.919 (0.887–0.948) | 1.000 | 0.000 | 1.000 | 0.000 | 0.908 / 0.967 |
| ResNet50 | **centre hidden** | **0.971 (0.955–0.984)** | 0.997 | 0.025 | 0.997 | 0.077 | 0.802 / 0.978 |
| VGG16 | unmasked | 0.991 (0.982–0.998) | 0.889 | 0.987 | 0.929 | 0.987 | 0.023 / 0.861 |
| VGG16 | border 12% | 0.939 (0.907–0.965) | 0.997 | 0.103 | 1.000 | 0.000 | 0.756 / 0.975 |
| VGG16 | border 20% | 0.799 (0.745–0.851) | 1.000 | 0.000 | 1.000 | 0.000 | 0.949 / 0.986 |
| VGG16 | **centre hidden** | **0.974 (0.959–0.987)** | 1.000 | 0.012 | 1.000 | 0.000 | 0.868 / 0.995 |
| EfficientNetB0 | unmasked | 0.996 (0.991–0.999) | 0.894 | 0.987 | 0.946 | 0.987 | 0.067 / 0.803 |
| EfficientNetB0 | border 12% | 0.981 (0.968–0.992) | 0.978 | 0.818 | 0.997 | 0.480 | 0.346 / 0.888 |
| EfficientNetB0 | border 20% | 0.969 (0.953–0.982) | 0.991 | 0.350 | 1.000 | 0.038 | 0.558 / 0.890 |
| EfficientNetB0 | **centre hidden** | **0.912 (0.879–0.944)** | 0.911 | 0.688 | 0.991 | 0.285 | 0.430 / 0.734 |

   **How to read this:**
   - **With the central 170×170 hidden, AUC stays at 0.912–0.974.** The models can still rank TB above Normal from what lies *outside* the lung field: corner markers ("L" on Normal, "R" on TB), annotations, collimation edges, the overall look of the image. **High AUC with the lungs hidden means the models rely on non-lung cues.**
   - It is not explained by the fill colour: mean brightness alone gives a test AUC of only 0.576.
   - This is direct evidence that the source confounding in §5.3 item 7 is learned by the models.
   - **Caveat:** the 27-px border still shows lung tops, outer lung edges and the diaphragm, so "lungs hidden" is approximate.
   - **Border masking:** AUC falls as more border is hidden.
     - VGG16 falls the most (0.991 → 0.939 → 0.799), so it leans most on the periphery.
     - ResNet50: 0.998 → 0.970 → 0.919.
     - EfficientNetB0 is the most robust to border masking (0.996 → 0.981 → 0.969) and the weakest with the centre hidden (0.912), so it relies relatively more on central content.
   - **Threshold-based numbers under masking are not meaningful.** Masking shifts every probability towards TB (mean P(TB) for Normal goes from 0.02–0.07 to 0.35–0.95), so specificity collapses at fixed thresholds. AUC is the measure to read here.
   - Part of each drop may be distribution shift: the models never saw grey borders in training. So the border drops are an upper bound on reliance on the border. The centre-hidden result doesn't have that problem in the same direction: shift should *hurt* AUC, yet it stays high.
   - **Conclusion:** the internal test scores (AUC 0.991–0.998) **overstate** how well the models detect TB from lung findings. External validation on a single-source dataset (Shenzhen/Montgomery) is needed before any claim of clinical relevance.

### Goal 5 — GRAD-CAM: done (2026-11-07). Full write-up: [failure_analysis.md](failure_analysis.md)
- **Fixed in [gradcam.py](../src/components/gradcam.py)** (§5.2 rows):
  - NaN guard: maps with max ≤ 1e-8 are returned as zeros and flagged empty.
  - Keras 3 gradient model built without the extra list around `model.inputs`.
  - No `plt.show()` and no cv2: `overlay()` returns an RGB array.
  - One documented layer per model (`model_builder.LAST_CONV_LAYER`): VGG16 `block5_conv3`, ResNet50 `conv5_block3_out`, EfficientNetB0 `top_conv`. The `block5_conv2`/`block5_conv3` mismatch is gone.
  - Grad-CAM now targets the pre-sigmoid logit and is batched.
  - `make_gradcam_heatmap()` keeps its old signature; checked to still work with the app's old VGG `.h5`.
  - [gradcam_pipeline.py](../src/pipeline/gradcam_pipeline.py) is now a command-line tool (`--model --image --out`) with no hard-coded paths.
- **Analysis:** [gradcam_analysis.py](../src/pipeline/gradcam_analysis.py). The full run takes 9 min 17 s on CPU; `--sanity-only` and `--galleries-only` are available. Outputs go to `artifacts/gradcam/`.
- **Sanity check (32 images):**
  - Probabilities match the saved ones (≤ 3.2e-7).
  - No NaN; 0 empty TB maps.
  - Randomised head: median r = −0.10 for VGG (passes) and **0.49 for ResNet (only partly passes**: ResNet maps partly reflect where features are active, so they are weaker evidence).
- **Attention: share of TB-evidence energy in the outer 12% border** (median; uniform reference 0.424):
  - **ResNet50, correct TB: 0.331** (mostly central, but the bottom strip gets 1.4× its area share).
  - **VGG16: 0.463 overall**, with the bottom strip at 2.1–2.9× its area share.
  - **Missed TB is more peripheral than correct TB**, robust across layers: ResNet conv5 0.388 vs 0.331 (p = 0.047); ResNet conv4 0.526 vs 0.308 (p = 0.010); VGG 0.624 vs 0.429 (p = 8.7e-9).
  - The ResNet Normal vs TB difference flips between conv5 and conv4, so it isn't robust.
  - Empty maps:
    - TB-evidence: ResNet conv5 1, conv4 13, VGG 0.
    - Predicted-class: 0 for all.
- **Galleries** (qualitative): the 7 shared misses + `others (23)`, plus 10 random correct Normal and 10 random correct TB.
  - Hot spots on **handwritten text and labels** in several correct TB images.
  - No consistent focus on the Normal "L" marker.
  - The Normal evidence for the misses lies mostly outside the lung fields.
- **The 7 shared misses:**
  - All 256×256.
  - 6 of 7 have Normal-like **large file sizes** (79th–94th percentile of TB) and lower contrast.
  - None has bottom handwriting.
  - TB.1892 shows visible opacities but is still missed.
  - The false positive `others (23)` is a Normal image with a TB-like small file.
  - **Hypothesis:** the models partly learned the image source (consistent with §5.3 item 7 and the goal 4 masking result).
- **Proposed next step (not done):** retrain on border-masked or lung-cropped images, then repeat the masking test; external validation; radiologist review of the misses.

### Goal 6 — APP + REQUIREMENTS + README: done (2026-11-07)
- **App** ([app/streamlit_app.py](../app/streamlit_app.py)). Fixes the §5.2 app rows and the §5.4 app paths.
  - Loads the primary **ResNet50 `.keras`** model and the **val-tuned threshold 0.6142** from `artifacts/clinical/thresholds.json`, so the threshold can't drift from the evaluated value.
  - Shows a **"Research demo, not for clinical use"** banner that mentions the source bias and links to the failure analysis.
  - Shows a "TB suspected" or "No TB flag" result and an **"Uncalibrated model score"**. The "Confidence" metric and progress bar are gone.
  - Grad-CAM is the predicted-class map from the new `gradcam.py`. No cv2.
  - Upload handling: `.convert("RGB")` (grayscale, RGBA and palette PNGs work); unreadable files show an error instead of crashing.
- **Shared inference** ([src/components/inference.py](../src/components/inference.py)): paths resolve from the repo root, and it's reusable for the FastAPI endpoint in goal 8.
- **Bug found and fixed: preprocessing parity.**
  - Decoding JPEGs with PIL moved test scores by up to **0.024** (mean 0.0019) compared with training, which decodes with TensorFlow (a different IDCT).
  - The model input now comes from the uploaded bytes through `data_loader.decode_and_resize()`, the same function training uses. PIL is only used to validate and display the image.
  - After the fix: app vs saved test predictions, max diff **1.6e-7** (32 images). The training data path is unchanged (2.2e-8 on 64 images).
  - `AppTest` render checks passed with no exceptions: no upload, two TB images, one Normal image, and garbage bytes.
- **requirements.txt:** direct dependencies only, pinned to the lock-file versions. All 12 pins match the lock, `pip check` is clean, and a dry-run install changes nothing.
  - `opencv-python` (the cause of the Docker libGL problem in §5.6) and `tqdm` are removed, since nothing uses them any more.
  - `requirements-lock.txt` stays as the exact Windows environment. It pins the Windows-only `tensorflow-intel`.
- **PyTorch files** (§5.7): moved with `git mv` to [legacy/pytorch/](../legacy/pytorch/README.md), with a README listing their known bugs. Nothing references them.
- **README** (§5.6, §5.8): rewritten.
  - A one-line honest TL;DR and the test results for all 3 models (tuned threshold, 95% CIs, exact bounds).
  - A before/after table of the preprocessing fix, with the legacy results marked invalid.
  - "Key finding: dataset confounding" (metadata AUC 0.918, lungs-hidden AUC 0.971/0.974/0.912, Grad-CAM gallery, TB.1892), plus limitations.
  - Reproduction steps with measured CPU times, the dataset link (Kaggle, with a pointer to its page for the source and licence), and a repo map.
  - Fixes: the real clone URL, closed code blocks, no "Fully containerized" claim.
- **failure_analysis.md §2:** added point 5. Attention on the bottom strip alone isn't evidence of a shortcut; hot spots exactly on handwriting are.
- **.gitignore:** the merged `Thumbs.db`/`*.npz` line was repaired and the UTF-16 line removed. The `.npz` heatmaps were untracked and removed from history (amended into the goal 5 commit).
- **Still open:** recalibrating the scores; the threshold margin decision from goal 4; Docker, API and model card (goal 8); tests and CI (goal 7).

### Goal 7 — ENGINEERING: done (2026-11-07, branch `upgrade-v4`)
- **Package:** `__init__.py` in `src/`, `src/components/`, `src/pipeline/`, `src/utils/` and `tests/` (fixes the §5.6 namespace-package note).
- **`pyproject.toml`** holds tool configuration only:
  - pytest runs from the repo root (`pythonpath = ["."]`), with markers `data` and `models`.
  - ruff uses the default rule set (E4/E7/E9/F), lint only with no formatting rules, line length 120. It excludes `legacy/` and `notebooks/`.
- **`requirements-dev.txt`:** the runtime requirements plus `pytest==8.3.5` and `ruff==0.9.10`. Installed into `tb_v2_env` after a dry run; only 5 new packages were added and none changed.
- **Ruff:** fixed 25 findings by hand, with no blanket auto-fix:
  - 13 one-line `a; b` statements split (E702).
  - 3 `x = lambda` assignments turned into functions (E731).
  - An unused variable (F841) and an f-string without placeholders (F541).
  - 2 unused imports (F401), checked one by one: `itertools.combinations` in `data_audit.py` and `pandas` in `evaluate.py`. The imports that register `CaffePreprocess` already carry `# noqa: F401` and were kept.
  - Logic is unchanged. Afterwards, all 3 saved models reload and reproduce their saved test predictions (within 1e-4).
- **Code change for testing:** `build_model(..., weights="imagenet")` gains a `weights` parameter (default unchanged), so tests build the real architectures untrained at 32×32 without downloading ImageNet weights.
- **Tests:** 8 files, 49 tests, all on synthetic data except where noted.
  - Label mapping.
  - Preprocessing parity: app path vs training path, exactly equal for JPEG and PNG. These are regression tests for both past bugs, together with a 0–255 range check.
  - `CaffePreprocess` vs Keras `preprocess_input`.
  - L, LA, RGBA and P inputs give `(1, 224, 224, 3)`.
  - Predictions within [0, 1], and save/reload gives identical output.
  - Grad-CAM NaN guard and logit consistency.
  - `make_splits` on a synthetic folder: duplicates dropped, no hash in two splits, stratification within ±1 image, determinism, cross-class duplicate error.
  - Metric, bootstrap and threshold functions on known inputs.
  - Committed `splits.csv`, `duplicates.csv` and `thresholds.json` integrity.
  - **Auto-skip without images or models:** the file-hash leakage check (1 test) and the app-path parity vs `artifacts/predictions/` for all 3 models (3 tests). `TB_TESTS_SIMULATE_CI=1` runs the suite as CI sees it.
- **Local timings** (`tb_v2_env`, CPU):
  - Full suite: **49 passed in 57.8 s**. The first cold run took 82 s, because hashing 2,982 images took 23.8 s when the files weren't cached.
  - CI-simulated: **45 passed, 4 skipped in 37.9 s**.
  - Local runs need `PYTHONNOUSERSITE=1`. A user-site `platformdirs` 4.11.0 (`AppData\Roaming\Python\Python310`) shadows the env's 4.12.4 and breaks pytest's plugin loading. The env is complete without the user site.
- **CI** (`.github/workflows/ci.yml`): ubuntu-latest, Python 3.10, pip cache. Steps: install `requirements-dev.txt`, `ruff check .`, `pytest`. Runs on push to `main`, on pull requests, and manually.
  - Linux install check: every pin has a Linux x86-64 / CPython 3.10 wheel on PyPI. The Linux TensorFlow 2.16.1 wheel requires `tensorflow-intel` only when `platform_system == "Windows"`.
  - Not yet run on GitHub.
- **`.gitattributes`:** `* text=auto eol=lf`, plus binary rules. `git add --renormalize .` changed nothing, because every tracked text file was already stored as LF (`core.autocrlf=true`); only the Windows working copies were CRLF.
- **README:** CI badge, a "Tests and lint" step, and `tests/` and CI in the repo map.

### Goal 9 — EXTERNAL VALIDATION: done (2026-11-07; merged in PR #12)

#### Pre-registered expectations (written 2026-11-07 14:01, before any external predictions existed)
At the time of writing, `artifacts/external/` held only the 10-image preprocessing check (examples, histograms, `sample_*.csv/json`). No external images had been passed through any model. The models, the val-tuned thresholds (ResNet50 0.6142, VGG16 0.2357, EfficientNetB0 0.3257) and the preprocessing were fixed in advance and won't change after seeing external results.

- **H1:** external ROC-AUC will be lower than the internal test AUC of 0.998 (ResNet50; VGG16 0.991, EfficientNetB0 0.996), on both Shenzhen and Montgomery. The size of the drop is not predicted.
- **H2:** Montgomery **specificity** at the val-tuned thresholds will be **low**. Its images carry "R" markers, white "PA ERECT" labels and dark bottom bands, which resemble the training TB source more than the training Normal source (§5.3 item 7, failure_analysis.md).
- **H3:** if some Shenzhen images overlap with the training data, results on those overlapping images will be **inflated** compared with the non-overlapping ones.
- **H4:** if the models rely on compression artefacts, the **JPEG variant** (quality 75, 4:2:0, like the training files) will score a **higher AUC than the lossless PNG** variant.

How they'll be judged (fixed now):
- H1 is supported for a site and model if the external AUC's 95% CI upper bound is below the internal point estimate.
- H2 is supported if Montgomery specificity at the tuned threshold is below 0.80 for the primary model (ResNet50).
- H3 is supported if AUC (or sensitivity/specificity, where AUC is undefined) on the overlapping images is higher than on the rest. It can only be judged if confirmed overlaps exist.
- H4 is supported if JPEG AUC > PNG AUC for a site and model. The size and consistency of the gap across models and sites will be reported, along with the paired bootstrap 95% CI of the difference.

Each hypothesis will be reported below as supported or not supported, with the numbers.

**Addendum (written 2026-11-07 14:21, still before any external predictions existed).** It was prompted by the overlap check: 785 of the 800 external images (98%), Normal and TB alike, have a *training TB* image as their closest thumbnail match.
- **H5:** the models will flag most external Normal images as TB, so **specificity at the val-tuned thresholds will be low on both sites**, while **AUC may hold up better than specificity**.
- **Judged by:** specificity < 0.80 on both sites for the primary model (ResNet50), and AUC − specificity > 0 on both sites. As with H2, ResNet50 is judged first and the other two models are reported alongside.

**Addendum 2 (written 2026-11-07 14:43, after the evaluation and before any external Grad-CAM had been computed).**
- **H6:** on Montgomery, the share of Grad-CAM energy **inside the true lung fields** (left + right `ManualMask`) will **not exceed** the lung-area share of the image, for ResNet50 and for VGG16.
- **Method, fixed now:**
  - Masks follow exactly the image geometry: original size → 256×256 (PIL, nearest) → 224×224 (TensorFlow, nearest). That's the same stretch as the image path (Lanczos to 256, bilinear to 224), with nearest-neighbour interpolation so masks stay binary.
  - A unit test checks mask/image alignment, and a 4-image overlay figure is saved for a visual check.
  - Maps: TB-evidence map (primary) and predicted-class map (secondary), at the same layers as goal 5 (ResNet50 `conv5_block3_out`, VGG16 `block5_conv3`).
  - Per image: inside-lung energy = Σ(heatmap · mask) / Σ heatmap, after upsampling to 224×224. Lung-area share = mean(mask).
- **Judged by:** H6 is supported for a model if the median of (inside-lung energy − lung-area share) over the Montgomery images is ≤ 0 on the primary map. Empty maps are excluded and counted. A Wilcoxon signed-rank p-value and per-class / correct-vs-wrong breakdowns are reported alongside. H6 is supported overall only if it holds for both models.

#### Data audit of the external sets
Same `data_audit.py` functions, unchanged; labels come from the `_0`/`_1` file suffix. Outputs: `artifacts/external/data_audit/`.
- **Duplicates:** no exact or near-duplicates (thumbnail correlation ≥ 0.99) within either site.
- **Metadata-only shortcut test** (5-fold CV ROC-AUC; logistic regression / depth-3 tree):

| features | training (Kaggle) | Shenzhen | Montgomery |
|---|---|---|---|
| width + height + aspect | 0.525 / 0.525 | 0.711 / 0.742 | 0.578 / 0.578 |
| file size | 0.784 / 0.861 | 0.532 / 0.655 | 0.558 / 0.533 |
| brightness / contrast | 0.744 / 0.803 | 0.705 / 0.721 | 0.602 / 0.585 |
| **all metadata** | **0.911 / 0.918** | **0.855** / 0.791 | **0.637** / 0.564 |

- **Shenzhen is flagged** (all-metadata AUC 0.855 ≥ 0.80): its labels are partly guessable from acquisition metadata, mostly image size and brightness/contrast.
  - That would limit Shenzhen as a test for **models trained on Shenzhen**. Our models are frozen and never saw it, so they can only exploit cues that also exist in the training data.
  - Shenzhen's brightness difference points the **opposite way** to the training data's: Shenzhen Normal is brighter than TB (median 161 vs 151), while in training TB is brighter than Normal (136 vs 132).
  - The cue that predicts Shenzhen labels is therefore not one the models learned. Image size is removed by the resize, except for the stretch.
- **Montgomery** (0.637, not flagged) is the **cleanest external test**, but it is small (n = 138: 80 Normal, 58 TB), so its CIs will be wide.
- **Intensity shift vs training** (median of 64×64 thumbnails): Shenzhen is brighter (151–161 vs 132–136). Montgomery has far more dark and saturated area (18% of pixels < 10 and 5% > 245, vs ≤ 0.3% and 0% in training). No intensity normalisation was applied; this shift is part of what is being tested.

#### Inventory, preprocessing and overlap
- **Script:** [src/pipeline/external_validation.py](../src/pipeline/external_validation.py), run in stages (`--stage sample | audit | prepare | readings | overlap | cropsearch | evaluate | gradcam`). Outputs go to `artifacts/external/`; converted images to `data/external/prepared/` (not committed).
- **Inventory:**
  - Shenzhen: 662 images (326 Normal / 336 TB), 635 palette + 27 RGB, all truly gray, 8-bit, 948–3001 px, aspect w/h 0.76–1.30.
  - Montgomery: 138 images (80 / 58), 8-bit grayscale, 4020×4892 portrait (97) or 4892×4020 landscape (41). The NLM ReadMe states "12 bits" of gray levels, but the distributed PNGs are 8-bit: mode `L`, maximum 255. So no bit-depth rescaling was possible or needed.
  - Both counts match the expected numbers. 11 macOS/Windows junk files were ignored.
- **Clinical readings:** parsed strictly. 7 Shenzhen readings were corrected by hand where the meaning was unambiguous (`femal` ×3; `16month` ×2 → 1.33 y; `64days` → 0.18 y; `Male , 38yrs,`), recorded in `readings_corrected.csv`. 2 stay unknown: `male 42` (no unit) and Montgomery sex "O".
- **Preprocessing:**
  1. Decode to 8-bit gray. A 16-bit path that scales by the real bit depth exists but was not needed.
  2. Lanczos downscale to 256×256, stretching non-square images like the training set.
  3. Lossless PNG (primary) and JPEG at the training files' quality (75, 4:2:0; read from their quantisation tables).
  4. The unchanged `decode_and_resize` to 224×224.
  - No intensity normalisation (decided in advance): the intensity shift is part of the test.
  - Parity: the evaluation path reproduces the saved internal test predictions for all 3 models (max diff ≤ 2.2e-8).
- **Overlap with the 2,982 training images** (32×32 thumbnail correlation, `data_audit.sim_vector`):
  - Whole image: best match median 0.833 (Shenzhen) / 0.750 (Montgomery), max 0.924 / 0.874. None ≥ 0.95.
  - For scale: different training images best-match each other at a median of 0.874.
  - **Crop search** (81 crops per image; a positive control of cropped, re-saved copies is recovered at 0.949–0.979, median 0.979): max 0.946 / 0.943. None ≥ 0.95.
  - Visual check of the top 10 pairs from each method (non-expert): all are different radiographs.
  - **Confirmed overlaps: 0.** The automatic ≥ 0.95 exclusion removes 0 images, so results with and without exclusions are identical.
  - **Limitation:** overlap after heavier cropping or other transformations can't be fully ruled out.

#### Results (frozen models, val-tuned thresholds unchanged, lossless PNG)
95% CIs: stratified bootstrap, 2,000 resamples, seed 42. Values truncated to 3 decimals. Full tables: `artifacts/external/evaluation/summary.md`.

| model | data | n (N/TB) | ROC-AUC | sensitivity | specificity | PPV | NPV | Brier |
|---|---|---|---|---|---|---|---|---|
| **ResNet50** | internal test | 448 (77/371) | 0.998 (0.996–0.999) | 0.935 (0.908–0.959) | 1.000 (77/77) | 1.000 | 0.762 | 0.026 |
| | Shenzhen | 662 (326/336) | **0.669** (0.628–0.709) | 0.907 (0.875–0.937) | **0.208** (0.165–0.254) | 0.541 | 0.686 | 0.320 |
| | Montgomery | 138 (80/58) | **0.656** (0.559–0.748) | 1.000 | **0.000** | 0.420 | undefined\* | 0.554 |
| | pooled (secondary) | 800 (406/394) | 0.619 (0.581–0.657) | 0.921 | 0.167 | 0.517 | 0.686 | 0.361 |
| VGG16 | internal test | 448 | 0.991 (0.982–0.998) | 0.929 | 0.987 | 0.997 | 0.745 | 0.072 |
| | Shenzhen | 662 | 0.470 (0.427–0.514) | 0.699 | 0.165 | 0.463 | 0.348 | 0.383 |
| | Montgomery | 138 | 0.581 (0.479–0.683) | 0.965 | 0.000 | 0.411 | 0.000 | 0.516 |
| EfficientNetB0 | internal test | 448 | 0.996 (0.991–0.999) | 0.946 | 0.987 | 0.997 | 0.791 | 0.071 |
| | Shenzhen | 662 | 0.521 (0.477–0.565) | 0.476 | 0.533 | 0.512 | 0.497 | 0.289 |
| | Montgomery | 138 | 0.537 (0.442–0.639) | 1.000 | 0.012 | 0.423 | 1.000 | 0.384 |

\* ResNet50 flags every Montgomery image as TB, so no image is predicted Normal and NPV is undefined.

**Oracle analysis** (*not a result*: the threshold is chosen on the external data itself). The threshold ResNet50 would need for 95% sensitivity is 0.517 on Shenzhen, giving specificity 0.141, and 0.911 on Montgomery, giving specificity 0.025. No operating point rescues the models on these sites.

**Interpretation:**
- **(a) The site affects the score far more than the disease does.**
  - EfficientNetB0's mean P(TB) moves by about 0.44 between sites (Shenzhen 0.327 / 0.351 → Montgomery 0.771 / 0.789 for Normal / TB), but only about 0.02 between Normal and TB within a site.
  - ResNet50's mean P(TB) for Normal images goes from 0.051 internally to 0.751 (Shenzhen) and 0.978 (Montgomery). The TB − Normal gap within a site is only +0.096 and +0.005.
- **(b) VGG16 ranks Normal slightly above TB on Shenzhen** (AUC 0.470, CI 0.427–0.514, which includes 0.5). On Montgomery its AUC is 0.581 and the median P(TB) is higher for TB (0.996 vs 0.986); only the mean is higher for Normal (0.908 vs 0.891).
- **(c) Every external Brier score is worse than a constant base-rate predictor:** π(1 − π) = 0.250 (Shenzhen), 0.244 (Montgomery), 0.250 (pooled). The models' Brier scores are 0.289–0.555.
- **(d) Shenzhen subgroups:** ResNet50 AUC is 0.609 (0.532–0.680) for women and 0.698 (0.651–0.744) for men. The CIs overlap, so **no sex disparity is claimed**.
  - Specificity is 0.142 for women and 0.240 for men.
  - The ≥ 60 group is small (17 Normal), with specificity 0.058.
  - Every Montgomery subgroup has fewer than 30 images in a class (flagged) and specificity 0.000.
- **Bottom line:** the internal AUC of 0.998 doesn't transfer. ResNet50 keeps some real signal (AUC about 0.66 on both sites, with CIs above 0.5), but at the val-tuned threshold it flags almost every external image as TB. VGG16 and EfficientNetB0 are at chance.

#### Pre-registered hypotheses: verdicts
| | verdict | numbers |
|---|---|---|
| **H1** external AUC < internal | **Supported**, 6 of 6 model–site pairs | Every external CI upper bound (≤ 0.748) is below the internal point estimate (≥ 0.991) |
| **H2** Montgomery specificity low | **Supported** | ResNet50 specificity 0.000 (< 0.80); VGG16 0.000, EfficientNetB0 0.012 |
| **H3** overlap inflates results | **Not testable** | 0 confirmed overlaps (whole-image check and crop search) |
| **H4** JPEG AUC > PNG AUC | **Not supported** | The literal rule was met in 4 of 6 point estimates, but every per-site paired 95% CI of (JPEG − PNG) includes 0, and \|diff\| ≤ 0.012. The pre-registered rule was **underspecified**: it had no CI requirement. The pooled EfficientNetB0 CI excluding 0 (+0.009, CI +0.0001 to +0.018) is a secondary analysis and negligible in size. |
| **H5** most external Normals flagged TB; AUC holds better than specificity | **Prediction supported; stated reason only partly supported** | ResNet50 specificity 0.208 / 0.000 (< 0.80) and AUC 0.669 / 0.656 (> specificity) on both sites, so the prediction holds. The reason (98% of external images most resemble a training TB image) depends on the matching method. With crop search, the share closest to a training **Normal** image rises from 2% to 24% (193/800), and all top-10 strongest matches are Normal, but 76% are still closest to a training TB image. *Exploratory, post hoc:* on Shenzhen, Normal images whose crop-search neighbour is a training TB image get higher ResNet50 scores than those closest to a training Normal (mean 0.791 vs 0.674, Mann–Whitney p = 3e-7), which is consistent with the reason. On Montgomery all scores are saturated (≈ 0.98), so it can't be tested there. |
| **H6** Grad-CAM lung energy ≤ lung area (Montgomery) | **Supported** for both models | Median inside-lung energy vs median lung-area share: ResNet50 0.186 vs 0.257 (median per-image difference −0.070, Wilcoxon p = 8e-21); VGG16 0.169 vs 0.257 (−0.086, p = 3e-16). 0 empty maps. Energy exceeds area in only 11% (ResNet50) / 22% (VGG16) of images. |

#### Grad-CAM on the external sets (stage `gradcam`; 13 min 18 s on CPU)
- **Mask geometry:** masks follow the image path (original → 256 → 224, nearest-neighbour). A unit test with a synthetic non-square image confirms alignment (IoU > 0.95), and [mask_alignment.png](../artifacts/external/gradcam/mask_alignment.png) shows 4 real Montgomery images with the mask contour on the original and on the model input. The Grad-CAM forward pass reproduced the saved predictions (< 1e-4).
- **Lung-mask energy (Montgomery, TB-evidence map, medians):**

| model | group | n | energy inside lungs | lung-area share | energy − area |
|---|---|---|---|---|---|
| ResNet50 | Normal | 80 | 0.162 | 0.246 | −0.091 |
| ResNet50 | TB | 58 | 0.230 | 0.263 | −0.035 |
| VGG16 | Normal | 80 | 0.143 | 0.246 | −0.100 |
| VGG16 | TB | 58 | 0.225 | 0.263 | −0.025 |
| VGG16 | correct / wrong | 56 / 82 | 0.231 / 0.143 | 0.262 / 0.250 | −0.023 / −0.100 |

  - ResNet50 flags every Montgomery image as TB, so for ResNet50 "correct" = TB and "wrong" = Normal.
  - **Resolution-ceiling control** (robustness check, not a new hypothesis; stage `ceiling`): each lung mask is averaged onto the Grad-CAM grid and upsampled exactly like the heatmaps. That gives the lung-energy share of a map that is *perfectly* on the lungs at that resolution.
    - Median ceiling **0.682** for ResNet50 (7×7) and **0.825** for VGG16 (14×14), against observed **0.186** / **0.169** and a lung-area share of 0.257.
    - The coarse grid would allow up to 0.68–0.82, so the low observed share is not a resolution artefact. Outputs: `artifacts/external/gradcam/h6_resolution_ceiling.{json,csv}`.
  - TB images draw more energy into the lungs than Normal images, but even for TB it stays below the lungs' area share.
- **Goal 4's "lungs hidden" box, checked with real anatomy:** on Montgomery, a median 94% (minimum 80%) of the true lung area lies inside the central 170×170 box masked in goal 4. So that test hid most of the lungs, at least for this image geometry.
- **Border energy vs the internal test** (TB-evidence map, outer 12% border, medians; uniform reference 0.424):

| model | class | internal test | Shenzhen | Montgomery |
|---|---|---|---|---|
| ResNet50 | Normal | 0.437 | 0.606 | 0.471 |
| ResNet50 | TB | 0.332 | 0.530 | 0.422 |
| VGG16 | Normal | 0.523 | 0.777 | 0.517 |
| VGG16 | TB | 0.449 | 0.690 | 0.444 |

  On Shenzhen both models put far more of their TB evidence on the border than on the internal test.
- **Galleries** ([gallery_shenzhen.png](../artifacts/external/gradcam/gallery_shenzhen.png), [gallery_montgomery.png](../artifacts/external/gradcam/gallery_montgomery.png); ResNet50, predicted-class maps; qualitative, non-expert):
  - **Montgomery:** hot spots in false and true positives often sit on the white "PA ERECT" label and the shoulders and edges; the lung fields are mostly cool. There are no FN or TN to show.
  - **Shenzhen:** false positives light up corner "L"/"ap" markers and lateral edges. Some true positives highlight visible lung opacities, consistent with the remaining signal (AUC 0.669). In FN and TN, the Normal evidence sits along the bottom band.
- **Tests:** `tests/test_external.py` (23 tests) covers the readings parser and corrections, age bands, grayscale and 16-bit conversion, crop search (finds a cropped copy), mask/image alignment, a real-mask check (skips without data), and the metric/subgroup helpers.

#### Data sources and use terms
- **Shenzhen:** National Library of Medicine, NIH (Bethesda, MD, USA) and Shenzhen No.3 People's Hospital, Guangdong Medical College (Shenzhen, China).
- **Montgomery:** National Library of Medicine, NIH, with the Montgomery County (MD) Department of Health and Human Services.
- **Cited:**
  - Jaeger S, Candemir S, Antani S, Wáng YX, Lu PX, Thoma G. Two public chest X-ray datasets for computer-aided screening of pulmonary diseases. *Quant Imaging Med Surg.* 2014;4(6):475–477.
  - Jaeger S, et al. Automatic tuberculosis screening using chest radiographs. *IEEE Trans Med Imaging.* 2014;33(2):233–245. doi:10.1109/TMI.2013.2284099.
  - Candemir S, et al. Lung segmentation in chest radiographs using anatomical atlases with nonrigid registration. *IEEE Trans Med Imaging.* 2014;33(2):577–590. doi:10.1109/TMI.2013.2290491.
- **Use terms:** both NLM ReadMes ask users **not to share the dataset outside their research group or organisation**. The images and clinical readings are not committed (`data/` is gitignored).
- **Decision (cautious option):** the repository contains only:
  - summary-level outputs;
  - per-image files without patient data (file name, label, scores, attention shares);
  - predictions reduced to `site, file, label, prob` (`artifacts/external/predictions_public/`);
  - image-free figures, including `gradcam/h6_lung_energy.png`.
- **Kept local** (gitignored):
  - per-image clinical readings: `inventory.csv`, `readings_*.csv`, `sample_preprocessing.csv`;
  - the full prediction CSVs (`predictions/`, with age, sex and findings);
  - the 6 figures that show NLM images (galleries, mask alignment, overlap pairs, crop-search top 10, preprocessing examples).
  - A dry-run staging check confirmed that none of these would be committed, and that no committed CSV has a sex, age or findings column.

### Goal 10 — SHORTCUT REMOVAL: done (2026-11-07, branch `upgrade-v4`). **Negative result:** the primary model failed S1, S4a and S4b.

#### Pre-registration (written 2026-11-07 15:40, before any goal 10 code, training or results)
At the time of writing, no goal 10 model, prediction or figure existed. The only models were the goal 3 models in `models/tensorflow/` and the old smoke-test folder.

**Design (fixed now):**
- **Training data:** Kaggle only (the train split of `data/splits.csv`). Val is used for early stopping and thresholds; test is the internal test. Shenzhen and Montgomery stay fully external. Kaggle + Shenzhen will be goal 10b, pre-registered separately.
- **Models:** ResNet50 and EfficientNetB0 (VGG16 dropped). ImageNet weights, same head as goals 2–3.
- **Inside the model** (so the external pipeline is unchanged and still feeds raw 0–255 RGB at 224×224):
  1. **Central crop:** remove 12% of each edge (27 px at 224), then bilinear resize back to 224×224.
  2. **Per-image standardisation:** z-score over the image, then rescale to mean 128 and SD 50, clipped to 0–255, so the backbone gets inputs in its expected range.
  3. ResNet50: then `CaffePreprocess`. EfficientNetB0: nothing more.
- **Training augmentation** (training split only). Every random choice is **independent of the label**, with the same probability for both classes:
  - synthetic markers and text ("L", "R", "PA", "AP", "PA ERECT", dates, handwriting-like strokes; white, black or boxed; 1–3 items placed anywhere), probability 0.5;
  - JPEG re-encoding at quality U(50, 95), probability 0.7;
  - brightness ± 10% of 255, contrast × U(0.7, 1.3), gamma U(0.7, 1.4);
  - the existing rotation ± 10° and zoom 0.1. No flips.
- **Arm 1:** frozen backbone, head trained (≤ 20 epochs, EarlyStopping on val AUC, patience 4).
- **Arm 2:** starts from Arm 1's weights. Fine-tunes ResNet50 `conv5_*` or EfficientNetB0 `block6a` onwards plus `top`. BatchNorm stays frozen. Learning rate 1e-5, ≤ 15 epochs, patience 4.
- **Primary model, fixed now: ResNet50 Arm 2.** The Kaggle val set is confounded, so no arm or model is chosen by val or external results. All 4 combinations (2 models × 2 arms) are reported.
- **Thresholds:** same rule as before (highest threshold with val sensitivity ≥ 0.95), on the new val predictions.
- **Evaluation:** the goal 9 external pipeline is unchanged: same prepared images, same decode path, same statistics.
  - Allowed changes: output paths via a config "tag", so goal 4/9 results are never overwritten.
  - One required geometry change for Grad-CAM: for models with the internal crop, heatmaps are mapped back into the 224×224 input frame (resized into the central 170×170, zero outside) before lung-mask and border shares are computed.
  - A regression test confirms the default configuration still reproduces the goal 9 predictions.

**Success criteria** (comparisons against the goal 9 model of the same architecture, on the same images; 95% CIs from a stratified bootstrap with 2,000 resamples, seed 42):
| | criterion | decision rule |
|---|---|---|
| **S1** (primary endpoint) | external AUC rises | ResNet50 Arm 2: the paired bootstrap 95% CI of ΔAUC (goal 10 − goal 9 ResNet50, lossless PNG) lies entirely above 0 on **both** Shenzhen and Montgomery. Secondary: AUC ≥ 0.80 on each site. |
| **S2** (primary version) | lungs-hidden AUC drops | Internal test, goal 4 method (fill = the image's own mean grey). The hidden region is the central box of the **cropped field**, leaving 12% of that field on each side: input pixels 47–176, so 130×130 hidden and a 20-px ring of the cropped field visible. Success: AUC ≤ 0.75 (goal 4 lungs-hidden AUC was 0.971). The original goal 4 version (central 170×170 of the input hidden) is also reported, labelled **trivially affected by the crop**: it hides the whole cropped field. |
| **S3** | more Grad-CAM energy inside the lungs (Montgomery) | Primary model, TB-evidence map mapped into the input frame: median over images of (energy inside the lung masks − lung-area share) **> 0**. Reported alongside: a paired Wilcoxon test against goal 9 ResNet50 (per image), and the resolution ceiling. |
| **S4a** | weaker shortcut signal in the processed inputs | The `data_audit` shortcut test (5-fold CV ROC-AUC, logistic regression and depth-3 tree, seed 42) on what the backbone sees: the cropped, standardised model input for all 2,982 Kaggle images. Features: dark share (< 10), bright share (> 245), the 5th / 50th / 95th percentiles, skewness and kurtosis of a 64×64 thumbnail, plus original width, height and aspect. Mean and SD are constant after standardisation by design, so they're excluded. Success: all-feature AUC ≤ 0.75 for both models' inputs. The raw audit was 0.918, and some real pathology signal in intensity is expected. |
| **S4b** | site effect no longer dominates | Primary model, PNG external predictions. Within-site gap gₛ = mean P(TB \| TB) − mean P(TB \| Normal), per site. Between-site shift d = \|mean P(TB \| Normal, Shenzhen) − mean P(TB \| Normal, Montgomery)\|. Success: min(g_Shenzhen, g_Montgomery) > d. Goal 9 ResNet50: g = 0.096 / 0.005 vs d = 0.227. |

- **Expected, not a failure:** the internal test AUC (0.998 for goal 9 ResNet50) may **drop** once the shortcuts are gone. It will be reported but not counted.
- **Secondary result (not a criterion):** external specificity at the new val-tuned thresholds.
- **Reporting:** each criterion is reported on its own for all 4 model–arm combinations, with the primary model's verdict decisive. Any analysis added after seeing results is labelled post hoc.

*(The pre-registration above is unchanged. Everything below was written after the results.)*

#### Training (run by the owner in PowerShell; logs in `logs/train_v10_arm{1,2}.log`)
| | epochs run | best epoch | best val AUC | median s / epoch | trainable params |
|---|---|---|---|---|---|
| ResNet50 Arm 1 | 20 (cap) | 20 | 0.9944 | 169.7 | 2,049 |
| EfficientNetB0 Arm 1 | 20 (cap) | 20 | 0.9964 | 85.8 | 1,281 |
| ResNet50 Arm 2 | 13 (early stop) | 9 | 1.0000 | 184.6 | 14,955,521 |
| EfficientNetB0 Arm 2 | 9 (early stop) | 5 | 1.0000 | 150.4 | 3,131,165 |

- **Logs:** each has one warning, the harmless `tf.placeholder` deprecation notice. No errors.
- **Reload:** all 4 models reload with the crop and standardisation layers present, and their trainable counts match the smoke tests.
- **Parity (CLAUDE.md rule):** the app path (`inference.preprocess_bytes` → `predict`) reproduces each model's saved test predictions on 32 images (16 Normal, 16 TB), max difference 6.3e-07. The external pipeline's own parity check passes too (max difference 5e-09). pytest: 87 passed, including the goal 9 regression test.

#### Results (truncated to 3 decimals; 95% CIs from a stratified bootstrap with 2,000 resamples, seed 42; external = lossless PNG)
| | **ResNet50 Arm 2 (primary)** | ResNet50 Arm 1 | EfficientNetB0 Arm 1 | EfficientNetB0 Arm 2 |
|---|---|---|---|---|
| Val-tuned threshold (pre-registered rule) | 0.9999972 (see flag 1) | 0.2437 | 0.1788 | 0.2267 |
| Internal test AUC (not a criterion) | 0.999 | 0.993 | 0.990 | 0.999 |
| **S1** Shenzhen AUC, goal 9 → goal 10; ΔAUC (95% CI) | 0.669 → 0.502; −0.167 (−0.207 to −0.125) | 0.669 → 0.445; −0.224 (−0.268 to −0.179) | 0.521 → 0.653; +0.131 (+0.089 to +0.175) | 0.521 → 0.605; +0.084 (+0.039 to +0.129) |
| **S1** Montgomery AUC, goal 9 → goal 10; ΔAUC (95% CI) | 0.656 → 0.572; −0.084 (−0.181 to +0.011) | 0.656 → 0.673; +0.017 (−0.091 to +0.118) | 0.537 → 0.642; +0.105 (+0.014 to +0.193) | 0.537 → 0.684; +0.146 (+0.049 to +0.239) |
| **S1** verdict (secondary: AUC ≥ 0.80 on both sites) | **not supported** (no) | not supported (no) | supported (no) | supported (no) |
| **S2** internal AUC with the centre of the cropped field hidden (≤ 0.75 needed) | **0.659 (0.600–0.715): supported** (flag 1) | 0.938: not supported | 0.965: not supported | 0.975: not supported |
| S2, goal 4 version (trivially affected by the crop) | 0.520 | 0.488 | 0.434 | 0.432 |
| **S3** Montgomery median (lung energy − lung area) (> 0 needed) | **+0.124 (93% of images > 0): supported** (flag 2) | −0.010: not supported | +0.004: supported | +0.025: supported |
| S3 median lung energy; lung-area share 0.257; resolution ceiling 0.787 | 0.368 | 0.239 | 0.245 | 0.269 |
| S3 paired Wilcoxon vs goal 9 ResNet50 (median paired difference; p) | +0.198; p = 2.1e-24 | +0.064; p = 4.4e-08 | +0.063; p = 5.9e-07 (flag 3) | +0.096; p = 9.3e-11 (flag 3) |
| S3 empty TB-evidence maps (of 138) | 0 | 0 | 8 | 4 |
| **S4a** processed-input shortcut test, all-feature AUC (≤ 0.75 needed) | **0.883: not supported** | the same inputs (identical layers, max difference 0.0) | the same | the same |
| **S4b** g Shenzhen / Montgomery vs d (min g > d needed) | **0.016 / −0.000 vs 0.094: not supported** | −0.015 / 0.079 vs 0.483: not supported | 0.061 / 0.099 vs 0.467: not supported | 0.079 / 0.155 vs 0.501: not supported |
| Secondary: external specificity (Shenzhen / Montgomery) | 0.625 / 0.062 | 0.592 / 0.012 | 0.880 / 0.025 | 0.926 / 0.162 |
| (external sensitivity at the same threshold) | 0.342 / 0.948 | 0.345 / 0.982 | 0.306 / 0.913 | 0.208 / 0.862 |

- **S4a details:** logistic regression 0.883, depth-3 tree 0.834. The rule uses the higher of the two, as the raw audit headline 0.918 did; both exceed 0.75. Width + height + aspect alone give 0.525, so the signal is in the intensity features.
- **Goal 9 reference values** recomputed from the saved files: g = 0.096 / 0.005, d = 0.2264 (ResNet50). The pre-registration printed d as 0.227, which rounds up; truncated, it is 0.226. The verdict is the same either way.
- **Sources:**
  - `artifacts/v10_comparison/criteria.json`, `s4a.json` and `s4a_shortcut_test.csv` (code: `src/pipeline/goal10_compare.py`);
  - `artifacts/v10_arm{1,2}/clinical/` (thresholds, metrics, masking);
  - `artifacts/v10_arm{1,2}/external/` (share-safe parts only; see "Sharing" below).

#### Verdicts by the pre-registered rules (primary model: ResNet50 Arm 2)
| criterion | verdict | comment |
|---|---|---|
| **S1** (primary endpoint) | **Not supported** | External AUC **fell** on both sites. On Shenzhen the CI is entirely below 0; Shenzhen AUC is now at chance (0.502). |
| **S2** | Supported by the rule, but **does not show the shortcut was removed** | Flag 1: the scores are saturated. |
| **S3** | Supported by the rule, but **does not show lung-focused attention** | Flag 2: below a uniform map over the cropped field. |
| **S4a** | **Not supported** | 0.883 on the processed inputs (raw audit 0.918). |
| **S4b** | **Not supported** | The Montgomery within-site gap is −0.000: no class separation there. |

- **Secondary models:**
  - EfficientNetB0 passes S1 in both arms, but from goal 9's chance level (0.521 / 0.537) to 0.60–0.68, below the 0.80 secondary bar.
  - EfficientNetB0 fails S2, S4a and S4b, and passes S3 only by small margins.
  - ResNet50 Arm 1 fails everything.
- **Internal test AUC rose** (0.999) instead of dropping, which is consistent with the shortcut still being learned.

#### Flags (found while evaluating; reported, not fixed)
1. **Sigmoid saturation behind the S2 pass and the threshold.**
   - ResNet50 Arm 2 gives exactly 1.0 (float32) to 329 of 371 val TB images and 312 of 371 test TB images.
   - The val classes separate completely (Normal max 0.333, TB min 0.641). The pre-registered rule (highest threshold with val sensitivity ≥ 0.95) therefore picks 0.9999972, inside the saturated band.
   - With the centre of the cropped field hidden, mean P(TB) is 1.0 for both Normal and TB. The S2 AUC of 0.659 comes from tiny differences between near-1 scores, not from a loss of the shortcut.
   - The same saturation explains the Montgomery result: every image scores about 0.999, so the Brier score is 0.579 with a zero-width CI (the Normal share, 80/138).
2. **S3 is mostly crop geometry (post hoc check).**
   - The crop puts all Grad-CAM energy inside the central 170×170 field, where lung covers a median share of **0.413**, not the whole-image 0.257.
   - Against this uniform-in-crop baseline, every model is below: median −0.041 for the primary (26% of images above), −0.173 / −0.143 / −0.130 for the others.
3. **The S3 Wilcoxon for EfficientNetB0 compares across architectures.** Goal 9 ran no EfficientNetB0 Grad-CAM, so the only goal 9 reference is ResNet50.
4. **Border 12% masked = unmasked, exactly, for all 4 models.** This is expected: the internal crop removes that same border.

#### Post hoc: which processed-input statistics carry the signal (S4a features; not a criterion)
- **Method:** the S4a depth-3 tree and logistic regression, refit on all 2,982 images without CV, to describe the signal (`goal10_compare --stage s4a_posthoc`, output `artifacts/v10_comparison/s4a_posthoc.json`).
- **Tree splits:** the tree uses only **kurtosis, p95 and skewness**.
  - Root: kurtosis ≤ −1.21 sends 1,067 images, of which only 20 are Normal (98% TB).
  - The remaining 1,915 images (489 Normal) split on p95 ≤ 195.2 (581 images, 29 Normal), then skewness.
  - Width, height, aspect, the percentiles p5 / p50 and the dark / bright shares are unused. Impurity importances: kurtosis 0.346, skewness 0.330, p95 0.323.
- **Class medians:**
  - kurtosis −1.039 (Normal) vs −1.153 (TB);
  - p95 205.1 vs 199.9;
  - skewness −0.020 vs −0.058.
- **Standardised LR coefficients:** p5 +3.61, p95 −2.34, bright share +1.78, skewness −1.60, dark share +1.31, kurtosis +0.74 (the rest |coef| < 0.73). The features are strongly collinear, so individual coefficients are not interpretable.
- **Reading:** after standardisation, the histogram **shape** (flatter, less bright tail in TB images) still separates the classes. This analysis cannot say how much of that is pathology and how much is source (scanner and post-processing). Given goal 9, the source is the likely main part.

#### Code changes made while evaluating (paths and labels only; no criterion, threshold rule or statistic changed)
- `external_validation.load_or_predict`: the parity check read the goal 4 predictions from a hard-coded `artifacts/predictions/`. It now reads the active config's `predictions_dir` (the same model's `evaluate.py` output). With the old path, the check would have compared goal 10 models against goal 3 predictions and failed.
- `evaluate_clinical`: the shared-errors table header was hard-coded for 3 models. It now lists the config's models. With 2 models, its "all 3" columns are 0 by construction.
- New: `src/pipeline/goal10_compare.py` (S4a, the post hoc S4a diagnostic, and the S1–S4b comparison).

#### Limitations
- **Early stopping, checkpoint selection and thresholds all use a confounded val set.** Val comes from the same two Kaggle sources as train, so it rewards the shortcut as much as TB signal.
  - Arm 2 reached val AUC 1.0000 within 4–8 epochs. The kept epoch was then a choice between ties, not evidence of generalisation.
  - The val-tuned thresholds inherit the same confounding; external specificity at those thresholds is 0.012–0.162 on Montgomery.
- **Arm 1 was stopped by the 20-epoch cap**, not by early stopping (val AUC still rising, 0.9940 → 0.9944 at epoch 20). A longer Arm 1 would also have changed Arm 2's starting point.
- **Sigmoid saturation:** float32 probabilities of exactly 1.0 create ties that distort AUCs, bootstrap CIs and threshold tuning. In future, rank and tune on **logits** (pre-sigmoid scores) rather than probabilities.
- One seed and one training run per arm; no estimate of run-to-run variation.

#### Conclusion
- Removing borders and markers, standardising intensities and adding label-independent anti-shortcut augmentation **did not remove the shortcut**.
- When image source and label coincide (Normal and TB from different collections), a model can still learn the source from what is left: histogram shape, texture and acquisition traits inside the lung field. A val set from the same sources cannot detect this.
- This is the failure mode described by DeGrave, Janizek & Lee (*AI for radiographic COVID-19 detection selects shortcuts over signal*, Nature Machine Intelligence 3, 610–619, 2021). Their models also learned source-specific features and lost accuracy on external data.
- A **mixed-source design**, where each source contributes both classes (goal 10b, Kaggle + Shenzhen), is required before any claim about TB signal. Its val and test splits should also be stratified by source.

#### Sharing (NLM data-use terms, as in goal 9)
- **Same rules as goal 9.**
- **Committed:**
  - the aggregate metrics;
  - the image-free `h6_lung_energy.png`;
  - the `predictions_public/` files (site, file, label, prob);
  - the Grad-CAM attention CSVs (no age, sex or findings).
- **Kept local** (gitignored under `artifacts/v10_arm*/external/`): full per-image predictions (with sex and age), Grad-CAM galleries and the mask-alignment figure, which show NLM images.

### Goal 8: not started. Goal 10b (mixed-source training): optional, needs its own pre-registration
