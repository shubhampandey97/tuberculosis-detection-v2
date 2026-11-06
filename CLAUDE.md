# Context
Portfolio project for ML/Data Science job applications (target: medtech/healthcare AI roles).
Owner: Shubham Pandey. Keep all results honest; never invent or round up metrics.
docs/audit.md is the full code audit (sections §1–§6). Refer to it by section number.

# Environment
- Windows, conda env tb_v2_env, CPU only. TF 2.16.1, Keras 3.12.4, protobuf<5, Streamlit 1.60.
- Always run Python with: C:\Users\Admin\anaconda3\envs\tb_v2_env\python.exe
- Run modules from the repo root, e.g. ...\python.exe -m src.pipeline.train_pipeline
- requirements-lock.txt = exact working environment. Never upgrade protobuf above 4.x.

# Dataset facts (from audit §2, §5.3)
- data/Dataset of Tuberculosis Chest X-rays Images/: 514 Normal ("Normal Chest X-rays/"), 2,494 TB ("TB Chest X-rays/").
- Labels: 0 = Normal, 1 = TB (alphabetical folder order). Sigmoid output = P(TB). This is correct; don't change it.
- 26 byte-identical duplicate pairs (52 files), within-class.
- Possible shortcut: 124 TB images are 96x96 (upscaled), all Normal are 256x256.
- No patient IDs, so patient-level splitting is impossible (state this as a limitation).

# Root causes of the bad results (audit §5.1–5.3), all to be fixed
- Training rescales /255, but evaluation, Grad-CAM and the app use raw 0–255. No model gets its own preprocess_input.
- The training loop saves all 3 models to vgg_model.h5 (train_pipeline.py:16,46,61-62).
- No test set: evaluation reuses the validation split from training.
- Old metrics in artifacts/metrics/ are INVALID. Move them to artifacts/legacy/ (keep them for a before/after comparison) and never report them as results.

# Working rules
- Show a plan first and wait for approval before editing.
- One goal per session. Keep changes small and reviewable.
- Before any full training run, do a 1-epoch smoke test on ~200 images.
- Full training is slow on CPU: tell me the expected time before starting.
- After each goal: list the changed files, summarise results, and update the "Status" section of docs/audit.md.
- Never delete data. Never commit data/ or models/.

# Upgrade goals (in order)
1. DATA: script to remove the duplicate pairs, record width/height per image, create a stratified seeded 70/15/15
   split saved to data/splits.csv (path, label, split, width, height). Report class counts per split.
2. PIPELINE FIX: load from splits.csv; remove /255; put per-model preprocessing inside build_model
   (VGG16/ResNet50 use preprocess_input, EfficientNetB0 none); fix the save-path bug; global seeds;
   EarlyStopping + ModelCheckpoint on val AUC; read batch/img size from config. Evaluation on the TEST split only.
3. RETRAIN all 3 models and evaluate on test. Save new metrics; compare with artifacts/legacy/.
4. EVALUATION: sensitivity, specificity, PPV, NPV with 95% bootstrap CIs; ROC + calibration curve + Brier score;
   choose the threshold on VAL for sensitivity >= 95%, then report on TEST.
   Shortcut check: metrics on the 96x96 TB subgroup vs the rest; a baseline that predicts from image size only.
5. GRAD-CAM: fix the NaN guard, one layer name per model; gallery of all test false negatives/positives;
   docs/failure_analysis.md with observations.
6. APP + REQUIREMENTS: .convert("RGB"); use the tuned threshold; "not for clinical use" disclaimer;
   pin requirements.txt from the lock file; fix README (closed code block, real clone URL, honest results for all 3 models).
7. ENGINEERING: __init__.py files, pytest tests (label mapping, preprocessing parity, split has no duplicates
   across splits, prediction range), GitHub Actions CI; remove or move the PyTorch files to legacy/.
8. MLOPS + SERVING: MLflow tracking; DVC for data; FastAPI /predict; working Dockerfile (libgl1, .dockerignore,
   serving command); MODEL_CARD.md.