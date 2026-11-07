# Context
Portfolio project for ML/Data Science job applications (target: medtech/healthcare AI roles).
Owner: Shubham Pandey. Keep all results honest; never invent or round up metrics.
docs/audit.md = full code audit (§1–§6) + Status of every goal. docs/failure_analysis.md = Grad-CAM analysis.
Refer to them by section number.

# Environment
- Windows, conda env tb_v2_env, CPU only. TF 2.16.1, Keras 3.12.4, protobuf<5, Streamlit 1.60.
- Always run Python with: C:\Users\Admin\anaconda3\envs\tb_v2_env\python.exe
- Run modules from the repo root, e.g. ...\python.exe -m src.pipeline.train_pipeline
- requirements.txt = 12 direct deps pinned (Linux-safe). requirements-lock.txt = exact Windows env.
  Never upgrade protobuf above 4.x. OpenCV and tqdm were removed (nothing uses them).
- Git: work on branch upgrade-v4. main has goals 1–6 merged (PR #10).

# Current state (goals 1–6 DONE; do not redo)
- Data: 2,982 images after removing 26 exact duplicates (509 Normal / 2,473 TB).
  data/splits.csv = stratified seeded 70/15/15 (train 2,087 / val 447 / test 448; test = 77 Normal, 371 TB).
  Labels 0 = Normal, 1 = TB. 124 TB images are non-square. No patient IDs.
- Pipeline: one decode path (data_loader.decode_and_resize, TF decode, raw 0–255 RGB) used by training,
  evaluation and the app (src/components/inference.py). Per-model preprocessing lives INSIDE the model.
  Models: models/tensorflow/{vgg,resnet,efficientnet}_model.keras (frozen ImageNet backbones).
- Primary model: ResNet50 (selected on val AUC). Val-tuned thresholds (sens >= 0.95 on val) in
  artifacts/clinical/thresholds.json: ResNet 0.6142, VGG 0.2357, EffNet 0.3257.
- Test results (ResNet @ tuned): sensitivity 0.935 (0.908–0.959), specificity 1.000 (77/77), AUC 0.998.
- KEY FINDING: dataset source confounding. Metadata-only AUC 0.918; lungs-hidden AUC 0.971 (ResNet),
  0.974 (VGG), 0.912 (EffNet); Grad-CAM attention on handwritten annotations; 7 TB images missed by all
  models look like the Normal source. Internal scores are an upper bound, not clinical performance.
- Per-image predictions: artifacts/predictions/{model}_{val,test}.csv. Old invalid results: artifacts/legacy/.

# Working rules
- Show a plan first and wait for approval before editing.
- One goal per session. Keep changes small and reviewable.
- Before any full training run, do a 1-epoch smoke test on ~200 images, and tell me the expected CPU time.
- After each goal: list the changed files, summarise results, update the Status section of docs/audit.md.
- Never delete data. Never commit data/ (except splits.csv, duplicates.csv) or models/.
- Any new inference path must pass a parity test against artifacts/predictions/ (two past bugs were
  train/inference mismatches).

# Remaining goals (order: 7 → 9 → 8 → 10)
7. ENGINEERING: __init__.py files, pytest (label mapping, preprocessing parity app vs training,
   CaffePreprocess vs keras preprocess_input, image modes, prediction range, reload, Grad-CAM NaN guard,
   make_splits logic, metric functions); data/model-dependent tests auto-skip when files are missing;
   GitHub Actions CI (Ubuntu, Python 3.10, ruff + pytest); .gitattributes for line endings; CI badge.
9. EXTERNAL VALIDATION: evaluate the frozen ResNet50/VGG16/EffNet (no retraining) on the NLM Shenzhen set
   (662 CXRs, 336 TB / 326 Normal, one hospital) and Montgomery set (138 CXRs, 58 TB / 80 Normal,
   one hospital), where both classes share one acquisition pipeline. FIRST check for overlap with the
   training data (some public TB collections include these images). Same metrics/CIs as goal 4,
   val-tuned thresholds applied unchanged, plus subgroups by sex/age from the clinical readings.
   Data in data/external/ (not committed).
8. MLOPS + SERVING: FastAPI /predict reusing inference.py; working Dockerfile (CPU, .dockerignore,
   serving command, no CUDA base); MODEL_CARD.md including external-validation results;
   optional MLflow tracking and DVC for data.
10. (optional) SHORTCUT REMOVAL: retrain with border masking / lung crops; re-run the lungs-hidden test
   and external validation.