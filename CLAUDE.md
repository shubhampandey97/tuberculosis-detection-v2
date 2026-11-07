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
- Git: work on branch upgrade-v4. main has goals 1–7 and 9 merged (PRs #10–#12); goal 10 is on upgrade-v4.
- Local pytest: set PYTHONNOUSERSITE=1 (a user-site platformdirs breaks pytest). Lint: ruff check .

# Current state (goals 1–7, 9 and 10 DONE; do not redo)
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
- Tests/CI (goal 7): pytest suite in tests/ (data/model tests auto-skip), GitHub Actions ruff + pytest.
- EXTERNAL VALIDATION (goal 9; src/pipeline/external_validation.py, docs/audit.md goal 9, artifacts/external/):
  frozen models, val-tuned thresholds unchanged, 0 overlaps with training (incl. crop search).
  ResNet50 AUC 0.669 (Shenzhen, 662) / 0.656 (Montgomery, 138) vs 0.998 internal; specificity at the
  tuned threshold 0.208 / 0.000 (flags every Montgomery image TB). VGG16 0.470 / 0.581, EffNet 0.521 / 0.537
  (chance). Pre-registered H1, H2, H6 supported; H3 not testable; H4 not supported; H5 prediction supported,
  stated reason only partly. Montgomery Grad-CAM energy inside true lung masks 0.186 (ResNet) / 0.169 (VGG)
  < lung-area share 0.257 (resolution ceiling 0.682 / 0.825). Shenzhen metadata AUC 0.855 (flagged).
  NLM terms: don't redistribute the NLM data; external data lives in data/external/ (not committed).
- GOAL 10 SHORTCUT REMOVAL: DONE, NEGATIVE (docs/audit.md goal 10; pre-registered; configs
  src/config/config_v10_arm{1,2}.yaml selected via TB_CONFIG; models/tensorflow_v10_arm{1,2}/; outputs
  artifacts/v10_arm{1,2}/, criteria in artifacts/v10_comparison/ via src/pipeline/goal10_compare.py).
  In-model 12% central crop + per-image standardisation (mean 128, SD 50) + label-independent marker/JPEG/
  intensity augmentation; ResNet50 + EffNetB0; Arm 1 frozen, Arm 2 fine-tunes last block. Primary ResNet50
  Arm 2: external AUC 0.502 / 0.572 (FELL; S1 not supported), S4a processed-input shortcut AUC 0.883,
  S4b not supported. S2 (0.659) and S3 pass by the rules only (sigmoid saturation; crop geometry,
  uniform-in-crop lung share 0.413). EffNet S1 supported but only to 0.60-0.68. Internal AUC 0.999.
  Lessons: val is confounded (early stopping/thresholds); rank on logits, not saturated probabilities.

# Working rules
- Show a plan first and wait for approval before editing.
- One goal per session. Keep changes small and reviewable.
- Before any full training run, do a 1-epoch smoke test on ~200 images, and tell me the expected CPU time.
- After each goal: list the changed files, summarise results, update the Status section of docs/audit.md.
- Never delete data. Never commit data/ (except splits.csv, duplicates.csv) or models/.
- Any new inference path must pass a parity test against artifacts/predictions/ (two past bugs were
  train/inference mismatches).

# Remaining goals (order: 8, then optional 10b)
8. MLOPS + SERVING: FastAPI /predict reusing inference.py; working Dockerfile (CPU, .dockerignore,
   serving command, no CUDA base); MODEL_CARD.md including external-validation results;
   optional MLflow tracking and DVC for data.
10b (optional). MIXED-SOURCE TRAINING: Kaggle + Shenzhen so each source has both classes; source-stratified
   val/test; thresholds on logits; Montgomery as external test. Needs its own pre-registration first.
