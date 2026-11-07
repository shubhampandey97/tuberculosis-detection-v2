# Failure analysis: where the TB models look (Grad-CAM)

Goal 5 of the upgrade plan. Models: **ResNet50 (primary, chosen on val AUC)** and VGG16. Data: the held-out **test split** (448 images: 77 Normal, 371 TB). Decisions use the val-tuned thresholds from goal 4 (ResNet 0.6142, VGG 0.2357).

**Short answer.** For TB images the models classify correctly, ResNet50's evidence lies mostly *inside* the central area, but both models also put weight on the image periphery. In the galleries this includes **handwritten text and labels at the edges of TB images**. VGG16 is much more peripheral than ResNet50. TB that the models miss gets its little TB evidence from the periphery. The 7 TB images missed by every model share a common acquisition profile (n = 7, so suggestive only): compared with other TB images, they look like the Normal source. This fits the source confounding found in the data audit (audit §5.3 item 7) and the goal 4 masking test (AUC 0.91–0.97 with the lungs hidden). It points to source cues, not lung findings, as a major driver of the predictions.

---

## 1. Method

- **Code:** [src/components/gradcam.py](../src/components/gradcam.py) (Grad-CAM) and [src/pipeline/gradcam_analysis.py](../src/pipeline/gradcam_analysis.py) (analysis). Outputs are in [artifacts/gradcam/](../artifacts/gradcam/).
- **Layers:** one documented layer per model (`model_builder.LAST_CONV_LAYER`):
  - ResNet50 `conv5_block3_out` (7×7 grid, 32 px per cell)
  - VGG16 `block5_conv3` (14×14, 16 px per cell)
  - Resolution check: ResNet50 is also measured at `conv4_block6_out` (14×14).
- **Target:** the pre-sigmoid **logit**, not P(TB), because sigmoid gradients vanish for confident predictions. Two maps come from one backward pass:
  - **TB-evidence map** (primary): ReLU(+CAM). It's the same target for every image.
  - **Predicted-class map** (secondary, used in the galleries): ReLU(+CAM) if the prediction is TB, ReLU(−CAM) if it's Normal.
- **Attention measure:**
  1. Upscale each map to 224×224 and normalise it to sum to 1.
  2. Measure the share of energy in the **outer 12% border** (27 px), split into top, bottom, left and right strips. Corners count as part of the top and bottom strips.
  3. Compare with the **uniform reference**, i.e. each region's share of the image area: border 0.424, top 0.121, bottom 0.121, left 0.091, right 0.091.
  4. Group tests use two-sided Mann–Whitney U. The p-values are **not** corrected for multiple comparisons (about 18 tests), so treat p ≈ 0.05 as weak.
- **Empty maps:** maps with no positive evidence are guarded (zeros, flagged) and left out of the fractions.

### Sanity checks (32 test images, [sanity.json](../artifacts/gradcam/sanity.json))
| check | ResNet50 | VGG16 |
|---|---|---|
| P(TB) vs saved predictions (max abs diff) | 3.2e-7 | 5.3e-9 |
| sigmoid(logit) vs model output | 6.0e-8 | 3.7e-9 |
| NaN / empty TB maps | none / 0 | none / 0 |
| **Randomised head**: Pearson r between original and random-head maps (median, range) | **0.49** (−0.14 to 0.77; 5 random maps empty) | **−0.10** (−0.24 to 0.44) |

The randomised-head check replaces the final Dense weights with random ones of the same scale.
- **VGG16 passes clearly:** its maps depend on the learned weights.
- **ResNet50 only partly passes.** Its maps stay moderately correlated with random-weight maps, so part of each ResNet map just shows *where conv5 features are strongly active*, not what the classifier uses. ResNet maps should be read as weaker evidence than VGG maps.

---

## 2. Where the models look (quantitative)

Border energy share, median [IQR], with the uniform reference 0.424. Full table: [attention_summary.json](../artifacts/gradcam/attention_summary.json); per-image values: `attention_resnet.csv` and `attention_vgg.csv`.

**TB-evidence map (primary):**

| model / layer | Normal (77) | TB (371) | p (Normal vs TB) | TB correct | TB missed (FN) | p (correct vs missed) | empty maps |
|---|---|---|---|---|---|---|---|
| ResNet50 conv5 (7×7) | 0.437 [0.264–0.676] | 0.332 [0.266–0.397] | 3.9e-5 | 0.331 (n=347) | 0.388 [0.281–0.514] (n=24) | 0.047 | 1 (Normal) |
| ResNet50 conv4 (14×14) | 0.148 [0.010–0.462] | 0.316 [0.196–0.449] | 1.1e-4 | 0.308 | 0.526 [0.227–0.690] (n=21) | 0.010 | 13 (10 Normal, 3 FN) |
| VGG16 block5 (14×14) | 0.523 [0.455–0.607] | 0.449 [0.367–0.548] | 1.0e-4 | 0.429 (n=345) | 0.624 [0.556–0.723] (n=26) | 8.7e-9 | 0 |

**Mean share per strip in the TB-evidence map** (uniform: top/bottom 0.121, left/right 0.091):

| model | group | top | bottom | left | right |
|---|---|---|---|---|---|
| ResNet50 conv5 | TB correct | 0.065 | **0.170** | 0.045 | 0.050 |
| ResNet50 conv5 | TB missed | **0.187** | 0.105 | 0.056 | 0.049 |
| ResNet50 conv5 | Normal | **0.368** | 0.032 | 0.036 | 0.034 |
| VGG16 | TB correct | 0.110 | **0.255** | 0.045 | 0.037 |
| VGG16 | TB missed | **0.214** | **0.350** | 0.032 | 0.031 |
| VGG16 | Normal | 0.164 | **0.311** | 0.033 | 0.014 |

**Predicted-class map (secondary):**
- ResNet50: the Normal-evidence maps of Normal images put 0.492 [0.460–0.538] of their energy on the border, above uniform. The same holds for missed TB (0.495 [0.439–0.547], vs 0.331 for correct TB; p = 4.0e-10). Empty maps: 0.
- VGG16 shows no Normal vs TB difference in these maps: 0.432 vs 0.431, p = 0.74. Empty maps: 0.

**What the numbers say:**
1. **ResNet50 on correctly detected TB is mostly central.** Only 0.331 of the energy falls on the border, below uniform. But the **bottom strip gets 1.4× its area share**, where the handwritten annotations sit (see §3).
2. **VGG16 is peripheral.** Nearly half its TB evidence (median 0.463 over all images) is on the border, and the **bottom strip gets 2.1–2.9× its area share in every group**. This matches goal 4, where border masking hurt VGG most (AUC 0.991 → 0.799 at 20%).
3. **Missed TB has more peripheral evidence, in both models.** The clearest rise relative to correct TB is in the **top strip**, where the side markers sit: ResNet 0.187 vs 0.065, VGG 0.214 vs 0.110, i.e. 1.5–1.8× uniform. VGG's bottom strip rises too (0.350 vs 0.255). This holds at both ResNet layers (conv5 p = 0.047; conv4 p = 0.010) and for VGG (p = 8.7e-9).
4. **Resolution check:** the "missed TB is more peripheral" result is robust to layer choice. The **Normal vs TB difference for ResNet isn't**: Normal is *more* peripheral at conv5 (0.437 vs 0.332) but *less* at conv4 (0.148 vs 0.316). At conv4, 10 of 77 Normal TB-evidence maps are empty, which leaves little to measure. Claims about where ResNet's TB evidence sits *in Normal images* should not be made.
5. **Attention on the bottom strip alone is not evidence of a shortcut.** That strip contains both annotations and lower-lung anatomy (diaphragm, costophrenic angles); the hot spots that land exactly on handwriting (§3) are the evidence.
6. Normalised maps show **where** the evidence is, not **how much** there is. On Normal images, the TB-evidence map shows where the small amount of TB evidence sits, not a strong signal.

---

## 3. Galleries (qualitative observations)

Galleries use the predicted-class maps at the primary layers. The dashed box marks the inner edge of the 12% border.
- [gallery_errors.png](../artifacts/gradcam/gallery_errors.png)
- [gallery_correct_tb.png](../artifacts/gradcam/gallery_correct_tb.png)
- [gallery_correct_normal.png](../artifacts/gradcam/gallery_correct_normal.png)

The random images are 10 Normal and 10 TB that both models got right (seed 42).

> **Everything in this section is qualitative:** my visual reading of the overlays, by a non-radiologist. It has not been reviewed by a clinician.

**Correct TB (10 random):**
- **Handwritten text.** Five of the 10 have handwritten annotations (names and dates in non-Latin script) at the bottom: TB.1860, TB.1904, TB.2071, TB.2186, TB.444.
  - In TB.1860 both models have a hot spot right on the text.
  - ResNet also lands on the text in TB.1904 and TB.2071 (TB.444 less clearly).
  - VGG shows a band along the text in TB.2071 and TB.2186.
- **Labels and markers.** In TB.1967, ResNet's map covers the printed "CXR" label at the top-left. TB.1406's maps sit in the bottom corners.
- **Lungs and heart.** Other images have attention on the lung fields or the mediastinum and heart region: ResNet TB.2186, TB.2369, TB.538, TB.830; VGG TB.2369.

**Correct Normal (10 random):**
- **No consistent hot spot on the small "L" marker** in the top-right corner that all Normal images share.
- **ResNet's** Normal evidence mostly sits along the **bottom band** (below the diaphragm and the lower edge) and along the lateral edges.
- **VGG's** usually sits at the **top centre** (trachea, neck and upper mediastinum) and along the shoulders.
- The vertical streak artefacts typical of the Normal source can't be confirmed or ruled out as a cue from these maps.

**Shared errors (7 TB missed by all 3 models, plus the one false positive):**
- **The Normal evidence for the misses is mostly outside the lung fields.**
  - ResNet: the bottom band in TB.1141, TB.1892, TB.2324 and TB.321; the lower corners and edges in TB.441 and TB.244.
  - VGG: the lateral edges in TB.1141 and TB.2145, the top corners in TB.1892, TB.441 and TB.244, and the top centre in TB.321 and TB.244.
- **No hot spot sits on the "R" markers** visible in TB.1141, TB.1892 and TB.2324, or on the "F" in TB.244.
- **None of the 7 misses has handwritten text at the bottom,** unlike half of the correct TB sample.
- **Several misses look like fairly clear lungs to a non-expert** (TB.1141, TB.2145, TB.2324, TB.321). The disease may be subtle, which a radiologist would need to confirm.
- **TB.1892 is a clear exception:** it shows visible patchy and nodular opacities, especially in the left lung, and is still called Normal by every model (P(TB) 0.361 ResNet, 0.124 VGG). This is the strongest single sign that the decision is not driven by lung findings alone.
- **TB.441** is washed out, with very low contrast.
- **The false positive, `others (23)` (Normal):**
  - VGG calls it TB (P = 0.912), with evidence over the lower-right heart and diaphragm region.
  - ResNet gives P = 0.513. That's a false positive at 0.5 but correct at the tuned threshold of 0.614. Its Normal-evidence map sits on both lateral edges (border share 0.71).

---

## 4. What the 7 shared misses have in common (quantitative)

From [image_stats.csv](../artifacts/gradcam/image_stats.csv) and [shared_misses_vs_groups.csv](../artifacts/gradcam/shared_misses_vs_groups.csv). Medians:

| group | n | file size (KB) | brightness | contrast (SD) | P(TB) ResNet | P(TB) VGG | TB-evidence border share, ResNet / VGG |
|---|---|---|---|---|---|---|---|
| shared misses | 7 | **10.28** | 134.2 | **45.4** | 0.249 | 0.057 | 0.332 / **0.671** |
| TB correct (both models) | 334 | 7.88 | 137.4 | 54.0 | 0.986 | 0.992 | 0.331 / 0.426 |
| Normal correct (both models) | 76 | 9.84 | 132.0 | 51.4 | 0.027 | 0.003 | 0.443 / 0.525 |

| image | size | KB (percentile among test TB) | contrast (percentile among test TB) | P(TB) ResNet / VGG |
|---|---|---|---|---|
| TB.1141 | 256×256 | 10.28 (85th) | 28th | 0.075 / 0.057 |
| TB.1892 | 256×256 | 12.10 (91st) | 10th | 0.361 / 0.124 |
| TB.2145 | 256×256 | 10.43 (86th) | 16th | 0.149 / 0.030 |
| TB.2324 | 256×256 | 9.80 (81st) | 71st | 0.371 / 0.066 |
| TB.244 | 256×256 | 13.63 (94th) | 17th | 0.453 / 0.052 |
| TB.321 | 256×256 | 9.68 (79th) | 21st | 0.174 / 0.032 |
| TB.441 | 256×256 | 6.99 (19th) | 0.3th | 0.249 / 0.077 |
| *others (23)* (Normal, the false positive) | 256×256 | 8.91 (71st among TB; 12th among Normal) | 30th | 0.513 / 0.912 |

**What the misses share:**
- **All 7 are standard 256×256**; none is a non-square image.
- **6 of 7 have large files** (79th–94th percentile of TB). The median of 10.28 KB is above the *Normal* median of 9.84, and TB is usually smaller (median 8.01 KB). The data audit found file size alone predicts the label with AUC 0.861.
- **6 of 7 have lower contrast than most TB images** (≤ 28th percentile).
- **None has the bottom handwriting** seen in many correct TB images (qualitative, §3).
- **VGG's remaining TB evidence for them is mostly on the border** (0.671).
- **The one false positive is the mirror image:** a Normal image with a TB-like small file size (12th percentile among Normals).

**Interpretation (hypothesis, not proven):** the shared misses are TB images that *look like the Normal source* on acquisition and compression. The shared false positive is a Normal image that looks like the TB source. Together with the goal 4 masking result (AUC 0.912–0.974 with the lungs hidden) and the attention on annotations, the evidence suggests the models partly learned **which source an image comes from**. Lung pathology is not the only thing they use. With n = 7 and 1, this can't be confirmed statistically.

---

## 5. Limitations

- **Grad-CAM is approximate and coarse.** A ResNet conv5 cell is 32 px, larger than the 27-px border, and the ResNet maps only partly pass the randomised-head check.
- **Normalised shares** show where the evidence is, not how strong it is.
- **The visual reading is a non-expert's,** and nobody has reviewed the misses clinically.
- **Internal test set only,** from the same two sources as training.
- **The p-values are not corrected** for multiple comparisons.
- **"Lungs hidden" in goal 4 is approximate** (lung tops and outer lung edges stay visible).

## 6. Next steps (proposed, not done)

1. **Remove the shortcut in training:** crop or mask the borders and annotations (or train on lung-field crops or segmentations), then retrain and repeat the goal 4 masking test. Success would mean a large *drop* in lungs-hidden AUC while the unmasked score stays close.
2. **External validation** on a single-source dataset (Shenzhen or Montgomery) where both classes share one acquisition pipeline.
3. **Radiologist review** of the 7 shared misses, especially TB.1892.
4. **Recalibrate** the probabilities (goal 4 found P(TB) under-predicted) before showing any "confidence" in the app (goal 6).
