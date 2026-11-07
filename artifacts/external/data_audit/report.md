# Data audit of the external sets

Same functions as `src/pipeline/data_audit.py` (unchanged): `inspect_file` (64x64 thumbnail pixel statistics, 32x32 z-normalised similarity vector, MD5), `exact_duplicates`, `near_duplicates` (thumbnail correlation >= 0.99) and `shortcut_test` (5-fold stratified CV ROC-AUC, seed 42). Labels come from the `_0`/`_1` file suffix. File size = original PNG.

## Shenzhen (662 images, unreadable: 0)

- Exact duplicate groups (MD5): **0** (cross-class: 0).
- Near-duplicate pairs (thumbnail corr >= 0.99): **0**.

## Montgomery (138 images, unreadable: 0)

- Exact duplicate groups (MD5): **0** (cross-class: 0).
- Near-duplicate pairs (thumbnail corr >= 0.99): **0**.

## Metadata-only shortcut test (5-fold CV ROC-AUC), next to the training data audit

```
                         training LR  training tree  shenzhen LR  shenzhen tree  montgomery LR  montgomery tree
features                                                                                                       
width + height + aspect       0.5249         0.5249     0.710798       0.741852       0.578220         0.578220
file size (KB)                0.7843         0.8607     0.531761       0.655291       0.557670         0.533333
brightness/contrast           0.7443         0.8028     0.704759       0.721089       0.601799         0.584612
all metadata                  0.9107         0.9183     0.854714       0.791319       0.637405         0.563542
```

Flag rule: all-metadata AUC >= 0.8 (either model) = the site's labels can be guessed from acquisition metadata, which limits it as an external test.

- shenzhen: best all-metadata AUC 0.855 -> **FLAGGED**
- montgomery: best all-metadata AUC 0.637 -> not flagged

## Pixel statistics per class (medians; 64x64 grayscale thumbnails of the original image)

```
                                          n     mean_px     std_px  dark_frac  bright_frac      file_kb
site              class                                                                                
training (Kaggle) Normal Chest X-rays   514  132.051025  48.332186   0.001953     0.000000     9.868652
                  TB Chest X-rays      2494  135.664550  54.286048   0.002686     0.000000     8.093262
shenzhen          Normal                326  161.057251  61.943451   0.012695     0.000000  5805.930664
                  TB                    336  150.817139  63.478371   0.008545     0.000000  5686.989258
montgomery        Normal                 80  115.538452  80.606415   0.181274     0.051514  4211.644043
                  TB                     58  114.690552  81.674240   0.168457     0.044800  4369.824707
```

