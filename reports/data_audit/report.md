# Data audit: `data/Dataset of Tuberculosis Chest X-rays Images`

## 1. Inventory

```
                     files  mean_kb  share_%
class                                       
Normal Chest X-rays    514      9.9     17.1
TB Chest X-rays       2494      9.5     82.9
```

Label mapping (alphabetical, same as Keras): `Normal Chest X-rays` = 0, `TB Chest X-rays` = 1

Extensions: {'.jpg': 3008}
Non-image files skipped: 0

## 2. Integrity

Unreadable / corrupt files: **0**

Colour modes per class:

```
mode                  RGB
class                    
Normal Chest X-rays   514
TB Chest X-rays      2494
```

## 3. Image size and metadata

Top sizes per class (width x height):

```
                              count
class               size_wxh       
Normal Chest X-rays 256x256     514
TB Chest X-rays     256x256    2370
                    256x215       8
                    256x217       7
                    256x213       6
                    256x203       5
```

Number of distinct sizes per class: {'Normal Chest X-rays': 1, 'TB Chest X-rays': 60}

Aspect ratio (width/height) per class:

```
                     min  50%       max
class                                  
Normal Chest X-rays  1.0  1.0  1.000000
TB Chest X-rays      1.0  1.0  1.383784
```

DPI values per class:

```
class     Normal Chest X-rays  TB Chest X-rays
dpi                                           
                          514             2370
(96, 96)                    0              124
```

## 4. Exact duplicates (MD5)

Files in duplicate groups: **52** (26 groups). Groups spanning more than one class (label conflicts): **0**.

Duplicates must be removed before splitting, or one copy can land in train and the other in test.

## 5. Near-duplicates (perceptual hash)

Pairs with thumbnail correlation >= 0.99 (not byte-identical): **2**. Cross-class pairs: **0**.

These are candidates only: open a few pairs and look before removing anything.

## 6. Pixel statistics (64x64 grayscale)

```
                        mean_px     std_px  dark_frac  bright_frac
class                                                             
Normal Chest X-rays  132.842239  48.297859   0.003839     0.000068
TB Chest X-rays      135.744614  53.796234   0.010041     0.005890
```

## 7. Shortcut test (metadata only, 5-fold CV ROC-AUC)

```
               features                   model  roc_auc_mean  roc_auc_std
width + height + aspect     logistic regression      0.524860     0.001611
width + height + aspect decision tree (depth 3)      0.524860     0.001611
         file size (KB)     logistic regression      0.784300     0.018287
         file size (KB) decision tree (depth 3)      0.860695     0.014796
    brightness/contrast     logistic regression      0.744256     0.006767
    brightness/contrast decision tree (depth 3)      0.802756     0.013175
           all metadata     logistic regression      0.910665     0.018742
           all metadata decision tree (depth 3)      0.918315     0.014119
```

How to read this: ~0.5 means metadata says nothing about the label. Values near 1.0 mean the classes can be told apart **without looking at the anatomy**, so a CNN might learn the same shortcut. Investigate before trusting any high accuracy.
