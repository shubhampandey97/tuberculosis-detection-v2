# External validation results (frozen models, val-tuned thresholds unchanged)

Thresholds: ResNet50 0.6142, VGG16 0.2357, EfficientNetB0 0.3257. 95% CIs: stratified bootstrap, 2000 resamples, seed 42. Values truncated to 3 decimals. Primary variant: lossless PNG. Overlap exclusions: confirmed 0, automatic (>= 0.95) 0 -> results with and without exclusions are identical.

## Main table (PNG, at the val-tuned threshold)

| model | data | n (N/TB) | ROC-AUC | sensitivity | specificity | PPV | NPV | F1 | Brier |
|---|---|---|---|---|---|---|---|---|---|
| ResNet50 | internal test | 448 (77/371) | 0.998 (0.996–0.999) | 0.935 (0.908–0.959) | 1.000 (1.000–1.000) | 1.000 (1.000–1.000) | 0.762 (0.693–0.836) | 0.966 (0.951–0.979) | 0.026 (0.018–0.035) |
| ResNet50 | shenzhen | 662 (326/336) | 0.669 (0.628–0.709) | 0.907 (0.875–0.937) | 0.208 (0.165–0.254) | 0.541 (0.526–0.558) | 0.686 (0.600–0.771) | 0.678 (0.659–0.696) | 0.320 (0.306–0.334) |
| ResNet50 | montgomery | 138 (80/58) | 0.656 (0.559–0.748) | 1.000 (1.000–1.000) | 0.000 (0.000–0.000) | 0.420 (0.420–0.420) | n/a (n/a–n/a) [2000 undef.] | 0.591 (0.591–0.591) | 0.554 (0.549–0.560) |
| ResNet50 | pooled (secondary) | 800 (406/394) | 0.619 (0.581–0.657) | 0.921 (0.893–0.946) | 0.167 (0.133–0.204) | 0.517 (0.505–0.531) | 0.686 (0.599–0.772) | 0.663 (0.646–0.678) | 0.361 (0.347–0.374) |
| VGG16 | internal test | 448 (77/371) | 0.991 (0.982–0.998) | 0.929 (0.902–0.954) | 0.987 (0.961–1.000) | 0.997 (0.991–1.000) | 0.745 (0.675–0.819) | 0.962 (0.947–0.976) | 0.072 (0.054–0.091) |
| VGG16 | shenzhen | 662 (326/336) | 0.470 (0.427–0.514) | 0.699 (0.648–0.747) | 0.165 (0.125–0.205) | 0.463 (0.441–0.483) | 0.348 (0.282–0.412) | 0.557 (0.526–0.585) | 0.383 (0.358–0.409) |
| VGG16 | montgomery | 138 (80/58) | 0.581 (0.479–0.683) | 0.965 (0.913–1.000) | 0.000 (0.000–0.000) | 0.411 (0.398–0.420) | 0.000 (0.000–0.000) [251 undef.] | 0.577 (0.554–0.591) | 0.516 (0.481–0.550) |
| VGG16 | pooled (secondary) | 800 (406/394) | 0.465 (0.425–0.505) | 0.738 (0.695–0.781) | 0.133 (0.103–0.165) | 0.452 (0.434–0.469) | 0.343 (0.278–0.409) | 0.561 (0.535–0.585) | 0.406 (0.382–0.430) |
| EfficientNetB0 | internal test | 448 (77/371) | 0.996 (0.991–0.999) | 0.946 (0.921–0.967) | 0.987 (0.961–1.000) | 0.997 (0.991–1.000) | 0.791 (0.723–0.863) | 0.970 (0.957–0.982) | 0.071 (0.058–0.086) |
| EfficientNetB0 | shenzhen | 662 (326/336) | 0.521 (0.477–0.565) | 0.476 (0.422–0.529) | 0.533 (0.478–0.585) | 0.512 (0.470–0.555) | 0.497 (0.459–0.534) | 0.493 (0.447–0.535) | 0.289 (0.277–0.301) |
| EfficientNetB0 | montgomery | 138 (80/58) | 0.537 (0.442–0.639) | 1.000 (1.000–1.000) | 0.012 (0.000–0.037) | 0.423 (0.420–0.429) | 1.000 (1.000–1.000) [688 undef.] | 0.594 (0.591–0.601) | 0.384 (0.356–0.411) |
| EfficientNetB0 | pooled (secondary) | 800 (406/394) | 0.498 (0.458–0.537) | 0.553 (0.505–0.604) | 0.431 (0.384–0.480) | 0.485 (0.456–0.515) | 0.498 (0.460–0.537) | 0.517 (0.481–0.552) | 0.305 (0.290–0.321) |

## At threshold 0.5 (for comparison)

| model | data | sensitivity | specificity | mean P(TB) Normal / TB |
|---|---|---|---|---|
| ResNet50 | shenzhen | 0.967 (0.946–0.985) | 0.134 (0.101–0.171) | 0.751 / 0.847 |
| ResNet50 | montgomery | 1.000 (1.000–1.000) | 0.000 (0.000–0.000) | 0.977 / 0.982 |
| VGG16 | shenzhen | 0.544 (0.491–0.595) | 0.380 (0.328–0.432) | 0.584 / 0.539 |
| VGG16 | montgomery | 0.913 (0.827–0.982) | 0.062 (0.012–0.112) | 0.907 / 0.891 |
| EfficientNetB0 | shenzhen | 0.172 (0.133–0.214) | 0.880 (0.843–0.914) | 0.327 / 0.351 |
| EfficientNetB0 | montgomery | 1.000 (1.000–1.000) | 0.062 (0.012–0.112) | 0.770 / 0.788 |

## PNG vs JPEG (quality 75, 4:2:0) ROC-AUC

| model | site | AUC PNG | AUC JPEG | JPEG - PNG (95% CI, paired) |
|---|---|---|---|---|
| ResNet50 | shenzhen | 0.669 | 0.671 | +0.0014 (-0.0106 to +0.0132) |
| ResNet50 | montgomery | 0.656 | 0.651 | -0.0043 (-0.0289 to +0.0190) |
| ResNet50 | pooled | 0.619 | 0.621 | +0.0023 (-0.0071 to +0.0109) |
| VGG16 | shenzhen | 0.470 | 0.477 | +0.0075 (-0.0002 to +0.0156) |
| VGG16 | montgomery | 0.581 | 0.576 | -0.0056 (-0.0203 to +0.0095) |
| VGG16 | pooled | 0.465 | 0.469 | +0.0033 (-0.0027 to +0.0098) |
| EfficientNetB0 | shenzhen | 0.521 | 0.531 | +0.0094 (-0.0026 to +0.0209) |
| EfficientNetB0 | montgomery | 0.537 | 0.549 | +0.0119 (-0.0097 to +0.0317) |
| EfficientNetB0 | pooled | 0.498 | 0.507 | +0.0092 (+0.0001 to +0.0176) |

## Oracle analysis (NOT a result: threshold chosen on the external data itself)

| model | site | val-tuned threshold | sens / spec at val-tuned | oracle threshold for 95% sens | spec at oracle |
|---|---|---|---|---|---|
| ResNet50 | shenzhen | 0.6142 | 0.907 / 0.208 | 0.5171 | 0.141 |
| ResNet50 | montgomery | 0.6142 | 1.000 / 0.000 | 0.9113 | 0.025 |
| ResNet50 | pooled | 0.6142 | 0.921 / 0.167 | 0.5315 | 0.123 |
| VGG16 | shenzhen | 0.2357 | 0.699 / 0.165 | 0.0231 | 0.006 |
| VGG16 | montgomery | 0.2357 | 0.965 / 0.000 | 0.2479 | 0.000 |
| VGG16 | pooled | 0.2357 | 0.738 / 0.133 | 0.0320 | 0.017 |
| EfficientNetB0 | shenzhen | 0.3257 | 0.476 / 0.533 | 0.1305 | 0.042 |
| EfficientNetB0 | montgomery | 0.3257 | 1.000 / 0.012 | 0.5267 | 0.075 |
| EfficientNetB0 | pooled | 0.3257 | 0.553 / 0.431 | 0.1356 | 0.041 |

## Subgroups (ResNet50, PNG, val-tuned threshold; * = fewer than 30 images in a class)

| site | variable | group | n (N/TB) | sensitivity | specificity | AUC |
|---|---|---|---|---|---|---|
| shenzhen | sex | F | 213 (105/108) | 0.870 (0.805–0.925) | 0.142 (0.076–0.209) | 0.609 (0.532–0.680) |
| shenzhen | sex | M | 448 (220/228) | 0.925 (0.890–0.956) | 0.240 (0.186–0.295) | 0.698 (0.651–0.744) |
| shenzhen | sex | unknown* | 1 (1/0) | n/a | 0.000 (0.000–0.000) | n/a |
| shenzhen | age band | 18-39 | 431 (218/213) | 0.887 (0.845–0.929) | 0.243 (0.192–0.298) | 0.670 (0.616–0.717) |
| shenzhen | age band | 40-59 | 147 (66/81) | 0.925 (0.864–0.975) | 0.212 (0.121–0.318) | 0.664 (0.575–0.745) |
| shenzhen | age band | <18* | 31 (24/7) | 0.857 (0.571–1.000) | 0.000 (0.000–0.000) | n/a |
| shenzhen | age band | >=60* | 52 (17/35) | 1.000 (1.000–1.000) | 0.058 (0.000–0.176) | 0.715 (0.569–0.848) |
| shenzhen | age band | unknown* | 1 (1/0) | n/a | 0.000 (0.000–0.000) | n/a |
| montgomery | sex | F* | 74 (53/21) | 1.000 (1.000–1.000) | 0.000 (0.000–0.000) | 0.673 (0.528–0.811) |
| montgomery | sex | M* | 63 (26/37) | 1.000 (1.000–1.000) | 0.000 (0.000–0.000) | 0.602 (0.455–0.743) |
| montgomery | sex | unknown* | 1 (1/0) | n/a | 0.000 (0.000–0.000) | n/a |
| montgomery | age band | 18-39* | 51 (34/17) | 1.000 (1.000–1.000) | 0.000 (0.000–0.000) | 0.577 (0.397–0.756) |
| montgomery | age band | 40-59* | 52 (31/21) | 1.000 (1.000–1.000) | 0.000 (0.000–0.000) | 0.677 (0.517–0.821) |
| montgomery | age band | <18* | 17 (13/4) | 1.000 (1.000–1.000) | 0.000 (0.000–0.000) | n/a |
| montgomery | age band | >=60* | 18 (2/16) | 1.000 (1.000–1.000) | 0.000 (0.000–0.000) | n/a |

## Pre-registered hypotheses

```
{
 "H1": {
  "resnet/shenzhen": {
   "external_auc": 0.6699167397020158,
   "ci_high": 0.7094377647531406,
   "internal_auc": 0.9983197395596318,
   "supported": true
  },
  "resnet/montgomery": {
   "external_auc": 0.65625,
   "ci_high": 0.7484967672413793,
   "internal_auc": 0.9983197395596318,
   "supported": true
  },
  "vgg/shenzhen": {
   "external_auc": 0.4701376716330704,
   "ci_high": 0.5140618152936021,
   "internal_auc": 0.9917037140756817,
   "supported": true
  },
  "vgg/montgomery": {
   "external_auc": 0.5818965517241379,
   "ci_high": 0.6834159482758619,
   "internal_auc": 0.9917037140756817,
   "supported": true
  },
  "efficientnet/shenzhen": {
   "external_auc": 0.5217371457785569,
   "ci_high": 0.5654691151767455,
   "internal_auc": 0.9964294465642174,
   "supported": true
  },
  "efficientnet/montgomery": {
   "external_auc": 0.5379310344827586,
   "ci_high": 0.6398760775862069,
   "internal_auc": 0.9964294465642174,
   "supported": true
  }
 },
 "H4": {
  "resnet/shenzhen": {
   "diff": 0.0013876716330702976,
   "ci": [
    -0.01062915388548063,
    0.013220767601519204
   ],
   "supported": true
  },
  "resnet/montgomery": {
   "diff": -0.004310344827586299,
   "ci": [
    -0.028879310344827713,
    0.018970905172413854
   ],
   "supported": false
  },
  "vgg/shenzhen": {
   "diff": 0.007513511539585238,
   "ci": [
    -0.00020107544551561384,
    0.01563002118025129
   ],
   "supported": true
  },
  "vgg/montgomery": {
   "diff": -0.005603448275862011,
   "ci": [
    -0.02025862068965512,
    0.009488146551724152
   ],
   "supported": false
  },
  "efficientnet/shenzhen": {
   "diff": 0.009366783523225175,
   "ci": [
    -0.0025566936897459674,
    0.02094311459246278
   ],
   "supported": true
  },
  "efficientnet/montgomery": {
   "diff": 0.0118534482758621,
   "ci": [
    -0.00969827586206895,
    0.03168103448275861
   ],
   "supported": true
  }
 },
 "H2": {
  "montgomery_resnet_specificity": 0.0,
  "supported": true
 },
 "H3": {
  "testable": false,
  "note": "no confirmed overlaps (whole-image and crop search)"
 },
 "H5": {
  "resnet_specificity": {
   "shenzhen": 0.2085889570552147,
   "montgomery": 0.0
  },
  "resnet_auc": {
   "shenzhen": 0.6699167397020158,
   "montgomery": 0.65625
  },
  "supported": true
 }
}
```
