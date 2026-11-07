# External validation results (frozen models, val-tuned thresholds unchanged)

Thresholds: ResNet50 0.2437, EfficientNetB0 0.1788. 95% CIs: stratified bootstrap, 2000 resamples, seed 42. Values truncated to 3 decimals. Primary variant: lossless PNG. Overlap exclusions: confirmed 0, automatic (>= 0.95) 0 -> results with and without exclusions are identical.

## Main table (PNG, at the val-tuned threshold)

| model | data | n (N/TB) | ROC-AUC | sensitivity | specificity | PPV | NPV | F1 | Brier |
|---|---|---|---|---|---|---|---|---|---|
| ResNet50 | internal test | 448 (77/371) | 0.993 (0.986–0.998) | 0.924 (0.897–0.948) | 0.987 (0.961–1.000) | 0.997 (0.991–1.000) | 0.730 (0.663–0.802) | 0.959 (0.944–0.973) | 0.094 (0.076–0.114) |
| ResNet50 | shenzhen | 662 (326/336) | 0.445 (0.401–0.487) | 0.345 (0.294–0.395) | 0.592 (0.539–0.644) | 0.465 (0.417–0.512) | 0.467 (0.438–0.495) | 0.396 (0.347–0.443) | 0.385 (0.364–0.405) |
| ResNet50 | montgomery | 138 (80/58) | 0.673 (0.579–0.767) | 0.982 (0.947–1.000) | 0.012 (0.000–0.037) | 0.419 (0.407–0.429) | 0.500 (0.000–1.000) [279 undef.] | 0.587 (0.569–0.601) | 0.374 (0.340–0.408) |
| ResNet50 | pooled (secondary) | 800 (406/394) | 0.452 (0.412–0.492) | 0.439 (0.390–0.487) | 0.477 (0.433–0.524) | 0.449 (0.413–0.484) | 0.467 (0.434–0.500) | 0.444 (0.403–0.484) | 0.383 (0.361–0.405) |
| EfficientNetB0 | internal test | 448 (77/371) | 0.990 (0.983–0.996) | 0.940 (0.916–0.964) | 0.961 (0.909–1.000) | 0.991 (0.980–1.000) | 0.770 (0.704–0.845) | 0.965 (0.951–0.978) | 0.107 (0.089–0.127) |
| EfficientNetB0 | shenzhen | 662 (326/336) | 0.653 (0.611–0.694) | 0.306 (0.258–0.357) | 0.880 (0.846–0.914) | 0.725 (0.659–0.792) | 0.551 (0.532–0.572) | 0.430 (0.377–0.485) | 0.371 (0.360–0.382) |
| EfficientNetB0 | montgomery | 138 (80/58) | 0.642 (0.548–0.742) | 0.913 (0.827–0.982) | 0.025 (0.000–0.062) | 0.404 (0.380–0.424) | 0.285 (0.000–0.666) [2 undef.] | 0.560 (0.524–0.590) | 0.286 (0.245–0.322) |
| EfficientNetB0 | pooled (secondary) | 800 (406/394) | 0.590 (0.553–0.629) | 0.395 (0.350–0.444) | 0.711 (0.669–0.756) | 0.571 (0.524–0.619) | 0.548 (0.524–0.573) | 0.467 (0.423–0.514) | 0.357 (0.340–0.372) |

## At threshold 0.5 (for comparison)

| model | data | sensitivity | specificity | mean P(TB) Normal / TB |
|---|---|---|---|---|
| ResNet50 | shenzhen | 0.196 (0.154–0.241) | 0.809 (0.766–0.852) | 0.269 / 0.253 |
| ResNet50 | montgomery | 0.931 (0.862–0.982) | 0.100 (0.037–0.175) | 0.752 / 0.831 |
| EfficientNetB0 | shenzhen | 0.047 (0.026–0.071) | 0.993 (0.984–1.000) | 0.103 / 0.165 |
| EfficientNetB0 | montgomery | 0.741 (0.620–0.844) | 0.362 (0.262–0.462) | 0.571 / 0.671 |

## PNG vs JPEG (quality 75, 4:2:0) ROC-AUC

| model | site | AUC PNG | AUC JPEG | JPEG - PNG (95% CI, paired) |
|---|---|---|---|---|
| ResNet50 | shenzhen | 0.445 | 0.448 | +0.0035 (-0.0059 to +0.0135) |
| ResNet50 | montgomery | 0.673 | 0.671 | -0.0028 (-0.0265 to +0.0207) |
| ResNet50 | pooled | 0.452 | 0.456 | +0.0039 (-0.0032 to +0.0110) |
| EfficientNetB0 | shenzhen | 0.653 | 0.651 | -0.0021 (-0.0122 to +0.0081) |
| EfficientNetB0 | montgomery | 0.642 | 0.643 | +0.0006 (-0.0138 to +0.0149) |
| EfficientNetB0 | pooled | 0.590 | 0.590 | -0.0002 (-0.0076 to +0.0073) |

## Oracle analysis (NOT a result: threshold chosen on the external data itself)

| model | site | val-tuned threshold | sens / spec at val-tuned | oracle threshold for 95% sens | spec at oracle |
|---|---|---|---|---|---|
| ResNet50 | shenzhen | 0.2437 | 0.345 / 0.592 | 0.0152 | 0.030 |
| ResNet50 | montgomery | 0.2437 | 0.982 / 0.012 | 0.4093 | 0.062 |
| ResNet50 | pooled | 0.2437 | 0.439 / 0.477 | 0.0163 | 0.029 |
| EfficientNetB0 | shenzhen | 0.1788 | 0.306 / 0.880 | 0.0411 | 0.101 |
| EfficientNetB0 | montgomery | 0.1788 | 0.913 / 0.025 | 0.1642 | 0.025 |
| EfficientNetB0 | pooled | 0.1788 | 0.395 / 0.711 | 0.0437 | 0.105 |

## Subgroups (ResNet50, PNG, val-tuned threshold; * = fewer than 30 images in a class)

| site | variable | group | n (N/TB) | sensitivity | specificity | AUC |
|---|---|---|---|---|---|---|
| shenzhen | sex | F | 213 (105/108) | 0.305 (0.222–0.398) | 0.523 (0.428–0.619) | 0.373 (0.298–0.449) |
| shenzhen | sex | M | 448 (220/228) | 0.364 (0.302–0.425) | 0.622 (0.559–0.686) | 0.483 (0.428–0.540) |
| shenzhen | sex | unknown* | 1 (1/0) | n/a | 1.000 (1.000–1.000) | n/a |
| shenzhen | age band | 18-39 | 431 (218/213) | 0.211 (0.159–0.267) | 0.674 (0.614–0.734) | 0.407 (0.353–0.461) |
| shenzhen | age band | 40-59 | 147 (66/81) | 0.469 (0.370–0.580) | 0.393 (0.272–0.515) | 0.414 (0.325–0.506) |
| shenzhen | age band | <18* | 31 (24/7) | 0.000 (0.000–0.000) | 0.583 (0.375–0.791) | n/a |
| shenzhen | age band | >=60* | 52 (17/35) | 0.942 (0.857–1.000) | 0.294 (0.117–0.529) | 0.623 (0.438–0.789) |
| shenzhen | age band | unknown* | 1 (1/0) | n/a | 1.000 (1.000–1.000) | n/a |
| montgomery | sex | F* | 74 (53/21) | 1.000 (1.000–1.000) | 0.018 (0.000–0.056) | 0.739 (0.609–0.866) |
| montgomery | sex | M* | 63 (26/37) | 0.972 (0.918–1.000) | 0.000 (0.000–0.000) | 0.686 (0.543–0.817) |
| montgomery | sex | unknown* | 1 (1/0) | n/a | 0.000 (0.000–0.000) | n/a |
| montgomery | age band | 18-39* | 51 (34/17) | 0.941 (0.823–1.000) | 0.029 (0.000–0.088) | 0.425 (0.245–0.617) |
| montgomery | age band | 40-59* | 52 (31/21) | 1.000 (1.000–1.000) | 0.000 (0.000–0.000) | 0.751 (0.599–0.872) |
| montgomery | age band | <18* | 17 (13/4) | 1.000 (1.000–1.000) | 0.000 (0.000–0.000) | n/a |
| montgomery | age band | >=60* | 18 (2/16) | 1.000 (1.000–1.000) | 0.000 (0.000–0.000) | n/a |

## Pre-registered hypotheses

```
{
 "H1": {
  "resnet/shenzhen": {
   "external_auc": 0.4454334647969618,
   "ci_high": 0.4871510279725387,
   "internal_auc": 0.9936290124969369,
   "supported": true
  },
  "resnet/montgomery": {
   "external_auc": 0.6739224137931035,
   "ci_high": 0.7674568965517241,
   "internal_auc": 0.9936290124969369,
   "supported": true
  },
  "efficientnet/shenzhen": {
   "external_auc": 0.6531642565001461,
   "ci_high": 0.6942927439380661,
   "internal_auc": 0.9907235621521338,
   "supported": true
  },
  "efficientnet/montgomery": {
   "external_auc": 0.6428879310344828,
   "ci_high": 0.7420312499999999,
   "internal_auc": 0.9907235621521338,
   "supported": true
  }
 },
 "H4": {
  "resnet/shenzhen": {
   "diff": 0.003487437919953207,
   "ci": [
    -0.0058731832456909,
    0.013494193689745885
   ],
   "supported": true
  },
  "resnet/montgomery": {
   "diff": -0.0028017241379311164,
   "ci": [
    -0.026514008620689686,
    0.020689655172413834
   ],
   "supported": false
  },
  "efficientnet/shenzhen": {
   "diff": -0.0021088957055215296,
   "ci": [
    -0.012169743280747927,
    0.008098935509786548
   ],
   "supported": false
  },
  "efficientnet/montgomery": {
   "diff": 0.000646551724137745,
   "ci": [
    -0.013793103448275784,
    0.014870689655172353
   ],
   "supported": true
  }
 },
 "H2": {
  "montgomery_resnet_specificity": 0.0125,
  "supported": true
 },
 "H3": {
  "testable": false,
  "note": "no confirmed overlaps (whole-image and crop search)"
 },
 "H5": {
  "resnet_specificity": {
   "shenzhen": 0.5920245398773006,
   "montgomery": 0.0125
  },
  "resnet_auc": {
   "shenzhen": 0.4454334647969618,
   "montgomery": 0.6739224137931035
  },
  "supported": false
 }
}
```
