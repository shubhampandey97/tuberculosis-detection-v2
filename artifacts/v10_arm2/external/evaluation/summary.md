# External validation results (frozen models, val-tuned thresholds unchanged)

Thresholds: ResNet50 1.0000, EfficientNetB0 0.2267. 95% CIs: stratified bootstrap, 2000 resamples, seed 42. Values truncated to 3 decimals. Primary variant: lossless PNG. Overlap exclusions: confirmed 0, automatic (>= 0.95) 0 -> results with and without exclusions are identical.

## Main table (PNG, at the val-tuned threshold)

| model | data | n (N/TB) | ROC-AUC | sensitivity | specificity | PPV | NPV | F1 | Brier |
|---|---|---|---|---|---|---|---|---|---|
| ResNet50 | internal test | 448 (77/371) | 0.999 (0.999–1.000) | 0.921 (0.892–0.948) | 1.000 (1.000–1.000) | 1.000 (1.000–1.000) | 0.726 (0.658–0.802) | 0.959 (0.943–0.973) | 0.002 (0.000–0.006) |
| ResNet50 | shenzhen | 662 (326/336) | 0.502 (0.461–0.545) | 0.342 (0.288–0.392) | 0.625 (0.573–0.674) | 0.485 (0.433–0.533) | 0.480 (0.452–0.507) | 0.401 (0.348–0.449) | 0.459 (0.440–0.477) |
| ResNet50 | montgomery | 138 (80/58) | 0.571 (0.504–0.640) | 0.948 (0.879–1.000) | 0.062 (0.012–0.125) | 0.423 (0.401–0.442) | 0.625 (0.250–1.000) | 0.585 (0.553–0.610) | 0.579 (0.579–0.579) |
| ResNet50 | pooled (secondary) | 800 (406/394) | 0.493 (0.455–0.531) | 0.431 (0.380–0.482) | 0.514 (0.467–0.561) | 0.463 (0.425–0.500) | 0.482 (0.451–0.513) | 0.446 (0.403–0.487) | 0.480 (0.466–0.494) |
| EfficientNetB0 | internal test | 448 (77/371) | 0.999 (0.998–1.000) | 0.954 (0.932–0.973) | 1.000 (1.000–1.000) | 1.000 (1.000–1.000) | 0.819 (0.754–0.885) | 0.976 (0.965–0.986) | 0.059 (0.044–0.075) |
| EfficientNetB0 | shenzhen | 662 (326/336) | 0.605 (0.561–0.646) | 0.208 (0.166–0.252) | 0.926 (0.898–0.953) | 0.744 (0.666–0.826) | 0.531 (0.516–0.547) | 0.325 (0.269–0.382) | 0.397 (0.382–0.412) |
| EfficientNetB0 | montgomery | 138 (80/58) | 0.684 (0.587–0.781) | 0.862 (0.775–0.948) | 0.162 (0.087–0.250) | 0.427 (0.392–0.462) | 0.619 (0.421–0.812) | 0.571 (0.523–0.616) | 0.310 (0.258–0.359) |
| EfficientNetB0 | pooled (secondary) | 800 (406/394) | 0.567 (0.529–0.605) | 0.304 (0.258–0.352) | 0.775 (0.733–0.815) | 0.568 (0.511–0.627) | 0.534 (0.514–0.556) | 0.396 (0.347–0.445) | 0.382 (0.362–0.402) |

## At threshold 0.5 (for comparison)

| model | data | sensitivity | specificity | mean P(TB) Normal / TB |
|---|---|---|---|---|
| ResNet50 | shenzhen | 0.925 (0.898–0.955) | 0.088 (0.058–0.119) | 0.905 / 0.922 |
| ResNet50 | montgomery | 1.000 (1.000–1.000) | 0.000 (0.000–0.000) | 0.999 / 0.999 |
| EfficientNetB0 | shenzhen | 0.092 (0.062–0.125) | 0.987 (0.975–0.996) | 0.072 / 0.152 |
| EfficientNetB0 | montgomery | 0.775 (0.672–0.879) | 0.375 (0.274–0.487) | 0.574 / 0.729 |

## PNG vs JPEG (quality 75, 4:2:0) ROC-AUC

| model | site | AUC PNG | AUC JPEG | JPEG - PNG (95% CI, paired) |
|---|---|---|---|---|
| ResNet50 | shenzhen | 0.502 | 0.520 | +0.0181 (+0.0091 to +0.0273) |
| ResNet50 | montgomery | 0.571 | 0.611 | +0.0398 (-0.0031 to +0.0870) |
| ResNet50 | pooled | 0.493 | 0.510 | +0.0171 (+0.0089 to +0.0258) |
| EfficientNetB0 | shenzhen | 0.605 | 0.605 | +0.0001 (-0.0091 to +0.0096) |
| EfficientNetB0 | montgomery | 0.684 | 0.689 | +0.0043 (-0.0093 to +0.0172) |
| EfficientNetB0 | pooled | 0.567 | 0.568 | +0.0011 (-0.0055 to +0.0078) |

## Oracle analysis (NOT a result: threshold chosen on the external data itself)

| model | site | val-tuned threshold | sens / spec at val-tuned | oracle threshold for 95% sens | spec at oracle |
|---|---|---|---|---|---|
| ResNet50 | shenzhen | 1.0000 | 0.342 / 0.625 | 0.3173 | 0.064 |
| ResNet50 | montgomery | 1.0000 | 0.948 / 0.062 | 1.0000 | 0.037 |
| ResNet50 | pooled | 1.0000 | 0.431 / 0.514 | 0.3740 | 0.056 |
| EfficientNetB0 | shenzhen | 0.2267 | 0.208 / 0.926 | 0.0056 | 0.064 |
| EfficientNetB0 | montgomery | 0.2267 | 0.862 / 0.162 | 0.0428 | 0.012 |
| EfficientNetB0 | pooled | 0.2267 | 0.304 / 0.775 | 0.0066 | 0.081 |

## Subgroups (ResNet50, PNG, val-tuned threshold; * = fewer than 30 images in a class)

| site | variable | group | n (N/TB) | sensitivity | specificity | AUC |
|---|---|---|---|---|---|---|
| shenzhen | sex | F | 213 (105/108) | 0.296 (0.212–0.379) | 0.571 (0.485–0.657) | 0.437 (0.361–0.514) |
| shenzhen | sex | M | 448 (220/228) | 0.364 (0.302–0.429) | 0.650 (0.590–0.713) | 0.537 (0.480–0.589) |
| shenzhen | sex | unknown* | 1 (1/0) | n/a | 1.000 (1.000–1.000) | n/a |
| shenzhen | age band | 18-39 | 431 (218/213) | 0.220 (0.164–0.276) | 0.711 (0.646–0.770) | 0.477 (0.423–0.532) |
| shenzhen | age band | 40-59 | 147 (66/81) | 0.456 (0.358–0.567) | 0.515 (0.393–0.636) | 0.513 (0.422–0.603) |
| shenzhen | age band | <18* | 31 (24/7) | 0.285 (0.000–0.571) | 0.375 (0.166–0.583) | n/a |
| shenzhen | age band | >=60* | 52 (17/35) | 0.828 (0.685–0.942) | 0.294 (0.117–0.529) | 0.584 (0.428–0.730) |
| shenzhen | age band | unknown* | 1 (1/0) | n/a | 1.000 (1.000–1.000) | n/a |
| montgomery | sex | F* | 74 (53/21) | 1.000 (1.000–1.000) | 0.037 (0.000–0.094) | 0.597 (0.512–0.680) |
| montgomery | sex | M* | 63 (26/37) | 0.918 (0.837–1.000) | 0.115 (0.000–0.269) | 0.570 (0.465–0.677) |
| montgomery | sex | unknown* | 1 (1/0) | n/a | 0.000 (0.000–0.000) | n/a |
| montgomery | age band | 18-39* | 51 (34/17) | 0.882 (0.705–1.000) | 0.058 (0.000–0.147) | 0.473 (0.332–0.616) |
| montgomery | age band | 40-59* | 52 (31/21) | 0.952 (0.857–1.000) | 0.096 (0.000–0.225) | 0.663 (0.557–0.758) |
| montgomery | age band | <18* | 17 (13/4) | 1.000 (1.000–1.000) | 0.000 (0.000–0.000) | n/a |
| montgomery | age band | >=60* | 18 (2/16) | 1.000 (1.000–1.000) | 0.000 (0.000–0.000) | n/a |

## Pre-registered hypotheses

```
{
 "H1": {
  "resnet/shenzhen": {
   "external_auc": 0.5025014607069822,
   "ci_high": 0.5450439809742914,
   "internal_auc": 0.999894983722477,
   "supported": true
  },
  "resnet/montgomery": {
   "external_auc": 0.5719827586206896,
   "ci_high": 0.6407435344827584,
   "internal_auc": 0.999894983722477,
   "supported": true
  },
  "efficientnet/shenzhen": {
   "external_auc": 0.6058464796961729,
   "ci_high": 0.6462140300905637,
   "internal_auc": 0.9993699023348619,
   "supported": true
  },
  "efficientnet/montgomery": {
   "external_auc": 0.684698275862069,
   "ci_high": 0.7819073275862068,
   "internal_auc": 0.9993699023348619,
   "supported": true
  }
 },
 "H4": {
  "resnet/shenzhen": {
   "diff": 0.018099072451066345,
   "ci": [
    0.009066539767747439,
    0.027337473524685903
   ],
   "supported": true
  },
  "resnet/montgomery": {
   "diff": 0.03976293103448281,
   "ci": [
    -0.003133081896551767,
    0.08696120689655162
   ],
   "supported": true
  },
  "efficientnet/shenzhen": {
   "diff": 0.00011868244230206493,
   "ci": [
    -0.009147905711364252,
    0.009559642491966024
   ],
   "supported": true
  },
  "efficientnet/montgomery": {
   "diff": 0.004310344827586299,
   "ci": [
    -0.009272629310344931,
    0.01724137931034475
   ],
   "supported": true
  }
 },
 "H2": {
  "montgomery_resnet_specificity": 0.0625,
  "supported": true
 },
 "H3": {
  "testable": false,
  "note": "no confirmed overlaps (whole-image and crop search)"
 },
 "H5": {
  "resnet_specificity": {
   "shenzhen": 0.6257668711656442,
   "montgomery": 0.0625
  },
  "resnet_auc": {
   "shenzhen": 0.5025014607069822,
   "montgomery": 0.5719827586206896
  },
  "supported": false
 }
}
```
