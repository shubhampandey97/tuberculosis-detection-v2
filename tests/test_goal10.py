"""Goal 10 (shortcut removal): in-model crop / standardisation, fine-tuning, label-independent augmentation,
crop-aware Grad-CAM geometry, the S2 mask, and a regression check that the default (goal 9) setup is unchanged."""
import inspect
import os

import keras
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.components import data_loader as dl
from src.components.model_builder import (CentralCrop, PerImageStandardize, build_model, crop_pixels,
                                          set_finetune)
from src.utils.config_loader import load_config
from tests.conftest import ROOT, TINY_SIZE, encode, synthetic_xray

V10_ARM1 = os.path.join(ROOT, "src", "config", "config_v10_arm1.yaml")
V10_ARM2 = os.path.join(ROOT, "src", "config", "config_v10_arm2.yaml")


# ---------------------------------------------------------------- configs
def test_default_config_unchanged(monkeypatch):
    monkeypatch.delenv("TB_CONFIG", raising=False)
    cfg = load_config()
    assert cfg["tag"] == "default" and cfg["models"] == ["vgg", "resnet", "efficientnet"]
    assert cfg["models_dir"] == "models/tensorflow" and "shortcut" not in cfg["augmentation"]


def test_v10_configs_select_by_env(monkeypatch):
    monkeypatch.setenv("TB_CONFIG", V10_ARM2)
    cfg = load_config()
    assert cfg["tag"] == "v10_arm2" and cfg["models"] == ["resnet", "efficientnet"]
    assert cfg["training"]["finetune"] and cfg["training"]["init_from"] == "models/tensorflow_v10_arm1"
    arm1 = load_config(V10_ARM1)
    assert arm1["model_options"] == cfg["model_options"] == {"crop_fraction": 0.12, "standardize": True}
    assert arm1["augmentation"]["shortcut"] == cfg["augmentation"]["shortcut"]


# ---------------------------------------------------------------- in-model layers
def test_central_crop_geometry(rng):
    x = rng.uniform(0, 255, (2, 224, 224, 3)).astype("float32")
    c = crop_pixels(224, 0.12)
    assert c == 27
    out = CentralCrop(0.12)(x).numpy()
    import tensorflow as tf
    expected = tf.image.resize(x[:, c:224 - c, c:224 - c, :], (224, 224)).numpy()
    assert out.shape == x.shape
    np.testing.assert_allclose(out, expected, atol=1e-4)


def test_standardize_gives_backbone_range(rng):
    x = rng.uniform(40, 90, (3, 64, 64, 3)).astype("float32")  # dark, low-contrast input
    y = PerImageStandardize()(x).numpy()
    assert y.min() >= 0 and y.max() <= 255
    for img in y:
        assert img.mean() == pytest.approx(128, abs=2.0) and img.std() == pytest.approx(50, abs=2.0)
    const = PerImageStandardize()(np.full((1, 8, 8, 3), 77.0, "float32")).numpy()
    assert np.all(np.isfinite(const)) and np.allclose(const, 128.0)  # no NaN on a constant image


def test_model_with_shortcut_layers(tmp_path, rng):
    m = build_model("resnet", img_size=TINY_SIZE, weights=None, crop_fraction=0.12, standardize=True)
    names = [layer.name for layer in m.layers]
    assert names.index("central_crop") < names.index("per_image_standardize") < names.index("caffe_preprocess")
    eff = build_model("efficientnet", img_size=TINY_SIZE, weights=None, crop_fraction=0.12, standardize=True)
    x = rng.uniform(0, 255, (4,) + TINY_SIZE + (3,)).astype("float32")
    p = eff.predict(x, verbose=0)
    assert np.all((p >= 0) & (p <= 1))
    eff.save(tmp_path / "e.keras")
    np.testing.assert_allclose(keras.models.load_model(tmp_path / "e.keras").predict(x, verbose=0), p, atol=1e-6)
    # what the backbone receives (after crop + standardisation) is in the 0-255 range
    probe = keras.Model(eff.inputs, eff.get_layer("per_image_standardize").output)
    seen = probe.predict(x, verbose=0)
    assert seen.min() >= 0 and seen.max() <= 255


@pytest.mark.parametrize("name, first", [("resnet", "conv5_block1_1_conv"), ("efficientnet", "block6a_expand_conv")])
def test_set_finetune(name, first):
    m = build_model(name, img_size=TINY_SIZE, weights=None, crop_fraction=0.12, standardize=True)
    head_only = sum(int(np.prod(w.shape)) for w in m.trainable_weights)
    set_finetune(m, first, 1e-5)
    start = [layer.name for layer in m.layers].index(first)
    for i, layer in enumerate(m.layers):
        if i < start or isinstance(layer, keras.layers.BatchNormalization):
            assert not layer.trainable, layer.name
    assert m.get_layer(first).trainable and m.get_layer("p_tb").trainable
    assert sum(int(np.prod(w.shape)) for w in m.trainable_weights) > head_only
    assert float(m.optimizer.learning_rate) == pytest.approx(1e-5)


# ---------------------------------------------------------------- augmentation
PARAMS = load_config(V10_ARM1)["augmentation"]["shortcut"]


def test_augmentation_never_sees_the_label():
    assert list(inspect.signature(dl.shortcut_augment).parameters) == ["image", "rng", "params", "force_markers"]


def test_augmentation_output_and_markers(rng, xray):
    img = np.asarray(Image.fromarray(xray).resize((224, 224)))
    out, info = dl.shortcut_augment(img, rng, PARAMS, force_markers=True)
    assert out.shape == img.shape and out.dtype == np.uint8 and info["markers"] >= 1
    assert np.array_equal(out[..., 0], out[..., 1])  # still gray
    no_change = dict(PARAMS, marker_p=0.0, jpeg_p=0.0, brightness=0.0, contrast=[1, 1], gamma=[1, 1])
    same, info = dl.shortcut_augment(img, rng, no_change)
    assert info["markers"] == 0 and np.array_equal(same, img)


def test_markers_applied_with_same_rate_in_both_classes(tmp_path, monkeypatch):
    """Through the real tf.data training pipeline: the share of images changed by markers is the same per class."""
    rng = np.random.default_rng(3)
    rows = []
    for i in range(160):
        p = tmp_path / f"img{i}.png"
        p.write_bytes(encode(synthetic_xray(rng, 224, 224), "PNG"))
        rows.append({"path": str(p), "label": i % 2})
    df = pd.DataFrame(rows)
    cfg = load_config(V10_ARM1)
    cfg["augmentation"]["shortcut"] = dict(PARAMS, marker_p=0.5, jpeg_p=0.0, brightness=0.0,
                                           contrast=[1, 1], gamma=[1, 1])
    cfg["training"]["batch_size"] = 16
    monkeypatch.setattr(dl, "build_augmentation", lambda config: (lambda x, training=True: x))  # no rotation/zoom
    plain = {r.path: np.asarray(Image.open(r.path).convert("RGB"), dtype=np.float32) for r in df.itertuples()}
    changed = {0: [], 1: []}
    ds = dl.make_dataset(cfg, df=df, training=True)
    for x, y in ds:
        for img, lab in zip(x.numpy(), y.numpy()):
            # identify the source image by its exact content where markers did not touch it: compare to all
            diffs = [np.abs(img - ref).mean() for ref in plain.values()]
            changed[int(lab)].append(min(diffs) > 0)
    r0, r1 = np.mean(changed[0]), np.mean(changed[1])
    assert 0.3 < r0 < 0.7 and 0.3 < r1 < 0.7 and abs(r0 - r1) < 0.2


# ---------------------------------------------------------------- evaluation geometry
def test_gradcam_maps_into_input_frame():
    from src.components.gradcam import upsample
    from src.pipeline import external_validation as ev
    maps = np.ones((1, 7, 7), dtype=np.float32)
    plain = ev.heatmaps_in_input_frame(maps, (224, 224))
    np.testing.assert_allclose(plain, upsample(maps, (224, 224)))
    framed = ev.heatmaps_in_input_frame(maps, (224, 224), 0.12)
    assert framed[0, :27].sum() == 0 and framed[0, :, -27:].sum() == 0  # nothing outside the crop
    assert np.allclose(framed[0, 27:197, 27:197], 1.0)


def test_resolution_ceiling_with_crop_is_bounded():
    from src.pipeline import external_validation as ev
    mask = np.zeros((224, 224), dtype=bool)
    mask[50:190, 40:100] = True
    mask[60:200, 125:185] = True
    c = ev.resolution_ceiling(mask, 7, 0.12)
    assert mask.mean() < c <= 1.0


def test_s2_mask_is_centre_of_cropped_field():
    from src.pipeline import evaluate_clinical as ec
    ec.configure(load_config(V10_ARM1))
    kind, frac = ec.MASKS["centre_in_crop"]
    assert kind == "centre" and int(round(frac * 224)) == 47  # 27 px crop + 20 px (12% of the 170-px field)
    apply, visible = ec.mask_fn((224, 224), kind, frac)
    out = apply(np.full((224, 224, 3), 200.0, "float32") * np.linspace(0, 1, 224)[None, :, None]).numpy()
    ref = 200.0 * np.linspace(0, 1, 224)
    assert np.allclose(out[100, :47, 0], ref[:47]) and np.allclose(out[100, 177:, 0], ref[177:])  # visible ring
    assert np.allclose(out[100, 47:177, 0], out[100, 47, 0])  # hidden centre (constant fill)
    ec.configure(load_config(os.path.join(ROOT, "src", "config", "config.yaml")))  # back to default
    assert ec.OUT_DIR == "artifacts/clinical" and ec.PRED_DIR == "artifacts/predictions"


# ---------------------------------------------------------------- regression: goal 9 setup reproduces itself
@pytest.mark.skipif(not (os.path.exists(os.path.join(ROOT, "artifacts/external/predictions/resnet_png.csv"))
                         and os.path.exists(os.path.join(ROOT, "models/tensorflow/resnet_model.keras"))
                         and os.path.isdir(os.path.join(ROOT, "data/external/prepared/png"))),
                    reason="goal 9 predictions, models or prepared external images not present")
def test_default_external_predictions_reproduce(monkeypatch):
    from src.components import inference
    from src.pipeline import external_validation as ev
    monkeypatch.delenv("TB_CONFIG", raising=False)
    cfg = load_config()
    ev.configure(cfg)
    assert ev.PRED_DIR == os.path.join("artifacts/external", "predictions") and ev.CLINICAL_DIR == "artifacts/clinical"
    saved = pd.read_csv(os.path.join(ROOT, "artifacts/external/predictions/resnet_png.csv")).head(16)
    inv = pd.read_csv(os.path.join(ROOT, "artifacts/external/inventory.csv")).set_index(["site", "file"])
    paths = [os.path.join(ROOT, inv.loc[(s, f), "png_path"]) for s, f in zip(saved.site, saved.file)]
    probs = ev.predict_paths(inference.load_model("resnet"), cfg, paths, saved.label)
    assert np.abs(probs - saved.prob.values).max() < 1e-4
