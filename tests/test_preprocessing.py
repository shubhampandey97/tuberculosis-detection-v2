"""Preprocessing parity between the app and training (regression tests for two past bugs).

Bug 1 (audit §5.1): training scaled /255 while evaluation/app used raw 0-255.
Bug 2 (goal 6): the app decoded JPEGs with PIL, training with TensorFlow -> scores moved up to 0.024.
"""
import io

import keras
import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

from src.components import inference
from src.components.data_loader import decode_and_resize, load_image
from src.components.model_builder import CaffePreprocess
from tests.conftest import encode

SIZE = (224, 224)


@pytest.mark.parametrize("fmt", ["JPEG", "PNG"])
def test_app_path_identical_to_training_path(tmp_path, xray, fmt):
    raw = encode(xray, fmt, **({"quality": 90} if fmt == "JPEG" else {}))
    path = tmp_path / f"img.{fmt.lower()}"
    path.write_bytes(raw)

    app = inference.preprocess_bytes(raw, SIZE)
    train_from_bytes = decode_and_resize(tf.constant(raw), SIZE).numpy()[None]
    train_from_file = load_image(tf.constant(str(path)), SIZE).numpy()[None]

    np.testing.assert_array_equal(app, train_from_bytes)
    np.testing.assert_array_equal(app, train_from_file)


def test_pixels_stay_in_0_255_float32(xray):
    x = inference.preprocess_bytes(encode(xray, "PNG"), SIZE)
    assert x.dtype == np.float32
    assert x.min() >= 0 and x.max() <= 255
    assert x.max() > 200  # catches a reintroduced /255 rescaling


@pytest.mark.parametrize("mode", ["L", "LA", "RGBA", "P", "RGB"])
def test_image_modes_give_rgb_224(xray, mode):
    raw = encode(xray, "PNG", mode=mode)
    x = inference.preprocess_bytes(raw)  # default size from config.yaml
    assert x.shape == (1, 224, 224, 3)
    # Same pixels as PIL's convert("RGB") on the same (lossless) file.
    via_pil = inference.preprocess(Image.open(io.BytesIO(raw)))
    np.testing.assert_allclose(x, via_pil, atol=1e-3)


def test_unreadable_bytes_raise():
    with pytest.raises(Exception):
        inference.preprocess_bytes(b"not an image")


@pytest.mark.parametrize("module", [keras.applications.vgg16, keras.applications.resnet50])
def test_caffe_preprocess_matches_keras(rng, module):
    x = rng.uniform(0, 255, (2, 16, 16, 3)).astype("float32")
    ours = CaffePreprocess()(x).numpy()
    reference = module.preprocess_input(x.copy())
    np.testing.assert_allclose(ours, reference, atol=1e-4)
