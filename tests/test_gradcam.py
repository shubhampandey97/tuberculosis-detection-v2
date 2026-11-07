"""Grad-CAM: NaN guard and gradient flow on a real (tiny) architecture."""
import numpy as np
import pytest

from src.components.gradcam import gradcam_signed, overlay, to_heatmap
from src.components.model_builder import LAST_CONV_LAYER
from tests.conftest import TINY_SIZE


def test_all_negative_map_returns_zeros_not_nan():
    hm, empty = to_heatmap(np.full((2, 3, 3), -1.0, dtype=np.float32))
    assert empty.tolist() == [True, True]
    assert not np.isnan(hm).any()
    assert np.all(hm == 0)


def test_mixed_batch_is_handled_per_image():
    cam = np.stack([np.full((3, 3), -2.0), np.arange(9, dtype=float).reshape(3, 3)]).astype(np.float32)
    hm, empty = to_heatmap(cam)
    assert empty.tolist() == [True, False]
    assert hm[1].max() == pytest.approx(1.0) and hm[1].min() == 0.0


def test_sign_flips_evidence():
    cam = np.array([[[1.0, -1.0]]], dtype=np.float32)
    tb, _ = to_heatmap(cam, 1.0)
    normal, _ = to_heatmap(cam, -1.0)
    assert tb[0, 0].tolist() == [1.0, 0.0] and normal[0, 0].tolist() == [0.0, 1.0]


@pytest.mark.parametrize("name", ["vgg", "efficientnet"])
def test_gradcam_on_model_is_finite_and_matches_output(tiny_models, rng, name):
    model = tiny_models[name]
    x = rng.uniform(0, 255, (2,) + TINY_SIZE + (3,)).astype("float32")
    layer = LAST_CONV_LAYER[name]
    cams, logits, probs = gradcam_signed(x, model, [layer])
    assert cams[layer].shape[0] == 2 and np.all(np.isfinite(cams[layer]))
    # The logit used as the Grad-CAM target reproduces the model output.
    np.testing.assert_allclose(1 / (1 + np.exp(-logits)), probs, atol=1e-5)
    np.testing.assert_allclose(probs, model.predict(x, verbose=0).ravel(), atol=1e-5)
    hm, _ = to_heatmap(cams[layer])
    out = overlay(x[0], hm[0])
    assert out.shape == TINY_SIZE + (3,) and out.dtype == np.uint8
