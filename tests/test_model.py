"""Model building, prediction range and save/reload (tiny untrained models, no downloads)."""
import keras
import numpy as np
import pytest

from src.components.model_builder import LAST_CONV_LAYER, NEEDS_CAFFE, build_model
from tests.conftest import MODEL_NAMES, TINY_SIZE


@pytest.fixture
def batch(rng):
    return rng.uniform(0, 255, (4,) + TINY_SIZE + (3,)).astype("float32")


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_predictions_in_unit_interval(tiny_models, batch, name):
    p = tiny_models[name].predict(batch, verbose=0)
    assert p.shape == (4, 1)
    assert np.all(np.isfinite(p)) and p.min() >= 0 and p.max() <= 1


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_structure(tiny_models, name):
    model = tiny_models[name]
    model.get_layer(LAST_CONV_LAYER[name])  # Grad-CAM layer exists (flat graph)
    has_caffe = any(layer.name == "caffe_preprocess" for layer in model.layers)
    assert has_caffe == (name in NEEDS_CAFFE)
    assert model.layers[-1].name == "p_tb"


@pytest.mark.parametrize("name", ["vgg", "efficientnet"])  # with and without the custom layer
def test_reload_gives_identical_output(tiny_models, batch, tmp_path, name):
    model = tiny_models[name]
    path = tmp_path / f"{name}.keras"
    model.save(path)
    reloaded = keras.models.load_model(path)
    np.testing.assert_allclose(reloaded.predict(batch, verbose=0), model.predict(batch, verbose=0), atol=1e-6)


def test_invalid_model_name():
    with pytest.raises(ValueError):
        build_model("alexnet", img_size=TINY_SIZE, weights=None)
