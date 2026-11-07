"""Shared fixtures. Unit tests use tiny synthetic images and tiny untrained models (no downloads).

Tests that need the real dataset images or the trained models skip automatically when they are
missing (e.g. in CI, where data/ images and models/ are not in the repository).
"""
import io
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "Dataset of Tuberculosis Chest X-rays Images"
MODELS_DIR = ROOT / "models" / "tensorflow"
MODEL_NAMES = ["vgg", "resnet", "efficientnet"]

# TB_TESTS_SIMULATE_CI=1 runs the suite as CI sees it (no images, no models) on a full local checkout.
SIMULATE_CI = os.environ.get("TB_TESTS_SIMULATE_CI") == "1"
HAS_IMAGES = (DATA_DIR / "TB Chest X-rays" / "TB.1.jpg").exists() and not SIMULATE_CI
HAS_MODELS = all((MODELS_DIR / f"{m}_model.keras").exists() for m in MODEL_NAMES) and not SIMULATE_CI

requires_data = pytest.mark.skipif(not HAS_IMAGES, reason="dataset images not present in data/")
requires_models = pytest.mark.skipif(not HAS_MODELS, reason="trained models not present in models/tensorflow/")

TINY_SIZE = (32, 32)  # smallest input the Keras application backbones accept


def synthetic_xray(rng, width=256, height=256):
    """A grayscale 'X-ray-like' uint8 RGB array: smooth gradient + noise + a bright square marker."""
    yy, xx = np.mgrid[0:height, 0:width]
    base = 60 + 120 * np.exp(-(((xx - width / 2) / (width / 3)) ** 2 + ((yy - height / 2) / (height / 2.5)) ** 2))
    img = np.clip(base + rng.normal(0, 18, (height, width)), 0, 255)
    img[5:15, width - 15:width - 5] = 250  # marker-like bright block
    return np.repeat(img.astype(np.uint8)[..., None], 3, axis=2)


def encode(array_or_image, fmt, mode=None, **kwargs):
    img = array_or_image if isinstance(array_or_image, Image.Image) else Image.fromarray(array_or_image)
    if mode:
        img = img.convert(mode)
    buf = io.BytesIO()
    img.save(buf, format=fmt, **kwargs)
    return buf.getvalue()


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def xray(rng):
    return synthetic_xray(rng, width=300, height=260)  # non-square on purpose


@pytest.fixture(scope="session")
def tiny_models():
    """The real build_model() architectures, untrained (weights=None), at 32x32 input."""
    from src.components.model_builder import build_model

    return {name: build_model(name, img_size=TINY_SIZE, weights=None) for name in MODEL_NAMES}
