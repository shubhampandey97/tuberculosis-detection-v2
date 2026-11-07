"""Parity of the app inference path with the saved evaluation predictions (CLAUDE.md rule).

Needs the dataset images and the trained models; skips automatically without them.
"""
import pandas as pd
import pytest

from src.components import inference
from tests.conftest import MODEL_NAMES, ROOT, requires_data, requires_models

pytestmark = [pytest.mark.data, pytest.mark.models, requires_data, requires_models]
N_IMAGES = 16


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_saved_model_reproduces_saved_test_predictions(name):
    model = inference.load_model(name)
    saved = pd.read_csv(ROOT / "artifacts" / "predictions" / f"{name}_test.csv").head(N_IMAGES)
    for path, prob in zip(saved["path"], saved["prob"]):
        score = inference.predict(model, inference.preprocess_bytes((ROOT / path).read_bytes()))
        assert abs(score - prob) < 1e-4, f"{name} {path}: {score} vs saved {prob}"
