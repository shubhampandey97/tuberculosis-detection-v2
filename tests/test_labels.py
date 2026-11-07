"""Label mapping: Normal = 0, TB = 1 everywhere (P(TB) is the sigmoid output)."""
import pandas as pd

from src.pipeline import make_splits
from tests.conftest import ROOT, encode, synthetic_xray


def test_class_dirs_mapping():
    assert make_splits.CLASS_DIRS == {"Normal Chest X-rays": 0, "TB Chest X-rays": 1}
    # Same as Keras' alphabetical folder order.
    assert sorted(make_splits.CLASS_DIRS) == list(make_splits.CLASS_DIRS)


def test_index_images_assigns_labels(tmp_path, rng):
    for folder, n in (("Normal Chest X-rays", 2), ("TB Chest X-rays", 3)):
        (tmp_path / folder).mkdir()
        for i in range(n):
            (tmp_path / folder / f"img{i}.jpg").write_bytes(encode(synthetic_xray(rng), "JPEG"))
    df = make_splits.index_images(str(tmp_path))
    assert (df[df.path.str.contains("/Normal Chest X-rays/")].label == 0).all()
    assert (df[df.path.str.contains("/TB Chest X-rays/")].label == 1).all()


def test_evaluate_class_names():
    from src.pipeline.evaluate import CLASS_NAMES
    assert CLASS_NAMES == ["Normal", "TB"]


def test_committed_splits_labels_match_folders():
    df = pd.read_csv(ROOT / "data" / "splits.csv")
    assert (df[df.path.str.contains("/TB Chest X-rays/")].label == 1).all()
    assert (df[df.path.str.contains("/Normal Chest X-rays/")].label == 0).all()
