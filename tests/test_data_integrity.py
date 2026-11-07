"""Integrity of the committed split. Hash checks need the images and skip without them."""
import hashlib

import pandas as pd
import pytest

from tests.conftest import ROOT, requires_data

SPLITS = ROOT / "data" / "splits.csv"
DUPLICATES = ROOT / "data" / "duplicates.csv"


@pytest.fixture(scope="module")
def splits():
    return pd.read_csv(SPLITS)


def test_splits_shape_and_counts(splits):
    assert list(splits.columns) == ["path", "label", "split", "width", "height"]
    assert len(splits) == 2982
    assert splits["path"].is_unique
    assert splits["label"].value_counts().to_dict() == {1: 2473, 0: 509}
    assert splits["split"].value_counts().to_dict() == {"train": 2087, "test": 448, "val": 447}
    test = splits[splits.split == "test"]
    assert (test.label == 0).sum() == 77 and (test.label == 1).sum() == 371


def test_duplicates_are_not_in_splits(splits):
    dups = pd.read_csv(DUPLICATES)
    assert len(dups) == 26
    assert not set(dups["dropped_path"]) & set(splits["path"])
    assert set(dups["kept_path"]) <= set(splits["path"])


def test_committed_thresholds():
    from src.components import inference
    assert inference.PRIMARY_MODEL == "resnet"
    assert inference.load_threshold("resnet") == pytest.approx(0.6142, abs=1e-4)


@pytest.mark.data
@requires_data
def test_hashes_unique_and_no_hash_in_two_splits(splits):
    def md5(rel):
        return hashlib.md5((ROOT / rel).read_bytes()).hexdigest()

    hashes = splits["path"].map(md5)
    assert hashes.is_unique
    assert splits.assign(h=hashes).groupby("h")["split"].nunique().max() == 1
