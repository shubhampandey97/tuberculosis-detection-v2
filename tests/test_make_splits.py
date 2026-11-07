"""make_splits on a tiny synthetic dataset: de-duplication, no leakage, stratification."""
import shutil

import pytest

from src.pipeline import make_splits
from tests.conftest import encode, synthetic_xray

N_NORMAL, N_TB, N_TB_NONSQUARE = 40, 160, 20
NORMAL_DUPS = {41: 3, 42: 7}          # new file number -> number of the original it copies
TB_DUPS = {161: 5, 162: 9, 163: 150}


def _write(folder, name, rng, width=256, height=256):
    (folder / name).write_bytes(encode(synthetic_xray(rng, width, height), "JPEG", quality=85))


@pytest.fixture
def dataset(tmp_path, rng):
    normal, tb = tmp_path / "Normal Chest X-rays", tmp_path / "TB Chest X-rays"
    normal.mkdir()
    tb.mkdir()
    for i in range(1, N_NORMAL + 1):
        _write(normal, f"others ({i}).jpg", rng)
    for i in range(1, N_TB + 1):
        height = 221 if i <= N_TB_NONSQUARE else 256  # first 20 TB images are non-square
        _write(tb, f"TB.{i}.jpg", rng, height=height)
    for new, orig in NORMAL_DUPS.items():
        shutil.copy(normal / f"others ({orig}).jpg", normal / f"others ({new}).jpg")
    for new, orig in TB_DUPS.items():
        shutil.copy(tb / f"TB.{orig}.jpg", tb / f"TB.{new}.jpg")
    return tmp_path


def test_duplicates_dropped_keeping_lower_number(dataset):
    df = make_splits.index_images(str(dataset))
    assert len(df) == N_NORMAL + N_TB + len(NORMAL_DUPS) + len(TB_DUPS)
    kept, dups = make_splits.drop_duplicates(df)
    assert len(dups) == len(NORMAL_DUPS) + len(TB_DUPS)
    assert kept["md5"].is_unique and len(kept) == N_NORMAL + N_TB
    for _, row in dups.iterrows():
        assert make_splits.file_number(row.kept_path) < make_splits.file_number(row.dropped_path)


def test_split_has_no_leakage_and_is_stratified(dataset):
    df = make_splits.index_images(str(dataset))
    n_files = len(df)
    kept, dups = make_splits.drop_duplicates(df)
    split = make_splits.stratified_split(kept)
    make_splits.check(split, n_files, dups)  # the script's own assertions

    assert split.groupby("md5")["split"].nunique().max() == 1  # no hash in two splits
    sizes = split["split"].value_counts()
    assert (sizes["train"], sizes["val"], sizes["test"]) == (140, 30, 30)
    expected_normal = {"train": 28, "val": 6, "test": 6}  # 20% Normal in every split
    for name, part in split.groupby("split"):
        assert abs((part.label == 0).sum() - expected_normal[name]) <= 1
        nonsquare = make_splits.is_nonstandard(part).sum()
        assert abs(nonsquare - N_TB_NONSQUARE * len(part) / len(split)) <= 1


def test_split_is_deterministic(dataset):
    kept, _ = make_splits.drop_duplicates(make_splits.index_images(str(dataset)))
    a = make_splits.stratified_split(kept).sort_values("path")["split"].tolist()
    b = make_splits.stratified_split(kept).sort_values("path")["split"].tolist()
    assert a == b


def test_cross_class_duplicate_raises(dataset):
    shutil.copy(dataset / "Normal Chest X-rays" / "others (1).jpg", dataset / "TB Chest X-rays" / "TB.999.jpg")
    with pytest.raises(ValueError, match="span both classes"):
        make_splits.drop_duplicates(make_splits.index_images(str(dataset)))


def test_unexpected_class_folders_raise(tmp_path):
    (tmp_path / "Normal").mkdir()
    (tmp_path / "TB").mkdir()
    with pytest.raises(ValueError, match="Expected class folders"):
        make_splits.index_images(str(tmp_path))
