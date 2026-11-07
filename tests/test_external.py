"""Goal 9 helpers: readings parser, grayscale/bit-depth conversion, crop search, lung-mask geometry, metrics."""
import os

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.pipeline import data_audit as da
from src.pipeline import external_validation as ev
from tests.conftest import ROOT, synthetic_xray


# ---------------------------------------------------------------- readings
@pytest.mark.parametrize("text, expected", [
    ("male 45yrs\nnormal", ("M", 45, "normal", True)),
    ("female 23yrs\nsecondary PTB in the right upper field", ("F", 23, "secondary PTB in the right upper field", True)),
    ("Male 35 years\nnormal", ("M", 35, "normal", True)),
    ("male 42\nnormal", (None, None, "male 42 | normal", False)),        # no unit: stays unknown
    ("femal 32yrs\nnormal", (None, None, "femal 32yrs | normal", False)),  # typo: rejected by the strict parser
])
def test_shenzhen_parser(text, expected):
    assert ev.parse_reading("shenzhen", text) == expected


def test_montgomery_parser():
    sex, age, findings, ok = ev.parse_reading("montgomery", "Patient's Sex: F \nPatient's Age: 027Y\nnormal\n")
    assert (sex, age, findings, ok) == ("F", 27, "normal", True)
    sex, age, _, ok = ev.parse_reading("montgomery", "Patient's Sex: O \nPatient's Age: 005Y\nnormal")
    assert sex is None and age == 5 and not ok  # sex "O" stays unknown


def test_reading_corrections_are_explicit():
    for (site, file), (sex, age, note) in ev.READING_CORRECTIONS.items():
        assert site == "shenzhen" and sex in ("M", "F") and note
    assert ev.READING_CORRECTIONS[("shenzhen", "CHNCXR_0115_0.png")][1] == pytest.approx(16 / 12)  # 1.33 y, not < 1
    assert ("shenzhen", "CHNCXR_0018_0.png") not in ev.READING_CORRECTIONS  # "male 42" stays unknown


@pytest.mark.parametrize("age, band", [(0.18, "<18"), (17.9, "<18"), (18, "18-39"), (39, "18-39"), (40, "40-59"),
                                       (59, "40-59"), (60, ">=60"), (None, "unknown"), (float("nan"), "unknown")])
def test_age_band(age, band):
    assert ev.age_band(age) == band


# ---------------------------------------------------------------- grayscale and bit depth
def test_palette_and_rgb_gray(xray):
    gray = xray[..., 0]
    pal_img = Image.fromarray(gray).convert("P")
    g, info = ev.to_gray8(pal_img)
    assert info["palette_gray"] and np.array_equal(g, np.asarray(pal_img.convert("L")))
    g, info = ev.to_gray8(Image.fromarray(xray))
    assert info["rgb_channels_equal"] and np.array_equal(g, gray)


def test_non_gray_rgb_is_flagged(xray):
    colour = xray.copy()
    colour[..., 0] = 255 - colour[..., 0]
    _, info = ev.to_gray8(Image.fromarray(colour))
    assert info["rgb_channels_equal"] is False


def test_16bit_scaled_by_real_bit_depth():
    a = np.array([[0, 2048, 4095]], dtype=np.uint16)  # 12-bit data stored in 16 bits
    g, info = ev.to_gray8(Image.fromarray(a), bits=12)
    assert info["bits"] == 12 and g.tolist() == [[0, 128, 255]]  # a blind /256 would give 15 as the maximum


# ---------------------------------------------------------------- crop search
def test_crop_search_finds_a_cropped_copy(rng):
    img = Image.fromarray(synthetic_xray(rng)[..., 0])
    combos, vecs = ev.crop_vectors(img)
    assert len(combos) == 81 and vecs.shape == (81, da.SIM_SIZE ** 2)
    assert combos[0] == (0.0, 0.0, 0.0, 0.0)
    np.testing.assert_allclose(vecs[0], da.sim_vector(img), atol=1e-5)
    w, h = img.size
    copy = img.crop((int(0.05 * w), int(0.1 * h), w - int(0.1 * w), h)).resize((256, 256), Image.LANCZOS)
    corr = vecs @ da.sim_vector(copy) / vecs.shape[1]
    assert corr.max() > 0.99 and corr.max() > corr[0]


# ---------------------------------------------------------------- lung-mask geometry
def test_mask_and_image_stay_aligned(tmp_path):
    """A bright region and its mask go through the image path and the mask path: they must coincide."""
    h, w = 410, 330  # non-square, so the stretch is exercised
    img = np.full((h, w), 40, dtype=np.uint8)
    mask = np.zeros((h, w), dtype=bool)
    mask[60:300, 50:150] = True   # "left lung"
    mask[90:350, 190:300] = True  # "right lung"
    img[mask] = 220
    path = tmp_path / "x.png"
    Image.fromarray(img).save(path)
    png, _, _ = ev.prepare_image(str(path), 75, 2)
    bright = ev.model_input(png)[..., 0] > 128
    grid = ev.mask_to_model_grid(mask, (224, 224))
    assert grid.shape == (224, 224) and grid.dtype == bool
    iou = (bright & grid).sum() / (bright | grid).sum()
    assert iou > 0.95


@pytest.mark.skipif(not os.path.isdir(os.path.join(ROOT, ev.MASK_ROOT)), reason="Montgomery masks not present")
def test_real_montgomery_masks_load():
    files = sorted(os.listdir(os.path.join(ROOT, ev.MASK_ROOT, "leftMask")))
    f = next(x for x in files if x.endswith(".png"))
    m = ev.load_lung_mask(f)
    with Image.open(os.path.join(ROOT, ev.SITES["montgomery"]["root"], "CXR_png", f)) as im:
        assert m.shape == (im.height, im.width)
    share = ev.mask_to_model_grid(m, (224, 224)).mean()
    assert 0.1 < share < 0.6


def test_resolution_ceiling_is_bounded_and_coarser_is_lower():
    mask = np.zeros((224, 224), dtype=bool)
    mask[40:180, 30:100] = True
    mask[50:190, 125:200] = True
    c7, c14 = ev.resolution_ceiling(mask, 7), ev.resolution_ceiling(mask, 14)
    assert mask.mean() < c7 <= c14 <= 1.0  # better than uniform, and finer grids get closer to 1
    assert ev.resolution_ceiling(np.ones((224, 224), dtype=bool), 7) == pytest.approx(1.0)


# ---------------------------------------------------------------- metric helpers
def test_evaluate_block_perfect_and_subgroups():
    y = np.array([0] * 12 + [1] * 12)
    p = np.where(y == 1, 0.9, 0.1)
    block, idx = ev.evaluate_block(y, p, 0.5, 200, 42)
    assert block["roc_auc"]["value"] == 1.0 and block["at_tuned"]["specificity"]["value"] == 1.0
    assert idx.shape == (200, 24)
    df = pd.DataFrame({"site": "montgomery", "label": y, "prob": p, "sex": ["F", "M"] * 12,
                       "age_band": ["18-39"] * 24})
    rows = pd.DataFrame(ev.subgroup_rows(df, "resnet", 0.5, 200, 42))
    by_sex = rows[rows.variable == "sex"]
    assert by_sex["small"].all()                      # 6 per class < 30
    assert by_sex["auc"].isna().all()                 # fewer than 10 per class -> no AUC
    age = rows[rows.variable == "age band"].iloc[0]
    assert age.auc == 1.0 and age.sensitivity == 1.0 and age.specificity == 1.0
