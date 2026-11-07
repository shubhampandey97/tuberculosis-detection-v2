"""Screening metrics, bootstrap and threshold tuning from evaluate_clinical, on known inputs."""
import numpy as np
import pytest

from src.pipeline import evaluate_clinical as ec

Y = np.array([0] * 5 + [1] * 15)


def test_perfect_classifier_scores_one():
    p = np.where(Y == 1, 0.95, 0.05)
    idx = ec.stratified_boot_indices(Y, 200, 42)
    res = ec.metrics_at(Y, p, 0.5, idx)
    for m in ec.METRICS:
        assert res[m]["value"] == 1.0 and res[m]["ci_low"] == 1.0 and res[m]["n_undefined"] == 0
    assert res["counts"] == {"tp": 15, "fp": 0, "tn": 5, "fn": 0}
    assert ec.auc_ci(Y, p, idx)["value"] == 1.0
    assert ec.brier_ci(Y, p, idx)["value"] == pytest.approx(0.0025)


def test_rates_on_known_counts():
    r = ec.rates(tp=8, fp=2, tn=6, fn=4)
    assert r["sensitivity"] == pytest.approx(8 / 12)
    assert r["specificity"] == pytest.approx(6 / 8)
    assert r["ppv"] == pytest.approx(8 / 10)
    assert r["npv"] == pytest.approx(6 / 10)
    assert r["f1"] == pytest.approx(16 / 22)


def test_undefined_ppv_is_nan_and_counted():
    p = np.full(len(Y), 0.1)  # never predicts TB
    idx = ec.stratified_boot_indices(Y, 100, 42)
    res = ec.metrics_at(Y, p, 0.5, idx)
    assert res["sensitivity"]["value"] == 0.0
    assert res["ppv"]["value"] is None and res["ppv"]["n_undefined"] == 100
    assert np.isnan(ec.rates(0, 0, 5, 5)["ppv"])


def test_counts_at_threshold_use_greater_or_equal():
    assert ec.counts(np.array([0, 1]), np.array([0.5, 0.5]), 0.5) == (1, 1, 0, 0)


def test_stratified_bootstrap_keeps_class_sizes():
    idx = ec.stratified_boot_indices(Y, 50, 42)
    assert idx.shape == (50, len(Y))
    assert np.all((Y[idx] == 0).sum(axis=1) == 5) and np.all((Y[idx] == 1).sum(axis=1) == 15)
    np.testing.assert_array_equal(idx, ec.stratified_boot_indices(Y, 50, 42))  # seeded


def test_tune_threshold_is_highest_meeting_target():
    y = np.array([0] * 10 + [1] * 20)
    p = np.concatenate([np.linspace(0.0, 0.3, 10), np.linspace(0.05, 1.0, 20)])
    res = ec.tune_threshold(y, p, target=0.95)  # needs 19 of 20 TB caught
    assert res["k_required"] == 19
    assert res["val_sensitivity"] >= 0.95
    tb_sorted = np.sort(p[y == 1])[::-1]
    assert res["threshold"] == pytest.approx(tb_sorted[18])
    above = np.nextafter(res["threshold"], 1.0)
    assert ec.rates(*ec.counts(y, p, above))["sensitivity"] < 0.95


def test_calibration_bins_cover_all_predictions(rng):
    p = rng.uniform(0, 1, 200)
    p[0], p[1] = 0.0, 1.0
    y = (rng.uniform(0, 1, 200) < p).astype(int)
    bins = ec.calibration_bins(y, p)
    assert len(bins) == ec.N_BINS and sum(b["n"] for b in bins) == 200


def test_truncation_never_rounds_up():
    assert ec.t3(0.9999) == "0.999"
    assert ec.t3(0.12345) == "0.123"
    assert ec.t3(1.0) == "1.000"
    assert ec.t3(None) == "n/a" and ec.t3(float("nan")) == "n/a"
