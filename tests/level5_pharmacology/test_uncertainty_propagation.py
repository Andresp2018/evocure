"""Potency uncertainty must reach the tissue output, not stop at the QSAR.

The failure this guards against is quiet: a point IC50 enters the simulation, a
single growth inhibition comes out, and a prediction with a thousandfold interval
is presented with the same confidence as a measurement.
"""

from __future__ import annotations

import math

import pytest

from simulation.pharmacology.uncertainty import (
    ic50_interval_um,
    potency_grid,
    propagate_potency_uncertainty,
)


def test_interval_matches_the_conformal_half_width():
    low, high = ic50_interval_um(0.1, 1.0)
    assert low == pytest.approx(0.01)
    assert high == pytest.approx(1.0)

    # The project's measured interval: 3.12 log units is a factor of about 1330
    # each way, so the full span is that squared.
    low, high = ic50_interval_um(0.1, 3.12)
    assert high / low == pytest.approx(10.0 ** 6.24, rel=0.01)


def test_zero_width_interval_collapses_to_the_point():
    assert ic50_interval_um(0.5, 0.0) == (0.5, 0.5)
    assert potency_grid(0.5, 0.0) == [0.5]


def test_grid_is_log_spaced_and_brackets_the_point():
    grid = potency_grid(1.0, 2.0, points=5)

    assert len(grid) == 5
    assert grid[0] == pytest.approx(0.01)
    assert grid[-1] == pytest.approx(100.0)
    assert grid[2] == pytest.approx(1.0)

    # Log spacing means constant ratios, not constant differences.
    ratios = [grid[i + 1] / grid[i] for i in range(len(grid) - 1)]
    assert all(r == pytest.approx(ratios[0], rel=1e-6) for r in ratios)


def test_invalid_point_estimates_are_refused():
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            ic50_interval_um(bad, 1.0)


def _threshold_tissue(dose_um: float = 5.0, maximum: float = 90.0):
    """A stand in for the tissue model: saturates below the dose, nothing above.

    This mirrors the measured transfer curve closely enough to test the wrapper
    without running a simulation.
    """
    def run(ic50_um: float) -> list[float]:
        value = maximum / (1.0 + (ic50_um / dose_um) ** 2)
        return [value - 1.0, value, value + 1.0]
    return run


def test_wide_interval_produces_a_wide_band():
    result = propagate_potency_uncertainty(
        _threshold_tissue(), point_ic50_um=0.1, q95_log_units=3.12, points=5)

    assert result["ic50_interval_fold"] == pytest.approx(10.0 ** 6.24, rel=0.01)
    assert len(result["samples"]) == 5
    # The potent end saturates, the weak end does nothing.
    assert result["tgi_range_pct"][0] < 1.0
    assert result["tgi_range_pct"][1] > 80.0
    assert result["tgi_band_width_pct"] > 75.0


def test_narrow_interval_produces_a_narrow_band():
    result = propagate_potency_uncertainty(
        _threshold_tissue(), point_ic50_um=0.1, q95_log_units=0.2, points=5)

    assert result["tgi_band_width_pct"] < 5.0
    assert not result["band_spans_zero"]


def test_point_estimate_is_reported_alongside_the_band():
    result = propagate_potency_uncertainty(
        _threshold_tissue(), point_ic50_um=1.0, q95_log_units=1.0, points=5)

    point = result["tgi_at_point_estimate_pct"]
    low, high = result["tgi_range_pct"]
    assert low <= point <= high, "the point estimate must sit inside its own band"


def test_band_spanning_zero_is_called_out():
    def sign_flipping(ic50_um: float) -> list[float]:
        return [20.0, 20.0] if ic50_um < 1.0 else [-5.0, -5.0]

    result = propagate_potency_uncertainty(
        sign_flipping, point_ic50_um=1.0, q95_log_units=2.0, points=5)

    assert result["band_spans_zero"] is True
    assert "does not establish" in result["interpretation"]


def test_no_distribution_statistics_are_reported():
    """Conformal prediction gives a set, so a mean over the band would be invented."""
    result = propagate_potency_uncertainty(
        _threshold_tissue(), point_ic50_um=0.1, q95_log_units=2.0, points=5)

    for forbidden in ("tgi_band_mean_pct", "tgi_band_sd_pct", "tgi_band_median_pct"):
        assert forbidden not in result
    assert "not a density" in result["method"]


def test_samples_record_every_evaluated_potency():
    result = propagate_potency_uncertainty(
        _threshold_tissue(), point_ic50_um=0.5, q95_log_units=1.5, points=7)

    assert len(result["samples"]) == 7
    for sample in result["samples"]:
        assert sample["ic50_um"] > 0
        assert len(sample["tgi_per_seed"]) == 3
        assert math.isfinite(sample["tgi_mean_pct"])
