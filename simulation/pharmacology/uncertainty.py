"""Carry potency uncertainty into the tissue model instead of dropping it.

The pipeline hands the tissue simulation a single IC50 and reports a single growth
inhibition, which makes a prediction with a 95% conformal interval of three log
units look as definite as a measurement. The uncertainty is largest exactly where
it is currently discarded.

Conformal prediction produces a set, not a probability density: the guarantee is
that the true value lies in the interval with 95% marginal coverage, and it says
nothing about where inside it. So this module does not sample a distribution and
report a mean, which would invent a shape the method never claimed. It evaluates
the tissue model across the interval and reports the resulting range of responses.

Read the output as "the simulation is consistent with anything in this band", not
as a confidence interval on an effect.
"""

from __future__ import annotations

from typing import Any, Callable

import math

import numpy as np


def ic50_interval_um(point_ic50_um: float, q95_log_units: float) -> tuple[float, float]:
    """The IC50 range implied by a conformal interval on pIC50.

    A conformal half width of q on the log scale is a factor of 10**q either way,
    so an interval of 3.12 log units spans a factor of about 1330.
    """
    if point_ic50_um <= 0 or not math.isfinite(point_ic50_um):
        raise ValueError("point IC50 must be positive and finite")
    if q95_log_units <= 0:
        return point_ic50_um, point_ic50_um
    factor = 10.0 ** float(q95_log_units)
    return point_ic50_um / factor, point_ic50_um * factor


def potency_grid(point_ic50_um: float, q95_log_units: float, points: int = 5) -> list[float]:
    """Evenly spaced IC50 values across the interval, on the log scale.

    Log spacing matters: the interval is symmetric in log units, so linear
    spacing would put almost every sample at the insensitive end.
    """
    low, high = ic50_interval_um(point_ic50_um, q95_log_units)
    if low == high:
        return [point_ic50_um]
    count = max(3, int(points))
    return [float(v) for v in np.logspace(math.log10(low), math.log10(high), count)]


def propagate_potency_uncertainty(
    run_tissue: Callable[[float], list[float]],
    point_ic50_um: float,
    q95_log_units: float,
    points: int = 5,
) -> dict[str, Any]:
    """Evaluate the tissue response across the potency interval.

    run_tissue takes an IC50 in micromolar and returns the growth inhibition
    percentages it produced, typically one per seed. Keeping the simulation
    outside this function keeps the module free of engine imports and testable
    with a stub.
    """
    grid = potency_grid(point_ic50_um, q95_log_units, points)
    samples = []
    for ic50 in grid:
        values = [float(v) for v in run_tissue(ic50)]
        samples.append({
            "ic50_um": round(ic50, 6),
            "tgi_mean_pct": round(float(np.mean(values)), 3),
            "tgi_per_seed": [round(v, 3) for v in values],
        })

    means = [s["tgi_mean_pct"] for s in samples]
    low, high = min(means), max(means)
    point_index = min(range(len(grid)), key=lambda i: abs(grid[i] - point_ic50_um))

    # The band is only informative if it is narrow relative to the effect being
    # claimed. Spanning zero means the simulation does not establish a direction.
    spans_zero = low <= 0.0 <= high

    return {
        "point_ic50_um": round(point_ic50_um, 6),
        "conformal_q_95_log_units": round(float(q95_log_units), 4),
        "ic50_interval_um": [round(grid[0], 6), round(grid[-1], 6)],
        "ic50_interval_fold": round(10.0 ** (2 * float(q95_log_units)), 1),
        "samples": samples,
        "tgi_at_point_estimate_pct": samples[point_index]["tgi_mean_pct"],
        "tgi_range_pct": [round(low, 3), round(high, 3)],
        "tgi_band_width_pct": round(high - low, 3),
        "band_spans_zero": bool(spans_zero),
        "interpretation": (
            "The potency interval admits tissue responses from "
            f"{low:.1f} to {high:.1f} percent growth inhibition. "
            + ("That band includes no effect, so the simulation does not establish "
               "that this compound does anything."
               if spans_zero else
               "The band excludes zero, so within this model the direction of effect "
               "is consistent across the potency interval. It remains a simulation "
               "driven by a predicted potency, not a measurement.")
        ),
        "method": (
            "Conformal prediction yields a set with 95% marginal coverage, not a "
            "density. Responses are evaluated on a log spaced grid across that set "
            "and reported as a range. No distribution over the interval is assumed, "
            "so no mean or standard deviation of the band is reported."
        ),
    }
