#!/usr/bin/env python3
"""Test the PDAC tissue model against published measurements.

No kinetic, diffusion or clonal parameter in the tissue model has ever been
fitted to a measurement, so simulated growth inhibition is a restatement of the
IC50 it was handed. The clearest symptom is in the platform's own report:
gemcitabine, scored the way a candidate is scored, returns -0.99% growth
inhibition with an interval spanning zero. The positive control fails its own
test.

This harness is the first step out of that. It reads
data/reference/tissue_calibration_observations.json, runs the model against each
observation, and reports the discrepancy.

Three design choices worth knowing, because each is a way this script could have
lied instead:

1. It compares RATIOS, not absolute potencies. Published PANC-1 monolayer
   gemcitabine IC50 values span 58 nM to 31.41 uM across studies, so absolute
   values cannot be pooled and cannot be targets. A ratio measured within one
   study has the assay confounders cancel.

2. It does NOT fit parameters. Four observations against five free parameters
   would be overfitting with extra steps, and a least-squares number would look
   like a calibration. Instead it sweeps one parameter at a time against the
   observations that constrain it, and reports which values are consistent.

3. It refuses to call the result a calibration while the observations are
   unverified. Every record carries a verification field, and the ones here were
   transcribed from search engine summaries rather than read in the papers.
   The comparison still runs, so the harness can be exercised and reviewed, but
   the output says plainly that it is not a calibration. Promoting a record to
   verified_primary needs no code, only someone opening the paper.

It also reports which observations the model cannot express at all. A harness
that silently drops those overstates its own coverage.

    python scripts/calibrate_tissue_model.py
    python scripts/calibrate_tissue_model.py --sweep-beta-ecm
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from rdkit import RDLogger

from simulation.biology.diseases.pdac import CellDeathMode, CellType, PDACModule
from simulation.chemistry.engine import ChemistryEngine
from simulation.core.paths import reference_data_dir

RDLogger.DisableLog("rdApp.*")

OBSERVATIONS = "tissue_calibration_observations.json"

# A ratio is "consistent" when it lands inside this factor of the measurement.
# Deliberately loose: these are biological ratios from single studies, and a
# tighter band would manufacture precision the observations do not carry.
TOLERANCE_FACTOR = 2.0


def _mean_at_distance(field: np.ndarray, source_x: np.ndarray, source_y: np.ndarray,
                      dx: float, dy: float, distance_um: float,
                      peak: float) -> float:
    """Mean relative concentration in the shell at a given distance from a source.

    Vectorised over the grid. The scalar loop this replaces was
    O(nx * ny * sources) in Python and dominated the harness runtime.
    """
    gy, gx = np.indices(field.shape)
    # (cells, sources) distance matrix, then the nearest source per cell.
    diff_x = (gx.ravel()[:, None] - source_x[None, :]) * dx
    diff_y = (gy.ravel()[:, None] - source_y[None, :]) * dy
    nearest = np.min(np.hypot(diff_x, diff_y), axis=1)
    shell = np.abs(nearest - distance_um) <= max(dx, dy)
    if not np.any(shell):
        return float("nan")
    return float(np.mean(field.ravel()[shell] / peak))


def _furthest_distance(field: np.ndarray, source_x: np.ndarray, source_y: np.ndarray,
                       dx: float, dy: float) -> float:
    """The longest distance from any grid point to its nearest drug source."""
    gy, gx = np.indices(field.shape)
    diff_x = (gx.ravel()[:, None] - source_x[None, :]) * dx
    diff_y = (gy.ravel()[:, None] - source_y[None, :]) * dy
    nearest = np.min(np.hypot(diff_x, diff_y), axis=1)
    return float(np.max(nearest)) if nearest.size else float("nan")


def load_observations() -> dict[str, Any]:
    path = reference_data_dir() / OBSERVATIONS
    if not path.is_file():
        raise SystemExit(f"No observation file at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def gemcitabine_profile(ic50_um: float) -> tuple[Any, dict[str, Any]]:
    """A dosable pharmacology profile at a stated sensitive IC50."""
    engine = ChemistryEngine()
    record = engine.resolve_compound_input("Gemcitabine")
    profile = {
        "ic50_sensitive_um": float(ic50_um),
        "ic50_resistant_um": float(ic50_um) * 4.5,
        "ic50_healthy_um": float(ic50_um) * 12.0,
        "hill_slope": 2.0,
        "emax": PDACModule.emax_default if hasattr(PDACModule, "emax_default") else 3.5e-05,
        "fraction_unbound_plasma": 0.25,
    }
    return record, profile


def run_tissue(ic50_um: float, hours: float, concentration_um: float,
               seed: int, strip_cafs: bool, nx: int = 36, ny: int = 28) -> float:
    """Growth inhibition against an untreated arm with the same seed.

    strip_cafs removes the fibroblast populations before dosing, which is the
    model's analogue of assaying tumour cells without co-cultured CAFs.
    """
    record, profile = gemcitabine_profile(ic50_um)

    def build() -> PDACModule:
        module = PDACModule(seed=seed, nx=nx, ny=ny)
        module.initialize_scenario(seed=seed)
        if strip_cafs:
            module.cells = [c for c in module.cells
                            if c.cell_type not in (CellType.CAF, CellType.FIBROBLAST)]
        return module

    def tumour_count(module: PDACModule) -> int:
        return sum(1 for c in module.cells
                   if c.death_mode == CellDeathMode.ALIVE
                   and c.cell_type in (CellType.TUMOR_SENSITIVE, CellType.TUMOR_RESISTANT))

    dt = 900.0
    steps = max(1, int(round(hours * 3600.0 / dt)))

    treated = build()
    treated.apply_compound_exposure(record, profile, concentration_um)
    control = build()

    for _ in range(steps):
        treated.step(dt)
        control.step(dt)

    treated_n, control_n = tumour_count(treated), tumour_count(control)
    if control_n <= 0:
        return float("nan")
    return 100.0 * (1.0 - treated_n / control_n)


def effective_ic50_for_target(target_tgi_pct: float, hours: float,
                              concentration_um: float, seeds: list[int],
                              strip_cafs: bool, iterations: int = 8,
                              nx: int = 28, ny: int = 22) -> float | None:
    """The IC50 the model needs to produce a given growth inhibition.

    Bisection on log IC50. This is what makes a model-side ratio comparable to a
    measured IC50 ratio: both become "the potency required for the same effect".
    """
    def response(ic50: float) -> float:
        values = [run_tissue(ic50, hours, concentration_um, s, strip_cafs, nx, ny)
                  for s in seeds]
        values = [v for v in values if np.isfinite(v)]
        return float(np.mean(values)) if values else float("nan")

    low, high = 1e-4, 1e3
    r_low, r_high = response(low), response(high)
    if not (np.isfinite(r_low) and np.isfinite(r_high)):
        return None
    # A potent compound gives high inhibition, so response falls as IC50 rises.
    if r_low < target_tgi_pct or r_high > target_tgi_pct:
        return None

    # Each iteration costs one tissue run per seed per arm, and each run is two
    # modules stepped over the whole exposure. Eight iterations narrow the
    # bracket to a factor of about 1.1 on a seven decade range, which is far
    # finer than the observations can distinguish.
    for _ in range(iterations):
        mid = 10.0 ** (0.5 * (np.log10(low) + np.log10(high)))
        value = response(mid)
        if not np.isfinite(value):
            return None
        if value > target_tgi_pct:
            low = mid
        else:
            high = mid
    return 10.0 ** (0.5 * (np.log10(low) + np.log10(high)))


def check_caf_fold(observation: dict[str, Any], seeds: list[int],
                   hours: float, concentration_um: float) -> dict[str, Any]:
    """Does removing CAFs change required potency the way co-culture does?"""
    measured = float(observation["fold"])
    target = 40.0  # a mid-range effect, away from the floor and the ceiling

    with_caf = effective_ic50_for_target(target, hours, concentration_um, seeds, False)
    without_caf = effective_ic50_for_target(target, hours, concentration_um, seeds, True)

    if with_caf is None or without_caf is None or without_caf <= 0:
        return {
            "observation_id": observation["id"],
            "status": "model_could_not_be_driven_to_the_target_effect",
            "measured_fold": measured,
            "model_fold": None,
            "note": ("Bisection did not bracket the target growth inhibition, so "
                     "the model cannot be compared on this observation at these "
                     "settings rather than disagreeing with it."),
        }

    model_fold = with_caf / without_caf
    ratio = max(model_fold, measured) / max(1e-9, min(model_fold, measured))
    return {
        "observation_id": observation["id"],
        "status": "compared",
        "measured_fold": measured,
        "model_fold": round(model_fold, 3),
        "ic50_required_with_caf_um": round(with_caf, 5),
        "ic50_required_without_caf_um": round(without_caf, 5),
        "disagreement_factor": round(ratio, 2),
        "within_tolerance": bool(ratio <= TOLERANCE_FACTOR),
        "tolerance_factor": TOLERANCE_FACTOR,
    }


def check_penetration(observation: dict[str, Any], seed: int = 20261027) -> dict[str, Any]:
    """Does the barrier allow a small molecule to cross dense tissue in time?

    The observation is doxorubicin reaching the core of a 480 um PANC-1 spheroid
    within 12 hours. The model is a vascularised slice rather than a spheroid, so
    this is read as a bound on transport over the equivalent distance: free drug
    at 240 um from the source must not be negligible after 12 hours.
    """
    record, profile = gemcitabine_profile(0.045)
    module = PDACModule(seed=seed, nx=40, ny=30)
    module.initialize_scenario(seed=seed)
    module.apply_compound_exposure(record, profile, 5.0)

    dt = 900.0
    for _ in range(int(round(float(observation["time_to_full_penetration_h"]) * 3600.0 / dt))):
        module.step(dt)

    field = module.grid_drug_free
    peak = float(np.max(field))
    if peak <= 0:
        return {"observation_id": observation["id"], "status": "no_drug_in_the_field"}

    # Distance of each cell from the nearest high-concentration source.
    threshold = 0.5 * peak
    ys, xs = np.nonzero(field >= threshold)
    if len(xs) == 0:
        return {"observation_id": observation["id"], "status": "no_source_region_resolved"}

    half_distance_um = 0.5 * float(observation["spheroid_diameter_um"])
    fraction = _mean_at_distance(field, xs, ys, module.dx, module.dy,
                                 half_distance_um, peak)
    # When no grid cell sits that far from a source the comparison is impossible,
    # and the reason matters: the model's slice is perfused throughout, so there
    # may be no avascular path of that length anywhere in it. Reporting the
    # longest distance available turns an opaque failure into a statement about
    # the geometry mismatch between a vascularised slice and a spheroid.
    furthest = _furthest_distance(field, xs, ys, module.dx, module.dy)
    return {
        "observation_id": observation["id"],
        "status": "compared" if np.isfinite(fraction) else "no_tissue_at_that_distance",
        "beta_ecm": module.beta_ecm,
        "distance_um": half_distance_um,
        "furthest_distance_from_a_source_um": (round(furthest, 1)
                                               if np.isfinite(furthest) else None),
        "geometry_note": (
            None if np.isfinite(fraction) else
            f"No point in the slice is {half_distance_um:.0f} um from the nearest "
            f"drug source; the furthest is {furthest:.0f} um. The model is a "
            f"perfused tumour slice and the observation is an avascular spheroid, "
            f"so this constraint needs either a larger avascular region in the "
            f"model or a spheroid geometry to be testable."
        ),
        "hours": observation["time_to_full_penetration_h"],
        "relative_concentration_at_distance": (round(fraction, 4)
                                               if np.isfinite(fraction) else None),
        "measurement_says": "full penetration to the core",
        # "Full penetration" is not quantified in the source as a percentage, so
        # this reads a tenth of the peak as the floor below which the model would
        # be claiming a barrier the experiment contradicts.
        "model_consistent_with_full_penetration": (bool(fraction >= 0.10)
                                                   if np.isfinite(fraction) else None),
        "floor_used": 0.10,
    }


def check_barrier_effect(hours: tuple[float, ...] = (1.0, 6.0, 12.0, 24.0),
                         betas: tuple[float, ...] = (0.0, 3.2, 12.8, 50.0),
                         seed: int = 20261027) -> dict[str, Any]:
    """Does the desmoplastic transport barrier actually do anything?

    This needs no published observation. It asks whether the model's headline
    mechanism, D_eff = D_0 / (1 + beta * ECM), produces a drug gradient that
    survives the exposure durations the platform actually runs, by comparing the
    free drug field at the default barrier strength against no barrier at all
    and against a much stronger one.

    Measured on the default grid: a gradient exists at one hour, coefficient of
    variation around 0.47, and is essentially gone by twelve, CV 0.049 at the
    default beta of 3.2. Exposures in the published report are 24 hours. Raising
    beta fifteen fold, 3.2 to 50, moves min/max at twelve hours only from 0.835
    to 0.540. The reason is in the ECM field itself: it averages about 0.105 of
    its maximum, so 1 / (1 + 3.2 * 0.105) is a 25% reduction in diffusivity, not
    a barrier.

    So the mechanism is present in the code and quantitatively inert over the
    timescales used, which is a candidate explanation both for the CAF sign
    inversion this harness finds and for the tissue model adding little to the
    QSAR it is handed.
    """
    record, profile = gemcitabine_profile(0.045)
    rows: list[dict[str, Any]] = []

    for beta in betas:
        module = PDACModule(seed=seed, nx=40, ny=30)
        module.beta_ecm = float(beta)
        module.initialize_scenario(seed=seed)
        module.apply_compound_exposure(record, profile, 5.0)

        elapsed = 0.0
        for target in hours:
            while elapsed < target * 3600.0 - 1e-9:
                module.step(900.0)
                elapsed += 900.0
            field = module.grid_drug_free
            mean = float(np.mean(field))
            peak = float(np.max(field))
            rows.append({
                "beta_ecm": float(beta),
                "hours": float(target),
                "drug_min": round(float(np.min(field)), 5),
                "drug_max": round(peak, 5),
                "drug_mean": round(mean, 5),
                "coefficient_of_variation": (round(float(np.std(field)) / mean, 4)
                                             if mean > 0 else None),
                "min_over_max": (round(float(np.min(field)) / peak, 4)
                                 if peak > 0 else None),
                "ecm_mean": round(float(np.mean(module.grid_ecm)), 4),
            })

    def at(beta: float, hour: float) -> dict[str, Any] | None:
        for row in rows:
            if row["beta_ecm"] == beta and row["hours"] == hour:
                return row
        return None

    default_long = at(3.2, max(hours))
    none_long = at(0.0, max(hours))
    strong_long = at(max(betas), max(hours))

    gradient_survives = (default_long is not None
                         and (default_long["coefficient_of_variation"] or 0.0) > 0.25)
    barrier_separates = (default_long is not None and none_long is not None
                         and abs((default_long["min_over_max"] or 0.0)
                                 - (none_long["min_over_max"] or 0.0)) > 0.1)

    return {
        "check": "does the transport barrier produce a lasting gradient",
        "needs_no_published_observation": True,
        "rows": rows,
        "gradient_survives_the_exposure": bool(gradient_survives),
        "default_barrier_separates_from_no_barrier": bool(barrier_separates),
        "finding": (
            "The barrier produces a lasting gradient and changes the field."
            if gradient_survives and barrier_separates else
            "The drug field equilibrates to near uniform within the exposure "
            "durations the platform runs, and the default barrier strength is "
            "close to indistinguishable from no barrier at all. The mechanism is "
            "present in the code and quantitatively inert here. The ECM field "
            "averages a small fraction of its maximum, so the attenuation term "
            "reduces diffusivity modestly rather than blocking transport."
        ),
        "why_it_matters": (
            "Desmoplastic protection is the model's headline mechanism and the "
            "reason it claims to add something a ligand-based score cannot. If it "
            "does not bite, simulated growth inhibition is a restatement of the "
            "IC50 handed to it, which is what the failing positive control shows."
        ),
    }


def sweep_beta_ecm(values: list[float], observation: dict[str, Any]) -> list[dict[str, Any]]:
    """Which barrier strengths are consistent with the penetration measurement?"""
    rows = []
    original = PDACModule.beta_ecm if hasattr(PDACModule, "beta_ecm") else None
    for value in values:
        module_beta = value

        def patched_check() -> dict[str, Any]:
            record, profile = gemcitabine_profile(0.045)
            module = PDACModule(seed=20261027, nx=40, ny=30)
            module.beta_ecm = module_beta
            module.initialize_scenario(seed=20261027)
            module.apply_compound_exposure(record, profile, 5.0)
            for _ in range(int(round(12.0 * 3600.0 / 900.0))):
                module.step(900.0)
            field = module.grid_drug_free
            peak = float(np.max(field))
            if peak <= 0:
                return {"relative": None}
            threshold = 0.5 * peak
            ys, xs = np.nonzero(field >= threshold)
            if len(xs) == 0:
                return {"relative": None}
            value = _mean_at_distance(field, xs, ys, module.dx, module.dy,
                                      240.0, peak)
            return {"relative": None if not np.isfinite(value) else value}

        result = patched_check()
        relative = result["relative"]
        rows.append({
            "beta_ecm": value,
            "relative_concentration_at_240um": (round(relative, 4)
                                                if relative is not None else None),
            "consistent": (bool(relative >= 0.10) if relative is not None else None),
        })
        print(f"    beta_ecm {value:5.2f} -> relative {relative if relative is None else round(relative,4)}")
    _ = original
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare the PDAC tissue model against published measurements.")
    parser.add_argument("--hours", type=float, default=24.0)
    parser.add_argument("--concentration-um", type=float, default=5.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20261027, 20261028])
    parser.add_argument("--sweep-beta-ecm", action="store_true",
                        help="Sweep the ECM attenuation coefficient against the "
                             "penetration constraint.")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reports" / "tissue_calibration.json")
    args = parser.parse_args()

    payload = load_observations()
    observations = payload["observations"]
    unverified = [o for o in observations if o.get("verification") != "verified_primary"]

    print("=" * 78)
    print("  PDAC tissue model against published measurements")
    print("=" * 78)
    print(f"\n  observations loaded : {len(observations)}")
    print(f"  unverified          : {len(unverified)} of {len(observations)}")

    inexpressible = [o for o in observations if o.get("model_can_express") == "no"]
    if inexpressible:
        print(f"\n  The model has no mechanism for {len(inexpressible)} of these:")
        for o in inexpressible:
            print(f"    {o['id']}")
            print(f"      {' '.join(o['model_mapping_note'])[:150]}")

    results: list[dict[str, Any]] = []

    caf = [o for o in observations if o["kind"] == "fold_resistance_caf_coculture"
           and o.get("model_can_express") == "yes"]
    for observation in caf:
        print(f"\n  CAF co-culture: {observation['id']}")
        print(f"    measured fold {observation['fold']}, driving the model to 40% inhibition ...")
        result = check_caf_fold(observation, args.seeds, args.hours, args.concentration_um)
        results.append(result)
        if result["status"] == "compared":
            print(f"    model fold {result['model_fold']}  "
                  f"disagreement factor {result['disagreement_factor']}  "
                  f"within tolerance: {result['within_tolerance']}")
        else:
            print(f"    {result['status']}")

    penetration = [o for o in observations if o["kind"] == "penetration_constraint"]
    for observation in penetration:
        print(f"\n  Penetration constraint: {observation['id']}")
        result = check_penetration(observation)
        results.append(result)
        if result.get("status") == "compared":
            print(f"    beta_ecm {result['beta_ecm']}, relative concentration at "
                  f"{result['distance_um']} um after {result['hours']} h: "
                  f"{result['relative_concentration_at_distance']}")
            print(f"    consistent with full penetration: "
                  f"{result['model_consistent_with_full_penetration']}")
        else:
            print(f"    {result.get('status')}")
            if result.get("geometry_note"):
                print(f"    {result['geometry_note']}")

    print("\n  Barrier effect, needs no published observation:")
    barrier = check_barrier_effect()
    for row in barrier["rows"]:
        if row["hours"] in (1.0, max(r["hours"] for r in barrier["rows"])):
            print(f"    beta {row['beta_ecm']:5.1f} at {row['hours']:5.1f} h: "
                  f"CV {row['coefficient_of_variation']}  "
                  f"min/max {row['min_over_max']}  ECM mean {row['ecm_mean']}")
    print(f"    gradient survives the exposure      : "
          f"{barrier['gradient_survives_the_exposure']}")
    print(f"    default barrier differs from none   : "
          f"{barrier['default_barrier_separates_from_no_barrier']}")
    print(f"    {barrier['finding']}")

    sweep: list[dict[str, Any]] = []
    if args.sweep_beta_ecm and penetration:
        print("\n  Sweeping beta_ecm against the penetration constraint:")
        sweep = sweep_beta_ecm([0.0, 1.6, 3.2, 6.4, 12.8], penetration[0])

    compared = [r for r in results if r.get("status") == "compared"]
    agreeing = [r for r in compared
                if r.get("within_tolerance") is True
                or r.get("model_consistent_with_full_penetration") is True]

    print("\n" + "=" * 78)
    print(f"  compared : {len(compared)} of {len(observations)} observations")
    print(f"  agreeing : {len(agreeing)} of {len(compared)} compared")

    if unverified:
        verdict = (
            f"NOT A CALIBRATION. {len(unverified)} of {len(observations)} observations are "
            f"unverified, transcribed from search engine summaries rather than read in the "
            f"cited papers. The comparison above exercises the harness and shows what the "
            f"model does; it does not license any claim about agreement with the "
            f"literature. Promote records to verified_primary in "
            f"data/reference/{OBSERVATIONS} first."
        )
    elif len(agreeing) == len(compared) and compared:
        verdict = ("Every comparable observation agrees within the stated tolerance. "
                   "That is a weak result on this few observations, not a fitted model.")
    else:
        verdict = (f"{len(compared) - len(agreeing)} of {len(compared)} comparable "
                   f"observations disagree with the model. Those are the parameters to "
                   f"revisit, and the barrier mechanism is the first candidate.")

    print(f"\n  {verdict}")
    print("\n  Nothing here fits a parameter. Four observations against five free")
    print("  parameters would be overfitting; the sweep reports consistent ranges.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "observation_file": str(reference_data_dir() / OBSERVATIONS),
        "settings": {"hours": args.hours, "concentration_um": args.concentration_um,
                     "seeds": args.seeds, "tolerance_factor": TOLERANCE_FACTOR},
        "observations_total": len(observations),
        "observations_unverified": len(unverified),
        "observations_model_cannot_express": [o["id"] for o in inexpressible],
        "results": results,
        "barrier_effect": barrier,
        "beta_ecm_sweep": sweep,
        "compared": len(compared),
        "agreeing": len(agreeing),
        "is_a_calibration": not unverified,
        "verdict": verdict,
        "contested_mechanism": payload.get("contested_mechanism"),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\n  Written to {args.output}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
