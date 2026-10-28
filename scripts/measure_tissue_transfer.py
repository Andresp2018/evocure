"""How much does the tissue simulation add beyond the IC50 it is handed?

When a candidate's potency comes from a model rather than a measurement, the
tissue simulation can only transform that model's output. This script measures
the transformation directly: it fixes everything except the assigned IC50 and
sweeps it, so the resulting curve is the tissue model's entire contribution to
discriminating between compounds at a given dose.

Read it next to two other numbers from the same run:

  the spread of IC50 across the published candidates, and
  the QSAR's conformal interval

If the candidates' IC50 spread is small compared to the interval, the tissue
model is ranking compounds on differences far smaller than the uncertainty in
any one of them, whatever the p values say.

    python scripts/measure_tissue_transfer.py

Writes reports/tissue_transfer_function.json.
"""

from __future__ import annotations

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from simulation.biology.registry import DiseaseRegistry
from simulation.chemistry.engine import ChemistryEngine
from simulation.core.engine import EVOCUREEngine

OUT = ROOT / "reports" / "tissue_transfer_function.json"
RESULTS = ROOT / "reports" / "pdac_publication_test" / "experiment_results.json"

SEED = 20261027
SEEDS = [20261027, 20261028, 20261029]
DOSE_UM = 5.0
DURATION_H = 24.0
GRID = {"nx": 36, "ny": 28}
IC50_SWEEP = [0.01, 0.03, 0.1, 0.3, 1.0, 2.0, 5.0, 10.0, 30.0, 100.0]


def main() -> int:
    engine = EVOCUREEngine(seed=SEED)
    engine.set_active_disease("PDAC", seed=SEED)
    carrier = ChemistryEngine().resolve_compound_input("Gemcitabine")
    base, _ = engine._tissue_pharmacology(carrier)

    print("=" * 74)
    print("Tissue model response as a function of the IC50 it is given")
    print("=" * 74)
    print(f"  Dose fixed at {DOSE_UM} uM, exposure {DURATION_H} h, "
          f"{len(SEEDS)} seeds, grid {GRID['nx']}x{GRID['ny']}")
    print(f"\n  {'IC50 (uM)':>11s} {'TGI mean %':>12s} {'TGI sd':>9s}")

    curve = []
    for ic50 in IC50_SWEEP:
        profile = dict(base)
        profile["ic50_sensitive_um"] = ic50
        profile["ic50_resistant_um"] = ic50 * 4.5
        profile["ic50_healthy_um"] = ic50 * 12.0

        values = []
        for seed in SEEDS:
            sim = DiseaseRegistry().create("PDAC", seed=seed, **GRID)
            values.append(sim.run_controlled_comparison(
                compound=carrier, pharmacology_profile=profile,
                applied_concentration_um=DOSE_UM, duration_hours=DURATION_H, seed=seed,
            )["deltas"]["tumor_growth_inhibition_pct"])

        entry = {"ic50_um": ic50,
                 "tgi_mean_pct": round(float(np.mean(values)), 2),
                 "tgi_sd_pct": round(float(np.std(values, ddof=1)), 2),
                 "tgi_per_seed": [round(v, 2) for v in values]}
        curve.append(entry)
        print(f"  {ic50:>11.3f} {entry['tgi_mean_pct']:>12.2f} {entry['tgi_sd_pct']:>9.2f}")

    report = {
        "dose_um": DOSE_UM, "duration_hours": DURATION_H, "seeds": SEEDS, "grid": GRID,
        "curve": curve,
        "method": (
            "Every pharmacology parameter except the IC50 triple is held fixed at the "
            "profile of a single carrier compound, so the curve isolates the tissue "
            "model's response to assigned potency alone."
        ),
    }

    # Put the curve next to the published candidates, if there are any.
    if RESULTS.is_file():
        published = json.loads(RESULTS.read_text(encoding="utf-8"))
        ic50s = []
        for candidate in published.get("candidates", []):
            value = (candidate.get("tissue_model_info") or {}).get("ic50_sensitive_um")
            if isinstance(value, (int, float)):
                ic50s.append(float(value))
        quantile = (published.get("qsar_validation") or {}).get("conformal_q_95")

        if ic50s:
            spread_fold = max(ic50s) / max(1e-12, min(ic50s))
            report["published_candidates"] = {
                "n": len(ic50s),
                "ic50_min_um": round(min(ic50s), 4),
                "ic50_max_um": round(max(ic50s), 4),
                "spread_fold": round(spread_fold, 2),
                "potency_provenance": sorted({
                    c.get("potency_provenance") for c in published.get("candidates", [])
                    if c.get("potency_provenance")
                }),
            }
            print("\n" + "-" * 74)
            print(f"  Published candidates span IC50 {min(ic50s):.4f} to {max(ic50s):.4f} uM, "
                  f"a factor of {spread_fold:.1f}.")

            if isinstance(quantile, (int, float)):
                interval_fold = 10.0 ** float(quantile)
                report["qsar_conformal_q_95_log_units"] = quantile
                report["qsar_interval_fold"] = round(interval_fold, 1)
                report["spread_vs_uncertainty"] = (
                    "The candidates are separated by less than the uncertainty in any one "
                    "of their potencies."
                    if spread_fold < interval_fold else
                    "The candidate spread exceeds the single-compound uncertainty."
                )
                print(f"  The QSAR's 95% conformal interval is a factor of {interval_fold:,.0f}.")
                print(f"  {report['spread_vs_uncertainty']}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWritten to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
