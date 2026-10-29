#!/usr/bin/env python3
"""Does expanding the training set actually improve the potency model?

The project's first priority is dataset expansion, and `train_qsar.py` now runs
on an imported set. But the two runs it prints cannot be compared directly, and
reading them as a before and after would be wrong in both directions.

Measured on PANC-1 IC50 from ChEMBL_37, 2106 usable records over 919 scaffolds
against the curated 43 over 32:

                    curated      expanded
    RMSE             1.3397        0.8524     better
    conformal q95    3.1237        1.7814     better
    R2               0.6987        0.3733     worse
    ROC AUC          0.9535        0.6945     worse
    Brier            0.0999        0.2126     worse

Those metrics are computed on different populations, so they are not a before
and after of the same quantity. The curated set was assembled to span potency,
pIC50 1.22 to 8.30 with variance 5.96. The ChEMBL set concentrates around
pIC50 5.46 with variance 1.16. Since R2 = 1 - MSE / variance, a fivefold
smaller denominator lowers R2 even as absolute error falls; 1 - 1.3397^2/5.96
is 0.699 and 1 - 0.8524^2/1.16 is 0.373, which reproduces both reported values.
The ROC AUC fall has the same shape: the active label is IC50 below 10 uM, which
cuts a curated bimodal distribution cleanly and falls inside the bulk of the
ChEMBL one.

So this script does the comparison those numbers cannot. Both models are trained
with the held-out scaffolds removed from each, and both are then scored on the
same curated compounds, with the same targets. A paired difference on an
identical test set is a comparison; two numbers from two populations is not.

What it still cannot tell you. The test set is curated compounds, so it measures
transfer onto curated chemistry and not onto whatever you plan to screen. The
held-out sets are small, which is the original problem restated, so the per seed
spread is reported rather than a single mean. And nothing here says the model
measures potency: whatever the interval becomes, it stays an ordering signal
across orders of magnitude in IC50.

    python scripts/fetch_chembl_bioactivity.py --cell-line PANC-1 --endpoint IC50
    python scripts/import_bioactivity.py data/raw/chembl_PANC-1_IC50.csv \\
        --endpoint IC50 --cell-line PANC-1 --disease PDAC --merge-into-curated \\
        --output data/processed/pdac_expanded_dataset.json
    python scripts/measure_dataset_expansion.py
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

from ai.predictors.data_connectors import ScientificDataManager
from ai.predictors.qsar import CalibratedQSARPredictor

RDLogger.DisableLog("rdApp.*")

CURATED = ROOT / "data" / "reference" / "pdac_reference_dataset.json"
HOLDOUT_FRACTION = 0.3
SEEDS = (0, 1, 2, 3, 4)


def load(path: Path) -> list[dict]:
    """PDAC-partition records with a usable pIC50, as the trainer sees them."""
    manager = ScientificDataManager(reference_dir=path.parent)
    manager.load_reference_dataset(filename=path.name, disease_id="PDAC")
    rows = []
    for record in manager.get_records():
        assay = record.get("bioactivity") or {}
        smiles = record.get("canonical_smiles")
        if not smiles or assay.get("pic50") is None:
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        rows.append({
            "smiles": smiles,
            "pic50": float(assay["pic50"]),
            "active": int(bool(assay.get("growth_inhibition_active"))),
            "scaffold": MurckoScaffold.MurckoScaffoldSmiles(mol=mol),
        })
    return rows


def fit_and_score(train: list[dict], test: list[dict]) -> dict[str, float] | None:
    """Train with no internal CV, then score on the given rows.

    evaluate_split is off because the point here is one fixed external test set,
    not another internal estimate.
    """
    if len(train) < 10 or not test:
        return None

    model = CalibratedQSARPredictor(random_state=42)
    model.fit([r["smiles"] for r in train], [r["pic50"] for r in train],
              [r["active"] for r in train], evaluate_split=False)

    truth, predicted = [], []
    for row in test:
        features = model.featurize(row["smiles"])
        if features is None:
            continue
        truth.append(row["pic50"])
        predicted.append(float(model.regressor.predict(features.reshape(1, -1))[0]))

    if len(truth) < 3:
        return None

    residuals = np.abs(np.array(truth) - np.array(predicted))
    n = len(residuals)
    # Same split-conformal rule the predictor uses, on this test set.
    k = min(n, max(1, int(np.ceil((n + 1) * 0.95))))
    return {
        "train_compounds": len(train),
        "test_compounds": n,
        "rmse": float(np.sqrt(np.mean(residuals ** 2))),
        "mae": float(np.mean(residuals)),
        "conformal_q95": float(np.sort(residuals)[k - 1]),
    }


def one_seed(curated: list[dict], expanded: list[dict], seed: int) -> dict | None:
    """One paired trial on a shared scaffold holdout.

    The holdout is drawn from scaffolds present in the curated set, because the
    test rows have to be curated rows for both models to be scored on identical
    targets. Those scaffolds are removed from both training sets, so neither
    model has seen the chemistry it is tested on.
    """
    rng = np.random.default_rng(seed)
    curated_scaffolds = sorted({r["scaffold"] for r in curated})
    count = max(2, int(round(len(curated_scaffolds) * HOLDOUT_FRACTION)))
    held = set(rng.choice(curated_scaffolds, size=count, replace=False).tolist())

    test = [r for r in curated if r["scaffold"] in held]
    curated_train = [r for r in curated if r["scaffold"] not in held]
    expanded_train = [r for r in expanded if r["scaffold"] not in held]

    base = fit_and_score(curated_train, test)
    wide = fit_and_score(expanded_train, test)
    if base is None or wide is None:
        return None

    return {
        "seed": seed,
        "held_out_scaffolds": len(held),
        "test_compounds": base["test_compounds"],
        "curated": base,
        "expanded": wide,
        "rmse_delta": round(wide["rmse"] - base["rmse"], 4),
        "q95_delta": round(wide["conformal_q95"] - base["conformal_q95"], 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare a curated and an expanded training set on one test set.")
    parser.add_argument("--expanded", type=Path,
                        default=ROOT / "data" / "processed" / "pdac_expanded_dataset.json",
                        help="Imported dataset to compare against the curated one.")
    args = parser.parse_args()

    if not args.expanded.is_file():
        print(f"No expanded dataset at {args.expanded}.\n"
              f"Build one with scripts/fetch_chembl_bioactivity.py and "
              f"scripts/import_bioactivity.py first.", file=sys.stderr)
        return 1

    curated = load(CURATED)
    expanded = load(args.expanded)
    print("=" * 78)
    print("  Does expanding the training set improve the potency model?")
    print("=" * 78)
    print(f"\n  curated  : {len(curated):5d} compounds, "
          f"{len({r['scaffold'] for r in curated}):4d} scaffolds, "
          f"pIC50 variance {np.var([r['pic50'] for r in curated]):.3f}")
    print(f"  expanded : {len(expanded):5d} compounds, "
          f"{len({r['scaffold'] for r in expanded}):4d} scaffolds, "
          f"pIC50 variance {np.var([r['pic50'] for r in expanded]):.3f}")
    print("\n  Both trained with the held-out scaffolds removed, both scored on the")
    print("  same curated compounds. Lower is better for all three columns.\n")

    header = (f"{'seed':>5} {'test':>5} {'RMSE cur':>9} {'RMSE exp':>9} "
              f"{'q95 cur':>8} {'q95 exp':>8}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    trials = []
    for seed in SEEDS:
        trial = one_seed(curated, expanded, seed)
        if trial is None:
            print(f"{seed:>5}   skipped, not enough usable rows")
            continue
        trials.append(trial)
        print(f"{seed:>5} {trial['test_compounds']:>5} "
              f"{trial['curated']['rmse']:>9.4f} {trial['expanded']['rmse']:>9.4f} "
              f"{trial['curated']['conformal_q95']:>8.4f} "
              f"{trial['expanded']['conformal_q95']:>8.4f}")

    if not trials:
        print("\n  No usable trials.")
        return 1

    rmse_deltas = [t["rmse_delta"] for t in trials]
    q95_deltas = [t["q95_delta"] for t in trials]
    improved = sum(1 for d in rmse_deltas if d < 0)

    print(f"\n  RMSE change      : {np.mean(rmse_deltas):+.4f} log units mean, "
          f"range {min(rmse_deltas):+.4f} to {max(rmse_deltas):+.4f}")
    print(f"  q95 change       : {np.mean(q95_deltas):+.4f} log units mean, "
          f"range {min(q95_deltas):+.4f} to {max(q95_deltas):+.4f}")
    print(f"  Seeds improved   : {improved} of {len(trials)}")

    if improved == len(trials):
        verdict = ("The expanded set gave lower error on every seed. On an identical "
                   "test set this is a real improvement, not a change of population.")
    elif improved == 0:
        verdict = ("The expanded set was worse on every seed. More data from a "
                   "different assay population did not transfer onto curated "
                   "chemistry.")
    elif np.mean(rmse_deltas) < 0:
        verdict = (f"Better on {improved} of {len(trials)} seeds and better on "
                   f"average, but not consistently. On test sets this small that is "
                   f"a weak result, and the spread is the original problem restated.")
    else:
        verdict = (f"Better on only {improved} of {len(trials)} seeds and worse on "
                   f"average. The expansion does not establish an improvement here.")

    print(f"\n  {verdict}")
    print("\n  Whatever the interval became, it stays orders of magnitude wide in")
    print("  IC50. This model orders compounds; it does not measure them.")

    destination = ROOT / "reports" / "dataset_expansion_comparison.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({
        "question": ("Does training on the imported set beat the curated set, "
                     "measured on one identical scaffold holdout?"),
        "curated_dataset": str(CURATED.relative_to(ROOT)),
        "expanded_dataset": str(args.expanded),
        "curated_compounds": len(curated),
        "curated_scaffolds": len({r["scaffold"] for r in curated}),
        "curated_target_variance": round(float(np.var([r["pic50"] for r in curated])), 4),
        "expanded_compounds": len(expanded),
        "expanded_scaffolds": len({r["scaffold"] for r in expanded}),
        "expanded_target_variance": round(float(np.var([r["pic50"] for r in expanded])), 4),
        "holdout_fraction": HOLDOUT_FRACTION,
        "trials": trials,
        "rmse_delta_mean": round(float(np.mean(rmse_deltas)), 4),
        "q95_delta_mean": round(float(np.mean(q95_deltas)), 4),
        "seeds_improved": improved,
        "seeds_total": len(trials),
        "verdict": verdict,
        "why_the_trainer_metrics_are_not_comparable": (
            "train_qsar.py reports R2 and ROC AUC on whichever dataset it trained "
            "on. R2 = 1 - MSE / target variance, and the curated set was built to "
            "span potency (variance 5.96) while the ChEMBL set concentrates "
            "(variance 1.16), so R2 falls on the wider set even as absolute error "
            "improves. The active label, IC50 below 10 uM, likewise cuts the "
            "curated bimodal distribution cleanly and lands inside the bulk of the "
            "ChEMBL one, which is why ROC AUC falls. Neither is evidence about "
            "model quality across the two runs."
        ),
        "caveat": (
            "The test rows are curated compounds, so this measures transfer onto "
            "curated chemistry, not onto arbitrary screening libraries. The "
            "holdouts are small, which is the original data limitation restated, "
            "so per seed results are reported rather than a single figure."
        ),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\n  Written to {destination}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
