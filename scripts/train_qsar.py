"""Rebuild the PDAC QSAR artifact from the curated reference dataset.

No model weights ship with this repository. This script regenerates
models/scientific/pdac_qsar_v1.joblib from data/reference/pdac_reference_dataset.json,
which is version controlled. Training takes a few seconds on a CPU and needs no GPU.

    python scripts/train_qsar.py

The random seed is fixed, so two runs of the same scikit-learn and RDKit versions
produce the same model. Different library versions can shift the numbers slightly;
the script prints the validation metrics so you can check what you got.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.predictors.data_connectors import ScientificDataManager
from ai.predictors.qsar import CalibratedQSARPredictor


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the EVOCURE PDAC QSAR model")
    parser.add_argument("--disease", default="PDAC",
                        help="Disease partition to train on (default: PDAC)")
    parser.add_argument("--output", default=None,
                        help="Destination joblib path (default: models/scientific/pdac_qsar_v1.joblib)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--dataset", default=None,
                        help="Dataset JSON to train on. Default is the curated "
                             "data/reference/pdac_reference_dataset.json. Point this at "
                             "the output of scripts/import_bioactivity.py to train on an "
                             "expanded set; without it the expanded file is written and "
                             "never read.")
    args = parser.parse_args()

    print("=" * 78)
    print("EVOCURE QSAR training")
    print("=" * 78)

    if args.dataset:
        dataset_path = Path(args.dataset).expanduser()
        if not dataset_path.is_file():
            print(f"\nERROR: no dataset at {dataset_path}", file=sys.stderr)
            return 1
        manager = ScientificDataManager(reference_dir=dataset_path.parent)
        records = manager.load_reference_dataset(filename=dataset_path.name,
                                                 disease_id=args.disease)
        print(f"  Dataset file   : {dataset_path}")
    else:
        manager = ScientificDataManager()
        records = manager.load_reference_dataset(disease_id=args.disease)
    print(f"  Dataset        : {manager.provenance.source_name} v{manager.provenance.version}")
    print(f"  Partition      : {args.disease}")
    print(f"  Compounds      : {len(records)}")

    raw = manager.get_records()
    smiles, pic50, active, diseases = [], [], [], []
    for row in raw:
        assay = row.get("bioactivity") or {}
        value = assay.get("pic50")
        smi = row.get("canonical_smiles")
        if smi is None or value is None:
            continue
        smiles.append(smi)
        pic50.append(float(value))
        active.append(int(bool(assay.get("growth_inhibition_active"))))
        diseases.append(str(assay.get("disease_id") or row.get("disease_id") or args.disease))

    if len(smiles) < 10:
        print(f"\nERROR: only {len(smiles)} usable records; refusing to train.", file=sys.stderr)
        return 1

    print(f"  Usable records : {len(smiles)} ({sum(active)} active, {len(active) - sum(active)} inactive)")
    print("\n  Fitting random forest regressor and calibrated classifier ...")

    predictor = CalibratedQSARPredictor(random_state=args.seed)
    metrics = predictor.fit(smiles, pic50, active, evaluate_split=True, disease_list=diseases)

    destination = Path(args.output) if args.output else predictor.DEFAULT_MODEL_PATH
    saved = predictor.save(destination)

    validation = metrics.get("validation_metrics", metrics)
    print("\n  Bemis-Murcko scaffold validation (no scaffold leakage):")
    for key in ("validation_scheme", "independent_scaffolds", "train_compounds",
                "test_compounds", "rmse", "r2", "roc_auc", "brier_score",
                "conformal_q_95"):
        if key in validation:
            print(f"    {key:22s} {validation[key]}")

    print(f"\n  Saved to       : {saved}")
    print(f"  Size           : {saved.stat().st_size / 1024:.1f} KB")

    r2 = validation.get("r2")
    quantile = validation.get("conformal_q_95")
    print("\n  Read these numbers before trusting any potency value this model emits.")
    if isinstance(r2, (int, float)):
        if r2 < 0.5:
            print(f"    The pIC50 regressor explains little variance (R2 = {r2}). Treat predicted")
            print("    potency as a coarse ordering signal, not a measurement.")
        else:
            print(f"    The regressor carries real ordering signal (R2 = {r2}), but see the")
            print("    interval below before reading any single value as a quantity.")
    if isinstance(quantile, (int, float)) and quantile > 1.5:
        print(f"    The 95% conformal interval is +/- {quantile} log units, still orders of")
        print("    magnitude in IC50. Rank ordering is far more defensible than point values.")
    print("    The active/inactive classifier is the stronger half of this model.")
    print("    These come from repeated grouped CV. A single scaffold split on 43 compounds")
    print("    returns anything from R2 -0.09 to 0.82 depending only on the seed.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
