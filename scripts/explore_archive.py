"""Query and re-score the archive of structures a search evaluated.

The archive exists because re-scoring is cheap and searching is not: re-ranking
2093 archived structures took 99 seconds against 6834 to produce them, a factor
of 69. Expanding the training set invalidates every score, so after retraining
you re-rank the archive rather than repeating the search.

    python scripts/explore_archive.py --summary
    python scripts/explore_archive.py --top 20
    python scripts/explore_archive.py --top 20 --exclude-alerts --unselected
    python scripts/explore_archive.py --rescore
    python scripts/explore_archive.py --export reports/search_trace.json

Read what comes back as a log of what an algorithm tried. It is not a shortlist:
the scores come from a potency model whose 95% conformal interval is a factor of
about 1330, which is far wider than the differences between rows.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulation.core.structure_archive import EXPORT_CAVEAT, StructureArchive


def banner() -> None:
    print("=" * 78)
    print("EVOCURE search trace archive")
    print("=" * 78)


def show_summary(archive: StructureArchive) -> None:
    summary = archive.summary()
    if not summary["total"]:
        print("  The archive is empty. Run a discovery experiment first.")
        return

    print(f"  Structures archived          : {summary['total']}")
    print(f"  Distinct Murcko scaffolds    : {summary['unique_scaffolds']}")
    print(f"  Reported as candidates       : {summary['selected_for_reporting']} "
          f"({100 * summary['reported_fraction']:.2f}% of the archive)")
    print(f"  Seed compounds               : {summary['seed_compounds']}")
    print(f"  Contributing experiments     : {summary['contributing_experiments']}")
    print(f"  With a medchem assessment    : {summary['with_medchem_assessment']}")

    if summary["rejection_reasons"]:
        print("\n  Why structures did not reach a report:")
        for reason, count in summary["rejection_reasons"].items():
            print(f"    {count:6d}  {reason}")


def show_rows(archive: StructureArchive, args) -> None:
    rows = archive.query(limit=args.top, min_utility=args.min_utility,
                         exclude_alerts=args.exclude_alerts,
                         only_unselected=args.unselected)
    if not rows:
        print("  Nothing matches those filters.")
        return

    print(f"\n  {'utility':>8s} {'pIC50':>7s} {'gen':>4s} {'sel':>4s}  structure")
    for row in rows:
        utility = row["composite_utility"]
        pic50 = row["pic50"]
        print(f"  {utility if utility is not None else float('nan'):>8.4f} "
              f"{pic50 if pic50 is not None else float('nan'):>7.2f} "
              f"{row['origin_generation'] or 0:>4d} "
              f"{'yes' if row['selected'] else 'no':>4s}  "
              f"{row['canonical_smiles'][:62]}")
        if row["not_selected_because"]:
            print(f"           rejected: {row['not_selected_because']}")


def do_rescore(archive: StructureArchive) -> int:
    from ai.predictors.qsar import CalibratedQSARPredictor
    import hashlib
    import math

    model_path = ROOT / "models" / "scientific" / "pdac_qsar_v1.joblib"
    if not model_path.is_file():
        print("  No QSAR model found. Build one with: python scripts/train_qsar.py",
              file=sys.stderr)
        return 1

    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()[:12]
    predictor = CalibratedQSARPredictor().load()
    print(f"  Re-scoring against model {digest} ...")

    def score(smiles: str):
        try:
            prediction = predictor.predict(smiles)
        except Exception:
            return None
        value = (prediction.get("pic50") or {}).get("value")
        if not isinstance(value, (int, float)):
            return None
        # Same sigmoidal potency utility the pipeline uses, centred at pIC50 6.
        potency = 1.0 / (1.0 + math.exp(-(float(value) - 6.0)))
        return {
            "composite_utility": round(potency, 4),
            "pic50": round(float(value), 4),
            "ic50_um": prediction.get("ic50_um_estimated"),
            "objectives": {"predicted_pdac_activity": round(potency, 4)},
        }

    # Potency only, while the archived scores came from the full objective set, so
    # the churn number must not be read as model drift.
    outcome = archive.rescore(score, new_model_version=digest,
                              comparable_objective=False).to_dict()
    print(f"\n  Archived            : {outcome['total']}")
    print(f"  Re-scored           : {outcome['rescored']}")
    print(f"  Improved / worsened : {outcome['improved']} / {outcome['worsened']}")
    print(f"  Previous model      : {outcome['previous_model']}")
    print(f"  New model           : {outcome['new_model']}")
    print(f"  Top 50 churn        : {outcome['rank_churn_top_50']} of 50")
    print(f"\n  {outcome['interpretation']}")
    print("\n  Note: this re-scores on predicted potency alone. A full re-rank also needs")
    print("  toxicology, drug likeness and the tissue objective, which the pipeline computes.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Explore and re-score the search trace archive",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--summary", action="store_true", help="what the archive holds")
    parser.add_argument("--top", type=int, default=0, help="show the N highest scoring rows")
    parser.add_argument("--min-utility", type=float, default=None)
    parser.add_argument("--exclude-alerts", action="store_true",
                        help="hide structures with medicinal chemistry alerts")
    parser.add_argument("--unselected", action="store_true",
                        help="only structures no report ever showed")
    parser.add_argument("--rescore", action="store_true",
                        help="re-score every row against the current QSAR model")
    parser.add_argument("--export", default=None, help="write a JSON export")
    parser.add_argument("--db", default=None, help="archive path")
    args = parser.parse_args()

    archive = StructureArchive(db_path=args.db)
    banner()

    if args.rescore:
        code = do_rescore(archive)
        if code:
            return code
        print()

    if args.summary or not any((args.top, args.rescore, args.export)):
        show_summary(archive)

    if args.top:
        show_rows(archive, args)

    if args.export:
        info = archive.export(args.export, limit=max(args.top, 1000))
        print(f"\n  Exported {info['rows']} rows to {info['path']}")

    print("\n" + "-" * 78)
    print("  " + EXPORT_CAVEAT.replace(". ", ".\n  "))
    print("-" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
