"""Import external bioactivity data into the curated dataset, safely.

The potency model is data limited: its learning curve is still descending at 32
scaffolds, and its 95% conformal interval is a factor of about 1330. More
assay-homogeneous data is the single largest improvement available.

This tool does the import without the failure mode that would make things worse,
which is pooling measurements that are not comparable. You name the assay context
you want; everything outside it is rejected and counted, so you can see what a
file actually contained.

No network access. Obtain the data yourself, for example:

  ChEMBL   a target or cell line activity export as CSV
  GDSC     the bulk IC50 download, filtered to your cell line
  Your own plate reader output, with the columns below

Expected columns, with common aliases accepted:

  canonical_smiles | smiles          required
  standard_value   | value | ic50    required unless pic50 is present
  standard_units   | units           required with standard_value
  standard_type    | measurement_type  for example IC50
  cell_line                            for example PANC-1
  standard_relation                    "=" only; bounds are rejected
  name | molecule_chembl_id            optional

Examples:

  # See what is in a file before importing anything
  python scripts/import_bioactivity.py data/raw/chembl_panc1.csv --inspect

  # Import one assay group, writing a new dataset next to the curated one
  python scripts/import_bioactivity.py data/raw/chembl_panc1.csv \\
      --endpoint IC50 --cell-line PANC-1 --disease PDAC \\
      --output data/reference/pdac_expanded_dataset.json

  # Then retrain and see whether the conformal interval narrowed
  python scripts/train_qsar.py
"""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ai.predictors.bioactivity_import import (
    AssayContext,
    merge_into_dataset,
    normalise_records,
    scaffold_summary,
)

CURATED = ROOT / "data" / "reference" / "pdac_reference_dataset.json"


def read_rows(path: Path) -> list[dict]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("compounds") or payload.get("data") or [payload]
        return list(payload)
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def cmd_inspect(rows: list[dict]) -> int:
    print("=" * 76)
    print(f"Inspecting {len(rows)} rows. Nothing is imported.")
    print("=" * 76)

    if not rows:
        print("  The file is empty.")
        return 1

    print(f"\n  Columns: {', '.join(sorted(rows[0].keys()))[:300]}")

    # Group the file by assay context so the operator can choose one.
    probe = normalise_records(rows, AssayContext(endpoint="__none__", cell_line="__none__"))
    print(f"\n  Assay groups present, largest first:")
    for key, count in list(probe.groups.items())[:20]:
        print(f"    {count:6d}  {key}")
    if len(probe.groups) > 20:
        print(f"    ... and {len(probe.groups) - 20} more")

    print("\n  Pick one group and pass its endpoint and cell line to import it.")
    print("  Pooling several of these into one training target is what this tool exists")
    print("  to prevent: they are different physical quantities.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import external bioactivity data into one assay context",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("source", help="CSV, TSV or JSON file you obtained yourself")
    parser.add_argument("--inspect", action="store_true",
                        help="report the assay groups in the file and exit")
    parser.add_argument("--endpoint", default="IC50", help="measurement type to keep")
    parser.add_argument("--cell-line", default="", help="cell line to keep")
    parser.add_argument("--organism", default="", help="organism, when relevant")
    parser.add_argument("--disease", default="PDAC", help="disease partition label")
    parser.add_argument("--output", default=None,
                        help="write the merged dataset here (default: dry run)")
    parser.add_argument("--merge-into-curated", action="store_true",
                        help="merge with the curated dataset rather than writing imports alone")
    args = parser.parse_args()

    path = Path(args.source)
    if not path.is_file():
        print(f"ERROR: no such file: {path}", file=sys.stderr)
        return 1

    rows = read_rows(path)
    if args.inspect:
        return cmd_inspect(rows)

    context = AssayContext(endpoint=args.endpoint, cell_line=args.cell_line,
                           organism=args.organism)

    print("=" * 76)
    print("Bioactivity import")
    print("=" * 76)
    print(f"  Source        : {path}")
    print(f"  Rows read     : {len(rows)}")
    print(f"  Target group  : {context.key}")

    report = normalise_records(rows, context)

    print(f"\n  Accepted      : {len(report.accepted)}")
    print(f"  Rejected      : {len(report.rejected)}")
    for reason, count in sorted(report.rejection_summary.items(), key=lambda kv: -kv[1]):
        print(f"      {count:6d}  {reason}")

    if report.conflicts:
        print(f"\n  {len(report.conflicts)} duplicate(s) disagreed by more than tenfold.")
        print("  These were combined by geometric mean and are listed in the output, but a")
        print("  tenfold disagreement usually means the sources are not comparable after all.")
        for conflict in report.conflicts[:5]:
            print(f"      {conflict['inchi_key'][:14]}  {conflict['values_um']}  "
                  f"{conflict['fold_disagreement']}x")

    if not report.accepted:
        print("\n  Nothing accepted. Run with --inspect to see which groups the file holds.")
        return 1

    summary = scaffold_summary(report.accepted)
    print(f"\n  Imported compounds        : {summary['compounds']}")
    print(f"  Independent scaffolds     : {summary['independent_scaffolds']}")
    print(f"  Singleton scaffolds       : {summary['singleton_scaffolds']}")
    print("  Scaffold count, not compound count, is the sample size a scaffold split sees.")

    final_records = report.accepted
    merge_info = None
    if args.merge_into_curated:
        payload = json.loads(CURATED.read_text(encoding="utf-8"))
        final_records, merge_info = merge_into_dataset(
            payload.get("compounds", []), report.accepted, args.disease)
        print(f"\n  Merged with the curated set:")
        for key, value in merge_info.items():
            print(f"      {key:26s} {value}")
        combined = scaffold_summary(
            [r for r in final_records if r.get("canonical_smiles")])
        print(f"      independent scaffolds      {combined['independent_scaffolds']}")

    if not args.output:
        print("\n  Dry run. Pass --output to write a dataset file.")
        return 0

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({
        "metadata": {
            "dataset_name": f"EVOCURE imported bioactivity, {context.key}",
            "assay_context": context.key,
            "source_file": str(path.name),
            "import_report": report.as_dict(),
            "merge": merge_info,
            "license": "Inherits the licence of the source data. Check it before redistributing.",
            "caveat": (
                "Imported values were normalised, not re-curated. They carry no citation "
                "and no independent check. Treat them as lower confidence than the "
                "hand-curated records."
            ),
        },
        "compounds": final_records,
    }, indent=2), encoding="utf-8")

    print(f"\n  Written to {destination}")
    print("  Retrain with: python scripts/train_qsar.py")
    print("  The training script prints the conformal interval, which is the number to watch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
