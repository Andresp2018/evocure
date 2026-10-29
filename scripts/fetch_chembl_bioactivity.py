#!/usr/bin/env python3
"""Fetch cell line cytotoxicity data from ChEMBL, ready for the importer.

The project's own measurements put this first: the potency model trains on 43
compounds over 32 scaffolds, its 95% conformal interval is a factor of about
1330 on IC50, and its learning curve over held out scaffolds is still descending
steeply at that size. The estimator is data limited, not architecture limited,
and everything downstream of it inherits that limit.

DATA_SOURCES.md already says where the data is and import_bioactivity.py already
turns a download into a training set without pooling incomparable assays. The
missing step was getting the download: ChEMBL's web interface does not make a
cell line activity export easy, so the instruction "obtain a PANC-1 IC50 export"
was the point where the work stopped.

This script does that step against ChEMBL's public REST API and nothing more. It
writes a CSV whose columns are the ones import_bioactivity.py reads, plus a JSON
sidecar recording the query, the ChEMBL release and the counts, so a dataset
built this way can be traced back to the exact request that produced it.

What it does not do. It does not clean, average, impute or interpret. Censored
values, missing structures and records ChEMBL itself flags as suspect are
dropped and counted here, and everything that survives still goes through the
importer's assay comparability gate afterwards. It writes to data/raw/, which is
gitignored, and it is the only part of EVOCURE that touches the network: the
pipeline itself stays offline.

Licence. ChEMBL data is CC BY-SA 3.0. You may use it; check the terms before
redistributing it, and note that the share alike clause may affect what you can
do with a dataset derived from it. This script records the source so the
obligation travels with the file.

Examples:

  # What is available, without downloading anything
  python scripts/fetch_chembl_bioactivity.py --cell-line PANC-1 --survey

  # Fetch PANC-1 IC50 values
  python scripts/fetch_chembl_bioactivity.py --cell-line PANC-1 --endpoint IC50

  # Then inspect, import and retrain, which is where the number that matters moves
  python scripts/import_bioactivity.py data/raw/chembl_PANC-1_IC50.csv --inspect
  python scripts/import_bioactivity.py data/raw/chembl_PANC-1_IC50.csv \\
      --endpoint IC50 --cell-line PANC-1 --disease PDAC --merge-into-curated \\
      --output data/processed/pdac_expanded_dataset.json
  python scripts/train_qsar.py --dataset data/processed/pdac_expanded_dataset.json
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import csv
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx

API = "https://www.ebi.ac.uk/chembl/api/data"
PAGE_SIZE = 1000
USER_AGENT = "EVOCURE-bioactivity-fetch (https://github.com/Andresp2018/evocure)"

# The columns import_bioactivity.py reads. cell_line is written explicitly
# because a ChEMBL activity record carries the cell line as a target name, and
# the importer groups on an assay context it has to be able to see.
CSV_COLUMNS = [
    "canonical_smiles",
    "standard_type",
    "standard_relation",
    "standard_value",
    "standard_units",
    "cell_line",
    "molecule_chembl_id",
    "assay_chembl_id",
    "document_chembl_id",
    "document_year",
    "assay_description",
    "data_validity_comment",
]

# Units the importer converts exactly. Anything else would be rejected there, so
# it is not worth downloading.
CONVERTIBLE_UNITS = {"M", "mM", "uM", "µM", "μM", "nM", "pM"}


def client(timeout: float) -> httpx.Client:
    return httpx.Client(
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )


def chembl_release(http: httpx.Client) -> str:
    """The database release, so a fetched file can be tied to a version."""
    try:
        payload = http.get(f"{API}/status.json").json()
        return str(payload.get("chembl_db_version") or payload.get("version") or "unknown")
    except Exception:
        return "unknown"


def resolve_cell_line_target(http: httpx.Client, name: str) -> dict[str, str]:
    """Find the CELL-LINE target for a cell line name.

    ChEMBL models a cell line both as a cell_line resource and as a CELL-LINE
    target, and only the latter can be used to filter activities directly. An
    exact name match is required: 'PANC-1' and 'Gemcitabine-resistant PANC-1'
    are different biology and must never be pooled by a loose match.
    """
    response = http.get(f"{API}/target.json",
                        params={"pref_name__iexact": name, "limit": 20})
    response.raise_for_status()
    targets = [t for t in response.json().get("targets", [])
               if t.get("target_type") == "CELL-LINE"]

    if not targets:
        wider = http.get(f"{API}/target.json",
                         params={"pref_name__icontains": name, "limit": 25})
        candidates = [t.get("pref_name") for t in wider.json().get("targets", [])
                      if t.get("target_type") == "CELL-LINE"]
        raise SystemExit(
            f"No CELL-LINE target in ChEMBL whose name is exactly {name!r}.\n"
            + ("Close names, pass one of these verbatim:\n  "
               + "\n  ".join(str(c) for c in candidates if c)
               if candidates else "No similar cell line names were found either.")
        )

    if len(targets) > 1:
        names = ", ".join(str(t.get("pref_name")) for t in targets)
        raise SystemExit(f"{name!r} matched several CELL-LINE targets ({names}). "
                         f"Pass the one you want exactly.")

    target = targets[0]
    return {
        "target_chembl_id": str(target["target_chembl_id"]),
        "pref_name": str(target.get("pref_name") or name),
        "organism": str(target.get("organism") or ""),
    }


def survey(http: httpx.Client, target_id: str) -> dict[str, int]:
    """How many convertible records exist per endpoint, without downloading them."""
    found: dict[str, int] = {}
    for endpoint in ("IC50", "GI50", "EC50", "Ki", "Kd"):
        total = 0
        for unit in ("nM", "uM"):
            response = http.get(f"{API}/activity.json", params={
                "target_chembl_id": target_id, "standard_type": endpoint,
                "standard_units": unit, "limit": 1,
            })
            if response.status_code == 200:
                total += int(response.json()["page_meta"]["total_count"])
        if total:
            found[endpoint] = total
    return found


def fetch_activities(http: httpx.Client, target_id: str, endpoint: str,
                     max_records: int, pause: float) -> tuple[list[dict], dict[str, int]]:
    """Page through activities, keeping only rows the importer can use.

    Rejections are counted by reason rather than dropped quietly, so the summary
    says what a fetch actually contained.
    """
    rows: list[dict] = []
    counts = {"seen": 0, "censored": 0, "no_smiles": 0, "no_value": 0,
              "unit_not_convertible": 0, "flagged_by_chembl": 0}
    offset = 0

    while len(rows) < max_records:
        response = http.get(f"{API}/activity.json", params={
            "target_chembl_id": target_id,
            "standard_type": endpoint,
            "limit": PAGE_SIZE,
            "offset": offset,
        })
        response.raise_for_status()
        payload = response.json()
        activities = payload.get("activities", [])
        if not activities:
            break

        for record in activities:
            counts["seen"] += 1

            # ChEMBL's own quality flag. "Outside typical range" and the like mean
            # the curators doubt the number; training on it would be a choice to
            # trust it more than they do.
            if record.get("data_validity_comment"):
                counts["flagged_by_chembl"] += 1
                continue

            relation = str(record.get("standard_relation") or "").strip()
            if relation not in ("=", "=="):
                # A bound such as >100 uM is not a measurement. The importer
                # rejects these too; dropping them here keeps the file honest.
                counts["censored"] += 1
                continue

            smiles = record.get("canonical_smiles")
            if not smiles:
                counts["no_smiles"] += 1
                continue

            if record.get("standard_value") in (None, ""):
                counts["no_value"] += 1
                continue

            units = str(record.get("standard_units") or "").strip()
            if units not in CONVERTIBLE_UNITS:
                counts["unit_not_convertible"] += 1
                continue

            rows.append({
                "canonical_smiles": smiles,
                "standard_type": record.get("standard_type") or endpoint,
                "standard_relation": "=",
                "standard_value": record.get("standard_value"),
                "standard_units": units,
                "cell_line": "",  # filled by the caller with the resolved name
                "molecule_chembl_id": record.get("molecule_chembl_id") or "",
                "assay_chembl_id": record.get("assay_chembl_id") or "",
                "document_chembl_id": record.get("document_chembl_id") or "",
                "document_year": record.get("document_year") or "",
                "assay_description": (str(record.get("assay_description") or "")
                                      .replace("\n", " ")[:300]),
                "data_validity_comment": "",
            })
            if len(rows) >= max_records:
                break

        total = int(payload["page_meta"]["total_count"])
        offset += PAGE_SIZE
        print(f"  fetched {counts['seen']} of {total}, kept {len(rows)}", flush=True)
        if offset >= total:
            break
        if pause:
            time.sleep(pause)

    return rows, counts


def assay_fragmentation(rows: list[dict]) -> dict[str, Any]:
    """How many separate protocols a fetch actually pools together.

    This is the number that decides whether a fetch is usable, and it is not
    visible from the row count. "IC50 | PANC-1" is one assay comparability group
    to scripts/import_bioactivity.py, which groups on endpoint, cell line and
    organism. Measured on ChEMBL_37 it is 355 distinct assays from 325 distinct
    papers, median three records each, largest 70. Importing it pools 355
    protocols with different incubation times, readouts and conventions into one
    training target, which is the exact failure the importer exists to prevent
    and which its key cannot see.

    The consequences were measured, see reports/dataset_expansion_comparison.json:
    scored on an identical scaffold holdout, the model trained on 2106 pooled
    records was worse than the 43 compound curated model on four of five seeds.
    The trainer's own metrics looked better only because they were computed on a
    different, lower variance population.

    So read a high distinct_assays with a low largest_assay_records as a warning,
    not as a large dataset.
    """
    from collections import Counter

    assays = Counter(str(r.get("assay_chembl_id") or "") for r in rows)
    documents = Counter(str(r.get("document_chembl_id") or "") for r in rows)
    assay_sizes = sorted(assays.values(), reverse=True)
    doc_sizes = sorted(documents.values(), reverse=True)

    return {
        "distinct_assays": len(assays),
        "distinct_documents": len(documents),
        "largest_assay_records": assay_sizes[0] if assay_sizes else 0,
        "median_assay_records": (assay_sizes[len(assay_sizes) // 2]
                                 if assay_sizes else 0),
        "assays_with_40_or_more": sum(1 for s in assay_sizes if s >= 40),
        "largest_document_records": doc_sizes[0] if doc_sizes else 0,
        "largest_assay_share_pct": (round(100.0 * assay_sizes[0] / len(rows), 1)
                                    if rows and assay_sizes else 0.0),
    }


def scaffold_report(rows: list[dict]) -> dict[str, int]:
    """Scaffolds, not compounds, are the sample size a scaffold split sees.

    Five hundred analogues of one series is not five hundred data points, so a
    fetch is worth judging on this number rather than on the row count.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    scaffolds: set[str] = set()
    unique: set[str] = set()
    unparseable = 0
    for row in rows:
        mol = Chem.MolFromSmiles(str(row["canonical_smiles"]))
        if mol is None:
            unparseable += 1
            continue
        unique.add(Chem.MolToInchiKey(mol))
        try:
            scaffolds.add(MurckoScaffold.MurckoScaffoldSmiles(mol=mol))
        except Exception:
            continue
    return {
        "rows": len(rows),
        "unique_compounds": len(unique),
        "distinct_murcko_scaffolds": len(scaffolds),
        "unparseable_structures": unparseable,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch cell line activity data from ChEMBL for the importer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument("--cell-line", required=True,
                        help="Exact ChEMBL cell line name, for example PANC-1 or MIA PaCa-2.")
    parser.add_argument("--endpoint", default="IC50",
                        help="Measurement type to fetch. Default IC50.")
    parser.add_argument("--survey", action="store_true",
                        help="Report what is available and exit without downloading.")
    parser.add_argument("--max-records", type=int, default=20000,
                        help="Stop after this many usable rows. Default 20000.")
    parser.add_argument("--output", type=Path, default=None,
                        help="CSV destination. Default data/raw/chembl_<cell>_<endpoint>.csv")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--pause", type=float, default=0.2,
                        help="Seconds between pages, to stay polite. Default 0.2.")
    args = parser.parse_args()

    print("=" * 78)
    print("  ChEMBL bioactivity fetch")
    print("=" * 78)

    with client(args.timeout) as http:
        try:
            target = resolve_cell_line_target(http, args.cell_line)
        except httpx.HTTPError as exc:
            print(f"\nCould not reach ChEMBL: {exc}")
            print("This script is the only part of EVOCURE that needs the network. "
                  "Everything else runs offline.")
            return 2

        release = chembl_release(http)
        print(f"\n  cell line     : {target['pref_name']} ({target['target_chembl_id']})")
        print(f"  organism      : {target['organism'] or 'unspecified'}")
        print(f"  ChEMBL release: {release}")

        available = survey(http, target["target_chembl_id"])
        print("\n  Available records in convertible units:")
        for endpoint, count in sorted(available.items(), key=lambda kv: -kv[1]):
            print(f"    {endpoint:6s} {count:>8d}")
        if not available:
            print("    none")

        if args.survey:
            print("\n  Survey only, nothing downloaded.")
            return 0

        if args.endpoint not in available:
            print(f"\n  No {args.endpoint} records for this cell line in convertible units.")
            print("  Pick one of the endpoints listed above.")
            return 1

        print(f"\n  Fetching {args.endpoint} ...")
        rows, counts = fetch_activities(http, target["target_chembl_id"],
                                        args.endpoint, args.max_records, args.pause)

    for row in rows:
        row["cell_line"] = target["pref_name"]

    if not rows:
        print("\n  Nothing usable was returned. Nothing written.")
        return 1

    destination = args.output or (
        ROOT / "data" / "raw"
        / f"chembl_{target['pref_name'].replace(' ', '-')}_{args.endpoint}.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)

    with open(destination, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    summary = scaffold_report(rows)
    fragmentation = assay_fragmentation(rows)
    provenance = {
        "source": "ChEMBL REST API",
        "api_base": API,
        "chembl_release": release,
        "query": {
            "target_chembl_id": target["target_chembl_id"],
            "cell_line": target["pref_name"],
            "organism": target["organism"],
            "standard_type": args.endpoint,
        },
        "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "records_examined": counts["seen"],
        "records_written": len(rows),
        "dropped": {k: v for k, v in counts.items() if k != "seen"},
        "dataset_shape": summary,
        "assay_fragmentation": fragmentation,
        "licence": "ChEMBL data is CC BY-SA 3.0. Check the terms before redistributing.",
        "caveat": (
            "A fetch is not a dataset. These rows still have to pass the assay "
            "comparability gate in scripts/import_bioactivity.py, and scaffold "
            "count rather than row count is what a scaffold split sees as its "
            "sample size."
        ),
    }
    sidecar = destination.with_suffix(".provenance.json")
    sidecar.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    print(f"\n  Written       : {destination}")
    print(f"  Provenance    : {sidecar}")
    print(f"\n  rows written            : {summary['rows']}")
    print(f"  unique compounds        : {summary['unique_compounds']}")
    print(f"  distinct Murcko scaffolds: {summary['distinct_murcko_scaffolds']}")
    print("\n  dropped before writing:")
    for reason, count in provenance["dropped"].items():
        if count:
            print(f"    {reason:24s} {count}")

    print(f"\n  distinct assays         : {fragmentation['distinct_assays']}")
    print(f"  distinct papers         : {fragmentation['distinct_documents']}")
    print(f"  largest single assay    : {fragmentation['largest_assay_records']} records "
          f"({fragmentation['largest_assay_share_pct']}% of the file)")
    print(f"  median assay size       : {fragmentation['median_assay_records']} records")

    if fragmentation["distinct_assays"] > 20 and fragmentation["largest_assay_share_pct"] < 25:
        print("\n  READ THIS BEFORE IMPORTING. One endpoint and one cell line is not one")
        print(f"  assay. This file pools {fragmentation['distinct_assays']} protocols from "
              f"{fragmentation['distinct_documents']} papers, with different")
        print("  incubation times, readouts and conventions. The importer's comparability")
        print("  key groups on endpoint, cell line and organism, so it cannot see this.")
        print("  Measured consequence on exactly this data: a model trained on the pooled")
        print("  set was worse than the 43 compound curated model on four of five seeds,")
        print("  scored on an identical scaffold holdout. The trainer's own metrics looked")
        print("  better only because they were computed on a lower variance population.")
        print("  Reproduce with scripts/measure_dataset_expansion.py, and check whether a")
        print("  single protocol gives you enough: the largest here is "
              f"{fragmentation['largest_assay_records']} records.")

    print("\n  Scaffolds, not rows, are the sample size a scaffold split sees.")
    print("  The curated set has 32. Next:")
    print(f"\n    python scripts/import_bioactivity.py {destination} --inspect")
    print(f"    python scripts/import_bioactivity.py {destination} \\")
    print(f"        --endpoint {args.endpoint} --cell-line '{target['pref_name']}' \\")
    print("        --disease PDAC --merge-into-curated \\")
    print("        --output data/processed/pdac_expanded_dataset.json")
    print("    python scripts/train_qsar.py \\")
    print("        --dataset data/processed/pdac_expanded_dataset.json")
    print("\n  Watch conformal_q_95. That is the number that decides whether this "
          "model\n  orders compounds or says something about them.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
