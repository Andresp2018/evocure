#!/usr/bin/env python3
"""Ask public databases whether a candidate is actually new, and what it resembles.

EVOCURE's novelty check compares a candidate against 111 structures held locally.
The report says so, and calls it a sanity check rather than a prior art search,
which is accurate and also means the question a reader cares about has not been
asked: does this molecule already exist, and has anyone taken it anywhere?

This script asks it. For each candidate it runs

  PubChem   exact structure lookup, so "already catalogued" is a yes or no
  ChEMBL    exact flexmatch, plus a similarity search returning the nearest
            known molecules with their highest clinical phase

and then computes Tanimoto locally, with RDKit, against a panel of drugs that
matter in pancreatic cancer: the chemotherapy backbone the disease is actually
treated with, and the KRAS-targeted agents that current progress is coming from.
Local computation rather than a server's number, so the figure is exact and
reproducible.

The clinical phase of the nearest neighbour is the part worth reading. A
candidate whose closest known relative never left preclinical is in different
company from one sitting next to an approved drug, and neither fact says
anything about the candidate itself.

What this still does not do. No patent search: SureChEMBL has no open REST API
of the kind used here, so prior art in the patent literature is not covered, and
that is where a great deal of chemical matter lives. Absence from PubChem and
ChEMBL is therefore not novelty. Presence is conclusive the other way: a
structure PubChem already holds was not invented by this search.

Nothing here is evidence of activity. Similarity to an approved drug does not
transfer its efficacy, and distance from one is not evidence of anything either.

    python scripts/check_prior_art.py
    python scripts/check_prior_art.py --results path/to/experiment_results.json
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import json
import sys
import time
import urllib.parse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"
USER_AGENT = "EVOCURE-prior-art (https://github.com/Andresp2018/evocure)"
MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# Drugs to measure the candidates against, by ChEMBL preferred name. Two groups,
# because the comparison only means something if both are present: what PDAC is
# treated with today, and what the agents driving current progress look like.
# Resolved from ChEMBL at run time so the structures are not transcribed here.
REFERENCE_PANEL: dict[str, str] = {
    "GEMCITABINE": "chemotherapy backbone",
    "FLUOROURACIL": "chemotherapy backbone",
    "PACLITAXEL": "chemotherapy backbone",
    "IRINOTECAN": "chemotherapy backbone",
    "ERLOTINIB": "targeted, EGFR",
    "SOTORASIB": "targeted, KRAS G12C",
    "ADAGRASIB": "targeted, KRAS G12C",
}

SIMILARITY_THRESHOLD = 70


def client(timeout: float) -> httpx.Client:
    return httpx.Client(timeout=timeout, follow_redirects=True,
                        headers={"User-Agent": USER_AGENT, "Accept": "application/json"})


def pubchem_exact(http: httpx.Client, smiles: str) -> dict[str, Any]:
    """Does PubChem already hold this exact structure?"""
    encoded = urllib.parse.quote(smiles, safe="")
    try:
        response = http.get(f"{PUBCHEM}/compound/smiles/{encoded}/cids/JSON")
        if response.status_code == 404:
            return {"found": False, "cids": []}
        response.raise_for_status()
        cids = response.json().get("IdentifierList", {}).get("CID", [])
        # PubChem returns CID 0 for "parsed but not present".
        cids = [c for c in cids if c]
        return {"found": bool(cids), "cids": cids[:5]}
    except Exception as exc:
        return {"found": None, "error": f"{type(exc).__name__}: {exc}"}


def chembl_exact(http: httpx.Client, smiles: str) -> dict[str, Any]:
    try:
        response = http.get(f"{CHEMBL}/molecule.json", params={
            "molecule_structures__canonical_smiles__flexmatch": smiles, "limit": 5})
        response.raise_for_status()
        payload = response.json()
        return {
            "found": bool(payload["page_meta"]["total_count"]),
            "matches": [{"chembl_id": m["molecule_chembl_id"],
                         "pref_name": m.get("pref_name"),
                         "max_phase": m.get("max_phase")}
                        for m in payload.get("molecules", [])],
        }
    except Exception as exc:
        return {"found": None, "error": f"{type(exc).__name__}: {exc}"}


def chembl_neighbours(http: httpx.Client, smiles: str,
                      threshold: int = SIMILARITY_THRESHOLD) -> dict[str, Any]:
    """Nearest known molecules, with how far anyone took them."""
    encoded = urllib.parse.quote(smiles, safe="")
    try:
        response = http.get(f"{CHEMBL}/similarity/{encoded}/{threshold}.json",
                            params={"limit": 20})
        if response.status_code >= 400:
            return {"count": 0, "nearest": [],
                    "note": f"similarity search returned {response.status_code}"}
        payload = response.json()
        rows = []
        for molecule in payload.get("molecules", []):
            rows.append({
                "chembl_id": molecule["molecule_chembl_id"],
                "pref_name": molecule.get("pref_name"),
                "similarity": (round(float(molecule["similarity"]), 2)
                               if molecule.get("similarity") is not None else None),
                "max_phase": molecule.get("max_phase"),
            })
        rows.sort(key=lambda r: -(r["similarity"] or 0))
        phases = [r["max_phase"] for r in rows if r["max_phase"] is not None]
        return {
            "count": int(payload["page_meta"]["total_count"]),
            "threshold_pct": threshold,
            "nearest": rows[:8],
            "highest_phase_among_neighbours": max(phases) if phases else None,
        }
    except Exception as exc:
        return {"count": None, "error": f"{type(exc).__name__}: {exc}"}


def resolve_panel(http: httpx.Client) -> dict[str, dict[str, Any]]:
    """Fetch the reference panel's structures from ChEMBL rather than hardcoding."""
    panel: dict[str, dict[str, Any]] = {}
    for name, role in REFERENCE_PANEL.items():
        try:
            response = http.get(f"{CHEMBL}/molecule.json",
                                params={"pref_name__iexact": name, "limit": 1})
            response.raise_for_status()
            molecules = response.json().get("molecules", [])
            if not molecules:
                print(f"  warning: {name} not found in ChEMBL, left out of the panel")
                continue
            molecule = molecules[0]
            smiles = (molecule.get("molecule_structures") or {}).get("canonical_smiles")
            mol = Chem.MolFromSmiles(smiles) if smiles else None
            if mol is None:
                print(f"  warning: {name} has no usable structure, left out")
                continue
            panel[name] = {
                "role": role,
                "chembl_id": molecule["molecule_chembl_id"],
                "max_phase": molecule.get("max_phase"),
                "smiles": Chem.MolToSmiles(mol),
                "fingerprint": MORGAN.GetFingerprint(mol),
            }
        except Exception as exc:
            print(f"  warning: {name} lookup failed ({type(exc).__name__})")
    return panel


def panel_similarity(smiles: str, panel: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    # An empty SMILES parses to a molecule with no atoms rather than to None, and
    # fingerprinting it returns Tanimoto 0.0 against everything. That reads as
    # "measured, and unrelated" when nothing was measured, so it is refused here
    # along with the unparseable case.
    mol = Chem.MolFromSmiles(smiles) if smiles and smiles.strip() else None
    if mol is None or mol.GetNumAtoms() == 0:
        return []
    own = MORGAN.GetFingerprint(mol)
    rows = [{
        "drug": name,
        "role": entry["role"],
        "chembl_id": entry["chembl_id"],
        "tanimoto": round(float(DataStructs.TanimotoSimilarity(own, entry["fingerprint"])), 4),
    } for name, entry in panel.items()]
    rows.sort(key=lambda r: -r["tanimoto"])
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check candidates against PubChem and ChEMBL, and against PDAC drugs.")
    parser.add_argument("--results", type=Path,
                        default=ROOT / "reports" / "pdac_publication_test" / "experiment_results.json")
    parser.add_argument("--threshold", type=int, default=SIMILARITY_THRESHOLD,
                        help="ChEMBL similarity threshold, percent. Default 70.")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reports" / "prior_art_check.json")
    args = parser.parse_args()

    if not args.results.is_file():
        print(f"No results at {args.results}", file=sys.stderr)
        return 1

    data = json.loads(args.results.read_text(encoding="utf-8"))
    candidates = data.get("candidates") or []
    if not candidates:
        print("No candidates in that results file.", file=sys.stderr)
        return 1

    print("=" * 78)
    print("  Prior art check against public databases")
    print("=" * 78)

    with client(args.timeout) as http:
        print("\n  Resolving the reference panel from ChEMBL ...")
        panel = resolve_panel(http)
        print(f"  Panel: {len(panel)} drugs")

        rows = []
        for candidate in candidates:
            label = candidate.get("label", "?")
            smiles = (candidate.get("compound") or {}).get("canonical_smiles")
            if not smiles:
                print(f"\n  {label}: no structure recorded, skipped")
                continue

            print(f"\n  {label}  {smiles}")
            pubchem = pubchem_exact(http, smiles)
            time.sleep(0.25)
            chembl = chembl_exact(http, smiles)
            time.sleep(0.25)
            neighbours = chembl_neighbours(http, smiles, args.threshold)
            time.sleep(0.25)
            against = panel_similarity(smiles, panel)

            exists = bool(pubchem.get("found")) or bool(chembl.get("found"))
            print(f"    PubChem exact      : "
                  + ("already catalogued, CID " + ", ".join(str(c) for c in pubchem["cids"])
                     if pubchem.get("found") else
                     "not found" if pubchem.get("found") is False else
                     f"lookup failed ({pubchem.get('error')})"))
            print(f"    ChEMBL exact       : "
                  + ("present as " + ", ".join(str(m["chembl_id"]) for m in chembl["matches"])
                     if chembl.get("found") else
                     "not found" if chembl.get("found") is False else
                     f"lookup failed ({chembl.get('error')})"))
            if neighbours.get("nearest"):
                top = neighbours["nearest"][0]
                print(f"    nearest in ChEMBL  : {top['chembl_id']} "
                      f"{top.get('pref_name') or '(unnamed)'} at {top['similarity']}% "
                      f"(highest phase among {neighbours['count']} neighbours: "
                      f"{neighbours.get('highest_phase_among_neighbours')})")
            else:
                print(f"    nearest in ChEMBL  : nothing above {args.threshold}%")
            if against:
                best = against[0]
                print(f"    closest PDAC drug  : {best['drug']} ({best['role']}) "
                      f"Tanimoto {best['tanimoto']}")
                kras = [r for r in against if "KRAS" in r["role"]]
                if kras:
                    print(f"    closest KRAS agent : {kras[0]['drug']} "
                          f"Tanimoto {kras[0]['tanimoto']}")

            rows.append({
                "label": label,
                "canonical_smiles": smiles,
                "already_in_public_databases": exists,
                "pubchem_exact": pubchem,
                "chembl_exact": chembl,
                "chembl_neighbours": neighbours,
                "tanimoto_to_pdac_drugs": against,
            })

    known = sum(1 for r in rows if r["already_in_public_databases"])
    print("\n" + "=" * 78)
    print(f"  {known} of {len(rows)} candidates already exist in PubChem or ChEMBL.")
    if rows and all(r["tanimoto_to_pdac_drugs"] for r in rows):
        chemo = [max((x["tanimoto"] for x in r["tanimoto_to_pdac_drugs"]
                      if x["role"] == "chemotherapy backbone"), default=0.0) for r in rows]
        kras = [max((x["tanimoto"] for x in r["tanimoto_to_pdac_drugs"]
                     if "KRAS" in x["role"]), default=0.0) for r in rows]
        print(f"  Tanimoto to the chemotherapy backbone: "
              f"{min(chemo):.2f} to {max(chemo):.2f}")
        print(f"  Tanimoto to the KRAS-targeted agents : "
              f"{min(kras):.2f} to {max(kras):.2f}")
        print("\n  Read that gap as a statement about what this platform can reach, not")
        print("  about what is worth reaching. A cytotoxicity QSAR trained on PANC-1")
        print("  cannot represent mutation-selective target engagement, so the search")
        print("  cannot propose that modality however many generations it runs.")

    print("\n  No patent search was performed. Absence here is not novelty, and")
    print("  similarity to an approved drug does not transfer its efficacy.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "results_file": str(args.results),
        "checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "similarity_threshold_pct": args.threshold,
        "reference_panel": {name: {k: v for k, v in entry.items() if k != "fingerprint"}
                            for name, entry in panel.items()},
        "candidates": rows,
        "already_in_public_databases": known,
        "candidates_checked": len(rows),
        "databases_searched": ["PubChem PUG-REST exact structure",
                               "ChEMBL exact flexmatch",
                               "ChEMBL similarity search"],
        "not_searched": ["SureChEMBL and the patent literature",
                         "internal or commercial compound collections"],
        "caveat": (
            "Absence from PubChem and ChEMBL is not novelty, because the patent "
            "literature was not searched. Presence is conclusive the other way. "
            "Tanimoto to an approved drug is a structural measurement and carries "
            "no information about activity in either direction."
        ),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\n  Written to {args.output}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
