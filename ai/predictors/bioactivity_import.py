"""Import external bioactivity data without silently mixing incomparable assays.

Expanding the training set is the project's largest lever: the potency model is
data limited, and its learning curve is still descending at 32 scaffolds. But a
larger set assembled carelessly is worse than a small clean one, because mixing
assay contexts produces a model that looks better on paper and predicts nothing.

This module enforces the rule the curated dataset already follows. A bacterial MIC
and a human cytotoxicity IC50 are different physical quantities and never share a
training target, however similar their column headers look.

Design choices worth knowing:

* Records are grouped by assay comparability key, not by disease label. Two PANC-1
  IC50 values are comparable; a PANC-1 IC50 and a PANC-1 GI50 are not, and neither
  are IC50 values from different cell lines pooled without saying so.
* Units convert only where the conversion is exact. Mass concentration needs a
  molecular weight; without one the record is rejected rather than guessed.
* Duplicates are resolved explicitly, and disagreeing duplicates are reported
  rather than averaged away, because a tenfold disagreement between sources is
  information about data quality.
* Nothing is fabricated. A record missing a value is dropped and counted.

No network calls. Point it at a file you obtained yourself.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

import math

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

# Exact multipliers into micromolar. Anything not listed is refused.
_TO_MICROMOLAR: dict[str, float] = {
    "m": 1_000_000.0,
    "mm": 1_000.0,
    "um": 1.0,
    "µm": 1.0,
    "μm": 1.0,
    "nm": 0.001,
    "pm": 0.000001,
}

_MASS_UNITS = {"ug.ml-1", "ug/ml", "mcg/ml", "mg.l-1", "mg/l"}

# Measurement types that describe the same underlying quantity closely enough to
# pool. Anything outside this map keeps its own comparability group.
_CANONICAL_ENDPOINT: dict[str, str] = {
    "ic50": "IC50",
    "ic50_um": "IC50",
    "pic50": "IC50",
    "gi50": "GI50",
    "ec50": "EC50",
    "ki": "Ki",
    "kd": "Kd",
    "mic": "MIC",
    "ld50": "LD50",
}


@dataclass
class AssayContext:
    """What makes two measurements comparable."""

    endpoint: str
    cell_line: str
    organism: str = ""

    @property
    def key(self) -> str:
        parts = [self.endpoint.upper(), self.cell_line.upper().replace(" ", "-")]
        if self.organism:
            parts.append(self.organism.upper())
        return " | ".join(parts)


@dataclass
class ImportReport:
    """Everything that happened, including what was thrown away and why."""

    accepted: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    groups: dict[str, int] = field(default_factory=dict)

    @property
    def rejection_summary(self) -> dict[str, int]:
        return dict(Counter(r["reason"] for r in self.rejected))

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted_count": len(self.accepted),
            "rejected_count": len(self.rejected),
            "rejection_summary": self.rejection_summary,
            "conflict_count": len(self.conflicts),
            "assay_groups": self.groups,
            "conflicts": self.conflicts[:50],
        }


def to_micromolar(value: float, unit: str, mol_weight: float | None = None) -> float | None:
    """Convert a concentration to micromolar, or return None if it cannot be done exactly."""
    if value is None or not math.isfinite(float(value)) or float(value) <= 0:
        return None
    key = str(unit or "").strip().lower().replace(" ", "")

    if key in _TO_MICROMOLAR:
        return float(value) * _TO_MICROMOLAR[key]

    if key in _MASS_UNITS:
        # ug/mL to uM needs the molecular weight; refuse rather than approximate.
        if not mol_weight or mol_weight <= 0:
            return None
        return float(value) * 1000.0 / float(mol_weight)

    return None


def pic50_from_micromolar(ic50_um: float) -> float:
    return 6.0 - math.log10(ic50_um)


def canonical_endpoint(raw: str) -> str:
    key = str(raw or "").strip().lower().replace("-", "").replace("_", "")
    for token, canonical in _CANONICAL_ENDPOINT.items():
        if key.startswith(token):
            return canonical
    return str(raw or "UNKNOWN").strip().upper() or "UNKNOWN"


def _first(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        for key in (name, name.upper(), name.lower(), name.title()):
            if key in row and row[key] not in (None, ""):
                return row[key]
    return None


def normalise_records(
    rows: Iterable[dict[str, Any]],
    required_context: AssayContext,
    *,
    allow_pic50_column: bool = True,
) -> ImportReport:
    """Normalise raw rows into training records within one assay context.

    required_context names the only group accepted. Everything else is rejected
    with a reason, so an import that quietly discards most of a file says so.
    """
    report = ImportReport()
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    group_counter: Counter[str] = Counter()

    for index, row in enumerate(rows):
        smiles = _first(row, "canonical_smiles", "smiles", "SMILES", "Smiles")
        if not smiles:
            report.rejected.append({"row": index, "reason": "no SMILES"})
            continue

        mol = Chem.MolFromSmiles(str(smiles).strip())
        if mol is None:
            report.rejected.append({"row": index, "reason": "SMILES does not parse",
                                    "smiles": str(smiles)[:120]})
            continue

        endpoint = canonical_endpoint(
            _first(row, "standard_type", "measurement_type", "endpoint", "type") or "")
        cell_line = str(_first(row, "cell_line", "assay_cell_line", "CELL_LINE_NAME") or "").strip()
        organism = str(_first(row, "organism", "assay_organism") or "").strip()
        context = AssayContext(endpoint=endpoint, cell_line=cell_line, organism=organism)
        group_counter[context.key] += 1

        if context.key != required_context.key:
            report.rejected.append({
                "row": index, "reason": "assay context does not match the target group",
                "found": context.key, "required": required_context.key,
            })
            continue

        relation = str(_first(row, "standard_relation", "relation") or "=").strip()
        if relation not in ("=", "==", ""):
            # A censored value such as >100 uM is a bound, not a measurement.
            report.rejected.append({"row": index, "reason": f"censored value ({relation})"})
            continue

        mol_weight = float(Descriptors.MolWt(mol))
        ic50_um: float | None = None

        raw_pic50 = _first(row, "pic50", "pIC50")
        raw_value = _first(row, "standard_value", "value", "ic50_um", "ic50")
        raw_unit = _first(row, "standard_units", "units", "unit")

        if raw_value is not None:
            try:
                ic50_um = to_micromolar(float(raw_value), str(raw_unit or "uM"), mol_weight)
            except (TypeError, ValueError):
                ic50_um = None
            if ic50_um is None:
                report.rejected.append({
                    "row": index, "reason": f"unit not convertible ({raw_unit})"})
                continue
        elif allow_pic50_column and raw_pic50 is not None:
            try:
                ic50_um = 10.0 ** (6.0 - float(raw_pic50))
            except (TypeError, ValueError):
                ic50_um = None
            if ic50_um is None:
                report.rejected.append({"row": index, "reason": "pIC50 not numeric"})
                continue
        else:
            report.rejected.append({"row": index, "reason": "no activity value"})
            continue

        if not math.isfinite(ic50_um) or ic50_um <= 0:
            report.rejected.append({"row": index, "reason": "non-positive activity"})
            continue

        inchikey = Chem.MolToInchiKey(mol)
        record = {
            "name": str(_first(row, "name", "compound_name", "pref_name",
                               "molecule_chembl_id", "DRUG_NAME") or inchikey[:14]),
            "canonical_smiles": Chem.MolToSmiles(mol, isomericSmiles=True),
            "inchi_key": inchikey,
            "molecular_weight": round(mol_weight, 3),
            "murcko_scaffold": MurckoScaffold.MurckoScaffoldSmiles(mol=mol),
            "ic50_um": round(ic50_um, 6),
            "pic50": round(pic50_from_micromolar(ic50_um), 4),
            "assay_context": context.key,
            "cell_line": cell_line,
            "measurement_type": endpoint,
            "source_id": str(_first(row, "molecule_chembl_id", "chembl_id",
                                    "pubchem_cid", "id") or ""),
        }

        dedup_key = (inchikey, context.key)
        if dedup_key in seen:
            previous = seen[dedup_key]
            fold = max(previous["ic50_um"], ic50_um) / max(1e-12, min(previous["ic50_um"], ic50_um))
            if fold > 10.0:
                # Sources disagreeing by more than tenfold is a data quality signal,
                # not something to average into silence.
                report.conflicts.append({
                    "inchi_key": inchikey, "assay_context": context.key,
                    "values_um": sorted([previous["ic50_um"], round(ic50_um, 6)]),
                    "fold_disagreement": round(fold, 1),
                })
            # Geometric mean is the right summary for a log-scaled quantity.
            merged = math.sqrt(previous["ic50_um"] * ic50_um)
            previous["ic50_um"] = round(merged, 6)
            previous["pic50"] = round(pic50_from_micromolar(merged), 4)
            previous["replicate_count"] = previous.get("replicate_count", 1) + 1
            continue

        record["replicate_count"] = 1
        seen[dedup_key] = record

    report.accepted = list(seen.values())
    report.groups = dict(group_counter.most_common())
    return report


def scaffold_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Independent scaffolds, which is the sample size that matters for validation."""
    groups: dict[str, int] = defaultdict(int)
    for record in records:
        groups[record.get("murcko_scaffold", "")] += 1
    sizes = sorted(groups.values(), reverse=True)
    return {
        "compounds": len(records),
        "independent_scaffolds": len(groups),
        "largest_scaffold_group": sizes[0] if sizes else 0,
        "singleton_scaffolds": sum(1 for s in sizes if s == 1),
    }


def merge_into_dataset(
    existing: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
    disease_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Add imported records to a curated set, keeping curated values on collision.

    Curated records carry assay provenance and citations that an imported row does
    not, so a hand-checked value is never overwritten by a bulk import.
    """
    by_key: dict[str, dict[str, Any]] = {}
    for record in existing:
        mol = Chem.MolFromSmiles(record.get("canonical_smiles", ""))
        if mol is None:
            continue
        by_key[Chem.MolToInchiKey(mol)] = record

    added, skipped = 0, 0
    merged = list(existing)
    for record in incoming:
        key = record.get("inchi_key")
        if not key or key in by_key:
            skipped += 1
            continue
        entry = dict(record)
        entry["disease_id"] = disease_id
        entry["dataset_source"] = "External import, normalised"
        entry["growth_inhibition_active"] = bool(record["ic50_um"] < 10.0)
        merged.append(entry)
        by_key[key] = entry
        added += 1

    return merged, {
        "added": added,
        "skipped_already_present": skipped,
        "total_after_merge": len(merged),
    }
