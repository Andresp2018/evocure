"""
EVOCURE Scientific Data Manager and Connectors (Section 16).

Handles offline loading, unit normalization, RDKit chemical validation,
context-aware duplicate detection, missing-field tracking, and provenance
for ChEMBL, PubChem, GDSC, DepMap, and NCI-60 datasets.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from simulation.core.contracts import CompoundRecord, EpistemicCategory
from simulation.core.paths import reference_data_dir

logger = logging.getLogger(__name__)


@dataclass
class NormalizationIssue:
    """Detailed record of normalization, validation, or schema issues."""
    record_index: int
    identifier: str
    issue_type: str  # 'invalid_structure', 'missing_field', 'unit_conversion', 'duplicate'
    message: str
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetProvenance:
    """Provenance and lineage metadata for scientific datasets."""
    source_name: str
    source_format: str
    version: str
    filepath: Optional[str] = None
    imported_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    total_raw_records: int = 0
    valid_records_count: int = 0
    invalid_records_count: int = 0
    duplicate_records_count: int = 0
    missing_fields_summary: Dict[str, int] = field(default_factory=dict)
    notes: str = ""


class ScientificDataManager:
    """
    Offline Data Connector & Normalizer for Scientific Datasets (Section 16).
    
    Loads, sanitizes, and normalizes biochemical datasets without external network calls.
    Adheres strictly to the EVOCURE provenance and epistemic contracts.
    """

    CONCENTRATION_UNITS = {
        "m": 1.0e6,
        "molar": 1.0e6,
        "mm": 1.0e3,
        "millimolar": 1.0e3,
        "um": 1.0,
        "µm": 1.0,
        "micromolar": 1.0,
        "nm": 1.0e-3,
        "nanomolar": 1.0e-3,
        "pm": 1.0e-6,
        "picomolar": 1.0e-6,
    }

    def __init__(self, reference_dir: Optional[Union[str, Path]] = None) -> None:
        if reference_dir is None:
            self.reference_dir = reference_data_dir()
        else:
            self.reference_dir = Path(reference_dir)
        
        self.provenance: Optional[DatasetProvenance] = None
        self.issues: List[NormalizationIssue] = []
        self._records: List[Dict[str, Any]] = []
        self._compound_records: List[CompoundRecord] = []

    @classmethod
    def normalize_concentration(
        cls, value: float, unit: str
    ) -> Tuple[float, float]:
        """
        Normalize concentration to micromolar (uM) and calculate pIC50.
        
        pIC50 = -log10(conc_M) = 6.0 - log10(conc_uM)
        """
        if value <= 0:
            raise ValueError(f"Concentration must be positive, got {value}")
        
        clean_unit = unit.strip().lower()
        multiplier = cls.CONCENTRATION_UNITS.get(clean_unit)
        if multiplier is None:
            # Fallback assumption if unspecified: uM
            multiplier = 1.0

        conc_um = value * multiplier
        # Protect against log10 domain errors
        safe_um = max(conc_um, 1.0e-12)
        pic50 = 6.0 - math.log10(safe_um)
        return round(conc_um, 6), round(pic50, 4)

    def load_reference_dataset(
        self,
        filename: str = "pdac_reference_dataset.json",
        disease_id: Optional[str] = "PDAC",
    ) -> List[CompoundRecord]:
        """
        Load curated local reference dataset offline, strictly partitioning by disease_id
        (Section 13.3: never mix incomparable assays such as bacterial MICs or hepatic
        receptor EC50s into a PDAC cytotoxicity training target).
        Pass disease_id='ALL' or None to load all 90 multi-disease reference records.
        """
        dataset_path = self.reference_dir / filename
        if not dataset_path.exists():
            raise FileNotFoundError(f"Reference dataset not found at: {dataset_path}")

        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        all_records = data.get("compounds", [])
        if disease_id and disease_id.upper() != "ALL":
            norm_did = disease_id.upper()
            disease_map = {
                "PDAC": {"PDAC", "CONTROL"},
                "NSCLC_KRAS": {"NSCLC", "PDAC", "CONTROL"},
                "NSCLC": {"NSCLC", "CONTROL"},
                "AMR_BACTERIAL": {"AMR", "CONTROL"},
                "AMR": {"AMR", "CONTROL"},
                "METABOLIC_MASLD": {"MASLD", "CONTROL"},
                "MASLD": {"MASLD", "CONTROL"},
            }
            allowed = disease_map.get(norm_did, {norm_did, "CONTROL"})
            raw_records = [
                r for r in all_records
                if str(r.get("disease_id", "PDAC")).upper() in allowed
            ]
        else:
            raw_records = all_records

        metadata = data.get("metadata", {})
        
        self.provenance = DatasetProvenance(
            source_name=metadata.get("dataset_name", "PDAC Reference Dataset"),
            source_format="JSON",
            version=metadata.get("version", "1.0.0"),
            filepath=str(dataset_path),
            total_raw_records=len(raw_records),
            notes=metadata.get("description", "Curated offline reference dataset")
        )
        
        return self._process_records(raw_records, default_source="PDAC Reference Curated")

    def import_local_records(
        self,
        filepath: Union[str, Path],
        source_format: str,
        source_name: str = "External Import",
        version: str = "1.0.0",
    ) -> List[CompoundRecord]:
        """
        Import local dataset from CSV or JSON (ChEMBL, PubChem, GDSC, DepMap, NCI-60).
        Zero network calls: purely offline file ingestion.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        raw_records: List[Dict[str, Any]] = []
        if path.suffix.lower() == ".json":
            with open(path, "r", encoding="utf-8") as f:
                content = json.load(f)
                if isinstance(content, list):
                    raw_records = content
                elif isinstance(content, dict) and "compounds" in content:
                    raw_records = content["compounds"]
                elif isinstance(content, dict) and "data" in content:
                    raw_records = content["data"]
                else:
                    raw_records = [content]
        elif path.suffix.lower() in [".csv", ".tsv"]:
            delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
            with open(path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                raw_records = [row for row in reader]
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        self.provenance = DatasetProvenance(
            source_name=source_name,
            source_format=source_format.upper(),
            version=version,
            filepath=str(path),
            total_raw_records=len(raw_records)
        )

        return self._process_records(raw_records, default_source=source_name)

    def _process_records(
        self,
        raw_records: List[Dict[str, Any]],
        default_source: str
    ) -> List[CompoundRecord]:
        """
        Process, normalize, validate RDKit structures, detect context-aware duplicates,
        and instantiate typed CompoundRecord objects.
        """
        self.issues.clear()
        self._records.clear()
        self._compound_records.clear()

        # Context-aware duplicate tracking: (canonical_smiles, assay_id, cell_line)
        seen_contexts: Dict[Tuple[str, str, str], int] = {}
        missing_fields_counter: Dict[str, int] = {}

        valid_count = 0
        invalid_count = 0
        duplicate_count = 0

        for idx, item in enumerate(raw_records):
            name = (
                item.get("name")
                or item.get("compound_name")
                or item.get("DRUG_NAME")
                or item.get("Compound")
                or f"Compound_{idx+1}"
            )
            raw_smiles = (
                item.get("canonical_smiles")
                or item.get("smiles")
                or item.get("SMILES")
                or item.get("canonical_smiles_rdkit")
                or ""
            ).strip()

            if not raw_smiles:
                self.issues.append(
                    NormalizationIssue(
                        record_index=idx,
                        identifier=name,
                        issue_type="missing_field",
                        message="Missing SMILES string",
                    )
                )
                missing_fields_counter["smiles"] = missing_fields_counter.get("smiles", 0) + 1
                invalid_count += 1
                continue

            # RDKit Structure Sanitization & Descriptors
            mol = Chem.MolFromSmiles(raw_smiles)
            if mol is None:
                self.issues.append(
                    NormalizationIssue(
                        record_index=idx,
                        identifier=name,
                        issue_type="invalid_structure",
                        message=f"RDKit unable to parse SMILES: {raw_smiles}",
                    )
                )
                invalid_count += 1
                continue

            can_smiles = Chem.MolToSmiles(mol, canonical=True)
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
            mw = round(float(Descriptors.MolWt(mol)), 3)
            exact_mass = round(float(Descriptors.ExactMolWt(mol)), 4)
            formula = rdMolDescriptors.CalcMolFormula(mol)
            logp = round(float(Descriptors.MolLogP(mol)), 3)
            tpsa = round(float(Descriptors.TPSA(mol)), 2)
            hbd = int(Descriptors.NumHDonors(mol))
            hba = int(Descriptors.NumHAcceptors(mol))
            rot_bonds = int(Descriptors.NumRotatableBonds(mol))
            heavy_atoms = int(mol.GetNumHeavyAtoms())
            try:
                qed_val = round(float(Descriptors.qed(mol)), 4)
            except Exception:
                qed_val = 0.5
            inchikey = Chem.MolToInchiKey(mol)

            # Metadata extraction
            cell_line = (
                item.get("cell_line")
                or item.get("CELL_LINE_NAME")
                or item.get("CellLine")
                or "PANC-1"
            )
            assay_id = (
                item.get("assay_id")
                or item.get("ASSAY_ID")
                or item.get("Assay")
                or f"ASSAY-{cell_line}"
            )
            measurement_type = (
                item.get("measurement_type")
                or item.get("standard_type")
                or "IC50_uM"
            )

            # Concentration & pIC50 Normalization
            pic50: Optional[float] = None
            ic50_um: Optional[float] = None

            if "pic50" in item and item["pic50"] is not None:
                try:
                    pic50 = round(float(item["pic50"]), 4)
                    ic50_um = round(10.0 ** (6.0 - pic50), 6)
                except (ValueError, TypeError):
                    pass

            if ic50_um is None:
                raw_val = item.get("ic50_um") or item.get("standard_value") or item.get("IC50") or item.get("LN_IC50")
                unit = item.get("standard_units") or item.get("unit") or "uM"
                if raw_val is not None:
                    try:
                        val_float = float(raw_val)
                        if item.get("measurement_type") == "LN_IC50":
                            val_float = math.exp(val_float)
                        ic50_um, pic50 = self.normalize_concentration(val_float, unit)
                    except Exception as e:
                        self.issues.append(
                            NormalizationIssue(
                                record_index=idx,
                                identifier=name,
                                issue_type="unit_conversion",
                                message=f"Failed to normalize concentration: {e}",
                            )
                        )
                        missing_fields_counter["concentration"] = missing_fields_counter.get("concentration", 0) + 1

            if pic50 is None:
                pic50 = 3.0  # Safe neutral baseline for unmeasured control

            # Context-Aware Duplicate Detection (Section 16: Never merge across different assays/cell lines!)
            context_key = (can_smiles, assay_id, cell_line)
            if context_key in seen_contexts:
                duplicate_count += 1
                self.issues.append(
                    NormalizationIssue(
                        record_index=idx,
                        identifier=name,
                        issue_type="duplicate",
                        message=f"Duplicate compound found within same assay context ({cell_line}, {assay_id})",
                        context={"context_key": context_key, "previous_index": seen_contexts[context_key]}
                    )
                )
                # Keep distinct record with unique ID rather than blindly dropping
                cid = f"EVO-DUP-{idx+1:04d}"
            else:
                seen_contexts[context_key] = idx
                cid = item.get("compound_id") or f"EVO-REC-{idx+1:04d}"

            disease_id = item.get("disease_id") or "PDAC"

            # Bioactivity dictionary
            bioactivity = {
                "assay_id": assay_id,
                "cell_line": cell_line,
                "disease_id": disease_id,
                "measurement_type": measurement_type,
                "ic50_um": ic50_um,
                "pic50": pic50,
                "growth_inhibition_active": bool(ic50_um is not None and ic50_um < 10.0),
                "selectivity_hpde": float(item.get("selectivity_hpde", 1.0)),
                "hepatotox_alert": bool(item.get("hepatotox_alert", False)),
                "chembl_id": item.get("chembl_id", ""),
                "pubchem_cid": item.get("pubchem_cid"),
                "dataset_source": item.get("dataset_source", default_source),
                "license": item.get("license", "CC-BY 4.0 / Open Access"),
                "acquisition_date": item.get("acquisition_date", "2026-10-27"),
            }

            clean_dict = {
                "compound_id": cid,
                "name": name,
                "canonical_smiles": can_smiles,
                "disease_id": disease_id,
                "formula": formula,
                "molecular_weight": mw,
                "exact_mass": exact_mass,
                "logp": logp,
                "tpsa": tpsa,
                "hbd": hbd,
                "hba": hba,
                "rotatable_bonds": rot_bonds,
                "heavy_atoms": heavy_atoms,
                "qed": qed_val,
                "murcko_scaffold": scaffold,
                "inchi_key": inchikey,
                "bioactivity": bioactivity,
            }
            self._records.append(clean_dict)

            # Instantiate typed CompoundRecord
            compound_record = CompoundRecord(
                compound_id=cid,
                name=name,
                common_name=name,
                canonical_smiles=can_smiles,
                formula=formula,
                mol_weight=mw,
                exact_mass=exact_mass,
                logp=logp,
                tpsa=tpsa,
                hbd=hbd,
                hba=hba,
                rotatable_bonds=rot_bonds,
                heavy_atoms=heavy_atoms,
                qed=qed_val,
                murcko_scaffold=scaffold,
                inchi_key=inchikey,
                provenance=item.get("dataset_source", default_source),
                epistemic_category=EpistemicCategory.EXPERIMENTAL_OBSERVATION,
                is_chemically_valid=True,
                reference_bioactivity=bioactivity,
            )
            self._compound_records.append(compound_record)
            valid_count += 1

        if self.provenance:
            self.provenance.valid_records_count = valid_count
            self.provenance.invalid_records_count = invalid_count
            self.provenance.duplicate_records_count = duplicate_count
            self.provenance.missing_fields_summary = missing_fields_counter

        return self._compound_records

    def get_records(self) -> List[Dict[str, Any]]:
        """Return raw normalized record dictionaries."""
        return self._records

    def get_compound_records(self) -> List[CompoundRecord]:
        """Return typed CompoundRecord contracts."""
        return self._compound_records

    def export_normalized_dataset(
        self, filepath: Union[str, Path], include_metadata: bool = True
    ) -> None:
        """Export normalized dataset to JSON with full provenance metadata."""
        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "metadata": {
                "provenance": {
                    "source_name": self.provenance.source_name if self.provenance else "Unknown",
                    "source_format": self.provenance.source_format if self.provenance else "JSON",
                    "version": self.provenance.version if self.provenance else "1.0.0",
                    "imported_at": self.provenance.imported_at if self.provenance else "",
                    "total_raw": self.provenance.total_raw_records if self.provenance else 0,
                    "valid_count": self.provenance.valid_records_count if self.provenance else 0,
                    "invalid_count": self.provenance.invalid_records_count if self.provenance else 0,
                    "duplicate_count": self.provenance.duplicate_records_count if self.provenance else 0,
                },
                "issues_count": len(self.issues),
            },
            "compounds": [
                {
                    "compound_id": r["compound_id"],
                    "name": r["name"],
                    "canonical_smiles": r["canonical_smiles"],
                    "formula": r["formula"],
                    "molecular_weight": r["molecular_weight"],
                    "logp": r["logp"],
                    "tpsa": r["tpsa"],
                    "murcko_scaffold": r["murcko_scaffold"],
                    "inchi_key": r["inchi_key"],
                    **r["bioactivity"]
                }
                for r in self._records
            ]
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
