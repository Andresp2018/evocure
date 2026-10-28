"""Medicinal chemistry filters, so the search stops proposing molecules nobody can make.

Until now the entire plausibility assessment was "RDKit parsed it and the valences
are legal". That admits structures a chemist would reject on sight: fused
nucleoside dimers, promiscuous assay interferers, strained or reactive groups, and
compounds with no plausible synthetic route.

This module adds four checks, all local and all from published rule sets:

  PAINS        pan-assay interference substructures (Baell and Holloway)
  Brenk        unstable, reactive and toxicophoric groups
  NIH          reactive and otherwise unsuitable functionality
  SA score     synthetic accessibility, 1 easy to 10 hard (Ertl and Schuffenhauer)

What these are and are not. They are coarse substructure and fragment-frequency
heuristics with well known false positives: real drugs trip them. Quercetin is a
PAINS hit, and several approved covalent inhibitors trip Brenk. So the default is
to score and flag, not to delete. Hard rejection is available but opt in, and the
report always states which structures were filtered and on what rule.

A compound passing every filter is still only "not obviously impossible". None of
this establishes that a molecule can be made, is stable, or does anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import math
import os
import sys

from rdkit import Chem
from rdkit.Chem import Descriptors, FilterCatalog
from rdkit.Chem.FilterCatalog import FilterCatalogParams

# Synthetic accessibility lives in RDKit's contrib tree, which is not importable
# as a package, so its directory is added to the path on first use.
_SA_SCORER = None


def _sa_scorer():
    global _SA_SCORER
    if _SA_SCORER is not None:
        return _SA_SCORER
    try:
        import rdkit
        contrib = os.path.join(os.path.dirname(rdkit.__file__), "Contrib", "SA_Score")
        if contrib not in sys.path:
            sys.path.append(contrib)
        import sascorer  # type: ignore
        _SA_SCORER = sascorer
    except Exception:
        _SA_SCORER = False
    return _SA_SCORER


@lru_cache(maxsize=1)
def _catalog() -> FilterCatalog.FilterCatalog:
    params = FilterCatalogParams()
    for name in ("PAINS", "BRENK", "NIH"):
        params.AddCatalog(getattr(FilterCatalogParams.FilterCatalogs, name))
    return FilterCatalog.FilterCatalog(params)


# Above this, a chemist would need a dedicated programme rather than a synthesis.
SA_HARD_THRESHOLD = 6.0
# Below this, routine medicinal chemistry.
SA_EASY_THRESHOLD = 4.0


@dataclass
class MedChemAssessment:
    """One structure's plausibility, with the reasoning kept attached."""

    smiles: str
    is_valid: bool = True
    sa_score: float | None = None
    sa_band: str = "not available"
    alerts: list[dict[str, str]] = field(default_factory=list)
    alert_families: list[str] = field(default_factory=list)
    heavy_atoms: int = 0
    stereocentres: int = 0
    macrocycle: bool = False
    verdict: str = "not assessed"
    flags: list[str] = field(default_factory=list)

    @property
    def alert_count(self) -> int:
        return len(self.alerts)

    @property
    def passes(self) -> bool:
        """No alerts and synthetically reachable. Not a claim that it works."""
        return self.is_valid and not self.alerts and (
            self.sa_score is None or self.sa_score < SA_HARD_THRESHOLD)

    def to_dict(self) -> dict[str, Any]:
        return {
            "smiles": self.smiles,
            "is_valid": self.is_valid,
            "sa_score": self.sa_score,
            "sa_band": self.sa_band,
            "alert_count": self.alert_count,
            "alert_families": self.alert_families,
            "alerts": self.alerts[:10],
            "heavy_atoms": self.heavy_atoms,
            "stereocentres": self.stereocentres,
            "macrocycle": self.macrocycle,
            "flags": self.flags,
            "verdict": self.verdict,
            "passes": self.passes,
            "caveat": (
                "Substructure alerts and SA score are coarse heuristics with known "
                "false positives; approved drugs trip them. Passing means not "
                "obviously implausible, not that the compound can be made or works."
            ),
        }


def _sa_band(score: float | None) -> str:
    if score is None:
        return "not available"
    if score < SA_EASY_THRESHOLD:
        return "routine"
    if score < SA_HARD_THRESHOLD:
        return "demanding"
    return "likely impractical"


def assess(smiles: str) -> MedChemAssessment:
    """Score one structure. Never raises on bad input; returns is_valid False."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return MedChemAssessment(smiles=smiles or "", is_valid=False,
                                 verdict="not a parseable structure")

    result = MedChemAssessment(smiles=Chem.MolToSmiles(mol, isomericSmiles=True))
    result.heavy_atoms = mol.GetNumHeavyAtoms()
    result.stereocentres = len(Chem.FindMolChiralCenters(
        mol, includeUnassigned=True, useLegacyImplementation=False))
    result.macrocycle = any(len(ring) >= 12 for ring in mol.GetRingInfo().AtomRings())

    for match in _catalog().GetMatches(mol):
        description = match.GetDescription() or "unnamed alert"
        # The catalogue exposes its source set as a property on some builds and
        # not others, so fall back to the description's prefix.
        family = ""
        try:
            if "FilterSet" in match.GetPropList():
                family = str(match.GetProp("FilterSet"))
        except Exception:
            family = ""
        if not family:
            family = description.split("_")[0].split("(")[0].strip()
        result.alerts.append({"family": family.upper() or "ALERT",
                              "description": description})
    result.alert_families = sorted({a["family"] for a in result.alerts})

    scorer = _sa_scorer()
    if scorer:
        try:
            result.sa_score = round(float(scorer.calculateScore(mol)), 3)
        except Exception:
            result.sa_score = None
    result.sa_band = _sa_band(result.sa_score)

    # Structural oddities the search is known to produce.
    if result.stereocentres >= 6:
        result.flags.append(f"{result.stereocentres} stereocentres, hard to make as a single isomer")
    if result.macrocycle:
        result.flags.append("macrocycle, specialised synthesis")
    if result.heavy_atoms > 50:
        result.flags.append(f"{result.heavy_atoms} heavy atoms, large for an oral drug")
    if "." in result.smiles:
        result.flags.append("multiple disconnected components, not a single molecule")

    if not result.is_valid:
        result.verdict = "not a parseable structure"
    elif result.alerts and result.sa_score and result.sa_score >= SA_HARD_THRESHOLD:
        result.verdict = "implausible: structural alerts and hard to synthesise"
    elif result.alerts:
        result.verdict = f"flagged: {', '.join(result.alert_families)}"
    elif result.sa_score and result.sa_score >= SA_HARD_THRESHOLD:
        result.verdict = "synthetically demanding, no structural alerts"
    else:
        result.verdict = "no alerts, synthetically reachable"
    return result


def plausibility_objective(assessment: MedChemAssessment) -> float:
    """Map an assessment to [0, 1] for use as a search objective, higher is better.

    Alerts cost but do not zero the score, because the rule sets have false
    positives and a hard zero would discard real chemistry.
    """
    if not assessment.is_valid:
        return 0.0

    score = 1.0
    score *= max(0.25, 1.0 - 0.25 * min(3, assessment.alert_count))

    if assessment.sa_score is not None:
        # SA runs 1 to 10; reward the routine end, penalise the impractical end.
        normalised = (10.0 - assessment.sa_score) / 9.0
        score *= max(0.1, min(1.0, normalised))

    if assessment.macrocycle:
        score *= 0.8
    if "." in assessment.smiles:
        score *= 0.5
    if assessment.stereocentres >= 6:
        score *= 0.8

    return round(max(0.0, min(1.0, score)), 4)


def screen(smiles_list: list[str], reject_hard: bool = False) -> dict[str, Any]:
    """Assess many structures and summarise what a filter would remove."""
    assessments = [assess(s) for s in smiles_list]
    removed = [a for a in assessments
               if not a.is_valid or (reject_hard and not a.passes)]
    kept = [a for a in assessments if a not in removed]

    families: dict[str, int] = {}
    for a in assessments:
        for family in a.alert_families:
            families[family] = families.get(family, 0) + 1

    scores = [a.sa_score for a in assessments if a.sa_score is not None]
    return {
        "total": len(assessments),
        "kept": len(kept),
        "removed": len(removed),
        "reject_hard": reject_hard,
        "alert_family_counts": dict(sorted(families.items(), key=lambda kv: -kv[1])),
        "with_any_alert": sum(1 for a in assessments if a.alerts),
        "sa_median": round(float(sorted(scores)[len(scores) // 2]), 3) if scores else None,
        "sa_impractical": sum(1 for s in scores if s >= SA_HARD_THRESHOLD),
        "assessments": [a.to_dict() for a in assessments],
        "note": (
            "Filtering is off by default. These rule sets flag approved drugs too, "
            "so removing every hit would discard real chemistry along with the noise."
        ),
    }
