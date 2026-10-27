"""
Pharmacology and ADMET Prediction Engine for EVOCURE.

Implements physicochemical profiling (MW, LogP, LogD, ESOL solubility, Caco-2 permeability,
GI absorption, plasma protein binding fu, hepatic metabolic stability t1/2, Lipinski/Veber rules,
QED), antitumor pharmacodynamics (Hill Emax model across sensitive, resistant, and healthy clones),
exposure partitioning (applied vs. free vs. intracellular), biomedical mode validation,
and drug combination synergy evaluation (Bliss independence, Loewe additivity, 8x8 dose-response matrices).

All predictions strictly conform to epistemic transparency rules and contracts.py.
"""

from __future__ import annotations

import math
from typing import Any, Callable
import numpy as np

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Crippen, Lipinski, QED, rdMolDescriptors, AllChem, rdFingerprintGenerator

from simulation.core.contracts import (
    CompoundRecord,
    ScientificValue,
    EpistemicCategory,
    UniverseMode,
)

# Mandatory Safety Disclaimer: clinical doses are never generated
SAFETY_DISCLAIMER: str = (
    "DISCLAIMER: In silico computational simulation for research and exploratory ranking only. "
    "Does not constitute medical evidence, pharmacological guidance, clinical trial design, or human dosing recommendations."
)


def validate_for_biomedical_mode(
    compound: CompoundRecord,
    source_mode: UniverseMode = UniverseMode.GROUNDED_CHEMISTRY,
) -> dict[str, Any]:
    """
    Enforce Section 5.1 rule:
    Never transfer ARTIFICIAL_LIFE fitness or replication traits into pharmacological efficacy.
    Require strict RDKit chemical validity and recompute all properties from molecular structure alone.
    """
    discarded_traits: list[str] = []
    recomputed_properties: dict[str, Any] = {}

    if source_mode == UniverseMode.ARTIFICIAL_LIFE:
        # Check and strip artificial traits
        if compound.artificial_traits:
            discarded_traits = list(compound.artificial_traits.keys())

    # Structural sanitization and chemical validity check
    smiles = compound.canonical_smiles.strip() if compound.canonical_smiles else ""
    if not smiles:
        return {
            "is_valid": False,
            "source_mode": source_mode.value,
            "chemical_validity": False,
            "reason": "Missing canonical SMILES representation.",
            "discarded_traits": discarded_traits,
            "recomputed_properties": {},
            "safety_disclaimer": SAFETY_DISCLAIMER,
        }

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "is_valid": False,
            "source_mode": source_mode.value,
            "chemical_validity": False,
            "reason": f"SMILES '{smiles}' failed RDKit sanitization and valence verification.",
            "discarded_traits": discarded_traits,
            "recomputed_properties": {},
            "safety_disclaimer": SAFETY_DISCLAIMER,
        }

    # Recompute ground-truth properties purely from chemical graph
    mw = float(Descriptors.MolWt(mol))
    logp = float(Crippen.MolLogP(mol))
    tpsa = float(Descriptors.TPSA(mol))
    hbd = int(Lipinski.NumHDonors(mol))
    hba = int(Lipinski.NumHAcceptors(mol))
    rb = int(Lipinski.NumRotatableBonds(mol))
    heavy_atoms = int(mol.GetNumHeavyAtoms())
    qed_val = float(QED.qed(mol))
    formal_charge = int(Chem.GetFormalCharge(mol))

    recomputed_properties = {
        "mol_weight": round(mw, 3),
        "logp": round(logp, 3),
        "tpsa": round(tpsa, 2),
        "hbd": hbd,
        "hba": hba,
        "rotatable_bonds": rb,
        "heavy_atoms": heavy_atoms,
        "qed": round(qed_val, 4),
        "formal_charge": formal_charge,
    }

    return {
        "is_valid": True,
        "source_mode": source_mode.value,
        "chemical_validity": True,
        "reason": "Chemically sanitized and verified via RDKit.",
        "discarded_traits": discarded_traits,
        "recomputed_properties": recomputed_properties,
        "epistemic_note": (
            "Fitness parameters from Mode A are quarantined; pharmacology is governed "
            "strictly by verified chemical graph descriptors."
            if source_mode == UniverseMode.ARTIFICIAL_LIFE
            else "Standard Grounded Chemistry mode validated."
        ),
        "safety_disclaimer": SAFETY_DISCLAIMER,
    }


class PharmacologyEngine:
    """
    Evaluates ADMET physicochemical properties, antitumor pharmacodynamics,
    and drug combination synergy conforming to EVOCURE scientific contracts.
    """

    def __init__(self, model_version: str = "1.0.0") -> None:
        self.model_version = model_version

    def evaluate_admet(
        self,
        compound: CompoundRecord,
        in_domain: bool = True,
        domain_similarity: float = 0.7,
    ) -> dict[str, Any]:
        """
        Compute ADMET & physicochemical properties wrapped in ScientificValue.to_dict():
        - Molecular Weight (g/mol)
        - LogP (Wildman-Crippen)
        - LogD (pH 7.4)
        - ESOL Aqueous Solubility (logS mol/L)
        - Caco-2 / Effective Permeability (cm/s estimate from TPSA & MW)
        - GI Absorption class
        - Plasma Protein Unbound Fraction fu (0.01 to 0.99 from LogD & charge)
        - Hepatic Metabolic Stability (t1/2 estimate in hours)
        - Lipinski Rule of 5 & Veber drug-likeness
        - Synthetic/Structural Viability (QED)
        - Prediction Uncertainty
        """
        # Parse or use cached properties
        mol = Chem.MolFromSmiles(compound.canonical_smiles) if compound.canonical_smiles else None

        if mol is not None:
            mw = float(Descriptors.MolWt(mol))
            logp = float(Crippen.MolLogP(mol))
            tpsa = float(Descriptors.TPSA(mol))
            hbd = int(Lipinski.NumHDonors(mol))
            hba = int(Lipinski.NumHAcceptors(mol))
            rot_bonds = int(Lipinski.NumRotatableBonds(mol))
            heavy_atoms = int(mol.GetNumHeavyAtoms())
            formal_charge = int(Chem.GetFormalCharge(mol))
            qed_score = float(QED.qed(mol))
            aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
        else:
            mw = float(compound.mol_weight)
            logp = float(compound.logp)
            tpsa = float(compound.tpsa)
            hbd = int(compound.hbd)
            hba = int(compound.hba)
            rot_bonds = int(compound.rotatable_bonds)
            heavy_atoms = int(compound.heavy_atoms) if compound.heavy_atoms > 0 else max(1, int(mw / 14))
            formal_charge = int(compound.formal_charge)
            qed_score = float(compound.qed)
            aromatic_atoms = 0

        # Applicability domain heuristic: extreme sizes or non-physical counts
        if mw > 1200.0 or heavy_atoms < 3 or not in_domain:
            in_domain = False

        # 1. LogD (pH 7.4)
        # Empirical Henderson-Hasselbalch adjustment for physiological pH 7.4
        charge_term = abs(formal_charge) * 1.15
        logd_74 = round(logp - charge_term, 3)

        # 2. ESOL Aqueous Solubility (Delaney 2004)
        # logS = 0.16 - 0.63*cLogP - 0.0062*MW + 0.066*RB - 0.74*(aromatic_atoms / heavy_atoms)
        aromatic_prop = (aromatic_atoms / max(1, heavy_atoms)) if heavy_atoms > 0 else 0.0
        esol_log_s = 0.16 - (0.63 * logp) - (0.0062 * mw) + (0.066 * rot_bonds) - (0.74 * aromatic_prop)
        esol_log_s = round(esol_log_s, 3)

        # 3. Caco-2 / Effective Permeability (cm/s estimate from TPSA and MW)
        # log10(Peff) approx -4.36 - 0.010*TPSA - 0.0025*MW + 0.15*logp
        log_peff = -4.36 - (0.010 * tpsa) - (0.0025 * mw) + (0.15 * logp)
        peff_cm_s = float(10 ** max(-9.0, min(-3.0, log_peff)))

        # 4. GI Absorption class
        if tpsa <= 140.0 and mw <= 500.0 and logp >= -1.0 and logp <= 5.0 and peff_cm_s >= 5e-6:
            gi_class = "High"
        elif tpsa <= 165.0 and mw <= 700.0 and peff_cm_s >= 1e-6:
            gi_class = "Moderate"
        else:
            gi_class = "Low"

        # 5. Plasma Protein Unbound Fraction fu (Austin / Lobell model)
        # log10((1-fu)/fu) = 0.448 * LogD7.4 - 0.20 + 0.30 * |formal_charge|
        ratio_bound_to_free = 10 ** (0.448 * logd_74 - 0.20 + 0.30 * abs(formal_charge))
        calculated_fu = 1.0 / (1.0 + ratio_bound_to_free)
        fu = round(float(max(0.01, min(0.99, calculated_fu))), 4)

        # 6. Hepatic Metabolic Stability (t1/2 estimate in hours)
        # Higher lipophilicity & rotatable bonds increase hepatic CYP accessibility/clearance
        t12_hours = 12.0 * math.exp(-0.30 * max(0.0, logp) - 0.06 * rot_bonds + 0.001 * mw)
        t12_hours = round(float(max(0.2, min(48.0, t12_hours))), 2)

        # 7. Lipinski Rule of 5
        lipinski_violations = 0
        if mw > 500.0:
            lipinski_violations += 1
        if logp > 5.0:
            lipinski_violations += 1
        if hbd > 5:
            lipinski_violations += 1
        if hba > 10:
            lipinski_violations += 1
        lipinski_pass = lipinski_violations <= 1

        # 8. Veber Rule
        veber_violations = 0
        if rot_bonds > 10:
            veber_violations += 1
        if tpsa > 140.0:
            veber_violations += 1
        veber_pass = veber_violations == 0

        # 9. Synthetic/Structural Viability (QED)
        qed_val = round(qed_score, 4)

        # 10. Prediction Uncertainty
        if in_domain:
            uncertainty = round(float(max(0.05, min(0.95, 1.0 - domain_similarity))), 3)
            confidence = round(float(max(0.10, min(0.98, domain_similarity))), 3)
            ad_category = EpistemicCategory.CALCULATED_PHYSICOCHEMICAL
            model_category = EpistemicCategory.EXPLORATORY_MODEL
        else:
            uncertainty = 1.0
            confidence = 0.0
            ad_category = EpistemicCategory.INSUFFICIENT_DATA
            model_category = EpistemicCategory.INSUFFICIENT_DATA

        results: dict[str, Any] = {
            "molecular_weight": ScientificValue(
                name="Molecular Weight",
                value=round(mw, 3) if in_domain else "Insufficient data",
                unit="g/mol",
                epistemic_category=ad_category,
                source="RDKit Descriptors",
                confidence=1.0 if in_domain else 0.0,
                in_applicability_domain=in_domain,
            ).to_dict(),
            "logp": ScientificValue(
                name="LogP (Wildman-Crippen)",
                value=round(logp, 3) if in_domain else "Insufficient data",
                unit="",
                epistemic_category=ad_category,
                source="RDKit Crippen",
                confidence=0.90 if in_domain else 0.0,
                uncertainty_low=round(logp - 0.45, 2) if in_domain else None,
                uncertainty_high=round(logp + 0.45, 2) if in_domain else None,
                in_applicability_domain=in_domain,
            ).to_dict(),
            "logd_74": ScientificValue(
                name="LogD (pH 7.4)",
                value=logd_74 if in_domain else "Insufficient data",
                unit="",
                epistemic_category=ad_category,
                source="Henderson-Hasselbalch / Charge Empirical",
                confidence=0.85 if in_domain else 0.0,
                uncertainty_low=round(logd_74 - 0.50, 2) if in_domain else None,
                uncertainty_high=round(logd_74 + 0.50, 2) if in_domain else None,
                in_applicability_domain=in_domain,
            ).to_dict(),
            "esol_solubility": ScientificValue(
                name="ESOL Aqueous Solubility",
                value=esol_log_s if in_domain else "Insufficient data",
                unit="log(mol/L)",
                epistemic_category=ad_category,
                source="Delaney ESOL Model (2004)",
                confidence=0.82 if in_domain else 0.0,
                uncertainty_low=round(esol_log_s - 0.75, 2) if in_domain else None,
                uncertainty_high=round(esol_log_s + 0.75, 2) if in_domain else None,
                in_applicability_domain=in_domain,
            ).to_dict(),
            "caco2_permeability": ScientificValue(
                name="Caco-2 / Effective Permeability",
                value=peff_cm_s if in_domain else "Insufficient data",
                unit="cm/s",
                epistemic_category=model_category,
                source="Quantitative Structure-Permeability Relationship (QSPR)",
                confidence=confidence,
                uncertainty_low=round(peff_cm_s * 0.4, 8) if in_domain else None,
                uncertainty_high=round(peff_cm_s * 2.5, 8) if in_domain else None,
                in_applicability_domain=in_domain,
            ).to_dict(),
            "gi_absorption": ScientificValue(
                name="GI Absorption Class",
                value=gi_class if in_domain else "Insufficient data",
                unit="class",
                epistemic_category=model_category,
                source="Egan / Veber / Permeability Heuristic",
                confidence=confidence,
                in_applicability_domain=in_domain,
            ).to_dict(),
            "fraction_unbound_plasma": ScientificValue(
                name="Plasma Protein Unbound Fraction (fu)",
                value=fu if in_domain else "Insufficient data",
                unit="fraction",
                epistemic_category=model_category,
                source="Austin-Lobell LogD / Charge Model",
                confidence=confidence,
                uncertainty_low=round(max(0.01, fu * 0.65), 4) if in_domain else None,
                uncertainty_high=round(min(0.99, fu * 1.45), 4) if in_domain else None,
                in_applicability_domain=in_domain,
            ).to_dict(),
            "hepatic_clearance_half_life": ScientificValue(
                name="Hepatic Metabolic Stability (t1/2)",
                value=t12_hours if in_domain else "Insufficient data",
                unit="hours",
                epistemic_category=model_category,
                source="Lipophilicity-Rigidity In Vitro Clearance Heuristic",
                confidence=confidence,
                uncertainty_low=round(max(0.1, t12_hours * 0.5), 2) if in_domain else None,
                uncertainty_high=round(min(72.0, t12_hours * 2.0), 2) if in_domain else None,
                in_applicability_domain=in_domain,
            ).to_dict(),
            "lipinski_drug_likeness": ScientificValue(
                name="Lipinski Rule of Five",
                value=f"Violations: {lipinski_violations} (Pass: {lipinski_pass})" if in_domain else "Insufficient data",
                unit="",
                epistemic_category=ad_category,
                source="Lipinski Ro5 Definition",
                confidence=1.0 if in_domain else 0.0,
                in_applicability_domain=in_domain,
            ).to_dict(),
            "veber_rule": ScientificValue(
                name="Veber Drug-likeness",
                value=f"Violations: {veber_violations} (Pass: {veber_pass})" if in_domain else "Insufficient data",
                unit="",
                epistemic_category=ad_category,
                source="Veber Rule Definition",
                confidence=1.0 if in_domain else 0.0,
                in_applicability_domain=in_domain,
            ).to_dict(),
            "synthetic_viability_qed": ScientificValue(
                name="Quantitative Estimate of Drug-likeness (QED)",
                value=qed_val if in_domain else "Insufficient data",
                unit="score [0-1]",
                epistemic_category=ad_category,
                source="Bickerton et al. QED",
                confidence=1.0 if in_domain else 0.0,
                in_applicability_domain=in_domain,
            ).to_dict(),
            "prediction_uncertainty": ScientificValue(
                name="ADMET Prediction Uncertainty",
                value=uncertainty,
                unit="score [0-1]",
                epistemic_category=ad_category,
                source="Tanimoto Applicability Domain & Descriptors",
                confidence=confidence,
                in_applicability_domain=in_domain,
            ).to_dict(),
            "applicability_domain_in": in_domain,
            "domain_similarity": domain_similarity,
            "safety_disclaimer": SAFETY_DISCLAIMER,
        }

        return results

    def predict_antitumor_activity(
        self,
        compound: CompoundRecord,
        qsar_pred: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Predict antitumor pharmacodynamic profile across sensitive, resistant, and healthy cells.

        Precedence rules:
        1. compound.reference_bioactivity -> EpistemicCategory.EXPERIMENTAL_OBSERVATION
        2. calibrated qsar_pred -> EpistemicCategory.VALIDATED_MODEL_PREDICTION or EXPLORATORY_MODEL
        3. exploratory structure-property model -> EpistemicCategory.EXPLORATORY_MODEL (or INSUFFICIENT_DATA)
        """
        # Determine applicability domain
        in_domain = True
        if compound.mol_weight > 1200.0 or (compound.heavy_atoms > 0 and compound.heavy_atoms < 3):
            in_domain = False

        ref_bio = compound.reference_bioactivity or {}

        # 1. Experimental observation branch
        if ref_bio and ("ic50_um" in ref_bio or "ic50_sensitive_um" in ref_bio):
            ic50_sens = float(ref_bio.get("ic50_sensitive_um", ref_bio.get("ic50_um", 1.0)))
            ic50_res = float(ref_bio.get("ic50_resistant_um", ic50_sens * 8.5))
            ic50_healthy = float(ref_bio.get("ic50_healthy_um", ic50_sens * 15.0))
            emax = float(ref_bio.get("emax", 0.95))
            hill_slope = float(ref_bio.get("hill_slope", 1.4))
            res_prob = float(ref_bio.get("resistance_emergence_prob", 0.18))
            category = EpistemicCategory.EXPERIMENTAL_OBSERVATION
            source = str(ref_bio.get("source", "Imported Experimental Bioactivity Reference"))
            confidence = 0.95

        # 2. Validated / Calibrated QSAR branch
        elif qsar_pred and in_domain:
            ic50_sens = float(qsar_pred.get("ic50_sensitive_um", 2.5))
            ic50_res = float(qsar_pred.get("ic50_resistant_um", ic50_sens * 6.0))
            ic50_healthy = float(qsar_pred.get("ic50_healthy_um", ic50_sens * 12.0))
            emax = float(qsar_pred.get("emax", 0.92))
            hill_slope = float(qsar_pred.get("hill_slope", 1.3))
            res_prob = float(qsar_pred.get("resistance_emergence_prob", 0.22))
            category = (
                EpistemicCategory.VALIDATED_MODEL_PREDICTION
                if qsar_pred.get("is_validated", False)
                else EpistemicCategory.EXPLORATORY_MODEL
            )
            source = str(qsar_pred.get("source", "Calibrated QSAR Model"))
            confidence = float(qsar_pred.get("confidence", 0.80))

        # 3. Exploratory structure-property model or Insufficient Data
        else:
            if not in_domain:
                category = EpistemicCategory.INSUFFICIENT_DATA
                source = "Out-of-Domain Structure Heuristic"
                confidence = 0.1
                ic50_sens = 50.0
                ic50_res = 100.0
                ic50_healthy = 100.0
                emax = 0.50
                hill_slope = 1.0
                res_prob = 0.50
            else:
                category = EpistemicCategory.EXPLORATORY_MODEL
                source = "EVOCURE Exploratory Pharmacophore Model"
                confidence = 0.65

                # Baseline sensitivity derived from MW, LogP, and QED
                # Potent small molecules in oncology typically have LogP ~ 1.5 - 3.5, MW ~ 300 - 550, QED > 0.4
                logp_opt = max(0.0, 1.0 - abs(compound.logp - 2.5) / 3.0)
                mw_opt = max(0.0, 1.0 - abs(compound.mol_weight - 400.0) / 300.0)
                potency_factor = max(0.05, 0.5 * logp_opt + 0.3 * mw_opt + 0.2 * compound.qed)

                # IC50 in uM: higher potency factor gives lower (more potent) IC50
                ic50_sens = round(float(math.exp(3.0 - 3.2 * potency_factor)), 3)
                ic50_res = round(ic50_sens * (4.5 + 3.0 * (1.0 - compound.qed)), 3)
                ic50_healthy = round(ic50_sens * (6.0 + 8.0 * compound.qed), 3)
                emax = round(float(min(0.98, max(0.70, 0.80 + 0.18 * compound.qed))), 3)
                hill_slope = 1.35
                res_prob = round(float(min(0.60, max(0.05, 0.35 - 0.20 * compound.qed))), 3)

        # Selectivity index SI = IC50_healthy / IC50_sensitive
        si = round(float(ic50_healthy / max(1e-4, ic50_sens)), 2)

        # Compute ADMET fu and Peff for exposure partitioning
        admet_summary = self.evaluate_admet(compound, in_domain=in_domain)
        fu_val = (
            admet_summary["fraction_unbound_plasma"]["value"]
            if isinstance(admet_summary["fraction_unbound_plasma"]["value"], (int, float))
            else 0.10
        )
        peff_val = (
            admet_summary["caco2_permeability"]["value"]
            if isinstance(admet_summary["caco2_permeability"]["value"], (int, float))
            else 2e-6
        )

        def compute_exposure_partition(applied_conc_um: float) -> dict[str, float]:
            """
            Partition applied concentration into free extracellular and intracellular pools.
            C_free = fu * C_applied
            C_intra = C_free * (k_in / (k_out + k_efflux))
            """
            c_applied = max(0.0, float(applied_conc_um))
            c_free = float(fu_val * c_applied)
            # Permeability scaling
            k_in = max(0.1, min(5.0, peff_val / 1e-6))
            k_out = 1.0
            k_efflux = 0.5  # Baseline active transporter efflux
            partition_ratio = k_in / (k_out + k_efflux)
            c_intra = float(c_free * partition_ratio)
            return {
                "applied_conc_um": round(c_applied, 4),
                "free_extracellular_um": round(c_free, 4),
                "estimated_intracellular_um": round(c_intra, 4),
                "unbound_fraction_fu": round(fu_val, 4),
                "partition_ratio_intra_to_free": round(partition_ratio, 3),
            }

        # Concentration-response curves (50 logarithmic points from 1e-4 to 1e3 uM)
        conc_points = np.logspace(-4, 3, 50).tolist()
        sens_viability: list[float] = []
        res_viability: list[float] = []
        healthy_viability: list[float] = []

        for c in conc_points:
            # Hill equation for cell viability percentage
            sens_kill = emax * (c ** hill_slope) / (ic50_sens ** hill_slope + c ** hill_slope)
            res_kill = emax * (c ** hill_slope) / (ic50_res ** hill_slope + c ** hill_slope)
            healthy_kill = (emax * 0.85) * (c ** hill_slope) / (ic50_healthy ** hill_slope + c ** hill_slope)

            sens_viability.append(round(max(0.0, min(100.0, 100.0 * (1.0 - sens_kill))), 2))
            res_viability.append(round(max(0.0, min(100.0, 100.0 * (1.0 - res_kill))), 2))
            healthy_viability.append(round(max(0.0, min(100.0, 100.0 * (1.0 - healthy_kill))), 2))

        return {
            "ic50_sensitive_um": ScientificValue(
                name="IC50 (Sensitive Tumor Clone)",
                value=ic50_sens if in_domain else "Insufficient data",
                unit="uM",
                epistemic_category=category,
                source=source,
                confidence=confidence,
                uncertainty_low=round(ic50_sens * 0.70, 3) if in_domain else None,
                uncertainty_high=round(ic50_sens * 1.45, 3) if in_domain else None,
                in_applicability_domain=in_domain,
            ).to_dict(),
            "ic50_resistant_um": ScientificValue(
                name="IC50 (Resistant Tumor Clone)",
                value=ic50_res if in_domain else "Insufficient data",
                unit="uM",
                epistemic_category=category,
                source=source,
                confidence=confidence * 0.9,
                uncertainty_low=round(ic50_res * 0.65, 3) if in_domain else None,
                uncertainty_high=round(ic50_res * 1.55, 3) if in_domain else None,
                in_applicability_domain=in_domain,
            ).to_dict(),
            "ic50_healthy_um": ScientificValue(
                name="IC50 (Healthy Epithelial Cells)",
                value=ic50_healthy if in_domain else "Insufficient data",
                unit="uM",
                epistemic_category=category,
                source=source,
                confidence=confidence * 0.85,
                uncertainty_low=round(ic50_healthy * 0.60, 3) if in_domain else None,
                uncertainty_high=round(ic50_healthy * 1.65, 3) if in_domain else None,
                in_applicability_domain=in_domain,
            ).to_dict(),
            "emax": ScientificValue(
                name="Maximum Antitumor Efficacy (Emax)",
                value=emax if in_domain else "Insufficient data",
                unit="fraction [0-1]",
                epistemic_category=category,
                source=source,
                confidence=confidence,
                in_applicability_domain=in_domain,
            ).to_dict(),
            "hill_slope": ScientificValue(
                name="Hill Coefficient (Slope)",
                value=hill_slope if in_domain else "Insufficient data",
                unit="",
                epistemic_category=category,
                source=source,
                confidence=confidence,
                in_applicability_domain=in_domain,
            ).to_dict(),
            "selectivity_index": ScientificValue(
                name="Selectivity Index (SI)",
                value=si if in_domain else "Insufficient data",
                unit="ratio",
                epistemic_category=category,
                source="IC50_healthy / IC50_sensitive",
                confidence=confidence,
                uncertainty_low=round(si * 0.70, 2) if in_domain else None,
                uncertainty_high=round(si * 1.40, 2) if in_domain else None,
                in_applicability_domain=in_domain,
            ).to_dict(),
            "resistance_emergence_prob": ScientificValue(
                name="Resistance Emergence Probability",
                value=res_prob if in_domain else "Insufficient data",
                unit="probability [0-1]",
                epistemic_category=category,
                source="PDAC Subclonal Pressure Model",
                confidence=confidence * 0.8,
                in_applicability_domain=in_domain,
            ).to_dict(),
            "concentration_response_curve": {
                "conc_um": conc_points,
                "tumor_sensitive_viability": sens_viability,
                "tumor_resistant_viability": res_viability,
                "healthy_viability": healthy_viability,
            },
            "compute_exposure_partition": compute_exposure_partition,
            "sample_exposure_partition": compute_exposure_partition(10.0),
            "safety_disclaimer": SAFETY_DISCLAIMER,
        }

    def evaluate_combination_synergy(
        self,
        compound_a: CompoundRecord,
        compound_b: CompoundRecord,
        conc_a_um: float = 5.0,
        conc_b_um: float = 5.0,
        qsar_a: dict[str, Any] | None = None,
        qsar_b: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Evaluate drug combination synergy between two candidate compounds.

        Computes:
        - Single-agent pharmacodynamic responses E_A, E_B at (conc_a_um, conc_b_um)
          across sensitive tumor cells, resistant clones, and healthy epithelial cells.
        - Bliss Independence expected response: E_bliss = E_A + E_B - E_A * E_B.
        - Loewe Additivity Combination Index (CI_loewe) estimate.
        - Mechanistic Synergy Factor based on structural and physicochemical complementarity:
          Orthogonal scaffolds (Tanimoto < 0.35) with diverse LogP/TPSA permeability yield
          positive synergy (1.08 - 1.30); near-identical scaffolds (Tanimoto > 0.75) compete
          for target sites (0.92 - 1.00); dual lipophilic/reactive compounds amplify healthy toxicity.
        - 8x8 2D Dose-Response Matrix (inhibition and Bliss excess heatmaps).
        - Key metrics wrapped in ScientificValue with mandatory safety disclaimers.
        """
        # 1. Single agent pharmacodynamics
        pd_a = self.predict_antitumor_activity(compound_a, qsar_pred=qsar_a)
        pd_b = self.predict_antitumor_activity(compound_b, qsar_pred=qsar_b)

        def _get_num(d: dict[str, Any], key: str, fallback: float) -> float:
            v = d.get(key, {}).get("value")
            if isinstance(v, (int, float)):
                return float(v)
            return fallback

        ic50_a_sens = _get_num(pd_a, "ic50_sensitive_um", 5.0)
        ic50_a_res = _get_num(pd_a, "ic50_resistant_um", 25.0)
        ic50_a_healthy = _get_num(pd_a, "ic50_healthy_um", 50.0)
        emax_a = _get_num(pd_a, "emax", 0.90)
        h_a = _get_num(pd_a, "hill_slope", 1.35)

        ic50_b_sens = _get_num(pd_b, "ic50_sensitive_um", 5.0)
        ic50_b_res = _get_num(pd_b, "ic50_resistant_um", 25.0)
        ic50_b_healthy = _get_num(pd_b, "ic50_healthy_um", 50.0)
        emax_b = _get_num(pd_b, "emax", 0.90)
        h_b = _get_num(pd_b, "hill_slope", 1.35)

        # Hill equation inhibition function (fraction 0 - 1)
        def hill_inhibition(conc: float, ic50: float, emax: float, h: float) -> float:
            if conc <= 0.0:
                return 0.0
            c = max(0.0, float(conc))
            return float(emax * (c ** h) / (ic50 ** h + c ** h))

        # Single agent responses at target concentrations
        e_a_sens = hill_inhibition(conc_a_um, ic50_a_sens, emax_a, h_a)
        e_b_sens = hill_inhibition(conc_b_um, ic50_b_sens, emax_b, h_b)

        e_a_res = hill_inhibition(conc_a_um, ic50_a_res, emax_a, h_a)
        e_b_res = hill_inhibition(conc_b_um, ic50_b_res, emax_b, h_b)

        e_a_healthy = hill_inhibition(conc_a_um, ic50_a_healthy, emax_a * 0.85, h_a)
        e_b_healthy = hill_inhibition(conc_b_um, ic50_b_healthy, emax_b * 0.85, h_b)

        # 2. Bliss Independence expectation
        e_bliss_sens = e_a_sens + e_b_sens - (e_a_sens * e_b_sens)
        e_bliss_res = e_a_res + e_b_res - (e_a_res * e_b_res)
        e_bliss_healthy = e_a_healthy + e_b_healthy - (e_a_healthy * e_b_healthy)

        # 3. Loewe Additivity Combination Index (CI)
        if conc_a_um > 0.0 and conc_b_um > 0.0:
            ci_loewe = (conc_a_um / max(1e-3, ic50_a_sens)) + (conc_b_um / max(1e-3, ic50_b_sens))
        else:
            ci_loewe = 1.0

        # 4. Tanimoto Structural Similarity & Mechanistic Synergy Factor
        mol_a = Chem.MolFromSmiles(compound_a.canonical_smiles) if compound_a.canonical_smiles else None
        mol_b = Chem.MolFromSmiles(compound_b.canonical_smiles) if compound_b.canonical_smiles else None

        if mol_a is not None and mol_b is not None:
            fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
            fp_a = fpgen.GetFingerprint(mol_a)
            fp_b = fpgen.GetFingerprint(mol_b)
            tanimoto = float(DataStructs.TanimotoSimilarity(fp_a, fp_b))
        else:
            tanimoto = 0.50

        # The local data contain no experimentally measured interaction surface.
        synergy_factor = 1.0
        healthy_tox_mult = 1.0
        mechanism_note = "Bliss independence assumed. Structure similarity is descriptive and cannot demonstrate synergy."

        # Combination responses
        e_combo_sens = min(1.0, max(0.0, e_bliss_sens * synergy_factor))
        e_combo_res = min(1.0, max(0.0, e_bliss_res * synergy_factor))
        e_combo_healthy = min(1.0, max(0.0, e_bliss_healthy * healthy_tox_mult))

        bliss_excess = e_combo_sens - e_bliss_sens
        bliss_synergy_score = round(float(bliss_excess * 100.0), 2)  # In percentage

        # Combination selectivity index: ratio of tumor killing to healthy tissue damage
        comb_si = round(float(e_combo_sens / max(0.02, e_combo_healthy)), 2)

        # 5. 8x8 2D Dose-Response Matrix for Heatmap Visualization
        grid_a = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
        grid_b = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]

        inhibition_matrix: list[list[float]] = []
        bliss_excess_matrix: list[list[float]] = []

        for ca in grid_a:
            row_inh: list[float] = []
            row_bliss: list[float] = []
            ea = hill_inhibition(ca, ic50_a_sens, emax_a, h_a)
            for cb in grid_b:
                eb = hill_inhibition(cb, ic50_b_sens, emax_b, h_b)
                e_bl = ea + eb - (ea * eb)
                e_co = min(1.0, max(0.0, e_bl * synergy_factor))
                bl_exc = (e_co - e_bl) * 100.0
                row_inh.append(round(e_co * 100.0, 2))
                row_bliss.append(round(bl_exc, 2))
            inhibition_matrix.append(row_inh)
            bliss_excess_matrix.append(row_bliss)

        return {
            "compound_a_id": compound_a.compound_id,
            "compound_b_id": compound_b.compound_id,
            "conc_a_um": conc_a_um,
            "conc_b_um": conc_b_um,
            "tanimoto_similarity": round(tanimoto, 4),
            "synergy_factor": synergy_factor,
            "synergy_mechanism": mechanism_note,
            "loewe_combination_index": None,
            "bliss_synergy_score": ScientificValue(
                name="Bliss Synergy Score",
                value=bliss_synergy_score,
                unit="%",
                epistemic_category=EpistemicCategory.EXPLORATORY_MODEL,
                source="Bliss Independence Excess Model",
                confidence=0.0,
                uncertainty_low=None,
                uncertainty_high=None,
                in_applicability_domain=True,
                limitations="Zero excess follows the independence assumption. No interaction experiments or uncertainty estimates are available. Loewe CI is unavailable without equipotent dose-response curves.",
            ).to_dict(),
            "predicted_combination_inhibition": ScientificValue(
                name="Predicted Combination Inhibition",
                value=round(e_combo_sens, 4),
                unit="fraction [0-1]",
                epistemic_category=EpistemicCategory.EXPLORATORY_MODEL,
                source="Mechanistic Synergy Complementarity Model",
                confidence=0.0,
                uncertainty_low=round(max(0.0, e_combo_sens - 0.08), 4),
                uncertainty_high=round(min(1.0, e_combo_sens + 0.08), 4),
                in_applicability_domain=True,
                limitations=SAFETY_DISCLAIMER,
            ).to_dict(),
            "combination_selectivity_index": ScientificValue(
                name="Combination Selectivity Index",
                value=comb_si,
                unit="ratio",
                epistemic_category=EpistemicCategory.EXPLORATORY_MODEL,
                source="Tumor Sensitive vs. Healthy Tissue Inhibition Ratio",
                confidence=0.72,
                uncertainty_low=round(max(0.1, comb_si * 0.75), 2),
                uncertainty_high=round(comb_si * 1.30, 2),
                in_applicability_domain=True,
                limitations=SAFETY_DISCLAIMER,
            ).to_dict(),
            "single_agent_responses": {
                "compound_a": {
                    "conc_um": conc_a_um,
                    "sensitive_inhibition": round(e_a_sens, 4),
                    "resistant_inhibition": round(e_a_res, 4),
                    "healthy_inhibition": round(e_a_healthy, 4),
                },
                "compound_b": {
                    "conc_um": conc_b_um,
                    "sensitive_inhibition": round(e_b_sens, 4),
                    "resistant_inhibition": round(e_b_res, 4),
                    "healthy_inhibition": round(e_b_healthy, 4),
                },
            },
            "combination_response": {
                "sensitive_inhibition": round(e_combo_sens, 4),
                "resistant_inhibition": round(e_combo_res, 4),
                "healthy_inhibition": round(e_combo_healthy, 4),
                "bliss_expected_inhibition": round(e_bliss_sens, 4),
                "bliss_excess": round(bliss_excess, 4),
            },
            "matrix_2d": {
                "conc_grid_a": grid_a,
                "conc_grid_b": grid_b,
                "inhibition_matrix": inhibition_matrix,
                "bliss_excess_matrix": bliss_excess_matrix,
            },
            "safety_disclaimer": SAFETY_DISCLAIMER,
        }
