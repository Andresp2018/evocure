"""
Toxicology Evaluation Engine for EVOCURE.

Implements Section 11.3 computational toxicology:
- RDKit SMARTS structural alert screening for reactive/toxicophores (nitroaromatics, epoxides,
  aziridines, acyl halides, Michael acceptors, hydrazines, quinones, polyhalogenated aliphatics,
  basic lipophilic amine hERG motif, and thiophene/furan reactive metabolites).
- Physicochemical liability rules (Pfizer 3/75 rule, lipophilicity burden, formal charge extremes).
- Experimental bioactivity reference integration (DILI, Ames test, hERG IC50, healthy cytotoxicity).
- 10 mandatory toxicological categories wrapped in ScientificValue.to_dict().
- Applicability domain enforcement (MW > 1200 or heavy_atoms < 3 yields Insufficient Data).
- Internal computational risk index (0-100) strictly labeled for exploratory ranking only.
"""

from __future__ import annotations

import math
from typing import Any

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski

from simulation.core.contracts import (
    CompoundRecord,
    ScientificValue,
    EpistemicCategory,
)

# Structural alert SMARTS patterns
ALERT_SMARTS: dict[str, str] = {
    "nitroaromatic": "[$([NX3](=O)=O),$([N+](=O)[O-])]c",
    "epoxide": "[C;R1]1[O;R1][C;R1]1",
    "aziridine": "[C;R1]1[N;R1][C;R1]1",
    "acyl_halide": "[CX3](=[OX1])[F,Cl,Br,I]",
    "michael_acceptor": "[CX3]=[CX3]-[CX3]=[O,S]",
    "hydrazine": "[NX3;!$(NC=O)][NX3;!$(NC=O)]",
    "quinone": "O=C1C=CC(=O)C=C1",
    "polyhalogenated": "[CX4]([F,Cl,Br,I])([F,Cl,Br,I])[F,Cl,Br,I]",
    "basic_aliphatic_amine": "[NX3;H2,H1,H0;!$(NC=O);!$(Nc)]",
    "thiophene": "c1ccsc1",
    "furan": "c1ccoc1",
    "pyridine": "c1ccncc1",
    "imidazole": "c1c[nH]cn1",
}

ALERT_DESCRIPTIONS: dict[str, str] = {
    "nitroaromatic": "Nitroaromatic moiety: mutagenicity, Ames positive liability, nitroreduction bioactivation.",
    "epoxide": "Epoxide ring: reactive alkylating electrophile, genotoxicity/hepatotoxicity risk.",
    "aziridine": "Aziridine ring: reactive nitrogen alkylating electrophile, DNA adduction liability.",
    "acyl_halide": "Acyl halide: highly reactive acylating species, non-specific protein acylation.",
    "michael_acceptor": "Michael acceptor (alpha,beta-unsaturated carbonyl): soft electrophile, covalent thiol adduction.",
    "hydrazine": "Hydrazine / hydrazide: oxidative DNA cleavage, reactive radical bioactivation, liver injury.",
    "quinone": "Quinone / quinone-imine: redox cycling, reactive oxygen species (ROS) formation, glutathione depletion.",
    "polyhalogenated": "Tri- or poly-halogenated aliphatic: cytochrome P450 haloalkane bioactivation, nephrotoxicity.",
    "basic_aliphatic_amine": "Basic aliphatic amine: potential hERG channel interacting motif when combined with lipophilic aromatics.",
    "thiophene": "Thiophene ring: cytochrome P450 oxidation to reactive thiophene sulfoxide / epoxide intermediates.",
    "furan": "Furan ring: cytochrome P450 oxidation to reactive cis-2-butene-1,4-dial intermediates.",
    "pyridine": "Pyridine ring: potential Cytochrome P450 (e.g. CYP3A4) coordination and inhibition.",
    "imidazole": "Imidazole ring: potent heme-iron Cytochrome P450 coordination and drug-drug interaction risk.",
}

RISK_INDEX_LABEL: str = (
    "Internal computational risk score (0-100) for exploratory ranking only; "
    "NOT a clinical human safety measurement."
)


class ToxicologyEngine:
    """
    10-category multi-organ toxicological risk assessment engine.
    Wraps every evaluation in ScientificValue conforming to EVOCURE epistemic boundaries.
    """

    def __init__(self, model_version: str = "0.1.0") -> None:
        self.model_version = model_version
        self._compiled_patterns: dict[str, Chem.Mol | None] = {}
        for name, smarts in ALERT_SMARTS.items():
            self._compiled_patterns[name] = Chem.MolFromSmarts(smarts)

    def screen_structural_alerts(self, mol: Chem.Mol | None) -> list[dict[str, str]]:
        """Identify matched toxicophoric substructures."""
        if mol is None:
            return []

        detected: list[dict[str, str]] = []
        for name, pattern in self._compiled_patterns.items():
            if pattern is not None and mol.HasSubstructMatch(pattern):
                detected.append({
                    "alert_id": name,
                    "description": ALERT_DESCRIPTIONS.get(name, "Structural toxicophore alert"),
                })
        return detected

    def evaluate_toxicology(
        self,
        compound: CompoundRecord,
        in_domain: bool = True,
    ) -> dict[str, Any]:
        """
        Evaluate all 10 mandatory toxicology categories:
        1. non_tumor_cytotoxicity
        2. hepatotoxicity
        3. nephrotoxicity
        4. cardiotoxicity
        5. neurotoxicity
        6. genotoxicity
        7. hematological_toxicity
        8. gastrointestinal_toxicity
        9. interaction_risk
        10. global_uncertainty

        When in_domain is False (or MW > 1200 / heavy_atoms < 3),
        returns value='Insufficient data' and EpistemicCategory.INSUFFICIENT_DATA.
        """
        mol = Chem.MolFromSmiles(compound.canonical_smiles) if compound.canonical_smiles else None

        # Check applicability domain
        mw = float(compound.mol_weight)
        heavy_atoms = int(compound.heavy_atoms)
        if mol is not None:
            mw = float(Descriptors.MolWt(mol))
            heavy_atoms = int(mol.GetNumHeavyAtoms())

        if mw > 1200.0 or heavy_atoms < 3 or not in_domain:
            in_domain = False

        # If outside domain, strictly return INSUFFICIENT_DATA for all 10 categories
        if not in_domain:
            insufficient_val = ScientificValue(
                name="Insufficient Data",
                value="Insufficient data",
                unit="",
                epistemic_category=EpistemicCategory.INSUFFICIENT_DATA,
                source="EVOCURE Applicability Domain Boundary",
                confidence=0.0,
                in_applicability_domain=False,
                limitations="Compound structure lies outside the verified applicability domain of the toxicology engine.",
            ).to_dict()

            return {
                "non_tumor_cytotoxicity": insufficient_val,
                "hepatotoxicity": insufficient_val,
                "nephrotoxicity": insufficient_val,
                "cardiotoxicity": insufficient_val,
                "neurotoxicity": insufficient_val,
                "genotoxicity": insufficient_val,
                "hematological_toxicity": insufficient_val,
                "gastrointestinal_toxicity": insufficient_val,
                "interaction_risk": insufficient_val,
                "global_uncertainty": ScientificValue(
                    name="Global Prediction Uncertainty",
                    value=1.0,
                    unit="score [0-1]",
                    epistemic_category=EpistemicCategory.INSUFFICIENT_DATA,
                    source="Out of Applicability Domain",
                    confidence=0.0,
                    in_applicability_domain=False,
                ).to_dict(),
                "internal_computational_risk_index_0_100": None,
                "risk_index_label": RISK_INDEX_LABEL,
                "detected_alerts": [],
                "applicability_domain_in": False,
            }

        # In-domain evaluation
        logp = float(compound.logp)
        tpsa = float(compound.tpsa)
        hbd = int(compound.hbd)
        formal_charge = int(compound.formal_charge)
        if mol is not None:
            logp = float(Crippen.MolLogP(mol))
            tpsa = float(Descriptors.TPSA(mol))
            hbd = int(Lipinski.NumHDonors(mol))
            formal_charge = int(Chem.GetFormalCharge(mol))

        detected_alerts = self.screen_structural_alerts(mol)
        alert_ids = {a["alert_id"] for a in detected_alerts}

        ref_bio = compound.reference_bioactivity or {}

        # Physicochemical risk flags
        # Pfizer 3/75 rule: LogP > 3.0 and TPSA < 75 A^2 (increases promiscuous toxicity)
        pfizer_3_75 = (logp > 3.0) and (tpsa < 75.0)
        high_lipophilicity = logp > 4.5
        extreme_charge = abs(formal_charge) >= 2

        # -------------------------------------------------------------
        # 1. Non-Tumor Cytotoxicity
        # -------------------------------------------------------------
        has_exp_healthy_ic50 = "cytotoxicity_healthy_ic50_um" in ref_bio
        if has_exp_healthy_ic50:
            exp_ic50 = float(ref_bio["cytotoxicity_healthy_ic50_um"])
            cyto_score = min(1.0, max(0.05, 10.0 / max(0.1, exp_ic50)))
            cyto_cat = EpistemicCategory.EXPERIMENTAL_OBSERVATION
            cyto_source = "Imported Healthy Cell Cytotoxicity Assay"
            cyto_conf = 0.95
        else:
            # Electrophiles and high lipophilicity drive non-specific cytotoxicity
            base_cyto = 0.15
            if "epoxide" in alert_ids or "aziridine" in alert_ids or "acyl_halide" in alert_ids:
                base_cyto += 0.40
            if "michael_acceptor" in alert_ids:
                base_cyto += 0.25
            if pfizer_3_75:
                base_cyto += 0.15
            cyto_score = round(min(1.0, base_cyto), 3)
            cyto_cat = EpistemicCategory.EXPLORATORY_MODEL
            cyto_source = "Reactive Electrophile & Desolvation Heuristic"
            cyto_conf = 0.70

        non_tumor_cytotoxicity = ScientificValue(
            name="Non-Tumor Cytotoxicity Risk",
            value=cyto_score,
            unit="risk score [0-1]",
            epistemic_category=cyto_cat,
            source=cyto_source,
            confidence=cyto_conf,
            uncertainty_low=round(max(0.0, cyto_score - 0.20), 2),
            uncertainty_high=round(min(1.0, cyto_score + 0.20), 2),
            in_applicability_domain=True,
        ).to_dict()

        # -------------------------------------------------------------
        # 2. Hepatotoxicity (DILI)
        # -------------------------------------------------------------
        has_dili_ref = "hepatotox_dili" in ref_bio
        if has_dili_ref:
            dili_flag = bool(ref_bio["hepatotox_dili"])
            hep_score = 0.85 if dili_flag else 0.15
            hep_cat = EpistemicCategory.EXPERIMENTAL_OBSERVATION
            hep_source = "Reference DILI Annotation"
            hep_conf = 0.92
        else:
            hep_risk = 0.10
            # Reactive metabolites are chief drivers of DILI
            if "epoxide" in alert_ids or "hydrazine" in alert_ids:
                hep_risk += 0.35
            if "thiophene" in alert_ids or "furan" in alert_ids:
                hep_risk += 0.25
            if "quinone" in alert_ids:
                hep_risk += 0.25
            if "michael_acceptor" in alert_ids:
                hep_risk += 0.20
            if pfizer_3_75 or high_lipophilicity:
                hep_risk += 0.15
            hep_score = round(min(1.0, hep_risk), 3)
            hep_cat = EpistemicCategory.EXPLORATORY_MODEL
            hep_source = "CYP Bioactivation & Reactive Metabolite Heuristic"
            hep_conf = 0.68

        hepatotoxicity = ScientificValue(
            name="Hepatotoxicity / DILI Risk",
            value=hep_score,
            unit="risk score [0-1]",
            epistemic_category=hep_cat,
            source=hep_source,
            confidence=hep_conf,
            uncertainty_low=round(max(0.0, hep_score - 0.22), 2),
            uncertainty_high=round(min(1.0, hep_score + 0.22), 2),
            in_applicability_domain=True,
        ).to_dict()

        # -------------------------------------------------------------
        # 3. Nephrotoxicity
        # -------------------------------------------------------------
        neph_risk = 0.08
        if "polyhalogenated" in alert_ids:
            neph_risk += 0.40
        if "acyl_halide" in alert_ids:
            neph_risk += 0.30
        if mw > 450.0 and logp < -1.0:  # High renal elimination of large polar species
            neph_risk += 0.20
        if extreme_charge:
            neph_risk += 0.15
        neph_score = round(min(1.0, neph_risk), 3)

        nephrotoxicity = ScientificValue(
            name="Nephrotoxicity Risk",
            value=neph_score,
            unit="risk score [0-1]",
            epistemic_category=EpistemicCategory.EXPLORATORY_MODEL,
            source="Renal Tubular Exposure & Haloalkane Bioactivation Model",
            confidence=0.65,
            uncertainty_low=round(max(0.0, neph_score - 0.20), 2),
            uncertainty_high=round(min(1.0, neph_score + 0.20), 2),
            in_applicability_domain=True,
        ).to_dict()

        # -------------------------------------------------------------
        # 4. Cardiotoxicity (hERG / QT Prolongation)
        # -------------------------------------------------------------
        has_herg_ref = "cardiotox_herg_ic50_um" in ref_bio
        if has_herg_ref:
            herg_ic50 = float(ref_bio["cardiotox_herg_ic50_um"])
            # IC50 < 1 uM: high risk, 1-10 uM: moderate, > 10 uM: low
            cardio_score = round(min(1.0, max(0.05, 1.0 / (1.0 + herg_ic50 / 3.0))), 3)
            cardio_cat = EpistemicCategory.EXPERIMENTAL_OBSERVATION
            cardio_source = "Imported In Vitro hERG Patch-Clamp Bioactivity"
            cardio_conf = 0.95
        else:
            # Basic aliphatic amine + lipophilic aromatic framework is classic hERG pharmacophore
            cardio_risk = 0.08
            is_basic_amine = "basic_aliphatic_amine" in alert_ids
            aromatic_rings = 0
            if mol is not None:
                aromatic_rings = Descriptors.NumAromaticRings(mol)

            if is_basic_amine and logp > 3.0 and aromatic_rings >= 2:
                cardio_risk += 0.65
            elif is_basic_amine and logp > 2.0:
                cardio_risk += 0.35
            elif high_lipophilicity and aromatic_rings >= 3:
                cardio_risk += 0.25

            cardio_score = round(min(1.0, cardio_risk), 3)
            cardio_cat = EpistemicCategory.EXPLORATORY_MODEL
            cardio_source = "Lipophilic Amine hERG Pharmacophore Heuristic"
            cardio_conf = 0.72

        cardiotoxicity = ScientificValue(
            name="Cardiotoxicity / hERG Liability",
            value=cardio_score,
            unit="risk score [0-1]",
            epistemic_category=cardio_cat,
            source=cardio_source,
            confidence=cardio_conf,
            uncertainty_low=round(max(0.0, cardio_score - 0.25), 2),
            uncertainty_high=round(min(1.0, cardio_score + 0.25), 2),
            in_applicability_domain=True,
        ).to_dict()

        # -------------------------------------------------------------
        # 5. Neurotoxicity
        # -------------------------------------------------------------
        # CNS penetration features: LogP 1.5 - 3.5, TPSA < 70, HBD <= 1, MW < 400
        neuro_risk = 0.05
        cns_permeable = (1.5 <= logp <= 3.8) and (tpsa <= 65.0) and (hbd <= 1) and (mw <= 420.0)
        if cns_permeable:
            neuro_risk += 0.30
            if "basic_aliphatic_amine" in alert_ids:
                neuro_risk += 0.25
        if "hydrazine" in alert_ids:
            neuro_risk += 0.25
        neuro_score = round(min(1.0, neuro_risk), 3)

        neurotoxicity = ScientificValue(
            name="Neurotoxicity / CNS Penetration Risk",
            value=neuro_score,
            unit="risk score [0-1]",
            epistemic_category=EpistemicCategory.EXPLORATORY_MODEL,
            source="Blood-Brain Barrier (BBB) Permeability & Amine Motif Model",
            confidence=0.62,
            uncertainty_low=round(max(0.0, neuro_score - 0.20), 2),
            uncertainty_high=round(min(1.0, neuro_score + 0.20), 2),
            in_applicability_domain=True,
        ).to_dict()

        # -------------------------------------------------------------
        # 6. Genotoxicity (Mutagenicity / DNA Reactivity)
        # -------------------------------------------------------------
        has_ames_ref = "ames_mutagenicity" in ref_bio
        if has_ames_ref:
            is_ames_pos = bool(ref_bio["ames_mutagenicity"])
            geno_score = 0.90 if is_ames_pos else 0.05
            geno_cat = EpistemicCategory.EXPERIMENTAL_OBSERVATION
            geno_source = "Imported Ames Salmonella Mutagenicity Assay"
            geno_conf = 0.95
        else:
            geno_risk = 0.05
            # Cohort of concern alerts for genotoxicity
            if "nitroaromatic" in alert_ids:
                geno_risk += 0.60
            if "epoxide" in alert_ids:
                geno_risk += 0.55
            if "aziridine" in alert_ids:
                geno_risk += 0.65
            if "hydrazine" in alert_ids:
                geno_risk += 0.45
            if "acyl_halide" in alert_ids:
                geno_risk += 0.35
            if "quinone" in alert_ids:
                geno_risk += 0.30
            geno_score = round(min(1.0, geno_risk), 3)
            geno_cat = EpistemicCategory.EXPLORATORY_MODEL
            geno_source = "Hansen / Ashby-Tennant Mutagenic Alert Screener"
            geno_conf = 0.80

        genotoxicity = ScientificValue(
            name="Genotoxicity / Ames Mutagenicity Risk",
            value=geno_score,
            unit="risk score [0-1]",
            epistemic_category=geno_cat,
            source=geno_source,
            confidence=geno_conf,
            uncertainty_low=round(max(0.0, geno_score - 0.18), 2),
            uncertainty_high=round(min(1.0, geno_score + 0.18), 2),
            in_applicability_domain=True,
        ).to_dict()

        # -------------------------------------------------------------
        # 7. Hematological Toxicity (Bone Marrow Suppression)
        # -------------------------------------------------------------
        hem_risk = 0.08
        if "epoxide" in alert_ids or "aziridine" in alert_ids:
            hem_risk += 0.45
        if "quinone" in alert_ids:
            hem_risk += 0.25
        if "hydrazine" in alert_ids:
            hem_risk += 0.25
        if cyto_score > 0.5:
            hem_risk += 0.20
        hem_score = round(min(1.0, hem_risk), 3)

        hematological_toxicity = ScientificValue(
            name="Hematological Toxicity / Myelosuppression Risk",
            value=hem_score,
            unit="risk score [0-1]",
            epistemic_category=EpistemicCategory.EXPLORATORY_MODEL,
            source="Bone Marrow Progenitor Sensitivity Heuristic",
            confidence=0.66,
            uncertainty_low=round(max(0.0, hem_score - 0.22), 2),
            uncertainty_high=round(min(1.0, hem_score + 0.22), 2),
            in_applicability_domain=True,
        ).to_dict()

        # -------------------------------------------------------------
        # 8. Gastrointestinal Toxicity
        # -------------------------------------------------------------
        gi_risk = 0.10
        if extreme_charge:
            gi_risk += 0.30
        if "michael_acceptor" in alert_ids:
            gi_risk += 0.25
        if "acyl_halide" in alert_ids:
            gi_risk += 0.35
        if cyto_score > 0.4:
            gi_risk += 0.15
        gi_score = round(min(1.0, gi_risk), 3)

        gastrointestinal_toxicity = ScientificValue(
            name="Gastrointestinal Toxicity / Mucositis Risk",
            value=gi_score,
            unit="risk score [0-1]",
            epistemic_category=EpistemicCategory.EXPLORATORY_MODEL,
            source="Intestinal Epithelial Barrier Permeation & Irritation Model",
            confidence=0.64,
            uncertainty_low=round(max(0.0, gi_score - 0.20), 2),
            uncertainty_high=round(min(1.0, gi_score + 0.20), 2),
            in_applicability_domain=True,
        ).to_dict()

        # -------------------------------------------------------------
        # 9. Interaction Risk (CYP / Transporter DDI)
        # -------------------------------------------------------------
        ddi_risk = 0.08
        if "imidazole" in alert_ids:
            ddi_risk += 0.45  # Classic potent CYP3A4 inhibitor motif
        if "pyridine" in alert_ids:
            ddi_risk += 0.25
        if high_lipophilicity:
            ddi_risk += 0.20
        if pfizer_3_75:
            ddi_risk += 0.15
        ddi_score = round(min(1.0, ddi_risk), 3)

        interaction_risk = ScientificValue(
            name="Metabolic Drug-Drug Interaction Risk (CYP/Transporter)",
            value=ddi_score,
            unit="risk score [0-1]",
            epistemic_category=EpistemicCategory.EXPLORATORY_MODEL,
            source="Cytochrome P450 Coordination & Lipophilic Inhibition Heuristic",
            confidence=0.70,
            uncertainty_low=round(max(0.0, ddi_score - 0.22), 2),
            uncertainty_high=round(min(1.0, ddi_score + 0.22), 2),
            in_applicability_domain=True,
        ).to_dict()

        # -------------------------------------------------------------
        # 10. Global Uncertainty
        # -------------------------------------------------------------
        # Driven by lack of experimental reference data, number of alerts, and molecular size
        exp_count = sum(1 for k in ["cytotoxicity_healthy_ic50_um", "hepatotox_dili", "cardiotox_herg_ic50_um", "ames_mutagenicity"] if k in ref_bio)
        alert_count = len(detected_alerts)

        uncertainty_base = 0.35 - (0.07 * exp_count) + (0.05 * alert_count)
        global_unc_val = round(max(0.05, min(0.95, uncertainty_base)), 3)

        global_uncertainty = ScientificValue(
            name="Global Toxicological Prediction Uncertainty",
            value=global_unc_val,
            unit="uncertainty score [0-1]",
            epistemic_category=EpistemicCategory.EXPLORATORY_MODEL,
            source="Multi-Category Evidence & Alert Entropy Composite",
            confidence=0.85,
            in_applicability_domain=True,
        ).to_dict()

        # -------------------------------------------------------------
        # Composite Internal Computational Risk Index (0 - 100)
        # -------------------------------------------------------------
        # Weights summing to 100
        weights = {
            "genotoxicity": 18.0,
            "cardiotoxicity": 16.0,
            "hepatotoxicity": 16.0,
            "hematological_toxicity": 12.0,
            "nephrotoxicity": 10.0,
            "non_tumor_cytotoxicity": 10.0,
            "neurotoxicity": 6.0,
            "gastrointestinal_toxicity": 6.0,
            "interaction_risk": 6.0,
        }

        weighted_sum = (
            weights["genotoxicity"] * geno_score
            + weights["cardiotoxicity"] * cardio_score
            + weights["hepatotoxicity"] * hep_score
            + weights["hematological_toxicity"] * hem_score
            + weights["nephrotoxicity"] * neph_score
            + weights["non_tumor_cytotoxicity"] * cyto_score
            + weights["neurotoxicity"] * neuro_score
            + weights["gastrointestinal_toxicity"] * gi_score
            + weights["interaction_risk"] * ddi_score
        )
        composite_index = round(float(max(0.0, min(100.0, weighted_sum))), 1)

        return {
            "non_tumor_cytotoxicity": non_tumor_cytotoxicity,
            "hepatotoxicity": hepatotoxicity,
            "nephrotoxicity": nephrotoxicity,
            "cardiotoxicity": cardiotoxicity,
            "neurotoxicity": neurotoxicity,
            "genotoxicity": genotoxicity,
            "hematological_toxicity": hematological_toxicity,
            "gastrointestinal_toxicity": gastrointestinal_toxicity,
            "interaction_risk": interaction_risk,
            "global_uncertainty": global_uncertainty,
            "internal_computational_risk_index_0_100": composite_index,
            "risk_index_label": RISK_INDEX_LABEL,
            "detected_alerts": detected_alerts,
            "applicability_domain_in": True,
        }
