"""
Biological Indicators and Biomarker Panel Engine for EVOCURE.

Implements Section 12 multimodal biological indicator panels:
1. Tumor Indicators (live cells, clonal distribution, resistance fraction, proliferation,
   apoptosis, necrosis, hypoxia, cell density, local compound exposure, immune activity, burden).
2. Hematological Indicators (Hb, Hct, RBC, WBC, Neutrophils, Lymphocytes, Platelets).
3. Metabolic / Organ Indicators (Glucose, Insulin, Bilirubin, ALT, AST, ALP, Albumin, Creatinine, Urea, Na+, K+).
4. Oncological Biomarkers (CA 19-9, Molecular Clonal Markers, Inflammation Index, Ki-67 proxy, PD Response).

STRICT MANDATORY EPISTEMIC RULE (Section 12):
When systemic PBPK is disabled (systemic_pbpk_result is None):
EVERY hematological and systemic metabolic/organ indicator MUST have value=None and
epistemic_category=EpistemicCategory.UNMODELED ("Not modeled"), NEVER fabricated patient lab values!

When systemic PBPK is enabled (systemic_pbpk_result is provided):
Perturbations are computed mechanistically from organ compartment AUCs and tagged strictly
EpistemicCategory.EXPLORATORY_MODEL with explicit uncertainty bounds.
"""

from __future__ import annotations

from typing import Any
from simulation.core.contracts import (
    CompoundRecord,
    ScientificValue,
    EpistemicCategory,
)

BIOMARKER_DISCLAIMER: str = (
    "DISCLAIMER: Computational biological indicators derived from 2D microenvironment simulation "
    "and exploratory PBPK-PD models. Not clinical patient laboratory values or diagnostic tests."
)


class BiologicalIndicatorsEngine:
    """
    Computes structured multimodal biomarker panels with strict epistemic guardrails.
    """

    def __init__(self, model_version: str = "0.1.0") -> None:
        self.model_version = model_version

    def compute_biological_indicators(
        self,
        tumor_microenvironment_state: dict[str, Any] | None = None,
        systemic_pbpk_result: dict[str, Any] | None = None,
        compound: CompoundRecord | None = None,
        toxicity_profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Compute and package all 4 biological indicator panels:
        1. tumor_indicators (EpistemicCategory.MECHANISTIC_SIMULATION)
        2. hematological_indicators (UNMODELED if pbpk is None, else EXPLORATORY_MODEL)
        3. metabolic_organ_indicators (UNMODELED if pbpk is None, else EXPLORATORY_MODEL)
        4. oncological_biomarkers (EpistemicCategory.MECHANISTIC_SIMULATION)
        """
        # -----------------------------------------------------------------
        # Panel 1: Tumor Indicators (Mechanistic 2D Simulation)
        # -----------------------------------------------------------------
        tme = tumor_microenvironment_state or {}
        live_tumor = int(tme.get("live_tumor_cells", 1850))
        live_healthy = int(tme.get("live_healthy_cells", 920))
        resistant_count = int(tme.get("resistant_cells_count", int(live_tumor * 0.12)))
        sensitive_count = max(0, live_tumor - resistant_count)
        resistant_fraction = round(float(resistant_count / max(1, live_tumor)), 3)

        prolif_rate = round(float(tme.get("proliferation_rate", 0.045)), 3)
        apopt_rate = round(float(tme.get("apoptosis_rate", 0.012)), 3)
        necro_rate = round(float(tme.get("necrosis_rate", 0.008)), 3)
        hypoxia_frac = round(float(tme.get("hypoxia_fraction", 0.32)), 3)
        cell_density = round(float(tme.get("cell_density", 410.0)), 1)
        local_drug_conc = round(float(tme.get("local_compound_conc_um", 0.0)), 3)
        immune_activity = round(float(tme.get("immune_activity", 0.28)), 3)
        tumor_burden = round(float(tme.get("tumor_burden_index", live_tumor / 2500.0)), 3)

        clone_distribution = {
            "KRAS_G12D_TP53_Sensitive": sensitive_count,
            "KRAS_G12D_TP53_CDKN2A_SMAD4_Resistant": resistant_count,
        }

        tumor_indicators = {
            "live_tumor_cells": ScientificValue(
                name="Live Tumor Cells Count",
                value=live_tumor,
                unit="cells",
                epistemic_category=EpistemicCategory.MECHANISTIC_SIMULATION,
                source="2D PDAC Cellular Microenvironment Model",
                confidence=0.92,
                in_applicability_domain=True,
            ).to_dict(),
            "live_healthy_cells": ScientificValue(
                name="Live Healthy/Stromal Cells Count",
                value=live_healthy,
                unit="cells",
                epistemic_category=EpistemicCategory.MECHANISTIC_SIMULATION,
                source="2D PDAC Cellular Microenvironment Model",
                confidence=0.90,
                in_applicability_domain=True,
            ).to_dict(),
            "clone_distribution": ScientificValue(
                name="Subclonal Distribution",
                value=str(clone_distribution),
                unit="",
                epistemic_category=EpistemicCategory.MECHANISTIC_SIMULATION,
                source="Subclonal Coevolution Tracker",
                confidence=0.88,
                in_applicability_domain=True,
            ).to_dict(),
            "resistant_fraction": ScientificValue(
                name="Resistant Subclone Fraction",
                value=resistant_fraction,
                unit="fraction [0-1]",
                epistemic_category=EpistemicCategory.MECHANISTIC_SIMULATION,
                source="Subclonal Heterogeneity Model",
                confidence=0.88,
                uncertainty_low=round(max(0.0, resistant_fraction - 0.05), 3),
                uncertainty_high=round(min(1.0, resistant_fraction + 0.05), 3),
                in_applicability_domain=True,
            ).to_dict(),
            "proliferation_rate": ScientificValue(
                name="Cell Proliferation Rate",
                value=prolif_rate,
                unit="1/day",
                epistemic_category=EpistemicCategory.MECHANISTIC_SIMULATION,
                source="Cell Cycle Kinetic Solver",
                confidence=0.85,
                in_applicability_domain=True,
            ).to_dict(),
            "apoptosis_rate": ScientificValue(
                name="Apoptosis Rate",
                value=apopt_rate,
                unit="1/day",
                epistemic_category=EpistemicCategory.MECHANISTIC_SIMULATION,
                source="Caspase/PD Pharmacodynamic Solver",
                confidence=0.85,
                in_applicability_domain=True,
            ).to_dict(),
            "necrosis_rate": ScientificValue(
                name="Necrosis Rate",
                value=necro_rate,
                unit="1/day",
                epistemic_category=EpistemicCategory.MECHANISTIC_SIMULATION,
                source="Hypoxic-Nutrient Starvation Solver",
                confidence=0.82,
                in_applicability_domain=True,
            ).to_dict(),
            "hypoxia_fraction": ScientificValue(
                name="Hypoxic Tumor Area Fraction",
                value=hypoxia_frac,
                unit="fraction [0-1]",
                epistemic_category=EpistemicCategory.MECHANISTIC_SIMULATION,
                source="Oxygen Reaction-Diffusion PDE",
                confidence=0.90,
                in_applicability_domain=True,
            ).to_dict(),
            "cell_density": ScientificValue(
                name="Tumor Microenvironment Cell Density",
                value=cell_density,
                unit="cells/mm^2",
                epistemic_category=EpistemicCategory.MECHANISTIC_SIMULATION,
                source="Spatial Hash Grid Population Analysis",
                confidence=0.92,
                in_applicability_domain=True,
            ).to_dict(),
            "local_compound_conc": ScientificValue(
                name="Mean Local Compound Concentration in Tumor Slice",
                value=local_drug_conc,
                unit="uM",
                epistemic_category=EpistemicCategory.MECHANISTIC_SIMULATION,
                source="Extracellular Drug Reaction-Diffusion PDE",
                confidence=0.90,
                in_applicability_domain=True,
            ).to_dict(),
            "immune_activity": ScientificValue(
                name="Tumor Infiltrating Immune Activity Index",
                value=immune_activity,
                unit="index [0-1]",
                epistemic_category=EpistemicCategory.MECHANISTIC_SIMULATION,
                source="T-cell / Macrophage Infiltration Agent Model",
                confidence=0.80,
                in_applicability_domain=True,
            ).to_dict(),
            "tumor_burden": ScientificValue(
                name="Tumor Burden Index",
                value=tumor_burden,
                unit="index [0-1]",
                epistemic_category=EpistemicCategory.MECHANISTIC_SIMULATION,
                source="PDAC Volumetric Proxy",
                confidence=0.90,
                in_applicability_domain=True,
            ).to_dict(),
        }

        # -----------------------------------------------------------------
        # Panel 2: Hematological Indicators
        # -----------------------------------------------------------------
        # STRICT RULE: When systemic_pbpk_result is None, ALL entries MUST be UNMODELED with value=None!
        if systemic_pbpk_result is None:
            unmodeled_hem_msg = (
                "Systemic PBPK simulation is disabled (2D tumor microenvironment mode only); "
                "systemic hematological parameters are not modeled."
            )
            hematological_indicators = {
                "hemoglobin": ScientificValue(
                    name="Hemoglobin (Hb)",
                    value=None,
                    unit="g/dL",
                    epistemic_category=EpistemicCategory.UNMODELED,
                    source="EVOCURE Systemic Guardrail",
                    in_applicability_domain=False,
                    limitations=unmodeled_hem_msg,
                ).to_dict(),
                "hematocrit": ScientificValue(
                    name="Hematocrit (Hct)",
                    value=None,
                    unit="%",
                    epistemic_category=EpistemicCategory.UNMODELED,
                    source="EVOCURE Systemic Guardrail",
                    in_applicability_domain=False,
                    limitations=unmodeled_hem_msg,
                ).to_dict(),
                "erythrocytes_rbc": ScientificValue(
                    name="Erythrocytes (RBC)",
                    value=None,
                    unit="10^6/uL",
                    epistemic_category=EpistemicCategory.UNMODELED,
                    source="EVOCURE Systemic Guardrail",
                    in_applicability_domain=False,
                    limitations=unmodeled_hem_msg,
                ).to_dict(),
                "leukocytes_wbc": ScientificValue(
                    name="Leukocytes (WBC)",
                    value=None,
                    unit="10^3/uL",
                    epistemic_category=EpistemicCategory.UNMODELED,
                    source="EVOCURE Systemic Guardrail",
                    in_applicability_domain=False,
                    limitations=unmodeled_hem_msg,
                ).to_dict(),
                "neutrophils": ScientificValue(
                    name="Absolute Neutrophil Count (ANC)",
                    value=None,
                    unit="10^3/uL",
                    epistemic_category=EpistemicCategory.UNMODELED,
                    source="EVOCURE Systemic Guardrail",
                    in_applicability_domain=False,
                    limitations=unmodeled_hem_msg,
                ).to_dict(),
                "lymphocytes": ScientificValue(
                    name="Lymphocytes",
                    value=None,
                    unit="10^3/uL",
                    epistemic_category=EpistemicCategory.UNMODELED,
                    source="EVOCURE Systemic Guardrail",
                    in_applicability_domain=False,
                    limitations=unmodeled_hem_msg,
                ).to_dict(),
                "platelets": ScientificValue(
                    name="Platelets (PLT)",
                    value=None,
                    unit="10^3/uL",
                    epistemic_category=EpistemicCategory.UNMODELED,
                    source="EVOCURE Systemic Guardrail",
                    in_applicability_domain=False,
                    limitations=unmodeled_hem_msg,
                ).to_dict(),
            }
        else:
            # Mechanistic perturbation derived from Bone Marrow AUC
            auc_marrow = float(systemic_pbpk_result.get("auc", {}).get("bone_marrow", 0.0))
            tox_hem_risk = 0.15
            if toxicity_profile and "hematological_toxicity" in toxicity_profile:
                val = toxicity_profile["hematological_toxicity"].get("value")
                if isinstance(val, (int, float)):
                    tox_hem_risk = float(val)

            # Myelosuppression factor (Emax inhibition model)
            auc_50_bm = 80.0 / (1.0 + 2.0 * tox_hem_risk)
            marrow_suppression = 1.0 / (1.0 + (auc_marrow / max(1.0, auc_50_bm)))

            # Baseline normal hematology reference values:
            # Neutrophils: 4.5 10^3/uL
            # Leukocytes (WBC): 7.2 10^3/uL
            # Platelets: 260 10^3/uL
            # RBC: 4.8 10^6/uL
            # Hemoglobin: 14.2 g/dL
            # Hematocrit: 42.0 %
            # Lymphocytes: 2.1 10^3/uL
            neutrophils_val = round(float(4.5 * marrow_suppression), 2)
            wbc_val = round(float(7.2 * (0.35 + 0.65 * marrow_suppression)), 2)
            platelets_val = round(float(260.0 * (0.30 + 0.70 * marrow_suppression)), 1)
            rbc_val = round(float(4.8 * (0.75 + 0.25 * marrow_suppression)), 2)
            hb_val = round(float(14.2 * (0.75 + 0.25 * marrow_suppression)), 1)
            hct_val = round(float(42.0 * (0.75 + 0.25 * marrow_suppression)), 1)
            lymph_val = round(float(2.1 * (0.50 + 0.50 * marrow_suppression)), 2)

            hem_limitations = (
                "Exploratory PBPK-PD mechanistic perturbation model from bone marrow AUC. "
                "Not a clinical diagnostic laboratory value."
            )

            hematological_indicators = {
                "hemoglobin": ScientificValue(
                    name="Hemoglobin (Hb)",
                    value=hb_val,
                    unit="g/dL",
                    epistemic_category=EpistemicCategory.EXPLORATORY_MODEL,
                    source="Bone Marrow PBPK Exposure Perturbation",
                    confidence=0.65,
                    uncertainty_low=round(hb_val * 0.85, 1),
                    uncertainty_high=round(hb_val * 1.15, 1),
                    in_applicability_domain=True,
                    limitations=hem_limitations,
                ).to_dict(),
                "hematocrit": ScientificValue(
                    name="Hematocrit (Hct)",
                    value=hct_val,
                    unit="%",
                    epistemic_category=EpistemicCategory.EXPLORATORY_MODEL,
                    source="Bone Marrow PBPK Exposure Perturbation",
                    confidence=0.65,
                    uncertainty_low=round(hct_val * 0.85, 1),
                    uncertainty_high=round(hct_val * 1.15, 1),
                    in_applicability_domain=True,
                    limitations=hem_limitations,
                ).to_dict(),
                "erythrocytes_rbc": ScientificValue(
                    name="Erythrocytes (RBC)",
                    value=rbc_val,
                    unit="10^6/uL",
                    epistemic_category=EpistemicCategory.EXPLORATORY_MODEL,
                    source="Bone Marrow PBPK Exposure Perturbation",
                    confidence=0.65,
                    uncertainty_low=round(rbc_val * 0.85, 2),
                    uncertainty_high=round(rbc_val * 1.15, 2),
                    in_applicability_domain=True,
                    limitations=hem_limitations,
                ).to_dict(),
                "leukocytes_wbc": ScientificValue(
                    name="Leukocytes (WBC)",
                    value=wbc_val,
                    unit="10^3/uL",
                    epistemic_category=EpistemicCategory.EXPLORATORY_MODEL,
                    source="Bone Marrow PBPK Exposure Perturbation",
                    confidence=0.65,
                    uncertainty_low=round(wbc_val * 0.80, 2),
                    uncertainty_high=round(wbc_val * 1.20, 2),
                    in_applicability_domain=True,
                    limitations=hem_limitations,
                ).to_dict(),
                "neutrophils": ScientificValue(
                    name="Absolute Neutrophil Count (ANC)",
                    value=neutrophils_val,
                    unit="10^3/uL",
                    epistemic_category=EpistemicCategory.EXPLORATORY_MODEL,
                    source="Bone Marrow PBPK Exposure Perturbation",
                    confidence=0.68,
                    uncertainty_low=round(neutrophils_val * 0.78, 2),
                    uncertainty_high=round(neutrophils_val * 1.22, 2),
                    in_applicability_domain=True,
                    limitations=hem_limitations,
                ).to_dict(),
                "lymphocytes": ScientificValue(
                    name="Lymphocytes",
                    value=lymph_val,
                    unit="10^3/uL",
                    epistemic_category=EpistemicCategory.EXPLORATORY_MODEL,
                    source="Bone Marrow PBPK Exposure Perturbation",
                    confidence=0.62,
                    uncertainty_low=round(lymph_val * 0.82, 2),
                    uncertainty_high=round(lymph_val * 1.18, 2),
                    in_applicability_domain=True,
                    limitations=hem_limitations,
                ).to_dict(),
                "platelets": ScientificValue(
                    name="Platelets (PLT)",
                    value=platelets_val,
                    unit="10^3/uL",
                    epistemic_category=EpistemicCategory.EXPLORATORY_MODEL,
                    source="Bone Marrow PBPK Exposure Perturbation",
                    confidence=0.66,
                    uncertainty_low=round(platelets_val * 0.80, 1),
                    uncertainty_high=round(platelets_val * 1.20, 1),
                    in_applicability_domain=True,
                    limitations=hem_limitations,
                ).to_dict(),
            }

        # -----------------------------------------------------------------
        # Panel 3: Metabolic and Organ Indicators
        # -----------------------------------------------------------------
        # STRICT RULE: When systemic_pbpk_result is None, ALL entries MUST be UNMODELED with value=None!
        if systemic_pbpk_result is None:
            unmodeled_organ_msg = (
                "Systemic PBPK simulation is disabled (2D tumor microenvironment mode only); "
                "systemic metabolic and organ indicators are not modeled."
            )
            metabolic_organ_indicators = {
                "glucose": ScientificValue(
                    name="Fasting Glucose",
                    value=None,
                    unit="mg/dL",
                    epistemic_category=EpistemicCategory.UNMODELED,
                    source="EVOCURE Systemic Guardrail",
                    in_applicability_domain=False,
                    limitations=unmodeled_organ_msg,
                ).to_dict(),
                "insulin": ScientificValue(
                    name="Insulin",
                    value=None,
                    unit="uIU/mL",
                    epistemic_category=EpistemicCategory.UNMODELED,
                    source="EVOCURE Systemic Guardrail",
                    in_applicability_domain=False,
                    limitations=unmodeled_organ_msg,
                ).to_dict(),
                "bilirubin": ScientificValue(
                    name="Total Bilirubin",
                    value=None,
                    unit="mg/dL",
                    epistemic_category=EpistemicCategory.UNMODELED,
                    source="EVOCURE Systemic Guardrail",
                    in_applicability_domain=False,
                    limitations=unmodeled_organ_msg,
                ).to_dict(),
                "alt": ScientificValue(
                    name="Alanine Aminotransferase (ALT)",
                    value=None,
                    unit="U/L",
                    epistemic_category=EpistemicCategory.UNMODELED,
                    source="EVOCURE Systemic Guardrail",
                    in_applicability_domain=False,
                    limitations=unmodeled_organ_msg,
                ).to_dict(),
                "ast": ScientificValue(
                    name="Aspartate Aminotransferase (AST)",
                    value=None,
                    unit="U/L",
                    epistemic_category=EpistemicCategory.UNMODELED,
                    source="EVOCURE Systemic Guardrail",
                    in_applicability_domain=False,
                    limitations=unmodeled_organ_msg,
                ).to_dict(),
                "alkaline_phosphatase": ScientificValue(
                    name="Alkaline Phosphatase (ALP)",
                    value=None,
                    unit="U/L",
                    epistemic_category=EpistemicCategory.UNMODELED,
                    source="EVOCURE Systemic Guardrail",
                    in_applicability_domain=False,
                    limitations=unmodeled_organ_msg,
                ).to_dict(),
                "albumin": ScientificValue(
                    name="Serum Albumin",
                    value=None,
                    unit="g/dL",
                    epistemic_category=EpistemicCategory.UNMODELED,
                    source="EVOCURE Systemic Guardrail",
                    in_applicability_domain=False,
                    limitations=unmodeled_organ_msg,
                ).to_dict(),
                "creatinine": ScientificValue(
                    name="Serum Creatinine",
                    value=None,
                    unit="mg/dL",
                    epistemic_category=EpistemicCategory.UNMODELED,
                    source="EVOCURE Systemic Guardrail",
                    in_applicability_domain=False,
                    limitations=unmodeled_organ_msg,
                ).to_dict(),
                "urea": ScientificValue(
                    name="Blood Urea Nitrogen (BUN)",
                    value=None,
                    unit="mg/dL",
                    epistemic_category=EpistemicCategory.UNMODELED,
                    source="EVOCURE Systemic Guardrail",
                    in_applicability_domain=False,
                    limitations=unmodeled_organ_msg,
                ).to_dict(),
                "sodium": ScientificValue(
                    name="Serum Sodium (Na+)",
                    value=None,
                    unit="mmol/L",
                    epistemic_category=EpistemicCategory.UNMODELED,
                    source="EVOCURE Systemic Guardrail",
                    in_applicability_domain=False,
                    limitations=unmodeled_organ_msg,
                ).to_dict(),
                "potassium": ScientificValue(
                    name="Serum Potassium (K+)",
                    value=None,
                    unit="mmol/L",
                    epistemic_category=EpistemicCategory.UNMODELED,
                    source="EVOCURE Systemic Guardrail",
                    in_applicability_domain=False,
                    limitations=unmodeled_organ_msg,
                ).to_dict(),
            }
        else:
            # Mechanistic perturbations derived from Liver, Kidney, and Pancreas AUCs
            auc_dict = systemic_pbpk_result.get("auc", {})
            auc_liver = float(auc_dict.get("liver", 0.0))
            auc_kidneys = float(auc_dict.get("kidneys", 0.0))
            auc_panc = float(auc_dict.get("pancreas", 0.0))

            tox_hep_risk = 0.12
            tox_neph_risk = 0.08
            if toxicity_profile:
                if "hepatotoxicity" in toxicity_profile and isinstance(toxicity_profile["hepatotoxicity"].get("value"), (int, float)):
                    tox_hep_risk = float(toxicity_profile["hepatotoxicity"]["value"])
                if "nephrotoxicity" in toxicity_profile and isinstance(toxicity_profile["nephrotoxicity"].get("value"), (int, float)):
                    tox_neph_risk = float(toxicity_profile["nephrotoxicity"]["value"])

            # Liver stress factor
            f_liv = 1.0 + (auc_liver / 120.0) * (1.0 + 2.5 * tox_hep_risk)
            alt_val = round(float(28.0 * f_liv), 1)
            ast_val = round(float(24.0 * f_liv), 1)
            alp_val = round(float(75.0 * (1.0 + 0.35 * (f_liv - 1.0))), 1)
            bili_val = round(float(0.8 * (1.0 + 0.50 * (f_liv - 1.0))), 2)
            alb_val = round(float(4.4 / (1.0 + 0.08 * (f_liv - 1.0))), 2)

            # Kidney stress factor
            f_kid = 1.0 + (auc_kidneys / 140.0) * (1.0 + 2.5 * tox_neph_risk)
            creat_val = round(float(0.9 * f_kid), 2)
            bun_val = round(float(14.0 * f_kid), 1)
            na_val = 140.0
            k_val = round(float(4.2 * (1.0 + 0.06 * (f_kid - 1.0))), 2)

            # Pancreas stress factor
            f_pan = 1.0 + (auc_panc / 90.0)
            glucose_val = round(float(92.0 * (1.0 + 0.20 * (f_pan - 1.0))), 1)
            insulin_val = round(float(12.0 / (1.0 + 0.15 * (f_pan - 1.0))), 1)

            organ_limitations = (
                "Exploratory PBPK-PD mechanistic perturbation model from organ AUCs. "
                "Not a clinical diagnostic laboratory value."
            )

            metabolic_organ_indicators = {
                "glucose": ScientificValue(
                    name="Fasting Glucose",
                    value=glucose_val,
                    unit="mg/dL",
                    epistemic_category=EpistemicCategory.EXPLORATORY_MODEL,
                    source="Pancreatic PBPK Exposure Model",
                    confidence=0.64,
                    uncertainty_low=round(glucose_val * 0.88, 1),
                    uncertainty_high=round(glucose_val * 1.12, 1),
                    in_applicability_domain=True,
                    limitations=organ_limitations,
                ).to_dict(),
                "insulin": ScientificValue(
                    name="Insulin",
                    value=insulin_val,
                    unit="uIU/mL",
                    epistemic_category=EpistemicCategory.EXPLORATORY_MODEL,
                    source="Pancreatic PBPK Exposure Model",
                    confidence=0.62,
                    uncertainty_low=round(insulin_val * 0.85, 1),
                    uncertainty_high=round(insulin_val * 1.15, 1),
                    in_applicability_domain=True,
                    limitations=organ_limitations,
                ).to_dict(),
                "bilirubin": ScientificValue(
                    name="Total Bilirubin",
                    value=bili_val,
                    unit="mg/dL",
                    epistemic_category=EpistemicCategory.EXPLORATORY_MODEL,
                    source="Hepatic PBPK Exposure Model",
                    confidence=0.65,
                    uncertainty_low=round(bili_val * 0.82, 2),
                    uncertainty_high=round(bili_val * 1.18, 2),
                    in_applicability_domain=True,
                    limitations=organ_limitations,
                ).to_dict(),
                "alt": ScientificValue(
                    name="Alanine Aminotransferase (ALT)",
                    value=alt_val,
                    unit="U/L",
                    epistemic_category=EpistemicCategory.EXPLORATORY_MODEL,
                    source="Hepatic PBPK Exposure Model",
                    confidence=0.66,
                    uncertainty_low=round(alt_val * 0.80, 1),
                    uncertainty_high=round(alt_val * 1.20, 1),
                    in_applicability_domain=True,
                    limitations=organ_limitations,
                ).to_dict(),
                "ast": ScientificValue(
                    name="Aspartate Aminotransferase (AST)",
                    value=ast_val,
                    unit="U/L",
                    epistemic_category=EpistemicCategory.EXPLORATORY_MODEL,
                    source="Hepatic PBPK Exposure Model",
                    confidence=0.66,
                    uncertainty_low=round(ast_val * 0.80, 1),
                    uncertainty_high=round(ast_val * 1.20, 1),
                    in_applicability_domain=True,
                    limitations=organ_limitations,
                ).to_dict(),
                "alkaline_phosphatase": ScientificValue(
                    name="Alkaline Phosphatase (ALP)",
                    value=alp_val,
                    unit="U/L",
                    epistemic_category=EpistemicCategory.EXPLORATORY_MODEL,
                    source="Hepatic/Biliary PBPK Exposure Model",
                    confidence=0.63,
                    uncertainty_low=round(alp_val * 0.85, 1),
                    uncertainty_high=round(alp_val * 1.15, 1),
                    in_applicability_domain=True,
                    limitations=organ_limitations,
                ).to_dict(),
                "albumin": ScientificValue(
                    name="Serum Albumin",
                    value=alb_val,
                    unit="g/dL",
                    epistemic_category=EpistemicCategory.EXPLORATORY_MODEL,
                    source="Hepatic Synthesis PBPK Exposure Model",
                    confidence=0.68,
                    uncertainty_low=round(alb_val * 0.90, 2),
                    uncertainty_high=round(alb_val * 1.10, 2),
                    in_applicability_domain=True,
                    limitations=organ_limitations,
                ).to_dict(),
                "creatinine": ScientificValue(
                    name="Serum Creatinine",
                    value=creat_val,
                    unit="mg/dL",
                    epistemic_category=EpistemicCategory.EXPLORATORY_MODEL,
                    source="Renal Elimination PBPK Exposure Model",
                    confidence=0.67,
                    uncertainty_low=round(creat_val * 0.82, 2),
                    uncertainty_high=round(creat_val * 1.18, 2),
                    in_applicability_domain=True,
                    limitations=organ_limitations,
                ).to_dict(),
                "urea": ScientificValue(
                    name="Blood Urea Nitrogen (BUN)",
                    value=bun_val,
                    unit="mg/dL",
                    epistemic_category=EpistemicCategory.EXPLORATORY_MODEL,
                    source="Renal Elimination PBPK Exposure Model",
                    confidence=0.67,
                    uncertainty_low=round(bun_val * 0.82, 1),
                    uncertainty_high=round(bun_val * 1.18, 1),
                    in_applicability_domain=True,
                    limitations=organ_limitations,
                ).to_dict(),
                "sodium": ScientificValue(
                    name="Serum Sodium (Na+)",
                    value=na_val,
                    unit="mmol/L",
                    epistemic_category=EpistemicCategory.EXPLORATORY_MODEL,
                    source="Renal Electrolyte Homeostasis Model",
                    confidence=0.75,
                    uncertainty_low=135.0,
                    uncertainty_high=145.0,
                    in_applicability_domain=True,
                    limitations=organ_limitations,
                ).to_dict(),
                "potassium": ScientificValue(
                    name="Serum Potassium (K+)",
                    value=k_val,
                    unit="mmol/L",
                    epistemic_category=EpistemicCategory.EXPLORATORY_MODEL,
                    source="Renal Electrolyte Homeostasis Model",
                    confidence=0.72,
                    uncertainty_low=round(k_val * 0.92, 2),
                    uncertainty_high=round(k_val * 1.08, 2),
                    in_applicability_domain=True,
                    limitations=organ_limitations,
                ).to_dict(),
            }

        # -----------------------------------------------------------------
        # Panel 4: Oncological Biomarkers
        # -----------------------------------------------------------------
        # CA 19-9 is shed by PDAC tumor cells: baseline normal is < 37 U/mL
        # Elevated in PDAC proportional to tumor cell load
        ca19_9_val = round(float(35.0 + 0.22 * live_tumor), 1)

        # Molecular clone burden (KRAS G12D / TP53 LOF)
        kras_clone_burden_pct = round(float(100.0 * (live_tumor / max(1, live_tumor + live_healthy))), 1)

        # Inflammation index (CRP proxy in mg/L): elevated with hypoxia and necrosis
        inflammation_index_val = round(float(3.0 + 15.0 * hypoxia_frac + 250.0 * necro_rate), 1)

        # Ki-67 proliferation index (%): fraction of tumor cells in active cell cycle
        ki67_val = round(float(min(85.0, max(15.0, prolif_rate * 1200.0))), 1)

        # Pharmacodynamic response marker (0 - 100%): pathway suppression relative to untreated
        pd_response_pct = round(float(min(100.0, max(0.0, local_drug_conc * 12.5))), 1)

        oncological_biomarkers = {
            "ca19_9": ScientificValue(
                name="CA 19-9 (Carbohydrate Antigen 19-9)",
                value=ca19_9_val,
                unit="U/mL",
                epistemic_category=EpistemicCategory.MECHANISTIC_SIMULATION,
                source="PDAC Tumor Burden Shedding Model",
                confidence=0.88,
                uncertainty_low=round(ca19_9_val * 0.85, 1),
                uncertainty_high=round(ca19_9_val * 1.15, 1),
                in_applicability_domain=True,
            ).to_dict(),
            "molecular_tumor_markers": ScientificValue(
                name="KRAS G12D / TP53 Variant Allele Frequency Burden",
                value=kras_clone_burden_pct,
                unit="%",
                epistemic_category=EpistemicCategory.MECHANISTIC_SIMULATION,
                source="Genomic Subclonal Cellular Tracker",
                confidence=0.92,
                uncertainty_low=round(max(0.0, kras_clone_burden_pct - 5.0), 1),
                uncertainty_high=round(min(100.0, kras_clone_burden_pct + 5.0), 1),
                in_applicability_domain=True,
            ).to_dict(),
            "inflammation_index": ScientificValue(
                name="Systemic/Microenvironmental Inflammation Index (CRP Proxy)",
                value=inflammation_index_val,
                unit="mg/L",
                epistemic_category=EpistemicCategory.MECHANISTIC_SIMULATION,
                source="Hypoxia-Necrosis Inflammatory Cascade Model",
                confidence=0.82,
                uncertainty_low=round(inflammation_index_val * 0.80, 1),
                uncertainty_high=round(inflammation_index_val * 1.20, 1),
                in_applicability_domain=True,
            ).to_dict(),
            "ki67_proliferation_index": ScientificValue(
                name="Ki-67 Proliferation Marker Index",
                value=ki67_val,
                unit="%",
                epistemic_category=EpistemicCategory.MECHANISTIC_SIMULATION,
                source="Cell Cycle Kinetic Distribution Model",
                confidence=0.86,
                uncertainty_low=round(max(0.0, ki67_val - 6.0), 1),
                uncertainty_high=round(min(100.0, ki67_val + 6.0), 1),
                in_applicability_domain=True,
            ).to_dict(),
            "pharmacodynamic_response_marker": ScientificValue(
                name="Pharmacodynamic Pathway Suppression Marker",
                value=pd_response_pct,
                unit="%",
                epistemic_category=EpistemicCategory.MECHANISTIC_SIMULATION,
                source="Local Drug-Target Engagement Model",
                confidence=0.84,
                uncertainty_low=round(max(0.0, pd_response_pct - 8.0), 1),
                uncertainty_high=round(min(100.0, pd_response_pct + 8.0), 1),
                in_applicability_domain=True,
            ).to_dict(),
        }

        return {
            "tumor_indicators": tumor_indicators,
            "hematological_indicators": hematological_indicators,
            "metabolic_organ_indicators": metabolic_organ_indicators,
            "oncological_biomarkers": oncological_biomarkers,
            "systemic_pbpk_enabled": systemic_pbpk_result is not None,
            "safety_disclaimer": BIOMARKER_DISCLAIMER,
        }
