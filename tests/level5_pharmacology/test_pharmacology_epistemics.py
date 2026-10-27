"""
Level-5 Pharmacology, Toxicology, Systemic PBPK, and Epistemic Guardrails Test Suite.

Validates:
1. Tripartite Exposure Partitioning:
   applied_conc_um -> free_extracellular_um -> estimated_intracellular_um.
2. validate_for_biomedical_mode:
   Quarantining Mode A artificial life traits and recomputing properties strictly from chemical graph.
3. ToxicologyEngine 10 Categories & Boundaries:
   All 10 required categories, ScientificValue metadata, internal 0-100 risk label,
   and "Insufficient data" (INSUFFICIENT_DATA) on out-of-domain molecules (MW > 1200 or heavy_atoms < 3).
4. SystemicPBPKEngine 7 Compartments:
   blood, liver, kidneys, bone_marrow, pancreas, tumor, peripheral.
5. BiologicalIndicatorsEngine Mandatory Section 12 Guardrail:
   When systemic_pbpk_result=None, every hematological & metabolic indicator is value is None
   and epistemic_code == 'UNMODELED'; when PBPK is provided, they are 'EXPLORATORY_MODEL'.
6. CalibratedQSARPredictor Applicability Domain Assessment:
   In-domain vs out-of-domain detection and uncertainty warnings.
7. LocalGemmaAssistant & LayaDecisionEngine:
   PLAY vs LAB tone adaptation, grounded numbers, cancer cure-claim interception,
   parameter change approval guardrails (proposed_action), and triage fallback resilience.
8. Clinical Safety Language Audit:
   Zero forbidden clinical claims ('proven cure', 'safe for humans', 'guaranteed clinical efficacy').
"""

from __future__ import annotations

import math
from typing import Any
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import Descriptors

from ai.laya.decision_engine import LayaDecisionEngine
from ai.local_llm.gemma_assistant import LocalGemmaAssistant
from ai.predictors.data_connectors import ScientificDataManager
from ai.predictors.qsar import CalibratedQSARPredictor
from simulation.chemistry.engine import ChemistryEngine
from simulation.core.contracts import (
    CompoundRecord,
    EpistemicCategory,
    InterfaceMode,
    ScientificValue,
    UniverseMode,
)
from simulation.pharmacology.admet import (
    PharmacologyEngine,
    validate_for_biomedical_mode,
)
from simulation.systemic.biomarkers import (
    BIOMARKER_DISCLAIMER,
    BiologicalIndicatorsEngine,
)
from simulation.systemic.pbpk import (
    PHYSIOLOGICAL_VOLUMES_L,
    SystemicPBPKEngine,
)
from simulation.toxicity.models import (
    RISK_INDEX_LABEL,
    ToxicologyEngine,
)


# =============================================================================
# 1. Tripartite Exposure Partitioning
# =============================================================================

def test_tripartite_exposure_partitioning() -> None:
    """
    Validate Section 11.2 Tripartite Exposure Partitioning:
    - Applied total concentration (applied_conc_um)
    - Free extracellular concentration (free_extracellular_um = fu * applied_conc_um)
    - Intracellular concentration (estimated_intracellular_um modulated by influx/efflux)
    """
    ce = ChemistryEngine()
    gem = ce.resolve_compound_input("Gemcitabine")
    pe = PharmacologyEngine()

    pd_profile = pe.predict_antitumor_activity(gem)
    assert "compute_exposure_partition" in pd_profile
    assert "sample_exposure_partition" in pd_profile

    # Test across multiple applied concentrations: 0.5, 2.0, 5.0, 10.0 uM
    for applied in [0.5, 2.0, 5.0, 10.0]:
        part = pd_profile["compute_exposure_partition"](applied)

        assert "applied_conc_um" in part
        assert "free_extracellular_um" in part
        assert "estimated_intracellular_um" in part
        assert "unbound_fraction_fu" in part
        assert "partition_ratio_intra_to_free" in part

        assert part["applied_conc_um"] == pytest.approx(applied, rel=1e-4)

        # Free fraction relationship: C_free = fu * C_applied
        fu = part["unbound_fraction_fu"]
        assert 0.0 < fu <= 1.0
        expected_free = round(fu * applied, 4)
        assert part["free_extracellular_um"] == pytest.approx(expected_free, abs=1e-3)

        # Intracellular accumulation
        assert part["estimated_intracellular_um"] >= 0.0
        assert part["partition_ratio_intra_to_free"] > 0.0


# =============================================================================
# 2. validate_for_biomedical_mode Quarantining Mode A Traits
# =============================================================================

def test_biomedical_mode_quarantining_artificial_traits() -> None:
    """
    Validate Section 5.1:
    Mode A artificial traits (fitness, replication_rate, predator_efficiency)
    must NEVER transfer into pharmacological activity. Properties must be recomputed
    strictly from RDKit chemical graph descriptors.
    """
    ce = ChemistryEngine()
    compound = ce.resolve_compound_input("Gemcitabine")

    # Inject artificial life traits
    compound.artificial_traits = {
        "replication_rate": 0.85,
        "fitness": 1.92,
        "predator_efficiency": 0.45,
    }

    result = validate_for_biomedical_mode(compound, source_mode=UniverseMode.ARTIFICIAL_LIFE)

    assert result["is_valid"] is True
    assert result["chemical_validity"] is True
    assert "discarded_traits" in result
    for trait in ["replication_rate", "fitness", "predator_efficiency"]:
        assert trait in result["discarded_traits"]

    # Verify recomputed properties exist and derive purely from molecular structure
    props = result["recomputed_properties"]
    assert props["mol_weight"] == pytest.approx(263.2, abs=0.5)
    assert props["heavy_atoms"] == 18
    assert props["hbd"] == 3
    assert props["hba"] == 6

    # Test invalid SMILES string rejection
    invalid_comp = CompoundRecord(
        compound_id="INVALID-001",
        name="Invalid Molecule",
        common_name="Invalid Molecule",
        canonical_smiles="CC(C)(C)(C)(C)C_INVALID_VALENCE",
        formula="C7H16",
        mol_weight=100.0,
    )
    res_invalid = validate_for_biomedical_mode(invalid_comp, source_mode=UniverseMode.GROUNDED_CHEMISTRY)
    assert res_invalid["is_valid"] is False
    assert res_invalid["chemical_validity"] is False
    assert "sanitization" in res_invalid["reason"].lower() or "valence" in res_invalid["reason"].lower()


# =============================================================================
# 3. ToxicologyEngine 10 Categories, Metadata & Applicability Domain Boundaries
# =============================================================================

def test_toxicology_engine_categories_metadata_and_boundaries() -> None:
    """
    Validate Section 11.3 computational toxicology:
    1. Evaluates all 10 required organ and systemic toxicity categories.
    2. Packages each evaluation in ScientificValue with required provenance metadata.
    3. Enforces internal computational risk score (0-100) disclaimer.
    4. Out-of-domain molecules (MW > 1200 or heavy_atoms < 3) return Insufficient Data (INSUFFICIENT_DATA).
    """
    tox_engine = ToxicologyEngine()
    ce = ChemistryEngine()
    gem = ce.resolve_compound_input("Gemcitabine")

    # In-domain evaluation
    res_in = tox_engine.evaluate_toxicology(gem, in_domain=True)

    required_categories = [
        "non_tumor_cytotoxicity",
        "hepatotoxicity",
        "nephrotoxicity",
        "cardiotoxicity",
        "neurotoxicity",
        "genotoxicity",
        "hematological_toxicity",
        "gastrointestinal_toxicity",
        "interaction_risk",
        "global_uncertainty",
    ]

    for cat in required_categories:
        assert cat in res_in, f"Missing required toxicology category: {cat}"
        item = res_in[cat]
        assert "name" in item
        assert "value" in item
        assert "unit" in item
        assert "source" in item
        assert "model_version" in item
        assert "confidence" in item
        assert "validation_status" in item
        assert "limitations" in item
        assert "epistemic_category" in item
        assert "epistemic_code" in item

    # Verify risk index disclaimer
    assert res_in["risk_index_label"] == RISK_INDEX_LABEL
    assert "NOT a clinical human safety measurement" in res_in["risk_index_label"]

    # -------------------------------------------------------------------------
    # Out-of-domain Boundary Test: MW > 1200
    # -------------------------------------------------------------------------
    big_smiles = "C" * 100  # Linear alkane C100H202, MW > 1400
    mol_big = Chem.MolFromSmiles(big_big_smi := ("CC" + "C" * 98))
    mw_big = float(Descriptors.MolWt(mol_big))
    assert mw_big > 1200.0

    comp_big = CompoundRecord(
        compound_id="EXOTIC-HEAVY",
        name="Exotic Heavy Polymer",
        common_name="Exotic Heavy Polymer",
        canonical_smiles=big_big_smi,
        formula="C100H202",
        mol_weight=mw_big,
        heavy_atoms=100,
    )
    res_big = tox_engine.evaluate_toxicology(comp_big, in_domain=True)
    assert res_big["applicability_domain_in"] is False
    assert res_big["internal_computational_risk_index_0_100"] is None

    for cat in required_categories[:9]:
        val = res_big[cat]
        assert val["value"] == "Insufficient data"
        assert val["epistemic_code"] == "INSUFFICIENT_DATA"
        assert val["in_applicability_domain"] is False

    # -------------------------------------------------------------------------
    # Out-of-domain Boundary Test: heavy_atoms < 3
    # -------------------------------------------------------------------------
    comp_small = CompoundRecord(
        compound_id="TINY-MOL",
        name="Ethane",
        common_name="Ethane",
        canonical_smiles="CC",
        formula="C2H6",
        mol_weight=30.07,
        heavy_atoms=2,
    )
    res_small = tox_engine.evaluate_toxicology(comp_small, in_domain=True)
    assert res_small["applicability_domain_in"] is False
    for cat in required_categories[:9]:
        assert res_small[cat]["value"] == "Insufficient data"
        assert res_small[cat]["epistemic_code"] == "INSUFFICIENT_DATA"


# =============================================================================
# 4. SystemicPBPKEngine 7 Anatomical Compartments
# =============================================================================

def test_systemic_pbpk_7_compartments() -> None:
    """
    Validate Section 13 7-Compartment Systemic PBPK Kinetics:
    - Compartments: blood, liver, kidneys, bone_marrow, pancreas, tumor, peripheral.
    - Coupled mass-balance ODE integration via solve_ivp.
    - Returns time-series concentrations and compartment AUCs.
    - Strict non-negativity across all compartments.
    """
    pbpk = SystemicPBPKEngine()
    ce = ChemistryEngine()
    gem = ce.resolve_compound_input("Gemcitabine")

    expected_compartments = {"blood", "liver", "kidneys", "bone_marrow", "pancreas", "tumor", "peripheral"}

    # Verify physiological volumes and flows
    assert set(pbpk.volumes.keys()) == expected_compartments
    assert set(pbpk.flows.keys()) == (expected_compartments - {"blood"})

    # Run 24-hour IV infusion simulation
    sim_res = pbpk.simulate(
        compound=gem,
        dose_mg=100.0,
        administration_route="iv_infusion",
        infusion_duration_hours=1.0,
        duration_hours=24.0,
        num_time_points=120,
    )

    assert sim_res["administration_route"] == "iv_infusion"
    assert len(sim_res["time_points_hours"]) == 120

    # Verify all 7 compartments are present in time-series concentrations and AUC
    concs = sim_res["concentrations"]
    aucs = sim_res["auc"]
    assert set(concs.keys()) == expected_compartments
    assert set(aucs.keys()) == expected_compartments

    for comp_name in expected_compartments:
        profile = np.array(concs[comp_name])
        assert len(profile) == 120
        assert np.all(profile >= 0.0), f"Negative concentration detected in {comp_name}"
        assert not np.any(np.isnan(profile)), f"NaN detected in {comp_name}"
        assert aucs[comp_name] > 0.0, f"AUC must be positive for {comp_name}"

    # Verify expected PK behavior: Blood Cmax occurs near end of 1.0h infusion
    t_arr = np.array(sim_res["time_points_hours"])
    c_blood = np.array(concs["blood"])
    idx_max = int(np.argmax(c_blood))
    t_cmax = t_arr[idx_max]
    assert 0.8 <= t_cmax <= 1.5, f"Expected Cmax near infusion end (1.0h), got {t_cmax:.2f}h"


# =============================================================================
# 5. BiologicalIndicatorsEngine Mandatory Section 12 Guardrail
# =============================================================================

def test_biological_indicators_section12_guardrail() -> None:
    """
    MANDATORY SECTION 12 EPISTEMIC GUARDRAIL:
    1. When systemic PBPK is disabled (systemic_pbpk_result is None):
       EVERY hematological and systemic metabolic/organ indicator MUST have:
       - value is None
       - epistemic_code == 'UNMODELED'
       (Never emit fabricated laboratory values!)
    2. When systemic PBPK is provided:
       Hematological and metabolic indicators have numeric values and are strictly
       tagged EpistemicCategory.EXPLORATORY_MODEL.
    """
    bie = BiologicalIndicatorsEngine()
    ce = ChemistryEngine()
    gem = ce.resolve_compound_input("Gemcitabine")

    # -------------------------------------------------------------------------
    # Branch 1: Systemic PBPK Disabled (systemic_pbpk_result is None)
    # -------------------------------------------------------------------------
    res_unmodeled = bie.compute_biological_indicators(
        tumor_microenvironment_state={"live_tumor_cells": 1200, "hypoxia_fraction": 0.25},
        systemic_pbpk_result=None,
        compound=gem,
    )

    hem = res_unmodeled["hematological_indicators"]
    met = res_unmodeled["metabolic_organ_indicators"]
    tumor = res_unmodeled["tumor_indicators"]

    # Tumor indicators remain modeled from 2D cellular slice
    assert tumor["live_tumor_cells"]["value"] == 1200
    assert tumor["live_tumor_cells"]["epistemic_code"] == "MECHANISTIC_SIMULATION"

    # All hematological indicators MUST be None and UNMODELED
    hem_keys = ["hemoglobin", "hematocrit", "erythrocytes_rbc", "leukocytes_wbc", "neutrophils", "lymphocytes", "platelets"]
    for k in hem_keys:
        assert k in hem, f"Missing hematological indicator: {k}"
        assert hem[k]["value"] is None, f"Hematological {k} must have value is None when PBPK is disabled"
        assert hem[k]["epistemic_code"] == "UNMODELED", f"Hematological {k} must be UNMODELED"
        assert hem[k]["in_applicability_domain"] is False

    # All metabolic/organ indicators MUST be None and UNMODELED
    met_keys = ["glucose", "insulin", "bilirubin", "alt", "ast", "alkaline_phosphatase", "albumin", "creatinine", "urea", "sodium", "potassium"]
    for k in met_keys:
        assert k in met, f"Missing metabolic indicator: {k}"
        assert met[k]["value"] is None, f"Metabolic {k} must have value is None when PBPK is disabled"
        assert met[k]["epistemic_code"] == "UNMODELED", f"Metabolic {k} must be UNMODELED"
        assert met[k]["in_applicability_domain"] is False

    # -------------------------------------------------------------------------
    # Branch 2: Systemic PBPK Enabled (systemic_pbpk_result provided)
    # -------------------------------------------------------------------------
    pbpk = SystemicPBPKEngine()
    pbpk_res = pbpk.simulate(gem, dose_mg=100.0)
    res_modeled = bie.compute_biological_indicators(
        tumor_microenvironment_state={"live_tumor_cells": 1200},
        systemic_pbpk_result=pbpk_res,
        compound=gem,
    )

    hem_mod = res_modeled["hematological_indicators"]
    met_mod = res_modeled["metabolic_organ_indicators"]

    for k in hem_keys:
        assert hem_mod[k]["value"] is not None
        assert isinstance(hem_mod[k]["value"], (int, float))
        assert hem_mod[k]["epistemic_code"] == "EXPLORATORY_MODEL"
        assert hem_mod[k]["uncertainty_low"] is not None
        assert hem_mod[k]["uncertainty_high"] is not None

    for k in met_keys:
        assert met_mod[k]["value"] is not None
        assert isinstance(met_mod[k]["value"], (int, float))
        assert met_mod[k]["epistemic_code"] == "EXPLORATORY_MODEL"


# =============================================================================
# 6. CalibratedQSARPredictor Applicability Domain Assessment
# =============================================================================

def test_qsar_applicability_domain_assessment() -> None:
    """
    Validate Tanimoto Applicability Domain assessment:
    - Query against reference training set (threshold s_max >= 0.25).
    - In-domain compounds yield in_domain=True with empty warning.
    - Out-of-domain compounds yield in_domain=False with an explicit warning.
    """
    dm = ScientificDataManager()
    dataset = dm.load_reference_dataset()
    ref_smiles = [c.canonical_smiles for c in dataset]

    predictor = CalibratedQSARPredictor(applicability_threshold=0.25)
    predictor.train_smiles = ref_smiles

    # Known reference drug in dataset: Gemcitabine
    gem_smi = next(c.canonical_smiles for c in dataset if c.name == "Gemcitabine")
    ad_in = predictor.assess_applicability_domain(gem_smi)
    assert ad_in["in_domain"] is True
    assert ad_in["max_similarity"] == pytest.approx(1.0, abs=1e-3)
    assert ad_in["warning"] == ""

    # Exotic structure with no analog in PDAC kinase/antimetabolite reference: e.g. long polyethylene glycol
    peg_smi = "CCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCO"
    ad_out = predictor.assess_applicability_domain(peg_smi)
    assert ad_out["in_domain"] is False
    assert ad_out["max_similarity"] < 0.25
    assert "Out-of-domain alert" in ad_out["warning"]
    assert "elevated uncertainty" in ad_out["warning"]


# =============================================================================
# 7. LocalGemmaAssistant & LayaDecisionEngine Resilience
# =============================================================================

def test_local_gemma_assistant_and_laya_decision_engine() -> None:
    """
    Validate Local Gemma Assistant and Laya Decision Engine:
    1. PLAY vs LAB tone differentiation.
    2. Strict grounding to simulation telemetry (no hallucinated numbers).
    3. Cancer cure-claim interception (Hard Guardrail 1).
    4. Parameter modification approval guardrail (proposed_action).
    5. Laya triage and event classification with deterministic fallback resilience.
    """
    assistant = LocalGemmaAssistant()
    assistant.configure_provider(provider="deterministic")
    ctx = {
        "step": 125,
        "generation": 5,
        "alive_cells": 78,
        "current_compound": "Gemcitabine",
        "ph": 7.35,
        "resistant_cells": 18,
    }

    # 1. PLAY vs LAB tone
    ans_play = assistant.ask("What is happening?", simulation_context=ctx, mode=InterfaceMode.PLAY)
    ans_lab = assistant.ask("What is happening?", simulation_context=ctx, mode=InterfaceMode.LAB)

    assert "step 125" in ans_play["answer"]
    assert "step 125" in ans_lab["answer"]
    # Tone difference: LAB mode explicitly details microenvironment parameters
    assert "pH=7.35" in ans_lab["answer"] or "Normalized O2" in ans_lab["answer"]

    # 2. Cure claim interception
    cure_ans = assistant.ask("Can this candidate compound cure pancreatic cancer?", simulation_context=ctx, mode=InterfaceMode.LAB)
    assert cure_ans["guardrail_checks"]["cancer_cure_claim_prevented"] is True
    assert "cannot be claimed to 'cure'" in cure_ans["answer"] or "cannot claim" in cure_ans["answer"].lower()

    # 3. Parameter mutation intent requiring user approval
    mut_ans = assistant.ask("Please change pH to 6.8", simulation_context=ctx, mode=InterfaceMode.LAB)
    assert "proposed_action" in mut_ans
    prop_action = mut_ans["proposed_action"]
    assert prop_action["action"] == "set_environment_parameter"
    assert prop_action["parameter"] == "ph"
    assert prop_action["requested_value"] == 6.8
    assert prop_action["requires_user_approval"] is True

    # 4. Laya Decision Engine Triage & Resilient Fallback
    laya = LayaDecisionEngine()
    assert laya.provider in ("deterministic_validated_rules", "laya_multilingual_agent")

    test_candidates = [
        {
            "compound_id": "CMPD-001",
            "name": "Gemcitabine",
            "pic50": 7.35,
            "selectivity_hpde": 12.0,
            "qed": 0.61,
            "hepatotox_alert": False,
            "in_applicability_domain": True,
        },
        {
            "compound_id": "CMPD-002",
            "name": "D-Glucose",
            "pic50": 2.10,
            "selectivity_hpde": 1.0,
            "qed": 0.45,
            "hepatotox_alert": False,
            "in_applicability_domain": True,
        },
    ]

    triage_res = laya.triage_candidates(test_candidates)
    assert triage_res["total_evaluated"] == 2
    rankings = triage_res["rankings"]
    assert len(rankings) == 2
    # Gemcitabine should rank first due to higher potency and selectivity
    assert rankings[0]["name"] == "Gemcitabine"
    assert rankings[0]["status"] in ("approved_for_testing", "flagged_with_warnings")
    assert rankings[0]["composite_utility"] > rankings[1]["composite_utility"]

    # 5. Laya Event Classification
    event_ext = laya.classify_event([
        {"entity_count": 100},
        {"entity_count": 40},  # 60% drop
    ])
    assert event_ext["event_type"] == "extinction_wave"

    event_bloom = laya.classify_event([
        {"entity_count": 50},
        {"entity_count": 85},  # 70% increase
    ])
    assert event_bloom["event_type"] == "autocatalytic_bloom"


# =============================================================================
# 8. Clinical Safety Language Audit
# =============================================================================

def test_zero_forbidden_clinical_strings() -> None:
    """
    Epistemic Audit (Section 27):
    Verify that ZERO forbidden clinical strings appear in generated outputs:
    - 'proven cure'
    - 'safe for humans'
    - 'guaranteed clinical efficacy'
    """
    forbidden_phrases = [
        "proven cure",
        "safe for humans",
        "guaranteed clinical efficacy",
    ]

    bie = BiologicalIndicatorsEngine()
    ce = ChemistryEngine()
    gem = ce.resolve_compound_input("Gemcitabine")
    pbpk = SystemicPBPKEngine()
    pbpk_res = pbpk.simulate(gem, dose_mg=100.0)
    indicators = bie.compute_biological_indicators(systemic_pbpk_result=pbpk_res, compound=gem)

    indicators_str = str(indicators).lower()
    for phrase in forbidden_phrases:
        assert phrase not in indicators_str, f"Forbidden clinical phrase detected in indicators: '{phrase}'"

    assistant = LocalGemmaAssistant()
    ctx = {"alive_cells": 100, "current_compound": "Gemcitabine"}
    ans = assistant.ask("Tell me about the clinical efficacy and safety of this cure.", simulation_context=ctx)
    ans_str = str(ans).lower()

    for phrase in forbidden_phrases:
        assert phrase not in ans_str, f"Forbidden clinical phrase detected in assistant response: '{phrase}'"
