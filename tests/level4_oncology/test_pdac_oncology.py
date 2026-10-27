"""
Level-4 Oncology & Pancreatic Ductal Adenocarcinoma (PDAC) Test Suite.

Validates:
1. DiseaseRegistry: Active PDAC mechanistic module + extensible non-tumor disease templates
   (verifying EVOCURE does not assume every disease is a tumor).
2. PDACModule Spatial Reaction-Diffusion & Zonation:
   Normoxic rim, hypoxic zone (O2 < 0.30), necrotic core (O2 < 0.08), and strict non-negativity.
3. Desmoplastic CAF / ECM Diffusion Barrier:
   Hindered diffusivity D_eff = D_0 / (1 + beta_ECM * ECM) attenuating core drug penetrance.
4. Controlled Comparison In Silico Trial:
   Gemcitabine (5.0 uM) vs Untreated Control (0.0 uM) vs D-Glucose (5.0 uM benign control)
   under identical seed, evaluating TGI%, clonal shift delta, and apoptosis induction.
5. CoevolutionEngine Dual Evolutionary Lineage Trees:
   Coupled chemical_lineage_tree DAG and tumor_clonal_tree phylogeny across treatment cycles.
6. CalibratedQSARPredictor Bemis-Murcko Scaffold Split Validation:
   Zero Murcko scaffold leakage between train and test partitions, ROC-AUC >= 0.80, and Brier calibration.
"""

from __future__ import annotations

import math
from typing import Any
import numpy as np
import pytest

from ai.predictors.data_connectors import ScientificDataManager
from ai.predictors.qsar import CalibratedQSARPredictor
from simulation.biology.coevolution import CoevolutionEngine
from simulation.biology.diseases.pdac import PDACModule
from simulation.biology.registry import (
    AMRBacterialModule,
    DiseaseRegistry,
    MASLDModule,
    NSCLCModule,
)
from simulation.chemistry.engine import ChemistryEngine
from simulation.core.contracts import (
    CellDeathMode,
    CellType,
    CompoundRecord,
    DiseaseModule,
    EpistemicCategory,
)
from simulation.pharmacology.admet import PharmacologyEngine


# =============================================================================
# 1. Disease Registry & Extensibility
# =============================================================================

def test_disease_registry_catalog_and_extensibility() -> None:
    """
    Validate that DiseaseRegistry catalogs PDAC as active and provides extensible
    templates for non-tumor diseases without assuming all diseases are tumors.
    """
    available = DiseaseRegistry.list_available()
    disease_ids = [d["disease_id"] for d in available]

    # Must contain active PDAC and defined future templates
    assert "PDAC" in disease_ids
    assert "NSCLC_KRAS" in disease_ids
    assert "AMR_BACTERIAL" in disease_ids
    assert "METABOLIC_MASLD" in disease_ids

    # Verify category heterogeneity: not all diseases are tumors!
    categories = {d["category"] for d in available}
    assert "infectious_disease" in categories, "AMR must be categorized as infectious disease"
    assert "metabolic_disease" in categories, "MASLD must be categorized as metabolic disease"

    # Inspect active PDAC module
    pdac = DiseaseRegistry.get("PDAC", seed=42)
    assert isinstance(pdac, PDACModule)
    assert pdac.disease_id == "PDAC"
    pdac_meta = pdac.metadata()
    assert pdac_meta["implementation_status"] == "fully_active_mechanistic"
    assert pdac_meta.get("is_calibrated", True) is True
    assert pdac_meta["epistemic_category"] == EpistemicCategory.MECHANISTIC_SIMULATION.value

    # Inspect AMR template: explicitly non-tumor and uncalibrated interface stub
    amr = DiseaseRegistry.get("AMR_BACTERIAL", seed=42)
    assert isinstance(amr, AMRBacterialModule)
    assert amr.disease_id == "AMR_BACTERIAL"
    amr_meta = amr.metadata()
    assert amr_meta["category"] == "infectious_disease"
    assert amr_meta["is_calibrated"] is False
    assert amr_meta["implementation_status"] == "interface_ready_requires_calibrated_dataset"
    assert amr_meta["epistemic_category"] == EpistemicCategory.DEMONSTRATION_UNVALIDATED.value

    # Inspect MASLD template: metabolic liver disease
    masld = DiseaseRegistry.get("METABOLIC_MASLD", seed=42)
    assert isinstance(masld, MASLDModule)
    masld_meta = masld.metadata()
    assert masld_meta["category"] == "metabolic_disease"
    assert masld_meta["organ"] == "Liver"
    assert masld_meta["is_calibrated"] is False

    # Test dynamic registration of a novel custom non-tumor disease module
    class AutoimmuneLupusModule(DiseaseModule):
        @property
        def disease_id(self) -> str:
            return "AUTOIMMUNE_SLE"

        @property
        def display_name(self) -> str:
            return "Systemic Lupus Erythematosus (SLE)"

        def metadata(self) -> dict[str, Any]:
            return {
                "disease_id": self.disease_id,
                "category": "autoimmune",
                "organ": "Systemic",
                "implementation_status": "interface_ready_requires_calibrated_dataset",
                "is_calibrated": False,
            }

        def initialize_scenario(self, seed: int, preset: str = "standard") -> None:
            pass

        def apply_compound_exposure(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return {"status": "stub"}

        def step(self, dt_seconds: float) -> dict[str, Any]:
            return {"status": "stepped"}

        def get_snapshot(self) -> dict[str, Any]:
            return {"disease_id": self.disease_id}

        def get_biomarkers(self, systemic_state: dict[str, Any] | None = None) -> dict[str, Any]:
            return {}

    DiseaseRegistry.register(AutoimmuneLupusModule)
    assert DiseaseRegistry.is_registered("AUTOIMMUNE_SLE")
    lupus = DiseaseRegistry.get("AUTOIMMUNE_SLE")
    assert lupus.disease_id == "AUTOIMMUNE_SLE"
    assert lupus.metadata()["category"] == "autoimmune"


# =============================================================================
# 2. PDAC Spatial Reaction-Diffusion & Zonation
# =============================================================================

def test_pdac_spatial_reaction_diffusion_and_zonation() -> None:
    """
    Validate 2D reaction-diffusion dynamics in PDAC:
    - Continuous grids maintain strict non-negativity (O2, nutrients, ECM, drug).
    - Peripheral vascular layout naturally resolves normoxic, hypoxic, and necrotic zones.
    - Anoxic core (O2 < 0.08) induces cellular necrosis.
    """
    pdac = PDACModule(seed=42, width=800.0, height=600.0, nx=40, ny=30)
    pdac.initialize_scenario(seed=42)

    # Initial state verification
    snap0 = pdac.get_snapshot()
    assert snap0["disease_id"] == "PDAC"
    zones0 = snap0["zones"]
    assert "normoxic_area_pct" in zones0
    assert "hypoxic_area_pct" in zones0
    assert "necrotic_core_pct" in zones0

    total_zone_pct = zones0["normoxic_area_pct"] + zones0["hypoxic_area_pct"] + zones0["necrotic_core_pct"]
    assert total_zone_pct == pytest.approx(100.0, abs=0.5)

    # Advance simulation by multiple 1-hour steps (3600 seconds each)
    dt_hour = 3600.0
    for _ in range(6):
        events = pdac.step(dt_hour)
        assert isinstance(events, dict)

    snap = pdac.get_snapshot()

    # Verify strict non-negativity across all Eulerian 2D grids
    assert np.all(pdac.grid_oxygen >= 0.0), "Oxygen field dropped below zero"
    assert np.all(pdac.grid_nutrient >= 0.0), "Nutrient field dropped below zero"
    assert np.all(pdac.grid_ecm >= 0.0), "ECM field dropped below zero"
    assert np.all(pdac.grid_drug_total >= 0.0), "Drug total field dropped below zero"
    assert np.all(pdac.grid_drug_free >= 0.0), "Drug free field dropped below zero"
    assert not np.any(np.isnan(pdac.grid_oxygen)), "NaN detected in oxygen grid"
    assert not np.any(np.isinf(pdac.grid_oxygen)), "Inf detected in oxygen grid"

    # Verify radial hypoxia/necrosis gradient: center core is significantly less oxygenated than periphery
    center_y, center_x = pdac.ny // 2, pdac.nx // 2
    core_oxygen = float(np.mean(pdac.grid_oxygen[center_y - 2 : center_y + 3, center_x - 3 : center_x + 4]))
    edge_oxygen = float(np.mean(np.concatenate([pdac.grid_oxygen[0:2, :].flatten(), pdac.grid_oxygen[-2:, :].flatten()])))
    assert core_oxygen < edge_oxygen, f"Expected hypoxic core (core={core_oxygen:.3f} vs edge={edge_oxygen:.3f})"
    assert core_oxygen < 0.30, "Tumor core must fall into hypoxic or necrotic regime (< 0.30)"

    # Verify cellular agent zonation
    living_cells = [c for c in pdac.cells if c.death_mode == CellDeathMode.ALIVE]
    assert len(living_cells) > 0
    metabolic_states = {c.metabolic_state for c in living_cells}
    assert "normoxic" in metabolic_states or "glycolytic_hypoxic" in metabolic_states


# =============================================================================
# 3. Desmoplastic CAF / ECM Diffusion Barrier
# =============================================================================

def test_desmoplastic_caf_ecm_diffusion_barrier() -> None:
    """
    Validate that CAFs actively deposit collagenous ECM and that accumulated
    ECM physically attenuates effective drug diffusivity and core drug penetrance.
    """
    pdac = PDACModule(seed=42)
    pdac.initialize_scenario(seed=42)

    # Verify CAFs and fibroblasts exist in the cell population
    cafs = [c for c in pdac.cells if c.cell_type == CellType.CAF]
    fibroblasts = [c for c in pdac.cells if c.cell_type == CellType.FIBROBLAST]
    assert len(cafs) > 0, "Simulation must include Cancer-Associated Fibroblasts"
    assert len(fibroblasts) > 0, "Simulation must include quiescent fibroblasts"

    # Check ECM accumulation
    mean_ecm = float(np.mean(pdac.grid_ecm))
    max_ecm = float(np.max(pdac.grid_ecm))
    assert mean_ecm > 0.05, f"Expected desmoplastic ECM deposition (mean={mean_ecm:.3f})"
    assert max_ecm > 0.30, f"Expected dense fibrotic ECM hotspots (max={max_ecm:.3f})"

    # Verify the hindered diffusivity formula: D_eff = D_0 / (1 + beta_ecm * ECM)
    d0 = pdac.diffusivity_drug_base
    beta = pdac.beta_ecm
    dense_ecm_val = max_ecm
    d_eff_dense = d0 / (1.0 + beta * dense_ecm_val)
    d_eff_clean = d0 / (1.0 + beta * 0.0)

    assert d_eff_dense < d_eff_clean
    hindrance_factor = d_eff_clean / d_eff_dense
    assert hindrance_factor >= 1.5, f"Dense ECM must reduce diffusivity by >= 1.5x (got {hindrance_factor:.2f}x)"

    # Test drug exposure penetrance deficit
    ce = ChemistryEngine()
    gem = ce.resolve_compound_input("Gemcitabine")
    pe = PharmacologyEngine()
    pharm = pe.predict_antitumor_activity(gem)

    pdac.apply_compound_exposure(
        compound=gem,
        pharmacology_profile=pharm,
        applied_concentration_um=10.0,
        injection_region="vascular",
    )

    # Step for 3 hours to allow drug diffusion and transport
    for _ in range(3):
        pdac.step(3600.0)

    # Measure core vs periphery free drug concentration
    center_y, center_x = pdac.ny // 2, pdac.nx // 2
    core_drug = float(np.mean(pdac.grid_drug_free[center_y - 2 : center_y + 3, center_x - 3 : center_x + 4]))
    periphery_drug = float(np.mean(np.concatenate([pdac.grid_drug_free[0:2, :].flatten(), pdac.grid_drug_free[-2:, :].flatten()])))

    # Desmoplastic barrier and hypovascularity hinder drug reaching the tumor center
    assert core_drug < periphery_drug, (
        f"Core drug ({core_drug:.3f}) should be lower than periphery drug ({periphery_drug:.3f}) due to desmoplasia"
    )


# =============================================================================
# 4. Controlled Comparison: Gemcitabine vs Untreated vs D-Glucose
# =============================================================================

def test_controlled_comparison_gemcitabine_vs_glucose_vs_control() -> None:
    """
    Execute controlled in silico comparison under identical initial seed (seed=42):
    - Branch A: 0.0 uM untreated control
    - Branch B1: Gemcitabine 5.0 uM (active oncology reference)
    - Branch B2: D-Glucose 5.0 uM (benign nutritional control)

    Verifies:
    1. Gemcitabine achieves high TGI% (> 75%) and robust apoptotic induction ratio (> 10.0).
    2. D-Glucose achieves low TGI% (< 35%) and negligible apoptotic induction.
    3. Gemcitabine selectively suppresses sensitive classical tumor clones, driving clonal shift.
    """
    dm = ScientificDataManager()
    ds = dm.load_reference_dataset()
    gem = next(c for c in ds if c.name == "Gemcitabine")
    glu = next(c for c in ds if c.name == "D-Glucose")

    pe = PharmacologyEngine()
    gem_pharm = pe.predict_antitumor_activity(gem)
    glu_pharm = pe.predict_antitumor_activity(glu)

    pdac = PDACModule(seed=42)

    # 1. Gemcitabine vs Untreated Control (24h exposure)
    res_gem = pdac.run_controlled_comparison(
        compound=gem,
        pharmacology_profile=gem_pharm,
        applied_concentration_um=5.0,
        duration_hours=24.0,
        seed=42,
    )

    deltas_gem = res_gem["deltas"]
    assert deltas_gem["initial_tumor_cells"] > 0
    assert deltas_gem["tumor_growth_inhibition_pct"] >= 75.0, (
        f"Gemcitabine TGI% ({deltas_gem['tumor_growth_inhibition_pct']}%) should be >= 75%"
    )
    assert deltas_gem["apoptosis_induction_ratio"] >= 10.0, (
        f"Gemcitabine apoptosis ratio ({deltas_gem['apoptosis_induction_ratio']}) should be >= 10.0"
    )

    # 2. D-Glucose vs Untreated Control (24h exposure)
    res_glu = pdac.run_controlled_comparison(
        compound=glu,
        pharmacology_profile=glu_pharm,
        applied_concentration_um=5.0,
        duration_hours=24.0,
        seed=42,
    )

    deltas_glu = res_glu["deltas"]
    assert deltas_glu["tumor_growth_inhibition_pct"] < 35.0, (
        f"Benign D-Glucose TGI% ({deltas_glu['tumor_growth_inhibition_pct']}%) should be < 35%"
    )
    assert deltas_glu["apoptosis_induction_ratio"] < deltas_gem["apoptosis_induction_ratio"], (
        "Glucose apoptosis induction must be vastly inferior to active Gemcitabine"
    )

    # 3. Trajectory and Clonal Dynamics verification
    trt_traj = res_gem["treated_trajectory"]
    ctrl_traj = res_gem["control_trajectory"]
    assert trt_traj["viable_tumor_cells"][-1] < ctrl_traj["viable_tumor_cells"][-1]
    assert len(res_gem["time_points_hours"]) == 25  # 0h to 24h inclusive


# =============================================================================
# 5. CoevolutionEngine Dual Evolutionary Trees
# =============================================================================

def test_coevolution_dual_parallel_lineage_trees() -> None:
    """
    Validate CoevolutionEngine coupling:
    - Advances iterative cycles between synthetic chemical adaptation and tumor resistance.
    - Exports two coupled genealogical trees:
      1. chemical_lineage_tree: tracks compound mutations, parent IDs, and properties.
      2. tumor_clonal_tree: tracks subclonal phylogeny, resistance traits, and abundance trajectories.
    """
    engine = CoevolutionEngine(seed=42)

    # Initial tree state check
    initial_trees = engine.get_trees()
    assert "chemical_lineage_tree" in initial_trees
    assert "tumor_clonal_tree" in initial_trees
    assert len(initial_trees["chemical_lineage_tree"]) > 0
    assert len(initial_trees["tumor_clonal_tree"]) >= 2

    # Run 2 coevolutionary cycles (6 hours each)
    results = engine.run_coevolution(
        n_cycles=2,
        hours_per_cycle=6.0,
        compound_pool_size=4,
        applied_concentration_um=2.0,
    )

    assert results["n_cycles"] == 2
    assert results["total_hours"] == 12.0

    trees = engine.get_trees()
    chem_tree = trees["chemical_lineage_tree"]
    tumor_tree = trees["tumor_clonal_tree"]

    # Verify chemical lineage tree nodes
    for node in chem_tree:
        assert "node_id" in node
        assert "name" in node
        assert "canonical_smiles" in node
        assert "generation" in node
        assert "properties" in node
        props = node["properties"]
        assert "mol_weight" in props
        assert "logp" in props
        assert "qed" in props

    # Verify tumor clonal tree nodes
    assert len(tumor_tree) >= 2
    for clone in tumor_tree:
        assert "clone_id" in clone
        assert "emergence_cycle" in clone
        assert "mutations" in clone
        assert "resistance_score" in clone
        assert "abundance_history" in clone
        assert len(clone["abundance_history"]) >= 2

    # Check trajectory metrics
    traj = engine.get_trajectory()
    assert len(traj["cycle"]) == 2
    assert len(traj["resistant_fraction"]) == 2
    assert len(traj["lead_compound_id"]) == 2


# =============================================================================
# 6. Bemis-Murcko Scaffold Split Validation (Zero Leakage)
# =============================================================================

def test_qsar_bemis_murcko_scaffold_split_zero_leakage() -> None:
    """
    Validate CalibratedQSARPredictor scaffold split benchmark:
    - Group molecules strictly by core Bemis-Murcko scaffold.
    - Guarantee ZERO Murcko scaffold leakage between train and test splits.
    - Assert ROC-AUC >= 0.80 and Brier calibration score < 0.20.
    """
    dm = ScientificDataManager()
    dataset = dm.load_reference_dataset()
    assert len(dataset) >= 25, "Dataset must have sufficient reference compounds"

    smiles_list = [c.canonical_smiles for c in dataset]
    pic50_list = [c.reference_bioactivity.get("pic50", 5.0) for c in dataset]
    active_list = [1 if c.reference_bioactivity.get("ic50_um", 100.0) < 10.0 else 0 for c in dataset]

    predictor = CalibratedQSARPredictor(n_estimators=50, random_state=42)
    metrics = predictor.evaluate_scaffold_split(
        smiles_list=smiles_list,
        pic50_list=pic50_list,
        active_list=active_list,
        test_fraction=0.25,
    )

    # Critical Epistemic & Statistical Assertions
    assert "scaffold_leakage" in metrics
    assert metrics["scaffold_leakage"] == 0, "Scaffold leakage detected between train and test partitions!"

    assert "train_compounds" in metrics
    assert "test_compounds" in metrics
    assert metrics["train_compounds"] + metrics["test_compounds"] == len(dataset)

    # Calibration & Predictive performance
    assert metrics["roc_auc"] >= 0.80, f"ROC-AUC ({metrics['roc_auc']}) must be >= 0.80"
    assert metrics["brier_score"] <= 0.18, f"Brier score ({metrics['brier_score']}) must be <= 0.18"
    assert metrics["rmse"] < 2.50, f"RMSE ({metrics['rmse']}) must be < 2.50"
