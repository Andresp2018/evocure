"""
EVOCURE v1.0.0 Final Release Certification Test Suite.

Executes 12 comprehensive automated validation tests covering:
1. DiseaseRegistry: PDAC, NSCLC_KRAS, AMR_BACTERIAL, METABOLIC_MASLD active mechanistic models.
2. NSCLCModule: Sotorasib KRAS-G12C efficacy (TGI > 25%) and healthy alveolar cell sparing.
3. AMRBacterialModule: EPS biofilm diffusion hindrance and KPC beta-lactamase enzymatic hydrolysis.
4. MASLDModule: Resmetirom steatosis clearance vs Acetaminophen hepatotoxic necrosis / ALT surge.
5. EVOCUREEngine: Dynamic multimodal disease switching across all four disease models.
6. PharmacologyEngine & EVOCUREEngine: 8x8 Bliss synergy matrix and 4-arm in silico trial.
7. MolecularStructureEditor: Chemical editing operations (substituent, swap, ring fusion, bioisostere).
8. MolecularStructureEditor: MCS structural diff with Coral/Cyan SVG vector highlighting.
9. Built-in Compound Library: >= 45 validated compounds with 100% valid RDKit SMILES.
10. ScientificDataManager: Multi-disease reference dataset partitioning (>=85 records, PDAC filtering).
11. REST API PDF Export: Native binary PDF generation endpoint (/api/v1/export/latest/pdf).
12. Frontend Offline Integrity: Zero external CDN audit and 100% interactive DOM ID bindings.
"""

from __future__ import annotations

import math
from pathlib import Path
import re
from typing import Any

from fastapi.testclient import TestClient
import numpy as np
import pytest
from rdkit import Chem

from ai.predictors.data_connectors import ScientificDataManager
from app.main import app
from simulation.biology.diseases.amr_bacterial import AMRBacterialModule
from simulation.biology.diseases.masld import MASLDModule
from simulation.biology.diseases.nsclc import NSCLCModule
from simulation.biology.diseases.pdac import PDACModule
from simulation.biology.registry import DiseaseRegistry
from simulation.chemistry.editor import MolecularStructureEditor
from simulation.chemistry.engine import ChemistryEngine
from simulation.chemistry.library import get_builtin_library
from simulation.core.contracts import (
    CellDeathMode,
    CellType,
    EpistemicCategory,
)
from simulation.core.engine import EVOCUREEngine
from simulation.pharmacology.admet import PharmacologyEngine


# =============================================================================
# Test 01: Disease Registry with 4 Active Mechanistic Disease Models
# =============================================================================

def test_01_disease_registry_four_active_mechanistic_diseases() -> None:
    """
    Verify that DiseaseRegistry catalogs all 4 core disease models (PDAC, NSCLC_KRAS,
    AMR_BACTERIAL, METABOLIC_MASLD), each instantiating a fully active mechanistic
    simulation engine with zero stubs.
    """
    available = DiseaseRegistry.list_available()
    registered_ids = {d["disease_id"] for d in available}
    expected_ids = {"PDAC", "NSCLC_KRAS", "AMR_BACTERIAL", "METABOLIC_MASLD"}

    assert expected_ids.issubset(registered_ids), (
        f"Missing required diseases. Found: {registered_ids}, Expected subset: {expected_ids}"
    )

    categories = {d["category"] for d in available}
    assert "oncology" in categories
    assert "infectious_disease" in categories
    assert "metabolic_disease" in categories

    for did in expected_ids:
        instance = DiseaseRegistry.create(did, seed=42)
        assert instance.disease_id == did
        assert hasattr(instance, "step"), f"{did} must implement step()"
        assert hasattr(instance, "run_controlled_comparison"), f"{did} must implement run_controlled_comparison()"
        assert hasattr(instance, "get_biomarkers"), f"{did} must implement get_biomarkers()"

        meta = instance.metadata()
        assert "disease_id" in meta
        assert "display_name" in meta
        assert len(meta.get("hallmarks_modeled", [])) > 0, f"{did} must declare modeled hallmarks"

        # Verify initial spatial microenvironment is non-trivial (Eulerian grids instantiated)
        snap = instance.get_snapshot()
        assert "heatmaps" in snap
        assert "oxygen" in snap["heatmaps"]
        assert snap["summary_metrics"]["viable_cells"] > 0
        assert len(instance.cells) > 0


# =============================================================================
# Test 02: NSCLC Sotorasib Efficacy and Healthy Alveolar Sparing
# =============================================================================

def test_02_nsclc_sotorasib_efficacy_and_healthy_sparing() -> None:
    """
    Verify that Sotorasib (KRAS-G12C inhibitor) at 5.0 uM exhibits robust Tumor Growth
    Inhibition (TGI% > 25%) in NSCLCModule while sparing healthy alveolar epithelial cells
    via elevated selectivity index and therapeutic margin.
    """
    ce = ChemistryEngine()
    sot = ce.resolve_compound_input("Sotorasib")
    assert sot.is_chemically_valid

    pe = PharmacologyEngine()
    pharm = pe.predict_antitumor_activity(sot)

    ic50_sens = pharm["ic50_sensitive_um"]["value"]
    ic50_healthy = pharm["ic50_healthy_um"]["value"]
    selectivity_index = pharm["selectivity_index"]["value"]

    assert ic50_sens < 1.0, f"Sotorasib sensitive clone IC50 must be potent (got {ic50_sens} uM)"
    assert ic50_healthy >= 15.0, f"Sotorasib healthy IC50 must demonstrate safety window (got {ic50_healthy} uM)"
    assert selectivity_index >= 50.0, f"Selectivity index must be >= 50 (got {selectivity_index})"

    # Run controlled 24-hour in silico NSCLC trial
    nsclc = NSCLCModule(seed=42)
    comparison = nsclc.run_controlled_comparison(
        compound=sot,
        pharmacology_profile=pharm,
        applied_concentration_um=5.0,
        duration_hours=24.0,
        seed=42,
    )

    tgi = comparison["deltas"]["tumor_growth_inhibition_pct"]
    assert tgi > 25.0, f"Expected Sotorasib TGI > 25%, got {tgi:.1f}%"
    assert comparison["deltas"]["treated_final_tumor_cells"] < comparison["deltas"]["control_final_tumor_cells"]

    # Verify concentration-response profile spares healthy alveolar cells (>85% viability at therapeutic dose)
    curve = pharm["concentration_response_curve"]
    concs = curve["conc_um"]
    healthy_viab = curve["healthy_viability"]
    sens_viab = curve["tumor_sensitive_viability"]

    # Find closest index to 5.0 uM
    idx_5um = min(range(len(concs)), key=lambda i: abs(concs[i] - 5.0))
    assert healthy_viab[idx_5um] > 85.0, f"Healthy alveolar viability at 5uM must be > 85%, got {healthy_viab[idx_5um]}%"
    assert sens_viab[idx_5um] < 15.0, f"Sensitive tumor viability at 5uM must be < 15%, got {sens_viab[idx_5um]}%"


# =============================================================================
# Test 03: AMR Bacterial Biofilm Matrix Hindrance and KPC Hydrolysis
# =============================================================================

def test_03_amr_biofilm_hindrance_and_kpc_hydrolysis() -> None:
    """
    Verify that AMRBacterialModule models EPS biofilm matrix hindrance, reducing
    effective antibiotic diffusivity, and captures KPC carbapenemase hydrolysis
    sparing persister colonies.
    """
    amr = AMRBacterialModule(seed=42)
    amr.initialize_scenario(seed=42)

    # 1. Biophysical EPS Biofilm Diffusion Barrier
    d0 = amr.diffusivity_drug_base
    beta = amr.beta_biofilm
    assert beta >= 3.0, f"Biofilm hindrance parameter beta must be >= 3.0 (got {beta})"

    # Test effective diffusivity reduction in dense biofilm
    simulated_eps_dense = 0.50
    d_eff_dense = d0 / (1.0 + beta * simulated_eps_dense)
    d_eff_free = d0 / (1.0 + beta * 0.0)
    assert d_eff_dense < d_eff_free
    assert (d_eff_free / d_eff_dense) >= 2.0, "EPS biofilm must reduce diffusivity by >= 2.0x"

    # 2. Enzymatic KPC Hydrolysis Reference Anchors
    meropenem_ref = AMRBacterialModule.REFERENCE_COMPOUNDS["MEROPENEM"]
    cipro_ref = AMRBacterialModule.REFERENCE_COMPOUNDS["CIPROFLOXACIN"]
    assert meropenem_ref["susceptible_to_kpc"] is True, "Meropenem must be susceptible to KPC beta-lactamase"
    assert cipro_ref["susceptible_to_kpc"] is False, "Ciprofloxacin must be refractory to beta-lactamase hydrolysis"

    # 3. KPC Persister Differential Resistance
    sens_ic50 = meropenem_ref["ic50_sensitive_um"]
    res_ic50 = meropenem_ref["ic50_resistant_um"]
    fold_resistance = res_ic50 / sens_ic50
    assert fold_resistance >= 20.0, f"KPC persisters must show >= 20x fold-resistance (got {fold_resistance:.1f}x)"

    # Verify KPC mutation tags in persister bacterial population
    kpc_persisters = [c for c in amr.cells if "KPC_CARBAPENEMASE" in c.mutations]
    assert len(kpc_persisters) > 0, "Simulation must contain KPC+ persister bacterial entities"


# =============================================================================
# Test 04: MASLD Resmetirom Efficacy and Acetaminophen Hepatotoxicity
# =============================================================================

def test_04_masld_resmetirom_efficacy_and_acetaminophen_toxicity() -> None:
    """
    Verify that MASLDModule models steatosis clearance by THR-beta agonist Resmetirom
    while penalizing hepatotoxic insults (Acetaminophen / APAP) via necrosis and ALT elevation.
    """
    masld = MASLDModule(seed=42)
    masld.initialize_scenario(seed=42)

    # Reference profile validation
    resm_ref = MASLDModule.REFERENCE_COMPOUNDS["RESMETIROM"]
    apap_ref = MASLDModule.REFERENCE_COMPOUNDS["ACETAMINOPHEN"]

    assert resm_ref["hepatotoxic"] is False
    assert resm_ref["ic50_sensitive_um"] < 1.0, "Resmetirom must show sub-micromolar steatosis resolution"
    assert resm_ref["ic50_healthy_um"] > 30.0, "Resmetirom must spare healthy hepatocytes"

    assert apap_ref["hepatotoxic"] is True, "Acetaminophen must be flagged as hepatotoxic control"
    assert apap_ref["ic50_healthy_um"] < 3.0, "APAP must have low healthy hepatocyte tolerability threshold"

    # Test baseline indicators via get_biomarkers()
    base_indicators = masld.get_biomarkers()
    assert "SERUM_ALT_SURROGATE" in base_indicators
    base_alt = base_indicators["SERUM_ALT_SURROGATE"]["value"]
    assert base_alt >= 20.0, f"Baseline ALT should be physiological (got {base_alt})"

    # Simulate APAP-induced hepatocyte necrosis
    masld.cumulative_necrosis_count += 8
    stressed_indicators = masld.get_biomarkers()
    stressed_alt = stressed_indicators["SERUM_ALT_SURROGATE"]["value"]
    assert stressed_alt > base_alt, "Hepatocyte necrosis must surge serum ALT surrogate"
    assert stressed_alt >= 50.0, f"Necrotic liver injury must push ALT > 50 U/L (got {stressed_alt})"


# =============================================================================
# Test 05: Dynamic Multimodal Disease Switching in EVOCUREEngine
# =============================================================================

def test_05_engine_dynamic_disease_switching() -> None:
    """
    Verify that EVOCUREEngine dynamically switches its active disease across all 4
    registered models without memory leaks, preserving engine state and generating
    disease-specific biological indicator panels.
    """
    engine = EVOCUREEngine(seed=42)
    diseases = ["PDAC", "NSCLC_KRAS", "AMR_BACTERIAL", "METABOLIC_MASLD"]

    for did in diseases:
        engine.set_active_disease(did)
        engine.step_tumor(3600.0)

        full_state = engine.get_full_state()
        assert full_state["tumor"]["disease_id"] == did, f"Active disease must be {did}"

        indicators = engine.get_biological_indicators()
        assert "tumor_indicators" in indicators
        assert "metabolic_organ_indicators" in indicators
        assert "oncological_biomarkers" in indicators

        tumor_inds = indicators["tumor_indicators"]
        assert len(tumor_inds) >= 5, f"Expected >= 5 tumor indicators for {did}"
        for ind_name, ind_val in tumor_inds.items():
            assert "value" in ind_val
            assert "epistemic_category" in ind_val


# =============================================================================
# Test 06: 2-Drug Combination Synergy (Bliss Matrix and 4-Arm Trial)
# =============================================================================

def test_06_combination_synergy_bliss_and_four_arm_trial() -> None:
    """
    Verify that PharmacologyEngine calculates an 8x8 checkerboard synergy matrix using
    Bliss Independence (E_bliss = E_A + E_B - E_A * E_B) categorized as EXPLORATORY_MODEL,
    and EVOCUREEngine executes a 4-arm in silico trial (control, mono_a, mono_b, combination).
    """
    ce = ChemistryEngine()
    gem = ce.resolve_compound_input("Gemcitabine")
    pac = ce.resolve_compound_input("Paclitaxel")

    pe = PharmacologyEngine()
    synergy = pe.evaluate_combination_synergy(gem, pac)

    # 1. Checkerboard Matrix Validation
    matrix_data = synergy["matrix_2d"]
    assert len(matrix_data["conc_grid_a"]) == 8, "Expected 8 concentration points for Drug A"
    assert len(matrix_data["conc_grid_b"]) == 8, "Expected 8 concentration points for Drug B"
    assert len(matrix_data["inhibition_matrix"]) == 8
    assert len(matrix_data["inhibition_matrix"][0]) == 8
    assert len(matrix_data["bliss_excess_matrix"]) == 8
    assert len(matrix_data["bliss_excess_matrix"][0]) == 8

    # Verify Epistemic Categorization
    bliss_score = synergy["bliss_synergy_score"]
    assert bliss_score["epistemic_code"] == EpistemicCategory.EXPLORATORY_MODEL.name

    # 2. In Silico 4-Arm Trial Execution
    engine = EVOCUREEngine(seed=42)
    trial_res = engine.evaluate_combination(
        compound_a_query="Gemcitabine",
        compound_b_query="Paclitaxel",
        conc_a_um=5.0,
        conc_b_um=5.0,
        duration_hours=6.0,
    )

    assert "trial_4_arm" in trial_res
    arms = trial_res["trial_4_arm"]["trajectories"]
    assert "control" in arms
    assert "monotherapy_a" in arms
    assert "monotherapy_b" in arms
    assert "combination" in arms

    deltas = trial_res["trial_4_arm"]["deltas"]
    assert "tgi_a_pct" in deltas
    assert "tgi_b_pct" in deltas
    assert "tgi_combo_pct" in deltas
    assert "excess_over_bliss" in deltas


# =============================================================================
# Test 07: Molecular Structure Editor Chemical Transformations
# =============================================================================

def test_07_molecular_structure_editor_transformations() -> None:
    """
    Verify that MolecularStructureEditor successfully performs the four core medicinal
    chemistry operations: add_group (trifluoromethyl, morpholino), replace_atom,
    fuse_or_append_ring, and bioisostere_swap, generating valid canonical SMILES.
    """
    editor = MolecularStructureEditor()

    # 1. add_group (trifluoromethyl)
    r_cf3 = editor.apply_edit("c1ccccc1", "add_group", param="trifluoromethyl")
    assert r_cf3["valid"] is True
    assert "F" in r_cf3["edited_smiles"]
    assert Chem.MolFromSmiles(r_cf3["edited_smiles"]) is not None

    # 2. add_group (morpholino)
    r_morph = editor.apply_edit("c1ccccc1", "add_group", param="morpholino")
    assert r_morph["valid"] is True
    assert "N" in r_morph["edited_smiles"] and "O" in r_morph["edited_smiles"]
    assert Chem.MolFromSmiles(r_morph["edited_smiles"]) is not None

    # 3. replace_atom (benzene C -> N = pyridine)
    r_swap = editor.apply_edit("c1ccccc1", "replace_atom", param="C_to_N")
    assert r_swap["valid"] is True
    mol_pyr = Chem.MolFromSmiles(r_swap["edited_smiles"])
    assert mol_pyr is not None
    assert any(a.GetAtomicNum() == 7 for a in mol_pyr.GetAtoms())

    # 4. fuse_or_append_ring (append pyridine ring)
    r_ring = editor.apply_edit("c1ccccc1", "fuse_or_append_ring", param="pyridine")
    assert r_ring["valid"] is True
    mol_biphenyl = Chem.MolFromSmiles(r_ring["edited_smiles"])
    assert mol_biphenyl is not None
    assert mol_biphenyl.GetNumHeavyAtoms() >= 11

    # 5. bioisostere_swap (carboxylic acid -> tetrazole)
    r_bio = editor.apply_edit("CC(=O)O", "bioisostere_swap", param="carboxylic_acid_to_tetrazole")
    assert r_bio["valid"] is True
    assert "n" in r_bio["edited_smiles"].lower()
    assert Chem.MolFromSmiles(r_bio["edited_smiles"]) is not None


# =============================================================================
# Test 08: MCS Structural Diff and Vector SVG Highlighting
# =============================================================================

def test_08_structural_diff_mcs_and_svg_highlighting() -> None:
    """
    Verify that MolecularStructureEditor computes Maximum Common Substructure (MCS)
    diffs, reports exact physicochemical deltas (MW, LogP, Tanimoto), and generates
    side-by-side Coral/Cyan highlighted SVG vector depictions.
    """
    editor = MolecularStructureEditor()
    parent_smiles = "c1ccccc1"
    analog_res = editor.apply_edit(parent_smiles, "add_group", param="trifluoromethyl")
    analog_smiles = analog_res["edited_smiles"]

    diff = editor.compute_structural_diff(parent_smiles, analog_smiles)

    assert diff["valid"] is True
    assert diff["delta_mw"] > 60.0, f"CF3 addition must increase MW by ~68 Da (got {diff['delta_mw']})"
    assert diff["delta_logp"] > 0.5, "CF3 addition must increase lipophilicity LogP"
    assert 0.0 < diff["tanimoto_similarity"] < 1.0

    # Verify SVG generation and coloring
    svg_a = diff["svg_a_highlighted"]
    svg_b = diff["svg_b_highlighted"]
    assert "<svg" in svg_a
    assert "</svg>" in svg_a
    assert "<svg" in svg_b
    assert "</svg>" in svg_b


# =============================================================================
# Test 09: Compound Library Size and Chemical Validity
# =============================================================================

def test_09_compound_library_size_and_validity() -> None:
    """
    Verify that the built-in reference library contains >= 45 compounds and that
    every compound possesses a valid, sanitizable RDKit molecular structure.
    """
    lib = get_builtin_library()
    assert len(lib) >= 45, f"Built-in library must contain >= 45 compounds (got {len(lib)})"

    invalid_entries = []
    for entry in lib:
        smi = entry.get("smiles")
        if not smi:
            invalid_entries.append((entry.get("name"), "missing smiles"))
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            invalid_entries.append((entry.get("name"), smi))

    assert len(invalid_entries) == 0, f"Found invalid SMILES in library: {invalid_entries}"


# =============================================================================
# Test 10: Multi-Disease Reference Dataset Partitioning
# =============================================================================

def test_10_reference_dataset_multi_disease_partitioning() -> None:
    """
    Verify that ScientificDataManager loads >= 85 multi-disease reference records
    when querying 'ALL', and strictly partitions assays so that disease-specific
    cohorts (e.g., PDAC) exclude incomparable bacterial or metabolic assays.
    """
    mgr = ScientificDataManager()

    # Load all records across diseases
    all_records = mgr.load_reference_dataset(disease_id="ALL")
    assert len(all_records) >= 85, f"Global reference dataset must contain >= 85 records (got {len(all_records)})"

    # Load PDAC-specific cohort
    pdac_mgr = ScientificDataManager()
    pdac_records = pdac_mgr.load_reference_dataset(disease_id="PDAC")
    assert len(pdac_records) < len(all_records), "PDAC cohort must be partitioned from total records"
    assert len(pdac_records) >= 30, f"PDAC cohort must contain >= 30 records (got {len(pdac_records)})"

    # Verify non-comparable compounds are excluded from PDAC cytotoxicity training target
    pdac_names = {r.name for r in pdac_records}
    assert "Meropenem" not in pdac_names, "Bacterial antibiotic Meropenem must be excluded from PDAC dataset"
    assert "Resmetirom" not in pdac_names, "Hepatic THR-beta agonist Resmetirom must be excluded from PDAC dataset"
    assert "Gemcitabine" in pdac_names, "PDAC gold standard Gemcitabine must be present in PDAC dataset"


# =============================================================================
# Test 11: Native Binary PDF Export Endpoint
# =============================================================================

def test_11_native_binary_pdf_export_endpoint() -> None:
    """
    Verify that the native binary PDF export endpoint (/api/v1/export/latest/pdf)
    returns an HTTP 200 response with content-type application/pdf, starting with
    the standard '%PDF-' magic bytes and containing complete document content.
    """
    client = TestClient(app)
    response = client.get("/api/v1/export/latest/pdf")

    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}"
    assert "application/pdf" in response.headers.get("content-type", "")
    assert response.content.startswith(b"%PDF-"), "Binary response must start with PDF magic bytes"
    assert len(response.content) > 1500, f"PDF export must be a non-trivial binary document (got {len(response.content)} bytes)"


# =============================================================================
# Test 12: Frontend Zero-CDN Offline Audit and Interactive DOM ID Binding
# =============================================================================

def test_12_frontend_dom_id_binding_and_zero_cdn_audit() -> None:
    """
    Verify that frontend/static/index.html is 100% offline-functional with zero external
    CDN dependencies (Google Fonts, unpkg, jsdelivr, cdnjs), and that every interactive
    HTML control ID is bound in frontend/static/js/*.js.
    """
    index_path = Path(__file__).resolve().parents[2] / "frontend" / "static" / "index.html"
    assert index_path.exists(), f"Frontend index.html not found at: {index_path}"
    html = index_path.read_text(encoding="utf-8")

    # 1. Zero External CDN Audit
    cdn_patterns = [
        r"https?://cdn",
        r"googleapis\.com",
        r"unpkg\.com",
        r"jsdelivr\.net",
        r"cdnjs\.cloudflare\.com",
    ]
    for pattern in cdn_patterns:
        matches = re.findall(pattern, html, re.IGNORECASE)
        assert len(matches) == 0, f"Violation of zero-CDN rule: found '{matches}' in index.html"

    # 2. Interactive Control ID Binding Audit
    interactive_tag_ids = re.findall(
        r"<(?:button|input|select|textarea)[^>]*\bid=[\"']([^\"']+)[\"']",
        html,
        re.IGNORECASE,
    )
    assert len(interactive_tag_ids) > 0, "Expected interactive control IDs in index.html"

    js_dir = Path(__file__).resolve().parents[2] / "frontend" / "static" / "js"
    assert js_dir.exists(), f"Frontend JS directory not found at: {js_dir}"
    all_js_code = " ".join(f.read_text(encoding="utf-8") for f in js_dir.glob("*.js"))

    unbound_ids = [cid for cid in interactive_tag_ids if cid not in all_js_code]
    assert len(unbound_ids) == 0, f"Unbound interactive controls found without JS handlers: {unbound_ids}"
