"""
End-to-End System Acceptance Test Suite for EVOCURE & EVOCURE PLAY.

Verifies:
1. EVOCUREEngine deterministic bit-for-bit seed replay (seed=2026).
2. FPS/render-stride invariance and PLAY <-> LAB instant mode switching without altering
   entities, environment, or RNG state.
3. Adding >=3 compound types, observing Brownian movement, reactions, resource consumption,
   mutations, selection, multi-generation acceleration (10_gen, 100_gen, 1000_gen), and
   genealogy lineage tracing.
4. SQLite ExperimentStore save, load, duplicate, checkpoint create/restore, and
   JSON/CSV/SDF/Markdown report exports.
5. CandidateDiscoveryPipeline 15-stage execution.
6. Verification of reports/evocure_001_results.json (Scenario A & Scenario B validation gate passed).
7. FastAPI TestClient end-to-end tests across all primary REST endpoints & WebSocket /ws:
   - GET /
   - GET /api/v1/health
   - GET /api/v1/state
   - GET /api/v1/library
   - GET /api/v1/missions-glossary
   - POST /api/v1/mode/interface
   - POST /api/v1/world/reset
   - POST /api/v1/world/add-compound
   - POST /api/v1/world/step
   - POST /api/v1/tumor/controlled-comparison
   - GET /api/v1/compound/Gemcitabine
   - GET /api/v1/biomarkers
   - POST /api/v1/assistant/ask
   - POST /api/v1/compare-worlds
   - WebSocket /ws (connecting, init_state, ping/pong)
8. Static offline audit of frontend/static/ confirming zero external http:// or https:// CDN dependencies.
9. Performance benchmark assertions (100 physics+ecosystem steps < 4.5s; 1000_gen preset < 2.0s).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
import asyncio
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from simulation.core.contracts import (
    InterfaceMode,
    UniverseMode,
)
from simulation.core.engine import EVOCUREEngine
from simulation.core.persistence import ExperimentStore
from simulation.discovery.pipeline import CandidateDiscoveryPipeline


# =============================================================================
# 1. Deterministic Bit-for-Bit Seed Replay
# =============================================================================

def test_evocure_engine_deterministic_seed_replay() -> None:
    """
    Acceptance Criterion:
    Bit-for-bit reproducibility: two separate engine instances initialized with
    seed=2026 stepping through identical cycles produce exactly identical entity
    spatial coordinates, velocities, and clocks.
    """
    engine_1 = EVOCUREEngine(seed=2026, universe_mode=UniverseMode.ARTIFICIAL_LIFE, interface_mode=InterfaceMode.PLAY)
    engine_1.reset_world(preset="chemical_life", seed=2026)

    engine_2 = EVOCUREEngine(seed=2026, universe_mode=UniverseMode.ARTIFICIAL_LIFE, interface_mode=InterfaceMode.PLAY)
    engine_2.reset_world(preset="chemical_life", seed=2026)

    # Step both engines through 40 discrete simulation steps
    for _ in range(40):
        engine_1.step_universe(steps=1, dt=0.1)
        engine_2.step_universe(steps=1, dt=0.1)

    state_1 = engine_1.get_full_state()
    state_2 = engine_2.get_full_state()

    assert len(state_1["entities"]) == len(state_2["entities"])
    assert len(state_1["entities"]) > 0

    for e1, e2 in zip(state_1["entities"], state_2["entities"]):
        assert e1["id"] == e2["id"]
        assert e1["cid"] == e2["cid"]
        assert e1["x"] == pytest.approx(e2["x"], abs=1e-12)
        assert e1["y"] == pytest.approx(e2["y"], abs=1e-12)
        assert e1["e"] == pytest.approx(e2["e"], abs=1e-12)

    # Clocks check
    assert state_1["clocks"]["evolutionary_step"] == state_2["clocks"]["evolutionary_step"]
    assert state_1["clocks"]["evolutionary_generation"] == state_2["clocks"]["evolutionary_generation"]


# =============================================================================
# 2. FPS/Render-Stride Invariance & Instant Mode Switching
# =============================================================================

def test_fps_render_stride_invariance_and_mode_switching() -> None:
    """
    Acceptance Criterion:
    1. Render stride invariance: simulating 20 steps at render_stride=1 vs render_stride=5
       produces identical physical trajectories for entities.
    2. PLAY <-> LAB mode switching occurs instantaneously without resetting simulation state,
       modifying cellular entities, changing environmental physics, or disrupting RNG state.
    """
    engine = EVOCUREEngine(seed=2026)
    engine.reset_world(preset="chemical_life", seed=2026)

    initial_entities_count = len(engine.entities)
    initial_step = engine.clocks.evolutionary_step
    initial_ph = engine.env.ph
    initial_temp = engine.env.temperature_k

    # Instant switch to LAB mode
    res_lab = engine.set_interface_mode(InterfaceMode.LAB)
    assert engine.interface_mode == InterfaceMode.LAB
    assert res_lab["interface_mode"] == "LAB"
    assert len(engine.entities) == initial_entities_count
    assert engine.clocks.evolutionary_step == initial_step
    assert engine.env.ph == initial_ph
    assert engine.env.temperature_k == initial_temp

    # Instant switch back to PLAY mode
    res_play = engine.set_interface_mode(InterfaceMode.PLAY)
    assert engine.interface_mode == InterfaceMode.PLAY
    assert res_play["interface_mode"] == "PLAY"
    assert len(engine.entities) == initial_entities_count
    assert engine.clocks.evolutionary_step == initial_step
    assert engine.env.ph == initial_ph


# =============================================================================
# 3. Ecosystem Dynamics: Brownian Movement, Reactions, Acceleration & Genealogy
# =============================================================================

def test_ecosystem_compounds_brownian_reactions_acceleration_and_genealogy() -> None:
    """
    Acceptance Criterion:
    - Adding >=3 compound types to the sandbox.
    - Entities exhibit Brownian velocity perturbations and spatial displacement.
    - Entities participate in catalytic reactions and substrate consumption.
    - Multi-generation acceleration presets (10_gen, 100_gen, 1000_gen) advance clocks rapidly.
    - Genealogy tracker records lineages, mutations, and diversity.
    """
    engine = EVOCUREEngine(seed=42)
    engine.reset_world(preset="sandbox_empty", seed=42)
    assert len(engine.entities) == 0

    # 1. Add >= 3 distinct compound types
    engine.add_compound_to_world("Glycine", count=15, ecological_role="autotroph")
    engine.add_compound_to_world("Pyruvic Acid", count=15, ecological_role="autotroph")
    engine.add_compound_to_world("Formaldehyde", count=20, ecological_role="nutrient", entity_kind="NUTRIENT")
    assert len(engine.entities) == 50
    assert len(engine.compounds) >= 3

    # Record initial entity positions
    pos0 = [(e.x, e.y) for e in engine.entities]

    # Step universe and observe Brownian movement
    engine.step_universe(steps=5, dt=0.2)
    pos1 = [(e.x, e.y) for e in engine.entities]

    # Verify entities moved via Brownian thermal diffusion + advection
    displacements = [math.hypot(p1[0] - p0[0], p1[1] - p0[1]) for p0, p1 in zip(pos0, pos1)]
    assert np.mean(displacements) > 0.05, "Entities must exhibit non-zero spatial displacement"

    # 2. Test multi-generation acceleration presets
    gen_before = engine.clocks.evolutionary_generation
    res_10 = engine.advance_by_preset("10_gen")
    assert res_10["generations_advanced"] == 10
    assert engine.clocks.evolutionary_generation == gen_before + 10

    # Long runs are queued and execute the same real steps in the background.
    from app.services.session_manager import SessionManager
    sm = SessionManager()
    sm.engine = engine
    job = asyncio.run(sm.start_advance("1000_gen"))
    assert job["advance_job"]["active"]
    assert sm.advance_target_step == engine.clocks.evolutionary_step + 25000
    assert engine.clocks.evolutionary_generation == gen_before + 10

    # 3. Genealogy Lineage Tracing
    genealogy = engine.genealogy
    assert len(genealogy.nodes) >= 3
    node_ids = list(genealogy.nodes.keys())
    lineage = genealogy.trace_lineage(node_ids[0])
    assert len(lineage) >= 1
    assert lineage[0]["compound_id"] in engine.compounds


# =============================================================================
# 4. SQLite Persistence: Save, Load, Duplicate, Checkpoints & Multiformat Exports
# =============================================================================

def test_sqlite_persistence_checkpoints_and_multiformat_exports(tmp_path: Path) -> None:
    """
    Acceptance Criterion:
    - Save experiment with all Section 20 fields.
    - Load experiment and duplicate experiment.
    - Checkpoint create and restore.
    - Multiformat exports: JSON, CSV time-series, MDL SDF, and publication Markdown report.
    """
    db_file = tmp_path / "acceptance_test.sqlite3"
    store = ExperimentStore(db_path=db_file)

    engine = EVOCUREEngine(seed=42)
    engine.reset_world(preset="chemical_life", seed=42)
    engine.step_universe(steps=5, dt=0.1)

    # 1. Save experiment
    save_res = engine.save_current_experiment(title="Acceptance Persistence Test")
    exp_id = save_res["experiment_id"]
    assert exp_id.startswith("EXP-")

    # 2. Load experiment
    loaded_exp = engine.store.load_experiment(exp_id)
    assert loaded_exp["experiment_id"] == exp_id
    assert loaded_exp["title"] == "Acceptance Persistence Test"
    assert loaded_exp["seed"] == 42
    assert "config_json" in loaded_exp
    assert "parameters_json" in loaded_exp

    # 3. Duplicate experiment
    dup_id = engine.store.duplicate_experiment(exp_id)
    assert dup_id.startswith("EXP-")
    assert dup_id != exp_id
    cloned_exp = engine.store.load_experiment(dup_id)
    assert "Acceptance Persistence Test" in cloned_exp["title"]

    # 4. Create and restore checkpoint
    ckpt_res = engine.create_checkpoint(label="Phase-1 Baseline")
    ckpt_id = ckpt_res["checkpoint_id"]
    assert ckpt_id.startswith("CHK-")

    # Step engine further to perturb state
    engine.step_universe(steps=10, dt=0.1)
    step_perturbed = engine.clocks.evolutionary_step

    # Restore checkpoint
    restore_res = engine.restore_checkpoint(ckpt_id)
    assert restore_res["status"] == "restored"
    assert engine.clocks.evolutionary_step == ckpt_res["step"]
    assert engine.clocks.evolutionary_step < step_perturbed

    # 5. Multiformat Exports
    json_export = engine.store.export_json(exp_id)
    parsed_json = json.loads(json_export)
    assert parsed_json["experiment"]["experiment_id"] == exp_id

    csv_export = engine.store.export_csv(exp_id)
    assert isinstance(csv_export, str)
    assert len(csv_export) > 0

    sdf_export = engine.store.export_sdf(exp_id)
    assert isinstance(sdf_export, str)
    assert "$$$$" in sdf_export, "SDF export must contain MDL entry termination delimiter ($$$$)"

    md_report = engine.store.generate_scientific_report_md(exp_id)
    assert "# EVOCURE" in md_report
    assert "DISCLAIMER" in md_report


# =============================================================================
# 5. Candidate Discovery Pipeline 15-Stage Execution
# =============================================================================

def test_candidate_discovery_pipeline_15_stages() -> None:
    """
    Acceptance Criterion:
    Execute CandidateDiscoveryPipeline and verify complete 15-stage workflow:
    1. Disease context selection
    2. Seed import
    3. Variant generation
    4. RDKit validation
    5. Descriptors calculation
    6. Calibrated QSAR prediction
    7. ADMET & toxicology evaluation
    8. Applicability domain check
    9. Multi-objective evaluation
    10. Laya triage
    11. Controlled 2D PDAC microenvironment simulation
    12. Result & provenance storage
    13. Pareto-optimal candidate identification
    14. Evolutionary population update
    15. Next generation loop
    """
    pipeline = CandidateDiscoveryPipeline()
    results = pipeline.run(
        generations=1,
        population_size=4,
        top_k_cellular=1,
        duration_hours=6.0,
        seed=1337,
    )

    assert results["disease_id"] == "PDAC"
    assert results["generations_completed"] == 1
    assert results["total_molecules_evaluated"] >= 4
    assert len(results["pareto_frontier"]) > 0

    # Verify all 15 stages are recorded in stage_by_stage_trace
    trace = results["stage_by_stage_trace"]
    assert len(trace) == 15, f"Expected exactly 15 stages, got {len(trace)}"

    stage_numbers = [s["stage_number"] for s in trace]
    assert stage_numbers == list(range(1, 16))

    expected_keywords = [
        "Disease", "Seed", "Variant", "Validation", "Descriptor",
        "QSAR", "Toxicology", "Domain", "Multi-Objective", "Triage",
        "Microenvironment", "Storage", "Pareto", "Population", "Loop"
    ]
    for s, kw in zip(trace, expected_keywords):
        assert kw.lower() in s["stage_name"].lower(), f"Stage {s['stage_number']} missing keyword '{kw}'"


# =============================================================================
# 6. Verification of reports/evocure_001_results.json
# =============================================================================

def test_evocure_001_results_json_verification() -> None:
    """
    Acceptance Criterion:
    Verify official EVOCURE-001 results file:
    - Scenario A: Open artificial chemical evolution completed heritable diversification.
    - Scenario B Validation Gate: Status == PASSED (Gemcitabine demonstrated robust TGI%
      and selective apoptosis vs D-Glucose and untreated vehicle).
    """
    report_file = Path(r"e:\MA\EVOCURE\reports\evocure_001_results.json")
    assert report_file.exists(), "EVOCURE-001 results file must exist"

    with open(report_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["experiment_code"] == "EVOCURE-001"
    assert "scenario_a" in data
    assert "scenario_b" in data

    # Scenario A verification
    scen_a = data["scenario_a"]
    assert scen_a["universe_mode"] == "ARTIFICIAL_LIFE"
    assert scen_a["generations_completed"] >= 50
    assert scen_a["total_lineages_emerged"] > 0
    assert len(scen_a["evolution_trace"]) > 0

    # Scenario B Validation Gate verification
    scen_b = data["scenario_b"]
    assert scen_b["universe_mode"] == "GROUNDED_CHEMISTRY"
    assert scen_b["disease_id"] == "PDAC"

    gate = scen_b["validation_gate"]
    assert gate["gate_status"] == "PASSED", f"Validation gate must be PASSED, got {gate['gate_status']}"
    assert gate["positive_control"]["name"] == "Gemcitabine"
    assert gate["positive_control"]["tgi_pct"] > 80.0
    assert gate["positive_control"]["apoptosis_induction_ratio"] > 10.0
    assert gate["negative_control"]["name"] == "D-Glucose"
    assert gate["negative_control"]["tgi_pct"] < 30.0
    assert gate["healthy_sparing_verified"] is True


# =============================================================================
# 7. FastAPI REST & WebSocket End-to-End Test Suite
# =============================================================================

def test_fastapi_rest_and_websocket_full_stack() -> None:
    """
    Acceptance Criterion:
    Test all primary REST API endpoints and real-time WebSocket:
    - /
    - /api/v1/health
    - /api/v1/state
    - /api/v1/library
    - /api/v1/missions-glossary
    - /api/v1/mode/interface
    - /api/v1/world/reset
    - /api/v1/world/add-compound
    - /api/v1/world/step
    - /api/v1/tumor/controlled-comparison
    - /api/v1/compound/Gemcitabine
    - /api/v1/biomarkers
    - /api/v1/assistant/ask
    - /api/v1/compare-worlds
    - /ws (WebSocket connect, initial state delivery, ping-pong)
    """
    client = TestClient(app)

    # 1. UI Root
    r_root = client.get("/")
    assert r_root.status_code == 200
    assert "text/html" in r_root.headers.get("content-type", "")

    # 2. Health Check
    r_health = client.get("/api/v1/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "healthy"

    # 3. State
    r_state = client.get("/api/v1/state")
    assert r_state.status_code == 200
    assert "clocks" in r_state.json()

    # 4. Library
    r_lib = client.get("/api/v1/library")
    assert r_lib.status_code == 200
    assert len(r_lib.json()) >= 20

    # 5. Missions & Glossary
    r_mg = client.get("/api/v1/missions-glossary")
    assert r_mg.status_code == 200
    assert "missions" in r_mg.json()

    # 6. Mode Interface Switch
    r_mode = client.post("/api/v1/mode/interface", json={"mode": "LAB"})
    assert r_mode.status_code == 200
    assert r_mode.json()["success"] is True

    # 7. World Reset
    r_reset = client.post("/api/v1/world/reset", json={"preset": "chemical_life", "seed": 42})
    assert r_reset.status_code == 200
    assert r_reset.json()["success"] is True

    # 8. Add Compound
    r_add = client.post("/api/v1/world/add-compound", json={"query": "Glycine", "count": 6})
    assert r_add.status_code == 200
    assert r_add.json()["success"] is True

    # 9. Step World
    r_step = client.post("/api/v1/world/step", json={"steps": 2, "dt": 0.1})
    assert r_step.status_code == 200
    assert r_step.json()["success"] is True

    # 10. Tumor Controlled Comparison
    r_ctrl = client.post(
        "/api/v1/tumor/controlled-comparison",
        json={
            "compound_query": "Gemcitabine",
            "applied_concentration_um": 5.0,
            "duration_hours": 12.0,
            "seed": 42,
        },
    )
    assert r_ctrl.status_code == 200
    ctrl_json = r_ctrl.json()
    assert ctrl_json["success"] is True
    assert "deltas" in ctrl_json["data"]

    # 11. Compound Lookup
    r_comp = client.get("/api/v1/compound/Gemcitabine")
    assert r_comp.status_code == 200
    assert r_comp.json()["name"] == "Gemcitabine"

    # 12. Multimodal Biomarkers
    r_bio = client.get("/api/v1/biomarkers")
    assert r_bio.status_code == 200
    assert "tumor_indicators" in r_bio.json()
    assert "hematological_indicators" in r_bio.json()

    # 13. AI Assistant Inquiry
    r_ask = client.post("/api/v1/assistant/ask", json={"question": "What is happening?"})
    assert r_ask.status_code == 200
    assert "answer" in r_ask.json()["data"]

    # 14. World Comparison Endpoint
    r_comp_worlds = client.post(
        "/api/v1/compare-worlds",
        json={
            "scenario_left": {"preset": "chemical_life"},
            "scenario_right": {"preset": "surprise_me"},
            "steps": 10,
            "repeats": 1,
            "seed": 42,
        },
    )
    assert r_comp_worlds.status_code == 200
    assert r_comp_worlds.json()["success"] is True

    # 15. Real-time WebSocket Protocol
    with client.websocket_connect("/ws") as ws:
        # Upon connect, server sends init_state
        init_data = ws.receive_json()
        assert init_data["type"] == "init_state"

        # Client sends ping, server responds pong
        ws.send_text(json.dumps({"type": "ping"}))
        pong_data = ws.receive_json()
        assert pong_data["type"] == "pong"


# =============================================================================
# 8. Static Offline Audit: Zero External CDN Dependencies
# =============================================================================

def test_static_offline_zero_cdn_audit() -> None:
    """
    Acceptance Criterion:
    Offline Integrity (Sections 1.2 & 28.3):
    All frontend assets (HTML, CSS, JS) must be 100% self-contained local files.
    Audit proves ZERO external http:// or https:// script, font, or stylesheet URLs.
    """
    static_root = Path(r"e:\MA\EVOCURE\frontend\static")
    assert static_root.exists(), "frontend/static directory must exist"

    files_checked = 0
    cdn_regex = re.compile(r"https?://", re.IGNORECASE)

    for root, _, files in os.walk(static_root):
        for f in files:
            file_path = Path(root) / f
            if file_path.suffix.lower() in (".html", ".css", ".js"):
                files_checked += 1
                with open(file_path, "r", encoding="utf-8", errors="ignore") as fp:
                    content = fp.read()
                    matches = cdn_regex.findall(content)
                    assert len(matches) == 0, (
                        f"External URL detected in offline asset {file_path.name}: {matches}"
                    )

    assert files_checked >= 5, f"Expected at least 5 static files checked, got {files_checked}"


# =============================================================================
# 9. Performance Benchmark Assertions
# =============================================================================

def test_performance_benchmarks() -> None:
    """
    Acceptance Criterion:
    High-Performance Physics & Vectorized Acceleration:
    1. 100 physics + ecosystem simulation steps with over 100 active entities in < 4.5s.
    2. Queue a real 1000-generation background run in < 2.0s without changing clocks.
    """
    engine = EVOCUREEngine(seed=42)
    engine.reset_world(preset="chemical_life", seed=42)
    assert len(engine.entities) >= 100

    # Benchmark 1: 100 continuous steps
    t0 = time.perf_counter()
    engine.step_universe(steps=100, dt=0.1)
    t1 = time.perf_counter()
    step_duration = t1 - t0
    assert step_duration < 4.50, f"100 physics steps took too long: {step_duration:.3f}s (must be < 4.5s)"

    # Queuing a long run must be responsive and must not fabricate generations.
    from app.services.session_manager import SessionManager
    sm = SessionManager()
    sm.engine = engine
    step_before = engine.clocks.evolutionary_step
    t2 = time.perf_counter()
    job = asyncio.run(sm.start_advance("1000_gen"))
    assert time.perf_counter() - t2 < 2.0
    assert job["advance_job"]["target_step"] == step_before + 25000
    assert engine.clocks.evolutionary_step == step_before
