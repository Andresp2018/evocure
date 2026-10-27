"""Regressions for the desktop simulation, persistence, and export repair."""
import asyncio
from collections import Counter
from dataclasses import asdict
import json
import re
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient
from simulation.core.engine import EVOCUREEngine
from simulation.core.contracts import EntityKind, UniverseMode
from app.services.session_manager import SessionManager
from app.main import app

@pytest.fixture
def engine(tmp_path, monkeypatch):
    monkeypatch.setenv("EVOCURE_DISABLE_LAYA_NEURAL", "1")
    return EVOCUREEngine(seed=314, db_path=tmp_path / "world.sqlite3")

def particles(engine):
    return [asdict(e) for e in engine.entities]

def atoms(engine):
    total = Counter()
    for e in engine.entities:
        if e.alive:
            total.update(engine.compounds[e.compound_id].atom_counts)
    return total

def test_collision_consumes_reactants_and_conserves_atoms(engine, monkeypatch):
    engine.reset_world("sandbox_empty")
    engine.add_compound_to_world("C", count=2, entity_kind="MOLECULE", x=400, y=300)
    original = list(engine.entities)
    monkeypatch.setattr(engine.physics, "step_entities", lambda **kw: {"collision_pairs": [(original[0], original[1])]})
    before = atoms(engine)
    for _ in range(20):
        engine.step_universe()
        if engine.reactions_count:
            break
    assert atoms(engine) == before
    assert len(engine.entities) == 1
    assert engine.entities[0].compound_id not in {e.compound_id for e in original}
    assert engine.reactions_count == 1

def test_checkpoint_restores_exact_particles_rng_fields_and_lineage(engine):
    engine.step_universe(3)
    checkpoint = engine.create_checkpoint()
    expected = particles(engine)
    expected_rng = json.dumps(engine.rng.bit_generator.state)
    expected_grid = engine.grid.fields["substrate"].copy()
    engine.step_universe(4)
    future = particles(engine)
    engine.restore_checkpoint(checkpoint["checkpoint_id"])
    assert particles(engine) == expected
    assert json.dumps(engine.rng.bit_generator.state) == expected_rng
    assert np.array_equal(engine.grid.fields["substrate"], expected_grid)
    assert all(e.compound_id in engine.compounds for e in engine.entities)
    assert engine.darwinian.genealogy_tracker is engine.genealogy
    engine.step_universe(4)
    assert particles(engine) == future

def test_saved_world_reopens_after_reset(engine):
    engine.step_universe(3)
    expected = particles(engine)
    saved = engine.save_current_experiment("Test universe")
    engine.reset_world("sandbox_empty")
    engine.load_saved_experiment(saved["experiment_id"])
    assert particles(engine) == expected

def test_grounded_molecules_do_not_reproduce(engine):
    engine.reset_world("sandbox_empty")
    engine.universe_mode = UniverseMode.GROUNDED_CHEMISTRY
    engine.add_compound_to_world("Water", count=1, entity_kind="MOLECULE")
    engine.entities[0].energy = 100
    engine.entities[0].stored_substrate = 100
    engine.step_universe(30)
    assert len(engine.entities) == 1
    assert engine.entities[0].generation == 0

def test_real_advancement_matches_manual_steps(engine):
    engine.reset_world("sandbox_empty")
    engine.add_compound_to_world("Water", count=1, entity_kind="MOLECULE")
    checkpoint = engine.create_checkpoint()
    result = engine.advance_by_preset("10_gen")
    expected = particles(engine)
    assert engine.clocks.evolutionary_step == 250
    assert result["generations_advanced"] == 10
    engine.restore_checkpoint(checkpoint["checkpoint_id"])
    engine.step_universe(250)
    assert particles(engine) == expected

def test_chat_receives_actual_population(engine):
    engine.assistant.configure_provider(provider="deterministic")
    answer = engine.ask_assistant("What is happening?")
    assert str(len(engine.entities)) in answer["answer"]
    assert answer["provider"] == "deterministic_grounded_assistant"

def test_assets_and_export_presets_are_present():
    root = Path(__file__).resolve().parents[2] / "godot_frontend"
    missing = []
    for script in root.rglob("*.gd"):
        for resource in re.findall(r'"(res://[^"\n]+\.(?:png|svg|gdshader|gd|tscn))"', script.read_text(encoding="utf-8")):
            if not (root / resource.removeprefix("res://")).is_file():
                missing.append(resource)
    assert not missing
    assert "Windows Desktop" in (root / "export_presets.cfg").read_text()

def test_api_genealogy_playback_jobs_and_all_exports(engine, monkeypatch):
    sm = SessionManager()
    sm.engine = engine
    monkeypatch.setattr(SessionManager, "_instance", sm)
    client = TestClient(app)
    assert client.get("/api/v1/genealogy").status_code == 200
    result = client.post("/api/v1/playback", json={"action": "play", "speed_multiplier": 10, "target": "universe"}).json()
    assert result["data"]["speed_multiplier"] == 10
    client.post("/api/v1/playback", json={"action": "pause"})
    job = client.post("/api/v1/world/advance-preset", json={"preset": "1000_gen"}).json()["data"]
    assert job["advance_job"]["active"]
    assert engine.clocks.evolutionary_step == 0
    assert job["advance_job"]["target_step"] == 25000
    client.post("/api/v1/playback", json={"action": "pause"})
    saved = client.post("/api/v1/experiments/save", json={"title": "Exports"}).json()["data"]["experiment_id"]
    for fmt in ("json", "csv", "sdf", "md", "pdf"):
        response = client.get(f"/api/v1/export/{saved}/{fmt}")
        assert response.status_code == 200, (fmt, response.text[:200])
        assert len(response.content) > 20
        if fmt == "pdf":
            assert response.content.startswith(b"%PDF")
        if fmt == "sdf":
            assert "$$$$" in response.text
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "init_state"


def test_stir_changes_particle_motion_without_creating_reactions(engine):
    before = [(e.vx, e.vy) for e in engine.entities]
    positions = [(e.x, e.y) for e in engine.entities]
    result = engine.stir_mixture()
    assert result["entities_stirred"] == len(before)
    assert before != [(e.vx, e.vy) for e in engine.entities]
    assert engine.reactions_count == 0
    engine.step_universe()
    assert positions != [(e.x, e.y) for e in engine.entities]


def test_tumor_checkpoint_continues_same_random_trajectory(engine):
    engine.step_tumor(steps=2)
    checkpoint = engine.create_checkpoint()
    engine.step_tumor(steps=2)
    expected = engine.tumor_module.get_snapshot()
    engine.restore_checkpoint(checkpoint["checkpoint_id"])
    engine.step_tumor(steps=2)
    assert engine.tumor_module.get_snapshot() == expected
