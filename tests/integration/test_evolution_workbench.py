import asyncio
import copy
import json
import time
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.services.session_manager import SessionManager
from app.services.evolution_workbench import EvolutionWorkbench
from simulation.core.engine import EVOCUREEngine
from simulation.discovery.workbench import compare_drugs, plan_guided_search
from simulation.biology.registry import DiseaseRegistry
from simulation.core.acceleration import compute_device, status as compute_status
from simulation.chemistry.engine import ChemistryEngine

@pytest.fixture
def session(tmp_path, monkeypatch):
    monkeypatch.setenv("EVOCURE_DISABLE_LAYA_NEURAL", "1")
    monkeypatch.setenv("EVOCURE_COMPUTE", "cpu")
    sm=SessionManager()
    sm.engine=EVOCUREEngine(seed=9, db_path=tmp_path/"world.sqlite")
    monkeypatch.setattr(sm.engine.assistant, "is_local_server_running", lambda:False)
    monkeypatch.setattr(SessionManager, "_instance", sm)
    return sm


def test_reference_comparison_exact_similarity_and_stereochemistry():
    chemistry=ChemistryEngine()
    aspirin=compare_drugs(chemistry.resolve_compound_input("Aspirin"))
    assert aspirin["exact_reference_name"] == "Aspirin"
    assert aspirin["matches"][0]["similarity"] == 1
    assert all(v==0 for v in aspirin["matches"][0]["candidate_minus_reference"].values())
    changed=compare_drugs(chemistry.build_compound("CCC(=O)Oc1ccccc1C(=O)O"))
    assert changed["exact_reference_name"] is None
    assert 0 < changed["matches"][0]["similarity"] < 1
    assert "does not establish novelty" in changed["scope"]


def test_prompt_becomes_bounded_executable_policy():
    plan=plan_guided_search("Find 5 potential cures for pancreatic cancer with low toxicity")
    assert plan["candidate_count"]==5
    assert plan["objective_weights"]["low_toxicological_risk"] > plan["objective_weights"]["qed_druglikeness"]
    assert "not cures" in plan["limitation"]
    assert plan["provider"]=="local_rule_based_planner"
    assert plan_guided_search("find 99 candidates")["candidate_count"]==20
    assert plan_guided_search("Find 5 candidates based on 5-Fluorouracil")["seed_queries"] == ["5-Fluorouracil"]


def test_live_registry_contains_real_mutants_and_no_samples(session):
    e=session.engine
    parent=e.compounds[next(iter(e.compounds))]
    child=e._mutate_compound_callback(parent,e.rng)
    e.genealogy.register_compound(child,birth_step=12)
    wb=EvolutionWorkbench(session)
    result=wb.compounds(source="universe")
    assert child.compound_id in {item["compound_id"] for item in result["compounds"]}
    assert all(not item["compound_id"].startswith("EVOCURE-00") for item in result["compounds"])
    assert result["registered_compounds"]==len(e.compounds)


@pytest.mark.parametrize("disease,reference",[("PDAC","Gemcitabine"),("NSCLC_KRAS","Sotorasib"),("AMR_BACTERIAL","Meropenem"),("METABOLIC_MASLD","Resmetirom")])
def test_every_disease_exposure_animated_state_and_restore(session,disease,reference):
    e=session.engine
    e.set_active_disease(disease)
    exposure=e.introduce_compound_to_tumor(reference,5,injection_region="uniform")
    assert exposure["model_info"]["status"].startswith("exploratory")
    assert np.max(e.tumor_module.grid_drug_free)>0
    e.step_tumor(2)
    state=e.get_full_state()
    assert state["tumor"]["disease_id"]==disease
    assert len(state["tumor"]["cells"])>0
    assert all(key in state["tumor"]["cells"][0] for key in ("st","cycle","generation","drg","o2","metabolic"))
    checkpoint=e.create_checkpoint()
    e.step_tumor(2)
    expected=e.tumor_module.get_snapshot()
    e.restore_checkpoint(checkpoint["checkpoint_id"])
    e.step_tumor(2)
    assert e.tumor_module.get_snapshot()==expected
    assert e.clocks.biological_seconds==3600


def test_unknown_disease_activity_is_not_silently_assigned(session):
    e=session.engine
    e.set_active_disease("AMR_BACTERIAL")
    with pytest.raises(ValueError,match="No disease-specific"):
        e.introduce_compound_to_tumor("CCO")


def test_reset_defaults_restores_start_and_preserves_saves(session):
    client=TestClient(app)
    saved=session.engine.save_current_experiment("Keep me")
    session.engine.update_environment({"ph":5.0})
    session.engine.step_universe(2)
    response=client.post("/api/v1/world/reset-defaults")
    assert response.status_code==200
    state=response.json()["data"]
    assert state["clocks"]["evolutionary_step"]==0
    assert state["environment"]["ph"]==7.4
    assert state["total_entity_count"]==106
    assert not state["playback"]["is_running"]
    assert session.engine.store.load_experiment(saved["experiment_id"])["title"]=="Keep me"


def test_background_search_real_results_registered_persisted_and_compared(session):
    async def run():
        wb=EvolutionWorkbench(session)
        result=await wb.start({"generations":1,"population_size":6,"candidate_count":5,"top_k_cellular":1})
        assert result["status"]=="running"
        assert session.engine.discovery_results=={}
        await asyncio.wait_for(wb.task,timeout=120)
        assert wb.job["status"]=="complete",wb.job
        results=session.engine.discovery_results
        assert results["total_molecules_evaluated"]>4
        assert len(results["top_candidates"])==5
        for candidate in results["top_candidates"]:
            assert not candidate["is_seed"]
            assert candidate["compound_id"] in session.engine.compounds
            assert candidate["drug_comparison"]["matches"]
        saved=session.engine.save_current_experiment("Search")
        session.engine.reset_world("sandbox_empty")
        session.engine.load_saved_experiment(saved["experiment_id"])
        assert len(session.engine.discovery_results["top_candidates"])==5
    asyncio.run(run())


def test_editor_catalog_operations_and_invalid_edit_reported(session):
    client=TestClient(app)
    catalog=client.get("/api/v1/editor/operations").json()
    assert len(catalog["substituents"])>=10
    for param in ("methyl","trifluoromethyl","fluoro"):
        result=client.post("/api/v1/editor/apply",json={"smiles":"CC(=O)Oc1ccccc1C(=O)O","operation":"add_group","param":param}).json()["data"]
        assert result["valid"],result
        assert result["edited_smiles"] != result["original_smiles"]
    invalid=client.post("/api/v1/editor/apply",json={"smiles":"bad smiles","operation":"add_group","param":"methyl"})
    assert invalid.status_code==400 or not invalid.json()["data"].get("valid",True)


@pytest.mark.parametrize("disease",["PDAC","NSCLC_KRAS","AMR_BACTERIAL","METABOLIC_MASLD"])
def test_gpu_cpu_field_parity(disease,monkeypatch):
    """Any vendor's accelerator must reproduce the CPU reference fields.

    float64 backends (CUDA, ROCm, XPU) are held to a tight tolerance. DirectML and
    MPS have no float64, so they are checked at a tolerance appropriate to float32.
    """
    monkeypatch.setenv("EVOCURE_COMPUTE","auto")
    device=compute_device()
    if device is None:
        pytest.skip("No GPU accelerator is available on this host")
    monkeypatch.setenv("EVOCURE_COMPUTE","cpu")
    cpu=DiseaseRegistry().create(disease,seed=7,nx=36,ny=28)
    gpu=copy.deepcopy(cpu)
    cpu._diffuse_and_react_substrates(900)
    monkeypatch.setenv("EVOCURE_COMPUTE","auto")
    gpu._diffuse_and_react_substrates(900)
    atol=1e-12 if device.precision=="float64" else 2e-5
    assert np.allclose(cpu.grid_oxygen,gpu.grid_oxygen,rtol=0,atol=atol)
    assert np.allclose(cpu.grid_nutrient,gpu.grid_nutrient,rtol=0,atol=atol)


def test_compute_status_reports_a_backend():
    report=compute_status()
    assert report["preference"] in ("auto","cpu","cuda","rocm","directml","xpu","mps")
    assert isinstance(report["detected_backends"],list)
    assert report["precision"] in ("float32","float64")
    if report["gpu_available"]:
        assert report["gpu_backend"] in ("cuda","rocm","directml","xpu","mps")
        assert report["gpu_vendor"]
    else:
        assert report["numerical_backend"]=="NumPy CPU float64"


def test_combination_has_no_invented_synergy_and_respects_request(session):
    client = TestClient(app)
    response = client.post("/api/v1/tumor/combination", json={
        "compound_a": "Gemcitabine", "compound_b": "Paclitaxel",
        "dose_a_um": 0, "dose_b_um": 0, "run_4_arm_trial": False,
        "injection_region": "uniform", "hours": 1,
    })
    assert response.status_code == 200, response.text
    result = response.json()["data"]
    assert result["trial_4_arm"] is None
    assert result["synergy_profile"]["synergy_factor"] == 1.0
    assert result["bliss_excess_pct"] == 0
    assert result["loewe_ci"] is None
    assert np.max(result["synergy_profile"]["matrix_2d"]["bliss_excess_matrix"]) == 0
    assert session.engine.tumor_module.applied_concentration_um == 0


def test_coevolution_changes_real_structures_and_advances_actual_clock(session):
    from rdkit.Chem import Descriptors
    from rdkit import Chem
    previous = set(session.engine.compounds)
    result = session.engine.run_coevolution_experiment(n_cycles=2, hours_per_cycle=1)
    new_records = [record for key, record in session.engine.compounds.items() if key not in previous]
    assert new_records
    for record in new_records:
        assert record.mol_weight == pytest.approx(Descriptors.MolWt(Chem.MolFromSmiles(record.canonical_smiles)), abs=0.01)
    assert session.engine.clocks.biological_seconds == 7200
    assert result["model_info"]["status"] == "exploratory coevolution"


@pytest.mark.parametrize("disease,reference", [("NSCLC_KRAS", "Sotorasib"), ("AMR_BACTERIAL", "Meropenem"), ("METABOLIC_MASLD", "Resmetirom")])
def test_other_disease_search_does_not_fabricate_activity(disease, reference, monkeypatch):
    from simulation.discovery.pipeline import CandidateDiscoveryPipeline
    monkeypatch.setenv("EVOCURE_DISABLE_LAYA_NEURAL", "1")
    monkeypatch.setenv("EVOCURE_COMPUTE", "cpu")
    result = CandidateDiscoveryPipeline(disease_id=disease).run(seed_queries=[reference], generations=1, population_size=4, top_k_cellular=1)
    assert result["controlled_comparisons"] == []
    assert "Chemistry-only" in result["activity_scope"]
    assert all(candidate["pic50"] is None for candidate in result["evaluated_candidates"])
    assert result["objective_weights"]["predicted_pdac_activity"] == 0


def test_local_gemma_planning_uses_constrained_json_payload(monkeypatch):
    import io
    import urllib.request
    from ai.local_llm.gemma_assistant import LocalGemmaAssistant
    from simulation.core.contracts import InterfaceMode
    seen = {}
    def answer(request, **kwargs):
        seen.update(json.loads(request.data))
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": '{"priorities":["safety"]}'}}]}).encode())
    monkeypatch.setattr(urllib.request, "urlopen", answer)
    result = LocalGemmaAssistant()._query_llama_server("Find 5 candidates", {"task": "bounded_evolution_planning"}, InterfaceMode.LAB)
    assert json.loads(result)["priorities"] == ["safety"]
    assert seen["response_format"] == {"type": "json_object"}
    assert seen["chat_template_kwargs"]["enable_thinking"] is False


def test_health_endpoint_works_without_any_gpu(monkeypatch):
    """A machine with no GPU must not break the API.

    The telemetry contract returns gpu=None rather than omitting the key, so a
    .get("gpu", {}) default does not protect callers. That exact mistake shipped
    and only surfaced on a GPU-less CI runner, because every development machine
    here has a card.
    """
    import ai.orchestration.registry as registry_module
    from simulation.core import hardware

    monkeypatch.setattr(hardware, "detect_gpus", lambda: [])
    monkeypatch.setattr(hardware, "primary_gpu", lambda: None)
    monkeypatch.setenv("EVOCURE_COMPUTE", "cpu")

    telemetry = registry_module.get_ai_registry().get_hardware_telemetry()
    assert telemetry["gpu"] is None
    assert telemetry["gpus"] == []

    client = TestClient(app)
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    payload = response.json()
    accel = payload.get("ai_status", {}).get("hardware_accel")
    assert accel, "health endpoint must report an accelerator string even with no GPU"


def test_compute_doctor_runs_without_a_gpu(monkeypatch):
    monkeypatch.setenv("EVOCURE_COMPUTE", "cpu")
    report = compute_status()
    assert report["gpu_available"] is False
    assert report["gpu_name"] is None
    assert report["numerical_backend"] == "NumPy CPU float64"
    assert report["precision"] == "float64"
