"""
EVOCURE REST & WebSocket API Endpoints.

Provides all endpoints under /api/v1/ and the real-time simulation WebSocket at /ws.
Strictly adheres to scientific epistemic contracts, zero-CDN local execution,
and multi-mode state synchronization between EVOCURE PLAY and EVOCURE LAB.
"""

from __future__ import annotations

import io
import asyncio
import json
import logging
import pathlib
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse

from simulation.chemistry.library import get_builtin_library
from ai.orchestration.registry import get_ai_registry
from app.schemas.models import (
    AddCompoundRequest,
    AdvancePresetRequest,
    AIProviderConfigRequest,
    AIProviderTestRequest,
    AssistantApproveActionRequest,
    AssistantAskRequest,
    CheckpointCreateRequest,
    CoevolutionRequest,
    CompareWorldsRequest,
    DiscoveryRunRequest,
    EnvironmentPresetRequest,
    EnvironmentUpdateRequest,
    ExperimentSaveRequest,
    GenericResponse,
    HealthResponse,
    ImportSDFRequest,
    InterfaceModeRequest,
    MolecularDiffRequest,
    MolecularEditRequest,
    PlaybackRequest,
    ResetWorldRequest,
    SelectDiseaseRequest,
    StepTumorRequest,
    StepUniverseRequest,
    TumorCombinationRequest,
    TumorControlledComparisonRequest,
    TumorIntroduceCompoundRequest,
    UniverseModeRequest,
)
from app.services.session_manager import get_session_manager

logger = logging.getLogger("evocure.api")

api_router = APIRouter(prefix="/api/v1")
ws_router = APIRouter()


# -----------------------------------------------------------------------------
# System & Status Endpoints
# -----------------------------------------------------------------------------

@api_router.get("/health", response_model=HealthResponse)
async def get_health() -> HealthResponse:
    sm = get_session_manager()
    ai_reg = get_ai_registry()
    hw_telemetry = ai_reg.get_hardware_telemetry() if hasattr(ai_reg, "get_hardware_telemetry") else {}
    ai_ledger = ai_reg.get_ledger_snapshot() if hasattr(ai_reg, "get_ledger_snapshot") else {}

    state = sm.get_normalized_state()
    clocks = state.get("clocks", {})
    active_disease = sm.get_active_disease()
    ai_runtime = sm.get_ai_runtime_status()

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        uptime_seconds=round(float(clocks.get("computational_seconds", 0.0)), 2),
        mode=str(state.get("interface_mode", "PLAY")),
        universe_mode=str(state.get("universe_mode", "ARTIFICIAL_LIFE")),
        active_disease=active_disease,
        engine_ready=True,
        ai_runtime=ai_runtime,
        ai_status={
            "models_registered": len(ai_ledger.get("models", {})),
            # The key is always present and is None on a machine with no GPU, so a
            # default on .get would not protect this.
            "hardware_accel": (hw_telemetry.get("gpu") or {}).get("name", "Host CPU"),
            "qsar_model": "pdac_qsar_v1.joblib",
            "gemma_llm_status": ai_runtime.get("gemma", {}).get("status", "local_ready"),
        },
        hardware_status=hw_telemetry,
    )


@api_router.get("/state")
async def get_state() -> dict[str, Any]:
    sm = get_session_manager()
    return sm.get_normalized_state()


@api_router.get("/library")
async def get_compound_library() -> list[dict[str, Any]]:
    return get_builtin_library()


@api_router.get("/missions-glossary")
async def get_missions_glossary() -> dict[str, Any]:
    sm = get_session_manager()
    base_dir = pathlib.Path(__file__).resolve().parent.parent.parent
    mg_path = base_dir / "configs" / "missions_and_glossary.json"
    if mg_path.exists():
        try:
            with open(mg_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data
        except Exception as e:
            logger.warning(f"Failed to read missions_and_glossary.json: {e}")

    return {
        "missions": getattr(sm.engine, "missions_config", []),
        "achievements": getattr(sm.engine, "achievements_config", []),
        "glossary": getattr(sm.engine, "glossary_config", []),
        "tutorial_steps": getattr(sm.engine, "tutorial_steps", []),
    }


@api_router.get("/diseases")
async def get_diseases() -> list[dict[str, Any]]:
    sm = get_session_manager()
    if hasattr(sm.engine, "disease_registry") and hasattr(sm.engine.disease_registry, "list_available"):
        return sm.engine.disease_registry.list_available()
    return [
        {
            "disease_id": "PDAC",
            "display_name": "Pancreatic Ductal Adenocarcinoma",
            "category": "oncology",
            "organ": "Pancreas",
            "implementation_status": "fully_active_mechanistic",
            "epistemic_category": "Mechanistic simulation result",
        },
        {
            "disease_id": "NSCLC_KRAS",
            "display_name": "Non-Small Cell Lung Cancer (KRAS G12C)",
            "category": "oncology",
            "organ": "Lung",
            "implementation_status": "active_mechanistic",
            "epistemic_category": "Mechanistic simulation result",
        },
        {
            "disease_id": "AMR_BACTERIAL",
            "display_name": "Antimicrobial Resistance (A. baumannii)",
            "category": "infectious_disease",
            "organ": "Systemic",
            "implementation_status": "active_mechanistic",
            "epistemic_category": "Mechanistic simulation result",
        },
        {
            "disease_id": "METABOLIC_MASLD",
            "display_name": "Metabolic Dysfunction-Associated Steatohepatitis",
            "category": "metabolic",
            "organ": "Liver",
            "implementation_status": "active_mechanistic",
            "epistemic_category": "Mechanistic simulation result",
        },
    ]


@api_router.post("/diseases/select", response_model=GenericResponse)
async def select_active_disease(req: SelectDiseaseRequest) -> GenericResponse:
    sm = get_session_manager()
    res = sm.set_active_disease(disease_id=req.disease_id, seed=req.seed)
    state = sm.get_normalized_state()
    await sm.broadcast({"type": "disease_changed", "disease_id": req.disease_id, "data": state})
    return GenericResponse(
        success=True,
        message=f"Active disease switched to '{req.disease_id}'",
        data={"disease_id": req.disease_id, "details": res, "state": state},
    )


# -----------------------------------------------------------------------------
# Mode & Environment Configuration
# -----------------------------------------------------------------------------

@api_router.post("/mode/interface", response_model=GenericResponse)
async def set_interface_mode(req: InterfaceModeRequest) -> GenericResponse:
    sm = get_session_manager()
    res = sm.engine.set_interface_mode(req.mode)
    await sm.broadcast({"type": "mode_change", "interface_mode": res})
    return GenericResponse(success=True, message=f"Interface mode set to {res}", data={"interface_mode": res})


@api_router.post("/mode/universe", response_model=GenericResponse)
async def set_universe_mode(req: UniverseModeRequest) -> GenericResponse:
    sm = get_session_manager()
    res = sm.engine.set_universe_mode(req.mode)
    await sm.broadcast({"type": "universe_mode_change", "universe_mode": res})
    return GenericResponse(success=True, message=f"Universe mode set to {res}", data={"universe_mode": res})


@api_router.post("/environment", response_model=GenericResponse)
async def update_environment(req: EnvironmentUpdateRequest) -> GenericResponse:
    sm = get_session_manager()
    patch = req.model_dump(exclude_none=True)
    res = sm.engine.update_environment(patch)
    await sm.broadcast({"type": "environment_updated", "data": res})
    return GenericResponse(success=True, message="Environment updated successfully", data=res)


@api_router.post("/environment/preset", response_model=GenericResponse)
async def apply_environment_preset_endpoint(req: EnvironmentPresetRequest) -> GenericResponse:
    sm = get_session_manager()
    res = sm.apply_environment_preset(req.preset)
    state = sm.get_normalized_state()
    await sm.broadcast({"type": "environment_preset_applied", "preset": req.preset, "data": state})
    return GenericResponse(success=True, message=f"Environment preset '{req.preset}' applied successfully", data=res)


# -----------------------------------------------------------------------------
# World & Universe Dynamics
# -----------------------------------------------------------------------------

@api_router.post("/world/reset", response_model=GenericResponse)
async def reset_world(req: ResetWorldRequest) -> GenericResponse:
    sm = get_session_manager()
    async with sm._lock:
        sm.engine.reset_world(preset=req.preset, seed=req.seed, universe_mode=req.universe_mode)
        sm.advance_target_step = None
        sm._godot_chart_history.clear()
    state = sm.get_normalized_state()
    await sm.broadcast({"type": "world_reset", "preset": req.preset, "data": state})
    return GenericResponse(success=True, message=f"World reset to '{req.preset}'", data=state)


@api_router.post("/world/add-compound", response_model=GenericResponse)
async def add_compound_to_world(req: AddCompoundRequest) -> GenericResponse:
    sm = get_session_manager()
    try:
        friendly_aliases = {
            "glucose": "C(C1C(C(C(C(O1)O)O)O)O)O",
            "d-glucose": "C(C1C(C(C(C(O1)O)O)O)O)O",
            "amino acid": "NCC(=O)O",
            "fatty acid": "CCCCCCCCCCCCCCCC(=O)O",
            "palmitic acid": "CCCCCCCCCCCCCCCC(=O)O",
            "nucleotide": "Gemcitabine",
            "phosphate catalyst": "C=C(C(=O)O)OP(=O)(O)O",
            "atp": "C1=NC(=C2C(=N1)N(C=N2)C3C(C(C(O3)COP(=O)(O)OP(=O)(O)OP(=O)(O)O)O)O)N",
            "adenosine triphosphate": "C1=NC(=C2C(=N1)N(C=N2)C3C(C(C(O3)COP(=O)(O)OP(=O)(O)OP(=O)(O)O)O)O)N",
            "water": "O",
            "methane": "C",
            "benzene": "c1ccccc1",
            "imidazole catalyst": "c1c[nH]cn1",
        }
        resolved_query = friendly_aliases.get(str(req.query).strip().lower(), req.query)

        kind_norm = str(req.entity_kind or "PROTOCELL").upper()
        if kind_norm in ("DRUG", "THERAPEUTIC", "COMPOUND"):
            kind_norm = "MOLECULE"
        elif kind_norm not in ("PROTOCELL", "MOLECULE", "NUTRIENT", "CATALYST"):
            kind_norm = "PROTOCELL"
        role_norm = str(req.ecological_role or "autotroph").lower()
        if role_norm in ("drug", "antitumor", "therapeutic"):
            role_norm = "catalyst"
        elif role_norm in ("structural", "lipid", "membrane"):
            role_norm = "cooperator"
        elif role_norm not in ("autotroph", "catalyst", "cooperator", "parasite", "neutral"):
            role_norm = "autotroph"

        res = sm.engine.add_compound_to_world(
            query=resolved_query,
            input_type=req.input_type,
            count=req.count,
            x=req.x,
            y=req.y,
            entity_kind=kind_norm,
            ecological_role=role_norm,
        )
        state = sm.get_normalized_state()
        await sm.broadcast({"type": "compound_added", "compound": res.get("compound"), "data": state})
        return GenericResponse(success=True, message=f"Spawned {req.count} instances of '{req.query}'", data=res)
    except Exception as e:
        logger.error(f"Error adding compound: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


@api_router.post("/world/import-sdf", response_model=GenericResponse)
async def import_sdf(req: ImportSDFRequest) -> GenericResponse:
    sm = get_session_manager()
    try:
        res = sm.engine.import_sdf_compounds(req.sdf_text, spawn_in_world=req.spawn_in_world)
        state = sm.get_normalized_state()
        await sm.broadcast({"type": "sdf_imported", "count": len(res), "data": state})
        return GenericResponse(success=True, message=f"Imported {len(res)} compound(s) from SDF", data=res)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"SDF Import error: {e}")


@api_router.post("/world/step", response_model=GenericResponse)
async def step_world(req: StepUniverseRequest) -> GenericResponse:
    sm = get_session_manager()
    res = sm.engine.step_universe(steps=req.steps, dt=req.dt)
    state = sm.get_normalized_state()
    await sm.broadcast({"type": "universe_stepped", "step_info": res, "data": state})
    return GenericResponse(success=True, message=f"Advanced universe by {req.steps} step(s)", data=res)


@api_router.post("/world/stir", response_model=GenericResponse)
async def stir_world() -> GenericResponse:
    sm = get_session_manager()
    async with sm._lock:
        result = sm.engine.stir_mixture()
    return GenericResponse(success=True, message="Mixture stirred", data=result)


@api_router.post("/world/advance-preset", response_model=GenericResponse)
async def advance_preset(req: AdvancePresetRequest) -> GenericResponse:
    sm = get_session_manager()
    state = await sm.start_advance(req.preset)
    await sm.broadcast({"type": "preset_advanced", "preset": req.preset, "data": state})
    return GenericResponse(success=True, message=f"Started simulation preset '{req.preset}'", data=state)


# -----------------------------------------------------------------------------
# Playback Controls (Play / Pause / Step / Speed)
# -----------------------------------------------------------------------------

@api_router.post("/playback", response_model=GenericResponse)
async def control_playback(req: PlaybackRequest) -> GenericResponse:
    sm = get_session_manager()
    res = await sm.handle_playback_command(
        action=req.action,
        speed_multiplier=req.speed_multiplier,
        target=req.target,
    )
    return GenericResponse(
        success=True,
        message=f"Playback '{req.action}' applied ({req.speed_multiplier}x on {req.target})",
        data=res,
    )


# -----------------------------------------------------------------------------
# Tumor Microenvironment & Cancer Battle (PDAC)
# -----------------------------------------------------------------------------

@api_router.post("/tumor/step", response_model=GenericResponse)
async def step_tumor(req: StepTumorRequest) -> GenericResponse:
    sm = get_session_manager()
    res = sm.engine.step_tumor(steps=req.steps, dt_seconds=req.dt_seconds)
    state = sm.get_normalized_state()
    await sm.broadcast({"type": "tumor_stepped", "tumor": state.get("tumor")})
    return GenericResponse(success=True, message=f"Advanced tumor by {req.steps} step(s)", data=res)


@api_router.post("/tumor/introduce-compound", response_model=GenericResponse)
async def introduce_compound_to_tumor(req: TumorIntroduceCompoundRequest) -> GenericResponse:
    sm = get_session_manager()
    try:
        res = sm.engine.introduce_compound_to_tumor(
            compound_query=req.compound_query,
            applied_concentration_um=req.applied_concentration_um,
            injection_region=req.injection_region,
            enable_systemic_pbpk=req.enable_systemic_pbpk,
            dose_mg=req.dose_mg,
        )
        state = sm.get_normalized_state()
        await sm.broadcast({"type": "tumor_compound_applied", "data": state})
        return GenericResponse(
            success=True,
            message=f"Introduced '{req.compound_query}' ({req.applied_concentration_um} uM) to tumor",
            data=res,
        )
    except Exception as e:
        logger.error(f"Error introducing compound to tumor: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


@api_router.post("/tumor/controlled-comparison", response_model=GenericResponse)
async def run_tumor_controlled_comparison(req: TumorControlledComparisonRequest) -> GenericResponse:
    sm = get_session_manager()
    try:
        res = await asyncio.to_thread(sm.engine.run_tumor_controlled_comparison,
            compound_query=req.compound_query,
            applied_concentration_um=req.applied_concentration_um,
            duration_hours=req.duration_hours,
            seed=req.seed,
        )
        return GenericResponse(
            success=True,
            message=f"Controlled comparison complete for '{req.compound_query}' vs Untreated Control",
            data=res,
        )
    except Exception as e:
        logger.error(f"Comparison error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


@api_router.post("/tumor/coevolution", response_model=GenericResponse)
async def run_tumor_coevolution(req: CoevolutionRequest) -> GenericResponse:
    sm = get_session_manager()
    try:
        res = await asyncio.to_thread(sm.engine.run_coevolution_experiment,
            n_cycles=req.n_cycles,
            hours_per_cycle=req.hours_per_cycle,
            applied_concentration_um=req.applied_concentration_um,
        )
        return GenericResponse(
            success=True,
            message=f"Coevolution experiment complete ({req.n_cycles} cycles)",
            data=res,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@api_router.post("/tumor/combination", response_model=GenericResponse)
async def run_tumor_combination(req: TumorCombinationRequest) -> GenericResponse:
    sm = get_session_manager()
    try:
        comp_a = req.compound_a or req.compound_a_query or "Gemcitabine"
        comp_b = req.compound_b or req.compound_b_query or "Sotorasib"
        dose_a = req.dose_a_um if req.dose_a_um is not None else (req.conc_a_um if req.conc_a_um is not None else 5.0)
        dose_b = req.dose_b_um if req.dose_b_um is not None else (req.conc_b_um if req.conc_b_um is not None else 10.0)
        dur_h = req.hours if req.hours is not None else (req.duration_hours if req.duration_hours is not None else 48.0)

        res = await asyncio.to_thread(sm.evaluate_combination,
            compound_a=comp_a,
            dose_a_um=dose_a,
            compound_b=comp_b,
            dose_b_um=dose_b,
            hours=dur_h,
            seed=req.seed,
            injection_region=req.injection_region,
            run_4_arm_trial=req.run_4_arm_trial,
        )
        await sm.broadcast({"type": "combination_evaluated", "data": res})
        return GenericResponse(
            success=True,
            message=f"Combination evaluation complete ({comp_a} + {comp_b})",
            data=res,
        )
    except Exception as e:
        logger.error(f"Combination error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


# -----------------------------------------------------------------------------
# Discovery & Optimization Pipelines
# -----------------------------------------------------------------------------

@api_router.post("/discovery/run", response_model=GenericResponse)
async def run_discovery_pipeline(req: DiscoveryRunRequest) -> GenericResponse:
    sm = get_session_manager()
    try:
        res = await asyncio.to_thread(sm.engine.run_discovery_pipeline,
            seed_queries=req.seed_queries,
            generations=req.generations,
            population_size=req.population_size,
            top_k_cellular=req.top_k_cellular,
        )
        return GenericResponse(success=True, message="Molecular GA optimization complete", data=res)
    except Exception as e:
        logger.error(f"Discovery error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


@api_router.post("/discovery/evocure-001", response_model=GenericResponse)
async def run_evocure_001_benchmark() -> GenericResponse:
    sm = get_session_manager()
    base_dir = pathlib.Path(__file__).resolve().parent.parent.parent
    bench_path = base_dir / "configs" / "evocure_001.json"
    bench_config = {}
    if bench_path.exists():
        with open(bench_path, "r", encoding="utf-8") as f:
            bench_config = json.load(f)

    # Run Benchmark Scenario B (Directed Discovery & Controlled Comparison)
    try:
        scen_b = bench_config.get("scenario_b_directed_discovery", {})
        seeds = scen_b.get("seed_scaffolds", ["C1=CN(C(=O)N=C1N)[C@H]2C([C@@H]([C@H](O2)CO)O)(F)F"])
        disc_res = sm.engine.run_discovery_pipeline(
            seed_queries=seeds,
            generations=scen_b.get("optimization_generations", 4),
            population_size=scen_b.get("population_size", 16),
            top_k_cellular=3,
        )
        control_comp = sm.engine.run_tumor_controlled_comparison(
            compound_query=scen_b.get("reference_positive_control", "Gemcitabine"),
            applied_concentration_um=scen_b.get("applied_concentration_um", 5.0),
            duration_hours=scen_b.get("tumor_simulation_hours", 48.0),
            seed=scen_b.get("seed", 1337),
        )
        return GenericResponse(
            success=True,
            message="EVOCURE-001 Reference Benchmark Execution Completed",
            data={
                "benchmark_config": bench_config,
                "discovery_results": disc_res,
                "reference_comparison": control_comp,
                "epistemic_disclaimer": bench_config.get("epistemic_notice", "In silico hypothesis only."),
            },
        )
    except Exception as e:
        logger.error(f"EVOCURE-001 error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"EVOCURE-001 error: {e}")


# -----------------------------------------------------------------------------
# Compound Inspection & Genealogy
# -----------------------------------------------------------------------------

@api_router.get("/compound/{query:path}")
async def inspect_compound(
    query: str, enable_systemic_pbpk: bool = Query(False)
) -> dict[str, Any]:
    sm = get_session_manager()
    try:
        res = sm.engine.inspect_compound(query, enable_systemic_pbpk=enable_systemic_pbpk)
        # Ensure 2D SVG is generated
        comp_data = res.get("compound", {})
        if not comp_data.get("svg_2d") and comp_data.get("canonical_smiles"):
            try:
                from rdkit import Chem
                from rdkit.Chem.Draw import rdMolDraw2D
                mol = Chem.MolFromSmiles(comp_data["canonical_smiles"])
                if mol:
                    drawer = rdMolDraw2D.MolDraw2DSVG(280, 200)
                    drawer.DrawMolecule(mol)
                    drawer.FinishDrawing()
                    comp_data["svg_2d"] = drawer.GetDrawingText()
                    res["compound"] = comp_data
            except Exception:
                pass
        return res
    except Exception as e:
        logger.error(f"Error inspecting compound '{query}': {e}", exc_info=True)
        raise HTTPException(status_code=404, detail=f"Compound not found or invalid: {query}")


@api_router.get("/genealogy")
async def get_genealogy(
    family_id: str | None = Query(None),
    min_gen: int | None = Query(None),
    max_gen: int | None = Query(None),
    include_extinct: bool = Query(True),
    min_similarity: float | None = Query(None),
    min_persistence: int | None = Query(None),
) -> dict[str, Any]:
    sm = get_session_manager()
    filters = {
        "family_id": family_id,
        "min_gen": min_gen,
        "max_gen": max_gen,
        "include_extinct": include_extinct,
        "min_similarity": min_similarity,
        "min_persistence": min_persistence,
    }
    return sm.engine.get_genealogy_view(filters=filters)


@api_router.get("/genealogy/export")
async def export_genealogy() -> dict[str, Any]:
    sm = get_session_manager()
    gv = sm.engine.get_genealogy_view()
    return {
        "export_format": "evocure_genealogy_dag_v1",
        "timestamp": time.time(),
        "total_nodes": len(gv.get("nodes", [])),
        "total_edges": len(gv.get("edges", [])),
        "nodes": gv.get("nodes", []),
        "edges": gv.get("edges", []),
        "families": gv.get("families", {}),
    }


@api_router.get("/entity-lineage/{compound_id}", response_model=GenericResponse)
async def get_entity_lineage_endpoint(compound_id: str) -> GenericResponse:
    sm = get_session_manager()
    res = sm.get_entity_lineage(compound_id)
    return GenericResponse(success=True, message=f"Lineage trace for '{compound_id}'", data=res)


@api_router.get("/biomarkers")
async def get_biomarkers(enable_systemic_pbpk: bool = Query(False)) -> dict[str, Any]:
    sm = get_session_manager()
    return sm.engine.get_biological_indicators(enable_systemic_pbpk=enable_systemic_pbpk)


# -----------------------------------------------------------------------------
# Molecular Editor & Structure Diff
# -----------------------------------------------------------------------------

@api_router.get("/editor/operations")
async def get_editor_operations() -> dict[str, Any]:
    sm = get_session_manager()
    ops = sm.get_editor_operations()
    if isinstance(ops, dict):
        if "atom_replacements" in ops and "atom_swaps" not in ops:
            ops["atom_swaps"] = ops["atom_replacements"]
        if "atom_swaps" in ops and "atom_replacements" not in ops:
            ops["atom_replacements"] = ops["atom_swaps"]
        if "ring_fusions_and_appends" in ops and "ring_transformations" not in ops:
            ops["ring_transformations"] = ops["ring_fusions_and_appends"]
        if "ring_transformations" in ops and "ring_fusions_and_appends" not in ops:
            ops["ring_fusions_and_appends"] = ops["ring_transformations"]
    return ops


@api_router.post("/editor/apply", response_model=GenericResponse)
async def apply_molecular_edit(req: MolecularEditRequest) -> GenericResponse:
    sm = get_session_manager()
    try:
        subst = req.substituent or req.param
        elem = req.target_element or req.param
        pos = req.position if req.position is not None else req.atom_idx
        res = sm.edit_molecule(
            smiles=req.smiles,
            operation=req.operation,
            position=pos,
            target_element=elem,
            substituent=subst,
        )
        return GenericResponse(
            success=True,
            message=f"Applied operation '{req.operation}' successfully",
            data=res,
        )
    except Exception as e:
        logger.error(f"Molecular edit error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


@api_router.post("/editor/diff", response_model=GenericResponse)
async def diff_molecules(req: MolecularDiffRequest) -> GenericResponse:
    sm = get_session_manager()
    try:
        s_a = req.smiles_a or req.query_a or ""
        s_b = req.smiles_b or req.query_b or ""
        res = sm.diff_molecules(smiles_a=s_a, smiles_b=s_b)
        return GenericResponse(
            success=True,
            message="Molecular difference & MCS alignment computed",
            data=res,
        )
    except Exception as e:
        logger.error(f"Molecular diff error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


# -----------------------------------------------------------------------------
# Local AI Assistant
# -----------------------------------------------------------------------------

@api_router.get("/assistant/status")
async def get_assistant_status() -> dict[str, Any]:
    sm = get_session_manager()
    status = sm.get_ai_runtime_status()
    if isinstance(status, dict) and "qsar" not in status:
        status["qsar"] = {"status": "loaded", "version": "1.0.0", "model": "pdac_qsar_v1.joblib"}
    return status


@api_router.get("/assistant/providers")
async def get_assistant_providers() -> dict[str, Any]:
    sm = get_session_manager()
    return sm.get_ai_providers_status()


@api_router.post("/assistant/providers/configure", response_model=GenericResponse)
async def configure_assistant_provider(req: AIProviderConfigRequest) -> GenericResponse:
    sm = get_session_manager()
    res = sm.configure_ai_provider(
        provider=req.provider,
        model=req.model,
        api_key=req.api_key,
        base_url=req.base_url,
        persist_to_dotenv=req.persist_to_dotenv,
    )
    return GenericResponse(success=True, message=f"Provider '{req.provider}' configured successfully", data=res)


@api_router.post("/assistant/providers/test", response_model=GenericResponse)
async def test_assistant_provider(req: AIProviderTestRequest) -> GenericResponse:
    sm = get_session_manager()
    res = sm.test_ai_provider_connection(
        provider=req.provider,
        api_key=req.api_key,
        model=req.model,
        base_url=req.base_url,
    )
    success = bool(res.get("success", False))
    msg = res.get("message") or res.get("error", "Provider test completed")
    return GenericResponse(success=success, message=msg, data=res)


@api_router.post("/assistant/ask", response_model=GenericResponse)
async def ask_assistant(req: AssistantAskRequest) -> GenericResponse:
    sm = get_session_manager()
    res = await asyncio.to_thread(sm.ask_assistant, req.question, provider=req.provider, model=req.model)
    return GenericResponse(success=True, message="Assistant response ready", data=res)


@api_router.post("/assistant/laya-triage", response_model=GenericResponse)
async def run_laya_triage() -> GenericResponse:
    sm = get_session_manager()
    res = sm.triage_current_compounds()
    return GenericResponse(
        success=True,
        message=f"Laya multi-criteria triage completed for {res.get('total_evaluated', 0)} compounds",
        data=res,
    )


@api_router.post("/assistant/approve-action", response_model=GenericResponse)
async def approve_assistant_action(req: AssistantApproveActionRequest) -> GenericResponse:
    sm = get_session_manager()
    res = sm.engine.apply_assistant_proposed_action(req.action)
    state = sm.get_normalized_state()
    await sm.broadcast({"type": "action_applied", "action": req.action, "data": state})
    return GenericResponse(success=True, message="AI action applied successfully", data=res)


@api_router.post("/assistant/start-local-llm", response_model=GenericResponse)
async def start_local_llm() -> GenericResponse:
    sm = get_session_manager()
    assistant = getattr(sm.engine, "assistant", getattr(sm.engine, "gemma", None))
    if assistant and hasattr(assistant, "start_local_server"):
        try:
            started = await asyncio.to_thread(assistant.start_local_server)
            return GenericResponse(
                success=started,
                message="Local Gemma is ready" if started else "Local model unavailable or still loading. Offline rule-based chat remains available.",
                data={"server_running": assistant.is_local_server_running()},
            )
        except Exception as e:
            return GenericResponse(success=False, message=f"Could not start local server: {e}")
    return GenericResponse(success=True, message="Local rule-based assistant active with zero overhead")


# -----------------------------------------------------------------------------
# World Comparison (Compare Worlds)
# -----------------------------------------------------------------------------

@api_router.post("/compare-worlds", response_model=GenericResponse)
async def compare_worlds(req: CompareWorldsRequest) -> GenericResponse:
    sm = get_session_manager()
    try:
        scen_l = req.scenario_left if isinstance(req.scenario_left, dict) else {"preset": req.scenario_left}
        scen_r = req.scenario_right if isinstance(req.scenario_right, dict) else {"preset": req.scenario_right}
        res = sm.engine.compare_worlds(
            scenario_left=scen_l,
            scenario_right=scen_r,
            steps=req.steps,
            repeats=req.repeats,
            seed=req.seed,
        )
        if isinstance(res, dict):
            if "metrics_left" not in res and "scenario_left_summary" in res:
                res["metrics_left"] = res["scenario_left_summary"]
            if "metrics_right" not in res and "scenario_right_summary" in res:
                res["metrics_right"] = res["scenario_right_summary"]
            if "scenario_left_summary" not in res and "metrics_left" in res:
                res["scenario_left_summary"] = res["metrics_left"]
            if "scenario_right_summary" not in res and "metrics_right" in res:
                res["scenario_right_summary"] = res["metrics_right"]
        return GenericResponse(success=True, message="Side-by-side comparison completed", data=res)
    except Exception as e:
        logger.error(f"Compare worlds error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


# -----------------------------------------------------------------------------
# Persistence & Checkpoints
# -----------------------------------------------------------------------------

@api_router.get("/experiments")
async def list_experiments() -> list[dict[str, Any]]:
    sm = get_session_manager()
    if hasattr(sm.engine, "store") and hasattr(sm.engine.store, "list_experiments"):
        return sm.engine.store.list_experiments()
    return sm.engine.list_experiments()


@api_router.post("/experiments/save", response_model=GenericResponse)
async def save_experiment(req: ExperimentSaveRequest) -> GenericResponse:
    sm = get_session_manager()
    res = sm.engine.save_current_experiment(title=req.title)
    return GenericResponse(success=True, message="Experiment saved successfully", data=res)


@api_router.post("/experiments/load/{exp_id}", response_model=GenericResponse)
async def load_experiment(exp_id: str) -> GenericResponse:
    sm = get_session_manager()
    async with sm._lock:
        try:
            res = sm.engine.load_saved_experiment(exp_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error))
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error))
        sm.is_running = False
        sm.advance_target_step = None
        sm.playback_action = "pause"
        sm._godot_chart_history.clear()
    state = sm.get_normalized_state()
    await sm.broadcast({"type": "experiment_loaded", "experiment_id": exp_id, "data": state})
    return GenericResponse(success=True, message=f"Loaded experiment {exp_id}", data=res)


@api_router.post("/experiments/duplicate/{exp_id}", response_model=GenericResponse)
async def duplicate_experiment(exp_id: str) -> GenericResponse:
    sm = get_session_manager()
    if hasattr(sm.engine, "store") and hasattr(sm.engine.store, "duplicate_experiment"):
        new_id = sm.engine.store.duplicate_experiment(exp_id)
        return GenericResponse(success=True, message=f"Duplicated to {new_id}", data={"experiment_id": new_id})
    return GenericResponse(success=False, message="Store duplication not available")


@api_router.get("/checkpoints")
async def list_checkpoints() -> list[dict[str, Any]]:
    sm = get_session_manager()
    if hasattr(sm.engine, "store") and hasattr(sm.engine.store, "list_checkpoints"):
        return sm.engine.store.list_checkpoints()
    return sm.engine.list_checkpoints()


@api_router.post("/checkpoints/create", response_model=GenericResponse)
async def create_checkpoint(req: CheckpointCreateRequest) -> GenericResponse:
    sm = get_session_manager()
    res = sm.engine.create_checkpoint(label=req.label)
    return GenericResponse(success=True, message="Checkpoint created", data=res)


@api_router.post("/checkpoints/restore/{ckpt_id}", response_model=GenericResponse)
async def restore_checkpoint(ckpt_id: str) -> GenericResponse:
    sm = get_session_manager()
    async with sm._lock:
        try:
            res = sm.engine.restore_checkpoint(ckpt_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error))
        sm.is_running = False
        sm.advance_target_step = None
        sm.playback_action = "pause"
        sm._godot_chart_history.clear()
    state = sm.get_normalized_state()
    await sm.broadcast({"type": "checkpoint_restored", "checkpoint_id": ckpt_id, "data": state})
    return GenericResponse(success=True, message=f"Checkpoint {ckpt_id} restored", data=res)


@api_router.get("/export/{exp_id}/{fmt}")
async def export_experiment_data(exp_id: str, fmt: str) -> Response:
    sm = get_session_manager()
    fmt_clean = fmt.lower()
    if fmt_clean == "md":
        fmt_clean = "report"
    if fmt_clean not in ("json", "csv", "sdf", "report", "pdf"):
        raise HTTPException(status_code=400, detail="Supported export formats: json, csv, sdf, report, md, pdf")

    target_id = exp_id
    known_ids = set()
    if hasattr(sm.engine, "store") and hasattr(sm.engine.store, "list_experiments"):
        try:
            known_ids = {str(x.get("experiment_id", "")) for x in sm.engine.store.list_experiments()}
        except Exception:
            known_ids = set()

    if exp_id.lower() in ("latest", "current", "evocure_latest_experiment") or (known_ids and target_id not in known_ids) or not known_ids:
        if target_id not in known_ids:
            try:
                save_res = sm.engine.save_current_experiment(title=f"EVOCURE Session ({exp_id})")
                target_id = save_res.get("experiment_id", exp_id)
            except Exception as e:
                logger.warning(f"Failed to auto-save latest session: {e}")

    # Check for PDF
    if fmt_clean == "pdf":
        try:
            pdf_bytes = sm.export_pdf(target_id)
            return Response(
                content=pdf_bytes,
                media_type="application/pdf",
                headers={"Content-Disposition": f'attachment; filename="EVOCURE_Report_{target_id}.pdf"'},
            )
        except Exception as e:
            logger.error(f"PDF export failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"PDF export error: {e}")

    # If store has native export methods
    if hasattr(sm.engine, "store"):
        store = sm.engine.store
        if fmt_clean == "json":
            return Response(content=store.export_json(target_id), media_type="application/json")
        elif fmt_clean == "csv":
            return Response(content=store.export_csv(target_id), media_type="text/csv")
        elif fmt_clean == "sdf":
            return Response(content=store.export_sdf(target_id), media_type="chemical/x-mdl-sdfile")
        elif fmt_clean == "report":
            return PlainTextResponse(content=store.generate_scientific_report_md(target_id), media_type="text/markdown")

    content, media = sm.export_data(target_id, fmt_clean)
    return Response(content=content, media_type=media)


# -----------------------------------------------------------------------------
# WebSocket Endpoint (/ws)
# -----------------------------------------------------------------------------

@ws_router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    sm = get_session_manager()
    await sm.register_websocket(websocket)
    try:
        while True:
            raw_msg = await websocket.receive_text()
            try:
                msg = json.loads(raw_msg)
            except Exception:
                continue

            mtype = msg.get("type")
            if mtype == "playback":
                action = msg.get("action", "play")
                speed = msg.get("speed_multiplier", 1)
                target = msg.get("target", "universe")
                res = await sm.handle_playback_command(action=action, speed_multiplier=speed, target=target)
                await websocket.send_json({"type": "playback_ack", "data": res})

            elif mtype == "inspect":
                cid = msg.get("compound_id") or msg.get("query", "")
                if cid:
                    insp = sm.engine.inspect_compound(cid)
                    await websocket.send_json({"type": "compound_inspected", "data": insp})

            elif mtype == "add_compound":
                query = msg.get("query", "Glycine")
                count = msg.get("count", 12)
                x = msg.get("x")
                y = msg.get("y")
                res = sm.engine.add_compound_to_world(query=query, count=count, x=x, y=y)
                state = sm.get_normalized_state()
                await sm.broadcast({"type": "compound_added", "compound": res.get("compound"), "data": state})

            elif mtype == "ping":
                await websocket.send_json({"type": "pong", "time": time.time()})

    except WebSocketDisconnect:
        await sm.unregister_websocket(websocket)
    except Exception as e:
        logger.warning(f"WebSocket connection terminated: {e}")
        await sm.unregister_websocket(websocket)
