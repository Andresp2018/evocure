import asyncio
import uuid
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.services.session_manager import get_session_manager
from app.services.evolution_workbench import EvolutionWorkbench
from simulation.discovery.workbench import compare_drugs, plan_guided_search
from simulation.core.acceleration import status as compute_status

router = APIRouter(prefix="/api/v1")


def workbench():
    session = get_session_manager()
    if not hasattr(session, "evolution_workbench"):
        session.evolution_workbench = EvolutionWorkbench(session)
    return session.evolution_workbench


class PlanRequest(BaseModel):
    prompt: str = Field(min_length=3, max_length=2000)


class SearchRequest(BaseModel):
    plan_id: str | None = None
    generations: int = Field(default=2, ge=1, le=6)
    population_size: int = Field(default=12, ge=4, le=40)
    candidate_count: int = Field(default=5, ge=1, le=20)


@router.get("/discovery/compounds")
async def compounds(source: str = Query("universe", pattern="^(universe|shortlist|directed|all)$"), search: str = "", offset: int = Query(0, ge=0), limit: int = Query(60, ge=1, le=100)):
    return workbench().compounds(source, search, offset, limit)


@router.get("/discovery/job")
async def discovery_job():
    return dict(workbench().job)


@router.post("/discovery/plan")
async def discovery_plan(request: PlanRequest):
    wb = workbench()
    plan = await asyncio.to_thread(plan_guided_search, request.prompt, wb.session.engine.assistant)
    plan["plan_id"] = uuid.uuid4().hex
    wb.plans[plan["plan_id"]] = plan
    return plan


@router.post("/discovery/start")
async def start_discovery(request: SearchRequest):
    wb = workbench()
    plan = wb.plans.get(request.plan_id) if request.plan_id else None
    if request.plan_id and not plan:
        raise HTTPException(404, "Search plan not found; create a new plan.")
    settings = {k: plan[k] for k in ("seed_queries", "generations", "population_size", "top_k_cellular", "objective_weights", "candidate_count")} if plan else {
        "generations": request.generations, "population_size": request.population_size,
        "candidate_count": request.candidate_count, "top_k_cellular": 1}
    try:
        return await wb.start(settings, plan)
    except ValueError as error:
        raise HTTPException(409, str(error))


@router.post("/discovery/cancel")
async def cancel_discovery():
    wb = workbench()
    wb.cancel()
    return dict(wb.job)


@router.get("/discovery/drug-comparison")
async def drug_comparison(query: str, limit: int = Query(5, ge=1, le=20)):
    try:
        record = workbench().session.engine._resolve_compound_rec(query)
        return compare_drugs(record, limit)
    except ValueError as error:
        raise HTTPException(400, str(error))


@router.post("/world/reset-defaults")
async def reset_defaults():
    wb = workbench()
    session = wb.session
    wb.cancel()
    async with session._lock:
        engine = session.engine
        engine.env = engine._load_default_environment()
        engine.set_active_disease("PDAC", seed=20261026)
        engine.set_interface_mode("PLAY")
        engine.reset_world("chemical_life", seed=20261026)
        session.is_running = False
        session.playback_action = "pause"
        session.playback_target = "universe"
        session.speed_multiplier = 1
        session.advance_target_step = None
        session._godot_chart_history.clear()
        wb.plans.clear()
        wb.job = {"status": "idle"}
        state = session.get_normalized_state()
    await session.broadcast({"type": "world_reset", "data": state})
    return {"success": True, "data": state, "message": "Default world restored. Saved experiments are retained."}


@router.post("/tumor/reset")
async def reset_tissue():
    session = get_session_manager()
    async with session._lock:
        session.engine.set_active_disease(session.engine.disease_id, seed=session.engine.seed)
        session.engine.tissue_events = []
        session.engine.tissue_model_info = {}
        session.engine.clocks.biological_seconds = 0
        session.is_running = False
        session.playback_action = "pause"
        state = session.get_normalized_state()
    await session.broadcast({"type": "disease_changed", "data": state})
    return {"success": True, "data": state}


@router.get("/compute/status")
async def acceleration_status():
    return compute_status()
