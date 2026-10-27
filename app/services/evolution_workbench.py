"""Responsive discovery jobs; published results are committed on the API thread."""
import asyncio
from collections import Counter
import threading
import time
import uuid

from simulation.discovery.pipeline import CandidateDiscoveryPipeline
from simulation.discovery.workbench import compare_drugs, drug_references


class SearchCancelled(Exception):
    pass


class EvolutionWorkbench:
    def __init__(self, session):
        self.session = session
        self.plans = {}
        self.job = {"status": "idle"}
        self._cancel = threading.Event()
        self.task = None

    def compounds(self, source="universe", search="", offset=0, limit=60):
        engine = self.session.engine
        counts = Counter(e.compound_id for e in engine.entities if e.alive)
        directed = {c["compound_id"]: c for c in engine.discovery_results.get("evaluated_candidates", [])}
        shortlist = {c["compound_id"] for c in engine.discovery_results.get("top_candidates", [])}
        reference_smiles = {r.canonical_smiles: entry["name"] for entry, r, _ in drug_references()}
        records = []
        generated = 0
        for record in engine.compounds.values():
            is_generated = record.transformation_type not in ("seed", "founder", "")
            if is_generated:
                generated += 1
            if source == "directed" and record.compound_id not in directed:
                continue
            if source == "shortlist" and record.compound_id not in shortlist:
                continue
            if source == "universe" and (not is_generated or record.provenance == "Directed discovery pipeline"):
                continue
            if search and search.lower() not in (record.name + record.compound_id + record.formula + record.canonical_smiles).lower():
                continue
            node = engine.genealogy.nodes.get(record.compound_id)
            item = {
                "compound_id": record.compound_id, "name": record.name, "formula": record.formula,
                "canonical_smiles": record.canonical_smiles, "origin_generation": record.origin_generation,
                "transformation_type": record.transformation_type, "parent_ids": record.parent_ids,
                "population": counts[record.compound_id], "birth_step": node.birth_step if node else 0,
                "source": "Directed search" if record.compound_id in directed else "Chemical universe",
                "exact_reference_name": reference_smiles.get(record.canonical_smiles),
                "mol_weight": record.mol_weight, "qed": record.qed, "logp": record.logp,
            }
            if record.compound_id in directed:
                for key in ("composite_utility", "risk_index", "pic50", "in_applicability_domain", "objectives"):
                    item[key] = directed[record.compound_id].get(key)
            records.append(item)
        records.sort(key=lambda c: (c.get("composite_utility") or 0, c["birth_step"], c["origin_generation"], c["compound_id"]), reverse=True)
        return {"compounds": records[offset:offset + limit], "total": len(records), "offset": offset,
                "experiment_id": engine.active_experiment_id,
                "local_ai_running": engine.assistant.is_local_server_running(),
                "registered_compounds": len(engine.compounds), "generated_compounds": generated,
                "population": sum(counts.values()), "generation": engine.clocks.evolutionary_generation,
                "reactions": engine.reactions_count, "job": dict(self.job),
                "shortlist": engine.discovery_results.get("top_candidates", [])}

    def cancel(self):
        self._cancel.set()
        if self.job.get("status") == "running":
            self.job = {**self.job, "status": "cancelled", "message": "Search stopped; unfinished results were not registered."}

    async def start(self, settings, plan=None):
        if self.task and not self.task.done():
            raise ValueError("A search is still running or stopping. Wait before starting another.")
        self._cancel = threading.Event()
        cancel_event = self._cancel
        engine = self.session.engine
        experiment_id = engine.active_experiment_id
        self.job = {"status": "running", "job_id": uuid.uuid4().hex[:12], "stage": 0,
                    "stage_name": "Preparing search", "generation": 0,
                    "target_generations": settings["generations"], "started_at": time.time(),
                    "plan": plan or {}, "message": "Generating real molecular variants"}

        def progress(update):
            if cancel_event.is_set():
                raise SearchCancelled()
            self.job = {**self.job, **update}

        def compute():
            pipeline = CandidateDiscoveryPipeline(disease_id=(plan or {}).get("disease_id", engine.disease_id),
                qsar_predictor=engine.qsar, decision_engine=engine.laya)
            return pipeline.run(**settings, seed=engine.seed, progress_callback=progress)

        async def execute():
            try:
                result = await asyncio.to_thread(compute)
                if cancel_event.is_set() or engine.active_experiment_id != experiment_id:
                    return
                # Includes automatic reference matches for the requested shortlist.
                for candidate in result.get("top_candidates", []):
                    record = engine.chemistry.build_compound(candidate["canonical_smiles"])
                    comparison = compare_drugs(record, limit=3)
                    candidate["drug_comparison"] = {"exact_reference_name": comparison["exact_reference_name"],
                                                    "matches": [{k: m[k] for k in ("name", "similarity", "exact_match")} for m in comparison["matches"]]}
                async with self.session._lock:
                    if cancel_event.is_set() or engine.active_experiment_id != experiment_id:
                        return
                    engine.register_discovery_results(result)
                self.job = {**self.job, "status": "complete", "stage": 15,
                            "message": f"Evaluated {result['total_molecules_evaluated']} structures; {len(result['top_candidates'])} candidates shortlisted.",
                            "completed_at": time.time()}
                await self.session.broadcast({"type": "discovery_updated", "data": dict(self.job)})
            except SearchCancelled:
                pass
            except Exception as error:
                self.job = {**self.job, "status": "failed", "message": str(error)}
        self.task = asyncio.create_task(execute())
        return dict(self.job)
