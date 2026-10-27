"""
EVOCURE Session and Simulation Manager.

Provides thread-safe state management, background simulation stepping loop,
WebSocket broadcasting, and full epistemic contracts bridging EVOCURE PLAY
and EVOCURE LAB across a single shared engine.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import math
import os
import pathlib
import time
import uuid
import hashlib
from typing import Any, Literal

import numpy as np
from fastapi import WebSocket

from simulation.chemistry.engine import ChemistryEngine
from simulation.chemistry.library import COMPOUND_LIBRARY, get_builtin_library
from simulation.core.contracts import (
    CellDeathMode,
    CellType,
    CellularEntity,
    CompartmentBarrier,
    CompoundRecord,
    EntityKind,
    EnvironmentConfig,
    EpistemicCategory,
    GenealogyNode,
    InterfaceMode,
    ScientificValue,
    SimulationClocks,
    SpatialEntity,
    SpatialRegion,
    UniverseMode,
)
from simulation.physics.spatial import (
    PhysicsEngine2D,
    ReactionDiffusionGrid2D,
    SpatialHashGrid,
)
from simulation.biology.diseases.pdac import PDACModule
from simulation.evolution.genealogy import GenealogyTracker
from simulation.evolution.darwinian import DarwinianEcosystemEngine
from simulation.evolution.molecular_ga import DirectedMolecularOptimizer
from simulation.biology.coevolution import CoevolutionEngine
from simulation.pharmacology.admet import PharmacologyEngine
from simulation.toxicity.models import ToxicologyEngine
from simulation.systemic.pbpk import SystemicPBPKEngine
from simulation.systemic.biomarkers import BiologicalIndicatorsEngine
from ai.laya.decision_engine import LayaDecisionEngine
from ai.local_llm.gemma_assistant import LocalGemmaAssistant
from ai.orchestration.registry import get_ai_registry

logger = logging.getLogger("evocure.session")
logging.basicConfig(level=logging.INFO)

# Optional lazy import of external EVOCUREEngine if provided by simulation agent
try:
    from simulation.core.engine import EVOCUREEngine as ExternalEVOCUREEngine  # type: ignore
except (ImportError, ModuleNotFoundError):
    ExternalEVOCUREEngine = None


class InternalEVOCUREEngine:
    """
    Self-contained, production-grade EVOCURE simulation engine.
    Orchestrates Chemistry, Physics, Evolution, PDAC, Pharmacology,
    Systemic PBPK, Toxicology, and Local AI with strict epistemic metadata.
    """

    def __init__(self, seed: int = 20261026) -> None:
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self.interface_mode: str = "PLAY"
        self.universe_mode: str = "ARTIFICIAL_LIFE"
        self.start_wall_time = time.time()

        # Load configs
        self.base_dir = pathlib.Path(__file__).resolve().parent.parent.parent
        self.configs_dir = self.base_dir / "configs"
        self.data_dir = self.base_dir / "data"
        self.checkpoints_dir = self.base_dir / "checkpoints"
        self.experiments_dir = self.base_dir / "experiments"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

        self._load_mission_glossary_config()

        # Clocks
        self.clocks = SimulationClocks(
            steps_per_generation=25,
            generations_per_virtual_year=10.0,
            biological_seconds=0.0,
        )

        # Core Engines
        self.chemistry = ChemistryEngine()
        self.physics = PhysicsEngine2D()
        self.spatial_hash = SpatialHashGrid(cell_size=25.0)
        self.rd_grid = ReactionDiffusionGrid2D(width=1000.0, height=700.0, nx=50, ny=35)
        self.rd_grid.fields["substrate"].fill(1.5)

        self.environment = EnvironmentConfig(
            width=1000.0,
            height=700.0,
            temperature_k=310.15,
            ph=7.4,
            viscosity_pa_s=0.00089,
            oxygen_norm=1.0,
            energy_source_flux=1.0,
            resource_influx_rate=0.4,
            current_vx=0.05,
            current_vy=0.0,
            barriers=[
                CompartmentBarrier(
                    barrier_id="BARRIER-MEMBRANE-CENTRAL",
                    x1=500.0,
                    y1=120.0,
                    x2=500.0,
                    y2=580.0,
                    permeability=0.25,
                )
            ],
            regions=[
                SpatialRegion(
                    region_id="NICHE-HYDROTHERMAL-WEST",
                    name="Warm Catalytic Vent",
                    x=60.0,
                    y=80.0,
                    width=280.0,
                    height=540.0,
                    temperature_delta_k=12.0,
                    ph_override=6.8,
                    oxygen_multiplier=0.85,
                    energy_flux=1.65,
                ),
                SpatialRegion(
                    region_id="NICHE-HOMEOSTATIC-EAST",
                    name="Physiological Buffer Basin",
                    x=620.0,
                    y=80.0,
                    width=320.0,
                    height=540.0,
                    temperature_delta_k=0.0,
                    ph_override=7.4,
                    oxygen_multiplier=1.15,
                    energy_flux=1.0,
                ),
            ],
        )

        # Biology & Disease
        self.tumor = PDACModule(seed=seed, width=800.0, height=600.0, nx=40, ny=30)
        self.tumor.initialize_scenario(seed=seed, preset="standard")

        # Evolution & Genealogy
        self.genealogy = GenealogyTracker()
        self.ecosystem = DarwinianEcosystemEngine(
            genealogy_tracker=self.genealogy,
            compounds_registry={},
            mutation_rate=0.08,
            optimal_temperature_k=310.15,
            optimal_ph=7.4,
        )
        self.ga_optimizer = DirectedMolecularOptimizer()

        # Pharmacology, Toxicology, Systemic
        self.pharmacology = PharmacologyEngine()
        self.toxicology = ToxicologyEngine()
        self.pbpk = SystemicPBPKEngine()
        self.biomarkers_engine = BiologicalIndicatorsEngine()

        # Local AI
        self.laya = LayaDecisionEngine()
        self.gemma = LocalGemmaAssistant()
        self.ai_registry = get_ai_registry()

        # Entities & Live State
        self.entities: list[SpatialEntity] = []
        self.compounds_registry: dict[str, CompoundRecord] = {}
        self.recent_reactions: list[dict[str, Any]] = []
        self.activity_log: list[str] = []
        self.metrics_history: list[dict[str, Any]] = []
        self.achievements_unlocked: set[str] = set()

        # Initialize Default World
        self.reset_world(preset="chemical_life", seed=seed, universe_mode="ARTIFICIAL_LIFE")

    def _load_mission_glossary_config(self) -> None:
        mg_path = self.configs_dir / "missions_and_glossary.json"
        if mg_path.exists():
            try:
                with open(mg_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.missions_config = data.get("missions", [])
                    self.achievements_config = data.get("achievements", [])
                    self.glossary_config = data.get("glossary", [])
                    self.tutorial_steps = data.get("tutorial_steps", [])
                    return
            except Exception as e:
                logger.warning(f"Could not load missions_and_glossary.json: {e}")
        self.missions_config = []
        self.achievements_config = []
        self.glossary_config = []
        self.tutorial_steps = []

    def set_interface_mode(self, mode: str | InterfaceMode) -> str:
        self.interface_mode = str(mode).upper()
        if self.interface_mode not in ("PLAY", "LAB"):
            self.interface_mode = "PLAY"
        return self.interface_mode

    def set_universe_mode(self, mode: str | UniverseMode) -> str:
        self.universe_mode = str(mode).upper()
        if self.universe_mode not in ("ARTIFICIAL_LIFE", "GROUNDED_CHEMISTRY"):
            self.universe_mode = "ARTIFICIAL_LIFE"
        return self.universe_mode

    def update_environment(self, patch: dict[str, Any]) -> dict[str, Any]:
        for k, v in patch.items():
            if hasattr(self.environment, k) and v is not None:
                setattr(self.environment, k, float(v) if isinstance(v, (int, float)) else v)
        self.achievements_unlocked.add("M04")
        return self.environment.to_dict()

    def reset_world(
        self,
        preset: str = "chemical_life",
        seed: int | None = None,
        universe_mode: str | None = None,
    ) -> dict[str, Any]:
        if seed is not None:
            self.seed = seed
            self._rng = np.random.default_rng(seed)

        if universe_mode:
            self.set_universe_mode(universe_mode)

        self.entities.clear()
        self.compounds_registry.clear()
        self.genealogy = GenealogyTracker()
        self.ecosystem.genealogy_tracker = self.genealogy
        self.ecosystem.compounds_registry = self.compounds_registry
        self.spatial_hash.clear()
        self.recent_reactions.clear()
        self.clocks = SimulationClocks(
            steps_per_generation=25,
            generations_per_virtual_year=10.0,
            biological_seconds=0.0,
        )

        preset_clean = preset.lower()
        if preset_clean == "surprise_me":
            choices = ["chemical_life", "cancer_battle", "evolve_compounds"]
            preset_clean = str(self._rng.choice(choices))

        if preset_clean == "chemical_life":
            self.set_universe_mode("ARTIFICIAL_LIFE")
            self._seed_prebiotic_compounds()
        elif preset_clean == "cancer_battle":
            self.set_universe_mode("GROUNDED_CHEMISTRY")
            self.tumor.initialize_scenario(seed=self.seed, preset="standard")
            self._seed_reference_chemo()
        elif preset_clean == "evolve_compounds":
            self.set_universe_mode("GROUNDED_CHEMISTRY")
            self._seed_evolution_compounds()
        elif preset_clean == "sandbox_empty":
            pass

        self.activity_log.append(
            f"Reset universe with preset '{preset_clean}' (Seed: {self.seed}, Mode: {self.universe_mode})"
        )
        return self.get_full_state()

    def _seed_prebiotic_compounds(self) -> None:
        prebiotics = [
            ("Glycine", 24, "autotroph", "PROTOCELL", 200.0, 350.0, "#38bdf8", "circle"),
            ("Pyruvic Acid", 20, "autotroph", "PROTOCELL", 280.0, 300.0, "#4ade80", "hexagon"),
            ("Imidazole", 16, "catalyst", "CATALYST", 150.0, 200.0, "#f59e0b", "diamond"),
            ("Adenine", 14, "cooperator", "PROTOCELL", 750.0, 350.0, "#a855f7", "star"),
            ("Formaldehyde", 30, "neutral", "MOLECULE", 500.0, 350.0, "#94a3b8", "ring"),
        ]
        for name, count, role, kind_str, cx, cy, col, shp in prebiotics:
            rec = self.chemistry.resolve_compound_input(name, input_type="auto")
            self.compounds_registry[rec.compound_id] = rec
            self.genealogy.register_compound(rec, birth_step=self.clocks.evolutionary_step)
            for i in range(count):
                ox = float(self._rng.normal(cx, 45.0))
                oy = float(self._rng.normal(cy, 45.0))
                ox = float(np.clip(ox, 20.0, self.environment.width - 20.0))
                oy = float(np.clip(oy, 20.0, self.environment.height - 20.0))
                ent = SpatialEntity(
                    entity_id=f"ENT-{rec.compound_id[:8]}-{i:03d}-{uuid.uuid4().hex[:4]}",
                    compound_id=rec.compound_id,
                    kind=EntityKind[kind_str],
                    x=ox,
                    y=oy,
                    vx=float(self._rng.normal(0, 0.4)),
                    vy=float(self._rng.normal(0, 0.4)),
                    radius=5.5 if kind_str == "PROTOCELL" else 4.0,
                    mass=max(20.0, rec.mol_weight),
                    energy=12.0,
                    family_id=rec.family_id,
                    color_hex=col,
                    shape=shp,
                    ecological_role=role,
                )
                self.entities.append(ent)
                self.spatial_hash.insert(ent)

    def _seed_reference_chemo(self) -> None:
        chemos = ["Gemcitabine", "5-Fluorouracil", "Erlotinib", "Aspirin"]
        for cname in chemos:
            rec = self.chemistry.resolve_compound_input(cname, input_type="auto")
            self.compounds_registry[rec.compound_id] = rec
            self.genealogy.register_compound(rec, birth_step=self.clocks.evolutionary_step)

    def _seed_evolution_compounds(self) -> None:
        seeds = [
            ("C1=CC=C(C=C1)O", "Phenol", 15, "autotroph", 300.0, 350.0, "#38bdf8", "circle"),
            ("CC(=O)O", "Acetic Acid", 20, "autotroph", 400.0, 300.0, "#34d399", "hexagon"),
            ("C1=CN=CN1", "Imidazole", 15, "catalyst", 700.0, 350.0, "#fbbf24", "diamond"),
        ]
        for smi, name, count, role, cx, cy, col, shp in seeds:
            rec = self.chemistry.resolve_compound_input(smi, input_type="smiles")
            self.compounds_registry[rec.compound_id] = rec
            self.genealogy.register_compound(rec, birth_step=self.clocks.evolutionary_step)
            for i in range(count):
                ox = float(np.clip(self._rng.normal(cx, 40.0), 20.0, self.environment.width - 20.0))
                oy = float(np.clip(self._rng.normal(cy, 40.0), 20.0, self.environment.height - 20.0))
                ent = SpatialEntity(
                    entity_id=f"ENT-{rec.compound_id[:8]}-{i:03d}-{uuid.uuid4().hex[:4]}",
                    compound_id=rec.compound_id,
                    kind=EntityKind.PROTOCELL,
                    x=ox,
                    y=oy,
                    vx=float(self._rng.normal(0, 0.3)),
                    vy=float(self._rng.normal(0, 0.3)),
                    energy=10.0,
                    family_id=rec.family_id,
                    color_hex=col,
                    shape=shp,
                    ecological_role=role,
                )
                self.entities.append(ent)
                self.spatial_hash.insert(ent)

    def add_compound_to_world(
        self,
        query: str,
        input_type: str = "auto",
        count: int = 12,
        x: float | None = None,
        y: float | None = None,
        entity_kind: str = "PROTOCELL",
        ecological_role: str = "autotroph",
    ) -> dict[str, Any]:
        rec = self.chemistry.resolve_compound_input(query, input_type=input_type)
        self.compounds_registry[rec.compound_id] = rec
        self.genealogy.register_compound(rec, birth_step=self.clocks.evolutionary_step)

        cx = float(x) if x is not None else float(self.environment.width / 2.0)
        cy = float(y) if y is not None else float(self.environment.height / 2.0)

        shapes = ["circle", "hexagon", "diamond", "ring", "star", "triangle"]
        colors = ["#38bdf8", "#34d399", "#fbbf24", "#f43f5e", "#a855f7", "#06b6d4"]
        idx = abs(hash(rec.compound_id)) % len(shapes)
        shape = shapes[idx]
        col = colors[idx]

        kind_val = EntityKind.PROTOCELL
        try:
            kind_val = EntityKind[entity_kind.upper()]
        except Exception:
            pass

        spawned = []
        for i in range(count):
            ox = float(np.clip(self._rng.normal(cx, 30.0), 20.0, self.environment.width - 20.0))
            oy = float(np.clip(self._rng.normal(cy, 30.0), 20.0, self.environment.height - 20.0))
            ent = SpatialEntity(
                entity_id=f"ENT-{rec.compound_id[:8]}-{i:03d}-{uuid.uuid4().hex[:4]}",
                compound_id=rec.compound_id,
                kind=kind_val,
                x=ox,
                y=oy,
                vx=float(self._rng.normal(0, 0.5)),
                vy=float(self._rng.normal(0, 0.5)),
                radius=6.0 if kind_val == EntityKind.PROTOCELL else 4.0,
                mass=max(15.0, rec.mol_weight),
                energy=14.0,
                family_id=rec.family_id,
                color_hex=col,
                shape=shape,
                ecological_role=ecological_role,
            )
            self.entities.append(ent)
            self.spatial_hash.insert(ent)
            spawned.append(ent.entity_id)

        self.achievements_unlocked.add("M01")
        self.activity_log.append(
            f"Introduced {count}x {rec.name} ({rec.canonical_smiles}) at ({round(cx, 1)}, {round(cy, 1)})"
        )
        return {"compound": rec.to_dict(), "spawned_count": len(spawned), "entity_ids": spawned}

    def import_sdf_compounds(self, sdf_text: str, spawn_in_world: bool = False) -> list[dict[str, Any]]:
        imported = self.chemistry.import_sdf(sdf_text)
        results = []
        for rec in imported:
            self.compounds_registry[rec.compound_id] = rec
            self.genealogy.register_compound(rec, birth_step=self.clocks.evolutionary_step)
            results.append(rec.to_dict())
            if spawn_in_world:
                self.add_compound_to_world(rec.canonical_smiles, input_type="smiles", count=6)
        self.activity_log.append(f"Imported {len(results)} compounds from SDF payload")
        return results

    def step_universe(self, steps: int = 1, dt: float = 0.25) -> dict[str, Any]:
        reactions_occurred = 0
        for _ in range(steps):
            # 1. Physics integration
            self.physics.step_entities(
                entities=self.entities,
                env=self.environment,
                grid=self.rd_grid,
                dt=dt,
                rng=self._rng,
            )

            # Rebuild spatial hash grid
            self.spatial_hash.clear()
            for ent in self.entities:
                if ent.alive:
                    self.spatial_hash.insert(ent)

            # 2. Reaction diffusion continuous substrate field
            self.rd_grid.step_field("substrate", dt=dt * 0.1, diffusion_coeff=0.08)
            self.rd_grid.fields["substrate"] += self.environment.resource_influx_rate * dt * 0.05
            np.clip(self.rd_grid.fields["substrate"], 0.0, 5.0, out=self.rd_grid.fields["substrate"])

            # 3. Darwinian ecosystem dynamics (feeding, division, decay, mutation)
            if self.universe_mode == "ARTIFICIAL_LIFE":
                eco_res = self.ecosystem.step(
                    entities=self.entities,
                    environment=self.environment,
                    substrate_source=self.rd_grid,
                    dt=dt,
                    rng=self._rng,
                    current_step=self.clocks.evolutionary_step,
                )
                if eco_res.get("births", 0) > 0:
                    self.achievements_unlocked.add("ACH-FIRST-REACTION")

            # 4. Stochastic collision chemical reactions
            new_reactions = self._check_chemical_reactions()
            reactions_occurred += len(new_reactions)

            # Clocks update
            self.clocks.evolutionary_step += 1
            if self.clocks.evolutionary_step % self.clocks.steps_per_generation == 0:
                self.clocks.evolutionary_generation += 1

        self.clocks.computational_seconds = time.time() - self.start_wall_time

        # Update genealogy abundances
        abundances: dict[str, int] = {}
        for ent in self.entities:
            if ent.alive:
                abundances[ent.compound_id] = abundances.get(ent.compound_id, 0) + 1
        self.genealogy.update_abundances(abundances, current_step=self.clocks.evolutionary_step)

        # Check mission milestones
        if self.clocks.evolutionary_generation >= 100:
            self.achievements_unlocked.add("M03")
            self.achievements_unlocked.add("ACH-CENTURY-LINEAGE")
        alive_entities = [e for e in self.entities if e.alive]
        if len(set(e.family_id for e in alive_entities)) >= 2:
            self.achievements_unlocked.add("M02")

        # Snapshot summary
        metrics = {
            "step": self.clocks.evolutionary_step,
            "generation": self.clocks.evolutionary_generation,
            "virtual_years": self.clocks.virtual_years,
            "population": len(alive_entities),
            "families": len(set(e.family_id for e in alive_entities)),
            "reactions": reactions_occurred,
            "mean_energy": (
                round(float(np.mean([e.energy for e in alive_entities])), 2)
                if alive_entities
                else 0.0
            ),
        }
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > 300:
            self.metrics_history.pop(0)

        return metrics

    def _check_chemical_reactions(self) -> list[dict[str, Any]]:
        reactions = []
        alive_ents = [e for e in self.entities if e.alive]
        if len(alive_ents) < 2:
            return reactions

        sample_size = min(12, len(alive_ents))
        sample_indices = self._rng.choice(len(alive_ents), size=sample_size, replace=False)
        for idx in sample_indices:
            ent = alive_ents[idx]
            neighbors = self.spatial_hash.query_neighbors(ent.x, ent.y, radius=12.0)
            for other in neighbors:
                if other.entity_id == ent.entity_id or not other.alive:
                    continue
                if ent.compound_id in self.compounds_registry and other.compound_id in self.compounds_registry:
                    rec_a = self.compounds_registry[ent.compound_id]
                    rec_b = self.compounds_registry[other.compound_id]
                    res = self.chemistry.attempt_reaction(
                        rec_a,
                        rec_b,
                        temperature_k=self.environment.temperature_k,
                        ph=self.environment.ph,
                        rng=self._rng,
                    )
                    if res.get("reaction_occurred") and res.get("products"):
                        reactions.append(res)
                        self.recent_reactions.append({
                            "x": (ent.x + other.x) / 2.0,
                            "y": (ent.y + other.y) / 2.0,
                            "name": res.get("reaction_name", "Transformation"),
                            "step": self.clocks.evolutionary_step,
                        })
                        if len(self.recent_reactions) > 20:
                            self.recent_reactions.pop(0)
                        self.achievements_unlocked.add("ACH-FIRST-REACTION")
                        break
        return reactions

    def advance_by_preset(self, preset: str) -> dict[str, Any]:
        jump_steps = {
            "10_gen": 250,
            "100_gen": 1000,
            "1000_gen": 2500,
            "10000_gen": 5000,
            "1_yr": 250,
            "10_yr": 1000,
            "100_yr": 2500,
            "1000_yr": 5000,
            "10000_yr": 10000,
        }
        steps = jump_steps.get(preset, 250)
        batch_size = 50
        remaining = steps
        while remaining > 0:
            cur = min(batch_size, remaining)
            self.step_universe(steps=cur, dt=0.5)
            remaining -= cur
        self.activity_log.append(f"Accelerated evolutionary time by preset '{preset}' ({steps} steps)")
        return self.get_full_state()

    def step_tumor(self, steps: int = 1, dt_seconds: float = 900.0) -> dict[str, Any]:
        res = {}
        for _ in range(steps):
            res = self.tumor.step(dt_seconds=dt_seconds)
            self.clocks.biological_seconds += dt_seconds
        self.clocks.computational_seconds = time.time() - self.start_wall_time

        if res.get("resistant_cells", 0) > 10:
            self.achievements_unlocked.add("M07")
        if self.tumor.step_count >= 10:
            self.achievements_unlocked.add("M05")

        return res

    def introduce_compound_to_tumor(
        self,
        compound_query: str,
        applied_concentration_um: float = 5.0,
        injection_region: str = "vascular",
        enable_systemic_pbpk: bool = False,
        dose_mg: float = 100.0,
    ) -> dict[str, Any]:
        rec = self.chemistry.resolve_compound_input(compound_query, input_type="auto")
        self.compounds_registry[rec.compound_id] = rec
        admet = self.pharmacology.evaluate_admet(rec)
        tox = self.toxicology.evaluate_toxicology(rec)

        pbpk_res = None
        effective_conc = applied_concentration_um
        if enable_systemic_pbpk:
            pbpk_res = self.pbpk.simulate(rec, dose_mg=dose_mg)
            pancreas_conc = pbpk_res.get("summary", {}).get("c_max_pancreas_um", applied_concentration_um)
            effective_conc = float(pancreas_conc)

        res = self.tumor.apply_compound_exposure(
            compound=rec,
            pharmacology_profile=admet,
            applied_concentration_um=effective_conc,
            injection_region=injection_region,
        )
        self.achievements_unlocked.add("M05")
        self.activity_log.append(
            f"Introduced candidate {rec.name} to PDAC microenvironment (Exposure: {round(effective_conc, 2)} uM, PBPK={enable_systemic_pbpk})"
        )
        return {
            "tumor_response": res,
            "compound": rec.to_dict(),
            "admet": admet,
            "toxicology": tox,
            "pbpk": pbpk_res,
        }

    def run_tumor_controlled_comparison(
        self,
        compound_query: str,
        applied_concentration_um: float = 5.0,
        duration_hours: float = 48.0,
        seed: int | None = None,
    ) -> dict[str, Any]:
        use_seed = seed if seed is not None else self.seed
        rec = self.chemistry.resolve_compound_input(compound_query, input_type="auto")
        admet = self.pharmacology.evaluate_admet(rec)

        comparison = self.tumor.run_controlled_comparison(
            compound=rec,
            pharmacology_profile=admet,
            applied_concentration_um=applied_concentration_um,
            duration_hours=duration_hours,
            seed=use_seed,
        )
        self.achievements_unlocked.add("M06")
        self.achievements_unlocked.add("ACH-RIGOROUS-CONTROL")
        self.activity_log.append(
            f"Completed controlled comparison for {rec.name} across {duration_hours}h (Seed: {use_seed})"
        )
        return comparison

    def run_coevolution_experiment(
        self,
        n_cycles: int = 4,
        hours_per_cycle: float = 24.0,
        applied_concentration_um: float = 3.0,
    ) -> dict[str, Any]:
        coev = CoevolutionEngine(
            tumor_module=self.tumor,
            pharmacology_callback=lambda c, ctx: self.pharmacology.evaluate_admet(c),
            seed=self.seed,
        )
        trajectory = coev.run_coevolution(
            n_cycles=n_cycles,
            hours_per_cycle=hours_per_cycle,
            applied_concentration_um=applied_concentration_um,
        )
        self.achievements_unlocked.add("M07")
        self.activity_log.append(f"Completed coevolution run: {n_cycles} cycles x {hours_per_cycle}h")
        return trajectory

    def run_discovery_pipeline(
        self,
        seed_queries: list[str] | None = None,
        generations: int = 4,
        population_size: int = 16,
        top_k_cellular: int = 3,
    ) -> dict[str, Any]:
        if not seed_queries:
            seed_queries = [
                "C1=CN(C(=O)N=C1N)[C@H]2C([C@@H]([C@H](O2)CO)O)(F)F",  # Gemcitabine
                "O=C1NC(=O)NC=C1F",                                      # 5-FU
                "COc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC",                 # Erlotinib
                "CC(=O)Oc1ccccc1C(=O)O",                                 # Aspirin
            ]

        pop_records = [self.chemistry.resolve_compound_input(q, input_type="auto") for q in seed_queries]
        pop_smiles = [r.canonical_smiles for r in pop_records]

        def evaluate_candidate(smi: str) -> dict[str, float]:
            try:
                rec = self.chemistry.resolve_compound_input(smi, input_type="smiles")
                admet = self.pharmacology.evaluate_admet(rec)
                tox = self.toxicology.evaluate_toxicology(rec)
                potency = admet.get("panc1_ic50_um", {}).get("value", 50.0)
                potency_score = 1.0 / (1.0 + float(potency if potency else 50.0))
                selectivity = admet.get("selectivity_index", {}).get("value", 1.0)
                tox_risk = tox.get("overall_risk_index", {}).get("value", 1.0)
                qed = rec.qed
                return {
                    "potency": float(potency_score),
                    "selectivity": float(selectivity if selectivity else 1.0),
                    "qed": float(qed),
                    "safety": float(1.0 - (tox_risk if tox_risk else 0.5)),
                }
            except Exception:
                return {"potency": 0.0, "selectivity": 0.0, "qed": 0.0, "safety": 0.0}

        ga_res = self.ga_optimizer.evolve_population(
            population_smiles=pop_smiles,
            evaluate_fn=evaluate_candidate,
            population_size=population_size,
            rng=self._rng,
            generation=generations,
        )

        candidates = []
        front_0 = ga_res.get("pareto_front_0", [])[:top_k_cellular]
        for ind in front_0:
            smi = ind.get("smiles", "")
            rec = self.chemistry.resolve_compound_input(smi, input_type="smiles")
            self.compounds_registry[rec.compound_id] = rec
            admet = self.pharmacology.evaluate_admet(rec)
            tox = self.toxicology.evaluate_toxicology(rec)
            candidates.append({
                "compound": rec.to_dict(),
                "objectives": ind.get("objectives", {}),
                "admet": admet,
                "toxicology": tox,
            })

        self.achievements_unlocked.add("M08")
        self.achievements_unlocked.add("ACH-PARETO-EXPLORER")
        self.achievements_unlocked.add("ACH-SELECTIVITY-GUARDIAN")
        self.activity_log.append(f"Completed directed discovery pipeline: {len(candidates)} Pareto candidates")
        return {
            "generations": generations,
            "population_size": population_size,
            "pareto_candidates": candidates,
            "diversity_score": ga_res.get("diversity_score", 0.75),
            "epistemic_notice": "In silico candidate ranking requires wet-lab validation before clinical inference.",
        }

    def inspect_compound(
        self, compound_id_or_query: str, enable_systemic_pbpk: bool = False
    ) -> dict[str, Any]:
        rec = None
        if compound_id_or_query in self.compounds_registry:
            rec = self.compounds_registry[compound_id_or_query]
        else:
            rec = self.chemistry.resolve_compound_input(compound_id_or_query, input_type="auto")
            self.compounds_registry[rec.compound_id] = rec

        admet = self.pharmacology.evaluate_admet(rec)
        tox = self.toxicology.evaluate_toxicology(rec)
        pbpk_data = None
        if enable_systemic_pbpk:
            pbpk_data = self.pbpk.simulate(rec)

        bio_indicators = self.biomarkers_engine.compute_biological_indicators(
            tumor_microenvironment_state=self.tumor.get_snapshot(),
            systemic_pbpk_result=pbpk_data,
            compound=rec,
            toxicity_profile=tox,
        )

        return {
            "compound": rec.to_dict(),
            "admet": admet,
            "toxicology": tox,
            "pbpk": pbpk_data,
            "biological_indicators": bio_indicators,
            "epistemic_tags": {
                "physicochemical": EpistemicCategory.CALCULATED_PHYSICOCHEMICAL.value,
                "admet": EpistemicCategory.VALIDATED_MODEL_PREDICTION.value,
                "toxicity": EpistemicCategory.EXPLORATORY_MODEL.value,
                "cellular_response": EpistemicCategory.MECHANISTIC_SIMULATION.value,
            },
        }

    def get_genealogy_view(self, filters: dict[str, Any] | None = None) -> dict[str, Any]:
        flt = filters or {}
        return self.genealogy.query_tree(
            family_id=flt.get("family_id"),
            min_gen=flt.get("min_gen"),
            max_gen=flt.get("max_gen"),
            include_extinct=flt.get("include_extinct", True),
            min_similarity=flt.get("min_similarity", 0.0),
            min_persistence=flt.get("min_persistence", 0),
        )

    def get_biological_indicators(self, enable_systemic_pbpk: bool | None = None) -> dict[str, Any]:
        active_comp = self.tumor.active_compound
        pbpk_res = None
        if enable_systemic_pbpk and active_comp:
            pbpk_res = self.pbpk.simulate(active_comp)

        return self.biomarkers_engine.compute_biological_indicators(
            tumor_microenvironment_state=self.tumor.get_snapshot(),
            systemic_pbpk_result=pbpk_res,
            compound=active_comp,
            toxicity_profile=self.toxicology.evaluate_toxicology(active_comp) if active_comp else None,
        )

    def ask_assistant(
        self,
        question: str,
        provider: str | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        context = {
            "interface_mode": self.interface_mode,
            "universe_mode": self.universe_mode,
            "clocks": self.clocks.to_dict(),
            "entity_count": len([e for e in self.entities if e.alive]),
            "temperature_k": self.environment.temperature_k,
            "ph": self.environment.ph,
            "tumor_step": self.tumor.step_count,
            "tumor_viability": self.tumor.get_snapshot().get("summary_metrics", {}).get("mean_viability", 1.0),
        }
        res = self.gemma.ask(
            question=question,
            simulation_context=context,
            mode=InterfaceMode.LAB if self.interface_mode == "LAB" else InterfaceMode.PLAY,
            provider_override=provider,
            model_override=model,
        )

        action_proposal = None
        q_lower = question.lower()
        if "cool" in q_lower or "decrease temp" in q_lower:
            action_proposal = {
                "type": "update_environment",
                "label": "Decrease Temperature by 10 K",
                "payload": {"temperature_k": max(273.15, self.environment.temperature_k - 10.0)},
            }
        elif "warm" in q_lower or "increase temp" in q_lower:
            action_proposal = {
                "type": "update_environment",
                "label": "Increase Temperature by 10 K",
                "payload": {"temperature_k": min(373.15, self.environment.temperature_k + 10.0)},
            }
        elif "gemcitabine" in q_lower and ("add" in q_lower or "introduce" in q_lower):
            action_proposal = {
                "type": "add_compound",
                "label": "Add Gemcitabine to Universe",
                "payload": {"query": "Gemcitabine", "count": 12},
            }
        elif "compare" in q_lower and "control" in q_lower:
            action_proposal = {
                "type": "run_comparison",
                "label": "Run Controlled Tumor Comparison (Gemcitabine)",
                "payload": {"compound_query": "Gemcitabine", "duration_hours": 48.0},
            }

        ans_text = res.get("answer") or res.get("response") or "EVOCURE scientific assistant ready."
        result_payload = {
            "question": question,
            "answer": ans_text,
            "response": ans_text,
            "model": res.get("provider") or res.get("model") or "Deterministic Scientific Fallback / Local Gemma",
            "provider": res.get("provider", "deterministic"),
            "proposed_action": action_proposal or res.get("proposed_action"),
            "epistemic_category": EpistemicCategory.AI_HYPOTHESIS.value,
            "guardrail_checks": res.get("guardrail_checks", {"cancer_cure_claim_prevented": True}),
        }
        if res.get("fallback_reason"):
            result_payload["fallback_reason"] = res["fallback_reason"]
        return result_payload

    def apply_assistant_proposed_action(self, action: dict[str, Any]) -> dict[str, Any]:
        atype = action.get("type")
        payload = action.get("payload", {})
        if atype == "update_environment":
            return self.update_environment(payload)
        elif atype == "add_compound":
            return self.add_compound_to_world(
                query=payload.get("query", "Glycine"),
                count=payload.get("count", 12),
            )
        elif atype == "run_comparison":
            return self.run_tumor_controlled_comparison(
                compound_query=payload.get("compound_query", "Gemcitabine"),
                duration_hours=payload.get("duration_hours", 48.0),
            )
        return {"success": False, "message": f"Unknown action type: {atype}"}

    def compare_worlds(
        self,
        scenario_left: dict[str, Any],
        scenario_right: dict[str, Any],
        steps: int = 60,
        repeats: int = 3,
        seed: int = 42,
    ) -> dict[str, Any]:
        results_left = []
        results_right = []
        for rep in range(repeats):
            rep_seed = seed + rep * 100
            eng_l = InternalEVOCUREEngine(seed=rep_seed)
            eng_l.reset_world(preset=scenario_left.get("preset", "chemical_life"))
            if scenario_left.get("compound"):
                eng_l.add_compound_to_world(scenario_left["compound"], count=16)
            eng_l.step_universe(steps=steps)
            results_left.append(eng_l.metrics_history[-1] if eng_l.metrics_history else {})

            eng_r = InternalEVOCUREEngine(seed=rep_seed)
            eng_r.reset_world(preset=scenario_right.get("preset", "chemical_life"))
            if scenario_right.get("compound"):
                eng_r.add_compound_to_world(scenario_right["compound"], count=16)
            eng_r.step_universe(steps=steps)
            results_right.append(eng_r.metrics_history[-1] if eng_r.metrics_history else {})

        self.activity_log.append(f"Completed Compare Worlds side-by-side experiment ({repeats} repeats)")
        return {
            "scenario_left": scenario_left,
            "scenario_right": scenario_right,
            "steps": steps,
            "repeats": repeats,
            "metrics_left_mean": {
                "population": float(np.mean([r.get("population", 0) for r in results_left])),
                "families": float(np.mean([r.get("families", 0) for r in results_left])),
                "mean_energy": float(np.mean([r.get("mean_energy", 0) for r in results_left])),
            },
            "metrics_right_mean": {
                "population": float(np.mean([r.get("population", 0) for r in results_right])),
                "families": float(np.mean([r.get("families", 0) for r in results_right])),
                "mean_energy": float(np.mean([r.get("mean_energy", 0) for r in results_right])),
            },
        }

    def create_checkpoint(self, label: str = "") -> dict[str, Any]:
        ckpt_id = f"CKPT-{int(time.time())}-{uuid.uuid4().hex[:6]}"
        record = {
            "checkpoint_id": ckpt_id,
            "label": label or f"Checkpoint at Gen {self.clocks.evolutionary_generation}",
            "timestamp": time.time(),
            "clocks": self.clocks.to_dict(),
            "environment": self.environment.to_dict(),
            "entities_count": len([e for e in self.entities if e.alive]),
            "universe_mode": self.universe_mode,
            "interface_mode": self.interface_mode,
        }
        file_path = self.checkpoints_dir / f"{ckpt_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        self.activity_log.append(f"Created checkpoint: {record['label']} ({ckpt_id})")
        return record

    def list_checkpoints(self) -> list[dict[str, Any]]:
        ckpts = []
        for p in self.checkpoints_dir.glob("*.json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    ckpts.append(json.load(f))
            except Exception:
                pass
        ckpts.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        return ckpts

    def restore_checkpoint(self, checkpoint_id: str) -> dict[str, Any]:
        file_path = self.checkpoints_dir / f"{checkpoint_id}.json"
        if not file_path.exists():
            return {"success": False, "message": f"Checkpoint {checkpoint_id} not found"}
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.activity_log.append(f"Restored checkpoint: {data.get('label', checkpoint_id)}")
        return {"success": True, "data": data}

    def save_current_experiment(self, title: str = "") -> dict[str, Any]:
        exp_id = f"EXP-{int(time.time())}-{uuid.uuid4().hex[:6]}"
        data = {
            "experiment_id": exp_id,
            "title": title or f"Experiment {self.universe_mode} Gen {self.clocks.evolutionary_generation}",
            "created_at": time.time(),
            "universe_mode": self.universe_mode,
            "interface_mode": self.interface_mode,
            "clocks": self.clocks.to_dict(),
            "environment": self.environment.to_dict(),
            "metrics_history": self.metrics_history,
            "activity_log": self.activity_log[-50:],
            "achievements": list(self.achievements_unlocked),
        }
        file_path = self.experiments_dir / f"{exp_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return data

    def list_experiments(self) -> list[dict[str, Any]]:
        exps = []
        for p in self.experiments_dir.glob("*.json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    exps.append(json.load(f))
            except Exception:
                pass
        exps.sort(key=lambda x: x.get("created_at", 0), reverse=True)
        return exps

    def load_saved_experiment(self, experiment_id: str) -> dict[str, Any]:
        file_path = self.experiments_dir / f"{experiment_id}.json"
        if not file_path.exists():
            return {"success": False, "message": f"Experiment {experiment_id} not found"}
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {"success": True, "data": data}

    def get_full_state(self, max_visible_particles: int = 1500) -> dict[str, Any]:
        alive_entities = [e for e in self.entities if e.alive]
        visible_entities = alive_entities[:max_visible_particles]
        entities_render = [e.to_render_dict() for e in visible_entities]
        tumor_snapshot = self.tumor.get_snapshot()
        substrate_heatmap = self.rd_grid.to_heatmap_list("substrate")

        return {
            "interface_mode": self.interface_mode,
            "universe_mode": self.universe_mode,
            "clocks": self.clocks.to_dict(),
            "environment": self.environment.to_dict(),
            "entities": entities_render,
            "entity_total_count": len(alive_entities),
            "entity_visible_count": len(visible_entities),
            "recent_reactions": self.recent_reactions[-10:],
            "tumor": tumor_snapshot,
            "substrate_heatmap": substrate_heatmap,
            "metrics": (
                self.metrics_history[-1]
                if self.metrics_history
                else {
                    "step": 0,
                    "generation": 0,
                    "population": len(alive_entities),
                    "reactions": 0,
                }
            ),
            "achievements_unlocked": list(self.achievements_unlocked),
            "activity_log": self.activity_log[-20:],
        }


class SessionManager:
    """
    Thread- and async-safe manager holding the live simulation engine,
    background stepping loop supporting 1x/10x/100x speeds for Universe
    or Tumor clocks, and WebSocket live broadcasting.
    """

    _instance: SessionManager | None = None

    def __init__(self) -> None:
        if ExternalEVOCUREEngine is not None:
            try:
                self.engine = ExternalEVOCUREEngine()
                logger.info("Instantiated external EVOCUREEngine from simulation.core.engine")
            except Exception as e:
                logger.warning(f"Could not instantiate external EVOCUREEngine: {e}, falling back to internal engine")
                self.engine = InternalEVOCUREEngine()
        else:
            self.engine = InternalEVOCUREEngine()
            logger.info("Instantiated internal EVOCUREEngine")

        self._lock = asyncio.Lock()
        self._subscribers: set[WebSocket] = set()
        self._background_task: asyncio.Task | None = None

        self.playback_action: str = "pause"
        self.speed_multiplier: int = 1
        self.playback_target: str = "universe"
        self.fps: float = 20.0
        self.is_running: bool = False
        self._godot_chart_history: list[dict[str, Any]] = []
        self.advance_target_step: int | None = None
        self.advance_start_step: int = 0

    @classmethod
    def get_instance(cls) -> SessionManager:
        if cls._instance is None:
            cls._instance = SessionManager()
        return cls._instance

    def get_normalized_state(self) -> dict[str, Any]:
        state = self.engine.get_full_state()
        if "tumor" not in state and hasattr(self.engine, "tumor_module"):
            state["tumor"] = self.engine.tumor_module.get_snapshot()
        clocks_dict = state.get("clocks", {})
        if hasattr(clocks_dict, "to_dict"):
            clocks_dict = clocks_dict.to_dict()
            state["clocks"] = clocks_dict
        if "metrics" not in state:
            alive_count = state.get("total_entity_count", len(state.get("entities", [])))
            state["metrics"] = {
                "step": clocks_dict.get("evolutionary_step", 0),
                "generation": clocks_dict.get("evolutionary_generation", 0),
                "virtual_years": clocks_dict.get("virtual_years", 0.0),
                "population": alive_count,
                "reactions": state.get("reactions_count", 0),
            }
        if "entity_total_count" not in state:
            state["entity_total_count"] = state.get("total_entity_count", len(state.get("entities", [])))
        if "entity_visible_count" not in state:
            state["entity_visible_count"] = state.get("visible_entity_count", len(state.get("entities", [])))
        if "achievements_unlocked" not in state:
            state["achievements_unlocked"] = list(getattr(self.engine, "unlocked_achievements", set()))
        if "activity_log" not in state:
            state["activity_log"] = getattr(self.engine, "recent_events", [])

        self._record_godot_chart_history(state)
        state = self.enrich_state_for_godot(state)
        state["playback"] = {"is_running": self.is_running, "speed_multiplier": self.speed_multiplier, "target": self.playback_target}
        state["advance_job"] = {
            "active": self.advance_target_step is not None,
            "target_step": self.advance_target_step,
            "start_step": self.advance_start_step,
        }
        return state

    def _record_godot_chart_history(self, state: dict[str, Any] | None = None) -> None:
        """Maintain rolling buffer of up to 120 points for Godot telemetry charts."""
        clocks = getattr(self.engine, "clocks", None)
        step = getattr(clocks, "evolutionary_step", 0) if clocks else 0
        gen = getattr(clocks, "evolutionary_generation", 0) if clocks else 0
        v_years = float(getattr(clocks, "virtual_years", gen * 0.1)) if clocks else float(gen) * 0.1

        entities = [e for e in getattr(self.engine, "entities", []) if getattr(e, "alive", True)]
        alive_entities = len(entities)

        fam_pops = {
            "catalysts": 0,
            "transporters": 0,
            "degraders": 0,
            "antitumor": 0,
            "structural": 0,
            "others": 0,
        }
        tox_vals: list[float] = []
        anti_vals: list[float] = []
        for e in entities:
            role = str(getattr(e, "ecological_role", "neutral")).lower()
            kind = str(getattr(getattr(e, "kind", None), "value", getattr(e, "kind", "MOLECULE"))).upper()
            if "catalyst" in role or kind == "CATALYST":
                fam_pops["catalysts"] += 1
            elif "cooperator" in role or "transport" in role:
                fam_pops["transporters"] += 1
            elif "parasite" in role or "degrad" in role:
                fam_pops["degraders"] += 1
            elif kind in ("DRUG", "THERAPEUTIC"):
                fam_pops["antitumor"] += 1
            elif "autotroph" in role or kind == "PROTOCELL":
                fam_pops["structural"] += 1
            else:
                fam_pops["others"] += 1
            e_energy = float(getattr(e, "energy", 10.0))
            tox_vals.append(0.0)
            anti_vals.append(0.0)

        mean_tox = round(float(np.mean(tox_vals)), 3) if tox_vals else 0.18
        std_tox = round(float(np.std(tox_vals)), 3) if len(tox_vals) > 1 else 0.06
        mean_anti = round(float(np.mean(anti_vals)), 3) if anti_vals else 0.42
        std_anti = round(float(np.std(anti_vals)), 3) if len(anti_vals) > 1 else 0.08

        tumor = getattr(self.engine, "tumor", getattr(self.engine, "tumor_module", None))
        tumor_cells = 0
        resistant_cells = 0
        if tumor and hasattr(tumor, "get_snapshot"):
            snap = tumor.get_snapshot()
            metrics = snap.get("summary_metrics", {})
            tumor_cells = int(metrics.get("viable_cell_count", metrics.get("tumor_cell_count", 0)))
            resistant_cells = int(metrics.get("resistant_cell_count", 0))

        env = getattr(self.engine, "environment", None)
        ph = float(getattr(env, "ph", 7.4)) if env else 7.4
        o2 = float(getattr(env, "oxygen_norm", 1.0)) if env else 1.0

        point = {
            "mean_energy_norm": float(np.mean([min(1.0, max(0.0, e.energy / max(1.0, e.replication_threshold))) for e in entities])) if entities else 0.0,
            "species_diversity_ratio": len({e.compound_id for e in entities}) / max(1, len(entities)),
            "step": step,
            "generation": gen,
            "virtual_years": round(v_years, 2),
            "population": alive_entities or tumor_cells,
            "family_populations": fam_pops,
            "mean_toxicity": mean_tox,
            "std_toxicity": std_tox,
            "mean_antitumor": mean_anti,
            "std_antitumor": std_anti,
            "tumor_cells": tumor_cells,
            "resistant_cells": resistant_cells,
            "oxygen_norm": round(o2, 3),
            "ph": round(ph, 3),
            "timestamp": round(time.time(), 2),
        }
        self._godot_chart_history.append(point)
        if len(self._godot_chart_history) > 120:
            self._godot_chart_history = self._godot_chart_history[-120:]

    def enrich_state_for_godot(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Enrich normalized simulation state with Godot 4.7 telemetry, rolling chart history,
        visual genome representations for 3D/2D procedural sprites, and multi-AI summary.
        """
        # 1. Rolling chart history
        from simulation.discovery.workbench import molecule_layout
        compounds_registry = getattr(self.engine, "compounds", {})
        active_compound_ids = {item.get("cid", item.get("compound_id", "")) for item in state.get("entities", [])}
        state["structure_catalog"] = {cid: molecule_layout(compounds_registry[cid].canonical_smiles) for cid in active_compound_ids if cid in compounds_registry}
        state["chart_history"] = list(self._godot_chart_history)

        # 2. Multi-AI provider summary
        assistant = getattr(self.engine, "assistant", getattr(self.engine, "gemma", None))
        if assistant and hasattr(assistant, "get_providers_status"):
            prov_status = assistant.get_providers_status()
            state["ai_provider_summary"] = {
                "active_provider": prov_status.get("active_provider", "auto_local"),
                "providers": {
                    k: {
                        "configured": v.get("configured", False),
                        "masked_key": v.get("masked_key", ""),
                        "current_model": v.get("current_model", ""),
                    }
                    for k, v in prov_status.get("providers", {}).items()
                },
            }
        else:
            state["ai_provider_summary"] = {
                "active_provider": "deterministic",
                "providers": {"deterministic": {"configured": True, "masked_key": ""}},
            }

        # 3. Visual genome derivation for entities
        raw_entities = state.get("entities", [])
        enriched_entities = []
        compounds = getattr(self.engine, "compounds", getattr(self.engine, "compounds_registry", {}))

        for ent in raw_entities:
            ent_dict = ent if isinstance(ent, dict) else (ent.to_dict() if hasattr(ent, "to_dict") else vars(ent))
            ent_dict = dict(ent_dict)

            cid = str(ent_dict.get("cid") or ent_dict.get("compound_id") or "")
            rec = compounds.get(cid)

            smiles = str(getattr(rec, "canonical_smiles", "") if rec else (ent_dict.get("smiles", "") or ""))
            display_name = str(getattr(rec, "name", "") if rec else (ent_dict.get("name", "") or cid))
            name = display_name.lower()
            formula = str(getattr(rec, "formula", "") if rec else (ent_dict.get("formula", "") or ""))

            if "c1ccccc1" in smiles.lower() or "benzene" in name:
                archetype = "benzene"
            elif "phosphate" in name or "atp" in name or "nucleotide" in name or "adenosine" in name or ("p" in formula.lower() and "o" in formula.lower()):
                archetype = "phosphate"
            elif "glucose" in name or "sugar" in name or formula == "C6H12O6":
                archetype = "glucose"
            elif "lipid" in name or "palmitic" in name or "fatty" in name or ("c" in formula.lower() and len(smiles) > 18 and "n" not in smiles.lower()):
                archetype = "lipid"
            elif "amino" in name or "alanine" in name or "glycine" in name or "cysteine" in name or "histidine" in name:
                archetype = "amino"
            elif formula in ("O2", "OO") or "oxygen" in name or "peroxide" in name:
                archetype = "dioxygen"
            elif formula in ("H2O", "OH2") or "water" in name:
                archetype = "water"
            elif formula == "CH4" or "methane" in name:
                archetype = "methane"
            else:
                archetype = "modular"

            ent_id_str = str(ent_dict.get("id", ent_dict.get("entity_id", cid)))
            seed_val = int(hashlib.sha256(cid.encode()).hexdigest()[:8], 16)

            head_id = int(ent_dict.get("head_id", seed_val % 12))
            body_id = int(ent_dict.get("body_id", (seed_val // 12) % 12))
            legs_id = int(ent_dict.get("legs_id", (seed_val // 144) % 8))
            appendage_id = int(ent_dict.get("appendage_id", (seed_val // 1152) % 8))
            crystal_id = int(ent_dict.get("crystal_id", (seed_val // 9216) % 6))

            ent_kind_str = str(ent_dict.get("k", ent_dict.get("kind", ent_dict.get("entity_kind", "PROTOCELL"))))
            eco_role_str = str(ent_dict.get("role", ent_dict.get("ecological_role", "autotroph")))
            family_label = str(ent_dict.get("fam", ent_dict.get("family_id", "Family Alpha")))

            default_tox = 0.0
            tox_raw = float(ent_dict.get("toxicity", default_tox) or default_tox)
            toxicity_norm = float(np.clip(tox_raw, 0.0, 1.0))

            is_therapeutic = ent_kind_str in ("DRUG", "THERAPEUTIC") or "drug" in eco_role_str.lower() or "therapeutic" in family_label.lower()
            default_eff = 0.0
            eff_raw = float(ent_dict.get("potency", ent_dict.get("antitumor_efficacy", default_eff)) or default_eff)
            antitumor_norm = float(np.clip(eff_raw, 0.0, 1.0))

            eng_raw = float(ent_dict.get("e", ent_dict.get("energy", 18.0)) or 18.0)
            energy_norm = float(np.clip(eng_raw / 28.0, 0.15, 1.0))

            mut_rate = float(ent_dict.get("mutation_rate", 0.12) or 0.12)

            # Bioluminescent glow calculation
            if toxicity_norm > 0.6:
                glow_color = "#ff3366"  # Bio-hazard red
            elif antitumor_norm > 0.6:
                glow_color = "#00f5d4"  # Antitumor teal
            elif energy_norm > 0.7:
                glow_color = "#fee440"  # High energy amber
            else:
                glow_color = "#70d6ff"  # Quiescent cyan

            ent_dict["entity_id"] = ent_id_str
            ent_dict["compound_id"] = cid
            ent_dict["name"] = display_name
            ent_dict["formula"] = formula
            ent_dict["smiles"] = smiles
            # Structures are fetched on selection, not repeated 800 times per frame.
            ent_dict["svg_2d"] = ""
            ent_dict["mol_weight"] = float(getattr(rec, "mol_weight", 180.0) if rec else 180.0)
            ent_dict["logp"] = float(getattr(rec, "logp", 0.5) if rec else 0.5)
            ent_dict["qed"] = float(getattr(rec, "qed", 0.62) if rec else 0.62)
            ent_dict["generation"] = int(ent_dict.get("gen", ent_dict.get("generation", 0)))
            ent_dict["ecological_role"] = eco_role_str
            ent_dict["entity_kind"] = ent_kind_str
            ent_dict["archetype"] = archetype
            ent_dict["head_id"] = head_id
            ent_dict["body_id"] = body_id
            ent_dict["legs_id"] = legs_id
            ent_dict["appendage_id"] = appendage_id
            ent_dict["crystal_id"] = crystal_id
            ent_dict["toxicity_norm"] = toxicity_norm
            ent_dict["bioactivity_available"] = "toxicity" in ent_dict or "potency" in ent_dict
            ent_dict["antitumor_norm"] = antitumor_norm
            ent_dict["energy_norm"] = energy_norm
            ent_dict["mutation_rate"] = mut_rate
            ent_dict["family_label"] = family_label
            ent_dict["glow_color"] = glow_color

            enriched_entities.append(ent_dict)

        state["entities"] = enriched_entities
        return state

    async def start_background_loop(self) -> None:
        if self._background_task is None or self._background_task.done():
            self._background_task = asyncio.create_task(self._simulation_loop())
            logger.info("Started background simulation stepping loop")

    async def stop_background_loop(self) -> None:
        if self._background_task and not self._background_task.done():
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
            logger.info("Stopped background simulation loop")

    async def _simulation_loop(self) -> None:
        frame_interval = 1.0 / self.fps
        last_broadcast = 0.0
        while True:
            try:
                start = time.monotonic()
                if self.is_running and self.playback_action == "play":
                    async with self._lock:
                        if self.playback_target == "universe":
                            steps = min(4, self.speed_multiplier)
                            if self.advance_target_step is not None:
                                steps = min(steps, self.advance_target_step - self.engine.clocks.evolutionary_step)
                            self.engine.step_universe(steps=steps, dt=0.25)
                            if self.advance_target_step is not None and self.engine.clocks.evolutionary_step >= self.advance_target_step:
                                self.advance_target_step = None
                                self.is_running = False
                                self.playback_action = "pause"
                        elif self.playback_target == "tumor":
                            steps = 1 if self.speed_multiplier == 1 else (4 if self.speed_multiplier == 10 else 12)
                            self.engine.step_tumor(steps=steps, dt_seconds=900.0)
                        elif self.playback_target == "both":
                            u_steps = 1 if self.speed_multiplier == 1 else (5 if self.speed_multiplier == 10 else 20)
                            t_steps = 1 if self.speed_multiplier == 1 else (4 if self.speed_multiplier == 10 else 12)
                            self.engine.step_universe(steps=u_steps, dt=0.25)
                            self.engine.step_tumor(steps=t_steps, dt_seconds=900.0)

                    if self._subscribers and (not self.is_running or time.monotonic() - last_broadcast >= 0.1):
                        state = self.get_normalized_state()
                        await self.broadcast({"type": "state_update", "data": state})
                        last_broadcast = time.monotonic()

                elapsed = time.monotonic() - start
                sleep_time = max(0.005, frame_interval - elapsed)
                await asyncio.sleep(sleep_time)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in simulation loop: {e}", exc_info=True)
                await asyncio.sleep(0.1)

    async def register_websocket(self, websocket: WebSocket) -> None:
        self._subscribers.add(websocket)
        logger.info(f"WebSocket client connected. Active subscribers: {len(self._subscribers)}")
        async with self._lock:
            state = self.get_normalized_state()
        await websocket.send_json({"type": "init_state", "data": state})

    async def unregister_websocket(self, websocket: WebSocket) -> None:
        if websocket in self._subscribers:
            self._subscribers.remove(websocket)
            logger.info(f"WebSocket client disconnected. Active subscribers: {len(self._subscribers)}")

    async def broadcast(self, message: dict[str, Any]) -> None:
        dead_sockets = []
        for ws in list(self._subscribers):
            try:
                await ws.send_json(message)
            except Exception:
                dead_sockets.append(ws)
        for ws in dead_sockets:
            self._subscribers.discard(ws)

    async def handle_playback_command(
        self,
        action: Literal["play", "pause", "step", "speed"],
        speed_multiplier: int = 1,
        target: str = "universe",
    ) -> dict[str, Any]:
        async with self._lock:
            self.playback_target = target
            self.speed_multiplier = speed_multiplier
            if action == "play":
                self.playback_action = "play"
                self.is_running = True
            elif action == "pause":
                self.playback_action = "pause"
                self.is_running = False
            elif action == "speed":
                self.speed_multiplier = speed_multiplier
            elif action == "step":
                self.playback_action = "pause"
                self.is_running = False
                if target == "universe":
                    self.engine.step_universe(steps=1, dt=0.25)
                elif target == "tumor":
                    self.engine.step_tumor(steps=1, dt_seconds=900.0)
                elif target == "both":
                    self.engine.step_universe(steps=1, dt=0.25)
                    self.engine.step_tumor(steps=1, dt_seconds=900.0)

            state = self.get_normalized_state()

        await self.broadcast({
            "type": "playback_change",
            "action": action,
            "speed": self.speed_multiplier,
            "target": self.playback_target,
        })
        return {
            "action": self.playback_action,
            "speed_multiplier": self.speed_multiplier,
            "target": self.playback_target,
            "is_running": self.is_running,
            "state": state,
        }

    async def start_advance(self, preset: str) -> dict[str, Any]:
        generations = {"10_gen": 10, "100_gen": 100, "1000_gen": 1000, "10000_gen": 10000,
                       "1_yr": 10, "10_yr": 100, "100_yr": 1000, "1000_yr": 10000, "10000_yr": 100000}
        if preset not in generations:
            raise ValueError("Unknown time preset")
        async with self._lock:
            self.advance_start_step = self.engine.clocks.evolutionary_step
            self.advance_target_step = self.advance_start_step + generations[preset] * self.engine.clocks.steps_per_generation
            self.playback_target = "universe"
            self.speed_multiplier = 100
            self.playback_action = "play"
            self.is_running = True
        return self.get_normalized_state()

    def get_active_disease(self) -> str:
        return str(getattr(self.engine, "disease_id", "PDAC"))

    def set_active_disease(self, disease_id: str, seed: int | None = None) -> dict[str, Any]:
        if hasattr(self.engine, "set_active_disease"):
            return self.engine.set_active_disease(disease_id, seed=seed)

        self.engine.disease_id = disease_id
        use_seed = seed if seed is not None else getattr(self.engine, "seed", 42)
        if hasattr(self.engine, "disease_registry") and self.engine.disease_registry.is_registered(disease_id):
            cls = self.engine.disease_registry.get_class(disease_id)
            mod = cls(seed=use_seed)
            mod.initialize_scenario(seed=use_seed)
            self.engine.tumor_module = mod
        elif disease_id == "PDAC":
            self.engine.tumor_module = PDACModule(seed=use_seed)
            self.engine.tumor_module.initialize_scenario(seed=use_seed)

        return {
            "disease_id": disease_id,
            "status": "active",
            "metadata": self.engine.tumor_module.metadata() if hasattr(self.engine.tumor_module, "metadata") else {"display_name": disease_id},
        }

    def evaluate_combination(
        self,
        compound_a: str | None = None,
        compound_a_query: str | None = None,
        dose_a_um: float | None = None,
        conc_a_um: float | None = None,
        compound_b: str | None = None,
        compound_b_query: str | None = None,
        dose_b_um: float | None = None,
        conc_b_um: float | None = None,
        injection_region: str = "vascular",
        run_4_arm_trial: bool = True,
        duration_hours: float | None = None,
        hours: float | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        cA = compound_a_query or compound_a or "Gemcitabine"
        cB = compound_b_query or compound_b or "C1=CN=CN1"
        dA = conc_a_um if conc_a_um is not None else (dose_a_um if dose_a_um is not None else 5.0)
        dB = conc_b_um if conc_b_um is not None else (dose_b_um if dose_b_um is not None else 5.0)
        dur = duration_hours if duration_hours is not None else (hours if hours is not None else 48.0)

        res = self.engine.evaluate_combination(
            compound_a_query=cA, compound_b_query=cB, conc_a_um=dA, conc_b_um=dB,
            injection_region=injection_region, run_4_arm_trial=run_4_arm_trial,
            duration_hours=dur,
        )
        sp = res["synergy_profile"]
        res["bliss_excess_pct"] = sp["bliss_synergy_score"]["value"]
        res["loewe_ci"] = sp.get("loewe_combination_index")
        res["synergy_matrix"] = sp["matrix_2d"]["inhibition_matrix"]
        res["synergy_matrix_8x8"] = res["synergy_matrix"]
        res["trial_arms"] = res["trial_4_arm"]
        res["four_arm_trial"] = res["trial_4_arm"]
        return res

    def get_editor_operations(self) -> dict[str, Any]:
        if hasattr(self.engine, "get_editor_operations"):
            return self.engine.get_editor_operations()

        return {
            "substituents": [
                {"name": "methyl", "smiles": "C", "label": "+Methyl (-CH3)", "mw": 15.03, "group": "alkyl"},
                {"name": "fluoro", "smiles": "F", "label": "+Fluoro (-F)", "mw": 18.99, "group": "halogen"},
                {"name": "chloro", "smiles": "Cl", "label": "+Chloro (-Cl)", "mw": 35.45, "group": "halogen"},
                {"name": "hydroxyl", "smiles": "O", "label": "+Hydroxyl (-OH)", "mw": 17.01, "group": "polar"},
                {"name": "amino", "smiles": "N", "label": "+Amino (-NH2)", "mw": 16.02, "group": "polar"},
                {"name": "cf3", "smiles": "C(F)(F)F", "label": "+Trifluoromethyl (-CF3)", "mw": 69.01, "group": "lipophilic"},
                {"name": "cyano", "smiles": "C#N", "label": "+Cyano (-CN)", "mw": 26.02, "group": "polar"},
                {"name": "methoxy", "smiles": "OC", "label": "+Methoxy (-OCH3)", "mw": 31.03, "group": "ether"},
                {"name": "cyclopropyl", "smiles": "C1CC1", "label": "+Cyclopropyl", "mw": 41.07, "group": "carbocycle"},
                {"name": "imidazole", "smiles": "c1c[nH]cn1", "label": "+Imidazole", "mw": 67.07, "group": "heterocycle"},
                {"name": "morpholino", "smiles": "N1CCOCC1", "label": "+Morpholino", "mw": 86.11, "group": "heterocycle"},
            ],
            "atom_swaps": [
                {"name": "c_to_n", "label": "Carbon to Nitrogen (C -> N)", "description": "Bioisosteric core heteroatom insertion"},
                {"name": "o_to_s", "label": "Oxygen to Sulfur (O -> S)", "description": "Chalcogen isosteric swap"},
                {"name": "n_to_o", "label": "Nitrogen to Oxygen (N -> O)", "description": "Amine to ether transition"},
            ],
            "ring_transformations": [
                {"name": "add_pyridine", "label": "+Pyridine Ring"},
                {"name": "add_pyrimidine", "label": "+Pyrimidine Ring"},
                {"name": "add_cyclopropane", "label": "+Cyclopropane Ring"},
                {"name": "bioisostere_swap", "label": "Carboxylic Acid -> Tetrazole Swap"},
            ],
        }

    def edit_molecule(
        self,
        smiles: str,
        operation: str,
        param: str = "methyl",
        atom_idx: int | None = None,
        position: int | None = None,
        target_element: str | None = None,
        substituent: str | None = None,
        register_in_world: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        pos = position if position is not None else atom_idx
        p = substituent or param or target_element or "methyl"
        p_clean = p.lower()
        if p_clean in ("fluorine", "fluoro"):
            p = "fluoro"
        elif p_clean in ("chlorine", "chloro"):
            p = "chloro"
        elif p_clean in ("bromine", "bromo"):
            p = "bromo"
        elif p_clean in ("hydroxy", "hydroxyl"):
            p = "hydroxyl"
        elif p_clean in ("amine", "amino"):
            p = "amino"
        elif p_clean in ("cf3", "trifluoromethyl"):
            p = "trifluoromethyl"

        op = operation.lower()
        if op in ("substitute", "substitution", "attach_substituent", "attach_group"):
            op = "add_group"
        elif op in ("swap", "swap_atom"):
            op = "replace_atom"

        if hasattr(self.engine, "edit_molecule"):
            res = self.engine.edit_molecule(
                smiles=smiles,
                operation=op,
                param=p,
                atom_idx=pos,
                register_in_world=register_in_world,
            )
            if isinstance(res, dict):
                parent_svg = res.get("parent_svg") or (res.get("diff", {}).get("svg_a_highlighted") if isinstance(res.get("diff"), dict) else None) or res.get("svg_2d") or ""
                deriv_svg = res.get("derivative_svg") or res.get("edited_svg") or (res.get("diff", {}).get("svg_b_highlighted") if isinstance(res.get("diff"), dict) else None) or res.get("svg_annotated") or res.get("svg_2d") or ""
                res["parent_svg"] = parent_svg
                res["derivative_svg"] = deriv_svg
                res["edited_svg"] = deriv_svg

                diff_obj = res.get("diff") if isinstance(res.get("diff"), dict) else {}
                deltas = res.get("property_deltas") or res.get("deltas") or {
                    "delta_mw": diff_obj.get("delta_mw", 0.0),
                    "delta_logp": diff_obj.get("delta_logp", 0.0),
                    "delta_tpsa": diff_obj.get("delta_tpsa", 0.0),
                    "delta_qed": diff_obj.get("delta_qed", 0.0),
                }
                res["property_deltas"] = deltas
                res["deltas"] = deltas
                res["is_chemically_valid"] = res.get("valid", True)
            return res

        from rdkit import Chem
        from rdkit.Chem import Descriptors, Crippen, QED, rdMolDescriptors
        from rdkit.Chem.Draw import rdMolDraw2D

        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            raise ValueError(f"Invalid input SMILES: '{smiles}'")

        rw = Chem.RWMol(mol)
        op_clean = operation.lower()
        param_clean = param.lower()

        highlighted_atom = None

        if "swap" in op_clean or op_clean == "swap_atom":
            if param_clean == "c_to_n":
                for a in rw.GetAtoms():
                    if a.GetSymbol() == "C":
                        a.SetAtomicNum(7)
                        highlighted_atom = a.GetIdx()
                        break
            elif param_clean == "o_to_s":
                for a in rw.GetAtoms():
                    if a.GetSymbol() == "O":
                        a.SetAtomicNum(16)
                        highlighted_atom = a.GetIdx()
                        break
            elif param_clean == "n_to_o":
                for a in rw.GetAtoms():
                    if a.GetSymbol() == "N":
                        a.SetAtomicNum(8)
                        highlighted_atom = a.GetIdx()
                        break
        else:
            # Add substituent or ring
            sub_dict = {
                "methyl": "C", "fluoro": "F", "chloro": "Cl", "hydroxyl": "O",
                "amino": "N", "cf3": "C(F)(F)F", "cyano": "C#N", "methoxy": "OC",
                "cyclopropyl": "C1CC1", "imidazole": "c1c[nH]cn1", "morpholino": "N1CCOCC1",
                "add_pyridine": "c1ccncc1", "add_pyrimidine": "c1cncnc1",
            }
            sub_smi = sub_dict.get(param_clean, "C")
            sub_mol = Chem.MolFromSmiles(sub_smi)

            target_idx = atom_idx
            if target_idx is None or target_idx >= rw.GetNumAtoms():
                for a in rw.GetAtoms():
                    if a.GetSymbol() == "C" and a.GetTotalNumHs() > 0:
                        target_idx = a.GetIdx()
                        break

            if target_idx is not None and sub_mol:
                offset = rw.GetNumAtoms()
                for a in sub_mol.GetAtoms():
                    rw.AddAtom(Chem.Atom(a.GetAtomicNum()))
                for b in sub_mol.GetBonds():
                    rw.AddBond(b.GetBeginAtomIdx() + offset, b.GetEndAtomIdx() + offset, b.GetBondType())
                rw.AddBond(target_idx, offset, Chem.BondType.SINGLE)
                highlighted_atom = offset

        edited_mol = rw.GetMol()
        Chem.SanitizeMol(edited_mol)
        new_smiles = Chem.MolToSmiles(edited_mol)

        # Draw Parent SVG
        drawer_p = rdMolDraw2D.MolDraw2DSVG(280, 200)
        drawer_p.DrawMolecule(mol)
        drawer_p.FinishDrawing()
        parent_svg = drawer_p.GetDrawingText()

        # Draw Edited SVG with Coral/Cyan Highlight
        drawer_e = rdMolDraw2D.MolDraw2DSVG(280, 200)
        highlight_list = [highlighted_atom] if highlighted_atom is not None else []
        drawer_e.DrawMolecule(edited_mol, highlightAtoms=highlight_list)
        drawer_e.FinishDrawing()
        edited_svg = drawer_e.GetDrawingText()

        # Deltas
        mw_orig = round(float(Descriptors.MolWt(mol)), 2)
        mw_new = round(float(Descriptors.MolWt(edited_mol)), 2)
        logp_orig = round(float(Crippen.MolLogP(mol)), 2)
        logp_new = round(float(Crippen.MolLogP(edited_mol)), 2)
        tpsa_orig = round(float(rdMolDescriptors.CalcTPSA(mol)), 2)
        tpsa_new = round(float(rdMolDescriptors.CalcTPSA(edited_mol)), 2)
        qed_orig = round(float(QED.qed(mol)), 3)
        qed_new = round(float(QED.qed(edited_mol)), 3)

        registered_rec = None
        if register_in_world:
            registered_rec = self.engine.chemistry.build_compound(new_smiles, name=f"Variant-{param}")
            self.engine.compounds_registry[registered_rec.compound_id] = registered_rec
            self.engine.genealogy.register_compound(registered_rec, birth_step=self.engine.clocks.evolutionary_step)

        return {
            "parent_smiles": smiles,
            "edited_smiles": new_smiles,
            "operation": operation,
            "param": param,
            "parent_svg": parent_svg,
            "edited_svg": edited_svg,
            "parent_properties": {"mw": mw_orig, "logp": logp_orig, "tpsa": tpsa_orig, "qed": qed_orig},
            "edited_properties": {"mw": mw_new, "logp": logp_new, "tpsa": tpsa_new, "qed": qed_new},
            "deltas": {
                "delta_mw": round(mw_new - mw_orig, 2),
                "delta_logp": round(logp_new - logp_orig, 2),
                "delta_tpsa": round(tpsa_new - tpsa_orig, 2),
                "delta_qed": round(qed_new - qed_orig, 3),
            },
            "is_chemically_valid": True,
            "registered_in_world": bool(registered_rec),
            "compound_id": registered_rec.compound_id if registered_rec else None,
        }

    def diff_molecules(
        self,
        query_a: str | None = None,
        query_b: str | None = None,
        smiles_a: str | None = None,
        smiles_b: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        qa = query_a or smiles_a or ""
        qb = query_b or smiles_b or ""
        if hasattr(self.engine, "diff_molecules"):
            diff_res = self.engine.diff_molecules(query_a=qa, query_b=qb)
            if isinstance(diff_res, dict):
                svg_a = diff_res.get("svg_a") or diff_res.get("svg_a_highlighted") or ""
                svg_b = diff_res.get("svg_b") or diff_res.get("svg_b_highlighted") or ""
                diff_res["svg_a"] = svg_a
                diff_res["svg_b"] = svg_b
                diff_res["svg_a_highlighted"] = svg_a
                diff_res["svg_b_highlighted"] = svg_b
                if "deltas" not in diff_res:
                    diff_res["deltas"] = {
                        "delta_mw": diff_res.get("delta_mw", 0.0),
                        "delta_logp": diff_res.get("delta_logp", 0.0),
                        "delta_tpsa": diff_res.get("delta_tpsa", 0.0),
                        "delta_qed": diff_res.get("delta_qed", 0.0),
                    }
                if "property_deltas" not in diff_res:
                    diff_res["property_deltas"] = diff_res["deltas"]
            return diff_res

        from rdkit import Chem
        from rdkit.Chem import Descriptors, Crippen, QED, rdFMCS, rdMolDescriptors, DataStructs
        from rdkit.Chem.Draw import rdMolDraw2D

        rec_a = self.engine.chemistry.resolve_compound_input(query_a)
        rec_b = self.engine.chemistry.resolve_compound_input(query_b)

        mol_a = Chem.MolFromSmiles(rec_a.canonical_smiles)
        mol_b = Chem.MolFromSmiles(rec_b.canonical_smiles)

        # Compute MCS
        mcs = rdFMCS.FindMCS([mol_a, mol_b], timeout=3)
        mcs_mol = Chem.MolFromSmarts(mcs.smartsString)

        match_a = mol_a.GetSubstructMatch(mcs_mol) if mcs_mol else ()
        match_b = mol_b.GetSubstructMatch(mcs_mol) if mcs_mol else ()

        diff_atoms_a = [i for i in range(mol_a.GetNumAtoms()) if i not in match_a]
        diff_atoms_b = [i for i in range(mol_b.GetNumAtoms()) if i not in match_b]

        # Draw SVGs highlighting different atoms
        drawer_a = rdMolDraw2D.MolDraw2DSVG(280, 200)
        drawer_a.DrawMolecule(mol_a, highlightAtoms=diff_atoms_a)
        drawer_a.FinishDrawing()
        svg_a = drawer_a.GetDrawingText()

        drawer_b = rdMolDraw2D.MolDraw2DSVG(280, 200)
        drawer_b.DrawMolecule(mol_b, highlightAtoms=diff_atoms_b)
        drawer_b.FinishDrawing()
        svg_b = drawer_b.GetDrawingText()

        # Tanimoto Similarity
        fpgen = self.engine.chemistry._fpgen
        fp_a = fpgen.GetFingerprint(mol_a)
        fp_b = fpgen.GetFingerprint(mol_b)
        similarity = round(float(DataStructs.TanimotoSimilarity(fp_a, fp_b)), 4)

        return {
            "compound_a": rec_a.to_dict(),
            "compound_b": rec_b.to_dict(),
            "tanimoto_similarity": similarity,
            "mcs_smarts": mcs.smartsString,
            "svg_a_highlighted": svg_a,
            "svg_b_highlighted": svg_b,
            "deltas": {
                "delta_mw": round(rec_b.mol_weight - rec_a.mol_weight, 2),
                "delta_logp": round(rec_b.logp - rec_a.logp, 2),
                "delta_tpsa": round(rec_b.tpsa - rec_a.tpsa, 2),
                "delta_qed": round(rec_b.qed - rec_a.qed, 3),
            },
        }

    def get_ai_runtime_status(self) -> dict[str, Any]:
        if hasattr(self.engine, "get_ai_runtime_status"):
            return self.engine.get_ai_runtime_status()

        reg = get_ai_registry()
        hw = reg.get_hardware_telemetry() if hasattr(reg, "get_hardware_telemetry") else {}
        base_dir = pathlib.Path(__file__).resolve().parent.parent.parent
        gemma_path = base_dir / "models" / "gemma" / "Gemma-4-E2B-Heretic-Q4_K_M.gguf"
        gemma_runtime = base_dir / "models" / "gemma" / "llama_runtime" / "llama.exe"
        laya_path = base_dir / "models" / "laya" / "multilingual" / "model.safetensors"
        qsar_path = base_dir / "models" / "scientific" / "pdac_qsar_v1.joblib"

        assistant = getattr(self.engine, "assistant", getattr(self.engine, "gemma", None))
        is_server = assistant.is_local_server_running() if assistant and hasattr(assistant, "is_local_server_running") else False

        return {
            "hardware_telemetry": hw,
            "models": {
                "gemma_4_llm": {
                    "name": "Gemma-4-E2B-Heretic-Q4_K_M",
                    "present": gemma_path.exists(),
                    "size_mb": round(gemma_path.stat().st_size / (1024*1024), 1) if gemma_path.exists() else 0,
                    "runtime_exe_present": gemma_runtime.exists(),
                    "is_server_running": is_server,
                },
                "laya_system_1": {
                    "name": "Laya Multilingual Neural System-1",
                    "present": laya_path.exists(),
                    "status": "ready_offline",
                },
                "qsar_predictor": {
                    "name": "Calibrated Random Forest QSAR (PANC-1 / MIA PaCa-2)",
                    "present": qsar_path.exists(),
                    "status": "active_calibrated",
                },
            },
            "epistemic_policy": "Section 27 Compliant: Deterministic scientific guardrails active; zero silent model hallucination.",
        }

    def triage_current_compounds(self) -> dict[str, Any]:
        if hasattr(self.engine, "triage_current_compounds"):
            res = self.engine.triage_current_compounds()
            if isinstance(res, dict):
                if "ranked_compounds" not in res and "rankings" in res:
                    res["ranked_compounds"] = res["rankings"]
                if "rankings" not in res and "ranked_compounds" in res:
                    res["rankings"] = res["ranked_compounds"]
            return res

        compounds = getattr(self.engine, "compounds", getattr(self.engine, "compounds_registry", {}))
        ranked = []
        for cid, rec in compounds.items():
            admet = self.engine.pharmacology.evaluate_admet(rec)
            tox = self.engine.toxicology.evaluate_toxicology(rec)

            ic50 = admet.get("panc1_ic50_um", {}).get("value") or 50.0
            potency_score = 1.0 / (1.0 + float(ic50))
            sel_index = admet.get("selectivity_index", {}).get("value") or 1.0
            sel_score = min(1.0, float(sel_index) / 10.0)
            qed_score = float(rec.qed)
            tox_risk = tox.get("overall_risk_index", {}).get("value") or 0.3
            safety_score = max(0.0, 1.0 - float(tox_risk))

            utility = round(0.35 * potency_score + 0.25 * sel_score + 0.20 * qed_score + 0.20 * safety_score, 3)
            verdict = "PRIORITY_CANDIDATE" if utility >= 0.65 else ("VIABLE_LEAD" if utility >= 0.45 else ("MONITOR" if utility >= 0.25 else "DEPRIORITIZE"))

            ranked.append({
                "compound_id": cid,
                "name": rec.name,
                "canonical_smiles": rec.canonical_smiles,
                "utility_score": utility,
                "recommendation": verdict,
                "metrics": {
                    "potency_score": round(potency_score, 3),
                    "selectivity_score": round(sel_score, 3),
                    "qed_score": round(qed_score, 3),
                    "safety_score": round(safety_score, 3),
                },
                "admet_summary": admet,
                "toxicology_summary": tox,
                "epistemic_category": EpistemicCategory.AI_HYPOTHESIS.value,
            })

        ranked.sort(key=lambda x: x["utility_score"], reverse=True)
        return {
            "ranked_compounds": ranked,
            "total_evaluated": len(ranked),
            "criteria_weights": {"potency": 0.35, "selectivity": 0.25, "qed": 0.20, "safety": 0.20},
            "epistemic_notice": "Algorithmic multi-criteria triage prioritizing in silico hypotheses for wet-lab confirmation.",
        }

    def export_pdf(self, exp_id: str) -> bytes:
        if hasattr(self.engine, "store") and hasattr(self.engine.store, "export_pdf"):
            return self.engine.store.export_pdf(exp_id)

        # Fallback simple PDF
        content = f"%PDF-1.3\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>\nendobj\n4 0 obj\n<< /Length 55 >>\nstream\nBT /F1 12 Tf 72 712 Td (EVOCURE Simulation Report - {exp_id}) Tj ET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000214 00000 n \ntrailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n318\n%%EOF\n"
        return content.encode("utf-8")

    def apply_environment_preset(self, preset: str) -> dict[str, Any]:
        """Apply one of the 6 validated biophysical environment presets."""
        presets = {
            "pancreatic_tissue": {
                "temperature_k": 310.15,
                "ph": 7.35,
                "viscosity_pa_s": 0.0012,
                "oxygen_norm": 0.80,
                "resource_influx_rate": 0.50,
                "energy_source_flux": 1.0,
                "current_vx": 0.05,
                "current_vy": 0.0,
            },
            "bloodstream": {
                "temperature_k": 310.15,
                "ph": 7.40,
                "viscosity_pa_s": 0.0035,
                "oxygen_norm": 1.00,
                "resource_influx_rate": 0.80,
                "energy_source_flux": 1.2,
                "current_vx": 0.25,
                "current_vy": 0.0,
            },
            "tumor_microenvironment": {
                "temperature_k": 310.65,
                "ph": 6.75,
                "viscosity_pa_s": 0.0020,
                "oxygen_norm": 0.35,
                "resource_influx_rate": 0.30,
                "energy_source_flux": 0.8,
                "current_vx": 0.02,
                "current_vy": 0.0,
            },
            "oxidative_stress": {
                "temperature_k": 311.15,
                "ph": 6.90,
                "viscosity_pa_s": 0.0010,
                "oxygen_norm": 1.60,
                "resource_influx_rate": 0.40,
                "energy_source_flux": 1.8,
                "current_vx": 0.05,
                "current_vy": 0.0,
            },
            "acidic_low_ph": {
                "temperature_k": 310.15,
                "ph": 6.20,
                "viscosity_pa_s": 0.0015,
                "oxygen_norm": 0.50,
                "resource_influx_rate": 0.30,
                "energy_source_flux": 0.9,
                "current_vx": 0.05,
                "current_vy": 0.0,
            },
            "hypoxia_low_o2": {
                "temperature_k": 310.15,
                "ph": 6.80,
                "viscosity_pa_s": 0.0018,
                "oxygen_norm": 0.10,
                "resource_influx_rate": 0.20,
                "energy_source_flux": 0.7,
                "current_vx": 0.05,
                "current_vy": 0.0,
            },
        }

        if preset not in presets:
            raise ValueError(f"Unknown preset '{preset}'. Valid presets: {list(presets.keys())}")

        vals = presets[preset]
        env = getattr(self.engine, "environment", None)
        if env is not None:
            for k, v in vals.items():
                if hasattr(env, k):
                    setattr(env, k, v)

        tumor = getattr(self.engine, "tumor", getattr(self.engine, "tumor_module", None))
        if tumor is not None:
            if hasattr(tumor, "ph"):
                tumor.ph = vals["ph"]
            if hasattr(tumor, "oxygen_norm"):
                tumor.oxygen_norm = vals["oxygen_norm"]

        return {"preset": preset, "environment": vals}

    def get_entity_lineage(self, compound_id: str) -> dict[str, Any]:
        """
        Return the ancestor lineage chain [G0..Gk] with modular parts and visual genome.
        """
        genealogy = getattr(self.engine, "genealogy", None)
        chain: list[dict[str, Any]] = []
        if genealogy and hasattr(genealogy, "trace_lineage"):
            chain = genealogy.trace_lineage(compound_id)

        if not chain:
            compounds = getattr(self.engine, "compounds", getattr(self.engine, "compounds_registry", {}))
            rec = compounds.get(compound_id)
            if rec:
                chain = [{
                    "node_id": f"GN-{compound_id}",
                    "compound_id": compound_id,
                    "name": rec.name or rec.common_name or compound_id,
                    "canonical_smiles": rec.canonical_smiles or "",
                    "family_id": rec.family_id or "FAM-ORIGIN",
                    "family_name": rec.family_name or "Founder Family",
                    "generation": rec.origin_generation,
                    "parent_ids": list(rec.parent_ids),
                    "transformation_type": "founder",
                    "mutation_description": "Founder compound",
                    "similarity_to_parent": 1.0,
                    "birth_step": 0,
                    "extinction_step": None,
                    "current_abundance": 1,
                    "peak_abundance": 1,
                    "cumulative_births": 1,
                    "persistence_steps": 1,
                    "properties_summary": {
                        "mol_weight": float(getattr(rec, "mol_weight", 0.0)),
                        "logp": float(getattr(rec, "logp", 0.0)),
                        "qed": float(getattr(rec, "qed", 0.0)),
                    },
                }]
            else:
                chain = [{
                    "node_id": f"GN-{compound_id}",
                    "compound_id": compound_id,
                    "name": compound_id,
                    "canonical_smiles": "",
                    "family_id": "FAM-ORIGIN",
                    "family_name": "Founder Lineage",
                    "generation": 0,
                    "parent_ids": [],
                    "transformation_type": "founder",
                    "mutation_description": "System founder",
                    "similarity_to_parent": 1.0,
                    "birth_step": 0,
                    "extinction_step": None,
                    "current_abundance": 1,
                    "peak_abundance": 1,
                    "cumulative_births": 1,
                    "persistence_steps": 1,
                    "properties_summary": {},
                }]

        enriched_chain = []
        for node in chain:
            enriched_node = dict(node)
            cid = str(node.get("compound_id", ""))
            seed_val = abs(hash(cid))
            smi = str(node.get("canonical_smiles", ""))
            name_lower = str(node.get("name", "")).lower()

            if "c1ccccc1" in smi.lower() or "benzene" in name_lower:
                archetype = "benzene"
            elif "p" in smi.lower() or "phosphate" in name_lower:
                archetype = "phosphate"
            elif "glucose" in name_lower:
                archetype = "glucose"
            else:
                archetype = "modular"

            enriched_node["visual_genome"] = {
                "archetype": archetype,
                "head_id": seed_val % 12,
                "body_id": (seed_val // 12) % 12,
                "legs_id": (seed_val // 144) % 8,
                "appendage_id": (seed_val // 1152) % 8,
                "crystal_id": (seed_val // 9216) % 6,
                "glow_color": "#00f5d4",
            }
            enriched_chain.append(enriched_node)

        return {
            "compound_id": compound_id,
            "lineage_depth": len(enriched_chain),
            "ancestors": enriched_chain,
        }

    def get_ai_providers_status(self) -> dict[str, Any]:
        assistant = getattr(self.engine, "assistant", getattr(self.engine, "gemma", None))
        if assistant and hasattr(assistant, "get_providers_status"):
            return assistant.get_providers_status()
        return {"active_provider": "deterministic", "providers": {}}

    def configure_ai_provider(
        self,
        provider: str,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        persist_to_dotenv: bool = False,
    ) -> dict[str, Any]:
        assistant = getattr(self.engine, "assistant", getattr(self.engine, "gemma", None))
        if assistant and hasattr(assistant, "configure_provider"):
            res = assistant.configure_provider(
                provider=provider,
                model=model,
                api_key=api_key,
                base_url=base_url,
                persist_to_dotenv=persist_to_dotenv,
            )
            reg = get_ai_registry()
            if reg:
                if provider == "gemini":
                    reg.update_model_status("google_gemini_api", status="configured" if api_key else "optional_unconfigured")
                elif provider == "openai":
                    reg.update_model_status("openai_api", status="configured" if api_key else "optional_unconfigured")
            return res
        return {"error": "Assistant does not support provider configuration"}

    def test_ai_provider_connection(
        self,
        provider: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> dict[str, Any]:
        assistant = getattr(self.engine, "assistant", getattr(self.engine, "gemma", None))
        if assistant and hasattr(assistant, "test_provider_connection"):
            return assistant.test_provider_connection(
                provider=provider,
                api_key=api_key,
                model=model,
                base_url=base_url,
            )
        return {"success": False, "error": "Assistant connection tester unavailable"}

    def ask_assistant(
        self,
        question: str,
        provider: str | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        if hasattr(self.engine, "ask_assistant"):
            try:
                return self.engine.ask_assistant(question, provider=provider, model=model)
            except TypeError:
                return self.engine.ask_assistant(question)
        assistant = getattr(self.engine, "assistant", getattr(self.engine, "gemma", None))
        if assistant:
            return assistant.ask(
                question=question,
                simulation_context=self.get_normalized_state(),
                provider_override=provider,
                model_override=model,
            )
        return {"answer": "Assistant offline", "provider": "deterministic"}


def get_session_manager() -> SessionManager:
    return SessionManager.get_instance()
