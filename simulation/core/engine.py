"""
EVOCURE Unified Simulation Engine Core.

Integrates all domain engines into a single deterministic simulation backend
powering both EVOCURE PLAY and EVOCURE LAB:
- 3 Separated Clocks (Computational, Evolutionary, Biological)
- 2D Spatial Physics, Collision Detection, & Eulerian Reaction-Diffusion Grid
- Stoichiometric Chemistry Engine & Emergent Molecular Reactions
- Darwinian Artificial Chemical Life & Lineage Genealogy Tracker
- Mechanistic PDAC Tumor Microenvironment & Hypoxic/Desmoplastic Reaction-Diffusion
- In Silico ADMET & 10-Category Multi-Organ Toxicology
- Whole-Body Systemic PBPK & Calibrated Biological Indicators
- Calibrated QSAR Predictor (pIC50 with 95% Confidence Intervals)
- Local Gemma AI Assistant & Laya Decision Triage Engine
- SQLite Persistence Store, Checkpoints, & Multi-Format Exporters
- 15-Stage Candidate Discovery Pipeline
- Compare Worlds Statistical Evaluator (mean +/- std across repeated seeds)
"""

from __future__ import annotations

import copy
import json
import math
import os
import time
import uuid
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from rdkit import RDLogger

# Silence noisy RDKit C++ stderr parser messages
RDLogger.DisableLog("rdApp.*")

from ai.laya import LayaDecisionEngine
from ai.local_llm import LocalGemmaAssistant
from ai.orchestration import ModelRegistry, get_ai_registry
from ai.predictors import CalibratedQSARPredictor, ScientificDataManager
from simulation.biology.coevolution import CoevolutionEngine
from simulation.biology.diseases.pdac import PDACModule
from simulation.biology.registry import DiseaseRegistry
from simulation.chemistry.editor import MolecularStructureEditor
from simulation.chemistry.engine import ChemistryEngine
from simulation.core.contracts import (
    CellularEntity,
    CellType,
    CellDeathMode,
    CompartmentBarrier,
    SpatialRegion,
    GenealogyNode,
    CompoundRecord,
    EntityKind,
    EnvironmentConfig,
    EpistemicCategory,
    InterfaceMode,
    ScientificValue,
    SimulationClocks,
    SpatialEntity,
    UniverseMode,
)
from simulation.core.persistence import ExperimentStore
from simulation.discovery.pipeline import CandidateDiscoveryPipeline
from simulation.evolution.darwinian import DarwinianEcosystemEngine
from simulation.evolution.genealogy import GenealogyTracker
from simulation.evolution.molecular_ga import DirectedMolecularOptimizer
from simulation.pharmacology.admet import (
    PharmacologyEngine,
    validate_for_biomedical_mode,
)
from simulation.physics.spatial import (
    PhysicsEngine2D,
    ReactionDiffusionGrid2D,
)
from simulation.systemic.biomarkers import BiologicalIndicatorsEngine
from simulation.systemic.pbpk import SystemicPBPKEngine
from simulation.toxicity.models import ToxicologyEngine
from simulation.core.paths import configs_dir

CONFIGS_DIR: Path = configs_dir()
DEFAULT_UNIVERSE_CONFIG: Path = CONFIGS_DIR / "default_universe.json"
MISSIONS_CONFIG: Path = CONFIGS_DIR / "missions_and_glossary.json"


class SubstrateSourceAdapter:
    """Adapts ReactionDiffusionGrid2D to DarwinianEcosystemEngine consumption protocol."""

    def __init__(self, grid: ReactionDiffusionGrid2D, field_name: str = "substrate") -> None:
        self.grid = grid
        self.field_name = field_name

    def __call__(self, x: float, y: float) -> float:
        return self.grid.sample(self.field_name, x, y)

    def deduct(self, x: float, y: float, amount: float) -> float:
        return self.grid.consume_at(self.field_name, x, y, amount)


class EVOCUREEngine:
    """
    Central, deterministic simulation engine powering EVOCURE PLAY and EVOCURE LAB.
    Seamlessly manages physical particle dynamics, biochemical reactions, artificial life
    evolution, tumor pharmacology, AI assistance, and persistent experiment storage.
    """

    def __init__(
        self,
        seed: int = 20261026,
        universe_mode: UniverseMode | str = UniverseMode.ARTIFICIAL_LIFE,
        interface_mode: InterfaceMode | str = InterfaceMode.PLAY,
        disease_id: str = "PDAC",
        db_path: str | Path | None = None,
    ) -> None:
        self.seed: int = int(seed)
        self.universe_mode: UniverseMode = (
            UniverseMode(universe_mode) if isinstance(universe_mode, str) else universe_mode
        )
        self.interface_mode: InterfaceMode = (
            InterfaceMode(interface_mode) if isinstance(interface_mode, str) else interface_mode
        )
        self.disease_id: str = disease_id
        self.rng: np.random.Generator = np.random.Generator(np.random.PCG64(self.seed))

        # 1. Three Strictly Separated Clocks
        self.clocks: SimulationClocks = SimulationClocks(
            steps_per_generation=25,
            generations_per_virtual_year=10.0,
            calibration_note="Virtual years represent abstract evolutionary cycles in Mode A; biological time represents mechanistic PK/PD in Mode B.",
        )

        # 2. Environment Configuration
        self.env: EnvironmentConfig = self._load_default_environment()

        # 3. Continuous 2D Reaction-Diffusion Grid
        self.grid: ReactionDiffusionGrid2D = ReactionDiffusionGrid2D(
            width=self.env.width,
            height=self.env.height,
            nx=50,
            ny=35,
            default_fields={"substrate": 12.0, "oxygen": 1.0, "waste": 0.0},
        )
        self.substrate_adapter = SubstrateSourceAdapter(self.grid, "substrate")

        # 4. Core Registry & Engines
        self.compounds: dict[str, CompoundRecord] = {}
        self.entities: list[SpatialEntity] = []
        self.genealogy: GenealogyTracker = GenealogyTracker()
        self.chemistry: ChemistryEngine = ChemistryEngine()
        self.physics: PhysicsEngine2D = PhysicsEngine2D()
        self.mol_optimizer: DirectedMolecularOptimizer = DirectedMolecularOptimizer()
        self.mol_editor: MolecularStructureEditor = MolecularStructureEditor()

        # Darwinian Ecosystem Engine (with structural mutation callback)
        self.darwinian: DarwinianEcosystemEngine = DarwinianEcosystemEngine(
            genealogy_tracker=self.genealogy,
            mutation_rate=0.08,
            mutate_structure_fn=self._mutate_compound_callback,
            compounds_registry=self.compounds,
        )

        # 5. Disease Module & Pharmacology
        self.disease_registry: DiseaseRegistry = DiseaseRegistry()
        self.tumor_module = self.disease_registry.create(self.disease_id, seed=self.seed, nx=36, ny=28)
        self.pharmacology: PharmacologyEngine = PharmacologyEngine()
        self.toxicology: ToxicologyEngine = ToxicologyEngine()
        self.pbpk: SystemicPBPKEngine = SystemicPBPKEngine()
        self.biomarkers_engine: BiologicalIndicatorsEngine = BiologicalIndicatorsEngine()
        self.data_manager: ScientificDataManager = ScientificDataManager()

        # 6. AI Predictors & Local Intelligence
        try:
            self.qsar: CalibratedQSARPredictor = CalibratedQSARPredictor().load()
        except Exception:
            self.qsar = CalibratedQSARPredictor()

        self.assistant: LocalGemmaAssistant = LocalGemmaAssistant()
        self.laya: LayaDecisionEngine = LayaDecisionEngine()
        self.model_registry: ModelRegistry = get_ai_registry()

        # 7. Persistence & Discovery Pipeline
        self.store: ExperimentStore = ExperimentStore(db_path=db_path)
        self.discovery_pipeline: CandidateDiscoveryPipeline = CandidateDiscoveryPipeline(
            disease_id=self.disease_id,
            chem_engine=self.chemistry,
            qsar_predictor=self.qsar,
            pharmacology_engine=self.pharmacology,
            toxicology_engine=self.toxicology,
            decision_engine=self.laya,
            disease_registry=self.disease_registry,
            molecular_optimizer=self.mol_optimizer,
        )

        # 8. Missions, Achievements, & History
        self.missions: list[dict[str, Any]] = []
        self.achievements: list[dict[str, Any]] = []
        self.unlocked_achievements: set[str] = set()
        self.mission_progress: dict[str, Any] = {
            "entities_added": 0,
            "families_competing": 0,
            "evolutionary_generation": 0,
            "env_shift_survived": 0,
            "tumor_steps_observed": 0,
            "controlled_comparison_run": 0,
            "resistance_observed": 0,
            "pareto_evaluation_completed": 0,
        }
        self._load_missions_and_achievements()

        # History tracking & period summary metrics
        self.history: list[dict[str, Any]] = []
        self.reactions_count: int = 0
        self.recent_events: list[dict[str, Any]] = []
        self.active_experiment_id: str = f"EXP-{datetime_tag()}-{uuid.uuid4().hex[:8].upper()}"

        # Deterministic entity counter & reaction cache for high performance
        self._entity_counter: int = 0
        self._reaction_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        self.discovery_results: dict[str, Any] = {}
        self.tissue_events: list[dict[str, Any]] = []
        self.tissue_model_info: dict[str, Any] = {}

        # Initialize baseline world
        self.reset_world(preset="chemical_life", seed=self.seed)

    def _load_default_environment(self) -> EnvironmentConfig:
        """Load default environment parameters from JSON configuration if available."""
        if DEFAULT_UNIVERSE_CONFIG.exists():
            try:
                with open(DEFAULT_UNIVERSE_CONFIG, "r", encoding="utf-8") as f:
                    data = json.load(f)
                env_dict = data.get("environment", {})
                return EnvironmentConfig(
                    width=float(env_dict.get("width", 1000.0)),
                    height=float(env_dict.get("height", 700.0)),
                    temperature_k=float(env_dict.get("temperature_k", 310.15)),
                    ph=float(env_dict.get("ph", 7.4)),
                    viscosity_pa_s=float(env_dict.get("viscosity_pa_s", 0.00089)),
                    oxygen_norm=float(env_dict.get("oxygen_norm", 1.0)),
                    resource_influx_rate=float(env_dict.get("resource_influx_rate", 0.4)),
                    resource_clearance_rate=float(env_dict.get("resource_clearance_rate", 0.01)),
                    current_vx=float(env_dict.get("current_vx", 0.05)),
                    current_vy=float(env_dict.get("current_vy", 0.0)),
                )
            except Exception:
                pass
        return EnvironmentConfig()

    def _load_missions_and_achievements(self) -> None:
        """Load gamified missions and educational glossary items."""
        if MISSIONS_CONFIG.exists():
            try:
                with open(MISSIONS_CONFIG, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.missions = data.get("missions", [])
                self.achievements = data.get("achievements", [])
            except Exception:
                pass

    def _mutate_compound_callback(
        self, parent_comp: CompoundRecord, rng: np.random.Generator
    ) -> CompoundRecord:
        """Produces a chemically valid RDKit structural variant on heritable division."""
        smiles = parent_comp.canonical_smiles
        mut_smiles, desc = self.mol_optimizer.mutate_smiles(smiles, rng=rng)

        # Fast cache lookup: reuse existing CompoundRecord if structure already registered
        for c in self.compounds.values():
            if c.canonical_smiles == mut_smiles:
                return c

        new_comp = self.chemistry.build_compound(
            smiles=mut_smiles,
            origin_generation=self.clocks.evolutionary_generation,
            parent_ids=[parent_comp.compound_id],
            transformation_type="molecular_mutation",
            provenance="Darwinian Heritable Variation",
        )
        self.compounds[new_comp.compound_id] = new_comp
        return new_comp

    # -------------------------------------------------------------------------
    # Mode & Environment Controls
    # -------------------------------------------------------------------------

    def set_interface_mode(self, mode: InterfaceMode | str) -> dict[str, Any]:
        """Instant switch between EVOCURE PLAY and EVOCURE LAB with zero parameter disruption."""
        self.interface_mode = InterfaceMode(mode) if isinstance(mode, str) else mode
        return {
            "interface_mode": self.interface_mode.value,
            "status": "active",
            "epistemic_note": "Switched interface mode; underlying simulation state and RNG sequence preserved.",
        }

    def set_universe_mode(self, mode: UniverseMode | str) -> dict[str, Any]:
        """Switch between Mode A (Artificial Chemical Life) and Mode B (Grounded Chemistry)."""
        prev_mode = self.universe_mode
        self.universe_mode = UniverseMode(mode) if isinstance(mode, str) else mode
        note = (
            "Mode B active. Fitness parameters from Mode A are strictly quarantined from pharmacology."
            if self.universe_mode == UniverseMode.GROUNDED_CHEMISTRY
            else "Mode A active. Emergent artificial chemical replicators evolving in continuous 2D space."
        )
        return {
            "previous_universe_mode": prev_mode.value,
            "universe_mode": self.universe_mode.value,
            "status": "active",
            "epistemic_note": note,
        }

    def set_active_disease(self, disease_id: str, seed: int | None = None) -> dict[str, Any]:
        """
        Dynamically switch the active 2D cellular disease microenvironment.
        Supported modules: 'PDAC', 'NSCLC_KRAS', 'AMR_BACTERIAL', 'METABOLIC_MASLD'.
        """
        d_seed = self.seed if seed is None else int(seed)
        self.tumor_module = self.disease_registry.create(disease_id, seed=d_seed, nx=36, ny=28)
        self.disease_id = disease_id
        self.discovery_pipeline.disease_id = disease_id
        self.clocks.biological_seconds = 0.0
        self.tissue_events = []
        self.tissue_model_info = {}
        return {
            "disease_id": self.disease_id,
            "display_name": self.tumor_module.display_name,
            "metadata": self.tumor_module.metadata(),
            "tumor": self.tumor_module.get_snapshot(),
        }

    def update_environment(self, patch: dict[str, Any]) -> dict[str, Any]:
        """Update physical environment parameters (temperature, pH, viscosity, flow)."""
        prev_temp = self.env.temperature_k
        prev_ph = self.env.ph

        for k, v in patch.items():
            if hasattr(self.env, k):
                setattr(self.env, k, float(v) if isinstance(v, (int, float)) else v)

        # Mission 4 Check: environmental survival
        if abs(self.env.temperature_k - prev_temp) > 5.0 or abs(self.env.ph - prev_ph) > 0.5:
            self.mission_progress["env_shift_survived"] += 1
            self._check_achievement("ACH-EXTREMOPHILE")

        return {
            "status": "updated",
            "environment": self.env.to_dict(),
        }

    # -------------------------------------------------------------------------
    # World Reset & Presets
    # -------------------------------------------------------------------------

    def reset_world(
        self,
        preset: str = "chemical_life",
        seed: int | None = None,
        universe_mode: str | None = None,
    ) -> dict[str, Any]:
        """
        Reset simulation world to a predefined or randomized scenario.
        Presets: 'chemical_life', 'cancer_battle', 'evolve_compounds', 'surprise_me', 'sandbox_empty'.
        """
        if seed is not None:
            self.seed = int(seed)
        self.rng = np.random.Generator(np.random.PCG64(self.seed))

        if universe_mode:
            self.universe_mode = UniverseMode(universe_mode)

        # Reset Clocks & State
        self.clocks = SimulationClocks(
            steps_per_generation=25,
            generations_per_virtual_year=10.0,
        )
        self.entities.clear()
        self.compounds.clear()
        self.genealogy = GenealogyTracker()
        self.darwinian = DarwinianEcosystemEngine(
            genealogy_tracker=self.genealogy,
            mutation_rate=0.08,
            mutate_structure_fn=self._mutate_compound_callback,
            compounds_registry=self.compounds,
        )
        self.history.clear()
        self.reactions_count = 0
        self.recent_events.clear()
        self._entity_counter = 0
        self._reaction_cache.clear()
        self.discovery_results = {}
        self.tissue_events = []
        self.tissue_model_info = {}
        self.active_experiment_id = f"EXP-{datetime_tag()}-{uuid.uuid4().hex[:8].upper()}"

        # Grid re-initialization
        self.grid = ReactionDiffusionGrid2D(
            width=self.env.width,
            height=self.env.height,
            nx=50,
            ny=35,
            default_fields={"substrate": 14.0, "oxygen": 1.0, "waste": 0.0},
        )
        self.substrate_adapter = SubstrateSourceAdapter(self.grid, "substrate")

        scenario_name = preset
        if preset == "chemical_life":
            self.universe_mode = UniverseMode.ARTIFICIAL_LIFE
            self.add_compound_to_world("Glycine", count=24, ecological_role="autotroph")
            self.add_compound_to_world("Pyruvic Acid", count=20, ecological_role="autotroph")
            self.add_compound_to_world("Imidazole", count=14, ecological_role="catalyst")
            self.add_compound_to_world("Adenine", count=16, ecological_role="cooperator")
            self.add_compound_to_world("Formaldehyde", count=32, ecological_role="nutrient", entity_kind="NUTRIENT")
            scenario_name = "Primordial Chemical Soup (Mode A)"

        elif preset == "cancer_battle":
            self.universe_mode = UniverseMode.GROUNDED_CHEMISTRY
            self.interface_mode = InterfaceMode.LAB
            self.tumor_module = self.disease_registry.create(self.disease_id, seed=self.seed, nx=36, ny=28)
            # Pre-register reference therapeutic
            gem = self.chemistry.resolve_compound_input("Gemcitabine")
            self.compounds[gem.compound_id] = gem
            self.genealogy.register_compound(gem, birth_step=0, mutation_description="Standard-of-care reference")
            scenario_name = f"Virtual {self.tumor_module.display_name} Microenvironment (Mode B)"

        elif preset == "evolve_compounds":
            self.universe_mode = UniverseMode.GROUNDED_CHEMISTRY
            self.interface_mode = InterfaceMode.LAB
            for ref_name in ["Gemcitabine", "5-Fluorouracil", "Erlotinib", "Aspirin"]:
                self.add_compound_to_world(ref_name, count=6, entity_kind="THERAPEUTIC", ecological_role="neutral")
            scenario_name = "Directed Oncology Molecular Evolution"

        elif preset == "surprise_me":
            # Procedural, scientifically valid randomized scenario with logged seed
            rand_temp = round(float(self.rng.uniform(290.0, 335.0)), 1)
            rand_ph = round(float(self.rng.uniform(6.2, 8.2)), 2)
            self.env.temperature_k = rand_temp
            self.env.ph = rand_ph

            library_candidates = ["Glycine", "L-Alanine", "Pyruvic Acid", "Imidazole", "Adenine", "Formaldehyde", "Urea", "Citric Acid", "Acetic Acid"]
            chosen = self.rng.choice(library_candidates, size=int(self.rng.integers(3, 6)), replace=False)
            for c_name in chosen:
                c_count = int(self.rng.integers(12, 28))
                role = str(self.rng.choice(["autotroph", "catalyst", "cooperator"]))
                self.add_compound_to_world(c_name, count=c_count, ecological_role=role)

            scenario_name = f"Procedural Niche (T={rand_temp}K, pH={rand_ph}, Seed={self.seed})"

        elif preset == "sandbox_empty":
            scenario_name = "Empty Chemical Sandbox"

        return {
            "status": "reset",
            "preset": preset,
            "scenario_name": scenario_name,
            "seed": self.seed,
            "universe_mode": self.universe_mode.value,
            "interface_mode": self.interface_mode.value,
            "initial_entities_count": len(self.entities),
            "initial_compounds_count": len(self.compounds),
        }

    # -------------------------------------------------------------------------
    # Compound & Entity Injection
    # -------------------------------------------------------------------------

    def _resolve_compound_rec(self, query: str, input_type: str = "auto") -> CompoundRecord:
        """Resolve query checking local world compounds, common aliases, and chemistry library."""
        q = (query or "").strip()
        # 1. Match local compounds by compound_id, name, or canonical_smiles
        for c in self.compounds.values():
            if q.lower() in [c.compound_id.lower(), c.name.lower(), c.canonical_smiles.lower()]:
                return c
        # 2. Match common aliases
        alias_map = {
            "fluorouracil": "5-Fluorouracil",
            "5fu": "5-Fluorouracil",
            "alanine": "L-Alanine",
        }
        mapped_query = alias_map.get(q.lower(), q)
        return self.chemistry.resolve_compound_input(mapped_query, input_type=input_type)

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
        """Resolve a compound by name/SMILES, register in genealogy, and spawn spatial entities."""
        rec = self._resolve_compound_rec(query, input_type=input_type)
        cid = rec.compound_id

        if cid not in self.compounds:
            self.compounds[cid] = rec
            self.genealogy.register_compound(
                compound=rec,
                birth_step=self.clocks.evolutionary_step,
                similarity_to_parent=1.0,
                mutation_description=f"Introduced into world: {rec.name}",
                color_hex=self._assign_entity_color(rec),
            )

        # Determine spatial injection center
        cx = float(self.rng.uniform(self.env.width * 0.15, self.env.width * 0.85)) if x is None else float(x)
        cy = float(self.rng.uniform(self.env.height * 0.15, self.env.height * 0.85)) if y is None else float(y)

        kind_enum = EntityKind.THERAPEUTIC if entity_kind == "DRUG" else (EntityKind(entity_kind) if entity_kind in EntityKind.__members__ else EntityKind.PROTOCELL)
        color = self._assign_entity_color(rec)
        shape = self._assign_entity_shape(rec)

        new_ids: list[str] = []
        for i in range(max(1, count)):
            self._entity_counter += 1
            eid = f"{cid[:8]}-E{self._entity_counter:06d}"
            spread_r = self.rng.uniform(0.0, 35.0)
            theta = self.rng.uniform(0.0, 2.0 * math.pi)
            px = float(np.clip(cx + spread_r * math.cos(theta), 10.0, self.env.width - 10.0))
            py = float(np.clip(cy + spread_r * math.sin(theta), 10.0, self.env.height - 10.0))

            entity = SpatialEntity(
                entity_id=eid,
                compound_id=cid,
                kind=kind_enum,
                x=px,
                y=py,
                vx=float(self.rng.normal(0.0, 0.2)),
                vy=float(self.rng.normal(0.0, 0.2)),
                radius=max(3.0, min(9.0, math.sqrt(rec.mol_weight) * 0.45)),
                mass=max(10.0, rec.mol_weight),
                energy=12.0,
                stored_substrate=6.0,
                generation=self.clocks.evolutionary_generation,
                family_id=rec.family_id,
                color_hex=color,
                shape=shape,
                ecological_role=ecological_role,
            )
            self.entities.append(entity)
            new_ids.append(eid)

        self.mission_progress["entities_added"] += len(new_ids)
        self._check_achievement("ACH-FIRST-COMPOUND")

        return {
            "status": "added",
            "compound_id": cid,
            "name": rec.name,
            "smiles": rec.canonical_smiles,
            "entities_spawned": len(new_ids),
            "total_world_entities": len(self.entities),
        }

    def import_sdf_compounds(self, sdf_text: str, spawn_in_world: bool = False) -> list[dict[str, Any]]:
        """Import compounds from MDL SD-File text, registering each in the compound library."""
        imported = self.chemistry.import_sdf(sdf_text)
        summaries: list[dict[str, Any]] = []

        for rec in imported:
            self.compounds[rec.compound_id] = rec
            self.genealogy.register_compound(
                compound=rec,
                birth_step=self.clocks.evolutionary_step,
                mutation_description="Imported from SD-File",
            )
            if spawn_in_world:
                self.add_compound_to_world(rec.canonical_smiles, count=8)

            summaries.append({
                "compound_id": rec.compound_id,
                "name": rec.name,
                "formula": rec.formula,
                "mol_weight": rec.mol_weight,
                "smiles": rec.canonical_smiles,
            })
        return summaries

    # -------------------------------------------------------------------------
    # Universe Stepping & Emergent Dynamics
    # -------------------------------------------------------------------------

    def step_universe(
        self,
        steps: int = 1,
        dt: float = 0.25,
        max_entities: int = 800,
    ) -> dict[str, Any]:
        """
        Advance the 2D universe by one or more discrete steps:
        - Reaction-diffusion grid transport
        - 2D Brownian spatial physics and collisions
        - Stoichiometric reactions on colliding pairs
        - Darwinian division, metabolism, and heritable structural variation
        - Configurable carrying capacity enforcement
        - Periodic auto-checkpointing and authentic period delta summaries.
        """
        t0 = time.perf_counter()
        total_steps = max(1, int(steps))
        step_births = 0
        step_deaths = 0
        step_reactions = 0
        initial_lineages_count = len(self.genealogy.nodes)

        for _ in range(total_steps):
            # 1. Step continuous substrate reaction-diffusion field
            self.grid.step_field(
                name="substrate",
                diff_coeff=18.0,
                dt=dt,
                influx_rate=self.env.resource_influx_rate,
                clearance_rate=self.env.resource_clearance_rate,
                vx=self.env.current_vx,
                vy=self.env.current_vy,
            )

            # 2. Physics & collision detection
            phys_res = self.physics.step_entities(
                entities=self.entities,
                env=self.env,
                grid=self.grid,
                dt=dt,
                rng=self.rng,
            )
            collision_pairs = phys_res.get("collision_pairs", [])

            # 3. Chemical reactions on colliding pairs (sampled at most 16 pairs for high performance)
            if len(collision_pairs) > 16:
                pair_indices = self.rng.choice(len(collision_pairs), size=16, replace=False)
                active_collision_pairs = [collision_pairs[i] for i in pair_indices]
            else:
                active_collision_pairs = collision_pairs

            for e1, e2 in active_collision_pairs:
                if not e1.alive or not e2.alive:
                    continue
                c1 = self.compounds.get(e1.compound_id)
                c2 = self.compounds.get(e2.compound_id)
                if c1 is not None and c2 is not None:
                    # Prevent unbounded polymer chaining in 2D Mode A universe
                    if c1.mol_weight + c2.mol_weight > 350.0:
                        continue

                    cat_present = (e1.ecological_role == "catalyst" or e2.ecological_role == "catalyst")
                    s1 = min(c1.canonical_smiles, c2.canonical_smiles)
                    s2 = max(c1.canonical_smiles, c2.canonical_smiles)
                    rxn_key = (s1, s2, cat_present, self.universe_mode.value, self.env.temperature_k, self.env.ph)
                    if rxn_key in self._reaction_cache:
                        rxn_res = self._reaction_cache[rxn_key]
                    else:
                        rxn_res = self.chemistry.attempt_reaction(
                            comp_a=c1,
                            comp_b=c2,
                            temperature_k=self.env.temperature_k,
                            ph=self.env.ph,
                            catalyst_present=cat_present,
                            mode=self.universe_mode,
                            rng=None,
                        )
                        self._reaction_cache[rxn_key] = rxn_res

                    # Cache possible transformations, never stochastic encounter outcomes.
                    if rxn_res.get("occurred", False) and self.rng.random() <= rxn_res.get("probability", 1.0):
                        step_reactions += 1
                        self.reactions_count += 1
                        self._check_achievement("ACH-FIRST-REACTION")
                        rxn_event = {
                            "type": "REACTION_OCCURRED",
                            "x": round(float((e1.x + e2.x) / 2.0), 2),
                            "y": round(float((e1.y + e2.y) / 2.0), 2),
                            "reactant_a": c1.name,
                            "reactant_b": c2.name,
                            "reaction_type": rxn_res.get("reaction_type", "transformation"),
                            "step": self.clocks.evolutionary_step,
                        }
                        self.recent_events.append(rxn_event)
                        if len(self.recent_events) > 50:
                            self.recent_events = self.recent_events[-50:]
                        # An encounter complex binds existing particles reversibly.
                        if rxn_res.get("reaction_type") == "non_covalent_complexation":
                            e1.bound_to, e2.bound_to = e2.entity_id, e1.entity_id
                            continue
                        # Each particle represents one molecule/complex. Consume both
                        # reactants and emit every product exactly once.
                        e1.alive = e2.alive = False
                        product_x, product_y = (e1.x + e2.x) / 2, (e1.y + e2.y) / 2
                        products = rxn_res.get("products", [])
                        for prod in rxn_res.get("products", []):
                            # Reuse existing CompoundRecord if known by canonical SMILES
                            matched_prod = prod
                            for known_c in self.compounds.values():
                                if known_c.canonical_smiles == prod.canonical_smiles:
                                    matched_prod = known_c
                                    break
                            if matched_prod.compound_id not in self.compounds:
                                self.compounds[matched_prod.compound_id] = matched_prod
                                self.genealogy.register_compound(
                                    compound=matched_prod,
                                    birth_step=self.clocks.evolutionary_step,
                                    similarity_to_parent=0.85,
                                    mutation_description=f"Reaction: {rxn_res.get('reaction_type', 'transformation')}",
                                )
                            self._entity_counter += 1
                            offset = (len(self.entities) % 3 - 1) * 3.0
                            self.entities.append(SpatialEntity(
                                entity_id=f"RXN-E{self._entity_counter:08d}",
                                compound_id=matched_prod.compound_id,
                                kind=e1.kind if len(products) == 1 else EntityKind.MOLECULE,
                                x=min(self.env.width - 10, max(10, product_x + offset)),
                                y=min(self.env.height - 10, max(10, product_y - offset)),
                                mass=matched_prod.mol_weight, charge=matched_prod.formal_charge,
                                radius=max(3.0, min(12.0, math.sqrt(matched_prod.mol_weight) * 0.45)),
                                energy=(e1.energy + e2.energy) / len(products),
                                stored_substrate=(e1.stored_substrate + e2.stored_substrate) / len(products),
                                family_id=matched_prod.family_id,
                                generation=max(e1.generation, e2.generation),
                                ecological_role=e1.ecological_role,
                                color_hex=self._assign_entity_color(matched_prod),
                                shape=self._assign_entity_shape(matched_prod),
                            ))

            # 4. Darwinian evolutionary step (metabolism, division, mutation, extinction)
            replicators = [e for e in self.entities if e.alive and e.kind == EntityKind.PROTOCELL]
            inert_entities = [e for e in self.entities if e.alive and e.kind != EntityKind.PROTOCELL]
            eco_metrics = self.darwinian.step(
                entities=replicators if self.universe_mode == UniverseMode.ARTIFICIAL_LIFE else [],
                environment=self.env,
                substrate_source=self.substrate_adapter,
                dt=dt,
                rng=self.rng,
                current_step=self.clocks.evolutionary_step,
            )
            self.entities = (eco_metrics.get("entities", replicators) + inert_entities
                             if self.universe_mode == UniverseMode.ARTIFICIAL_LIFE
                             else [e for e in self.entities if e.alive])
            abundances: dict[str, int] = {}
            for entity in self.entities:
                if entity.alive:
                    abundances[entity.compound_id] = abundances.get(entity.compound_id, 0) + 1
            self.genealogy.update_abundances(abundances, current_step=self.clocks.evolutionary_step)
            step_births += eco_metrics.get("births", 0)
            step_deaths += eco_metrics.get("deaths", 0)

            # 5. Carrying capacity enforcement
            if self.universe_mode == UniverseMode.ARTIFICIAL_LIFE and len(self.entities) > max_entities:
                self.entities.sort(key=lambda e: e.energy, reverse=True)
                self.entities = self.entities[:max_entities]

            # 6. Clocks update
            self.clocks.evolutionary_step += 1
            if self.clocks.evolutionary_step % self.clocks.steps_per_generation == 0:
                self.clocks.evolutionary_generation += 1

        wall_delta = time.perf_counter() - t0
        self.clocks.computational_seconds += wall_delta

        # Mission & Achievement Updates
        self.mission_progress["evolutionary_generation"] = self.clocks.evolutionary_generation
        if self.clocks.evolutionary_generation >= 100:
            self._check_achievement("ACH-CENTURY-LINEAGE")

        # Event classification via Laya
        alive_entities = [e for e in self.entities if e.alive]
        tree_state = self.genealogy.to_dict()
        laya_eval = self.laya.classify_event({
            "step": self.clocks.evolutionary_step,
            "generation": self.clocks.evolutionary_generation,
            "entity_count": len(alive_entities),
            "shannon_diversity": eco_metrics.get("shannon_diversity", 0.0),
        })

        # Compute authentic period summary from actual deltas
        new_lineages = len(self.genealogy.nodes) - initial_lineages_count
        extinct_nodes = [n for n in self.genealogy.nodes.values() if n.is_extinct]
        active_cids = {e.compound_id for e in alive_entities}
        active_fams: dict[str, int] = {}
        for e in alive_entities:
            active_fams[e.family_id] = active_fams.get(e.family_id, 0) + 1

        dominant_fam = max(active_fams.items(), key=lambda kv: kv[1])[0] if active_fams else "None"
        dominant_comp_name = "None"
        if alive_entities:
            cid_counts: dict[str, int] = {}
            for e in alive_entities:
                cid_counts[e.compound_id] = cid_counts.get(e.compound_id, 0) + 1
            top_cid = max(cid_counts.items(), key=lambda kv: kv[1])[0]
            if top_cid in self.compounds:
                dominant_comp_name = self.compounds[top_cid].name

        summary_sentences = [
            f"Simulated {total_steps} step(s) advancing evolutionary time to Generation {self.clocks.evolutionary_generation} ({self.clocks.virtual_years} virtual years).",
            f"Population dynamics: {step_births} births, {step_deaths} deaths; active carrying capacity at {len(alive_entities)} replicators.",
            f"Chemical transformations: executed {step_reactions} reaction event(s), discovering {new_lineages} novel chemical lineage(s).",
            f"Ecosystem structure: {len(active_fams)} active family/families; dominant family '{dominant_fam}' and top compound '{dominant_comp_name}'.",
        ]

        summary = {
            "steps_executed": total_steps,
            "computational_seconds_step": round(wall_delta, 4),
            "clocks": self.clocks.to_dict(),
            "active_entities": len(alive_entities),
            "step_births": step_births,
            "step_deaths": step_deaths,
            "reactions_executed": step_reactions,
            "new_lineages_born": new_lineages,
            "families_extinct": len(extinct_nodes),
            "dominant_family_name": dominant_fam,
            "top_persistent_compound": dominant_comp_name,
            "shannon_diversity": eco_metrics.get("shannon_diversity", 0.0),
            "laya_event_classification": laya_eval,
            "summary_sentences": summary_sentences,
        }

        # Record trajectory snapshot in history
        self.history.append({
            "step": self.clocks.evolutionary_step,
            "generation": self.clocks.evolutionary_generation,
            "virtual_years": self.clocks.virtual_years,
            "active_entities": len(alive_entities),
            "shannon_diversity": eco_metrics.get("shannon_diversity", 0.0),
            "reactions": step_reactions,
        })
        if len(self.history) > 500:
            self.history = self.history[-500:]

        return summary

    def stir_mixture(self) -> dict[str, Any]:
        """User-applied mechanical agitation in abstract simulation units."""
        for entity in self.entities:
            if entity.alive:
                entity.vx = float(self.rng.normal(0, 28))
                entity.vy = float(self.rng.normal(0, 28))
        self.recent_events.append({"type": "MIXTURE_STIRRED", "step": self.clocks.evolutionary_step,
                                   "description": "User applied mechanical agitation (abstract velocity units)."})
        return {"entities_stirred": len(self.entities)}

    def advance_by_preset(self, preset: str) -> dict[str, Any]:
        """
        Accelerate simulation by predefined generation or virtual-year milestones:
        '10_gen', '100_gen', '1000_gen', '10000_gen', '1_yr', '10_yr', '100_yr', '1000_yr', '10000_yr'.
        Uses fast vectorized/sampled stepping for large jumps (>25 gen) so it finishes in <3 seconds deterministically!
        """
        gen_map: dict[str, int] = {
            "10_gen": 10,
            "100_gen": 100,
            "1000_gen": 1000,
            "10000_gen": 10000,
            "1_yr": 10,
            "10_yr": 100,
            "100_yr": 1000,
            "1000_yr": 10000,
            "10000_yr": 100000,
        }

        target_gens = gen_map.get(preset, 10)
        t0 = time.perf_counter()

        result = self.step_universe(steps=target_gens * self.clocks.steps_per_generation, dt=0.25)
        result.update(preset=preset, acceleration_mode="standard_stepping", generations_advanced=target_gens)
        return result

    # -------------------------------------------------------------------------
    # Oncology & Cellular Microenvironment (PDAC)
    # -------------------------------------------------------------------------

    def step_tumor(self, steps: int = 1, dt_seconds: float = 900.0) -> dict[str, Any]:
        """Advance the 2D PDAC cellular tumor microenvironment by dt_seconds (default 15 min)."""
        total_steps = max(1, int(steps))
        last_snap = {}
        for _ in range(total_steps):
            step_res = self.tumor_module.step(dt_seconds)
            events = step_res.get("events", {})
            if any(events.values()):
                self.tissue_events.append({"biological_hours": step_res.get("biological_hours"), **events})
                self.tissue_events = self.tissue_events[-30:]
            self.clocks.biological_seconds += dt_seconds
            self.mission_progress["tumor_steps_observed"] += 1

        last_snap = self.tumor_module.get_snapshot()
        self._check_achievement("ACH-FIRST-TUMOR-STEP")

        return {
            "steps_executed": total_steps,
            "biological_seconds": self.clocks.biological_seconds,
            "biological_hours": self.clocks.biological_hours,
            "biological_days": self.clocks.biological_days,
            "tumor_snapshot": last_snap,
        }

    def introduce_compound_to_tumor(
        self,
        compound_query: str,
        applied_concentration_um: float = 5.0,
        injection_region: str = "vascular",
        enable_systemic_pbpk: bool = False,
        dose_mg: float = 100.0,
    ) -> dict[str, Any]:
        """Introduce a compound into the 2D tumor microenvironment with optional systemic PBPK simulation."""
        rec = self._resolve_compound_rec(compound_query)
        if "." in rec.canonical_smiles:
            raise ValueError("Artificial multicomponent complexes are not supported as tissue exposures. Choose a single validated molecular structure.")
        admet = self.pharmacology.evaluate_admet(rec)
        tox = self.toxicology.evaluate_toxicology(rec)

        pbpk_res = None
        if enable_systemic_pbpk:
            pbpk_res = self.pbpk.simulate(rec, dose_mg=dose_mg)

        # Adapt profile for PDACModule
        clean_pharma, model_info = self._tissue_pharmacology(rec)
        self.tissue_model_info = model_info

        apply_res = self.tumor_module.apply_compound_exposure(
            compound=rec,
            pharmacology_profile=clean_pharma,
            applied_concentration_um=applied_concentration_um,
            injection_region=injection_region,
        )

        return {
            "compound_id": rec.compound_id,
            "name": rec.name,
            "applied_concentration_um": applied_concentration_um,
            "injection_region": injection_region,
            "exposure_result": apply_res,
            "pbpk_simulation": pbpk_res,
            "toxicology_profile": tox,
            "model_info": model_info,
        }

    def _tissue_pharmacology(self, record):
        profile = dict(self.pharmacology.evaluate_admet(record))
        if self.disease_id != "PDAC":
            references = getattr(self.tumor_module, "REFERENCE_COMPOUNDS", {})
            for reference in references.values():
                try:
                    reference_record = self.chemistry.resolve_compound_input(reference["name"].split("(")[0].strip())
                except ValueError:
                    reference_record = self.chemistry.build_compound(reference["smiles"])
                if reference_record.canonical_smiles == record.canonical_smiles:
                    profile.update(reference)
                    profile["emax"] = self.tumor_module.emax_default
                    profile["hill_slope"] = getattr(self.tumor_module, "hill_slope_default", 2.0)
                    return profile, {"status": "exploratory reference-anchored model", "source": reference["name"],
                                     "note": "Disease-specific reference parameters are model assumptions; they are not a validated prediction for a patient."}
            raise ValueError(f"No disease-specific response model is available for this compound in {self.disease_id}. Structural similarity alone cannot supply one.")
        # Offer the calibrated QSAR for structures that have no curated bioactivity.
        # predict_antitumor_activity still prefers curated measurement when present,
        # so reference drugs are unaffected and only novel structures change path.
        from simulation.pharmacology.admet import qsar_potency_hint
        hint = qsar_potency_hint(getattr(self, "qsar", None), record)
        response = self.pharmacology.predict_antitumor_activity(record, qsar_pred=hint)
        for key in ("ic50_sensitive_um", "ic50_resistant_um", "ic50_healthy_um", "hill_slope"):
            value = response.get(key, {})
            raw = value.get("value") if isinstance(value, dict) else value
            if not isinstance(raw, (int, float)):
                raise ValueError("Insufficient pharmacology data to simulate this structure.")
            profile[key] = float(raw)
        emax = response["emax"]["value"]
        # Fractional efficacy and a per-second death rate have different units.
        profile["emax"] = float(emax) * self.tumor_module.emax_default
        profile["selectivity_index"] = profile["ic50_healthy_um"] / max(0.001, profile["ic50_sensitive_um"])
        profile["fraction_unbound_plasma"] = profile["fraction_unbound_plasma"]["value"]
        category = response["ic50_sensitive_um"].get("epistemic_category")
        if record.reference_bioactivity:
            provenance = "curated_experimental_bioactivity"
        elif hint is not None:
            provenance = "calibrated_qsar"
        else:
            provenance = "structure_property_heuristic"
        info = {"status": "exploratory", "source": response["ic50_sensitive_um"]["source"],
                "ic50_sensitive_um": profile["ic50_sensitive_um"], "maximum_kill_rate_per_second": profile["emax"],
                "potency_provenance": provenance,
                "epistemic_category": category,
                "note": "Uncalibrated tissue response model; reference-associated or structure-property parameters are hypotheses, not clinical evidence."}
        if provenance == "calibrated_qsar" and hint is not None:
            info["conformal_q_95_log_units"] = hint.get("conformal_q_95")
            info["potency_caveat"] = (
                "Potency comes from the calibrated QSAR, which orders compounds but does "
                "not measure them. Its 95% conformal interval spans orders of magnitude in "
                "IC50, and that uncertainty is not propagated through the tissue simulation."
            )
        return profile, info

    def run_tumor_controlled_comparison(
        self,
        compound_query: str,
        applied_concentration_um: float = 5.0,
        duration_hours: float = 48.0,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """
        Execute identical-initial-condition comparison:
        Control (0.0 uM) vs Treated (applied_concentration_um).
        """
        rec = self._resolve_compound_rec(compound_query)
        clean_pharma, _model_info = self._tissue_pharmacology(rec)

        comp_seed = self.seed if seed is None else int(seed)
        res = self.tumor_module.run_controlled_comparison(
            compound=rec,
            pharmacology_profile=clean_pharma,
            applied_concentration_um=applied_concentration_um,
            duration_hours=duration_hours,
            seed=comp_seed,
        )
        self.mission_progress["controlled_comparison_run"] += 1
        self._check_achievement("ACH-CONTROLLED-TRIAL")
        return res

    def run_coevolution_experiment(
        self,
        n_cycles: int = 4,
        hours_per_cycle: float = 24.0,
        applied_concentration_um: float = 3.0,
    ) -> dict[str, Any]:
        """Run co-evolutionary experiment between evolving drug variants and adapting resistant PDAC subclones."""
        if self.disease_id != "PDAC":
            raise ValueError("Chemical-tumour coevolution currently supports the PDAC model only.")
        initial_compounds = [self.chemistry.resolve_compound_input(name) for name in ("Gemcitabine", "5-Fluorouracil", "Erlotinib")]
        if not initial_compounds:
            gem = self.chemistry.resolve_compound_input("Gemcitabine")
            initial_compounds = [gem]

        coev = CoevolutionEngine(
            tumor_module=self.tumor_module,
            initial_compounds=initial_compounds,
            variation_callback=self._mutate_compound_callback,
            pharmacology_callback=lambda record, state: self._tissue_pharmacology(record)[0],
            seed=self.seed,
        )
        res = coev.run_coevolution(
            n_cycles=n_cycles,
            hours_per_cycle=hours_per_cycle,
            applied_concentration_um=applied_concentration_um,
        )
        self.clocks.biological_seconds = self.tumor_module.biological_seconds
        self.tissue_model_info = {"status": "exploratory coevolution", "note": "RDKit-valid chemical variation coupled to an uncalibrated PDAC tissue model."}
        res["model_info"] = self.tissue_model_info
        self.mission_progress["resistance_observed"] += 1
        return res

    def evaluate_combination(
        self,
        compound_a_query: str,
        compound_b_query: str,
        conc_a_um: float = 5.0,
        conc_b_um: float = 5.0,
        injection_region: str = "vascular",
        run_4_arm_trial: bool = True,
        duration_hours: float = 48.0,
    ) -> dict[str, Any]:
        """
        Evaluate 2-drug combination synergy, Bliss independence, and optional 4-arm in silico trial
        (Control vs Monotherapy A vs Monotherapy B vs Combination).
        """
        rec_a = self._resolve_compound_rec(compound_a_query)
        rec_b = self._resolve_compound_rec(compound_b_query)

        try:
            qsar_a = self.qsar.predict(rec_a.canonical_smiles, disease_id=self.disease_id)
        except TypeError:
            qsar_a = self.qsar.predict(rec_a.canonical_smiles)

        try:
            qsar_b = self.qsar.predict(rec_b.canonical_smiles, disease_id=self.disease_id)
        except TypeError:
            qsar_b = self.qsar.predict(rec_b.canonical_smiles)

        synergy_profile = self.pharmacology.evaluate_combination_synergy(
            compound_a=rec_a,
            compound_b=rec_b,
            conc_a_um=conc_a_um,
            conc_b_um=conc_b_um,
            qsar_a=qsar_a,
            qsar_b=qsar_b,
        )

        # Structural dissimilarity cannot establish drug synergy.
        synergy_factor = 1.0
        clean_a, info_a = self._tissue_pharmacology(rec_a)
        clean_b, info_b = self._tissue_pharmacology(rec_b)
        if self.disease_id != "PDAC":
            raise ValueError("The two-drug interaction model is currently implemented for PDAC only. Use the disease-specific single-agent comparison.")
        self.tissue_model_info = {"status": "exploratory combination", "note": "Independent drug hazards; synergy is not assumed or experimentally established.", "a": info_a, "b": info_b}

        exposure_res = self.tumor_module.apply_combination_exposure(
            compound_a=rec_a,
            compound_b=rec_b,
            pharma_a=clean_a,
            pharma_b=clean_b,
            conc_a_um=conc_a_um,
            conc_b_um=conc_b_um,
            injection_region=injection_region,
            synergy_factor=synergy_factor,
        )

        trial_res = None
        if run_4_arm_trial:
            trial_res = self.tumor_module.run_combination_comparison(
                compound_a=rec_a,
                compound_b=rec_b,
                pharma_a=clean_a,
                pharma_b=clean_b,
                conc_a_um=conc_a_um,
                conc_b_um=conc_b_um,
                duration_hours=duration_hours,
                seed=self.seed,
                synergy_factor=synergy_factor,
            )

        return {
            "disease_id": self.disease_id,
            "compound_a": rec_a.to_dict(),
            "compound_b": rec_b.to_dict(),
            "synergy_profile": synergy_profile,
            "exposure_result": exposure_res,
            "trial_4_arm": trial_res,
        }

    # -------------------------------------------------------------------------
    # Discovery Pipeline & Compound Inspection
    # -------------------------------------------------------------------------

    def run_discovery_pipeline(
        self,
        seed_queries: list[str] | None = None,
        generations: int = 4,
        population_size: int = 16,
        top_k_cellular: int = 3,
    ) -> dict[str, Any]:
        """Execute the 15-stage multi-objective candidate discovery pipeline."""
        res = self.discovery_pipeline.run(
            seed_queries=seed_queries,
            generations=generations,
            population_size=population_size,
            top_k_cellular=top_k_cellular,
            seed=self.seed,
        )
        self.register_discovery_results(res)
        return res

    def register_discovery_results(self, res: dict[str, Any]) -> None:
        self.discovery_results = res
        for item in res.get("evaluated_candidates", res.get("pareto_frontier", [])):
            record = self.chemistry.build_compound(
                item["canonical_smiles"], name=item.get("name", ""),
                origin_generation=item.get("origin_generation", 0),
                parent_ids=item.get("parent_ids", []), transformation_type="directed_optimization",
                provenance="Directed discovery pipeline",
            )
            if record.compound_id not in self.compounds:
                self.compounds[record.compound_id] = record
                self.genealogy.register_compound(record, birth_step=self.clocks.evolutionary_step,
                                                 mutation_description="Directed search candidate; not deployed")
        self.mission_progress["pareto_evaluation_completed"] += 1
        self._check_achievement("ACH-PARETO-OPTIMAL")

    def inspect_compound(
        self,
        compound_id_or_query: str,
        enable_systemic_pbpk: bool = False,
    ) -> dict[str, Any]:
        """Full scientific profiling of any molecule with explicit EpistemicCategory tagging."""
        rec = self._resolve_compound_rec(compound_id_or_query)

        admet = self.pharmacology.evaluate_admet(rec)
        tox = self.toxicology.evaluate_toxicology(rec)
        try:
            qsar_res = self.qsar.predict(rec.canonical_smiles, disease_id=self.disease_id)
        except TypeError:
            qsar_res = self.qsar.predict(rec.canonical_smiles)
        ad_check = self.qsar.assess_applicability_domain(rec.canonical_smiles)
        antitumor_raw = self.pharmacology.predict_antitumor_activity(rec)
        antitumor = {k: v for k, v in antitumor_raw.items() if not callable(v)}

        pbpk_res = None
        if enable_systemic_pbpk:
            pbpk_res = self.pbpk.simulate(rec)

        bio_indicators = self.biomarkers_engine.compute_biological_indicators(
            tumor_microenvironment_state=self.tumor_module.get_snapshot(),
            systemic_pbpk_result=pbpk_res,
            compound=rec,
            toxicity_profile=tox,
        )

        rec_dict = rec.to_dict()
        if not rec_dict.get("svg_2d"):
            try:
                rec_dict["svg_2d"] = self.chemistry._render_svg_2d(rec.canonical_smiles)
            except Exception:
                rec_dict["svg_2d"] = ""

        return {
            # Top-level flat keys for direct access
            "compound_id": rec.compound_id,
            "name": rec.name,
            "canonical_smiles": rec.canonical_smiles,
            "formula": rec.formula,
            "molecular_weight": rec.mol_weight,
            "qed": rec.qed,
            "logp": rec.logp,
            "tpsa": rec.tpsa,
            "qsar_prediction": qsar_res,
            "applicability_domain": ad_check,
            "admet_profile": admet,
            "admet": admet,
            "toxicology_profile": tox,
            "toxicology": tox,
            "antitumor_activity": antitumor,
            "systemic_pbpk": pbpk_res,
            "pbpk": pbpk_res,
            "biological_indicators": bio_indicators,
            "compound": rec_dict,
            "epistemic_disclaimer": "Computational estimate; does not constitute clinical pharmacology.",
        }

    # -------------------------------------------------------------------------
    # Molecular Editor & Structural Diffing
    # -------------------------------------------------------------------------

    def get_editor_operations(self) -> dict[str, Any]:
        """List all supported structural edits, substituents, atom replacements, and bioisosteres."""
        return self.mol_editor.list_available_operations()

    def edit_molecule(
        self,
        smiles: str,
        operation: str,
        param: str = "methyl",
        atom_idx: int | None = None,
        register_in_world: bool = False,
    ) -> dict[str, Any]:
        """Apply an interactive chemical edit to a molecule with structural diff and optional world registration."""
        op_map = {
            "attach_substituent": "add_group",
            "attach_group": "add_group",
            "swap_atom": "replace_atom",
            "fuse_ring": "fuse_or_append_ring",
            "bioisostere": "bioisostere_swap",
            "sanitize": "clean_and_canonicalize",
        }
        op = op_map.get(operation.lower(), operation)
        res = self.mol_editor.apply_edit(
            smiles=smiles,
            operation=op,
            param=param,
            atom_idx=atom_idx,
            seed=self.seed,
        )

        if register_in_world and res.get("valid") and res.get("edited_smiles"):
            edited_smi = res["edited_smiles"]
            rec = self.chemistry.resolve_compound_input(edited_smi)
            if rec.compound_id not in self.compounds:
                self.compounds[rec.compound_id] = rec
                self.genealogy.register_compound(
                    compound=rec,
                    birth_step=self.clocks.evolutionary_step,
                    similarity_to_parent=0.9,
                    mutation_description=f"Molecular edit ({operation}:{param})",
                )
            self.add_compound_to_world(rec.canonical_smiles, count=6, entity_kind="THERAPEUTIC", ecological_role="neutral")
            res["registered_in_world"] = True
            res["registered_compound_id"] = rec.compound_id

        return res

    def diff_molecules(self, query_a: str, query_b: str) -> dict[str, Any]:
        """Compute visual, atom-level, and property structural diff between two molecules."""
        rec_a = self._resolve_compound_rec(query_a)
        rec_b = self._resolve_compound_rec(query_b)
        diff_res = self.mol_editor.compute_structural_diff(rec_a.canonical_smiles, rec_b.canonical_smiles)
        diff_res["name_a"] = rec_a.name
        diff_res["name_b"] = rec_b.name
        return diff_res

    def triage_current_compounds(self, weights: dict[str, float] | None = None) -> dict[str, Any]:
        """Run Laya multi-objective candidate triage on active world and reference compounds."""
        candidates = list(self.compounds.values())
        if not candidates:
            from simulation.chemistry.library import get_builtin_library
            for item in get_builtin_library()[:6]:
                try:
                    c = self.chemistry.resolve_compound_input(item["smiles"])
                    candidates.append(c)
                except Exception:
                    pass
        return self.laya.triage_candidates(candidates=candidates, weights=weights)

    def get_genealogy_view(self, filters: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return evolutionary lineage DAG tree view."""
        return self.genealogy.query_tree(**{k: v for k, v in (filters or {}).items() if v is not None})

    def get_biological_indicators(self, enable_systemic_pbpk: bool | None = None) -> dict[str, Any]:
        """Compute integrated biological indicators across tumor and systemic compartments."""
        snap = self.tumor_module.get_snapshot()
        pbpk_data = None
        if enable_systemic_pbpk and self.tumor_module.active_compound:
            pbpk_data = self.pbpk.simulate(self.tumor_module.active_compound)

        return self.biomarkers_engine.compute_biological_indicators(
            tumor_microenvironment_state=snap,
            systemic_pbpk_result=pbpk_data,
            compound=self.tumor_module.active_compound,
        )

    # -------------------------------------------------------------------------
    # Local AI Assistant & Action Application
    # -------------------------------------------------------------------------

    def ask_assistant(
        self,
        question: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> dict[str, Any]:
        """Consult Local Gemma Assistant with grounded simulation context and audit logging."""
        ctx = {
            "universe_mode": self.universe_mode.value,
            "interface_mode": self.interface_mode.value,
            "active_entities": len([e for e in self.entities if e.alive]),
            "entity_count": len([e for e in self.entities if e.alive]),
            "evolutionary_step": self.clocks.evolutionary_step,
            "family_counts": self.genealogy.query_tree().get("family_counts", {}),
            "ph": self.env.ph,
            "oxygen_norm": self.env.oxygen_norm,
            "reactions_count": self.reactions_count,
            "evolutionary_generation": self.clocks.evolutionary_generation,
            "biological_hours": self.clocks.biological_hours,
            "disease_id": self.disease_id,
            "mutation_rate": self.darwinian.mutation_rate,
            "registered_compounds": len(self.compounds),
        }
        family_population: dict[str, int] = {}
        for entity in self.entities:
            if entity.alive:
                family_population[entity.family_id] = family_population.get(entity.family_id, 0) + 1
        if family_population:
            dominant, count = max(family_population.items(), key=lambda item: item[1])
            ctx.update(dominant_family=dominant, dominant_share_pct=count / max(1, sum(family_population.values())) * 100)
        tissue = self.tumor_module.get_snapshot()
        summary = tissue.get("summary_metrics", {})
        ctx.update(tumor_cells_count=summary.get("viable_tumor_cells", 0),
                   resistant_cells=round(summary.get("resistant_fraction", 0) * summary.get("viable_tumor_cells", 0)),
                   current_compound=self.tumor_module.active_compound.name if self.tumor_module.active_compound else "None")
        res = self.assistant.ask(
            question,
            simulation_context=ctx,
            mode=self.interface_mode,
            provider_override=provider,
            model_override=model,
        )
        # Log to AI Model Registry audit ledger
        self.model_registry.log_ai_decision(
            module="EVOCUREEngine.ask_assistant",
            action_type="chat_query",
            query_summary=question[:100],
            output_summary=res.get("answer", "")[:100],
            provider=res.get("provider", "local_gemma"),
            epistemic_category=res.get("epistemic_category", EpistemicCategory.AI_HYPOTHESIS),
        )
        return res

    def get_ai_runtime_status(self) -> dict[str, Any]:
        """Query real-time health, provider configuration, and model telemetry across all AI engines."""
        ledger = getattr(self.assistant, "ledger", []) or getattr(self.model_registry, "ledger", [])
        return {
            "gemma": self.assistant.get_status(),
            "laya": self.laya.get_status(),
            "qsar_version": "1.0.0",
            "ledger": ledger[-50:] if isinstance(ledger, list) else [],
        }

    def apply_assistant_proposed_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Safely execute parameter or world actions proposed by the local AI assistant or UI."""
        act = action.get("action") or action.get("action_type") or action.get("type", "")
        params = action.get("parameters") or action

        if act in ["update_environment", "set_environment_parameter"]:
            if "parameter" in action and "requested_value" in action:
                param_name = action["parameter"]
                val = action["requested_value"]
                return self.update_environment({param_name: val})
            return self.update_environment(params.get("patch", params))

        elif act in ["add_compound", "introduce_compound"]:
            query = params.get("query") or params.get("compound") or "Glycine"
            count = int(params.get("count", 12))
            return self.add_compound_to_world(query, count=count)

        elif act in ["set_mode", "switch_mode"]:
            mode = params.get("mode", "PLAY")
            return self.set_interface_mode(mode)

        elif act in ["set_active_disease", "switch_disease"]:
            d_id = params.get("disease_id", "PDAC")
            return self.set_active_disease(d_id)

        elif act in ["set_exposure_concentration"]:
            conc = float(params.get("concentration_um", params.get("applied_concentration_um", 5.0)))
            return {"status": "applied", "exposure_concentration_um": conc}

        else:
            return {"status": "unsupported_action", "action": act, "payload": action}

    # -------------------------------------------------------------------------
    # Compare Worlds Statistical Evaluator
    # -------------------------------------------------------------------------

    def compare_worlds(
        self,
        scenario_left: dict[str, Any],
        scenario_right: dict[str, Any],
        steps: int = 60,
        repeats: int = 3,
        seed: int = 42,
    ) -> dict[str, Any]:
        """
        Compare two distinct simulation configurations (chemical or tumor)
        reporting mean +/- standard deviation across repeated random seeds.
        """
        left_results: list[dict[str, Any]] = []
        right_results: list[dict[str, Any]] = []

        is_tumor_comparison = (
            "compound" in scenario_left or "applied_concentration_um" in scenario_left
        )

        for rep in range(max(1, repeats)):
            rep_seed = seed + rep * 101

            if is_tumor_comparison:
                # Compare cellular tumor response across any disease
                c_left = scenario_left.get("compound", "Control")
                conc_left = float(scenario_left.get("applied_concentration_um", 0.0))
                dur = float(scenario_left.get("duration_hours", 48.0))
                d_id_left = scenario_left.get("disease_id", self.disease_id)

                mod_left = self.disease_registry.create(d_id_left, seed=rep_seed, nx=30, ny=24)
                rec_left = self.chemistry.resolve_compound_input(c_left if c_left != "Control" else "D-Glucose")
                adm_left = self.pharmacology.evaluate_admet(rec_left)
                adm_left_clean = dict(adm_left)
                if isinstance(adm_left_clean.get("fraction_unbound_plasma"), dict):
                    adm_left_clean["fraction_unbound_plasma"] = adm_left_clean["fraction_unbound_plasma"].get("value", 0.25)
                comp_left = mod_left.run_controlled_comparison(
                    compound=rec_left,
                    pharmacology_profile=adm_left_clean,
                    applied_concentration_um=conc_left,
                    duration_hours=dur,
                    seed=rep_seed,
                )
                left_results.append(comp_left["deltas"])

                c_right = scenario_right.get("compound", "Gemcitabine")
                conc_right = float(scenario_right.get("applied_concentration_um", 5.0))
                d_id_right = scenario_right.get("disease_id", self.disease_id)

                mod_right = self.disease_registry.create(d_id_right, seed=rep_seed, nx=30, ny=24)
                rec_right = self.chemistry.resolve_compound_input(c_right)
                adm_right = self.pharmacology.evaluate_admet(rec_right)
                adm_right_clean = dict(adm_right)
                if isinstance(adm_right_clean.get("fraction_unbound_plasma"), dict):
                    adm_right_clean["fraction_unbound_plasma"] = adm_right_clean["fraction_unbound_plasma"].get("value", 0.25)
                comp_right = mod_right.run_controlled_comparison(
                    compound=rec_right,
                    pharmacology_profile=adm_right_clean,
                    applied_concentration_um=conc_right,
                    duration_hours=dur,
                    seed=rep_seed,
                )
                right_results.append(comp_right["deltas"])

            else:
                # Compare Artificial Chemical Worlds
                # Run Branch Left
                eng_l = EVOCUREEngine(seed=rep_seed)
                eng_l.update_environment(scenario_left.get("environment", {}))
                for c_item in scenario_left.get("compounds", [{"query": "Glycine", "count": 20}]):
                    eng_l.add_compound_to_world(c_item.get("query", "Glycine"), count=c_item.get("count", 12))
                res_l = eng_l.step_universe(steps=steps)
                left_results.append(res_l)

                # Run Branch Right
                eng_r = EVOCUREEngine(seed=rep_seed)
                eng_r.update_environment(scenario_right.get("environment", {}))
                for c_item in scenario_right.get("compounds", [{"query": "Glycine", "count": 20}]):
                    eng_r.add_compound_to_world(c_item.get("query", "Glycine"), count=c_item.get("count", 12))
                res_r = eng_r.step_universe(steps=steps)
                right_results.append(res_r)

        # Compute mean +/- std across repeats
        def aggregate_metrics(res_list: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
            agg: dict[str, list[float]] = {}
            for item in res_list:
                for k, v in item.items():
                    if isinstance(v, (int, float)):
                        agg.setdefault(k, []).append(float(v))
            return {
                k: {
                    "mean": round(float(np.mean(vals)), 3),
                    "std": round(float(np.std(vals)), 3),
                }
                for k, vals in agg.items()
            }

        return {
            "comparison_type": f"{self.tumor_module.display_name} Cellular Microenvironment" if is_tumor_comparison else "Chemical Universe",
            "repeats": repeats,
            "base_seed": seed,
            "scenario_left_summary": aggregate_metrics(left_results),
            "scenario_right_summary": aggregate_metrics(right_results),
            "raw_repeats_left": left_results,
            "raw_repeats_right": right_results,
        }

    # -------------------------------------------------------------------------
    # Checkpointing & Experiment Persistence
    # -------------------------------------------------------------------------

    def create_checkpoint(self, label: str = "") -> dict[str, Any]:
        """Serialize complete engine state into SQLite checkpoint."""
        lbl = label or f"Checkpoint-Step-{self.clocks.evolutionary_step}"
        state_dict = self._serialize_runtime()
        chk_id = self.store.save_checkpoint(
            experiment_id=self.active_experiment_id,
            step=self.clocks.evolutionary_step,
            generation=self.clocks.evolutionary_generation,
            label=lbl,
            state_dict=state_dict,
        )
        return {"checkpoint_id": chk_id, "label": lbl, "step": self.clocks.evolutionary_step}

    def _serialize_runtime(self) -> dict[str, Any]:
        """Lossless local state, including random streams and continuous fields."""
        tumor = self.tumor_module
        tumor_attributes = {}
        for key, value in vars(tumor).items():
            if isinstance(value, np.ndarray):
                tumor_attributes[key] = {"array": value.tolist()}
            elif isinstance(value, (str, int, float, bool)) or value is None:
                tumor_attributes[key] = value
            elif key == "vessels":
                tumor_attributes[key] = copy.deepcopy(value)
        return {
            "clocks": self.clocks.to_dict(),
            "universe_mode": self.universe_mode.value,
            "interface_mode": self.interface_mode.value,
            "disease_id": self.disease_id,
            "seed": self.seed,
            "env": self.env.to_dict(),
            "entities": [asdict(e) for e in self.entities],
            "compounds": {cid: c.to_dict() for cid, c in self.compounds.items()},
            "genealogy": self.genealogy.to_dict(),
            "tumor_snapshot": self.tumor_module.get_snapshot(),
            "_entity_counter": self._entity_counter,
            "rng_state": self.rng.bit_generator.state,
            "grid_fields": {k: v.tolist() for k, v in self.grid.fields.items()},
            "reactions_count": self.reactions_count,
            "history": copy.deepcopy(self.history),
            "recent_events": copy.deepcopy(self.recent_events),
            "missions_progress": dict(self.mission_progress),
            "discovery_results": self.discovery_results,
            "tissue_events": self.tissue_events,
            "tissue_model_info": self.tissue_model_info,
            "tumor_runtime": {
                "attributes": tumor_attributes,
                "cells": [{**asdict(cell), "intracellular_drug_b": getattr(cell, "intracellular_drug_b", 0.0)} for cell in tumor.cells],
                "active_pharmacology": copy.deepcopy(tumor.active_pharmacology),
                "active_pharmacology_b": copy.deepcopy(tumor.active_pharmacology_b),
                "rng_state": tumor._rng.bit_generator.state,
                "active_compound": tumor.active_compound.to_dict() if tumor.active_compound else None,
                "active_compound_b": tumor.active_compound_b.to_dict() if tumor.active_compound_b else None,
            },
        }

    def restore_checkpoint(self, checkpoint_id: str) -> dict[str, Any]:
        """Restore engine state from saved SQLite checkpoint."""
        chk = self.store.load_checkpoint(checkpoint_id)
        st = chk.get("state", {})
        return self._restore_runtime(st, checkpoint_id)

    def _restore_runtime(self, st: dict[str, Any], checkpoint_id: str = "") -> dict[str, Any]:
        if not st:
            return {"status": "error", "message": "Empty checkpoint state payload."}

        self.seed = int(st.get("seed", self.seed))
        # Restore clocks
        clk_dict = st.get("clocks", {})
        self.clocks.computational_seconds = float(clk_dict.get("computational_seconds", 0.0))
        self.clocks.evolutionary_step = int(clk_dict.get("evolutionary_step", 0))
        self.clocks.evolutionary_generation = int(clk_dict.get("evolutionary_generation", 0))
        self.clocks.biological_seconds = float(clk_dict.get("biological_seconds", 0.0))

        # Restore modes and disease
        if "universe_mode" in st:
            self.universe_mode = UniverseMode(st["universe_mode"])
        if "interface_mode" in st:
            self.interface_mode = InterfaceMode(st["interface_mode"])
        if "disease_id" in st:
            self.set_active_disease(st["disease_id"], seed=self.seed)
            self.clocks.biological_seconds = float(clk_dict.get("biological_seconds", 0.0))
        if "_entity_counter" in st:
            self._entity_counter = int(st["_entity_counter"])

        # Restore environment
        if "env" in st:
            env_data = dict(st["env"])
            env_data["barriers"] = [CompartmentBarrier(**b) for b in env_data.get("barriers", [])]
            env_data["regions"] = [SpatialRegion(**r) for r in env_data.get("regions", [])]
            self.env = EnvironmentConfig(**env_data)

        # Restore compounds
        if "compounds" in st and isinstance(st["compounds"], dict):
            self.compounds.clear()
            for cid, c_data in st["compounds"].items():
                try:
                    valid_fields = {f.name for f in fields(CompoundRecord)}
                    values = {k: v for k, v in c_data.items() if k in valid_fields}
                    values["epistemic_category"] = EpistemicCategory(values.get("epistemic_category", EpistemicCategory.DEMONSTRATION_UNVALIDATED))
                    c_rec = CompoundRecord(**values)
                    self.compounds[cid] = c_rec
                except Exception:
                    pass

        # Restore entities
        if "entities" in st and isinstance(st["entities"], list):
            self.entities.clear()
            for e_data in st["entities"]:
                try:
                    aliases = {"id": "entity_id", "cid": "compound_id", "k": "kind", "r": "radius", "e": "energy", "gen": "generation", "fam": "family_id", "col": "color_hex", "shp": "shape", "role": "ecological_role", "bnd": "bound_to"}
                    valid_fields = {f.name for f in fields(SpatialEntity)}
                    values = {aliases.get(k, k): v for k, v in e_data.items() if aliases.get(k, k) in valid_fields}
                    values["kind"] = EntityKind(values.get("kind", "PROTOCELL"))
                    ent = SpatialEntity(**values)
                    self.entities.append(ent)
                except Exception:
                    pass

        self.rng = np.random.default_rng(self.seed)
        if st.get("rng_state"):
            self.rng.bit_generator.state = st["rng_state"]
        self.grid = ReactionDiffusionGrid2D(width=self.env.width, height=self.env.height, nx=50, ny=35)
        for key, value in st.get("grid_fields", {}).items():
            self.grid.fields[key] = np.array(value, dtype=float)
        self.substrate_adapter = SubstrateSourceAdapter(self.grid)
        self.genealogy = GenealogyTracker.from_dict(st.get("genealogy", {}))
        self.darwinian.genealogy_tracker = self.genealogy
        self.darwinian.compounds_registry = self.compounds
        self.reactions_count = st.get("reactions_count", 0)
        self.history = st.get("history", [])
        self.recent_events = st.get("recent_events", [])
        self.mission_progress.update(st.get("missions_progress", {}))
        self.discovery_results = st.get("discovery_results", {})
        self.tissue_events = st.get("tissue_events", [])
        self.tissue_model_info = st.get("tissue_model_info", {})
        self._reaction_cache.clear()
        runtime = st.get("tumor_runtime", {})
        for key, value in runtime.get("attributes", {}).items():
            setattr(self.tumor_module, key, np.array(value["array"], dtype=float) if isinstance(value, dict) and "array" in value else value)
        if "cells" in runtime:
            self.tumor_module.cells = []
            for cell in runtime["cells"]:
                cell = dict(cell)
                intracellular_b = cell.pop("intracellular_drug_b", 0.0)
                cell["cell_type"] = CellType(cell["cell_type"])
                cell["death_mode"] = CellDeathMode(cell["death_mode"])
                restored_cell = CellularEntity(**cell)
                restored_cell.intracellular_drug_b = intracellular_b
                self.tumor_module.cells.append(restored_cell)
            self.tumor_module._rng.bit_generator.state = runtime["rng_state"]
            for attr in ("active_compound", "active_compound_b"):
                value = runtime.get(attr)
                valid_fields = {f.name for f in fields(CompoundRecord)}
                if value:
                    values = {k: v for k, v in value.items() if k in valid_fields}
                    values["epistemic_category"] = EpistemicCategory(values.get("epistemic_category", EpistemicCategory.DEMONSTRATION_UNVALIDATED))
                    compound = CompoundRecord(**values)
                else:
                    compound = None
                setattr(self.tumor_module, attr, compound)
                profile_key = "active_pharmacology_b" if attr.endswith("_b") else "active_pharmacology"
                setattr(self.tumor_module, profile_key, runtime.get(profile_key, self.pharmacology.evaluate_admet(compound) if compound else {}))
        return {
            "status": "restored",
            "checkpoint_id": checkpoint_id,
            "step": self.clocks.evolutionary_step,
            "generation": self.clocks.evolutionary_generation,
            "entities_count": len(self.entities),
            "compounds_count": len(self.compounds),
        }

    def save_current_experiment(self, title: str = "") -> dict[str, Any]:
        """Save complete experiment session with all Section 20 fields to SQLite store."""
        t = title or f"EVOCURE Session ({self.universe_mode.value})"
        alive_count = len([e for e in self.entities if e.alive])

        results_summary = {
            "active_entities": alive_count,
            "total_compounds_registered": len(self.compounds),
            "reactions_executed": self.reactions_count,
            "history": self.history,
        }

        final_state = {
            "runtime_state": self._serialize_runtime(),
            "active_entities": alive_count,
            "compounds": {cid: c.to_dict() for cid, c in self.compounds.items()},
            "tumor_snapshot": self.tumor_module.get_snapshot(),
        }

        exp_id = self.store.save_experiment(
            experiment_id=self.active_experiment_id,
            title=t,
            simulator_version="1.0.0",
            universe_mode=self.universe_mode.value,
            interface_mode=self.interface_mode.value,
            disease_id=self.disease_id,
            seed=self.seed,
            config=self.env.to_dict(),
            parameters={"max_entities": 800, "dt": 0.25},
            dataset_version="1.0.0",
            model_versions=self.model_registry.get_ledger_snapshot().get("models"),
            duration_clocks=self.clocks.to_dict(),
            final_state=final_state,
            results_summary=results_summary,
            performance_metrics=self.model_registry.get_hardware_telemetry(),
            warnings_limitations=[
                "In silico research simulation; never constitutes human clinical guidance."
            ],
        )
        self.active_experiment_id = exp_id
        return {"experiment_id": exp_id, "title": t, "status": "saved"}

    def load_saved_experiment(self, experiment_id: str) -> dict[str, Any]:
        """Load an experiment from the persistence store."""
        exp = self.store.load_experiment(experiment_id)
        self.active_experiment_id = exp.get("experiment_id", self.active_experiment_id)
        if "seed" in exp:
            self.seed = int(exp["seed"])
        runtime = exp.get("final_state_json", {}).get("runtime_state")
        if not runtime:
            raise ValueError("This older experiment has no resumable world state. Load a checkpoint or create a new save.")
        self._restore_runtime(runtime)
        return exp

    # -------------------------------------------------------------------------
    # Full State Serialization (UI & WebSockets)
    # -------------------------------------------------------------------------

    def get_full_state(self, max_visible_particles: int = 1500) -> dict[str, Any]:
        """
        Return complete serializable snapshot for UI and WebSockets.
        Explicitly distinguishes total_entity_count vs visible_entity_count when subsampled!
        """
        alive_entities = [e for e in self.entities if e.alive]
        total_count = len(alive_entities)

        # Particle subsampling if total entities exceed max_visible_particles
        if total_count > max_visible_particles:
            visible_entities = alive_entities[:max_visible_particles]
        else:
            visible_entities = alive_entities

        tumor_snap = self.tumor_module.get_snapshot()
        tumor_summary = tumor_snap.get("summary_metrics") or tumor_snap.get("metrics") or {}

        return {
            "experiment_id": self.active_experiment_id,
            "seed": self.seed,
            "universe_mode": self.universe_mode.value,
            "interface_mode": self.interface_mode.value,
            "disease_id": self.disease_id,
            "clocks": self.clocks.to_dict(),
            "environment": self.env.to_dict(),
            "total_entity_count": total_count,
            "visible_entity_count": len(visible_entities),
            "is_subsampled": total_count > max_visible_particles,
            "entities": [e.to_render_dict() for e in visible_entities],
            "compounds_count": len(self.compounds),
            "reactions_count": self.reactions_count,
            "substrate_heatmap": self.grid.to_heatmap_list("substrate", downsample=2),
            "tumor": tumor_snap,
            "tumor_summary": tumor_summary,
            "tissue_events": self.tissue_events[-10:],
            "tissue_model_info": self.tissue_model_info,
            "history": self.history[-120:],
            "missions_progress": self.mission_progress,
            "unlocked_achievements": list(self.unlocked_achievements),
            "recent_events": self.recent_events[-10:],
            "recent_reactions": self.recent_events[-10:],
        }

    # -------------------------------------------------------------------------
    # Visual Styling & Achievement Helpers
    # -------------------------------------------------------------------------

    def _assign_entity_color(self, compound: CompoundRecord) -> str:
        """Assign accessible, scientifically meaningful color hex code by chemical family."""
        fam = compound.family_id.upper()
        if "AMINO" in fam or "PROTEIN" in fam:
            return "#38bdf8"  # Sky Blue
        elif "ACID" in fam or "CARBOXYL" in fam:
            return "#f87171"  # Coral Red
        elif "CATALYST" in fam or "IMIDAZOLE" in fam:
            return "#fbbf24"  # Amber Gold
        elif "NUCLEOTIDE" in fam or "PURINE" in fam:
            return "#a78bfa"  # Violet Purple
        elif "ALIPHATIC" in fam or "NUTRIENT" in fam:
            return "#34d399"  # Emerald Green
        elif "THERAPEUTIC" in fam or "ONCOLOGY" in fam:
            return "#ec4899"  # Pink / Magenta
        return "#94a3b8"      # Slate

    def _assign_entity_shape(self, compound: CompoundRecord) -> str:
        """Assign distinct accessible 2D shape for color-blind accessibility."""
        fam = compound.family_id.upper()
        if "AMINO" in fam:
            return "circle"
        elif "ACID" in fam:
            return "diamond"
        elif "CATALYST" in fam:
            return "hexagon"
        elif "NUCLEOTIDE" in fam:
            return "star"
        elif "THERAPEUTIC" in fam:
            return "ring"
        return "circle"

    def _check_achievement(self, achievement_id: str) -> None:
        """Unlock and log educational achievement milestone."""
        if achievement_id not in self.unlocked_achievements:
            self.unlocked_achievements.add(achievement_id)
            for a in self.achievements:
                if a.get("achievement_id") == achievement_id:
                    self.recent_events.append({
                        "type": "ACHIEVEMENT_UNLOCKED",
                        "title": a.get("title", achievement_id),
                        "description": a.get("description", ""),
                    })
                    break


def datetime_tag() -> str:
    """Return compact UTC date string."""
    return time.strftime("%Y%m%d", time.gmtime())
