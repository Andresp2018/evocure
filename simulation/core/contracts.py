"""
Fundamental scientific, architectural, and epistemic contracts for EVOCURE & EVOCURE PLAY.

Every module in EVOCURE (Chemistry, Physics, Evolution, Biology/PDAC, Pharmacology,
Toxicology, Systemic PBPK, Local AI, Persistence, and UI) communicates strictly
through these typed contracts.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class EpistemicCategory(str, Enum):
    """Mandatory scientific transparency classification (Section 27)."""

    EXPERIMENTAL_OBSERVATION = "Imported experimental observation"
    CALCULATED_PHYSICOCHEMICAL = "Calculated physicochemical property"
    VALIDATED_MODEL_PREDICTION = "Validated model prediction"
    MECHANISTIC_SIMULATION = "Mechanistic simulation result"
    EXPLORATORY_MODEL = "Exploratory model result"
    AI_HYPOTHESIS = "AI-generated hypothesis"
    DEMONSTRATION_UNVALIDATED = "Demonstrative result without external validation"
    INSUFFICIENT_DATA = "Insufficient data"
    UNMODELED = "Not modeled"


class UniverseMode(str, Enum):
    """Distinction between Artificial Chemical Life and Grounded Chemistry (Section 5.1)."""

    ARTIFICIAL_LIFE = "ARTIFICIAL_LIFE"       # Mode A: Artificial replicators & exploratory evolution
    GROUNDED_CHEMISTRY = "GROUNDED_CHEMISTRY" # Mode B: Chemically validated structures & pharmacology


class InterfaceMode(str, Enum):
    """Shared-engine presentation modes (EVOCURE PLAY vs EVOCURE LAB)."""

    PLAY = "PLAY"
    LAB = "LAB"


class EntityKind(str, Enum):
    """Types of entities in the 2D chemical universe."""

    NUTRIENT = "NUTRIENT"
    MOLECULE = "MOLECULE"
    CATALYST = "CATALYST"
    PROTOCELL = "PROTOCELL"
    AGGREGATE = "AGGREGATE"
    THERAPEUTIC = "THERAPEUTIC"


class CellType(str, Enum):
    """Cellular entities in the 2D tumor microenvironment (Section 8.1)."""

    TUMOR_SENSITIVE = "TUMOR_SENSITIVE"
    TUMOR_RESISTANT = "TUMOR_RESISTANT"
    EPITHELIAL_HEALTHY = "EPITHELIAL_HEALTHY"
    FIBROBLAST = "FIBROBLAST"
    CAF = "CAF"                    # Cancer-Associated Fibroblast (desmoplastic stroma)
    IMMUNE_T = "IMMUNE_T"          # Cytotoxic T lymphocyte
    MACROPHAGE = "MACROPHAGE"
    ENDOTHELIAL = "ENDOTHELIAL"    # Blood vessel lining


class CellDeathMode(str, Enum):
    """Cell state transitions."""

    ALIVE = "ALIVE"
    APOPTOSIS = "APOPTOSIS"
    NECROSIS = "NECROSIS"


@dataclass
class ScientificValue:
    """
    Epistemically transparent value container (Sections 11.3, 12, 27).
    Never emits naked clinical claims without provenance, uncertainty, and domain check.
    """

    name: str
    value: float | int | str | None
    unit: str = ""
    epistemic_category: EpistemicCategory = EpistemicCategory.EXPLORATORY_MODEL
    source: str = "EVOCURE Internal Engine"
    model_version: str = "0.1.0"
    confidence: float | None = None
    uncertainty_low: float | None = None
    uncertainty_high: float | None = None
    in_applicability_domain: bool = True
    validation_status: str = "Internal / exploratory calibration"
    limitations: str = "In silico computational estimate; does not constitute clinical evidence."

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["epistemic_category"] = self.epistemic_category.value
        data["epistemic_code"] = self.epistemic_category.name
        return data


@dataclass
class SimulationClocks:
    """
    Strictly separated clocks (Section 7.1):
    - computational_seconds: wall-clock execution time on the host CPU/GPU.
    - evolutionary_step / generation / virtual_years: artificial evolutionary time.
    - biological_seconds: mechanistic time for cellular & PK/PD dynamics.
    """

    computational_seconds: float = 0.0
    evolutionary_step: int = 0
    evolutionary_generation: int = 0
    steps_per_generation: int = 25
    generations_per_virtual_year: float = 10.0
    biological_seconds: float = 0.0
    is_calibrated_equivalence: bool = False
    calibration_note: str = (
        "Virtual years represent abstract evolutionary time in the artificial universe "
        "and do not directly equate to real-world chemical or clinical time."
    )

    @property
    def virtual_years(self) -> float:
        if self.generations_per_virtual_year <= 0:
            return 0.0
        total_gen = self.evolutionary_generation + (
            (self.evolutionary_step % max(1, self.steps_per_generation))
            / max(1, self.steps_per_generation)
        )
        return round(total_gen / self.generations_per_virtual_year, 3)

    @property
    def biological_hours(self) -> float:
        return round(self.biological_seconds / 3600.0, 3)

    @property
    def biological_days(self) -> float:
        return round(self.biological_seconds / 86400.0, 4)

    def to_dict(self) -> dict[str, Any]:
        return {
            "computational_seconds": round(self.computational_seconds, 3),
            "evolutionary_step": self.evolutionary_step,
            "evolutionary_generation": self.evolutionary_generation,
            "virtual_years": self.virtual_years,
            "biological_seconds": round(self.biological_seconds, 2),
            "biological_hours": self.biological_hours,
            "biological_days": self.biological_days,
            "is_calibrated_equivalence": self.is_calibrated_equivalence,
            "calibration_note": self.calibration_note,
        }


@dataclass
class CompartmentBarrier:
    """Semi-permeable spatial barrier in the 2D universe (Section 4.1)."""

    barrier_id: str
    x1: float
    y1: float
    x2: float
    y2: float
    permeability: float = 0.15  # 0.0 = impermeable wall, 1.0 = open boundary


@dataclass
class SpatialRegion:
    """Localized environmental niche in the 2D universe (Sections 4.1, 6)."""

    region_id: str
    name: str
    x: float
    y: float
    width: float
    height: float
    temperature_delta_k: float = 0.0
    ph_override: float | None = None
    oxygen_multiplier: float = 1.0
    energy_flux: float = 1.0


@dataclass
class EnvironmentConfig:
    """Physical and dimensionless parameters of the 2D universe (Section 4.1)."""

    width: float = 1000.0
    height: float = 700.0
    temperature_k: float = 310.15       # Physical unit: Kelvin (37 °C)
    ph: float = 7.4                     # pH scale (0 - 14)
    viscosity_pa_s: float = 0.00089     # Physical unit: Pa·s (water/plasma ~0.7-0.9 mPa·s)
    oxygen_norm: float = 1.0            # Normalized oxygen availability [0.0 - 2.0]
    energy_source_flux: float = 1.0     # Dimensionless simulation energy flux
    resource_influx_rate: float = 0.35  # Substrate replenishment rate (simulation units/step)
    resource_clearance_rate: float = 0.01
    current_vx: float = 0.0             # Advection drift X
    current_vy: float = 0.0             # Advection drift Y
    boundary_permeability: float = 0.0  # 0.0 = reflective closed box; >0 allows edge exchange
    use_physical_units: bool = True
    barriers: list[CompartmentBarrier] = field(default_factory=list)
    regions: list[SpatialRegion] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CompoundRecord:
    """
    Chemically validated or artificial compound entity definition (Sections 4.2, 6.2).
    """

    compound_id: str
    name: str
    common_name: str
    canonical_smiles: str
    formula: str
    mol_weight: float
    exact_mass: float = 0.0
    formal_charge: int = 0
    logp: float = 0.0
    logd_74: float = 0.0
    tpsa: float = 0.0
    hbd: int = 0
    hba: int = 0
    rotatable_bonds: int = 0
    heavy_atoms: int = 0
    qed: float = 0.0
    esol_log_s: float | None = None
    functional_groups: list[str] = field(default_factory=list)
    murcko_scaffold: str = ""
    inchi_key: str = ""
    family_id: str = "FAM-ORIGIN"
    family_name: str = "Founder Family"
    origin_generation: int = 0
    parent_ids: list[str] = field(default_factory=list)
    transformation_type: str = "seed"
    provenance: str = "EVOCURE Local Compound Library"
    epistemic_category: EpistemicCategory = EpistemicCategory.CALCULATED_PHYSICOCHEMICAL
    is_chemically_valid: bool = True
    svg_2d: str = ""
    atom_counts: dict[str, int] = field(default_factory=dict)
    artificial_traits: dict[str, float] = field(default_factory=dict)
    reference_bioactivity: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["epistemic_category"] = self.epistemic_category.value
        data["epistemic_code"] = self.epistemic_category.name
        return data


@dataclass
class SpatialEntity:
    """Active particle, molecule, aggregate, or artificial replicator in 2D space."""

    entity_id: str
    compound_id: str
    kind: EntityKind
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    radius: float = 5.0
    mass: float = 100.0
    charge: int = 0
    energy: float = 10.0
    stored_substrate: float = 0.0
    age: int = 0
    generation: int = 0
    lineage_id: str = ""
    family_id: str = "FAM-ORIGIN"
    color_hex: str = "#38bdf8"
    shape: str = "circle"  # circle, hexagon, diamond, ring, star, triangle (accessibility!)
    bound_to: str | None = None
    colony_id: str | None = None
    ecological_role: str = "autotroph"  # autotroph, catalyst, cooperator, parasite, neutral
    replication_threshold: float = 22.0
    decay_rate: float = 0.015
    catalytic_efficiency: float = 1.0
    stability_score: float = 0.8
    alive: bool = True

    def to_render_dict(self) -> dict[str, Any]:
        """Compact representation for high-FPS WebSocket streaming."""
        return {
            "id": self.entity_id,
            "cid": self.compound_id,
            "k": self.kind.value,
            "x": round(self.x, 1),
            "y": round(self.y, 1),
            "r": round(self.radius, 1),
            "e": round(self.energy, 2),
            "gen": self.generation,
            "fam": self.family_id,
            "col": self.color_hex,
            "shp": self.shape,
            "role": self.ecological_role,
            "bnd": self.bound_to,
        }


@dataclass
class GenealogyNode:
    """Node in the evolutionary & molecular lineage DAG (Sections 6.2, 18)."""

    node_id: str
    compound_id: str
    name: str
    canonical_smiles: str
    family_id: str
    family_name: str
    generation: int
    parent_ids: list[str] = field(default_factory=list)
    transformation_type: str = "founder"
    mutation_description: str = "Founder compound introduced into the universe."
    similarity_to_parent: float = 1.0
    birth_step: int = 0
    extinction_step: int | None = None
    extinction_cause: str | None = None
    current_abundance: int = 0
    peak_abundance: int = 0
    cumulative_births: int = 0
    persistence_steps: int = 0
    color_hex: str = "#38bdf8"
    properties_summary: dict[str, float] = field(default_factory=dict)

    @property
    def is_extinct(self) -> bool:
        return self.extinction_step is not None and self.current_abundance == 0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["is_extinct"] = self.is_extinct
        return data


@dataclass
class CellularEntity:
    """Individual cell agent in the 2D tumor microenvironment (Section 8.2)."""

    cell_id: str
    cell_type: CellType
    clone_id: str
    x: float
    y: float
    radius: float = 6.5
    cell_cycle_phase: str = "G1"       # G0, G1, S, G2_M
    division_progress: float = 0.0
    growth_rate: float = 0.025
    metabolic_state: str = "normoxic"  # normoxic, glycolytic_hypoxic, quiescent, stressed
    local_oxygen: float = 1.0
    local_nutrient: float = 1.0
    local_drug_conc: float = 0.0
    intracellular_drug: float = 0.0
    exposure_auc: float = 0.0
    mutations: list[str] = field(default_factory=list)
    resistance_score: float = 0.05     # [0.0 - 1.0]
    efflux_pump_level: float = 0.1
    stress_level: float = 0.0
    viability: float = 1.0
    death_mode: CellDeathMode = CellDeathMode.ALIVE
    generation: int = 0
    parent_clone_id: str | None = None

    def to_render_dict(self) -> dict[str, Any]:
        return {
            "id": self.cell_id,
            "t": self.cell_type.value,
            "cl": self.clone_id,
            "x": round(self.x, 1),
            "y": round(self.y, 1),
            "r": round(self.radius, 1),
            "o2": round(self.local_oxygen, 2),
            "drg": round(self.intracellular_drug, 3),
            "res": round(self.resistance_score, 2),
            "v": round(self.viability, 2),
            "st": self.death_mode.value,
            "mut": self.mutations,
            "cycle": round(self.division_progress, 4),
            "phase": self.cell_cycle_phase,
            "metabolic": self.metabolic_state,
            "local_drug": round(self.local_drug_conc, 4),
            "generation": self.generation,
            "stress": round(self.stress_level, 3),
        }


class DiseaseModule(abc.ABC):
    """
    Abstract extensible interface for biomedical disease models (Section 23).
    PDAC is the inaugural implementation; future cancers, infectious, metabolic,
    or neurodegenerative modules implement this contract.
    """

    @property
    @abc.abstractmethod
    def disease_id(self) -> str:
        """Unique identifier, e.g., 'PDAC'."""

    @property
    @abc.abstractmethod
    def display_name(self) -> str:
        """Human-readable English disease name."""

    @abc.abstractmethod
    def metadata(self) -> dict[str, Any]:
        """Return biological context, entities, biomarkers, assumptions, and limitations."""

    @abc.abstractmethod
    def initialize_scenario(self, seed: int, preset: str = "standard") -> None:
        """Initialize spatial microenvironment, cellular populations, and fields."""

    @abc.abstractmethod
    def apply_compound_exposure(
        self,
        compound: CompoundRecord,
        pharmacology_profile: dict[str, Any],
        applied_concentration_um: float,
        injection_region: str = "vascular",
    ) -> dict[str, Any]:
        """Introduce a candidate compound into the microenvironment."""

    @abc.abstractmethod
    def step(self, dt_seconds: float) -> dict[str, Any]:
        """Advance cellular, reaction-diffusion, and pharmacodynamic state by dt_seconds."""

    @abc.abstractmethod
    def get_snapshot(self) -> dict[str, Any]:
        """Return serializable state for visualization, metrics, and checkpoints."""

    @abc.abstractmethod
    def get_biomarkers(self, systemic_state: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return disease-specific and general biological indicators with epistemic tags."""
