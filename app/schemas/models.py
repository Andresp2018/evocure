"""
Pydantic Request and Response Models for EVOCURE API v1.0.0.
"""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class InterfaceModeRequest(BaseModel):
    mode: Literal["PLAY", "LAB"] = Field(..., description="Target UI presentation mode")


class UniverseModeRequest(BaseModel):
    mode: Literal["ARTIFICIAL_LIFE", "GROUNDED_CHEMISTRY"] = Field(
        ..., description="Universe chemical grounding mode"
    )


class EnvironmentUpdateRequest(BaseModel):
    temperature_k: float | None = Field(None, ge=200.0, le=450.0)
    ph: float | None = Field(None, ge=0.0, le=14.0)
    viscosity_pa_s: float | None = Field(None, ge=0.0001, le=0.1)
    oxygen_norm: float | None = Field(None, ge=0.0, le=2.0)
    energy_source_flux: float | None = Field(None, ge=0.0, le=10.0)
    resource_influx_rate: float | None = Field(None, ge=0.0, le=5.0)
    resource_clearance_rate: float | None = Field(None, ge=0.0, le=1.0)
    current_vx: float | None = Field(None, ge=-10.0, le=10.0)
    current_vy: float | None = Field(None, ge=-10.0, le=10.0)
    boundary_permeability: float | None = Field(None, ge=0.0, le=1.0)


class ResetWorldRequest(BaseModel):
    preset: Literal[
        "chemical_life", "cancer_battle", "evolve_compounds", "surprise_me", "sandbox_empty"
    ] = Field("chemical_life", description="Initialization scenario preset")
    seed: int | None = Field(None, description="Optional deterministic PRNG seed")
    universe_mode: Literal["ARTIFICIAL_LIFE", "GROUNDED_CHEMISTRY"] | None = Field(
        None, description="Optional override of universe mode"
    )


class AddCompoundRequest(BaseModel):
    query: str = Field(..., description="Compound name, ID, or SMILES string")
    input_type: Literal["auto", "name", "smiles", "id"] = Field(
        "auto", description="Query resolution hint"
    )
    count: int = Field(12, ge=1, le=500, description="Number of instances to spawn")
    x: float | None = Field(None, description="Center X spawn coordinate")
    y: float | None = Field(None, description="Center Y spawn coordinate")
    entity_kind: str = Field(
        "PROTOCELL", description="EntityKind enum value (PROTOCELL, MOLECULE, NUTRIENT, CATALYST)"
    )
    ecological_role: str = Field(
        "autotroph", description="Ecological role (autotroph, catalyst, cooperator, parasite, neutral)"
    )


class ImportSDFRequest(BaseModel):
    sdf_text: str = Field(..., description="SDF or Molfile formatted text")
    spawn_in_world: bool = Field(False, description="Whether to immediately spawn parsed compounds")


class StepUniverseRequest(BaseModel):
    steps: int = Field(1, ge=1, le=1000, description="Number of simulation steps to advance")
    dt: float = Field(0.25, ge=0.01, le=2.0, description="Integration time-step delta")


class AdvancePresetRequest(BaseModel):
    preset: Literal[
        "10_gen",
        "100_gen",
        "1000_gen",
        "10000_gen",
        "1_yr",
        "10_yr",
        "100_yr",
        "1000_yr",
        "10000_yr",
    ] = Field("10_gen", description="Pre-configured multi-step evolutionary jump")


class PlaybackRequest(BaseModel):
    action: Literal["play", "pause", "step", "speed"] = Field(
        ..., description="Playback control command"
    )
    speed_multiplier: Literal[1, 10, 100] = Field(
        1, description="Simulation speed multiplier"
    )
    target: Literal["universe", "tumor", "both"] = Field(
        "universe", description="Target simulation domain to advance in real-time"
    )


class StepTumorRequest(BaseModel):
    steps: int = Field(1, ge=1, le=500, description="Number of cellular PDAC steps")
    dt_seconds: float = Field(
        900.0, ge=60.0, le=86400.0, description="Biological seconds per step (default 15 min)"
    )


class TumorIntroduceCompoundRequest(BaseModel):
    compound_query: str = Field(..., description="Compound identifier, name, or SMILES")
    applied_concentration_um: float = Field(
        5.0, ge=0.0, le=1000.0, description="Extracellular exposure concentration in micromolar"
    )
    injection_region: Literal["vascular", "periphery", "diffuse", "intratumoral", "uniform", "topical_interstitial"] = Field(
        "vascular", description="Delivery location / route in the microenvironment"
    )
    enable_systemic_pbpk: bool = Field(
        False, description="Whether to couple with 7-compartment systemic PBPK kinetics"
    )
    dose_mg: float = Field(100.0, ge=1.0, le=5000.0, description="Systemic bolus dose in mg")


class TumorControlledComparisonRequest(BaseModel):
    compound_query: str = Field(..., description="Compound to compare against untreated control")
    applied_concentration_um: float = Field(5.0, ge=0.001, le=500.0)
    duration_hours: float = Field(48.0, ge=1.0, le=168.0)
    seed: int | None = Field(None, description="Deterministic seed for identical comparison")


class TumorCombinationRequest(BaseModel):
    compound_a: str | None = Field(None, description="First combination partner compound")
    compound_a_query: str | None = Field(None, description="First combination partner compound")
    compound_b: str | None = Field(None, description="Second combination partner compound")
    compound_b_query: str | None = Field(None, description="Second combination partner compound")
    dose_a_um: float | None = Field(None, description="Dose A in micromolar")
    conc_a_um: float | None = Field(None, description="Dose A in micromolar")
    dose_b_um: float | None = Field(None, description="Dose B in micromolar")
    conc_b_um: float | None = Field(None, description="Dose B in micromolar")
    hours: float | None = Field(None, description="Simulation duration in hours")
    duration_hours: float | None = Field(None, description="Simulation duration in hours")
    seed: int | None = Field(None, description="Deterministic seed")
    injection_region: Literal["vascular", "periphery", "diffuse", "intratumoral", "uniform", "topical_interstitial"] = Field(
        "vascular", description="Delivery route"
    )
    run_4_arm_trial: bool = Field(True, description="Run Control vs A vs B vs A+B arms")


class CoevolutionRequest(BaseModel):
    n_cycles: int = Field(4, ge=1, le=12, description="Number of treatment-response cycles")
    hours_per_cycle: float = Field(24.0, ge=1.0, le=72.0)
    applied_concentration_um: float = Field(3.0, ge=0.01, le=50.0)


class DiscoveryRunRequest(BaseModel):
    seed_queries: list[str] | None = Field(
        None, description="Starting seed scaffold names or SMILES"
    )
    generations: int = Field(4, ge=1, le=20, description="Molecular GA optimization generations")
    population_size: int = Field(16, ge=4, le=64)
    top_k_cellular: int = Field(3, ge=1, le=10)


class SelectDiseaseRequest(BaseModel):
    disease_id: Literal["PDAC", "NSCLC_KRAS", "AMR_BACTERIAL", "METABOLIC_MASLD"] = Field(
        "PDAC", description="Active disease module identifier"
    )
    seed: int | None = Field(None, description="Optional seed for microenvironment reset")


class MolecularEditRequest(BaseModel):
    smiles: str = Field(..., description="Base SMILES string to modify")
    operation: str = Field(..., description="Operation: add_substituent, atom_swap, swap_atom, ring_transformation, bioisostere, add_ring")
    substituent: str | None = Field(None, description="Substituent identifier")
    param: str | None = Field(None, description="Substituent or atom parameter")
    target_element: str | None = Field(None, description="Target element for swap")
    position: int | None = Field(None, description="Optional atom position index")
    atom_idx: int | None = Field(None, description="Optional target atom index")
    register_in_world: bool = Field(False, description="Register and optionally spawn in live simulation")


class MolecularDiffRequest(BaseModel):
    query_a: str | None = Field(None, description="First molecule (parent / reference)")
    smiles_a: str | None = Field(None, description="First molecule SMILES")
    query_b: str | None = Field(None, description="Second molecule (mutant / variant)")
    smiles_b: str | None = Field(None, description="Second molecule SMILES")


class AssistantAskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question in English")
    provider: str | None = Field(
        None, description="Optional provider override ('auto_local', 'local_gemma', 'gemini', 'openai', 'deterministic')"
    )
    model: str | None = Field(None, description="Optional model override")


class AIProviderConfigRequest(BaseModel):
    provider: Literal["auto_local", "local_gemma", "gemini", "openai", "deterministic"] = Field(
        ..., description="Target AI provider to configure"
    )
    api_key: str | None = Field(None, description="API key (masked when stored, never logged)")
    model: str | None = Field(None, description="Model identifier (e.g. 'gemini-2.5-flash', 'gpt-4o-mini')")
    base_url: str | None = Field(None, description="Base URL for OpenAI-compatible endpoints")
    persist_to_dotenv: bool = Field(False, description="Whether to persist configuration to .env")


class AIProviderTestRequest(BaseModel):
    provider: Literal["auto_local", "local_gemma", "gemini", "openai", "deterministic"] | None = Field(
        None, description="Target provider to test connection (defaults to active provider)"
    )
    api_key: str | None = Field(None, description="Optional API key override for testing")
    model: str | None = Field(None, description="Optional model override for testing")
    base_url: str | None = Field(None, description="Optional base URL override for testing")


class EnvironmentPresetRequest(BaseModel):
    preset: Literal[
        "pancreatic_tissue",
        "bloodstream",
        "tumor_microenvironment",
        "oxidative_stress",
        "acidic_low_ph",
        "hypoxia_low_o2",
    ] = Field(..., description="Target biophysical environment preset")


class AssistantApproveActionRequest(BaseModel):
    action: dict[str, Any] = Field(..., description="Structured action payload proposed by AI")


class CompareWorldsRequest(BaseModel):
    scenario_left: str | dict[str, Any] = Field(..., description="Left world configuration or preset name")
    scenario_right: str | dict[str, Any] = Field(..., description="Right world configuration or preset name")
    steps: int = Field(60, ge=5, le=500)
    repeats: int = Field(3, ge=1, le=10)
    seed: int = Field(42)


class ExperimentSaveRequest(BaseModel):
    title: str = Field("", description="Human-readable title for saved experiment")


class CheckpointCreateRequest(BaseModel):
    label: str = Field("", description="Checkpoint bookmark label")


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "1.0.0"
    frontend_edition: str = "Godot 4.7 Bioluminescent + Web Dual Edition"
    uptime_seconds: float = 0.0
    mode: str = "PLAY"
    universe_mode: str = "ARTIFICIAL_LIFE"
    active_disease: str = "PDAC"
    engine_ready: bool = True
    ai_status: dict[str, Any] = Field(default_factory=dict)
    ai_runtime: dict[str, Any] = Field(default_factory=dict)
    hardware_status: dict[str, Any] = Field(default_factory=dict)


class GenericResponse(BaseModel):
    success: bool
    message: str
    data: Any = None
