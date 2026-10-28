"""
Darwinian Evolution and Emergent Chemical Life Engine (Mode A).

Implements Section 6 (Artificial Chemical Life):
- Purely emergent selection without artificial fitness increment per generation.
- Exact thermodynamic/mass conservation at binary division:
    child1.energy + child2.energy + division_overhead == parent.energy
    child1.stored_substrate + child2.stored_substrate == parent.stored_substrate
- 15 mechanistic evolutionary dynamics:
    1. Substrate resource uptake (catalytic efficiency)
    2. Basal metabolic maintenance cost (decay_rate)
    3. Environmental stress modulation (temperature/pH deviations buffered by stability)
    4. Cooperative cross-feeding boost near cooperator/catalyst neighbors
    5. Parasitic energy drain by parasite entities from nearby hosts
    6. Colony kin energy sharing
    7. Death/lysis recycling of stored mass back to environment
    8. Binary division threshold gating
    9. Exact mass and energy division conservation
    10. Heritable traits with stochastic mutation
    11. Molecular variation structure callback & genealogy registration
    12. Spatial dispersal on division
    13. Differential selection under substrate scarcity
    14. Neutral genetic drift under identical traits
    15. Extinction and lineage abundance logging in GenealogyTracker
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Callable
import numpy as np

from simulation.core.contracts import (
    CompoundRecord,
    EnvironmentConfig,
    EntityKind,
    SpatialEntity,
)
from simulation.evolution.chemical_phenotype import phenotype_from_compound
from simulation.evolution.genealogy import GenealogyTracker


class DarwinianEcosystemEngine:
    """
    Mode A Artificial Chemical Life Evolutionary Engine.
    
    Evolves spatial populations of chemical replicators mechanistically
    under mass and energy conservation, resource competition, environmental
    niches, and cooperative/parasitic interactions.
    """

    def __init__(
        self,
        genealogy_tracker: GenealogyTracker | None = None,
        mutation_rate: float = 0.05,
        division_overhead: float = 2.0,
        min_replication_substrate: float = 4.0,
        base_uptake_rate: float = 1.5,
        base_metabolic_rate: float = 1.0,
        substrate_energy_yield: float = 2.0,
        recycling_fraction: float = 0.75,
        cooperation_radius: float = 25.0,
        cooperation_boost: float = 0.25,
        parasite_radius: float = 20.0,
        parasite_drain_rate: float = 1.0,
        parasite_efficiency: float = 0.8,
        colony_radius: float = 30.0,
        colony_sharing_rate: float = 0.1,
        optimal_temperature_k: float = 310.15,
        temp_tolerance: float = 10.0,
        optimal_ph: float = 7.4,
        ph_tolerance: float = 1.0,
        mutate_structure_fn: Callable[[CompoundRecord, np.random.Generator], CompoundRecord] | None = None,
        compounds_registry: dict[str, CompoundRecord] | None = None,
        chemical_phenotype: bool = True,
        crowding_reference: int = 1200,
    ) -> None:
        self.genealogy_tracker = genealogy_tracker
        self.mutation_rate = float(mutation_rate)
        self.division_overhead = float(division_overhead)
        self.min_replication_substrate = float(min_replication_substrate)
        self.base_uptake_rate = float(base_uptake_rate)
        self.base_metabolic_rate = float(base_metabolic_rate)
        self.substrate_energy_yield = float(substrate_energy_yield)
        self.recycling_fraction = float(recycling_fraction)
        self.cooperation_radius = float(cooperation_radius)
        self.cooperation_boost = float(cooperation_boost)
        self.parasite_radius = float(parasite_radius)
        self.parasite_drain_rate = float(parasite_drain_rate)
        self.parasite_efficiency = float(parasite_efficiency)
        self.colony_radius = float(colony_radius)
        self.colony_sharing_rate = float(colony_sharing_rate)
        self.optimal_temperature_k = float(optimal_temperature_k)
        self.temp_tolerance = float(temp_tolerance)
        self.optimal_ph = float(optimal_ph)
        self.ph_tolerance = float(ph_tolerance)
        self.mutate_structure_fn = mutate_structure_fn
        self.compounds_registry = compounds_registry if compounds_registry is not None else {}
        # Survival traits derived from the molecule rather than drifting on their
        # own. Without this the structure is a label and selection acts on noise.
        self.chemical_phenotype = bool(chemical_phenotype)
        # Population scale at which crowding starts to bite. This is NOT a hard
        # ceiling: the equilibrium population depends on substrate supply and on
        # the traits in play, and with a generous supply it settles well above
        # this number. Set it to 0 to disable competition entirely, which
        # restores the old behaviour of unbounded exponential growth.
        self.crowding_reference = max(0, int(crowding_reference))

        # With traits derived from structure, a structural mutation is the only
        # thing that can change them. Without a way to produce one there is no
        # heritable variation at all, so the ecosystem cannot evolve. That is the
        # correct behaviour, and silently inert is the wrong way to deliver it.
        if self.chemical_phenotype and mutate_structure_fn is None:
            warnings.warn(
                "DarwinianEcosystemEngine: chemical_phenotype is on but no "
                "mutate_structure_fn was supplied. Survival traits are derived "
                "from the molecule, so with no way to alter the molecule there is "
                "no heritable variation and the population cannot evolve. Supply "
                "mutate_structure_fn, or pass chemical_phenotype=False for the "
                "legacy numeric drift behaviour.",
                RuntimeWarning, stacklevel=2,
            )

        # Ecological counters
        self.total_births: int = 0
        self.total_deaths: int = 0
        self.recycled_substrate_pool: float = 0.0

    # -------------------------------------------------------------------------
    # Spatial Interaction Pre-Processing
    # -------------------------------------------------------------------------

    def _compute_spatial_interactions(
        self,
        entities: list[SpatialEntity],
        dt: float,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """
        Compute cooperative boosts, parasitic energy drain, and colony energy sharing.
        Returns:
            cooperative_boosts: {entity_id: boost_multiplier}
            parasitic_energy_deltas: {entity_id: energy_delta}
        """
        coop_boosts: dict[str, float] = {e.entity_id: 0.0 for e in entities}
        energy_deltas: dict[str, float] = {e.entity_id: 0.0 for e in entities}

        n = len(entities)
        if n < 2:
            return coop_boosts, energy_deltas

        # Extract coordinates for spatial neighborhood search
        coords = np.array([[e.x, e.y] for e in entities], dtype=float)

        for i in range(n):
            ei = entities[i]
            if not ei.alive:
                continue

            for j in range(i + 1, n):
                ej = entities[j]
                if not ej.alive:
                    continue

                dx = coords[i, 0] - coords[j, 0]
                dy = coords[i, 1] - coords[j, 1]
                dist_sq = dx * dx + dy * dy

                # 1. Cooperative cross-feeding boost
                if dist_sq <= self.cooperation_radius * self.cooperation_radius:
                    if ei.ecological_role in ("cooperator", "catalyst"):
                        coop_boosts[ej.entity_id] += self.cooperation_boost
                    if ej.ecological_role in ("cooperator", "catalyst"):
                        coop_boosts[ei.entity_id] += self.cooperation_boost

                # 2. Parasitic energy drain
                if dist_sq <= self.parasite_radius * self.parasite_radius:
                    # Case A: ei is parasite, ej is non-parasite host
                    if ei.ecological_role == "parasite" and ej.ecological_role != "parasite":
                        drain = min(max(0.0, ej.energy * 0.1), self.parasite_drain_rate * dt)
                        energy_deltas[ej.entity_id] -= drain
                        energy_deltas[ei.entity_id] += drain * self.parasite_efficiency
                    # Case B: ej is parasite, ei is non-parasite host
                    elif ej.ecological_role == "parasite" and ei.ecological_role != "parasite":
                        drain = min(max(0.0, ei.energy * 0.1), self.parasite_drain_rate * dt)
                        energy_deltas[ei.entity_id] -= drain
                        energy_deltas[ej.entity_id] += drain * self.parasite_efficiency

                # 3. Colony kin energy sharing
                if (
                    ei.colony_id is not None
                    and ei.colony_id == ej.colony_id
                    and dist_sq <= self.colony_radius * self.colony_radius
                ):
                    e_diff = ei.energy - ej.energy
                    transfer = e_diff * self.colony_sharing_rate * dt * 0.5
                    energy_deltas[ei.entity_id] -= transfer
                    energy_deltas[ej.entity_id] += transfer

        return coop_boosts, energy_deltas

    # -------------------------------------------------------------------------
    # Core Evolutionary Simulation Step
    # -------------------------------------------------------------------------

    def step(
        self,
        entities: list[SpatialEntity],
        environment: EnvironmentConfig | None = None,
        substrate_source: Any = None,
        dt: float = 1.0,
        rng: np.random.Generator | None = None,
        current_step: int = 0,
    ) -> dict[str, Any]:
        """
        Advance the Darwinian ecosystem by dt time units.
        
        Args:
            entities: Active population of SpatialEntity replicators.
            environment: Physical environment config (temperature, pH, boundaries).
            substrate_source: Substrate availability callback, 2D array, or scalar.
            dt: Time step duration.
            rng: NumPy random generator.
            current_step: Current simulation step index.
            
        Returns:
            Dictionary with surviving entities, ecological metrics, and event counts.
        """
        if rng is None:
            rng = np.random.default_rng(current_step + 42)

        # Environmental parameters
        env_width = environment.width if environment else 1000.0
        env_height = environment.height if environment else 700.0
        temp_k = environment.temperature_k if environment else self.optimal_temperature_k
        ph_val = environment.ph if environment else self.optimal_ph

        # Environmental stress factor
        temp_dev = abs(temp_k - self.optimal_temperature_k) / self.temp_tolerance
        ph_dev = abs(ph_val - self.optimal_ph) / self.ph_tolerance
        base_stress = temp_dev + ph_dev

        # 1. Neighbor interactions (cooperation, parasitism, colony sharing)
        living_entities = [e for e in entities if e.alive]
        coop_boosts, energy_deltas = self._compute_spatial_interactions(living_entities, dt)

        # Logistic crowding. At the carrying capacity maintenance costs double, so
        # the weakest metabolisms stop paying their way and the population settles
        # instead of growing without limit.
        # Quadratic, because a linear term is too weak to bind: a lineage with a
        # low decay rate simply outruns it and the population overshoots the
        # ceiling several fold, which makes "carrying capacity" a misnomer.
        crowding = 1.0
        if self.crowding_reference > 0:
            load = len(living_entities) / float(self.crowding_reference)
            crowding = 1.0 + load * load

        # 2. Resource uptake, metabolism, maintenance, and interaction resolution
        step_deaths: int = 0
        step_recycled_substrate: float = 0.0

        for entity in living_entities:
            # Apply parasitic drain / colony sharing
            entity.energy += energy_deltas.get(entity.entity_id, 0.0)

            # A. Determine available substrate
            avail_substrate: float = 10.0
            if callable(substrate_source):
                avail_substrate = float(substrate_source(entity.x, entity.y))
            elif isinstance(substrate_source, (int, float)):
                avail_substrate = float(substrate_source)
            elif hasattr(substrate_source, "__getitem__"):
                # 2D Grid
                try:
                    gx = int(np.clip(entity.x / env_width * substrate_source.shape[1], 0, substrate_source.shape[1] - 1))
                    gy = int(np.clip(entity.y / env_height * substrate_source.shape[0], 0, substrate_source.shape[0] - 1))
                    avail_substrate = float(substrate_source[gy, gx])
                except Exception:
                    avail_substrate = 10.0

            # B. Resource uptake, shared out as the population grows.
            # Competition has to act here rather than on maintenance: uptake is
            # two orders of magnitude larger than the decay cost, so throttling
            # maintenance alone leaves growth effectively unbounded.
            coop_mult = 1.0 + min(2.0, coop_boosts.get(entity.entity_id, 0.0))
            max_uptake = self.base_uptake_rate * entity.catalytic_efficiency * coop_mult * dt
            actual_uptake = min(avail_substrate / crowding, max_uptake)
            entity.stored_substrate += actual_uptake

            # Deduct substrate from source if callback supports it
            if callable(getattr(substrate_source, "deduct", None)):
                substrate_source.deduct(entity.x, entity.y, actual_uptake)

            # C. Intracellular Metabolism (converting stored substrate to energy)
            # Biologically regulated: when energy reaches replication threshold + buffer,
            # entity conserves stored substrate to accumulate the biomass required for division.
            needed_energy = max(0.0, (entity.replication_threshold + 1.0) - entity.energy)
            needed_substrate = needed_energy / max(0.1, self.substrate_energy_yield * entity.stability_score)
            metabolic_capacity = self.base_metabolic_rate * entity.catalytic_efficiency * dt
            metabolized = min(entity.stored_substrate, metabolic_capacity, needed_substrate)
            entity.stored_substrate -= metabolized
            entity.energy += metabolized * self.substrate_energy_yield * entity.stability_score

            # D. Basal Metabolic Maintenance Cost (modulated by environmental stress)
            # High stability score buffers against temperature and pH stress
            stress_buffer = max(0.1, 1.0 - 0.5 * entity.stability_score)
            stress_mult = 1.0 + base_stress * stress_buffer
            # Crowding raises the cost of staying alive, so the population finds a
            # ceiling and entities compete instead of all surviving together.
            maintenance_cost = entity.decay_rate * stress_mult * crowding * dt
            entity.energy -= maintenance_cost

            entity.age += 1

            # E. Lysis check: starvation / death
            if entity.energy <= 0.0:
                entity.alive = False
                step_deaths += 1
                # Death/lysis mass recycling
                recycled = (
                    entity.stored_substrate * self.recycling_fraction
                    + (entity.mass * 0.05)
                )
                step_recycled_substrate += recycled
                self.recycled_substrate_pool += recycled

        self.total_deaths += step_deaths

        # 3. Binary Division & Conservation
        surviving_entities: list[SpatialEntity] = []
        step_births: int = 0

        for entity in living_entities:
            if not entity.alive:
                continue

            # Binary Division Threshold Condition
            if (
                entity.energy >= entity.replication_threshold
                and entity.stored_substrate >= self.min_replication_substrate
            ):
                step_births += 2
                self.total_births += 2

                # Exact Division Conservation:
                # child1.energy + child2.energy + overhead == parent.energy
                # child1.stored_substrate + child2.stored_substrate == parent.stored_substrate
                overhead = min(self.division_overhead, entity.energy * 0.5)
                usable_energy = entity.energy - overhead
                c1_energy = usable_energy * 0.5
                c2_energy = usable_energy * 0.5

                c1_substrate = entity.stored_substrate * 0.5
                c2_substrate = entity.stored_substrate * 0.5

                c1_mass = entity.mass * 0.5
                c2_mass = entity.mass * 0.5

                # Spatial dispersal
                disp_dist = max(3.0, entity.radius * 1.5)
                angle = float(rng.uniform(0, 2 * math.pi))
                dx = disp_dist * math.cos(angle)
                dy = disp_dist * math.sin(angle)

                c1_x = float(np.clip(entity.x - dx, entity.radius, env_width - entity.radius))
                c1_y = float(np.clip(entity.y - dy, entity.radius, env_height - entity.radius))
                c2_x = float(np.clip(entity.x + dx, entity.radius, env_width - entity.radius))
                c2_y = float(np.clip(entity.y + dy, entity.radius, env_height - entity.radius))

                # Daughter cells trait inheritance & stochastic mutation
                for c_idx, (cx, cy, c_energy, c_sub, c_m) in enumerate([
                    (c1_x, c1_y, c1_energy, c1_substrate, c1_mass),
                    (c2_x, c2_y, c2_energy, c2_substrate, c2_mass),
                ]):
                    child_id = f"{entity.entity_id}-d{entity.generation + 1}-{c_idx}"
                    child_cat = entity.catalytic_efficiency
                    child_stab = entity.stability_score
                    child_rep = entity.replication_threshold
                    child_decay = entity.decay_rate
                    child_role = entity.ecological_role
                    child_cid = entity.compound_id
                    child_fam = entity.family_id
                    child_color = entity.color_hex

                    # Stochastic Mutation
                    is_mutated = rng.random() < self.mutation_rate
                    if is_mutated and not self.chemical_phenotype:
                        # Legacy path: traits drift on their own, detached from any
                        # structure. Retained so the two regimes can be compared.
                        child_cat = max(0.05, child_cat + float(rng.normal(0, 0.05)))
                        child_stab = float(np.clip(child_stab + rng.normal(0, 0.05), 0.1, 1.0))
                        child_rep = max(5.0, child_rep + float(rng.normal(0, 0.5)))
                        child_decay = max(0.001, child_decay + float(rng.normal(0, 0.002)))
                    if is_mutated:

                        # Ecological role mutation (rare)
                        if rng.random() < 0.15:
                            roles = ["autotroph", "catalyst", "cooperator", "parasite"]
                            child_role = str(rng.choice(roles))

                        # Structural mutation callback if configured
                        if self.mutate_structure_fn and self.compounds_registry:
                            parent_comp = self.compounds_registry.get(entity.compound_id)
                            if parent_comp is not None:
                                new_comp = self.mutate_structure_fn(parent_comp, rng)
                                self.compounds_registry[new_comp.compound_id] = new_comp
                                child_cid = new_comp.compound_id
                                child_fam = new_comp.family_id

                                # The whole point of a chemical evolution engine: the
                                # new structure determines the child's survival traits.
                                # Without this the molecule is a label and selection
                                # acts on drifting numbers that mean nothing.
                                if self.chemical_phenotype:
                                    phenotype = phenotype_from_compound(new_comp)
                                    child_cat = phenotype.catalytic_efficiency
                                    child_stab = phenotype.stability_score
                                    child_rep = phenotype.replication_threshold
                                    child_decay = phenotype.decay_rate

                                if self.genealogy_tracker:
                                    self.genealogy_tracker.register_compound(
                                        compound=new_comp,
                                        birth_step=current_step,
                                        similarity_to_parent=0.9,
                                        mutation_description=f"Structural mutation from {entity.compound_id}",
                                        color_hex=child_color,
                                    )

                    child_entity = SpatialEntity(
                        entity_id=child_id,
                        compound_id=child_cid,
                        kind=entity.kind,
                        x=cx,
                        y=cy,
                        vx=float(rng.normal(0, 0.1)),
                        vy=float(rng.normal(0, 0.1)),
                        radius=entity.radius,
                        mass=c_m,
                        charge=entity.charge,
                        energy=c_energy,
                        stored_substrate=c_sub,
                        age=0,
                        generation=entity.generation + 1,
                        lineage_id=entity.lineage_id or entity.entity_id,
                        family_id=child_fam,
                        color_hex=child_color,
                        shape=entity.shape,
                        colony_id=entity.colony_id,
                        ecological_role=child_role,
                        replication_threshold=child_rep,
                        decay_rate=child_decay,
                        catalytic_efficiency=child_cat,
                        stability_score=child_stab,
                        alive=True,
                    )
                    surviving_entities.append(child_entity)

            else:
                surviving_entities.append(entity)

        # 4. Abundances & Lineage Tracking
        compound_counts: dict[str, int] = {}
        for e in surviving_entities:
            compound_counts[e.compound_id] = compound_counts.get(e.compound_id, 0) + 1

        if self.genealogy_tracker:
            self.genealogy_tracker.update_abundances(
                counts_by_compound=compound_counts,
                current_step=current_step,
                default_extinction_cause="competitive_exclusion_or_decay",
            )

        # 5. Ecological Metrics
        metrics = self.compute_ecological_metrics(surviving_entities)
        metrics.update({
            "step": current_step,
            "births": step_births,
            "deaths": step_deaths,
            "recycled_substrate": float(round(step_recycled_substrate, 3)),
            "entities": surviving_entities,
        })
        return metrics

    # -------------------------------------------------------------------------
    # Ecological Diversity & Population Metrics
    # -------------------------------------------------------------------------

    def compute_ecological_metrics(self, entities: list[SpatialEntity]) -> dict[str, Any]:
        """
        Compute Shannon diversity, Simpson index, family abundances,
        and ecological niche occupancy across living entities.
        """
        living = [e for e in entities if e.alive]
        total = len(living)

        if total == 0:
            return {
                "active_entities": 0,
                "shannon_diversity": 0.0,
                "simpson_diversity": 0.0,
                "family_abundances": {},
                "niche_occupancy": {},
                "mean_energy": 0.0,
                "mean_catalytic_efficiency": 0.0,
                "mean_stability_score": 0.0,
                "mean_generation": 0.0,
            }

        # Compound frequencies
        cid_counts: dict[str, int] = {}
        fam_counts: dict[str, int] = {}
        role_counts: dict[str, int] = {}

        for e in living:
            cid_counts[e.compound_id] = cid_counts.get(e.compound_id, 0) + 1
            fam_counts[e.family_id] = fam_counts.get(e.family_id, 0) + 1
            role_counts[e.ecological_role] = role_counts.get(e.ecological_role, 0) + 1

        # Shannon diversity: H = - sum(p * ln(p))
        proportions = [c / total for c in cid_counts.values()]
        shannon = float(-sum(p * np.log(p) for p in proportions if p > 0))

        # Simpson diversity: D = 1 - sum(p^2)
        simpson = float(1.0 - sum(p * p for p in proportions))

        return {
            "active_entities": total,
            "shannon_diversity": float(round(shannon, 4)),
            "simpson_diversity": float(round(simpson, 4)),
            "family_abundances": fam_counts,
            "niche_occupancy": role_counts,
            "mean_energy": float(round(np.mean([e.energy for e in living]), 3)),
            "mean_catalytic_efficiency": float(round(np.mean([e.catalytic_efficiency for e in living]), 4)),
            "mean_stability_score": float(round(np.mean([e.stability_score for e in living]), 4)),
            "mean_generation": float(round(np.mean([e.generation for e in living]), 2)),
        }
