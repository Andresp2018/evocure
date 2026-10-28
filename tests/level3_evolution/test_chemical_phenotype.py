"""Survival must depend on the molecule, not on a number drifting beside it.

Before this map existed, the module deciding who lives contained zero references
to SMILES, molecular weight, logP or QED. Entities evolved four free-floating
numbers while their structures were mutated in parallel and never read back.
Replacing every molecule with a random label would have changed nothing.

These tests pin the map, and the chemical reasoning behind each trait.
"""

from __future__ import annotations

import pytest

from simulation.evolution.chemical_phenotype import (
    FALLBACK,
    apply_to_entity,
    phenotype_from_compound,
    phenotype_from_smiles,
)

BENZENE = "c1ccccc1"
HISTIDINE = "N[C@@H](Cc1c[nH]cn1)C(=O)O"
ANHYDRIDE = "CC(=O)OC(C)=O"
EPOXIDE = "C1OC1CCO"
ETHANOL = "CCO"
PACLITAXEL = (
    "CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1OC(=O)C(C(c1ccccc1)NC(=O)c1ccccc1)O)O)"
    "OC(=O)c1ccccc1)(CO4)OC(C)=O)O)C)OC(C)=O"
)


# ---------------------------------------------------------------------------
# The map must actually be a map
# ---------------------------------------------------------------------------

def test_same_structure_gives_the_same_traits():
    assert phenotype_from_smiles(BENZENE) == phenotype_from_smiles(BENZENE)


def test_different_structures_give_different_traits():
    """The failure this whole module exists to prevent."""
    a = phenotype_from_smiles(BENZENE)
    b = phenotype_from_smiles(HISTIDINE)

    assert a.catalytic_efficiency != b.catalytic_efficiency
    assert a.stability_score != b.stability_score
    assert a.replication_threshold != b.replication_threshold


def test_unparseable_structures_get_mediocre_traits_not_good_ones():
    """A molecule nobody can read must never be advantaged by accident."""
    broken = phenotype_from_smiles("not_a_molecule")
    assert broken == FALLBACK

    good = phenotype_from_smiles(HISTIDINE)
    assert broken.catalytic_efficiency < good.catalytic_efficiency
    assert broken.stability_score < good.stability_score


# ---------------------------------------------------------------------------
# Each trait must follow its stated chemical rationale
# ---------------------------------------------------------------------------

def test_catalytic_groups_raise_catalytic_efficiency():
    """Histidine's imidazole is the textbook catalytic residue; benzene is inert."""
    assert (phenotype_from_smiles(HISTIDINE).catalytic_efficiency
            > phenotype_from_smiles(BENZENE).catalytic_efficiency)


def test_hydrolysable_bonds_lower_stability_and_raise_decay():
    """An anhydride falls apart in water; an aromatic ring does not."""
    labile = phenotype_from_smiles(ANHYDRIDE)
    stable = phenotype_from_smiles(BENZENE)

    assert labile.stability_score < stable.stability_score
    assert labile.decay_rate > stable.decay_rate
    assert labile.labile_groups > 0


def test_ring_strain_lowers_stability():
    strained = phenotype_from_smiles(EPOXIDE)
    unstrained = phenotype_from_smiles(ETHANOL)
    assert strained.stability_score < unstrained.stability_score


def test_bigger_molecules_cost_more_to_copy():
    """Replication threshold is the energy a cell must bank before dividing."""
    small = phenotype_from_smiles(ETHANOL)
    large = phenotype_from_smiles(PACLITAXEL)

    assert large.replication_threshold > small.replication_threshold
    assert large.heavy_atoms > small.heavy_atoms


def test_aromatic_content_supports_stability():
    assert phenotype_from_smiles(BENZENE).aromatic_fraction == pytest.approx(1.0)
    assert phenotype_from_smiles(ETHANOL).aromatic_fraction == 0.0


# ---------------------------------------------------------------------------
# Bounds, so no structure can break the simulation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smiles", [
    BENZENE, HISTIDINE, ANHYDRIDE, EPOXIDE, ETHANOL, PACLITAXEL,
    "C", "O", "[Na+].[Cl-]", "C1=CC=C(C=C1)C(=O)O",
])
def test_every_trait_stays_in_a_usable_range(smiles):
    p = phenotype_from_smiles(smiles)
    assert 0.05 <= p.catalytic_efficiency <= 2.5
    assert 0.05 <= p.stability_score <= 0.98
    assert 0.001 <= p.decay_rate <= 0.12
    assert 6.0 <= p.replication_threshold <= 80.0


# ---------------------------------------------------------------------------
# The reasoning must be retained, not just the number
# ---------------------------------------------------------------------------

def test_every_trait_carries_its_rationale():
    p = phenotype_from_smiles(HISTIDINE)
    for trait in ("catalytic_efficiency", "stability_score", "decay_rate",
                  "replication_threshold"):
        assert trait in p.rationale
        assert p.rationale[trait]

    assert "imidazole" in p.rationale["catalytic_efficiency"]


def test_the_map_does_not_claim_to_be_biochemistry():
    payload = phenotype_from_smiles(BENZENE).to_dict()
    assert "not biochemistry" in payload["caveat"]
    assert "not calibrated" in payload["caveat"]


# ---------------------------------------------------------------------------
# Applying to an entity
# ---------------------------------------------------------------------------

def test_applying_to_an_entity_overwrites_its_traits():
    from simulation.core.contracts import EntityKind, SpatialEntity

    entity = SpatialEntity(entity_id="E1", compound_id="C1", kind=EntityKind.PROTOCELL,
                           x=0.0, y=0.0, catalytic_efficiency=1.0,
                           stability_score=0.8, decay_rate=0.015,
                           replication_threshold=22.0)
    phenotype = apply_to_entity(entity, HISTIDINE)

    assert entity.catalytic_efficiency == phenotype.catalytic_efficiency
    assert entity.stability_score == phenotype.stability_score
    assert entity.decay_rate == phenotype.decay_rate
    assert entity.replication_threshold == phenotype.replication_threshold
    # The defaults were replaced, not merely confirmed.
    assert entity.replication_threshold != 22.0


def test_compound_records_work_directly():
    from simulation.chemistry.engine import ChemistryEngine

    record = ChemistryEngine().build_compound(HISTIDINE)
    assert phenotype_from_compound(record) == phenotype_from_smiles(record.canonical_smiles)


# ---------------------------------------------------------------------------
# End to end: does the engine now select on chemistry?
# ---------------------------------------------------------------------------

def _population(chem, seeds, per_seed=12):
    import numpy as np
    from simulation.core.contracts import EntityKind, SpatialEntity

    rng = np.random.default_rng(7)
    registry, pop = {}, []
    index = 0
    for label, smiles in seeds.items():
        for _ in range(per_seed):
            record = chem.build_compound(smiles)
            registry[record.compound_id] = record
            entity = SpatialEntity(
                entity_id=f"E{index}", compound_id=record.compound_id,
                kind=EntityKind.PROTOCELL, x=float(rng.uniform(0, 400)),
                y=float(rng.uniform(0, 300)), energy=14.0, stored_substrate=6.0,
                lineage_id=label)
            apply_to_entity(entity, record.canonical_smiles)
            pop.append(entity)
            index += 1
    return registry, pop


def test_chemistry_alone_decides_who_survives():
    """No structural mutation, no trait drift. Only the phenotype map differs.

    A hydrolysable anhydride against a stable aromatic: if the map is wired in,
    the anhydride lineage is eliminated. If the molecule were still a label,
    both lineages would behave identically.
    """
    import numpy as np
    from simulation.chemistry.engine import ChemistryEngine
    from simulation.core.contracts import EnvironmentConfig
    from simulation.evolution.darwinian import DarwinianEcosystemEngine

    chem = ChemistryEngine()
    _, pop = _population(chem, {"stable": BENZENE, "labile": ANHYDRIDE})

    engine = DarwinianEcosystemEngine(mutation_rate=0.0, crowding_reference=150)
    rng = np.random.default_rng(7)
    env = EnvironmentConfig()

    for step in range(1, 81):
        result = engine.step(pop, environment=env, substrate_source=lambda x, y: 5.0,
                             dt=1.0, rng=rng, current_step=step)
        pop = [e for e in result["entities"] if e.alive]
        if not pop:
            break

    survivors = {label: sum(1 for e in pop if e.lineage_id == label)
                 for label in ("stable", "labile")}

    assert survivors["labile"] == 0, "the hydrolysable lineage should be eliminated"
    assert survivors["stable"] > 12, "the stable lineage should have grown"


def test_disabling_the_map_restores_structure_blind_behaviour():
    """With the map off, two different molecules are treated identically."""
    import numpy as np
    from simulation.chemistry.engine import ChemistryEngine
    from simulation.core.contracts import EnvironmentConfig
    from simulation.evolution.darwinian import DarwinianEcosystemEngine

    chem = ChemistryEngine()
    _, pop = _population(chem, {"stable": BENZENE, "labile": ANHYDRIDE})
    # Identical default traits: the structures differ, the entities do not.
    for entity in pop:
        entity.catalytic_efficiency = 1.0
        entity.stability_score = 0.8
        entity.decay_rate = 0.015
        entity.replication_threshold = 22.0

    engine = DarwinianEcosystemEngine(mutation_rate=0.0, chemical_phenotype=False,
                                      crowding_reference=150)
    rng = np.random.default_rng(7)
    env = EnvironmentConfig()
    for step in range(1, 41):
        result = engine.step(pop, environment=env, substrate_source=lambda x, y: 5.0,
                             dt=1.0, rng=rng, current_step=step)
        pop = [e for e in result["entities"] if e.alive]

    survivors = {label: sum(1 for e in pop if e.lineage_id == label)
                 for label in ("stable", "labile")}
    assert survivors["stable"] == survivors["labile"], (
        "without the map the two structures must be indistinguishable, which is "
        "exactly the defect the map removes"
    )


def test_crowding_throttles_uptake():
    """What the competition term actually does, and nothing more.

    It divides available substrate by a crowding factor. It is deliberately not
    asserted to cap the population: in these regimes reproduction is driven by
    banked energy rather than by uptake, so a crowded population still grows.
    Population regulation in this engine is weak and remains an open problem;
    see docs/SCIENTIFIC_LIMITATIONS_AND_EPISTEMICS.md.
    """
    import numpy as np
    from simulation.chemistry.engine import ChemistryEngine
    from simulation.core.contracts import EnvironmentConfig
    from simulation.evolution.darwinian import DarwinianEcosystemEngine

    chem = ChemistryEngine()
    env = EnvironmentConfig()
    stored = {}

    for reference in (5, 100000):
        _, pop = _population(chem, {"stable": BENZENE}, per_seed=20)
        for entity in pop:
            entity.stored_substrate = 0.0
            entity.energy = 5.0
        engine = DarwinianEcosystemEngine(mutation_rate=0.0, crowding_reference=reference)
        rng = np.random.default_rng(3)
        engine.step(pop, environment=env, substrate_source=lambda x, y: 5.0,
                    dt=1.0, rng=rng, current_step=1)
        stored[reference] = float(np.mean([e.stored_substrate for e in pop]))

    assert stored[5] < stored[100000], (
        "heavy crowding must reduce how much substrate each entity takes up"
    )


def test_crowding_can_be_switched_off():
    from simulation.evolution.darwinian import DarwinianEcosystemEngine

    assert DarwinianEcosystemEngine(crowding_reference=0).crowding_reference == 0
    assert DarwinianEcosystemEngine().crowding_reference > 0
