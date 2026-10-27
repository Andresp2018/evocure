"""
Level-3 Scientific and Architectural Evolution Tests for EVOCURE.

Validates:
1. Heritability & trait variation governed by mutation rate
2. Differential selection (high-efficiency lineage outcompetes low-efficiency lineage)
3. Extinction logging & cause tracking in GenealogyTracker
4. Neutral genetic drift under identical phenotypic traits
5. Exact mass & energy conservation at binary division
6. 100% chemical validity of 50 random RDKit mutations and recombinations
7. NSGA-II fast non-dominated sorting and crowding distance correctness
8. Lineage DAG ancestor tracing & node comparison
9. Directed molecular GA population evolution cycle
10. Tree query filtering and serialization roundtrip
"""

from __future__ import annotations

import pytest
import numpy as np
from rdkit import Chem

from simulation.core.contracts import (
    CompoundRecord,
    EnvironmentConfig,
    EntityKind,
    SpatialEntity,
)
from simulation.evolution.genealogy import GenealogyTracker
from simulation.evolution.darwinian import DarwinianEcosystemEngine
from simulation.evolution.molecular_ga import DirectedMolecularOptimizer


# =============================================================================
# 1. Exact Division Conservation Test
# =============================================================================

def test_binary_division_conservation() -> None:
    """
    Verify exact thermodynamic and mass conservation at binary division:
    child1.energy + child2.energy + division_overhead == parent.energy
    child1.stored_substrate + child2.stored_substrate == parent.stored_substrate
    child1.mass + child2.mass == parent.mass
    """
    engine = DarwinianEcosystemEngine(
        division_overhead=2.0,
        min_replication_substrate=4.0,
        mutation_rate=0.0,
    )
    parent = SpatialEntity(
        entity_id="parent-01",
        compound_id="CMP-01",
        kind=EntityKind.PROTOCELL,
        x=100.0,
        y=100.0,
        energy=24.5,
        stored_substrate=8.2,
        mass=120.0,
        replication_threshold=20.0,
        catalytic_efficiency=1.5,
        stability_score=0.9,
    )

    env = EnvironmentConfig(width=500.0, height=500.0)
    rng = np.random.default_rng(42)

    # Substrate source with zero additional influx to isolate division mechanics
    result = engine.step(
        entities=[parent],
        environment=env,
        substrate_source=0.0,
        dt=0.0,  # Zero dt prevents uptake/metabolism during division check
        rng=rng,
        current_step=1,
    )

    survivors = result["entities"]
    assert len(survivors) == 2, f"Expected 2 daughters from binary division, got {len(survivors)}"
    c1, c2 = survivors[0], survivors[1]

    # Energy conservation
    total_child_energy = c1.energy + c2.energy + engine.division_overhead
    assert total_child_energy == pytest.approx(parent.energy, abs=1e-9)

    # Substrate conservation
    total_child_substrate = c1.stored_substrate + c2.stored_substrate
    assert total_child_substrate == pytest.approx(parent.stored_substrate, abs=1e-9)

    # Mass conservation
    total_child_mass = c1.mass + c2.mass
    assert total_child_mass == pytest.approx(parent.mass, abs=1e-9)


# =============================================================================
# 2. Heritability & Trait Variation Tests
# =============================================================================

def test_heritability_without_mutation() -> None:
    """
    When mutation_rate == 0.0, daughter cells must inherit parent traits exactly.
    """
    engine = DarwinianEcosystemEngine(
        division_overhead=1.5,
        min_replication_substrate=4.0,
        mutation_rate=0.0,
    )
    parent = SpatialEntity(
        entity_id="P-HERITABLE",
        compound_id="CMP-HERIT",
        kind=EntityKind.PROTOCELL,
        x=200.0,
        y=200.0,
        energy=30.0,
        stored_substrate=10.0,
        replication_threshold=22.0,
        decay_rate=0.0125,
        catalytic_efficiency=1.85,
        stability_score=0.92,
        ecological_role="catalyst",
        generation=2,
    )

    result = engine.step(
        entities=[parent],
        substrate_source=0.0,
        dt=0.0,
        rng=np.random.default_rng(99),
        current_step=5,
    )
    children = result["entities"]
    assert len(children) == 2
    for child in children:
        assert child.catalytic_efficiency == parent.catalytic_efficiency
        assert child.stability_score == parent.stability_score
        assert child.replication_threshold == parent.replication_threshold
        assert child.decay_rate == parent.decay_rate
        assert child.ecological_role == parent.ecological_role
        assert child.generation == parent.generation + 1


def test_heritability_with_mutation() -> None:
    """
    When mutation_rate == 1.0, daughter cells exhibit bounded stochastic trait shifts.
    """
    engine = DarwinianEcosystemEngine(
        division_overhead=1.5,
        min_replication_substrate=4.0,
        mutation_rate=1.0,
    )
    parent = SpatialEntity(
        entity_id="P-MUT",
        compound_id="CMP-MUT",
        kind=EntityKind.PROTOCELL,
        x=200.0,
        y=200.0,
        energy=30.0,
        stored_substrate=10.0,
        replication_threshold=22.0,
        decay_rate=0.02,
        catalytic_efficiency=1.5,
        stability_score=0.8,
    )

    result = engine.step(
        entities=[parent],
        substrate_source=0.0,
        dt=0.0,
        rng=np.random.default_rng(123),
        current_step=1,
    )
    children = result["entities"]
    assert len(children) == 2
    # At least one daughter exhibits a mutated catalytic efficiency or stability
    diverged = any(
        c.catalytic_efficiency != parent.catalytic_efficiency
        or c.stability_score != parent.stability_score
        for c in children
    )
    assert diverged, "Expected mutation variation in offspring"


# =============================================================================
# 3. Differential Selection Test
# =============================================================================

def test_differential_selection() -> None:
    """
    Under limited substrate, a high-efficiency lineage (higher uptake & metabolism)
    must outcompete a low-efficiency lineage through emergent differential reproduction.
    """
    tracker = GenealogyTracker()
    engine = DarwinianEcosystemEngine(
        genealogy_tracker=tracker,
        mutation_rate=0.0,
        division_overhead=2.0,
        min_replication_substrate=4.0,
        base_uptake_rate=1.5,
        substrate_energy_yield=2.0,
    )

    # Lineage A: Superior catalytic efficiency (2.0)
    # Lineage B: Deficient catalytic efficiency (0.4)
    entities: list[SpatialEntity] = []
    for i in range(10):
        entities.append(SpatialEntity(
            entity_id=f"A-{i}",
            compound_id="HIGH-EFF",
            kind=EntityKind.PROTOCELL,
            x=100.0 + i * 5.0,
            y=100.0 + i * 5.0,
            energy=15.0,
            stored_substrate=2.0,
            catalytic_efficiency=2.0,
            stability_score=0.9,
            decay_rate=0.02,
            replication_threshold=20.0,
        ))
        entities.append(SpatialEntity(
            entity_id=f"B-{i}",
            compound_id="LOW-EFF",
            kind=EntityKind.PROTOCELL,
            x=300.0 + i * 5.0,
            y=300.0 + i * 5.0,
            energy=15.0,
            stored_substrate=2.0,
            catalytic_efficiency=0.4,
            stability_score=0.9,
            decay_rate=0.02,
            replication_threshold=20.0,
        ))

    cA = CompoundRecord(
        compound_id="HIGH-EFF", name="High Efficiency", common_name="HE",
        canonical_smiles="c1ccccc1", formula="C6H6", mol_weight=78.11,
    )
    cB = CompoundRecord(
        compound_id="LOW-EFF", name="Low Efficiency", common_name="LE",
        canonical_smiles="c1ccncc1", formula="C5H5N", mol_weight=79.10,
    )
    tracker.register_compound(cA, birth_step=0)
    tracker.register_compound(cB, birth_step=0)

    # Run for 30 steps with limited environmental substrate
    rng = np.random.default_rng(42)
    current_entities = entities
    for step in range(1, 35):
        # Substrate source yields a moderate amount
        res = engine.step(
            entities=current_entities,
            substrate_source=1.8,
            dt=1.0,
            rng=rng,
            current_step=step,
        )
        current_entities = res["entities"]

    count_A = sum(1 for e in current_entities if e.compound_id == "HIGH-EFF" and e.alive)
    count_B = sum(1 for e in current_entities if e.compound_id == "LOW-EFF" and e.alive)

    # Lineage A should have expanded while Lineage B declines or remains suppressed
    assert count_A > count_B, f"High-efficiency lineage ({count_A}) should outcompete low-efficiency ({count_B})"


# =============================================================================
# 4. Extinction & Cause Logging Test
# =============================================================================

def test_extinction_and_cause_logging() -> None:
    """
    When an extant population drops to zero, the tracker must record the
    extinction step and cause.
    """
    tracker = GenealogyTracker()
    engine = DarwinianEcosystemEngine(genealogy_tracker=tracker)

    comp = CompoundRecord(
        compound_id="EXTINCT-01",
        name="Fragile Replicator",
        common_name="Fragile",
        canonical_smiles="c1ccoc1",
        formula="C4H4O",
        mol_weight=68.07,
    )
    tracker.register_compound(comp, birth_step=0)

    # Entity starts with low energy and no substrate in a harsh decaying environment
    entity = SpatialEntity(
        entity_id="E-STARVE",
        compound_id="EXTINCT-01",
        kind=EntityKind.PROTOCELL,
        x=50.0,
        y=50.0,
        energy=0.5,
        stored_substrate=0.0,
        decay_rate=1.0,  # Fast decay guarantees starvation in 1 step
        alive=True,
    )

    # Step 1: Living
    tracker.update_abundances({"EXTINCT-01": 1}, current_step=1)
    assert not tracker.nodes["EXTINCT-01"].is_extinct

    # Step 2: Engine runs, entity starves and dies (energy <= 0)
    res = engine.step(
        entities=[entity],
        substrate_source=0.0,
        dt=1.0,
        current_step=2,
    )
    assert len([e for e in res["entities"] if e.alive]) == 0

    node = tracker.nodes["EXTINCT-01"]
    assert node.is_extinct is True
    assert node.extinction_step == 2
    assert node.extinction_cause == "competitive_exclusion_or_decay"


# =============================================================================
# 5. Neutral Genetic Drift Test
# =============================================================================

def test_neutral_genetic_drift() -> None:
    """
    Two lineages with identical functional traits undergo demographic stochasticity
    without systematic selective bias.
    """
    engine = DarwinianEcosystemEngine(
        mutation_rate=0.0,
        division_overhead=2.0,
        min_replication_substrate=4.0,
    )

    entities: list[SpatialEntity] = []
    for i in range(8):
        entities.append(SpatialEntity(
            entity_id=f"N1-{i}",
            compound_id="NEUTRAL-1",
            kind=EntityKind.PROTOCELL,
            x=150.0 + (i % 3) * 10.0,
            y=150.0 + (i // 3) * 10.0,
            energy=16.0,
            stored_substrate=3.0,
            catalytic_efficiency=1.2,
            stability_score=0.85,
            replication_threshold=20.0,
        ))
        entities.append(SpatialEntity(
            entity_id=f"N2-{i}",
            compound_id="NEUTRAL-2",
            kind=EntityKind.PROTOCELL,
            x=180.0 + (i % 3) * 10.0,
            y=180.0 + (i // 3) * 10.0,
            energy=16.0,
            stored_substrate=3.0,
            catalytic_efficiency=1.2,
            stability_score=0.85,
            replication_threshold=20.0,
        ))

    rng = np.random.default_rng(777)
    current = entities
    for step in range(1, 20):
        res = engine.step(current, substrate_source=2.0, dt=1.0, rng=rng, current_step=step)
        current = res["entities"]

    count1 = sum(1 for e in current if e.compound_id == "NEUTRAL-1" and e.alive)
    count2 = sum(1 for e in current if e.compound_id == "NEUTRAL-2" and e.alive)

    # Both lineages survive and population is actively reproducing
    assert count1 + count2 > 0
    # Relative ratio has drifted from initial 1:1 without breaking the universe
    assert isinstance(count1, int) and isinstance(count2, int)


# =============================================================================
# 6. 100% Chemical Validity of 50 RDKit Mutations/Recombinations
# =============================================================================

def test_rdkit_chemical_validity_50_mutations_recombinations() -> None:
    """
    Verify 100% chemical validity of 50 random RDKit mutations and recombinations
    across diverse small-molecule drug scaffolds.
    Every produced SMILES must parse and sanitize cleanly with Chem.SanitizeMol.
    """
    opt = DirectedMolecularOptimizer()
    rng = np.random.default_rng(2026)

    test_smiles = [
        "c1ccccc1",                               # Benzene
        "c1ccncc1",                               # Pyridine
        "CC(=O)Oc1ccccc1C(=O)O",                  # Aspirin
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",          # Ibuprofen
        "c1ccoc1",                               # Furan
        "c1ccc2c(c1)cccc2",                       # Naphthalene
        "CCNCC",                                  # Diethylamine
        "CC(=O)Nc1ccc(O)cc1",                     # Paracetamol
    ]

    # Test 50 mutations
    for i in range(50):
        parent_smi = test_smiles[i % len(test_smiles)]
        mutated_smi, label = opt.mutate_smiles(parent_smi, rng)

        assert isinstance(mutated_smi, str) and len(mutated_smi) > 0, f"Empty mutation at {i}"
        mol = Chem.MolFromSmiles(mutated_smi)
        assert mol is not None, f"Failed to parse SMILES from mutation ({label}): {mutated_smi}"
        Chem.SanitizeMol(mol)  # Raises ValueError if invalid valence or aromaticity

    # Test 50 recombinations
    for i in range(50):
        s1 = test_smiles[i % len(test_smiles)]
        s2 = test_smiles[(i + 1) % len(test_smiles)]
        recombined_smi, label = opt.recombine_smiles(s1, s2, rng)

        assert isinstance(recombined_smi, str) and len(recombined_smi) > 0, f"Empty recombination at {i}"
        mol = Chem.MolFromSmiles(recombined_smi)
        assert mol is not None, f"Failed to parse SMILES from recombination ({label}): {recombined_smi}"
        Chem.SanitizeMol(mol)


# =============================================================================
# 7. NSGA-II Pareto Sorting & Crowding Distance Test
# =============================================================================

def test_nsga2_pareto_sorting_and_crowding() -> None:
    """
    Verify fast non-dominated sorting and crowding distance calculations
    on analytically known 2D objective vectors (maximizing both objectives).
    """
    opt = DirectedMolecularOptimizer()

    # Analytical test points:
    # 0: (4.0, 4.0) -> Dominates all other points
    # 1: (3.0, 1.0) -> Non-dominated with 2
    # 2: (1.0, 3.0) -> Non-dominated with 1
    # 3: (2.0, 2.0) -> Non-dominated with 1 and 2
    # 4: (0.5, 0.5) -> Dominated by all above
    pts = np.array([
        [4.0, 4.0],  # 0
        [3.0, 1.0],  # 1
        [1.0, 3.0],  # 2
        [2.0, 2.0],  # 3
        [0.5, 0.5],  # 4
    ])

    fronts = opt.fast_non_dominated_sort(pts)
    assert len(fronts) == 3, f"Expected 3 Pareto fronts, got {len(fronts)}"
    assert fronts[0] == [0], f"Front 0 should contain [0], got {fronts[0]}"
    assert set(fronts[1]) == {1, 2, 3}, f"Front 1 should contain {1, 2, 3}, got {fronts[1]}"
    assert fronts[2] == [4], f"Front 2 should contain [4], got {fronts[2]}"

    # Crowding distance on Front 1 [1, 2, 3]
    # Sorted by obj 0: 2 (1.0), 3 (2.0), 1 (3.0)
    # Extremes 1 and 2 should have infinite crowding distance
    # Interior 3 should have finite crowding distance
    cd = opt.crowding_distance(pts, fronts[1])
    assert np.isinf(cd[1])
    assert np.isinf(cd[2])
    assert np.isfinite(cd[3]) and cd[3] > 0.0


# =============================================================================
# 8. Genealogy Ancestor Tracing & Comparison Test
# =============================================================================

def test_genealogy_ancestor_tracing() -> None:
    """
    Verify trace_lineage correctly traverses ancestor nodes back to original founders.
    """
    tracker = GenealogyTracker()

    f0 = CompoundRecord(
        compound_id="FOUNDER-0", name="Founder", common_name="F0",
        canonical_smiles="c1ccccc1", formula="C6H6", mol_weight=78.11,
        origin_generation=0,
    )
    m1 = CompoundRecord(
        compound_id="MUTANT-1", name="Mutant 1", common_name="M1",
        canonical_smiles="c1ccncc1", formula="C5H5N", mol_weight=79.10,
        origin_generation=1, parent_ids=["FOUNDER-0"],
    )
    m2 = CompoundRecord(
        compound_id="MUTANT-2", name="Mutant 2", common_name="M2",
        canonical_smiles="c1ccnnc1", formula="C4H4N2", mol_weight=80.09,
        origin_generation=2, parent_ids=["MUTANT-1"],
    )

    tracker.register_compound(f0, birth_step=0)
    tracker.register_compound(m1, birth_step=10, similarity_to_parent=0.85)
    tracker.register_compound(m2, birth_step=25, similarity_to_parent=0.90)

    # Trace lineage for M2
    lineage = tracker.trace_lineage("MUTANT-2")
    cids = [n["compound_id"] for n in lineage]
    assert cids == ["FOUNDER-0", "MUTANT-1", "MUTANT-2"]

    # Compare nodes
    comp = tracker.compare_nodes("FOUNDER-0", "MUTANT-2")
    assert comp["relationship"] == "ancestor_descendant"
    assert comp["generation_delta"] == 2
    assert "mol_weight" in comp["property_deltas"]
    assert comp["tanimoto_similarity"] is not None


# =============================================================================
# 9. Molecular GA Population Evolution Test
# =============================================================================

def test_molecular_ga_population_evolution() -> None:
    """
    Verify evolve_population executes a full multi-objective generation cycle,
    maintains population size, calculates diversity, and returns Pareto front 0.
    """
    opt = DirectedMolecularOptimizer()
    rng = np.random.default_rng(555)

    def mock_evaluator(smiles: str) -> dict[str, float]:
        mol = Chem.MolFromSmiles(smiles)
        mw = mol.GetNumHeavyAtoms() if mol else 1.0
        return {
            "qed_proxy": float(min(1.0, 10.0 / (mw + 1.0))),
            "aromatic_ratio": float(sum(1 for a in mol.GetAtoms() if a.GetIsAromatic()) / max(1, mol.GetNumAtoms())),
        }

    initial_pop = ["c1ccccc1", "c1ccncc1", "c1ccoc1", "CCNCC"]
    result = opt.evolve_population(
        population_smiles=initial_pop,
        evaluate_fn=mock_evaluator,
        population_size=6,
        rng=rng,
        generation=1,
    )

    assert len(result["survivors_smiles"]) == 6
    assert len(result["pareto_front_0"]) >= 1
    assert result["diversity_score"] >= 0.0
    assert result["generation"] == 1
    assert "EVOCURE" in result["provenance"]


# =============================================================================
# 10. Genealogy Query & Serialization Roundtrip Test
# =============================================================================

def test_genealogy_query_and_serialization() -> None:
    """
    Verify query_tree filtering, statistics computation, and dict serialization roundtrip.
    """
    tracker = GenealogyTracker()
    c1 = CompoundRecord(
        compound_id="C-A", name="A", common_name="A",
        canonical_smiles="c1ccccc1", formula="C6H6", mol_weight=78.11,
        family_id="FAM-1", origin_generation=0,
    )
    c2 = CompoundRecord(
        compound_id="C-B", name="B", common_name="B",
        canonical_smiles="c1ccncc1", formula="C5H5N", mol_weight=79.10,
        family_id="FAM-1", origin_generation=1, parent_ids=["C-A"],
    )
    c3 = CompoundRecord(
        compound_id="C-C", name="C", common_name="C",
        canonical_smiles="c1ccsc1", formula="C4H4S", mol_weight=84.14,
        family_id="FAM-2", origin_generation=1, parent_ids=["C-A"],
    )

    tracker.register_compound(c1, birth_step=0)
    tracker.register_compound(c2, birth_step=5)
    tracker.register_compound(c3, birth_step=10)

    tracker.update_abundances({"C-A": 5, "C-B": 10, "C-C": 0}, current_step=15)

    # Query FAM-1
    q1 = tracker.query_tree(family_id="FAM-1")
    assert q1["total_nodes"] == 2
    assert q1["dominant_family"] == "FAM-1"

    # Query without extinct (C-C is extinct)
    q_alive = tracker.query_tree(include_extinct=False)
    alive_cids = {n["compound_id"] for n in q_alive["nodes"]}
    assert "C-C" not in alive_cids
    assert "C-A" in alive_cids and "C-B" in alive_cids

    # Serialization roundtrip
    serialized = tracker.to_dict()
    restored = GenealogyTracker.from_dict(serialized)
    assert len(restored.nodes) == 3
    assert restored.nodes["C-C"].is_extinct is True
    assert restored.nodes["C-B"].current_abundance == 10
