"""Regression tests for the two defects found auditing the evolutionary engine.

1. Coevolution reported descriptor changes on a molecule it never altered.
2. The tumour simulation was computed but excluded from the selection objectives,
   so the microenvironment exerted no selective pressure.

Both are the kind of fault that leaves the output looking entirely reasonable, so
they are pinned here rather than left to manual inspection.
"""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import Descriptors, QED

from simulation.biology.coevolution import CoevolutionEngine
from simulation.discovery.pipeline import (
    BASE_OBJECTIVE_KEYS,
    TISSUE_OBJECTIVE_KEY,
    CandidateDiscoveryPipeline,
)


# ---------------------------------------------------------------------------
# 1. Coevolution must change the structure it claims to have changed
# ---------------------------------------------------------------------------

def test_coevolution_mutation_changes_the_actual_structure():
    engine = CoevolutionEngine(seed=11)
    parent = engine.compound_pool[0]

    children = [engine.mutate_compound(parent, generation=g) for g in range(1, 9)]
    altered = [c for c in children if c.canonical_smiles != parent.canonical_smiles]

    assert altered, "no structural variant was produced across eight attempts"
    for child in altered:
        assert Chem.MolFromSmiles(child.canonical_smiles) is not None


def test_coevolution_descriptors_match_the_molecule_they_describe():
    """The original bug: MW and logP were shifted while the SMILES stayed identical."""
    engine = CoevolutionEngine(seed=23)
    parent = engine.compound_pool[0]

    for generation in range(1, 10):
        child = engine.mutate_compound(parent, generation=generation)
        if child.canonical_smiles == parent.canonical_smiles:
            continue  # parent retained, nothing claimed
        mol = Chem.MolFromSmiles(child.canonical_smiles)
        assert mol is not None

        assert child.mol_weight == pytest.approx(Descriptors.MolWt(mol), abs=0.05)
        assert child.logp == pytest.approx(Descriptors.MolLogP(mol), abs=0.05)
        assert child.qed == pytest.approx(QED.qed(mol), abs=0.01)
        assert child.heavy_atoms == mol.GetNumHeavyAtoms()


def test_coevolution_never_reports_a_derivative_of_an_unchanged_molecule():
    """If no valid variant is found the parent is returned, not relabelled."""
    engine = CoevolutionEngine(seed=5)
    parent = engine.compound_pool[0]

    for generation in range(1, 12):
        child = engine.mutate_compound(parent, generation=generation)
        if child.canonical_smiles == parent.canonical_smiles:
            assert child.compound_id == parent.compound_id, (
                "an unchanged molecule was given a derivative identity"
            )


def test_efflux_pressure_steers_towards_less_lipophilic_structures():
    """Tumour resistance state must actually reach the chemistry."""
    engine = CoevolutionEngine(seed=3)
    parent = engine.compound_pool[0]

    evasive = [engine.mutate_compound(parent, 1, "efflux_evasion").logp for _ in range(8)]
    balanced = [engine.mutate_compound(parent, 1, "balanced").logp for _ in range(8)]

    assert float(np.mean(evasive)) < float(np.mean(balanced))


# ---------------------------------------------------------------------------
# 2. The tumour model must be able to drive selection
# ---------------------------------------------------------------------------

def _run(tissue_in_fitness: bool):
    pipeline = CandidateDiscoveryPipeline(disease_id="PDAC")
    return pipeline.run(
        seed_queries=["Gemcitabine", "5-Fluorouracil"],
        generations=1,
        population_size=6,
        top_k_cellular=1,
        duration_hours=6.0,
        seed=4242,
        candidate_count=3,
        tissue_in_fitness=tissue_in_fitness,
    )


def test_open_loop_leaves_tumour_response_out_of_selection():
    result = _run(tissue_in_fitness=False)

    assert result["tissue_in_fitness"] is False
    assert list(result["objective_keys"]) == list(BASE_OBJECTIVE_KEYS)
    for candidate in result["evaluated_candidates"]:
        assert TISSUE_OBJECTIVE_KEY not in candidate["objectives"]
    assert "does not affect selection" in result["selection_basis"]


def test_closed_loop_makes_tumour_response_an_objective_for_every_candidate():
    result = _run(tissue_in_fitness=True)

    assert result["tissue_in_fitness"] is True
    assert TISSUE_OBJECTIVE_KEY in result["objective_keys"]

    evaluated = result["evaluated_candidates"]
    assert evaluated, "pipeline produced no candidates"

    # Every individual needs the objective, otherwise Pareto sorting compares
    # candidates on different criteria.
    for candidate in evaluated:
        value = candidate["objectives"].get(TISSUE_OBJECTIVE_KEY)
        assert value is not None, "a candidate entered selection without the objective"
        assert 0.0 <= value <= 1.0

    simulated = [c for c in evaluated if c.get("tissue_simulated")]
    assert simulated, "closed loop ran no tissue simulations"


def test_closed_loop_fitness_potency_matches_the_reporting_path():
    """The optimiser and the judge must score potency the same way.

    The closed loop originally drove its tissue simulation from the raw QSAR IC50
    while the report evaluated with the pharmacology model's IC50. On the published
    candidates those differed by roughly seventy fold, so selection optimised a
    target nothing ever measured.
    """
    from simulation.chemistry.engine import ChemistryEngine
    from simulation.pharmacology.admet import PharmacologyEngine

    chem, pharm = ChemistryEngine(), PharmacologyEngine()
    record = chem.resolve_compound_input("Gemcitabine")

    response = pharm.predict_antitumor_activity(record)
    reported = response["ic50_sensitive_um"]
    reported = reported.get("value") if isinstance(reported, dict) else reported

    pipeline = CandidateDiscoveryPipeline(disease_id="PDAC")
    captured: dict[str, float] = {}
    original = pipeline.disease_registry.create

    def capture(disease_id, **kwargs):
        captured.update(kwargs)
        return original(disease_id, **kwargs)

    pipeline.disease_registry.create = capture  # type: ignore[method-assign]
    pipeline.run(
        seed_queries=["Gemcitabine"], generations=1, population_size=4,
        top_k_cellular=1, duration_hours=4.0, seed=99, candidate_count=2,
        tissue_in_fitness=True,
    )

    # Same grid as the engine's reporting tumour module.
    assert captured.get("nx") == 36 and captured.get("ny") == 28
    assert isinstance(reported, (int, float)) and reported > 0


def test_closed_loop_simulates_more_than_the_top_candidate():
    """The original design simulated one individual, which cannot steer a population."""
    open_loop = _run(tissue_in_fitness=False)
    closed_loop = _run(tissue_in_fitness=True)

    assert len(closed_loop["controlled_comparisons"]) > len(open_loop["controlled_comparisons"])


# ---------------------------------------------------------------------------
# 3. Medicinal chemistry plausibility as a search objective
# ---------------------------------------------------------------------------

def test_medchem_objective_is_off_by_default():
    result = _run(tissue_in_fitness=False)
    assert result["medchem_filters"] is False
    assert "medicinal_chemistry_plausibility" not in result["objective_keys"]
    for candidate in result["evaluated_candidates"]:
        assert candidate.get("medchem") is None


def test_medchem_objective_enters_selection_when_enabled():
    from simulation.discovery.pipeline import MEDCHEM_OBJECTIVE_KEY

    pipeline = CandidateDiscoveryPipeline(disease_id="PDAC")
    result = pipeline.run(
        seed_queries=["Gemcitabine", "5-Fluorouracil"], generations=1,
        population_size=6, top_k_cellular=1, duration_hours=6.0, seed=4242,
        candidate_count=3, medchem_filters=True,
    )

    assert result["medchem_filters"] is True
    assert MEDCHEM_OBJECTIVE_KEY in result["objective_keys"]

    evaluated = result["evaluated_candidates"]
    assert evaluated
    for candidate in evaluated:
        assert candidate.get("medchem") is not None
        value = candidate["objectives"].get(MEDCHEM_OBJECTIVE_KEY)
        assert value is not None and 0.0 <= value <= 1.0


def test_medchem_assessment_travels_with_the_candidate():
    pipeline = CandidateDiscoveryPipeline(disease_id="PDAC")
    result = pipeline.run(
        seed_queries=["Gemcitabine"], generations=1, population_size=4,
        top_k_cellular=1, duration_hours=4.0, seed=7, candidate_count=2,
        medchem_filters=True,
    )

    for candidate in result["top_candidates"]:
        medchem = candidate.get("medchem")
        assert medchem is not None
        assert "sa_score" in medchem and "alert_count" in medchem
        assert "verdict" in medchem
        # The caveat must survive into anything a reader sees.
        assert "not that the compound can be made or works" in medchem["caveat"]
