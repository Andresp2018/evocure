"""Tests for the chemical-space tie-break, which was measured and did not work.

diverse_truncation blends Tanimoto distance into NSGA-II's crowding tie-break.
It was built to address the structural convergence in the project's candidate
review and, measured over three seeds, it did not: mean pairwise Tanimoto
distance moved -0.012 and scaffold count -0.3, both inside seed noise. See
scripts/measure_chemical_diversity.py.

So these tests pin what it actually does, which is a correct implementation of a
diversity-preserving truncation, and they deliberately do not assert that it
broadens any search. A test claiming that would contradict the measurement.
"""

from __future__ import annotations

import numpy as np
import pytest

from simulation.discovery.pipeline import CandidateDiscoveryPipeline
from simulation.evolution.molecular_ga import DirectedMolecularOptimizer

# Two tight clusters plus one outlier. A chemistry-blind tie-break can keep both
# members of one cluster; a chemistry-aware one should reach across.
BENZENES = ["c1ccccc1", "Cc1ccccc1", "CCc1ccccc1"]
SUGARS = ["OC[C@H]1OC(O)[C@H](O)[C@@H]1O", "OC[C@H]1OC(O)[C@H](O)[C@@H]1OC"]
OUTLIER = "CC(=O)Oc1ccccc1C(=O)O"


def test_it_returns_the_requested_count():
    optimizer = DirectedMolecularOptimizer()
    structures = BENZENES + SUGARS + [OUTLIER]
    matrix = np.random.default_rng(0).random((len(structures), 3))
    front = list(range(len(structures)))
    mapping = dict(enumerate(structures))

    for needed in (1, 2, 3, 5):
        kept = optimizer.diverse_truncation(matrix, front, needed, mapping)
        assert len(kept) == needed
        assert len(set(kept)) == needed, "an index was selected twice"
        assert set(kept).issubset(set(front))


def test_degenerate_requests_are_handled():
    optimizer = DirectedMolecularOptimizer()
    matrix = np.zeros((3, 2))
    assert optimizer.diverse_truncation(matrix, [0, 1, 2], 0, {}) == []
    assert optimizer.diverse_truncation(matrix, [], 2, {}) == []
    # Asking for more than the front holds returns the whole front.
    assert set(optimizer.diverse_truncation(matrix, [0, 1, 2], 9, {})) == {0, 1, 2}


def test_full_chemical_weight_spreads_across_clusters():
    """With objectives tied, the pick must be driven by structure alone."""
    optimizer = DirectedMolecularOptimizer()
    structures = BENZENES + SUGARS
    # Identical objectives, so crowding carries no information at all.
    matrix = np.ones((len(structures), 3))
    front = list(range(len(structures)))
    mapping = dict(enumerate(structures))

    kept = optimizer.diverse_truncation(matrix, front, 2, mapping,
                                        chemical_weight=1.0)
    benzene_indices = {0, 1, 2}
    assert len(set(kept) & benzene_indices) <= 1, (
        "both picks came from the same structural cluster")


def test_zero_chemical_weight_reproduces_crowding_order():
    optimizer = DirectedMolecularOptimizer()
    structures = BENZENES + SUGARS + [OUTLIER]
    matrix = np.array([
        [0.1, 0.9], [0.2, 0.8], [0.5, 0.5], [0.8, 0.2], [0.9, 0.1], [0.3, 0.7],
    ])
    front = list(range(len(structures)))
    mapping = dict(enumerate(structures))

    crowding = optimizer.crowding_distance(matrix, front)
    kept = optimizer.diverse_truncation(matrix, front, 2, mapping,
                                        chemical_weight=0.0)
    boundary = {idx for idx, value in crowding.items() if np.isinf(value)}
    assert set(kept).issubset(boundary), (
        "at zero chemical weight the boundary individuals NSGA-II keeps "
        "must be the ones kept")


def test_an_unreadable_structure_gets_no_diversity_bonus():
    """A structure nobody can parse must not win on an assumed max distance."""
    optimizer = DirectedMolecularOptimizer()
    structures = ["c1ccccc1", "Cc1ccccc1", "not a molecule"]
    matrix = np.ones((3, 2))
    mapping = dict(enumerate(structures))

    kept = optimizer.diverse_truncation(matrix, [0, 1, 2], 2, mapping,
                                        chemical_weight=1.0)
    assert 2 not in kept, "the unparseable structure was rewarded"


def test_weights_outside_zero_to_one_are_clamped():
    optimizer = DirectedMolecularOptimizer()
    structures = BENZENES + SUGARS
    matrix = np.ones((len(structures), 2))
    mapping = dict(enumerate(structures))

    for weight in (-5.0, 7.0):
        kept = optimizer.diverse_truncation(matrix, list(range(len(structures))),
                                            3, mapping, chemical_weight=weight)
        assert len(kept) == 3


# ---------------------------------------------------------------------------
# Pipeline wiring
# ---------------------------------------------------------------------------

def test_the_pipeline_reports_the_weight_it_was_given():
    pipeline = CandidateDiscoveryPipeline(disease_id="PDAC")
    result = pipeline.run(
        seed_queries=["Gemcitabine", "Aspirin"], generations=2, population_size=6,
        top_k_cellular=1, duration_hours=3.0, seed=404, candidate_count=3,
        chemical_diversity_weight=0.5,
    )

    assert result["chemical_diversity_weight"] == 0.5
    assert 0.0 <= result["population_diversity"] <= 1.0


def test_it_is_off_by_default():
    """Turning it on changes which structures a run explores, so it is opt in."""
    pipeline = CandidateDiscoveryPipeline(disease_id="PDAC")
    result = pipeline.run(
        seed_queries=["Gemcitabine"], generations=1, population_size=4,
        top_k_cellular=1, duration_hours=2.0, seed=404, candidate_count=2,
    )

    assert result["chemical_diversity_weight"] == 0.0
