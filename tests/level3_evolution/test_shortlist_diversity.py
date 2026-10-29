"""The reported shortlist must not present stereoisomers as separate candidates.

Measured on a 100 generation run before the fix: the top five candidates held
three distinct constitutions, two of them stereoisomer pairs, and the report
presented all five as prioritised candidates. A 20 generation run of the same
pipeline gave five distinct constitutions, so the defect got worse with more
search, which is the opposite of what a larger budget should buy.

The cause is that every score in this pipeline is blind to stereochemistry. The
descriptors, the Morgan fingerprint the QSAR reads and the tissue model all
return identical values for a stereoisomer pair, so their composite utility
matches to the last digit, they sort adjacently, and taking the top N fills
consecutive slots from one constitution.
"""

from __future__ import annotations

from rdkit import Chem

from simulation.discovery.pipeline import CandidateDiscoveryPipeline


def constitution(smiles: str) -> str:
    """The InChIKey skeleton block, which stereoisomers share."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    return Chem.MolToInchiKey(mol).split("-")[0]


def _run(candidate_count: int = 5, **kwargs):
    pipeline = CandidateDiscoveryPipeline(disease_id="PDAC")
    return pipeline.run(
        seed_queries=["Gemcitabine", "5-Fluorouracil", "Erlotinib"],
        generations=3, population_size=12, top_k_cellular=1,
        duration_hours=4.0, seed=515, candidate_count=candidate_count,
        **kwargs,
    )


def test_stereoisomers_do_not_fill_separate_slots():
    """The original defect, pinned."""
    result = _run()
    shortlist = result["top_candidates"]
    assert shortlist, "the search returned no candidates"

    skeletons = [constitution(c["canonical_smiles"]) for c in shortlist]
    assert len(set(skeletons)) == len(skeletons), (
        f"the shortlist presents {len(skeletons)} candidates over only "
        f"{len(set(skeletons))} constitutions: {skeletons}"
    )


def test_the_shortlist_reports_how_many_constitutions_it_holds():
    result = _run()
    assert result["distinct_constitutions_reported"] == len(result["top_candidates"])


def test_stereoisomer_families_are_reported_not_hidden():
    """Collapsing a family must leave a record of what was collapsed."""
    result = _run()
    families = result["stereoisomer_families"]

    assert isinstance(families, dict)
    for skeleton, ids in families.items():
        assert len(ids) > 1, "a single member is not a family"
        assert len(set(ids)) == len(ids), "an id appears twice in one family"


def test_the_requested_count_is_still_honoured():
    """De-duplicating must not quietly return a shorter list."""
    for count in (2, 3, 5):
        result = _run(candidate_count=count)
        evaluated = [c for c in result["evaluated_candidates"] if not c["is_seed"]]
        expected = min(count, len(evaluated))
        assert len(result["top_candidates"]) == expected, (
            f"asked for {count}, got {len(result['top_candidates'])} "
            f"from {len(evaluated)} available variants"
        )


def test_candidates_are_still_ordered_by_utility():
    """Preferring distinct constitutions must not scramble the ranking."""
    result = _run()
    utilities = [c.get("composite_utility") for c in result["top_candidates"]]
    scored = [u for u in utilities if u is not None]
    assert scored == sorted(scored, reverse=True), utilities


def test_no_seed_reaches_the_shortlist():
    result = _run()
    for candidate in result["top_candidates"]:
        assert candidate["is_seed"] is False
