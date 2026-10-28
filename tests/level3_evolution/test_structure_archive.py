"""The archive must stay a search trace and never become a candidate library.

Its value is cheap re-scoring: re-ranking 2093 archived structures takes 99
seconds against 6834 to repeat the search. Its risk is that a table of machine
generated molecules with scores reads as a shortlist. These tests pin both.
"""

from __future__ import annotations

import json

import pytest

from simulation.core.structure_archive import (
    EXPORT_CAVEAT,
    RECORD_TYPE,
    StructureArchive,
)

GEMCITABINE = "C1=CN(C(=O)N=C1N)[C@H]2C([C@@H]([C@H](O2)CO)O)(F)F"
ERLOTINIB = "COc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"


@pytest.fixture
def archive(tmp_path):
    return StructureArchive(db_path=tmp_path / "archive.sqlite3")


def candidate(smiles, compound_id, utility=0.5, **extra):
    base = {
        "compound_id": compound_id,
        "canonical_smiles": smiles,
        "composite_utility": utility,
        "objectives": {"predicted_pdac_activity": 0.7, "qed_druglikeness": 0.6,
                       "low_toxicological_risk": 0.8},
        "in_applicability_domain": True,
        "origin_generation": 2,
        "parent_ids": ["CMP-SEED"],
        "is_seed": False,
    }
    base.update(extra)
    return base


def result(evaluated, selected_ids=()):
    return {
        "evaluated_candidates": evaluated,
        "top_candidates": [c for c in evaluated if c["compound_id"] in selected_ids],
    }


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

def test_records_every_evaluated_structure_not_just_the_winners(archive):
    evaluated = [
        candidate(GEMCITABINE, "A", 0.9),
        candidate(ERLOTINIB, "B", 0.5),
        candidate(ASPIRIN, "C", 0.3),
        candidate(CAFFEINE, "D", 0.2),
    ]
    stats = archive.record_search(result(evaluated, selected_ids={"A"}), experiment_id="EXP-1")

    assert stats["inserted"] == 4
    assert archive.count() == 4

    summary = archive.summary()
    assert summary["selected_for_reporting"] == 1
    # The point of the archive: almost nothing a run evaluates gets reported.
    assert summary["reported_fraction"] == pytest.approx(0.25)


def test_invalid_structures_are_skipped_and_counted(archive):
    evaluated = [candidate(GEMCITABINE, "A"), candidate("not_a_molecule", "B")]
    stats = archive.record_search(result(evaluated), experiment_id="EXP-1")

    assert stats["inserted"] == 1
    assert stats["skipped_invalid"] == 1


def test_reseeing_a_structure_updates_scores_but_keeps_first_sighting(archive):
    archive.record_search(result([candidate(GEMCITABINE, "A", 0.4)]), experiment_id="EXP-1")
    archive.record_search(result([candidate(GEMCITABINE, "A", 0.8)]), experiment_id="EXP-2")

    assert archive.count() == 1
    row = archive.query(limit=1)[0]
    assert row["composite_utility"] == pytest.approx(0.8)
    assert row["first_experiment_id"] == "EXP-1"


def test_rejection_reason_is_stored_for_every_unselected_structure(archive):
    evaluated = [
        candidate(GEMCITABINE, "A", 0.9),
        candidate(ERLOTINIB, "B", 0.4, in_applicability_domain=False),
        candidate(ASPIRIN, "C", 0.3,
                  medchem={"alert_count": 2, "alert_families": ["BRENK", "PAINS_B"]}),
        candidate(CAFFEINE, "D", 0.2,
                  objectives={"predicted_pdac_activity": 0.1, "qed_druglikeness": 0.6,
                              "low_toxicological_risk": 0.8}),
    ]
    archive.record_search(result(evaluated, selected_ids={"A"}), experiment_id="EXP-1")

    reasons = {r["canonical_smiles"]: r["not_selected_because"]
               for r in archive.query(limit=10)}
    assert all(v for k, v in reasons.items() if "F)F" not in k), "every reject needs a reason"

    joined = " ".join(v for v in reasons.values() if v)
    assert "applicability domain" in joined
    assert "medicinal chemistry alerts" in joined
    assert "low predicted activity" in joined


# ---------------------------------------------------------------------------
# It must not read as a candidate library
# ---------------------------------------------------------------------------

def test_every_row_is_typed_as_a_search_trace(archive):
    archive.record_search(result([candidate(GEMCITABINE, "A")]), experiment_id="EXP-1")
    row = archive.query(limit=1)[0]

    assert row["record_type"] == RECORD_TYPE == "search_trace"
    assert row["caveat"] == EXPORT_CAVEAT


def test_the_schema_refuses_a_row_typed_as_anything_else(archive):
    """A CHECK constraint, so the framing cannot be edited away in a later patch."""
    import sqlite3

    archive.record_search(result([candidate(GEMCITABINE, "A")]), experiment_id="EXP-1")
    with pytest.raises(sqlite3.IntegrityError):
        with archive._connect() as conn:
            conn.execute("UPDATE structures SET record_type = 'candidate_library'")


def test_exports_carry_the_caveat_as_a_field(tmp_path, archive):
    archive.record_search(
        result([candidate(GEMCITABINE, "A"), candidate(ERLOTINIB, "B")]),
        experiment_id="EXP-1")

    destination = tmp_path / "export.json"
    archive.export(destination, limit=10)
    payload = json.loads(destination.read_text(encoding="utf-8"))

    assert payload["record_type"] == "search_trace"
    assert "NOT A CANDIDATE LIBRARY" in payload["caveat"]
    assert "1330" in payload["caveat"], "the export must carry the potency uncertainty"
    assert payload["summary"]["caveat"] == EXPORT_CAVEAT
    for row in payload["structures"]:
        assert row["caveat"] == EXPORT_CAVEAT


def test_model_versions_are_recorded_so_stale_scores_are_visible(archive):
    archive.record_search(result([candidate(GEMCITABINE, "A")]), experiment_id="EXP-1",
                          model_versions={"qsar": "pdac_qsar_v1", "pipeline": "1.1.0"})
    row = archive.query(limit=1)[0]

    versions = json.loads(row["model_versions_json"])
    assert versions["qsar"] == "pdac_qsar_v1"
    assert row["scored_at_utc"]


# ---------------------------------------------------------------------------
# Re-scoring, the reason the archive exists
# ---------------------------------------------------------------------------

def test_rescoring_updates_scores_and_reports_rank_churn(archive):
    evaluated = [candidate(GEMCITABINE, "A", 0.9), candidate(ERLOTINIB, "B", 0.5),
                 candidate(ASPIRIN, "C", 0.3), candidate(CAFFEINE, "D", 0.1)]
    archive.record_search(result(evaluated, selected_ids={"A"}), experiment_id="EXP-1",
                          model_versions={"qsar": "v1"})

    # A new model that reverses the ordering entirely.
    order = {GEMCITABINE: 0.1, ERLOTINIB: 0.3, ASPIRIN: 0.5, CAFFEINE: 0.9}

    def new_model(smiles):
        from rdkit import Chem
        canonical = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True)
        for key, value in order.items():
            if Chem.MolToSmiles(Chem.MolFromSmiles(key), isomericSmiles=True) == canonical:
                return {"composite_utility": value, "pic50": 6.0}
        return None

    outcome = archive.rescore(new_model, new_model_version="v2")

    assert outcome.rescored == 4
    assert outcome.improved + outcome.worsened == 4
    assert outcome.new_model == "v2"
    assert outcome.previous_model == "v1"

    top = archive.query(limit=1)[0]
    assert top["canonical_smiles"].count("N1C=NC") or "caffeine" in top["canonical_smiles"].lower() \
        or top["composite_utility"] == pytest.approx(0.9)
    assert json.loads(top["model_versions_json"])["qsar"] == "v2"


def test_rescoring_leaves_rows_the_model_cannot_score(archive):
    archive.record_search(
        result([candidate(GEMCITABINE, "A", 0.9), candidate(ERLOTINIB, "B", 0.5)]),
        experiment_id="EXP-1")

    outcome = archive.rescore(lambda smiles: None, new_model_version="v2")

    assert outcome.rescored == 0
    assert archive.query(limit=1)[0]["composite_utility"] == pytest.approx(0.9)


def test_rescore_outcome_explains_what_churn_means(archive):
    archive.record_search(result([candidate(GEMCITABINE, "A", 0.9)]), experiment_id="EXP-1")
    payload = archive.rescore(lambda s: {"composite_utility": 0.2},
                              new_model_version="v2").to_dict()

    assert "driven by the model rather than by the chemistry" in payload["interpretation"]


# ---------------------------------------------------------------------------
# Querying the explored space
# ---------------------------------------------------------------------------

def test_query_can_exclude_structures_with_alerts(archive):
    evaluated = [
        candidate(GEMCITABINE, "A", 0.9),
        candidate(ERLOTINIB, "B", 0.8,
                  medchem={"alert_count": 1, "alert_families": ["BRENK"]}),
    ]
    archive.record_search(result(evaluated), experiment_id="EXP-1")

    assert len(archive.query(limit=10)) == 2
    clean = archive.query(limit=10, exclude_alerts=True)
    assert len(clean) == 1


def test_query_can_surface_what_selection_discarded(archive):
    evaluated = [candidate(GEMCITABINE, "A", 0.9), candidate(ERLOTINIB, "B", 0.85)]
    archive.record_search(result(evaluated, selected_ids={"A"}), experiment_id="EXP-1")

    unselected = archive.query(limit=10, only_unselected=True)
    assert len(unselected) == 1
    assert unselected[0]["selected"] == 0


def test_empty_archive_still_reports_its_caveat(archive):
    summary = archive.summary()
    assert summary["total"] == 0
    assert "NOT A CANDIDATE LIBRARY" in summary["caveat"]


def test_churn_is_not_called_model_drift_when_the_objective_changed(archive):
    """Re-scoring on different criteria reorders by construction, not by drift."""
    archive.record_search(
        result([candidate(GEMCITABINE, "A", 0.9), candidate(ERLOTINIB, "B", 0.2)]),
        experiment_id="EXP-1")

    drift = archive.rescore(lambda s: {"composite_utility": 0.5},
                            new_model_version="v2",
                            comparable_objective=True).to_dict()
    assert drift["comparable_objective"] is True
    assert "model drift" in drift["interpretation"]

    changed = archive.rescore(lambda s: {"composite_utility": 0.5},
                              new_model_version="v3",
                              comparable_objective=False).to_dict()
    assert changed["comparable_objective"] is False
    assert "does not measure" in changed["interpretation"]
    assert "Compare like with like" in changed["interpretation"]


def test_scaffold_is_computed_when_the_caller_omits_it(archive):
    """The pipeline's candidate summary carries no scaffold, and scaffold count is
    the archive's most informative statistic, so it is derived rather than trusted."""
    evaluated = [candidate(GEMCITABINE, "A"), candidate(ERLOTINIB, "B"),
                 candidate(CAFFEINE, "C")]
    for item in evaluated:
        item.pop("murcko_scaffold", None)
    archive.record_search(result(evaluated), experiment_id="EXP-1")

    summary = archive.summary()
    assert summary["unique_scaffolds"] >= 2, "scaffolds must not collapse to one empty string"
    for row in archive.query(limit=10):
        assert row["murcko_scaffold"] is not None
