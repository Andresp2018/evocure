"""The importer must never make the training set worse than it was.

Expanding the dataset is the project's largest lever, and also the easiest way to
destroy it: pooling measurements that are not comparable produces a model that
scores better on paper and predicts nothing. These tests pin the refusals.
"""

from __future__ import annotations

import math

import pytest

from ai.predictors.bioactivity_import import (
    AssayContext,
    canonical_endpoint,
    merge_into_dataset,
    normalise_records,
    pic50_from_micromolar,
    scaffold_summary,
    to_micromolar,
)

PANC1 = AssayContext(endpoint="IC50", cell_line="PANC-1")
GEMCITABINE = "C1=CN(C(=O)N=C1N)[C@H]2C([C@@H]([C@H](O2)CO)O)(F)F"
ERLOTINIB = "COc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


def row(smiles=GEMCITABINE, value="45", units="nM", endpoint="IC50",
        cell_line="PANC-1", **extra):
    base = {"canonical_smiles": smiles, "standard_value": value,
            "standard_units": units, "standard_type": endpoint,
            "cell_line": cell_line}
    base.update(extra)
    return base


# ---------------------------------------------------------------------------
# Unit conversion is exact or it does not happen
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value,unit,expected", [
    (1.0, "uM", 1.0),
    (1000.0, "nM", 1.0),
    (0.001, "mM", 1.0),
    (1e-6, "M", 1.0),
    (1_000_000.0, "pM", 1.0),
])
def test_known_units_convert_exactly(value, unit, expected):
    assert to_micromolar(value, unit) == pytest.approx(expected)


def test_mass_concentration_needs_a_molecular_weight():
    assert to_micromolar(1.0, "ug.mL-1") is None
    converted = to_micromolar(1.0, "ug.mL-1", mol_weight=263.2)
    assert converted == pytest.approx(1000.0 / 263.2, rel=1e-6)


def test_unknown_units_are_refused_not_guessed():
    for unit in ("percent", "%", "log10", "arbitrary", ""):
        assert to_micromolar(1.0, unit) is None


def test_nonsense_values_are_refused():
    assert to_micromolar(0.0, "uM") is None
    assert to_micromolar(-1.0, "uM") is None
    assert to_micromolar(float("nan"), "uM") is None


def test_pic50_round_trip():
    assert pic50_from_micromolar(1.0) == pytest.approx(6.0)
    assert pic50_from_micromolar(0.045) == pytest.approx(7.3468, abs=1e-3)


# ---------------------------------------------------------------------------
# Assay homogeneity, the rule this module exists for
# ---------------------------------------------------------------------------

def test_other_cell_lines_are_rejected():
    rows = [row(), row(smiles=ERLOTINIB, cell_line="MIA PaCa-2")]
    report = normalise_records(rows, PANC1)

    assert len(report.accepted) == 1
    assert len(report.rejected) == 1
    assert "assay context" in report.rejected[0]["reason"]


def test_other_endpoints_are_rejected():
    """An IC50 and a GI50 are different quantities even on the same cells."""
    rows = [row(), row(smiles=ERLOTINIB, endpoint="GI50")]
    report = normalise_records(rows, PANC1)

    assert len(report.accepted) == 1
    assert any("assay context" in r["reason"] for r in report.rejected)


def test_bacterial_mic_never_enters_a_cytotoxicity_target():
    rows = [
        row(),
        row(smiles=ERLOTINIB, endpoint="MIC", cell_line="", organism="E. coli"),
    ]
    report = normalise_records(rows, PANC1)

    assert len(report.accepted) == 1
    assert report.accepted[0]["measurement_type"] == "IC50"


def test_censored_values_are_rejected():
    """A result reported as >100 uM is a bound, not a measurement."""
    rows = [row(), row(smiles=ERLOTINIB, standard_relation=">")]
    report = normalise_records(rows, PANC1)

    assert len(report.accepted) == 1
    assert any("censored" in r["reason"] for r in report.rejected)


def test_inspection_reports_every_group_present():
    rows = [row(), row(smiles=ERLOTINIB, cell_line="MIA PaCa-2"),
            row(smiles=ASPIRIN, endpoint="GI50")]
    report = normalise_records(rows, AssayContext(endpoint="__none__", cell_line="__none__"))

    assert len(report.groups) == 3
    assert sum(report.groups.values()) == 3
    assert not report.accepted


# ---------------------------------------------------------------------------
# Bad input is dropped and counted, never invented
# ---------------------------------------------------------------------------

def test_unparseable_and_missing_input_is_counted():
    rows = [
        row(),
        row(smiles="not a molecule"),
        {"standard_value": "10", "standard_units": "uM", "standard_type": "IC50",
         "cell_line": "PANC-1"},
        row(smiles=ERLOTINIB, value=None, units=None),
    ]
    report = normalise_records(rows, PANC1)

    assert len(report.accepted) == 1
    assert len(report.rejected) == 3
    reasons = report.rejection_summary
    assert reasons.get("SMILES does not parse") == 1
    assert reasons.get("no SMILES") == 1
    assert reasons.get("no activity value") == 1


# ---------------------------------------------------------------------------
# Duplicates
# ---------------------------------------------------------------------------

def test_agreeing_duplicates_are_combined_without_a_conflict():
    rows = [row(value="45"), row(value="50")]
    report = normalise_records(rows, PANC1)

    assert len(report.accepted) == 1
    assert report.accepted[0]["replicate_count"] == 2
    assert not report.conflicts
    # Geometric mean of 0.045 and 0.050 uM.
    assert report.accepted[0]["ic50_um"] == pytest.approx(math.sqrt(0.045 * 0.050), rel=1e-3)


def test_disagreeing_duplicates_are_reported_not_hidden():
    rows = [row(value="45"), row(value="45000")]
    report = normalise_records(rows, PANC1)

    assert len(report.accepted) == 1
    assert len(report.conflicts) == 1
    assert report.conflicts[0]["fold_disagreement"] == pytest.approx(1000.0, rel=0.01)


# ---------------------------------------------------------------------------
# Merging must not overwrite curated values
# ---------------------------------------------------------------------------

def test_merge_keeps_curated_records_on_collision():
    curated = [{"name": "Gemcitabine", "canonical_smiles": GEMCITABINE,
                "pic50": 7.347, "dataset_source": "curated"}]
    incoming = normalise_records([row(value="9000")], PANC1).accepted

    merged, info = merge_into_dataset(curated, incoming, "PDAC")

    assert info["added"] == 0
    assert info["skipped_already_present"] == 1
    assert merged[0]["dataset_source"] == "curated"
    assert merged[0]["pic50"] == 7.347


def test_merge_adds_genuinely_new_structures():
    curated = [{"name": "Gemcitabine", "canonical_smiles": GEMCITABINE, "pic50": 7.347}]
    incoming = normalise_records([row(smiles=ERLOTINIB, value="450")], PANC1).accepted

    merged, info = merge_into_dataset(curated, incoming, "PDAC")

    assert info["added"] == 1
    assert len(merged) == 2
    assert merged[1]["disease_id"] == "PDAC"
    assert merged[1]["growth_inhibition_active"] is True


# ---------------------------------------------------------------------------
# Scaffold accounting
# ---------------------------------------------------------------------------

def test_scaffold_summary_counts_independent_scaffolds():
    records = normalise_records(
        [row(), row(smiles=ERLOTINIB, value="450"), row(smiles=ASPIRIN, value="9000")],
        PANC1,
    ).accepted

    summary = scaffold_summary(records)
    assert summary["compounds"] == 3
    assert summary["independent_scaffolds"] >= 2


def test_canonical_endpoint_folds_known_aliases():
    assert canonical_endpoint("IC50") == "IC50"
    assert canonical_endpoint("ic50_um") == "IC50"
    assert canonical_endpoint("pIC50") == "IC50"
    assert canonical_endpoint("MIC") == "MIC"
    # An unrecognised endpoint keeps its own group rather than being folded in.
    assert canonical_endpoint("Thermal shift") == "THERMAL SHIFT"
