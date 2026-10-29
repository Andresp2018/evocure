"""Tests for the ChEMBL fetcher, with no network access.

The fetcher exists because the project's first priority, expanding the 43
compound training set, stopped at "obtain a PANC-1 IC50 export". It is the only
part of EVOCURE that touches the network, which makes it the part most likely to
rot unnoticed, so its filtering rules and its contract with
scripts/import_bioactivity.py are pinned here against a stub transport.

What matters most is the last test: the columns the fetcher writes have to be the
columns the importer reads. A mismatch there would produce a file that fetches
cleanly and imports as nothing, and the whole chain would look like it worked.
"""

from __future__ import annotations

import csv
import importlib.util
import io
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    """Load a scripts/ module by path; scripts/ is deliberately not a package."""
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fetcher = _load_script("fetch_chembl_bioactivity")


class StubResponse:
    def __init__(self, payload: dict, status: int = 200):
        self._payload = payload
        self.status_code = status

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"status {self.status_code}")


class StubClient:
    """Answers the two endpoints the fetcher uses, from canned payloads."""

    def __init__(self, targets: list[dict] | None = None,
                 activities: list[dict] | None = None):
        self.targets = targets if targets is not None else []
        self.activities = activities if activities is not None else []
        self.calls: list[tuple[str, dict]] = []

    def get(self, url: str, params: dict | None = None) -> StubResponse:
        params = params or {}
        self.calls.append((url, params))
        if "target.json" in url:
            return StubResponse({"targets": self.targets,
                                 "page_meta": {"total_count": len(self.targets)}})
        if "activity.json" in url:
            return StubResponse({"activities": self.activities,
                                 "page_meta": {"total_count": len(self.activities)}})
        if "status.json" in url:
            return StubResponse({"chembl_db_version": "ChEMBL_37"})
        return StubResponse({}, status=404)


def activity(**overrides) -> dict:
    record = {
        "canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O",
        "standard_type": "IC50",
        "standard_relation": "=",
        "standard_value": "1200.0",
        "standard_units": "nM",
        "molecule_chembl_id": "CHEMBL25",
        "assay_chembl_id": "CHEMBL765373",
        "document_chembl_id": "CHEMBL1131426",
        "document_year": 2009,
        "assay_description": "Cytotoxicity against PANC-1.",
        "data_validity_comment": None,
    }
    record.update(overrides)
    return record


PANC1 = {"target_chembl_id": "CHEMBL614139", "target_type": "CELL-LINE",
         "pref_name": "PANC-1", "organism": "Homo sapiens"}


# ---------------------------------------------------------------------------
# Resolving a cell line
# ---------------------------------------------------------------------------

def test_an_exact_cell_line_resolves():
    stub = StubClient(targets=[PANC1])
    resolved = fetcher.resolve_cell_line_target(stub, "PANC-1")

    assert resolved["target_chembl_id"] == "CHEMBL614139"
    assert resolved["pref_name"] == "PANC-1"


def test_a_non_cell_line_target_is_not_accepted():
    """A protein target named like a cell line is different biology."""
    protein = {"target_chembl_id": "CHEMBL1", "target_type": "SINGLE PROTEIN",
               "pref_name": "PANC-1", "organism": "Homo sapiens"}
    stub = StubClient(targets=[protein])

    with pytest.raises(SystemExit):
        fetcher.resolve_cell_line_target(stub, "PANC-1")


def test_an_inexact_name_is_refused_and_lists_the_near_misses():
    """PANC-1 and Gemcitabine-resistant PANC-1 must never be pooled.

    Silently resolving a loose match is exactly the assay pooling the importer
    exists to prevent, so the fetcher refuses and names the candidates.
    """
    resistant = {"target_chembl_id": "CHEMBL6066779", "target_type": "CELL-LINE",
                 "pref_name": "Gemcitabine-resistant PANC-1", "organism": "Homo sapiens"}
    # First query is the exact one and finds nothing; the widening query follows.
    stub = StubClient(targets=[resistant])

    class ExactMissClient(StubClient):
        def get(self, url, params=None):
            params = params or {}
            if "target.json" in url and "pref_name__iexact" in params:
                return StubResponse({"targets": [], "page_meta": {"total_count": 0}})
            return super().get(url, params)

    with pytest.raises(SystemExit) as raised:
        fetcher.resolve_cell_line_target(
            ExactMissClient(targets=[resistant]), "PANC")

    message = str(raised.value)
    assert "Gemcitabine-resistant PANC-1" in message
    assert "verbatim" in message


def test_an_ambiguous_name_is_refused():
    duplicate = dict(PANC1, target_chembl_id="CHEMBL999")
    stub = StubClient(targets=[PANC1, duplicate])

    with pytest.raises(SystemExit):
        fetcher.resolve_cell_line_target(stub, "PANC-1")


# ---------------------------------------------------------------------------
# What gets dropped
# ---------------------------------------------------------------------------

def test_unusable_records_are_dropped_and_counted():
    records = [
        activity(),                                                   # keep
        activity(standard_relation=">"),                              # censored bound
        activity(standard_relation="<", standard_value="10"),         # censored bound
        activity(data_validity_comment="Outside typical range"),      # ChEMBL doubts it
        activity(canonical_smiles=None),                              # no structure
        activity(standard_value=None),                                # no value
        activity(standard_units="ug.mL-1"),                           # not exactly convertible
    ]
    stub = StubClient(activities=records)

    rows, counts = fetcher.fetch_activities(stub, "CHEMBL614139", "IC50",
                                            max_records=100, pause=0.0)

    assert len(rows) == 1
    assert counts["seen"] == 7
    assert counts["censored"] == 2
    assert counts["flagged_by_chembl"] == 1
    assert counts["no_smiles"] == 1
    assert counts["no_value"] == 1
    assert counts["unit_not_convertible"] == 1


def test_everything_written_carries_an_uncensored_relation():
    stub = StubClient(activities=[activity(standard_relation="=="),
                                  activity(standard_relation="=")])
    rows, _ = fetcher.fetch_activities(stub, "CHEMBL614139", "IC50",
                                       max_records=100, pause=0.0)

    assert rows
    for row in rows:
        assert row["standard_relation"] == "="


def test_max_records_is_respected():
    stub = StubClient(activities=[activity() for _ in range(50)])
    rows, _ = fetcher.fetch_activities(stub, "CHEMBL614139", "IC50",
                                       max_records=5, pause=0.0)
    assert len(rows) == 5


def test_scaffold_report_counts_scaffolds_not_rows():
    """Scaffold count, not row count, is what a scaffold split sees."""
    rows = [
        {"canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O"},
        {"canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O"},   # duplicate compound
        {"canonical_smiles": "c1ccc2c(c1)cccc2"},
        {"canonical_smiles": "not a molecule"},
    ]
    summary = fetcher.scaffold_report(rows)

    assert summary["rows"] == 4
    assert summary["unique_compounds"] == 2
    assert summary["unparseable_structures"] == 1
    assert summary["distinct_murcko_scaffolds"] >= 1


# ---------------------------------------------------------------------------
# The contract with the importer, which is the one that silently breaks
# ---------------------------------------------------------------------------

def test_the_importer_accepts_what_the_fetcher_writes():
    from ai.predictors.bioactivity_import import AssayContext, normalise_records

    stub = StubClient(activities=[
        activity(canonical_smiles="CC(=O)Oc1ccccc1C(=O)O", molecule_chembl_id="CHEMBL25"),
        activity(canonical_smiles="Nc1ccn([C@@H]2O[C@H](CO)[C@@H](O)C2(F)F)c(=O)n1",
                 molecule_chembl_id="CHEMBL888", standard_value="45.0"),
    ])
    rows, _ = fetcher.fetch_activities(stub, "CHEMBL614139", "IC50",
                                       max_records=10, pause=0.0)
    for row in rows:
        row["cell_line"] = "PANC-1"

    # Round trip through CSV, because that is what actually reaches the importer.
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fetcher.CSV_COLUMNS)
    writer.writeheader()
    writer.writerows(rows)
    buffer.seek(0)
    parsed = list(csv.DictReader(buffer))

    report = normalise_records(
        parsed, AssayContext(endpoint="IC50", cell_line="PANC-1"))

    assert len(report.accepted) == 2, report.rejection_summary
    assert not report.rejected
    for record in report.accepted:
        assert record["ic50_um"] > 0
        assert record["assay_context"] == "IC50 | PANC-1"
        assert record["measurement_type"] == "IC50"


def test_written_columns_cover_what_the_importer_needs():
    needed = {"canonical_smiles", "standard_value", "standard_units",
              "standard_type", "cell_line", "standard_relation"}
    assert needed.issubset(set(fetcher.CSV_COLUMNS))


def test_fragmentation_counts_the_protocols_being_pooled():
    """One endpoint and one cell line is not one assay, and that is the number
    that decides whether a fetch is usable.

    Measured on ChEMBL_37, "IC50 | PANC-1" is 355 assays from 325 papers, median
    three records. Pooling them produced a model worse than the 43 compound
    curated one on four of five seeds, on an identical holdout.
    """
    rows = (
        [{"assay_chembl_id": "A1", "document_chembl_id": "D1"}] * 60
        + [{"assay_chembl_id": "A2", "document_chembl_id": "D2"}] * 3
        + [{"assay_chembl_id": f"A{i}", "document_chembl_id": f"D{i}"}
           for i in range(3, 40)]
    )
    report = fetcher.assay_fragmentation(rows)

    assert report["distinct_assays"] == 39
    assert report["distinct_documents"] == 39
    assert report["largest_assay_records"] == 60
    assert report["median_assay_records"] == 1
    assert report["assays_with_40_or_more"] == 1
    assert 0.0 < report["largest_assay_share_pct"] < 100.0


def test_fragmentation_handles_an_empty_fetch():
    report = fetcher.assay_fragmentation([])

    assert report["distinct_assays"] == 0
    assert report["largest_assay_records"] == 0
    assert report["largest_assay_share_pct"] == 0.0


def test_a_single_protocol_fetch_is_not_flagged():
    """The warning must not fire on data that is genuinely one assay."""
    rows = [{"assay_chembl_id": "A1", "document_chembl_id": "D1"}] * 50
    report = fetcher.assay_fragmentation(rows)

    assert report["distinct_assays"] == 1
    assert report["largest_assay_share_pct"] == 100.0


def test_only_units_the_importer_converts_exactly_are_requested():
    from ai.predictors.bioactivity_import import to_micromolar

    for unit in fetcher.CONVERTIBLE_UNITS:
        assert to_micromolar(1.0, unit) is not None, (
            f"the fetcher keeps {unit}, which the importer cannot convert")
