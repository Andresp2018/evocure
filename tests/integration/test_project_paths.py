"""Regression tests for the path resolution defect and the silence around it.

Measured before the fix. Every module located its data with
`Path(__file__).resolve().parents[2]`, which is the checkout root only while the
packages sit inside a checkout. After `pip install .`, the documented install
route, the same expression resolves to site-packages, so:

    .venv/lib/python3.13/site-packages/models/scientific/pdac_qsar_v1.joblib

does not exist. The discovery pipeline caught that FileNotFoundError and carried
on with an unfitted predictor, so a PDAC run returned predicted_pdac_activity
0.0 for every candidate, no IC50, no tissue simulation, and still described
itself as an exploratory PDAC potency screen.

The suite never caught it because pytest sets `pythonpath = ["."]`, putting the
checkout ahead of site-packages, so the tests only ever load the in-repo copy.
These tests therefore drive the resolver directly rather than relying on import
order, and assert the pipeline says so when the model is missing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from simulation.core import paths
from simulation.discovery.pipeline import CandidateDiscoveryPipeline


@pytest.fixture(autouse=True)
def _clear_root_cache():
    """project_root is cached per process; these tests move the root."""
    paths.project_root.cache_clear()
    paths.writable_root.cache_clear()
    yield
    paths.project_root.cache_clear()
    paths.writable_root.cache_clear()


def _make_root(base: Path) -> Path:
    """A directory carrying the markers that identify an EVOCURE root."""
    (base / "configs").mkdir(parents=True, exist_ok=True)
    (base / "configs" / "default_universe.json").write_text(json.dumps({}), encoding="utf-8")
    (base / "data" / "reference").mkdir(parents=True, exist_ok=True)
    return base


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def test_the_checkout_is_found_from_the_package_itself():
    """Running inside the repository must resolve to the repository."""
    root = paths.project_root()
    assert paths.is_resolved_root()
    assert (root / "configs" / "default_universe.json").is_file()
    assert (root / "data" / "reference").is_dir()


def test_evocure_home_overrides_everything(tmp_path, monkeypatch):
    target = _make_root(tmp_path / "relocated")
    monkeypatch.setenv("EVOCURE_HOME", str(target))
    paths.project_root.cache_clear()

    assert paths.project_root() == target.resolve()
    assert paths.configs_dir() == target.resolve() / "configs"
    assert paths.scientific_model_path().name == "pdac_qsar_v1.joblib"
    assert paths.scientific_model_path().is_relative_to(target.resolve())


def test_an_unresolvable_root_is_reported_rather_than_assumed(tmp_path, monkeypatch):
    """The honest answer when no root exists is 'not found', not a wrong path.

    This is the case that used to fail silently: a packaged install with no
    checkout anywhere above it.
    """
    monkeypatch.delenv("EVOCURE_HOME", raising=False)
    monkeypatch.setattr(paths, "_FALLBACK_ROOT", tmp_path / "nowhere")
    monkeypatch.setattr(paths, "_search_upwards", lambda _start: None)
    paths.project_root.cache_clear()

    assert paths.is_resolved_root() is False
    description = paths.describe()
    assert description["root_resolved_from_markers"] is False
    assert description["configs_present"] is False


def test_describe_reports_what_a_missing_file_error_needs(monkeypatch):
    monkeypatch.delenv("EVOCURE_HOME", raising=False)
    paths.project_root.cache_clear()
    description = paths.describe()

    for key in ("project_root", "root_resolved_from_markers", "writable_root",
                "configs_present", "reference_data_present", "qsar_artifact_present",
                "markers_checked"):
        assert key in description


# ---------------------------------------------------------------------------
# Writable state must never land in site-packages
# ---------------------------------------------------------------------------

def test_writable_root_follows_its_own_override(tmp_path, monkeypatch):
    monkeypatch.setenv("EVOCURE_DATA_HOME", str(tmp_path / "state"))
    paths.writable_root.cache_clear()

    assert paths.writable_root() == tmp_path / "state"
    written = paths.writable_path("experiments", "evocure.sqlite3")
    assert written.parent.is_dir(), "writable_path must create the parent directory"


def test_a_read_only_root_does_not_become_a_write_target(tmp_path, monkeypatch):
    """A system-wide install gives a read-only root; writes must go elsewhere."""
    read_only = _make_root(tmp_path / "ro")
    monkeypatch.delenv("EVOCURE_DATA_HOME", raising=False)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("EVOCURE_HOME", str(read_only))
    paths.project_root.cache_clear()
    paths.writable_root.cache_clear()

    import os
    monkeypatch.setattr(os, "access", lambda path, mode: False)

    assert paths.writable_root() == tmp_path / "xdg" / "evocure"


# ---------------------------------------------------------------------------
# No silent degradation when the potency model is absent
# ---------------------------------------------------------------------------

def test_a_loadable_model_reports_itself_available():
    pipeline = CandidateDiscoveryPipeline(disease_id="PDAC")
    if not pipeline.potency_model_available:
        pytest.skip("no trained QSAR artifact in this checkout")
    assert pipeline.qsar_unavailable_reason is None


def test_a_missing_model_is_named_not_swallowed(monkeypatch):
    """The original defect: an unfitted predictor installed without a word."""
    from ai.predictors import qsar as qsar_module

    def explode(self, filepath=None):
        raise FileNotFoundError("Model artifact not found at: /nowhere/pdac_qsar_v1.joblib")

    monkeypatch.setattr(qsar_module.CalibratedQSARPredictor, "load", explode)

    pipeline = CandidateDiscoveryPipeline(disease_id="PDAC")
    assert pipeline.potency_model_available is False
    reason = pipeline.qsar_unavailable_reason
    assert reason and "FileNotFoundError" in reason
    assert "scripts/train_qsar.py" in reason, "the reason must say how to fix it"


def test_a_run_without_a_potency_model_does_not_claim_a_potency_scope(monkeypatch):
    """A chemistry-only run must not be labelled an exploratory PDAC potency screen."""
    from ai.predictors import qsar as qsar_module

    monkeypatch.setattr(
        qsar_module.CalibratedQSARPredictor, "load",
        lambda self, filepath=None: (_ for _ in ()).throw(FileNotFoundError("absent")),
    )

    pipeline = CandidateDiscoveryPipeline(disease_id="PDAC")
    result = pipeline.run(
        seed_queries=["CC(=O)Oc1ccccc1C(=O)O"], generations=1, population_size=4,
        top_k_cellular=1, duration_hours=2.0, seed=17, candidate_count=2,
    )

    assert result["potency_model_available"] is False
    assert result["potency_model_unavailable_reason"]
    assert "No fitted activity predictor" in result["activity_scope"]
    # The zeros are real, so the scope statement has to account for them.
    for candidate in result["evaluated_candidates"]:
        assert candidate["objectives"]["predicted_pdac_activity"] == 0.0
    assert result["controlled_comparisons"] == []
