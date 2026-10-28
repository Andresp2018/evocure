"""Potency must come from the best available source, and say which one it used.

Three sources exist, in descending order of what they can support:

  curated_experimental_bioactivity  a measurement from the reference dataset
  calibrated_qsar                   a model, RMSE 1.34 log units under grouped CV
  structure_property_heuristic      a rule of thumb, RMSE 2.34, and its reachable
                                    range cannot span the curated drugs at all

The tissue path used to skip the middle one entirely: predict_antitumor_activity
had a QSAR branch but nothing passed it a prediction, so every generated structure
fell to the heuristic. These tests pin the precedence and the reporting of it.
"""

from __future__ import annotations

import pytest

from simulation.chemistry.engine import ChemistryEngine
from simulation.core.engine import EVOCUREEngine
from simulation.pharmacology.admet import PharmacologyEngine, qsar_potency_hint

# A gemcitabine-derived variant: valid, inside the applicability domain, and not
# present in the curated library.
NOVEL_SMILES = "Nc1ccn([C@H]2O[C@@H](n3ccc(N)nc3=O)C(F)(F)[C@H]2C(=O)O)c(=O)n1"


@pytest.fixture(scope="module")
def engine():
    instance = EVOCUREEngine(seed=1234)
    instance.set_active_disease("PDAC", seed=1234)
    return instance


def test_curated_drugs_keep_their_measured_potency(engine):
    """A model prediction must never override an experimental measurement."""
    record = engine._resolve_compound_rec("Gemcitabine")
    profile, info = engine._tissue_pharmacology(record)

    assert info["potency_provenance"] == "curated_experimental_bioactivity"
    assert profile["ic50_sensitive_um"] == pytest.approx(0.045, abs=0.01)


def test_novel_structures_use_the_qsar_not_the_heuristic(engine):
    record = engine.chemistry.build_compound(NOVEL_SMILES)
    profile, info = engine._tissue_pharmacology(record)

    assert info["potency_provenance"] == "calibrated_qsar"
    assert "conformal" in info.get("potency_caveat", "").lower()

    # The heuristic cannot return anything below about 0.87 uM for any molecule.
    # Landing under that is proof the QSAR branch is the one being used.
    assert profile["ic50_sensitive_um"] < 0.87


def test_provenance_is_always_reported(engine):
    for query in ("Gemcitabine", "Caffeine", "5-Fluorouracil"):
        _, info = engine._tissue_pharmacology(engine._resolve_compound_rec(query))
        assert info["potency_provenance"] in {
            "curated_experimental_bioactivity",
            "calibrated_qsar",
            "structure_property_heuristic",
        }


def test_hint_declines_when_it_cannot_help():
    """Out of domain, unfitted, or no prediction must fall back, not guess."""
    chem = ChemistryEngine()
    record = chem.build_compound(NOVEL_SMILES)

    assert qsar_potency_hint(None, record) is None

    class Unfitted:
        is_fitted = False

    assert qsar_potency_hint(Unfitted(), record) is None

    class OutOfDomain:
        is_fitted = True

        def predict(self, smiles):
            return {"in_applicability_domain": False, "ic50_um_estimated": 0.01}

    assert qsar_potency_hint(OutOfDomain(), record) is None

    class Broken:
        is_fitted = True

        def predict(self, smiles):
            raise RuntimeError("model unavailable")

    assert qsar_potency_hint(Broken(), record) is None


def test_hint_is_not_labelled_validated():
    """A 3 log unit conformal interval is not a validated measurement."""
    class Predictor:
        is_fitted = True

        def predict(self, smiles):
            return {"in_applicability_domain": True, "ic50_um_estimated": 0.5,
                    "tanimoto_similarity_max": 0.7, "conformal_quantile_95": 3.12}

    chem = ChemistryEngine()
    hint = qsar_potency_hint(Predictor(), chem.build_compound(NOVEL_SMILES))

    assert hint is not None
    assert hint["is_validated"] is False
    assert hint["ic50_sensitive_um"] == pytest.approx(0.5)


def test_heuristic_remains_the_fallback_when_qsar_is_absent():
    """Removing the predictor must degrade to the heuristic, not to an error."""
    pharmacology = PharmacologyEngine()
    record = ChemistryEngine().build_compound(NOVEL_SMILES)

    response = pharmacology.predict_antitumor_activity(record, qsar_pred=None)
    ic50 = response["ic50_sensitive_um"]["value"]

    assert isinstance(ic50, (int, float))
    # The heuristic's reachable floor; see investigate_potency_asymmetry.py.
    assert ic50 >= 0.8
