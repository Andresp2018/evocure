"""Reported QSAR metrics must describe the model, not the split that produced them.

A single Bemis-Murcko scaffold split on this dataset leaves about ten test compounds.
Every metric from ten points swings with the split: measured across ten random_state
values, R2 ranged from -0.09 to 0.82 and the conformal quantile from 1.97 to 4.92. The
shipped headline figure was one draw from that spread, and it understated the model.

These tests pin the reporting path to repeated scaffold-grouped cross validation and
pin the instability of the single split, so nobody quietly reverts to it.
"""

from __future__ import annotations

import numpy as np
import pytest

from ai.predictors.data_connectors import ScientificDataManager
from ai.predictors.qsar import CalibratedQSARPredictor


@pytest.fixture(scope="module")
def pdac_training_data():
    manager = ScientificDataManager()
    manager.load_reference_dataset(disease_id="PDAC")
    smiles, pic50, active = [], [], []
    for record in manager.get_records():
        assay = record.get("bioactivity") or {}
        if record.get("canonical_smiles") and assay.get("pic50") is not None:
            smiles.append(record["canonical_smiles"])
            pic50.append(float(assay["pic50"]))
            active.append(int(bool(assay.get("growth_inhibition_active"))))
    assert len(smiles) >= 40
    return smiles, pic50, active


def test_single_split_metrics_are_unstable(pdac_training_data):
    """The reason the reporting path does not use a single split."""
    smiles, pic50, active = pdac_training_data

    r2_values = []
    for state in range(6):
        predictor = CalibratedQSARPredictor(random_state=state)
        metrics = predictor.evaluate_scaffold_split(smiles, pic50, active)
        r2_values.append(metrics["r2"])
        # Each split tests a handful of compounds, which is the root of the problem.
        assert metrics["test_compounds"] <= 15

    spread = max(r2_values) - min(r2_values)
    assert spread > 0.3, (
        "the single split has become stable; if the dataset grew, revisit whether the "
        "repeated CV path is still necessary"
    )


def test_reported_metrics_use_repeated_grouped_cv(pdac_training_data):
    smiles, pic50, active = pdac_training_data
    predictor = CalibratedQSARPredictor(random_state=42)
    predictor.fit(smiles, pic50, active, evaluate_split=True)

    metrics = predictor.validation_metrics
    assert "repeated scaffold-grouped CV" in metrics["validation_scheme"]
    assert metrics["n_repeats"] >= 5 and metrics["n_folds"] >= 3
    assert metrics["independent_scaffolds"] > 0
    assert metrics["scaffold_leakage"] == 0

    # Pooled over repeats, so far more test predictions than compounds.
    assert metrics["test_compounds"] > len(smiles)


def test_cv_metrics_are_reproducible(pdac_training_data):
    """Same seed, same numbers. Reported figures must not drift between runs."""
    smiles, pic50, active = pdac_training_data

    first = CalibratedQSARPredictor(random_state=42).evaluate_scaffold_cv(
        smiles, pic50, active, n_folds=5, n_repeats=3)
    second = CalibratedQSARPredictor(random_state=42).evaluate_scaffold_cv(
        smiles, pic50, active, n_folds=5, n_repeats=3)

    for key in ("rmse", "r2", "roc_auc", "conformal_q_95"):
        assert first[key] == second[key], f"{key} is not reproducible"


def test_cv_beats_predicting_the_mean(pdac_training_data):
    """A model that cannot beat the training mean should not be shipped."""
    smiles, pic50, active = pdac_training_data

    metrics = CalibratedQSARPredictor(random_state=42).evaluate_scaffold_cv(
        smiles, pic50, active, n_folds=5, n_repeats=4)

    baseline_rmse = float(np.std(pic50))
    assert metrics["rmse"] < baseline_rmse, (
        f"RMSE {metrics['rmse']} does not beat the spread of the labels "
        f"({baseline_rmse:.3f}); the regressor carries no usable signal"
    )
    assert metrics["r2"] > 0.0


def test_conformal_interval_is_reported_and_wide(pdac_training_data):
    """Ordering is usable; single values are not. Both halves must stay true."""
    smiles, pic50, active = pdac_training_data
    metrics = CalibratedQSARPredictor(random_state=42).evaluate_scaffold_cv(
        smiles, pic50, active, n_folds=5, n_repeats=4)

    quantile = metrics["conformal_q_95"]
    assert quantile > 0.0
    # If this ever drops below one log unit on this dataset, the claim in the report
    # that point IC50 values are unusable would need revisiting rather than repeating.
    assert quantile > 1.0, (
        "the conformal interval narrowed substantially; update the report's wording "
        "instead of carrying the old caveat forward"
    )
