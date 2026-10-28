"""Medicinal chemistry filters must flag without quietly deleting real chemistry.

These rule sets have well documented false positives. Measured here, the PAINS,
Brenk and NIH catalogues between them flag aspirin, quercetin and paclitaxel, and
paclitaxel is a pancreatic cancer standard of care. A filter that hard rejected
every hit would remove approved drugs from a drug discovery platform.

So the contract is: score everything, flag honestly, reject only on request, and
never claim that passing means a molecule is good.
"""

from __future__ import annotations

import pytest

from simulation.chemistry.medchem_filters import (
    SA_HARD_THRESHOLD,
    assess,
    plausibility_objective,
    screen,
)

GEMCITABINE = "C1=CN(C(=O)N=C1N)[C@H]2C([C@@H]([C@H](O2)CO)O)(F)F"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
QUERCETIN = "O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12"
PACLITAXEL = (
    "CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1OC(=O)C(C(c1ccccc1)NC(=O)c1ccccc1)O)O)"
    "OC(=O)c1ccccc1)(CO4)OC(C)=O)O)C)OC(C)=O"
)
# A gemcitabine-derived variant from the published 100 generation run.
GENERATED = "Nc1ccn([C@H]2O[C@@H](n3ccc(N)nc3=O)C(F)(F)[C@H]2C(=O)O)c(=O)n1"
DISCONNECTED = "CCO.CCN"


def test_invalid_input_never_raises():
    for bad in ("not_a_molecule", "", "C(((", "@@@"):
        result = assess(bad)
        assert result.is_valid is False
        assert result.verdict == "not a parseable structure"
        assert plausibility_objective(result) == 0.0


def test_a_clean_drug_passes():
    result = assess(GEMCITABINE)
    assert result.is_valid
    assert result.alert_count == 0
    assert result.sa_score is not None and result.sa_score < SA_HARD_THRESHOLD
    assert result.passes is True


def test_known_false_positives_are_flagged_but_not_deleted():
    """The reason filtering is opt in: these are all real, useful molecules."""
    for smiles, name in ((ASPIRIN, "aspirin"), (QUERCETIN, "quercetin")):
        result = assess(smiles)
        assert result.is_valid, name
        assert result.alert_count > 0, f"{name} should trip a published alert set"
        # Flagged, yet still scored above zero so a search can weigh it.
        assert plausibility_objective(result) > 0.0, name


def test_hard_rejection_would_remove_a_standard_of_care():
    """Paclitaxel is used in pancreatic cancer and does not pass a strict filter."""
    result = assess(PACLITAXEL)
    assert result.is_valid
    assert result.passes is False

    strict = screen([PACLITAXEL], reject_hard=True)
    assert strict["removed"] == 1

    lenient = screen([PACLITAXEL], reject_hard=False)
    assert lenient["removed"] == 0
    assert lenient["kept"] == 1


def test_synthetic_accessibility_orders_easy_before_hard():
    easy = assess(ASPIRIN).sa_score
    hard = assess(PACLITAXEL).sa_score
    assert easy is not None and hard is not None
    assert easy < hard, "aspirin must score as easier to make than paclitaxel"


def test_structural_oddities_the_generator_produces_are_flagged():
    result = assess(PACLITAXEL)
    joined = " ".join(result.flags)
    assert "stereocentres" in joined
    assert "heavy atoms" in joined

    multi = assess(DISCONNECTED)
    assert any("disconnected" in f for f in multi.flags)
    # A salt or mixture should score below a single clean molecule.
    assert plausibility_objective(multi) < plausibility_objective(assess(GEMCITABINE))


def test_objective_is_bounded_and_ordered():
    for smiles in (GEMCITABINE, ASPIRIN, QUERCETIN, PACLITAXEL, GENERATED):
        value = plausibility_objective(assess(smiles))
        assert 0.0 <= value <= 1.0

    clean = plausibility_objective(assess(GEMCITABINE))
    messy = plausibility_objective(assess(QUERCETIN))
    assert clean > messy, "a clean structure must outrank one with three alert families"


def test_generated_candidate_is_assessed_not_assumed():
    """The published candidates were never checked for this before."""
    result = assess(GENERATED)
    assert result.is_valid
    assert result.sa_score is not None
    assert result.verdict != "not assessed"


def test_screen_summarises_without_hiding_anything():
    report = screen([GEMCITABINE, ASPIRIN, QUERCETIN, PACLITAXEL, "broken"])

    assert report["total"] == 5
    assert report["with_any_alert"] >= 3
    assert report["sa_median"] is not None
    assert len(report["assessments"]) == 5
    # The invalid structure is removed even in lenient mode.
    assert report["removed"] == 1
    assert "false positives" in report["note"] or "approved drugs" in report["note"]


def test_every_assessment_carries_its_caveat():
    payload = assess(GEMCITABINE).to_dict()
    assert "not that the compound can be made or works" in payload["caveat"]
    assert payload["passes"] is True
