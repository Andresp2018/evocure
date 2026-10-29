"""Tests for the prior art check, with no network access.

The project's own novelty check compares a candidate against 111 local
structures and says, correctly, that this is a sanity check and not a prior art
search. scripts/check_prior_art.py asks the question the report could not: does
PubChem or ChEMBL already hold this structure, what is the nearest known
molecule, and how far did anyone take it.

Two things here matter more than the rest. A lookup failure must never be
reported as "not found", because that would turn a network error into a novelty
claim. And the Tanimoto comparison against the reference panel has to be
computed locally and exactly, since it is the number used to say how far the
candidates sit from the drugs PDAC is actually treated with.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


art = _load_script("check_prior_art")

GEMCITABINE = "Nc1ccn([C@@H]2O[C@H](CO)[C@@H](O)C2(F)F)c(=O)n1"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


class StubResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"status {self.status_code}")


class ExplodingClient:
    """Every request fails, which is how a lookup error reaches the code."""

    def get(self, url, params=None):
        raise RuntimeError("network unreachable")


# ---------------------------------------------------------------------------
# A failed lookup must not read as novelty
# ---------------------------------------------------------------------------

def test_pubchem_failure_is_unknown_not_absent():
    result = art.pubchem_exact(ExplodingClient(), GEMCITABINE)

    assert result["found"] is None, "a network failure was reported as 'not found'"
    assert "error" in result


def test_chembl_failure_is_unknown_not_absent():
    result = art.chembl_exact(ExplodingClient(), GEMCITABINE)

    assert result["found"] is None
    assert "error" in result


def test_a_404_from_pubchem_is_a_real_absence():
    """PubChem answers 404 for a structure it does not hold, which is an answer."""
    class NotFound:
        def get(self, url, params=None):
            return StubResponse({}, status=404)

    result = art.pubchem_exact(NotFound(), ASPIRIN)
    assert result["found"] is False
    assert result["cids"] == []


def test_pubchem_cid_zero_is_not_a_match():
    """PubChem returns CID 0 for 'parsed but not present'."""
    class ZeroCid:
        def get(self, url, params=None):
            return StubResponse({"IdentifierList": {"CID": [0]}})

    result = art.pubchem_exact(ZeroCid(), ASPIRIN)
    assert result["found"] is False


def test_a_pubchem_hit_is_reported_with_its_cid():
    class Hit:
        def get(self, url, params=None):
            return StubResponse({"IdentifierList": {"CID": [60750]}})

    result = art.pubchem_exact(Hit(), GEMCITABINE)
    assert result["found"] is True
    assert 60750 in result["cids"]


# ---------------------------------------------------------------------------
# Neighbours, and how far anyone took them
# ---------------------------------------------------------------------------

def test_neighbours_are_sorted_and_carry_the_highest_phase():
    class Neighbours:
        def get(self, url, params=None):
            return StubResponse({
                "page_meta": {"total_count": 3},
                "molecules": [
                    {"molecule_chembl_id": "CHEMBL2", "pref_name": "MID",
                     "similarity": "80.0", "max_phase": 2.0},
                    {"molecule_chembl_id": "CHEMBL1", "pref_name": "TOP",
                     "similarity": "95.5", "max_phase": 4.0},
                    {"molecule_chembl_id": "CHEMBL3", "pref_name": None,
                     "similarity": "72.1", "max_phase": None},
                ],
            })

    result = art.chembl_neighbours(Neighbours(), GEMCITABINE)

    assert result["count"] == 3
    assert [r["chembl_id"] for r in result["nearest"]] == ["CHEMBL1", "CHEMBL2", "CHEMBL3"]
    assert result["nearest"][0]["similarity"] == 95.5
    assert result["highest_phase_among_neighbours"] == 4.0


def test_no_neighbours_above_the_threshold_is_not_an_error():
    class Empty:
        def get(self, url, params=None):
            return StubResponse({"page_meta": {"total_count": 0}, "molecules": []})

    result = art.chembl_neighbours(Empty(), ASPIRIN)
    assert result["count"] == 0
    assert result["nearest"] == []
    assert result["highest_phase_among_neighbours"] is None


def test_a_similarity_error_status_does_not_raise():
    """ChEMBL rejects some structures; that must not abort the whole check."""
    class Rejects:
        def get(self, url, params=None):
            return StubResponse({}, status=500)

    result = art.chembl_neighbours(Rejects(), ASPIRIN)
    assert result["count"] == 0
    assert "note" in result


# ---------------------------------------------------------------------------
# The panel comparison, computed locally
# ---------------------------------------------------------------------------

def _panel():
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    entries = {
        "GEMCITABINE": ("chemotherapy backbone", GEMCITABINE),
        "SOTORASIB": ("targeted, KRAS G12C",
                      "CC1=CC(=CC=C1)N1C(=O)N(C2=NC(=NC3=C2N=CN3)N3CCN(CC3)C(=O)C=C)C=C1"),
    }
    panel = {}
    for name, (role, smiles) in entries.items():
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        panel[name] = {"role": role, "chembl_id": f"TEST-{name}",
                       "max_phase": 4.0, "smiles": smiles,
                       "fingerprint": gen.GetFingerprint(mol)}
    return panel


def test_a_structure_is_identical_to_itself():
    panel = _panel()
    rows = art.panel_similarity(GEMCITABINE, panel)

    assert rows, "the panel produced no comparison"
    assert rows[0]["drug"] == "GEMCITABINE"
    assert rows[0]["tanimoto"] == pytest.approx(1.0)


def test_rows_are_sorted_by_similarity_and_keep_the_role():
    panel = _panel()
    rows = art.panel_similarity(GEMCITABINE, panel)

    scores = [r["tanimoto"] for r in rows]
    assert scores == sorted(scores, reverse=True)
    assert all("role" in r and "chembl_id" in r for r in rows)


def test_an_unparseable_candidate_yields_no_comparison():
    assert art.panel_similarity("not a molecule", _panel()) == []
    assert art.panel_similarity("", _panel()) == []


def test_the_panel_covers_both_modalities():
    """The comparison is only meaningful with chemotherapy and KRAS agents in it."""
    roles = set(art.REFERENCE_PANEL.values())
    assert any("chemotherapy" in r for r in roles)
    assert any("KRAS" in r for r in roles)
