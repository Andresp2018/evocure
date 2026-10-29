"""Regression tests for unbounded molecular growth in the generator.

Measured before the fix, with the operators driven directly:

  Four successive crossovers took erlotinib from 349 to 1131 Da, growing
  monotonically, because the fragment operator returned the first valid
  fragment pair and that is biased towards the largest fragment of each parent.

  Ten generations with selection favouring size reached a mean molecular weight
  of 76 kDa, from a seed population averaging 403 Da.

Nothing in the search bounded this. The only size pressure anywhere was QED, an
indirect graded preference that a caller can zero out through the public
objective_weights argument, and the applicability-domain gate at 1200 Da only
relabels a compound as out of domain rather than keeping it out of the
population. This is the mechanism behind the fused dimers in the project's own
candidate review.

These tests pin the bound itself, not the particular numbers above, and they
check that an oversized seed can still be worked on rather than freezing the
search.
"""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import Descriptors

from simulation.evolution.molecular_ga import (
    DEFAULT_ENVELOPE,
    ChemicalEnvelope,
    DirectedMolecularOptimizer,
)

ERLOTINIB = "COc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC"
GEMCITABINE = "Nc1ccn([C@@H]2O[C@H](CO)[C@@H](O)C2(F)F)c(=O)n1"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
SEEDS = [GEMCITABINE, "O=c1[nH]cc(F)c(=O)[nH]1", ERLOTINIB, ASPIRIN]


def mw(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    return Descriptors.MolWt(mol)


# ---------------------------------------------------------------------------
# The envelope itself
# ---------------------------------------------------------------------------

def test_the_default_envelope_admits_real_drugs():
    """A bound that rejects gemcitabine or erlotinib is the wrong bound."""
    for smiles in (GEMCITABINE, ERLOTINIB, ASPIRIN,
                   "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"):  # ibuprofen
        assert DEFAULT_ENVELOPE.admits_smiles(smiles), smiles


def test_the_envelope_names_what_a_molecule_breaks():
    dimer = ERLOTINIB + "." + ERLOTINIB
    violations = DEFAULT_ENVELOPE.violations(Chem.MolFromSmiles(dimer))
    assert "disconnected" in violations
    assert "mol_weight" in violations

    assert "too_small" in DEFAULT_ENVELOPE.violations(Chem.MolFromSmiles("CO"))
    assert DEFAULT_ENVELOPE.violations(None) == ("unparseable",)


# ---------------------------------------------------------------------------
# Operators stay inside it
# ---------------------------------------------------------------------------

def test_repeated_crossover_does_not_inflate_the_molecule():
    """The original defect: every crossover grew the product."""
    current = ERLOTINIB
    for step in range(8):
        current, _label = DirectedMolecularOptimizer.recombine_smiles(
            current, ERLOTINIB, rng=np.random.default_rng(100 + step))
        assert DEFAULT_ENVELOPE.admits_smiles(current), (
            f"step {step} left the envelope at {mw(current):.1f} Da: {current}")

    assert mw(current) <= DEFAULT_ENVELOPE.max_mol_weight


def test_a_long_run_with_size_favouring_selection_stays_bounded():
    """Selection that rewards size must not be able to run the generator away.

    Keeping the heaviest half each generation is the worst case for a generator
    with no bound, and is what produced 76 kDa before the fix.
    """
    rng = np.random.default_rng(7)
    population = list(SEEDS)

    for _generation in range(10):
        children = []
        for _ in range(16):
            if len(population) >= 2 and rng.random() < 0.65:
                a, b = rng.choice(population, size=2, replace=False)
                child, _ = DirectedMolecularOptimizer.recombine_smiles(
                    str(a), str(b), rng=rng)
            else:
                child, _ = DirectedMolecularOptimizer.mutate_smiles(
                    str(rng.choice(population)), rng=rng)
            children.append(child)

        pool = list(dict.fromkeys(population + children))
        population = sorted(pool, key=mw, reverse=True)[:10]

        for smiles in population:
            assert DEFAULT_ENVELOPE.admits_smiles(smiles), (
                f"{mw(smiles):.1f} Da escaped the envelope: {smiles}")

    assert max(mw(s) for s in population) <= DEFAULT_ENVELOPE.max_mol_weight


@pytest.mark.parametrize("seed", [1, 2, 3, 11, 29])
def test_every_mutation_and_crossover_product_is_admissible(seed):
    rng = np.random.default_rng(seed)
    for parent in SEEDS:
        mutated, _ = DirectedMolecularOptimizer.mutate_smiles(parent, rng)
        assert Chem.MolFromSmiles(mutated) is not None
        assert DEFAULT_ENVELOPE.admits_smiles(mutated)

    for a in SEEDS:
        for b in SEEDS:
            child, _ = DirectedMolecularOptimizer.recombine_smiles(a, b, rng)
            assert Chem.MolFromSmiles(child) is not None
            assert DEFAULT_ENVELOPE.admits_smiles(child)


# ---------------------------------------------------------------------------
# The bound must not freeze the search
# ---------------------------------------------------------------------------

def test_an_oversized_parent_can_still_be_altered():
    """A seed above the bound must be workable, or the population cannot move.

    Rejecting everything from an inadmissible parent would leave an imported
    oversized seed permanently stuck, so a product is also accepted when it is
    no worse than the parent it came from.
    """
    oversized = ".".join([ERLOTINIB] * 3)
    assert not DEFAULT_ENVELOPE.admits_smiles(oversized)

    altered = 0
    for seed in range(12):
        child, label = DirectedMolecularOptimizer.mutate_smiles(
            oversized, np.random.default_rng(seed))
        if child != oversized:
            altered += 1
            assert Chem.MolFromSmiles(child) is not None
            assert label != "unmodified_envelope_bound"

    assert altered, "an oversized parent was frozen rather than worked on"


def test_mutation_reports_when_the_bound_blocked_it(monkeypatch):
    """A caller must be able to tell a refusal from a no-op.

    Driven with a stub operator that only ever returns an inadmissible product,
    so the refusal branch is exercised without depending on which mutation the
    random strategy order happens to pick.
    """
    monkeypatch.setattr(
        DirectedMolecularOptimizer, "_mutate_once",
        staticmethod(lambda smiles, rng: (".".join([ERLOTINIB] * 4), "stub_dimer")),
    )

    child, label = DirectedMolecularOptimizer.mutate_smiles(
        ASPIRIN, np.random.default_rng(3))

    assert child == ASPIRIN, "an inadmissible product was let through"
    assert label == "unmodified_envelope_bound"


def test_crossover_degrades_to_a_bounded_mutation_not_a_weld(monkeypatch):
    """When no crossover is admissible, the fallback must stay inside the bound."""
    monkeypatch.setattr(
        DirectedMolecularOptimizer, "_recombine_once",
        staticmethod(lambda a, b, rng: (a + "." + b, "stub_weld")),
    )

    child, label = DirectedMolecularOptimizer.recombine_smiles(
        ERLOTINIB, GEMCITABINE, rng=np.random.default_rng(5))

    assert DEFAULT_ENVELOPE.admits_smiles(child)
    assert label.startswith("crossover_envelope_bound_")


def test_a_widened_envelope_restores_the_previous_behaviour():
    """The bound is a default, not a hard-wired policy."""
    unbounded = ChemicalEnvelope(
        max_mol_weight=float("inf"), max_heavy_atoms=10**6,
        max_rotatable_bonds=10**6, max_rings=10**6,
        min_heavy_atoms=0, allow_disconnected=True)

    current = ERLOTINIB
    for step in range(4):
        current, _ = DirectedMolecularOptimizer.recombine_smiles(
            current, ERLOTINIB, rng=np.random.default_rng(100 + step),
            envelope=unbounded)

    assert unbounded.admits_smiles(current)


# ---------------------------------------------------------------------------
# The constructor carries it, so a caller can configure one place
# ---------------------------------------------------------------------------

def test_the_optimizer_exposes_its_envelope():
    assert DirectedMolecularOptimizer().envelope is DEFAULT_ENVELOPE

    custom = ChemicalEnvelope(max_mol_weight=450.0)
    assert DirectedMolecularOptimizer(envelope=custom).envelope is custom
