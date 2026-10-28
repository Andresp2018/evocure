"""Make the molecule decide the fate of the entity carrying it.

Until now the artificial life engine evolved four free-floating numbers,
catalytic_efficiency, stability_score, replication_threshold and decay_rate, by
adding Gaussian noise at each division. The entity's molecular structure was
mutated in parallel, recorded in the genealogy, and never read back. Measured
directly: the module that decides who survives contained zero references to
SMILES, molecular weight, logP or QED.

That meant the simulation was a numeric genetic algorithm with molecule names
painted on it. Replacing every structure with a random label would have changed
nothing. There was no genotype to phenotype map, which is the thing that makes a
chemical evolution simulation chemical.

This module is that map. Each survival trait is derived from the structure by a
stated chemical rationale:

  catalytic_efficiency   groups that can donate, accept or shuttle, plus the
                         polar surface available to interact with a substrate
  stability_score        aromatic and ring content resist degradation, strained
                         rings and labile linkages do not
  decay_rate             hydrolysable bonds set the rate of falling apart
  replication_threshold  a larger, more complex molecule costs more to copy

What this is. A deterministic, defensible heuristic mapping, so that selection
acts on chemistry instead of on noise. Two entities with the same structure get
the same traits, and a mutation that changes the structure changes the traits.

What this is not. Biochemistry. These are not measured rate constants, there is
no enzymology here, and nothing is calibrated against an experiment. The mapping
is a model of the form "this kind of structure should fare better than that
kind", which is enough to make evolution structure-driven and not enough to say
anything about real molecules.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

# Substructures that make a molecule come apart in water. Rates are not modelled;
# the count sets a relative lability, which is what selection needs.
_LABILE_PATTERNS: tuple[tuple[str, str, float], ...] = (
    ("anhydride", "[CX3](=O)[OX2][CX3](=O)", 3.0),
    ("acyl_halide", "[CX3](=O)[F,Cl,Br,I]", 3.0),
    ("epoxide", "C1OC1", 2.5),
    ("ester", "[CX3](=O)[OX2H0][#6]", 1.5),
    ("thioester", "[CX3](=O)[SX2][#6]", 1.8),
    ("carbamate", "[NX3][CX3](=O)[OX2]", 1.0),
    ("imine", "[CX3]=[NX2]", 1.2),
    ("acetal", "[CX4]([OX2])([OX2])", 1.3),
    ("amide", "[NX3][CX3](=O)[#6]", 0.4),
)

# Groups that can take part in catalysis: acid/base pairs, nucleophiles, and
# systems that can delocalise charge.
_CATALYTIC_PATTERNS: tuple[tuple[str, str, float], ...] = (
    ("carboxylic_acid", "[CX3](=O)[OX2H1]", 1.0),
    ("primary_amine", "[NX3;H2][#6]", 0.9),
    ("secondary_amine", "[NX3;H1]([#6])[#6]", 0.7),
    ("hydroxyl", "[OX2H][#6]", 0.6),
    ("thiol", "[SX2H]", 1.1),
    ("imidazole", "c1cnc[nH]1", 1.4),
    ("phosphate", "[PX4](=O)([OX2])([OX2])", 1.2),
    ("aromatic_nitrogen", "[nX2]", 0.5),
)

_STRAINED_RINGS: tuple[str, ...] = ("C1CC1", "C1CCC1", "C1OC1", "C1NC1")


@dataclass(frozen=True)
class ChemicalPhenotype:
    """Survival traits derived from a structure, with the reasoning retained."""

    catalytic_efficiency: float
    stability_score: float
    decay_rate: float
    replication_threshold: float
    heavy_atoms: int
    catalytic_groups: float
    labile_groups: float
    aromatic_fraction: float
    rationale: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "catalytic_efficiency": self.catalytic_efficiency,
            "stability_score": self.stability_score,
            "decay_rate": self.decay_rate,
            "replication_threshold": self.replication_threshold,
            "heavy_atoms": self.heavy_atoms,
            "catalytic_groups": self.catalytic_groups,
            "labile_groups": self.labile_groups,
            "aromatic_fraction": self.aromatic_fraction,
            "rationale": self.rationale,
            "caveat": (
                "A heuristic structure to trait mapping, not biochemistry. It makes "
                "selection act on chemistry rather than on noise; it is not calibrated "
                "against any measurement and says nothing about real molecules."
            ),
        }


@lru_cache(maxsize=512)
def _compiled(patterns: tuple[tuple[str, str, float], ...]):
    out = []
    for name, smarts, weight in patterns:
        query = Chem.MolFromSmarts(smarts)
        if query is not None:
            out.append((name, query, weight))
    return tuple(out)


def _weighted_matches(mol, patterns) -> tuple[float, list[str]]:
    total, found = 0.0, []
    for name, query, weight in _compiled(patterns):
        hits = len(mol.GetSubstructMatches(query))
        if hits:
            total += weight * hits
            found.append(f"{name} x{hits}")
    return total, found


# Defaults used when a structure cannot be parsed. Deliberately mediocre rather
# than favourable, so an unreadable molecule is never advantaged by accident.
FALLBACK = ChemicalPhenotype(
    catalytic_efficiency=0.5, stability_score=0.4, decay_rate=0.03,
    replication_threshold=24.0, heavy_atoms=0, catalytic_groups=0.0,
    labile_groups=0.0, aromatic_fraction=0.0,
    rationale={"structure": "unparseable, fallback traits applied"},
)


@lru_cache(maxsize=4096)
def phenotype_from_smiles(smiles: str) -> ChemicalPhenotype:
    """Derive survival traits from a structure. Deterministic and cached."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return FALLBACK

    heavy = mol.GetNumHeavyAtoms()
    if heavy == 0:
        return FALLBACK

    catalytic_score, catalytic_found = _weighted_matches(mol, _CATALYTIC_PATTERNS)
    labile_score, labile_found = _weighted_matches(mol, _LABILE_PATTERNS)

    aromatic_atoms = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    aromatic_fraction = aromatic_atoms / heavy
    rings = rdMolDescriptors.CalcNumRings(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    logp = Crippen.MolLogP(mol)

    strained = sum(len(mol.GetSubstructMatches(q))
                   for q in (Chem.MolFromSmarts(s) for s in _STRAINED_RINGS)
                   if q is not None)

    # Catalytic efficiency: functional groups that can act, scaled by the polar
    # surface available to meet a substrate, normalised by size so that a large
    # inert molecule is not rewarded for having many groups in absolute terms.
    density = catalytic_score / max(1.0, heavy / 10.0)
    surface = min(1.0, tpsa / 140.0)
    catalytic = 0.25 + 0.9 * min(2.0, density) * (0.5 + 0.5 * surface)
    catalytic = round(max(0.05, min(2.5, catalytic)), 4)

    # Stability: aromatic and ring systems resist degradation; strained rings and
    # labile linkages do not. Extreme lipophilicity is treated as destabilising
    # in an aqueous compartment.
    stability = 0.35 + 0.35 * aromatic_fraction + 0.06 * min(4, rings)
    stability -= 0.08 * min(5.0, labile_score)
    stability -= 0.10 * min(3, strained)
    stability -= 0.04 * max(0.0, abs(logp) - 4.0)
    stability = round(max(0.05, min(0.98, stability)), 4)

    # Decay: driven by hydrolysable bonds, buffered by how stable the rest is.
    decay = 0.004 + 0.009 * min(5.0, labile_score)
    decay *= (1.25 - 0.5 * stability)
    decay = round(max(0.001, min(0.12, decay)), 5)

    # Replication threshold: the energetic cost of building a copy rises with
    # size and ring complexity. A large molecule must accumulate more before it
    # can divide, which is a real cost and not a free parameter.
    threshold = 8.0 + 0.42 * heavy + 1.1 * rings
    threshold = round(max(6.0, min(80.0, threshold)), 3)

    return ChemicalPhenotype(
        catalytic_efficiency=catalytic,
        stability_score=stability,
        decay_rate=decay,
        replication_threshold=threshold,
        heavy_atoms=heavy,
        catalytic_groups=round(catalytic_score, 3),
        labile_groups=round(labile_score, 3),
        aromatic_fraction=round(aromatic_fraction, 4),
        rationale={
            "catalytic_efficiency": (
                f"catalytic groups [{', '.join(catalytic_found) or 'none'}], "
                f"TPSA {tpsa:.0f}, {heavy} heavy atoms"),
            "stability_score": (
                f"{aromatic_fraction:.0%} aromatic, {rings} ring(s), "
                f"labile [{', '.join(labile_found) or 'none'}], "
                f"{strained} strained ring(s), cLogP {logp:.1f}"),
            "decay_rate": f"hydrolysable load {labile_score:.1f}, buffered by stability",
            "replication_threshold": f"{heavy} heavy atoms and {rings} ring(s) to rebuild",
        },
    )


def phenotype_from_compound(compound: Any) -> ChemicalPhenotype:
    """Convenience wrapper for a CompoundRecord or anything with canonical_smiles."""
    return phenotype_from_smiles(getattr(compound, "canonical_smiles", "") or "")


def apply_to_entity(entity: Any, smiles: str) -> ChemicalPhenotype:
    """Write the derived traits onto a SpatialEntity, in place."""
    phenotype = phenotype_from_smiles(smiles)
    entity.catalytic_efficiency = phenotype.catalytic_efficiency
    entity.stability_score = phenotype.stability_score
    entity.decay_rate = phenotype.decay_rate
    entity.replication_threshold = phenotype.replication_threshold
    return phenotype
