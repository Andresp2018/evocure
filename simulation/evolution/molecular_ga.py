"""
Directed Molecular Optimization (NSGA-II) with RDKit Chemically Validated Operators.

Implements Section 6.1 (Mode B Grounded Chemistry):
- RDKit chemically validated functional group mutation, bioisosteric substitution, and bond modification.
- Chemically sanitized BRICS / single-bond fragment crossover recombination.
- Full NSGA-II multi-objective optimization (Pareto non-dominated sorting and crowding distance).
- Strict chemical validity (100% RDKit sanitization guarantee; invalid valences rejected).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import numpy as np

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors

from simulation.core.contracts import EpistemicCategory


@dataclass(frozen=True)
class ChemicalEnvelope:
    """The region of chemical space the search is allowed to propose.

    The operators had no size constraint of any kind, and nothing downstream
    supplied one: the only bound on molecular size anywhere in the search was
    QED, an indirect soft preference that a caller can switch off through the
    public objective_weights argument. Measured on the operators alone, four
    successive crossovers take erlotinib from 349 to 1131 Da, and ten
    generations under size-favouring selection reach a mean of 76 kDa. A
    generator that can wander to 76 kDa is not proposing drugs, and this is the
    mechanism behind the fused dimers the project's own candidate review flags.

    The bounds are the published ones, loosened where oncology practice has
    moved past them, and they are deliberately generous rather than tight: this
    is a boundary on what the search may emit, not a drug-likeness score. QED
    and the medicinal chemistry filters remain the graded signals.

      max_mol_weight        Lipinski's rule of five sets 500 Da. Approved
                            oncology small molecules routinely exceed it, so
                            600 Da admits that chemistry while still excluding
                            dimerisation products.
      max_heavy_atoms       45 heavy atoms is roughly 600 Da for typical
                            organics, and bounds the graph independently of
                            halogen mass.
      max_rotatable_bonds   Veber sets 10 for oral bioavailability; 12 leaves
                            headroom for the flexible linkers in kinase
                            inhibitors.
      max_rings             6 rings, counted as SSSR rings rather than ring
                            systems, is already a complex polycycle.
      min_heavy_atoms       Below this a fragment is not a candidate. The
                            search otherwise drifts to tiny molecules that
                            score well on size-penalised objectives.
      allow_disconnected    Multi-component SMILES are salts or mixtures, not
                            one molecule, and the tissue model doses a single
                            species.
    """

    max_mol_weight: float = 600.0
    max_heavy_atoms: int = 45
    max_rotatable_bonds: int = 12
    max_rings: int = 6
    min_heavy_atoms: int = 5
    allow_disconnected: bool = False

    def violations(self, mol: Any) -> tuple[str, ...]:
        """Which bounds this molecule breaks. Empty means it is admitted."""
        if mol is None:
            return ("unparseable",)
        broken: list[str] = []
        heavy = mol.GetNumHeavyAtoms()
        if heavy > self.max_heavy_atoms:
            broken.append("heavy_atoms")
        if heavy < self.min_heavy_atoms:
            broken.append("too_small")
        if Descriptors.MolWt(mol) > self.max_mol_weight:
            broken.append("mol_weight")
        if rdMolDescriptors.CalcNumRotatableBonds(mol) > self.max_rotatable_bonds:
            broken.append("rotatable_bonds")
        if rdMolDescriptors.CalcNumRings(mol) > self.max_rings:
            broken.append("rings")
        if not self.allow_disconnected and len(Chem.GetMolFrags(mol)) > 1:
            broken.append("disconnected")
        return tuple(broken)

    def admits(self, mol: Any) -> bool:
        return not self.violations(mol)

    def admits_smiles(self, smiles: str) -> bool:
        return self.admits(Chem.MolFromSmiles(smiles) if smiles else None)


# Applied by every operator unless a caller passes its own.
DEFAULT_ENVELOPE = ChemicalEnvelope()


def _accepts_product(product: Any, parents: list[Any],
                     envelope: ChemicalEnvelope) -> bool:
    """Whether an operator's product may be returned.

    Inside the envelope is always acceptable. A product that is outside it is
    accepted only when its parents were already outside on the same bounds and
    it is no larger than the largest of them, so that a seed imported above the
    bounds can still be mutated towards compliance instead of freezing the
    search. Without that allowance, an oversized seed would make every operator
    return the parent unchanged and the population could never move.
    """
    if product is None:
        return False
    broken = envelope.violations(product)
    if not broken:
        return True

    parent_broken: set[str] = set()
    largest_parent = 0
    for parent in parents:
        if parent is None:
            continue
        parent_broken.update(envelope.violations(parent))
        largest_parent = max(largest_parent, parent.GetNumHeavyAtoms())

    if not set(broken).issubset(parent_broken):
        return False
    return product.GetNumHeavyAtoms() <= max(largest_parent, envelope.max_heavy_atoms)


class DirectedMolecularOptimizer:
    """
    NSGA-II Multi-Objective Directed Molecular Optimizer for Grounded Chemistry (Mode B).
    
    Optimizes multi-property candidate profiles across Pareto frontiers with
    guaranteed RDKit chemical sanitization and valence validity.
    """

    # Functional groups for attachment and swap
    FUNCTIONAL_GROUPS: dict[str, str] = {
        "-F": "*F",
        "-Cl": "*Cl",
        "-OH": "*O",
        "-NH2": "*N",
        "-CH3": "*C",
        "-OCH3": "*OC",
        "-CF3": "*C(F)(F)F",
        "-CN": "*C#N",
        "-C(=O)O": "*C(=O)O",
    }

    def __init__(self, reference_dataset_smiles: list[str] | None = None,
                 envelope: ChemicalEnvelope | None = None) -> None:
        self.morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        self.envelope = envelope or DEFAULT_ENVELOPE
        self.reference_dataset_smiles = reference_dataset_smiles or []
        self._reference_fps: list[Any] = []
        if self.reference_dataset_smiles:
            for s in self.reference_dataset_smiles:
                m = Chem.MolFromSmiles(s)
                if m is not None:
                    self._reference_fps.append(self.morgan_gen.GetFingerprint(m))

    # -------------------------------------------------------------------------
    # Envelope-constrained public operators
    # -------------------------------------------------------------------------

    @staticmethod
    def mutate_smiles(smiles: str, rng: np.random.Generator,
                      envelope: ChemicalEnvelope | None = None,
                      attempts: int = 6) -> tuple[str, str]:
        """A chemically valid mutation that stays inside the chemical envelope.

        The underlying operators are unchanged; their products are now checked
        against the envelope and retried on rejection. When no attempt lands
        inside it the parent is returned unchanged, which is the honest outcome:
        the search found no admissible move from here, and inventing an
        oversized one would put a molecule nobody can make into the population.

        envelope defaults to DEFAULT_ENVELOPE. Pass a ChemicalEnvelope to widen
        or tighten it, or ChemicalEnvelope(max_mol_weight=float("inf"), ...) to
        recover the previous unbounded behaviour.
        """
        bounds = envelope or DEFAULT_ENVELOPE
        parent = Chem.MolFromSmiles(smiles) if smiles else None
        if parent is None:
            return smiles, "invalid_input"

        rejected = False
        for _ in range(max(1, attempts)):
            candidate, label = DirectedMolecularOptimizer._mutate_once(smiles, rng)
            if candidate == smiles:
                continue
            product = Chem.MolFromSmiles(candidate)
            if _accepts_product(product, [parent], bounds):
                return candidate, label
            rejected = True

        # Two different outcomes, and a caller auditing a run needs to tell them
        # apart: the envelope turned every product down, or the operators found
        # no move to make from this structure at all.
        return smiles, ("unmodified_envelope_bound" if rejected
                        else "unmodified_identity")

    @staticmethod
    def recombine_smiles(smiles_a: str, smiles_b: str, rng: np.random.Generator,
                         envelope: ChemicalEnvelope | None = None,
                         attempts: int = 6) -> tuple[str, str]:
        """A crossover whose product stays inside the chemical envelope.

        Crossover is where molecular size ran away: the fragment operator
        recombines the first fragment pair it finds valid, which is biased
        towards the largest fragment of each parent, and the fallback linking
        operator welds two whole molecules together. Both are now bounded, and
        a rejected crossover degrades to a bounded mutation of the first parent
        rather than to an unbounded weld.
        """
        bounds = envelope or DEFAULT_ENVELOPE
        parent_a = Chem.MolFromSmiles(smiles_a) if smiles_a else None
        parent_b = Chem.MolFromSmiles(smiles_b) if smiles_b else None
        if parent_a is None or parent_b is None:
            return smiles_a, "invalid_input"
        parents = [parent_a, parent_b]

        for _ in range(max(1, attempts)):
            candidate, label = DirectedMolecularOptimizer._recombine_once(
                smiles_a, smiles_b, rng)
            if candidate in (smiles_a, smiles_b):
                continue
            product = Chem.MolFromSmiles(candidate)
            if _accepts_product(product, parents, bounds):
                return candidate, label

        mutated, label = DirectedMolecularOptimizer.mutate_smiles(
            smiles_a, rng, envelope=bounds, attempts=attempts)
        return mutated, f"crossover_envelope_bound_{label}"

    # -------------------------------------------------------------------------
    # RDKit Chemical Mutation Operators
    # -------------------------------------------------------------------------

    @staticmethod
    def _mutate_once(smiles: str, rng: np.random.Generator) -> tuple[str, str]:
        """One unconstrained mutation attempt, verified only by Chem.SanitizeMol.

        Chemical validity but no size or property bound. Callers go through
        mutate_smiles, which applies the chemical envelope.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles, "invalid_input"

        # Mutation strategy candidates
        strategies = [0, 1, 2, 3]
        rng.shuffle(strategies)

        for strategy in strategies:
            try:
                if strategy == 0:
                    # Strategy 0: Functional Group Attachment
                    candidate_atoms = [
                        a.GetIdx()
                        for a in mol.GetAtoms()
                        if a.GetTotalNumHs() > 0 and a.GetAtomicNum() in (6, 7)
                    ]
                    if not candidate_atoms:
                        candidate_atoms = [
                            a.GetIdx() for a in mol.GetAtoms() if a.GetTotalNumHs() > 0
                        ]
                    if candidate_atoms:
                        target_idx = int(rng.choice(candidate_atoms))
                        fg_keys = list(DirectedMolecularOptimizer.FUNCTIONAL_GROUPS.keys())
                        fg_name = str(rng.choice(fg_keys))
                        fg_smiles = DirectedMolecularOptimizer.FUNCTIONAL_GROUPS[fg_name]
                        fg_mol = Chem.MolFromSmiles(fg_smiles)

                        combo = Chem.CombineMols(mol, fg_mol)
                        rw = Chem.RWMol(combo)
                        dummies = [a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 0]
                        fg_att = rw.GetAtomWithIdx(dummies[0]).GetNeighbors()[0].GetIdx()
                        rw.AddBond(target_idx, fg_att, Chem.BondType.SINGLE)
                        rw.RemoveAtom(dummies[0])
                        prod = rw.GetMol()
                        Chem.SanitizeMol(prod)
                        res_smi = Chem.MolToSmiles(prod, canonical=True)
                        if res_smi and res_smi != smiles:
                            return res_smi, f"attach_{fg_name}"

                elif strategy == 1:
                    # Strategy 1: Bioisosteric Atom Substitution (C <-> N, O <-> S)
                    sub_candidates: list[tuple[int, int, str]] = []
                    for a in mol.GetAtoms():
                        # Aromatic C <-> N
                        if a.GetAtomicNum() == 6 and a.GetIsAromatic():
                            sub_candidates.append((a.GetIdx(), 7, "bioisostere_aromatic_C_to_N"))
                        elif a.GetAtomicNum() == 7 and a.GetIsAromatic():
                            sub_candidates.append((a.GetIdx(), 6, "bioisostere_aromatic_N_to_C"))
                        # O <-> S
                        elif a.GetAtomicNum() == 8:
                            sub_candidates.append((a.GetIdx(), 16, "bioisostere_O_to_S"))
                        elif a.GetAtomicNum() == 16:
                            sub_candidates.append((a.GetIdx(), 8, "bioisostere_S_to_O"))

                    if sub_candidates:
                        idx = int(rng.integers(0, len(sub_candidates)))
                        t_idx, new_atomic_num, label = sub_candidates[idx]
                        rw = Chem.RWMol(mol)
                        rw.GetAtomWithIdx(t_idx).SetAtomicNum(new_atomic_num)
                        prod = rw.GetMol()
                        Chem.SanitizeMol(prod)
                        res_smi = Chem.MolToSmiles(prod, canonical=True)
                        if res_smi and res_smi != smiles:
                            return res_smi, label

                elif strategy == 2:
                    # Strategy 2: Functional Group Swap / Substituent Pruning
                    terminal_bonds: list[tuple[int, int]] = []
                    for b in mol.GetBonds():
                        if not b.IsInRing() and b.GetBondType() == Chem.BondType.SINGLE:
                            a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
                            if a1.GetDegree() == 1 and a2.GetDegree() > 1:
                                terminal_bonds.append((a1.GetIdx(), a2.GetIdx()))
                            elif a2.GetDegree() == 1 and a1.GetDegree() > 1:
                                terminal_bonds.append((a2.GetIdx(), a1.GetIdx()))

                    if terminal_bonds:
                        idx = int(rng.integers(0, len(terminal_bonds)))
                        term_idx, core_idx = terminal_bonds[idx]
                        # Option A: Replace with another functional group
                        fg_keys = list(DirectedMolecularOptimizer.FUNCTIONAL_GROUPS.keys())
                        fg_name = str(rng.choice(fg_keys))
                        fg_mol = Chem.MolFromSmiles(DirectedMolecularOptimizer.FUNCTIONAL_GROUPS[fg_name])
                        combo = Chem.CombineMols(mol, fg_mol)
                        rw = Chem.RWMol(combo)
                        dummies = [a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 0]
                        fg_att = rw.GetAtomWithIdx(dummies[0]).GetNeighbors()[0].GetIdx()
                        rw.AddBond(core_idx, fg_att, Chem.BondType.SINGLE)
                        rw.RemoveAtom(dummies[0])
                        # Remove the original terminal atom
                        rw.RemoveAtom(term_idx)
                        prod = rw.GetMol()
                        Chem.SanitizeMol(prod)
                        res_smi = Chem.MolToSmiles(prod, canonical=True)
                        if res_smi and res_smi != smiles:
                            return res_smi, f"swap_substituent_for_{fg_name}"
                    
                elif strategy == 3:
                    # Strategy 3: Bond Modification (Acyclic single <-> double)
                    bonds_to_mod: list[tuple[int, Chem.BondType, str]] = []
                    for b in mol.GetBonds():
                        if not b.IsInRing():
                            a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
                            if b.GetBondType() == Chem.BondType.SINGLE:
                                if a1.GetTotalNumHs() > 0 and a2.GetTotalNumHs() > 0:
                                    bonds_to_mod.append((b.GetIdx(), Chem.BondType.DOUBLE, "bond_single_to_double"))
                            elif b.GetBondType() == Chem.BondType.DOUBLE:
                                bonds_to_mod.append((b.GetIdx(), Chem.BondType.SINGLE, "bond_double_to_single"))

                    if bonds_to_mod:
                        idx = int(rng.integers(0, len(bonds_to_mod)))
                        b_idx, new_type, label = bonds_to_mod[idx]
                        rw = Chem.RWMol(mol)
                        rw.GetBondWithIdx(b_idx).SetBondType(new_type)
                        prod = rw.GetMol()
                        Chem.SanitizeMol(prod)
                        res_smi = Chem.MolToSmiles(prod, canonical=True)
                        if res_smi and res_smi != smiles:
                            return res_smi, label

            except Exception:
                continue

        # Invariant Fallback: Guaranteed attachment to any accessible hydrogen
        candidate_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetTotalNumHs() > 0]
        if candidate_atoms:
            for fg_name in ["-F", "-OH", "-CH3", "-Cl"]:
                try:
                    target_idx = candidate_atoms[0]
                    fg_mol = Chem.MolFromSmiles(DirectedMolecularOptimizer.FUNCTIONAL_GROUPS[fg_name])
                    combo = Chem.CombineMols(mol, fg_mol)
                    rw = Chem.RWMol(combo)
                    dummies = [a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 0]
                    fg_att = rw.GetAtomWithIdx(dummies[0]).GetNeighbors()[0].GetIdx()
                    rw.AddBond(target_idx, fg_att, Chem.BondType.SINGLE)
                    rw.RemoveAtom(dummies[0])
                    prod = rw.GetMol()
                    Chem.SanitizeMol(prod)
                    res_smi = Chem.MolToSmiles(prod, canonical=True)
                    if res_smi:
                        return res_smi, f"fallback_attach_{fg_name}"
                except Exception:
                    continue

        return smiles, "unmodified_identity"

    # -------------------------------------------------------------------------
    # RDKit Chemical Recombination Operators
    # -------------------------------------------------------------------------

    @staticmethod
    def _recombine_once(smiles_a: str, smiles_b: str,
                        rng: np.random.Generator) -> tuple[str, str]:
        """One unconstrained crossover attempt, verified only by Chem.SanitizeMol.

        Callers go through recombine_smiles, which applies the chemical envelope.
        """
        mol_a = Chem.MolFromSmiles(smiles_a)
        mol_b = Chem.MolFromSmiles(smiles_b)
        if mol_a is None or mol_b is None:
            return smiles_a, "invalid_input"

        # Strategy 1: Acyclic Single-Bond Fragment Crossover
        bonds_a = [
            b.GetIdx()
            for b in mol_a.GetBonds()
            if not b.IsInRing() and b.GetBondType() == Chem.BondType.SINGLE
        ]
        bonds_b = [
            b.GetIdx()
            for b in mol_b.GetBonds()
            if not b.IsInRing() and b.GetBondType() == Chem.BondType.SINGLE
        ]

        if bonds_a and bonds_b:
            shuffled_ba = list(bonds_a)
            shuffled_bb = list(bonds_b)
            rng.shuffle(shuffled_ba)
            rng.shuffle(shuffled_bb)

            for ba in shuffled_ba[:3]:
                for bb in shuffled_bb[:3]:
                    try:
                        f_a = Chem.FragmentOnBonds(mol_a, [ba], dummyLabels=[(1, 1)])
                        frags_a = Chem.GetMolFrags(f_a, asMols=True)
                        f_b = Chem.FragmentOnBonds(mol_b, [bb], dummyLabels=[(1, 1)])
                        frags_b = Chem.GetMolFrags(f_b, asMols=True)

                        # Try fragment pairs in order of how close the child's
                        # size is to the parents' mean. Taking the first valid
                        # pair instead favoured the largest fragment of each
                        # parent, so every crossover grew the molecule: four in
                        # a row took erlotinib from 349 to 1131 Da. A crossover
                        # should recombine chemistry, not accumulate it.
                        target = 0.5 * (mol_a.GetNumHeavyAtoms() + mol_b.GetNumHeavyAtoms())
                        pairs = [
                            (fa, fb) for fa in frags_a for fb in frags_b
                        ]
                        pairs.sort(key=lambda pair: abs(
                            (pair[0].GetNumHeavyAtoms() - 1)
                            + (pair[1].GetNumHeavyAtoms() - 1) - target))
                        for fa, fb in pairs:
                            dummies_a = [
                                at.GetIdx() for at in fa.GetAtoms() if at.GetAtomicNum() == 0
                            ]
                            dummies_b = [
                                at.GetIdx() for at in fb.GetAtoms() if at.GetAtomicNum() == 0
                            ]
                            if len(dummies_a) == 1 and len(dummies_b) == 1:
                                combo = Chem.CombineMols(fa, fb)
                                rw = Chem.RWMol(combo)
                                dummies = [
                                    at.GetIdx() for at in rw.GetAtoms() if at.GetAtomicNum() == 0
                                ]
                                n1 = rw.GetAtomWithIdx(dummies[0]).GetNeighbors()[0].GetIdx()
                                n2 = rw.GetAtomWithIdx(dummies[1]).GetNeighbors()[0].GetIdx()
                                rw.AddBond(n1, n2, Chem.BondType.SINGLE)
                                for d in sorted(dummies, reverse=True):
                                    rw.RemoveAtom(d)
                                prod = rw.GetMol()
                                Chem.SanitizeMol(prod)
                                smi = Chem.MolToSmiles(prod, canonical=True)
                                if smi and smi != smiles_a and smi != smiles_b:
                                    return smi, f"fragment_crossover_{ba}_{bb}"
                    except Exception:
                        continue

        # Strategy 2: Direct Inter-Molecular Fragment Linking
        cands_a = [
            a.GetIdx()
            for a in mol_a.GetAtoms()
            if a.GetTotalNumHs() > 0 and a.GetAtomicNum() in (6, 7)
        ]
        cands_b = [
            a.GetIdx()
            for a in mol_b.GetAtoms()
            if a.GetTotalNumHs() > 0 and a.GetAtomicNum() in (6, 7)
        ]
        if cands_a and cands_b:
            try:
                ia = int(rng.choice(cands_a))
                ib = int(rng.choice(cands_b))
                combo = Chem.CombineMols(mol_a, mol_b)
                rw = Chem.RWMol(combo)
                offset_ib = ib + mol_a.GetNumAtoms()
                rw.AddBond(ia, offset_ib, Chem.BondType.SINGLE)
                prod = rw.GetMol()
                Chem.SanitizeMol(prod)
                smi = Chem.MolToSmiles(prod, canonical=True)
                if smi:
                    return smi, f"fragment_link_{ia}_{ib}"
            except Exception:
                pass

        # Strategy 3: Clean Fallback to Mutation
        mutated, label = DirectedMolecularOptimizer._mutate_once(smiles_a, rng)
        return mutated, f"crossover_fallback_{label}"

    # -------------------------------------------------------------------------
    # NSGA-II Multi-Objective Sorting & Crowding Distance
    # -------------------------------------------------------------------------

    @staticmethod
    def fast_non_dominated_sort(objectives_matrix: np.ndarray) -> list[list[int]]:
        """
        Fast non-dominated sorting (Deb et al., NSGA-II).
        All objective columns are MAXIMIZED.
        
        Args:
            objectives_matrix: Shape (N, M) array of fitness objectives.
            
        Returns:
            List of fronts [F_0, F_1, ...], where each front contains row indices.
        """
        n_points = objectives_matrix.shape[0]
        if n_points == 0:
            return []

        domination_counts = np.zeros(n_points, dtype=int)
        dominated_sets: list[list[int]] = [[] for _ in range(n_points)]
        fronts: list[list[int]] = [[]]

        for p in range(n_points):
            for q in range(n_points):
                if p == q:
                    continue
                # p dominates q if p >= q on all objectives and p > q on at least one
                p_ge_q = np.all(objectives_matrix[p] >= objectives_matrix[q])
                p_gt_q = np.any(objectives_matrix[p] > objectives_matrix[q])
                q_ge_p = np.all(objectives_matrix[q] >= objectives_matrix[p])
                q_gt_p = np.any(objectives_matrix[q] > objectives_matrix[p])

                if p_ge_q and p_gt_q:
                    dominated_sets[p].append(q)
                elif q_ge_p and q_gt_p:
                    domination_counts[p] += 1

            if domination_counts[p] == 0:
                fronts[0].append(p)

        curr_front = 0
        while curr_front < len(fronts) and len(fronts[curr_front]) > 0:
            next_front: list[int] = []
            for p in fronts[curr_front]:
                for q in dominated_sets[p]:
                    domination_counts[q] -= 1
                    if domination_counts[q] == 0:
                        next_front.append(q)
            curr_front += 1
            if next_front:
                fronts.append(next_front)

        return [f for f in fronts if len(f) > 0]

    @staticmethod
    def crowding_distance(objectives_matrix: np.ndarray, front: list[int]) -> dict[int, float]:
        """
        Calculate crowding distance for individuals within a Pareto front.
        Boundary individuals are assigned float('inf').
        """
        if len(front) == 0:
            return {}
        if len(front) <= 2:
            return {idx: float("inf") for idx in front}

        distances = {idx: 0.0 for idx in front}
        n_objs = objectives_matrix.shape[1]

        for m in range(n_objs):
            sorted_front = sorted(front, key=lambda idx: float(objectives_matrix[idx, m]))
            distances[sorted_front[0]] = float("inf")
            distances[sorted_front[-1]] = float("inf")

            obj_range = float(objectives_matrix[sorted_front[-1], m] - objectives_matrix[sorted_front[0], m])
            if obj_range > 1e-12:
                for i in range(1, len(sorted_front) - 1):
                    prev_idx = sorted_front[i - 1]
                    next_idx = sorted_front[i + 1]
                    diff = float(objectives_matrix[next_idx, m] - objectives_matrix[prev_idx, m])
                    distances[sorted_front[i]] += diff / obj_range

        return distances

    def diverse_truncation(
        self,
        objectives_matrix: np.ndarray,
        front: list[int],
        needed: int,
        smiles_by_index: dict[int, str],
        chemical_weight: float = 0.5,
    ) -> list[int]:
        """Truncate a Pareto front on objective crowding and chemical distance.

        NSGA-II breaks ties inside an over-full front by crowding distance, which
        is measured in objective space, so two structures that score alike are
        interchangeable to it however different they are chemically. This blends
        Tanimoto distance into that tie-break.

        Selection is greedy. The most crowded-out individual goes in first, then
        each next pick maximises

            chemical_weight * (distance to the nearest already selected member)
          + (1 - chemical_weight) * (normalised objective crowding)

        Tanimoto distance on Morgan fingerprints, radius 2. chemical_weight 0
        reproduces plain NSGA-II crowding, 1 ignores the objectives entirely.

        MEASURED, AND IT DID NOT WORK. This was built to address the structural
        convergence in the project's own candidate review, where generated
        structures cluster at Tanimoto 0.6 to the seed drug. It does not do that.
        Over three seeds at six generations and population 16, against the plain
        crowding tie-break, scripts/measure_chemical_diversity.py measured

            mean pairwise Tanimoto distance   -0.012
            distinct Murcko scaffolds          -0.3
            mean of the five base objectives   +0.008

        all inside seed noise, and the surviving population's own diversity ran
        from 0.56 to 0.71 across seeds against 0.67 to 0.69 for the control. The
        tie-break does fire, every generation: front 0 holds 20 to 28 members
        against a population of 16. It simply does not move the chemistry, so
        the convergence is set somewhere else, most likely in how offspring are
        generated rather than in which survivors are kept. Downweighting the
        applicability-domain objective, which is maximised similarity to the
        training set, did not widen the chemistry either.

        It is kept, off by default, because the mechanism is sound and the
        measurement is worth having: it may matter at a population size where
        front overflow is larger, and it saves the next person implementing it
        from scratch. It is not a fix for structural convergence and must not be
        described as one. Nor does it establish that any extra structures are
        good; a more diverse population of implausible molecules is still
        implausible.
        """
        if needed <= 0 or not front:
            return []
        if needed >= len(front):
            return list(front)

        weight = min(1.0, max(0.0, float(chemical_weight)))
        crowding = self.crowding_distance(objectives_matrix, front)

        finite = [v for v in crowding.values() if np.isfinite(v)]
        ceiling = max(finite) if finite else 1.0
        if ceiling <= 0:
            ceiling = 1.0

        def normalised_crowding(idx: int) -> float:
            value = crowding.get(idx, 0.0)
            # Boundary individuals carry inf. They are the extremes of the front
            # and NSGA-II keeps them ahead of any interior member, so they take
            # 1.0 and the finite values are compressed below it. Scaling the
            # finite values to the full range instead tied the best interior
            # member with the boundary ones and lost that ordering, which at
            # chemical_weight 0 means not reproducing plain NSGA-II.
            if np.isinf(value):
                return 1.0
            return 0.99 * min(1.0, float(value) / ceiling)

        fingerprints: dict[int, Any] = {}
        for idx in front:
            mol = Chem.MolFromSmiles(smiles_by_index.get(idx, "") or "")
            if mol is not None:
                fingerprints[idx] = self.morgan_gen.GetFingerprint(mol)

        remaining = list(front)
        first = max(remaining, key=normalised_crowding)
        selected = [first]
        remaining.remove(first)

        while len(selected) < needed and remaining:
            def score(idx: int) -> float:
                own = fingerprints.get(idx)
                if own is None:
                    # An unreadable structure gets no diversity credit rather
                    # than a free maximum.
                    distance = 0.0
                else:
                    similarities = [
                        DataStructs.TanimotoSimilarity(own, fingerprints[chosen])
                        for chosen in selected if chosen in fingerprints
                    ]
                    distance = 1.0 - max(similarities) if similarities else 1.0
                return weight * distance + (1.0 - weight) * normalised_crowding(idx)

            pick = max(remaining, key=score)
            selected.append(pick)
            remaining.remove(pick)

        return selected

    def calculate_diversity(self, smiles_list: list[str]) -> float:
        """
        Calculate mean pairwise Tanimoto distance across the population
        using Morgan fingerprints.
        """
        if len(smiles_list) <= 1:
            return 0.0

        fps: list[Any] = []
        for s in smiles_list:
            m = Chem.MolFromSmiles(s)
            if m is not None:
                fps.append(self.morgan_gen.GetFingerprint(m))

        if len(fps) <= 1:
            return 0.0

        sims: list[float] = []
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                sims.append(float(DataStructs.TanimotoSimilarity(fps[i], fps[j])))

        mean_sim = float(np.mean(sims)) if sims else 1.0
        return float(round(1.0 - mean_sim, 4))

    # -------------------------------------------------------------------------
    # Full Evolutionary Generation Cycle
    # -------------------------------------------------------------------------

    def evolve_population(
        self,
        population_smiles: list[str],
        evaluate_fn: Callable[[str], dict[str, float]],
        population_size: int,
        rng: np.random.Generator,
        generation: int = 1,
        crossover_prob: float = 0.7,
        mutation_prob: float = 0.3,
    ) -> dict[str, Any]:
        """
        Execute one complete generation of NSGA-II multi-objective molecular optimization.
        
        Args:
            population_smiles: List of candidate SMILES.
            evaluate_fn: Callback mapping SMILES -> objective dict {obj_name: value}.
            population_size: Target size of survivor population.
            rng: NumPy random generator.
            generation: Current generation index.
            crossover_prob: Probability of recombination vs direct reproduction.
            mutation_prob: Probability of mutating offspring.
            
        Returns:
            Dictionary with Pareto front 0 candidates, full ranked population,
            diversity score, generation, and provenance metadata.
        """
        # 1. Clean & sanitize initial population
        valid_pop: list[str] = []
        for s in population_smiles:
            m = Chem.MolFromSmiles(s)
            if m is not None:
                valid_pop.append(Chem.MolToSmiles(m, canonical=True))

        if not valid_pop:
            valid_pop = ["c1ccccc1"]  # Minimal default seed

        # 2. Evaluate current population
        eval_cache: dict[str, dict[str, float]] = {}
        for s in valid_pop:
            if s not in eval_cache:
                eval_cache[s] = evaluate_fn(s)

        # 3. Generate Offspring
        offspring: list[str] = []
        offspring_provenance: list[str] = []

        while len(offspring) < population_size:
            # Parent selection (random or tournament)
            p1 = str(rng.choice(valid_pop))
            p2 = str(rng.choice(valid_pop))

            if rng.random() < crossover_prob:
                child_smi, trans_label = self.recombine_smiles(p1, p2, rng)
            else:
                child_smi = p1
                trans_label = "clone"

            if rng.random() < mutation_prob or child_smi == p1:
                child_smi, mut_label = self.mutate_smiles(child_smi, rng)
                trans_label = f"{trans_label}+{mut_label}"

            # Verify valid sanitized RDKit molecule
            m_child = Chem.MolFromSmiles(child_smi)
            if m_child is not None:
                can_child = Chem.MolToSmiles(m_child, canonical=True)
                offspring.append(can_child)
                offspring_provenance.append(trans_label)
                if can_child not in eval_cache:
                    eval_cache[can_child] = evaluate_fn(can_child)

        # 4. Combined Population Pool (Parents + Offspring)
        combined_smiles = list(dict.fromkeys(valid_pop + offspring))
        for s in combined_smiles:
            if s not in eval_cache:
                eval_cache[s] = evaluate_fn(s)

        # Build objective matrix
        first_eval = eval_cache[combined_smiles[0]]
        obj_names = sorted(list(first_eval.keys()))

        obj_matrix = np.zeros((len(combined_smiles), len(obj_names)), dtype=float)
        for i, s in enumerate(combined_smiles):
            res = eval_cache[s]
            for j, name in enumerate(obj_names):
                obj_matrix[i, j] = float(res.get(name, 0.0))

        # 5. Non-Dominated Sorting
        fronts = self.fast_non_dominated_sort(obj_matrix)

        # 6. Environmental Selection for Next Generation
        selected_indices: list[int] = []
        individual_ranks: dict[int, int] = {}
        individual_crowding: dict[int, float] = {}

        for rank, front in enumerate(fronts):
            cd_dict = self.crowding_distance(obj_matrix, front)
            for idx in front:
                individual_ranks[idx] = rank
                individual_crowding[idx] = cd_dict.get(idx, 0.0)

            if len(selected_indices) + len(front) <= population_size:
                selected_indices.extend(front)
            else:
                needed = population_size - len(selected_indices)
                sorted_by_crowding = sorted(
                    front, key=lambda idx: individual_crowding[idx], reverse=True
                )
                selected_indices.extend(sorted_by_crowding[:needed])
                break

        # 7. Build Pareto Front 0 and Full Ranked Output
        survivor_smiles = [combined_smiles[idx] for idx in selected_indices]

        ranked_population: list[dict[str, Any]] = []
        for idx in selected_indices:
            smi = combined_smiles[idx]
            ranked_population.append({
                "smiles": smi,
                "rank": individual_ranks[idx],
                "crowding_distance": (
                    "inf"
                    if np.isinf(individual_crowding[idx])
                    else float(round(individual_crowding[idx], 4))
                ),
                "objectives": {
                    name: float(round(obj_matrix[idx, j], 4))
                    for j, name in enumerate(obj_names)
                },
            })

        pareto_front_0: list[dict[str, Any]] = []
        if fronts:
            for idx in fronts[0]:
                smi = combined_smiles[idx]
                pareto_front_0.append({
                    "smiles": smi,
                    "crowding_distance": (
                        "inf"
                        if np.isinf(individual_crowding.get(idx, 0.0))
                        else float(round(individual_crowding.get(idx, 0.0), 4))
                    ),
                    "objectives": {
                        name: float(round(obj_matrix[idx, j], 4))
                        for j, name in enumerate(obj_names)
                    },
                })

        diversity_score = self.calculate_diversity(survivor_smiles)

        return {
            "pareto_front_0": pareto_front_0,
            "ranked_population": ranked_population,
            "diversity_score": diversity_score,
            "generation": generation,
            "evaluations_count": len(eval_cache),
            "survivors_smiles": survivor_smiles,
            "provenance": (
                "EVOCURE Directed Molecular Optimizer "
                "(NSGA-II + RDKit Grounded Chemistry Mode B)"
            ),
            "epistemic_category": EpistemicCategory.CALCULATED_PHYSICOCHEMICAL.value,
        }
