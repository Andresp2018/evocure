"""
Directed Molecular Optimization (NSGA-II) with RDKit Chemically Validated Operators.

Implements Section 6.1 (Mode B Grounded Chemistry):
- RDKit chemically validated functional group mutation, bioisosteric substitution, and bond modification.
- Chemically sanitized BRICS / single-bond fragment crossover recombination.
- Full NSGA-II multi-objective optimization (Pareto non-dominated sorting and crowding distance).
- Strict chemical validity (100% RDKit sanitization guarantee; invalid valences rejected).
"""

from __future__ import annotations

from typing import Any, Callable
import numpy as np

from rdkit import Chem, DataStructs
from rdkit.Chem import BRICS, rdFingerprintGenerator

from simulation.core.contracts import EpistemicCategory


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

    def __init__(self, reference_dataset_smiles: list[str] | None = None) -> None:
        self.morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        self.reference_dataset_smiles = reference_dataset_smiles or []
        self._reference_fps: list[Any] = []
        if self.reference_dataset_smiles:
            for s in self.reference_dataset_smiles:
                m = Chem.MolFromSmiles(s)
                if m is not None:
                    self._reference_fps.append(self.morgan_gen.GetFingerprint(m))

    # -------------------------------------------------------------------------
    # RDKit Chemical Mutation Operators
    # -------------------------------------------------------------------------

    @staticmethod
    def mutate_smiles(smiles: str, rng: np.random.Generator) -> tuple[str, str]:
        """
        Perform chemically valid functional group attachment/swap, bioisosteric
        atom substitution, or bond modification, strictly verified by Chem.SanitizeMol.
        
        Returns:
            tuple[canonical_smiles, transformation_label]
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
    def recombine_smiles(smiles_a: str, smiles_b: str, rng: np.random.Generator) -> tuple[str, str]:
        """
        Recombine two molecules using BRICS / single-bond fragment crossover
        sanitized by RDKit, cleanly falling back to valid mutation if no crossover
        site exists.
        
        Returns:
            tuple[canonical_smiles, transformation_label]
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

                        for fa in frags_a:
                            for fb in frags_b:
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
        mutated, label = DirectedMolecularOptimizer.mutate_smiles(smiles_a, rng)
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
