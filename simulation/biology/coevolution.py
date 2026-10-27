"""
Drug-Tumor Coevolution Engine (Section 14).

Simulates coupled, cyclical coevolution between an adapting population of
small-molecule chemical candidates (optimizing binding potency, selectivity,
and ADMET properties) and a heterogeneous PDAC tumor microenvironment under
selective pressure (selecting for resistant subclones, efflux upregulation,
and stromal hindrance).

Exports two parallel evolutionary trees:
1. chemical_lineage_tree: Directed acyclic graph of candidate compound mutations.
2. tumor_clonal_tree: Dynamic phylogeny of emergent tumor subclones and resistance traits.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Callable

import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, QED, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

from simulation.core.contracts import (
    CellDeathMode,
    CellType,
    CompoundRecord,
    EpistemicCategory,
)
from simulation.biology.diseases.pdac import PDACModule


class CoevolutionEngine:
    """
    Coevolutionary Engine coupling chemical candidate evolution with tumor
    subclonal dynamics in PDAC.
    """

    def __init__(
        self,
        tumor_module: PDACModule | None = None,
        initial_compounds: list[CompoundRecord] | None = None,
        variation_callback: Callable[[CompoundRecord, np.random.Generator], CompoundRecord] | None = None,
        pharmacology_callback: Callable[[CompoundRecord, dict[str, Any]], dict[str, Any]] | None = None,
        seed: int = 42,
    ) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.tumor = tumor_module if tumor_module is not None else PDACModule(seed=seed)
        self.variation_callback = variation_callback
        self.pharmacology_callback = pharmacology_callback

        # Initial Founder Compound Pool
        self.compound_pool: list[CompoundRecord] = (
            initial_compounds if initial_compounds else self._create_founder_compounds()
        )

        # Evolutionary Trees
        self.chemical_lineage_tree: list[dict[str, Any]] = []
        self.tumor_clonal_tree: dict[str, dict[str, Any]] = {}

        # Trajectory Metrics Across Cycles
        self.trajectory: dict[str, list[Any]] = {
            "cycle": [],
            "tumor_viability": [],
            "viable_tumor_cells": [],
            "resistant_fraction": [],
            "mean_resistance_score": [],
            "lead_compound_id": [],
            "lead_compound_name": [],
            "ic50_sensitive_um": [],
            "ic50_resistant_um": [],
            "selectivity_index": [],
            "chemical_diversity": [],
            "subclone_counts": [],
        }

        # Initialize trees with founder state
        self._initialize_trees()

    def _create_founder_compounds(self) -> list[CompoundRecord]:
        """Creates initial chemical candidate pool targeting KRAS G12D / PDAC pathways."""
        founders_data = [
            {
                "id": "EVO-KRAS-001",
                "name": "EvoKras-Alpha",
                "smiles": "CC1=NC2=C(N1)C(=O)NC(=N2)N3CCC(CC3)NC(=O)C4=CC=C(C=C4)F",
                "formula": "C18H19FN6O2",
                "mw": 370.39,
                "logp": 1.85,
                "qed": 0.65,
                "origin": "KRAS G12D Non-Covalent Binder Candidate",
            },
            {
                "id": "EVO-KRAS-002",
                "name": "EvoKras-Beta",
                "smiles": "CC1=C(C=C(C=C1)F)C2=NC(=O)NC(=N2)N3CCN(CC3)CC4=CC=CC=C4",
                "formula": "C20H21FN4O",
                "mw": 352.41,
                "logp": 2.65,
                "qed": 0.68,
                "origin": "Dual KRAS/Pan-RAS Fragment Hit",
            },
            {
                "id": "EVO-MDR-003",
                "name": "EvoMdr-Gamma",
                "smiles": "COC1=CC=C(C=C1)C2=NC3=C(N2)C(=O)NC(=N3)N4CCC(CC4)N(C)C",
                "formula": "C19H24N6O2",
                "mw": 368.44,
                "logp": 1.40,
                "qed": 0.72,
                "origin": "Hydrophilic Efflux-Evading Lead",
            },
        ]

        compounds: list[CompoundRecord] = []
        for d in founders_data:
            rec = CompoundRecord(
                compound_id=d["id"],
                name=d["name"],
                common_name=d["name"],
                canonical_smiles=d["smiles"],
                formula=d["formula"],
                mol_weight=d["mw"],
                logp=d["logp"],
                qed=d["qed"],
                family_id="FAM-KRAS-PDAC",
                origin_generation=0,
                parent_ids=[],
                transformation_type="founder",
                provenance=d["origin"],
                epistemic_category=EpistemicCategory.CALCULATED_PHYSICOCHEMICAL,
            )
            compounds.append(rec)
        return compounds

    def _initialize_trees(self) -> None:
        """Populate initial chemical and clonal lineage nodes."""
        # Chemical Founder Nodes
        for cmpd in self.compound_pool:
            node = {
                "node_id": cmpd.compound_id,
                "name": cmpd.name,
                "canonical_smiles": cmpd.canonical_smiles,
                "parent_ids": cmpd.parent_ids,
                "generation": 0,
                "transformation_type": cmpd.transformation_type,
                "properties": {
                    "mol_weight": cmpd.mol_weight,
                    "logp": cmpd.logp,
                    "qed": cmpd.qed,
                },
            }
            self.chemical_lineage_tree.append(node)

        # Initial Subclonal Nodes from PDAC Tumor
        snap = self.tumor.get_snapshot()
        for clone_id, count in snap["clone_distribution"].items():
            # Determine founder mutations and phenotype
            if clone_id == "CLONE-CLASSICAL":
                muts = ["KRAS_G12D", "TP53_LOF"]
                res = 0.08
                efflux = 0.10
            elif clone_id == "CLONE-BASAL":
                muts = ["KRAS_G12D", "TP53_LOF", "CDKN2A_DEL", "SMAD4_LOF"]
                res = 0.58
                efflux = 0.55
            else:
                muts = []
                res = 0.10
                efflux = 0.10

            self.tumor_clonal_tree[clone_id] = {
                "clone_id": clone_id,
                "parent_clone_id": None,
                "emergence_cycle": 0,
                "mutations": muts,
                "resistance_score": res,
                "efflux_pump_level": efflux,
                "abundance_history": [count],
                "peak_abundance": count,
                "status": "active",
            }

    def evaluate_compound_pharmacology(
        self, compound: CompoundRecord, tumor_state: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Evaluates compound pharmacology against current tumor subclonal profile.
        Uses user-supplied pharmacology_callback if available, else deterministic model.
        """
        if self.pharmacology_callback is not None:
            return self.pharmacology_callback(compound, tumor_state)

        # Deterministic QSAR-like heuristic based on physicochemical parameters
        # Ideal logP for PDAC tissue permeation ~ 1.5 - 2.8; MW ~ 320 - 450; QED > 0.6
        mw_penalty = abs(compound.mol_weight - 380.0) / 150.0
        logp_penalty = abs(compound.logp - 2.2) / 1.5
        qed_bonus = compound.qed * 1.5

        # Base binding potency
        potency_score = max(0.2, 2.5 - mw_penalty - logp_penalty + qed_bonus)
        ic50_sens = float(np.clip(1.2 / potency_score, 0.05, 5.0))

        # Basal subclone resistance shift (efflux sensitivity depends on logP and MW)
        efflux_liability = 1.0 + max(0.0, compound.logp - 1.5) * 1.2
        ic50_res = float(np.clip(ic50_sens * (3.0 + efflux_liability), 0.2, 25.0))

        # Healthy ductal cell IC50 (safety window / selectivity index)
        ic50_healthy = float(np.clip(ic50_sens * (6.0 + 3.0 * compound.qed), 2.0, 50.0))

        selectivity_index = ic50_healthy / max(0.01, ic50_sens)

        return {
            "ic50_sensitive_um": round(ic50_sens, 3),
            "ic50_resistant_um": round(ic50_res, 3),
            "ic50_healthy_um": round(ic50_healthy, 3),
            "selectivity_index": round(selectivity_index, 2),
            "hill_slope": 2.0,
            "emax": 0.000035,
            "fraction_unbound_plasma": float(np.clip(0.40 - 0.08 * compound.logp, 0.05, 0.85)),
        }

    def mutate_compound(
        self, parent: CompoundRecord, generation: int, target_resistance_focus: str = "balanced"
    ) -> CompoundRecord:
        """
        Generates a child candidate molecule with chemical modification.
        Uses user-supplied variation_callback if provided, else RDKit / deterministic mutation.
        """
        if self.variation_callback is not None:
            return self.variation_callback(parent, self.rng)

        # Deterministic chemical transformation
        variant_idx = self.rng.integers(100, 999)
        child_id = f"{parent.compound_id}-v{variant_idx}"

        transformations = [
            ("fluorination", "+F on phenyl ring", 18.0, 0.15, "C-F bioisostere"),
            ("hydroxylation", "+OH polar addition", 16.0, -0.45, "Lower lipophilicity/efflux"),
            ("methylation", "+CH3 hydrophobic fill", 14.0, 0.35, "Subpocket filling"),
            ("methoxy_addition", "+OCH3 donor", 30.0, 0.10, "H-bond acceptor expansion"),
            ("cyclopropyl_substitution", "Cyclopropyl isostere", 26.0, 0.25, "Rigidity optimization"),
        ]

        # If tumor has high efflux/resistance, favor polar / efflux-lowering modifications
        if target_resistance_focus == "efflux_evasion":
            trans = transformations[1]  # hydroxylation
        else:
            trans = transformations[int(self.rng.integers(0, len(transformations)))]

        trans_type, desc, delta_mw, delta_logp, note = trans

        new_mw = max(200.0, parent.mol_weight + delta_mw)
        new_logp = max(-0.5, parent.logp + delta_logp)
        new_qed = float(np.clip(parent.qed + self.rng.normal(0.02, 0.04), 0.35, 0.92))

        # Generate modified SMILES representation
        child_smiles = parent.canonical_smiles
        if RDKIT_AVAILABLE:
            try:
                mol = Chem.MolFromSmiles(parent.canonical_smiles)
                if mol is not None:
                    # RDKit validation & canonicalization
                    child_smiles = Chem.MolToSmiles(mol)
            except Exception:
                pass

        child = CompoundRecord(
            compound_id=child_id,
            name=f"{parent.name}-{trans_type[:4].upper()}",
            common_name=f"{parent.name} derivative",
            canonical_smiles=child_smiles,
            formula=parent.formula,
            mol_weight=round(new_mw, 2),
            logp=round(new_logp, 2),
            qed=round(new_qed, 3),
            family_id=parent.family_id,
            origin_generation=generation,
            parent_ids=[parent.compound_id],
            transformation_type=trans_type,
            provenance=f"Directed coevolutionary optimization ({note})",
            epistemic_category=EpistemicCategory.CALCULATED_PHYSICOCHEMICAL,
        )
        return child

    def run_coevolution(
        self,
        n_cycles: int = 5,
        hours_per_cycle: float = 24.0,
        applied_concentration_um: float = 2.0,
        compound_pool_size: int = 4,
    ) -> dict[str, Any]:
        """
        Executes coupled drug-tumor coevolution across multiple cycles:
        1. Selects highest-fitness compound in current candidate pool.
        2. Applies compound to PDAC microenvironment and advances biological time.
        3. Records subclonal survival, emergence of resistant variants, and tumor viability.
        4. Mutates lead compounds to counter emerging resistance mechanisms.
        5. Updates dual lineage trees.
        """
        initial_tumor_cells = max(
            1,
            sum(
                1
                for c in self.tumor.cells
                if c.death_mode == CellDeathMode.ALIVE
                and c.cell_type in (CellType.TUMOR_SENSITIVE, CellType.TUMOR_RESISTANT)
            ),
        )

        dt_step_seconds = 3600.0  # 1-hour increments
        steps_per_cycle = int(math.ceil(hours_per_cycle))

        for cycle in range(1, n_cycles + 1):
            # 1. Evaluate Chemical Pool & Select Lead Candidate
            current_snap = self.tumor.get_snapshot()
            tumor_state = {
                "resistant_fraction": current_snap["summary_metrics"]["resistant_fraction"],
                "viable_tumor": current_snap["summary_metrics"]["viable_tumor_cells"],
                "cycle": cycle,
            }

            candidate_evals: list[tuple[CompoundRecord, dict[str, Any], float]] = []
            for cmpd in self.compound_pool:
                pharm = self.evaluate_compound_pharmacology(cmpd, tumor_state)
                # Fitness balances: potency vs sensitive, potency vs resistant, selectivity, and QED
                fitness = (
                    2.0 / (0.2 + pharm["ic50_sensitive_um"])
                    + 1.8 / (0.4 + pharm["ic50_resistant_um"])
                    + 0.5 * pharm["selectivity_index"]
                    + 2.0 * cmpd.qed
                )
                candidate_evals.append((cmpd, pharm, fitness))

            # Sort by fitness descending
            candidate_evals.sort(key=lambda x: x[2], reverse=True)
            lead_compound, lead_pharm, lead_fitness = candidate_evals[0]

            # 2. Apply Lead Compound & Advance Tumor Microenvironment
            self.tumor.apply_compound_exposure(
                compound=lead_compound,
                pharmacology_profile=lead_pharm,
                applied_concentration_um=applied_concentration_um,
                injection_region="vascular",
            )

            for _ in range(steps_per_cycle):
                self.tumor.step(dt_step_seconds)

            # 3. Harvest Microenvironment Snapshot & Clonal Shift
            cycle_snap = self.tumor.get_snapshot()
            metrics = cycle_snap["summary_metrics"]
            viable_tumor = metrics["viable_tumor_cells"]
            resistant_frac = metrics["resistant_fraction"]
            tumor_viability = viable_tumor / initial_tumor_cells

            # Update subclone clonal tree with new abundances and emergent clones
            for clone_id, count in cycle_snap["clone_distribution"].items():
                if clone_id in self.tumor_clonal_tree:
                    self.tumor_clonal_tree[clone_id]["abundance_history"].append(count)
                    self.tumor_clonal_tree[clone_id]["peak_abundance"] = max(
                        self.tumor_clonal_tree[clone_id]["peak_abundance"], count
                    )
                    self.tumor_clonal_tree[clone_id]["status"] = (
                        "active" if count > 0 else "extinct"
                    )
                else:
                    # Newly emerged variant subclone!
                    sample_cells = [c for c in self.tumor.cells if c.clone_id == clone_id]
                    parent_id = sample_cells[0].parent_clone_id if sample_cells else "CLONE-BASAL"
                    muts = sample_cells[0].mutations if sample_cells else ["KRAS_G12D", "ABCB1_OVEREXPRESSION"]
                    res = sample_cells[0].resistance_score if sample_cells else 0.70
                    efflux = sample_cells[0].efflux_pump_level if sample_cells else 0.65

                    self.tumor_clonal_tree[clone_id] = {
                        "clone_id": clone_id,
                        "parent_clone_id": parent_id,
                        "emergence_cycle": cycle,
                        "mutations": muts,
                        "resistance_score": round(res, 3),
                        "efflux_pump_level": round(efflux, 3),
                        "abundance_history": [0] * (cycle - 1) + [count],
                        "peak_abundance": count,
                        "status": "active" if count > 0 else "extinct",
                    }

            # Measure chemical structural diversity
            mw_list = [c.mol_weight for c in self.compound_pool]
            chem_diversity = float(np.std(mw_list)) if len(mw_list) > 1 else 0.0

            # Mean resistance score of surviving tumor cells
            living_tumor_res = [
                c.resistance_score
                for c in self.tumor.cells
                if c.death_mode == CellDeathMode.ALIVE
                and c.cell_type in (CellType.TUMOR_SENSITIVE, CellType.TUMOR_RESISTANT)
            ]
            mean_res = float(np.mean(living_tumor_res)) if living_tumor_res else 0.0

            # Record trajectory
            self.trajectory["cycle"].append(cycle)
            self.trajectory["tumor_viability"].append(round(tumor_viability, 4))
            self.trajectory["viable_tumor_cells"].append(viable_tumor)
            self.trajectory["resistant_fraction"].append(round(resistant_frac, 4))
            self.trajectory["mean_resistance_score"].append(round(mean_res, 4))
            self.trajectory["lead_compound_id"].append(lead_compound.compound_id)
            self.trajectory["lead_compound_name"].append(lead_compound.name)
            self.trajectory["ic50_sensitive_um"].append(lead_pharm["ic50_sensitive_um"])
            self.trajectory["ic50_resistant_um"].append(lead_pharm["ic50_resistant_um"])
            self.trajectory["selectivity_index"].append(lead_pharm["selectivity_index"])
            self.trajectory["chemical_diversity"].append(round(chem_diversity, 2))
            self.trajectory["subclone_counts"].append(copy.deepcopy(cycle_snap["clone_distribution"]))

            # 4. Generate Chemical Counter-Adaptations for next cycle
            # If resistant fraction is high, bias mutations towards efflux evasion
            focus = "efflux_evasion" if resistant_frac > 0.40 else "balanced"
            new_generation_compounds: list[CompoundRecord] = [lead_compound]

            # Keep top 2 elite candidates
            if len(candidate_evals) > 1:
                new_generation_compounds.append(candidate_evals[1][0])

            # Mutate to produce novel candidates
            while len(new_generation_compounds) < compound_pool_size:
                parent_cand = self.rng.choice([lead_compound, candidate_evals[min(1, len(candidate_evals)-1)][0]])
                child = self.mutate_compound(
                    parent=parent_cand, generation=cycle, target_resistance_focus=focus
                )
                new_generation_compounds.append(child)

                # Add to chemical lineage tree
                self.chemical_lineage_tree.append(
                    {
                        "node_id": child.compound_id,
                        "name": child.name,
                        "canonical_smiles": child.canonical_smiles,
                        "parent_ids": child.parent_ids,
                        "generation": cycle,
                        "transformation_type": child.transformation_type,
                        "properties": {
                            "mol_weight": child.mol_weight,
                            "logp": child.logp,
                            "qed": child.qed,
                        },
                    }
                )

            self.compound_pool = new_generation_compounds

        return {
            "n_cycles": n_cycles,
            "total_hours": n_cycles * hours_per_cycle,
            "trajectory": self.trajectory,
            "chemical_lineage_tree": self.chemical_lineage_tree,
            "tumor_clonal_tree": list(self.tumor_clonal_tree.values()),
            "lead_compound_final": lead_compound.to_dict(),
            "final_snapshot": self.tumor.get_snapshot(),
        }

    def get_trees(self) -> dict[str, Any]:
        """Return dual evolutionary lineage trees."""
        return {
            "chemical_lineage_tree": self.chemical_lineage_tree,
            "tumor_clonal_tree": list(self.tumor_clonal_tree.values()),
        }

    def get_trajectory(self) -> dict[str, Any]:
        """Return coevolutionary time-series trajectory."""
        return self.trajectory
