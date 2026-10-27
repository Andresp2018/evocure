"""
Lineage Genealogy and Phylogeny DAG Tracking for EVOCURE.

Maintains complete historical records of compound origins, mutations,
abundances, and extinction dynamics without artificial fitness inflation.
"""

from __future__ import annotations

from typing import Any
import numpy as np

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    _MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
except ImportError:
    Chem = None
    _MORGAN_GEN = None

from simulation.core.contracts import CompoundRecord, GenealogyNode


class GenealogyTracker:
    """
    Lineage genealogy DAG tracker.
    
    Tracks ancestry, transformations, persistence, and extinction events
    across evolutionary generations for both Mode A (artificial chemical life)
    and Mode B (grounded chemistry).
    """

    def __init__(self, nodes: dict[str, GenealogyNode] | None = None) -> None:
        self.nodes: dict[str, GenealogyNode] = nodes if nodes is not None else {}

    def register_compound(
        self,
        compound: CompoundRecord,
        birth_step: int,
        similarity_to_parent: float = 1.0,
        mutation_description: str = "",
        color_hex: str = "#38bdf8",
        properties_summary: dict[str, float] | None = None,
    ) -> GenealogyNode:
        """
        Register a compound in the genealogy DAG.
        If already registered, increments cumulative births.
        """
        cid = compound.compound_id
        if cid in self.nodes:
            existing = self.nodes[cid]
            existing.cumulative_births += 1
            return existing

        # Extract numerical properties summary
        if properties_summary is None:
            summary: dict[str, float] = {
                "mol_weight": float(getattr(compound, "mol_weight", 0.0)),
                "logp": float(getattr(compound, "logp", 0.0)),
                "qed": float(getattr(compound, "qed", 0.0)),
                "tpsa": float(getattr(compound, "tpsa", 0.0)),
                "rotatable_bonds": float(getattr(compound, "rotatable_bonds", 0)),
                "heavy_atoms": float(getattr(compound, "heavy_atoms", 0)),
            }
            if compound.artificial_traits:
                for k, v in compound.artificial_traits.items():
                    try:
                        summary[k] = float(v)
                    except (TypeError, ValueError):
                        pass
        else:
            summary = {
                k: float(v)
                for k, v in properties_summary.items()
                if isinstance(v, (int, float, np.floating, np.integer))
            }

        # Resolve transformation and mutation description
        parent_ids = list(compound.parent_ids)
        if not parent_ids:
            transformation_type = getattr(compound, "transformation_type", "founder") or "founder"
            desc = mutation_description or "Founder compound introduced into the universe."
        else:
            transformation_type = getattr(compound, "transformation_type", "mutation") or "mutation"
            desc = mutation_description or f"Derived from {', '.join(parent_ids)}"

        node = GenealogyNode(
            node_id=f"GN-{cid}",
            compound_id=cid,
            name=compound.name or compound.common_name or cid,
            canonical_smiles=compound.canonical_smiles or "",
            family_id=compound.family_id or "FAM-ORIGIN",
            family_name=compound.family_name or "Founder Family",
            generation=compound.origin_generation,
            parent_ids=parent_ids,
            transformation_type=transformation_type,
            mutation_description=desc,
            similarity_to_parent=float(similarity_to_parent),
            birth_step=birth_step,
            extinction_step=None,
            extinction_cause=None,
            current_abundance=1,
            peak_abundance=1,
            cumulative_births=1,
            persistence_steps=1,
            color_hex=color_hex,
            properties_summary=summary,
        )
        self.nodes[cid] = node
        return node

    def update_abundances(
        self,
        counts_by_compound: dict[str, int],
        current_step: int,
        default_extinction_cause: str = "competitive_exclusion_or_decay",
    ) -> None:
        """
        Update current and peak abundances, track persistence steps,
        and log extinction events when extant population drops to zero.
        """
        for cid, node in self.nodes.items():
            new_count = counts_by_compound.get(cid, 0)
            if node.current_abundance > 0 and new_count == 0:
                # Extinction event
                if node.extinction_step is None:
                    node.extinction_step = current_step
                    node.extinction_cause = default_extinction_cause
            elif new_count > 0:
                # Extant species
                if node.extinction_step is not None:
                    # Species resurged / revived from low count
                    node.extinction_step = None
                    node.extinction_cause = None
                node.persistence_steps += 1
                if new_count > node.peak_abundance:
                    node.peak_abundance = new_count

            node.current_abundance = new_count

    def trace_lineage(self, compound_id: str) -> list[dict[str, Any]]:
        """
        Traverse ancestors all the way back to original founder compounds.
        Returns ordered list of ancestor node dictionaries starting from founders
        down to the target compound.
        """
        if compound_id not in self.nodes:
            return []

        visited: set[str] = set()
        lineage_nodes: list[GenealogyNode] = []

        def dfs(cid: str) -> None:
            if cid in visited or cid not in self.nodes:
                return
            visited.add(cid)
            curr = self.nodes[cid]
            for pid in curr.parent_ids:
                dfs(pid)
            lineage_nodes.append(curr)

        dfs(compound_id)
        return [node.to_dict() for node in lineage_nodes]

    def compare_generations(self, gen_a: int, gen_b: int) -> dict[str, Any]:
        """
        Compare structural and property distributions between two generations.
        """
        nodes_a = [n for n in self.nodes.values() if n.generation == gen_a]
        nodes_b = [n for n in self.nodes.values() if n.generation == gen_b]

        def get_means(nodes: list[GenealogyNode]) -> dict[str, float]:
            if not nodes:
                return {}
            all_keys: set[str] = set()
            for n in nodes:
                all_keys.update(n.properties_summary.keys())
            means: dict[str, float] = {}
            for k in all_keys:
                vals = [n.properties_summary[k] for n in nodes if k in n.properties_summary]
                if vals:
                    means[k] = float(np.mean(vals))
            return means

        means_a = get_means(nodes_a)
        means_b = get_means(nodes_b)

        deltas: dict[str, float] = {}
        for k in set(means_a.keys()).union(means_b.keys()):
            val_a = means_a.get(k, 0.0)
            val_b = means_b.get(k, 0.0)
            deltas[k] = float(val_b - val_a)

        families_a = {n.family_id for n in nodes_a}
        families_b = {n.family_id for n in nodes_b}

        sims_b = [n.similarity_to_parent for n in nodes_b if n.parent_ids]
        mean_sim_b = float(np.mean(sims_b)) if sims_b else 1.0

        return {
            "gen_a": gen_a,
            "gen_b": gen_b,
            "count_a": len(nodes_a),
            "count_b": len(nodes_b),
            "property_means_a": means_a,
            "property_means_b": means_b,
            "property_deltas": deltas,
            "families_a": sorted(list(families_a)),
            "families_b": sorted(list(families_b)),
            "new_families": sorted(list(families_b - families_a)),
            "extinct_families": sorted(list(families_a - families_b)),
            "mean_similarity_to_parent_b": mean_sim_b,
        }

    def compare_nodes(self, cid_a: str, cid_b: str) -> dict[str, Any]:
        """
        Compare two nodes and return structural and property deltas.
        """
        node_a = self.nodes.get(cid_a)
        node_b = self.nodes.get(cid_b)

        if node_a is None or node_b is None:
            return {"error": f"One or both compounds not found: {cid_a}, {cid_b}"}

        # Calculate property deltas
        props_a = node_a.properties_summary
        props_b = node_b.properties_summary
        all_props = set(props_a.keys()).union(props_b.keys())
        deltas = {k: float(props_b.get(k, 0.0) - props_a.get(k, 0.0)) for k in all_props}

        # Chemical Tanimoto similarity if SMILES available
        tanimoto: float | None = None
        if Chem is not None and _MORGAN_GEN is not None:
            if node_a.canonical_smiles and node_b.canonical_smiles:
                try:
                    ma = Chem.MolFromSmiles(node_a.canonical_smiles)
                    mb = Chem.MolFromSmiles(node_b.canonical_smiles)
                    if ma is not None and mb is not None:
                        fp_a = _MORGAN_GEN.GetFingerprint(ma)
                        fp_b = _MORGAN_GEN.GetFingerprint(mb)
                        tanimoto = float(DataStructs.TanimotoSimilarity(fp_a, fp_b))
                except Exception:
                    tanimoto = None

        # Determine lineage relationship
        if cid_a == cid_b:
            relationship = "identical"
        elif cid_a in node_b.parent_ids or cid_b in node_a.parent_ids:
            relationship = "parent_child"
        elif any(p in node_b.parent_ids for p in node_a.parent_ids if node_a.parent_ids):
            relationship = "siblings"
        else:
            ancestors_a = {n["compound_id"] for n in self.trace_lineage(cid_a)}
            ancestors_b = {n["compound_id"] for n in self.trace_lineage(cid_b)}
            if cid_a in ancestors_b:
                relationship = "ancestor_descendant"
            elif cid_b in ancestors_a:
                relationship = "descendant_ancestor"
            elif ancestors_a.intersection(ancestors_b):
                relationship = "shared_ancestor"
            else:
                relationship = "unrelated"

        return {
            "compound_a": cid_a,
            "compound_b": cid_b,
            "generation_delta": node_b.generation - node_a.generation,
            "property_deltas": deltas,
            "tanimoto_similarity": tanimoto,
            "relationship": relationship,
        }

    def query_tree(
        self,
        family_id: str | None = None,
        min_gen: int | None = None,
        max_gen: int | None = None,
        include_extinct: bool = True,
        min_similarity: float = 0.0,
        min_persistence: int = 0,
    ) -> dict[str, Any]:
        """
        Query DAG nodes and edges matching filter criteria, returning graph
        structure and macro-evolutionary summary statistics.
        """
        filtered_nodes: list[GenealogyNode] = []
        for node in self.nodes.values():
            if family_id is not None and node.family_id != family_id:
                continue
            if min_gen is not None and node.generation < min_gen:
                continue
            if max_gen is not None and node.generation > max_gen:
                continue
            if not include_extinct and node.is_extinct:
                continue
            if node.similarity_to_parent < min_similarity:
                continue
            if node.persistence_steps < min_persistence:
                continue
            filtered_nodes.append(node)

        filtered_cids = {n.compound_id for n in filtered_nodes}

        # Build edges: source -> target
        edges: list[dict[str, Any]] = []
        for node in filtered_nodes:
            for pid in node.parent_ids:
                if pid in self.nodes:
                    edges.append({
                        "source": pid,
                        "target": node.compound_id,
                        "transformation": node.transformation_type,
                        "similarity": node.similarity_to_parent,
                    })

        # Summary statistics
        active_count = sum(1 for n in filtered_nodes if n.current_abundance > 0)
        extinct_count = sum(1 for n in filtered_nodes if n.is_extinct)

        family_counts: dict[str, int] = {}
        for n in filtered_nodes:
            family_counts[n.family_id] = family_counts.get(n.family_id, 0) + 1

        dominant_family = (
            max(family_counts.items(), key=lambda kv: kv[1])[0]
            if family_counts
            else "None"
        )

        # Shannon diversity over extant species
        active_nodes = [n for n in filtered_nodes if n.current_abundance > 0]
        total_active_abundance = sum(n.current_abundance for n in active_nodes)
        if total_active_abundance > 0 and len(active_nodes) > 1:
            props = [n.current_abundance / total_active_abundance for n in active_nodes]
            shannon = float(-sum(p * np.log(p) for p in props if p > 0))
        else:
            shannon = 0.0

        return {
            "nodes": [n.to_dict() for n in filtered_nodes],
            "edges": edges,
            "total_nodes": len(filtered_nodes),
            "active_count": active_count,
            "extinct_count": extinct_count,
            "dominant_family": dominant_family,
            "family_counts": family_counts,
            "shannon_diversity_index": float(round(shannon, 4)),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize tracker to dictionary."""
        return {
            "nodes": {cid: node.to_dict() for cid, node in self.nodes.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GenealogyTracker:
        """Deserialize tracker from dictionary."""
        tracker = cls()
        nodes_raw = data.get("nodes", {})
        for cid, nd in nodes_raw.items():
            node = GenealogyNode(
                node_id=nd["node_id"],
                compound_id=nd["compound_id"],
                name=nd["name"],
                canonical_smiles=nd.get("canonical_smiles", ""),
                family_id=nd.get("family_id", "FAM-ORIGIN"),
                family_name=nd.get("family_name", "Founder Family"),
                generation=nd.get("generation", 0),
                parent_ids=nd.get("parent_ids", []),
                transformation_type=nd.get("transformation_type", "founder"),
                mutation_description=nd.get("mutation_description", ""),
                similarity_to_parent=nd.get("similarity_to_parent", 1.0),
                birth_step=nd.get("birth_step", 0),
                extinction_step=nd.get("extinction_step"),
                extinction_cause=nd.get("extinction_cause"),
                current_abundance=nd.get("current_abundance", 0),
                peak_abundance=nd.get("peak_abundance", 0),
                cumulative_births=nd.get("cumulative_births", 1),
                persistence_steps=nd.get("persistence_steps", 0),
                color_hex=nd.get("color_hex", "#38bdf8"),
                properties_summary=nd.get("properties_summary", {}),
            )
            tracker.nodes[cid] = node
        return tracker
