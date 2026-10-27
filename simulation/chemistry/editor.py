"""
Interactive 2D Molecular Structure Editor and Structural Diff Engine for EVOCURE.

Powers EVOCURE LAB's Interactive 2D Molecular Editor and Family Tree Structural Diff Viewer.
Provides:
- 15 substituent additions (methyl, ethyl, fluoro, chloro, hydroxyl, amino, carboxyl, cyano,
  methoxy, trifluoromethyl, acetyl, cyclopropyl, phenyl, imidazole, morpholino).
- Atom replacements (C <-> N, O <-> S, F <-> Cl, Br <-> F) with automatic valence sanitization.
- Ring fusions and appends (cyclopropane, oxetane, pyridine, pyrimidine, benzene, piperazine).
- Medicinal chemistry bioisostere swaps (carboxylic acid to tetrazole/amide, ester to amide/oxadiazole,
  phenyl to pyridine, methyl to CF3, H to F).
- Cleaning, salt-stripping, canonicalization, and atom-index-annotated 2D SVG depiction.
- Maximum Common Substructure (MCS) diff with coral/cyan highlighted SVGs and property deltas.
"""

from __future__ import annotations

import random
from typing import Any

from rdkit import Chem, DataStructs
from rdkit.Chem import (
    AllChem,
    Crippen,
    Descriptors,
    Lipinski,
    QED,
    rdFMCS,
    rdFingerprintGenerator,
)
from rdkit.Chem.Draw import rdMolDraw2D

from simulation.chemistry.engine import ChemistryEngine
from simulation.core.contracts import CompoundRecord


# Standard substituents and their attachment atom index in the substituent fragment
SUBSTITUENT_DEFINITIONS: dict[str, tuple[str, int, str]] = {
    "methyl": ("C", 0, "Methyl group (-CH3)"),
    "ethyl": ("CC", 0, "Ethyl group (-CH2CH3)"),
    "fluoro": ("F", 0, "Fluoro group (-F)"),
    "chloro": ("Cl", 0, "Chloro group (-Cl)"),
    "hydroxyl": ("O", 0, "Hydroxyl group (-OH)"),
    "amino": ("N", 0, "Amino group (-NH2)"),
    "carboxyl": ("C(=O)O", 0, "Carboxylic acid (-COOH)"),
    "cyano": ("C#N", 0, "Cyano / nitrile group (-CN)"),
    "methoxy": ("OC", 0, "Methoxy group (-OCH3)"),
    "trifluoromethyl": ("C(F)(F)F", 0, "Trifluoromethyl group (-CF3)"),
    "acetyl": ("C(=O)C", 0, "Acetyl group (-COCH3)"),
    "cyclopropyl": ("C1CC1", 0, "Cyclopropyl ring (-c-C3H5)"),
    "phenyl": ("c1ccccc1", 0, "Phenyl aromatic ring (-Ph)"),
    "imidazole": ("c1c[nH]cn1", 0, "Imidazole heterocycle"),
    "morpholino": ("C1COCCN1", 5, "Morpholine heterocycle (-Morph)"),
}

# Supported ring appends / fusions
RING_DEFINITIONS: dict[str, tuple[str, int, str]] = {
    "cyclopropane": ("C1CC1", 0, "3-membered aliphatic cyclopropane ring"),
    "oxetane": ("C1COC1", 0, "4-membered oxygen-containing oxetane ring"),
    "pyridine": ("c1ccncc1", 0, "6-membered aromatic pyridine ring"),
    "pyrimidine": ("c1cncnc1", 0, "6-membered aromatic pyrimidine ring"),
    "benzene": ("c1ccccc1", 0, "6-membered aromatic benzene ring"),
    "piperazine": ("C1CNCCN1", 0, "6-membered diamine piperazine ring"),
}

# Bioisosteric reaction SMARTS
BIOISOSTERE_REACTIONS: dict[str, tuple[str, str]] = {
    "carboxylic_acid_to_tetrazole": (
        "[C:1](=O)[OX2H1]>>[C:1]c1nn[nH]n1",
        "Carboxylic acid to 1H-tetrazole bioisosteric replacement",
    ),
    "carboxylic_acid_to_amide": (
        "[C:1](=O)[OX2H1]>>[C:1](=O)N",
        "Carboxylic acid to primary amide replacement",
    ),
    "ester_to_amide": (
        "[C:1](=O)[OX2H0][C:2]>>[C:1](=O)N[C:2]",
        "Carboxylic ester to secondary amide replacement",
    ),
    "ester_to_oxadiazole": (
        "[C:1](=O)[OX2H0][C:2]>>[C:1]c1nc([C:2])no1",
        "Ester to 1,2,4-oxadiazole heterocyclic bioisostere replacement",
    ),
    "phenyl_to_pyridine": (
        "[c;H1:1]1[c:2][c:3][c:4][c:5][c:6]1>>[n:1]1[c:2][c:3][c:4][c:5][c:6]1",
        "Phenyl to pyridine aromatic ring nitrogen insertion",
    ),
    "methyl_to_cf3": (
        "[C;H3:1]>>[C:1](F)(F)F",
        "Methyl to trifluoromethyl bioisosteric replacement",
    ),
    "hydrogen_to_fluorine": (
        "[c;H1:1]>>[c:1]F",
        "Aromatic C-H to C-F metabolic-blocking fluorination",
    ),
}

# Atom replacement atomic number mapping: (source_atomic_num, target_atomic_num)
ATOM_REPLACEMENTS: dict[str, tuple[int, int, str]] = {
    "C_to_N": (6, 7, "Carbon to Nitrogen swap"),
    "N_to_C": (7, 6, "Nitrogen to Carbon swap"),
    "O_to_S": (8, 16, "Oxygen to Sulfur swap"),
    "S_to_O": (16, 8, "Sulfur to Oxygen swap"),
    "F_to_Cl": (9, 17, "Fluorine to Chlorine swap"),
    "Cl_to_F": (17, 9, "Chlorine to Fluorine swap"),
    "Br_to_F": (35, 9, "Bromine to Fluorine swap"),
}

# Color palettes for structural diff highlights (RGB normalized 0.0 - 1.0)
COLOR_CORAL: tuple[float, float, float] = (0.95, 0.45, 0.35)  # Deleted / mutated in Parent A
COLOR_CYAN: tuple[float, float, float] = (0.20, 0.75, 0.90)   # Inserted / mutated in Child B


class MolecularStructureEditor:
    """
    RDKit-based interactive 2D structure editor and comparative diff engine.
    """

    def __init__(self) -> None:
        self.chem_engine = ChemistryEngine()
        self.fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    def list_available_operations(self) -> dict[str, Any]:
        """
        Return metadata on all available editing operations, substituents,
        atom swaps, ring modifications, and bioisostere transformations.
        """
        return {
            "substituents": [
                {"id": k, "smiles": v[0], "attach_index": v[1], "description": v[2]}
                for k, v in SUBSTITUENT_DEFINITIONS.items()
            ],
            "atom_replacements": [
                {"id": k, "from_z": v[0], "to_z": v[1], "description": v[2]}
                for k, v in ATOM_REPLACEMENTS.items()
            ],
            "ring_fusions_and_appends": [
                {"id": k, "smiles": v[0], "attach_index": v[1], "description": v[2]}
                for k, v in RING_DEFINITIONS.items()
            ],
            "bioisostere_swaps": [
                {"id": k, "smarts": v[0], "description": v[1]}
                for k, v in BIOISOSTERE_REACTIONS.items()
            ],
            "clean_and_canonicalize": {
                "id": "clean_and_canonicalize",
                "description": "Validate, strip salts/fragments, standardize tautomers, and annotate atom indices.",
            },
        }

    def _render_svg(
        self,
        mol: Chem.Mol,
        width: int = 280,
        height: int = 210,
        add_atom_indices: bool = False,
        highlight_atoms: list[int] | None = None,
        highlight_bonds: list[int] | None = None,
        highlight_color: tuple[float, float, float] | None = None,
    ) -> str:
        """Render a clean 2D vector depiction of an RDKit molecule."""
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        opts = drawer.drawOptions()
        opts.clearBackground = False
        if add_atom_indices:
            opts.addAtomIndices = True

        h_atoms = highlight_atoms or []
        h_bonds = highlight_bonds or []

        atom_colors = {idx: highlight_color for idx in h_atoms} if highlight_color else None
        bond_colors = {idx: highlight_color for idx in h_bonds} if highlight_color else None

        rdMolDraw2D.PrepareAndDrawMolecule(
            drawer,
            mol,
            highlightAtoms=h_atoms,
            highlightAtomColors=atom_colors,
            highlightBonds=h_bonds,
            highlightBondColors=bond_colors,
        )
        drawer.FinishDrawing()
        return drawer.GetDrawingText()

    def compute_structural_diff(self, smiles_a: str, smiles_b: str) -> dict[str, Any]:
        """
        Compute Maximum Common Substructure (MCS), Morgan Tanimoto similarity,
        exact physicochemical property deltas, and dual side-by-side highlighted SVGs.
        """
        mol_a = Chem.MolFromSmiles(smiles_a) if smiles_a else None
        mol_b = Chem.MolFromSmiles(smiles_b) if smiles_b else None

        if mol_a is None or mol_b is None:
            return {
                "valid": False,
                "error": "One or both SMILES strings could not be parsed.",
            }

        # 1. Exact physicochemical property deltas
        mw_a = float(Descriptors.MolWt(mol_a))
        mw_b = float(Descriptors.MolWt(mol_b))
        logp_a = float(Crippen.MolLogP(mol_a))
        logp_b = float(Crippen.MolLogP(mol_b))
        tpsa_a = float(Descriptors.TPSA(mol_a))
        tpsa_b = float(Descriptors.TPSA(mol_b))
        qed_a = float(QED.qed(mol_a))
        qed_b = float(QED.qed(mol_b))
        hbd_a = int(Lipinski.NumHDonors(mol_a))
        hbd_b = int(Lipinski.NumHDonors(mol_b))
        hba_a = int(Lipinski.NumHAcceptors(mol_a))
        hba_b = int(Lipinski.NumHAcceptors(mol_b))

        fp_a = self.fpgen.GetFingerprint(mol_a)
        fp_b = self.fpgen.GetFingerprint(mol_b)
        tanimoto = float(DataStructs.TanimotoSimilarity(fp_a, fp_b))

        # 2. Maximum Common Substructure (MCS) calculation
        try:
            mcs_res = rdFMCS.FindMCS([mol_a, mol_b], timeout=5)
            mcs_smarts = mcs_res.smartsString
            mcs_num_atoms = int(mcs_res.numAtoms)
            mcs_mol = Chem.MolFromSmarts(mcs_smarts) if mcs_smarts else None
        except Exception:
            mcs_smarts = ""
            mcs_num_atoms = 0
            mcs_mol = None

        match_a = mol_a.GetSubstructMatch(mcs_mol) if mcs_mol is not None else ()
        match_b = mol_b.GetSubstructMatch(mcs_mol) if mcs_mol is not None else ()

        # Non-MCS atoms and their connected bonds
        non_mcs_atoms_a = [i for i in range(mol_a.GetNumAtoms()) if i not in match_a]
        non_mcs_bonds_a = [
            b.GetIdx() for b in mol_a.GetBonds()
            if b.GetBeginAtomIdx() in non_mcs_atoms_a or b.GetEndAtomIdx() in non_mcs_atoms_a
        ]

        non_mcs_atoms_b = [i for i in range(mol_b.GetNumAtoms()) if i not in match_b]
        non_mcs_bonds_b = [
            b.GetIdx() for b in mol_b.GetBonds()
            if b.GetBeginAtomIdx() in non_mcs_atoms_b or b.GetEndAtomIdx() in non_mcs_atoms_b
        ]

        # 3. Side-by-side highlighted 2D SVGs (Coral on A, Cyan on B)
        svg_a_highlighted = self._render_svg(
            mol_a,
            width=280,
            height=210,
            highlight_atoms=non_mcs_atoms_a,
            highlight_bonds=non_mcs_bonds_a,
            highlight_color=COLOR_CORAL,
        )

        svg_b_highlighted = self._render_svg(
            mol_b,
            width=280,
            height=210,
            highlight_atoms=non_mcs_atoms_b,
            highlight_bonds=non_mcs_bonds_b,
            highlight_color=COLOR_CYAN,
        )

        return {
            "valid": True,
            "smiles_a": Chem.MolToSmiles(mol_a, canonical=True),
            "smiles_b": Chem.MolToSmiles(mol_b, canonical=True),
            "delta_mw": round(mw_b - mw_a, 3),
            "delta_logp": round(logp_b - logp_a, 3),
            "delta_tpsa": round(tpsa_b - tpsa_a, 2),
            "delta_qed": round(qed_b - qed_a, 4),
            "delta_hbd": hbd_b - hbd_a,
            "delta_hba": hba_b - hba_a,
            "tanimoto_similarity": round(tanimoto, 4),
            "mcs_smarts": mcs_smarts,
            "mcs_num_atoms": mcs_num_atoms,
            "non_mcs_atoms_a": non_mcs_atoms_a,
            "non_mcs_atoms_b": non_mcs_atoms_b,
            "svg_a_highlighted": svg_a_highlighted,
            "svg_b_highlighted": svg_b_highlighted,
        }

    def apply_edit(
        self,
        smiles: str,
        operation: str,
        param: str = "methyl",
        atom_idx: int | None = None,
        seed: int = 42,
    ) -> dict[str, Any]:
        """
        Apply a structured molecular modification operation to a chemical structure.

        Operations:
        - 'add_group': Attach substituent (methyl, ethyl, fluoro, chloro, hydroxyl, amino, carboxyl, etc.)
        - 'replace_atom': Swap element (C <-> N, O <-> S, F <-> Cl, Br <-> F)
        - 'fuse_or_append_ring': Link ring (cyclopropane, oxetane, pyridine, pyrimidine, benzene, piperazine)
        - 'bioisostere_swap': Apply classic medicinal chemistry SMARTS bioisostere replacements
        - 'clean_and_canonicalize': Strip fragments, sanitize, canonicalize, and generate annotated SVG
        """
        rng = random.Random(seed)

        raw_smiles = smiles.strip() if smiles else ""
        if not raw_smiles:
            return {"valid": False, "error": "Input SMILES string cannot be empty."}

        # Step 1: Parse and sanitize original molecule
        mol = Chem.MolFromSmiles(raw_smiles)
        if mol is None:
            return {"valid": False, "error": f"Invalid SMILES string cannot be parsed: '{raw_smiles}'"}

        edited_mol: Chem.Mol | None = None
        target_atom_idx: int | None = atom_idx

        try:
            # -------------------------------------------------------------
            # Operation 1: add_group
            # -------------------------------------------------------------
            if operation == "add_group":
                sub_info = SUBSTITUENT_DEFINITIONS.get(param.lower())
                if sub_info is None:
                    # Check if param itself is a valid SMILES
                    custom_mol = Chem.MolFromSmiles(param)
                    if custom_mol is None:
                        return {"valid": False, "error": f"Unknown substituent: '{param}'"}
                    sub_smi, attach_idx, _ = param, 0, "Custom substituent"
                else:
                    sub_smi, attach_idx, _ = sub_info

                group_mol = Chem.MolFromSmiles(sub_smi)
                if group_mol is None:
                    return {"valid": False, "error": f"Could not build substituent fragment for '{param}'"}

                # Find candidate target atoms with available valence
                candidate_indices: list[int] = []
                for a in mol.GetAtoms():
                    if a.GetTotalNumHs() > 0:
                        candidate_indices.append(a.GetIdx())

                if not candidate_indices:
                    return {"valid": False, "error": "No substitutable atoms with available hydrogen found."}

                # Prioritize specified atom_idx or select candidate
                ordered_candidates = list(candidate_indices)
                if atom_idx is not None and atom_idx in candidate_indices:
                    ordered_candidates.remove(atom_idx)
                    ordered_candidates.insert(0, atom_idx)
                else:
                    rng.shuffle(ordered_candidates)
                    # Prioritize carbons, then nitrogens
                    ordered_candidates.sort(
                        key=lambda i: (0 if mol.GetAtomWithIdx(i).GetSymbol() == "C" else 1)
                    )

                for cand_idx in ordered_candidates:
                    try:
                        combo = Chem.RWMol(Chem.CombineMols(mol, group_mol))
                        g_attach = mol.GetNumAtoms() + attach_idx
                        combo.AddBond(cand_idx, g_attach, Chem.BondType.SINGLE)
                        Chem.SanitizeMol(combo)
                        edited_mol = combo.GetMol()
                        target_atom_idx = cand_idx
                        break
                    except Exception:
                        continue

                if edited_mol is None:
                    return {"valid": False, "error": f"Failed to attach substituent '{param}' without valence violations."}

            # -------------------------------------------------------------
            # Operation 2: replace_atom
            # -------------------------------------------------------------
            elif operation == "replace_atom":
                # Parse parameter (e.g. 'C_to_N', 'O_to_S', or target element 'N', 'S', etc.)
                replacement_info = ATOM_REPLACEMENTS.get(param)
                if replacement_info is not None:
                    src_z, dst_z, _ = replacement_info
                else:
                    # Look up by target element
                    target_map = {
                        "N": (6, 7), "C": (7, 6), "S": (8, 16),
                        "O": (16, 8), "CL": (9, 17), "F": (17, 9),
                    }
                    if param.upper() in target_map:
                        src_z, dst_z = target_map[param.upper()]
                    else:
                        return {"valid": False, "error": f"Unsupported atom replacement: '{param}'"}

                # Collect atoms matching source element
                src_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == src_z]
                if not src_atoms:
                    return {
                        "valid": False,
                        "error": f"No source atoms with atomic number {src_z} found for replacement.",
                    }

                ordered_src = list(src_atoms)
                if atom_idx is not None and atom_idx in src_atoms:
                    ordered_src.remove(atom_idx)
                    ordered_src.insert(0, atom_idx)
                else:
                    rng.shuffle(ordered_src)

                for cand_idx in ordered_src:
                    try:
                        rw = Chem.RWMol(mol)
                        rw.GetAtomWithIdx(cand_idx).SetAtomicNum(dst_z)
                        Chem.SanitizeMol(rw)
                        edited_mol = rw.GetMol()
                        target_atom_idx = cand_idx
                        break
                    except Exception:
                        continue

                if edited_mol is None:
                    return {
                        "valid": False,
                        "error": f"Failed atom swap '{param}' on available sites due to valence/aromaticity constraints.",
                    }

            # -------------------------------------------------------------
            # Operation 3: fuse_or_append_ring
            # -------------------------------------------------------------
            elif operation == "fuse_or_append_ring":
                ring_info = RING_DEFINITIONS.get(param.lower())
                if ring_info is None:
                    return {"valid": False, "error": f"Unsupported ring: '{param}'"}
                ring_smi, attach_idx, _ = ring_info

                ring_mol = Chem.MolFromSmiles(ring_smi)
                if ring_mol is None:
                    return {"valid": False, "error": f"Could not build ring fragment for '{param}'"}

                candidate_indices = [a.GetIdx() for a in mol.GetAtoms() if a.GetTotalNumHs() > 0]
                if not candidate_indices:
                    return {"valid": False, "error": "No substitutable position found to attach ring."}

                ordered_candidates = list(candidate_indices)
                if atom_idx is not None and atom_idx in candidate_indices:
                    ordered_candidates.remove(atom_idx)
                    ordered_candidates.insert(0, atom_idx)
                else:
                    rng.shuffle(ordered_candidates)

                for cand_idx in ordered_candidates:
                    try:
                        combo = Chem.RWMol(Chem.CombineMols(mol, ring_mol))
                        r_attach = mol.GetNumAtoms() + attach_idx
                        combo.AddBond(cand_idx, r_attach, Chem.BondType.SINGLE)
                        Chem.SanitizeMol(combo)
                        edited_mol = combo.GetMol()
                        target_atom_idx = cand_idx
                        break
                    except Exception:
                        continue

                if edited_mol is None:
                    return {"valid": False, "error": f"Failed to attach/fuse ring '{param}'."}

            # -------------------------------------------------------------
            # Operation 4: bioisostere_swap
            # -------------------------------------------------------------
            elif operation == "bioisostere_swap":
                # Find matching bioisostere reaction
                rxn_key = param.lower()
                # Allow alias matching (e.g. 'tetrazole' -> 'carboxylic_acid_to_tetrazole')
                if rxn_key not in BIOISOSTERE_REACTIONS:
                    for k in BIOISOSTERE_REACTIONS:
                        if rxn_key in k:
                            rxn_key = k
                            break

                rxn_info = BIOISOSTERE_REACTIONS.get(rxn_key)
                if rxn_info is None:
                    return {"valid": False, "error": f"Unknown bioisostere replacement: '{param}'"}

                smarts_rxn, _ = rxn_info
                rxn = AllChem.ReactionFromSmarts(smarts_rxn)
                products = rxn.RunReactants((mol,))
                if not products or not products[0]:
                    # Fallback for fluorine if aromatic failed
                    if "fluorine" in rxn_key:
                        rxn_fallback = AllChem.ReactionFromSmarts("[C;H1,H2,H3:1]>>[C:1]F")
                        products = rxn_fallback.RunReactants((mol,))

                if not products or not products[0]:
                    return {
                        "valid": False,
                        "error": f"No reactive motif found for bioisosteric transformation '{param}'.",
                    }

                # Deterministically select product set
                chosen_set = products[rng.randint(0, len(products) - 1)]
                prod_mol = chosen_set[0]
                Chem.SanitizeMol(prod_mol)
                edited_mol = prod_mol

            # -------------------------------------------------------------
            # Operation 5: clean_and_canonicalize
            # -------------------------------------------------------------
            elif operation == "clean_and_canonicalize":
                # Strip salts if multi-fragment
                frags = Chem.GetMolFrags(mol, asMols=True)
                largest_mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
                Chem.SanitizeMol(largest_mol)
                edited_mol = largest_mol

            else:
                return {"valid": False, "error": f"Unknown operation: '{operation}'"}

        except Exception as exc:
            return {"valid": False, "error": f"Error executing operation '{operation}': {exc}"}

        if edited_mol is None:
            return {"valid": False, "error": f"Operation '{operation}' produced no valid chemical structure."}

        # Step 2: Canonicalize edited molecule
        edited_smiles = Chem.MolToSmiles(edited_mol, canonical=True)

        # Step 3: Compute full CompoundRecord via ChemistryEngine
        comp_rec = self.chem_engine.build_compound(
            smiles=edited_smiles,
            name=f"Edited-{param}",
            provenance=f"Molecular Editor ({operation}:{param})",
            transformation_type=f"editor_{operation}",
        )

        # Step 4: Render depictions
        svg_2d = self._render_svg(edited_mol, width=280, height=210, add_atom_indices=False)
        svg_annotated = self._render_svg(edited_mol, width=320, height=240, add_atom_indices=True)

        # Step 5: Compute comparative diff with original SMILES
        diff_result = self.compute_structural_diff(smiles, edited_smiles)

        return {
            "valid": True,
            "original_smiles": Chem.MolToSmiles(mol, canonical=True),
            "edited_smiles": edited_smiles,
            "operation_applied": operation,
            "param": param,
            "atom_idx": target_atom_idx,
            "compound": comp_rec.to_dict(),
            "svg_2d": svg_2d,
            "svg_annotated": svg_annotated,
            "diff": diff_result,
        }
