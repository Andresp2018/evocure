"""
EVOCURE Computational Chemistry Engine.

Implements rigorous RDKit-based molecular validation, physicochemical descriptor
calculation, 2D vector depiction, Tanimoto similarity metrics, Bemis-Murcko
scaffolding, stoichiometry validation, and biochemical reaction execution across
Mode B (Grounded Chemistry) and Mode A (Artificial Chemical Life).
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import uuid
from typing import Any

from rdkit import Chem, DataStructs
from rdkit.Chem import (
    AllChem,
    Crippen,
    Descriptors,
    QED,
    rdDepictor,
    rdFingerprintGenerator,
    rdMolDescriptors,
)
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Scaffolds import MurckoScaffold

from simulation.chemistry.library import COMPOUND_LIBRARY
from simulation.core.contracts import CompoundRecord, EpistemicCategory, UniverseMode


class ChemistryEngine:
    """
    Central computational chemistry engine for EVOCURE.

    Provides deterministic, validated molecular property estimation and
    reaction transformation preserving elemental stoichiometry and formal charge.
    """

    def __init__(self) -> None:
        # Pre-compile Morgan fingerprint generator (radius 2, 2048 bits)
        self._fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

        # Pre-compile functional group identification SMARTS
        self._functional_group_patterns: list[tuple[str, Chem.Mol]] = [
            ("hydroxyl", Chem.MolFromSmarts("[OX2H]")),
            ("amine", Chem.MolFromSmarts("[NX3;H2,H1;!$(NC=O)]")),
            ("carboxylic_acid", Chem.MolFromSmarts("[CX3](=O)[OX2H1]")),
            ("ester", Chem.MolFromSmarts("[CX3](=O)[OX2H0][#6]")),
            ("amide", Chem.MolFromSmarts("[CX3](=O)[NX3]")),
            ("carbonyl", Chem.MolFromSmarts("[CX3]=O")),
            ("aromatic_ring", Chem.MolFromSmarts("a")),
            ("halogen", Chem.MolFromSmarts("[F,Cl,Br,I]")),
            ("nitrile", Chem.MolFromSmarts("[CX2]#[NX1]")),
            ("phosphate", Chem.MolFromSmarts("[PX4](=[OX1])([OX2])([OX2])")),
            ("thiol", Chem.MolFromSmarts("[SX2H]")),
        ]

        # Pre-compile ionizable group patterns for physiological LogD 7.4 estimation
        self._ionizable_patterns: list[tuple[str, Chem.Mol, str, float]] = [
            ("carboxylic_acid", Chem.MolFromSmarts("[CX3](=O)[OX2H1]"), "acid", 4.2),
            ("sulfonic_acid", Chem.MolFromSmarts("[SX4](=O)(=O)[OX2H1]"), "acid", 1.3),
            ("phosphate", Chem.MolFromSmarts("[PX4](=[OX1])([OX2H])"), "acid", 2.2),
            ("phenol", Chem.MolFromSmarts("[OX2H][c]"), "acid", 9.9),
            ("aliphatic_amine", Chem.MolFromSmarts("[NX3;H2,H1,H0;!$(NC=O);!$(Nc);!$(N~S);!$(N#*)]"), "base", 9.5),
            ("guanidine", Chem.MolFromSmarts("[NX3][CX3](=[NX2])[NX3]"), "base", 12.5),
            ("aromatic_amine", Chem.MolFromSmarts("[NX3;H2,H1;$(Nc)]"), "base", 4.6),
            ("imidazole", Chem.MolFromSmarts("[n;H0;$(n1cc[nH]c1)]"), "base", 6.9),
            ("pyridine", Chem.MolFromSmarts("[n;H0;$(n1ccccc1)]"), "base", 5.2),
        ]

        # Pre-compile reaction SMARTS templates
        self._grounded_reactions: list[tuple[str, Any]] = [
            (
                "amide_coupling",
                AllChem.ReactionFromSmarts("[C:1](=O)[OX2H].[N;H2,H1:2]>>[C:1](=O)[N:2].[OX2H2]"),
            ),
            (
                "amide_hydrolysis",
                AllChem.ReactionFromSmarts("[C:1](=O)[N;H1,H0:2].[OX2H2]>>[C:1](=O)[OX2H].[N:2]"),
            ),
            (
                "esterification",
                AllChem.ReactionFromSmarts("[C:1](=O)[OX2H].[C;!$(C=O):2][OX2H]>>[C:1](=O)[OX2][C:2].[OX2H2]"),
            ),
            (
                "ester_hydrolysis",
                AllChem.ReactionFromSmarts("[C:1](=O)[OX2][C:2].[OX2H2]>>[C:1](=O)[OX2H].[C:2][OX2H]"),
            ),
            (
                "imine_condensation",
                AllChem.ReactionFromSmarts("[C:1]=[O].[N;H2:2]>>[C:1]=[N:2].[OX2H2]"),
            ),
            (
                "imine_hydrolysis",
                AllChem.ReactionFromSmarts("[C:1]=[N:2].[OX2H2]>>[C:1]=[O].[N:2]"),
            ),
            (
                "aldol_addition",
                AllChem.ReactionFromSmarts("[C:1]=[O:2].[C;H3,H2,H1:3][C:4]=[O:5]>>[O:2][C:1][C:3][C:4]=[O:5]"),
            ),
        ]

    # -------------------------------------------------------------------------
    # Core Compound Construction & Validation
    # -------------------------------------------------------------------------

    def build_compound(
        self,
        smiles: str,
        name: str = "",
        common_name: str = "",
        provenance: str = "User Input",
        origin_generation: int = 0,
        parent_ids: list[str] | None = None,
        transformation_type: str = "seed",
        family_id: str | None = None,
        family_name: str | None = None,
        reference_bioactivity: dict[str, Any] | None = None,
    ) -> CompoundRecord:
        """
        Validate chemical structure with RDKit, compute all required physicochemical
        descriptors, determine Murcko scaffold and chemical family, render 2D SVG,
        and return a validated CompoundRecord.

        Raises:
            ValueError: If the SMILES string is invalid or fails RDKit valence sanitization.
        """
        if not smiles or not smiles.strip():
            raise ValueError("SMILES string cannot be empty.")

        raw_smiles = smiles.strip()

        # Step 1: Parse without immediate sanitization to capture valence errors accurately
        mol = Chem.MolFromSmiles(raw_smiles, sanitize=False)
        if mol is None:
            raise ValueError(f"Invalid SMILES string cannot be parsed: '{raw_smiles}'")

        # Step 2: Explicitly sanitize to reject valence or aromaticity violations
        try:
            Chem.SanitizeMol(mol)
        except Exception as exc:
            raise ValueError(
                f"Chemical structure failed valence/sanitization checks: {exc}"
            ) from exc

        # Step 3: Canonical SMILES and elemental formula
        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
        formula = rdMolDescriptors.CalcMolFormula(mol)

        # Step 4: Molecular weight, exact mass, and formal charge
        mol_weight = round(float(Descriptors.MolWt(mol)), 3)
        exact_mass = round(float(Descriptors.ExactMolWt(mol)), 4)
        formal_charge = int(Chem.GetFormalCharge(mol))

        # Step 5: Partition coefficient (Crippen LogP) and physiological LogD at pH 7.4
        logp = round(float(Crippen.MolLogP(mol)), 3)
        logd_74 = round(float(self._estimate_logd_74(mol, logp, ph=7.4)), 3)

        # Step 6: Polar surface area, hydrogen bond counts, rotatable bonds, heavy atoms
        tpsa = round(float(rdMolDescriptors.CalcTPSA(mol)), 2)
        hbd = int(rdMolDescriptors.CalcNumHBD(mol))
        hba = int(rdMolDescriptors.CalcNumHBA(mol))
        rotatable_bonds = int(rdMolDescriptors.CalcNumRotatableBonds(mol))
        heavy_atoms = int(mol.GetNumHeavyAtoms())

        # Step 7: Drug-likeness (QED) and Delaney ESOL aqueous solubility
        qed_score = round(float(QED.qed(mol)), 4)
        esol_log_s = round(float(self._calculate_esol(mol, logp, mol_weight, rotatable_bonds, heavy_atoms)), 3)

        # Step 8: Bemis-Murcko scaffold
        murcko_scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)

        # Step 9: InChIKey with safe fallback
        try:
            inchi_key = Chem.MolToInchiKey(mol) or ""
        except Exception:
            inchi_key = ""

        # Step 10: Explicit atom counts (including H via Chem.AddHs) for exact stoichiometry
        mol_with_hs = Chem.AddHs(mol)
        atom_counts: dict[str, int] = {}
        for atom in mol_with_hs.GetAtoms():
            sym = atom.GetSymbol()
            atom_counts[sym] = atom_counts.get(sym, 0) + 1

        # Step 11: Functional group detection via SMARTS
        functional_groups: list[str] = []
        for fg_name, fg_smarts in self._functional_group_patterns:
            if fg_smarts is not None and mol.HasSubstructMatch(fg_smarts):
                functional_groups.append(fg_name)

        # Step 12: 2D vector depiction SVG (240 x 180)
        svg_2d = self._render_svg_2d(mol, width=240, height=180)

        # Step 13: Deterministic family assignment if not provided
        if not family_id or not family_name:
            calc_fid, calc_fname = self.assign_chemical_family(canonical_smiles)
            family_id = family_id or calc_fid
            family_name = family_name or calc_fname

        # Step 14: Deterministic compound ID and name assignment
        # Check if canonical SMILES exists in builtin library
        lib_match = self._find_in_builtin(canonical_smiles=canonical_smiles)
        if lib_match:
            compound_id = lib_match["compound_id"]
            if not name:
                name = lib_match["name"]
            if not common_name:
                common_name = lib_match["common_name"]
            if provenance == "User Input":
                provenance = lib_match["provenance"]
            if reference_bioactivity is None and "reference_bioactivity" in lib_match:
                reference_bioactivity = lib_match["reference_bioactivity"]
        else:
            hash_id = hashlib.sha256(canonical_smiles.encode("utf-8")).hexdigest()[:10].upper()
            compound_id = f"CMP-{hash_id}"
            if not name:
                name = common_name or f"Compound-{hash_id[:6]}"
            if not common_name:
                common_name = name

        return CompoundRecord(
            compound_id=compound_id,
            name=name,
            common_name=common_name,
            canonical_smiles=canonical_smiles,
            formula=formula,
            mol_weight=mol_weight,
            exact_mass=exact_mass,
            formal_charge=formal_charge,
            logp=logp,
            logd_74=logd_74,
            tpsa=tpsa,
            hbd=hbd,
            hba=hba,
            rotatable_bonds=rotatable_bonds,
            heavy_atoms=heavy_atoms,
            qed=qed_score,
            esol_log_s=esol_log_s,
            functional_groups=functional_groups,
            murcko_scaffold=murcko_scaffold,
            inchi_key=inchi_key,
            family_id=family_id,
            family_name=family_name,
            origin_generation=origin_generation,
            parent_ids=parent_ids or [],
            transformation_type=transformation_type,
            provenance=provenance,
            epistemic_category=EpistemicCategory.CALCULATED_PHYSICOCHEMICAL,
            is_chemically_valid=True,
            svg_2d=svg_2d,
            atom_counts=atom_counts,
            artificial_traits={},
            reference_bioactivity=reference_bioactivity or {},
        )

    # -------------------------------------------------------------------------
    # Compound Input Resolution
    # -------------------------------------------------------------------------

    def resolve_compound_input(self, query: str, input_type: str = "auto") -> CompoundRecord:
        """
        Resolve a compound query by library name, common name, molecular formula,
        SMILES string, or MOL block text.

        Args:
            query: Input string to resolve.
            input_type: One of "auto", "smiles", "name", "formula", "mol".

        Returns:
            Fully validated CompoundRecord.

        Raises:
            ValueError: If the query cannot be resolved or parsed.
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")

        raw_query = query
        q = query.strip()
        mode = input_type.lower()

        # Specific resolution modes
        if mode == "smiles":
            return self.build_compound(smiles=q, provenance="Direct SMILES Input")

        if mode == "mol":
            mol = Chem.MolFromMolBlock(raw_query) or Chem.MolFromMolBlock(q)
            if mol is None:
                raise ValueError("Failed to parse provided MOL block.")
            smi = Chem.MolToSmiles(mol)
            name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
            return self.build_compound(smiles=smi, name=name, provenance="MOL Block Input")

        if mode == "name":
            for entry in COMPOUND_LIBRARY:
                if (
                    q.lower() == entry["name"].lower()
                    or q.lower() == entry["common_name"].lower()
                    or q.lower() == entry["compound_id"].lower()
                ):
                    return self.build_compound(
                        smiles=entry["smiles"],
                        name=entry["name"],
                        common_name=entry["common_name"],
                        provenance=entry["provenance"],
                        reference_bioactivity=entry.get("reference_bioactivity"),
                    )
            raise ValueError(f"Compound with name '{query}' not found in library.")

        if mode == "formula":
            for entry in COMPOUND_LIBRARY:
                rec = self.build_compound(smiles=entry["smiles"])
                if q.upper() == rec.formula.upper():
                    return self.build_compound(
                        smiles=entry["smiles"],
                        name=entry["name"],
                        common_name=entry["common_name"],
                        provenance=entry["provenance"],
                        reference_bioactivity=entry.get("reference_bioactivity"),
                    )
            raise ValueError(f"No compound with molecular formula '{query}' found in library.")

        # Auto-detection mode
        # 1. Check for MOL block markers
        if ("M  END" in raw_query) or ("V2000" in raw_query) or ("V3000" in raw_query):
            try:
                mol = Chem.MolFromMolBlock(raw_query) or Chem.MolFromMolBlock(q)
                if mol is not None and mol.GetNumAtoms() > 0:
                    smi = Chem.MolToSmiles(mol)
                    name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
                    return self.build_compound(smiles=smi, name=name, provenance="MOL Block Input")
            except Exception:
                pass

        # 2. Check compound name or common name in builtin library
        for entry in COMPOUND_LIBRARY:
            if (
                q.lower() == entry["name"].lower()
                or q.lower() == entry["common_name"].lower()
                or q.lower() == entry["compound_id"].lower()
            ):
                return self.build_compound(
                    smiles=entry["smiles"],
                    name=entry["name"],
                    common_name=entry["common_name"],
                    provenance=entry["provenance"],
                    reference_bioactivity=entry.get("reference_bioactivity"),
                )

        # 3. Try parsing as SMILES
        try:
            mol = Chem.MolFromSmiles(q, sanitize=False)
            if mol is not None and mol.GetNumAtoms() > 0:
                Chem.SanitizeMol(mol)
                return self.build_compound(smiles=q, provenance="SMILES Input")
        except Exception:
            pass

        # 4. Check formula in builtin library
        q_upper = q.upper()
        for entry in COMPOUND_LIBRARY:
            try:
                mol = Chem.MolFromSmiles(entry["smiles"])
                if mol is not None:
                    entry_formula = rdMolDescriptors.CalcMolFormula(mol)
                    if q_upper == entry_formula.upper():
                        return self.build_compound(
                            smiles=entry["smiles"],
                            name=entry["name"],
                            common_name=entry["common_name"],
                            provenance=entry["provenance"],
                            reference_bioactivity=entry.get("reference_bioactivity"),
                        )
            except Exception:
                continue

        raise ValueError(
            f"Could not resolve compound input: '{query}'. Query is neither a recognized "
            f"library compound, valid SMILES, molecular formula, nor MOL block."
        )

    # -------------------------------------------------------------------------
    # SDF Import & Export
    # -------------------------------------------------------------------------

    def import_sdf(self, sdf_text: str) -> list[CompoundRecord]:
        """
        Import molecular structures and metadata from an SDF text stream.

        Args:
            sdf_text: Raw multi-molecule SDF string.

        Returns:
            List of validated CompoundRecord instances.
        """
        if not sdf_text or not sdf_text.strip():
            return []

        records: list[CompoundRecord] = []
        bio = io.BytesIO(sdf_text.encode("utf-8"))
        supplier = Chem.ForwardSDMolSupplier(bio)

        for idx, mol in enumerate(supplier):
            if mol is None:
                continue
            try:
                smiles = Chem.MolToSmiles(mol)
                name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"Imported_{idx+1}"
                common_name = mol.GetProp("common_name") if mol.HasProp("common_name") else name
                provenance = mol.GetProp("provenance") if mol.HasProp("provenance") else "SDF Import"
                fam_id = mol.GetProp("family_id") if mol.HasProp("family_id") else None
                fam_name = mol.GetProp("family_name") if mol.HasProp("family_name") else None
                cid = mol.GetProp("compound_id") if mol.HasProp("compound_id") else None

                ref_bio: dict[str, Any] = {}
                if mol.HasProp("reference_bioactivity"):
                    try:
                        ref_bio = json.loads(mol.GetProp("reference_bioactivity"))
                    except Exception:
                        ref_bio = {}

                record = self.build_compound(
                    smiles=smiles,
                    name=name,
                    common_name=common_name,
                    provenance=provenance,
                    family_id=fam_id,
                    family_name=fam_name,
                    reference_bioactivity=ref_bio,
                )
                if cid:
                    record.compound_id = cid
                records.append(record)
            except Exception:
                continue

        return records

    def export_sdf(self, compounds: list[CompoundRecord]) -> str:
        """
        Export a list of CompoundRecords into an SDF formatted string with 2D coordinates
        and embedded physicochemical properties.

        Args:
            compounds: List of CompoundRecord objects to export.

        Returns:
            SDF multi-molecule string.
        """
        sio = io.StringIO()
        writer = Chem.SDWriter(sio)

        for comp in compounds:
            mol = Chem.MolFromSmiles(comp.canonical_smiles)
            if mol is None:
                continue
            rdDepictor.Compute2DCoords(mol)
            mol.SetProp("_Name", comp.common_name or comp.name or comp.compound_id)
            mol.SetProp("compound_id", comp.compound_id)
            mol.SetProp("common_name", comp.common_name)
            mol.SetProp("formula", comp.formula)
            mol.SetProp("mol_weight", str(comp.mol_weight))
            mol.SetProp("exact_mass", str(comp.exact_mass))
            mol.SetProp("formal_charge", str(comp.formal_charge))
            mol.SetProp("logp", str(comp.logp))
            mol.SetProp("logd_74", str(comp.logd_74))
            mol.SetProp("tpsa", str(comp.tpsa))
            mol.SetProp("qed", str(comp.qed))
            mol.SetProp("murcko_scaffold", comp.murcko_scaffold)
            mol.SetProp("inchi_key", comp.inchi_key)
            mol.SetProp("provenance", comp.provenance)
            mol.SetProp("family_id", comp.family_id)
            mol.SetProp("family_name", comp.family_name)
            if comp.reference_bioactivity:
                mol.SetProp("reference_bioactivity", json.dumps(comp.reference_bioactivity))
            writer.write(mol)

        writer.close()
        return sio.getvalue()

    # -------------------------------------------------------------------------
    # Structural Similarity & Family Assignment
    # -------------------------------------------------------------------------

    def tanimoto_similarity(self, smiles_a: str, smiles_b: str) -> float:
        """
        Compute Morgan fingerprint Tanimoto similarity between two SMILES strings.

        Args:
            smiles_a: SMILES string of first molecule.
            smiles_b: SMILES string of second molecule.

        Returns:
            Tanimoto coefficient in range [0.0, 1.0].

        Raises:
            ValueError: If either SMILES string cannot be parsed.
        """
        mol_a = Chem.MolFromSmiles(smiles_a)
        mol_b = Chem.MolFromSmiles(smiles_b)

        if mol_a is None:
            raise ValueError(f"Invalid SMILES string for molecule A: '{smiles_a}'")
        if mol_b is None:
            raise ValueError(f"Invalid SMILES string for molecule B: '{smiles_b}'")

        fp_a = self._fpgen.GetFingerprint(mol_a)
        fp_b = self._fpgen.GetFingerprint(mol_b)
        return round(float(DataStructs.TanimotoSimilarity(fp_a, fp_b)), 4)

    def assign_chemical_family(self, smiles: str) -> tuple[str, str]:
        """
        Deterministically assign a chemical family ID and human-readable family name
        based on the Bemis-Murcko scaffold or acyclic functional class.

        Args:
            smiles: SMILES string of the compound.

        Returns:
            Tuple of (family_id, family_name).

        Raises:
            ValueError: If SMILES string is invalid.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Cannot assign chemical family to invalid SMILES: '{smiles}'")

        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        if scaffold:
            scaffold_hash = hashlib.sha256(scaffold.encode("utf-8")).hexdigest()[:8].upper()
            family_id = f"FAM-SCF-{scaffold_hash}"

            if scaffold == "c1ccccc1":
                family_name = "Benzene Scaffold Family"
            elif any(het in scaffold for het in ["n", "N"]):
                family_name = f"Nitrogen Heterocycle Family ({scaffold})"
            elif any(het in scaffold for het in ["o", "O"]):
                family_name = f"Oxygen Heterocycle Family ({scaffold})"
            else:
                family_name = f"Carbocyclic Scaffold Family ({scaffold})"
            return family_id, family_name

        # Acyclic molecules: classify by functional groups and molecular size
        has_amine = mol.HasSubstructMatch(Chem.MolFromSmarts("[NX3;H2,H1;!$(NC=O)]"))
        has_acid = mol.HasSubstructMatch(Chem.MolFromSmarts("[CX3](=O)[OX2H1]"))
        has_carbonyl = mol.HasSubstructMatch(Chem.MolFromSmarts("[CX3]=O"))
        hydroxyl_matches = mol.GetSubstructMatches(Chem.MolFromSmarts("[OX2H]"))

        if has_amine and has_acid:
            return "FAM-AMINOACID", "Amino Acid Family"
        if has_acid:
            return "FAM-CARBOXYL", "Carboxylic Acid Family"
        if len(hydroxyl_matches) >= 3:
            return "FAM-CARBOHYDRATE", "Carbohydrate / Polyol Family"
        if has_carbonyl:
            return "FAM-CARBONYL", "Acyclic Carbonyl Derivative Family"
        if has_amine:
            return "FAM-AMINE", "Aliphatic Amine Family"
        if mol.GetNumHeavyAtoms() <= 2:
            return "FAM-PREBIOTIC-FOUNDER", "Small Prebiotic Founder Family"

        return "FAM-ACYCLIC", "Acyclic Aliphatic Family"

    # -------------------------------------------------------------------------
    # Stoichiometry & Chemical Reaction Dynamics
    # -------------------------------------------------------------------------

    def check_stoichiometry(
        self,
        reactants: list[CompoundRecord],
        products: list[CompoundRecord],
    ) -> dict[str, Any]:
        """
        Verify exact atomic conservation of every chemical element (including explicit H)
        and net formal charge between reactants and products.

        Args:
            reactants: List of reactant CompoundRecords.
            products: List of product CompoundRecords.

        Returns:
            Dictionary containing:
            - balanced: bool, True if all element deltas are 0 and charge delta is 0.
            - atom_delta: dict[str, int], delta for each element (products - reactants).
            - charge_delta: int, net change in formal charge (products - reactants).
            - reactants_atoms: dict[str, int], total atom counts in reactants.
            - products_atoms: dict[str, int], total atom counts in products.
        """
        react_atoms: dict[str, int] = {}
        prod_atoms: dict[str, int] = {}

        react_charge = sum(r.formal_charge for r in reactants)
        prod_charge = sum(p.formal_charge for p in products)
        charge_delta = prod_charge - react_charge

        for r in reactants:
            for element, count in r.atom_counts.items():
                react_atoms[element] = react_atoms.get(element, 0) + count

        for p in products:
            for element, count in p.atom_counts.items():
                prod_atoms[element] = prod_atoms.get(element, 0) + count

        all_elements = set(react_atoms.keys()) | set(prod_atoms.keys())
        atom_delta: dict[str, int] = {}
        for el in sorted(all_elements):
            delta = prod_atoms.get(el, 0) - react_atoms.get(el, 0)
            atom_delta[el] = delta

        balanced = (charge_delta == 0) and all(d == 0 for d in atom_delta.values())

        return {
            "balanced": balanced,
            "atom_delta": atom_delta,
            "charge_delta": charge_delta,
            "reactants_atoms": react_atoms,
            "products_atoms": prod_atoms,
            "reactants_charge": react_charge,
            "products_charge": prod_charge,
        }

    def attempt_reaction(
        self,
        comp_a: CompoundRecord,
        comp_b: CompoundRecord,
        temperature_k: float = 310.15,
        ph: float = 7.4,
        catalyst_present: bool = False,
        mode: UniverseMode = UniverseMode.GROUNDED_CHEMISTRY,
        rng: Any = None,
    ) -> dict[str, Any]:
        """
        Simulate a chemical reaction encounter between two compounds under defined
        environmental conditions (temperature, pH, catalytic state).

        Governing Rate Law (Section 3.1):
            k(T, pH) = A0 * exp(-Ea / (R * T)) * exp(-(pH - pH_opt)^2 / (2 * sigma_pH^2)) * (1 + gamma_cat * [Cat])

        In Mode B (GROUNDED_CHEMISTRY):
            Applies validated RDKit reaction SMARTS (esterification, amide coupling, imine condensation,
            aldol addition) and verifies exact atomic & formal charge conservation.
            If no covalent template matches, returns reversible non-covalent encounter complex metadata.

        In Mode A (ARTIFICIAL_LIFE):
            Allows documented simplified condensation/autocatalytic replication reactions while
            preserving strict closed-box mass and element bookkeeping metadata.

        Returns:
            Dictionary with reaction status, products, stoichiometry, probability, and kinetic details.
        """
        # Step 1: Kinetic rate calculation via Arrhenius + pH + Catalytic law
        r_gas = 8.314462618  # J / (mol * K)
        ea = 45000.0         # Activation energy in J / mol
        a0 = 1.0e8           # Frequency pre-exponential factor (s^-1)
        ph_opt = 7.4         # Optimal pH for biochemical condensation
        sigma_ph = 1.5       # pH tolerance width
        gamma_cat = 5.0      # Catalytic rate enhancement multiplier
        cat_conc = 1.0 if catalyst_present else 0.0

        temp = max(200.0, min(500.0, temperature_k))
        arrhenius_factor = math.exp(-ea / (r_gas * temp))
        ph_factor = math.exp(-((ph - ph_opt) ** 2) / (2.0 * (sigma_ph ** 2)))
        catalyst_factor = 1.0 + (gamma_cat * cat_conc)

        rate_k = a0 * arrhenius_factor * ph_factor * catalyst_factor
        # Discrete reaction probability over an effective collision step (dt = 0.5 s)
        dt = 0.5
        prob = max(0.001, min(1.0, 1.0 - math.exp(-rate_k * dt)))

        # Step 2: Stochastic check
        if rng is not None:
            roll = rng.random() if hasattr(rng, "random") else rng()
            reaction_triggered = roll <= prob
        else:
            # Deterministic evaluation mode
            reaction_triggered = True

        if not reaction_triggered:
            return {
                "occurred": False,
                "reaction_type": "none",
                "products": [],
                "stoichiometry": {"balanced": True, "atom_delta": {}, "charge_delta": 0},
                "probability": round(prob, 4),
                "rate_constant": round(rate_k, 4),
                "notes": "Kinetic activation barrier or probability threshold not overcome at current T/pH.",
            }

        # Step 3: Mode-dependent transformation
        mol_a = Chem.MolFromSmiles(comp_a.canonical_smiles)
        mol_b = Chem.MolFromSmiles(comp_b.canonical_smiles)

        if mol_a is None or mol_b is None:
            raise ValueError("Failed to regenerate RDKit molecules from reactant canonical SMILES.")

        # Try grounded reaction templates
        for rxn_name, rxn in self._grounded_reactions:
            for r1, r2, p_ids in [
                (mol_a, mol_b, [comp_a.compound_id, comp_b.compound_id]),
                (mol_b, mol_a, [comp_b.compound_id, comp_a.compound_id]),
            ]:
                try:
                    product_sets = rxn.RunReactants((r1, r2))
                    if not product_sets:
                        continue

                    for p_tuple in product_sets:
                        try:
                            prod_records: list[CompoundRecord] = []
                            for idx, p_mol in enumerate(p_tuple):
                                Chem.SanitizeMol(p_mol)
                                p_smi = Chem.MolToSmiles(p_mol)
                                p_name = f"{rxn_name.capitalize()} Product {idx+1}"
                                prod_record = self.build_compound(
                                    smiles=p_smi,
                                    name=p_name,
                                    common_name=p_name,
                                    provenance=f"Grounded Reaction: {rxn_name}",
                                    parent_ids=p_ids,
                                    transformation_type=rxn_name,
                                    origin_generation=max(comp_a.origin_generation, comp_b.origin_generation) + 1,
                                )
                                prod_records.append(prod_record)

                            stoich = self.check_stoichiometry([comp_a, comp_b], prod_records)
                            if stoich["balanced"]:
                                return {
                                    "occurred": True,
                                    "reaction_type": rxn_name,
                                    "products": prod_records,
                                    "stoichiometry": stoich,
                                    "probability": round(prob, 4),
                                    "rate_constant": round(rate_k, 4),
                                    "notes": (
                                        f"Grounded reaction '{rxn_name}' executed. "
                                        f"Exact stoichiometry and formal charge conservation verified."
                                    ),
                                }
                        except Exception:
                            continue
                except Exception:
                    continue

        # If no covalent template matched:
        if mode == UniverseMode.GROUNDED_CHEMISTRY:
            # Reversible non-covalent encounter complex
            all_elements = set(comp_a.atom_counts.keys()) | set(comp_b.atom_counts.keys())
            return {
                "occurred": True,
                "reaction_type": "non_covalent_complexation",
                "products": [comp_a, comp_b],
                "stoichiometry": {
                    "balanced": True,
                    "atom_delta": {el: 0 for el in all_elements},
                    "charge_delta": 0,
                },
                "probability": round(prob, 4),
                "rate_constant": round(rate_k, 4),
                "notes": (
                    "No covalent reaction template matched. Formed transient reversible "
                    "non-covalent encounter complex without bond rearrangement."
                ),
            }

        # UniverseMode.ARTIFICIAL_LIFE: simplified condensation / adduct formation
        comb_atoms: dict[str, int] = {}
        all_els = set(comp_a.atom_counts.keys()) | set(comp_b.atom_counts.keys())
        for el in all_els:
            comb_atoms[el] = comp_a.atom_counts.get(el, 0) + comp_b.atom_counts.get(el, 0)

        total_mass = round(comp_a.mol_weight + comp_b.mol_weight, 3)
        total_exact = round(comp_a.exact_mass + comp_b.exact_mass, 4)
        total_charge = comp_a.formal_charge + comp_b.formal_charge
        comb_smiles = f"{comp_a.canonical_smiles}.{comp_b.canonical_smiles}"
        hash_key = hashlib.sha256(comb_smiles.encode("utf-8")).hexdigest()[:8].upper()
        art_id = f"CMP-ART-{hash_key}"
        art_name = f"Artificial Complex [{comp_a.common_name} + {comp_b.common_name}]"
        art_formula = self._format_hill_formula(comb_atoms)

        art_record = CompoundRecord(
            compound_id=art_id,
            name=art_name,
            common_name=art_name,
            canonical_smiles=f"{comp_a.canonical_smiles}.{comp_b.canonical_smiles}",
            formula=art_formula,
            mol_weight=total_mass,
            exact_mass=total_exact,
            formal_charge=total_charge,
            logp=round((comp_a.logp + comp_b.logp) / 2.0, 3),
            logd_74=round((comp_a.logd_74 + comp_b.logd_74) / 2.0, 3),
            tpsa=round(comp_a.tpsa + comp_b.tpsa, 2),
            hbd=comp_a.hbd + comp_b.hbd,
            hba=comp_a.hba + comp_b.hba,
            rotatable_bonds=comp_a.rotatable_bonds + comp_b.rotatable_bonds,
            heavy_atoms=comp_a.heavy_atoms + comp_b.heavy_atoms,
            qed=round((comp_a.qed + comp_b.qed) / 2.0, 4),
            esol_log_s=round(((comp_a.esol_log_s or 0.0) + (comp_b.esol_log_s or 0.0)) / 2.0, 3),
            functional_groups=list(set(comp_a.functional_groups + comp_b.functional_groups)),
            murcko_scaffold="",
            inchi_key="",
            family_id="FAM-ARTIFICIAL-LIFE",
            family_name="Artificial Life Condensation Family",
            origin_generation=max(comp_a.origin_generation, comp_b.origin_generation) + 1,
            parent_ids=[comp_a.compound_id, comp_b.compound_id],
            transformation_type="artificial_condensation",
            provenance="Mode A Artificial Chemical Life Condensation",
            epistemic_category=EpistemicCategory.DEMONSTRATION_UNVALIDATED,
            is_chemically_valid=True,
            svg_2d="",
            atom_counts=comb_atoms,
            artificial_traits={"catalytic_replication_potential": 0.85, "energy_density": 1.25},
            reference_bioactivity={},
        )

        stoich = self.check_stoichiometry([comp_a, comp_b], [art_record])
        return {
            "occurred": True,
            "reaction_type": "artificial_condensation",
            "products": [art_record],
            "stoichiometry": stoich,
            "probability": round(prob, 4),
            "rate_constant": round(rate_k, 4),
            "notes": "Mode A artificial chemical condensation executed with exact atomic and mass conservation.",
        }

    # -------------------------------------------------------------------------
    # Internal Computational Helpers
    # -------------------------------------------------------------------------

    def _estimate_logd_74(self, mol: Chem.Mol, logp: float, ph: float = 7.4) -> float:
        """
        Estimate physiological LogD at pH 7.4 using Henderson-Hasselbalch shifts
        for acidic and basic ionizable functional groups detected via SMARTS.
        """
        total_delta = 0.0
        for name, pattern, group_type, pka in self._ionizable_patterns:
            if pattern is None:
                continue
            matches = mol.GetSubstructMatches(pattern)
            n = len(matches)
            if n > 0:
                if group_type == "acid":
                    exponent = min(15.0, ph - pka)
                    delta = math.log10(1.0 + (10.0 ** exponent))
                else:
                    exponent = min(15.0, pka - ph)
                    delta = math.log10(1.0 + (10.0 ** exponent))
                total_delta += n * delta

        bounded_delta = min(total_delta, 10.0)
        return logp - bounded_delta

    def _calculate_esol(
        self,
        mol: Chem.Mol,
        clogp: float,
        mw: float,
        rotatable_bonds: int,
        heavy_atoms: int,
    ) -> float:
        """
        Calculate aqueous solubility (LogS) via Delaney's ESOL linear model:
        LogS = 0.16 - 0.63 * cLogP - 0.0062 * MW + 0.066 * RB - 0.74 * AP
        """
        if heavy_atoms <= 0:
            ap = 0.0
        else:
            aromatic_heavy_count = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
            ap = aromatic_heavy_count / float(heavy_atoms)

        log_s = 0.16 - (0.63 * clogp) - (0.0062 * mw) + (0.066 * rotatable_bonds) - (0.74 * ap)
        return log_s

    def _render_svg_2d(self, mol: Chem.Mol, width: int = 240, height: int = 180) -> str:
        """
        Render a 2D depiction SVG string with specified dimensions.
        """
        mol_copy = Chem.Mol(mol)
        try:
            rdDepictor.Compute2DCoords(mol_copy)
            drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
            drawer.DrawMolecule(mol_copy)
            drawer.FinishDrawing()
            return drawer.GetDrawingText()
        except Exception:
            return ""

    def _find_in_builtin(self, canonical_smiles: str) -> dict[str, Any] | None:
        """
        Look up a compound in the builtin library by canonical SMILES match.
        """
        for item in COMPOUND_LIBRARY:
            try:
                m = Chem.MolFromSmiles(item["smiles"])
                if m is not None and Chem.MolToSmiles(m, canonical=True) == canonical_smiles:
                    return item
            except Exception:
                continue
        return None

    def _format_hill_formula(self, atom_counts: dict[str, int]) -> str:
        """
        Format an atom count dictionary into standard Hill system notation
        (C first, H second, then remaining elements in alphabetical order).
        """
        elements: list[str] = []
        if "C" in atom_counts:
            elements.append("C")
            if "H" in atom_counts:
                elements.append("H")
            remaining = sorted([k for k in atom_counts.keys() if k not in ("C", "H")])
            elements.extend(remaining)
        else:
            elements = sorted(list(atom_counts.keys()))

        res = ""
        for el in elements:
            count = atom_counts.get(el, 0)
            if count > 0:
                res += el + (str(count) if count > 1 else "")
        return res
