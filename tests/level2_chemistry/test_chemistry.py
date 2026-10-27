"""
Unit and integration tests for EVOCURE Computational Chemistry module (Level 2).

Tests validate:
- Curated compound library completeness and metadata rigor
- RDKit structure parsing, canonicalization, and valence violation rejection
- Physicochemical descriptors (MW, exact mass, formal charge, LogP, LogD7.4, TPSA, QED, ESOL LogS)
- Explicit hydrogen atom counting for atomic conservation
- Functional group SMARTS matching and 2D vector SVG depiction
- SDF serialization and deserialization roundtrip
- Morgan fingerprint Tanimoto similarity metrics
- Bemis-Murcko scaffolding and chemical family classification
- Exact elemental and formal charge stoichiometry checking
- Mode B (Grounded Chemistry) reaction execution (esterification, amide coupling, imine condensation, aldol addition)
- Mode B non-covalent complexation fallback
- Mode A (Artificial Chemical Life) mass-conserving condensation
"""

import math
import pytest
from rdkit import Chem

from simulation.chemistry.engine import ChemistryEngine
from simulation.chemistry.library import COMPOUND_LIBRARY, get_builtin_library
from simulation.core.contracts import CompoundRecord, EpistemicCategory, UniverseMode


@pytest.fixture
def engine() -> ChemistryEngine:
    """Provide a shared ChemistryEngine instance."""
    return ChemistryEngine()


# =============================================================================
# 1. Compound Library Tests
# =============================================================================

class TestCompoundLibrary:
    """Validates the static EVOCURE compound library."""

    def test_library_minimum_size_and_immutability(self) -> None:
        lib1 = get_builtin_library()
        lib2 = get_builtin_library()
        assert len(lib1) >= 28, f"Expected at least 28 compounds, found {len(lib1)}"
        assert len(lib1) == len(COMPOUND_LIBRARY)
        # Verify independence of copied lists
        lib1.pop()
        assert len(lib1) != len(lib2)

    def test_library_categories_coverage(self) -> None:
        lib = get_builtin_library()
        categories = {entry["category"] for entry in lib}
        expected = {"pdac_reference", "control", "nutrient", "prebiotic", "cofactor"}
        assert expected.issubset(categories), f"Missing categories: {expected - categories}"

    def test_all_library_compounds_build_successfully(self, engine: ChemistryEngine) -> None:
        lib = get_builtin_library()
        for entry in lib:
            record = engine.build_compound(
                smiles=entry["smiles"],
                name=entry["name"],
                common_name=entry["common_name"],
                provenance=entry["provenance"],
                reference_bioactivity=entry.get("reference_bioactivity"),
            )
            assert isinstance(record, CompoundRecord)
            assert record.is_chemically_valid is True
            assert record.mol_weight > 0.0
            assert record.formula != ""
            assert len(record.atom_counts) > 0
            assert record.compound_id.startswith("CMP-")

    def test_pdac_reference_bioactivity_metadata(self) -> None:
        lib = get_builtin_library()
        pdac_refs = [entry for entry in lib if entry["category"] == "pdac_reference"]
        assert len(pdac_refs) >= 10

        for entry in pdac_refs:
            bio = entry.get("reference_bioactivity")
            assert bio is not None, f"Missing bioactivity for {entry['name']}"
            assert "target" in bio
            assert "assay_source" in bio
            assert "cell_lines" in bio
            cell_lines = bio["cell_lines"]
            assert ("PANC-1" in cell_lines) or ("MIA_PaCa-2" in cell_lines)


# =============================================================================
# 2. Structure Validation & Descriptor Calculation Tests
# =============================================================================

class TestStructureValidationAndDescriptors:
    """Validates RDKit sanitization, descriptor calculations, and error rejection."""

    def test_valid_compound_descriptors_aspirin(self, engine: ChemistryEngine) -> None:
        asp = engine.build_compound("CC(=O)Oc1ccccc1C(=O)O", common_name="Aspirin")
        assert asp.formula == "C9H8O4"
        assert math.isclose(asp.mol_weight, 180.159, abs_tol=0.1)
        assert math.isclose(asp.exact_mass, 180.0423, abs_tol=0.01)
        assert asp.formal_charge == 0
        assert asp.heavy_atoms == 13
        assert asp.rotatable_bonds == 2
        assert asp.murcko_scaffold == "c1ccccc1"
        assert asp.inchi_key != ""
        assert "carboxylic_acid" in asp.functional_groups
        assert "ester" in asp.functional_groups
        assert "aromatic_ring" in asp.functional_groups

    def test_invalid_smiles_syntax_raises_value_error(self, engine: ChemistryEngine) -> None:
        with pytest.raises(ValueError, match="empty"):
            engine.build_compound("")

        with pytest.raises(ValueError, match="cannot be parsed"):
            engine.build_compound("NotAValidSmilesString")

        with pytest.raises(ValueError, match="cannot be parsed"):
            engine.build_compound("C1CC")  # Unclosed ring

    def test_valence_violation_raises_value_error(self, engine: ChemistryEngine) -> None:
        # Carbon with 7 bonds (pentavalent/heptavalent)
        with pytest.raises(ValueError, match="valence/sanitization"):
            engine.build_compound("C(=O)(=O)(=O)O")

        # Hypervalent neutral nitrogen with 5 single bonds
        with pytest.raises(ValueError, match="valence/sanitization"):
            engine.build_compound("CN(C)(C)(C)C")

    def test_atom_counts_include_explicit_hydrogens(self, engine: ChemistryEngine) -> None:
        water = engine.build_compound("O", common_name="Water")
        assert water.atom_counts == {"H": 2, "O": 1}

        methane = engine.build_compound("C", common_name="Methane")
        assert methane.atom_counts == {"C": 1, "H": 4}

        glycine = engine.build_compound("NCC(=O)O", common_name="Glycine")
        assert glycine.atom_counts == {"C": 2, "H": 5, "N": 1, "O": 2}

    def test_delaney_esol_solubility(self, engine: ChemistryEngine) -> None:
        glucose = engine.build_compound("OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O")
        aspirin = engine.build_compound("CC(=O)Oc1ccccc1C(=O)O")
        # Glucose is significantly more water-soluble than aspirin
        assert glucose.esol_log_s is not None
        assert aspirin.esol_log_s is not None
        assert glucose.esol_log_s > aspirin.esol_log_s

    def test_logd_74_adjustment(self, engine: ChemistryEngine) -> None:
        aspirin = engine.build_compound("CC(=O)Oc1ccccc1C(=O)O")
        # At pH 7.4, aspirin's carboxylic acid is ionized (deprotonated), so LogD7.4 < LogP
        assert aspirin.logd_74 < aspirin.logp
        assert math.isclose(aspirin.logd_74, aspirin.logp - 3.2, abs_tol=0.2)

    def test_svg_2d_depiction_generation(self, engine: ChemistryEngine) -> None:
        gemcitabine = engine.resolve_compound_input("Gemcitabine")
        svg = gemcitabine.svg_2d
        assert isinstance(svg, str)
        assert len(svg) > 500
        assert ("<svg" in svg) or ("<?xml" in svg)
        assert "</svg>" in svg


# =============================================================================
# 3. Compound Input Resolution Tests
# =============================================================================

class TestCompoundResolution:
    """Validates multi-format compound input parsing."""

    def test_resolve_by_common_name(self, engine: ChemistryEngine) -> None:
        rec = engine.resolve_compound_input("Gemcitabine")
        assert rec.compound_id == "CMP-GEMCITABINE"
        assert rec.formula == "C9H11F2N3O4"

        rec2 = engine.resolve_compound_input("aspirin")
        assert rec2.compound_id == "CMP-ASPIRIN"

    def test_resolve_by_smiles(self, engine: ChemistryEngine) -> None:
        rec = engine.resolve_compound_input("CC(=O)O")
        assert rec.canonical_smiles == "CC(=O)O"
        assert rec.formula == "C2H4O2"

    def test_resolve_by_molecular_formula(self, engine: ChemistryEngine) -> None:
        rec = engine.resolve_compound_input("H2O", input_type="formula")
        assert rec.common_name == "Water"

    def test_resolve_by_mol_block(self, engine: ChemistryEngine) -> None:
        mol = Chem.MolFromSmiles("CC(=O)O")
        mol_block = Chem.MolToMolBlock(mol)
        rec_auto = engine.resolve_compound_input(mol_block, input_type="auto")
        assert rec_auto.canonical_smiles == "CC(=O)O"
        rec_explicit = engine.resolve_compound_input(mol_block, input_type="mol")
        assert rec_explicit.canonical_smiles == "CC(=O)O"

    def test_resolve_unresolvable_raises_value_error(self, engine: ChemistryEngine) -> None:
        with pytest.raises(ValueError, match="Could not resolve"):
            engine.resolve_compound_input("NonExistentCompound123456XYZ")


# =============================================================================
# 4. SDF Serialization Roundtrip Tests
# =============================================================================

class TestSDFSerialization:
    """Validates export and import of SDF text streams."""

    def test_export_and_import_sdf_roundtrip(self, engine: ChemistryEngine) -> None:
        c1 = engine.resolve_compound_input("Gemcitabine")
        c2 = engine.resolve_compound_input("Aspirin")
        c3 = engine.resolve_compound_input("Glycine")

        sdf_text = engine.export_sdf([c1, c2, c3])
        assert isinstance(sdf_text, str)
        assert "$$$$" in sdf_text

        imported = engine.import_sdf(sdf_text)
        assert len(imported) == 3

        for orig, imp in zip([c1, c2, c3], imported):
            assert orig.canonical_smiles == imp.canonical_smiles
            assert orig.formula == imp.formula
            assert orig.common_name == imp.common_name
            assert orig.compound_id == imp.compound_id
            assert math.isclose(orig.mol_weight, imp.mol_weight, abs_tol=0.01)

    def test_import_empty_sdf(self, engine: ChemistryEngine) -> None:
        assert engine.import_sdf("") == []
        assert engine.import_sdf("   ") == []


# =============================================================================
# 5. Similarity & Scaffolding Tests
# =============================================================================

class TestSimilarityAndScaffolding:
    """Validates Tanimoto Morgan similarity and Bemis-Murcko classification."""

    def test_tanimoto_self_similarity_is_one(self, engine: ChemistryEngine) -> None:
        smi = "CC(=O)Oc1ccccc1C(=O)O"
        sim = engine.tanimoto_similarity(smi, smi)
        assert math.isclose(sim, 1.0, abs_tol=1e-4)

    def test_tanimoto_distinct_compounds(self, engine: ChemistryEngine) -> None:
        smi_a = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
        smi_b = "NCC(=O)O"                # Glycine
        sim = engine.tanimoto_similarity(smi_a, smi_b)
        assert 0.0 <= sim < 0.5

    def test_tanimoto_invalid_smiles_raises(self, engine: ChemistryEngine) -> None:
        with pytest.raises(ValueError):
            engine.tanimoto_similarity("CC(=O)O", "InvalidSmiles")

    def test_assign_chemical_family(self, engine: ChemistryEngine) -> None:
        fid1, fname1 = engine.assign_chemical_family("CC(=O)Oc1ccccc1C(=O)O")
        assert fid1.startswith("FAM-SCF-")
        assert "Benzene" in fname1

        fid2, fname2 = engine.assign_chemical_family("NCC(=O)O")
        assert fid2 == "FAM-AMINOACID"
        assert "Amino Acid" in fname2


# =============================================================================
# 6. Stoichiometry & Chemical Reaction Tests
# =============================================================================

class TestStoichiometryAndReactions:
    """Validates conservation laws and Mode B / Mode A reaction behavior."""

    def test_check_stoichiometry_balanced_pair(self, engine: ChemistryEngine) -> None:
        acid = engine.build_compound("CC(=O)O")
        alc = engine.build_compound("CO")
        ester = engine.build_compound("COC(C)=O")
        water = engine.build_compound("O")

        result = engine.check_stoichiometry([acid, alc], [ester, water])
        assert result["balanced"] is True
        assert result["charge_delta"] == 0
        assert all(delta == 0 for delta in result["atom_delta"].values())

    def test_check_stoichiometry_unbalanced_fails(self, engine: ChemistryEngine) -> None:
        acid = engine.build_compound("CC(=O)O")
        alc = engine.build_compound("CO")
        ester = engine.build_compound("COC(C)=O")

        # Missing water product
        result = engine.check_stoichiometry([acid, alc], [ester])
        assert result["balanced"] is False
        assert result["atom_delta"]["H"] == -2
        assert result["atom_delta"]["O"] == -1

    def test_mode_b_esterification_reaction(self, engine: ChemistryEngine) -> None:
        acid = engine.build_compound("CC(=O)O", common_name="Acetic Acid")
        alc = engine.build_compound("CO", common_name="Methanol")
        rxn = engine.attempt_reaction(acid, alc, mode=UniverseMode.GROUNDED_CHEMISTRY)

        assert rxn["occurred"] is True
        assert rxn["reaction_type"] == "esterification"
        assert rxn["stoichiometry"]["balanced"] is True
        assert len(rxn["products"]) == 2
        smiles_set = {p.canonical_smiles for p in rxn["products"]}
        assert "COC(C)=O" in smiles_set
        assert "O" in smiles_set

    def test_mode_b_amide_coupling_reaction(self, engine: ChemistryEngine) -> None:
        acid = engine.build_compound("CC(=O)O", common_name="Acetic Acid")
        amine = engine.build_compound("CN", common_name="Methylamine")
        rxn = engine.attempt_reaction(acid, amine, mode=UniverseMode.GROUNDED_CHEMISTRY)

        assert rxn["occurred"] is True
        assert rxn["reaction_type"] == "amide_coupling"
        assert rxn["stoichiometry"]["balanced"] is True
        assert len(rxn["products"]) == 2
        smiles_set = {p.canonical_smiles for p in rxn["products"]}
        assert "CNC(C)=O" in smiles_set
        assert "O" in smiles_set

    def test_mode_b_glycine_peptide_condensation(self, engine: ChemistryEngine) -> None:
        gly = engine.build_compound("NCC(=O)O", common_name="Glycine")
        rxn = engine.attempt_reaction(gly, gly, mode=UniverseMode.GROUNDED_CHEMISTRY)

        assert rxn["occurred"] is True
        assert rxn["reaction_type"] == "amide_coupling"
        assert rxn["stoichiometry"]["balanced"] is True
        smiles_set = {p.canonical_smiles for p in rxn["products"]}
        assert "NCC(=O)NCC(=O)O" in smiles_set
        assert "O" in smiles_set

    def test_mode_b_imine_condensation(self, engine: ChemistryEngine) -> None:
        ald = engine.build_compound("CC=O", common_name="Acetaldehyde")
        amine = engine.build_compound("CN", common_name="Methylamine")
        rxn = engine.attempt_reaction(ald, amine, mode=UniverseMode.GROUNDED_CHEMISTRY)

        assert rxn["occurred"] is True
        assert rxn["reaction_type"] == "imine_condensation"
        assert rxn["stoichiometry"]["balanced"] is True

    def test_mode_b_aldol_addition(self, engine: ChemistryEngine) -> None:
        ald = engine.build_compound("CC=O", common_name="Acetaldehyde")
        rxn = engine.attempt_reaction(ald, ald, mode=UniverseMode.GROUNDED_CHEMISTRY)

        assert rxn["occurred"] is True
        assert rxn["reaction_type"] == "aldol_addition"
        assert rxn["stoichiometry"]["balanced"] is True
        assert len(rxn["products"]) == 1
        assert rxn["products"][0].canonical_smiles == "CC(O)CC=O"

    def test_mode_b_non_covalent_complex_fallback(self, engine: ChemistryEngine) -> None:
        asp = engine.resolve_compound_input("Aspirin")
        caf = engine.resolve_compound_input("Caffeine")
        rxn = engine.attempt_reaction(asp, caf, mode=UniverseMode.GROUNDED_CHEMISTRY)

        assert rxn["occurred"] is True
        assert rxn["reaction_type"] == "non_covalent_complexation"
        assert rxn["stoichiometry"]["balanced"] is True
        assert len(rxn["products"]) == 2

    def test_mode_a_artificial_condensation(self, engine: ChemistryEngine) -> None:
        asp = engine.resolve_compound_input("Aspirin")
        caf = engine.resolve_compound_input("Caffeine")
        rxn = engine.attempt_reaction(asp, caf, mode=UniverseMode.ARTIFICIAL_LIFE)

        assert rxn["occurred"] is True
        assert rxn["reaction_type"] == "artificial_condensation"
        assert rxn["stoichiometry"]["balanced"] is True
        assert len(rxn["products"]) == 1
        art = rxn["products"][0]
        assert art.epistemic_category == EpistemicCategory.DEMONSTRATION_UNVALIDATED
        assert math.isclose(art.mol_weight, asp.mol_weight + caf.mol_weight, abs_tol=0.01)

    def test_rate_law_temperature_and_catalytic_effect(self, engine: ChemistryEngine) -> None:
        acid = engine.build_compound("CC(=O)O")
        alc = engine.build_compound("CO")

        r_low_t = engine.attempt_reaction(acid, alc, temperature_k=280.0, catalyst_present=False)
        r_high_t = engine.attempt_reaction(acid, alc, temperature_k=320.0, catalyst_present=False)
        r_cat = engine.attempt_reaction(acid, alc, temperature_k=310.15, catalyst_present=True)
        r_uncat = engine.attempt_reaction(acid, alc, temperature_k=310.15, catalyst_present=False)

        # Arrhenius: k increases with T
        assert r_high_t["rate_constant"] > r_low_t["rate_constant"]
        # Catalytic enhancement: k increases when catalyst is present
        assert r_cat["rate_constant"] > r_uncat["rate_constant"]
