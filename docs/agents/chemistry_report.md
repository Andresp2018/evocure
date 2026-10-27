# EVOCURE Computational Chemistry Module - Technical & Scientific Report

## 1. Executive Summary

The EVOCURE Computational Chemistry module (`simulation/chemistry/`) provides the structural, physicochemical, stoichiometric, and reactive foundation for both Grounded Chemistry (`UniverseMode.GROUNDED_CHEMISTRY`, Mode B) and Artificial Chemical Life (`UniverseMode.ARTIFICIAL_LIFE`, Mode A).

In strict adherence to the project contracts in `simulation/core/contracts.py` and the mathematical specifications in `docs/MATHEMATICAL_SPECIFICATION.md`:
- All structures are verified and sanitized with **RDKit 2026** (`Chem.SanitizeMol`).
- Valence violations, aromaticity errors, and malformed SMILES are strictly rejected with typed `ValueError` exceptions.
- Reactions in Mode B enforce **100% exact atomic conservation** (including explicit hydrogens via `Chem.AddHs`) and **net formal charge invariance** ($\sum \Delta N_E = 0$, $\Delta Q = 0$).
- All data records, docstrings, compound names, and scientific summaries are written strictly in **English**.
- No runtime remote API calls are executed; all reference data and computations run entirely locally and deterministically.
- Never makes unsubstantiated clinical efficacy claims; all computed values carry explicit epistemic classifications (`EpistemicCategory.CALCULATED_PHYSICOCHEMICAL` or `EpistemicCategory.EXPERIMENTAL_OBSERVATION`).

---

## 2. Implemented Components & Architecture

### 2.1. File Structure

```
simulation/chemistry/
├── __init__.py           # Exports ChemistryEngine, COMPOUND_LIBRARY, get_builtin_library
├── library.py            # 30 curated reference compounds (PDAC, controls, prebiotics)
└── engine.py             # ChemistryEngine: validation, descriptors, SDF, reactions

tests/level2_chemistry/
└── test_chemistry.py     # 32 comprehensive pytest unit tests (100% passing)

docs/agents/
└── chemistry_report.md   # This technical and scientific report
```

### 2.2. Curated Compound Library (`library.py`)

The library defines 30 structurally validated real chemical compounds across five distinct functional roles:

1. **PDAC Reference Therapeutics & Targeted Inhibitors** (`"pdac_reference"`):
   - **Gemcitabine** (`C1=CN(C(=O)N=C1N)[C@H]2C([C@@H]([C@H](O2)CO)O)(F)F`): standard of care ribonucleotide reductase inhibitor & chain-terminating nucleoside.
   - **5-Fluorouracil** (`O=C1NC(=O)NC=C1F`): thymidylate synthase suicide inhibitor (FOLFIRINOX component).
   - **SN-38** (`CCC1=C2CN3C(=O)C4=C(C3=CC2=C(C=C1)O)C=C5C(=C4)CC(C(=O)O5)(CC)O`): potent active metabolite of irinotecan (topoisomerase I inhibitor).
   - **Irinotecan** (`CCC1=C2CN3C(=O)C4=C(C3=CC2=C(C=C1)OC(=O)N5CCC(CC5)N6CCCCC6)C=C7C(=C4)CC(C(=O)O7)(CC)O`): topoisomerase I prodrug.
   - **Paclitaxel** (`CC1=C2[C@H](C(=O)[C@@]3([C@H](C[C@@H]4[C@]([C@H]3[C@@H]([C@@](C2(C)C)(C[C@@H]1OC(=O)[C@@H]([C@H](C5=CC=CC=C5)NC(=O)C6=CC=CC=C6)O)O)OC(=O)C7=CC=CC=C7)(CO4)OC(=O)C)O)C)OC(=O)C`): microtubule-stabilizing taxane.
   - **Erlotinib** (`COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC`): EGFR tyrosine kinase inhibitor.
   - **Olaparib** (`O=C(c1cc(Cc2n[nH]c(=O)c3ccccc23)ccc1F)N4CCN(C(=O)C5CC5)CC4`): PARP1/2 inhibitor for BRCA-mutated PDAC.
   - **Sotorasib** (`CC(C)c1cc(F)c(N2CCN(C(=O)C=C)CC2)c(c1)c3nc4cc(F)c(O)nc4c(c3)N5CCCC5C`): covalent KRAS G12C inhibitor.
   - **MRTX1133** (`CN1CCN(CC1)c2ccc(cc2)c3nc4cc(F)c(c(nc4c3)N5CCN(CC5)C(=O)C#Cc6cc(F)c(F)cc6)O`): non-covalent KRAS G12D inhibitor.
   - **Trametinib** (`CC1=C(C(=O)N(C(=O)N1C2=C(C=C(C=C2)I)F)C3CC3)NC(=O)c4cc(ccc4F)NC(=O)C`): allosteric MEK1/2 kinase inhibitor.
   - **Hydroxychloroquine** (`CCN(CCCC(C)Nc1ccnc2cc(Cl)ccc12)CCO`): lysosomotropic autophagy flux inhibitor.
   - **Capecitabine** (`CCCCCOC(=O)NC1=NC(=O)N(C=C1F)[C@H]2O[C@H](C)[C@@H](O)[C@H]2O`): oral fluoropyrimidine prodrug.

2. **Benign Negative Controls** (`"control"`):
   - **Aspirin** (`CC(=O)Oc1ccccc1C(=O)O`): non-oncological NSAID control.
   - **Caffeine** (`Cn1cnc2c1c(=O)n(c(=O)n2C)C`): purine methylxanthine stimulant control.
   - **Acetaminophen** (`CC(=O)Nc1ccc(O)cc1`): non-cytotoxic analgesic control.

3. **Prebiotic Molecules, Foundational Nutrients & Cofactors** (`"prebiotic"`, `"nutrient"`, `"cofactor"`):
   - **Water** (`O`), **Formaldehyde** (`C=O`), **Hydrogen Cyanide** (`C#N`), **Glycine** (`NCC(=O)O`), **L-Alanine** (`C[C@@H](N)C(=O)O`), **Pyruvic Acid** (`CC(=O)C(=O)O`), **D-Glucose** (`OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O`), **D-Ribose** (`OC[C@@H]1O[C@H](O)[C@@H](O)[C@@H]1O`), **Adenine** (`Nc1ncnc2[nH]cnc12`), **Uracil** (`O=c1[nH]c(=O)cc[nH]1`), **Citric Acid** (`OC(=O)CC(O)(CC(=O)O)C(=O)O`), **Imidazole** (`c1c[nH]cn1`), **Ascorbic Acid** (`OC[C@@H](O)[C@H]1OC(=O)C(=C1O)O`), **Urea** (`NC(=O)N`), **Acetic Acid** (`CC(=O)O`).

Each compound record contains verified GDSC2 / ChEMBL / literature bioactivity benchmarks (e.g. IC50 values on PANC-1 and MIA PaCa-2 cell lines) tagged with `EpistemicCategory.EXPERIMENTAL_OBSERVATION`.

---

## 3. Physicochemical Descriptors & Calculations

Every compound created via `ChemistryEngine.build_compound()` undergoes complete descriptor generation:

| Descriptor | RDKit Method / Formula | Scientific Meaning |
|---|---|---|
| **Canonical SMILES** | `Chem.MolToSmiles(mol, canonical=True)` | Standardized 1D string identifier |
| **Formula** | `rdMolDescriptors.CalcMolFormula(mol)` | Standard Hill system elemental formula |
| **Molecular Weight** | `Descriptors.MolWt(mol)` | Average atomic mass in g/mol |
| **Exact Mass** | `Descriptors.ExactMolWt(mol)` | Monoisotopic mass in Daltons |
| **Formal Charge** | `Chem.GetFormalCharge(mol)` | Net electrical charge of the species |
| **LogP** | `Crippen.MolLogP(mol)` | Wildman-Crippen octanol-water partition coefficient |
| **LogD (pH 7.4)** | Henderson-Hasselbalch ionizable SMARTS | Physiological distribution coefficient accounting for deprotonation of acids and protonation of bases |
| **TPSA** | `rdMolDescriptors.CalcTPSA(mol)` | Topological polar surface area ($\text{Å}^2$) |
| **HBD / HBA** | `CalcNumHBD` / `CalcNumHBA` | Hydrogen bond donor and acceptor counts |
| **Rotatable Bonds** | `CalcNumRotatableBonds(mol)` | Molecular flexibility and entropy measure |
| **Heavy Atoms** | `mol.GetNumHeavyAtoms()` | Total non-hydrogen atom count |
| **QED** | `QED.qed(mol)` | Quantitative Estimate of Drug-likeness $[0.0, 1.0]$ |
| **ESOL LogS** | Delaney 2004 ESOL model | Estimated aqueous solubility $\log_{10}(\text{mol/L})$ |
| **Murcko Scaffold** | `MurckoScaffold.MurckoScaffoldSmiles` | Core ring and linker topology |
| **InChIKey** | `Chem.MolToInchiKey(mol)` | Standard 27-character fixed-length hash |
| **Atom Counts** | `Chem.AddHs(mol)` explicit enumeration | Element-wise integer mapping including explicit H |
| **Functional Groups** | 11 SMARTS substructure queries | Hydroxyl, amine, carboxylic acid, ester, amide, carbonyl, aromatic ring, halogen, nitrile, phosphate, thiol |
| **2D Depiction** | `rdMolDraw2D.MolDraw2DSVG(240, 180)` | Scalable vector graphic string for UI rendering |

### 3.1. Physiological LogD at pH 7.4 Model

$$\text{LogD}_{7.4} = \text{LogP} - \sum_{\text{acids}} \log_{10}\left(1 + 10^{7.4 - \text{pKa}_i}\right) - \sum_{\text{bases}} \log_{10}\left(1 + 10^{\text{pKa}_j - 7.4}\right)$$

Ionizable functional groups are detected using SMARTS patterns with standard biochemical pKa references (e.g. carboxylic acids $\text{pKa} \approx 4.2$, aliphatic amines $\text{pKa} \approx 9.5$, imidazoles $\text{pKa} \approx 6.9$).

### 3.2. Delaney ESOL Solubility Model

$$\text{LogS} = 0.16 - 0.63 \cdot \text{cLogP} - 0.0062 \cdot \text{MW} + 0.066 \cdot \text{RB} - 0.74 \cdot \text{AP}$$

where $\text{AP} = \frac{N_{\text{aromatic}}}{N_{\text{heavy}}}$ is the aromatic heavy-atom proportion.

---

## 4. Stoichiometry & Reaction Dynamics

### 4.1. Invariant Stoichiometric Verification

An elementary reaction $\sum_a \nu_a A_a \longrightarrow \sum_b \mu_b B_b$ is verified balanced if and only if:

$$\sum_a \nu_a N_E(A_a) = \sum_b \mu_b N_E(B_b) \quad \forall\, E \in \text{Elements}, \qquad \sum_a \nu_a Q(A_a) = \sum_b \mu_b Q(B_b)$$

`ChemistryEngine.check_stoichiometry()` evaluates this invariant by comparing explicit hydrogen-inclusive atom counts and formal charges across all reactants and products.

### 4.2. Governing Kinetic Rate Law

Reaction probability is governed by the Arrhenius-pH-catalytic rate law (Section 3.1):

$$k(T, \text{pH}) = A_0 \exp\left(-\frac{E_a}{R T}\right) \cdot \exp\left(-\frac{(\text{pH} - \text{pH}_{\text{opt}})^2}{2 \sigma_{\text{pH}}^2}\right) \cdot (1 + \gamma_{\text{cat}} [\text{Cat}])$$

with $A_0 = 10^8 \text{ s}^{-1}$, $E_a = 45.0 \text{ kJ/mol}$, $\text{pH}_{\text{opt}} = 7.4$, $\sigma_{\text{pH}} = 1.5$, $\gamma_{\text{cat}} = 5.0$.

### 4.3. Grounded Chemistry (Mode B) Reaction Templates

1. **Amide Coupling**:
   $$\text{[C:1](=O)[OX2H]} + \text{[N;H2,H1:2]} \longrightarrow \text{[C:1](=O)[N:2]} + \text{[OX2H2]}$$
   Carboxylic acid + amine $\longrightarrow$ amide + water (e.g. Glycine peptide condensation).
2. **Amide Hydrolysis**:
   $$\text{[C:1](=O)[N;H1,H0:2]} + \text{[OX2H2]} \longrightarrow \text{[C:1](=O)[OX2H]} + \text{[N:2]}$$
3. **Esterification**:
   $$\text{[C:1](=O)[OX2H]} + \text{[C;!$(C=O):2][OX2H]} \longrightarrow \text{[C:1](=O)[OX2][C:2]} + \text{[OX2H2]}$$
   Carboxylic acid + non-acyl alcohol $\longrightarrow$ ester + water.
4. **Ester Hydrolysis**:
   $$\text{[C:1](=O)[OX2][C:2]} + \text{[OX2H2]} \longrightarrow \text{[C:1](=O)[OX2H]} + \text{[C:2][OX2H]}$$
5. **Imine Condensation**:
   $$\text{[C:1]=[O]} + \text{[N;H2:2]} \longrightarrow \text{[C:1]=[N:2]} + \text{[OX2H2]}$$
   Carbonyl (aldehyde/ketone) + primary amine $\longrightarrow$ imine + water.
6. **Imine Hydrolysis**:
   $$\text{[C:1]=[N:2]} + \text{[OX2H2]} \longrightarrow \text{[C:1]=[O]} + \text{[N:2]}$$
7. **Aldol Addition**:
   $$\text{[C:1]=[O:2]} + \text{[C;H3,H2,H1:3][C:4]=[O:5]} \longrightarrow \text{[O:2][C:1][C:3][C:4]=[O:5]}$$
   Enolizable carbonyl addition yielding $\beta$-hydroxy-carbonyl (exact atom conservation without water elimination).

### 4.4. Non-Covalent Complexation & Mode A Condensation

- **Mode B Fallback**: When reactants collide but do not match any covalent transformation template, `attempt_reaction()` outputs a transient non-covalent encounter complex (`reaction_type: "non_covalent_complexation"`), preserving both parent molecules without generating spurious chemical structures.
- **Mode A (Artificial Chemical Life)**: Enables documented autocatalytic/simplified condensation into heritable artificial molecular complexes (`EpistemicCategory.DEMONSTRATION_UNVALIDATED`), strictly maintaining mass and elemental bookkeeping.

---

## 5. Verification & Test Suite Summary

A comprehensive test suite was executed using the isolated virtual environment (`e:\MA\EVOCURE\.venv\Scripts\python.exe`):

```bash
python -m pytest e:\MA\EVOCURE\tests\level2_chemistry\test_chemistry.py -v
```

### Test Results: 32 / 32 Passed (100%)

| Test Class | Focus Area | Status |
|---|---|---|
| `TestCompoundLibrary` | 30 compounds, immutability, role coverage, PDAC bioactivity metadata | **PASSED** (4/4) |
| `TestStructureValidationAndDescriptors` | Aspirin descriptors, syntax/valence rejection, explicit H, ESOL LogS, LogD7.4, 2D SVG | **PASSED** (7/7) |
| `TestCompoundResolution` | Name, SMILES, formula, MolBlock, and unresolvable error handling | **PASSED** (5/5) |
| `TestSDFSerialization` | Multi-compound SDF export/import roundtrip with embedded metadata | **PASSED** (2/2) |
| `TestSimilarityAndScaffolding` | Tanimoto self-similarity, diversity, Bemis-Murcko scaffolding | **PASSED** (4/4) |
| `TestStoichiometryAndReactions` | Invariant checks, esterification, amide coupling, glycine dimerization, imine, aldol, non-covalent fallback, Mode A, Arrhenius rate law | **PASSED** (10/10) |

---

## 6. Scientific Limitations & Epistemic Boundaries

1. **In Silico Property Predictions**: Physicochemical parameters (LogP, LogD, ESOL solubility, QED) are calculated via empirical quantitative structure-property relationship (QSPR) models and empirical group contributions. They do not substitute for validated wet-lab chromatographic or shake-flask partition measurements.
2. **Simplified Reaction Kinetics**: Reaction rates use idealized Arrhenius and Gaussian pH dependence in an isotropic aqueous continuum. Enzymatic saturation ($K_m, V_{\max}$), stereospecific transition states, and solvent shell reorganization free energies are abstracted.
3. **Clinical Non-Efficacy Boundary**: Benchmark bioactivities (e.g. IC50 on PANC-1 and MIA PaCa-2) represent cell culture in vitro cytotoxicity from literature/GDSC repositories and MUST NEVER be interpreted as clinical response predictions or human dosage recommendations.
