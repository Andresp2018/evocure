# Scientific Documentation, Epistemic Classification & Limitations - EVOCURE v1.0.0

EVOCURE enforces strict scientific honesty across every calculation, visualization, export, and AI explanation (Sections 13.3, 21 & 27).

---

## 1. Mandatory Epistemic Classification (Section 27)

Every quantitative or categorical metric returned by EVOCURE carries an explicit `EpistemicCategory`:

1. **`Imported experimental observation`** (`EXPERIMENTAL_OBSERVATION`) - Measured literature/database values (e.g., curated GDSC2 / ChEMBL cell-line $\text{IC}_{50}$, bacterial MIC, or biochemical $\text{EC}_{50}$ entries with cited assay provenance).
2. **`Calculated physicochemical property`** (`CALCULATED_PHYSICOCHEMICAL`) - Deterministic chemical graph invariants computed via RDKit (exact molecular weight, formula, topological polar surface area, H-bond donors/acceptors, rotatable bonds, Crippen $\log P$).
3. **`Validated model prediction`** (`VALIDATED_MODEL_PREDICTION`) - Split-conformal calibrated QSAR predictions inside the verified Tanimoto applicability domain ($s_{\max} \ge 0.25$) evaluated under zero-leakage Bemis-Murcko scaffold splitting on strictly assay-homogeneous datasets.
4. **`Mechanistic simulation result`** (`MECHANISTIC_SIMULATION`) - Dynamical outputs arising directly from governing PDEs/ODEs and conservation rules (2D reaction-diffusion hypoxia/biofilm/zonation fractions, closed-box mass balance, clonal cell counts).
5. **`Exploratory model result`** (`EXPLORATORY_MODEL`) - Heuristic QSPR, out-of-domain estimates, Bliss/Loewe combination surfaces, internal 0-100 toxicological risk scores, and simplified 7-compartment PBPK systemic projections.
6. **`AI-generated hypothesis`** (`AI_HYPOTHESIS`) - Natural-language interpretations or experimental suggestions produced by the local Gemma 4 assistant.
7. **`Demonstrative result without external validation`** (`DEMONSTRATION_UNVALIDATED`) - Toy or Mode A artificial chemical life abstractions.
8. **`Insufficient data`** (`INSUFFICIENT_DATA`) & **`Not modeled`** (`UNMODELED`) - Explicitly displayed whenever molecules fall outside model bounds or when systemic physiology (e.g., hemoglobin, creatinine, ALT in 2D-only slice mode) is not modeled.

---

## 2. Hard Scientific & Ethical Guardrails

1. **Artificial Life $\neq$ Living Chemistry or Clinical Proof**:
   - Mode A (`ARTIFICIAL_LIFE`) models self-replication, catalysis, competition, and parasitism as an abstract computational system. Fitness or persistence in Mode A is **permanently quarantined** (`validate_for_biomedical_mode`) and never transferred into pharmacological efficacy claims.
2. **Strict Separation of Non-Comparable Assays (Section 13.3)**:
   - Reference records in `data/reference/pdac_reference_dataset.json` (90 compounds across `PDAC`, `NSCLC`, `AMR`, `MASLD`, and `CONTROL`) are strictly partitioned by `disease_id` and `assay_comparability_group`. Bacterial MICs (`AMR`) or hepatic nuclear receptor $\text{EC}_{50}$s (`MASLD`) are never pooled into human `PDAC` cellular cytotoxicity QSAR training.
3. **No LLM Authority Over Efficacy**:
   - Neither Gemma 4 nor neural Laya AI ever determines whether a molecule inhibits a disease target or is safe. All biological responses derive strictly from explicit mathematical/QSAR models with documented split-conformal uncertainty.
4. **Prohibition of Clinical Claims & Human Dosing**:
   - EVOCURE never outputs terms such as *"proven cure"*, *"safe for humans"*, or *"guaranteed clinical efficacy"*, and never recommends human doses, schedules, or self-administration.
5. **2D Tissue & Microenvironment Scope (`PDAC`, `NSCLC_KRAS`, `AMR_BACTERIAL`, `METABOLIC_MASLD`)**:
   - Captures spatial $\text{O}_2$/nutrient/lipid gradients, ECM/CAF desmoplasia, EPS biofilm hindrance, enzymatic drug hydrolysis, tripartite free-drug partitioning, Bliss/Loewe 2-drug interactions, and multifactorial subclonal resistance.
   - Does **not** capture full 3D organ vasculature, metastatic dissemination, adaptive systemic T-cell priming in lymph nodes, or human clinical trial pharmacokinetics.
6. **Deterministic Reproducibility Boundaries**:
   - Given an identical configuration and `PCG64` seed on the same CPU architecture and Python/NumPy build, simulations are deterministically reproducible and invariant to graphical FPS. Cross-architecture floating-point or GPU thread scheduling is not claimed to be bit-identical without verification.
