# User Guide - EVOCURE Godot 4.7 Bioluminescent & Web Workbench (v1.0.0)

EVOCURE provides a hardware-accelerated **Godot 4.7 Bioluminescent Desktop Client** (`godot_frontend/`) alongside a zero-CDN local **Web Workbench** (`frontend/`), operating over a **single unified Python `.venv` scientific simulation engine**:
- **Lab Mode (`EVOCURE LAB`)**: Full quantitative research workbench for computational scientists, students, and investigators.
- **Play / Battle Mode (`EVOCURE PLAY`)**: Intuitive, accessible bioluminescent discovery experience with guided missions, fast-forward time jumps, and visual evolutionary creatures.

You can switch between **Lab Mode** and **Play / Battle Mode** at any time without losing state or altering simulation parameters, select any of the **4 active 2D biomedical disease models** (`PDAC`, `NSCLC_KRAS`, `AMR_BACTERIAL`, `METABOLIC_MASLD`) from the top header, and configure **Local Gemma 4**, **Google Gemini**, or **OpenAI** from **⚙ AI Settings**.

---

## 1. Quick Start in EVOCURE PLAY (3 Clicks or Fewer)

1. Launch EVOCURE (`start_evocure.bat` or `.\.venv\Scripts\python.exe run.py`).
2. On the **EVOCURE PLAY** welcome hub, click one of the **4 Launch Cards**:
   - **Create Chemical Life** - Watch molecules move, react, compete for substrate, mutate, and branch into evolutionary families.
   - **Fight Cancer & Disease** - Open a virtual 2D tumor/tissue slice (default: Pancreatic Cancer `PDAC`), introduce a reference compound (such as Gemcitabine), and compare against an untreated control.
   - **Evolve Compounds** - Breed chemically valid molecular variants balanced across multiple objectives.
   - **Explore Discoveries** - Browse saved experiments, chemical family trees, achievements, and glossary terms.
   - *(Or click **"Surprise Me"** to generate a seeded, reproducible discovery scenario immediately!)*
3. Click **Play** (or drag any of the 45 built-in compounds from the dock directly onto the 2D world).

### Game-Style Tools, Dual View & Time Controls
- **Dual Canvas View Toggle**: Switch effortlessly between the **2D Chemical Universe** and the **2D Disease Battle** right inside `PLAY` mode.
- **Toolbar**: *Add* (drag-and-drop onto canvas), *Select* (inspect 2D structure & plain-English compound explainer modal), *Observe* (track a lineage), *Accelerate* (`10x`, `100x`, or jump `10` / `100` / `1,000` / `10,000` generations or virtual years), *Compare Worlds*, and *Save / Experiments Manager*.
- **Period Summaries**: After a multi-generation jump, EVOCURE reports real simulation counts (new lineages born, extinct families, dominant scaffold family).
- **Missions & Glossary**: Complete the 8 optional guided missions to learn how diffusion, competition, hypoxia, stromal barriers, clonal resistance, and Pareto trade-offs work. Open the **Glossary** (`G`) or **Tutorial** (`T`) anytime.

---

## 2. Advanced Research in EVOCURE LAB (10 Scientific Views)

Toggle **LAB** in the top bar to unlock all 10 specialized research views:

1. **Chemical Universe**: Configure physical Kelvin temperature, pH, viscosity, advection currents, semi-permeable barriers, and spatial niches; add compounds via SMILES, formula, MOL, or `.sdf`/`.mol` file upload.
2. **Evolution Laboratory**: Track real-time Shannon diversity and population trajectories; compare Mode A (*Artificial Chemical Life*) ecological selection against Mode B (*Grounded Chemistry*) **15-Stage NSGA-II** multi-objective molecular optimization.
3. **Cancer & Disease Battle**: Real-time 2D cellular & PDE visualization across **PDAC**, **KRAS-G12C Lung Cancer (`NSCLC_KRAS`)**, **Carbapenem-Resistant Bacterial Biofilm (`AMR_BACTERIAL`)**, and **Metabolic Steatohepatitis (`METABOLIC_MASLD`)**. Run controlled single-agent comparisons, **2-Drug Combination `8x8` Bliss/Loewe Synergy & 4-Arm Trials**, or coupled drug-tumor **Coevolution**.
4. **Tumor & Tissue Microenvironment**: Inspect continuous spatial reaction-diffusion grids ($\text{O}_2$, nutrients, ECM/biofilm/collagen matrix density, hindered drug diffusivity).
5. **Molecular Explorer & Interactive 2D Editor**: Inspect RDKit atom-indexed 2D structures, Bemis-Murcko scaffolds, Delaney ESOL solubility, $\log D_{7.4}$, TPSA, QED, and SMARTS groups; perform **35 live RDKit-validated structural edits** (substituents, heteroatom swaps, ring fusions, bioisosteres), compute **Coral/Cyan MCS structural diffs**, and spawn edited molecules directly into the Universe or Disease slice.
6. **Family Tree**: Navigate the interactive molecular genealogy DAG from founders to extant and extinct descendants, diff any two lineage nodes structurally, and export the full lineage graph as JSON.
7. **Toxicity & ADMET**: Inspect tripartite exposure ($C_{\text{applied}}$, $C_{\text{free}}$, $C_{\text{intracellular}}$), split-conformal QSAR prediction intervals, clone-specific Hill curves, Tanimoto Applicability Domain warnings, and the 10-category toxicology panel with uncertainty bounds.
8. **Biological Indicators**: View 2D tissue biomarkers alongside systemic hematological/organ panels. Toggle the **7-Compartment Systemic PBPK** ODE engine on/off to see why systemic blood markers are strictly `"Not modeled"` in 2D-only mode vs. `"Exploratory model result"` under multicompartment PBPK.
9. **AI Research Assistant**: Check live **Gemma 4** (local GGUF + CUDA `llama-server`) and neural **Laya AI** (`system_one` multilingual checkpoint) status, launch the local LLM on demand, run neural + safety-rule candidate triage, and approve/reject AI-proposed simulation actions.
10. **Experiment Comparison & Reports**: Run **Compare Worlds** side-by-side across compounds, environments, or diseases with multi-seed replication (`mean ± std`), and export reproducible **`PDF`**, **`Markdown`**, **`JSON`**, **`CSV`**, and **`SDF`** artifacts.

---

## 3. Accessibility & Keyboard Shortcuts

- **Colorblind Modes**: Switch between *Lab Bioluminescent*, *Okabe-Ito Colorblind-Safe*, and *High Contrast*. Shapes (`circle`, `hexagon`, `diamond`, `star`, `square`, `ring`) always distinguish entity and cell types independently of color.
- **Reduced Motion & Font Scaling**: Toggle reduced FX/animations and scale UI text (`Normal / Large / XL`).
- **Keyboard Shortcuts**: `Space` (Play/Pause), `M` (Switch PLAY/LAB), `G` (Glossary), `T` (Tutorial).
