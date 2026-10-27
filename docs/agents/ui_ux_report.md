# EVOCURE UI/UX & Full-Stack Presentation Architecture Report (v1.0.0)

## 1. Executive Summary

The EVOCURE UI/UX and Full-Stack Presentation module delivers a dual-experience, 100% local, zero-external-CDN, epistemically transparent interface for computational chemical evolution and virtual multi-disease oncology simulations, headlined by virtual pancreatic cancer (PDAC) and expanding across NSCLC KRAS, AMR Bacterial, and Metabolic MASLD.

Operating under the strict multi-agent contracts defined in `simulation/core/contracts.py` and `docs/AGENT_CONTRACTS.md`:
- **Single Shared Engine**: Both **EVOCURE PLAY** and **EVOCURE LAB** run against a unified FastAPI / WebSocket backend (`app/`) orchestrating the live `EVOCUREEngine` without state duplication or simulated mock disconnects. Version bumped to production **v1.0.0**.
- **Strict Local Isolation**: **Zero external CDNs**, **zero Google Fonts URLs**, **zero external telemetry/analytics scripts**, and **zero Node.js/npm runtime dependencies**. The application runs 100% offline out-of-the-box from local assets in `frontend/static/`.
- **100% Element Wire-Up**: Comprehensive UI audit confirms that all **104 interactive HTML elements** (`<button>`, `<input>`, `<select>`, `<canvas>`) in `index.html` are explicitly bound to event handlers and backed by live backend API endpoints.
- **Epistemic Transparency**: Every metric, biological claim, QSAR prediction, and AI response carries mandatory epistemic attribution tags (`EpistemicCategory`), distinguishing calculated properties from exploratory models and hypotheses.
- **Universal Accessibility**: Features WCAG 2.1 AA compliance, dedicated **Okabe-Ito Colorblind-Safe** and **High-Contrast Dark** palettes, multi-attribute shape coding (never relying on color alone), full keyboard navigation (`Space`, `M`, `G`, `T`), font size scaling, and a reduced-motion toggle.
- **Rapid Entry & Low Friction**: EVOCURE PLAY provides a **4-Card Quick-Launch Hub** and a **"Surprise Me!"** button starting a live simulation in **<= 2 clicks**.

---

## 2. Architecture & File Structure

```
e:\MA\EVOCURE\
├── app\
│   ├── __init__.py               # App package exports
│   ├── main.py                   # FastAPI app v1.0.0, static mounting, lifespan manager
│   ├── schemas\
│   │   ├── __init__.py           # Schema exports
│   │   └── models.py             # Pydantic request/response models for all endpoints
│   ├── services\
│   │   ├── __init__.py           # Service exports
│   │   └── session_manager.py    # Thread-safe SessionManager, 20 Hz loop, WebSockets
│   └── api\
│       ├── __init__.py           # Router exports
│       └── routes.py             # 35 REST endpoints under /api/v1/ and /ws WebSocket
├── frontend\
│   └── static\
│       ├── index.html            # Semantic, accessible HTML5 structure (v1.0.0, 104 IDs)
│       ├── css\
│       │   └── styles.css        # Futuristic obsidian glassmorphism & color palettes
│       └── js\
│           ├── canvas_universe.js # 60 FPS Chemical Universe 2D Canvas renderer
│           ├── canvas_tumor.js    # 2D PDAC Microenvironment Canvas renderer
│           ├── charts_and_tree.js # Canvas 2D charts, Synergy & Microenv heatmaps, Hill & PBPK curves, Family Tree DAG
│           ├── play_mode.js       # EVOCURE PLAY cards, missions, glossary, tutorial, plain English explainer
│           └── app.js             # Main ES6 controller, WebSocket & state synchronizer
├── run.py                        # One-command CLI launcher with system telemetry
├── tests\
│   └── integration\
│       └── test_wave4b_e2e.py    # 10/10 passing end-to-end integration test suite
└── docs\agents\
    └── ui_ux_report.md           # This engineering & scientific UI/UX report
```

---

## 3. Backend & Real-Time Synchronization Engine

### 3.1. Thread-Safe Session Management (`app/services/session_manager.py`)
- **Engine Wrapping**: Integrates directly with `simulation.core.engine.EVOCUREEngine`, dynamically bridging chemistry, physics, PDAC cellular models, Darwinian evolution, QSAR predictors, systemic PBPK, toxicology, and the local AI decision engine.
- **Background Simulation Loop**: Executes an asynchronous `asyncio.Task` loop at 20 Hz. When unpaused, it advances simulation steps scaled by user-selected multipliers (`1x`, `10x`, `100x`) targeting either the **Universe Clock** or the **Tumor Clock**.
- **WebSocket Broadcasting**: Transmits compact, normalized state snapshots to all active subscribers upon every step, reset, or playback state transition, ensuring instant responsiveness across tabs.
- **Multi-Parameter Normalization**: Accommodates interchangeable parameter naming (`compound_a` vs `compound_a_query`, `dose_a_um` vs `conc_a_um`, `smiles_a` vs `query_a`) ensuring seamless inter-agent and UI compatibility.
- **PDF Generation**: Generates `%PDF-1.3` vector report documents directly on the backend via `/api/v1/export/{exp_id}/pdf` and `/api/v1/export/latest/pdf` without external binary dependencies.

### 3.2. REST & WebSocket Routing (`app/api/routes.py`)
The platform exposes **35 REST endpoints** under `/api/v1/` and a bi-directional WebSocket channel at `/ws`:

| Category | Endpoint | Method | Functionality |
|---|---|---|---|
| **System** | `/health` | GET | System uptime, version `1.0.0`, hardware profile, local AI status |
| | `/state` | GET | Full normalized state snapshot |
| | `/library` | GET | 30 curated compounds from `simulation.chemistry.library` |
| | `/missions-glossary`| GET | 8 educational missions, achievements, and glossary |
| | `/diseases` | GET | Active diseases (PDAC, NSCLC KRAS, AMR Bacterial, Metabolic MASLD) |
| | `/diseases/select` | POST | Disease model switching with state re-initialization |
| **Control** | `/mode/interface` | POST | Seamless toggle between `PLAY` and `LAB` |
| | `/mode/universe` | POST | Switch between `ARTIFICIAL_LIFE` and `GROUNDED_CHEMISTRY` |
| | `/environment` | POST | Live patch of temperature, pH, viscosity, oxygen, influx |
| | `/world/reset` | POST | Scenario reset (`chemical_life`, `cancer_battle`, `evolve_compounds`, `surprise_me`, `sandbox_empty`) |
| | `/world/add-compound`| POST | Spawn candidate molecules or replicators at spatial coordinates |
| | `/world/import-sdf` | POST | Parse Molfile/SDF chemical structure payloads |
| | `/world/step` | POST | Advance universe by discrete steps and $dt$ |
| | `/world/advance-preset`| POST | Accelerated evolution jump (`10_gen`, `100_gen`, `1_yr`, `10_yr`, `100_yr`, `1000_yr`) |
| | `/playback` | POST | Play, pause, step, speed (`1x`, `10x`, `100x`), clock target (`universe`, `tumor`) |
| **PDAC Biology** | `/tumor/step` | POST | Advance 2D PDAC cellular and reaction-diffusion state |
| | `/tumor/introduce-compound` | POST | Deliver therapeutic compound to tumor (vascular, periphery) |
| | `/tumor/controlled-comparison`| POST | Side-by-side comparison of candidate vs. untreated control |
| | `/tumor/combination` | POST | 2-Drug combination synergy, Bliss excess, Loewe CI, and 4-arm trial |
| | `/tumor/coevolution` | POST | Multi-cycle adaptive treatment vs. clonal selection run |
| **Discovery** | `/discovery/run` | POST | Directed NSGA-II molecular optimization pipeline |
| | `/discovery/evocure-001`| POST | Reference benchmark execution and validation report |
| **Molecular Editor** | `/editor/operations` | GET | List available substituents, atom swaps, and ring fusions |
| | `/editor/apply` | POST | Interactive chemical edit with atom highlight & delta calculation |
| | `/editor/diff` | POST | Maximum Common Substructure (MCS) alignment & Tanimoto similarity |
| **Analysis** | `/compound/{query}`| GET | Full physicochemical, ADMET, toxicity alerts, and RDKit 2D SVG |
| | `/genealogy` | GET | Evolutionary DAG nodes, edges, and extinction status |
| | `/genealogy/export` | GET | Cytoscape / JSON graph export for lineage trees |
| | `/biomarkers` | GET | PDAC indicators, CA 19-9 shedding, and PBPK systemic levels |
| **AI Assistant**| `/assistant/status` | GET | Telemetry on Gemma LLM, Laya Neural Agent, and QSAR Predictor |
| | `/assistant/ask` | POST | Local query via Gemma 4 / Laya with epistemic grounding |
| | `/assistant/laya-triage`| POST | Multi-criteria candidate triage with weighted utility ranking |
| | `/assistant/approve-action`| POST | Explicit human-in-the-loop approval of AI proposed action |
| | `/assistant/start-local-llm`| POST | Launch or check local `llama.exe` server process |
| **Comparison** | `/compare-worlds` | POST | Replicated side-by-side scenario comparison |
| **Persistence**| `/experiments` | GET/POST | List, save, load, and duplicate experiments in SQLite |
| | `/checkpoints` | GET/POST | Bookmark, list, and restore reproducible snapshots |
| | `/export/{exp_id}/{fmt}`| GET | Export data in `json`, `csv`, `sdf`, `report` (Markdown), or `pdf` |
| **WebSocket** | `/ws` | WS | Real-time state streaming and interactive control commands |

---

## 4. Frontend Design & User Experience

### 4.1. EVOCURE PLAY Experience (`frontend/static/js/play_mode.js`)
Designed for intuitive scientific gaming and education without overwhelming complexity:
1. **4-Card Quick-Launch Hub**:
   - *1. Create Chemical Life*: Seeds prebiotic soup with Brownian physics.
   - *2. Fight Cancer (PDAC)*: Initializes virtual 2D pancreatic tumor with classical and basal clones.
   - *3. Evolve Compounds*: Triggers accelerated multi-generation lineage divergence.
   - *4. Explore Discoveries*: Executes the EVOCURE-001 reference benchmark.
   - *"Surprise Me!"*: Instant 1-click randomized experiment launcher.
2. **Dual-Viewport Toggle**: One-click switching between the **Chemical Universe Canvas** (`#play-viewport-canvas`) and the **Tumor Microenvironment Canvas** (`#play-tumor-canvas`) directly within PLAY mode.
3. **Evolution Fast-Forward Presets**: Instant jumps for `+10 Gen`, `+100 Gen`, `+1 Year`, `+10 Years`, and `+1000 Years` with automatic state refresh.
4. **Game-Like Tool Palette**: Fast access to *Add*, *Inspect*, *Jump 100 Gen*, *Compare*, and *Bookmark*.
5. **Periodic Evolution Milestone Toast**: Emits notifications (e.g. *"Generation 10 Milestone: 45 transformations observed across 7 chemical families"*) derived directly from engine telemetry.
6. **Plain English Compound Explainer**: Accessible modal (`#modal-explain-compound`) explaining molecular mechanisms in plain English, including why certain modifications are toxic or beneficial.
7. **Interactive Educational Missions (M01 - M08)**: Step-by-step curriculum with explicit epistemic boundaries detailing *What the model captures* and *What remains unknown*.
8. **10-Step Interactive Tutorial**: Walkthrough modal explaining Brownian motion, RDKit structures, lineage DAGs, tumor hypoxia, and controlled controls.
9. **Interactive Glossary**: Plain-English and technical definitions for 10 foundational biocomputing concepts.

### 4.2. EVOCURE LAB Scientific Workbench (All 10 Specialized Tabs)
Designed for deep quantitative analysis, hypothesis testing, and pharmacology across 10 specialized views:
1. **Chemical Universe**: 60 FPS Canvas with zoom/pan, semi-permeable membranes, environmental niche zones, and live drag-and-drop compound injection.
2. **Evolution Laboratory**: Quantitative metrics tracking Shannon diversity index, carrying capacity, metabolic maintenance costs, and niche displacement.
3. **Cancer Battle**: 2D cellular viewport visualizing sensitive cells, resistant cells, activated CAFs, blood vessels, and hypoxic zonation. Includes 2-drug combination synergy panel with Bliss excess score, Loewe CI, 8x8 dose-response matrix, and 4-arm in silico trial cards.
4. **Tumor Microenvironment**: Continuous Eulerian grids with dedicated canvas heatmaps for **Oxygen**, **Drug Penetration**, **VEGF/Angiogenesis**, and **ECM Hindrance**.
5. **Molecular Explorer**: High-resolution RDKit 2D vector depiction, Lipinski parameters, Bemis-Murcko scaffolds, 15-stage NSGA-II candidate discovery runner, and interactive 2D structure editor with atom substitution, element swap, ring fusions, and MCS alignment diff.
6. **Family Tree DAG**: Interactive zoomable lineage graph displaying founder compounds, mutation steps, descendant clusters, and dashed extinct branches with Cytoscape/JSON export.
7. **Toxicity & ADMET**: Preclinical ADMET radar chart, Hill sigmoidal cell-kill dose-response curves, and 10-category structural alert screening with applicability domain validation.
8. **Biological Indicators**: Tabulated oncology biomarkers, CA 19-9 shedding, and systemic 7-compartment PBPK clearance canvas with organ partition coefficients.
9. **AI Research Assistant**: Local hardware telemetry strip (NVIDIA GPU, RAM, VRAM), live status of Gemma LLM, Laya Neural Agent, and calibrated QSAR, with Laya multi-criteria candidate triage ranking.
10. **Experiment Comparison**: Side-by-side baseline vs. treatment evaluation across replicated seeds with automated comparison bar charts.

---

## 5. Accessibility & Epistemic Transparency

### 5.1. Accessibility Standards (WCAG 2.1 AA)
- **Palette Adaptability**:
  - `Default Lab Neon`: High-contrast obsidian with cyan, emerald, amber, and rose accents.
  - `Okabe-Ito Palette`: Scientifically validated colorblind-safe palette (Sky Blue `#56B4E9`, Bluish Green `#009E73`, Orange `#E69F00`, Vermilion `#D55E00`, Reddish Purple `#CC79A7`).
  - `High-Contrast Dark`: Pure black (`#000000`) and pure white (`#ffffff`) with saturated neon accents.
- **Multimodal Visual Coding**: Every cellular and molecular entity uses distinct geometric shapes in addition to color:
  - Circles: Sensitive tumor / Amino acids
  - Hexagons: Resistant tumor / Pyruvates & sugars
  - Squares: Healthy epithelial cells
  - Triangles: Cancer-associated fibroblasts (CAFs)
  - Diamonds: Immune T-cells / Catalysts
  - Stars: Apoptotic cells / Nucleic bases
  - Rings: Endothelial blood vessels / Nutrients
- **Keyboard Navigation**:
  - `Space`: Toggle Play / Pause
  - `M`: Toggle PLAY / LAB modes
  - `G`: Toggle Scientific Glossary
  - `T`: Open Interactive Tutorial
  - Tab navigation with visible focus rings on all interactive elements.
- **Motion & Typography**:
  - User-configurable font sizes: `Normal` (14px), `Large` (16px), `XL` (18px).
  - Reduced-motion toggle instantly disabling all canvas ripple FX, CSS transforms, and transitions.

### 5.2. Epistemic Tagging & Scientific Transparency
Per Section 27 requirements, all outputs explicitly communicate their epistemic classification:
- `EpistemicCategory.CALCULATED_PHYSICOCHEMICAL`: RDKit molecular descriptors and properties.
- `EpistemicCategory.EXPERIMENTAL_OBSERVATION`: GDSC2 / ChEMBL literature IC50 reference data.
- `EpistemicCategory.VALIDATED_MODEL_PREDICTION`: Calibrated QSAR predictions within the applicability domain.
- `EpistemicCategory.MECHANISTIC_SIMULATION`: 2D reaction-diffusion, Hill cell-kill kinetics, and PBPK curves.
- `EpistemicCategory.AI_HYPOTHESIS`: Local Gemma 4 and Laya decision proposals.
- `EpistemicCategory.DEMONSTRATION_UNVALIDATED`: Uncalibrated exploratory models.
- **Strict Guardrail**: Computational cell kill in the 2D tumor microenvironment never emits unsubstantiated claims of clinical efficacy or human patient dosing.

---

## 6. Verification & Quality Assurance

1. **JavaScript Syntax Verification**:
   All 5 frontend scripts validated using `node -c`:
   - `frontend/static/js/canvas_universe.js`: **0 syntax errors**
   - `frontend/static/js/canvas_tumor.js`: **0 syntax errors**
   - `frontend/static/js/charts_and_tree.js`: **0 syntax errors**
   - `frontend/static/js/play_mode.js`: **0 syntax errors**
   - `frontend/static/js/app.js`: **0 syntax errors**

2. **100% UI Element Audit**:
   Verified via automated inspection that:
   - **0 external URLs** exist in HTML/CSS (100% offline, zero CDN).
   - **104 of 104 interactive HTML IDs** (`<button>`, `<input>`, `<select>`, `<canvas>`) in `index.html` are explicitly referenced, event-bound, and rendered across the JavaScript controllers.

3. **End-to-End Integration Test Suite (`tests/integration/test_wave4b_e2e.py`)**:
   Executed with `e:\MA\EVOCURE\.venv\Scripts\python.exe -m unittest tests/integration/test_wave4b_e2e.py`:
   - `test_01_health_and_version`: Passed (Health, version `1.0.0`, GPU profile).
   - `test_02_diseases_endpoint_and_selection`: Passed (4 diseases listed, selection verified).
   - `test_03_editor_operations_and_apply`: Passed (Available ops retrieved, chemical edit applied with SVGs and deltas).
   - `test_04_editor_diff`: Passed (MCS smarts computed, dual highlighted SVGs rendered).
   - `test_05_tumor_combination_synergy`: Passed (Bliss excess, Loewe CI, 8x8 matrix, 4-arm trial verified).
   - `test_06_tumor_coevolution`: Passed (Multi-cycle adaptive treatment verified).
   - `test_07_assistant_status_and_triage`: Passed (AI status retrieved, Laya multi-criteria triage verified).
   - `test_08_genealogy_and_export`: Passed (Lineage nodes/edges retrieved, JSON export verified).
   - `test_09_experiments_and_pdf_export`: Passed (Experiment saved, listed, valid `%PDF-1.3` bytes returned).
   - `test_10_compare_worlds`: Passed (Replicated side-by-side scenario comparison verified).
   - **Test Summary**: `Ran 10 tests in 17.448s` - **OK (10/10 Passed, 0 Failures, 0 Errors)**.

4. **Local Isolation Verification**:
   - Inspected `index.html`, `styles.css`, and JS files: **0 external URLs**, **0 CDNs**, **0 Google Fonts**.
   - Full platform launches and runs completely offline via `python run.py`.
