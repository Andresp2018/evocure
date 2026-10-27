# Local REST & WebSocket API Reference - EVOCURE v1.0.0

Base URL: `http://127.0.0.1:8000`  
Interactive OpenAPI Schema (local): `http://127.0.0.1:8000/docs`  
WebSocket Stream: `ws://127.0.0.1:8000/ws`

---

## 1. System, Multi-Disease Selection, Modes & Configuration

- `GET /api/v1/health` - Returns simulator version (`1.0.0`), active disease, 3-clock telemetry, hardware CPU/RAM/GPU telemetry, and live `Gemma 4` / `Laya AI` / `QSAR` runtime status.
- `GET /api/v1/state` - Full snapshot of the active universe, active 2D disease microenvironment, genealogy summary, biomarkers, mission progress, and recent events.
- `GET /api/v1/library` - Returns the curated **45-compound** offline chemical & multi-disease reference library (PDAC, NSCLC, AMR, MASLD, prebiotics, controls).
- `GET /api/v1/missions-glossary` - Returns the 8 EVOCURE PLAY missions, achievements, interactive scientific glossary, and 10-step tutorial.
- `GET /api/v1/diseases` - Lists all 4 active 2D mechanistic `DiseaseModule` engines (`PDAC`, `NSCLC_KRAS`, `AMR_BACTERIAL`, `METABOLIC_MASLD`).
- `POST /api/v1/diseases/select` - Body `{"disease_id": "PDAC" | "NSCLC_KRAS" | "AMR_BACTERIAL" | "METABOLIC_MASLD", "seed": 42}`. Switches the active 2D biomedical disease engine dynamically and broadcasts `disease_changed`.
- `POST /api/v1/mode/interface` - Body `{"mode": "PLAY" | "LAB"}`. Switches presentation mode instantaneously without altering simulation state or RNG seed.
- `POST /api/v1/mode/universe` - Body `{"mode": "ARTIFICIAL_LIFE" | "GROUNDED_CHEMISTRY"}`.
- `POST /api/v1/environment` - Updates 2D physical/environmental parameters (`temperature_k`, `ph`, `viscosity_pa_s`, `oxygen_norm`, `resource_influx_rate`, `current_vx`, `current_vy`, `use_physical_units`).
- `POST /api/v1/environment/preset` - Body `{"preset": "pancreatic_tissue" | "bloodstream" | "tumor_microenvironment" | "oxidative_stress" | "acidic_low_ph" | "hypoxia_low_o2"}`. Applies calibrated biophysical presets instantaneously.
- `GET /api/v1/entity-lineage/{compound_id}` - Traces the ancestor-to-descendant evolutionary chain (`G0 -> Gn`) with visual genome descriptors and SMILES scaffolds for the Godot 4.7 Generational Evolution Strip.

---

## 2. Chemical Universe, 2D Molecular Editor & Evolutionary Time Control

- `POST /api/v1/world/reset` - Body `{"preset": "chemical_life" | "cancer_battle" | "evolve_compounds" | "surprise_me" | "sandbox_empty", "seed": 42}`.
- `POST /api/v1/world/add-compound` - Introduces compounds by name, formula, SMILES, or MOL block at optional `(x, y)` coordinates.
- `POST /api/v1/world/import-sdf` - Parses multi-molecule `.sdf` / `.mol` content via RDKit and registers structures locally.
- `POST /api/v1/world/step` - Advances `steps` evolutionary/physical ticks and returns delta telemetry + `period_summary`.
- `POST /api/v1/world/advance-preset` - Executes accelerated headless evolution (`10_gen`, `100_gen`, `1000_gen`, `10000_gen`, `1_yr`, `10_yr`, `100_yr`, `1000_yr`, `10000_yr`).
- `POST /api/v1/playback` - Controls continuous WebSocket playback (`play`, `pause`, `step`, `1x`, `10x`, `100x`).
- `GET /api/v1/editor/operations` - Lists all 35 supported 2D medicinal chemistry operations (`substituents`, `atom_replacements`, `ring_fusions_and_appends`, `bioisosteres`).
- `POST /api/v1/editor/apply` - Applies an RDKit-sanitized structural transformation (`attach_substituent`, `replace_atom`, `fuse_or_append_ring`, `apply_bioisostere`) and returns the edited canonical SMILES, atom-indexed 2D SVG, and updated physicochemical descriptors.
- `POST /api/v1/editor/diff` - Computes Maximum Common Substructure (`rdFMCS`), Tanimoto similarity, property deltas, and Coral/Cyan highlighted 2D SVG diffs between two molecules.

---

## 3. Multi-Disease Microenvironment, Combination Synergy, Toxicology & PBPK

- `POST /api/v1/tumor/step` - Steps the active 2D disease cellular & reaction-diffusion PDE engine.
- `POST /api/v1/tumor/introduce-compound` - Administers a validated compound (`applied_concentration_um`, `injection_region`, optional `enable_systemic_pbpk`, `dose_mg`).
- `POST /api/v1/tumor/controlled-comparison` - Runs identical-seed untreated vehicle control (`0.0 uM`) vs. exposed 2D disease microenvironment over `duration_hours`.
- `POST /api/v1/tumor/combination` - Evaluates 2-drug combination synergy (`8x8` Bliss Independence & Loewe Additivity CI matrix) plus a 4-arm *in silico* trial (`Control` vs. `Mono A` vs. `Mono B` vs. `Combination A+B`).
- `POST /api/v1/tumor/coevolution` - Executes coupled drug-disease coevolution and returns dual parallel phylogenetic trees (`chemical_lineage_tree` and `tumor_clonal_tree`).
- `POST /api/v1/discovery/run` - Executes the 15-stage NSGA-II multi-objective candidate discovery pipeline.
- `POST /api/v1/discovery/evocure-001` - Executes mandatory benchmark experiment **EVOCURE-001** (Scenarios A & B).
- `GET /api/v1/compound/{query}` - Returns RDKit 2D SVG, descriptors, ADMET, split-conformal QSAR interval, Hill concentration-response curves, 10-category toxicology with epistemic tags, Tanimoto Applicability Domain check, and optional 7-compartment PBPK trajectories.
- `GET /api/v1/genealogy` & `GET /api/v1/genealogy/export` - Queries or exports the molecular lineage DAG with filters and generation diffing.
- `GET /api/v1/biomarkers` - Returns Disease/Tumor, Hematological, Metabolic/Organ, and Oncological biomarker panels (`UNMODELED` when PBPK is disabled; `EXPLORATORY_MODEL` when PBPK is enabled).

---

## 4. Multi-Provider AI (`Local Gemma 4` + `Google Gemini` + `OpenAI` + `Laya`), Persistence & Exports

- `GET /api/v1/assistant/status` - Returns live runtime telemetry for local `Gemma 4`, neural `Laya AI`, active provider, and split-conformal `QSAR`.
- `GET /api/v1/assistant/providers` - Lists all supported AI Assistant providers (`auto_local`, `local_gemma`, `gemini`, `openai`, `deterministic`), recommended models, availability, and masked key hints (`never exposes raw API keys`).
- `POST /api/v1/assistant/providers/configure` - Configures the active AI provider (`provider`, `model`, optional masked `api_key`, optional OpenAI-compatible `base_url`, and `persist_to_dotenv`).
- `POST /api/v1/assistant/providers/test` - Runs a live diagnostic ping against the selected AI provider (`Local Gemma 4`, `Google Gemini`, `OpenAI`, or `Deterministic`) and returns latency and verification status.
- `POST /api/v1/assistant/start-local-llm` - Spawns local CUDA `llama-server.exe` on `Gemma-4-E2B-Heretic-Q4_K_M.gguf` (on-demand background inference server).
- `POST /api/v1/assistant/ask` - Grounded natural-language query (`question`, optional `provider` & `model` override) returning `answer`, `provider`, `epistemic_category`, and optional `proposed_action`.
- `POST /api/v1/assistant/laya-triage` - Runs live neural `Laya` (`system_one`) + hard physical/toxicological safety rule triage across active candidate molecules.
- `POST /api/v1/assistant/approve-action` - Explicitly applies a user-approved parameter change suggested by the assistant.
- `POST /api/v1/compare-worlds` - Runs side-by-side world, compound, or multi-disease comparisons across repeated seeds (`mean ± std`).
- `GET /api/v1/experiments`, `POST /api/v1/experiments/save`, `POST /api/v1/experiments/load/{exp_id}`, `POST /api/v1/experiments/duplicate/{exp_id}`.
- `GET /api/v1/checkpoints`, `POST /api/v1/checkpoints/create`, `POST /api/v1/checkpoints/restore/{ckpt_id}`.
- `GET /api/v1/export/{exp_id}/{fmt}` - Exports reproducible `json`, `csv`, `sdf`, Markdown `report`, or native binary vector **`pdf`** (`exp_id` accepts `"latest"` to snapshot and export the live session automatically).

---

## 5. WebSocket Protocol (`/ws`)

- **Server → Client**: JSON frames (`state_update`, `disease_changed`, `combination_evaluated`, `checkpoint_restored`, etc.) containing entity coordinates, downsampled concentration heatmaps, 3-clock telemetry, cell states, and event notifications.
- **Client → Server**: JSON commands (`playback`, `inspect`, `add_compound`, `ping`).
