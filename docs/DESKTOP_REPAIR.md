# Desktop repair and export guide

Double-click `start_evocure.bat` on Windows. It starts the local Python engine and the exported Godot application. Simulation playback begins automatically. On Linux use `./start_evocure.sh` after setting up the virtual environment.

In **Chemical Universe**, add compounds with the buttons or the library. Click **Stir / mix populations** to agitate the actual particles. Watch collisions, reaction rings, complex formation, resource competition, reproduction, and inherited variants. Select a particle to inspect its structure and family tree. Small worlds show every particle; larger worlds indicate how many are shown.

**Play / Pause**, speed controls, and **Run 100 gen / Run 1,000 gen** advance the same simulation. Long runs report progress and can be paused/resumed. Their duration depends on the population and your computer. Generations and virtual years are abstract simulation clocks.

**Chat & Export** provides offline explanations from actual world data. A local Gemma model can be started separately if its local runtime and weights are installed. The UI identifies which provider answered. Failed requests produce an error instead of an invented scientific result. Chat does not automatically change the world.

## Discover compounds and guide evolution

Open **Evolution Lab** to see actual compounds from the running chemical universe. The list includes structure identity, origin generation, transformation and current population; extinct structures remain in the registry. Choose **Directed shortlist** after a search to see the requested new variants, or **All directed structures** to inspect starting controls as well.

Enter a research objective such as “Find 5 experimental candidates for pancreatic cancer with low predicted toxicity.” Select **Ask AI to plan**, review the priorities and seeds, then **Run this guided search**. Gemma proposes categorical priorities locally when it is running; a named rule-based planner remains available offline. The proposed weights actually affect selection. The search creates and validates real RDKit variants, reports progress, and registers its results. Stop cancels unfinished work. Candidates are hypotheses, not cures.

The selected structure is compared against 28 curated local drug references. An exact canonical isomeric match is distinct from fingerprint similarity. The interface shows the nearest reference structure and descriptor differences. A missing match does not prove novelty; similarity does not prove efficacy. Selected candidates and reference drugs can be transferred into the tissue workspace.

## View each disease and its response

Choose the disease in the top bar and open **Disease & Tissue**. PDAC and KRAS NSCLC show tumour cells, resistant clones, healthy cells, stroma and vessels. AMR shows bacteria, biofilm and host cells. MASLD shows steatotic hepatocytes, activated stellate cells and liver tissue. AMR and MASLD are not tumour diseases.

The view uses actual model positions, cell radius, death state, oxygen, intracellular drug and clone identity. Membrane motion, nuclei and categorical steatosis droplets are illustrative. Death transitions and new-cell pulses require corresponding state changes. Select overlays for oxygen, drug, matrix/biofilm/fibrosis or nutrients; click a cell for details and use the wheel to zoom.

Apply the disease's default reference exposure and choose **Play tissue** or step biological time. Drug concentration, uptake, deaths and resistance update from the engine. **Compare with untreated control** runs matched-seed simulations. Novel compounds have an exploratory PDAC response surrogate; unknown responses in the other diseases are explicitly unavailable. Their guided searches remain chemistry-only. PDAC also provides independent-hazard four-arm comparisons and real structural coevolution. These models are not clinically validated.

**Reset this tissue** resets the selected disease while retaining chemistry. **Reset defaults** restores the original seed, environment, world, PDAC tissue, playback and view defaults, retaining saved worlds. Both resets explain their effect before replacing unsaved state.

## GPU acceleration

On this machine, Vulkan rendering uses the RTX 3060. Local Gemma uses GPU layer offload; Laya and tissue oxygen/nutrient field calculations use CUDA. The top bar reports the device/backend. Chemistry, stochastic cell/particle decisions, QSAR, PBPK and SQLite still run on the CPU. A small field can cost more to transfer than to compute; GPU use does not guarantee a speedup. Set `EVOCURE_COMPUTE=cpu` to force the numerical CPU fallback. CPU/GPU substrate parity is tested for all four diseases to an absolute tolerance of 1e-12.

Use **Save current world**, **Reopen selected world**, or create/restore a checkpoint. New saves contain particle properties, compound IDs, genealogy, fields, random generator state, and the disease microenvironment. Older saves may contain reports only and cannot reconstruct a full world; the app explains this if you try to reopen one.

Export the current world using **JSON**, **CSV**, **SDF**, **Markdown**, or **PDF**. Each export saves the current state first and opens a destination picker. JSON includes the resumable runtime state; CSV contains recorded trajectory points; SDF contains the experiment's structures. Reports and model outputs remain exploratory simulation results.

## Export the Godot application

The Godot project includes Windows and Linux export presets. Windows and Linux binaries are in `build/windows/` and `build/linux/`. The Windows executable embeds its game resources. Linux uses the executable plus its `.pck` file.

To rebuild using installed Godot 4.7 and matching export templates:

```powershell
.\.venv\Scripts\python.exe scripts\export_desktop.py --platform all
```

The executable is the frontend. It still needs the project's local Python engine, dependencies and any optional model files. The root launcher starts both. Moving just the executable to another machine does not package Python or the models.

## Scientific scope

Artificial life is a computational abstraction. Reactions consume reactants and emit balanced products; only artificial protocells reproduce. The current rates and environmental scales are exploratory, without experimental calibration. Mechanical stirring uses abstract velocity units. The energy and diversity charts describe the ecosystem; they are not therapeutic efficacy measurements. Unavailable toxicity/activity values are shown as unknown. This repair does not establish clinical validity or complete every research feature in the original brief.

## Validation

Run `.venv\Scripts\python.exe -m pytest -q` for the automated checks. Set `EVOCURE_DISABLE_LAYA_NEURAL=1` to exercise the required offline fallback and keep optional neural inference out of simulation timing measurements.

Godot resource imports, native exports and an executable desktop audit cover all six workspaces, actual guided searches, reference matching, all four disease views/overlays/exposures, PDAC combination/coevolution, editing, PBPK, chat, checkpoints and JSON/CSV/SDF/Markdown/PDF file exports. Run the explicit `--desktop-audit=<absolute-output-folder>` frontend option only against an isolated test backend: it resets that backend and writes screenshots/results. Validation evidence is in `reports/desktop_feature_audit/` and the test logs. Linux export was built on Windows; Linux execution still requires validation on a Linux machine.
