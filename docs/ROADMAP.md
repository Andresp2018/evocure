# Post-v1.0.0 Research Horizons - EVOCURE

EVOCURE `v1.0.0` delivers the **100% complete local scientific simulation & discovery platform**, having already completed and verified all planned `v0.1 -> v1.0` milestones:
- **Completed in `v1.0.0`**: Split-conformal QSAR prediction intervals across a 90-compound assay-partitioned multi-disease reference dataset (`0%` scaffold leakage, `ROC-AUC = 0.9167`).
- **Completed in `v1.0.0`**: All 4 mechanistic 2D disease engines active with zero stubs (`PDAC`, `NSCLC_KRAS`, `AMR_BACTERIAL`, `METABOLIC_MASLD`), dual-drug reaction-diffusion PDEs, Bliss/Loewe `8x8` synergy matrices, and 4-arm *in silico* trials.
- **Completed in `v1.0.0`**: Interactive 2D RDKit Molecular Structure Editor + MCS Coral/Cyan structural diff depictions.
- **Completed in `v1.0.0`**: Live local neural `Laya` System-1 triage + local `Gemma 4` CUDA/CPU `llama-server` integration + native binary `PDF` (`fpdf2`) publication reports.

---

## Future Scientific Extensions (`v2.0+` Research Horizons)

### Horizon 1 - 3D Volumetric Organoid PDEs & Spatial Transcriptomics Import
- Extending the 2D finite-volume reaction-diffusion solver to 3D patient-derived organoid (PDO) meshes and importing local 10x Visium spatial transcriptomics slices as initial cellular coordinates.

### Horizon 2 - Structure-Based 3D Pocket Docking Integration
- Optional local integration with open-source 3D protein-ligand scoring (e.g., AutoDock Vina / Gnina on local `.pdb`/`.cif` targets such as KRAS G12D, EGFR T790M, KPC-2, and THR-$\beta$) alongside ligand-based QSAR.

### Horizon 3 - Multi-Gigabyte Offline ChEMBL / DepMap Warehouse Importers
- Streaming SQLite/Parquet ingestion of full multi-gigabyte ChEMBL v34+ and DepMap PRISM dumps through `ScientificDataManager.import_local_records`.

### Horizon 4 - Prospective Wet-Lab & Usability Validation Loops
- Active-learning Bayesian experimental design closing the loop with robotic *in vitro* 384-well spheroid assays, plus formal multi-cohort human usability studies on `EVOCURE PLAY`.
