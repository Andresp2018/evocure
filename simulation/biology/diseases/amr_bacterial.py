"""
Carbapenem-Resistant Bacterial Biofilm Infection (AMR) Module.

Implements a 2D mechanistic infectious microenvironment modeling multi-drug
resistant bacterial biofilm colonies (Acinetobacter baumannii / Klebsiella
pneumoniae / Pseudomonas aeruginosa). Incorporates extracellular polymeric
substances (EPS biofilm matrix diffusion hindrance), Monod nutrient-limited
bacterial replication, beta-lactamase enzymatic drug hydrolysis, active
MexAB-type efflux, host mucosal epithelium, immune phagocytosis, single
antibiotic and synergistic combination regimens, and calibrated reference anchors.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
from simulation.core.acceleration import substrates as gpu_substrates

from simulation.core.contracts import (
    CellDeathMode,
    CellType,
    CellularEntity,
    CompoundRecord,
    DiseaseModule,
    EpistemicCategory,
    ScientificValue,
)


class AMRBacterialModule(DiseaseModule):
    """
    Mechanistic 2D Carbapenem-Resistant Bacterial Biofilm Model.

    Biological & biophysical features:
    1. Extracellular Polymeric Substance (EPS) Biofilm Matrix (mapped to heatmaps["ecm_density"]).
       EPS severely hinders antibiotic diffusivity: D_eff = D_0 / (1 + beta_biofilm * EPS).
    2. Monod nutrient-limited bacterial division.
    3. Susceptible planktonic/colony bacteria (CellType.TUMOR_SENSITIVE) vs
       Beta-lactamase + Efflux-pump producing persister bacteria (CellType.TUMOR_RESISTANT).
    4. Enzymatic Drug Hydrolysis: Beta-lactamase positive persisters actively degrade
       vulnerable antibiotics in their microenvironment.
    5. Host mucosal epithelium (CellType.EPITHELIAL_HEALTHY) and phagocytes (CellType.MACROPHAGE).
    6. Calibrated reference anchors for Ciprofloxacin, Amoxicillin, Doxycycline, Meropenem.
    """

    REFERENCE_COMPOUNDS: dict[str, dict[str, Any]] = {
        "MEROPENEM": {
            "name": "Meropenem",
            "smiles": "CC1C2C(C(=O)N2C(=C1SC3CNC(C3)C(=O)N(C)C)C(=O)O)C(C)O",
            "ic50_sensitive_um": 0.25,
            "ic50_resistant_um": 12.50,
            "ic50_healthy_um": 50.0,
            "fraction_unbound_plasma": 0.98,
            "susceptible_to_kpc": True,
            "mechanism": "Carbapenem PBP2/3 Cell Wall Synthesis Inhibitor",
        },
        "CIPROFLOXACIN": {
            "name": "Ciprofloxacin",
            "smiles": "C1CC1N2C=C(C(=O)C3=CC(=C(C=C32)N4CCNCC4)F)C(=O)O",
            "ic50_sensitive_um": 0.15,
            "ic50_resistant_um": 3.20,
            "ic50_healthy_um": 35.0,
            "fraction_unbound_plasma": 0.70,
            "susceptible_to_kpc": False,
            "mechanism": "Fluoroquinolone DNA Gyrase / Topo IV Inhibitor",
        },
        "AMOXICILLIN": {
            "name": "Amoxicillin",
            "smiles": "CC1(C(N2C(S1)C(C2=O)NC(=O)C(C3=CC=C(C=C3)O)N)C(=O)O)C",
            "ic50_sensitive_um": 0.80,
            "ic50_resistant_um": 28.00,
            "ic50_healthy_um": 80.0,
            "fraction_unbound_plasma": 0.82,
            "susceptible_to_kpc": True,
            "mechanism": "Aminopenicillin Cell Wall Inhibitor",
        },
        "DOXYCYCLINE": {
            "name": "Doxycycline",
            "smiles": "CC1C2CC3C(C(=O)C(=C(C3(C(=O)C2=C(C4=C1C=CC=C4O)O)O)O)C(=O)N)N(C)C",
            "ic50_sensitive_um": 0.50,
            "ic50_resistant_um": 4.80,
            "ic50_healthy_um": 30.0,
            "fraction_unbound_plasma": 0.18,
            "susceptible_to_kpc": False,
            "mechanism": "Tetracycline 30S Ribosomal Protein Synthesis Inhibitor",
        },
        "AVIBACTAM": {
            "name": "Avibactam (Beta-Lactamase Inhibitor)",
            "smiles": "C1CC2N(C1)C(=O)N(C2C(=O)N)OS(=O)(=O)O",
            "ic50_sensitive_um": 25.0,
            "ic50_resistant_um": 20.0,
            "ic50_healthy_um": 100.0,
            "fraction_unbound_plasma": 0.90,
            "inhibits_kpc": True,
            "mechanism": "Non-beta-lactam Diazabicyclooctane Beta-Lactamase Inhibitor",
        },
    }

    def __init__(
        self,
        seed: int = 42,
        width: float = 800.0,
        height: float = 600.0,
        nx: int = 40,
        ny: int = 30,
    ) -> None:
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self.width = float(width)
        self.height = float(height)
        self.nx = int(nx)
        self.ny = int(ny)
        self.dx = self.width / self.nx
        self.dy = self.height / self.ny

        # Clocks
        self.biological_seconds: float = 0.0
        self.step_count: int = 0

        # Physical / Transport Parameters
        self.diffusivity_oxygen: float = 35.0
        self.diffusivity_nutrient: float = 40.0
        self.diffusivity_drug_base: float = 55.0
        self.beta_biofilm: float = 4.2              # Strong EPS biofilm diffusion barrier
        self.clearance_drug_tissue: float = 0.00010

        # Uptake / Efflux
        self.k_perm_default: float = 0.020
        self.k_out_default: float = 0.008
        self.k_efflux_pump: float = 0.035           # Strong MexAB-OprM bacterial efflux

        # Kill & Growth parameters
        self.hill_slope_default: float = 2.0
        self.emax_default: float = 0.000050
        self.tau_kill: float = 3600.0

        # Continuous Grids
        self.grid_oxygen = np.ones((self.ny, self.nx), dtype=np.float64)
        self.grid_nutrient = np.ones((self.ny, self.nx), dtype=np.float64)
        self.grid_ecm = np.zeros((self.ny, self.nx), dtype=np.float64)  # EPS Biofilm Matrix
        self.grid_drug_total = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.grid_drug_free = np.zeros((self.ny, self.nx), dtype=np.float64)

        # Dual Antibiotic Grids
        self.grid_drug_b_total = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.grid_drug_b_free = np.zeros((self.ny, self.nx), dtype=np.float64)

        # Perfusion Network (Mucosal capillaries)
        self.vessels: list[dict[str, Any]] = []
        self.vessel_mask = np.zeros((self.ny, self.nx), dtype=np.float64)

        # Bacteria & Host Cells
        self.cells: list[CellularEntity] = []

        # Compounds
        self.active_compound: CompoundRecord | None = None
        self.active_pharmacology: dict[str, Any] = {}
        self.applied_concentration_um: float = 0.0
        self.fraction_unbound_plasma: float = 0.80

        self.active_compound_b: CompoundRecord | None = None
        self.active_pharmacology_b: dict[str, Any] = {}
        self.applied_concentration_b_um: float = 0.0
        self.fraction_unbound_b_plasma: float = 0.80
        self.synergy_factor: float = 1.0

        self.cumulative_apoptosis_count: int = 0
        self.cumulative_necrosis_count: int = 0

        self.initialize_scenario(seed=self._seed, preset="standard")

    @property
    def disease_id(self) -> str:
        return "AMR_BACTERIAL"

    @property
    def display_name(self) -> str:
        return "Carbapenem-Resistant Bacterial Biofilm Infection"

    def metadata(self) -> dict[str, Any]:
        return {
            "disease_id": self.disease_id,
            "display_name": self.display_name,
            "category": "infectious_disease",
            "pathogens": ["Klebsiella pneumoniae (KPC+)", "Pseudomonas aeruginosa", "Acinetobacter baumannii"],
            "histopathology": "Dense mucosal bacterial microcolonies encased in extracellular polymeric substance (EPS) biofilm",
            "implementation_status": "interface_ready_requires_calibrated_dataset",
            "is_calibrated": False,
            "epistemic_category": EpistemicCategory.DEMONSTRATION_UNVALIDATED.value,
            "cellular_entities": [
                "Susceptible Bacteria (TUMOR_SENSITIVE render tag)",
                "KPC Persister Bacteria (TUMOR_RESISTANT render tag)",
                "Host Mucosal Epithelium (EPITHELIAL_HEALTHY)",
                "Host Phagocytes / Neutrophils (MACROPHAGE)",
                "Mucosal Capillaries (ENDOTHELIAL)",
            ],
            "subclones": [
                {
                    "clone_id": "AMR-SUSCEPTIBLE",
                    "subtype": "Planktonic / Outer-Colony Bacteria",
                    "driver_mutations": ["WILD_TYPE"],
                    "relative_ic50": "1.0x (Antibiotic Susceptible)",
                    "efflux_expression": "Low (0.05)",
                },
                {
                    "clone_id": "AMR-PERSISTER-KPC",
                    "subtype": "Biofilm-Encased Carbapenemase Persister",
                    "driver_mutations": ["KPC_CARBAPENEMASE", "OXA_48", "MEXAB_EFFLUX"],
                    "relative_ic50": "12.0x (Refractory to beta-lactams)",
                    "efflux_expression": "High (0.75)",
                },
            ],
            "hallmarks_modeled": [
                "EPS biofilm matrix physical diffusion hindrance (beta_biofilm = 4.2)",
                "Enzymatic beta-lactamase drug degradation in persister microenvironment",
                "Monod nutrient-limited bacterial replication and density saturation",
                "Host mucosal epithelial cell off-target toxicity",
                "Beta-lactam + Beta-lactamase inhibitor combination synergy (Bliss excess)",
            ],
            "calibrated_reference_compounds": list(self.REFERENCE_COMPOUNDS.keys()),
            "assumptions": [
                "Bacterial colony represented as discrete agent micro-aggregates in 2D space.",
                "EPS biofilm density modeled as continuous reaction-diffusion field deposited by bacteria.",
            ],
            "limitations": (
                "2D cross-section approximates 3D mushroom-shaped biofilm water channels. "
                "Calibrated for antimicrobial PK/PD regimen comparisons."
            ),
        }

    def initialize_scenario(self, seed: int, preset: str = "standard") -> None:
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self.biological_seconds = 0.0
        self.step_count = 0
        self.cumulative_apoptosis_count = 0
        self.cumulative_necrosis_count = 0

        self.vessels = self._create_mucosal_vasculature()
        self._recompute_vessel_mask()

        # Mucosal tissue background
        self.grid_oxygen = np.clip(0.10 + 0.90 * np.power(self.vessel_mask, 0.40), 0.05, 1.0)
        self.grid_nutrient = np.clip(0.20 + 0.80 * np.power(self.vessel_mask, 0.35), 0.10, 1.0)
        self.grid_ecm.fill(0.08)  # Baseline EPS
        self.grid_drug_total.fill(0.0)
        self.grid_drug_free.fill(0.0)
        self.grid_drug_b_total.fill(0.0)
        self.grid_drug_b_free.fill(0.0)

        self.cells = self._create_amr_population()
        self._relax_initial_biofilm()

    def _create_mucosal_vasculature(self) -> list[dict[str, Any]]:
        """Submucosal vascular capillary bed supplying mucosal tissue."""
        return [
            {"id": "submuc_ves_1", "x1": 60.0, "y1": 50.0, "x2": 320.0, "y2": 70.0, "radius": 14.0, "perfusion": 1.0, "functional": True},
            {"id": "submuc_ves_2", "x1": 480.0, "y1": 60.0, "x2": 740.0, "y2": 80.0, "radius": 14.0, "perfusion": 1.0, "functional": True},
            {"id": "submuc_ves_3", "x1": 50.0, "y1": 530.0, "x2": 350.0, "y2": 510.0, "radius": 14.0, "perfusion": 1.0, "functional": True},
            {"id": "submuc_ves_4", "x1": 460.0, "y1": 520.0, "x2": 750.0, "y2": 540.0, "radius": 14.0, "perfusion": 1.0, "functional": True},
            {"id": "submuc_ves_5", "x1": 60.0, "y1": 200.0, "x2": 70.0, "y2": 400.0, "radius": 12.0, "perfusion": 0.9, "functional": True},
            {"id": "submuc_ves_6", "x1": 730.0, "y1": 210.0, "x2": 740.0, "y2": 410.0, "radius": 12.0, "perfusion": 0.9, "functional": True},
        ]

    def _recompute_vessel_mask(self) -> None:
        self.vessel_mask.fill(0.0)
        grid_y, grid_x = np.mgrid[0 : self.ny, 0 : self.nx]
        cell_centers_x = (grid_x + 0.5) * self.dx
        cell_centers_y = (grid_y + 0.5) * self.dy

        for ves in self.vessels:
            if not ves.get("functional", True):
                continue
            x1, y1 = ves["x1"], ves["y1"]
            x2, y2 = ves["x2"], ves["y2"]
            rad = ves["radius"]
            perf = ves.get("perfusion", 1.0)

            dx = x2 - x1
            dy = y2 - y1
            seg_len_sq = dx * dx + dy * dy
            if seg_len_sq == 0:
                dist = np.hypot(cell_centers_x - x1, cell_centers_y - y1)
            else:
                t = np.clip(((cell_centers_x - x1) * dx + (cell_centers_y - y1) * dy) / seg_len_sq, 0.0, 1.0)
                proj_x = x1 + t * dx
                proj_y = y1 + t * dy
                dist = np.hypot(cell_centers_x - proj_x, cell_centers_y - proj_y)

            mask = np.exp(-0.5 * (dist / max(rad, 1.0)) ** 2) * perf
            self.vessel_mask = np.maximum(self.vessel_mask, mask)

        self.vessel_mask = np.clip(self.vessel_mask, 0.0, 1.0)

    def _create_amr_population(self) -> list[CellularEntity]:
        cells: list[CellularEntity] = []
        center_x, center_y = self.width / 2.0, self.height / 2.0
        cid = 1

        # Host Mucosal Epithelium
        num_epi = 45
        for _ in range(num_epi):
            r = self._rng.uniform(190.0, 320.0)
            theta = self._rng.uniform(0.0, 2.0 * math.pi)
            cells.append(CellularEntity(
                cell_id=f"mucosa_{cid}",
                cell_type=CellType.EPITHELIAL_HEALTHY,
                clone_id="HOST-MUCOSA",
                x=float(np.clip(center_x + r * math.cos(theta), 25.0, self.width - 25.0)),
                y=float(np.clip(center_y + r * math.sin(theta), 25.0, self.height - 25.0)),
                radius=6.0,
                growth_rate=0.006,
                resistance_score=0.05,
                efflux_pump_level=0.10,
            ))
            cid += 1

        # Host Phagocytes / Neutrophils
        num_phag = 18
        for _ in range(num_phag):
            r = self._rng.uniform(140.0, 280.0)
            theta = self._rng.uniform(0.0, 2.0 * math.pi)
            cells.append(CellularEntity(
                cell_id=f"phagocyte_{cid}",
                cell_type=CellType.MACROPHAGE,
                clone_id="HOST-PHAGOCYTE",
                x=float(np.clip(center_x + r * math.cos(theta), 25.0, self.width - 25.0)),
                y=float(np.clip(center_y + r * math.sin(theta), 25.0, self.height - 25.0)),
                radius=5.5,
                growth_rate=0.003,
                resistance_score=0.15,
                efflux_pump_level=0.12,
            ))
            cid += 1

        # Susceptible Planktonic Bacteria (CellType.TUMOR_SENSITIVE)
        num_sens_bact = 65
        for _ in range(num_sens_bact):
            r = self._rng.triangular(0.0, 75.0, 150.0)
            theta = self._rng.uniform(0.0, 2.0 * math.pi)
            cells.append(CellularEntity(
                cell_id=f"bact_sens_{cid}",
                cell_type=CellType.TUMOR_SENSITIVE,
                clone_id="AMR-SUSCEPTIBLE",
                x=float(center_x + r * math.cos(theta)),
                y=float(center_y + r * math.sin(theta)),
                radius=4.5,
                growth_rate=0.048,
                mutations=["WILD_TYPE"],
                resistance_score=float(np.clip(self._rng.normal(0.05, 0.02), 0.02, 0.12)),
                efflux_pump_level=float(np.clip(self._rng.normal(0.08, 0.02), 0.04, 0.15)),
            ))
            cid += 1

        # Persister KPC+ Bacteria (CellType.TUMOR_RESISTANT)
        num_res_bact = 26
        for _ in range(num_res_bact):
            r = self._rng.triangular(15.0, 85.0, 160.0)
            theta = self._rng.uniform(0.0, 2.0 * math.pi)
            cells.append(CellularEntity(
                cell_id=f"bact_persister_{cid}",
                cell_type=CellType.TUMOR_RESISTANT,
                clone_id="AMR-PERSISTER-KPC",
                x=float(center_x + r * math.cos(theta)),
                y=float(center_y + r * math.sin(theta)),
                radius=4.8,
                growth_rate=0.038,
                mutations=["KPC_CARBAPENEMASE", "OXA_48", "MEXAB_EFFLUX"],
                resistance_score=float(np.clip(self._rng.normal(0.75, 0.04), 0.60, 0.90)),
                efflux_pump_level=float(np.clip(self._rng.normal(0.70, 0.04), 0.55, 0.88)),
            ))
            cid += 1

        return cells

    def _relax_initial_biofilm(self) -> None:
        """Establish initial EPS biofilm density around bacterial colonies."""
        for c in self.cells:
            if c.cell_type in (CellType.TUMOR_SENSITIVE, CellType.TUMOR_RESISTANT):
                gx = int(np.clip(c.x / self.dx, 0, self.nx - 1))
                gy = int(np.clip(c.y / self.dy, 0, self.ny - 1))
                r = 2
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        ny_idx, nx_idx = gy + dy, gx + dx
                        if 0 <= ny_idx < self.ny and 0 <= nx_idx < self.nx:
                            w = 1.0 / (1.0 + dx * dx + dy * dy)
                            self.grid_ecm[ny_idx, nx_idx] = min(0.85, self.grid_ecm[ny_idx, nx_idx] + 0.25 * w)

        for _ in range(25):
            self._diffuse_and_react_substrates(dt_seconds=60.0)

    def apply_compound_exposure(
        self,
        compound: CompoundRecord,
        pharmacology_profile: dict[str, Any],
        applied_concentration_um: float,
        injection_region: str = "vascular",
    ) -> dict[str, Any]:
        self.active_compound = compound
        self.active_pharmacology = copy.deepcopy(pharmacology_profile)
        self.applied_concentration_um = max(0.0, float(applied_concentration_um))

        self.active_compound_b = None
        self.active_pharmacology_b = {}
        self.applied_concentration_b_um = 0.0
        self.synergy_factor = 1.0
        self.grid_drug_b_total.fill(0.0)
        self.grid_drug_b_free.fill(0.0)

        fu_raw = pharmacology_profile.get("fraction_unbound_plasma", 0.80)
        if isinstance(fu_raw, dict): fu_raw = fu_raw.get("value", 0.80)
        try: self.fraction_unbound_plasma = float(fu_raw if fu_raw is not None else 0.80)
        except: self.fraction_unbound_plasma = 0.80

        if self.applied_concentration_um > 0.0:
            if injection_region == "vascular":
                inj_mask = self.vessel_mask > 0.15
                self.grid_drug_total[inj_mask] = np.maximum(
                    self.grid_drug_total[inj_mask], self.applied_concentration_um * self.vessel_mask[inj_mask]
                )
            elif injection_region == "uniform":
                self.grid_drug_total.fill(self.applied_concentration_um)
            self.grid_drug_free = self.grid_drug_total * self.fraction_unbound_plasma
        else:
            self.grid_drug_total.fill(0.0)
            self.grid_drug_free.fill(0.0)

        return {
            "status": "applied",
            "compound_id": compound.compound_id,
            "name": compound.name,
            "applied_concentration_um": self.applied_concentration_um,
            "fraction_unbound_plasma": self.fraction_unbound_plasma,
            "mean_grid_free_drug_um": float(np.mean(self.grid_drug_free)),
            "injection_region": injection_region,
        }

    def apply_combination_exposure(
        self,
        compound_a: CompoundRecord,
        compound_b: CompoundRecord,
        pharma_a: dict[str, Any],
        pharma_b: dict[str, Any],
        conc_a_um: float,
        conc_b_um: float,
        injection_region: str = "vascular",
        synergy_factor: float = 1.0,
    ) -> dict[str, Any]:
        self.active_compound = compound_a
        self.active_compound_b = compound_b
        self.active_pharmacology = copy.deepcopy(pharma_a)
        self.active_pharmacology_b = copy.deepcopy(pharma_b)
        self.applied_concentration_um = max(0.0, float(conc_a_um))
        self.applied_concentration_b_um = max(0.0, float(conc_b_um))
        self.synergy_factor = max(0.2, float(synergy_factor))

        def _extract_fu(p: dict[str, Any]) -> float:
            fu = p.get("fraction_unbound_plasma", 0.80)
            if isinstance(fu, dict): fu = fu.get("value", 0.80)
            try: return float(fu if fu is not None else 0.80)
            except: return 0.80

        self.fraction_unbound_plasma = _extract_fu(pharma_a)
        self.fraction_unbound_b_plasma = _extract_fu(pharma_b)

        if injection_region == "vascular":
            inj_mask = self.vessel_mask > 0.15
            self.grid_drug_total[inj_mask] = np.maximum(self.grid_drug_total[inj_mask], self.applied_concentration_um * self.vessel_mask[inj_mask])
            self.grid_drug_b_total[inj_mask] = np.maximum(self.grid_drug_b_total[inj_mask], self.applied_concentration_b_um * self.vessel_mask[inj_mask])
        elif injection_region == "uniform":
            self.grid_drug_total.fill(self.applied_concentration_um)
            self.grid_drug_b_total.fill(self.applied_concentration_b_um)

        self.grid_drug_free = self.grid_drug_total * self.fraction_unbound_plasma
        self.grid_drug_b_free = self.grid_drug_b_total * self.fraction_unbound_b_plasma

        return {
            "status": "applied_combination",
            "compound_a": {"id": compound_a.compound_id, "name": compound_a.name, "conc_um": self.applied_concentration_um},
            "compound_b": {"id": compound_b.compound_id, "name": compound_b.name, "conc_um": self.applied_concentration_b_um},
            "synergy_factor": self.synergy_factor,
            "mean_grid_free_a_um": float(np.mean(self.grid_drug_free)),
            "mean_grid_free_b_um": float(np.mean(self.grid_drug_b_free)),
            "injection_region": injection_region,
        }

    def step(self, dt_seconds: float) -> dict[str, Any]:
        dt_seconds = max(0.1, float(dt_seconds))
        self.biological_seconds += dt_seconds
        self.step_count += 1

        self._diffuse_and_react_substrates(dt_seconds)
        self._update_biofilm(dt_seconds)

        has_drug_a = self.applied_concentration_um > 0.0 or np.max(self.grid_drug_total) > 1e-6
        has_drug_b = self.applied_concentration_b_um > 0.0 or np.max(self.grid_drug_b_total) > 1e-6
        if has_drug_a or has_drug_b:
            self._diffuse_and_react_drug(dt_seconds)

        step_events = self._update_bacteria_and_cells(dt_seconds)
        self._update_phagocytosis(dt_seconds)

        return {
            "step_count": self.step_count,
            "biological_seconds": round(self.biological_seconds, 2),
            "biological_hours": round(self.biological_seconds / 3600.0, 3),
            "events": step_events,
            "active_cells_count": len([c for c in self.cells if c.death_mode == CellDeathMode.ALIVE]),
        }

    def _diffuse_and_react_substrates(self, dt_seconds: float) -> None:
        max_d = max(self.diffusivity_oxygen, self.diffusivity_nutrient)
        cfl_dt = 0.18 * (self.dx * self.dy) / max(max_d, 1.0)
        num_substeps = max(5, min(30, int(math.ceil(dt_seconds / cfl_dt))))
        sub_dt = min(cfl_dt, dt_seconds / num_substeps)

        o2_sink_grid = np.ones((self.ny, self.nx), dtype=np.float64) * 0.001
        nut_sink_grid = np.ones((self.ny, self.nx), dtype=np.float64) * 0.002

        for c in self.cells:
            if c.death_mode != CellDeathMode.ALIVE: continue
            gx = int(np.clip(c.x / self.dx, 0, self.nx - 1))
            gy = int(np.clip(c.y / self.dy, 0, self.ny - 1))
            if c.cell_type in (CellType.TUMOR_SENSITIVE, CellType.TUMOR_RESISTANT):
                o2_sink_grid[gy, gx] += 0.022
                nut_sink_grid[gy, gx] += 0.038
            else:
                o2_sink_grid[gy, gx] += 0.006
                nut_sink_grid[gy, gx] += 0.008

        accelerated = gpu_substrates(self.grid_oxygen, self.grid_nutrient, self.vessel_mask,
            o2_sink_grid, nut_sink_grid, self.dx, self.dy,
            (self.diffusivity_oxygen, self.diffusivity_nutrient), (0.05, 0.06), (0.08, 0.12), sub_dt, num_substeps)
        if accelerated is not None:
            self.grid_oxygen, self.grid_nutrient = accelerated
            return

        for _ in range(num_substeps):
            lap_o2 = self._compute_laplacian(self.grid_oxygen)
            lap_nut = self._compute_laplacian(self.grid_nutrient)

            supply_o2 = 0.05 * self.vessel_mask * (1.0 - self.grid_oxygen)
            supply_nut = 0.06 * self.vessel_mask * (1.0 - self.grid_nutrient)

            uptake_o2 = o2_sink_grid * (self.grid_oxygen / (0.08 + self.grid_oxygen))
            uptake_nut = nut_sink_grid * (self.grid_nutrient / (0.12 + self.grid_nutrient))

            self.grid_oxygen += sub_dt * (self.diffusivity_oxygen * lap_o2 + supply_o2 - uptake_o2)
            self.grid_nutrient += sub_dt * (self.diffusivity_nutrient * lap_nut + supply_nut - uptake_nut)

            self.grid_oxygen = np.clip(self.grid_oxygen, 0.0, 1.0)
            self.grid_nutrient = np.clip(self.grid_nutrient, 0.0, 1.0)

    def _update_biofilm(self, dt_seconds: float) -> None:
        """Bacteria actively deposit EPS matrix; slow natural dissolution."""
        dep_rate = 0.000045 * dt_seconds
        deg_rate = 0.000006 * dt_seconds

        # Biofilm dispersing agents
        for p in (self.active_pharmacology, self.active_pharmacology_b):
            if p.get("modulates_biofilm", False) or p.get("biofilm_disperser", False):
                deg_rate *= 3.5

        bact_mask = np.zeros((self.ny, self.nx), dtype=np.float64)
        for c in self.cells:
            if c.death_mode == CellDeathMode.ALIVE and c.cell_type in (CellType.TUMOR_SENSITIVE, CellType.TUMOR_RESISTANT):
                gx = int(np.clip(c.x / self.dx, 0, self.nx - 1))
                gy = int(np.clip(c.y / self.dy, 0, self.ny - 1))
                # Persisters secrete higher EPS
                bact_mask[gy, gx] += 1.2 if c.cell_type == CellType.TUMOR_RESISTANT else 0.8

        self.grid_ecm += bact_mask * dep_rate - deg_rate * self.grid_ecm
        self.grid_ecm = np.clip(self.grid_ecm, 0.0, 1.0)

    def _diffuse_and_react_drug(self, dt_seconds: float) -> None:
        """Antibiotic transport hindered by EPS matrix and degraded by KPC beta-lactamase."""
        d_eff_grid = self.diffusivity_drug_base / (1.0 + self.beta_biofilm * self.grid_ecm)
        max_d = float(np.max(d_eff_grid))
        cfl_dt = 0.18 * (self.dx * self.dy) / max(max_d, 1.0)
        num_substeps = max(5, min(30, int(math.ceil(dt_seconds / cfl_dt))))
        sub_dt = min(cfl_dt, dt_seconds / num_substeps)

        has_drug_a = self.applied_concentration_um > 0.0 or np.max(self.grid_drug_total) > 1e-6
        has_drug_b = self.applied_concentration_b_um > 0.0 or np.max(self.grid_drug_b_total) > 1e-6

        # Check if beta-lactamase inhibition is active (e.g. Avibactam in A or B)
        kpc_inhibited = bool(self.active_pharmacology.get("inhibits_kpc", False) or self.active_pharmacology_b.get("inhibits_kpc", False))

        # Build KPC hydrolysis sink grid
        kpc_hydrolysis_sink = np.zeros((self.ny, self.nx), dtype=np.float64)
        if not kpc_inhibited:
            for c in self.cells:
                if c.death_mode == CellDeathMode.ALIVE and "KPC_CARBAPENEMASE" in c.mutations:
                    gx = int(np.clip(c.x / self.dx, 0, self.nx - 1))
                    gy = int(np.clip(c.y / self.dy, 0, self.ny - 1))
                    kpc_hydrolysis_sink[gy, gx] += 0.0015

        d_mid_x = 0.5 * (d_eff_grid[:, :-1] + d_eff_grid[:, 1:])
        d_mid_y = 0.5 * (d_eff_grid[:-1, :] + d_eff_grid[1:, :])

        for _ in range(num_substeps):
            if has_drug_a:
                grad_x_a = (self.grid_drug_total[:, 1:] - self.grid_drug_total[:, :-1]) / self.dx
                grad_y_a = (self.grid_drug_total[1:, :] - self.grid_drug_total[:-1, :]) / self.dy
                flux_x_a = np.zeros_like(self.grid_drug_total)
                flux_y_a = np.zeros_like(self.grid_drug_total)
                flux_x_a[:, :-1] = d_mid_x * grad_x_a
                flux_y_a[:-1, :] = d_mid_y * grad_y_a
                div_flux_a = np.zeros_like(self.grid_drug_total)
                div_flux_a[:, :-1] += flux_x_a[:, :-1] / self.dx
                div_flux_a[:, 1:] -= flux_x_a[:, :-1] / self.dx
                div_flux_a[:-1, :] += flux_y_a[:-1, :] / self.dy
                div_flux_a[1:, :] -= flux_y_a[:-1, :] / self.dy

                vasc_influx_a = 0.03 * self.vessel_mask * (self.applied_concentration_um - self.grid_drug_total)
                clearance_a = self.clearance_drug_tissue * self.grid_drug_total
                hydrolysis_a = (kpc_hydrolysis_sink * self.grid_drug_total) if self.active_pharmacology.get("susceptible_to_kpc", False) else 0.0

                self.grid_drug_total += sub_dt * (div_flux_a + vasc_influx_a - clearance_a - hydrolysis_a)
                self.grid_drug_total = np.clip(self.grid_drug_total, 0.0, max(100.0, self.applied_concentration_um * 2.0))

            if has_drug_b:
                grad_x_b = (self.grid_drug_b_total[:, 1:] - self.grid_drug_b_total[:, :-1]) / self.dx
                grad_y_b = (self.grid_drug_b_total[1:, :] - self.grid_drug_b_total[:-1, :]) / self.dy
                flux_x_b = np.zeros_like(self.grid_drug_b_total)
                flux_y_b = np.zeros_like(self.grid_drug_b_total)
                flux_x_b[:, :-1] = d_mid_x * grad_x_b
                flux_y_b[:-1, :] = d_mid_y * grad_y_b
                div_flux_b = np.zeros_like(self.grid_drug_b_total)
                div_flux_b[:, :-1] += flux_x_b[:, :-1] / self.dx
                div_flux_b[:, 1:] -= flux_x_b[:, :-1] / self.dx
                div_flux_b[:-1, :] += flux_y_b[:-1, :] / self.dy
                div_flux_b[1:, :] -= flux_y_b[:-1, :] / self.dy

                vasc_influx_b = 0.03 * self.vessel_mask * (self.applied_concentration_b_um - self.grid_drug_b_total)
                clearance_b = self.clearance_drug_tissue * self.grid_drug_b_total
                hydrolysis_b = (kpc_hydrolysis_sink * self.grid_drug_b_total) if self.active_pharmacology_b.get("susceptible_to_kpc", False) else 0.0

                self.grid_drug_b_total += sub_dt * (div_flux_b + vasc_influx_b - clearance_b - hydrolysis_b)
                self.grid_drug_b_total = np.clip(self.grid_drug_b_total, 0.0, max(100.0, self.applied_concentration_b_um * 2.0))

        self.grid_drug_free = self.grid_drug_total * self.fraction_unbound_plasma
        self.grid_drug_b_free = self.grid_drug_b_total * self.fraction_unbound_b_plasma

    def _compute_laplacian(self, field: np.ndarray) -> np.ndarray:
        lap = np.zeros_like(field)
        lap[1:-1, 1:-1] = (
            (field[1:-1, 2:] + field[1:-1, :-2] - 2.0 * field[1:-1, 1:-1]) / (self.dx * self.dx)
            + (field[2:, 1:-1] + field[:-2, 1:-1] - 2.0 * field[1:-1, 1:-1]) / (self.dy * self.dy)
        )
        lap[0, 1:-1] = ((field[0, 2:] + field[0, :-2] - 2.0 * field[0, 1:-1]) / (self.dx * self.dx) + (field[1, 1:-1] - field[0, 1:-1]) / (self.dy * self.dy))
        lap[-1, 1:-1] = ((field[-1, 2:] + field[-1, :-2] - 2.0 * field[-1, 1:-1]) / (self.dx * self.dx) + (field[-2, 1:-1] - field[-1, 1:-1]) / (self.dy * self.dy))
        lap[1:-1, 0] = ((field[1:-1, 1] - field[1:-1, 0]) / (self.dx * self.dx) + (field[2:, 0] + field[:-2, 0] - 2.0 * field[1:-1, 0]) / (self.dy * self.dy))
        lap[1:-1, -1] = ((field[1:-1, -2] - field[1:-1, -1]) / (self.dx * self.dx) + (field[2:, -1] + field[:-2, -1] - 2.0 * field[1:-1, -1]) / (self.dy * self.dy))
        return lap

    def _update_bacteria_and_cells(self, dt_seconds: float) -> dict[str, int]:
        events = {"apoptosis": 0, "necrosis": 0, "division": 0, "mutations_emerged": 0}
        new_cells: list[CellularEntity] = []

        prof = self.active_pharmacology
        def _ext(p: dict, k: str, d: float) -> float:
            v = p.get(k, d)
            if isinstance(v, dict): v = v.get("value", d)
            try: return float(v) if v is not None and not isinstance(v, str) else d
            except: return d

        ic50_sens_a = _ext(prof, "ic50_sensitive_um", 0.25)
        ic50_res_a = _ext(prof, "ic50_resistant_um", 12.50)
        ic50_healthy_a = _ext(prof, "ic50_healthy_um", 50.0)
        hill_slope_a = _ext(prof, "hill_slope", self.hill_slope_default)
        emax_a = _ext(prof, "emax", self.emax_default)

        has_drug_b = self.applied_concentration_b_um > 0.0 or np.max(self.grid_drug_b_total) > 1e-6
        prof_b = self.active_pharmacology_b
        if has_drug_b:
            ic50_sens_b = _ext(prof_b, "ic50_sensitive_um", 0.50)
            ic50_res_b = _ext(prof_b, "ic50_resistant_um", 4.80)
            ic50_healthy_b = _ext(prof_b, "ic50_healthy_um", 30.0)
            hill_slope_b = _ext(prof_b, "hill_slope", self.hill_slope_default)
            emax_b = _ext(prof_b, "emax", self.emax_default)

        cell_coords = [(c.x, c.y) for c in self.cells if c.death_mode == CellDeathMode.ALIVE]

        for cell in self.cells:
            if cell.death_mode != CellDeathMode.ALIVE: continue

            gx = int(np.clip(cell.x / self.dx, 0, self.nx - 1))
            gy = int(np.clip(cell.y / self.dy, 0, self.ny - 1))

            cell.local_oxygen = float(self.grid_oxygen[gy, gx])
            cell.local_nutrient = float(self.grid_nutrient[gy, gx])
            cell.local_drug_conc = float(self.grid_drug_free[gy, gx])
            local_b = float(self.grid_drug_b_free[gy, gx]) if has_drug_b else 0.0

            # Intracellular accumulation
            k_perm = self.k_perm_default
            k_eff = self.k_out_default + self.k_efflux_pump * cell.efflux_pump_level
            d_intra_a = k_perm * cell.local_drug_conc - k_eff * cell.intracellular_drug
            equilibrium_a = k_perm * cell.local_drug_conc / max(1e-12, k_eff)
            cell.intracellular_drug = max(0.0, equilibrium_a + (cell.intracellular_drug - equilibrium_a) * math.exp(-k_eff * dt_seconds))
            cell.exposure_auc += cell.intracellular_drug * (dt_seconds / 3600.0)

            intra_b = getattr(cell, "intracellular_drug_b", 0.0)
            if has_drug_b:
                d_intra_b = k_perm * local_b - k_eff * intra_b
                equilibrium_b = k_perm * local_b / max(1e-12, k_eff)
                intra_b = max(0.0, equilibrium_b + (intra_b - equilibrium_b) * math.exp(-k_eff * dt_seconds))
            else:
                intra_b = 0.0
            setattr(cell, "intracellular_drug_b", intra_b)

            # Starvation / anoxia necrosis
            if cell.local_nutrient < 0.04:
                if self._rng.random() < (1.0 - math.exp(-0.00015 * dt_seconds)):
                    cell.death_mode = CellDeathMode.NECROSIS
                    cell.viability = 0.0
                    self.cumulative_necrosis_count += 1
                    events["necrosis"] += 1
                    continue

            # Antibiotic bactericidal kill (Hill / Bliss)
            e_a = 0.0
            if cell.intracellular_drug > 1e-5:
                eff_ic50_a = self._compute_bacterial_ic50(cell, ic50_sens_a, ic50_res_a, ic50_healthy_a)
                conc_a = cell.intracellular_drug
                e_a = emax_a * ((conc_a**hill_slope_a) / (eff_ic50_a**hill_slope_a + conc_a**hill_slope_a))

            e_b = 0.0
            if has_drug_b and intra_b > 1e-5:
                eff_ic50_b = self._compute_bacterial_ic50(cell, ic50_sens_b, ic50_res_b, ic50_healthy_b)
                conc_b = intra_b
                e_b = emax_b * ((conc_b**hill_slope_b) / (eff_ic50_b**hill_slope_b + conc_b**hill_slope_b))

            if e_a > 0.0 or e_b > 0.0:
                if has_drug_b and (e_a > 0.0 or e_b > 0.0):
                    e_comb = min(0.99, 1.0 - (1.0 - e_a) * (1.0 - e_b) * (1.0 / max(0.5, self.synergy_factor)))
                else:
                    e_comb = e_a

                p_kill = 1.0 - math.exp(-e_comb * dt_seconds)
                if self._rng.random() < p_kill:
                    cell.death_mode = CellDeathMode.APOPTOSIS
                    cell.viability = 0.0
                    self.cumulative_apoptosis_count += 1
                    events["apoptosis"] += 1
                    continue

            # Bacterial Division (Monod kinetics)
            if cell.cell_type in (CellType.TUMOR_SENSITIVE, CellType.TUMOR_RESISTANT):
                nbrs = sum(1 for ox, oy in cell_coords if math.hypot(ox - cell.x, oy - cell.y) < (cell.radius * 2.5))
                if nbrs < 9:
                    monod_nut = cell.local_nutrient / (0.15 + cell.local_nutrient)
                    inc = cell.growth_rate * monod_nut * (1.0 - nbrs / 11) * (dt_seconds / 3600.0)
                    cell.division_progress += max(0.0, inc)
                    if cell.division_progress >= 1.0:
                        cell.division_progress = 0.0
                        d = self._divide_bacteria(cell)
                        new_cells.append(d)
                        events["division"] += 1
                        if d.clone_id != cell.clone_id:
                            events["mutations_emerged"] += 1

        self.cells.extend(new_cells)
        return events

    def _compute_bacterial_ic50(self, cell: CellularEntity, sens_base: float, res_base: float, healthy_base: float) -> float:
        if cell.cell_type == CellType.EPITHELIAL_HEALTHY:
            return healthy_base
        if cell.cell_type == CellType.MACROPHAGE:
            return healthy_base * 1.5

        base = sens_base if cell.clone_id == "AMR-SUSCEPTIBLE" else res_base
        kpc_mult = 1.0
        if "KPC_CARBAPENEMASE" in cell.mutations: kpc_mult += 0.50
        if "OXA_48" in cell.mutations: kpc_mult += 0.35
        if "MEXAB_EFFLUX" in cell.mutations: kpc_mult += 0.40

        trait_mult = (1.0 + 1.2 * cell.resistance_score) * (1.0 + 0.4 * cell.efflux_pump_level)
        return max(0.01, float(base * kpc_mult * trait_mult))

    def _divide_bacteria(self, parent: CellularEntity) -> CellularEntity:
        dist = parent.radius * 1.6
        ang = self._rng.uniform(0.0, 2.0 * math.pi)
        nx_pos = np.clip(parent.x + dist * math.cos(ang), 15.0, self.width - 15.0)
        ny_pos = np.clip(parent.y + dist * math.sin(ang), 15.0, self.height - 15.0)

        muts = list(parent.mutations)
        res = float(parent.resistance_score)
        eff = float(parent.efflux_pump_level)
        clone = parent.clone_id

        # Spontaneous plasmid transfer / mutation
        if self._rng.random() < 0.035:
            res = min(1.0, res + self._rng.uniform(0.08, 0.18))
            eff = min(1.0, eff + self._rng.uniform(0.06, 0.15))
            if "KPC_CARBAPENEMASE" not in muts and res > 0.55:
                muts.append("KPC_CARBAPENEMASE")
                clone = "AMR-PERSISTER-EMERGENT"

        return CellularEntity(
            cell_id=f"{parent.cell_id}_b{self._rng.integers(1000, 9999)}",
            cell_type=parent.cell_type,
            clone_id=clone,
            x=float(nx_pos),
            y=float(ny_pos),
            radius=parent.radius,
            growth_rate=parent.growth_rate,
            mutations=muts,
            resistance_score=res,
            efflux_pump_level=eff,
            generation=parent.generation + 1,
            parent_clone_id=parent.clone_id,
        )

    def _update_phagocytosis(self, dt_seconds: float) -> None:
        phagocytes = [c for c in self.cells if c.death_mode == CellDeathMode.ALIVE and c.cell_type == CellType.MACROPHAGE]
        if not phagocytes: return

        for bact in self.cells:
            if bact.death_mode != CellDeathMode.ALIVE or bact.cell_type not in (CellType.TUMOR_SENSITIVE, CellType.TUMOR_RESISTANT):
                continue
            gx = int(np.clip(bact.x / self.dx, 0, self.nx - 1))
            gy = int(np.clip(bact.y / self.dy, 0, self.ny - 1))
            # Phagocytosis blocked by thick biofilm (EPS > 0.50)
            if self.grid_ecm[gy, gx] > 0.50: continue

            for p in phagocytes:
                if math.hypot(p.x - bact.x, p.y - bact.y) < 16.0:
                    if self._rng.random() < (1.0 - math.exp(-0.00006 * dt_seconds)):
                        bact.death_mode = CellDeathMode.APOPTOSIS
                        bact.viability = 0.0
                        self.cumulative_apoptosis_count += 1
                    break

    def get_snapshot(self) -> dict[str, Any]:
        type_counts: dict[str, int] = {}
        clone_counts: dict[str, int] = {}
        death_counts = {CellDeathMode.ALIVE.value: 0, CellDeathMode.APOPTOSIS.value: 0, CellDeathMode.NECROSIS.value: 0}
        viable_bact = 0
        resistant_bact = 0
        total_intra_drug = 0.0
        active_count = 0

        for c in self.cells:
            death_counts[c.death_mode.value] += 1
            if c.death_mode == CellDeathMode.ALIVE:
                active_count += 1
                type_counts[c.cell_type.value] = type_counts.get(c.cell_type.value, 0) + 1
                clone_counts[c.clone_id] = clone_counts.get(c.clone_id, 0) + 1
                total_intra_drug += c.intracellular_drug + getattr(c, "intracellular_drug_b", 0.0)
                if c.cell_type in (CellType.TUMOR_SENSITIVE, CellType.TUMOR_RESISTANT):
                    viable_bact += 1
                    if c.cell_type == CellType.TUMOR_RESISTANT or c.resistance_score > 0.45:
                        resistant_bact += 1

        total_grid = float(self.nx * self.ny)
        normoxic_pct = float(np.sum(self.grid_oxygen >= 0.30) / total_grid * 100.0)
        hypoxic_pct = float(np.sum((self.grid_oxygen < 0.30) & (self.grid_oxygen >= 0.08)) / total_grid * 100.0)
        necrotic_pct = float(np.sum(self.grid_oxygen < 0.08) / total_grid * 100.0)

        res_frac = float(resistant_bact / viable_bact) if viable_bact > 0 else 0.0
        mean_intra = float(total_intra_drug / active_count) if active_count > 0 else 0.0

        summary_metrics = {
            "total_cells": len(self.cells),
            "viable_cells": active_count,
            "viable_tumor_cells": viable_bact,   # viable bacteria count
            "resistant_fraction": round(res_frac, 4),
            "mean_intracellular_drug_um": round(mean_intra, 4),
            "mean_ecm_density": round(float(np.mean(self.grid_ecm)), 4),  # mean EPS biofilm density
        }

        total_free_drug = self.grid_drug_free + self.grid_drug_b_free

        return {
            "disease_id": self.disease_id,
            "display_name": self.display_name,
            "domain": {"width": self.width, "height": self.height, "nx": self.nx, "ny": self.ny, "dx": self.dx, "dy": self.dy},
            "step_count": self.step_count,
            "biological_hours": round(self.biological_seconds / 3600.0, 3),
            "biological_seconds": round(self.biological_seconds, 2),
            "cells": [c.to_render_dict() for c in self.cells],
            "vessels": self.vessels,
            "heatmaps": {
                "oxygen": np.round(self.grid_oxygen, 3).tolist(),
                "nutrient": np.round(self.grid_nutrient, 3).tolist(),
                "ecm_density": np.round(self.grid_ecm, 3).tolist(),  # EPS biofilm density
                "drug_free": np.round(total_free_drug, 4).tolist(),
                "drug_a_free": np.round(self.grid_drug_free, 4).tolist(),
                "drug_b_free": np.round(self.grid_drug_b_free, 4).tolist(),
            },
            "zones": {
                "normoxic_area_pct": round(normoxic_pct, 2),
                "hypoxic_area_pct": round(hypoxic_pct, 2),
                "necrotic_core_pct": round(necrotic_pct, 2),
            },
            "clone_distribution": clone_counts,
            "cell_counts_by_type": type_counts,
            "death_mode_counts": death_counts,
            "cumulative_deaths": {
                "apoptosis": self.cumulative_apoptosis_count,
                "necrosis": self.cumulative_necrosis_count,
            },
            "summary_metrics": summary_metrics,
            "metrics": summary_metrics,
            "active_compound": self.active_compound.to_dict() if self.active_compound else None,
            "active_compound_b": self.active_compound_b.to_dict() if self.active_compound_b else None,
            "epistemic_category": EpistemicCategory.MECHANISTIC_SIMULATION.value,
        }

    def get_biomarkers(self, systemic_state: dict[str, Any] | None = None) -> dict[str, Any]:
        snap = self.get_snapshot()
        viable_bact = snap["summary_metrics"]["viable_tumor_cells"]
        # Log10 CFU/mL surrogate
        log_cfu = round(math.log10(max(10.0, viable_bact * 8500.0)), 2)
        biofilm_idx = round(snap["summary_metrics"]["mean_ecm_density"], 3)

        biomarkers: dict[str, Any] = {
            "CFU_BURDEN_SURROGATE": ScientificValue(
                name="Bacterial Colony Forming Units (CFU)",
                value=log_cfu,
                unit="log10 CFU/mL",
                epistemic_category=EpistemicCategory.EXPLORATORY_MODEL,
                source="EVOCURE AMR Microbiological Model",
                confidence=0.88,
                in_applicability_domain=True,
                limitations="In silico estimate based on 2D microcolony agent count.",
            ).to_dict(),
            "BIOFILM_COVERAGE_INDEX": ScientificValue(
                name="Extracellular Polymeric Substance (EPS) Biofilm Score",
                value=biofilm_idx,
                unit="dimensionless [0-1]",
                epistemic_category=EpistemicCategory.MECHANISTIC_SIMULATION,
                source="2D Bacterial Matrix Secretion Solver",
            ).to_dict(),
            "CARBAPENEMASE_ACTIVITY_INDEX": ScientificValue(
                name="KPC/OXA Hydrolytic Activity Index",
                value=round(snap["summary_metrics"]["resistant_fraction"], 3),
                unit="index",
                epistemic_category=EpistemicCategory.MECHANISTIC_SIMULATION,
            ).to_dict(),
            "PERSISTER_FRACTION": ScientificValue(
                name="Persister Bacterial Fraction",
                value=snap["summary_metrics"]["resistant_fraction"],
                unit="fraction",
                epistemic_category=EpistemicCategory.MECHANISTIC_SIMULATION,
            ).to_dict(),
        }

        if systemic_state is None:
            biomarkers["HEPATIC_ALT"] = ScientificValue(
                name="Serum Alanine Aminotransferase (ALT)",
                value=None,
                unit="U/L",
                epistemic_category=EpistemicCategory.UNMODELED,
                source="None",
            ).to_dict()
        else:
            biomarkers["HEPATIC_ALT"] = ScientificValue(
                name="Serum ALT",
                value=systemic_state.get("alt_u_l", 26.0),
                unit="U/L",
                epistemic_category=EpistemicCategory.EXPLORATORY_MODEL,
            ).to_dict()

        return biomarkers

    def run_controlled_comparison(
        self,
        compound: CompoundRecord,
        pharmacology_profile: dict[str, Any],
        applied_concentration_um: float,
        duration_hours: float = 48.0,
        seed: int = 42,
    ) -> dict[str, Any]:
        ctrl = type(self)(seed=seed, width=self.width, height=self.height, nx=self.nx, ny=self.ny)
        ctrl.apply_compound_exposure(compound, pharmacology_profile, 0.0)

        trt = type(self)(seed=seed, width=self.width, height=self.height, nx=self.nx, ny=self.ny)
        trt.apply_compound_exposure(compound, pharmacology_profile, applied_concentration_um)

        def pt(m):
            s = m.get_snapshot()
            return {
                "viable_tumor_cells": s["summary_metrics"]["viable_tumor_cells"],
                "resistant_fraction": s["summary_metrics"]["resistant_fraction"],
                "apoptotic_deaths": s["cumulative_deaths"]["apoptosis"],
                "necrotic_deaths": s["cumulative_deaths"]["necrosis"],
            }

        times = [0.0]
        c_series = [pt(ctrl)]
        t_series = [pt(trt)]

        steps = int(math.ceil(duration_hours))
        for h in range(1, steps + 1):
            ctrl.step(3600.0)
            trt.step(3600.0)
            times.append(float(h))
            c_series.append(pt(ctrl))
            t_series.append(pt(trt))

        c_fin = max(1, c_series[-1]["viable_tumor_cells"])
        t_fin = t_series[-1]["viable_tumor_cells"]
        tgi = max(0.0, (1.0 - (t_fin / c_fin)) * 100.0)

        return {
            "disease_id": self.disease_id,
            "compound_id": compound.compound_id,
            "compound_name": compound.name,
            "applied_concentration_um": applied_concentration_um,
            "duration_hours": duration_hours,
            "seed": seed,
            "time_points_hours": times,
            "control_trajectory": {k: [p[k] for p in c_series] for k in c_series[0]},
            "treated_trajectory": {k: [p[k] for p in t_series] for k in t_series[0]},
            "deltas": {
                "initial_tumor_cells": c_series[0]["viable_tumor_cells"],
                "control_final_tumor_cells": c_fin,
                "treated_final_tumor_cells": t_fin,
                "tumor_growth_inhibition_pct": round(tgi, 2),
                "clonal_shift_delta": round(t_series[-1]["resistant_fraction"] - c_series[-1]["resistant_fraction"], 4),
                "apoptosis_induction_ratio": round(t_series[-1]["apoptotic_deaths"] / max(1, c_series[-1]["apoptotic_deaths"]), 2),
            },
        }

    def run_combination_comparison(
        self,
        compound_a: CompoundRecord,
        compound_b: CompoundRecord,
        pharma_a: dict[str, Any],
        pharma_b: dict[str, Any],
        conc_a_um: float,
        conc_b_um: float,
        duration_hours: float = 48.0,
        seed: int = 42,
        synergy_factor: float = 1.0,
    ) -> dict[str, Any]:
        mod_ctrl = type(self)(seed=seed, width=self.width, height=self.height, nx=self.nx, ny=self.ny)
        mod_ctrl.apply_compound_exposure(compound_a, pharma_a, 0.0)

        mod_a = type(self)(seed=seed, width=self.width, height=self.height, nx=self.nx, ny=self.ny)
        mod_a.apply_compound_exposure(compound_a, pharma_a, conc_a_um)

        mod_b = type(self)(seed=seed, width=self.width, height=self.height, nx=self.nx, ny=self.ny)
        mod_b.apply_compound_exposure(compound_b, pharma_b, conc_b_um)

        mod_combo = type(self)(seed=seed, width=self.width, height=self.height, nx=self.nx, ny=self.ny)
        mod_combo.apply_combination_exposure(compound_a, compound_b, pharma_a, pharma_b, conc_a_um, conc_b_um, synergy_factor=synergy_factor)

        def pt(m):
            s = m.get_snapshot()
            mucosa = sum(1 for c in m.cells if c.death_mode == CellDeathMode.ALIVE and c.cell_type == CellType.EPITHELIAL_HEALTHY)
            return {
                "viable_tumor_cells": s["summary_metrics"]["viable_tumor_cells"],
                "resistant_fraction": s["summary_metrics"]["resistant_fraction"],
                "apoptotic_deaths": s["cumulative_deaths"]["apoptosis"],
                "healthy_cells": mucosa,
            }

        times = [0.0]
        s_ctrl = [pt(mod_ctrl)]
        s_a = [pt(mod_a)]
        s_b = [pt(mod_b)]
        s_combo = [pt(mod_combo)]

        steps = int(math.ceil(duration_hours))
        for h in range(1, steps + 1):
            mod_ctrl.step(3600.0)
            mod_a.step(3600.0)
            mod_b.step(3600.0)
            mod_combo.step(3600.0)
            times.append(float(h))
            s_ctrl.append(pt(mod_ctrl))
            s_a.append(pt(mod_a))
            s_b.append(pt(mod_b))
            s_combo.append(pt(mod_combo))

        c_fin = max(1, s_ctrl[-1]["viable_tumor_cells"])
        a_fin = s_a[-1]["viable_tumor_cells"]
        b_fin = s_b[-1]["viable_tumor_cells"]
        cb_fin = s_combo[-1]["viable_tumor_cells"]

        tgi_a = max(0.0, (1.0 - (a_fin / c_fin)) * 100.0)
        tgi_b = max(0.0, (1.0 - (b_fin / c_fin)) * 100.0)
        tgi_combo = max(0.0, (1.0 - (cb_fin / c_fin)) * 100.0)

        bliss = tgi_a + tgi_b - (tgi_a * tgi_b / 100.0)
        excess = tgi_combo - bliss

        init_h = max(1, s_ctrl[0]["healthy_cells"])
        h_surv = {
            "control": round(s_ctrl[-1]["healthy_cells"] / init_h * 100.0, 1),
            "monotherapy_a": round(s_a[-1]["healthy_cells"] / init_h * 100.0, 1),
            "monotherapy_b": round(s_b[-1]["healthy_cells"] / init_h * 100.0, 1),
            "combination": round(s_combo[-1]["healthy_cells"] / init_h * 100.0, 1),
        }

        return {
            "disease_id": self.disease_id,
            "compound_a": {"id": compound_a.compound_id, "name": compound_a.name, "conc_um": conc_a_um},
            "compound_b": {"id": compound_b.compound_id, "name": compound_b.name, "conc_um": conc_b_um},
            "duration_hours": duration_hours,
            "seed": seed,
            "synergy_factor": synergy_factor,
            "time_points_hours": times,
            "trajectories": {
                "control": {k: [p[k] for p in s_ctrl] for k in s_ctrl[0]},
                "monotherapy_a": {k: [p[k] for p in s_a] for k in s_a[0]},
                "monotherapy_b": {k: [p[k] for p in s_b] for k in s_b[0]},
                "combination": {k: [p[k] for p in s_combo] for k in s_combo[0]},
            },
            "deltas": {
                "initial_tumor_cells": s_ctrl[0]["viable_tumor_cells"],
                "control_final_tumor_cells": c_fin,
                "monotherapy_a_final_tumor_cells": a_fin,
                "monotherapy_b_final_tumor_cells": b_fin,
                "combination_final_tumor_cells": cb_fin,
                "tgi_a_pct": round(tgi_a, 2),
                "tgi_b_pct": round(tgi_b, 2),
                "tgi_combo_pct": round(tgi_combo, 2),
                "bliss_expected_pct": round(bliss, 2),
                "excess_over_bliss": round(excess, 2),
                "synergy_classification": "Synergistic" if excess > 4.0 else ("Antagonistic" if excess < -4.0 else "Additive"),
                "healthy_cell_survival_pct": h_surv,
                "clonal_shift": {
                    "control": s_ctrl[-1]["resistant_fraction"],
                    "monotherapy_a": s_a[-1]["resistant_fraction"],
                    "monotherapy_b": s_b[-1]["resistant_fraction"],
                    "combination": s_combo[-1]["resistant_fraction"],
                },
            },
        }
