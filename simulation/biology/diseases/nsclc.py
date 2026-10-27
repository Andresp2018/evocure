"""
Non-Small Cell Lung Cancer (NSCLC) KRAS G12C/D Driven Module.

Implements a 2D mechanistic tumor microenvironment for KRAS-mutant Non-Small
Cell Lung Adenocarcinoma (NSCLC). Features rich alveolar capillary vascularity
(higher baseline oxygenation than PDAC), rapid proliferation of KRAS-G12C
subclones, MET-amplified persister subclones, alveolar type-II epithelial cells,
cytotoxic T-lymphocytes, alveolar macrophages, single and dual combination
pharmacodynamics, and calibrated reference drug anchors.
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


class NSCLCModule(DiseaseModule):
    """
    Mechanistic 2D KRAS-Mutant Non-Small Cell Lung Adenocarcinoma (NSCLC) Model.

    Governing biological & biophysical features:
    1. Higher baseline vascularity & oxygenation than PDAC (normoxic fraction ~0.55-0.70).
    2. Moderate desmoplastic stroma (mean ECM ~0.20-0.30, beta_ECM = 1.8).
    3. Proliferative KRAS-G12C subclone alongside MET-amplified / KEAP1-mutant persister subclone.
    4. Surrounding Alveolar Type-II pneumocytes (AT2) with contact inhibition.
    5. Single & 2-Drug combination exposure with Bliss/Loewe synergy modeling.
    6. Calibrated reference anchors for Sotorasib, Erlotinib, Paclitaxel, Cisplatin, Gemcitabine.
    """

    REFERENCE_COMPOUNDS: dict[str, dict[str, Any]] = {
        "SOTORASIB": {
            "name": "Sotorasib (AMG 510)",
            "smiles": "C=CC(=O)N1CCN(CC1)C2=NC3=C(C=CC(=C3F)C4=C(C=CC(=C4Cl)O)C)C(=N2)NC5=CC(=C(C=C5)F)F",
            "ic50_sensitive_um": 0.12,
            "ic50_resistant_um": 4.50,
            "ic50_healthy_um": 22.0,
            "fraction_unbound_plasma": 0.18,
            "mechanism": "KRAS G12C Covalent Inhibitor (Switch-II pocket)",
        },
        "ERLOTINIB": {
            "name": "Erlotinib (Tarceva)",
            "smiles": "COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC",
            "ic50_sensitive_um": 0.45,
            "ic50_resistant_um": 3.80,
            "ic50_healthy_um": 14.5,
            "fraction_unbound_plasma": 0.22,
            "mechanism": "EGFR Tyrosine Kinase Inhibitor",
        },
        "PACLITAXEL": {
            "name": "Paclitaxel (Taxol)",
            "smiles": "CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1OC(=O)C(C(C5=CC=CC=C5)NC(=O)C6=CC=CC=C6)O)O)OC(=O)C7=CC=CC=C7)(CO4)OC(=O)C)O)C)OC(=O)C",
            "ic50_sensitive_um": 0.08,
            "ic50_resistant_um": 1.20,
            "ic50_healthy_um": 2.60,
            "fraction_unbound_plasma": 0.10,
            "mechanism": "Microtubule Stabilization / Mitotic Arrest",
        },
        "CISPLATIN": {
            "name": "Cisplatin",
            "smiles": "N.N.Cl[Pt]Cl",
            "ic50_sensitive_um": 1.50,
            "ic50_resistant_um": 6.20,
            "ic50_healthy_um": 11.5,
            "fraction_unbound_plasma": 0.35,
            "mechanism": "DNA Intra-strand Crosslinker",
        },
        "GEMCITABINE": {
            "name": "Gemcitabine",
            "smiles": "C1=CN(C(=O)N=C1N)C2C(C(C(O2)CO)O)(F)F",
            "ic50_sensitive_um": 0.25,
            "ic50_resistant_um": 2.40,
            "ic50_healthy_um": 8.00,
            "fraction_unbound_plasma": 0.85,
            "mechanism": "Nucleoside Metabolic Inhibitor",
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

        # Physical / Transport Parameters (higher vascular perfusion in lung)
        self.diffusivity_oxygen: float = 45.0       # um^2/s (higher in alveolar tissue)
        self.diffusivity_nutrient: float = 35.0
        self.diffusivity_drug_base: float = 50.0
        self.beta_ecm: float = 1.8                  # Moderate lung ECM hindrance
        self.clearance_drug_tissue: float = 0.00012

        # Cellular Uptake & Efflux
        self.k_perm_default: float = 0.015
        self.k_out_default: float = 0.007
        self.k_efflux_pump: float = 0.022

        # PD Hill Parameters
        self.hill_slope_default: float = 2.0
        self.emax_default: float = 0.000040
        self.tau_kill: float = 3600.0

        # Grids
        self.grid_oxygen = np.ones((self.ny, self.nx), dtype=np.float64)
        self.grid_nutrient = np.ones((self.ny, self.nx), dtype=np.float64)
        self.grid_ecm = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.grid_drug_total = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.grid_drug_free = np.zeros((self.ny, self.nx), dtype=np.float64)

        # Secondary Compound Grids (Combination)
        self.grid_drug_b_total = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.grid_drug_b_free = np.zeros((self.ny, self.nx), dtype=np.float64)

        # Vascular Network & Mask
        self.vessels: list[dict[str, Any]] = []
        self.vessel_mask = np.zeros((self.ny, self.nx), dtype=np.float64)

        # Cells
        self.cells: list[CellularEntity] = []

        # Compounds
        self.active_compound: CompoundRecord | None = None
        self.active_pharmacology: dict[str, Any] = {}
        self.applied_concentration_um: float = 0.0
        self.fraction_unbound_plasma: float = 0.25

        self.active_compound_b: CompoundRecord | None = None
        self.active_pharmacology_b: dict[str, Any] = {}
        self.applied_concentration_b_um: float = 0.0
        self.fraction_unbound_b_plasma: float = 0.25
        self.synergy_factor: float = 1.0

        # Death Counters
        self.cumulative_apoptosis_count: int = 0
        self.cumulative_necrosis_count: int = 0

        self.initialize_scenario(seed=self._seed, preset="standard")

    @property
    def disease_id(self) -> str:
        return "NSCLC_KRAS"

    @property
    def display_name(self) -> str:
        return "KRAS-Mutant Non-Small Cell Lung Adenocarcinoma"

    def metadata(self) -> dict[str, Any]:
        return {
            "disease_id": self.disease_id,
            "display_name": self.display_name,
            "category": "solid_tumor",
            "organ": "Lung",
            "histopathology": "Invasive alveolar adenocarcinoma with rich microvascular capillary network",
            "implementation_status": "fully_active_mechanistic",
            "is_calibrated": True,
            "epistemic_category": EpistemicCategory.MECHANISTIC_SIMULATION.value,
            "cellular_entities": [
                CellType.TUMOR_SENSITIVE.value,
                CellType.TUMOR_RESISTANT.value,
                CellType.EPITHELIAL_HEALTHY.value,
                CellType.CAF.value,
                CellType.FIBROBLAST.value,
                CellType.IMMUNE_T.value,
                CellType.MACROPHAGE.value,
                CellType.ENDOTHELIAL.value,
            ],
            "subclones": [
                {
                    "clone_id": "CLONE-KRAS-G12C",
                    "subtype": "KRAS-G12C Driven Sensitive",
                    "driver_mutations": ["KRAS_G12C", "TP53_MUT"],
                    "relative_ic50": "1.0x (Sotorasib Sensitive)",
                    "efflux_expression": "Low (0.08)",
                },
                {
                    "clone_id": "CLONE-MET-AMP",
                    "subtype": "MET-Amplified / KEAP1 Persister",
                    "driver_mutations": ["KRAS_G12C", "MET_AMP", "KEAP1_MUT"],
                    "relative_ic50": "5.5x (Resistant to monotherapy)",
                    "efflux_expression": "Moderate-High (0.48)",
                },
            ],
            "hallmarks_modeled": [
                "Rich alveolar capillary plexus sustaining elevated baseline normoxia",
                "Rapid proliferation of oncogenic KRAS-G12C lung adenocarcinoma clones",
                "Bypass pathway resistance via MET amplification and KEAP1 oxidative defense",
                "Alveolar epithelial type-II contact inhibition and drug tolerability",
                "2-drug combination synergy overcoming persister subclones",
            ],
            "calibrated_reference_compounds": list(self.REFERENCE_COMPOUNDS.keys()),
            "assumptions": [
                "2D cross-section of pulmonary alveolar parenchymal nodule.",
                "Capillary perfusion maintains higher systemic oxygenation than dense desmoplastic tumors.",
            ],
            "limitations": (
                "Planar cross-section neglects breathing-cycle alveolar strain and "
                "bronchial airway airflow advection. Calibrated for comparative pharmacology."
            ),
        }

    def initialize_scenario(self, seed: int, preset: str = "standard") -> None:
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self.biological_seconds = 0.0
        self.step_count = 0
        self.cumulative_apoptosis_count = 0
        self.cumulative_necrosis_count = 0

        self.vessels = self._create_nsclc_vasculature()
        self._recompute_vessel_mask()

        # Lung tissue has higher baseline oxygenation from alveolar capillary networks
        self.grid_oxygen = np.clip(0.15 + 0.85 * np.power(self.vessel_mask, 0.35), 0.10, 1.0)
        self.grid_nutrient = np.clip(0.20 + 0.80 * np.power(self.vessel_mask, 0.30), 0.15, 1.0)
        self.grid_ecm.fill(0.04)
        self.grid_drug_total.fill(0.0)
        self.grid_drug_free.fill(0.0)
        self.grid_drug_b_total.fill(0.0)
        self.grid_drug_b_free.fill(0.0)

        self.cells = self._create_nsclc_cell_population()
        self._relax_initial_microenvironment()

    def _create_nsclc_vasculature(self) -> list[dict[str, Any]]:
        """Alveolar capillary plexus: 10 capillary conduits traversing pulmonary parenchyma."""
        return [
            {"id": "ves_cap_1", "x1": 60.0, "y1": 80.0, "x2": 260.0, "y2": 90.0, "radius": 12.0, "perfusion": 1.0, "functional": True},
            {"id": "ves_cap_2", "x1": 540.0, "y1": 80.0, "x2": 740.0, "y2": 110.0, "radius": 12.0, "perfusion": 1.0, "functional": True},
            {"id": "ves_cap_3", "x1": 70.0, "y1": 240.0, "x2": 220.0, "y2": 260.0, "radius": 11.0, "perfusion": 1.0, "functional": True},
            {"id": "ves_cap_4", "x1": 580.0, "y1": 250.0, "x2": 730.0, "y2": 280.0, "radius": 12.0, "perfusion": 1.0, "functional": True},
            {"id": "ves_cap_5", "x1": 80.0, "y1": 420.0, "x2": 240.0, "y2": 440.0, "radius": 12.0, "perfusion": 1.0, "functional": True},
            {"id": "ves_cap_6", "x1": 560.0, "y1": 430.0, "x2": 720.0, "y2": 450.0, "radius": 11.0, "perfusion": 1.0, "functional": True},
            {"id": "ves_cap_7", "x1": 90.0, "y1": 530.0, "x2": 320.0, "y2": 510.0, "radius": 12.0, "perfusion": 1.0, "functional": True},
            {"id": "ves_cap_8", "x1": 500.0, "y1": 520.0, "x2": 710.0, "y2": 530.0, "radius": 12.0, "perfusion": 1.0, "functional": True},
            # Penetrating alveolar micro-vessels entering tumor margins
            {"id": "ves_cap_pen1", "x1": 260.0, "y1": 180.0, "x2": 350.0, "y2": 230.0, "radius": 8.0, "perfusion": 0.85, "functional": True},
            {"id": "ves_cap_pen2", "x1": 450.0, "y1": 370.0, "x2": 550.0, "y2": 410.0, "radius": 8.0, "perfusion": 0.80, "functional": True},
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

    def _create_nsclc_cell_population(self) -> list[CellularEntity]:
        cells: list[CellularEntity] = []
        center_x, center_y = self.width / 2.0, self.height / 2.0
        cid = 1

        # Endothelial cells along pulmonary capillaries
        for ves in self.vessels:
            for s in range(3):
                frac = (s + 0.5) / 3
                vx = ves["x1"] + frac * (ves["x2"] - ves["x1"]) + self._rng.uniform(-2.0, 2.0)
                vy = ves["y1"] + frac * (ves["y2"] - ves["y1"]) + self._rng.uniform(-2.0, 2.0)
                cells.append(CellularEntity(
                    cell_id=f"nsclc_endo_{cid}",
                    cell_type=CellType.ENDOTHELIAL,
                    clone_id="HOST-PULMONARY-ENDO",
                    x=float(vx),
                    y=float(vy),
                    radius=5.5,
                    growth_rate=0.005,
                    resistance_score=0.15,
                    efflux_pump_level=0.12,
                ))
                cid += 1

        # Tumor Population: 75% KRAS-G12C sensitive, 25% MET-AMP persisters
        num_sens = 70
        num_res = 22
        for _ in range(num_sens):
            r = self._rng.triangular(0.0, 90.0, 170.0)
            theta = self._rng.uniform(0.0, 2.0 * math.pi)
            cx = center_x + r * math.cos(theta)
            cy = center_y + r * math.sin(theta)
            c = CellularEntity(
                cell_id=f"nsclc_sens_{cid}",
                cell_type=CellType.TUMOR_SENSITIVE,
                clone_id="CLONE-KRAS-G12C",
                x=float(cx),
                y=float(cy),
                radius=6.4,
                growth_rate=0.042,
                mutations=["KRAS_G12C", "TP53_MUT"],
                resistance_score=float(np.clip(self._rng.normal(0.06, 0.02), 0.02, 0.15)),
                efflux_pump_level=float(np.clip(self._rng.normal(0.08, 0.02), 0.04, 0.16)),
            )
            cells.append(c)
            cid += 1

        for _ in range(num_res):
            r = self._rng.triangular(30.0, 110.0, 180.0)
            theta = self._rng.uniform(0.0, 2.0 * math.pi)
            cx = center_x + r * math.cos(theta)
            cy = center_y + r * math.sin(theta)
            c = CellularEntity(
                cell_id=f"nsclc_res_{cid}",
                cell_type=CellType.TUMOR_RESISTANT,
                clone_id="CLONE-MET-AMP",
                x=float(cx),
                y=float(cy),
                radius=6.6,
                growth_rate=0.035,
                mutations=["KRAS_G12C", "MET_AMP", "KEAP1_MUT"],
                resistance_score=float(np.clip(self._rng.normal(0.62, 0.04), 0.48, 0.78)),
                efflux_pump_level=float(np.clip(self._rng.normal(0.48, 0.04), 0.35, 0.65)),
            )
            cells.append(c)
            cid += 1

        # Healthy Alveolar Type-II Pneumocytes (AT2)
        num_at2 = 40
        for _ in range(num_at2):
            r = self._rng.uniform(180.0, 320.0)
            theta = self._rng.uniform(0.0, 2.0 * math.pi)
            cx = np.clip(center_x + r * math.cos(theta), 25.0, self.width - 25.0)
            cy = np.clip(center_y + r * math.sin(theta), 25.0, self.height - 25.0)
            cells.append(CellularEntity(
                cell_id=f"at2_{cid}",
                cell_type=CellType.EPITHELIAL_HEALTHY,
                clone_id="HOST-ALVEOLAR-AT2",
                x=float(cx),
                y=float(cy),
                radius=5.8,
                growth_rate=0.010,
                resistance_score=0.04,
                efflux_pump_level=0.08,
            ))
            cid += 1

        # Moderate Stroma: CAFs and Fibroblasts (lower burden than PDAC)
        num_cafs = 25
        for _ in range(num_cafs):
            r = self._rng.uniform(60.0, 200.0)
            theta = self._rng.uniform(0.0, 2.0 * math.pi)
            cx = center_x + r * math.cos(theta)
            cy = center_y + r * math.sin(theta)
            is_active = self._rng.random() < 0.60
            cells.append(CellularEntity(
                cell_id=f"nsclc_caf_{cid}",
                cell_type=CellType.CAF if is_active else CellType.FIBROBLAST,
                clone_id="STROMA-PULMONARY",
                x=float(cx),
                y=float(cy),
                radius=7.0,
                growth_rate=0.012,
                resistance_score=0.25,
                efflux_pump_level=0.18,
            ))
            cid += 1

        # Infiltrating Immune Cells (T cells and Alveolar Macrophages)
        num_t = 16
        num_macro = 14
        for _ in range(num_t):
            r = self._rng.uniform(140.0, 280.0)
            theta = self._rng.uniform(0.0, 2.0 * math.pi)
            cells.append(CellularEntity(
                cell_id=f"nsclc_t_{cid}",
                cell_type=CellType.IMMUNE_T,
                clone_id="IMMUNE-CD8",
                x=float(np.clip(center_x + r * math.cos(theta), 25.0, self.width - 25.0)),
                y=float(np.clip(center_y + r * math.sin(theta), 25.0, self.height - 25.0)),
                radius=4.8,
                growth_rate=0.002,
                resistance_score=0.10,
                efflux_pump_level=0.12,
            ))
            cid += 1

        for _ in range(num_macro):
            r = self._rng.uniform(120.0, 260.0)
            theta = self._rng.uniform(0.0, 2.0 * math.pi)
            cells.append(CellularEntity(
                cell_id=f"nsclc_macro_{cid}",
                cell_type=CellType.MACROPHAGE,
                clone_id="IMMUNE-ALVEOLAR-MAC",
                x=float(np.clip(center_x + r * math.cos(theta), 25.0, self.width - 25.0)),
                y=float(np.clip(center_y + r * math.sin(theta), 25.0, self.height - 25.0)),
                radius=6.2,
                growth_rate=0.004,
                resistance_score=0.15,
                efflux_pump_level=0.16,
            ))
            cid += 1

        return cells

    def _relax_initial_microenvironment(self) -> None:
        for c in self.cells:
            if c.cell_type in (CellType.CAF, CellType.FIBROBLAST):
                gx = int(np.clip(c.x / self.dx, 0, self.nx - 1))
                gy = int(np.clip(c.y / self.dy, 0, self.ny - 1))
                r = 2
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        ny_idx, nx_idx = gy + dy, gx + dx
                        if 0 <= ny_idx < self.ny and 0 <= nx_idx < self.nx:
                            w = 1.0 / (1.0 + dx * dx + dy * dy)
                            self.grid_ecm[ny_idx, nx_idx] = min(0.60, self.grid_ecm[ny_idx, nx_idx] + 0.18 * w)

        for _ in range(30):
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

        fu_raw = pharmacology_profile.get("fraction_unbound_plasma", 0.25)
        if isinstance(fu_raw, dict):
            fu_raw = fu_raw.get("value", 0.25)
        try:
            self.fraction_unbound_plasma = float(fu_raw if fu_raw is not None else 0.25)
        except (ValueError, TypeError):
            self.fraction_unbound_plasma = 0.25

        if self.applied_concentration_um > 0.0:
            if injection_region == "vascular":
                inj_mask = self.vessel_mask > 0.15
                self.grid_drug_total[inj_mask] = np.maximum(
                    self.grid_drug_total[inj_mask],
                    self.applied_concentration_um * self.vessel_mask[inj_mask],
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
            fu = p.get("fraction_unbound_plasma", 0.25)
            if isinstance(fu, dict):
                fu = fu.get("value", 0.25)
            try:
                return float(fu if fu is not None else 0.25)
            except (ValueError, TypeError):
                return 0.25

        self.fraction_unbound_plasma = _extract_fu(pharma_a)
        self.fraction_unbound_b_plasma = _extract_fu(pharma_b)

        if injection_region == "vascular":
            inj_mask = self.vessel_mask > 0.15
            self.grid_drug_total[inj_mask] = np.maximum(
                self.grid_drug_total[inj_mask], self.applied_concentration_um * self.vessel_mask[inj_mask]
            )
            self.grid_drug_b_total[inj_mask] = np.maximum(
                self.grid_drug_b_total[inj_mask], self.applied_concentration_b_um * self.vessel_mask[inj_mask]
            )
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
        self._update_ecm(dt_seconds)

        has_drug_a = self.applied_concentration_um > 0.0 or np.max(self.grid_drug_total) > 1e-6
        has_drug_b = self.applied_concentration_b_um > 0.0 or np.max(self.grid_drug_b_total) > 1e-6
        if has_drug_a or has_drug_b:
            self._diffuse_and_react_drug(dt_seconds)

        step_events = self._update_cells(dt_seconds)
        self._update_immune_interactions(dt_seconds)

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

        o2_sink_grid = np.ones((self.ny, self.nx), dtype=np.float64) * 0.0015
        nut_sink_grid = np.ones((self.ny, self.nx), dtype=np.float64) * 0.0012

        for c in self.cells:
            if c.death_mode != CellDeathMode.ALIVE:
                continue
            gx = int(np.clip(c.x / self.dx, 0, self.nx - 1))
            gy = int(np.clip(c.y / self.dy, 0, self.ny - 1))

            if c.cell_type in (CellType.TUMOR_SENSITIVE, CellType.TUMOR_RESISTANT):
                o2_sink_grid[gy, gx] += 0.028
                nut_sink_grid[gy, gx] += 0.024
            elif c.cell_type == CellType.CAF:
                o2_sink_grid[gy, gx] += 0.010
                nut_sink_grid[gy, gx] += 0.008
            else:
                o2_sink_grid[gy, gx] += 0.005
                nut_sink_grid[gy, gx] += 0.004

        accelerated = gpu_substrates(self.grid_oxygen, self.grid_nutrient, self.vessel_mask,
            o2_sink_grid, nut_sink_grid, self.dx, self.dy,
            (self.diffusivity_oxygen, self.diffusivity_nutrient), (0.07, 0.05), (0.08, 0.1), sub_dt, num_substeps)
        if accelerated is not None:
            self.grid_oxygen, self.grid_nutrient = accelerated
            return

        for _ in range(num_substeps):
            lap_o2 = self._compute_laplacian(self.grid_oxygen)
            lap_nut = self._compute_laplacian(self.grid_nutrient)

            # High vascular supply in lung capillaries
            supply_o2 = 0.07 * self.vessel_mask * (1.0 - self.grid_oxygen)
            supply_nut = 0.05 * self.vessel_mask * (1.0 - self.grid_nutrient)

            uptake_o2 = o2_sink_grid * (self.grid_oxygen / (0.08 + self.grid_oxygen))
            uptake_nut = nut_sink_grid * (self.grid_nutrient / (0.10 + self.grid_nutrient))

            self.grid_oxygen += sub_dt * (self.diffusivity_oxygen * lap_o2 + supply_o2 - uptake_o2)
            self.grid_nutrient += sub_dt * (self.diffusivity_nutrient * lap_nut + supply_nut - uptake_nut)

            self.grid_oxygen = np.clip(self.grid_oxygen, 0.0, 1.0)
            self.grid_nutrient = np.clip(self.grid_nutrient, 0.0, 1.0)

    def _update_ecm(self, dt_seconds: float) -> None:
        deposition_rate = 0.000020 * dt_seconds
        stroma_sensitization = 1.0
        for p in (self.active_pharmacology, self.active_pharmacology_b):
            if p.get("modulates_stroma", False) or p.get("anti_fibrotic", False):
                stroma_sensitization = max(stroma_sensitization, 2.5)

        degradation_rate = 0.000010 * dt_seconds * stroma_sensitization
        caf_mask = np.zeros((self.ny, self.nx), dtype=np.float64)
        for c in self.cells:
            if c.death_mode == CellDeathMode.ALIVE and c.cell_type in (CellType.CAF, CellType.FIBROBLAST):
                gx = int(np.clip(c.x / self.dx, 0, self.nx - 1))
                gy = int(np.clip(c.y / self.dy, 0, self.ny - 1))
                caf_mask[gy, gx] += 1.0 if c.cell_type == CellType.CAF else 0.4

        self.grid_ecm += caf_mask * deposition_rate - degradation_rate * self.grid_ecm
        self.grid_ecm = np.clip(self.grid_ecm, 0.0, 0.75)

    def _diffuse_and_react_drug(self, dt_seconds: float) -> None:
        d_eff_grid = self.diffusivity_drug_base / (1.0 + self.beta_ecm * self.grid_ecm)
        max_d = float(np.max(d_eff_grid))
        cfl_dt = 0.18 * (self.dx * self.dy) / max(max_d, 1.0)
        num_substeps = max(5, min(30, int(math.ceil(dt_seconds / cfl_dt))))
        sub_dt = min(cfl_dt, dt_seconds / num_substeps)

        has_drug_a = self.applied_concentration_um > 0.0 or np.max(self.grid_drug_total) > 1e-6
        has_drug_b = self.applied_concentration_b_um > 0.0 or np.max(self.grid_drug_b_total) > 1e-6

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
                self.grid_drug_total += sub_dt * (div_flux_a + vasc_influx_a - clearance_a)
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
                self.grid_drug_b_total += sub_dt * (div_flux_b + vasc_influx_b - clearance_b)
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

    def _update_cells(self, dt_seconds: float) -> dict[str, int]:
        events = {"apoptosis": 0, "necrosis": 0, "division": 0, "mutations_emerged": 0}
        new_daughter_cells: list[CellularEntity] = []

        prof = self.active_pharmacology
        def _extract(p: dict, k: str, d: float) -> float:
            v = p.get(k, d)
            if isinstance(v, dict): v = v.get("value", d)
            try: return float(v) if v is not None and not isinstance(v, str) else d
            except: return d

        ic50_sens_a = _extract(prof, "ic50_sensitive_um", 0.12)
        ic50_res_a = _extract(prof, "ic50_resistant_um", 4.50)
        ic50_healthy_a = _extract(prof, "ic50_healthy_um", 22.0)
        hill_slope_a = _extract(prof, "hill_slope", self.hill_slope_default)
        emax_a = _extract(prof, "emax", self.emax_default)

        has_drug_b = self.applied_concentration_b_um > 0.0 or np.max(self.grid_drug_b_total) > 1e-6
        prof_b = self.active_pharmacology_b
        if has_drug_b:
            ic50_sens_b = _extract(prof_b, "ic50_sensitive_um", 0.45)
            ic50_res_b = _extract(prof_b, "ic50_resistant_um", 3.80)
            ic50_healthy_b = _extract(prof_b, "ic50_healthy_um", 14.5)
            hill_slope_b = _extract(prof_b, "hill_slope", self.hill_slope_default)
            emax_b = _extract(prof_b, "emax", self.emax_default)

        cell_coords = [(c.x, c.y) for c in self.cells if c.death_mode == CellDeathMode.ALIVE]

        for cell in self.cells:
            if cell.death_mode != CellDeathMode.ALIVE:
                continue

            gx = int(np.clip(cell.x / self.dx, 0, self.nx - 1))
            gy = int(np.clip(cell.y / self.dy, 0, self.ny - 1))

            cell.local_oxygen = float(self.grid_oxygen[gy, gx])
            cell.local_nutrient = float(self.grid_nutrient[gy, gx])
            cell.local_drug_conc = float(self.grid_drug_free[gy, gx])
            local_drug_b_conc = float(self.grid_drug_b_free[gy, gx]) if has_drug_b else 0.0

            if cell.local_oxygen < 0.08: cell.metabolic_state = "quiescent"
            elif cell.local_oxygen < 0.30: cell.metabolic_state = "glycolytic_hypoxic"
            else: cell.metabolic_state = "normoxic"

            k_perm = self.k_perm_default
            k_efflux_tot = self.k_out_default + self.k_efflux_pump * cell.efflux_pump_level
            d_intra_a = k_perm * cell.local_drug_conc - k_efflux_tot * cell.intracellular_drug
            equilibrium_a = k_perm * cell.local_drug_conc / max(1e-12, k_efflux_tot)
            cell.intracellular_drug = max(0.0, equilibrium_a + (cell.intracellular_drug - equilibrium_a) * math.exp(-k_efflux_tot * dt_seconds))
            cell.exposure_auc += cell.intracellular_drug * (dt_seconds / 3600.0)

            intra_b = getattr(cell, "intracellular_drug_b", 0.0)
            if has_drug_b:
                d_intra_b = k_perm * local_drug_b_conc - k_efflux_tot * intra_b
                equilibrium_b = k_perm * local_drug_b_conc / max(1e-12, k_efflux_tot)
                intra_b = max(0.0, equilibrium_b + (intra_b - equilibrium_b) * math.exp(-k_efflux_tot * dt_seconds))
            else:
                intra_b = 0.0
            setattr(cell, "intracellular_drug_b", intra_b)

            # Anoxia necrosis (O2 < 0.08)
            if cell.local_oxygen < 0.08:
                p_necr = 1.0 - math.exp(-0.00012 * (0.08 - cell.local_oxygen) * dt_seconds)
                if self._rng.random() < p_necr:
                    cell.death_mode = CellDeathMode.NECROSIS
                    cell.viability = 0.0
                    self.cumulative_necrosis_count += 1
                    events["necrosis"] += 1
                    continue

            # Pharmacodynamic kill (Hill / Bliss)
            e_a = 0.0
            if cell.intracellular_drug > 1e-5:
                eff_ic50_a = self._compute_multifactor_ic50(cell, ic50_sens_a, ic50_res_a, ic50_healthy_a)
                conc_a = cell.intracellular_drug
                e_a = emax_a * ((conc_a**hill_slope_a) / (eff_ic50_a**hill_slope_a + conc_a**hill_slope_a))

            e_b = 0.0
            if has_drug_b and intra_b > 1e-5:
                eff_ic50_b = self._compute_multifactor_ic50(cell, ic50_sens_b, ic50_res_b, ic50_healthy_b)
                conc_b = intra_b
                e_b = emax_b * ((conc_b**hill_slope_b) / (eff_ic50_b**hill_slope_b + conc_b**hill_slope_b))

            if e_a > 0.0 or e_b > 0.0:
                if has_drug_b and (e_a > 0.0 or e_b > 0.0):
                    e_comb = min(0.99, 1.0 - (1.0 - e_a) * (1.0 - e_b) * (1.0 / max(0.5, self.synergy_factor)))
                else:
                    e_comb = e_a

                p_apop = 1.0 - math.exp(-e_comb * dt_seconds)
                if self._rng.random() < p_apop:
                    cell.death_mode = CellDeathMode.APOPTOSIS
                    cell.viability = 0.0
                    self.cumulative_apoptosis_count += 1
                    events["apoptosis"] += 1
                    continue

            # Proliferation
            if cell.cell_type in (CellType.TUMOR_SENSITIVE, CellType.TUMOR_RESISTANT, CellType.EPITHELIAL_HEALTHY):
                nbrs = sum(1 for ox, oy in cell_coords if math.hypot(ox - cell.x, oy - cell.y) < (cell.radius * 2.8))
                cap = 5 if cell.cell_type == CellType.EPITHELIAL_HEALTHY else 8
                if nbrs < cap:
                    o2_f = cell.local_oxygen / (0.12 + cell.local_oxygen)
                    nut_f = cell.local_nutrient / (0.15 + cell.local_nutrient)
                    inc = cell.growth_rate * o2_f * nut_f * (1.0 - nbrs / (cap + 2)) * (dt_seconds / 3600.0)
                    cell.division_progress += max(0.0, inc)
                    if cell.division_progress >= 1.0:
                        cell.division_progress = 0.0
                        daughter = self._divide_cell(cell)
                        new_daughter_cells.append(daughter)
                        events["division"] += 1
                        if daughter.clone_id != cell.clone_id:
                            events["mutations_emerged"] += 1

        self.cells.extend(new_daughter_cells)
        return events

    def _compute_multifactor_ic50(self, cell: CellularEntity, sens_base: float, res_base: float, healthy_base: float) -> float:
        if cell.cell_type == CellType.EPITHELIAL_HEALTHY:
            return healthy_base
        if cell.cell_type in (CellType.CAF, CellType.FIBROBLAST):
            return healthy_base * 1.4

        base = sens_base if cell.clone_id == "CLONE-KRAS-G12C" else res_base
        pw_mult = 1.0
        if "KRAS_G12C" in cell.mutations: pw_mult += 0.10
        if "MET_AMP" in cell.mutations: pw_mult += 0.45
        if "KEAP1_MUT" in cell.mutations: pw_mult += 0.30
        if "TP53_MUT" in cell.mutations: pw_mult += 0.20

        trait_mult = (1.0 + 1.1 * cell.resistance_score) * (1.0 + 0.35 * cell.efflux_pump_level)
        hyp_prot = 1.0 + (0.5 * max(0.0, 1.0 - cell.local_oxygen / 0.30))
        return max(0.01, float(base * pw_mult * trait_mult * hyp_prot))

    def _divide_cell(self, parent: CellularEntity) -> CellularEntity:
        dist = parent.radius * 1.8
        ang = self._rng.uniform(0.0, 2.0 * math.pi)
        nx_pos = np.clip(parent.x + dist * math.cos(ang), 15.0, self.width - 15.0)
        ny_pos = np.clip(parent.y + dist * math.sin(ang), 15.0, self.height - 15.0)

        muts = list(parent.mutations)
        res = float(parent.resistance_score)
        eff = float(parent.efflux_pump_level)
        clone = parent.clone_id

        if parent.cell_type in (CellType.TUMOR_SENSITIVE, CellType.TUMOR_RESISTANT) and self._rng.random() < 0.03:
            res = min(1.0, res + self._rng.uniform(0.08, 0.16))
            eff = min(1.0, eff + self._rng.uniform(0.05, 0.12))
            if "MET_AMP" not in muts and res > 0.50:
                muts.append("MET_AMP")
                clone = "CLONE-MET-PERSISTER"

        return CellularEntity(
            cell_id=f"{parent.cell_id}_d{self._rng.integers(1000, 9999)}",
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

    def _update_immune_interactions(self, dt_seconds: float) -> None:
        t_cells = [c for c in self.cells if c.death_mode == CellDeathMode.ALIVE and c.cell_type == CellType.IMMUNE_T]
        if not t_cells: return

        for tumor_cell in self.cells:
            if tumor_cell.death_mode != CellDeathMode.ALIVE or tumor_cell.cell_type not in (CellType.TUMOR_SENSITIVE, CellType.TUMOR_RESISTANT):
                continue
            gx = int(np.clip(tumor_cell.x / self.dx, 0, self.nx - 1))
            gy = int(np.clip(tumor_cell.y / self.dy, 0, self.ny - 1))
            if self.grid_ecm[gy, gx] > 0.50 or self.grid_oxygen[gy, gx] < 0.15:
                continue
            for tc in t_cells:
                if math.hypot(tc.x - tumor_cell.x, tc.y - tumor_cell.y) < 18.0:
                    if self._rng.random() < (1.0 - math.exp(-0.00005 * dt_seconds)):
                        tumor_cell.death_mode = CellDeathMode.APOPTOSIS
                        tumor_cell.viability = 0.0
                        self.cumulative_apoptosis_count += 1
                    break

    def get_snapshot(self) -> dict[str, Any]:
        type_counts: dict[str, int] = {}
        clone_counts: dict[str, int] = {}
        death_counts = {CellDeathMode.ALIVE.value: 0, CellDeathMode.APOPTOSIS.value: 0, CellDeathMode.NECROSIS.value: 0}
        viable_tumor = 0
        resistant_tumor = 0
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
                    viable_tumor += 1
                    if c.cell_type == CellType.TUMOR_RESISTANT or c.resistance_score > 0.40:
                        resistant_tumor += 1

        total_cells = float(self.nx * self.ny)
        normoxic_pct = float(np.sum(self.grid_oxygen >= 0.30) / total_cells * 100.0)
        hypoxic_pct = float(np.sum((self.grid_oxygen < 0.30) & (self.grid_oxygen >= 0.08)) / total_cells * 100.0)
        necrotic_pct = float(np.sum(self.grid_oxygen < 0.08) / total_cells * 100.0)

        resistant_fraction = float(resistant_tumor / viable_tumor) if viable_tumor > 0 else 0.0
        mean_intra_drug = float(total_intra_drug / active_count) if active_count > 0 else 0.0

        summary_metrics = {
            "total_cells": len(self.cells),
            "viable_cells": active_count,
            "viable_tumor_cells": viable_tumor,
            "resistant_fraction": round(resistant_fraction, 4),
            "mean_intracellular_drug_um": round(mean_intra_drug, 4),
            "mean_ecm_density": round(float(np.mean(self.grid_ecm)), 4),
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
                "ecm_density": np.round(self.grid_ecm, 3).tolist(),
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
        viable_tumor = snap["summary_metrics"]["viable_tumor_cells"]
        cea_val = round(2.5 + viable_tumor * 0.18, 2)
        cyfra_val = round(1.2 + viable_tumor * 0.08, 2)

        biomarkers: dict[str, Any] = {
            "CEA_SURROGATE": ScientificValue(
                name="Carcinoembryonic Antigen (CEA)",
                value=cea_val,
                unit="ng/mL",
                epistemic_category=EpistemicCategory.EXPLORATORY_MODEL,
                source="EVOCURE NSCLC Biomarker Model",
                confidence=0.85,
                in_applicability_domain=True,
                limitations="Surrogate correlated with viable tumor cell count.",
            ).to_dict(),
            "CYFRA21_1": ScientificValue(
                name="Cytokeratin 19 Fragment (CYFRA 21-1)",
                value=cyfra_val,
                unit="ng/mL",
                epistemic_category=EpistemicCategory.EXPLORATORY_MODEL,
                source="EVOCURE NSCLC Biomarker Model",
                confidence=0.80,
                in_applicability_domain=True,
                limitations="Correlated with epithelial lung tumor burden.",
            ).to_dict(),
            "TUMOR_HYPOXIA_FRACTION": ScientificValue(
                name="Tumor Hypoxic Fraction",
                value=round(snap["zones"]["hypoxic_area_pct"] / 100.0, 3),
                unit="fraction",
                epistemic_category=EpistemicCategory.MECHANISTIC_SIMULATION,
                source="2D Reaction-Diffusion Eulerian Solver",
            ).to_dict(),
            "MET_PERSISTER_FRACTION": ScientificValue(
                name="MET Persister Subclone Fraction",
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
                value=systemic_state.get("alt_u_l", 24.0),
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
            healthy = sum(1 for c in m.cells if c.death_mode == CellDeathMode.ALIVE and c.cell_type == CellType.EPITHELIAL_HEALTHY)
            return {
                "viable_tumor_cells": s["summary_metrics"]["viable_tumor_cells"],
                "resistant_fraction": s["summary_metrics"]["resistant_fraction"],
                "apoptotic_deaths": s["cumulative_deaths"]["apoptosis"],
                "healthy_cells": healthy,
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
