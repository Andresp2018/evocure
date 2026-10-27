"""
Pancreatic Ductal Adenocarcinoma (PDAC) Mechanistic Microenvironment Module.

Implements the 2D tumor microenvironment for PDAC, incorporating subclonal
heterogeneity (classical sensitive vs basal resistant), desmoplastic stroma
(CAFs depositing collagenous ECM), hypovascular spatial architecture,
reaction-diffusion transport of oxygen, nutrients, and candidate compounds,
normoxic, hypoxic, and necrotic zonation, and Hill pharmacodynamics.
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


class PDACModule(DiseaseModule):
    """
    Mechanistic 2D Pancreatic Ductal Adenocarcinoma (PDAC) Tumor Microenvironment.

    Governing biological & biophysical features:
    1. Spatial Reaction-Diffusion (Eulerian 2D grids for O2, nutrients, ECM, drug).
    2. Desmoplasia: CAFs actively deposit ECM, physically attenuating drug diffusivity:
       D_eff(x) = D_0 / (1 + beta_ECM * ECM(x)).
    3. Spatial Hypovascularity: Vessels concentrated at periphery/invasive margin;
       hypovascular core produces a natural normoxic rim, hypoxic mantle (O2 < 0.30),
       and necrotic core (O2 < 0.08).
    4. Multi-factor Subclonal Sensitivity:
       - Classical Subclone: KRAS G12D, TP53 LOF (sensitive, low efflux).
       - Basal Subclone: KRAS G12D, TP53 LOF, CDKN2A DEL, SMAD4 LOF (resistant, high efflux).
       - Sensitivity is evaluated via multifactorial pathway + efflux + microenvironment state.
    5. Hill-equation Pharmacodynamics: Intracellular exposure drives apoptosis, while
       severe anoxia drives necrosis.
    """

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

        # Clocks & Mechanistic time
        self.biological_seconds: float = 0.0
        self.step_count: int = 0

        # Physical / Transport Parameters
        self.diffusivity_oxygen: float = 30.0       # um^2/s effective on scaled grid
        self.diffusivity_nutrient: float = 25.0     # um^2/s
        self.diffusivity_drug_base: float = 40.0    # um^2/s D0
        self.beta_ecm: float = 3.2                  # Desmoplastic hindrance coefficient
        self.clearance_drug_tissue: float = 0.00015 # 1/s tissue clearance rate

        # Cellular Uptake & Efflux Parameters
        self.k_perm_default: float = 0.012          # 1/s membrane permeation rate
        self.k_out_default: float = 0.006           # 1/s passive efflux rate
        self.k_efflux_pump: float = 0.024           # 1/s active pump-mediated efflux scale

        # PD Hill Parameters (defaults, overridden by compound profile if supplied)
        self.hill_slope_default: float = 2.0
        self.emax_default: float = 0.000035         # Max kill rate (1/s, calibrated for 24-72h regimens)
        self.tau_kill: float = 3600.0               # Scale factor for killing timescale

        # Continuous 2D Grids (ny, nx)
        self.grid_oxygen = np.ones((self.ny, self.nx), dtype=np.float64)
        self.grid_nutrient = np.ones((self.ny, self.nx), dtype=np.float64)
        self.grid_ecm = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.grid_drug_total = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.grid_drug_free = np.zeros((self.ny, self.nx), dtype=np.float64)

        # Vascular Network & Precomputed Perfusion Mask
        self.vessels: list[dict[str, Any]] = []
        self.vessel_mask = np.zeros((self.ny, self.nx), dtype=np.float64)

        # Cellular Entities
        self.cells: list[CellularEntity] = []

        # Current Compound & Pharmacology Context
        self.active_compound: CompoundRecord | None = None
        self.active_pharmacology: dict[str, Any] = {}
        self.applied_concentration_um: float = 0.0
        self.fraction_unbound_plasma: float = 0.25

        # Secondary Compound Context (2-Drug Combination)
        self.active_compound_b: CompoundRecord | None = None
        self.active_pharmacology_b: dict[str, Any] = {}
        self.applied_concentration_b_um: float = 0.0
        self.fraction_unbound_b_plasma: float = 0.25
        self.synergy_factor: float = 1.0
        self.grid_drug_b_total = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.grid_drug_b_free = np.zeros((self.ny, self.nx), dtype=np.float64)

        # Historical Death Counters
        self.cumulative_apoptosis_count: int = 0
        self.cumulative_necrosis_count: int = 0

        # Initialize default baseline scenario
        self.initialize_scenario(seed=self._seed, preset="standard")

    @property
    def disease_id(self) -> str:
        return "PDAC"

    @property
    def display_name(self) -> str:
        return "Pancreatic Ductal Adenocarcinoma"

    def metadata(self) -> dict[str, Any]:
        """Return biological context, entity types, biomarkers, assumptions, and limitations."""
        return {
            "disease_id": self.disease_id,
            "display_name": self.display_name,
            "organ": "Pancreas",
            "histopathology": "Infiltrating ductal adenocarcinoma with prominent desmoplastic stroma",
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
                    "clone_id": "CLONE-CLASSICAL",
                    "subtype": "Classical / Progenitor",
                    "driver_mutations": ["KRAS_G12D", "TP53_LOF"],
                    "relative_ic50": "1.0x (Sensitive baseline)",
                    "efflux_expression": "Low (baseline 0.10)",
                },
                {
                    "clone_id": "CLONE-BASAL",
                    "subtype": "Basal-like / Squamoid",
                    "driver_mutations": ["KRAS_G12D", "TP53_LOF", "CDKN2A_DEL", "SMAD4_LOF"],
                    "relative_ic50": "3.5x - 7.0x (Resistant)",
                    "efflux_expression": "High (baseline 0.55)",
                },
            ],
            "hallmarks_modeled": [
                "Desmoplastic stromal barrier with physically hindered drug diffusivity",
                "Spatial hypovascularity and peripheral vessel confinement",
                "Severe hypoxia gradient and central necrosis core (O2 < 0.08)",
                "Contact inhibition in healthy epithelial cells vs oncogenic override",
                "Multifactorial resistance (efflux + pathway alteration + microenvironment)",
                "Immune exclusion from dense fibrotic stroma and hypoxic regions",
            ],
            "epistemic_category": EpistemicCategory.MECHANISTIC_SIMULATION.value,
            "implementation_status": "fully_active_mechanistic",
            "assumptions": [
                "2D planar cross-section representing a representative microenvironmental slice of PDAC.",
                "Fickian diffusion with localized hindrance factor D_eff = D_0 / (1 + beta * ECM).",
                "Steady vascular influx at functional vessel coordinates.",
            ],
            "limitations": (
                "2D cross-section neglects out-of-plane capillary loops and convective interstitial "
                "fluid pressure (IFP) gradients. Useful for qualitative and comparative in silico "
                "regimen screening, not for single-patient prognostic clinical claims."
            ),
        }

    def initialize_scenario(self, seed: int, preset: str = "standard") -> None:
        """Initialize spatial microenvironment, cellular populations, and grids."""
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self.biological_seconds = 0.0
        self.step_count = 0
        self.cumulative_apoptosis_count = 0
        self.cumulative_necrosis_count = 0

        # Setup Vascular Architecture (Peripherally weighted vessels, typical of PDAC)
        self.vessels = self._create_pdac_vasculature()
        self._recompute_vessel_mask()

        # Reset Grids with physiological vascular gradient
        self.grid_oxygen = np.clip(0.04 + 0.96 * np.power(self.vessel_mask, 0.4), 0.02, 1.0)
        self.grid_nutrient = np.clip(0.08 + 0.92 * np.power(self.vessel_mask, 0.35), 0.05, 1.0)
        self.grid_ecm.fill(0.05)
        self.grid_drug_total.fill(0.0)
        self.grid_drug_free.fill(0.0)
        self.grid_drug_b_total.fill(0.0)
        self.grid_drug_b_free.fill(0.0)

        # Seed Cells
        self.cells = self._create_pdac_cell_population()

        # Initial relaxation of ECM and Fields around the initialized cells
        self._relax_initial_microenvironment()

    def _create_pdac_vasculature(self) -> list[dict[str, Any]]:
        """
        Creates hypovascular architecture typical of PDAC.
        Vessels skirt the peripheral margin of the tumor mass, leaving the center starved.
        """
        vessels = [
            # Top-left margin arteriole
            {"id": "ves_margin_tl", "x1": 80.0, "y1": 60.0, "x2": 260.0, "y2": 110.0, "radius": 14.0, "perfusion": 1.0, "functional": True},
            # Top-right margin arteriole
            {"id": "ves_margin_tr", "x1": 540.0, "y1": 100.0, "x2": 720.0, "y2": 70.0, "radius": 14.0, "perfusion": 1.0, "functional": True},
            # Right peripheral venule
            {"id": "ves_margin_r", "x1": 710.0, "y1": 200.0, "x2": 730.0, "y2": 440.0, "radius": 15.0, "perfusion": 1.0, "functional": True},
            # Bottom-right margin vessel
            {"id": "ves_margin_br", "x1": 560.0, "y1": 510.0, "x2": 710.0, "y2": 530.0, "radius": 13.0, "perfusion": 1.0, "functional": True},
            # Bottom-left margin vessel
            {"id": "ves_margin_bl", "x1": 90.0, "y1": 520.0, "x2": 280.0, "y2": 490.0, "radius": 14.0, "perfusion": 1.0, "functional": True},
            # Left peripheral vessel
            {"id": "ves_margin_l", "x1": 70.0, "y1": 180.0, "x2": 80.0, "y2": 420.0, "radius": 15.0, "perfusion": 1.0, "functional": True},
            # A single tortuous, partially compressed penetrating capillary near tumor periphery
            {"id": "ves_pen_margin", "x1": 250.0, "y1": 170.0, "x2": 310.0, "y2": 230.0, "radius": 9.0, "perfusion": 0.45, "functional": True},
        ]
        return vessels

    def _recompute_vessel_mask(self) -> None:
        """Rasterize vessel paths onto the 2D grid as a source density mask [0, 1]."""
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

            # Distance from cell centers to line segment (x1, y1) -> (x2, y2)
            dx = x2 - x1
            dy = y2 - y1
            seg_len_sq = dx * dx + dy * dy
            if seg_len_sq == 0:
                dist = np.hypot(cell_centers_x - x1, cell_centers_y - y1)
            else:
                t = ((cell_centers_x - x1) * dx + (cell_centers_y - y1) * dy) / seg_len_sq
                t = np.clip(t, 0.0, 1.0)
                proj_x = x1 + t * dx
                proj_y = y1 + t * dy
                dist = np.hypot(cell_centers_x - proj_x, cell_centers_y - proj_y)

            # Smooth Gaussian-like footprint within vessel radius
            mask = np.exp(-0.5 * (dist / max(rad, 1.0)) ** 2) * perf
            self.vessel_mask = np.maximum(self.vessel_mask, mask)

        self.vessel_mask = np.clip(self.vessel_mask, 0.0, 1.0)

    def _create_pdac_cell_population(self) -> list[CellularEntity]:
        """
        Creates an initial spatial population of tumor subclones, CAFs,
        healthy ductal epithelium, and infiltrating immune cells.
        Center: (400, 300) with radius ~180 um.
        """
        cells: list[CellularEntity] = []
        center_x, center_y = self.width / 2.0, self.height / 2.0
        cell_id_counter = 1

        # 1. Endothelial cells along vessels
        for idx, ves in enumerate(self.vessels):
            steps = 4
            for s in range(steps):
                frac = (s + 0.5) / steps
                vx = ves["x1"] + frac * (ves["x2"] - ves["x1"]) + self._rng.uniform(-3.0, 3.0)
                vy = ves["y1"] + frac * (ves["y2"] - ves["y1"]) + self._rng.uniform(-3.0, 3.0)
                c = CellularEntity(
                    cell_id=f"endo_{cell_id_counter}",
                    cell_type=CellType.ENDOTHELIAL,
                    clone_id="HOST-VESSEL",
                    x=float(vx),
                    y=float(vy),
                    radius=5.5,
                    growth_rate=0.005,
                    resistance_score=0.20,
                    efflux_pump_level=0.15,
                )
                cells.append(c)
                cell_id_counter += 1

        # 2. Dense central Tumor Population: mix of sensitive (classical) & resistant (basal)
        num_tumor_sensitive = 65
        num_tumor_resistant = 25

        for _ in range(num_tumor_sensitive):
            # Dense clustering in the core and inner mantle
            r = self._rng.triangular(0.0, 80.0, 150.0)
            theta = self._rng.uniform(0.0, 2.0 * math.pi)
            cx = center_x + r * math.cos(theta)
            cy = center_y + r * math.sin(theta)
            c = CellularEntity(
                cell_id=f"pdac_sens_{cell_id_counter}",
                cell_type=CellType.TUMOR_SENSITIVE,
                clone_id="CLONE-CLASSICAL",
                x=float(cx),
                y=float(cy),
                radius=6.5,
                growth_rate=0.032,
                mutations=["KRAS_G12D", "TP53_LOF"],
                resistance_score=float(self._rng.normal(0.08, 0.02)),
                efflux_pump_level=float(self._rng.normal(0.10, 0.02)),
            )
            c.resistance_score = float(np.clip(c.resistance_score, 0.02, 0.20))
            c.efflux_pump_level = float(np.clip(c.efflux_pump_level, 0.05, 0.20))
            cells.append(c)
            cell_id_counter += 1

        for _ in range(num_tumor_resistant):
            # Interspersed subclonal resistant pockets
            r = self._rng.triangular(20.0, 100.0, 160.0)
            theta = self._rng.uniform(0.0, 2.0 * math.pi)
            cx = center_x + r * math.cos(theta)
            cy = center_y + r * math.sin(theta)
            c = CellularEntity(
                cell_id=f"pdac_res_{cell_id_counter}",
                cell_type=CellType.TUMOR_RESISTANT,
                clone_id="CLONE-BASAL",
                x=float(cx),
                y=float(cy),
                radius=6.8,
                growth_rate=0.028,
                mutations=["KRAS_G12D", "TP53_LOF", "CDKN2A_DEL", "SMAD4_LOF"],
                resistance_score=float(self._rng.normal(0.58, 0.04)),
                efflux_pump_level=float(self._rng.normal(0.55, 0.04)),
            )
            c.resistance_score = float(np.clip(c.resistance_score, 0.45, 0.75))
            c.efflux_pump_level = float(np.clip(c.efflux_pump_level, 0.40, 0.75))
            cells.append(c)
            cell_id_counter += 1

        # 3. Cancer-Associated Fibroblasts (CAFs) and Fibroblasts (Dense desmoplastic ring)
        num_cafs = 40
        for _ in range(num_cafs):
            # Ring around and interspersed in the tumor
            r = self._rng.uniform(50.0, 210.0)
            theta = self._rng.uniform(0.0, 2.0 * math.pi)
            cx = center_x + r * math.cos(theta)
            cy = center_y + r * math.sin(theta)
            is_active_caf = self._rng.random() < 0.75
            ctype = CellType.CAF if is_active_caf else CellType.FIBROBLAST
            c = CellularEntity(
                cell_id=f"caf_{cell_id_counter}",
                cell_type=ctype,
                clone_id="STROMA-ACTIVATED" if is_active_caf else "STROMA-QUIESCENT",
                x=float(cx),
                y=float(cy),
                radius=7.5,
                growth_rate=0.015,
                resistance_score=0.35,
                efflux_pump_level=0.25,
            )
            cells.append(c)
            cell_id_counter += 1

        # 4. Healthy Pancreatic Ductal Epithelium (Surrounding host parenchyma)
        num_healthy = 35
        for _ in range(num_healthy):
            r = self._rng.uniform(190.0, 320.0)
            theta = self._rng.uniform(0.0, 2.0 * math.pi)
            cx = np.clip(center_x + r * math.cos(theta), 30.0, self.width - 30.0)
            cy = np.clip(center_y + r * math.sin(theta), 30.0, self.height - 30.0)
            c = CellularEntity(
                cell_id=f"healthy_{cell_id_counter}",
                cell_type=CellType.EPITHELIAL_HEALTHY,
                clone_id="HOST-DUCTAL",
                x=float(cx),
                y=float(cy),
                radius=6.0,
                growth_rate=0.008,
                resistance_score=0.05,
                efflux_pump_level=0.10,
            )
            cells.append(c)
            cell_id_counter += 1

        # 5. Infiltrating Immune Cells (T cells and Macrophages, largely held at margin)
        num_t_cells = 15
        num_macrophages = 12

        for _ in range(num_t_cells):
            r = self._rng.uniform(160.0, 300.0)
            theta = self._rng.uniform(0.0, 2.0 * math.pi)
            cx = np.clip(center_x + r * math.cos(theta), 25.0, self.width - 25.0)
            cy = np.clip(center_y + r * math.sin(theta), 25.0, self.height - 25.0)
            c = CellularEntity(
                cell_id=f"tcell_{cell_id_counter}",
                cell_type=CellType.IMMUNE_T,
                clone_id="IMMUNE-CD8",
                x=float(cx),
                y=float(cy),
                radius=4.8,
                growth_rate=0.002,
                resistance_score=0.12,
                efflux_pump_level=0.15,
            )
            cells.append(c)
            cell_id_counter += 1

        for _ in range(num_macrophages):
            r = self._rng.uniform(130.0, 270.0)
            theta = self._rng.uniform(0.0, 2.0 * math.pi)
            cx = np.clip(center_x + r * math.cos(theta), 25.0, self.width - 25.0)
            cy = np.clip(center_y + r * math.sin(theta), 25.0, self.height - 25.0)
            c = CellularEntity(
                cell_id=f"macro_{cell_id_counter}",
                cell_type=CellType.MACROPHAGE,
                clone_id="IMMUNE-TAM",
                x=float(cx),
                y=float(cy),
                radius=6.2,
                growth_rate=0.004,
                resistance_score=0.18,
                efflux_pump_level=0.20,
            )
            cells.append(c)
            cell_id_counter += 1

        return cells

    def _relax_initial_microenvironment(self) -> None:
        """Deposit baseline ECM near CAFs and settle the initial oxygen gradient."""
        for c in self.cells:
            if c.cell_type in (CellType.CAF, CellType.FIBROBLAST):
                gx = int(np.clip(c.x / self.dx, 0, self.nx - 1))
                gy = int(np.clip(c.y / self.dy, 0, self.ny - 1))
                # Local deposition footprint
                r = 2
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        ny_idx = gy + dy
                        nx_idx = gx + dx
                        if 0 <= ny_idx < self.ny and 0 <= nx_idx < self.nx:
                            weight = 1.0 / (1.0 + dx * dx + dy * dy)
                            self.grid_ecm[ny_idx, nx_idx] = min(
                                1.0, self.grid_ecm[ny_idx, nx_idx] + 0.35 * weight
                            )

        # Run 40 pseudo-steps of reaction-diffusion to form realistic baseline hypoxia
        for _ in range(40):
            self._diffuse_and_react_substrates(dt_seconds=60.0)

    def apply_compound_exposure(
        self,
        compound: CompoundRecord,
        pharmacology_profile: dict[str, Any],
        applied_concentration_um: float,
        injection_region: str = "vascular",
    ) -> dict[str, Any]:
        """
        Introduce a candidate compound into the microenvironment.
        Vascular administration introduces drug proportional to the vessel mask.
        """
        self.active_compound = compound
        self.active_pharmacology = copy.deepcopy(pharmacology_profile)
        self.applied_concentration_um = max(0.0, float(applied_concentration_um))
        fu_raw = pharmacology_profile.get("fraction_unbound_plasma", 0.25)
        if isinstance(fu_raw, dict):
            fu_raw = fu_raw.get("value", 0.25)
        try:
            self.fraction_unbound_plasma = float(fu_raw if fu_raw is not None else 0.25)
        except (ValueError, TypeError):
            self.fraction_unbound_plasma = 0.25

        # Reset secondary compound
        self.active_compound_b = None
        self.active_pharmacology_b = {}
        self.applied_concentration_b_um = 0.0
        self.synergy_factor = 1.0
        self.grid_drug_b_total.fill(0.0)
        self.grid_drug_b_free.fill(0.0)

        # Baseline injection: vascular injection drives boundary/vessel concentration
        if self.applied_concentration_um > 0.0:
            if injection_region == "vascular":
                injection_mask = self.vessel_mask > 0.15
                self.grid_drug_total[injection_mask] = np.maximum(
                    self.grid_drug_total[injection_mask],
                    self.applied_concentration_um * self.vessel_mask[injection_mask],
                )
            elif injection_region == "uniform":
                self.grid_drug_total.fill(self.applied_concentration_um)
            elif injection_region == "topical_interstitial":
                # Interstitial deposit at tumor boundary
                self.grid_drug_total[0:4, :] = self.applied_concentration_um

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
        """
        Introduce a 2-drug combination into the microenvironment.
        Models dual drug reaction-diffusion, combined Bliss/Loewe pharmacodynamics,
        stromal modulation/sensitization, and combined off-target stress.
        """
        self.active_compound = compound_a
        self.active_compound_b = compound_b
        self.active_pharmacology = copy.deepcopy(pharma_a)
        self.active_pharmacology_b = copy.deepcopy(pharma_b)
        self.applied_concentration_um = max(0.0, float(conc_a_um))
        self.applied_concentration_b_um = max(0.0, float(conc_b_um))
        self.synergy_factor = max(0.2, float(synergy_factor))

        def _extract_fu(prof: dict[str, Any]) -> float:
            fu = prof.get("fraction_unbound_plasma", 0.25)
            if isinstance(fu, dict):
                fu = fu.get("value", 0.25)
            try:
                return float(fu if fu is not None else 0.25)
            except (ValueError, TypeError):
                return 0.25

        self.fraction_unbound_plasma = _extract_fu(pharma_a)
        self.fraction_unbound_b_plasma = _extract_fu(pharma_b)

        # Baseline injection for both compounds
        if injection_region == "vascular":
            injection_mask = self.vessel_mask > 0.15
            self.grid_drug_total[injection_mask] = np.maximum(
                self.grid_drug_total[injection_mask],
                self.applied_concentration_um * self.vessel_mask[injection_mask],
            )
            self.grid_drug_b_total[injection_mask] = np.maximum(
                self.grid_drug_b_total[injection_mask],
                self.applied_concentration_b_um * self.vessel_mask[injection_mask],
            )
        elif injection_region == "uniform":
            self.grid_drug_total.fill(self.applied_concentration_um)
            self.grid_drug_b_total.fill(self.applied_concentration_b_um)
        elif injection_region == "topical_interstitial":
            self.grid_drug_total[0:4, :] = self.applied_concentration_um
            self.grid_drug_b_total[0:4, :] = self.applied_concentration_b_um

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
        """
        Advance cellular, reaction-diffusion, and pharmacodynamic state by dt_seconds.
        """
        dt_seconds = max(0.1, float(dt_seconds))
        self.biological_seconds += dt_seconds
        self.step_count += 1

        # 1. Reaction-Diffusion of Oxygen and Nutrients
        self._diffuse_and_react_substrates(dt_seconds)

        # 2. Desmoplastic Deposition & Degradation of ECM
        self._update_ecm(dt_seconds)

        # 3. Transport and Diffusion of Candidate Drug(s)
        has_drug_a = self.applied_concentration_um > 0.0 or np.max(self.grid_drug_total) > 1e-6
        has_drug_b = self.applied_concentration_b_um > 0.0 or np.max(self.grid_drug_b_total) > 1e-6
        if has_drug_a or has_drug_b:
            self._diffuse_and_react_drug(dt_seconds)

        # 4. Cell Mechanics: Sampling, Uptake, Pharmacodynamics, Apoptosis, Necrosis, Division
        step_events = self._update_cells(dt_seconds)

        # 5. Immune surveillance & clearance
        self._update_immune_interactions(dt_seconds)

        return {
            "step_count": self.step_count,
            "biological_seconds": round(self.biological_seconds, 2),
            "biological_hours": round(self.biological_seconds / 3600.0, 3),
            "events": step_events,
            "active_cells_count": len([c for c in self.cells if c.death_mode == CellDeathMode.ALIVE]),
        }

    def _diffuse_and_react_substrates(self, dt_seconds: float) -> None:
        """
        Eulerian finite-difference reaction diffusion for oxygen & nutrients.
        Sub-steps ensure numerical stability under explicit CFL criterion.
        """
        max_d = max(self.diffusivity_oxygen, self.diffusivity_nutrient)
        cfl_dt = 0.18 * (self.dx * self.dy) / max(max_d, 1.0)
        # Cap sub-steps between 5 and 30 for stable quasi-steady relaxation
        num_substeps = max(5, min(30, int(math.ceil(dt_seconds / cfl_dt))))
        sub_dt = min(cfl_dt, dt_seconds / num_substeps)

        # Rasterize cell sinks for oxygen and nutrient with basal tissue metabolism
        o2_sink_grid = np.ones((self.ny, self.nx), dtype=np.float64) * 0.002
        nut_sink_grid = np.ones((self.ny, self.nx), dtype=np.float64) * 0.0015

        for c in self.cells:
            if c.death_mode != CellDeathMode.ALIVE:
                continue
            gx = int(np.clip(c.x / self.dx, 0, self.nx - 1))
            gy = int(np.clip(c.y / self.dy, 0, self.ny - 1))

            # Consumption weights based on cell type
            if c.cell_type in (CellType.TUMOR_SENSITIVE, CellType.TUMOR_RESISTANT):
                o2_rate = 0.035  # Warburg & high proliferative respiration
                nut_rate = 0.028
            elif c.cell_type == CellType.CAF:
                o2_rate = 0.015
                nut_rate = 0.012
            else:
                o2_rate = 0.008
                nut_rate = 0.006

            o2_sink_grid[gy, gx] += o2_rate
            nut_sink_grid[gy, gx] += nut_rate

        accelerated = gpu_substrates(self.grid_oxygen, self.grid_nutrient, self.vessel_mask,
            o2_sink_grid, nut_sink_grid, self.dx, self.dy,
            (self.diffusivity_oxygen, self.diffusivity_nutrient), (0.05, 0.04), (0.1, 0.12), sub_dt, num_substeps)
        if accelerated is not None:
            self.grid_oxygen, self.grid_nutrient = accelerated
            return

        for _ in range(num_substeps):
            # Oxygen & Nutrient Laplacian
            lap_o2 = self._compute_laplacian(self.grid_oxygen)
            lap_nut = self._compute_laplacian(self.grid_nutrient)

            # Vascular Supply: vessels replenish substrates toward 1.0
            supply_o2 = 0.05 * self.vessel_mask * (1.0 - self.grid_oxygen)
            supply_nut = 0.04 * self.vessel_mask * (1.0 - self.grid_nutrient)

            # Michaelis-Menten cellular uptake
            uptake_o2 = o2_sink_grid * (self.grid_oxygen / (0.10 + self.grid_oxygen))
            uptake_nut = nut_sink_grid * (self.grid_nutrient / (0.12 + self.grid_nutrient))

            # Forward Euler step
            self.grid_oxygen += sub_dt * (self.diffusivity_oxygen * lap_o2 + supply_o2 - uptake_o2)
            self.grid_nutrient += sub_dt * (self.diffusivity_nutrient * lap_nut + supply_nut - uptake_nut)

            # Strict physical bounds [0, 1]
            self.grid_oxygen = np.clip(self.grid_oxygen, 0.0, 1.0)
            self.grid_nutrient = np.clip(self.grid_nutrient, 0.0, 1.0)

    def _update_ecm(self, dt_seconds: float) -> None:
        """
        CAFs actively deposit ECM matrix; basal turnover slowly degrades it.
        """
        # Check if either active drug modulates/sensitizes stroma (e.g. PEGPH20 or TGF-beta inhibitor)
        stroma_sensitization = 1.0
        for p in (self.active_pharmacology, self.active_pharmacology_b):
            if p.get("modulates_stroma", False) or p.get("anti_fibrotic", False) or p.get("degrades_ecm", False):
                stroma_sensitization = max(stroma_sensitization, 3.0)

        deposition_rate = 0.000035 * dt_seconds
        degradation_rate = 0.000008 * dt_seconds * stroma_sensitization

        caf_mask = np.zeros((self.ny, self.nx), dtype=np.float64)
        for c in self.cells:
            if c.death_mode == CellDeathMode.ALIVE and c.cell_type in (CellType.CAF, CellType.FIBROBLAST):
                gx = int(np.clip(c.x / self.dx, 0, self.nx - 1))
                gy = int(np.clip(c.y / self.dy, 0, self.ny - 1))
                mult = 1.0 if c.cell_type == CellType.CAF else 0.4
                caf_mask[gy, gx] += mult

        self.grid_ecm += caf_mask * deposition_rate - degradation_rate * self.grid_ecm
        self.grid_ecm = np.clip(self.grid_ecm, 0.0, 1.0)

    def _diffuse_and_react_drug(self, dt_seconds: float) -> None:
        """
        Reaction-diffusion of candidate compound(s) in desmoplastic tissue.
        Local effective diffusivity is hindered by ECM:
        D_eff(y, x) = D_base / (1 + beta_ecm * ECM(y, x)).
        """
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
                vasc_influx_a = 0.02 * self.vessel_mask * (self.applied_concentration_um - self.grid_drug_total)
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
                vasc_influx_b = 0.02 * self.vessel_mask * (self.applied_concentration_b_um - self.grid_drug_b_total)
                clearance_b = self.clearance_drug_tissue * self.grid_drug_b_total
                self.grid_drug_b_total += sub_dt * (div_flux_b + vasc_influx_b - clearance_b)
                self.grid_drug_b_total = np.clip(self.grid_drug_b_total, 0.0, max(100.0, self.applied_concentration_b_um * 2.0))

        self.grid_drug_free = self.grid_drug_total * self.fraction_unbound_plasma
        self.grid_drug_b_free = self.grid_drug_b_total * self.fraction_unbound_b_plasma

    def _compute_laplacian(self, field: np.ndarray) -> np.ndarray:
        """Finite-difference 5-point discrete Laplacian with zero-flux Neumann boundaries."""
        lap = np.zeros_like(field)

        # Interior
        lap[1:-1, 1:-1] = (
            (field[1:-1, 2:] + field[1:-1, :-2] - 2.0 * field[1:-1, 1:-1]) / (self.dx * self.dx)
            + (field[2:, 1:-1] + field[:-2, 1:-1] - 2.0 * field[1:-1, 1:-1]) / (self.dy * self.dy)
        )

        # Neumann Boundaries (zero normal gradient: duplicate edge for ghost cell)
        # Top row
        lap[0, 1:-1] = (
            (field[0, 2:] + field[0, :-2] - 2.0 * field[0, 1:-1]) / (self.dx * self.dx)
            + (field[1, 1:-1] - field[0, 1:-1]) / (self.dy * self.dy)
        )
        # Bottom row
        lap[-1, 1:-1] = (
            (field[-1, 2:] + field[-1, :-2] - 2.0 * field[-1, 1:-1]) / (self.dx * self.dx)
            + (field[-2, 1:-1] - field[-1, 1:-1]) / (self.dy * self.dy)
        )
        # Left column
        lap[1:-1, 0] = (
            (field[1:-1, 1] - field[1:-1, 0]) / (self.dx * self.dx)
            + (field[2:, 0] + field[:-2, 0] - 2.0 * field[1:-1, 0]) / (self.dy * self.dy)
        )
        # Right column
        lap[1:-1, -1] = (
            (field[1:-1, -2] - field[1:-1, -1]) / (self.dx * self.dx)
            + (field[2:, -1] + field[:-2, -1] - 2.0 * field[1:-1, -1]) / (self.dy * self.dy)
        )

        return lap

    def _update_cells(self, dt_seconds: float) -> dict[str, int]:
        """
        Updates cellular states:
        - Samples local continuous fields (O2, nutrient, free drug).
        - Computes intracellular drug accumulation with efflux modulation.
        - Computes multi-factor subclonal IC50 and Hill kill probability.
        - Triggers apoptosis and anoxia-driven necrosis.
        - Evaluates contact-inhibited division and emergence of resistant variants.
        """
        events = {"apoptosis": 0, "necrosis": 0, "division": 0, "mutations_emerged": 0}
        new_daughter_cells: list[CellularEntity] = []

        # Read pharmacology parameters if provided (handling both raw numbers and ScientificValue dicts)
        prof = self.active_pharmacology

        def _extract_param(key: str, default: float) -> float:
            v = prof.get(key, default)
            if isinstance(v, dict):
                v = v.get("value", default)
            try:
                return float(v) if v is not None and not isinstance(v, str) else default
            except (ValueError, TypeError):
                return default

        ic50_sens_base = _extract_param("ic50_sensitive_um", 0.45)
        ic50_res_base = _extract_param("ic50_resistant_um", 2.80)
        ic50_healthy_base = _extract_param("ic50_healthy_um", 8.50)
        hill_slope = _extract_param("hill_slope", self.hill_slope_default)
        emax = _extract_param("emax", self.emax_default)

        # Drug B pharmacology parameters (if combination active)
        prof_b = self.active_pharmacology_b
        has_drug_b = self.applied_concentration_b_um > 0.0 or np.max(self.grid_drug_b_total) > 1e-6
        if has_drug_b:
            def _extract_param_b(key: str, default: float) -> float:
                v = prof_b.get(key, default)
                if isinstance(v, dict):
                    v = v.get("value", default)
                try:
                    return float(v) if v is not None and not isinstance(v, str) else default
                except (ValueError, TypeError):
                    return default

            ic50_sens_b = _extract_param_b("ic50_sensitive_um", 0.60)
            ic50_res_b = _extract_param_b("ic50_resistant_um", 3.20)
            ic50_healthy_b = _extract_param_b("ic50_healthy_um", 9.00)
            hill_slope_b = _extract_param_b("hill_slope", self.hill_slope_default)
            emax_b = _extract_param_b("emax", self.emax_default)

        # Spatial density lookup for contact inhibition
        cell_coords = [(c.x, c.y) for c in self.cells if c.death_mode == CellDeathMode.ALIVE]

        for cell in self.cells:
            if cell.death_mode != CellDeathMode.ALIVE:
                continue

            # Grid sampling
            gx = int(np.clip(cell.x / self.dx, 0, self.nx - 1))
            gy = int(np.clip(cell.y / self.dy, 0, self.ny - 1))

            cell.local_oxygen = float(self.grid_oxygen[gy, gx])
            cell.local_nutrient = float(self.grid_nutrient[gy, gx])
            cell.local_drug_conc = float(self.grid_drug_free[gy, gx])
            local_drug_b_conc = float(self.grid_drug_b_free[gy, gx]) if has_drug_b else 0.0

            # Update metabolic state
            if cell.local_oxygen < 0.08:
                cell.metabolic_state = "quiescent"
            elif cell.local_oxygen < 0.30:
                cell.metabolic_state = "glycolytic_hypoxic"
            else:
                cell.metabolic_state = "normoxic"

            # Intracellular drug transport for Drug A
            k_perm = self.k_perm_default
            k_efflux_total = self.k_out_default + self.k_efflux_pump * cell.efflux_pump_level
            d_intra_a = k_perm * cell.local_drug_conc - k_efflux_total * cell.intracellular_drug
            # Exact first-order uptake/efflux update: Euler is unstable at 15-minute steps.
            equilibrium_a = k_perm * cell.local_drug_conc / max(1e-12, k_efflux_total)
            cell.intracellular_drug = max(0.0, equilibrium_a + (cell.intracellular_drug - equilibrium_a) * math.exp(-k_efflux_total * dt_seconds))
            cell.exposure_auc += cell.intracellular_drug * (dt_seconds / 3600.0)

            # Intracellular drug transport for Drug B
            intra_b = getattr(cell, "intracellular_drug_b", 0.0)
            if has_drug_b:
                d_intra_b = k_perm * local_drug_b_conc - k_efflux_total * intra_b
                equilibrium_b = k_perm * local_drug_b_conc / max(1e-12, k_efflux_total)
                intra_b = max(0.0, equilibrium_b + (intra_b - equilibrium_b) * math.exp(-k_efflux_total * dt_seconds))
            else:
                intra_b = 0.0
            setattr(cell, "intracellular_drug_b", intra_b)

            # Check Severe Anoxia Necrosis (O2 < 0.08)
            if cell.local_oxygen < 0.08:
                p_necrosis = 1.0 - math.exp(-0.00012 * (0.08 - cell.local_oxygen) * dt_seconds)
                if self._rng.random() < p_necrosis:
                    cell.death_mode = CellDeathMode.NECROSIS
                    cell.viability = 0.0
                    self.cumulative_necrosis_count += 1
                    events["necrosis"] += 1
                    continue

            # Pharmacodynamic Kill (Apoptosis via Hill equation / Bliss combination)
            e_a = 0.0
            if cell.intracellular_drug > 1e-5:
                eff_ic50_a = self._compute_multifactor_ic50(
                    cell=cell,
                    ic50_sens_base=ic50_sens_base,
                    ic50_res_base=ic50_res_base,
                    ic50_healthy_base=ic50_healthy_base,
                )
                conc_a = cell.intracellular_drug
                hill_factor_a = (conc_a**hill_slope) / (eff_ic50_a**hill_slope + conc_a**hill_slope)
                e_a = emax * hill_factor_a

            e_b = 0.0
            if has_drug_b and intra_b > 1e-5:
                eff_ic50_b = self._compute_multifactor_ic50(
                    cell=cell,
                    ic50_sens_base=ic50_sens_b,
                    ic50_res_base=ic50_res_b,
                    ic50_healthy_base=ic50_healthy_b,
                )
                conc_b = intra_b
                hill_factor_b = (conc_b**hill_slope_b) / (eff_ic50_b**hill_slope_b + conc_b**hill_slope_b)
                e_b = emax_b * hill_factor_b

            if e_a > 0.0 or e_b > 0.0:
                if has_drug_b and (e_a > 0.0 or e_b > 0.0):
                    # Independent per-second hazards add; these are not fractions.
                    e_comb = max(0.0, (e_a + e_b) * self.synergy_factor)
                else:
                    e_comb = e_a

                p_apoptosis = 1.0 - math.exp(-e_comb * dt_seconds)
                if self._rng.random() < p_apoptosis:
                    cell.death_mode = CellDeathMode.APOPTOSIS
                    cell.viability = 0.0
                    self.cumulative_apoptosis_count += 1
                    events["apoptosis"] += 1
                    continue

            # Proliferation & Division
            # Growth is suppressed by hypoxia, nutrient starvation, and contact density
            if cell.cell_type in (
                CellType.TUMOR_SENSITIVE,
                CellType.TUMOR_RESISTANT,
                CellType.EPITHELIAL_HEALTHY,
                CellType.CAF,
            ):
                local_neighbors = sum(
                    1
                    for ox, oy in cell_coords
                    if math.hypot(ox - cell.x, oy - cell.y) < (cell.radius * 2.8)
                )

                # Contact inhibition
                if cell.cell_type == CellType.EPITHELIAL_HEALTHY:
                    capacity = 4  # Strict contact inhibition
                elif cell.cell_type == CellType.CAF:
                    capacity = 6
                else:
                    capacity = 8  # Relaxed contact inhibition in malignant subclones

                if local_neighbors < capacity:
                    o2_growth_factor = cell.local_oxygen / (0.15 + cell.local_oxygen)
                    nut_growth_factor = cell.local_nutrient / (0.18 + cell.local_nutrient)
                    growth_increment = (
                        cell.growth_rate
                        * o2_growth_factor
                        * nut_growth_factor
                        * (1.0 - local_neighbors / (capacity + 2))
                        * (dt_seconds / 3600.0)
                    )
                    cell.division_progress += max(0.0, growth_increment)

                    if cell.division_progress >= 1.0:
                        cell.division_progress = 0.0
                        daughter = self._divide_cell(cell)
                        new_daughter_cells.append(daughter)
                        events["division"] += 1
                        if daughter.clone_id != cell.clone_id:
                            events["mutations_emerged"] += 1

        self.cells.extend(new_daughter_cells)
        return events

    def _compute_multifactor_ic50(
        self,
        cell: CellularEntity,
        ic50_sens_base: float,
        ic50_res_base: float,
        ic50_healthy_base: float,
    ) -> float:
        """
        Multifactorial computation of effective IC50:
        Never a single mutation lookup! Incorporates:
        1. Base subclonal genetic background (KRAS, TP53, CDKN2A, SMAD4).
        2. Continuous resistance score and active transporter pump level.
        3. Hypoxic protective niche (HIF-1alpha induced chemoresistance).
        4. Stroma / ECM physical and paracrine cushioning.
        """
        if cell.cell_type == CellType.EPITHELIAL_HEALTHY:
            return ic50_healthy_base
        elif cell.cell_type in (CellType.CAF, CellType.FIBROBLAST):
            return ic50_healthy_base * 1.5
        elif cell.cell_type in (CellType.IMMUNE_T, CellType.MACROPHAGE):
            return ic50_healthy_base * 0.9

        # Tumor Subclones
        if cell.clone_id == "CLONE-CLASSICAL":
            base = ic50_sens_base
        else:
            base = ic50_res_base

        # Pathway-derived multiplier
        pathway_multiplier = 1.0
        if "KRAS_G12D" in cell.mutations:
            pathway_multiplier += 0.15
        if "TP53_LOF" in cell.mutations:
            pathway_multiplier += 0.30
        if "CDKN2A_DEL" in cell.mutations:
            pathway_multiplier += 0.25
        if "SMAD4_LOF" in cell.mutations:
            pathway_multiplier += 0.35
        if "ABCB1_OVEREXPRESSION" in cell.mutations:
            pathway_multiplier += 0.50

        # Continuous trait scaling
        trait_multiplier = (1.0 + 1.2 * cell.resistance_score) * (
            1.0 + 0.4 * cell.efflux_pump_level
        )

        # Hypoxic resistance factor: low O2 triggers HIF-1alpha, slowing proliferation and apoptosis
        hypoxic_protection = 1.0
        if cell.local_oxygen < 0.30:
            hypoxic_protection += 0.8 * (1.0 - cell.local_oxygen / 0.30)

        eff_ic50 = base * pathway_multiplier * trait_multiplier * hypoxic_protection
        return max(0.01, float(eff_ic50))

    def _divide_cell(self, parent: CellularEntity) -> CellularEntity:
        """Binary division producing daughter cell with potential mutation / clonal shift."""
        offset_dist = parent.radius * 1.8
        angle = self._rng.uniform(0.0, 2.0 * math.pi)
        nx_pos = np.clip(parent.x + offset_dist * math.cos(angle), 15.0, self.width - 15.0)
        ny_pos = np.clip(parent.y + offset_dist * math.sin(angle), 15.0, self.height - 15.0)

        daughter_mutations = list(parent.mutations)
        daughter_res = float(parent.resistance_score)
        daughter_efflux = float(parent.efflux_pump_level)
        daughter_clone = parent.clone_id

        # Stochastic subclonal mutation / clonal drift under selective pressure
        if parent.cell_type in (CellType.TUMOR_SENSITIVE, CellType.TUMOR_RESISTANT):
            # 2.5% chance of spontaneous mutation or resistance escalation
            if self._rng.random() < 0.025:
                daughter_res = min(1.0, daughter_res + self._rng.uniform(0.08, 0.18))
                daughter_efflux = min(1.0, daughter_efflux + self._rng.uniform(0.06, 0.15))
                if "ABCB1_OVEREXPRESSION" not in daughter_mutations and daughter_efflux > 0.40:
                    daughter_mutations.append("ABCB1_OVEREXPRESSION")
                    daughter_clone = f"{parent.clone_id}-MDR1"
                elif daughter_res > 0.65 and "SMAD4_LOF" not in daughter_mutations:
                    daughter_mutations.append("SMAD4_LOF")
                    daughter_clone = "CLONE-BASAL-EMERGENT"

        daughter = CellularEntity(
            cell_id=f"{parent.cell_id}_d{self._rng.integers(1000, 9999)}",
            cell_type=parent.cell_type,
            clone_id=daughter_clone,
            x=float(nx_pos),
            y=float(ny_pos),
            radius=parent.radius,
            growth_rate=parent.growth_rate,
            mutations=daughter_mutations,
            resistance_score=daughter_res,
            efflux_pump_level=daughter_efflux,
            generation=parent.generation + 1,
            parent_clone_id=parent.clone_id,
        )
        return daughter

    def _update_immune_interactions(self, dt_seconds: float) -> None:
        """
        Cytotoxic T cells & Macrophages mediate antitumor clearance,
        heavily suppressed in dense ECM and hypoxic regions.
        """
        t_cells = [
            c
            for c in self.cells
            if c.death_mode == CellDeathMode.ALIVE and c.cell_type == CellType.IMMUNE_T
        ]
        if not t_cells:
            return

        for tumor_cell in self.cells:
            if (
                tumor_cell.death_mode != CellDeathMode.ALIVE
                or tumor_cell.cell_type not in (CellType.TUMOR_SENSITIVE, CellType.TUMOR_RESISTANT)
            ):
                continue

            # Sample local ECM and oxygen
            gx = int(np.clip(tumor_cell.x / self.dx, 0, self.nx - 1))
            gy = int(np.clip(tumor_cell.y / self.dy, 0, self.ny - 1))
            ecm = self.grid_ecm[gy, gx]
            o2 = self.grid_oxygen[gy, gx]

            # Immune suppression by desmoplasia and severe hypoxia
            if ecm > 0.65 or o2 < 0.20:
                continue

            # Find nearby active T cell
            for tc in t_cells:
                dist = math.hypot(tc.x - tumor_cell.x, tc.y - tumor_cell.y)
                if dist < 18.0:
                    # Cytotoxic clearance probability
                    p_kill = 1.0 - math.exp(-0.00004 * dt_seconds)
                    if self._rng.random() < p_kill:
                        tumor_cell.death_mode = CellDeathMode.APOPTOSIS
                        tumor_cell.viability = 0.0
                        self.cumulative_apoptosis_count += 1
                    break

    def get_snapshot(self) -> dict[str, Any]:
        """
        Return serializable state for visualization, metrics, and Cancer Battle View.
        """
        # Cell and Death counts
        type_counts: dict[str, int] = {}
        clone_counts: dict[str, int] = {}
        death_counts = {
            CellDeathMode.ALIVE.value: 0,
            CellDeathMode.APOPTOSIS.value: 0,
            CellDeathMode.NECROSIS.value: 0,
        }

        viable_tumor_count = 0
        resistant_tumor_count = 0
        total_intracellular_drug = 0.0
        active_count = 0

        for c in self.cells:
            death_counts[c.death_mode.value] = death_counts.get(c.death_mode.value, 0) + 1
            if c.death_mode == CellDeathMode.ALIVE:
                active_count += 1
                type_counts[c.cell_type.value] = type_counts.get(c.cell_type.value, 0) + 1
                clone_counts[c.clone_id] = clone_counts.get(c.clone_id, 0) + 1
                total_intracellular_drug += c.intracellular_drug

                if c.cell_type in (CellType.TUMOR_SENSITIVE, CellType.TUMOR_RESISTANT):
                    viable_tumor_count += 1
                    if c.cell_type == CellType.TUMOR_RESISTANT or c.resistance_score > 0.40:
                        resistant_tumor_count += 1

        total_grid_cells = float(self.nx * self.ny)
        normoxic_pct = float(np.sum(self.grid_oxygen >= 0.30) / total_grid_cells * 100.0)
        hypoxic_pct = float(
            np.sum((self.grid_oxygen < 0.30) & (self.grid_oxygen >= 0.08))
            / total_grid_cells
            * 100.0
        )
        necrotic_pct = float(np.sum(self.grid_oxygen < 0.08) / total_grid_cells * 100.0)

        resistant_fraction = (
            float(resistant_tumor_count / viable_tumor_count) if viable_tumor_count > 0 else 0.0
        )
        mean_intra_drug = (
            float(total_intracellular_drug / active_count) if active_count > 0 else 0.0
        )

        total_free_drug = self.grid_drug_free + self.grid_drug_b_free

        summary_metrics = {
            "total_cells": len(self.cells),
            "viable_cells": active_count,
            "viable_tumor_cells": viable_tumor_count,
            "resistant_fraction": round(resistant_fraction, 4),
            "mean_intracellular_drug_um": round(mean_intra_drug, 4),
            "mean_ecm_density": round(float(np.mean(self.grid_ecm)), 4),
        }

        return {
            "disease_id": self.disease_id,
            "display_name": self.display_name,
            "biological_seconds": round(self.biological_seconds, 2),
            "biological_hours": round(self.biological_seconds / 3600.0, 3),
            "step_count": self.step_count,
            "domain": {
                "width": self.width,
                "height": self.height,
                "nx": self.nx,
                "ny": self.ny,
                "dx": self.dx,
                "dy": self.dy,
            },
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
        """
        Return disease-specific and general biological indicators with epistemic tags.
        Systemic organ markers are strictly marked UNMODELED unless systemic_state is provided.
        """
        snap = self.get_snapshot()
        metrics = snap["summary_metrics"]
        zones = snap["zones"]
        viable_tumor = metrics["viable_tumor_cells"]

        # CA19-9 surrogate: serum shedding ~ proportional to viable tumor burden
        # Baseline normal human reference: < 37 U/mL. In advanced PDAC: 100 - 10000+ U/mL.
        ca19_9_val = round(35.0 + viable_tumor * 4.8, 1)

        biomarkers: dict[str, Any] = {
            "CA19_9_SURROGATE": ScientificValue(
                name="CA19-9 Pancreatic Tumor Marker",
                value=ca19_9_val,
                unit="U/mL",
                epistemic_category=EpistemicCategory.EXPLORATORY_MODEL,
                source="EVOCURE PDAC Mechanistic Shedding Model",
                confidence=0.82,
                uncertainty_low=round(ca19_9_val * 0.85, 1),
                uncertainty_high=round(ca19_9_val * 1.18, 1),
                in_applicability_domain=True,
                validation_status="Correlated to 2D viable cell biomass; computational surrogate",
                limitations="In silico surrogate; CA19-9 shedding is non-linear and Lewis-antigen dependent in humans.",
            ).to_dict(),
            "HYPOXIA_FRACTION": ScientificValue(
                name="Tumor Hypoxic Fraction (O2 < 0.30)",
                value=round(zones["hypoxic_area_pct"] / 100.0, 3),
                unit="fraction",
                epistemic_category=EpistemicCategory.MECHANISTIC_SIMULATION,
                source="2D Reaction-Diffusion Eulerian Solver",
                confidence=0.90,
                uncertainty_low=round(max(0.0, (zones["hypoxic_area_pct"] - 5.0) / 100.0), 3),
                uncertainty_high=round(min(1.0, (zones["hypoxic_area_pct"] + 5.0) / 100.0), 3),
                in_applicability_domain=True,
                validation_status="Mechanistically derived from vascular geometry and cellular sinks",
                limitations="2D planar approximation without 3D convective vascular interconnects.",
            ).to_dict(),
            "NECROTIC_CORE_FRACTION": ScientificValue(
                name="Tumor Necrotic Core Area (O2 < 0.08)",
                value=round(zones["necrotic_core_pct"] / 100.0, 3),
                unit="fraction",
                epistemic_category=EpistemicCategory.MECHANISTIC_SIMULATION,
                source="2D Reaction-Diffusion Eulerian Solver",
                confidence=0.92,
                uncertainty_low=round(max(0.0, (zones["necrotic_core_pct"] - 3.0) / 100.0), 3),
                uncertainty_high=round(min(1.0, (zones["necrotic_core_pct"] + 3.0) / 100.0), 3),
                in_applicability_domain=True,
                validation_status="Mechanistically derived anoxic threshold",
                limitations="Assumes constant 0.08 anoxic necrosis trigger threshold.",
            ).to_dict(),
            "DESMOPLASIA_SCORE": ScientificValue(
                name="Stroma Desmoplasia ECM Score",
                value=metrics["mean_ecm_density"],
                unit="dimensionless [0-1]",
                epistemic_category=EpistemicCategory.MECHANISTIC_SIMULATION,
                source="EVOCURE CAF Secretion & Hindered Transport Model",
                confidence=0.88,
                in_applicability_domain=True,
                validation_status="Mechanistic CAF deposition and degradation balance",
                limitations="Does not model enzymatic cross-linking by lysyl oxidase (LOX).",
            ).to_dict(),
            "CLONAL_RESISTANCE_INDEX": ScientificValue(
                name="Clonal Resistance Proportion",
                value=metrics["resistant_fraction"],
                unit="ratio",
                epistemic_category=EpistemicCategory.MECHANISTIC_SIMULATION,
                source="EVOCURE Subclonal Population Tracker",
                confidence=0.95,
                in_applicability_domain=True,
                validation_status="Directly tracked agent-based subclonal genotypes",
                limitations="Tracks modeled driver alterations (KRAS, TP53, CDKN2A, SMAD4, ABCB1).",
            ).to_dict(),
        }

        # Systemic Markers: Strictly UNMODELED if systemic PBPK is disabled
        if systemic_state is None:
            biomarkers["HEPATIC_ALT"] = ScientificValue(
                name="Serum Alanine Aminotransferase (ALT)",
                value=None,
                unit="U/L",
                epistemic_category=EpistemicCategory.UNMODELED,
                source="None",
                in_applicability_domain=False,
                validation_status="Systemic PBPK disabled; organ toxicity unmodeled in 2D slice",
                limitations="Requires active 7-compartment systemic PBPK module.",
            ).to_dict()
            biomarkers["RENAL_CREATININE"] = ScientificValue(
                name="Serum Creatinine",
                value=None,
                unit="mg/dL",
                epistemic_category=EpistemicCategory.UNMODELED,
                source="None",
                in_applicability_domain=False,
                validation_status="Systemic PBPK disabled; renal clearance unmodeled in 2D slice",
                limitations="Requires active 7-compartment systemic PBPK module.",
            ).to_dict()
        else:
            biomarkers["HEPATIC_ALT"] = ScientificValue(
                name="Serum Alanine Aminotransferase (ALT)",
                value=systemic_state.get("alt_u_l", 28.0),
                unit="U/L",
                epistemic_category=EpistemicCategory.EXPLORATORY_MODEL,
                source="EVOCURE 7-Compartment Systemic PBPK",
                confidence=0.75,
                in_applicability_domain=True,
                validation_status="Calibrated exploratory hepatotoxicity prediction",
                limitations="Computational ODE estimate based on liver AUC.",
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
        """
        Runs identical initial tumor state (seed) under two parallel branches:
        - Control: 0.0 uM untreated control.
        - Treated: applied_concentration_um exposed scenario.

        Returns side-by-side time series and delta metrics.
        """
        duration_hours = max(1.0, float(duration_hours))
        applied_concentration_um = max(0.0, float(applied_concentration_um))

        # Instantiate Branch A (Control)
        control_module = PDACModule(
            seed=seed, width=self.width, height=self.height, nx=self.nx, ny=self.ny
        )
        control_module.apply_compound_exposure(
            compound=compound,
            pharmacology_profile=pharmacology_profile,
            applied_concentration_um=0.0,
        )

        # Instantiate Branch B (Treated)
        treated_module = PDACModule(
            seed=seed, width=self.width, height=self.height, nx=self.nx, ny=self.ny
        )
        treated_module.apply_compound_exposure(
            compound=compound,
            pharmacology_profile=pharmacology_profile,
            applied_concentration_um=applied_concentration_um,
        )

        time_points_hours: list[float] = [0.0]

        def extract_trajectory_point(mod: PDACModule) -> dict[str, Any]:
            snap = mod.get_snapshot()
            m = snap["summary_metrics"]
            d = snap["cumulative_deaths"]
            z = snap["zones"]
            c_dist = snap["clone_distribution"]
            return {
                "viable_tumor_cells": m["viable_tumor_cells"],
                "classical_clone_cells": c_dist.get("CLONE-CLASSICAL", 0),
                "basal_clone_cells": c_dist.get("CLONE-BASAL", 0),
                "resistant_fraction": m["resistant_fraction"],
                "apoptotic_deaths": d["apoptosis"],
                "necrotic_deaths": d["necrosis"],
                "hypoxic_fraction": round(z["hypoxic_area_pct"] / 100.0, 3),
                "mean_intracellular_drug_um": m["mean_intracellular_drug_um"],
            }

        control_series: list[dict[str, Any]] = [extract_trajectory_point(control_module)]
        treated_series: list[dict[str, Any]] = [extract_trajectory_point(treated_module)]

        # Step size: 1 hour (3600 seconds)
        dt_step = 3600.0
        steps = int(math.ceil(duration_hours))

        for h in range(1, steps + 1):
            control_module.step(dt_step)
            treated_module.step(dt_step)

            time_points_hours.append(float(h))
            control_series.append(extract_trajectory_point(control_module))
            treated_series.append(extract_trajectory_point(treated_module))

        # Calculate final delta metrics
        final_ctrl = control_series[-1]
        final_trt = treated_series[-1]

        initial_tumor = control_series[0]["viable_tumor_cells"]
        ctrl_final_tumor = max(1, final_ctrl["viable_tumor_cells"])
        trt_final_tumor = final_trt["viable_tumor_cells"]

        # Tumor Growth Inhibition (TGI %)
        # TGI = (1 - (T_final / C_final)) * 100%
        tgi_pct = (1.0 - (trt_final_tumor / ctrl_final_tumor)) * 100.0

        # Clonal Shift Delta (Treated resistant fraction minus Control resistant fraction)
        clonal_shift_delta = final_trt["resistant_fraction"] - final_ctrl["resistant_fraction"]

        # Apoptosis Induction Ratio
        ctrl_apop = max(1, final_ctrl["apoptotic_deaths"])
        trt_apop = final_trt["apoptotic_deaths"]
        apoptosis_induction_ratio = trt_apop / ctrl_apop

        # Desmoplastic Barrier Impact: ratio of core drug conc to edge drug conc
        trt_snap = treated_module.get_snapshot()
        drug_grid = np.array(trt_snap["heatmaps"]["drug_free"])
        center_y, center_x = self.ny // 2, self.nx // 2
        core_drug = float(np.mean(drug_grid[center_y - 2 : center_y + 3, center_x - 3 : center_x + 4]))
        periphery_drug = float(np.mean(np.concatenate([drug_grid[0:2, :].flatten(), drug_grid[-2:, :].flatten()])))
        barrier_attenuation_ratio = (core_drug / max(1e-6, periphery_drug)) if periphery_drug > 1e-6 else 1.0

        return {
            "compound_id": compound.compound_id,
            "compound_name": compound.name,
            "applied_concentration_um": applied_concentration_um,
            "duration_hours": duration_hours,
            "seed": seed,
            "time_points_hours": time_points_hours,
            "control_trajectory": {
                k: [pt[k] for pt in control_series] for k in control_series[0].keys()
            },
            "treated_trajectory": {
                k: [pt[k] for pt in treated_series] for k in treated_series[0].keys()
            },
            "deltas": {
                "initial_tumor_cells": initial_tumor,
                "control_final_tumor_cells": ctrl_final_tumor,
                "treated_final_tumor_cells": trt_final_tumor,
                "tumor_growth_inhibition_pct": round(tgi_pct, 2),
                "clonal_shift_delta": round(clonal_shift_delta, 4),
                "apoptosis_induction_ratio": round(apoptosis_induction_ratio, 2),
                "barrier_attenuation_ratio": round(barrier_attenuation_ratio, 3),
                "observed_selectivity": (
                    "High sensitivity in Classical subclone, marked resistance in Basal subclone"
                    if clonal_shift_delta > 0.10
                    else "Uniform response or low concentration"
                ),
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
        """
        Runs identical initial tumor state under 4 arms:
        1. Untreated Control (0.0 uM)
        2. Monotherapy A (conc_a_um)
        3. Monotherapy B (conc_b_um)
        4. Combination A + B (conc_a_um + conc_b_um, synergy_factor)
        """
        duration_hours = max(1.0, float(duration_hours))
        conc_a_um = max(0.0, float(conc_a_um))
        conc_b_um = max(0.0, float(conc_b_um))

        # Instantiate 4 arms
        mod_ctrl = type(self)(seed=seed, width=self.width, height=self.height, nx=self.nx, ny=self.ny)
        mod_ctrl.apply_compound_exposure(compound_a, pharma_a, applied_concentration_um=0.0)

        mod_a = type(self)(seed=seed, width=self.width, height=self.height, nx=self.nx, ny=self.ny)
        mod_a.apply_compound_exposure(compound_a, pharma_a, applied_concentration_um=conc_a_um)

        mod_b = type(self)(seed=seed, width=self.width, height=self.height, nx=self.nx, ny=self.ny)
        mod_b.apply_compound_exposure(compound_b, pharma_b, applied_concentration_um=conc_b_um)

        mod_combo = type(self)(seed=seed, width=self.width, height=self.height, nx=self.nx, ny=self.ny)
        mod_combo.apply_combination_exposure(
            compound_a, compound_b, pharma_a, pharma_b, conc_a_um, conc_b_um, synergy_factor=synergy_factor
        )

        def extract_point(mod: Any) -> dict[str, Any]:
            snap = mod.get_snapshot()
            m = snap["summary_metrics"]
            d = snap["cumulative_deaths"]
            healthy_count = sum(
                1 for c in mod.cells
                if c.death_mode == CellDeathMode.ALIVE and c.cell_type == CellType.EPITHELIAL_HEALTHY
            )
            return {
                "viable_tumor_cells": m["viable_tumor_cells"],
                "resistant_fraction": m["resistant_fraction"],
                "apoptotic_deaths": d["apoptosis"],
                "necrotic_deaths": d["necrosis"],
                "healthy_cells": healthy_count,
            }

        time_points: list[float] = [0.0]
        series_ctrl = [extract_point(mod_ctrl)]
        series_a = [extract_point(mod_a)]
        series_b = [extract_point(mod_b)]
        series_combo = [extract_point(mod_combo)]

        dt_step = 3600.0
        steps = int(math.ceil(duration_hours))
        for h in range(1, steps + 1):
            mod_ctrl.step(dt_step)
            mod_a.step(dt_step)
            mod_b.step(dt_step)
            mod_combo.step(dt_step)

            time_points.append(float(h))
            series_ctrl.append(extract_point(mod_ctrl))
            series_a.append(extract_point(mod_a))
            series_b.append(extract_point(mod_b))
            series_combo.append(extract_point(mod_combo))

        # Final metrics
        initial_tumor = series_ctrl[0]["viable_tumor_cells"]
        ctrl_final = max(1, series_ctrl[-1]["viable_tumor_cells"])
        a_final = series_a[-1]["viable_tumor_cells"]
        b_final = series_b[-1]["viable_tumor_cells"]
        combo_final = series_combo[-1]["viable_tumor_cells"]

        tgi_a = max(0.0, (1.0 - (a_final / ctrl_final)) * 100.0)
        tgi_b = max(0.0, (1.0 - (b_final / ctrl_final)) * 100.0)
        tgi_combo = max(0.0, (1.0 - (combo_final / ctrl_final)) * 100.0)

        # Expected Bliss independence
        bliss_expected = tgi_a + tgi_b - (tgi_a * tgi_b / 100.0)
        excess_over_bliss = tgi_combo - bliss_expected

        if excess_over_bliss > 4.0:
            classification = "Synergistic"
        elif excess_over_bliss < -4.0:
            classification = "Antagonistic"
        else:
            classification = "Additive"

        initial_healthy = max(1, series_ctrl[0]["healthy_cells"])
        healthy_survival = {
            "control": round(series_ctrl[-1]["healthy_cells"] / initial_healthy * 100.0, 1),
            "monotherapy_a": round(series_a[-1]["healthy_cells"] / initial_healthy * 100.0, 1),
            "monotherapy_b": round(series_b[-1]["healthy_cells"] / initial_healthy * 100.0, 1),
            "combination": round(series_combo[-1]["healthy_cells"] / initial_healthy * 100.0, 1),
        }

        return {
            "disease_id": self.disease_id,
            "compound_a": {"id": compound_a.compound_id, "name": compound_a.name, "conc_um": conc_a_um},
            "compound_b": {"id": compound_b.compound_id, "name": compound_b.name, "conc_um": conc_b_um},
            "duration_hours": duration_hours,
            "seed": seed,
            "synergy_factor": synergy_factor,
            "time_points_hours": time_points,
            "trajectories": {
                "control": {k: [pt[k] for pt in series_ctrl] for k in series_ctrl[0]},
                "monotherapy_a": {k: [pt[k] for pt in series_a] for k in series_a[0]},
                "monotherapy_b": {k: [pt[k] for pt in series_b] for k in series_b[0]},
                "combination": {k: [pt[k] for pt in series_combo] for k in series_combo[0]},
            },
            "deltas": {
                "initial_tumor_cells": initial_tumor,
                "control_final_tumor_cells": ctrl_final,
                "monotherapy_a_final_tumor_cells": a_final,
                "monotherapy_b_final_tumor_cells": b_final,
                "combination_final_tumor_cells": combo_final,
                "tgi_a_pct": round(tgi_a, 2),
                "tgi_b_pct": round(tgi_b, 2),
                "tgi_combo_pct": round(tgi_combo, 2),
                "bliss_expected_pct": round(bliss_expected, 2),
                "excess_over_bliss": round(excess_over_bliss, 2),
                "synergy_classification": classification,
                "healthy_cell_survival_pct": healthy_survival,
                "clonal_shift": {
                    "control": series_ctrl[-1]["resistant_fraction"],
                    "monotherapy_a": series_a[-1]["resistant_fraction"],
                    "monotherapy_b": series_b[-1]["resistant_fraction"],
                    "combination": series_combo[-1]["resistant_fraction"],
                },
            },
        }
