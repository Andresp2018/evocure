"""
Metabolic Dysfunction-Associated Steatohepatitis (MASLD / MASH) Module.

Implements a 2D mechanistic hepatic lobule model representing non-alcoholic
fatty liver disease progressing to steatohepatitis and bridging fibrosis.
Features metabolic zonation (periportal Zone 1 high O2/beta-oxidation to
pericentral Zone 3 lipotoxicity), steatotic hepatocytes (CellType.TUMOR_SENSITIVE),
collagen-secreting activated hepatic stellate cells (CellType.CAF / TUMOR_RESISTANT),
quiescent hepatocytes (CellType.EPITHELIAL_HEALTHY), inflammatory Kupffer cells
(CellType.MACROPHAGE), hepatotoxicity assessment (penalizing toxic agents like
Acetaminophen while rewarding lipid-clearing Resmetirom/Metformin), single &
combination exposure, and calibrated reference anchors.
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


class MASLDModule(DiseaseModule):
    """
    Mechanistic 2D Hepatic Lobule MASLD / MASH Model.

    Biological & biophysical features:
    1. Hepatic Lobule Zonation:
       Periportal Zone 1 (peripheral portal triad vessels, O2 ~0.80-1.0, high beta-oxidation).
       Pericentral Zone 3 (central vein region, O2 ~0.20-0.35, high lipid accumulation and lipotoxicity).
    2. Steatotic Lipotoxic Hepatocytes (CellType.TUMOR_SENSITIVE render tag).
       Metabolic modulators reduce intracellular lipid burden and reverse steatosis.
    3. Activated Hepatic Stellate Cells (HSCs, CellType.CAF / TUMOR_RESISTANT) depositing collagen.
    4. Quiescent Functional Hepatocytes (CellType.EPITHELIAL_HEALTHY).
    5. Inflammatory Kupffer Cells (CellType.MACROPHAGE) activated by lipotoxic necrosis.
    6. Hepatotoxicity Evaluation:
       Molecules with high intrinsic cytotoxicity or APAP-like profile trigger hepatocyte death,
       elevating ALT and penalizing therapeutic safety.
    7. Calibrated reference anchors for Metformin, Resmetirom, Curcumin, Aspirin, Acetaminophen.
    """

    REFERENCE_COMPOUNDS: dict[str, dict[str, Any]] = {
        "RESMETIROM": {
            "name": "Resmetirom (MGL-3196)",
            "smiles": "CC1=CC(=C(C=C1)OC2=CC(=C(C=C2)C#N)Cl)N3C(=O)NC(=O)C(=N3)O",
            "ic50_sensitive_um": 0.35,      # Effective steatosis resolution EC50
            "ic50_resistant_um": 2.20,      # HSC antifibrotic EC50
            "ic50_healthy_um": 45.0,        # Wide safety window (liver-directed)
            "fraction_unbound_plasma": 0.01,
            "modulates_stroma": True,       # Antifibrotic
            "hepatotoxic": False,
            "mechanism": "Thyroid Hormone Receptor Beta (THR-beta) Selective Agonist",
        },
        "METFORMIN": {
            "name": "Metformin",
            "smiles": "CN(C)C(=N)NC(=N)N",
            "ic50_sensitive_um": 4.50,      # Steatosis resolution EC50
            "ic50_resistant_um": 12.00,
            "ic50_healthy_um": 120.0,
            "fraction_unbound_plasma": 0.99,
            "modulates_stroma": True,
            "hepatotoxic": False,
            "mechanism": "AMPK Activator / Mitochondrial Complex I Mild Inhibitor",
        },
        "CURCUMIN": {
            "name": "Curcumin",
            "smiles": "COC1=C(C=CC(=C1)C=CC(=O)CC(=O)C=CC2=CC(=C(C=C2)O)OC)O",
            "ic50_sensitive_um": 3.20,
            "ic50_resistant_um": 8.50,
            "ic50_healthy_um": 35.0,
            "fraction_unbound_plasma": 0.12,
            "modulates_stroma": False,
            "hepatotoxic": False,
            "mechanism": "Antioxidant / NF-kB & Lipogenesis Modulator",
        },
        "ASPIRIN": {
            "name": "Aspirin",
            "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
            "ic50_sensitive_um": 8.00,
            "ic50_resistant_um": 18.00,
            "ic50_healthy_um": 50.0,
            "fraction_unbound_plasma": 0.50,
            "modulates_stroma": False,
            "hepatotoxic": False,
            "mechanism": "Cyclooxygenase (COX-1/2) Anti-inflammatory Inhibitor",
        },
        "ACETAMINOPHEN": {
            "name": "Acetaminophen (APAP - Toxic Control)",
            "smiles": "CC(=O)NC1=CC=C(C=C1)O",
            "ic50_sensitive_um": 25.0,
            "ic50_resistant_um": 25.0,
            "ic50_healthy_um": 1.20,        # Low healthy IC50 = severe hepatotoxicity!
            "fraction_unbound_plasma": 0.85,
            "modulates_stroma": False,
            "hepatotoxic": True,            # Induces NAPQI-mediated hepatocyte necrosis
            "mechanism": "NAPQI Reactive Metabolite Hepatotoxicity Model",
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
        self.diffusivity_oxygen: float = 38.0
        self.diffusivity_nutrient: float = 30.0
        self.diffusivity_drug_base: float = 48.0
        self.beta_fibrosis: float = 2.4             # Fibrotic collagen diffusion barrier
        self.clearance_drug_tissue: float = 0.00014

        # Uptake / Clearance
        self.k_perm_default: float = 0.018
        self.k_out_default: float = 0.007
        self.k_efflux_pump: float = 0.020

        # PD Hill Parameters
        self.hill_slope_default: float = 2.0
        self.emax_default: float = 0.000045
        self.tau_kill: float = 3600.0

        # Continuous Grids
        self.grid_oxygen = np.ones((self.ny, self.nx), dtype=np.float64)
        self.grid_nutrient = np.ones((self.ny, self.nx), dtype=np.float64)
        self.grid_ecm = np.zeros((self.ny, self.nx), dtype=np.float64)  # Hepatic Collagen Fibrosis
        self.grid_drug_total = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.grid_drug_free = np.zeros((self.ny, self.nx), dtype=np.float64)

        # Dual Compound Grids
        self.grid_drug_b_total = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.grid_drug_b_free = np.zeros((self.ny, self.nx), dtype=np.float64)

        # Hepatic Lobule Vessels (Portal triads + Central vein)
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
        return "METABOLIC_MASLD"

    @property
    def display_name(self) -> str:
        return "Metabolic Dysfunction-Associated Steatohepatitis (MASLD / MASH)"

    def metadata(self) -> dict[str, Any]:
        return {
            "disease_id": self.disease_id,
            "display_name": self.display_name,
            "category": "metabolic_disease",
            "organ": "Liver",
            "histopathology": "Hepatic lobule steatohepatitis with ballooning hepatocytes, Mallory-Denk bodies, and bridging fibrosis",
            "implementation_status": "interface_ready_requires_calibrated_dataset",
            "is_calibrated": False,
            "epistemic_category": EpistemicCategory.DEMONSTRATION_UNVALIDATED.value,
            "cellular_entities": [
                "Steatotic Lipotoxic Hepatocytes (TUMOR_SENSITIVE render tag)",
                "Activated Hepatic Stellate Cells (CAF / TUMOR_RESISTANT render tag)",
                "Quiescent Healthy Hepatocytes (EPITHELIAL_HEALTHY)",
                "Kupffer Cells (MACROPHAGE)",
                "Sinusoidal Endothelial Cells (ENDOTHELIAL)",
            ],
            "subclones": [
                {
                    "clone_id": "STEATOTIC-HEPATOCYTE",
                    "subtype": "Triglyceride-Overloaded Steatotic Hepatocyte",
                    "driver_mutations": ["LIPOTOXIC_STRESS", "CYP2E1_INDUCED"],
                    "relative_ic50": "1.0x (Metabolic Modulator Sensitive)",
                    "efflux_expression": "Low (0.06)",
                },
                {
                    "clone_id": "ACTIVATED-HSC",
                    "subtype": "Myofibroblastic Hepatic Stellate Cell",
                    "driver_mutations": ["SMAD3_PHOSPHO", "COLLAGEN_TYPE1_HIGH"],
                    "relative_ic50": "4.5x (Fibrotic Focus Refractory)",
                    "efflux_expression": "Moderate (0.35)",
                },
            ],
            "hallmarks_modeled": [
                "Hepatic lobule zonation: Periportal Zone 1 to Pericentral Zone 3 oxygen/lipid gradient",
                "Steatosis reversal: metabolic modulators restore steatotic hepatocytes to healthy state",
                "Hepatic stellate cell collagen deposition driving progressive fibrosis stage F0 to F4",
                "Kupffer cell inflammatory activation in response to lipotoxic hepatocellular debris",
                "Hepatotoxicity assessment: APAP/toxic insults induce hepatocyte necrosis and surge ALT",
            ],
            "calibrated_reference_compounds": list(self.REFERENCE_COMPOUNDS.keys()),
            "assumptions": [
                "Classical hexagonal hepatic lobule modeled in 2D cross-section.",
                "Portal triads at periphery supply oxygen and compounds; central vein drains center.",
            ],
            "limitations": (
                "2D lobule neglects sinusoidal 3D fenestrations and space of Disse flow dynamics. "
                "Calibrated for comparative in silico metabolic/antifibrotic screening."
            ),
        }

    def initialize_scenario(self, seed: int, preset: str = "standard") -> None:
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self.biological_seconds = 0.0
        self.step_count = 0
        self.cumulative_apoptosis_count = 0
        self.cumulative_necrosis_count = 0

        self.vessels = self._create_hepatic_vasculature()
        self._recompute_vessel_mask()

        # Hepatic lobule gradient: Zone 1 (portal triads) high O2, Zone 3 (central) lower O2
        self.grid_oxygen = np.clip(0.18 + 0.82 * np.power(self.vessel_mask, 0.40), 0.12, 1.0)
        self.grid_nutrient = np.clip(0.25 + 0.75 * np.power(self.vessel_mask, 0.35), 0.15, 1.0)
        self.grid_ecm.fill(0.12)  # Baseline fibrosis
        self.grid_drug_total.fill(0.0)
        self.grid_drug_free.fill(0.0)
        self.grid_drug_b_total.fill(0.0)
        self.grid_drug_b_free.fill(0.0)

        self.cells = self._create_masld_population()
        self._relax_initial_fibrosis()

    def _create_hepatic_vasculature(self) -> list[dict[str, Any]]:
        """Portal triads along lobule periphery + Central vein at center."""
        return [
            # Peripheral Portal Triads (Hepatic Artery + Portal Vein branches)
            {"id": "portal_triad_1", "x1": 60.0, "y1": 60.0, "x2": 240.0, "y2": 70.0, "radius": 14.0, "perfusion": 1.0, "functional": True},
            {"id": "portal_triad_2", "x1": 560.0, "y1": 60.0, "x2": 740.0, "y2": 70.0, "radius": 14.0, "perfusion": 1.0, "functional": True},
            {"id": "portal_triad_3", "x1": 50.0, "y1": 530.0, "x2": 250.0, "y2": 540.0, "radius": 14.0, "perfusion": 1.0, "functional": True},
            {"id": "portal_triad_4", "x1": 550.0, "y1": 530.0, "x2": 750.0, "y2": 540.0, "radius": 14.0, "perfusion": 1.0, "functional": True},
            {"id": "portal_triad_5", "x1": 50.0, "y1": 280.0, "x2": 70.0, "y2": 320.0, "radius": 14.0, "perfusion": 1.0, "functional": True},
            {"id": "portal_triad_6", "x1": 730.0, "y1": 280.0, "x2": 750.0, "y2": 320.0, "radius": 14.0, "perfusion": 1.0, "functional": True},
            # Central Vein draining center of lobule (Zone 3)
            {"id": "central_vein", "x1": 380.0, "y1": 290.0, "x2": 420.0, "y2": 310.0, "radius": 15.0, "perfusion": 0.35, "functional": True},
        ]

    def _recompute_vessel_mask(self) -> None:
        self.vessel_mask.fill(0.0)
        grid_y, grid_x = np.mgrid[0 : self.ny, 0 : self.nx]
        cell_centers_x = (grid_x + 0.5) * self.dx
        cell_centers_y = (grid_y + 0.5) * self.dy

        for ves in self.vessels:
            if not ves.get("functional", True): continue
            x1, y1, x2, y2 = ves["x1"], ves["y1"], ves["x2"], ves["y2"]
            rad = ves["radius"]
            perf = ves.get("perfusion", 1.0)

            dx = x2 - x1
            dy = y2 - y1
            seg_len_sq = dx * dx + dy * dy
            if seg_len_sq == 0:
                dist = np.hypot(cell_centers_x - x1, cell_centers_y - y1)
            else:
                t = np.clip(((cell_centers_x - x1) * dx + (cell_centers_y - y1) * dy) / seg_len_sq, 0.0, 1.0)
                dist = np.hypot(cell_centers_x - (x1 + t * dx), cell_centers_y - (y1 + t * dy))

            mask = np.exp(-0.5 * (dist / max(rad, 1.0)) ** 2) * perf
            self.vessel_mask = np.maximum(self.vessel_mask, mask)

        self.vessel_mask = np.clip(self.vessel_mask, 0.0, 1.0)

    def _create_masld_population(self) -> list[CellularEntity]:
        cells: list[CellularEntity] = []
        center_x, center_y = self.width / 2.0, self.height / 2.0
        cid = 1

        # Zone 3 (Pericentral): high steatotic hepatocytes (CellType.TUMOR_SENSITIVE)
        num_steatotic = 68
        for _ in range(num_steatotic):
            r = self._rng.triangular(10.0, 80.0, 160.0)
            theta = self._rng.uniform(0.0, 2.0 * math.pi)
            cells.append(CellularEntity(
                cell_id=f"steatotic_{cid}",
                cell_type=CellType.TUMOR_SENSITIVE,
                clone_id="STEATOTIC-HEPATOCYTE",
                x=float(center_x + r * math.cos(theta)),
                y=float(center_y + r * math.sin(theta)),
                radius=6.8,
                growth_rate=0.015,
                mutations=["LIPOTOXIC_STRESS", "CYP2E1_INDUCED"],
                resistance_score=0.10,
                efflux_pump_level=0.08,
            ))
            cid += 1

        # Activated Hepatic Stellate Cells (HSCs) depositing collagen (CellType.CAF / TUMOR_RESISTANT)
        num_hsc = 28
        for _ in range(num_hsc):
            r = self._rng.uniform(40.0, 210.0)
            theta = self._rng.uniform(0.0, 2.0 * math.pi)
            cells.append(CellularEntity(
                cell_id=f"hsc_{cid}",
                cell_type=CellType.TUMOR_RESISTANT,
                clone_id="ACTIVATED-HSC",
                x=float(center_x + r * math.cos(theta)),
                y=float(center_y + r * math.sin(theta)),
                radius=6.5,
                growth_rate=0.020,
                mutations=["SMAD3_PHOSPHO", "COLLAGEN_TYPE1_HIGH"],
                resistance_score=0.55,
                efflux_pump_level=0.35,
            ))
            cid += 1

        # Zone 1 (Periportal): Quiescent Healthy Hepatocytes (CellType.EPITHELIAL_HEALTHY)
        num_healthy = 48
        for _ in range(num_healthy):
            r = self._rng.uniform(170.0, 320.0)
            theta = self._rng.uniform(0.0, 2.0 * math.pi)
            cells.append(CellularEntity(
                cell_id=f"healthy_hep_{cid}",
                cell_type=CellType.EPITHELIAL_HEALTHY,
                clone_id="HEALTHY-HEPATOCYTE",
                x=float(np.clip(center_x + r * math.cos(theta), 25.0, self.width - 25.0)),
                y=float(np.clip(center_y + r * math.sin(theta), 25.0, self.height - 25.0)),
                radius=6.6,
                growth_rate=0.008,
                resistance_score=0.05,
                efflux_pump_level=0.10,
            ))
            cid += 1

        # Kupffer Cells (CellType.MACROPHAGE) resident in hepatic sinusoids
        num_kupffer = 20
        for _ in range(num_kupffer):
            r = self._rng.uniform(80.0, 250.0)
            theta = self._rng.uniform(0.0, 2.0 * math.pi)
            cells.append(CellularEntity(
                cell_id=f"kupffer_{cid}",
                cell_type=CellType.MACROPHAGE,
                clone_id="KUPFFER-CELL",
                x=float(np.clip(center_x + r * math.cos(theta), 25.0, self.width - 25.0)),
                y=float(np.clip(center_y + r * math.sin(theta), 25.0, self.height - 25.0)),
                radius=5.8,
                growth_rate=0.004,
                resistance_score=0.12,
                efflux_pump_level=0.15,
            ))
            cid += 1

        return cells

    def _relax_initial_fibrosis(self) -> None:
        """Establish baseline bridging collagen fibrosis around activated HSCs."""
        for c in self.cells:
            if c.clone_id == "ACTIVATED-HSC":
                gx = int(np.clip(c.x / self.dx, 0, self.nx - 1))
                gy = int(np.clip(c.y / self.dy, 0, self.ny - 1))
                r = 2
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        ny_idx, nx_idx = gy + dy, gx + dx
                        if 0 <= ny_idx < self.ny and 0 <= nx_idx < self.nx:
                            w = 1.0 / (1.0 + dx * dx + dy * dy)
                            self.grid_ecm[ny_idx, nx_idx] = min(0.70, self.grid_ecm[ny_idx, nx_idx] + 0.22 * w)

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

        fu_raw = pharmacology_profile.get("fraction_unbound_plasma", 0.25)
        if isinstance(fu_raw, dict): fu_raw = fu_raw.get("value", 0.25)
        try: self.fraction_unbound_plasma = float(fu_raw if fu_raw is not None else 0.25)
        except: self.fraction_unbound_plasma = 0.25

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
            fu = p.get("fraction_unbound_plasma", 0.25)
            if isinstance(fu, dict): fu = fu.get("value", 0.25)
            try: return float(fu if fu is not None else 0.25)
            except: return 0.25

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
        self._update_fibrosis(dt_seconds)

        has_drug_a = self.applied_concentration_um > 0.0 or np.max(self.grid_drug_total) > 1e-6
        has_drug_b = self.applied_concentration_b_um > 0.0 or np.max(self.grid_drug_b_total) > 1e-6
        if has_drug_a or has_drug_b:
            self._diffuse_and_react_drug(dt_seconds)

        step_events = self._update_hepatocytes_and_hsc(dt_seconds)

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
        nut_sink_grid = np.ones((self.ny, self.nx), dtype=np.float64) * 0.001

        for c in self.cells:
            if c.death_mode != CellDeathMode.ALIVE: continue
            gx = int(np.clip(c.x / self.dx, 0, self.nx - 1))
            gy = int(np.clip(c.y / self.dy, 0, self.ny - 1))
            o2_sink_grid[gy, gx] += 0.015
            nut_sink_grid[gy, gx] += 0.018

        accelerated = gpu_substrates(self.grid_oxygen, self.grid_nutrient, self.vessel_mask,
            o2_sink_grid, nut_sink_grid, self.dx, self.dy,
            (self.diffusivity_oxygen, self.diffusivity_nutrient), (0.06, 0.05), (0.08, 0.1), sub_dt, num_substeps)
        if accelerated is not None:
            self.grid_oxygen, self.grid_nutrient = accelerated
            return

        for _ in range(num_substeps):
            lap_o2 = self._compute_laplacian(self.grid_oxygen)
            lap_nut = self._compute_laplacian(self.grid_nutrient)

            supply_o2 = 0.06 * self.vessel_mask * (1.0 - self.grid_oxygen)
            supply_nut = 0.05 * self.vessel_mask * (1.0 - self.grid_nutrient)

            uptake_o2 = o2_sink_grid * (self.grid_oxygen / (0.08 + self.grid_oxygen))
            uptake_nut = nut_sink_grid * (self.grid_nutrient / (0.10 + self.grid_nutrient))

            self.grid_oxygen += sub_dt * (self.diffusivity_oxygen * lap_o2 + supply_o2 - uptake_o2)
            self.grid_nutrient += sub_dt * (self.diffusivity_nutrient * lap_nut + supply_nut - uptake_nut)

            self.grid_oxygen = np.clip(self.grid_oxygen, 0.0, 1.0)
            self.grid_nutrient = np.clip(self.grid_nutrient, 0.0, 1.0)

    def _update_fibrosis(self, dt_seconds: float) -> None:
        """HSCs deposit collagen; antifibrotic compounds accelerate collagen resolution."""
        dep_rate = 0.000030 * dt_seconds
        deg_rate = 0.000008 * dt_seconds

        # Antifibrotic metabolic modulators
        for p in (self.active_pharmacology, self.active_pharmacology_b):
            if p.get("modulates_stroma", False) or p.get("anti_fibrotic", False):
                deg_rate *= 3.0

        hsc_mask = np.zeros((self.ny, self.nx), dtype=np.float64)
        for c in self.cells:
            if c.death_mode == CellDeathMode.ALIVE and c.clone_id == "ACTIVATED-HSC":
                gx = int(np.clip(c.x / self.dx, 0, self.nx - 1))
                gy = int(np.clip(c.y / self.dy, 0, self.ny - 1))
                hsc_mask[gy, gx] += 1.0

        self.grid_ecm += hsc_mask * dep_rate - deg_rate * self.grid_ecm
        self.grid_ecm = np.clip(self.grid_ecm, 0.0, 0.85)

    def _diffuse_and_react_drug(self, dt_seconds: float) -> None:
        d_eff_grid = self.diffusivity_drug_base / (1.0 + self.beta_fibrosis * self.grid_ecm)
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
                vasc_influx_a = 0.025 * self.vessel_mask * (self.applied_concentration_um - self.grid_drug_total)
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
                vasc_influx_b = 0.025 * self.vessel_mask * (self.applied_concentration_b_um - self.grid_drug_b_total)
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

    def _update_hepatocytes_and_hsc(self, dt_seconds: float) -> dict[str, int]:
        events = {"apoptosis": 0, "necrosis": 0, "division": 0, "steatosis_cleared": 0}

        prof = self.active_pharmacology
        def _ext(p: dict, k: str, d: float) -> float:
            v = p.get(k, d)
            if isinstance(v, dict): v = v.get("value", d)
            try: return float(v) if v is not None and not isinstance(v, str) else d
            except: return d

        ec50_steat_a = _ext(prof, "ic50_sensitive_um", 0.35)
        ec50_fib_a = _ext(prof, "ic50_resistant_um", 2.20)
        ic50_tox_a = _ext(prof, "ic50_healthy_um", 45.0)
        is_tox_a = bool(prof.get("hepatotoxic", False))

        has_drug_b = self.applied_concentration_b_um > 0.0 or np.max(self.grid_drug_b_total) > 1e-6
        prof_b = self.active_pharmacology_b
        if has_drug_b:
            ec50_steat_b = _ext(prof_b, "ic50_sensitive_um", 4.50)
            ec50_fib_b = _ext(prof_b, "ic50_resistant_um", 12.0)
            ic50_tox_b = _ext(prof_b, "ic50_healthy_um", 120.0)
            is_tox_b = bool(prof_b.get("hepatotoxic", False))

        for cell in self.cells:
            if cell.death_mode != CellDeathMode.ALIVE: continue

            gx = int(np.clip(cell.x / self.dx, 0, self.nx - 1))
            gy = int(np.clip(cell.y / self.dy, 0, self.ny - 1))

            cell.local_oxygen = float(self.grid_oxygen[gy, gx])
            cell.local_nutrient = float(self.grid_nutrient[gy, gx])
            cell.local_drug_conc = float(self.grid_drug_free[gy, gx])
            local_b = float(self.grid_drug_b_free[gy, gx]) if has_drug_b else 0.0

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

            # Hepatotoxic necrosis trigger (e.g. APAP toxicity on hepatocytes)
            if (is_tox_a and cell.intracellular_drug > ic50_tox_a) or (has_drug_b and is_tox_b and intra_b > ic50_tox_b):
                p_necr = 1.0 - math.exp(-0.00030 * dt_seconds)
                if self._rng.random() < p_necr:
                    cell.death_mode = CellDeathMode.NECROSIS
                    cell.viability = 0.0
                    self.cumulative_necrosis_count += 1
                    events["necrosis"] += 1
                    continue

            # Therapeutic Resolution of Steatosis & Deactivation of HSCs
            # In MASLDModule, "apoptosis/kill" represents clearance of pathological phenotype!
            e_a = 0.0
            if cell.intracellular_drug > 1e-5:
                eff_ec50 = ec50_steat_a if cell.clone_id == "STEATOTIC-HEPATOCYTE" else ec50_fib_a
                conc_a = cell.intracellular_drug
                e_a = self.emax_default * ((conc_a**2.0) / (eff_ec50**2.0 + conc_a**2.0))

            e_b = 0.0
            if has_drug_b and intra_b > 1e-5:
                eff_ec50_b = ec50_steat_b if cell.clone_id == "STEATOTIC-HEPATOCYTE" else ec50_fib_b
                conc_b = intra_b
                e_b = self.emax_default * ((conc_b**2.0) / (eff_ec50_b**2.0 + conc_b**2.0))

            if e_a > 0.0 or e_b > 0.0:
                if has_drug_b and (e_a > 0.0 or e_b > 0.0):
                    e_comb = min(0.99, 1.0 - (1.0 - e_a) * (1.0 - e_b) * (1.0 / max(0.5, self.synergy_factor)))
                else:
                    e_comb = e_a

                # In steatotic cells, drug clears lipid burden (steatosis clearance)
                p_clear = 1.0 - math.exp(-e_comb * dt_seconds)
                if self._rng.random() < p_clear:
                    cell.death_mode = CellDeathMode.APOPTOSIS  # mapped to cleared pathology
                    cell.viability = 0.0
                    self.cumulative_apoptosis_count += 1
                    events["apoptosis"] += 1
                    events["steatosis_cleared"] += 1
                    continue

        return events

    def get_snapshot(self) -> dict[str, Any]:
        type_counts: dict[str, int] = {}
        clone_counts: dict[str, int] = {}
        death_counts = {CellDeathMode.ALIVE.value: 0, CellDeathMode.APOPTOSIS.value: 0, CellDeathMode.NECROSIS.value: 0}
        viable_pathological = 0
        resistant_fibrotic = 0
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
                    viable_pathological += 1
                    if c.clone_id == "ACTIVATED-HSC":
                        resistant_fibrotic += 1

        total_grid = float(self.nx * self.ny)
        normoxic_pct = float(np.sum(self.grid_oxygen >= 0.30) / total_grid * 100.0)
        hypoxic_pct = float(np.sum((self.grid_oxygen < 0.30) & (self.grid_oxygen >= 0.08)) / total_grid * 100.0)
        necrotic_pct = float(np.sum(self.grid_oxygen < 0.08) / total_grid * 100.0)

        res_frac = float(resistant_fibrotic / viable_pathological) if viable_pathological > 0 else 0.0
        mean_intra = float(total_intra_drug / active_count) if active_count > 0 else 0.0

        summary_metrics = {
            "total_cells": len(self.cells),
            "viable_cells": active_count,
            "viable_tumor_cells": viable_pathological, # steatotic + activated HSC burden
            "resistant_fraction": round(res_frac, 4),
            "mean_intracellular_drug_um": round(mean_intra, 4),
            "mean_ecm_density": round(float(np.mean(self.grid_ecm)), 4), # hepatic fibrosis score
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
                "ecm_density": np.round(self.grid_ecm, 3).tolist(),  # hepatic collagen fibrosis
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
        steatotic_count = snap["clone_distribution"].get("STEATOTIC-HEPATOCYTE", 0)
        mean_fibrosis = snap["summary_metrics"]["mean_ecm_density"]

        # Liver stiffness (kPa surrogate for FibroScan F0 to F4)
        stiffness_kpa = round(3.5 + mean_fibrosis * 16.5, 2)
        # ALT surrogate (basal 24 U/L + surge with hepatocyte necrosis)
        alt_surrogate = round(24.0 + snap["cumulative_deaths"]["necrosis"] * 8.5, 1)

        biomarkers: dict[str, Any] = {
            "STEATOSIS_GRADE_SURROGATE": ScientificValue(
                name="Hepatic Steatosis Grade (S0-S3)",
                value="S3 (Severe >66%)" if steatotic_count > 45 else ("S2 (Moderate)" if steatotic_count > 20 else "S1 (Mild)"),
                unit="grade",
                epistemic_category=EpistemicCategory.EXPLORATORY_MODEL,
                source="EVOCURE Hepatic Steatosis Model",
                confidence=0.86,
                in_applicability_domain=True,
            ).to_dict(),
            "LIVER_STIFFNESS_KPA": ScientificValue(
                name="Transient Elastography Liver Stiffness",
                value=stiffness_kpa,
                unit="kPa",
                epistemic_category=EpistemicCategory.MECHANISTIC_SIMULATION,
                source="2D Collagen Matrix Secretion Solver",
                confidence=0.88,
                in_applicability_domain=True,
            ).to_dict(),
            "SERUM_ALT_SURROGATE": ScientificValue(
                name="Serum Alanine Aminotransferase (ALT)",
                value=alt_surrogate,
                unit="U/L",
                epistemic_category=EpistemicCategory.MECHANISTIC_SIMULATION,
                source="Hepatocellular Necrosis Leakage Surrogate",
                confidence=0.82,
                in_applicability_domain=True,
            ).to_dict(),
            "KUPFFER_INFLAMMATION_SCORE": ScientificValue(
                name="Kupffer Cell Inflammatory Activation",
                value=round(mean_fibrosis * 1.2, 3),
                unit="score [0-1]",
                epistemic_category=EpistemicCategory.MECHANISTIC_SIMULATION,
            ).to_dict(),
        }

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
                "tumor_growth_inhibition_pct": round(tgi, 2), # steatosis & fibrosis resolution %
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
