"""
7-Compartment Systemic Physiologically-Based Pharmacokinetic (PBPK) Module for EVOCURE.

Implements Section 13 & Mathematical Specification Section 5.2:
Coupled ODE system across 7 anatomical compartments:
1. Blood (vascular pool)
2. Liver
3. Kidneys
4. Bone Marrow
5. Pancreas
6. Tumor (PDAC primary lesion)
7. Peripheral (adipose, muscle, skin, viscera)

Governing mass-balance differential equations:
V_blood * dC_blood/dt = sum_{k != blood} Q_k * (C_k / K_p,k - C_blood)
                        - CL_hep * fu * C_blood - CL_ren * fu * C_blood + I_dose(t)

V_k * dC_k/dt = Q_k * (C_blood - C_k / K_p,k)  for each tissue compartment k

Solved numerically via scipy.integrate.solve_ivp (RK45).
"""

from __future__ import annotations

import math
from typing import Any, Literal
import numpy as np
from scipy.integrate import solve_ivp

from simulation.core.contracts import (
    CompoundRecord,
    ScientificValue,
    EpistemicCategory,
)

PBPK_SAFETY_DISCLAIMER: str = (
    "DISCLAIMER: Exploratory computational PBPK simulation for preclinical research only. "
    "Does not constitute medical advice or human clinical dosing guidance."
)

# Standard reference physiological values for an adult 70 kg human
PHYSIOLOGICAL_VOLUMES_L: dict[str, float] = {
    "blood": 5.0,        # Total vascular volume (arterial + venous)
    "liver": 1.8,        # Hepatic tissue
    "kidneys": 0.31,     # Renal tissue
    "bone_marrow": 1.5,  # Active red/yellow bone marrow
    "pancreas": 0.10,    # Pancreatic parenchyma
    "tumor": 0.05,       # 50 mL primary pancreatic tumor mass
    "peripheral": 55.0,  # Rest of body (muscle, fat, bone, viscera)
}

PHYSIOLOGICAL_FLOWS_L_PER_H: dict[str, float] = {
    "liver": 90.0,       # Hepatic arterial + portal inflow (~30% cardiac output)
    "kidneys": 60.0,     # Renal perfusion (~20% cardiac output)
    "bone_marrow": 15.0, # Bone marrow perfusion (~5% cardiac output)
    "pancreas": 6.0,     # Pancreatic perfusion (~2% cardiac output)
    "tumor": 1.5,        # Desmoplastic hypoperfused PDAC microvasculature (~0.5%)
    "peripheral": 127.5, # Remainder of systemic cardiac output (~42.5%)
}

# Total cardiac output = 90 + 60 + 15 + 6 + 1.5 + 127.5 = 300.0 L/h (5.0 L/min)
CARDIAC_OUTPUT_L_PER_H: float = 300.0

# Glomerular filtration rate = 125 mL/min = 7.5 L/h
GLOMERULAR_FILTRATION_RATE_L_PER_H: float = 7.5


class SystemicPBPKEngine:
    """
    7-Compartment PBPK Numerical Simulation Engine.
    """

    def __init__(self, model_version: str = "0.1.0") -> None:
        self.model_version = model_version
        self.volumes = dict(PHYSIOLOGICAL_VOLUMES_L)
        self.flows = dict(PHYSIOLOGICAL_FLOWS_L_PER_H)

    def calculate_partition_coefficients(
        self,
        logp: float,
        fu: float,
    ) -> dict[str, float]:
        """
        Estimate tissue-to-blood partition coefficients (Kp) from lipophilicity (LogP)
        and plasma unbound fraction (fu).
        """
        fu_safe = max(0.01, min(0.99, fu))
        logp_pos = max(0.0, logp)

        # Rodgers-Rowland / Poulin-Theil simplified empirical organ partition factors
        # Lipophilic molecules accumulate in liver, peripheral adipose/tissue
        # Polar compounds partition primarily according to tissue water fraction
        kp_liver = max(0.2, (0.7 + 0.35 * logp_pos) * (0.3 + 0.7 / fu_safe) ** 0.25)
        kp_kidneys = max(0.2, 0.8 + 0.25 * logp_pos)
        kp_bone_marrow = max(0.1, 0.6 + 0.20 * logp_pos)
        kp_pancreas = max(0.1, 0.7 + 0.25 * logp_pos)
        # Tumor penetration is attenuated by desmoplastic stroma and interstitial fluid pressure
        kp_tumor = max(0.05, 0.45 + 0.15 * logp_pos)
        kp_peripheral = max(0.1, 0.55 + 0.40 * logp_pos)

        return {
            "liver": round(float(min(20.0, kp_liver)), 3),
            "kidneys": round(float(min(15.0, kp_kidneys)), 3),
            "bone_marrow": round(float(min(12.0, kp_bone_marrow)), 3),
            "pancreas": round(float(min(12.0, kp_pancreas)), 3),
            "tumor": round(float(min(8.0, kp_tumor)), 3),
            "peripheral": round(float(min(25.0, kp_peripheral)), 3),
        }

    def calculate_clearances(
        self,
        fu: float,
        metabolic_t12_hours: float = 6.0,
    ) -> dict[str, float]:
        """
        Compute hepatic and renal elimination clearances (L/h).
        - Renal: GFR * fu (filtration only, baseline assumption)
        - Hepatic: Well-stirred model using intrinsic clearance CL_int,hep
        """
        fu_safe = max(0.01, min(0.99, fu))
        t12_safe = max(0.1, metabolic_t12_hours)

        # Renal clearance: GFR * fu
        cl_ren = GLOMERULAR_FILTRATION_RATE_L_PER_H * fu_safe

        # In vitro to whole-liver intrinsic clearance scaling
        # Faster metabolic degradation (lower t1/2) yields higher CL_int
        cl_int_hep = max(1.0, (self.volumes["liver"] * math.log(2) / t12_safe) * 20.0)

        # Well-stirred hepatic clearance: Q_hep * fu * CL_int / (Q_hep + fu * CL_int)
        q_liver = self.flows["liver"]
        cl_hep = (q_liver * fu_safe * cl_int_hep) / (q_liver + fu_safe * cl_int_hep)

        cl_tot = cl_hep + cl_ren

        return {
            "cl_hepatic_l_per_h": round(float(cl_hep), 3),
            "cl_renal_l_per_h": round(float(cl_ren), 3),
            "cl_total_l_per_h": round(float(cl_tot), 3),
            "cl_intrinsic_hepatic_l_per_h": round(float(cl_int_hep), 3),
        }

    def simulate(
        self,
        compound: CompoundRecord,
        dose_mg: float = 100.0,
        administration_route: Literal["iv_bolus", "iv_infusion", "oral"] = "iv_bolus",
        infusion_duration_hours: float = 1.0,
        duration_hours: float = 24.0,
        num_time_points: int = 240,
        metabolic_t12_hours: float = 6.0,
    ) -> dict[str, Any]:
        """
        Solve the 7-compartment PBPK ODE system over [0, duration_hours].
        Units:
        - Concentrations: uM (micromolar)
        - Time: hours
        - Dose: converted from mg to umol via molecular weight
        """
        mw = max(10.0, float(compound.mol_weight))
        fu = float(compound.reference_bioactivity.get("fraction_unbound", 0.15))
        logp = float(compound.logp)

        # Partition coefficients and clearances
        kp = self.calculate_partition_coefficients(logp, fu)
        clearance = self.calculate_clearances(fu, metabolic_t12_hours)

        cl_hep = clearance["cl_hepatic_l_per_h"]
        cl_ren = clearance["cl_renal_l_per_h"]

        # Dose conversion: mg -> umol
        # dose_umol = (dose_mg / (mw g/mol)) * 1000 umol/mmol
        dose_umol = (dose_mg / mw) * 1000.0

        v_blood = self.volumes["blood"]
        v_liver = self.volumes["liver"]
        v_kidneys = self.volumes["kidneys"]
        v_marrow = self.volumes["bone_marrow"]
        v_panc = self.volumes["pancreas"]
        v_tumor = self.volumes["tumor"]
        v_periph = self.volumes["peripheral"]

        q_liver = self.flows["liver"]
        q_kidneys = self.flows["kidneys"]
        q_marrow = self.flows["bone_marrow"]
        q_panc = self.flows["pancreas"]
        q_tumor = self.flows["tumor"]
        q_periph = self.flows["peripheral"]

        kp_liver = kp["liver"]
        kp_kidneys = kp["kidneys"]
        kp_marrow = kp["bone_marrow"]
        kp_panc = kp["pancreas"]
        kp_tumor = kp["tumor"]
        kp_periph = kp["peripheral"]

        # Initial conditions: [C_blood, C_liver, C_kidneys, C_marrow, C_panc, C_tumor, C_periph, A_gut]
        # State index mapping:
        # 0: blood, 1: liver, 2: kidneys, 3: bone_marrow, 4: pancreas, 5: tumor, 6: peripheral, 7: gut_depot
        y0 = np.zeros(8, dtype=np.float64)

        ka_oral = 1.2  # 1/h absorption rate constant
        f_oral = 0.65  # 65% oral bioavailability fraction

        if administration_route == "iv_bolus":
            y0[0] = dose_umol / v_blood
        elif administration_route == "oral":
            y0[7] = dose_umol  # umol deposited in gut lumen
        # For iv_infusion, dose is delivered over infusion_duration_hours in the ODE function

        def ode_system(t: float, y: np.ndarray) -> np.ndarray:
            c_b = max(0.0, y[0])
            c_liv = max(0.0, y[1])
            c_kid = max(0.0, y[2])
            c_mar = max(0.0, y[3])
            c_pan = max(0.0, y[4])
            c_tum = max(0.0, y[5])
            c_per = max(0.0, y[6])
            a_gut = max(0.0, y[7])

            # Dose input rate I_dose(t) in umol/h
            i_dose = 0.0
            if administration_route == "iv_infusion" and t <= infusion_duration_hours:
                i_dose = dose_umol / max(0.01, infusion_duration_hours)
            elif administration_route == "oral":
                i_dose = ka_oral * a_gut * f_oral

            # Sum of returning flows from organs: sum Q_k * (C_k / Kp_k - C_blood)
            # which is (sum Q_k * C_k / Kp_k) - (sum Q_k) * C_blood
            return_flow = (
                q_liver * (c_liv / kp_liver - c_b)
                + q_kidneys * (c_kid / kp_kidneys - c_b)
                + q_marrow * (c_mar / kp_marrow - c_b)
                + q_panc * (c_pan / kp_panc - c_b)
                + q_tumor * (c_tum / kp_tumor - c_b)
                + q_periph * (c_per / kp_periph - c_b)
            )

            # Elimination from blood
            elimination = (cl_hep * fu * c_b) + (cl_ren * fu * c_b)

            dc_b_dt = (return_flow - elimination + i_dose) / v_blood

            # Organ compartment ODEs: V_k * dC_k/dt = Q_k * (C_blood - C_k / Kp_k)
            dc_liv_dt = (q_liver * (c_b - c_liv / kp_liver)) / v_liver
            dc_kid_dt = (q_kidneys * (c_b - c_kid / kp_kidneys)) / v_kidneys
            dc_mar_dt = (q_marrow * (c_b - c_mar / kp_marrow)) / v_marrow
            dc_pan_dt = (q_panc * (c_b - c_pan / kp_panc)) / v_panc
            dc_tum_dt = (q_tumor * (c_b - c_tum / kp_tumor)) / v_tumor
            dc_per_dt = (q_periph * (c_b - c_per / kp_periph)) / v_periph

            # Gut absorption depot
            da_gut_dt = -ka_oral * a_gut if administration_route == "oral" else 0.0

            return np.array([
                dc_b_dt,
                dc_liv_dt,
                dc_kid_dt,
                dc_mar_dt,
                dc_pan_dt,
                dc_tum_dt,
                dc_per_dt,
                da_gut_dt,
            ])

        t_eval = np.linspace(0, duration_hours, num_time_points)
        sol = solve_ivp(
            fun=ode_system,
            t_span=(0, duration_hours),
            y0=y0,
            method="RK45",
            t_eval=t_eval,
            rtol=1e-5,
            atol=1e-8,
        )

        time_points = sol.t.tolist()
        c_blood = np.maximum(0.0, sol.y[0]).tolist()
        c_liver = np.maximum(0.0, sol.y[1]).tolist()
        c_kidneys = np.maximum(0.0, sol.y[2]).tolist()
        c_marrow = np.maximum(0.0, sol.y[3]).tolist()
        c_pancreas = np.maximum(0.0, sol.y[4]).tolist()
        c_tumor = np.maximum(0.0, sol.y[5]).tolist()
        c_peripheral = np.maximum(0.0, sol.y[6]).tolist()

        # Compute Cmax for each compartment
        cmax_dict = {
            "blood": round(float(max(c_blood)), 4),
            "liver": round(float(max(c_liver)), 4),
            "kidneys": round(float(max(c_kidneys)), 4),
            "bone_marrow": round(float(max(c_marrow)), 4),
            "pancreas": round(float(max(c_pancreas)), 4),
            "tumor": round(float(max(c_tumor)), 4),
            "peripheral": round(float(max(c_peripheral)), 4),
        }

        # Compute AUC using trapezoidal integration
        t_arr = np.array(time_points)
        auc_dict = {
            "blood": round(float(np.trapezoid(c_blood, t_arr)), 3),
            "liver": round(float(np.trapezoid(c_liver, t_arr)), 3),
            "kidneys": round(float(np.trapezoid(c_kidneys, t_arr)), 3),
            "bone_marrow": round(float(np.trapezoid(c_marrow, t_arr)), 3),
            "pancreas": round(float(np.trapezoid(c_pancreas, t_arr)), 3),
            "tumor": round(float(np.trapezoid(c_tumor, t_arr)), 3),
            "peripheral": round(float(np.trapezoid(c_peripheral, t_arr)), 3),
        }

        # Estimate terminal elimination half-life from blood curve
        # Use last 30% of time course where concentrations are decaying log-linearly
        terminal_cutoff = int(len(c_blood) * 0.70)
        t_term = t_arr[terminal_cutoff:]
        c_term = np.array(c_blood[terminal_cutoff:])
        valid_idx = c_term > 1e-4

        half_life_hours = None
        if np.sum(valid_idx) >= 5:
            log_c = np.log(c_term[valid_idx])
            slope, _ = np.polyfit(t_term[valid_idx], log_c, 1)
            if slope < -1e-5:
                half_life_hours = round(float(-math.log(2) / slope), 2)

        if half_life_hours is None or half_life_hours <= 0:
            # Fallback estimation from total clearance and volume of distribution
            vd_ss = sum(self.volumes[k] * kp.get(k, 1.0) for k in kp) + v_blood
            half_life_hours = round(float(math.log(2) * vd_ss / max(0.1, clearance["cl_total_l_per_h"])), 2)

        # Organ exposure ratios
        auc_b = max(1e-4, auc_dict["blood"])
        auc_mar = max(1e-4, auc_dict["bone_marrow"])
        exposure_ratios = {
            "tumor_to_blood": round(auc_dict["tumor"] / auc_b, 3),
            "tumor_to_bone_marrow": round(auc_dict["tumor"] / auc_mar, 3),
            "liver_to_blood": round(auc_dict["liver"] / auc_b, 3),
            "kidneys_to_blood": round(auc_dict["kidneys"] / auc_b, 3),
            "pancreas_to_blood": round(auc_dict["pancreas"] / auc_b, 3),
        }

        return {
            "compound_id": compound.compound_id,
            "dose_mg": dose_mg,
            "dose_umol": round(dose_umol, 3),
            "administration_route": administration_route,
            "duration_hours": duration_hours,
            "time_points_hours": [round(t, 3) for t in time_points],
            "concentrations": {
                "blood": [round(c, 4) for c in c_blood],
                "liver": [round(c, 4) for c in c_liver],
                "kidneys": [round(c, 4) for c in c_kidneys],
                "bone_marrow": [round(c, 4) for c in c_marrow],
                "pancreas": [round(c, 4) for c in c_pancreas],
                "tumor": [round(c, 4) for c in c_tumor],
                "peripheral": [round(c, 4) for c in c_peripheral],
            },
            "cmax": cmax_dict,
            "auc": auc_dict,
            "elimination_half_life_hours": ScientificValue(
                name="Systemic Elimination Half-life (t1/2)",
                value=half_life_hours,
                unit="hours",
                epistemic_category=EpistemicCategory.MECHANISTIC_SIMULATION,
                source="7-Compartment PBPK Numerical Simulation",
                confidence=0.82,
                uncertainty_low=round(half_life_hours * 0.70, 2),
                uncertainty_high=round(half_life_hours * 1.35, 2),
                in_applicability_domain=True,
            ).to_dict(),
            "clearances": clearance,
            "partition_coefficients_kp": kp,
            "organ_exposure_ratios": exposure_ratios,
            "volumes_l": self.volumes,
            "flows_l_per_h": self.flows,
            "epistemic_category": EpistemicCategory.MECHANISTIC_SIMULATION.value,
            "safety_disclaimer": PBPK_SAFETY_DISCLAIMER,
        }
