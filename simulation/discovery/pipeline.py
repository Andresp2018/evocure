"""
EVOCURE 15-Stage Candidate Discovery Pipeline (Section 15).

Implements the complete 15-stage workflow integrating:
1. Disease & experimental context selection (e.g. PDAC).
2. Initial seed compound import & lookup.
3. Structural variant generation (point mutations & scaffold recombinations).
4. Strict RDKit chemical validation (quarantining Mode A artificial traits).
5. Physicochemical descriptor calculation (MW, LogP, TPSA, HBD, HBA, QED).
6. Calibrated QSAR activity prediction (pIC50, growth-inhibition probability, 95% CI).
7. ADMET & 10-category toxicology risk evaluation.
8. Applicability domain check (Tanimoto max similarity to reference set).
9. Multi-objective evaluation (potency, selectivity index, QED, low tox, domain confidence).
10. Progressive triage via LayaDecisionEngine.
11. Controlled 2D PDAC microenvironment simulation (vs untreated control).
12. Result & provenance storage.
13. Pareto-optimal candidate identification (NSGA-II non-dominated sorting).
14. Evolutionary population update.
15. Next generation loop with full trace, control comparisons, and epistemic disclaimers.
"""

from __future__ import annotations

import math
import uuid
from typing import Any, Callable

import numpy as np
from rdkit import Chem

from ai.laya import LayaDecisionEngine
from ai.predictors import CalibratedQSARPredictor
from simulation.biology.diseases.pdac import PDACModule
from simulation.biology.registry import DiseaseRegistry
from simulation.chemistry.engine import ChemistryEngine
from simulation.core.contracts import (
    CompoundRecord,
    EpistemicCategory,
    ScientificValue,
    UniverseMode,
)
from simulation.evolution.molecular_ga import DirectedMolecularOptimizer
from simulation.pharmacology.admet import (
    PharmacologyEngine,
    validate_for_biomedical_mode,
)
from simulation.toxicity.models import ToxicologyEngine

NON_CLINICAL_DISCLAIMER: str = (
    "DISCLAIMER: In silico computational simulation for research and exploratory ranking only. "
    "Does not constitute medical evidence, pharmacological guidance, clinical trial design, "
    "or human dosing recommendations. Never maximize tumor cell kill alone without balancing "
    "toxicology, selectivity, and drug-likeness."
)

DEFAULT_SEEDS: list[str] = [
    "Nc1ccn([C@@H]2O[C@H](CO)[C@@H](O)C2(F)F)c(=O)n1",  # Gemcitabine
    "O=c1[nH]cc(F)c(=O)[nH]1",                          # 5-Fluorouracil
    "COc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC",             # Erlotinib
    "CC(=O)Oc1ccccc1C(=O)O",                             # Aspirin
]


class CandidateDiscoveryPipeline:
    """
    Executes the 15-stage multi-objective candidate discovery and controlled
    cellular evaluation pipeline conforming to EVOCURE scientific contracts.
    """

    def __init__(
        self,
        disease_id: str = "PDAC",
        chem_engine: ChemistryEngine | None = None,
        qsar_predictor: CalibratedQSARPredictor | None = None,
        pharmacology_engine: PharmacologyEngine | None = None,
        toxicology_engine: ToxicologyEngine | None = None,
        decision_engine: LayaDecisionEngine | None = None,
        disease_registry: DiseaseRegistry | None = None,
        molecular_optimizer: DirectedMolecularOptimizer | None = None,
    ) -> None:
        self.disease_id = disease_id
        self.chem_engine = chem_engine or ChemistryEngine()
        self.pharmacology_engine = pharmacology_engine or PharmacologyEngine()
        self.toxicology_engine = toxicology_engine or ToxicologyEngine()
        self.decision_engine = decision_engine or LayaDecisionEngine()
        self.disease_registry = disease_registry or DiseaseRegistry()
        self.molecular_optimizer = molecular_optimizer or DirectedMolecularOptimizer()

        # Initialize or load QSAR model
        if qsar_predictor is not None:
            self.qsar_predictor = qsar_predictor
        else:
            try:
                self.qsar_predictor = CalibratedQSARPredictor().load()
            except Exception:
                self.qsar_predictor = CalibratedQSARPredictor()

    def run(
        self,
        seed_queries: list[str] | None = None,
        generations: int = 4,
        population_size: int = 16,
        top_k_cellular: int = 3,
        applied_concentration_um: float = 5.0,
        duration_hours: float = 48.0,
        seed: int = 1337,
        progress_callback: Callable | None = None,
        objective_weights: dict[str, float] | None = None,
        candidate_count: int = 5,
    ) -> dict[str, Any]:
        """
        Execute the full 15-stage iterative discovery pipeline across the specified
        number of evolutionary generations.
        """
        rng = np.random.default_rng(seed)
        if self.disease_id != "PDAC":
            objective_weights = {"predicted_pdac_activity": 0.0, "selectivity_index": 0.0,
                                 "qed_druglikeness": 1.0, "low_toxicological_risk": 1.0,
                                 "applicability_domain_confidence": 1.0, **(objective_weights or {})}
            objective_weights["predicted_pdac_activity"] = 0.0
            objective_weights["selectivity_index"] = 0.0
        class ProgressTrace(list):
            def append(self, item):
                super().append(item)
                if progress_callback:
                    progress_callback({"stage": item["stage_number"], "stage_name": item["stage_name"],
                                       "generation": item["generation"], "target_generations": generations})
        stage_trace: list[dict[str, Any]] = ProgressTrace()
        all_evaluated: dict[str, dict[str, Any]] = {}
        all_controlled_comparisons: list[dict[str, Any]] = []

        # ---------------------------------------------------------------------
        # STAGE 1: Disease & Experimental Context Selection
        # ---------------------------------------------------------------------
        disease_meta = {}
        if self.disease_registry.is_registered(self.disease_id):
            mod_cls = self.disease_registry.get_class(self.disease_id)
            tmp_mod = mod_cls(seed=seed)
            disease_meta = tmp_mod.metadata()
        else:
            disease_meta = {
                "disease_id": self.disease_id,
                "display_name": "Pancreatic Ductal Adenocarcinoma (PDAC)",
                "context": "Virtual 2D microenvironment with hypovascularity and dense desmoplasia.",
            }

        stage_trace.append({
            "stage_number": 1,
            "stage_name": "Disease & Experimental Context Selection",
            "generation": 0,
            "details": {
                "disease_id": self.disease_id,
                "disease_metadata": disease_meta,
                "target_clones": ["Classical Sensitive (KRAS/TP53)", "Basal Resistant (KRAS/TP53/CDKN2A/SMAD4)"],
                "microenvironment_factors": ["Reaction-diffusion O2", "Dense Collagen ECM", "Hypovascularity"],
            },
        })

        # ---------------------------------------------------------------------
        # STAGE 2: Initial Seed Compound Import & Lookup
        # ---------------------------------------------------------------------
        raw_seeds = seed_queries or DEFAULT_SEEDS
        current_compounds: list[CompoundRecord] = []
        for s in raw_seeds:
            try:
                rec = self.chem_engine.resolve_compound_input(s)
                current_compounds.append(rec)
            except Exception:
                # If direct lookup fails, try building from SMILES
                try:
                    rec = self.chem_engine.build_compound(s, provenance="Seed Scaffold")
                    current_compounds.append(rec)
                except Exception:
                    pass

        if not current_compounds:
            # Fallback guarantee
            current_compounds.append(self.chem_engine.build_compound("CC(=O)Oc1ccccc1C(=O)O", name="Aspirin"))

        stage_trace.append({
            "stage_number": 2,
            "stage_name": "Initial Seed Compound Import & Lookup",
            "generation": 0,
            "details": {
                "initial_seed_count": len(current_compounds),
                "seed_compounds": [
                    {"id": c.compound_id, "name": c.name, "smiles": c.canonical_smiles}
                    for c in current_compounds
                ],
            },
        })

        # Population of canonical SMILES strings to evolve
        current_smiles_pop: list[str] = [c.canonical_smiles for c in current_compounds if c.canonical_smiles]
        seed_smiles = set(current_smiles_pop)
        structure_ids = {c.canonical_smiles: c.compound_id for c in current_compounds}

        # Begin Evolutionary Multi-Generation Loop (Stages 3 to 15)
        for gen_idx in range(max(1, generations)):
            gen_number = gen_idx + 1

            # -----------------------------------------------------------------
            # STAGE 3: Structural Variant Generation
            # -----------------------------------------------------------------
            candidates_smiles: list[str] = []
            variants_provenance: list[dict[str, str]] = []
            variant_parents: dict[str, list[str]] = {}

            # Retain current population members
            for s in current_smiles_pop:
                candidates_smiles.append(s)
                variants_provenance.append({"smiles": s, "type": "parent_retention", "source": "population"})

            # Generate offspring through point mutations and recombinations
            num_to_generate = max(8, population_size * 2 - len(candidates_smiles))
            for _ in range(num_to_generate):
                use_crossover = len(current_smiles_pop) >= 2 and rng.random() < 0.65
                if use_crossover:
                    p1, p2 = rng.choice(current_smiles_pop, size=2, replace=False)
                    child_smiles, desc = self.molecular_optimizer.recombine_smiles(p1, p2, rng=rng)
                    candidates_smiles.append(child_smiles)
                    variant_parents[child_smiles] = [structure_ids[s] for s in (str(p1), str(p2)) if s in structure_ids]
                    variants_provenance.append({"smiles": child_smiles, "type": "recombination", "source": desc})
                else:
                    parent = str(rng.choice(current_smiles_pop))
                    child_smiles, desc = self.molecular_optimizer.mutate_smiles(parent, rng=rng)
                    candidates_smiles.append(child_smiles)
                    variant_parents[child_smiles] = [structure_ids[parent]] if parent in structure_ids else []
                    variants_provenance.append({"smiles": child_smiles, "type": "mutation", "source": desc})

            stage_trace.append({
                "stage_number": 3,
                "stage_name": "Structural Variant Generation",
                "generation": gen_number,
                "details": {
                    "total_generated_variants": len(candidates_smiles),
                    "parent_pool_size": len(current_smiles_pop),
                    "provenance_sample": variants_provenance[:5],
                },
            })

            # -----------------------------------------------------------------
            # STAGE 4: Strict RDKit Chemical Validation
            # -----------------------------------------------------------------
            valid_compounds: list[CompoundRecord] = []
            quarantined_count: int = 0

            for sm in candidates_smiles:
                mol = Chem.MolFromSmiles(sm)
                if mol is None:
                    quarantined_count += 1
                    continue

                can_smi = Chem.MolToSmiles(mol, canonical=True)
                # Build candidate record
                rec = self.chem_engine.build_compound(
                    smiles=can_smi,
                    parent_ids=variant_parents.get(sm, []),
                    origin_generation=gen_number,
                    transformation_type="directed_optimization",
                    provenance="Section 15 Discovery Pipeline",
                )
                rec.parent_ids = [parent_id for parent_id in rec.parent_ids if parent_id != rec.compound_id]
                structure_ids[can_smi] = rec.compound_id

                # Strict Mode B validation (quarantining Mode A traits)
                val_res = validate_for_biomedical_mode(rec, source_mode=UniverseMode.GROUNDED_CHEMISTRY)
                if val_res.get("is_valid", False):
                    valid_compounds.append(rec)
                else:
                    quarantined_count += 1

            # De-duplicate by canonical SMILES
            unique_compounds: dict[str, CompoundRecord] = {}
            for c in valid_compounds:
                if c.canonical_smiles not in unique_compounds:
                    unique_compounds[c.canonical_smiles] = c
            valid_compounds = list(unique_compounds.values())

            stage_trace.append({
                "stage_number": 4,
                "stage_name": "Strict RDKit Chemical Validation",
                "generation": gen_number,
                "details": {
                    "valid_chemical_structures": len(valid_compounds),
                    "quarantined_invalid_structures": quarantined_count,
                    "validation_mode": "Strict RDKit Graph + Valence Enforcement",
                },
            })

            # -----------------------------------------------------------------
            # STAGE 5: Physicochemical Descriptor Calculation
            # -----------------------------------------------------------------
            # Descriptors already computed during build_compound: MW, LogP, TPSA, HBD, HBA, QED
            stage_trace.append({
                "stage_number": 5,
                "stage_name": "Physicochemical Descriptor Calculation",
                "generation": gen_number,
                "details": {
                    "descriptors_computed": len(valid_compounds),
                    "metrics": ["MW", "LogP", "TPSA", "HBD", "HBA", "RotBonds", "HeavyAtoms", "QED"],
                },
            })

            # -----------------------------------------------------------------
            # STAGE 6: Calibrated QSAR Activity Prediction
            # -----------------------------------------------------------------
            qsar_predictions: dict[str, dict[str, Any]] = {}
            for comp in valid_compounds:
                sm = comp.canonical_smiles
                try:
                    if self.disease_id != "PDAC":
                        raise ValueError("No disease-specific activity predictor is available")
                    pred = self.qsar_predictor.predict(sm)
                except Exception as e:
                    pred = {
                        "is_chemically_valid": True,
                        "smiles": sm,
                        "in_applicability_domain": False,
                        "tanimoto_similarity_max": 0.0,
                        "pic50": {"value": None},
                        "ic50_um_estimated": None,
                        "active_probability": {"value": None},
                        "unavailable_reason": str(e),
                    }
                qsar_predictions[sm] = pred

            stage_trace.append({
                "stage_number": 6,
                "stage_name": "Calibrated QSAR Activity Prediction",
                "generation": gen_number,
                "details": {
                    "compounds_predicted": len(qsar_predictions),
                    "target_endpoint": "Exploratory PDAC pIC50" if self.disease_id == "PDAC" else "Unavailable: chemistry-only ranking for this disease",
                    "calibration_method": "Sigmoidal CalibratedClassifierCV + RandomForest 95% CI",
                },
            })

            # -----------------------------------------------------------------
            # STAGE 7: ADMET & 10-Category Toxicology Risk Evaluation
            # -----------------------------------------------------------------
            admet_profiles: dict[str, dict[str, Any]] = {}
            tox_profiles: dict[str, dict[str, Any]] = {}

            for comp in valid_compounds:
                sm = comp.canonical_smiles
                q_res = qsar_predictions.get(sm, {})
                in_domain = bool(q_res.get("in_applicability_domain", True))
                s_max = float(q_res.get("tanimoto_similarity_max", 0.5))

                # ADMET
                admet = self.pharmacology_engine.evaluate_admet(
                    compound=comp, in_domain=in_domain, domain_similarity=s_max
                )
                admet_profiles[sm] = admet

                # 10-Category Toxicology
                tox = self.toxicology_engine.evaluate_toxicology(
                    compound=comp, in_domain=in_domain
                )
                tox_profiles[sm] = tox

            stage_trace.append({
                "stage_number": 7,
                "stage_name": "ADMET & 10-Category Toxicology Risk Evaluation",
                "generation": gen_number,
                "details": {
                    "profiles_evaluated": len(admet_profiles),
                    "toxicology_categories_checked": [
                        "DILI/Hepatotoxicity", "hERG/Cardiotoxicity", "Ames/Mutagenicity",
                        "Nephrotoxicity", "Hematotoxicity", "Pfizer 3/75 Rule",
                        "Reactive Metabolites", "Off-target Liability", "Structural Alerts",
                        "Aggregate Computational Risk Index",
                    ],
                },
            })

            # -----------------------------------------------------------------
            # STAGE 8: Applicability Domain Check
            # -----------------------------------------------------------------
            domain_assessments: dict[str, dict[str, Any]] = {}
            in_domain_count = 0
            for comp in valid_compounds:
                sm = comp.canonical_smiles
                q_res = qsar_predictions.get(sm, {})
                in_ad = bool(q_res.get("in_applicability_domain", False))
                s_max = float(q_res.get("tanimoto_similarity_max", 0.0))
                if in_ad:
                    in_domain_count += 1
                domain_assessments[sm] = {
                    "in_domain": in_ad,
                    "max_similarity": s_max,
                    "epistemic_tag": (
                        EpistemicCategory.VALIDATED_MODEL_PREDICTION.value
                        if in_ad
                        else EpistemicCategory.EXPLORATORY_MODEL.value
                    ),
                }

            stage_trace.append({
                "stage_number": 8,
                "stage_name": "Applicability Domain Check",
                "generation": gen_number,
                "details": {
                    "total_checked": len(valid_compounds),
                    "in_domain_count": in_domain_count,
                    "out_of_domain_flagged": len(valid_compounds) - in_domain_count,
                    "threshold_tanimoto": 0.25,
                },
            })

            # -----------------------------------------------------------------
            # STAGE 9: Multi-Objective Evaluation
            # -----------------------------------------------------------------
            # MANDATORY RULE: NEVER maximize tumor cell kill alone!
            # Evaluate balanced multi-objective criteria across all 5 dimensions.
            evaluated_candidates: list[dict[str, Any]] = []
            for comp in valid_compounds:
                sm = comp.canonical_smiles
                q_res = qsar_predictions.get(sm, {})
                admet = admet_profiles.get(sm, {})
                tox = tox_profiles.get(sm, {})
                dom = domain_assessments.get(sm, {})

                pic50_dict = q_res.get("pic50", {})
                pic50_raw = pic50_dict.get("value")
                pic50_val = float(pic50_raw) if isinstance(pic50_raw, (int, float)) else None

                # Objective 1: Predicted Potency (sigmoidal utility centered at pIC50 = 6.0)
                potency_obj = 1.0 / (1.0 + math.exp(-1.0 * (pic50_val - 6.0))) if pic50_val is not None else 0.0

                # Objective 2: Selectivity Index (healthy cell vs tumor sparing estimate)
                sel_val = 1.0
                if "selectivity_index" in admet:
                    sel_val = float(admet["selectivity_index"].get("value") or 1.0)
                elif "therapeutic_window_score" in tox:
                    sel_val = float(tox["therapeutic_window_score"].get("value") or 1.0)
                sel_obj = min(1.0, max(0.0, math.log10(max(1.0, sel_val)) / 1.5)) if pic50_val is not None else 0.0

                # Objective 3: Drug-likeness (QED)
                qed_obj = float(comp.qed)

                # Objective 4: Low Toxicological Risk
                raw_risk = tox.get("internal_computational_risk_index_0_100")
                risk_idx = float(raw_risk) if isinstance(raw_risk, (int, float)) else None
                low_tox_obj = max(0.0, min(1.0, 1.0 - (risk_idx / 100.0))) if risk_idx is not None else 0.0

                # Objective 5: Applicability Domain Confidence
                s_max = float(dom.get("max_similarity", 0.3))
                domain_obj = max(0.0, min(1.0, s_max))

                candidate_entry = {
                    "compound_id": comp.compound_id,
                    "name": comp.name,
                    "canonical_smiles": sm,
                    "compound_record": comp,
                    "pic50": pic50_val,
                    "ic50_um": q_res.get("ic50_um_estimated"),
                    "qed": qed_obj,
                    "selectivity_hpde": sel_val,
                    "hepatotox_alert": bool(tox.get("hepatotoxicity_risk", {}).get("risk_category") == "HIGH"),
                    "in_applicability_domain": dom.get("in_domain", True),
                    "domain_similarity": s_max,
                    "risk_index": risk_idx,
                    "admet_profile": admet,
                    "toxicology_profile": tox,
                    "objectives": {
                        "predicted_pdac_activity": round(potency_obj, 4),
                        "selectivity_index": round(sel_obj, 4),
                        "qed_druglikeness": round(qed_obj, 4),
                        "low_toxicological_risk": round(low_tox_obj, 4),
                        "applicability_domain_confidence": round(domain_obj, 4),
                    },
                }
                evaluated_candidates.append(candidate_entry)
                all_evaluated[sm] = candidate_entry

            stage_trace.append({
                "stage_number": 9,
                "stage_name": "Multi-Objective Evaluation",
                "generation": gen_number,
                "details": {
                    "candidates_scored": len(evaluated_candidates),
                    "objectives_evaluated": [
                        "predicted_pdac_activity", "selectivity_index",
                        "qed_druglikeness", "low_toxicological_risk",
                        "applicability_domain_confidence",
                    ],
                    "core_principle": "Strict multi-objective balance; tumor kill is NEVER maximized in isolation.",
                },
            })

            # -----------------------------------------------------------------
            # STAGE 10: Progressive Triage via LayaDecisionEngine
            # -----------------------------------------------------------------
            triage_inputs = []
            for cand in evaluated_candidates:
                if cand["pic50"] is None:
                    cand["triage_status"] = "chemistry_only"
                    cand["triage_rationale"] = "Activity unavailable; no efficacy or safety approval inferred."
                    continue
                triage_inputs.append({
                    "compound_id": cand["compound_id"],
                    "name": cand["name"],
                    "canonical_smiles": cand["canonical_smiles"],
                    "qed": cand["qed"],
                    "pic50": cand["pic50"],
                    "selectivity_hpde": cand["selectivity_hpde"],
                    "hepatotox_alert": cand["hepatotox_alert"],
                    "in_applicability_domain": cand["in_applicability_domain"],
                })

            triage_result = self.decision_engine.triage_candidates(triage_inputs)
            rankings = triage_result.get("rankings", [])

            # Match rankings back to evaluated candidates
            for r in rankings:
                for c in evaluated_candidates:
                    if c["compound_id"] == r["compound_id"]:
                        c["triage_status"] = r["status"]
                        c["composite_utility"] = r["composite_utility"]
                        c["triage_rationale"] = r["rationale"]
                        c["triage_warnings"] = r["warnings"]
                        break

            # Sort by composite utility descending
            evaluated_candidates.sort(key=lambda x: x.get("composite_utility", 0.0), reverse=True)
            if objective_weights:
                weight_total = max(0.1, sum(max(0.0, float(objective_weights.get(key, 1.0))) for key in evaluated_candidates[0]["objectives"])) if evaluated_candidates else 1.0
                for candidate in evaluated_candidates:
                    candidate["composite_utility"] = sum(value * max(0.0, float(objective_weights.get(key, 1.0))) for key, value in candidate["objectives"].items()) / weight_total
                evaluated_candidates.sort(key=lambda c: c["composite_utility"], reverse=True)

            # Select top K candidates for cellular microenvironment simulation
            top_for_cellular = evaluated_candidates[: max(1, top_k_cellular)]

            stage_trace.append({
                "stage_number": 10,
                "stage_name": "Progressive Triage (Laya Decision Engine)",
                "generation": gen_number,
                "details": {
                    "total_triaged": len(evaluated_candidates),
                    "approved_count": triage_result.get("approved_count", 0),
                    "flagged_count": triage_result.get("flagged_count", 0),
                    "rejected_count": triage_result.get("rejected_count", 0),
                    "selected_for_2d_cellular": [c["compound_id"] for c in top_for_cellular],
                },
            })

            # -----------------------------------------------------------------
            # STAGE 11: Controlled 2D PDAC Microenvironment Simulation
            # -----------------------------------------------------------------
            cellular_simulation_results: list[dict[str, Any]] = []
            for cand in top_for_cellular:
                if self.disease_id != "PDAC" or cand.get("ic50_um") is None:
                    continue
                comp = cand["compound_record"]
                admet = cand["admet_profile"]

                # Adapt pharmacology profile for PDACModule consumption
                adapted_pharma = dict(admet)
                if isinstance(adapted_pharma.get("fraction_unbound_plasma"), dict):
                    adapted_pharma["fraction_unbound_plasma"] = adapted_pharma["fraction_unbound_plasma"].get("value", 0.25)
                if "ic50_um" in cand and cand["ic50_um"] is not None:
                    ic50_val = max(0.001, float(cand["ic50_um"]))
                    adapted_pharma["ic50_sensitive_um"] = ic50_val
                    adapted_pharma["ic50_resistant_um"] = ic50_val * 4.5
                    adapted_pharma["ic50_healthy_um"] = ic50_val * 12.0

                # Convert fractional Emax to the tissue model's per-second hazard.
                adapted_pharma["emax"] = PDACModule.emax_default if hasattr(PDACModule, "emax_default") else 0.000035
                # Run controlled comparison vs untreated vehicle control (0.0 uM)
                disease_sim = self.disease_registry.create(self.disease_id, seed=seed, nx=30, ny=24)
                comparison = disease_sim.run_controlled_comparison(
                    compound=comp,
                    pharmacology_profile=adapted_pharma,
                    applied_concentration_um=applied_concentration_um,
                    duration_hours=duration_hours,
                    seed=seed,
                )
                cand["controlled_cellular_comparison"] = comparison
                cand["controlled_pdac_comparison"] = comparison
                cellular_simulation_results.append({
                    "compound_id": cand["compound_id"],
                    "name": cand["name"],
                    "tgi_pct": comparison["deltas"]["tumor_growth_inhibition_pct"],
                    "clonal_shift_delta": comparison["deltas"]["clonal_shift_delta"],
                    "apoptosis_induction_ratio": comparison["deltas"]["apoptosis_induction_ratio"],
                    "barrier_attenuation_ratio": comparison["deltas"]["barrier_attenuation_ratio"],
                })
                all_controlled_comparisons.append(comparison)

            stage_trace.append({
                "stage_number": 11,
                "stage_name": f"Controlled 2D {self.disease_id} Microenvironment Simulation",
                "generation": gen_number,
                "details": {
                    "cellular_evaluations_run": len(cellular_simulation_results),
                    "duration_hours": duration_hours,
                    "applied_concentration_um": applied_concentration_um,
                    "summary_deltas": cellular_simulation_results,
                },
            })

            # -----------------------------------------------------------------
            # STAGE 12: Result & Provenance Storage
            # -----------------------------------------------------------------
            stage_trace.append({
                "stage_number": 12,
                "stage_name": "Result & Provenance Storage",
                "generation": gen_number,
                "details": {
                    "evaluated_in_generation": len(evaluated_candidates),
                    "cumulative_unique_structures": len(all_evaluated),
                    "provenance_tracked": True,
                },
            })

            # -----------------------------------------------------------------
            # STAGE 13: Pareto-Optimal Candidate Identification
            # -----------------------------------------------------------------
            # Matrix of 5 objectives for NSGA-II non-dominated sorting
            obj_matrix = np.zeros((len(evaluated_candidates), 5), dtype=np.float64)
            for idx, cand in enumerate(evaluated_candidates):
                objs = cand["objectives"]
                obj_matrix[idx, 0] = objs["predicted_pdac_activity"]
                obj_matrix[idx, 1] = objs["selectivity_index"]
                obj_matrix[idx, 2] = objs["qed_druglikeness"]
                obj_matrix[idx, 3] = objs["low_toxicological_risk"]
                obj_matrix[idx, 4] = objs["applicability_domain_confidence"]

            fronts = self.molecular_optimizer.fast_non_dominated_sort(obj_matrix)
            pareto_front_indices = fronts[0] if fronts else []
            pareto_front_candidates = [evaluated_candidates[i] for i in pareto_front_indices]

            stage_trace.append({
                "stage_number": 13,
                "stage_name": "Pareto-Optimal Candidate Identification",
                "generation": gen_number,
                "details": {
                    "total_fronts": len(fronts),
                    "pareto_front_0_count": len(pareto_front_candidates),
                    "pareto_front_members": [
                        {
                            "id": c["compound_id"],
                            "name": c["name"],
                            "composite_utility": c.get("composite_utility"),
                            "objectives": c["objectives"],
                        }
                        for c in pareto_front_candidates
                    ],
                },
            })

            # -----------------------------------------------------------------
            # STAGE 14: Evolutionary Population Update
            # -----------------------------------------------------------------
            # Select survivor population of size population_size using NSGA-II front priority & crowding
            new_pop_smiles: list[str] = []
            for front in fronts:
                if len(new_pop_smiles) + len(front) <= population_size:
                    for f_idx in front:
                        new_pop_smiles.append(evaluated_candidates[f_idx]["canonical_smiles"])
                else:
                    needed = population_size - len(new_pop_smiles)
                    # Use crowding distance to break ties
                    cd = self.molecular_optimizer.crowding_distance(obj_matrix, front)
                    sorted_front = sorted(front, key=lambda idx: (evaluated_candidates[idx].get("composite_utility", 0.0), cd.get(idx, 0.0)) if objective_weights else (cd.get(idx, 0.0), 0.0), reverse=True)
                    for f_idx in sorted_front[:needed]:
                        new_pop_smiles.append(evaluated_candidates[f_idx]["canonical_smiles"])
                    break

            if not new_pop_smiles:
                new_pop_smiles = [c["canonical_smiles"] for c in evaluated_candidates[:population_size]]

            current_smiles_pop = new_pop_smiles

            stage_trace.append({
                "stage_number": 14,
                "stage_name": "Evolutionary Population Update",
                "generation": gen_number,
                "details": {
                    "updated_survivor_pool_size": len(current_smiles_pop),
                    "selection_mechanism": "NSGA-II Non-dominated Sorting with Crowding Distance",
                },
            })

            # -----------------------------------------------------------------
            # STAGE 15: Next Generation Loop Execution
            # -----------------------------------------------------------------
            stage_trace.append({
                "stage_number": 15,
                "stage_name": "Next Generation Loop",
                "generation": gen_number,
                "details": {
                    "completed_generation": gen_number,
                    "target_generations": generations,
                    "loop_status": "Advancing to next generation" if gen_number < generations else "Discovery cycles complete",
                },
            })

        # Final Pareto Frontier extraction across ALL evaluated compounds
        final_list = list(all_evaluated.values())
        final_matrix = np.zeros((len(final_list), 5), dtype=np.float64)
        for idx, cand in enumerate(final_list):
            objs = cand["objectives"]
            final_matrix[idx, 0] = objs["predicted_pdac_activity"]
            final_matrix[idx, 1] = objs["selectivity_index"]
            final_matrix[idx, 2] = objs["qed_druglikeness"]
            final_matrix[idx, 3] = objs["low_toxicological_risk"]
            final_matrix[idx, 4] = objs["applicability_domain_confidence"]

        final_fronts = self.molecular_optimizer.fast_non_dominated_sort(final_matrix)
        final_pareto_indices = final_fronts[0] if final_fronts else []
        final_pareto_candidates = [final_list[i] for i in final_pareto_indices]

        # Sort Pareto front by composite utility descending
        final_pareto_candidates.sort(key=lambda x: x.get("composite_utility", 0.0), reverse=True)

        def candidate_summary(candidate):
            record = candidate["compound_record"]
            return {
                "compound_id": record.compound_id, "name": record.name,
                "canonical_smiles": record.canonical_smiles, "formula": record.formula,
                "origin_generation": record.origin_generation, "parent_ids": record.parent_ids,
                "transformation_type": "directed_optimization", "svg_2d": record.svg_2d,
                "mol_weight": record.mol_weight, "logp": record.logp, "qed": record.qed,
                "risk_index": candidate["risk_index"], "pic50": candidate["pic50"],
                "in_applicability_domain": candidate["in_applicability_domain"],
                "objectives": candidate["objectives"], "composite_utility": candidate.get("composite_utility"),
                "source": "Directed optimization", "epistemic_category": "Exploratory model result",
                "is_seed": record.canonical_smiles in seed_smiles,
            }
        ranked = sorted(final_list, key=lambda c: c.get("composite_utility", 0.0), reverse=True)
        variants = [candidate for candidate in ranked if candidate["canonical_smiles"] not in seed_smiles]

        return {
            "evaluated_candidates": [candidate_summary(c) for c in final_list],
            "top_candidates": [candidate_summary(c) for c in variants[:candidate_count]],
            "seed_controls": [candidate_summary(c) for c in ranked if c["canonical_smiles"] in seed_smiles],
            "requested_candidate_count": candidate_count,
            "objective_weights": objective_weights or {},
            "disease_id": self.disease_id,
            "activity_scope": "Exploratory PDAC surrogate; not clinically validated" if self.disease_id == "PDAC" else "Chemistry-only ranking. No activity or novel-compound cellular response is available for this disease.",
            "pipeline_version": "1.1.0",
            "simulator_version": "1.0.0",
            "generations_completed": generations,
            "total_molecules_evaluated": len(all_evaluated),
            "pareto_frontier_count": len(final_pareto_candidates),
            "pareto_frontier": [
                {
                    "compound_id": c["compound_id"],
                    "name": c["name"],
                    "canonical_smiles": c["canonical_smiles"],
                    "qed": c["qed"],
                    "pic50": c["pic50"],
                    "ic50_um": c["ic50_um"],
                    "risk_index": c["risk_index"],
                    "in_applicability_domain": c["in_applicability_domain"],
                    "composite_utility": c.get("composite_utility"),
                    "triage_status": c.get("triage_status"),
                    "triage_rationale": c.get("triage_rationale"),
                    "objectives": c["objectives"],
                    "cellular_simulation": (
                        c.get("controlled_cellular_comparison", {}).get("deltas")
                        or c.get("controlled_pdac_comparison", {}).get("deltas")
                    ),
                }
                for c in final_pareto_candidates
            ],
            "controlled_comparisons": all_controlled_comparisons,
            "stage_by_stage_trace": stage_trace,
            "epistemic_disclaimer": NON_CLINICAL_DISCLAIMER,
        }
