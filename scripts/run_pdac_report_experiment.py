"""Reproducible PDAC candidate screening run.

This is a software verification experiment. It exercises the full discovery pipeline
end to end and records every number it produces, including the ones that undercut the
candidates. It is not a clinical efficacy evaluation and produces no evidence about
human disease.

Design notes that matter when reading the output:

* Two reference controls bracket the tissue assay. Gemcitabine is the positive control
  and should show a large growth inhibition. Caffeine is the negative control and should
  show roughly none. If the controls do not separate, the assay is not measuring
  anything and the candidate numbers mean nothing.
* Every compound is run across several independent seeds. A candidate is only called
  distinguishable from the negative control when Welch's t-test on those replicates says
  so. Point estimates without that comparison are not reported as effects.
* The compute backend is recorded. Force EVOCURE_COMPUTE=cpu for a bit reproducible run.

    python scripts/run_pdac_report_experiment.py
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version

import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, rdMolDescriptors
from scipy import stats

from simulation.core.engine import EVOCUREEngine
from simulation.core.acceleration import status as compute_status
from simulation.discovery.pipeline import CandidateDiscoveryPipeline
from simulation.discovery.workbench import plan_guided_search, compare_drugs

OUT = ROOT / "reports" / "pdac_publication_test"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20261027
# Five independent tissue seeds per compound. Three was too few to tell a real effect
# from seed noise, since the per seed spread on these grids crosses zero.
REPLICATES = [20261027, 20261028, 20261029, 20261030, 20261031]
POSITIVE_CONTROL = "Gemcitabine"
NEGATIVE_CONTROL = "Caffeine"

PROMPT = (
    "Find 5 experimental candidates for pancreatic cancer based on Gemcitabine, "
    "5-Fluorouracil, Erlotinib and Paclitaxel. Prioritize activity, low predicted "
    "toxicity and drug-likeness."
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe_spread(values):
    """Mean, sample standard deviation and a t based 95% interval."""
    array = np.asarray(values, dtype=float)
    n = array.size
    mean = float(array.mean())
    if n < 2:
        return {"mean": round(mean, 4), "n": n, "std": None,
                "ci95_low": None, "ci95_high": None, "excludes_zero": None}
    std = float(array.std(ddof=1))
    half = float(stats.t.ppf(0.975, n - 1) * std / np.sqrt(n))
    low, high = mean - half, mean + half
    return {"mean": round(mean, 4), "n": n, "std": round(std, 4),
            "ci95_low": round(low, 4), "ci95_high": round(high, 4),
            "excludes_zero": bool(low > 0 or high < 0)}


def compare_to_control(values, control_values):
    """Welch's t-test plus Hedges' g. Welch is used because variances differ by arm."""
    treated = np.asarray(values, dtype=float)
    control = np.asarray(control_values, dtype=float)
    if treated.size < 2 or control.size < 2:
        return {"test": "unavailable", "reason": "fewer than two replicates"}

    result = stats.ttest_ind(treated, control, equal_var=False)
    n1, n2 = treated.size, control.size
    s1, s2 = treated.std(ddof=1), control.std(ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled > 0:
        d = float((treated.mean() - control.mean()) / pooled)
        correction = 1.0 - (3.0 / (4.0 * (n1 + n2) - 9.0))
        g = d * correction
    else:
        g = float("nan")

    p_value = float(result.pvalue)
    return {
        "test": "Welch two sided t-test versus negative control",
        "mean_difference_pct": round(float(treated.mean() - control.mean()), 4),
        "t_statistic": round(float(result.statistic), 4),
        "p_value": p_value,
        "hedges_g": None if np.isnan(g) else round(g, 4),
        "separates_from_negative_control": bool(p_value < 0.05),
        "note": (
            "Uncorrected for multiple comparisons. With five candidates tested, a "
            "Bonferroni threshold would be p < 0.01."
        ),
        "bonferroni_separates": bool(p_value < 0.01),
    }


def main():
    started = time.perf_counter()
    engine = EVOCUREEngine(seed=SEED)
    engine.set_active_disease("PDAC", seed=SEED)

    # The deterministic planner is the default so the published run reproduces on a
    # machine with no language model installed. Set EVOCURE_REPORT_USE_LLM_PLANNER=1
    # to let a local Gemma server propose the search priorities instead.
    use_llm_planner = os.environ.get("EVOCURE_REPORT_USE_LLM_PLANNER") == "1"
    plan = plan_guided_search(PROMPT, engine.assistant if use_llm_planner else None)
    settings = {
        "seed_queries": plan["seed_queries"], "generations": 4,
        "population_size": 24, "top_k_cellular": 1,
        "applied_concentration_um": 5.0, "duration_hours": 24.0,
        "seed": SEED, "objective_weights": plan["objective_weights"],
        "candidate_count": 5,
        # Closed selection loop: every candidate that can be dosed is simulated in the
        # microenvironment and its growth inhibition is a selection objective, so the
        # tumour model drives the search instead of only reporting on its output.
        "tissue_in_fitness": True,
    }
    print("PLAN", plan["provider"], settings, flush=True)

    pipeline = CandidateDiscoveryPipeline(
        disease_id="PDAC", qsar_predictor=engine.qsar, decision_engine=engine.laya,
    )
    result = pipeline.run(
        **settings,
        progress_callback=lambda event: print("SEARCH", event, flush=True),
    )
    engine.register_discovery_results(result)
    (OUT / "pipeline_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    report = {
        "report_id": "EVOCURE-PDAC-20261027",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Exploratory software verification and candidate prioritisation. Not evidence "
            "of a cure, patient benefit, safety or novelty."
        ),
        "plan": plan,
        "search_settings": settings,
        "replicate_seeds": REPLICATES,
        "controls": {"positive": POSITIVE_CONTROL, "negative": NEGATIVE_CONTROL},
        "compute": compute_status(),
        "laya": engine.laya.get_status(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: version(name)
            for name in ("rdkit", "numpy", "scipy", "scikit-learn", "torch")
        },
        "hashes": {
            str(path.relative_to(ROOT)): digest(path) for path in [
                ROOT / "models/scientific/pdac_qsar_v1.joblib",
                ROOT / "data/reference/pdac_reference_dataset.json",
                ROOT / "simulation/discovery/pipeline.py",
                ROOT / "simulation/biology/diseases/pdac.py",
                ROOT / "simulation/core/engine.py",
                ROOT / "simulation/pharmacology/admet.py",
                ROOT / "simulation/discovery/workbench.py",
            ]
        },
        "selection_basis": result["selection_basis"],
        "tissue_in_fitness": result["tissue_in_fitness"],
        "objective_keys": result["objective_keys"],
        "tissue_simulations_run": len(result["controlled_comparisons"]),
        "qsar_validation": dict(engine.qsar.validation_metrics),
        "training_structure_count": len(engine.qsar.train_smiles),
        "search_total_structures": result["total_molecules_evaluated"],
        "grid": {key: getattr(engine.tumor_module, key)
                 for key in ("nx", "ny", "width", "height")},
        "candidates": [],
        "reference_controls": [],
    }

    def evaluate(query, label, candidate=None):
        record = engine._resolve_compound_rec(query)
        mol = Chem.MolFromSmiles(record.canonical_smiles)
        comparison = compare_drugs(record, 5)
        profile, info = engine._tissue_pharmacology(record)

        trials = []
        for seed in REPLICATES:
            print("TISSUE", label, seed, flush=True)
            trials.append(engine.run_tumor_controlled_comparison(
                record.compound_id, 5.0, 24.0, seed))

        tgi = [t["deltas"]["tumor_growth_inhibition_pct"] for t in trials]
        resistant = [t["treated_trajectory"]["resistant_fraction"][-1] for t in trials]

        return {
            "label": label,
            "compound": record.to_dict(),
            "ranking": candidate,
            "inchikey": Chem.MolToInchiKey(mol),
            "descriptors": {
                "molecular_weight": Descriptors.MolWt(mol),
                "logp": Descriptors.MolLogP(mol),
                "tpsa": rdMolDescriptors.CalcTPSA(mol),
                "hbd": rdMolDescriptors.CalcNumHBD(mol),
                "hba": rdMolDescriptors.CalcNumHBA(mol),
                "qed": record.qed,
            },
            "qsar": engine.qsar.predict(record.canonical_smiles),
            "drug_comparison": comparison,
            "tissue_profile": profile,
            "tissue_model_info": info,
            "replicates": trials,
            "tgi_per_replicate": tgi,
            "summary": {
                "tgi": describe_spread(tgi),
                "resistant_fraction": describe_spread(resistant),
                # Kept for anything reading the previous schema.
                "tgi_mean_pct": float(np.mean(tgi)),
                "tgi_min_pct": min(tgi),
                "tgi_max_pct": max(tgi),
                "resistant_fraction_mean": float(np.mean(resistant)),
            },
        }

    # Controls first, so the candidate arms can be tested against the negative control
    # in the same pass and the run fails fast if the assay is not separating.
    for reference in (POSITIVE_CONTROL, NEGATIVE_CONTROL):
        report["reference_controls"].append(evaluate(reference, reference))

    negative_tgi = next(c["tgi_per_replicate"] for c in report["reference_controls"]
                        if c["label"] == NEGATIVE_CONTROL)
    positive_tgi = next(c["tgi_per_replicate"] for c in report["reference_controls"]
                        if c["label"] == POSITIVE_CONTROL)

    # Like for like control. The positive control draws its potency from curated
    # experimental bioactivity; a generated structure cannot, and the structure
    # property model that scores it has a floor well above the reference value. So
    # the positive control is also run with its curated bioactivity removed, which
    # is the only comparator the candidates are actually measured against.
    def modelled_potency_arm(query: str) -> dict | None:
        from simulation.biology.registry import DiseaseRegistry
        import copy as _copy

        record = engine._resolve_compound_rec(query)
        bare = _copy.deepcopy(record)
        bare.reference_bioactivity = {}

        profile = dict(engine.pharmacology.evaluate_admet(bare))
        response = engine.pharmacology.predict_antitumor_activity(bare)
        for key in ("ic50_sensitive_um", "ic50_resistant_um", "ic50_healthy_um", "hill_slope"):
            value = response.get(key, {})
            raw = value.get("value") if isinstance(value, dict) else value
            if not isinstance(raw, (int, float)):
                return None
            profile[key] = float(raw)
        profile["emax"] = float(response["emax"]["value"]) * engine.tumor_module.emax_default
        if isinstance(profile.get("fraction_unbound_plasma"), dict):
            profile["fraction_unbound_plasma"] = profile["fraction_unbound_plasma"]["value"]

        values = []
        for seed in REPLICATES:
            print("TISSUE modelled_potency_control", seed, flush=True)
            sim = DiseaseRegistry().create("PDAC", seed=seed, nx=36, ny=28)
            values.append(sim.run_controlled_comparison(
                compound=bare, pharmacology_profile=profile,
                applied_concentration_um=5.0, duration_hours=24.0, seed=seed,
            )["deltas"]["tumor_growth_inhibition_pct"])

        curated_ic50 = engine._tissue_pharmacology(record)[0]["ic50_sensitive_um"]
        return {
            "compound": query,
            "curated_ic50_um": round(float(curated_ic50), 4),
            "modelled_ic50_um": round(profile["ic50_sensitive_um"], 4),
            "fold_weaker_when_modelled": round(
                profile["ic50_sensitive_um"] / max(1e-9, float(curated_ic50)), 1),
            "tgi_per_replicate": values,
            "summary": describe_spread(values),
            "versus_negative_control": compare_to_control(values, negative_tgi),
        }

    report["modelled_potency_control"] = modelled_potency_arm(POSITIVE_CONTROL)
    print("MODELLED_CONTROL", json.dumps(report["modelled_potency_control"]), flush=True)

    assay = compare_to_control(positive_tgi, negative_tgi)
    report["assay_validity"] = {
        "positive_vs_negative_control": assay,
        "assay_separates_controls": assay.get("separates_from_negative_control", False),
        "interpretation": (
            "The positive control separates from the negative control, so the tissue "
            "model responds to a known active in this configuration."
            if assay.get("separates_from_negative_control")
            else "The controls do not separate. Candidate tissue numbers from this run "
                 "carry no weight and should not be interpreted."
        ),
    }
    print("ASSAY", json.dumps(report["assay_validity"]), flush=True)

    modelled_arm = report.get("modelled_potency_control") or {}
    modelled_tgi = modelled_arm.get("tgi_per_replicate")

    for index, candidate in enumerate(result["top_candidates"], 1):
        details = evaluate(candidate["compound_id"], f"Candidate_{index:02}", candidate)
        details["versus_negative_control"] = compare_to_control(
            details["tgi_per_replicate"], negative_tgi)
        if modelled_tgi:
            details["versus_modelled_potency_control"] = compare_to_control(
                details["tgi_per_replicate"], modelled_tgi)
        report["candidates"].append(details)
        (OUT / "experiment_results.partial.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")

    # Structure images for the report. Written last so a failure here cannot lose a run.
    for entry in report["candidates"] + report["reference_controls"]:
        mol = Chem.MolFromSmiles(entry["compound"]["canonical_smiles"])
        if mol is not None:
            Draw.MolToImage(mol, size=(1200, 600)).save(OUT / (entry["label"] + ".png"))

    separated = [c["label"] for c in report["candidates"]
                 if c["versus_negative_control"].get("separates_from_negative_control")]
    corrected = [c["label"] for c in report["candidates"]
                 if c["versus_negative_control"].get("bonferroni_separates")]
    report["candidates_separating_from_negative_control"] = separated
    report["candidates_separating_bonferroni"] = corrected

    total = len(report["candidates"])
    control_fails_too = (
        modelled_arm.get("versus_negative_control", {}).get("separates_from_negative_control")
        is False
    )
    if not separated and control_fails_too:
        # Reporting the bare count would read as a verdict on the molecules. It is not
        # one while the positive control fails the same test under the same scoring.
        report["headline"] = (
            f"No candidate of {total} separated from the negative control, and neither "
            f"did the positive control when scored by the same potency model."
        )
    elif not separated:
        report["headline"] = (
            f"No candidate of {total} separated from the negative control across "
            f"{len(REPLICATES)} seeds."
        )
    elif corrected:
        report["headline"] = (
            f"{len(corrected)} of {total} candidates separated from the negative control "
            f"and survived correction for {total} comparisons."
        )
    else:
        # Reporting the uncorrected count alone would overstate a marginal result.
        report["headline"] = (
            f"{len(separated)} of {total} candidates separated from the negative control "
            f"at an uncorrected p < 0.05, and none survived correction for {total} "
            f"comparisons."
        )

    saved = engine.save_current_experiment("PDAC five-candidate software test | 2026-10-27")
    report["saved_experiment_id"] = saved["experiment_id"]
    report["elapsed_seconds"] = time.perf_counter() - started
    report["results_hash"] = digest(OUT / "pipeline_results.json")
    (OUT / "experiment_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("COMPLETE", report["search_total_structures"], "structures; experiment",
          saved["experiment_id"], "elapsed seconds", round(report["elapsed_seconds"], 1),
          flush=True)
    print("HEADLINE", report["headline"], flush=True)


if __name__ == "__main__":
    main()
