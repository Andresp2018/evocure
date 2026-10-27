"""Why no generated candidate matches the positive control.

The published screening run ends with every candidate indistinguishable from the
negative control while the positive control separates cleanly. Two explanations fit
that observation and they have opposite consequences:

  A. The generated molecules are genuinely weak in this tissue model.
  B. Reference drugs and generated molecules get their potency from different
     sources, and the comparison is rigged before any simulation runs.

This script separates them. Four experiments:

  1. Provenance. Where each compound's IC50 comes from, and the range the
     structure-property model can produce at all.
  2. Like for like. Score the positive control through the same structure-property
     model used for generated molecules, then re-run the tissue comparison. If
     gemcitabine also stops separating, B is doing the work.
  3. Dose response. Sweep the applied concentration to see whether the candidates
     are weak or merely underdosed.
  4. Alternative potency model. Drive the tissue model from the calibrated QSAR
     instead of the structure-property heuristic.

    python scripts/investigate_potency_asymmetry.py

Writes reports/potency_asymmetry.json and prints a summary. Nothing here is
evidence about human disease; it characterises the software's own behaviour.
"""

from __future__ import annotations

from pathlib import Path
import copy
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy import stats

from simulation.biology.registry import DiseaseRegistry
from simulation.chemistry.engine import ChemistryEngine
from simulation.core.engine import EVOCUREEngine
from simulation.pharmacology.admet import PharmacologyEngine

OUT = ROOT / "reports" / "potency_asymmetry.json"
RESULTS = ROOT / "reports" / "pdac_publication_test" / "experiment_results.json"

SEED = 20261027
REPLICATES = [20261027, 20261028, 20261029, 20261030, 20261031]
DOSES_UM = [1.0, 5.0, 25.0, 100.0, 500.0]
DURATION_H = 24.0
GRID = {"nx": 36, "ny": 28}


def welch(treated: list[float], control: list[float]) -> dict:
    a, b = np.asarray(treated, float), np.asarray(control, float)
    if a.size < 2 or b.size < 2:
        return {"p_value": None, "separates": None}
    result = stats.ttest_ind(a, b, equal_var=False)
    return {
        "mean_difference_pct": round(float(a.mean() - b.mean()), 3),
        "p_value": float(result.pvalue),
        "separates": bool(result.pvalue < 0.05),
    }


def tissue_profile(engine: EVOCUREEngine, record, qsar_pred=None) -> dict | None:
    """Rebuild the reporting path's profile, optionally through the QSAR branch."""
    pharm: PharmacologyEngine = engine.pharmacology
    profile = dict(pharm.evaluate_admet(record))
    response = pharm.predict_antitumor_activity(record, qsar_pred=qsar_pred)
    for key in ("ic50_sensitive_um", "ic50_resistant_um", "ic50_healthy_um", "hill_slope"):
        value = response.get(key, {})
        raw = value.get("value") if isinstance(value, dict) else value
        if not isinstance(raw, (int, float)):
            return None
        profile[key] = float(raw)
    emax = response["emax"]["value"]
    profile["emax"] = float(emax) * engine.tumor_module.emax_default
    profile["selectivity_index"] = profile["ic50_healthy_um"] / max(0.001, profile["ic50_sensitive_um"])
    if isinstance(profile.get("fraction_unbound_plasma"), dict):
        profile["fraction_unbound_plasma"] = profile["fraction_unbound_plasma"]["value"]
    profile["_source"] = response["ic50_sensitive_um"].get("source")
    profile["_category"] = response["ic50_sensitive_um"].get("epistemic_category")
    return profile


def run_tissue(record, profile: dict, dose_um: float, seeds: list[int]) -> list[float]:
    clean = {k: v for k, v in profile.items() if not k.startswith("_")}
    values = []
    for seed in seeds:
        sim = DiseaseRegistry().create("PDAC", seed=seed, **GRID)
        comparison = sim.run_controlled_comparison(
            compound=record, pharmacology_profile=clean,
            applied_concentration_um=dose_um, duration_hours=DURATION_H, seed=seed,
        )
        values.append(comparison["deltas"]["tumor_growth_inhibition_pct"])
    return values


def strip_reference(record):
    """A copy of the compound with its curated bioactivity removed.

    This is what the pharmacology model sees for any newly generated structure.
    """
    bare = copy.deepcopy(record)
    bare.reference_bioactivity = {}
    return bare


def heuristic_ic50(mol_weight: float, logp: float, qed: float) -> float:
    """The structure-property branch, reproduced so its range can be mapped."""
    logp_opt = max(0.0, 1.0 - abs(logp - 2.5) / 3.0)
    mw_opt = max(0.0, 1.0 - abs(mol_weight - 400.0) / 300.0)
    potency_factor = max(0.05, 0.5 * logp_opt + 0.3 * mw_opt + 0.2 * qed)
    return float(math.exp(3.0 - 3.2 * potency_factor))


def main() -> int:
    engine = EVOCUREEngine(seed=SEED)
    engine.set_active_disease("PDAC", seed=SEED)
    chem = ChemistryEngine()
    report: dict = {"seed": SEED, "replicates": REPLICATES, "duration_hours": DURATION_H,
                    "grid": GRID, "doses_um": DOSES_UM}

    published = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.is_file() else {}
    candidate_smiles = [c["compound"]["canonical_smiles"] for c in published.get("candidates", [])]

    # ---------------------------------------------------------------- 1. provenance
    print("=" * 78)
    print("1. Where potency comes from")
    print("=" * 78)

    provenance = []
    for label, query in ([("Gemcitabine", "Gemcitabine"), ("5-Fluorouracil", "5-Fluorouracil"),
                          ("Erlotinib", "Erlotinib"), ("Caffeine", "Caffeine")]
                         + [(f"Candidate_{i:02}", s) for i, s in enumerate(candidate_smiles, 1)]):
        try:
            record = chem.resolve_compound_input(query)
        except ValueError:
            record = chem.build_compound(query)
        profile = tissue_profile(engine, record)
        if profile is None:
            continue
        entry = {
            "label": label,
            "ic50_sensitive_um": round(profile["ic50_sensitive_um"], 4),
            "source": profile["_source"],
            "epistemic_category": profile["_category"],
            "has_curated_bioactivity": bool(record.reference_bioactivity),
            "mol_weight": round(record.mol_weight, 1),
            "logp": round(record.logp, 2),
            "qed": round(record.qed, 3),
        }
        provenance.append(entry)
        print(f"  {label:16s} IC50 {entry['ic50_sensitive_um']:9.4f} uM   "
              f"{'curated' if entry['has_curated_bioactivity'] else 'modelled':9s} "
              f"{str(entry['source'])[:44]}")
    report["provenance"] = provenance

    # The achievable range of the structure-property branch
    grid_ic50 = [heuristic_ic50(mw, logp, qed)
                 for mw in range(250, 601, 25)
                 for logp in [x / 2 for x in range(0, 11)]
                 for qed in [0.3, 0.5, 0.7, 0.9]]
    best, worst = min(grid_ic50), max(grid_ic50)
    curated = [e for e in provenance if e["has_curated_bioactivity"]]
    report["structure_property_range_um"] = {"best_possible": round(best, 4),
                                             "worst": round(worst, 4)}
    print(f"\n  Structure-property model can only ever return IC50 between "
          f"{best:.3f} and {worst:.1f} uM.")
    for entry in curated:
        ratio = best / entry["ic50_sensitive_um"] if entry["ic50_sensitive_um"] > 0 else float("inf")
        print(f"    {entry['label']} is curated at {entry['ic50_sensitive_um']:.4f} uM, "
              f"{ratio:.0f}x below anything the model can output.")
        entry["best_model_output_is_x_times_weaker"] = round(ratio, 1)

    # ------------------------------------------------------- 2. like for like control
    print("\n" + "=" * 78)
    print("2. The positive control scored as if it were a generated structure")
    print("=" * 78)

    gem = chem.resolve_compound_input("Gemcitabine")
    caf = chem.resolve_compound_input("Caffeine")

    gem_curated = tissue_profile(engine, gem)
    gem_bare = tissue_profile(engine, strip_reference(gem))
    caf_profile = tissue_profile(engine, caf)

    negative = run_tissue(caf, caf_profile, 5.0, REPLICATES)
    curated_tgi = run_tissue(gem, gem_curated, 5.0, REPLICATES)
    bare_tgi = run_tissue(gem, gem_bare, 5.0, REPLICATES)

    like = {
        "negative_control_tgi": negative,
        "gemcitabine_curated": {
            "ic50_um": round(gem_curated["ic50_sensitive_um"], 4),
            "tgi": curated_tgi, "mean": round(float(np.mean(curated_tgi)), 2),
            **welch(curated_tgi, negative),
        },
        "gemcitabine_as_novel_structure": {
            "ic50_um": round(gem_bare["ic50_sensitive_um"], 4),
            "tgi": bare_tgi, "mean": round(float(np.mean(bare_tgi)), 2),
            **welch(bare_tgi, negative),
        },
    }
    report["like_for_like"] = like
    print(f"  Gemcitabine, curated potency   IC50 {like['gemcitabine_curated']['ic50_um']:8.4f} uM  "
          f"TGI {like['gemcitabine_curated']['mean']:6.2f} %  "
          f"separates={like['gemcitabine_curated']['separates']}")
    print(f"  Gemcitabine, modelled potency  IC50 {like['gemcitabine_as_novel_structure']['ic50_um']:8.4f} uM  "
          f"TGI {like['gemcitabine_as_novel_structure']['mean']:6.2f} %  "
          f"separates={like['gemcitabine_as_novel_structure']['separates']}")

    if like["gemcitabine_curated"]["separates"] and not like["gemcitabine_as_novel_structure"]["separates"]:
        verdict = ("The same molecule separates with its curated potency and fails with its "
                   "modelled potency. The screening result is driven by where the number "
                   "comes from, not by the chemistry of the candidates.")
    elif like["gemcitabine_as_novel_structure"]["separates"]:
        verdict = ("Gemcitabine separates even when scored by the structure-property model, "
                   "so the asymmetry alone does not explain why candidates fail.")
    else:
        verdict = "Inconclusive: the positive control did not separate in either configuration."
    report["like_for_like"]["verdict"] = verdict
    print(f"\n  {verdict}")

    # ------------------------------------------------------------- 3. dose response
    print("\n" + "=" * 78)
    print("3. Dose response")
    print("=" * 78)

    dose_seeds = REPLICATES[:3]
    dose_rows = []
    subjects = [("Gemcitabine", gem, gem_curated), ("Caffeine", caf, caf_profile)]
    for index, smiles in enumerate(candidate_smiles[:2], 1):
        record = chem.build_compound(smiles)
        profile = tissue_profile(engine, record)
        if profile is not None:
            subjects.append((f"Candidate_{index:02}", record, profile))

    print(f"  {'compound':16s} " + " ".join(f"{d:>8.0f}uM" for d in DOSES_UM))
    for label, record, profile in subjects:
        means = []
        for dose in DOSES_UM:
            values = run_tissue(record, profile, dose, dose_seeds)
            means.append(round(float(np.mean(values)), 2))
        dose_rows.append({"label": label,
                          "ic50_um": round(profile["ic50_sensitive_um"], 4),
                          "tgi_by_dose": dict(zip(map(str, DOSES_UM), means))})
        print(f"  {label:16s} " + " ".join(f"{m:>10.2f}" for m in means))
    report["dose_response"] = dose_rows

    candidate_rows = [r for r in dose_rows if r["label"].startswith("Candidate")]
    rescued = [r["label"] for r in candidate_rows
               if max(r["tgi_by_dose"].values()) >= 50.0]
    report["candidates_reaching_50pct_at_any_dose"] = rescued
    print(f"\n  Candidates reaching 50% inhibition at any tested dose: "
          f"{', '.join(rescued) if rescued else 'none'}")

    # ------------------------------------------------- 4. alternative potency model
    print("\n" + "=" * 78)
    print("4. Driving the tissue model from the calibrated QSAR instead")
    print("=" * 78)

    qsar_rows = []
    for index, smiles in enumerate(candidate_smiles, 1):
        record = chem.build_compound(smiles)
        prediction = engine.qsar.predict(record.canonical_smiles)
        ic50 = prediction.get("ic50_um_estimated")
        if not isinstance(ic50, (int, float)):
            continue
        qsar_pred = {
            "ic50_sensitive_um": float(ic50),
            "ic50_resistant_um": float(ic50) * 6.0,
            "ic50_healthy_um": float(ic50) * 12.0,
            "source": "Calibrated QSAR (investigation only)",
            "confidence": 0.8,
        }
        heuristic = tissue_profile(engine, record)
        qsar_profile = tissue_profile(engine, record, qsar_pred=qsar_pred)
        if heuristic is None or qsar_profile is None:
            continue
        tgi_q = run_tissue(record, qsar_profile, 5.0, REPLICATES)
        row = {
            "label": f"Candidate_{index:02}",
            "heuristic_ic50_um": round(heuristic["ic50_sensitive_um"], 4),
            "qsar_ic50_um": round(float(ic50), 4),
            "fold_difference": round(heuristic["ic50_sensitive_um"] / max(1e-9, float(ic50)), 1),
            "tgi_with_qsar": tgi_q,
            "mean_tgi_with_qsar": round(float(np.mean(tgi_q)), 2),
            **welch(tgi_q, negative),
        }
        qsar_rows.append(row)
        print(f"  {row['label']} heuristic {row['heuristic_ic50_um']:8.3f} uM  "
              f"QSAR {row['qsar_ic50_um']:8.4f} uM  ({row['fold_difference']:>6.1f}x)  "
              f"TGI {row['mean_tgi_with_qsar']:6.2f} %  separates={row['separates']}")
    report["qsar_driven"] = qsar_rows
    separating = [r["label"] for r in qsar_rows if r["separates"]]
    report["candidates_separating_with_qsar_potency"] = separating

    print(f"\n  Separating from the negative control with QSAR potency: "
          f"{', '.join(separating) if separating else 'none'} of {len(qsar_rows)}")
    print("\n  Note: the QSAR regressor's 95% conformal interval is about 4.2 log units, "
          "so these\n  IC50 values are not measurements. This shows the tissue model's "
          "sensitivity to the\n  potency input, not that the candidates are potent.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n" + "=" * 78)
    print(f"Written to {OUT}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
