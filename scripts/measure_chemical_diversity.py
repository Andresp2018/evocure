#!/usr/bin/env python3
"""Measure what chemical-space tie-breaking does to the search.

The project's own candidate review records that generated structures cluster at
Tanimoto 0.6 to the seed drug, and names the cause: NSGA-II's crowding distance
is computed in objective space, so two molecules that score alike are
interchangeable to it however different they are chemically. The population
therefore converges structurally while still looking diverse on the objectives.

This script runs the same search twice, once with the objective-space tie-break
and once with chemistry blended in, and reports:

  population diversity     mean pairwise Tanimoto distance of the survivors
  distinct Murcko scaffolds how many scaffold families the search visited
  similarity to the seeds   how far the candidates got from where they started
  objective quality         mean of the five base objectives, to show what the
                            diversity cost

A diversity gain that comes with an objective-quality loss is a trade, not an
improvement, so both are printed. Nothing here says the extra structures are
any good: a more diverse population of implausible molecules is still
implausible.

    python scripts/measure_chemical_diversity.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

from simulation.discovery.pipeline import BASE_OBJECTIVE_KEYS, CandidateDiscoveryPipeline

RDLogger.DisableLog("rdApp.*")

GENERATIONS = 6
POPULATION = 16
SEEDS = (11, 2024, 777)
MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def fingerprint(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    return MORGAN.GetFingerprint(mol) if mol is not None else None


def mean_pairwise_distance(smiles_list: list[str]) -> float:
    fps = [fp for fp in (fingerprint(s) for s in smiles_list) if fp is not None]
    if len(fps) < 2:
        return 0.0
    sims = [
        DataStructs.TanimotoSimilarity(fps[i], fps[j])
        for i in range(len(fps)) for j in range(i + 1, len(fps))
    ]
    return 1.0 - float(np.mean(sims))


def scaffold_count(smiles_list: list[str]) -> int:
    scaffolds = set()
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        try:
            scaffolds.add(MurckoScaffold.MurckoScaffoldSmiles(mol=mol))
        except Exception:
            continue
    return len(scaffolds)


def max_similarity_to_seeds(smiles_list: list[str], seed_smiles: list[str]) -> float:
    """Mean over candidates of the closest seed. Lower means the search moved."""
    seed_fps = [fp for fp in (fingerprint(s) for s in seed_smiles) if fp is not None]
    if not seed_fps:
        return float("nan")
    closest = []
    for smiles in smiles_list:
        own = fingerprint(smiles)
        if own is None:
            continue
        closest.append(max(DataStructs.TanimotoSimilarity(own, fp) for fp in seed_fps))
    return float(np.mean(closest)) if closest else float("nan")


def mean_objective_quality(candidates: list[dict]) -> float:
    values = [
        np.mean([c["objectives"].get(key, 0.0) for key in BASE_OBJECTIVE_KEYS])
        for c in candidates
    ]
    return float(np.mean(values)) if values else 0.0


def run_once(seed: int, diversity_weight: float) -> dict:
    pipeline = CandidateDiscoveryPipeline(disease_id="PDAC")
    result = pipeline.run(
        generations=GENERATIONS, population_size=POPULATION, top_k_cellular=1,
        duration_hours=4.0, seed=seed, candidate_count=8,
        chemical_diversity_weight=diversity_weight,
    )
    evaluated = result["evaluated_candidates"]
    variants = [c for c in evaluated if not c["is_seed"]]
    seed_smiles = [c["canonical_smiles"] for c in evaluated if c["is_seed"]]
    structures = [c["canonical_smiles"] for c in variants]

    return {
        "seed": seed,
        "diversity_weight": diversity_weight,
        "potency_model_available": result["potency_model_available"],
        "structures_evaluated": len(evaluated),
        "variants": len(variants),
        "population_diversity": round(result["population_diversity"], 4),
        "variant_diversity": round(mean_pairwise_distance(structures), 4),
        "distinct_scaffolds": scaffold_count(structures),
        "mean_similarity_to_seeds": round(max_similarity_to_seeds(structures, seed_smiles), 4),
        "mean_objective_quality": round(mean_objective_quality(variants), 4),
    }


def main() -> int:
    rows: list[dict] = []
    for seed in SEEDS:
        for weight in (0.0, 0.5):
            print(f"running seed {seed} with diversity weight {weight} ...", flush=True)
            rows.append(run_once(seed, weight))

    def group(weight: float) -> list[dict]:
        return [r for r in rows if r["diversity_weight"] == weight]

    print()
    header = (f"{'weight':>7} {'variants':>9} {'diversity':>10} {'scaffolds':>10} "
              f"{'sim_to_seed':>12} {'objectives':>11}")
    print(header)
    print("-" * len(header))
    summary: dict[str, dict] = {}
    for weight in (0.0, 0.5):
        rows_w = group(weight)
        means = {
            "variants": float(np.mean([r["variants"] for r in rows_w])),
            "variant_diversity": float(np.mean([r["variant_diversity"] for r in rows_w])),
            "distinct_scaffolds": float(np.mean([r["distinct_scaffolds"] for r in rows_w])),
            "mean_similarity_to_seeds": float(np.mean([r["mean_similarity_to_seeds"] for r in rows_w])),
            "mean_objective_quality": float(np.mean([r["mean_objective_quality"] for r in rows_w])),
        }
        summary[f"weight_{weight}"] = {k: round(v, 4) for k, v in means.items()}
        print(f"{weight:>7.1f} {means['variants']:>9.1f} "
              f"{means['variant_diversity']:>10.4f} {means['distinct_scaffolds']:>10.1f} "
              f"{means['mean_similarity_to_seeds']:>12.4f} "
              f"{means['mean_objective_quality']:>11.4f}")

    base, div = summary["weight_0.0"], summary["weight_0.5"]
    diversity_delta = div["variant_diversity"] - base["variant_diversity"]
    scaffold_delta = div["distinct_scaffolds"] - base["distinct_scaffolds"]
    quality_delta = div["mean_objective_quality"] - base["mean_objective_quality"]

    print()
    print(f"Diversity change      : {diversity_delta:+.4f} mean pairwise Tanimoto distance")
    print(f"Scaffold change       : {scaffold_delta:+.1f} distinct Murcko scaffolds")
    print(f"Objective change      : {quality_delta:+.4f} mean of the five base objectives")
    if diversity_delta <= 0.0 and scaffold_delta <= 0.0:
        verdict = ("Chemical tie-breaking did not broaden the search on these seeds. "
                   "On this population size the fronts may rarely overflow, which is "
                   "where the tie-break applies at all.")
    elif quality_delta < -0.02:
        verdict = ("Broader chemistry bought at a measurable objective cost. That is a "
                   "trade to make deliberately, not a free improvement.")
    else:
        verdict = ("Broader chemistry at no material objective cost on these seeds. "
                   "Three seeds is a small sample and this is not a benchmark against "
                   "any published generator.")
    print()
    print(verdict)

    out = ROOT / "reports" / "chemical_diversity_measurement.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "generations": GENERATIONS,
        "population_size": POPULATION,
        "seeds": list(SEEDS),
        "runs": rows,
        "summary": summary,
        "diversity_delta": round(diversity_delta, 4),
        "scaffold_delta": round(scaffold_delta, 4),
        "objective_delta": round(quality_delta, 4),
        "verdict": verdict,
        "caveat": (
            "Mean pairwise Tanimoto distance and Murcko scaffold counts measure "
            "spread, not quality. Neither says the structures can be made or do "
            "anything, and three seeds is too few to claim a general effect."
        ),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\nWritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
