"""Compare the molecular generator against baselines, instead of describing it alone.

A generator reported in isolation cannot be judged. The standard question is
whether it beats random mutation and whether it beats doing nothing clever, and
until that is answered its quality is unknown rather than good.

This runs three strategies on the same objective, the same seed compounds, the
same evaluation budget and the same scoring code:

  random_sampling   draw from the local compound library, no optimisation
  random_mutation   mutate seeds at random, keep nothing, the drift baseline
  hill_climbing     mutate and keep improvements, a greedy baseline
  nsga2_directed    the platform's own NSGA-II search

Budget is counted in objective evaluations, not generations, because comparing a
population of 24 over 4 generations with a population of 4 over 24 generations on
"generations" would be meaningless.

What this does not do. It is not GuacaMol or MOSES. Those need their own packages
and reference datasets and are the right next step; this measures the generator
against baselines that can run offline today, which is enough to answer "is the
directed search doing anything".

    python scripts/benchmark_generator.py --budget 400

Writes reports/generator_benchmark.json.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

from ai.predictors.qsar import CalibratedQSARPredictor
from simulation.chemistry.library import COMPOUND_LIBRARY
from simulation.chemistry.medchem_filters import assess, plausibility_objective
from simulation.evolution.molecular_ga import DirectedMolecularOptimizer

RDLogger.DisableLog("rdApp.*")

OUT = ROOT / "reports" / "generator_benchmark.json"
SEEDS = ["Nc1ccn([C@@H]2O[C@H](CO)[C@@H](O)C2(F)F)c(=O)n1",  # Gemcitabine
         "O=c1[nH]cc(F)c(=O)[nH]1",                          # 5-Fluorouracil
         "COc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC"]            # Erlotinib


class Objective:
    """One scoring function, shared by every strategy, counting its own calls."""

    def __init__(self) -> None:
        try:
            self.qsar = CalibratedQSARPredictor().load()
        except Exception:
            self.qsar = None
        self.calls = 0
        self._cache: dict[str, float] = {}

    def __call__(self, smiles: str) -> float:
        if smiles in self._cache:
            return self._cache[smiles]
        self.calls += 1

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            self._cache[smiles] = 0.0
            return 0.0

        # Potency where the model can speak, plausibility always. Deliberately the
        # same balance the platform uses: never potency alone.
        potency = 0.0
        if self.qsar is not None and getattr(self.qsar, "is_fitted", False):
            try:
                prediction = self.qsar.predict(smiles)
                value = (prediction.get("pic50") or {}).get("value")
                if isinstance(value, (int, float)):
                    potency = 1.0 / (1.0 + math.exp(-(float(value) - 6.0)))
            except Exception:
                potency = 0.0

        plausibility = plausibility_objective(assess(smiles))
        score = 0.6 * potency + 0.4 * plausibility
        self._cache[smiles] = score
        return score


def diversity(smiles_list: list[str]) -> dict[str, float]:
    """Internal diversity and scaffold count: a high score on one scaffold is not a win."""
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    mols = [m for m in mols if m is not None]
    if len(mols) < 2:
        return {"mean_pairwise_distance": 0.0, "unique_scaffolds": len(mols)}

    fps = [generator.GetFingerprint(m) for m in mols]
    distances = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            distances.append(1.0 - DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    scaffolds = {MurckoScaffold.MurckoScaffoldSmiles(mol=m) for m in mols}
    return {
        "mean_pairwise_distance": round(float(np.mean(distances)), 4),
        "unique_scaffolds": len(scaffolds),
    }


def _valid(smiles: str) -> bool:
    return Chem.MolFromSmiles(smiles) is not None


def run_random_sampling(objective: Objective, budget: int, rng) -> list[str]:
    pool = [e["smiles"] for e in COMPOUND_LIBRARY if e.get("smiles") and _valid(e["smiles"])]
    picks = [str(rng.choice(pool)) for _ in range(budget)]
    for smiles in picks:
        objective(smiles)
    return picks


def run_random_mutation(objective: Objective, budget: int, rng) -> list[str]:
    optimizer = DirectedMolecularOptimizer()
    seen = []
    population = list(SEEDS)
    while objective.calls < budget:
        parent = str(rng.choice(population))
        child, _ = optimizer.mutate_smiles(parent, rng=rng)
        if not _valid(child):
            continue
        objective(child)
        seen.append(child)
        # Drift: the child always replaces, regardless of score.
        population = [child] + population[:2]
    return seen


def run_hill_climbing(objective: Objective, budget: int, rng) -> list[str]:
    optimizer = DirectedMolecularOptimizer()
    current = str(rng.choice(SEEDS))
    best_score = objective(current)
    seen = [current]
    while objective.calls < budget:
        child, _ = optimizer.mutate_smiles(current, rng=rng)
        if not _valid(child):
            continue
        score = objective(child)
        seen.append(child)
        if score >= best_score:
            current, best_score = child, score
    return seen


def run_nsga2(objective: Objective, budget: int, rng, population_size: int = 20) -> list[str]:
    optimizer = DirectedMolecularOptimizer()
    population = list(SEEDS)
    seen = list(population)

    while objective.calls < budget:
        offspring = []
        for _ in range(population_size):
            if len(population) >= 2 and rng.random() < 0.65:
                a, b = rng.choice(population, size=2, replace=False)
                child, _ = optimizer.recombine_smiles(str(a), str(b), rng=rng)
            else:
                child, _ = optimizer.mutate_smiles(str(rng.choice(population)), rng=rng)
            if _valid(child):
                offspring.append(child)

        pool = list(dict.fromkeys(population + offspring))
        if not pool:
            break
        scored = [(s, objective(s)) for s in pool]
        seen.extend(offspring)
        scored.sort(key=lambda item: item[1], reverse=True)
        population = [s for s, _ in scored[:population_size]]

    return seen


STRATEGIES = {
    "random_sampling": run_random_sampling,
    "random_mutation": run_random_mutation,
    "hill_climbing": run_hill_climbing,
    "nsga2_directed": run_nsga2,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the generator against baselines")
    parser.add_argument("--budget", type=int, default=400,
                        help="objective evaluations per strategy (default: 400)")
    parser.add_argument("--repeats", type=int, default=3, help="independent runs per strategy")
    parser.add_argument("--top", type=int, default=10, help="top N kept for scoring")
    args = parser.parse_args()

    print("=" * 76)
    print("Generator benchmark")
    print("=" * 76)
    print(f"  Budget {args.budget} objective evaluations per run, "
          f"{args.repeats} runs per strategy")
    print(f"\n  {'strategy':18s} {'best':>7s} {'top10 mean':>11s} {'diversity':>10s} "
          f"{'scaffolds':>10s} {'valid':>7s} {'sec':>6s}")

    results: dict[str, dict] = {}
    for name, strategy in STRATEGIES.items():
        bests, means, divs, scaffs, valids, times = [], [], [], [], [], []
        for repeat in range(args.repeats):
            rng = np.random.default_rng(20261028 + repeat)
            objective = Objective()
            started = time.perf_counter()
            generated = strategy(objective, args.budget, rng)
            times.append(time.perf_counter() - started)

            unique = list(dict.fromkeys(g for g in generated if _valid(g)))
            valids.append(len(unique) / max(1, len(generated)))
            scored = sorted(((s, objective(s)) for s in unique),
                            key=lambda item: item[1], reverse=True)
            top = scored[:args.top]
            bests.append(top[0][1] if top else 0.0)
            means.append(float(np.mean([v for _, v in top])) if top else 0.0)
            metrics = diversity([s for s, _ in top])
            divs.append(metrics["mean_pairwise_distance"])
            scaffs.append(metrics["unique_scaffolds"])

        results[name] = {
            "best_score": round(float(np.mean(bests)), 4),
            "best_score_sd": round(float(np.std(bests, ddof=1)) if len(bests) > 1 else 0.0, 4),
            "top_mean_score": round(float(np.mean(means)), 4),
            "internal_diversity": round(float(np.mean(divs)), 4),
            "unique_scaffolds_in_top": round(float(np.mean(scaffs)), 1),
            "valid_fraction": round(float(np.mean(valids)), 4),
            "seconds_per_run": round(float(np.mean(times)), 1),
        }
        row = results[name]
        print(f"  {name:18s} {row['best_score']:>7.4f} {row['top_mean_score']:>11.4f} "
              f"{row['internal_diversity']:>10.4f} {row['unique_scaffolds_in_top']:>10.1f} "
              f"{row['valid_fraction']:>7.2f} {row['seconds_per_run']:>6.1f}")

    directed = results["nsga2_directed"]["top_mean_score"]
    baselines = {k: v["top_mean_score"] for k, v in results.items() if k != "nsga2_directed"}
    strongest = max(baselines, key=baselines.get)
    margin = directed - baselines[strongest]

    verdict = (
        f"The directed search beats its strongest baseline ({strongest}) by "
        f"{margin:+.4f} on top {args.top} mean score."
        if margin > 0 else
        f"The directed search does not beat {strongest}, which scores "
        f"{-margin:+.4f} higher. On this objective the extra machinery is not "
        f"earning its cost."
    )
    print(f"\n  {verdict}")

    diversity_note = ""
    best_div = max(results, key=lambda k: results[k]["internal_diversity"])
    if best_div != "nsga2_directed":
        diversity_note = (
            f"{best_div} explores more chemical space than the directed search "
            f"({results[best_div]['internal_diversity']} against "
            f"{results['nsga2_directed']['internal_diversity']}), which matches the "
            f"structural convergence seen in long production runs."
        )
        print(f"  {diversity_note}")

    # random_sampling draws from a library of real, mostly approved drugs. It is a
    # retrieval baseline, not a generative one, and its score says as much about the
    # objective as about search. Reporting it without that note would be misleading
    # in the direction that flatters nobody.
    retrieval_note = (
        "random_sampling retrieves known compounds from the local library rather than "
        "generating new ones, so a high score there reflects the library containing "
        "approved drugs. It is included as the question every generative method should "
        "have to answer: would picking from a catalogue have done as well?"
    )
    print(f"\n  {retrieval_note}")

    payload = {
        "retrieval_baseline_note": retrieval_note,
        "budget_evaluations": args.budget,
        "repeats": args.repeats,
        "top_n": args.top,
        "seed_compounds": SEEDS,
        "objective": (
            "0.6 x calibrated QSAR potency utility, sigmoidal at pIC50 6.0, plus "
            "0.4 x medicinal chemistry plausibility. Potency alone is never optimised."
        ),
        "results": results,
        "verdict": verdict,
        "diversity_note": diversity_note,
        "scope": (
            "Offline baselines only. This is not GuacaMol or MOSES, which require their "
            "own packages and reference sets and remain the right external comparison. "
            "It answers whether the directed search beats random and greedy search on "
            "this objective, nothing more."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWritten to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
