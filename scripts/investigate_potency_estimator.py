"""Can the potency estimator for novel structures be fixed, and how.

The screening report's binding constraint is that generated structures get their
potency from a structure-property heuristic whose output range cannot reach the
curated values of reference drugs. Two repairs are usually proposed: train on more
data, or calibrate the heuristic against the drugs we do have. This script measures
which, if either, is worth doing.

A note on the headline metric already in the repository. The reported scaffold-split
R2 of 0.32 comes from a single split with about ten test compounds. An R2 from ten
points carries enormous variance, so it is not a usable estimate of anything. Every
number here instead comes from repeated scaffold-grouped cross validation, which is
slower and much harder to flatter.

Experiments:

  1. Inventory. How much data, how many independent scaffolds, what range.
  2. Baselines. Does the current model beat predicting the training mean? Any model
     that does not has learned nothing worth shipping.
  3. Learning curve. Is performance still climbing at the current sample size? This
     is what decides whether more data is the answer.
  4. Calibration. Refit the heuristic's parameters against curated drugs and see
     whether its range and accuracy improve.
  5. Range check. After calibration, can it reach the reference drugs at all?

    python scripts/investigate_potency_estimator.py

Writes reports/potency_estimator_study.json. This characterises models trained on
90 curated literature values; it is not a claim about any molecule.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error

from ai.predictors.data_connectors import ScientificDataManager
from ai.predictors.qsar import CalibratedQSARPredictor

OUT = ROOT / "reports" / "potency_estimator_study.json"
SEED = 20261027
N_REPEATS = 12
N_FOLDS = 5


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load(partition: str) -> list[dict]:
    manager = ScientificDataManager()
    manager.load_reference_dataset(disease_id=partition)
    rows = []
    for record in manager.get_records():
        assay = record.get("bioactivity") or {}
        smiles, pic50 = record.get("canonical_smiles"), assay.get("pic50")
        if not smiles or pic50 is None:
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        rows.append({
            "name": record.get("name"),
            "smiles": smiles,
            "pic50": float(pic50),
            "ic50_um": 10.0 ** (6.0 - float(pic50)),
            "scaffold": MurckoScaffold.MurckoScaffoldSmiles(mol=mol),
            "mol_weight": float(Descriptors.MolWt(mol)),
            "logp": float(Descriptors.MolLogP(mol)),
            "qed": float(QED.qed(mol)),
            "disease": str(assay.get("disease_id", partition)),
        })
    return rows


# ---------------------------------------------------------------------------
# The heuristic, and a refittable version of it
# ---------------------------------------------------------------------------

def heuristic_terms(row: dict) -> tuple[float, float, float]:
    logp_opt = max(0.0, 1.0 - abs(row["logp"] - 2.5) / 3.0)
    mw_opt = max(0.0, 1.0 - abs(row["mol_weight"] - 400.0) / 300.0)
    return logp_opt, mw_opt, row["qed"]


def potency_factor(row: dict) -> float:
    logp_opt, mw_opt, qed = heuristic_terms(row)
    return max(0.05, 0.5 * logp_opt + 0.3 * mw_opt + 0.2 * qed)


def heuristic_pic50(row: dict, intercept: float = 3.0, slope: float = 3.2) -> float:
    ic50_um = math.exp(intercept - slope * potency_factor(row))
    return 6.0 - math.log10(max(1e-12, ic50_um))


# ---------------------------------------------------------------------------
# Models, each exposing fit(rows) -> predict(rows)
# ---------------------------------------------------------------------------

def model_mean(train: list[dict]):
    value = float(np.mean([r["pic50"] for r in train]))
    return lambda rows: np.full(len(rows), value)


def model_heuristic_current(train: list[dict]):
    return lambda rows: np.array([heuristic_pic50(r) for r in rows])


def model_heuristic_refit_ab(train: list[dict]):
    """Keep the hand-chosen descriptor weights, refit only intercept and slope."""
    x = np.array([[potency_factor(r)] for r in train])
    y = np.array([math.log(max(1e-12, r["ic50_um"])) for r in train])
    fit = LinearRegression().fit(x, y)

    def predict(rows):
        xs = np.array([[potency_factor(r)] for r in rows])
        ln_ic50 = fit.predict(xs)
        return 6.0 - np.log10(np.clip(np.exp(ln_ic50), 1e-12, None))
    predict.params = {"intercept": float(fit.intercept_), "slope": float(-fit.coef_[0])}
    return predict


def model_heuristic_refit_full(train: list[dict]):
    """Refit the three descriptor terms as well, rather than trusting 0.5/0.3/0.2."""
    x = np.array([heuristic_terms(r) for r in train])
    y = np.array([math.log(max(1e-12, r["ic50_um"])) for r in train])
    fit = LinearRegression().fit(x, y)

    def predict(rows):
        xs = np.array([heuristic_terms(r) for r in rows])
        ln_ic50 = fit.predict(xs)
        return 6.0 - np.log10(np.clip(np.exp(ln_ic50), 1e-12, None))
    predict.params = {"intercept": float(fit.intercept_),
                      "coef_logp_opt": float(fit.coef_[0]),
                      "coef_mw_opt": float(fit.coef_[1]),
                      "coef_qed": float(fit.coef_[2])}
    return predict


_FEATURISER = CalibratedQSARPredictor()


def _fingerprints(rows: list[dict]) -> np.ndarray:
    return np.array([_FEATURISER.featurize(r["smiles"]) for r in rows])


def _descriptors(rows: list[dict]) -> np.ndarray:
    return np.array([[r["mol_weight"] / 1000.0, r["logp"] / 5.0, r["qed"]] for r in rows])


def model_rf_fingerprint(train: list[dict]):
    """The shipped model: random forest over 1024-bit Morgan plus 8 descriptors."""
    fit = RandomForestRegressor(n_estimators=100, random_state=SEED)
    fit.fit(_fingerprints(train), [r["pic50"] for r in train])
    return lambda rows: fit.predict(_fingerprints(rows))


def model_rf_descriptors(train: list[dict]):
    fit = RandomForestRegressor(n_estimators=100, random_state=SEED)
    fit.fit(_descriptors(train), [r["pic50"] for r in train])
    return lambda rows: fit.predict(_descriptors(rows))


def model_ridge_descriptors(train: list[dict]):
    fit = Ridge(alpha=1.0).fit(_descriptors(train), [r["pic50"] for r in train])
    return lambda rows: fit.predict(_descriptors(rows))


MODELS = {
    "predict_training_mean": model_mean,
    "heuristic_as_shipped": model_heuristic_current,
    "heuristic_refit_2_params": model_heuristic_refit_ab,
    "heuristic_refit_4_params": model_heuristic_refit_full,
    "random_forest_fingerprint": model_rf_fingerprint,
    "random_forest_descriptors": model_rf_descriptors,
    "ridge_descriptors": model_ridge_descriptors,
}


# ---------------------------------------------------------------------------
# Scaffold grouped cross validation
# ---------------------------------------------------------------------------

def scaffold_folds(rows: list[dict], n_folds: int, rng: np.random.Generator):
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["scaffold"]].append(index)
    keys = list(groups)
    rng.shuffle(keys)
    folds: list[list[int]] = [[] for _ in range(n_folds)]
    # Largest scaffold groups first into the currently smallest fold.
    for key in sorted(keys, key=lambda k: -len(groups[k])):
        target = min(range(n_folds), key=lambda f: len(folds[f]))
        folds[target].extend(groups[key])
    return folds


def cross_validate(rows: list[dict], builder, n_repeats: int = N_REPEATS) -> dict:
    errors, actuals, predictions = [], [], []
    for repeat in range(n_repeats):
        rng = np.random.default_rng(SEED + repeat)
        folds = scaffold_folds(rows, N_FOLDS, rng)
        for fold in folds:
            test = [rows[i] for i in fold]
            train = [rows[i] for i in range(len(rows)) if i not in set(fold)]
            if len(train) < 8 or not test:
                continue
            predict = builder(train)
            yhat = np.asarray(predict(test), dtype=float)
            ytrue = np.array([r["pic50"] for r in test], dtype=float)
            errors.extend((yhat - ytrue).tolist())
            actuals.extend(ytrue.tolist())
            predictions.extend(yhat.tolist())

    errors = np.asarray(errors)
    actuals = np.asarray(actuals)
    predictions = np.asarray(predictions)
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))
    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((actuals - actuals.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    spearman = float(stats.spearmanr(actuals, predictions).statistic) if len(set(predictions)) > 1 else float("nan")
    return {"rmse_log_units": round(rmse, 3), "mae_log_units": round(mae, 3),
            "r2": round(r2, 3), "spearman": round(spearman, 3),
            "n_predictions": int(errors.size)}


def main() -> int:
    report: dict = {"seed": SEED, "n_repeats": N_REPEATS, "n_folds": N_FOLDS}

    pdac = load("PDAC")
    everything = load("ALL")

    # ------------------------------------------------------------ 1. inventory
    print("=" * 78)
    print("1. What there is to learn from")
    print("=" * 78)
    for label, rows in (("PDAC partition (used for training)", pdac),
                        ("All partitions (not assay comparable)", everything)):
        scaffolds = {r["scaffold"] for r in rows}
        pic50s = [r["pic50"] for r in rows]
        print(f"  {label:38s} n={len(rows):3d}  scaffolds={len(scaffolds):3d}  "
              f"pIC50 {min(pic50s):.2f} to {max(pic50s):.2f}  sd={np.std(pic50s):.2f}")
        report[("pdac" if "PDAC" in label else "all") + "_inventory"] = {
            "n": len(rows), "scaffolds": len(scaffolds),
            "pic50_min": round(min(pic50s), 3), "pic50_max": round(max(pic50s), 3),
            "pic50_sd": round(float(np.std(pic50s)), 3),
        }
    print(f"\n  {len({r['scaffold'] for r in pdac})} independent scaffolds is the real sample")
    print("  size for a scaffold split, not the compound count.")

    # ------------------------------------------------------------ 2. baselines
    print("\n" + "=" * 78)
    print("2. Does anything beat predicting the mean?")
    print("=" * 78)
    print(f"  Repeated scaffold-grouped CV, {N_FOLDS} folds x {N_REPEATS} repeats, PDAC partition")
    print(f"\n  {'model':28s} {'RMSE':>7s} {'MAE':>7s} {'R2':>7s} {'Spearman':>9s}")

    baselines = {}
    for name, builder in MODELS.items():
        result = cross_validate(pdac, builder)
        baselines[name] = result
        print(f"  {name:28s} {result['rmse_log_units']:>7.3f} {result['mae_log_units']:>7.3f} "
              f"{result['r2']:>7.3f} {result['spearman']:>9.3f}")
    report["baselines_pdac"] = baselines

    mean_rmse = baselines["predict_training_mean"]["rmse_log_units"]
    beat = {k: v for k, v in baselines.items()
            if k != "predict_training_mean" and v["rmse_log_units"] < mean_rmse}
    report["models_beating_the_mean"] = sorted(beat, key=lambda k: beat[k]["rmse_log_units"])
    print(f"\n  Beating the trivial mean baseline (RMSE {mean_rmse}): "
          f"{', '.join(report['models_beating_the_mean']) or 'none'}")

    # ------------------------------------------------------- 3. learning curve
    print("\n" + "=" * 78)
    print("3. Would more data help?")
    print("=" * 78)

    best_name = (report["models_beating_the_mean"] or ["random_forest_fingerprint"])[0]
    print(f"  Learning curve for {best_name}, scaffolds held out for testing.\n")
    print(f"  {'train scaffolds':>16s} {'train rows':>11s} {'RMSE':>8s} {'Spearman':>9s}")

    curve = []
    groups = defaultdict(list)
    for index, row in enumerate(pdac):
        groups[row["scaffold"]].append(index)
    keys = list(groups)

    for fraction in (0.25, 0.40, 0.55, 0.70, 0.85):
        rmses, spearmans, sizes, scaffold_counts = [], [], [], []
        for repeat in range(N_REPEATS):
            rng = np.random.default_rng(SEED + 100 + repeat)
            shuffled = list(keys)
            rng.shuffle(shuffled)
            cut = max(2, int(len(shuffled) * fraction))
            train_keys, test_keys = shuffled[:cut], shuffled[cut:]
            train = [pdac[i] for k in train_keys for i in groups[k]]
            test = [pdac[i] for k in test_keys for i in groups[k]]
            if len(train) < 8 or len(test) < 4:
                continue
            predict = MODELS[best_name](train)
            yhat = np.asarray(predict(test), dtype=float)
            ytrue = np.array([r["pic50"] for r in test], dtype=float)
            rmses.append(float(np.sqrt(mean_squared_error(ytrue, yhat))))
            if len(set(yhat.tolist())) > 1:
                spearmans.append(float(stats.spearmanr(ytrue, yhat).statistic))
            sizes.append(len(train))
            scaffold_counts.append(len(train_keys))
        if not rmses:
            continue
        entry = {"fraction": fraction,
                 "train_scaffolds": round(float(np.mean(scaffold_counts)), 1),
                 "train_rows": round(float(np.mean(sizes)), 1),
                 "rmse": round(float(np.mean(rmses)), 3),
                 "spearman": round(float(np.mean(spearmans)), 3) if spearmans else None}
        curve.append(entry)
        print(f"  {entry['train_scaffolds']:>16.1f} {entry['train_rows']:>11.1f} "
              f"{entry['rmse']:>8.3f} "
              f"{(entry['spearman'] if entry['spearman'] is not None else float('nan')):>9.3f}")
    report["learning_curve"] = curve

    if len(curve) >= 2:
        improvement = curve[0]["rmse"] - curve[-1]["rmse"]
        report["rmse_gained_from_quadrupling_data"] = round(improvement, 3)
        still_climbing = improvement > 0.15
        report["learning_curve_still_climbing"] = bool(still_climbing)
        print(f"\n  RMSE change from {curve[0]['train_scaffolds']:.0f} to "
              f"{curve[-1]['train_scaffolds']:.0f} training scaffolds: {improvement:+.3f} log units")
        print("  " + ("Still improving: more data is the lever." if still_climbing else
                      "Essentially flat: more data of this kind is not the lever."))

    # --------------------------------------------------------- 4. calibration
    print("\n" + "=" * 78)
    print("4. What calibration buys")
    print("=" * 78)

    fitted_ab = model_heuristic_refit_ab(pdac)
    fitted_full = model_heuristic_refit_full(pdac)
    report["calibrated_parameters"] = {
        "two_parameter": fitted_ab.params,
        "four_parameter": fitted_full.params,
        "as_shipped": {"intercept": 3.0, "slope": 3.2},
    }
    print(f"  As shipped        IC50 = exp(3.000 - 3.200 f)")
    print(f"  Refit on curated  IC50 = exp({fitted_ab.params['intercept']:.3f} - "
          f"{fitted_ab.params['slope']:.3f} f)")

    shipped = baselines["heuristic_as_shipped"]
    refit2 = baselines["heuristic_refit_2_params"]
    refit4 = baselines["heuristic_refit_4_params"]
    print(f"\n  RMSE as shipped          {shipped['rmse_log_units']:.3f}")
    print(f"  RMSE refit 2 parameters  {refit2['rmse_log_units']:.3f}")
    print(f"  RMSE refit 4 parameters  {refit4['rmse_log_units']:.3f}")
    print(f"  RMSE trivial mean        {mean_rmse:.3f}")

    # ---------------------------------------------------------- 5. range check
    print("\n" + "=" * 78)
    print("5. Can a calibrated heuristic reach the reference drugs?")
    print("=" * 78)

    sweep = [{"mol_weight": mw, "logp": lp, "qed": q}
             for mw in range(200, 701, 20)
             for lp in [x / 4 for x in range(-8, 25)]
             for q in [0.2, 0.4, 0.6, 0.8, 0.95]]

    ranges = {}
    for label, predictor in (("as_shipped", lambda r: heuristic_pic50(r)),
                             ("refit_2_params", lambda r: float(fitted_ab([r])[0])),
                             ("refit_4_params", lambda r: float(fitted_full([r])[0]))):
        values = [predictor(r) for r in sweep]
        best_ic50 = 10.0 ** (6.0 - max(values))
        worst_ic50 = 10.0 ** (6.0 - min(values))
        ranges[label] = {"most_potent_ic50_um": round(best_ic50, 4),
                         "least_potent_ic50_um": round(worst_ic50, 2)}
        print(f"  {label:16s} reachable IC50 range: "
              f"{best_ic50:.3f} to {worst_ic50:.1f} uM")
    report["reachable_ranges"] = ranges

    references = [r for r in pdac if r["ic50_um"] < 1.0]
    references.sort(key=lambda r: r["ic50_um"])
    print(f"\n  Curated compounds below 1 uM that a generated structure would have to match:")
    covered = {label: 0 for label in ranges}
    for row in references[:6]:
        line = f"    {str(row['name'])[:22]:22s} curated {row['ic50_um']:8.4f} uM"
        for label, info in ranges.items():
            reachable = row["ic50_um"] >= info["most_potent_ic50_um"]
            covered[label] += int(reachable)
            line += f"   {label}:{'reachable' if reachable else 'OUT OF RANGE'}"
        print(line)
    report["sub_micromolar_references"] = len(references)
    report["reference_coverage"] = {k: f"{v}/{min(6, len(references))}" for k, v in covered.items()}

    # ------------------------------------------------------------- conclusion
    print("\n" + "=" * 78)
    print("Conclusion")
    print("=" * 78)
    conclusions = []
    if not report["models_beating_the_mean"]:
        conclusions.append(
            "No model tested beats predicting the training mean under scaffold-grouped "
            "CV. At this sample size the estimator carries no usable signal, and "
            "calibration cannot create any.")
    else:
        conclusions.append(
            f"Best under scaffold CV: {report['models_beating_the_mean'][0]}. "
            f"It beats the trivial mean baseline, but check the margin before relying on it.")
    if report.get("learning_curve_still_climbing"):
        conclusions.append(
            "The learning curve is still improving at the current size, so a larger "
            "assay-homogeneous training set is the highest value work.")
    else:
        conclusions.append(
            "The learning curve is flat across the range tested, so simply adding more "
            "compounds of this kind is unlikely to help on its own. The representation "
            "or the assay heterogeneity is the limit.")
    worst_cover = min(covered.values()) if covered else 0
    if worst_cover < min(6, len(references)):
        conclusions.append(
            "Even refitted, the heuristic's functional form cannot reach the most potent "
            "curated drugs, so recalibration alone will not make generated structures "
            "comparable to reference therapeutics.")
    for line in conclusions:
        print(f"  - {line}")
    report["conclusions"] = conclusions

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWritten to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
