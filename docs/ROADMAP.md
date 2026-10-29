# Roadmap

Where the project actually stands, and what would move it forward.

The platform is engineered and tested. It is not validated. The code runs end to end,
the suite passes, CI is green and the reporting states its own limits, which is exactly
why the remaining work is scientific rather than technical.

Three measurements set the agenda below, all reproducible from this repository:

- The PDAC potency model trains on 43 compounds over 32 independent scaffolds. Repeated
  scaffold-grouped cross validation gives RMSE 1.34 log units and a 95% conformal
  interval of 3.12 log units, a factor of about 1330 on IC50.
- A learning curve over held-out scaffolds falls from RMSE 2.04 to 1.04 between 8 and 27
  training scaffolds and is still descending. The estimator is data limited.
- Holding everything fixed except the assigned IC50 shows the tissue model behaves close
  to a threshold on the potency it is handed. In the published 20 generation run the
  five candidates span a factor of 1.7 in IC50 inside that 1330 fold uncertainty.

Reproduce with `scripts/investigate_potency_estimator.py`,
`scripts/investigate_potency_asymmetry.py` and `scripts/measure_tissue_transfer.py`.

[DATA_SOURCES.md](DATA_SOURCES.md) says where the missing data lives and how to
import it. Items 1 and 2 below are both data problems before they are code problems.

---

## 1. Scientific usefulness, highest priority

| Action | Why |
|---|---|
| Expand the QSAR dataset | 43 compounds is the bottleneck. A conformal interval around 1330 fold makes any ranking nearly useless. **This has now been attempted and measured, and ChEMBL's PANC-1 IC50 data does not supply it.** See the section below before repeating the attempt. |
| Calibrate the tissue model | It is exploratory today. Without a fit to real data, meaning published spheroid, organoid or dose response measurements, simulated growth inhibition is only a restatement of the QSAR. |
| Propagate uncertainty | Instead of passing a point IC50 into the tissue model, sample from the conformal interval and report the resulting distribution of growth inhibition. The present design hides the uncertainty at the moment it matters most. |
| Add minimum medicinal chemistry filters | PAINS, synthetic accessibility, reactivity alerts and stability filters, so the search stops producing candidates no chemist would want to make. |

### What happened when the expansion was tried

The tooling now runs end to end, and the answer it gives is negative. This
section records it so the next person does not spend the same effort reaching the
same place.

`scripts/fetch_chembl_bioactivity.py` pulls the data, `scripts/import_bioactivity.py`
gates it, and `scripts/train_qsar.py --dataset` trains on it. That last argument
did not exist: the expanded file the previous instructions told you to build was
written and never read, so the retrain step silently used the curated 43
compounds again.

**What ChEMBL supplies.** PANC-1 IC50, ChEMBL_37: 3140 records examined, 2253
usable, 2070 unique compounds, 900 Murcko scaffolds. Merged with the curated set,
2153 compounds over 952 scaffolds. On the face of it, exactly the 200 to 500
scaffolds this roadmap asked for, nearly twice over.

**What the trainer reported.** Retraining on it looks like a decisive win:

| | curated | expanded |
|---|---|---|
| compounds / scaffolds | 43 / 32 | 2106 / 919 |
| RMSE, log units | 1.3397 | 0.8524 |
| conformal q95, log units | 3.1237 | 1.7814 |
| R2 | 0.6987 | 0.3733 |
| ROC AUC | 0.9535 | 0.6945 |

A conformal quantile of 3.1237 is a factor of 1330 on IC50; 1.7814 is a factor
of 60. That reads as the 22 fold tightening this roadmap predicted.

**It is not one.** Those columns are computed on different populations, so none
of them is a before and after of the same quantity. The curated set was built to
span potency, pIC50 variance 5.96; the ChEMBL set concentrates, variance 1.16.
Since R2 = 1 - MSE / variance, the smaller denominator alone accounts for the R2
fall: 1 - 1.3397^2/5.96 = 0.699 and 1 - 0.8524^2/1.16 = 0.373 reproduce both
reported values. By the same logic the RMSE and conformal improvements are also
population effects, not accuracy. The ROC AUC fall is the matching artifact on
the classifier: the active label, IC50 below 10 uM, cuts the curated bimodal
distribution cleanly and lands inside the bulk of the ChEMBL one.

**The comparison that is valid.** `scripts/measure_dataset_expansion.py` trains
both models with the held-out scaffolds removed from each and scores both on the
same curated compounds with the same targets. Over five holdout draws the
expanded model was **worse on four of five**, RMSE +0.2304 log units on average
(range -0.4218 to +0.8130). The conformal quantile improved on average, -0.3850,
but ranged from -1.4642 to +1.3925, so that is not established either.
`reports/dataset_expansion_comparison.json` has the per seed numbers.

**Why.** "IC50 | PANC-1" is not one assay. On ChEMBL_37 it is **355 distinct
assays from 325 distinct papers, median three records each, largest 70, which is
3.1% of the file**. Only four assays have 40 or more records. The importer's
comparability key groups on endpoint, cell line and organism, so it sees one
group where there are 355 protocols with different incubation times, readouts and
conventions. The 26 duplicate disagreements it reported, up to 203 fold on the
same compound, are that heterogeneity measured directly. The fetcher now prints
this fragmentation and warns before you import.

**What this means for the plan.** The single largest protocol in ChEMBL's PANC-1
IC50 data holds 70 records. An assay-homogeneous set of hundreds of scaffolds
therefore **does not exist in this source**, and the instruction to assemble one
from it cannot be followed as written. More compounds from more papers is not
more data for this purpose.

Where to look instead:

- **GDSC**, which runs one consistent protocol across a large compound panel.
  That is the shape of source this needs, and it is now the first recommendation
  rather than one of four.
- **A single large published panel**, imported on its own, accepting a few hundred
  compounds from one protocol over thousands from many.
- **Hierarchical or multi-task modelling** that treats each assay as its own task
  with a shared representation, rather than pooling them into one target. This
  uses the heterogeneous data honestly instead of pretending it is homogeneous,
  and is the only route that makes the 952 scaffolds usable.
- **Extending the comparability key** to include the source assay or document, so
  the importer can see the fragmentation its own design is meant to prevent.

What has not changed: the curated 43 compound model is still the one shipped, and
its conformal interval is still a factor of about 1330.

## 2. Methodological novelty

| Direction | Description | Degree of novelty |
|---|---|---|
| Closed loop with real experimental data | Rather than simulating the tissue, couple the loop to a real assay or to a calibrated digital twin. Few open source platforms do this honestly. | High |
| Multi-scale evolution | Combine molecular generation with evolution of microenvironment parameters, or of dosing regimens, inside the same loop. | Medium to high |
| Rigorous public benchmark | A standard task set with external data, always reported against baselines such as REINVENT, classical graph GAs and random search. | Medium, but very useful |
| Decision under uncertainty | Have selection optimise explicitly for robustness, for example maximising the probability of exceeding a threshold rather than the mean value. | Medium |

## 3. Practical usefulness

- **Medicinal chemist mode.** Produce not only the structure but a simplified synthetic
  route proposal, a synthetic accessibility estimate, and flags for known problems.
- **Fair comparison always visible.** The report already does something useful with the
  third control arm. Make that standard and make it clearer still.
- **Integration with public databases.** For each candidate, query PubChem, ChEMBL and
  SureChEMBL automatically and show the nearest Tanimoto and whether the structure
  already exists. The current novelty check searches 122 local structures, which is a
  sanity check and not a prior art search.
- **Focus on a concrete problem.** Rather than a generic platform, specialise in a niche
  where open source is weak, for example nucleosides for PDAC, or resistance models in a
  hypoxic microenvironment.

## 4. Recommended order of work

1. **Data first.** Expand and homogenise the QSAR dataset. This is the largest lever and
   everything else is downstream of it.
2. **Calibrate or remove the tissue model.** Failing that, be even more explicit that it
   is illustrative only.
3. **Add medicinal chemistry filters and automatic prior art.**
4. **Public benchmark against existing methods.**
5. **Only then** explore more sophisticated or multi-scale loops.

---

## Longer horizons

Worth doing eventually, but none of these fixes the constraint above, and attempting them
first would add capability to a platform whose outputs cannot yet be trusted.

- **3D volumetric organoid PDEs.** Extend the 2D finite-volume reaction-diffusion solver
  to 3D patient-derived organoid meshes, with spatial transcriptomics slices as initial
  cellular coordinates.
- **Structure-based docking.** Optional local integration with open source 3D
  protein-ligand scoring against targets such as KRAS G12D or EGFR T790M, alongside the
  existing ligand-based QSAR.
- **Offline ChEMBL and DepMap warehouse importers.** Streaming ingestion of full ChEMBL
  and DepMap dumps through `ScientificDataManager.import_local_records`. This one is
  partly item 1 above and is the version of it worth building.
- **Prospective wet-lab loops.** Active-learning experimental design closing the loop
  with in vitro spheroid assays.

---

## A note on the previous version of this file

Earlier releases described the platform as "100% complete" with all milestones "verified",
and quoted a classifier ROC AUC of 0.9167 as a headline figure.

That figure came from a single scaffold split with roughly ten test compounds. Re-running
the same routine across ten random seeds returns R2 anywhere from -0.09 to 0.82. The
number described the split, not the model.

Development since then found fabricated descriptors in the coevolution engine, a selection
loop that excluded the tumour model it claimed to optimise against, a potency scoring
asymmetry that made the comparison against reference drugs unwinnable by construction, and
several portability faults that only a GPU-less CI runner could catch. All had passing
tests and plausible-looking output.

"Complete" was not an accurate description, and this file no longer claims it.
