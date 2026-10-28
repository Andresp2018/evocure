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
  to a threshold on the potency it is handed. In the published 100 generation run the
  five candidates span a factor of 1.7 in IC50 inside that 1330 fold uncertainty.

Reproduce with `scripts/investigate_potency_estimator.py`,
`scripts/investigate_potency_asymmetry.py` and `scripts/measure_tissue_transfer.py`.

---

## 1. Scientific usefulness, highest priority

| Action | Why |
|---|---|
| Expand the QSAR dataset | 43 compounds is the bottleneck. A conformal interval around 1330 fold makes any ranking nearly useless. With 200 to 500 homogeneous scaffolds from PANC-1 and MIA PaCa-2 in ChEMBL and GDSC, the model moves from ordering compounds to saying something about them. |
| Calibrate the tissue model | It is exploratory today. Without a fit to real data, meaning published spheroid, organoid or dose response measurements, simulated growth inhibition is only a restatement of the QSAR. |
| Propagate uncertainty | Instead of passing a point IC50 into the tissue model, sample from the conformal interval and report the resulting distribution of growth inhibition. The present design hides the uncertainty at the moment it matters most. |
| Add minimum medicinal chemistry filters | PAINS, synthetic accessibility, reactivity alerts and stability filters, so the search stops producing candidates no chemist would want to make. |

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
