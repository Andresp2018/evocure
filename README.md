# EVOCURE

An open source in silico platform for early stage compound screening, built around a
simulated tumour microenvironment, a calibrated QSAR model and a bounded evolutionary
search over molecular structures. It runs entirely on your own machine, on any GPU or on
no GPU at all.

**What this is:** a hypothesis generator. It proposes chemically valid structures and
ranks them against predicted activity, toxicity alerts, drug likeness and applicability
domain, then exercises the top ranked ones in a simulated tissue.

**What this is not:** evidence about human disease. Nothing this software produces has
been synthesised or tested in a cell, an animal or a person. Read
[SCIENTIFIC_LIMITATIONS_AND_EPISTEMICS.md](docs/SCIENTIFIC_LIMITATIONS_AND_EPISTEMICS.md)
before drawing a conclusion from any number it prints.

---

## Example output

[reports/EVOCURE_PDAC_Candidate_Report.pdf](reports/EVOCURE_PDAC_Candidate_Report.pdf)
is a complete screening run on pancreatic ductal adenocarcinoma: five prioritised
candidates, their predicted properties with uncertainty intervals, their behaviour in the
simulated microenvironment, and a statistical comparison against positive and negative
controls.

The report is generated directly from the run's results file. Nothing in it is written by
hand, and it reports what the controls showed whether or not that flatters the candidates.

Reproduce it:

```bash
python scripts/train_qsar.py
EVOCURE_COMPUTE=cpu EVOCURE_DISABLE_LAYA_NEURAL=1 python scripts/run_pdac_report_experiment.py
python scripts/build_candidate_pdf.py
```

---

## Install

Full instructions for both platforms: [docs/INSTALL.md](docs/INSTALL.md).

### Windows

```powershell
git clone https://github.com/Andresp2018/evocure.git
cd evocure
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
.\.venv\Scripts\python.exe scripts\train_qsar.py
.\start_evocure.bat
```

### Linux

```bash
git clone https://github.com/Andresp2018/evocure.git
cd evocure
bash scripts/setup_env.sh
./.venv/bin/python scripts/train_qsar.py
./start_evocure.sh
```

Then open http://127.0.0.1:8000.

---

## GPU support

Works on NVIDIA, AMD, Intel and Apple GPUs through one code path, and on the CPU alone.
Check what your machine will use:

```bash
python -m simulation.core.acceleration
```

| Vendor | Backend | PyTorch install |
|---|---|---|
| NVIDIA | CUDA | `pip install torch --index-url https://download.pytorch.org/whl/cu124` |
| AMD, Linux | ROCm | `pip install torch --index-url https://download.pytorch.org/whl/rocm6.2` |
| AMD or Intel, Windows | DirectML | `pip install torch-directml` |
| Intel Arc | XPU | `pip install torch --index-url https://download.pytorch.org/whl/xpu` |
| Apple Silicon | Metal | `pip install torch` |
| Any | CPU | `pip install torch --index-url https://download.pytorch.org/whl/cpu` |

Select one explicitly with `EVOCURE_COMPUTE=cuda|rocm|directml|xpu|mps|cpu`. An
unavailable backend falls back to the CPU rather than failing.

DirectML and Metal have no float64, so those run the substrate fields in float32 and say
so in the status output. The CPU path is the reproducibility reference.

Details, per vendor troubleshooting and the precision notes:
[docs/GPU_SUPPORT.md](docs/GPU_SUPPORT.md).

---

## AI models are not in this repository

No model weights are committed here. The `models/` directory ships only a README and a
manifest of filenames, sizes and SHA-256 checksums.

```bash
python scripts/fetch_models.py --status   # what you have
python scripts/fetch_models.py --verify   # checksum it
```

| Artifact | Needed for | How to get it |
|---|---|---|
| PDAC QSAR | Activity prediction for novel compounds | `python scripts/train_qsar.py`, trains in seconds from the committed dataset |
| Gemma GGUF | Optional language assistant | Download yourself, see [models/README.md](models/README.md) |
| Laya checkpoint | Optional neural triage | Download yourself |
| llama.cpp runtime | Running the GGUF | [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases), build matching your GPU |

EVOCURE runs with all of these absent. Each has a deterministic fallback, and the
interface always names which provider answered. The example report was produced with the
fallbacks active, so you do not need any downloaded weights to reproduce it.

---

## What it does

**Chemical universe.** Particles move, collide and react under a rule based chemistry with
reproduction and inherited variation. A sandbox for artificial chemical ecosystems, kept
strictly separate from the biomedical models.

Survival traits are derived from each entity's molecular structure, so selection acts on
chemistry: catalytic groups and polar surface set metabolic efficiency, hydrolysable bonds
set the decay rate, ring systems set stability, and molecular size sets the cost of
copying. With structural mutation disabled, a hydrolysable anhydride lineage is eliminated
while a stable aromatic one grows, driven by nothing but the structures.

Earlier releases evolved four free-standing numbers while the molecule was a label that
nothing read. Open-endedness is measured with Bedau style activity statistics and is **not**
claimed: runs so far classify as class 1, selecting and then settling.

**Candidate discovery.** A fifteen stage pipeline: seed import, structural variant
generation, RDKit validation, descriptor calculation, calibrated QSAR with conformal
intervals, ADMET and ten category toxicology, applicability domain check, multi objective
scoring, triage, controlled tissue simulation, NSGA-II Pareto sorting, and population
update across generations.

Selection runs in one of two modes, and which one you use changes what the search
actually optimises:

| `tissue_in_fitness` | Objectives | What drives selection |
|---|---|---|
| `False` (default) | 5 | QSAR prediction, toxicity, drug likeness, domain. The tissue simulation is a readout only |
| `True` | 6 | The above plus simulated growth inhibition, with every dosable candidate simulated |

The closed loop is what makes this evolution *against a tumour model* rather than against
a QSAR surrogate. It costs roughly one second of CPU per candidate per generation. The
published example report uses it. Be aware that the tissue model's drug response is
driven by the predicted IC50, so the sixth objective correlates with the first rather
than being independent evidence.

**Disease models.** Pancreatic ductal adenocarcinoma, KRAS mutant lung cancer, resistant
bacterial biofilm and MASLD liver disease. Each is a two dimensional agent based tissue
with oxygen and nutrient reaction diffusion, drug transport, clonal populations and
resistance. The latter two are not cancers.

Novel compound activity prediction exists for PDAC only. Other diseases rank on chemistry
and report activity as unavailable rather than inventing a number.

**Pharmacology and toxicology.** ADMET estimation, a ten category structural alert and
rule based risk index, and PBPK modelling. Outputs are labelled with their epistemic
status; values without supporting model data are reported as unknown rather than guessed.

**Interfaces.** A Godot 4 desktop client and a browser frontend, both driven by the same
FastAPI engine. OpenAPI documentation at `/docs` when running.

---

## Verify the install

```bash
# Linux and macOS
EVOCURE_DISABLE_LAYA_NEURAL=1 EVOCURE_COMPUTE=cpu python -m pytest -q
```

```powershell
# Windows
$env:EVOCURE_DISABLE_LAYA_NEURAL = "1"; $env:EVOCURE_COMPUTE = "cpu"
.\.venv\Scripts\python.exe -m pytest -q
```

Forcing the CPU backend makes the run deterministic. Dropping `EVOCURE_COMPUTE` exercises
your GPU and enables the GPU to CPU parity test, which checks that your accelerator
reproduces the reference fields at a tolerance matched to its precision.

---

## Repository layout

```
ai/              QSAR predictor, decision engine, optional LLM assistant, model registry
app/             FastAPI application, routes, schemas, session management
simulation/      Core engine, chemistry, physics, evolution, disease models, pharmacology
  core/
    acceleration.py   vendor neutral GPU backend
    hardware.py       vendor neutral GPU inventory
  discovery/
    pipeline.py       the fifteen stage screening pipeline
  biology/diseases/   PDAC, NSCLC, AMR, MASLD tissue models
scripts/         Setup, model management, QSAR training, experiments, report generation
docs/            Architecture, installation, GPU support, limitations, API reference
data/reference/  Curated 90 compound reference dataset, CC-BY 4.0
tests/           Five validation levels plus integration suites
godot_frontend/  Godot 4 desktop client
frontend/        Browser client
```

---

## Scientific scope and honesty

This platform is built so that its limits are visible in its output rather than buried in
a disclaimer. Specifically:

- Predicted potency carries a conformal interval, and on the current training set that
  interval is wide enough that pIC50 is usable as an ordering signal and nothing more.
  The report says so on the page where the numbers appear.
- Validation metrics come from repeated scaffold-grouped cross validation, not a single
  split. On 43 compounds a single split leaves about ten test points and returns
  anything from R2 -0.09 to 0.82 depending only on the seed. Earlier releases quoted one
  such draw; `scripts/investigate_potency_estimator.py` documents the correction.
- Every tissue run is compared against an untreated arm with the same seed, replicated
  across several seeds, and tested against a negative control. A point estimate without
  that comparison is not reported as an effect.
- Positive and negative controls are run first. If they do not separate, candidate
  numbers from that run are not interpretable, and the report states this.
- Activity predictions are made only where a validated predictor exists. Elsewhere the
  value is "unavailable", never a plausible looking substitute.
- Reference drugs and generated structures do not get their potency from the same place,
  and the example report says so on the page where the comparison appears. A curated drug
  uses measured bioactivity; a new structure uses a heuristic whose most potent possible
  output is about nineteen times weaker than gemcitabine's measured value. Scoring
  gemcitabine itself through that heuristic drops it from 67% growth inhibition to -1%.
  Published runs therefore carry a third control arm, the positive control re-scored the
  way a candidate is scored, because that is the only fair comparator.
  `scripts/investigate_potency_asymmetry.py` reproduces this.
- Computed descriptors always describe the molecule actually present. The coevolution
  engine mutates a real structure with RDKit and measures the result; it never shifts a
  parent's properties by a fixed delta and labels the unchanged parent a derivative.
  `tests/level3_evolution/test_selection_loop.py` pins this.
- Structural similarity to a known drug does not transfer efficacy, and absence from a
  small local library does not establish novelty. Both are stated wherever similarity is
  reported.

Virtual generations and simulated hours index the simulation and do not correspond to
chemical, biological or clinical timescales.

---

## Open results

The platform is open source and the results are too. The reference dataset, the full
results JSON and the generated report are all in the repository, so any claim here can be
reproduced, checked or contradicted without asking anyone's permission.

If you run this and get something different, that is worth knowing. Open an issue with
your backend, your library versions and your results file.

---

## Scientific collaboration needed

This platform is engineered but not validated. The code runs, the tests pass and the
reporting is honest about its own limits, which is precisely why the remaining gaps are
scientific rather than technical. Three concrete challenges, each with the measurement
that defines it and what finishing it would look like.

If you work on any of these, the entry point is an issue describing what you measured.
Negative results are as welcome as positive ones and will be published the same way.

### Challenge 1. Validate and calibrate the QSAR and the biological simulations

**Where it stands.** The PDAC potency model trains on 43 compounds spanning 32
independent Bemis-Murcko scaffolds. Under repeated scaffold-grouped cross validation it
reaches RMSE 1.34 log units, R2 0.70 and ROC AUC 0.95, with a 95% conformal interval of
3.12 log units. That interval is a factor of roughly 1300 on IC50, so the model orders
compounds and does not measure them. The tissue model's kinetics, diffusion constants and
clonal parameters have never been fitted to experimental pancreatic tumour data at all.

**Why it is the binding constraint.** A learning curve over held-out scaffolds shows RMSE
falling from 2.04 to 1.04 as training scaffolds rise from 8 to 27, and still descending
steeply at the current size. The estimator is data limited, not architecture limited.
Reproduce it with `python scripts/investigate_potency_estimator.py`.

**This has been tried, and ChEMBL does not supply it.** The obvious plan was to pull
PANC-1 and MIA PaCa-2 IC50 values from ChEMBL. The tooling for that now runs end to end,
and the result is negative. Recorded here so nobody spends the effort twice.

ChEMBL_37 yields 2070 unique PANC-1 IC50 compounds over 900 Murcko scaffolds, which looks
like the 200 scaffolds this section used to ask for, nearly five times over. Retraining on
it reports RMSE 1.34 to 0.85 and the conformal quantile 3.12 to 1.78 log units, a factor
of 1330 down to a factor of 60.

None of that is an improvement. Those numbers are computed on different populations: the
curated set was built to span potency (pIC50 variance 5.96), the ChEMBL set concentrates
(variance 1.16), and since R2 = 1 - MSE/variance the smaller denominator alone reproduces
the reported R2 of both models. Scored properly, both trained with the held-out scaffolds
removed and both tested on the same curated compounds, the 2106 compound model was **worse
on four of five holdout draws**, RMSE +0.23 log units on average.

The reason is that "IC50 | PANC-1" is not one assay. It is **355 distinct assays from 325
distinct papers, median three records each, largest 70**, and only four with 40 or more.
The importer groups on endpoint, cell line and organism, so it cannot see that it is
pooling 355 protocols; the 26 duplicate disagreements it did report, up to 203 fold on the
same compound, are that heterogeneity measured directly. More compounds from more papers
is not more data for this purpose. Reproduce with
`python scripts/measure_dataset_expansion.py`.

**What would help, given that.**
- **GDSC**, or any single large panel run under one protocol. One protocol across a few
  hundred compounds beats thousands across hundreds of protocols, and this is now the
  first recommendation rather than one of four.
- **Hierarchical or multi-task modelling** that treats each assay as its own task over a
  shared representation instead of pooling them into one target. That is the only route
  that makes the 952 available scaffolds usable, and it is a modelling change rather than
  a data one.
- Calibration of the tissue model against published spheroid or organoid dose response
  data, so that simulated growth inhibition means something outside this software.
- Propagation of QSAR uncertainty through the tissue simulation. At present a point IC50
  enters the simulation and the interval is dropped, which flatters the output.

**Done looks like.** A fitted model with its validation protocol, a dataset with
provenance per record, and a statement of what the conformal interval became **measured on
a test set the previous model was also scored on**. The last clause is the part this
project got wrong first time.

[docs/DATA_SOURCES.md](docs/DATA_SOURCES.md) lists the sources and the exact commands.
`scripts/fetch_chembl_bioactivity.py` obtains the data and reports how many protocols a
fetch pools before you import it; `scripts/import_bioactivity.py` gates assay
comparability on endpoint and cell line, which this exercise showed is necessary and not
sufficient.

### Challenge 2. Improve molecular generation and evolutionary optimisation

**Where it stands.** The generator applies RDKit point mutations over nine functional
groups, bioisosteric substitutions and BRICS fragment crossover, selected by NSGA-II over
six objectives with the tumour simulation in the loop, inside a bounded drug-like
envelope. The published run evaluates 467 unique valid structures over 115 Bemis-Murcko
scaffolds, across 20 generations at population 24, in 23 minutes on four CPU cores.

**Why that is small.** Established molecular optimisation studies operate at 10^4 to 10^6
evaluations. At a few hundred the search barely leaves the seed scaffolds, which is
visible in the results: all five reported candidates are gemcitabine analogues at Tanimoto
0.62 to 0.78 to that seed. The platform has also never
been benchmarked against the standard baselines, so its generator's quality relative to
GB-GA, REINVENT or a SELFIES method is simply unknown.

**What would help.**
- A GuacaMol or MOSES benchmark run, so the generator can be compared to published
  baselines rather than described in isolation.
- Richer transformations: ring opening and closing, scaffold hopping, a larger
  bioisostere table, matched molecular pair transformations.
- Scaling the search. The tissue objective costs roughly one second per candidate per
  generation and currently dominates; surrogate-assisted or multi-fidelity selection
  would let the population grow by orders of magnitude.
- Diversity preservation. NSGA-II crowding distance operates on objective space, not
  chemical space, so the population converges structurally. Blending Tanimoto distance
  into that tie-break has been tried and **does not fix it**, which narrows the problem
  rather than solving it. Over three seeds at six generations and population 16 it moved
  mean pairwise Tanimoto distance by -0.012 and distinct Murcko scaffolds by -0.3, both
  inside seed noise, with the front overflowing every generation (20 to 28 members
  against a population of 16) so the tie-break was certainly firing. Downweighting the
  applicability-domain objective, which is maximised similarity to the training set, did
  not widen the chemistry either. So the convergence is set somewhere other than
  survivor selection, most likely in how offspring are generated. Reproduce with
  `python scripts/measure_chemical_diversity.py`; the mechanism is available but off by
  default as `chemical_diversity_weight`.

**Done looks like.** A benchmark number against a published baseline, or a demonstrated
increase in explored chemical diversity at equal compute.

### Challenge 3. Chemical plausibility, synthesizability and experimental validation

**Where it stands.** Generated structures are RDKit-valid, pass valence checks, and are
now bounded to a drug-like envelope: 600 Da, 45 heavy atoms, 12 rotatable bonds, 6 rings,
a single connected component. PAINS, Brenk, NIH and synthetic accessibility are
computed and reported, and can enter selection with `medchem_filters`. There is still no
retrosynthetic analysis and no stability or reactivity screening.

The envelope is a correctness fix, not a plausibility assessment. Before it, nothing in
the search bounded molecular size at all: four successive crossovers took erlotinib from
349 to 1131 Da, growing every step, and ten generations under size-favouring selection
reached a mean of 76 kDa. The only size pressure was QED, which a caller can zero out
through `objective_weights`. That is the mechanism behind the fused dimers this section
used to describe. Reproduce either direction from
`tests/level3_evolution/test_chemical_envelope.py`.

What that buys is narrow, and the published report shows how narrow. Regenerating it with
the envelope in place changed nothing a reader would notice: candidates at 356 to 403 Da
against 356 to 385 before, no structural alerts either way, SA 4.10 to 4.32 against 4.10
to 4.36, and the same five gemcitabine analogues. This run never needed the bound, because
it already weights QED at 2.0 and runs the medicinal chemistry filters. The envelope earns
its place against the operators themselves and against a run that downweights QED, not
against this one, and it is not offered as an improvement to these candidates.

A molecule inside the envelope with no structural alert and a routine SA score is still
only "not obviously impossible", and the rule sets flag approved drugs, so passing them
means little on its own.

**Why it matters.** A ranked list whose members cannot be made, or would decompose, or
would trigger every assay they touch, is not a shortlist. Chemical judgement is the
filter this platform most obviously lacks, and it is the one that cannot be automated
away.

**What would help.**
- Synthetic accessibility scoring (SAscore, SCScore or RAscore) as a search objective,
  not merely a report column.
- Retrosynthetic feasibility on the top candidates, through AiZynthFinder or ASKCOS.
- PAINS and reactive substructure filters applied during generation rather than after.
- A medicinal chemist's written review of a published candidate list, including the ones
  that are obviously wrong and why. This is genuinely valuable and needs no code.
- A realistic proposal for what first experimental validation would look like: which
  assay, which cell lines, what a negative result would rule out, and what it would cost.

**Done looks like.** A filter that removes implausible structures before they are ranked,
or a written assessment of a published list from someone who makes molecules for a
living.

### Also useful

- Verification on AMD, Intel and Apple hardware. The vendor-neutral backend was written
  against each vendor's documented PyTorch API and exercised only on NVIDIA. Paste the
  output of `python -m simulation.core.acceleration` into an issue.
- Additional disease models following the `DiseaseModule` contract.

### Where this is going

[docs/ROADMAP.md](docs/ROADMAP.md) sets out the work in priority order, with the
measurements behind each item. The short version: data first, because the potency
estimator is data limited and everything else is downstream of it.

### One ground rule

Keep the epistemic labelling intact. A pull request that reports a number without its
uncertainty, or presents a simulation result as an experimental finding, will be asked to
change that before anything else is discussed. The same standard applies to results that
favour the project: two defects found during development, fabricated descriptors in the
coevolution engine and a selection loop that excluded the tumour model, both survived
because the output looked reasonable.

---

## Licence

MIT for the code. CC-BY 4.0 for the curated reference dataset in `data/reference/`.
Optional model weights, which are not distributed here, carry their publishers' own terms.

---

## A closing note on the subject matter

Pancreatic cancer has one of the worst survival rates in oncology. That is a reason to
work on it and a reason to describe computational output about it precisely. This software
generates hypotheses at the earliest stage of a long pipeline, where the great majority of
hypotheses fail. It is not a cure, it does not find cures, and no output of it should be
presented to anyone as medical information.
