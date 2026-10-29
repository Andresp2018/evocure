# What would make this useful

This document exists because the obvious ambition for a platform like this one,
"generate better drug candidates than anything else does", is both unreachable
and not where the value is. It sets out what is actually blocking EVOCURE, what
niche is genuinely unoccupied, and in what order the work pays off. Every claim
here is measured in this repository or cited.

It replaces aspiration with a plan. [ROADMAP.md](ROADMAP.md) remains the
itemised task list.

---

## 1. Generation is not the bottleneck

Producing valid molecules against arbitrary objectives is a solved problem.
REINVENT, graph genetic algorithms and SELFIES methods operate at 10^4 to 10^6
evaluations. EVOCURE's published run evaluates 467. Closing that gap would make
EVOCURE ordinary rather than useful.

The field's own numbers say where the failure is. The best-supported
retrospective of AI-discovered molecules puts Phase I success at 80 to 90%
against a historical 40 to 65%, and Phase II at roughly 40%, which is the
industry norm. Safety transferred; efficacy did not. On a sample of ten
concluded molecules, so treat the magnitudes loosely and the shape seriously.

A generator proposing a million structures still cannot say which one survives a
tumour. That is the problem worth working on.

---

## 2. What blocks EVOCURE today, in order

### 2.1 The endpoint cannot express the thing that works

The QSAR is trained on PANC-1 bulk cytotoxicity. That endpoint cannot
distinguish a mutation-selective KRAS G12D inhibitor from a general cytotoxin,
because both reduce viability in a well. There is no target structure, no
docking, and no representation of which mutation a compound engages.

Measured consequence. `scripts/check_prior_art.py` run against the published
candidates returns Tanimoto 0.62 to 0.78 against gemcitabine and **0.12 to 0.14
against sotorasib and adagrasib**. All five candidates are nucleoside
antimetabolites, the class gemcitabine has occupied since 1997.

Meanwhile the FDA approved daraxonrasib for previously treated metastatic PDAC
in August 2026 on median overall survival of 13.2 against 6.7 months, and
several KRAS G12D agents are in early trials. The survival gains are coming from
a modality this platform structurally cannot propose, however long the search
runs. This is a design limit, not a budget limit.

### 2.2 Predicted potency has no resolution

The 95% conformal interval is 3.12 log units, a factor of about 1330 on IC50.
Propagated through the tissue model it admits growth inhibition from -5% to
+90% for every published candidate; the report's own headline says each interval
"admits no effect at all".

The obvious fix was tried and failed. Importing 2070 PANC-1 IC50 compounds over
900 scaffolds from ChEMBL made the model **worse on four of five holdout draws**
when both models were scored on an identical test set, because "IC50 | PANC-1"
is 355 distinct assays from 325 papers, median three records each. See
[ROADMAP.md](ROADMAP.md) and `reports/dataset_expansion_comparison.json`. More
data from more papers is not more data.

### 2.3 The tissue model is not calibrated

No kinetic, diffusion or clonal parameter has been fitted to a measurement. So
simulated growth inhibition is a restatement of the IC50 it was handed, which is
visible in the sharpest failure the platform reports about itself: gemcitabine,
scored the way a candidate is scored, is assigned 8.05 uM against its curated
0.045 uM, 179 fold weaker, and returns growth inhibition of **-0.99%** with a
confidence interval spanning zero. The positive control fails its own test.

Candidates then "beat" that control by 65 to 77 percentage points. That is a
statement about the scoring function, not about the molecules, and it is a
warning rather than a result.

---

## 3. The niche that is actually unoccupied

Look at what the PDAC module already represents:

- reaction-diffusion of oxygen, nutrients, ECM and drug on a 2D slice
- CAFs depositing ECM that attenuates drug diffusivity, `D_eff = D_0 / (1 + beta * ECM)`
- normoxic rim, hypoxic mantle and necrotic core zonation
- per-cell efflux pump levels and continuous resistance scores
- two clonal populations, classical and basal, that can shift under treatment
- immune exclusion from fibrotic and hypoxic regions

No ligand-based generative platform can answer a question posed in those terms.
And the question that decides outcomes in pancreatic cancer is not "what kills
PANC-1 cells in a well", which has been answerable since the 1990s. It is:

> **Why does a drug that works in the well fail to reach the tumour, and when
> does resistance emerge?**

Desmoplasia, hypoxia, interstitial pressure and clonal heterogeneity. That is
why this disease has the survival it has.

So the defensible product is not a candidate generator. It is **an open,
calibrated simulator of the tumour microenvironment that chooses the next
experiment**: which schedule, which combination, in which microenvironment, and
when it fails. Decision support under stated uncertainty.

Open generators are plentiful. An open, honestly calibrated microenvironment
model is not.

---

## 4. What that requires, in order of leverage

### Step 1. Calibrate the tissue model against published measurements

The only step that needs neither a laboratory nor new data. It is also harder
than it looks, for the same reason the ChEMBL import failed.

**Measured obstacle.** Published PANC-1 monolayer gemcitabine IC50 values span
58 nM to 31.41 uM, three orders of magnitude, because they differ in assay
duration, readout and passage. Absolute values cannot be pooled across papers,
so they cannot be calibration targets.

**What can.** Ratios measured within one study, where the confounders cancel:

| Observation | Value | Source |
|---|---|---|
| Panc02 spheroid against monolayer | 18.38 against 9.99 uM, 1.84x | one study, paired |
| PANC-1 spheroid-derived against parental | 1.5x gemcitabine, 1.9x doxorubicin, 2.8x oxaliplatin | one study, paired |
| PANC-1 with CAFs at 9:1 against alone | 331 against 58 nM, 5.7x | one study, paired |
| Doxorubicin in 480 um Panc-1 spheroids | full penetration to the core within 12 h | one study |

Two of these the model can express directly. CAF-mediated resistance is a
with-and-without run, and the penetration observation is a falsifiable
constraint on `beta_ECM`: if the model predicts a steep gradient where the
experiment shows a small molecule reaching the core of a 480 um spheroid in 12
hours, the barrier is too strong.

**And the barrier may be the wrong mechanism entirely.** The literature is
contested. One study of KPC mice reports that gemcitabine does not penetrate far
and attributes poor response to stroma and interstitial pressure. Another, in
two orthotopic models, found gemcitabine distributed nearly homogeneously into
both hypoxic and normoxic regions and concluded that penetration did not limit
efficacy. The platform's headline mechanism rests on the first. Calibration has
to be able to report that it is wrong.

`scripts/calibrate_tissue_model.py` is the harness. It holds the observations
with provenance, runs the model against each, reports the discrepancy, and names
the observations the model cannot express at all. It does not fit five free
parameters to four observations, which would be overfitting with extra steps.

**Done looks like.** A statement of which parameters are consistent with the
observations, which are not, and which observations the model has no way to
represent.

### Step 2. Give potency a usable interval

Not more ChEMBL. Two routes, both measurable:

- **GDSC**, or any single large panel run under one protocol. One protocol
  across a few hundred compounds beats thousands across hundreds of protocols.
- **Hierarchical or multi-task modelling** that treats each assay as its own
  task over a shared representation. This is the only route that makes the 952
  scaffolds already imported usable without pretending they are homogeneous, and
  it is a modelling change rather than a data acquisition problem.

### Step 3. Add a target-based arm

The KRAS G12D structure is public and local docking is feasible. Necessary to
touch the modality that is working, and **not sufficient**: docking scores
correlate poorly with activity, so this buys the ability to propose that
chemistry, not the ability to rank it.

### Step 4. Close the loop with one real experiment

Twenty to fifty compounds purchased and tested in a spheroid assay, fed back
into the model. Small, unglamorous, and the only thing here that converts the
platform from a hypothesis generator into an instrument. Nobody does this
openly.

### Step 5. A medicinal chemist's written review

Of a published candidate list, including the structures that are obviously wrong
and why. Needs no code. The README has asked for it from the start.

---

## 5. What to stop doing

- **More generations.** A 100 generation run was executed to test exactly this.
  Whatever it returns, the search budget is not the constraint; sections 2.1 to
  2.3 are.
- **More objectives and more mutation operators.** Polishing the part that
  already works.
- **Reading a lower conformal interval as a better model** when it was computed
  on a different population. That mistake was made during this work and caught
  by scoring both models on one test set.

---

## 6. The honest summary

As a candidate generator for pancreatic cancer, the output is not currently
useful, and comparing it to the recent clinical record makes that clearer rather
than less clear. The candidates are novel, in that PubChem and ChEMBL hold none
of them, and they are novel members of a class with a long record of clinical
disappointment, proposed by a scoring function that cannot recover gemcitabine's
own activity.

What is genuinely unusual is that the platform says so itself. The headline
states that the intervals admit no effect. The modelled-control arm exists to
catch exactly the unfair comparison it caught. Most software in this category
would report "76% growth inhibition, p < 0.001" and stop.

That is an engineering and epistemic achievement rather than a biological one.
It is also the foundation that makes sections 3 and 4 worth attempting, because
a platform that reports its own limits can be calibrated. One that flatters
itself cannot.
