# Where to get the data this platform is missing

The two gaps that limit everything else:

1. **A larger, homogeneous PDAC cytotoxicity dataset**, for the QSAR model.
2. **Dose response data in 3D models**, spheroids or organoids, to calibrate the
   tissue model.

Neither can be filled from inside this repository. There is no network access in
the pipeline by design, and fabricating bioactivity values would be worse than
having none. This page says where the real data lives so that someone with access
can close the gap, and `scripts/import_bioactivity.py` turns a download into a
training set without silently mixing incomparable assays.

Why it matters, in numbers measured here: the potency model trains on 43 compounds
over 32 independent scaffolds, its 95% conformal interval is a factor of about
1330 on IC50, and its learning curve is still descending steeply at the current
size. See [ROADMAP.md](ROADMAP.md).

---

## 1. Activity data for the QSAR model

| Source | What it has | Where | Notes |
|---|---|---|---|
| ChEMBL | IC50 and GI50 in PDAC lines (PANC-1, MIA PaCa-2, BxPC-3 and others) | https://www.ebi.ac.uk/chembl/ | Filter by target or by cell line. The best open source. |
| GDSC, Genomics of Drug Sensitivity in Cancer | Dose response across many lines, PDAC included | https://www.cancerrxgene.org/ | Good for assay consistency. |
| PubChem BioAssay | Cytotoxicity assays | https://pubchem.ncbi.nlm.nih.gov/ | Useful as a complement. |
| CCLE / DepMap | Some drug sensitivity data | https://depmap.org/ | More focused on CRISPR and expression, but has drug data. |

**Practical recommendation, revised after measuring it.** Start with **GDSC**, or any
single published panel run under one protocol. Not ChEMBL.

ChEMBL has the volume: 2070 unique PANC-1 IC50 compounds over 900 Murcko
scaffolds, against the curated set's 32. It does not have the homogeneity. On
ChEMBL_37 those records come from **355 distinct assays in 325 distinct papers,
median three records each, largest 70**, so importing them pools 355 protocols
into one training target. Measured consequence: scored on an identical scaffold
holdout, a model trained on the pooled 2106 records was worse than the 43
compound curated model on four of five draws. The trainer's own metrics looked
better only because they were computed on a lower variance population. See
[ROADMAP.md](ROADMAP.md) for the full numbers and
`python scripts/measure_dataset_expansion.py` to reproduce them.

So the goal is not hundreds of compounds with diverse scaffolds. It is hundreds
of compounds **measured the same way**, and the two are not the same target.
ChEMBL remains useful as a source for a single large panel imported on its own,
and `scripts/fetch_chembl_bioactivity.py` reports the protocol count so you can
see what you are about to pool.

That last point is not a stylistic preference. The published 20 generation run
produced five candidates that are all gemcitabine analogues at Tanimoto 0.62 to
0.78, because the training set is narrow and the applicability domain check
rewards staying close to it. Scaffold diversity in the training data is what
would let the search leave that neighbourhood with any confidence.

### Getting a ChEMBL download

The instruction above used to stop at "obtain a PANC-1 IC50 export", which is
where the work stopped: ChEMBL's web interface does not make a cell line
activity export straightforward. `scripts/fetch_chembl_bioactivity.py` does that
one step against ChEMBL's public REST API.

```bash
# What is available for a cell line, without downloading anything
python scripts/fetch_chembl_bioactivity.py --cell-line PANC-1 --survey

# Fetch it, into the columns the importer reads
python scripts/fetch_chembl_bioactivity.py --cell-line PANC-1 --endpoint IC50
python scripts/fetch_chembl_bioactivity.py --cell-line "MIA PaCa-2" --endpoint IC50
```

It writes a CSV to `data/raw/`, which is gitignored, and a `.provenance.json`
beside it recording the exact query, the ChEMBL release, the counts and what was
dropped, so a dataset built this way can be traced back to the request that
produced it.

It drops and counts, rather than keeping: records ChEMBL itself flags with a
`data_validity_comment`, censored values such as `>100 uM`, rows with no
structure, and units the importer cannot convert exactly. It does not clean,
average, impute or interpret, and everything it writes still has to pass the
assay comparability gate below.

This is the only part of EVOCURE that touches the network, and only when you run
it. The pipeline stays offline.

The cell line name must match ChEMBL exactly. `PANC-1` and
`Gemcitabine-resistant PANC-1` are separate CELL-LINE targets and separate
biology; a loose match that pooled them would be the exact failure the importer
exists to prevent. The script refuses an inexact match and lists the near
misses.

### Importing a download

```bash
# See what assay groups the file actually contains before importing anything
python scripts/import_bioactivity.py data/raw/chembl_panc1.csv --inspect

# Import one group, merged with the curated records
python scripts/import_bioactivity.py data/raw/chembl_panc1.csv \
    --endpoint IC50 --cell-line PANC-1 --disease PDAC \
    --merge-into-curated \
    --output data/processed/pdac_expanded_dataset.json

# Retrain and watch the conformal interval, which is the number that matters.
# --dataset is required: without it the script loads the curated file and the
# import you just made is written and never read.
python scripts/train_qsar.py --dataset data/processed/pdac_expanded_dataset.json
```

The output goes to `data/processed/`, which is gitignored. That is deliberate:
`data/reference/` is published under CC-BY 4.0, and a dataset derived from ChEMBL
carries CC BY-SA 3.0, so writing one into the other would misstate the licence of
the result.

The importer rejects, and counts, anything outside the assay group you name:
different cell lines, different endpoints, censored values such as `>100 uM`,
unconvertible units, and unparseable structures. It keeps curated values on
collision, because a hand-checked record with a citation outranks a bulk import.
Duplicates that disagree by more than tenfold are reported rather than averaged
into silence.

**Count scaffolds, not compounds.** A scaffold split sees independent scaffolds
as its sample size. Five hundred analogues of one series is not five hundred data
points.

**Check the licence before redistributing.** ChEMBL is CC BY-SA, GDSC and DepMap
have their own terms. The importer records the source file but cannot know what
you are allowed to publish.

---

## 2. Data to calibrate the tissue model

| Kind of data | Useful sources | Where to look |
|---|---|---|
| Dose response in PDAC spheroids and organoids | PDAC organoid papers (Tuveson and others) | PubMed, including supplementary figures |
| Hypoxia, diffusion, desmoplasia | Tumour microenvironment studies in PDAC | Literature, plus some imaging datasets |
| Kinetic parameters (uptake, clearance) | Pharmacology papers on gemcitabine and other nucleosides | PubMed |

This data is scattered, often inside figures, and usually needs manual extraction
or contact with the authors. There is no single clean repository like ChEMBL.

That difficulty is the honest reason the tissue model is uncalibrated today, and
it is why the roadmap offers "calibrate **or** remove" as the second priority. A
model that cannot be fitted to anything is better described as illustrative than
carried as though it were predictive.

What calibration would need, concretely:

- Dose response curves in a 3D PDAC model for at least a handful of compounds
  whose potency in 2D is also known, so the 2D to 3D shift can be estimated
  rather than assumed.
- Oxygen and nutrient gradient measurements, or published diffusion coefficients
  for the relevant species in dense stroma.
- A resistant subpopulation frequency and growth rate from a real model system.

With those, `simulation/biology/diseases/pdac.py` has the parameters exposed to
fit. Without them, its output restates whatever potency it was handed; this is
measured in `reports/tissue_transfer_function.json`.

---

## 3. Other useful data, less critical

- **PAINS and toxicity alerts.** Already implemented in RDKit's filter catalogue
  and available in public databases. No download needed.
- **Synthetic accessibility.** SA Score ships with RDKit contrib; AiZynthFinder
  covers full retrosynthesis.
- **Prior art and similarity.** PubChem, ChEMBL and SureChEMBL, by API or bulk
  download. The current novelty check searches 122 local structures, which is a
  sanity check and not a prior art search.

---

## Suggested order

1. **ChEMBL and GDSC**, to expand the QSAR. Largest immediate effect.
2. **Extract spheroid and organoid dose response from the literature**, to
   calibrate the tissue model.
3. **Only then** consider more advanced data such as PBPK parameters.

---

## If you contribute data

Open an issue with the source, the filter you applied, the assay group, and the
output of `scripts/import_bioactivity.py --inspect`. The useful contribution is
the provenance as much as the numbers: a dataset nobody can trace is a dataset
nobody can check.

Negative results count. If an expanded set does not narrow the conformal
interval, that is worth publishing here too, because it would mean the limit is
the representation rather than the sample size, and it would change the roadmap.
