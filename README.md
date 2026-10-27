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
git clone https://github.com/OWNER/REPO.git
cd REPO
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
.\.venv\Scripts\python.exe scripts\train_qsar.py
.\start_evocure.bat
```

### Linux

```bash
git clone https://github.com/OWNER/REPO.git
cd REPO
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

**Candidate discovery.** A fifteen stage pipeline: seed import, structural variant
generation, RDKit validation, descriptor calculation, calibrated QSAR with conformal
intervals, ADMET and ten category toxicology, applicability domain check, five objective
scoring, triage, controlled tissue simulation, NSGA-II Pareto sorting, and population
update across generations.

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
  interval is wide enough that pIC50 is usable as a loose ordering signal and nothing
  more. The report says so on the page where the numbers appear.
- Every tissue run is compared against an untreated arm with the same seed, replicated
  across several seeds, and tested against a negative control. A point estimate without
  that comparison is not reported as an effect.
- Positive and negative controls are run first. If they do not separate, candidate
  numbers from that run are not interpretable, and the report states this.
- Activity predictions are made only where a validated predictor exists. Elsewhere the
  value is "unavailable", never a plausible looking substitute.
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

## Contributing

Useful contributions, roughly in order of value:

1. Experimental calibration of the tissue model parameters against published data.
2. A larger, better curated bioactivity training set, which is the binding constraint on
   the QSAR model today.
3. Verification on AMD, Intel and Apple hardware. The vendor neutral backend was written
   against the documented PyTorch APIs for each and exercised on NVIDIA; the other paths
   are implemented but have not been run on real hardware by the maintainers. Reports of
   `python -m simulation.core.acceleration` output from other vendors are welcome.
4. Additional disease models following the `DiseaseModule` contract.

Keep the epistemic labelling intact. A pull request that reports a number without its
uncertainty, or that presents a simulation result as an experimental finding, will be
asked to change that before anything else.

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
