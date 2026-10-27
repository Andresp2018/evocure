# Installation

Two supported platforms: Windows 10/11 and Linux. macOS works for the Python engine and
the web frontend; the Godot desktop client is not built for it.

Requirements:

- Python 3.11, 3.12 or 3.13
- About 2 GB of disk for the environment, plus whatever optional model weights you add
- A GPU is optional. See [GPU_SUPPORT.md](GPU_SUPPORT.md)

---

## Windows

### 1. Install Python

Get Python 3.12 from [python.org](https://www.python.org/downloads/windows/) or:

```powershell
winget install Python.Python.3.12
```

Tick **Add python.exe to PATH** in the installer. Verify:

```powershell
python --version
```

### 2. Get the code

```powershell
git clone https://github.com/Andresp2018/evocure.git
cd evocure
```

No `git`? Install with `winget install Git.Git`, or download the ZIP from GitHub and
extract it.

### 3. Create the environment

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
```

This creates `.venv`, installs dependencies and copies `.env.example` to `.env`. It uses
[uv](https://github.com/astral-sh/uv) when present, which is much faster, and falls back
to `pip`.

If PowerShell blocks the script, the `-ExecutionPolicy Bypass` above handles it for that
one invocation without changing your system policy.

### 4. Pick your GPU build (optional)

The setup script installs the default PyTorch wheel. Replace it to match your hardware:

```powershell
# NVIDIA
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu124

# AMD or Intel on Windows
.\.venv\Scripts\python.exe -m pip install torch-directml

# CPU only
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Confirm what EVOCURE will use:

```powershell
.\.venv\Scripts\python.exe -m simulation.core.acceleration
```

### 5. Build the QSAR model

```powershell
.\.venv\Scripts\python.exe scripts\train_qsar.py
```

A few seconds. No weights are downloaded; it trains from the dataset in the repository.

### 6. Run

```powershell
.\start_evocure.bat
```

Or directly:

```powershell
.\.venv\Scripts\python.exe run.py
```

Open http://127.0.0.1:8000 if a browser does not open by itself. The Godot desktop
client needs to be exported first (step 8); until then use `--frontend web`.

### 7. Verify the install

```powershell
$env:EVOCURE_DISABLE_LAYA_NEURAL = "1"
$env:EVOCURE_COMPUTE = "cpu"
.\.venv\Scripts\python.exe -m pytest -q
```

Forcing CPU makes the run deterministic. Drop `EVOCURE_COMPUTE` to exercise your GPU,
which also enables the GPU/CPU parity test.

### 8. Desktop client (optional)

Install Godot 4.7 and its export templates, then:

```powershell
.\.venv\Scripts\python.exe scripts\export_desktop.py --platform windows
```

Binaries are not committed to this repository.

---

## Linux

### 1. Install Python and build tools

Debian or Ubuntu:

```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3-pip git build-essential
```

Fedora:

```bash
sudo dnf install -y python3.12 python3-pip git gcc gcc-c++ make
```

Arch:

```bash
sudo pacman -S --needed python python-pip git base-devel
```

### 2. Get the code

```bash
git clone https://github.com/Andresp2018/evocure.git
cd evocure
```

### 3. Create the environment

```bash
bash scripts/setup_env.sh
```

### 4. Pick your GPU build (optional)

```bash
# NVIDIA
./.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cu124

# AMD, ROCm 6.x
./.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/rocm6.2

# Intel Arc
./.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/xpu

# CPU only
./.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

AMD also needs group membership:

```bash
sudo usermod -aG render,video $USER
```

Log out and back in. Then confirm:

```bash
./.venv/bin/python -m simulation.core.acceleration
```

### 5. Build the QSAR model

```bash
./.venv/bin/python scripts/train_qsar.py
```

### 6. Run

```bash
./start_evocure.sh
```

Headless server, no GUI:

```bash
./.venv/bin/python run.py --frontend none --no-browser
```

### 7. Verify the install

```bash
EVOCURE_DISABLE_LAYA_NEURAL=1 EVOCURE_COMPUTE=cpu ./.venv/bin/python -m pytest -q
```

### 8. Desktop client (optional)

```bash
./.venv/bin/python scripts/export_desktop.py --platform linux
chmod +x build/linux/EVOCURE.x86_64
```

The exported client needs its adjacent `.pck` file.

---

## Optional AI models

Neither is required. Both have deterministic fallbacks, and the published example report
was produced with the fallbacks active.

```bash
python scripts/fetch_models.py --status
```

See [models/README.md](../models/README.md).

---

## Configuration

`setup_env` copies `.env.example` to `.env`. Useful keys:

| Variable | Default | Meaning |
|---|---|---|
| `EVOCURE_HOST` | `127.0.0.1` | Bind address. Leave on loopback unless you mean otherwise |
| `EVOCURE_PORT` | `8000` | Port. The launcher scans upward if it is taken |
| `EVOCURE_COMPUTE` | `auto` | `auto`, `cpu`, `cuda`, `rocm`, `directml`, `xpu`, `mps` |
| `EVOCURE_FRONTEND` | `godot` | `godot`, `web`, `both`, `none` |
| `EVOCURE_DISABLE_LAYA_NEURAL` | unset | `1` forces deterministic triage |
| `EVOCURE_MAX_ENTITIES` | `4000` | Particle ceiling |

`.env` is gitignored. Keep API keys out of version control; the optional cloud providers
are off by default and EVOCURE is designed to run fully offline.

---

## Reproduce the example report

```bash
EVOCURE_DISABLE_LAYA_NEURAL=1 EVOCURE_COMPUTE=cpu python scripts/run_pdac_report_experiment.py
python scripts/build_candidate_pdf.py
```

Expect roughly 10 to 30 minutes depending on your CPU. Output lands in
`reports/pdac_publication_test/`. Forcing CPU is what makes the run reproducible;
the experiment records the backend it used either way.

---

## Troubleshooting

**`No working Python 3.11 or 3.12 interpreter found`.** The setup script looks for
`python3.12`, `python3.11`, then `python3`. Point it at a specific interpreter by
creating the venv yourself: `/path/to/python3.12 -m venv .venv`, then
`./.venv/bin/python -m pip install -r requirements.txt`.

**RDKit fails to install.** RDKit ships wheels for CPython 3.11 to 3.13 on the major
platforms. On an unusual platform use conda: `conda install -c conda-forge rdkit`.

**Port 8000 in use.** The launcher scans the next 20 ports and prints the one it picked.
Override with `--port`.

**Godot client does not start.** The launcher falls back to the browser frontend.
Force it with `--frontend web`. Set `EVOCURE_GODOT_BIN` to point at a specific binary.

**Tests fail only with the GPU enabled.** Run `python -m simulation.core.acceleration`
and check the reported precision. A float32 backend (DirectML, MPS) will not match the
CPU reference to 1e-12; the parity test accounts for this automatically, so a failure
here is worth reporting as a bug.
