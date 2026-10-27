# Installation & Hardware Requirements Guide - EVOCURE

All EVOCURE runtime dependencies, scientific calculators, and local AI model weights operate strictly offline inside the isolated project folder `EVOCURE/`.

---

## 1. Hardware Requirements (Section 2.12 & 2.13)

| Tier | CPU | System RAM | GPU / VRAM | Disk Space | Supported Capabilities |
|---|---|---|---|---|---|
| **Minimum (CPU-Only)** | 4-core x86_64 / ARM64 | 8 GB RAM | None required (CPU fallback) | 2 GB (core) / 6 GB (with Q4 LLM) | Full 2D Chemical Universe, PDAC Battle, RDKit, NSGA-II, PBPK, Calibrated QSAR, Deterministic Grounded Assistant, Quantized CPU Gemma |
| **Recommended (GPU-Accelerated)** | 6+ cores (e.g., Ryzen 5 3600X / Core i5+) | 16-32 GB RAM | NVIDIA RTX GPU with 8-12+ GB VRAM | 10+ GB free | Real-time 60 FPS rendering, fast batch evolution (1,000+ generations), GPU-accelerated local **Gemma 4** (`llama.cpp` CUDA) + **Laya AI** |

### Non-Encapsulable Host Components (Mandatory Disclosure - Section 2.14)
While 100% of Python libraries reside inside `EVOCURE/.venv` and model weights reside inside `EVOCURE/models/`, the following operating-system components cannot physically exist inside a Python virtual environment:
1. Host Operating System kernel (Windows 10/11, Linux, or macOS).
2. Native GPU display/CUDA kernel drivers (`nvml` / NVIDIA Driver).
3. Local HTML5/Canvas-capable web browser (Microsoft Edge, Chrome, Firefox, or Safari).
4. Initial host Python 3.11 or 3.12 interpreter used once to bootstrap `.venv`.

---

## 2. Automated `.venv` Setup

### Windows (PowerShell or Command Prompt)
```powershell
cd EVOCURE
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
```
Or simply double-click **`start_evocure.bat`** - it automatically creates `.venv` if missing, installs all pinned dependencies locally, starts the local server on `127.0.0.1:8000`, and opens your browser.

### Linux / macOS
```bash
cd EVOCURE
bash scripts/setup_env.sh
```

---

## 3. Starting EVOCURE Offline

Once `.venv` is initialized, no internet connection is required:

```powershell
# Windows (.venv Python)
.\.venv\Scripts\python.exe run.py

# Linux / macOS (.venv Python)
./.venv/bin/python run.py
```

Optional CLI flags:
- `--host 127.0.0.1` - bind address (strictly local by default).
- `--port 8000` - local HTTP/WebSocket port.
- `--frontend godot|web|both|none` - launch native **Godot 4.7** desktop client (`godot`, default when detected), local browser (`web`), `both`, or headless backend (`none`).
- `--godot-bin <path>` - optional explicit path to Godot 4.x executable (or set `GODOT_BIN` in `.env`).
- `--no-browser` - start headless backend server without launching Godot or a browser window.

---

## 4. Managing Local AI Models (`models/`)

Local weights reside inside `EVOCURE/models/`:
- `models/gemma/Gemma-4-E2B-Heretic-Q4_K_M.gguf` + `models/gemma/gemma4.jinja` + `models/gemma/llama_runtime/llama-server.exe`
- `models/laya/multilingual/model.safetensors` (+ tokenizer & encoder configs)
- `models/scientific/pdac_qsar_v1.joblib`

To inspect hardware compatibility, licenses, exact file sizes, and free disk space (without any unauthorized automatic download):
```powershell
.\.venv\Scripts\python.exe scripts\manage_models.py --list
.\.venv\Scripts\python.exe scripts\manage_models.py --hardware-check
```
