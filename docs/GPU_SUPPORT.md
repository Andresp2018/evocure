# GPU support

EVOCURE runs on NVIDIA, AMD, Intel and Apple GPUs, and on the CPU alone. There is one
code path; the backend is chosen at runtime. No part of the platform requires a GPU.

Check what your machine will use:

```bash
python -m simulation.core.acceleration
```

Example output on a machine with an NVIDIA card:

```
EVOCURE compute backend
============================================================
  Requested (EVOCURE_COMPUTE) : auto
  Active backend              : NVIDIA CUDA / PyTorch float64
  Device                      : NVIDIA GeForce RTX 3060
  Field precision             : float64
  PyTorch installed           : yes
  Usable accelerators         : 1
    - cuda      NVIDIA     NVIDIA GeForce RTX 3060 (float64)
```

## What actually runs on the GPU

Be aware of the scope before you buy hardware for this. GPU acceleration covers the
oxygen and nutrient reaction-diffusion fields in the tissue models, and local AI model
inference when you install the optional weights. These stay on the CPU:

| Work | Device | Why |
|---|---|---|
| Oxygen / nutrient reaction-diffusion | GPU | Dense grid stencil, batches well |
| Laya and Gemma inference (optional) | GPU | Standard tensor and GGUF offload |
| Godot rendering | GPU | Vulkan |
| RDKit chemistry | CPU | Graph algorithms on a C++ library |
| Per-particle and per-cell agent decisions | CPU | Branch heavy, small per-item work |
| scikit-learn QSAR | CPU | Random forest inference |
| PBPK ODE integration | CPU | Small stiff system, SciPy |
| SQLite persistence | CPU | Disk bound |

For the default grid sizes the CPU path is already fast. A GPU helps on large grids and
long runs. Maximum GPU utilization is not a goal.

## Choosing a backend

Set `EVOCURE_COMPUTE`:

| Value | Meaning |
|---|---|
| `auto` | Default. Tries CUDA/ROCm, then XPU, then MPS, then DirectML, then CPU |
| `cpu` | Force the NumPy reference path |
| `cuda` | NVIDIA through PyTorch CUDA |
| `rocm` | AMD through PyTorch ROCm (Linux) |
| `directml` | AMD, Intel or NVIDIA through torch-directml (Windows) |
| `xpu` | Intel Arc or Data Center GPU through PyTorch XPU |
| `mps` | Apple Silicon through Metal |

If you name a backend that is not available, EVOCURE falls back to the CPU rather than
failing. The doctor command tells you which ones it can actually reach.

## Install per vendor

PyTorch is the only piece that differs. Install the rest of EVOCURE normally, then
replace PyTorch with the build for your hardware.

### NVIDIA, Windows or Linux

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Needs a recent NVIDIA driver. CUDA Toolkit is not required separately; the wheel bundles
what it needs. Runs float64.

### AMD on Linux (ROCm)

```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
```

Needs ROCm 6.x installed and your user in the `render` and `video` groups:

```bash
sudo usermod -aG render,video $USER
```

Log out and back in for that to take effect. A ROCm PyTorch build reports AMD cards
through the `torch.cuda` namespace, which is expected; EVOCURE tells the two apart with
`torch.version.hip` and reports the backend as `rocm`. Runs float64.

Check the [ROCm compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)
for your card. Consumer RDNA cards that are not on the official list often work by
setting an override, for example:

```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
```

### AMD or Intel on Windows (DirectML)

```bash
pip install torch-directml
```

DirectML works through DirectX 12, so it covers essentially any modern GPU on Windows
including AMD Radeon, Intel Arc and integrated graphics. It is the most portable option
on Windows and needs no vendor SDK.

Precision: DirectML does not implement float64. EVOCURE detects this, runs the fields in
float32 and reports `precision: float32`. Expect differences of roughly 1e-6 relative
against the CPU reference. Use `EVOCURE_COMPUTE=cpu` when you need bit identical output.

### Intel Arc and Data Center GPUs (XPU)

```bash
pip install torch --index-url https://download.pytorch.org/whl/xpu
```

Needs recent Intel GPU drivers. Reported as backend `xpu`.

### Apple Silicon (MPS)

```bash
pip install torch
```

The standard macOS wheel includes Metal support. Like DirectML, MPS has no float64, so
fields run float32 and are reported as such.

### CPU only

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Or skip PyTorch entirely. EVOCURE uses NumPy when PyTorch is absent. This path is fully
supported and is the reproducibility reference.

## Reproducibility

The CPU path is the reference. It is the only backend guaranteed to produce bit identical
results across machines.

- float64 backends (CUDA, ROCm, XPU) agree with the CPU to within 1e-12 on these fields.
  This is verified by `test_gpu_cpu_field_parity`, which runs on whatever accelerator the
  host has and is skipped when there is none.
- float32 backends (DirectML, MPS) agree to roughly 1e-5 absolute. The same test applies
  the looser tolerance automatically after reading the backend's reported precision.

For any run you intend to publish, record the backend. `simulation.core.acceleration.status()`
returns it, and the experiment runner embeds it in the report JSON.

## Local AI model offload

The optional Gemma assistant runs through llama.cpp, which has its own backend selection
separate from PyTorch. EVOCURE detects your GPU vendor and reports the matching llama.cpp
build in `gpu_offload_backend`:

| Vendor | llama.cpp build |
|---|---|
| NVIDIA | CUDA |
| AMD | HIP, or Vulkan as a portable fallback |
| Intel | Vulkan or SYCL |
| Apple | Metal |

The llama.cpp runtime is not bundled with this repository. Download the build matching
your GPU from [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases) and
place it in `models/gemma/llama_runtime/`. Vulkan builds are the simplest choice for AMD
and Intel, since they need no vendor compute SDK.

## Troubleshooting

**"No accelerator found" but you have a GPU.** The usual cause is a CPU-only PyTorch
build. Check:

```python
import torch; print(torch.__version__, torch.version.cuda, torch.version.hip)
```

A version string ending in `+cpu` means you need to reinstall with the right index URL.

**ROCm: "HIP error: invalid device function".** Your card is not in the build's target
list. Try `HSA_OVERRIDE_GFX_VERSION` as above with the nearest supported architecture.

**DirectML is slower than the CPU.** On small grids the transfer overhead dominates. This
is expected; use `EVOCURE_COMPUTE=cpu` for small runs.

**Results differ slightly between two machines.** Check `precision` in the status output.
A float32 backend explains differences around 1e-6. Set `EVOCURE_COMPUTE=cpu` on both.

**Multiple GPUs.** EVOCURE uses device 0. Select a different one with the vendor's
environment variable: `CUDA_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES` or `ONEAPI_DEVICE_SELECTOR`.
