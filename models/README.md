# Models directory

**No model weights are committed to this repository.** Everything in this directory
other than this file and `manifest.json` is ignored by git.

EVOCURE runs with this directory empty. Every optional artifact has a deterministic
fallback, and the dashboard always names which provider answered.

## What goes here

```
models/
  scientific/pdac_qsar_v1.joblib     rebuild locally, see below
  gemma/                             optional LLM assistant, GGUF + llama.cpp runtime
  laya/multilingual/                 optional neural triage, safetensors
```

## Check what you have

```bash
python scripts/fetch_models.py --status
python scripts/fetch_models.py --verify
```

## The one you should rebuild

The PDAC QSAR model trains in seconds on a CPU from the reference dataset that is
committed in `data/reference/`:

```bash
python scripts/train_qsar.py
```

Without it, candidate search still runs but ranks on physicochemical properties,
toxicity alerts and domain similarity only, and reports activity as unavailable.

Read the validation metrics the training script prints. The pIC50 regressor is weak
(R2 around 0.32 on a scaffold split, 95% conformal interval of about 4.2 log units).
The active/inactive classifier is the stronger half. Treat predicted potency as a
coarse ordering signal, never as a measurement.

## The optional ones

`gemma-assistant` and `laya-decision` are convenience layers, not scientific
components. They affect how search priorities are phrased and how candidates are
triaged, not the chemistry, the QSAR model or the tissue simulation.

The published example report was produced with both fallbacks active, so you do not
need either to reproduce it.

These weights are not redistributed here. Their licenses are the publisher's to grant,
and a 4 GB binary does not belong in git. `manifest.json` records the exact size and
SHA-256 of each file in the reference install so you can verify whatever you obtain.

To install from a location you control:

```bash
EVOCURE_MODEL_MIRROR=https://your-host/evocure python scripts/fetch_models.py --fetch all
```

## llama.cpp runtime

If you use the Gemma assistant, put a llama.cpp build matching your GPU in
`models/gemma/llama_runtime/`. Get it from
[llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases).

| GPU | Build |
|---|---|
| NVIDIA | CUDA |
| AMD | HIP, or Vulkan for portability |
| Intel | Vulkan or SYCL |
| Apple | Metal |
| None | CPU |

Vulkan builds are the simplest choice on AMD and Intel since they need no vendor
compute SDK. See [docs/GPU_SUPPORT.md](../docs/GPU_SUPPORT.md).
