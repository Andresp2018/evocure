"""Vendor neutral compute backend for the reaction-diffusion substrate fields.

EVOCURE runs on NVIDIA, AMD, Intel and Apple GPUs through a single code path, and
falls back to NumPy on the CPU. The CPU path stays the reproducibility reference:
it is the only backend guaranteed to be bit identical across machines.

Backend selection is controlled by the EVOCURE_COMPUTE environment variable:

    auto        pick the best available accelerator (default)
    cpu         force the NumPy reference path
    cuda        NVIDIA via PyTorch CUDA
    rocm        AMD via PyTorch ROCm (Linux)
    directml    AMD, Intel or NVIDIA via torch-directml (Windows)
    xpu         Intel Arc / Data Center GPU via PyTorch XPU
    mps         Apple Silicon via PyTorch Metal

A PyTorch ROCm build exposes AMD hardware through the torch.cuda namespace, so the
cuda and rocm backends share an implementation and are told apart by torch.version.hip.

Precision: CUDA, ROCm and XPU run float64, matching the CPU reference closely. DirectML
and MPS do not implement float64, so those backends run float32 and report
precision="float32". Expect small numerical differences there, of order 1e-6 relative on
these fields. Use EVOCURE_COMPUTE=cpu for runs that must be exactly reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import os
import numpy as np

VALID_PREFERENCES = ("auto", "cpu", "cuda", "rocm", "directml", "xpu", "mps")

VENDOR_BY_BACKEND = {
    "cuda": "NVIDIA",
    "rocm": "AMD",
    "directml": "GPU vendor reported by DirectML",
    "xpu": "Intel",
    "mps": "Apple",
    "cpu": "CPU",
}


@dataclass(frozen=True)
class ComputeDevice:
    """A resolved accelerator plus the dtype it can actually run."""

    backend: str
    vendor: str
    name: str
    device: Any
    dtype: Any
    precision: str
    notes: tuple[str, ...] = field(default=())

    @property
    def is_gpu(self) -> bool:
        return self.backend != "cpu"


def preference() -> str:
    """Read the requested backend, defaulting to auto for unrecognised values."""
    raw = os.environ.get("EVOCURE_COMPUTE", "auto").strip().lower()
    return raw if raw in VALID_PREFERENCES else "auto"


def _torch():
    try:
        import torch
    except ImportError:
        return None
    return torch


def _supports_float64(torch, device) -> bool:
    """Probe the device rather than trusting a vendor table."""
    try:
        probe = torch.ones(2, dtype=torch.float64, device=device)
        return float((probe + probe).sum().item()) == 4.0
    except Exception:
        return False


def _build(torch, backend: str, device, name: str, notes: tuple[str, ...] = ()) -> ComputeDevice:
    float64 = _supports_float64(torch, device)
    if not float64:
        notes = notes + (
            "float64 is unavailable on this backend; fields run in float32. "
            "Use EVOCURE_COMPUTE=cpu for bit reproducible output.",
        )
    return ComputeDevice(
        backend=backend,
        vendor=VENDOR_BY_BACKEND.get(backend, "Unknown"),
        name=name,
        device=device,
        dtype=torch.float64 if float64 else torch.float32,
        precision="float64" if float64 else "float32",
        notes=notes,
    )


def _try_cuda_or_rocm(torch) -> ComputeDevice | None:
    """PyTorch ROCm presents AMD cards through the torch.cuda namespace."""
    try:
        if not torch.cuda.is_available():
            return None
        name = torch.cuda.get_device_name(0)
    except Exception:
        return None

    hip = getattr(torch.version, "hip", None)
    if hip:
        return _build(torch, "rocm", torch.device("cuda:0"), name, (f"ROCm/HIP runtime {hip}",))
    cuda_version = getattr(torch.version, "cuda", None)
    note = (f"CUDA runtime {cuda_version}",) if cuda_version else ()
    return _build(torch, "cuda", torch.device("cuda:0"), name, note)


def _try_xpu(torch) -> ComputeDevice | None:
    try:
        xpu = getattr(torch, "xpu", None)
        if xpu is None or not xpu.is_available():
            return None
        try:
            name = xpu.get_device_name(0)
        except Exception:
            name = "Intel XPU device 0"
    except Exception:
        return None
    return _build(torch, "xpu", torch.device("xpu:0"), name)


def _try_mps(torch) -> ComputeDevice | None:
    try:
        backend = getattr(torch.backends, "mps", None)
        if backend is None or not backend.is_available():
            return None
    except Exception:
        return None
    return _build(torch, "mps", torch.device("mps"), "Apple Silicon GPU (Metal)")


def _try_directml(torch) -> ComputeDevice | None:
    """torch-directml covers AMD, Intel and NVIDIA GPUs on Windows."""
    try:
        import torch_directml
    except ImportError:
        return None
    try:
        if not torch_directml.is_available() or torch_directml.device_count() < 1:
            return None
        device = torch_directml.device(0)
        try:
            name = torch_directml.device_name(0)
        except Exception:
            name = "DirectML device 0"
    except Exception:
        return None
    return _build(torch, "directml", device, name, ("Windows DirectML (DirectX 12) backend",))


_DETECTORS = {
    "cuda": _try_cuda_or_rocm,
    "rocm": _try_cuda_or_rocm,
    "xpu": _try_xpu,
    "mps": _try_mps,
    "directml": _try_directml,
}

# Order matters: native vendor runtimes first, DirectML as the Windows catch-all.
_AUTO_ORDER = ("cuda", "xpu", "mps", "directml")


@lru_cache(maxsize=len(VALID_PREFERENCES))
def _resolve(want: str) -> ComputeDevice | None:
    """Detection is cached per requested backend, not globally, so that changing
    EVOCURE_COMPUTE at runtime (tests do this) takes effect without a stale device."""
    if want == "cpu":
        return None

    torch = _torch()
    if torch is None:
        return None

    if want != "auto":
        detector = _DETECTORS.get(want)
        resolved = detector(torch) if detector else None
        if resolved is None:
            return None
        if want == "rocm" and resolved.backend != "rocm":
            return None
        if want == "cuda" and resolved.backend != "cuda":
            return None
        return resolved

    for backend in _AUTO_ORDER:
        resolved = _DETECTORS[backend](torch)
        if resolved is not None:
            return resolved
    return None


def compute_device() -> ComputeDevice | None:
    """Resolve the active accelerator, or None when running on the CPU."""
    return _resolve(preference())


def available_backends() -> list[dict[str, Any]]:
    """Enumerate every backend this install can reach. Used by the doctor command."""
    torch = _torch()
    if torch is None:
        return []
    found: list[dict[str, Any]] = []
    for backend in _AUTO_ORDER:
        resolved = _DETECTORS[backend](torch)
        if resolved is not None:
            found.append({
                "backend": resolved.backend,
                "vendor": resolved.vendor,
                "name": resolved.name,
                "precision": resolved.precision,
            })
    return found


def gpu_enabled() -> bool:
    return compute_device() is not None


def torch_device_string() -> str:
    """Device string for consumers that load their own models, such as Laya."""
    resolved = compute_device()
    if resolved is None:
        return "cpu"
    if resolved.backend == "directml":
        return str(resolved.device)
    return {"cuda": "cuda", "rocm": "cuda", "xpu": "xpu", "mps": "mps"}[resolved.backend]


def cuda_runtime():
    """Deprecated alias retained for existing callers and tests."""
    return _torch() if gpu_enabled() else None


def status() -> dict[str, Any]:
    resolved = compute_device()
    torch = _torch()
    return {
        "numerical_backend": (
            f"{resolved.vendor} {resolved.backend.upper()} / PyTorch {resolved.precision}"
            if resolved else "NumPy CPU float64"
        ),
        "gpu_available": resolved is not None,
        "gpu_backend": resolved.backend if resolved else None,
        "gpu_vendor": resolved.vendor if resolved else None,
        "gpu_name": resolved.name if resolved else None,
        "precision": resolved.precision if resolved else "float64",
        "backend_notes": list(resolved.notes) if resolved else [],
        "torch_installed": torch is not None,
        "detected_backends": available_backends(),
        "accelerated_work": ["Disease oxygen/nutrient reaction-diffusion"] if resolved else [],
        "cpu_work": [
            "RDKit chemistry",
            "particle and cellular agents",
            "scikit-learn QSAR",
            "SQLite",
            "PBPK ODE",
        ],
        "rendering": "Godot Vulkan GPU",
        "preference": preference(),
        "reproducibility_note": (
            "EVOCURE_COMPUTE=cpu is the reference path and the only backend guaranteed "
            "to reproduce bit for bit across machines."
        ),
    }


def substrates(oxygen, nutrient, vessel, oxygen_sink, nutrient_sink, dx, dy,
               diffusivities, supply_rates, km, dt, steps):
    """Advance both substrate fields on the accelerator.

    Returns None when no accelerator is active, which tells the caller to run its
    NumPy loop. The finite-difference stencil here matches the CPU implementation.
    """
    resolved = compute_device()
    if resolved is None:
        return None

    torch = _torch()
    device, dtype = resolved.device, resolved.dtype

    def tensor(value):
        return torch.as_tensor(np.asarray(value), dtype=dtype, device=device)

    try:
        with torch.inference_mode():
            field = tensor(np.stack((oxygen, nutrient)))
            sinks = tensor(np.stack((oxygen_sink, nutrient_sink)))
            vascular = tensor(vessel)
            diff = tensor(diffusivities)[:, None, None]
            supply = tensor(supply_rates)[:, None, None]
            michaelis = tensor(km)[:, None, None]
            inv_dx2 = 1.0 / (dx * dx)
            inv_dy2 = 1.0 / (dy * dy)

            for _ in range(steps):
                lap = torch.zeros_like(field)
                lap[:, 1:-1, 1:-1] = (
                    (field[:, 1:-1, 2:] + field[:, 1:-1, :-2] - 2 * field[:, 1:-1, 1:-1]) * inv_dx2
                    + (field[:, 2:, 1:-1] + field[:, :-2, 1:-1] - 2 * field[:, 1:-1, 1:-1]) * inv_dy2
                )
                lap[:, 0, 1:-1] = (
                    (field[:, 0, 2:] + field[:, 0, :-2] - 2 * field[:, 0, 1:-1]) * inv_dx2
                    + (field[:, 1, 1:-1] - field[:, 0, 1:-1]) * inv_dy2
                )
                lap[:, -1, 1:-1] = (
                    (field[:, -1, 2:] + field[:, -1, :-2] - 2 * field[:, -1, 1:-1]) * inv_dx2
                    + (field[:, -2, 1:-1] - field[:, -1, 1:-1]) * inv_dy2
                )
                lap[:, 1:-1, 0] = (
                    (field[:, 1:-1, 1] - field[:, 1:-1, 0]) * inv_dx2
                    + (field[:, 2:, 0] + field[:, :-2, 0] - 2 * field[:, 1:-1, 0]) * inv_dy2
                )
                lap[:, 1:-1, -1] = (
                    (field[:, 1:-1, -2] - field[:, 1:-1, -1]) * inv_dx2
                    + (field[:, 2:, -1] + field[:, :-2, -1] - 2 * field[:, 1:-1, -1]) * inv_dy2
                )
                field = torch.clamp(
                    field + dt * (
                        diff * lap
                        + supply * vascular * (1 - field)
                        - sinks * (field / (michaelis + field))
                    ),
                    0, 1,
                )
            result = field.to("cpu").numpy().astype(np.float64, copy=False)
    except Exception:
        # A backend that accepts the tensors but fails mid-kernel must not take the
        # simulation down; the caller's NumPy path produces the reference result.
        return None

    return result[0], result[1]


def _doctor() -> int:
    report = status()
    print("EVOCURE compute backend")
    print("=" * 60)
    print(f"  Requested (EVOCURE_COMPUTE) : {report['preference']}")
    print(f"  Active backend              : {report['numerical_backend']}")
    print(f"  Device                      : {report['gpu_name'] or 'host CPU'}")
    print(f"  Field precision             : {report['precision']}")
    print(f"  PyTorch installed           : {'yes' if report['torch_installed'] else 'no'}")
    detected = report["detected_backends"]
    print(f"  Usable accelerators         : {len(detected)}")
    for entry in detected:
        print(f"    - {entry['backend']:9s} {entry['vendor']:10s} {entry['name']} ({entry['precision']})")
    for note in report["backend_notes"]:
        print(f"  Note: {note}")
    if not detected:
        print("\n  No accelerator found. The CPU path is fully supported and is the")
        print("  reproducibility reference; see docs/GPU_SUPPORT.md to enable a GPU.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_doctor())
