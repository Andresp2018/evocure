"""Vendor neutral GPU inventory.

Reports the GPUs present on the host regardless of who made them, by trying the
vendor tools in turn and falling back to what PyTorch and the OS can tell us:

    nvidia-smi      NVIDIA, with live VRAM and utilization
    rocm-smi        AMD on Linux, with live VRAM
    PyTorch         any vendor with a working torch backend, total VRAM only
    Windows CIM     any vendor, name and adapter RAM only
    Linux sysfs     any vendor, name only

Nothing here is required for the simulation to run; it feeds the dashboard and
the setup doctor. Every probe is wrapped so a missing or broken tool is silent.
"""

from __future__ import annotations

from typing import Any

import json
import os
import platform
import shutil
import subprocess

_TIMEOUT_S = 2.0


def _run(cmd: list[str], timeout: float = _TIMEOUT_S) -> str | None:
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0,
        )
    except Exception:
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return result.stdout.strip()


def _probe_nvidia_smi() -> list[dict[str, Any]]:
    binary = shutil.which("nvidia-smi")
    if not binary:
        return []
    out = _run([
        binary,
        "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ])
    if not out:
        return []
    found = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            found.append({
                "name": parts[0],
                "vendor": "NVIDIA",
                "vram_total_mb": float(parts[1]),
                "vram_used_mb": float(parts[2]),
                "vram_free_mb": float(parts[3]),
                "gpu_utilization_pct": float(parts[4]),
                "source": "nvidia-smi",
            })
        except ValueError:
            continue
    return found


def _probe_rocm_smi() -> list[dict[str, Any]]:
    binary = shutil.which("rocm-smi")
    if not binary:
        return []
    out = _run([binary, "--showproductname", "--showmeminfo", "vram", "--json"])
    if not out:
        return []
    try:
        payload = json.loads(out)
    except ValueError:
        return []
    found = []
    for key, card in payload.items():
        if not key.lower().startswith("card") or not isinstance(card, dict):
            continue
        name = (
            card.get("Card Series")
            or card.get("Card series")
            or card.get("Card Model")
            or card.get("Card SKU")
            or "AMD GPU"
        )
        entry: dict[str, Any] = {"name": str(name), "vendor": "AMD", "source": "rocm-smi"}
        for field, target in (
            ("VRAM Total Memory (B)", "vram_total_mb"),
            ("VRAM Total Used Memory (B)", "vram_used_mb"),
        ):
            try:
                entry[target] = round(float(card[field]) / (1024 ** 2), 1)
            except (KeyError, TypeError, ValueError):
                pass
        if "vram_total_mb" in entry and "vram_used_mb" in entry:
            entry["vram_free_mb"] = round(entry["vram_total_mb"] - entry["vram_used_mb"], 1)
        found.append(entry)
    return found


def _probe_torch() -> list[dict[str, Any]]:
    try:
        import torch
    except ImportError:
        return []
    found: list[dict[str, Any]] = []
    try:
        if torch.cuda.is_available():
            amd = bool(getattr(torch.version, "hip", None))
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                found.append({
                    "name": props.name,
                    "vendor": "AMD" if amd else "NVIDIA",
                    "vram_total_mb": round(props.total_memory / (1024 ** 2), 1),
                    "source": "torch.cuda (ROCm)" if amd else "torch.cuda",
                })
    except Exception:
        pass
    try:
        xpu = getattr(torch, "xpu", None)
        if xpu is not None and xpu.is_available():
            for index in range(xpu.device_count()):
                found.append({
                    "name": xpu.get_device_name(index),
                    "vendor": "Intel",
                    "source": "torch.xpu",
                })
    except Exception:
        pass
    try:
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            found.append({
                "name": "Apple Silicon GPU (Metal)",
                "vendor": "Apple",
                "source": "torch.mps",
            })
    except Exception:
        pass
    return found


def _probe_windows_cim() -> list[dict[str, Any]]:
    if os.name != "nt":
        return []
    out = _run([
        "powershell", "-NoProfile", "-NonInteractive", "-Command",
        "Get-CimInstance Win32_VideoController | "
        "Select-Object Name,AdapterRAM,AdapterCompatibility | ConvertTo-Json -Compress",
    ], timeout=6.0)
    if not out:
        return []
    try:
        payload = json.loads(out)
    except ValueError:
        return []
    if isinstance(payload, dict):
        payload = [payload]
    found = []
    for card in payload:
        if not isinstance(card, dict) or not card.get("Name"):
            continue
        entry: dict[str, Any] = {
            "name": str(card["Name"]),
            "vendor": str(card.get("AdapterCompatibility") or "Unknown").strip(),
            "source": "Windows CIM",
        }
        ram = card.get("AdapterRAM")
        if isinstance(ram, (int, float)) and ram > 0:
            # Win32_VideoController reports a signed 32 bit value, so it saturates at 4 GB.
            entry["vram_total_mb"] = round(float(ram) / (1024 ** 2), 1)
            entry["vram_total_truncated"] = float(ram) >= (4 * 1024 ** 3) - 1
        found.append(entry)
    return found


def _probe_linux_sysfs() -> list[dict[str, Any]]:
    if platform.system() != "Linux":
        return []
    out = _run(["lspci", "-mm"], timeout=3.0)
    if not out:
        return []
    found = []
    for line in out.splitlines():
        lowered = line.lower()
        if "vga compatible controller" not in lowered and "3d controller" not in lowered:
            continue
        fields = [part.strip('"') for part in line.split('" "')]
        vendor = fields[2] if len(fields) > 2 else "Unknown"
        model = fields[3] if len(fields) > 3 else line
        found.append({
            "name": f"{vendor} {model}".strip(),
            "vendor": vendor.split()[0] if vendor else "Unknown",
            "source": "lspci",
        })
    return found


def detect_gpus() -> list[dict[str, Any]]:
    """Return every GPU found, richest source first, de-duplicated by name."""
    seen: dict[str, dict[str, Any]] = {}
    for probe in (_probe_nvidia_smi, _probe_rocm_smi, _probe_torch,
                  _probe_windows_cim, _probe_linux_sysfs):
        try:
            entries = probe()
        except Exception:
            continue
        for entry in entries:
            key = entry["name"].strip().lower()
            if key in seen:
                # Keep the first (richer) source, fill any gaps from the later one.
                for field, value in entry.items():
                    seen[key].setdefault(field, value)
            else:
                seen[key] = entry
    return list(seen.values())


def primary_gpu() -> dict[str, Any] | None:
    """The GPU most likely to be used for compute: the one with the most known VRAM."""
    gpus = detect_gpus()
    if not gpus:
        return None
    return max(gpus, key=lambda g: g.get("vram_total_mb") or 0.0)
