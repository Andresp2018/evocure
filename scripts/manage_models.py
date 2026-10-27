"""
EVOCURE Model Manager CLI (Section 2.15).

Provides safe management of local AI models and weights:
- Hardware capability inspection and model quantization recommendations.
- Local model inventory listing (size in GB, license, local presence).
- Safe download preparation with disk space verification and explicit confirmation gate.
- NEVER auto-downloads large weights without the explicit '--confirm-download' flag.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.local_llm.gemma_assistant import LocalGemmaAssistant
from ai.orchestration.registry import get_ai_registry


AVAILABLE_MODELS: Dict[str, Dict[str, Any]] = {
    "gemma-4-E2B-it-Q4_K_M": {
        "model_id": "gemma-4-E2B-it-Q4_K_M",
        "description": "Gemma 4 2B Parameter Instruction-Tuned (4-bit quantized GGUF)",
        "size_gb": 1.85,
        "license": "Gemma Terms of Use / Open Weights",
        "target_dir": "models/gemma",
        "filename": "Gemma-4-E2B-Heretic-Q4_K_M.gguf",
        "min_ram_gb": 8.0,
        "min_vram_gb": 4.0,
        "recommended_for": "Standard desktop CPU or budget GPU (<= 8GB VRAM)",
    },
    "gemma-4-E4B-it-Q4_K_M": {
        "model_id": "gemma-4-E4B-it-Q4_K_M",
        "description": "Gemma 4 4B Parameter Instruction-Tuned (4-bit quantized GGUF)",
        "size_gb": 3.65,
        "license": "Gemma Terms of Use / Open Weights",
        "target_dir": "models/gemma",
        "filename": "Gemma-4-E4B-it-Q4_K_M.gguf",
        "min_ram_gb": 16.0,
        "min_vram_gb": 8.0,
        "recommended_for": "Workstation GPU with 8-12GB VRAM (e.g. RTX 3060/4060)",
    },
    "gemma-4-27B-A4B-Q4_K_M": {
        "model_id": "gemma-4-27B-A4B-Q4_K_M",
        "description": "Gemma 4 27B Parameter High-Precision (4-bit quantized GGUF)",
        "size_gb": 17.2,
        "license": "Gemma Terms of Use / Open Weights",
        "target_dir": "models/gemma",
        "filename": "Gemma-4-27B-A4B-Q4_K_M.gguf",
        "min_ram_gb": 32.0,
        "min_vram_gb": 16.0,
        "recommended_for": "High-end workstation (16GB+ VRAM & 32GB+ RAM)",
    },
    "laya-multilingual": {
        "model_id": "laya-multilingual",
        "description": "Laya AI Multilingual Decision Checkpoint (RL Agent & Encoders)",
        "size_gb": 0.45,
        "license": "Apache 2.0 / Open Source",
        "target_dir": "models/laya/multilingual",
        "filename": "model.safetensors",
        "min_ram_gb": 4.0,
        "min_vram_gb": 0.0,
        "recommended_for": "Structured multi-attribute candidate triage and event policy",
    },
    "pdac-qsar-v1": {
        "model_id": "pdac-qsar-v1",
        "description": "Multi-Disease & PDAC Calibrated QSAR Regressor with Split-Conformal Intervals",
        "size_gb": 0.08,
        "license": "CC-BY 4.0 / EVOCURE Open Science",
        "target_dir": "models/scientific",
        "filename": "pdac_qsar_v1.joblib",
        "min_ram_gb": 2.0,
        "min_vram_gb": 0.0,
        "recommended_for": "In silico pIC50 and growth-inhibition probability prediction across diseases",
    },
}


def get_free_disk_space_gb(path: Path) -> float:
    """Calculate free disk space in gigabytes on target drive."""
    # Ensure path or ancestor exists for disk_usage
    check_dir = path
    while not check_dir.exists() and check_dir.parent != check_dir:
        check_dir = check_dir.parent
    usage = shutil.disk_usage(check_dir)
    return round(usage.free / (1024 ** 3), 2)


def cmd_list() -> None:
    """List all registered models and local availability."""
    print("=" * 80)
    print("EVOCURE AI Model Inventory & Availability Ledger")
    print("=" * 80)
    print(f"{'MODEL ID':<25} {'SIZE (GB)':<10} {'STATUS':<12} {'LICENSE':<22} {'TARGET DIR'}")
    print("-" * 80)

    for mid, info in AVAILABLE_MODELS.items():
        target_path = PROJECT_ROOT / info["target_dir"] / info["filename"]
        status = "INSTALLED" if target_path.exists() else "AVAILABLE"
        print(
            f"{mid:<25} {info['size_gb']:<10.2f} {status:<12} {info['license'][:20]:<22} {info['target_dir']}"
        )

    print("=" * 80)
    print("Note: To inspect hardware compatibility, run: python scripts/manage_models.py --hardware-check")


def cmd_hardware_check() -> None:
    """Run hardware inspection and output quantization recommendation."""
    assistant = LocalGemmaAssistant()
    hw = assistant.hardware_profile

    free_disk = get_free_disk_space_gb(PROJECT_ROOT / "models")

    print("=" * 80)
    print("EVOCURE Host Hardware Capability Inspection (Section 2.15)")
    print("=" * 80)
    print(f"  CPU Cores:              {hw.cpu_cores_logical} logical ({hw.cpu_cores_physical} physical)")
    print(f"  System RAM:             {hw.ram_total_gb:.2f} GB total ({hw.ram_available_gb:.2f} GB available)")
    if hw.gpu_name:
        print(f"  Detected GPU:           {hw.gpu_name}")
        print(f"  GPU Vendor:             {hw.gpu_vendor or 'Unknown'}")
        total = f"{hw.vram_total_gb:.2f} GB total" if hw.vram_total_gb else "VRAM unknown"
        free = f" ({hw.vram_free_gb:.2f} GB free)" if hw.vram_free_gb else ""
        print(f"  GPU VRAM:               {total}{free}")
        print(f"  LLM Offload Backend:    {hw.gpu_offload_backend}")
    else:
        print("  Detected GPU:           None detected (CPU-only execution)")

    try:
        from simulation.core.acceleration import status as compute_status
        report = compute_status()
        print(f"  Simulation Backend:     {report['numerical_backend']}")
        print(f"  Field Precision:        {report['precision']}")
    except Exception:
        pass

    print(f"  Free Disk Space:        {free_disk:.2f} GB (drive: {PROJECT_ROOT.drive or '/'})")
    print("-" * 80)
    print(f"  RECOMMENDED MODEL:      {hw.recommended_model}")
    print(f"  RECOMMENDATION REASON:  {hw.recommendation_reason}")
    print("=" * 80)


def cmd_prepare_download(model_id: str, confirm_download: bool) -> None:
    """Safely prepare or gate a model download with explicit consent verification."""
    info = AVAILABLE_MODELS.get(model_id)
    if not info:
        print(f"ERROR: Unknown model ID '{model_id}'. Run with --list to see available models.", file=sys.stderr)
        sys.exit(1)

    target_dir = PROJECT_ROOT / info["target_dir"]
    target_file = target_dir / info["filename"]
    free_disk = get_free_disk_space_gb(target_dir)

    print("=" * 80)
    print(f"Model Download Pre-Flight Verification: {model_id}")
    print("=" * 80)
    print(f"  Description:            {info['description']}")
    print(f"  Exact Model Size:       {info['size_gb']:.2f} GB")
    print(f"  License Terms:          {info['license']}")
    print(f"  Destination Path:       {target_file}")
    print(f"  Current Free Disk:      {free_disk:.2f} GB")
    
    # Check if file already exists
    if target_file.exists():
        actual_size = round(target_file.stat().st_size / (1024 ** 3), 2)
        print(f"\n[OK] Model is ALREADY INSTALLED locally at {target_file} ({actual_size} GB).")
        print("No download necessary. Offline execution is ready.")
        return

    # Check disk space buffer
    required_space = info["size_gb"] + 1.0
    if free_disk < required_space:
        print(
            f"\n[ERROR] Insufficient disk space! Required: {required_space:.2f} GB, Available: {free_disk:.2f} GB.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Section 2.15 Gate: Refuse without explicit confirm flag
    if not confirm_download:
        print("\n" + "!" * 80)
        print("SAFETY REFUSAL: Automatic model download blocked per EVOCURE Policy (Section 2.15).")
        print("EVOCURE operates 100% offline and will NEVER auto-download large weights in the background.")
        print(f"To authorize downloading {info['size_gb']:.2f} GB under '{info['license']}', execute:")
        print(f"    python scripts/manage_models.py --prepare-download {model_id} --confirm-download")
        print("!" * 80)
        sys.exit(2)

    # User explicitly confirmed download
    print(f"\n[AUTHORIZED] Explicit confirmation received for {model_id}.")
    print(f"Preparing destination directory: {target_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Pre-flight checks passed: {free_disk:.2f} GB available on target volume.")
    print("Note: In production offline deployments, weights should be staged via secure artifact copy.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EVOCURE Safe AI Model Manager CLI (Section 2.15)"
    )
    parser.add_argument(
        "--list", action="store_true", help="List all available AI models and local status"
    )
    parser.add_argument(
        "--hardware-check", action="store_true", help="Inspect host hardware and recommend model quantization"
    )
    parser.add_argument(
        "--prepare-download", metavar="MODEL_ID", type=str, help="Pre-flight verification for downloading model weights"
    )
    parser.add_argument(
        "--confirm-download", action="store_true", help="Explicit consent flag required to authorize model download"
    )
    parser.add_argument(
        "--status", action="store_true", help="Output JSON snapshot of model registry and hardware telemetry"
    )

    args = parser.parse_args()

    if args.list:
        cmd_list()
    elif args.hardware_check:
        cmd_hardware_check()
    elif args.prepare_download:
        cmd_prepare_download(args.prepare_download, args.confirm_download)
    elif args.status:
        registry = get_ai_registry()
        print(json.dumps(registry.get_ledger_snapshot(), indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
