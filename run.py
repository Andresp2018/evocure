"""
EVOCURE Unified Launcher & Entrypoint (Godot 4.7 Bioluminescent Frontend + Python Scientific Server).

One-command runner for EVOCURE & EVOCURE PLAY:
- Starts the isolated .venv FastAPI + WebSocket scientific backend on 127.0.0.1:8000.
- Automatically launches the native Godot 4.7 desktop frontend (default), with fallback
  or co-launch support for the local HTML5 web workbench (--frontend web | both | godot | none).
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser

BASE_DIR = pathlib.Path(__file__).resolve().parent
GODOT_PROJECT_DIR = BASE_DIR / "godot_frontend"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def find_godot_executable() -> str | None:
    """Locate Godot 4 executable across env vars, bundled tools, PATH, and WinGet."""
    env_bin = os.environ.get("EVOCURE_GODOT_BIN", "").strip()
    if env_bin and pathlib.Path(env_bin).exists():
        return env_bin

    # Check local tools/godot directory
    tools_godot = BASE_DIR / "tools" / "godot"
    if tools_godot.exists():
        for exe in sorted(tools_godot.rglob("*Godot*.exe"), reverse=True):
            if "console" not in exe.name.lower():
                return str(exe)

    for cmd_name in ("godot4", "godot"):
        found = shutil.which(cmd_name)
        if found:
            return found

    if os.name == "nt":
        winget_pkgs = pathlib.Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
        if winget_pkgs.exists():
            candidates = [
                p for p in winget_pkgs.rglob("Godot_v4*_win64.exe")
                if "console" not in p.name.lower()
            ]
            if candidates:
                candidates.sort(key=lambda p: p.name, reverse=True)
                return str(candidates[0])

        for prog_env in ("ProgramFiles", "ProgramFiles(x86)"):
            root = pathlib.Path(os.environ.get(prog_env, ""))
            if root.exists() and (root / "Godot").exists():
                for exe in (root / "Godot").rglob("*Godot*.exe"):
                    if "console" not in exe.name.lower():
                        return str(exe)

    return None


def detect_system_telemetry() -> dict[str, str]:
    """Detect available local CPU, RAM, GPU, Godot 4 runtime, and AI model status."""
    godot_bin = find_godot_executable()
    info = {
        "cpu": f"{os.cpu_count() or 'Unknown'} logical cores",
        "ram": "Unknown",
        "gpu": "Host CPU only",
        "godot": pathlib.Path(godot_bin).name if godot_bin else "Not detected (Web UI fallback ready)",
        "gemma": "Not found",
        "cloud_ai": "Optional (Google Gemini & OpenAI configurable in Settings)",
        "laya": "Not found",
        "qsar": "Not found",
    }

    try:
        import psutil
        vm = psutil.virtual_memory()
        info["ram"] = f"{round(vm.total / (1024**3), 1)} GB (Free: {round(vm.available / (1024**3), 1)} GB)"
    except Exception:
        pass

    try:
        from ai.orchestration.registry import get_ai_registry
        reg = get_ai_registry()
        hw = reg.get_hardware_telemetry()
        gpu = hw.get("gpu", {})
        if gpu and gpu.get("name"):
            info["gpu"] = f"{gpu.get('name')} ({round(gpu.get('vram_total_mb', 0)/1024.0, 1)} GB VRAM)"
    except Exception:
        pass

    gemma_path = BASE_DIR / "models" / "gemma" / "Gemma-4-E2B-Heretic-Q4_K_M.gguf"
    if gemma_path.exists():
        size_gb = round(gemma_path.stat().st_size / (1024**3), 2)
        info["gemma"] = f"Gemma-4-E2B Heretic Q4_K_M ({size_gb} GB)"

    laya_path = BASE_DIR / "models" / "laya" / "multilingual" / "model.safetensors"
    if laya_path.exists():
        info["laya"] = "Multilingual SafeTensors (Offline Validated)"

    qsar_path = BASE_DIR / "models" / "scientific" / "pdac_qsar_v1.joblib"
    if qsar_path.exists():
        info["qsar"] = "Split-Conformal Calibrated Random Forest QSAR"

    return info


def print_banner(host: str, port: int, frontend_mode: str, telem: dict[str, str]) -> None:
    url = f"http://{host}:{port}/"
    border = "=" * 82
    print("\n" + border)
    print("   EVOCURE - Evolutionary Chemical Universe & Cancer Research Engine (Godot 4)")
    print("           Local Scientific Core • Bioluminescent Godot Client • Multi-AI")
    print(border)
    print(f"  Active Frontend    : {frontend_mode.upper()} ({telem['godot']})")
    print(f"  Local Web / API    : {url}  |  OpenAPI: {url}docs")
    print(f"  WebSocket Channel  : ws://{host}:{port}/ws")
    print("  ------------------------------------------------------------------------------")
    print(f"  Hardware           : CPU: {telem['cpu']} | RAM: {telem['ram']}")
    print(f"  GPU Acceleration   : {telem['gpu']}")
    print(f"  Local Gemma 4 AI   : {telem['gemma']}")
    print(f"  Cloud AI Optional  : {telem['cloud_ai']}")
    print(f"  Laya Neural Engine : {telem['laya']}")
    print(f"  Predictive QSAR    : {telem['qsar']}")
    print("  Epistemic Standard : Section 27 Transparent Attribution (Zero Uncalibrated Claims)")
    print(border + "\n")


def _wait_for_backend(url: str, timeout_s: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url + "api/v1/health", timeout=1.5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.35)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="EVOCURE Local Platform Launcher")
    parser.add_argument(
        "--host",
        default=os.environ.get("EVOCURE_HOST", "127.0.0.1"),
        help="Host address to bind (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("EVOCURE_PORT", "8000")),
        help="Port to listen on (default: 8000)",
    )
    parser.add_argument(
        "--frontend",
        choices=["godot", "web", "both", "none"],
        default=os.environ.get("EVOCURE_FRONTEND", "godot"),
        help="Frontend client to launch automatically (default: godot)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Headless server mode: do not launch Godot GUI or web browser automatically",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn hot reloading",
    )
    args = parser.parse_args()
    requested_port = args.port
    for candidate_port in range(requested_port, requested_port + 20):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.bind((args.host, candidate_port))
            args.port = candidate_port
            break
        except OSError:
            continue
    else:
        raise SystemExit("No free local port was found. Close an older EVOCURE server and try again.")
    if args.port != requested_port:
        print(f"[EVOCURE] Port {requested_port} is already in use; starting this session on {args.port}.")

    effective_frontend = "none" if args.no_browser else args.frontend
    telemetry = detect_system_telemetry()
    print_banner(args.host, args.port, effective_frontend, telemetry)

    godot_proc_holder: list[subprocess.Popen] = []

    if effective_frontend != "none":
        target_url = f"http://{args.host}:{args.port}/"
        exported_client = BASE_DIR / "build" / ("windows/EVOCURE.exe" if os.name == "nt" else "linux/EVOCURE.x86_64")
        godot_exe = str(exported_client) if exported_client.is_file() else find_godot_executable()

        def _launch_clients() -> None:
            if not _wait_for_backend(target_url, timeout_s=120.0):
                return

            try:
                request = urllib.request.Request(target_url + "api/v1/playback", data=json.dumps({"action": "play", "speed_multiplier": 1, "target": "universe"}).encode(), headers={"Content-Type": "application/json"})
                urllib.request.urlopen(request, timeout=5).close()
            except Exception as exc:
                print(f"[EVOCURE] Could not start playback: {exc}")
            launched_godot = False
            if effective_frontend in ("godot", "both"):
                if godot_exe and (GODOT_PROJECT_DIR / "project.godot").exists():
                    try:
                        cmd = [godot_exe]
                        if not exported_client.is_file():
                            cmd += ["--path", str(GODOT_PROJECT_DIR)]
                        cmd += ["--", f"--backend-url=http://{args.host}:{args.port}"]
                        p = subprocess.Popen(cmd, cwd=str(GODOT_PROJECT_DIR))
                        godot_proc_holder.append(p)
                        launched_godot = True
                    except Exception as exc:
                        print(f"[EVOCURE] Could not launch Godot binary ({exc}); falling back to browser.")

            if effective_frontend in ("web", "both") or (effective_frontend == "godot" and not launched_godot):
                try:
                    webbrowser.open(target_url)
                except Exception:
                    pass

        threading.Thread(target=_launch_clients, daemon=True).start()

    import uvicorn
    try:
        uvicorn.run("app.main:app", host=args.host, port=args.port, reload=args.reload)
    finally:
        for proc in godot_proc_holder:
            if proc.poll() is None:
                try:
                    proc.terminate()
                except Exception:
                    pass


if __name__ == "__main__":
    main()
