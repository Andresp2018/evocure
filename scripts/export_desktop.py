"""Build native Godot clients using the locally installed export templates."""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run import find_godot_executable


def main():
    parser = argparse.ArgumentParser(description="Export the EVOCURE desktop frontend")
    parser.add_argument("--platform", choices=["windows", "linux", "all"], default="windows")
    args = parser.parse_args()
    godot = find_godot_executable()
    if not godot:
        raise SystemExit("Install Godot 4.7 and its matching export templates, or set EVOCURE_GODOT_BIN.")
    platforms = ["windows", "linux"] if args.platform == "all" else [args.platform]
    for platform in platforms:
        destination = ROOT / "build" / platform
        destination.mkdir(parents=True, exist_ok=True)
        preset = "Windows Desktop" if platform == "windows" else "Linux Desktop"
        subprocess.run([godot, "--headless", "--path", str(ROOT / "godot_frontend"), "--editor", "--import", "--quit"], check=True)
        subprocess.run([godot, "--headless", "--path", str(ROOT / "godot_frontend"), "--export-release", preset], check=True)
        print(f"Exported {preset} to {destination}")
    print("Use start_evocure.bat / start_evocure.sh to launch the client together with its Python engine.")


if __name__ == "__main__":
    main()
