"""Inspect, verify and fetch the optional local AI artifacts.

No model weights are stored in this repository. This script reads models/manifest.json
and tells you what is present, what is missing and what each missing piece costs you.

    python scripts/fetch_models.py --status     what is installed
    python scripts/fetch_models.py --verify     checksum what is installed
    python scripts/fetch_models.py --fetch all  download what has a known source

EVOCURE runs with none of these installed. Every optional artifact has a deterministic
fallback, and the published example report was produced with the fallbacks active.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "models" / "manifest.json"
CHUNK = 1024 * 1024


def human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.is_file():
        print(f"ERROR: manifest not found at {MANIFEST_PATH}", file=sys.stderr)
        raise SystemExit(1)
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_url(artifact: dict[str, Any], entry: dict[str, Any]) -> str | None:
    """A manifest url wins; otherwise fall back to a user supplied mirror."""
    if entry.get("url"):
        return str(entry["url"])
    if artifact.get("url"):
        return str(artifact["url"]).rstrip("/") + "/" + Path(entry["path"]).name
    mirror = os.environ.get("EVOCURE_MODEL_MIRROR", "").strip()
    if mirror:
        return mirror.rstrip("/") + "/" + entry["path"]
    return None


def cmd_status(manifest: dict[str, Any], verify: bool) -> int:
    print("=" * 78)
    print("EVOCURE optional AI artifacts")
    print("=" * 78)
    problems = 0

    for artifact in manifest["artifacts"]:
        files = artifact.get("files") or []
        present = [f for f in files if (PROJECT_ROOT / f["path"]).is_file()]
        total = len(files)

        if total == 0:
            state = "not tracked file by file"
        elif len(present) == total:
            state = "installed"
        elif present:
            state = f"partial ({len(present)}/{total} files)"
        else:
            state = "not installed"

        print(f"\n  {artifact['id']}  [{state}]")
        print(f"    Purpose     : {artifact['required_for']}")
        print(f"    License     : {artifact.get('license', 'see homepage')}")
        if artifact.get("rebuildable"):
            print(f"    Rebuild     : {artifact['rebuild_command']}")
        elif artifact.get("homepage"):
            print(f"    Obtain from : {artifact['homepage']}")
        if artifact.get("install_dir"):
            print(f"    Install to  : {artifact['install_dir']}")
        if artifact.get("note"):
            print(f"    Note        : {artifact['note']}")

        for entry in files:
            path = PROJECT_ROOT / entry["path"]
            if not path.is_file():
                print(f"      missing  {entry['path']}  ({human(entry['bytes'])})")
                continue
            size = path.stat().st_size
            if not verify:
                flag = "ok" if size == entry["bytes"] else "size differs"
                print(f"      {flag:12s} {entry['path']}  ({human(size)})")
                continue
            actual = sha256_of(path)
            if actual == entry["sha256"]:
                print(f"      verified     {entry['path']}")
            else:
                problems += 1
                print(f"      CHECKSUM MISMATCH  {entry['path']}")
                print(f"         expected {entry['sha256']}")
                print(f"         actual   {actual}")
                if entry.get("note"):
                    print(f"         {entry['note']}")

    print("\n" + "-" * 78)
    print("  EVOCURE runs without any of these. Missing artifacts fall back to")
    print("  deterministic rule based paths, and the dashboard names the active provider.")
    if problems:
        print(f"\n  {problems} checksum mismatch(es). A rebuilt QSAR model is expected to differ;")
        print("  an unexpected difference in a downloaded weight file is worth investigating.")
    print("=" * 78)
    return 0


def download(url: str, destination: Path, expected_bytes: int) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    print(f"    downloading {url}")
    try:
        with urllib.request.urlopen(url, timeout=30) as response, partial.open("wb") as out:
            fetched = 0
            while True:
                block = response.read(CHUNK)
                if not block:
                    break
                out.write(block)
                fetched += len(block)
                if expected_bytes:
                    percent = 100.0 * fetched / expected_bytes
                    print(f"\r      {human(fetched)} / {human(expected_bytes)} ({percent:.1f}%)",
                          end="", flush=True)
            print()
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        print(f"    failed: {exc}", file=sys.stderr)
        partial.unlink(missing_ok=True)
        return False
    partial.replace(destination)
    return True


def cmd_fetch(manifest: dict[str, Any], target: str) -> int:
    wanted = [a for a in manifest["artifacts"] if target in ("all", a["id"])]
    if not wanted:
        ids = ", ".join(a["id"] for a in manifest["artifacts"])
        print(f"ERROR: unknown artifact '{target}'. Known ids: {ids}, all", file=sys.stderr)
        return 1

    unresolved: list[dict[str, Any]] = []
    for artifact in wanted:
        print(f"\n  {artifact['id']}")
        if artifact.get("rebuildable"):
            print(f"    Rebuild locally instead of downloading: {artifact['rebuild_command']}")
            continue

        for entry in artifact.get("files") or []:
            destination = PROJECT_ROOT / entry["path"]
            if destination.is_file() and destination.stat().st_size == entry["bytes"]:
                print(f"    already present: {entry['path']}")
                continue
            url = resolve_url(artifact, entry)
            if not url:
                unresolved.append({"artifact": artifact, "entry": entry})
                continue
            if download(url, destination, entry["bytes"]):
                actual = sha256_of(destination)
                if actual == entry["sha256"]:
                    print(f"    verified: {entry['path']}")
                else:
                    print(f"    WARNING: checksum differs for {entry['path']}")
                    print(f"      expected {entry['sha256']}")
                    print(f"      actual   {actual}")

    if unresolved:
        print("\n" + "=" * 78)
        print("  These artifacts have no pinned download URL in the manifest.")
        print("=" * 78)
        print("  Model weights are not redistributed through this repository, and their")
        print("  licenses are the publisher's to grant. Download them yourself, or point")
        print("  EVOCURE_MODEL_MIRROR at a location you control:")
        print("\n    EVOCURE_MODEL_MIRROR=https://your-host/evocure python scripts/fetch_models.py --fetch all\n")
        for item in unresolved:
            artifact, entry = item["artifact"], item["entry"]
            print(f"    {entry['path']}")
            print(f"      {human(entry['bytes'])}, sha256 {entry['sha256']}")
            if artifact.get("homepage"):
                print(f"      source: {artifact['homepage']}")
            print(f"      license: {artifact.get('license', 'see source')}")
        print("\n  Nothing here is required. EVOCURE falls back to deterministic paths.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect, verify and fetch EVOCURE's optional AI artifacts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--status", action="store_true", help="show what is installed (default)")
    group.add_argument("--verify", action="store_true", help="checksum every installed file")
    group.add_argument("--fetch", metavar="ID", help="download an artifact id, or 'all'")
    args = parser.parse_args()

    manifest = load_manifest()
    if args.fetch:
        return cmd_fetch(manifest, args.fetch)
    return cmd_status(manifest, verify=args.verify)


if __name__ == "__main__":
    raise SystemExit(main())
