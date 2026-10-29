"""Where EVOCURE's data, models and configs actually live.

Every module used to locate its data with `Path(__file__).resolve().parents[2]`,
which says "two directories above this package". That is true only while the
packages sit directly inside a checkout. After `pip install .`, the documented
install route, the packages sit in site-packages, so `parents[2]` resolves to
site-packages itself and nothing is found there:

    .venv/lib/python3.13/site-packages/models/scientific/pdac_qsar_v1.joblib

Measured consequence, which is worse than a missing file. The discovery pipeline
catches the resulting FileNotFoundError and continues with an unfitted QSAR, so
every candidate comes back with predicted_pdac_activity 0.0, no IC50, no tissue
simulation and triage "chemistry_only" - a run that looks complete and has no
potency model in it at all. The tests never caught it because pytest puts the
checkout on sys.path ahead of site-packages, so the suite only ever exercises
the in-repo copy.

Resolution order, first match wins:

  1. EVOCURE_HOME, if set. The explicit answer for a packaged or relocated install.
  2. The nearest ancestor of this file that looks like an EVOCURE root.
  3. The current working directory, or an ancestor of it, that looks like one.
  4. The historical parents[2], so nothing gets worse than it was.

"Looks like an EVOCURE root" means it carries the read-only payload this project
cannot run without: configs/default_universe.json and data/reference. Checking
for content rather than for pyproject.toml matters because a wheel install has
no pyproject.toml and an unrelated parent directory may have one.

Writable state is resolved separately. Checkpoints, SQLite databases and reports
must never be written into site-packages, which is often read-only and is not
the user's data in any case.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

# A directory is an EVOCURE root when it carries the payload the engine loads at
# import time. Both must be present; either alone is too weak a signal.
_ROOT_MARKERS: tuple[str, ...] = (
    "configs/default_universe.json",
    "data/reference",
)

_FALLBACK_ROOT: Path = Path(__file__).resolve().parents[2]


def _looks_like_root(candidate: Path) -> bool:
    try:
        return all((candidate / marker).exists() for marker in _ROOT_MARKERS)
    except OSError:
        return False


def _search_upwards(start: Path) -> Path | None:
    for directory in (start, *start.parents):
        if _looks_like_root(directory):
            return directory
    return None


@lru_cache(maxsize=1)
def project_root() -> Path:
    """The directory holding configs/, data/, models/ and reports/.

    Cached, because it is read on nearly every import and the answer cannot
    change within a process. Call project_root.cache_clear() in a test that
    moves the root.
    """
    override = os.environ.get("EVOCURE_HOME")
    if override:
        path = Path(override).expanduser()
        try:
            return path.resolve()
        except OSError:
            return path

    from_package = _search_upwards(Path(__file__).resolve().parent)
    if from_package is not None:
        return from_package

    try:
        from_cwd = _search_upwards(Path.cwd().resolve())
    except OSError:
        from_cwd = None
    if from_cwd is not None:
        return from_cwd

    return _FALLBACK_ROOT


def is_resolved_root() -> bool:
    """Whether the located root actually carries the expected payload.

    False means EVOCURE fell back to a guess, so a caller reporting a missing
    model can say the root itself was never found rather than blaming the file.
    """
    return _looks_like_root(project_root())


def project_path(*parts: str) -> Path:
    """A path inside the project root. Nothing is created."""
    return project_root().joinpath(*parts)


def configs_dir() -> Path:
    return project_path("configs")


def reference_data_dir() -> Path:
    return project_path("data", "reference")


def models_dir() -> Path:
    return project_path("models")


def scientific_model_path(filename: str = "pdac_qsar_v1.joblib") -> Path:
    return project_path("models", "scientific", filename)


@lru_cache(maxsize=1)
def writable_root() -> Path:
    """Where this process may write checkpoints, databases and reports.

    EVOCURE_DATA_HOME wins when set. Otherwise the project root is used when it
    is writable, which keeps a checkout's experiments/ and reports/ exactly where
    they have always been. A read-only root, which is what a system-wide install
    gives you, falls back to the user's state directory so a run does not die
    trying to write into site-packages.
    """
    override = os.environ.get("EVOCURE_DATA_HOME")
    if override:
        return Path(override).expanduser()

    root = project_root()
    if os.access(root, os.W_OK):
        return root

    base = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(base).expanduser() / "evocure"


def writable_path(*parts: str) -> Path:
    """A path under the writable root, with its parent directory created."""
    target = writable_root().joinpath(*parts)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return target


def describe() -> dict[str, object]:
    """Everything a diagnostic needs to explain a missing-file error."""
    root = project_root()
    return {
        "project_root": str(root),
        "root_resolved_from_markers": is_resolved_root(),
        "evocure_home_env": os.environ.get("EVOCURE_HOME"),
        "evocure_data_home_env": os.environ.get("EVOCURE_DATA_HOME"),
        "writable_root": str(writable_root()),
        "configs_present": configs_dir().is_dir(),
        "reference_data_present": reference_data_dir().is_dir(),
        "qsar_artifact_present": scientific_model_path().exists(),
        "markers_checked": list(_ROOT_MARKERS),
    }


if __name__ == "__main__":  # pragma: no cover - operator diagnostic
    for key, value in describe().items():
        print(f"{key:28s} {value}")
