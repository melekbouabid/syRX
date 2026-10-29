"""Checks run before spending bandwidth or disk: space, permissions, identity, tooling."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from synergyrx_data.config import Settings

# Raw file + parsed copy + working room.
SPACE_FACTOR = 2.5
MIN_FREE_BYTES = 512 * 1024**2


class PreflightError(RuntimeError):
    """A precondition for a run is not met."""


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = True


def _existing_parent(path: Path) -> Path:
    path = path.resolve()
    while not path.exists() and path.parent != path:
        path = path.parent
    return path


def free_bytes(path: Path) -> int:
    return shutil.disk_usage(_existing_parent(path)).free


def ensure_disk_space(path: Path, need_bytes: int) -> None:
    """Raise before a download that would not fit, counting room for the parsed copy."""
    required = int(need_bytes * SPACE_FACTOR) + MIN_FREE_BYTES
    available = free_bytes(path)
    if available < required:
        raise PreflightError(
            f"not enough disk space under {path}: need about {required / 1e9:.1f} GB "
            f"({need_bytes / 1e9:.2f} GB download), {available / 1e9:.1f} GB free"
        )


def run_preflight(settings: Settings, planned_bytes: int = 0) -> list[Check]:
    """Return the full checklist; callers decide whether a failed required check is fatal."""
    checks: list[Check] = []
    version = sys.version_info
    checks.append(
        Check("python", version >= (3, 11), f"{version.major}.{version.minor}.{version.micro}")
    )
    checks.append(
        Check("rdkit", importlib.util.find_spec("rdkit") is not None, "needed for SMILES cleaning")
    )
    checks.append(
        Check(
            "pytdc",
            importlib.util.find_spec("tdc") is not None,
            "optional: the TDC connector falls back to the same Harvard Dataverse file",
            required=False,
        )
    )
    env_file = settings.project_root / ".env"
    checks.append(Check(".env", env_file.is_file(), str(env_file), required=False))
    checks.append(
        Check(
            "contact_email",
            bool(settings.secrets.contact_email),
            "CONTACT_EMAIL is sent in the User-Agent; required for live requests",
            required=settings.http.require_contact_email,
        )
    )
    checks.append(
        Check(
            "ncbi_api_key",
            bool(settings.secrets.api_key),
            "optional: E-utilities limit is about 3 req/s without it, 10 req/s with it",
            required=False,
        )
    )
    probe_dir = _existing_parent(settings.data_root)
    try:
        probe = probe_dir / ".synergyrx_write_test"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
        writable = True
    except OSError:
        writable = False
    checks.append(Check("data_dir_writable", writable, str(settings.data_root)))
    available = free_bytes(settings.data_root)
    required = int(planned_bytes * SPACE_FACTOR) + MIN_FREE_BYTES
    checks.append(
        Check(
            "disk_space",
            available >= required,
            f"{available / 1e9:.1f} GB free, about {required / 1e9:.1f} GB needed "
            f"for the planned downloads",
        )
    )
    return checks
