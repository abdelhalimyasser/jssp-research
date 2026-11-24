"""Real git, toolchain, OS, and hardware provenance capture (no fabricated fields)."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from jssp_research.benchmarks.checksums import sha256_bytes, sha256_file
from jssp_research.native import get_native_bridge_info


class GitProvenance(BaseModel):
    """Git repository commit, branch, clean/dirty state, and diff digest."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    commit: str | None
    branch: str | None
    dirty: bool | None
    diff_sha256: str | None


class EnvironmentProvenance(BaseModel):
    """Host OS, CPU/RAM hardware, compiler, Python toolchain, and lockfile provenance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    os_name: str
    os_release: str
    kernel_version: str
    architecture: str
    cpu_model: str
    physical_core_count: int | None
    logical_core_count: int | None
    ram_total_bytes: int | None
    python_version: str
    uv_version: str
    rustc_version: str
    cargo_version: str
    pyo3_version: str
    maturin_version: str
    cargo_lock_sha256: str | None
    uv_lock_sha256: str | None
    compiler_profile: str
    native_extension_id: str


def _run_text_command(args: list[str], cwd: Path | None = None) -> str | None:
    try:
        completed = subprocess.run(
            args,
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except OSError, subprocess.SubprocessError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _run_bytes_command(args: list[str], cwd: Path | None = None) -> bytes | None:
    try:
        completed = subprocess.run(
            args,
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except OSError, subprocess.SubprocessError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def _read_linux_cpu_info() -> tuple[str, int | None]:
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.is_file():
        return "unavailable", None

    model_name = "unavailable"
    physical_pairs: set[tuple[str, str]] = set()
    curr_phys = ""
    curr_core = ""

    for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            if curr_phys and curr_core:
                physical_pairs.add((curr_phys, curr_core))
            curr_phys = ""
            curr_core = ""
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        val = value.strip()
        if key == "model name" and model_name == "unavailable" and val:
            model_name = val
        elif key == "physical id":
            curr_phys = val
        elif key == "core id":
            curr_core = val

    if curr_phys and curr_core:
        physical_pairs.add((curr_phys, curr_core))

    physical_count = len(physical_pairs) if physical_pairs else None
    return model_name, physical_count


def _read_linux_ram_bytes() -> int | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        return None
    for line in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("MemTotal:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1]) * 1024
    return None


def _installed_package_version(dist_name: str) -> str:
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _extract_cargo_lock_package_version(cargo_lock_path: Path, package_name: str) -> str:
    if not cargo_lock_path.is_file():
        return "unavailable"
    in_pkg = False
    matched_name = False
    for line in cargo_lock_path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped == "[[package]]":
            in_pkg = True
            matched_name = False
            continue
        if not in_pkg:
            continue
        if stripped == f'name = "{package_name}"':
            matched_name = True
        elif matched_name and stripped.startswith('version = "'):
            return stripped.split('"', 2)[1]
    return "unavailable"


def collect_git_provenance(repo_root: Path) -> GitProvenance:
    """Inspect real Git commit, branch, dirty flag, and working-tree diff SHA-256."""
    commit = _run_text_command(["git", "rev-parse", "HEAD"], cwd=repo_root)
    if commit is None:
        return GitProvenance(commit=None, branch=None, dirty=None, diff_sha256=None)

    branch = _run_text_command(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root)
    status_bytes = _run_bytes_command(["git", "status", "--porcelain=v1"], cwd=repo_root)
    if status_bytes is None:
        return GitProvenance(commit=commit, branch=branch, dirty=None, diff_sha256=None)

    dirty = len(status_bytes.strip()) > 0
    if not dirty:
        return GitProvenance(commit=commit, branch=branch, dirty=False, diff_sha256=None)

    diff_bytes = _run_bytes_command(["git", "diff", "HEAD"], cwd=repo_root) or b""
    combined = status_bytes + b"\n---GIT_DIFF_HEAD---\n" + diff_bytes
    return GitProvenance(
        commit=commit,
        branch=branch,
        dirty=True,
        diff_sha256=sha256_bytes(combined),
    )


def collect_environment_provenance(
    repo_root: Path,
    *,
    compiler_profile: str = "dev_editable",
) -> EnvironmentProvenance:
    """Capture actual OS, CPU, RAM, toolchain, lockfile, and native bridge provenance."""
    cpu_model, physical_cores = _read_linux_cpu_info()
    ram_total = _read_linux_ram_bytes()
    cargo_lock = repo_root / "Cargo.lock"
    uv_lock = repo_root / "uv.lock"

    uv_ver = _run_text_command(["uv", "--version"], cwd=repo_root) or "unavailable"
    rustc_ver = _run_text_command(["rustc", "--version"], cwd=repo_root) or "unavailable"
    cargo_ver = _run_text_command(["cargo", "--version"], cwd=repo_root) or "unavailable"
    pyo3_ver = _extract_cargo_lock_package_version(cargo_lock, "pyo3")
    maturin_ver = _installed_package_version("maturin")
    bridge_info = get_native_bridge_info()

    return EnvironmentProvenance(
        os_name=platform.system() or "unavailable",
        os_release=platform.release() or "unavailable",
        kernel_version=platform.version() or "unavailable",
        architecture=platform.machine() or "unavailable",
        cpu_model=cpu_model,
        physical_core_count=physical_cores,
        logical_core_count=os.cpu_count(),
        ram_total_bytes=ram_total,
        python_version=platform.python_version(),
        uv_version=uv_ver,
        rustc_version=rustc_ver,
        cargo_version=cargo_ver,
        pyo3_version=pyo3_ver,
        maturin_version=maturin_ver,
        cargo_lock_sha256=sha256_file(cargo_lock) if cargo_lock.is_file() else None,
        uv_lock_sha256=sha256_file(uv_lock) if uv_lock.is_file() else None,
        compiler_profile=compiler_profile,
        native_extension_id=(
            f"{bridge_info['bridge_info']} (mod v{bridge_info['module_version']})"
        ),
    )
