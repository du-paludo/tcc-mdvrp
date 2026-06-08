"""Wrapper to run the standalone simulation log animator from src/tools."""

from __future__ import annotations

import subprocess
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]


def _venv_python_candidates() -> list[Path]:
    return [
        REPO_ROOT / "venv" / "Scripts" / "python.exe",
        REPO_ROOT / ".venv" / "Scripts" / "python.exe",
    ]


def _same_executable(path_a: Path, path_b: Path) -> bool:
    return str(path_a.resolve()).lower() == str(path_b.resolve()).lower()


def _run_on_venv_python_if_needed() -> int | None:
    current_exe = Path(sys.executable)

    for candidate in _venv_python_candidates():
        if not candidate.exists():
            continue
        if _same_executable(current_exe, candidate):
            return None

        completed = subprocess.run(
            [str(candidate), str(Path(__file__).resolve()), *sys.argv[1:]],
            check=False,
        )
        return int(completed.returncode)

    return None


def main() -> int:
    delegated_code = _run_on_venv_python_if_needed()
    if delegated_code is not None:
        return delegated_code

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from tools.animate_simulation_log import main as animator_main

    return int(animator_main())


if __name__ == "__main__":
    raise SystemExit(main())
