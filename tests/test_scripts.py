"""Every command-line script must at least import and print its help."""
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted((ROOT / "scripts").rglob("*.py"))


def test_scripts_are_found():
    assert len(SCRIPTS) >= 12


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: str(path.relative_to(ROOT)))
def test_script_help(script):
    result = subprocess.run(
        [sys.executable, str(script), "--help"], cwd=ROOT, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert "usage" in result.stdout.lower()
