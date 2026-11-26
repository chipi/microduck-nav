"""Writes tests/fixtures/obs_parity.json: constructed states and the 1343-vector the training code builds from each.

robotics-lab's duck navigator copies this file (robots/duck/tests/fixtures/) and checks its own builder against it.
Run: uv run python scripts/export_obs_parity.py
"""

import json
from pathlib import Path

from microduck_nav.obs_parity import build_fixture

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "obs_parity.json"

if __name__ == "__main__":
    fixture = build_fixture()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fixture, indent=1) + "\n")
    print(f"wrote {OUT}: {len(fixture['states'])} states")
