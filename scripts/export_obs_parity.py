"""Writes tests/fixtures/obs_parity.json and obs_parity_deploy.json: constructed states and the 1343-vector the training code
builds from each, for the full task (the walker's 50 Hz step) and for the deploy task (a policy tick of five steps, 0.1 s).

robotics-lab's duck navigator copies these files (robots/duck/tests/fixtures/) and checks its own builder against them.
Run: uv run python scripts/export_obs_parity.py
"""

import json
from pathlib import Path

from microduck_nav.obs_parity import DEPLOY_TASK, TASK, build_fixture

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
FILES = {TASK: "obs_parity.json", DEPLOY_TASK: "obs_parity_deploy.json"}

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for task, name in FILES.items():
        fixture = build_fixture(task)
        (OUT / name).write_text(json.dumps(fixture, indent=1) + "\n")
        print(f"wrote {OUT / name}: {len(fixture['states'])} states, step_s {fixture['step_s']}")
