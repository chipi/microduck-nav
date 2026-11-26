"""M1 dry run on CPU: the DGX command with small numbers, then the replay that decides "done"."""

import json
import re
from pathlib import Path

import pytest

from microduck_nav import replay, train_check

pytestmark = pytest.mark.slow

DOC = Path(__file__).resolve().parent.parent / "docs" / "M1.md"
TASK = "Mjlab-Nav-OneWall-MicroDuck"


def test_the_m1_doc_carries_the_commands_this_repo_builds():
    text = DOC.read_text()
    smoke = " ".join(train_check.train_command(TASK, 64, 20, 10, cpu=False)[1:])
    full = " ".join(train_check.train_command(TASK, 4096, 200, 50, cpu=False)[1:])
    assert f"train {smoke}" in text and f"train {full}" in text
    assert "--agent.logger tensorboard" in text and "--agent.save-interval 50" in text


@pytest.fixture(scope="module")
def run(tmp_path_factory, walker_path):
    return train_check.run(TASK, 16, 5, 5, tmp_path_factory.mktemp("m1"))


def test_the_command_trains_saves_and_exports(run):
    assert run["returncode"] == 0, run["output"][-3000:]
    assert train_check.all_finite(run["scalars"])
    assert {"model_0.pt", "model_4.pt"} <= set(run["checkpoints"])
    assert train_check.check_onnx(run["log_dir"] / "model_4.onnx")["output"] == (1, 3)


def test_the_replay_runs_on_the_checkpoint_and_reports_a_verdict(run, tmp_path, capsys):
    checkpoint = run["log_dir"] / "model_4.pt"
    out = tmp_path / "replay.json"
    code = replay.main([str(checkpoint), "--task", TASK, "--num-envs", "4", "--steps", "150", "--json", str(out)])
    data = json.loads(out.read_text())
    assert code in (0, 1)
    assert data["summary"]["envs"] == 4 and len(data["envs"]) == 4
    assert all(e["steps"] > 0 for e in data["envs"])
    assert all(e["initial_clearance"] > 0 for e in data["envs"])
    with capsys.disabled():
        print("\nM1-dry " + json.dumps(data["summary"]))
