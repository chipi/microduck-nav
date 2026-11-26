import hashlib
import os
from pathlib import Path

import pytest

WALKER = Path(os.environ.get("MICRODUCK_WALKER_ONNX", "~/.cache/duck-sim/policies/current/velstand.onnx")).expanduser()
WALKER_SHA_PREFIX = "ce894d30"


@pytest.fixture(scope="session")
def walker_path():
    if not WALKER.exists():
        pytest.skip(f"walker not found at {WALKER}")
    digest = hashlib.sha256(WALKER.read_bytes()).hexdigest()
    assert digest.startswith(WALKER_SHA_PREFIX), f"unexpected walker {digest}"
    return WALKER
