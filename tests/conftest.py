import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data._client import load_config  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config(ROOT / "config.yaml")


@pytest.fixture(scope="session")
def root():
    return ROOT
