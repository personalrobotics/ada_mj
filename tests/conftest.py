"""Each test's evidence directory."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def artifacts(request: pytest.FixtureRequest) -> Path:
    """Where a test writes its evidence: ``$ADA_ARTIFACTS`` (default ``artifacts/``)
    / <test name>."""
    base = Path(os.environ.get("ADA_ARTIFACTS", ROOT / "artifacts"))
    out = base / str(request.node.name)
    out.mkdir(parents=True, exist_ok=True)
    return out
