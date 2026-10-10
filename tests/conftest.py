"""Shared fixtures. All tests are offline and use small synthetic data."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from pitwall_ai.data.cleaning import clean_laps
from pitwall_ai.data.demo_data import generate_demo_laps

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def demo_clean():
    """Cleaned synthetic laps (6720 raw rows -> ~6000 eligible)."""
    cleaned, _ = clean_laps(generate_demo_laps())
    return cleaned


def load_script(name: str):
    """Import ``scripts/<name>.py`` as a module (scripts is not a package)."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
