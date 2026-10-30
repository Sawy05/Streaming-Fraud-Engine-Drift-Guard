"""Pytest configuration and compatibility hooks for pytest-benchmark on pytest >= 9.0."""

from __future__ import annotations

import pytest


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Restore `_nodeid` alias removed in pytest 9 so `pytest-benchmark` fixture initializes cleanly."""
    if not hasattr(item, "_nodeid"):
        item._nodeid = item.nodeid  # type: ignore[attr-defined]
