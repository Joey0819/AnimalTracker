"""Automatic initialization utilities for no-click tracking workflows."""

from densetrack.initialization.auto_seed import (
    AutoInitConfig,
    AutoInitResult,
    create_manual_initialization_points,
    find_auto_initialization_points,
    write_auto_init_outputs,
)

__all__ = [
    "AutoInitConfig",
    "AutoInitResult",
    "create_manual_initialization_points",
    "find_auto_initialization_points",
    "write_auto_init_outputs",
]
