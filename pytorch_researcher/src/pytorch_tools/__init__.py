"""pytorch_tools package

This package will contain PyTorch-specific tool implementations used by the
PyTorch ML Research Agent MVP. Submodules expected to live under this package
include (but are not limited to):

- model_assembler     : Generate PyTorch `nn.Module` code from structured configs.
- model_summary       : Analyze PyTorch model source and produce structured summaries.
- quick_evaluator     : Run lightweight training/evaluation loops for rapid feedback.

The package initializer aims to be resilient to partial implementations during
development: imports are attempted but failures are caught so tests and other
parts of the system can import the package without requiring all tools to be
present.

Provides:
- __all__ for discoverability
- __version__ string
- `available_tools()` helper to list tools that successfully imported
- `get_tool(name)` helper to access an imported tool module
"""

from __future__ import annotations

import importlib
import logging
from typing import Dict, List, Optional

__all__ = [
    "__version__",
    "available_tools",
    "get_tool",
    "llm",
    "model_assembler",
    "model_summary",
    "quick_evaluator",
]

__version__ = "0.1.0"

_logger = logging.getLogger(__name__)

# Attempt to import known submodules. If a submodule is not present yet,
# leave the variable as None so callers can handle absence gracefully.
model_assembler = None
model_summary = None
quick_evaluator = None
llm = None

_import_map: dict[str, object | None] = {
    "model_assembler": None,
    "model_summary": None,
    "quick_evaluator": None,
    "llm": None,
}

for _name in list(_import_map.keys()):
    try:
        # Use relative import; submodules may not exist yet during early development.
        _mod = importlib.import_module(f"{__name__}.{_name}")  # type: ignore[arg-type]
        globals()[_name] = _mod
        _import_map[_name] = _mod
    except Exception as _exc:  # pragma: no cover - defensive import
        # Don't raise here; the package should be importable even if some tools are missing.
        _import_map[_name] = None
        _logger.debug("pytorch_tools: could not import %s: %s", _name, _exc)


def available_tools() -> list[str]:
    """Return a list of tool submodule names that were successfully imported.

    Examples
    --------
    >>> from pytorch_researcher.src.pytorch_tools import available_tools
    >>> available_tools()
    ['model_assembler', 'model_summary']

    """
    return [name for name, mod in _import_map.items() if mod is not None]


def get_tool(name: str):
    """Retrieve the imported tool module by name.

    Args:
        name: One of the known submodule names (e.g., 'model_assembler').

    Returns:
        The imported module object.

    Raises:
        KeyError: If the name is unknown or the tool is not available.

    """
    if name not in _import_map:
        raise KeyError(f"Unknown pytorch_tools module: {name!r}")
    mod = _import_map[name]
    if mod is None:
        raise KeyError(
            f"Requested pytorch_tools module {name!r} is not available (not implemented/importable)."
        )
    return mod
