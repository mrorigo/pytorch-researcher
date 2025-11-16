import os
import sys
from pathlib import Path

# Ensure the project root (the directory containing this tests/ folder) is on sys.path
# so that imports like `import pytorch_researcher.src.utils` work during tests.
#
# This file is placed at: <project_root>/tests/conftest.py
# We add <project_root> to sys.path so the `pytorch_researcher` package (under
# <project_root>/pytorch_researcher) can be imported.
project_root = Path(__file__).resolve().parents[1]
project_root_str = str(project_root)

if project_root_str not in sys.path:
    # Insert at the front so test-local packages take precedence over any installed packages.
    sys.path.insert(0, project_root_str)

# Optional: make it easy to locate the top-level src directory from tests if needed.
# e.g., project_root / "src"
SRC_DIR = project_root / "src"
if SRC_DIR.exists():
    SRC_DIR_STR = str(SRC_DIR)
    if SRC_DIR_STR not in sys.path:
        # Add the sibling top-level `src` to sys.path as well, to support direct imports
        # if tests or utilities import modules directly from the top-level src dir.
        sys.path.insert(0, SRC_DIR_STR)
