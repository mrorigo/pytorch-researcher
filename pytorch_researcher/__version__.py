"""
Version information for PyTorch ML Research Agent.

This module provides version information for the package.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Tuple


def _get_version_from_git() -> str | None:
    """Get version from git tags if available."""
    try:
        # Check if we're in a git repository
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            return None

        # Get the latest git tag
        result = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0"],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode == 0:
            version = result.stdout.strip()
            if version.startswith("v"):
                version = version[1:]
            return version

        # If no tags, get commit hash
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode == 0:
            commit = result.stdout.strip()
            return f"0.1.0+dev.{commit}"

    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    return None


def _get_version_from_file() -> str:
    """Get version from VERSION file if it exists."""
    version_file = Path(__file__).parent / "VERSION"
    if version_file.exists():
        return version_file.read_text().strip()
    return "0.1.0"


def get_version() -> str:
    """
    Get the current version of the package.

    Version priority:
    1. Environment variable PACKAGE_VERSION
    2. Git tags (if in git repository)
    3. VERSION file
    4. Default fallback version
    """
    # Check environment variable first
    env_version = os.environ.get("PACKAGE_VERSION")
    if env_version:
        return env_version

    # Try git-based versioning
    git_version = _get_version_from_git()
    if git_version:
        return git_version

    # Try VERSION file
    return _get_version_from_file()


# Public API
__version__ = get_version()


# Version components for programmatic access
def version_info() -> Tuple[str, str, str]:
    """
    Get version information as a tuple.

    Returns:
        (version, build_type, build_info)
        - version: Semantic version string
        - build_type: 'release', 'dev', or 'unknown'
        - build_info: Additional build information
    """
    version = __version__

    if "+dev." in version:
        return version, "dev", version.split("+dev.")[-1]
    elif "+" in version:
        return version, "custom", version.split("+")[-1]
    else:
        return version, "release", ""


# Package metadata
__author__ = "PyTorch ML Research Agent Team"
__email__ = "research@ml-agent.dev"
__license__ = "MIT"
__description__ = (
    "Autonomous PyTorch ML Research Agent with LLM-driven architectural exploration"
)
__url__ = "https://github.com/pytorch-researcher/pytorch-ml-research-agent"
__project_name__ = "pytorch-ml-research-agent"


# Development version check
def is_development_version() -> bool:
    """Check if this is a development version."""
    return "+dev." in __version__ or "+" in __version__


# Version compatibility
def check_version_compatibility(min_version: str) -> bool:
    """
    Check if current version meets minimum version requirement.

    Args:
        min_version: Minimum required version (e.g., "0.1.0")

    Returns:
        True if current version is compatible
    """
    try:
        current = tuple(map(int, __version__.split("+")[0].split(".")))
        required = tuple(map(int, min_version.split(".")))
        return current >= required
    except (ValueError, AttributeError):
        return False


if __name__ == "__main__":
    # Simple CLI for version information
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "--version":
            print(__version__)
        elif command == "--info":
            version, build_type, build_info = version_info()
            print(f"Version: {version}")
            print(f"Build Type: {build_type}")
            if build_info:
                print(f"Build Info: {build_info}")
        elif command == "--check":
            if len(sys.argv) > 2:
                min_ver = sys.argv[2]
                compatible = check_version_compatibility(min_ver)
                print(f"Compatible with {min_ver}: {compatible}")
            else:
                print("Usage: python __version__.py --check <min_version>")
        elif command == "--dev":
            print(f"Development version: {is_development_version()}")
        else:
            print("Unknown command. Available: --version, --info, --check, --dev")
    else:
        print(f"PyTorch ML Research Agent v{__version__}")
