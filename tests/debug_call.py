#!/usr/bin/env python3
"""
Debug script to import `python_run` from the project package and call it.

This script:
- Ensures the project root and top-level `src/` directory are on `sys.path`.
- Imports `python_run` from `pytorch_researcher.src.utils`.
- Creates a temporary Python script that echoes its argv and prints a greeting.
- Calls `python_run(script_path, args=['TestUser'])` and prints the captured result.
- For comparison, also runs the script using subprocess with 'python' and with sys.executable.

Save this file under: pytorch-researcher/tests/debug_call.py
Run it from the project environment to observe differences in behavior.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

# Ensure project root and top-level src are on sys.path so imports resolve
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]  # <project_root>/tests/.. -> project_root
SRC_DIR = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Try to import python_run from the package shim we provided earlier
python_run = None
import_error = None
try:
    from pytorch_researcher.src.utils import python_run as _py_run  # type: ignore

    python_run = _py_run
except Exception as e:
    import_error = e

SCRIPT_CONTENT = textwrap.dedent(
    """\
    import sys
    import json
    # Print argv as JSON for reliable parsing
    print(json.dumps({"argv": sys.argv}), flush=True)
    if len(sys.argv) > 1:
        name = sys.argv[1]
    else:
        name = "World"
    print(f"Hello, {name}!", flush=True)
    """
)


def write_temp_script(tmpdir: Path) -> Path:
    script_path = tmpdir / "debug_args_script.py"
    script_path.write_text(SCRIPT_CONTENT, encoding="utf-8")
    script_path.chmod(0o755)
    return script_path


def pretty_print_result(label: str, result: dict | subprocess.CompletedProcess):
    border = "-" * 70
    print(border)
    print(f"[{label}]")
    if isinstance(result, dict):
        stdout = result.get("stdout", "")
        stderr = result.get("stderr", "")
        exit_code = result.get("exit_code", None)
    else:
        stdout = result.stdout
        stderr = result.stderr
        exit_code = result.returncode

    print("exit_code:", exit_code)
    print("stdout:")
    if stdout:
        print(stdout.rstrip())
    else:
        print("<EMPTY STDOUT>")
    print("stderr:")
    if stderr:
        print(stderr.rstrip())
    else:
        print("<EMPTY STDERR>")

    # Try to parse the first stdout line as JSON to show argv
    first_line = (stdout or "").splitlines()[0] if (stdout or "").splitlines() else ""
    try:
        parsed = json.loads(first_line)
        print("parsed argv from first stdout line:", parsed.get("argv"))
    except Exception:
        print("could not parse argv JSON from first stdout line")
    print(border)
    print()


def main():
    print("Debugging python_run import and invocation")
    print("Project root:", PROJECT_ROOT)
    print("Top-level src dir:", SRC_DIR)
    print("sys.executable:", sys.executable)
    print()

    if import_error:
        print("WARNING: failed to import `python_run` from package shim:")
        print(repr(import_error))
        print("Continuing with raw subprocess runs for comparison.\n")
    else:
        print("Successfully imported `python_run` from package shim.\n")

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        script_path = write_temp_script(tmpdir)
        print("Created temporary script at:", script_path)
        print()

        # 1) Call python_run from package, if available
        if python_run is not None:
            try:
                print("Calling python_run(script_path, args=['TestUser'])")
                res = python_run(str(script_path), args=["TestUser"])
                pretty_print_result("python_run (args=['TestUser'])", res)
            except Exception as e:
                print("python_run raised an exception:", repr(e))
                print()

            try:
                print("Calling python_run(script_path, args=[])  # no args")
                res = python_run(str(script_path), args=[])
                pretty_print_result("python_run (args=[])", res)
            except Exception as e:
                print("python_run raised an exception:", repr(e))
                print()

        # 2) subprocess.run(['python', script_path, 'TestUser'])
        try:
            print("Calling subprocess.run(['python', script_path, 'TestUser'])")
            cp = subprocess.run(
                ["python", str(script_path), "TestUser"],
                capture_output=True,
                text=True,
                check=False,
            )
            pretty_print_result("subprocess ['python' ...]", cp)
        except Exception as e:
            print("subprocess ['python' ...] raised:", repr(e))

        # 3) subprocess.run([sys.executable, script_path, 'TestUser'])
        try:
            print("Calling subprocess.run([sys.executable, script_path, 'TestUser'])")
            cp = subprocess.run(
                [sys.executable, str(script_path), "TestUser"],
                capture_output=True,
                text=True,
                check=False,
            )
            pretty_print_result("subprocess [sys.executable ...]", cp)
        except Exception as e:
            print("subprocess [sys.executable ...] raised:", repr(e))

        # 4) subprocess.run(['/usr/bin/env', 'python', script_path, 'TestUser'])
        try:
            env_invoker = ["/usr/bin/env", "python", str(script_path), "TestUser"]
            print(
                "Calling subprocess.run(['/usr/bin/env', 'python', script_path, 'TestUser'])"
            )
            cp = subprocess.run(
                env_invoker, capture_output=True, text=True, check=False
            )
            pretty_print_result("subprocess ['/usr/bin/env python' ...]", cp)
        except Exception as e:
            print("subprocess ['/usr/bin/env python' ...] raised:", repr(e))

    print("Debug run complete.")


if __name__ == "__main__":
    main()
