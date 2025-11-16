#!/usr/bin/env python3
"""
Debug script for `python_run` behavior with arguments.

Creates a temporary Python script that inspects sys.argv and prints a greeting using
the first argument (if present). Then runs several variants:

- Calls the project's `python_run` helper (imported from the package shim).
- Calls subprocess.run with "python" (which is what python_run uses).
- Calls subprocess.run with sys.executable (explicit current interpreter).
- Calls subprocess.run with "/usr/bin/env python" style invocation.

This helps surface differences in environment or invocation that could explain
why `python_run(..., args=["..."])` produced empty stdout in a test run.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

# Import the function under test. Depending on test layout, this should resolve to
# the utils module we implemented earlier.
try:
    from pytorch_researcher.src.utils import python_run
except Exception as e:
    # If import fails, print the error and continue; the script will still run
    # the subprocess comparisons using the system Python.
    python_run = None
    import_error = e
else:
    import_error = None


SCRIPT_CONTENT = textwrap.dedent(
    """
    import sys
    import json
    # Print argv as JSON so it's easy to parse even if buffering behaves oddly.
    print(json.dumps({"argv": sys.argv}), flush=True)
    # Print a friendly greeting if an argument was provided.
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
    # Make it executable (not required for python invocation, but useful)
    script_path.chmod(0o755)
    return script_path


def run_and_print(label: str, proc_result: dict | subprocess.CompletedProcess):
    sep = "-" * 60
    print(sep)
    print(f"[{label}]")
    if isinstance(proc_result, dict):
        # expected keys: stdout, stderr, exit_code
        stdout = proc_result.get("stdout", "")
        stderr = proc_result.get("stderr", "")
        exit_code = proc_result.get("exit_code", None)
    else:
        # CompletedProcess
        stdout = proc_result.stdout
        stderr = proc_result.stderr
        exit_code = proc_result.returncode

    print("exit_code:", exit_code)
    print("stdout:")
    print(stdout if stdout else "<EMPTY STDOUT>")
    print("stderr:")
    print(stderr if stderr else "<EMPTY STDERR>")
    # Try to parse first line as JSON showing argv
    first_line = (stdout or "").splitlines()[0] if (stdout or "").splitlines() else ""
    try:
        parsed = json.loads(first_line)
        print("parsed argv from stdout JSON:", parsed.get("argv"))
    except Exception:
        print("could not parse argv JSON from stdout first line")
    print(sep)
    print()


def main():
    print("Debugging python_run argument behavior")
    print("System interpreter (sys.executable):", sys.executable)
    if import_error:
        print("WARNING: failed to import `python_run` from package shim:", import_error)
        print("Continuing with raw subprocess tests only.\n")

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        script_path = write_temp_script(tmpdir)

        # 1) If available, call the project's python_run with args list
        if python_run is not None:
            try:
                print("Calling python_run(script_path, args=['TestUser'])")
                res = python_run(str(script_path), args=["TestUser"])
                run_and_print("python_run (args=['TestUser'])", res)
            except Exception as e:
                print("python_run raised an exception:", repr(e))
                print()

            try:
                print("Calling python_run(script_path, args=[])  # no args")
                res = python_run(str(script_path), args=[])
                run_and_print("python_run (args=[])", res)
            except Exception as e:
                print("python_run raised an exception:", repr(e))
                print()

            try:
                print(
                    "Calling python_run(script_path, args=['One','Two'])  # multiple args"
                )
                res = python_run(str(script_path), args=["One", "Two"])
                run_and_print("python_run (args=['One','Two'])", res)
            except Exception as e:
                print("python_run raised an exception:", repr(e))
                print()

            # Also try passing args as a single string to see behavior (even though API expects list)
            try:
                print(
                    "Calling python_run(script_path, args='TestUser')  # string instead of list"
                )
                res = python_run(
                    str(script_path), args="TestUser"
                )  # intentionally incorrect type
                run_and_print("python_run (args='TestUser')", res)
            except Exception as e:
                print("python_run (string-args) raised an exception:", repr(e))
                print()

        # 2) Call subprocess.run with "python" (like python_run does)
        try:
            print("Calling subprocess.run(['python', script_path, 'TestUser'])")
            cp = subprocess.run(
                ["python", str(script_path), "TestUser"],
                capture_output=True,
                text=True,
                check=False,
            )
            run_and_print("subprocess ['python' ...]", cp)
        except Exception as e:
            print("subprocess ['python' ...] raised:", repr(e))

        # 3) Call subprocess.run with sys.executable
        try:
            print("Calling subprocess.run([sys.executable, script_path, 'TestUser'])")
            cp = subprocess.run(
                [sys.executable, str(script_path), "TestUser"],
                capture_output=True,
                text=True,
                check=False,
            )
            run_and_print("subprocess [sys.executable ...]", cp)
        except Exception as e:
            print("subprocess [sys.executable ...] raised:", repr(e))

        # 4) Call subprocess.run using env lookup (/usr/bin/env python)
        try:
            print(
                "Calling subprocess.run(['/usr/bin/env', 'python', script_path, 'TestUser'])"
            )
            cp = subprocess.run(
                ["/usr/bin/env", "python", str(script_path), "TestUser"],
                capture_output=True,
                text=True,
                check=False,
            )
            run_and_print("subprocess ['/usr/bin/env python' ...]", cp)
        except Exception as e:
            print("subprocess ['/usr/bin/env python' ...] raised:", repr(e))

        # 5) For completeness, run the script with no args via subprocess to compare
        try:
            print("Calling subprocess.run([sys.executable, script_path])  # no args")
            cp = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                check=False,
            )
            run_and_print("subprocess [no args]", cp)
        except Exception as e:
            print("subprocess [no args] raised:", repr(e))

    print("Debug run complete. Inspect outputs above to determine discrepancies.")


if __name__ == "__main__":
    main()
