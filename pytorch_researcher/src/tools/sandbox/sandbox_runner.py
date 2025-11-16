"""
Sandbox runner and harness utilities.

This module provides a safe-ish subprocess-based harness to import and execute
LLM-generated model code in an isolated Python subprocess. It is intended for
quick smoke tests: instantiate the model class, run a single forward pass on a
dummy tensor, and print a small JSON summary to stdout for the orchestrator to
consume.

Design goals:
- Do not import generated code in-process; run in a separate Python process.
- Limit runtime with a timeout.
- Capture stdout/stderr and return structured result for logging and registry.
- Keep the harness simple and focused on smoke checks (existence, instantiation,
  forward pass, param counts, output shape).
- Allow tests to run without a GPU by default (CPU-only).

Usage (example):
    from pytorch_researcher.src.tools.sandbox.sandbox_runner import run_sandboxed_harness
    res = run_sandboxed_harness(
        model_path="src/models/generated.py",
        class_name="AssembledModel",
        input_size=(1, 3, 32, 32),
        timeout=60,
    )
    print(res)

Returns:
    dict with keys:
      - success: bool
      - returncode: int or None (if timed out)
      - stdout: str
      - stderr: str
      - duration: float seconds
      - parsed: dict or None (parsed JSON from harness stdout when available)
      - error: str when success is False and no parsed info is present
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# A small, airtight harness template that will be written to a temporary .py
# file and executed in a subprocess. The harness reads command-line args:
#   argv[1] = model_path (path to .py file)
#   argv[2] = class_name (model class to instantiate; optional, may be "AssembledModel")
#   argv[3] = input_size JSON (e.g., "[1,3,32,32]")
#
# The harness prints a single JSON object to stdout describing:
#   {"ok": True, "total_params": ..., "trainable_params": ..., "output_shape": (...)}
#
# On failure it prints JSON {"ok": False, "error": "<msg>"} and exits non-zero.
_HARNESS_TEMPLATE = r"""
# Auto-generated harness for sandboxed model validation
import json, sys, traceback
from pathlib import Path

def _emit(obj):
    try:
        print(json.dumps(obj))
    except Exception:
        # Fallback to a minimal string if JSON serialization fails
        print(json.dumps({"ok": False, "error": "serialization failure"}))

def main():
    try:
        if len(sys.argv) < 2:
            _emit({"ok": False, "error": "missing model_path argument"})
            sys.exit(2)
        model_path = sys.argv[1]
        class_name = sys.argv[2] if len(sys.argv) >= 3 and sys.argv[2] else None
        input_size_json = sys.argv[3] if len(sys.argv) >= 4 else None

        # Basic validation
        p = Path(model_path)
        if not p.exists():
            _emit({"ok": False, "error": f"model file not found: {model_path}"})
            sys.exit(3)

        # Parse input size if provided
        input_size = None
        if input_size_json:
            try:
                input_size = tuple(json.loads(input_size_json))
            except Exception as e:
                _emit({"ok": False, "error": f"invalid input_size JSON: {e}"})
                sys.exit(4)

        # Import torch lazily; report friendly error if not available
        try:
            import torch
            import torch.nn as nn
        except Exception as e:
            _emit({"ok": False, "error": f\"PyTorch import failed: {e}\"})
            sys.exit(5)

        # Import module from path
        import importlib.util
        spec = importlib.util.spec_from_file_location("sandbox_model_module", str(p))
        if spec is None or spec.loader is None:
            _emit({"ok": False, "error": "could not create module spec"})
            sys.exit(6)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            _emit({"ok": False, "error": f\"Failed to execute module: {e}\\n{traceback.format_exc()}\"})
            sys.exit(7)

        # Locate class
        cls = None
        if class_name:
            cls = getattr(module, class_name, None)
            if cls is None:
                # try AssembledModel fallback
                cls = getattr(module, "AssembledModel", None)
        else:
            cls = getattr(module, "AssembledModel", None)
            if cls is None:
                # try find first nn.Module subclass defined in module
                import inspect
                for _, obj in inspect.getmembers(module, inspect.isclass):
                    if getattr(obj, "__module__", None) == module.__name__:
                        try:
                            if issubclass(obj, nn.Module):
                                cls = obj
                                break
                        except Exception:
                            continue

        if cls is None:
            _emit({"ok": False, "error": "could not locate nn.Module subclass in module"})
            sys.exit(8)

        # Instantiate
        try:
            inst = cls()
        except Exception as e:
            _emit({"ok": False, "error": f\"Failed to instantiate {cls.__name__}: {e}\\n{traceback.format_exc()}\"})
            sys.exit(9)

        # Move to cpu
        try:
            inst.to("cpu")
        except Exception:
            pass

        # Count params
        total = 0
        trainable = 0
        try:
            for p in inst.parameters():
                n = int(p.numel())
                total += n
                if p.requires_grad:
                    trainable += n
        except Exception:
            # If parameters() fails, still continue to try forward pass
            pass

        # Build dummy input and run forward if input_size is provided
        output_shape = None
        if input_size is not None:
            try:
                inst.eval()
                with torch.no_grad():
                    dummy = torch.randn(input_size, device="cpu")
                    out = inst(dummy)
                # normalize output shape information
                import torch as _torch
                if isinstance(out, _torch.Tensor):
                    output_shape = tuple(out.shape)
                elif isinstance(out, (list, tuple)):
                    shapes = []
                    for o in out:
                        if isinstance(o, _torch.Tensor):
                            shapes.append(tuple(o.shape))
                        else:
                            shapes.append(type(o).__name__)
                    output_shape = tuple(shapes)
                else:
                    output_shape = (type(out).__name__,)
            except Exception as e:
                _emit({"ok": False, "error": f\"Forward pass failed: {e}\\n{traceback.format_exc()}\"})
                sys.exit(10)

        # Success: emit structured summary
        _emit({
            "ok": True,
            "class_name": cls.__name__,
            "total_params": total,
            "trainable_params": trainable,
            "output_shape": output_shape,
        })
        sys.exit(0)

    except SystemExit:
        raise
    except Exception as e:
        _emit({"ok": False, "error": f\"Unhandled exception in harness: {e}\\n{traceback.format_exc()}\"})
        sys.exit(99)

if __name__ == '__main__':
    main()
"""


def _write_harness_temp() -> Tuple[Path, bool]:
    """
    Write the harness template to a temporary file and return its Path.

    Returns:
        (path, delete_on_exit) - path of the file, and whether the caller should
        attempt to delete it after use (True).
    """
    tf = tempfile.NamedTemporaryFile(
        delete=False, suffix=".py", prefix="sandbox_harness_"
    )
    tf_path = Path(tf.name)
    try:
        tf.write(_HARNESS_TEMPLATE.encode("utf-8"))
        tf.flush()
    finally:
        tf.close()
    return tf_path, True


def run_sandboxed_harness(
    model_path: str,
    class_name: Optional[str] = None,
    input_size: Optional[Sequence[int]] = (1, 3, 32, 32),
    timeout: int = 60,
    python_executable: Optional[str] = None,
    extra_env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Run the harness in a subprocess to validate the generated model.

    Args:
        model_path: path to the generated Python source file for the model.
        class_name: optional class name to instantiate (falls back to AssembledModel).
        input_size: sequence describing input tensor shape including batch dim.
        timeout: time limit in seconds for the subprocess.
        python_executable: path to Python interpreter to use (defaults to sys.executable).
        extra_env: optional environment variables to set for the subprocess (merged with a minimal safe env).

    Returns:
        A dict containing keys:
          - success (bool)
          - returncode (int or None if timed out)
          - stdout (str)
          - stderr (str)
          - duration (float)
          - parsed (dict or None)
          - error (str when applicable)
    """
    start = time.time()
    python_executable = python_executable or sys.executable

    harness_path, should_delete = _write_harness_temp()

    # Build command-line args: model_path, class_name, input_size(json)
    args = [python_executable, str(harness_path), str(model_path)]
    if class_name:
        args.append(str(class_name))
    else:
        args.append("")  # empty placeholder
    # Serialize input_size
    try:
        input_json = json.dumps(list(input_size)) if input_size is not None else ""
    except Exception:
        input_json = ""
    args.append(input_json)

    # Prepare a minimal, controlled environment for the subprocess.
    # Keep PYTHONPATH so local package imports work, but avoid leaking secrets.
    safe_env = {}
    # preserve essential PATH for Python exec to work
    if "PATH" in os.environ:
        safe_env["PATH"] = os.environ["PATH"]
    # preserve PYTHONPATH if present so the project modules are importable
    if "PYTHONPATH" in os.environ:
        safe_env["PYTHONPATH"] = os.environ["PYTHONPATH"]
    # Add LANG to avoid localization issues
    safe_env["LANG"] = os.environ.get("LANG", "en_US.UTF-8")

    # Merge any extra env supplied by caller (explicit)
    if extra_env:
        safe_env.update(extra_env)

    # Execute subprocess
    try:
        proc = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=safe_env,
            timeout=timeout,
            text=True,
        )
        duration = time.time() - start
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        parsed = None
        # Attempt to parse stdout as JSON - harness prints exactly one JSON line on success/failure
        try:
            # strip leading/trailing whitespace and parse last non-empty line
            last_line = None
            for line in reversed([ln for ln in stdout.splitlines() if ln.strip()]):
                last_line = line
                break
            if last_line:
                parsed = json.loads(last_line)
        except Exception:
            parsed = None

        result: Dict[str, Any] = {
            "success": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "duration": duration,
            "parsed": parsed,
        }
        if (
            not result["success"]
            and parsed
            and isinstance(parsed, dict)
            and parsed.get("error")
        ):
            result["error"] = parsed.get("error")
        elif not result["success"] and not parsed:
            result["error"] = stderr.strip() or "subprocess failed without JSON output"
        return result

    except subprocess.TimeoutExpired as e:
        duration = time.time() - start
        # Kill is implicit when TimeoutExpired; compile partial output if any
        stdout = (
            e.stdout.decode("utf-8")
            if isinstance(e.stdout, bytes)
            else (e.stdout or "")
        )
        stderr = (
            e.stderr.decode("utf-8")
            if isinstance(e.stderr, bytes)
            else (e.stderr or "")
        )
        return {
            "success": False,
            "returncode": None,
            "stdout": stdout,
            "stderr": stderr,
            "duration": duration,
            "parsed": None,
            "error": f"timeout after {timeout}s",
        }
    except Exception as exc:
        duration = time.time() - start
        logger.exception("Unexpected error running sandbox harness: %s", exc)
        return {
            "success": False,
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "duration": duration,
            "parsed": None,
            "error": f"unexpected error: {exc}",
        }
    finally:
        # Best-effort cleanup of the harness script
        try:
            if should_delete and harness_path.exists():
                harness_path.unlink()
        except Exception:
            # not fatal; leave file for inspection if deletion fails
            pass
