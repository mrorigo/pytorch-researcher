# pytorch-researcher/pytorch_researcher/src/pytorch_tools/model_summary.py
"""
pytorch_model_summary - Lightweight model introspection utility

This module provides helper functions to:
- load a Python module from a source file path
- locate and instantiate an nn.Module subclass
- produce a structured summary of the model using `torchinfo` when available,
  or using a fallback that inspects parameters and performs a single forward
  pass with a dummy tensor.

Functions
---------
- summarize_model_from_path(path, class_name, input_size=(1,3,32,32), init_kwargs=None, device='cpu')
    High-level function to produce the summary and parameter counts.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import logging
import sys
import types
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Optional LLM client abstractions (factored out for DRYness).
# Import is best-effort so environments without the llm module still work.
try:
    from pytorch_researcher.src.pytorch_tools.llm import BaseLLMClient, HTTPLLMClient
except Exception:
    BaseLLMClient = None  # type: ignore
    HTTPLLMClient = None  # type: ignore

_logger = logging.getLogger(__name__)

# Optional imports
try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - environment dependent
    torch = None  # type: ignore
    nn = None  # type: ignore

# torchinfo (formerly torchsummaryX/torchsummary) optional
try:
    # `torchinfo.summary` is the modern helper (installable as `torchinfo`)
    from torchinfo import summary as torchinfo_summary  # type: ignore
except Exception:  # pragma: no cover - optional
    torchinfo_summary = None  # type: ignore


def _load_module_from_path(
    path: str, module_name: Optional[str] = None
) -> types.ModuleType:
    """
    Dynamically load a Python module from a file path.

    Parameters
    ----------
    path:
        Path to the .py file to import as a module.
    module_name:
        Optional name to assign to the loaded module. If not provided,
        a deterministic name derived from the absolute path will be used.

    Returns
    -------
    module:
        The loaded Python module object.

    Raises
    ------
    FileNotFoundError if the file doesn't exist.
    RuntimeError if module cannot be loaded.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Module file not found: {path}")

    if module_name is None:
        # Create a reproducible module name from path
        safe_name = "model_src_" + "_".join(p.resolve().parts[-4:])
        module_name = safe_name

    spec = importlib.util.spec_from_file_location(module_name, str(p))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not create module spec for {path}")

    module = importlib.util.module_from_spec(spec)
    try:
        # Execute the module in its own namespace
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Failed to execute module {path}: {e}") from e

    return module


def _find_model_class(
    module: types.ModuleType, class_name: Optional[str] = None
) -> type:
    """
    Find an nn.Module subclass in the module.

    If `class_name` is provided, it tries to retrieve that class specifically.
    Otherwise, it searches for the first subclass of `torch.nn.Module`.

    Returns the class object.

    Raises KeyError if not found.
    """
    if torch is None or nn is None:
        raise RuntimeError(
            "PyTorch is required for model inspection but is not available."
        )

    if class_name:
        cls = getattr(module, class_name, None)
        if cls is None:
            raise KeyError(
                f"Class named {class_name!r} not found in module {module.__name__}"
            )
        if not inspect.isclass(cls) or not issubclass(cls, nn.Module):
            raise KeyError(f"Found {class_name!r} but it is not an nn.Module subclass")
        return cls

    # Search for first class that is subclass of nn.Module
    for _, obj in inspect.getmembers(module, inspect.isclass):
        # Only consider classes defined in the module to avoid imported classes
        if getattr(obj, "__module__", None) != module.__name__:
            continue
        if issubclass(obj, nn.Module):
            return obj

    raise KeyError("No nn.Module subclass found in the provided module")


def _instantiate_model(
    cls: type, init_kwargs: Optional[Dict[str, Any]] = None, device: str = "cpu"
) -> nn.Module:
    """
    Instantiate model class with init_kwargs (if any) and move to device.

    Raises TypeError/RuntimeError if instantiation fails.
    """
    if init_kwargs is None:
        init_kwargs = {}
    try:
        model = cls(**init_kwargs)
    except Exception as e:
        # Provide a helpful error if simple instantiation failed
        raise RuntimeError(
            f"Failed to instantiate model class {cls.__name__}: {e}"
        ) from e

    if torch is not None:
        try:
            model.to(device)
        except Exception:
            # Non-fatal: some modules may not support .to(device) at construction
            pass
    return model


def _count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Return (total_params, trainable_params) for the model.
    """
    total = 0
    trainable = 0
    for p in model.parameters():
        num = int(p.numel())
        total += num
        if p.requires_grad:
            trainable += num
    return total, trainable


def _run_dummy_forward(
    model: nn.Module, input_size: Tuple[int, ...], device: str = "cpu"
) -> Dict[str, Any]:
    """
    Run a single forward pass with a dummy tensor to validate shapes and gather a brief
    operational summary. Returns a dict containing output shape and any exception info.

    Note: This executes model.forward once under torch.no_grad(); it is only used for
    verification in environments where torchinfo is not available.
    """
    if torch is None:
        raise RuntimeError(
            "PyTorch is required to run dummy forward but is not available."
        )

    try:
        model.eval()
        with torch.no_grad():
            dummy = torch.randn(input_size, device=device)
            out = model(dummy)
        # If output is a tensor or tuple/list of tensors, capture shape(s)
        if isinstance(out, torch.Tensor):
            out_shape = tuple(out.shape)
        elif isinstance(out, (list, tuple)):
            out_shape = []
            for o in out:
                if isinstance(o, torch.Tensor):
                    out_shape.append(tuple(o.shape))
                else:
                    out_shape.append(type(o).__name__)
            out_shape = tuple(out_shape)
        else:
            out_shape = (type(out).__name__,)
        return {"ok": True, "output_shape": out_shape}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def summarize_model_from_path(
    path: str,
    class_name: Optional[str] = None,
    input_size: Tuple[int, ...] = (1, 3, 32, 32),
    init_kwargs: Optional[Dict[str, Any]] = None,
    device: str = "cpu",
    use_torchinfo: bool = True,
    use_llm: bool = False,
    llm_client: Optional[BaseLLMClient] = None,
) -> Dict[str, Any]:
    """
    High-level helper to summarize a model defined in a Python source file.

    Parameters
    ----------
    path:
        Path to the Python source file containing the model class.
    class_name:
        Optional class name to use. If omitted, the first nn.Module subclass found
        in the module will be used.
    input_size:
        Input shape tuple including batch dimension (e.g., (1,3,32,32)).
    init_kwargs:
        Optional dict of kwargs to pass to the model's constructor.
    device:
        Device string for instantiation and dummy forward (e.g., 'cpu' or 'cuda:0').
    use_torchinfo:
        If True and the optional `torchinfo` package is available, use it for an
        extended summary. Otherwise, use a lightweight fallback.

    Returns
    -------
    A dictionary with keys:
    - success (bool)
    - summary_str (str) : detailed textual summary when available
    - total_params (int)
    - trainable_params (int)
    - output_shape (tuple) or None
    - error (str) when success is False
    """
    result: Dict[str, Any] = {
        "success": False,
        "summary_str": "",
        "total_params": None,
        "trainable_params": None,
        "output_shape": None,
        "error": None,
    }

    # Optional LLM-based summarization path.
    # If requested, require a provided `llm_client` implementing the minimal
    # interface `call(prompt, temperature=..., timeout=...) -> {"raw": ...}`.
    if use_llm:
        # If llm support isn't available in this environment, fail fast.
        if BaseLLMClient is None:
            result["error"] = "LLM support not available in this environment."
            result["success"] = False
            return result

        if llm_client is None:
            # Require explicit injection of a client to avoid accidental network calls.
            result["error"] = "LLM summarization requested but no llm_client provided"
            result["success"] = False
            return result

        try:
            # Read the model source to provide context to the LLM
            src_text = Path(path).read_text(encoding="utf-8")
            prompt = (
                "Analyze the following PyTorch model source code and return a single JSON object "
                'with keys: "summary_str", "total_params", "trainable_params". '
                "Respond ONLY with JSON.\n\n"
                f"{src_text}"
            )
            raw = llm_client.call(prompt, temperature=0.0, timeout=60)

            # Extract assistant text from common LLM response shapes
            assistant_text = None
            raw_parsed = raw.get("raw") if isinstance(raw, dict) else raw
            if isinstance(raw_parsed, dict):
                choices = raw_parsed.get("choices") or []
                if choices:
                    first = choices[0]
                    if isinstance(first, dict):
                        msg = first.get("message") or first.get("text") or {}
                        if isinstance(msg, dict):
                            assistant_text = msg.get("content") or msg.get("text")
                        elif isinstance(msg, str):
                            assistant_text = msg
                    elif isinstance(first, str):
                        assistant_text = first
            if assistant_text is None:
                assistant_text = (
                    json.dumps(raw_parsed)
                    if not isinstance(raw_parsed, str)
                    else raw_parsed
                )

            # Parse the assistant JSON and map fields into the result
            parsed = json.loads(assistant_text)
            if isinstance(parsed, dict):
                result["summary_str"] = (
                    parsed.get("summary_str") or parsed.get("summary") or ""
                )
                if parsed.get("total_params") is not None:
                    try:
                        result["total_params"] = int(parsed.get("total_params"))
                    except Exception:
                        result["total_params"] = parsed.get("total_params")
                if parsed.get("trainable_params") is not None:
                    try:
                        result["trainable_params"] = int(parsed.get("trainable_params"))
                    except Exception:
                        result["trainable_params"] = parsed.get("trainable_params")
                result["success"] = True
                return result
            else:
                result["error"] = "LLM returned non-dict JSON"
                result["success"] = False
                return result
        except Exception as e:
            result["error"] = f"LLM summarization failed: {e}"
            result["success"] = False
            return result

    try:
        if torch is None:
            raise RuntimeError(
                "PyTorch is required but is not installed in the environment."
            )

        module = _load_module_from_path(path)
        cls = _find_model_class(module, class_name)
        model = _instantiate_model(cls, init_kwargs=init_kwargs, device=device)

        total, trainable = _count_parameters(model)
        result["total_params"] = total
        result["trainable_params"] = trainable

        # Prefer torchinfo if available and requested
        if use_torchinfo and (torchinfo_summary is not None):
            try:
                # torchinfo expects input_size excluding batch dimension, or with batch?
                # Accept both common API shapes; pass `input_size` as-is.
                ti = torchinfo_summary(model, input_size=input_size, verbose=0)
                # torchinfo returns an object which stringifies to a table summary
                result["summary_str"] = str(ti)
                # torchinfo object may include total params; attempt to extract if available
                try:
                    # some torchinfo versions expose summary.total_params / trainable_params
                    total_p = getattr(ti, "total_params", None)
                    trainable_p = getattr(ti, "trainable_params", None)
                    if total_p is not None:
                        result["total_params"] = int(total_p)
                    if trainable_p is not None:
                        result["trainable_params"] = int(trainable_p)
                except Exception:
                    pass
                # Attempt a dummy forward to capture output shape as verification
                df = _run_dummy_forward(model, input_size, device=device)
                if df.get("ok"):
                    result["output_shape"] = df.get("output_shape")
                else:
                    result["output_shape"] = None
                result["success"] = True
                return result
            except Exception as e:
                _logger.debug("torchinfo summary failed: %s; falling back", e)

        # Fallback: build a simple textual summary and run a dummy forward
        lines = []
        lines.append(f"Module: {cls.__name__}")
        lines.append(f"Device: {device}")
        lines.append(f"Total parameters (computed): {total:,}")
        lines.append(f"Trainable parameters (computed): {trainable:,}")
        lines.append("Layers:")
        # List named modules excluding top-level container
        try:
            for name, module_obj in model.named_modules():
                # skip the top-level module itself
                if name == "":
                    continue
                # show brief representation
                rep = module_obj.__class__.__name__
                lines.append(f"  {name}: {rep}")
        except Exception:
            # If named_modules fails for some reason, ignore
            pass

        # Run dummy forward to capture output shape or error
        df = _run_dummy_forward(model, input_size, device=device)
        if df.get("ok"):
            result["output_shape"] = df.get("output_shape")
            lines.append(f"Dummy forward output shape: {result['output_shape']}")
            result["summary_str"] = "\n".join(lines)
            result["success"] = True
            return result
        else:
            lines.append(f"Dummy forward failed: {df.get('error')}")
            result["summary_str"] = "\n".join(lines)
            result["error"] = df.get("error")
            result["success"] = False
            return result

    except Exception as exc:  # pragma: no cover - surface errors to the caller
        result["error"] = str(exc)
        result["success"] = False
        return result
