# pytorch-researcher/pytorch_researcher/src/pytorch_tools/model_assembler.py
"""Minimal `pytorch_model_assembler` skeleton.

This module provides a small, testable implementation of a PyTorch model assembler
that can generate a Python source string implementing a torch.nn.Module class
from a structured `ModelConfig` dataclass.

The implementation is intentionally simple and deterministic so it can be used
for unit tests and early integration before wiring up any LLM-based code
generation backend. It supports a limited set of layer types and produces
syntactically valid PyTorch code which can be written to disk and imported.

Planned extension points:
- Replace or augment the programmatic assembler with a call to an LLM-based
  assembler for more sophisticated architecture generation.
- Add richer validation and more layer types.
- Support templating and style conventions.

Example usage
-------------
from pytorch_researcher.src.pytorch_tools.model_assembler import (
    ModelConfig, assemble_model_code, save_model_code
)

cfg = ModelConfig(
    class_name="SimpleCNN",
    input_shape=(3, 32, 32),
    layers=[
        {"type": "Conv2d", "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1},
        {"type": "ReLU"},
        {"type": "MaxPool2d", "kernel_size": 2},
        {"type": "Flatten"},
        {"type": "Linear", "out_features": 10},
    ],
)

code = assemble_model_code(cfg)
save_model_code("src/models/simple_cnn.py", code)
"""

from __future__ import annotations

import ast
import dataclasses
import logging
import textwrap
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclasses.dataclass
class ModelConfig:
    """Structured configuration describing a simple PyTorch model.

    Attributes:
        class_name: Name of the generated nn.Module subclass.
        input_shape: Tuple describing the input tensor (batch dim excluded).
        layers: Ordered list of layer descriptions. Each layer is a dict with a
            required 'type' key and optional parameters depending on the layer.
        docstring: Optional module/class docstring to include in generated code.

    """

    class_name: str
    input_shape: tuple | None = None
    layers: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    docstring: str | None = None


class ModelAssemblerError(Exception):
    """Raised when assembler encounters invalid configuration or generation errors."""


# Supported layer types and how they map to nn.* constructors.
# This is intentionally minimal and extendable.
_SUPPORTED_LAYER_MAP = {
    "Conv2d": "nn.Conv2d",
    "Linear": "nn.Linear",
    "ReLU": "nn.ReLU",
    "LeakyReLU": "nn.LeakyReLU",
    "MaxPool2d": "nn.MaxPool2d",
    "AvgPool2d": "nn.AvgPool2d",
    "BatchNorm2d": "nn.BatchNorm2d",
    "Dropout": "nn.Dropout",
    "Flatten": "Flatten",  # special-cased in forward
}


def _format_param_value(v: Any) -> str:
    """Return a Python source snippet for a simple parameter value."""
    if isinstance(v, str):
        return repr(v)
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, (list, tuple)):
        inner = ", ".join(_format_param_value(x) for x in v)
        # single-element tuple must have trailing comma
        if isinstance(v, tuple) and len(v) == 1:
            inner = inner + ","
        return f"({inner})"
    return repr(v)


def _layer_init_snippet(
    name: str, layer: dict[str, Any], idx: int, prev_channels: int | None
) -> str:
    """Construct a line of code for __init__ assigning a layer to self.<name>.

    Returns code string and the new channel count (if applicable).
    """
    t = layer.get("type")
    if t is None:
        raise ModelAssemblerError(f"Layer at index {idx} missing required 'type' key")

    if t == "Flatten":
        # No parameterized nn.Module needed for Flatten in this simple assembler
        return f"        self.{name} = nn.Flatten()", prev_channels

    if t not in _SUPPORTED_LAYER_MAP:
        raise ModelAssemblerError(f"Unsupported layer type: {t!r}")

    ctor = _SUPPORTED_LAYER_MAP[t]

    # Build parameter map from layer dict excluding 'type'
    params = {k: v for k, v in layer.items() if k != "type"}

    # Special-case Conv2d and Linear to provide sensible positional ordering and defaults
    if t == "Conv2d":
        # Default kernel/stride/padding values if not provided
        kernel_size = params.pop("kernel_size", 3)
        stride = params.pop("stride", 1)
        padding = params.pop("padding", 1)
        dilation = params.pop("dilation", 1)
        groups = params.pop("groups", 1)
        bias = params.pop("bias", True)

        # in_channels: prefer explicit, else use prev_channels if available
        in_channels = params.pop("in_channels", None)
        out_channels = params.pop("out_channels", None)
        if in_channels is None:
            if prev_channels is not None:
                in_channels = prev_channels
            else:
                raise ModelAssemblerError(
                    f"Conv2d at index {idx} requires 'in_channels' or a valid prev_channels"
                )
        if out_channels is None:
            raise ModelAssemblerError(f"Conv2d at index {idx} requires 'out_channels'")

        # Build ordered positional args: in_channels, out_channels, kernel_size
        positional = [
            _format_param_value(in_channels),
            _format_param_value(out_channels),
            _format_param_value(kernel_size),
        ]
        # Append optional keyword args with sensible defaults only if different from defaults
        kw_parts = []
        if stride != 1:
            kw_parts.append(f"stride={_format_param_value(stride)}")
        if padding != 0:
            kw_parts.append(f"padding={_format_param_value(padding)}")
        if dilation != 1:
            kw_parts.append(f"dilation={_format_param_value(dilation)}")
        if groups != 1:
            kw_parts.append(f"groups={_format_param_value(groups)}")
        if bias is not True:
            kw_parts.append(f"bias={_format_param_value(bias)}")

        param_list = ", ".join(positional + kw_parts)
        line = f"        self.{name} = {ctor}({param_list})"
        # update prev_channels to reflect conv output channels.
        # Note: this variable tracks the number of output channels only.
        # Spatial dimensions (height/width) depend on kernel_size/stride/padding
        # and are not tracked by this simple assembler. If padding preserves
        # spatial dims (e.g., kernel_size=3 with padding=1), using the channel
        # count alone is sufficient for downstream in_features inference.
        return line, int(out_channels)

    if t == "Linear":
        # Default bias True
        bias = params.pop("bias", True)
        out_features = params.pop("out_features", None)
        in_features = params.pop("in_features", None)

        # If in_features missing and prev_channels looks like a scalar, try to reuse it
        if in_features is None:
            if prev_channels is not None:
                in_features = prev_channels
            else:
                # leave in_features absent and rely on named parameter if provided
                in_features = None

        if out_features is None:
            raise ModelAssemblerError(f"Linear at index {idx} requires 'out_features'")

        # If we have in_features, prefer positional args (in_features, out_features)
        if in_features is not None:
            positional = [
                _format_param_value(in_features),
                _format_param_value(out_features),
            ]
            kw_parts = []
            if bias is not True:
                kw_parts.append(f"bias={_format_param_value(bias)}")
            param_list = ", ".join(positional + kw_parts)
            line = f"        self.{name} = {ctor}({param_list})"
        else:
            # Fallback to keyword arguments (out_features=...)
            kw_parts = [f"out_features={_format_param_value(out_features)}"]
            if bias is not True:
                kw_parts.append(f"bias={_format_param_value(bias)}")
            param_list = ", ".join(kw_parts)
            line = f"        self.{name} = {ctor}({param_list})"

        # Linear changes channel semantics; set prev_channels to out_features (if int)
        try:
            new_prev = int(out_features)
        except Exception:
            new_prev = prev_channels
        return line, new_prev

    # Generic handling for other supported layers: preserve explicit params order
    param_strs = []
    # Keep deterministic ordering by sorting keys except preserve tuple/list ordering when present
    for k in sorted(params.keys()):
        v = params[k]
        param_strs.append(f"{k}={_format_param_value(v)}")
    param_list = ", ".join(param_strs)
    line = (
        f"        self.{name} = {ctor}({param_list})"
        if param_list
        else f"        self.{name} = {ctor}()"
    )
    return line, prev_channels


def assemble_model_code(cfg: ModelConfig) -> str:
    """Assemble Python source code for a PyTorch `nn.Module` from a ModelConfig.

    The generated code imports torch/nn, defines a Module subclass named
    cfg.class_name, implements a minimal __init__ registering layers and a
    forward method that applies them in order. For layers that require special
    handling (e.g., Flatten), the forward will include the appropriate calls.

    This routine is intentionally conservative: it will not attempt to infer
    tensor shapes or compute `in_features` for Linear layers. For more complex
    generation (automatic shape inference, residual links, dynamic graphs),
    replace this with a more advanced generator or an LLM-backed tool.

    Returns:
        A string containing the Python module source code.

    """
    if not cfg.class_name.isidentifier():
        raise ModelAssemblerError(
            f"class_name {cfg.class_name!r} is not a valid Python identifier"
        )

    lines: list[str] = []
    lines.append("import torch")
    lines.append("import torch.nn as nn")
    lines.append("")
    if cfg.docstring:
        doc = textwrap.indent(textwrap.dedent(cfg.docstring), "    ")
    else:
        doc = None

    lines.append(f"class {cfg.class_name}(nn.Module):")
    if doc:
        lines.append('    """')
        for l in cfg.docstring.splitlines():
            lines.append(f"    {l}")
        lines.append('    """')
    lines.append("    def __init__(self):")
    lines.append("        super().__init__()")

    # Build __init__ layer registrations
    prev_channels = None
    init_lines: list[str] = []
    forward_lines: list[str] = []
    forward_lines.append("        x = input")
    for idx, layer in enumerate(cfg.layers):
        lname = f"layer_{idx}"
        try:
            init_snippet, prev_channels = _layer_init_snippet(
                lname, layer, idx, prev_channels
            )
        except ModelAssemblerError:
            raise
        init_lines.append(init_snippet)

        # Forward behavior: for most nn.* modules we can just call them
        ltype = layer.get("type")
        if ltype == "Flatten":
            forward_lines.append(f"        x = self.{lname}(x)")
        else:
            forward_lines.append(f"        x = self.{lname}(x)")

    if not init_lines:
        # Add a simple identity layer to ensure module is valid
        init_lines.append("        self.identity = nn.Identity()")
        forward_lines = ["        x = self.identity(input)"]

    # Attach init lines
    lines.extend(init_lines)
    lines.append("")
    lines.append("    def forward(self, input):")
    lines.extend(forward_lines)
    lines.append("        return x")
    lines.append("")

    src = "\n".join(lines)
    # Final formatting
    src = textwrap.dedent(src)
    # Validate Python syntax
    try:
        ast.parse(src)
    except SyntaxError as e:
        log.error("Generated model code contains a syntax error: %s", e)
        raise ModelAssemblerError(f"Generated code is invalid Python: {e}") from e

    return src


def save_model_code(path: str | Path, code: str, overwrite: bool = True) -> Path:
    """Save generated model source string to a file.

    Args:
        path: Destination file path (will be created if necessary).
        code: Python source code to write.
        overwrite: If False and file exists, raises FileExistsError.

    Returns:
        The path that was written as a pathlib.Path.

    """
    p = Path(path)
    if p.exists() and not overwrite:
        raise FileExistsError(f"File exists and overwrite=False: {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(code, encoding="utf-8")
    return p


__all__ = [
    "ModelAssemblerError",
    "ModelConfig",
    "assemble_model_code",
    "save_model_code",
]
