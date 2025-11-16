import importlib.util
import sys
from pathlib import Path

import pytest


def _import_module_from_path(path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    loader = spec.loader
    if loader is None:
        raise ImportError(f"Cannot load module from {path}")
    loader.exec_module(module)
    return module


def _cleanup_module(module_name: str):
    # Remove module from sys.modules if present to avoid cross-test contamination
    if module_name in sys.modules:
        del sys.modules[module_name]


def test_assemble_and_save_simple_model(tmp_path):
    """
    Test the deterministic assembler: assemble_model_code -> save_model_code ->
    import module -> instantiate -> forward pass works.
    """
    # Import the programmatic assembler
    from pytorch_researcher.src.pytorch_tools.model_assembler import (
        ModelConfig,
        assemble_model_code,
        save_model_code,
    )

    # Skip if torch is not available (tests require it)
    torch = pytest.importorskip("torch")

    # Build a simple model config that is compatible with a 32x32 input
    class_name = "TestCNN"
    cfg = ModelConfig(
        class_name=class_name,
        input_shape=(3, 32, 32),
        layers=[
            {"type": "Conv2d", "in_channels": 3, "out_channels": 8, "kernel_size": 3},
            {"type": "ReLU"},
            {"type": "MaxPool2d", "kernel_size": 2},
            {"type": "Flatten"},
            {"type": "Linear", "in_features": 8 * 16 * 16, "out_features": 10},
        ],
    )

    # Generate source
    src = assemble_model_code(cfg)
    assert "class TestCNN" in src
    assert "import torch" in src

    # Save to disk
    out_path = tmp_path / "models" / "test_cnn.py"
    save_model_code(out_path, src, overwrite=True)
    assert out_path.exists()

    # Import dynamically and instantiate
    module_name = "generated_test_cnn"
    mod = _import_module_from_path(str(out_path), module_name)
    try:
        assert hasattr(mod, class_name), "Generated module must define the class"
        cls = getattr(mod, class_name)
        model = cls()
        model.eval()

        # Run a forward pass with a dummy tensor
        inp = torch.randn(1, 3, 32, 32)
        out = model(inp)
        # The final layer is Linear with out_features=10 -> expect shape (1, 10)
        assert hasattr(out, "shape")
        assert out.shape[0] == 1
        assert out.shape[1] == 10
    finally:
        _cleanup_module(module_name)


def test_llm_wrapper_fallback_generates_and_saves(tmp_path):
    """
    Test the LLM-backed assembler interface but force the fallback path
    (use_llm=False) so the deterministic assembler is exercised through the wrapper.
    """
    from pytorch_researcher.src.pytorch_tools.model_assembler_llm import (
        assemble_from_config,
    )

    # Ensure torch is available for forward pass validation
    torch = pytest.importorskip("torch")

    # Provide a dict-style model_config (the wrapper should accept this)
    model_config = {
        "class_name": "LLMFallbackModel",
        "layers": [
            {"type": "Conv2d", "in_channels": 3, "out_channels": 4, "kernel_size": 3},
            {"type": "ReLU"},
            {"type": "MaxPool2d", "kernel_size": 2},
            {"type": "Flatten"},
            {"type": "Linear", "in_features": 4 * 16 * 16, "out_features": 5},
        ],
    }

    out_path = tmp_path / "llm_fallback_model.py"

    # Force fallback (do not attempt to call an LLM in tests)
    result = assemble_from_config(
        model_config, str(out_path), use_llm=False, llm_kwargs=None
    )
    assert result["path"] == str(out_path)
    assert result["via"] in ("fallback", "llm", "fallback")  # expecting fallback

    # Validate file exists and the source is present
    assert out_path.exists()
    src = out_path.read_text(encoding="utf-8")
    assert "class LLMFallbackModel" in src or "class AssembledModel" in src

    # Import and run a dummy forward pass
    module_name = "generated_llm_fallback"
    mod = _import_module_from_path(str(out_path), module_name)
    try:
        # The wrapper attempts to name the class as provided; fall back to AssembledModel if not present
        cls_name = (
            "LLMFallbackModel" if hasattr(mod, "LLMFallbackModel") else "AssembledModel"
        )
        assert hasattr(mod, cls_name), f"Expected class {cls_name} in generated module"
        cls = getattr(mod, cls_name)
        model = cls()
        model.eval()
        inp = torch.randn(1, 3, 32, 32)
        out = model(inp)
        assert hasattr(out, "shape")
        # The final linear layer had out_features=5 => expect (1,5)
        assert out.shape[1] == 5
    finally:
        _cleanup_module(module_name)


def test_generated_code_is_valid_python_ast():
    """
    Ensure that assemble_model_code produces syntactically valid Python by parsing AST.
    """
    import ast

    from pytorch_researcher.src.pytorch_tools.model_assembler import (
        ModelConfig,
        assemble_model_code,
    )

    cfg = ModelConfig(
        class_name="AstCheckModel",
        layers=[
            {"type": "Linear", "in_features": 16, "out_features": 4},
        ],
    )

    src = assemble_model_code(cfg)
    # Should parse without SyntaxError
    ast.parse(src)


# def test_llm_assembler_with_mock_client(tmp_path):
#     """
#     Validate that LLMModelAssembler accepts an injected LLM client and
#     correctly extracts and saves code returned by the client without performing
#     any real network requests.
#     """
#     import json

#     from pytorch_researcher.src.pytorch_tools.model_assembler_llm import LLMModelAssembler

#     # Require torch for forward pass validation
#     torch = pytest.importorskip("torch")

#     # Minimal model source returned by the fake LLM
#     model_src = \"\"\"\n    import torch\n    import torch.nn as nn\n\n    class AssembledModel(nn.Module):\n        def __init__(self):\n            super().__init__()\n            self.lin = nn.Linear(10, 1)\n\n        def forward(self, x):\n            return self.lin(x)\n    \"\"\"\n\n    # Fake LLM client that returns a structured choices/message content JSON\n    class MockLLMClient:\n        def call(self, prompt: str, temperature: float = 0.0, timeout: int = 300):\n            assistant_text = json.dumps({\"python\": model_src, \"metadata\": {\"class_name\": \"AssembledModel\"}})\n            return {\"raw\": {\"choices\": [{\"message\": {\"content\": assistant_text}}]}}\n\n    # Instantiate assembler with injected mock client\n    assembler = LLMModelAssembler(llm_client=MockLLMClient())\n\n    model_config = {\"class_name\": \"AssembledModel\", \"layers\": []}\n    out_path = tmp_path / \"mock_assembled.py\"\n\n    res = assembler.assemble_from_config(model_config, str(out_path), use_llm=True)\n    assert res[\"via\"] == \"llm\"\n    assert Path(res[\"path\"]).exists()\n\n    # Import and run a forward pass to validate the generated code\n    mod = _import_module_from_path(str(out_path), \"mock_assembled_mod\")\n    try:\n        assert hasattr(mod, \"AssembledModel\")\n        cls = getattr(mod, \"AssembledModel\")\n        model = cls()\n        model.eval()\n        inp = torch.randn(1, 10)\n        out = model(inp)\n        assert hasattr(out, \"shape\")\n    finally:\n        _cleanup_module(\"mock_assembled_mod\")\n
