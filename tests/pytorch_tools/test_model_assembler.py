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


def test_deterministic_first_strategy_simple_config(tmp_path):
    """
    Test deterministic-first strategy: simple config uses deterministic path by default.
    """
    from pytorch_researcher.src.pytorch_tools.model_assembler_llm import (
        assemble_from_config,
    )

    torch = pytest.importorskip("torch")

    # Simple config that should use deterministic-first with proper Linear layer config
    simple_config = {
        "class_name": "SimpleCNN",
        "input_shape": (3, 32, 32),
        "layers": [
            {"type": "Conv2d", "in_channels": 3, "out_channels": 16, "kernel_size": 3},
            {"type": "ReLU"},
            {"type": "MaxPool2d", "kernel_size": 2},
            {"type": "Flatten"},
            {"type": "Linear", "in_features": 16 * 16 * 16, "out_features": 10},
        ],
    }

    out_path = tmp_path / "simple_cnn.py"
    result = assemble_from_config(simple_config, str(out_path), use_llm=True)
    
    assert result["path"] == str(out_path)
    assert result["via"] == "deterministic"
    assert "class SimpleCNN" in result["source"]


def test_deterministic_first_complex_config_uses_llm_fallback(tmp_path):
    """
    Test that complex/unsupported configs fall back to LLM path when use_llm=True.
    Since we can't mock LLM in unit tests easily, test the logic path exists.
    """
    from pytorch_researcher.src.pytorch_tools.model_assembler_llm import (
        assemble_from_config,
    )

    torch = pytest.importorskip("torch")

    # Complex config with unsupported layer that should fallback to LLM
    complex_config = {
        "class_name": "ComplexModel",
        "input_shape": (3, 32, 32),
        "layers": [
            {"type": "CustomAttention", "heads": 8},  # Unsupported layer type
            {"type": "ResBlock"},  # Unsupported
        ],
    }

    out_path = tmp_path / "complex_model.py"
    
    # Since no LLM client is configured, expect LLM client error
    with pytest.raises(Exception) as exc_info:
        assemble_from_config(complex_config, str(out_path), use_llm=True)
    
    # Should fail with generic error since both deterministic and LLM failed
    assert "Both deterministic and LLM assembly failed" in str(exc_info.value)
    
    # File should not exist since assembly failed completely
    assert not out_path.exists()


def test_complexity_heuristic_classification():
    """
    Test the config complexity heuristic directly.
    """
    from pytorch_researcher.src.pytorch_tools.model_assembler_llm import LLMModelAssembler
    
    assembler = LLMModelAssembler()
    
    # Simple config should be deterministic-friendly
    simple = {
        "input_shape": (3, 32, 32),
        "layers": [
            {"type": "Conv2d", "out_channels": 32},
            {"type": "ReLU"},
            {"type": "Flatten"},
        ]
    }
    assert assembler.is_deterministic_friendly(simple) == True

    # Complex/unsupported should not
    complex = {
        "input_shape": (3, 32, 32),
        "layers": [{"type": "TransformerBlock"}]
    }
    assert assembler.is_deterministic_friendly(complex) == False

    # Edge cases
    empty = {"layers": []}
    assert assembler.is_deterministic_friendly(empty) == False

    no_input_shape = {
        "input_shape": None,
        "layers": [{"type": "Linear", "out_features": 10}]
    }
    assert assembler.is_deterministic_friendly(no_input_shape) == False


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
