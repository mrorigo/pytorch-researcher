import importlib.util
from pathlib import Path
from textwrap import dedent

import pytest

# Import the modules under test from the project package
from pytorch_researcher.src.pytorch_tools import model_summary, quick_evaluator


def _write_module(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(source), encoding="utf-8")


def _import_module_from_path(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    module = importlib.util.module_from_spec(spec)
    loader = spec.loader
    if loader is None:
        raise ImportError(f"Cannot load module from {path}")
    loader.exec_module(module)
    return module


def test_summarize_model_from_path_success(tmp_path):
    """
    Create a simple PyTorch model source file, use model_summary to analyze it,
    and assert that the returned summary contains expected fields.
    """
    torch = pytest.importorskip("torch")

    src = """
    import torch
    import torch.nn as nn

    class TinyModel(nn.Module):
        def __init__(self, in_channels=3, num_classes=10):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(8 * 16 * 16, num_classes),
            )

        def forward(self, x):
            return self.net(x)
    """

    mod_path = tmp_path / "tiny_model.py"
    _write_module(mod_path, src)

    # Call the summary helper
    res = model_summary.summarize_model_from_path(
        str(mod_path), class_name="TinyModel", input_size=(1, 3, 32, 32), device="cpu"
    )

    assert isinstance(res, dict)
    assert res.get("success") is True, (
        f"Expected success, got error: {res.get('error')}"
    )
    assert isinstance(res.get("total_params"), int) and res["total_params"] > 0
    assert isinstance(res.get("trainable_params"), int) and res["trainable_params"] > 0
    # Output shape should be a tuple like (1, num_classes)
    assert res.get("output_shape") is not None


def test_summarize_model_from_path_missing_class(tmp_path):
    """
    If the module does not define the requested class, the summary function
    should return a dict with success=False and an error message.
    """
    src = """
    # empty module: no model class here
    x = 1
    """

    mod_path = tmp_path / "no_model.py"
    _write_module(mod_path, src)

    res = model_summary.summarize_model_from_path(
        str(mod_path), class_name="DoesNotExist", input_size=(1, 3, 32, 32)
    )

    assert isinstance(res, dict)
    assert res.get("success") is False
    assert res.get("error") is not None


def test_quick_evaluator_with_inmemory_model():
    """
    Build a tiny model in-test, run EnhancedQuickEvaluator.quick_evaluate (via quick_evaluate_once),
    and assert the returned structure contains expected enhanced evaluation metrics.
    """
    torch = pytest.importorskip("torch")
    nn = torch.nn

    # Define a tiny model compatible with 32x32 inputs
    class TinyCNN(nn.Module):
        def __init__(self, in_channels=3, num_classes=5):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, 4, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(4 * 16 * 16, num_classes),
            )

        def forward(self, x):
            return self.net(x)

    model = TinyCNN()
    # Configure a very small quick-eval to keep test fast
    cfg = quick_evaluator.QuickEvalConfig(
        dataset_name="synthetic",  # use synthetic fallback
        subset_size=64,
        batch_size=8,
        epochs=1,
        learning_rate=1e-2,
        random_seed=123,
        num_seeds=1,  # single seed for testing
        device="cpu",
    )

    result = quick_evaluator.quick_evaluate_once(model, cfg)

    assert isinstance(result, dict)
    assert "config" in result
    assert "model_name" in result
    assert "num_seeds" in result

    # Enhanced framework structure
    assert "seed_results" in result
    assert "aggregated" in result
    assert "final" in result

    # Validate seed_results structure
    seed_results = result["seed_results"]
    assert isinstance(seed_results, list)
    assert len(seed_results) == 1  # single seed test

    # Validate seed result structure
    seed_result = seed_results[0]
    assert "seed" in seed_result
    assert "history" in seed_result
    assert "final" in seed_result
    assert isinstance(seed_result["history"], list)
    assert isinstance(seed_result["final"], dict)

    # Validate aggregated structure
    aggregated = result["aggregated"]
    assert isinstance(aggregated, dict)

    # Validate final metrics
    final = result["final"]
    assert isinstance(final, dict)
    # If val_accuracy present ensure it's a float in [0,1]
    if "val_accuracy" in final:
        assert 0.0 <= final["val_accuracy"] <= 1.0


def test_quick_evaluator_integration_with_saved_model(tmp_path):
    """
    Test EnhancedQuickEvaluator by saving a generated model file, importing it, instantiating,
    and running the enhanced evaluator on the instantiated model.
    """
    torch = pytest.importorskip("torch")
    nn = torch.nn

    src = """
    import torch
    import torch.nn as nn

    class SavedModel(nn.Module):
        def __init__(self, in_channels=3, num_classes=4):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, 4, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(4 * 16 * 16, num_classes),
            )

        def forward(self, x):
            return self.net(x)
    """

    mod_path = tmp_path / "saved_model.py"
    _write_module(mod_path, src)

    # Import module and instantiate class
    mod = _import_module_from_path(mod_path, "saved_mod_test")
    assert hasattr(mod, "SavedModel")
    cls = mod.SavedModel
    model = cls()

    cfg = quick_evaluator.QuickEvalConfig(
        dataset_name="synthetic",
        subset_size=32,
        batch_size=8,
        epochs=1,
        device="cpu",
        num_seeds=1,
    )

    qe = quick_evaluator.EnhancedQuickEvaluator(cfg)
    out = qe.quick_evaluate(model, model_name="SavedModel")

    assert isinstance(out, dict)
    # Enhanced evaluation framework structure
    assert "num_seeds" in out
    assert "aggregated" in out
    assert "final" in out
    assert "seed_results" in out
    assert "model_name" in out

    # Validate aggregated results
    aggregated = out["aggregated"]
    assert isinstance(aggregated, dict)
    if "val_accuracy" in aggregated:
        acc_stats = aggregated["val_accuracy"]
        assert "mean" in acc_stats
        assert "std" in acc_stats
        assert isinstance(acc_stats["mean"], float)

    # Validate final results
    final = out["final"]
    assert isinstance(final, dict)
    if "val_accuracy" in final:
        assert 0.0 <= final["val_accuracy"] <= 1.0

    # Validate seed results
    seed_results = out["seed_results"]
    assert isinstance(seed_results, list)
    assert len(seed_results) > 0
    for seed_result in seed_results:
        assert "seed" in seed_result
        assert "final" in seed_result
        assert "history" in seed_result


def test_enhanced_evaluation_multi_seed():
    """
    Test that multi-seed evaluation works correctly and provides statistical aggregation.
    """
    torch = pytest.importorskip("torch")
    nn = torch.nn

    # Simple model for multi-seed testing
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes=3):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((4, 4))
            self.fc = nn.Linear(8 * 4 * 4, num_classes)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.conv(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    model = SimpleCNN()

    # Configure multi-seed evaluation
    cfg = quick_evaluator.QuickEvalConfig(
        dataset_name="synthetic",
        subset_size=48,
        batch_size=8,
        epochs=1,
        num_seeds=3,  # Multiple seeds
        device="cpu",
        verbose=False,
    )

    result = quick_evaluator.quick_evaluate_once(model, cfg)

    # Validate multi-seed structure
    assert "num_seeds" in result
    assert result["num_seeds"] == 3

    # Validate seed results
    seed_results = result["seed_results"]
    assert len(seed_results) == 3

    # Each seed result should have different seed values
    seeds = [sr["seed"] for sr in seed_results]
    assert len(set(seeds)) == 3  # All different seeds

    # Validate aggregated statistics
    aggregated = result["aggregated"]
    assert "val_accuracy" in aggregated

    acc_stats = aggregated["val_accuracy"]
    assert "mean" in acc_stats
    assert "std" in acc_stats
    assert "values" in acc_stats
    assert len(acc_stats["values"]) == 3
    assert 0.0 <= acc_stats["mean"] <= 1.0
    assert acc_stats["std"] >= 0.0


def test_evaluation_goal_achievement():
    """
    Test that goal achievement detection works correctly.
    """
    torch = pytest.importorskip("torch")
    nn = torch.nn

    class TestModel(nn.Module):
        def __init__(self, num_classes=2):
            super().__init__()
            self.fc = nn.Linear(
                3072, num_classes
            )  # Match synthetic dataset input shape

        def forward(self, x):
            x = x.view(x.size(0), -1)
            return self.fc(x)

    model = TestModel()

    # Test with high target (should not achieve)
    cfg_high = quick_evaluator.QuickEvalConfig(
        dataset_name="synthetic",
        subset_size=32,
        epochs=1,
        num_seeds=1,
        target_accuracy=0.95,  # Very high target
        device="cpu",
    )

    result_high = quick_evaluator.quick_evaluate_once(model, cfg_high)
    assert "goal_achieved" in result_high
    assert result_high["goal_achieved"] is False

    # Test with low target (should achieve)
    cfg_low = quick_evaluator.QuickEvalConfig(
        dataset_name="synthetic",
        subset_size=32,
        epochs=1,
        num_seeds=1,
        target_accuracy=0.01,  # Very low target
        device="cpu",
    )

    result_low = quick_evaluator.quick_evaluate_once(model, cfg_low)
    assert "goal_achieved" in result_low
    assert result_low["goal_achieved"] is True
