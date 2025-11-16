"""
Test script for Enhanced Evaluation Framework

This script demonstrates the enhanced evaluation capabilities including:
- Multi-seed evaluation for statistical significance
- Hugging Face datasets integration
- Enhanced metrics aggregation and reporting
- Reproducible evaluation with fixed seeds

Usage:
    python test_enhanced_evaluation.py
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add the source directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "pytorch_researcher" / "src"))

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Please install with: pip install torch")

try:
    import torchvision
    import torchvision.transforms as transforms

    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    print("TorchVision not available. Some tests will be skipped.")

try:
    from datasets import list_datasets, load_dataset

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Hugging Face datasets not available. HF tests will be skipped.")

# Import our enhanced modules
from pytorch_tools.dataset_loader import (
    DatasetConfig,
    FlexibleDatasetLoader,
    create_flexible_dataset_config,
    get_flexible_dataset_loader,
    list_supported_datasets,
)
from pytorch_tools.quick_evaluator import (
    EnhancedQuickEvaluator,
    QuickEvalConfig,
    quick_evaluate_legacy,
    quick_evaluate_once,
)


class TestModels:
    """Collection of test models for evaluation."""

    @staticmethod
    def get_simple_cnn(num_classes=10):
        """Simple CNN for image classification."""

        class SimpleCNN(nn.Module):
            def __init__(self, num_classes=10):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d((4, 4))
                self.fc1 = nn.Linear(64 * 4 * 4, 128)
                self.fc2 = nn.Linear(128, num_classes)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.2)

            def forward(self, x):
                x = self.relu(self.conv1(x))
                x = self.relu(self.conv2(x))
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.fc2(x)
                return x

        return SimpleCNN(num_classes)

    @staticmethod
    def get_mlp(input_size=784, num_classes=10):
        """Simple MLP for tabular data."""

        class MLP(nn.Module):
            def __init__(self, input_size=784, num_classes=10):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(input_size, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, num_classes),
                )

            def forward(self, x):
                x = x.view(x.size(0), -1)  # Flatten
                return self.layers(x)

        return MLP(input_size, num_classes)


def test_dataset_loader():
    """Test the enhanced dataset loader functionality."""
    print("\n" + "=" * 60)
    print("TESTING ENHANCED DATASET LOADER")
    print("=" * 60)

    # Test supported datasets listing
    if HF_AVAILABLE:
        print("✓ Hugging Face datasets available")
        supported_datasets = list_supported_datasets()
        print(f"Supported datasets: {supported_datasets}")
    else:
        print("✗ Hugging Face datasets not available")

    # Test dataset configuration creation
    print("\n--- Testing Dataset Configuration ---")

    # Test CIFAR-10 configuration
    try:
        config = create_flexible_dataset_config(
            "cifar10", subset_size=100, seed=42, transforms=["basic", "normalize"]
        )
        print(f"✓ Created CIFAR-10 config: {config.name}")
        print(f"  - Subset size: {config.subset_size}")
        print(f"  - Transforms: {config.transforms}")
        print(f"  - Input shape: {config.input_shape}")
    except Exception as e:
        print(f"✗ Failed to create CIFAR-10 config: {e}")

    # Test synthetic dataset configuration
    synthetic_config = None
    try:
        synthetic_config = create_flexible_dataset_config(
            "synthetic", subset_size=200, input_shape=(3, 32, 32), num_classes=10
        )
        print(f"✓ Created synthetic config: {synthetic_config.name}")
        print(f"  - Input shape: {synthetic_config.input_shape}")
        print(f"  - Num classes: {synthetic_config.num_classes}")
    except Exception as e:
        print(f"✗ Failed to create synthetic config: {e}")

    # Test dataset loading (if dependencies available)
    print("\n--- Testing Dataset Loading ---")

    if TORCH_AVAILABLE and synthetic_config:
        try:
            loader = get_flexible_dataset_loader(synthetic_config)
            train_dataset, val_dataset, test_dataset = loader.load()
            print(f"✓ Loaded synthetic dataset")
            print(f"  - Training samples: {len(train_dataset)}")
            if val_dataset:
                print(f"  - Validation samples: {len(val_dataset)}")
            if test_dataset:
                print(f"  - Test samples: {len(test_dataset)}")

            # Test DataLoader creation
            train_loader = loader.create_dataloader(train_dataset, shuffle=True)
            print(f"  - Training batches: {len(train_loader)}")

        except Exception as e:
            print(f"✗ Failed to load synthetic dataset: {e}")
    else:
        print("✗ PyTorch not available for dataset loading test")

    # Test Hugging Face dataset loading (if available)
    if HF_AVAILABLE and TORCH_AVAILABLE:
        try:
            print("\n--- Testing Hugging Face Dataset Loading ---")
            hf_config = create_flexible_dataset_config(
                "imdb",
                subset_size=50,  # Small subset for testing
                seed=42,
            )
            loader = get_flexible_dataset_loader(hf_config)
            train_dataset, val_dataset, test_dataset = loader.load()
            print(f"✓ Loaded IMDB dataset")
            print(f"  - Training samples: {len(train_dataset)}")
            if val_dataset:
                print(f"  - Validation samples: {len(val_dataset)}")
            if test_dataset:
                print(f"  - Test samples: {len(test_dataset)}")
        except Exception as e:
            print(f"✗ Failed to load HF dataset: {e}")
    else:
        print("\n--- Skipping HF Dataset Tests ---")
        print("  (Missing dependencies: HF datasets or PyTorch)")


def test_enhanced_evaluator():
    """Test the enhanced quick evaluator with multi-seed support."""
    print("\n" + "=" * 60)
    print("TESTING ENHANCED QUICK EVALUATOR")
    print("=" * 60)

    if not TORCH_AVAILABLE:
        print("✗ PyTorch not available. Skipping evaluator tests.")
        return

    # Test 1: Basic single-seed evaluation
    print("\n--- Test 1: Single-Seed Evaluation ---")
    try:
        model = TestModels.get_simple_cnn(num_classes=10)
        config = QuickEvalConfig(
            dataset_name="synthetic",
            subset_size=100,
            epochs=2,
            num_seeds=1,
            verbose=True,
        )

        result = quick_evaluate_once(model, config)
        print("✓ Single-seed evaluation completed")
        print(f"  - Model: {result['model_name']}")
        print(f"  - Final accuracy: {result['final'].get('val_accuracy', 'N/A')}")
        print(f"  - Goal achieved: {result.get('goal_achieved', False)}")

    except Exception as e:
        print(f"✗ Single-seed evaluation failed: {e}")

    # Test 2: Multi-seed evaluation
    print("\n--- Test 2: Multi-Seed Evaluation ---")
    try:
        model = TestModels.get_simple_cnn(num_classes=10)
        config = QuickEvalConfig(
            dataset_name="synthetic",
            subset_size=100,
            epochs=2,
            num_seeds=3,  # Multiple seeds for statistical analysis
            verbose=False,
        )

        start_time = time.time()
        result = quick_evaluate_once(model, config)
        evaluation_time = time.time() - start_time

        print("✓ Multi-seed evaluation completed")
        print(f"  - Model: {result['model_name']}")
        print(f"  - Seeds evaluated: {result['num_seeds']}")
        print(f"  - Evaluation time: {evaluation_time:.2f}s")

        # Display statistical results
        if "aggregated" in result:
            agg = result["aggregated"]
            if "val_accuracy" in agg:
                acc_stats = agg["val_accuracy"]
                print(f"  - Mean accuracy: {acc_stats['mean']:.4f}")
                print(f"  - Std deviation: {acc_stats['std']:.4f}")
                print(f"  - Min accuracy: {acc_stats['min']:.4f}")
                print(f"  - Max accuracy: {acc_stats['max']:.4f}")
                if "ci_95" in acc_stats:
                    ci_low, ci_high = acc_stats["ci_95"]
                    print(f"  - 95% CI: [{ci_low:.4f}, {ci_high:.4f}]")

        print(f"  - Goal achieved: {result.get('goal_achieved', False)}")

    except Exception as e:
        print(f"✗ Multi-seed evaluation failed: {e}")

    # Test 3: Legacy compatibility
    print("\n--- Test 3: Legacy Compatibility ---")
    try:
        model = TestModels.get_simple_cnn(num_classes=10)
        config = QuickEvalConfig(
            dataset_name="synthetic", subset_size=50, epochs=1, num_seeds=1
        )

        result = quick_evaluate_legacy(model, config)
        print("✓ Legacy evaluation completed")
        print(f"  - Seeds evaluated: {result['num_seeds']}")
        print(f"  - Final accuracy: {result['final'].get('val_accuracy', 'N/A')}")

    except Exception as e:
        print(f"✗ Legacy evaluation failed: {e}")

    # Test 4: CIFAR-10 evaluation (if torchvision available)
    if TORCHVISION_AVAILABLE:
        print("\n--- Test 4: Real Dataset Evaluation (CIFAR-10) ---")
        try:
            model = TestModels.get_simple_cnn(num_classes=10)
            config = QuickEvalConfig(
                dataset_name="cifar10",
                subset_size=200,
                epochs=2,
                num_seeds=2,
                verbose=False,
            )

            result = quick_evaluate_once(model, config)
            print("✓ CIFAR-10 evaluation completed")
            print(f"  - Final accuracy: {result['final'].get('val_accuracy', 'N/A')}")

        except Exception as e:
            print(f"✗ CIFAR-10 evaluation failed: {e}")
    else:
        print("\n--- Test 4: Skipped (TorchVision not available) ---")


def test_enhanced_features():
    """Test enhanced features like early stopping, target accuracy, etc."""
    print("\n" + "=" * 60)
    print("TESTING ENHANCED FEATURES")
    print("=" * 60)

    if not TORCH_AVAILABLE:
        print("✗ PyTorch not available. Skipping enhanced features tests.")
        return

    # Test early stopping
    print("\n--- Test: Early Stopping ---")
    try:
        model = TestModels.get_simple_cnn(num_classes=10)
        config = QuickEvalConfig(
            dataset_name="synthetic",
            subset_size=100,
            epochs=10,  # More epochs to trigger early stopping
            early_stopping_patience=2,
            verbose=False,
        )

        result = quick_evaluate_once(model, config)
        print("✓ Early stopping test completed")

        # Check if early stopping was triggered
        if result["seed_results"] and len(result["seed_results"]) > 0:
            history = result["seed_results"][0]["history"]
            epochs_run = len(history)
            print(f"  - Epochs run: {epochs_run} (out of {config.epochs} max)")
            if epochs_run < config.epochs:
                print("  - Early stopping was triggered!")

    except Exception as e:
        print(f"✗ Early stopping test failed: {e}")

    # Test target accuracy detection
    print("\n--- Test: Target Accuracy Detection ---")
    try:
        model = TestModels.get_simple_cnn(num_classes=10)
        config = QuickEvalConfig(
            dataset_name="synthetic",
            subset_size=100,
            epochs=2,
            target_accuracy=0.8,  # High target that likely won't be met
            verbose=False,
        )

        result = quick_evaluate_once(model, config)
        print("✓ Target accuracy test completed")
        print(f"  - Target accuracy: {config.target_accuracy}")
        print(f"  - Achieved accuracy: {result['final'].get('val_accuracy', 'N/A')}")
        print(f"  - Goal achieved: {result.get('goal_achieved', False)}")

    except Exception as e:
        print(f"✗ Target accuracy test failed: {e}")

    # Test deterministic evaluation
    print("\n--- Test: Deterministic Evaluation ---")
    try:
        model1 = TestModels.get_simple_cnn(num_classes=10)
        model2 = TestModels.get_simple_cnn(num_classes=10)

        # Use same seed for deterministic evaluation
        config = QuickEvalConfig(
            dataset_name="synthetic",
            subset_size=50,
            epochs=1,
            random_seed=42,
            deterministic=True,
            verbose=False,
        )

        result1 = quick_evaluate_once(model1, config)

        # Same configuration, different model instance
        result2 = quick_evaluate_once(model2, config)

        print("✓ Deterministic evaluation test completed")
        print(f"  - Run 1 accuracy: {result1['final'].get('val_accuracy', 'N/A')}")
        print(f"  - Run 2 accuracy: {result2['final'].get('val_accuracy', 'N/A')}")

        # Compare results (should be identical with deterministic evaluation)
        acc1 = result1["final"].get("val_accuracy", 0)
        acc2 = result2["final"].get("val_accuracy", 0)
        if abs(acc1 - acc2) < 1e-6:
            print("  - Results are deterministic ✓")
        else:
            print("  - Results differ (non-deterministic) ⚠")

    except Exception as e:
        print(f"✗ Deterministic evaluation test failed: {e}")


def run_comprehensive_demo():
    """Run a comprehensive demonstration of all features."""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE DEMONSTRATION")
    print("=" * 60)

    if not TORCH_AVAILABLE:
        print("✗ PyTorch not available. Cannot run demonstration.")
        return

    print("This demonstration will showcase the enhanced evaluation framework")
    print("with multiple datasets, models, and statistical analysis.\n")

    # Demo configuration
    demo_configs = [
        {
            "name": "Quick CNN Test",
            "model": TestModels.get_simple_cnn(num_classes=10),
            "config": QuickEvalConfig(
                dataset_name="synthetic",
                subset_size=100,
                epochs=2,
                num_seeds=2,
                target_accuracy=0.7,
                verbose=False,
            ),
        },
        {
            "name": "MLP Tabular Test",
            "model": TestModels.get_mlp(input_size=100, num_classes=5),
            "config": QuickEvalConfig(
                dataset_name="synthetic",
                subset_size=150,
                epochs=3,
                num_seeds=3,
                target_accuracy=0.6,
                verbose=False,
            ),
        },
    ]

    # Add CIFAR-10 test if torchvision available
    if TORCHVISION_AVAILABLE:
        demo_configs.append(
            {
                "name": "Real Dataset Test (CIFAR-10)",
                "model": TestModels.get_simple_cnn(num_classes=10),
                "config": QuickEvalConfig(
                    dataset_name="cifar10",
                    subset_size=300,
                    epochs=3,
                    num_seeds=2,
                    target_accuracy=0.5,
                    verbose=False,
                ),
            }
        )

    results = []

    for i, demo in enumerate(demo_configs, 1):
        print(f"\n--- Demo {i}: {demo['name']} ---")

        try:
            start_time = time.time()
            result = quick_evaluate_once(demo["model"], demo["config"])
            demo_time = time.time() - start_time

            print(f"✓ Completed in {demo_time:.2f}s")

            # Display key results
            final_acc = result["final"].get("val_accuracy", "N/A")
            print(f"  - Final accuracy: {final_acc}")
            print(f"  - Goal achieved: {result.get('goal_achieved', False)}")

            if result["num_seeds"] > 1 and "aggregated" in result:
                agg = result["aggregated"]
                if "val_accuracy" in agg:
                    acc_stats = agg["val_accuracy"]
                    print(
                        f"  - Mean ± Std: {acc_stats['mean']:.4f} ± {acc_stats['std']:.4f}"
                    )
                    if "ci_95" in acc_stats:
                        ci_low, ci_high = acc_stats["ci_95"]
                        print(f"  - 95% CI: [{ci_low:.4f}, {ci_high:.4f}]")

            results.append(
                {
                    "name": demo["name"],
                    "success": True,
                    "accuracy": final_acc,
                    "goal_achieved": result.get("goal_achieved", False),
                    "time": demo_time,
                }
            )

        except Exception as e:
            print(f"✗ Failed: {e}")
            results.append({"name": demo["name"], "success": False, "error": str(e)})

    # Summary
    print("\n" + "=" * 60)
    print("DEMONSTRATION SUMMARY")
    print("=" * 60)

    for result in results:
        status = "✓" if result["success"] else "✗"
        print(f"{status} {result['name']}")
        if result["success"]:
            if isinstance(result["accuracy"], (int, float)):
                print(f"    Accuracy: {result['accuracy']:.4f}")
            else:
                print(f"    Accuracy: {result['accuracy']}")
            print(f"    Goal achieved: {result['goal_achieved']}")
            print(f"    Time: {result['time']:.2f}s")
        else:
            print(f"    Error: {result.get('error', 'Unknown')}")

    successful_tests = sum(1 for r in results if r["success"])
    print(
        f"\nSuccess rate: {successful_tests}/{len(results)} ({100 * successful_tests / len(results):.1f}%)"
    )


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test Enhanced Evaluation Framework")
    parser.add_argument(
        "--demo", action="store_true", help="Run comprehensive demonstration"
    )
    parser.add_argument(
        "--dataset", action="store_true", help="Test dataset loader only"
    )
    parser.add_argument("--evaluator", action="store_true", help="Test evaluator only")
    parser.add_argument(
        "--features", action="store_true", help="Test enhanced features only"
    )

    args = parser.parse_args()

    print("Enhanced Evaluation Framework Test Suite")
    print("=" * 50)
    print(f"PyTorch available: {TORCH_AVAILABLE}")
    print(f"TorchVision available: {TORCHVISION_AVAILABLE}")
    print(f"HuggingFace datasets available: {HF_AVAILABLE}")

    if not TORCH_AVAILABLE:
        print("\n⚠️  WARNING: PyTorch is required for most tests.")
        print("Please install with: pip install torch")
        return

    if args.dataset:
        test_dataset_loader()
    elif args.evaluator:
        test_enhanced_evaluator()
    elif args.features:
        test_enhanced_features()
    elif args.demo:
        run_comprehensive_demo()
    else:
        # Run all tests
        test_dataset_loader()
        test_enhanced_evaluator()
        test_enhanced_features()
        run_comprehensive_demo()

    print("\n" + "=" * 50)
    print("Test suite completed!")


if __name__ == "__main__":
    main()
