"""
Enhanced Metrics Test Suite

This module tests the enhanced evaluation metrics implementation including:
- F1, precision, recall calculations
- AUC and ROC analysis
- Confusion matrix and per-class analysis
- Learning dynamics tracking
- Statistical aggregation across seeds
"""

import json
import os

# Test the enhanced evaluator
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# Import PyTorch for testing
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Some tests will be skipped.")

# Test the enhanced evaluator
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pytorch_researcher/src"))

from pytorch_tools.quick_evaluator import (
    SKLEARN_AVAILABLE,
    EnhancedQuickEvaluator,
    QuickEvalConfig,
)


class TestEnhancedMetrics(unittest.TestCase):
    """Test suite for enhanced metrics functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = QuickEvalConfig(
            dataset_name="cifar10",
            subset_size=128,
            epochs=1,
            num_seeds=2,
            compute_confusion_matrix=True,
            compute_per_class_metrics=True,
            compute_auc_scores=True,
            track_learning_curves=True,
            metrics_to_track=[
                "accuracy",
                "f1_macro",
                "f1_weighted",
                "precision_macro",
                "recall_macro",
            ],
        )
        self.evaluator = EnhancedQuickEvaluator(self.config)

        # Create a simple mock model
        self.mock_model = MagicMock()
        self.mock_model.parameters.return_value = [MagicMock(grad=None)]
        self.mock_model.named_children.return_value = []

    def test_configuration_enhancement(self):
        """Test that QuickEvalConfig includes all enhanced options."""
        # Test default values
        config = QuickEvalConfig()
        self.assertEqual(config.metrics_to_track, ["accuracy", "loss"])
        self.assertFalse(config.compute_confusion_matrix)
        self.assertFalse(config.compute_per_class_metrics)
        self.assertFalse(config.compute_auc_scores)
        self.assertFalse(config.track_learning_curves)

        # Test enhanced values
        self.assertTrue(self.config.compute_confusion_matrix)
        self.assertTrue(self.config.compute_per_class_metrics)
        self.assertTrue(self.config.compute_auc_scores)
        self.assertTrue(self.config.track_learning_curves)

    def test_classification_metrics_computation(self):
        """Test F1, precision, recall computation."""
        # Create test data
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 2, 1, 0, 0, 1, 0, 1, 2])

        # Compute metrics
        metrics = self.evaluator._compute_classification_metrics(y_pred, y_true)

        # Verify metrics are computed
        self.assertIn("f1_macro", metrics)
        self.assertIn("f1_weighted", metrics)
        self.assertIn("precision_macro", metrics)
        self.assertIn("recall_macro", metrics)
        self.assertIn("accuracy", metrics)

        # Verify metric values are reasonable
        self.assertGreaterEqual(metrics["f1_macro"], 0.0)
        self.assertLessEqual(metrics["f1_macro"], 1.0)
        self.assertGreaterEqual(metrics["precision_macro"], 0.0)
        self.assertLessEqual(metrics["precision_macro"], 1.0)

    def test_auc_metrics_computation(self):
        """Test AUC score computation including PR-AUC."""
        # Binary classification test
        y_true_binary = np.array([0, 1, 1, 0, 1, 0, 1, 0])
        y_prob_binary = np.array([0.1, 0.9, 0.8, 0.2, 0.7, 0.3, 0.95, 0.05])

        metrics = self.evaluator._compute_auc_metrics(
            np.column_stack([1 - y_prob_binary, y_prob_binary]), y_true_binary
        )

        if SKLEARN_AVAILABLE:
            self.assertIn("auc_binary", metrics)
            self.assertGreaterEqual(metrics["auc_binary"], 0.0)
            self.assertLessEqual(metrics["auc_binary"], 1.0)

            # Test PR-AUC for binary classification
            self.assertIn("pr_auc_binary", metrics)
            self.assertGreaterEqual(metrics["pr_auc_binary"], 0.0)
            self.assertLessEqual(metrics["pr_auc_binary"], 1.0)

        # Multi-class test
        y_true_multi = np.array([0, 1, 2, 0, 1, 2])
        y_prob_multi = np.array(
            [
                [0.7, 0.2, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
                [0.6, 0.3, 0.1],
                [0.2, 0.7, 0.1],
                [0.1, 0.2, 0.7],
            ]
        )

        metrics = self.evaluator._compute_auc_metrics(y_prob_multi, y_true_multi)

        if SKLEARN_AVAILABLE:
            self.assertIn("auc_macro", metrics)
            self.assertGreaterEqual(metrics["auc_macro"], 0.0)
            self.assertLessEqual(metrics["auc_macro"], 1.0)

            # Test per-class AUC
            self.assertIn("per_class_auc", metrics)
            per_class_auc = metrics["per_class_auc"]
            self.assertIn("0", per_class_auc)
            self.assertIn("1", per_class_auc)
            self.assertIn("2", per_class_auc)

            # Verify per-class AUC structure
            for class_id in ["0", "1", "2"]:
                class_metrics = per_class_auc[class_id]
                self.assertIn("roc_auc", class_metrics)
                self.assertIn("fpr", class_metrics)
                self.assertIn("tpr", class_metrics)
                self.assertIn("thresholds", class_metrics)
                self.assertGreaterEqual(class_metrics["roc_auc"], 0.0)
                self.assertLessEqual(class_metrics["roc_auc"], 1.0)

    def test_per_class_roc_auc(self):
        """Test per-class ROC AUC computation."""
        # Multi-class test data
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        y_prob = np.array(
            [
                [0.8, 0.1, 0.1],  # Class 0
                [0.1, 0.8, 0.1],  # Class 1
                [0.1, 0.1, 0.8],  # Class 2
                [0.7, 0.2, 0.1],  # Class 0
                [0.2, 0.7, 0.1],  # Class 1
                [0.1, 0.2, 0.7],  # Class 2
                [0.9, 0.05, 0.05],  # Class 0
                [0.05, 0.9, 0.05],  # Class 1
                [0.05, 0.05, 0.9],  # Class 2
            ]
        )

        per_class_metrics = self.evaluator._compute_per_class_roc_auc(y_prob, y_true)

        if SKLEARN_AVAILABLE:
            # Verify all classes are present
            self.assertEqual(len(per_class_metrics), 3)

            for class_id in ["0", "1", "2"]:
                self.assertIn(class_id, per_class_metrics)
                class_data = per_class_metrics[class_id]

                # Verify required fields
                self.assertIn("roc_auc", class_data)
                self.assertIn("fpr", class_data)
                self.assertIn("tpr", class_data)
                self.assertIn("thresholds", class_data)

                # Verify data types and ranges
                self.assertIsInstance(class_data["roc_auc"], float)
                self.assertGreaterEqual(class_data["roc_auc"], 0.0)
                self.assertLessEqual(class_data["roc_auc"], 1.0)

                # Verify arrays are numpy arrays or lists
                self.assertIsInstance(class_data["fpr"], (list, np.ndarray))
                self.assertIsInstance(class_data["tpr"], (list, np.ndarray))
                self.assertIsInstance(class_data["thresholds"], (list, np.ndarray))

                # Verify ROC curve properties
                fpr = np.array(class_data["fpr"])
                tpr = np.array(class_data["tpr"])

                # FPR and TPR should be between 0 and 1
                self.assertTrue(np.all(fpr >= 0) and np.all(fpr <= 1))
                self.assertTrue(np.all(tpr >= 0) and np.all(tpr <= 1))

                # ROC curve should start at (0,0) and end at (1,1)
                self.assertEqual(fpr[0], 0.0)
                self.assertEqual(tpr[0], 0.0)
                self.assertEqual(fpr[-1], 1.0)
                self.assertEqual(tpr[-1], 1.0)
        else:
            # When sklearn is not available, should return empty dict
            self.assertEqual(per_class_metrics, {})

    def test_pr_auc_computation(self):
        """Test Precision-Recall AUC computation for binary classification."""
        # Binary classification with imbalanced data
        y_true = np.array([0, 0, 0, 0, 1, 1, 1])  # 4 negatives, 3 positives
        y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8])

        # Test with sklearn available
        if SKLEARN_AVAILABLE:
            with patch("sklearn.metrics.average_precision_score") as mock_ap:
                mock_ap.return_value = 0.75

                metrics = self.evaluator._compute_auc_metrics(
                    np.column_stack([1 - y_prob, y_prob]), y_true
                )

                self.assertIn("pr_auc_binary", metrics)
                self.assertEqual(metrics["pr_auc_binary"], 0.75)
                mock_ap.assert_called_once()

    def test_roc_curve_data_structure(self):
        """Test that ROC curve data has correct structure."""
        # Simple test case
        y_true = np.array([0, 1, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.8, 0.2])

        per_class_auc = self.evaluator._compute_per_class_roc_auc(
            np.column_stack([1 - y_prob, y_prob]), y_true
        )

        if SKLEARN_AVAILABLE and per_class_auc:
            # Check class 1 (positive class)
            class_1_data = per_class_auc["1"]

            # Verify all required fields exist
            required_fields = ["roc_auc", "fpr", "tpr", "thresholds"]
            for field in required_fields:
                self.assertIn(field, class_1_data)

            # Verify data lengths are consistent
            fpr_len = len(class_1_data["fpr"])
            tpr_len = len(class_1_data["tpr"])
            thresholds_len = len(class_1_data["thresholds"])

            # FPR and TPR should have same length, thresholds may differ
            self.assertEqual(fpr_len, tpr_len)
            self.assertGreater(thresholds_len, 0)

            # Verify thresholds are sorted in descending order
            thresholds = class_1_data["thresholds"]
            self.assertEqual(thresholds, sorted(thresholds, reverse=True))

    def test_fallback_auc_behavior(self):
        """Test fallback behavior when sklearn functions fail."""
        # Test with invalid probability data
        y_true = np.array([0, 1, 2])
        y_prob = np.array([[0.5, 0.3, 0.2], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])

        # Mock sklearn functions to raise exceptions
        with patch(
            "sklearn.metrics.roc_auc_score", side_effect=Exception("Mock error")
        ):
            metrics = self.evaluator._compute_auc_metrics(y_prob, y_true)

            # Should return empty dict when computation fails
            self.assertEqual(metrics, {})

    def test_per_class_analysis(self):
        """Test per-class performance analysis."""
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 2, 1, 0, 0, 1, 0, 1, 2])

        result = self.evaluator._compute_per_class_analysis(y_pred, y_true)

        # Verify structure
        self.assertIn("confusion_matrix", result)
        self.assertIn("per_class_metrics", result)
        self.assertIn("class_names", result)

        # Verify confusion matrix
        cm = result["confusion_matrix"]
        self.assertEqual(len(cm), 3)  # 3 classes
        self.assertEqual(len(cm[0]), 3)  # 3x3 matrix

        # Verify per-class metrics
        per_class = result["per_class_metrics"]
        for class_id in [0, 1, 2]:
            self.assertIn(class_id, per_class)
            self.assertIn("precision", per_class[class_id])
            self.assertIn("recall", per_class[class_id])
            self.assertIn("f1", per_class[class_id])
            self.assertIn("support", per_class[class_id])

    def test_predictions_collection(self):
        """Test prediction and target collection."""
        # Mock torch operations
        with patch("torch.no_grad"), patch.object(self.evaluator, "device", "cpu"):
            # Mock model output
            mock_outputs = MagicMock()
            mock_outputs.argmax.return_value = torch.tensor([0, 1, 0, 1])
            mock_outputs.softmax.return_value = torch.tensor(
                [[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.4, 0.6]]
            )

            # Mock data loader
            mock_loader = [(torch.randn(4, 3, 32, 32), torch.tensor([0, 1, 0, 1]))]

            predictions, targets, probabilities = (
                self.evaluator._collect_predictions_and_targets(
                    self.mock_model, mock_loader
                )
            )

            self.assertEqual(len(predictions), 4)
            self.assertEqual(len(targets), 4)
            self.assertEqual(len(probabilities), 4)

    def test_enhanced_evaluation(self):
        """Test enhanced evaluation pipeline."""
        with (
            patch.object(self.evaluator, "_evaluate") as mock_evaluate,
            patch.object(
                self.evaluator, "_collect_predictions_and_targets"
            ) as mock_collect,
            patch.object(
                self.evaluator, "_compute_classification_metrics"
            ) as mock_class_metrics,
            patch.object(self.evaluator, "_compute_auc_metrics") as mock_auc_metrics,
            patch.object(
                self.evaluator, "_compute_per_class_analysis"
            ) as mock_per_class,
        ):
            # Mock return values
            mock_evaluate.return_value = {"val_accuracy": 0.85, "val_loss": 0.5}
            mock_collect.return_value = (
                np.array([0, 1]),
                np.array([0, 1]),
                np.array([[0.8, 0.2], [0.3, 0.7]]),
            )
            mock_class_metrics.return_value = {"f1_macro": 0.8}
            mock_auc_metrics.return_value = {"auc_binary": 0.9}
            mock_per_class.return_value = {"confusion_matrix": [[1, 0], [0, 1]]}

            # Mock criterion
            mock_criterion = MagicMock()

            result = self.evaluator._enhanced_evaluate(
                self.mock_model, MagicMock(), mock_criterion
            )

            # Verify enhanced metrics are included
            self.assertEqual(result["val_accuracy"], 0.85)
            self.assertEqual(result["f1_macro"], 0.8)
            self.assertEqual(result["auc_binary"], 0.9)
            self.assertIn("confusion_matrix", result)

    def test_learning_dynamics_tracking(self):
        """Test learning curves and dynamics tracking."""
        # Test disabled tracking
        self.config.track_learning_curves = False
        result = self.evaluator._track_learning_dynamics(
            self.mock_model, MagicMock(), MagicMock(), MagicMock(), 3
        )
        self.assertEqual(result, {})

        # Test enabled tracking
        self.config.track_learning_curves = True
        with (
            patch.object(self.evaluator, "_train_one_epoch") as mock_train,
            patch.object(self.evaluator, "_evaluate") as mock_eval,
        ):
            mock_train.return_value = {"loss": 0.5, "accuracy": 0.8}
            mock_eval.return_value = {"val_loss": 0.6, "val_accuracy": 0.75}

            result = self.evaluator._track_learning_dynamics(
                self.mock_model, MagicMock(), MagicMock(), MagicMock(), 3
            )

            # Verify structure
            expected_keys = ["train_loss", "train_accuracy", "val_loss", "val_accuracy"]
            for key in expected_keys:
                self.assertIn(key, result)
                self.assertEqual(len(result[key]), 3)

    def test_aggregate_results_enhanced(self):
        """Test enhanced results aggregation."""
        # Mock seed results with enhanced metrics
        seed_results = [
            {
                "final": {
                    "val_accuracy": 0.85,
                    "val_loss": 0.5,
                    "f1_macro": 0.8,
                    "precision_macro": 0.82,
                }
            },
            {
                "final": {
                    "val_accuracy": 0.87,
                    "val_loss": 0.45,
                    "f1_macro": 0.85,
                    "precision_macro": 0.84,
                }
            },
        ]

        aggregated = self.evaluator._aggregate_results(seed_results)

        # Verify basic metrics
        self.assertIn("val_accuracy", aggregated)
        self.assertIn("val_loss", aggregated)

        # Verify enhanced metrics
        self.assertIn("f1_macro", aggregated)
        self.assertIn("precision_macro", aggregated)

        # Verify statistics
        self.assertIn("mean", aggregated["val_accuracy"])
        self.assertIn("std", aggregated["val_accuracy"])
        self.assertIn("ci_95", aggregated["val_accuracy"])

    def test_configuration_defaults(self):
        """Test that enhanced configuration has sensible defaults."""
        config = QuickEvalConfig()

        # Test metrics tracking defaults
        self.assertIn("accuracy", config.metrics_to_track)
        self.assertIn("f1_macro", config.metrics_to_track)
        self.assertIn("f1_weighted", config.metrics_to_track)
        self.assertIn("precision_macro", config.metrics_to_track)
        self.assertIn("recall_macro", config.metrics_to_track)

        # Test analysis options defaults
        self.assertTrue(config.compute_confidence_intervals)
        self.assertEqual(config.confidence_level, 0.95)

    def test_fallback_behavior(self):
        """Test fallback behavior when sklearn is not available."""
        with patch("pytorch_tools.quick_evaluator.SKLEARN_AVAILABLE", False):
            # Re-import to get fallback implementations
            import importlib

            import pytorch_tools.quick_evaluator as qe

            importlib.reload(qe)

            evaluator = qe.EnhancedQuickEvaluator(self.config)

            # Test fallback F1 score
            y_true = [0, 1, 2, 0, 1, 2]
            y_pred = [0, 2, 1, 0, 0, 1]

            metrics = evaluator._compute_classification_metrics(
                np.array(y_pred), np.array(y_true)
            )

            # Should still compute something (fallback)
            self.assertIn("accuracy", metrics)
            self.assertGreaterEqual(metrics["accuracy"], 0.0)
            self.assertLessEqual(metrics["accuracy"], 1.0)

    def test_end_to_end_evaluation(self):
        """Test complete evaluation pipeline."""
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch.object(self.evaluator, "_run_single_seed_evaluation") as mock_run,
        ):
            # Mock single seed evaluation
            mock_run.return_value = {
                "seed": 42,
                "history": [],
                "final": {
                    "val_accuracy": 0.85,
                    "f1_macro": 0.83,
                    "precision_macro": 0.82,
                },
                "best_val_accuracy": 0.85,
            }

            result = self.evaluator.quick_evaluate(self.mock_model, "TestModel")

            # Verify result structure
            self.assertIn("config", result)
            self.assertIn("model_name", result)
            self.assertIn("num_seeds", result)
            self.assertIn("seed_results", result)
            self.assertIn("aggregated", result)
            self.assertIn("final", result)

            # Verify model name
            self.assertEqual(result["model_name"], "TestModel")

            # Verify number of seeds
            self.assertEqual(result["num_seeds"], 2)

    def test_metrics_saving(self):
        """Test that metrics can be saved to file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_path = f.name

        try:
            config = QuickEvalConfig(save_metrics_path=temp_path)
            evaluator = EnhancedQuickEvaluator(config)

            with patch.object(evaluator, "_run_single_seed_evaluation") as mock_run:
                mock_run.return_value = {
                    "seed": 42,
                    "history": [],
                    "final": {"val_accuracy": 0.85},
                    "best_val_accuracy": 0.85,
                }

                result = evaluator.quick_evaluate(self.mock_model)

                # Verify file was created and contains valid JSON
                with open(temp_path, "r") as read_f:
                    saved_data = json.load(read_f)

                self.assertIn("model_name", saved_data)
                self.assertIn("val_accuracy", saved_data.get("final", {}))

        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)


if __name__ == "__main__":
    # Add torch imports for testing
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset

        HAS_TORCH = True
    except ImportError:
        HAS_TORCH = False
        print("Warning: PyTorch not available. Some tests will be skipped.")

    unittest.main()
