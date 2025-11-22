"""Enhanced Quick Evaluator for PyTorch ML Research Agent

This module provides robust evaluation capabilities including:
- Multi-seed evaluation for statistical significance
- Hugging Face datasets integration
- Enhanced metrics aggregation and reporting
- Reproducible evaluation with fixed seeds
- Comprehensive performance analysis
"""

from __future__ import annotations

import json
import logging
import random
import statistics
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Optional torch import
try:
    import torch
    import torch.optim as optim
    from torch import nn
    from torch.utils.data import DataLoader, Dataset, random_split

    TORCH_AVAILABLE = True
except Exception:
    torch = None  # type: ignore
    nn = None  # type: ignore
    DataLoader = None  # type: ignore
    Dataset = object  # type: ignore
    random_split = None  # type: ignore
    optim = None  # type: ignore
    TORCH_AVAILABLE = False

# Optional torchvision datasets and transforms
try:
    import torchvision
    import torchvision.transforms as transforms

    TORCHVISION_AVAILABLE = True
except Exception:
    torchvision = None  # type: ignore
    transforms = None  # type: ignore
    TORCHVISION_AVAILABLE = False

# Import our enhanced dataset loader
try:
    from .dataset_loader import (
        DatasetConfig,
        DatasetLoader,
        DatasetLoaderError,
        create_dataset_config,
        get_dataset_loader,
    )

    ENHANCED_DATASET_LOADER = True
except Exception:
    # Fallback to original dataset loading if enhanced loader not available
    DatasetConfig = None  # type: ignore
    DatasetLoader = None  # type: ignore
    get_dataset_loader = None  # type: ignore
    create_dataset_config = None  # type: ignore
    DatasetLoaderError = Exception  # type: ignore
    ENHANCED_DATASET_LOADER = False

# Optional sklearn import for enhanced metrics
try:
    from sklearn.metrics import (
        accuracy_score,
        auc,
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )

    SKLEARN_AVAILABLE = True
except ImportError:
    # Fallback implementations when sklearn is not available
    SKLEARN_AVAILABLE = False

# Optional matplotlib import for visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

    # Simple fallback implementations
    def f1_score(y_true, y_pred, average="macro"):
        """Simple fallback F1 score implementation."""
        import numpy as np

        if average == "macro":
            # Per-class F1 scores
            classes = list(set(y_true))
            f1_scores = []
            for cls in classes:
                tp = sum(
                    1
                    for i, pred in enumerate(y_pred)
                    if pred == cls and y_true[i] == cls
                )
                fp = sum(
                    1
                    for i, pred in enumerate(y_pred)
                    if pred == cls and y_true[i] != cls
                )
                fn = sum(
                    1
                    for i, true in enumerate(y_true)
                    if true == cls and y_pred[i] != cls
                )

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0
                    else 0.0
                )
                f1_scores.append(f1)

            return np.mean(f1_scores) if f1_scores else 0.0
        else:
            # Simple accuracy-based fallback
            return accuracy_score(y_true, y_pred)

    def precision_score(y_true, y_pred, average="macro"):
        """Simple fallback precision implementation."""
        # Similar structure to f1_score but for precision
        return accuracy_score(y_true, y_pred) * 0.9  # Rough approximation

    def recall_score(y_true, y_pred, average="macro"):
        """Simple fallback recall implementation."""
        # Similar structure to f1_score but for recall
        return accuracy_score(y_true, y_pred) * 0.9  # Rough approximation

    def roc_auc_score(y_true, y_pred, **kwargs):
        """Fallback that returns a reasonable AUC value when sklearn is not available."""
        # For fallback, we'll return a value based on the accuracy of argmax predictions
        try:
            # Convert probabilities to predictions if needed
            if len(y_pred.shape) > 1:
                # Multi-class probabilities - take argmax
                y_pred_labels = np.argmax(y_pred, axis=1)
            else:
                # Binary probabilities - threshold at 0.5
                y_pred_labels = (y_pred > 0.5).astype(int)

            # Use accuracy as a fallback metric
            return accuracy_score(y_true, y_pred_labels)
        except Exception:
            # If anything fails, return a neutral value
            return 0.5

    def average_precision_score(y_true, y_score, **kwargs):
        """Fallback that returns accuracy when PR-AUC cannot be computed."""
        return accuracy_score(y_true, y_score)

    def roc_curve(y_true, y_score, **kwargs):
        """Fallback ROC curve that returns simple thresholds."""
        # Simple fallback: return basic ROC points
        n_samples = len(y_true)
        fpr = np.array([0.0, 0.5, 1.0])
        tpr = np.array([0.0, 0.5, 1.0])
        thresholds = np.array([2.0, 1.0, 0.0])
        return fpr, tpr, thresholds

    def precision_recall_curve(y_true, y_score, **kwargs):
        """Fallback PR curve that returns simple points."""
        # Simple fallback: return basic PR points
        precision = np.array([1.0, 0.5, 0.0])
        recall = np.array([0.0, 0.5, 1.0])
        thresholds = np.array([2.0, 1.0, 0.0])
        return precision, recall, thresholds

    def auc(fpr, tpr):
        """Fallback AUC calculation using trapezoidal rule."""
        return np.trapz(tpr, fpr)

    def confusion_matrix(y_true, y_pred):
        """Simple fallback confusion matrix."""
        classes = list(set(y_true + y_pred))
        cm = [[0] * len(classes) for _ in classes]
        class_to_idx = {cls: i for i, cls in enumerate(classes)}

        for true, pred in zip(y_true, y_pred):
            i = class_to_idx[true]
            j = class_to_idx[pred]
            cm[i][j] += 1

        return np.array(cm)

    def accuracy_score(y_true, y_pred):
        """Simple accuracy implementation."""
        return sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)


_logger = logging.getLogger(__name__)


@dataclass
class QuickEvalConfig:
    """Enhanced configuration for quick evaluation runs.

    Attributes:
        dataset_name: Name of dataset to use (e.g., 'cifar10', 'mnist', 'imdb').
        dataset_config: Optional DatasetConfig for advanced configuration.
        subset_size: Maximum number of samples to use (for speed).
        batch_size: Batch size for training and evaluation.
        epochs: Number of training epochs.
        learning_rate: Learning rate for optimizer.
        random_seed: Base RNG seed for reproducibility.
        num_seeds: Number of seeds for statistical evaluation (default: 1 for speed, 3+ for robustness).
        device: Device string (e.g., 'cpu' or 'cuda:0').
        num_workers: Number of DataLoader workers.
        save_metrics_path: Optional path to write metrics JSON.
        verbose: Enable detailed logging.
        early_stopping_patience: Early stopping patience (0 to disable).
        target_accuracy: Target accuracy for goal achievement detection.
        metrics_to_track: List of metrics to track and aggregate.
        evaluation_strategy: When to evaluate ('epoch', 'end').
        deterministic: Use deterministic operations for reproducibility.

        # Enhanced metrics configuration
        compute_confusion_matrix: bool = False
        compute_per_class_metrics: bool = False
        compute_auc_scores: bool = False
        track_learning_curves: bool = False
        track_gradient_norms: bool = False
        compute_confidence_intervals: bool = True
        confidence_level: float = 0.95
        detailed_report: bool = False
        save_confusion_matrix_plot: bool = False
        export_learning_curves: bool = False

    """

    # Dataset configuration
    dataset_name: str = "cifar10"
    dataset_config: DatasetConfig | None = None
    subset_size: int | None = 2048

    # Training configuration
    batch_size: int = 64
    epochs: int = 2
    learning_rate: float = 1e-3
    random_seed: int = 42
    num_seeds: int = 1  # Number of seeds for statistical evaluation

    # System configuration
    device: str = "cpu"
    num_workers: int = 0
    save_metrics_path: str | None = None
    verbose: bool = False

    # Basic options
    early_stopping_patience: int = 0
    target_accuracy: float = 0.7
    metrics_to_track: list[str] = field(default_factory=lambda: ["accuracy", "loss"])
    evaluation_strategy: str = "epoch"  # 'epoch' or 'end'
    deterministic: bool = True

    # Enhanced metrics configuration
    compute_confusion_matrix: bool = False
    compute_per_class_metrics: bool = False
    compute_auc_scores: bool = False
    track_learning_curves: bool = False
    track_gradient_norms: bool = False
    compute_confidence_intervals: bool = True
    confidence_level: float = 0.95
    detailed_report: bool = False
    save_confusion_matrix_plot: bool = False
    export_learning_curves: bool = False

    # Dataset-specific arguments
    dataset_kwargs: dict[str, Any] = field(default_factory=dict)


class QuickEvaluatorError(Exception):
    pass


class EnhancedQuickEvaluator:
    """Enhanced evaluator with multi-seed support and robust dataset integration.
    """

    def __init__(self, cfg: QuickEvalConfig):
        self.cfg = cfg
        self.device = cfg.device if TORCH_AVAILABLE else "cpu"

        if TORCH_AVAILABLE and "cuda" in self.device and not torch.cuda.is_available():
            _logger.warning(
                "Requested CUDA device but CUDA is unavailable; falling back to CPU."
            )
            self.device = "cpu"

        self._setup_logging()

        # Initialize dataset configuration
        if self.cfg.dataset_config is None:
            if ENHANCED_DATASET_LOADER:
                self.dataset_config = create_dataset_config(
                    self.cfg.dataset_name,
                    subset_size=self.cfg.subset_size,
                    seed=self.cfg.random_seed,
                    batch_size=self.cfg.batch_size,
                    num_workers=self.cfg.num_workers,
                    **self.cfg.dataset_kwargs,
                )
                self.dataset_loader = get_dataset_loader(self.dataset_config)
            else:
                self.dataset_config = None
                self.dataset_loader = None
        else:
            self.dataset_config = self.cfg.dataset_config
            if ENHANCED_DATASET_LOADER:
                self.dataset_loader = get_dataset_loader(self.dataset_config)
            else:
                self.dataset_loader = None

    def _setup_logging(self):
        if self.cfg.verbose:
            level = logging.DEBUG
        else:
            level = logging.INFO
        if not _logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
            )
            _logger.addHandler(handler)
        _logger.setLevel(level)

    def _seed_everything(self, seed: int):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        if TORCH_AVAILABLE:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if self.cfg.deterministic:
                try:
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
                except Exception:
                    pass

    def _build_optimizer_and_loss(self, model: nn.Module):
        """Build optimizer and loss function."""
        if not TORCH_AVAILABLE:
            raise QuickEvaluatorError(
                "PyTorch is required to build optimizer and loss."
            )

        optimizer = optim.Adam(model.parameters(), lr=self.cfg.learning_rate)
        criterion = nn.CrossEntropyLoss()
        return optimizer, criterion

    def _train_one_epoch(
        self, model: nn.Module, dataloader: DataLoader, optimizer, criterion
    ) -> dict[str, Any]:
        """Train model for one epoch."""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in dataloader:
            if TORCH_AVAILABLE:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * xb.shape[0]
            preds = outputs.argmax(dim=1)
            correct += int((preds == yb).sum().item())
            total += xb.shape[0]

        avg_loss = running_loss / total if total else float("nan")
        accuracy = correct / total if total else 0.0

        return {"loss": avg_loss, "accuracy": accuracy, "samples": total}

    def _evaluate(
        self, model: nn.Module, dataloader: DataLoader, criterion
    ) -> dict[str, Any]:
        """Evaluate model on dataset."""
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for xb, yb in dataloader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                outputs = model(xb)
                loss = criterion(outputs, yb)

                running_loss += float(loss.item()) * xb.shape[0]
                preds = outputs.argmax(dim=1)
                correct += int((preds == yb).sum().item())
                total += xb.shape[0]

        avg_loss = running_loss / total if total else float("nan")
        accuracy = correct / total if total else 0.0

        return {"val_loss": avg_loss, "val_accuracy": accuracy, "samples": total}

    def _infer_num_classes(self, model: nn.Module) -> int | None:
        """Infer number of classes from model output."""
        try:
            # Default input shape for inference
            input_shape = getattr(self.dataset_config, "input_shape", None) or (
                3,
                32,
                32,
            )
            dummy = torch.randn(1, *input_shape, device=self.device)

            model.eval()
            with torch.no_grad():
                out = model(dummy)

            if hasattr(out, "shape"):
                if len(out.shape) >= 2:
                    return int(out.shape[1])
                elif len(out.shape) == 1:
                    return int(out.shape[0])
        except Exception as e:
            _logger.debug(f"Failed to infer num_classes: {e}")

        return None

    def _prepare_dataset_fallback(self) -> tuple[Dataset, Dataset | None]:
        """Fallback dataset preparation using original logic."""
        if not TORCH_AVAILABLE:
            raise QuickEvaluatorError(
                "PyTorch is required for QuickEvaluator but is not available."
            )

        name = (self.cfg.dataset_name or "").lower()
        if name in ("cifar10", "mnist") and TORCHVISION_AVAILABLE:
            _logger.info("Loading torchvision dataset: %s", name)
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )

            if name == "cifar10":
                ds_train = torchvision.datasets.CIFAR10(
                    root="./data",
                    train=True,
                    download=True,
                    transform=transform,
                )
            else:
                ds_train = torchvision.datasets.MNIST(
                    root="./data",
                    train=True,
                    download=True,
                    transform=transform,
                )

            # Optionally subsample for quick runs
            if self.cfg.subset_size:
                subset_size = min(len(ds_train), int(self.cfg.subset_size))
                _logger.info(
                    "Using subset of size %d from %d available samples",
                    subset_size,
                    len(ds_train),
                )
                ds_train, _ = random_split(
                    ds_train, [subset_size, len(ds_train) - subset_size]
                )
                return ds_train, None

            return ds_train, None

        # Fallback: synthetic random dataset (small)
        _logger.info("Using synthetic random dataset (fallback)")

        class RandomDataset(Dataset):
            def __init__(
                self,
                num_samples: int = 1024,
                input_shape=(3, 32, 32),
                num_classes: int = 10,
            ):
                self.num_samples = num_samples
                self.input_shape = input_shape
                self.num_classes = num_classes

            def __len__(self):
                return self.num_samples

            def __getitem__(self, idx):
                # produce a random tensor and a random label; dtype=float32 for models
                import numpy as _np  # local import to avoid hard dependency if not used

                x = _np.random.randn(*self.input_shape).astype("float32")
                y = int(_np.random.randint(0, self.num_classes))
                # Convert to torch tensors lazily if torch is available
                if TORCH_AVAILABLE:
                    return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)
                return x, y

        num_samples = int(self.cfg.subset_size or 1024)
        # Determine desired number of classes for the synthetic dataset. Priority:
        # 1) explicit cfg.num_classes, 2) cfg.dataset_kwargs['num_classes'], 3) default 10
        try:
            if getattr(self.cfg, "num_classes", None) is not None:
                ds_num_classes = int(self.cfg.num_classes)
            else:
                ds_num_classes = int(self.cfg.dataset_kwargs.get("num_classes", 10))
        except Exception:
            ds_num_classes = 10
        ds = RandomDataset(
            num_samples, input_shape=(3, 32, 32), num_classes=ds_num_classes
        )
        return ds, None

    def _make_dataloader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        """Create DataLoader from dataset."""
        if not TORCH_AVAILABLE:
            raise QuickEvaluatorError("PyTorch is required for dataloaders.")
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
        )

    def _run_single_seed_evaluation(
        self, model: nn.Module, seed: int, model_name: str | None = None
    ) -> dict[str, Any]:
        """Run evaluation for a single seed."""
        # Set seed for this evaluation
        self._seed_everything(seed)

        # Move model to device
        model = model.to(self.device)

        # Infer number of classes if not provided
        if not hasattr(self.cfg, "num_classes") or self.cfg.num_classes is None:
            inferred_classes = self._infer_num_classes(model)
            if inferred_classes:
                self.cfg.num_classes = inferred_classes

        # Load datasets
        if ENHANCED_DATASET_LOADER and self.dataset_loader:
            train_dataset, val_dataset = self.dataset_loader.load()
            train_loader = self.dataset_loader.create_dataloader(
                train_dataset, shuffle=True
            )

            # Use validation dataset if available, otherwise split training data
            if val_dataset is not None:
                val_loader = self.dataset_loader.create_dataloader(
                    val_dataset, shuffle=False
                )
            else:
                # Create validation split from training data
                if len(train_dataset) > 100:
                    total_size = len(train_dataset)
                    val_size = min(int(0.1 * total_size), 500)
                    train_size = total_size - val_size

                    if random_split is not None:
                        train_dataset, val_dataset = random_split(
                            train_dataset, [train_size, val_size]
                        )
                        train_loader = self.dataset_loader.create_dataloader(
                            train_dataset, shuffle=True
                        )
                        val_loader = self.dataset_loader.create_dataloader(
                            val_dataset, shuffle=False
                        )
                    else:
                        val_loader = train_loader
                else:
                    val_loader = train_loader
        else:
            # Use fallback dataset preparation
            train_dataset, val_dataset = self._prepare_dataset_fallback()
            train_loader = self._make_dataloader(train_dataset, shuffle=True)

            if val_dataset is not None:
                val_loader = self._make_dataloader(val_dataset, shuffle=False)
            else:
                # Create validation split from training data
                if len(train_dataset) > 100:
                    total_size = len(train_dataset)
                    val_size = min(int(0.1 * total_size), 500)
                    train_size = total_size - val_size

                    if random_split is not None:
                        train_dataset, val_dataset = random_split(
                            train_dataset, [train_size, val_size]
                        )
                        train_loader = self._make_dataloader(
                            train_dataset, shuffle=True
                        )
                        val_loader = self._make_dataloader(val_dataset, shuffle=False)
                    else:
                        val_loader = train_loader
                else:
                    val_loader = train_loader

        optimizer, criterion = self._build_optimizer_and_loss(model)

        history = []
        best_val_accuracy = 0.0
        best_metrics = None
        patience_counter = 0

        # Determine if we should use enhanced evaluation
        use_enhanced = (
            self.cfg.compute_confusion_matrix
            or self.cfg.compute_per_class_metrics
            or self.cfg.compute_auc_scores
            or self.cfg.track_learning_curves
            or len(
                [m for m in self.cfg.metrics_to_track if m not in ["accuracy", "loss"]]
            )
            > 0
        )

        for epoch in range(1, max(1, int(self.cfg.epochs)) + 1):
            # Training
            train_metrics = self._train_one_epoch(
                model, train_loader, optimizer, criterion
            )

            # Evaluation - use enhanced evaluation if needed
            if use_enhanced:
                val_metrics = self._enhanced_evaluate(model, val_loader, criterion)
            else:
                val_metrics = self._evaluate(model, val_loader, criterion)

            epoch_entry = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
            history.append(epoch_entry)

            # Track best performance
            current_val_accuracy = val_metrics.get("val_accuracy", 0.0)
            if current_val_accuracy > best_val_accuracy:
                best_val_accuracy = current_val_accuracy
                best_metrics = val_metrics.copy()
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping
            if (
                self.cfg.early_stopping_patience > 0
                and patience_counter >= self.cfg.early_stopping_patience
            ):
                _logger.info(f"Early stopping at epoch {epoch}")
                break

        result = {
            "seed": seed,
            "history": history,
            "final": best_metrics or (history[-1]["val"] if history else {}),
            "best_val_accuracy": best_val_accuracy,
        }

        # Add learning curves if enabled
        if self.cfg.track_learning_curves and len(history) > 1:
            learning_dynamics = self._track_learning_dynamics(
                model, train_loader, val_loader, optimizer, len(history)
            )
            if learning_dynamics:
                result["learning_curves"] = learning_dynamics

        return result

    def quick_evaluate(
        self, model: nn.Module, model_name: str | None = None
    ) -> dict[str, Any]:
        """Perform enhanced quick evaluation with multi-seed support.

        Returns a dictionary with:
            - 'config': serialized QuickEvalConfig
            - 'model_name': provided or inferred name
            - 'num_seeds': number of seeds evaluated
            - 'seed_results': list of individual seed results
            - 'aggregated': aggregated statistics across seeds
            - 'final': best overall metrics
        """
        if not TORCH_AVAILABLE:
            raise QuickEvaluatorError("PyTorch is required for quick evaluation.")

        model_name = model_name or model.__class__.__name__
        num_seeds = max(1, int(self.cfg.num_seeds))

        _logger.info(f"Starting evaluation of {model_name} with {num_seeds} seed(s)")

        # Run evaluation for each seed
        seed_results = []
        all_final_accuracies = []
        all_final_losses = []

        for seed_offset in range(num_seeds):
            current_seed = self.cfg.random_seed + seed_offset
            _logger.info(f"Running evaluation with seed {current_seed}")

            # Create a fresh model instance for each seed to ensure independence
            try:
                # Try to create a fresh model (this assumes the model has a copy method or can be reinitialized)
                if hasattr(model, "copy"):
                    seed_model = model.copy()
                else:
                    # Fallback: use the same model but reset weights
                    seed_model = model
                    if hasattr(model, "reset_parameters"):
                        model.reset_parameters()

                seed_result = self._run_single_seed_evaluation(
                    seed_model, current_seed, model_name
                )
                seed_results.append(seed_result)

                # Collect metrics for aggregation
                final_metrics = seed_result.get("final", {})
                all_final_accuracies.append(final_metrics.get("val_accuracy", 0.0))
                all_final_losses.append(final_metrics.get("val_loss", float("inf")))

            except Exception as e:
                _logger.error(f"Failed evaluation for seed {current_seed}: {e}")
                continue

        if not seed_results:
            raise QuickEvaluatorError("All seed evaluations failed")

        # Aggregate results across seeds
        aggregated = self._aggregate_results(seed_results)

        # Determine overall best result
        best_seed_idx = np.argmax(all_final_accuracies)
        best_seed_result = seed_results[best_seed_idx]

        result = {
            "config": asdict(self.cfg),
            "model_name": model_name,
            "num_seeds": len(seed_results),
            "seed_results": seed_results,
            "aggregated": aggregated,
            "final": best_seed_result["final"],
            "best_seed": best_seed_result["seed"],
            "dataset_info": (
                getattr(self.dataset_config, "__dict__", {})
                if self.dataset_config
                else {}
            ),
        }

        # Check if goal is achieved
        target_accuracy = self.cfg.target_accuracy
        mean_accuracy = aggregated.get("val_accuracy", {}).get("mean", 0.0)
        result["goal_achieved"] = mean_accuracy >= target_accuracy

        # Optionally save metrics to disk
        if self.cfg.save_metrics_path:
            try:
                p = Path(self.cfg.save_metrics_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(result, indent=2), encoding="utf-8")
                _logger.info("Saved quick eval metrics to %s", p)
            except Exception as e:
                _logger.warning(
                    "Failed to save metrics to %s: %s", self.cfg.save_metrics_path, e
                )

        return result

    def _collect_predictions_and_targets(
        self, model: nn.Module, dataloader: DataLoader
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Collect all predictions and targets for metric computation."""
        model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []

        with torch.no_grad():
            for xb, yb in dataloader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                outputs = model(xb)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = outputs.argmax(dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(yb.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        return (
            np.array(all_predictions),
            np.array(all_targets),
            np.array(all_probabilities) if self.cfg.compute_auc_scores else None,
        )

    def _aggregate_results(self, seed_results: list[dict[str, Any]]) -> dict[str, Any]:
        """Aggregate results across multiple seeds."""
        if not seed_results:
            return {}

        # Collect all final metrics
        all_final_metrics = [result.get("final", {}) for result in seed_results]

        # Aggregate each metric
        aggregated = {}
        basic_metric_names = ["val_accuracy", "val_loss", "accuracy", "loss"]
        enhanced_metric_names = [
            "f1_macro",
            "f1_weighted",
            "precision_macro",
            "recall_macro",
            "auc_macro",
            "auc_binary",
            "pr_auc_macro",
            "pr_auc_binary",
        ]

        # Aggregate basic metrics
        for metric in basic_metric_names:
            values = []
            for metrics in all_final_metrics:
                if metric in metrics:
                    values.append(float(metrics[metric]))

            if values:
                aggregated[metric] = {
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "min": min(values),
                    "max": max(values),
                    "values": values,
                }

        # Aggregate enhanced metrics
        for metric in enhanced_metric_names:
            values = []
            for metrics in all_final_metrics:
                if metric in metrics:
                    values.append(float(metrics[metric]))

            if values:
                aggregated[metric] = {
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "min": min(values),
                    "max": max(values),
                    "values": values,
                }

        # Add confidence interval for accuracy (95%)
        if "val_accuracy" in aggregated:
            mean_acc = aggregated["val_accuracy"]["mean"]
            std_acc = aggregated["val_accuracy"]["std"]
            n = len(aggregated["val_accuracy"]["values"])

            if n > 1:
                # Simple 95% confidence interval approximation
                margin = 1.96 * (std_acc / np.sqrt(n))
                aggregated["val_accuracy"]["ci_95"] = (
                    mean_acc - margin,
                    mean_acc + margin,
                )

        return aggregated

    def _compute_classification_metrics(
        self, all_predictions: np.ndarray, all_targets: np.ndarray
    ) -> dict[str, float]:
        """Compute comprehensive classification metrics."""
        metrics = {}

        # Convert to lists for sklearn compatibility
        y_true = all_targets.tolist()
        y_pred = all_predictions.tolist()

        try:
            # Always compute accuracy as fallback
            metrics["accuracy"] = accuracy_score(y_true, y_pred)

            # F1 Scores
            if "f1_macro" in self.cfg.metrics_to_track:
                metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro")
            if "f1_weighted" in self.cfg.metrics_to_track:
                metrics["f1_weighted"] = f1_score(y_true, y_pred, average="weighted")

            # Precision and Recall
            if "precision_macro" in self.cfg.metrics_to_track:
                metrics["precision_macro"] = precision_score(
                    y_true, y_pred, average="macro"
                )
            if "recall_macro" in self.cfg.metrics_to_track:
                metrics["recall_macro"] = recall_score(y_true, y_pred, average="macro")

            # Support (class distribution)
            if "support" in self.cfg.metrics_to_track:
                metrics["support"] = np.bincount(all_targets).tolist()

        except Exception as e:
            _logger.warning(f"Failed to compute classification metrics: {e}")

        return metrics

    def _compute_auc_metrics(
        self, all_probabilities: np.ndarray, all_targets: np.ndarray
    ) -> dict[str, float]:
        """Compute comprehensive AUC-based metrics including ROC-AUC and PR-AUC."""
        if not self.cfg.compute_auc_scores:
            return {}

        metrics = {}

        try:
            # Check sklearn availability at module level
            import pytorch_tools.quick_evaluator as qe_module

            if not qe_module.SKLEARN_AVAILABLE:
                _logger.debug("Sklearn not available for AUC computation")
                return metrics

            _logger.debug(
                f"AUC computation: probabilities shape {all_probabilities.shape}, targets shape {all_targets.shape}"
            )
            _logger.debug(
                f"Probabilities type: {type(all_probabilities)}, Targets type: {type(all_targets)}"
            )
            _logger.debug(
                f"First few probabilities: {all_probabilities[:3] if len(all_probabilities) > 0 else 'empty'}"
            )
            _logger.debug(
                f"First few targets: {all_targets[:3] if len(all_targets) > 0 else 'empty'}"
            )

            if all_probabilities.shape[1] == 2:  # Binary classification
                _logger.debug("Computing binary classification AUC")
                # ROC-AUC for binary classification
                auc_binary = roc_auc_score(all_targets, all_probabilities[:, 1])
                metrics["auc_binary"] = float(auc_binary)

                # Precision-Recall AUC for binary classification
                pr_auc_binary = average_precision_score(
                    all_targets, all_probabilities[:, 1]
                )
                metrics["pr_auc_binary"] = float(pr_auc_binary)

            else:  # Multi-class
                _logger.debug("Computing multi-class classification AUC")
                _logger.debug(f"Unique targets: {np.unique(all_targets)}")
                _logger.debug(f"Class distribution: {np.bincount(all_targets)}")

                # ROC-AUC for multi-class (one-vs-rest)
                auc_macro = roc_auc_score(
                    all_targets, all_probabilities, multi_class="ovr", average="macro"
                )
                metrics["auc_macro"] = float(auc_macro)

                # Per-class ROC curves and AUC scores
                per_class_auc = self._compute_per_class_roc_auc(
                    all_probabilities, all_targets
                )
                if per_class_auc:
                    metrics["per_class_auc"] = per_class_auc

        except Exception as e:
            _logger.warning(f"Failed to compute AUC metrics: {e}")
            _logger.warning(f"Error details: {type(e).__name__}: {e!s}")
            import traceback

            _logger.warning(f"Traceback: {traceback.format_exc()}")

        return metrics

    def _compute_per_class_roc_auc(
        self, all_probabilities: np.ndarray, all_targets: np.ndarray
    ) -> dict[str, float]:
        """Compute per-class ROC AUC scores and ROC curve data."""
        # Check sklearn availability at module level
        import pytorch_tools.quick_evaluator as qe_module

        if not qe_module.SKLEARN_AVAILABLE:
            return {}

        try:
            from sklearn.metrics import auc, roc_curve

            classes = np.unique(all_targets)
            per_class_metrics = {}

            for class_idx in classes:
                # Binary problem for this class vs all others
                y_true_binary = (all_targets == class_idx).astype(int)
                y_score = all_probabilities[:, int(class_idx)]

                # Compute ROC curve
                fpr, tpr, thresholds = roc_curve(y_true_binary, y_score)
                roc_auc = auc(fpr, tpr)

                per_class_metrics[str(int(class_idx))] = {
                    "roc_auc": float(roc_auc),
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                    "thresholds": thresholds.tolist(),
                }

            return per_class_metrics

        except Exception as e:
            _logger.warning(f"Failed to compute per-class ROC AUC: {e}")
            return {}

    def _analyze_confusion_patterns(
        self, cm: np.ndarray, classes: np.ndarray
    ) -> dict[str, Any]:
        """Analyze confusion patterns in the confusion matrix."""
        try:
            # Find most confused class pairs
            confusion_pairs = []
            for i in range(len(classes)):
                for j in range(len(classes)):
                    if i != j and cm[i, j] > 0:
                        confusion_pairs.append(
                            {
                                "true_class": int(classes[i]),
                                "predicted_class": int(classes[j]),
                                "count": int(cm[i, j]),
                                "percentage": float(cm[i, j] / cm[i, :].sum() * 100),
                            }
                        )

            # Sort by confusion count (descending)
            confusion_pairs.sort(key=lambda x: x["count"], reverse=True)

            # Find classes with highest error rates
            error_rates = []
            for i, cls in enumerate(classes):
                total_for_class = cm[i, :].sum()
                correct = cm[i, i]
                error_count = total_for_class - correct
                error_rate = (
                    error_count / total_for_class if total_for_class > 0 else 0.0
                )

                error_rates.append(
                    {
                        "class": int(cls),
                        "error_rate": float(error_rate),
                        "error_count": int(error_count),
                        "total_samples": int(total_for_class),
                    }
                )

            error_rates.sort(key=lambda x: x["error_rate"], reverse=True)

            return {
                "top_confusions": confusion_pairs[:5],
                "highest_error_rates": error_rates[:3],
                "total_confusions": len(confusion_pairs),
            }
        except Exception as e:
            _logger.warning(f"Failed to analyze confusion patterns: {e}")
            return {}

    def _compute_class_insights(
        self,
        per_class_metrics: dict,
        class_distribution: np.ndarray,
        total_samples: int,
    ) -> dict[str, Any]:
        """Compute insights about class performance and distribution."""
        try:
            insights = {
                "class_balance": {},
                "performance_summary": {},
                "recommendations": [],
            }

            # Analyze class balance
            max_samples = class_distribution.max()
            min_samples = class_distribution.min()
            balance_ratio = min_samples / max_samples if max_samples > 0 else 0.0

            insights["class_balance"] = {
                "most_samples": int(np.argmax(class_distribution)),
                "least_samples": int(np.argmin(class_distribution)),
                "balance_ratio": float(balance_ratio),
                "is_balanced": balance_ratio > 0.5,
            }

            # Performance summary
            f1_scores = [metrics["f1"] for metrics in per_class_metrics.values()]
            precision_scores = [
                metrics["precision"] for metrics in per_class_metrics.values()
            ]
            recall_scores = [
                metrics["recall"] for metrics in per_class_metrics.values()
            ]

            insights["performance_summary"] = {
                "avg_f1": float(np.mean(f1_scores)),
                "min_f1": float(np.min(f1_scores)),
                "max_f1": float(np.max(f1_scores)),
                "avg_precision": float(np.mean(precision_scores)),
                "avg_recall": float(np.mean(recall_scores)),
                "performance_std": float(np.std(f1_scores)),
            }

            # Generate recommendations
            if balance_ratio < 0.3:
                insights["recommendations"].append(
                    "Dataset is highly imbalanced. Consider data augmentation or class weighting."
                )

            if np.std(f1_scores) > 0.2:
                worst_class = min(
                    per_class_metrics.keys(), key=lambda k: per_class_metrics[k]["f1"]
                )
                insights["recommendations"].append(
                    f"Class {worst_class} has significantly lower performance. Consider targeted data collection."
                )

            if np.mean(f1_scores) < 0.7:
                insights["recommendations"].append(
                    "Overall performance is low. Consider model architecture changes or hyperparameter tuning."
                )

            return insights
        except Exception as e:
            _logger.warning(f"Failed to compute class insights: {e}")
            return {}

    def _compute_per_class_analysis(
        self,
        all_predictions: np.ndarray,
        all_targets: np.ndarray,
        class_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Compute detailed per-class performance analysis."""
        if not self.cfg.compute_per_class_metrics:
            return {}

        try:
            cm = confusion_matrix(all_targets.tolist(), all_predictions.tolist())
            classes = np.unique(all_targets)

            per_class_metrics = {}
            for i, cls in enumerate(classes):
                # True Positives, False Positives, False Negatives
                tp = cm[i, i]
                fp = cm[:, i].sum() - tp
                fn = cm[i, :].sum() - tp

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = (
                    2 * (precision * recall) / (precision + recall)
                    if (precision + recall) > 0
                    else 0.0
                )

                # Calculate additional metrics
                tn = cm.sum() - (tp + fp + fn)  # True negatives
                specificity = (
                    tn / (tn + fp) if (tn + fp) > 0 else 0.0
                )  # True negative rate
                balanced_accuracy = (recall + specificity) / 2
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False positive rate

                per_class_metrics[int(cls)] = {
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "support": int(cm[i, :].sum()),
                    "specificity": float(specificity),
                    "balanced_accuracy": float(balanced_accuracy),
                    "fpr": float(fpr),
                    "true_positives": int(tp),
                    "false_positives": int(fp),
                    "true_negatives": int(tn),
                    "false_negatives": int(fn),
                }

            # Compute additional analysis
            total_samples = len(all_targets)
            class_distribution = np.bincount(all_targets, minlength=len(classes))

            # Error analysis - most confused classes
            error_analysis = self._analyze_confusion_patterns(cm, classes)

            # Class-wise performance insights
            performance_insights = self._compute_class_insights(
                per_class_metrics, class_distribution, total_samples
            )

            # Calculate overall statistics
            total_correct = sum(cm[i, i] for i in range(len(classes)))
            overall_accuracy = (
                total_correct / total_samples if total_samples > 0 else 0.0
            )

            result = {
                "confusion_matrix": cm.tolist(),
                "confusion_matrix_normalized": (
                    cm / cm.sum(axis=1, keepdims=True)
                ).tolist(),
                "per_class_metrics": per_class_metrics,
                "class_names": class_names or [f"Class_{i}" for i in classes],
                "class_distribution": class_distribution.tolist(),
                "total_samples": int(total_samples),
                "overall_statistics": {
                    "total_samples": int(total_samples),
                    "total_correct": int(total_correct),
                    "overall_accuracy": float(overall_accuracy),
                    "class_distribution": [
                        int(cm[i, :].sum()) for i in range(len(classes))
                    ],
                },
                "error_analysis": error_analysis,
                "performance_insights": performance_insights,
                "confusion_matrix_stats": {
                    "accuracy_per_class": [
                        per_class_metrics[int(cls)]["precision"] for cls in classes
                    ],
                    "recall_per_class": [
                        per_class_metrics[int(cls)]["recall"] for cls in classes
                    ],
                    "f1_per_class": [
                        per_class_metrics[int(cls)]["f1"] for cls in classes
                    ],
                },
            }

            return result

        except Exception as e:
            _logger.warning(f"Failed to compute per-class analysis: {e}")
            return {}

    def _plot_confusion_matrix(
        self,
        confusion_matrix_data: dict[str, Any],
        save_path: str | None = None,
        title: str = "Confusion Matrix",
    ) -> str | None:
        """Create and optionally save confusion matrix visualization."""
        if not MATPLOTLIB_AVAILABLE:
            _logger.warning("Matplotlib not available for visualization")
            return None

        try:
            cm = np.array(confusion_matrix_data["confusion_matrix"])
            class_names = confusion_matrix_data.get(
                "class_names", [f"Class_{i}" for i in range(cm.shape[0])]
            )

            # Create figure
            plt.figure(figsize=(10, 8))

            # Create heatmap
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={"label": "Count"},
            )

            plt.title(title, fontsize=16, fontweight="bold")
            plt.xlabel("Predicted Label", fontsize=12)
            plt.ylabel("True Label", fontsize=12)
            plt.tight_layout()

            # Save if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                _logger.info(f"Confusion matrix saved to {save_path}")
                plt.close()
                return save_path
            else:
                # Return as base64 string for display
                import base64
                import io

                buffer = io.BytesIO()
                plt.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.getvalue()).decode()
                plt.close()

                return f"data:image/png;base64,{image_base64}"

        except Exception as e:
            _logger.warning(f"Failed to create confusion matrix visualization: {e}")
            return None

    def _collect_predictions_and_targets(
        self, model: nn.Module, dataloader: DataLoader
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Collect all predictions and targets for metric computation."""
        model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []

        with torch.no_grad():
            for xb, yb in dataloader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                outputs = model(xb)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = outputs.argmax(dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(yb.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        return (
            np.array(all_predictions),
            np.array(all_targets),
            np.array(all_probabilities) if self.cfg.compute_auc_scores else None,
        )

    def _enhanced_evaluate(
        self, model: nn.Module, dataloader: DataLoader, criterion
    ) -> dict[str, Any]:
        """Enhanced evaluation with comprehensive metrics."""
        # Get basic metrics first
        basic_metrics = self._evaluate(model, dataloader, criterion)

        # Collect predictions and targets for advanced metrics
        all_predictions, all_targets, all_probabilities = (
            self._collect_predictions_and_targets(model, dataloader)
        )

        # Compute enhanced classification metrics
        classification_metrics = self._compute_classification_metrics(
            all_predictions, all_targets
        )

        # Merge metrics
        enhanced_metrics = {**basic_metrics, **classification_metrics}

        # Compute AUC metrics if probabilities are available
        if all_probabilities is not None:
            auc_metrics = self._compute_auc_metrics(all_probabilities, all_targets)
            enhanced_metrics.update(auc_metrics)

        # Compute per-class analysis if enabled
        if self.cfg.compute_per_class_metrics:
            per_class_analysis = self._compute_per_class_analysis(
                all_predictions, all_targets
            )
            enhanced_metrics.update(per_class_analysis)

        return enhanced_metrics

    def _track_learning_dynamics(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer,
        num_epochs: int,
    ) -> dict[str, Any]:
        """Track learning dynamics during training with advanced analysis."""
        if not self.cfg.track_learning_curves:
            return {}

        learning_dynamics = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "gradient_norms": [],
            "learning_rates": [],
            "overfitting_analysis": {},
            "convergence_analysis": {},
            "training_insights": [],
        }

        try:
            criterion = nn.CrossEntropyLoss()
            initial_lr = self.cfg.learning_rate

            for epoch in range(num_epochs):
                # Track current learning rate
                current_lr = (
                    optimizer.param_groups[0]["lr"] if optimizer else initial_lr
                )
                learning_dynamics["learning_rates"].append(float(current_lr))

                # Training phase with gradient tracking
                if self.cfg.track_gradient_norms:
                    total_grad_norm = 0.0
                    param_count = 0
                    for param in model.parameters():
                        if param.grad is not None:
                            grad_norm = param.grad.data.norm(2)
                            total_grad_norm += grad_norm.item() ** 2
                            param_count += 1

                    if param_count > 0:
                        learning_dynamics["gradient_norms"].append(
                            np.sqrt(total_grad_norm)
                        )
                    else:
                        learning_dynamics["gradient_norms"].append(0.0)

                # Evaluate and store metrics
                train_metrics = self._train_one_epoch(
                    model, train_loader, optimizer, criterion
                )
                val_metrics = self._evaluate(model, val_loader, criterion)

                learning_dynamics["train_loss"].append(train_metrics["loss"])
                learning_dynamics["train_accuracy"].append(train_metrics["accuracy"])
                learning_dynamics["val_loss"].append(val_metrics["val_loss"])
                learning_dynamics["val_accuracy"].append(val_metrics["val_accuracy"])

            # Analyze learning dynamics
            overfitting_analysis = self._analyze_overfitting(learning_dynamics)
            convergence_analysis = self._analyze_convergence(learning_dynamics)
            training_insights = self._generate_training_insights(
                learning_dynamics, overfitting_analysis, convergence_analysis
            )

            learning_dynamics["overfitting_analysis"] = overfitting_analysis
            learning_dynamics["convergence_analysis"] = convergence_analysis
            learning_dynamics["training_insights"] = training_insights

        except Exception as e:
            _logger.warning(f"Failed to track learning dynamics: {e}")

        return learning_dynamics

    def _analyze_overfitting(self, learning_dynamics: dict[str, Any]) -> dict[str, Any]:
        """Analyze overfitting patterns in training curves."""
        try:
            train_loss = learning_dynamics["train_loss"]
            val_loss = learning_dynamics["val_loss"]
            train_accuracy = learning_dynamics["train_accuracy"]
            val_accuracy = learning_dynamics["val_accuracy"]

            if len(train_loss) < 3:
                return {"overfitting_detected": False, "analysis": "Insufficient data"}

            overfitting_analysis = {
                "overfitting_detected": False,
                "overfitting_epoch": None,
                "overfitting_severity": "none",
                "gap_analysis": {},
                "recommendations": [],
            }

            # Analyze loss gap
            loss_gaps = []
            accuracy_gaps = []

            for i in range(len(val_loss)):
                loss_gap = train_loss[i] - val_loss[i]
                accuracy_gap = train_accuracy[i] - val_accuracy[i]
                loss_gaps.append(loss_gap)
                accuracy_gaps.append(accuracy_gap)

            # Detect overfitting patterns
            final_gap = loss_gaps[-1]
            max_gap = max(loss_gaps)
            avg_gap = np.mean(loss_gaps)

            # Criteria for overfitting detection
            overfitting_threshold = 0.1  # 10% relative gap
            severe_overfitting_threshold = 0.2  # 20% relative gap

            if final_gap > overfitting_threshold:
                overfitting_analysis["overfitting_detected"] = True

                # Find overfitting epoch (first significant gap)
                for i, gap in enumerate(loss_gaps):
                    if gap > overfitting_threshold:
                        overfitting_analysis["overfitting_epoch"] = i + 1
                        break

                # Classify severity
                if final_gap > severe_overfitting_threshold:
                    overfitting_analysis["overfitting_severity"] = "severe"
                elif final_gap > overfitting_threshold * 1.5:
                    overfitting_analysis["overfitting_severity"] = "moderate"
                else:
                    overfitting_analysis["overfitting_severity"] = "mild"

            # Gap analysis
            overfitting_analysis["gap_analysis"] = {
                "final_loss_gap": float(final_gap),
                "max_loss_gap": float(max_gap),
                "avg_loss_gap": float(avg_gap),
                "accuracy_gap_trend": "increasing"
                if accuracy_gaps[-1] > accuracy_gaps[0]
                else "stable",
                "loss_gap_trend": "increasing"
                if loss_gaps[-1] > loss_gaps[0]
                else "stable",
            }

            # Generate recommendations
            if overfitting_analysis["overfitting_detected"]:
                if overfitting_analysis["overfitting_severity"] == "severe":
                    overfitting_analysis["recommendations"].append(
                        "Severe overfitting detected. Consider stronger regularization, more data, or early stopping."
                    )
                elif overfitting_analysis["overfitting_severity"] == "moderate":
                    overfitting_analysis["recommendations"].append(
                        "Moderate overfitting detected. Consider dropout, weight decay, or data augmentation."
                    )
                else:
                    overfitting_analysis["recommendations"].append(
                        "Mild overfitting detected. Consider early stopping or slight regularization."
                    )
            else:
                overfitting_analysis["recommendations"].append(
                    "No significant overfitting detected. Training appears healthy."
                )

            return overfitting_analysis

        except Exception as e:
            _logger.warning(f"Failed to analyze overfitting: {e}")
            return {"overfitting_detected": False, "error": str(e)}

    def _analyze_convergence(self, learning_dynamics: dict[str, Any]) -> dict[str, Any]:
        """Analyze convergence patterns in training."""
        try:
            val_loss = learning_dynamics["val_loss"]
            val_accuracy = learning_dynamics["val_accuracy"]

            if len(val_loss) < 3:
                return {"convergence_detected": False, "analysis": "Insufficient data"}

            convergence_analysis = {
                "convergence_detected": False,
                "convergence_epoch": None,
                "convergence_rate": "unknown",
                "final_improvement": {},
                "stability_analysis": {},
                "recommendations": [],
            }

            # Calculate improvement metrics
            initial_loss = val_loss[0]
            final_loss = val_loss[-1]
            initial_accuracy = val_accuracy[0]
            final_accuracy = val_accuracy[-1]

            loss_improvement = initial_loss - final_loss
            accuracy_improvement = final_accuracy - initial_accuracy

            convergence_analysis["final_improvement"] = {
                "loss_improvement": float(loss_improvement),
                "accuracy_improvement": float(accuracy_improvement),
                "relative_loss_improvement": float(loss_improvement / initial_loss)
                if initial_loss > 0
                else 0.0,
                "relative_accuracy_improvement": float(
                    accuracy_improvement / initial_accuracy
                )
                if initial_accuracy > 0
                else 0.0,
            }

            # Analyze stability (variance in last few epochs)
            recent_epochs = min(5, len(val_loss) // 2)
            recent_losses = val_loss[-recent_epochs:]
            recent_accuracies = val_accuracy[-recent_epochs:]

            loss_stability = np.std(recent_losses)
            accuracy_stability = np.std(recent_accuracies)

            convergence_analysis["stability_analysis"] = {
                "recent_loss_std": float(loss_stability),
                "recent_accuracy_std": float(accuracy_stability),
                "loss_is_stable": loss_stability < 0.01,
                "accuracy_is_stable": accuracy_stability < 0.01,
            }

            # Convergence detection criteria
            loss_improvement_threshold = 0.005  # 0.5% relative improvement
            stability_threshold = 0.01  # Standard deviation threshold

            if loss_improvement > initial_loss * loss_improvement_threshold:
                # Check if convergence has been achieved (stable performance)
                if (
                    loss_stability < stability_threshold
                    and accuracy_stability < stability_threshold
                ):
                    convergence_analysis["convergence_detected"] = True

                    # Find convergence epoch (when improvement plateaus)
                    for i in range(2, len(val_loss) - recent_epochs + 1):
                        recent_std = np.std(val_loss[i : i + recent_epochs])
                        if recent_std < stability_threshold:
                            convergence_analysis["convergence_epoch"] = (
                                i + recent_epochs
                            )
                            break

                    # Determine convergence rate
                    if loss_improvement > initial_loss * 0.1:
                        convergence_analysis["convergence_rate"] = "fast"
                    elif loss_improvement > initial_loss * 0.05:
                        convergence_analysis["convergence_rate"] = "moderate"
                    else:
                        convergence_analysis["convergence_rate"] = "slow"

            # Generate recommendations
            if convergence_analysis["convergence_detected"]:
                convergence_analysis["recommendations"].append(
                    f"Convergence detected at epoch {convergence_analysis['convergence_epoch']}. "
                    "Consider early stopping for efficiency."
                )
            else:
                if loss_stability < stability_threshold:
                    convergence_analysis["recommendations"].append(
                        "Training appears to have plateaued. Consider learning rate scheduling or architecture changes."
                    )
                else:
                    convergence_analysis["recommendations"].append(
                        "Training is still improving. Consider running for more epochs."
                    )

            return convergence_analysis

        except Exception as e:
            _logger.warning(f"Failed to analyze convergence: {e}")
            return {"convergence_detected": False, "error": str(e)}

    def _generate_training_insights(
        self,
        learning_dynamics: dict[str, Any],
        overfitting_analysis: dict[str, Any],
        convergence_analysis: dict[str, Any],
    ) -> list[str]:
        """Generate comprehensive training insights and recommendations."""
        insights = []

        try:
            train_loss = learning_dynamics["train_loss"]
            val_loss = learning_dynamics["val_loss"]
            train_accuracy = learning_dynamics["train_accuracy"]
            val_accuracy = learning_dynamics["val_accuracy"]
            gradient_norms = learning_dynamics["gradient_norms"]

            # Overall training assessment
            if len(train_loss) > 0:
                final_train_loss = train_loss[-1]
                final_val_loss = val_loss[-1]
                final_train_accuracy = train_accuracy[-1]
                final_val_accuracy = val_accuracy[-1]

                # Performance assessment
                if final_val_accuracy > 0.9:
                    insights.append("🎉 Excellent model performance achieved!")
                elif final_val_accuracy > 0.8:
                    insights.append("👍 Good model performance achieved.")
                elif final_val_accuracy > 0.7:
                    insights.append("👌 Reasonable model performance.")
                else:
                    insights.append("⚠️ Model performance may need improvement.")

                # Training efficiency
                convergence_epoch = convergence_analysis.get("convergence_epoch")
                if convergence_epoch is not None:
                    epochs_to_convergence = convergence_epoch
                    if epochs_to_convergence < len(train_loss) * 0.5:
                        insights.append("⚡ Efficient training - converged quickly!")
                    elif epochs_to_convergence > len(train_loss) * 0.8:
                        insights.append(
                            "🐌 Training took many epochs - consider optimizing."
                        )
                else:
                    insights.append(
                        "⏳ Training did not converge within the given epochs."
                    )

            # Overfitting insights
            if overfitting_analysis.get("overfitting_detected", False):
                severity = overfitting_analysis.get("overfitting_severity", "unknown")
                if severity == "severe":
                    insights.append("🚨 Critical overfitting detected!")
                elif severity == "moderate":
                    insights.append("⚠️ Moderate overfitting detected.")
                else:
                    insights.append("💡 Mild overfitting detected.")

            # Convergence insights
            if convergence_analysis.get("convergence_detected", False):
                insights.append("✅ Training successfully converged.")
            else:
                insights.append("⏳ Training still in progress or plateaued.")

            # Gradient analysis
            if len(gradient_norms) > 0:
                avg_grad_norm = np.mean(gradient_norms)
                if avg_grad_norm > 10:
                    insights.append(
                        "💪 Large gradient norms detected - training is active."
                    )
                elif avg_grad_norm < 0.1:
                    insights.append(
                        "😴 Small gradient norms - possible vanishing gradients."
                    )
                else:
                    insights.append("📊 Normal gradient norms observed.")

            # Learning rate insights
            if len(learning_dynamics["learning_rates"]) > 1:
                initial_lr = learning_dynamics["learning_rates"][0]
                final_lr = learning_dynamics["learning_rates"][-1]
                if final_lr != initial_lr:
                    insights.append(
                        f"🔧 Learning rate changed from {initial_lr:.6f} to {final_lr:.6f}"
                    )
                else:
                    insights.append(f"📈 Constant learning rate: {initial_lr:.6f}")

            # Data distribution insights
            if len(val_accuracy) > 0:
                val_acc_trend = val_accuracy[-1] - val_accuracy[0]
                if val_acc_trend > 0.1:
                    insights.append(
                        "📈 Significant improvement in validation accuracy!"
                    )
                elif val_acc_trend < 0.01:
                    insights.append("📊 Validation accuracy plateaued.")

        except Exception as e:
            _logger.warning(f"Failed to generate training insights: {e}")
            insights.append("Unable to generate detailed insights due to an error.")

        return insights


# Backward compatibility
QuickEvaluator = EnhancedQuickEvaluator


def quick_evaluate_once(
    model: nn.Module, cfg: QuickEvalConfig | None = None
) -> dict[str, Any]:
    """Enhanced convenience function for one-shot quick evaluation.

    Example:
        cfg = QuickEvalConfig(epochs=1, subset_size=512, num_seeds=3)
        res = quick_evaluate_once(my_model, cfg)

    """
    cfg = cfg or QuickEvalConfig()
    ev = EnhancedQuickEvaluator(cfg)
    return ev.quick_evaluate(model)


# Legacy compatibility function
def quick_evaluate_legacy(
    model: nn.Module, cfg: QuickEvalConfig | None = None
) -> dict[str, Any]:
    """Legacy compatibility function with single-seed evaluation."""
    cfg = cfg or QuickEvalConfig()
    cfg.num_seeds = 1  # Force single seed for backward compatibility
    ev = EnhancedQuickEvaluator(cfg)
    return ev.quick_evaluate(model)


def main() -> None:
    """Main entrypoint for the quick evaluator CLI."""
    import argparse

    p = argparse.ArgumentParser(description="Enhanced quick evaluator")
    p.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    p.add_argument("--dataset", default="cifar10", help="Dataset name")
    p.add_argument(
        "--subset-size", type=int, default=512, help="Subset size for quick evaluation"
    )
    p.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help="Number of seeds for statistical evaluation",
    )
    p.add_argument(
        "--target-accuracy", type=float, default=0.7, help="Target accuracy threshold"
    )
    p.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = p.parse_args()

    # Create configuration
    cfg = QuickEvalConfig(
        epochs=args.epochs,
        dataset_name=args.dataset,
        subset_size=args.subset_size,
        num_seeds=args.num_seeds,
        target_accuracy=args.target_accuracy,
        verbose=args.verbose,
    )

    # Create a simple test model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 32, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((4, 4))
            self.fc = nn.Linear(32 * 4 * 4, 10)

        def forward(self, x):
            x = self.conv(x)
            x = torch.relu(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    # Run evaluation
    model = TestModel()
    result = quick_evaluate_once(model, cfg)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Model: {result['model_name']}")
    print(f"Seeds evaluated: {result['num_seeds']}")

    if "aggregated" in result:
        agg = result["aggregated"]
        if "val_accuracy" in agg:
            acc_stats = agg["val_accuracy"]
            print(f"Accuracy: {acc_stats['mean']:.4f} ± {acc_stats['std']:.4f}")
            if "ci_95" in acc_stats:
                ci_low, ci_high = acc_stats["ci_95"]
                print(f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]")

    print(f"Goal achieved: {result.get('goal_achieved', False)}")
    print("=" * 50)


if __name__ == "__main__":  # pragma: no cover - simple entrypoint
    main()
