"""Enhanced Dataset Loader for PyTorch ML Research Agent

This module provides robust dataset loading capabilities including:
- Hugging Face Datasets integration with caching
- Support for computer vision, NLP, and tabular datasets
- Reproducible sampling with fixed seeds
- Subset support for quick evaluation
- Automatic preprocessing pipelines
- Flexible configuration system
"""
# ruff: noqa: T201

from __future__ import annotations

import hashlib
import json
import logging
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Optional dependencies
try:
    import torch
    from torch.utils.data import DataLoader, Dataset, random_split

    TORCH_AVAILABLE = True
except Exception:
    torch = None  # type: ignore
    Dataset = object  # type: ignore
    DataLoader = object  # type: ignore
    random_split = None  # type: ignore
    TORCH_AVAILABLE = False

try:
    import torchvision
    import torchvision.transforms as transforms

    TORCHVISION_AVAILABLE = True
except Exception:
    torchvision = None  # type: ignore
    transforms = None  # type: ignore
    TORCHVISION_AVAILABLE = False

try:
    from datasets import Dataset as HFDataset
    from datasets import load_dataset, load_from_disk

    HF_DATASETS_AVAILABLE = True
except Exception:
    HFDataset = None  # type: ignore
    load_dataset = None  # type: ignore
    load_from_disk = None  # type: ignore
    HF_DATASETS_AVAILABLE = False

_logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Flexible configuration for dataset loading and preprocessing.

    This class provides a comprehensive configuration system for various types of
    datasets used in machine learning research, supporting both traditional and
    modern dataset formats with extensive customization options.

    Attributes:
        name: Dataset name (e.g., "cifar10", "glue", "imdb", "custom")
        subset: Optional subset name (e.g., "sst2", "cola" for GLUE)
        split: Dataset split to use ("train", "test", "validation")

        # Sampling and Performance
        subset_size: Maximum number of samples to use (for quick evaluation)
        cache_dir: Directory for caching datasets locally
        seed: Random seed for reproducible sampling and operations

        # Preprocessing and Transforms
        transforms: List of transform steps to apply
        preprocessing_steps: Custom preprocessing pipeline
        augmentations: Data augmentation configuration
        normalization: Normalization parameters

        # Dataset Loading
        batch_size: Batch size for DataLoader
        num_workers: Number of DataLoader workers
        shuffle: Whether to shuffle the dataset
        drop_last: Whether to drop last incomplete batch
        pin_memory: Whether to pin memory for DataLoader

        # Data Specifications
        input_shape: Expected input shape for models
        target_shape: Expected target shape for models
        num_classes: Number of classes for classification
        feature_columns: Column names for features (tabular/NLP)
        target_column: Column name for target labels

        # Advanced Options
        max_sequence_length: Maximum sequence length for text
        image_size: Target image size for preprocessing
        offine_mode: Use only cached datasets (no downloads)
        validation_split: Fraction of data to use for validation
        test_split: Fraction of data to use for testing

        # Custom Arguments
        dataset_kwargs: Additional dataset-specific arguments
        custom_configs: Custom configuration dictionary

    """

    # Core dataset identification
    name: str = "cifar10"
    subset: str | None = None
    split: str = "train"

    # Sampling and caching
    subset_size: int | None = None
    cache_dir: str = "./data/cache"
    seed: int = 42
    offline_mode: bool = False

    # DataLoader configuration
    batch_size: int = 64
    num_workers: int = 0
    shuffle: bool = True
    drop_last: bool = False
    pin_memory: bool = True

    # Data specifications
    input_shape: tuple[int, ...] | None = None
    target_shape: tuple[int, ...] | None = None
    num_classes: int | None = None

    # Feature specifications
    feature_columns: list[str] | None = None
    target_column: str = "label"
    text_column: str | None = None
    image_column: str | None = None

    # Preprocessing pipeline
    transforms: list[str] = field(default_factory=lambda: ["basic"])
    preprocessing_steps: list[str] | None = None
    augmentations: dict[str, Any] | None = None
    normalization: dict[str, Any] | None = None

    # Advanced specifications
    max_sequence_length: int | None = None
    image_size: tuple[int, int] | None = None
    validation_split: float = 0.1
    test_split: float = 0.1

    # Custom configurations
    dataset_kwargs: dict[str, Any] = field(default_factory=dict)
    custom_configs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Post-initialization processing and validation."""
        # Ensure required fields are properly initialized
        if self.transforms is None:
            self.transforms = ["basic"]

        if self.preprocessing_steps is None:
            self.preprocessing_steps = self.transforms

        # Auto-detect common configurations based on dataset name
        self._auto_configure()

        # Validate configuration
        self._validate()

    def _auto_configure(self):
        """Automatically configure reasonable defaults based on dataset name."""
        dataset_name = self.name.lower()

        # Computer Vision datasets
        if dataset_name in ["cifar10", "cifar100", "svhn"]:
            if self.input_shape is None:
                self.input_shape = (3, 32, 32)
            if self.num_classes is None:
                self.num_classes = 10 if dataset_name == "cifar10" else 100
            if "normalize" not in self.transforms:
                self.transforms.append("normalize")
            if self.image_size is None:
                self.image_size = (32, 32)

        elif dataset_name == "mnist":
            if self.input_shape is None:
                self.input_shape = (1, 28, 28)
            if self.num_classes is None:
                self.num_classes = 10
            if "normalize" not in self.transforms:
                self.transforms.append("normalize")

        elif dataset_name in ["imagenet", "coco"]:
            if self.image_size is None:
                self.image_size = (224, 224)
            if self.input_shape is None:
                self.input_shape = (3, 224, 224)

        # NLP datasets
        elif dataset_name in ["glue", "super_glue", "imdb", "sst2"]:
            if self.text_column is None:
                self.text_column = "text" if dataset_name == "imdb" else "sentence"
            if self.max_sequence_length is None:
                self.max_sequence_length = 512
            if "tokenize" not in self.transforms:
                self.transforms.append("tokenize")

        # Tabular datasets
        elif dataset_name in ["titanic", "adult", "credit"]:
            if self.feature_columns is None:
                self.feature_columns = ["feature_" + str(i) for i in range(10)]
            if "tabular" not in self.transforms:
                self.transforms.append("tabular")

    def _validate(self):
        """Validate configuration parameters."""
        errors = []

        # Basic validation
        if not self.name:
            errors.append("Dataset name cannot be empty")

        if self.batch_size <= 0:
            errors.append("Batch size must be positive")

        if self.num_workers < 0:
            errors.append("Number of workers cannot be negative")

        if not 0 <= self.validation_split < 1:
            errors.append("Validation split must be between 0 and 1")

        if not 0 <= self.test_split < 1:
            errors.append("Test split must be between 0 and 1")

        if self.validation_split + self.test_split >= 1:
            errors.append("Validation and test splits cannot exceed 1")

        # Dataset-specific validation
        if (
            self.name.lower() in ["cifar10", "cifar100", "mnist"]
            and not TORCHVISION_AVAILABLE
        ):
            errors.append(f"torchvision required for {self.name} dataset")

        if (
            self.text_column
            and self.name.lower() in ["glue", "super_glue"]
            and not HF_DATASETS_AVAILABLE
        ):
            errors.append(f"Hugging Face datasets required for {self.name} dataset")

        if errors:
            raise ValueError(f"DatasetConfig validation failed: {'; '.join(errors)}")

    def get_cache_key(self) -> str:
        """Generate a unique cache key for this configuration."""
        config_dict = asdict(self)
        # Remove non-hashable items
        config_dict = {
            k: v
            for k, v in config_dict.items()
            if isinstance(v, (str, int, float, bool, type(None)))
        }

        # Create hash
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:16]

    def create_subset_config(self, new_subset_size: int) -> DatasetConfig:
        """Create a new configuration with a different subset size."""
        new_config = asdict(self)
        new_config["subset_size"] = new_subset_size
        return DatasetConfig(**new_config)

    def get_transform_config(self) -> dict[str, Any]:
        """Get transform configuration for dataset-specific preprocessing."""
        config = {
            "basic": self._get_basic_transforms(),
            "normalization": self._get_normalization_config(),
            "augmentation": self._get_augmentation_config(),
            "tokenization": self._get_tokenization_config(),
        }
        return config

    def _get_basic_transforms(self) -> list:
        """Get basic transform pipeline."""
        if not TORCHVISION_AVAILABLE:
            return []

        transforms_list = []

        # Convert to tensor
        transforms_list.append(transforms.ToTensor())

        # Add dataset-specific basic transforms
        if self.name.lower() in ["cifar10", "cifar100"]:
            transforms_list.extend(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(0.5),
                ]
            )

        return transforms_list

    def _get_normalization_config(self) -> dict[str, Any] | None:
        """Get normalization configuration."""
        if self.normalization:
            return self.normalization

        # Default normalization values
        if self.name.lower() == "cifar10":
            return {"mean": [0.4914, 0.4822, 0.4465], "std": [0.2023, 0.1994, 0.2010]}
        elif self.name.lower() == "cifar100":
            return {"mean": [0.5071, 0.4867, 0.4408], "std": [0.2675, 0.2565, 0.2761]}
        elif self.name.lower() == "mnist":
            return {"mean": [0.1307], "std": [0.3081]}

        return None

    def _get_augmentation_config(self) -> dict[str, Any] | None:
        """Get data augmentation configuration."""
        if self.augmentations:
            return self.augmentations

        # Default augmentation for image datasets
        if self.name.lower() in ["cifar10", "cifar100", "imagenet"]:
            return {
                "random_crop": {"size": 32, "padding": 4},
                "random_horizontal_flip": {"p": 0.5},
                "color_jitter": {
                    "brightness": 0.2,
                    "contrast": 0.2,
                    "saturation": 0.2,
                    "hue": 0.1,
                },
            }

        return None

    def _get_tokenization_config(self) -> dict[str, Any] | None:
        """Get tokenization configuration for NLP datasets."""
        if not HF_DATASETS_AVAILABLE:
            return None

        config = {
            "max_length": self.max_sequence_length or 512,
            "truncation": True,
            "padding": "max_length",
        }

        # Add dataset-specific tokenization settings
        if self.name.lower() == "imdb":
            config.update(
                {
                    "max_length": 512,
                    "return_tensors": "pt",
                }
            )

        return config


class DatasetLoaderError(Exception):
    """Base exception for dataset loading errors."""

    pass


class UnsupportedDatasetError(DatasetLoaderError):
    """Raised when dataset is not supported."""

    pass


class PreprocessingError(DatasetLoaderError):
    """Raised when preprocessing fails."""

    pass


class CacheError(DatasetLoaderError):
    """Raised when cache operations fail."""

    pass


class FlexibleDatasetLoader:
    """Enhanced dataset loader with extensive flexibility and support for multiple formats."""

    def __init__(self, config: DatasetConfig):
        """Initialize loader with dataset configuration and cache directory."""
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Register supported dataset loaders
        self._init_loaders()

    def _init_loaders(self):
        """Initialize dataset-specific loaders."""
        self.loaders = {
            # Computer Vision
            "cifar10": self._load_cifar10,
            "cifar100": self._load_cifar100,
            "mnist": self._load_mnist,
            "fashion_mnist": self._load_fashion_mnist,
            "svhn": self._load_svhn,
            # Natural Language Processing
            "glue": self._load_glue,
            "super_glue": self._load_super_glue,
            "imdb": self._load_imdb,
            "sst2": self._load_sst2,
            "cola": self._load_cola,
            "qnli": self._load_qnli,
            # Tabular
            "titanic": self._load_titanic,
            "adult": self._load_adult,
            "credit": self._load_credit,
            # Synthetic
            "synthetic": self._load_synthetic,
        }

    def load_dataset(self) -> tuple[Dataset, Dataset | None, Dataset | None]:
        """Load dataset with flexible configuration support.

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
            where val_dataset and test_dataset may be None

        """
        if not TORCH_AVAILABLE:
            raise DatasetLoaderError("PyTorch is required for dataset loading.")

        dataset_name = self.config.name.lower()

        if dataset_name in self.loaders:
            return self.loaders[dataset_name]()
        else:
            # Try Hugging Face datasets as fallback
            if HF_DATASETS_AVAILABLE:
                return self._load_hf_dataset(dataset_name)
            else:
                _logger.warning(
                    "Dataset '%s' not supported and Hugging Face datasets not available. "
                    "Falling back to synthetic dataset.",
                    dataset_name,
                )
                return self._load_synthetic()

    # Alias for backward compatibility
    def load(self):
        """Alias for load_dataset method."""
        return self.load_dataset()

    def _load_cifar10(self) -> tuple[Dataset, Dataset | None, Dataset | None]:
        """Load CIFAR-10 dataset with flexible configuration."""
        if not TORCHVISION_AVAILABLE:
            raise DatasetLoaderError("torchvision is required for CIFAR-10.")

        transform = self._build_transforms()

        train_dataset = torchvision.datasets.CIFAR10(
            root=str(self.cache_dir / "cifar10"),
            train=True,
            download=True,
            transform=transform,
        )

        test_dataset = torchvision.datasets.CIFAR10(
            root=str(self.cache_dir / "cifar10"),
            train=False,
            download=True,
            transform=transform,
        )

        return self._apply_splits(train_dataset, test_dataset)

    def _load_cifar100(self) -> tuple[Dataset, Dataset | None, Dataset | None]:
        """Load CIFAR-100 dataset with flexible configuration."""
        if not TORCHVISION_AVAILABLE:
            raise DatasetLoaderError("torchvision is required for CIFAR-100.")

        transform = self._build_transforms()

        train_dataset = torchvision.datasets.CIFAR100(
            root=str(self.cache_dir / "cifar100"),
            train=True,
            download=True,
            transform=transform,
        )

        test_dataset = torchvision.datasets.CIFAR100(
            root=str(self.cache_dir / "cifar100"),
            train=False,
            download=True,
            transform=transform,
        )

        return self._apply_splits(train_dataset, test_dataset)

    def _load_mnist(self) -> tuple[Dataset, Dataset | None, Dataset | None]:
        """Load MNIST dataset with flexible configuration."""
        if not TORCHVISION_AVAILABLE:
            raise DatasetLoaderError("torchvision is required for MNIST.")

        transform = self._build_transforms()

        train_dataset = torchvision.datasets.MNIST(
            root=str(self.cache_dir / "mnist"),
            train=True,
            download=True,
            transform=transform,
        )

        test_dataset = torchvision.datasets.MNIST(
            root=str(self.cache_dir / "mnist"),
            train=False,
            download=True,
            transform=transform,
        )

        return self._apply_splits(train_dataset, test_dataset)

    def _load_fashion_mnist(
        self,
    ) -> tuple[Dataset, Dataset | None, Dataset | None]:
        """Load Fashion-MNIST dataset with flexible configuration."""
        if not TORCHVISION_AVAILABLE:
            raise DatasetLoaderError("torchvision is required for Fashion-MNIST.")

        transform = self._build_transforms()

        train_dataset = torchvision.datasets.FashionMNIST(
            root=str(self.cache_dir / "fashion_mnist"),
            train=True,
            download=True,
            transform=transform,
        )

        test_dataset = torchvision.datasets.FashionMNIST(
            root=str(self.cache_dir / "fashion_mnist"),
            train=False,
            download=True,
            transform=transform,
        )

        return self._apply_splits(train_dataset, test_dataset)

    def _load_svhn(self) -> tuple[Dataset, Dataset | None, Dataset | None]:
        """Load SVHN dataset with flexible configuration."""
        if not TORCHVISION_AVAILABLE:
            raise DatasetLoaderError("torchvision is required for SVHN.")

        transform = self._build_transforms()

        train_dataset = torchvision.datasets.SVHN(
            root=str(self.cache_dir / "svhn"),
            split="train",
            download=True,
            transform=transform,
        )

        test_dataset = torchvision.datasets.SVHN(
            root=str(self.cache_dir / "svhn"),
            split="test",
            download=True,
            transform=transform,
        )

        return self._apply_splits(train_dataset, test_dataset)

    def _load_glue(self) -> tuple[Dataset, Dataset | None, Dataset | None]:
        """Load GLUE dataset with flexible configuration."""
        if not HF_DATASETS_AVAILABLE:
            raise DatasetLoaderError("Hugging Face datasets is required for GLUE.")

        subset = self.config.subset or "sst2"
        dataset = load_dataset("glue", subset, cache_dir=str(self.cache_dir / "glue"))

        return self._process_hf_classification_dataset(dataset)

    def _load_super_glue(self) -> tuple[Dataset, Dataset | None, Dataset | None]:
        """Load SuperGLUE dataset with flexible configuration."""
        if not HF_DATASETS_AVAILABLE:
            raise DatasetLoaderError("Hugging Face datasets is required for SuperGLUE.")

        subset = self.config.subset or "boolq"
        dataset = load_dataset(
            "super_glue", subset, cache_dir=str(self.cache_dir / "super_glue")
        )

        text_field = "passage" if subset == "boolq" else "text"
        return self._process_hf_classification_dataset(dataset, text_field=text_field)

    def _load_imdb(self) -> tuple[Dataset, Dataset | None, Dataset | None]:
        """Load IMDB dataset with flexible configuration."""
        if not HF_DATASETS_AVAILABLE:
            raise DatasetLoaderError("Hugging Face datasets is required for IMDB.")

        dataset = load_dataset("imdb", cache_dir=str(self.cache_dir / "imdb"))
        return self._process_hf_classification_dataset(dataset)

    def _load_sst2(self) -> tuple[Dataset, Dataset | None, Dataset | None]:
        """Load SST-2 dataset with flexible configuration."""
        if not HF_DATASETS_AVAILABLE:
            raise DatasetLoaderError("Hugging Face datasets is required for SST-2.")

        dataset = load_dataset("glue", "sst2", cache_dir=str(self.cache_dir / "sst2"))
        return self._process_hf_classification_dataset(dataset, text_field="sentence")

    def _load_cola(self) -> tuple[Dataset, Dataset | None, Dataset | None]:
        """Load CoLA dataset with flexible configuration."""
        if not HF_DATASETS_AVAILABLE:
            raise DatasetLoaderError("Hugging Face datasets is required for CoLA.")

        dataset = load_dataset("glue", "cola", cache_dir=str(self.cache_dir / "cola"))
        return self._process_hf_classification_dataset(dataset, text_field="sentence")

    def _load_qnli(self) -> tuple[Dataset, Dataset | None, Dataset | None]:
        """Load QNLI dataset with flexible configuration."""
        if not HF_DATASETS_AVAILABLE:
            raise DatasetLoaderError("Hugging Face datasets is required for QNLI.")

        dataset = load_dataset("glue", "qnli", cache_dir=str(self.cache_dir / "qnli"))
        return self._process_hf_classification_dataset(dataset, text_field="question")

    def _load_titanic(self) -> tuple[Dataset, Dataset | None, Dataset | None]:
        """Load Titanic dataset with flexible configuration."""
        if not HF_DATASETS_AVAILABLE:
            raise DatasetLoaderError("Hugging Face datasets is required for Titanic.")

        dataset = load_dataset("titanic", cache_dir=str(self.cache_dir / "titanic"))
        return self._process_hf_classification_dataset(
            dataset, target_column="Survived"
        )

    def _load_adult(self) -> tuple[Dataset, Dataset | None, Dataset | None]:
        """Load Adult dataset with flexible configuration."""
        if not HF_DATASETS_AVAILABLE:
            raise DatasetLoaderError("Hugging Face datasets is required for Adult.")

        dataset = load_dataset("adult", cache_dir=str(self.cache_dir / "adult"))
        return self._process_hf_classification_dataset(dataset, target_column="income")

    def _load_credit(self) -> tuple[Dataset, Dataset | None, Dataset | None]:
        """Load Credit Card Fraud dataset with flexible configuration."""
        if not HF_DATASETS_AVAILABLE:
            raise DatasetLoaderError(
                "Hugging Face datasets is required for Credit dataset."
            )

        dataset = load_dataset(
            "creditcard_fraud", cache_dir=str(self.cache_dir / "credit")
        )
        return self._process_hf_classification_dataset(dataset, target_column="Class")

    def _load_synthetic(self) -> tuple[Dataset, Dataset | None, Dataset | None]:
        """Load synthetic dataset for testing and rapid prototyping."""
        num_samples = self.config.subset_size or 1024
        input_shape = self.config.input_shape or (3, 32, 32)
        num_classes = self.config.num_classes or 10

        class SyntheticDataset(Dataset):
            def __init__(
                self, num_samples: int, input_shape: tuple[int, ...], num_classes: int
            ):
                self.num_samples = num_samples
                self.input_shape = input_shape
                self.num_classes = num_classes

            def __len__(self):
                return self.num_samples

            def __getitem__(self, idx):
                # Generate diverse synthetic data
                x = np.random.randn(*self.input_shape).astype(np.float32)
                x = x / (np.linalg.norm(x) + 1e-8)  # Normalize

                # Generate label based on simple pattern
                y = int(np.sum(x) * 10) % self.num_classes

                return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

        dataset = SyntheticDataset(num_samples, input_shape, num_classes)
        return dataset, None, None

    def _load_hf_dataset(
        self, dataset_name: str
    ) -> tuple[Dataset, Dataset | None, Dataset | None]:
        """Load dataset from Hugging Face Hub with flexible configuration."""
        if not HF_DATASETS_AVAILABLE:
            raise DatasetLoaderError("Hugging Face datasets library is not available.")

        try:
            dataset = load_dataset(
                dataset_name,
                self.config.subset,
                cache_dir=str(self.cache_dir / dataset_name),
            )

            # Auto-detect dataset structure
            if "text" in dataset[self.config.split].features:
                return self._process_hf_classification_dataset(
                    dataset, text_field="text"
                )
            elif "sentence" in dataset[self.config.split].features:
                return self._process_hf_classification_dataset(
                    dataset, text_field="sentence"
                )
            else:
                _logger.warning("Could not detect dataset format for %s", dataset_name)
                return self._load_synthetic()

        except Exception as e:
            _logger.warning(
                "Failed to load Hugging Face dataset %s: %s", dataset_name, e
            )
            return self._load_synthetic()

    def _process_hf_classification_dataset(
        self,
        dataset: dict[str, HFDataset],
        text_field: str | None = None,
        target_column: str | None = None,
    ) -> tuple[Dataset, Dataset | None, Dataset | None]:
        """Process Hugging Face classification dataset with flexible configuration."""

        class HFDatasetWrapper(Dataset):
            def __init__(
                self,
                hf_dataset: HFDataset,
                text_field: str | None,
                target_column: str,
                config: DatasetConfig,
            ):
                self.hf_dataset = hf_dataset
                self.text_field = text_field
                self.target_column = target_column
                self.config = config
                self.tokenizer = None  # Could be enhanced with actual tokenization

            def __len__(self):
                return len(self.hf_dataset)

            def __getitem__(self, idx):
                item = self.hf_dataset[idx]

                if self.text_field and self.config.name.lower() in [
                    "imdb",
                    "sst2",
                    "cola",
                ]:
                    # Text classification
                    text = item[self.text_field]
                    label = item[self.target_column]

                    # Simple text encoding (could be enhanced with actual tokenizers)
                    if isinstance(text, str):
                        # Hash-based encoding for compatibility
                        features = torch.tensor(
                            [
                                hash(char.lower()) % 1000
                                for char in text[
                                    : self.config.max_sequence_length or 100
                                ]
                            ],
                            dtype=torch.float,
                        )
                    else:
                        features = torch.zeros(self.config.max_sequence_length or 100)

                elif self.feature_columns:
                    # Tabular data
                    features = torch.tensor(
                        [
                            float(item.get(col, 0.0))
                            for col in self.config.feature_columns
                        ],
                        dtype=torch.float,
                    )
                    label = item[self.target_column]
                else:
                    # Default fallback
                    features = torch.zeros(100)
                    label = item.get(self.target_column, 0)

                return features, torch.tensor(label, dtype=torch.long)

        target_column = target_column or self.config.target_column

        # Handle dataset splits
        split_name = self.config.split
        if split_name not in dataset:
            split_name = "train" if "train" in dataset else next(iter(dataset.keys()))

        train_dataset = HFDatasetWrapper(
            dataset[split_name], text_field, target_column, self.config
        )

        # Handle validation and test splits
        val_dataset = None
        test_dataset = None

        if "validation" in dataset:
            val_dataset = HFDatasetWrapper(
                dataset["validation"], text_field, target_column, self.config
            )

        if "test" in dataset:
            test_dataset = HFDatasetWrapper(
                dataset["test"], text_field, target_column, self.config
            )

        return self._apply_splits(train_dataset, val_dataset, test_dataset)

    def _apply_splits(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset | None = None,
        test_dataset: Dataset | None = None,
    ) -> tuple[Dataset, Dataset | None, Dataset | None]:
        """Apply subset size and create additional splits if needed."""
        # Apply subset size to training data
        if self.config.subset_size and len(train_dataset) > self.config.subset_size:
            random.seed(self.config.seed)
            indices = random.sample(range(len(train_dataset)), self.config.subset_size)

            class SubsetDataset(Dataset):
                def __init__(self, dataset: Dataset, indices: list[int]):
                    self.dataset = dataset
                    self.indices = indices

                def __len__(self):
                    return len(self.indices)

                def __getitem__(self, idx):
                    return self.dataset[self.indices[idx]]

            train_dataset = SubsetDataset(train_dataset, indices)

        # Create validation split if not provided
        if val_dataset is None and len(train_dataset) > 100:
            total_size = len(train_dataset)
            val_size = int(total_size * self.config.validation_split)

            if val_size > 0 and random_split is not None:
                train_dataset, val_dataset = random_split(
                    train_dataset, [total_size - val_size, val_size]
                )

        return train_dataset, val_dataset, test_dataset

    def _build_transforms(self):
        """Build transform pipeline based on configuration."""
        if not TORCHVISION_AVAILABLE:
            return None

        transform_list = []

        # Basic transforms
        if "basic" in self.config.transforms:
            transform_list.append(transforms.ToTensor())

        # Normalization
        norm_config = self.config.get_transform_config()["normalization"]
        if norm_config:
            transform_list.append(transforms.Normalize(**norm_config))

        # Data augmentation
        if "augment" in self.config.transforms:
            aug_config = self.config.get_transform_config()["augmentation"]
            if aug_config:
                # Apply basic augmentations
                if "random_crop" in aug_config:
                    params = aug_config["random_crop"]
                    transform_list.append(transforms.RandomCrop(**params))

                if "random_horizontal_flip" in aug_config:
                    transform_list.append(
                        transforms.RandomHorizontalFlip(
                            **aug_config["random_horizontal_flip"]
                        )
                    )

        # Resize if needed
        if self.config.image_size and "resize" in self.config.transforms:
            transform_list.append(transforms.Resize(self.config.image_size))

        return transforms.Compose(transform_list) if transform_list else None

    def create_dataloader(
        self,
        dataset: Dataset,
        shuffle: bool | None = None,
        drop_last: bool | None = None,
    ) -> DataLoader:
        """Create DataLoader with flexible configuration."""
        if not TORCH_AVAILABLE:
            raise DatasetLoaderError("PyTorch required for DataLoader")

        shuffle = shuffle if shuffle is not None else self.config.shuffle
        drop_last = drop_last if drop_last is not None else self.config.drop_last

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            drop_last=drop_last,
            pin_memory=self.config.pin_memory and torch.cuda.is_available(),
        )

        return dataloader

    def get_dataset_info(self) -> dict[str, Any]:
        """Get comprehensive information about the configured dataset."""
        return {
            "name": self.config.name,
            "subset": self.config.subset,
            "split": self.config.split,
            "input_shape": self.config.input_shape,
            "num_classes": self.config.num_classes,
            "subset_size": self.config.subset_size,
            "transforms": self.config.transforms,
            "cache_dir": str(self.cache_dir),
            "hf_available": HF_DATASETS_AVAILABLE,
            "torchvision_available": TORCHVISION_AVAILABLE,
            "cache_key": self.config.get_cache_key(),
        }


# Factory functions and utilities
def create_flexible_dataset_config(
    dataset_name: str,
    *,
    subset_size: int | None = None,
    batch_size: int = 64,
    transforms: list[str] | None = None,
    input_shape: tuple[int, ...] | None = None,
    num_classes: int | None = None,
    **kwargs,
) -> DatasetConfig:
    """Create a flexible dataset configuration with sensible defaults.

    Args:
        dataset_name: Name of the dataset
        subset_size: Maximum number of samples to use
        batch_size: Batch size for DataLoader
        transforms: List of transform steps
        input_shape: Expected input shape for models
        num_classes: Number of classes for classification
        **kwargs: Additional configuration parameters

    Returns:
        DatasetConfig instance with automatic configuration

    """
    config_params = {
        "name": dataset_name,
        "subset_size": subset_size,
        "batch_size": batch_size,
        "input_shape": input_shape,
        "num_classes": num_classes,
        **kwargs,
    }

    if transforms is not None:
        config_params["transforms"] = transforms

    return DatasetConfig(**config_params)


def get_flexible_dataset_loader(config: DatasetConfig) -> FlexibleDatasetLoader:
    """Factory function to create flexible dataset loader."""
    return FlexibleDatasetLoader(config)


def list_supported_datasets() -> list[str]:
    """List all supported dataset names."""
    return [
        # Computer Vision
        "cifar10",
        "cifar100",
        "mnist",
        "fashion_mnist",
        "svhn",
        # Natural Language Processing
        "glue",
        "super_glue",
        "imdb",
        "sst2",
        "cola",
        "qnli",
        # Tabular Data
        "titanic",
        "adult",
        "credit",
        # Synthetic
        "synthetic",
    ]


def validate_flexible_dataset_config(config: DatasetConfig) -> bool:
    """Validate flexible dataset configuration."""
    try:
        config._validate()
        return True
    except Exception:
        return False


# Export main classes and functions
DatasetConfig = DatasetConfig
FlexibleDatasetLoader = FlexibleDatasetLoader
DatasetLoaderError = DatasetLoaderError

__all__ = [
    "DatasetConfig",
    "DatasetLoaderError",
    "FlexibleDatasetLoader",
    "create_flexible_dataset_config",
    "get_flexible_dataset_loader",
    "list_supported_datasets",
    "validate_flexible_dataset_config",
]


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Flexible dataset loader test")
    parser.add_argument("--dataset", default="cifar10", help="Dataset name")
    parser.add_argument("--subset-size", type=int, default=100, help="Subset size")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--transforms",
        nargs="+",
        default=["basic", "normalize"],
        help="Transform steps",
    )

    args = parser.parse_args()

    # Create flexible configuration
    config = create_flexible_dataset_config(
        args.dataset,
        subset_size=args.subset_size,
        batch_size=args.batch_size,
        transforms=args.transforms,
        seed=42,
    )

    print(f"Created configuration for {config.name}")
    print(f"Input shape: {config.input_shape}")
    print(f"Num classes: {config.num_classes}")
    print(f"Transforms: {config.transforms}")

    # Test configuration validation
    if validate_flexible_dataset_config(config):
        print("✓ Configuration is valid")
    else:
        print("✗ Configuration validation failed")

    # Load dataset if possible
    try:
        loader = get_flexible_dataset_loader(config)
        train_ds, val_ds, test_ds = loader.load_dataset()

        print("✓ Dataset loaded successfully")
        print(f"  Training samples: {len(train_ds)}")
        if val_ds:
            print(f"  Validation samples: {len(val_ds)}")
        if test_ds:
            print(f"  Test samples: {len(test_ds)}")

        # Test DataLoader creation
        train_loader = loader.create_dataloader(train_ds)
        print(f"  Training batches: {len(train_loader)}")

    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
