# Dependencies Documentation

This document provides comprehensive information about the PyTorch ML Research Agent's dependency management, installation procedures, and troubleshooting guide.

## Overview

The PyTorch ML Research Agent uses a modern Python packaging approach with `pyproject.toml` and UV for dependency management. The project is designed with a modular dependency structure to support different use cases and development scenarios.

## Core Dependencies

### Machine Learning & Data Science
- **torch** (>=2.0.0) - Core PyTorch library for tensor operations and neural networks
- **torchvision** (>=0.15.0) - Computer vision utilities and pre-trained models
- **numpy** (>=1.24.0) - Numerical computing foundation
- **pandas** (>=2.0.0) - Data manipulation and analysis

### Dataset Management
- **datasets** (>=2.0.0) - Hugging Face datasets library for loading and processing datasets
- **transformers** (>=4.20.0) - State-of-the-art NLP models and utilities
- **huggingface-hub** (>=0.15.0) - Interface to Hugging Face Hub for model sharing
- **safetensors** (>=0.3.0) - Safe tensor serialization for ML models
- **tokenizers** (>=0.13.0) - Fast tokenizers for NLP preprocessing

### HTTP & API Integration
- **httpx** (>=0.24.0) - Modern HTTP client for API calls
- **aiohttp** (>=3.8.0) - Async HTTP client/server framework
- **requests** (>=2.28.0) - HTTP library for compatibility

### LLM Integration
- **openai** (>=1.0.0) - OpenAI API client for LLM access
- **pydantic** (>=2.0.0) - Data validation and settings management

### Data Processing
- **pyarrow** (>=10.0.0) - Columnar data processing
- **pillow** (>=9.0.0) - Image processing library

### Configuration & Utilities
- **pyyaml** (>=6.0) - YAML configuration file parsing
- **tqdm** (>=4.64.0) - Progress bars for long-running operations
- **jinja2** (>=3.1.0) - Template engine for configuration
- **typing-extensions** (>=4.5.0) - Extended type hints support

## Optional Dependency Groups

### Development Dependencies (`dev`)
```bash
uv pip install -e ".[dev]"
```

Includes:
- **pytest** - Testing framework
- **pytest-mock** - Mocking utilities for tests
- **pytest-asyncio** - Async test support
- **pytest-cov** - Test coverage reporting
- **pytest-xdist** - Parallel test execution
- **black** - Code formatting
- **isort** - Import sorting
- **flake8** - Code linting
- **mypy** - Static type checking
- **pre-commit** - Git hooks for code quality
- **ruff** - Fast Python linter

### Enhanced Evaluation (`evaluation`)
```bash
uv pip install -e ".[evaluation]"
```

Includes additional packages for advanced evaluation:
- Extended dataset support
- Enhanced transformers functionality
- Additional Hugging Face integrations

### Computer Vision (`vision`)
```bash
uv pip install -e ".[vision]"
```

Specialized computer vision dependencies:
- **opencv-python** (>=4.8.0) - Computer vision library
- **albumentations** (>=1.3.0) - Image augmentation library

### Natural Language Processing (`nlp`)
```bash
uv pip install -e ".[nlp]"
```

NLP-specific dependencies:
- **spacy** (>=3.6.0) - Industrial-strength NLP
- **nltk** (>=3.8.0) - Natural Language Toolkit

### Production Deployment (`production`)
```bash
uv pip install -e ".[production]"
```

Production-ready dependencies:
- **gunicorn** (>=21.0.0) - WSGI HTTP Server
- **fastapi** (>=0.100.0) - Modern web framework
- **uvicorn** (>=0.23.0) - ASGI server
- **prometheus-client** (>=0.17.0) - Metrics collection
- **structlog** (>=23.0.0) - Structured logging

### Documentation (`docs`)
```bash
uv pip install -e ".[docs]"
```

Documentation generation:
- **sphinx** (>=7.0.0) - Documentation generator
- **sphinx-rtd-theme** (>=1.3.0) - Read the Docs theme
- **myst-parser** (>=2.0.0) - Markdown parser for Sphinx
- **nbsphinx** (>=0.9.0) - Jupyter notebook integration
- **jupyter** (>=1.0.0) - Jupyter ecosystem

### GPU Support (`gpu`)
```bash
uv pip install -e ".[gpu]"
```

GPU-specific dependencies:
- **torch[cuda]** - PyTorch with CUDA support
- **torchvision[cuda]** - TorchVision with CUDA support
- **nvidia-ml-py** (>=12.535.0) - NVIDIA GPU monitoring
- **psutil** (>=5.9.0) - System and process utilities

### All Dependencies
```bash
uv pip install -e ".[all]"
```

Installs all optional dependency groups.

## Installation Guide

### 1. Basic Installation
```bash
# Clone the repository
git clone https://github.com/pytorch-researcher/pytorch-ml-research-agent.git
cd pytorch-ml-research-agent

# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install core dependencies
uv pip install -e .
```

### 2. Development Installation
```bash
# Install with development dependencies
uv pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### 3. Full Installation
```bash
# Install with all optional dependencies
uv pip install -e ".[all]"
```

### 4. Selective Installation
```bash
# Install specific feature sets
uv pip install -e ".[dev,evaluation,gpu]"

# Install for specific use case
uv pip install -e ".[dev,vision]"  # For computer vision development
uv pip install -e ".[dev,nlp]"     # For NLP development
```

## Environment Setup

### Python Version
- **Minimum**: Python 3.11
- **Recommended**: Python 3.11 or 3.12
- **Tested**: Python 3.11.14

### UV Package Manager
This project uses UV for dependency management. Install UV first:

```bash
# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using pip
pip install uv
```

### Virtual Environment
```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -e ".[dev]"
```

## Development Workflow

### 1. Code Quality
```bash
# Run all linting tools
uv run black pytorch_researcher tests
uv run isort pytorch_researcher tests
uv run ruff check pytorch_researcher tests
uv run mypy pytorch_researcher

# Or use the combined command
uv run pre-commit run --all-files
```

### 2. Testing
```bash
# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=pytorch_researcher --cov-report=html

# Run specific test categories
uv run pytest -m "not slow"        # Skip slow tests
uv run pytest -m "unit"            # Run only unit tests
uv run pytest -m "integration"     # Run only integration tests
```

### 3. Documentation
```bash
# Build documentation
uv run sphinx-build -b html docs docs/_build/html

# Serve documentation locally
cd docs/_build/html && python -m http.server 8000
```

## Dependency Management

### Adding New Dependencies
1. Edit `pyproject.toml` to add the dependency
2. Update the dependency groups as appropriate
3. Test the installation: `uv pip install -e ".[dev]"`
4. Update this documentation

### Updating Dependencies
```bash
# Update all dependencies
uv pip install --upgrade -e ".[dev]"

# Update specific package
uv pip install --upgrade torch

# Update with constraints
uv pip install --upgrade "torch>=2.0.0,<3.0.0"
```

### Dependency Resolution
```bash
# Check for dependency conflicts
uv pip check

# Resolve dependencies and create lock file
uv lock

# Install from lock file
uv pip install --no-deps -r requirements.txt
```

## Troubleshooting

### Common Issues

#### 1. CUDA/GPU Issues
```bash
# Install CUDA-enabled PyTorch
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

#### 2. Hugging Face Dataset Downloads
```bash
# Set cache directory
export HF_HOME=/path/to/cache
export TRANSFORMERS_CACHE=/path/to/cache

# Disable telemetry
export TRANSFORMERS_VERBOSITY=error
```

#### 3. Memory Issues
```bash
# Reduce batch size in evaluation
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0  # Disable MPS on macOS

# Use CPU-only mode
export CUDA_VISIBLE_DEVICES=""  # Force CPU usage
```

#### 4. Import Errors
```bash
# Ensure PYTHONPATH is set correctly
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Reinstall in development mode
uv pip install -e ".[dev]" --force-reinstall
```

### Environment Variables

#### Recommended Environment Variables
```bash
# Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# PyTorch settings
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export CUDA_VISIBLE_DEVICES=""

# Hugging Face settings
export HF_HOME="${HOME}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"

# OpenAI API (if using)
export OPENAI_API_KEY="your-api-key-here"

# Development settings
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export TORCH_BACKENDS_CUDNN_ALLOW_TF32=True
```

### Performance Optimization

#### 1. Memory Usage
- Use smaller batch sizes for evaluation
- Enable gradient checkpointing for large models
- Use mixed precision training when possible

#### 2. CPU Optimization
```bash
# Set number of CPU threads
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

#### 3. GPU Optimization
```bash
# Enable TF32 for better performance on RTX cards
export TORCH_BACKENDS_CUDNN_ALLOW_TF32=True

# Use pinned memory for data loading
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

## CI/CD Integration

### GitHub Actions Example
```yaml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.11, 3.12]
    
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install UV
      run: pip install uv
    
    - name: Install dependencies
      run: uv pip install -e ".[dev]"
    
    - name: Run tests
      run: uv run pytest
    
    - name: Run linting
      run: |
        uv run black --check pytorch_researcher tests
        uv run ruff check pytorch_researcher tests
        uv run mypy pytorch_researcher
```

## Security Considerations

### Dependency Scanning
```bash
# Install safety for vulnerability scanning
uv pip install safety

# Scan for known vulnerabilities
safety check

# Scan with JSON output
safety check --json
```

### License Compliance
```bash
# Install pip-licenses
uv pip install pip-licenses

# Generate license report
uv run pip-licenses --format=table --with-urls
```

## Version Compatibility

### Tested Combinations
- **Python 3.11** + **PyTorch 2.0+** + **CUDA 11.8**
- **Python 3.11** + **PyTorch 2.0+** + **CPU-only**
- **Python 3.12** + **PyTorch 2.0+** + **CUDA 12.1**

### Known Issues
- **Python 3.10**: Not officially supported (may work but not tested)
- **Very old PyTorch versions**: May have compatibility issues with newer dependencies
- **Apple Silicon**: Some packages may require specific installation flags

## Support

For dependency-related issues:
1. Check this documentation first
2. Search existing GitHub issues
3. Create a new issue with:
   - Python version (`python --version`)
   - UV version (`uv --version`)
   - Operating system
   - Complete error message
   - Steps to reproduce

## Contributing

When adding new dependencies:
1. Update `pyproject.toml` with appropriate version constraints
2. Add to relevant optional dependency groups
3. Update this documentation
4. Test installation with `uv pip install -e ".[dev]"`
5. Ensure CI passes with new dependencies

---

*This documentation is maintained alongside the project. Last updated: 2025-11-17*