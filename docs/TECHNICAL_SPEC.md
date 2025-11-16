# PyTorch ML Research Agent - Technical Specification (MVP)

## Enhanced Evaluation Metrics (Recently Implemented)

This specification has been updated to include comprehensive evaluation metrics capabilities that transform the system from basic accuracy tracking to a research-grade evaluation platform suitable for independent ML researchers and academic institutions.

### Overview

The enhanced evaluation metrics system provides:
- **Publication-Ready Metrics**: F1, precision, recall, AUC, PR-AUC, confusion matrices
- **Statistical Rigor**: Multi-seed evaluation with confidence intervals
- **Per-Class Analysis**: Detailed performance analysis for imbalanced datasets
- **Learning Dynamics**: Overfitting detection, convergence analysis, training insights
- **Advanced Visualizations**: Optional confusion matrix plots and learning curves

### Technical Implementation

**Core Files:**
- `pytorch_researcher/src/pytorch_tools/quick_evaluator.py` - Enhanced evaluator (880+ lines)
- `test_enhanced_metrics.py` - Comprehensive test suite (400+ lines)
- `docs/METRICS_PLAN.md` - Complete implementation documentation

**New Methods Added (11 total):**
1. `_compute_classification_metrics()` - F1, precision, recall with sklearn integration
2. `_compute_auc_metrics()` - ROC-AUC and PR-AUC calculations
3. `_compute_per_class_roc_auc()` - Per-class ROC curve analysis
4. `_compute_per_class_analysis()` - Comprehensive per-class metrics
5. `_analyze_confusion_patterns()` - Error pattern analysis
6. `_compute_class_insights()` - Performance insights generation
7. `_plot_confusion_matrix()` - Visualization support
8. `_track_learning_dynamics()` - Training curve analysis
9. `_analyze_overfitting()` - Overfitting detection with severity levels
10. `_analyze_convergence()` - Convergence analysis with stability detection
11. `_generate_training_insights()` - Actionable training recommendations

### Configuration Architecture

The `QuickEvalConfig` class has been extended with comprehensive configuration options:

```python
@dataclass
class QuickEvalConfig:
    # Enhanced metrics configuration
    metrics_to_track: List[str] = field(default_factory=lambda: ["accuracy", "loss"])
    compute_confusion_matrix: bool = False
    compute_per_class_metrics: bool = False
    compute_auc_scores: bool = False
    track_learning_curves: bool = False
    track_gradient_norms: bool = False
    
    # Statistical analysis
    compute_confidence_intervals: bool = True
    confidence_level: float = 0.95
    
    # Output formatting
    detailed_report: bool = False
    save_confusion_matrix_plot: bool = False
    export_learning_curves: bool = False
```

### Metric Categories

**1. Classification Metrics:**
- F1 scores (macro, weighted averages)
- Precision and recall (macro, weighted)
- Support and class distribution
- Accuracy with fallback implementations

**2. AUC Analysis:**
- ROC-AUC (binary and multi-class classification)
- Precision-Recall AUC (PR-AUC) for imbalanced datasets
- Per-class ROC curves with FPR, TPR, thresholds
- Comprehensive AUC metric aggregation

**3. Per-Class Analysis:**
- Individual class precision, recall, F1, specificity
- Confusion matrices (raw and normalized formats)
- Error analysis with top misclassification patterns
- Performance insights with actionable recommendations
- Class distribution analysis and imbalance detection

**4. Learning Dynamics:**
- Training and validation curve tracking
- Overfitting detection with severity classification (mild/moderate/severe)
- Convergence analysis with stability detection
- Gradient norm monitoring and learning rate tracking
- Automated training insights with emoji-enhanced recommendations

### Statistical Framework

**Multi-Seed Evaluation:**
- Configurable number of seeds for statistical significance
- Automatic aggregation across seeds with mean, std, min, max
- 95% confidence intervals for accuracy metrics
- Best seed identification and reporting

**Result Structure:**
```python
{
    "config": { /* QuickEvalConfig serialization */ },
    "model_name": "ResNet18",
    "num_seeds": 3,
    "seed_results": [ /* individual seed results */ ],
    "aggregated": {
        "val_accuracy": {
            "mean": 0.847,
            "std": 0.023,
            "ci_95": [0.824, 0.870],
            "values": [0.85, 0.82, 0.87]
        }
        /* additional aggregated metrics */
    },
    "final": { /* best seed results */ },
    "best_seed": 42,
    "goal_achieved": True,
    "learning_curves": { /* if enabled */ }
}
```

### Dependency Management

**Optional Dependencies:**
- `scikit-learn`: Comprehensive metric calculations (with robust fallbacks)
- `matplotlib`: Visualization capabilities (optional, graceful degradation)

**Fallback Strategy:**
- When sklearn unavailable: Implement basic metric calculations using numpy
- When matplotlib unavailable: Skip visualization, focus on numerical metrics
- Graceful degradation ensures system works in minimal environments

### Performance Considerations

**Computational Overhead:**
- Basic metrics: +10-15% evaluation time
- AUC calculations: +5-10% evaluation time
- Confusion matrix: <5% overhead
- Learning curves: +2-5% per epoch (if enabled)

**Memory Usage:**
- Prediction storage: ~2x current memory for comprehensive analysis
- Learning curves: +10-20% additional memory during training
- Overall: <30% additional memory for full feature set

### Usage Integration

**Basic Enhanced Evaluation:**
```python
from pytorch_researcher.src.pytorch_tools.quick_evaluator import QuickEvalConfig, quick_evaluate_once

config = QuickEvalConfig(
    dataset_name="cifar10",
    num_seeds=3,
    metrics_to_track=["f1_macro", "precision_macro", "recall_macro"],
    compute_confusion_matrix=True
)

result = quick_evaluate_once(model, config)
```

**Advanced Analysis:**
```python
config = QuickEvalConfig(
    track_learning_curves=True,
    compute_per_class_metrics=True,
    detailed_report=True
)

result = quick_evaluate_once(model, config)

# Access comprehensive results
for class_id, metrics in result['final']['per_class_metrics'].items():
    print(f"Class {class_id}: F1={metrics['f1']:.3f}")

if 'learning_curves' in result:
    curves = result['learning_curves']
    print(f"Overfitting: {curves['overfitting_analysis']['overfitting_detected']}")
```

### Testing and Validation

**Test Coverage:**
- Comprehensive unit tests for all new functionality
- Integration tests for complete evaluation pipeline
- Fallback behavior testing when sklearn unavailable
- End-to-end evaluation testing with multiple scenarios

**Success Metrics:**
- All existing functionality preserved (backward compatibility)
- Performance overhead <20% for basic metrics
- Memory overhead <30% for full feature set
- Graceful degradation when optional dependencies unavailable
- Publication-ready statistical analysis capabilities

### Architectural Principles

**Modular Design:**
- Each enhancement adds functionality without breaking existing code
- Optional features with sensible defaults
- Progressive disclosure of advanced capabilities
- Clear separation between basic and enhanced functionality

**Research-Grade Quality:**
- Standard ML evaluation metrics following academic conventions
- Statistical rigor with confidence intervals and multi-seed evaluation
- Comprehensive documentation and usage examples
- Suitable for independent researchers and academic institutions

**Backward Compatibility:**
- All existing APIs preserved
- Default configurations maintain original behavior
- Optional enhancements don't affect existing workflows
- Legacy functions available for specific use cases


## 1. Introduction

This document provides a technical specification for the Minimum Viable Product (MVP) of the PyTorch ML Research Agent. It outlines the architectural decisions, LLM interaction strategy, core components, and development principles to guide the implementation. The primary goal is to ensure a clear, concise, and maintainable codebase that fulfills the objectives laid out in the PRD and MVP documents.

## 2. Overall Architecture

The agent's architecture will follow a modular design, primarily composed of a central orchestrator that interacts with a Planning LLM and a suite of specialized tools. Some tools may, in turn, leverage their own Internal LLMs for specific tasks like code generation or analysis.

```
  +---------------------+      +------------------------+
  |   User Input/Goal   |----->| Agent Orchestrator     |
  +---------------------+      | (Python Application)   |
                               +-----------+------------+
                                           |
                                           v
+------------------------+        +------------------------+
| Planning LLM           |<-------| Decision Making        |
| (e.g., gpt-5.1-mini)   |        | (Agent Orchestrator)   |
|                        |------->| Configuration/Strategy |
+------------------------+        +------------------------+
                                           |
                                           v
+---------------------------------------------------------------+
| Tooling Layer (Python Modules)                                |
|   +--------------------------+  +--------------------------+  |
|   | Foundational Tools       |  | PyTorch-Specific Tools   |  |
|   | (e.g., read_file,        |  | (e.g., model_assembler,  |  |
|   |  write_file, bash_run)   |<->|  model_summary,         |  |
|   +--------------------------+  |  quick_evaluator)        |  |
|                                 +-----------+--------------+  |
|                                             |                 |
|                                             v                 |
|                                  +------------------------+   |
|                                  | Tools' Internal LLM    |   |
|                                  | (e.g., specialized     |   |
|                                  |  backend for code gen) |   |
|                                  +------------------------+   |
+---------------------------------------------------------------+
                                           |
                                           v
+------------------------+        +------------------------+
| Experiment Registry    |<-------| Results Storage &      |
| (registry.json)        |        | Tracking (Orchestrator)|
+------------------------+        +------------------------+
```

## 3. LLM Interaction Strategy

We will use the `openai` Python dependency for all LLM interactions, providing a consistent interface for both the Planning LLM and any Tools' Internal LLMs.

### 3.1 LLM Client Configuration

The `openai` client will be configured to allow independent settings for each type of LLM used by the agent, maximizing versatility. This configuration will be managed dynamically, primarily via a YAML configuration file and environment variables, to avoid hardcoding and facilitate easy switching between environments.

*   **General `openai` client configuration**:
    *   All LLM interactions will use the `openai` Python client.

*   **Planning LLM Configuration**:
    *   **Local Testing**:
        *   `base_url`: `http://localhost:11434/v1` (for Ollama)
        *   `model`: `gpt-oss:20b`
        *   `api_key`: `ollama` (placeholder, or could be empty if not required by Ollama setup)
    *   **Production**:
        *   `base_url`: (default OpenAI API endpoint)
        *   `model`: `gpt-5.1-mini`
        *   `api_key`: Loaded from `OPENAI_API_KEY` environment variable.

*   **Tools' Internal LLM Configuration**:
    *   These LLMs are typically specialized for tasks like code generation or analysis. Their configuration should also be independent.
    *   **Local Testing**:
        *   `base_url`: `http://localhost:11434/v1` (for Ollama)
        *   `model`: `gpt-oss:20b` (or a more specialized local model if available)
        *   `api_key`: `ollama`
    *   **Production**:
        *   `base_url`: (default OpenAI API endpoint)
        *   `model`: `gpt-5.1-mini` (or a more specialized production model if applicable)
        *   `api_key`: Loaded from `OPENAI_API_KEY` environment variable.

The agent will select the appropriate LLM configuration based on the current environment and the LLM's role (Planning or Internal Tool).

### 3.2 Prompt Engineering Principles

*   **Clarity and Conciseness**: Prompts will be designed to be as clear and concise as possible, minimizing ambiguity.
*   **Structured Output**: Where possible, we will leverage LLM capabilities for structured output (e.g., JSON format) to simplify parsing.
*   **Contextual Information**: Prompts will include relevant contextual information (e.g., current goal, experiment history, previous model config) to enable informed decision-making by the LLM.

## 4. Core Components

### 4.1 Agent Orchestrator

The central control flow of the agent. It will:
*   Operate within a dynamically created unique project directory (provided by `create_pytorch_project_scaffold`), using it as its current working directory for all operations related to a specific goal.
*   Parse initial goals.
*   Interact with the Planning LLM for strategic decisions (initial config, next actions).
*   Call appropriate tools based on LLM directives.
*   Manage the experiment lifecycle and update the `registry.json`.
*   Implement stopping criteria.

### 4.2 Planning LLM

The "brain" of the agent, responsible for high-level reasoning:
*   Translating natural language goals into concrete `model_config` dictionaries.
*   Analyzing experiment results from `registry.json`.
*   Deciding the next action: refine model, declare goal achieved, or stop.
*   Generating refinement strategies.

### 4.3 Tools' Internal LLMs

Specialized LLMs integrated within specific tools:
*   **`pytorch_model_assembler`**: Generates PyTorch `nn.Module` code from a `model_config` dictionary.
*   **`pytorch_model_summary`**: Analyzes generated PyTorch code and provides a structured summary (e.g., layers, parameters).
*   Other tools may integrate LLMs as needed for specific, domain-focused tasks.

## 5. Tool Implementation Guidelines

Tools will be implemented as modular, self-contained Python functions or classes. Each tool should:
*   Have a clear, single responsibility.
*   Accept structured input (e.g., dictionaries, specific types).
*   Return structured output for easy parsing by the orchestrator or other tools.
*   Be thoroughly unit-tested.
*   Adhere to the "Keep It Simple, Stupid" (KISS) principle to ensure maintainability.

Reference the `PRD2.md` and `MVP.md` for a detailed list of tools and their specifications.

## 6. Experiment Management

A central `registry.json` file (as outlined in `HIGH_LEVEL_PLAN.md` and `MVP.md`) will be used to track all experiments. Each entry in the registry will contain:
*   A unique experiment ID.
*   Timestamp.
*   Model configuration used.
*   Key performance metrics (e.g., accuracy, loss).
*   Path to generated model code.
*   Decision made by the Planning LLM (e.g., "refine", "goal achieved").

The `write_file` tool will be enhanced or a dedicated utility will be created to manage atomic updates to this registry, ensuring data integrity.

## 7. Error Handling and Logging Strategy

A robust error handling and logging strategy is crucial for an autonomous agent.

*   **Structured Logging**: All logs will be structured (e.g., JSON format) to facilitate automated parsing and analysis.
*   **Per-Agent Logging**: Logs will be organized per agent run, allowing clear isolation of an agent's activities.
*   **Correlation IDs**: A unique correlation ID will be generated for each main goal initiated. This ID will be propagated through all subsequent experiments and actions, enabling end-to-end tracing of an agent's workflow from the initial goal to individual experiments and tool calls.
*   **Granular Log Levels**: Standard logging levels (DEBUG, INFO, WARNING, ERROR, CRITICAL) will be used appropriately.
*   **Actionable Error Messages**: Error messages will be designed to be clear and provide sufficient context for debugging and decision-making.

## 8. Global Configuration Management

A central configuration mechanism will be implemented using a YAML file to manage agent-wide settings. This includes, but is not limited to:

*   **Default Paths**: Configuration for standard directories, relative to the agent's CWD (e.g., `src/`, `configs/`, `experiments/`).
*   **LLM Configuration**: Default models, base URLs, and API key environment variable names for Planning and Tools' Internal LLMs.
*   **Maximum Iterations**: Limits on the number of iterations an agent can perform for a given goal.
*   **Retry Policies**: Configuration for retrying failed tool calls or LLM interactions (e.g., number of retries, backoff strategy).
*   **Evaluation Settings**: Default epochs, batch sizes, or dataset splits for `pytorch_quick_evaluator`.

This YAML file will be loaded at agent startup, providing a flexible way to tune the agent's behavior without code changes.

## 9. Testing Framework and Methodology

To ensure the reliability and correctness of the agent and its tools, we will adhere to a rigorous testing methodology.

*   **Framework**: `pytest` will be used as the primary testing framework.
*   **Code Coverage**: `pytest-cov` will be integrated to monitor and enforce code coverage, ensuring that a high percentage of the codebase is covered by tests.
*   **Unit Tests**: Comprehensive unit tests for all individual functions, classes, and tools. Mocking will be extensively used for external dependencies (e.g., LLM APIs, file system operations) to isolate the component under test.
*   **Integration Tests**: Tests that verify the interaction between multiple components or tools.
*   **End-to-End Tests**: Scenarios that simulate the full agent workflow, from goal initiation to completion, verifying the overall logic and decision-making.

## 10. Development Principles

*   **KISS (Keep It Simple, Stupid)**: Prioritize simplicity and clarity in design and implementation. Avoid unnecessary complexity.
*   **Test-Driven Development (TDD)**: All features, especially tools and core agent logic, will be developed with a strong emphasis on comprehensive unit and integration tests.
*   **Modularity**: Components should be loosely coupled to allow for independent development, testing, and future extensibility.
*   **Readability**: Code will be well-commented and adhere to standard Python style guides (e.g., PEP 8).
*   **Security**: Be mindful of potential security implications, especially when executing `bash_run` or interacting with external APIs.

## 11. Environment Setup

*   **Package Management**: `uv` will be used for dependency management.
*   **Virtual Environment**: All development and execution will occur within a Python virtual environment located at `.venv`. It will be activated using `.venv/bin/activate`.
*   **Dependencies**: The `openai` library will be a core dependency. Specific PyTorch and data science libraries will be managed via `pip_install_pytorch_package` tool where appropriate.
