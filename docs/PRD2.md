# PyTorch ML Research Agent - Product Requirements Document (PRD.md)

## 1. Introduction

This document details the requirements for building an autonomous agent specialized in PyTorch-based machine learning research. The agent's mission is to efficiently explore, develop, train, and evaluate novel model architectures, layer types, training regimes, and optimization strategies. A key innovation in this design is the strategic integration of a highly capable **Internal Large Language Model (LLM)** within the toolchain, acting as an intelligent backend for complex code generation, analysis, and contextual interpretation. This approach significantly reduces the implementation complexity of the tools while empowering the agent with advanced capabilities for rapid architectural exploration and experimentation.

The core of this system is a set of precisely defined tools that allow the agent to interact with its environment, manage code, rapidly prototype, run experiments, and evaluate results. These tools are designed to maximize the efficiency of the ML research loop, enabling a "pick-and-place" approach to model construction and a quick iteration cycle for testing, now further accelerated by LLM-powered intelligence.

## 2. Goals

*   **Accelerate ML Research:** Drastically reduce the time and effort for the agent to explore new PyTorch model architectures and training configurations.
*   **Enable Autonomous Exploration:** Provide a toolkit that allows the agent to independently define, train, and evaluate models with minimal human intervention.
*   **Promote Modularity via LLM-Driven Assembly:** Facilitate rapid, flexible composition of model components by offloading complex code generation to an LLM, enabling "pick-and-place" of standard layers, custom blocks, and novel streams.
*   **Standardize Experimentation:** Offer consistent interfaces for training, evaluation, and hyperparameter management.
*   **Provide Timely Feedback:** Deliver quick insights into model viability and performance at various stages of the research process.
*   **Reduce Tool Implementation Complexity:** Leverage an internal LLM to handle intricate code generation, interpretation, and analysis logic, thereby simplifying the development effort for the tool creators.

## 3. Agent Capabilities

Upon implementation of these tools, the agent will be capable of:
*   Reading and writing code and configuration files.
*   Executing shell commands and Python scripts.
*   Navigating the file system.
*   **Dynamically assembling PyTorch `nn.Module` architectures from declarative configurations, with the LLM generating the underlying Python code.**
*   **Getting instant structural and complexity summaries of generated or existing models, with the LLM performing the code analysis.**
*   Performing rapid, low-epoch training and evaluation on models to gauge initial promise.
*   Executing full-scale, robust training runs with detailed logging and checkpointing.
*   **Configuring and experimenting with advanced optimizer settings and custom optimizers, potentially with LLM assistance in custom optimizer generation.**
*   Managing Python package dependencies specific to PyTorch.
*   Training, loading, and applying tokenizers for sequence-based tasks.
*   Scaffolding new PyTorch projects with best practices.

## 4. Runtime Environment

The agent will operate within a Linux environment with the following components:
*   **Bash Shell:** For general system commands.
*   **Python 3.11:** The primary programming language.
*   **Python Virtual Environment (venv):** The agent will operate within an isolated Python environment for dependency management.
*   **PyTorch:** The core deep learning framework.
*   **CUDA (if available):** GPU acceleration is assumed and desired for PyTorch operations.
*   **Internal LLM Integration:** A dedicated interface and runtime for interacting with the LLM (details in Section 6).

## 5. Tools Specification

The following tools will be provided to the agent. They are categorized for clarity, but all are part of the core toolkit. The LLM's role, where applicable, is highlighted in the tool descriptions.

---

### A. Foundational Environment & File Management Tools

These are basic, essential tools for environment interaction and file system manipulation.

#### 1. `read_file`
*   **Description**: Reads and returns the entire content of a specified file.
*   **Purpose**: Essential for understanding existing code, configurations, data files, or logs before making modifications or taking action.
*   **Exact Syntax**:
    ```python
    read_file(file_path: str) -> str
    ```
*   **Example Usage**:
    ```python
    read_file("src/models/resnet.py")
    ```

#### 2. `write_file`
*   **Description**: Overwrites the entire content of a specified file with the provided string content. If the file does not exist, it will be created.
*   **Purpose**: Creating new Python modules, configuration files, scripts, or saving modified versions of existing files.
*   **Exact Syntax**:
    ```python
    write_file(file_path: str, content: str) -> None
    ```
*   **Example Usage**:
    ```python
    write_file("src/models/my_new_layer.py", "import torch\nclass MyNewLayer(torch.nn.Module):\n    def __init__(self): super().__init__(); self.linear = torch.nn.Linear(10, 10)\n    def forward(self, x): return self.linear(x)\n")
    ```

#### 3. `append_to_file`
*   **Description**: Appends the provided string content to the end of a specified file. If the file does not exist, it will be created.
*   **Purpose**: Adding logs, extending lists of dependencies, or incrementally adding content.
*   **Exact Syntax**:
    ```python
    append_to_file(file_path: str, content: str) -> None
    ```
*   **Example Usage**:
    ```python
    append_to_file("requirements.txt", "\ntorchmetrics==0.11.4")
    ```

#### 4. `bash_run`
*   **Description**: Executes a given shell command in the bash environment.
*   **Purpose**: Running `ls`, `mkdir`, `rm`, `git` commands, installing Python packages, checking environment status (`nvidia-smi`), or executing shell scripts.
*   **Exact Syntax**:
    ```python
    bash_run(command: str) -> str # Returns stdout and stderr combined
    ```
*   **Example Usage**:
    ```python
    bash_run("ls -lR")
    bash_run("nvidia-smi")
    ```

#### 5. `python_run`
*   **Description**: Executes a Python script using the active `python3.11` virtual environment. Optional command-line arguments can be passed.
*   **Purpose**: Running training scripts, evaluation routines, data preprocessing pipelines, or any other standalone Python program.
*   **Exact Syntax**:
    ```python
    python_run(script_path: str, *args: str) -> str # Returns stdout and stderr combined
    ```
*   **Example Usage**:
    ```python
    python_run("train.py", "--config", "configs/base.yaml", "--epochs", "10")
    ```

#### 6. `list_directory`
*   **Description**: Lists the contents (files and directories) of a specified directory.
*   **Purpose**: Helps the agent understand the project structure, locate relevant files, and navigate the codebase.
*   **Exact Syntax**:
    ```python
    list_directory(path: str = '.', recursive: bool = False) -> list[str]
    ```
*   **Example Usage**:
    ```python
    list_directory("src/models")
    list_directory(".", recursive=True)
    ```

#### 7. `create_directory`
*   **Description**: Creates a new directory at the specified path. It will create parent directories as needed (equivalent to `mkdir -p`).
*   **Purpose**: Organizing experiment results, creating new module directories, or setting up project structure.
*   **Exact Syntax**:
    ```python
    create_directory(path: str) -> None
    ```
*   **Example Usage**:
    ```python
    create_directory("experiments/my_new_run")
    ```

---

### B. PyTorch Model Development & Prototyping Tools

These tools enable rapid model architecture definition and introspection, significantly enhanced by the internal LLM.

#### 8. `pytorch_model_assembler`
*   **Description**: Dynamically constructs a PyTorch `nn.Module` class based on a declarative configuration (dictionary) by **prompting the internal LLM to generate the corresponding Python code**. This code is then saved to a specified Python file. This tool leverages the LLM's understanding of PyTorch patterns to compose standard `torch.nn` layers, custom `nn.Module` blocks (from other Python files), and complex architectural streams, handling boilerplate, imports, `__init__`, and `forward` methods.
*   **Purpose**: Rapid prototyping of new model architectures. The LLM handles the intricate task of converting a high-level model blueprint into functional PyTorch code, drastically accelerating architectural exploration and minimizing manual coding effort for the agent.
*   **Exact Syntax**:
    ```python
    pytorch_model_assembler(
        model_config: dict[str, Any], # Dictionary defining the model's structure.
        output_script_path: str       # Path where the generated Python file with the nn.Module class will be saved.
    ) -> None
    ```
    *   **`model_config` Structure**:
        ```python
        {
            "model_name": "MyNewModel", # Name of the generated nn.Module class.
            "input_shape": [3, 224, 224], # Expected input shape (e.g., [C, H, W]). Crucial for LLM to infer dimensions.
            "layers": [
                # Example: Standard PyTorch layer
                {"type": "Conv2d", "params": {"in_channels": 3, "out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1}},
                {"type": "ReLU", "params": {"inplace": True}},
                {"type": "MaxPool2d", "params": {"kernel_size": 2, "stride": 2}},
                # Example: Custom block from another file (LLM will handle importing)
                {"type": "CustomAttentionBlock", "module_path": "src/models/custom_blocks.py", "params": {"dim": 64, "heads": 8}},
                {"type": "AdaptiveAvgPool2d", "params": {"output_size": [1, 1]}},
                {"type": "Flatten"},
                # Example: Linear layer (LLM will infer in_features from input_shape and preceding layers)
                {"type": "Linear", "params": {"out_features": 10}},
            ]
        }
        ```
*   **Example Usage**:
    ```python
    pytorch_model_assembler(
        model_config={
            "model_name": "SimpleCNN",
            "input_shape": [3, 32, 32],
            "layers": [
                {"type": "Conv2d", "params": {"in_channels": 3, "out_channels": 16, "kernel_size": 3, "padding": 1}},
                {"type": "ReLU"},
                {"type": "MaxPool2d", "params": {"kernel_size": 2}},
                {"type": "Flatten"},
                {"type": "Linear", "params": {"out_features": 10}}
            ]
        },
        output_script_path="src/models/assembled_simple_cnn.py"
    )
    ```

#### 9. `pytorch_model_summary`
*   **Description**: Dynamically loads a PyTorch `nn.Module` from a script and, leveraging the **internal LLM's code analysis capabilities**, generates a comprehensive summary. This includes total parameters, trainable parameters, estimated FLOPs, and a layer-by-layer breakdown with output shapes and parameter counts. The LLM processes the model's source code and configuration to provide these details.
*   **Purpose**: Rapidly inspect and understand the structure and complexity of custom or external PyTorch models. The LLM removes the need for complex internal parsing or reliance on external libraries like `torchinfo` for the tool's implementation.
*   **Exact Syntax**:
    ```python
    pytorch_model_summary(
        model_script_path: str, # Path to the Python file defining the model class.
        model_class_name: str,  # The name of the nn.Module class within the script.
        input_size: tuple[int, ...] # Dummy input tensor shape (e.g., (1, 3, 224, 224)).
    ) -> dict[str, Any]
    ```
    *   **Returns**: A dictionary containing `summary_str` (formatted by LLM), `total_params`, `trainable_params`, `flops`, and `layer_details`.
*   **Example Usage**:
    ```python
    pytorch_model_summary("src/models/my_resnet.py", "MyResNet", (1, 3, 256, 256))
    ```

---

### C. PyTorch Training & Evaluation Tools

These tools streamline the execution and evaluation of training runs, from quick checks to full-scale experiments. Their core logic remains structured for reliability, but the LLM could assist the agent in configuring them.

#### 10. `pytorch_quick_evaluator`
*   **Description**: Performs an ultra-fast, minimal training and evaluation loop on a specified PyTorch model using common benchmark datasets or a dummy dataset.
*   **Purpose**: Rapidly test new model architectures or training regimes to get immediate feedback ("is this working?") during the exploration phase.
*   **Exact Syntax**:
    ```python
    pytorch_quick_evaluator(
        model_script_path: str,       # Path to the Python file defining the nn.Module class.
        model_class_name: str,        # The name of the nn.Module class within the script.
        eval_config: dict[str, Any]   # Dictionary specifying evaluation parameters.
    ) -> dict[str, Any]
    ```
    *   **`eval_config` Structure**:
        ```python
        {
            "dataset": "cifar10",       # One of: "mnist", "cifar10", "dummy". "dummy" generates random data.
            "input_shape": [1, 3, 32, 32], # Required for "dummy" dataset, ignored for others.
            "num_classes": 10,
            "batch_size": 64,
            "epochs": 1,                # Small number of epochs for quick feedback.
            "optimizer": "Adam",        # One of: "Adam", "SGD".
            "learning_rate": 0.001,
            "loss_function": "CrossEntropyLoss", # One of: "CrossEntropyLoss", "MSELoss".
            "metrics": ["accuracy", "f1"],
            "device": "cuda"            # One of: "cuda", "cpu". Defaults to "cuda" if available.
        }
        ```
    *   **Returns**: A dictionary containing final evaluation metrics and training loss.
*   **Example Usage**:
    ```python
    pytorch_quick_evaluator(
        model_script_path="src/models/assembled_simple_cnn.py",
        model_class_name="SimpleCNN",
        eval_config={
            "dataset": "cifar10", "num_classes": 10, "batch_size": 128, "epochs": 3,
            "optimizer": "Adam", "learning_rate": 0.0005, "loss_function": "CrossEntropyLoss",
            "metrics": ["accuracy"]
        }
    )
    ```

#### 11. `pytorch_full_trainer`
*   **Description**: Initiates a full-scale training and evaluation run for a specified PyTorch model, allowing for comprehensive hyperparameter configuration, checkpointing, and detailed logging.
*   **Purpose**: Execute thorough training cycles, save best models, and generate complete performance reports after initial promising results from `pytorch_quick_evaluator`.
*   **Exact Syntax**:
    ```python
    pytorch_full_trainer(
        model_script_path: str,       # Path to the Python file defining the nn.Module class.
        model_class_name: str,        # The name of the nn.Module class within the script.
        trainer_config: dict[str, Any], # Dictionary specifying comprehensive training parameters.
        experiment_name: str = None,  # Optional: Unique name for the experiment run. If None, a timestamped name is generated.
        log_base_dir: str = 'full_experiments' # Base directory for this tool's logs.
    ) -> dict[str, Any]
    ```
    *   **`trainer_config` Structure**:
        ```python
        {
            "dataset": "imagenet",      # Dataset name ("cifar10", "imagenet_subset", "custom_dataset_script").
            "dataset_params": {
                "data_path": "/mnt/data/imagenet",
                "split": "train",
                "tokenizer_config": {"name": "my_bpe_tokenizer", "type": "registered"} # Optional tokenizer configuration.
            },
            "num_classes": 1000,
            "batch_size": 128,
            "epochs": 100,
            "optimizer": {
                "type": "AdamW",
                "params": {"lr": 0.001, "weight_decay": 0.01} # 'type' can also refer to a custom optimizer registered via pytorch_optimizer_configurator
            },
            "scheduler": {
                "type": "CosineAnnealingLR",
                "params": {"T_max": 100, "eta_min": 1e-6}
            },
            "loss_function": "CrossEntropyLoss",
            "metrics": ["accuracy", "f1_macro"],
            "device": "cuda",
            "checkpoint_interval": 5,
            "save_best_model": True,
            "early_stopping": {
                "metric": "val_loss",
                "patience": 10,
                "mode": "min"
            }
        }
        ```
    *   **Returns**: A dictionary with detailed training/validation metrics, best checkpoint path, and final evaluation results.
*   **Example Usage**:
    ```python
    pytorch_full_trainer(
        model_script_path="src/models/my_vision_transformer.py",
        model_class_name="VisionTransformer",
        trainer_config={
            "dataset": "imagenet_subset", "dataset_params": {"data_path": "./data/tiny-imagenet"}, "num_classes": 200,
            "batch_size": 256, "epochs": 50,
            "optimizer": {"type": "AdamW", "params": {"lr": 0.0002, "weight_decay": 0.05}},
            "scheduler": {"type": "ReduceLROnPlateau", "params": {"mode": "min", "factor": 0.1, "patience": 5}},
            "loss_function": "CrossEntropyLoss", "metrics": ["accuracy", "top_5_accuracy"], "device": "cuda",
            "save_best_model": True, "early_stopping": {"metric": "val_loss", "patience": 10}
        },
        experiment_name="vit_imagenet_full_run"
    )
    ```

---

### D. Advanced Configuration & Utility Tools

These tools provide advanced control over dependencies, tokenizers, and core components like optimizers, with the LLM assisting in custom component generation.

#### 12. `pytorch_optimizer_configurator`
*   **Description**: Provides a flexible way to configure, modify, or inject custom optimizers for use in training tools. For custom optimizers, it can leverage the **internal LLM to generate the `torch.optim.Optimizer` subclass code** based on a high-level description or specified algorithm, which is then registered for use.
*   **Purpose**: Empower the agent to experiment with advanced optimizer configurations and novel optimization algorithms, with LLM assistance in rapid custom optimizer creation.
*   **Exact Syntax**:
    ```python
    pytorch_optimizer_configurator(
        action: str,                        # "generate_config", "register_custom_optimizer"
        config_name: str,                   # Name for the generated/registered config/optimizer.
        optimizer_details: dict[str, Any] = None, # Details for config generation or LLM prompt for custom optimizer.
        output_script_path: str = None      # Path to save generated custom optimizer code.
    ) -> dict[str, Any]
    ```
    *   **Action: `generate_config`**
        *   **`optimizer_details`**:
            ```python
            {
                "type": "Adam",             # Base PyTorch optimizer (e.g., Adam, SGD, AdamW).
                "default_params": {"lr": 0.001, "weight_decay": 0.0001},
                "param_groups": [           # Optional: specific settings for parameter groups.
                    {"name_regex": "backbone.*", "lr": 0.0001},
                    {"name_regex": "head.*", "lr": 0.01, "weight_decay": 0.001}
                ]
            }
            ```
        *   **Returns**: A dictionary representation of the optimizer configuration. This implicitly registers the config by making it available for future use by `config_name`.
    *   **Action: `register_custom_optimizer`**
        *   **`optimizer_details`**: A dictionary describing the custom optimizer, which will be **passed to the LLM to generate the Python code**.
            ```python
            {
                "description": "Implement a custom optimizer called 'MomentumAdam' that combines Adam's adaptive learning rates with a stronger Nesterov momentum term, especially for parameters in BatchNorm layers.",
                "base_optimizer": "Adam", # Optional hint for LLM
                "additional_params": {"nesterov_momentum": 0.9}
            }
            ```
        *   **`output_script_path`**: Path where the **LLM-generated Python script** for the custom optimizer will be saved. The class name in the script will be `config_name`.
        *   **Returns**: Confirmation of registration. The generated custom optimizer can then be used by `pytorch_full_trainer`.
*   **Example Usage (Generate Config):**
    ```python
    pytorch_optimizer_configurator(
        action="generate_config",
        config_name="tuned_adamw",
        optimizer_details={
            "type": "AdamW",
            "default_params": {"lr": 0.0005, "weight_decay": 0.01},
            "param_groups": [
                {"name_regex": "features.*", "lr": 0.0001},
                {"name_regex": "classifier.*", "lr": 0.005}
            ]
        }
    )
    ```
*   **Example Usage (Register Custom Optimizer via LLM):**
    ```python
    pytorch_optimizer_configurator(
        action="register_custom_optimizer",
        config_name="MomentumAdam",
        optimizer_details={
            "description": "A new optimizer that is a variant of Adam but applies an aggressive cosine decay schedule to the learning rate for the first 10% of training steps and then a fixed learning rate. It should also have a special 'focal_weight_decay' parameter."
        },
        output_script_path="src/optimizers/momentum_adam.py"
    )
    ```

#### 13. `pytorch_tokenizer_manager`
*   **Description**: Provides comprehensive capabilities for training, loading, and applying tokenizers. It supports common tokenization algorithms (e.g., BPE, WordPiece) using underlying libraries (`tokenizers`, `transformers`) and can integrate with pre-trained Hugging Face tokenizers. The **internal LLM could assist in suggesting optimal tokenizer configurations** for the 'train' action based on dataset characteristics.
*   **Purpose**: Facilitate research into tokenization schemes, allowing the agent to quickly prepare data for models with custom or standard tokenizers.
*   **Exact Syntax**:
    ```python
    pytorch_tokenizer_manager(
        action: str,                         # "train", "load", "apply"
        tokenizer_name: str,                 # A unique name for this tokenizer.
        config: dict[str, Any] = None,       # Configuration for 'train' action.
        tokenizer_path: str = None,          # Path to load/save tokenizer assets.
        text_input: str | list[str] = None,  # Text to tokenize (for 'apply').
        hf_pretrained_id: str = None         # Hugging Face pretrained model ID (for 'load' action for HF tokenizers).
    ) -> dict[str, Any]
    ```
    *   **Action: `train`**
        *   Trains a new tokenizer from a given text corpus.
        *   **`config`**:
            ```python
            {
                "type": "BPE",              # One of: "BPE", "WordPiece", "CharLevel".
                "vocab_size": 30000,
                "data_files": ["path/to/corpus1.txt", "path/to/corpus2.txt"], # List of text files for training.
                "special_tokens": ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
                "min_frequency": 2          # For BPE/WordPiece.
            }
            ```
        *   **Returns**: `{'status': 'success', 'tokenizer_name': tokenizer_name, 'tokenizer_path': tokenizer_path, 'vocab_size': actual_vocab_size}`.
    *   **Action: `load`**
        *   Loads an existing tokenizer, either from a local path or a Hugging Face pretrained ID.
        *   **Returns**: `{'status': 'success', 'tokenizer_name': tokenizer_name, 'tokenizer_info': dict_with_vocab_size_etc}`. This implicitly registers it for use by name.
    *   **Action: `apply`**
        *   Applies a loaded/registered tokenizer to input text.
        *   **Returns**: `{'input_ids': list[list[int]], 'attention_mask': list[list[int]]}`.
*   **Example Usage (Train):**
    ```python
    pytorch_tokenizer_manager(
        action="train",
        tokenizer_name="my_bpe_tokenizer",
        config={
            "type": "BPE",
            "vocab_size": 10000,
            "data_files": ["data/nlp/text_corpus.txt"],
            "special_tokens": ["[PAD]", "[UNK]"]
        },
        tokenizer_path="models/tokenizers/my_bpe"
    )
    ```

#### 14. `pip_install_pytorch_package`
*   **Description**: Installs a PyTorch-related Python package ensuring compatibility with a specified PyTorch and CUDA version.
*   **Purpose**: Quickly and reliably add common PyTorch ecosystem libraries with correct versions, avoiding common installation and compatibility issues.
*   **Exact Syntax**:
    ```python
    pip_install_pytorch_package(
        package_name: str,     # The name of the package to install (e.g., "torchvision", "timm").
        version: str = None,   # Optional specific version (e.g., "0.15.2").
        pytorch_version: str = "2.1.0", # The PyTorch version installed (used for dependency resolution).
        cuda_version: str = "cu118", # The CUDA version for PyTorch (e.g., "cu118", "cpu").
        upgrade: bool = False  # If True, upgrade existing packages.
    ) -> str
    ```
    *   **Returns**: The standard output and error messages from the `pip` installation command.
*   **Example Usage**:
    ```python
    pip_install_pytorch_package("torchmetrics", version="0.11.4", pytorch_version="2.1.0", cuda_version="cu118")
    ```

#### 15. `create_pytorch_project_scaffold`
*   **Description**: Generates a standard, opinionated directory structure for a new PyTorch research project, populating it with essential template files.
*   **Purpose**: Quickly set up a well-organized project, reducing initial setup time and providing a robust framework for new research initiatives.
*   **Exact Syntax**:
    ```python
    create_pytorch_project_scaffold(
        project_name: str,  # The name of the new project (this will be the root directory name).
        base_dir: str = '.' # The directory where the new project folder will be created.
    ) -> None
    ```
    *   **Creates**: A directory structure as detailed in the previous PRD.
*   **Example Usage**:
    ```python
    create_pytorch_project_scaffold("NewImageClassifier")
    ```

---

## 6. Internal LLM Capabilities and Role

The core innovation in this design is the strategic integration of an internal LLM. This LLM is not directly exposed to the agent as a general "chat" tool but is leveraged *within* the specialized tools to perform complex programmatic tasks.

**LLM Requirements:**
*   **Code Generation:** Capable of generating correct, idiomatic Python code for PyTorch `nn.Module` classes, `torch.optim.Optimizer` subclasses, and potentially helper functions, given structured input configurations or high-level descriptions. This includes handling imports, class definitions, `__init__` and `forward` methods, and inferring component interconnections (e.g., `in_features`).
*   **Code Analysis & Interpretation:** Able to analyze existing Python code (e.g., a PyTorch `nn.Module` definition) to extract structural information, parameter counts, and logical flow for summarization.
*   **Contextual Understanding:** Possesses deep knowledge of PyTorch API, common ML patterns, and best practices to ensure generated code is functional and robust.
*   **Error Handling (Internal):** The LLM should be capable of self-correcting or generating explanations for why it couldn't generate valid code, allowing the tools to provide meaningful error messages to the agent.
*   **Reliability & Determinism (as much as possible):** While LLMs are inherently stochastic, for code generation tasks within tools, efforts should be made to ensure high reliability and consistency of output given identical inputs. Temperature settings or few-shot examples could be used.

**LLM's Role in Specific Tools:**

*   **`pytorch_model_assembler` (Primary User of LLM):** The tool sends the `model_config` to the LLM and requests Python code for the `nn.Module`. The LLM returns the code string, which the tool then saves. This offloads all the complex dynamic code generation, import management, and dimension inference logic.
*   **`pytorch_model_summary` (Primary User of LLM):** The tool reads the target model's Python script and potentially a dummy input shape. It sends this information to the LLM, asking for a detailed summary. The LLM then performs the analysis and structures the summary output. This simplifies the tool's implementation by not requiring internal AST parsing or dependency on `torchinfo`/`thop` within the tool itself.
*   **`pytorch_optimizer_configurator` (User of LLM):** For the `register_custom_optimizer` action, the tool sends the `optimizer_details` to the LLM, prompting it to generate the `torch.optim.Optimizer` subclass code. The tool then saves and registers this generated code.
*   **`pytorch_tokenizer_manager` (Potential User of LLM):** For the `train` action, the tool could query the LLM with the dataset characteristics (e.g., average token length, vocabulary diversity from a small sample) to recommend optimal `config` parameters (e.g., `vocab_size`, `type`). This would be an optional, value-add capability.

## 7. Tool Dependencies (External Python Libraries)

The implementation of these tools will likely rely on the following Python libraries, which should be pre-installed or easily installable within the agent's `venv`:

*   `torch` (PyTorch)
*   `torchvision` (for standard datasets in quick/full evaluators)
*   `huggingface/transformers` (for tokenization, and potentially model components in `assembler`)
*   `huggingface/tokenizers` (for custom tokenizer training)
*   `PyYAML` (for config management)
*   `tqdm` (for progress bars in training tools)
*   `scikit-learn` (for F1, precision, recall metrics)
*   **LLM API Client:** (e.g., OpenAI Python client, Hugging Face `transformers` for local LLMs, custom API client for internal LLM) - **Crucial for LLM integration.**

## 8. Agent Interaction Model

The agent is expected to use these tools in an iterative research loop:

1.  **Project Setup**: `create_pytorch_project_scaffold`
2.  **Architecture Definition**: `pytorch_model_assembler` (for new ideas, leveraging LLM code gen) or `read_file` (to inspect existing code).
3.  **Model Inspection**: `pytorch_model_summary` (LLM-powered analysis) to understand complexity.
4.  **Quick Validation**: `pytorch_quick_evaluator` with a dummy or small dataset and few epochs.
5.  **Refinement**: If metrics are poor, go back to step 2 (LLM for new arch) or adjust `eval_config`.
6.  **Optimizer Experimentation**: `pytorch_optimizer_configurator` to define custom optimizers (LLM could generate code) or parameter groups.
7.  **Tokenizer Research**: `pytorch_tokenizer_manager` (LLM could assist in config) to train/load/apply tokenizers for sequence data.
8.  **Full Training**: If quick evaluation shows promise, `pytorch_full_trainer` with a larger dataset, more epochs, and detailed logging.
9.  **Analysis**: `read_file` to review experiment logs and saved results.
10. **Further Exploration**: Based on analysis, repeat the loop with new architectures, training regimes, or hyperparameter adjustments.

## 9. Implementation Considerations for Architect

*   **LLM Integration Layer:** A robust, fault-tolerant interface to the internal LLM is paramount. This includes prompt engineering for code generation/analysis, handling LLM response parsing, error handling (e.g., invalid code generated), and retry mechanisms.
*   **Dynamic Module Loading & Execution:** Still critical. `pytorch_model_assembler`, `pytorch_model_summary`, `pytorch_quick_evaluator`, `pytorch_full_trainer`, and `pytorch_optimizer_configurator` will require robust mechanisms for dynamically loading Python modules and classes (including LLM-generated ones) from arbitrary paths.
*   **Code Validation (Generated Code):** While the LLM generates code, a basic validation step (e.g., `ast.parse`, simple `exec` in a sandbox, `pylint` check) after generation and before writing to file is highly recommended to catch trivial syntax errors.
*   **Environment Isolation:** When `exec`ing or importing LLM-generated code, ensure it runs in a sandboxed environment to prevent side effects or security risks.
*   **Resource Management:** `pytorch_full_trainer` and `pytorch_quick_evaluator` need to gracefully handle GPU availability, memory limits, and process management.
*   **Dependency Management:** The `pip_install_pytorch_package` tool should carefully construct `pip install` commands.
*   **State Management:** How registered optimizers/tokenizers are stored and looked up across tool calls needs to be robust (e.g., a simple in-memory registry for the duration of the agent's run, or a small file-based manifest).
*   **Performance (LLM):** Consider the latency and cost of LLM calls, especially for frequently used tools. Optimize prompts and potentially cache common responses.
*   **Security:** `bash_run` and execution of LLM-generated code are powerful. Strict security measures, including sandboxing and minimal permissions for the agent's runtime, are essential.
