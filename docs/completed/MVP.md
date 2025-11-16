# PyTorch ML Research Agent - Minimum Viable Product (MVP.md)

## 1. Introduction

This MVP document outlines the essential features and tools required to rapidly demonstrate the core thesis of the PyTorch ML Research Agent: **the strategic integration of LLMs at multiple levels to accelerate PyTorch model architectural exploration through intelligent planning, code generation, and analysis, underpinned by a clear feedback loop and results tracking.**

The goal of this MVP is to establish a foundational system where an autonomous agent can receive a high-level architectural description, leverage its **own planning LLM** to define the concrete `model_config`, then use a **tool's internal LLM** to generate functional PyTorch code, quickly understand its structure, and run a minimal validation to gauge its immediate promise. This proves the critical LLM-empowered loop and the capability to make iterative decisions based on structured feedback, before scaling to full training, hyperparameter optimization, or complex project management.

## 2. MVP Goals

*   **Prove Agent-LLM Driven Configuration:** Demonstrate the agent's ability to utilize its dedicated "planning LLM" to translate a high-level architectural intent into a structured `model_config` dictionary.
*   **Prove Tool-LLM Driven Architecture Prototyping:** Demonstrate the agent's ability to use the `pytorch_model_assembler`'s **internal LLM** to generate valid PyTorch `nn.Module` code from a declarative `model_config`.
*   **Validate Tool-LLM Powered Model Introspection:** Show that the agent can utilize the `pytorch_model_summary`'s **internal LLM** to analyze generated (or existing) model code and produce structural summaries.
*   **Establish Rapid Feedback Loop:** Enable the agent to perform quick, low-epoch evaluations on generated models to provide immediate viability feedback. Crucially, this feedback will be structured and used by the agent for subsequent decisions.
*   **Implement Basic Experiment Tracking:** Store generated artifacts (configs, code, summaries, results) in a structured way that allows the agent to review past experiments.
*   **Minimize Tool Implementation Complexity:** Highlight how LLM integration (both agent's planning LLM and tools' internal LLMs) simplifies development effort by offloading complex reasoning and generation.
*   **Enable Basic Autonomous Iteration:** Allow the agent to perform a rudimentary cycle of "plan config -> generate code -> summarize -> quick evaluate -> analyze & revise" until a stopping criterion is met.

## 3. Agent Capabilities (MVP)

Upon implementation of these tools, the agent will be capable of:

*   Reading and writing code/files.
*   Executing basic shell commands.
*   **Using its planning LLM (e.g., via OpenAI API) to dynamically define PyTorch `nn.Module` architectures as declarative `model_config` dictionaries, based on a high-level goal and past experiment results.**
*   **Leveraging the `pytorch_model_assembler`'s internal LLM to generate the actual Python code for `nn.Module` architectures from these `model_config`s.**
*   **Getting instant structural and complexity summaries of generated models, with the `pytorch_model_summary`'s internal LLM performing the code analysis.**
*   Performing rapid, low-epoch training and evaluation on models using common benchmark/dummy datasets to gauge initial promise.
*   Scaffolding a basic PyTorch project directory with an experiment tracking mechanism.
*   **Analyzing quick evaluation results and model summaries, using its planning LLM, to decide whether to iterate on the architecture, declare success (based on a predefined threshold), or stop due to lack of progress.**

## 4. Runtime Environment (MVP)

The MVP will operate within a Linux environment with the following components, as detailed in the full PRD:

*   **Bash Shell**
*   **Python 3.11** (with `venv`)
*   **PyTorch**
*   **CUDA (or MPS device for local testing)**
*   **LLM Integration:**
    *   **Agent's Planning LLM:** Access to a general-purpose LLM (e.g., gpt-oss:20b) via an API client (e.g., OpenAI API). This LLM assists the *agent itself* in higher-level reasoning, generating structured inputs for tools, and interpreting outcomes.
    *   **Tools' Internal LLM:** A dedicated interface and runtime for interacting with an LLM specifically used *within* tools like `pytorch_model_assembler` and `pytorch_model_summary` for specialized code generation and analysis tasks.

## 5. Tools Specification (MVP)

The MVP will focus on a critical subset of tools from the full PRD. These tools are sufficient to demonstrate the core LLM-accelerated research loop.

### A. Foundational Environment & File Management Tools (MVP)

These are the absolute minimum required for agent operation.

#### 1. `read_file`
*   **Description**: Reads and returns the entire content of a specified file.
*   **Purpose**: Essential for understanding existing code, configurations, or logs.
*   **Exact Syntax**: `read_file(file_path: str) -> str`

#### 2. `write_file`
*   **Description**: Overwrites the entire content of a specified file. If the file does not exist, it will be created.
*   **Purpose**: Creating new Python modules, saving LLM-generated code, or updating experiment logs/registry.
*   **Exact Syntax**: `write_file(file_path: str, content: str) -> None`

#### 3. `bash_run`
*   **Description**: Executes a given shell command.
*   **Purpose**: Running `ls`, `mkdir`, `pip install` for core dependencies.
*   **Exact Syntax**: `bash_run(command: str) -> str`

#### 4. `python_run`
*   **Description**: Executes a Python script using the active `python3.11` virtual environment.
*   **Purpose**: Running data preprocessing or simple utility scripts.
*   **Exact Syntax**: `python_run(script_path: str, *args: str) -> str`

#### 5. `list_directory`
*   **Description**: Lists the contents (files and directories) of a specified directory.
*   **Purpose**: Helps the agent understand the project structure and locate relevant files.
*   **Exact Syntax**: `list_directory(path: str = '.', recursive: bool = False) -> list[str]`

#### 6. `create_pytorch_project_scaffold`
*   **Description**: Generates a standard, opinionated directory structure for a new PyTorch research project, including an `experiments/registry.json`.
*   **Purpose**: Quickly set up a well-organized project for experiments and their tracking.
*   **Exact Syntax**: `create_pytorch_project_scaffold(project_name: str, base_dir: str = '.') -> None`
    *   *Note*: The scaffold will be very minimal for MVP, primarily focused on `src/models/`, `configs/`, and `experiments/` containing `registry.json`.

### B. PyTorch Model Development & Prototyping Tools (MVP Core)

These are the central tools demonstrating LLM integration.

#### 7. `pytorch_model_assembler`
*   **Description**: Dynamically constructs a PyTorch `nn.Module` class based on a declarative configuration (dictionary) by **prompting its *internal LLM* to generate the corresponding Python code**. This code is then saved to a specified Python file.
*   **Purpose**: Rapid prototyping of new model architectures, leveraging the tool's internal LLM for complex code generation.
*   **Exact Syntax**:
    ```python
    pytorch_model_assembler(
        model_config: dict[str, Any], # Dictionary defining the model's structure.
        output_script_path: str       # Path where the generated Python file with the nn.Module class will be saved.
    ) -> None
    ```
    *   **`model_config` Structure**: As defined in the full PRD, but MVP would focus on simpler layer sequences.

#### 8. `pytorch_model_summary`
*   **Description**: Dynamically loads a PyTorch `nn.Module` from a script and, leveraging the **tool's *internal LLM*'s code analysis capabilities**, generates a comprehensive summary.
*   **Purpose**: Rapidly inspect and understand the structure and complexity of generated PyTorch models.
*   **Exact Syntax**:
    ```python
    pytorch_model_summary(
        model_script_path: str, # Path to the Python file defining the model class.
        model_class_name: str,  # The name of the nn.Module class within the script.
        input_size: tuple[int, ...] # Dummy input tensor shape.
    ) -> dict[str, Any]
    ```

### C. PyTorch Training & Evaluation Tools (MVP)

A single tool for quick validation.

#### 9. `pytorch_quick_evaluator`
*   **Description**: Performs an ultra-fast, minimal training and evaluation loop on a specified PyTorch model using common benchmark datasets or a dummy dataset.
*   **Purpose**: Rapidly test new model architectures or training regimes to get immediate feedback.
*   **Exact Syntax**:
    ```python
    pytorch_quick_evaluator(
        model_script_path: str,       # Path to the Python file defining the nn.Module class.
        model_class_name: str,        # The name of the nn.Module class within the script.
        eval_config: dict[str, Any]   # Dictionary specifying evaluation parameters.
    ) -> dict[str, Any]
    ```
    *   **`eval_config` Structure**: As defined in the full PRD, supporting "mnist", "cifar10", "dummy" datasets.
    *   **Returns**: A dictionary containing final evaluation metrics (e.g., `{'accuracy': 0.75, 'val_loss': 0.23}`) and training loss, along with a unique `run_id` for this evaluation.

## 6. Internal LLM Capabilities and Role (MVP Focus)

The MVP will clearly distinguish between two types of LLM integration:

### A. Agent's Planning LLM (e.g., gpt-oss:20b via OpenAI API)

This LLM is directly invoked by the *agent itself* for higher-level reasoning, planning, and generation of structured inputs for the specialized tools.

*   **Role**:
    *   **Configuration Generation (Primary MVP Role):** Takes a high-level natural language prompt describing a desired model architecture (e.g., "a small CNN for CIFAR-10 with 3 conv layers and 2 linear layers") and any relevant prior experiment results, then generates the corresponding declarative `model_config` dictionary (as expected by `pytorch_model_assembler`).
    *   **Task Planning & Tool Argument Generation:** Assists the agent in breaking down a goal into a sequence of tool calls and generating precise arguments for those calls (e.g., determining `input_size` for `pytorch_model_summary` based on the `model_config`).
    *   **Results Analysis & Decision Making (Crucial for Feedback Loop):** Interprets structured tool outputs (model summaries, evaluation results) and the experiment registry, providing natural language insights, assessing goal progress, and deciding the next action (e.g., iterate, stop, revise initial goal).
*   **Requirements**: General language understanding, ability to generate structured JSON/dict outputs, basic knowledge of ML concepts, context management (to remember past interactions and results).

### B. Tools' Internal LLM (Specialized Backend for Tools)

This LLM is *embedded within* specific tools (`pytorch_model_assembler`, `pytorch_model_summary`) and is invoked by the tool itself to perform specialized, programmatic tasks that require deep code understanding or generation.

*   **Role**:
    *   **Code Generation (Critical for `pytorch_model_assembler`):** Takes a structured `model_config` dictionary and generates correct, idiomatic Python code for PyTorch `nn.Module` classes. This includes handling imports, class definitions, `__init__`, `forward` methods, and inferring component interconnections (e.g., `in_features`).
    *   **Code Analysis & Interpretation (Critical for `pytorch_model_summary`):** Takes PyTorch `nn.Module` Python code and a dummy input shape, then analyzes it to extract structural information, parameter counts, and generate a formatted summary string.
*   **Requirements**: Deep knowledge of PyTorch API, code generation capabilities, code analysis capabilities, context awareness for dimension inference and layer interdependencies.

## 7. Tool Dependencies (External Python Libraries - MVP)

The MVP will rely on these core libraries:

*   `torch`
*   `torchvision` (for `mnist`, `cifar10` in `pytorch_quick_evaluator`)
*   `PyYAML` (for config parsing if needed for `model_config` persistence)
*   **LLM API Client:** (e.g., OpenAI Python client, Hugging Face `transformers` for local LLMs, or custom API client for internal LLM) - **Essential for *both* LLM integrations.**

## 8. Agent Interaction Model (MVP) - The Feedback Loop

The MVP agent will follow a simplified, direct research loop, highlighting the dual LLM interaction and iterative decision-making:

1.  **Project Setup**:
    *   The agent receives a high-level initial research goal from a human user (e.g., "Design a simple CNN for CIFAR-10 aiming for >70% accuracy on quick evaluation").
    *   Agent calls `create_pytorch_project_scaffold("MyMVPCIFAR10Project")`. This creates the project directory and an empty `experiments/registry.json`.

2.  **Architecture Definition (Agent's Planning LLM)**:
    *   The agent prompts its **planning LLM** with the initial goal, potentially including any current project context or the `registry.json` (if empty, it's just the goal).
    *   **Planning LLM Task**: Generate a suitable `model_config` dictionary based on the goal.
    *   The **planning LLM** returns the structured `model_config` dictionary.
    *   The agent saves this `model_config` to `configs/models/current_model_config.yaml` using `write_file`.

3.  **Code Generation (Tool's Internal LLM)**:
    *   The agent calls `pytorch_model_assembler` using the saved `model_config`, specifying `output_script_path="src/models/current_model.py"`.
    *   The `pytorch_model_assembler` uses its **internal LLM** to generate the actual PyTorch `nn.Module` Python code, saving it to `src/models/current_model.py`.

4.  **Model Inspection (Tool's Internal LLM)**:
    *   The agent calls `pytorch_model_summary("src/models/current_model.py", "CurrentModel", (1, 3, 32, 32))`. (The `input_size` might be determined by the agent's planning LLM based on dataset info in the initial goal).
    *   The `pytorch_model_summary` tool uses its **internal LLM** to analyze the code and return a structured summary.
    *   The agent saves this summary as `experiments/<run_id>/summary.json` (see Section 9).

5.  **Quick Validation**:
    *   The agent calls `pytorch_quick_evaluator` with the generated model (`"src/models/current_model.py"`, `"CurrentModel"`) and an `eval_config` (e.g., `{"dataset": "cifar10", "num_classes": 10, "epochs": 3, ...}`).
    *   The tool returns the `EvaluationResults` including metrics and a `run_id`.
    *   The agent saves these results to `experiments/<run_id>/results.json` and updates the `experiments/registry.json`.

6.  **Analysis & Decision Making (Agent's Planning LLM) - The Feedback Loop Closure**:
    *   The agent gathers all relevant information:
        *   Original high-level goal.
        *   Current `model_config`.
        *   Current `model_summary` from `experiments/<run_id>/summary.json`.
        *   Current `EvaluationResults` from `experiments/<run_id>/results.json`.
        *   The full `experiments/registry.json` to provide historical context.
    *   The agent prompts its **planning LLM** with all this context, asking for:
        *   An assessment of current progress towards the original goal.
        *   Identification of strengths/weaknesses of the current model/config.
        *   A clear decision:
            *   **Iterate**: Suggest specific modifications to the `model_config` for the next attempt (e.g., "add another convolutional layer," "reduce kernel size," "try ReLU instead of GELU").
            *   **Goal Achieved**: Declare success if performance threshold is met.
            *   **Stop/Unpromising**: Advise stopping if repeated attempts yield no progress or the current path seems unfeasible.
    *   The **planning LLM** returns this analysis and decision.

7.  **Iteration / Stopping**:
    *   **If Iterate**: The agent uses the **planning LLM's** suggested modifications to generate a *new* `model_config` (re-prompting the planning LLM as in step 2, but with additional context on desired changes). The loop restarts from step 2.
    *   **If Goal Achieved or Stop/Unpromising**: The agent formally terminates the research loop and provides a final report to the user.

## 9. Experiment Tracking & Results Storage (MVP)

To enable the feedback loop, structured storage of experiment artifacts is crucial.

*   **Project Structure**: The `create_pytorch_project_scaffold` tool creates:
    ```
    MyMVPCIFAR10Project/
    ├── src/
    │   └── models/
    │       └── current_model.py     # Latest LLM-generated model code
    ├── configs/
    │   └── models/
    │       └── current_model_config.yaml # Latest LLM-generated model config
    ├── experiments/
    │   ├── registry.json             # Central manifest of all quick eval runs
    │   └── <run_id>/                 # Unique directory for each quick eval run
    │       ├── model_config.yaml     # Snapshot of model_config used for this run
    │       ├── model_code.py         # Snapshot of model code used for this run
    │       ├── summary.json          # Model summary from pytorch_model_summary
    │       └── results.json          # Quick evaluation metrics
    └── requirements.txt
    ```

*   **`registry.json`**: This central file is a list of all quick evaluation runs.
    *   **Purpose**: Allows the agent's Planning LLM to quickly get an overview of past attempts without parsing individual files, providing historical context for decision-making.
    *   **Structure**:
        ```json
        [
            {
                "run_id": "qe_20231027_153001",
                "timestamp": "2023-10-27T15:30:01Z",
                "model_name": "CurrentModel",
                "dataset": "cifar10",
                "metrics_summary": {"accuracy": 0.65, "val_loss": 0.35},
                "experiment_dir": "experiments/qe_20231027_153001/",
                "decision": "Iterate: suggested increasing layers"
            },
            {
                "run_id": "qe_20231027_153520",
                "timestamp": "2023-10-27T15:35:20Z",
                "model_name": "CurrentModel",
                "dataset": "cifar10",
                "metrics_summary": {"accuracy": 0.72, "val_loss": 0.28},
                "experiment_dir": "experiments/qe_20231027_153520/",
                "decision": "Goal Achieved: >70% accuracy met"
            }
        ]
        ```
    *   The agent updates this `registry.json` after each `pytorch_quick_evaluator` call and subsequent decision point.

## 10. Stopping Criteria & Goal Achievement (MVP)

The agent needs a clear mechanism to determine when its research goal is met or when to abandon a path.

*   **Explicit Goal Definition**: The initial prompt from the user will include clear, measurable criteria for success (e.g., "achieve >70% accuracy on CIFAR-10 quick evaluation").
*   **Planning LLM Evaluation**: After each quick evaluation, the agent's **planning LLM** is provided with the `registry.json`, the current `EvaluationResults`, and the original goal. It is tasked with assessing if the success criteria have been met.
    *   **Goal Achieved**: If the `EvaluationResults` meet or exceed the specified criteria (e.g., `metrics_summary.accuracy >= 0.70`), the planning LLM advises the agent that the goal is achieved.
    *   **No Progress / Unpromising**: If, after a predefined number of iterations (e.g., 3-5), the model's performance does not significantly improve, or even degrades, the planning LLM can advise the agent to stop the current exploration path. This prevents infinite loops on unpromising directions.
    *   **Iteration Limit**: A hard limit on the total number of iterations or an overall time budget can also serve as a stopping criterion, ensuring the MVP doesn't run indefinitely.
*   **Reporting**: When a stopping criterion (goal achieved or unpromising) is met, the agent generates a final report to the user, summarizing the findings, the best performing model, and the rationale for stopping. This report will also be saved in the `experiments/` directory.

This enhanced MVP provides a complete, demonstrable loop of intelligent ML research, from high-level goal to iterative refinement and a clear stopping point, critically leveraging LLMs for both strategic planning and tactical code operations.
