# PyTorch ML Research Agent - Phase 0 Plan: Foundations & Core Utilities

## 1. Introduction

Phase 0 focuses on establishing the fundamental operational environment and core file system interaction tools required by the PyTorch ML Research Agent. This initial phase is crucial for laying a robust groundwork before integrating any LLM capabilities. The primary goal is to ensure the agent can reliably manage project directories, read/write files, and execute arbitrary shell and Python commands.

## 2. Objective

Establish the basic operational environment and essential file system interaction tools. This phase does not involve LLMs directly, but ensures the agent can perform fundamental operations:
*   Project directory creation and management.
*   Reliable file system operations (read, write, list).
*   Execution of `bash` commands with output capture.
*   Execution of Python scripts with output capture.

## 3. Environment Setup & Core Dependencies

Before implementing the tools, ensure the development environment is correctly set up as per project guidelines:

### A. Package Management
*   **Tool**: `uv` will be used for dependency management.
*   **Action**: Ensure `uv` is installed and accessible.

### B. Virtual Environment
*   **Location**: The Python virtual environment must be located in `.venv` within the project root.
*   **Activation**: All development and execution will occur within this virtual environment, activated using `.venv/bin/activate`.
*   **Action**: Create and activate the `.venv` if it doesn't exist.

### C. Core Dependencies
*   **Initial Dependencies**: While not strictly used in Phase 0, the `openai` Python library is a core project dependency and should be installed from the outset to avoid future integration issues.
*   **Action**: Install `openai` into the `.venv` using `uv pip install openai`.

## 4. Key Tools to Implement (with Detailed Specifications)

Each tool will be implemented as a modular Python function or class, adhering to clear input/output contracts and robust error handling.

### 4.1. `read_file`

*   **Purpose**: Safely read the content of a specified file.
*   **Input**: `file_path` (string) - The absolute or relative path to the file to be read.
*   **Output**: `file_content` (string) - The content of the file.
*   **Error Handling**:
    *   Raise a `FileNotFoundError` if the `file_path` does not exist.
    *   Raise a `PermissionError` if the agent does not have read permissions.
    *   Handle encoding issues (e.g., default to UTF-8, provide an option for others).

### 4.2. `write_file`

*   **Purpose**: Safely write content to a specified file, creating it if it doesn't exist or overwriting it if it does.
*   **Input**:
    *   `file_path` (string) - The absolute or relative path to the file.
    *   `content` (string) - The content to write to the file.
    *   `overwrite` (boolean, default: `True`) - If `False`, raise an error if the file already exists.
*   **Output**: `True` upon successful write.
*   **Error Handling**:
    *   Raise a `FileExistsError` if `overwrite` is `False` and the file already exists.
    *   Raise a `PermissionError` if the agent does not have write permissions to the directory or file.
    *   Ensure parent directories are created if they don't exist.

### 4.3. `bash_run`

*   **Purpose**: Execute a shell command in the environment and capture its standard output, standard error, and exit code.
*   **Input**: `command` (string) - The shell command to execute.
*   **Output**: A dictionary containing:
    *   `stdout` (string) - The standard output of the command.
    *   `stderr` (string) - The standard error of the command.
    *   `exit_code` (integer) - The exit code of the command.
*   **Error Handling**:
    *   Should *not* raise an exception for non-zero exit codes. Instead, capture `stderr` and `exit_code` to allow the calling agent to decide how to handle the result.
    *   Handle timeout conditions if a command takes too long (optional for MVP, but good to consider for future).

### 4.4. `python_run`

*   **Purpose**: Execute a Python script using the currently active Python interpreter and capture its standard output, standard error, and exit code. This is particularly useful for running generated models or utility scripts.
*   **Input**:
    *   `script_path` (string) - The absolute or relative path to the Python script to execute.
    *   `args` (list of strings, optional) - Command-line arguments to pass to the script.
*   **Output**: A dictionary containing:
    *   `stdout` (string) - The standard output of the script.
    *   `stderr` (string) - The standard error of the script.
    *   `exit_code` (integer) - The exit code of the script.
*   **Error Handling**:
    *   Similar to `bash_run`, capture `stderr` and `exit_code` rather than raising exceptions directly.
    *   Raise `FileNotFoundError` if `script_path` does not exist.

### 4.5. `list_directory`

*   **Purpose**: List the contents (files and subdirectories) of a specified directory.
*   **Input**: `directory_path` (string) - The absolute or relative path to the directory.
*   **Output**: A list of strings, where each string is the name of a file or subdirectory within `directory_path`.
*   **Error Handling**:
    *   Raise a `FileNotFoundError` if the `directory_path` does not exist or is not a directory.
    *   Raise a `PermissionError` if the agent does not have read permissions for the directory.

### 4.6. `create_pytorch_project_scaffold`

*   **Purpose**: Create a standardized project directory structure for a new PyTorch research project, including necessary subdirectories and an initial `registry.json` file.
*   **Input**: `project_name` (string) - The name of the project.
*   **Output**: `project_root_path` (string) - The absolute path to the created project root directory.
*   **Details**:
    *   The project root directory will be created under the agent's current working directory.
    *   Inside the project root, the following structure will be created:
        *   `src/`: For model code and utility scripts.
        *   `configs/`: For model configurations (e.g., YAML files).
        *   `experiments/`: To store experiment results and the `registry.json`.
    *   An empty `registry.json` file will be created inside the `experiments/` directory. This file should contain an empty JSON array `[]` initially.
*   **Error Handling**:
    *   Raise an `FileExistsError` if a directory with `project_name` already exists.
    *   Raise a `PermissionError` if the agent cannot create directories.

## 5. Minimal Agent Orchestrator (Phase 0)

For Phase 0, a minimal orchestrator will be a simple Python script or a basic function that sequentially calls these newly implemented tools. Its purpose is solely to demonstrate the correct functioning and integration of the foundational tools. It will not involve any LLM interaction or complex decision-making.

## 6. Deliverables

*   Python module(s) containing the implemented `read_file`, `write_file`, `bash_run`, `python_run`, `list_directory`, and `create_pytorch_project_scaffold` tools.
*   A minimal Python script demonstrating the sequential use of these tools, proving their basic functionality.
*   Comprehensive unit tests for each tool.

## 7. Test-Verified Approach

A rigorous testing methodology using `pytest` will ensure the reliability of each foundational tool.

### A. Unit Tests

Each tool will have dedicated unit tests covering various scenarios:

*   **`read_file`**:
    *   Test reading an existing file with content.
    *   Test reading an empty file.
    *   Test attempting to read a non-existent file (expect `FileNotFoundError`).
    *   Test reading a file with specific permissions (e.g., no read access, if feasible in testing environment).
*   **`write_file`**:
    *   Test creating a new file with content.
    *   Test overwriting an existing file.
    *   Test writing to a file in a non-existent subdirectory (expect directory creation).
    *   Test attempting to write where no permissions exist (expect `PermissionError`).
    *   Test `overwrite=False` when file exists (expect `FileExistsError`).
*   **`bash_run`**:
    *   Test a simple command that succeeds (e.g., `echo "hello"`), verify `stdout` and `exit_code`.
    *   Test a command that fails (e.g., `false` or `nonexistent_command`), verify `stderr` and `exit_code` (non-zero).
    *   Test commands that produce both `stdout` and `stderr`.
*   **`python_run`**:
    *   Test executing a simple Python script that prints to `stdout`.
    *   Test executing a Python script that takes arguments.
    *   Test executing a Python script that raises an exception (verify `stderr` and non-zero `exit_code`).
    *   Test attempting to run a non-existent script (expect `FileNotFoundError`).
*   **`list_directory`**:
    *   Test listing a directory with multiple files and subdirectories.
    *   Test listing an empty directory.
    *   Test attempting to list a non-existent directory (expect `FileNotFoundError`).
    *   Test attempting to list a file path instead of a directory path (expect `FileNotFoundError` or similar).
*   **`create_pytorch_project_scaffold`**:
    *   Test creating a new project with the specified structure (`src/`, `configs/`, `experiments/`).
    *   Verify that an empty `registry.json` (`[]`) is created within `experiments/`.
    *   Test attempting to create a project with a name that already exists (expect `FileExistsError`).

### B. Integration Test Scenario

An end-to-end integration test will simulate a basic agent workflow using the foundational tools.

1.  **Setup**: The test should create a temporary working directory to prevent interference with other tests or the actual project structure.
2.  **`create_pytorch_project_scaffold("MyTestProject")`**:
    *   Call the scaffold tool.
    *   **Assertion**: Verify `MyTestProject` directory exists, and `src/`, `configs/`, `experiments/` subdirectories exist within it.
    *   **Assertion**: Verify `MyTestProject/experiments/registry.json` exists and contains `[]`.
3.  **`list_directory("MyTestProject")`**:
    *   Call the list directory tool on the newly created project.
    *   **Assertion**: Verify the output list contains `src`, `configs`, `experiments`.
4.  **`write_file("MyTestProject/src/temp_script.py", "print('Hello from temp script')")`**:
    *   Call the write file tool.
    *   **Assertion**: Verify `temp_script.py` exists and contains the correct content by using `read_file`.
5.  **`python_run("MyTestProject/src/temp_script.py")`**:
    *   Call the python run tool.
    *   **Assertion**: Verify `stdout` contains "Hello from temp script" and `exit_code` is 0.
6.  **`bash_run("mkdir MyTestProject/configs/temp_config")`**:
    *   Call the bash run tool.
    *   **Assertion**: Verify `exit_code` is 0.
    *   **Assertion**: Verify `MyTestProject/configs/temp_config` exists by using `list_directory` on `MyTestProject/configs`.
7.  **Cleanup**: `bash_run("rm -rf MyTestProject")` or use a context manager/fixture for temporary directories.
    *   **Assertion**: Verify `MyTestProject` no longer exists.

### C. Verification

All unit tests and the integration test scenario must pass successfully. This verifies that:
*   Each tool functions correctly in isolation.
*   The tools can be used together in a sequence to perform basic project management and command execution.
*   The agent has a reliable foundation for subsequent LLM-driven phases.