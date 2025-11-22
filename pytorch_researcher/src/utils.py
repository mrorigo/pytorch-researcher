"""Utility helpers for file I/O, subprocess management, and scaffolding."""

import os
import subprocess
import sys


def read_file(file_path: str) -> str:
    """Safely reads the content of a specified file.

    Args:
        file_path (str): The absolute or relative path to the file to be read.

    Returns:
        str: The content of the file.

    Raises:
        FileNotFoundError: If the file_path does not exist.
        PermissionError: If the agent does not have read permissions.

    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    if not os.access(file_path, os.R_OK):
        raise PermissionError(f"Permission denied: Cannot read file {file_path}")

    with open(file_path, encoding="utf-8") as f:
        return f.read()


def write_file(file_path: str, content: str, overwrite: bool = True) -> bool:
    """Safely writes content to a specified file, creating it if it doesn't exist or overwriting it if it does.

    Args:
        file_path (str): The absolute or relative path to the file.
        content (str): The content to write to the file.
        overwrite (bool, optional): If False, raise an error if the file already exists. Defaults to True.

    Returns:
        bool: True upon successful write.

    Raises:
        FileExistsError: If overwrite is False and the file already exists.
        PermissionError: If the agent does not have write permissions to the directory or file.

    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    if os.path.exists(file_path) and not overwrite:
        raise FileExistsError(
            f"File already exists: {file_path}. Set overwrite=True to overwrite."
        )

    if not os.access(os.path.dirname(file_path) or ".", os.W_OK):
        raise PermissionError(
            f"Permission denied: Cannot write to directory {os.path.dirname(file_path) or '.'}"
        )

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return True


def bash_run(command: str) -> dict:
    """Execute a shell command and capture its standard output, standard error, and exit code.

    Args:
        command (str): The shell command to execute.

    Returns:
        dict: A dictionary containing 'stdout' (str), 'stderr' (str), and 'exit_code' (int).

    """
    process = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        check=False,  # Do not raise an exception for non-zero exit codes
    )
    return {
        "stdout": process.stdout,
        "stderr": process.stderr,
        "exit_code": process.returncode,
    }


def list_directory(directory_path: str) -> list[str]:
    """List the contents (files and subdirectories) of a specified directory.

    Args:
        directory_path (str): The absolute or relative path to the directory.

    Returns:
        list[str]: A list of strings, where each string is the name of a file or subdirectory within directory_path.

    Raises:
        FileNotFoundError: If the directory_path does not exist or is not a directory.
        PermissionError: If the agent does not have read permissions for the directory.

    """
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    if not os.path.isdir(directory_path):
        raise FileNotFoundError(f"Path is not a directory: {directory_path}")
    if not os.access(directory_path, os.R_OK):
        raise PermissionError(
            f"Permission denied: Cannot read directory {directory_path}"
        )

    return os.listdir(directory_path)


def python_run(script_path: str, args: list[str] | None = None) -> dict:
    """Execute a Python script using the active interpreter and capture output.

    Args:
        script_path (str): The absolute or relative path to the Python script to execute.
        args (list of strings, optional): Command-line arguments to pass to the script.

    Returns:
        dict: A dictionary containing 'stdout' (str), 'stderr' (str), and 'exit_code' (int).

    Raises:
        FileNotFoundError: If script_path does not exist.

    """
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Python script not found: {script_path}")

    if args is None:
        args = []

    # Coerce all args to strings and ensure unbuffered output using -u
    args = [str(a) for a in args]
    # Use sys.executable with -u for reliable interpreter invocation and unbuffered IO
    command = [sys.executable, "-u", script_path, *args]

    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,  # Do not raise an exception for non-zero exit codes
    )

    return {
        "stdout": process.stdout,
        "stderr": process.stderr,
        "exit_code": process.returncode,
    }


def create_pytorch_project_scaffold(project_name: str) -> str:
    """Create a standardized project directory structure and initial registry.

    Args:
        project_name (str): The name of the project.

    Returns:
        str: The absolute path to the created project root directory.

    Raises:
        FileExistsError: If a directory with project_name already exists.
        PermissionError: If the agent cannot create directories.

    """
    project_root_path = os.path.abspath(project_name)

    if os.path.exists(project_root_path):
        raise FileExistsError(f"Project directory already exists: {project_root_path}")

    try:
        os.makedirs(project_root_path)
    except OSError as e:
        raise PermissionError(f"Permission denied: Cannot create directory {project_root_path} - {e}") from e

    subdirectories = ["src", "configs", "experiments"]
    for subdir in subdirectories:
        os.makedirs(os.path.join(project_root_path, subdir))

    registry_path = os.path.join(project_root_path, "experiments", "registry.json")
    with open(registry_path, "w", encoding="utf-8") as f:
        f.write("[]")  # Initialize with an empty JSON array

    return project_root_path
