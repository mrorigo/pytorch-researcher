import json
import os

import pytest

from pytorch_researcher.src.utils import (
    bash_run,
    create_pytorch_project_scaffold,
    list_directory,
    python_run,
    read_file,
    write_file,
)


# Unit tests for read_file
def test_read_file_success(tmp_path):
    file_path = tmp_path / "test.txt"
    file_path.write_text("Hello, world!")
    content = read_file(str(file_path))
    assert content == "Hello, world!"


def test_read_file_empty(tmp_path):
    file_path = tmp_path / "empty.txt"
    file_path.write_text("")
    content = read_file(str(file_path))
    assert content == ""


def test_read_file_not_found():
    with pytest.raises(FileNotFoundError):
        read_file("non_existent_file.txt")


def test_read_file_permission_denied(tmp_path):
    file_path = tmp_path / "no_permission.txt"
    file_path.write_text("secret")
    os.chmod(file_path, 0o000)  # Remove all permissions

    with pytest.raises(PermissionError):
        read_file(str(file_path))

    os.chmod(file_path, 0o644)  # Restore permissions for cleanup


# Unit tests for write_file
def test_write_file_create_new(tmp_path):
    file_path = tmp_path / "new_file.txt"
    assert write_file(str(file_path), "New content") is True
    assert file_path.read_text() == "New content"


def test_write_file_overwrite_existing(tmp_path):
    file_path = tmp_path / "existing_file.txt"
    file_path.write_text("Original content")
    assert write_file(str(file_path), "Overwritten content", overwrite=True) is True
    assert file_path.read_text() == "Overwritten content"


def test_write_file_no_overwrite_existing(tmp_path):
    file_path = tmp_path / "existing_file_no_overwrite.txt"
    file_path.write_text("Original content")
    with pytest.raises(FileExistsError):
        write_file(str(file_path), "New content", overwrite=False)
    assert file_path.read_text() == "Original content"


def test_write_file_create_in_subdirectory(tmp_path):
    sub_dir = tmp_path / "sub_dir"
    file_path = sub_dir / "nested_file.txt"
    assert write_file(str(file_path), "Nested content") is True
    assert sub_dir.is_dir()
    assert file_path.read_text() == "Nested content"


def test_write_file_permission_denied(tmp_path):
    # Create a directory with no write permissions
    no_permission_dir = tmp_path / "no_write_dir"
    no_permission_dir.mkdir()
    os.chmod(no_permission_dir, 0o444)  # Read-only directory

    file_path = no_permission_dir / "blocked_file.txt"
    with pytest.raises(PermissionError):
        write_file(str(file_path), "Should not be written")

    os.chmod(no_permission_dir, 0o755)  # Restore permissions for cleanup


# Unit tests for bash_run
def test_bash_run_success():
    result = bash_run("echo 'Hello from bash!'")
    assert result["stdout"].strip() == "Hello from bash!"
    assert result["stderr"] == ""
    assert result["exit_code"] == 0


def test_bash_run_fail_command_not_found():
    result = bash_run("non_existent_command_xyz")
    assert "not found" in result["stderr"] or "command not found" in result["stderr"]
    assert result["stdout"] == ""
    assert result["exit_code"] != 0


def test_bash_run_fail_exit_code():
    result = bash_run("exit 1")
    assert result["stdout"] == ""
    assert result["stderr"] == ""
    assert result["exit_code"] == 1


def test_bash_run_stdout_stderr():
    result = bash_run("echo 'output'; >&2 echo 'error'")
    assert result["stdout"].strip() == "output"
    assert result["stderr"].strip() == "error"
    assert result["exit_code"] == 0


# Unit tests for python_run
def test_python_run_success(tmp_path):
    script_path = tmp_path / "hello.py"
    script_path.write_text("print('Hello from Python script!')")
    result = python_run(str(script_path))
    assert result["stdout"].strip() == "Hello from Python script!"
    assert result["stderr"] == ""
    assert result["exit_code"] == 0


def test_python_run_with_args(tmp_path):
    script_path = tmp_path / "args_script.py"
    script_path.write_text(
        "import sys\nname = sys.argv[1] if len(sys.argv) > 1 else 'World'\nprint(f'Hello, {name}!')\n"
    )
    result = python_run(str(script_path), args=["TestUser"])
    assert result["stdout"].strip() == "Hello, TestUser!"
    assert result["stderr"] == ""
    assert result["exit_code"] == 0


def test_python_run_fail_script_error(tmp_path):
    script_path = tmp_path / "error_script.py"
    script_path.write_text("raise ValueError('Something went wrong!')")
    result = python_run(str(script_path))
    assert "ValueError: Something went wrong!" in result["stderr"]
    assert result["stdout"] == ""
    assert result["exit_code"] != 0


def test_python_run_not_found():
    with pytest.raises(FileNotFoundError):
        python_run("non_existent_python_script.py")


# Unit tests for list_directory
def test_list_directory_success(tmp_path):
    (tmp_path / "file1.txt").write_text("content")
    (tmp_path / "subdir1").mkdir()
    (tmp_path / "file2.txt").write_text("content")
    contents = list_directory(str(tmp_path))
    assert set(contents) == {"file1.txt", "subdir1", "file2.txt"}


def test_list_directory_empty(tmp_path):
    contents = list_directory(str(tmp_path))
    assert contents == []


def test_list_directory_not_found():
    with pytest.raises(FileNotFoundError):
        list_directory("non_existent_dir_xyz")


def test_list_directory_not_a_directory(tmp_path):
    file_path = tmp_path / "a_file.txt"
    file_path.write_text("content")
    with pytest.raises(FileNotFoundError):  # Should be FileNotFoundError as per plan
        list_directory(str(file_path))


def test_list_directory_permission_denied(tmp_path):
    no_permission_dir = tmp_path / "no_read_dir"
    no_permission_dir.mkdir()
    os.chmod(no_permission_dir, 0o000)  # No permissions at all

    with pytest.raises(PermissionError):
        list_directory(str(no_permission_dir))

    os.chmod(no_permission_dir, 0o755)  # Restore permissions for cleanup


# Unit tests for create_pytorch_project_scaffold
def test_create_pytorch_project_scaffold_success(tmp_path):
    project_name = tmp_path / "MyTestProject"
    project_root = create_pytorch_project_scaffold(str(project_name))
    assert os.path.isdir(project_root)
    assert os.path.isdir(os.path.join(project_root, "src"))
    assert os.path.isdir(os.path.join(project_root, "configs"))
    assert os.path.isdir(os.path.join(project_root, "experiments"))
    registry_path = os.path.join(project_root, "experiments", "registry.json")
    assert os.path.exists(registry_path)
    with open(registry_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert content.strip() == "[]"


def test_create_pytorch_project_scaffold_already_exists(tmp_path):
    project_name = tmp_path / "ExistingProject"
    os.makedirs(project_name)
    with pytest.raises(FileExistsError):
        create_pytorch_project_scaffold(str(project_name))


def test_create_pytorch_project_scaffold_permission_denied(tmp_path, monkeypatch):
    project_name = tmp_path / "BlockedProject"
    # Simulate permission error by monkeypatching os.makedirs to raise OSError
    original_makedirs = os.makedirs

    def raise_os_error(path, *args, **kwargs):
        raise OSError("Permission denied")

    monkeypatch.setattr(os, "makedirs", raise_os_error)
    with pytest.raises(PermissionError):
        create_pytorch_project_scaffold(str(project_name))
    # Restore original for safety
    monkeypatch.setattr(os, "makedirs", original_makedirs)
