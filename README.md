# PyTorch Researcher — README

This repository contains the PyTorch ML Research Agent (prototype). It includes a planning-driven research orchestrator that uses a Planning LLM for strategic decisions, an LLM-backed model assembler, sandbox validation, model summarization, and **comprehensive evaluation metrics**. The system provides end-to-end autonomous research that iteratively explores model architectures to achieve specified goals.

**Enhanced Evaluation Metrics (New!):**
- **Publication-Ready Metrics**: F1, precision, recall, AUC, PR-AUC, confusion matrices
- **Statistical Rigor**: Multi-seed evaluation with confidence intervals
- **Per-Class Analysis**: Detailed performance analysis for each class
- **Learning Dynamics**: Overfitting detection, convergence analysis, training insights
- **Advanced Visualizations**: Confusion matrix plots and learning curves

This README explains how to set up the development environment, run the orchestrator, run tests, and where to find important files and artifacts.

---

Table of Contents
- Project layout
- Prerequisites
- Create and activate virtual environment
- Install dependencies (uv)
- Running tests
- Running the orchestrator
  - Basic usage
  - Advanced configuration
- Using the assembler programmatically
- Generated artifacts and registry
- Configuration and tuning
- Troubleshooting & common failures
- Security & safety notes
- Contributing

---

Project layout (key files)
- `pytorch_researcher/src/utils.py` — Core filesystem & process utilities.
- `pytorch_researcher/src/pytorch_tools/model_assembler.py` — Deterministic programmatic assembler.
- `pytorch_researcher/src/pytorch_tools/model_assembler_llm.py` — HTTP-only LLM assembler (MVP).
- `pytorch_researcher/src/pytorch_tools/model_summary.py` — Model summary utilities.
- `pytorch_researcher/src/pytorch_tools/quick_evaluator.py` — **Enhanced** quick train/eval loops with comprehensive metrics, AUC analysis, confusion matrices, per-class analysis, and learning dynamics tracking.
- `pytorch_researcher/src/agent_orchestrator.py` — Primary orchestrator CLI entrypoint.
- `pytorch_researcher/tests/` — pytest test suite for the above components.
- `test_enhanced_metrics.py` — Comprehensive test suite for enhanced evaluation metrics (400+ lines).
- `docs/METRICS_PLAN.md` — Complete implementation plan and documentation for enhanced metrics.
- `STATUS.md` — human-maintained status notes from local runs.

---

Prerequisites
- Python 3.11+ recommended (the dev environment used Python 3.11).
- `uv` is the preferred package manager for this repo (project conventions). If you do not have `uv`, you can use `pip` within the `.venv`, but follow repo guidance.
- For running the full orchestrator (summary + quick evaluation), install `torch` (CPU build is sufficient) and optionally `torchvision`:
  - `torch` (and `torchvision` if you want dataset support).
- For local LLM generation you'll need an LLM server (Ollama or another local endpoint that implements a `/v1/chat/completions`-compatible API). The planner currently implements a simple HTTP POST to `{base_url}/chat/completions`.

---

Create & activate virtual environment
1. Create a local venv named `.venv` in the repo root (recommended)
```/dev/null/README.md#L1-3
python -m venv .venv
```

2. Activate the venv
```/dev/null/README.md#L4-4
. .venv/bin/activate
```

(If you use `uv` you can still activate `.venv` — the repo's agent notes expect `.venv/bin/activate`.)

---

Install dependencies
- Minimal / project dependencies (install `openai` per project guidelines even though the LLM assembler uses HTTP by default):
```/dev/null/README.md#L1-3
. .venv/bin/activate
uv pip install openai
```

- Install PyTorch (+ torchvision) to enable model summary and quick evaluation:
```/dev/null/README.md#L1-3
. .venv/bin/activate
uv pip install torch torchvision
```
Note: Use an appropriate CPU/GPU wheel for your environment. If you cannot install PyTorch, the project falls back to synthetic datasets and some tests will be skipped.

- Optionally install dev/test tools:
```/dev/null/README.md#L1-2
. .venv/bin/activate
uv pip install pytest
```

If your environment does not use `uv`, you can use `pip` inside the activated `.venv`.

---

Running tests
Run the whole test suite with `pytest` from the repository root (activate `.venv` first):
```/dev/null/README.md#L1-1
pytest -q
```

Expected final state (after dependencies are installed, especially `torch`/`numpy`):
- All tests pass (e.g. `32 passed` in the working dev run that validated the assembler + quick evaluator).

If tests fail, see the Troubleshooting section below.

---

Running the orchestrator
The orchestrator provides a single, unified entrypoint for autonomous research. It lives inside the package; run it as a module.

Important: use the module form so imports resolve correctly.

Basic usage:

```bash
PYTHONPATH=. python -m pytorch_researcher.src.agent_orchestrator \
  --goal "Design a small CIFAR-10 CNN targeting >70% accuracy" \
  --llm-base-url http://localhost:11434/v1 \
  --llm-model gpt-oss:20b \
  --keep \
  --verbose
```

Flags:
- `--goal` — high-level research goal (required)
- `--name` — optional run name (auto-generated if not provided)
- `--keep` — keep the generated run directory (default removes temporary files)
- `--llm-base-url`, `--llm-model`, `--llm-api-key` — LLM settings

What happens during an orchestrator run:
- Creates a timestamped project scaffold `run_<timestamp>` in repo root.
- Uses the Planning LLM to propose model configurations and make iteration decisions.
- Calls the LLM-backed assembler to generate model source (requests structured JSON with a `python` field).
- Validates generated source using `ast.parse`.
- Saves the model in `src/models/model.py` under the run directory.
- Validates generated code in a sandbox subprocess.
- Summarizes the model (counts params; runs a dummy forward pass) — requires PyTorch.
- Performs quick evaluation to assess performance.
- Runs a quick evaluator (small training loop) on a synthetic dataset (fallback) or torchvision dataset if requested.

---

Using the assembler programmatically
You can call the deterministic assembler or the LLM wrapper from Python code.

- Deterministic assembler:
```/dev/null/README.md#L1-8
PYTHONPATH=. python - <<'PY'
from pytorch_researcher.src.pytorch_tools.model_assembler import ModelConfig, assemble_model_code, save_model_code
cfg = ModelConfig(class_name='MyModel', input_shape=(3,32,32), layers=[{"type":"Conv2d","in_channels":3,"out_channels":8,"kernel_size":3}, {"type":"Flatten"}, {"type":"Linear","out_features":10}])
src = assemble_model_code(cfg)
print(src[:400])
save_model_code('out_models/my_model.py', src, overwrite=True)
PY
```

- LLM assembler (HTTP-only):
```/dev/null/README.md#L1-8
PYTHONPATH=. python - <<'PY'
from pytorch_researcher.src.pytorch_tools.model_assembler_llm import assemble_from_config
cfg = {"class_name":"TestModel","layers":[{"type":"Conv2d","in_channels":3,"out_channels":8,"kernel_size":3},{"type":"Flatten"},{"type":"Linear","out_features":10}]}
res = assemble_from_config(cfg, "run_tmp/src/models/model.py", use_llm=True, llm_kwargs={"base_url":"http://localhost:11434/v1","model_name":"gpt-oss:20b","max_retries":2})
print(res.keys())
PY
```

Notes:
- `model_assembler_llm.py` is intentionally implemented as an HTTP-only MVP (no `openai` SDK). It POSTs a prompt to `{base_url}/chat/completions` and expects a response shape containing `choices[0].message.content` (a JSON string with a `python` key).
- The default HTTP timeout is set to 300 seconds. You can adjust `http_timeout` in `pytorch_researcher/pytorch_researcher/src/pytorch_tools/model_assembler_llm.py`.

---

Generated artifacts and registry
- Orchestrator runs create a directory `run_<timestamp>/` in the repository root when run with `--keep`.
- Inside each run directory:
  - `src/models/model.py` — saved model source (LLM or fallback).
  - `experiments/registry.json` — JSON array where each run appends an entry with metadata (prompts, LLM responses, assembly results, sandbox validation, summary, evaluation, planning decisions).
  - Additional artifacts (logs, summaries) may be present.

Example: `run_20251116T214854Z/src/models/model.py`

---

Configuration and tuning
- LLM assembler settings:
  - File: `pytorch_researcher/pytorch_researcher/src/pytorch_tools/model_assembler_llm.py`
  - Key configurable values:
    - `model_name` — LLM model to request (e.g., `gpt-oss:20b`).
    - `base_url` — Base URL for the local LLM server (e.g., `http://localhost:11434/v1`).
    - `api_key` — Optional; included as `Authorization: Bearer <api_key>` if provided.
    - `http_timeout` — HTTP call timeout (default 300s).
    - `max_retries` and `retry_backoff` — retry policy for transient errors.
- Orchestrator:
  - `pytorch_researcher/pytorch_researcher/src/agent_orchestrator.py` — orchestrator entrypoint and flags.

---

Troubleshooting & common failures

1. LLM endpoint unreachable / timeouts
   - Verify the endpoint is running:
```/dev/null/README.md#L1-3
curl -s -X POST 'http://localhost:11434/v1/chat/completions' -H 'Content-Type: application/json' -d '{"model":"gpt-oss:20b","messages":[{"role":"user","content":"ping"}]}' | jq .
```
   - If it times out, either increase `http_timeout` in the assembler or ensure the local model is running and responsive.

2. `ModuleNotFoundError: No module named 'torch'`
   - Install PyTorch in the `.venv`:
```/dev/null/README.md#L1-2
. .venv/bin/activate
uv pip install torch torchvision
```

3. Tests failing due to shape mismatches in generated deterministic models
   - The deterministic assembler attempts to choose sensible defaults (e.g., Conv2d padding defaults changed to preserve spatial dims). If a generated model's `Linear` layer `in_features` does not match the flattened tensor size, check:
     - The assembler's `ModelConfig` `input_shape` usage.
     - Whether padding/stride/kernel_size assumptions match your intended configuration.
   - Unit tests are designed to validate common shapes (32x32 inputs). If you change defaults, adjust tests accordingly.

4. Permission or path issues when running the orchestrator
   - Run from the repository root and ensure `PYTHONPATH=.`, or invoke as module with `python -m pytorch_researcher.src.agent_orchestrator`.

---

Security & safety notes
- The orchestrator and summary/evaluation code may import and execute generated Python code created by the LLM. This is dangerous if the LLM or prompts can cause arbitrary or malicious code to be written.
- For any untrusted LLM outputs, **do not** import or execute the generated code in the main process. Instead:
  - Run a sandboxed subprocess or container that instantiates the model and runs a small smoke test (instantiate class, pass a small random tensor).
  - Only accept artifacts into your experiments registry after passing sandboxed validation.

---

Contributing
- Tests are included — please add tests for new features.
- Follow the project's coding conventions; keep deterministic behavior where tests rely on defaults.
- If you add more LLM integrations, prefer small, testable abstractions and provide fallback deterministic behavior for CI and offline tests.
