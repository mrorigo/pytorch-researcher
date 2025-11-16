# NEXT_MAJOR_MILESTONE: Phase 4 — Planning LLM & Iteration Loop (Actionable Plan)

Status: draft
Goal: Implement the Agent's Planning LLM and the autonomous iteration/feedback loop (Phase 4 from the High-Level Plan). This milestone turns the existing Phase 0/1/2/3 tooling into an autonomous research loop: Planning LLM → assembler → summarize → quick-eval → Planning LLM decision → iterate/stop.

This document lays out:
- concrete sub-tasks and deliverables,
- test strategy (unit + integration + E2E),
- safety and sandboxing requirements,
- API/CLI design changes,
- acceptance criteria and success metrics,
- a suggested timeline with estimates and priorities.

Target audience: next developer(s) who will implement Phase 4.

---

## High-level objectives (what "done" looks like)

1. A Planning LLM client and prompt layer that can:
   - accept a high-level research goal and produce an initial `model_config` (JSON/dict).
   - accept experiment results and registry context and return a decision: { "action": "refine"|"stop"|"achieve", "reason": "...", "next_config": {...} (optional) }.

2. An orchestrator loop that:
   - orchestrates iterations (assemble → summarize → evaluate → record → decide),
   - respects stopping criteria (goal achieved, max iterations, no improvement),
   - persists detailed per-iteration artifacts to `phaseX_demo_<ts>/experiments/registry.json`.

3. Tests & mocks enabling reliable offline development:
   - Mock Planning LLM responses for unit tests.
   - Deterministic integration tests for the iteration logic.

4. Safety and reproducibility:
   - Generated code is validated and executed in a sandboxed subprocess for summary & evaluation.
   - All decisions and prompts are stored in the registry for auditing and debugging.

---

## Prerequisites / Assumptions

- The repo has Phase 0–3 tooling already implemented and test-verified (as present).
- You have an LLM endpoint available for testing (local or hosted); for local testing `gpt-oss:20b` on a local server is a known working option.
- `torch`, `numpy` (and optionally `torchvision`) are installed in the environment used for summary/eval.
- CI runner may need to skip heavy tests (Mark LLM-dependent & GPU tests appropriately).

---

## API & Design Decisions (recommended)

1. Planning LLM client abstraction
   - Create `pytorch_researcher/src/planning_llm/client.py` exposing:
     - `class PlanningLLMClient` with methods:
       - `propose_initial_config(goal: str, constraints: Optional[dict]) -> Dict`
       - `decide_next_action(goal: str, registry: List[Dict], latest_result: Dict) -> Dict`
     - The client implements HTTP-only MVP (POST to `<base_url>/chat/completions`), parses structured JSON responses, and raises typed exceptions on parse/validation errors.
   - Accept `base_url`, `model_name`, `api_key`, `http_timeout`, `max_retries` in the client constructor.

2. Standardized Planning LLM response schema
   - For deterministic parsing and testing, request Planning LLM to return structured JSON with the exact keys:
     - For initial proposal:
       ```
       {
         "action": "propose",
         "model_config": { ... },
         "notes": "...",
         "metadata": { "assumptions": {...} }
       }
       ```
     - For decision responses:
       ```
       {
         "action": "refine" | "achieve" | "stop",
         "reason": "...",
         "next_config": { ... } // present if action == "refine"
       }
       ```

3. Orchestrator loop interface
   - Add a new orchestrator method: `run_phase4_loop(goal: str, max_iterations: int = 5, target_accuracy: float = 0.7, keep: bool = True, llm_kwargs: dict = {}) -> dict`
   - The loop should:
     1. Call `PlanningLLMClient.propose_initial_config`.
     2. Use existing assembler (LLM-backed or fallback) to create model file.
     3. Validate code syntactically, then run sandboxed summary & evaluation.
     4. Record iteration details into `experiments/registry.json` (include prompts, raw LLM responses, generated code checksum, summary, eval metrics).
     5. Call `PlanningLLMClient.decide_next_action` with full context.
     6. Repeat until `achieve` or `stop` or max iterations reached.

4. Sandboxed execution
   - Implement a small sandbox runner `tools/sandbox_runner.py` (or under package). It should:
     - Launch a subprocess (not in-process import) with a tightly-scoped environment.
     - Accept the path to the generated model file and a small harness script that:
       - imports the module,
       - instantiates the model,
       - runs a single dummy forward pass,
       - prints serialized summary JSON (or returns non-zero on failure).
     - Time-limit the subprocess (e.g., 60s) and capture stdout/stderr for diagnostics.
   - Use the sandbox for both summary and quick evaluation steps to reduce code-execution risks.

---

## Concrete tasks & deliverables

Priority order (shortest path to working Phase 4 MVP):

A. Planning LLM client & prompt templates (Core)
- Task A1: Implement `planning_llm/client.py` (HTTP-only), unit-tested with request/response parsing.
- Task A2: Create prompt templates:
  - `prompts/propose_initial_config.txt` — includes schema, constraints, and example model_config.
  - `prompts/decide_next_action.txt` — instructs LLM how to interpret registry & results and return the decision JSON.
- Deliverable: `PlanningLLMClient` that can be instantiated and returns validated JSON objects.

B. Sandbox runner & validation harness (Safety)
- Task B1: Implement `sandbox_runner` to run an isolated Python process and enforce a timeout.
- Task B2: Create a deterministic harness script template that imports model, instantiates class, runs a forward, prints JSON summary.
- Deliverable: Sandboxed summary & quick-eval harness used by orchestrator.

C. Orchestrator loop & registry integration (Core)
- Task C1: Add `run_phase4_loop` to the orchestrator with CLI flags:
  - `--phase 4`, `--goal`, `--max-iter`, `--target-accuracy`, plus LLM flags.
- Task C2: Implement iteration persistence: append per-iteration objects to `experiments/registry.json` containing:
  - run_id, iteration, timestamp, model_config, assembler.via, llm_response, code_path, code_checksum, summary, eval_metrics, planning_decision.
- Deliverable: CLI-invocable loop that stores full audit trail.

D. Tests, mocking, and CI integration (Quality)
- Task D1: Unit tests for PlanningLLMClient using mocked HTTP responses.
- Task D2: Unit tests for orchestrator decision logic with mocked Planning LLM responses (e.g., always refine, then achieve).
- Task D3: Integration test that runs a short loop with:
  - a mocked Planning LLM (returns pre-canned configs/decisions),
  - deterministic assembler (fallback),
  - sandboxed harness for summary (actual subprocess but using a simple model),
  - verify registry entries and stopping behavior.
- Deliverable: Test suite covering the critical loop paths: refine → refine → achieve, and stop-on-max-iterations.

E. UX / Logging / Observability (Ops)
- Task E1: Log prompts and LLM raw responses to `experiments/` for auditing.
- Task E2: Add verbosity flag `--llm-raw` to optionally save raw LLM body for each call.
- Task E3: Create a small `scripts/diagnose_llm.sh` to test LLM endpoint connectivity and sample responses.

F. Documentation & Examples
- Task F1: Add `docs/NEXT_MAJOR_MILESTONE.md` (this file).
- Task F2: Add a Phase 4 quickstart in `README.md` with example CLI usage and recommended local setup (llm server running).
- Deliverable: Clear instructions for reproducing Phase 4 demos locally.

---

## Testing plan (detailed)

1. Unit tests
   - `tests/planning_llm/test_client.py`:
     - Mock HTTP responses for `propose_initial_config` and `decide_next_action`.
     - Validate schema parsing and error handling (malformed JSON, missing keys).
   - `tests/orchestrator/test_decision_logic.py`:
     - Mock the PlanningLLMClient to return sequences of decisions.
     - Test function `evaluate_decision` (isolated) that interprets LLM decisions correctly.

2. Integration tests (fast)
   - `tests/phase4/integration_loop_mocked_llm.py`:
     - Use a fake PlanningLLM server (or mock client) providing:
       - Iteration 1: propose a simple config.
       - Iteration 1 result: returns "refine" with new config.
       - Iteration 2 result: returns "achieve".
     - Use deterministic assembler (fallback) to avoid flakiness.
     - Use sandbox runner but the harness will be simple and fast (tiny model with forward pass).
     - Assertions:
       - registry entries count == number of iterations,
       - final decision == "achieve",
       - persisted artifacts (model files) exist.

3. End-to-end functional test (manual / CI optional)
   - Runs against a real Planning LLM endpoint.
   - Measures wall-clock time and whether it completes within `max_iter`.
   - For CI, this should be optional/flagged because it requires network and may be slow.

4. Safety tests
   - Ensure sandbox runner rejects models that attempt file system or network access.
   - Provide a malicious-model test case that tries to DoS or leak env info — sandbox must exit non-zero and orchestrator should record failure and mark decision appropriately.

---

## Acceptance criteria / success metrics

- Functional:
  - You can run `python -m pytorch_researcher.src.agent_orchestrator --phase 4 --goal "Design a small CNN for CIFAR10" --max-iter 3` and observe:
    - The loop runs for at most 3 iterations,
    - Each iteration generates a model file, saves summary and evaluation,
    - The registry contains an entry per iteration with LLM prompts and raw responses (when enabled),
    - Loop respects the Planning LLM decision (refine/achieve/stop).

- Tests:
  - Unit tests for Planning LLM client and orchestrator decision logic pass.
  - Integration test with mocked Planning LLM completes deterministically.

- Safety:
  - Generated code is not imported in the main process; sandbox subprocess validates it.
  - Sandbox enforces a timeout and resource constraints adequate for the quick-eval use case.

- Observability:
  - Logs and registry entries provide enough data to reconstruct prompts, LLM responses, and evaluation results for each iteration.

---

## Risks & Mitigations

1. LLM variability and hallucinations
   - Risk: Planning LLM returns malformed JSON or illogical decisions.
   - Mitigation: Enforce strict response schema; validate and fallback to a conservative policy (e.g., stop or use deterministic heuristic) when parsing fails.

2. Unsafe generated code
   - Risk: LLM produces code that does arbitrary I/O or system calls.
   - Mitigation: Sandbox runs in subprocess with limited privileges; do not import generated modules in the main process.

3. Long LLM latency
   - Risk: Local big models are slow; loop becomes unusable interactively.
   - Mitigation: Keep high default `http_timeout` (currently 300s) and provide `--llm-timeout` override. Allow a "dry-run" mode with mocked LLM for fast local development.

4. Flaky evaluation metrics
   - Risk: Quick evaluations on synthetic data are noisy.
   - Mitigation: Use small but deterministic seeds; record random seeds and run multiple seeds for reliability if needed.

---

## Suggested timeline (rough estimates)

Note: estimates assume 1 developer familiar with the codebase, working hours per task.

- Week 0.5 (2–3 days)
  - A1: PlanningLLM client MVP + prompt templates (A1, A2)
  - D1: basic unit tests mocking HTTP responses

- Week 0.5 (2–3 days)
  - B1–B2: Sandbox runner + harness templates
  - Unit tests for sandbox (safety checks)

- Week 1 (4–5 days)
  - C1–C2: Orchestrator loop implementation and registry integration
  - D2: Unit tests for decision & loop behavior with mocked LLM

- Week 0.5 (2–3 days)
  - E1–E3: Logging, CLI flags, diagnostics script
  - F1–F2: Documentation & examples

- Week 0.5 (2–3 days)
  - Integration tests (D3), fix issues, polish.

Total: ~3–4 weeks (approx 15–20 work days) to a solid Phase 4 MVP with tests and sandboxing. You can shorten this by focusing first on a "mocked LLM" loop for internal testing (omit real-LLM integration initially).

---

## Example iteration (concise)

1. User runs orchestrator:
   ```
   PYTHONPATH=. python -m pytorch_researcher.src.agent_orchestrator --phase 4 --goal "Optimize a small CIFAR10 CNN to reach 70% accuracy" --max-iter 5 --keep --llm-base-url http://localhost:11434/v1
   ```

2. Orchestrator:
   - Calls `PlanningLLMClient.propose_initial_config(goal)`.
   - Saves prompt + raw response to run/experiments.
   - Assembles model (LLM assembler or deterministic fallback).
   - Writes model file and computes checksum.
   - Calls sandbox harness to run summary + quick-eval.
   - Appends results to `registry.json`.
   - Calls `PlanningLLMClient.decide_next_action` with registry context and latest result.
   - Repeats until decision == `achieve` or `stop` or max iterations reached.

3. End result:
   - `phase4_run_<ts>/experiments/registry.json` contains the full audit trail of iterations, prompts, raw LLM outputs, model paths, summaries, evals, and final decision.

---

## Next immediate actions I recommend you take now

1. Implement the Planning LLM client and the prompt templates (A1, A2). This unlocks the rest because you can mock it for tests.
2. Implement the sandbox runner (B1–B2) to make all subsequent steps safe.
3. Wire a simple orchestrator loop using mocked Planning LLM responses to validate persistence and stopping criteria.
4. Add unit tests & an integration test that uses the mocked Planning LLM to exercise the full loop.

If you want, I can produce:
- starter skeletons for `planning_llm/client.py`, `tools/sandbox_runner.py`, and the orchestrator loop function with typed signatures and TODOs; or
- example prompt templates for `propose_initial_config` and `decide_next_action` that adhere to the structured JSON schema above.

Which of those would you like me to produce next?
