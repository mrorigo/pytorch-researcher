# ML Research Agent - Examples

This directory contains examples and guides for using the ML Research Agent Orchestrator.

## Quick Start

### Prerequisites

1. **Activate the virtual environment:**
   ```bash
   . .venv/bin/activate
   ```

2. **Set PYTHONPATH to the repository root:**
   ```bash
   export PYTHONPATH=.
   ```

3. **Ensure a Planning LLM HTTP endpoint is running:**
   - For local testing with Ollama: `http://localhost:11434/v1`
   - For production with OpenAI: `https://api.openai.com/v1`

### Basic Usage

Run the orchestrator with a high-level research goal:

```bash
PYTHONPATH=. python -m pytorch_researcher.src.agent_orchestrator \
  --goal "Design a small CIFAR-10 CNN targeting >70% accuracy" \
  --llm-base-url http://localhost:11434/v1 \
  --llm-model gpt-oss:20b \
  --keep \
  --verbose
```

### Command-Line Options

- `--goal, -g`: High-level research goal (required)
- `--name, -n`: Optional project/run name (auto-generated if not provided)
- `--keep`: Keep artifacts after run (default: removes temporary files)
- `--llm-base-url`: Planning LLM base URL (required)
- `--llm-model`: Planning LLM model name (default: gpt-oss:20b)
- `--llm-api-key`: Optional LLM API key
- `--max-iter`: Maximum iterations (default: 5)
- `--target-accuracy`: Target accuracy for stopping (default: 0.7)
- `--verbose, -v`: Enable debug logging

### Example Goals

Here are some example research goals you can try:

1. **CIFAR-10 CNN Optimization:**
   ```
   "Design a small CNN for CIFAR-10 aiming for >70% accuracy with minimal parameters"
   ```

2. **Architecture Exploration:**
   ```
   "Explore different CNN architectures for CIFAR-10 to find the best accuracy-efficiency trade-off"
   ```

3. **Transfer Learning:**
   ```
   "Design a transfer learning approach for CIFAR-10 using a pre-trained backbone"
   ```

4. **Custom Dataset:**
   ```
   "Create a simple MLP for a custom tabular dataset with 10 features and binary classification"
   ```

### Understanding the Output

When you run the orchestrator, it creates a timestamped directory (e.g., `run_20251116T214854Z/`) containing:

```
run_<timestamp>/
├── src/
│   └── models/          # Generated model sources
├── experiments/
│   └── registry.json    # Complete audit trail
└── configs/             # Configuration files (if any)
```

#### Registry Structure

The `experiments/registry.json` contains a complete audit trail:

```json
[
  {
    "run_id": "run-2025-11-16T21:48:54Z",
    "timestamp": "2025-11-16T21:48:54Z",
    "goal": "Design a small CIFAR-10 CNN targeting >70% accuracy",
    "report": {
      "goal": "...",
      "iterations": [...],
      "final_status": "achieved"
    }
  }
]
```

Each iteration entry includes:
- Model configuration used
- Assembly results (LLM prompts, responses, generated code)
- Sandbox validation results
- Model summaries
- Quick evaluation metrics
- Planning LLM decisions

### Advanced Usage

#### Custom Project Names

```bash
python -m pytorch_researcher.src.agent_orchestrator \
  --goal "Your research goal" \
  --name "my_custom_experiment" \
  --llm-base-url http://localhost:11434/v1 \
  --llm-model gpt-oss:20b
```

#### Production Setup

For production use with OpenAI:

```bash
export OPENAI_API_KEY="your-api-key-here"
python -m pytorch_researcher.src.agent_orchestrator \
  --goal "Your research goal" \
  --llm-base-url https://api.openai.com/v1 \
  --llm-model gpt-5.1-mini \
  --keep
```

#### Multiple Iterations

```bash
python -m pytorch_researcher.src.agent_orchestrator \
  --goal "Your research goal" \
  --llm-base-url http://localhost:11434/v1 \
  --llm-model gpt-oss:20b \
  --max-iter 10 \
  --target-accuracy 0.8
```

### Troubleshooting

#### Common Issues

1. **LLM Connection Errors:**
   - Verify the LLM endpoint is running and accessible
   - Check the base URL format (must end with `/v1`)
   - Ensure API key is correct if required

2. **Import Errors:**
   - Make sure PYTHONPATH is set to the repository root
   - Verify the virtual environment is activated

3. **Sandbox Validation Failures:**
   - Generated model code may have syntax errors
   - Check the registry.json for detailed error messages
   - The orchestrator will attempt to refine the model in subsequent iterations

4. **PyTorch Not Available:**
   - Install PyTorch: `uv pip install torch torchvision`
   - The quick evaluation step will be skipped if PyTorch is not available

#### Debug Mode

Enable verbose logging to see detailed execution information:

```bash
python -m pytorch_researcher.src.agent_orchestrator \
  --goal "Your research goal" \
  --llm-base-url http://localhost:11434/v1 \
  --verbose
```

### Integration with Other Tools

#### Direct Sandbox Usage

You can use the sandbox harness directly for testing generated models:

```python
from pytorch_researcher.src.tools.sandbox.sandbox_runner import run_sandboxed_harness

result = run_sandboxed_harness(
    "path/to/generated_model.py",
    class_name="AssembledModel",
    input_size=(1, 3, 32, 32),
    timeout=60
)
print(result)
```

#### Model Summary

Get detailed information about generated models:

```python
from pytorch_researcher.src.pytorch_tools.model_summary import summarize_model_from_path

summary = summarize_model_from_path(
    "path/to/model.py",
    class_name="AssembledModel",
    input_size=(1, 3, 32, 32)
)
print(summary)
```

### Best Practices

1. **Start Simple:** Begin with straightforward goals and gradually increase complexity
2. **Use Descriptive Goals:** Clear, specific goals yield better results
3. **Monitor Iterations:** Check the registry.json to understand the agent's decision-making
4. **Keep Artifacts:** Use `--keep` to preserve runs for analysis and debugging
5. **Set Realistic Targets:** Adjust `--target-accuracy` based on your dataset and constraints

### Next Steps

- Explore the generated models in the `src/models/` directory
- Analyze the decision patterns in `experiments/registry.json`
- Use insights from successful runs to inform future experiments
- Consider extending the orchestrator for your specific research needs

For more detailed information about the implementation, see the main [README.md](../README.md) and [STATUS.md](../STATUS.md).