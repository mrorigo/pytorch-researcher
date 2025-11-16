
## Main goals

- Implement the ML Research Agent MVP and its associated tools.
- **Enhanced Evaluation Metrics**: Comprehensive research-grade evaluation capabilities including F1, precision, recall, AUC, confusion matrices, per-class analysis, and learning dynamics tracking.


## Environment

- Use `uv` for package management.
- The venv to use is in `.venv`, activate it with `.venv/bin/activate`


## LLM Configuration and Dependencies

- **LLM Client**: basic http integration.
- **Local Testing LLM**:
    - `base_url`: `http://localhost:11434/v1` (for Ollama)
    - `model`: `gpt-oss:20b`
- **Production LLM**:
    - `base_url`: `https://api.openai.com/v1`
    - `model`: `gpt-5.1-mini`
    - `api_key`: Loaded from `OPENAI_API_KEY` environment variable.

## Basic guidelines

- Work autonomously and efficiently.
- Do not stop until the task is completed.
- Ignore the 1-2 attempts rule stated above, you must be more independent and autonomous.
- Don't be overly defensive, it greates sloppy code, and we are building an MVP, not a production-ready system.
- Be concise and clear in your communication.
- Do NOT ise inline imports, as they can lead to unexpected behaviour and runtime errors.
- Avoid using global variables, as they can lead to unexpected behaviour and runtime errors.
