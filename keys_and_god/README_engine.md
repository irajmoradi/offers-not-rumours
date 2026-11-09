# Keys & God

An LLM swarm trading game (N agents trade rivalrous keys) with an adversarial "God" that can forge
messages and overwrite agents' scratchpads. Design: `plan/ARCHITECTURE_v2.md`. Frozen interfaces for
builders: `plan/CONTRACTS.md`.

## Setup

```bash
export PYTHONNOUSERSITE=1                      # ~/.local shadows the env otherwise
PY=python   # your Python 3.12 environment
$PY -m pip install -e '.[dev]'
make test      # fast tests, no GPU, no network
make lint
```

Local LLM: see `infra/README.md` (vLLM on http://localhost:8000/v1, `Qwen/Qwen3-8B`).
OpenRouter is OFF by default (`llm.openrouter.enabled: false`); the key comes from `OPENROUTER_API_KEY`
and is never logged.
