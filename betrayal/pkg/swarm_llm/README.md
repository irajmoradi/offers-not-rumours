# Swarm LLM: cooperation, betrayal, covert influence, and emergence

Open-model experiments for the swarm alignment sprint. Everything runs from
`swarm_llm.ipynb` (Kaggle / your own GPU / Colab). Modules:

| File | What it does |
|---|---|
| engine.py | Core key-trading game, ledger, instances, scripted baseline |
| llm_game.py | LLM agents: give / post / warn / defect, covert D-Evil influencer, visibility, metrics, interviews |
| llm_deterrence.py | Deterrence vs. containment with LLM agents |
| emergence.py | Leader emergence, role types, scaling analysis (sim) |
| real_agents.py | Route 1: run a real framework (AutoGen/CrewAI) you control; framework-agnostic graph core |
| public_data.py | Route 2: analyze public multi-agent datasets (e.g. MAST) with the same metrics |
| transparent_probe.py | Route 3: transparent, identified, rate-limited observer for authorized public endpoints |
| exports.py | Datasets for Adaption (messages, decisions, skeptic examples, d_evil_messages) |
| viz.py | Charts for every experiment (colorblind-safe, multi-panel) |
| backends.py | transformers / vLLM / offline mock model backends |

The emergence metrics (leader share, in-degree centralization, role mix, single-hub)
are the SAME across the simulator and all three real-world routes, so you can compare
how leaders and failures scale in sim vs. real agents vs. public data.

Real-world routes are measurement, not attack. Route 3 only runs against endpoints
you list and mark authorized, identifies itself, and sends no hidden text.
