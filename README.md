# Active-Graph Visual RAG — pre-refactor backup (2026-07-06)

Snapshot of the working method + eval pipeline before large refactoring.

## Layout
- `visual_rag_agent/` — the method: agent loop (`src/agent.py`), Active Clue Graph (`src/active_clue_graph.py`), retriever (Qwen3-VL-Embedding dense + ColQwen multi-vector, `src/retriever.py`), DeepSeek judge (`src/judge.py`), prompts/schemas, index-building & eval entrypoints (`scripts/`), eval configs (`config/`), RL reward fns (`rewards/`).
- `eval/` — benchmark run scripts (serve vLLM → predict → DeepSeek judge → per-category scoring) for full SlideVQA-2215, ViDoSeek-1142, MMLongBench-847.

## Notes
- Secrets scrubbed: set `DEEPSEEK_API_KEY` (judge) and `HF_TOKEN` (slurm index build) via env.
- Best checkpoints (on training box, not in repo): 7B v9 step_75 = SlideVQA 75.76 / ViDoSeek 71.54 / MMLB-847 35.30; 3B sj step_75 = 69.16 / 65.15 / 29.52.
