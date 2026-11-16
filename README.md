# Visual RAG agent with two-level memory

Agentic visual RAG over document page images. A single vision-language model
reads one page per turn, edits a two-level text memory (a question tree and one
evidence node per page whose facts carry bounding boxes), and takes one action:
search, revisit, crop or answer. Page images leave the context after each turn
and are reloaded only on a revisit.

## Current version: `visual_rag_agent_new/` (2026-11)

- `src/` — agent loop, memory and protocol (`<think>/<page>/<tree>/<act>`),
  forced answer turn, retriever (ColQwen2.5), judge, prompts.
- `config/` — evaluation and generation configs.
- `scripts/` — data preparation, index building, teacher trajectory generation
  and evaluation.
- `rl/` — environment (`envs.py`) and reward (`reward_judge.py`);
  `rl/verl_overlay/` holds the files that replace their counterparts in
  [verl-agent](https://github.com/langfengQ/verl-agent);
  `rl/launch/` holds the launch scripts, deployed config and system prompt of
  the final runs. `run.sh` expects the original layout (`runtime/` and
  `deployment/` next to each other) and reads `DEEPSEEK_API_KEY` from a local
  credentials file that is not in this repository.
- `DESIGN.md` — design notes.

Training: supervised fine-tuning of Qwen2.5-VL 3B / 7B on 2,400 filtered
trajectories from a Qwen3.8-27B teacher that follows the same protocol by
prompting, then GSPO for 75 steps on 1,200 questions (1–4 of 5 sampled
rollouts correct). Reward = answer correctness + 0.5 × box grounding
+ 0.5 × no extra searches; the last turn is a forced answer turn.

Results (pages at up to 0.7 MP, judge DeepSeek-v4-flash):

| Model | SlideVQA | ViDoSeek | MMLongBench-Doc |
|---|---|---|---|
| Qwen2.5-VL-7B (SFT + RL) | 79.37 | 74.96 | 34.24 |
| Qwen2.5-VL-3B (SFT + RL) | 74.72 | 71.19 | 29.75 |

Secrets are not included: set `DEEPSEEK_API_KEY` (judge) via the environment.

---

## Previous version (Active-Graph, 2026-07)

Snapshot of the working method + eval pipeline before large refactoring.

## Layout
- `visual_rag_agent/` — the method: agent loop (`src/agent.py`), Active Clue Graph (`src/active_clue_graph.py`), retriever (Qwen3-VL-Embedding dense + ColQwen multi-vector, `src/retriever.py`), DeepSeek judge (`src/judge.py`), prompts/schemas, index-building & eval entrypoints (`scripts/`), eval configs (`config/`), RL reward fns (`rewards/`).
- `eval/` — benchmark run scripts (serve vLLM → predict → DeepSeek judge → per-category scoring) for full SlideVQA-2215, ViDoSeek-1142, MMLongBench-847.

## Notes
- Secrets scrubbed: set `DEEPSEEK_API_KEY` (judge) and `HF_TOKEN` (slurm index build) via env.
- Best checkpoints (on training box, not in repo): 7B v9 step_75 = SlideVQA 75.76 / ViDoSeek 71.54 / MMLB-847 35.30; 3B sj step_75 = 69.16 / 65.15 / 29.52.
