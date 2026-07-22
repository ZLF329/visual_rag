# visual_rag_agent_agv2 — AGv2 refactored implementation (2026-07)

Post-refactor code for the Active-Graph visual-RAG agent. The pre-refactor snapshot lives in
`../visual_rag_agent/`; this directory supersedes it.

## AGv2 protocol (merged-action)

Every turn = `<think>` + [`<update_graph>{...}` iff an observation is pending] + exactly one
action (`<search>` / `<bbox>[x1,y1,x2,y2]` / `<answer>`). Key changes vs the old per-turn
single-action protocol:

- `update_graph` is inlined before the action instead of occupying its own turn; the final
  turn is MERGED (final commit + answer together, single- and multi-hop alike).
- UniDoc-style `<bbox>` zoom added: legal right after an accept/expand page commit; crops the
  committed page in displayed-pixel coordinates (±28px pad + smart_resize); zoom commits
  append facts to the node that received the page's facts and refresh its answer; zoom may be
  retried after a zoom-reject. The active node stays pinned on the crop target during a chain.
- Supporting facts are plain strings (no bbox payloads in the graph).
- Structural violations are a hard format-error lane (episode-terminating on both sides);
  malformed bbox rectangles are a soft per-step error.
- Teacher-side coordinate-frame adapter: Qwen3-VL-family teachers ground in 0-1000 normalized
  coordinates regardless of prompt instructions; `Agent(bbox_frame="norm1000")` converts each
  box to displayed pixels and rewrites the recorded response, so persisted SFT targets stay in
  the canonical pixel frame (see DESIGN_AGV2.md, D14).

Design decisions D1-D14 are recorded in `DESIGN_AGV2.md`.

## Layout

- `src/` — eval-side implementation. `src/protocol.py` is the single shared parser +
  crop/commit helper module; `src/agent.py` is the AGv2 eval loop.
- `rl/envs.py` — verl-agent RL environment (imports the same `src/protocol.py`; drop into
  `verl-agent/agent_system/environments/env_package/slidevqa/`).
- `scripts/generate_active_graph_sft_trajectories.py` — teacher-distillation SFT generator
  (reads per-turn `sft.messages/target` straight from traces; judge + reference-page +
  clean-termination keep filters).
- `scripts/serve_teacher.sh`, `scripts/run_generation.sh` — teacher-box launchers
  (Qwen3-VL-235B-A22B-Thinking-FP8 via vLLM, TP=4). Set your own `DEEPSEEK_API_KEY`.
- `config/teacher_qwen3vl235b_sft.yaml` — teacher generation config (`agent.bbox_frame:
  norm1000`).
- `test_agv2.py` (69 checks), `test_agv2_env.py` (43), `test_gen_extract.py` (16) — mechanical
  suites: parser violation matrix, graph transitions, crop pin/resume + retry, coordinate
  mapping, scripted fake-VLM episodes, reward channels, SFT extraction.

Run tests: `VRA_PATH=<this dir> python test_agv2.py` (env suite needs the verl-agent tree on
PYTHONPATH).
