# visual_rag_agent_new — editable page memory + active revisit

The current method ("new"). Design: `DESIGN.md` (§1–§10 method, §11 implementation decisions).
`../visual_rag_agent_agv3/` is frozen as the replay baseline required by §10.

## Turn format

```
<think>...</think>
<act>{"memory_edit": {"page_desc": null | "...",
                      "page_ops": [{"op": "add_fact", "text", "bbox"} | {"op": "update_fact", "fact_id", "text", "bbox"}],
                      "task_ops": [{"op": "expand", "answered", "answer", "pages", "next"} |
                                   {"op": "update", "node_id", "question", "status", "answer", "pages"}]},
      "action": {"type": "search", "query"} | {"type": "revisit", "page_id"} |
                {"type": "crop", "bbox" | "fact_id"} | {"type": "answer", "text", "support_nodes"}}</act>
```

Memory = question tree (N0 + ordered sub-questions, active = first unresolved) + page nodes
(P1, P2, ... each owning editable facts P1.F1, ...) + question→page references. Edits are atomic;
correcting a fact reopens the resolved questions that relied on its page. One crop per visit;
any seen page can be revisited. `answer` ends the episode; E_final = pages of the support nodes.

## Layout

- `src/protocol.py` — schema, parser, structural rules, coordinate frames (`qwen2.5-vl` px /
  `qwen3` 0-1000) and SCORE crop geometry (shared by eval and RL; DESIGN.md §11.3).
- `src/memory.py` — `Memory` / `PageNode` / `QNode` / `View`, `apply_edit`, rendering.
- `src/prompts.py` — `POLICY_SYSTEM` and `build_turn_messages` (shared by eval and RL).
- `src/agent.py` — eval loop (cursor search, revisit, crop, soft/hard error lanes, forced
  answer turn, trace + per-turn SFT records rendered in the student's `sft_frame`).
- `src/vlm.py`, `src/retriever.py`, `src/judge.py`, `src/config.py` — infrastructure.
- `rl/envs.py` — verl-agent environment (set `VISUAL_RAG_AGENT_PATH` to this checkout).
- `scripts/prepare_slidevqa.py`, `scripts/build_colqwen_index.py`, `scripts/retrieval_recall.py`,
  `scripts/evaluate.py`, `scripts/generate_sft_trajectories.py`, `scripts/serve_teacher.sh`,
  `scripts/run_generation.sh`, `scripts/run_eval.sh` (resumable, flat corpus, DeepSeek judge);
  `config/default.yaml` (student), `config/teacher.yaml`, overlays `config/blackwell.yaml`, `config/turns10.yaml`.
- `tests/` — `python tests/run_tests.py` (no pytest needed).

## RL knobs

`COVERAGE_W=0.5 FULL_COVERAGE_W=0 CMP_W=0.2 GROUNDING_W=0.5 ACCESS_COST_W=0.1 AG_SOFT_ERROR_REWARD=-1
ACTIVE_GRAPH_RETRIEVE_K=5 SLIDEVQA_STEP_WORKERS=32`; env config `slidevqa.frame` (default `qwen2.5-vl`),
`slidevqa.retrieval_scope` (`flat` | `deck`). Dataset rows may carry
`keyfacts: [{"page": 3, "bbox_2d": [x1,y1,x2,y2]}]` (original page pixels).

The next experiment uses `rl/experiments/hybrid_coverage_gspo.args` (DESIGN.md §11.36):
`answer + 0.2*coverage_fraction + 0.3*full_coverage + 0.5*grounding`, with CMP/access-cost
weights zero and GSPO clip low/high `0.0003/0.0004`. Coverage uses the final cited pages;
full coverage requires nonempty gold pages. The grounding calculation is unchanged.
Explicit `env.slidevqa.coverage_mode`, `coverage_w`, `full_coverage_w`, `grounding_w`,
`cmp_w`, and `access_cost_w` override the corresponding uppercase environment variables.

This is an override file, not a complete launcher. After deploying the updated `rl/envs.py`
to the environment module actually imported by verl-agent, load the file on the Linux host:

```bash
mapfile -t experiment_overrides < rl/experiments/hybrid_coverage_gspo.args
```

Append `"${experiment_overrides[@]}"` as the **last** arguments to the verified training
entry point. Use a new experiment/output directory, verify current GPU allocation and model
paths, and retain cosine scheduling with 20% warmup. The file does not change the advantage
estimator or enable GiGPO/pruning; verify the base configuration matches the intended estimator.
Historical launch snapshots are not updated by this preset.

## Runpod preparation and GPU changes

The preparation Pod uses a 500 GB **regional Network Volume** at `/workspace` in AP-IN-1.
Keep code, managed Python, environments, model files and data under this mount. A replacement
Pod in the same data center can attach the volume; use a compatible CUDA 12.8 development
image and retain the `/workspace` path. Other data centers require a separate copy of the files.

```bash
source /workspace/code/visual_rag_agent_new/scripts/activate_runpod.sh
python /workspace/code/visual_rag_agent_new/scripts/runpod_preflight.py
```

`prepare_runpod_env.sh` installs the dependencies into `/workspace/envs/rl`, records resolved
versions, checks a FlashAttention operation against PyTorch SDPA and runs the agent tests.
It does not install model weights or launch training. Source data metadata retains its original
`/root/rivermind-data` paths; the activation script creates a compatibility symlink to `/workspace`.

The source's selected assets are recorded in `/workspace/preparation/source_assets.json`.
After all transfers and training-index assembly finish, verify the files with:

```bash
python /workspace/code/visual_rag_agent_new/scripts/runpod_asset_manifest.py \
  --root /workspace --verify /workspace/preparation/source_assets.json \
  --output /workspace/preparation/assets_verified.json
```

After provisioning the intended **one-node, four-GPU** Pod, check communication with:

```bash
torchrun --standalone --nproc_per_node=4 \
  /workspace/code/visual_rag_agent_new/scripts/runpod_nccl_smoke.py
```

Four-GPU communication and a complete training step still need validation on that Pod.
Resolve the starting model, experiment/output directory, GPU count, retriever placement,
FSDP and rollout settings before formal training. The 3B starting policy is the merged SFT v5
checkpoint at `/workspace/models/student_qwen25vl3b_lora_v5`, matching the checkpoint used
to initialize the completed H800 RL run. Check `/workspace/preparation/sft3b_files_verified.json`
and `sft3b_runtime_validation.json` for original-file hashes and the GPU loading/image-forward
results. The 7B starting checkpoint remains deferred by the user.
