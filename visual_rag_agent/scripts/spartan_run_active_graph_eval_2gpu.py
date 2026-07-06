#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import threading
from pathlib import Path

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


def main() -> None:
    # cursor retrieval K=5 to match RL training (agent.py defaults to 1 otherwise).
    # set before shards are spawned so base_env=os.environ.copy() propagates it.
    os.environ.setdefault("ACTIVE_GRAPH_RETRIEVE_K", "5")
    parser = argparse.ArgumentParser(description='Run active-graph SlideVQA eval in 2 GPU shards on Spartan.')
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--num-samples', type=int, default=500)
    parser.add_argument('--shards', type=int, default=2)
    parser.add_argument('--judge', choices=['none', 'deepseek'], default=os.environ.get('ACTIVE_GRAPH_EVAL_JUDGE', 'deepseek'))
    parser.add_argument('--judge-model', default=os.environ.get('ACTIVE_GRAPH_EVAL_JUDGE_MODEL', 'deepseek-v4-flash'))
    parser.add_argument('--judge-max-tokens', default=os.environ.get('DEEPSEEK_JUDGE_MAX_TOKENS', '1024'))
    parser.add_argument('--project', default=os.environ.get('PROJECT', '/scratch/punim0614/lifuzhang/visual_rag_agent'))
    parser.add_argument('--dataset-file', default=os.environ.get('ACTIVE_GRAPH_EVAL_DATASET', '/scratch/punim0614/lifuzhang/visual_rag_agent/data/corpora/slidevqa/test.jsonl'))
    parser.add_argument('--index-path', default=os.environ.get('ACTIVE_GRAPH_EVAL_INDEX', '/scratch/punim0614/lifuzhang/visual_rag_agent/data/indexes/slidevqa_test_main'))
    parser.add_argument('--retriever-path', default=os.environ.get('ACTIVE_GRAPH_RETRIEVER', '/scratch/punim0614/lifuzhang/models/Qwen3-VL-Embedding-8B'))
    parser.add_argument('--output-root', default=None)
    args = parser.parse_args()

    project = Path(args.project)
    model_path = Path(args.model_path)
    dataset_file = Path(args.dataset_file)
    index_path = Path(args.index_path)
    output_root = Path(args.output_root) if args.output_root else project / 'outputs' / args.run_id
    config_path = project / 'config' / f'{args.run_id}.yaml'

    required = [project / 'scripts/evaluate.py', project / 'scripts/merge_eval_shards.py', model_path / 'config.json', dataset_file, index_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError('Missing required paths:\n' + '\n'.join(missing))

    output_root.mkdir(parents=True, exist_ok=True)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    config = {
        'models': {
            'vlm': {
                'provider': 'qwen',
                'name': str(model_path),
                'max_tokens': 1024,
                'temperature': 0.0,
                'prompt_mode': 'system_in_user',
            },
            'retriever': {
                'name': str(args.retriever_path),
                'index_path': str(index_path),
            },
        },
        'agent': {
            'top_k': 1,
            'max_iters': int(os.environ.get('ACTIVE_GRAPH_EVAL_MAX_ITERS', '15')),
            'partial_memory_capacity': 2,
        },
        'image_budget': {'yes_pixels': 400000, 'partial_pixels': 400000},
        'dataset': {'name': 'slidevqa', 'split': 'test', 'num_samples': args.num_samples},
        'runtime': {
            'output_dir': str(output_root),
            'device': 'cuda',
            'dtype': 'bfloat16',
            'attn_implementation': 'flash_attention_2',
        },
    }
    if yaml is not None:
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    else:
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')

    shard_sizes = [args.num_samples // args.shards] * args.shards
    for i in range(args.num_samples % args.shards):
        shard_sizes[i] += 1
    shard_starts = []
    cursor = 0
    for size in shard_sizes:
        shard_starts.append(cursor)
        cursor += size

    python = os.environ.get('PYTHON', sys.executable)
    base_env = os.environ.copy()
    base_env['PYTHONPATH'] = str(project) + os.pathsep + base_env.get('PYTHONPATH', '')
    base_env['TOKENIZERS_PARALLELISM'] = 'false'
    base_env['PYTHONUNBUFFERED'] = '1'

    procs: list[tuple[int, subprocess.Popen[str]]] = []
    shard_roots: list[Path] = []
    for shard_id, (start, size) in enumerate(zip(shard_starts, shard_sizes)):
        shard_root = output_root / f'shard{shard_id}'
        shard_root.mkdir(parents=True, exist_ok=True)
        shard_roots.append(shard_root)
        env = base_env.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(shard_id % max(1, int(os.environ.get("EVAL_NUM_GPUS", args.shards))))
        cmd = [
            python, 'scripts/evaluate.py',
            '--dataset-file', str(dataset_file),
            '--config', str(config_path),
            '--output', str(shard_root),
            '--baseline', 'agent',
            '--start-index', str(start),
            '--num-samples', str(size),
            '--judge', args.judge,
            '--judge-model', args.judge_model,
            '--judge-max-tokens', str(args.judge_max_tokens),
        ]
        print(f'[launch shard{shard_id}] cuda={env["CUDA_VISIBLE_DEVICES"]} start={start} size={size}', flush=True)
        print(' '.join(shlex.quote(x) for x in cmd), flush=True)
        procs.append((shard_id, subprocess.Popen(cmd, cwd=str(project), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)))

    def stream(prefix: str, proc: subprocess.Popen[str]) -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(f'[{prefix}] {line}', end='', flush=True)

    threads = []
    for shard_id, proc in procs:
        t = threading.Thread(target=stream, args=(f'shard{shard_id}', proc), daemon=True)
        t.start()
        threads.append(t)

    failed = []
    for shard_id, proc in procs:
        rc = proc.wait()
        if rc != 0:
            failed.append((shard_id, rc))
    for t in threads:
        t.join(timeout=2)
    if failed:
        raise RuntimeError(f'eval shard failures: {failed}')

    merge_cmd = [python, 'scripts/merge_eval_shards.py', '--roots', *[str(root) for root in shard_roots], '--output', str(output_root / 'merged')]
    print(' '.join(shlex.quote(x) for x in merge_cmd), flush=True)
    subprocess.run(merge_cmd, cwd=str(project), env=base_env, check=True)
    print('[done]', output_root, flush=True)


if __name__ == '__main__':
    main()
