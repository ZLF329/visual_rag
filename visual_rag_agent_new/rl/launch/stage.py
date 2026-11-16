"""Deploy the candidate to its own paths and validate data against the live image indices."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys

import pyarrow.parquet as pq
import yaml
from PIL import Image

prep = Path(__file__).resolve().parents[1]
workspace = Path('/workspace')
local = Path('/opt') / prep.name
code = local / 'verl-agent'
slide_data = workspace / 'data/slidevqa'
cfg = yaml.safe_load((prep / 'candidate_config.yaml').read_text())
data = Path(cfg['data']['train_files']).parent

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

manifest = json.loads((prep / 'runtime_manifest.json').read_text())
for name, expected in manifest.items():
    assert sha(prep / name) == expected, name
selection = json.loads((prep / 'data/selection_summary.json').read_text())
assert sha(prep / 'data/selected_1200.jsonl') == selection['selected_sha256']
assert sha(prep / 'data/train.parquet') == json.loads((prep / 'data/parquet_verified.json').read_text())['sha256']

data.mkdir(parents=True, exist_ok=True)
for src in (prep / 'data').iterdir():
    target = data / src.name
    if not target.exists():
        shutil.copyfile(src, target)
    assert sha(src) == sha(target)

shutil.copytree(workspace / 'code/verl-agent', code, dirs_exist_ok=True,
                ignore=shutil.ignore_patterns('.git', '__pycache__', '*.pyc', 'outputs', 'wandb', '.env', '.secrets.env'))
for src in (prep / 'runtime/verl_overlay').rglob('*.py'):
    relative = src.relative_to(prep / 'runtime/verl_overlay')
    dst = code / relative
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    assert sha(src) == sha(dst)
assert sha(workspace / 'code/verl-agent/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py') == \
    'a4fad0cfc37873d5a2632dd6af22a95bc68cb0891c01729f8d7409e2d17bc69b'

def read_jsonl(path):
    return [json.loads(line) for line in path.open() if line.strip()]

indices = {}
for corpus in ('train', 'val'):
    src = slide_data / 'index' / f'{corpus}_colqwen'
    dst = local / 'indices' / corpus
    dst.mkdir(parents=True, exist_ok=True)
    for name in ('meta.json', 'mv_embeddings.pt'):
        link = dst / name
        if not link.is_symlink():
            link.symlink_to(src / name)
        assert link.resolve() == (src / name).resolve() and link.is_file()
    entries = json.loads((src / 'filenames.json').read_text())
    for entry in entries:
        path = entry['image_path']
        for prefix in ('/root/rivermind-data/', '/opt/visual-rag-local/', '/workspace/'):
            if path.startswith(prefix):
                path = '/workspace/' + path.removeprefix(prefix)
                break
        assert path.startswith(f'/workspace/data/slidevqa/pages/{corpus}/'), path
        entry['image_path'] = path
    (dst / 'filenames.json').write_text(json.dumps(entries, ensure_ascii=False))
    indices[corpus] = {str(Path(entry['image_path']).relative_to(slide_data / 'pages' / corpus).with_suffix('')):
                       Path(entry['image_path']) for entry in entries}
    assert len(indices[corpus]) == len(entries)
    cfg['env']['slidevqa']['index_paths'][corpus] = str(dst)
cfg['env']['slidevqa']['index_path'] = cfg['env']['slidevqa']['index_paths']['train']
cfg['data']['val_files'] = str(slide_data / 'rl_set_7b_1600_multi500_kf/val.parquet')
assert Path(cfg['data']['val_files']).is_file()
assert Path(cfg['actor_rollout_ref']['model']['path'], 'model.safetensors.index.json').is_file()
assert Path(cfg['env']['slidevqa']['retriever_path'], 'adapter_config.json').is_file()
rows = read_jsonl(data / 'selected_1200.jsonl')
assert Counter(row['hop'] for row in rows) == {'single': 800, 'multi': 400}
test_decks = {row['deck_name'] for row in read_jsonl(slide_data / 'test.jsonl')}
assert not (test_decks & {row['deck_name'] for row in rows})
image_sizes = {}
boxes = 0
for row in rows:
    for fact in row['keyfacts']:
        key = f"{row['deck_name']}/page_{fact['page']:02d}"
        path = indices[row['corpus']][key]
        if path not in image_sizes:
            with Image.open(path) as image:
                image_sizes[path] = image.size
        width, height = image_sizes[path]
        x1, y1, x2, y2 = fact['bbox_2d']
        assert 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height, (row['sample_id'], key, fact)
        boxes += 1

table = pq.read_table(data / 'train.parquet')
assert table.num_rows == 1200
assert [row['sample_id'] for row in table['env_kwargs'].to_pylist()] == [row['sample_id'] for row in rows]
# Multi-hop prompts create a real mix of ordinary and G endings under the shortened smoke budget.
smoke_indices = [i for i, row in enumerate(rows) if row['hop'] == 'multi'][:16]
pq.write_table(table.take(smoke_indices), data / 'smoke.parquet')
validation = pq.read_table(cfg['data']['val_files'])['env_kwargs'].to_pylist()
overlap = len({r['sample_id'] for r in validation} & {r['sample_id'] for r in rows})

(prep / 'deployed_config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
report = dict(questions=len(rows), gold_images=len(image_sizes), boxes=boxes, test_deck_overlap=0,
              validation_question_overlap=overlap, overlay_files=len(manifest), train_sha256=sha(data / 'train.parquet'))
(prep / 'deployment_verified.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report), flush=True)
