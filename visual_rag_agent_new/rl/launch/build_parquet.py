"""Local-only conversion using the prior formal parquet schema, then exact round-trip checks."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
source = DATA / 'selected_1200.jsonl'
selection = json.loads((DATA / 'selection_summary.json').read_text())
assert hashlib.sha256(source.read_bytes()).hexdigest() == selection['selected_sha256']
raw = [json.loads(s) for s in source.read_text().splitlines()]
assert len(raw) == len({q['sample_id'] for q in raw}) == 1200
assert Counter(q['hop'] for q in raw) == {'single': 800, 'multi': 400}
assert all(1 <= q['correct_count'] <= 4 for q in raw)
assert all(q['keyfacts'] and all(len(k['bbox_2d']) == 4 for k in q['keyfacts']) for q in raw)
assert all({int(k['page']) for k in q['keyfacts']} <= set(q['evidence_pages']) for q in raw)
import os
schema_source = Path(os.environ.get('SCHEMA_SOURCE') or HERE.parent / 'final1200_quality_multi400_20261110/train.parquet')
schema = pq.read_schema(schema_source)
records = []
for i, q in enumerate(raw):
    pages = sorted(set(q['evidence_pages']))
    labels = [f"{q['deck_name']}/page_{n:02d}" for n in pages]
    extra = dict(index=i, split='train', sample_id=q['sample_id'], qa_id=q['qa_id'],
                 deck_name=q['deck_name'], question=q['question'], answer=str(q['answer']),
                 hop_type=q['hop'], evidence_count=len(pages), evidence_pages=pages,
                 reference_pages=pages, page_labels=labels, corpus=q['corpus'],
                 crop_label=None, crop_confidence=None, k_correct=q['correct_count'], teacher_v4=None)
    kwargs = dict(question=q['question'], answer=str(q['answer']), ground_truth=str(q['answer']),
                  deck_name=q['deck_name'], evidence_pages=pages, reference_pages=labels,
                  page_labels=labels, data_source='slidevqa_new_train', sample_id=q['sample_id'],
                  corpus=q['corpus'], extra_info=extra, keyfacts=json.dumps(q['keyfacts']))
    records.append(dict(data_source='slidevqa_new_train',
                        prompt=[dict(role='user', content='Question: ' + q['question'])],
                        ability='visual_document_qa', reward_model=dict(style='rule', ground_truth=str(q['answer'])),
                        extra_info=extra, env_kwargs=kwargs))
path = DATA / 'train.parquet'
pq.write_table(pa.Table.from_pylist(records, schema=schema), path)
saved = pq.read_table(path).to_pylist()
assert saved == records, 'Parquet round trip changed content'
assert [q['env_kwargs']['sample_id'] for q in saved] == [q['sample_id'] for q in raw]
report = dict(rows=1200, single=800, multi=400, batches_of_16=75, duplicates=0,
              same_schema_as_previous_formal=True, all_rows_round_trip_equal=True,
              jsonl_order_preserved=True, per_step_composition_quotas=False,
              sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
              tokenizer_dataset_preflight_run=False, gpu_smoke_test_run=False,
              training_started=False)
(DATA / 'parquet_verified.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
