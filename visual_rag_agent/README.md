# Visual RAG Agent with Task-State Memory

This is an inference-only research prototype for SlideVQA-style visual document QA.

The agent uses exactly three tools:

- `search`: retrieve document page images.
- `analyse_summarise`: automatically analyse every retrieved page.
- `answer`: produce the final answer from structured memory.

The loop has one decision point per iteration: search again or answer. Memory is structured as confirmed findings, partial findings capped to the two most recent entries, and typed warnings for failed pages.

## Layout

```text
visual_rag_agent/
├── config/default.yaml
├── src/
├── scripts/
├── tests/
├── data/
└── outputs/
```

## CPU-safe checks

These tests do not load the retriever or VLM:

```bash
cd /root/autodl-tmp/visual_rag_agent
python -m pytest tests
```

## Build an index on a GPU machine

```bash
python scripts/build_index.py \
  --corpus data/corpora/slidevqa \
  --output data/indexes/slidevqa \
  --model Alibaba-NLP/GVE-7B
```

The index format is intentionally simple:

- `embeddings.npy`
- `filenames.json`

## Run one query

```bash
python scripts/run_inference.py \
  --query "How did Apple's services revenue change from Q2 to Q3 2024?" \
  --index data/indexes/slidevqa \
  --config config/default.yaml
```

## Evaluate first 200 examples

```bash
python scripts/evaluate.py \
  --dataset-file data/corpora/slidevqa/test.jsonl \
  --num-samples 200 \
  --config config/default.yaml \
  --output outputs/runs
```

Use `--baseline agentic_summary` for the baseline that decides `search` vs
`answer` from query memory plus the most recent two rounds of images, then
summarizes every searched page into memory.
# visual_rag
