#!/usr/bin/env bash
#SBATCH --job-name=ag25-evalidx
#SBATCH --partition=gpu-a100-short
#SBATCH --account=punim0614
#SBATCH --qos=publicgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=03:30:00
#SBATCH --output=/scratch/punim0614/lifuzhang/slurm_logs/%x-%j.out
#SBATCH --error=/scratch/punim0614/lifuzhang/slurm_logs/%x-%j.err
set -euo pipefail
module load Miniconda3/23.10.0-1
module load CUDA/12.4.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/punim0614/.cache/conda/envs/verl-gspo
cd /scratch/punim0614/lifuzhang/visual_rag_agent
export PYTHONPATH=/scratch/punim0614/lifuzhang/visual_rag_agent:${PYTHONPATH:-}
PY=/scratch/punim0614/.cache/conda/envs/verl-gspo/bin/python
if [ -s data/corpora/slidevqa/test.jsonl ] && [ "$(find data/corpora/slidevqa/pages -name '*.png' 2>/dev/null | wc -l)" -gt 100 ]; then echo "test.jsonl+corpus already present, skipping prepare"; else
echo "=== 1) prepare official SlideVQA test (500) -> test.jsonl + pages ==="
"$PY" scripts/prepare_official_slidevqa.py --split test --num-samples 500 \
  --output data/corpora/slidevqa \
  --hf-endpoint https://huggingface.co \
  --token ${HF_TOKEN}
echo "test.jsonl lines: $(wc -l < data/corpora/slidevqa/test.jsonl) | pages: $(find data/corpora/slidevqa/pages -name '*.png'|wc -l)"
fi
echo "=== 2) build slidevqa_test_main index ==="
"$PY" scripts/build_index.py --corpus data/corpora/slidevqa/pages \
  --output data/indexes/slidevqa_test_main \
  --model /scratch/punim0614/lifuzhang/models/Qwen3-VL-Embedding-8B \
  --device cuda --dtype bfloat16 --batch-size 8
echo "EVAL INDEX BUILD DONE: $(ls -la data/indexes/slidevqa_test_main/embeddings.npy)"
