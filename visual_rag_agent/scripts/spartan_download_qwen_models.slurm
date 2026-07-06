#!/usr/bin/env bash
#SBATCH --job-name=qwen3vl-model-download
#SBATCH --partition=sapphire
#SBATCH --account=punim0614
#SBATCH --qos=publiccpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd /scratch/punim0614/lifuzhang/visual_rag_agent
mkdir -p logs
bash scripts/spartan_download_qwen_models.sh
