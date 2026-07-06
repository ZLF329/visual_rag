#!/usr/bin/env bash
#SBATCH --job-name=verl-envbuild
#SBATCH --partition=sapphire
#SBATCH --account=punim0614
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/punim0614/lifuzhang/slurm_logs/verl-envbuild-%j.out
#SBATCH --error=/scratch/punim0614/lifuzhang/slurm_logs/verl-envbuild-%j.err

set -xeo pipefail
export LANG=C.UTF-8; export LC_ALL=C.UTF-8
echo "HOST=$(hostname)  START=$(date)"
# quick internet check
curl -sI --max-time 20 https://pypi.org/simple/ | head -1 || echo "WARN: pypi check failed"
bash /scratch/punim0614/lifuzhang/visual_rag_agent/scripts/build_verl_vllm011.sh
echo "SLURM_BUILD_DONE=$(date)"
