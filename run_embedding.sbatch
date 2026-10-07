#!/bin/bash
#SBATCH -J embed_bio
#SBATCH -p a40
#SBATCH --qos=a40
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=run_embed.log
#SBATCH --error=run_embed.log

set -eo pipefail

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export PYTHONUNBUFFERED=1

source ~/ml-training-env/bin/activate
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen-embed

SCRIPT_DIR="/home/compiling-ganesh/24m0829/long_document/wiki_pipeline"
DOMAIN="biology"

echo "========================================================"
echo " Running Qwen Concept Embedding & Semantic Island Bridge"
echo "========================================================"

python3 "${SCRIPT_DIR}/embed_and_bridge_islands.py" \
    --domain "${DOMAIN}" \
    --wiki_base_dir "/home/compiling-ganesh/24m0829/long_document/wiki" \
    --sim_threshold 0.78

echo "Embedding bridging complete!"
