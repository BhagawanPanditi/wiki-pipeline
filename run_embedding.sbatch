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

set -Eeuo pipefail

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export PYTHONUNBUFFERED=1

source ~/ml-training-env/bin/activate
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen-embed

SCRIPT_DIR="/home/compiling-ganesh/24m0829/long_document/wiki_pipeline"
WIKI_BASE_DIR="/home/compiling-ganesh/24m0829/long_document/wiki"
DOMAIN="biology"
STATE_DIR="${WIKI_BASE_DIR}/${DOMAIN}/.pipeline_state"
mkdir -p "${STATE_DIR}"

# Any embedding attempt may update the graph; force downstream stages to
# validate/rebuild rather than trusting markers from an earlier graph version.
for step in stage2-noise-audit stage2-bidirectional-relations stage2-markdown-export stage2-final-analysis \
    embedding-bridging noise-audit bidirectional-relations final-consolidation final-analysis; do
    rm -f -- "${STATE_DIR}/${step}.done" "${STATE_DIR}/${step}.failed"
done

echo "========================================================"
echo " Running Qwen Concept Embedding & Semantic Island Bridge"
echo "========================================================"

MAX_ATTEMPTS="${EMBED_MAX_ATTEMPTS:-3}"
if ! [[ "${MAX_ATTEMPTS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "EMBED_MAX_ATTEMPTS must be a positive integer." >&2
    exit 2
fi

attempt=1
delay=30
while true; do
    echo "Embedding attempt ${attempt}/${MAX_ATTEMPTS}..."
    if python3 "${SCRIPT_DIR}/embed_and_bridge_islands.py" \
        --domain "${DOMAIN}" \
        --wiki_base_dir "${WIKI_BASE_DIR}" \
        --sim_threshold 0.78; then
        break
    else
        rc=$?
    fi

    if ((attempt >= MAX_ATTEMPTS)); then
        echo "Embedding failed after ${attempt} attempts (exit ${rc}). Resubmit the job after reviewing run_embed.log." >&2
        exit "$rc"
    fi

    echo "Embedding attempt failed (exit ${rc}); retrying in ${delay}s."
    sleep "$delay"
    attempt=$((attempt + 1))
    delay=$((delay * 2))
    if ((delay > 300)); then delay=300; fi
done

echo "Embedding bridging complete!"
