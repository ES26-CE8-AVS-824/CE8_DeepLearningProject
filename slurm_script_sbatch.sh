#!/bin/bash

#SBATCH --job-name=mini-whisper
#SBATCH --output=logs/mini-whisper_%j.out
#SBATCH --error=logs/mini-whisper_%j.err
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00

singularity exec --nv \
    -B "$HOME/.singularity:/scratch/singularity" \
    --env WANDB_API_KEY="[wandb-api-key]" \
    --env WANDB_ENTITY="mini-whisper" \
    --env WANDB_PROJECT="mini-whisper" \
    "/ceph/project/es26-ce8-avs-824/whispers-in-the-storm/sgmse_env_cu130_v1.sif" \
    /bin/bash -c "
        set -euo pipefail
        export TMPDIR=/scratch/singularity/tmp && \
        export TRITON_LIBCUDA_PATH=/.singularity.d/libs && \
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
        python train.py
    "