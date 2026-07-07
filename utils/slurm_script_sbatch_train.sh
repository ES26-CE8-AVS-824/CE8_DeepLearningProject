#!/bin/bash

#SBATCH --job-name=whispers
#SBATCH --output=logs/mini-whisper-train_%j.out
#SBATCH --error=logs/mini-whisper-train_%j.err
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00

singularity exec --nv --cleanenv \
    -B "$HOME/.singularity:/scratch/singularity" \
    --env WANDB_API_KEY="[wandb_api_key]" \
    --env WANDB_ENTITY="mini-whisper" \
    --env WANDB_PROJECT="mini-whisper" \
    "/ceph/project/es26-ce8-avs-824/whispers-in-the-storm/sgmse_env_cu130_v6.sif" \
    /bin/bash -c "
        set -euo pipefail
        export TMPDIR=/scratch/singularity/tmp && \
        \
        export NUMBA_CACHE_DIR=/scratch/singularity/tmp/numba_cache && \
        export MPLCONFIGDIR=/scratch/singularity/tmp/matplotlib_cache && \
        export TORCH_EXTENSIONS_DIR=/scratch/singularity/tmp/torch_extensions && \
        export TORCH_KERNEL_CACHE=/scratch/singularity/tmp/torch_kernels && \
        mkdir -p \$NUMBA_CACHE_DIR \$MPLCONFIGDIR \$TORCH_EXTENSIONS_DIR \$TORCH_KERNEL_CACHE && \
        \
        export TRITON_LIBCUDA_PATH=/.singularity.d/libs && \
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
        torchrun --nproc_per_node=4 train.py
    "
