#!/bin/bash

#SBATCH --output=/lustre/fswork/projects/rech/unv/usi32yh/myria3d/logs/slurm/%j.out
#SBATCH --error=/lustre/fswork/projects/rech/unv/usi32yh/myria3d/logs/slurm/%j.err
#SBATCH -A uhn@h100
#SBATCH -C h100

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread

#SBATCH --job-name=myria3d-200k
#SBATCH --time=19:45:00
# 200k iterations is ~20x the reference 10k run this --time budget comes from
# (scripts/jz/launch_h100.sh) -- it will very likely need several resubmissions.
# Adjust --time to your actual throughput/QOS. To resume past the time limit,
# resubmit the same command with `model.ckpt_path=<run_dir>/checkpoints/last.ckpt`
# appended below (save_last: true is on by default, see configs/callbacks/multitask.yaml,
# and this goes through the real Lightning resume path -- optimizer/scheduler/epoch state
# included -- not `task.task_name=finetune`, which restarts the schedule from epoch 0).

# Usage:
#   sbatch run_flair3d_plus_multitask_200k.slurm            # base myria3d lr (1e-3)
#   sbatch run_flair3d_plus_multitask_200k.slurm 2e-3        # lr override
# Tip: `sbatch --job-name=myria3d-200k-lr2e-3 run_flair3d_plus_multitask_200k.slurm 2e-3`
# to tell the two runs apart in squeue (the #SBATCH job-name above can't see $1).
LR="${1:-1e-3}"

JOB_DIR=/lustre/fswork/projects/rech/unv/usi32yh/myria3d/logs/slurm/${SLURM_JOB_ID}
mkdir -p ${JOB_DIR}

# --output/--error above must stay flat in logs/slurm/ (that directory is
# guaranteed to exist at submission time; a %j subdirectory might not be,
# and SLURM will silently fail to create the file if its parent dir is
# missing). Symlink them into JOB_DIR instead so everything for this job
# is browsable from one place.
ln -sf ../${SLURM_JOB_ID}.out ${JOB_DIR}/${SLURM_JOB_ID}.out
ln -sf ../${SLURM_JOB_ID}.err ${JOB_DIR}/${SLURM_JOB_ID}.err

cp $0 ${JOB_DIR}/script.slurm

{
    echo "Job ID: $SLURM_JOB_ID"
    echo "Starting job at: $(date)"
    echo "Running on host: $(hostname)"
    echo "Working directory: $(pwd)"
    echo "Python executable: $(which python)"
    echo "lr: ${LR}"
    nvidia-smi  # État initial du GPU
} > ${JOB_DIR}/job_info.log

module purge
module load arch/h100
module load cuda/12.1.0
module load miniforge/24.9.0

conda activate myria3d


set -euo pipefail

conda list > ${JOB_DIR}/conda_env.txt

export LOGS_DIR=/lustre/fswork/projects/rech/unv/usi32yh/myria3d/logs
export WANDB_MODE=offline
export FLAIR3D_DATA_ROOT=/lustre/fsn1/projects/rech/unv/usi32yh/data_pointcept/flair3d_plus
export FLAIR3D_CSV_MANIFEST=${FLAIR3D_DATA_ROOT}/raw/scene_split_manifest.csv
cd /lustre/fswork/projects/rech/unv/usi32yh/myria3d

START_TIME=$(date +%s)


### --- Myria3D syntax ---
python run.py experiment=flair3d_plus/multitask_200k \
  model.lr=${LR}

### ------------------------

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

{
    echo "Job finished at: $(date)"
    echo "Duration: ${DURATION} seconds"
    nvidia-smi  # État final du GPU
} >> ${JOB_DIR}/job_info.log
