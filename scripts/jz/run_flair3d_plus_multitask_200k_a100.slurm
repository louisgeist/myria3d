#!/bin/bash

#SBATCH --output=/lustre/fswork/projects/rech/unv/usi32yh/myria3d/logs/slurm/%j.out
#SBATCH --error=/lustre/fswork/projects/rech/unv/usi32yh/myria3d/logs/slurm/%j.err
#SBATCH -A uhn@a100
#SBATCH -C a100
# ^ TODO verify against `idrct_info`/your allocation notice before sbatch'ing:
#   - account/hours pot: uhn@a100 (separate quota from the exhausted ppm@h100)
#   - `-C a100` constraint name, and whether this partition also needs an
#     explicit --qos= for a >20h job or a different --cpus-per-task ratio
#     than the H100 nodes below (kept at 24 from the H100 script, unverified
#     for A100).

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread

#SBATCH --job-name=myria3d-200k-a100
#SBATCH --time=19:45:00
#SBATCH --signal=USR1@120
#SBATCH --requeue
# Same auto-requeue mechanism as run_flair3d_plus_multitask_200k.slurm (see
# that file for the full explanation) -- kept here in case this leg also
# needs several requeue hops to reach 200k iters.

# Resuming the run that was killed on ppm@h100 (job 2175326) for lack of
# project hours. This is a real Lightning fit-resume via model.ckpt_path
# (optimizer/scheduler/epoch/global_step/GradNorm-lite EMA all restored) --
# NOT task.task_name=finetune, which would restart the schedule at epoch 0.
#
# hpc_ckpt_1.ckpt was picked over checkpoints/last.ckpt because it's the
# freshest state (saved by the USR1-signal handler ~120s before job 2175326
# hit its --time limit, ~2h more recent than the last periodic validation
# checkpoint). Lightning will resume mid-epoch (it fast-forwards the batch
# counter rather than restarting the epoch from batch 0), but since our
# dataloader doesn't implement the stateful state_dict/load_state_dict
# interface, expect a `PossibleUserWarning` about the dataloader not being
# resumable in the logs -- harmless, not a sign of a botched resume.
#
# Usage:
#   sbatch run_flair3d_plus_multitask_200k_a100.slurm
#   sbatch run_flair3d_plus_multitask_200k_a100.slurm 2e-3                     # lr override
#   sbatch run_flair3d_plus_multitask_200k_a100.slurm 1e-3 /path/to/other.ckpt # different ckpt
LR="${1:-1e-3}"
CKPT_PATH="${2:-/lustre/fswork/projects/rech/unv/usi32yh/myria3d/logs/slurm/2175326/hydra_run/hpc_ckpt_1.ckpt}"

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
    echo "resuming from: ${CKPT_PATH}"
    nvidia-smi  # État initial du GPU
} > ${JOB_DIR}/job_info.log

module purge
# TODO verify: A100 nodes on JZ typically don't need an `arch/*` module swap
# (H100 needed `module load arch/h100` because it's the non-default arch) --
# drop this line or replace it with the correct one per `module avail arch`.
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
# `srun` is required for PyTorch Lightning to detect it's running under SLURM
# and actually receive the USR1 signal declared above. `hydra.run.dir` is
# pinned to a job-scoped (not timestamped) path so it stays identical across
# requeues of the same $SLURM_JOB_ID.
srun python run.py experiment=flair3d_plus/multitask_200k \
  model.lr=${LR} \
  model.ckpt_path=${CKPT_PATH} \
  hydra.run.dir=${JOB_DIR}/hydra_run \
  '+trainer.plugins=[{_target_:pytorch_lightning.plugins.environments.SLURMEnvironment}]'

### ------------------------

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

{
    echo "Job finished at: $(date)"
    echo "Duration: ${DURATION} seconds"
    nvidia-smi  # État final du GPU
} >> ${JOB_DIR}/job_info.log
