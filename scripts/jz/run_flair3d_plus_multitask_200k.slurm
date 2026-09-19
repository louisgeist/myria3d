#!/bin/bash

#SBATCH --output=/lustre/fswork/projects/rech/unv/usi32yh/myria3d/logs/slurm/%j.out
#SBATCH --error=/lustre/fswork/projects/rech/unv/usi32yh/myria3d/logs/slurm/%j.err
#SBATCH -A ppm@h100
#SBATCH -C h100

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread

#SBATCH --job-name=myria3d-200k
#SBATCH --time=19:45:00
#SBATCH --signal=USR1@120
#SBATCH --requeue
# 200k iterations is ~20x the reference 10k run this --time budget comes from
# (scripts/jz/launch_h100.sh) -- it will very likely need several requeues.
# Adjust --time to your actual throughput/QOS.
#
# Auto-requeue: `--signal=USR1@120` has SLURM deliver SIGUSR1 to the running
# `srun` step 120s before the `--time` limit hits. PyTorch Lightning's SLURM
# auto-requeue (enabled below via `+trainer.plugins=[...SLURMEnvironment]`, only
# armed because we run under `srun` -- see `_SignalConnector` in
# pytorch_lightning) catches that signal, saves a full checkpoint (weights,
# optimizer/scheduler state, epoch/global_step, callback state, and -- via
# MultiTaskModel.on_save_checkpoint -- the GradNorm-lite EMA, which otherwise
# lives outside any state_dict and would silently reset on resume) to
# `hpc_ckpt_N.ckpt` under `hydra.run.dir`, then calls `scontrol requeue`
# itself and exits. SLURM puts the job back in the queue under the *same*
# $SLURM_JOB_ID; when it restarts, `hydra.run.dir` is pinned below to the same
# job-scoped path, so Lightning's own hpc-checkpoint auto-detection finds that
# file and resumes from it with no flag needed on our end (see
# `_parse_ckpt_path` / `_hpc_resume_path` in pytorch_lightning) -- fully
# automatic, no manual resubmission.
#
# This only fires on a clean time-limit requeue. If the job instead dies some
# other way (node failure before the signal, manual scancel, OOM, ...), fall
# back to resubmitting by hand with
# `model.ckpt_path=<JOB_DIR>/hydra_run/checkpoints/last.ckpt` appended to the
# command below (save_last: true is on, see configs/callbacks/multitask.yaml;
# this is the same real Lightning resume path, just picking the last
# periodically-saved checkpoint instead of the exact-at-signal one -- not
# `task.task_name=finetune`, which restarts the schedule from epoch 0).

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
# `srun` is required for PyTorch Lightning to detect it's running under SLURM
# and actually receive the USR1 signal declared above. `hydra.run.dir` is
# pinned to a job-scoped (not timestamped) path so it stays identical across
# requeues of the same $SLURM_JOB_ID -- both for the hpc auto-resume checkpoint
# lookup and for the manual `checkpoints/last.ckpt` fallback mentioned above.
srun python run.py experiment=flair3d_plus/multitask_200k \
  model.lr=${LR} \
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
