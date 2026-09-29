#!/bin/bash

#SBATCH --output=/lustre/fswork/projects/rech/unv/usi32yh/myria3d/logs/slurm/%j.out
#SBATCH --error=/lustre/fswork/projects/rech/unv/usi32yh/myria3d/logs/slurm/%j.err
#SBATCH -A uhn@a100
#SBATCH -C a100

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread

#SBATCH --job-name=myria3d-mt-test
#SBATCH --time=19:45:00

# Full 104/52 test pass takes ~10h15 on 1 A100 (5872 batches @ ~0.14 it/s; job 274440 was
# killed by a 10h limit at 5684/5872, and since metrics + Pointcept dumps are only written
# at the end of the test epoch, that run left nothing behind).
#
# Test-only pass of a Flair3D+ multitask checkpoint (full test split), writing the
# Pointcept-format `result/{patch_id}_logits_{network,forest_2d}.npy` dumps for
# Pointcept's tools/eval_network_apls.py.
#
# The val/test mosaic is 104/52 by default (preparations eval_tile_width/eval_subtile_width,
# see points_budget_flair3d_plus_multitask.yaml): Pointcept patches are ~102.4 x 102.5 m, and
# the former 100/50 mosaic left the east/north ~2.5 m strip of every patch unpredicted ->
# NaN cuts every ~102 m in the stitched ROI raster that collapsed APLS G->G' (perfect
# GT-as-prediction scored 1.0 instead of 47.8 on D068-2021/UU-S1-2). Override with
# EVAL_TILE_WIDTH / EVAL_SUBTILE_WIDTH (keep a 2:1 ratio -> 4 quadrants).
#
# Usage:
#   sbatch run_flair3d_plus_multitask_test_a100.slurm /path/to/checkpoint.ckpt
CKPT_PATH="${1:?usage: sbatch $0 /path/to/checkpoint.ckpt}"
EVAL_TILE_WIDTH="${EVAL_TILE_WIDTH:-104}"
EVAL_SUBTILE_WIDTH="${EVAL_SUBTILE_WIDTH:-52}"
NUM_WORKERS="${NUM_WORKERS:-8}"

JOB_DIR=/lustre/fswork/projects/rech/unv/usi32yh/myria3d/logs/slurm/${SLURM_JOB_ID}
mkdir -p ${JOB_DIR}

ln -sf ../${SLURM_JOB_ID}.out ${JOB_DIR}/${SLURM_JOB_ID}.out
ln -sf ../${SLURM_JOB_ID}.err ${JOB_DIR}/${SLURM_JOB_ID}.err

cp $0 ${JOB_DIR}/script.slurm

{
    echo "Job ID: $SLURM_JOB_ID"
    echo "Starting job at: $(date)"
    echo "Running on host: $(hostname)"
    echo "Working directory: $(pwd)"
    echo "Python executable: $(which python)"
    echo "checkpoint: ${CKPT_PATH}"
    echo "eval tile_width/subtile_width: ${EVAL_TILE_WIDTH}/${EVAL_SUBTILE_WIDTH}"
    echo "num_workers: ${NUM_WORKERS}"
    nvidia-smi
} > ${JOB_DIR}/job_info.log

module purge
module load arch/a100
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
# Dumps land in ${JOB_DIR}/hydra_run/result (PointceptPredictionDump output_dir is
# relative to the Hydra run dir).
srun python run.py experiment=flair3d_plus/multitask_200k \
  task.task_name=test \
  model.ckpt_path=${CKPT_PATH} \
  datamodule.transforms.preparations.eval_tile_width=${EVAL_TILE_WIDTH} \
  datamodule.transforms.preparations.eval_subtile_width=${EVAL_SUBTILE_WIDTH} \
  datamodule.num_workers=${NUM_WORKERS} \
  logger.wandb.name=test_tw${EVAL_TILE_WIDTH}_job${SLURM_JOB_ID} \
  logger.wandb.job_type=test \
  hydra.run.dir=${JOB_DIR}/hydra_run \
  '+trainer.plugins=[{_target_:pytorch_lightning.plugins.environments.SLURMEnvironment}]'

### ------------------------

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

{
    echo "Job finished at: $(date)"
    echo "Duration: ${DURATION} seconds"
    echo "Pointcept dumps: ${JOB_DIR}/hydra_run/result"
    echo "wandb (offline): wandb sync ${JOB_DIR}/hydra_run/wandb/offline-run-*"
    nvidia-smi
} >> ${JOB_DIR}/job_info.log
