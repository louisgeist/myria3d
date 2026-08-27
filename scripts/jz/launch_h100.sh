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


#SBATCH --job-name=myria3d
#SBATCH --time=16:00:00


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
cd /lustre/fswork/projects/rech/unv/usi32yh/myria3d

START_TIME=$(date +%s)


###  --- Compiled code ---
# Export path to code compiled for H100

# Pointops
POINTOPS_PATH=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/pointops_build_h100/lib/python3.10/site-packages/pointops-1.0-py3.10-linux-x86_64.egg

# Add pointops to PYTHONPATH | not overriden if already set
export PYTHONPATH="${POINTOPS_PATH}${PYTHONPATH:+:$PYTHONPATH}"


### --- Myria3D syntax ---
python run.py experiment=flair3d_plus/multitask \
  datamodule.data_root=/lustre/fsn1/projects/rech/unv/usi32yh/data_pointcept/flair3d_plus \
  datamodule.csv_manifest=/lustre/fsn1/projects/rech/unv/usi32yh/data_pointcept/flair3d_plus/raw/scene_split_manifest.csv

### ------------------------

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

{
    echo "Job finished at: $(date)"
    echo "Duration: ${DURATION} seconds"
    nvidia-smi  # État final du GPU
} >> ${JOB_DIR}/job_info.log
