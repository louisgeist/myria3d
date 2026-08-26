# Flair3D+ with Flair3D-build labels

This workflow trains on the full Flair3D+ dataset using precomputed semantic labels from [Flair3D-build](https://github.com/IGNF/Flair3D-build). Use **`label=v20`** in Flair3D-build (current); myria3d only reads the `semantic` field from the output PLY (16 classes including Void). Config and file names still say `v12` for historical reasons.

The steps are the same on every machine:

1. Build label PLY files with Flair3D-build.
2. Generate a `basename,split` CSV with myria3d (absolute PLY paths).
3. Create an HDF5 cache (`task.task_name=create_hdf5`), then train.

Output PLY layout (Flair3D-build):

`{output_root}/LIDARHD/{dept_year}_LIDARHD/{roi}/{dept_year}_LIDARHD_{roi}_{scene_i_j}.ply`

Each file contains a `semantic` field (uint8). Raw `cosia_class` / `lidarhd_class` are removed.

Before any `python run.py` command, ensure a `.env` file exists at the myria3d repo root with at least:

```bash
LOGS_DIR="logs/"
```

---

## Hecate

Local workstation. Repositories and data live under `/data/geist/`.

| Role | Path |
|------|------|
| Flair3D-build | `/data/geist/Flair3D-build` |
| myria3d | `/data/geist/myria3d` |
| Pointcept raw (manifest, tiles) | `/data/geist/Pointcept/data/flair3d_plus` |
| Built labels (`output_root`) | `/data/geist/Flair3D-build/data/flair3d_label_enhanced` |

Conda envs: `flair3d-label` (Flair3D-build), `myria3d` (training).

### 1. Build labels (Flair3D-build)

```bash
cd /data/geist/Flair3D-build
conda activate flair3d-label

# Default Hydra config: config_hecate (D067 manifest)
python src/main.py --config-name=config_hecate

# Other departments, e.g. D010 or D075:
# python src/main.py --config-name=config_hecate_D010
# python src/main.py --config-name=config_hecate_D075
```

### 2. Generate split CSV (myria3d)

Example for department **D067**:

```bash
cd /data/geist/myria3d
conda activate myria3d

python -m myria3d.pctl.dataset.flair3d \
  --split-manifest-csv /data/geist/Pointcept/data/flair3d_plus/raw/scene_split_manifest_D067.csv \
  --labels-root /data/geist/Flair3D-build/data/flair3d_label_enhanced \
  --output-csv /data/geist/myria3d/tests/data/flair3d_plus_split_D067.csv \
  --excluded-tiles-details-csv /data/geist/Pointcept/data/flair3d_plus/missing_coord_tiles.details.csv \
  --require-existing-ply
```

The CSV uses absolute paths in `basename` (compatible with `datamodule.data_dir: "."`).

### 3. Create HDF5 and train

```bash
cd /data/geist/myria3d
conda activate myria3d

python run.py experiment=flair3d_plus/base_v12 \
  datamodule.split_csv_path=/data/geist/myria3d/tests/data/flair3d_plus_split_D067.csv \
  datamodule.hdf5_file_path=/data/geist/myria3d/tests/data/flair3d_plus_v12_D067.hdf5 \
  task.task_name=create_hdf5
```

Then fit:

```bash
python run.py experiment=flair3d_plus/base_v12 \
  datamodule.split_csv_path=/data/geist/myria3d/tests/data/flair3d_plus_split_D067.csv \
  datamodule.hdf5_file_path=/data/geist/myria3d/tests/data/flair3d_plus_v12_D067.hdf5
```

After HDF5 creation, you may set `datamodule.data_dir` and `datamodule.split_csv_path` to `null` and only pass `datamodule.hdf5_file_path`.

---

## Multitask

See the repo guide [`readme_flair3d.md`](../../../readme_flair3d.md) (segment + forest_2d + roads + nathab axes + elevation).

---

## Jean Zay

IDRIS cluster (account `unv@h100`). Persistent data on `fsn1`, working copies on `fswork`. Adjust `usi32yh` if your login differs.

| Role | Path |
|------|------|
| Flair3D-build | `/lustre/fswork/projects/rech/unv/usi32yh/Flair3D-build` |
| myria3d | `/lustre/fswork/projects/rech/unv/usi32yh/myria3d` |
| Pointcept raw | `/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/data/flair3d_plus/raw` |
| Built labels (`output_root`) | `/lustre/fsn1/projects/rech/unv/usi32yh/data_flair3d_build/flair3d_label_enhanced` |

Load modules before activating conda (see [Flair3D-build `build_env_cuda12.md`](https://github.com/IGNF/Flair3D-build/blob/main/build_env_cuda12.md)):

```bash
module purge
module load arch/h100
module load cuda/12.1.0
module load miniforge/24.9.0
```

Conda envs: `flair3d-label` (Flair3D-build), `myria3d` (training).

### 1. Build labels (Flair3D-build)

Interactive (login or compute node):

```bash
cd /lustre/fswork/projects/rech/unv/usi32yh/Flair3D-build
conda activate flair3d-label

python src/main.py --config-name=config_jz
```

Or submit the provided SLURM script (GPU, ~3 h):

```bash
cd /lustre/fswork/projects/rech/unv/usi32yh/Flair3D-build
mkdir -p logs
sbatch scripts/jz/run_flair3d_build.slurm
```

`config_jz` uses the full `scene_split_manifest.csv` (all departments), not the D067-only manifest used on Hecate.

### 2. Generate split CSV (myria3d)

```bash
cd /lustre/fswork/projects/rech/unv/usi32yh/myria3d
conda activate myria3d

python -m myria3d.pctl.dataset.flair3d \
  --split-manifest-csv /lustre/fswork/projects/rech/unv/usi32yh/Pointcept/data/flair3d_plus/raw/scene_split_manifest.csv \
  --labels-root /lustre/fsn1/projects/rech/unv/usi32yh/data_flair3d_build/flair3d_label_enhanced \
  --output-csv /lustre/fsn1/projects/rech/unv/usi32yh/data/myria3d/flair3d_plus_split.csv \
  --excluded-tiles-details-csv /lustre/fswork/projects/rech/unv/usi32yh/Pointcept/data/flair3d_plus/missing_coord_tiles.details.csv \
  --require-existing-ply
```

For a single department on Jean Zay, point `--split-manifest-csv` to e.g. `scene_split_manifest_D067.csv` and adjust `--output-csv` accordingly.

### 3. Create HDF5 and train

HDF5 creation is I/O-heavy; a login node is fine for a smoke test, but prefer a CPU compute job for the full dataset.

```bash
cd /lustre/fswork/projects/rech/unv/usi32yh/myria3d
conda activate myria3d
mkdir -p logs

sbatch scripts/jz/run_flair3d_plus_hdf5.slurm
```

Or interactively:

```bash
python run.py experiment=flair3d_plus/base_v12 \
  datamodule.split_csv_path=/lustre/fsn1/projects/rech/unv/usi32yh/data/myria3d/flair3d_plus_split.csv \
  datamodule.hdf5_file_path=/lustre/fsn1/projects/rech/unv/usi32yh/data/myria3d/flair3d_plus_v12.hdf5 \
  task.task_name=create_hdf5
```

Training (GPU). Submit with SLURM — creates the HDF5 on first run if it does not exist yet:

```bash
sbatch scripts/jz/run_flair3d_plus_train.slurm
```

Or with `srun` (one H100):

```bash
srun --account=unv@h100 -C h100 --gres=gpu:1 --cpus-per-task=10 --time=20:00:00 \
  python run.py experiment=flair3d_plus/base_v12 \
  datamodule.split_csv_path=/lustre/fsn1/projects/rech/unv/usi32yh/data/myria3d/flair3d_plus_split.csv \
  datamodule.hdf5_file_path=/lustre/fsn1/projects/rech/unv/usi32yh/data/myria3d/flair3d_plus_v12.hdf5 \
  datamodule.data_dir=.
```

`run_flair3d_plus_hdf5.slurm` remains optional (CPU-only HDF5 build before training).

Scripts live under `scripts/jz/`. Override paths without editing the file, e.g.:

```bash
export JZ_USER=usi32yh
export SPLIT_CSV=/lustre/fsn1/projects/rech/unv/usi32yh/data/myria3d/flair3d_plus_split_D067.csv
export HDF5_PATH=/lustre/fsn1/projects/rech/unv/usi32yh/data/myria3d/flair3d_plus_v12_D067.hdf5
sbatch scripts/jz/run_flair3d_plus_train.slurm
```
