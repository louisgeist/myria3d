# Flair3D+ in myria3d

Guide for training on Flair3D+ with myria3d: single-task semantic segmentation (PLY labels) and **multitask** (PLY + GeoTIFF rasters).

Use **`label=v14`** in Flair3D-build. Config paths still say `v12` for historical reasons.

Before any `python run.py` command, create a `.env` at the myria3d repo root:

```bash
LOGS_DIR="logs/"
```

---

## Overview

| Mode | Experiment config | Labels source | Model |
|------|-------------------|---------------|-------|
| Single-task (segment) | `experiment=flair3d_plus/base_v12` | PLY `semantic` (Flair3D-build) | `PyGRandLANet` |
| Multitask | `experiment=flair3d_plus/multitask_v12` | PLY + GeoTIFF rasters | `PyGRandLANetMultiTask` |

Common pipeline:

1. Build label PLY files with Flair3D-build.
2. Generate a `basename,split` CSV with myria3d (absolute PLY paths).
3. Create an HDF5 cache (`task.task_name=create_hdf5`), then train.

PLY layout (Flair3D-build output):

```
{output_root}/LIDARHD/{dept_year}_LIDARHD/{roi}/{dept_year}_LIDARHD_{roi}_{scene_i_j}.ply
```

Each PLY contains a `semantic` field (uint8, 16 classes including Void).

---

## Single-task (segment only)

### Hecate (local)

| Role | Path |
|------|------|
| Flair3D-build | `/data/geist/Flair3D-build` |
| myria3d | `/data/geist/myria3d` |
| Pointcept raw (manifest) | `/data/geist/Pointcept/data/flair3d_plus` |
| Built labels | `/data/geist/Flair3D-build/data/flair3d_label_enhanced` |

**1. Build labels**

```bash
cd /data/geist/Flair3D-build && conda activate flair3d-label
python src/main.py --config-name=config_hecate
```

**2. Split CSV (example D067)**

```bash
cd /data/geist/myria3d && conda activate myria3d

python -m myria3d.pctl.dataset.flair3d \
  --split-manifest-csv /data/geist/Pointcept/data/flair3d_plus/raw/scene_split_manifest_D067.csv \
  --labels-root /data/geist/Flair3D-build/data/flair3d_label_enhanced \
  --output-csv /data/geist/myria3d/tests/data/flair3d_plus_split_D067.csv \
  --excluded-tiles-details-csv /data/geist/Pointcept/data/flair3d_plus/missing_coord_tiles.details.csv \
  --require-existing-ply
```

**3. HDF5 + train**

```bash
python run.py experiment=flair3d_plus/base_v12 \
  datamodule.split_csv_path=/data/geist/myria3d/tests/data/flair3d_plus_split_D067.csv \
  datamodule.hdf5_file_path=/data/geist/myria3d/tests/data/flair3d_plus_v12_D067.hdf5 \
  task.task_name=create_hdf5

python run.py experiment=flair3d_plus/base_v12 \
  datamodule.split_csv_path=/data/geist/myria3d/tests/data/flair3d_plus_split_D067.csv \
  datamodule.hdf5_file_path=/data/geist/myria3d/tests/data/flair3d_plus_v12_D067.hdf5
```

After HDF5 creation, you may set `datamodule.data_dir` and `datamodule.split_csv_path` to `null` and only pass `datamodule.hdf5_file_path`.

### Jean Zay

See also `scripts/jz/run_flair3d_plus_hdf5.slurm` and `scripts/jz/run_flair3d_plus_train.slurm`.

```bash
sbatch scripts/jz/run_flair3d_plus_hdf5.slurm
sbatch scripts/jz/run_flair3d_plus_train.slurm
```

Override paths without editing the script:

```bash
export JZ_USER=usi32yh
export SPLIT_CSV=/lustre/fsn1/projects/rech/unv/usi32yh/data/myria3d/flair3d_plus_split_D067.csv
export HDF5_PATH=/lustre/fsn1/projects/rech/unv/usi32yh/data/myria3d/flair3d_plus_v12_D067.hdf5
sbatch scripts/jz/run_flair3d_plus_train.slurm
```

---

## Multitask

Trains **five tasks** jointly with a shared RandLA-Net backbone:

| Task | Source | Remap (Pointcept-aligned) | Output dim | `ignore_index` |
|------|--------|---------------------------|------------|----------------|
| `segment` | PLY `semantic` | default | 16 | 15 |
| `forest` | GeoTIFF `FOREST` | default | 2 | 2 |
| `land_use` | GeoTIFF `LAND_USE` | filtered | 10 | 10 |
| `natural_habitat` | GeoTIFF `NATURAL_HABITAT` | by_habitat_x_domain | 11 | 11 |
| `elevation` | GeoTIFF `DEM_ELEV` band 2 (DTM) | — (regression) | 1 | NaN masked |

Elevation target: `z_lidar - DTM` (meters). Training uses scale `0.01` and `SmoothL1Loss`.

### Raster layout

GeoTIFFs live under `datamodule.raster_root` (same tree as Pointcept):

```
{raster_root}/FOREST/{dept_year}_FOREST/{roi}/{dept_year}_FOREST_{roi}_{scene_i_j}.tif
{raster_root}/LAND_USE/...
{raster_root}/NATURAL_HABITAT/...
{raster_root}/DEM_ELEV/...
```

Stem rule: replace `_LIDARHD_` with `_{MODALITY}_` in the PLY filename.

Raster sampling runs **once per patch** (before subtiling). Any change to preprocessing or remaps requires **recreating the HDF5**.

### Commands (Hecate example)

```bash
python run.py experiment=flair3d_plus/multitask_v12 \
  datamodule.split_csv_path=/data/geist/myria3d/tests/data/flair3d_plus_split_D067.csv \
  datamodule.hdf5_file_path=/data/geist/myria3d/tests/data/flair3d_plus_multitask_v12_D067.hdf5 \
  datamodule.raster_root=/data/geist/Pointcept/data/flair3d_plus/raw \
  task.task_name=create_hdf5

python run.py experiment=flair3d_plus/multitask_v12 \
  datamodule.split_csv_path=/data/geist/myria3d/tests/data/flair3d_plus_split_D067.csv \
  datamodule.hdf5_file_path=/data/geist/myria3d/tests/data/flair3d_plus_multitask_v12_D067.hdf5 \
  datamodule.raster_root=/data/geist/Pointcept/data/flair3d_plus/raw
```

Jean Zay: same overrides as single-task, but use `experiment=flair3d_plus/multitask_v12`, a distinct `HDF5_PATH`, and set `RASTER_ROOT` to the Pointcept raw directory on lustre.

### Training from Pointcept `.npy` (shortcut)

If Flair3D+ tiles are already preprocessed by [Pointcept](https://github.com/Pointcept/Pointcept) (folders under `train/`, `val/`, `test/` with `coord.npy`, `color.npy`, `strength.npy`, multitask label arrays), you can train **without** `create_hdf5`:

```bash
python run.py experiment=flair3d_plus/multitask_v12_pointcept \
  datamodule.data_root=/data/geist/Pointcept/data/flair3d_plus \
  datamodule.csv_manifest=/data/geist/Pointcept/data/flair3d_plus/raw/scene_split_manifest_D067.csv
```

Prerequisites:

- Pointcept output layout: `{data_root}/{split}/{dept_year}_LIDARHD/{roi}/{patch_id}/`
- `coord.npy` present (completion marker)
- Same manifest / exclusions as Pointcept (`missing_coord_tiles.details.csv`, `too_small_tiles.csv`)

Each Pointcept patch is ~100 m on disk; myria3d crops it to 50×50 m subtiles on the fly (same mosaic as the HDF5 pipeline). Train: one random quadrant per tile per epoch. Val/test: four deterministic quadrants per tile (`subtile_index` 0–3). Downsampling is handled at train time (`SubtileCrop` → `GridSampling` → `MaximumNumNodes`). RGB stays in 0–255 float (Flair3D+ convention, not `/255`). Set `datamodule.tile_width=100` and `datamodule.subtile_width=50` for correct `NormalizePos`.

**Perf note:** val/test reload the full `.npy` scene once per quadrant (4× I/O per tile). Usually negligible vs GPU forward (OS page cache helps: quadrants are consecutive, no val shuffle). If val I/O becomes a bottleneck, add a per-scene in-memory cache in `PointceptNpyDataset`.

For faster validation (metrics on subsampled points, skip full-res KNN) while keeping full-res test metrics:

```bash
model.interpolate_at_val=false
```

#### Iter-limited training schedule (Pointcept-compatible)

The Flair3D+ train set is very large. By default, `multitask_v12_pointcept` uses **iter-limited** mode (same semantics as Pointcept): one training epoch is a fixed number of optimizer steps, not a full dataset pass.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `total_iters` | `10000` | Total optimizer steps for the run |
| `iter_per_epoch` | `1000` | Batch steps per training epoch |
| `eval_every` | `5` | Validate every N training epochs (last epoch always validated) |

Derived values: `num_epochs = total_iters // iter_per_epoch` (exact division required). Example with defaults: 10 epochs × 1000 steps = 10 000 steps; validation at epochs 5 and 10.

Override from CLI:

```bash
python run.py experiment=flair3d_plus/multitask_v12_pointcept \
  total_iters=10000 iter_per_epoch=1000 eval_every=5 \
  datamodule.data_root=/data/geist/Pointcept/data/flair3d_plus \
  datamodule.csv_manifest=/data/geist/Pointcept/data/flair3d_plus/raw/scene_split_manifest_D067.csv
```

To use classic epoch-based training instead, set `total_iters=null` and configure `trainer.max_epochs` explicitly.

**Callbacks:** `early_stopping.patience` and `ReduceLROnPlateau.patience` count **validation events**, not training epochs. With `eval_every=5`, `patience=6` means 6 validations ≈ 30 training epochs (30 000 steps with default `iter_per_epoch`).

### Key config files

| File | Role |
|------|------|
| `configs/experiment/flair3d_plus/multitask_v12.yaml` | Experiment entry point (HDF5 pipeline) |
| `configs/experiment/flair3d_plus/multitask_v12_pointcept.yaml` | Multitask from Pointcept `.npy` tiles (iter-limited by default) |
| `configs/training_schedule/default.yaml` | `total_iters`, `iter_per_epoch`, `eval_every` defaults |
| `configs/datamodule/pointcept_npy_datamodule.yaml` | Datamodule for Pointcept preprocessed data |
| `configs/dataset_description/flair3d_plus_multitask.yaml` | Task dims, weights, elevation scale |
| `configs/model/pyg_randla_net_multitask_model.yaml` | `PyGRandLANetMultiTask` hparams |
| `configs/callbacks/multitask.yaml` | Per-task metrics, early stopping on `val/iou` (segment) |

### Code map

| Component | Path |
|-----------|------|
| Label LUTs / remaps | `myria3d/pctl/dataset/flair3d_label_remap.py` |
| GeoTIFF sampling | `myria3d/pctl/dataset/raster_utils.py` |
| Raster enrichment + pre-transform | `myria3d/pctl/dataset/flair3d.py` |
| HDF5 create/load | `myria3d/pctl/dataset/hdf5.py` |
| Pointcept `.npy` loader | `myria3d/pctl/dataset/pointcept_npy.py` |
| Pointcept datamodule | `myria3d/pctl/datamodule/pointcept_npy.py` |
| Iter-limited sampler | `myria3d/pctl/dataloader/iter_limited_sampler.py` |
| Training schedule resolver | `myria3d/utils/training_schedule.py` |
| Multitask backbone | `myria3d/models/modules/pyg_randla_net_multitask.py` |
| Lightning module | `myria3d/models/multitask_model.py` |
| Metrics callback | `myria3d/callbacks/multitask_metric_callbacks.py` |

### Dependencies

Multitask preprocessing requires `rasterio` (in `environment.yml`). Update the conda env if needed:

```bash
mamba env update -f environment.yml
```

### Monitoring

- Main metric: `val/iou` on **segment**
- Per-task: `val/iou_forest`, `val/iou_land_use`, `val/iou_natural_habitat`
- Regression: `val/elevation_mae`, `val/elevation_rmse`
- Per-task losses: `train/loss_segment`, `train/loss_forest`, …

---

## Sphinx tutorial

A longer machine-specific walkthrough (Hecate + Jean Zay paths) also lives in `docs/source/tutorials/flair3d_plus.md`.
