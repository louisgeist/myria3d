# Flair3D+ in myria3d

Guide for training on Flair3D+ with myria3d: single-task semantic segmentation (PLY labels) and **multitask** (PLY + GeoTIFF rasters).

Use **`label=v20`** in Flair3D-build. Config paths still say `v12` for historical reasons.

Before any `python run.py` command, create a `.env` at the myria3d repo root:

```bash
LOGS_DIR="logs/"
```

---

## Overview

| Mode | Experiment config | Labels source | Model |
|------|-------------------|---------------|-------|
| Single-task (segment) | `experiment=flair3d_plus/base_v12` | PLY `semantic` (Flair3D-build) | `PyGRandLANet` |
| Multitask | `experiment=flair3d_plus/multitask` | Pointcept `.npy` tiles | `PyGRandLANetMultiTask` |

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

Default entry point: `experiment=flair3d_plus/multitask`. It trains from Pointcept `.npy` tiles (no HDF5) with GradNorm-lite enabled. The RandLA-Net backbone is shared across eight heads:

| Task | Source | Type | Output dim | `ignore_index` |
|------|--------|------|------------|----------------|
| `segment` | `segment.npy` (Flair3D-build v20) | per-point semantic | 16 | 15 |
| `forest_2d` | `forest_2d.npy` (1 m raster) | pixel_semantic (mean-pool) | 2 | 2 |
| `roads` | `network.npy` channel `ROADS` | pixel_semantic (max-pool) | 2 | 2 |
| `nathab_habitat_type` | `natural_habitat.npy` → `by_habitat_type_ecological` | tile_distribution (WeightedKL) | 4 | 4 |
| `nathab_moisture_regime` | same → `by_moisture_regime` | tile_distribution | 3 | 3 |
| `nathab_soil_chemistry` | same → `by_soil_chemistry` | tile_distribution | 2 | 2 |
| `nathab_bioclimatic_zone` | same → `by_climatic_domain` | tile_distribution | 3 | 3 |
| `elevation` | `elevation.npy` (`z_lidar - DTM`) | per-point regression | 1 | NaN masked |

`forest_2d.npy` / `network.npy` are produced by Pointcept's `rasterize_forest.py` / `rasterize_network.py` (run outside myria3d). Roads training is raster CE with foreground weight 5; graph APLS evaluation is a follow-up, not part of this recipe.

Elevation target: `z_lidar - DTM` (meters). Training uses scale `0.01` and `SmoothL1Loss`.

### Training from Pointcept `.npy`

Tiles must already be preprocessed by [Pointcept](https://github.com/Pointcept/Pointcept) (folders under `train/`, `val/`, `test/` with `coord.npy`, `color.npy`, `strength.npy`, multitask label arrays):

```bash
python run.py experiment=flair3d_plus/multitask
```

Hecate defaults in the experiment YAML: `data_root=/data/geist/Pointcept/data/flair3d_plus` and the D067 manifest. Jean Zay overrides `data_root` / `csv_manifest` (see `scripts/jz/launch_h100.sh`).

Prerequisites:

- Pointcept output layout: `{data_root}/{split}/{dept_year}_LIDARHD/{roi}/{patch_id}/`
- `coord.npy` present (completion marker)
- Same manifest / exclusions as Pointcept (`missing_coord_tiles.details.csv`, `too_small_tiles.csv`)

Each Pointcept patch is ~100 m on disk; myria3d crops it to 50×50 m subtiles on the fly. Train: one random quadrant per tile per epoch. Val/test: four deterministic quadrants per tile (`subtile_index` 0–3). Downsampling is handled at train time (`SubtileCrop` → `GridSampling` → `MaximumNumNodes`). RGB stays in 0–255 float (Flair3D+ convention, not `/255`). Set `datamodule.tile_width=100` and `datamodule.subtile_width=50` for correct `NormalizePos`.

Train augmentations (`light_radiometry.yaml`) match Pointcept: XY flips, then `RandomDropColor` / `RandomDropStrength` (20% of scenes drop all RGB or intensity; 50% drop 10% of points). Dropped channels are zeroed as a placeholder; `MultiTaskModel` replaces them with learned fill-in parameters (`color_mask_value`, `strength_mask_value`, logged as `train/learned_mask/*`). Val/test do not drop.

**Perf note:** val/test reload the full `.npy` scene once per quadrant (4× I/O per tile). Usually negligible vs GPU forward (OS page cache helps: quadrants are consecutive, no val shuffle). If val I/O becomes a bottleneck, add a per-scene in-memory cache in `PointceptNpyDataset`.

For faster validation (metrics on subsampled points, skip full-res KNN) while keeping full-res test metrics:

```bash
model.interpolate_at_val=false
```

#### Iter-limited training schedule (Pointcept-compatible)

The Flair3D+ train set is very large. By default, `multitask` uses **iter-limited** mode (same semantics as Pointcept): one training epoch is a fixed number of optimizer steps, not a full dataset pass.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `total_iters` | `10000` | Total optimizer steps for the run |
| `iter_per_epoch` | `1000` | Batch steps per training epoch |
| `eval_every` | `5` | Validate every N training epochs (last epoch always validated) |

Derived values: `num_epochs = total_iters // iter_per_epoch` (exact division required). Example with defaults: 10 epochs × 1000 steps = 10 000 steps; validation at epochs 5 and 10.

Override from CLI:

```bash
python run.py experiment=flair3d_plus/multitask \
  total_iters=10000 iter_per_epoch=1000 eval_every=5
```

To use classic epoch-based training instead, set `total_iters=null` and configure `trainer.max_epochs` explicitly.

**Callbacks:** `early_stopping.patience` and `ReduceLROnPlateau.patience` count **validation events**, not training epochs. With `eval_every=5`, `patience=6` means 6 validations ≈ 30 training epochs (30 000 steps with default `iter_per_epoch`).

### Key config files

| File | Role |
|------|------|
| `configs/experiment/flair3d_plus/multitask.yaml` | Default multitask experiment (Pointcept `.npy`, GradNorm-lite) |
| `configs/training_schedule/default.yaml` | `total_iters`, `iter_per_epoch`, `eval_every` defaults |
| `configs/datamodule/pointcept_npy_datamodule.yaml` | Datamodule for Pointcept preprocessed data |
| `configs/dataset_description/flair3d_plus_multitask.yaml` | Task dims, weights, elevation scale |
| `configs/model/pyg_randla_net_multitask_model.yaml` | `PyGRandLANetMultiTask` hparams |
| `configs/model/multitask_default.yaml` | GradNorm-lite, learned RGB/intensity fill-in, task-weight defaults |
| `configs/datamodule/transforms/augmentations/light_radiometry.yaml` | XY flips + RandomDropColor / RandomDropStrength |
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
| Pixel-semantic pooling | `myria3d/models/modules/pixel_pooling.py` |
| Tile-distribution loss | `myria3d/models/losses.py` |
| Lightning module | `myria3d/models/multitask_model.py` |
| GradNorm-lite + diagnostic gradient-norm utilities | `myria3d/models/gradnorm_lite.py` |
| Metrics callback | `myria3d/callbacks/multitask_metric_callbacks.py` |

### Dependencies

Multitask preprocessing requires `rasterio` (in `environment.yml`). Update the conda env if needed:

```bash
mamba env update -f environment.yml
```

### Monitoring

Logged every epoch (`train/` / `val/` / `test/`). `val/iou` (segment mIoU) remains the checkpoint / early-stopping key.

| Task | Headline | Also logged |
|------|----------|-------------|
| `segment` | `{split}/segment/mIoU` | `{split}/segment/iou/{class}` (val/test), alias `{split}/iou` |
| `forest_2d`, `roads` | `{split}/{task}/mIoU` | cell-pooled IoU; `{split}/{task}/iou/{class}` (val/test) |
| `elevation` | `{split}/reg/elevation/mae` | RMSE; values in **meters** |
| `nathab_*` | `{split}/tv/{task}` | weighted KL; per-class frequency MAE `{split}/mae/{task}/{class}` |
| all 4 nathab axes | `{split}/tv/nathab_total` | **unweighted sum** of the 4 axis TVs (Pointcept) |

Nathab TV: pool each tile to distributions `(pi_hat, q_t)`, accumulate `n_t · |pi_hat - q_t|` over tiles, then `TV = sum_c (abs_weighted_c / sum n_t)`. That is the point-count-weighted mean of per-tile L1. `nathab_total` then **adds the four axis TVs** (no extra weighting).

Graph **APLS** is not logged here (needs dense raster export + ROI stitching; Pointcept `NetworkAPLSEvaluator`). Roads monitoring is raster mIoU / per-class IoU only.

Per-task losses: `train/loss_segment`, `train/loss_forest_2d`, `train/loss_roads`, `train/loss_nathab_*`, …

### Loss balancing (GradNorm-lite)

Ported from Pointcept (`pointcept/utils/gradient_norm.py`, wired in `pointcept/engines/train.py`)
— not the original GradNorm (Chen et al. 2018), a simpler heuristic that keeps every task's
raw-loss gradient contributing at a similar scale: an EMA of each task's loss gradient norm
w.r.t. the last shared backbone layer (`fp1` in `PyGRandLANetMultiTask`), refreshed every
`grad_norm_lite_interval` steps, used to rescale each task's loss by `1 / max(EMA, eps)` before
summing with the static `task_weights`:

```
train/loss = Σ_t  loss_t · task_weights[t] · (1 / max(EMA_t, grad_norm_lite_eps))
```

Config (`configs/model/multitask_default.yaml`): `grad_norm_lite` (off by default; the
`flair3d_plus/multitask` experiment turns it on), `grad_norm_lite_interval` (default 100),
`grad_norm_lite_ema_alpha` (EMA smoothing, default 0.1), `grad_norm_lite_eps` (default 1e-3),
`grad_norm_lite_task_groups` (the 4 nathab axes share one `nathab` group, matching Pointcept).

Logged keys: `train/grad_norm_lite_scale_{task}` (every step), `train/grad_norm_lite_norm_{task}`
(only on measurement steps). A separate, diagnostic-only toggle `log_task_gradient_norms` (off by
default — costs one extra gradient probe per task **every** step) logs
`train/task_grad_norm_backbone_{task}`, `train/task_grad_norm_head_{task}`, and pairwise
`train/task_grad_cos_{task_a}__{task_b}` backbone-gradient cosine similarities, without affecting
the loss — useful to check whether tasks are fighting each other during training.

---

## Sphinx tutorial

A longer machine-specific walkthrough (Hecate + Jean Zay paths) also lives in `docs/source/tutorials/flair3d_plus.md`.
