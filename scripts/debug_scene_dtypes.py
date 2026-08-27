"""Find the Flair3D+ multitask scene(s) that break PyG collate on Jean Zay.

Symptom (DataLoader worker crash, intermittent):

    RuntimeError: torch.cat(): input types can't be cast to the desired output type Long

= one ``Data`` attribute has a different dtype / ndim on different scenes in the same
batch. PyG sizes the shared-memory output buffer from the *first* scene in the batch, so
whether it crashes depends on shuffle order -- hence the intermittency.

Two modes:

* ``--mode collate`` (default, fast): build the *real* training dataloader (same config,
  transforms, sampler, ``num_workers``) and iterate it. A wrapped collater checks per-key
  dtype/ndim/shape consistency of every mini-batch *before* the fragile ``torch.cat`` and,
  on the first mismatch, prints the offending key + the ``patch_id`` of every scene on
  each side, then stops. Parallel workers + early stop -> minutes, not hours. Detects the
  bug even with ``--num-workers 0`` (does not rely on the buffer-typed cat actually
  raising).

* ``--mode scan``: exhaustively load every scene of a split (optionally through the real
  transform pipeline) and report, per attribute, every (dtype, ndim, trailing-shape)
  seen. Use ``--jobs`` to parallelise. Slower but gives the full picture.

Usage on Jean Zay (env vars exported by run_flair3d_plus_multitask_200k.slurm)::

    python scripts/debug_scene_dtypes.py                       # collate mode, defaults
    python scripts/debug_scene_dtypes.py --epochs 30 --num-workers 8
    python scripts/debug_scene_dtypes.py --mode scan --transform train --jobs 24 --limit 0
"""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from myria3d.pctl.dataset.pointcept_npy import PointceptNpyDataset  # noqa: E402


# --------------------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------------------
def tensor_signature(value):
    """(dtype, ndim, trailing-shape) for a tensor; None for anything else.

    The leading dim (point count) legitimately varies per scene, so it is excluded --
    only a dtype / ndim / trailing-shape difference breaks collate.
    """
    if isinstance(value, torch.Tensor):
        return (str(value.dtype), value.dim(), tuple(value.shape[1:]))
    return None


def batch_key_mismatches(data_list):
    """Return {key: {signature: [patch_id, ...]}} for keys that are not collate-safe.

    Covers both a dtype/ndim/shape split across scenes and a key present on some scenes
    but missing on others.
    """
    present = defaultdict(int)
    sigs = defaultdict(lambda: defaultdict(list))
    for d in data_list:
        pid = getattr(d, "patch_id", "?")
        keys = set(d.keys())
        for key in keys:
            present[key] += 1
            sig = tensor_signature(d[key])
            if sig is not None:
                sigs[key][sig].append(pid)

    bad = {}
    n = len(data_list)
    for key, per_sig in sigs.items():
        split_dtype = len(per_sig) > 1
        missing = present[key] != n
        if split_dtype or missing:
            groups = {sig: pids for sig, pids in per_sig.items()}
            if missing:
                groups["<key absent>"] = [
                    getattr(d, "patch_id", "?") for d in data_list if key not in set(d.keys())
                ]
            bad[key] = groups
    return bad


def format_mismatch_report(bad, header):
    lines = [f"{'=' * 78}", header, f"{'=' * 78}"]
    for key, groups in bad.items():
        lines.append(f"\nkey {key!r} -- {len(groups)} incompatible variants in one batch:")
        for sig, pids in groups.items():
            shown = ", ".join(pids[:8]) + (" ..." if len(pids) > 8 else "")
            lines.append(f"  {str(sig):45s} x{len(pids):<4d}  {shown}")
    return "\n".join(lines)


def compose_cfg(experiment, data_root, csv_manifest, extra_overrides=()):
    from hydra import compose, initialize_config_dir

    overrides = [
        f"experiment={experiment}",
        "logger=csv",
        f"datamodule.data_root={data_root}",
        f"datamodule.csv_manifest={csv_manifest}",
        *extra_overrides,
    ]
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs")):
        return compose(config_name="config", overrides=overrides)


def build_transform(experiment, which, data_root, csv_manifest):
    from hydra.utils import instantiate

    cfg = compose_cfg(experiment, data_root, csv_manifest)
    dm = instantiate(cfg.datamodule)
    return dm.train_transform if which == "train" else dm.eval_transform


# --------------------------------------------------------------------------------------
# mode: collate  (fast -- iterate the real dataloader, stop at first mismatch)
# --------------------------------------------------------------------------------------
def run_collate_mode(args):
    from hydra.utils import instantiate

    import myria3d.pctl.dataloader.dataloader as dl_mod

    cfg = compose_cfg(args.experiment, args.data_root, args.csv_manifest)
    dm = instantiate(cfg.datamodule)
    dm.num_workers = args.num_workers
    if args.batch_size:
        dm.batch_size = args.batch_size
    if args.num_workers == 0:
        dm.prefetch_factor = None
    # Drop the iter-limited sampler: one epoch here = a full deterministic sweep of every
    # train scene (with fresh random subtile crops), so --epochs N covers the dataset N
    # times instead of N random 12k-scene subsets.
    dm.iter_per_epoch = None
    print(f"experiment={args.experiment}  batch_size={dm.batch_size}  "
          f"num_workers={dm.num_workers}")
    print("building scene list / dataset ...")
    dm.setup()
    n_train = len(dm.dataset.traindata)
    print(f"  {n_train} train entries -- one --epoch is a full pass over all of them")

    orig_call = dl_mod.GeometricNoneProofCollater.__call__

    def checked_call(self, data_list):
        clean = [d for d in data_list if d is not None]
        if len(clean) > 1:
            bad = batch_key_mismatches(clean)
            if bad:
                report = format_mismatch_report(
                    bad, "COLLATE-BREAKING MISMATCH (this is the Jean Zay crash)"
                )
                print("\n" + report, file=sys.stderr, flush=True)
                raise RuntimeError("\n" + report)
        return orig_call(self, data_list)

    dl_mod.GeometricNoneProofCollater.__call__ = checked_call

    loader = dm.train_dataloader()
    sampler = getattr(loader, "sampler", None)
    seen_batches = 0
    seen_scenes = 0
    for epoch in range(args.epochs):
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        for batch in loader:
            seen_batches += 1
            seen_scenes += int(getattr(batch, "num_graphs", dm.batch_size))
            if seen_batches % 200 == 0:
                print(f"  epoch {epoch}  {seen_batches} batches  ~{seen_scenes} scenes  ok")
        print(f"epoch {epoch} done -- {seen_batches} batches, ~{seen_scenes} scenes, no mismatch")

    print(f"\nNo collate mismatch in {seen_batches} batches (~{seen_scenes} scene draws). "
          f"Try more --epochs or --mode scan --limit 0.")


# --------------------------------------------------------------------------------------
# mode: scan  (exhaustive, optionally parallel and/or through the transform pipeline)
# --------------------------------------------------------------------------------------
_SCAN = {}


def _scan_init(data_root, csv_manifest, tile_width, subtile_width, transform_mode,
               experiment):
    transform = None
    if transform_mode != "none":
        transform = build_transform(experiment, transform_mode, data_root, csv_manifest)
    _SCAN["ds"] = PointceptNpyDataset(
        data_root=data_root,
        csv_manifest=csv_manifest,
        tile_width=tile_width,
        subtile_width=subtile_width,
        train_transform=transform,
        eval_transform=transform,
    )


def _scan_one(idx):
    ds = _SCAN["ds"]
    try:
        data = ds[idx]
    except Exception as e:  # noqa: BLE001 -- report, keep going
        return (ds._scenes[idx][0], "ERROR", repr(e))
    if data is None:
        return None
    out = {}
    for key in data.keys():
        v = data[key]
        sig = tensor_signature(v)
        if sig is None:
            if isinstance(v, np.ndarray):
                sig = (f"numpy:{v.dtype}", v.ndim, tuple(v.shape[1:]))
            else:
                sig = (type(v).__name__,)
        out[key] = sig
    return (ds._scenes[idx][0], "OK", out)


def run_scan_mode(args):
    from multiprocessing import Pool

    tmp_ds = PointceptNpyDataset(
        data_root=args.data_root,
        csv_manifest=args.csv_manifest,
        tile_width=args.tile_width,
        subtile_width=args.subtile_width,
        train_transform=None,
        eval_transform=None,
    )
    indices = tmp_ds._indices_for_split(args.split)
    if args.limit:
        indices = indices[: args.limit]
    del tmp_ds
    print(f"scanning {len(indices)} {args.split!r} scenes  transform={args.transform}  "
          f"jobs={args.jobs}")

    combos = defaultdict(set)
    present = defaultdict(int)
    examples = defaultdict(lambda: defaultdict(list))
    errors = []

    init_args = (args.data_root, args.csv_manifest, args.tile_width, args.subtile_width,
                 args.transform, args.experiment)

    def handle(res):
        if res is None:
            return
        scene_dir, status, payload = res
        if status == "ERROR":
            errors.append((scene_dir, payload))
            return
        for key, sig in payload.items():
            present[key] += 1
            combos[key].add(sig)
            if len(examples[key][sig]) < 5:
                examples[key][sig].append(scene_dir)

    if args.jobs > 1:
        with Pool(args.jobs, initializer=_scan_init, initargs=init_args) as pool:
            for i, res in enumerate(pool.imap_unordered(_scan_one, indices, chunksize=32)):
                handle(res)
                if (i + 1) % 5000 == 0:
                    print(f"  ... {i + 1}/{len(indices)}")
    else:
        _scan_init(*init_args)
        for i, idx in enumerate(indices):
            handle(_scan_one(idx))
            if (i + 1) % 1000 == 0:
                print(f"  ... {i + 1}/{len(indices)}")

    print("\n=== per-key (dtype, ndim, trailing-shape)  [MISMATCH if >1] ===")
    mismatched = []
    for key in sorted(combos):
        variants = sorted(str(s) for s in combos[key])
        flag = "   <-- MISMATCH" if len(combos[key]) > 1 else ""
        if len(combos[key]) > 1:
            mismatched.append(key)
        print(f"{key:26s} in {present[key]:7d}/{len(indices)}  {variants}{flag}")

    for key in mismatched:
        print(f"\n{key}:")
        for sig in sorted(combos[key], key=str):
            print(f"  {sig}")
            for sd in examples[key][sig]:
                print(f"      {sd}")

    if errors:
        print(f"\n=== {len(errors)} scenes raised while loading ===")
        for sd, err in errors[:20]:
            print(f"  {sd}\n    {err}")

    if not mismatched and not errors:
        print("\nno mismatch found. try --transform train, a bigger --limit, or --mode collate.")


# --------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mode", choices=["collate", "scan"], default="collate")
    p.add_argument("--data-root", default=os.environ.get("FLAIR3D_DATA_ROOT"))
    p.add_argument("--csv-manifest", default=os.environ.get("FLAIR3D_CSV_MANIFEST"))
    p.add_argument("--experiment", default="flair3d_plus/multitask_200k")
    p.add_argument("--tile-width", type=float, default=100)
    p.add_argument("--subtile-width", type=float, default=50)
    # collate mode
    p.add_argument("--epochs", type=int, default=3,
                   help="[collate] full passes over the train split (fresh random crops each)")
    p.add_argument("--num-workers", type=int, default=8, help="[collate] dataloader workers")
    p.add_argument("--batch-size", type=int, default=0, help="[collate] 0 = config default")
    # scan mode
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--transform", default="none", choices=["none", "train", "eval"],
                   help="[scan] apply the real transform pipeline before comparing")
    p.add_argument("--limit", type=int, default=0, help="[scan] max scenes (0 = all)")
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                   help="[scan] parallel worker processes")
    args = p.parse_args()

    if not args.data_root or not args.csv_manifest:
        p.error("set --data-root/--csv-manifest or FLAIR3D_DATA_ROOT / FLAIR3D_CSV_MANIFEST")

    if args.mode == "collate":
        run_collate_mode(args)
    else:
        run_scan_mode(args)


if __name__ == "__main__":
    main()
