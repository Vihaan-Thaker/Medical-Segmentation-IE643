#!/usr/bin/env python
"""Arm-2 data: AMOS train/val slices exactly as arm 1 reads them (ARM2_CODE_PLAN.md §1a, §3.4).

* Same files: data/processed/manifest.csv + the stored .npz (img uint8, msk 0/1/2). Nothing
  is re-windowed, resampled, flipped or augmented; the uint8 image reaches the model as stored.
* Same kidneys: per side, only if the manifest has a box for that side (>= 50 px).
* Validation boxes: sam_metrics.jitter_box(clean, level, h, w, jitter_seed(case, z, side,
  level)) -- the evaluation's own seeding, so validation sees exactly the test's prompts.
* Training boxes: level ~ U[0, 0.20] per kidney, seeded per (seed, epoch, draw index), so the
  r4 and r8 runs see identical prompts regardless of worker count or batch size.
* Epoch draws are made explicitly in the main process as (draw index, slice index) pairs:
  50/50 CT/MRI in expectation, with replacement.
* Single-modality teachers (D70-D71): a dataset restricted to one modality draws uniformly
  over its slices instead; the joint (both-modality) path is unchanged.
"""
from __future__ import annotations

import csv
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import eval_sam as E
import sam_metrics as M
from sam_train_core import prepare_image

TRAIN_SPLITS = ("train", "val")
# D33 counts of slices with >= 1 kidney box (prep job 329158, "boxed" column).
EXPECTED_SLICES = {"train": {"ct": 5760, "mri": 2535}, "val": {"ct": 1901, "mri": 843}}
VAL_LEVELS = (0.0, 0.10)
TRAIN_MAX_LEVEL = 0.20
MODALITIES = ("ct", "mri")


def check_split_integrity(processed: Path) -> dict:
    """Every patient belongs to exactly one split (D10). Read straight from the raw manifest,
    independently of load_manifest. Returns {split: n_patients}."""
    with open(Path(processed) / "manifest.csv") as fh:
        rows = list(csv.DictReader(fh))
    splits_of: dict = {}
    for r in rows:
        splits_of.setdefault(r["patient"], set()).add(r["split"])
    multi = {p: sorted(s) for p, s in splits_of.items() if len(s) > 1}
    if multi:
        raise SystemExit(f"patients in more than one split (leak): {dict(list(multi.items())[:5])}")
    return dict(Counter(next(iter(s)) for s in splits_of.values()))


def manifest_sha256(processed: Path) -> str:
    h = hashlib.sha256()
    with open(Path(processed) / "manifest.csv", "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def kidney_sides(row: dict) -> list[tuple[int, str]]:
    """(side, prefix) for each side that has a box, in eval order (right, then left)."""
    return [(side, prefix) for side, prefix in E.SIDES if row[f"{prefix}_box"] is not None]


class SliceDataset:
    """The boxed slices of one split, sorted by (case, z). Loads one slice on demand.
    modality=None keeps both modalities (arm 2); "ct" / "mri" keeps that one only (D71)."""

    def __init__(self, processed: Path, split: str, expected: dict | None = EXPECTED_SLICES,
                 modality: str | None = None):
        if split not in TRAIN_SPLITS:
            raise ValueError(f"arm-2 data may only use {TRAIN_SPLITS}, not {split!r}")
        if modality is not None and modality not in MODALITIES:
            raise ValueError(f"modality must be None or one of {MODALITIES}, not {modality!r}")
        self.processed = Path(processed)
        self.split = split
        self.modality = modality
        rows = E.load_manifest(self.processed, splits=(split,))
        bad = [r for r in rows if r["split"] != split]
        if bad:
            raise AssertionError(f"load_manifest returned foreign split rows: {bad[0]['split']}")
        rows = [r for r in rows if r["sam_eligible"]]
        rows.sort(key=lambda r: (r["case"], r["z"]))
        for r in rows:
            if not kidney_sides(r):
                raise AssertionError(f"{r['case']} z={r['z']}: sam_eligible but no side has a box")
        if modality is not None:
            rows = [r for r in rows if r["modality"] == modality]
        counts = dict(Counter(r["modality"] for r in rows))
        if modality is not None and set(counts) != {modality}:
            raise SystemExit(f"{split} restricted to {modality}: modalities present {counts}")
        want = None if expected is None else (
            expected[split] if modality is None else {modality: expected[split][modality]})
        if want is not None and counts != want:
            raise SystemExit(f"{split}: boxed slices per modality {counts} != expected "
                             f"{want} (D33)")
        self.rows = rows
        self.counts = counts

    def __len__(self) -> int:
        return len(self.rows)

    def n_kidneys(self, idx: int) -> int:
        return len(kidney_sides(self.rows[idx]))

    def load(self, idx: int) -> dict:
        r = self.rows[idx]
        with np.load(self.processed / r["path"]) as d:
            img, msk = d["img"], d["msk"]
        if img.dtype != np.uint8 or img.shape != (r["h"], r["w"]) or msk.shape != img.shape:
            raise AssertionError(f"{r['case']} z={r['z']}: stored slice {img.dtype} {img.shape} "
                                 f"does not match the manifest ({r['h']}, {r['w']})")
        kidneys = []
        for side, prefix in kidney_sides(r):
            gt = msk == side
            if int(gt.sum()) != r[f"{prefix}_area"] or not gt.any():
                raise AssertionError(f"{r['case']} z={r['z']} {prefix}: GT area {int(gt.sum())} "
                                     f"!= manifest {r[f'{prefix}_area']}")
            kidneys.append({"side": side, "clean": tuple(int(v) for v in r[f"{prefix}_box"]),
                            "gt": gt})
        return {"img": img, "kidneys": kidneys, "case": r["case"], "patient": r["patient"],
                "modality": r["modality"], "z": r["z"], "h": r["h"], "w": r["w"],
                "sp_row": r["sp_row"], "sp_col": r["sp_col"]}


def make_epoch_draws(ds: SliceDataset, epoch: int, seed: int, n_draws: int | None = None) -> list:
    """This epoch's training draws as [(i, slice_index)]: with replacement, weight 1/n_ct per
    CT slice and 1/n_mri per MRI slice (50/50 in expectation). A single-modality dataset
    (D71) draws uniformly over its slices. Deterministic in (seed, epoch)."""
    n_draws = len(ds) if n_draws is None else n_draws
    n_mod = Counter(r["modality"] for r in ds.rows)
    single = getattr(ds, "modality", None)
    if single is None:
        if set(n_mod) != {"ct", "mri"}:
            raise ValueError(f"need both modalities to balance, have {dict(n_mod)}")
        w = torch.tensor([1.0 / n_mod[r["modality"]] for r in ds.rows], dtype=torch.float64)
    else:
        if set(n_mod) != {single}:
            raise ValueError(f"{single} dataset holds {dict(n_mod)}")
        w = torch.ones(len(ds.rows), dtype=torch.float64)
    g = torch.Generator().manual_seed(int(seed) + int(epoch))
    idx = torch.multinomial(w, n_draws, replacement=True, generator=g).tolist()
    return list(enumerate(idx))


def draw_rng(seed: int, epoch: int, i: int) -> np.random.Generator:
    key = hashlib.sha256(f"{seed}|{epoch}|{i}".encode()).digest()
    return np.random.default_rng(int.from_bytes(key[:8], "big"))


class TrainDraws(Dataset):
    """Training items for one epoch, keyed by draw position k -> (i, slice index)."""

    def __init__(self, base: SliceDataset, draws: list, epoch: int, seed: int, target: int = 1024,
                 max_level: float = TRAIN_MAX_LEVEL):
        self.base, self.draws, self.epoch, self.seed = base, draws, epoch, seed
        self.target, self.max_level = target, max_level

    def __len__(self) -> int:
        return len(self.draws)

    def boxes_for(self, i: int, item: dict) -> list[tuple]:
        rng = draw_rng(self.seed, self.epoch, i)
        out = []
        for k in item["kidneys"]:
            level = float(rng.uniform(0.0, self.max_level))
            jseed = int(rng.integers(0, 2 ** 62))
            out.append(M.jitter_box(k["clean"], level, item["h"], item["w"], jseed))
        return out

    def __getitem__(self, k: int) -> dict:
        i, idx = self.draws[k]
        item = self.base.load(idx)
        t, in_size, orig = prepare_image(item["img"], self.target)
        return {"draw": i, "slice_index": idx, "image": t, "input_size": in_size,
                "original_size": orig, "boxes": self.boxes_for(i, item),
                "gts": [torch.from_numpy(kd["gt"]) for kd in item["kidneys"]],
                "modality": item["modality"], "case": item["case"], "z": item["z"]}


class ValItems(Dataset):
    """Validation slices with the evaluation's deterministic boxes at VAL_LEVELS."""

    def __init__(self, base: SliceDataset, indices: list[int] | None = None, target: int = 1024,
                 levels: tuple = VAL_LEVELS):
        self.base = base
        self.indices = list(range(len(base))) if indices is None else list(indices)
        self.target, self.levels = target, tuple(levels)

    def __len__(self) -> int:
        return len(self.indices)

    @staticmethod
    def val_box(item: dict, kidney: dict, level: float) -> tuple:
        return M.jitter_box(kidney["clean"], level, item["h"], item["w"],
                            M.jitter_seed(item["case"], int(item["z"]), int(kidney["side"]), level))

    def __getitem__(self, k: int) -> dict:
        item = self.base.load(self.indices[k])
        t, in_size, orig = prepare_image(item["img"], self.target)
        return {"image": t, "input_size": in_size, "original_size": orig,
                "boxes": {lev: [self.val_box(item, kd, lev) for kd in item["kidneys"]]
                          for lev in self.levels},
                "gts": [kd["gt"] for kd in item["kidneys"]],
                "sides": [kd["side"] for kd in item["kidneys"]],
                "modality": item["modality"], "patient": item["patient"], "case": item["case"],
                "z": item["z"], "sp": (item["sp_row"], item["sp_col"])}


def identity_subset(base: SliceDataset, step: int = 14) -> list[int]:
    """Fixed val subset for the identity and merge checks: every 14th slice of the (case, z)
    sorted list (196 slices for the 2,744 val slices)."""
    return list(range(0, len(base), step))


def collate_list(batch: list) -> list:
    """No stacking: slices differ in size; padding happens per image in to_model_input."""
    return batch


def worker_init(_worker_id: int) -> None:
    torch.set_num_threads(1)
