#!/usr/bin/env python
"""Arm-4 teacher maps: on-disk format, save, load and verify. No SAM import, so the arm-4
student training can read the maps without depending on SAM (plan invariant as arm 3).

Layout of a teacher folder (written by teacher_maps.py):
    teacher.json                 identity stamp: teacher checkpoint sha256, manifest sha256, format
    slices/<case>/<z:04d>.npz    one file per boxed training slice:
        logit_1 / logit_2        float16 h x w: SAM's pre-threshold mask logits at the slice's
                                 native size, for the right (1) / left (2) kidney -- only the
                                 sides that have a manifest box (>= 50 px)
        meta                     JSON (0-d unicode array): case, z, h, w, sides, boxes, SAM
                                 iou_pred, Dice of (logit > 0) vs ground truth, fp16 sign flips
    teacher_manifest.csv         one row per slice, built by re-reading every file
    run.json                     status TEACHER_OK + counts, QC and timings

Exactly what arm 2's evaluation computed: SamPredictor.set_image on the uint8 slice repeated
to 3 channels, predict(box=clean box, multimask_output=False, return_logits=True). The
evaluation's mask is (logit > 0); the logit itself is kept so arm 4 can soften it.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

FORMAT_VERSION = 1
STAMP = "teacher.json"
SIDE_KEYS = {1: "logit_1", 2: "logit_2"}         # mask value -> array name (1 right, 2 left)
FP16_MAX = 60000.0                                # |logit| must stay inside float16 range
MANIFEST_FIELDS = ["case", "patient", "modality", "z", "h", "w", "path", "sides",
                   "box_1", "box_2", "iou_1", "iou_2", "dice_1", "dice_2", "fp16_flips"]


def slice_path(out: Path, case: str, z: int) -> Path:
    return Path(out) / "slices" / str(case) / f"{int(z):04d}.npz"


def rel_path(case: str, z: int) -> str:
    return f"slices/{case}/{int(z):04d}.npz"


def save_slice(out: Path, meta: dict, logits: dict) -> Path:
    """Atomic write of one slice: {side: float32 h x w logits} -> float16 arrays + meta.
    Refuses non-finite or out-of-range logits and sides that are not 1/2."""
    h, w = int(meta["h"]), int(meta["w"])
    arrays = {}
    for side, lg in logits.items():
        if side not in SIDE_KEYS:
            raise ValueError(f"side {side!r} not in {list(SIDE_KEYS)}")
        lg = np.asarray(lg)
        if lg.shape != (h, w):
            raise ValueError(f"{meta['case']} z={meta['z']} side {side}: logits {lg.shape} != {(h, w)}")
        if not np.isfinite(lg).all():
            raise ValueError(f"{meta['case']} z={meta['z']} side {side}: non-finite logits")
        if float(np.abs(lg).max()) >= FP16_MAX:
            raise ValueError(f"{meta['case']} z={meta['z']} side {side}: |logit| beyond float16 range")
        arrays[SIDE_KEYS[side]] = lg.astype(np.float16)
    if sorted(logits) != sorted(meta["sides"]):
        raise ValueError(f"{meta['case']} z={meta['z']}: logits for {sorted(logits)} but meta sides {meta['sides']}")
    path = slice_path(out, meta["case"], meta["z"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, meta=np.array(json.dumps(meta, sort_keys=True)), **arrays)
    with open(tmp, "rb+") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path


def load_slice(out: Path, case: str, z: int) -> tuple[dict, dict]:
    """-> (meta, {side: float16 h x w logits}). Raises on any inconsistency."""
    path = slice_path(out, case, z)
    with np.load(path, allow_pickle=False) as d:
        meta = json.loads(str(d["meta"]))
        logits = {side: d[key] for side, key in SIDE_KEYS.items() if key in d.files}
    if meta.get("case") != case or int(meta.get("z", -1)) != int(z):
        raise ValueError(f"{path}: meta names {meta.get('case')} z={meta.get('z')}")
    if sorted(logits) != sorted(meta["sides"]):
        raise ValueError(f"{path}: arrays for sides {sorted(logits)}, meta says {meta['sides']}")
    for side, lg in logits.items():
        if lg.dtype != np.float16 or lg.shape != (meta["h"], meta["w"]):
            raise ValueError(f"{path}: side {side} array {lg.dtype} {lg.shape}")
    return meta, logits


def expected_meta(row: dict, boxes: dict, identity: dict) -> dict:
    """The identity part of a slice's meta, derived from the manifest row and the run stamp:
    a stored file is valid only if all of these match."""
    sides = [s for s in (1, 2) if boxes.get(s) is not None]
    return {"case": row["case"], "z": int(row["z"]), "h": int(row["h"]), "w": int(row["w"]),
            "sides": sides, "boxes": {str(s): list(boxes[s]) for s in sides},
            "teacher_sha256": identity["teacher_sha256"],
            "manifest_sha256": identity["manifest_sha256"], "format": FORMAT_VERSION}


def meta_matches(meta: dict, want: dict) -> bool:
    return all(meta.get(k) == v for k, v in want.items())


def read_stamp(out: Path) -> dict | None:
    p = Path(out) / STAMP
    return json.loads(p.read_text()) if p.exists() else None
