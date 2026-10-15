#!/usr/bin/env python
"""Arm 1 — off-the-shelf SAM on the IE643 kidney evaluation set (DECISIONS.md D24/D24a/D26).

Runs SAM with ground-truth-derived box prompts over the test and external splits, scoring
every kidney instance with Dice and NSD. One SLURM job covers all four test sets.

What it does, per slice:

  * encode ONCE with `set_image`, then decode once per kidney per jitter level. The
    encoder is 632M parameters and the decoder ~4M, so decoding again is nearly free:
    this makes the whole box-perturbation sweep (D12) a by-product of arm 1 rather than a
    separate pass. Encoding per kidney instead would be 13,145 encodes against 7,386 --
    1.78x the cost for identical output.
  * jitter the box at 0 / 5 / 10 / 20 % of its side length (10 % is SAM's own recipe),
    deterministically seeded per instance so results are reproducible under resume.

Every case's rows are flushed as soon as it finishes, and `--resume` keeps only cases whose
instance AND volume rows are complete and unique, rewriting the CSVs (with a backup) if a
partial case is found -- so a crashed or timed-out job can always be continued without
losing or duplicating a row. The final row counts are then re-verified from the files on
disk, for fresh and resumed runs alike.

QC overlays are generated AFTER scoring, from `instances.csv`: the worst and median
instances per group are re-run through SAM with their stored jittered box. This survives
resume (every group gets overlays, whichever job scored it) and doubles as a
reproducibility check -- the re-run Dice must match the stored Dice.

Outputs (under --out):
  instances.csv      one row per (kidney instance x jitter level): 2D Dice + NSD
  volumes_3d.csv     one row per (case, side, jitter): 3D Dice + NSD over the stacked volume
  spacing_report.csv in-plane spacing per dataset -- fixes the headline NSD tolerance
  run.json           config, versions, per-stage timing, throughput, self-checks
  qc/                overlay PNGs, deliberately including the worst cases

Usage
-----
    python eval_sam.py --processed .../data/processed \
                       --checkpoint .../checkpoints/sam_vit_h_4b8939.pth \
                       --out .../outputs/arm1
    --self-test   verify the metrics and exit (seconds)
    --limit N     N cases per group, for a smoke run
    --resume      continue an interrupted run (a fresh run over existing results is
                  refused unless --overwrite)

Safety checks (D28): at start-up, before any GPU time, the evaluation set must match the
D22a counts per group and every case must be a full z-stack. Every stored row -- kept on
resume, inherited by a shard, merged, or final -- is re-derived against the manifest
(slice, kidney side, clean box, jittered box from its seed, GT area, identity, spacing,
score ranges). Damaged CSVs (cut-off last line, empty, missing file) are repaired by
redoing the affected case; a CSV with different columns is refused outright.

Parallel (D27): N jobs each take `--shard K --num-shards N` (K = 0..N-1) and write only to
--out/shards/K_of_N/, so no two jobs ever share a file. Each shard skips cases already
complete in --out, takes a slice-balanced share of the rest, and resumes itself when
resubmitted. Then one `--merge N` job combines --out + shards, re-verifies every count and
writes the QC overlays.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sam_metrics as M  # noqa: E402

EVAL_SPLITS = ("test", "external")
# D22a anchors, counted independently from manifest.csv on Prajna (2026-10-06):
# (dataset, modality, sequence) -> (cases, scored slices, kidney instances). Checked at
# start-up, before any GPU time, and the instance total again on the final files.
# Skipped only under --limit.
EXPECTED_GROUPS = {
    ("amos", "ct", ""): (59, 1870, 3315),
    ("amos", "mri", ""): (12, 691, 1233),
    ("chaos", "mri", "t1in"): (20, 297, 526),
    ("chaos", "mri", "t1out"): (20, 297, 526),
    ("chaos", "mri", "t2"): (20, 284, 506),
    ("word", "ct", ""): (100, 3947, 7039),
}                                    # totals: 231 cases, 7,386 slices, 13,145 instances
SIDES = ((M.RIGHT, "rk"), (M.LEFT, "lk"))
STAGES = ("io", "encode", "decode", "metric2d", "metric3d")

INSTANCE_FIELDS = [
    "dataset", "patient", "case", "sequence", "modality", "split", "z", "side",
    "jitter", "box_clean", "box_jit", "box_iou", "gt_area", "pred_area",
    "dice", *[f"nsd_{t:g}" for t in M.TOLERANCES_MM], "sam_iou", "sp_row", "sp_col",
]
VOLUME_FIELDS = [
    "dataset", "patient", "case", "sequence", "modality", "split", "side", "jitter",
    "dice3d", *[f"nsd3d_{t:g}" for t in M.TOLERANCES_MM],
    "gt_vox", "pred_vox", "sliver_vox", "sliver_frac", "n_slices", "n_scored",
    "sp_row", "sp_col", "sp_slice",
]


# ------------------------------------------------------------------------- model
def build_predictor(checkpoint: Path, model_type: str, device: str):
    """Real SAM. Separated so the test suite can substitute a stub and still exercise
    every line of the surrounding loop."""
    import torch
    from segment_anything import SamPredictor, sam_model_registry

    # Match CPU threads to the SLURM allocation; the default is every core on the node,
    # which oversubscribes the 8 CPUs we were given. GPU numerics are unaffected.
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "8"))))
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
    sam.to(device=device)
    sam.eval()
    torch.set_grad_enabled(False)
    return SamPredictor(sam)


_SYNC = None


def _cuda_sync() -> None:
    """Block until queued GPU work finishes, so the encode/decode split in the timings is
    real. `set_image` returns before the encoder finishes; without this its cost would be
    charged to the first decode. Adds no work -- the first decode would wait anyway."""
    global _SYNC
    if _SYNC is None:
        try:
            import torch
            _SYNC = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
        except Exception:
            _SYNC = lambda: None  # noqa: E731
    _SYNC()


# -------------------------------------------------------------------- manifest IO
def load_manifest(processed: Path) -> list[dict]:
    with open(processed / "manifest.csv") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"{processed / 'manifest.csv'} is empty")
    out = []
    for r in rows:
        if r["split"] not in EVAL_SPLITS:
            continue
        r["z"] = int(r["z"])
        r["n_slices"] = int(r["n_slices"])
        r["rk_area"] = int(r["rk_area"])
        r["lk_area"] = int(r["lk_area"])
        r["sam_eligible"] = int(r["sam_eligible"])
        r["h"], r["w"] = int(r["h"]), int(r["w"])
        for k in ("sp_row", "sp_col", "sp_slice"):
            r[k] = float(r[k])
        r["rk_box"] = json.loads(r["rk_box"])
        r["lk_box"] = json.loads(r["lk_box"])
        # A degenerate or out-of-bounds box would silently give SAM a meaningless prompt.
        for prefix in ("rk", "lk"):
            b = r[f"{prefix}_box"]
            if b is None:
                continue
            x0, y0, x1, y1 = b
            if not (0 <= x0 < x1 < r["w"] and 0 <= y0 < y1 < r["h"]):
                raise SystemExit(f"invalid {prefix}_box {b} for {r['case']} z={r['z']} "
                                 f"(image {r['h']}x{r['w']})")
            if r[f"{prefix}_area"] <= 0:
                raise SystemExit(f"{r['case']} z={r['z']}: {prefix} has a box but zero area")
        out.append(r)
    return out


def group_cases(rows: list[dict]) -> dict[str, list[dict]]:
    cases: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        cases[r["case"]].append(r)
    for c in cases.values():
        c.sort(key=lambda r: r["z"])
    return dict(cases)


def expected_rows(case_rows: list[dict]) -> tuple[int, int]:
    """(instance rows, volume rows) a complete case must have, derived from the manifest."""
    n_inst = sum((r["rk_box"] is not None) + (r["lk_box"] is not None)
                 for r in case_rows) * len(M.JITTER_LEVELS)
    n_sides = sum(any(r[f"{p}_area"] > 0 for r in case_rows) for _, p in SIDES)
    return n_inst, n_sides * len(M.JITTER_LEVELS)


def expected_keys(case: str, case_rows: list[dict]) -> tuple[set, set]:
    """Exactly which (instance, volume) keys a complete case has -- stronger than a count."""
    ki, kv = set(), set()
    for j in M.JITTER_LEVELS:
        jk = f"{j:.4f}"
        for side, prefix in SIDES:
            name = M.SIDE_NAME[side]
            for r in case_rows:
                if r[f"{prefix}_box"] is not None:
                    ki.add((case, str(r["z"]), name, jk))
            if any(r[f"{prefix}_area"] > 0 for r in case_rows):
                kv.add((case, name, jk))
    return ki, kv


def inst_key(r: dict) -> tuple:
    return (r["case"], str(r["z"]), r["side"], f"{float(r['jitter']):.4f}")


def vol_key(r: dict) -> tuple:
    return (r["case"], r["side"], f"{float(r['jitter']):.4f}")


def read_results(path: Path, fields: list[str]) -> tuple[list[dict], list[str]]:
    """Rows of a results CSV, plus its defects. Tolerates what a killed job can leave
    behind -- an empty file, or a last line cut off mid-write (no trailing newline) -- by
    dropping the damaged row, so its case reads as incomplete and is redone. A header that
    differs from this code's columns is NOT tolerated: those rows came from a different
    version of the code and must not be mixed in or rewritten."""
    if not path.exists():
        return [], []
    with open(path, newline="") as fh:
        text = fh.read()
    if not text:
        return [], [f"{path.name}: empty file"]
    defects = []
    lines = text.splitlines(keepends=True)
    if not lines[-1].endswith(("\n", "\r")):
        defects.append(f"{path.name}: last line cut off mid-write; dropped")
        lines = lines[:-1]
    rdr = csv.reader(lines)
    header = next(rdr, None)
    if header is None:
        return [], defects + [f"{path.name}: no header"]
    if header != fields:
        raise SystemExit(f"{path}: columns differ from this code's -- written by another "
                         f"version? Refusing to mix or rewrite it.\n  file: {header}\n"
                         f"  code: {fields}")
    rows, n_bad = [], 0
    for rec in rdr:
        if len(rec) != len(fields):
            n_bad += 1
            continue
        rows.append(dict(zip(fields, rec)))
    if n_bad:
        defects.append(f"{path.name}: {n_bad} malformed rows dropped")
    return rows, defects


def audit_outputs(inst_path: Path, vol_path: Path, cases: dict, scope: list[str]) -> dict:
    """Per-case completeness of the CSVs on disk. A case is complete only when BOTH files
    hold exactly its expected keys, each once. `clean` means the files can be appended to
    as they are: both exist, nothing damaged, partial or out of scope."""
    inst, d_i = read_results(inst_path, INSTANCE_FIELDS)
    vols, d_v = read_results(vol_path, VOLUME_FIELDS)
    ci, cv = defaultdict(list), defaultdict(list)
    for r in inst:
        ci[r["case"]].append(inst_key(r))
    for r in vols:
        cv[r["case"]].append(vol_key(r))
    complete, partial = set(), {}
    for c in scope:
        want_i, want_v = expected_keys(c, cases[c])
        ki, kv = ci.get(c, []), cv.get(c, [])
        ok = (len(ki) == len(want_i) and set(ki) == want_i and
              len(kv) == len(want_v) and set(kv) == want_v)
        if ok:
            complete.add(c)
        elif ki or kv:
            partial[c] = (len(ki), len(want_i), len(kv), len(want_v))
    stray = sorted(({r["case"] for r in inst} | {r["case"] for r in vols}) - set(scope))
    defects = d_i + d_v
    clean = (inst_path.exists() and vol_path.exists() and not defects and not partial
             and not stray)
    return {"inst": inst, "vols": vols, "complete": complete, "partial": partial,
            "stray": stray, "defects": defects, "clean": clean}


def count_groups(cases: dict, scope: list[str]) -> dict:
    """(dataset, modality, sequence) -> (cases, scored slices, kidney instances)."""
    out: dict = defaultdict(lambda: [0, 0, 0])
    for c in scope:
        m = cases[c][0]
        g = out[(m["dataset"], m["modality"], m["sequence"])]
        g[0] += 1
        g[1] += n_encode_slices(cases[c])
        g[2] += expected_rows(cases[c])[0] // len(M.JITTER_LEVELS)
    return {k: tuple(v) for k, v in out.items()}


def expected_total_instances() -> int:
    return sum(v[2] for v in EXPECTED_GROUPS.values())


def check_case_layout(case: str, rows: list[dict]) -> list[str]:
    """process_case stacks the slices into a volume: that needs every z from 0 to
    n_slices-1 exactly once (test/external are stored as full volumes, D20) and one
    geometry and identity for the whole case."""
    bad = []
    zs = [r["z"] for r in rows]
    if zs != list(range(len(rows))) or any(r["n_slices"] != len(rows) for r in rows):
        bad.append(f"{case}: slices are not z = 0..{rows[0]['n_slices'] - 1} exactly once "
                   f"(have {len(rows)}, z {zs[0]}..{zs[-1]})")
    for k in ("h", "w", "sp_row", "sp_col", "sp_slice", "dataset", "patient", "modality",
              "sequence", "split"):
        if len({r[k] for r in rows}) != 1:
            bad.append(f"{case}: '{k}' differs between slices")
    return bad


def verify_rows(inst: list[dict], vols: list[dict], cases: dict, manifest_by_key: dict) -> list[str]:
    """Check every stored row against the manifest, independently of how it was produced:
    the right slice and kidney (clean box, GT area), the jittered box re-derived from its
    seed, identity fields, spacing, and metric ranges. A row count cannot see rows from a
    different code version, a z or left/right misalignment, or a corrupted merge; this can.
    Returns problems (empty = all rows consistent)."""
    bad: list[str] = []
    levels = {f"{j:.4f}": j for j in M.JITTER_LEVELS}
    side_of = {v: k for k, v in M.SIDE_NAME.items()}
    nsd2 = [f"nsd_{t:g}" for t in M.TOLERANCES_MM]
    nsd3 = [f"nsd3d_{t:g}" for t in M.TOLERANCES_MM]

    def scores_ok(vals: list[float]) -> bool:
        nsd = vals[1:]
        return (all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in vals)
                and all(b >= a - 1e-9 for a, b in zip(nsd, nsd[1:])))   # NSD grows with tolerance

    for r in inst:
        tag = f"{r.get('case')} z={r.get('z')} {r.get('side')} jitter={r.get('jitter')}"
        try:
            m = manifest_by_key.get((r["case"], int(r["z"])))
            j = levels.get(f"{float(r['jitter']):.4f}")
            side = side_of.get(r["side"])
            if m is None or j is None or side is None:
                bad.append(f"{tag}: not an evaluation instance")
                continue
            prefix = "rk" if side == M.RIGHT else "lk"
            box = m[f"{prefix}_box"]
            if box is None:
                bad.append(f"{tag}: the manifest has no box here")
                continue
            if json.loads(r["box_clean"]) != list(box):
                bad.append(f"{tag}: box_clean {r['box_clean']} != manifest {box}")
            want = list(M.jitter_box(tuple(box), j, m["h"], m["w"],
                                     M.jitter_seed(r["case"], m["z"], side, j)))
            if json.loads(r["box_jit"]) != want:
                bad.append(f"{tag}: box_jit {r['box_jit']} != re-derived {want}")
            if int(r["gt_area"]) != m[f"{prefix}_area"]:
                bad.append(f"{tag}: gt_area {r['gt_area']} != manifest {m[f'{prefix}_area']}")
            for k in ("dataset", "patient", "modality", "sequence", "split"):
                if r[k] != m[k]:
                    bad.append(f"{tag}: {k} '{r[k]}' != manifest '{m[k]}'")
            if abs(float(r["sp_row"]) - m["sp_row"]) > 1e-9 or abs(float(r["sp_col"]) - m["sp_col"]) > 1e-9:
                bad.append(f"{tag}: spacing differs from the manifest")
            vals = [float(r["dice"])] + [float(r[c]) for c in nsd2]
            if not scores_ok(vals):
                bad.append(f"{tag}: scores out of range or NSD not increasing: {vals}")
            if int(r["pred_area"]) == 0 and any(v != 0.0 for v in vals):
                bad.append(f"{tag}: empty prediction with non-zero score")
        except (KeyError, ValueError, TypeError) as e:
            bad.append(f"{tag}: unreadable row ({type(e).__name__}: {e})")

    for r in vols:
        tag = f"{r.get('case')} {r.get('side')} jitter={r.get('jitter')} (3D)"
        try:
            rows = cases.get(r["case"])
            side = side_of.get(r["side"])
            if rows is None or side is None or f"{float(r['jitter']):.4f}" not in levels:
                bad.append(f"{tag}: not an evaluation volume")
                continue
            prefix = "rk" if side == M.RIGHT else "lk"
            gt_vox = sum(x[f"{prefix}_area"] for x in rows)
            sliver = sum(x[f"{prefix}_area"] for x in rows if x[f"{prefix}_box"] is None)
            if int(r["gt_vox"]) != gt_vox:
                bad.append(f"{tag}: gt_vox {r['gt_vox']} != manifest {gt_vox}")
            if int(r["sliver_vox"]) != sliver:
                bad.append(f"{tag}: sliver_vox {r['sliver_vox']} != manifest {sliver}")
            if int(r["n_slices"]) != len(rows):
                bad.append(f"{tag}: n_slices {r['n_slices']} != manifest {len(rows)}")
            vals = [float(r["dice3d"])] + [float(r[c]) for c in nsd3]
            if not scores_ok(vals):
                bad.append(f"{tag}: scores out of range or NSD not increasing: {vals}")
        except (KeyError, ValueError, TypeError) as e:
            bad.append(f"{tag}: unreadable row ({type(e).__name__}: {e})")
    return bad


def summarise_problems(label: str, bad: list[str], k: int = 8) -> list[str]:
    if not bad:
        return []
    return [f"{label}: {len(bad)} inconsistent rows, e.g. " + " | ".join(bad[:k])]


def spacing_report(rows: list[dict]) -> list[dict]:
    """In-plane spacing per dataset. D24a/C5: a tolerance below the pixel size cannot be
    resolved, so the headline NSD tolerance must be >= the coarsest pixel here."""
    by: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        by[(r["dataset"], r["modality"])].extend([r["sp_row"], r["sp_col"]])
    out = []
    for (ds, mod), vals in sorted(by.items()):
        a = np.asarray(vals)
        out.append({"dataset": ds, "modality": mod, "n_slices": len(vals) // 2,
                    "sp_min": round(float(a.min()), 4),
                    "sp_median": round(float(np.median(a)), 4),
                    "sp_max": round(float(a.max()), 4),
                    "nsd1_subpixel": int(a.max() > 1.0),
                    "nsd2_subpixel": int(a.max() > 2.0)})
    return out


# ------------------------------------------------------------------------- QC image
def save_overlay(img: np.ndarray, gt: np.ndarray, pred: np.ndarray,
                 box: tuple[int, int, int, int], path: Path, title: str) -> None:
    from PIL import Image, ImageDraw
    rgb = np.repeat(img[..., None], 3, axis=2).astype(np.float32)
    tp = np.logical_and(gt, pred)
    fn = np.logical_and(gt, ~pred)
    fp = np.logical_and(~gt, pred)
    for sel, colour in ((tp, (60, 220, 60)), (fn, (230, 60, 60)), (fp, (70, 130, 255))):
        rgb[sel] = 0.45 * rgb[sel] + 0.55 * np.array(colour, np.float32)
    im = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    scale = max(1, int(round(512 / max(im.size))))
    if scale > 1:
        im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    d = ImageDraw.Draw(im)
    x0, y0, x1, y1 = (v * scale for v in box)
    d.rectangle([x0, y0, x1 + scale - 1, y1 + scale - 1], outline=(255, 230, 0), width=2)
    d.text((4, 4), title, fill=(255, 230, 0))
    d.text((4, im.height - 24), "green=correct  red=missed  blue=false", fill=(255, 230, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def write_qc_from_csv(inst_rows: list[dict], manifest_by_key: dict, processed: Path,
                      predictor, out: Path, k: int) -> tuple[int, float, list[str]]:
    """Re-run SAM on the worst and median instances per group (headline jitter), using the
    exact jittered box stored in instances.csv, and write overlays. Returns (n written,
    max |Dice difference| vs the stored value, problems). The Dice comparison checks that a
    resumed run is consistent with the job that originally scored each instance."""
    by: dict[str, list[dict]] = defaultdict(list)
    for r in inst_rows:
        if abs(float(r["jitter"]) - M.HEADLINE_JITTER) < 1e-9:
            by["_".join(x for x in (r["dataset"], r["modality"], r["sequence"]) if x)].append(r)
    n, max_diff, problems = 0, 0.0, []
    for group, rs in sorted(by.items()):
        rs.sort(key=lambda r: float(r["dice"]))
        mid = len(rs) // 2
        picks = [("worst", r) for r in rs[:k // 2]] + \
                [("median", r) for r in rs[mid: mid + (k - k // 2)]]
        for tag, r in picks:
            m = manifest_by_key[(r["case"], int(r["z"]))]
            side = M.RIGHT if r["side"] == "right" else M.LEFT
            with np.load(processed / m["path"]) as d:
                img, gt = d["img"], d["msk"] == side
            predictor.set_image(np.repeat(img[..., None], 3, axis=2), image_format="RGB")
            box = tuple(json.loads(r["box_jit"]))
            masks, _, _ = predictor.predict(box=np.array(box, dtype=float), multimask_output=False)
            pred = np.asarray(masks[0], dtype=bool)
            dice = M.score_masks(gt, pred, (m["sp_row"], m["sp_col"]))["dice"]
            diff = abs(dice - float(r["dice"]))
            max_diff = max(max_diff, diff)
            if diff > 0.01:
                problems.append(f"QC re-run of {r['case']} z={r['z']} {r['side']}: Dice "
                                f"{dice:.4f} vs stored {float(r['dice']):.4f} -- not reproducible")
            save_overlay(img, gt, pred, box,
                         out / "qc" / group / f"{tag}_{dice:.3f}_{r['case']}_z{int(r['z']):04d}_{r['side']}.png",
                         f"{r['case']} z={r['z']} {r['side']} dice={dice:.3f}")
            n += 1
    return n, max_diff, problems


# ---------------------------------------------------------------------- one case
def process_case(case: str, rows: list[dict], processed: Path, predictor,
                 jitters: tuple[float, ...]) -> tuple[list, list, dict]:
    """Returns (instance rows, 3D volume rows, per-stage seconds). Loads every slice of the
    case: scored slices for prediction, the rest so the 3D ground truth is complete."""
    tm = dict.fromkeys(STAGES, 0.0)
    meta = rows[0]
    nz = len(rows)
    h, w = meta["h"], meta["w"]
    z_index = {r["z"]: i for i, r in enumerate(rows)}

    t = time.perf_counter()
    gt_vol = np.zeros((nz, h, w), np.uint8)
    imgs: dict[int, np.ndarray] = {}
    for i, r in enumerate(rows):
        with np.load(processed / r["path"]) as d:
            gt_vol[i] = d["msk"]
            if r["sam_eligible"]:
                imgs[r["z"]] = d["img"]
    tm["io"] += time.perf_counter() - t

    pred_vol = {(j, side): np.zeros((nz, h, w), bool) for j in jitters for side, _ in SIDES}
    inst_rows: list[dict] = []

    for r in rows:
        if not r["sam_eligible"]:
            continue
        img = imgs[r["z"]]
        t = time.perf_counter()
        predictor.set_image(np.repeat(img[..., None], 3, axis=2), image_format="RGB")
        _cuda_sync()
        tm["encode"] += time.perf_counter() - t
        zi = z_index[r["z"]]
        for side, prefix in SIDES:
            box = r[f"{prefix}_box"]
            if box is None:
                continue
            gt = gt_vol[zi] == side
            if not gt.any():
                raise RuntimeError(f"{case} z={r['z']} {prefix}: box present but mask empty")
            clean = tuple(int(v) for v in box)
            for j in jitters:
                jb = M.jitter_box(clean, j, h, w, M.jitter_seed(case, r["z"], side, j))
                t = time.perf_counter()
                masks, scores, _ = predictor.predict(
                    box=np.array(jb, dtype=float), multimask_output=False)
                pred = np.asarray(masks[0], dtype=bool)
                tm["decode"] += time.perf_counter() - t
                t = time.perf_counter()
                s = M.score_masks(gt, pred, (r["sp_row"], r["sp_col"]))
                tm["metric2d"] += time.perf_counter() - t
                pred_vol[(j, side)][zi] = pred
                inst_rows.append({
                    "dataset": r["dataset"], "patient": r["patient"], "case": case,
                    "sequence": r["sequence"], "modality": r["modality"], "split": r["split"],
                    "z": r["z"], "side": M.SIDE_NAME[side], "jitter": j,
                    "box_clean": json.dumps(list(clean)), "box_jit": json.dumps(list(jb)),
                    "box_iou": round(M.box_iou(clean, jb), 4),
                    "gt_area": s["gt_vox"], "pred_area": s["pred_vox"],
                    "dice": round(s["dice"], 6),
                    **{f"nsd_{t_:g}": round(s[f"nsd_{t_:g}"], 6) for t_ in M.TOLERANCES_MM},
                    "sam_iou": round(float(scores[0]), 6),
                    "sp_row": r["sp_row"], "sp_col": r["sp_col"],
                })

    # 3D: stack the per-slice predictions and score the whole kidney (D24a/C4).
    t = time.perf_counter()
    vol_rows: list[dict] = []
    sp3 = (meta["sp_slice"], meta["sp_row"], meta["sp_col"])
    for side, prefix in SIDES:
        gt3 = gt_vol == side
        if not gt3.any():
            continue
        # GT on slices SAM was never prompted on (sub-50px tips): missed by construction.
        sliver = sum(r[f"{prefix}_area"] for r in rows
                     if r[f"{prefix}_box"] is None and r[f"{prefix}_area"] > 0)
        n_scored = sum(1 for r in rows if r[f"{prefix}_box"] is not None)
        for j in jitters:
            s = M.score_masks(gt3, pred_vol[(j, side)], sp3)
            vol_rows.append({
                "dataset": meta["dataset"], "patient": meta["patient"], "case": case,
                "sequence": meta["sequence"], "modality": meta["modality"],
                "split": meta["split"], "side": M.SIDE_NAME[side], "jitter": j,
                "dice3d": round(s["dice"], 6),
                **{f"nsd3d_{t_:g}": round(s[f"nsd_{t_:g}"], 6) for t_ in M.TOLERANCES_MM},
                "gt_vox": s["gt_vox"], "pred_vox": s["pred_vox"],
                "sliver_vox": sliver,
                "sliver_frac": round(sliver / max(1, s["gt_vox"]), 6),
                "n_slices": nz, "n_scored": n_scored,
                "sp_row": meta["sp_row"], "sp_col": meta["sp_col"], "sp_slice": meta["sp_slice"],
            })
    tm["metric3d"] += time.perf_counter() - t
    return inst_rows, vol_rows, tm


def _backup(paths: tuple[Path, ...]) -> str:
    """Copy each existing file to <stem>.bak-<stamp>.csv, with a stamp no earlier backup
    used, so two backups within the same second never overwrite each other. Returns the
    stamp, or "" when there was nothing to back up."""
    existing = [pth for pth in paths if pth.exists()]
    if not existing:
        return ""
    base, i = time.strftime("%Y%m%d-%H%M%S"), 0
    stamp = base
    while any(pth.with_name(f"{pth.stem}.bak-{stamp}.csv").exists() for pth in existing):
        i += 1
        stamp = f"{base}-{i}"
    for pth in existing:
        shutil.copy2(pth, pth.with_name(f"{pth.stem}.bak-{stamp}.csv"))
    return stamp


def _rewrite(path: Path, fields: list[str], rows: list[dict]) -> None:
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
        fh.flush()
        os.fsync(fh.fileno())                 # on disk before it replaces the original
    os.replace(tmp, path)                     # atomic: never a half-written results file



# --------------------------------------------------------------------- sharding
# Parallel jobs never touch the same file. Shard k of N reads the base files in --out
# (read-only) to see which cases are already complete, takes its fixed share of the rest,
# and writes only to --out/shards/<k>_of_<N>/. A separate --merge step combines base +
# shards into --out, verifies the totals and runs QC. Base files change only in --merge,
# and only once every case is complete somewhere, so every shard computes the same split.
def shard_dir(out: Path, k: int, n: int) -> Path:
    return out / "shards" / f"{k}_of_{n}"


def n_encode_slices(case_rows: list[dict]) -> int:
    return sum(1 for r in case_rows if r["sam_eligible"])


def assign_shards(todo: list[str], cases: dict, n: int) -> list[list[str]]:
    """Split cases over n shards, balanced by slices to encode (greedy, largest first).
    Deterministic: depends only on the case list."""
    load, parts = [0] * n, [[] for _ in range(n)]
    for c in sorted(todo, key=lambda c: (-n_encode_slices(cases[c]), c)):
        i = min(range(n), key=lambda i: (load[i], i))
        parts[i].append(c)
        load[i] += n_encode_slices(cases[c])
    return parts


def setup_shard(out: Path, k: int, n: int, cases: dict, scope: list[str],
                manifest_by_key: dict) -> tuple[Path, list[str]]:
    """This shard's directory and cases. The assignment is frozen in assigned.json on first
    start, so a resubmitted shard keeps exactly its own cases."""
    sd = shard_dir(out, k, n)
    plan = sd / "assigned.json"
    if plan.exists():
        with open(plan) as fh:
            pj = json.load(fh)
        mine = pj["cases"]
        if (pj.get("shard"), pj.get("num_shards")) != (k, n):
            raise SystemExit(f"{plan} belongs to shard {pj.get('shard')}/{pj.get('num_shards')}, "
                             f"not {k}/{n}")
        if set(mine) - set(scope):
            raise SystemExit(f"{plan} names cases outside this run's scope "
                             f"(different --limit?): {sorted(set(mine) - set(scope))[:5]}")
        print(f"shard {k}/{n}: reusing frozen assignment of {len(mine)} cases")
    else:
        base = audit_outputs(out / "instances.csv", out / "volumes_3d.csv", cases, scope)
        # Cases the base already holds are skipped, so they must be right: re-derive them.
        bad = verify_rows([r for r in base["inst"] if r["case"] in base["complete"]],
                          [r for r in base["vols"] if r["case"] in base["complete"]],
                          cases, manifest_by_key)
        if bad:
            raise SystemExit("\n".join(summarise_problems("base results", bad)) +
                             "\nrefusing to build on them")
        remaining = [c for c in scope if c not in base["complete"]]
        parts = assign_shards(remaining, cases, n)
        mine = parts[k]
        sd.mkdir(parents=True, exist_ok=True)
        tmp = plan.with_suffix(".json.tmp")
        with open(tmp, "w") as fh:
            json.dump({"shard": k, "num_shards": n, "cases": mine,
                       "n_base_complete": len(base["complete"]), "n_remaining": len(remaining),
                       "slices_per_shard": [sum(n_encode_slices(cases[c]) for c in p_)
                                            for p_ in parts]}, fh, indent=2)
        os.replace(tmp, plan)
        print(f"shard {k}/{n}: base has {len(base['complete'])}/{len(scope)} cases complete; "
              f"{len(remaining)} remaining split as "
              f"{[sum(n_encode_slices(cases[c]) for c in p_) for p_ in parts]} slices")
    mine_set = set(mine)
    return sd, [c for c in scope if c in mine_set]


def merge_shards(out: Path, n: int, cases: dict, scope: list[str],
                 manifest_by_key: dict) -> tuple[list[str], dict]:
    """Combine the base files and n shard directories into --out. Every case must be complete
    in at least one source. A case complete in two sources (e.g. a stray extra job) must
    agree on Dice to 0.01, and the first source wins. Writes nothing unless all checks pass.
    Returns (problems, info)."""
    sources = [("base", out)] + [(f"shard{k}", shard_dir(out, k, n)) for k in range(n)]
    problems: list[str] = []
    for lbl, d in sources[1:]:
        if not (d / "instances.csv").exists():
            problems.append(f"{lbl}: no results at {d}")
    audits = {lbl: audit_outputs(d / "instances.csv", d / "volumes_3d.csv", cases, scope)
              for lbl, d in sources}
    by_inst, by_vol = {}, {}
    for lbl, a in audits.items():
        if a["stray"]:
            problems.append(f"{lbl}: rows for cases outside scope {a['stray'][:5]}")
        bi, bv = defaultdict(list), defaultdict(list)
        for r in a["inst"]:
            bi[r["case"]].append(r)
        for r in a["vols"]:
            bv[r["case"]].append(r)
        by_inst[lbl], by_vol[lbl] = bi, bv

    chosen: dict[str, str] = {}
    missing: list[str] = []
    n_dup, dup_diff, n_disagree = 0, 0.0, 0
    for c in scope:
        have = [lbl for lbl, _ in sources if c in audits[lbl]["complete"]]
        if not have:
            part = [f"{lbl} {audits[lbl]['partial'][c]}" for lbl, _ in sources
                    if c in audits[lbl]["partial"]]
            missing.append(c + (f" (partial: {'; '.join(part)})" if part else ""))
            continue
        chosen[c] = have[0]
        if len(have) > 1:
            n_dup += 1
            ref = {inst_key(r): float(r["dice"]) for r in by_inst[have[0]][c]}
            case_diff = 0.0
            for other in have[1:]:
                for r in by_inst[other][c]:
                    d_ref = ref.get(inst_key(r))
                    case_diff = max(case_diff, 1.0 if d_ref is None else abs(float(r["dice"]) - d_ref))
            dup_diff = max(dup_diff, case_diff)
            n_disagree += case_diff > 0.01
    if missing:
        problems.append(f"{len(missing)} cases complete in no source -- resubmit the shards "
                        f"(they resume): {missing[:10]}")
    if dup_diff > 0.01:
        problems.append(f"{n_disagree} of {n_dup} cases scored twice disagree on Dice "
                        f"(by up to {dup_diff:.4f})")

    shard_runs = {}
    for lbl, d in sources[1:]:
        if (d / "run.json").exists():
            with open(d / "run.json") as fh:
                rj = json.load(fh)
            shard_runs[lbl] = {k_: rj.get(k_) for k_ in (
                "host", "elapsed_min_this_run", "n_slices_encoded_this_run",
                "slices_per_second", "seconds_per_slice_by_stage", "problems")}
    info = {"num_shards": n, "cases_from": {lbl: sum(v == lbl for v in chosen.values())
                                            for lbl, _ in sources},
            "cases_scored_twice": n_dup, "max_dice_diff_scored_twice": dup_diff,
            "shard_runs": shard_runs}
    if problems:
        return problems, info

    inst = [r for c in scope for r in by_inst[chosen[c]][c]]
    vols = [r for c in scope for r in by_vol[chosen[c]][c]]
    bad = summarise_problems("merged rows", verify_rows(inst, vols, cases, manifest_by_key))
    if bad:
        return bad, info
    stamp = _backup((out / "instances.csv", out / "volumes_3d.csv"))
    _rewrite(out / "instances.csv", INSTANCE_FIELDS, inst)
    _rewrite(out / "volumes_3d.csv", VOLUME_FIELDS, vols)
    print(f"merge: wrote {len(inst)} instance + {len(vols)} volume rows; cases from "
          f"{info['cases_from']}; {n_dup} cases scored twice (max Dice diff {dup_diff:.1e})"
          + (f"; backup suffix .bak-{stamp}" if stamp else ""))
    return [], info


# ------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--processed", type=Path, help="prep_data.py output directory")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--model-type", default="vit_h", choices=("vit_h", "vit_l", "vit_b"))
    p.add_argument("--out", type=Path)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=None, help="cases per group (smoke run)")
    p.add_argument("--qc-per-group", type=int, default=20)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true",
                   help="start afresh over existing results (otherwise refused)")
    p.add_argument("--shard", type=int, default=None,
                   help="score only shard K (0-based) of --num-shards; writes to --out/shards/")
    p.add_argument("--num-shards", type=int, default=None)
    p.add_argument("--merge", type=int, default=None, metavar="N",
                   help="combine --out with its N shard outputs, verify, write QC")
    p.add_argument("--self-test", action="store_true", help="check metrics and exit")
    args = p.parse_args(argv)
    if (args.shard is None) != (args.num_shards is None):
        p.error("--shard and --num-shards go together")
    if args.num_shards is not None and not (args.num_shards >= 1 and 0 <= args.shard < args.num_shards):
        p.error(f"need 0 <= --shard < --num-shards (got {args.shard}, {args.num_shards})")
    if args.merge is not None and (args.shard is not None or args.merge < 1):
        p.error("--merge N (N >= 1) is a separate step, not combined with --shard")

    t_self = time.time()
    M.self_test()
    print(f"metrics self-test OK ({time.time() - t_self:.1f}s)")
    if args.self_test:
        return 0
    for name in ("processed", "checkpoint", "out"):
        if getattr(args, name) is None:
            p.error(f"--{name.replace('_', '-')} is required unless --self-test")
    if not args.checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")

    rows = load_manifest(args.processed)
    cases = group_cases(rows)
    manifest_by_key = {(r["case"], r["z"]): r for r in rows}
    args.out.mkdir(parents=True, exist_ok=True)

    # Case order is grouped so --limit takes a balanced sample across datasets.
    def gkey(c):
        m = cases[c][0]
        return (m["dataset"], m["split"], m["modality"], m["sequence"])
    scope = sorted(cases, key=lambda c: (gkey(c), c))
    if args.limit:
        seen: dict = defaultdict(int)
        keep = []
        for c in scope:
            if seen[gkey(c)] < args.limit:
                seen[gkey(c)] += 1
                keep.append(c)
        scope = keep

    # ---- pre-flight, before any GPU time: the data must be what D22a counted, and every
    # case must stack into a volume.
    layout = [msg for c in scope for msg in check_case_layout(c, cases[c])]
    if layout:
        raise SystemExit("case layout problems:\n  " + "\n  ".join(layout[:10]))
    got = count_groups(cases, scope)
    print(f"\n{'group':<22}{'cases':>7}{'slices':>8}{'instances':>11}")
    for g in sorted(got):
        print(f"{'/'.join(x for x in g if x):<22}{got[g][0]:>7}{got[g][1]:>8}{got[g][2]:>11}")
    if not args.limit and got != EXPECTED_GROUPS:
        diff = {g: (got.get(g), EXPECTED_GROUPS.get(g))
                for g in set(got) | set(EXPECTED_GROUPS) if got.get(g) != EXPECTED_GROUPS.get(g)}
        print(f"\nPROBLEMS:\n  !! evaluation set differs from the D22a counts "
              f"(group: (found, expected)): {diff}\n\nEVAL_FAILED")
        return 1

    # Where this job writes, and which cases it owns.
    is_shard = args.shard is not None
    if is_shard:
        work, my_scope = setup_shard(args.out, args.shard, args.num_shards, cases, scope,
                                     manifest_by_key)
        resume = True                       # a shard always continues its own files
    else:
        work, my_scope, resume = args.out, scope, args.resume
    label = (f"shard {args.shard}/{args.num_shards}" if is_shard
             else f"merge of {args.merge} shards" if args.merge else "single job")
    print(f"mode: {label}; writing to {work}")

    sp_rep = spacing_report([r for r in rows if r["sam_eligible"]])
    with open(work / "spacing_report.csv", "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(sp_rep[0].keys()))
        wtr.writeheader()
        wtr.writerows(sp_rep)
    coarsest = max(r["sp_max"] for r in sp_rep)
    print(f"\n{'dataset/modality':<18}{'slices':>8}{'sp min':>9}{'sp med':>9}{'sp max':>9}  note")
    for r in sp_rep:
        note = "NSD@1 is sub-pixel here" if r["nsd1_subpixel"] else ""
        print(f"{r['dataset'] + '/' + r['modality']:<18}{r['n_slices']:>8}{r['sp_min']:>9}"
              f"{r['sp_median']:>9}{r['sp_max']:>9}  {note}")
    print(f"coarsest in-plane pixel = {coarsest} mm -> headline NSD tolerance "
          f"{M.HEADLINE_TOLERANCE_MM} mm is "
          f"{'VALID' if M.HEADLINE_TOLERANCE_MM >= coarsest else 'TOO TIGHT (D24a/C5)'}")

    inst_path, vol_path = work / "instances.csv", work / "volumes_3d.csv"
    todo = list(my_scope)
    mode = "w"
    merge_info: dict = {}
    if args.merge:
        merge_problems, merge_info = merge_shards(args.out, args.merge, cases, scope,
                                                  manifest_by_key)
        if merge_problems:
            print("\nPROBLEMS (base files left untouched):")
            for m in merge_problems:
                print(f"  !! {m}")
            print("\nEVAL_FAILED")
            return 1
        todo, mode = [], "a"                # nothing to score: verify + QC only
    elif resume and (inst_path.exists() or vol_path.exists()):
        a = audit_outputs(inst_path, vol_path, cases, my_scope)
        for c, (gi, wi_, gv, wv_) in sorted(a["partial"].items()):
            print(f"  resume: {c} is partial (instances {gi}/{wi_}, volumes {gv}/{wv_}) — redoing")
        for d_ in a["defects"]:
            print(f"  resume: {d_}")
        # Complete cases are kept and skipped, so they must be right: re-derive them.
        bad = verify_rows([r for r in a["inst"] if r["case"] in a["complete"]],
                          [r for r in a["vols"] if r["case"] in a["complete"]],
                          cases, manifest_by_key)
        if bad:
            raise SystemExit("\n".join(summarise_problems("existing results", bad)) +
                             "\nrefusing to resume on top of them")
        if not a["clean"]:
            # Drop partial / damaged / out-of-scope rows so the redo cannot duplicate them,
            # and give both files a header. Back up first: they hold hours of GPU time.
            stamp = _backup((inst_path, vol_path))
            _rewrite(inst_path, INSTANCE_FIELDS, [r for r in a["inst"] if r["case"] in a["complete"]])
            _rewrite(vol_path, VOLUME_FIELDS, [r for r in a["vols"] if r["case"] in a["complete"]])
            print(f"  resume: rewrote CSVs keeping {len(a['complete'])} complete cases"
                  + (f" (backup suffix .bak-{stamp})" if stamp else ""))
        todo = [c for c in my_scope if c not in a["complete"]]
        mode = "a"
        print(f"resume: {len(a['complete'])} cases complete, {len(todo)} to do")
    elif (inst_path.exists() or vol_path.exists()) and not args.overwrite:
        raise SystemExit(f"{work} already holds results. Pass --resume to continue them "
                         f"(RESUME=1 in the sbatch), or --overwrite to discard them.")

    expected_new = sum(expected_rows(cases[c])[0] for c in todo)
    print(f"\n{len(todo)} cases, {len(M.JITTER_LEVELS)} jitter levels -> "
          f"{expected_new} instance scores to compute this run")

    predictor = build_predictor(args.checkpoint, args.model_type, args.device)
    print(f"loaded {args.model_type} from {args.checkpoint.name}")

    stage = dict.fromkeys(STAGES, 0.0)
    n_inst = n_vol = n_encode = 0
    t0 = time.time()
    fi = open(inst_path, mode, newline="")
    fv = open(vol_path, mode, newline="")
    wi = csv.DictWriter(fi, fieldnames=INSTANCE_FIELDS, extrasaction="ignore")
    wv = csv.DictWriter(fv, fieldnames=VOLUME_FIELDS, extrasaction="ignore")
    if mode == "w":
        wi.writeheader()
        wv.writeheader()
    try:
        for n, case in enumerate(todo, 1):
            ir, vr, tm = process_case(case, cases[case], args.processed, predictor,
                                      M.JITTER_LEVELS)
            wi.writerows(ir)
            wv.writerows(vr)
            fi.flush()
            fv.flush()                      # a crash never loses completed cases
            os.fsync(fi.fileno())
            os.fsync(fv.fileno())
            n_inst += len(ir)
            n_vol += len(vr)
            n_encode += n_encode_slices(cases[case])
            for s_ in STAGES:
                stage[s_] += tm[s_]
            if n % 10 == 0 or n == len(todo):
                el = time.time() - t0
                per = " ".join(f"{s_} {stage[s_] / max(n_encode, 1):.2f}" for s_ in STAGES)
                print(f"  [{n}/{len(todo)}] {case}  {n_encode} slices  "
                      f"{n_encode / max(el, 1e-9):.3f} slices/s  eta "
                      f"{el / n * (len(todo) - n) / 60:.1f} min  | s/slice: {per}", flush=True)
    finally:
        fi.close()
        fv.close()
    elapsed = time.time() - t0

    # ---- re-verify the files on disk, for every mode
    problems: list[str] = []
    a = audit_outputs(inst_path, vol_path, cases, my_scope)
    n_inst_total, n_vol_total = len(a["inst"]), len(a["vols"])
    if len(a["complete"]) != len(my_scope):
        missing = [c for c in my_scope if c not in a["complete"]]
        problems.append(f"{len(missing)} cases incomplete after the run: {missing[:5]}")
    if a["stray"]:
        problems.append(f"rows for cases outside scope: {a['stray'][:5]}")
    problems += a["defects"]
    problems += summarise_problems("final rows", verify_rows(a["inst"], a["vols"], cases,
                                                             manifest_by_key))
    n_want = expected_total_instances() * len(M.JITTER_LEVELS)
    if not is_shard and not args.limit and n_inst_total != n_want:
        problems.append(f"{n_inst_total} instance rows on disk, expected {n_want} "
                        f"({expected_total_instances()} instances x {len(M.JITTER_LEVELS)} "
                        f"jitter levels)")
    if M.HEADLINE_TOLERANCE_MM < coarsest:
        problems.append(f"headline NSD tolerance {M.HEADLINE_TOLERANCE_MM} mm is below the "
                        f"coarsest pixel {coarsest} mm (D24a/C5)")

    # ---- QC overlays from the CSV: every group, whichever job scored it. Shards skip
    # this; the merge step does it once over the combined results.
    t_qc = time.time()
    n_qc, qc_diff, qc_problems = (0, 0.0, [])
    if not problems and not is_shard:
        if (args.out / "qc").is_dir():
            shutil.rmtree(args.out / "qc")  # our own regenerated output; no stale picks
        n_qc, qc_diff, qc_problems = write_qc_from_csv(
            a["inst"], manifest_by_key, args.processed, predictor, args.out, args.qc_per_group)
        problems += qc_problems
    t_qc = time.time() - t_qc

    per_slice = {s_: round(stage[s_] / max(n_encode, 1), 4) for s_ in STAGES}
    print(f"\nthis run: {n_inst} instance + {n_vol} volume scores in {elapsed / 60:.1f} min, "
          f"{n_encode} slices at {n_encode / max(elapsed, 1e-9):.3f} slices/s")
    print("per-slice seconds by stage: " + "  ".join(f"{k} {v:.3f}" for k, v in per_slice.items()))
    print(f"on disk ({work.name}): {n_inst_total} instance rows, {n_vol_total} volume rows, "
          f"{len(a['complete'])}/{len(my_scope)} cases complete")
    if not is_shard:
        print(f"QC: {n_qc} overlays in {t_qc / 60:.1f} min; max |Dice| difference on re-run = "
              f"{qc_diff:.2e}")

    import PIL
    with open(work / "run.json", "w") as fh:
        json.dump({
            "args": {k: str(v) for k, v in vars(args).items()},
            "mode": label,
            "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "host": platform.node(),
            "python": platform.python_version(),
            "versions": {"numpy": np.__version__, "pillow": PIL.__version__},
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
            "tolerances_mm": list(M.TOLERANCES_MM),
            "headline_tolerance_mm": M.HEADLINE_TOLERANCE_MM,
            "jitter_levels": list(M.JITTER_LEVELS), "headline_jitter": M.HEADLINE_JITTER,
            "coarsest_in_plane_mm": coarsest,
            "n_cases_in_scope": len(my_scope), "n_cases_this_run": len(todo),
            "n_slices_encoded_this_run": n_encode,
            "n_instance_rows_on_disk": n_inst_total, "n_volume_rows_on_disk": n_vol_total,
            "elapsed_min_this_run": round(elapsed / 60, 2),
            "slices_per_second": round(n_encode / max(elapsed, 1e-9), 4),
            "seconds_per_slice_by_stage": per_slice,
            "qc_overlays": n_qc, "qc_max_dice_diff": qc_diff,
            "merge": merge_info,
            "problems": problems,
        }, fh, indent=2)

    ok_tag, fail_tag = ("SHARD_OK", "SHARD_FAILED") if is_shard else ("EVAL_OK", "EVAL_FAILED")
    if problems:
        print("\nPROBLEMS:")
        for m in problems:
            print(f"  !! {m}")
        print(f"\n{fail_tag}")
        return 1
    print(f"\n{ok_tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
