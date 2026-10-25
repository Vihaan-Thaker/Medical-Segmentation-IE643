#!/usr/bin/env python
"""Arm 4: the r4 teacher's mask logits on every boxed AMOS training slice (DECISIONS D46).

    python teacher_maps.py --processed P --checkpoint C --runs R --out O [--expect-gpu A40]

* Teacher = arm 2's selected, merged checkpoint (r = 4, D40). Its sha256 must equal the
  recorded constant, runs/arm2_r4/run.json's merged sha256 and runs/arm2_selection.json's
  r4 sha256, and the selection must have chosen r = 4. Anything else -> refuse.
* Model and forward = arm 2's test evaluation exactly: eval_sam.build_predictor (fp32, eval,
  no grad), set_image(uint8 slice repeated to 3 channels), predict(box, multimask_output=False,
  return_logits=True). The evaluation thresholds these logits at 0.
* Boxes: the CLEAN manifest boxes (the teacher's best estimate of each kidney). Only boxed
  sides (>= 50 px), only the train split, read through student_data.SliceStore -- the same
  integrity-checked rows the student trains on.
* Output: teacher_io's format; resumable (a valid existing file is kept, an invalid one is
  rewritten); the manifest is built by re-reading every file. QC: teacher Dice vs ground truth.
Ends TEACHER_OK / TEACHER_FAILED.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_sam as E  # noqa: E402
import sam_metrics as M  # noqa: E402
import student_data as D  # noqa: E402
import teacher_io as TIO  # noqa: E402

# arm 2's selected teacher: checkpoints/sam_vit_b_arm2_r4.pth (D40, FINALIZE_OK merge check 1.000000)
TEACHER_SHA256 = "c3a6ce163c073132a1835f0213b2071fc039bd8f346dc52c1590846d3137383d"
TEACHER_RANK = 4
QC_MIN_DICE = 0.90               # per-modality patient-mean teacher Dice on train (val was 0.959)
FLIP_MAX_FRACTION = 1e-4         # fp16 storage may flip the sign of at most this pixel fraction


def log(msg: str) -> None:
    print(msg, flush=True)


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_write_json(path: Path, obj: dict) -> None:
    tmp = Path(path).with_name(Path(path).name + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def check_teacher(checkpoint: Path, runs: Path, expected_sha: str) -> dict:
    """The checkpoint must be arm 2's selected r4 merge, by three independent records."""
    ident = E.checkpoint_identity(Path(checkpoint), "vit_b")
    sha = ident["checkpoint_sha256"]
    problems = []
    if sha != expected_sha:
        problems.append(f"checkpoint sha256 {sha[:16]}... != recorded teacher {expected_sha[:16]}...")
    rj = json.loads((Path(runs) / f"arm2_r{TEACHER_RANK}" / "run.json").read_text())
    if rj.get("status") != "TRAIN_OK" or not rj.get("merged", {}).get("check", {}).get("passed"):
        problems.append("arm2_r4 run.json is not TRAIN_OK with a passed merge check")
    if rj.get("merged", {}).get("sha256") != sha:
        problems.append("checkpoint sha256 differs from arm2_r4 run.json's merged sha256")
    sel = json.loads((Path(runs) / "arm2_selection.json").read_text())
    if sel.get("chosen_rank") != TEACHER_RANK:
        problems.append(f"arm2_selection.json chose r={sel.get('chosen_rank')}, not r={TEACHER_RANK}")
    if sel.get(f"r{TEACHER_RANK}", {}).get("merged_sha256") != sha:
        problems.append("checkpoint sha256 differs from arm2_selection.json's r4 sha256")
    if problems:
        raise SystemExit("REFUSING -- not arm 2's selected teacher: " + "; ".join(problems))
    return ident


def check_gpu(expect: str | None) -> dict:
    import torch
    if not torch.cuda.is_available():
        if expect:
            raise SystemExit(f"expected GPU {expect}, but CUDA is not available")
        return {"cuda": False}
    name = torch.cuda.get_device_name(0)
    if expect and expect not in name:
        raise SystemExit(f"expected a GPU named *{expect}*, got {name!r}")
    return {"cuda": True, "name": name, "torch": torch.__version__, "cuda_version": torch.version.cuda}


def teacher_slice(predictor, img: np.ndarray, msk: np.ndarray, boxes: dict) -> tuple[dict, dict]:
    """One slice through the evaluation's exact SAM calls. -> ({side: float32 logits},
    {side: (iou_pred, dice, fp16_flips)})."""
    predictor.set_image(np.repeat(img[..., None], 3, axis=2), image_format="RGB")
    logits, stats = {}, {}
    for side in (1, 2):
        box = boxes.get(side)
        if box is None:
            continue
        lg, scores, _ = predictor.predict(box=np.array(box, dtype=float), multimask_output=False,
                                          return_logits=True)
        lg = np.asarray(lg[0], dtype=np.float32)
        gt = msk == side
        if not gt.any():
            raise RuntimeError(f"side {side}: box present but mask empty")
        pred = lg > 0.0                                     # == the evaluation's mask
        flips = int(((lg.astype(np.float16) > 0) != pred).sum())
        logits[side] = lg
        stats[side] = (float(np.asarray(scores).reshape(-1)[0]), M.dice_coefficient(gt, pred), flips)
    return logits, stats


def run(processed: Path, checkpoint: Path, runs: Path, out: Path, device: str = "cuda",
        expect_gpu: str | None = None, expected_sha: str = TEACHER_SHA256,
        predictor_factory=None, expected_counts=D.EXPECTED_COUNTS, qc_min_dice: float = QC_MIN_DICE,
        threads: int = 8) -> int:
    t0 = time.time()
    out = Path(out)
    ident = check_teacher(checkpoint, runs, expected_sha)
    gpu = check_gpu(expect_gpu) if predictor_factory is None else {"cuda": False, "stub": True}
    D.check_split_integrity(Path(processed))
    identity = {"teacher_sha256": ident["checkpoint_sha256"],
                "manifest_sha256": D.manifest_sha256(Path(processed)), "format": TIO.FORMAT_VERSION,
                "split": "train", "boxes": "clean (manifest), no jitter", "model_type": "vit_b"}
    have = TIO.read_stamp(out)
    if have is not None and {k: have.get(k) for k in identity} != identity:
        raise SystemExit(f"REFUSING -- {out} holds maps from a different teacher/data/format: "
                         f"{ {k: (have.get(k), identity[k]) for k in identity if have.get(k) != identity[k]} }")
    out.mkdir(parents=True, exist_ok=True)
    if have is None:
        atomic_write_json(out / TIO.STAMP, {**identity, "created_utc": utc(),
                                            "checkpoint": ident})
    store = D.SliceStore(Path(processed), "train", expected_counts, threads, log)
    todo = [i for i, r in enumerate(store.rows) if r["sam_eligible"]]
    n_kid = sum(sum(b is not None for b in store.boxes(i).values()) for i in todo)
    log(f"teacher {ident['checkpoint_name']} ({identity['teacher_sha256'][:16]}...): "
        f"{len(todo)} boxed train slices, {n_kid} kidneys -> {out}")

    predictor = (predictor_factory() if predictor_factory is not None
                 else E.build_predictor(Path(checkpoint), "vit_b", device))
    done = kept = rewritten = 0
    t_run = time.time()
    for k, idx in enumerate(todo):
        r = store.rows[idx]
        boxes = store.boxes(idx)
        want = TIO.expected_meta(r, boxes, identity)
        path = TIO.slice_path(out, r["case"], r["z"])
        if path.exists():
            try:
                meta, _ = TIO.load_slice(out, r["case"], r["z"])
                if TIO.meta_matches(meta, want):
                    kept += 1
                    continue
                log(f"  {r['case']} z={r['z']}: stored file does not match -- rewriting")
            except Exception as e:                       # truncated / corrupt -> recompute
                log(f"  {r['case']} z={r['z']}: unreadable ({type(e).__name__}) -- rewriting")
            rewritten += 1
        img, msk = store.get(idx)
        try:
            logits, stats = teacher_slice(predictor, img, msk, boxes)
        except RuntimeError as e:
            raise SystemExit(f"{r['case']} z={r['z']}: {e}")
        meta = {**want, "patient": r["patient"], "modality": r["modality"],
                "iou": {str(s): stats[s][0] for s in stats}, "dice": {str(s): stats[s][1] for s in stats},
                "fp16_flips": {str(s): stats[s][2] for s in stats}, "written_utc": utc()}
        TIO.save_slice(out, meta, logits)
        done += 1
        if done % 500 == 0:
            el = time.time() - t_run
            log(f"  {k + 1}/{len(todo)} slices  ({done} computed, {el / done:.3f} s/slice)")
    log(f"computed {done}, kept {kept} valid existing ({rewritten} rewritten) in {time.time() - t_run:.0f}s")
    return finalize(out, store, todo, identity, ident, gpu, qc_min_dice, t0)


def finalize(out: Path, store, todo: list, identity: dict, ident: dict, gpu: dict,
             qc_min_dice: float, t0: float) -> int:
    """Re-read EVERY file, check it against the store row, build the manifest and the QC."""
    rows, problems = [], []
    per = defaultdict(lambda: defaultdict(list))         # modality -> patient -> [dice]
    flips = pixels = 0
    nbytes = 0
    for idx in todo:
        r = store.rows[idx]
        boxes = store.boxes(idx)
        want = TIO.expected_meta(r, boxes, identity)
        try:
            meta, logits = TIO.load_slice(out, r["case"], r["z"])
        except Exception as e:
            problems.append(f"{r['case']} z={r['z']}: {type(e).__name__}: {e}")
            continue
        if not TIO.meta_matches(meta, want):
            problems.append(f"{r['case']} z={r['z']}: meta does not match the store row")
            continue
        _, msk = store.get(idx)
        for s, lg in logits.items():
            d = M.dice_coefficient(msk == s, lg.astype(np.float32) > 0)
            # with no fp16 sign flips, the stored mask must reproduce the stored Dice exactly
            if meta["fp16_flips"][str(s)] == 0 and abs(d - meta["dice"][str(s)]) > 1e-9:
                problems.append(f"{r['case']} z={r['z']} side {s}: stored Dice {meta['dice'][str(s)]} "
                                f"!= recomputed {d}")
            per[r["modality"]][r["patient"]].append(meta["dice"][str(s)])
            flips += meta["fp16_flips"][str(s)]
            pixels += lg.size
        nbytes += TIO.slice_path(out, r["case"], r["z"]).stat().st_size
        rows.append({"case": r["case"], "patient": r["patient"], "modality": r["modality"], "z": r["z"],
                     "h": r["h"], "w": r["w"], "path": TIO.rel_path(r["case"], r["z"]),
                     "sides": "".join(str(s) for s in meta["sides"]),
                     "box_1": json.dumps(meta["boxes"].get("1")), "box_2": json.dumps(meta["boxes"].get("2")),
                     "iou_1": meta["iou"].get("1", ""), "iou_2": meta["iou"].get("2", ""),
                     "dice_1": meta["dice"].get("1", ""), "dice_2": meta["dice"].get("2", ""),
                     "fp16_flips": sum(meta["fp16_flips"].values())})
    expected_paths = {x["path"] for x in rows}
    stray = sorted(str(p.relative_to(out)) for p in (out / "slices").rglob("*")
                   if p.is_file() and str(p.relative_to(out)) not in expected_paths)
    if stray:
        problems.append(f"{len(stray)} unexpected files under slices/ (e.g. {stray[:3]})")
    n_kid = sum(len(v) for pats in per.values() for v in pats.values())
    want_kid = sum(sum(b is not None for b in store.boxes(i).values()) for i in todo)
    if len(rows) != len(todo) or n_kid != want_kid:
        problems.append(f"{len(rows)}/{len(todo)} slices and {n_kid}/{want_kid} kidneys verified")
    qc = {m: float(np.mean([np.mean(v) for v in pats.values()])) for m, pats in per.items()}
    qc_kid = {m: sum(len(v) for v in pats.values()) for m, pats in per.items()}
    flip_frac = flips / pixels if pixels else 0.0
    for m, v in qc.items():
        if v < qc_min_dice:
            problems.append(f"teacher Dice on {m} train {v:.4f} < {qc_min_dice} (wrong teacher or pipeline?)")
    if flip_frac > FLIP_MAX_FRACTION:
        problems.append(f"float16 storage flipped {flip_frac:.2e} of mask pixels (> {FLIP_MAX_FRACTION})")
    with open(out / "teacher_manifest.csv.tmp", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=TIO.MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)
    os.replace(out / "teacher_manifest.csv.tmp", out / "teacher_manifest.csv")
    h = hashlib.sha256((out / "teacher_manifest.csv").read_bytes()).hexdigest()
    status = "TEACHER_OK" if not problems else "TEACHER_FAILED"
    atomic_write_json(out / "run.json", {
        "status": status, "slices": len(rows), "kidneys": n_kid, "kidneys_per_modality": qc_kid,
        "teacher_train_dice_patient_mean": qc, "fp16_flip_fraction": flip_frac, "bytes": nbytes,
        "teacher_manifest_sha256": h, **identity, "checkpoint": ident, "gpu": gpu,
        "host": platform.node(), "seconds": round(time.time() - t0, 1), "finished_utc": utc(),
        "problems": problems[:50]})
    log("teacher Dice on train (patient mean, clean boxes): "
        + ", ".join(f"{m} {v:.4f} ({qc_kid[m]} kidneys)" for m, v in sorted(qc.items())))
    log(f"{len(rows)} slices, {n_kid} kidneys, {nbytes / 2 ** 30:.2f} GiB, fp16 sign flips {flip_frac:.2e}")
    if problems:
        for p in problems[:20]:
            log(f"!! {p}")
        log("TEACHER_FAILED")
        return 1
    log("TEACHER_OK")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--processed", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--runs", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--device", default="cuda")
    p.add_argument("--expect-gpu", default=None)
    a = p.parse_args(argv)
    return run(a.processed, a.checkpoint, a.runs, a.out, a.device, a.expect_gpu)


if __name__ == "__main__":
    raise SystemExit(main())
