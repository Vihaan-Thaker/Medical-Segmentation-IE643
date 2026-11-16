#!/usr/bin/env python
"""Arms 3-4: score a trained student on the test + external sets (ARM3_EVAL_PLAN.md, D53).
Arm 4 (ARM4_CODE_PLAN.md §11): an arm4_plain_s<seed> run (run.json with a kd block and the
selected teacher's sha256) is evaluated exactly like a plain arm-3 student; its model type is
student_plain_kd and the eval run.json records the kd settings.

    python eval_student.py --processed P --run-dir runs/arm3_plain_s643 \
                           --out outputs/arm3_plain_s643 --expect-gpu A40 --resume

Same evaluation as arms 1, 1b and 2 (D22/D22a): the same 231 cases, 7,386 scored slices and
13,145 kidney instances, scored with sam_metrics; the output is in eval_sam.py's exact format
(instances.csv, volumes_3d.csv, model.json stamp, run.json), so the unchanged aggregate_arm.py
and compare_arms.py work on it. eval_sam's own machinery is IMPORTED, not copied: the D22a
anchors, case-layout check, verify_rows, audit_outputs/resume, check_stamp and save_overlay.
Only the prediction step differs: student_core.load_student + net_input +
predict_native_batch, the exact path that produced the recorded validation score.

* plain student: one prediction per slice; an instance's prediction is (labels == side). It
  is written into the rows of all 4 jitter levels with the evaluation's boxes recorded, so
  rows, counts and compare_arms' identical-prompt proof match the SAM arms. Flat across
  jitter by construction.
* box student: per jitter level, one forward pass per scored slice with both sides'
  jittered boxes as input channels (as in training and validation).
* volumes_3d.csv: SAM's rule -- a side's predicted volume only on that side's boxed slices.
* plain only: volumes_3d_full.csv (every slice predicted, jitter 0) and fp_slices.csv
  (kidney-free slices with >= 50 predicted kidney pixels), the deployed, prompt-free setting.
* sam_iou = nan (no IoU head). Before any test slice is read, AMOS val is re-scored from
  best.pt on this GPU and must match the recorded best score within 0.002.
Ends EVAL_OK / EVAL_FAILED.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
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
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_sam as E  # noqa: E402
import sam_metrics as M  # noqa: E402
import student_core as C  # noqa: E402
import student_data as D  # noqa: E402
import student_unet as U  # noqa: E402
import train_student as T  # noqa: E402

STAGES = ("io", "forward", "metric2d", "metric3d")
FP_FIELDS = ["dataset", "patient", "case", "sequence", "modality", "split", "n_slices",
             "n_kidney_free", "n_fp_slices", "fp_pixels"]
VAL_RECHECK_TOL = 0.002          # A100 (training) vs A40 (evaluation) numerics
QC_TOL = 0.01                    # as eval_sam's QC re-run check
FP_MIN_PX = D.MIN_AREA           # a predicted kidney of >= 50 px on a kidney-free slice
PRED_BATCH = 16


def log(msg: str) -> None:
    print(msg, flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 23), b""):
            h.update(block)
    return h.hexdigest()


# ------------------------------------------------------------------ training record
def check_training(run_dir: Path, processed: Path) -> dict:
    """The student must be a finished, untouched training run on the current data."""
    run_dir = Path(run_dir)
    rj_path, best = run_dir / "run.json", run_dir / "best.pt"
    if not rj_path.exists() or not best.exists():
        raise SystemExit(f"REFUSING -- {run_dir} has no run.json / best.pt (not a finished training run)")
    rj = json.loads(rj_path.read_text())
    problems = []
    if rj.get("status") != "TRAIN_OK":
        problems.append(f"run.json status {rj.get('status')!r}, not TRAIN_OK")
    if rj.get("epochs_completed") != rj.get("settings", {}).get("epochs"):
        problems.append(f"{rj.get('epochs_completed')} epochs completed of {rj.get('settings', {}).get('epochs')}")
    kd = rj.get("kd")               # arm 4 (distilled) runs carry a kd block; arm 3 runs have none
    want_name = f"{'arm4' if kd else 'arm3'}_{rj.get('variant')}_s{rj.get('seed')}"
    if run_dir.name != want_name:
        problems.append(f"folder {run_dir.name!r} does not match variant/seed {want_name!r}")
    if kd is not None:
        import teacher_maps as TM   # local: arm-3 evaluations never load it
        if rj.get("variant") != "plain":
            problems.append(f"a distilled run must be a plain student, got {rj.get('variant')!r}")
        if not isinstance(kd, dict) or set(kd) != {"lambda", "temperature", "margin", "gate"} \
                or any(not isinstance(v, (int, float)) for v in kd.values()):
            problems.append(f"run.json kd block incomplete: {kd!r}")
        if (rj.get("teacher") or {}).get("teacher_sha256") != TM.TEACHER_SHA256:
            problems.append("run.json teacher_sha256 is not the selected arm-2 teacher's")
    sha = sha256_file(best)
    if sha != rj.get("best_sha256"):
        problems.append(f"best.pt sha256 {sha[:16]}... != run.json best_sha256 {str(rj.get('best_sha256'))[:16]}...")
    msha = D.manifest_sha256(Path(processed))
    if msha != rj.get("manifest_sha256"):
        problems.append("the manifest changed since training (sha256 differs)")
    if problems:
        raise SystemExit(f"REFUSING -- {run_dir}: " + "; ".join(problems))
    ident = {"model_type": f"student_{rj['variant']}{'_kd' if kd is not None else ''}",
             "checkpoint_name": f"{run_dir.name}/best.pt",
             "checkpoint_bytes": best.stat().st_size, "checkpoint_sha256": sha}
    return {"rj": rj, "ident": ident}


def load_model(run_dir: Path, rj: dict, device):
    model, norm, variant, size, ck = C.load_student(Path(run_dir) / "best.pt", device)
    if variant != rj["variant"] or int(ck["seed"]) != int(rj["seed"]) or int(ck["epoch"]) != int(rj["best_epoch"]):
        raise SystemExit(f"best.pt (variant {variant}, seed {ck['seed']}, epoch {ck['epoch']}) does not "
                         f"match run.json ({rj['variant']}, {rj['seed']}, {rj['best_epoch']})")
    n = U.count_params(model)
    if tuple(ck["arch"]["channels"]) == U.CHANNELS and n != U.EXPECTED_PARAMS[ck["arch"]["in_ch"]]:
        raise SystemExit(f"best.pt network has {n:,} parameters, expected {U.EXPECTED_PARAMS[ck['arch']['in_ch']]:,}")
    return model, norm, variant, size, n


def val_recheck(model, variant: str, norm, size: int, processed: Path, best_score: float, device,
                expected_counts, workers: int = 4) -> dict:
    """Re-score AMOS val with the training code's own validation; must match best_score."""
    store = D.SliceStore(Path(processed), "val", expected_counts, 8, log)
    items = D.ValItems(store, variant, norm, size)
    s = T.validate(model, items, T.val_loader(items, T.Cfg(val_workers=workers, val_batch=16)),
                   device, T.select_level_of(variant))
    diff = abs(float(s["selection"]) - float(best_score))
    return {"selection": float(s["selection"]), "recorded": float(best_score), "abs_diff": diff,
            "ok": diff <= VAL_RECHECK_TOL}


# --------------------------------------------------------------------- one case
def eval_boxes(r: dict, case: str, level: float) -> dict:
    """{side: jittered box at `level` | None}, the evaluation's own seeding (as eval_sam)."""
    out = {}
    for side, prefix in E.SIDES:
        b = r[f"{prefix}_box"]
        out[side] = None if b is None else M.jitter_box(
            tuple(int(v) for v in b), level, r["h"], r["w"], M.jitter_seed(case, r["z"], side, level))
    return out


def predict_slices(model, imgs: list, boxes: list | None, variant: str, norm, size: int, device) -> list:
    """Native-size label maps for a list of slices (batched; == per-slice predict_native)."""
    out = []
    for a in range(0, len(imgs), PRED_BATCH):
        xs, gs = [], []
        for k in range(a, min(a + PRED_BATCH, len(imgs))):
            x, _, g = C.net_input(imgs[k], norm, variant, None if boxes is None else boxes[k], size)
            xs.append(x)
            gs.append(g)
        out += [lab for lab, _ in C.predict_native_batch(model, xs, gs, device)]
    return out


def inst_row(r: dict, case: str, side: int, j: float, clean: tuple, jb: tuple, s: dict) -> dict:
    return {"dataset": r["dataset"], "patient": r["patient"], "case": case,
            "sequence": r["sequence"], "modality": r["modality"], "split": r["split"],
            "z": r["z"], "side": M.SIDE_NAME[side], "jitter": j,
            "box_clean": json.dumps(list(clean)), "box_jit": json.dumps(list(jb)),
            "box_iou": round(M.box_iou(clean, jb), 4),
            "gt_area": s["gt_vox"], "pred_area": s["pred_vox"], "dice": round(s["dice"], 6),
            **{f"nsd_{t:g}": round(s[f"nsd_{t:g}"], 6) for t in M.TOLERANCES_MM},
            "sam_iou": float("nan"), "sp_row": r["sp_row"], "sp_col": r["sp_col"]}


def vol_row(meta: dict, case: str, side: int, j: float, s: dict, sliver: int, nz: int, n_scored: int) -> dict:
    return {"dataset": meta["dataset"], "patient": meta["patient"], "case": case,
            "sequence": meta["sequence"], "modality": meta["modality"], "split": meta["split"],
            "side": M.SIDE_NAME[side], "jitter": j, "dice3d": round(s["dice"], 6),
            **{f"nsd3d_{t:g}": round(s[f"nsd_{t:g}"], 6) for t in M.TOLERANCES_MM},
            "gt_vox": s["gt_vox"], "pred_vox": s["pred_vox"], "sliver_vox": sliver,
            "sliver_frac": round(sliver / max(1, s["gt_vox"]), 6), "n_slices": nz, "n_scored": n_scored,
            "sp_row": meta["sp_row"], "sp_col": meta["sp_col"], "sp_slice": meta["sp_slice"]}


def process_case_student(case: str, rows: list, processed: Path, model, variant: str, norm, size: int,
                         device) -> dict:
    """-> {inst, vols, full, fp, tm}: instance rows (all 4 levels), 3D rows (SAM's rule),
    and for the plain student the whole-volume 3D rows and the false-positive record."""
    tm = dict.fromkeys(STAGES, 0.0)
    meta = rows[0]
    nz, h, w = len(rows), meta["h"], meta["w"]
    t = time.perf_counter()
    gt_vol = np.zeros((nz, h, w), np.uint8)
    imgs = {}
    for i, r in enumerate(rows):
        with np.load(processed / r["path"]) as d:
            gt_vol[i] = d["msk"]
            if variant == "plain" or r["sam_eligible"]:
                imgs[i] = d["img"]
    tm["io"] += time.perf_counter() - t
    scored = [i for i, r in enumerate(rows) if r["sam_eligible"]]
    levels = M.JITTER_LEVELS
    sp3 = (meta["sp_slice"], meta["sp_row"], meta["sp_col"])
    inst, vols, full, fp = [], [], [], None

    # ---- predictions: plain once per slice (all slices); box once per scored slice per level
    t = time.perf_counter()
    if variant == "plain":
        order = list(range(nz))
        labs = predict_slices(model, [imgs[i] for i in order], None, variant, norm, size, device)
        lab_of = {lev: dict(zip(order, labs)) for lev in levels}          # level-invariant
    else:
        lab_of = {}
        for lev in levels:
            bx = [eval_boxes(rows[i], case, lev) for i in scored]
            labs = predict_slices(model, [imgs[i] for i in scored], bx, variant, norm, size, device)
            lab_of[lev] = dict(zip(scored, labs))
    tm["forward"] += time.perf_counter() - t

    # ---- 2D instances
    t = time.perf_counter()
    for i in scored:
        r = rows[i]
        for side, prefix in E.SIDES:
            box = r[f"{prefix}_box"]
            if box is None:
                continue
            gt = gt_vol[i] == side
            if not gt.any():
                raise RuntimeError(f"{case} z={r['z']} {prefix}: box present but mask empty")
            clean = tuple(int(v) for v in box)
            cache = None
            for lev in levels:
                jb = M.jitter_box(clean, lev, h, w, M.jitter_seed(case, r["z"], side, lev))
                if variant == "plain" and cache is not None:
                    s = cache                                              # identical prediction
                else:
                    s = M.score_masks(gt, lab_of[lev][i] == side, (r["sp_row"], r["sp_col"]))
                    cache = s
                inst.append(inst_row(r, case, side, lev, clean, jb, s))
    tm["metric2d"] += time.perf_counter() - t

    # ---- 3D: SAM's rule (boxed slices of that side only); plain also whole volume
    t = time.perf_counter()
    for side, prefix in E.SIDES:
        gt3 = gt_vol == side
        if not gt3.any():
            continue
        boxed = [i for i, r in enumerate(rows) if r[f"{prefix}_box"] is not None]
        sliver = sum(r[f"{prefix}_area"] for r in rows if r[f"{prefix}_box"] is None and r[f"{prefix}_area"] > 0)
        cache = None
        for lev in levels:
            if variant == "plain" and cache is not None:
                s = cache
            else:
                pred3 = np.zeros((nz, h, w), bool)
                for i in boxed:
                    pred3[i] = lab_of[lev][i] == side
                s = M.score_masks(gt3, pred3, sp3)
                cache = s
            vols.append(vol_row(meta, case, side, lev, s, sliver, nz, len(boxed)))
        if variant == "plain":
            pred_full = np.stack([lab_of[0.0][i] == side for i in range(nz)])
            full.append(vol_row(meta, case, side, 0.0, M.score_masks(gt3, pred_full, sp3), sliver, nz, nz))
    if variant == "plain":
        free = [i for i, r in enumerate(rows) if int(r["has_kidney"]) == 0]
        px = [int((lab_of[0.0][i] > 0).sum()) for i in free]
        fp = {"dataset": meta["dataset"], "patient": meta["patient"], "case": case,
              "sequence": meta["sequence"], "modality": meta["modality"], "split": meta["split"],
              "n_slices": nz, "n_kidney_free": len(free), "n_fp_slices": sum(p >= FP_MIN_PX for p in px),
              "fp_pixels": sum(px)}
    tm["metric3d"] += time.perf_counter() - t
    return {"inst": inst, "vols": vols, "full": full, "fp": fp, "tm": tm}


# ------------------------------------------------------- the plain-only extra files
def audit_extras(out: Path, cases: dict, scope: list) -> dict:
    """Per-case completeness of volumes_3d_full.csv (one row per side with GT) and
    fp_slices.csv (one row per case)."""
    full, d1 = E.read_results(out / "volumes_3d_full.csv", E.VOLUME_FIELDS)
    fps, d2 = E.read_results(out / "fp_slices.csv", FP_FIELDS)
    cf, cp = defaultdict(list), defaultdict(list)
    for r in full:
        cf[r["case"]].append((r["side"], f"{float(r['jitter']):.4f}"))
    for r in fps:
        cp[r["case"]].append(r["case"])
    complete = set()
    for c in scope:
        want = {(M.SIDE_NAME[s], "0.0000") for s, p in E.SIDES if any(x[f"{p}_area"] > 0 for x in cases[c])}
        if len(cf.get(c, [])) == len(want) and set(cf.get(c, [])) == want and len(cp.get(c, [])) == 1:
            complete.add(c)
    stray = sorted(({r["case"] for r in full} | {r["case"] for r in fps}) - set(scope))
    return {"full": full, "fps": fps, "complete": complete, "defects": d1 + d2, "stray": stray}


def verify_extras(full: list, fps: list, cases: dict) -> list:
    bad = []
    side_of = {v: k for k, v in M.SIDE_NAME.items()}
    for r in full:
        rows = cases.get(r["case"])
        side = side_of.get(r["side"])
        if rows is None or side is None or float(r["jitter"]) != 0.0:
            bad.append(f"{r.get('case')} {r.get('side')}: not a whole-volume row")
            continue
        prefix = "rk" if side == M.RIGHT else "lk"
        if int(r["gt_vox"]) != sum(x[f"{prefix}_area"] for x in rows) or int(r["n_slices"]) != len(rows):
            bad.append(f"{r['case']} {r['side']}: whole-volume GT voxels / slices differ from the manifest")
        vals = [float(r["dice3d"])] + [float(r[f"nsd3d_{t:g}"]) for t in M.TOLERANCES_MM]
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in vals):
            bad.append(f"{r['case']} {r['side']}: whole-volume scores out of range")
    for r in fps:
        rows = cases.get(r["case"])
        if rows is None:
            bad.append(f"{r.get('case')}: fp row for a case outside the evaluation")
            continue
        n_free = sum(int(x["has_kidney"]) == 0 for x in rows)
        if int(r["n_kidney_free"]) != n_free or not 0 <= int(r["n_fp_slices"]) <= n_free:
            bad.append(f"{r['case']}: fp counts inconsistent with the manifest")
    return bad


# ------------------------------------------------------------------------- QC
def write_qc(inst_rows: list, manifest_by_key: dict, processed: Path, model, variant: str, norm, size: int,
             device, out: Path, k: int) -> tuple[int, float, list]:
    """Worst and median instances per group at the headline jitter: re-predict through the
    same path, compare with the stored Dice, write overlays (eval_sam.save_overlay)."""
    by = defaultdict(list)
    for r in inst_rows:
        if abs(float(r["jitter"]) - M.HEADLINE_JITTER) < 1e-9:
            by["_".join(x for x in (r["dataset"], r["modality"], r["sequence"]) if x)].append(r)
    n, max_diff, problems = 0, 0.0, []
    for group, rs in sorted(by.items()):
        rs.sort(key=lambda r: float(r["dice"]))
        mid = len(rs) // 2
        picks = [("worst", r) for r in rs[:k // 2]] + [("median", r) for r in rs[mid: mid + (k - k // 2)]]
        for tag, r in picks:
            m = manifest_by_key[(r["case"], int(r["z"]))]
            side = M.RIGHT if r["side"] == "right" else M.LEFT
            with np.load(processed / m["path"]) as d:
                img, gt = d["img"], d["msk"] == side
            boxes = None if variant == "plain" else [eval_boxes(m, r["case"], float(r["jitter"]))]
            lab = predict_slices(model, [img], boxes, variant, norm, size, device)[0]
            pred = lab == side
            dice = M.score_masks(gt, pred, (m["sp_row"], m["sp_col"]))["dice"]
            diff = abs(dice - float(r["dice"]))
            max_diff = max(max_diff, diff)
            if diff > QC_TOL:
                problems.append(f"QC re-run of {r['case']} z={r['z']} {r['side']}: Dice {dice:.4f} vs stored "
                                f"{float(r['dice']):.4f} -- not reproducible")
            E.save_overlay(img, gt, pred, tuple(json.loads(r["box_jit"])),
                           out / "qc" / group / f"{tag}_{dice:.3f}_{r['case']}_z{int(r['z']):04d}_{r['side']}.png",
                           f"{r['case']} z={r['z']} {r['side']} dice={dice:.3f}")
            n += 1
    return n, max_diff, problems


def speed_benchmark(model, variant: str, norm, size: int, device, cases: dict, scope: list,
                    processed: Path, n: int = 200, warm: int = 20) -> dict:
    """Batch-1 latency: network forward only and end-to-end predict_native (letterbox,
    forward, back-mapping), median over n scored slices after `warm` warm-up runs."""
    picks = [r for c in scope for r in cases[c] if r["sam_eligible"]][: n + warm]
    if not picks:
        return {}
    dev = torch.device(device)
    sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
    fwd, e2e = [], []
    model.eval()
    for k, r in enumerate(picks):
        with np.load(processed / r["path"]) as d:
            img = d["img"]
        boxes = None if variant == "plain" else eval_boxes(r, r["case"], M.HEADLINE_JITTER)
        x, _, _ = C.net_input(img, norm, variant, boxes, size)
        xb = x[None].to(dev)
        with torch.no_grad():
            sync()
            t = time.perf_counter()
            model(xb)
            sync()
            tf = time.perf_counter() - t
        t = time.perf_counter()
        C.predict_native(model, img, norm, variant, boxes, dev, size)
        sync()
        te = time.perf_counter() - t
        if k >= warm:
            fwd.append(tf)
            e2e.append(te)
    if not fwd:
        return {}
    return {"n_slices": len(fwd), "forward_ms_median": round(1000 * float(np.median(fwd)), 3),
            "end_to_end_ms_median": round(1000 * float(np.median(e2e)), 3),
            "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"}


# ------------------------------------------------------------------------- main
def run(args, model_factory=None, val_expected_counts=D.EXPECTED_COUNTS) -> int:
    M.self_test()
    processed, out, run_dir = Path(args.processed), Path(args.out), Path(args.run_dir)
    tr = check_training(run_dir, processed)
    rj, ident = tr["rj"], tr["ident"]
    device = torch.device(args.device)
    if args.expect_gpu:
        if not torch.cuda.is_available() or args.expect_gpu not in torch.cuda.get_device_name(0):
            raise SystemExit(f"expected a GPU named *{args.expect_gpu}*, got "
                             f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA'}")
    if model_factory is None:
        model, norm, variant, size, n_params = load_model(run_dir, rj, device)
    else:
        model, norm, variant, size, n_params = model_factory()
    log(f"student {ident['checkpoint_name']} ({variant}, seed {rj['seed']}, best epoch {rj['best_epoch']}, "
        f"val {rj['best_score']:.6f}): {n_params:,} parameters, sha256 {ident['checkpoint_sha256'][:16]}...")

    # ---- sanity gate: the evaluated weights are the selected ones, on this GPU
    vr = val_recheck(model, variant, norm, size, processed, rj["best_score"], device, val_expected_counts,
                     workers=args.val_workers)
    log(f"val re-check on this device: {vr['selection']:.6f} vs recorded {vr['recorded']:.6f} "
        f"(|diff| {vr['abs_diff']:.2e}, tolerance {VAL_RECHECK_TOL})")
    if not vr["ok"]:
        raise SystemExit("REFUSING -- best.pt does not reproduce its recorded validation score on this device")

    # ---- the evaluation set, exactly as eval_sam
    rows = E.load_manifest(processed)
    cases = E.group_cases(rows)
    manifest_by_key = {(r["case"], r["z"]): r for r in rows}

    def gkey(c):
        m = cases[c][0]
        return (m["dataset"], m["split"], m["modality"], m["sequence"])
    scope = sorted(cases, key=lambda c: (gkey(c), c))
    if args.limit:
        seen, keep = defaultdict(int), []
        for c in scope:
            if seen[gkey(c)] < args.limit:
                seen[gkey(c)] += 1
                keep.append(c)
        scope = keep
    layout = [msg for c in scope for msg in E.check_case_layout(c, cases[c])]
    if layout:
        raise SystemExit("case layout problems:\n  " + "\n  ".join(layout[:10]))
    got = E.count_groups(cases, scope)
    if not args.limit and got != E.EXPECTED_GROUPS:
        raise SystemExit(f"evaluation set differs from the D22a counts: found {got}, expected {E.EXPECTED_GROUPS}")
    log(f"evaluation set: {len(scope)} cases, "
        f"{sum(v[1] for v in got.values())} scored slices, {sum(v[2] for v in got.values())} instances")
    # as eval_sam (D24a/C5): the headline NSD tolerance must not be finer than the coarsest pixel
    sp_rep = E.spacing_report([r for r in rows if r["sam_eligible"]])
    coarsest = max(r["sp_max"] for r in sp_rep)

    # ---- stamp, then resume (eval_sam's logic)
    out.mkdir(parents=True, exist_ok=True)
    log(f"model stamp: {E.check_stamp(out, ident, write=True)}")
    with open(out / "spacing_report.csv", "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(sp_rep[0].keys()))
        wtr.writeheader()
        wtr.writerows(sp_rep)
    inst_path, vol_path = out / "instances.csv", out / "volumes_3d.csv"
    full_path, fp_path = out / "volumes_3d_full.csv", out / "fp_slices.csv"
    plain = variant == "plain"
    files = (inst_path, vol_path) + ((full_path, fp_path) if plain else ())
    todo, mode = list(scope), "w"
    if args.resume and any(p.exists() for p in files):
        a = E.audit_outputs(inst_path, vol_path, cases, scope)
        complete = set(a["complete"])
        defects, stray = list(a["defects"]), list(a["stray"])
        if plain:
            x = audit_extras(out, cases, scope)
            complete &= x["complete"]
            defects += x["defects"]
            stray += x["stray"]
        bad = E.verify_rows([r for r in a["inst"] if r["case"] in complete],
                            [r for r in a["vols"] if r["case"] in complete], cases, manifest_by_key)
        if plain:
            bad += verify_extras([r for r in x["full"] if r["case"] in complete],
                                 [r for r in x["fps"] if r["case"] in complete], cases)
        if bad:
            raise SystemExit("\n".join(E.summarise_problems("existing results", bad)) +
                             "\nrefusing to resume on top of them")
        present = {r["case"] for r in a["inst"]} | {r["case"] for r in a["vols"]}
        if plain:
            present |= {r["case"] for r in x["full"]} | {r["case"] for r in x["fps"]}
        # appendable only if every case present in ANY file is complete in ALL of them
        clean = not defects and not stray and all(p.exists() for p in files) and present == complete
        if not clean:
            stamp = E._backup(files)
            E._rewrite(inst_path, E.INSTANCE_FIELDS, [r for r in a["inst"] if r["case"] in complete])
            E._rewrite(vol_path, E.VOLUME_FIELDS, [r for r in a["vols"] if r["case"] in complete])
            if plain:
                E._rewrite(full_path, E.VOLUME_FIELDS, [r for r in x["full"] if r["case"] in complete])
                E._rewrite(fp_path, FP_FIELDS, [r for r in x["fps"] if r["case"] in complete])
            log(f"resume: rewrote the CSVs keeping {len(complete)} complete cases"
                + (f" (backup suffix .bak-{stamp})" if stamp else ""))
        todo = [c for c in scope if c not in complete]
        mode = "a"
        log(f"resume: {len(complete)} cases complete, {len(todo)} to do")
    elif any(p.exists() for p in files) and not args.overwrite:
        raise SystemExit(f"{out} already holds results. Pass --resume to continue them, or --overwrite to discard them.")

    # ---- score
    stage = dict.fromkeys(STAGES, 0.0)
    n_slices = 0
    t0 = time.time()
    handles = [open(p, mode, newline="") for p in files]
    fields = [E.INSTANCE_FIELDS, E.VOLUME_FIELDS] + ([E.VOLUME_FIELDS, FP_FIELDS] if plain else [])
    writers = [csv.DictWriter(h_, fieldnames=f_, extrasaction="raise") for h_, f_ in zip(handles, fields)]
    if mode == "w":
        for w_ in writers:
            w_.writeheader()
    try:
        for n, case in enumerate(todo, 1):
            res = process_case_student(case, cases[case], processed, model, variant, norm, size, device)
            writers[0].writerows(res["inst"])
            writers[1].writerows(res["vols"])
            if plain:
                writers[2].writerows(res["full"])
                writers[3].writerow(res["fp"])
            for h_ in handles:
                h_.flush()
                os.fsync(h_.fileno())
            for s_ in STAGES:
                stage[s_] += res["tm"][s_]
            n_slices += len(cases[case])
            if n % 20 == 0 or n == len(todo):
                el = time.time() - t0
                log(f"  [{n}/{len(todo)}] {case}  {n_slices / max(el, 1e-9):.1f} slices/s  "
                    f"eta {el / n * (len(todo) - n) / 60:.1f} min")
    finally:
        for h_ in handles:
            h_.close()
    elapsed = time.time() - t0

    # ---- re-verify everything on disk
    problems = []
    a = E.audit_outputs(inst_path, vol_path, cases, scope)
    if len(a["complete"]) != len(scope):
        problems.append(f"{len(scope) - len(a['complete'])} cases incomplete after the run")
    if a["stray"]:
        problems.append(f"rows for cases outside scope: {a['stray'][:5]}")
    problems += a["defects"]
    problems += E.summarise_problems("final rows", E.verify_rows(a["inst"], a["vols"], cases, manifest_by_key))
    if M.HEADLINE_TOLERANCE_MM < coarsest:
        problems.append(f"headline NSD tolerance {M.HEADLINE_TOLERANCE_MM} mm is below the coarsest "
                        f"pixel {coarsest} mm (D24a/C5)")
    n_want = E.expected_total_instances() * len(M.JITTER_LEVELS)
    if not args.limit and len(a["inst"]) != n_want:
        problems.append(f"{len(a['inst'])} instance rows on disk, expected {n_want}")
    fp_by_group = {}
    if plain:
        x = audit_extras(out, cases, scope)
        if len(x["complete"]) != len(scope) or x["stray"] or x["defects"]:
            problems.append(f"whole-volume / fp files incomplete: {len(x['complete'])}/{len(scope)} cases, "
                            f"defects {x['defects'][:3]}, stray {x['stray'][:3]}")
        problems += E.summarise_problems("whole-volume rows", verify_extras(x["full"], x["fps"], cases))
        acc = defaultdict(lambda: [0, 0])
        for r in x["fps"]:
            g = "/".join(v for v in (r["dataset"], r["modality"], r["sequence"]) if v)
            acc[g][0] += int(r["n_kidney_free"])
            acc[g][1] += int(r["n_fp_slices"])
        fp_by_group = {g: {"kidney_free_slices": v[0], "fp_slices": v[1],
                           "fp_rate": round(v[1] / v[0], 6) if v[0] else None} for g, v in sorted(acc.items())}
        for g, v in fp_by_group.items():
            log(f"  false positives on kidney-free slices, {g}: {v['fp_slices']}/{v['kidney_free_slices']} "
                f"({v['fp_rate']})")

    n_qc, qc_diff = 0, 0.0
    if not problems:
        if (out / "qc").is_dir():
            shutil.rmtree(out / "qc")
        n_qc, qc_diff, qc_probs = write_qc(a["inst"], manifest_by_key, processed, model, variant, norm, size,
                                           device, out, args.qc_per_group)
        problems += qc_probs
    speed = speed_benchmark(model, variant, norm, size, device, cases, scope, processed) if not problems else {}
    if speed:
        log(f"speed (batch 1): forward {speed['forward_ms_median']} ms, end to end "
            f"{speed['end_to_end_ms_median']} ms per slice on {speed['device']}")

    import PIL
    with open(out / "run.json", "w") as fh:
        json.dump({
            "args": {k: str(v) for k, v in vars(args).items()}, "mode": "single job (student)",
            "model": {k: ident[k] for k in E.STAMP_KEYS},
            "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "host": platform.node(),
            "python": platform.python_version(),
            "versions": {"numpy": np.__version__, "torch": torch.__version__, "pillow": PIL.__version__},
            "variant": variant, "seed": rj["seed"], "best_epoch": rj["best_epoch"], "best_score": rj["best_score"],
            "params": n_params, "val_recheck": vr, "tolerances_mm": list(M.TOLERANCES_MM),
            "headline_tolerance_mm": M.HEADLINE_TOLERANCE_MM, "jitter_levels": list(M.JITTER_LEVELS),
            "headline_jitter": M.HEADLINE_JITTER, "coarsest_in_plane_mm": coarsest,
            "jitter_note": ("prediction independent of the box: identical rows at every level" if plain
                            else "evaluated at every level with that level's boxes"),
            "n_cases_in_scope": len(scope), "n_cases_this_run": len(todo),
            "n_instance_rows_on_disk": len(a["inst"]), "n_volume_rows_on_disk": len(a["vols"]),
            "fp_kidney_free_slices": fp_by_group, "speed": speed,
            "elapsed_min_this_run": round(elapsed / 60, 2),
            "seconds_by_stage": {k: round(v, 2) for k, v in stage.items()},
            "qc_overlays": n_qc, "qc_max_dice_diff": qc_diff, "problems": problems,
            **({"arm": 4, "kd": rj["kd"], "teacher_sha256": rj["teacher"]["teacher_sha256"]}
               if rj.get("kd") is not None else {})}, fh, indent=2, default=str)
    if problems:
        log("\nPROBLEMS:")
        for m in problems:
            log(f"  !! {m}")
        log("\nEVAL_FAILED")
        return 1
    log(f"\n{len(a['inst'])} instance rows, {len(a['vols'])} volume rows; QC {n_qc} overlays, "
        f"max |Dice| re-run difference {qc_diff:.2e}")
    log("EVAL_OK")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--processed", required=True, type=Path)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--device", default="cuda")
    p.add_argument("--expect-gpu", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="cases per group (smoke run)")
    p.add_argument("--qc-per-group", type=int, default=20)
    p.add_argument("--val-workers", type=int, default=4)
    return run(p.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
