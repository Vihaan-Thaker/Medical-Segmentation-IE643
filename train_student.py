#!/usr/bin/env python
"""Arm 3: train the prompt-free student U-Net from scratch (ARM3_CODE_PLAN.md, DECISIONS D42).

    train_student.py --processed P --run-dir R --variant {plain,box} --seed S [--preflight]

Default mode trains (resumable; resubmitting a finished run prints TRAIN_OK at once).
--preflight runs the one-off checks and timing on the GPU and ends PREFLIGHT_OK.

Recipe = nnU-Net v2 (verified in its source, plan §1a): SGD(lr 0.01, momentum 0.99, Nesterov,
weight decay 3e-5), poly decay per step, clip 12, CE + batch Dice with deep supervision,
nnU-Net augmentation without mirroring and with rotation <= 0.52 rad. Batch 12,
100 epochs x 1,000 steps. bf16 autocast for training, fp32 validation.
This file deliberately imports nothing from the arm-2 code (plan invariant 1).

Arm 4 (ARM4_CODE_PLAN.md, D59/D60): --kd --teacher-dir T --kd-lambda --kd-temperature
--kd-margin --kd-gate adds the distillation term of student_kd.kd_loss to arm 3's loss and
nothing else. With --kd off every code path, stamp and output is arm 3's (proven against
frozen_arm3/ by test_arm4.py and the arm-4 preflight). An arm-4 run refuses to start unless
its same-seed arm-3 partner (runs/arm3_plain_s<seed>) is TRAIN_OK with the same settings,
normalisation, manifest and initial weights.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import dataclasses
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import student_core as C  # noqa: E402
import student_data as D  # noqa: E402
import student_unet as U  # noqa: E402

CODE_FILES = ("student_unet.py", "student_core.py", "student_aug.py", "student_data.py",
              "train_student.py", "sam_metrics.py", "eval_sam.py")
KD_CODE_FILES = CODE_FILES + ("student_kd.py", "teacher_io.py", "teacher_maps.py")
A100_MIN_GIB = 75.0          # A100-80GB reports ~79.3 GiB; A100-40GB ~39.4 GiB
TRAIN_FIELDS = ["epoch", "train_loss", "train_ce", "train_dice", "grad_norm", "lr", "steps",
                "ct_draws", "mri_draws"]
KD_FIELDS = ["train_seg", "train_kd", "kd_ct", "kd_mri", "kd_terms", "kd_draw_frac"]
TAIL_FIELDS = ["union_dice", "fp_slice_rate", "epoch_s", "train_s", "val_s", "data_wait_s",
               "peak_mem_gib"]
KD_SETTINGS = ("kd", "kd_lambda", "kd_temperature", "kd_margin", "kd_gate")


def val_fields(variant: str) -> list[str]:
    if variant == "plain":
        return ["ct", "mri", "selection"]
    return ["ct@0.00", "mri@0.00", "ct@0.10", "mri@0.10", "selection"]


def metric_fields(variant: str, kd: bool = False) -> list[str]:
    return TRAIN_FIELDS + (KD_FIELDS if kd else []) + val_fields(variant) + TAIL_FIELDS


class TrainFailed(RuntimeError):
    pass


# ============================================================================ config
@dataclass
class Cfg:
    processed: str = ""
    run_dir: str = ""
    variant: str = "plain"
    seed: int = 643
    epochs: int = 100
    steps_per_epoch: int = 1000
    batch: int = 12
    lr: float = 0.01
    momentum: float = 0.99
    nesterov: bool = True
    weight_decay: float = 3e-5
    poly_exponent: float = 0.9
    clip: float = 12.0
    size: int = C.NET_SIZE
    channels: tuple = U.CHANNELS
    deep_supervision: bool = True
    amp: bool = True
    augment: bool = True                 # internal only (no CLI flag): preflight/tests switch it off
    max_level: float = D.TRAIN_MAX_LEVEL
    device: str = "cuda"
    expect_gpu: str | None = None
    workers: int = 12
    val_workers: int = 4
    val_batch: int = 16
    threads: int = 8
    expected_counts: dict | None = field(default_factory=lambda: copy.deepcopy(D.EXPECTED_COUNTS))
    check_params: bool = True
    allow_code_change: str | None = None
    stop_after_epoch: int | None = None  # test hook: simulate a job ending mid-run
    # arm 4 (ARM4_CODE_PLAN.md §1): distillation; all off = arm 3 exactly
    kd: bool = False
    teacher_dir: str = ""
    kd_lambda: float = 1.0
    kd_temperature: float = 1.0
    kd_margin: float = 0.20
    kd_gate: float = 0.80
    # arm-4 preflight (ARM4_CODE_PLAN.md §8)
    pf_kd_items: int = 200
    pf_kd_steps: int = 1000
    pf_kd_ratio: tuple = (0.1, 10.0)
    pf_kd_align: float = 0.90
    pf_kd_inside_aug: float = 0.99
    pf_kd_gate_fail: float = 0.10
    pf_kd_gate_warn: float = 0.05
    pf_kd_frozen: int = 100              # positions per block (consecutive + spread) for P3
    # preflight
    pf_learn_steps: int = 300
    pf_learn_slices: int = 16
    pf_learn_loss_ratio: float = 0.5
    pf_learn_dice_gain: float = 0.2
    pf_steps_plain: int = 1000
    pf_steps_box: int = 200
    pf_probe: int = 32
    pf_aug_items: int = 200
    pf_ceiling_fail: float = 0.90
    pf_wait_warn: float = 0.20


# Run-local settings that cannot change results; everything else must match on resume.
STAMP_EXCLUDE = {"processed", "run_dir", "workers", "val_workers", "val_batch", "threads",
                 "allow_code_change", "stop_after_epoch", "teacher_dir"}


def stamp_of(cfg: Cfg) -> dict:
    """Result-relevant settings. With kd off the kd keys are dropped, so arm 3's stamp is
    exactly what the arm-3 code wrote; the teacher's identity is stamped separately."""
    d = dataclasses.asdict(cfg)
    out = {k: v for k, v in d.items() if k not in STAMP_EXCLUDE and not k.startswith("pf_")}
    if not cfg.kd:
        for k in KD_SETTINGS:
            out.pop(k)
    return json.loads(json.dumps(out))


def run_dir_name(cfg: Cfg) -> str:
    return f"{'arm4' if cfg.kd else 'arm3'}_{cfg.variant}_s{cfg.seed}"


def kd_settings(cfg: Cfg) -> dict:
    return {"lambda": cfg.kd_lambda, "temperature": cfg.kd_temperature, "margin": cfg.kd_margin,
            "gate": cfg.kd_gate}


def check_kd_cfg(cfg: Cfg) -> None:
    """Arm-4 settings must be complete and sane (ARM4_CODE_PLAN.md §7.2)."""
    if not cfg.kd:
        return
    probs = []
    if cfg.variant != "plain":
        probs.append(f"distillation is defined for the plain student only, not {cfg.variant!r}")
    if not cfg.teacher_dir:
        probs.append("--teacher-dir is required with --kd")
    if not (math.isfinite(cfg.kd_lambda) and cfg.kd_lambda > 0):
        probs.append(f"kd_lambda {cfg.kd_lambda} must be > 0")
    if not (math.isfinite(cfg.kd_temperature) and cfg.kd_temperature > 0):
        probs.append(f"kd_temperature {cfg.kd_temperature} must be > 0")
    if not 0.0 <= cfg.kd_margin <= 1.0:
        probs.append(f"kd_margin {cfg.kd_margin} outside [0, 1]")
    if not 0.0 <= cfg.kd_gate <= 1.0:
        probs.append(f"kd_gate {cfg.kd_gate} outside [0, 1]")
    if probs:
        raise SystemExit("invalid arm-4 settings: " + "; ".join(probs))


# ============================================================================ helpers
def log(msg: str) -> None:
    print(msg, flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 23), b""):
            h.update(block)
    return h.hexdigest()


def code_hashes(kd: bool = False) -> dict:
    here = Path(__file__).resolve().parent
    return {f: sha256_file(here / f) for f in (KD_CODE_FILES if kd else CODE_FILES)}


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_write_json(path: Path, obj: dict) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def atomic_torch_save(obj, path: Path) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        torch.save(obj, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def write_metrics(path: Path, rows: list[dict], fields: list[str]) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def read_metrics(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh))


def gpu_info() -> dict:
    if not torch.cuda.is_available():
        return {"cuda": False}
    p = torch.cuda.get_device_properties(0)
    return {"cuda": True, "name": torch.cuda.get_device_name(0),
            "total_gib": round(p.total_memory / 2 ** 30, 2),
            "capability": list(torch.cuda.get_device_capability(0)),
            "bf16": bool(torch.cuda.is_bf16_supported()), "cuda_version": torch.version.cuda,
            "torch": torch.__version__}


def gpu_check(expect: str | None, info: dict | None = None) -> dict:
    """Refuse the wrong hardware. `info` is injectable for tests."""
    info = gpu_info() if info is None else info
    if expect is None:
        return info
    if not info.get("cuda"):
        raise SystemExit(f"expected GPU {expect}, but CUDA is not available")
    if expect not in info["name"]:
        raise SystemExit(f"expected a GPU named *{expect}*, got {info['name']!r}")
    if expect == "A100" and info["total_gib"] < A100_MIN_GIB:
        raise SystemExit(f"expected an A100-80GB, got {info['total_gib']} GiB")
    if not info.get("bf16"):
        raise SystemExit(f"{info['name']} does not support bf16")
    if tuple(info["capability"]) < (8, 0):
        raise SystemExit(f"{info['name']} capability {info['capability']} < 8.0")
    return info


def rng_state() -> dict:
    st = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st: dict) -> None:
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def weights_digest(model) -> str:
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def peak_gib(device) -> float:
    if torch.device(device).type != "cuda":
        return 0.0
    return round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)


def autocast_ctx(cfg: Cfg, device):
    dev = torch.device(device)
    return torch.autocast(device_type=dev.type, dtype=torch.bfloat16,
                          enabled=bool(cfg.amp and dev.type == "cuda"))


def float_outputs(out):
    """Model output (list with deep supervision, tensor without) -> fp32 for the loss."""
    return [o.float() for o in out] if isinstance(out, (list, tuple)) else out.float()


def per_epoch(cfg: Cfg) -> int:
    return cfg.steps_per_epoch * cfg.batch


def total_steps(cfg: Cfg) -> int:
    return cfg.epochs * cfg.steps_per_epoch


# ======================================================================= model / data
def build_model(cfg: Cfg, device, seed: int | None = None):
    torch.manual_seed(cfg.seed if seed is None else seed)
    in_ch = C.in_channels(cfg.variant)
    model = U.StudentUNet(in_ch, channels=tuple(cfg.channels), deep_supervision=cfg.deep_supervision)
    n = U.count_params(model)
    if cfg.check_params and tuple(cfg.channels) == U.CHANNELS:
        want = U.EXPECTED_PARAMS[in_ch] if cfg.deep_supervision else U.EXPECTED_PARAMS_NO_DS[in_ch]
        if n != want:
            raise SystemExit(f"{cfg.variant} student has {n:,} parameters, expected {want:,} (plan §3.1)")
    return model.to(device), n


def make_optimizer(model, cfg: Cfg):
    return torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
                           nesterov=cfg.nesterov, weight_decay=cfg.weight_decay)


def load_stores(cfg: Cfg):
    D.check_split_integrity(Path(cfg.processed))
    tr = D.SliceStore(Path(cfg.processed), "train", cfg.expected_counts, cfg.threads, log)
    va = D.SliceStore(Path(cfg.processed), "val", cfg.expected_counts, cfg.threads, log)
    return tr, va


def train_loader(ds, cfg: Cfg):
    w = int(cfg.workers)
    return DataLoader(ds, batch_size=cfg.batch, shuffle=False, drop_last=True, num_workers=w,
                      collate_fn=D.collate_train, worker_init_fn=D.worker_init if w else None,
                      persistent_workers=bool(w), prefetch_factor=2 if w else None,
                      multiprocessing_context=D.mp_context() if w else None)


def val_loader(items, cfg: Cfg):
    w = int(cfg.val_workers)
    return DataLoader(items, batch_size=cfg.val_batch, shuffle=False, drop_last=False, num_workers=w,
                      collate_fn=D.collate_list, worker_init_fn=D.worker_init if w else None,
                      persistent_workers=bool(w),
                      multiprocessing_context=D.mp_context() if w else None)


# ========================================================================= validation
def validate(model, items: D.ValItems, loader, device, select_level) -> dict:
    """Prompt-free (or box) predictions on every val slice, scored as D22: one row per boxed
    kidney instance per level; plus side-agnostic Dice on boxed slices and the false-positive
    rate on kidney-free slices (both reported only, at select_level)."""
    store = items.store
    rows, union, fp = [], {}, []
    sk = C.level_key(select_level)
    for batch in loader:
        for lev in items.levels:
            lk = C.level_key(lev)
            preds = C.predict_native_batch(model, [b["xs"][lk] for b in batch],
                                           [b["geom"] for b in batch], device)
            for b, (labels, _) in zip(batch, preds):
                idx = b["idx"]
                r = store.rows[idx]
                _, msk = store.get(idx)
                for side, clean in store.boxes(idx).items():
                    if clean is None:
                        continue
                    rows.append({"modality": r["modality"], "patient": r["patient"], "level": lev,
                                 "dice": C.dice_bool(labels == side, msk == side)})
                if lk != sk:
                    continue
                if r["sam_eligible"]:
                    union.setdefault(r["patient"], []).append(C.dice_bool(labels > 0, msk > 0))
                if r["has_kidney"] == 0:
                    fp.append(int((labels > 0).sum()) >= D.MIN_AREA)
    s = C.val_metric(rows, select_level)
    s["union_dice"] = float(np.mean([np.mean(v) for v in union.values()])) if union else float("nan")
    s["fp_slice_rate"] = float(np.mean(fp)) if fp else float("nan")
    s["n_rows"] = len(rows)
    return s


def select_level_of(variant: str):
    return C.NO_BOX_LEVEL if variant == "plain" else 0.10


# ============================================================================= stamp
def check_stamp_and_write(cfg: Cfg, run_dir: Path, extra: dict) -> dict:
    """config.json: written on first start; later starts must match settings, data, norm and
    GPU type exactly, and the code (unless --allow-code-change with a reason)."""
    path = run_dir / "config.json"
    now = {"settings": stamp_of(cfg), **extra}
    if not path.exists():
        atomic_write_json(path, {**now, "created_utc": utc(), "code_changes": []})
        return now
    have = json.loads(path.read_text())
    diffs = [k for k in ("settings", "manifest_sha256", "gpu_type", "norm", "teacher", "partner")
             if have.get(k) != now.get(k)]
    if diffs:
        raise SystemExit(f"{run_dir}: refusing to resume -- changed {diffs}: "
                         f"{ {k: (have.get(k), now.get(k)) for k in diffs} }")
    if have.get("code") != now.get("code"):
        changed = sorted(f for f in now["code"] if have.get("code", {}).get(f) != now["code"][f])
        if not cfg.allow_code_change:
            raise SystemExit(f"{run_dir}: code changed since the run started ({changed}); "
                             f"resume only with --allow-code-change \"<reason>\"")
        have.setdefault("code_changes", []).append({"utc": utc(), "files": changed,
                                                    "reason": cfg.allow_code_change,
                                                    "new_hashes": {f: now["code"][f] for f in changed}})
        have["code"] = now["code"]
        atomic_write_json(path, have)
        log(f"code change accepted and logged: {changed} -- {cfg.allow_code_change}")
    return now


def startup(cfg: Cfg) -> dict:
    if cfg.variant not in C.VARIANTS:
        raise SystemExit(f"--variant must be one of {C.VARIANTS}")
    gpu = gpu_check(cfg.expect_gpu)
    if gpu.get("cuda"):
        log(f"GPU: {gpu['name']}  {gpu['total_gib']} GiB  capability {gpu['capability']}  "
            f"bf16 {gpu['bf16']}  CUDA {gpu['cuda_version']}  torch {gpu['torch']}")
    return {"gpu": gpu, "gpu_type": gpu.get("name", "cpu") if gpu.get("cuda") else "cpu",
            "manifest_sha256": D.manifest_sha256(Path(cfg.processed))}


def partner_dir(cfg: Cfg) -> Path:
    """The arm-3 run an arm-4 run is paired with: same parent folder, same seed."""
    return Path(cfg.run_dir).parent / f"arm3_plain_s{cfg.seed}"


def check_partner(cfg: Cfg, norm: C.Norm, manifest_sha: str) -> dict:
    """Arm 4 is arm 3 + one loss term: the same-seed arm-3 run must be finished and agree on
    every setting except the kd ones, the normalisation, the data and the initial weights
    (ARM4_CODE_PLAN.md invariant 2). Refuses, naming the field, otherwise."""
    pdir = partner_dir(cfg)
    rj_path, st_path = pdir / "run.json", pdir / "state.json"
    if not rj_path.exists() or not st_path.exists():
        raise SystemExit(f"REFUSING -- arm-4 partner {pdir} has no run.json / state.json (arm 3 seed {cfg.seed} not trained)")
    rj, st = json.loads(rj_path.read_text()), json.loads(st_path.read_text())
    probs = []
    if rj.get("status") != "TRAIN_OK":
        probs.append(f"status {rj.get('status')!r}, not TRAIN_OK")
    want = stamp_of(dataclasses.replace(cfg, kd=False))
    if rj.get("settings") != want:
        keys = sorted(k for k in set(want) | set(rj.get("settings") or {})
                      if (rj.get("settings") or {}).get(k) != want.get(k))
        probs.append(f"settings differ in {keys}")
    if rj.get("norm") != norm.as_dict():
        probs.append(f"normalisation {rj.get('norm')} != {norm.as_dict()}")
    if rj.get("manifest_sha256") != manifest_sha:
        probs.append("manifest sha256 differs")
    probe, _ = build_model(cfg, "cpu")
    digest = weights_digest(probe)
    del probe
    if st.get("initial_digest") != digest:
        probs.append("initial weights differ (state.json initial_digest)")
    if probs:
        raise SystemExit(f"REFUSING -- not paired with {pdir}: " + "; ".join(probs))
    return {"path": str(pdir.name), "best_sha256": rj.get("best_sha256"), "initial_digest": digest}


# ========================================================================== training
def train_step(model, opt, batch: dict, step: int, cfg: Cfg, device) -> dict:
    lr = C.poly_lr(step, total_steps(cfg), cfg.lr, cfg.poly_exponent)
    for gr in opt.param_groups:
        gr["lr"] = lr
    x = batch["x"].to(device, non_blocking=True)
    y = batch["label"].to(device, non_blocking=True).long()
    v = batch["valid"].to(device, non_blocking=True)
    with autocast_ctx(cfg, device):
        out = model(x)
    outs = float_outputs(out)
    loss = C.ds_loss(outs, y, v)
    total = loss["total"]
    kd = None
    if cfg.kd:                                # arm 4: + lambda x L_KD on the full-resolution output
        import student_kd as K
        full = outs[0] if isinstance(outs, (list, tuple)) else outs
        kd = K.kd_loss(full, batch["t_logit"].to(device, non_blocking=True),
                       batch["t_region"].to(device, non_blocking=True), v,
                       batch["t_has"].to(device, non_blocking=True), cfg.kd_temperature)
        if not torch.isfinite(kd["total"]):
            raise TrainFailed(f"non-finite distillation loss at step {step}: {float(kd['total'])}")
        total = total + cfg.kd_lambda * kd["total"]
    if not torch.isfinite(total):
        raise TrainFailed(f"non-finite loss at step {step}: {float(total)}")
    opt.zero_grad(set_to_none=True)
    total.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
    if not torch.isfinite(norm):
        raise TrainFailed(f"non-finite gradient norm at step {step}: {float(norm)}")
    opt.step()
    res = {"loss": float(total.detach()), "ce": float(loss["ce"].detach()),
           "dice": float(loss["dice"].detach()), "grad_norm": float(norm), "lr": lr}
    if kd is not None:
        mods = [batch["modality"][j] for j in kd["term_image"].tolist()]
        tv = kd["terms"].tolist()
        res.update(seg=float(loss["total"].detach()), kd=float(kd["total"].detach()),
                   kd_n_ct=sum(m == "ct" for m in mods), kd_n_mri=sum(m == "mri" for m in mods),
                   kd_sum_ct=sum(t for t, m in zip(tv, mods) if m == "ct"),
                   kd_sum_mri=sum(t for t, m in zip(tv, mods) if m == "mri"),
                   kd_images=kd["n_images_with_terms"], kd_empty=kd["n_empty_region"])
    return res


def run_training(cfg: Cfg, stores=None) -> int:
    run_dir = Path(cfg.run_dir)
    check_kd_cfg(cfg)
    if run_dir.name != run_dir_name(cfg):
        raise SystemExit(f"run directory {run_dir.name!r} must be named {run_dir_name(cfg)!r} "
                         f"(variant and seed)")
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.device)
    env = startup(cfg)
    rj_path = run_dir / "run.json"
    if rj_path.exists():
        rj = json.loads(rj_path.read_text())
        if rj.get("status") == "TRAIN_OK" and rj.get("epochs_completed") == cfg.epochs:
            if rj.get("settings") != stamp_of(cfg) or rj.get("manifest_sha256") != env["manifest_sha256"]:
                raise SystemExit(f"{run_dir}: finished run has different settings or data")
            log(f"{run_dir}: already complete ({cfg.epochs} epochs, best epoch {rj['best_epoch']}).")
            log("TRAIN_OK")
            return 0
    tr_store, va_store = stores if stores is not None else load_stores(cfg)
    norm = C.norm_from_buffer(tr_store.img)
    stamp_extra = {"manifest_sha256": env["manifest_sha256"], "gpu_type": env["gpu_type"],
                   "norm": norm.as_dict(), "code": code_hashes(cfg.kd)}
    tstore = partner = None
    if cfg.kd:
        import student_kd as K
        partner = check_partner(cfg, norm, env["manifest_sha256"])
        tstore = K.TeacherStore(cfg.teacher_dir, tr_store, cfg.kd_gate, cfg.threads, log)
        stamp_extra.update(teacher=tstore.identity, partner=partner)
        log(f"arm 4: paired with {partner['path']} (settings, norm, manifest, initial weights); "
            f"lambda {cfg.kd_lambda}, T {cfg.kd_temperature}, margin {cfg.kd_margin}, gate {cfg.kd_gate}")
    stamp = check_stamp_and_write(cfg, run_dir, stamp_extra)
    model, n_params = build_model(cfg, device)
    opt = make_optimizer(model, cfg)
    pe, spe, total = per_epoch(cfg), cfg.steps_per_epoch, total_steps(cfg)
    log(f"{cfg.variant} student, seed {cfg.seed}: {n_params:,} parameters; {spe} steps/epoch x "
        f"{cfg.epochs} = {total} steps at batch {cfg.batch}; norm mean {norm.mean:.4f} std {norm.std:.4f}")
    fields = metric_fields(cfg.variant, cfg.kd)
    sel_level = select_level_of(cfg.variant)
    val_items = D.ValItems(va_store, cfg.variant, norm, cfg.size)
    vload = val_loader(val_items, cfg)

    state_path = run_dir / "state.json"
    if not state_path.exists():
        atomic_write_json(state_path, {"initial_digest": weights_digest(model), "params": n_params})
    state = json.loads(state_path.read_text())
    metrics_path, last_path, best_path = run_dir / "metrics.csv", run_dir / "last.pt", run_dir / "best.pt"
    rows = read_metrics(metrics_path)

    def cpu_state() -> dict:
        return {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}

    def export_best(state_dict: dict, epoch: int, score: float) -> None:
        kd_part = ({"kd": kd_settings(cfg), "teacher_sha256": tstore.identity["teacher_sha256"]}
                   if cfg.kd else {})
        atomic_torch_save({"model": state_dict,
                           "arch": {"in_ch": C.in_channels(cfg.variant), "channels": list(cfg.channels),
                                    "n_classes": C.N_CLASSES, "deep_supervision": cfg.deep_supervision},
                           "variant": cfg.variant, "norm": norm.as_dict(), "size": cfg.size,
                           "epoch": epoch, "score": score, "seed": cfg.seed, **kd_part}, best_path)

    def commit(epoch: int, best: dict, rows_: list, best_changed: bool) -> None:
        """last.pt is the single source of truth, written atomically FIRST; metrics.csv and
        best.pt are exports of it. A job killed between the writes is repaired on resume."""
        atomic_torch_save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                           "epoch": epoch, "step": epoch * spe, "best_epoch": best["epoch"],
                           "best_score": best["score"], "best_model": best["state"],
                           "rows": rows_, "rng": rng_state()}, last_path)
        write_metrics(metrics_path, rows_, fields)
        if best_changed:
            export_best(best["state"], best["epoch"], best["score"])

    def best_file_matches(best: dict) -> bool:
        if not best_path.exists():
            return False
        bk = torch.load(best_path, map_location="cpu", weights_only=False)
        return (bk.get("epoch") == best["epoch"] and bk.get("score") == best["score"]
                and bk["model"].keys() == best["state"].keys()
                and all(torch.equal(bk["model"][k], best["state"][k]) for k in best["state"]))

    def val_row(epoch: int, extra: dict) -> dict:
        tv = time.time()
        s = validate(model, val_items, vload, device, sel_level)
        row = {"epoch": epoch, **extra}
        for k in val_fields(cfg.variant) + ["union_dice", "fp_slice_rate"]:
            row[k] = round(float(s[k]), 6)
        row["val_s"] = round(time.time() - tv, 1)
        row["peak_mem_gib"] = peak_gib(device)
        return row

    if last_path.exists():
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        set_rng_state(ck["rng"])
        done, rows = ck["epoch"], ck["rows"]
        best = {"epoch": ck["best_epoch"], "score": ck["best_score"], "state": ck["best_model"]}
        if [int(r["epoch"]) for r in rows] != list(range(done + 1)):
            raise SystemExit(f"last.pt rows {[r['epoch'] for r in rows]} are not 0..{done}")
        if ck["step"] != done * spe:
            raise SystemExit(f"last.pt step {ck['step']} != {done} x {spe}")
        if [int(r["epoch"]) for r in read_metrics(metrics_path)] != list(range(done + 1)):
            log("repairing metrics.csv from last.pt (the job ended between writes)")
        write_metrics(metrics_path, rows, fields)
        if not best_file_matches(best):
            log(f"repairing best.pt from last.pt (best epoch {best['epoch']})")
            export_best(best["state"], best["epoch"], best["score"])
        log(f"resuming after epoch {done}; best so far epoch {best['epoch']} ({best['score']:.4f})")
    else:
        if rows or best_path.exists():
            raise SystemExit(f"{run_dir}: metrics.csv / best.pt exist without last.pt -- inconsistent run folder")
        row0 = val_row(0, {"steps": 0})
        rows = [row0]
        done = 0
        best = {"epoch": 0, "score": row0["selection"], "state": cpu_state()}
        commit(0, best, rows, True)
        log(f"epoch 0 (untrained): selection {best['score']:.4f}")
    if done > cfg.epochs:
        raise SystemExit(f"{run_dir}: at epoch {done}, beyond the configured {cfg.epochs}")

    if done < cfg.epochs:
        if cfg.kd:
            ds = D.TrainItems(tr_store, cfg.variant, cfg.seed, norm, pe, done * pe, cfg.epochs * pe,
                              cfg.size, cfg.augment, cfg.max_level, teacher=tstore, kd_margin=cfg.kd_margin)
        else:
            ds = D.TrainItems(tr_store, cfg.variant, cfg.seed, norm, pe, done * pe, cfg.epochs * pe,
                              cfg.size, cfg.augment, cfg.max_level)
        it = iter(train_loader(ds, cfg))
        model.train()
        for epoch in range(done + 1, cfg.epochs + 1):
            te = time.time()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            sums = {"loss": 0.0, "ce": 0.0, "dice": 0.0, "grad_norm": 0.0}
            kd_sums = dict.fromkeys(("seg", "kd", "kd_n_ct", "kd_n_mri", "kd_sum_ct", "kd_sum_mri",
                                     "kd_images", "kd_empty"), 0.0)
            wait, n_mri, lr = 0.0, 0, 0.0
            for s in range(spe):
                step = (epoch - 1) * spe + s
                tw = time.time()
                batch = next(it)
                wait += time.time() - tw
                if batch["epoch"][0] != epoch or batch["pos"][0] != step * cfg.batch:
                    raise AssertionError(f"data/step misalignment at step {step}: "
                                         f"epoch {batch['epoch'][0]}, position {batch['pos'][0]}")
                n_mri += sum(m == "mri" for m in batch["modality"])
                r = train_step(model, opt, batch, step, cfg, device)
                for k in sums:
                    sums[k] += r[k]
                if cfg.kd:
                    for k in kd_sums:
                        kd_sums[k] += r[k]
                lr = r["lr"]
            train_s = time.time() - te
            extra = {"train_loss": sums["loss"] / spe, "train_ce": sums["ce"] / spe,
                     "train_dice": sums["dice"] / spe, "grad_norm": sums["grad_norm"] / spe,
                     "lr": lr, "steps": spe, "ct_draws": pe - n_mri, "mri_draws": n_mri,
                     "train_s": round(train_s, 1), "data_wait_s": round(wait, 1)}
            if cfg.kd:
                n_ct, n_mr = int(kd_sums["kd_n_ct"]), int(kd_sums["kd_n_mri"])
                extra.update(train_seg=kd_sums["seg"] / spe, train_kd=kd_sums["kd"] / spe,
                             kd_ct=kd_sums["kd_sum_ct"] / n_ct if n_ct else float("nan"),
                             kd_mri=kd_sums["kd_sum_mri"] / n_mr if n_mr else float("nan"),
                             kd_terms=n_ct + n_mr, kd_draw_frac=kd_sums["kd_images"] / pe)
            row = val_row(epoch, {k: (round(v, 6) if isinstance(v, float) else v) for k, v in extra.items()})
            model.train()
            row["epoch_s"] = round(time.time() - te, 1)
            rows.append(row)
            improved = row["selection"] > best["score"]
            if improved:
                best = {"epoch": epoch, "score": row["selection"], "state": cpu_state()}
            commit(epoch, best, rows, improved)
            kd_txt = (f"  kd {row['train_kd']:.4f} (ct {row['kd_ct']:.4f} / mri {row['kd_mri']:.4f}, "
                      f"{row['kd_terms']} terms, {int(kd_sums['kd_empty'])} empty)" if cfg.kd else "")
            log(f"epoch {epoch}/{cfg.epochs}: loss {row['train_loss']:.4f}  val {row['selection']:.4f}  "
                f"best {best['epoch']} ({best['score']:.4f})  union {row['union_dice']:.4f}  "
                f"fp {row['fp_slice_rate']:.3f}  {row['epoch_s']:.0f}s  wait {wait:.0f}s  "
                f"peak {row['peak_mem_gib']} GiB" + kd_txt)
            if cfg.stop_after_epoch is not None and epoch == cfg.stop_after_epoch and epoch < cfg.epochs:
                log(f"stopping after epoch {epoch} (test hook)")
                return 0

    if weights_digest(model) == state["initial_digest"]:
        raise TrainFailed("weights never changed -- training had no effect")
    if not best_file_matches(best):
        raise TrainFailed("best.pt does not hold the best epoch's weights")
    best_epoch, best_score = best["epoch"], best["score"]
    sel = [float(r["selection"]) for r in rows]
    late = best_epoch >= int(0.9 * cfg.epochs)
    if cfg.epochs >= 100:
        late = late or (np.mean(sel[81:101]) - np.mean(sel[61:81]) > 0.002)
    if late:
        log("WARNING: still improving at the end of the budget (best epoch "
            f"{best_epoch}); decide before the remaining runs (plan §3.5)")
    kd_part = ({"arm": 4, "kd": kd_settings(cfg), "teacher": tstore.identity, "partner": partner}
               if cfg.kd else {})
    atomic_write_json(rj_path, {
        "status": "TRAIN_OK", "epochs_completed": cfg.epochs, "best_epoch": best_epoch,
        "best_score": best_score, "best_sha256": sha256_file(best_path), "late_best": bool(late),
        "variant": cfg.variant, "seed": cfg.seed, "params": n_params, "norm": norm.as_dict(),
        "settings": stamp["settings"], "manifest_sha256": env["manifest_sha256"], "gpu": env["gpu"],
        "gpu_type": env["gpu_type"], "steps_per_epoch": spe, "total_steps": total,
        "host": platform.node(), "finished_utc": utc(), **kd_part})
    log(f"best epoch {best_epoch}: val selection {best_score:.4f}")
    log("TRAIN_OK")
    return 0


# ========================================================================= preflight
def _items_both_kidneys(store: D.SliceStore, n: int) -> list[int]:
    """n slices with both kidneys boxed, half CT half MRI, evenly spaced in (case, z) order."""
    out = []
    for mod, k in (("ct", n - n // 2), ("mri", n // 2)):
        cand = [i for i, r in enumerate(store.rows) if r["modality"] == mod
                and r["rk_box"] is not None and r["lk_box"] is not None]
        if len(cand) < k:
            raise SystemExit(f"preflight: only {len(cand)} {mod} slices with both kidneys")
        out += [cand[int(j * len(cand) / k)] for j in range(k)]
    return out


def _fixed_batch(store, idxs, variant, norm, size):
    """Unaugmented inputs for fixed slices (clean boxes for the box variant)."""
    xs, ys, vs = [], [], []
    for idx in idxs:
        img, msk = store.get(idx)
        boxes = store.boxes(idx) if variant == "box" else None
        x, v, g = C.net_input(img, norm, variant, boxes, size)
        xs.append(x)
        ys.append(C.to_net_label(msk, g))
        vs.append(v)
    return {"x": torch.stack(xs), "label": torch.stack(ys), "valid": torch.stack(vs)}


def _batch_dice(model, store, idxs, variant, norm, size, device) -> float:
    xs, gs = [], []
    for idx in idxs:
        img, _ = store.get(idx)
        x, _, g = C.net_input(img, norm, variant, store.boxes(idx) if variant == "box" else None, size)
        xs.append(x)
        gs.append(g)
    preds = C.predict_native_batch(model, xs, gs, device)
    vals = []
    for idx, (labels, _) in zip(idxs, preds):
        _, msk = store.get(idx)
        for side, clean in store.boxes(idx).items():
            if clean is not None:
                vals.append(C.dice_bool(labels == side, msk == side))
    return float(np.mean(vals))


@torch.no_grad()
def _probe_loss(model, batch, cfg, device) -> float:
    was = model.training
    model.train()                  # DS outputs; InstanceNorm has no running stats
    try:
        out = model(batch["x"].to(device))
        return float(C.ds_loss(float_outputs(out), batch["label"].to(device).long(),
                               batch["valid"].to(device))["total"])
    finally:
        model.train(was)


def _ceiling(va_store, size: int) -> dict:
    """Letterbox ceiling: perfect one-hot logits through the back-mapping only."""
    per = {}                          # (modality, resized) -> patient -> [dice]
    exact_fail = []
    for idx, r in enumerate(va_store.rows):
        if not r["sam_eligible"]:
            continue
        _, msk = va_store.get(idx)
        g = C.letterbox_geometry(r["h"], r["w"], size)
        lab = C.to_net_label(msk, g)
        logits = torch.nn.functional.one_hot(lab, C.N_CLASSES).permute(2, 0, 1).float() * 10.0
        labels = C.to_native_logits(logits, g).argmax(0).numpy()
        resized = (g["h2"], g["w2"]) != (g["h"], g["w"])
        for side, clean in va_store.boxes(idx).items():
            if clean is None:
                continue
            d = C.dice_bool(labels == side, msk == side)
            per.setdefault((r["modality"], resized), {}).setdefault(r["patient"], []).append(d)
            if not resized and d != 1.0:
                exact_fail.append(f"{r['case']} z={r['z']} side {side}: {d}")
    out = {"exact_fail": exact_fail[:10], "n_exact_fail": len(exact_fail)}
    for (mod, resized), pats in per.items():
        key = f"{mod}_{'resized' if resized else 'native512'}"
        out[key] = float(np.mean([np.mean(v) for v in pats.values()]))
        out[f"n_inst_{key}"] = sum(len(v) for v in pats.values())
    return out


def _aug_examples(path: Path, items: list) -> None:
    from PIL import Image, ImageDraw
    tiles = []
    for it in items[:16]:
        x = it["x"][0].numpy()
        lo, hi = np.percentile(x, (1, 99))
        g = np.clip((x - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
        rgb = np.stack([g, g, g], -1)
        lab = it["label"].numpy()
        for side, col in ((1, (255, 60, 60)), (2, (60, 160, 255))):
            m = lab == side
            edge = m & ~(np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1))
            rgb[edge] = col
        if it["x"].shape[0] == 3:
            for c, col in ((1, (255, 200, 0)), (2, (0, 255, 120))):
                b = it["x"][c].numpy() > 0.5
                edge = b & ~(np.roll(b, 1, 0) & np.roll(b, -1, 0) & np.roll(b, 1, 1) & np.roll(b, -1, 1))
                rgb[edge] = col
        tiles.append(Image.fromarray(rgb).resize((256, 256)))
    sheet = Image.new("RGB", (4 * 256, 4 * 256))
    for k, t in enumerate(tiles):
        sheet.paste(t, ((k % 4) * 256, (k // 4) * 256))
    ImageDraw.Draw(sheet).text((4, 4), "red = right kidney, blue = left (patient's right on image LEFT)",
                               fill=(255, 255, 0))
    sheet.save(path)


def run_preflight(cfg: Cfg, stores=None) -> int:
    if cfg.kd:
        return run_preflight_kd(cfg, stores)
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.device)
    pf: dict = {"started_utc": utc(), "checks": {}}
    fails: list[str] = []

    def check(name: str, ok: bool, info) -> None:
        pf["checks"][name] = {"ok": bool(ok), "info": info}
        log(f"[{'PASS' if ok else 'FAIL'}] {name}: {info}")
        if not ok:
            fails.append(name)

    # 1. environment
    env = startup(dataclasses.replace(cfg, variant="plain"))
    pf["gpu"] = env["gpu"]
    if device.type == "cuda":
        check("environment", torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
              {"device": torch.cuda.get_device_name(0)})
    # 2. data
    tr_store, va_store = stores if stores is not None else load_stores(cfg)
    norm = C.norm_from_buffer(tr_store.img)
    pf["data"] = {"train_counts": tr_store.counts, "val_counts": va_store.counts,
                  "load_s": round(tr_store.load_s + va_store.load_s, 1),
                  "ram_gib": round(tr_store.gib + va_store.gib, 2), "norm": norm.as_dict(),
                  "train_sizes": tr_store.size_table(), "val_sizes": va_store.size_table()}
    log(f"data: {json.dumps(pf['data'], indent=1)}")
    # 3. letterbox ceiling
    ceil_ = _ceiling(va_store, cfg.size)
    resized_vals = [v for k, v in ceil_.items() if k.endswith("_resized")]
    check("letterbox_ceiling", ceil_["n_exact_fail"] == 0 and all(v >= cfg.pf_ceiling_fail for v in resized_vals),
          ceil_)
    # 4. pipeline identity + augmentation sanity
    pe = per_epoch(cfg)
    for variant in C.VARIANTS:
        n_id = 20
        plain_ds = D.TrainItems(tr_store, variant, cfg.seed, norm, pe, 0, n_id, cfg.size, augment=False)
        ok = True
        for k in range(n_id):
            it = plain_ds[k]
            img, _ = tr_store.get(it["idx"])
            ref, _, _ = C.to_net_image(img, norm, cfg.size)
            ok &= bool(torch.equal(it["x"][:1], ref))
        aug_ds = D.TrainItems(tr_store, variant, cfg.seed, norm, pe, 0, cfg.pf_aug_items, cfg.size, augment=True)
        items = [aug_ds[k] for k in range(cfg.pf_aug_items)]
        labels_ok = all(set(torch.unique(i["label"]).tolist()) <= {0, 1, 2} for i in items)
        finite = all(bool(torch.isfinite(i["x"]).all()) for i in items)
        box_ok = variant == "plain" or all(set(torch.unique(i["x"][1:]).tolist()) <= {0.0, 1.0} for i in items)
        check(f"pipeline_{variant}", ok and labels_ok and finite and box_ok,
              {"unaugmented_equals_net_input": ok, "labels_in_012": labels_ok, "finite": finite,
               "box_channels_binary": box_ok})
        _aug_examples(run_dir / f"aug_examples_{variant}.png", items)
    # 5. can it learn (plain, no augmentation, constant lr)
    idxs = _items_both_kidneys(tr_store, cfg.pf_learn_slices)
    fixed = _fixed_batch(tr_store, idxs, "plain", norm, cfg.size)
    lc = dataclasses.replace(cfg, variant="plain")
    model, _ = build_model(lc, device)
    opt = make_optimizer(model, lc)
    l0, d0 = _probe_loss(model, fixed, lc, device), _batch_dice(model, tr_store, idxs, "plain", norm, cfg.size, device)
    model.train()
    for _ in range(cfg.pf_learn_steps):
        x, y, v = fixed["x"].to(device), fixed["label"].to(device), fixed["valid"].to(device)
        with autocast_ctx(lc, device):
            out = model(x)
        loss = C.ds_loss(float_outputs(out), y.long(), v)["total"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), lc.clip)
        opt.step()
    l1, d1 = _probe_loss(model, fixed, lc, device), _batch_dice(model, tr_store, idxs, "plain", norm, cfg.size, device)
    check("can_learn", l1 <= cfg.pf_learn_loss_ratio * l0 and d1 - d0 >= cfg.pf_learn_dice_gain,
          {"loss": [round(l0, 4), round(l1, 4)], "dice": [round(d0, 4), round(d1, 4)]})
    del model, opt
    # 6. real steps + timing, then 7. estimate
    timing = {}
    for variant, n_steps in (("plain", cfg.pf_steps_plain), ("box", cfg.pf_steps_box)):
        vc = dataclasses.replace(cfg, variant=variant)
        model, _ = build_model(vc, device)
        opt = make_optimizer(model, vc)
        probe_ds = D.TrainItems(tr_store, variant, cfg.seed, norm, pe, 0, cfg.pf_probe, cfg.size, augment=False)
        probe = D.collate_train([probe_ds[k] for k in range(cfg.pf_probe)])
        p0 = _probe_loss(model, probe, vc, device)
        ds = D.TrainItems(tr_store, variant, cfg.seed, norm, pe, 0, n_steps * cfg.batch, cfg.size, cfg.augment)
        it = iter(train_loader(ds, vc))
        model.train()
        step_s, waits, finite = [], [], True
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for s in range(n_steps):
            tw = time.time()
            batch = next(it)
            ts = time.time()
            try:
                train_step(model, opt, batch, s, vc, device)
            except TrainFailed as e:
                finite = False
                log(f"non-finite in preflight {variant}: {e}")
                break
            if device.type == "cuda":
                torch.cuda.synchronize()
            step_s.append(time.time() - ts)
            waits.append(ts - tw)
        p1 = _probe_loss(model, probe, vc, device)
        skip = min(len(step_s) // 10, 100) if variant == "plain" else min(len(step_s) // 4, 50)
        mean_step = float(np.mean(step_s[skip:])) if len(step_s) > skip else float("nan")
        mean_wait = float(np.mean(waits[skip:])) if len(waits) > skip else float("nan")
        items = D.ValItems(va_store, variant, norm, cfg.size)
        tv = time.time()
        vs = validate(model, items, val_loader(items, vc), device, select_level_of(variant))
        val_s = time.time() - tv
        frac = mean_wait / (mean_step + mean_wait) if math.isfinite(mean_step + mean_wait) else float("nan")
        timing[variant] = {"s_per_step": mean_step, "wait_per_step": mean_wait, "wait_fraction": frac,
                           "peak_gib": peak_gib(device), "val_s": val_s, "probe_loss": [p0, p1],
                           "val_selection_after": vs["selection"]}
        ok = finite and math.isfinite(mean_step) and (variant == "box" or p1 < p0)
        check(f"real_steps_{variant}", ok, timing[variant])
        if math.isfinite(frac) and frac > cfg.pf_wait_warn:
            log(f"WARNING: data loading is {timing[variant]['wait_fraction']:.0%} of step time ({variant})")
        del model, opt, it
    run_h = {v: cfg.epochs * (cfg.steps_per_epoch * timing[v]["s_per_step"] + timing[v]["val_s"]) / 3600
             for v in timing}
    worst = max(run_h.values())
    rec_h = max(6, int(math.ceil(1.5 * 2 * worst)))
    pf["estimate"] = {"per_run_hours": run_h, "two_runs_hours": 2 * worst,
                      "recommended_ARM3_TRAIN_TIME": f"{rec_h:02d}:00:00"}
    pf["finished_utc"] = utc()
    pf["passed"] = not fails
    atomic_write_json(run_dir / "preflight.json", pf)
    log(f"estimate: {json.dumps(pf['estimate'])}")
    if fails:
        log(f"failed checks: {fails}")
        log("PREFLIGHT_FAILED")
        return 1
    log(f"set ARM3_TRAIN_TIME = \"{rec_h:02d}:00:00\" in gen_sbatch_arm3.py and regenerate")
    log("PREFLIGHT_OK")
    return 0


# ===================================================================== arm-4 preflight
ARM4_CFG_FIELDS = tuple(f.name for f in dataclasses.fields(Cfg)
                        if f.name in KD_SETTINGS or f.name == "teacher_dir" or f.name.startswith("pf_kd_"))


@contextlib.contextmanager
def one_thread():
    """Training items are made in DataLoader workers with ONE torch thread (D.worker_init).
    Multi-threaded CPU reductions round differently (measured: about half of augmented images
    differ in the last bits between 1 and 8 threads), so items are compared under the training
    condition."""
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(n)


def item_digests(ds, positions) -> dict:
    import arm3_reference as R
    with one_thread():
        return {str(q): R.item_digest(ds[q]) for q in positions}


def item_mismatch(mine: dict, ref: dict) -> dict:
    """Which fields differ, over the positions that differ (empty = identical)."""
    diff = [q for q in ref if mine.get(q) != ref[q]]
    names = ("idx", "epoch", "i", "x", "label", "valid")
    return {"positions": len(diff), **{n: sum(mine.get(q, [None] * 6)[k] != ref[q][k] for q in diff)
                                       for k, n in enumerate(names)}}


def arm3_overrides(cfg: Cfg, variant: str) -> dict:
    """The arm-3 configuration that corresponds to cfg, as keyword arguments the FROZEN Cfg
    accepts (an unknown key would make the frozen Cfg raise)."""
    d = dataclasses.asdict(dataclasses.replace(cfg, variant=variant, kd=False))
    return json.loads(json.dumps({k: v for k, v in d.items() if k not in ARM4_CFG_FIELDS}))


def _kd_alignment(item: dict) -> list:
    """Per used side with ground truth present: (modality, Dice(teacher > 0 & valid, GT),
    GT pixels inside the region, GT pixels)."""
    lab, out = item["label"].long(), []
    for c in (0, 1):
        if not bool(item["t_has"][c]):
            continue
        gt = lab == c + 1
        n = int(gt.sum())
        if n == 0:
            continue
        pred = (item["t_logit"][c].float() > 0) & item["valid"]
        out.append((item["modality"], C.dice_bool(pred.numpy(), gt.numpy()),
                    int((item["t_region"][c] & gt).sum()), n))
    return out


def _kd_examples(path: Path, items: list) -> None:
    from PIL import Image, ImageDraw

    def edge(m):
        return m & ~(np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1))
    tiles = []
    for it in items[:16]:
        x = it["x"][0].numpy()
        lo, hi = np.percentile(x, (1, 99))
        g = np.clip((x - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
        rgb = np.stack([g, g, g], -1)
        lab = it["label"].numpy()
        for c, (gcol, tcol) in enumerate((((255, 60, 60), (255, 230, 0)), ((60, 160, 255), (0, 255, 120)))):
            if bool(it["t_has"][c]):
                rgb[edge(it["t_region"][c].numpy())] = (255, 255, 255)
                rgb[edge(it["t_logit"][c].float().numpy() > 0)] = tcol
            rgb[edge(lab == c + 1)] = gcol
        tiles.append(Image.fromarray(rgb).resize((256, 256)))
    sheet = Image.new("RGB", (4 * 256, 4 * 256))
    for k, t in enumerate(tiles):
        sheet.paste(t, ((k % 4) * 256, (k // 4) * 256))
    ImageDraw.Draw(sheet).text((4, 4), "GT red/blue; teacher p>0.5 yellow/green; region white", fill=(255, 255, 0))
    sheet.save(path)


def run_preflight_kd(cfg: Cfg, stores=None, frozen_dir: Path | None = None) -> int:
    """ARM4_CODE_PLAN.md §8: P1-P11. Ends PREFLIGHT_OK / PREFLIGHT_FAILED."""
    import arm3_reference as R
    import student_kd as K
    check_kd_cfg(cfg)
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.device)
    frozen_dir = Path(frozen_dir) if frozen_dir else Path(__file__).resolve().parent / "frozen_arm3"
    pdir = partner_dir(cfg)
    pf: dict = {"started_utc": utc(), "checks": {}, "kd": kd_settings(cfg), "partner": str(pdir)}
    fails: list[str] = []

    def check(name: str, ok: bool, info) -> None:
        pf["checks"][name] = {"ok": bool(ok), "info": info}
        log(f"[{'PASS' if ok else 'FAIL'}] {name}: {info}")
        if not ok:
            fails.append(name)

    def finish() -> int:
        pf["finished_utc"] = utc()
        pf["passed"] = not fails
        atomic_write_json(run_dir / "preflight.json", pf)
        if fails:
            log(f"failed checks: {fails}")
            log("PREFLIGHT_FAILED")
            return 1
        log(f"set ARM4_TRAIN_TIME = \"{pf['estimate']['recommended_ARM4_TRAIN_TIME']}\" in "
            "gen_sbatch_arm4_train.py and regenerate")
        log("PREFLIGHT_OK")
        return 0

    # P1 environment
    env = startup(cfg)
    pf["gpu"] = env["gpu"]
    if device.type == "cuda":
        check("P1_environment", torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
              {"device": torch.cuda.get_device_name(0)})
    # P2 data, against the arm-3 partner
    tr_store, va_store = stores if stores is not None else load_stores(cfg)
    norm = C.norm_from_buffer(tr_store.img)
    prj = json.loads((pdir / "run.json").read_text()) if (pdir / "run.json").exists() else {}
    check("P2_data", prj.get("norm") == norm.as_dict() and prj.get("manifest_sha256") == env["manifest_sha256"],
          {"train_counts": tr_store.counts, "val_counts": va_store.counts, "norm": norm.as_dict(),
           "partner_norm": prj.get("norm"), "ram_gib": round(tr_store.gib + va_store.gib, 2)})
    # P3 frozen arm-3 identity (subprocess)
    pe, total = per_epoch(cfg), cfg.epochs * per_epoch(cfg)
    positions = R.spread_positions(total, cfg.pf_kd_frozen)
    frozen_probs = R.check_frozen(frozen_dir)
    pcfg = json.loads((pdir / "config.json").read_text()) if (pdir / "config.json").exists() else {}
    info3: dict = {"frozen_files": frozen_probs or "7 files match the recorded sha256",
                   "partner_code_matches_frozen": pcfg.get("code") == R.FROZEN_SHA256,
                   "positions": len(positions)}
    ok3 = not frozen_probs and info3["partner_code_matches_frozen"]
    if not frozen_probs:
        for variant in C.VARIANTS:
            ref = R.frozen_items(frozen_dir, cfg.processed, variant, cfg.seed, positions, pe, cfg.size,
                                 cfg.augment, cfg.max_level, cfg.expected_counts)
            ds = D.TrainItems(tr_store, variant, cfg.seed, norm, pe, 0, total, cfg.size, cfg.augment, cfg.max_level)
            mine = item_digests(ds, positions)
            bad = item_mismatch(mine, ref["items"])
            info3[f"items_{variant}_identical"] = not bad["positions"]
            if bad["positions"]:
                info3[f"items_{variant}_mismatch"] = bad
            info3[f"norm_{variant}_identical"] = ref["norm"] == norm.as_dict()
            ok3 &= not bad["positions"] and ref["norm"] == norm.as_dict()
        ovs = [arm3_overrides(cfg, v) for v in C.VARIANTS]
        fs = R.frozen_stamps(frozen_dir, ovs)
        mine = []
        for v in C.VARIANTS:
            c3 = dataclasses.replace(cfg, variant=v, kd=False)
            mine.append({"stamp": stamp_of(c3), "metric_fields": metric_fields(v), "run_dir_name": run_dir_name(c3)})
        info3["stamps_identical"] = json.loads(json.dumps(mine)) == fs
        ok3 &= info3["stamps_identical"]
    check("P3_frozen_arm3_identity", ok3, info3)
    # P4 teacher store
    try:
        tstore = K.TeacherStore(cfg.teacher_dir, tr_store, cfg.kd_gate, cfg.threads, log)
    except SystemExit as e:
        check("P4_teacher_store", False, str(e))
        pf["estimate"] = {"recommended_ARM4_TRAIN_TIME": None}
        return finish()
    gated = {m: tstore.n_gated[m] / tstore.n_kidneys[m] for m in tstore.n_kidneys}
    check("P4_teacher_store", max(gated.values()) <= cfg.pf_kd_gate_fail,
          {**tstore.identity, "gated_fraction": gated, "load_s": round(tstore.load_s, 1), "gib": round(tstore.gib, 2)})
    if max(gated.values()) > cfg.pf_kd_gate_warn:
        log(f"WARNING: more than {cfg.pf_kd_gate_warn:.0%} of a modality's kidneys are gated: {gated}")
    # P5-P7 alignment and pairing on positions that carry teacher terms
    un = D.TrainItems(tr_store, "plain", cfg.seed, norm, pe, 0, total, cfg.size, False, cfg.max_level,
                      teacher=tstore, kd_margin=cfg.kd_margin)
    au = D.TrainItems(tr_store, "plain", cfg.seed, norm, pe, 0, total, cfg.size, True, cfg.max_level,
                      teacher=tstore, kd_margin=cfg.kd_margin)
    a3 = D.TrainItems(tr_store, "plain", cfg.seed, norm, pe, 0, total, cfg.size, True, cfg.max_level)
    sel, q = [], 0
    al = {"un": [], "au": []}
    paired, aug_items = True, []
    with one_thread():                         # as in the DataLoader workers
        while len(sel) < cfg.pf_kd_items and q < total:
            if bool(un[q]["t_has"].any()):
                sel.append(q)
            q += 1
        for q in sel:
            iu, ia, i3 = un[q], au[q], a3[q]
            al["un"] += _kd_alignment(iu)
            al["au"] += _kd_alignment(ia)
            aug_items.append(ia)
            paired &= (iu["idx"] == ia["idx"] == i3["idx"] and torch.equal(ia["x"], i3["x"])
                       and torch.equal(ia["label"], i3["label"]) and torch.equal(ia["valid"], i3["valid"]))

    def summ(rows):
        out = {}
        for m in ("ct", "mri"):
            rs = [r for r in rows if r[0] == m]
            out[m] = {"n": len(rs), "dice": float(np.mean([r[1] for r in rs])) if rs else float("nan"),
                      "inside": (sum(r[2] for r in rs) / sum(r[3] for r in rs)) if rs else float("nan")}
        return out
    s5, s6 = summ(al["un"]), summ(al["au"])
    check("P5_alignment_unaugmented", all(v["n"] > 0 and v["dice"] >= cfg.pf_kd_align and v["inside"] == 1.0
                                          for v in s5.values()), {"items": len(sel), **s5})
    check("P6_alignment_augmented", all(v["n"] > 0 and v["dice"] >= cfg.pf_kd_align and v["inside"] >= cfg.pf_kd_inside_aug
                                        for v in s6.values()), s6)
    try:
        partner = check_partner(cfg, norm, env["manifest_sha256"])
        p_ok, p_info = True, partner
    except SystemExit as e:
        p_ok, p_info = False, str(e)
    check("P7_pairing", paired and p_ok, {"items_identical_to_arm3": paired, "partner": p_info})
    # P8 picture, P9 softness
    _kd_examples(run_dir / "kd_examples.png", aug_items)
    pf["softness"] = K.softness(tstore, sorted({un[q]["idx"] for q in sel}), cfg.kd_margin)
    log(f"P9 softness (share of region pixels with teacher p in (0.05, 0.95)): {pf['softness']}")
    # P10 real steps
    model, _ = build_model(cfg, device)
    opt = make_optimizer(model, cfg)
    probe = D.collate_train([un[q] for q in sel[:cfg.pf_probe]])
    p0 = _probe_loss(model, probe, cfg, device)
    n_steps = cfg.pf_kd_steps
    ds = D.TrainItems(tr_store, "plain", cfg.seed, norm, pe, 0, n_steps * cfg.batch, cfg.size, cfg.augment,
                      cfg.max_level, teacher=tstore, kd_margin=cfg.kd_margin)
    it = iter(train_loader(ds, cfg))
    model.train()
    segs, kds, step_s, waits, finite = [], [], [], [], True
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for s in range(n_steps):
        tw = time.time()
        batch = next(it)
        ts = time.time()
        try:
            r = train_step(model, opt, batch, s, cfg, device)
        except TrainFailed as e:
            finite = False
            log(f"non-finite in the arm-4 preflight: {e}")
            break
        if device.type == "cuda":
            torch.cuda.synchronize()
        step_s.append(time.time() - ts)
        waits.append(ts - tw)
        segs.append(r["seg"])
        kds.append(r["kd"])
    p1 = _probe_loss(model, probe, cfg, device)
    half = len(segs) // 2
    ratio = (cfg.kd_lambda * float(np.mean(kds[half:])) / float(np.mean(segs[half:]))
             if len(segs) >= 2 else float("nan"))
    skip = min(len(step_s) // 10, 100)
    mean_step = float(np.mean(step_s[skip:])) if len(step_s) > skip else float("nan")
    mean_wait = float(np.mean(waits[skip:])) if len(waits) > skip else float("nan")
    items = D.ValItems(va_store, "plain", norm, cfg.size)
    tv = time.time()
    vs = validate(model, items, val_loader(items, cfg), device, select_level_of("plain"))
    val_s = time.time() - tv
    frac = mean_wait / (mean_step + mean_wait) if math.isfinite(mean_step + mean_wait) else float("nan")
    lo, hi = cfg.pf_kd_ratio
    info10 = {"steps": len(segs), "s_per_step": mean_step, "wait_per_step": mean_wait, "wait_fraction": frac,
              "peak_gib": peak_gib(device), "val_s": val_s, "probe_seg_loss": [p0, p1],
              "kd_over_seg_ratio": ratio, "ratio_band": [lo, hi],
              "seg_first_last": [segs[0], segs[-1]] if segs else None,
              "kd_first_last": [kds[0], kds[-1]] if kds else None, "val_selection_after": vs["selection"]}
    check("P10_real_steps", finite and math.isfinite(mean_step) and p1 < p0 and math.isfinite(ratio) and lo <= ratio <= hi,
          info10)
    if math.isfinite(frac) and frac > cfg.pf_wait_warn:
        log(f"WARNING: data loading is {frac:.0%} of step time")
    # P11 estimate
    run_h = cfg.epochs * (cfg.steps_per_epoch * mean_step + val_s) / 3600 if math.isfinite(mean_step) else float("nan")
    rec = f"{max(6, int(math.ceil(1.5 * run_h))):02d}:00:00" if math.isfinite(run_h) else None
    pf["estimate"] = {"per_run_hours": run_h, "recommended_ARM4_TRAIN_TIME": rec}
    log(f"estimate: {json.dumps(pf['estimate'])}")
    return finish()


# ============================================================================== CLI
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--processed", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--variant", choices=C.VARIANTS, default="plain")
    p.add_argument("--seed", type=int, default=643)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--steps-per-epoch", type=int, default=1000)
    p.add_argument("--batch", type=int, default=12)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--device", default="cuda")
    p.add_argument("--expect-gpu", default=None)
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--allow-code-change", default=None, metavar="REASON")
    p.add_argument("--kd", action="store_true", help="arm 4: add the distillation term (all --kd-* required)")
    p.add_argument("--teacher-dir", default=None)
    p.add_argument("--kd-lambda", type=float, default=None)
    p.add_argument("--kd-temperature", type=float, default=None)
    p.add_argument("--kd-margin", type=float, default=None)
    p.add_argument("--kd-gate", type=float, default=None)
    a = p.parse_args(argv)
    kd_args = {"teacher_dir": a.teacher_dir, "kd_lambda": a.kd_lambda, "kd_temperature": a.kd_temperature,
               "kd_margin": a.kd_margin, "kd_gate": a.kd_gate}
    if a.kd:
        missing = [k for k, v in kd_args.items() if v is None]
        if missing:
            p.error(f"--kd needs every distillation setting given explicitly; missing {missing}")
    elif any(v is not None for v in kd_args.values()):
        p.error("distillation settings given without --kd (this would silently train arm 3)")
    cfg = Cfg(processed=a.processed, run_dir=a.run_dir, variant=a.variant, seed=a.seed,
              epochs=a.epochs, steps_per_epoch=a.steps_per_epoch, batch=a.batch, lr=a.lr,
              workers=a.workers, device=a.device, expect_gpu=a.expect_gpu,
              allow_code_change=a.allow_code_change, kd=a.kd,
              **{k: v for k, v in kd_args.items() if v is not None})
    try:
        return run_preflight(cfg) if a.preflight else run_training(cfg)
    except TrainFailed as e:
        diag = Path(cfg.run_dir) / "train_failed.json"
        try:
            atomic_write_json(diag, {"error": str(e), "utc": utc()})
        except OSError:
            pass
        log(f"!! {e}")
        log("TRAIN_FAILED")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
