#!/usr/bin/env python
"""Arm 2: fine-tune SAM ViT-B with LoRA (q, v; all 12 encoder blocks) + the full mask decoder
on AMOS train, select on AMOS val (DECISIONS.md D33-D34, ARM2_CODE_PLAN.md).

Modes (one per SLURM step):
  --preflight     identity vs SamPredictor, can-it-learn, LR sanity, timing/memory -> preflight.json
  (default)       train for --epochs, resumable at epoch boundaries          -> TRAIN_OK
  --finalize      merge the best epoch's LoRA into a plain ViT-B checkpoint + merge check
  --check-merged  verify a merged checkpoint against its run.json (used by the eval scripts)

Everything that turns a stored slice into a mask follows arm 1 step for step (sam_train_core).
The logic lives in run_preflight / run_training / finalize / check_merged, so the CPU tests run
the same code on a tiny SAM; main() only parses arguments.

--modality ct / mri (D70-D71, ARM2_MOD_CODE_PLAN.md): a single-modality teacher. Train and val
are restricted to that modality, draws are uniform within it, selection is its val Dice; the
default (both) is arm 2 unchanged.
"""
from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
import gc
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
import eval_sam as E  # noqa: E402
import sam_metrics as M  # noqa: E402
import sam_lora as L  # noqa: E402
import sam_train_core as C  # noqa: E402
import train_data as D  # noqa: E402

# The off-the-shelf ViT-B checkpoint every arm-2 run starts from (verified on Prajna, D32).
BASE_SHA256 = "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912"
BASE_BYTES = 375042383
CODE_FILES = ("sam_lora.py", "sam_train_core.py", "train_data.py", "train_sam_lora.py",
              "sam_metrics.py", "eval_sam.py")
A100_MIN_GIB = 75.0          # A100-80GB reports ~79.3 GiB; A100-40GB ~39.4 GiB
METRIC_FIELDS = ["epoch", "train_loss", "train_dice", "train_bce", "train_iou", "grad_norm",
                 "ct@0.00", "mri@0.00", "ct@0.10", "mri@0.10", "selection",
                 "nsd3_ct@0.10", "nsd3_mri@0.10", "lr_lora", "lr_dec", "steps",
                 "ct_draws", "mri_draws", "epoch_s", "train_s", "val_s", "data_wait_s",
                 "peak_mem_gib"]
# The modalities a run trains and selects on (D71); "both" is arm 2.
MODALITY_SETS = {"both": ("ct", "mri"), "ct": ("ct",), "mri": ("mri",)}
VAL_COLUMNS = ("ct@0.00", "mri@0.00", "ct@0.10", "mri@0.10", "nsd3_ct@0.10", "nsd3_mri@0.10")


class TrainFailed(RuntimeError):
    pass


# ============================================================================ config
@dataclass
class Cfg:
    processed: str = ""
    checkpoint: str = ""
    rank: int = 4
    alpha: float = 4.0
    run_dir: str = ""
    merged_out: str = ""
    epochs: int = 35
    batch: int = 8
    accum: int = 1
    lr_lora: float = 5e-4
    lr_dec: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    warmup: int = 250
    lr_floor: float = 1e-6
    clip: float = 1.0
    seed: int = 643
    modality: str = "both"               # both (arm 2) | ct | mri (D71)
    workers: int = 10
    val_batch: int = 8
    device: str = "cuda"
    amp: bool = True
    grad_checkpoint: bool = False
    expect_gpu: str | None = None
    n_draws: int | None = None           # default: len(train) = 8,295
    img_size: int = 1024
    max_level: float = D.TRAIN_MAX_LEVEL
    val_levels: tuple = D.VAL_LEVELS
    select_level: float = 0.10
    nsd: bool = True
    expect_blocks: int | None = 12
    expect_dim: int | None = 768
    check_base_sha: bool = True
    expected_slices: dict | None = field(default_factory=lambda: copy.deepcopy(D.EXPECTED_SLICES))
    allow_code_change: str | None = None
    force_new: bool = False
    stop_after_epoch: int | None = None  # test hook: simulate a job ending mid-run
    # preflight
    pf_learn_steps: int = 300
    pf_learn_slices: int = 16
    pf_learn_loss_ratio: float = 0.5     # loss on the 16 slices must at least halve
    pf_lr_steps: int = 300
    pf_probe_slices: int = 32
    pf_lr_max_drop: float = 0.01
    pf_identity_min_agree: float = 0.999
    pf_identity_max_ddice: float = 0.0005
    pf_identity_step: int = 14


# Settings that change results and must match on every resume. Run-local paths, worker count
# (jitter is seeded per draw, so workers cannot change results) and test hooks are excluded.
STAMP_EXCLUDE = {"run_dir", "merged_out", "workers", "allow_code_change", "force_new",
                 "stop_after_epoch", "processed", "checkpoint"}


def stamp_of(cfg: Cfg) -> dict:
    d = dataclasses.asdict(cfg)
    out = {k: v for k, v in d.items() if k not in STAMP_EXCLUDE and not k.startswith("pf_")}
    if out.get("modality") == "both":           # arm 2's stamp stays exactly as it was
        del out["modality"]
    return json.loads(json.dumps(out))          # tuples -> lists, canonical for comparison


def modalities_of(cfg: Cfg) -> tuple:
    if cfg.modality not in MODALITY_SETS:
        raise ValueError(f"modality must be one of {sorted(MODALITY_SETS)}, not {cfg.modality!r}")
    return MODALITY_SETS[cfg.modality]


def dataset_modality(cfg: Cfg) -> str | None:
    """The SliceDataset filter for this run: None (both) or "ct" / "mri"."""
    return None if len(modalities_of(cfg)) == 2 else cfg.modality


# ============================================================================ helpers
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 23), b""):
            h.update(block)
    return h.hexdigest()


def code_hashes() -> dict:
    here = Path(__file__).resolve().parent
    return {f: sha256_file(here / f) for f in CODE_FILES}


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


def write_metrics(path: Path, rows: list[dict]) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=METRIC_FIELDS, extrasaction="ignore")
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
    """Refuse the wrong hardware (§6a). `info` is injectable for tests."""
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


def build_plain_vitb(checkpoint, device):
    from segment_anything import sam_model_registry
    sam = sam_model_registry["vit_b"](checkpoint=str(checkpoint))
    return sam.to(device)


def build_lora(cfg: Cfg, build_plain, device):
    sam = build_plain(cfg.checkpoint, device)
    torch.manual_seed(cfg.seed)                         # LoRA A init is reproducible
    L.inject_lora(sam, cfg.rank, cfg.alpha, cfg.expect_blocks, cfg.expect_dim)
    info = L.freeze_for_arm2(sam)
    if cfg.grad_checkpoint:
        from torch.utils.checkpoint import checkpoint as ckpt
        for blk in sam.image_encoder.blocks:
            f = blk.forward
            blk.forward = lambda x, f=f: ckpt(f, x, use_reentrant=False)
    return sam.to(device), info


def trainable_groups(sam) -> tuple[list, list]:
    lora = [lin.weight for m in L.lora_modules(sam) for lin in (m.A_q, m.B_q, m.A_v, m.B_v)]
    dec = list(sam.mask_decoder.parameters())
    return lora, dec


def make_optimizer(sam, cfg: Cfg, total_steps: int):
    lora, dec = trainable_groups(sam)
    opt = torch.optim.AdamW([{"params": lora, "lr": cfg.lr_lora},
                             {"params": dec, "lr": cfg.lr_dec}],
                            betas=tuple(cfg.betas), weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, [
        lambda s: C.lr_factor(s, cfg.lr_lora, cfg.warmup, total_steps, cfg.lr_floor),
        lambda s: C.lr_factor(s, cfg.lr_dec, cfg.warmup, total_steps, cfg.lr_floor)])
    return opt, sched


def steps_per_epoch(n_draws: int, cfg: Cfg) -> int:
    n = (n_draws // cfg.batch) // cfg.accum
    if n < 1:
        raise ValueError(f"{n_draws} draws give no full step at batch {cfg.batch} x accum {cfg.accum}")
    return n


def autocast_ctx(cfg: Cfg, device):
    dev = torch.device(device)
    return torch.autocast(device_type=dev.type, dtype=torch.bfloat16,
                          enabled=bool(cfg.amp and dev.type == "cuda"))


def loader(ds, batch, workers, drop_last):
    return DataLoader(ds, batch_size=batch, shuffle=False, drop_last=drop_last,
                      num_workers=workers, collate_fn=D.collate_list,
                      worker_init_fn=D.worker_init if workers else None, pin_memory=False)


# ======================================================================= predict / val
@torch.no_grad()
def predict_val(sam, items: D.ValItems, device, batch: int, workers: int, keep_logits=False):
    """The validation forward: fp32, eval(), no_grad, one encoding per slice, every kidney
    decoded at every level. Returns per-instance rows (and logits if asked)."""
    was_training = sam.training
    sam.eval()
    rows, logits_out = [], []
    try:
        for b in loader(items, batch, workers, drop_last=False):
            feats = sam.image_encoder(C.to_model_input(sam, [it["image"] for it in b], device))
            for j, it in enumerate(b):
                for lev in items.levels:
                    for k, box in enumerate(it["boxes"][lev]):
                        bt = C.transform_box(box, it["original_size"], sam.image_encoder.img_size, device)
                        lg, _ = C.decode_one(sam, feats[j:j + 1], bt, it["input_size"], it["original_size"])
                        pred = (lg > 0).cpu().numpy()
                        gt = it["gts"][k]
                        row = {"case": it["case"], "z": it["z"], "side": it["sides"][k],
                               "modality": it["modality"], "patient": it["patient"],
                               "level": lev, "dice": C.dice_bool(pred, gt), "box": box}
                        if lev == 0.10 and items_nsd(items):
                            row["nsd3"] = M.score_masks(gt, pred, it["sp"])["nsd_3"]
                        rows.append(row)
                        if keep_logits:
                            logits_out.append(lg.float().cpu())
    finally:
        sam.train(was_training)
    return rows, logits_out


def items_nsd(items) -> bool:
    return getattr(items, "nsd", True)


def summarise_val(rows: list[dict], select_level: float, modalities: tuple = ("ct", "mri")) -> dict:
    s = C.val_metric(rows, select_level=select_level, modalities=modalities)
    for mod in ("ct", "mri"):
        per = {}
        for r in rows:
            if r["modality"] == mod and "nsd3" in r:
                per.setdefault(r["patient"], []).append(r["nsd3"])
        s[f"nsd3_{mod}@0.10"] = (sum(sum(v) / len(v) for v in per.values()) / len(per)) if per else float("nan")
    return s


@torch.no_grad()
def reference_logits(ref_sam, base: D.SliceDataset, indices: list[int], level: float) -> list:
    """Arm 1's own steps on the same slices: np.load -> repeat -> SamPredictor.set_image ->
    jitter_box(jitter_seed) -> predict(box, multimask_output=False) logits."""
    from segment_anything import SamPredictor
    ref_sam.eval()
    pred = SamPredictor(ref_sam)
    out = []
    for idx in indices:
        item = base.load(idx)
        pred.set_image(np.repeat(item["img"][..., None], 3, axis=2), image_format="RGB")
        for kd in item["kidneys"]:
            jb = D.ValItems.val_box(item, kd, level)
            masks, _, _ = pred.predict(box=np.array(jb, dtype=float), multimask_output=False,
                                       return_logits=True)
            out.append((torch.as_tensor(np.asarray(masks[0], dtype=np.float32)), kd["gt"]))
    return out


def compare_logits(ours: list, ref: list) -> dict:
    """Pixel agreement of logit>0, mean-Dice difference and max |logit difference|."""
    if len(ours) != len(ref):
        raise AssertionError(f"identity: {len(ours)} vs {len(ref)} kidneys")
    agree = total = 0
    d_ours, d_ref, max_dl = [], [], 0.0
    for lo, (lr, gt) in zip(ours, ref):
        po, pr = (lo > 0).numpy(), (lr > 0).numpy()
        agree += int((po == pr).sum())
        total += po.size
        d_ours.append(C.dice_bool(po, gt))
        d_ref.append(C.dice_bool(pr, gt))
        max_dl = max(max_dl, float((lo - lr).abs().max()))
    return {"pixel_agreement": agree / total, "mean_dice_ours": float(np.mean(d_ours)),
            "mean_dice_ref": float(np.mean(d_ref)),
            "abs_mean_dice_diff": abs(float(np.mean(d_ours)) - float(np.mean(d_ref))),
            "max_abs_logit_diff": max_dl, "n_kidneys": len(ours)}


@torch.no_grad()
def probe_loss(sam, items: list[dict], device) -> float:
    """Mean per-kidney training loss on a FIXED set of items (same slices, same boxes), fp32,
    eval mode. Comparing it before/after a few hundred steps is a noise-free learning signal,
    unlike comparing the losses of different training batches."""
    was = sam.training
    sam.eval()
    tot, n = 0.0, 0
    try:
        for k in range(0, len(items), 8):
            mb = items[k:k + 8]
            outs = C.forward_batch(sam, [it["image"] for it in mb], [it["boxes"] for it in mb],
                                   [it["input_size"] for it in mb], [it["original_size"] for it in mb], device)
            for it, res in zip(mb, outs):
                for (lg, iou), gt in zip(res, it["gts"]):
                    tot += float(C.kidney_loss(lg, iou, gt)["total"])
                    n += 1
    finally:
        sam.train(was)
    return tot / n


def free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================================ training
def train_one_step(sam, micro_batches: list, kids_eff: int, cfg: Cfg, device) -> dict:
    """Forward + backward over one optimiser step's micro-batches. Each kidney weighs
    1 / kids_eff, so batch 8 == 4 x 2 accumulation. Returns summed loss components."""
    sums = {"dice": 0.0, "bce": 0.0, "iou": 0.0, "total": 0.0, "kidneys": 0}
    for mb in micro_batches:
        with autocast_ctx(cfg, device):
            outs = C.forward_batch(sam, [it["image"] for it in mb], [it["boxes"] for it in mb],
                                   [it["input_size"] for it in mb],
                                   [it["original_size"] for it in mb], device)
        loss = 0.0
        for it, res in zip(mb, outs):
            for (lg, iou), gt in zip(res, it["gts"]):
                kl = C.kidney_loss(lg, iou, gt)
                loss = loss + kl["total"]
                for k in ("dice", "bce", "iou", "total"):
                    sums[k] += float(kl[k].detach())
                sums["kidneys"] += 1
        loss = loss / kids_eff
        if not torch.isfinite(loss):
            raise TrainFailed(f"non-finite loss {float(loss)}")
        loss.backward()
    if sums["kidneys"] != kids_eff:
        raise AssertionError(f"step saw {sums['kidneys']} kidneys, manifest says {kids_eff}")
    return sums


def optimiser_step(sam, opt, sched, cfg: Cfg) -> float:
    lora, dec = trainable_groups(sam)
    norm = torch.nn.utils.clip_grad_norm_(lora + dec, cfg.clip)
    if not torch.isfinite(norm):
        raise TrainFailed(f"non-finite gradient norm {float(norm)}")
    opt.step()
    sched.step()
    opt.zero_grad(set_to_none=True)
    return float(norm)


def run_epoch(sam, opt, sched, train_ds, epoch: int, cfg: Cfg, device, n_draws: int,
              max_steps: int | None = None, record_steps: list | None = None) -> dict:
    draws = D.make_epoch_draws(train_ds, epoch, cfg.seed, n_draws)
    n_steps = steps_per_epoch(n_draws, cfg)
    if max_steps is not None:
        n_steps = min(n_steps, max_steps)
    e_eff = cfg.batch * cfg.accum
    used = draws[: n_steps * e_eff]
    ds = D.TrainDraws(train_ds, used, epoch, cfg.seed, cfg.img_size, cfg.max_level)
    it = iter(loader(ds, cfg.batch, cfg.workers, drop_last=True))
    sam.train()
    tot = {"dice": 0.0, "bce": 0.0, "iou": 0.0, "total": 0.0, "kidneys": 0}
    norms, wait = [], 0.0
    t0 = time.time()
    for s in range(n_steps):
        tw = time.time()
        mbs = [next(it) for _ in range(cfg.accum)]
        ts = time.time()
        wait += ts - tw
        kids = sum(train_ds.n_kidneys(idx) for _, idx in used[s * e_eff:(s + 1) * e_eff])
        sums = train_one_step(sam, mbs, kids, cfg, device)
        norms.append(optimiser_step(sam, opt, sched, cfg))
        if record_steps is not None:
            if torch.device(device).type == "cuda":
                torch.cuda.synchronize()
            record_steps.append({"loss": sums["total"] / sums["kidneys"],
                                 "step_s": time.time() - ts, "wait_s": ts - tw})
        for k in tot:
            tot[k] += sums[k]
    n_mri = sum(1 for _, idx in used if train_ds.rows[idx]["modality"] == "mri")
    return {"train_loss": tot["total"] / tot["kidneys"], "train_dice": tot["dice"] / tot["kidneys"],
            "train_bce": tot["bce"] / tot["kidneys"], "train_iou": tot["iou"] / tot["kidneys"],
            "grad_norm": float(np.mean(norms)), "steps": n_steps, "ct_draws": len(used) - n_mri,
            "mri_draws": n_mri, "train_s": time.time() - t0, "data_wait_s": wait}


def peak_gib(device) -> float:
    if torch.device(device).type != "cuda":
        return 0.0
    return round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)


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


def adapter_digest(sam) -> str:
    h = hashlib.sha256()
    for k, v in L.adapter_state(sam).items():
        h.update(k.encode())
        h.update(v.float().numpy().tobytes())
    return h.hexdigest()


def load_datasets(cfg: Cfg):
    mod = dataset_modality(cfg)
    return (D.SliceDataset(Path(cfg.processed), "train", cfg.expected_slices, modality=mod),
            D.SliceDataset(Path(cfg.processed), "val", cfg.expected_slices, modality=mod))


def check_run_data(cfg: Cfg, train_ds, val_ds, n_draws: int) -> dict:
    """Startup guard (D71): train and val hold exactly the run's modalities, and an epoch is
    one draw per training slice unless n_draws was set explicitly (test hook). Returns the
    counts recorded in run.json."""
    want = set(modalities_of(cfg))
    for name, ds in (("train", train_ds), ("val", val_ds)):
        have = {r["modality"] for r in ds.rows}
        if have != want:
            raise SystemExit(f"{name} data holds modalities {sorted(have)}, the run ({cfg.modality}) "
                             f"needs exactly {sorted(want)}")
    if cfg.n_draws is None and n_draws != len(train_ds):
        raise SystemExit(f"{n_draws} draws per epoch, but {len(train_ds)} training slices")
    val_patients: dict = {}
    for r in val_ds.rows:
        val_patients.setdefault(r["modality"], set()).add(r["patient"])
    return {"train_slices": dict(train_ds.counts), "val_slices": dict(val_ds.counts),
            "val_patients": {m: len(p) for m, p in sorted(val_patients.items())}}


def check_stamp_and_write(cfg: Cfg, run_dir: Path, extra: dict, log) -> dict:
    """config.json: written on first start; on every later start the settings, data, base
    checkpoint, GPU type and code must be the same (code only with --allow-code-change)."""
    path = run_dir / "config.json"
    now = {"settings": stamp_of(cfg), **extra}
    if not path.exists():
        atomic_write_json(path, {**now, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                 "code_changes": []})
        return now
    have = json.loads(path.read_text())
    diffs = [k for k in ("settings", "base_sha256", "manifest_sha256", "gpu_type")
             if have.get(k) != now.get(k)]
    if diffs:
        detail = {k: (have.get(k), now.get(k)) for k in diffs}
        raise SystemExit(f"{run_dir}: refusing to resume -- changed {diffs}: {detail}")
    if have.get("code") != now.get("code"):
        changed = sorted(f for f in now["code"] if have.get("code", {}).get(f) != now["code"][f])
        if not cfg.allow_code_change:
            raise SystemExit(f"{run_dir}: training code changed since the run started ({changed}); "
                             f"resume only with --allow-code-change \"<reason>\"")
        have.setdefault("code_changes", []).append({
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "files": changed,
            "reason": cfg.allow_code_change, "new_hashes": {f: now["code"][f] for f in changed}})
        have["code"] = now["code"]
        atomic_write_json(path, have)
        log(f"code change accepted and logged: {changed} -- {cfg.allow_code_change}")
    return now


def startup(cfg: Cfg, device, log, expected_ckpt: bool = True) -> dict:
    """GPU, split integrity, base checkpoint identity, manifest hash."""
    gpu = gpu_check(cfg.expect_gpu)
    if cfg.expect_gpu:
        log(f"GPU: {gpu['name']}  {gpu['total_gib']} GiB  capability {gpu['capability']}  "
            f"bf16 {gpu['bf16']}  CUDA {gpu['cuda_version']}  torch {gpu['torch']}")
    D.check_split_integrity(Path(cfg.processed))
    ident = E.checkpoint_identity(Path(cfg.checkpoint), "vit_b")
    if cfg.check_base_sha and expected_ckpt:
        if (ident["checkpoint_sha256"], ident["checkpoint_bytes"]) != (BASE_SHA256, BASE_BYTES):
            raise SystemExit(f"base checkpoint {cfg.checkpoint} is not the verified SAM ViT-B "
                             f"(sha256 {ident['checkpoint_sha256'][:16]}..., {ident['checkpoint_bytes']} bytes)")
    return {"gpu": gpu, "base_sha256": ident["checkpoint_sha256"],
            "manifest_sha256": D.manifest_sha256(Path(cfg.processed)),
            "gpu_type": gpu.get("name", "cpu") if gpu.get("cuda") else "cpu"}


def run_training(cfg: Cfg, build_plain=build_plain_vitb, datasets=None, log=print) -> int:
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.device)
    env = startup(cfg, device, log)
    stamp = check_stamp_and_write(cfg, run_dir, {"base_sha256": env["base_sha256"],
                                                 "manifest_sha256": env["manifest_sha256"],
                                                 "gpu_type": env["gpu_type"], "code": code_hashes()}, log)
    rj_path = run_dir / "run.json"
    if rj_path.exists():
        rj = json.loads(rj_path.read_text())
        if rj.get("status") == "TRAIN_OK" and rj.get("epochs_completed") == cfg.epochs:
            log(f"{run_dir}: already complete ({cfg.epochs} epochs, best epoch {rj['best_epoch']}).")
            log("TRAIN_OK")
            return 0

    train_ds, val_ds = datasets if datasets is not None else load_datasets(cfg)
    mods = modalities_of(cfg)
    n_draws = cfg.n_draws or len(train_ds)
    data_info = check_run_data(cfg, train_ds, val_ds, n_draws)
    spe = steps_per_epoch(n_draws, cfg)
    total = spe * cfg.epochs
    log(f"modality {cfg.modality}: train slices {data_info['train_slices']}, val slices "
        f"{data_info['val_slices']}, val patients {data_info['val_patients']}; {n_draws} draws/epoch")
    sam, info = build_lora(cfg, build_plain, device)
    log(f"rank {cfg.rank}, alpha {cfg.alpha}: trainable {info['trainable']:,} "
        f"(LoRA {info['lora']:,} + decoder {info['decoder']:,}) of {info['total']:,}; "
        f"{spe} steps/epoch x {cfg.epochs} = {total} steps; batch {cfg.batch} x accum {cfg.accum}")
    opt, sched = make_optimizer(sam, cfg, total)
    val_items = D.ValItems(val_ds, None, cfg.img_size, tuple(cfg.val_levels))
    val_items.nsd = cfg.nsd

    state_path = run_dir / "state.json"
    if not state_path.exists():
        atomic_write_json(state_path, {"frozen_digest": L.frozen_state_digest(sam),
                                       "initial_adapter_digest": adapter_digest(sam), "info": info})
    state = json.loads(state_path.read_text())
    if L.frozen_state_digest(sam) != state["frozen_digest"]:
        raise SystemExit("frozen weights differ from the run's start (base checkpoint changed?)")

    metrics_path, last_path, best_path = run_dir / "metrics.csv", run_dir / "last.pt", run_dir / "best.pt"
    rows = read_metrics(metrics_path)

    def validate_row(epoch: int, extra: dict) -> dict:
        tv = time.time()
        vrows, _ = predict_val(sam, val_items, device, cfg.val_batch, cfg.workers)
        s = summarise_val(vrows, cfg.select_level, mods)
        # a single-modality run has no rows for the other modality: its columns are nan
        return {"epoch": epoch, **extra, **{k: round(s.get(k, float("nan")), 6) for k in VAL_COLUMNS},
                "selection": round(s["selection"], 6),
                "val_s": round(time.time() - tv, 1), "lr_lora": opt.param_groups[0]["lr"],
                "lr_dec": opt.param_groups[1]["lr"], "peak_mem_gib": peak_gib(device)}

    if last_path.exists():
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        L.load_adapter_state(sam, ck["adapter"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        set_rng_state(ck["rng"])
        done, best_epoch, best_score = ck["epoch"], ck["best_epoch"], ck["best_score"]
        if [int(r["epoch"]) for r in rows] != list(range(done + 1)):
            raise SystemExit(f"metrics.csv epochs {[r['epoch'] for r in rows]} are not 0..{done}")
        if sched.last_epoch != done * spe:
            raise SystemExit(f"scheduler at step {sched.last_epoch}, expected {done * spe}")
        log(f"resuming after epoch {done}; best so far epoch {best_epoch} ({best_score:.4f})")
    else:
        if rows:
            raise SystemExit(f"{metrics_path} exists without last.pt -- inconsistent run folder")
        row0 = validate_row(0, {"steps": 0})
        rows = [row0]
        best_epoch, best_score, done = 0, row0["selection"], 0
        atomic_torch_save({"adapter": L.adapter_state(sam), "epoch": 0, "score": best_score}, best_path)
        atomic_torch_save({"adapter": L.adapter_state(sam), "optimizer": opt.state_dict(),
                           "scheduler": sched.state_dict(), "rng": rng_state(), "epoch": 0,
                           "best_epoch": 0, "best_score": best_score}, last_path)
        write_metrics(metrics_path, rows)
        log(f"epoch 0 (off-the-shelf baseline): selection {best_score:.4f}  "
            + "  ".join(f"{m.upper()}@0.10 {row0[f'{m}@0.10']:.4f}" for m in mods))

    if done > cfg.epochs:
        raise SystemExit(f"{run_dir}: at epoch {done}, beyond the configured {cfg.epochs}")
    # done == epochs (all trained, job ended before run.json) -> the loop is empty and the
    # final checks + run.json below complete the run.
    for epoch in range(done + 1, cfg.epochs + 1):
        te = time.time()
        if torch.device(device).type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        tr = run_epoch(sam, opt, sched, train_ds, epoch, cfg, device, n_draws)
        row = validate_row(epoch, {k: (round(v, 6) if isinstance(v, float) else v) for k, v in tr.items()})
        row["epoch_s"] = round(time.time() - te, 1)
        rows.append(row)
        if row["selection"] > best_score:
            best_epoch, best_score = epoch, row["selection"]
            atomic_torch_save({"adapter": L.adapter_state(sam), "epoch": epoch, "score": best_score}, best_path)
        atomic_torch_save({"adapter": L.adapter_state(sam), "optimizer": opt.state_dict(),
                           "scheduler": sched.state_dict(), "rng": rng_state(), "epoch": epoch,
                           "best_epoch": best_epoch, "best_score": best_score}, last_path)
        write_metrics(metrics_path, rows)
        log(f"epoch {epoch}/{cfg.epochs}: loss {row['train_loss']:.4f}  val selection "
            f"{row['selection']:.4f} ("
            + ", ".join(f"{m.upper()} {row[f'{m}@0.10']:.4f}" for m in mods) + ")  best "
            f"{best_epoch} ({best_score:.4f})  {row['epoch_s']:.0f}s  wait {tr['data_wait_s']:.0f}s  "
            f"peak {row['peak_mem_gib']} GiB")
        if cfg.stop_after_epoch is not None and epoch == cfg.stop_after_epoch and epoch < cfg.epochs:
            log(f"stopping after epoch {epoch} (test hook)")
            return 0

    if L.frozen_state_digest(sam) != state["frozen_digest"]:
        raise TrainFailed("frozen weights changed during training")
    if adapter_digest(sam) == state["initial_adapter_digest"]:
        raise TrainFailed("adapter never changed -- training had no effect")
    late_best = best_epoch >= int(0.9 * cfg.epochs)      # the students' rule (D70)
    atomic_write_json(rj_path, {
        "status": "TRAIN_OK", "epochs_completed": cfg.epochs, "best_epoch": best_epoch,
        "best_score": best_score, "rank": cfg.rank, "alpha": cfg.alpha, "settings": stamp["settings"],
        "modality": cfg.modality, "n_draws": n_draws, **data_info, "late_best": late_best,
        "base_sha256": env["base_sha256"], "manifest_sha256": env["manifest_sha256"],
        "gpu": env["gpu"], "gpu_type": env["gpu_type"], "info": info, "steps_per_epoch": spe,
        "total_steps": total, "frozen_digest": state["frozen_digest"], "host": platform.node(),
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    log(f"best epoch {best_epoch}: val selection {best_score:.4f}")
    if late_best:
        log(f"WARNING late best epoch ({best_epoch} of {cfg.epochs}): under-trained? (D70 extension rule)")
    log("TRAIN_OK")
    return 0


# ============================================================================ finalize
def load_plain_vitb(path, device):
    from segment_anything import sam_model_registry
    return sam_model_registry["vit_b"](checkpoint=str(path)).to(device)


def finalize(cfg: Cfg, build_plain=build_plain_vitb, load_plain=load_plain_vitb, datasets=None,
             log=print) -> int:
    run_dir, out = Path(cfg.run_dir), Path(cfg.merged_out)
    rj_path = run_dir / "run.json"
    if not rj_path.exists():
        raise SystemExit(f"{rj_path} missing -- train first")
    rj = json.loads(rj_path.read_text())
    if rj.get("status") != "TRAIN_OK":
        raise SystemExit(f"{rj_path}: status {rj.get('status')}, not TRAIN_OK")
    run_mod = rj.get("settings", {}).get("modality", "both")
    if run_mod != cfg.modality:
        raise SystemExit(f"{rj_path}: run trained on modality {run_mod}, finalize called with {cfg.modality}")
    if out.exists():
        rec = rj.get("merged", {})
        if rec.get("sha256") == sha256_file(out) and rec.get("check", {}).get("passed"):
            log(f"{out} already merged and checked (sha256 {rec['sha256'][:16]}...)")
            log("FINALIZE_OK")
            return 0
        if not cfg.force_new:
            raise SystemExit(f"{out} exists but does not match {rj_path}; refusing to overwrite "
                             f"(--force-new to replace)")
    device = torch.device(cfg.device)
    if cfg.expect_gpu:
        gpu_check(cfg.expect_gpu)
    _, val_ds = datasets if datasets is not None else load_datasets(cfg)
    subset = D.identity_subset(val_ds, cfg.pf_identity_step)
    items = D.ValItems(val_ds, subset, cfg.img_size, (0.10,))
    items.nsd = False

    sam, _ = build_lora(cfg, build_plain, device)
    best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    if best["epoch"] != rj["best_epoch"]:
        raise SystemExit(f"best.pt is epoch {best['epoch']}, run.json says {rj['best_epoch']}")
    L.load_adapter_state(sam, best["adapter"])
    _, ours = predict_val(sam, items, device, cfg.val_batch, 0, keep_logits=True)

    L.merge_lora(sam)
    plain_keys = L.plain_state_keys(build_plain(cfg.checkpoint, "cpu"))
    if L.plain_state_keys(sam) != plain_keys:
        raise SystemExit("merged state-dict keys differ from a plain SAM ViT-B")
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save({k: v.detach().to("cpu").contiguous() for k, v in sam.state_dict().items()}, out)
    sha = sha256_file(out)
    del sam

    ref = load_plain(out, device)
    refs = reference_logits(ref, val_ds, subset, 0.10)
    chk = compare_logits(ours, refs)
    chk["passed"] = (chk["pixel_agreement"] >= cfg.pf_identity_min_agree
                     and chk["abs_mean_dice_diff"] < cfg.pf_identity_max_ddice)
    rj["merged"] = {"path": str(out), "sha256": sha, "bytes": out.stat().st_size,
                    "best_epoch": best["epoch"], "check": chk}
    atomic_write_json(rj_path, rj)
    log(f"merge check: pixel agreement {chk['pixel_agreement']:.6f}, |dDice| "
        f"{chk['abs_mean_dice_diff']:.2e}, max |dlogit| {chk['max_abs_logit_diff']:.2e} "
        f"over {chk['n_kidneys']} kidneys")
    if not chk["passed"]:
        log("FINALIZE_FAILED")
        return 1
    log(f"merged -> {out} (sha256 {sha[:16]}...)")
    log("FINALIZE_OK")
    return 0


def check_merged(cfg: Cfg, log=print) -> int:
    rj_path = Path(cfg.run_dir) / "run.json"
    if not rj_path.exists():
        log(f"MERGED_FAILED: {rj_path} missing")
        return 1
    rj = json.loads(rj_path.read_text())
    rec = rj.get("merged", {})
    out = Path(cfg.merged_out)
    problems = []
    if rj.get("status") != "TRAIN_OK":
        problems.append(f"status {rj.get('status')}")
    if not rec.get("check", {}).get("passed"):
        problems.append("merge check not passed")
    if not out.exists():
        problems.append(f"{out} missing")
    elif rec.get("sha256") != sha256_file(out):
        problems.append("merged checkpoint sha256 differs from run.json")
    if Path(rec.get("path", "")).resolve() != out.resolve():
        problems.append(f"run.json records {rec.get('path')}, not {out}")
    run_mod = rj.get("settings", {}).get("modality", "both")     # arm 2's run.json has no key
    if run_mod != cfg.modality:
        problems.append(f"run trained on modality {run_mod}, expected {cfg.modality}")
    if problems:
        log("MERGED_FAILED: " + "; ".join(problems))
        return 1
    log(f"MERGED_OK {out} sha256 {rec['sha256'][:16]}... best epoch {rec['best_epoch']}")
    return 0


# ============================================================================ preflight
def run_preflight(cfg: Cfg, build_plain=build_plain_vitb, datasets=None, log=print) -> int:
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.device)
    env = startup(cfg, device, log)
    train_ds, val_ds = datasets if datasets is not None else load_datasets(cfg)
    report = {"gpu": env["gpu"], "settings": stamp_of(cfg), "checks": {}}
    ok = True
    subset = D.identity_subset(val_ds, cfg.pf_identity_step)

    # 1. identity: our forward (LoRA, B = 0) vs SamPredictor with the original checkpoint
    sam, info = build_lora(cfg, build_plain, device)
    items = D.ValItems(val_ds, subset, cfg.img_size, (0.10,))
    items.nsd = False
    t = time.time()
    _, ours = predict_val(sam, items, device, cfg.val_batch, cfg.workers, keep_logits=True)
    val_subset_s = time.time() - t
    del sam
    ref = build_plain(cfg.checkpoint, device)
    chk = compare_logits(ours, reference_logits(ref, val_ds, subset, 0.10))
    del ref
    chk["passed"] = (chk["pixel_agreement"] >= cfg.pf_identity_min_agree
                     and chk["abs_mean_dice_diff"] < cfg.pf_identity_max_ddice)
    report["checks"]["identity"] = chk
    ok &= chk["passed"]
    log(f"[1] identity over {chk['n_kidneys']} kidneys / {len(subset)} slices: agreement "
        f"{chk['pixel_agreement']:.6f}, |dDice| {chk['abs_mean_dice_diff']:.2e}, max |dlogit| "
        f"{chk['max_abs_logit_diff']:.2e} -> {'PASS' if chk['passed'] else 'FAIL'}")

    # 2. timing / memory / LR sanity, with the OOM fallback ladder
    n_draws = cfg.n_draws or len(train_ds)
    total = steps_per_epoch(n_draws, cfg) * cfg.epochs
    lr_res, chosen = None, None
    # Fixed probe for the LR check: the first draws of epoch 0, with their (seeded) jitter.
    probe_ds = D.TrainDraws(train_ds, D.make_epoch_draws(train_ds, 0, cfg.seed, n_draws)[:cfg.pf_probe_slices],
                            0, cfg.seed, cfg.img_size, cfg.max_level)
    probe = [probe_ds[k] for k in range(len(probe_ds))]
    ladder = [(cfg.batch, cfg.accum, cfg.grad_checkpoint), (4, 2, cfg.grad_checkpoint), (4, 2, True)]
    for b, a, gck in ladder:
        trial = dataclasses.replace(cfg, batch=b, accum=a, grad_checkpoint=gck)
        sam = opt = sched = None
        try:
            free_gpu()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            sam, _ = build_lora(trial, build_plain, device)
            sub_items = D.ValItems(val_ds, subset, cfg.img_size, (0.10,))
            sub_items.nsd = False
            d0 = np.mean([r["dice"] for r in predict_val(sam, sub_items, device, cfg.val_batch, cfg.workers)[0]])
            p0 = probe_loss(sam, probe, device)
            opt, sched = make_optimizer(sam, trial, steps_per_epoch(n_draws, trial) * cfg.epochs)
            steps = []
            run_epoch(sam, opt, sched, train_ds, 1, trial, device, n_draws,
                      max_steps=cfg.pf_lr_steps, record_steps=steps)
            p1 = probe_loss(sam, probe, device)
            d1 = np.mean([r["dice"] for r in predict_val(sam, sub_items, device, cfg.val_batch, cfg.workers)[0]])
            timed = steps[50:] or steps                 # skip warm-up / cuDNN autotune steps
            lr_res = {"steps": len(steps), "probe_slices": len(probe),
                      "probe_loss_before": p0, "probe_loss_after": p1,
                      "val_subset_dice_before": float(d0), "val_subset_dice_after": float(d1),
                      "running_loss_first": float(np.mean([s_["loss"] for s_ in steps[:50]])),
                      "running_loss_last": float(np.mean([s_["loss"] for s_ in steps[-50:]])),
                      "s_per_step": float(np.mean([s_["step_s"] for s_ in timed])),
                      "wait_per_step": float(np.mean([s_["wait_s"] for s_ in timed])),
                      "peak_mem_gib": peak_gib(device)}
            chosen = {"batch": b, "accum": a, "grad_checkpoint": gck}
            break
        except torch.cuda.OutOfMemoryError:
            log(f"    out of memory at batch {b} x accum {a} (grad checkpoint {gck}); trying the next setting")
            continue
        finally:
            del sam, opt, sched                         # free before the next attempt / next check
            free_gpu()
    if chosen is None:
        report["checks"]["lr_sanity"] = {"passed": False, "error": "out of memory at every setting"}
        ok = False
        log("[2] LR sanity: out of memory at every setting -> FAIL")
    else:
        lr_res["passed"] = (lr_res["probe_loss_after"] < lr_res["probe_loss_before"] and
                            lr_res["val_subset_dice_after"] >= lr_res["val_subset_dice_before"] - cfg.pf_lr_max_drop)
        report["checks"]["lr_sanity"] = lr_res
        report["recommended"] = chosen
        ok &= lr_res["passed"]
        log(f"[2] LR sanity ({lr_res['steps']} steps at batch {chosen['batch']} x accum {chosen['accum']}): "
            f"fixed-probe loss {lr_res['probe_loss_before']:.4f} -> {lr_res['probe_loss_after']:.4f}, val-subset Dice "
            f"{lr_res['val_subset_dice_before']:.4f} -> {lr_res['val_subset_dice_after']:.4f} -> "
            f"{'PASS' if lr_res['passed'] else 'FAIL'}")
        log(f"    {lr_res['s_per_step']:.3f} s/step, data wait {lr_res['wait_per_step']:.3f} s/step, "
            f"peak {lr_res['peak_mem_gib']} GiB")

        # 3. can it learn: 16 fixed slices, clean boxes, constant LR
        trial = dataclasses.replace(cfg, **{k: chosen[k] for k in ("batch", "accum", "grad_checkpoint")})
        sam, _ = build_lora(trial, build_plain, device)
        idx = [int(round(i * (len(train_ds) - 1) / max(1, cfg.pf_learn_slices - 1)))
               for i in range(cfg.pf_learn_slices)]
        lora, dec = trainable_groups(sam)
        opt = torch.optim.AdamW([{"params": lora, "lr": cfg.lr_lora}, {"params": dec, "lr": cfg.lr_dec}],
                                betas=tuple(cfg.betas), weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
        clean_draws = [(i, j) for i, j in enumerate(idx)]
        learn_ds = D.TrainDraws(train_ds, clean_draws, 0, cfg.seed, cfg.img_size, max_level=0.0)
        learn_items = [learn_ds[k] for k in range(len(learn_ds))]
        clean_items = D.ValItems(train_ds, idx, cfg.img_size, (0.0,))
        clean_items.nsd = False
        dice_before = float(np.mean([r["dice"] for r in predict_val(sam, clean_items, device, cfg.val_batch, 0)[0]]))
        loss_before = probe_loss(sam, learn_items, device)
        e_eff = trial.batch * trial.accum
        for s in range(cfg.pf_learn_steps):
            start = (s * e_eff) % len(clean_draws)
            sel = [learn_items[(start + q) % len(clean_draws)] for q in range(e_eff)]
            mbs = [sel[m * trial.batch:(m + 1) * trial.batch] for m in range(trial.accum)]
            kids = sum(len(it["boxes"]) for it in sel)
            train_one_step(sam, mbs, kids, trial, device)
            optimiser_step(sam, opt, sched, trial)
        learn_dice = float(np.mean([r["dice"] for r in predict_val(sam, clean_items, device, cfg.val_batch, 0)[0]]))
        loss_after = probe_loss(sam, learn_items, device)
        del sam, opt
        free_gpu()
        # Gradients reach LoRA + decoder and the model fits: the loss on these slices at least
        # halves and Dice rises. (An absolute Dice target is not used: SAM's 256x256 mask
        # resolution caps boundary Dice on small kidney slices however well it learns.)
        learn = {"slices": len(idx), "steps": cfg.pf_learn_steps, "dice_before": dice_before,
                 "dice_after": learn_dice, "loss_before": loss_before, "loss_after": loss_after,
                 "passed": loss_after <= cfg.pf_learn_loss_ratio * loss_before and learn_dice > dice_before}
        report["checks"]["learn"] = learn
        ok &= learn["passed"]
        log(f"[3] can it learn ({len(idx)} slices, {cfg.pf_learn_steps} steps, clean boxes): loss "
            f"{loss_before:.4f} -> {loss_after:.4f} (need <= {cfg.pf_learn_loss_ratio:.0%} of start), Dice "
            f"{dice_before:.4f} -> {learn_dice:.4f} (must rise) -> {'PASS' if learn['passed'] else 'FAIL'}")

        # 4. time estimate for the SLURM limit
        spe = steps_per_epoch(n_draws, dataclasses.replace(cfg, batch=chosen["batch"], accum=chosen["accum"]))
        val_full_s = val_subset_s * (len(val_ds) / max(1, len(subset))) * len(cfg.val_levels)
        epoch_s = spe * (lr_res["s_per_step"] + lr_res["wait_per_step"]) + val_full_s
        total_h = (cfg.epochs * epoch_s + val_full_s) / 3600
        report["estimate"] = {"steps_per_epoch": spe, "val_full_s": round(val_full_s),
                              "epoch_s": round(epoch_s), "total_h": round(total_h, 2),
                              "recommended_time_h": int(math.ceil(1.5 * total_h)), "total_steps": total}
        log(f"[4] estimate: {epoch_s / 60:.1f} min/epoch, {total_h:.1f} h for {cfg.epochs} epochs "
            f"-> set ARM2_TRAIN_TIME to {int(math.ceil(1.5 * total_h))}:00:00 in gen_sbatch.py")

    report["passed"] = bool(ok)
    atomic_write_json(run_dir / "preflight.json", report)
    log("PREFLIGHT_OK" if ok else "PREFLIGHT_FAILED")
    return 0 if ok else 1


# ============================================================================ main
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--processed", required=True)
    p.add_argument("--checkpoint", required=True, help="the base SAM ViT-B checkpoint")
    p.add_argument("--rank", type=int, choices=(4, 8, 16), required=True)
    p.add_argument("--alpha", type=float, default=4.0)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--merged-out", default="")
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--lr-lora", type=float, default=5e-4)
    p.add_argument("--lr-dec", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=643)
    p.add_argument("--modality", choices=sorted(MODALITY_SETS), default="both",
                   help="both = arm 2; ct / mri = a single-modality teacher (D71)")
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--grad-checkpoint", action="store_true")
    p.add_argument("--expect-gpu", default=None)
    p.add_argument("--allow-code-change", default=None, metavar="REASON")
    p.add_argument("--force-new", action="store_true")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--finalize", action="store_true")
    mode.add_argument("--check-merged", action="store_true")
    a = p.parse_args(argv)
    if a.alpha != 4.0:
        p.error("alpha is fixed at 4 for both ranks (D33)")
    if (a.finalize or a.check_merged) and not a.merged_out:
        p.error("--finalize / --check-merged need --merged-out")
    cfg = Cfg(processed=a.processed, checkpoint=a.checkpoint, rank=a.rank, alpha=a.alpha,
              run_dir=a.run_dir, merged_out=a.merged_out, epochs=a.epochs, batch=a.batch,
              accum=a.accum, lr_lora=a.lr_lora, lr_dec=a.lr_dec, seed=a.seed,
              modality=a.modality, workers=a.workers,
              device=a.device, grad_checkpoint=a.grad_checkpoint, expect_gpu=a.expect_gpu,
              allow_code_change=a.allow_code_change, force_new=a.force_new)
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "8"))))
    try:
        if a.check_merged:
            return check_merged(cfg)
        if a.preflight:
            return run_preflight(cfg)
        if a.finalize:
            return finalize(cfg)
        return run_training(cfg)
    except TrainFailed as e:
        atomic_write_json(Path(cfg.run_dir) / "failure.json",
                          {"error": str(e), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        print(f"!! {e}\nTRAIN_FAILED")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
