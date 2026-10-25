#!/usr/bin/env python
"""Diagnostic for the arm-4 preflight's P10 (1.0 s/step vs arm 3's 0.065 s): time the training
step by phase, with and without distillation, on a fixed batch and through the real loader.
Changes nothing; writes nothing but its log.

    python kd_speed_probe.py --processed P --teacher-dir T [--steps 30] [--workers 12]
"""
from __future__ import annotations

import argparse
import os
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import student_core as C  # noqa: E402
import student_data as D  # noqa: E402
import student_kd as K  # noqa: E402
import train_student as T  # noqa: E402


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def phase_step(model, opt, batch, cfg, device, kd: bool) -> dict:
    """train_step's operations, synchronised after each phase (ms)."""
    t = {}
    sync()
    t0 = time.perf_counter()
    x = batch["x"].to(device)
    y = batch["label"].to(device).long()
    v = batch["valid"].to(device)
    if kd:
        tl, tr, th = (batch[k].to(device) for k in ("t_logit", "t_region", "t_has"))
    sync()
    t1 = time.perf_counter()
    with T.autocast_ctx(cfg, device):
        out = model(x)
    sync()
    t2 = time.perf_counter()
    outs = T.float_outputs(out)
    seg = C.ds_loss(outs, y, v)
    sync()
    t3 = time.perf_counter()
    total = seg["total"]
    if kd:
        k = K.kd_loss(outs[0], tl, tr, v, th, cfg.kd_temperature)
        total = total + cfg.kd_lambda * k["total"]
    float(total.detach())
    sync()
    t4 = time.perf_counter()
    opt.zero_grad(set_to_none=True)
    total.backward()
    sync()
    t5 = time.perf_counter()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
    opt.step()
    sync()
    t6 = time.perf_counter()
    for name, a, b in (("to_gpu", t0, t1), ("forward", t1, t2), ("seg_loss", t2, t3), ("kd_loss", t3, t4),
                       ("backward", t4, t5), ("clip_step", t5, t6), ("total", t0, t6)):
        t[name] = (b - a) * 1000
    return t


def median_table(rows: list) -> dict:
    return {k: round(statistics.median(r[k] for r in rows), 1) for k in rows[0]}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--processed", required=True)
    p.add_argument("--teacher-dir", required=True)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--device", default="cuda")
    a = p.parse_args(argv)
    run = Path(tempfile.mkdtemp(prefix="kdprobe_")) / "arm4_plain_s643"
    cfg = T.Cfg(processed=a.processed, run_dir=str(run), kd=True, teacher_dir=a.teacher_dir,
                expect_gpu="A100" if a.device == "cuda" else None, workers=a.workers, device=a.device)
    env = T.startup(cfg)
    print(f"host {platform.node()}  gpu {env['gpu'].get('name')}  cpus (affinity) {len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()}  "
          f"torch threads {torch.get_num_threads()}  OMP {os.environ.get('OMP_NUM_THREADS')}  "
          f"loadavg {os.getloadavg()}", flush=True)
    device = torch.device(a.device)
    tr, _ = T.load_stores(cfg)
    norm = C.norm_from_buffer(tr.img)
    ts = K.TeacherStore(cfg.teacher_dir, tr, cfg.kd_gate, cfg.threads, print)
    pe = T.per_epoch(cfg)
    un = D.TrainItems(tr, "plain", cfg.seed, norm, pe, 0, pe, cfg.size, False, cfg.max_level,
                      teacher=ts, kd_margin=cfg.kd_margin)
    with T.one_thread():
        items = [un[q] for q in range(cfg.batch)]
    fixed = D.collate_train(items)
    print(f"fixed batch: {int(fixed['t_has'].sum())} teacher sides in {cfg.batch} images", flush=True)
    model, _ = T.build_model(cfg, device)
    opt = T.make_optimizer(model, cfg)
    model.train()
    res = {}
    for name, kd in (("A fixed batch, arm-3 step (no kd), phases", False),
                     ("B fixed batch, arm-4 step (kd), phases", True)):
        rows = [phase_step(model, opt, fixed, cfg, device, kd) for _ in range(a.steps + 5)][5:]
        res[name] = median_table(rows)
        print(f"{name}: {res[name]}", flush=True)
    for name, kd in (("C fixed batch, train_step() arm 3", False), ("D fixed batch, train_step() arm 4", True)):
        c = T.dataclasses.replace(cfg, kd=kd)
        ms = []
        for s in range(a.steps + 5):
            sync()
            t0 = time.perf_counter()
            T.train_step(model, opt, fixed, s, c, device)
            sync()
            ms.append((time.perf_counter() - t0) * 1000)
        print(f"{name}: median {statistics.median(ms[5:]):.1f} ms/step", flush=True)
    for name, kd in (("E real loader, arm 3", False), ("F real loader, arm 4", True)):
        c = T.dataclasses.replace(cfg, kd=kd)
        n = 2 * a.steps + 10
        ds = (D.TrainItems(tr, "plain", cfg.seed, norm, pe, 0, n * cfg.batch, cfg.size, True, cfg.max_level,
                           teacher=ts, kd_margin=cfg.kd_margin) if kd else
              D.TrainItems(tr, "plain", cfg.seed, norm, pe, 0, n * cfg.batch, cfg.size, True, cfg.max_level))
        it = iter(T.train_loader(ds, c))
        step_ms, wait_ms = [], []
        for s in range(n):
            tw = time.perf_counter()
            batch = next(it)
            ts_ = time.perf_counter()
            T.train_step(model, opt, batch, s, c, device)
            sync()
            step_ms.append((time.perf_counter() - ts_) * 1000)
            wait_ms.append((ts_ - tw) * 1000)
        del it
        print(f"{name}: median step {statistics.median(step_ms[10:]):.1f} ms, median wait "
              f"{statistics.median(wait_ms[10:]):.1f} ms  (loadavg {os.getloadavg()})", flush=True)
    g = torch.Generator(device=device).manual_seed(0)
    B, S = cfg.batch, cfg.size
    o = torch.randn(B, 3, S, S, device=device, generator=g, requires_grad=True)
    tl = torch.randn(B, 2, S, S, device=device, generator=g)
    rg = torch.rand(B, 2, S, S, device=device, generator=g) > 0.9
    vv = torch.ones(B, S, S, dtype=torch.bool, device=device)
    hh = torch.ones(B, 2, dtype=torch.bool, device=device)
    ms = []
    for _ in range(a.steps + 5):
        sync()
        t0 = time.perf_counter()
        K.kd_loss(o, tl, rg, vv, hh, 1.0)["total"].backward()
        sync()
        ms.append((time.perf_counter() - t0) * 1000)
    print(f"G kd_loss alone (forward + backward, random GPU tensors): median {statistics.median(ms[5:]):.1f} ms", flush=True)
    print("PROBE_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
