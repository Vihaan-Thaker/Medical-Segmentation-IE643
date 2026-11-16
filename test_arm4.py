#!/usr/bin/env python
"""CPU tests for the arm-4 training code and its evaluation (ARM4_CODE_PLAN.md §12; Part B = 15).

    python test_arm4.py        # exits 0 and prints ALL TESTS PASSED

Synthetic processed/ folder of test_student.py plus a synthetic teacher folder written through
teacher_io (smooth logits around each kidney; one kidney deliberately wrong, so the gate is
exercised). The frozen arm-3 code (frozen_arm3/) runs in subprocesses via arm3_reference.py.
"""
from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import math
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arm3_reference as R  # noqa: E402
import sam_metrics as M  # noqa: E402
import student_aug as A  # noqa: E402
import student_core as C  # noqa: E402
import student_data as D  # noqa: E402
import student_kd as K  # noqa: E402
import teacher_io as TIO  # noqa: E402
import teacher_maps as TM  # noqa: E402
import test_student as TS  # noqa: E402
import train_student as T  # noqa: E402

torch.set_num_threads(1)
check, raises, quiet = TS.check, TS.raises, TS.quiet
FRAME, FROZEN = TS.FRAME, HERE / "frozen_arm3"
TIMING = {"epoch_s", "train_s", "val_s", "data_wait_s", "peak_mem_gib"}


# ------------------------------------------------------------------------ fixtures
def write_teacher(tdir: Path, proc: Path, store, bad: tuple | None) -> dict:
    """A teacher folder in teacher_maps' exact layout. Logits: a smooth ramp around each kidney
    (Gaussian-blurred mask, float16-exact); `bad` = (case, z, side) gets a shifted map."""
    msha = D.manifest_sha256(proc)
    ident = {"teacher_sha256": TM.TEACHER_SHA256, "manifest_sha256": msha, "format": TIO.FORMAT_VERSION,
             "split": "train", "boxes": "clean (manifest), no jitter", "model_type": "vit_b"}
    tdir.mkdir(parents=True)
    T.atomic_write_json(tdir / TIO.STAMP, {**ident, "created_utc": "test"})
    rows, n_kid, dices = [], 0, {}
    for i, r in enumerate(store.rows):
        if not r["sam_eligible"]:
            continue
        boxes = store.boxes(i)
        _, msk = store.get(i)
        logits, dice = {}, {}
        for side, b in boxes.items():
            if b is None:
                continue
            gt = msk == side
            lg = 8.0 * (2.0 * ndimage.gaussian_filter(gt.astype(np.float64), 1.0) - 1.0)
            if bad == (r["case"], r["z"], side):
                lg = np.roll(lg, 12, axis=1)
            lg = lg.astype(np.float16).astype(np.float32)
            logits[side] = lg
            dice[str(side)] = M.dice_coefficient(gt, lg > 0)
            dices[(r["case"], r["z"], side)] = dice[str(side)]
            n_kid += 1
        meta = {**TIO.expected_meta(r, boxes, ident), "patient": r["patient"], "modality": r["modality"],
                "iou": {s: 0.9 for s in dice}, "dice": dice, "fp16_flips": {s: 0 for s in dice}, "written_utc": "test"}
        TIO.save_slice(tdir, meta, logits)
        rows.append({"case": r["case"], "z": r["z"]})
    with open(tdir / "teacher_manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["case", "z"])
        w.writeheader()
        w.writerows(rows)
    T.atomic_write_json(tdir / "run.json", {
        "status": "TEACHER_OK", "slices": len(rows), "kidneys": n_kid, "problems": [],
        "teacher_manifest_sha256": hashlib.sha256((tdir / "teacher_manifest.csv").read_bytes()).hexdigest(),
        **ident})
    return dices


def kd_cfg(proc: Path, run: Path, tdir: Path, exp: dict, **kw) -> T.Cfg:
    return TS.tiny_cfg(proc, run, expected_counts=exp, kd=True, teacher_dir=str(tdir), **kw)


def nontiming(rows: list) -> list:
    return [{k: v for k, v in r.items() if k not in TIMING} for r in rows]


def same_weights(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


# ============================================================================ tests
def test_frozen_identity(proc, exp, tmp, tr) -> Path:
    print("\n--- 1. arm 3 is bit-identical to the frozen arm-3 code ---")
    check(R.check_frozen(FROZEN) == [], "frozen_arm3/ holds the 7 recorded arm-3 files")
    norm = C.norm_from_buffer(tr.img)
    pe, total = 24, 72
    pos = R.spread_positions(total, 30)
    for v in C.VARIANTS:
        ref = R.frozen_items(FROZEN, proc, v, 643, pos, pe, FRAME, True, D.TRAIN_MAX_LEVEL, exp)
        ds = D.TrainItems(tr, v, 643, norm, pe, 0, total, FRAME, True)
        check(all(R.item_digest(ds[q]) == ref["items"][str(q)] for q in pos) and ref["norm"] == norm.as_dict(),
              f"{v}: {len(pos)} augmented training items bitwise equal to the frozen code's")
    cfgs = [TS.tiny_cfg(proc, tmp / "x" / "arm3_plain_s643", expected_counts=exp, variant=v) for v in C.VARIANTS]
    fs = R.frozen_stamps(FROZEN, [T.arm3_overrides(c, c.variant) for c in cfgs])
    mine = [{"stamp": T.stamp_of(c), "metric_fields": T.metric_fields(c.variant), "run_dir_name": T.run_dir_name(c)}
            for c in cfgs]
    check(json.loads(json.dumps(mine)) == fs, "stamp_of / metric_fields / run_dir_name equal the frozen code's")
    fdir, ndir = tmp / "frozen_run" / "arm3_plain_s643", tmp / "pair" / "arm3_plain_s643"
    cj = tmp / "tiny_cfg.json"
    cj.write_text(json.dumps(T.arm3_overrides(TS.tiny_cfg(proc, fdir, expected_counts=exp), "plain")))
    r = R.run_frozen(FROZEN, ["--mode", "tiny_train", "--cfg-json", cj])
    check(r.returncode == 0 and "TRAIN_OK" in r.stdout, "frozen arm-3 code: tiny training run TRAIN_OK (subprocess)")
    rc, _ = quiet(T.run_training, TS.tiny_cfg(proc, ndir, expected_counts=exp))
    lf, ln = (torch.load(d / "last.pt", weights_only=False) for d in (fdir, ndir))
    bf, bn = (torch.load(d / "best.pt", weights_only=False) for d in (fdir, ndir))
    check(rc == 0 and nontiming(T.read_metrics(fdir / "metrics.csv")) == nontiming(T.read_metrics(ndir / "metrics.csv"))
          and same_weights(lf["model"], ln["model"]) and same_weights(lf["optimizer"]["state"][0], ln["optimizer"]["state"][0])
          and same_weights(bf["model"], bn["model"]) and bf["epoch"] == bn["epoch"] and bf["score"] == bn["score"],
          "full tiny training: new code (kd off) == frozen code (metrics, last/best weights, optimizer state)")
    rj_f, rj_n = (json.loads((d / "run.json").read_text()) for d in (fdir, ndir))
    check(rj_f["settings"] == rj_n["settings"] and "kd" not in rj_n and "arm" not in rj_n,
          "run.json settings identical; no arm-4 keys in an arm-3 run")
    check(json.loads((fdir / "config.json").read_text())["code"] == R.FROZEN_SHA256,
          "the frozen run's config.json records exactly the frozen hashes (as arm 3's real runs)")
    return fdir          # the arm-4 partner in the following tests: a run made by the arm-3 code itself


def test_threads_real_frame(tmp: Path) -> None:
    print("\n--- 1b. frozen identity at the real 512 frame, parent with 4 threads ---")
    saved = TS.CASES
    TS.CASES = [("train", "ct", "amos_0001", (768, 768)), ("train", "ct", "amos_0002", (512, 512)),
                ("train", "mri", "amos_0501", (300, 400)), ("train", "mri", "amos_0502", (85, 260))]
    try:
        proc = tmp / "processed512"
        exp = TS.build_processed(proc)
    finally:
        TS.CASES = saved
    st, _ = quiet(D.SliceStore, proc, "train", exp, 2)
    norm = C.norm_from_buffer(st.img)
    pos = list(range(40))
    ref = R.frozen_items(FROZEN, proc, "plain", 643, pos, 40, 512, True, D.TRAIN_MAX_LEVEL, exp)["items"]
    ds = D.TrainItems(st, "plain", 643, norm, 40, 0, 40, 512, True)
    torch.set_num_threads(4)
    try:
        naive = {str(q): R.item_digest(ds[q]) for q in pos}
        fixed = T.item_digests(ds, pos)
        threads_after = torch.get_num_threads()
    finally:
        torch.set_num_threads(1)
    bad_naive, bad_fixed = T.item_mismatch(naive, ref), T.item_mismatch(fixed, ref)
    check(bad_naive["positions"] > 0 and bad_naive["x"] == bad_naive["positions"] and bad_naive["idx"] == 0
          and bad_naive["label"] == 0, f"sensitivity: with 4 threads {bad_naive['positions']}/40 images differ in the "
          "last bits (only x; idx/label/valid equal) -- the preflight's first P3 failure")
    check(bad_fixed["positions"] == 0 and threads_after == 4,
          "item_digests (one thread, as the DataLoader workers): bitwise equal to the frozen code; thread count restored")


def numpy_kd(out, tl, reg, val, has, T_):
    B = out.shape[0]
    total, n = 0.0, 0
    for b in range(B):
        z = out[b].astype(np.float64) / T_
        lse = np.log(np.exp(z).sum(0))
        for c, k in ((0, 1), (1, 2)):
            m = reg[b, c] & val[b]
            if not has[b, c] or m.sum() == 0:
                continue
            q = np.exp(z[k] - lse)
            p = 1.0 / (1.0 + np.exp(-tl[b, c].astype(np.float64) / T_))
            kl = p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
            total += T_ * T_ * kl[m].mean()
            n += 1
    return total / B, n


def test_loss() -> None:
    print("\n--- 2-5. kd_loss ---")
    g = torch.Generator().manual_seed(0)
    B, S = 3, 10
    out = torch.randn(B, 3, S, S, generator=g) * 3
    tl = torch.randn(B, 2, S, S, generator=g) * 4
    reg = torch.rand(B, 2, S, S, generator=g) > 0.4
    val = torch.ones(B, S, S, dtype=torch.bool)
    val[:, -2:] = False
    has = torch.tensor([[True, False], [True, True], [False, False]])
    for T_ in (1.0, 2.0):
        r = K.kd_loss(out, tl, reg, val, has, T_)
        ref, n = numpy_kd(out.numpy(), tl.numpy(), reg.numpy(), val.numpy(), has.numpy(), T_)
        check(abs(float(r["total"]) - ref) < 1e-5 and r["n_terms"] == n == 3 and r["n_images_with_terms"] == 2,
              f"T={T_:g}: equals a brute-force float64 reference ({float(r['total']):.6f} vs {ref:.6f}); "
              f"3 terms, divided by the full batch of 3")
    r2 = K.kd_loss(out[:2], tl[:2], reg[:2], val[:2], has[:2], 1.0)
    check(abs(float(r2["total"]) * 2 - float(K.kd_loss(out, tl, reg, val, has, 1.0)["total"]) * 3) < 1e-5,
          "an image without a teacher adds no term but counts in B")
    t = torch.randn(1, S, S, generator=g) * 5
    agree = torch.stack([torch.zeros(1, S, S), t, torch.full((1, S, S), -30.0)], 1)
    tl1 = torch.stack([t, torch.full((1, S, S), K.KD_FILL)], 1)
    full = torch.ones(1, 2, S, S, dtype=torch.bool)
    hv = torch.tensor([[True, False]])
    z0 = float(K.kd_loss(agree, tl1, full, torch.ones(1, S, S, dtype=torch.bool), hv, 1.0)["total"])
    z1 = float(K.kd_loss(agree, -tl1, full, torch.ones(1, S, S, dtype=torch.bool), hv, 1.0)["total"])
    check(abs(z0) < 1e-6 and z1 > 0.1, f"student = teacher -> {z0:.1e}; opposite teacher -> {z1:.3f}")
    m = (reg & val[:, None])
    out2, tl2 = out.clone(), tl.clone()
    out2[~(m.any(1))[:, None].expand_as(out2)] = 777.0
    tl2[~m] = -55.0
    check(abs(float(K.kd_loss(out2, tl2, reg, val, has, 1.0)["total"]) - float(K.kd_loss(out, tl, reg, val, has, 1.0)["total"])) < 1e-5,
          "values outside region & valid do not change the loss")
    has0 = has.clone()
    has0[1, 1] = False
    reg_e = reg.clone()
    reg_e[0, 0] = False
    r3 = K.kd_loss(out, tl, reg, val, has0, 1.0)
    r4 = K.kd_loss(out, tl, reg_e, val, has, 1.0)
    check(r3["n_terms"] == 2 and r4["n_terms"] == 2 and r4["n_empty_region"] == 1,
          "a gated / missing side adds no term; an empty region adds no term and is counted")
    x = out.clone().requires_grad_(True)
    K.kd_loss(x, tl, reg, val, has, 1.0)["total"].backward()
    used = torch.zeros(B, S, S, dtype=torch.bool)
    for b in range(B):
        for c in (0, 1):
            if has[b, c]:
                used[b] |= reg[b, c] & val[b]
    check(bool(torch.isfinite(x.grad).all()) and float(x.grad.abs().sum(1)[~used].max()) == 0.0
          and float(x.grad.abs().sum()) > 0, "gradients finite, exactly zero outside the used pixels")
    xe = (torch.randn(B, 3, S, S, generator=g) * 1e4).requires_grad_(True)
    le = K.kd_loss(xe, torch.sign(tl) * 60.0, reg, val, has, 1.0)["total"]
    le.backward()
    check(bool(torch.isfinite(le)) and bool(torch.isfinite(xe.grad).all()), "teacher +-60, student +-1e4: finite loss and gradients")
    check(raises(lambda: K.kd_loss(out, tl[:, :1], reg, val, has, 1.0)), "shape mismatch refused")
    # analytic gradient of the binary KL through the 3-class softmax (= soft-target BCE gradient):
    # dKL_k/dz_j = (q_k - p)/T x (1 if j == k else -q_j/(1 - q_k)), then x T^2 / n_region / B
    for T_ in (1.0, 2.0):
        x = out.clone().requires_grad_(True)
        K.kd_loss(x, tl, reg, val, has, T_)["total"].backward()
        q = torch.softmax(out.double() / T_, 1)
        want = torch.zeros_like(q)
        for b in range(B):
            for c, k in ((0, 1), (1, 2)):
                m = reg[b, c] & val[b]
                if not has[b, c] or not m.any():
                    continue
                p = torch.sigmoid(tl[b, c].double() / T_)
                qk = q[b, k]
                for j in range(3):
                    gj = (qk - p) / T_ * (1.0 if j == k else -q[b, j] / (1 - qk))
                    want[b, j] += torch.where(m, gj, torch.zeros_like(gj)) * T_ * T_ / m.sum() / B
        check(float((x.grad.double() - want).abs().max()) < 1e-6,
              f"T={T_:g}: autograd gradient == the analytic soft-target gradient (max diff "
              f"{float((x.grad.double() - want).abs().max()):.1e})")
    # minimising the term alone drives the student's P_R / P_L onto the teacher's probabilities
    pt = torch.rand(1, 2, S, S, generator=g) * torch.tensor([0.6, 0.35]).view(1, 2, 1, 1) + 0.02   # p_R + p_L < 1
    tt = torch.log(pt / (1 - pt))
    z = torch.zeros(1, 3, S, S, requires_grad=True)
    opt = torch.optim.Adam([z], lr=0.1)
    allr = torch.ones(1, 2, S, S, dtype=torch.bool)
    for _ in range(1500):
        opt.zero_grad()
        K.kd_loss(z, tt, allr, torch.ones(1, S, S, dtype=torch.bool), torch.tensor([[True, True]]), 1.0)["total"].backward()
        opt.step()
    pr = torch.softmax(z.detach(), 1)
    err = float((pr[:, 1:] - pt).abs().max())
    check(err < 0.01, f"optimising L_KD alone: student P_R, P_L -> the teacher's probabilities (max |diff| {err:.4f})")


def test_geometry() -> None:
    print("\n--- 6-8. region, letterbox, augmentation-follow ---")
    check(math.ceil(0.28 * 25) == 8 and K.region_box((10, 10, 34, 20), 100, 100, 0.28) == (3, 6, 41, 24)
          and K.region_box((10, 10, 14, 20), 100, 100, 0.2) == (9, 7, 15, 23),
          "margin uses exact arithmetic: ceil(0.28 x 25) = 7 (float gives 8); 0.2: ceil(1) = 1, ceil(2.2) = 3")
    check(K.region_box((0, 0, 4, 4), 50, 50, 0.2) == (0, 0, 5, 5) and K.region_box((45, 40, 49, 49), 50, 50, 0.5) == (42, 35, 49, 49),
          "clipped at every border")
    check(raises(lambda: K.region_box((0, 0, 60, 4), 50, 50, 0.2)), "box outside the slice refused")
    for hw in ((128, 80), (60, 100), (160, 160)):
        g = C.letterbox_geometry(*hw, FRAME)
        b = (5, 6, 20, 30)
        rb = C.box_to_frame(K.region_box(b, *hw, 0.2), g)
        fb = C.box_to_frame(b, g)
        check(rb[0] <= fb[0] and rb[1] <= fb[1] and rb[2] >= fb[2] and rb[3] >= fb[3], f"{hw}: frame region contains the frame box")
    norm = C.Norm(34.85, 58.93)
    ok = True
    for hw in ((FRAME, FRAME), (60, 100), (160, 160), (96, 128)):
        img = np.random.default_rng(sum(hw)).integers(0, 256, hw).astype(np.uint8)
        x, _, g = C.to_net_image(img, norm, FRAME)
        arr = ((torch.from_numpy(img).to(torch.float32) - norm.mean) / norm.std).numpy()
        ok &= torch.equal(K.to_net_soft(arr, g, norm.pad_value), x[0])
    for hw in ((512, 512), (768, 768), (85, 260), (468, 576)):          # real AMOS sizes, real frame
        img = np.random.default_rng(sum(hw)).integers(0, 256, hw).astype(np.uint8)
        x, _, g = C.to_net_image(img, norm, 512)
        arr = ((torch.from_numpy(img).to(torch.float32) - norm.mean) / norm.std).numpy()
        ok &= torch.equal(K.to_net_soft(arr, g, norm.pad_value), x[0])
    check(ok, "to_net_soft == to_net_image bit for bit (test frame, and 512 frame at 512/768/85x260/468x576)")
    S = 64
    g = {"h": S, "w": S, "h2": S, "w2": S, "m": S, "size": S}
    lab = torch.zeros(S, S, dtype=torch.int64)
    lab[torch.from_numpy(TS.disk(S, S, 32, 18, 7))] = 1
    lab[torch.from_numpy(TS.disk(S, S, 32, 46, 7))] = 2
    img = (lab > 0).float()[None] * 100.0 - 1.0
    valid = torch.ones(S, S, dtype=torch.bool)
    onehot = torch.stack([lab == 1, lab == 2])
    tl = torch.where(onehot, 8.0, -8.0)
    exact, dices, spatial = True, [], 0
    for k in range(300):
        im, lb, va, bx, ap = A.augment(img, lab, valid, torch.zeros(0, S, S), g, -1.0, np.random.default_rng(4000 + k))
        tlo, reg = K.follow_augmentation(tl, onehot, ap, g, va)
        exact &= torch.equal(reg[0], lb == 1) and torch.equal(reg[1], lb == 2)
        spatial += int(ap["rotation"] or ap["scaling"])
        for c in (0, 1):
            if (lb == c + 1).sum() > 20:
                dices.append(M.dice_coefficient((lb == c + 1).numpy(), ((tlo[c] > 0) & va).numpy()))
    check(exact and spatial > 60, f"nearest-followed one-hot == augment's own labels exactly ({spatial}/300 spatial draws)")
    check(float(np.mean(dices)) > 0.97 and min(dices) > 0.90,
          f"bilinear-followed teacher logits stay on the labels (mean Dice {np.mean(dices):.3f}, min {min(dices):.3f}; "
          "5-7 px test disks, so boundary pixels differ between bilinear and nearest)")


def test_items_and_store(proc, exp, tmp, tr, va, tdir, dices) -> None:
    print("\n--- 9-10. pairing, teacher store ---")
    ts, out = quiet(K.TeacherStore, tdir, tr, 0.80, 2)
    n_slices = sum(r["sam_eligible"] for r in tr.rows)
    n_kid = sum(sum(b is not None for b in tr.boxes(i).values()) for i in range(len(tr)) if tr.rows[i]["sam_eligible"])
    bad_mod = next(r["modality"] for r in tr.rows if r["case"] == BAD[0])
    check(len(ts.index) == n_slices and sum(ts.n_kidneys.values()) == n_kid
          and ts.n_gated[bad_mod] == 1 and sum(ts.n_gated.values()) == 1 and dices[BAD] < 0.8,
          f"store: {n_slices} slices / {n_kid} kidneys verified; exactly the one bad kidney gated (Dice {dices[BAD]:.2f})")
    norm = C.norm_from_buffer(tr.img)
    pe, total = 24, 72
    kd_ds = D.TrainItems(tr, "plain", 643, norm, pe, 0, total, FRAME, True, teacher=ts, kd_margin=0.2)
    a3_ds = D.TrainItems(tr, "plain", 643, norm, pe, 0, total, FRAME, True)
    same, n_t = True, 0
    st0 = (torch.get_rng_state(), np.random.get_state()[1].copy())
    for q in range(60):
        a, b = kd_ds[q], a3_ds[q]
        same &= torch.equal(a["x"], b["x"]) and torch.equal(a["label"], b["label"]) and torch.equal(a["valid"], b["valid"])
        n_t += int(a["t_has"].sum())
        same &= a["t_logit"].dtype == torch.float16 and a["t_region"].dtype == torch.bool
    st1 = (torch.get_rng_state(), np.random.get_state()[1])
    check(same and n_t > 20, f"60 arm-4 items == the arm-3 items (x, label, valid) bitwise; {n_t} teacher sides")
    check(torch.equal(st0[0], st1[0]) and np.array_equal(st0[1], st1[1]), "the teacher path draws no global random numbers")
    un = D.TrainItems(tr, "plain", 643, norm, pe, 0, total, FRAME, False, teacher=ts, kd_margin=0.2)
    al = []
    for q in range(40):
        al += T._kd_alignment(un[q])
    check(al and min(d for _, d, _, _ in al) > 0.85 and all(i == n for _, _, i, n in al),
          f"unaugmented: teacher (logit > 0) matches the label (min Dice {min(d for _, d, _, _ in al):.3f}); kidney inside the region")
    check(raises(lambda: D.collate_train([kd_ds[0], a3_ds[1]])), "a batch mixing teacher / no-teacher items is refused")
    cb = D.collate_train([a3_ds[0], a3_ds[1]])
    check(set(cb) == {"x", "label", "valid", "pos", "epoch", "i", "idx", "modality", "has_kidney"},
          "arm-3 batches carry exactly the arm-3 keys")
    check(raises(lambda: D.TrainItems(tr, "box", 643, norm, pe, 0, total, FRAME, True, teacher=ts)),
          "a teacher with the box variant is refused")

    def mutated(name, fn):
        d = tmp / f"t_{name}"
        shutil.copytree(tdir, d)
        fn(d)
        return raises(lambda: K.TeacherStore(d, tr, 0.80, 2))

    def edit_json(path, **kv):
        j = json.loads(path.read_text())
        j.update(kv)
        path.write_text(json.dumps(j))
    first = next(r for r in tr.rows if r["sam_eligible"])
    fpath = TIO.slice_path(tdir, first["case"], first["z"]).relative_to(tdir)

    def resave(d, key, val):
        meta, lg = TIO.load_slice(d, first["case"], first["z"])
        meta[key] = val(meta[key])
        TIO.save_slice(d, meta, {s: v.astype(np.float32) for s, v in lg.items()})
    cases = {
        "wrong teacher sha": lambda d: edit_json(d / TIO.STAMP, teacher_sha256="0" * 64),
        "wrong manifest sha": lambda d: edit_json(d / TIO.STAMP, manifest_sha256="0" * 64),
        "run.json not TEACHER_OK": lambda d: edit_json(d / "run.json", status="TEACHER_FAILED"),
        "count mismatch": lambda d: edit_json(d / "run.json", kidneys=1),
        "teacher_manifest changed": lambda d: (d / "teacher_manifest.csv").write_text("x\n"),
        "missing file": lambda d: (d / fpath).unlink(),
        "meta box differs": lambda d: resave(d, "boxes", lambda b: {k: [v[0] + 1, *v[1:]] for k, v in b.items()}),
        "stored Dice does not recompute": lambda d: resave(d, "dice", lambda x: {k: v - 0.01 for k, v in x.items()}),
    }
    for name, fn in cases.items():
        check(mutated(name.replace(" ", "_").replace(".", ""), fn), f"teacher store refuses: {name}")
    check(raises(lambda: K.TeacherStore(tdir, va, 0.80, 2)), "teacher store refuses a val-split store")


def test_config_and_training(proc, exp, tmp, stores, tdir, partner_dir) -> None:
    print("\n--- 11-12. configuration, pairing guard, distillation training ---")
    P = partner_dir.parent
    check(raises(lambda: T.check_kd_cfg(kd_cfg(proc, P / "arm4_box_s643", tdir, exp, variant="box"))),
          "--kd with the box variant refused")
    check(raises(lambda: T.check_kd_cfg(kd_cfg(proc, P / "x", "", exp))) and
          raises(lambda: T.check_kd_cfg(kd_cfg(proc, P / "x", tdir, exp, kd_lambda=0.0))), "missing teacher / lambda 0 refused")
    base = ["--processed", str(proc), "--run-dir", str(P / "arm4_plain_s643")]
    check(raises(lambda: T.main(base + ["--kd-lambda", "1"])), "a distillation setting without --kd refused (CLI)")
    check(raises(lambda: T.main(base + ["--kd", "--teacher-dir", str(tdir), "--kd-lambda", "1"])),
          "--kd without every setting refused (CLI)")
    c4 = kd_cfg(proc, P / "arm4_plain_s643", tdir, exp)
    s4, s3 = T.stamp_of(c4), T.stamp_of(dataclasses.replace(c4, kd=False))
    check(all(k in s4 for k in T.KD_SETTINGS) and not any(k in s3 for k in T.KD_SETTINGS) and "teacher_dir" not in s4,
          "stamp: kd keys only with kd on; the teacher path is never stamped")
    check(raises(lambda: T.run_training(kd_cfg(proc, P / "arm3_plain_s643", tdir, exp), stores)),
          "an arm-4 run in an arm3_* folder refused")
    # pairing guard
    check(raises(lambda: T.run_training(kd_cfg(proc, tmp / "lonely" / "arm4_plain_s643", tdir, exp), stores)),
          "no arm-3 partner -> refused")
    check(raises(lambda: T.run_training(kd_cfg(proc, P / "arm4_plain_s643", tdir, exp, lr=0.02), stores)),
          "partner trained with other settings -> refused")
    for what, fn in (("status", lambda d: (d / "run.json").write_text(json.dumps({**json.loads((d / "run.json").read_text()), "status": "x"}))),
                     ("initial weights", lambda d: (d / "state.json").write_text(json.dumps({"initial_digest": "0"}))),
                     ("normalisation", lambda d: (d / "run.json").write_text(json.dumps({**json.loads((d / "run.json").read_text()),
                                                                                       "norm": {"mean": 1.0}})))):
        q = tmp / f"p_{what.replace(' ', '_')}"
        shutil.copytree(partner_dir, q / "arm3_plain_s643")
        fn(q / "arm3_plain_s643")
        check(raises(lambda: T.run_training(kd_cfg(proc, q / "arm4_plain_s643", tdir, exp), stores)),
              f"partner with a different {what} -> refused")
    # training
    a4 = P / "arm4_plain_s643"
    rc, out = quiet(T.run_training, kd_cfg(proc, a4, tdir, exp), stores)
    rj = json.loads((a4 / "run.json").read_text())
    rows = T.read_metrics(a4 / "metrics.csv")
    check(rc == 0 and rj["status"] == "TRAIN_OK" and rj["arm"] == 4 and rj["kd"] == T.kd_settings(c4)
          and rj["teacher"]["teacher_sha256"] == TM.TEACHER_SHA256 and rj["partner"]["path"] == "arm3_plain_s643",
          "distilled tiny run: TRAIN_OK; run.json records arm 4, kd settings, teacher, partner")
    check(list(rows[0].keys()) == T.metric_fields("plain", True) and all(float(r["train_kd"]) >= 0 for r in rows[1:])
          and sum(int(r["kd_terms"]) for r in rows[1:]) > 0 and "kd " in out,
          f"metrics carry the KD columns ({sum(int(r['kd_terms']) for r in rows[1:])} terms)")
    check(all(abs(float(r["train_loss"]) - float(r["train_seg"]) - float(r["train_kd"])) < 1e-4 for r in rows[1:]),
          "train_loss = train_seg + lambda x train_kd (lambda 1)")
    ck = torch.load(a4 / "best.pt", weights_only=False)
    model, norm, variant, size, _ = C.load_student(a4 / "best.pt")
    check(ck["kd"] == T.kd_settings(c4) and variant == "plain" and size == FRAME, "best.pt records kd; load_student reads it")
    p3 = json.loads((partner_dir / "state.json").read_text())["initial_digest"]
    check(json.loads((a4 / "state.json").read_text())["initial_digest"] == p3, "arm-4 initial weights == arm-3 partner's")
    lw3 = torch.load(partner_dir / "last.pt", weights_only=False)["model"]
    check(not same_weights(torch.load(a4 / "last.pt", weights_only=False)["model"], lw3),
          "the distillation term changes training (final weights differ from arm 3)")
    rc, out = quiet(T.run_training, kd_cfg(proc, a4, tdir, exp), stores)
    check(rc == 0 and "TRAIN_OK" in out and "epoch 1/" not in out, "finished arm-4 run: TRAIN_OK without training")
    # resume exactness and changed settings
    b4 = tmp / "resume" / "arm4_plain_s643"
    shutil.copytree(partner_dir, b4.parent / "arm3_plain_s643")
    quiet(T.run_training, kd_cfg(proc, b4, tdir, exp, stop_after_epoch=1), stores)
    check(raises(lambda: T.run_training(kd_cfg(proc, b4, tdir, exp, kd_lambda=2.0), stores)), "resume with another lambda refused")
    check(raises(lambda: T.run_training(kd_cfg(proc, b4, tdir, exp, kd_gate=0.5), stores)), "resume with another gate refused")
    quiet(T.run_training, kd_cfg(proc, b4, tdir, exp), stores)
    check(same_weights(torch.load(a4 / "last.pt", weights_only=False)["model"], torch.load(b4 / "last.pt", weights_only=False)["model"])
          and nontiming(T.read_metrics(a4 / "metrics.csv")) == nontiming(T.read_metrics(b4 / "metrics.csv")),
          "resume exactness with distillation: 1 + resume + 1 == 2 straight")
    rc, out = quiet(T.run_training, TS.tiny_cfg(proc, partner_dir, expected_counts=exp), stores)
    check(rc == 0 and "TRAIN_OK" in out and "epoch 1/" not in out, "the finished arm-3 run, resubmitted with the new code: TRAIN_OK")


def test_preflight(proc, exp, tmp, stores, tdir, partner_dir) -> None:
    print("\n--- 13. arm-4 preflight (reduced) ---")
    run = partner_dir.parent / "arm4_preflight"
    small = dict(pf_kd_items=6, pf_kd_steps=4, pf_probe=4, pf_kd_frozen=10, pf_kd_ratio=(0.0, 1e9))
    rc, out = quiet(T.run_preflight, kd_cfg(proc, run, tdir, exp, **small), stores)
    pf = json.loads((run / "preflight.json").read_text())
    want = {"P2_data", "P3_frozen_arm3_identity", "P4_teacher_store", "P5_alignment_unaugmented",
            "P6_alignment_augmented", "P7_pairing", "P10_real_steps"}
    check(rc == 0 and "PREFLIGHT_OK" in out and want <= set(pf["checks"]) and all(v["ok"] for v in pf["checks"].values()),
          f"PREFLIGHT_OK, every check passes ({len(pf['checks'])})")
    check((run / "kd_examples.png").exists() and pf["softness"] and pf["estimate"]["recommended_ARM4_TRAIN_TIME"].endswith(":00:00"),
          "kd_examples.png, softness and a time recommendation written")
    for name, kw, failing in (("ratio band", {"pf_kd_ratio": (1e8, 1e9)}, "P10_real_steps"),
                              ("gate fraction", {"pf_kd_gate_fail": 0.0}, "P4_teacher_store"),
                              ("alignment", {"pf_kd_align": 1.01}, "P5_alignment_unaugmented")):
        r2 = tmp / f"pf_{failing}" / "arm4_preflight"
        shutil.copytree(partner_dir, r2.parent / "arm3_plain_s643")
        rc, out = quiet(T.run_preflight, kd_cfg(proc, r2, tdir, exp, **{**small, **kw}), stores)
        pf2 = json.loads((r2 / "preflight.json").read_text())
        check(rc == 1 and "PREFLIGHT_FAILED" in out and not pf2["checks"][failing]["ok"], f"{name} out of bounds -> {failing} FAILS")


def test_sbatch(tmp: Path) -> None:
    print("\n--- 14. SLURM scripts ---")
    names = ["arm4_preflight.sbatch", "arm4_train_s643.sbatch", "arm4_train_s644.sbatch", "arm4_train_s645.sbatch"]
    regen = tmp / "regen"
    regen.mkdir()
    subprocess.run([sys.executable, str(HERE / "gen_sbatch_arm4_train.py"), str(regen)], check=True, capture_output=True)
    check(sorted(p.name for p in regen.glob("*.sbatch")) == names, "generator writes exactly the 4 arm-4 scripts")
    for n in names:
        txt = (HERE / n).read_text()
        check(txt == (regen / n).read_text() and subprocess.run(["bash", "-n", str(HERE / n)]).returncode == 0,
              f"{n} == generator output, passes bash -n")
        check("#SBATCH --partition=dgx" in txt and "*A100*" in txt and "--expect-gpu A100" in txt
              and "--kd --teacher-dir" in txt and "--kd-lambda 1.0 --kd-temperature 1.0 --kd-margin 0.2 --kd-gate 0.8" in txt,
              f"{n}: dgx / A100 guards and every distillation setting")
    stub = tmp / "stub"
    stub.mkdir()
    calls = tmp / "calls.txt"
    (stub / "squeue").write_text('#!/bin/bash\necho "${SQUEUE_OUT:-}"\n')
    (stub / "nvidia-smi").write_text('#!/bin/bash\necho "${GPU_NAME:-NVIDIA A100-SXM4-80GB}"\n')
    (stub / "py").write_text(f'#!/bin/bash\nif [ "$1" = "-c" ]; then echo env-ok; exit 0; fi\necho "$@" >> {calls}\nexit 0\n')
    for f in stub.iterdir():
        f.chmod(f.stat().st_mode | stat.S_IEXEC)
    seg = tmp / "seg"
    for d in ("data/processed", "teacher/arm2_r4_train", "runs/arm3_plain_s643", "runs/arm3_plain_s644", "runs/arm3_plain_s645"):
        (seg / d).mkdir(parents=True)
    (seg / "data/processed/manifest.csv").write_text("x\n")
    (seg / "teacher/arm2_r4_train/run.json").write_text("{}")
    for s in (643, 644, 645):
        (seg / f"runs/arm3_plain_s{s}/run.json").write_text("{}")

    def run(name, **env):
        t = (HERE / name).read_text().replace("SEG=/home/rethinkingai-self/vihaan/Segmentation", f"SEG={seg}")
        t = t.replace("PY=/home/rethinkingai-self/vihaan/miniconda3/envs/seg311/bin/python", f"PY={stub / 'py'}")
        sp = tmp / f"run_{name}"
        sp.write_text(t)
        calls.unlink(missing_ok=True)
        r = subprocess.run(["bash", str(sp)], env={**os.environ, "PATH": f"{stub}:{os.environ['PATH']}", **env},
                           capture_output=True, text=True)
        return r.returncode, (calls.read_text().split() if calls.exists() else [])
    rc, argv = run("arm4_train_s644.sbatch")
    check(rc == 0 and argv[0].endswith("scripts/train_student.py") and argv[argv.index("--seed") + 1] == "644"
          and argv[argv.index("--run-dir") + 1] == f"{seg}/runs/arm4_plain_s644"
          and argv[argv.index("--teacher-dir") + 1] == f"{seg}/teacher/arm2_r4_train" and "--preflight" not in argv,
          "train_s644: seed 644 -> runs/arm4_plain_s644 with the teacher folder")
    rc, argv = run("arm4_preflight.sbatch")
    check(rc == 0 and "--preflight" in argv and argv[argv.index("--run-dir") + 1] == f"{seg}/runs/arm4_preflight",
          "preflight command")
    rc, argv = run("arm4_train_s643.sbatch", SQUEUE_OUT="9 arm4_t644 RUNNING\n10 arm4_t645 RUNNING")
    check(rc == 0 and argv, "the three seed scripts run alongside each other")
    rc, argv = run("arm4_train_s643.sbatch", SQUEUE_OUT="9 arm4_pre RUNNING")
    check(rc == 3 and not argv, "training refuses next to the preflight")
    rc, argv = run("arm4_train_s643.sbatch", SQUEUE_OUT="9 arm4_t643 RUNNING")
    check(rc == 3 and not argv, "a second copy of the same seed is refused")
    rc, argv = run("arm4_preflight.sbatch", SQUEUE_OUT="9 arm4_t645 RUNNING")
    check(rc == 3, "the preflight refuses next to a training job")
    rc, argv = run("arm4_train_s645.sbatch", GPU_NAME="NVIDIA A40")
    check(rc == 4 and not argv, "an A40 node is refused")
    (seg / "runs/arm3_plain_s645/run.json").unlink()
    rc, argv = run("arm4_train_s645.sbatch")
    check(rc == 2 and not argv, "a missing arm-3 partner stops the script before python")
    (seg / "teacher/arm2_r4_train/run.json").unlink()
    rc, argv = run("arm4_train_s643.sbatch")
    check(rc == 2 and not argv, "missing teacher maps stop the script before python")


def make_kd_run(root: Path, proc: Path, exp: dict, va, seed: int) -> Path:
    """A finished arm-4 training folder: test_student_eval's run fixture with different weights
    (made under seed + 1000, then re-labelled), plus run.json's kd block and teacher identity."""
    import eval_student as ES
    import test_student_eval as TSE
    src = TSE.make_run(root / "kdsrc", proc, exp, va, "plain", seed + 1000)
    ck = torch.load(src / "best.pt", map_location="cpu", weights_only=False)
    ck["seed"] = seed
    torch.save(ck, src / "best.pt")
    rj = json.loads((src / "run.json").read_text())
    rj.update(seed=seed, best_sha256=ES.sha256_file(src / "best.pt"), arm=4,
              kd={"lambda": 1.0, "temperature": 1.0, "margin": 0.2, "gate": 0.8},
              teacher={"teacher_sha256": TM.TEACHER_SHA256, "manifest_sha256": D.manifest_sha256(proc)})
    (src / "run.json").write_text(json.dumps(rj))
    dst = root / "runs" / f"arm4_plain_s{seed}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return dst


def test_part_b(tmp: Path) -> None:
    print("\n--- 15. Part B: evaluation of arm 4 ---")
    import eval_sam as E
    import eval_student as ES
    import seed_summary as SS
    import test_student_eval as TSE
    root = tmp / "partb"
    old_cases, old_groups = TS.CASES, dict(E.EXPECTED_GROUPS)
    try:
        proc, exp, rows, cases, va = TSE.setup(root)
        kd = {s: make_kd_run(root, proc, exp, va, s) for s in (643, 644)}
        a3 = {s: TSE.make_run(root, proc, exp, va, "plain", s) for s in (643, 644)}

        tr = ES.check_training(kd[643], proc)
        check(tr["ident"]["model_type"] == "student_plain_kd"
              and tr["ident"]["checkpoint_name"] == "arm4_plain_s643/best.pt",
              "check_training accepts arm4_plain_s643: model type student_plain_kd")
        check(ES.check_training(a3[643], proc)["ident"]["model_type"] == "student_plain",
              "an arm-3 run is unchanged: student_plain")

        def refused(mutate, folder: str, why: str, msg: str) -> None:
            d = root / "bad" / folder
            shutil.rmtree(d.parent, ignore_errors=True)
            shutil.copytree(kd[643], d)
            rj = json.loads((d / "run.json").read_text())
            mutate(rj)
            (d / "run.json").write_text(json.dumps(rj))
            try:
                quiet(ES.check_training, d, proc)
                check(False, msg)
            except SystemExit as e:
                check(why in str(e), msg + f" ({str(e)[:90]})" if why not in str(e) else msg)
        refused(lambda rj: None, "arm3_plain_s643", "does not match", "refuses a distilled run in an arm3_ folder")
        refused(lambda rj: rj.pop("kd"), "arm4_plain_s643", "does not match", "refuses an arm4_ folder without a kd block")
        refused(lambda rj: rj["kd"].pop("gate"), "arm4_plain_s643", "kd block incomplete", "refuses an incomplete kd block")
        refused(lambda rj: rj["teacher"].update(teacher_sha256="0" * 64), "arm4_plain_s643", "teacher_sha256",
                "refuses a run distilled from another teacher")

        out4 = {s: root / "outputs" / f"arm4_plain_s{s}" for s in kd}
        out3 = {s: root / "outputs" / f"arm3_plain_s{s}" for s in a3}
        for s in kd:
            rc4, _ = TSE.evaluate(proc, kd[s], out4[s], exp)
            rc3, _ = TSE.evaluate(proc, a3[s], out3[s], exp)
            check(rc4 == 0 and rc3 == 0, f"seed {s}: arm-4 and arm-3 evaluations end EVAL_OK")
        ej = json.loads((out4[643] / "run.json").read_text())
        check(ej["model"]["model_type"] == "student_plain_kd" and ej["kd"]["gate"] == 0.8 and ej["arm"] == 4
              and ej["teacher_sha256"] == TM.TEACHER_SHA256 and "kd" not in json.loads((out3[643] / "run.json").read_text()),
              "eval run.json records the kd settings (arm 4 only)")
        i4, i3 = TSE.read(out4[643] / "instances.csv"), TSE.read(out3[643] / "instances.csv")
        keys = ("case", "z", "side", "jitter", "box_clean", "box_jit")
        check([[r[k] for k in keys] for r in i4] == [[r[k] for k in keys] for r in i3]
              and any(a["dice"] != b["dice"] for a, b in zip(i4, i3)),
              "arm 4 is scored on the identical instances and boxes as arm 3 (with different predictions)")

        so = root / "outputs" / "arm4_seeds"
        argv = ["--runs", str(out4[643]), str(out4[644]), "--ref-runs", str(out3[644]), str(out3[643]),
                "--label-ref", "Arm 3 plain student", "--out", str(so), "--n-boot", "200"]
        rc, log = quiet(SS.main, argv)
        check(rc == 0 and "SUMMARY_OK" in log and "# Arm 4 distilled plain student" in log and "Seed-paired" in log,
              "seed_summary --ref-runs: SUMMARY_OK, arm-4 title")
        t2 = [r for r in TSE.read(so / "seed_compare_vs_arm_3_plain_student.csv")
              if r["level"] == "2d" and r["metric"] == "dice" and abs(float(r["jitter"]) - 0.10) < 1e-9]

        def pmeans(o: Path) -> dict:
            acc: dict = {}
            for r in TSE.read(o / "instances.csv"):
                if abs(float(r["jitter"]) - 0.10) < 1e-9:
                    g = "/".join(x for x in (r["dataset"], r["modality"], r["sequence"]) if x)
                    acc.setdefault(g, {}).setdefault(r["patient"], []).append(float(r["dice"]))
            return {g: {p_: sum(v) / len(v) for p_, v in d.items()} for g, d in acc.items()}
        p4, p3 = {s: pmeans(out4[s]) for s in kd}, {s: pmeans(out3[s]) for s in a3}
        ok = bool(t2)
        for row in t2:
            g = row["group"]
            pats = sorted(p3[643][g])
            ref = [(p3[643][g][q] + p3[644][g][q]) / 2 for q in pats]
            stu = [(p4[643][g][q] + p4[644][g][q]) / 2 for q in pats]
            ok &= abs(float(row["ref_mean"]) - round(sum(ref) / len(ref), 4)) < 1e-9
            ok &= abs(float(row["delta"]) - round(sum(b - a for a, b in zip(ref, stu)) / len(pats), 4)) < 1e-9
            for s in kd:     # each seed against its OWN partner, not the seed-mean reference
                want = sum(p4[s][g][q] - p3[s][g][q] for q in pats) / len(pats)
                ok &= abs(float(row[f"delta_arm4_plain_s{s}"]) - round(want, 4)) < 1e-9
        check(ok, "--ref-runs: reference = mean over arm-3 seeds, Δ and per-seed Δ (vs own partner) = hand computation")
        rc, log = quiet(SS.main, ["--runs", str(out4[643]), str(out4[644]), "--ref-runs", str(out3[643]),
                                  "--out", str(so), "--n-boot", "50"])
        check(rc == 1 and "seed sets must be identical" in log, "--ref-runs refuses mismatched seed sets")
        rc, log = quiet(SS.main, ["--runs", str(out4[643]), str(out3[644]), "--ref-runs", str(out3[643]), str(out4[644]),
                                  "--out", str(so), "--n-boot", "50"])
        check(rc == 1 and "plain student, all of one type" in log, "refuses arm-3 and arm-4 runs mixed as seeds")
        rc, log = quiet(SS.main, ["--runs", str(out4[643]), str(out4[644]), "--ref", str(out3[643]),
                                  "--out", str(so), "--n-boot", "50"])
        check(rc == 0 and (so / "seed_summary_vs_arm3_plain_s643.md").read_text().startswith("# Arm 4 distilled"),
              "--ref mode works on arm-4 seeds (single reference)")
    finally:
        TS.CASES = old_cases
        E.EXPECTED_GROUPS = old_groups


def test_eval_sbatch(tmp: Path) -> None:
    print("\n--- 15b. arm-4 evaluation SLURM scripts ---")
    names = ["arm4_compare.sbatch", "arm4_eval_a.sbatch", "arm4_eval_b.sbatch"]
    regen = tmp / "regen_eval"
    regen.mkdir()
    subprocess.run([sys.executable, str(HERE / "gen_sbatch_arm4_eval.py"), str(regen)], check=True, capture_output=True)
    check(sorted(p.name for p in regen.glob("*.sbatch")) == names, "generator writes exactly the 3 evaluation scripts")
    for n in names:
        txt = (HERE / n).read_text()
        check(txt == (regen / n).read_text() and subprocess.run(["bash", "-n", str(HERE / n)]).returncode == 0
              and "#SBATCH --partition=a40" in txt and "*A40*" in txt, f"{n}: == generator, bash -n, a40/A40")
    stub, calls = tmp / "stub_eval", tmp / "calls_eval.txt"
    stub.mkdir()
    (stub / "squeue").write_text('#!/bin/bash\necho "${SQUEUE_OUT:-}"\n')
    (stub / "nvidia-smi").write_text('#!/bin/bash\necho "${GPU_NAME:-NVIDIA A40}"\n')
    (stub / "py").write_text('#!/bin/bash\nif [ "$1" = "-c" ]; then echo env-ok; exit 0; fi\n'
                             f'echo "$@" >> {calls}\n'
                             'if [ -n "${FAIL_MATCH:-}" ] && echo "$@" | grep -q -- "$FAIL_MATCH"; then exit 7; fi\nexit 0\n')
    for f in stub.iterdir():
        f.chmod(f.stat().st_mode | stat.S_IEXEC)
    seg = tmp / "seg_eval"
    (seg / "data/processed").mkdir(parents=True)
    (seg / "data/processed/manifest.csv").write_text("x\n")
    for d in ("arm1_vit_b", "arm2_r4", "arm3_plain_s643", "arm3_plain_s644", "arm3_plain_s645"):
        (seg / "outputs" / d).mkdir(parents=True)
        (seg / "outputs" / d / "run.json").write_text("{}")

    def run(name, **env):
        t = (HERE / name).read_text().replace("SEG=/home/rethinkingai-self/vihaan/Segmentation", f"SEG={seg}")
        t = t.replace("PY=/home/rethinkingai-self/vihaan/miniconda3/envs/seg311/bin/python", f"PY={stub / 'py'}")
        sp = tmp / "run_eval.sh"
        sp.write_text(t)
        calls.unlink(missing_ok=True)
        r = subprocess.run(["bash", str(sp)], env={**os.environ, "PATH": f"{stub}:{os.environ['PATH']}", **env},
                           capture_output=True, text=True)
        return r.returncode, (calls.read_text().splitlines() if calls.exists() else []), r.stdout
    rc, cl, _ = run("arm4_eval_a.sbatch")
    steps = [c.split()[0].rsplit("/", 1)[-1] for c in cl]
    check(rc == 0 and steps == ["eval_student.py", "aggregate_arm.py"] + ["compare_arms.py"] * 3
          + ["eval_student.py", "aggregate_arm.py"] + ["compare_arms.py"] * 3
          and f"{seg}/runs/arm4_plain_s643" in cl[0] and "--resume" in cl[0] and "--expect-gpu A40" in cl[0]
          and f"--a {seg}/outputs/arm3_plain_s643 --b {seg}/outputs/arm4_plain_s643" in cl[4]
          and f"{seg}/runs/arm4_plain_s644" in cl[5] and f"--a {seg}/outputs/arm3_plain_s644" in cl[9],
          "eval_a: s643 then s644, each vs arm 1b, arm 2 and its own arm-3 seed")
    rc, cl, _ = run("arm4_eval_b.sbatch")
    check(rc == 0 and len(cl) == 5 and f"{seg}/runs/arm4_plain_s645" in cl[0] and "arm3_plain_s645" in cl[-1],
          "eval_b: s645 vs arm 1b, arm 2 and arm-3 s645")
    rc, cl, out = run("arm4_eval_a.sbatch", FAIL_MATCH="arm4_plain_s643 --out")
    check(rc == 7 and len(cl) == 1 and "STOPPING" in out, "a failing evaluation stops the script")
    rc, cl, out = run("arm4_eval_a.sbatch", FAIL_MATCH="--a .*arm3_plain_s643")
    check(rc == 5 and sum("eval_student.py" in c for c in cl) == 2 and "a comparison FAILED" in out,
          "a failing comparison: next seed still evaluated, exit 5")
    rc, cl, _ = run("arm4_compare.sbatch")
    check(rc == 0 and len(cl) == 3 and all("seed_summary.py" in c for c in cl) and "--ref-runs" in cl[2]
          and all(f"arm4_plain_s{s}" in cl[2] and f"arm3_plain_s{s}" in cl[2] for s in SEEDS_EVAL)
          and "outputs/arm4_seeds" in cl[2], "compare: vs arm 1b, vs arm 2, then seed-paired vs arm 3")
    for job in ("arm4_t644", "arm4_ea"):
        rc, cl, _ = run("arm4_eval_a.sbatch", SQUEUE_OUT=f"9 {job} RUNNING")
        check(rc == 3 and not cl, f"evaluation refused next to {job}")
    rc, cl, _ = run("arm4_eval_a.sbatch", SQUEUE_OUT="9 arm4_eb RUNNING")
    check(rc == 0 and cl, "the two evaluation scripts may run together")
    rc, cl, _ = run("arm4_compare.sbatch", SQUEUE_OUT="9 arm4_eb RUNNING")
    check(rc == 3 and not cl, "compare refused while an evaluation runs")
    rc, cl, _ = run("arm4_eval_b.sbatch", GPU_NAME="NVIDIA A100-SXM4-80GB")
    check(rc == 4 and not cl, "non-A40 node refused")


SEEDS_EVAL = (643, 644, 645)


def test_isolation() -> None:
    print("\n--- 16. isolation, Python 3.10 compile ---")
    code = ("import sys; sys.path.insert(0, %r); import train_student, student_data; "
            "print([m for m in ('student_kd', 'teacher_maps', 'teacher_io', 'arm3_reference', 'segment_anything') "
            "if m in sys.modules])") % str(HERE)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    check(r.returncode == 0 and r.stdout.strip() == "[]", "arm-3 imports load no arm-4 / SAM module (import graph unchanged)")
    py310 = Path("/opt/anaconda3/envs/ner_env/bin/python")
    files = ["student_kd.py", "student_data.py", "train_student.py", "arm3_reference.py", "gen_sbatch_arm4_train.py",
             "test_arm4.py", "eval_student.py", "seed_summary.py", "gen_sbatch_arm4_eval.py"]
    if py310.exists():
        r = subprocess.run([str(py310), "-m", "py_compile", *[str(HERE / f) for f in files]], capture_output=True, text=True)
        check(r.returncode == 0, f"all arm-4 files compile under Python 3.10 {r.stderr.strip()}")
    else:
        print("  (Python 3.10 not available here; compile check done on the Mac)")


BAD: tuple = ()


def main() -> int:
    global BAD
    tmp = Path(tempfile.mkdtemp(prefix="arm4_test_"))
    try:
        proc = tmp / "processed"
        exp = TS.build_processed(proc)
        tr, _ = quiet(D.SliceStore, proc, "train", exp, 2)
        va, _ = quiet(D.SliceStore, proc, "val", exp, 2)
        r = next(r for r in tr.rows if r["sam_eligible"] and r["modality"] == "mri" and r["rk_box"] is not None)
        BAD = (r["case"], r["z"], 1)
        tdir = tmp / "teacher"
        dices = write_teacher(tdir, proc, tr, BAD)
        partner = test_frozen_identity(proc, exp, tmp, tr)
        test_threads_real_frame(tmp)
        test_loss()
        test_geometry()
        test_items_and_store(proc, exp, tmp, tr, va, tdir, dices)
        test_config_and_training(proc, exp, tmp, (tr, va), tdir, partner)
        test_preflight(proc, exp, tmp, (tr, va), tdir, partner)
        test_sbatch(tmp)
        test_part_b(tmp)
        test_eval_sbatch(tmp)
        test_isolation()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\nALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
