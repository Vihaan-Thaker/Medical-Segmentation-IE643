#!/usr/bin/env python
"""Tests for arm 2 (ARM2_CODE_PLAN.md §7): LoRA, the evaluation-identical forward path, data,
losses, schedule, selection metric, resume, run stamp, finalize/merge, rank selection, GPU
guards and the generated SLURM scripts. CPU only, a tiny SAM, a synthetic processed/ folder.

    python test_train_sam.py        # exits 0 and prints ALL TESTS PASSED

Needs the official segment_anything (as on Prajna), not the MedSAM fork.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import contextlib
import csv
import io
import json
import shutil
import subprocess
import sys
import tempfile
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eval_sam as E  # noqa: E402
import sam_lora as L  # noqa: E402
import sam_metrics as M  # noqa: E402
import sam_train_core as C  # noqa: E402
import select_rank as S  # noqa: E402
import train_data as D  # noqa: E402
import train_sam_lora as T  # noqa: E402
from segment_anything import SamPredictor  # noqa: E402
from segment_anything.modeling import (ImageEncoderViT, MaskDecoder, PromptEncoder, Sam,  # noqa: E402
                                       TwoWayTransformer)

torch.set_num_threads(1)


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)
    print(f"  ok  {msg}")


def raises(fn, exc=(SystemExit, AssertionError, ValueError, KeyError, RuntimeError, TypeError)) -> bool:
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            fn()
    except exc:
        return True
    return False


# ---------------------------------------------------------------- tiny SAM + synthetic data
def tiny_sam(seed: int = 0) -> Sam:
    torch.manual_seed(seed)
    enc = ImageEncoderViT(img_size=64, patch_size=16, in_chans=3, embed_dim=64, depth=2, num_heads=2,
                          mlp_ratio=4, out_chans=32, qkv_bias=True,
                          norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU,
                          use_abs_pos=True, use_rel_pos=True, rel_pos_zero_init=False,
                          window_size=14, global_attn_indexes=(1,))
    pe = PromptEncoder(embed_dim=32, image_embedding_size=(4, 4), input_image_size=(64, 64),
                       mask_in_chans=16)
    md = MaskDecoder(num_multimask_outputs=3,
                     transformer=TwoWayTransformer(depth=2, embedding_dim=32, mlp_dim=64, num_heads=2),
                     transformer_dim=32, iou_head_depth=3, iou_head_hidden_dim=32)
    return Sam(image_encoder=enc, prompt_encoder=pe, mask_decoder=md,
               pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375]).eval()


def tiny_plain_builder(ck, device):
    m = tiny_sam()
    m.load_state_dict(torch.load(ck, map_location="cpu"))
    return m.to(device)


def disk(h, w, cy, cx, r):
    yy, xx = np.mgrid[0:h, 0:w]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r


def build_processed(root: Path) -> dict:
    """Train/val/test rows, CT (48x48) and MRI (40x56, non-square), both kidneys, a
    right-only slice with a left sliver (no left box), and a kidney-free slice."""
    rows = []
    (root / "slices").mkdir(parents=True)
    plan = [("amos_0001", "train", "ct"), ("amos_0002", "train", "ct"),
            ("amos_0501", "train", "mri"), ("amos_0502", "train", "mri"),
            ("amos_0003", "val", "ct"), ("amos_0503", "val", "mri"),
            ("amos_0004", "test", "ct")]
    rng = np.random.default_rng(7)
    for case, split, mod in plan:
        H, W = (48, 48) if mod == "ct" else (40, 56)
        sp = (0.8, 0.8) if mod == "ct" else (1.2, 1.2)
        cdir = root / "slices" / "amos" / case
        cdir.mkdir(parents=True)
        for z in range(5):
            msk = np.zeros((H, W), np.uint8)
            img = (rng.random((H, W)) * 60).astype(np.uint8)
            if 1 <= z <= 3:
                msk[disk(H, W, H // 2, W // 4, 7)] = M.RIGHT
                msk[disk(H, W, H // 2, 3 * W // 4, 7)] = M.LEFT
            elif z == 4:
                msk[disk(H, W, H // 2, W // 4, 7)] = M.RIGHT
                msk[disk(H, W, H // 2, 3 * W // 4, 2)] = M.LEFT        # sliver, < 50 px
            img[msk > 0] = 200
            rel = f"slices/amos/{case}/{z:04d}.npz"
            np.savez_compressed(root / rel, img=img, msk=msk)

            def bbox(m):
                if m.sum() < 50:
                    return None
                ys, xs = np.nonzero(m)
                return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            rb, lb = bbox(msk == M.RIGHT), bbox(msk == M.LEFT)
            rows.append({"dataset": "amos", "patient": case, "case": case, "sequence": "",
                         "modality": mod, "split": split, "z": z, "n_slices": 5,
                         "has_kidney": int(msk.any()), "rk_area": int((msk == M.RIGHT).sum()),
                         "lk_area": int((msk == M.LEFT).sum()), "rk_box": json.dumps(rb),
                         "lk_box": json.dumps(lb), "sam_eligible": int(rb is not None or lb is not None),
                         "h": H, "w": W, "sp_row": sp[0], "sp_col": sp[1], "sp_slice": 3.0, "path": rel})
    with open(root / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return {"rows": rows}


def tiny_cfg(tmp: Path, proc: Path, ck: Path, **kw) -> T.Cfg:
    base = dict(processed=str(proc), checkpoint=str(ck), rank=4, alpha=4.0,
                run_dir=str(tmp / "run"), merged_out=str(tmp / "merged_tiny.pth"),
                epochs=2, batch=2, accum=1, warmup=2, workers=0, val_batch=3, device="cpu",
                amp=False, n_draws=8, img_size=64, expect_blocks=2, expect_dim=64,
                check_base_sha=False, expected_slices=None, pf_identity_step=1)
    base.update(kw)
    return T.Cfg(**base)


def quiet(fn, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = fn(*a, **k, log=lambda m: print(m))
    return rc, buf.getvalue()


# ============================================================================ tests
def test_lora() -> None:
    print("\n--- 1-4. LoRA: identity, merge, trainable set, frozen weights ---")
    s0 = tiny_sam()
    img = (np.random.default_rng(1).random((40, 56)) * 255).astype(np.uint8)
    t, ins, orig = C.prepare_image(img, 64)
    box = [(5, 6, 30, 25), (20, 10, 50, 35)]
    ref = C.forward_batch(s0, [t], [box], [ins], [orig], "cpu")[0]
    s = tiny_sam()
    L.inject_lora(s, 4, 4.0, expect_blocks=2, expect_dim=64)
    got = C.forward_batch(s, [t], [box], [ins], [orig], "cpu")[0]
    check(all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) for a, b in zip(ref, got)),
          "1. injected model with B = 0 gives exactly the original logits and IoU")
    check(raises(lambda: L.inject_lora(s, 4, 4.0)), "1. double injection is refused")
    check(raises(lambda: L.inject_lora(tiny_sam(), 4, 4.0, expect_blocks=12)),
          "1. a wrong block count is refused")

    with torch.no_grad():
        for m in L.lora_modules(s):
            for lin in (m.A_q, m.B_q, m.A_v, m.B_v):
                lin.weight.normal_(0, 0.2)
    lora_out = C.forward_batch(s, [t], [box], [ins], [orig], "cpu")[0]
    x = torch.randn(1, 4, 4, 64)
    m0 = L.lora_modules(s)[0]
    delta = m0(x) - m0.qkv(x)
    check(torch.count_nonzero(delta[..., 64:128]) == 0 and torch.count_nonzero(delta[..., :64]) > 0
          and torch.count_nonzero(delta[..., 128:]) > 0,
          "3. LoRA changes the q and v output slices only; k is untouched")
    L.merge_lora(s)
    merged = C.forward_batch(s, [t], [box], [ins], [orig], "cpu")[0]
    diff = max(float((a[0] - b[0]).abs().max()) for a, b in zip(lora_out, merged))
    check(diff <= 1e-5, f"2. merged model == LoRA model (max |dlogit| {diff:.1e})")
    check(L.plain_state_keys(s) == L.plain_state_keys(tiny_sam()) and not L.lora_modules(s),
          "2. merged state-dict keys equal a plain SAM's; no LoRA modules left")

    from segment_anything import sam_model_registry
    for r, want in ((4, 147456 + 4058340), (8, 294912 + 4058340)):
        vb = sam_model_registry["vit_b"](checkpoint=None)
        L.inject_lora(vb, r, 4.0, expect_blocks=12, expect_dim=768)
        info = L.freeze_for_arm2(vb)
        check(info["trainable"] == want and info["lora"] == 36864 * r and info["decoder"] == 4058340,
              f"3. real ViT-B r={r}: trainable {info['trainable']:,} = 36,864*{r} + 4,058,340")
        names = [n for n, p in vb.named_parameters() if p.requires_grad]
        check(all(".attn.qkv.A_" in n or ".attn.qkv.B_" in n or n.startswith("mask_decoder.") for n in names)
              and not any(n.startswith("prompt_encoder.") for n in names),
              f"3. r={r}: only LoRA A/B and the mask decoder are trainable")
        del vb

    s = tiny_sam()
    L.inject_lora(s, 4, 4.0, 2, 64)
    L.freeze_for_arm2(s)
    d0, a0 = L.frozen_state_digest(s), T.adapter_digest(s)
    opt = torch.optim.AdamW([p for p in s.parameters() if p.requires_grad], lr=1e-2)
    for _ in range(3):
        out = C.forward_batch(s, [t], [box], [ins], [orig], "cpu")[0]
        loss = sum(C.kidney_loss(lg, iou, torch.ones_like(lg, dtype=torch.bool))["total"] for lg, iou in out)
        loss.backward()
        opt.step()
        opt.zero_grad()
    check(L.frozen_state_digest(s) == d0 and T.adapter_digest(s) != a0,
          "4. after 3 optimiser steps the frozen weights are byte-identical and the adapter changed")
    st = L.adapter_state(s)
    s2 = tiny_sam()
    L.inject_lora(s2, 4, 4.0, 2, 64)
    L.load_adapter_state(s2, st)
    check(T.adapter_digest(s2) == T.adapter_digest(s), "4. adapter save/load round-trips exactly")
    bad = dict(st)
    bad.pop(next(iter(bad)))
    check(raises(lambda: L.load_adapter_state(s2, bad)), "4. a missing adapter key is refused")


def test_forward_equals_predictor(proc: Path) -> None:
    print("\n--- 5, 5b, 5c. the forward path is arm 1's, step for step ---")
    s = tiny_sam()
    rng = np.random.default_rng(2)
    sizes = [(48, 64), (64, 40), (64, 64), (30, 50)]
    imgs = [(rng.random(sz) * 255).astype(np.uint8) for sz in sizes]
    boxes = [[(3, 4, 30, 20)], [(2, 5, 30, 50), (10, 10, 20, 30)], [(0, 0, 63, 63)], [(5, 3, 40, 25)]]
    prep = [C.prepare_image(im, 64) for im in imgs]
    ours = C.forward_batch(s, [p[0] for p in prep], boxes, [p[1] for p in prep], [p[2] for p in prep], "cpu")
    pred = SamPredictor(s)
    worst, flips = 0.0, 0
    for im, bx, res in zip(imgs, boxes, ours):
        pred.set_image(np.repeat(im[..., None], 3, axis=2), image_format="RGB")
        for b, (lg, _) in zip(bx, res):
            m, _, _ = pred.predict(box=np.array(b, dtype=float), multimask_output=False, return_logits=True)
            ref = torch.as_tensor(m[0])
            worst = max(worst, float((lg.detach() - ref).abs().max()))
            sure = ref.abs() > 1e-4
            flips += int(((lg.detach() > 0) != (ref > 0))[sure].sum())
    check(worst <= 1e-5 and flips == 0,
          f"5. mixed-size batch: logits match SamPredictor (max |d| {worst:.1e}), masks equal")

    # negative control: pad raw images to a stack first, normalise after -> must differ
    raw = torch.stack([torch.nn.functional.pad(p[0], (0, 64 - p[0].shape[2], 0, 64 - p[0].shape[1]))
                       for p in prep]).float()
    wrong = (raw - s.pixel_mean) / s.pixel_std
    right = C.to_model_input(s, [p[0] for p in prep], "cpu")
    check(not torch.equal(wrong, right), "5. negative control: pad-then-normalise is detected as different")

    captured = []
    h = s.image_encoder.register_forward_hook(lambda mod, inp, out: captured.append(inp[0].detach().clone()))
    for im in (imgs[0], imgs[2]):
        pred.set_image(np.repeat(im[..., None], 3, axis=2), image_format="RGB")
        mine = C.to_model_input(s, [C.prepare_image(im, 64)[0]], "cpu")
        check(torch.equal(captured[-1], mine),
              f"5c. encoder input byte-equal to SamPredictor's ({im.shape[0]}x{im.shape[1]} slice)")
    h.remove()

    val = D.SliceDataset(proc, "val", expected=None)
    items = D.ValItems(val, None, 64, D.VAL_LEVELS)
    items.nsd = False
    rows, logits = T.predict_val(s, items, "cpu", batch=3, workers=0, keep_logits=True)
    k = 0
    ok_img = ok_box = True
    worst, flips, n = 0.0, 0, 0
    for idx in range(len(val)):
        item = val.load(idx)
        r = val.rows[idx]
        with np.load(proc / r["path"]) as dd:
            ok_img &= np.array_equal(item["img"], dd["img"])
        pred.set_image(np.repeat(item["img"][..., None], 3, axis=2), image_format="RGB")
        sides = [kd["side"] for kd in item["kidneys"]]
        expect_sides = [sd for sd, pfx in E.SIDES if r[f"{pfx}_box"] is not None]
        ok_box &= sides == expect_sides and all(
            kd["clean"] == tuple(r[f"{'rk' if kd['side'] == M.RIGHT else 'lk'}_box"]) for kd in item["kidneys"])
        for lev in D.VAL_LEVELS:
            for kd in item["kidneys"]:
                jb = M.jitter_box(kd["clean"], lev, r["h"], r["w"], M.jitter_seed(r["case"], r["z"], kd["side"], lev))
                m, _, _ = pred.predict(box=np.array(jb, dtype=float), multimask_output=False, return_logits=True)
                ref = torch.as_tensor(m[0])
                lg = logits[k]
                worst = max(worst, float((lg - ref).abs().max()))
                flips += int(((lg > 0) != (ref > 0))[ref.abs() > 1e-4].sum())
                check_row = rows[k]
                ok_box &= check_row["box"] == jb and check_row["level"] == lev
                k += 1
                n += 1
    check(k == len(logits) and worst <= 1e-5 and flips == 0,
          f"5b. whole val data path == arm 1's own steps for all {n} kidney x level pairs (max |d| {worst:.1e})")
    check(ok_img, "5b. the dataset's image is byte-equal to the stored .npz")
    check(ok_box, "5b. clean boxes = manifest, unboxed sides skipped, val boxes = jitter_box(jitter_seed)")


def test_losses_schedule_metric() -> None:
    print("\n--- 6, 7, 9. losses, LR schedule, selection metric ---")
    gt = torch.zeros(20, 20, dtype=torch.bool)
    gt[5:15, 5:15] = True
    perfect = torch.where(gt, 30.0, -30.0)
    d, b = C.seg_losses(perfect, gt)
    check(float(d) < 1e-6 and float(b) < 1e-6, "6. perfect prediction: Dice loss ~ 0, BCE ~ 0")
    d, _ = C.seg_losses(-perfect, gt)
    check(float(d) > 0.999, "6. inverted prediction: Dice loss ~ 1")
    lg = torch.zeros(2, 2)
    g = torch.tensor([[True, False], [False, False]])
    d, b = C.seg_losses(lg, g)
    p = 0.5
    want_d = 1 - (2 * p + 1e-6) / (4 * p * p + 1 + 1e-6)
    check(abs(float(d) - want_d) < 1e-6 and abs(float(b) - np.log(2)) < 1e-6,
          "6. hand-computed case: soft Dice (squared denominators) and BCE match")
    lg2 = torch.tensor([[1.0, -1.0], [1.0, -1.0]])
    check(abs(float(C.iou_target(lg2, g)) - 0.5) < 1e-9, "6. IoU target uses logit > 0, like the test threshold")

    base, warm, total, floor = 5e-4, 250, 36260, 1e-6
    f = [C.lr_factor(s, base, warm, total, floor) * base for s in range(total + 1)]
    check(abs(f[0] - base / warm) < 1e-15 and abs(f[warm - 1] - base) < 1e-15 and abs(f[warm] - base) < 1e-15,
          "7. linear warm-up reaches the base LR at step 250")
    check(abs(f[total] - floor) < 1e-12 and all(a >= b - 1e-18 for a, b in zip(f[warm:], f[warm + 1:])),
          "7. cosine decay is monotone and ends exactly at 1e-6")

    rows = ([{"modality": "ct", "patient": "a", "level": 0.1, "dice": v} for v in (0.8, 1.0)]
            + [{"modality": "ct", "patient": "b", "level": 0.1, "dice": 0.6}]
            + [{"modality": "mri", "patient": "c", "level": 0.1, "dice": 0.5}]
            + [{"modality": "ct", "patient": "a", "level": 0.0, "dice": 0.95},
               {"modality": "mri", "patient": "c", "level": 0.0, "dice": 0.7}])
    m = C.val_metric(rows)
    check(abs(m["ct@0.10"] - 0.75) < 1e-12 and abs(m["mri@0.10"] - 0.5) < 1e-12
          and abs(m["selection"] - 0.625) < 1e-12,
          "9. patient-mean then mean over patients per modality; selection = mean(CT, MRI) at 10%")


def test_data(proc: Path) -> None:
    print("\n--- 8, 16. data: splits, sampler, draw-indexed jitter, kidney counts ---")
    check(raises(lambda: D.SliceDataset(proc, "test", expected=None)), "8. a test split is refused")
    check(raises(lambda: D.SliceDataset(proc, "external", expected=None)), "8. the external split is refused")
    tr = D.SliceDataset(proc, "train", expected=None)
    check(all(r["split"] == "train" for r in tr.rows) and len(tr) == 16,
          "8. train dataset holds only train rows, boxed slices only (16 of 20)")
    check(raises(lambda: D.SliceDataset(proc, "train", expected={"train": {"ct": 1, "mri": 1}})),
          "8. a wrong expected slice count is refused")
    check(D.check_split_integrity(proc) == {"train": 4, "val": 2, "test": 1},
          "8. split integrity: one split per patient")
    leak = proc.parent / "leak"
    shutil.copytree(proc, leak)
    with open(leak / "manifest.csv") as fh:
        rs = list(csv.DictReader(fh))
    rs[-1]["patient"] = "amos_0001"
    with open(leak / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rs[0].keys()))
        w.writeheader()
        w.writerows(rs)
    check(raises(lambda: D.check_split_integrity(leak)), "8. a patient in two splits is detected")

    dr = D.make_epoch_draws(tr, 1, 643, 8295)
    mri = sum(tr.rows[i]["modality"] == "mri" for _, i in dr) / len(dr)
    check(abs(mri - 0.5) <= 0.03, f"8. weighted draws are 50/50 CT/MRI ({mri:.3f} MRI over 8,295)")
    check(D.make_epoch_draws(tr, 1, 643, 8295) == dr and D.make_epoch_draws(tr, 2, 643, 8295) != dr,
          "8. draws are deterministic in (seed, epoch) and change with the epoch")

    td = D.TrainDraws(tr, dr[:12], 1, 643, 64)
    direct = [td[k]["boxes"] for k in range(12)]
    loader_boxes = []
    for b in T.loader(td, 3, 2, drop_last=False):
        loader_boxes += [it["boxes"] for it in b]
    check(direct == loader_boxes, "10. training jitter is identical with 0 and 2 workers (seeded per draw)")
    cfg8 = T.Cfg(batch=8, accum=1)
    cfg42 = T.Cfg(batch=4, accum=2)
    check(T.steps_per_epoch(8295, cfg8) == T.steps_per_epoch(8295, cfg42) == 1036,
          "16. 8,295 draws -> 1,036 steps at batch 8 and at 4 x 2 alike")
    check(all(td[k]["boxes"].__len__() == tr.n_kidneys(dr[k][1]) for k in range(12)),
          "16. kidneys per item = manifest count (right-only slices give one)")


def test_training(proc: Path, tmp: Path, ck: Path) -> None:
    print("\n--- 10, 11. training: resume exactness, run stamp, idempotence ---")
    tr = D.SliceDataset(proc, "train", expected=None)
    va = D.SliceDataset(proc, "val", expected=None)
    ds = (tr, va)
    a_dir, b_dir = tmp / "runA", tmp / "runB"
    rc, _ = quiet(T.run_training, tiny_cfg(tmp, proc, ck, run_dir=str(a_dir)), tiny_plain_builder, ds)
    check(rc == 0 and json.loads((a_dir / "run.json").read_text())["status"] == "TRAIN_OK",
          "10. a straight 2-epoch run ends TRAIN_OK")
    rc, out1 = quiet(T.run_training, tiny_cfg(tmp, proc, ck, run_dir=str(b_dir), stop_after_epoch=1),
                     tiny_plain_builder, ds)
    check(rc == 0 and not (b_dir / "run.json").exists() and "stopping after epoch 1" in out1,
          "10. a run interrupted after epoch 1 has no run.json")
    rc, out2 = quiet(T.run_training, tiny_cfg(tmp, proc, ck, run_dir=str(b_dir)), tiny_plain_builder, ds)
    check(rc == 0 and "resuming after epoch 1" in out2, "10. it resumes after epoch 1")
    la = torch.load(a_dir / "last.pt", weights_only=False)["adapter"]
    lb = torch.load(b_dir / "last.pt", weights_only=False)["adapter"]
    check(set(la) == set(lb) and all(torch.equal(la[k], lb[k]) for k in la),
          "10. 2 epochs straight == 1 + resume + 1: identical adapter tensors")
    skip = {"epoch_s", "train_s", "val_s", "data_wait_s", "peak_mem_gib"}
    ma, mb = T.read_metrics(a_dir / "metrics.csv"), T.read_metrics(b_dir / "metrics.csv")
    check(len(ma) == 3 and [{k: v for k, v in r.items() if k not in skip} for r in ma]
          == [{k: v for k, v in r.items() if k not in skip} for r in mb],
          "10. ...and identical metrics rows (epoch 0 baseline + 2 epochs)")
    check(json.loads((a_dir / "run.json").read_text())["best_epoch"] in (0, 1, 2)
          and (a_dir / "best.pt").exists(), "10. best.pt exists (epoch-0 baseline counts)")

    rc, out = quiet(T.run_training, tiny_cfg(tmp, proc, ck, run_dir=str(a_dir)), tiny_plain_builder, ds)
    check(rc == 0 and "already complete" in out and "TRAIN_OK" in out and "epoch 1/" not in out,
          "11. resubmitting a finished run prints TRAIN_OK without training")
    check(raises(lambda: quiet(T.run_training, tiny_cfg(tmp, proc, ck, run_dir=str(a_dir), lr_dec=2e-4),
                               tiny_plain_builder, ds)), "11. a changed hyper-parameter is refused on resume")
    real = T.code_hashes
    try:
        T.code_hashes = lambda: {**real(), "sam_lora.py": "0" * 64}
        check(raises(lambda: quiet(T.run_training, tiny_cfg(tmp, proc, ck, run_dir=str(a_dir)),
                                   tiny_plain_builder, ds)), "11. changed training code is refused on resume")
        rc, out = quiet(T.run_training, tiny_cfg(tmp, proc, ck, run_dir=str(a_dir),
                                                 allow_code_change="comment-only fix"), tiny_plain_builder, ds)
        log = json.loads((a_dir / "config.json").read_text())["code_changes"]
        check(rc == 0 and log and log[-1]["reason"] == "comment-only fix",
              "11. --allow-code-change accepts it and logs the reason")
    finally:
        T.code_hashes = real
    s = T.stamp_of(tiny_cfg(tmp, proc, ck))
    check("workers" not in s and "run_dir" not in s and "rank" in s and "lr_lora" in s,
          "11. the stamp covers result-changing settings, not run-local ones")

    # 16. accumulation: batch 2 x 1 and batch 1 x 2 give the SAME gradient for the same step.
    # (Compared before the optimiser: Adam divides each update by the gradient's own scale,
    # which magnifies summation-order rounding, so post-Adam weights are not a fair test.)
    draws = D.make_epoch_draws(tr, 1, 643, 8)[:2]
    items = [D.TrainDraws(tr, draws, 1, 643, 64)[k] for k in range(2)]
    kids = sum(len(it["boxes"]) for it in items)
    grads = []
    for b, a, mbs in ((2, 1, [items]), (1, 2, [[items[0]], [items[1]]])):
        c = tiny_cfg(tmp, proc, ck, batch=b, accum=a)
        sam, _ = T.build_lora(c, tiny_plain_builder, "cpu")
        with torch.no_grad():                       # non-zero B, so the LoRA A matrices get gradients too
            for m in L.lora_modules(sam):
                m.B_q.weight.normal_(0, 0.05)
                m.B_v.weight.normal_(0, 0.05)
        T.train_one_step(sam, mbs, kids, c, "cpu")
        lora, dec = T.trainable_groups(sam)
        grads.append([p.grad.detach().clone() for p in lora + dec])
    scale = max(float(g.abs().max()) for g in grads[0])
    gap = max(float((g1 - g2).abs().max()) for g1, g2 in zip(*grads))
    check(scale > 0 and gap <= 1e-6 * max(1.0, scale),
          f"16. batch 2 x 1 and batch 1 x 2 give the same gradient (max |d| {gap:.1e}, max |g| {scale:.1e})")
    # ...and a full epoch runs end to end with accumulation
    c = tiny_cfg(tmp, proc, ck, batch=1, accum=2)
    sam, _ = T.build_lora(c, tiny_plain_builder, "cpu")
    opt, sched = T.make_optimizer(sam, c, T.steps_per_epoch(8, c) * c.epochs)
    r = T.run_epoch(sam, opt, sched, tr, 1, c, "cpu", 8)
    check(r["steps"] == 4 and np.isfinite(r["train_loss"]), "16. a full epoch with 1 x 2 accumulation runs (4 steps)")

    print("\n--- finalize: merge, merge check, idempotence, check-merged ---")
    cfg = tiny_cfg(tmp, proc, ck, run_dir=str(a_dir), merged_out=str(tmp / "sam_vit_b_tiny.pth"))
    rc, out = quiet(T.finalize, cfg, tiny_plain_builder, tiny_plain_builder, ds)
    rj = json.loads((a_dir / "run.json").read_text())
    check(rc == 0 and "FINALIZE_OK" in out and rj["merged"]["check"]["passed"]
          and rj["merged"]["check"]["pixel_agreement"] >= 0.999,
          f"finalize: merged checkpoint reproduces the LoRA model "
          f"(agreement {rj['merged']['check']['pixel_agreement']:.6f})")
    sd = torch.load(cfg.merged_out, weights_only=True)
    check(sorted(sd) == L.plain_state_keys(tiny_sam()), "finalize: the merged file is a plain SAM state dict")
    check(quiet(T.check_merged, cfg)[0] == 0, "check-merged: verified against run.json")
    rc, out = quiet(T.finalize, cfg, tiny_plain_builder, tiny_plain_builder, ds)
    check(rc == 0 and "already merged" in out, "finalize: rerun is a no-op")
    with open(cfg.merged_out, "ab") as fh:
        fh.write(b"tamper")
    check(quiet(T.check_merged, cfg)[0] == 1, "check-merged: a modified merged file is refused")
    check(raises(lambda: quiet(T.finalize, cfg, tiny_plain_builder, tiny_plain_builder, ds)),
          "finalize: refuses to overwrite a mismatching merged file without --force-new")
    cfg.force_new = True
    rc, _ = quiet(T.finalize, cfg, tiny_plain_builder, tiny_plain_builder, ds)
    check(rc == 0 and quiet(T.check_merged, cfg)[0] == 0, "finalize: --force-new rebuilds it")

    print("\n--- preflight (smoke on CPU; real thresholds apply on the A100) ---")
    pcfg = tiny_cfg(tmp, proc, ck, run_dir=str(tmp / "pre"), pf_lr_steps=4, pf_learn_steps=3,
                    pf_learn_slices=4, pf_probe_slices=4, pf_lr_max_drop=1.0)
    rc, out = quiet(T.run_preflight, pcfg, tiny_plain_builder, ds)
    rep = json.loads((tmp / "pre" / "preflight.json").read_text())
    check(set(rep["checks"]) == {"identity", "lr_sanity", "learn"} and "estimate" in rep
          and "recommended" in rep and rep["checks"]["identity"]["passed"],
          "preflight: writes all checks; identity (B = 0 vs SamPredictor) passes")
    check(("PREFLIGHT_OK" in out) == rep["passed"] and rc == (0 if rep["passed"] else 1),
          "preflight: the end marker and exit code follow the checks")
    lr = rep["checks"]["lr_sanity"]
    lrn = rep["checks"]["learn"]
    check(lr["passed"] == (lr["probe_loss_after"] < lr["probe_loss_before"]
                           and lr["val_subset_dice_after"] >= lr["val_subset_dice_before"] - 1.0)
          and lrn["passed"] == (lrn["loss_after"] <= 0.5 * lrn["loss_before"] and lrn["dice_after"] > lrn["dice_before"]),
          "preflight: LR check uses the fixed probe; learn check = loss halves and Dice rises")
    # the learn criterion genuinely passes once the model is given enough steps on a tiny set
    pcfg2 = tiny_cfg(tmp, proc, ck, run_dir=str(tmp / "pre2"), pf_lr_steps=4, pf_learn_steps=150,
                     pf_learn_slices=2, pf_probe_slices=4, pf_lr_max_drop=1.0, lr_dec=1e-2, lr_lora=1e-2)
    quiet(T.run_preflight, pcfg2, tiny_plain_builder, ds)
    lrn2 = json.loads((tmp / "pre2" / "preflight.json").read_text())["checks"]["learn"]
    check(lrn2["passed"] and lrn2["loss_after"] < 0.5 * lrn2["loss_before"],
          f"preflight: the learn check passes when the model can fit (loss {lrn2['loss_before']:.3f} -> "
          f"{lrn2['loss_after']:.3f}, Dice {lrn2['dice_before']:.3f} -> {lrn2['dice_after']:.3f})")


def test_gpu_and_selection(tmp: Path) -> None:
    print("\n--- 12, 14. GPU guard and rank selection ---")
    a100 = {"cuda": True, "name": "NVIDIA A100-SXM4-80GB", "total_gib": 79.3, "capability": [8, 0], "bf16": True}
    check(T.gpu_check("A100", a100)["name"].startswith("NVIDIA A100"), "14. an A100-80GB passes")
    check(raises(lambda: T.gpu_check("A100", {**a100, "name": "NVIDIA A40", "total_gib": 44.4})),
          "14. an A40 is refused when an A100 is expected")
    check(raises(lambda: T.gpu_check("A100", {**a100, "total_gib": 39.4})), "14. an A100-40GB is refused")
    check(raises(lambda: T.gpu_check("A100", {"cuda": False})), "14. no CUDA is refused")

    check(S.decide(0.900, 0.901)["chosen_rank"] == 4, "12. a difference < 0.002 goes to r = 4")
    d = S.decide(0.900, 0.904)
    check(d["chosen_rank"] == 8 and not d["consider_r16"], "12. r8 ahead by 0.004 -> r = 8")
    d = S.decide(0.900, 0.906)
    check(d["chosen_rank"] == 8 and d["consider_r16"], "12. r8 ahead by 0.006 -> r = 8, consider r = 16")
    check(S.decide(0.905, 0.900)["chosen_rank"] == 4, "12. r4 ahead -> r = 4")

    runs, outs = tmp / "sel" / "runs", tmp / "sel" / "outputs"
    settings = {"epochs": 35, "batch": 8, "lr_lora": 5e-4, "seed": 643}
    for r, score in ((4, 0.910), (8, 0.913)):
        rd = runs / f"arm2_r{r}"
        rd.mkdir(parents=True)
        (rd / "run.json").write_text(json.dumps({
            "status": "TRAIN_OK", "rank": r, "epochs_completed": 35, "best_epoch": 20 + r,
            "best_score": score, "settings": {**settings, "rank": r},
            "merged": {"sha256": f"{r}" * 64, "check": {"passed": True}}}))
        (rd / "config.json").write_text(json.dumps({
            "settings": {**settings, "rank": r}, "base_sha256": "b", "manifest_sha256": "m",
            "gpu_type": "NVIDIA A100-SXM4-80GB", "code": {"x": "1"}}))
    outs.mkdir(parents=True)
    for r in (4, 8):                       # the training jobs' SLURM logs sit in outputs/ too
        for ext in ("out", "err"):
            (outs / f"arm2_r{r}_train.36260{r}.{ext}").write_text("TRAIN_OK\n")
    with contextlib.redirect_stdout(io.StringIO()):
        rc = S.main(["--runs", str(runs), "--outputs", str(outs)])
    sel = json.loads((runs / "arm2_selection.json").read_text())
    check(rc == 0 and sel["chosen_rank"] == 8 and sel["r8"]["best_epoch"] == 28,
          "12. selection from val only: r8 (0.913) over r4 (0.910)")
    check(sel["existing_eval_folders"] == [],
          "12. training log files outputs/arm2_r*_train.*.out/.err are not evaluation folders")
    (outs / "arm2_r4").mkdir()
    with contextlib.redirect_stdout(io.StringIO()):
        rc = S.main(["--runs", str(runs), "--outputs", str(outs)])
    check(rc == 1, "12. refuses once an outputs/arm2_r* evaluation folder exists")
    (outs / "arm2_r4").rmdir()
    c8 = json.loads((runs / "arm2_r8" / "config.json").read_text())
    for key, val in (("settings", {**settings, "rank": 8, "batch": 4}), ("gpu_type", "NVIDIA A40")):
        bad = dict(c8)
        bad[key] = val
        (runs / "arm2_r8" / "config.json").write_text(json.dumps(bad))
        with contextlib.redirect_stdout(io.StringIO()):
            rc = S.main(["--runs", str(runs), "--outputs", str(outs)])
        check(rc == 1, f"12/14. refuses when the runs differ in {key} ({'batch' if key == 'settings' else 'GPU type'})")
    (runs / "arm2_r8" / "config.json").write_text(json.dumps(c8))


def gen_values() -> dict:
    """The ARM2_* constants, read from gen_sbatch.py's source (importing it would run it)."""
    import ast
    vals = {}
    for node in ast.parse((HERE / "gen_sbatch.py").read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name.startswith("ARM2_"):
                vals[name] = ast.literal_eval(node.value)
    return vals


def run_scripts(tmp: Path) -> None:
    print("\n--- 13, 15. generated arm-2 SLURM scripts ---")
    names = ["arm2_preflight.sbatch", "arm2_r4_train.sbatch", "arm2_r8_train.sbatch",
             "arm2_select.sbatch", "arm2_r4_eval.sbatch", "arm2_r8_eval.sbatch", "arm2_compare.sbatch"]
    regen = tmp / "regen"
    regen.mkdir()
    subprocess.run([sys.executable, str(HERE / "gen_sbatch.py"), str(regen)], check=True, capture_output=True)
    for n in names:
        check((HERE / n).read_text() == (regen / n).read_text(), f"13. {n} == generator output")
        check(subprocess.run(["bash", "-n", str(HERE / n)]).returncode == 0, f"13. {n} passes bash -n")
    texts = {n: (HERE / n).read_text() for n in names}

    def header(n, key):
        return next(line.split("=", 1)[1] for line in texts[n].splitlines() if line.startswith(f"#SBATCH --{key}="))
    for n in names:
        want = "dgx" if ("train" in n or "preflight" in n) else "a40"
        check(header(n, "partition") == want and header(n, "qos") == want,
              f"13. {n}: partition and QOS {want}")
    G = gen_values()
    for r in (4, 8):
        check(header(f"arm2_r{r}_train.sbatch", "time") == G["ARM2_TRAIN_TIME"],
              f"13. arm2_r{r}_train time limit = ARM2_TRAIN_TIME ({G['ARM2_TRAIN_TIME']})")

    # 15. twins
    def mapped(n, pairs):
        t = texts[n]
        for a, b in pairs:
            t = t.replace(a, b)
        return t
    tr_map = [("r4", "r8"), ("--rank 4", "--rank 8"), ("LoRA rank 4", "LoRA rank 8")]
    ev_map = [("r4", "r8"), ("--rank 4", "--rank 8"), ("rank 4,", "rank 8,"), ("r=4", "r=8")]
    check(mapped("arm2_r4_train.sbatch", tr_map) == texts["arm2_r8_train.sbatch"],
          "15. the r4 and r8 training scripts are twins (identical after mapping the rank)")
    check(mapped("arm2_r4_eval.sbatch", ev_map) == texts["arm2_r8_eval.sbatch"],
          "15. the r4 and r8 evaluation scripts are twins")

    seg = tmp / "Segmentation"
    for d in ("checkpoints", "runs", "outputs", "scripts", "data/processed"):
        (seg / d).mkdir(parents=True, exist_ok=True)
    (seg / "checkpoints" / "sam_vit_b_01ec64.pth").write_bytes(b"x")
    stub = tmp / "bin"
    stub.mkdir()
    calls = tmp / "calls.txt"
    fake_py = stub / "python"
    fake_py.write_text(
        "#!/bin/bash\n"
        'if [ "$1" = "-c" ]; then exit 0; fi\n'
        f'echo "$@" >> "{calls}"\n'
        'case "$*" in *--check-merged*) exit "${FAKE_CHECK_RC:-0}";; esac\n'
        'case "$*" in *--finalize*) exit 0;; esac\n'
        'case "$*" in *train_sam_lora.py*) exit "${FAKE_TRAIN_RC:-0}";; esac\n'
        "exit 0\n")
    (stub / "nvidia-smi").write_text(
        "#!/bin/bash\n"
        'case "$*" in *query-gpu=name\\ *|*"query-gpu=name --format"*) echo "$FAKE_GPU";;'
        ' *) echo "0, $FAKE_GPU, 81920 MiB";; esac\n')
    (stub / "squeue").write_text('#!/bin/bash\ncat "$FAKE_SQUEUE"\n')
    for f in stub.iterdir():
        f.chmod(0o755)

    def run(n, gpu="NVIDIA A100-SXM4-80GB", queue="", **env_extra):
        sc = texts[n].replace("/home/rethinkingai-self/vihaan/miniconda3/envs/seg311/bin/python", str(fake_py))
        sc = sc.replace("/home/rethinkingai-self/vihaan/Segmentation", str(seg))
        sp = tmp / n
        sp.write_text(sc)
        q = tmp / "q.txt"
        q.write_text(queue)
        if calls.exists():
            calls.unlink()
        env = dict(os.environ, PATH=f"{stub}:{os.environ['PATH']}", FAKE_SQUEUE=str(q), FAKE_GPU=gpu,
                   SLURM_JOB_ID="900", SLURM_CPUS_ON_NODE="16", **env_extra)
        r = subprocess.run(["bash", str(sp)], env=env, capture_output=True, text=True)
        return r.returncode, (calls.read_text().splitlines() if calls.exists() else []), r.stdout

    A40 = "NVIDIA A40"
    for r in (4, 8):
        n = f"arm2_r{r}_train.sbatch"
        rc, py, _ = run(n, queue=f"900 arm2_r{r}_tr RUNNING\n901 arm2_r{12 - r}_tr RUNNING\n")
        tr = [c for c in py if "train_sam_lora.py" in c]
        ok = (rc == 0 and len(tr) == 2 and "--finalize" not in tr[0] and tr[1].endswith("--finalize")
              and all(f"--rank {r} " in c and f"{seg}/runs/arm2_r{r} " in c
                      and f"{seg}/checkpoints/sam_vit_b_arm2_r{r}.pth" in c and "--expect-gpu A100" in c
                      and f"--batch {G['ARM2_BATCH']} --accum {G['ARM2_ACCUM']}" in c and "--workers 10" in c
                      and "--epochs 35" in c and "--alpha 4" in c and "sam_vit_b_01ec64.pth" in c for c in tr))
        check(ok, f"13. {n}: trains rank {r} on the A100, then finalizes (runs alongside the other rank)")
        rc, py, _ = run(n, FAKE_TRAIN_RC="1")
        check(rc == 1 and not any("--finalize" in c for c in py), f"13. {n}: a failed training skips finalize")
        rc, py, _ = run(n, gpu=A40)
        check(rc == 4 and not py, f"13. {n}: refuses to start on an A40")
        for clash in (f"902 arm2_r{r}_tr PENDING", f"903 arm2_r{r}_ev RUNNING", "904 arm2_pre RUNNING"):
            rc, py, _ = run(n, queue=clash + "\n")
            check(rc == 3 and not py, f"13. {n}: refuses next to '{clash}'")

    rc, py, _ = run("arm2_preflight.sbatch")
    check(rc == 0 and len(py) == 1 and py[0].endswith("--preflight") and "--rank 4 " in py[0]
          and "--merged-out" not in py[0] and "--expect-gpu A100" in py[0],
          "13. preflight: rank 4, A100, --preflight")
    rc, py, _ = run("arm2_preflight.sbatch", queue="905 arm2_r8_tr RUNNING\n")
    check(rc == 3, "13. preflight refuses while a training job runs")

    sel = seg / "runs" / "arm2_selection.json"
    for r in (4, 8):
        n = f"arm2_r{r}_eval.sbatch"
        if sel.exists():
            sel.unlink()
        rc, py, _ = run(n, gpu=A40)
        check(rc == 5 and not py, f"13. {n}: refuses without arm2_selection.json")
        sel.write_text("{}")
        rc, py, _ = run(n, gpu=A40, FAKE_CHECK_RC="1")
        check(rc == 6 and len(py) == 1 and "--check-merged" in py[0], f"13. {n}: refuses an unverified merged checkpoint")
        (seg / "outputs" / "arm1_vit_b").mkdir(exist_ok=True)
        (seg / "outputs" / "arm1_vit_b" / "run.json").write_text("{}")
        rc, py, _ = run(n, gpu=A40)
        ev = [c for c in py if "eval_sam.py" in c]
        ag = [c for c in py if "aggregate_arm.py" in c]
        cm = [c for c in py if "compare_arms.py" in c]
        check(rc == 0 and len(ev) == 1 and f"{seg}/checkpoints/sam_vit_b_arm2_r{r}.pth" in ev[0]
              and "--model-type vit_b" in ev[0] and f"{seg}/outputs/arm2_r{r} " in ev[0] + " "
              and ev[0].rstrip().endswith("--resume") and len(ag) == 1 and f"r={r}" in ag[0]
              and len(cm) == 1 and f"{seg}/outputs/arm1_vit_b" in cm[0],
              f"13. {n}: verified merged r{r} -> eval_sam on A40 -> aggregate -> compare vs arm 1b")
        rc, py, _ = run(n)
        check(rc == 4 and not py, f"13. {n}: refuses to start on an A100 (evaluation must match arm 1b's A40)")
        rc, py, _ = run(n, gpu=A40, queue=f"906 arm2_r{r}_tr RUNNING\n")
        check(rc == 3, f"13. {n}: refuses while the r{r} training runs")

    rc, py, _ = run("arm2_select.sbatch", gpu=A40)
    check(rc == 0 and len(py) == 1 and "select_rank.py" in py[0] and f"--runs {seg}/runs" in py[0],
          "13. select: select_rank.py on runs/ and outputs/")
    rc, py, _ = run("arm2_select.sbatch", gpu=A40, queue="907 arm2_r4_tr RUNNING\n")
    check(rc == 3, "13. select refuses while a training job runs")
    rc, py, _ = run("arm2_compare.sbatch", gpu=A40)
    check(rc == 5 and not py, "13. compare refuses until both arm-2 evaluations exist")
    for r in (4, 8):
        (seg / "outputs" / f"arm2_r{r}").mkdir(exist_ok=True)
        (seg / "outputs" / f"arm2_r{r}" / "run.json").write_text("{}")
    rc, py, _ = run("arm2_compare.sbatch", gpu=A40)
    check(rc == 0 and len(py) == 1 and f"--a {seg}/outputs/arm2_r4" in py[0] and f"--b {seg}/outputs/arm2_r8" in py[0],
          "13. compare: r4 (A) -> r8 (B)")


def test_python_compat() -> None:
    import ast
    bad = []
    for f in ("sam_lora.py", "sam_train_core.py", "train_data.py", "train_sam_lora.py",
              "select_rank.py", "gen_sbatch.py", "test_train_sam.py"):
        try:
            ast.parse((HERE / f).read_text(), feature_version=(3, 11))
        except SyntaxError as e:
            bad.append(f"{f}:{e.lineno}")
    check(not bad, "all arm-2 files parse as Python 3.11" + (f" -- {bad}" if bad else ""))


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="arm2_test_"))
    proc = tmp / "processed"
    build_processed(proc)
    ck = tmp / "tiny.pth"
    torch.save(tiny_sam().state_dict(), ck)
    test_python_compat()
    test_lora()
    test_forward_equals_predictor(proc)
    test_losses_schedule_metric()
    test_data(proc)
    test_training(proc, tmp, ck)
    test_gpu_and_selection(tmp)
    run_scripts(tmp)
    print(f"\nscratch: {tmp}\n\nALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
