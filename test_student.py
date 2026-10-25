#!/usr/bin/env python
"""CPU tests for the arm-3 student code (ARM3_CODE_PLAN.md §7).

    python test_student.py        # exits 0 and prints ALL TESTS PASSED

Synthetic processed/ folder: CT 128x128, MRI of mixed sizes (up- and down-scaled into the
frame), kidney-free, sliver and right-only slices, plus a test case that must never be read.
A narrow 7-level network in a 128 frame keeps the real depth and deep-supervision layout.
"""
from __future__ import annotations

import contextlib
import csv
import io
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sam_metrics as M  # noqa: E402
import student_aug as A  # noqa: E402
import student_core as C  # noqa: E402
import student_data as D  # noqa: E402
import student_unet as U  # noqa: E402
import train_student as T  # noqa: E402

torch.set_num_threads(1)
NARROW = (2, 2, 4, 4, 4, 4, 4)
FRAME = 128


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)
    print(f"  ok  {msg}")


def raises(fn, exc=(SystemExit, AssertionError, ValueError, KeyError, RuntimeError, IndexError)) -> bool:
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            fn()
    except exc:
        return True
    return False


def quiet(fn, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = fn(*a, **k)
    return r, buf.getvalue()


# ---------------------------------------------------------------- synthetic data
CASES = [  # split, modality, patient, (h, w)
    ("train", "ct", "amos_0001", (128, 128)), ("train", "ct", "amos_0002", (128, 128)),
    ("train", "ct", "amos_0003", (128, 128)),
    ("train", "mri", "amos_0501", (96, 128)), ("train", "mri", "amos_0502", (128, 80)),
    ("train", "mri", "amos_0503", (160, 160)),
    ("val", "ct", "amos_0011", (128, 128)), ("val", "ct", "amos_0012", (128, 128)),
    ("val", "mri", "amos_0511", (60, 100)), ("val", "mri", "amos_0512", (160, 160)),
    ("test", "ct", "amos_0021", (128, 128)),
]
NZ = 8      # z 0 and 7 kidney-free; 1..5 both kidneys; 6 right sliver + left kidney


def disk(h, w, cy, cx, r):
    yy, xx = np.ogrid[:h, :w]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


def tight(m):
    ys, xs = np.nonzero(m)
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def build_processed(root: Path) -> dict:
    rng = np.random.default_rng(0)
    rows = []
    for split, mod, pat, (h, w) in CASES:
        cdir = root / "slices" / "amos" / pat
        cdir.mkdir(parents=True)
        r = max(4, round(0.09 * min(h, w)))
        for z in range(NZ):
            msk = np.zeros((h, w), np.uint8)
            if 1 <= z <= 6:
                msk[disk(h, w, 0.55 * h, 0.7 * w, r)] = 2                  # left kidney (image right)
                msk[disk(h, w, 0.55 * h, 0.3 * w, r if z <= 5 else 2)] = 1  # right (image left); z6 sliver
            img = (30 + rng.integers(0, 10, (h, w))).astype(np.uint8)
            img[: int(0.4 * h), : int(0.45 * w)] = 120                      # "liver" on the image left
            img[msk > 0] = 190
            rel = f"slices/amos/{pat}/{z:04d}.npz"
            np.savez_compressed(root / rel, img=img, msk=msk)
            rk, lk = msk == 1, msk == 2
            rb = tight(rk) if rk.sum() >= 50 else None
            lb = tight(lk) if lk.sum() >= 50 else None
            rows.append({"dataset": "amos", "patient": pat, "case": pat, "sequence": "",
                         "modality": mod, "split": split, "z": z, "n_slices": NZ,
                         "has_kidney": int(rk.any() or lk.any()), "rk_area": int(rk.sum()),
                         "lk_area": int(lk.sum()), "rk_box": json.dumps(rb), "lk_box": json.dumps(lb),
                         "sam_eligible": int(rb is not None or lb is not None), "h": h, "w": w,
                         "sp_row": 1.0, "sp_col": 1.0, "sp_slice": 3.0, "path": rel})
    with open(root / "manifest.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    exp = {}
    for split in ("train", "val"):
        exp[split] = {}
        for mod in ("ct", "mri"):
            rs = [r for r in rows if r["split"] == split and r["modality"] == mod]
            exp[split][mod] = (len(rs), sum(r["has_kidney"] for r in rs), sum(r["sam_eligible"] for r in rs))
    return exp


def tiny_cfg(proc: Path, run: Path, **kw) -> T.Cfg:
    base = dict(processed=str(proc), run_dir=str(run), epochs=2, steps_per_epoch=3, batch=2,
                size=FRAME, channels=NARROW, device="cpu", amp=False, workers=0, val_workers=0,
                val_batch=4, threads=2, check_params=False)
    base.update(kw)
    return T.Cfg(**base)


# ============================================================================ tests
def test_architecture() -> None:
    print("\n--- 1. architecture ---")
    for cin in (1, 3):
        m = U.StudentUNet(cin)
        check(U.count_params(m) == U.EXPECTED_PARAMS[cin] and U.count_params_no_ds(m) == U.EXPECTED_PARAMS_NO_DS[cin],
              f"in_ch {cin}: {U.count_params(m):,} parameters ({U.count_params_no_ds(m):,} without DS heads)")
    m = U.StudentUNet(1).train()
    with torch.no_grad():
        out = m(torch.zeros(1, 1, 512, 512))
    check([tuple(o.shape[-2:]) for o in out] == [(512, 512), (256, 256), (128, 128), (64, 64)],
          "real network at 512: DS outputs at 512 / 256 / 128 / 64")
    n = U.StudentUNet(1, channels=NARROW).train()
    x = torch.randn(2, 1, FRAME, FRAME)
    out = n(x)
    check(len(out) == 4 and [o.shape[-1] for o in out] == [128, 64, 32, 16], "narrow 7-level: 4 DS outputs at 128..16")
    n.eval()
    check(torch.is_tensor(n(x)) and n(x).shape == (2, 3, FRAME, FRAME), "eval mode returns the full-resolution logits only")
    nd = U.StudentUNet(1, channels=NARROW, deep_supervision=False)
    nd.train()
    a = nd(x)
    nd.eval()
    check(torch.allclose(a, nd(x), atol=1e-6), "InstanceNorm keeps no running stats: train == eval output")
    check(raises(lambda: n(torch.zeros(1, 1, 96, 96))), "a frame not divisible by 64 is refused")
    check(raises(lambda: n(torch.zeros(1, 3, FRAME, FRAME))), "wrong input channel count is refused")
    w = U.StudentUNet(1).enc[4].net[3].weight.detach()          # 256 x 256 x 3 x 3
    want = math.sqrt(2.0 / (1 + 0.01 ** 2) / (256 * 9))
    check(abs(float(w.std()) / want - 1) < 0.05, f"He init: std {float(w.std()):.5f} vs {want:.5f}")


def test_letterbox() -> None:
    print("\n--- 2-3. letterbox, normalisation, back-mapping ---")
    norm = C.Norm(50.0, 20.0)
    g = C.letterbox_geometry(512, 512)
    check((g["h2"], g["w2"]) == (512, 512), "512x512 -> no resize")
    for (h, w), (h2, w2) in {(300, 200): (512, 341), (700, 512): (512, 374), (48, 64): (384, 512),
                             (128, 80): (512, 320)}.items():
        g = C.letterbox_geometry(h, w)
        check((g["h2"], g["w2"]) == (h2, w2), f"{h}x{w} -> {g['h2']}x{g['w2']}")
    img = np.random.default_rng(1).integers(0, 256, (FRAME, FRAME)).astype(np.uint8)
    x, valid, g = C.to_net_image(img, norm, FRAME)
    check(torch.equal(x[0], (torch.from_numpy(img).float() - 50.0) / 20.0) and bool(valid.all()),
          "same-size slice: bit-exact normalisation, no interpolation, all valid")
    img2 = np.zeros((60, 100), np.uint8)
    x, valid, g = C.to_net_image(img2, norm, FRAME)
    check(g["h2"] == 77 and g["w2"] == 128 and bool(valid[:77].all()) and not bool(valid[77:].any()),
          "60x100 -> 77x128 top-left, valid exactly the content")
    check(bool((x[0, 77:] == norm.pad_value).all()) and abs(norm.pad_value + 2.5) < 1e-12,
          "padding = normalised black (-mean/std)")
    # perfect one-hot logits -> exact native labels for same-size slices
    msk = np.zeros((FRAME, FRAME), np.uint8)
    msk[disk(FRAME, FRAME, 60, 40, 9)] = 1
    msk[disk(FRAME, FRAME, 60, 90, 9)] = 2
    g = C.letterbox_geometry(FRAME, FRAME, FRAME)
    lab = C.to_net_label(msk, g)
    lg = torch.nn.functional.one_hot(lab, 3).permute(2, 0, 1).float() * 10
    check(np.array_equal(C.to_native_logits(lg, g).argmax(0).numpy(), msk), "same size: perfect logits -> exact labels")
    for hw in ((60, 100), (160, 160), (96, 128)):
        h, w = hw
        m2 = np.zeros(hw, np.uint8)
        m2[disk(h, w, h * 0.55, w * 0.3, 9)] = 1
        g = C.letterbox_geometry(h, w, FRAME)
        lab = C.to_net_label(m2, g)
        back = C.to_native_logits(torch.nn.functional.one_hot(lab, 3).permute(2, 0, 1).float() * 10, g)
        d = M.dice_coefficient(m2 == 1, back.argmax(0).numpy() == 1)
        check(back.shape[-2:] == hw and d > 0.85, f"{h}x{w}: back to native shape, ceiling Dice {d:.3f}")
    g = C.letterbox_geometry(60, 100, FRAME)
    lg = torch.zeros(3, FRAME, FRAME)
    lg[1, g["h2"]:, :] = 100.0                             # class 1 only in the padding
    check(int((C.to_native_logits(lg, g).argmax(0) == 1).sum()) == 0, "padding never reaches the native output")
    torch.manual_seed(0)
    model = U.StudentUNet(1, channels=NARROW)
    imgs = [np.random.default_rng(k).integers(0, 255, s).astype(np.uint8) for k, s in
            enumerate(((60, 100), (128, 128), (160, 160)))]
    xs, gs = zip(*[(C.net_input(i, norm, "plain", None, FRAME)[0], C.net_input(i, norm, "plain", None, FRAME)[2])
                   for i in imgs])
    batch = C.predict_native_batch(model, list(xs), list(gs), "cpu")
    single = [C.predict_native(model, i, norm, "plain", None, "cpu", FRAME) for i in imgs]
    check(all(np.array_equal(b[0], s[0]) for b, s in zip(batch, single)), "predict_native_batch == per-slice predict_native")
    buf = np.random.default_rng(3).integers(0, 256, 10000).astype(np.uint8)
    n1 = C.norm_from_buffer(buf)
    check(abs(n1.mean - buf.mean()) < 1e-9 and abs(n1.std - buf.std()) < 1e-9 and n1 == C.norm_from_buffer(buf),
          "global norm = exact mean/std, deterministic")
    check(C.norm_from_buffer(buf, chunk=7) == n1, "chunked counting == one-pass counting")
    check(raises(lambda: C.Norm(1.0, 0.0)), "zero std refused")
    check(raises(lambda: C.net_input(img, norm, "plain", {1: None})) and raises(lambda: C.net_input(img, norm, "box")),
          "plain refuses boxes; box requires them")


def test_augmentation() -> None:
    print("\n--- 4. augmentation ---")
    S = 64
    g = {"h": S, "w": S, "h2": S, "w2": S, "m": S, "size": S}
    lab = torch.zeros(S, S, dtype=torch.int64)
    lab[torch.from_numpy(disk(S, S, 32, 18, 7))] = 1
    lab[torch.from_numpy(disk(S, S, 32, 46, 7))] = 2
    img = (lab > 0).float()[None] * 100.0 - 1.0
    valid = torch.ones(S, S, dtype=torch.bool)
    box = torch.zeros(2, S, S)
    pad = -1.0
    a1 = A.augment(img, lab, valid, box, g, pad, np.random.default_rng(5))
    a2 = A.augment(img, lab, valid, box, g, pad, np.random.default_rng(5))
    check(all(torch.equal(p, q) for p, q in zip(a1[:4], a2[:4])), "same rng -> identical augmentation")
    counts = {k: 0 for k in A.AUG}
    order_ok = align_ok = labels_ok = pad_ok = True
    n_both = 0
    for k in range(3000):
        im, lb, va, bx, ap = A.augment(img, lab, valid, box, g, pad, np.random.default_rng(1000 + k))
        for key in counts:
            counts[key] += bool(ap.get(key, False))
        labels_ok &= set(torch.unique(lb).tolist()) <= {0, 1, 2}
        pad_ok &= bool((im[0][~va] == pad).all()) and bool((lb[~va] == 0).all())
        if not (ap["noise"] or ap["blur"] or ap["lowres"] or ap["brightness"] or ap["contrast"]
                or ap["gamma"] or ap["gamma_inverted"]):
            agree = ((im[0] > 49) == (lb > 0))[va].float().mean()
            align_ok &= float(agree) >= 0.99
        if k < 500 and (lb == 1).any() and (lb == 2).any():
            n_both += 1
            xs = torch.arange(S).float()
            c1 = float((lb == 1).float().sum(0) @ xs / (lb == 1).sum())
            c2 = float((lb == 2).float().sum(0) @ xs / (lb == 2).sum())
            order_ok &= c1 < c2
    check(labels_ok and pad_ok, "labels stay in {0,1,2}; padding = pad_value and label 0 outside valid")
    check(align_ok, "image and label stay aligned (>= 99% inside valid) under the spatial transform")
    check(order_ok and n_both > 400, f"no mirroring: right kidney stays left of the left kidney ({n_both} draws)")
    eff = {"rotation": 0.2, "scaling": 0.2, "noise": 0.1, "blur": 0.1, "brightness": 0.15,
           "contrast": 0.15, "lowres": 0.125, "gamma_inverted": 0.1, "gamma": 0.3}
    bad = {k: counts[k] / 3000 for k, p in eff.items()
           if abs(counts[k] / 3000 - p) > 3 * math.sqrt(p * (1 - p) / 3000) + 1e-9}
    check(not bad, f"effective probabilities match the plan within 3 sigma {bad or ''}".strip())
    rng = np.random.default_rng(7)
    vals = [A.bg_sample(rng, 0.75, 1.25) for _ in range(4000)]
    below = np.mean([v < 1 for v in vals])
    check(0.47 < below < 0.53 and min(vals) >= 0.75 and max(vals) <= 1.25, f"BGContrast: {below:.3f} of draws below 1")
    v = torch.randn(5000) * 2 + 3
    for inv in (False, True):
        out = A._gamma(v, 1.4, inv)
        check(abs(float(out.mean() - v.mean())) < 1e-4 and abs(float(out.std() - v.std())) < 1e-4,
              f"gamma (inverted={inv}) retains mean and std")
    # out-of-field: zoom out a full-field image -> corners become invalid, label 0, image = pad
    va_ = None
    for k in range(2000):
        r = np.random.default_rng(50000 + k)
        im, lb, va, bx, ap = A.augment(img, lab, valid, box, g, pad, r)
        if ap["scaling"] and not bool(va.all()):
            va_ = (im, lb, va)
            break
    check(va_ is not None and bool((va_[1][~va_[2]] == 0).all()) and bool((va_[0][0][~va_[2]] == pad).all()),
          "out-of-field pixels: valid False, label 0, image = pad")
    bx_in = torch.stack([(lab == 1).float(), (lab == 2).float()])
    ok = True
    for k in range(200):
        im, lb, va, bx, ap = A.augment(img, lab, valid, bx_in, g, pad, np.random.default_rng(9000 + k))
        ok &= torch.equal(bx[0] > 0.5, lb == 1) and torch.equal(bx[1] > 0.5, lb == 2)
    check(ok, "box channels follow the spatial transform exactly like the labels")
    check("flip" not in (HERE / "student_aug.py").read_text().lower(), "student_aug.py contains no flip at all")
    # nnU-Net source semantics (batchgeneratorsv2, checked 2026-10-24)
    stds, shrink_ok, grow_ok = [], True, True
    area0 = int((lab > 0).sum())
    only = ("noise", "blur", "brightness", "contrast", "lowres", "gamma", "gamma_inverted", "rotation")
    for k in range(4000):
        im, lb, va, bx, ap = A.augment(img, lab, valid, box, g, pad, np.random.default_rng(70000 + k))
        if ap.get("noise") and not any(ap.get(o) for o in only if o != "noise") and not ap["scaling"]:
            stds.append((float((im[0] - img[0]).std()), ap["noise_std"]))
        if ap["scaling"] and not any(ap.get(o) for o in only):
            a = int((lb > 0).sum())
            if ap["scale_value"] > 1.05:
                shrink_ok &= a < area0
            if ap["scale_value"] < 0.95:
                grow_ok &= a > area0
    check(stds and max(sd for sd, _ in stds) <= 0.1 + 0.01 and all(abs(sd - want) < 0.01 for sd, want in stds),
          f"noise: std ~ U(0, 0.1), used directly as the std ({len(stds)} draws, max {max(sd for sd, _ in stds):.3f})")
    check(shrink_ok and grow_ok, "scaling: scale > 1 shrinks the kidneys, scale < 1 enlarges them (as nnU-Net)")
    imp = torch.zeros(1, 41, 41)
    imp[0, 20, 20] = 1.0
    bl = A._gauss_blur(imp, 0.5, 1.0)[0]
    d2 = (torch.arange(41) - 20.0) ** 2

    def spread(v):
        return float((v * d2).sum() / v.sum())
    check(spread(bl[20]) > 2 * spread(bl[:, 20]) and len(A._kernel(1.0)) == 7 and len(A._kernel(0.5)) == 3,
          "blur: separate sigma per axis; kernel size from truncate 6 (sigma 1.0 -> 7 taps)")


def test_boxes() -> None:
    print("\n--- 5. box channels ---")
    g = C.letterbox_geometry(256, 256, 512)
    check(C.box_to_frame((10, 20, 30, 40), g) == (20, 40, 61, 81), "x2 upscale: inclusive box covers 2x the pixels")
    g = C.letterbox_geometry(1024, 1024, 512)
    check(C.box_to_frame((10, 20, 11, 21), g) == (5, 10, 5, 10), "x0.5 downscale: a 2-px box -> 1 px, never inverted")
    g = C.letterbox_geometry(60, 100, FRAME)
    b = C.box_to_frame((90, 50, 99, 59), g)
    check(b[2] <= g["w2"] - 1 and b[3] <= g["h2"] - 1, f"edge box clipped to the content {b}")
    ch = C.box_channels({1: (10, 10, 20, 20), 2: None}, C.letterbox_geometry(FRAME, FRAME, FRAME))
    check(float(ch[0].sum()) == 121 and float(ch[1].sum()) == 0, "right box filled inclusively; unboxed left side = zeros")
    check(raises(lambda: C.box_to_frame((0, 0, 100, 5), C.letterbox_geometry(60, 100, FRAME))), "out-of-slice box refused")


def test_losses_schedule() -> None:
    print("\n--- 6-7. losses, schedule ---")
    t = torch.zeros(2, 16, 16, dtype=torch.int64)
    t[:, 4:8, 4:8] = 1
    t[0, 10:13, 10:13] = 2
    v = torch.ones(2, 16, 16, dtype=torch.bool)
    perfect = torch.nn.functional.one_hot(t, 3).permute(0, 3, 1, 2).float() * 50
    lp = C.ce_dice_loss(perfect, t, v)
    check(float(lp["total"]) < 1e-3, f"perfect logits -> loss {float(lp['total']):.2e}")
    wrong = torch.nn.functional.one_hot((t + 1) % 3, 3).permute(0, 3, 1, 2).float() * 50
    check(float(C.ce_dice_loss(wrong, t, v)["total"]) > 10, "inverted logits -> large loss")
    v2 = v.clone()
    v2[:, 14:, :] = False
    garbage = perfect.clone()
    garbage[:, :, 14:, :] = torch.randn(2, 3, 2, 16) * 100
    check(torch.allclose(C.ce_dice_loss(garbage, t, v2)["total"], C.ce_dice_loss(perfect, t, v2)["total"]),
          "garbage in the invalid region does not change the loss")
    t2 = t.clone()
    t2[t2 == 2] = 0
    check(bool(torch.isfinite(C.ce_dice_loss(torch.randn(2, 3, 16, 16), t2, v)["total"])), "batch without a left kidney: finite")
    check(np.allclose(C.ds_weights(4), [8 / 15, 4 / 15, 2 / 15, 1 / 15]), "DS weights (8, 4, 2, 1)/15")
    outs = [torch.nn.functional.one_hot(t[:, ::2 ** k, ::2 ** k], 3).permute(0, 3, 1, 2).float() * 50 for k in range(4)]
    check(float(C.ds_loss(outs, t, v)["total"]) < 1e-3, "DS targets = nearest downsampling (perfect at every scale)")
    check(raises(lambda: C.ds_loss(outs[::-1], t, v)), "DS outputs in the wrong order are refused")
    lrs = [C.poly_lr(s, 100000, 0.01) for s in (0, 50000, 99999)]
    want = [0.01, 0.01 * 0.5 ** 0.9, 0.01 * (1e-5) ** 0.9]
    check(all(math.isclose(a, b, rel_tol=1e-9) for a, b in zip(lrs, want)) and lrs[2] > 0, f"poly lr {lrs}")
    check(all(C.poly_lr(s, 1000, 0.01) > C.poly_lr(s + 1, 1000, 0.01) for s in range(999)), "poly lr strictly decreasing")
    check(raises(lambda: C.poly_lr(100000, 100000, 0.01)), "step == total refused")


def test_data(proc: Path, exp: dict) -> tuple:
    print("\n--- 8. data ---")
    tr, _ = quiet(D.SliceStore, proc, "train", exp, 2)
    va, _ = quiet(D.SliceStore, proc, "val", exp, 2)
    check(raises(lambda: D.SliceStore(proc, "test", exp)), "the test split is refused")
    check(all(r["split"] == "train" for r in tr.rows) and len(tr) == 48, "train store holds only train rows (48)")
    ok = True
    for i, r in enumerate(tr.rows + va.rows):
        st = tr if i < len(tr) else va
        j = i if i < len(tr) else i - len(tr)
        with np.load(proc / r["path"]) as d:
            img, msk = st.get(j)
            ok &= np.array_equal(img, d["img"]) and np.array_equal(msk, d["msk"])
    check(ok, "store bytes == .npz bytes for every train/val slice")
    check(isinstance(tr.rows[0]["has_kidney"], int), "has_kidney converted to int")
    bad_exp = json.loads(json.dumps(exp))
    bad_exp["train"]["ct"][0] += 1
    check(raises(lambda: D.SliceStore(proc, "train", bad_exp)), "count mismatch refused")
    dr = D.make_draws(tr, 1, 643, 12000)
    mri = np.mean([tr.rows[i]["modality"] == "mri" for _, i in dr])
    kinds = {(tr.rows[i]["has_kidney"], tr.rows[i]["sam_eligible"]) for _, i in dr}
    check(abs(mri - 0.5) < 0.03, f"draws 50/50 CT/MRI ({mri:.3f} MRI)")
    check({(0, 0), (1, 1)} <= kinds and any(tr.rows[i]["z"] == 6 for _, i in dr), "draws include kidney-free and sliver slices")
    check(D.make_draws(tr, 1, 643, 500) == D.make_draws(tr, 1, 643, 500)
          and D.make_draws(tr, 1, 644, 500) != D.make_draws(tr, 1, 643, 500)
          and D.make_draws(tr, 2, 643, 500) != D.make_draws(tr, 1, 643, 500), "draws: pure in (seed, epoch)")
    check(D.position(0, 12000) == (1, 0) and D.position(11999, 12000) == (1, 11999) and D.position(12000, 12000) == (2, 0),
          "global positions -> (epoch, i)")
    # leak detector + corrupted manifest box
    leak = proc.parent / "leak"
    shutil.copytree(proc, leak)
    rows = list(csv.DictReader(open(leak / "manifest.csv")))
    rows[-1]["patient"] = rows[0]["patient"]
    with open(leak / "manifest.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    check(raises(lambda: D.check_split_integrity(leak)), "a patient in two splits is detected")
    rows = list(csv.DictReader(open(proc / "manifest.csv")))
    k = next(i for i, r in enumerate(rows) if r["split"] == "train" and r["rk_box"] != "null")
    b = json.loads(rows[k]["rk_box"])
    rows[k]["rk_box"] = json.dumps([b[0] + 1, b[1], b[2], b[3]])
    shutil.rmtree(leak)
    shutil.copytree(proc, leak)
    with open(leak / "manifest.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    check(raises(lambda: D.SliceStore(leak, "train", exp)), "a manifest box that is not the tight mask box is refused")
    shutil.rmtree(leak)
    return tr, va


def test_items(tr, va) -> None:
    print("\n--- 4/5/8. datasets ---")
    norm = C.norm_from_buffer(tr.img)
    ds = D.TrainItems(tr, "box", 643, norm, 24, 0, 48, FRAME, augment=True)
    a, b = ds[5], ds[5]
    check(torch.equal(a["x"], b["x"]) and torch.equal(a["label"], b["label"]), "training item deterministic by position")
    check(a["x"].shape == (3, FRAME, FRAME) and a["label"].dtype == torch.uint8, "box item: 3 channels; uint8 label")
    nd = D.TrainItems(tr, "plain", 643, norm, 24, 0, 48, FRAME, augment=False)
    ok = True
    for k in range(12):
        it = nd[k]
        img, msk = tr.get(it["idx"])
        ref, _, g = C.to_net_image(img, norm, FRAME)
        ok &= torch.equal(it["x"], ref) and torch.equal(it["label"].long(), C.to_net_label(msk, g))
    check(ok, "unaugmented training item == net_input / to_net_label exactly")
    levels = []
    orig = M.jitter_box

    def spy(box, level, h, w, seed):
        levels.append(level)
        return orig(box, level, h, w, seed)
    D.M.jitter_box = spy
    try:
        for k in range(48):
            ds[k]
    finally:
        D.M.jitter_box = orig
    check(levels and min(levels) >= 0 and max(levels) <= 0.20, f"training jitter levels in [0, 0.20] ({len(levels)} boxes)")
    vi = D.ValItems(va, "box", norm, FRAME)
    ok = True
    for k in range(len(vi)):
        it = vi[k]
        r = va.rows[it["idx"]]
        for lev in (0.0, 0.10):
            want = {s: (None if c is None else M.jitter_box(c, lev, r["h"], r["w"], M.jitter_seed(r["case"], r["z"], s, lev)))
                    for s, c in va.boxes(it["idx"]).items()}
            ok &= torch.equal(it["xs"][C.level_key(lev)][1:], C.box_channels(want, it["geom"]))
    check(ok, "validation boxes == jitter_box(jitter_seed(case, z, side, level)), as the evaluation")
    # 0 vs 2 workers give identical batches
    cfg0 = T.Cfg(batch=4, workers=0)
    cfg2 = T.Cfg(batch=4, workers=2)
    b0 = [x for _, x in zip(range(3), T.train_loader(ds, cfg0))]
    b2 = [x for _, x in zip(range(3), T.train_loader(ds, cfg2))]
    check(all(torch.equal(p["x"], q["x"]) and p["pos"] == q["pos"] for p, q in zip(b0, b2)),
          "0 vs 2 DataLoader workers: identical batches")


def test_val_metric(tr, va) -> None:
    print("\n--- 9. validation metric ---")
    rows = [{"modality": "ct", "patient": "a", "level": "none", "dice": 1.0},
            {"modality": "ct", "patient": "a", "level": "none", "dice": 0.0},
            {"modality": "ct", "patient": "b", "level": "none", "dice": 1.0},
            {"modality": "mri", "patient": "c", "level": "none", "dice": 0.4}]
    s = C.val_metric(rows, "none")
    check(abs(s["ct"] - 0.75) < 1e-12 and abs(s["selection"] - 0.575) < 1e-12, "patient-mean, then modality mean")
    norm = C.norm_from_buffer(tr.img)

    class Const(torch.nn.Module):
        def __init__(self, cls):
            super().__init__()
            self.cls = cls
            self.p = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x):
            out = torch.zeros(x.shape[0], 3, x.shape[2], x.shape[3])
            out[:, self.cls] = 1.0
            return out
    items = D.ValItems(va, "plain", norm, FRAME)
    loader = T.val_loader(items, T.Cfg(val_workers=0, val_batch=4))
    s0 = T.validate(Const(0), items, loader, "cpu", "none")
    check(s0["selection"] == 0 and s0["union_dice"] == 0 and s0["fp_slice_rate"] == 0,
          "all-background model: Dice 0, FP rate 0")
    s1 = T.validate(Const(1), items, loader, "cpu", "none")
    n_inst = sum((r["rk_box"] is not None) + (r["lk_box"] is not None) for r in va.rows)
    check(s1["fp_slice_rate"] == 1.0 and s1["n_rows"] == n_inst and 0 < s1["selection"] < 0.5,
          f"all-'right' model: FP rate 1, left instances score 0 ({n_inst} instances)")


def test_training(proc: Path, exp: dict, tmp: Path, stores) -> None:
    print("\n--- 10-13. training: resume, stamp, seeds, end to end ---")
    a_dir, b_dir = tmp / "a" / "arm3_plain_s643", tmp / "b" / "arm3_plain_s643"
    rc, _ = quiet(T.run_training, tiny_cfg(proc, a_dir, expected_counts=exp), stores)
    check(rc == 0 and json.loads((a_dir / "run.json").read_text())["status"] == "TRAIN_OK", "2-epoch run: TRAIN_OK")
    quiet(T.run_training, tiny_cfg(proc, b_dir, expected_counts=exp, stop_after_epoch=1), stores)
    check(not (b_dir / "run.json").exists() and len(T.read_metrics(b_dir / "metrics.csv")) == 2, "stopped after epoch 1")
    quiet(T.run_training, tiny_cfg(proc, b_dir, expected_counts=exp), stores)
    la, lb = (torch.load(d / "last.pt", weights_only=False) for d in (a_dir, b_dir))
    same_w = all(torch.equal(la["model"][k], lb["model"][k]) for k in la["model"])
    vf = T.val_fields("plain") + ["train_loss", "union_dice", "fp_slice_rate"]
    ma, mb = T.read_metrics(a_dir / "metrics.csv"), T.read_metrics(b_dir / "metrics.csv")
    check(same_w and [[r[k] for k in vf] for r in ma] == [[r[k] for k in vf] for r in mb],
          "resume exactness: 1 + resume + 1 == 2 straight (weights and metrics)")
    best = torch.load(a_dir / "best.pt", weights_only=False)
    check(set(best) >= {"model", "arch", "variant", "norm", "epoch", "score", "seed", "size"}
          and best["arch"]["in_ch"] == 1, "best.pt carries model, arch, variant, norm, epoch, score, seed")
    rj = json.loads((a_dir / "run.json").read_text())
    check(rj["best_sha256"] == T.sha256_file(a_dir / "best.pt"), "run.json records the sha256 of best.pt")
    rc, out = quiet(T.run_training, tiny_cfg(proc, a_dir, expected_counts=exp), stores)
    check(rc == 0 and "TRAIN_OK" in out and "epoch 1/" not in out, "finished run: TRAIN_OK without training")
    check(raises(lambda: T.run_training(tiny_cfg(proc, b_dir, expected_counts=exp, lr=0.02, epochs=3), stores)),
          "changed setting on resume refused")
    c_dir = tmp / "c" / "arm3_plain_s643"
    quiet(T.run_training, tiny_cfg(proc, c_dir, expected_counts=exp, stop_after_epoch=1), stores)
    cj = json.loads((c_dir / "config.json").read_text())
    cj["code"]["student_core.py"] = "0" * 64
    T.atomic_write_json(c_dir / "config.json", cj)
    check(raises(lambda: T.run_training(tiny_cfg(proc, c_dir, expected_counts=exp), stores)), "code change refused")
    rc, _ = quiet(T.run_training, tiny_cfg(proc, c_dir, expected_counts=exp, allow_code_change="test"), stores)
    cj = json.loads((c_dir / "config.json").read_text())
    check(rc == 0 and cj["code_changes"] and cj["code_changes"][0]["reason"] == "test", "--allow-code-change allowed and logged")
    cj["norm"]["mean"] += 1.0
    T.atomic_write_json(c_dir / "config.json", cj)
    (c_dir / "run.json").unlink(missing_ok=True)
    check(raises(lambda: T.run_training(tiny_cfg(proc, c_dir, expected_counts=exp, epochs=2), stores)),
          "different normalisation on resume refused")
    check(raises(lambda: T.run_training(tiny_cfg(proc, tmp / "d" / "arm3_plain_s644", expected_counts=exp), stores)),
          "run folder not named after variant and seed refused")
    # crash consistency: a job killed between last.pt and the exports is repaired on resume
    x_dir = tmp / "x" / "arm3_plain_s643"
    quiet(T.run_training, tiny_cfg(proc, x_dir, expected_counts=exp, stop_after_epoch=1), stores)
    rows_x = T.read_metrics(x_dir / "metrics.csv")
    T.write_metrics(x_dir / "metrics.csv", rows_x[:1], T.metric_fields("plain"))    # lost row 1
    (x_dir / "best.pt").unlink()                                                     # lost export
    rc, out = quiet(T.run_training, tiny_cfg(proc, x_dir, expected_counts=exp), stores)
    lx = torch.load(x_dir / "last.pt", weights_only=False)
    check(rc == 0 and "repairing metrics.csv" in out and "repairing best.pt" in out
          and all(torch.equal(la["model"][k], lx["model"][k]) for k in la["model"])
          and [[r[k] for k in vf] for r in T.read_metrics(x_dir / "metrics.csv")] == [[r[k] for k in vf] for r in ma],
          "killed between writes: metrics.csv and best.pt repaired from last.pt; result == uninterrupted run")
    y_dir = tmp / "y" / "arm3_plain_s643"
    y_dir.mkdir(parents=True)
    shutil.copy(a_dir / "best.pt", y_dir / "best.pt")
    check(raises(lambda: T.run_training(tiny_cfg(proc, y_dir, expected_counts=exp), stores)),
          "best.pt without last.pt: inconsistent folder refused")
    # evaluation readiness: best.pt reloads into a model that reproduces the recorded best score
    model, norm, variant, size, ck = C.load_student(a_dir / "best.pt")
    _, va_ = stores
    items = D.ValItems(va_, variant, norm, size)
    s = T.validate(model, items, T.val_loader(items, T.Cfg(val_workers=0, val_batch=4)), "cpu", "none")
    check(variant == "plain" and size == FRAME and abs(round(s["selection"], 6) - rj["best_score"]) < 1e-9,
          f"load_student(best.pt) reproduces the recorded best val score ({rj['best_score']:.6f})")
    m1, _ = T.build_model(tiny_cfg(proc, a_dir), "cpu")
    m2, _ = T.build_model(tiny_cfg(proc, a_dir), "cpu")
    m3, _ = T.build_model(tiny_cfg(proc, a_dir, seed=644), "cpu")
    check(T.weights_digest(m1) == T.weights_digest(m2) != T.weights_digest(m3), "same seed -> same init; other seed -> different")
    bx = tmp / "e" / "arm3_box_s643"
    rc, _ = quiet(T.run_training, tiny_cfg(proc, bx, expected_counts=exp, variant="box"), stores)
    bm = T.read_metrics(bx / "metrics.csv")
    check(rc == 0 and len(bm) == 3 and "ct@0.10" in bm[0] and torch.load(bx / "best.pt", weights_only=False)["arch"]["in_ch"] == 3,
          "box variant end to end: 3 input channels, per-level metrics, TRAIN_OK")
    check(raises(lambda: T.build_model(T.Cfg(), "cpu")) is False and T.build_model(T.Cfg(variant="box"), "cpu")[1] == U.EXPECTED_PARAMS[3],
          "production build asserts the exact parameter counts")


def test_preflight(proc: Path, exp: dict, tmp: Path, stores) -> None:
    print("\n--- 13. preflight (reduced) ---")
    run = tmp / "pre" / "arm3_preflight"
    cfg = tiny_cfg(proc, run, expected_counts=exp, pf_learn_steps=5, pf_learn_slices=4, pf_steps_plain=3,
                   pf_steps_box=2, pf_probe=4, pf_aug_items=8)
    rc, out = quiet(T.run_preflight, cfg, stores)
    pf = json.loads((run / "preflight.json").read_text())
    names = set(pf["checks"])
    check({"letterbox_ceiling", "pipeline_plain", "pipeline_box", "can_learn", "real_steps_plain", "real_steps_box"} <= names,
          f"preflight ran every check ({len(names)})")
    check(pf["checks"]["letterbox_ceiling"]["ok"] and pf["checks"]["pipeline_plain"]["ok"] and pf["checks"]["pipeline_box"]["ok"],
          f"letterbox ceiling and pipeline checks pass ({ {k: round(v, 3) for k, v in pf['checks']['letterbox_ceiling']['info'].items() if isinstance(v, float)} })")
    check((run / "aug_examples_plain.png").exists() and (run / "aug_examples_box.png").exists()
          and pf["estimate"]["recommended_ARM3_TRAIN_TIME"].endswith(":00:00"), "augmentation sheets and a time recommendation written")


def test_sbatch(tmp: Path) -> None:
    print("\n--- 14. SLURM scripts ---")
    names = ["arm3_preflight.sbatch", "arm3_train_a.sbatch", "arm3_train_b.sbatch"]
    regen = tmp / "regen"
    regen.mkdir()
    subprocess.run([sys.executable, str(HERE / "gen_sbatch_arm3.py"), str(regen)], check=True, capture_output=True)
    check(sorted(p.name for p in regen.glob("*.sbatch")) == names, "generator writes exactly the 3 arm-3 scripts")
    for n in names:
        txt = (HERE / n).read_text()
        check(txt == (regen / n).read_text(), f"{n} == generator output")
        check(subprocess.run(["bash", "-n", str(HERE / n)]).returncode == 0, f"{n} passes bash -n")
        check("#SBATCH --partition=dgx" in txt and "#SBATCH --qos=dgx" in txt and "*A100*" in txt
              and "--expect-gpu A100" in txt, f"{n}: dgx / A100 guards")
    ta, tb = (HERE / "arm3_train_a.sbatch").read_text(), (HERE / "arm3_train_b.sbatch").read_text()
    import re

    def skeleton(t: str) -> str:
        for pat, rep_ in ((r"arm3_train_[ab]", "LOG"), (r"arm3_t[ab]\b", "JOB"),
                          (r"--variant (plain|box) --seed \d+", "VS"), (r"arm3_(plain|box)_s\d+", "RUN"),
                          (r"(plain|box) (student, )?seed \d+", "RS")):
            t = re.sub(pat, rep_, t)
        return t
    mapped = skeleton(ta)
    tb_sk = skeleton(tb)
    check(mapped == tb_sk and ta != tb, "train_a and train_b differ only in run tokens (variant/seed/run-dir/job)")
    # run the scripts with stubs
    stub = tmp / "stub"
    stub.mkdir()
    (stub / "squeue").write_text('#!/bin/bash\necho "${SQUEUE_OUT:-}"\n')
    (stub / "nvidia-smi").write_text('#!/bin/bash\necho "${GPU_NAME:-NVIDIA A100-SXM4-80GB}"\n')
    calls = tmp / "calls.txt"
    (stub / "py").write_text(f'#!/bin/bash\nif [ "$1" = "-c" ]; then echo env-ok; exit 0; fi\n'
                             f'echo "$@" >> {calls}\nn=$(wc -l < {calls} | tr -d \" \")\n'
                             f'if [ "$n" = "${{FAIL_AT:-0}}" ]; then exit 7; fi\nexit 0\n')
    for f in stub.iterdir():
        f.chmod(f.stat().st_mode | stat.S_IEXEC)
    seg = tmp / "seg"
    (seg / "data" / "processed").mkdir(parents=True)
    (seg / "data" / "processed" / "manifest.csv").write_text("x\n")

    def run(name, **env):
        txt = (HERE / name).read_text()
        txt = txt.replace("SEG=/home/rethinkingai-self/vihaan/Segmentation", f"SEG={seg}")
        txt = txt.replace("PY=/home/rethinkingai-self/vihaan/miniconda3/envs/seg311/bin/python", f"PY={stub / 'py'}")
        sp = tmp / f"run_{name}"
        sp.write_text(txt)
        calls.unlink(missing_ok=True)
        e = {**os.environ, "PATH": f"{stub}:{os.environ['PATH']}", **env}
        r = subprocess.run(["bash", str(sp)], env=e, capture_output=True, text=True)
        return r.returncode, (calls.read_text().splitlines() if calls.exists() else []), r.stdout
    rc, cl, _ = run("arm3_train_a.sbatch")
    check(rc == 0 and len(cl) == 2 and "--variant plain --seed 643" in cl[0] and f"--run-dir {seg}/runs/arm3_plain_s643" in cl[0]
          and "--variant plain --seed 644" in cl[1] and "--epochs 100 --steps-per-epoch 1000 --batch 12" in cl[1],
          "train_a: plain 643 then plain 644 with the exact arguments")
    rc, cl, _ = run("arm3_train_b.sbatch")
    check(rc == 0 and "--seed 645" in cl[0] and "--variant box --seed 643" in cl[1] and "arm3_box_s643" in cl[1],
          "train_b: plain 645 then box 643")
    rc, cl, out = run("arm3_train_a.sbatch", FAIL_AT="1")
    check(rc == 7 and len(cl) == 1 and "run 2 not started" in out, "a failing run 1 stops the script before run 2")
    rc, cl, out = run("arm3_train_a.sbatch", SQUEUE_OUT="123 arm3_ta RUNNING")
    check(rc == 3 and not cl, "a second copy of the same script is refused")
    rc, cl, out = run("arm3_train_b.sbatch", SQUEUE_OUT="123 arm3_ta RUNNING")
    check(rc == 0 and len(cl) == 2, "train_b runs alongside train_a")
    rc, cl, out = run("arm3_preflight.sbatch", SQUEUE_OUT="123 arm3_tb RUNNING")
    check(rc == 3, "the preflight refuses next to a training job")
    rc, cl, out = run("arm3_train_a.sbatch", GPU_NAME="NVIDIA A40")
    check(rc == 4 and not cl, "an A40 node is refused")
    rc, cl, out = run("arm3_preflight.sbatch")
    check(rc == 0 and len(cl) == 1 and "--preflight" in cl[0] and "arm3_preflight" in cl[0], "preflight command")


def test_gpu_guard_and_isolation() -> None:
    print("\n--- 15-16. GPU guard, isolation ---")
    a100 = {"cuda": True, "name": "NVIDIA A100-SXM4-80GB", "total_gib": 79.3, "capability": [8, 0], "bf16": True}
    check(T.gpu_check("A100", a100)["name"].startswith("NVIDIA A100"), "A100-80GB accepted")
    check(raises(lambda: T.gpu_check("A100", {**a100, "name": "NVIDIA A40", "total_gib": 44.4})), "A40 refused")
    check(raises(lambda: T.gpu_check("A100", {**a100, "total_gib": 39.4})), "A100-40GB refused")
    code = ("import sys; sys.path.insert(0, %r); import student_unet, student_core, student_aug, student_data, "
            "train_student; bad = [m for m in ('segment_anything', 'train_sam_lora', 'sam_train_core', 'sam_lora', "
            "'train_data') if m in sys.modules]; print(bad); sys.exit(1 if bad else 0)") % str(HERE)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    check(r.returncode == 0, f"arm-3 modules import no SAM / arm-2 code {r.stdout.strip()}")


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="arm3_test_"))
    try:
        proc = tmp / "processed"
        exp = build_processed(proc)
        test_architecture()
        test_letterbox()
        test_augmentation()
        test_boxes()
        test_losses_schedule()
        tr, va = test_data(proc, exp)
        test_items(tr, va)
        test_val_metric(tr, va)
        test_training(proc, exp, tmp, (tr, va))
        test_preflight(proc, exp, tmp, (tr, va))
        test_sbatch(tmp)
        test_gpu_guard_and_isolation()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\nALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
