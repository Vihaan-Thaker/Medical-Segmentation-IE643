#!/usr/bin/env python
"""Tests for sam_metrics.py, eval_sam.py and aggregate_arm.py.

Runs the whole arm-1 pipeline against a synthetic dataset and a stub SAM whose errors are
known in advance, so every reported number can be checked against a value derived
independently of the code under test.

    python test_eval_sam.py        # exits 0 and prints ALL TESTS PASSED
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aggregate_arm  # noqa: E402
import eval_sam  # noqa: E402
import sam_metrics as M  # noqa: E402


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)
    print(f"  ok  {msg}")


def disk(h, w, cy, cx, r):
    yy, xx = np.mgrid[0:h, 0:w]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r


# ------------------------------------------------------------------ 1. metrics
def test_metrics() -> None:
    print("\n--- metrics ---")
    M.self_test()
    check(True, "recorded sanity values hold (self_test)")

    gt = disk(120, 120, 60, 60, 20)
    sp = (1.0, 1.0)

    # Dice must equal the reference implementation, not just our own arithmetic.
    import surface_distance as sd
    for pred in (gt, np.roll(gt, 3, axis=1), disk(120, 120, 60, 60, 14)):
        ours = M.dice_coefficient(gt, pred)
        ref = float(sd.compute_dice_coefficient(gt, pred))
        assert abs(ours - ref) < 1e-12, (ours, ref)
    check(True, "Dice matches DeepMind's reference implementation exactly")

    # C2: spacing is (row, col); swapping it must change the answer.
    pr = np.roll(gt, 2, axis=0)
    a = M.score_masks(gt, pr, (3.0, 1.0))["nsd_2"]
    b = M.score_masks(gt, pr, (1.0, 3.0))["nsd_2"]
    check(a < 0.9 < b, f"spacing order reaches the metric: (3,1)->{a:.4f}, (1,3)->{b:.4f}")

    # A shift along rows with coarse row spacing must score worse than the same shift
    # along columns with coarse column spacing -- i.e. mm, not pixels, drive NSD.
    shift_r = M.score_masks(gt, np.roll(gt, 2, axis=0), (3.0, 1.0))["nsd_2"]
    shift_c = M.score_masks(gt, np.roll(gt, 2, axis=1), (3.0, 1.0))["nsd_2"]
    check(shift_r < shift_c, f"NSD is in mm, not pixels ({shift_r:.3f} < {shift_c:.3f})")

    for bad, why in (
        (lambda: M.score_masks(gt.astype(np.uint8), gt.astype(np.uint8), sp), "uint8 rejected"),
        (lambda: M.score_masks(np.zeros_like(gt), gt, sp), "empty GT rejected"),
        (lambda: M.score_masks(gt, gt, (1.0, 1.0, 1.0)), "wrong spacing length rejected"),
        (lambda: M.score_masks(gt, gt[:, :60], sp), "shape mismatch rejected"),
        (lambda: M.score_masks(gt, gt, (0.0, 1.0)), "non-positive spacing rejected"),
    ):
        try:
            bad()
            raise AssertionError(f"{why}: did not raise")
        except M.MetricError:
            pass
    check(True, "preconditions enforced: uint8, empty GT, bad spacing, shape mismatch")

    s = M.score_masks(gt, np.zeros_like(gt), sp)
    check(s["dice"] == 0.0 and all(s[f"nsd_{t:g}"] == 0.0 for t in M.TOLERANCES_MM),
          "empty prediction scores 0 on every metric without raising")

    # 3D, anisotropic: a one-slice shift at 5 mm spacing must hurt more than at 1 mm.
    g3 = np.zeros((20, 60, 60), bool)
    g3[5:15, 15:45, 15:45] = True
    p3 = np.roll(g3, 1, axis=0)
    aniso = M.score_masks(g3, p3, (5.0, 0.7, 0.7))["nsd_2"]
    iso = M.score_masks(g3, p3, (1.0, 1.0, 1.0))["nsd_2"]
    check(aniso < iso, f"3D anisotropic spacing honoured ({aniso:.3f} < {iso:.3f})")

    # ANALYTIC ground truth. Dilating the GT by exactly d mm puts every boundary point d mm
    # from the other boundary, so NSD must step 0 -> 1 as the tolerance crosses d. This is
    # derived from the definition of NSD, independent of any library's conventions, and is
    # therefore a stronger correctness check than comparing against a second implementation.
    big = disk(240, 240, 120, 120, 40)
    s3 = M.score_masks(big, disk(240, 240, 120, 120, 43), (1.0, 1.0))
    check(s3["nsd_1"] < 0.02 and s3["nsd_3"] > 0.90 and s3["nsd_5"] > 0.99,
          f"3 mm dilation: NSD steps 0->1 at tolerance 3 mm "
          f"(@1={s3['nsd_1']:.3f} @2={s3['nsd_2']:.3f} @3={s3['nsd_3']:.3f} @5={s3['nsd_5']:.3f})")
    s6 = M.score_masks(big, disk(240, 240, 120, 120, 46), (1.0, 1.0))
    check(s6["nsd_3"] < 0.02 and s6["nsd_5"] < 0.40,
          f"6 mm dilation: still ~0 at tolerance 3 and 5 mm "
          f"(@3={s6['nsd_3']:.3f} @5={s6['nsd_5']:.3f})")

    prev = {t: 1.1 for t in M.TOLERANCES_MM}
    mono = True
    for d in range(0, 9):
        sd_ = M.score_masks(big, disk(240, 240, 120, 120, 40 + d), (1.0, 1.0))
        for t in M.TOLERANCES_MM:
            if sd_[f"nsd_{t:g}"] > prev[t] + 1e-9:
                mono = False
            prev[t] = sd_[f"nsd_{t:g}"]
    check(mono, "NSD never increases as the prediction degrades (dilations 0-8 mm)")

    # MONAI agrees on ORDERING but not on absolute value: it defines boundaries by
    # morphological erosion while DeepMind shifts the grid half a pixel, so the two differ
    # by ~0.14 even on square pixels (arXiv 2410.02630; D24a/C1 is why we use DeepMind).
    # Equal ranking still catches API misuse -- swapped arguments or transposed spacing
    # would break monotonicity.
    try:
        import torch
        from monai.metrics import compute_surface_dice
        oh = lambda m: torch.from_numpy(np.stack([~m, m])[None]).float()
        ours, mon = [], []
        for d in (0, 2, 4, 6):
            pr = disk(240, 240, 120, 120, 40 + d)
            ours.append(M.score_masks(big, pr, (1.0, 1.0))["nsd_3"])
            mon.append(float(compute_surface_dice(oh(pr), oh(big), class_thresholds=[3.0],
                                                  include_background=False,
                                                  spacing=(1.0, 1.0))[0, 0]))
        order_ok = all(a >= b - 1e-9 for a, b in zip(ours, ours[1:])) and \
                   all(a >= b - 1e-9 for a, b in zip(mon, mon[1:]))
        check(order_ok, "MONAI and DeepMind rank the same degradations identically "
                        f"(ours {[round(v, 2) for v in ours]}, MONAI {[round(v, 2) for v in mon]}; "
                        "absolute values differ by design)")
    except ImportError:
        print("  --  MONAI not installed here; ranking cross-check skipped (runs in seg311)")


# ------------------------------------------------------------------- 2. jitter
def test_jitter() -> None:
    print("\n--- box jitter ---")
    box, h, w = (40, 50, 80, 110), 200, 200
    check(M.jitter_box(box, 0.0, h, w, 1) == box, "jitter 0% returns the clean box exactly")

    s = M.jitter_seed("case", 7, M.RIGHT, 0.1)
    check(M.jitter_box(box, 0.1, h, w, s) == M.jitter_box(box, 0.1, h, w, s),
          "same seed gives the same box (reproducible under resume)")
    check(M.jitter_seed("a", 1, 1, 0.1) != M.jitter_seed("a", 2, 1, 0.1),
          "seed varies with slice index")
    check(M.jitter_seed("a", 1, 1, 0.1) == M.jitter_seed("a", 1, 1, 0.1),
          "seed is stable across processes (sha256, not salted hash)")

    rng = np.random.default_rng(0)
    worst = 1.0
    for _ in range(3000):
        # Non-degenerate, like every real box: a scored kidney has >= 50 px.
        x0 = int(rng.integers(0, 35)); x1 = int(rng.integers(x0 + 1, 40))
        y0 = int(rng.integers(0, 35)); y1 = int(rng.integers(y0 + 1, 40))
        b = (x0, y0, x1, y1)
        for lv in M.JITTER_LEVELS:
            j = M.jitter_box(b, lv, 40, 40, int(rng.integers(1 << 30)))
            assert 0 <= j[0] < j[2] < 40 and 0 <= j[1] < j[3] < 40, (b, lv, j)
            worst = min(worst, M.box_iou(b, j))
    check(True, "3000 random boxes x 4 levels: never inverted, never out of bounds")
    check(0.0 <= worst < 1.0, f"box_iou degrades with jitter (worst seen {worst:.3f})")
    check(M.box_iou(box, box) == 1.0, "box_iou of a box with itself is 1.0")

    big = [M.box_iou(box, M.jitter_box(box, 0.20, h, w, i)) for i in range(400)]
    small = [M.box_iou(box, M.jitter_box(box, 0.05, h, w, i)) for i in range(400)]
    check(np.mean(big) < np.mean(small),
          f"20% jitter is worse than 5% ({np.mean(big):.3f} < {np.mean(small):.3f})")


# -------------------------------------------------------------- 3. aggregation
def test_aggregation() -> None:
    print("\n--- aggregation ---")
    # One patient with 100 perfect instances, nine with 1 poor instance each.
    rows = [{"patient": "p0", "dice": 1.0} for _ in range(100)]
    rows += [{"patient": f"p{i}", "dice": 0.0} for i in range(1, 10)]
    flat = float(np.mean([r["dice"] for r in rows]))
    agg = M.aggregate_by_patient(rows, "dice", n_boot=2000)
    check(abs(flat - 0.917) < 0.01 and abs(agg["mean"] - 0.1) < 1e-9,
          f"per-patient mean {agg['mean']:.3f} resists the dominant patient "
          f"(flat mean would be {flat:.3f})")
    check(agg["n_patients"] == 10 and agg["n_instances"] == 109,
          "patient and instance counts both reported")

    vals = [0.5] * 20
    m, lo, hi = M.bootstrap_ci(vals, n_boot=2000)
    check(m == 0.5 and lo == 0.5 and hi == 0.5, "zero-variance data gives a zero-width CI")
    rng = np.random.default_rng(0)
    m, lo, hi = M.bootstrap_ci(list(rng.normal(0.8, 0.1, 200)), n_boot=4000)
    check(lo < m < hi and (hi - lo) < 0.1, f"CI brackets the mean and is tight (n=200): "
                                           f"[{lo:.3f}, {hi:.3f}]")
    m1 = M.bootstrap_ci([0.1, 0.9] * 6, n_boot=3000)
    m2 = M.bootstrap_ci([0.1, 0.9] * 60, n_boot=3000)
    check((m1[2] - m1[1]) > (m2[2] - m2[1]), "fewer patients give a wider CI")
    check(all(np.isnan(x) for x in M.bootstrap_ci([])), "empty input gives NaN, not a crash")


# ---------------------------------------------------- 4. end-to-end with a stub
class StubPredictor:
    """Stands in for SamPredictor. Returns the GT disk eroded/dilated by a known amount,
    so the expected Dice is computable by hand, and counts encoder calls."""

    def __init__(self, truth: dict, radius_delta: int = -2):
        self.truth = truth
        self.radius_delta = radius_delta
        self.n_set_image = 0
        self.n_predict = 0
        self._key = None

    def set_image(self, image, image_format="RGB"):
        assert image.ndim == 3 and image.shape[2] == 3, "SAM needs HWC RGB"
        assert image.dtype == np.uint8, "SAM needs uint8"
        self.n_set_image += 1
        self._key = image.shape[:2]

    def predict(self, box, multimask_output=False):
        assert multimask_output is False, "arm 1 uses a single mask per box (D24a)"
        assert box.shape == (4,), "box must be XYXY"
        self.n_predict += 1
        h, w = self._key
        x0, y0, x1, y1 = box
        cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
        r = max(1.0, (min(x1 - x0, y1 - y0) / 2) + self.radius_delta)
        m = disk(h, w, cy, cx, r)
        return m[None], np.array([0.9], np.float32), None


def build_synthetic(root: Path) -> dict:
    """Two datasets x two patients, each a small volume with two kidneys, plus one
    sub-threshold sliver slice so the 3D sliver accounting is exercised."""
    H = W = 48
    rows = []
    (root / "slices").mkdir(parents=True)
    for ds, split, modality, npat in (("amos", "test", "ct", 2), ("word", "external", "ct", 2)):
        for p in range(npat):
            case = f"{ds}_{p:04d}"
            cdir = root / "slices" / ds / case
            cdir.mkdir(parents=True)
            nz = 7
            for z in range(nz):
                msk = np.zeros((H, W), np.uint8)
                img = np.full((H, W), 40, np.uint8)
                if 1 <= z <= 5:
                    r = 7 if z != 5 else 2          # z=5 is a sliver (<50 px)
                    msk[disk(H, W, 24, 14, r)] = M.RIGHT
                    msk[disk(H, W, 24, 34, r)] = M.LEFT
                    img[msk > 0] = 200
                rel = f"slices/{ds}/{case}/{z:04d}.npz"
                np.savez_compressed(root / rel, img=img, msk=msk)
                rk, lk = msk == M.RIGHT, msk == M.LEFT

                def bbox(m):
                    if m.sum() < 50:
                        return None
                    ys, xs = np.nonzero(m)
                    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
                rb, lb = bbox(rk), bbox(lk)
                rows.append({
                    "dataset": ds, "patient": case, "case": case, "sequence": "",
                    "modality": modality, "split": split, "z": z, "n_slices": nz,
                    "has_kidney": int(rk.any() or lk.any()),
                    "rk_area": int(rk.sum()), "lk_area": int(lk.sum()),
                    "rk_box": json.dumps(rb), "lk_box": json.dumps(lb),
                    "sam_eligible": int(rb is not None or lb is not None),
                    "h": H, "w": W, "sp_row": 1.0, "sp_col": 1.0, "sp_slice": 3.0,
                    "path": rel,
                })
            # a train case that must never be evaluated
    for p in range(1):
        case = "amos_9000"
        cdir = root / "slices" / "amos" / case
        cdir.mkdir(parents=True)
        msk = np.zeros((H, W), np.uint8)
        msk[disk(H, W, 24, 14, 7)] = M.RIGHT
        img = np.full((H, W), 40, np.uint8)
        rel = f"slices/amos/{case}/0000.npz"
        np.savez_compressed(root / rel, img=img, msk=msk)
        ys, xs = np.nonzero(msk == M.RIGHT)
        rows.append({"dataset": "amos", "patient": case, "case": case, "sequence": "",
                     "modality": "ct", "split": "train", "z": 0, "n_slices": 1, "has_kidney": 1,
                     "rk_area": int((msk == M.RIGHT).sum()), "lk_area": 0,
                     "rk_box": json.dumps([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]),
                     "lk_box": "null", "sam_eligible": 1, "h": H, "w": W,
                     "sp_row": 1.0, "sp_col": 1.0, "sp_slice": 3.0, "path": rel})

    with open(root / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return {"n_eval_instances": sum(
        (json.loads(r["rk_box"]) is not None) + (json.loads(r["lk_box"]) is not None)
        for r in rows if r["split"] in eval_sam.EVAL_SPLITS)}


PRODUCTION_GROUPS = dict(eval_sam.EXPECTED_GROUPS)


def use_synthetic_anchors(proc: Path) -> None:
    """The production anchors describe the real 231-case set; point them at the synthetic
    one so a correct synthetic run can pass. Restored by restore_anchors()."""
    cases = eval_sam.group_cases(eval_sam.load_manifest(proc))
    eval_sam.EXPECTED_GROUPS = eval_sam.count_groups(cases, sorted(cases))


def restore_anchors() -> None:
    eval_sam.EXPECTED_GROUPS = dict(PRODUCTION_GROUPS)


def test_end_to_end() -> int:
    print("\n--- end to end (stub SAM) ---")
    tmp = Path(tempfile.mkdtemp(prefix="arm1_test_"))
    proc, out = tmp / "processed", tmp / "arm1"
    info = build_synthetic(proc)

    stub = StubPredictor({})
    eval_sam.build_predictor = lambda *a, **k: stub
    ckpt = tmp / "fake.pth"
    ckpt.write_bytes(b"x")
    args = ["--processed", str(proc), "--checkpoint", str(ckpt), "--out", str(out),
            "--device", "cpu", "--qc-per-group", "4"]

    # The production anchors (D22a) demand the real 231-case set; anything else must fail
    # at start-up -- before the model is even loaded...
    stub.n_set_image = 0
    rc = eval_sam.main(args)
    check(rc == 1 and stub.n_set_image == 0 and not (out / "instances.csv").exists(),
          "D22a anchor guard fails at start-up, before any GPU work or output")
    # ...and with the anchors set to the synthetic truth, a correct run must SUCCEED.
    use_synthetic_anchors(proc)
    shutil.rmtree(out)
    stub.n_set_image = stub.n_predict = 0
    rc = eval_sam.main(args)
    check(rc == 0, "a correct run exits 0 (EVAL_OK), with the guard set to the true count")

    with open(out / "instances.csv") as fh:
        inst = list(csv.DictReader(fh))
    with open(out / "volumes_3d.csv") as fh:
        vols = list(csv.DictReader(fh))

    n_expect = info["n_eval_instances"] * len(M.JITTER_LEVELS)
    check(len(inst) == n_expect,
          f"{len(inst)} instance rows = {info['n_eval_instances']} instances x "
          f"{len(M.JITTER_LEVELS)} jitter levels")
    check("amos_9000" not in {r["case"] for r in inst}, "train split never evaluated")

    n_slices = len({(r["case"], r["z"]) for r in inst})
    run = json.load(open(out / "run.json"))
    n_qc = run["qc_overlays"]
    # QC re-runs each overlay instance once (1 encode + 1 decode), after scoring.
    check(stub.n_set_image == n_slices + n_qc and stub.n_predict == len(inst) + n_qc,
          f"scoring encoded once per slice ({n_slices}) and decoded per instance "
          f"({len(inst)}); QC added exactly {n_qc} re-runs")
    check(set(run["seconds_per_slice_by_stage"]) == set(eval_sam.STAGES),
          f"per-stage timing recorded: {run['seconds_per_slice_by_stage']}")
    check(run["qc_max_dice_diff"] < 1e-5,
          f"QC re-run reproduces the stored Dice (max diff {run['qc_max_dice_diff']:.1e})")
    groups_with_qc = {p.parent.name for p in (out / "qc").rglob("*.png")}
    check(groups_with_qc == {"amos_ct", "word_ct"},
          f"every group gets QC overlays, generated from the CSV: {sorted(groups_with_qc)}")

    zero = [r for r in inst if r["jitter"] == "0.0"]
    check(all(r["box_iou"] == "1.0" for r in zero), "jitter 0% keeps box_iou = 1.0")
    check(all(0.0 < float(r["dice"]) < 1.0 for r in zero),
          "stub's deliberately shrunken mask gives a partial Dice, as designed")

    # Independent recomputation of one row straight from the stored npz.
    r = zero[0]
    row_m = next(x for x in csv.DictReader(open(proc / "manifest.csv"))
                 if x["case"] == r["case"] and x["z"] == r["z"])
    with np.load(proc / row_m["path"]) as d:
        gt = d["msk"] == (M.RIGHT if r["side"] == "right" else M.LEFT)
    box = np.array(json.loads(r["box_jit"]), float)
    stub.set_image(np.zeros((int(row_m["h"]), int(row_m["w"]), 3), np.uint8))
    pred = np.asarray(stub.predict(box=box)[0][0], bool)
    want = M.score_masks(gt, pred, (float(row_m["sp_row"]), float(row_m["sp_col"])))
    check(abs(want["dice"] - float(r["dice"])) < 1e-6 and
          abs(want["nsd_2"] - float(r["nsd_2"])) < 1e-6,
          "a row recomputed independently from the npz matches instances.csv")

    sl = [r for r in vols if float(r["sliver_frac"]) > 0]
    check(bool(sl), f"3D sliver accounting non-zero for {len(sl)} (case, side, jitter) rows")
    check(all(int(r["n_scored"]) < int(r["n_slices"]) for r in vols),
          "3D rows record that fewer slices were scored than exist")
    check(any((out / "qc").rglob("*.png")), "QC overlays written")

    rep = list(csv.DictReader(open(out / "spacing_report.csv")))
    check(len(rep) == 2 and all(r["nsd1_subpixel"] == "0" for r in rep),
          "spacing report flags sub-pixel tolerances (none here at 1.0 mm)")

    # ---- resume: the three failure modes that matter after a timeout or crash
    def read_rows(name):
        with open(out / name) as fh:
            return list(csv.DictReader(fh))

    def rewrite(name, fields, keep):
        rs = [r for r in read_rows(name) if keep(r)]
        with open(out / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rs)

    def assert_complete(label):
        ii, vv = read_rows("instances.csv"), read_rows("volumes_3d.csv")
        ki = [eval_sam.inst_key(r) for r in ii]
        kv = [eval_sam.vol_key(r) for r in vv]
        check(len(ii) == n_expect and len(ki) == len(set(ki)) and
              len(vv) == len(vols) and len(kv) == len(set(kv)),
              f"{label}: {len(ii)} instance + {len(vv)} volume rows, no duplicates")

    # (a) clean resume: nothing to do, nothing added
    rc = eval_sam.main(args + ["--resume"])
    check(rc == 0, "resume on a complete run exits 0")
    assert_complete("clean resume")

    # (b) partial case: the crash hit mid-write, leaving half of one case's rows
    victim = "word_0001"
    vrows = [r for r in read_rows("instances.csv") if r["case"] == victim]
    drop = {eval_sam.inst_key(r) for r in vrows[: len(vrows) // 2]}
    rewrite("instances.csv", eval_sam.INSTANCE_FIELDS, lambda r: eval_sam.inst_key(r) not in drop)
    rc = eval_sam.main(args + ["--resume"])
    check(rc == 0, f"resume redoes partial case {victim} and exits 0")
    assert_complete("after partial-case resume")
    check(any(p.name.startswith("instances.bak-") for p in out.iterdir()),
          "resume backed up the CSVs before rewriting them")

    # (c) instances complete but the case's volume rows never written
    victim = "amos_0001"
    rewrite("volumes_3d.csv", eval_sam.VOLUME_FIELDS, lambda r: r["case"] != victim)
    rc = eval_sam.main(args + ["--resume"])
    check(rc == 0, f"resume detects {victim} has no volume rows and redoes it")
    assert_complete("after missing-volumes resume")

    # (d) the last line was cut off mid-write by the kill (no trailing newline)
    raw = (out / "instances.csv").read_text()
    (out / "instances.csv").write_text(raw[: len(raw) - 25])
    rc = eval_sam.main(args + ["--resume"])
    check(rc == 0, "resume drops a cut-off last line and redoes that case")
    assert_complete("after cut-off-line resume")

    # (e) volumes_3d.csv missing entirely (and an empty instances file in another dir)
    (out / "volumes_3d.csv").unlink()
    rc = eval_sam.main(args + ["--resume"])
    check(rc == 0, "resume with volumes_3d.csv missing rebuilds it with a header")
    assert_complete("after missing-file resume")
    check((out / "volumes_3d.csv").read_text().splitlines()[0].split(",") == eval_sam.VOLUME_FIELDS,
          "the rebuilt volumes_3d.csv starts with the right header")

    # (f) a fresh start over existing results is refused (forgot RESUME=1)
    before = (out / "instances.csv").read_bytes()
    try:
        eval_sam.main(args)
        check(False, "a fresh run over existing results must be refused")
    except SystemExit as e:
        check("--resume" in str(e) and (out / "instances.csv").read_bytes() == before,
              "a fresh run over existing results is refused; files untouched")

    # (g) a header from a different code version is refused, not rewritten
    good = (out / "volumes_3d.csv").read_text()
    lines = good.splitlines(keepends=True)
    (out / "volumes_3d.csv").write_text(lines[0].replace("dice3d", "dice_3d") + "".join(lines[1:]))
    try:
        eval_sam.main(args + ["--resume"])
        check(False, "a foreign header must be refused")
    except SystemExit as e:
        check("columns differ" in str(e), "resume refuses a CSV whose columns differ from the code's")
    (out / "volumes_3d.csv").write_text(good)

    # (h) a complete case whose content is wrong (left/right swapped) is refused on resume
    good_i = (out / "instances.csv").read_text()
    ii = read_rows("instances.csv")
    for r in ii:
        if r["case"] == "amos_0000":
            r["side"] = {"right": "left", "left": "right"}[r["side"]]
    rewrite_rows = ii
    with open(out / "instances.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=eval_sam.INSTANCE_FIELDS)
        w.writeheader()
        w.writerows(rewrite_rows)
    try:
        eval_sam.main(args + ["--resume"])
        check(False, "swapped sides must be refused")
    except SystemExit as e:
        check("inconsistent rows" in str(e), "resume refuses rows inconsistent with the manifest")
    (out / "instances.csv").write_text(good_i)
    rc = eval_sam.main(args + ["--resume"])
    check(rc == 0, "restored files resume cleanly")
    assert_complete("after restore")

    restore_anchors()

    rc = aggregate_arm.main(["--arm-dir", str(out), "--label", "test", "--n-boot", "500"])
    check(rc == 0, "aggregate_arm runs on the stub output")
    summ = list(csv.DictReader(open(out / "summary.csv")))
    check(any(r["level"] == "3d" for r in summ) and any(r["level"] == "2d" for r in summ),
          "summary.csv holds both 2D and 3D rows")
    g = next(r for r in summ if r["level"] == "2d" and r["metric"] == "dice"
             and float(r["jitter"]) == M.HEADLINE_JITTER and r["group"] == "word/ct")
    check(int(g["n_patients"]) == 2, "summary groups by dataset/modality and counts patients")
    check((out / "summary.md").exists() and (out / "breakdowns.csv").exists(),
          "summary.md and breakdowns.csv written")

    print(f"\nscratch: {tmp}")
    return 0


def test_sharding() -> None:
    """N parallel jobs (--shard K --num-shards N) + one --merge N must give exactly the rows
    a single job gives, never write a file another job writes, and refuse to merge
    anything incomplete or inconsistent."""
    print("\n--- sharding: parallel jobs + merge ---")
    import hashlib
    tmp = Path(tempfile.mkdtemp(prefix="arm1_shard_"))
    proc = tmp / "processed"
    build_synthetic(proc)
    stub = StubPredictor({})
    eval_sam.build_predictor = lambda *a, **k: stub
    use_synthetic_anchors(proc)
    ckpt = tmp / "fake.pth"
    ckpt.write_bytes(b"x")

    def argv(out, *extra):
        return ["--processed", str(proc), "--checkpoint", str(ckpt), "--out", str(out),
                "--device", "cpu", "--qc-per-group", "4", *extra]

    def rows(path):
        with open(path) as fh:
            return list(csv.DictReader(fh))

    def table(out):
        """key -> every scored column, for instances and volumes."""
        ti = {eval_sam.inst_key(r): tuple(r[f] for f in eval_sam.INSTANCE_FIELDS)
              for r in rows(out / "instances.csv")}
        tv = {eval_sam.vol_key(r): tuple(r[f] for f in eval_sam.VOLUME_FIELDS)
              for r in rows(out / "volumes_3d.csv")}
        return ti, tv, len(rows(out / "instances.csv")), len(rows(out / "volumes_3d.csv"))

    def digest(out):
        return {n: hashlib.sha256((out / n).read_bytes()).hexdigest()
                for n in ("instances.csv", "volumes_3d.csv") if (out / n).exists()}

    # Reference: one ordinary job.
    ref = tmp / "ref"
    check(eval_sam.main(argv(ref)) == 0, "reference single job exits 0")
    ref_i, ref_v, ref_ni, ref_nv = table(ref)

    # Unit: verify_rows accepts genuine rows and catches each misalignment it exists for.
    rws = eval_sam.load_manifest(proc)
    cases = eval_sam.group_cases(rws)
    mbk = {(r["case"], r["z"]): r for r in rws}
    gi, gv = rows(ref / "instances.csv"), rows(ref / "volumes_3d.csv")
    check(eval_sam.verify_rows(gi, gv, cases, mbk) == [], "verify_rows accepts every genuine row")
    for what, mutate in (
        ("z shifted by one", lambda r: r.update(z=str(int(r["z"]) + 1))),
        ("left/right swapped", lambda r: r.update(side={"right": "left", "left": "right"}[r["side"]])),
        ("jittered box from another seed", lambda r: r.update(box_jit=json.dumps(
            [int(v) + 1 for v in json.loads(r["box_jit"])]))),
        ("Dice out of range", lambda r: r.update(dice="1.5")),
        ("NSD falling with tolerance", lambda r: r.update(nsd_5="0.0", nsd_1="0.9")),
        ("wrong patient id", lambda r: r.update(patient="someone_else")),
    ):
        bad_rows = [dict(r) for r in gi]
        target = next(r for r in bad_rows if r["jitter"] != "0.0" and 0 < int(r["z"]) < 4
                      and float(r["nsd_1"]) > 0)
        mutate(target)
        check(len(eval_sam.verify_rows(bad_rows, gv, cases, mbk)) >= 1, f"verify_rows catches: {what}")
    bad_v = [dict(r) for r in gv]
    bad_v[0]["gt_vox"] = str(int(bad_v[0]["gt_vox"]) + 1)
    check(eval_sam.verify_rows(gi, bad_v, cases, mbk) != [], "verify_rows catches a wrong 3D gt_vox")

    # Unit: the layout check catches a missing slice (the 3D stack would silently compress).
    c0 = sorted(cases)[0]
    check(eval_sam.check_case_layout(c0, cases[c0]) == [], "layout check accepts a full volume")
    holed = [r for r in cases[c0] if r["z"] != 3]
    check(eval_sam.check_case_layout(c0, holed) != [], "layout check catches a missing slice")

    # Unit: the production anchors add up to the D22a totals.
    tot = [sum(v[i] for v in PRODUCTION_GROUPS.values()) for i in range(3)]
    check(tot == [231, 7386, 13145], f"D22a anchors total 231 cases / 7,386 slices / 13,145 instances: {tot}")

    # Unit: the split is deterministic, disjoint, complete and balanced by slices.
    todo = sorted(c for c in cases if cases[c][0]["split"] in eval_sam.EVAL_SPLITS)
    for n in (1, 2, 3, 4, 7):
        parts = eval_sam.assign_shards(todo, cases, n)
        flat = [c for p_ in parts for c in p_]
        check(len(parts) == n and sorted(flat) == todo and len(flat) == len(set(flat)) and
              parts == eval_sam.assign_shards(list(reversed(todo)), cases, n),
              f"assign_shards N={n}: disjoint, covers all {len(todo)} cases, order-independent")

    # Argument guards.
    for bad in (["--shard", "0"], ["--num-shards", "2"], ["--shard", "2", "--num-shards", "2"],
                ["--shard", "0", "--num-shards", "2", "--merge", "2"]):
        try:
            eval_sam.main(argv(tmp / "bad", *bad))
            check(False, f"{bad} should be rejected")
        except SystemExit as e:
            check(e.code == 2, f"rejects {' '.join(bad)}")

    # ---- Scenario A, the real one: base already holds the AMOS cases complete (as job
    # 336480 did) plus a PARTIAL word case; three shards cover the rest (one gets nothing).
    out = tmp / "arm1"
    out.mkdir()
    keep_full = {"amos_0000", "amos_0001"}
    half = [r for r in rows(ref / "instances.csv") if r["case"] == "word_0000"]
    half = half[: len(half) // 2]
    base_i = [r for r in rows(ref / "instances.csv") if r["case"] in keep_full] + half
    base_v = [r for r in rows(ref / "volumes_3d.csv") if r["case"] in keep_full]
    eval_sam._rewrite(out / "instances.csv", eval_sam.INSTANCE_FIELDS, base_i)
    eval_sam._rewrite(out / "volumes_3d.csv", eval_sam.VOLUME_FIELDS, base_v)
    before = digest(out)

    N = 3
    for k in (0, 1):
        check(eval_sam.main(argv(out, "--shard", str(k), "--num-shards", str(N))) == 0,
              f"shard {k}/{N} exits 0 (SHARD_OK)")
    check(digest(out) == before, "shards never modify the base files")

    rc = eval_sam.main(argv(out, "--merge", str(N)))
    check(rc == 1 and digest(out) == before,
          "merge with a shard missing fails and leaves the base files untouched")

    check(eval_sam.main(argv(out, "--shard", "2", "--num-shards", str(N))) == 0,
          f"shard 2/{N} exits 0")
    plans = [json.load(open(eval_sam.shard_dir(out, k, N) / "assigned.json"))["cases"]
             for k in range(N)]
    flat = [c for p_ in plans for c in p_]
    check(sorted(flat) == ["word_0000", "word_0001"] and len(flat) == len(set(flat)),
          f"shards took exactly the incomplete cases, once each (partial base case redone): {plans}")
    check(any(len(p_) == 0 for p_ in plans), "a shard with no cases still succeeds")

    stub.n_set_image = stub.n_predict = 0
    rc = eval_sam.main(argv(out, "--merge", str(N)))
    check(rc == 0, "merge of base + 3 shards exits 0 (EVAL_OK)")
    got_i, got_v, ni, nv = table(out)
    check(ni == ref_ni and nv == ref_nv and got_i == ref_i and got_v == ref_v,
          f"merged output is row-for-row identical to the single job ({ni} + {nv} rows)")
    run = json.load(open(out / "run.json"))
    check(run["merge"]["cases_from"] == {"base": 2, "shard0": 1, "shard1": 1, "shard2": 0} or
          sorted(run["merge"]["cases_from"].values()) == [0, 1, 1, 2],
          f"run.json records where each case came from: {run['merge']['cases_from']}")
    check(stub.n_set_image == run["qc_overlays"] and run["qc_overlays"] > 0,
          f"merge re-scores nothing; only the {run['qc_overlays']} QC re-runs touch the model")
    check({p.parent.name for p in (out / "qc").rglob("*.png")} == {"amos_ct", "word_ct"},
          "merge writes QC overlays for every group")
    check(any(p.name.startswith("instances.bak-") for p in out.iterdir()),
          "merge backs up the base files before replacing them")

    # Resubmitting a shard after the merge: frozen assignment, nothing to redo.
    stub.n_set_image = 0
    check(eval_sam.main(argv(out, "--shard", "0", "--num-shards", str(N))) == 0 and
          stub.n_set_image == 0, "a resubmitted finished shard does no work")

    # Merging again: every shard case is now complete in base AND its shard -> accepted,
    # because the two copies agree.
    check(eval_sam.main(argv(out, "--merge", str(N))) == 0, "re-merge exits 0")
    run = json.load(open(out / "run.json"))
    check(run["merge"]["cases_scored_twice"] == 2 and
          run["merge"]["max_dice_diff_scored_twice"] < 1e-9,
          "cases scored twice are detected and agree exactly")
    check(table(out)[:2] == (ref_i, ref_v), "re-merge still identical to the single job")

    # ...but if two copies disagree, the merge refuses.
    sd = next(eval_sam.shard_dir(out, k, N) for k in range(N) if plans[k])
    si = rows(sd / "instances.csv")
    si[0]["dice"] = str(float(si[0]["dice"]) * 0.5)
    eval_sam._rewrite(sd / "instances.csv", eval_sam.INSTANCE_FIELDS, si)
    before = digest(out)
    check(eval_sam.main(argv(out, "--merge", str(N))) == 1 and digest(out) == before,
          "merge refuses when two copies of a case disagree, base untouched")

    # ---- Scenario B: fresh start, 4 shards, no base files; one shard interrupted mid-case.
    out = tmp / "arm1_fresh"
    N = 4
    for k in range(N):
        check(eval_sam.main(argv(out, "--shard", str(k), "--num-shards", str(N))) == 0,
              f"fresh shard {k}/{N} exits 0")
    check(not (out / "instances.csv").exists(), "fresh shards create no base file")
    sd = eval_sam.shard_dir(out, 1, N)
    si = rows(sd / "instances.csv")
    eval_sam._rewrite(sd / "instances.csv", eval_sam.INSTANCE_FIELDS, si[: len(si) // 2])
    check(eval_sam.main(argv(out, "--merge", str(N))) == 1, "merge refuses an interrupted shard")
    check(eval_sam.main(argv(out, "--shard", "1", "--num-shards", str(N))) == 0,
          "resubmitting the interrupted shard redoes its partial case")
    check(any(p.name.startswith("instances.bak-") for p in sd.iterdir()),
          "the shard backed up its own files before rewriting them")
    check(eval_sam.main(argv(out, "--merge", str(N))) == 0, "merge of 4 fresh shards exits 0")
    check(table(out) == (ref_i, ref_v, ref_ni, ref_nv),
          "4-shard output is row-for-row identical to the single job")

    # Aggregation only runs on certified results.
    try:
        aggregate_arm.main(["--arm-dir", str(eval_sam.shard_dir(out, 0, N)), "--n-boot", "50"])
        check(False, "aggregate must refuse a shard directory")
    except SystemExit as e:
        check("does not certify" in str(e), "aggregate_arm refuses a shard's partial results")
    try:
        aggregate_arm.main(["--arm-dir", str(tmp / "nowhere"), "--n-boot", "50"])
        check(False, "aggregate must refuse a directory without run.json")
    except SystemExit as e:
        check("not verified" in str(e), "aggregate_arm refuses results without run.json")
    check(aggregate_arm.main(["--arm-dir", str(out), "--n-boot", "50"]) == 0,
          "aggregate_arm runs on the merged, certified results")

    restore_anchors()
    print(f"scratch: {tmp}")


def test_sbatch_scripts() -> None:
    """The 4 independent shard scripts + the merge script (D29): in sync with their
    generator, each runs exactly its own shard, and each refuses to start next to a job
    that writes the same files. Executed for real under bash with stub squeue / python /
    nvidia-smi, so the shell logic itself is tested, not just its text."""
    print("\n--- SLURM scripts: 4 shards + merge ---")
    import subprocess
    here = Path(__file__).resolve().parent
    names = [f"arm1_shard{k}.sbatch" for k in range(4)] + ["arm1_merge.sbatch"]
    tmp = Path(tempfile.mkdtemp(prefix="arm1_sbatch_"))
    regen = tmp / "regen"
    regen.mkdir()
    subprocess.run([sys.executable, str(here / "gen_sbatch.py"), str(regen)], check=True,
                   capture_output=True)
    for n in names:
        check((here / n).read_text() == (regen / n).read_text(),
              f"{n} is exactly what gen_sbatch.py produces (no hand edits, no drift)")
        r = subprocess.run(["bash", "-n", str(here / n)], capture_output=True)
        check(r.returncode == 0, f"{n} passes bash -n")

    texts = {n: (here / n).read_text() for n in names}
    logs = [next(line for line in texts[n].splitlines() if line.startswith("#SBATCH --output"))
            for n in names]
    check(len(set(logs)) == len(logs), "every script logs to its own file")
    jobs = [next(line for line in texts[n].splitlines() if line.startswith("#SBATCH --job-name"))
            for n in names]
    check(len(set(jobs)) == len(jobs), f"every script has its own job name: {jobs}")

    # Run each script with the cluster paths redirected into tmp and stub binaries.
    seg = tmp / "Segmentation"
    (seg / "checkpoints").mkdir(parents=True)
    (seg / "checkpoints" / "sam_vit_h_4b8939.pth").write_bytes(b"x")
    stub_bin = tmp / "bin"
    stub_bin.mkdir()
    calls = tmp / "python_calls.txt"
    fake_py = stub_bin / "python"
    fake_py.write_text(f'#!/bin/bash\nif [ "$1" = "-c" ]; then exit 0; fi\n'
                       f'echo "$@" >> "{calls}"\nexit 0\n')
    (stub_bin / "nvidia-smi").write_text("#!/bin/bash\necho '0, NVIDIA A40, 46068 MiB'\n")
    (stub_bin / "squeue").write_text('#!/bin/bash\ncat "$FAKE_SQUEUE"\n')
    for f in stub_bin.iterdir():
        f.chmod(0o755)

    def run(name: str, queue: str) -> tuple[int, list[str]]:
        script = texts[name].replace("/home/rethinkingai-self/vihaan/miniconda3/envs/seg311/bin/python",
                                     str(fake_py))
        script = script.replace("/home/rethinkingai-self/vihaan/Segmentation", str(seg))
        sp = tmp / name
        sp.write_text(script)
        q = tmp / "queue.txt"
        q.write_text(queue)
        if calls.exists():
            calls.unlink()
        env = dict(os.environ, PATH=f"{stub_bin}:{os.environ['PATH']}", FAKE_SQUEUE=str(q),
                   SLURM_JOB_ID="900", SLURM_CPUS_ON_NODE="8")
        r = subprocess.run(["bash", str(sp)], env=env, capture_output=True, text=True)
        return r.returncode, (calls.read_text().splitlines() if calls.exists() else [])

    for k in range(4):
        rc, py = run(f"arm1_shard{k}.sbatch", "900 arm1_s%d RUNNING\n" % k)   # only itself
        evals = [c for c in py if "eval_sam.py" in c]
        check(rc == 0 and len(evals) == 1 and f"--shard {k} --num-shards 4" in evals[0]
              and f"--out {seg}/outputs/arm1" in evals[0],
              f"arm1_shard{k}.sbatch runs eval_sam.py --shard {k} --num-shards 4 on outputs/arm1")
        other = (k + 1) % 4
        rc, py = run(f"arm1_shard{k}.sbatch", f"901 arm1_s{other} RUNNING\n")
        check(rc == 0 and py, f"shard {k} runs alongside shard {other} (independent)")
        for clash in (f"902 arm1_s{k} PENDING", "351837_2 arm1_sam RUNNING", "903 arm1_merge PENDING"):
            rc, py = run(f"arm1_shard{k}.sbatch", clash + "\n")
            check(rc == 3 and not py, f"shard {k} refuses to start next to '{clash}'")

    rc, py = run("arm1_merge.sbatch", "")
    check(rc == 0 and any("--merge 4" in c for c in py) and any("aggregate_arm.py" in c for c in py),
          "arm1_merge.sbatch runs --merge 4, then aggregate_arm.py")
    for clash in ("904 arm1_s1 RUNNING", "905 arm1_s3 PENDING", "351837_0 arm1_sam RUNNING"):
        rc, py = run("arm1_merge.sbatch", clash + "\n")
        check(rc == 3 and not py, f"merge refuses to start while '{clash}'")


def test_python_compatibility() -> None:
    """Prajna's seg311 is Python 3.11; a dev machine may be newer. Parse every script
    against 3.11 so 3.12-only syntax (e.g. PEP 701 nested same-quote f-strings) is caught
    here rather than as a SyntaxError on the cluster."""
    print("\n--- python 3.11 compatibility ---")
    import ast
    bad = []
    for f in sorted(Path(__file__).resolve().parent.glob("*.py")):
        try:
            ast.parse(f.read_text(), filename=str(f), feature_version=(3, 11))
        except SyntaxError as e:
            bad.append(f"{f.name}:{e.lineno}: {e.msg}")
    check(not bad, f"all {len(list(Path(__file__).resolve().parent.glob('*.py')))} scripts "
                   f"parse under Python 3.11" + (f" -- {bad}" if bad else ""))


def main() -> int:
    test_python_compatibility()
    test_metrics()
    test_jitter()
    test_aggregation()
    test_end_to_end()
    test_sharding()
    test_sbatch_scripts()
    print("\nALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
