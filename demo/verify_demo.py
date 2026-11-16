"""DEMO_PLAN §5 checks 1-4 on the real samples: the app reproduces the reported evaluation.

  1 boxes   every kidney instance of every sample: the app's clean and jittered boxes (all 4
            noise levels) equal box_clean / box_jit of the stored instances.csv, and no
            instance is missing or extra
  2 masks   per instance, the ground-truth area is identical and the predicted mask is within
            PIX_TOL pixels of the stored one (|pred area diff| + |intersection diff|, recovered
            from the stored Dice): plain student (arm 3 s644) on every sample; box student
            (arm 3 box) and SAM (arm 2) at 10% noise on one CT and one CHAOS case.
            Why pixels, not a Dice tolerance: the evaluation ran on an A40 (CUDA, TF32 convs by
            default), the demo on the Mac; a logit sitting on the 0.5 boundary flips one pixel,
            which on a 100-px kidney moves Dice by ~0.01 (D65).
  3 export  the saved NIfTI, reloaded through prep_data's own loader, gives back the
            predicted labels exactly (orientation and label numbers survive the round trip)
  4 speed   student time per slice on this machine

Run:  python verify_demo.py            (about 2 minutes on an M-series Mac)
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

import demo_core as D

PIX_TOL = 2
SAM_CASES = ("amos_0167", "chaos_39_t1in")


def stored(fname: str) -> dict:
    """case -> {(z, side, jitter): row} from a stored instances.csv."""
    out = defaultdict(dict)
    with open(D.DEMO_DIR / "results" / fname) as f:
        for r in csv.DictReader(f):
            out[r["case"]][(int(r["z"]), r["side"], round(float(r["jitter"]), 4))] = r
    return out


def check_boxes(case: D.Case, ref: dict) -> list:
    bad, seen = [], set()
    for z in range(case.img.shape[0]):
        for lev in D.NOISE_LEVELS:
            jit = D.eval_boxes(case, z, lev)
            for side, name in D.SIDES:
                clean = case.boxes[z][side]
                if clean is None:
                    continue
                key = (z, name, round(lev, 4))
                seen.add(key)
                r = ref.get(key)
                if r is None:
                    bad.append(f"{case.name} {key}: instance not in instances.csv")
                elif json.loads(r["box_clean"]) != list(clean) or json.loads(r["box_jit"]) != list(jit[side]):
                    bad.append(f"{case.name} {key}: box {clean}/{jit[side]} vs {r['box_clean']}/{r['box_jit']}")
    for key in set(ref) - seen:
        bad.append(f"{case.name} {key}: stored instance the app does not produce")
    return bad


def check_scores(case: D.Case, pred: D.Prediction, ref: dict, level: float) -> tuple[list, dict]:
    s = D.score(pred, case)
    bad, st = [], {"n": 0, "exact": 0, "max_pix": 0.0, "max_dice": 0.0}
    for row in s["instances"]:
        r = ref.get((row["z"], row["side"], round(level, 4)))
        if r is None:
            bad.append(f"{case.name} z={row['z']} {row['side']}: not stored")
            continue
        g, p_ref = int(r["gt_area"]), int(r["pred_area"])
        if g != row["gt_area"]:
            bad.append(f"{case.name} z={row['z']} {row['side']}: GT area {row['gt_area']} vs {g}")
            continue
        pix = (abs(p_ref - row["pred_area"])
               + abs(float(r["dice"]) * (g + p_ref) - row["dice"] * (g + row["pred_area"])) / 2)
        st["n"] += 1
        st["exact"] += pix < 0.5
        st["max_pix"] = max(st["max_pix"], pix)
        st["max_dice"] = max(st["max_dice"], abs(row["dice"] - float(r["dice"])))
        if pix > PIX_TOL + 0.01:
            bad.append(f"{case.name} z={row['z']} {row['side']}: {pix:.1f} px differ "
                       f"(Dice {row['dice']:.6f} vs {r['dice']})")
    return bad, st


def merge(a: dict, b: dict) -> dict:
    return {"n": a["n"] + b["n"], "exact": a["exact"] + b["exact"],
            "max_pix": max(a["max_pix"], b["max_pix"]), "max_dice": max(a["max_dice"], b["max_dice"])}


def check_export(case: D.Case, pred: D.Prediction) -> list:
    import prep_data as P
    with tempfile.TemporaryDirectory() as td:
        p = D.save_nifti(pred.labels(), case, Path(td) / "pred.nii.gz")
        # reload as a "label" through the evaluation loader, with labels 1/2 as right/left
        P.NIFTI_LABELS["_roundtrip"] = {"right": 1, "left": 2, "liver": 99}
        try:
            _, kid, _, _, _ = P.load_nifti({"image": str(p), "label": str(p), "dataset": "_roundtrip"})
        finally:
            del P.NIFTI_LABELS["_roundtrip"]
    back = np.stack([P.make_slice(np.zeros(kid.shape, np.uint8), kid, z, case.out_hw)[1]
                     for z in range(kid.shape[2])])
    return [] if np.array_equal(back, pred.labels()) else [f"{case.name}: exported mask does not round-trip"]


def main() -> int:
    models = D.model_catalogue()
    refs = {"arm3_s644": stored("arm3_plain_s644.instances.csv"),
            "arm3box_s643": stored("arm3_box_s643.instances.csv"),
            "arm2": stored("arm2_r4.instances.csv")}
    fails, n_inst = [], 0
    tot = {"n": 0, "exact": 0, "max_pix": 0.0, "max_dice": 0.0}
    for label, entry in D.sample_catalogue().items():
        case = D.load_sample(entry)
        ref = refs["arm3_s644"][case.name]
        if not ref:
            fails.append(f"{case.name}: no stored rows (is it a test case?)")
            continue
        b = check_boxes(case, ref)
        fails += b
        n_inst += case.n_instances()
        pred = D.predict(models["arm3_s644"], case)
        sb, st = check_scores(case, pred, ref, 0.0)
        fails += sb
        tot = merge(tot, st)
        line = (f"{case.name:16s} {case.n_instances():4d} kidneys  boxes {'OK' if not b else 'FAIL'}  "
                f"arm3 masks {'OK' if not sb else 'FAIL'}  ({pred.seconds / pred.n_forward * 1000:.0f} ms/slice)")
        if case.name == SAM_CASES[0]:
            fails += check_export(case, pred)
        if case.name in SAM_CASES:
            for key in ("arm3box_s643", "arm2"):
                p2 = D.predict(models[key], case, 0.10)
                s2, st2 = check_scores(case, p2, refs[key][case.name], 0.10)
                fails += s2
                tot = merge(tot, st2)
                line += f"  {key}@10% {'OK' if not s2 else 'FAIL'} ({p2.seconds:.0f} s)"
        print(line, flush=True)
    print(f"\nscored masks: {tot['n']}, pixel-identical to the evaluation: {tot['exact']}, "
          f"largest difference {tot['max_pix']:.0f} px (tolerance {PIX_TOL}), "
          f"largest Dice difference {tot['max_dice']:.4f}")
    for f in fails[:30]:
        print("FAIL", f)
    print("ALL CHECKS PASSED" if not fails else f"{len(fails)} FAILURES", f"({n_inst} kidney instances)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
