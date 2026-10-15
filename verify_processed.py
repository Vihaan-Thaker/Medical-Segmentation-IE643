#!/usr/bin/env python
"""Independent check of a finished prep_data.py output.

Reads EVERY stored slice back from disk and checks it against manifest.csv. Checks the
split counts against the plan written in DECISIONS.md (D14, D15, D18), NOT against
prep_data.py's own logic, so a bug there cannot also hide here.

    python verify_processed.py --out .../data/processed      # ends VERIFY_OK / VERIFY_FAILED
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# (dataset, split, modality) -> volumes. DECISIONS.md D14 + D15 + D18.
EXPECTED_VOLUMES = {
    ("amos", "train", "ct"): 177, ("amos", "val", "ct"): 59, ("amos", "test", "ct"): 59,
    ("amos", "train", "mri"): 36, ("amos", "val", "mri"): 12, ("amos", "test", "mri"): 12,
    ("word", "external", "ct"): 100,
    ("chaos", "external", "mri"): 60,          # 20 patients x {t1in, t1out, t2}
}
EXCLUDED = {"amos_0085", "amos_0320", "amos_0112", "amos_0142", "amos_0323"}   # D18
FULL_VOLUME_SPLITS = {"test", "external"}
SQUARE_TOL = 0.05                                                              # D19


def tight_box(mask: np.ndarray):
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return None
    return [int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True, help="prep_data.py --out directory")
    p.add_argument("--min-area", type=int, default=50, help="must match prep_data.py --min-area")
    p.add_argument("--skip-count-check", action="store_true", help="for --limit / synthetic runs")
    args = p.parse_args(argv)

    problems: list[str] = []

    def bad(msg: str) -> None:
        problems.append(msg)

    for name in ("manifest.csv", "volumes.csv", "run.json"):
        if not (args.out / name).exists():
            print(f"missing {args.out / name}\nVERIFY_FAILED")
            return 1
    with open(args.out / "manifest.csv") as fh:
        rows = list(csv.DictReader(fh))
    with open(args.out / "volumes.csv") as fh:
        vols = list(csv.DictReader(fh))
    run = json.load(open(args.out / "run.json"))

    # ---- volume level
    for v in vols:
        if v["error"]:
            bad(f"volume error {v['case']}: {v['error']}")
        for c in ("q_modality", "q_liver_right", "q_kidney_lr", "q_liver_above", "q_kidney_post"):
            if v[c] == "fail":
                bad(f"QC fail {v['case']}: {c}")
    if run.get("problems"):
        bad(f"prep_data.py itself reported {len(run['problems'])} problem(s)")

    counts = Counter((v["dataset"], v["split"], v["modality"]) for v in vols if not v["error"])
    if not args.skip_count_check:
        for key in sorted(set(counts) | set(EXPECTED_VOLUMES)):
            if counts.get(key, 0) != EXPECTED_VOLUMES.get(key, 0):
                bad(f"count {'/'.join(key)}: {counts.get(key, 0)} volumes, "
                    f"expected {EXPECTED_VOLUMES.get(key, 0)}")
    present_excluded = EXCLUDED & {v["case"] for v in vols}
    if present_excluded:
        bad(f"excluded cases present: {sorted(present_excluded)}")

    split_of: dict = {}
    for v in vols:
        split_of.setdefault(v["patient"], set()).add(v["split"])
    leaks = sorted(pt for pt, s in split_of.items() if len(s) > 1)
    if leaks:
        bad(f"patient in two splits: {leaks[:10]}")

    # ---- slice level: read every file back
    per_case: dict = {}
    for i, r in enumerate(rows, 1):
        per_case.setdefault(r["case"], []).append(int(r["z"]))
        f = args.out / r["path"]
        if not f.exists():
            bad(f"missing file {r['path']}")
            continue
        try:
            with np.load(f) as d:
                img, msk = d["img"], d["msk"]
        except Exception as exc:
            bad(f"unreadable {r['path']}: {exc}")
            continue
        h, w = int(r["h"]), int(r["w"])
        if img.shape != (h, w) or msk.shape != (h, w):
            bad(f"{r['path']}: shape img {img.shape} msk {msk.shape}, manifest {(h, w)}")
            continue
        if img.dtype != np.uint8 or msk.dtype != np.uint8:
            bad(f"{r['path']}: dtype img {img.dtype} msk {msk.dtype}")
        extra = set(np.unique(msk).tolist()) - {0, 1, 2}
        if extra:
            bad(f"{r['path']}: mask values {sorted(extra)} outside 0/1/2")
        rk, lk = int((msk == 1).sum()), int((msk == 2).sum())
        if (rk, lk) != (int(r["rk_area"]), int(r["lk_area"])):
            bad(f"{r['path']}: areas on disk {(rk, lk)} vs manifest {(r['rk_area'], r['lk_area'])}")
        if int(rk + lk > 0) != int(r["has_kidney"]):
            bad(f"{r['path']}: has_kidney mismatch")
        boxes = []
        for code, area, col in ((1, rk, "rk_box"), (2, lk, "lk_box")):
            box = json.loads(r[col])
            want = tight_box(msk == code) if area >= args.min_area else None
            if box != want:
                bad(f"{r['path']}: {col} {box} should be {want}")
            boxes.append(box)
        if int(any(b is not None for b in boxes)) != int(r["sam_eligible"]):
            bad(f"{r['path']}: sam_eligible mismatch")
        if abs(float(r["sp_row"]) / float(r["sp_col"]) - 1.0) > SQUARE_TOL + 1e-6:
            bad(f"{r['path']}: non-square pixels {r['sp_row']} x {r['sp_col']} mm")
        if len(problems) > 200:
            break
        if i % 10000 == 0:
            print(f"  read {i}/{len(rows)} slices", flush=True)

    # ---- every volume's slices are present, contiguous, and full where they must be
    for v in vols:
        if v["error"]:
            continue
        zs = sorted(per_case.get(v["case"], []))
        if len(zs) != int(v["n_kept"]):
            bad(f"{v['case']}: {len(zs)} slices in manifest, volumes.csv says {v['n_kept']}")
        elif zs and zs != list(range(zs[0], zs[-1] + 1)):
            bad(f"{v['case']}: slice indices not contiguous")
        if v["split"] in FULL_VOLUME_SPLITS and v["n_kept"] != v["n_slices"]:
            bad(f"{v['case']}: {v['split']} must be a full volume ({v['n_kept']}/{v['n_slices']})")

    print(f"\nchecked {len(rows)} slices from {len(vols)} volumes")
    for key in sorted(counts):
        print(f"  {'/'.join(key):<26}{counts[key]:>5} volumes")
    resampled = [v["case"] for v in vols if v.get("resampled")]
    print(f"  square-pixel resampled volumes: {len(resampled)}")
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for msg in problems[:50]:
            print(f"  !! {msg}")
        print("\nVERIFY_FAILED")
        return 1
    print("\nevery slice readable and consistent with manifest; counts match DECISIONS.md\nVERIFY_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
