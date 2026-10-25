#!/usr/bin/env python
"""What makes CHAOS T1 different? Image statistics per dataset (diagnostic only, no model).

    python domain_stats.py --processed .../data/processed --out .../outputs/domain_stats \
        --arms .../outputs/arm1_vit_b .../outputs/arm2_r4 .../outputs/arm3_plain_s643 .../outputs/arm3_box_s643

Groups: AMOS CT / MRI train (what the models learned from), AMOS CT / MRI test, WORD, CHAOS
t2 / t1in / t1out. Per case, up to --per-case kidney slices evenly spaced in z; per kidney
(the stored 0-255 image, the stored mask):
  k_mean, ring_mean    kidney interior vs a 1-5 mm ring outside it (other kidney excluded)
  contrast_d           (k_mean - ring_mean) / pooled SD; sign: + = kidney brighter than its ring
  k_pct_body           share of body pixels darker than the kidney median (1 = brightest organ)
  ring_pct_body        same for the ring (is the surrounding tissue / fat bright?)
  k_cv                 kidney interior SD / mean (texture, cortex-medulla)
  noise                Immerkaer noise SD over flat body pixels; cnr = |k - ring| / noise
  edge_mm              |k - ring| / mean boundary gradient (mm per unit): larger = blurrier edge
  grad_ratio           boundary gradient / median body gradient: does the kidney edge stand out?
  area_mm2, net_px     kidney area, and its area in the student's 512 letterbox
Geometry per case: h x w, in-plane spacing, slice thickness, field of view.
Each case's kidneys are averaged first, then the median [IQR] over cases.

--arms: per group at 10 % jitter, missed kidneys (Dice < 0.1), empty predictions, median
pred/gt area, and (CHAOS) the Spearman correlation across patients between each arm's Dice and
each statistic. Montages: one mid-kidney slice per case for 4 cases per group (CHAOS rows use
the same 4 patients) -- full slice with the GT outline, and a 100 mm crop around the right
kidney without it (to judge edges and blur).
Ends STATS_OK / STATS_FAILED.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import ndimage

GROUPS = (("amos/ct/[train]", "AMOS CT train"), ("amos/mri/[train]", "AMOS MRI train"),
          ("amos/ct/[test]", "AMOS CT test"), ("amos/mri/[test]", "AMOS MRI test"),
          ("word/ct/", "WORD CT"), ("chaos/mri/t2", "CHAOS T2"),
          ("chaos/mri/t1in", "CHAOS T1-in"), ("chaos/mri/t1out", "CHAOS T1-out"))
STATS = ("k_mean", "ring_mean", "contrast_d", "k_pct_body", "ring_pct_body", "k_cv", "noise", "cnr",
         "edge_mm", "grad_ratio", "area_mm2", "net_px")
GEOM = ("h", "w", "sp_inplane", "sp_slice", "fov_mm")
MIN_AREA = 30            # px: kidneys smaller than this give no stable interior / ring
RING_MM = (1.0, 5.0)
NET = 512
JITTER = 0.10


def group_of(r: dict) -> str | None:
    if r["dataset"] == "amos":
        return f"amos/{r['modality']}/[{r['split']}]" if r["split"] in ("train", "test") else None
    return f"{r['dataset']}/{r['modality']}/{r['sequence']}"


def body_mask(img: np.ndarray) -> np.ndarray:
    m = ndimage.binary_fill_holes(img > 12)
    lab, n = ndimage.label(m)
    if n == 0:
        return m
    return lab == (1 + int(np.argmax(ndimage.sum(m, lab, range(1, n + 1)))))


def noise_sd(img: np.ndarray, flat: np.ndarray) -> float:
    """Immerkaer (1996) fast noise estimate, restricted to flat body pixels."""
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float64)
    conv = np.abs(ndimage.convolve(img, k, mode="reflect"))
    return float(math.sqrt(math.pi / 2) * conv[flat].mean() / 6) if flat.any() else float("nan")


def slice_stats(img8: np.ndarray, msk: np.ndarray, sp: float) -> list[dict]:
    img = img8.astype(np.float64)
    body = body_mask(img8)
    grad = np.hypot(ndimage.sobel(img, 0), ndimage.sobel(img, 1)) / 8.0     # units per pixel
    body_in = ndimage.binary_erosion(body, iterations=3)
    gb = grad[body_in] if body_in.any() else grad[body]
    flat = body_in & (grad <= np.median(gb))
    sd_noise = noise_sd(img, flat)
    body_vals = np.sort(img[body])
    g_med = float(np.median(gb))
    r_in, r_out = (max(1, int(round(r / sp))) for r in RING_MM)
    scale = NET / max(img.shape)
    out = []
    for side in (1, 2):
        k = msk == side
        area = int(k.sum())
        if area < MIN_AREA:
            continue
        inner = ndimage.binary_erosion(k, iterations=1)
        inner = inner if inner.sum() >= 10 else k
        ring = (ndimage.binary_dilation(k, iterations=r_out) & ~ndimage.binary_dilation(k, iterations=r_in)
                & (msk == 0) & body)
        edge = k & ~ndimage.binary_erosion(k)
        if ring.sum() < 10:
            continue
        kv, rv = img[inner], img[ring]
        km, rm = float(kv.mean()), float(rv.mean())
        pooled = math.sqrt((kv.var() + rv.var()) / 2) or float("nan")
        g_edge = float(grad[edge].mean())
        out.append({
            "k_mean": km, "ring_mean": rm, "contrast_d": (km - rm) / pooled,
            "k_pct_body": float(np.searchsorted(body_vals, np.median(kv)) / len(body_vals)),
            "ring_pct_body": float(np.searchsorted(body_vals, np.median(rv)) / len(body_vals)),
            "k_cv": float(kv.std() / km) if km > 0 else float("nan"),
            "noise": sd_noise, "cnr": abs(km - rm) / sd_noise if sd_noise > 0 else float("nan"),
            "edge_mm": abs(km - rm) / g_edge * sp if g_edge > 0 else float("nan"),
            "grad_ratio": g_edge / g_med if g_med > 0 else float("nan"),
            "area_mm2": area * sp * sp, "net_px": area * scale * scale,
        })
    return out


def pick(seq: list, n: int) -> list:
    """n items evenly spaced (deterministic), all of them if there are fewer."""
    if len(seq) <= n:
        return list(seq)
    return [seq[round(i * (len(seq) - 1) / (n - 1))] for i in range(n)] if n > 1 else [seq[len(seq) // 2]]


def nanmed(v):
    v = [x for x in v if x == x]
    return statistics.median(v) if v else float("nan")


def q(v, p):
    v = [x for x in v if x == x]
    return float(np.percentile(v, p)) if v else float("nan")


def case_table(manifest: list[dict], processed: Path, per_case: int) -> dict:
    by_case = defaultdict(list)
    for r in manifest:
        g = group_of(r)
        if g is not None:
            by_case[(g, r["case"])].append(r)
    cases = {}
    for (g, case), rows in sorted(by_case.items()):
        rows.sort(key=lambda r: int(r["z"]))
        r0 = rows[0]
        sp = (float(r0["sp_row"]) + float(r0["sp_col"])) / 2
        geom = {"h": int(r0["h"]), "w": int(r0["w"]), "sp_inplane": sp, "sp_slice": float(r0["sp_slice"]),
                "fov_mm": max(int(r0["h"]) * float(r0["sp_row"]), int(r0["w"]) * float(r0["sp_col"]))}
        kid = [r for r in rows if int(r["rk_area"]) >= MIN_AREA or int(r["lk_area"]) >= MIN_AREA]
        ks = []
        for r in pick(kid, per_case):
            with np.load(processed / r["path"]) as d:
                ks += slice_stats(d["img"], d["msk"], sp)
        if not ks:
            continue
        cases[(g, case)] = {"group": g, "case": case, "patient": r0["patient"], "n_kidneys": len(ks), **geom,
                            **{s: float(np.nanmean([k[s] for k in ks])) for s in STATS}}
    return cases


def arm_table(arm: Path) -> tuple[dict, dict]:
    """{group: summary} and {(group, case): patient-mean Dice} at the headline jitter."""
    per = defaultdict(list)
    with open(arm / "instances.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            if abs(float(r["jitter"]) - JITTER) < 1e-9:
                g = group_of(r)
                if g is not None:
                    per[g].append(r)
    summ, case_dice = {}, {}
    for g, rows in per.items():
        pm = defaultdict(list)
        for r in rows:
            pm[r["case"]].append(float(r["dice"]))
        for c, v in pm.items():
            case_dice[(g, c)] = sum(v) / len(v)
        ratio = [float(r["pred_area"]) / float(r["gt_area"]) for r in rows if float(r["gt_area"]) > 0]
        summ[g] = {"dice": statistics.mean(case_dice[(g, c)] for c in pm),
                   "missed": sum(float(r["dice"]) < 0.1 for r in rows) / len(rows),
                   "empty": sum(float(r["pred_area"]) == 0 for r in rows) / len(rows),
                   "area_ratio": nanmed(ratio)}
    return summ, case_dice


def montage(manifest: list[dict], processed: Path, cases: dict, path_full: Path, path_zoom: Path) -> None:
    from PIL import Image, ImageDraw
    tile, crop_mm = 256, 100.0
    rows_by_case = defaultdict(list)
    for r in manifest:
        g = group_of(r)
        if g is not None:
            rows_by_case[(g, r["case"])].append(r)
    chaos_pats = pick(sorted({c["patient"] for c in cases.values() if c["group"].startswith("chaos/")}), 4)
    full = Image.new("RGB", (4 * tile + 120, len(GROUPS) * tile), (0, 0, 0))
    zoom = Image.new("RGB", (4 * tile + 120, len(GROUPS) * tile), (0, 0, 0))
    for gi, (g, name) in enumerate(GROUPS):
        mine = sorted(k for k in cases if k[0] == g)
        sel = ([k for k in mine if cases[k]["patient"] in chaos_pats] if g.startswith("chaos/") else pick(mine, 4))
        for ci, key in enumerate(sel[:4]):
            rows = rows_by_case[key]
            r = max(rows, key=lambda r: int(r["rk_area"]) + int(r["lk_area"]))
            with np.load(processed / r["path"]) as d:
                img, msk = d["img"], d["msk"]
            rgb = np.repeat(img[..., None], 3, axis=2).copy()
            edge = (msk > 0) & ~ndimage.binary_erosion(msk > 0)
            rgb[edge] = (60, 230, 60)
            for canvas, arr, crop in ((full, rgb, False), (zoom, img, True)):
                if crop:
                    sp = float(r["sp_row"])
                    ys, xs = np.nonzero(msk == 1) if (msk == 1).any() else np.nonzero(msk > 0)
                    cy, cx, half = int(ys.mean()), int(xs.mean()), int(round(crop_mm / 2 / sp))
                    a = np.pad(arr, half, mode="constant")[cy:cy + 2 * half, cx:cx + 2 * half]
                    im = Image.fromarray(a).convert("RGB").resize((tile, tile), Image.BILINEAR)
                else:
                    im = Image.fromarray(arr)
                    s = tile / max(im.size)
                    im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.BILINEAR)
                canvas.paste(im, (120 + ci * tile, gi * tile))
                ImageDraw.Draw(canvas).text((124 + ci * tile, gi * tile + 4), f"{key[1]} z{r['z']}", fill=(255, 230, 0))
        for canvas in (full, zoom):
            ImageDraw.Draw(canvas).text((6, gi * tile + tile // 2), name, fill=(255, 255, 255))
    full.save(path_full)
    zoom.save(path_zoom)


def fmt(v, nd=2):
    return "nan" if v != v else f"{v:.{nd}f}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--processed", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--arms", nargs="*", type=Path, default=[])
    p.add_argument("--per-case", type=int, default=8)
    a = p.parse_args(argv)
    with open(a.processed / "manifest.csv", newline="") as fh:
        manifest = list(csv.DictReader(fh))
    cases = case_table(manifest, a.processed, a.per_case)
    missing = [g for g, _ in GROUPS if not any(c["group"] == g for c in cases.values())]
    if missing:
        print(f"!! no cases for {missing}\nSTATS_FAILED")
        return 1
    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.out / "domain_cases.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(next(iter(cases.values())).keys()))
        w.writeheader()
        w.writerows(cases.values())
    L = ["# Image statistics per dataset (median [IQR] over cases)", "",
         "| statistic | " + " | ".join(n for _, n in GROUPS) + " |", "|---|" + "---|" * len(GROUPS)]
    summary = []
    for s in ("n_cases", "n_kidneys") + GEOM + STATS:
        cells = []
        for g, name in GROUPS:
            cs = [c for c in cases.values() if c["group"] == g]
            if s == "n_cases":
                cells.append(str(len(cs)))
                continue
            if s == "n_kidneys":
                cells.append(str(sum(c["n_kidneys"] for c in cs)))
                continue
            v = [c[s] for c in cs]
            m, lo, hi = nanmed(v), q(v, 25), q(v, 75)
            summary.append({"group": name, "statistic": s, "median": m, "p25": lo, "p75": hi})
            nd = 0 if s in ("h", "w", "fov_mm", "area_mm2", "net_px") else 2
            cells.append(f"{fmt(m, nd)} [{fmt(lo, nd)}, {fmt(hi, nd)}]")
        L.append(f"| {s} | " + " | ".join(cells) + " |")
    with open(a.out / "domain_stats.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["group", "statistic", "median", "p25", "p75"])
        w.writeheader()
        w.writerows(summary)
    for arm in a.arms:
        if not (arm / "instances.csv").is_file():
            L += ["", f"({arm.name}: instances.csv not found, skipped)"]
            continue
        summ, case_dice = arm_table(arm)
        L += ["", f"## {arm.name} at {JITTER:.0%} jitter", "",
              "| | " + " | ".join(n for _, n in GROUPS) + " |", "|---|" + "---|" * len(GROUPS)]
        for k, lab in (("dice", "Dice 2D (patient mean)"), ("missed", "kidneys with Dice < 0.1"),
                       ("empty", "empty predictions"), ("area_ratio", "median pred / gt area")):
            L.append(f"| {lab} | " + " | ".join(fmt(summ[g][k], 3) if g in summ else "-" for g, _ in GROUPS) + " |")
        from scipy.stats import spearmanr
        L += ["", "Spearman rho across patients, Dice vs statistic (CHAOS):", "",
              "| statistic | CHAOS T2 | CHAOS T1-in | CHAOS T1-out |", "|---|---|---|---|"]
        for s in STATS:
            cells = []
            for g in ("chaos/mri/t2", "chaos/mri/t1in", "chaos/mri/t1out"):
                ks = [k for k in cases if k[0] == g and k in case_dice]
                x, y = [cases[k][s] for k in ks], [case_dice[k] for k in ks]
                cells.append(fmt(float(spearmanr(x, y)[0]), 2) if len(ks) >= 5 else "-")
            L.append(f"| {s} | " + " | ".join(cells) + " |")
    montage(manifest, a.processed, cases, a.out / "montage_full.png", a.out / "montage_zoom.png")
    (a.out / "domain_stats.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {a.out}\nSTATS_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
