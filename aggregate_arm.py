#!/usr/bin/env python
"""Turn an arm's per-instance scores into the reported tables (DECISIONS.md D22/D24a).

No GPU and no model: it reads only `instances.csv` / `volumes_3d.csv`, so re-plotting,
changing the CI method or switching the headline tolerance is free and never requires
re-running the evaluation pass.

Aggregation is per-patient first, then across patients, with a percentile bootstrap CI
over patients (D22). A flat mean over instances would let one AMOS MRI patient drive 19.5%
of that set's result (D22a) and would imply far more independent samples than we have.

Writes summary.csv, breakdowns.csv and summary.md; prints the headline table.

    python aggregate_arm.py --arm-dir .../outputs/arm1 --label "Arm 1: off-the-shelf SAM"
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sam_metrics as M  # noqa: E402

GROUP = ("dataset", "modality", "sequence")


def read_csv(path: Path, floats: tuple[str, ...], ints: tuple[str, ...] = ()) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in floats:
            if k in r and r[k] != "":
                r[k] = float(r[k])
        for k in ints:
            if k in r and r[k] != "":
                r[k] = int(r[k])
    return rows


def group_label(r: dict) -> str:
    return "/".join(x for x in (r["dataset"], r["modality"], r.get("sequence", "")) if x)


def summarise(rows: list[dict], metrics: tuple[str, ...], n_boot: int, seed: int) -> list[dict]:
    """One row per (group x jitter x metric), aggregated per patient then across."""
    out = []
    by: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by[(group_label(r), r["jitter"])].append(r)
    for (g, j), rs in sorted(by.items()):
        for m in metrics:
            if m not in rs[0]:
                continue
            a = M.aggregate_by_patient(rs, m, n_boot=n_boot, seed=seed)
            pm = a["patient_means"]
            out.append({
                "group": g, "jitter": j, "metric": m,
                "n_patients": a["n_patients"], "n_instances": a["n_instances"],
                "mean": round(a["mean"], 4), "ci_lo": round(a["ci_lo"], 4),
                "ci_hi": round(a["ci_hi"], 4),
                "patient_min": round(min(pm), 4) if pm else float("nan"),
                "patient_median": round(float(np.median(pm)), 4) if pm else float("nan"),
                "patient_max": round(max(pm), 4) if pm else float("nan"),
            })
    return out


def breakdowns(inst: list[dict], headline_metric: str, n_boot: int, seed: int) -> list[dict]:
    """Cheap extra cuts from the same rows (D24a): per side, per GT size quartile, and
    complete-failure counts. All at the headline jitter only."""
    out = []
    rows = [r for r in inst if r["jitter"] == M.HEADLINE_JITTER]
    if not rows:
        return out

    for side in ("right", "left"):
        sel = [r for r in rows if r["side"] == side]
        if sel:
            a = M.aggregate_by_patient(sel, headline_metric, n_boot=n_boot, seed=seed)
            out.append({"kind": "side", "key": side, "group": "ALL",
                        "n_patients": a["n_patients"], "n_instances": a["n_instances"],
                        "mean": round(a["mean"], 4), "ci_lo": round(a["ci_lo"], 4),
                        "ci_hi": round(a["ci_hi"], 4)})

    areas = np.asarray([r["gt_area"] for r in rows], float)
    qs = np.percentile(areas, [25, 50, 75])
    for i, (lo, hi) in enumerate(zip([-np.inf, *qs], [*qs, np.inf])):
        sel = [r for r in rows if lo < r["gt_area"] <= hi]
        if sel:
            a = M.aggregate_by_patient(sel, headline_metric, n_boot=n_boot, seed=seed)
            out.append({"kind": "gt_area_quartile", "key": f"Q{i + 1} ({int(max(lo, 0))}-"
                                                          f"{int(hi) if math.isfinite(hi) else 'max'} px)",
                        "group": "ALL", "n_patients": a["n_patients"],
                        "n_instances": a["n_instances"], "mean": round(a["mean"], 4),
                        "ci_lo": round(a["ci_lo"], 4), "ci_hi": round(a["ci_hi"], 4)})

    by_g: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_g[group_label(r)].append(r)
    for g, rs in sorted(by_g.items()):
        zero = [r for r in rs if r["dice"] == 0.0]
        out.append({"kind": "zero_dice", "key": g, "group": g, "n_patients": len({r["patient"] for r in zero}),
                    "n_instances": len(zero), "mean": round(len(zero) / len(rs), 4),
                    "ci_lo": "", "ci_hi": ""})
        d = np.asarray([r["dice"] for r in rs])
        s = np.asarray([r["sam_iou"] for r in rs])
        corr = float(np.corrcoef(d, s)[0, 1]) if d.size > 2 and d.std() > 0 and s.std() > 0 else float("nan")
        out.append({"kind": "sam_iou_vs_dice_corr", "key": g, "group": g,
                    "n_patients": len({r["patient"] for r in rs}), "n_instances": len(rs),
                    "mean": round(corr, 4) if math.isfinite(corr) else "", "ci_lo": "", "ci_hi": ""})
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--arm-dir", type=Path, required=True)
    p.add_argument("--label", default="arm")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=643)
    p.add_argument("--allow-unverified", action="store_true",
                   help="summarise even if run.json does not certify the results")
    args = p.parse_args(argv)

    # Only summarise results the evaluation itself certified: a shard holds part of the
    # data, and a failed or interrupted run may hold partial cases.
    rj_path = args.arm_dir / "run.json"
    if not args.allow_unverified:
        if not rj_path.exists():
            raise SystemExit(f"{rj_path} missing: results not verified (--allow-unverified to override)")
        rj = json.loads(rj_path.read_text())
        if rj.get("problems") or str(rj.get("mode", "")).startswith("shard"):
            raise SystemExit(f"{rj_path} does not certify complete results "
                             f"(mode={rj.get('mode')}, problems={rj.get('problems')}); "
                             f"--allow-unverified to override")

    nsd_cols = tuple(f"nsd_{t:g}" for t in M.TOLERANCES_MM)
    inst = read_csv(args.arm_dir / "instances.csv",
                    floats=("jitter", "dice", "sam_iou", "box_iou", *nsd_cols),
                    ints=("z", "gt_area", "pred_area"))
    if not inst:
        raise SystemExit(f"no instances.csv in {args.arm_dir}")
    nsd3_cols = tuple(f"nsd3d_{t:g}" for t in M.TOLERANCES_MM)
    vols = read_csv(args.arm_dir / "volumes_3d.csv",
                    floats=("jitter", "dice3d", "sliver_frac", *nsd3_cols),
                    ints=("gt_vox", "pred_vox", "sliver_vox"))

    headline_nsd = f"nsd_{M.HEADLINE_TOLERANCE_MM:g}"
    summary = summarise(inst, ("dice", *nsd_cols), args.n_boot, args.seed)
    for r in summary:
        r["level"] = "2d"
    if vols:
        v = summarise(vols, ("dice3d", *nsd3_cols), args.n_boot, args.seed)
        for r in v:
            r["level"] = "3d"
        summary += v
    write_csv(args.arm_dir / "summary.csv", summary)

    bd = breakdowns(inst, "dice", args.n_boot, args.seed)
    write_csv(args.arm_dir / "breakdowns.csv", bd)

    # ---- headline table
    def pick(level, metric, jitter):
        return {r["group"]: r for r in summary
                if r["level"] == level and r["metric"] == metric and r["jitter"] == jitter}
    d2 = pick("2d", "dice", M.HEADLINE_JITTER)
    n2 = pick("2d", headline_nsd, M.HEADLINE_JITTER)
    d3 = pick("3d", "dice3d", M.HEADLINE_JITTER)

    lines = [f"# {args.label}", "",
             f"Headline: jitter {M.HEADLINE_JITTER:.0%} (SAM's own recipe), "
             f"NSD tolerance {M.HEADLINE_TOLERANCE_MM:g} mm. "
             f"Mean over patients, 95% bootstrap CI ({args.n_boot} resamples).", "",
             "| group | patients | instances | Dice 2D | "
             f"NSD@{M.HEADLINE_TOLERANCE_MM:g}mm 2D | Dice 3D |",
             "|---|---|---|---|---|---|"]
    print(f"\n{args.label} — jitter {M.HEADLINE_JITTER:.0%}, NSD@{M.HEADLINE_TOLERANCE_MM:g}mm")
    print(f"{'group':<24}{'pts':>5}{'inst':>7}{'Dice 2D (95% CI)':>26}"
          f"{'NSD 2D (95% CI)':>26}{'Dice 3D':>10}")
    def fmt(r: dict | None) -> str:
        if not r:
            return "-"
        return "{:.3f} [{:.3f},{:.3f}]".format(r["mean"], r["ci_lo"], r["ci_hi"])

    for g in sorted(d2):
        a, b, c = d2[g], n2.get(g), d3.get(g)
        c3 = "{:.3f}".format(c["mean"]) if c else "-"
        n_pat, n_inst = a["n_patients"], a["n_instances"]
        print(f"{g:<24}{n_pat:>5}{n_inst:>7}{fmt(a):>26}{fmt(b):>26}{c3:>10}")
        lines.append(f"| {g} | {n_pat} | {n_inst} | {fmt(a)} | {fmt(b)} | {c3} |")

    # ---- jitter sweep (D12's ablation, free from the same pass)
    lines += ["", "## Box-perturbation sweep (Dice 2D)", "",
              "| group | " + " | ".join(f"{j:.0%}" for j in M.JITTER_LEVELS) + " |",
              "|---" * (len(M.JITTER_LEVELS) + 1) + "|"]
    print(f"\nbox-perturbation sweep (Dice 2D)\n{'group':<24}" +
          "".join(f"{j:>10.0%}" for j in M.JITTER_LEVELS))
    for g in sorted(d2):
        cells = []
        for j in M.JITTER_LEVELS:
            r = {x["group"]: x for x in summary
                 if x["level"] == "2d" and x["metric"] == "dice" and x["jitter"] == j}.get(g)
            cells.append(f"{r['mean']:.3f}" if r else "-")
        print(f"{g:<24}" + "".join(f"{c:>10}" for c in cells))
        lines.append(f"| {g} | " + " | ".join(cells) + " |")

    if bd:
        lines += ["", "## Breakdowns (headline jitter)", "",
                  "| cut | key | patients | instances | Dice |", "|---|---|---|---|---|"]
        print("\nbreakdowns (headline jitter)")
        for r in bd:
            val = r["mean"]
            note = {"zero_dice": " (fraction of instances with Dice=0)",
                    "sam_iou_vs_dice_corr": " (corr of SAM's predicted IoU with Dice)"}.get(r["kind"], "")
            print(f"  {r['kind']:<22}{r['key']:<26}{r['n_instances']:>7}  {val}{note}")
            lines.append(f"| {r['kind']} | {r['key']} | {r['n_patients']} | {r['n_instances']} | {val} |")

    if vols:
        sl = [r["sliver_frac"] for r in vols if r["jitter"] == M.HEADLINE_JITTER]
        if sl:
            msg = (f"3D caveat: SAM is never prompted on sub-50px kidney tips, so those voxels "
                   f"are missed by construction — median {np.median(sl):.4%} of kidney volume, "
                   f"max {max(sl):.4%}.")
            print(f"\n{msg}")
            lines += ["", msg]

    (args.arm_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"\nwrote summary.csv, breakdowns.csv, summary.md -> {args.arm_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
