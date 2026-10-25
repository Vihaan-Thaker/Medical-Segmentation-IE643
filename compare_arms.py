#!/usr/bin/env python
"""Paired comparison of two SAM evaluation runs on identical prompts (DECISIONS.md D32).

    python compare_arms.py --a outputs/arm1 --b outputs/arm1_vit_b \
                           --label-a "SAM ViT-H" --label-b "SAM ViT-B"

Both runs must have scored the same kidneys with the same boxes, so any difference is due
to the model alone. Before computing anything this script PROVES that, row by row:
  * both runs are certified complete by their run.json (no problems, not a shard);
  * the instance keys (case, slice, side, jitter) are identical sets, each exactly once;
  * every pair agrees on the clean box, the jittered box SAM was given, the GT area,
    identity fields and pixel spacing; every 3D pair on GT / sliver voxels and slice count.
Any mismatch -> COMPARE_FAILED and no numbers.

Differences are B - A, per group x jitter x metric. Each patient's mean is taken under
both arms (the same two-stage aggregation as aggregate_arm.py, so the means equal its
summary.csv), and the 95% CI is a paired bootstrap over patients.

Scope: SAM-vs-SAM runs sharing eval_sam.py's output format (arms 1, 1b, 2). The
prompt-free students (arms 3-4) have no boxes; extend this when their format is fixed.

Writes compare.csv and compare.md into --out (default: <b>/compare_vs_<a folder name>/).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_sam as E  # noqa: E402
import sam_metrics as M  # noqa: E402

INST_SAME = ("box_clean", "box_jit", "gt_area", "dataset", "patient", "sequence", "modality",
             "split", "sp_row", "sp_col")
VOL_SAME = ("gt_vox", "sliver_vox", "n_slices", "dataset", "patient", "sequence", "modality",
            "split", "sp_row", "sp_col", "sp_slice")
METRICS_2D = ("dice", *[f"nsd_{t:g}" for t in M.TOLERANCES_MM])
METRICS_3D = ("dice3d", *[f"nsd3d_{t:g}" for t in M.TOLERANCES_MM])


def group_label(r: dict) -> str:
    return "/".join(x for x in (r["dataset"], r["modality"], r["sequence"]) if x)


def certify(run_dir: Path) -> tuple[list[str], dict]:
    """Problems (empty = certified) and the run's model identity, from run.json."""
    rj_path = run_dir / "run.json"
    if not rj_path.exists():
        return [f"{run_dir}: no run.json -- not a verified evaluation"], {}
    rj = json.loads(rj_path.read_text())
    probs = []
    if rj.get("problems"):
        probs.append(f"{run_dir}: run.json reports problems: {rj['problems']}")
    if str(rj.get("mode", "")).startswith("shard"):
        probs.append(f"{run_dir}: is a shard (partial results)")
    model = rj.get("model") or {"model_type": rj.get("args", {}).get("model_type"),
                                "checkpoint_name": Path(rj.get("args", {}).get("checkpoint", "")).name}
    return probs, model


def load(run_dir: Path, name: str, fields: list[str], key) -> tuple[dict, list[str]]:
    rows, defects = E.read_results(run_dir / name, fields)
    probs = [f"{run_dir.name}/{d}" for d in defects]
    if not rows:
        probs.append(f"{run_dir / name}: no rows")
    by: dict = {}
    dup = 0
    for r in rows:
        k = key(r)
        dup += k in by
        by[k] = r
    if dup:
        probs.append(f"{run_dir / name}: {dup} duplicate keys")
    return by, probs


def pair_check(a: dict, b: dict, same: tuple[str, ...], what: str) -> list[str]:
    probs = []
    only_a, only_b = set(a) - set(b), set(b) - set(a)
    if only_a or only_b:
        probs.append(f"{what}: key sets differ ({len(only_a)} only in A, {len(only_b)} only in B), "
                     f"e.g. {sorted(only_a)[:2]} / {sorted(only_b)[:2]}")
    bad = []
    for k in sorted(set(a) & set(b)):
        diff = [f for f in same if a[k][f] != b[k][f]]
        if diff:
            bad.append(f"{k}: {', '.join(f'{f} {a[k][f]!r} vs {b[k][f]!r}' for f in diff)}")
    if bad:
        probs.append(f"{what}: {len(bad)} pairs disagree on prompt/ground truth, e.g. "
                     + " | ".join(bad[:3]))
    return probs


def paired_table(a: dict, b: dict, metrics: tuple[str, ...], level: str, n_boot: int,
                 seed: int) -> list[dict]:
    """One row per group x jitter x metric; patient means under each arm, paired bootstrap."""
    cells: dict = defaultdict(lambda: defaultdict(lambda: ([], [])))
    for k, ra in a.items():
        rb = b[k]
        cell = cells[(group_label(ra), float(ra["jitter"]))][ra["patient"]]
        cell[0].append(ra)
        cell[1].append(rb)
    out = []
    for (g, j), pats in sorted(cells.items()):
        n_inst = sum(len(v[0]) for v in pats.values())
        for m in metrics:
            pa = [sum(float(r[m]) for r in v[0]) / len(v[0]) for v in pats.values()]
            pb = [sum(float(r[m]) for r in v[1]) / len(v[1]) for v in pats.values()]
            ma, mb, d, lo, hi = M.paired_bootstrap_diff(pa, pb, n_boot=n_boot, seed=seed)
            out.append({"level": level, "group": g, "jitter": j, "metric": m,
                        "n_patients": len(pa), "n_instances": n_inst,
                        "mean_a": round(ma, 4), "mean_b": round(mb, 4), "delta": round(d, 4),
                        "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                        "n_patients_b_better": sum(y > x for x, y in zip(pa, pb)),
                        "n_patients_a_better": sum(x > y for x, y in zip(pa, pb))})
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--a", type=Path, required=True, help="reference run folder")
    p.add_argument("--b", type=Path, required=True, help="run compared against it")
    p.add_argument("--label-a", default="A")
    p.add_argument("--label-b", default="B")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=643)
    args = p.parse_args(argv)
    out = args.out or args.b / f"compare_vs_{args.a.resolve().name}"

    problems: list[str] = []
    pa_, model_a = certify(args.a)
    pb_, model_b = certify(args.b)
    problems += pa_ + pb_
    ia, p1 = load(args.a, "instances.csv", E.INSTANCE_FIELDS, E.inst_key)
    ib, p2 = load(args.b, "instances.csv", E.INSTANCE_FIELDS, E.inst_key)
    va, p3 = load(args.a, "volumes_3d.csv", E.VOLUME_FIELDS, E.vol_key)
    vb, p4 = load(args.b, "volumes_3d.csv", E.VOLUME_FIELDS, E.vol_key)
    problems += p1 + p2 + p3 + p4
    if not problems:
        problems += pair_check(ia, ib, INST_SAME, "instances")
        problems += pair_check(va, vb, VOL_SAME, "3D volumes")
    def describe(m: dict) -> str:
        if not m:
            return "model unknown"
        sha = m.get("checkpoint_sha256")
        return (f"{m.get('model_type')} / {m.get('checkpoint_name')}"
                + (f" (sha256 {sha[:12]}...)" if sha else " (pre-D32 run: no checksum recorded)"))
    print(f"A = {args.label_a}: {args.a}  [{describe(model_a)}]")
    print(f"B = {args.label_b}: {args.b}  [{describe(model_b)}]")
    if problems:
        print("\nPROBLEMS:")
        for m in problems:
            print(f"  !! {m}")
        print("\nCOMPARE_FAILED")
        return 1
    same_model = bool(model_a) and model_a == model_b
    print(f"\npaired: {len(ia)} instance rows and {len(va)} volume rows, identical keys; every "
          f"pair has the same clean box, jittered box, GT area and spacing -> any difference "
          f"below is due to the model alone"
          + ("\nNOTE: both runs report the SAME model -- this compares a model with itself" if same_model else ""))

    rows = (paired_table(ia, ib, METRICS_2D, "2d", args.n_boot, args.seed)
            + paired_table(va, vb, METRICS_3D, "3d", args.n_boot, args.seed))
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "compare.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def pick(level, metric, j):
        return {r["group"]: r for r in rows
                if r["level"] == level and r["metric"] == metric and abs(r["jitter"] - j) < 1e-9}

    def cell(r):
        return "{:+.3f} [{:+.3f},{:+.3f}]".format(r["delta"], r["ci_lo"], r["ci_hi"]) if r else "-"

    hn = f"nsd_{M.HEADLINE_TOLERANCE_MM:g}"
    d2, n2, d3 = (pick("2d", "dice", M.HEADLINE_JITTER), pick("2d", hn, M.HEADLINE_JITTER),
                  pick("3d", "dice3d", M.HEADLINE_JITTER))
    title = f"{args.label_b} vs {args.label_a} (B - A), jitter {M.HEADLINE_JITTER:.0%}"
    md = [f"# {title}", "",
          f"Paired over identical prompts: {len(ia)} instance rows, {len(va)} volume rows. "
          f"Mean over patients; 95% paired bootstrap CI over patients ({args.n_boot} resamples).", "",
          f"| group | pts | Dice 2D {args.label_a} | Dice 2D {args.label_b} | Δ Dice 2D | "
          f"Δ NSD@{M.HEADLINE_TOLERANCE_MM:g}mm | Δ Dice 3D | B better / A better (pts) |",
          "|---|---|---|---|---|---|---|---|"]
    print(f"\n{title}")
    print(f"{'group':<18}{'pts':>4}{'A dice':>8}{'B dice':>8}{'d Dice 2D [95% CI]':>27}"
          f"{'d NSD 2D [95% CI]':>27}{'d Dice 3D':>11}{'B/A better':>12}")
    for g in sorted(d2):
        r = d2[g]
        d3s = "{:+.3f}".format(d3[g]["delta"]) if g in d3 else "-"
        ba = f"{r['n_patients_b_better']}/{r['n_patients_a_better']}"
        print(f"{g:<18}{r['n_patients']:>4}{r['mean_a']:>8.3f}{r['mean_b']:>8.3f}"
              f"{cell(r):>27}{cell(n2.get(g)):>27}{d3s:>11}{ba:>12}")
        md.append(f"| {g} | {r['n_patients']} | {r['mean_a']:.3f} | {r['mean_b']:.3f} | {cell(r)} | "
                  f"{cell(n2.get(g))} | {d3s} | {ba} |")

    md += ["", "## Δ Dice 2D by box jitter (B - A)", "",
           "| group | " + " | ".join(f"{j:.0%}" for j in M.JITTER_LEVELS) + " |",
           "|---" * (len(M.JITTER_LEVELS) + 1) + "|"]
    print(f"\nd Dice 2D by jitter (B - A)\n{'group':<18}" + "".join(f"{j:>10.0%}" for j in M.JITTER_LEVELS))
    for g in sorted(d2):
        cells = []
        for j in M.JITTER_LEVELS:
            r = pick("2d", "dice", j).get(g)
            cells.append("{:+.3f}".format(r["delta"]) if r else "-")
        print(f"{g:<18}" + "".join(f"{c:>10}" for c in cells))
        md.append(f"| {g} | " + " | ".join(cells) + " |")
    (out / "compare.md").write_text("\n".join(md) + "\n")
    print(f"\nwrote compare.csv, compare.md -> {out}\nCOMPARE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
