#!/usr/bin/env python
"""Arm 3 across seeds (ARM3_EVAL_PLAN.md §5, D53): mean ± SD over the plain-student seeds,
and the seed-averaged paired comparison against a reference arm.

    python seed_summary.py --runs outputs/arm3_plain_s643 outputs/arm3_plain_s644 outputs/arm3_plain_s645 \
        --ref outputs/arm1_vit_b --label-ref "SAM ViT-B (arm 1b)" --out outputs/arm3_seeds

Every input is certified (compare_arms.certify) and every seed is proven to share the
reference's instance keys and prompts (compare_arms.pair_check), so all numbers rest on the
same kidneys. Aggregation is the evaluation's own: each patient's mean first, then the mean
over patients (as aggregate_arm / compare_arms).
* Table 1: per group x metric (Dice 2D, NSD@3 mm 2D, Dice 3D) at the headline jitter, each
  seed's score, then mean ± SD (n - 1) across seeds.
* Table 2 (per jitter level): seed-averaged paired comparison. A patient's student score is
  the mean over seeds of that patient's mean; Δ = student - reference with
  sam_metrics.paired_bootstrap_diff (10,000 resamples, seed 643), plus each seed's own Δ.
Ends SUMMARY_OK / SUMMARY_FAILED.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_arms as CA  # noqa: E402
import eval_sam as E  # noqa: E402
import sam_metrics as M  # noqa: E402

METRICS = (("2d", "dice"), ("2d", f"nsd_{M.HEADLINE_TOLERANCE_MM:g}"), ("3d", "dice3d"))


def patient_means(rows: dict, metric: str) -> dict:
    """{(group, jitter): {patient: mean of the metric over that patient's rows}}."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows.values():
        acc[(CA.group_label(r), round(float(r["jitter"]), 4))][r["patient"]].append(float(r[metric]))
    return {k: {p: sum(v) / len(v) for p, v in pats.items()} for k, pats in acc.items()}


def summarise(runs: list[Path], ref: Path, n_boot: int = 10000, seed: int = 643):
    problems = []
    models = {}
    for d in runs + [ref]:
        pr, model = CA.certify(d)
        problems += pr
        models[d] = model or {}
    if ref.resolve() in {d.resolve() for d in runs}:
        problems.append("the reference is one of the seed runs")
    if any(models[d].get("model_type") != "student_plain" for d in runs):
        problems.append(f"every seed run must be a plain student: {[models[d].get('model_type') for d in runs]}")
    shas = [models[d].get("checkpoint_sha256") for d in runs]
    if None in shas or len(set(shas)) != len(shas):
        problems.append("the seed runs do not hold distinct models (checkpoint sha256s repeat or are missing)")
    data = {}
    for d in runs + [ref]:
        ia, p1 = CA.load(d, "instances.csv", E.INSTANCE_FIELDS, E.inst_key)
        va, p2 = CA.load(d, "volumes_3d.csv", E.VOLUME_FIELDS, E.vol_key)
        problems += p1 + p2
        data[d] = {"2d": ia, "3d": va}
    if not problems:
        for d in runs:
            problems += CA.pair_check(data[ref]["2d"], data[d]["2d"], CA.INST_SAME, f"{d.name} vs {ref.name} instances")
            problems += CA.pair_check(data[ref]["3d"], data[d]["3d"], CA.VOL_SAME, f"{d.name} vs {ref.name} 3D")
    if problems:
        return None, problems
    t1, t2 = [], []
    for level, metric in METRICS:
        per_seed = [patient_means(data[d][level], metric) for d in runs]
        ref_pm = patient_means(data[ref][level], metric)
        for key in sorted(ref_pm):
            g, j = key
            pats = sorted(ref_pm[key])
            for pm in per_seed:
                if sorted(pm[key]) != pats:
                    return None, [f"{g} jitter {j}: patient sets differ between seeds and reference"]
            seed_scores = [sum(pm[key][p] for p in pats) / len(pats) for pm in per_seed]
            if abs(j - M.HEADLINE_JITTER) < 1e-9:
                t1.append({"level": level, "metric": metric, "group": g, "n_patients": len(pats),
                           **{f"seed_{d.name}": round(s, 4) for d, s in zip(runs, seed_scores)},
                           "mean": round(statistics.mean(seed_scores), 4),
                           "sd": round(statistics.stdev(seed_scores), 4) if len(seed_scores) > 1 else 0.0})
            a = [ref_pm[key][p] for p in pats]
            b = [sum(pm[key][p] for pm in per_seed) / len(per_seed) for p in pats]
            ma, mb, dlt, lo, hi = M.paired_bootstrap_diff(a, b, n_boot=n_boot, seed=seed)
            t2.append({"level": level, "metric": metric, "group": g, "jitter": j, "n_patients": len(pats),
                       "ref_mean": round(ma, 4), "student_seed_mean": round(mb, 4), "delta": round(dlt, 4),
                       "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                       "n_patients_student_better": sum(y > x for x, y in zip(a, b)),
                       "n_patients_ref_better": sum(x > y for x, y in zip(a, b)),
                       **{f"delta_{d.name}": round(sum(pm[key][p] - ref_pm[key][p] for p in pats) / len(pats), 4)
                          for d, pm in zip(runs, per_seed)}})
    return (t1, t2), []


def write(out: Path, runs: list[Path], ref: Path, label_ref: str, t1: list, t2: list) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("seed_table.csv", t1), (f"seed_compare_vs_{ref.name}.csv", t2)):
        with open(out / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    names = [d.name for d in runs]
    L = [f"# Arm 3 plain student: {len(runs)} seeds ({', '.join(names)})", "",
         f"Headline jitter {M.HEADLINE_JITTER:.0%}; each patient's mean first, then over patients. "
         "The plain student ignores the box, so its scores are identical at every jitter level.", "",
         "## Score per seed, mean ± SD across seeds", "",
         "| level | metric | group | pts | " + " | ".join(names) + " | mean ± SD |",
         "|---|---|---|---|" + "---|" * len(names) + "---|"]
    for r in t1:
        L.append(f"| {r['level']} | {r['metric']} | {r['group']} | {r['n_patients']} | "
                 + " | ".join(f"{r[f'seed_{n}']:.4f}" for n in names) + f" | {r['mean']:.4f} ± {r['sd']:.4f} |")
    L += ["", f"## Seed-averaged paired comparison vs {label_ref} (student − reference)", "",
          "| level | metric | group | jitter | pts | reference | student (seed mean) | Δ [95% CI] | "
          "pts student / ref better | Δ per seed |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in t2:
        L.append(f"| {r['level']} | {r['metric']} | {r['group']} | {r['jitter']:.2f} | {r['n_patients']} | "
                 f"{r['ref_mean']:.4f} | {r['student_seed_mean']:.4f} | {r['delta']:+.4f} "
                 f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | {r['n_patients_student_better']} / "
                 f"{r['n_patients_ref_better']} | " + ", ".join(f"{r[f'delta_{n}']:+.4f}" for n in names) + " |")
    (out / f"seed_summary_vs_{ref.name}.md").write_text("\n".join(L) + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--runs", nargs="+", type=Path, required=True)
    p.add_argument("--ref", type=Path, required=True)
    p.add_argument("--label-ref", default="reference")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-boot", type=int, default=10000)
    a = p.parse_args(argv)
    if len(a.runs) < 2 or len({d.resolve() for d in a.runs}) != len(a.runs):
        print("!! need at least two distinct seed folders\nSUMMARY_FAILED")
        return 1
    res, problems = summarise(a.runs, a.ref, a.n_boot)
    if problems:
        print("PROBLEMS:")
        for m in problems:
            print(f"  !! {m}")
        print("SUMMARY_FAILED")
        return 1
    t1, t2 = res
    write(a.out, a.runs, a.ref, a.label_ref, t1, t2)
    print((a.out / f"seed_summary_vs_{a.ref.name}.md").read_text()[:3000])
    print(f"wrote {a.out}\nSUMMARY_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
