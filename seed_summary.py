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
Arm 4 (ARM4_CODE_PLAN.md §11): the seed runs may instead all be distilled students
(student_plain_kd), and --ref-runs replaces --ref with a SEED-PAIRED reference: e.g. the three
arm-3 seeds, each paired with the arm-4 run of the same training seed (eval run.json "seed").
A patient's reference score is then the mean over the reference seeds, and "Δ per seed" is
run_s - ref_s. Output: seed_compare_vs_<label slug>.csv and seed_summary_vs_<label slug>.md.
Ends SUMMARY_OK / SUMMARY_FAILED.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_arms as CA  # noqa: E402
import eval_sam as E  # noqa: E402
import sam_metrics as M  # noqa: E402

PLAIN_TYPES = ("student_plain", "student_plain_kd")
TITLES = {"student_plain": "Arm 3 plain student", "student_plain_kd": "Arm 4 distilled plain student"}
METRICS = (("2d", "dice"), ("2d", f"nsd_{M.HEADLINE_TOLERANCE_MM:g}"), ("3d", "dice3d"))


def patient_means(rows: dict, metric: str) -> dict:
    """{(group, jitter): {patient: mean of the metric over that patient's rows}}."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows.values():
        acc[(CA.group_label(r), round(float(r["jitter"]), 4))][r["patient"]].append(float(r[metric]))
    return {k: {p: sum(v) / len(v) for p, v in pats.items()} for k, pats in acc.items()}


def run_seed(d: Path):
    try:
        return int(json.loads((d / "run.json").read_text())["seed"])
    except Exception:
        return None


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "reference"


def summarise(runs: list[Path], ref: Path | None, n_boot: int = 10000, seed: int = 643,
              ref_runs: list[Path] | None = None):
    """ref: one reference arm (arms 1b, 2). ref_runs: seed-paired references (arm 3 for arm 4)."""
    problems = []
    refs = [ref] if ref_runs is None else list(ref_runs)
    models = {}
    for d in runs + refs:
        pr, model = CA.certify(d)
        problems += pr
        models[d] = model or {}
    if {r.resolve() for r in refs} & {d.resolve() for d in runs}:
        problems.append("the reference is one of the seed runs")
    types = {models[d].get("model_type") for d in runs}
    if len(types) != 1 or not types <= set(PLAIN_TYPES):
        problems.append(f"every seed run must be a plain student, all of one type {PLAIN_TYPES}: "
                        f"{[models[d].get('model_type') for d in runs]}")
    shas = [models[d].get("checkpoint_sha256") for d in runs]
    if None in shas or len(set(shas)) != len(shas):
        problems.append("the seed runs do not hold distinct models (checkpoint sha256s repeat or are missing)")
    partner = {}
    if ref_runs is not None:
        rshas = [models[d].get("checkpoint_sha256") for d in refs]
        if None in rshas or len(set(rshas)) != len(rshas):
            problems.append("the reference runs do not hold distinct models")
        rs, fs = [run_seed(d) for d in runs], [run_seed(d) for d in refs]
        if None in rs + fs or len(set(rs)) != len(rs) or len(set(fs)) != len(fs) or set(rs) != set(fs):
            problems.append(f"seed sets must be identical and distinct: runs {rs}, references {fs}")
        else:
            by_seed = dict(zip(fs, refs))
            partner = {d: by_seed[s_] for d, s_ in zip(runs, rs)}
    data = {}
    for d in runs + refs:
        ia, p1 = CA.load(d, "instances.csv", E.INSTANCE_FIELDS, E.inst_key)
        va, p2 = CA.load(d, "volumes_3d.csv", E.VOLUME_FIELDS, E.vol_key)
        problems += p1 + p2
        data[d] = {"2d": ia, "3d": va}
    if not problems:
        for d in runs:
            r = ref if ref_runs is None else partner[d]
            problems += CA.pair_check(data[r]["2d"], data[d]["2d"], CA.INST_SAME, f"{d.name} vs {r.name} instances")
            problems += CA.pair_check(data[r]["3d"], data[d]["3d"], CA.VOL_SAME, f"{d.name} vs {r.name} 3D")
    if problems:
        return None, problems
    t1, t2 = [], []
    for level, metric in METRICS:
        per_seed = [patient_means(data[d][level], metric) for d in runs]
        ref_pms = {r: patient_means(data[r][level], metric) for r in refs}
        ref_pm = ref_pms[refs[0]]
        for key in sorted(ref_pm):
            g, j = key
            pats = sorted(ref_pm[key])
            for pm in per_seed + list(ref_pms.values()):
                if sorted(pm.get(key, {})) != pats:
                    return None, [f"{g} jitter {j}: patient sets differ between seeds and reference"]
            seed_scores = [sum(pm[key][p] for p in pats) / len(pats) for pm in per_seed]
            if abs(j - M.HEADLINE_JITTER) < 1e-9:
                t1.append({"level": level, "metric": metric, "group": g, "n_patients": len(pats),
                           **{f"seed_{d.name}": round(s, 4) for d, s in zip(runs, seed_scores)},
                           "mean": round(statistics.mean(seed_scores), 4),
                           "sd": round(statistics.stdev(seed_scores), 4) if len(seed_scores) > 1 else 0.0})
            if ref_runs is None:
                a = [ref_pm[key][p] for p in pats]
                own_ref = {d: ref_pm for d in runs}
            else:
                a = [sum(ref_pms[r][key][p] for r in refs) / len(refs) for p in pats]
                own_ref = {d: ref_pms[partner[d]] for d in runs}
            b = [sum(pm[key][p] for pm in per_seed) / len(per_seed) for p in pats]
            ma, mb, dlt, lo, hi = M.paired_bootstrap_diff(a, b, n_boot=n_boot, seed=seed)
            t2.append({"level": level, "metric": metric, "group": g, "jitter": j, "n_patients": len(pats),
                       "ref_mean": round(ma, 4), "student_seed_mean": round(mb, 4), "delta": round(dlt, 4),
                       "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                       "n_patients_student_better": sum(y > x for x, y in zip(a, b)),
                       "n_patients_ref_better": sum(x > y for x, y in zip(a, b)),
                       **{f"delta_{d.name}": round(sum(pm[key][p] - own_ref[d][key][p] for p in pats) / len(pats), 4)
                          for d, pm in zip(runs, per_seed)}})
    return (t1, t2), []


def write(out: Path, runs: list[Path], tag: str, label_ref: str, t1: list, t2: list,
          title: str = TITLES["student_plain"], paired_note: str = "") -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("seed_table.csv", t1), (f"seed_compare_vs_{tag}.csv", t2)):
        with open(out / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    names = [d.name for d in runs]
    L = [f"# {title}: {len(runs)} seeds ({', '.join(names)})", "",
         f"Headline jitter {M.HEADLINE_JITTER:.0%}; each patient's mean first, then over patients. "
         "The plain student ignores the box, so its scores are identical at every jitter level.", "",
         "## Score per seed, mean ± SD across seeds", "",
         "| level | metric | group | pts | " + " | ".join(names) + " | mean ± SD |",
         "|---|---|---|---|" + "---|" * len(names) + "---|"]
    for r in t1:
        L.append(f"| {r['level']} | {r['metric']} | {r['group']} | {r['n_patients']} | "
                 + " | ".join(f"{r[f'seed_{n}']:.4f}" for n in names) + f" | {r['mean']:.4f} ± {r['sd']:.4f} |")
    L += ["", f"## Seed-averaged paired comparison vs {label_ref} (student − reference)", ""]
    if paired_note:
        L += [paired_note, ""]
    L += [
          "| level | metric | group | jitter | pts | reference | student (seed mean) | Δ [95% CI] | "
          "pts student / ref better | Δ per seed |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in t2:
        L.append(f"| {r['level']} | {r['metric']} | {r['group']} | {r['jitter']:.2f} | {r['n_patients']} | "
                 f"{r['ref_mean']:.4f} | {r['student_seed_mean']:.4f} | {r['delta']:+.4f} "
                 f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | {r['n_patients_student_better']} / "
                 f"{r['n_patients_ref_better']} | " + ", ".join(f"{r[f'delta_{n}']:+.4f}" for n in names) + " |")
    (out / f"seed_summary_vs_{tag}.md").write_text("\n".join(L) + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--runs", nargs="+", type=Path, required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--ref", type=Path)
    g.add_argument("--ref-runs", nargs="+", type=Path,
                   help="seed-paired references, matched to --runs by training seed")
    p.add_argument("--label-ref", default="reference")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-boot", type=int, default=10000)
    a = p.parse_args(argv)
    if len(a.runs) < 2 or len({d.resolve() for d in a.runs}) != len(a.runs):
        print("!! need at least two distinct seed folders\nSUMMARY_FAILED")
        return 1
    if a.ref_runs is not None and len({d.resolve() for d in a.ref_runs}) != len(a.ref_runs):
        print("!! the reference folders repeat\nSUMMARY_FAILED")
        return 1
    res, problems = summarise(a.runs, a.ref, a.n_boot, ref_runs=a.ref_runs)
    if problems:
        print("PROBLEMS:")
        for m in problems:
            print(f"  !! {m}")
        print("SUMMARY_FAILED")
        return 1
    t1, t2 = res
    mtype = CA.certify(a.runs[0])[1].get("model_type")
    if a.ref_runs is None:
        tag, note = a.ref.name, ""
    else:
        tag = slug(a.label_ref)
        note = ("Seed-paired: each run is paired with the reference run of the same training seed; a "
                "patient's reference score is the mean over the reference seeds; Δ per seed = run − its "
                "own partner (" + ", ".join(d.name for d in a.ref_runs) + ").")
    write(a.out, a.runs, tag, a.label_ref, t1, t2, TITLES[mtype], note)
    print((a.out / f"seed_summary_vs_{tag}.md").read_text()[:3000])
    print(f"wrote {a.out}\nSUMMARY_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
