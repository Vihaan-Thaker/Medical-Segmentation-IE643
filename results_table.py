#!/usr/bin/env python
"""One combined, cross-checked results table for arm 1 (ViT-H), arm 1b (ViT-B), arm 2 r=4 and
arm 2 r=8 (ARM2_POST_PLAN.md §3.3). Read-only; Mac Python (stdlib only).

    python results_table.py --results ../results      # -> results/arms_summary.md + .csv, POST_OK

Every number is copied from a certified file -- means/CIs from each arm's summary.csv, paired
differences from compare_arms.py's compare.csv -- never re-scored. Refuses if an arm is not
certified, the evaluation sets differ, two arms share a model, arm 2's evaluated checkpoint is not
the trained-and-merged one, or the rank selection did not precede the test.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import post_common as P  # noqa: E402

H = P.HEADLINE_JITTER


def load_all(results: Path) -> dict:
    arms = {name: P.load_arm(results, name, label) for name, label in P.ARMS}
    P.validate_arms(list(arms.values()))
    sel = P.selected_rank(results)
    for r in P.RANKS:
        tr = P.read_json(results / "runs" / f"arm2_r{r}" / "run.json")
        if tr.get("status") != "TRAIN_OK" or not tr.get("merged", {}).get("check", {}).get("passed"):
            raise P.PostError(f"arm2_r{r}: training not certified (status {tr.get('status')!r}, merge check "
                              f"{tr.get('merged', {}).get('check', {}).get('passed')!r}) -- run check_training.py")
        merged = tr.get("merged", {}).get("sha256")
        evaluated = arms[f"arm2_r{r}"]["model"].get("checkpoint_sha256")
        if not merged or evaluated != merged:
            raise P.PostError(f"arm2_r{r}: evaluated checkpoint sha256 {str(evaluated)[:16]}... is not the "
                              f"merged checkpoint from training {str(merged)[:16]}...")
        if sel[f"r{r}"]["merged_sha256"] != merged:
            raise P.PostError(f"arm2_r{r}: arm2_selection.json records a different merged checkpoint")
    same = arms["arm2_r4"]["model"].get("checkpoint_sha256") == arms["arm2_r8"]["model"].get("checkpoint_sha256")
    best0 = [P.read_json(results / "runs" / f"arm2_r{r}" / "run.json").get("best_epoch") for r in P.RANKS]
    if same and best0 != [0, 0]:
        raise P.PostError(f"arm2_r4 and arm2_r8 evaluated identical checkpoints, but their best epochs are "
                          f"{best0} (identical weights are only expected when both are 0)")
    compares = {(b, a): P.load_compare(results, b, a, arms) for b, a, _ in P.COMPARES}
    return {"arms": arms, "sel": sel, "compares": compares, "no_improvement": same}


def build(results: Path, data: dict) -> tuple[list[str], list[dict]]:
    arms, sel, compares = data["arms"], data["sel"], data["compares"]
    chosen = f"arm2_r{sel['chosen_rank']}"
    label = {n: arms[n]["label"] + (" (selected)" if n == chosen else "") for n, _ in P.ARMS}
    names = [n for n, _ in P.ARMS]
    L = [f"# Results — arms 1, 1b and 2 (selected rank r = {sel['chosen_rank']})", "",
         "Mean over patients (each patient's mean first), 95% bootstrap CI over patients. All numbers "
         "copied from each arm's certified `summary.csv`; paired differences from `compare_arms.py` "
         "(identical prompts proven per instance). Box jitter 10% unless stated; NSD at 3 mm.", ""]
    if data.get("no_improvement"):
        L += ["> **Note:** neither rank beat its epoch-0 (off-the-shelf) score on AMOS val, so both merged the "
              "untouched ViT-B weights: arm 2 r=4 and r=8 are the same model, numerically equal to arm 1b.", ""]

    def s(n, g, m, lvl="2d", j=H):
        return arms[n]["summary"][(lvl, g, m, P.jkey(j))]

    # 1. headline
    for title, lvl, m, with_ci in (("Dice 2D", "2d", "dice", True), ("NSD @ 3 mm (2D)", "2d", P.HEADLINE_NSD, True),
                                   ("Dice 3D", "3d", "dice3d", False)):
        L += [f"## {title} — 10% jitter", "",
              "| group | pts | " + " | ".join(label[n] for n in names) + " |",
              "|---|---|" + "---|" * len(names)]
        for g in P.GROUPS:
            cells = [P.fmt_ci(s(n, g, m, lvl)["mean"], s(n, g, m, lvl)["ci_lo"], s(n, g, m, lvl)["ci_hi"])
                     if with_ci else f"{s(n, g, m, lvl)['mean']:.3f}" for n in names]
            L.append(f"| {g} | {s(names[0], g, m, lvl)['n_patients']} | " + " | ".join(cells) + " |")
        L.append("")

    # 2. jitter sweep
    L += ["## Dice 2D vs box jitter (0 / 5 / 10 / 20%)", "",
          "| group | " + " | ".join(label[n] for n in names) + " |", "|---|" + "---|" * len(names)]
    for g in P.GROUPS:
        L.append(f"| {g} | " + " | ".join(" / ".join(f"{s(n, g, 'dice', j=j)['mean']:.3f}" for j in P.JITTERS)
                                          for n in names) + " |")
    L.append("")

    # 3. paired differences
    for b, a, title in P.COMPARES:
        c = compares[(b, a)]
        L += [f"## Paired: {title} (B − A)", ""]
        if c is None:
            L += [f"_not available: outputs/{b}/compare_vs_{a}/compare.csv was not fetched or not produced._", ""]
            continue
        rows = c["rows"]
        L += ["| group | pts | Δ Dice 2D [95% CI] | Δ NSD@3 [95% CI] | Δ Dice 3D | pts B / A better |",
              "|---|---|---|---|---|---|"]
        for g in P.GROUPS:
            d, n, d3 = rows[("2d", g, "dice", P.jkey(H))], rows[("2d", g, P.HEADLINE_NSD, P.jkey(H))], \
                rows[("3d", g, "dice3d", P.jkey(H))]
            L.append(f"| {g} | {d['n_patients']} | {P.fmt_ci(d['delta'], d['ci_lo'], d['ci_hi'], True)} | "
                     f"{P.fmt_ci(n['delta'], n['ci_lo'], n['ci_hi'], True)} | {d3['delta']:+.3f} | "
                     f"{d['n_patients_b_better']} / {d['n_patients_a_better']} |")
        L += ["", "Δ Dice 2D by jitter (0 / 5 / 10 / 20%): " + "; ".join(
            f"{g} " + " / ".join(f"{rows[('2d', g, 'dice', P.jkey(j))]['delta']:+.3f}" for j in P.JITTERS)
            for g in P.GROUPS), ""]

    # 4. in-domain vs external, for the selected arm 2
    c = compares[(chosen, "arm1_vit_b")]
    L += [f"## In-domain vs external: Arm 1b → {label[chosen]}", ""]
    if c is None:
        L += ["_not available (compare.csv missing)._", ""]
    else:
        L += ["| | group | Δ Dice 2D @10% [95% CI] | Δ Dice 2D @20% [95% CI] |", "|---|---|---|---|"]
        for g in P.GROUPS:
            d10, d20 = c["rows"][("2d", g, "dice", P.jkey(0.10))], c["rows"][("2d", g, "dice", P.jkey(0.20))]
            kind = "in-domain (AMOS)" if g in P.IN_DOMAIN else "external"
            L.append(f"| {kind} | {g} | {P.fmt_ci(d10['delta'], d10['ci_lo'], d10['ci_hi'], True)} | "
                     f"{P.fmt_ci(d20['delta'], d20['ci_lo'], d20['ci_hi'], True)} |")
        L += ["", "_Per group only: pooling groups would need patient-level re-aggregation, which this "
              "table deliberately does not do._", ""]

    # 5. provenance
    L += ["## Provenance", "", "| arm | model | checkpoint sha256 | finished (Prajna local time) |",
          "|---|---|---|---|"]
    for n in names:
        m, run = arms[n]["model"], arms[n]["run"]
        sha = m.get("checkpoint_sha256")
        sha_txt = f"{sha[:16]}..." if sha else "not recorded (pre-stamp run)"
        L.append(f"| {n} | {m.get('model_type')} / {m.get('checkpoint_name')} | {sha_txt} | "
                 f"{run.get('finished', '?')} |")
    L += ["", f"Rank selection: r = {sel['chosen_rank']} at {sel.get('selected_utc')} (UTC), "
          f"with no `outputs/arm2_r*` folder present (recorded by `select_rank.py`).", "",
          "| input file | sha256 |", "|---|---|"]
    files = [f for n in names for f in arms[n]["files"]] + \
            [c["path"] for c in compares.values() if c is not None] + [results / "runs" / "arm2_selection.json"]
    for f in files:
        L.append(f"| {Path(f).relative_to(results)} | {P.file_sha256(f)[:16]} |")

    long_rows = []
    for n in names:
        for (lvl, g, m, j), v in sorted(arms[n]["summary"].items()):
            long_rows.append({"arm": n, "selected": int(n == chosen), "group": g, "level": lvl, "metric": m,
                              "jitter": j, **v})
    return L, long_rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results", type=Path, required=True)
    a = p.parse_args(argv)
    try:
        data = load_all(a.results)
        lines, long_rows = build(a.results, data)
    except P.PostError as e:
        print(f"!! {e}\nPOST_FAILED")
        return 1
    (a.results / "arms_summary.md").write_text("\n".join(lines) + "\n")
    with open(a.results / "arms_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(long_rows[0].keys()))
        w.writeheader()
        w.writerows(long_rows)
    print("\n".join(lines[:40]))
    print(f"...\nwrote {a.results / 'arms_summary.md'} and arms_summary.csv ({len(long_rows)} rows)\nPOST_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
