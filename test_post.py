#!/usr/bin/env python
"""Tests for the arm-2 post-processing (ARM2_POST_PLAN.md §5): fetch_results.sh, post_common,
check_training, results_table, plot_results. Mac-only (needs matplotlib), seconds.

    python test_post.py        # exits 0 and prints ALL TESTS PASSED

A synthetic results/ tree is built whose numbers are internally consistent the way the real
pipeline's are: summary means come from per-patient means of instances.csv, compare means from
the summaries, compare deltas = mean_b - mean_a. Then every refusal path is exercised.
"""
from __future__ import annotations

import contextlib
import csv
import io
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import check_training as CT  # noqa: E402
import plot_results as PR  # noqa: E402
import post_common as P  # noqa: E402
import results_table as RT  # noqa: E402
import select_rank as S  # noqa: E402

SHA = {"arm1_vit_b": "e" * 64, "arm2_r4": "4" * 64, "arm2_r8": "8" * 64}
QUALITY = {"arm1": 0.90, "arm1_vit_b": 0.905, "arm2_r4": 0.94, "arm2_r8": 0.945}


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)
    print(f"  ok  {msg}")


def run(mod, results: Path) -> tuple[int, str]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = mod.main(["--results", str(results)])
    return rc, buf.getvalue()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------- synthetic results tree
def make_instances(arm: str, rng: np.random.Generator) -> list[dict]:
    rows = []
    for gi, g in enumerate(P.GROUPS):
        parts = g.split("/")
        ds, mod, seq = parts[0], parts[1], parts[2] if len(parts) > 2 else ""
        for p in range(3):
            patient = f"{ds}_{gi}{p}"
            case = patient + (f"_{seq}" if seq else "")
            for z in range(2):
                for side in ("right", "left"):
                    for j in P.JITTERS:
                        q = QUALITY[arm] - 0.3 * j
                        row = {"dataset": ds, "patient": patient, "case": case, "sequence": seq,
                               "modality": mod, "split": "test", "z": z, "side": side, "jitter": j,
                               "box_clean": json.dumps([z, 1, 9, 9]), "box_jit": json.dumps([z, 1, 9 + int(10 * j), 9]),
                               "box_iou": 1.0, "gt_area": 100 + z, "pred_area": 99}
                        for m in P.METRICS_2D:
                            row[m] = round(float(np.clip(q + rng.normal(0, 0.02), 0, 1)), 6)
                        rows.append(row)
    return rows


def summary_from(inst: list[dict], arm: str, rng: np.random.Generator) -> list[dict]:
    out = []
    for g in P.GROUPS:
        for j in P.JITTERS:
            sel = [r for r in inst if P.group_of(r) == g and abs(r["jitter"] - j) < 1e-9]
            for m in P.METRICS_2D:
                per = {}
                for r in sel:
                    per.setdefault(r["patient"], []).append(r[m])
                pm = [sum(v) / len(v) for v in per.values()]
                mean = sum(pm) / len(pm)
                out.append({"group": g, "jitter": j, "metric": m, "n_patients": len(pm), "n_instances": len(sel),
                            "mean": round(mean, 4), "ci_lo": round(mean - 0.01, 4), "ci_hi": round(mean + 0.01, 4),
                            "patient_min": 0, "patient_median": 0, "patient_max": 0, "level": "2d"})
            for m in P.METRICS_3D:
                mean = QUALITY[arm] - 0.2 * j + 0.001 * P.GROUPS.index(g)
                out.append({"group": g, "jitter": j, "metric": m, "n_patients": 3, "n_instances": 6,
                            "mean": round(mean, 4), "ci_lo": round(mean - 0.01, 4), "ci_hi": round(mean + 0.01, 4),
                            "patient_min": 0, "patient_median": 0, "patient_max": 0, "level": "3d"})
    return out


def compare_from(sa: list[dict], sb: list[dict]) -> list[dict]:
    ka = {(r["level"], r["group"], r["metric"], round(r["jitter"], 4)): r for r in sa}
    out = []
    for r in sb:
        a = ka[(r["level"], r["group"], r["metric"], round(r["jitter"], 4))]
        d = round(r["mean"] - a["mean"], 4)
        out.append({"level": r["level"], "group": r["group"], "jitter": r["jitter"], "metric": r["metric"],
                    "n_patients": r["n_patients"], "n_instances": r["n_instances"], "mean_a": a["mean"],
                    "mean_b": r["mean"], "delta": d, "ci_lo": round(d - 0.005, 4), "ci_hi": round(d + 0.005, 4),
                    "n_patients_b_better": 2, "n_patients_a_better": 1})
    return out


def metrics_rows(r: int, rng: np.random.Generator) -> list[dict]:
    rows = []
    for e in range(36):
        sel = 0.91 + 0.05 * (1 - np.exp(-e / 4)) + (0.002 if (e == 20 and r == 8) else 0) - (0.001 if e > 25 else 0)
        row = {"epoch": e, "ct@0.00": round(sel - 0.002, 6), "mri@0.00": round(sel + 0.003, 6),
               "ct@0.10": round(sel - 0.003, 6), "mri@0.10": round(sel + 0.003, 6), "selection": round(sel, 6),
               "nsd3_ct@0.10": 0.9, "nsd3_mri@0.10": 0.9, "val_s": 300.0, "steps": 0 if e == 0 else 1036}
        for c in P.TRAIN_COLS:
            row[c] = "" if e == 0 else (1500.0 if c == "epoch_s" else round(0.05 * np.exp(-e / 10) + 0.02, 6))
        rows.append(row)
    return rows


def build_tree(root: Path) -> None:
    rng = np.random.default_rng(1)
    inst = {a: make_instances(a, rng) for a, _ in P.ARMS}
    summ = {a: summary_from(inst[a], a, rng) for a, _ in P.ARMS}
    for a, _ in P.ARMS:
        d = root / "outputs" / a
        write_csv(d / "summary.csv", summ[a])
        run_ = {"problems": [], "mode": "merge of 4 shards" if a == "arm1" else "single job",
                "finished": "2026-10-25 10:00:00",
                "args": {"model_type": "vit_h" if a == "arm1" else "vit_b",
                         "checkpoint": "/x/sam_vit_h_4b8939.pth" if a == "arm1" else f"/x/{a}.pth"}}
        if a != "arm1":
            model = {"model_type": "vit_b", "checkpoint_name": f"{a}.pth", "checkpoint_bytes": 1,
                     "checkpoint_sha256": SHA[a]}
            run_["model"] = model
            (d / "model.json").write_text(json.dumps(model))
        (d / "run.json").write_text(json.dumps(run_))
        if a in ("arm1_vit_b", "arm2_r4", "arm2_r8"):
            write_csv(d / "instances.csv", inst[a])
    for b, a, _ in P.COMPARES:
        write_csv(root / "outputs" / b / f"compare_vs_{a}" / "compare.csv", compare_from(summ[a], summ[b]))
    best = {}
    for r in P.RANKS:
        d = root / "runs" / f"arm2_r{r}"
        rows = metrics_rows(r, rng)
        write_csv(d / "metrics.csv", rows)
        sel = [float(x["selection"]) for x in rows]
        b = sel.index(max(sel))
        best[r] = (b, max(sel))
        settings = {"epochs": 35, "batch": 8, "accum": 1, "grad_checkpoint": False, "rank": r, "lr_lora": 5e-4}
        (d / "run.json").write_text(json.dumps({
            "status": "TRAIN_OK", "rank": r, "epochs_completed": 35, "best_epoch": b, "best_score": max(sel),
            "settings": settings, "gpu": {"name": "NVIDIA A100-SXM4-80GB"}, "gpu_type": "NVIDIA A100-SXM4-80GB",
            "info": {"trainable": 36864 * r + 4058340, "lora": 36864 * r},
            "merged": {"sha256": SHA[f"arm2_r{r}"], "best_epoch": b,
                       "check": {"passed": True, "pixel_agreement": 1.0}}}))
        (d / "config.json").write_text(json.dumps({
            "settings": settings, "base_sha256": "b", "manifest_sha256": "m",
            "gpu_type": "NVIDIA A100-SXM4-80GB", "code": {"f": "1"}}))
        (d / "state.json").write_text("{}")
    pre = root / "runs" / "arm2_preflight"
    pre.mkdir(parents=True)
    (pre / "preflight.json").write_text(json.dumps({"passed": True, "recommended":
                                                    {"batch": 8, "accum": 1, "grad_checkpoint": False}}))
    dec = S.decide(best[4][1], best[8][1])
    (root / "runs" / "arm2_selection.json").write_text(json.dumps({
        **dec, "r4": {"best_score": best[4][1], "best_epoch": best[4][0], "merged_sha256": SHA["arm2_r4"]},
        "r8": {"best_score": best[8][1], "best_epoch": best[8][0], "merged_sha256": SHA["arm2_r8"]},
        "rule": "D33", "selected_utc": "2026-10-25T03:00:00Z", "existing_eval_folders": [],
        "ack_existing_eval": None}))


def variant(base: Path, tmp: Path, tag: str) -> Path:
    v = tmp / tag
    shutil.copytree(base, v)
    return v


def edit_json(path: Path, fn) -> None:
    d = json.loads(path.read_text())
    fn(d)
    path.write_text(json.dumps(d))


def edit_csv(path: Path, fn) -> None:
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    rows = fn(rows)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ============================================================================ tests
def test_environment() -> None:
    print("\n--- environment: Mac-Python safe ---")
    code = ("import sys; sys.path.insert(0, %r); import post_common, check_training, results_table, plot_results; "
            "bad = [m for m in ('torch', 'surface_distance', 'eval_sam', 'sam_metrics', 'compare_arms', "
            "'train_sam_lora') if m in sys.modules]; print(bad)") % str(HERE)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    check(out.returncode == 0 and out.stdout.strip() == "[]",
          f"importing the post scripts loads no torch / surface-distance / eval code ({out.stdout.strip()})")
    import ast
    bad = []
    for f in ("post_common.py", "check_training.py", "results_table.py", "plot_results.py", "test_post.py"):
        try:
            ast.parse((HERE / f).read_text(), feature_version=(3, 11))
        except SyntaxError as e:
            bad.append(f"{f}:{e.lineno}")
    check(not bad, "post files parse as Python 3.11")


def test_happy(base: Path) -> None:
    print("\n--- happy path ---")
    rc, out = run(CT, base)
    check(rc == 0 and "POST_OK" in out and (base / "training_summary.md").exists(), "check_training: POST_OK")
    sel = json.loads((base / "runs" / "arm2_selection.json").read_text())
    check(f"Selected rank: r = {sel['chosen_rank']}" in (base / "training_summary.md").read_text(),
          f"training summary names the selected rank (r = {sel['chosen_rank']})")
    rc, out = run(RT, base)
    md = (base / "arms_summary.md").read_text()
    check(rc == 0 and "POST_OK" in out, "results_table: POST_OK")
    s = {(r["level"], r["group"], r["metric"], round(float(r["jitter"]), 4)): r
         for r in csv.DictReader(open(base / "outputs" / "arm2_r8" / "summary.csv"))}
    cell = s[("2d", "word/ct", "dice", 0.1)]
    want = "{:.3f} [{:.3f}, {:.3f}]".format(float(cell["mean"]), float(cell["ci_lo"]), float(cell["ci_hi"]))
    check(want in md, f"headline cell copied from summary.csv ({want})")
    c = {(r["level"], r["group"], r["metric"], round(float(r["jitter"]), 4)): r
         for r in csv.DictReader(open(base / "outputs" / "arm2_r8" / "compare_vs_arm1_vit_b" / "compare.csv"))}
    d = c[("2d", "chaos/mri/t1out", "dice", 0.1)]
    want = "{:+.3f} [{:+.3f}, {:+.3f}]".format(float(d["delta"]), float(d["ci_lo"]), float(d["ci_hi"]))
    check(want in md, f"paired delta copied from compare.csv ({want})")
    n = sum(1 for _ in csv.DictReader(open(base / "arms_summary.csv")))
    check(n == 4 * len(P.expected_grid()), f"arms_summary.csv has all {n} rows (4 arms x full grid)")
    check("(selected)" in md and "Provenance" in md and "not recorded (pre-stamp run)" in md,
          "the selected arm is marked; provenance lists arm 1 as a pre-stamp run")
    rc, out = run(PR, base)
    figs = sorted(p.name for p in (base / "figures").glob("*.png"))
    check(rc == 0 and figs == ["gain_by_dataset.png", "jitter_robustness.png", "patient_delta.png",
                               "training_curves.png"] and all((base / "figures" / f).stat().st_size > 10000 for f in figs),
          f"plot_results: 4 non-empty figures {figs}")


def test_check_training_refusals(base: Path, tmp: Path) -> None:
    print("\n--- check_training refusals ---")
    cases = [
        ("non-contiguous epochs", lambda v: edit_csv(v / "runs/arm2_r4/metrics.csv", lambda rs: rs[:10] + rs[11:])),
        ("a NaN validation cell", lambda v: edit_csv(v / "runs/arm2_r4/metrics.csv",
                                                      lambda rs: [dict(r, **({"mri@0.10": "nan"} if r["epoch"] == "7" else {})) for r in rs])),
        ("a training value in the epoch-0 row", lambda v: edit_csv(v / "runs/arm2_r8/metrics.csv",
                                                                   lambda rs: [dict(rs[0], train_loss="0.1")] + rs[1:])),
        ("best epoch != first maximum", lambda v: edit_json(v / "runs/arm2_r4/run.json", lambda d: d.update(best_epoch=3))),
        ("merge check not passed", lambda v: edit_json(v / "runs/arm2_r8/run.json",
                                                        lambda d: d["merged"]["check"].update(passed=False))),
        ("merged checkpoint not from the best epoch", lambda v: edit_json(v / "runs/arm2_r4/run.json",
                                                                          lambda d: d["merged"].update(best_epoch=1))),
        ("settings differ beyond rank", lambda v: edit_json(v / "runs/arm2_r8/config.json",
                                                             lambda d: d["settings"].update(lr_lora=1e-3))),
        ("batch settings differ from the preflight", lambda v: edit_json(v / "runs/arm2_preflight/preflight.json",
                                                                         lambda d: d["recommended"].update(batch=4))),
        ("selection inconsistent with the rule", lambda v: edit_json(v / "runs/arm2_selection.json",
                                                                     lambda d: d.update(chosen_rank=12 - d["chosen_rank"]))),
        ("status not TRAIN_OK", lambda v: edit_json(v / "runs/arm2_r4/run.json", lambda d: d.update(status="RUNNING"))),
        ("a NaN NSD cell when NSD was computed", lambda v: edit_csv(v / "runs/arm2_r4/metrics.csv",
                                                                     lambda rs: [dict(r, **{"nsd3_ct@0.10": "nan"}) for r in rs])),
    ]
    for i, (why, mutate) in enumerate(cases):
        v = variant(base, tmp, f"ct{i}")
        mutate(v)
        rc, out = run(CT, v)
        check(rc == 1 and "POST_FAILED" in out, f"check_training refuses: {why}")


def test_table_refusals(base: Path, tmp: Path) -> None:
    print("\n--- results_table refusals ---")
    cases = [
        ("an arm reporting problems", lambda v: edit_json(v / "outputs/arm2_r4/run.json", lambda d: d.update(problems=["x"]))),
        ("a shard run.json", lambda v: edit_json(v / "outputs/arm1_vit_b/run.json", lambda d: d.update(mode="shard 0/2"))),
        ("evaluation sets differ", lambda v: edit_csv(v / "outputs/arm2_r8/summary.csv",
                                                      lambda rs: [dict(r, n_instances=str(int(r["n_instances"]) + (1 if i == 0 else 0)))
                                                                  for i, r in enumerate(rs)])),
        ("a missing arm", lambda v: shutil.rmtree(v / "outputs/arm2_r4")),
        ("an incomplete summary grid", lambda v: edit_csv(v / "outputs/arm1/summary.csv", lambda rs: rs[:-1])),
        ("arm 2 evaluated a different checkpoint", lambda v: (
            edit_json(v / "outputs/arm2_r8/model.json", lambda d: d.update(checkpoint_sha256="f" * 64)),
            edit_json(v / "outputs/arm2_r8/run.json", lambda d: d["model"].update(checkpoint_sha256="f" * 64)))),
        ("model.json and run.json disagree", lambda v: edit_json(v / "outputs/arm2_r4/model.json",
                                                                 lambda d: d.update(checkpoint_name="other.pth"))),
        ("two arms with the same model", lambda v: (
            edit_json(v / "outputs/arm2_r4/model.json", lambda d: d.update(checkpoint_sha256=SHA["arm2_r8"])),
            edit_json(v / "outputs/arm2_r4/run.json", lambda d: d["model"].update(checkpoint_sha256=SHA["arm2_r8"])))),
        ("selection made after an evaluation folder existed", lambda v: edit_json(
            v / "runs/arm2_selection.json", lambda d: d.update(existing_eval_folders=["outputs/arm2_r4"]))),
        ("selection made with an override", lambda v: edit_json(v / "runs/arm2_selection.json",
                                                                lambda d: d.update(ack_existing_eval="because"))),
        ("training not TRAIN_OK", lambda v: edit_json(v / "runs/arm2_r4/run.json", lambda d: d.update(status="RUNNING"))),
        ("training merge check not passed", lambda v: edit_json(v / "runs/arm2_r8/run.json",
                                                                 lambda d: d["merged"]["check"].update(passed=False))),
        ("a compare.csv inconsistent with the summaries", lambda v: edit_csv(
            v / "outputs/arm2_r4/compare_vs_arm1_vit_b/compare.csv",
            lambda rs: [dict(r, mean_b=str(float(r["mean_b"]) + 0.01)) if i == 5 else r for i, r in enumerate(rs)])),
    ]
    for i, (why, mutate) in enumerate(cases):
        v = variant(base, tmp, f"rt{i}")
        mutate(v)
        rc, out = run(RT, v)
        check(rc == 1 and "POST_FAILED" in out, f"results_table refuses: {why}")
    v = variant(base, tmp, "rt_same_ok")      # neither rank improved: identical merged weights
    for r in P.RANKS:
        edit_json(v / f"runs/arm2_r{r}/run.json", lambda d: (d.update(best_epoch=0),
                                                             d["merged"].update(sha256=SHA["arm2_r4"])))
    edit_json(v / "outputs/arm2_r8/model.json", lambda d: d.update(checkpoint_sha256=SHA["arm2_r4"]))
    edit_json(v / "outputs/arm2_r8/run.json", lambda d: d["model"].update(checkpoint_sha256=SHA["arm2_r4"]))
    edit_json(v / "runs/arm2_selection.json", lambda d: d["r8"].update(merged_sha256=SHA["arm2_r4"]))
    rc, out = run(RT, v)
    check(rc == 0 and "neither rank beat its epoch-0" in (v / "arms_summary.md").read_text(),
          "identical r4/r8 checkpoints are accepted (and noted) when both best epochs are 0")
    edit_json(v / "runs/arm2_r8/run.json", lambda d: d.update(best_epoch=12))
    rc, out = run(RT, v)
    check(rc == 1 and "identical checkpoints" in out, "...but refused when a best epoch is not 0")
    v = variant(base, tmp, "nsd_off")
    for r in P.RANKS:
        edit_csv(v / f"runs/arm2_r{r}/metrics.csv", lambda rs: [dict(x, **{"nsd3_ct@0.10": "nan", "nsd3_mri@0.10": "nan"}) for x in rs])
        edit_json(v / f"runs/arm2_r{r}/run.json", lambda d: d["settings"].update(nsd=False))
        edit_json(v / f"runs/arm2_r{r}/config.json", lambda d: d["settings"].update(nsd=False))
    rc, out = run(CT, v)
    check(rc == 0, "NaN NSD columns are accepted when the run did not compute NSD")
    v = variant(base, tmp, "rt_nocmp")
    shutil.rmtree(v / "outputs/arm1_vit_b/compare_vs_arm1")
    rc, out = run(RT, v)
    check(rc == 0 and "not available" in (v / "arms_summary.md").read_text(),
          "a missing optional compare is reported as 'not available', not a crash")


def test_plot_checks(base: Path, tmp: Path) -> None:
    print("\n--- plot_results cross-checks ---")
    sel = json.loads((base / "runs" / "arm2_selection.json").read_text())["chosen_rank"]
    chosen = f"arm2_r{sel}"
    v = variant(base, tmp, "pl_tamper")
    edit_csv(v / f"outputs/{chosen}/instances.csv",
             lambda rs: [dict(r, dice=str(float(r["dice"]) + 0.2)) if (r["dataset"] == "word" and r["jitter"] == "0.1")
                         else r for r in rs])
    rc, out = run(PR, v)
    check(rc == 1 and "per-patient mean" in out, "(c) refuses when per-patient deltas disagree with compare.csv")
    v = variant(base, tmp, "pl_box")
    edit_csv(v / f"outputs/{chosen}/instances.csv",
             lambda rs: [dict(r, box_jit="[0, 0, 1, 1]") if i == 3 else r for i, r in enumerate(rs)])
    rc, out = run(PR, v)
    check(rc == 1 and "box_jit" in out, "(c) refuses when the two arms' prompts differ")
    v = variant(base, tmp, "pl_noinst")             # carries the 4 figures of the happy-path run
    (v / f"outputs/{chosen}/instances.csv").unlink()
    rc, out = run(PR, v)
    figs = sorted(p.name for p in (v / "figures").glob("*.png"))
    check(rc == 0 and "skipped" in out and len(figs) == 3 and "patient_delta.png" not in figs,
          "(c) is skipped cleanly without instances.csv, and no stale patient_delta.png is left behind")
    v = variant(base, tmp, "pl_round")
    path = v / f"outputs/{chosen}/compare_vs_arm1_vit_b/compare.csv"
    edit_csv(path, lambda rs: [dict(r, delta=str(round(float(r["delta"]) + 1e-4, 4)))
                               if (r["level"], r["group"], r["metric"], r["jitter"]) == ("2d", "word/ct", "dice", "0.1")
                               else r for r in rs])
    rc, out = run(PR, v)
    check(rc == 0, "a delta off by 1e-4 (4-decimal rounding) is accepted")
    edit_csv(path, lambda rs: [dict(r, delta=str(round(float(r["delta"]) + 2e-4, 4)))
                               if (r["level"], r["group"], r["metric"], r["jitter"]) == ("2d", "word/ct", "dice", "0.1")
                               else r for r in rs])
    rc, out = run(PR, v)
    check(rc == 1, "a delta off by 3e-4 in total is refused")


def test_fetch(tmp: Path) -> None:
    print("\n--- fetch_results.sh (real rsync on a local fake cluster tree) ---")
    check(subprocess.run(["bash", "-n", str(HERE / "fetch_results.sh")]).returncode == 0, "fetch_results.sh passes bash -n")
    seg = tmp / "seg"
    for a, _ in P.ARMS:
        for f in ("summary.csv", "summary.md", "breakdowns.csv", "run.json", "model.json", "spacing_report.csv",
                  "instances.csv", "volumes_3d.csv", "qc/x.png", "shards/0_of_4/run.json", "compare_vs_z/compare.csv",
                  "compare_vs_z/compare.md"):
            (seg / "outputs" / a / f).parent.mkdir(parents=True, exist_ok=True)
            (seg / "outputs" / a / f).write_text("x")
    for r in P.RANKS:
        for f in ("metrics.csv", "run.json", "config.json", "state.json", "last.pt", "best.pt"):
            (seg / "runs" / f"arm2_r{r}" / f).parent.mkdir(parents=True, exist_ok=True)
            (seg / "runs" / f"arm2_r{r}" / f).write_text("x")
    (seg / "runs/arm2_preflight").mkdir(parents=True)
    (seg / "runs/arm2_preflight/preflight.json").write_text("x")
    (seg / "runs/arm2_selection.json").write_text("x")
    (seg / "checkpoints").mkdir()
    (seg / "checkpoints/sam_vit_b_arm2_r4.pth").write_text("x")
    (seg / "data/processed").mkdir(parents=True)
    (seg / "data/processed/manifest.csv").write_text("x")
    (seg / "outputs/arm2_r4_train.1.out").write_text("x")
    proj = tmp / "proj"
    (proj / "Code").mkdir(parents=True)
    shutil.copy(HERE / "fetch_results.sh", proj / "Code")
    env = {**__import__("os").environ, "FETCH_SRC": str(seg) + "/"}
    out = subprocess.run(["bash", str(proj / "Code/fetch_results.sh"), "--dry-run"], env=env, capture_output=True, text=True)
    check(out.returncode == 0 and "--include=/outputs/arm1/run.json" in out.stdout and ".pt" not in out.stdout.split("would run:")[1].split("\n")[1],
          "dry run prints the anchored rules and copies nothing")
    check(not (proj / "results" / "outputs").exists(), "dry run wrote no files")
    subprocess.run(["bash", str(proj / "Code/fetch_results.sh")], env=env, capture_output=True, text=True, check=True)
    got = sorted(str(p.relative_to(proj / "results")) for p in (proj / "results").rglob("*") if p.is_file())
    check(len(got) == 42 and not any(x.endswith((".pt", ".pth", ".png", "volumes_3d.csv", "instances.csv", ".out"))
                                     or "shards" in x or "data/" in x for x in got),
          f"copies exactly the 42 small files; no checkpoints, QC, shards, volumes, instances or logs ({len(got)})")
    subprocess.run(["bash", str(proj / "Code/fetch_results.sh"), "--with-instances"], env=env, capture_output=True,
                   text=True, check=True)
    inst = sorted(str(p.relative_to(proj / "results")) for p in (proj / "results").rglob("instances.csv"))
    check(inst == ["outputs/arm1_vit_b/instances.csv", "outputs/arm2_r4/instances.csv", "outputs/arm2_r8/instances.csv"],
          "--with-instances adds exactly the three instances.csv files")
    bad = subprocess.run(["bash", str(proj / "Code/fetch_results.sh"), "--nope"], env=env, capture_output=True, text=True)
    check(bad.returncode == 2, "an unknown option is refused")


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="post_test_"))
    base = tmp / "results"
    build_tree(base)
    test_environment()
    test_happy(base)
    test_check_training_refusals(base, tmp)
    test_table_refusals(base, tmp)
    test_plot_checks(base, tmp)
    test_fetch(tmp)
    print(f"\nscratch: {tmp}\n\nALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
