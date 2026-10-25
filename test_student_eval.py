#!/usr/bin/env python
"""CPU tests for the arm-3 test evaluation (eval_student.py, seed_summary.py,
gen_sbatch_arm3_eval.py) -- ARM3_EVAL_PLAN.md §7.

    python test_student_eval.py        # exits 0 and prints ALL TESTS PASSED
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import aggregate_arm  # noqa: E402
import compare_arms  # noqa: E402
import eval_sam as E  # noqa: E402
import eval_student as ES  # noqa: E402
import sam_metrics as M  # noqa: E402
import seed_summary as SS  # noqa: E402
import student_core as C  # noqa: E402
import student_data as D  # noqa: E402
import student_unet as U  # noqa: E402
import test_student as TS  # noqa: E402
import train_student as T  # noqa: E402

torch.set_num_threads(1)
check, raises = TS.check, TS.raises
NARROW, FRAME = TS.NARROW, TS.FRAME
CASES = [("train", "ct", "amos_0001", (128, 128)), ("train", "mri", "amos_0501", (96, 128)),
         ("val", "ct", "amos_0011", (128, 128)), ("val", "mri", "amos_0511", (60, 100)),
         ("test", "ct", "amos_0021", (128, 128)), ("test", "ct", "amos_0022", (128, 128)),
         ("test", "mri", "amos_0521", (96, 128)), ("external", "ct", "amos_0031", (160, 160))]


def quiet(fn, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = fn(*a, **k)
    return r, buf.getvalue()


class Const(torch.nn.Module):
    """Predicts one class everywhere (for hand-countable 3D / false-positive checks)."""
    def __init__(self, cls: int):
        super().__init__()
        self.cls = cls
        self.p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        out = torch.zeros(x.shape[0], 3, x.shape[2], x.shape[3])
        out[:, self.cls] = 5.0
        return out


def setup(root: Path):
    TS.CASES = CASES
    proc = root / "processed"
    exp = TS.build_processed(proc)
    rows = E.load_manifest(proc)
    cases = E.group_cases(rows)
    E.EXPECTED_GROUPS = E.count_groups(cases, sorted(cases))          # synthetic D22a anchors
    with contextlib.redirect_stdout(io.StringIO()):
        va = D.SliceStore(proc, "val", exp, 2)
    return proc, exp, rows, cases, va


def make_run(root: Path, proc: Path, exp: dict, va, variant: str, seed: int, model=None,
             best_score: float | None = None) -> Path:
    """A finished training-run folder: best.pt + run.json (TRAIN_OK) whose best_score is
    what the model really scores on val (so the evaluation's val re-check passes)."""
    rd = root / "runs" / f"arm3_{variant}_s{seed}"
    rd.mkdir(parents=True)
    norm = C.Norm(40.0, 50.0)
    torch.manual_seed(seed)
    net = U.StudentUNet(C.in_channels(variant), channels=NARROW)
    torch.save({"model": net.state_dict(), "arch": {"in_ch": C.in_channels(variant), "channels": list(NARROW),
                                                    "n_classes": 3, "deep_supervision": True},
                "variant": variant, "norm": norm.as_dict(), "size": FRAME, "epoch": 2, "score": 0.0,
                "seed": seed}, rd / "best.pt")
    m = model
    if m is None:
        m, _, _, _, _ = C.load_student(rd / "best.pt")
    if best_score is None:
        items = D.ValItems(va, variant, norm, FRAME)
        best_score = T.validate(m, items, T.val_loader(items, T.Cfg(val_workers=0, val_batch=4)), "cpu",
                                T.select_level_of(variant))["selection"]
    (rd / "run.json").write_text(json.dumps({
        "status": "TRAIN_OK", "epochs_completed": 2, "settings": {"epochs": 2}, "variant": variant,
        "seed": seed, "best_epoch": 2, "best_score": best_score, "best_sha256": ES.sha256_file(rd / "best.pt"),
        "manifest_sha256": D.manifest_sha256(proc), "params": U.count_params(m)}))
    return rd


def args_for(proc, rd, out, **kw):
    a = dict(processed=proc, run_dir=rd, out=out, device="cpu", expect_gpu=None, resume=False,
             overwrite=False, limit=None, qc_per_group=4, val_workers=0)
    a.update(kw)
    return argparse.Namespace(**a)


def evaluate(proc, rd, out, exp, factory=None, **kw):
    return quiet(ES.run, args_for(proc, rd, out, **kw), model_factory=factory, val_expected_counts=exp)


def read(p: Path) -> list:
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


# ============================================================================ tests
def test_format_and_path(root, proc, exp, rows, cases, va):
    print("\n--- 1-3. format, prediction path, jitter semantics ---")
    n_inst = sum((r["rk_box"] is not None) + (r["lk_box"] is not None) for r in rows)
    out = {}
    for variant, seed in (("plain", 643), ("box", 643)):
        rd = make_run(root, proc, exp, va, variant, seed)
        o = root / "outputs" / f"arm3_{variant}_s{seed}"
        (rc, log) = evaluate(proc, rd, o, exp)
        check(rc == 0 and "EVAL_OK" in log, f"{variant}: EVAL_OK")
        out[variant] = (rd, o)
        inst, vols = read(o / "instances.csv"), read(o / "volumes_3d.csv")
        with open(o / "instances.csv") as fh:
            header = next(csv.reader(fh))
        check(header == E.INSTANCE_FIELDS and len(inst) == n_inst * 4 and
              not E.verify_rows(inst, vols, cases, {(r["case"], r["z"]): r for r in rows}),
              f"{variant}: eval_sam's format and verify_rows; {n_inst} instances x 4 levels")
        check(all(np.isnan(float(r["sam_iou"])) for r in inst), f"{variant}: sam_iou is NaN (no IoU head)")
        model, norm, var, size, _ = C.load_student(rd / "best.pt")
        mk = {(r["case"], r["z"]): r for r in rows}
        ok = True
        for r in inst:
            m = mk[(r["case"], int(r["z"]))]
            side = M.RIGHT if r["side"] == "right" else M.LEFT
            with np.load(proc / m["path"]) as d:
                img, msk = d["img"], d["msk"]
            j = float(r["jitter"])
            boxes = None
            if variant == "box":
                boxes = {s: (None if m[f"{p}_box"] is None else
                             M.jitter_box(tuple(m[f"{p}_box"]), j, m["h"], m["w"], M.jitter_seed(r["case"], m["z"], s, j)))
                         for s, p in E.SIDES}
            lab, _ = C.predict_native(model, img, norm, variant, boxes, "cpu", size)
            s = M.score_masks(msk == side, lab == side, (m["sp_row"], m["sp_col"]))
            ok &= abs(round(s["dice"], 6) - float(r["dice"])) < 1e-9 and s["pred_vox"] == int(r["pred_area"])
        check(ok, f"{variant}: every stored score == an independent predict_native + score_masks (same path as validation)")
    by = {}
    for r in read(out["plain"][1] / "instances.csv"):
        by.setdefault((r["case"], r["z"], r["side"]), set()).add((r["dice"], r["pred_area"]))
    check(all(len(v) == 1 for v in by.values()), "plain: identical prediction at all 4 jitter levels")
    by = {}
    for r in read(out["box"][1] / "instances.csv"):
        by.setdefault((r["case"], r["z"], r["side"]), set()).add(r["pred_area"])
    check(any(len(v) > 1 for v in by.values()), "box: predictions change with the jittered boxes")
    rj = json.loads((out["plain"][1] / "run.json").read_text())
    # Batch size changes float rounding in the convolutions (val re-check uses 16, the test's
    # reference 4, QC re-runs 1), which can flip a pixel on an exact class tie: ~1e-7 in Dice.
    check(rj["val_recheck"]["ok"] and rj["val_recheck"]["abs_diff"] < 1e-5 and rj["qc_max_dice_diff"] < 1e-5
          and (out["plain"][1] / "qc").is_dir() and rj["speed"].get("n_slices", 0) > 0,
          f"val re-check |diff| {rj['val_recheck']['abs_diff']:.1e}, QC re-run |diff| {rj['qc_max_dice_diff']:.1e} "
          f"(batch-size rounding only); speed recorded")
    check((out["plain"][1] / "spacing_report.csv").exists() and rj["coarsest_in_plane_mm"] > 0,
          "spacing_report.csv written and the NSD-tolerance check applied, as eval_sam")
    return out


def test_3d_and_fp(root, proc, exp, rows, cases, va):
    print("\n--- 4-5. 3D rule, whole volume, false positives ---")
    res = {}
    for cls, seed in ((1, 701), (0, 702)):
        model = Const(cls)
        rd = make_run(root, proc, exp, va, "plain", seed, model=model)
        o = root / "outputs" / f"const{cls}"
        (rc, log) = evaluate(proc, rd, o, exp, factory=lambda m=model: (m, C.Norm(40.0, 50.0), "plain", FRAME, 1))
        check(rc == 0, f"constant-{cls} model: EVAL_OK")
        res[cls] = o
    vols = {(r["case"], r["side"], r["jitter"]): r for r in read(res[1] / "volumes_3d.csv")}
    full = {(r["case"], r["side"]): r for r in read(res[1] / "volumes_3d_full.csv")}
    ok_rule = ok_full = True
    for c, rs in cases.items():
        h, w, nz = rs[0]["h"], rs[0]["w"], len(rs)
        for side, prefix in E.SIDES:
            if not any(r[f"{prefix}_area"] > 0 for r in rs):
                continue
            boxed = sum(1 for r in rs if r[f"{prefix}_box"] is not None)
            want = boxed * h * w if side == M.RIGHT else 0
            ok_rule &= all(int(vols[(c, M.SIDE_NAME[side], f"{j}")]["pred_vox"]) == want for j in M.JITTER_LEVELS)
            ok_full &= int(full[(c, M.SIDE_NAME[side])]["pred_vox"]) == (nz * h * w if side == M.RIGHT else 0)
    check(ok_rule, "volumes_3d.csv: a side's prediction counted only on that side's boxed slices (SAM's rule)")
    check(ok_full, "volumes_3d_full.csv: every slice of the volume counted")
    fp1 = json.loads((res[1] / "run.json").read_text())["fp_kidney_free_slices"]
    fp0 = json.loads((res[0] / "run.json").read_text())["fp_kidney_free_slices"]
    n_free = sum(int(r["has_kidney"]) == 0 for r in rows)
    check(sum(v["kidney_free_slices"] for v in fp1.values()) == n_free and all(v["fp_rate"] == 1.0 for v in fp1.values())
          and all(v["fp_rate"] == 0.0 for v in fp0.values()),
          f"false-positive rate: 1.0 for an all-kidney model, 0.0 for an all-background model ({n_free} kidney-free slices)")


def test_downstream(root, proc, exp, rows, cases, va, out):
    print("\n--- 6. downstream tools unchanged ---")
    o_plain = out["plain"][1]
    rc, _ = quiet(aggregate_arm.main, ["--arm-dir", str(o_plain), "--label", "test", "--n-boot", "200"])
    check(rc == 0 and (o_plain / "summary.csv").exists(), "aggregate_arm.py runs on a student folder")
    # a SAM-format reference run: the real eval_sam.main with a stub SAM on the same data
    import test_eval_sam as TE
    stub = TE.StubPredictor({})
    orig = E.build_predictor
    E.build_predictor = lambda *a, **k: stub
    ck = root / "stub_vit_b.pth"
    ck.write_bytes(b"stub")
    ref = root / "outputs" / "ref_sam"
    try:
        rc, _ = quiet(E.main, ["--processed", str(proc), "--checkpoint", str(ck), "--model-type", "vit_b",
                               "--out", str(ref), "--device", "cpu", "--qc-per-group", "2"])
    finally:
        E.build_predictor = orig
    check(rc == 0, "reference: eval_sam.py with a stub SAM on the same synthetic set")
    rc, log = quiet(compare_arms.main, ["--a", str(ref), "--b", str(o_plain), "--n-boot", "200"])
    check(rc == 0 and "COMPARE_OK" in log, "compare_arms.py: SAM-format run vs student -> identical keys and boxes, COMPARE_OK")
    rc, log = quiet(compare_arms.main, ["--a", str(o_plain), "--b", str(out["box"][1]), "--n-boot", "200"])
    check(rc == 0, "compare_arms.py: plain vs box student (the D8 gap)")
    seeds = [o_plain]
    for s in (644, 645):
        rd = make_run(root, proc, exp, va, "plain", s)
        o = root / "outputs" / f"arm3_plain_s{s}"
        rc, _ = evaluate(proc, rd, o, exp)
        seeds.append(o)
    so = root / "outputs" / "arm3_seeds"
    rc, log = quiet(SS.main, ["--runs", *map(str, seeds), "--ref", str(ref), "--out", str(so), "--n-boot", "200"])
    check(rc == 0 and "SUMMARY_OK" in log, "seed_summary.py: SUMMARY_OK")
    t1 = {(r["level"], r["metric"], r["group"]): r for r in read(so / "seed_table.csv")}
    t2 = [r for r in read(so / "seed_compare_vs_ref_sam.csv") if r["level"] == "2d" and r["metric"] == "dice"
          and abs(float(r["jitter"]) - 0.10) < 1e-9]
    pm = []
    for o in seeds:
        acc = {}
        for r in read(o / "instances.csv"):
            if abs(float(r["jitter"]) - 0.10) < 1e-9:
                g = "/".join(x for x in (r["dataset"], r["modality"], r["sequence"]) if x)
                acc.setdefault(g, {}).setdefault(r["patient"], []).append(float(r["dice"]))
        pm.append({g: {p: sum(v) / len(v) for p, v in d.items()} for g, d in acc.items()})
    ref_pm = {}
    for r in read(ref / "instances.csv"):
        if abs(float(r["jitter"]) - 0.10) < 1e-9:
            g = "/".join(x for x in (r["dataset"], r["modality"], r["sequence"]) if x)
            ref_pm.setdefault(g, {}).setdefault(r["patient"], []).append(float(r["dice"]))
    ok = True
    for row in t2:
        g = row["group"]
        pats = sorted(ref_pm[g])
        seed_scores = [sum(p_[g][p] for p in pats) / len(pats) for p_ in pm]
        want_mean = sum(seed_scores) / len(seed_scores)
        ok &= abs(float(t1[("2d", "dice", g)]["mean"]) - round(want_mean, 4)) < 1e-9
        want_d = sum(sum(p_[g][p] for p_ in pm) / len(pm) - sum(ref_pm[g][p]) / len(ref_pm[g][p]) for p in pats) / len(pats)
        ok &= abs(float(row["delta"]) - round(want_d, 4)) < 1e-9
    check(ok, "seed table mean and seed-averaged Δ equal hand computation")
    bad = root / "outputs" / "broken"
    shutil.copytree(seeds[1], bad)
    rr = read(bad / "instances.csv")
    rr[0]["box_jit"] = "[0, 0, 1, 1]"
    with open(bad / "instances.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=E.INSTANCE_FIELDS)
        w.writeheader()
        w.writerows(rr)
    rc, _ = quiet(SS.main, ["--runs", str(seeds[0]), str(bad), "--ref", str(ref), "--out", str(so), "--n-boot", "50"])
    check(rc == 1, "seed_summary refuses a seed whose prompts differ from the reference")
    dup = root / "outputs" / "dup_of_s643"
    shutil.copytree(seeds[0], dup)
    rc, log = quiet(SS.main, ["--runs", str(seeds[0]), str(dup), "--ref", str(ref), "--out", str(so), "--n-boot", "50"])
    check(rc == 1 and "distinct models" in log, "seed_summary refuses the same model counted as two seeds")
    rc, log = quiet(SS.main, ["--runs", str(seeds[0]), str(seeds[1]), "--ref", str(seeds[0]), "--out", str(so), "--n-boot", "50"])
    check(rc == 1 and "reference is one of the seed runs" in log, "seed_summary refuses a reference that is also a seed")
    rc, log = quiet(SS.main, ["--runs", str(seeds[0]), str(out["box"][1]), "--ref", str(ref), "--out", str(so), "--n-boot", "50"])
    check(rc == 1 and "plain student" in log, "seed_summary refuses a box run among the plain seeds")


def test_guards_resume(root, proc, exp, rows, cases, va, out):
    print("\n--- 7-9. guards, resume ---")
    rd, o = out["plain"]
    tmp_rd = root / "g" / "arm3_plain_s643"
    shutil.copytree(rd, tmp_rd)
    rj = json.loads((tmp_rd / "run.json").read_text())

    def with_rj(**kw):
        (tmp_rd / "run.json").write_text(json.dumps({**rj, **kw}))
        return lambda: ES.run(args_for(proc, tmp_rd, root / "g" / "out_x"), val_expected_counts=exp)
    check(raises(with_rj(best_sha256="0" * 64)), "best.pt sha256 != run.json -> refused")
    check(raises(with_rj(status="RUNNING")), "not TRAIN_OK -> refused")
    check(raises(with_rj(seed=999)), "folder name / seed mismatch -> refused")
    check(raises(with_rj(manifest_sha256="1" * 64)), "manifest changed since training -> refused")
    check(raises(with_rj(best_score=rj["best_score"] + 0.01)), "val re-check off by 0.01 -> refused")
    (tmp_rd / "run.json").write_text(json.dumps(rj))
    other = root / "g" / "out_other"
    other.mkdir(parents=True)
    (other / "model.json").write_text(json.dumps({"model_type": "student_plain", "checkpoint_name": "x/best.pt",
                                                  "checkpoint_bytes": 1, "checkpoint_sha256": "2" * 64}))
    check(raises(lambda: ES.run(args_for(proc, tmp_rd, other), val_expected_counts=exp)),
          "an output folder stamped by another model -> refused")
    check(raises(lambda: ES.run(args_for(proc, rd, o), val_expected_counts=exp)),
          "existing results without --resume -> refused")
    # resume: a run killed mid-case (instances cut mid-line, last fp row lost)
    o2 = root / "g" / "resume"
    shutil.copytree(o, o2)
    for f in ("instances.csv", "volumes_3d.csv"):
        t = (o2 / f).read_text()
        (o2 / f).write_text(t[: int(len(t) * 0.6)])
    t = (o2 / "fp_slices.csv").read_text().splitlines(keepends=True)
    (o2 / "fp_slices.csv").write_text("".join(t[:-1]))
    rc, log = evaluate(proc, rd, o2, exp, resume=True)
    same = all(sorted(map(tuple, (r.values() for r in read(o2 / f)))) == sorted(map(tuple, (r.values() for r in read(o / f))))
               for f in ("instances.csv", "volumes_3d.csv", "volumes_3d_full.csv", "fp_slices.csv"))
    check(rc == 0 and "resume:" in log and same, "resume after a mid-write kill: final files == uninterrupted run")


def test_sbatch(root):
    print("\n--- 10. SLURM scripts ---")
    names = ["arm3_compare.sbatch", "arm3_eval_a.sbatch", "arm3_eval_b.sbatch"]
    regen = root / "regen"
    regen.mkdir(parents=True)
    subprocess.run([sys.executable, str(HERE / "gen_sbatch_arm3_eval.py"), str(regen)], check=True, capture_output=True)
    check(sorted(p.name for p in regen.glob("*.sbatch")) == names, "generator writes exactly the 3 scripts")
    for n in names:
        txt = (HERE / n).read_text()
        check(txt == (regen / n).read_text() and subprocess.run(["bash", "-n", str(HERE / n)]).returncode == 0
              and "#SBATCH --partition=a40" in txt and "*A40*" in txt, f"{n}: == generator, bash -n, a40/A40")
    stub, calls = root / "stub", root / "calls.txt"
    stub.mkdir()
    (stub / "squeue").write_text('#!/bin/bash\necho "${SQUEUE_OUT:-}"\n')
    (stub / "nvidia-smi").write_text('#!/bin/bash\necho "${GPU_NAME:-NVIDIA A40}"\n')
    (stub / "py").write_text('#!/bin/bash\nif [ "$1" = "-c" ]; then echo env-ok; exit 0; fi\n'
                             f'echo "$@" >> {calls}\n'
                             'if [ -n "${FAIL_MATCH:-}" ] && echo "$@" | grep -q -- "$FAIL_MATCH"; then exit 7; fi\nexit 0\n')
    for f in stub.iterdir():
        f.chmod(f.stat().st_mode | stat.S_IEXEC)
    seg = root / "seg"
    for d in ("data/processed", "outputs/arm1_vit_b", "outputs/arm2_r4", "outputs/arm3_plain_s643", "outputs/arm3_box_s643"):
        (seg / d).mkdir(parents=True)
    (seg / "data/processed/manifest.csv").write_text("x\n")
    for d in ("arm1_vit_b", "arm2_r4", "arm3_plain_s643", "arm3_box_s643"):
        (seg / "outputs" / d / "run.json").write_text("{}")

    def run(name, **env):
        t = (HERE / name).read_text().replace("SEG=/home/rethinkingai-self/vihaan/Segmentation", f"SEG={seg}")
        t = t.replace("PY=/home/rethinkingai-self/vihaan/miniconda3/envs/seg311/bin/python", f"PY={stub / 'py'}")
        sp = root / "run.sh"
        sp.write_text(t)
        calls.unlink(missing_ok=True)
        r = subprocess.run(["bash", str(sp)], env={**os.environ, "PATH": f"{stub}:{os.environ['PATH']}", **env},
                           capture_output=True, text=True)
        return r.returncode, (calls.read_text().splitlines() if calls.exists() else []), r.stdout
    rc, cl, _ = run("arm3_eval_a.sbatch")
    steps = [c.split()[0].rsplit("/", 1)[-1] for c in cl]
    check(rc == 0 and steps == ["eval_student.py", "aggregate_arm.py", "compare_arms.py", "compare_arms.py"] * 2
          and "arm3_plain_s643" in cl[0] and "--resume" in cl[0] and "--expect-gpu A40" in cl[0] and "arm3_plain_s644" in cl[4],
          "eval_a: (eval -> aggregate -> 2 compares) for s643 then s644")
    rc, cl, out = run("arm3_eval_a.sbatch", FAIL_MATCH="arm3_plain_s643 --out")
    check(rc == 7 and len(cl) == 1 and "STOPPING" in out, "a failing evaluation stops the script")
    rc, cl, out = run("arm3_eval_b.sbatch", FAIL_MATCH="--a .*arm2_r4")
    check(rc == 5 and sum("eval_student.py" in c for c in cl) == 2 and "a comparison FAILED" in out,
          "a failing comparison: next run still evaluated, script ends with exit 5")
    rc, cl, _ = run("arm3_compare.sbatch")
    check(rc == 0 and sum("seed_summary.py" in c for c in cl) == 2 and "arm3_box_s643" in cl[-1],
          "compare script: 2 seed summaries, then box vs plain s643")
    rc, cl, _ = run("arm3_eval_a.sbatch", SQUEUE_OUT="9 arm3_ea RUNNING")
    check(rc == 3 and not cl, "second copy refused")
    rc, cl, _ = run("arm3_eval_b.sbatch", GPU_NAME="NVIDIA A100-SXM4-80GB")
    check(rc == 4 and not cl, "non-A40 node refused")


def test_isolation():
    print("\n--- 11. isolation ---")
    code = ("import sys; sys.path.insert(0, %r); import eval_student, seed_summary; "
            "bad = [m for m in ('segment_anything', 'train_sam_lora', 'sam_lora') if m in sys.modules]; "
            "print(bad); sys.exit(1 if bad else 0)") % str(HERE)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    check(r.returncode == 0, f"eval_student / seed_summary import no SAM code {r.stdout.strip()}")


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="arm3_eval_test_"))
    groups = dict(E.EXPECTED_GROUPS)
    try:
        proc, exp, rows, cases, va = setup(root)
        out = test_format_and_path(root, proc, exp, rows, cases, va)
        test_3d_and_fp(root, proc, exp, rows, cases, va)
        test_downstream(root, proc, exp, rows, cases, va, out)
        test_guards_resume(root, proc, exp, rows, cases, va, out)
        test_sbatch(root / "sb")
        test_isolation()
    finally:
        E.EXPECTED_GROUPS = groups
        shutil.rmtree(root, ignore_errors=True)
    print("\nALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
