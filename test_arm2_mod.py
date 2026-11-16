#!/usr/bin/env python
"""Tests for the single-modality teachers (ARM2_MOD_CODE_PLAN.md §6, DECISIONS.md D70-D71):
modality filter, uniform draws, selection metric, run identity, CT / MRI training end to end,
resume, finalize / check-merged guards, the joint arm-2 path unchanged, and the generated SLURM
scripts. CPU only, the tiny SAM and synthetic processed/ folder of test_train_sam.py.

    python test_arm2_mod.py        # exits 0 and prints ALL TESTS PASSED

On Prajna it also checks the real manifest (counts, steps, draws) and the joint arm 2's real
config.json stamp; elsewhere those checks print "skipped".
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_train_sam as TT  # noqa: E402  (fixture, tiny SAM, helpers; its main() does not run)
import sam_train_core as C  # noqa: E402
import train_data as D  # noqa: E402
import train_sam_lora as T  # noqa: E402

check, raises, quiet = TT.check, TT.raises, TT.quiet
SEG_REAL = Path("/home/rethinkingai-self/vihaan/Segmentation")
SCRIPTS = ["arm2_ct_train.sbatch", "arm2_mri_train.sbatch", "arm2_ct_eval.sbatch", "arm2_mri_eval.sbatch"]

# --------------------------------------------------------------- frozen copies of the old code
# Verbatim from train_data.py / sam_train_core.py / train_sam_lora.py before D71 (the code that
# trained arm 2). The joint path must reproduce them exactly.


def old_make_epoch_draws(ds, epoch, seed, n_draws=None):
    n_draws = len(ds) if n_draws is None else n_draws
    n_mod = Counter(r["modality"] for r in ds.rows)
    if set(n_mod) != {"ct", "mri"}:
        raise ValueError(f"need both modalities to balance, have {dict(n_mod)}")
    w = torch.tensor([1.0 / n_mod[r["modality"]] for r in ds.rows], dtype=torch.float64)
    g = torch.Generator().manual_seed(int(seed) + int(epoch))
    idx = torch.multinomial(w, n_draws, replacement=True, generator=g).tolist()
    return list(enumerate(idx))


def old_val_metric(rows, levels=(0.0, 0.10), select_level=0.10):
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[(r["modality"], round(float(r["level"]), 4))][r["patient"]].append(float(r["dice"]))
    out = {}
    for (mod, lev), pats in by.items():
        pm = [sum(v) / len(v) for v in pats.values()]
        out[f"{mod}@{lev:.2f}"] = sum(pm) / len(pm)
        out[f"n_patients_{mod}@{lev:.2f}"] = len(pm)
        out[f"n_inst_{mod}@{lev:.2f}"] = sum(len(v) for v in pats.values())
    need = [f"{m}@{round(select_level, 4):.2f}" for m in ("ct", "mri")]
    missing = [k for k in need if k not in out]
    if missing:
        raise ValueError(f"val_metric: no rows for {missing}")
    out["selection"] = (out[need[0]] + out[need[1]]) / 2.0
    return out


# stamp_of(Cfg()) computed with the pre-D71 train_sam_lora.py (2026-11-16)
OLD_DEFAULT_STAMP = json.loads(
    '{"accum": 1, "alpha": 4.0, "amp": true, "batch": 8, "betas": [0.9, 0.999], "check_base_sha": true,'
    ' "clip": 1.0, "device": "cuda", "epochs": 35, "expect_blocks": 12, "expect_dim": 768, "expect_gpu": null,'
    ' "expected_slices": {"train": {"ct": 5760, "mri": 2535}, "val": {"ct": 1901, "mri": 843}},'
    ' "grad_checkpoint": false, "img_size": 1024, "lr_dec": 0.0001, "lr_floor": 1e-06, "lr_lora": 0.0005,'
    ' "max_level": 0.2, "n_draws": null, "nsd": true, "rank": 4, "seed": 643, "select_level": 0.1,'
    ' "val_batch": 8, "val_levels": [0.0, 0.1], "warmup": 250, "weight_decay": 0.01}')


def arm2_cfg(**kw) -> T.Cfg:
    """The Cfg main() builds from arm2_r4_train.sbatch's arguments."""
    base = dict(rank=4, alpha=4.0, epochs=35, batch=8, accum=1, lr_lora=5e-4, lr_dec=1e-4, seed=643,
                workers=10, device="cuda", grad_checkpoint=False, expect_gpu="A100")
    base.update(kw)
    return T.Cfg(**base)


# ------------------------------------------------------------------------------------ tests
def test_dataset(proc: Path) -> None:
    print("\n--- T1. SliceDataset modality filter ---")
    both = D.SliceDataset(proc, "train", expected=None)
    ct = D.SliceDataset(proc, "train", expected=None, modality="ct")
    mri = D.SliceDataset(proc, "train", expected=None, modality="mri")
    check(both.modality is None and set(both.counts) == {"ct", "mri"}, "T1. modality=None keeps both modalities")
    check(all(r["modality"] == "ct" for r in ct.rows) and ct.modality == "ct" and len(ct) > 0,
          f"T1. ct keeps only CT rows ({len(ct)})")
    check(all(r["modality"] == "mri" for r in mri.rows) and len(mri) > 0, f"T1. mri keeps only MRI rows ({len(mri)})")
    check([r for r in both.rows if r["modality"] == "ct"] == ct.rows
          and [r for r in both.rows if r["modality"] == "mri"] == mri.rows,
          "T1. the filtered rows are exactly the joint rows of that modality, in the same order")
    check(raises(lambda: D.SliceDataset(proc, "train", expected=None, modality="pet")), "T1. an unknown modality is refused")
    check(raises(lambda: D.SliceDataset(proc, "test", expected=None, modality="ct")),
          "T1. the test split is still refused with a modality")
    n = len(ct)
    exp_ok = {"train": {"ct": n, "mri": 999}, "val": {"ct": 1, "mri": 1}}
    check(len(D.SliceDataset(proc, "train", expected=exp_ok, modality="ct")) == n,
          "T1. the expected count is checked for the run's modality only")
    exp_bad = {"train": {"ct": n + 1, "mri": len(mri)}, "val": {"ct": 1, "mri": 1}}
    check(raises(lambda: D.SliceDataset(proc, "train", expected=exp_bad, modality="ct")),
          "T1. a wrong expected count for the modality is refused")


def test_draws(proc: Path) -> None:
    print("\n--- T2, T3, T4. epoch draws and steps ---")
    both = D.SliceDataset(proc, "train", expected=None)
    same = all(D.make_epoch_draws(both, e, s, 8295) == old_make_epoch_draws(both, e, s, 8295)
               for s in (643, 644, 7) for e in (0, 1, 35))
    check(same, "T2. joint draws == the frozen pre-D71 function (3 seeds x 3 epochs, 8,295 draws)")
    check(D.make_epoch_draws(both, 3, 643) == old_make_epoch_draws(both, 3, 643),
          "T2. ...also with the default draw count")
    for mod in ("ct", "mri"):
        ds = D.SliceDataset(proc, "train", expected=None, modality=mod)
        dr = D.make_epoch_draws(ds, 1, 643)
        check(len(dr) == len(ds) and [i for i, _ in dr] == list(range(len(ds))),
              f"T3. {mod}: default draws per epoch = number of {mod} slices ({len(ds)})")
        check(all(ds.rows[j]["modality"] == mod for _, j in dr), f"T3. {mod}: every draw is a {mod} slice")
        check(D.make_epoch_draws(ds, 1, 643) == dr and D.make_epoch_draws(ds, 2, 643) != dr,
              f"T3. {mod}: deterministic in (seed, epoch), changes with the epoch")
        g = torch.Generator().manual_seed(643 + 1)
        ref = torch.multinomial(torch.ones(len(ds), dtype=torch.float64), len(ds), replacement=True, generator=g).tolist()
        check([j for _, j in dr] == ref, f"T3. {mod}: draws are uniform (== multinomial over equal weights)")
        big = D.make_epoch_draws(ds, 1, 643, 20000)
        freq = Counter(j for _, j in big)
        exp = 20000 / len(ds)
        check(len(freq) == len(ds) and max(abs(v - exp) / exp for v in freq.values()) < 0.15,
              f"T3. {mod}: every slice drawn about equally often over 20,000 draws")
    fake = D.SliceDataset(proc, "train", expected=None, modality="ct")
    fake.rows = list(both.rows)
    check(raises(lambda: D.make_epoch_draws(fake, 1, 643)), "T3. a mixed dataset flagged as ct is refused")
    for n, want in ((5760, 720), (2535, 316), (8295, 1036)):
        check(T.steps_per_epoch(n, T.Cfg(batch=8, accum=1)) == want == T.steps_per_epoch(n, T.Cfg(batch=4, accum=2)),
              f"T4. {n} draws -> {want} steps at batch 8 and at 4 x 2 alike")


def test_metric_and_stamp() -> None:
    print("\n--- T5, T6. selection metric and run stamp ---")
    rng = random.Random(0)
    rows = [{"modality": m, "patient": f"{m}{p}", "level": lev, "dice": rng.random()}
            for m in ("ct", "mri") for p in range(5) for lev in (0.0, 0.10) for _ in range(rng.randint(1, 6))]
    new, old = C.val_metric(rows), old_val_metric(rows)
    check(new == old, "T5. joint val_metric == the frozen pre-D71 function (bit-identical, every key)")
    ct_rows = [r for r in rows if r["modality"] == "ct"]
    s = C.val_metric(ct_rows, modalities=("ct",))
    check(s["selection"] == s["ct@0.10"] == old["ct@0.10"], "T5. ct-only: selection = CT Dice @ 0.10")
    s = C.val_metric([r for r in rows if r["modality"] == "mri"], modalities=("mri",))
    check(s["selection"] == old["mri@0.10"], "T5. mri-only: selection = MRI Dice @ 0.10")
    check(raises(lambda: C.val_metric(rows, modalities=("ct",))), "T5. an MRI row in a CT-only call is refused")
    check(raises(lambda: C.val_metric(ct_rows)), "T5. a missing modality in a joint call is refused")
    check(raises(lambda: C.val_metric(ct_rows, modalities=())), "T5. no modality is refused")

    check(T.stamp_of(T.Cfg()) == OLD_DEFAULT_STAMP, "T6. stamp_of(Cfg()) == the pre-D71 stamp, key for key")
    st = T.stamp_of(T.Cfg(modality="ct"))
    check(st.get("modality") == "ct" and {k: v for k, v in st.items() if k != "modality"} == OLD_DEFAULT_STAMP,
          "T6. a ct run's stamp = arm 2's + modality ct")
    check(T.stamp_of(T.Cfg(modality="mri"))["modality"] == "mri", "T6. an mri run's stamp records mri")
    check(raises(lambda: T.modalities_of(T.Cfg(modality="pet"))), "T6. an unknown modality is refused")


def cli(argv: list) -> int:
    """train_sam_lora.main(argv) with its output silenced (stderr too: argparse errors)."""
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return T.main(argv)


def run_dir_cfg(tmp, proc, ck, name, mod, **kw):
    return TT.tiny_cfg(tmp, proc, ck, run_dir=str(tmp / name), merged_out=str(tmp / f"{name}_merged.pth"),
                       modality=mod, n_draws=None, **kw)


def test_training(proc: Path, tmp: Path, ck: Path) -> None:
    print("\n--- T7-T11. single-modality training, resume, finalize, check-merged ---")
    for mod, other in (("ct", "mri"), ("mri", "ct")):
        n_train = len(D.SliceDataset(proc, "train", expected=None, modality=mod))
        n_val = len(D.SliceDataset(proc, "val", expected=None, modality=mod))
        cfg = run_dir_cfg(tmp, proc, ck, f"run_{mod}", mod)
        rc, out = quiet(T.run_training, cfg, TT.tiny_plain_builder)
        rj = json.loads((tmp / f"run_{mod}" / "run.json").read_text())
        check(rc == 0 and rj["status"] == "TRAIN_OK" and rj["modality"] == mod
              and rj["settings"]["modality"] == mod, f"T7. {mod}: 2-epoch run ends TRAIN_OK, run.json modality {mod}")
        check(rj["n_draws"] == n_train and rj["train_slices"] == {mod: n_train} and rj["val_slices"] == {mod: n_val}
              and set(rj["val_patients"]) == {mod} and "late_best" in rj,
              f"T7. {mod}: run.json records {n_train} draws/epoch, {mod}-only train/val counts, late_best")
        check(rj["steps_per_epoch"] == n_train // cfg.batch and rj["total_steps"] == 2 * (n_train // cfg.batch),
              f"T7. {mod}: steps = draws // batch per epoch")
        rows = T.read_metrics(tmp / f"run_{mod}" / "metrics.csv")
        check(len(rows) == 3 and all(r[f"{other}@0.10"] == "nan" and r[f"{other}@0.00"] == "nan"
                                     and r[f"nsd3_{other}@0.10"] == "nan" for r in rows),
              f"T7. {mod}: every metrics row has the {other} columns nan (no {other} rows were scored)")
        check(all(float(r["selection"]) == float(r[f"{mod}@0.10"]) for r in rows),
              f"T7. {mod}: selection == {mod.upper()} Dice @ 0.10 in every row")
        check(all(int(r[f"{other}_draws"]) == 0 and int(r[f"{mod}_draws"]) == int(r["steps"]) * cfg.batch
                  for r in rows[1:]), f"T7. {mod}: every training draw is {mod} ({other}_draws == 0)")
        check(f"modality {mod}: train slices {{'{mod}': {n_train}}}" in out,
              f"T7. {mod}: the startup line reports the {mod}-only counts")

        # T8: resume exactness
        cfg_b = run_dir_cfg(tmp, proc, ck, f"run_{mod}_b", mod, stop_after_epoch=1)
        quiet(T.run_training, cfg_b, TT.tiny_plain_builder)
        rc, out2 = quiet(T.run_training, run_dir_cfg(tmp, proc, ck, f"run_{mod}_b", mod), TT.tiny_plain_builder)
        la = torch.load(tmp / f"run_{mod}" / "last.pt", weights_only=False)["adapter"]
        lb = torch.load(tmp / f"run_{mod}_b" / "last.pt", weights_only=False)["adapter"]
        skip = {"epoch_s", "train_s", "val_s", "data_wait_s", "peak_mem_gib"}
        ma = [{k: v for k, v in r.items() if k not in skip} for r in rows]
        mb = [{k: v for k, v in r.items() if k not in skip} for r in T.read_metrics(tmp / f"run_{mod}_b" / "metrics.csv")]
        check(rc == 0 and "resuming after epoch 1" in out2 and set(la) == set(lb)
              and all(torch.equal(la[k], lb[k]) for k in la) and ma == mb,
              f"T8. {mod}: 2 epochs straight == 1 + resume + 1 (adapter tensors and metrics rows)")

        # T9: another modality on the same run folder
        check(raises(lambda: quiet(T.run_training, run_dir_cfg(tmp, proc, ck, f"run_{mod}", other),
                                   TT.tiny_plain_builder)), f"T9. resuming the {mod} run as {other} is refused")
        check(raises(lambda: quiet(T.run_training, run_dir_cfg(tmp, proc, ck, f"run_{mod}", "both"),
                                   TT.tiny_plain_builder)), f"T9. resuming the {mod} run as joint is refused")

        # T11: finalize + check-merged
        check(raises(lambda: quiet(T.finalize, run_dir_cfg(tmp, proc, ck, f"run_{mod}", other),
                                   TT.tiny_plain_builder, TT.tiny_plain_builder)),
              f"T11. finalize of the {mod} run called with {other} is refused")
        rc, out = quiet(T.finalize, cfg, TT.tiny_plain_builder, TT.tiny_plain_builder)
        rj = json.loads((tmp / f"run_{mod}" / "run.json").read_text())
        n_sub = len(D.identity_subset(D.SliceDataset(proc, "val", expected=None, modality=mod), cfg.pf_identity_step))
        n_kid = sum(D.SliceDataset(proc, "val", expected=None, modality=mod).n_kidneys(i) for i in range(n_val))
        check(rc == 0 and "FINALIZE_OK" in out and rj["merged"]["check"]["passed"]
              and rj["merged"]["check"]["n_kidneys"] == n_kid and n_sub == n_val,
              f"T11. {mod}: FINALIZE_OK, merge check over the {mod} val kidneys only ({n_kid})")
        check(quiet(T.check_merged, cfg)[0] == 0, f"T11. {mod}: check-merged --modality {mod} -> OK")
        for wrong in (other, "both"):
            rc, out = quiet(T.check_merged, run_dir_cfg(tmp, proc, ck, f"run_{mod}", wrong))
            check(rc == 1 and "MERGED_FAILED" in out and f"modality {mod}" in out,
                  f"T11. {mod}: check-merged --modality {wrong} -> MERGED_FAILED")
        # the command line: --modality reaches the Cfg through main()
        argv = ["--processed", str(proc), "--checkpoint", str(ck), "--rank", "4",
                "--run-dir", str(tmp / f"run_{mod}"), "--merged-out", str(tmp / f"run_{mod}_merged.pth"),
                "--check-merged"]
        check(cli(argv + ["--modality", mod]) == 0 and cli(argv + ["--modality", other]) == 1 and cli(argv) == 1,
              f"T11. CLI: --check-merged passes with --modality {mod}, fails with {other} and with the default (both)")
    check(raises(lambda: cli(["--processed", "p", "--checkpoint", "c", "--rank", "4", "--run-dir", "r",
                              "--modality", "pet"])), "T11. CLI: an unknown --modality is rejected")

    # the joint run's run.json has no modality key: still accepted as "both"
    j = TT.tiny_cfg(tmp, proc, ck, run_dir=str(tmp / "run_joint"), merged_out=str(tmp / "joint_merged.pth"))
    quiet(T.run_training, j, TT.tiny_plain_builder)
    rj = json.loads((tmp / "run_joint" / "run.json").read_text())
    check("modality" not in rj["settings"] and rj["modality"] == "both",
          "T6. a joint run's settings carry no modality key (arm 2's stamp)")
    quiet(T.finalize, j, TT.tiny_plain_builder, TT.tiny_plain_builder)
    check(quiet(T.check_merged, j)[0] == 0, "T11. joint run: check-merged with the default modality -> OK")
    rj = json.loads((tmp / "run_joint" / "run.json").read_text())      # now with the merge record
    del rj["modality"]
    (tmp / "run_joint" / "run.json").write_text(json.dumps(rj))
    check(quiet(T.check_merged, j)[0] == 0, "T11. an arm-2 run.json without any modality key is accepted as both")
    rc, _ = quiet(T.check_merged, TT.tiny_cfg(tmp, proc, ck, run_dir=str(tmp / "run_joint"),
                                            merged_out=str(tmp / "joint_merged.pth"), modality="ct"))
    check(rc == 1, "T11. ...and refused as ct")

    # startup guard: a val set of the wrong modality is refused before training
    tr = D.SliceDataset(proc, "train", expected=None, modality="ct")
    va = D.SliceDataset(proc, "val", expected=None, modality="mri")
    check(raises(lambda: quiet(T.run_training, run_dir_cfg(tmp, proc, ck, "run_guard", "ct"),
                               TT.tiny_plain_builder, (tr, va))), "T7. a ct run given MRI val data is refused at startup")
    va_both = D.SliceDataset(proc, "val", expected=None)
    check(raises(lambda: quiet(T.run_training, run_dir_cfg(tmp, proc, ck, "run_guard2", "ct"),
                               TT.tiny_plain_builder, (tr, va_both))), "T7. a ct run given joint val data is refused")


def test_joint_suite() -> None:
    print("\n--- T12. the joint arm-2 suite, unmodified ---")
    r = subprocess.run([sys.executable, str(HERE / "test_train_sam.py")], capture_output=True, text=True)
    check(r.returncode == 0 and "ALL TESTS PASSED" in r.stdout,
          "T12. test_train_sam.py passes unmodified" + ("" if r.returncode == 0 else f"\n{r.stdout[-2000:]}{r.stderr[-2000:]}"))


def test_scripts(tmp: Path) -> None:
    print("\n--- T13. generated SLURM scripts ---")
    regen = tmp / "regen"
    regen.mkdir()
    subprocess.run([sys.executable, str(HERE / "gen_sbatch_arm2_mod.py"), str(regen)], check=True, capture_output=True)
    for n in SCRIPTS:
        check((HERE / n).read_text() == (regen / n).read_text(), f"T13. {n} == generator output")
        check(subprocess.run(["bash", "-n", str(HERE / n)]).returncode == 0, f"T13. {n} passes bash -n")
    texts = {n: (HERE / n).read_text() for n in SCRIPTS}

    def header(n, key):
        return next(line.split("=", 1)[1] for line in texts[n].splitlines() if line.startswith(f"#SBATCH --{key}="))
    for n in SCRIPTS:
        want = "dgx" if "train" in n else "a40"
        check(header(n, "partition") == want and header(n, "qos") == want, f"T13. {n}: partition and QOS {want}")
        check("--exclude" not in texts[n], f"T13. {n}: no --exclude line")
    check(header("arm2_ct_train.sbatch", "time") == "20:00:00" and header("arm2_mri_train.sbatch", "time") == "10:00:00",
          "T13. time limits: CT 20 h, MRI 10 h")
    old_jobs = {"arm2_pre", "arm2_r4_tr", "arm2_r8_tr", "arm2_sel", "arm2_r4_ev", "arm2_r8_ev", "arm2_cmp"}
    jobs = {header(n, "job-name") for n in SCRIPTS}
    check(len(jobs) == 4 and not jobs & old_jobs, f"T13. 4 distinct job names, none of the joint arm 2's ({sorted(jobs)})")
    pairs = [("arm2_ct_", "arm2_mri_"), ("a2ct_", "a2mri_"), ("--modality ct", "--modality mri"),
             ("CT-only", "MRI-only"), ("CT only", "MRI only"), ("AMOS train CT", "AMOS train MRI"),
             ("AMOS val CT", "AMOS val MRI"), ("AMOS CT only", "AMOS MRI only"), ("merged CT-only", "merged MRI-only")]

    def mapped(n, extra=()):
        t = texts[n]
        for a, b in list(pairs) + list(extra):
            t = t.replace(a, b)
        return t
    check(mapped("arm2_ct_train.sbatch", [("--time=20:00:00", "--time=10:00:00")]) == texts["arm2_mri_train.sbatch"],
          "T13. the CT and MRI training scripts are twins (modality and time limit only)")
    check(mapped("arm2_ct_eval.sbatch") == texts["arm2_mri_eval.sbatch"], "T13. the CT and MRI evaluation scripts are twins")
    # the training arguments are arm 2's exactly, plus --modality
    arm2 = (HERE / "arm2_r4_train.sbatch").read_text()
    a2_call = [ln for ln in arm2.splitlines() if "--lr-lora" in ln]
    for m in ("ct", "mri"):
        calls = [ln for ln in texts[f"arm2_{m}_train.sbatch"].splitlines() if "--lr-lora" in ln]
        check(len(calls) == 2 and calls[0] == a2_call[0] + f" --modality {m}"
              and calls[1] == a2_call[0] + f" --modality {m} --finalize",
              f"T13. arm2_{m}_train: arm 2's exact arguments + --modality {m} (train and finalize)")

    # ---- run the scripts against stubs (python, nvidia-smi, squeue)
    seg = tmp / "Segmentation"
    for d in ("checkpoints", "runs", "outputs", "scripts", "data/processed"):
        (seg / d).mkdir(parents=True, exist_ok=True)
    (seg / "checkpoints" / "sam_vit_b_01ec64.pth").write_bytes(b"x")
    stub = tmp / "bin"
    stub.mkdir()
    calls_f = tmp / "calls.txt"
    fake_py = stub / "python"
    fake_py.write_text(
        "#!/bin/bash\n"
        'if [ "$1" = "-c" ]; then exit 0; fi\n'
        f'echo "$@" >> "{calls_f}"\n'
        'case "$*" in *--check-merged*) exit "${FAKE_CHECK_RC:-0}";; esac\n'
        'case "$*" in *--finalize*) exit 0;; esac\n'
        'case "$*" in *train_sam_lora.py*) exit "${FAKE_TRAIN_RC:-0}";; esac\n'
        'case "$*" in *eval_sam.py*) exit "${FAKE_EVAL_RC:-0}";; esac\n'
        'case "$*" in *compare_arms.py*) exit "${FAKE_CMP_RC:-0}";; esac\n'
        "exit 0\n")
    (stub / "nvidia-smi").write_text(
        "#!/bin/bash\n"
        'case "$*" in *query-gpu=name\\ *|*"query-gpu=name --format"*) echo "$FAKE_GPU";;'
        ' *) echo "0, $FAKE_GPU, 81920 MiB";; esac\n')
    (stub / "squeue").write_text('#!/bin/bash\ncat "$FAKE_SQUEUE"\n')
    for f in stub.iterdir():
        f.chmod(0o755)

    def run(n, gpu="NVIDIA A100-SXM4-80GB", queue="", **env_extra):
        sc = texts[n].replace("/home/rethinkingai-self/vihaan/miniconda3/envs/seg311/bin/python", str(fake_py))
        sc = sc.replace(str(SEG_REAL), str(seg))
        sp = tmp / n
        sp.write_text(sc)
        q = tmp / "q.txt"
        q.write_text(queue)
        if calls_f.exists():
            calls_f.unlink()
        env = dict(os.environ, PATH=f"{stub}:{os.environ['PATH']}", FAKE_SQUEUE=str(q), FAKE_GPU=gpu,
                   SLURM_JOB_ID="900", SLURM_CPUS_ON_NODE="16", **env_extra)
        r = subprocess.run(["bash", str(sp)], env=env, capture_output=True, text=True)
        return r.returncode, (calls_f.read_text().splitlines() if calls_f.exists() else []), r.stdout

    A40 = "NVIDIA A40"
    for m, other in (("ct", "mri"), ("mri", "ct")):
        n = f"arm2_{m}_train.sbatch"
        rc, py, _ = run(n, queue=f"900 a2{m}_tr RUNNING\n901 a2{other}_tr RUNNING\n902 arm2_r4_tr RUNNING\n")
        tr = [c for c in py if "train_sam_lora.py" in c]
        ok = (rc == 0 and len(tr) == 2 and "--finalize" not in tr[0] and tr[1].endswith("--finalize")
              and all(f"--modality {m}" in c and f"{seg}/runs/arm2_{m}_r4 " in c
                      and f"{seg}/checkpoints/sam_vit_b_arm2_{m}_r4.pth" in c and "sam_vit_b_01ec64.pth" in c
                      and "--rank 4 " in c and "--expect-gpu A100" in c for c in tr))
        check(ok, f"T13. {n}: trains {m} on the A100 then finalizes (runs alongside the other modality)")
        rc, py, _ = run(n, FAKE_TRAIN_RC="1")
        check(rc == 1 and not any("--finalize" in c for c in py), f"T13. {n}: a failed training skips finalize")
        rc, py, _ = run(n, gpu=A40)
        check(rc == 4 and not py, f"T13. {n}: refuses to start on an A40")
        for clash in (f"903 a2{m}_tr PENDING", f"904 a2{m}_ev RUNNING"):
            rc, py, _ = run(n, queue=clash + "\n")
            check(rc == 3 and not py, f"T13. {n}: refuses next to '{clash}'")

        n = f"arm2_{m}_eval.sbatch"
        rc, py, _ = run(n, gpu=A40, FAKE_CHECK_RC="1")
        check(rc == 6 and len(py) == 1 and "--check-merged" in py[0] and f"--modality {m}" in py[0],
              f"T13. {n}: refuses an unverified merged checkpoint (check-merged --modality {m})")
        for ref in ("arm1_vit_b", "arm2_r4"):
            (seg / "outputs" / ref).mkdir(exist_ok=True)
            (seg / "outputs" / ref / "run.json").write_text("{}")
        rc, py, out = run(n, gpu=A40)
        ev = [c for c in py if "eval_sam.py" in c]
        ag = [c for c in py if "aggregate_arm.py" in c]
        cm = [c for c in py if "compare_arms.py" in c]
        check(rc == 0 and py[0].count("--check-merged") == 1 and len(ev) == 1
              and f"{seg}/checkpoints/sam_vit_b_arm2_{m}_r4.pth" in ev[0] and "--model-type vit_b" in ev[0]
              and f"--out {seg}/outputs/arm2_{m}_r4 " in ev[0] + " " and ev[0].rstrip().endswith("--resume")
              and len(ag) == 1 and f"{m.upper()}-only" in ag[0]
              and len(cm) == 2 and f"--a {seg}/outputs/arm1_vit_b" in cm[0] and f"--a {seg}/outputs/arm2_r4" in cm[1]
              and all(f"--b {seg}/outputs/arm2_{m}_r4" in c for c in cm) and "DONE: evaluated" in out,
              f"T13. {n}: check-merged -> eval_sam (A40) -> aggregate -> compare vs arm 1b and joint arm 2")
        rc, py, _ = run(n, gpu=A40, FAKE_EVAL_RC="7")
        check(rc == 7 and not any("aggregate_arm.py" in c for c in py), f"T13. {n}: a failed evaluation stops the script")
        rc, py, out = run(n, gpu=A40, FAKE_CMP_RC="1")
        check(rc == 5 and len([c for c in py if "compare_arms.py" in c]) == 2 and "comparison FAILED" in out,
              f"T13. {n}: a failed comparison is reported, the other still runs, exit 5")
        (seg / "outputs" / "arm2_r4" / "run.json").unlink()
        rc, py, out = run(n, gpu=A40)
        check(rc == 5 and "compare vs arm2_r4 skipped" in out, f"T13. {n}: a missing reference is reported, exit 5")
        rc, py, _ = run(n)
        check(rc == 4 and not py, f"T13. {n}: refuses to start on an A100")
        rc, py, _ = run(n, gpu=A40, queue=f"905 a2{m}_tr RUNNING\n")
        check(rc == 3 and not py, f"T13. {n}: refuses while the {m} training runs")


def test_python_compat() -> None:
    print("\n--- T14. Python 3.10 / 3.11 syntax ---")
    import ast
    bad = []
    for f in ("train_data.py", "sam_train_core.py", "train_sam_lora.py", "gen_sbatch_arm2_mod.py", "test_arm2_mod.py"):
        for v in ((3, 10), (3, 11)):
            try:
                ast.parse((HERE / f).read_text(), feature_version=v)
            except SyntaxError as e:
                bad.append(f"{f}:{e.lineno} (py{v[0]}.{v[1]})")
    check(not bad, "all changed / new files parse as Python 3.10 and 3.11" + (f" -- {bad}" if bad else ""))


def test_real_data() -> None:
    print("\n--- real data (Prajna only) ---")
    proc = SEG_REAL / "data" / "processed"
    if not (proc / "manifest.csv").exists():
        print("  skipped (no real manifest here)")
        return
    for split, want in (("train", {"ct": 5760, "mri": 2535}), ("val", {"ct": 1901, "mri": 843})):
        for m in ("ct", "mri"):
            ds = D.SliceDataset(proc, split, modality=m)        # default expected counts (D33)
            check(len(ds) == want[m] and ds.counts == {m: want[m]}, f"real: {split} {m} = {want[m]} boxed slices")
    pats = {m: len({r["patient"] for r in D.SliceDataset(proc, "val", modality=m).rows}) for m in ("ct", "mri")}
    check(pats == {"ct": 59, "mri": 12}, f"real: val patients {pats} (59 CT / 12 MRI)")
    both = D.SliceDataset(proc, "train")
    check(D.make_epoch_draws(both, 1, 643) == old_make_epoch_draws(both, 1, 643),
          "real: joint draws == the frozen pre-D71 function")
    for m, steps in (("ct", 720), ("mri", 316)):
        ds = D.SliceDataset(proc, "train", modality=m)
        dr = D.make_epoch_draws(ds, 1, 643)
        check(len(dr) == len(ds) and all(ds.rows[j]["modality"] == m for _, j in dr)
              and T.steps_per_epoch(len(dr), T.Cfg()) == steps,
              f"real: {m} epoch = {len(ds)} {m} draws, {steps} steps")
        cfg = T.Cfg(processed=str(proc), modality=m)
        tr, va = T.load_datasets(cfg)
        info = T.check_run_data(cfg, tr, va, len(tr))
        check(info["train_slices"] == {m: len(ds)} and set(info["val_slices"]) == {m},
              f"real: load_datasets + startup guard for {m}")
    cj = SEG_REAL / "runs" / "arm2_r4" / "config.json"
    if cj.exists():
        have, now = json.loads(cj.read_text())["settings"], T.stamp_of(arm2_cfg())
        diff = {k: (have.get(k), now.get(k)) for k in sorted(set(have) | set(now)) if have.get(k) != now.get(k)}
        check(not diff, "real: the joint arm 2's recorded config.json stamp is reproduced exactly"
              + (f" -- differs (recorded, now): {diff}" if diff else ""))
    else:
        print(f"  skipped stamp check ({cj} not found)")


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="arm2mod_test_"))
    proc = tmp / "processed"
    TT.build_processed(proc)
    ck = tmp / "tiny.pth"
    torch.save(TT.tiny_sam().state_dict(), ck)
    test_python_compat()
    test_dataset(proc)
    test_draws(proc)
    test_metric_and_stamp()
    test_training(proc, tmp, ck)
    test_scripts(tmp)
    test_real_data()
    test_joint_suite()
    print(f"\nscratch: {tmp}\n\nALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
