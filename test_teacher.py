#!/usr/bin/env python
"""CPU tests for the arm-4 teacher maps (teacher_io.py, teacher_maps.py, gen_sbatch_arm4.py).

    python test_teacher.py        # exits 0 and prints ALL TESTS PASSED

A tiny real SAM (official segment_anything) on the synthetic processed/ folder of
test_student.py. Every saved logit is compared with an independent SamPredictor run that
uses arm 2's evaluation calls.
"""
from __future__ import annotations

import contextlib
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
import student_data as D  # noqa: E402
import teacher_io as TIO  # noqa: E402
import teacher_maps as TM  # noqa: E402
import test_student as TS  # noqa: E402
from segment_anything import SamPredictor  # noqa: E402
from segment_anything.modeling import (ImageEncoderViT, MaskDecoder, PromptEncoder, Sam,  # noqa: E402
                                       TwoWayTransformer)

torch.set_num_threads(1)
check, raises = TS.check, TS.raises


def quiet(fn, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = fn(*a, **k)
    return r, buf.getvalue()


def tiny_sam(seed: int = 0) -> Sam:
    torch.manual_seed(seed)
    enc = ImageEncoderViT(img_size=64, patch_size=16, in_chans=3, embed_dim=64, depth=2, num_heads=2,
                          mlp_ratio=4.0, out_chans=32, qkv_bias=True, use_rel_pos=True,
                          window_size=14, global_attn_indexes=(1,))
    pe = PromptEncoder(embed_dim=32, image_embedding_size=(4, 4), input_image_size=(64, 64), mask_in_chans=16)
    dec = MaskDecoder(num_multimask_outputs=3, transformer_dim=32,
                      transformer=TwoWayTransformer(depth=2, embedding_dim=32, mlp_dim=64, num_heads=2),
                      iou_head_depth=3, iou_head_hidden_dim=32)
    return Sam(image_encoder=enc, prompt_encoder=pe, mask_decoder=dec,
               pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375]).eval()


def setup(tmp: Path):
    proc = tmp / "processed"
    exp = TS.build_processed(proc)
    sam = tiny_sam()
    ckpt = tmp / "teacher.pth"
    torch.save(sam.state_dict(), ckpt)
    sha = TM.E.checkpoint_identity(ckpt, "vit_b")["checkpoint_sha256"]
    runs = tmp / "runs"
    (runs / "arm2_r4").mkdir(parents=True)
    (runs / "arm2_r4" / "run.json").write_text(json.dumps(
        {"status": "TRAIN_OK", "merged": {"sha256": sha, "check": {"passed": True}}}))
    (runs / "arm2_selection.json").write_text(json.dumps({"chosen_rank": 4, "r4": {"merged_sha256": sha}}))
    return proc, exp, sam, ckpt, sha, runs


def go(proc, ckpt, runs, out, sam, sha, exp, qc=0.0, **kw):
    torch.set_grad_enabled(False)
    try:
        return quiet(TM.run, proc, ckpt, runs, out, device="cpu", expected_sha=sha,
                     predictor_factory=lambda: SamPredictor(sam), expected_counts=exp,
                     qc_min_dice=qc, threads=2, **kw)
    finally:
        torch.set_grad_enabled(True)


def test_run(tmp: Path) -> None:
    print("\n--- 1-4. teacher maps: content, exactness, resume ---")
    proc, exp, sam, ckpt, sha, runs = setup(tmp)
    out = tmp / "teacher"
    (rc, log) = go(proc, ckpt, runs, out, sam, sha, exp)
    rj = json.loads((out / "run.json").read_text())
    with contextlib.redirect_stdout(io.StringIO()):
        store = D.SliceStore(proc, "train", exp, 2)
    boxed = [i for i, r in enumerate(store.rows) if r["sam_eligible"]]
    n_kid = sum(sum(b is not None for b in store.boxes(i).values()) for i in boxed)
    check(rc == 0 and rj["status"] == "TEACHER_OK" and "TEACHER_OK" in log, "run ends TEACHER_OK")
    files = sorted(str(p.relative_to(out)) for p in (out / "slices").rglob("*.npz"))
    want = sorted(TIO.rel_path(store.rows[i]["case"], store.rows[i]["z"]) for i in boxed)
    check(files == want and rj["slices"] == len(boxed) and rj["kidneys"] == n_kid,
          f"exactly one file per boxed TRAIN slice ({len(boxed)} slices, {n_kid} kidneys); none for val/test/kidney-free")
    pred = SamPredictor(sam)
    exact = mask_eq = sides_ok = boxes_ok = True
    with torch.no_grad():
        for i in boxed:
            r = store.rows[i]
            img, msk = store.get(i)
            meta, logits = TIO.load_slice(out, r["case"], r["z"])
            boxes = store.boxes(i)
            sides_ok &= sorted(logits) == [s for s in (1, 2) if boxes[s] is not None]
            boxes_ok &= all(meta["boxes"][str(s)] == list(boxes[s]) for s in logits)
            pred.set_image(np.repeat(img[..., None], 3, axis=2), image_format="RGB")
            for s, lg in logits.items():
                ref, _, _ = pred.predict(box=np.array(boxes[s], dtype=float), multimask_output=False, return_logits=True)
                exact &= np.array_equal(lg, ref[0].astype(np.float16))
                m, _, _ = pred.predict(box=np.array(boxes[s], dtype=float), multimask_output=False)   # eval's call
                mask_eq &= np.array_equal(lg.astype(np.float32) > 0, m[0]) or meta["fp16_flips"][str(s)] > 0
    check(exact, "every stored logit == float16(SamPredictor.predict(return_logits=True)) bit for bit")
    check(mask_eq, "stored logit > 0 == the evaluation's own predict() mask")
    check(sides_ok and boxes_ok, "only boxed sides stored; meta boxes == the CLEAN manifest boxes")
    check(len((out / "teacher_manifest.csv").read_text().strip().splitlines()) == len(boxed) + 1
          and json.loads((out / "teacher.json").read_text())["teacher_sha256"] == sha,
          "teacher_manifest.csv has one row per slice; teacher.json names the teacher")
    # resume: valid files kept untouched
    snap = {p: p.stat().st_mtime_ns for p in (out / "slices").rglob("*.npz")}
    (rc, log) = go(proc, ckpt, runs, out, sam, sha, exp)
    check(rc == 0 and "computed 0, kept" in log and all(p.stat().st_mtime_ns == t for p, t in snap.items()),
          "re-run: every valid file kept, none rewritten")
    first = {p: np.load(p)["logit_1" if "logit_1" in np.load(p).files else "logit_2"].copy() for p in snap}
    victims = sorted(snap)[:2]
    victims[0].unlink()
    victims[1].write_bytes(b"truncated garbage")
    (rc, log) = go(proc, ckpt, runs, out, sam, sha, exp)
    same = all(np.array_equal(first[p], np.load(p)["logit_1" if "logit_1" in np.load(p).files else "logit_2"])
               for p in victims)
    check(rc == 0 and "computed 2" in log and "unreadable" in log and same,
          "a deleted and a corrupted file are recomputed, identical to the originals")
    (out / "slices" / "amos_0001" / "9999.npz").write_bytes(b"x")
    rc, _ = go(proc, ckpt, runs, out, sam, sha, exp)
    check(rc == 1 and json.loads((out / "run.json").read_text())["status"] == "TEACHER_FAILED",
          "an unexpected extra file -> TEACHER_FAILED")
    (out / "slices" / "amos_0001" / "9999.npz").unlink()
    rc, _ = go(proc, ckpt, runs, out, sam, sha, exp, qc=TM.QC_MIN_DICE)
    check(rc == 1, "teacher Dice below the QC threshold -> TEACHER_FAILED (a random tiny SAM is not a teacher)")


def test_guards(tmp: Path) -> None:
    print("\n--- 5-6. guards ---")
    proc, exp, sam, ckpt, sha, runs = setup(tmp)
    out = tmp / "t2"
    check(raises(lambda: go(proc, ckpt, runs, out, sam, "0" * 64, exp)), "a checkpoint that is not the recorded teacher is refused")
    sel = json.loads((runs / "arm2_selection.json").read_text())
    (runs / "arm2_selection.json").write_text(json.dumps({**sel, "chosen_rank": 8}))
    check(raises(lambda: go(proc, ckpt, runs, out, sam, sha, exp)), "selection chose r=8 -> refused")
    (runs / "arm2_selection.json").write_text(json.dumps(sel))
    rj = json.loads((runs / "arm2_r4" / "run.json").read_text())
    (runs / "arm2_r4" / "run.json").write_text(json.dumps({**rj, "merged": {"sha256": "1" * 64, "check": {"passed": True}}}))
    check(raises(lambda: go(proc, ckpt, runs, out, sam, sha, exp)), "arm2_r4 merged sha differs -> refused")
    (runs / "arm2_r4" / "run.json").write_text(json.dumps(rj))
    rc, _ = go(proc, ckpt, runs, out, sam, sha, exp)
    st = json.loads((out / "teacher.json").read_text())
    (out / "teacher.json").write_text(json.dumps({**st, "manifest_sha256": "2" * 64}))
    check(rc == 0 and raises(lambda: go(proc, ckpt, runs, out, sam, sha, exp)),
          "an output folder made from different data/teacher is refused")
    m = {"case": "c", "z": 1, "h": 4, "w": 5, "sides": [1]}
    check(raises(lambda: TIO.save_slice(tmp / "x", m, {1: np.full((4, 5), np.nan)})), "non-finite logits refused")
    check(raises(lambda: TIO.save_slice(tmp / "x", m, {1: np.zeros((5, 4))})), "wrong shape refused")
    check(raises(lambda: TIO.save_slice(tmp / "x", m, {2: np.zeros((4, 5))})), "sides inconsistent with meta refused")
    check(raises(lambda: TIO.save_slice(tmp / "x", m, {1: np.full((4, 5), 1e6)})), "logits beyond float16 range refused")
    TIO.save_slice(tmp / "x", m, {1: np.zeros((4, 5), np.float32)})
    meta, lg = TIO.load_slice(tmp / "x", "c", 1)
    check(lg[1].dtype == np.float16 and meta["sides"] == [1], "save/load round trip: float16 arrays, meta intact")
    os.replace(TIO.slice_path(tmp / "x", "c", 1), TIO.slice_path(tmp / "x", "c", 2))
    check(raises(lambda: TIO.load_slice(tmp / "x", "c", 2)), "a file whose meta names another slice is refused")
    code = ("import sys; sys.path.insert(0, %r); import teacher_io; "
            "bad = [m for m in ('torch', 'segment_anything') if m in sys.modules]; print(bad); sys.exit(1 if bad else 0)") % str(HERE)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    check(r.returncode == 0, f"teacher_io imports neither torch nor SAM {r.stdout.strip()}")


def test_sbatch(tmp: Path) -> None:
    print("\n--- 7. SLURM script ---")
    regen = tmp / "regen"
    regen.mkdir(parents=True)
    subprocess.run([sys.executable, str(HERE / "gen_sbatch_arm4.py"), str(regen)], check=True, capture_output=True)
    txt = (HERE / "teacher_maps.sbatch").read_text()
    check(sorted(p.name for p in regen.glob("*.sbatch")) == ["teacher_maps.sbatch"]
          and txt == (regen / "teacher_maps.sbatch").read_text(), "teacher_maps.sbatch == generator output")
    check(subprocess.run(["bash", "-n", str(HERE / "teacher_maps.sbatch")]).returncode == 0, "passes bash -n")
    check("#SBATCH --partition=a40" in txt and "#SBATCH --qos=a40" in txt and "*A40*" in txt
          and "--expect-gpu A40" in txt, "a40 partition, A40 guards (as arm 2's evaluation)")
    stub = tmp / "stub"
    stub.mkdir()
    calls = tmp / "calls.txt"
    (stub / "squeue").write_text('#!/bin/bash\necho "${SQUEUE_OUT:-}"\n')
    (stub / "nvidia-smi").write_text('#!/bin/bash\necho "${GPU_NAME:-NVIDIA A40}"\n')
    (stub / "py").write_text(f'#!/bin/bash\nif [ "$1" = "-c" ]; then echo env-ok; exit 0; fi\necho "$@" >> {calls}\nexit 0\n')
    for f in stub.iterdir():
        f.chmod(f.stat().st_mode | stat.S_IEXEC)
    seg = tmp / "seg"
    for d in ("checkpoints", "runs", "data/processed"):
        (seg / d).mkdir(parents=True)
    (seg / "checkpoints" / "sam_vit_b_arm2_r4.pth").write_text("x")
    (seg / "runs" / "arm2_selection.json").write_text("{}")
    (seg / "data" / "processed" / "manifest.csv").write_text("x\n")

    def run(**env):
        t = txt.replace("SEG=/home/rethinkingai-self/vihaan/Segmentation", f"SEG={seg}")
        t = t.replace("PY=/home/rethinkingai-self/vihaan/miniconda3/envs/seg311/bin/python", f"PY={stub / 'py'}")
        sp = tmp / "run.sh"
        sp.write_text(t)
        calls.unlink(missing_ok=True)
        r = subprocess.run(["bash", str(sp)], env={**os.environ, "PATH": f"{stub}:{os.environ['PATH']}", **env},
                           capture_output=True, text=True)
        return r.returncode, (calls.read_text().split() if calls.exists() else [])
    rc, argv = run()
    check(rc == 0 and argv[0].endswith("scripts/teacher_maps.py") and f"{seg}/checkpoints/sam_vit_b_arm2_r4.pth" in argv
          and f"{seg}/teacher/arm2_r4_train" in argv and argv[-2:] == ["--expect-gpu", "A40"],
          "runs teacher_maps.py on the r4 checkpoint -> teacher/arm2_r4_train, --expect-gpu A40")
    rc, argv = run(SQUEUE_OUT="5 teacher_r4 RUNNING")
    check(rc == 3 and not argv, "a second copy is refused")
    rc, argv = run(GPU_NAME="NVIDIA A100-SXM4-80GB")
    check(rc == 4 and not argv, "a non-A40 node is refused")


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="arm4_teacher_test_"))
    try:
        test_run(tmp / "a")
        test_guards(tmp / "b")
        test_sbatch(tmp / "c")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\nALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
