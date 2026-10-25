#!/usr/bin/env python
"""Run the FROZEN arm-3 student code (frozen_arm3/) in its own process, for comparison with the
current code (ARM4_CODE_PLAN.md §9). Proves that arm 4's changes leave arm 3 bit-identical.

    python arm3_reference.py --code-dir frozen_arm3 --mode items --processed P --variant plain \
        --seed 643 --positions 0,1,2 --per-epoch 12000 --out items.json [--expected-json E]
    python arm3_reference.py --code-dir frozen_arm3 --mode stamps --cfg-json cfgs.json --out stamps.json
    python arm3_reference.py --code-dir frozen_arm3 --mode tiny_train --cfg-json cfg.json

frozen_arm3/ holds byte copies of the 7 arm-3 CODE_FILES, sha256 recorded below (taken from the
files uploaded for arm 3; the arm-4 preflight also matches them against
runs/arm3_plain_s643/config.json). This script removes its own folder from sys.path, so the
frozen modules can never import a current one; every loaded project module is checked to come
from the frozen folder. Importing this file (for its constants and helpers) runs nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

FROZEN_SHA256 = {
    "student_unet.py": "3304201f30e8e28a017cb73c2508c25dadcf4ec8388070fb1f3fc9c4dfbced64",
    "student_core.py": "0bad44ae8e9b9d4a431255ae0ba822e03a88e98128e2ca351f8c2069f89ffc0e",
    "student_aug.py": "6c761dadb092dd92080339ee18c0854de8cd8946d1d04c8762887a8a2ce0eef5",
    "student_data.py": "20562fd494228df4a2f7f2939ab003801cea35610da1a2f3d1acfea3bd169db5",
    "train_student.py": "18e56c65768934c41a3bbb8f022ffbf22281209e6df8c2c0f43ae114f4b621a1",
    "sam_metrics.py": "1702d323f5f9264ef5362b467aa9a23bbf3c22e481ca04a3538ed45b227d75b6",
    "eval_sam.py": "2c05e9f334f0df0bd9ea067bf5ab6162c01251c187d9c1188a9e67204e0a5282",
}
PROJECT_MODULES = ("student_unet", "student_core", "student_aug", "student_data", "train_student",
                   "sam_metrics", "eval_sam", "student_kd", "teacher_io", "teacher_maps")
HERE = Path(__file__).resolve().parent


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def check_frozen(code_dir: Path) -> list[str]:
    """Problems with the frozen folder (empty = exactly the recorded arm-3 files)."""
    probs = []
    for f, want in FROZEN_SHA256.items():
        p = Path(code_dir) / f
        if not p.exists():
            probs.append(f"{f} missing from {code_dir}")
        elif file_sha256(p) != want:
            probs.append(f"{f} in {code_dir} is not the arm-3 file (sha256 differs)")
    return probs


def tensor_digest(t) -> str:
    a = t.detach().cpu().contiguous().numpy()
    h = hashlib.sha256(f"{a.dtype}|{a.shape}|".encode())
    h.update(a.tobytes())
    return h.hexdigest()


def item_digest(item: dict) -> list:
    """What must be identical between the frozen and the current code for one training item."""
    return [int(item["idx"]), int(item["epoch"]), int(item["i"]), tensor_digest(item["x"]),
            tensor_digest(item["label"]), tensor_digest(item["valid"])]


def spread_positions(total: int, n: int) -> list[int]:
    """n consecutive positions from 0 plus n spread over the whole run (deterministic)."""
    step = max(1, total // max(n, 1))
    return sorted(set(range(min(n, total))) | {min(total - 1, k * step + 7) for k in range(n)})


def run_frozen(code_dir: Path, args: list, timeout: float | None = None) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(HERE / "arm3_reference.py"), "--code-dir", str(code_dir), *[str(a) for a in args]]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def frozen_items(code_dir: Path, processed, variant: str, seed: int, positions: list, per_epoch: int,
                 size: int, augment: bool, max_level: float, expected: dict | None) -> dict:
    with tempfile.TemporaryDirectory(prefix="arm3_ref_") as td:
        out = Path(td) / "items.json"
        args = ["--mode", "items", "--processed", processed, "--variant", variant, "--seed", seed,
                "--positions", ",".join(str(p) for p in positions), "--per-epoch", per_epoch, "--size", size,
                "--augment", int(bool(augment)), "--max-level", repr(float(max_level)), "--out", out]
        if expected is not None:
            ej = Path(td) / "expected.json"
            ej.write_text(json.dumps(expected))
            args += ["--expected-json", ej]
        r = run_frozen(code_dir, args)
        if r.returncode != 0:
            raise RuntimeError(f"frozen arm-3 items failed:\n{r.stdout[-2000:]}\n{r.stderr[-4000:]}")
        return json.loads(out.read_text())


def frozen_stamps(code_dir: Path, overrides: list) -> list:
    with tempfile.TemporaryDirectory(prefix="arm3_ref_") as td:
        cj, out = Path(td) / "cfgs.json", Path(td) / "stamps.json"
        cj.write_text(json.dumps(overrides))
        r = run_frozen(code_dir, ["--mode", "stamps", "--cfg-json", cj, "--out", out])
        if r.returncode != 0:
            raise RuntimeError(f"frozen arm-3 stamps failed:\n{r.stdout[-2000:]}\n{r.stderr[-4000:]}")
        return json.loads(out.read_text())


# ------------------------------------------------------------------- the frozen process
def isolate(code_dir: Path) -> None:
    code_dir = Path(code_dir).resolve()
    loaded = [m for m in PROJECT_MODULES if m in sys.modules]
    if loaded:
        raise SystemExit(f"project modules already imported before isolation: {loaded}")
    sys.path[:] = [p for p in sys.path if Path(p or os.getcwd()).resolve() != HERE]
    sys.path.insert(0, str(code_dir))
    probs = check_frozen(code_dir)
    if probs:
        raise SystemExit("REFUSING -- " + "; ".join(probs))


def verify_loaded(code_dir: Path) -> None:
    code_dir = Path(code_dir).resolve()
    for m in PROJECT_MODULES:
        mod = sys.modules.get(m)
        if mod is None:
            continue
        f = Path(mod.__file__).resolve()
        if f.parent != code_dir:
            raise SystemExit(f"module {m} was loaded from {f}, not from the frozen folder {code_dir}")
        if FROZEN_SHA256.get(f.name) != file_sha256(f):
            raise SystemExit(f"module {m} ({f}) is not the recorded arm-3 file")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--code-dir", required=True, type=Path)
    p.add_argument("--mode", required=True, choices=("items", "stamps", "tiny_train"))
    p.add_argument("--processed")
    p.add_argument("--variant", default="plain")
    p.add_argument("--seed", type=int, default=643)
    p.add_argument("--positions", default="")
    p.add_argument("--per-epoch", type=int, default=12000)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--augment", type=int, default=1)
    p.add_argument("--max-level", type=float, default=0.20)
    p.add_argument("--expected-json", type=Path)
    p.add_argument("--cfg-json", type=Path)
    p.add_argument("--out", type=Path)
    a = p.parse_args(argv)
    isolate(a.code_dir)
    import torch
    torch.set_num_threads(1)
    import student_core as C
    import student_data as D
    import train_student as T
    verify_loaded(a.code_dir)
    if a.mode == "items":
        expected = json.loads(a.expected_json.read_text()) if a.expected_json else D.EXPECTED_COUNTS
        store = D.SliceStore(Path(a.processed), "train", expected, 8, lambda m: print(m, file=sys.stderr))
        norm = C.norm_from_buffer(store.img)
        pos = [int(x) for x in a.positions.split(",") if x != ""]
        ds = D.TrainItems(store, a.variant, a.seed, norm, a.per_epoch, 0, max(pos) + 1, a.size,
                          bool(a.augment), a.max_level)
        res = {"norm": norm.as_dict(), "items": {str(q): item_digest(ds[q]) for q in pos}}
        verify_loaded(a.code_dir)
        a.out.write_text(json.dumps(res))
        return 0
    if a.mode == "stamps":
        res = []
        for ov in json.loads(a.cfg_json.read_text()):
            cfg = T.Cfg(**ov)
            res.append({"stamp": T.stamp_of(cfg), "metric_fields": T.metric_fields(cfg.variant),
                        "run_dir_name": T.run_dir_name(cfg)})
        a.out.write_text(json.dumps(res))
        return 0
    ov = json.loads(a.cfg_json.read_text())
    rc = T.run_training(T.Cfg(**ov))
    verify_loaded(a.code_dir)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
