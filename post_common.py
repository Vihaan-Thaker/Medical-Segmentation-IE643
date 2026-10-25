#!/usr/bin/env python
"""Shared, validated loading of the arm-2 post-processing inputs (ARM2_POST_PLAN.md §3.5).

Reads only the small files copied back from Prajna by fetch_results.sh, laid out as on the
cluster under <results>/outputs/... and <results>/runs/... . Stdlib only (plus nothing): it must
run in the Mac's ordinary Python, which has no torch and no surface-distance, so it never
imports eval_sam / sam_metrics / compare_arms / train_sam_lora (plan invariant 7).

Every inconsistency raises PostError; the entry points turn that into POST_FAILED + exit 1.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

GROUPS = ("amos/ct", "amos/mri", "chaos/mri/t1in", "chaos/mri/t1out", "chaos/mri/t2", "word/ct")
IN_DOMAIN = ("amos/ct", "amos/mri")
JITTERS = (0.0, 0.05, 0.10, 0.20)
HEADLINE_JITTER = 0.10
TOLERANCES = (1, 2, 3, 5)
METRICS_2D = ("dice",) + tuple(f"nsd_{t}" for t in TOLERANCES)
METRICS_3D = ("dice3d",) + tuple(f"nsd3d_{t}" for t in TOLERANCES)
HEADLINE_NSD = "nsd_3"
# (folder under outputs/, label). Order = column order everywhere.
ARMS = (("arm1", "Arm 1 · SAM ViT-H (off-the-shelf)"),
        ("arm1_vit_b", "Arm 1b · SAM ViT-B (off-the-shelf)"),
        ("arm2_r4", "Arm 2 · r=4"),
        ("arm2_r8", "Arm 2 · r=8"))
# Paired comparisons as written by compare_arms.py: (B folder, A folder, label "A -> B").
COMPARES = (("arm1_vit_b", "arm1", "ViT-H → ViT-B"),
            ("arm2_r4", "arm1_vit_b", "Arm 1b → Arm 2 r=4"),
            ("arm2_r8", "arm1_vit_b", "Arm 1b → Arm 2 r=8"),
            ("arm2_r8", "arm2_r4", "Arm 2 r=4 → r=8"))
RANKS = (4, 8)
ROUND4_TOL = 1.5e-4          # summary.csv / compare.csv store 4-decimal values


class PostError(RuntimeError):
    pass


# ----------------------------------------------------------------------------- basics
def read_json(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise PostError(f"missing file: {path}")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise PostError(f"{path}: invalid JSON ({e})")


def read_csv(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise PostError(f"missing file: {path}")
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise PostError(f"{path}: no rows")
    return rows


def fnum(x, what: str) -> float:
    """A finite float, or PostError naming the field."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise PostError(f"{what}: not a number ({x!r})")
    if not math.isfinite(v):
        raise PostError(f"{what}: not finite ({x!r})")
    return v


def jkey(x) -> float:
    """Jitter as a dict key: float rounded to 4 decimals (0.1 == 0.10 == '0.1')."""
    return round(float(x), 4)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def expected_grid() -> list[tuple]:
    out = []
    for lvl, metrics in (("2d", METRICS_2D), ("3d", METRICS_3D)):
        for g in GROUPS:
            for m in metrics:
                for j in JITTERS:
                    out.append((lvl, g, m, jkey(j)))
    return out


# ------------------------------------------------------------------------------ arms
def load_arm(results: Path, name: str, label: str = "") -> dict:
    """One evaluated arm: certified run.json, complete summary.csv grid, model identity."""
    d = Path(results) / "outputs" / name
    run = read_json(d / "run.json")
    if "problems" not in run or run["problems"]:
        raise PostError(f"{name}: run.json reports problems {run.get('problems')!r} -- not certified")
    if str(run.get("mode", "")).startswith("shard"):
        raise PostError(f"{name}: run.json is a shard (partial results)")
    summary = {}
    for r in read_csv(d / "summary.csv"):
        key = (r["level"], r["group"], r["metric"], jkey(r["jitter"]))
        if key in summary:
            raise PostError(f"{name}/summary.csv: duplicate row {key}")
        what = f"{name}/summary.csv {key}"
        summary[key] = {"mean": fnum(r["mean"], what + " mean"), "ci_lo": fnum(r["ci_lo"], what + " ci_lo"),
                        "ci_hi": fnum(r["ci_hi"], what + " ci_hi"),
                        "n_patients": int(r["n_patients"]), "n_instances": int(r["n_instances"])}
    grid = expected_grid()
    missing = [k for k in grid if k not in summary]
    extra = sorted({k[1] for k in summary} - set(GROUPS))
    if missing:
        raise PostError(f"{name}/summary.csv: {len(missing)} expected rows missing, e.g. {missing[:3]}")
    if extra:
        raise PostError(f"{name}/summary.csv: unexpected groups {extra}")
    model_json = d / "model.json"
    rmodel = run.get("model")
    if model_json.exists():
        model = read_json(model_json)
        if rmodel and {k: rmodel.get(k) for k in model} != model:
            raise PostError(f"{name}: model.json and run.json['model'] disagree")
    elif rmodel:
        model = rmodel
    else:                                    # arm 1 (ViT-H) predates the stamp (D32): expected
        a = run.get("args", {})
        model = {"model_type": a.get("model_type"), "checkpoint_name": Path(a.get("checkpoint", "")).name,
                 "checkpoint_sha256": None}
    return {"name": name, "label": label or name, "dir": d, "run": run, "summary": summary,
            "model": model, "files": [d / "run.json", d / "summary.csv"] + ([model_json] if model_json.exists() else [])}


def validate_arms(arms: list[dict]) -> None:
    """Same evaluation set in every arm; distinct models."""
    ref = arms[0]
    for a in arms[1:]:
        for k in expected_grid():
            r0, r1 = ref["summary"][k], a["summary"][k]
            if (r0["n_patients"], r0["n_instances"]) != (r1["n_patients"], r1["n_instances"]):
                raise PostError(f"evaluation sets differ: {k} has {r0['n_patients']} pts / {r0['n_instances']} "
                                f"inst in {ref['name']} but {r1['n_patients']} / {r1['n_instances']} in {a['name']}")
    ident = {}
    for a in arms:
        m = a["model"]
        ident[a["name"]] = m.get("checkpoint_sha256") or f"{m.get('model_type')}/{m.get('checkpoint_name')}"
    names = list(ident)
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            # arm2_r4 == arm2_r8 is legitimate when neither run beat its epoch-0 baseline (both then
            # merge the untouched weights); results_table checks that case against the training runs.
            if {x, y} == {"arm2_r4", "arm2_r8"}:
                continue
            if ident[x] == ident[y]:
                raise PostError(f"two arms report the same model: {x} and {y} ({str(ident[x])[:16]})")


def load_compare(results: Path, b: str, a: str, arms: dict | None = None) -> dict | None:
    """compare.csv of a -> b (None if absent). Cross-checked against both arms' summaries."""
    path = Path(results) / "outputs" / b / f"compare_vs_{a}" / "compare.csv"
    if not path.exists():
        return None
    rows = {}
    for r in read_csv(path):
        key = (r["level"], r["group"], r["metric"], jkey(r["jitter"]))
        what = f"{path.parent.name}/compare.csv {key}"
        rows[key] = {k: fnum(r[k], f"{what} {k}") for k in ("mean_a", "mean_b", "delta", "ci_lo", "ci_hi")}
        rows[key].update({k: int(r[k]) for k in ("n_patients", "n_instances", "n_patients_b_better",
                                                 "n_patients_a_better")})
    missing = [k for k in expected_grid() if k not in rows]
    if missing:
        raise PostError(f"{path}: {len(missing)} expected rows missing, e.g. {missing[:3]}")
    if arms is not None:
        for k, r in rows.items():
            sa, sb = arms[a]["summary"][k], arms[b]["summary"][k]
            if abs(r["mean_a"] - sa["mean"]) > ROUND4_TOL or abs(r["mean_b"] - sb["mean"]) > ROUND4_TOL:
                raise PostError(f"{path}: {k} means ({r['mean_a']}, {r['mean_b']}) disagree with the "
                                f"summaries ({sa['mean']}, {sb['mean']})")
            if (r["n_patients"], r["n_instances"]) != (sa["n_patients"], sa["n_instances"]):
                raise PostError(f"{path}: {k} counts disagree with {a}/summary.csv")
            if abs(r["delta"] - (r["mean_b"] - r["mean_a"])) > ROUND4_TOL:
                raise PostError(f"{path}: {k} delta {r['delta']} != mean_b - mean_a")
    return {"path": path, "rows": rows}


# -------------------------------------------------------------------------- training
TRAIN_COLS = ("train_loss", "train_dice", "train_bce", "train_iou", "grad_norm", "epoch_s")
VAL_COLS = ("ct@0.00", "mri@0.00", "ct@0.10", "mri@0.10", "selection", "nsd3_ct@0.10", "nsd3_mri@0.10")


def load_training(results: Path, r: int) -> dict:
    d = Path(results) / "runs" / f"arm2_r{r}"
    return {"rank": r, "dir": d, "run": read_json(d / "run.json"), "config": read_json(d / "config.json"),
            "metrics": read_csv(d / "metrics.csv"),
            "files": [d / "run.json", d / "config.json", d / "metrics.csv"]}


def selected_rank(results: Path) -> dict:
    """runs/arm2_selection.json, with the val-before-test proof select_rank.py recorded:
    no outputs/arm2_r* folder existed at selection time, and no override was used."""
    sel = read_json(Path(results) / "runs" / "arm2_selection.json")
    if sel.get("chosen_rank") not in RANKS:
        raise PostError(f"arm2_selection.json: chosen_rank {sel.get('chosen_rank')!r}")
    if sel.get("existing_eval_folders"):
        raise PostError(f"arm2_selection.json: evaluation folders already existed at selection time "
                        f"{sel['existing_eval_folders']} -- the selection did not precede the test")
    if sel.get("ack_existing_eval") is not None:
        raise PostError("arm2_selection.json: selection made with --ack-existing-eval (override)")
    return sel


# ------------------------------------------------------------------------- instances
def load_instances(path: Path) -> dict:
    """instances.csv keyed by (case, z, side, jitter) -- eval_sam.inst_key, re-implemented."""
    out = {}
    for r in read_csv(path):
        k = (r["case"], str(r["z"]), r["side"], f"{float(r['jitter']):.4f}")
        if k in out:
            raise PostError(f"{path}: duplicate instance {k}")
        out[k] = r
    return out


def group_of(r: dict) -> str:
    return "/".join(x for x in (r["dataset"], r["modality"], r.get("sequence", "")) if x)


def pair_instances(a: dict, b: dict, what: str) -> None:
    """Identical keys; per key identical prompt and ground truth (compare_arms.pair_check)."""
    if set(a) != set(b):
        raise PostError(f"{what}: instance keys differ ({len(set(a) - set(b))} only in A, "
                        f"{len(set(b) - set(a))} only in B)")
    for k in a:
        for f in ("box_clean", "box_jit", "gt_area", "patient", "dataset", "modality", "sequence"):
            if a[k][f] != b[k][f]:
                raise PostError(f"{what}: {k} differs in {f} ({a[k][f]!r} vs {b[k][f]!r})")


def patient_deltas(a: dict, b: dict, group: str, jitter: float = HEADLINE_JITTER,
                   metric: str = "dice") -> tuple[list[str], list[float], list[float]]:
    """Per patient (within one group): mean metric under A and under B -- the same two-stage
    aggregation as aggregate_arm / compare_arms. Returns (patients, mean_a, mean_b)."""
    pa, pb = defaultdict(list), defaultdict(list)
    for k, ra in a.items():
        if group_of(ra) != group or abs(float(ra["jitter"]) - jitter) > 1e-9:
            continue
        pa[ra["patient"]].append(fnum(ra[metric], f"{k} {metric}"))
        pb[ra["patient"]].append(fnum(b[k][metric], f"{k} {metric}"))
    pats = sorted(pa)
    return (pats, [sum(pa[p]) / len(pa[p]) for p in pats], [sum(pb[p]) / len(pb[p]) for p in pats])


def fmt_ci(mean: float, lo: float, hi: float, signed: bool = False) -> str:
    f = "{:+.3f}" if signed else "{:.3f}"
    return (f + " [" + f + ", " + f + "]").format(mean, lo, hi)
