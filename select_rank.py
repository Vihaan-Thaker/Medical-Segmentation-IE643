#!/usr/bin/env python
"""Choose the arm-2 LoRA rank from AMOS-val results only, by the rule fixed in D33, BEFORE any
test result exists (ARM2_CODE_PLAN.md §3.6).

    python select_rank.py --runs .../runs --outputs .../outputs     # ends SELECT_OK / SELECT_FAILED

Rule: arm 2 = the rank with the higher best-epoch val selection score (mean of CT and MRI
patient-mean Dice at 10% jitter). A difference below 0.002 goes to r = 4, the smaller model.
"Consider r = 16" is flagged when r8 - r4 >= 0.005.

Refuses unless both runs finished (TRAIN_OK, merged + merge-checked), and their configs differ
ONLY in rank: same data, base checkpoint, GPU type, sampling, jitter, validation protocol,
optimiser settings, batch, epochs and seed. Otherwise the comparison is not paired.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

TIE = 0.002
GO_HIGHER = 0.005
RANKS = (4, 8)


def decide(s4: float, s8: float) -> dict:
    diff = s8 - s4
    chosen = 4 if abs(diff) < TIE else (8 if diff > 0 else 4)
    return {"chosen_rank": chosen, "r8_minus_r4": diff, "tie": abs(diff) < TIE,
            "consider_r16": diff >= GO_HIGHER}


def load_run(runs: Path, r: int) -> tuple[dict, dict, list[str]]:
    probs = []
    rd = runs / f"arm2_r{r}"
    rj = json.loads((rd / "run.json").read_text()) if (rd / "run.json").exists() else {}
    cfg = json.loads((rd / "config.json").read_text()) if (rd / "config.json").exists() else {}
    if not rj:
        probs.append(f"{rd}/run.json missing")
        return rj, cfg, probs
    if rj.get("status") != "TRAIN_OK":
        probs.append(f"r{r}: status {rj.get('status')}")
    if rj.get("rank") != r:
        probs.append(f"r{r}: run.json says rank {rj.get('rank')}")
    epochs = rj.get("settings", {}).get("epochs")
    if rj.get("epochs_completed") != epochs:
        probs.append(f"r{r}: {rj.get('epochs_completed')} of {epochs} epochs")
    if not rj.get("merged", {}).get("check", {}).get("passed"):
        probs.append(f"r{r}: not finalised (no passed merge check)")
    if not cfg:
        probs.append(f"{rd}/config.json missing")
    return rj, cfg, probs


def paired_problems(c4: dict, c8: dict) -> list[str]:
    probs = []
    s4, s8 = dict(c4.get("settings", {})), dict(c8.get("settings", {}))
    if s4.pop("rank", None) != 4 or s8.pop("rank", None) != 8:
        probs.append("config ranks are not 4 and 8")
    diff = sorted(k for k in set(s4) | set(s8) if s4.get(k) != s8.get(k))
    if diff:
        probs.append(f"settings differ beyond rank: {diff}")
    for k in ("base_sha256", "manifest_sha256", "gpu_type", "code"):
        if c4.get(k) != c8.get(k):
            probs.append(f"{k} differs between the two runs")
    return probs


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--outputs", type=Path, required=True)
    p.add_argument("--ack-existing-eval", default=None, metavar="REASON")
    a = p.parse_args(argv)

    problems = []
    runs = {}
    for r in RANKS:
        rj, cfg, pr = load_run(a.runs, r)
        runs[r] = (rj, cfg)
        problems += pr
    if not problems:
        problems += paired_problems(runs[4][1], runs[8][1])
    # evaluation results are FOLDERS outputs/arm2_r4, arm2_r8; the training logs
    # outputs/arm2_r*_train.<job>.out/.err also live in outputs/ and must not count
    existing = sorted(str(x) for x in a.outputs.glob("arm2_r*") if x.is_dir()) if a.outputs.exists() else []
    if existing and not a.ack_existing_eval:
        problems.append(f"evaluation folders already exist {existing}: the selection must precede "
                        f"any test result (--ack-existing-eval \"<reason>\" to override)")
    if problems:
        print("PROBLEMS:")
        for m in problems:
            print(f"  !! {m}")
        print("SELECT_FAILED")
        return 1

    s4, s8 = runs[4][0]["best_score"], runs[8][0]["best_score"]
    d = decide(s4, s8)
    out = {**d, "r4": {"best_score": s4, "best_epoch": runs[4][0]["best_epoch"],
                       "merged_sha256": runs[4][0]["merged"]["sha256"]},
           "r8": {"best_score": s8, "best_epoch": runs[8][0]["best_epoch"],
                  "merged_sha256": runs[8][0]["merged"]["sha256"]},
           "rule": f"higher best-epoch AMOS-val selection score (mean CT/MRI patient-mean Dice at "
                   f"10% jitter); |r8 - r4| < {TIE} -> r4; consider r16 iff r8 - r4 >= {GO_HIGHER}",
           "gpu_type": runs[4][1].get("gpu_type"),
           "selected_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "existing_eval_folders": existing, "ack_existing_eval": a.ack_existing_eval}
    a.runs.mkdir(parents=True, exist_ok=True)
    path = a.runs / "arm2_selection.json"
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    print(f"r4 best val {s4:.4f} (epoch {out['r4']['best_epoch']}), r8 best val {s8:.4f} "
          f"(epoch {out['r8']['best_epoch']}); r8 - r4 = {d['r8_minus_r4']:+.4f}")
    print(f"chosen rank: {d['chosen_rank']}" + (" (tie -> smaller model)" if d["tie"] else "")
          + ("; r8 leads by >= 0.005: consider r = 16" if d["consider_r16"] else ""))
    print(f"wrote {path}\nSELECT_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
