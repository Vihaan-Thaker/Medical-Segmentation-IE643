#!/usr/bin/env python
"""Phase A: verify both arm-2 training runs from the copied files, and summarise them
(ARM2_POST_PLAN.md §3.2). Read-only; Mac Python (stdlib only).

    python check_training.py --results ../results          # ends POST_OK / POST_FAILED

Checks, per rank: TRAIN_OK, 35 epochs, merge check passed; metrics.csv epochs 0..35 contiguous,
validation columns finite in every row, training columns finite in epochs 1-35 and empty in the
epoch-0 baseline row; best epoch = the first epoch attaining the maximum selection score (the
training loop's strictly-greater rule) and equal to run.json and the merged checkpoint's epoch.
Across ranks: settings identical except rank (select_rank.paired_problems); the preflight passed
and recommended the batch settings both runs used; the selection file is consistent with the
D33 rule (select_rank.decide) and proves it preceded any test result.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import post_common as P  # noqa: E402
import select_rank as S  # noqa: E402  (stdlib-only)


def check_rank(t: dict) -> dict:
    r, run, rows = t["rank"], t["run"], t["metrics"]
    tag = f"r={r}"
    if run.get("status") != "TRAIN_OK":
        raise P.PostError(f"{tag}: run.json status {run.get('status')!r}, not TRAIN_OK")
    if run.get("rank") != r:
        raise P.PostError(f"{tag}: run.json says rank {run.get('rank')}")
    epochs = run.get("settings", {}).get("epochs")
    if run.get("epochs_completed") != epochs or epochs != 35:
        raise P.PostError(f"{tag}: {run.get('epochs_completed')} of {epochs} epochs (expected 35)")
    merged = run.get("merged", {})
    chk = merged.get("check", {})
    if not chk.get("passed") or P.fnum(chk.get("pixel_agreement"), f"{tag} merge agreement") < 0.999:
        raise P.PostError(f"{tag}: merge check not passed ({chk})")
    if not merged.get("sha256"):
        raise P.PostError(f"{tag}: no merged sha256 in run.json")

    eps = [int(x["epoch"]) for x in rows]
    if eps != list(range(36)):
        raise P.PostError(f"{tag}: metrics.csv epochs {eps[:5]}...{eps[-3:]} are not 0..35 contiguous")
    nsd_on = run.get("settings", {}).get("nsd", True)
    val_cols = [c for c in P.VAL_COLS if nsd_on or not c.startswith("nsd3_")]
    for x in rows:
        e = int(x["epoch"])
        for c in val_cols:
            P.fnum(x.get(c), f"{tag} epoch {e} {c}")
        for c in P.TRAIN_COLS:
            if e == 0:
                if x.get(c, "") not in ("", None):
                    raise P.PostError(f"{tag}: epoch-0 baseline row has a training value in {c}")
            else:
                P.fnum(x.get(c), f"{tag} epoch {e} {c}")
    sel = [P.fnum(x["selection"], f"{tag} selection") for x in rows]
    best = max(sel)
    first = sel.index(best)                                  # strictly-greater rule => first max
    if first != run.get("best_epoch"):
        raise P.PostError(f"{tag}: best epoch in run.json {run.get('best_epoch')} != first max in "
                          f"metrics.csv {first}")
    if abs(best - P.fnum(run.get("best_score"), f"{tag} best_score")) > 1e-9:
        raise P.PostError(f"{tag}: best score {run.get('best_score')} != metrics.csv {best}")
    if merged.get("best_epoch") != first:
        raise P.PostError(f"{tag}: merged checkpoint is epoch {merged.get('best_epoch')}, best is {first}")
    hours = (sum(P.fnum(x["epoch_s"], "epoch_s") for x in rows[1:]) + P.fnum(rows[0].get("val_s"), "val_s")) / 3600
    return {"rank": r, "best_epoch": first, "best_score": best, "baseline": sel[0], "final": sel[-1],
            "gain": best - sel[0], "hours": hours, "mean_epoch_min": hours * 60 / 35,
            "trainable": run.get("info", {}).get("trainable"), "lora": run.get("info", {}).get("lora"),
            "gpu": run.get("gpu", {}).get("name") or run.get("gpu_type"),
            "merged_sha": merged["sha256"], "agreement": chk["pixel_agreement"],
            "best_row": rows[first], "base_row": rows[0]}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results", type=Path, required=True)
    a = p.parse_args(argv)
    try:
        tr = {r: P.load_training(a.results, r) for r in P.RANKS}
        res = {r: check_rank(tr[r]) for r in P.RANKS}
        probs = S.paired_problems(tr[4]["config"], tr[8]["config"])
        if probs:
            raise P.PostError("the two runs are not paired: " + "; ".join(probs))
        pre = P.read_json(a.results / "runs" / "arm2_preflight" / "preflight.json")
        if not pre.get("passed"):
            raise P.PostError("preflight.json: passed is not true")
        rec = pre.get("recommended", {})
        for r in P.RANKS:
            st = tr[r]["run"]["settings"]
            if (st.get("batch"), st.get("accum"), st.get("grad_checkpoint")) != \
                    (rec.get("batch"), rec.get("accum"), rec.get("grad_checkpoint")):
                raise P.PostError(f"r={r}: batch settings {st.get('batch')}x{st.get('accum')} "
                                  f"(gc {st.get('grad_checkpoint')}) differ from the preflight's {rec}")
        sel = P.selected_rank(a.results)
        d = S.decide(res[4]["best_score"], res[8]["best_score"])
        for k in ("chosen_rank", "tie", "consider_r16"):
            if sel.get(k) != d[k]:
                raise P.PostError(f"arm2_selection.json {k}={sel.get(k)!r} but the rule gives {d[k]!r}")
        for r in P.RANKS:
            if sel[f"r{r}"]["merged_sha256"] != res[r]["merged_sha"]:
                raise P.PostError(f"arm2_selection.json r{r} sha differs from the training run")
            if abs(sel[f"r{r}"]["best_score"] - res[r]["best_score"]) > 1e-9:
                raise P.PostError(f"arm2_selection.json r{r} score differs from the training run")
    except P.PostError as e:
        print(f"!! {e}\nPOST_FAILED")
        return 1

    lines = ["# Arm 2 training summary", "",
             "| | r = 4 | r = 8 |", "|---|---|---|"]

    def row(label, f):
        lines.append(f"| {label} | {f(res[4])} | {f(res[8])} |")
    row("trainable parameters", lambda x: f"{x['trainable']:,} (LoRA {x['lora']:,})")
    row("GPU", lambda x: x["gpu"])
    row("epoch 0 (off-the-shelf) val selection", lambda x: f"{x['baseline']:.4f}")
    row("best epoch", lambda x: str(x["best_epoch"]))
    row("best val selection", lambda x: f"{x['best_score']:.4f}  (+{x['gain']:.4f})")
    row("  CT / MRI Dice @10% at best", lambda x: f"{float(x['best_row']['ct@0.10']):.4f} / "
                                                   f"{float(x['best_row']['mri@0.10']):.4f}")
    row("  CT / MRI Dice @0% at best", lambda x: f"{float(x['best_row']['ct@0.00']):.4f} / "
                                                  f"{float(x['best_row']['mri@0.00']):.4f}")
    row("epoch-35 val selection", lambda x: f"{x['final']:.4f}")
    if any(res[r]["best_epoch"] == 0 for r in P.RANKS):
        lines.append("| **note** | " + " | ".join("best epoch 0: no improvement over off-the-shelf on val"
                                                  if res[r]["best_epoch"] == 0 else "" for r in P.RANKS) + " |")
    row("training time", lambda x: f"{x['hours']:.1f} h ({x['mean_epoch_min']:.1f} min/epoch)")
    row("merge check agreement", lambda x: f"{x['agreement']:.6f}")
    row("merged checkpoint sha256", lambda x: x["merged_sha"][:16] + "...")
    lines += ["", f"**Selected rank: r = {sel['chosen_rank']}** (rule: {sel.get('rule', '')}); "
              f"r8 − r4 = {sel.get('r8_minus_r4', 0):+.4f}"
              + ("; tie → smaller model" if sel.get("tie") else "")
              + ("; **r8 leads by ≥ 0.005: consider r = 16 (user's decision)**" if sel.get("consider_r16") else ""),
              f"Selection made at {sel.get('selected_utc')} (UTC) with no arm-2 evaluation folder present."]
    out = a.results / "training_summary.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}\nPOST_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
