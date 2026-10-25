#!/usr/bin/env python
"""Report figures for arms 1, 1b and 2 (ARM2_POST_PLAN.md §3.4). Read-only; Mac Python
(numpy + matplotlib; no torch / surface-distance).

    python plot_results.py --results ../results      # -> results/figures/*.png, POST_OK

  (a) training_curves.png    r4 vs r8: train loss; val Dice@10% CT / MRI and selection; epoch-0
                             baseline and best epoch marked
  (b) jitter_robustness.png  Dice 2D vs box jitter per test group, one line per arm, 95% CI band
  (c) patient_delta.png      per-patient Δ Dice (selected arm 2 − arm 1b) at 10%, per group, with the
                             paired mean + CI from compare.csv. Needs the instances.csv files
                             (fetch_results.sh --with-instances); cross-checked against compare.csv
  (d) gain_by_dataset.png    Δ Dice 2D (arm 1b → selected arm 2) at 10% and 20%, paired CIs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import post_common as P  # noqa: E402
import results_table as RT  # noqa: E402  (stdlib-only loaders + validation)

FIGURES = ("training_curves.png", "jitter_robustness.png", "patient_delta.png", "gain_by_dataset.png")
COLOR = {"arm1": "#9a9a9a", "arm1_vit_b": "#222222", "arm2_r4": "#1f77b4", "arm2_r8": "#ff7f0e"}
DPI = 200


def plt_module():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def fig_training(results: Path, out: Path, sel: dict) -> Path:
    plt = plt_module()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    for r in P.RANKS:
        t = P.load_training(results, r)
        rows = t["metrics"]
        ep = [int(x["epoch"]) for x in rows]
        c = COLOR[f"arm2_r{r}"]
        ax1.plot(ep[1:], [float(x["train_loss"]) for x in rows[1:]], color=c, label=f"r={r}")
        sel_ = [float(x["selection"]) for x in rows]
        ax2.plot(ep, [float(x["ct@0.10"]) for x in rows], color=c, ls="-", lw=1, alpha=0.8, label=f"r={r} CT")
        ax2.plot(ep, [float(x["mri@0.10"]) for x in rows], color=c, ls="--", lw=1, alpha=0.8, label=f"r={r} MRI")
        ax2.plot(ep, sel_, color=c, lw=2.2, label=f"r={r} selection (mean CT, MRI)")
        b = t["run"]["best_epoch"]
        ax2.plot([b], [sel_[b]], marker="*", ms=14, color=c, mec="k", zorder=5)
        ax2.axhline(sel_[0], color=c, ls=":", lw=1)
    ax1.set(xlabel="epoch", ylabel="train loss (Dice + BCE + IoU-MSE)", title="Training loss")
    ax2.set(xlabel="epoch", ylabel="AMOS-val Dice @10% jitter",
            title="Validation\n(dotted: epoch-0 off-the-shelf; ★ best epoch)")
    ax1.legend()
    ax2.legend(fontsize=7, ncol=2, loc="center right")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
    fig.suptitle(f"Arm 2 training, LoRA r = 4 vs r = 8 (selected on AMOS val: r = {sel['chosen_rank']})")
    fig.tight_layout()
    path = out / "training_curves.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def fig_jitter(arms: dict, out: Path, chosen: str) -> Path:
    plt = plt_module()
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharey=True)
    x = [100 * j for j in P.JITTERS]
    for ax, g in zip(axes.ravel(), P.GROUPS):
        for name, label in P.ARMS:
            s = [arms[name]["summary"][("2d", g, "dice", P.jkey(j))] for j in P.JITTERS]
            m = [v["mean"] for v in s]
            ax.plot(x, m, marker="o", color=COLOR[name], lw=2.4 if name == chosen else 1.4,
                    label=label + (" (selected)" if name == chosen else ""))
            ax.fill_between(x, [v["ci_lo"] for v in s], [v["ci_hi"] for v in s], color=COLOR[name], alpha=0.12)
        ax.set(title=g, xticks=x, xlabel="box jitter (% of box side)")
        ax.grid(alpha=0.3)
    for ax in axes[:, 0]:
        ax.set_ylabel("Dice 2D (mean over patients)")
    axes[0, 0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Robustness to the box prompt: Dice 2D vs jitter (95% CI over patients)")
    fig.tight_layout()
    path = out / "jitter_robustness.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def fig_patients(results: Path, out: Path, chosen: str, compare: dict) -> Path | None:
    pa = results / "outputs" / "arm1_vit_b" / "instances.csv"
    pb = results / "outputs" / chosen / "instances.csv"
    if not (pa.exists() and pb.exists()):
        print(f"  (c) skipped: {pa.name} for arm1_vit_b and/or {chosen} not fetched "
              f"(run fetch_results.sh --with-instances)")
        return None
    A, B = P.load_instances(pa), P.load_instances(pb)
    P.pair_instances(A, B, f"arm1_vit_b vs {chosen} instances")
    per_group = []                      # all cross-checks first, so a failure never leaves a half figure
    for g in P.GROUPS:
        pats, ma, mb = P.patient_deltas(A, B, g)
        d = np.array(mb) - np.array(ma)
        c = compare["rows"][("2d", g, "dice", P.jkey(P.HEADLINE_JITTER))]
        if len(pats) != c["n_patients"]:
            raise P.PostError(f"{g}: {len(pats)} patients in instances.csv, {c['n_patients']} in compare.csv")
        if abs(float(d.mean()) - c["delta"]) > P.ROUND4_TOL:
            raise P.PostError(f"{g}: per-patient mean Δ {d.mean():.5f} != compare.csv delta {c['delta']:.4f}")
        per_group.append((g, d, c))
    plt = plt_module()
    fig, ax = plt.subplots(figsize=(11, 4.6))
    rng = np.random.default_rng(643)
    for i, (g, d, c) in enumerate(per_group):
        ax.scatter(i + rng.uniform(-0.18, 0.18, d.size), d, s=10, alpha=0.55,
                   color="#2ca02c" if g in P.IN_DOMAIN else "#9467bd")
        ax.errorbar([i + 0.3], [c["delta"]], yerr=[[c["delta"] - c["ci_lo"]], [c["ci_hi"] - c["delta"]]],
                    fmt="D", color="k", capsize=4)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(P.GROUPS)), P.GROUPS, rotation=15)
    ax.set(ylabel=f"Δ Dice 2D per patient ({chosen} − arm1_vit_b)",
           title="Per-patient change at 10% jitter (green: in-domain AMOS, purple: external); ◆ paired mean + 95% CI")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = out / "patient_delta.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def fig_gain(out: Path, chosen: str, compare: dict) -> Path:
    plt = plt_module()
    fig, ax = plt.subplots(figsize=(10, 4.4))
    w = 0.38
    for k, (j, col) in enumerate(((0.10, "#1f77b4"), (0.20, "#d62728"))):
        rows = [compare["rows"][("2d", g, "dice", P.jkey(j))] for g in P.GROUPS]
        d = np.array([r["delta"] for r in rows])
        err = np.array([[r["delta"] - r["ci_lo"] for r in rows], [r["ci_hi"] - r["delta"] for r in rows]])
        xs = np.arange(len(P.GROUPS)) + (k - 0.5) * w
        ax.bar(xs, d, w, yerr=err, capsize=3, color=col, alpha=0.85, label=f"{int(j * 100)}% jitter")
    ax.axvline(len(P.IN_DOMAIN) - 0.5, color="k", ls="--", lw=0.8)
    # Region labels just above the axes (data-independent: correct even if gains are negative).
    tr = ax.get_xaxis_transform()
    ax.text((len(P.IN_DOMAIN) - 1) / 2, 1.01, "in-domain (AMOS)", transform=tr, ha="center", va="bottom", fontsize=8)
    ax.text((len(P.IN_DOMAIN) + len(P.GROUPS) - 1) / 2, 1.01, "external (WORD, CHAOS)", transform=tr,
            ha="center", va="bottom", fontsize=8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(P.GROUPS)), P.GROUPS, rotation=15)
    ax.set_ylabel("Δ Dice 2D (paired, 95% CI)")
    ax.set_title(f"Gain from fine-tuning: arm 1b → {chosen}", pad=18)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = out / "gain_by_dataset.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results", type=Path, required=True)
    a = p.parse_args(argv)
    out = a.results / "figures"
    try:
        data = RT.load_all(a.results)
        chosen = f"arm2_r{data['sel']['chosen_rank']}"
        compare = data["compares"][(chosen, "arm1_vit_b")]
        if compare is None:
            raise P.PostError(f"outputs/{chosen}/compare_vs_arm1_vit_b/compare.csv missing -- run the arm-2 eval")
        out.mkdir(parents=True, exist_ok=True)
        for f in FIGURES:                   # our own outputs: never leave a stale figure behind
            (out / f).unlink(missing_ok=True)
        written = [fig_training(a.results, out, data["sel"]), fig_jitter(data["arms"], out, chosen),
                   fig_patients(a.results, out, chosen, compare), fig_gain(out, chosen, compare)]
    except P.PostError as e:
        print(f"!! {e}\nPOST_FAILED")
        return 1
    for w in written:
        if w is not None:
            print(f"wrote {w}")
    print("POST_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
