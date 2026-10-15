#!/usr/bin/env python
"""Segmentation metrics and aggregation for the IE643 arm comparison.

ONE scoring implementation shared by all four arms (DECISIONS.md D22, D24a). If an arm
ever scores itself differently, the four rows stop being comparable, so every arm must
import from here.

Dice and NSD come from Google DeepMind's `surface-distance`, which weights boundary
elements by their physical size (contour length in 2D, surfel area in 3D). MONAI and
Metrics Reloaded instead *count* boundary elements under the tolerance, which deviates by
-37.5 to +7.0 percentage points on anisotropic grids (arXiv 2410.02630) -- exactly our 3D
case at e.g. 0.7 x 0.7 x 5.0 mm. See D24a/C1.

    pip install git+https://github.com/google-deepmind/surface-distance.git

Three rules the callers must respect, each enforced here rather than trusted (D24a):

  C2  spacing is (row, col) in 2D and (slice, row, col) in 3D, matching the array axes.
      Verified: swapping a 2D pair turns NSD@2 = 1.0000 into 0.7455 on identical masks.
  C3  masks must be numpy bool. `surface-distance` rejects uint8 outright.
  C6  ground truth must be non-empty. Two empty masks give NaN, which would silently
      poison any mean it entered.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

try:
    import surface_distance as _sd
except ImportError as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "surface-distance is required. Install it on the LOGIN node (compute nodes have "
        "no internet):\n"
        "  pip install git+https://github.com/google-deepmind/surface-distance.git"
    ) from exc

# D24a/C5: the headline tolerance must be >= the coarsest in-plane pixel in the evaluation
# set, otherwise it is sub-pixel and unmeasurable. Measured on job 336480: CHAOS reaches
# 2.0313 mm, so 2 mm is sub-pixel there and the rule selects 3 mm (D26). All four
# tolerances are always computed and stored; only the headline column changes.
TOLERANCES_MM: tuple[float, ...] = (1.0, 2.0, 3.0, 5.0)
HEADLINE_TOLERANCE_MM = 3.0

# Fraction of box side length used as the std of the Gaussian box jitter, capped at
# JITTER_MAX_PX. 0.10 is SAM's own training recipe (D5); 0.0 is the clean oracle box.
JITTER_LEVELS: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20)
HEADLINE_JITTER = 0.10
JITTER_MAX_PX = 20.0

RIGHT, LEFT = 1, 2
SIDE_NAME = {RIGHT: "right", LEFT: "left"}


class MetricError(ValueError):
    """A scoring precondition was violated. Never swallowed -- a silent NaN here would
    propagate into a reported mean."""


# --------------------------------------------------------------------------- checks
def _check_masks(gt: np.ndarray, pred: np.ndarray, spacing: tuple[float, ...]) -> None:
    for name, m in (("gt", gt), ("pred", pred)):
        if not isinstance(m, np.ndarray):
            raise MetricError(f"{name} must be a numpy array, got {type(m).__name__}")
        if m.dtype != np.bool_:
            raise MetricError(f"{name} must be dtype bool, got {m.dtype} (D24a/C3)")
    if gt.shape != pred.shape:
        raise MetricError(f"shape mismatch: gt {gt.shape} vs pred {pred.shape}")
    if len(spacing) != gt.ndim:
        raise MetricError(
            f"spacing has {len(spacing)} entries for a {gt.ndim}-D mask; it must be "
            f"{'(row, col)' if gt.ndim == 2 else '(slice, row, col)'} (D24a/C2)"
        )
    if any(not math.isfinite(s) or s <= 0 for s in spacing):
        raise MetricError(f"spacing must be finite and positive, got {spacing}")
    if not gt.any():
        raise MetricError("ground truth is empty; nothing to score (D24a/C6)")


def dice_coefficient(gt: np.ndarray, pred: np.ndarray) -> float:
    """2|A and B| / (|A| + |B|). Written out rather than imported so the definition is
    visible and testable; the test suite asserts it equals the reference implementation."""
    denom = int(gt.sum()) + int(pred.sum())
    if denom == 0:
        raise MetricError("both masks empty; Dice undefined (D24a/C6)")
    return 2.0 * float(np.logical_and(gt, pred).sum()) / denom


def score_masks(gt: np.ndarray, pred: np.ndarray, spacing: tuple[float, ...],
                tolerances: tuple[float, ...] = TOLERANCES_MM) -> dict:
    """Dice plus NSD at each tolerance, for a 2D slice or a 3D volume.

    `spacing` is mm per array axis, in axis order (D24a/C2). The surface distances are
    computed once and reused for every tolerance.
    """
    _check_masks(gt, pred, spacing)
    out = {"dice": dice_coefficient(gt, pred),
           "gt_vox": int(gt.sum()), "pred_vox": int(pred.sum())}
    if not pred.any():
        # Verified behaviour: NSD(gt, empty) = 0.0 cleanly. Short-circuited anyway so the
        # all-inf distance path is never relied on.
        out.update({f"nsd_{t:g}": 0.0 for t in tolerances})
        return out
    d = _sd.compute_surface_distances(gt, pred, spacing_mm=tuple(float(s) for s in spacing))
    for t in tolerances:
        v = _sd.compute_surface_dice_at_tolerance(d, float(t))
        if not math.isfinite(v):
            raise MetricError(f"NSD@{t} came back non-finite on non-empty masks")
        out[f"nsd_{t:g}"] = float(v)
    return out


# ----------------------------------------------------------------------- box jitter
def jitter_seed(case: str, z: int, side: int, level: float) -> int:
    """Deterministic per (instance, level): reproducible regardless of ordering, resume
    or parallelism (D24a). Derived from a stable hash, not Python's salted hash()."""
    key = f"{case}|{z}|{side}|{level:.4f}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


def jitter_box(box: tuple[int, int, int, int], level: float, h: int, w: int,
               seed: int) -> tuple[int, int, int, int]:
    """SAM's recipe (D5): Gaussian noise per coordinate with std = `level` x box side
    length, capped at 20 px. Clamped to the image and to a minimum 1 px side, so a large
    jitter on a small box can never produce an inverted or out-of-bounds box (D24a/C7).
    """
    x0, y0, x1, y1 = (int(v) for v in box)
    if level <= 0:
        return x0, y0, x1, y1
    rng = np.random.default_rng(seed)
    sx = min(level * (x1 - x0 + 1), JITTER_MAX_PX)
    sy = min(level * (y1 - y0 + 1), JITTER_MAX_PX)
    nx0, nx1 = x0 + rng.normal(0, sx), x1 + rng.normal(0, sx)
    ny0, ny1 = y0 + rng.normal(0, sy), y1 + rng.normal(0, sy)
    if nx1 < nx0:
        nx0, nx1 = nx1, nx0
    if ny1 < ny0:
        ny0, ny1 = ny1, ny0
    jx0 = int(np.clip(round(nx0), 0, w - 1))
    jx1 = int(np.clip(round(nx1), 0, w - 1))
    jy0 = int(np.clip(round(ny0), 0, h - 1))
    jy1 = int(np.clip(round(ny1), 0, h - 1))
    if jx1 == jx0:
        jx1 = min(jx0 + 1, w - 1)
        jx0 = max(jx1 - 1, 0)
    if jy1 == jy0:
        jy1 = min(jy0 + 1, h - 1)
        jy0 = max(jy1 - 1, 0)
    return jx0, jy0, jx1, jy1


def box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """IoU of two inclusive XYXY boxes. Recorded per instance so the jitter degradation
    curve can be read against how bad the prompt actually was (D24a/C7)."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    iw = max(0, min(ax1, bx1) - max(ax0, bx0) + 1)
    ih = max(0, min(ay1, by1) - max(ay0, by0) + 1)
    inter = iw * ih
    area_a = (ax1 - ax0 + 1) * (ay1 - ay0 + 1)
    area_b = (bx1 - bx0 + 1) * (by1 - by0 + 1)
    union = area_a + area_b - inter
    return float(inter) / float(union) if union > 0 else 0.0


# ----------------------------------------------------------------------- aggregation
def bootstrap_ci(values: list[float], n_boot: int = 10000, seed: int = 643,
                 alpha: float = 0.05) -> tuple[float, float, float]:
    """Mean and percentile bootstrap CI, resampling the given values (one per PATIENT).

    Bootstrap rather than a normal approximation because AMOS MRI has only 12 patients
    (D22), where a t-interval leans on a normality assumption we cannot check.
    """
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    if v.size == 1:
        return float(v[0]), float(v[0]), float(v[0])
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
    return (float(v.mean()),
            float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def aggregate_by_patient(rows: list[dict], metric: str, n_boot: int = 10000,
                         seed: int = 643) -> dict:
    """Mean within each patient, then across patients, with a bootstrap CI over patients.

    Two-stage because one AMOS MRI patient holds 19.5% of that set's instances (D22a):
    a flat mean over instances would let a single patient drive a fifth of the result,
    and would also pretend the CI rests on thousands of independent samples.
    """
    per_patient: dict[str, list[float]] = {}
    for r in rows:
        v = r.get(metric)
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            continue
        per_patient.setdefault(r["patient"], []).append(float(v))
    patient_means = [float(np.mean(v)) for v in per_patient.values() if v]
    mean, lo, hi = bootstrap_ci(patient_means, n_boot=n_boot, seed=seed)
    return {"metric": metric, "n_patients": len(patient_means), "n_instances": len(rows),
            "mean": mean, "ci_lo": lo, "ci_hi": hi,
            "patient_means": patient_means}


# ------------------------------------------------------------------------ self-test
# Values measured with this implementation on 2026-10-06 (D24a). Asserted before every
# run so a dependency change cannot silently alter scoring.
SELF_TEST_EXPECTED = {
    ("identical", 1.0): 1.0,
    ("shift2", 1.0): 0.5775, ("shift2", 2.0): 1.0,
    ("shift5", 1.0): 0.2129, ("shift5", 2.0): 0.3333,
    ("shift5", 3.0): 0.5000, ("shift5", 5.0): 1.0,
}
SELF_TEST_DICE = {"identical": 1.0, "shift2": 0.9575, "shift5": 0.8947}


def _disk(h: int, w: int, cy: float, cx: float, r: float) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r


def self_test(tol: float = 5e-4) -> None:
    """Assert the recorded sanity values. Fast (<1 s); run before touching real data so a
    wrong `surface-distance` version fails in seconds, not after a 40-minute GPU pass."""
    gt = _disk(200, 200, 100, 100, 30)
    sp = (1.0, 1.0)
    cases = {"identical": gt, "shift2": np.roll(gt, 2, axis=1), "shift5": np.roll(gt, 5, axis=1)}
    for name, pred in cases.items():
        s = score_masks(gt, pred, sp)
        if abs(s["dice"] - SELF_TEST_DICE[name]) > tol:
            raise MetricError(f"self-test Dice {name}: {s['dice']:.4f} != {SELF_TEST_DICE[name]}")
        for (cname, t), want in SELF_TEST_EXPECTED.items():
            if cname != name:
                continue
            got = s[f"nsd_{t:g}"]
            if abs(got - want) > tol:
                raise MetricError(f"self-test NSD@{t} {name}: {got:.4f} != {want}")

    # C2: asymmetric spacing must change the answer, proving spacing reaches the metric
    # in axis order rather than being ignored or transposed.
    pred_rows = np.roll(gt, 2, axis=0)
    a = score_masks(gt, pred_rows, (3.0, 1.0))["nsd_2"]
    b = score_masks(gt, pred_rows, (1.0, 3.0))["nsd_2"]
    if not (a < 0.9 < b):
        raise MetricError(f"self-test spacing order: expected (3,1)<0.9<(1,3), got {a:.4f}, {b:.4f}")

    # C6 / C3: preconditions are enforced, not assumed.
    for bad, why in (
        (lambda: score_masks(gt.astype(np.uint8), gt.astype(np.uint8), sp), "uint8 masks"),
        (lambda: score_masks(np.zeros_like(gt), np.zeros_like(gt), sp), "both masks empty"),
        (lambda: score_masks(gt, gt, (1.0,)), "wrong spacing length"),
    ):
        try:
            bad()
        except MetricError:
            continue
        raise MetricError(f"self-test: {why} should have raised MetricError")

    # Empty prediction scores zero rather than raising.
    s = score_masks(gt, np.zeros_like(gt), sp)
    if s["dice"] != 0.0 or s["nsd_2"] != 0.0:
        raise MetricError(f"self-test empty prediction: {s}")


if __name__ == "__main__":
    self_test()
    print("sam_metrics self-test OK")
