#!/usr/bin/env python
"""The students' shared contract (ARM3_CODE_PLAN.md §3.2): stored slice -> network input,
network output -> native label map, losses, LR schedule and the validation metric.

Training, validation and the later test evaluation all go through these functions, so a slice
is processed one way only.

* Letterbox: longest side -> `size` (512), top-left, padding = the normalised value of a black
  pixel. Geometry uses integer arithmetic only, so it cannot differ between machines. A slice
  already size x size is never interpolated (bit-exact).
* Always on the CPU in float32 (letterboxing); logits are mapped back on the model's device.
* Classes 0 / 1 / 2 = background / right kidney / left kidney, as the stored masks (D16).
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

NET_SIZE = 512
N_CLASSES = 3
CLASS_NAMES = ("background", "right", "left")
SIDES = ((1, "rk"), (2, "lk"))          # (mask value, manifest prefix), right first (as eval)
VARIANTS = ("plain", "box")
DICE_SMOOTH = 1e-5                      # nnU-Net MemoryEfficientSoftDiceLoss smooth
NO_BOX_LEVEL = "none"                   # the box-free student's single validation "level"


# ----------------------------------------------------------------------- normalisation
@dataclass(frozen=True)
class Norm:
    """One global linear intensity map (D42): x = (u8 - mean) / std."""
    mean: float
    std: float

    def __post_init__(self):
        if not (math.isfinite(self.mean) and math.isfinite(self.std) and self.std > 0):
            raise ValueError(f"invalid normalisation mean={self.mean} std={self.std}")

    @property
    def pad_value(self) -> float:
        return (0.0 - self.mean) / self.std

    def as_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std, "pad_value": self.pad_value}


def norm_from_buffer(buf: np.ndarray, chunk: int = 1 << 26) -> Norm:
    """Mean/std over every uint8 pixel, exact integer accumulation (deterministic). Counted
    in chunks: np.bincount converts its input to int64, which on the ~2.9e9-pixel training
    buffer would be a ~23 GB temporary."""
    if buf.dtype != np.uint8 or buf.size == 0:
        raise ValueError("normalisation needs a non-empty uint8 buffer")
    flat = buf.reshape(-1)
    counts = np.zeros(256, dtype=np.int64)
    for a in range(0, flat.size, int(chunk)):
        counts += np.bincount(flat[a:a + int(chunk)], minlength=256)
    v = np.arange(256, dtype=np.int64)
    n = int(counts.sum())
    s1, s2 = int((counts * v).sum()), int((counts * v * v).sum())
    mean = s1 / n
    var = s2 / n - mean * mean
    return Norm(float(mean), float(math.sqrt(max(var, 0.0))))


# --------------------------------------------------------------------------- geometry
def letterbox_geometry(h: int, w: int, size: int = NET_SIZE) -> dict:
    """Integer geometry: scale = size / max(h, w); h2, w2 rounded half-up."""
    h, w, size = int(h), int(w), int(size)
    if h < 1 or w < 1:
        raise ValueError(f"bad slice size {h}x{w}")
    m = max(h, w)
    h2 = (2 * h * size + m) // (2 * m)
    w2 = (2 * w * size + m) // (2 * m)
    if max(h2, w2) != size or min(h2, w2) < 1:
        raise AssertionError(f"letterbox {h}x{w} -> {h2}x{w2} (size {size})")
    return {"h": h, "w": w, "h2": int(h2), "w2": int(w2), "m": m, "size": size}


def _resized(g: dict) -> bool:
    return (g["h2"], g["w2"]) != (g["h"], g["w"])


def to_net_image(img_u8: np.ndarray, norm: Norm, size: int = NET_SIZE):
    """uint8 h x w -> (x float32 1 x size x size, valid bool size x size, geometry)."""
    if img_u8.dtype != np.uint8 or img_u8.ndim != 2:
        raise ValueError(f"expected a uint8 2D slice, got {img_u8.dtype} {img_u8.shape}")
    g = letterbox_geometry(*img_u8.shape, size)
    x = (torch.from_numpy(np.ascontiguousarray(img_u8)).to(torch.float32) - norm.mean) / norm.std
    if _resized(g):
        x = F.interpolate(x[None, None], (g["h2"], g["w2"]), mode="bilinear",
                          align_corners=False, antialias=True)[0, 0]
    out = torch.full((1, size, size), norm.pad_value, dtype=torch.float32)
    out[0, :g["h2"], :g["w2"]] = x
    valid = torch.zeros((size, size), dtype=torch.bool)
    valid[:g["h2"], :g["w2"]] = True
    return out, valid, g


def to_net_label(msk_u8: np.ndarray, g: dict) -> torch.Tensor:
    """Stored 0/1/2 mask -> int64 size x size in the network frame (nearest only)."""
    if msk_u8.shape != (g["h"], g["w"]):
        raise ValueError(f"mask {msk_u8.shape} != slice {(g['h'], g['w'])}")
    m = torch.from_numpy(np.ascontiguousarray(msk_u8)).to(torch.float32)
    if _resized(g):
        m = F.interpolate(m[None, None], (g["h2"], g["w2"]), mode="nearest-exact")[0, 0]
    out = torch.zeros((g["size"], g["size"]), dtype=torch.int64)
    out[:g["h2"], :g["w2"]] = m.round().to(torch.int64)
    return out


def box_to_frame(box, g: dict) -> tuple[int, int, int, int]:
    """Inclusive native XYXY -> inclusive network-frame XYXY. x0' = floor(x0 s),
    x1' = ceil((x1 + 1) s) - 1, s = size / m; exact integer arithmetic; clipped to the content."""
    x0, y0, x1, y1 = (int(v) for v in box)
    if not (0 <= x0 <= x1 < g["w"] and 0 <= y0 <= y1 < g["h"]):
        raise ValueError(f"box {box} outside the {g['h']}x{g['w']} slice")
    S, m = g["size"], g["m"]
    fx0, fy0 = (x0 * S) // m, (y0 * S) // m
    fx1 = -((-(x1 + 1) * S) // m) - 1
    fy1 = -((-(y1 + 1) * S) // m) - 1
    fx1, fy1 = min(fx1, g["w2"] - 1), min(fy1, g["h2"] - 1)
    fx0, fy0 = min(fx0, fx1), min(fy0, fy1)
    return fx0, fy0, fx1, fy1


def box_channels(boxes_by_side: dict, g: dict) -> torch.Tensor:
    """{1: box | None, 2: box | None} -> float32 2 x size x size (right, left); 1 inside."""
    if set(boxes_by_side) - {1, 2}:
        raise ValueError(f"unknown sides {set(boxes_by_side)}")
    out = torch.zeros((2, g["size"], g["size"]), dtype=torch.float32)
    for c, side in enumerate((1, 2)):
        b = boxes_by_side.get(side)
        if b is not None:
            x0, y0, x1, y1 = box_to_frame(b, g)
            out[c, y0:y1 + 1, x0:x1 + 1] = 1.0
    return out


def net_input(img_u8: np.ndarray, norm: Norm, variant: str, boxes_by_side: dict | None = None,
              size: int = NET_SIZE):
    """-> (x C x size x size, valid, geometry); C = 1 (plain) or 3 (box: image, right, left)."""
    if variant not in VARIANTS:
        raise ValueError(f"variant {variant!r} not in {VARIANTS}")
    x, valid, g = to_net_image(img_u8, norm, size)
    if variant == "plain":
        if boxes_by_side is not None:
            raise ValueError("the plain student takes no boxes")
        return x, valid, g
    if boxes_by_side is None:
        raise ValueError("the box student needs boxes_by_side (use {} for none)")
    return torch.cat([x, box_channels(boxes_by_side, g)], 0), valid, g


def in_channels(variant: str) -> int:
    return {"plain": 1, "box": 3}[variant]


# ------------------------------------------------------------------- back to the slice
def to_native_logits(logits: torch.Tensor, g: dict) -> torch.Tensor:
    """C x size x size (any device) -> float32 C x h x w on the same device."""
    lg = logits[:, :g["h2"], :g["w2"]].float()
    if _resized(g):
        lg = F.interpolate(lg[None], (g["h"], g["w"]), mode="bilinear", align_corners=False,
                           antialias=False)[0]
    return lg


@torch.no_grad()
def predict_native_batch(model, xs: list, geoms: list, device) -> list:
    """Network inputs (C x size x size, CPU) -> [(labels uint8 h x w, native logits float32
    C x h x w on CPU)]. fp32, eval mode, no grad; argmax ties -> lowest class index."""
    was = model.training
    model.eval()
    try:
        out = model(torch.stack(xs).to(device, torch.float32))
        res = []
        for k, g in enumerate(geoms):
            lg = to_native_logits(out[k], g)
            res.append((lg.argmax(0).to(torch.uint8).cpu().numpy(), lg.cpu()))
        return res
    finally:
        model.train(was)


def predict_native(model, img_u8, norm: Norm, variant: str, boxes_by_side=None, device="cpu",
                   size: int = NET_SIZE):
    x, _, g = net_input(img_u8, norm, variant, boxes_by_side, size)
    return predict_native_batch(model, [x], [g], device)[0]


def load_student(path, device="cpu"):
    """best.pt -> (model in eval mode on `device`, Norm, variant, frame size, raw checkpoint).
    The one loader for evaluation: rebuilds the exact architecture and refuses anything
    inconsistent (strict state-dict keys, variant vs input channels, stored pad value)."""
    import student_unet as U        # local import: student_core stays importable on its own
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck["arch"]
    if ck["variant"] not in VARIANTS or in_channels(ck["variant"]) != a["in_ch"]:
        raise ValueError(f"{path}: variant {ck['variant']!r} does not match in_ch {a['in_ch']}")
    model = U.StudentUNet(a["in_ch"], a["n_classes"], tuple(a["channels"]), a["deep_supervision"])
    model.load_state_dict(ck["model"], strict=True)
    norm = Norm(float(ck["norm"]["mean"]), float(ck["norm"]["std"]))
    if abs(norm.pad_value - float(ck["norm"]["pad_value"])) > 1e-9:
        raise ValueError(f"{path}: stored pad value inconsistent with mean/std")
    return model.to(device).eval(), norm, ck["variant"], int(ck["size"]), ck


# ------------------------------------------------------------------------------ losses
def ce_dice_loss(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor,
                 smooth: float = DICE_SMOOTH) -> dict:
    """CE over valid pixels + batch soft Dice over the foreground classes (nnU-Net:
    batch_dice, do_bg False). fp32. Returns total, ce, dice (dice = 1 - mean class Dice)."""
    logits = logits.float()
    vf = valid.to(logits.dtype)
    nv = vf.sum()
    if float(nv) <= 0:
        raise ValueError("no valid pixels in the batch")
    ce = (F.cross_entropy(logits, target, reduction="none") * vf).sum() / nv
    probs = torch.softmax(logits, 1)
    ds = []
    for c in range(1, logits.shape[1]):
        g = (target == c).to(logits.dtype) * vf
        p = probs[:, c] * vf
        inter, denom = (p * g).sum(), p.sum() + g.sum()
        ds.append((2.0 * inter + smooth) / torch.clamp(denom + smooth, min=1e-8))
    dice = 1.0 - torch.stack(ds).mean()
    return {"total": ce + dice, "ce": ce, "dice": dice}


def ds_weights(n: int) -> list[float]:
    w = [1.0 / (2 ** i) for i in range(n)]
    s = sum(w)
    return [v / s for v in w]


def ds_loss(outputs, target: torch.Tensor, valid: torch.Tensor) -> dict:
    """Deep supervision: output k is compared with target[:, ::2^k, ::2^k] (nearest); weights
    1/2^k normalised. A single tensor (no DS) is scored at full resolution only."""
    outs = outputs if isinstance(outputs, (list, tuple)) else [outputs]
    ws = ds_weights(len(outs))
    total = None
    first = None
    for k, (o, w) in enumerate(zip(outs, ws)):
        st = 2 ** k
        t, v = target[:, ::st, ::st], valid[:, ::st, ::st]
        if o.shape[-2:] != t.shape[-2:]:
            raise AssertionError(f"DS output {k}: {tuple(o.shape[-2:])} != target {tuple(t.shape[-2:])}")
        part = ce_dice_loss(o, t, v)
        total = part["total"] * w if total is None else total + part["total"] * w
        if k == 0:
            first = part
    return {"total": total, "ce": first["ce"], "dice": first["dice"]}


# ---------------------------------------------------------------------------- schedule
def poly_lr(step: int, total: int, base: float, exponent: float = 0.9) -> float:
    """nnU-Net's poly decay, per optimiser step (0-based): base * (1 - step/total)^exponent."""
    if not 0 <= step < total:
        raise ValueError(f"step {step} outside [0, {total})")
    return base * (1.0 - step / total) ** exponent


# ---------------------------------------------------------------------------- metrics
def dice_bool(pred: np.ndarray, gt: np.ndarray) -> float:
    import sam_metrics as M          # shared scoring definition (D22); raises if both empty
    return M.dice_coefficient(np.asarray(gt, bool), np.asarray(pred, bool))


def level_key(level) -> str:
    return NO_BOX_LEVEL if level == NO_BOX_LEVEL else f"{float(level):.2f}"


def val_metric(rows: list[dict], select_level) -> dict:
    """rows: {modality, patient, level, dice} per kidney instance. Per (modality, level): mean
    within each patient, then over patients (D22). Selection = mean(CT, MRI) at select_level."""
    by: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[(r["modality"], level_key(r["level"]))][r["patient"]].append(float(r["dice"]))
    out = {}
    for (mod, lev), pats in by.items():
        pm = [sum(v) / len(v) for v in pats.values()]
        key = mod if lev == NO_BOX_LEVEL else f"{mod}@{lev}"
        out[key] = sum(pm) / len(pm)
        out[f"n_patients_{key}"] = len(pm)
        out[f"n_inst_{key}"] = sum(len(v) for v in pats.values())
    sk = level_key(select_level)
    need = ["ct", "mri"] if sk == NO_BOX_LEVEL else [f"ct@{sk}", f"mri@{sk}"]
    missing = [k for k in need if k not in out]
    if missing:
        raise ValueError(f"val_metric: no rows for {missing}")
    out["selection"] = (out[need[0]] + out[need[1]]) / 2.0
    return out
