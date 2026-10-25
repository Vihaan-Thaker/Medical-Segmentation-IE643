#!/usr/bin/env python
"""Training augmentation for the students (ARM3_CODE_PLAN.md §1a, §3.3).

nnU-Net v2's `get_training_transforms`, in its order, values read from its source on
2026-10-24, with two recorded deviations (DECISIONS D42):
  * no mirroring at all (left/right kidney labels; TotalSegmentator);
  * rotation limited to +-0.52 rad (nnU-Net's 3D default) instead of its 2D +-180 deg, which
    would likewise put the right kidney on the image's right.
Details checked against batchgeneratorsv2 (2026-10-24): noise std ~ U(0, 0.1) (its
"noise_variance" is used as the std); blur sigma drawn per axis, kernel truncate 6; low
resolution = nearest-exact down, bilinear up; scale multiplies the sampling coordinates
(scale > 1 = smaller objects).
Intensity operations touch only valid (non-padding) pixels; padding is reset afterwards. All
randomness comes from the `rng` argument (one numpy Generator per training draw), so results
do not depend on worker count, process or global RNG state.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

AUG = {   # name: (probability, then per-channel probability where nnU-Net has one, range)
    "rotation": {"p": 0.2, "range": (-0.52, 0.52)},
    "scaling": {"p": 0.2, "range": (0.7, 1.4)},
    "noise": {"p": 0.1, "std": (0.0, 0.1)},     # nnU-Net's "noise_variance" is used as the std
    "blur": {"p": 0.2, "p_channel": 0.5, "sigma": (0.5, 1.0)},
    "brightness": {"p": 0.15, "range": (0.75, 1.25)},
    "contrast": {"p": 0.15, "range": (0.75, 1.25)},
    "lowres": {"p": 0.25, "p_channel": 0.5, "zoom": (0.5, 1.0)},
    "gamma_inverted": {"p": 0.1, "range": (0.7, 1.5)},
    "gamma": {"p": 0.3, "range": (0.7, 1.5)},
}


def bg_sample(rng: np.random.Generator, lo: float, hi: float) -> float:
    """batchgeneratorsv2 BGContrast: half the draws below 1, half at or above 1."""
    if rng.random() < 0.5 and lo < 1:
        return float(rng.uniform(lo, 1.0))
    return float(rng.uniform(max(lo, 1.0), hi))


def _grid(size: int, center: tuple[float, float], angle: float, scale: float) -> torch.Tensor:
    """Sampling grid (align_corners=False): output pixel p (pixel-edge coordinates, centre at
    index + 0.5) reads input R(angle) S (p - c) + c, as batchgeneratorsv2's SpatialTransform
    (coordinates multiplied by the scale): scale > 1 shrinks the content ("larger numbers =
    smaller objects"), scale < 1 magnifies it."""
    idx = torch.arange(size, dtype=torch.float64) + 0.5
    yy, xx = torch.meshgrid(idx, idx, indexing="ij")
    cy, cx = center
    dx, dy = (xx - cx) * scale, (yy - cy) * scale
    ca, sa = math.cos(angle), math.sin(angle)
    xin = ca * dx + sa * dy + cx
    yin = -sa * dx + ca * dy + cy
    g = torch.stack([2.0 * xin / size - 1.0, 2.0 * yin / size - 1.0], -1)
    return g.to(torch.float32)[None]


def _sample(t: torch.Tensor, grid: torch.Tensor, mode: str) -> torch.Tensor:
    return F.grid_sample(t[None], grid, mode=mode, padding_mode="zeros", align_corners=False)[0]


def _kernel(sigma: float, truncate: float = 6.0) -> torch.Tensor:
    """batchgeneratorsv2 _build_kernel: size = nearest odd integer to sigma * truncate + 0.5,
    taps on linspace(-(k-1)/2, (k-1)/2, k), normalised Gaussian."""
    v = sigma * truncate + 0.5
    k = max(1, 2 * int(round((v - 1) / 2)) + 1)
    x = torch.linspace(-(k - 1) / 2, (k - 1) / 2, k)
    pdf = torch.exp(-0.5 * (x / sigma) ** 2)
    return pdf / pdf.sum()


def _gauss_blur(img: torch.Tensor, sigma_y: float, sigma_x: float) -> torch.Tensor:
    """Separable blur, one sigma per axis (nnU-Net: synchronize_axes=False), reflect padding."""
    kx, ky = _kernel(sigma_x), _kernel(sigma_y)
    rx, ry = len(kx) // 2, len(ky) // 2
    x = F.conv2d(F.pad(img[None], (rx, rx, 0, 0), mode="reflect"), kx.view(1, 1, 1, -1))
    return F.conv2d(F.pad(x, (0, 0, ry, ry), mode="reflect"), ky.view(1, 1, -1, 1))[0]


def _gamma(v: torch.Tensor, gamma: float, invert: bool) -> torch.Tensor:
    """nnU-Net GammaTransform with retain_stats on the valid pixels `v` (1-D)."""
    if v.numel() < 2:
        return v
    mean0, std0 = v.mean(), v.std()
    x = -v if invert else v
    mn, mx = x.min(), x.max()
    rng_ = mx - mn
    if float(rng_) < 1e-7:
        return v
    x = ((x - mn) / rng_).clamp(0, 1) ** gamma * rng_ + mn
    if invert:
        x = -x
    return (x - x.mean()) / (x.std() + 1e-8) * std0 + mean0


def augment(img: torch.Tensor, label: torch.Tensor, valid: torch.Tensor, box_ch: torch.Tensor,
            g: dict, pad_value: float, rng: np.random.Generator):
    """img 1 x S x S float32, label S x S int64, valid S x S bool, box_ch K x S x S (K = 0 or 2).
    Returns (img, label, valid, box_ch, applied) -- new tensors; inputs are not modified."""
    S = img.shape[-1]
    img, label, valid, box_ch = img.clone(), label.clone(), valid.clone(), box_ch.clone()
    applied = {}

    # 1. rotation + scaling, one affine (decision values always drawn: stable stream)
    u_rot, a = rng.random(), rng.uniform(*AUG["rotation"]["range"])
    u_sc, s = rng.random(), rng.uniform(*AUG["scaling"]["range"])
    angle = float(a) if u_rot < AUG["rotation"]["p"] else 0.0
    scale = float(s) if u_sc < AUG["scaling"]["p"] else 1.0
    applied["rotation"], applied["scaling"] = angle != 0.0, scale != 1.0
    applied["angle_value"], applied["scale_value"] = angle, scale
    if applied["rotation"] or applied["scaling"]:
        grid = _grid(S, (g["h2"] / 2.0, g["w2"] / 2.0), angle, scale)
        img = _sample(img - pad_value, grid, "bilinear") + pad_value
        valid = _sample(valid[None].float(), grid, "nearest")[0] > 0.5
        label = _sample(label[None].float(), grid, "nearest")[0].round().to(torch.int64)
        if box_ch.shape[0]:
            box_ch = _sample(box_ch, grid, "nearest")
        label[~valid] = 0
        box_ch[:, ~valid] = 0.0

    x = img[0]
    m = valid
    if not bool(m.any()):                       # everything rotated out of the field
        img[0][~m] = pad_value
        return img, label, valid, box_ch, applied

    # 2. Gaussian noise
    applied["noise"] = rng.random() < AUG["noise"]["p"]
    if applied["noise"]:
        std = float(rng.uniform(*AUG["noise"]["std"]))
        applied["noise_std"] = std
        noise = torch.from_numpy(rng.standard_normal((S, S)).astype(np.float32)) * std
        x[m] = x[m] + noise[m]
    # 3. Gaussian blur (probability x per-channel probability)
    applied["blur"] = rng.random() < AUG["blur"]["p"] and rng.random() < AUG["blur"]["p_channel"]
    if applied["blur"]:
        sy, sx = (float(rng.uniform(*AUG["blur"]["sigma"])) for _ in range(2))
        x = _gauss_blur(x[None], sy, sx)[0]
    # 4. brightness
    applied["brightness"] = rng.random() < AUG["brightness"]["p"]
    if applied["brightness"]:
        x[m] = x[m] * bg_sample(rng, *AUG["brightness"]["range"])
    # 5. contrast (about the valid mean, range preserved)
    applied["contrast"] = rng.random() < AUG["contrast"]["p"]
    if applied["contrast"]:
        f = bg_sample(rng, *AUG["contrast"]["range"])
        v = x[m]
        mu, lo, hi = v.mean(), v.min(), v.max()
        x[m] = ((v - mu) * f + mu).clamp(lo, hi)
    # 6. simulate low resolution
    applied["lowres"] = rng.random() < AUG["lowres"]["p"] and rng.random() < AUG["lowres"]["p_channel"]
    if applied["lowres"]:
        z = float(rng.uniform(*AUG["lowres"]["zoom"]))
        n = max(1, int(round(S * z)))
        small = F.interpolate(x[None, None], (n, n), mode="nearest-exact")
        x = F.interpolate(small, (S, S), mode="bilinear", align_corners=False)[0, 0]
    # 7. inverted gamma, 8. gamma (retain stats)
    for name, inv in (("gamma_inverted", True), ("gamma", False)):
        applied[name] = rng.random() < AUG[name]["p"]
        if applied[name]:
            x[m] = _gamma(x[m], bg_sample(rng, *AUG[name]["range"]), inv)

    x = x.clone()
    x[~m] = pad_value
    img = x[None].contiguous()
    if not torch.isfinite(img).all():
        raise FloatingPointError("augmentation produced a non-finite value")
    return img, label, valid, box_ch, applied
