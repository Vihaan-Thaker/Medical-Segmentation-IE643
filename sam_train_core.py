#!/usr/bin/env python
"""Arm-2 core: a batched SAM forward pass that is step-for-step the evaluation path, the
training losses, the LR schedule and the validation selection score (ARM2_CODE_PLAN.md §1a, §3.3).

The forward path reproduces SamPredictor.set_image + predict(box, multimask_output=False):

  uint8 HxW  -> np.repeat to HxWx3 -> ResizeLongestSide(img_size).apply_image (PIL, uint8)
  -> torch.as_tensor(uint8) CHW   -> sam.preprocess PER IMAGE (normalise, then pad), fp32
  -> image_encoder                -> per box: apply_boxes in float64 numpy, then float32
  -> prompt_encoder(points=None, boxes, masks=None) -> mask_decoder(one image, one box,
     multimask_output=False) -> postprocess_masks (fp32) -> logits; mask = logits > 0

Each box is decoded on its own, exactly as eval_sam.py does, so batching never changes a
result. Only the encoder runs batched (one image per row; SAM has no cross-sample ops).
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from segment_anything.utils.transforms import ResizeLongestSide

EPS = 1e-6


# --------------------------------------------------------------------- images and boxes
def prepare_image(img_u8: np.ndarray, target: int = 1024) -> tuple[torch.Tensor, tuple, tuple]:
    """uint8 HxW slice -> (uint8 CxHrxWr tensor, input_size, original_size). No normalisation,
    no padding: those happen in to_model_input, as in SamPredictor."""
    if img_u8.dtype != np.uint8 or img_u8.ndim != 2:
        raise ValueError(f"expected a uint8 HxW slice, got {img_u8.dtype} {img_u8.shape}")
    rgb = np.repeat(img_u8[..., None], 3, axis=2)                   # as eval_sam.process_case
    resized = ResizeLongestSide(target).apply_image(rgb)             # as SamPredictor.set_image
    t = torch.as_tensor(resized).permute(2, 0, 1).contiguous()       # uint8, as set_image
    return t, tuple(resized.shape[:2]), tuple(img_u8.shape[:2])


def to_model_input(sam, images: list[torch.Tensor], device) -> torch.Tensor:
    """sam.preprocess on each image separately (normalise, THEN pad), then stack. Always fp32,
    outside autocast. Byte-identical to what SamPredictor.set_torch_image feeds the encoder."""
    size = sam.image_encoder.img_size
    out = []
    with torch.autocast(device_type=torch.device(device).type, enabled=False):
        for t in images:
            if t.dtype != torch.uint8 or t.ndim != 3 or max(t.shape[-2:]) != size:
                raise ValueError(f"image tensor must be uint8 CxHxW with long side {size}, "
                                 f"got {t.dtype} {tuple(t.shape)}")
            out.append(sam.preprocess(t.to(device)[None]))
    return torch.cat(out, dim=0)


def transform_box(box, original_size: tuple, target: int, device) -> torch.Tensor:
    """XYXY box in slice pixels -> SAM input coordinates, exactly as SamPredictor.predict:
    numpy float64 apply_boxes, then float32, shape (1, 1, 4)."""
    arr = np.array(box, dtype=float)                                 # as eval_sam: dtype=float
    tb = ResizeLongestSide(target).apply_boxes(arr, original_size)   # float64 numpy
    t = torch.as_tensor(tb, dtype=torch.float, device=device)        # as predict()
    return t[None, :]


def decode_one(sam, feature: torch.Tensor, box_t: torch.Tensor, input_size: tuple,
               original_size: tuple) -> tuple[torch.Tensor, torch.Tensor]:
    """One box on one image's features [1, C, h, w] -> (logits HxW fp32, iou_pred scalar)."""
    sparse, dense = sam.prompt_encoder(points=None, boxes=box_t, masks=None)
    low_res, iou_pred = sam.mask_decoder(
        image_embeddings=feature, image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
        multimask_output=False)
    logits = sam.postprocess_masks(low_res.float(), input_size, original_size)
    return logits[0, 0], iou_pred.reshape(-1)[0]


def forward_batch(sam, images: list[torch.Tensor], boxes_per_image: list[list],
                  input_sizes: list[tuple], original_sizes: list[tuple], device):
    """Encode each slice once (batched), decode every box separately. Returns, per image, a
    list of (logits, iou_pred) in the order of its boxes."""
    if not (len(images) == len(boxes_per_image) == len(input_sizes) == len(original_sizes)):
        raise ValueError("forward_batch: per-image lists differ in length")
    target = sam.image_encoder.img_size
    feats = sam.image_encoder(to_model_input(sam, images, device))
    out = []
    for i, boxes in enumerate(boxes_per_image):
        res = []
        for box in boxes:
            bt = transform_box(box, original_sizes[i], target, device)
            res.append(decode_one(sam, feats[i:i + 1], bt, input_sizes[i], original_sizes[i]))
        out.append(res)
    return out


# ----------------------------------------------------------------------------- losses
def seg_losses(logits: torch.Tensor, gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Soft Dice (squared denominators, as MedSAM's MONAI DiceLoss(squared_pred=True)) and
    BCE-with-logits averaged over pixels, for ONE kidney. fp32."""
    logits = logits.float()
    g = gt.to(device=logits.device, dtype=torch.float32)
    p = torch.sigmoid(logits)
    dice = 1.0 - (2.0 * (p * g).sum() + EPS) / ((p * p).sum() + (g * g).sum() + EPS)
    bce = F.binary_cross_entropy_with_logits(logits, g, reduction="mean")
    return dice, bce


@torch.no_grad()
def iou_target(logits: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """IoU of the thresholded prediction (logit > 0, the test threshold) with the GT."""
    pred = logits > 0
    g = gt.to(device=logits.device, dtype=torch.bool)
    union = (pred | g).sum()
    if int(union) == 0:
        raise ValueError("iou_target: empty prediction and empty GT")
    return (pred & g).sum().float() / union.float()


def kidney_loss(logits, iou_pred, gt) -> dict:
    dice, bce = seg_losses(logits, gt)
    iou_mse = (iou_pred.float() - iou_target(logits, gt)) ** 2
    return {"dice": dice, "bce": bce, "iou": iou_mse, "total": dice + bce + iou_mse}


# --------------------------------------------------------------------------- schedule
def lr_factor(step: int, base: float, warmup: int, total: int, floor: float) -> float:
    """Multiplier on `base` at optimiser step `step` (0-based): linear warm-up to 1 over
    `warmup` steps, then cosine from base down to `floor` at step == total."""
    if total <= warmup:
        raise ValueError(f"total steps {total} must exceed warm-up {warmup}")
    if step < warmup:
        return (step + 1) / warmup
    prog = min(1.0, (step - warmup) / (total - warmup))
    lr = floor + (base - floor) * 0.5 * (1.0 + math.cos(math.pi * prog))
    return lr / base


# ------------------------------------------------------------------------- selection
def dice_bool(pred: np.ndarray, gt: np.ndarray) -> float:
    """Hard Dice through the shared scoring code (sam_metrics), on bool masks."""
    import sam_metrics as M
    return M.dice_coefficient(np.asarray(gt, bool), np.asarray(pred, bool))


def val_metric(rows: list[dict], levels=(0.0, 0.10), select_level: float = 0.10,
               modalities: tuple = ("ct", "mri")) -> dict:
    """rows: {modality, patient, level, dice}. Per (modality, level): mean within each
    patient, then over patients. Selection score = mean(CT, MRI) at select_level; for a
    single-modality teacher (D71, modalities=("ct",) or ("mri",)) that modality's score. Rows
    of any modality outside `modalities` are refused."""
    present = {r["modality"] for r in rows}
    extra = sorted(present - set(modalities))
    if extra:
        raise ValueError(f"val_metric: rows of {extra}, expected only {list(modalities)}")
    by: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[(r["modality"], round(float(r["level"]), 4))][r["patient"]].append(float(r["dice"]))
    out = {}
    for (mod, lev), pats in by.items():
        pm = [sum(v) / len(v) for v in pats.values()]
        out[f"{mod}@{lev:.2f}"] = sum(pm) / len(pm)
        out[f"n_patients_{mod}@{lev:.2f}"] = len(pm)
        out[f"n_inst_{mod}@{lev:.2f}"] = sum(len(v) for v in pats.values())
    need = [f"{m}@{round(select_level, 4):.2f}" for m in modalities]
    missing = [k for k in need if k not in out]
    if missing:
        raise ValueError(f"val_metric: no rows for {missing}")
    if len(need) == 2:
        out["selection"] = (out[need[0]] + out[need[1]]) / 2.0
    elif len(need) == 1:
        out["selection"] = out[need[0]]
    else:
        raise ValueError(f"val_metric: modalities must be one or two, got {list(modalities)}")
    return out
