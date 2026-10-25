#!/usr/bin/env python
"""LoRA on SAM's image-encoder attention, for arm 2 (DECISIONS.md D33, ARM2_CODE_PLAN.md §3.2).

Each encoder block's attention has one fused projection `qkv = nn.Linear(d, 3d)`. SAM's
Attention.forward reshapes its output as (B, HW, 3, heads, head_dim), so the output columns
are laid out q = [0:d], k = [d:2d], v = [2d:3d]. LoRA adds a trainable low-rank correction to
the q and v parts only; k is never touched (LoRA paper, Table 5):

    out[..., 0:d]   += (alpha / r) * B_q(A_q(x))
    out[..., 2d:3d] += (alpha / r) * B_v(A_v(x))

B starts at zero, so an injected model is exactly the original until training moves B.
`merge_lora` folds the corrections back into the qkv weight, giving a plain SAM state dict
that `sam_model_registry["vit_b"](checkpoint=...)` loads unchanged.
"""
from __future__ import annotations

import hashlib
import math

import torch
from torch import nn


class LoRAQKV(nn.Module):
    """A frozen fused qkv Linear plus LoRA on its q and v output slices."""

    def __init__(self, qkv: nn.Linear, r: int, alpha: float):
        super().__init__()
        if not isinstance(qkv, nn.Linear):
            raise TypeError(f"expected nn.Linear for qkv, got {type(qkv).__name__}")
        d = qkv.in_features
        if qkv.out_features != 3 * d:
            raise ValueError(f"qkv must map d -> 3d, got {qkv.in_features} -> {qkv.out_features}")
        if r < 1:
            raise ValueError(f"LoRA rank must be >= 1, got {r}")
        self.qkv = qkv
        self.d, self.r, self.alpha = d, int(r), float(alpha)
        self.scale = self.alpha / self.r
        self.A_q = nn.Linear(d, r, bias=False)
        self.B_q = nn.Linear(r, d, bias=False)
        self.A_v = nn.Linear(d, r, bias=False)
        self.B_v = nn.Linear(r, d, bias=False)
        for a in (self.A_q, self.A_v):
            nn.init.kaiming_uniform_(a.weight, a=math.sqrt(5))
        for b in (self.B_q, self.B_v):
            nn.init.zeros_(b.weight)
        # Match the wrapped layer's device and dtype.
        self.to(device=qkv.weight.device, dtype=qkv.weight.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.qkv(x)
        dq = self.scale * self.B_q(self.A_q(x))
        dv = self.scale * self.B_v(self.A_v(x))
        zeros_k = torch.zeros_like(dq)
        # Non-in-place: build the full-width delta and add it once (autograd-safe).
        return out + torch.cat([dq, zeros_k, dv], dim=-1)


def lora_modules(sam) -> list[LoRAQKV]:
    return [blk.attn.qkv for blk in sam.image_encoder.blocks if isinstance(blk.attn.qkv, LoRAQKV)]


def inject_lora(sam, r: int, alpha: float, expect_blocks: int | None = None,
                expect_dim: int | None = None) -> list[LoRAQKV]:
    """Wrap blk.attn.qkv of EVERY image-encoder block. Production passes expect_blocks=12,
    expect_dim=768 (ViT-B); tests pass the tiny model's values."""
    blocks = list(sam.image_encoder.blocks)
    if expect_blocks is not None and len(blocks) != expect_blocks:
        raise ValueError(f"expected {expect_blocks} encoder blocks, found {len(blocks)}")
    if lora_modules(sam):
        raise RuntimeError("LoRA is already injected into this model")
    mods = []
    for i, blk in enumerate(blocks):
        qkv = blk.attn.qkv
        if expect_dim is not None and (qkv.in_features, qkv.out_features) != (expect_dim, 3 * expect_dim):
            raise ValueError(f"block {i}: qkv is {qkv.in_features}->{qkv.out_features}, expected "
                             f"{expect_dim}->{3 * expect_dim}")
        blk.attn.qkv = LoRAQKV(qkv, r, alpha)
        mods.append(blk.attn.qkv)
    return mods


def lora_param_count(d: int, r: int, n_blocks: int) -> int:
    """A_q, B_q, A_v, B_v per block: 4 * d * r."""
    return 4 * d * r * n_blocks


def freeze_for_arm2(sam) -> dict:
    """Train ONLY the LoRA A/B matrices and the mask decoder. Asserts the exact counts."""
    mods = lora_modules(sam)
    if not mods:
        raise RuntimeError("freeze_for_arm2 called before inject_lora")
    for p in sam.parameters():
        p.requires_grad_(False)
    for m in mods:
        for lin in (m.A_q, m.B_q, m.A_v, m.B_v):
            lin.weight.requires_grad_(True)
    for p in sam.mask_decoder.parameters():
        p.requires_grad_(True)

    d, r = mods[0].d, mods[0].r
    n_lora = sum(lin.weight.numel() for m in mods for lin in (m.A_q, m.B_q, m.A_v, m.B_v))
    n_dec = sum(p.numel() for p in sam.mask_decoder.parameters())
    n_train = sum(p.numel() for p in sam.parameters() if p.requires_grad)
    want_lora = lora_param_count(d, r, len(mods))
    if n_lora != want_lora or n_train != want_lora + n_dec:
        raise AssertionError(f"trainable count wrong: LoRA {n_lora} (want {want_lora}), total "
                             f"{n_train} (want {want_lora + n_dec})")
    # Nothing else may be trainable: prompt encoder, original encoder weights, qkv itself.
    lora_ids = {id(lin.weight) for m in mods for lin in (m.A_q, m.B_q, m.A_v, m.B_v)}
    dec_ids = {id(p) for p in sam.mask_decoder.parameters()}
    stray = [n for n, p in sam.named_parameters()
             if p.requires_grad and id(p) not in lora_ids | dec_ids]
    if stray:
        raise AssertionError(f"unexpected trainable parameters: {stray[:5]}")
    return {"lora": n_lora, "decoder": n_dec, "trainable": n_train,
            "total": sum(p.numel() for p in sam.parameters()), "rank": r, "dim": d,
            "blocks": len(mods)}


def frozen_state_digest(sam) -> str:
    """sha256 over every non-trainable parameter (name, dtype, shape, bytes), in name order.
    Equal before and after training <=> the frozen weights were never touched."""
    h = hashlib.sha256()
    for name, p in sorted(sam.named_parameters(), key=lambda kv: kv[0]):
        if p.requires_grad:
            continue
        t = p.detach().to("cpu").contiguous()
        h.update(name.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes() if t.dtype != torch.bfloat16 else t.float().numpy().tobytes())
    return h.hexdigest()


def adapter_state(sam) -> dict:
    """Only the trainable part: LoRA matrices + mask decoder (CPU tensors)."""
    sd = sam.state_dict()
    keep = {k for k in sd if (".attn.qkv.A_" in k or ".attn.qkv.B_" in k) or k.startswith("mask_decoder.")}
    if not any(".attn.qkv.A_" in k for k in keep):
        raise RuntimeError("adapter_state: no LoRA tensors found (inject first)")
    return {k: sd[k].detach().to("cpu").clone() for k in sorted(keep)}


def load_adapter_state(sam, state: dict) -> None:
    """Load exactly the adapter keys, strict both ways."""
    sd = sam.state_dict()
    want = {k for k in sd if (".attn.qkv.A_" in k or ".attn.qkv.B_" in k) or k.startswith("mask_decoder.")}
    got = set(state)
    if want != got:
        raise KeyError(f"adapter keys differ: missing {sorted(want - got)[:5]}, unexpected "
                       f"{sorted(got - want)[:5]}")
    for k in want:
        if tuple(sd[k].shape) != tuple(state[k].shape):
            raise ValueError(f"adapter shape mismatch for {k}: {tuple(state[k].shape)} vs {tuple(sd[k].shape)}")
    with torch.no_grad():
        for k in want:
            sd[k].copy_(state[k].to(device=sd[k].device, dtype=sd[k].dtype))


@torch.no_grad()
def merge_lora(sam) -> None:
    """Fold each LoRA correction into its qkv weight (fp32), then put the plain Linear back.
    Afterwards the model is an ordinary SAM with no LoRA modules."""
    mods = lora_modules(sam)
    if not mods:
        raise RuntimeError("merge_lora: no LoRA modules to merge")
    for blk in sam.image_encoder.blocks:
        m = blk.attn.qkv
        if not isinstance(m, LoRAQKV):
            raise RuntimeError("merge_lora: a block without LoRA (partial injection)")
        d, s = m.d, m.scale
        w = m.qkv.weight
        dq = (m.B_q.weight.float() @ m.A_q.weight.float()) * s
        dv = (m.B_v.weight.float() @ m.A_v.weight.float()) * s
        w_new = w.float().clone()
        w_new[0:d] += dq
        w_new[2 * d:3 * d] += dv
        w.copy_(w_new.to(w.dtype))
        blk.attn.qkv = m.qkv


def plain_state_keys(sam) -> list[str]:
    return sorted(sam.state_dict().keys())
