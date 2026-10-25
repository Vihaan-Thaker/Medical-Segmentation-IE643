#!/usr/bin/env python
"""Arm-3/4 student: a prompt-free 2D U-Net built from nnU-Net's blocks (ARM3_CODE_PLAN.md §3.1).

    conv 3x3 -> InstanceNorm(affine) -> LeakyReLU(0.01), twice per level
    encoder: level 1 at stride 1, levels 2..L start with a stride-2 conv (halves H x W)
    decoder: ConvTranspose 2x2 stride 2, cat([skip, up]), then a block
    head:    1x1 conv -> 3 classes (background / right kidney / left kidney)
    deep supervision: 1x1 heads on every decoder output except the two lowest-resolution ones

Production: channels (16, 32, 64, 128, 256, 256, 256) -> 8,372,979 parameters for one input
channel (+681 training-only deep-supervision heads). No dependency on SAM.
"""
from __future__ import annotations

import torch
import torch.nn as nn

CHANNELS = (16, 32, 64, 128, 256, 256, 256)
N_CLASSES = 3
# Exact counts of the production network (built and counted, D42); asserted at start-up.
EXPECTED_PARAMS = {1: 8_373_660, 3: 8_373_948}          # in_ch -> total incl. DS heads
EXPECTED_PARAMS_NO_DS = {1: 8_372_979, 3: 8_373_267}


class Block(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride, 1, bias=True),
            nn.InstanceNorm2d(cout, affine=True, track_running_stats=False),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(cout, cout, 3, 1, 1, bias=True),
            nn.InstanceNorm2d(cout, affine=True, track_running_stats=False),
            nn.LeakyReLU(0.01, inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StudentUNet(nn.Module):
    def __init__(self, in_ch: int, n_classes: int = N_CLASSES, channels=CHANNELS,
                 deep_supervision: bool = True):
        super().__init__()
        channels = tuple(int(c) for c in channels)
        if len(channels) < 3:
            raise ValueError(f"need at least 3 levels, got {channels}")
        self.in_ch, self.n_classes, self.channels = int(in_ch), int(n_classes), channels
        self.deep_supervision = bool(deep_supervision)
        L = len(channels)
        self.enc = nn.ModuleList()
        c = self.in_ch
        for lvl, ch in enumerate(channels):
            self.enc.append(Block(c, ch, 1 if lvl == 0 else 2))
            c = ch
        # decoder stages run from level L-1 (just above the bottleneck) up to level 1
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for lvl in range(L - 1, 0, -1):
            hi, lo = channels[lvl], channels[lvl - 1]
            self.up.append(nn.ConvTranspose2d(hi, lo, 2, 2))
            self.dec.append(Block(2 * lo, lo, 1))
        self.head = nn.Conv2d(channels[0], self.n_classes, 1)
        # decoder outputs, highest resolution first: levels 1..L-1 (0-based 0..L-2). nnU-Net
        # supervises all but the two lowest-resolution ones; level 1 is the main head.
        self.ds_levels = list(range(1, (L - 1) - 2))     # 0-based levels; L=7 -> [1, 2, 3]
        self.ds_heads = nn.ModuleList([nn.Conv2d(channels[lv], self.n_classes, 1)
                                       for lv in self.ds_levels])
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, a=0.01)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.InstanceNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @property
    def divisor(self) -> int:
        return 2 ** (len(self.channels) - 1)

    def forward(self, x: torch.Tensor):
        h, w = x.shape[-2:]
        d = self.divisor
        if h % d or w % d:
            raise ValueError(f"input {h}x{w} not divisible by {d}")
        if x.shape[1] != self.in_ch:
            raise ValueError(f"expected {self.in_ch} input channels, got {x.shape[1]}")
        skips = []
        for blk in self.enc:
            x = blk(x)
            skips.append(x)
        x = skips.pop()                                   # bottleneck
        outs = {}                                         # 0-based level -> decoder output
        for k, (up, dec) in enumerate(zip(self.up, self.dec)):
            lvl = len(self.channels) - 2 - k
            x = dec(torch.cat([skips.pop(), up(x)], 1))
            outs[lvl] = x
        main = self.head(outs[0])
        if self.training and self.deep_supervision:
            return [main] + [hd(outs[lv]) for hd, lv in zip(self.ds_heads, self.ds_levels)]
        return main


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_params_no_ds(model: StudentUNet) -> int:
    return count_params(model) - sum(p.numel() for p in model.ds_heads.parameters())
