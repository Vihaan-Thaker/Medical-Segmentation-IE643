#!/usr/bin/env python
"""Arm 4: distillation from the arm-2 r4 teacher's stored maps (ARM4_CODE_PLAN.md §3, §5;
DECISIONS D59 / D60).

* TeacherStore: every teacher map of the train split in RAM, after verifying the folder's
  identity (teacher.json, run.json, teacher_manifest.csv) and EVERY file against the student's
  own SliceStore (meta, boxes, and the stored teacher Dice, recomputed).
* Targets in the network frame: the teacher logits letterboxed with the image's own resampling
  call; the distillation region = the clean box + `margin` x its side on each side.
* follow_augmentation: the maps take the geometry the image transform already drew (from
  augment's `applied`), so the teacher path consumes no randomness and the arm-3 item at the
  same position is unchanged.
* kd_loss (Option B, D59): per kidney, a per-pixel binary KL(teacher || student) between the
  teacher's sigmoid map and the student's softmax probability of that kidney, at temperature T,
  x T^2, averaged over the region; summed over kidneys and divided by the FULL batch size.
Channel 0 = right kidney (mask value 1), channel 1 = left (2), as C.box_channels.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sam_metrics as M  # noqa: E402
import student_aug as A  # noqa: E402
import student_core as C  # noqa: E402
import student_data as D  # noqa: E402
import teacher_io as TIO  # noqa: E402
import teacher_maps as TM  # noqa: E402

KD_FILL = -20.0                     # teacher logit outside the slice (p ~ 2e-9); always masked out
TRANSPORT_DTYPE = torch.float16     # worker -> main process (the stored maps are float16 anyway)
TEACHER_IDENTITY = {"split": "train", "boxes": "clean (manifest), no jitter", "model_type": "vit_b"}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- teacher store
class TeacherStore:
    """All teacher maps of a train SliceStore in RAM (one flat float16 buffer), verified."""

    def __init__(self, teacher_dir, store, gate: float, threads: int = 8, log=print):
        t0 = time.time()
        if store.split != "train":
            raise SystemExit(f"teacher maps exist for the train split only, not {store.split!r}")
        if not 0.0 <= float(gate) <= 1.0:
            raise SystemExit(f"gate {gate} outside [0, 1]")
        self.dir, self.store, self.gate = Path(teacher_dir), store, float(gate)
        stamp = TIO.read_stamp(self.dir)
        if stamp is None:
            raise SystemExit(f"REFUSING -- {self.dir} has no {TIO.STAMP} (not a teacher folder)")
        msha = D.manifest_sha256(store.processed)
        want = {"teacher_sha256": TM.TEACHER_SHA256, "manifest_sha256": msha,
                "format": TIO.FORMAT_VERSION, **TEACHER_IDENTITY}
        diff = {k: (stamp.get(k), v) for k, v in want.items() if stamp.get(k) != v}
        if diff:
            raise SystemExit(f"REFUSING -- {self.dir} is not the arm-2 r4 teacher on this data: {diff}")
        rj_path, man_path = self.dir / "run.json", self.dir / "teacher_manifest.csv"
        if not rj_path.exists() or not man_path.exists():
            raise SystemExit(f"REFUSING -- {self.dir}: run.json / teacher_manifest.csv missing (teacher job unfinished)")
        rj = json.loads(rj_path.read_text())
        todo = [i for i, r in enumerate(store.rows) if r["sam_eligible"]]
        n_kid = sum(sum(b is not None for b in store.boxes(i).values()) for i in todo)
        man_sha = _sha256(man_path)
        problems = []
        if rj.get("status") != "TEACHER_OK" or rj.get("problems"):
            problems.append(f"run.json status {rj.get('status')!r}, problems {rj.get('problems')}")
        if rj.get("slices") != len(todo) or rj.get("kidneys") != n_kid:
            problems.append(f"run.json counts {rj.get('slices')} slices / {rj.get('kidneys')} kidneys != "
                            f"the store's {len(todo)} boxed slices / {n_kid} kidneys")
        if rj.get("teacher_sha256") != want["teacher_sha256"] or rj.get("manifest_sha256") != msha:
            problems.append("run.json teacher / manifest sha256 differ from teacher.json")
        if rj.get("teacher_manifest_sha256") != man_sha:
            problems.append("teacher_manifest.csv changed since the teacher job (sha256 differs)")
        if problems:
            raise SystemExit(f"REFUSING -- {self.dir}: " + "; ".join(problems))
        ident = {"teacher_sha256": want["teacher_sha256"], "manifest_sha256": msha}
        # layout: one float16 block per (slice, boxed side), in (store index, side) order
        self.index: dict = {}
        sizes, keys = [], []
        for i in todo:
            r = store.rows[i]
            for side, box in store.boxes(i).items():
                if box is not None:
                    keys.append((i, side))
                    sizes.append(r["h"] * r["w"])
        offs = np.concatenate([[0], np.cumsum(np.array(sizes, dtype=np.int64))]).astype(np.int64)
        self.buf = np.empty(int(offs[-1]), dtype=np.float16)
        self._off = {k: int(o) for k, o in zip(keys, offs[:-1])}
        bad: list = []

        def load(i: int) -> None:
            r = store.rows[i]
            tag = f"{r['case']} z={r['z']}"
            try:
                meta, logits = TIO.load_slice(self.dir, r["case"], r["z"])
            except Exception as e:                       # noqa: BLE001 -- reported, never skipped
                bad.append(f"{tag}: {type(e).__name__}: {e}")
                return
            boxes = store.boxes(i)
            if not TIO.meta_matches(meta, TIO.expected_meta(r, boxes, ident)):
                bad.append(f"{tag}: meta does not match the store row (case, z, size, sides, boxes or identity)")
                return
            if meta.get("patient") != r["patient"] or meta.get("modality") != r["modality"]:
                bad.append(f"{tag}: meta patient/modality {meta.get('patient')}/{meta.get('modality')}")
                return
            _, msk = store.get(i)
            out = {}
            for side, lg in logits.items():
                gt = msk == side
                d = M.dice_coefficient(gt, lg.astype(np.float32) > 0)
                stored, flips = float(meta["dice"][str(side)]), int(meta["fp16_flips"][str(side)])
                tol = 1e-9 if flips == 0 else flips / max(int(gt.sum()), 1)
                if abs(d - stored) > tol:
                    bad.append(f"{tag} side {side}: stored teacher Dice {stored} != recomputed {d}")
                    return
                a = self._off[(i, side)]
                self.buf[a:a + lg.size] = lg.ravel()
                out[side] = (stored, stored >= self.gate)
            self.index[i] = out

        with ThreadPoolExecutor(max_workers=max(1, int(threads))) as ex:
            for fut in [ex.submit(load, i) for i in todo]:
                fut.result()
        if bad:
            raise SystemExit(f"REFUSING -- {len(bad)} teacher files failed verification, e.g. " + " | ".join(bad[:10]))
        if len(self.index) != len(todo) or sum(len(v) for v in self.index.values()) != n_kid:
            raise SystemExit(f"teacher store holds {len(self.index)} slices / "
                             f"{sum(len(v) for v in self.index.values())} kidneys, expected {len(todo)} / {n_kid}")
        self.n_kidneys = Counter()
        self.n_gated = Counter()
        for i, sides in self.index.items():
            mod = store.rows[i]["modality"]
            for _, used in sides.values():
                self.n_kidneys[mod] += 1
                self.n_gated[mod] += int(not used)
        self.identity = {"teacher_sha256": ident["teacher_sha256"], "manifest_sha256": msha,
                         "teacher_manifest_sha256": man_sha, "slices": len(todo), "kidneys": n_kid,
                         "kidneys_by_modality": dict(sorted(self.n_kidneys.items())),
                         "gated_by_modality": {m: self.n_gated[m] for m in sorted(self.n_kidneys)}}
        self.load_s = time.time() - t0
        self.gib = self.buf.nbytes / 2 ** 30
        log(f"teacher: {len(todo)} slices, {n_kid} kidneys {dict(self.n_kidneys)}; gated (Dice < {self.gate}) "
            f"{dict(self.n_gated)}; verified and loaded in {self.load_s:.0f}s ({self.gib:.2f} GiB)")

    def maps(self, i: int) -> dict:
        """{side: (float16 h x w logits view, teacher Dice, used)}; {} for a slice without a box."""
        r = self.store.rows[i]
        out = {}
        for side, (dice, used) in self.index.get(i, {}).items():
            a = self._off[(i, side)]
            out[side] = (self.buf[a:a + r["h"] * r["w"]].reshape(r["h"], r["w"]), dice, used)
        return out


# ------------------------------------------------------------------------------ geometry
def region_box(box, h: int, w: int, margin: float) -> tuple[int, int, int, int]:
    """Inclusive native XYXY box enlarged by ceil(margin x side) on each side, clipped to the
    slice. Exact (Fraction) arithmetic: in floats e.g. 0.28 x 25 = 7.000000000000001, whose
    ceil is 8, not 7 (0.2 happens to be safe for every side length, but nothing relies on it)."""
    x0, y0, x1, y1 = (int(v) for v in box)
    if not (0 <= x0 <= x1 < w and 0 <= y0 <= y1 < h):
        raise ValueError(f"box {box} outside the {h}x{w} slice")
    fr = Fraction(str(float(margin)))
    if fr < 0:
        raise ValueError(f"negative margin {margin}")
    mx, my = math.ceil(fr * (x1 - x0 + 1)), math.ceil(fr * (y1 - y0 + 1))
    return max(0, x0 - mx), max(0, y0 - my), min(w - 1, x1 + mx), min(h - 1, y1 + my)


def to_net_soft(arr, g: dict, fill: float) -> torch.Tensor:
    """Native float map h x w -> float32 size x size, with C.to_net_image's exact resampling
    call (bilinear, align_corners False, antialias True), top-left, `fill` elsewhere."""
    x = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    if x.shape != (g["h"], g["w"]):
        raise ValueError(f"map {tuple(x.shape)} != slice {(g['h'], g['w'])}")
    if (g["h2"], g["w2"]) != (g["h"], g["w"]):
        x = F.interpolate(x[None, None], (g["h2"], g["w2"]), mode="bilinear",
                          align_corners=False, antialias=True)[0, 0]
    out = torch.full((g["size"], g["size"]), float(fill), dtype=torch.float32)
    out[:g["h2"], :g["w2"]] = x
    return out


def teacher_targets(tstore: TeacherStore, i: int, g: dict, margin: float):
    """Unaugmented network-frame targets for store slice i: (t_logit 2 x S x S float32,
    t_region 2 x S x S bool, t_has 2 bool). Only sides with a map that passes the gate."""
    S = g["size"]
    t_logit = torch.full((2, S, S), KD_FILL, dtype=torch.float32)
    t_region = torch.zeros((2, S, S), dtype=torch.bool)
    t_has = torch.zeros(2, dtype=torch.bool)
    boxes = tstore.store.boxes(i)
    for side, (lg, _, used) in tstore.maps(i).items():
        if not used:
            continue
        c = side - 1
        t_logit[c] = to_net_soft(lg, g, KD_FILL)
        x0, y0, x1, y1 = C.box_to_frame(region_box(boxes[side], g["h"], g["w"], margin), g)
        t_region[c, y0:y1 + 1, x0:x1 + 1] = True
        t_has[c] = True
    return t_logit, t_region, t_has


def follow_augmentation(t_logit: torch.Tensor, t_region: torch.Tensor, applied: dict, g: dict,
                        valid_after: torch.Tensor, fill: float = KD_FILL):
    """Apply the spatial transform augment() already drew (its `applied` angle / scale, same
    grid): logits bilinear (with the image's fill trick), region nearest; then region & valid.
    The intensity augmentations do not touch the teacher. No random number is drawn."""
    if applied["rotation"] or applied["scaling"]:
        S = t_logit.shape[-1]
        grid = A._grid(S, (g["h2"] / 2.0, g["w2"] / 2.0), applied["angle_value"], applied["scale_value"])
        t_logit = A._sample(t_logit - fill, grid, "bilinear") + fill
        t_region = A._sample(t_region.float(), grid, "nearest") > 0.5
    return t_logit, t_region & valid_after[None]


# ---------------------------------------------------------------------------------- loss
def kd_loss(out_full: torch.Tensor, t_logit: torch.Tensor, t_region: torch.Tensor, valid: torch.Tensor,
            t_has: torch.Tensor, temperature: float) -> dict:
    """ARM4_CODE_PLAN.md §3. out_full B x 3 x S x S student logits (full resolution);
    t_logit B x 2 x S x S; t_region B x 2 x S x S bool; valid B x S x S bool; t_has B x 2 bool.
    total = (1/B) sum over used (image, kidney) of T^2 x mean over region&valid of the binary
    KL(teacher || student)."""
    B, n_cls, H, W = out_full.shape
    if n_cls != C.N_CLASSES:
        raise AssertionError(f"student output has {n_cls} classes")
    for name, t, shape in (("t_logit", t_logit, (B, 2, H, W)), ("t_region", t_region, (B, 2, H, W)),
                           ("valid", valid, (B, H, W)), ("t_has", t_has, (B, 2))):
        if tuple(t.shape) != shape:
            raise AssertionError(f"kd_loss: {name} {tuple(t.shape)} != {shape}")
    T = float(temperature)
    if not T > 0:
        raise ValueError(f"temperature {temperature}")
    z = out_full.float() / T
    lse = torch.logsumexp(z, 1)
    total = z.new_zeros(())
    terms, images, n_empty = [], [], 0
    for c, k, other in ((0, 1, [0, 2]), (1, 2, [0, 1])):
        log_q = z[:, k] - lse
        log_1mq = torch.logsumexp(z[:, other], 1) - lse
        a = t_logit[:, c].float() / T
        lp, l1mp = F.logsigmoid(a), F.logsigmoid(-a)
        kl = torch.exp(lp) * (lp - log_q) + torch.exp(l1mp) * (l1mp - log_1mq)
        m = t_region[:, c].bool() & valid.bool()
        n = m.sum((1, 2))
        has = t_has[:, c].bool()
        use = has & (n > 0)
        n_empty += int((has & (n == 0)).sum())
        per = (T * T) * torch.where(m, kl, torch.zeros_like(kl)).sum((1, 2)) / n.clamp(min=1).to(kl.dtype)
        total = total + per[use].sum()
        terms.append(per[use].detach())
        images.append(torch.nonzero(use).flatten())
    terms_t, images_t = torch.cat(terms), torch.cat(images)
    return {"total": total / B, "terms": terms_t, "term_image": images_t, "n_terms": int(terms_t.numel()),
            "n_empty_region": n_empty, "n_images_with_terms": int(torch.unique(images_t).numel())}


# ------------------------------------------------------------------------------ diagnostic
def softness(tstore: TeacherStore, indices, margin: float, temps=(1.0, 2.0)) -> dict:
    """Report only: per modality and T, the fraction of native region pixels whose teacher
    probability lies in (0.05, 0.95) -- how much 'soft' information the targets carry."""
    acc: dict = {}
    for i in indices:
        r = tstore.store.rows[i]
        boxes = tstore.store.boxes(i)
        for side, (lg, _, used) in tstore.maps(i).items():
            if not used:
                continue
            x0, y0, x1, y1 = region_box(boxes[side], r["h"], r["w"], margin)
            a = lg[y0:y1 + 1, x0:x1 + 1].astype(np.float64)
            for T in temps:
                p = 1.0 / (1.0 + np.exp(-a / T))
                s = acc.setdefault((r["modality"], T), [0, 0])
                s[0] += int(((p > 0.05) & (p < 0.95)).sum())
                s[1] += a.size
    return {f"{m}@T{T:g}": (v[0] / v[1] if v[1] else float("nan")) for (m, T), v in sorted(acc.items())}
