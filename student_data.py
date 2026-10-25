#!/usr/bin/env python
"""Students' data (ARM3_CODE_PLAN.md §3.4): AMOS train/val slices held in RAM, deterministic
draws, training and validation datasets.

* Same files as every arm: data/processed/manifest.csv + .npz (img uint8, msk 0/1/2).
* All band slices (kidney, sliver, kidney-free) of a split, sorted by (case, z), loaded once into
  two flat uint8 buffers, so DataLoader workers (fork) share them with no per-item disk reads.
* Draws: CT/MRI 50/50, uniform within modality, with replacement, a pure function of
  (seed, epoch); per-draw randomness from sha256("arm3|seed|epoch|i").
* Arm 4 (ARM4_CODE_PLAN.md §6): TrainItems(teacher=...) adds the teacher's network-frame maps
  (student_kd); the "arm3" draw key is kept on purpose, so arm 4 sees arm 3's exact draws.
"""
from __future__ import annotations

import csv
import hashlib
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_sam as E  # noqa: E402  (load_manifest only; imports no SAM code)
import sam_metrics as M  # noqa: E402
import student_aug as A  # noqa: E402
import student_core as C  # noqa: E402

SPLITS = ("train", "val")
MIN_AREA = 50                     # prep_data --min-area (D16): a side has a box iff area >= 50
# DECISIONS D21 (prep job 329158): {split: {modality: (all slices, has_kidney, boxed)}}
EXPECTED_COUNTS = {"train": {"ct": (11033, 5786, 5760), "mri": (3442, 2567, 2535)},
                   "val": {"ct": (3688, 1912, 1901), "mri": (1131, 846, 843)}}
VAL_LEVELS_BOX = (0.0, 0.10)
TRAIN_MAX_LEVEL = 0.20
KD_KEYS = ("t_logit", "t_region", "t_has")   # arm 4 only (ARM4_CODE_PLAN.md §6)


def check_split_integrity(processed: Path) -> dict:
    """Every patient in exactly one split (D10), read from the raw manifest."""
    with open(Path(processed) / "manifest.csv") as fh:
        rows = list(csv.DictReader(fh))
    splits_of: dict = {}
    for r in rows:
        splits_of.setdefault(r["patient"], set()).add(r["split"])
    multi = {p: sorted(s) for p, s in splits_of.items() if len(s) > 1}
    if multi:
        raise SystemExit(f"patients in more than one split (leak): {dict(list(multi.items())[:5])}")
    return dict(Counter(next(iter(s)) for s in splits_of.values()))


def manifest_sha256(processed: Path) -> str:
    h = hashlib.sha256()
    with open(Path(processed) / "manifest.csv", "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _tight_box(mask: np.ndarray):
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return None
    return [int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())]


class SliceStore:
    """Every slice of one split in RAM: rows (manifest, parsed), flat image/mask buffers."""

    def __init__(self, processed: Path, split: str, expected: dict | None = EXPECTED_COUNTS,
                 threads: int = 8, log=print):
        if split not in SPLITS:
            raise ValueError(f"student data may only use {SPLITS}, not {split!r}")
        t0 = time.time()
        self.processed, self.split = Path(processed), split
        rows = E.load_manifest(self.processed, splits=(split,))
        if any(r["split"] != split for r in rows):
            raise AssertionError("load_manifest returned rows of another split")
        rows.sort(key=lambda r: (r["case"], r["z"]))
        for r in rows:
            hk = int(r["has_kidney"])
            if hk not in (0, 1):
                raise AssertionError(f"{r['case']} z={r['z']}: has_kidney {r['has_kidney']!r}")
            r["has_kidney"] = hk
            if r["modality"] not in ("ct", "mri"):
                raise AssertionError(f"{r['case']}: modality {r['modality']!r}")
        counts = {}
        for mod in ("ct", "mri"):
            rs = [r for r in rows if r["modality"] == mod]
            counts[mod] = (len(rs), sum(r["has_kidney"] for r in rs), sum(r["sam_eligible"] for r in rs))
        if expected is not None and counts != {m: tuple(v) for m, v in expected[split].items()}:
            raise SystemExit(f"{split}: (slices, has_kidney, boxed) per modality {counts} != expected "
                             f"{expected[split]} (DECISIONS D21)")
        self.rows, self.counts = rows, counts
        sizes = np.array([r["h"] * r["w"] for r in rows], dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        self.img = np.empty(int(self.offsets[-1]), dtype=np.uint8)
        self.msk = np.empty(int(self.offsets[-1]), dtype=np.uint8)
        with ThreadPoolExecutor(max_workers=max(1, int(threads))) as ex:
            for fut in [ex.submit(self._load_one, i) for i in range(len(rows))]:
                fut.result()                      # re-raises any per-slice failure
        self.load_s = time.time() - t0
        self.gib = 2 * self.img.nbytes / 2 ** 30
        log(f"{split}: {len(rows)} slices {counts} loaded in {self.load_s:.0f}s ({self.gib:.2f} GiB)")

    def _load_one(self, i: int) -> None:
        r = self.rows[i]
        with np.load(self.processed / r["path"]) as d:
            img, msk = d["img"], d["msk"]
        tag = f"{r['case']} z={r['z']}"
        if img.dtype != np.uint8 or msk.dtype != np.uint8:
            raise AssertionError(f"{tag}: dtypes {img.dtype}/{msk.dtype}, expected uint8")
        if img.shape != (r["h"], r["w"]) or msk.shape != img.shape:
            raise AssertionError(f"{tag}: shapes {img.shape}/{msk.shape} != manifest ({r['h']}, {r['w']})")
        if not set(np.unique(msk).tolist()) <= {0, 1, 2}:
            raise AssertionError(f"{tag}: mask values {np.unique(msk).tolist()}")
        for side, prefix in C.SIDES:
            sm = msk == side
            area = int(sm.sum())
            if area != r[f"{prefix}_area"]:
                raise AssertionError(f"{tag} {prefix}: area {area} != manifest {r[f'{prefix}_area']}")
            box = r[f"{prefix}_box"]
            if (box is not None) != (area >= MIN_AREA):
                raise AssertionError(f"{tag} {prefix}: box {box} inconsistent with area {area}")
            if box is not None and list(box) != _tight_box(sm):
                raise AssertionError(f"{tag} {prefix}: manifest box {box} != tight box {_tight_box(sm)}")
        if r["has_kidney"] != int(r["rk_area"] + r["lk_area"] > 0):
            raise AssertionError(f"{tag}: has_kidney inconsistent with areas")
        if r["sam_eligible"] != int(r["rk_box"] is not None or r["lk_box"] is not None):
            raise AssertionError(f"{tag}: sam_eligible inconsistent with boxes")
        a, b = int(self.offsets[i]), int(self.offsets[i + 1])
        self.img[a:b] = img.ravel()
        self.msk[a:b] = msk.ravel()

    def __len__(self) -> int:
        return len(self.rows)

    def get(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        r, a, b = self.rows[i], int(self.offsets[i]), int(self.offsets[i + 1])
        return self.img[a:b].reshape(r["h"], r["w"]), self.msk[a:b].reshape(r["h"], r["w"])

    def boxes(self, i: int) -> dict:
        r = self.rows[i]
        return {side: (None if r[f"{p}_box"] is None else tuple(int(v) for v in r[f"{p}_box"]))
                for side, p in C.SIDES}

    def size_table(self) -> dict:
        t = Counter((r["modality"], r["h"], r["w"]) for r in self.rows)
        return {f"{m} {h}x{w}": n for (m, h, w), n in sorted(t.items())}


# ------------------------------------------------------------------------------ draws
def make_draws(store: SliceStore, epoch: int, seed: int, n: int) -> list[tuple[int, int]]:
    """Epoch `epoch`'s draws [(i, slice index)]: weight 1/n_modality per slice (50/50 CT/MRI,
    uniform within modality), with replacement. Pure function of (seed, epoch)."""
    n_mod = Counter(r["modality"] for r in store.rows)
    if set(n_mod) != {"ct", "mri"}:
        raise ValueError(f"need both modalities, have {dict(n_mod)}")
    w = torch.tensor([1.0 / n_mod[r["modality"]] for r in store.rows], dtype=torch.float64)
    g = torch.Generator().manual_seed(int(seed) * 1000 + int(epoch))
    return list(enumerate(torch.multinomial(w, int(n), replacement=True, generator=g).tolist()))


def draw_rng(seed: int, epoch: int, i: int) -> np.random.Generator:
    key = hashlib.sha256(f"arm3|{seed}|{epoch}|{i}".encode()).digest()
    return np.random.default_rng(int.from_bytes(key[:8], "big"))


def position(p: int, per_epoch: int) -> tuple[int, int]:
    """Global training position -> (epoch 1.., draw index i)."""
    return p // per_epoch + 1, p % per_epoch


# --------------------------------------------------------------------------- datasets
class TrainItems(Dataset):
    """Training positions [start, end): position p -> epoch, draw i -> slice -> network input,
    label, valid (+ jittered box channels) -> augmentation. Deterministic per position."""

    def __init__(self, store: SliceStore, variant: str, seed: int, norm: C.Norm, per_epoch: int,
                 start: int, end: int, size: int = C.NET_SIZE, augment: bool = True,
                 max_level: float = TRAIN_MAX_LEVEL, teacher=None, kd_margin: float = 0.20):
        if variant not in C.VARIANTS:
            raise ValueError(variant)
        if not 0 <= start <= end:
            raise ValueError(f"bad position range [{start}, {end})")
        if teacher is not None and variant != "plain":
            raise ValueError("distillation (a teacher) is defined for the plain student only")
        self.store, self.variant, self.seed, self.norm = store, variant, int(seed), norm
        self.per_epoch, self.start, self.end, self.size = int(per_epoch), int(start), int(end), int(size)
        self.augment, self.max_level = bool(augment), float(max_level)
        self.teacher, self.kd_margin = teacher, float(kd_margin)    # arm 4; None = arm 3 exactly
        self._draws: dict = {}

    def __len__(self) -> int:
        return self.end - self.start

    def draws(self, epoch: int) -> list:
        if epoch not in self._draws:
            self._draws = {epoch: make_draws(self.store, epoch, self.seed, self.per_epoch)}
        return self._draws[epoch]

    def train_boxes(self, idx: int, rng: np.random.Generator) -> dict:
        """Per boxed side: level ~ U[0, max_level], then jitter_box (arm 2's recipe)."""
        r, out = self.store.rows[idx], {}
        for side, clean in self.store.boxes(idx).items():
            if clean is None:
                out[side] = None
                continue
            level = float(rng.uniform(0.0, self.max_level))
            jseed = int(rng.integers(0, 2 ** 62))
            out[side] = M.jitter_box(clean, level, r["h"], r["w"], jseed)
        return out

    def __getitem__(self, k: int) -> dict:
        p = self.start + int(k)
        if not self.start <= p < self.end:
            raise IndexError(k)
        epoch, i = position(p, self.per_epoch)
        idx = self.draws(epoch)[i][1]
        rng = draw_rng(self.seed, epoch, i)
        img, msk = self.store.get(idx)
        boxes = self.train_boxes(idx, rng) if self.variant == "box" else None
        x, valid, g = C.net_input(img, self.norm, self.variant, boxes, self.size)
        label = C.to_net_label(msk, g)
        if self.augment:
            im, label, valid, bx, applied = A.augment(x[:1], label, valid, x[1:], g, self.norm.pad_value, rng)
            x = torch.cat([im, bx], 0)
        r = self.store.rows[idx]
        if int(label.max()) > 2 or int(label.min()) < 0:
            raise AssertionError(f"label values outside 0..2 at position {p}")
        label = label.to(torch.uint8)            # 8x less inter-process traffic than int64
        item = {"x": x, "label": label, "valid": valid, "pos": p, "epoch": epoch, "i": i,
                "idx": idx, "modality": r["modality"], "has_kidney": r["has_kidney"]}
        if self.teacher is not None:
            # arm 4: the teacher's maps follow the geometry already drawn above; no rng is used,
            # so x / label / valid are exactly the arm-3 item at this position (plan invariant 4)
            import student_kd as K
            t_logit, t_region, t_has = K.teacher_targets(self.teacher, idx, g, self.kd_margin)
            if self.augment:
                t_logit, t_region = K.follow_augmentation(t_logit, t_region, applied, g, valid, K.KD_FILL)
            else:
                t_region = t_region & valid[None]
            item.update(t_logit=t_logit.to(K.TRANSPORT_DTYPE), t_region=t_region, t_has=t_has)
        return item


def collate_train(batch: list) -> dict:
    out = {k: torch.stack([b[k] for b in batch]) for k in ("x", "label", "valid")}
    for k in ("pos", "epoch", "i", "idx", "modality", "has_kidney"):
        out[k] = [b[k] for b in batch]
    kd = [k for k in KD_KEYS if k in batch[0]]
    if any((k in b) != (k in batch[0]) for b in batch for k in KD_KEYS):
        raise AssertionError("a batch mixes items with and without teacher targets")
    for k in kd:
        out[k] = torch.stack([b[k] for b in batch])
    return out


class ValItems(Dataset):
    """Every val slice, no augmentation. Box variant: one input per level, boxes from the
    evaluation's own jitter_box(jitter_seed) (as arm 2 and the test)."""

    def __init__(self, store: SliceStore, variant: str, norm: C.Norm, size: int = C.NET_SIZE,
                 indices: list | None = None):
        self.store, self.variant, self.norm, self.size = store, variant, norm, int(size)
        self.indices = list(range(len(store))) if indices is None else list(indices)
        self.levels = (C.NO_BOX_LEVEL,) if variant == "plain" else VAL_LEVELS_BOX

    def __len__(self) -> int:
        return len(self.indices)

    @staticmethod
    def val_box(row: dict, side: int, clean, level: float):
        return M.jitter_box(clean, level, row["h"], row["w"],
                            M.jitter_seed(row["case"], int(row["z"]), int(side), float(level)))

    def boxes_at(self, idx: int, level: float) -> dict:
        r = self.store.rows[idx]
        return {side: (None if clean is None else self.val_box(r, side, clean, level))
                for side, clean in self.store.boxes(idx).items()}

    def __getitem__(self, k: int) -> dict:
        idx = self.indices[k]
        img, _ = self.store.get(idx)
        xs, geom = {}, None
        for lev in self.levels:
            boxes = None if self.variant == "plain" else self.boxes_at(idx, lev)
            x, _, geom = C.net_input(img, self.norm, self.variant, boxes, self.size)
            xs[C.level_key(lev)] = x
        return {"xs": xs, "geom": geom, "idx": idx}


def collate_list(batch: list) -> list:
    return batch


def worker_init(_worker_id: int) -> None:
    torch.set_num_threads(1)


def mp_context():
    """fork on Linux (workers share the in-RAM store without copying); the default elsewhere."""
    return "fork" if sys.platform.startswith("linux") else None
