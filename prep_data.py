#!/usr/bin/env python
"""Build the 2D kidney-slice dataset for the IE643 SAM-distillation project.

One script, three datasets, one manifest (DECISIONS.md D2-D16):

  amos   AMOS22  NIfTI  CT + MRI   internal: the 360 labelled cases (imagesTr + imagesVa)
                                   pooled, then a seeded per-modality 60/20/20 split
                                   (CT 180/60/60, MRI 36/12/12)
  word   WORD    NIfTI  CT         external test: imagesTr only (100 scans)
  chaos  CHAOS   DICOM  MRI        external test: Train_Sets/MR, 20 patients x
                 + PNG             {T1 in-phase, T1 out-of-phase, T2-SPIR}

Geometry. Every volume is put in canonical RAS orientation (voxel axes increase
toward the patient's Right, Anterior, Superior), then every axial slice is stored in
radiological view: anterior at the top, the PATIENT'S RIGHT ON THE IMAGE LEFT.

Square pixels (D19). Slices whose two in-plane spacings differ by more than 5% (the 18
coronally acquired AMOS MRI) are stretched to square pixels at the finer spacing: image
bilinear, mask nearest-neighbour. Upsampling only, so no labelled pixel is ever lost.
Every stored slice therefore shows anatomy in its true proportions.

Labels. Masks are unified to 0 = background, 1 = right kidney, 2 = left kidney.
Each dataset numbers its organs differently (AMOS R=2 L=3, WORD R=4 L=3, CHAOS grey
levels), so the maps live in one table below and nowhere else.

Ground truth is never edited. Slivers (kidney area < --min-area px) keep their mask
pixels but get no box: SAM is never prompted with a degenerate box, while the
student's targets stay exact.

Anatomy-anchored QC runs on every volume (canonical RAS index space) and catches the
silent failures that would otherwise corrupt every downstream number:
  q_modality      CT contains air (< -500 HU); MRI does not
  q_liver_right   liver centroid is on the patient's right of the kidney midpoint
                  (catches a left/right orientation flip)
  q_kidney_lr     right-kidney centroid is right of the left-kidney centroid
                  (catches a swapped label map)
  q_liver_above   liver dome is superior to both kidneys   (catches reversed slice order)
  q_kidney_post   kidneys are posterior to the liver centroid  (catches an A/P flip)

Outputs, under --out:
  manifest.csv   one row per stored slice: boxes, areas, mm spacing, split, path
  volumes.csv    one row per volume: geometry, intensity range, QC verdicts, errors
  run.json       arguments, label maps, constants, library versions
  slices/<dataset>/<case>/<z>.npz   img (uint8 HxW), msk (uint8 HxW, 0/1/2)
  qc/<dataset>/<case>_z<z>.png      overlays for visual verification

Boxes are XYXY in pixel coordinates of the stored slice (x = column, y = row),
inclusive and tight. Box noise is added at load time, not here (D16).

Usage
-----
    python prep_data.py --amos-root  .../data/amos22/amos22 \
                        --word-root  .../data/word/WORD-V0.1.0 \
                        --word-test-labels .../data/word/labelsTs \
                        --chaos-root .../data/chaos/Train_Sets \
                        --out .../data/processed --workers 8
    add --limit 2 for a smoke test (split assignment is unchanged by --limit)
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import re
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

try:
    import nibabel as nib
    import pydicom
    from PIL import Image, ImageDraw
    from tqdm import tqdm
except ImportError as exc:  # pragma: no cover - environment guard
    sys.exit(f"missing dependency '{exc.name}'. Activate the seg311 env first.")

# ----------------------------------------------------------------------- constants
CT_WINDOW = (-150.0, 250.0)        # abdominal soft tissue, W400 L50 (D16)
MRI_PERCENTILES = (0.5, 99.5)      # per-volume normalisation (D16)
AMOS_MRI_MIN_ID = 500              # D10b: id >= 500 is MRI, verified on disk (Tr 200/40, Va 100/20)
SPLIT_FRACTIONS = (0.6, 0.2, 0.2)  # D14: train/val/test of the pooled AMOS labelled cases
# D18: removed from the pool BEFORE the split (user decision 2026-10-04), so the 295
# remaining CT split exactly 177/59/59. MRI is untouched (36/12/12).
AMOS_EXCLUDE = {
    "amos_0085": "label noise: left-kidney specks (0.1-0.2 mL) in the chest, slices 102-107",
    "amos_0320": "renal transplant: 336 mL pelvic graft plus two atrophic native kidneys",
    "amos_0112": "single kidney: no right-kidney label",
    "amos_0142": "single kidney: no left-kidney label",
    "amos_0323": "single kidney: no right-kidney label",
}
SYSTEMATIC_FAIL_RATE = 0.2         # a QC check failing on >20% of a dataset is a pipeline bug
SQUARE_TOL = 0.05                  # D19: in-plane pixels >5% non-square are resampled to square

# Organ label values per NIfTI dataset, from each dataset.json (verified on Prajna).
NIFTI_LABELS = {
    "amos": {"right": 2, "left": 3, "liver": 6},
    "word": {"right": 4, "left": 3, "liver": 1},   # NOTE: L/R order opposite to AMOS
}
# CHAOS MR ground truth: one PNG per slice, organs encoded as grey ranges (lo, hi].
# Ranges from nnU-Net's official CHAOS conversion (Task037_038_Chaos_Challenge.py).
CHAOS_GREY = {"liver": (55, 70), "right": (110, 135), "left": (175, 200), "spleen": (240, 255)}
# sequence -> (group folder that owns the Ground/ PNGs, DICOM sub-folder)
CHAOS_SEQUENCES = {
    "t1in": ("T1DUAL", "DICOM_anon/InPhase"),
    "t1out": ("T1DUAL", "DICOM_anon/OutPhase"),
    "t2": ("T2SPIR", "DICOM_anon"),
}

MANIFEST_FIELDS = [
    "dataset", "patient", "case", "sequence", "modality", "split", "z", "n_slices",
    "has_kidney", "rk_area", "lk_area", "rk_box", "lk_box", "sam_eligible",
    "h", "w", "sp_row", "sp_col", "sp_slice", "path",
]
VOLUME_FIELDS = [
    "dataset", "patient", "case", "sequence", "modality", "split", "orig_axcodes",
    "shape", "sp_row", "sp_col", "sp_slice", "resampled", "img_min", "img_max", "n_slices", "n_kept",
    "kidney_z_first", "kidney_z_last", "n_kidney_slices", "rk_vox", "lk_vox", "liver_vox",
    "q_modality", "q_liver_right", "q_kidney_lr", "q_liver_above", "q_kidney_post",
    "pairing", "unmapped_frac", "warnings", "error", "seconds",
]
QC_CHECKS = ["q_modality", "q_liver_right", "q_kidney_lr", "q_liver_above", "q_kidney_post"]


class VolumeError(Exception):
    """A volume that cannot be processed safely. Recorded in volumes.csv, never silent."""


# ------------------------------------------------------------------------ helpers
def natural_key(name: str) -> list:
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def case_number(path: Path) -> int:
    m = re.search(r"(\d+)", path.name)
    if not m:
        raise ValueError(f"no case number in {path.name}")
    return int(m.group(1))


def voxel_sizes(affine: np.ndarray) -> tuple[float, float, float]:
    return tuple(float(v) for v in np.sqrt((affine[:3, :3] ** 2).sum(axis=0)))


def to_display(slice_xy: np.ndarray) -> np.ndarray:
    """Canonical RAS slice [x, y] -> radiological view [row, col]:
    anterior at the top, the patient's right on the image LEFT."""
    return np.ascontiguousarray(slice_xy.T[::-1, ::-1])


def square_geometry(h: int, w: int, sp_row: float, sp_col: float):
    """D19. For square-enough pixels returns (None, sp_row, sp_col). Otherwise returns the
    (h, w) that makes pixels square at the FINER spacing, plus the exact resulting spacing
    (physical extent / new pixel count), so NSD stays correct in mm."""
    if abs(sp_row / sp_col - 1.0) <= SQUARE_TOL:
        return None, sp_row, sp_col
    target = min(sp_row, sp_col)
    nh = max(1, int(round(h * sp_row / target)))
    nw = max(1, int(round(w * sp_col / target)))
    return (nh, nw), h * sp_row / nh, w * sp_col / nw


def make_slice(image_u8: np.ndarray, kid: np.ndarray, z: int, out_hw):
    """Canonical volumes -> stored (img, msk) for slice z: radiological view, then the
    D19 square-pixel resample. Upsampling only, so nearest-neighbour keeps every label."""
    img2d = to_display(image_u8[:, :, z])
    m2d = to_display(kid[:, :, z])
    if out_hw is not None:
        nh, nw = out_hw
        img2d = np.array(Image.fromarray(img2d).resize((nw, nh), Image.BILINEAR))
        m2d = np.array(Image.fromarray(m2d).resize((nw, nh), Image.NEAREST))
    return img2d, m2d


def to_uint8(volume: np.ndarray, modality: str) -> np.ndarray:
    if modality == "ct":
        lo, hi = CT_WINDOW
    else:
        lo, hi = (float(v) for v in np.percentile(volume, MRI_PERCENTILES))
        if hi <= lo:                       # blank or constant volume
            hi = lo + 1.0
    out = (np.clip(volume, lo, hi) - lo) * (255.0 / (hi - lo))
    return np.rint(out).astype(np.uint8)


def bbox_xyxy(mask: np.ndarray) -> list[int] | None:
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return None
    return [int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())]


def centroid(mask: np.ndarray) -> np.ndarray | None:
    idx = np.nonzero(mask)
    if idx[0].size == 0:
        return None
    return np.array([float(i.mean()) for i in idx])


def per_slice_any(vol: np.ndarray) -> np.ndarray:
    """[X, Y, Z] -> bool[Z]: does slice z contain any nonzero voxel."""
    return vol.reshape(-1, vol.shape[2]).any(axis=0)


# ------------------------------------------------------------------------ loaders
def load_nifti(job: dict):
    img_file = nib.load(job["image"])
    orig_axcodes = "".join(nib.aff2axcodes(img_file.affine))
    img = nib.as_closest_canonical(img_file)
    lbl = nib.as_closest_canonical(nib.load(job["label"]))
    if len(img.shape) != 3:
        raise VolumeError(f"expected a 3D image, got shape {img.shape}")
    if img.shape != lbl.shape:
        raise VolumeError(f"image/label shape mismatch {img.shape} vs {lbl.shape}")
    if not np.allclose(img.affine, lbl.affine, atol=0.05):
        raise VolumeError("image/label affine mismatch after canonicalisation")
    image = img.get_fdata(dtype=np.float32)
    raw = np.asarray(lbl.dataobj)
    if not np.issubdtype(raw.dtype, np.integer):
        raw = np.rint(raw)
    raw = raw.astype(np.int16)
    lab = NIFTI_LABELS[job["dataset"]]
    kid = np.zeros(raw.shape, np.uint8)
    kid[raw == lab["right"]] = 1
    kid[raw == lab["left"]] = 2
    liver = raw == lab["liver"]
    return image, kid, liver, img.affine, {"orig_axcodes": orig_axcodes, "pairing": "nifti"}


def read_dicom_series(folder: Path) -> dict:
    """Read one DICOM series, sorted spatially (ascending along the slice normal),
    with a NIfTI-convention (RAS) affine for array index [col, row, slice]."""
    if not folder.is_dir():
        raise VolumeError(f"missing DICOM folder {folder}")
    entries = []
    for f in sorted((p for p in folder.iterdir() if p.is_file() and not p.name.startswith(".")),
                    key=lambda p: natural_key(p.name)):
        try:
            ds = pydicom.dcmread(str(f))
        except Exception:
            continue                                   # not a DICOM file (e.g. a readme)
        if "ImagePositionPatient" in ds and "PixelData" in ds:
            entries.append((f, ds))
    if not entries:
        raise VolumeError(f"no DICOM images in {folder}")

    iop = np.array(entries[0][1].ImageOrientationPatient, dtype=float)
    for _, ds in entries:
        if not np.allclose(np.array(ds.ImageOrientationPatient, dtype=float), iop, atol=1e-4):
            raise VolumeError(f"mixed slice orientations in {folder}")
    row_dir, col_dir = iop[:3], iop[3:]     # DICOM: direction of increasing column / row index
    normal = np.cross(row_dir, col_dir)
    pos = np.array([np.array(ds.ImagePositionPatient, dtype=float) for _, ds in entries])
    dist = pos @ normal
    if len(np.unique(np.round(dist, 3))) != len(dist):
        raise VolumeError(f"duplicate slice positions in {folder}")
    order = np.argsort(dist)
    entries = [entries[i] for i in order]
    pos, dist = pos[order], dist[order]

    rows, cols = int(entries[0][1].Rows), int(entries[0][1].Columns)
    stack = np.empty((len(entries), rows, cols), np.float32)
    for k, (_, ds) in enumerate(entries):
        if (int(ds.Rows), int(ds.Columns)) != (rows, cols):
            raise VolumeError(f"mixed slice sizes in {folder}")
        slope = float(getattr(ds, "RescaleSlope", 1) or 1)
        intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
        stack[k] = ds.pixel_array.astype(np.float32) * slope + intercept

    d_row, d_col = (float(v) for v in entries[0][1].PixelSpacing)   # [between rows, between cols]
    affine_lps = np.eye(4)
    affine_lps[:3, 0] = row_dir * d_col          # +i (column index) moves along row_dir
    affine_lps[:3, 1] = col_dir * d_row          # +j (row index) moves along col_dir
    if len(entries) > 1:
        steps = np.diff(dist)
        if np.max(np.abs(steps - np.median(steps))) > 0.05 * np.median(steps):
            raise VolumeError(f"non-uniform slice spacing in {folder}")
        affine_lps[:3, 2] = (pos[-1] - pos[0]) / (len(entries) - 1)
    else:
        affine_lps[:3, 2] = normal * float(getattr(entries[0][1], "SliceThickness", 1.0) or 1.0)
    affine_lps[:3, 3] = pos[0]
    affine = np.diag([-1.0, -1.0, 1.0, 1.0]) @ affine_lps            # DICOM LPS -> NIfTI RAS
    return {
        "files": [f for f, _ in entries],
        "inst": [int(getattr(ds, "InstanceNumber", 0) or 0) for _, ds in entries],
        "dist": dist, "rows": rows, "cols": cols, "stack": stack, "affine": affine,
    }


def read_png(path: Path) -> np.ndarray:
    a = np.array(Image.open(path))
    if a.ndim == 3:
        if not (np.array_equal(a[..., 0], a[..., 1]) and np.array_equal(a[..., 1], a[..., 2])):
            raise VolumeError(f"colour ground-truth PNG {path.name}")
        a = a[..., 0]
    a = a.astype(np.int32)
    if a.max() > 255:
        raise VolumeError(f"ground-truth PNG {path.name} is not 8-bit (max {a.max()})")
    return a


def chaos_codes(png: np.ndarray):
    kid = np.zeros(png.shape, np.uint8)
    lo, hi = CHAOS_GREY["right"]
    kid[(png > lo) & (png <= hi)] = 1
    lo, hi = CHAOS_GREY["left"]
    kid[(png > lo) & (png <= hi)] = 2
    lo, hi = CHAOS_GREY["liver"]
    liver = (png > lo) & (png <= hi)
    lo, hi = CHAOS_GREY["spleen"]
    spleen = (png > lo) & (png <= hi)
    labelled = int((png > 0).sum())
    unmapped = int(((png > 0) & (kid == 0) & ~liver & ~spleen).sum())
    return kid, liver, labelled, unmapped


def load_chaos(job: dict):
    pdir = Path(job["patient_dir"])
    group, sub = CHAOS_SEQUENCES[job["sequence"]]
    series = read_dicom_series(pdir / group / sub)
    ground = pdir / group / "Ground"
    if not ground.is_dir():
        raise VolumeError(f"missing ground-truth folder {ground}")
    pngs = sorted((p for p in ground.iterdir() if p.suffix.lower() == ".png"),
                  key=lambda p: natural_key(p.name))
    n = len(series["files"])
    if len(pngs) != n:
        raise VolumeError(f"{len(pngs)} ground-truth PNGs for {n} DICOM slices")
    png_by_stem = {p.stem: p for p in pngs}

    # Pair each spatially ordered DICOM slice with its PNG, safest method first.
    if {f.stem for f in series["files"]} == set(png_by_stem):
        ordered, pairing = [png_by_stem[f.stem] for f in series["files"]], "stem"
    else:
        sibling, sibling_seq = None, ""
        for other_seq, (other_group, other_sub) in CHAOS_SEQUENCES.items():
            if other_group == group and other_seq != job["sequence"]:
                other = read_dicom_series(pdir / other_group / other_sub)
                if {f.stem for f in other["files"]} == set(png_by_stem):
                    sibling, sibling_seq = other, other_seq
                    break
        if sibling is not None:
            # T1 in- and out-of-phase are one dual-echo acquisition: same slice positions.
            if len(sibling["files"]) != n or (sibling["rows"], sibling["cols"]) != (series["rows"], series["cols"]):
                raise VolumeError(f"{job['sequence']} geometry differs from {sibling_seq}")
            step = abs(float(np.median(np.diff(series["dist"])))) if n > 1 else 1.0
            if not np.allclose(series["dist"], sibling["dist"], atol=0.25 * step):
                raise VolumeError(f"{job['sequence']} slice positions differ from {sibling_seq}")
            ordered = [png_by_stem[f.stem] for f in sibling["files"]]
            pairing = f"position-via-{sibling_seq}"
        else:
            # Last resort: PNG name order <-> DICOM InstanceNumber order. q_liver_above
            # verifies the direction; --chaos-png-order reverse flips it if it fails.
            by_inst = np.argsort(series["inst"], kind="stable")
            seq = pngs if job["png_order"] == "instance" else pngs[::-1]
            ordered = [None] * n
            for rank, spatial_idx in enumerate(by_inst):
                ordered[spatial_idx] = seq[rank]
            pairing = f"rank-{job['png_order']}"

    rows, cols = series["rows"], series["cols"]
    kid = np.zeros((n, rows, cols), np.uint8)
    liver = np.zeros((n, rows, cols), np.uint8)
    labelled = unmapped = 0
    for k, p in enumerate(ordered):
        png = read_png(p)
        if png.shape != (rows, cols):
            raise VolumeError(f"PNG {p.name} is {png.shape}, DICOM slice is {(rows, cols)}")
        kk, ll, lab, unm = chaos_codes(png)
        kid[k], liver[k] = kk, ll
        labelled += lab
        unmapped += unm

    aff = series["affine"]
    orig_axcodes = "".join(nib.aff2axcodes(aff))
    # [slice, row, col] -> NIfTI array index [col, row, slice]
    img_c = nib.as_closest_canonical(nib.Nifti1Image(series["stack"].transpose(2, 1, 0), aff))
    kid_c = nib.as_closest_canonical(nib.Nifti1Image(kid.transpose(2, 1, 0), aff))
    liv_c = nib.as_closest_canonical(nib.Nifti1Image(liver.transpose(2, 1, 0), aff))
    image = img_c.get_fdata(dtype=np.float32)
    return (image, np.asarray(kid_c.dataobj).astype(np.uint8),
            np.asarray(liv_c.dataobj).astype(bool), img_c.affine,
            {"orig_axcodes": orig_axcodes, "pairing": pairing,
             "unmapped_frac": round(unmapped / max(1, labelled), 6)})


LOADERS = {"amos": load_nifti, "word": load_nifti, "chaos": load_chaos}


# ---------------------------------------------------------------------------- QC
def run_qc(image: np.ndarray, kid: np.ndarray, liver: np.ndarray, modality: str) -> dict:
    """Anatomy-anchored checks in canonical RAS index space: axis 0 -> patient Right,
    axis 1 -> Anterior, axis 2 -> Superior."""
    q = {}
    lo = float(image.min())
    q["q_modality"] = "pass" if ((lo < -500) if modality == "ct" else (lo > -100)) else "fail"

    cr, cl, cv = centroid(kid == 1), centroid(kid == 2), centroid(liver)
    q["q_kidney_lr"] = "n/a" if cr is None or cl is None else ("pass" if cr[0] > cl[0] else "fail")
    if cv is None or cr is None or cl is None:
        q["q_liver_right"] = "n/a"
    else:
        q["q_liver_right"] = "pass" if cv[0] > (cr[0] + cl[0]) / 2 else "fail"

    z_last = kid.shape[2] - 1
    kid_z, liv_z = per_slice_any(kid), per_slice_any(liver)
    if cv is None or not kid_z.any():
        q["q_liver_above"] = "n/a"
    else:
        top_kid = int(np.nonzero(kid_z)[0].max())
        top_liv = int(np.nonzero(liv_z)[0].max())
        if top_kid == z_last and top_liv == z_last:
            q["q_liver_above"] = "n/a"             # both truncated by the field of view
        else:
            q["q_liver_above"] = "pass" if top_liv > top_kid else "fail"

    ck = centroid(kid > 0)
    q["q_kidney_post"] = "n/a" if cv is None or ck is None else ("pass" if ck[1] < cv[1] else "fail")
    return q


def save_overlay(img2d: np.ndarray, m2d: np.ndarray, rk_box, lk_box, path: Path, title: str) -> None:
    rgb = np.repeat(img2d[..., None], 3, axis=2).astype(np.float32)
    for code, colour in ((1, (255, 40, 40)), (2, (40, 220, 40))):
        sel = m2d == code
        rgb[sel] = 0.55 * rgb[sel] + 0.45 * np.array(colour, np.float32)
    im = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    scale = max(1, int(round(512 / max(im.size))))
    if scale > 1:
        im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    d = ImageDraw.Draw(im)
    for box, colour, tag in ((rk_box, (255, 40, 40), "RK"), (lk_box, (40, 220, 40), "LK")):
        if box:
            x0, y0, x1, y1 = (v * scale for v in box)
            d.rectangle([x0, y0, x1 + scale - 1, y1 + scale - 1], outline=colour, width=2)
            d.text((x0 + 2, y0 + 2), tag, fill=colour)
    yellow = (255, 230, 0)
    d.text((im.width // 2 - 3, 2), "A", fill=yellow)
    d.text((2, im.height // 2), "pt-R", fill=yellow)
    d.text((im.width - 30, im.height // 2), "pt-L", fill=yellow)
    d.text((2, im.height - 12), title, fill=yellow)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


# ---------------------------------------------------------------------- one volume
def process_volume(job: dict) -> tuple[list[dict], dict]:
    t0 = time.time()
    rec = {k: job.get(k, "") for k in ("dataset", "patient", "case", "sequence", "modality", "split")}
    rows: list[dict] = []
    out = Path(job["out"])
    case_dir = out / "slices" / job["dataset"] / job["case"]
    try:
        image, kid, liver, affine, extra = LOADERS[job["dataset"]](job)
        rec.update(extra)
        X, Y, Z = image.shape
        sx, sy, sz = voxel_sizes(affine)
        # Display rows run along y, columns along x.
        out_hw, sp_r, sp_c = square_geometry(Y, X, sy, sx)
        sp_row, sp_col, sp_slice = round(sp_r, 4), round(sp_c, 4), round(sz, 4)
        rec.update(shape=f"{X}x{Y}x{Z}", sp_row=sp_row, sp_col=sp_col, sp_slice=sp_slice,
                   img_min=round(float(image.min()), 2), img_max=round(float(image.max()), 2))
        warnings = []
        if max(sx, sy) > 1.2 * sz:
            warnings.append("non-axial acquisition: axial slices are reformatted")
        if out_hw is not None:
            rec["resampled"] = (f"{Y}x{X} -> {out_hw[0]}x{out_hw[1]} px; "
                                f"{sy:.4g}x{sx:.4g} -> {sp_r:.4g}x{sp_c:.4g} mm")

        rec.update(run_qc(image, kid, liver, job["modality"]))
        image_u8 = to_uint8(image, job["modality"])
        del image

        kid_any = per_slice_any(kid)
        if not kid_any.any():
            raise VolumeError("no kidney voxels in the label")
        first = int(np.argmax(kid_any))
        last = int(Z - 1 - np.argmax(kid_any[::-1]))
        if job["full_volume"]:
            lo, hi = 0, Z - 1
        else:
            lo, hi = max(0, first - job["neg_margin"]), min(Z - 1, last + job["neg_margin"])

        case_dir.mkdir(parents=True, exist_ok=True)
        best_z, best_area = first, -1
        for z in range(lo, hi + 1):
            img2d, m2d = make_slice(image_u8, kid, z, out_hw)
            rk, lk = m2d == 1, m2d == 2
            rk_area, lk_area = int(rk.sum()), int(lk.sum())
            rk_box = bbox_xyxy(rk) if rk_area >= job["min_area"] else None
            lk_box = bbox_xyxy(lk) if lk_area >= job["min_area"] else None
            rel = f"slices/{job['dataset']}/{job['case']}/{z:04d}.npz"
            np.savez_compressed(out / rel, img=img2d, msk=m2d)
            rows.append({
                "dataset": job["dataset"], "patient": job["patient"], "case": job["case"],
                "sequence": job["sequence"], "modality": job["modality"], "split": job["split"],
                "z": z, "n_slices": Z, "has_kidney": int(rk_area + lk_area > 0),
                "rk_area": rk_area, "lk_area": lk_area,
                "rk_box": json.dumps(rk_box), "lk_box": json.dumps(lk_box),
                "sam_eligible": int(rk_box is not None or lk_box is not None),
                "h": img2d.shape[0], "w": img2d.shape[1],
                "sp_row": sp_row, "sp_col": sp_col, "sp_slice": sp_slice, "path": rel,
            })
            if rk_area + lk_area > best_area:
                best_z, best_area = z, rk_area + lk_area

        rec.update(n_slices=Z, n_kept=hi - lo + 1, kidney_z_first=first, kidney_z_last=last,
                   n_kidney_slices=int(kid_any.sum()), rk_vox=int((kid == 1).sum()),
                   lk_vox=int((kid == 2).sum()), liver_vox=int(liver.sum()))

        failed = any(rec.get(c) == "fail" for c in QC_CHECKS)
        if job["qc_overlay"] or failed:
            span = last - first
            for z in sorted({best_z, first + span // 4, first + (3 * span) // 4}):
                img2d, m2d = make_slice(image_u8, kid, z, out_hw)
                rk_area, lk_area = int((m2d == 1).sum()), int((m2d == 2).sum())
                save_overlay(
                    img2d, m2d,
                    bbox_xyxy(m2d == 1) if rk_area >= job["min_area"] else None,
                    bbox_xyxy(m2d == 2) if lk_area >= job["min_area"] else None,
                    out / "qc" / job["dataset"] / f"{job['case']}_z{z:04d}.png",
                    f"{job['case']} z={z} {'QC-FAIL' if failed else ''}".strip())
        rec["warnings"] = "; ".join(warnings)
    except Exception as exc:                     # recorded, reported, never silent
        if isinstance(exc, VolumeError):
            rec["error"] = str(exc)
        else:
            tb = traceback.format_exc(limit=4).strip().splitlines()
            rec["error"] = f"{type(exc).__name__}: {exc} | {' / '.join(tb[-3:])}"
        rows = []
        shutil.rmtree(case_dir, ignore_errors=True)   # no orphan slices for a failed volume
    rec["seconds"] = round(time.time() - t0, 1)
    return rows, rec


# ------------------------------------------------------------------------ job lists
def assign_amos_splits(pool: list[dict], seed: int) -> None:
    """D14: whole patients, per modality, seeded 60/20/20. Mutates job['split']."""
    for modality in ("ct", "mri"):
        subset = sorted((j for j in pool if j["modality"] == modality), key=lambda j: j["case"])
        random.Random(seed).shuffle(subset)
        n = len(subset)
        n_train = round(n * SPLIT_FRACTIONS[0])
        n_val = round(n * SPLIT_FRACTIONS[1])
        for i, job in enumerate(subset):
            job["split"] = "train" if i < n_train else "val" if i < n_train + n_val else "test"


def amos_jobs(root: Path, seed: int) -> tuple[list[dict], list[str]]:
    """Returns (jobs, excluded case names). Exclusions (D18) leave the pool before the
    split, so they shift the split counts instead of leaving holes in them."""
    pool, seen = [], set()
    for img_dir, lbl_dir in (("imagesTr", "labelsTr"), ("imagesVa", "labelsVa")):
        if not (root / img_dir).is_dir():
            raise SystemExit(f"AMOS: missing {root / img_dir}")
        for img in sorted(p for p in (root / img_dir).glob("*.nii.gz") if not p.name.startswith(".")):
            lbl = root / lbl_dir / img.name
            if not lbl.exists():
                raise SystemExit(f"AMOS: missing label {lbl}")
            cid = case_number(img)
            case = f"amos_{cid:04d}"
            if case in seen:
                raise SystemExit(f"AMOS: duplicate case {case}")
            seen.add(case)
            pool.append({"dataset": "amos", "case": case, "patient": case, "sequence": "",
                         "modality": "mri" if cid >= AMOS_MRI_MIN_ID else "ct",
                         "image": str(img), "label": str(lbl)})
    excluded = sorted(j["case"] for j in pool if j["case"] in AMOS_EXCLUDE)
    pool = [j for j in pool if j["case"] not in AMOS_EXCLUDE]
    assign_amos_splits(pool, seed)
    return pool, excluded


def word_jobs(root: Path, test_labels: Path | None) -> list[dict]:
    """External CT test = WORD imagesTr (D15). test_labels is only checked for
    presence so a mis-pointed --word-root fails loudly instead of yielding nothing."""
    img_dir, lbl_dir = root / "imagesTr", root / "labelsTr"
    if not img_dir.is_dir() or not lbl_dir.is_dir():
        raise SystemExit(f"WORD: missing imagesTr/labelsTr under {root}")
    jobs = []
    for img in sorted(p for p in img_dir.glob("*.nii.gz") if not p.name.startswith(".")):
        lbl = lbl_dir / img.name
        if not lbl.exists():
            raise SystemExit(f"WORD: missing label {lbl}")
        case = f"word_{case_number(img):04d}"
        jobs.append({"dataset": "word", "case": case, "patient": case, "sequence": "",
                     "modality": "ct", "split": "external", "image": str(img), "label": str(lbl)})
    if test_labels is not None and not test_labels.is_dir():
        raise SystemExit(f"WORD: --word-test-labels {test_labels} does not exist")
    return jobs


def chaos_jobs(root: Path, png_order: str) -> list[dict]:
    mr = root / "MR"
    if not mr.is_dir():
        raise SystemExit(f"CHAOS: missing {mr} (point --chaos-root at Train_Sets)")
    jobs = []
    for pdir in sorted((p for p in mr.iterdir() if p.is_dir() and p.name.isdigit()),
                       key=lambda p: int(p.name)):
        pid = int(pdir.name)
        for seq in CHAOS_SEQUENCES:
            jobs.append({"dataset": "chaos", "case": f"chaos_{pid:02d}_{seq}",
                         "patient": f"chaos_{pid:02d}", "sequence": seq, "modality": "mri",
                         "split": "external", "patient_dir": str(pdir), "png_order": png_order})
    return jobs


# ----------------------------------------------------------------------------- main
def prepare_out(out: Path, overwrite: bool) -> None:
    ours = [out / "slices", out / "qc", out / "manifest.csv", out / "volumes.csv", out / "run.json"]
    existing = [p for p in ours if p.exists()]
    if existing and not overwrite:
        raise SystemExit(f"{out} already holds a previous run ({existing[0].name}). "
                         "Re-run with --overwrite to replace it, or pick a new --out.")
    for p in existing:                           # only ever our own outputs
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    out.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--amos-root", type=Path, help="amos22/amos22 (holds imagesTr, imagesVa)")
    p.add_argument("--word-root", type=Path, help="word/WORD-V0.1.0 (holds imagesTr)")
    p.add_argument("--word-test-labels", type=Path, help="word/labelsTs (presence check only)")
    p.add_argument("--chaos-root", type=Path, help="chaos/Train_Sets (holds MR/)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--min-area", type=int, default=50, help="sliver threshold in px: no box below it")
    p.add_argument("--neg-margin", type=int, default=15,
                   help="kidney-free slices kept beyond each kidney end (band splits)")
    p.add_argument("--full-volume-splits", default="test,external",
                   help="splits stored as whole volumes, so students are scored on every slice")
    p.add_argument("--chaos-png-order", choices=("instance", "reverse"), default="instance",
                   help="only used when CHAOS PNG names match no DICOM file names")
    p.add_argument("--seed", type=int, default=643)
    p.add_argument("--limit", type=int, default=None,
                   help="max volumes per dataset/split/modality/sequence (smoke tests)")
    p.add_argument("--qc-volumes", type=int, default=3,
                   help="overlay PNGs for this many volumes per dataset/modality/sequence")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    if not (args.amos_root or args.word_root or args.chaos_root):
        p.error("give at least one of --amos-root / --word-root / --chaos-root")

    jobs: list[dict] = []
    amos_excluded: list[str] = []
    if args.amos_root:
        amos_pool, amos_excluded = amos_jobs(args.amos_root, args.seed)
        jobs += amos_pool
    if args.word_root:
        jobs += word_jobs(args.word_root, args.word_test_labels)
    if args.chaos_root:
        jobs += chaos_jobs(args.chaos_root, args.chaos_png_order)

    def group_key(j):
        return (j["dataset"], j["split"], j["modality"], j["sequence"])

    if args.limit:                               # applied AFTER split assignment
        kept, count = [], {}
        for j in sorted(jobs, key=lambda j: (group_key(j), j["case"])):
            if count.get(group_key(j), 0) < args.limit:
                count[group_key(j)] = count.get(group_key(j), 0) + 1
                kept.append(j)
        jobs = kept

    full = {s.strip() for s in args.full_volume_splits.split(",") if s.strip()}
    overlay_count: dict = {}
    for j in sorted(jobs, key=lambda j: (j["dataset"], j["modality"], j["sequence"], j["case"])):
        k = (j["dataset"], j["modality"], j["sequence"])
        overlay_count[k] = overlay_count.get(k, 0) + 1
        j.update(out=str(args.out), min_area=args.min_area, neg_margin=args.neg_margin,
                 full_volume=j["split"] in full, qc_overlay=overlay_count[k] <= args.qc_volumes)

    prepare_out(args.out, args.overwrite)
    print(f"{len(jobs)} volumes -> {args.out}")
    for case in amos_excluded:
        print(f"  excluded {case}: {AMOS_EXCLUDE[case]}")
    plan: dict = {}
    for j in jobs:
        plan[group_key(j)] = plan.get(group_key(j), 0) + 1
    for k in sorted(plan):
        print(f"  {'/'.join(x for x in k if x):<28}{plan[k]:>5} volumes")

    started = time.strftime("%Y-%m-%d %H:%M:%S")
    rows: list[dict] = []
    recs: list[dict] = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process_volume, j) for j in jobs]
            for fut in tqdm(as_completed(futures), total=len(futures), unit="vol"):
                r, rec = fut.result()
                rows += r
                recs.append(rec)
    else:
        for j in tqdm(jobs, unit="vol"):
            r, rec = process_volume(j)
            rows += r
            recs.append(rec)

    rows.sort(key=lambda r: (r["dataset"], r["case"], r["z"]))
    recs.sort(key=lambda r: (r["dataset"], r["case"]))
    write_csv(args.out / "manifest.csv", MANIFEST_FIELDS, rows)
    write_csv(args.out / "volumes.csv", VOLUME_FIELDS, recs)

    # ---- summary
    print(f"\nwrote {len(rows)} slices from {len(recs)} volumes")
    print(f"{'dataset/split/modality/seq':<30}{'vols':>5}{'slices':>8}{'kidney':>8}{'boxed':>7}")
    groups: dict = {}
    for r in rows:
        groups.setdefault((r["dataset"], r["split"], r["modality"], r["sequence"]), []).append(r)
    for k in sorted(groups):
        g = groups[k]
        print(f"{'/'.join(x for x in k if x):<30}{len({r['case'] for r in g}):>5}{len(g):>8}"
              f"{sum(r['has_kidney'] for r in g):>8}{sum(r['sam_eligible'] for r in g):>7}")

    problems = []
    # D10: a patient in two splits invalidates every number.
    split_of: dict = {}
    for r in rows:
        split_of.setdefault(r["patient"], set()).add(r["split"])
    leaks = sorted(pt for pt, s in split_of.items() if len(s) > 1)
    if leaks:
        problems.append(f"PATIENT LEAK across splits: {leaks[:10]}")

    errors = [r for r in recs if r.get("error")]
    for r in errors:
        problems.append(f"ERROR {r['case']}: {r['error']}")

    print(f"\n{'QC check':<16}" + "".join(f"{d:>26}" for d in sorted({r['dataset'] for r in recs})))
    for check in QC_CHECKS:
        line = f"{check:<16}"
        for d in sorted({r["dataset"] for r in recs}):
            ok = [r for r in recs if r["dataset"] == d and not r.get("error")]
            n_pass = sum(r.get(check) == "pass" for r in ok)
            n_fail = sum(r.get(check) == "fail" for r in ok)
            n_na = sum(r.get(check) == "n/a" for r in ok)
            line += f"{f'{n_pass} ok / {n_fail} FAIL / {n_na} n/a':>26}"
            if ok and n_fail / len(ok) > SYSTEMATIC_FAIL_RATE:
                problems.append(f"SYSTEMATIC {check} failure on {d} ({n_fail}/{len(ok)}): "
                                "pipeline bug, do NOT use this output")
        print(line)
    for r in recs:
        failed = [c for c in QC_CHECKS if r.get(c) == "fail"]
        if failed and not r.get("error"):
            problems.append(f"QC FAIL {r['case']}: {', '.join(failed)}  (see qc/{r['dataset']}/)")
    for r in recs:
        if r.get("warnings"):
            print(f"  warning {r['case']}: {r['warnings']}")

    import PIL  # noqa: E402  (version only)
    with open(args.out / "run.json", "w") as fh:
        json.dump({
            "args": {k: str(v) for k, v in vars(args).items()},
            "started": started, "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
            "host": platform.node(), "python": platform.python_version(),
            "versions": {"numpy": np.__version__, "nibabel": nib.__version__,
                         "pydicom": pydicom.__version__, "pillow": PIL.__version__},
            "nifti_labels": NIFTI_LABELS, "chaos_grey": CHAOS_GREY,
            "ct_window": CT_WINDOW, "mri_percentiles": MRI_PERCENTILES,
            "amos_mri_min_id": AMOS_MRI_MIN_ID, "split_fractions": SPLIT_FRACTIONS,
            "amos_excluded": {c: AMOS_EXCLUDE[c] for c in amos_excluded},
            "mask_codes": {"0": "background", "1": "right kidney", "2": "left kidney"},
            "slice_view": "radiological: anterior at top, patient's right on image left",
            "box_format": "XYXY pixel coords of the stored slice, inclusive, tight",
            "n_slices": len(rows), "n_volumes": len(recs), "problems": problems,
        }, fh, indent=2)

    if problems:
        print("\nPROBLEMS:")
        for msg in problems:
            print(f"  !! {msg}")
        print("\nPREP_FAILED")
        return 1
    print("\nno patient leak, no errors, all QC checks passed\nPREP_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
