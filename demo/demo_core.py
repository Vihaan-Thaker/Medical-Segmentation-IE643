"""Demo core (DEMO_PLAN.md): load a test-set scan + its ground truth, run the models, score them.

No Gradio here, so every function is testable. Everything that defines a result is IMPORTED
from the project code that produced the reported numbers, never re-implemented:
  prep_data   load_nifti / load_chaos, voxel_sizes, square_geometry, to_uint8, make_slice,
              bbox_xyxy (the exact slices and boxes of the evaluation data)
  sam_metrics jitter_box / jitter_seed (the evaluation's seeded box noise), score_masks
  student_core load_student, net_input, predict_native_batch (the students' only path)
  eval_sam    build_predictor (SAM exactly as evaluated: set_image once per slice, then
              predict(box, multimask_output=False) per kidney)
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DEMO_DIR = Path(__file__).resolve().parent
CODE_DIR = DEMO_DIR.parent
WEIGHTS = DEMO_DIR / "weights"
SAMPLES = DEMO_DIR / "samples"
sys.path.insert(0, str(CODE_DIR))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 8))   # read by eval_sam.build_predictor

import prep_data as P  # noqa: E402
import sam_metrics as M  # noqa: E402

MIN_AREA = 50                       # prep_data --min-area: a kidney gets a box iff it covers >= 50 px
SIDES = ((M.RIGHT, "right"), (M.LEFT, "left"))
NOISE_LEVELS = M.JITTER_LEVELS      # 0, 5, 10, 20 % of the box side
PRED_BATCH = 16


class DemoError(Exception):
    """A user-facing problem with the input (shown in the app, never a crash)."""


# ----------------------------------------------------------------------------- the case
@dataclass
class Case:
    name: str                       # evaluation case id; seeds the box noise exactly as in testing
    dataset: str                    # amos | word | chaos
    modality: str                   # ct | mri
    sequence: str                   # "" | t1in | t1out | t2
    img: np.ndarray                 # Z x h x w uint8, the stored (radiological) view
    msk: np.ndarray                 # Z x h x w uint8, 0 bg / 1 right / 2 left
    sp_row: float
    sp_col: float
    sp_slice: float
    affine: np.ndarray              # canonical (RAS) affine of the scan
    canon_shape: tuple              # (X, Y, Z) of the canonical volume
    out_hw: tuple | None            # square-pixel resample applied by make_slice, or None
    boxes: list = field(default_factory=list)   # per z: {side: clean box | None}

    @property
    def label(self) -> str:
        seq = f" {self.sequence}" if self.sequence else ""
        return f"{self.name} ({self.dataset.upper()} {self.modality.upper()}{seq})"

    @property
    def voxel_ml(self) -> float:
        return self.sp_row * self.sp_col * self.sp_slice / 1000.0

    def n_instances(self) -> int:
        return sum(b is not None for bz in self.boxes for b in bz.values())


def _case_from_volume(name, dataset, modality, sequence, image, kid, affine) -> Case:
    """prep_data.process_volume's slicing, for every slice (test volumes are stored whole)."""
    X, Y, Z = image.shape
    sx, sy, sz = P.voxel_sizes(affine)
    out_hw, sp_r, sp_c = P.square_geometry(Y, X, sy, sx)
    image_u8 = P.to_uint8(image, modality)
    imgs, msks, boxes = [], [], []
    for z in range(Z):
        img2d, m2d = P.make_slice(image_u8, kid, z, out_hw)
        imgs.append(img2d)
        msks.append(m2d)
        bz = {}
        for side, _ in SIDES:
            bz[side] = P.bbox_xyxy(m2d == side) if int((m2d == side).sum()) >= MIN_AREA else None
        boxes.append(bz)
    if not any(b is not None for bz in boxes for b in bz.values()):
        raise DemoError("the ground-truth mask contains no kidney of at least 50 pixels on any slice")
    return Case(name=name, dataset=dataset, modality=modality, sequence=sequence,
                img=np.stack(imgs), msk=np.stack(msks), sp_row=round(sp_r, 4), sp_col=round(sp_c, 4),
                sp_slice=round(sz, 4), affine=affine, canon_shape=(X, Y, Z), out_hw=out_hw, boxes=boxes)


def case_name_from(path: Path) -> str:
    n = Path(path).name
    for ext in (".nii.gz", ".nii"):
        if n.endswith(ext):
            return n[: -len(ext)]
    raise DemoError(f"{n}: expected a NIfTI file (.nii or .nii.gz)")


def load_nifti_case(image_path, label_path, dataset: str) -> Case:
    """An AMOS or WORD test scan + its label file, exactly as stored in the dataset."""
    if dataset not in P.NIFTI_LABELS:
        raise DemoError(f"unknown dataset {dataset!r}; choose AMOS or WORD")
    name = case_name_from(Path(image_path))
    if case_name_from(Path(label_path)) != name:
        raise DemoError(f"image {name} and label {case_name_from(Path(label_path))} are different cases")
    if dataset == "amos":
        modality = "mri" if P.case_number(Path(image_path)) >= P.AMOS_MRI_MIN_ID else "ct"
    else:
        modality = "ct"
    try:
        image, kid, _, affine, _ = P.load_nifti({"image": str(image_path), "label": str(label_path),
                                                 "dataset": dataset})
    except P.VolumeError as e:
        raise DemoError(str(e)) from e
    except Exception as e:                                   # unreadable file
        raise DemoError(f"could not read the scan: {type(e).__name__}: {e}") from e
    return _case_from_volume(name, dataset, modality, "", image, kid, affine)


def load_chaos_case(patient_dir, sequence: str) -> Case:
    """A CHAOS MR patient folder (DICOM + label PNGs) for one sequence, via prep_data.load_chaos."""
    pdir = Path(patient_dir)
    m = re.fullmatch(r"\d+", pdir.name)
    if not m or sequence not in P.CHAOS_SEQUENCES:
        raise DemoError(f"not a CHAOS patient folder / sequence: {pdir.name} {sequence}")
    try:
        image, kid, _, affine, _ = P.load_chaos({"patient_dir": str(pdir), "sequence": sequence,
                                                 "png_order": "instance"})
    except P.VolumeError as e:
        raise DemoError(str(e)) from e
    return _case_from_volume(f"chaos_{int(pdir.name):02d}_{sequence}", "chaos", "mri", sequence,
                             image, kid, affine)


# ---------------------------------------------------------------------------- samples
def sample_catalogue() -> dict:
    """{label: loader} for the test cases copied into demo/samples (found by glob)."""
    out = {}
    for img in sorted(SAMPLES.glob("amos22/amos22/images??/amos_*.nii.gz"), key=lambda p: P.case_number(p) >= P.AMOS_MRI_MIN_ID):
        lbl = img.parent.parent / img.parent.name.replace("images", "labels") / img.name
        if lbl.exists():
            kind = "MRI" if P.case_number(img) >= P.AMOS_MRI_MIN_ID else "CT"
            out[f"AMOS {kind} test: {case_name_from(img)}"] = ("nifti", str(img), str(lbl), "amos")
    for img in sorted(SAMPLES.glob("word/*/imagesTr/word_*.nii.gz")):
        lbl = img.parent.parent / "labelsTr" / img.name
        if lbl.exists():
            out[f"WORD CT (external): {case_name_from(img)}"] = ("nifti", str(img), str(lbl), "word")
    names = {"t2": "T2-SPIR", "t1in": "T1 in-phase", "t1out": "T1 out-phase"}
    for pdir in sorted(SAMPLES.glob("chaos/Train_Sets/MR/*"), key=lambda p: int(p.name) if p.name.isdigit() else 0):
        if pdir.is_dir() and pdir.name.isdigit():
            for seq in ("t2", "t1in", "t1out"):
                out[f"CHAOS MRI (external) patient {int(pdir.name):02d}: {names[seq]}"] = ("chaos", str(pdir), seq, "chaos")
    return out


def load_sample(entry) -> Case:
    kind = entry[0]
    if kind == "nifti":
        return load_nifti_case(entry[1], entry[2], entry[3])
    return load_chaos_case(entry[1], entry[2])


# ----------------------------------------------------------------------------- models
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


@dataclass
class ModelSpec:
    key: str
    title: str
    kind: str                       # student | sam
    path: Path
    needs_box: bool
    variant: str = "plain"          # student variant
    sam_type: str = "vit_b"


def model_catalogue() -> dict:
    """Every model whose weights are in demo/weights; arm 4 appears once its files are copied in."""
    specs = {}
    sam = {"arm1b": ("Arm 1b: SAM ViT-B, off the shelf (+ box)", "sam_vit_b_01ec64.pth"),
           "arm2": ("Arm 2: SAM ViT-B fine-tuned with LoRA, the teacher (+ box)", "sam_vit_b_arm2_r4.pth")}
    for key, (title, fname) in sam.items():
        if (WEIGHTS / fname).exists():
            specs[key] = ModelSpec(key, title, "sam", WEIGHTS / fname, True)
    for p in sorted(WEIGHTS.glob("arm3_plain_s*.pt")):
        seed = p.stem.split("_s")[-1]
        specs[f"arm3_s{seed}"] = ModelSpec(f"arm3_s{seed}", f"Arm 3: U-Net from scratch, no prompt (seed {seed})",
                                           "student", p, False)
    for p in sorted(WEIGHTS.glob("arm3_box_s*.pt")):
        seed = p.stem.split("_s")[-1]
        specs[f"arm3box_s{seed}"] = ModelSpec(f"arm3box_s{seed}", f"Arm 3 ablation: U-Net + box (seed {seed})",
                                              "student", p, True, variant="box")
    for p in sorted(WEIGHTS.glob("arm4_plain_s*.pt")):
        seed = p.stem.split("_s")[-1]
        specs[f"arm4_s{seed}"] = ModelSpec(f"arm4_s{seed}", f"Arm 4: U-Net distilled from arm 2, no prompt (seed {seed})",
                                           "student", p, False)
    return specs


def verify_weights(spec: ModelSpec) -> str:
    """A student checkpoint is checked against the sha256 its training run recorded."""
    rj = spec.path.with_name(spec.path.stem + ".run.json")
    if spec.kind != "student" or not rj.exists():
        return "not checked"
    want = json.loads(rj.read_text()).get("best_sha256")
    return "verified" if sha256(spec.path) == want else "MISMATCH with run.json"


def pick_device(prefer: str = "auto") -> str:
    import torch
    if prefer != "auto":
        return prefer
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


_CACHE: dict = {}


def _load(spec: ModelSpec, device: str):
    key = (spec.key, device)
    if key not in _CACHE:
        if spec.kind == "student":
            import student_core as C
            model, norm, variant, size, _ = C.load_student(spec.path, device)
            if variant != spec.variant:
                raise DemoError(f"{spec.path.name} is a {variant} student, expected {spec.variant}")
            _CACHE[key] = (model, norm, size)
        else:
            import eval_sam as E
            _CACHE[key] = E.build_predictor(spec.path, spec.sam_type, device)
    return _CACHE[key]


def eval_boxes(case: Case, z: int, level: float) -> dict:
    """{side: jittered box | None}: the evaluation's own boxes (eval_student.eval_boxes)."""
    h, w = case.img.shape[1:]
    return {side: (None if case.boxes[z][side] is None else
                   M.jitter_box(case.boxes[z][side], level, h, w, M.jitter_seed(case.name, z, side, level)))
            for side, _ in SIDES}


@dataclass
class Prediction:
    spec: ModelSpec
    level: float | None             # box noise used (None for prompt-free models)
    side_masks: dict                # side -> Z x h x w bool
    used_boxes: dict                # z -> {side: box} actually given to the model
    seconds: float
    n_forward: int                  # slices passed through the network

    def labels(self) -> np.ndarray:
        lab = np.zeros(self.side_masks[M.RIGHT].shape, np.uint8)
        lab[self.side_masks[M.RIGHT]] = 1
        lab[self.side_masks[M.LEFT]] = 2
        return lab


def predict(spec: ModelSpec, case: Case, level: float = M.HEADLINE_JITTER, device: str = "auto",
            progress=None) -> Prediction:
    import torch
    device = pick_device(device)
    obj = _load(spec, device)
    Z, h, w = case.img.shape
    masks = {side: np.zeros((Z, h, w), bool) for side, _ in SIDES}
    used: dict = {}
    t0 = time.perf_counter()
    if spec.kind == "student":
        import student_core as C
        model, norm, size = obj
        if spec.needs_box:          # box student: only on slices that have a box, as in the evaluation
            zs = [z for z in range(Z) if any(b is not None for b in case.boxes[z].values())]
        else:
            zs = list(range(Z))
        for a in range(0, len(zs), PRED_BATCH):
            chunk = zs[a:a + PRED_BATCH]
            xs, gs = [], []
            for z in chunk:
                bx = None
                if spec.needs_box:
                    bx = eval_boxes(case, z, level)
                    used[z] = {s: b for s, b in bx.items() if b is not None}
                x, _, g = C.net_input(case.img[z], norm, spec.variant, bx, size)
                xs.append(x)
                gs.append(g)
            with torch.no_grad():
                for z, (lab, _) in zip(chunk, C.predict_native_batch(model, xs, gs, device)):
                    for side, _ in SIDES:
                        masks[side][z] = lab == side
            if progress:
                progress(min(1.0, (a + len(chunk)) / max(1, len(zs))))
        n = len(zs)
    else:
        predictor = obj
        zs = [z for z in range(Z) if any(b is not None for b in case.boxes[z].values())]
        for k, z in enumerate(zs):
            predictor.set_image(np.repeat(case.img[z][..., None], 3, axis=2), image_format="RGB")
            bx = eval_boxes(case, z, level)
            used[z] = {}
            for side, _ in SIDES:
                if bx[side] is None:
                    continue
                out, _, _ = predictor.predict(box=np.array(bx[side], dtype=float), multimask_output=False)
                masks[side][z] = np.asarray(out[0], dtype=bool)
                used[z][side] = bx[side]
            if progress:
                progress((k + 1) / max(1, len(zs)))
        n = len(zs)
    if device == "mps":
        torch.mps.synchronize()
    return Prediction(spec, level if spec.needs_box else None, masks, used, time.perf_counter() - t0, n)


# ----------------------------------------------------------------------------- scoring
def score(pred: Prediction, case: Case) -> dict:
    """The evaluation's scoring: one row per kidney on a slice with a box (>= 50 px), Dice and
    NSD from sam_metrics.score_masks at (sp_row, sp_col); the case score is the instance mean
    (= the patient mean the reported tables use, for one case)."""
    rows = []
    for z, bz in enumerate(case.boxes):
        for side, name in SIDES:
            if bz[side] is None:
                continue
            gt = case.msk[z] == side
            s = M.score_masks(gt, pred.side_masks[side][z], (case.sp_row, case.sp_col))
            rows.append({"z": z, "side": name, "dice": s["dice"], "nsd_3": s["nsd_3"],
                         "gt_area": s["gt_vox"], "pred_area": s["pred_vox"]})
    out = {"instances": rows}
    for key in ("dice", "nsd_3"):
        out[key] = float(np.mean([r[key] for r in rows])) if rows else float("nan")
    for side, name in SIDES:
        out[f"vol_pred_{name}"] = float(pred.side_masks[side].sum()) * case.voxel_ml
        out[f"vol_gt_{name}"] = float((case.msk == side).sum()) * case.voxel_ml
    return out


# ------------------------------------------------------------------------------ export
def to_canonical(labels: np.ndarray, case: Case) -> np.ndarray:
    """Stored view (Z x h x w) -> canonical volume (X x Y x Z): undo make_slice."""
    from PIL import Image
    X, Y, Z = case.canon_shape
    vol = np.zeros((X, Y, Z), np.uint8)
    for z in range(Z):
        s = labels[z]
        if case.out_hw is not None:
            s = np.array(Image.fromarray(s).resize((X, Y), Image.NEAREST))
        vol[:, :, z] = s[::-1, ::-1].T          # inverse of prep_data.to_display
    return vol


def save_nifti(labels: np.ndarray, case: Case, path: Path) -> Path:
    import nibabel as nib
    img = nib.Nifti1Image(to_canonical(labels, case), case.affine)
    img.set_data_dtype(np.uint8)
    nib.save(img, str(path))
    return Path(path)


# ----------------------------------------------------------------------------- viewing
COLORS = {M.RIGHT: (232, 89, 60), M.LEFT: (66, 133, 244)}       # right kidney / left kidney


def _edge(m: np.ndarray) -> np.ndarray:
    e = m.copy()
    e[1:-1, 1:-1] = m[1:-1, 1:-1] & ~(m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:])
    return e


def _font(size: int):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                                         # Pillow < 10.1
        return ImageFont.load_default()


def kidney_crop(case: Case, margin: float = 0.35) -> tuple:
    """A square (r0, c0, r1, c1) around every ground-truth kidney pixel of the scan, with a
    margin, so the same crop is used on every slice and the slider does not jump."""
    _, h, w = case.img.shape
    rows, cols = np.nonzero((case.msk > 0).any(0))
    if rows.size == 0:
        return 0, 0, h, w
    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    side = int(max(r1 - r0, c1 - c0) * (1 + 2 * margin))
    side = min(max(side, 64), h, w)
    rc, cc = (r0 + r1) // 2, (c0 + c1) // 2
    r0 = int(np.clip(rc - side // 2, 0, h - side))
    c0 = int(np.clip(cc - side // 2, 0, w - side))
    return r0, c0, r0 + side, c0 + side


def slice_dice(case: Case, z: int, pred: Prediction) -> str:
    out = []
    for side, name in SIDES:
        g, p = case.msk[z] == side, pred.side_masks[side][z]
        if g.sum() + p.sum() == 0:
            continue
        out.append(f"{name[0].upper()} {2 * (g & p).sum() / (g.sum() + p.sum()):.2f}")
    return ("slice Dice " + ", ".join(out)) if out else "no kidney on this slice"


def render_panel(case: Case, z: int, pred: Prediction | None, title: str, size: int = 420,
                 crop: tuple | None = None):
    """One panel: the slice, the prediction filled (right orange, left blue), the ground-truth
    outline in yellow, and the boxes the model was given."""
    from PIL import Image, ImageDraw
    img = case.img[z]
    rgb = np.repeat(img[..., None], 3, axis=2).astype(np.float32)
    if pred is not None:
        for side, _ in SIDES:
            m = pred.side_masks[side][z]
            rgb[m] = 0.45 * rgb[m] + 0.55 * np.array(COLORS[side], np.float32)
    for side, _ in SIDES:
        rgb[_edge(case.msk[z] == side)] = (250, 220, 60)
    r0, c0 = 0, 0
    if crop is not None:
        r0, c0, r1, c1 = crop
        rgb = rgb[r0:r1, c0:c1]
    im = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    scale = size / max(im.size)
    im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.NEAREST)
    d = ImageDraw.Draw(im)
    if pred is not None:
        for side, b in pred.used_boxes.get(z, {}).items():
            x0, y0, x1, y1 = ((b[0] - c0) * scale, (b[1] - r0) * scale, (b[2] - c0) * scale, (b[3] - r0) * scale)
            d.rectangle([x0, y0, x1 + scale - 1, y1 + scale - 1], outline=COLORS[side], width=2)
    head = 54
    canvas = Image.new("RGB", (size, size + head), (24, 28, 36))
    canvas.paste(im, ((size - im.width) // 2, head + (size - im.height) // 2))
    d = ImageDraw.Draw(canvas)
    d.text((8, 6), title, fill=(240, 240, 240), font=_font(15))
    d.text((8, 30), slice_dice(case, z, pred) if pred is not None else f"slice {z} of {case.img.shape[0] - 1}",
           fill=(180, 190, 205), font=_font(14))
    return canvas


def render_grid(case: Case, z: int, preds: list, size: int = 420, zoom: bool = True):
    from PIL import Image
    crop = kidney_crop(case) if zoom else None
    panels = [render_panel(case, z, None, "Scan + ground truth", size, crop)]
    panels += [render_panel(case, z, p, short_title(p), size, crop) for p in preds]
    cols = min(3, len(panels))
    rows = (len(panels) + cols - 1) // cols
    ph = panels[0].height
    grid = Image.new("RGB", (cols * size, rows * ph), (24, 28, 36))
    for k, p in enumerate(panels):
        grid.paste(p, ((k % cols) * size, (k // cols) * ph))
    return grid


def short_title(p: Prediction) -> str:
    t = p.spec.title.split(":")[0]
    extra = {"arm1b": "SAM off the shelf", "arm2": "SAM fine-tuned"}.get(p.spec.key)
    if extra is None:
        extra = "U-Net + box" if p.spec.needs_box else ("U-Net distilled" if p.spec.key.startswith("arm4") else "U-Net")
    noise = f", box noise {int(round(p.level * 100))}%" if p.level is not None else ", no prompt"
    return f"{t}: {extra}{noise}"


def best_slice(case: Case) -> int:
    return int(np.argmax((case.msk > 0).reshape(case.msk.shape[0], -1).sum(1)))
