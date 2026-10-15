#!/usr/bin/env python
"""End-to-end test of prep_data.py on synthetic AMOS / WORD / CHAOS data.

Anatomy is defined once in patient (RAS, mm) coordinates and rasterised into every
file through that file's own affine, so ground truth is independent of the code
under test. Files are stored in deliberately awkward orientations (flipped, permuted,
DICOM LPS with superior-first slices) to prove the canonicalisation, the radiological
view, the label maps, the CHAOS DICOM<->PNG pairing and every QC check.

    python test_prep_data.py            # exits 0 and prints ALL TESTS PASSED
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
from PIL import Image
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prep_data  # noqa: E402

# ------------------------------------------------------------- synthetic anatomy (RAS mm)
ORGANS = {                       # name: (centre, radii) of an ellipsoid
    "liver": ((60.0, 20.0, 60.0), (60.0, 50.0, 50.0)),
    "rk": ((55.0, -40.0, 10.0), (25.0, 20.0, 45.0)),
    "lk": ((-55.0, -40.0, 15.0), (25.0, 20.0, 45.0)),
    "spleen": ((-70.0, 5.0, 50.0), (25.0, 25.0, 35.0)),
}
HALF = np.array([160.0, 160.0, 90.0])    # field of view half-extent per world axis
SPACING = np.array([4.0, 4.0, 5.0])      # mm per voxel along world x, y, z


def organ_map(world: np.ndarray) -> dict[str, np.ndarray]:
    """world: [..., 3] RAS mm -> boolean masks (kidneys win over liver/spleen overlaps)."""
    out = {}
    for name, (c, r) in ORGANS.items():
        out[name] = (((world - np.array(c)) / np.array(r)) ** 2).sum(-1) <= 1.0
    body = (world[..., 0] / 140.0) ** 2 + (world[..., 1] / 100.0) ** 2 <= 1.0
    out["body"] = body
    out["liver"] &= ~(out["rk"] | out["lk"])
    out["spleen"] &= ~(out["rk"] | out["lk"] | out["liver"])
    return out


def intensities(m: dict, modality: str, rng: np.random.Generator) -> np.ndarray:
    if modality == "ct":
        vals = {"bg": -1000.0, "body": 40.0, "liver": 60.0, "spleen": 50.0, "kid": 150.0}
    else:
        vals = {"bg": 0.0, "body": 300.0, "liver": 500.0, "spleen": 450.0, "kid": 800.0}
    img = np.full(m["body"].shape, vals["bg"], np.float32)
    img[m["body"]] = vals["body"]
    img[m["liver"]] = vals["liver"]
    img[m["spleen"]] = vals["spleen"]
    img[m["rk"] | m["lk"]] = vals["kid"]
    return img + rng.normal(0, 3, img.shape).astype(np.float32)


def make_affine(perm: tuple[int, int, int], signs: tuple[int, int, int], spacing=SPACING):
    """Voxel axis a runs along world axis perm[a] in direction signs[a]."""
    aff = np.eye(4)
    shape = []
    for a, (w, s) in enumerate(zip(perm, signs)):
        aff[:3, a] = 0.0
        aff[w, a] = s * spacing[w]
        aff[w, 3] = -s * (HALF[w] - spacing[w] / 2)
        shape.append(int(round(2 * HALF[w] / spacing[w])))
    return aff, tuple(shape)


ORIENTATIONS = {                  # name: (perm, signs)
    "RAS": ((0, 1, 2), (1, 1, 1)),
    "LAS": ((0, 1, 2), (-1, 1, 1)),
    "LPS": ((0, 1, 2), (-1, -1, 1)),
    "PERM": ((1, 0, 2), (1, -1, 1)),
    "ZFIRST": ((2, 0, 1), (-1, 1, 1)),
}


def write_nifti_case(img_path: Path, lbl_path: Path, orient: str, modality: str,
                     codes: dict, rng: np.random.Generator, spacing=SPACING) -> None:
    aff, shape = make_affine(*ORIENTATIONS[orient], spacing=spacing)
    idx = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), -1).astype(float)
    world = idx @ aff[:3, :3].T + aff[:3, 3]
    m = organ_map(world)
    lab = np.zeros(shape, np.uint8)
    for organ, code in codes.items():
        lab[m[organ]] = code
    img_dtype = np.int16 if modality == "ct" else np.float32
    nib.save(nib.Nifti1Image(intensities(m, modality, rng).astype(img_dtype), aff), str(img_path))
    nib.save(nib.Nifti1Image(lab, aff), str(lbl_path))


def build_amos(root: Path, rng) -> None:
    codes = {"rk": 2, "lk": 3, "liver": 6, "spleen": 1}
    orients = list(ORIENTATIONS)
    # amos_0085 is on prep_data.AMOS_EXCLUDE: it must leave the pool before the split.
    cases = [(i, "ct") for i in range(1, 11)] + [(85, "ct")] + [(i, "mri") for i in range(501, 506)]
    va = {3, 7, 9, 503, 505}                      # these go to imagesVa
    for d in ("imagesTr", "labelsTr", "imagesVa", "labelsVa"):
        (root / d).mkdir(parents=True)
    for n, (cid, mod) in enumerate(cases):
        sub = "Va" if cid in va else "Tr"
        name = f"amos_{cid:04d}.nii.gz"
        write_nifti_case(root / f"images{sub}" / name, root / f"labels{sub}" / name,
                         orients[n % len(orients)], mod, codes, rng)
    # Coronal-style MRI (like the 18 real ones): 8 mm front-to-back vs 4 mm in-plane, so
    # its axial slices have non-square pixels and must be resampled to square (D19).
    write_nifti_case(root / "imagesTr/amos_0506.nii.gz", root / "labelsTr/amos_0506.nii.gz",
                     "RAS", "mri", codes, rng, spacing=np.array([4.0, 8.0, 4.0]))


def build_word(root: Path, rng) -> None:
    codes = {"rk": 4, "lk": 3, "liver": 1, "spleen": 2}
    for d in ("imagesTr", "labelsTr", "imagesVal", "labelsVal"):
        (root / d).mkdir(parents=True)
    for n in range(1, 5):
        name = f"word_{n:04d}.nii.gz"
        write_nifti_case(root / "imagesTr" / name, root / "labelsTr" / name,
                         list(ORIENTATIONS)[n % 5], "ct", codes, rng)
    write_nifti_case(root / "imagesVal" / "word_0099.nii.gz", root / "labelsVal" / "word_0099.nii.gz",
                     "RAS", "ct", codes, rng)                                  # must be ignored


# ------------------------------------------------------------------- synthetic CHAOS
N_SL, N_PIX, PIX = 36, 64, 5.0
IOP = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]          # standard axial: rows -> P, cols -> L (LPS)


def write_dicom(path: Path, arr: np.ndarray, ipp, inst: int) -> None:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID, ds.SOPInstanceUID = MRImageStorage, meta.MediaStorageSOPInstanceUID
    ds.Modality, ds.PatientID = "MR", "synthetic"
    ds.Rows, ds.Columns = arr.shape
    ds.PixelSpacing = [PIX, PIX]
    ds.ImageOrientationPatient = IOP
    ds.ImagePositionPatient = [float(v) for v in ipp]
    ds.InstanceNumber = inst
    ds.SliceThickness = PIX
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
    ds.PixelData = arr.astype(np.uint16).tobytes()
    if int(pydicom.__version__.split(".")[0]) >= 3:
        ds.save_as(str(path), enforce_file_format=True)
    else:
        ds.is_little_endian, ds.is_implicit_VR = True, False
        ds.save_as(str(path), write_like_original=False)


def chaos_slice(n: int):
    """Instance n (1-based) is superior-first. Returns LPS IPP and the world grid."""
    z = 87.5 - (n - 1) * PIX
    ipp = (-157.5, -157.5, z)
    r, c = np.meshgrid(np.arange(N_PIX), np.arange(N_PIX), indexing="ij")
    lps = np.stack([ipp[0] + c * PIX, ipp[1] + r * PIX, np.full(r.shape, z)], -1)
    ras = lps * np.array([-1.0, -1.0, 1.0])
    return ipp, organ_map(ras)


def grey(m: dict) -> np.ndarray:
    g = np.zeros(m["body"].shape, np.uint8)
    g[m["liver"]], g[m["spleen"]], g[m["rk"]], g[m["lk"]] = 63, 252, 126, 189
    return g


def build_chaos_patient(root: Path, pid: int, png_naming: str, rng) -> None:
    """png_naming: 'stem'    -> T2 and T1 Ground named after T2 / InPhase files
                   'rank'    -> Ground named gt_NNN in instance order
                   'reverse' -> Ground named gt_NNN in REVERSED instance order (a trap)"""
    pd = root / "MR" / str(pid)
    for d in ("T1DUAL/DICOM_anon/InPhase", "T1DUAL/DICOM_anon/OutPhase", "T1DUAL/Ground",
              "T2SPIR/DICOM_anon", "T2SPIR/Ground"):
        (pd / d).mkdir(parents=True)
    for n in range(1, N_SL + 1):
        ipp, m = chaos_slice(n)
        base = intensities(m, "mri", rng)
        g = grey(m)
        gt_name = {"stem": None, "rank": f"gt_{n:03d}.png",
                   "reverse": f"gt_{N_SL + 1 - n:03d}.png"}[png_naming]
        write_dicom(pd / "T2SPIR/DICOM_anon" / f"IMG-0002-{n:05d}.dcm", base * 0.8 + 20, ipp, n)
        Image.fromarray(g).save(pd / "T2SPIR/Ground" / (gt_name or f"IMG-0002-{n:05d}.png"))
        write_dicom(pd / "T1DUAL/DICOM_anon/InPhase" / f"IMG-0004-{2 * n:05d}.dcm", base, ipp, 2 * n)
        write_dicom(pd / "T1DUAL/DICOM_anon/OutPhase" / f"IMG-0004-{2 * n - 1:05d}.dcm",
                    base * 0.9, ipp, 2 * n - 1)
        Image.fromarray(g).save(pd / "T1DUAL/Ground" / (gt_name or f"IMG-0004-{2 * n:05d}.png"))


# ----------------------------------------------------------------------------- checks
def read_csv(path: Path) -> list[dict]:
    with open(path) as fh:
        return list(csv.DictReader(fh))


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)
    print(f"  ok  {msg}")


def run(args: list[str]) -> int:
    print(f"\n$ prep_data {' '.join(a if '/' not in a else Path(a).name for a in args)}")
    return prep_data.main(args)


def main() -> int:
    rng = np.random.default_rng(0)
    tmp = Path(tempfile.mkdtemp(prefix="prep_test_"))
    amos, word, chaos = tmp / "amos", tmp / "word", tmp / "chaos"
    build_amos(amos, rng)
    build_word(word, rng)
    (word.parent / "labelsTs").mkdir()
    build_chaos_patient(chaos, 1, "stem", rng)
    build_chaos_patient(chaos, 2, "rank", rng)
    out = tmp / "out"
    common = ["--amos-root", str(amos), "--word-root", str(word),
              "--word-test-labels", str(word.parent / "labelsTs"),
              "--chaos-root", str(chaos)]

    # 1. full positive run, parallel ------------------------------------------------
    rc = run(common + ["--out", str(out), "--workers", "2", "--neg-margin", "3"])
    check(rc == 0, "positive run exits 0 (PREP_OK)")
    rows, vols = read_csv(out / "manifest.csv"), read_csv(out / "volumes.csv")
    check(all(not v["error"] for v in vols), "no volume errors")
    for c in prep_data.QC_CHECKS:
        check(all(v[c] == "pass" for v in vols), f"{c} passes on every volume")

    def vols_in(ds, split, mod):
        return {v["case"] for v in vols if (v["dataset"], v["split"], v["modality"]) == (ds, split, mod)}

    check([len(vols_in("amos", s, "ct")) for s in ("train", "val", "test")] == [6, 2, 2],
          "AMOS CT pooled 60/20/20 -> 6/2/2 of 10 (11 minus 1 excluded)")
    check("amos_0085" not in {v["case"] for v in vols} and "amos_0085" not in {r["case"] for r in rows},
          "excluded amos_0085 is in neither volumes.csv nor manifest.csv")
    run_json = json.load(open(out / "run.json"))
    check(list(run_json["amos_excluded"]) == ["amos_0085"], "run.json records the exclusion and its reason")
    check([len(vols_in("amos", s, "mri")) for s in ("train", "val", "test")] == [4, 1, 1],
          "AMOS MRI pooled 60/20/20 -> 4/1/1 of 6")
    check({v["modality"] for v in vols if v["case"] >= "amos_0500" and v["dataset"] == "amos"} == {"mri"},
          "AMOS ids >= 500 are MRI")
    check(len(vols_in("word", "external", "ct")) == 4, "WORD: imagesTr only (imagesVal ignored)")
    check(len([v for v in vols if v["dataset"] == "chaos"]) == 6, "CHAOS: 2 patients x 3 sequences")
    pair = {v["case"]: v["pairing"] for v in vols if v["dataset"] == "chaos"}
    check(pair["chaos_01_t2"] == "stem" and pair["chaos_01_t1in"] == "stem", "CHAOS stem pairing")
    check(pair["chaos_01_t1out"] == "position-via-t1in", "CHAOS out-phase reuses in-phase GT by position")
    check(all(pair[f"chaos_02_{s}"] == "rank-instance" for s in ("t1in", "t1out", "t2")),
          "CHAOS rank pairing when PNG names match no DICOM")
    check(all(float(v["unmapped_frac"]) == 0 for v in vols if v["dataset"] == "chaos"),
          "CHAOS: every labelled PNG pixel maps to a known organ")

    leak = {}
    for r in rows:
        leak.setdefault(r["patient"], set()).add(r["split"])
    check(all(len(s) == 1 for s in leak.values()), "no patient appears in two splits")

    # every kidney voxel survives (ground truth never edited)
    for v in vols:
        if v["resampled"]:
            continue                      # upsampled: checked in mm below instead
        kept = [r for r in rows if r["case"] == v["case"]]
        assert sum(int(r["rk_area"]) for r in kept) == int(v["rk_vox"]), v["case"]
        assert sum(int(r["lk_area"]) for r in kept) == int(v["lk_vox"]), v["case"]
    check(True, f"all {sum(not v['resampled'] for v in vols)} square-pixel volumes: "
                "stored kidney pixels == label voxels (no GT edited)")

    # D19: the coronal-style MRI is stretched to square pixels with true proportions
    cor = next(v for v in vols if v["case"] == "amos_0506")
    check(cor["resampled"].startswith("40x80 -> 80x80 px") and "non-axial" in cor["warnings"],
          f"amos_0506 resampled to square: {cor['resampled']}")
    cor_rows = [r for r in rows if r["case"] == "amos_0506"]
    check({(r["sp_row"], r["sp_col"], r["sp_slice"]) for r in cor_rows} == {("4.0", "4.0", "4.0")},
          "amos_0506 stored spacing is square: 4 x 4 mm (was 8 x 4)")
    for side, vox in (("rk", "rk_vox"), ("lk", "lk_vox")):
        mm2 = sum(int(r[f"{side}_area"]) for r in cor_rows) * 4.0 * 4.0
        assert abs(mm2 / (int(cor[vox]) * 8.0 * 4.0) - 1) < 0.01, (side, mm2)
    check(True, "amos_0506: kidney area in mm^2 conserved exactly through the resample")

    def kidney_aspect_mm(case):          # height/width of the right kidney, in mm, at its widest
        r = max((r for r in rows if r["case"] == case), key=lambda r: int(r["rk_area"]))
        x0, y0, x1, y1 = json.loads(r["rk_box"])
        return ((y1 - y0 + 1) * float(r["sp_row"])) / ((x1 - x0 + 1) * float(r["sp_col"]))
    ratio = kidney_aspect_mm("amos_0506") / kidney_aspect_mm("amos_0001")
    # Unfixed, rows would be half as many -> ratio halves. The residual gap from 1.0 is the
    # coarse 8 mm sampling blurring the kidney edge (lost detail, not distortion).
    check(abs(ratio - 1) < 0.25, f"kidney proportions match a normal scan after resampling "
                                 f"(ratio {ratio:.2f}; unfixed would be ~{ratio / 2:.2f})")

    sliver = [r for r in rows if 0 < int(r["rk_area"]) < 50]
    check(bool(sliver) and all(r["rk_box"] == "null" for r in sliver),
          f"{len(sliver)} sliver slices keep their pixels but get no box")
    s = np.load(out / sliver[0]["path"])
    check(int((s["msk"] == 1).sum()) == int(sliver[0]["rk_area"]), "sliver mask pixels preserved on disk")

    # geometry: radiological view, spacing, full volumes vs bands
    both = [r for r in rows if r["rk_box"] != "null" and r["lk_box"] != "null"]
    for r in both:
        d = np.load(out / r["path"])
        rk_c = np.argwhere(d["msk"] == 1).mean(0)
        lk_c = np.argwhere(d["msk"] == 2).mean(0)
        assert rk_c[1] < lk_c[1], f"{r['case']} z{r['z']}: right kidney not on image left"
        assert rk_c[0] > d["msk"].shape[0] / 2, f"{r['case']} z{r['z']}: kidney not posterior (bottom)"
        rb, lb = json.loads(r["rk_box"]), json.loads(r["lk_box"])
        rr, cc = np.nonzero(d["msk"] == 1)
        assert [cc.min(), rr.min(), cc.max(), rr.max()] == rb, "rk box not tight"
        rr, cc = np.nonzero(d["msk"] == 2)
        assert [cc.min(), rr.min(), cc.max(), rr.max()] == lb, "lk box not tight"
        assert d["img"].dtype == np.uint8 and d["msk"].dtype == np.uint8
    check(len(both) > 0, f"{len(both)} two-kidney slices: right kidney on image LEFT, kidneys "
                         "posterior (bottom), boxes tight XYXY")
    nifti_sp = {(r["sp_row"], r["sp_col"], r["sp_slice"]) for r in rows
                if r["dataset"] != "chaos" and r["case"] != "amos_0506"}
    check(nifti_sp == {("4.0", "4.0", "5.0")}, "NIfTI spacing (row, col, slice) = 4, 4, 5 mm")
    chaos_sp = {(r["sp_row"], r["sp_col"], r["sp_slice"]) for r in rows if r["dataset"] == "chaos"}
    check(chaos_sp == {("5.0", "5.0", "5.0")}, "CHAOS spacing from DICOM = 5, 5, 5 mm")
    for v in vols:
        z_n, z0, z1 = int(v["n_slices"]), int(v["kidney_z_first"]), int(v["kidney_z_last"])
        if v["split"] in ("test", "external"):
            expected = z_n
        else:
            expected = min(z_n - 1, z1 + 3) - max(0, z0 - 3) + 1
        assert int(v["n_kept"]) == expected, (v["case"], v["n_kept"], expected)
        assert expected < z_n or v["split"] in ("test", "external"), v["case"]
    check(True, "test/external stored as full volumes; train/val as kidney band +/- margin")
    check(any((out / "qc").rglob("*.png")), "QC overlays written")

    # 2. --limit keeps the split assignment --------------------------------------------
    out2 = tmp / "out_limit"
    rc = run(common + ["--out", str(out2), "--limit", "1"])
    split_full = {v["case"]: v["split"] for v in vols}
    lim = read_csv(out2 / "volumes.csv")
    check(rc == 0 and all(split_full[v["case"]] == v["split"] for v in lim),
          "--limit subsamples without changing any case's split")

    # 2b. the independent verifier: passes clean output, catches a corrupted slice -----
    import verify_processed
    print("\n$ verify_processed (positive run)")
    check(verify_processed.main(["--out", str(out), "--skip-count-check"]) == 0,
          "independent verifier passes on the positive run")
    victim = read_csv(out2 / "manifest.csv")[0]["path"]
    with np.load(out2 / victim) as d:
        img, msk = d["img"], d["msk"].copy()
    msk[0, 0] = 7
    np.savez_compressed(out2 / victim, img=img, msk=msk)
    print("\n$ verify_processed (one corrupted slice)")
    check(verify_processed.main(["--out", str(out2), "--skip-count-check"]) == 1,
          "verifier catches a corrupted slice")

    # 3. refuses to overwrite a previous run -------------------------------------------
    try:
        run(common + ["--out", str(out)])
        check(False, "overwrite guard")
    except SystemExit as e:
        check("--overwrite" in str(e), "refuses to mix with a previous run without --overwrite")
    check((out / "manifest.csv").exists(), "previous run left intact")

    # 4. negative: swapped WORD label map -> q_kidney_lr must fail ----------------------
    saved = dict(prep_data.NIFTI_LABELS["word"])
    prep_data.NIFTI_LABELS["word"] = {"right": 3, "left": 4, "liver": 1}
    rc = run(["--word-root", str(word), "--out", str(tmp / "neg_word"), "--workers", "1"])
    prep_data.NIFTI_LABELS["word"] = saved
    neg = read_csv(tmp / "neg_word" / "volumes.csv")
    check(rc == 1 and all(v["q_kidney_lr"] == "fail" for v in neg),
          "swapped L/R label map is caught (q_kidney_lr FAIL, run fails)")

    # 5. negative: CHAOS PNGs in reversed order -> q_liver_above must fail ---------------
    chaos_bad = tmp / "chaos_bad"
    build_chaos_patient(chaos_bad, 3, "reverse", rng)
    rc = run(["--chaos-root", str(chaos_bad), "--out", str(tmp / "neg_chaos")])
    neg = read_csv(tmp / "neg_chaos" / "volumes.csv")
    check(rc == 1 and all(v["q_liver_above"] == "fail" for v in neg),
          "reversed CHAOS slice pairing is caught (q_liver_above FAIL)")
    rc = run(["--chaos-root", str(chaos_bad), "--out", str(tmp / "neg_chaos_fix"),
              "--chaos-png-order", "reverse"])
    check(rc == 0, "--chaos-png-order reverse repairs it (all QC pass)")

    # 6. negative: a CT volume filed under an MRI id -> q_modality must fail -------------
    amos_bad = tmp / "amos_bad"
    for d in ("imagesTr", "labelsTr", "imagesVa", "labelsVa"):
        (amos_bad / d).mkdir(parents=True)
    write_nifti_case(amos_bad / "imagesTr/amos_0600.nii.gz", amos_bad / "labelsTr/amos_0600.nii.gz",
                     "RAS", "ct", {"rk": 2, "lk": 3, "liver": 6}, rng)
    rc = run(["--amos-root", str(amos_bad), "--out", str(tmp / "neg_amos")])
    neg = read_csv(tmp / "neg_amos" / "volumes.csv")
    check(rc == 1 and neg[0]["q_modality"] == "fail", "CT data under an MRI case id is caught")

    # 7. negative: label/image geometry mismatch -> recorded error, no orphan slices -----
    amos_geo = tmp / "amos_geo"
    for d in ("imagesTr", "labelsTr", "imagesVa", "labelsVa"):
        (amos_geo / d).mkdir(parents=True)
    write_nifti_case(amos_geo / "imagesTr/amos_0001.nii.gz", amos_geo / "labelsTr/amos_0001.nii.gz",
                     "RAS", "ct", {"rk": 2, "lk": 3, "liver": 6}, rng)
    lbl = nib.load(str(amos_geo / "labelsTr/amos_0001.nii.gz"))
    shifted = lbl.affine.copy()
    shifted[0, 3] += 12.0
    nib.save(nib.Nifti1Image(np.asarray(lbl.dataobj), shifted), str(amos_geo / "labelsTr/amos_0001.nii.gz"))
    rc = run(["--amos-root", str(amos_geo), "--out", str(tmp / "neg_geo")])
    neg = read_csv(tmp / "neg_geo" / "volumes.csv")
    check(rc == 1 and "affine mismatch" in neg[0]["error"], "image/label affine mismatch is an error")
    check(not (tmp / "neg_geo" / "slices" / "amos" / "amos_0001").exists(), "failed volume leaves no orphan slices")

    print(f"\nALL TESTS PASSED  (scratch data in {tmp})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
