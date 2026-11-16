"""Unit tests for demo_core on synthetic volumes (DEMO_PLAN §5): export orientation round
trip, scoring and volume formula, and clear errors for bad inputs. No model weights needed.
The checks against the real evaluation are in verify_demo.py.

Run:  python test_demo.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

import demo_core as D

FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def synthetic(td: Path, spacing=(0.8, 0.8, 3.0), shape=(64, 48, 12), label_shape=None, ndim4=False,
              kidneys=True, name="amos_0001"):
    rng = np.random.default_rng(0)
    img = rng.normal(0, 100, shape).astype(np.float32)
    lab = np.zeros(label_shape or shape, np.int16)
    if kidneys:
        lab[40:52, 15:30, 3:9] = 2          # AMOS right kidney: RAS, so high x = patient's right
        lab[10:22, 15:28, 4:10] = 3         # AMOS left kidney
    aff = np.diag([*spacing, 1.0])
    if ndim4:
        img = img[..., None]
    ip, lp = td / f"{name}.nii.gz", td / f"{name}_lbl" / f"{name}.nii.gz"
    lp.parent.mkdir(exist_ok=True)
    nib.save(nib.Nifti1Image(img, aff), str(ip))
    nib.save(nib.Nifti1Image(lab, aff), str(lp))
    return ip, lp, lab


def expect_error(name, fn):
    try:
        fn()
        check(name, False, "no error raised")
    except D.DemoError as e:
        check(name, True)
        print("      message:", e)
    except Exception as e:                       # a crash, not a clear message
        check(name, False, f"{type(e).__name__}: {e}")


def main() -> int:
    with tempfile.TemporaryDirectory() as t:
        td = Path(t)
        for tag, sp in (("square pixels", (0.8, 0.8, 3.0)), ("resampled to square", (0.7, 1.4, 3.0))):
            ip, lp, lab = synthetic(td, sp)
            case = D.load_nifti_case(ip, lp, "amos")
            check(f"{tag}: resample flag", (case.out_hw is None) == (sp[0] == sp[1]))
            # the stored ground truth, exported, must equal the original label volume (AMOS 2/3 -> 1/2)
            back = D.to_canonical(case.msk, case)
            want = np.where(lab == 2, 1, np.where(lab == 3, 2, 0))
            check(f"{tag}: export inverts make_slice exactly", np.array_equal(back, want),
                  f"{int((back != want).sum())} voxels differ")
            # the right kidney is drawn on the image's left (radiological view)
            cols = np.nonzero(case.msk == 1)[2]
            check(f"{tag}: right kidney on the image's left", cols.mean() < case.img.shape[2] / 2)
            # a perfect prediction scores 1 and its volume is voxels x voxel volume
            perfect = D.Prediction(D.ModelSpec("gt", "gt", "student", Path("."), False),
                                   None, {s: case.msk == s for s, _ in D.SIDES}, {}, 0.0, 1)
            sc = D.score(perfect, case)
            check(f"{tag}: perfect prediction Dice = NSD = 1", sc["dice"] == 1.0 and sc["nsd_3"] == 1.0)
            vox_ml = float(np.prod(sp)) / 1000
            check(f"{tag}: kidney volume in mL", abs(sc["vol_gt_right"] - (lab == 2).sum() * vox_ml) < 1e-6,
                  f"{sc['vol_gt_right']} vs {(lab == 2).sum() * vox_ml}")
            out = D.save_nifti(case.msk, case, td / "out.nii.gz")
            re = nib.load(str(out))
            check(f"{tag}: saved NIfTI has the scan's affine and shape",
                  np.allclose(re.affine, case.affine) and re.shape == lab.shape)
            for f in td.rglob("*.nii.gz"):
                f.unlink()

        ip, lp, _ = synthetic(td, ndim4=True)
        expect_error("4D image -> clear message", lambda: D.load_nifti_case(ip, lp, "amos"))
        ip, lp, _ = synthetic(td, label_shape=(64, 48, 11))
        expect_error("label of another shape -> clear message", lambda: D.load_nifti_case(ip, lp, "amos"))
        ip, lp, _ = synthetic(td, kidneys=False)
        expect_error("label without kidney -> clear message", lambda: D.load_nifti_case(ip, lp, "amos"))
        bad = td / "scan.png"
        bad.write_bytes(b"not a scan")
        expect_error("wrong file type -> clear message", lambda: D.load_nifti_case(bad, bad, "amos"))
        ip, _, _ = synthetic(td)
        _, lp2, _ = synthetic(td, name="amos_0002")
        expect_error("scan and label of different cases -> clear message", lambda: D.load_nifti_case(ip, lp2, "amos"))
    print("ALL TESTS PASSED" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
