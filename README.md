# Medical-Segmentation-IE643

**Task-specific distillation of the Segment Anything Model (SAM) into a lightweight, prompt-free
student for cross-modality (CT / MRI) kidney segmentation.**
Course project for IE643 (Deep Learning: Theory and Practice), IIT Bombay.

SAM segments almost anything when you give it a box, but it is large (ViT-H: 632 M encoder
parameters), needs a prompt for every object, and was trained on natural images. This project
asks how much of SAM's ability survives on medical images, how much fine-tuning recovers, and
whether that ability can be distilled into a small network that needs no prompt at all.

| | Arm | Prompt | Status |
|---|---|---|---|
| 1 | Off-the-shelf SAM (ViT-H, zero-shot) | ground-truth box | **done** |
| 2 | Fine-tuned SAM (the teacher) | ground-truth box | in progress |
| 3 | Lightweight student trained from scratch | none | planned |
| 4 | Lightweight student distilled from arm 2 | none | planned |
| + | Cross-modality transfer (train on CT → test on MRI, and vice versa) | | planned |

All four arms are scored on **exactly the same 13,145 kidney instances**, with the same metric
code, so every difference between rows is due to the model alone.

---

## Arm 1 results — off-the-shelf SAM ViT-H

Box prompt from the ground-truth mask, perturbed by 10% (SAM's own training noise). Mean over
patients with 95% bootstrap CI. All metrics range 0–1, higher is better.

| Test set | Patients | Kidney instances | Dice 2D | NSD @ 3 mm (2D) | Dice 3D |
|---|---|---|---|---|---|
| AMOS CT | 59 | 3,315 | 0.905 [0.899, 0.911] | 0.881 [0.870, 0.891] | 0.913 |
| AMOS MRI | 12 | 1,233 | 0.909 [0.899, 0.919] | 0.904 [0.887, 0.918] | 0.917 |
| CHAOS MRI T1 in-phase | 20 | 526 | 0.903 [0.888, 0.918] | 0.861 [0.816, 0.900] | 0.912 |
| CHAOS MRI T1 out-of-phase | 20 | 526 | 0.890 [0.876, 0.901] | 0.856 [0.822, 0.886] | 0.902 |
| CHAOS MRI T2-SPIR | 20 | 506 | 0.923 [0.912, 0.932] | 0.915 [0.894, 0.934] | 0.931 |
| WORD CT (external) | 100 | 7,039 | 0.925 [0.921, 0.928] | 0.935 [0.930, 0.940] | 0.928 |

**Sensitivity to the box prompt** (Dice 2D; each box edge independently shifted by Gaussian
noise with σ = level × box side, capped at 20 px):

| Test set | 0% (tight box) | 5% | 10% | 20% |
|---|---|---|---|---|
| AMOS CT | 0.923 | 0.922 | 0.905 | 0.774 |
| AMOS MRI | 0.928 | 0.928 | 0.909 | 0.767 |
| CHAOS T1-in | 0.917 | 0.913 | 0.903 | 0.784 |
| CHAOS T1-out | 0.905 | 0.904 | 0.890 | 0.790 |
| CHAOS T2 | 0.939 | 0.937 | 0.923 | 0.799 |
| WORD CT | 0.941 | 0.941 | 0.925 | 0.794 |

**Takeaways**
- With a good box, SAM segments kidneys well on every set, with no meaningful CT-vs-MRI gap and no
  drop on the external datasets. For reference, SAM-H with ground-truth boxes reaches
  77.8 / 79.1 mIoU on COCO / LVIS natural images; our clean-box Dice of 0.905–0.941 corresponds to
  roughly 0.83–0.89 IoU, so there is no degradation for this (easy, high-contrast) target.
- Performance hinges on the prompt: flat up to 5% jitter, −1.5 to −2 points at 10%, and a
  **13–16 point cliff at 20%** on every dataset. A real deployment has no oracle box, which is
  the motivation for the prompt-free students.
- Errors concentrate on small kidney cross-sections at the poles (Dice 0.874 for the smallest
  quartile vs 0.924 for the rest). Only 2 of 13,145 instances failed completely (Dice = 0).

---

## Data

| Dataset | Modality | Role | Patients / volumes |
|---|---|---|---|
| [AMOS22](https://amos22.grand-challenge.org/) | CT + MRI | train / val / in-distribution test | CT 177 / 59 / 59, MRI 36 / 12 / 12 |
| [WORD](https://github.com/HiLab-git/WORD) | CT | external test | 100 (imagesTr) |
| [CHAOS](https://chaos.grand-challenge.org/) | MRI | external test | 20 patients × {T1 in-phase, T1 out-of-phase, T2-SPIR} |

- AMOS22: the 360 labelled cases are pooled and split 60/20/20 per modality with seed 643.
  Case id ≥ 500 is MRI. Five CT cases are excluded (label noise, a renal transplant, three
  single-kidney cases).
- CHAOS CT has liver labels only, so only CHAOS MRI is used.
- Splits are **by patient**; `prep_data.py` asserts no patient appears in two splits.

**Evaluation set** (test + external splits, slices containing a promptable kidney):

| Group | Cases | Slices | Kidney instances |
|---|---|---|---|
| AMOS CT | 59 | 1,870 | 3,315 |
| AMOS MRI | 12 | 691 | 1,233 |
| CHAOS T1-in / T1-out / T2 | 20 / 20 / 20 | 297 / 297 / 284 | 526 / 526 / 506 |
| WORD CT | 100 | 3,947 | 7,039 |
| **Total** | **231** | **7,386** | **13,145** |

## Preprocessing (`prep_data.py`)

- Every volume is reoriented to canonical RAS. Axial slices are stored in radiological view
  (anterior at the top, the patient's right on the image left).
- Labels are unified to `0` background, `1` right kidney, `2` left kidney. The source label maps
  differ (AMOS R=2 / L=3, WORD R=4 / L=3, CHAOS PNG grey levels) and live in one table.
- Intensities go to uint8: CT window [−150, 250] HU; MRI per-volume 0.5–99.5 percentile.
- Slices with non-square pixels (18 coronal AMOS MRI) are upsampled to square pixels at the finer
  spacing, so anatomy keeps its true proportions.
- Each slice stores its in-plane spacing in mm, so surface metrics are computed in physical units.
- Kidney cross-sections under 50 px keep their mask but get no box: SAM is never prompted with a
  degenerate box, and the ground truth is never edited.
- Test and external volumes are stored in full (for 3D scoring). Train/val keep the kidney band
  ±15 slices.
- Anatomy-anchored QC on every volume catches orientation flips, swapped labels and reversed slice
  order (liver right of the kidneys, liver above them, kidneys posterior, CT contains air).
- `verify_processed.py` independently re-reads every slice and checks the split counts against the
  plan.

Output: `manifest.csv` (one row per slice: split, boxes, areas, spacing, path), `volumes.csv`,
`run.json`, and `slices/<dataset>/<case>/<z>.npz` holding `img` and `msk`.

## Evaluation protocol (shared by all arms: `sam_metrics.py`)

- **Dice** and **normalized surface distance** (NSD at 1, 2, 3 and 5 mm), per kidney per slice
  (2D) and per kidney volume (3D). NSD uses Google DeepMind's
  [`surface-distance`](https://github.com/google-deepmind/surface-distance), which weights boundary
  elements by their physical size. This is the correct choice on anisotropic grids, where counting
  boundary voxels can be off by tens of points.
- **Headline NSD tolerance is 3 mm.** It must be at least the coarsest in-plane pixel in the test
  data (2.03 mm, CHAOS), otherwise it would be sub-pixel. All four tolerances are stored.
- **Aggregation:** mean within each patient, then across patients, with a percentile bootstrap
  95% CI over patients. A flat mean over instances would let one AMOS MRI patient carry ~20% of
  that set.
- **Box prompts** for the SAM arms: one tight box per kidney from the ground truth, plus jittered
  copies at 0 / 5 / 10 / 20%. The jitter is seeded per (case, slice, side, level), so every run is
  reproducible. SAM encodes each slice once and decodes once per kidney per jitter level, so the
  robustness sweep costs almost nothing extra.
- Preconditions are enforced, not assumed: boolean masks, spacing in array-axis order, non-empty
  ground truth. A metrics self-test runs before every evaluation.

## Repository layout

| File | Purpose |
|---|---|
| `prep_data.py` | Build the 2D kidney-slice dataset from AMOS22, WORD and CHAOS |
| `verify_processed.py` | Independent re-check of every stored slice and of the split counts |
| `test_prep_data.py` | Tests for preprocessing on synthetic volumes (orientation, labels, boxes, QC) |
| `prep_data.sbatch` | SLURM job: preprocess, then verify |
| `sam_metrics.py` | Shared metrics: Dice, NSD (2D/3D), box jitter, box IoU, per-patient bootstrap, self-test |
| `eval_sam.py` | Arm 1 evaluation: SAM with box prompts, resumable, shardable, self-verifying |
| `aggregate_arm.py` | Per-instance scores → `summary.csv`, `breakdowns.csv`, `summary.md` |
| `test_eval_sam.py` | 149 checks: metrics, jitter, aggregation, end-to-end with a stub SAM, resume, sharding, merge, SLURM scripts |
| `eval_sam.sbatch` | SLURM job: one-job evaluation (also supports `RESUME=1`) |
| `arm1_shard{0..3}.sbatch`, `arm1_merge.sbatch` | Four independent shard jobs plus a manual merge job |
| `gen_sbatch.py` | Generates the shard and merge scripts from a single template |

## Setup

Python 3.11, PyTorch with CUDA. Developed on the Prajna HPC cluster (IIT Bombay, NVIDIA A40).

```bash
conda create -n seg311 python=3.11 -y && conda activate seg311
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install numpy scipy scikit-image pandas pillow nibabel pydicom SimpleITK pyzipper monai
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install git+https://github.com/google-deepmind/surface-distance.git
```

SAM checkpoints come from the
[segment-anything repository](https://github.com/facebookresearch/segment-anything#model-checkpoints)
(`sam_vit_h_4b8939.pth`, `sam_vit_l_0b3195.pth`, `sam_vit_b_01ec64.pth`).

The `.sbatch` files hard-code the cluster paths (`SEG=/home/.../Segmentation`, the conda python,
partition and account). Edit those lines for another machine. The expected layout is
`Segmentation/{scripts,data,checkpoints,outputs}`.

## Running

**1. Preprocess and verify**
```bash
python prep_data.py --amos-root data/amos22/amos22 --word-root data/word/WORD-V0.1.0 \
    --word-test-labels data/word/labelsTs --chaos-root data/chaos/Train_Sets \
    --out data/processed --workers 8
python verify_processed.py --out data/processed          # ends VERIFY_OK
```

**2. Run the tests** (CPU only, about a minute; a stub replaces SAM)
```bash
python test_prep_data.py
python test_eval_sam.py                                  # ends ALL TESTS PASSED
```

**3. Arm 1, as a single job**
```bash
python eval_sam.py --processed data/processed \
    --checkpoint checkpoints/sam_vit_h_4b8939.pth --model-type vit_h --out outputs/arm1
python aggregate_arm.py --arm-dir outputs/arm1 --label "Arm 1: off-the-shelf SAM"
```
On SLURM: `sbatch eval_sam.sbatch`. After a timeout, continue with
`sbatch --export=ALL,RESUME=1 eval_sam.sbatch`.

**3b. Arm 1, as four parallel jobs**
```bash
sbatch arm1_shard0.sbatch; sbatch arm1_shard1.sbatch
sbatch arm1_shard2.sbatch; sbatch arm1_shard3.sbatch
# when all four logs end with SHARD_OK:
sbatch arm1_merge.sbatch                                 # ends EVAL_OK + results table
```
Each shard scores a slice-balanced quarter of the unfinished cases and writes only to its own
folder. The merge combines everything, re-verifies it and aggregates. A full ViT-H pass takes about
0.7 s per slice on one A40, so about 1.5 h for all 7,386 slices.

**Outputs** (`outputs/arm1/`):
- `instances.csv`: one row per kidney × jitter level, holding the boxes, Dice, NSD at every
  tolerance, and SAM's predicted IoU;
- `volumes_3d.csv`: one row per kidney volume × jitter level;
- `summary.csv` / `summary.md` / `breakdowns.csv`: the reported tables;
- `spacing_report.csv`, `run.json` (config, versions, per-stage timing, checks) and `qc/`
  (worst and median overlays per group).

## Safeguards in the evaluation code

A long GPU job must not silently corrupt a result, so `eval_sam.py` checks the following.

- **Before any GPU work:** the evaluation set must match the expected per-group counts, and every
  case must be a complete z-stack.
- **Resume:** keeps only cases whose instance and volume rows are complete, with exactly the
  expected keys. It drops a line cut off by a killed job and redoes that case, backing up the
  files first. It refuses result files with foreign columns, and refuses to overwrite existing
  results unless `--resume` or `--overwrite` is given.
- **Every stored row is re-derived from the manifest:** the slice, the kidney side, the clean box,
  the jittered box from its seed, the ground-truth area, the identity fields, the spacing, and the
  score ranges. Any row from a different code version or a misaligned merge is caught.
- **Sharding:** shards never share a file. The merge refuses missing, partial or inconsistent
  shards and writes nothing in that case. The SLURM scripts refuse to start next to a job writing
  the same files.
- **QC overlays** re-run SAM on the worst and median instances. The re-run Dice must match the
  stored value, which is a cross-job reproducibility check: the arm 1 run reproduced to 5 × 10⁻⁷.
- `aggregate_arm.py` only summarises results that `run.json` certifies as complete.

## Roadmap

- **Arm 2:** fine-tune SAM on the AMOS training split (Dice + BCE, AdamW), with box jitter in
  training aimed at the 20% robustness cliff. The checkpoint is chosen on AMOS validation.
  Reference recipe: MedSAM.
- **Arms 3–4:** a lightweight prompt-free student, trained from scratch and distilled from arm 2.
- **Cross-modality transfer:** CT-only and MRI-only training, tested on the other modality.

## References

- Kirillov et al., *Segment Anything*, ICCV 2023 —
  [paper](https://arxiv.org/abs/2304.02643), [code](https://github.com/facebookresearch/segment-anything)
- Ma et al., *Segment anything in medical images* (MedSAM), Nature Communications 15:654, 2024 —
  [code](https://github.com/bowang-lab/MedSAM)
- Ji et al., *AMOS: A Large-Scale Abdominal Multi-Organ Benchmark*, NeurIPS 2022
- Luo et al., *WORD: A large scale dataset, benchmark and clinical applicable study for abdominal
  organ segmentation from CT image*, Medical Image Analysis 2022
- Kavur et al., *CHAOS Challenge — combined (CT-MR) healthy abdominal organ segmentation*,
  Medical Image Analysis 2021
- Nikolov et al., *Clinically applicable segmentation of head and neck anatomy for radiotherapy*
  (surface Dice / NSD), JMIR 2021 — [surface-distance](https://github.com/google-deepmind/surface-distance)
- Mazurowski et al., *Segment anything model for medical image analysis: an experimental study*,
  Medical Image Analysis 2023

The datasets are not redistributed here. Obtain them from their official sources under their
respective licences.

## License

MIT — see [LICENSE](LICENSE).
