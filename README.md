Task-specific distillation of the Segment Anything Model (SAM) into a lightweight, prompt-free student for kidney segmentation across CT and MRI (IE643 course project, IIT Bombay).
Four arms on identical test slices: (1) off-the-shelf SAM ViT-H with ground-truth box prompts, (2) fine-tuned SAM, (3) a student trained from scratch, (4) a student distilled from fine-tuned SAM; plus cross-modality transfer (CT↔MRI).
Data: AMOS22 (CT + MRI) for training, validation and in-distribution test; WORD (100 CT) and CHAOS (20 MRI patients × T1-in/T1-out/T2) as external test sets. Splits by patient.
Evaluation: 13,145 kidney instances on 7,386 slices, scored with 2D/3D Dice and normalized surface distance (NSD at 1/2/3/5 mm, physical spacing), averaged per patient with bootstrap 95% CIs, plus a box-prompt robustness sweep (0/5/10/20% jitter).
Includes a verified preprocessing pipeline, resumable and shardable SLURM evaluation scripts, a shared metrics module with self-tests, and a full decision log (DECISIONS.md).
Status: arm 1 complete. Off-the-shelf SAM reaches 0.89–0.93 Dice at 10% box jitter on all test sets, but drops to about 0.78 at 20%. Arms 2–4 in progress.
