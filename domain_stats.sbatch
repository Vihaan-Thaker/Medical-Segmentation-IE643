#!/bin/bash
#SBATCH --job-name=domain_stats
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --account=vihaan
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/rethinkingai-self/vihaan/Segmentation/outputs/domain_stats.%j.out
#SBATCH --error=/home/rethinkingai-self/vihaan/Segmentation/outputs/domain_stats.%j.err
#
# Diagnostic (no model, CPU only): image statistics per dataset -- why CHAOS T1 is different.
#   sbatch domain_stats.sbatch      -> outputs/domain_stats/{domain_stats.md, *.csv, montage_*.png}
#
set -u
SEG=/home/rethinkingai-self/vihaan/Segmentation
PY=/home/rethinkingai-self/vihaan/miniconda3/envs/seg311/bin/python
O=$SEG/outputs
echo "=== job ${SLURM_JOB_ID:-manual} on $(hostname)"
"$PY" "$SEG/scripts/domain_stats.py" --processed "$SEG/data/processed" --out "$O/domain_stats" \
    --arms "$O/arm1_vit_b" "$O/arm2_r4" "$O/arm3_plain_s643" "$O/arm3_plain_s644" "$O/arm3_plain_s645" "$O/arm3_box_s643"
RC=$?; echo "=== exit $RC"; exit $RC
