#!/bin/bash
#SBATCH --job-name=kd_probe
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:1
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --account=vihaan
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=/home/rethinkingai-self/vihaan/Segmentation/outputs/kd_speed_probe.%j.out
#SBATCH --error=/home/rethinkingai-self/vihaan/Segmentation/outputs/kd_speed_probe.%j.err
#
# Diagnostic (writes nothing but its log): why the arm-4 preflight's steps took 1.0 s (arm 3: 0.065 s).
# Same node type, CPUs and thread settings as the training jobs.
#   sbatch kd_speed_probe.sbatch
set -u
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS NUMEXPR_NUM_THREADS=$OMP_NUM_THREADS
SEG=/home/rethinkingai-self/vihaan/Segmentation
PY=/home/rethinkingai-self/vihaan/miniconda3/envs/seg311/bin/python
echo "=== job ${SLURM_JOB_ID:-manual} (kd_probe) on $(hostname)"
nvidia-smi
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
case "$GPU" in
    *A100*) ;;
    *) echo "REFUSING TO START -- must run on an A100, got '$GPU'"; exit 4 ;;
esac
"$PY" "$SEG/scripts/kd_speed_probe.py" --processed "$SEG/data/processed" --teacher-dir "$SEG/teacher/arm2_r4_train" --workers 12
RC=$?
echo "=== probe exit $RC"
nvidia-smi --query-gpu=utilization.gpu,clocks.sm,clocks.max.sm,temperature.gpu,power.draw --format=csv
exit "$RC"
