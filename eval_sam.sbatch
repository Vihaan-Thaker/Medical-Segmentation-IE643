#!/bin/bash
#SBATCH --job-name=arm1_sam
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --account=vihaan
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/home/rethinkingai-self/vihaan/Segmentation/outputs/arm1_sam.%J.out
#SBATCH --error=/home/rethinkingai-self/vihaan/Segmentation/outputs/arm1_sam.%J.err
#
# Arm 1 — off-the-shelf SAM on all four test sets in ONE job (DECISIONS.md D24).
#
#   smoke:  sbatch --export=ALL,LIMIT=2 eval_sam.sbatch   -> outputs/arm1_smoke
#   full:   sbatch eval_sam.sbatch                         -> outputs/arm1
#           (refused if outputs/arm1 already holds results -- use RESUME=1 or the shards)
#   resume: sbatch --export=ALL,RESUME=1 eval_sam.sbatch   (continues an interrupted run)
#   model:  sbatch --export=ALL,MODEL=vit_b eval_sam.sbatch
#
# PARALLEL (D27) -- N GPUs at once, then one merge. Never shares a file between jobs.
#   shards: sbatch --array=0-3 --export=ALL,SHARDS=4 eval_sam.sbatch
#           each task skips cases already complete in outputs/arm1, takes 1/4 of the rest
#           (balanced by slices), writes only outputs/arm1/shards/K_of_4/. Resubmit the
#           same command after a timeout: every shard resumes its own files.
#   merge:  sbatch --dependency=afterok:<ARRAY_JOB_ID> --export=ALL,MERGE=4 eval_sam.sbatch
#           combines base + shards into outputs/arm1, re-verifies all counts, QC, aggregate.
#
# MEASURED (job 336480, A40, fp32): ~5.8 s/slice average -- the full 7,386 slices do not fit
# in 8 h. That job scored AMOS + CHAOS completely and 38/100 WORD before timing out;
# RESUME=1 continues it without losing or duplicating a row. Progress lines now print
# seconds/slice per stage (io, encode, decode, metric2d, metric3d) so the bottleneck is
# measured rather than guessed.
#
# The metrics self-test runs first and aborts in seconds if `surface-distance` is wrong
# or missing, rather than after a full pass.

set -u
# Pin CPU threads to the allocation. Libraries default to every core on the node, which
# oversubscribes our 8 CPUs; GPU numerics are unaffected, so resumed rows stay consistent.
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS NUMEXPR_NUM_THREADS=$OMP_NUM_THREADS
SEG=/home/rethinkingai-self/vihaan/Segmentation
PY=$SEG/../miniconda3/envs/seg311/bin/python
[ -x "$PY" ] || PY=/home/rethinkingai-self/vihaan/miniconda3/envs/seg311/bin/python

MODEL=${MODEL:-vit_h}
case "$MODEL" in
    vit_h) CKPT=sam_vit_h_4b8939.pth ;;
    vit_l) CKPT=sam_vit_l_0b3195.pth ;;
    vit_b) CKPT=sam_vit_b_01ec64.pth ;;
    *) echo "unknown MODEL=$MODEL"; exit 2 ;;
esac

LIMIT=${LIMIT:-}
RESUME=${RESUME:-}
SHARDS=${SHARDS:-}
MERGE=${MERGE:-}
if [ -n "$LIMIT" ]; then
    OUT=$SEG/outputs/arm1_smoke
    EXTRA="--limit $LIMIT"
else
    OUT=$SEG/outputs/arm1
    EXTRA=""
fi
[ -n "$RESUME" ] && EXTRA="$EXTRA --resume"
if [ -n "$SHARDS" ] && [ -n "$MERGE" ]; then
    echo "set SHARDS or MERGE, not both"; exit 2
fi
if [ -n "$SHARDS" ]; then
    if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
        echo "SHARDS=$SHARDS needs an array job: sbatch --array=0-$((SHARDS - 1)) ..."; exit 2
    fi
    EXTRA="$EXTRA --shard $SLURM_ARRAY_TASK_ID --num-shards $SHARDS"
fi
[ -n "$MERGE" ] && EXTRA="$EXTRA --merge $MERGE"

echo "=== job ${SLURM_JOB_ID:-manual} on $(hostname)"
echo "=== model=$MODEL  limit=${LIMIT:-none}  resume=${RESUME:-no}  shard=${SLURM_ARRAY_TASK_ID:-}/${SHARDS:-}  merge=${MERGE:-no}  out=$OUT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
"$PY" -c "import torch,surface_distance,numpy;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'numpy',numpy.__version__)"
echo

# shellcheck disable=SC2086
"$PY" "$SEG/scripts/eval_sam.py" \
    --processed "$SEG/data/processed" \
    --checkpoint "$SEG/checkpoints/$CKPT" \
    --model-type "$MODEL" \
    --out "$OUT" $EXTRA
RC=$?
echo "=== eval exit $RC"
[ "$RC" -ne 0 ] && exit "$RC"
# A shard holds only part of the results; aggregation runs once, after the merge.
[ -n "$SHARDS" ] && exit 0

echo; echo "=== aggregate"
"$PY" "$SEG/scripts/aggregate_arm.py" --arm-dir "$OUT" \
    --label "Arm 1: off-the-shelf SAM ($MODEL, zero-shot, oracle box prompts)"
ARC=$?
echo "=== aggregate exit $ARC"
echo "=== disk: $(du -sh "$OUT" 2>/dev/null | cut -f1)"
exit "$ARC"
