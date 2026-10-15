#!/bin/bash
#SBATCH --job-name=prep_data
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:1
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --account=vihaan
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/home/rethinkingai-self/vihaan/Segmentation/outputs/prep_data.%J.out
#SBATCH --error=/home/rethinkingai-self/vihaan/Segmentation/outputs/prep_data.%J.err
#
# Builds the 2D kidney-slice dataset (prep_data.py) on a compute node.
#
#   smoke test:  sbatch --export=ALL,LIMIT=2 prep_data.sbatch        -> data/processed_smoke
#   full run:    sbatch prep_data.sbatch                              -> data/processed
#   re-run:      sbatch --export=ALL,OVERWRITE=1 prep_data.sbatch     (replaces data/processed)
#
# The log header dumps CHAOS's own definitions.txt / notes and file layout, so the
# grey levels and DICOM<->PNG naming can be cross-checked against prep_data.py.
# After building, verify_processed.py re-reads every slice and checks the split counts
# against DECISIONS.md. The job exits non-zero unless BOTH end OK.

set -u
SEG=/home/rethinkingai-self/vihaan/Segmentation
PY=/home/rethinkingai-self/vihaan/miniconda3/envs/seg311/bin/python
D=$SEG/data
LIMIT=${LIMIT:-}
OVERWRITE=${OVERWRITE:-}
if [ -n "$LIMIT" ]; then
    OUT=$D/processed_smoke
    EXTRA="--limit $LIMIT --overwrite"
    VERIFY_EXTRA="--skip-count-check"
else
    OUT=$D/processed
    EXTRA=""
    VERIFY_EXTRA=""
fi
[ -n "$OVERWRITE" ] && EXTRA="$EXTRA --overwrite"

echo "=== job ${SLURM_JOB_ID:-manual} on $(hostname)   limit=${LIMIT:-none}   out=$OUT"
echo "=== CHAOS definitions.txt";       cat "$D/chaos/Train_Sets/definitions.txt" 2>/dev/null
echo; echo "=== CHAOS NOTES_PLEASE_READ.txt"; cat "$D/chaos/Train_Sets/NOTES_PLEASE_READ.txt" 2>/dev/null
echo; echo "=== CHAOS MR/1 layout"
find "$D/chaos/Train_Sets/MR/1" -maxdepth 3 -type d | sort
for f in T1DUAL/DICOM_anon/InPhase T1DUAL/DICOM_anon/OutPhase T1DUAL/Ground T2SPIR/DICOM_anon T2SPIR/Ground; do
    printf '  %-28s %4s files, e.g. %s\n' "$f" "$(ls "$D/chaos/Train_Sets/MR/1/$f" | wc -l)" \
        "$(ls "$D/chaos/Train_Sets/MR/1/$f" | head -2 | tr '\n' ' ')"
done
echo

# shellcheck disable=SC2086
"$PY" "$SEG/scripts/prep_data.py" \
    --amos-root  "$D/amos22/amos22" \
    --word-root  "$D/word/WORD-V0.1.0" --word-test-labels "$D/word/labelsTs" \
    --chaos-root "$D/chaos/Train_Sets" \
    --out "$OUT" --workers 14 $EXTRA
RC=$?
echo "=== prep exit $RC"

echo; echo "=== independent verification"
# shellcheck disable=SC2086
"$PY" "$SEG/scripts/verify_processed.py" --out "$OUT" $VERIFY_EXTRA
VRC=$?
echo "=== verify exit $VRC"
echo "=== disk: $(du -sh "$OUT" 2>/dev/null | cut -f1)"
[ "$RC" -ne 0 ] && exit "$RC"
exit "$VRC"
