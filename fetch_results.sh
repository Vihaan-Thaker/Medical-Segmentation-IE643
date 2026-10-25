#!/bin/bash
# Copy the small arm-1/1b/2 result files from Prajna to IE643_Project/results/ (ARM2_POST_PLAN.md §3.1).
# Run on the Mac, by you:
#
#   bash fetch_results.sh                    # summaries, compares, run/model records, training logs
#   bash fetch_results.sh --with-instances   # + instances.csv of arm1_vit_b, arm2_r4, arm2_r8 (~9 MB each;
#                                            #   needed only for figure (c) of plot_results.py)
#   bash fetch_results.sh --dry-run          # show what would be copied, copy nothing
#
# Never copies checkpoints (*.pt, *.pth), QC images, volumes_3d.csv, shard folders or logs.
# First real use: run with --dry-run and check the listed files (the Mac's openrsync talking to
# Prajna's GNU rsync was not testable locally; the rules were tested with openrsync on both ends).
set -euo pipefail

REMOTE="vihaan@prajna.iitb.ac.in"
SEGR="/home/rethinkingai-self/vihaan/Segmentation"
SRC="${FETCH_SRC:-$REMOTE:$SEGR/}"                   # FETCH_SRC: tests only (a local folder)
ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"          # IE643_Project
DEST="$ROOT/results"
WITH_INST=0
DRY=0
for arg in "$@"; do
    case "$arg" in
        --with-instances) WITH_INST=1 ;;
        --dry-run) DRY=1 ;;
        *) echo "unknown option: $arg"; exit 2 ;;
    esac
done

mkdir -p "$DEST"
DEST_REAL="$(cd "$DEST" && pwd -P)"
case "$DEST_REAL" in
    "$ROOT"/*) ;;
    *) echo "REFUSING: destination $DEST_REAL is outside $ROOT"; exit 3 ;;
esac

ARMS="arm1 arm1_vit_b arm2_r4 arm2_r8"
RULES=()
# Anchored rules (leading /): outputs/arm1/run.json matches, outputs/arm1/shards/*/run.json does not.
for a in $ARMS; do
    for f in summary.csv summary.md breakdowns.csv run.json model.json spacing_report.csv; do
        RULES+=("--include=/outputs/$a/$f")
    done
    RULES+=("--include=/outputs/$a/compare_vs_*/" "--include=/outputs/$a/compare_vs_*/compare.csv"
            "--include=/outputs/$a/compare_vs_*/compare.md")
done
for r in 4 8; do
    for f in metrics.csv run.json config.json state.json; do
        RULES+=("--include=/runs/arm2_r$r/$f")
    done
done
RULES+=("--include=/runs/arm2_preflight/preflight.json" "--include=/runs/arm2_selection.json")
if [ "$WITH_INST" -eq 1 ]; then
    for a in arm1_vit_b arm2_r4 arm2_r8; do RULES+=("--include=/outputs/$a/instances.csv"); done
fi
# Directories to walk (only these), then exclude everything else.
RULES+=("--include=/outputs/" "--include=/runs/")
for a in $ARMS; do RULES+=("--include=/outputs/$a/"); done
RULES+=("--include=/runs/arm2_r4/" "--include=/runs/arm2_r8/" "--include=/runs/arm2_preflight/"
        "--exclude=*")

CMD=(rsync -av --prune-empty-dirs "${RULES[@]}" "$SRC" "$DEST_REAL/")

if [ "$DRY" -eq 1 ]; then
    echo "would run:"
    printf '  %q' "${CMD[@]}"; echo
    if command -v rsync >/dev/null 2>&1; then
        rsync -n -av --prune-empty-dirs "${RULES[@]}" "$SRC" "$DEST_REAL/"
    fi
    exit 0
fi

if ! command -v rsync >/dev/null 2>&1; then
    echo "rsync is not installed on this Mac. Equivalent scp commands:"
    for a in $ARMS; do echo "  scp '$REMOTE:$SEGR/outputs/$a/{summary.csv,summary.md,breakdowns.csv,run.json,model.json}' '$DEST_REAL/outputs/$a/'"; done
    exit 4
fi
if ! "${CMD[@]}"; then
    echo "rsync failed (if Prajna lacks rsync, copy with scp: same files as listed in this script)."
    exit 5
fi
echo
echo "copied into $DEST_REAL: $(find "$DEST_REAL" -type f | wc -l | tr -d ' ') files, $(du -sh "$DEST_REAL" | cut -f1)"
