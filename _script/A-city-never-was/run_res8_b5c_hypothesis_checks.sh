#!/usr/bin/env bash
set -euo pipefail

PAIRWISE_ROOT="${PAIRWISE_ROOT:-/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_classifiier_prob_similarity_by_pair}"
RESOLUTION="${RESOLUTION:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/res8_b5c_hypothesis_checks}"
CITY1="${CITY1:-}"
CITY2="${CITY2:-}"
LIMIT="${LIMIT:-}"

mkdir -p "$OUTPUT_DIR"

label_args=(
  --resolution "$RESOLUTION"
  --pairwise-root "$PAIRWISE_ROOT"
  --only-problematic
  --output-csv "$OUTPUT_DIR/res${RESOLUTION}_shard_label_check.csv"
)

compare_args=(
  --resolution "$RESOLUTION"
  --pairwise-root "$PAIRWISE_ROOT"
  --output-prefix "$OUTPUT_DIR/res${RESOLUTION}_b5c_compare"
)

if [[ -n "$CITY1" ]]; then
  label_args+=(--city1 "$CITY1")
  compare_args+=(--city1 "$CITY1")
fi

if [[ -n "$CITY2" ]]; then
  label_args+=(--city2 "$CITY2")
  compare_args+=(--city2 "$CITY2")
fi

if [[ -n "$LIMIT" ]]; then
  label_args+=(--limit "$LIMIT")
  compare_args+=(--limit "$LIMIT")
fi

echo "Running shard row-label check"
python3 check_b5c_shard_row_labels.py "${label_args[@]}" | tee "$OUTPUT_DIR/res${RESOLUTION}_shard_label_check.stdout.txt"

echo
echo "Running current-vs-row-label comparison"
python3 compare_b5c_current_vs_row_labels.py "${compare_args[@]}" | tee "$OUTPUT_DIR/res${RESOLUTION}_b5c_compare.stdout.txt"

echo
echo "Outputs written to: $OUTPUT_DIR"
echo "  - $OUTPUT_DIR/res${RESOLUTION}_shard_label_check.csv"
echo "  - $OUTPUT_DIR/res${RESOLUTION}_shard_label_check.stdout.txt"
echo "  - $OUTPUT_DIR/res${RESOLUTION}_b5c_compare_diff.csv"
echo "  - $OUTPUT_DIR/res${RESOLUTION}_b5c_compare_leakage.csv"
echo "  - $OUTPUT_DIR/res${RESOLUTION}_b5c_compare.stdout.txt"
