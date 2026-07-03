#!/usr/bin/env bash
set -euo pipefail

PAIRWISE_ROOT="${PAIRWISE_ROOT:-/lustre1/g/geog_pyloo/05_timemachine/_curated/c_city_classifiier_prob_similarity_by_pair}"
RESOLUTION="${RESOLUTION:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-/lustre1/g/geog_pyloo/05_timemachine/_tmp/res8_b5c_hypothesis_checks}"
CITY1="${CITY1:-}"
CITY2="${CITY2:-}"
LIMIT="${LIMIT:-}"
COMPARE_LIMIT="${COMPARE_LIMIT:-$LIMIT}"

mkdir -p "$OUTPUT_DIR"

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
  compare_args+=(--city2 "$CITY2")
fi

if [[ -n "$COMPARE_LIMIT" ]]; then
  compare_args+=(--limit "$COMPARE_LIMIT")
fi

echo "Running current-vs-row-label comparison"
echo "Hint: CITY1/CITY2 accept either raw shard names like 'Hong Kong' or normalized forms like 'hongkong'."
python3 compare_b5c_current_vs_row_labels.py "${compare_args[@]}" | tee "$OUTPUT_DIR/res${RESOLUTION}_b5c_compare.stdout.txt"

echo
echo "Outputs written to: $OUTPUT_DIR"
echo "  - $OUTPUT_DIR/res${RESOLUTION}_b5c_compare_diff.csv"
echo "  - $OUTPUT_DIR/res${RESOLUTION}_b5c_compare_leakage.csv"
echo "  - $OUTPUT_DIR/res${RESOLUTION}_b5c_compare.stdout.txt"
