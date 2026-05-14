#!/usr/bin/env bash
# Wipe the comparison output files so the next run starts from scratch.
# Use this when you want to reset the append-mode table.
#
# Files removed (only these four):
#   data/model_comparison.csv
#   data/model_comparison.md
#   data/model_comparison.jsonl
#   data/model_comparison_full.csv
#
# Everything else in data/ (the Chroma index, eval results, rag_meta) is
# untouched.
set -euo pipefail

cd "$(dirname "$0")"

REMOVED=0
for f in \
  data/model_comparison.csv \
  data/model_comparison.md \
  data/model_comparison.jsonl \
  data/model_comparison_full.csv
do
  if [[ -f "$f" ]]; then
    rm "$f"
    echo "removed $f"
    REMOVED=$((REMOVED + 1))
  fi
done

if [[ $REMOVED -eq 0 ]]; then
  echo "No comparison files to remove; you're already clean."
else
  echo "OK — cleared $REMOVED file(s). Next run starts fresh."
fi
