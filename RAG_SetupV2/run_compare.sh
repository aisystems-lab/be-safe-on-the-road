set -euo pipefail

export JUDGE_PROVIDER="${JUDGE_PROVIDER:-openai}"

TRIALS="${COMPARE_TRIALS:-3}"

COOLDOWN="${COMPARE_COOLDOWN:-0.5}"

python compare_models.py --trials "$TRIALS" --cooldown "$COOLDOWN" "$@"
