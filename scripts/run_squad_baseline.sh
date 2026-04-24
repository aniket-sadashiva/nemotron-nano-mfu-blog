#!/usr/bin/env bash
# SQuAD fine-tune, BASELINE (stock HF for-loop over 128 experts), 100 steps.
# This is the reference curve the gmm_patch run is compared against.
#
# Usage:
#   bash scripts/run_squad_baseline.sh
#
# Env vars: see scripts/_common.sh
#
# Output: results/squad_baseline/validation.jsonl
#         results/squad_baseline/training.jsonl

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"
check_prereqs

print_banner "SQuAD fine-tune, baseline (100 steps)"

run_in_container \
    nemo_automodel/recipes/llm/train_ft.py \
    /workspace/configs/squad_baseline.yaml \
    "$@"

echo ""
echo "Done. Validation curve:"
ls -la "${REPO_DIR}/results/squad_baseline/"*.jsonl 2>/dev/null || echo "(no JSONLs found)"
