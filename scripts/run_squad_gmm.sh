#!/usr/bin/env bash
# SQuAD fine-tune with gmm_patch, 100 steps. Produces validation loss curve.
#
# Usage:
#   bash scripts/run_squad_gmm.sh
#
# Env vars: see scripts/_common.sh
#
# Output: results/squad_gmm/validation.jsonl  (val loss every 10 steps)
#         results/squad_gmm/training.jsonl    (train loss every step)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"
check_prereqs

print_banner "SQuAD fine-tune, gmm_patch (100 steps)"

run_in_container \
    nemo_automodel/recipes/llm/train_ft.py \
    /workspace/configs/squad_gmm.yaml \
    "$@"

echo ""
echo "Done. Validation curve:"
ls -la "${REPO_DIR}/results/squad_gmm/"*.jsonl 2>/dev/null || echo "(no JSONLs found)"
