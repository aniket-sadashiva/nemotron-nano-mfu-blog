#!/usr/bin/env bash
# Reproduce the ~19.7% MFU result on 8x H100.
#
# Usage:
#   bash scripts/run_mfu_bench.sh
#
# Env vars (all optional):
#   DOCKER_IMAGE  - docker image            (default: nvcr.io/nvidia/nemo-automodel:25.11)
#   NGPUS         - GPUs                    (default: 8)
#   HF_CACHE_DIR  - host HF cache           (default: ${HOME}/.cache/huggingface)
#   EXTRA_MOUNT   - extra read-only mount   (e.g. /mnt/weights) - useful if you already
#                                             have the 60 GB Nemotron-3-Nano weights somewhere
#                                             and want to pass a local path via --model.pretrained_model_name_or_path
#
# Output: results/mfu_bench.json  (peak MFU + timings)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"
check_prereqs

print_banner "MFU benchmark (Nemotron-3-Nano, seq=20480, MBS=1, Adagrad, compile)"

run_in_container \
    nemo_automodel/recipes/llm/benchmark.py \
    /workspace/configs/mfu_bench.yaml \
    --benchmark.json_output_path=/workspace/results/mfu_bench.json \
    "$@"

echo ""
echo "Done. Summary:"
grep -E '"avg_mfu_percent"|"avg_iter_time_seconds"|"tflops_per_gpu"' \
    "${REPO_DIR}/results/mfu_bench.json" || true
