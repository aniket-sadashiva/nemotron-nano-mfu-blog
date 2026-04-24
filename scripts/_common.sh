# Shared helpers for all run scripts.
# Sourced by run_mfu_bench.sh, run_squad_gmm.sh, run_squad_baseline.sh.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# -- overridable knobs --------------------------------------------------------
DOCKER_IMAGE="${DOCKER_IMAGE:-nvcr.io/nvidia/nemo-automodel:25.11}"
NGPUS="${NGPUS:-8}"
# Host path for HF cache. First run will download weights + dataset into this
# directory (~60 GB for Nemotron-3-Nano-30B-A3B-BF16). Override if short on
# space under ${HOME}.
HF_CACHE_DIR="${HF_CACHE_DIR:-${HOME}/.cache/huggingface}"
# Optional: if set, the container also binds this read-only so configs can
# reference local weight paths (skip the 60 GB HF download).
EXTRA_MOUNT="${EXTRA_MOUNT:-}"
# Path to the Automodel source tree inside the container. The NGC image ships
# it at /opt/Automodel; we bind-mount only our patches on top, no clone needed.
AUTOMODEL_IN_CONTAINER="/opt/Automodel"
# -----------------------------------------------------------------------------

die()  { echo "ERROR: $*" >&2; exit 1; }

check_prereqs() {
    command -v docker >/dev/null || die "docker not found on PATH"
    [[ -f "${REPO_DIR}/patch/gmm_patch.py" ]]   || die "patch/gmm_patch.py missing"
    [[ -f "${REPO_DIR}/patch/flops_utils.py" ]] || die "patch/flops_utils.py missing"
    mkdir -p "${HF_CACHE_DIR}" "${REPO_DIR}/results"
}

# Build the docker run cmd as an array. Callers append the recipe invocation.
#   $1 - recipe script (e.g. nemo_automodel/recipes/llm/benchmark.py)
#   $2 - config yaml path inside container (e.g. /workspace/configs/mfu_bench.yaml)
#   $3..$N - extra torchrun args
run_in_container() {
    local recipe="$1"; shift
    local config="$1"; shift
    local extra_mount_args=()
    if [[ -n "${EXTRA_MOUNT}" ]]; then
        extra_mount_args=(-v "${EXTRA_MOUNT}:${EXTRA_MOUNT}:ro")
    fi
    docker run --gpus all --ipc=host \
        --ulimit memlock=-1 --ulimit stack=67108864 \
        -v "${REPO_DIR}:/workspace" \
        -v "${REPO_DIR}/patch/flops_utils.py:${AUTOMODEL_IN_CONTAINER}/nemo_automodel/components/utils/flops_utils.py:ro" \
        -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
        "${extra_mount_args[@]}" \
        -e NCCL_PROTO=Simple \
        -e NCCL_NVLS_ENABLE=0 \
        -e TORCH_NCCL_AVOID_RECORD_STREAMS=1 \
        -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        -e WANDB_MODE=disabled \
        -e SKIP_OPTIMIZER_SAVE=1 \
        --rm "${DOCKER_IMAGE}" bash -lc "
            set -euo pipefail
            # Put gmm_patch on PYTHONPATH so configs can resolve '_target_: gmm_patch.*'
            export PYTHONPATH=/workspace/patch:\${PYTHONPATH:-}
            cd ${AUTOMODEL_IN_CONTAINER}
            torchrun --nproc-per-node=${NGPUS} --nnodes=1 \
                ${recipe} \
                --config ${config} \
                $*
        "
}

print_banner() {
    cat <<EOF
============================================================
 Run:         $1
 Docker image: ${DOCKER_IMAGE}
 GPUs:        ${NGPUS}
 HF cache:    ${HF_CACHE_DIR}
 Repo:        ${REPO_DIR}
 Extra mount: ${EXTRA_MOUNT:-<none>}
============================================================
EOF
}
