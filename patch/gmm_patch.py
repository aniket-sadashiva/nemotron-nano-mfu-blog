"""
GroupedGEMM patch for stock HF NemotronHForCausalLM.

Reference: ~/.cache/huggingface/modules/transformers_modules/nvidia/
  NVIDIA_hyphen_Nemotron_hyphen_3_hyphen_Nano_hyphen_30B_hyphen_A3B_hyphen_BF16/
  378df16e4b54901a3f514f38ea9a34db9d061634/modeling_nemotron_h.py

=== OG HF CODE ===

    # NemotronHMLP.forward (line 814-815) — what each expert computes:
    def forward(self, x):
        return self.down_proj(self.act_fn(self.up_proj(x)))

    # NemotronHMOE.forward (line 864-871) — outer MoE layer (WE KEEP THIS IDENTICAL):
    def forward(self, hidden_states):
        residuals = hidden_states                                        # line 865
        orig_shape = hidden_states.shape                                 # line 866
        topk_indices, topk_weights = self.gate(hidden_states)            # line 867
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])  # line 868
        hidden_states = self.moe(hidden_states, topk_indices, topk_weights).view(*orig_shape)  # line 869
        hidden_states = hidden_states + self.shared_experts(residuals)   # line 870
        return hidden_states                                             # line 871

    # NemotronHMOE.moe (line 833-862) — THE SLOW FOR-LOOP WE REPLACE:
    def moe(self, hidden_states, topk_indices, topk_weights):
        final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
        expert_mask = F.one_hot(topk_indices, num_classes=len(self.experts)).permute(2,0,1)
        for expert_idx in range(len(self.experts)):                       # <-- 128 sequential iterations
            expert = self.experts[expert_idx]
            token_indices, weight_indices = torch.where(expert_mask[expert_idx])
            if token_indices.numel() > 0:
                expert_output = expert(hidden_states[token_indices])       # = down_proj(act_fn(up_proj(x)))
                final_hidden_states.index_add_(0, token_indices,
                    expert_output * topk_weights[token_indices, weight_indices].unsqueeze(-1))
        return final_hidden_states.type(hidden_states.dtype)

=== CHANGES ===

    GroupedMoEExperts.forward replaces ONLY the moe() for-loop above with:
      1. te_perm.moe_permute  — sort tokens by expert      (replaces one_hot + torch.where)
      2. ops.gmm(x, up_w)    — all 128 up_proj at once     (replaces expert.up_proj)
      3. act_fn(h)            — same activation as HF       (replaces expert.act_fn, read from config)
      4. ops.gmm(h, down_w)  — all 128 down_proj at once   (replaces expert.down_proj)
      5. te_perm.moe_unpermute — unsort + apply weights     (replaces index_add_ + weight multiply)

    GroupedMoEWrapper.forward is a COPY of NemotronHMOE.forward (lines 864-871).
    The ONLY difference: line 869 calls self.experts() instead of self.moe().
    gate, shared_experts, residuals, view, everything else is IDENTICAL.
"""

import logging
import torch
import torch.nn as nn
from transformers.activations import ACT2FN
from grouped_gemm import ops
import transformer_engine.pytorch.permutation as te_perm
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.distributed.fsdp import fully_shard
from torch.distributed.fsdp._fully_shard import MixedPrecisionPolicy

try:
    from torch.distributed.tensor import DTensor
except ImportError:
    DTensor = None

logger = logging.getLogger(__name__)

def _local(t):
    """Unwrap FSDP DTensor to local tensor."""
    return t.to_local() if DTensor is not None and isinstance(t, DTensor) else t


class GroupedMoEExperts(nn.Module):
    """Replaces NemotronHMOE.moe() for-loop (line 833-862) with batched ops.gmm."""

    def __init__(self, n_experts, up_projs, down_projs, act_fn):
        super().__init__()
        self.n_experts = n_experts
        self.up_projs = nn.Parameter(up_projs)     # [128, 2688, 1856] = stacked up_proj.weight.T
        self.down_projs = nn.Parameter(down_projs)  # [128, 1856, 2688] = stacked down_proj.weight.T
        self.act_fn = act_fn                        # ACT2FN[config.mlp_hidden_act] = relu2

    def forward(self, x, weights, indices):
        """Replaces: for expert_idx in range(128): expert(hidden_states[token_indices])"""
        up_w, down_w = _local(self.up_projs), _local(self.down_projs)
        idx = _local(indices).to(torch.int32)
        wts = _local(weights)

        # Step 1: Sort tokens by expert (replaces one_hot + torch.where in HF moe())
        permuted_x, row_map = te_perm.moe_permute(x, idx, num_out_tokens=-1, max_token_num=x.shape[0], map_type="index")
        counts = torch.bincount(idx.reshape(-1).to(torch.int64), minlength=self.n_experts).cpu()

        # Step 2: up_proj for all 128 experts at once (replaces: expert.up_proj(x))
        h = ops.gmm(permuted_x.to(up_w.dtype), up_w, counts, trans_b=False)

        # Step 3: activation (replaces: expert.act_fn(...) — relu2, read from HF config)
        h = self.act_fn(h)

        # Step 4: down_proj for all 128 experts at once (replaces: expert.down_proj(...))
        out = ops.gmm(h, down_w, counts, trans_b=False)

        # Step 5: Unsort + apply routing weights (replaces: index_add_ + weight multiply)
        return te_perm.moe_unpermute(out, row_map, merging_probs=wts.to(torch.float32), map_type="index").to(x.dtype)


class GroupedMoEWrapper(nn.Module):
    """COPY of NemotronHMOE.forward (line 864-871). Only change: line 869 uses GroupedMoEExperts."""

    def __init__(self, gate, experts, shared_experts):
        super().__init__()
        self.gate = gate                     # HF NemotronHTopkRouter — unchanged
        self.experts = experts               # GroupedMoEExperts — replaces for-loop
        self.shared_experts = shared_experts  # HF NemotronHMLP — unchanged

    def forward(self, hidden_states):
        residuals = hidden_states                                                        # line 865
        orig_shape = hidden_states.shape                                                 # line 866
        topk_indices, topk_weights = self.gate(hidden_states)                            # line 867
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])                  # line 868
        hidden_states = self.experts(hidden_states, topk_weights, topk_indices)          # line 869 (gmm)
        hidden_states = hidden_states.view(*orig_shape) + self.shared_experts(residuals)  # line 870
        return hidden_states                                                             # line 871


def patch_moe_with_gmm(model):
    """Walk model, stack expert weights into [128, in, out] tensors, replace MoE modules."""
    act_fn = ACT2FN[model.config.mlp_hidden_act]  # "relu2" for Nemotron-3-Nano (NOT silu)
    backbone = model.backbone if hasattr(model, "backbone") else model.model
    n = 0
    for layer in backbone.layers:
        if not (hasattr(layer, "block_type") and layer.block_type == "moe"):
            continue
        hf_moe = layer.mixer
        experts = list(hf_moe.experts)
        # HF nn.Linear.weight is [out, in]. ops.gmm needs [in, out]. So transpose.
        up = torch.stack([e.up_proj.weight.detach().T.contiguous() for e in experts])
        down = torch.stack([e.down_proj.weight.detach().T.contiguous() for e in experts])
        layer.mixer = GroupedMoEWrapper(
            gate=hf_moe.gate,
            experts=GroupedMoEExperts(len(experts), up, down, act_fn),
            shared_experts=hf_moe.shared_experts,
        )
        del hf_moe, experts
        n += 1
    logger.info(f"[gmm_patch] Replaced {n} MoE layers")


def parallelize_nemotronh_gmm(model, world_mesh, *, dp_axis_names,
                              activation_checkpointing=True, reshard_after_forward=True, **_):
    """Patch MoE + AC + FSDP."""
    backbone = model.backbone if hasattr(model, "backbone") else model.model
    patch_moe_with_gmm(model)
    if activation_checkpointing:
        for lid, block in backbone.layers.named_children():
            backbone.layers.register_module(lid, checkpoint_wrapper(block, preserve_rng_state=True))
    model.to(torch.bfloat16)
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, output_dtype=torch.bfloat16)
    mesh = world_mesh[tuple(dp_axis_names)]
    for block in backbone.layers.children():
        fully_shard(block, mesh=mesh, reshard_after_forward=reshard_after_forward, mp_policy=mp)
    fully_shard(backbone, mesh=mesh, reshard_after_forward=reshard_after_forward, mp_policy=mp)
    fully_shard(model, mesh=mesh, reshard_after_forward=False, mp_policy=mp)
