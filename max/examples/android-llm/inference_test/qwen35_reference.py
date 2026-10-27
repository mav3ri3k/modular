#!/usr/bin/env python3
# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the Apache License v2.0 with LLVM Exceptions: https://llvm.org/LICENSE.txt
"""Independent transformers FP32 Qwen3.5 text reference from tested GGUF."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFReader, dequantize
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5TextRotaryEmbedding,
)

p = argparse.ArgumentParser(description=__doc__)
for name in ["gguf", "tokens", "config", "output"]:
    p.add_argument("--" + name, type=Path, required=True)
a = p.parse_args()
torch.set_num_threads(4)
c = Qwen3_5TextConfig(**json.loads(a.config.read_text())["text_config"])
c._attn_implementation = "eager"
with torch.device("meta"):
    model = Qwen3_5ForCausalLM(c)
lookup = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
}
names = {
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
    "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
    "attn_gate.weight": "linear_attn.in_proj_z.weight",
    "ssm_out.weight": "linear_attn.out_proj.weight",
    "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
    "ssm_beta.weight": "linear_attn.in_proj_b.weight",
    "ssm_a": "linear_attn.A_log",
    "ssm_dt.bias": "linear_attn.dt_bias",
    "ssm_conv1d.weight": "linear_attn.conv1d.weight",
    "ssm_norm.weight": "linear_attn.norm.weight",
}
state = {}
for t in GGUFReader(a.gguf).tensors:
    name = lookup.get(t.name)
    if name is None:
        parts = t.name.split(".")
        layer = int(parts[1])
        if layer >= 24:
            continue
        kind = ".".join(parts[2:])
        name = f"model.layers.{layer}.{names[kind]}"
    x = dequantize(t.data, t.tensor_type).copy()
    if name.endswith("linear_attn.A_log"):
        x = np.log(-x)
    if name.endswith("conv1d.weight"):
        x = x[:, None, :]
    if name.endswith("norm.weight") and not name.endswith(
        "linear_attn.norm.weight"
    ):
        x = x - 1
    state[name] = torch.from_numpy(x)
state["lm_head.weight"] = state["model.embed_tokens.weight"]
model.load_state_dict(state, assign=True)
del state
model.model.rotary_emb = Qwen3_5TextRotaryEmbedding(c)
model.eval()
ids = list(map(int, a.tokens.read_text().split()))
with torch.no_grad():
    # Token-by-token execution exercises the recurrent update path, like the phone.
    cache = None
    outputs = []
    for token in ids:
        result = model(
            torch.tensor([[token]]), past_key_values=cache, use_cache=True
        )
        cache = result.past_key_values
        outputs.append(result.logits[0, -1].float().numpy())
np.stack(outputs).astype("<f4").tofile(a.output)
print("Dumped", len(outputs), "reference steps", flush=True)
