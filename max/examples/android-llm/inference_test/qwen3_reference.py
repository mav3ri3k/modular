#!/usr/bin/env python3
# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the Apache License v2.0 with LLVM Exceptions: https://llvm.org/LICENSE.txt
"""Independent FP32 PyTorch Qwen3 reference from exactly the tested GGUF values.

llama.cpp CPU quantized matvecs also quantize activations; this separates that
arithmetic difference from bugs in the Mojo packed weight decoding/architecture.
"""

import argparse
from pathlib import Path

import torch
from gguf import GGUFReader, dequantize
from transformers import Qwen3Config, Qwen3ForCausalLM

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--gguf", type=Path, required=True)
p.add_argument("--tokens", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
torch.set_num_threads(4)
r = GGUFReader(a.gguf)
config = Qwen3Config(
    hidden_size=2048,
    intermediate_size=6144,
    num_hidden_layers=28,
    num_attention_heads=16,
    num_key_value_heads=8,
    head_dim=128,
    vocab_size=151936,
    rms_norm_eps=1e-6,
    rope_theta=1e6,
    tie_word_embeddings=False,
    attn_implementation="eager",
)
with torch.device("meta"):
    model = Qwen3ForCausalLM(config)
lookup = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output.weight": "lm_head.weight",
    "output_norm.weight": "model.norm.weight",
}
names = {
    "attn_norm": "input_layernorm",
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "attn_q_norm": "self_attn.q_norm",
    "attn_k_norm": "self_attn.k_norm",
    "ffn_norm": "post_attention_layernorm",
    "ffn_gate": "mlp.gate_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
}
state = {}
for t in r.tensors:
    name = lookup.get(t.name)
    if name is None:
        _, layer, kind, _ = t.name.split(".")
        name = f"model.layers.{layer}.{names[kind]}.weight"
    state[name] = torch.from_numpy(dequantize(t.data, t.tensor_type).copy())
model.load_state_dict(state, assign=True)
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

model.model.rotary_emb = Qwen3RotaryEmbedding(config)
model.eval()
del state
ids = list(map(int, a.tokens.read_text().split()))
with torch.no_grad():
    logits = (
        model(torch.tensor([ids]), use_cache=False).logits[0].float().numpy()
    )
logits.astype("<f4").tofile(a.output)
print("Dumped", logits.shape, flush=True)
