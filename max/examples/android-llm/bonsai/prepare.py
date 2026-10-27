#!/usr/bin/env python3
# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the Apache License v2.0 with LLVM Exceptions: https://llvm.org/LICENSE.txt
"""Export Bonsai's packed text weights and an FP32-activation MLX reference.

Uses mlx-lm's published Hadamard-aware architecture. The embedding's hardcoded
FP16 output conversion is replaced with FP32; packed codes/scales are unchanged.
This matches the prototype's FP32 arithmetic rather than promising bitwise
agreement with the publisher's FP16 activation execution.
"""

import argparse
import hashlib
import json
import struct
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.prism_hadamard_qwen35 import (
    HadamardQuantizedEmbedding,
    HadamardQuantizedLinear,
    hadamard_rotate,
)


def embedding_fp32(
    self: HadamardQuantizedEmbedding, indices: mx.array
) -> mx.array:
    shape = indices.shape
    indices = indices.reshape(-1)
    value = mx.dequantize(
        self.weight[indices],
        self.scales[indices],
        self.biases[indices],
        group_size=128,
        bits=2,
    ).reshape(*shape, -1)
    return (
        hadamard_rotate(value, self.block, self.signs, inverse=True)
        if self.block
        else value
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prompt", default="What is the capital of France?")
    p.add_argument("--tokens", type=int, default=8)
    p.add_argument("--capacity", type=int, default=128)
    args = p.parse_args()
    if args.capacity < 1 or args.capacity > 8192 or args.tokens < 1:
        raise ValueError("Invalid capacity or token count")
    config = json.loads((args.checkpoint / "config.json").read_text())
    generation = json.loads(
        (args.checkpoint / "generation_config.json").read_text()
    )
    if set(generation["eos_token_id"]) != {248044, 248046}:
        raise ValueError("Unsupported generation stop tokens")
    c = config["text_config"]
    expected = dict(
        hidden_size=2048,
        intermediate_size=6144,
        num_hidden_layers=24,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=256,
        linear_num_key_heads=16,
        linear_num_value_heads=16,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        full_attention_interval=4,
        vocab_size=248320,
        rms_norm_eps=1e-6,
    )
    for key, val in expected.items():
        if c.get(key) != val:
            raise ValueError(f"Unsupported {key}: {c.get(key)}")
    if config["quantization"] != dict(bits=2, group_size=128, mode="affine"):
        raise ValueError("Unsupported packed format")
    if any(r["block"] != 512 for r in config["modules"]):
        raise ValueError("Only block-512 Hadamard packs are implemented")
    HadamardQuantizedEmbedding.__call__ = embedding_fp32
    model, tokenizer = load(
        str(args.checkpoint),
        lazy=True,
        tokenizer_config={"fix_mistral_regex": True},
    )
    model.set_dtype(mx.float32)
    text = tokenizer.apply_chat_template(
        [dict(role="user", content=args.prompt)],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) + args.tokens > args.capacity:
        raise ValueError("Prompt plus generation exceeds capacity")
    args.output.mkdir(parents=True, exist_ok=True)
    names = []
    arrays = []

    def add(name: str, value: mx.array | np.ndarray) -> None:
        value = np.ascontiguousarray(np.asarray(value))
        if value.dtype not in [np.dtype("float32"), np.dtype("uint32")]:
            raise ValueError(f"Unexpected export dtype: {name}: {value.dtype}")
        if value.dtype == np.float32 and not np.isfinite(value).all():
            raise ValueError(f"Non-finite exported tensor: {name}")
        names.append(
            dict(
                name=name,
                shape=list(value.shape),
                dtype=str(value.dtype),
                bytes=value.nbytes,
            )
        )
        arrays.append(value)

    def packed(name: str, module: HadamardQuantizedLinear) -> None:
        add(name + ".weight", module.weight.T)
        params = mx.stack([module.scales.T, module.biases.T], axis=-1)
        add(name + ".params", params)
        add(name + ".signs", module.signs)

    packed("embedding", model.model.embed_tokens)
    add("final_norm", model.model.norm.weight)
    for i, layer in enumerate(model.layers):
        prefix = f"layer.{i}."
        add(prefix + "input_norm", layer.input_layernorm.weight)
        add(prefix + "post_norm", layer.post_attention_layernorm.weight)
        for name in ["gate_proj", "up_proj", "down_proj"]:
            packed(prefix + name, getattr(layer.mlp, name))
        if layer.is_linear:
            a = layer.linear_attn
            for name in ["in_proj_qkv", "in_proj_z", "out_proj"]:
                packed(prefix + name, getattr(a, name))
            add(prefix + "in_proj_a", a.in_proj_a.weight.T)
            add(prefix + "in_proj_b", a.in_proj_b.weight.T)
            add(
                prefix + "decay_constants", mx.concatenate([a.A_log, a.dt_bias])
            )
            add(prefix + "conv", a.conv1d.weight[:, :, 0].T)
            add(prefix + "gated_norm", a.norm.weight)
        else:
            a = layer.self_attn
            for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                packed(prefix + name, getattr(a, name))
            add(prefix + "q_norm", a.q_norm.weight)
            add(prefix + "k_norm", a.k_norm.weight)
    inv = 1 / (10000000 ** (np.arange(0, 64, 2, dtype=np.float32) / 64))
    angles = np.arange(args.capacity, dtype=np.float32)[:, None] * inv[None, :]
    add("rope", np.concatenate([np.cos(angles), np.sin(angles)], axis=1))
    if len(arrays) != 605:
        raise ValueError(f"Incorrect tensor count: {len(arrays)}")
    with (args.output / "weights.bin").open("wb") as f:
        f.write(b"BONSAI01" + struct.pack("<II", len(arrays), args.capacity))
        for a in arrays:
            f.write(struct.pack("<Q", a.nbytes))
            f.write(a.tobytes())
    del arrays
    (args.output / "prompt.txt").write_text(" ".join(map(str, ids)) + "\n")
    cache = model.make_cache()
    generated = []
    inputs = mx.array([ids], dtype=mx.int32)
    with (args.output / "reference.f32").open("wb") as f:
        for _ in range(args.tokens):
            logits = model(inputs, cache=cache)[:, -1, :].astype(mx.float32)
            mx.eval(logits)
            actual = np.asarray(logits)[0]
            actual.astype("<f4").tofile(f)
            token = int(actual.argmax())
            generated.append(token)
            print("REFERENCE", len(generated) - 1, token, flush=True)
            inputs = mx.array([[token]], dtype=mx.int32)
    manifest = dict(
        model="prism-ml/Ternary-Bonsai-2-2B-mlx-2bit",
        revision="1c6cbae21f8c38830ce337972b46fdae7b18eaaa",
        checkpoint_sha256=hashlib.file_digest(
            (args.checkpoint / "model.safetensors").open("rb"), "sha256"
        ).hexdigest(),
        reference="mlx-lm FP32 activations; packed codes/scales preserved",
        prompt=args.prompt,
        prompt_ids=ids,
        expected_ids=generated,
        expected_text=tokenizer.decode(generated),
        capacity=args.capacity,
        generation_tokens=args.tokens,
        tensors=names,
    )
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print("REFERENCE TEXT", manifest["expected_text"], flush=True)


if __name__ == "__main__":
    main()
