#!/usr/bin/env python3
# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
"""Export SmolLM2 through ordinary MAX operators for Android graph validation.

Constants retain the checkpoint's BF16 values promoted to FP32. Cache tensors
are explicit graph inputs/outputs; this exercises prefill and decode without a
custom forward operator or an example-specific graph executor.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops
from safetensors.torch import load_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence", type=int, default=2)
    parser.add_argument("--capacity", type=int, default=16)
    parser.add_argument("--layers", type=int, default=30)
    parser.add_argument(
        "--intermediates",
        action="store_true",
        help="Include first-layer intermediate tensors for numerical diagnosis",
    )
    args = parser.parse_args()
    config = json.loads((args.checkpoint / "config.json").read_text())
    if not 1 <= args.layers <= config["num_hidden_layers"]:
        parser.error("Invalid layer count")
    if not 1 <= args.sequence <= args.capacity:
        parser.error("Sequence must fit capacity")
    weights = load_file(args.checkpoint / "model.safetensors")
    s, c, layers = args.sequence, args.capacity, args.layers
    h = config["hidden_size"]
    heads, kv_heads = (
        config["num_attention_heads"],
        config["num_key_value_heads"],
    )
    d = h // heads
    cpu, gpu = DeviceRef.CPU(), DeviceRef.GPU(0)
    graph = Graph(
        f"smollm2_{layers}_layers_s{s}",
        input_types=[
            TensorType(DType.int64, [s], device=cpu),
            TensorType(DType.int64, [s], device=cpu),
            TensorType(DType.float32, [layers, 2, c, kv_heads, d], device=cpu),
        ],
    )
    with graph:
        ids, positions, past = [value.tensor.to(gpu) for value in graph.inputs]

        def weight(name: str) -> TensorValue:
            return ops.constant(weights[name].float().numpy(), device=gpu)

        def linear(x: TensorValue, name: str) -> TensorValue:
            return ops.matmul(x, weight(name).transpose(0, 1))

        inv = 1 / (
            config["rope_theta"] ** (np.arange(0, d, 2, dtype=np.float32) / d)
        )
        angles = np.arange(c, dtype=np.float32)[:, None] * inv[None, :]
        angles = np.concatenate([angles, angles], axis=1)
        cos = ops.gather(
            ops.constant(np.cos(angles), device=gpu), positions, axis=0
        ).reshape([s, 1, d])
        sin = ops.gather(
            ops.constant(np.sin(angles), device=gpu), positions, axis=0
        ).reshape([s, 1, d])

        def rope(x: TensorValue) -> TensorValue:
            return (
                x * cos
                + ops.concat([-x[:, :, d // 2 :], x[:, :, : d // 2]], axis=2)
                * sin
            )

        def repeat(x: TensorValue) -> TensorValue:
            return ops.broadcast_to(
                x.transpose(0, 1).reshape([kv_heads, 1, c, d]),
                [kv_heads, heads // kv_heads, c, d],
            ).reshape([heads, c, d])

        mask_data = np.where(
            np.arange(c)[:, None] >= np.arange(c)[None, :],
            np.float32(0),
            np.finfo(np.float32).min,
        )
        mask = ops.gather(
            ops.constant(mask_data, device=gpu), positions, axis=0
        )
        embedding = weight("model.embed_tokens.weight")
        x = ops.gather(embedding, ids, axis=0)
        caches = []
        intermediates = []

        def observe(name: str, value: TensorValue) -> TensorValue:
            if args.intermediates and layer == 0:
                intermediates.append((name, value))
            return value

        for layer in range(layers):
            prefix = f"model.layers.{layer}."
            z = ops.rms_norm(
                x,
                weight(prefix + "input_layernorm.weight"),
                config["rms_norm_eps"],
            )
            q = rope(
                linear(z, prefix + "self_attn.q_proj.weight").reshape(
                    [s, heads, d]
                )
            ).transpose(0, 1)
            k = rope(
                linear(z, prefix + "self_attn.k_proj.weight").reshape(
                    [s, kv_heads, d]
                )
            )
            v = linear(z, prefix + "self_attn.v_proj.weight").reshape(
                [s, kv_heads, d]
            )
            k = ops.scatter_nd(past[layer, 0], k, positions.reshape([s, 1]))
            v = ops.scatter_nd(past[layer, 1], v, positions.reshape([s, 1]))
            caches.append(ops.stack([k, v], axis=0))
            scores = observe(
                "scores",
                ops.matmul(q, repeat(k).transpose(1, 2)) * (d**-0.5) + mask,
            )
            probabilities = observe("probabilities", ops.softmax(scores))
            y = (
                ops.matmul(probabilities, repeat(v))
                .transpose(0, 1)
                .reshape([s, h])
            )
            y = observe("attention", y)
            x = observe(
                "post_attention",
                x + linear(y, prefix + "self_attn.o_proj.weight"),
            )
            z = ops.rms_norm(
                x,
                weight(prefix + "post_attention_layernorm.weight"),
                config["rms_norm_eps"],
            )
            z = observe("mlp_input", z)
            x = x + linear(
                ops.silu(linear(z, prefix + "mlp.gate_proj.weight"))
                * linear(z, prefix + "mlp.up_proj.weight"),
                prefix + "mlp.down_proj.weight",
            )
            x = observe("block_output", x)
        x = ops.rms_norm(x, weight("model.norm.weight"), config["rms_norm_eps"])
        if args.intermediates:
            intermediates.append(("final_norm", x))
        logits = ops.matmul(x[s - 1 : s], embedding.transpose(0, 1))
        graph.output(
            logits.to(cpu),
            ops.stack(caches, axis=0).to(cpu),
            *[value.to(cpu) for _, value in intermediates],
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as out:
        graph._mlir_op.write_bytecode(out)
    if args.intermediates:
        args.output.with_suffix(".outputs.json").write_text(
            json.dumps(
                ["logits", "cache", *[name for name, _ in intermediates]],
                indent=2,
            )
            + "\n"
        )
    print(f"Exported {layers} layers: {args.output.stat().st_size} bytes")


if __name__ == "__main__":
    main()
