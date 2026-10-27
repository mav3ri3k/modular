#!/usr/bin/env python3
# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the Apache License v2.0 with LLVM Exceptions: https://llvm.org/LICENSE.txt
"""Build a resident-state Bonsai decode graph using MAX ops and custom kernels.

Packed weights remain external constants. State is explicit mutable BufferType
input, so MAX tracks effects and allocations. No custom whole-model forward op
or hidden runtime state is used. The same single-token graph performs prefill.
"""

import argparse
import json
from pathlib import Path

from max import mlir
from max.dtype import DType
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    Graph,
    KernelLibrary,
    TensorType,
    TensorValue,
    ops,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--kernels", type=Path, required=True)
    p.add_argument(
        "--analysis-kernels",
        type=Path,
        help="Same source compiled with the Python frontend toolchain",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--layers", type=int, default=24)
    args = p.parse_args()
    if not 1 <= args.layers <= 24:
        p.error("Expected 1-24 layers")
    metadata = json.loads((args.model / "manifest.json").read_text())
    capacity = metadata["capacity"]
    cpu, gpu = DeviceRef.CPU(), DeviceRef.GPU(0)
    library = KernelLibrary([(args.analysis_kernels or args.kernels).resolve()])
    state_sizes = []
    for layer in range(args.layers):
        state_sizes.extend(
            [512 * capacity, 512 * capacity] if (layer + 1) % 4
            == 0 else [3 * 6144, 16 * 128 * 128]
        )
    graph = Graph(
        "bonsai_decode",
        input_types=[
            TensorType(DType.int32, [1], device=cpu),
            TensorType(DType.int32, [1], device=cpu),
            *[BufferType(DType.float32, [n], device=gpu) for n in state_sizes],
        ],
        kernel_library=library,
    )
    with graph:
        token, position = [v.tensor for v in graph.inputs[:2]]
        states = [v.buffer for v in graph.inputs[2:]]
        weights = []
        for i, tensor in enumerate(metadata["tensors"]):
            compact_scale = tensor["name"].endswith(".params")
            # ARM CPU graph ops reject float16. Carry the packed half bits as
            # uint16 storage; the GPU custom op resolves their Float16 view.
            dtype = DType.uint16 if compact_scale else (
                DType.uint32 if tensor["dtype"] == "uint32" else DType.float32
            )
            elements = tensor["bytes"] // (8 if compact_scale else 4)
            weights.append(
                ops.constant_external(
                    f"w{i:03d}", TensorType(dtype, [elements], device=cpu)
                ).to(gpu)
            )

        def custom(
            name: str,
            inputs: list[TensorValue | BufferValue],
            sizes: list[int],
            *,
            inplace: bool = False,
            dtypes: list[DType] | None = None,
            **params: int,
        ) -> list[TensorValue]:
            return [
                v.tensor
                for v in (ops.inplace_custom if inplace else ops.custom)(
                    "bonsai_" + name,
                    device=gpu,
                    values=inputs,
                    out_types=[
                        TensorType(dtype, [n], device=gpu)
                        for n, dtype in zip(
                            sizes, dtypes or [DType.float32] * len(sizes)
                        )
                    ],
                    parameters=params,
                )
            ]

        def rotate(
            x: TensorValue, signs: TensorValue, width: int, inverse: int = 0
        ) -> list[TensorValue]:
            return custom(
                "hadamard",
                [x, signs],
                [width, width // 2],
                dtypes=[DType.float32, DType.uint32],
                width=width,
                inverse=inverse,
            )

        def project(
            q: TensorValue, slot: int, width: int, reduction: int = 2048
        ) -> TensorValue:
            if width == 248320:
                return custom(
                    "packed_head",
                    [q, weights[slot], weights[slot + 1]],
                    [width],
                    width=width,
                    reduction=reduction,
                )[0]
            partial_groups = (reduction + 1023) // 1024
            partials = custom(
                "packed_linear",
                [q, weights[slot], weights[slot + 1]],
                [width * partial_groups],
                width=width,
                reduction=reduction,
            )[0]
            # Let the ordinary MAX reduction compiler handle the FP32 join.
            return ops.sum(
                partials.reshape([partial_groups, width]), axis=0
            ).reshape([width])

        raw = custom("embedding", [weights[0], weights[1], token], [2048])[0]
        x = rotate(raw, weights[2], 2048, inverse=1)[0]
        for layer in range(args.layers):
            off = 4 + layer * 25
            history, state = states[2 * layer : 2 * layer + 2]
            z = ops.rms_norm(x, weights[off], 1e-6)
            q = rotate(z, weights[off + 13], 2048)[1]
            if (layer + 1) % 4 != 0:
                qkv = project(q, off + 11, 6144)
                gate = project(q, off + 14, 2048)
                partials = custom(
                    "gate_partials",
                    [z, weights[off + 20], weights[off + 21]],
                    [1024],
                )[0]
                convolved = custom(
                    "convolve",
                    [qkv, history, weights[off + 23]],
                    [6144],
                    inplace=True,
                )[0]
                qkv = custom("normalize_qk", [convolved], [6144])[0]
                gates = custom(
                    "delta_gates", [partials, weights[off + 22]], [32]
                )[0]
                recurrent = custom(
                    "delta_update", [qkv, gates, state], [2048], inplace=True
                )[0]
                z = custom(
                    "gated_norm", [recurrent, gate, weights[off + 24]], [2048]
                )[0]
                q = rotate(z, weights[off + 19], 2048)[1]
                y = project(q, off + 17, 2048)
            else:
                qg = project(q, off + 11, 4096)
                k = project(q, off + 14, 512)
                v = project(q, off + 17, 512)
                q = custom(
                    "head_rope",
                    [qg, weights[off + 23], weights[604], position],
                    [2048],
                    width=2048,
                    mode=0,
                )[0]
                k = custom(
                    "head_rope",
                    [k, weights[off + 24], weights[604], position],
                    [512],
                    width=512,
                    mode=1,
                )[0]
                custom(
                    "cache_kv",
                    [k, v, history, state, position],
                    [],
                    inplace=True,
                )
                scores = custom(
                    "attention_scores",
                    [q, history[...], position],
                    [8 * capacity],
                    capacity=capacity,
                )[0]
                z = custom(
                    "attention_values",
                    [scores, state[...], qg, position],
                    [2048],
                    capacity=capacity,
                )[0]
                q = rotate(z, weights[off + 22], 2048)[1]
                y = project(q, off + 20, 2048)
            x = x + y
            z = ops.rms_norm(x, weights[off + 1], 1e-6)
            q = rotate(z, weights[off + 4], 2048)[1]
            gate = project(q, off + 2, 6144)
            up = project(q, off + 5, 6144)
            z = ops.silu(gate) * up
            q = rotate(z, weights[off + 10], 6144)[1]
            x = x + project(q, off + 8, 2048, reduction=6144)
        x = ops.rms_norm(x, weights[3], 1e-6)
        q = rotate(x, weights[2], 2048)[1]
        graph.output(project(q, 0, 248320).to(cpu))

    module = graph._mlir_op.block.owner
    # The normal native pipeline imports and verifies declarations using its
    # own package. Python verified these ops with the same source compiled for
    # its frontend; precompiled package checksums are never bypassed.
    graph._mlir_op.attributes["_kernel_library_paths"] = mlir.ArrayAttr.get(
        [mlir.StringAttr.get(str(args.kernels.resolve()))]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        module.write_bytecode(stream)
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "layers": args.layers,
                "capacity": capacity,
                "state_sizes": state_sizes,
                "model": metadata["model"],
                "checkpoint_sha256": metadata["checkpoint_sha256"],
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"Exported {args.layers}-layer Bonsai MAX graph:"
        f" {args.output.stat().st_size} bytes"
    )


if __name__ == "__main__":
    main()
