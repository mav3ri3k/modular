#!/usr/bin/env python3
# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
"""Export a minimal custom-op graph proving the Bonsai kernel library path.

The graph doubles a 1024-element FP32 tensor through the registered
``bonsai_probe_scale`` custom op on the GPU device.
"""

import argparse
from pathlib import Path

from max.dtype import DType
from max.graph import DeviceRef, Graph, KernelLibrary, TensorType, ops


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cpu, gpu = DeviceRef.CPU(), DeviceRef.GPU(0)
    library = KernelLibrary([args.kernels.resolve()])
    graph = Graph(
        "bonsai_probe",
        input_types=[TensorType(DType.float32, [1024], device=cpu)],
        kernel_library=library,
    )
    with graph:
        x = graph.inputs[0].tensor.to(gpu)
        scaled = ops.custom(
            name="bonsai_probe_scale",
            device=gpu,
            values=[x],
            out_types=[TensorType(DType.float32, [1024], device=gpu)],
        )
        graph.output(scaled[0].to(cpu))

    # Seed the registered kernel declarations into the owning module so the
    # native cross-compiler resolves the custom op without loading the
    # package itself (mirrors the in-process engine compile path).
    module = graph._mlir_op.block.owner
    library._analysis.seed_kernel_decls(module)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as out:
        module.write_bytecode(out)
    print(f"Exported probe graph: {args.output.stat().st_size} bytes")


if __name__ == "__main__":
    main()
