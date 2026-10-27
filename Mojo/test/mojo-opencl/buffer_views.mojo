# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the Apache License v2.0 with LLVM Exceptions: https://llvm.org/LICENSE.txt
# RUN: %mojo-build-no-debug-no-assert %s --target-triple aarch64-unknown-opencl --target-cpu generic --emit asm -I %S/../../../max/mojo -o %t.cl
# RUN: FileCheck %s --input-file=%t.cl
# CHECK: __kernel void mojo_kernel_{{[0-9]+}}(__global uchar *{{.*}}_base, ulong {{.*}}_offset
# CHECK: _base +

from std.memory import Pointer
from max.gpu import global_idx


@export("view_copy")
def view_copy(
    source: Pointer[Float32, MutAnyOrigin],
    output: Pointer[Float32, MutAnyOrigin],
    count: Int32,
) abi("C"):
    """Copies through offset pointer arguments with a scalar argument after them.
    """
    var i = global_idx.x
    if i < Int(count):
        output[unsafe_offset=i] = source[unsafe_offset=i] * 2 + 1
