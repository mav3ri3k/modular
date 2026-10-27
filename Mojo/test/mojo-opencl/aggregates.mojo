# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the Apache License v2.0 with LLVM Exceptions: https://llvm.org/LICENSE.txt
# RUN: %mojo-build-no-debug-no-assert %s --target-triple aarch64-unknown-opencl --target-cpu generic --emit llvm -o %t.ll
# RUN: FileCheck %s --check-prefix=IR --input-file=%t.ll
# RUN: %mojo-build-no-debug-no-assert %s --target-triple aarch64-unknown-opencl --target-cpu generic --emit asm -o %t.cl
# RUN: FileCheck %s --check-prefix=CL --input-file=%t.cl
# IR: define{{.*}}spir_kernel void @aggregate_probe({
# IR: extractvalue {{.*}} 0, 0
# CL: __kernel void mojo_kernel_{{[0-9]+}}(__global uchar *{{.*}}_base, ulong {{.*}}_offset,uint

from std.memory import Pointer


@export("aggregate_probe")
def aggregate_probe(
    value: Tuple[Pointer[Float32, MutAnyOrigin], Int32]
) abi("C"):
    value[0][unsafe_offset=0] = Float32(value[1])
