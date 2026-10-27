# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the Apache License v2.0 with LLVM Exceptions: https://llvm.org/LICENSE.txt
# RUN: %mojo-build-no-debug-no-assert %s --target-triple aarch64-unknown-opencl --target-cpu generic --emit asm -o %t.cl
# RUN: FileCheck %s --input-file=%t.cl
# CHECK-DAG: #pragma OPENCL EXTENSION cl_khr_fp16 : enable
# CHECK-DAG: __kernel void mojo_kernel_{{[0-9]+}}(
# CHECK-DAG: convert_half2_rte(
# CHECK-DAG: convert_half2(
# CHECK-DAG: mad(

from std.memory import Pointer
from std.simd import FastMathFlag


@export("packed_math")
def packed_math(
    x: Pointer[Float32, MutAnyOrigin], y: Pointer[Float32, MutAnyOrigin]
) abi("C"):
    # The same standard SIMD and fast-math operations as Bonsai.
    var a = SIMD[.float32, 2](x[unsafe_offset=0], x[unsafe_offset=1]).cast[
        .float16
    ]()
    var w = SIMD[.int32, 2](
        Int32(x[unsafe_offset=2]), Int32(x[unsafe_offset=3])
    ).cast[.float16]()
    var previous = SIMD[.float32, 2](
        x[unsafe_offset=4], x[unsafe_offset=5]
    ).cast[.float16]()
    var acc = a.fma[FastMathFlag.AFN](w, previous)
    var total = acc.cast[.float32]().reduce_add()
    y[unsafe_offset=0] = total
