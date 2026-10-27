# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the Apache License v2.0 with LLVM Exceptions: https://llvm.org/LICENSE.txt
# RUN: %mojo-build-no-debug-no-assert %s --target-triple aarch64-unknown-opencl --target-cpu generic --emit asm -I %S/../../../max/mojo -o %t.cl
# RUN: FileCheck %s --input-file=%t.cl
# RUN: %mojo-build-no-debug-no-assert %s --target-triple aarch64-unknown-opencl --target-cpu generic --emit llvm -I %S/../../../max/mojo -o %t.ll
# RUN: FileCheck %s --check-prefix=IR --input-file=%t.ll
# IR-DAG: define{{.*}}spir_kernel void @matmul(
# IR-DAG: define{{.*}}spir_kernel void @dense_relu(
# IR-DAG: define{{.*}}spir_kernel void @phi_swap(
# IR-DAG: define{{.*}}spir_kernel void @narrow_multiply(
# IR-DAG: define{{.*}}spir_kernel void @float_predicates(
# CHECK: get_local_id(

from std.memory import Pointer
from std.sys.info import CompilationTarget
from max.gpu import global_idx


@export("matmul")
def matmul(
    a: Pointer[Float32, MutAnyOrigin],
    b: Pointer[Float32, MutAnyOrigin],
    output: Pointer[Float32, MutAnyOrigin],
    m: Int32,
    n: Int32,
    k: Int32,
) abi("C"):
    """Computes one matrix multiplication output per GPU thread."""
    comptime assert CompilationTarget.current().is_gpu()
    comptime assert not CompilationTarget.current().is_cpu()
    var i = global_idx.y
    var j = global_idx.x
    if i < Int(m) and j < Int(n):
        var value: Float32 = 0
        for p in range(Int(k)):
            value += (
                a[unsafe_offset=i * Int(k) + p]
                * b[unsafe_offset=p * Int(n) + j]
            )
        output[unsafe_offset=i * Int(n) + j] = value


@export("dense_relu")
def dense_relu(
    inputs: Pointer[Float32, MutAnyOrigin],
    weights: Pointer[Float32, MutAnyOrigin],
    bias: Pointer[Float32, MutAnyOrigin],
    output: Pointer[Float32, MutAnyOrigin],
    batch: Int32,
    width: Int32,
    reduction: Int32,
) abi("C"):
    """Computes a dense layer with a bias and ReLU, one output per GPU thread.
    """
    var row = global_idx.y
    var col = global_idx.x
    if row < Int(batch) and col < Int(width):
        var value = bias[unsafe_offset=col]
        for i in range(Int(reduction)):
            value += (
                inputs[unsafe_offset=row * Int(reduction) + i]
                * weights[unsafe_offset=i * Int(width) + col]
            )
        output[unsafe_offset=row * Int(width) + col] = max(value, Float32(0))


@export("phi_swap")
def phi_swap(output: Pointer[Int32, MutAnyOrigin], rounds: Int32) abi("C"):
    """Exercises parallel copies for cyclic loop-carried SSA values."""
    var a: Int32 = 1
    var b: Int32 = 2
    for _ in range(Int(rounds)):
        a, b = b, a
    output[unsafe_offset=global_idx.x] = a * 10 + b


@export("narrow_multiply")
def narrow_multiply(
    output: Pointer[UInt64, MutAnyOrigin], value: UInt16
) abi("C"):
    """Exercises integer promotion without introducing signed C overflow."""
    output[unsafe_offset=global_idx.x] = UInt64(value * value)


@export("float_predicates")
def float_predicates(
    output: Pointer[UInt32, MutAnyOrigin], a: Float32, b: Float32
) abi("C"):
    """Exercises ordered/unordered FP32 comparisons, including NaNs and zeros.
    """
    var value = (
        UInt32(a < b)
        + UInt32(a <= b) * 2
        + UInt32(a == b) * 4
        + UInt32(a != b) * 8
        + UInt32(a > b) * 16
        + UInt32(a >= b) * 32
    )
    output[unsafe_offset=global_idx.x] = value
