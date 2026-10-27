# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the Apache License v2.0 with LLVM Exceptions: https://llvm.org/LICENSE.txt
# RUN: %mojo-build-no-debug-no-assert %s --target-triple aarch64-unknown-opencl --target-cpu generic -D ASSERT=all --emit llvm -o %t.ll
# RUN: FileCheck %s --input-file=%t.ll --implicit-check-not=_FixedWriteBuffer --implicit-check-not=printf
# RUN: %mojo-build-no-debug-no-assert %s --target-triple aarch64-unknown-opencl --target-cpu generic -D ASSERT=all --emit asm -o %t.cl
# RUN: FileCheck %s --check-prefix=CL --input-file=%t.cl --implicit-check-not=__builtin_trap
# CL: __global uint *mojo_error
# CL: atomic_or(
# CHECK: define{{.*}}spir_kernel void @formatted_assertion(
# CHECK: icmp eq i32
# CHECK: br i1
# CHECK-LABEL: define{{.*}}spir_kernel void @nullable_input(
# CHECK: [[ADDRESS:%.*]] = ptrtoint ptr {{.*}} to i64
# CHECK: icmp eq i64 [[ADDRESS]], 0
# CHECK: call void @llvm.trap()

from std.memory import Pointer
from std.collections import OptionalReg


@export("formatted_assertion")
def formatted_assertion(value: Pointer[Int32, MutAnyOrigin]) abi("C"):
    # The OpenCL assertion hook has no message sink. Formatting must not
    # instantiate printf, and the runtime assertion must still abort.
    var x = value[unsafe_offset=0]
    debug_assert(x != 0, "expected nonzero value, got ", x)
    value[unsafe_offset=0] = x + 1


@export("nullable_input")
def nullable_input(
    output: Pointer[Int32, MutAnyOrigin],
    input: OptionalReg[Pointer[Int32, MutAnyOrigin]],
) abi("C"):
    output[unsafe_offset=0] = Int32(Bool(input))
