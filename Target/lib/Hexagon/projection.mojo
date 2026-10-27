# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the Apache License v2.0 with LLVM Exceptions: https://llvm.org/LICENSE.txt
"""Packed Bonsai projection: HVX staging/reduction and HMX FP16 tiles.

The HVX producer and HMX consumer run on separate persistent QuRT threads.
Weights stay W2 in shared memory; only the current output batch is expanded.
"""
from std.memory import Pointer, bitcast
from std.sys import llvm_intrinsic, inlined_assembly

comptime VI = SIMD[DType.int32, 32]
comptime VU = SIMD[DType.uint32, 32]


@always_inline
def _half_bits(code: VU) -> VU:
    return code.eq(0).select(
        VU(0xBC00),
        code.eq(1).select(VU(0), code.eq(2).select(VU(0x3C00), VU(0x4000))),
    )


@export("hexagon_prepare_activations")
def prepare_activations(
    x: Pointer[UInt16, MutAnyOrigin],
    a: Pointer[UInt32, MutAnyOrigin],
    reduction: Int32,
) abi("C"):
    # One Crouton holds 32 input channels and 32 identical spatial rows.
    for tile in range(Int(reduction) // 32):
        var v = (
            x.unsafe_offset(tile * 32)
            .unsafe_load[width=32, alignment=64]()
            .cast[DType.uint32]()
        )
        var pairs = v | (v << 16)
        for rows in range(16):
            a.unsafe_offset(tile * 512 + rows * 32).unsafe_store[alignment=128](
                pairs
            )


@export("hexagon_prepare_weights")
def prepare_weights(
    w: Pointer[UInt32, MutAnyOrigin],
    tiles: Pointer[UInt32, MutAnyOrigin],
    width: Int32,
    reduction: Int32,
    first: Int32,
    count: Int32,
) abi("C"):
    var groups = Int(reduction) // 128
    var bytes = UInt64(reduction) * 8
    var descriptor = (bytes << 32) | (bytes << 16) | 1
    inlined_assembly["l2fetch($0,$1)", NoneType, constraints="r,r,~{memory}"](
        w.unsafe_offset((Int(first) // 32) * (Int(reduction) // 16) * 32),
        descriptor,
    )
    for block in range(Int(count) // 32):
        if block + 1 < Int(count) // 32:
            inlined_assembly[
                "l2fetch($0,$1)", NoneType, constraints="r,r,~{memory}"
            ](
                w.unsafe_offset(
                    (Int(first) // 32 + block + 1) * (Int(reduction) // 16) * 32
                ),
                descriptor,
            )
        for group in range(groups):
            for word in range(8):
                var codes = w.unsafe_offset(
                    (
                        (Int(first) // 32 + block) * (Int(reduction) // 16)
                        + group * 8
                        + word
                    )
                    * 32
                ).unsafe_load[width=32, alignment=128]()
                comptime for pair in range(8):
                    var lo = _half_bits((codes >> UInt32(pair * 4)) & 3)
                    var hi = _half_bits((codes >> UInt32(pair * 4 + 2)) & 3)
                    var bits = lo | (hi << 16)
                    var offset = (
                        (block * groups + group) * 2048
                        + (word // 2) * 512
                        + ((word % 2) * 8 + pair) * 32
                    )
                    tiles.unsafe_offset(offset).unsafe_store[alignment=128](
                        bits
                    )


@export("hexagon_hmx_project")
def hmx_project(
    a: Pointer[UInt32, MutAnyOrigin],
    weights: Pointer[UInt32, MutAnyOrigin],
    scales: Pointer[UInt32, MutAnyOrigin],
    partials: Pointer[UInt32, MutAnyOrigin],
    reduction: Int32,
    count: Int32,
) abi("C"):
    var groups = Int(reduction) // 128
    for block in range(Int(count) // 32):
        for group in range(groups):
            inlined_assembly[
                ".word 0xa6e0c013", NoneType, constraints="~{memory}"
            ]()
            for tile in range(4):
                # activation.hf = mxmem(r0,r1); weight.hf = mxmem(r2,r3)
                inlined_assembly[
                    ".word 0x920041e4\n.word 0x9202e3ef",
                    NoneType,
                    constraints="{r0},{r1},{r2},{r3},~{memory}",
                ](
                    a.unsafe_offset((group * 4 + tile) * 512),
                    UInt32(2047),
                    weights.unsafe_offset(
                        ((block * groups + group) * 4 + tile) * 512
                    ),
                    UInt32(2047),
                )
            # Bias registers contain unit output scales; preserve group scales in FP32.
            inlined_assembly[
                ".word 0x9204c3fe\n.word 0xa6e5e604",
                NoneType,
                constraints="{r4},{r5},{r6},~{memory}",
            ](
                scales,
                partials.unsafe_offset((block * groups + group) * 512),
                UInt32(0),
            )


@export("hexagon_reduce_projection")
def reduce_projection(
    partials: Pointer[Int32, MutAnyOrigin],
    scales: Pointer[Float32, MutAnyOrigin],
    y: Pointer[Int32, MutAnyOrigin],
    shape: Pointer[Int32, MutAnyOrigin],
) abi("C"):
    var width = shape[unsafe_offset=0]
    var reduction = shape[unsafe_offset=1]
    var first = shape[unsafe_offset=2]
    var count = shape[unsafe_offset=3]
    var groups = Int(reduction) // 128
    for block in range(Int(count) // 32):
        var total = VI(0)
        for group in range(groups):
            # Low vector contains the even spatial row (row zero), all 32 channels.
            var halves = partials.unsafe_offset(
                (block * groups + group) * 512
            ).unsafe_load[width=32, alignment=128]()
            var h = bitcast[DType.uint32](halves) & 65535
            var magnitude = h & 32767
            var bits = (magnitude << 13) + VU(0x38000000)
            # Fast path flushes FP16 subnormal group partials, like relaxed GPU math.
            bits = (magnitude & 0x7C00).eq(0).select(VU(0), bits)
            bits |= (h & 32768) << 16
            var values = bitcast[DType.int32](bits)
            var scale = bitcast[DType.int32](
                scales.unsafe_offset(
                    group * Int(width) + Int(first) + block * 32
                ).unsafe_load[width=32, alignment=128]()
            )
            var product = llvm_intrinsic[
                "llvm.hexagon.V6.vmpy.qf32.sf.128B", VI
            ](values, scale)
            product = llvm_intrinsic["llvm.hexagon.V6.vconv.sf.qf32.128B", VI](
                product
            )
            total = llvm_intrinsic["llvm.hexagon.V6.vadd.sf.128B", VI](
                total, product
            )
            total = llvm_intrinsic["llvm.hexagon.V6.vconv.sf.qf32.128B", VI](
                total
            )
        y.unsafe_offset(Int(first) + block * 32).unsafe_store[alignment=128](
            total
        )
