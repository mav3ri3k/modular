# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the Apache License v2.0 with LLVM Exceptions: https://llvm.org/LICENSE.txt
"""Batch-one packed W2 GEMV on HVX, without HMX weight expansion.

Each byte encodes four K coefficients for one output. A per-activation lookup
maps pairs of two-bit codes to FP16 sums, shared by all output tiles. Each HVX
worker accumulates 128 output channels with QFloat16 and scales groups in FP32.
This approximate path flushes FP16 subnormal group sums, like the HMX prototype.
"""
from std.memory import Pointer, bitcast
from std.sys import llvm_intrinsic, inlined_assembly

comptime VI = SIMD[DType.int32, 32]
comptime VU = SIMD[DType.uint32, 32]
comptime VP = SIMD[DType.int32, 64]


# Preserve the existing prototype's FTZ conversion contract.
@always_inline
def half_sf(h: VU) -> VI:
    var m = h & 32767
    var b = (m << 13) + VU(0x38000000)
    b = (m & 0x7C00).eq(0).select(VU(0), b) | ((h & 32768) << 16)
    return bitcast[DType.int32](b)


@export("hexagon_prepare_lut")
def prepare(
    x: Pointer[UInt16, MutAnyOrigin],
    lut: Pointer[Int32, MutAnyOrigin],
    reduction: Int32,
) abi("C"):
    """Build FP16 pair-sum lookup tables for one activation vector."""
    # vlut4 indexes the upper two bits of each halfword. Duplicate each
    # lookup entry in both halves; vlut16 reads the low half of each word.
    var ca = VU(0)
    var cb = VU(0)
    comptime for lane in range(32):
        ca[lane] = UInt32(((lane % 4) << 14) | ((lane % 4) << 30))
        cb[lane] = UInt32((((lane // 4) % 4) << 14) | (((lane // 4) % 4) << 30))
    var wa = llvm_intrinsic["llvm.hexagon.V6.vlut4.128B", VI](
        bitcast[DType.int32](ca), UInt64(0x40003C000000BC00)
    )
    var wb = llvm_intrinsic["llvm.hexagon.V6.vlut4.128B", VI](
        bitcast[DType.int32](cb), UInt64(0x40003C000000BC00)
    )
    for pair in range(Int(reduction) // 2):
        var a = UInt32(x[unsafe_offset=pair * 2])
        var b = UInt32(x[unsafe_offset=pair * 2 + 1])
        var va = VI(bitcast[DType.int32](a | (a << 16)))
        var vb = VI(bitcast[DType.int32](b | (b << 16)))
        var pa = llvm_intrinsic["llvm.hexagon.V6.vmpy.qf16.hf.128B", VI](wa, va)
        var pb = llvm_intrinsic["llvm.hexagon.V6.vmpy.qf16.hf.128B", VI](wb, vb)
        var sum = llvm_intrinsic["llvm.hexagon.V6.vadd.qf16.128B", VI](pa, pb)
        var hf = llvm_intrinsic["llvm.hexagon.V6.vconv.hf.qf16.128B", VI](sum)
        lut.unsafe_offset(pair * 32).unsafe_store[alignment=128](hf)


@export("hexagon_hvx_project")
def project(
    w: Pointer[UInt8, MutAnyOrigin],
    scales: Pointer[Int32, MutAnyOrigin],
    lut: Pointer[Int32, MutAnyOrigin],
    y: Pointer[Int32, MutAnyOrigin],
    width: Int32,
    reduction: Int32,
    first: Int32,
    count: Int32,
) abi("C"):
    """Compute an aligned output slice using the resident packed W2 weights."""
    # Packed weights are blocked by 128 outputs, then by four input channels.
    for block in range(Int(first) // 128, (Int(first) + Int(count)) // 128):
        if block + 1 < (Int(first) + Int(count)) // 128:
            var bytes = UInt64(reduction) // 4
            var descriptor = (UInt64(128) << 32) | (UInt64(128) << 16) | bytes
            inlined_assembly[
                "l2fetch($0,$1)", NoneType, constraints="r,r,~{memory}"
            ](w.unsafe_offset((block + 1) * Int(reduction) * 32), descriptor)
        var total0 = VI(0)
        var total1 = VI(0)
        var total2 = VI(0)
        var total3 = VI(0)
        for group in range(Int(reduction) // 128):
            var acc0 = VI(0)
            var acc1 = VI(0)
            var acc2 = VI(0)
            var acc3 = VI(0)
            for byte in range(32):
                var codes = bitcast[DType.uint32, 32](
                    w.unsafe_offset(
                        (block * (Int(reduction) // 4) + group * 32 + byte)
                        * 128
                    ).unsafe_load[width=128, alignment=128]()
                )
                var t0 = lut.unsafe_offset(
                    (group * 64 + byte * 2) * 32
                ).unsafe_load[width=32, alignment=128]()
                var t1 = lut.unsafe_offset(
                    (group * 64 + byte * 2 + 1) * 32
                ).unsafe_load[width=32, alignment=128]()
                var v0 = llvm_intrinsic["llvm.hexagon.V6.vlutvwh.128B", VP](
                    bitcast[DType.int32](codes & 0x0F0F0F0F), t0, Int32(0)
                )
                var v1 = llvm_intrinsic["llvm.hexagon.V6.vlutvwh.128B", VP](
                    bitcast[DType.int32]((codes >> 4) & 0x0F0F0F0F),
                    t1,
                    Int32(0),
                )
                acc0 = llvm_intrinsic["llvm.hexagon.V6.vadd.qf16.mix.128B", VI](
                    acc0, v0.slice[32]()
                )
                acc1 = llvm_intrinsic["llvm.hexagon.V6.vadd.qf16.mix.128B", VI](
                    acc1, v0.slice[32, offset=32]()
                )
                acc2 = llvm_intrinsic["llvm.hexagon.V6.vadd.qf16.mix.128B", VI](
                    acc2, v1.slice[32]()
                )
                acc3 = llvm_intrinsic["llvm.hexagon.V6.vadd.qf16.mix.128B", VI](
                    acc3, v1.slice[32, offset=32]()
                )
            var even = llvm_intrinsic["llvm.hexagon.V6.vadd.qf16.128B", VI](
                acc0, acc2
            )
            var odd = llvm_intrinsic["llvm.hexagon.V6.vadd.qf16.128B", VI](
                acc1, acc3
            )
            even = llvm_intrinsic["llvm.hexagon.V6.vconv.hf.qf16.128B", VI](
                even
            )
            odd = llvm_intrinsic["llvm.hexagon.V6.vconv.hf.qf16.128B", VI](odd)
            var hf = bitcast[DType.uint16, 64](even).interleave(
                bitcast[DType.uint16, 64](odd)
            )
            var dot0 = half_sf(hf.slice[32, offset=0]().cast[DType.uint32]())
            var scale0 = scales.unsafe_offset(
                group * Int(width) + block * 128 + 0
            ).unsafe_load[width=32, alignment=128]()
            var product0 = llvm_intrinsic[
                "llvm.hexagon.V6.vmpy.qf32.sf.128B", VI
            ](dot0, scale0)
            product0 = llvm_intrinsic["llvm.hexagon.V6.vconv.sf.qf32.128B", VI](
                product0
            )
            total0 = llvm_intrinsic["llvm.hexagon.V6.vadd.sf.128B", VI](
                total0, product0
            )
            total0 = llvm_intrinsic["llvm.hexagon.V6.vconv.sf.qf32.128B", VI](
                total0
            )
            var dot1 = half_sf(hf.slice[32, offset=32]().cast[DType.uint32]())
            var scale1 = scales.unsafe_offset(
                group * Int(width) + block * 128 + 32
            ).unsafe_load[width=32, alignment=128]()
            var product1 = llvm_intrinsic[
                "llvm.hexagon.V6.vmpy.qf32.sf.128B", VI
            ](dot1, scale1)
            product1 = llvm_intrinsic["llvm.hexagon.V6.vconv.sf.qf32.128B", VI](
                product1
            )
            total1 = llvm_intrinsic["llvm.hexagon.V6.vadd.sf.128B", VI](
                total1, product1
            )
            total1 = llvm_intrinsic["llvm.hexagon.V6.vconv.sf.qf32.128B", VI](
                total1
            )
            var dot2 = half_sf(hf.slice[32, offset=64]().cast[DType.uint32]())
            var scale2 = scales.unsafe_offset(
                group * Int(width) + block * 128 + 64
            ).unsafe_load[width=32, alignment=128]()
            var product2 = llvm_intrinsic[
                "llvm.hexagon.V6.vmpy.qf32.sf.128B", VI
            ](dot2, scale2)
            product2 = llvm_intrinsic["llvm.hexagon.V6.vconv.sf.qf32.128B", VI](
                product2
            )
            total2 = llvm_intrinsic["llvm.hexagon.V6.vadd.sf.128B", VI](
                total2, product2
            )
            total2 = llvm_intrinsic["llvm.hexagon.V6.vconv.sf.qf32.128B", VI](
                total2
            )
            var dot3 = half_sf(hf.slice[32, offset=96]().cast[DType.uint32]())
            var scale3 = scales.unsafe_offset(
                group * Int(width) + block * 128 + 96
            ).unsafe_load[width=32, alignment=128]()
            var product3 = llvm_intrinsic[
                "llvm.hexagon.V6.vmpy.qf32.sf.128B", VI
            ](dot3, scale3)
            product3 = llvm_intrinsic["llvm.hexagon.V6.vconv.sf.qf32.128B", VI](
                product3
            )
            total3 = llvm_intrinsic["llvm.hexagon.V6.vadd.sf.128B", VI](
                total3, product3
            )
            total3 = llvm_intrinsic["llvm.hexagon.V6.vconv.sf.qf32.128B", VI](
                total3
            )
        y.unsafe_offset(block * 128 + 0).unsafe_store[alignment=128](total0)
        y.unsafe_offset(block * 128 + 32).unsafe_store[alignment=128](total1)
        y.unsafe_offset(block * 128 + 64).unsafe_store[alignment=128](total2)
        y.unsafe_offset(block * 128 + 96).unsafe_store[alignment=128](total3)
