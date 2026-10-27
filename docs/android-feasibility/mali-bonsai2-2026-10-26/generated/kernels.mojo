# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the Apache License v2.0 with LLVM Exceptions: https://llvm.org/LICENSE.txt
"""Bonsai 2 2B: packed weights, FP16 matvecs and FP32 recurrent state."""
from std.memory import Pointer
from std.memory import bitcast
from std.math import exp, isfinite, log, rsqrt
from std.simd import FastMathFlag
from max.gpu import global_idx, block_idx
from max.gpu.primitives.warp import shuffle_xor


@always_inline
def sigmoid(x: Float32) -> Float32:
    return 1 / (1 + exp(-x))


def hadamard(
    x: Pointer[Float32, MutAnyOrigin],
    signs: Pointer[Float32, MutAnyOrigin],
    y: Pointer[Float32, MutAnyOrigin],
    q: Pointer[UInt32, MutAnyOrigin],
    width: Int32,
    stride: Int32,
    inverse: Int32,
):
    # Adreno 840 uses 64-lane subgroups for this 64-thread workgroup.
    var lane = global_idx.x % 64
    var base = (global_idx.x // 64) * 512
    var v0 = x[unsafe_offset=base + lane + 0]
    if inverse == 0:
        v0 *= signs[unsafe_offset=base + lane + 0]
    var v1 = x[unsafe_offset=base + lane + 64]
    if inverse == 0:
        v1 *= signs[unsafe_offset=base + lane + 64]
    var v2 = x[unsafe_offset=base + lane + 128]
    if inverse == 0:
        v2 *= signs[unsafe_offset=base + lane + 128]
    var v3 = x[unsafe_offset=base + lane + 192]
    if inverse == 0:
        v3 *= signs[unsafe_offset=base + lane + 192]
    var v4 = x[unsafe_offset=base + lane + 256]
    if inverse == 0:
        v4 *= signs[unsafe_offset=base + lane + 256]
    var v5 = x[unsafe_offset=base + lane + 320]
    if inverse == 0:
        v5 *= signs[unsafe_offset=base + lane + 320]
    var v6 = x[unsafe_offset=base + lane + 384]
    if inverse == 0:
        v6 *= signs[unsafe_offset=base + lane + 384]
    var v7 = x[unsafe_offset=base + lane + 448]
    if inverse == 0:
        v7 *= signs[unsafe_offset=base + lane + 448]
    comptime for stage in range(6):
        comptime mask = 1 << stage
        var t0 = shuffle_xor(v0, UInt32(mask))
        v0 = t0 - v0 if (lane & mask) != 0 else v0 + t0
        var t1 = shuffle_xor(v1, UInt32(mask))
        v1 = t1 - v1 if (lane & mask) != 0 else v1 + t1
        var t2 = shuffle_xor(v2, UInt32(mask))
        v2 = t2 - v2 if (lane & mask) != 0 else v2 + t2
        var t3 = shuffle_xor(v3, UInt32(mask))
        v3 = t3 - v3 if (lane & mask) != 0 else v3 + t3
        var t4 = shuffle_xor(v4, UInt32(mask))
        v4 = t4 - v4 if (lane & mask) != 0 else v4 + t4
        var t5 = shuffle_xor(v5, UInt32(mask))
        v5 = t5 - v5 if (lane & mask) != 0 else v5 + t5
        var t6 = shuffle_xor(v6, UInt32(mask))
        v6 = t6 - v6 if (lane & mask) != 0 else v6 + t6
        var t7 = shuffle_xor(v7, UInt32(mask))
        v7 = t7 - v7 if (lane & mask) != 0 else v7 + t7
    var t0_0 = v0
    v0 += v1
    v1 = t0_0 - v1
    var t0_2 = v2
    v2 += v3
    v3 = t0_2 - v3
    var t0_4 = v4
    v4 += v5
    v5 = t0_4 - v5
    var t0_6 = v6
    v6 += v7
    v7 = t0_6 - v7
    var t1_0 = v0
    v0 += v2
    v2 = t1_0 - v2
    var t1_1 = v1
    v1 += v3
    v3 = t1_1 - v3
    var t1_4 = v4
    v4 += v6
    v6 = t1_4 - v6
    var t1_5 = v5
    v5 += v7
    v7 = t1_5 - v7
    var t2_0 = v0
    v0 += v4
    v4 = t2_0 - v4
    var t2_1 = v1
    v1 += v5
    v5 = t2_1 - v5
    var t2_2 = v2
    v2 += v6
    v6 = t2_2 - v6
    var t2_3 = v3
    v3 += v7
    v7 = t2_3 - v7
    v0 *= Float32(0.04419417382415922)
    if inverse != 0:
        v0 *= signs[unsafe_offset=base + lane + 0]
    y[unsafe_offset=base + lane + 0] = v0
    if inverse == 0:
        var next0 = shuffle_xor(v0, UInt32(1))
        if lane % 2 == 0:
            q[unsafe_offset=base // 2 + lane // 2 + 0] = bitcast[.uint32, 1](
                SIMD[.float32, 2](v0, next0).cast[.float16]()
            )
    v1 *= Float32(0.04419417382415922)
    if inverse != 0:
        v1 *= signs[unsafe_offset=base + lane + 64]
    y[unsafe_offset=base + lane + 64] = v1
    if inverse == 0:
        var next1 = shuffle_xor(v1, UInt32(1))
        if lane % 2 == 0:
            q[unsafe_offset=base // 2 + lane // 2 + 32] = bitcast[.uint32, 1](
                SIMD[.float32, 2](v1, next1).cast[.float16]()
            )
    v2 *= Float32(0.04419417382415922)
    if inverse != 0:
        v2 *= signs[unsafe_offset=base + lane + 128]
    y[unsafe_offset=base + lane + 128] = v2
    if inverse == 0:
        var next2 = shuffle_xor(v2, UInt32(1))
        if lane % 2 == 0:
            q[unsafe_offset=base // 2 + lane // 2 + 64] = bitcast[.uint32, 1](
                SIMD[.float32, 2](v2, next2).cast[.float16]()
            )
    v3 *= Float32(0.04419417382415922)
    if inverse != 0:
        v3 *= signs[unsafe_offset=base + lane + 192]
    y[unsafe_offset=base + lane + 192] = v3
    if inverse == 0:
        var next3 = shuffle_xor(v3, UInt32(1))
        if lane % 2 == 0:
            q[unsafe_offset=base // 2 + lane // 2 + 96] = bitcast[.uint32, 1](
                SIMD[.float32, 2](v3, next3).cast[.float16]()
            )
    v4 *= Float32(0.04419417382415922)
    if inverse != 0:
        v4 *= signs[unsafe_offset=base + lane + 256]
    y[unsafe_offset=base + lane + 256] = v4
    if inverse == 0:
        var next4 = shuffle_xor(v4, UInt32(1))
        if lane % 2 == 0:
            q[unsafe_offset=base // 2 + lane // 2 + 128] = bitcast[.uint32, 1](
                SIMD[.float32, 2](v4, next4).cast[.float16]()
            )
    v5 *= Float32(0.04419417382415922)
    if inverse != 0:
        v5 *= signs[unsafe_offset=base + lane + 320]
    y[unsafe_offset=base + lane + 320] = v5
    if inverse == 0:
        var next5 = shuffle_xor(v5, UInt32(1))
        if lane % 2 == 0:
            q[unsafe_offset=base // 2 + lane // 2 + 160] = bitcast[.uint32, 1](
                SIMD[.float32, 2](v5, next5).cast[.float16]()
            )
    v6 *= Float32(0.04419417382415922)
    if inverse != 0:
        v6 *= signs[unsafe_offset=base + lane + 384]
    y[unsafe_offset=base + lane + 384] = v6
    if inverse == 0:
        var next6 = shuffle_xor(v6, UInt32(1))
        if lane % 2 == 0:
            q[unsafe_offset=base // 2 + lane // 2 + 192] = bitcast[.uint32, 1](
                SIMD[.float32, 2](v6, next6).cast[.float16]()
            )
    v7 *= Float32(0.04419417382415922)
    if inverse != 0:
        v7 *= signs[unsafe_offset=base + lane + 448]
    y[unsafe_offset=base + lane + 448] = v7
    if inverse == 0:
        var next7 = shuffle_xor(v7, UInt32(1))
        if lane % 2 == 0:
            q[unsafe_offset=base // 2 + lane // 2 + 224] = bitcast[.uint32, 1](
                SIMD[.float32, 2](v7, next7).cast[.float16]()
            )


@always_inline
def _group_dot(
    x: Pointer[UInt32, MutAnyOrigin],
    w: Pointer[UInt32, MutAnyOrigin],
    width: Int,
    i: Int,
    group: Int,
) -> Float32:
    # Permit the same approximate FP16 multiply-add used by this quantized
    # model, through the standard fast-math contract rather than a device hook.
    var d0 = SIMD[.float16, 2](0)
    var d1 = SIMD[.float16, 2](0)
    var d2 = SIMD[.float16, 2](0)
    var d3 = SIMD[.float16, 2](0)
    for word in range(8):
        var codes = w[unsafe_offset=(group * 8 + word) * width + i]
        comptime for pair in range(8):
            var q = bitcast[.float16, 2](
                x[unsafe_offset=group * 64 + word * 8 + pair]
            )
            var c = (codes >> UInt32(pair * 4)) & 15
            var weight = SIMD[.int32, 2](
                Int32(c & 3) - 1, Int32((c >> 2) & 3) - 1
            ).cast[.float16]()
            comptime if pair % 4 == 0:
                d0 = q.fma[FastMathFlag.AFN](weight, d0)
            elif pair % 4 == 1:
                d1 = q.fma[FastMathFlag.AFN](weight, d1)
            elif pair % 4 == 2:
                d2 = q.fma[FastMathFlag.AFN](weight, d2)
            else:
                d3 = q.fma[FastMathFlag.AFN](weight, d3)
    return (
        d0.cast[.float32]().reduce_add() + d1.cast[.float32]().reduce_add()
    ) + (d2.cast[.float32]().reduce_add() + d3.cast[.float32]().reduce_add())


@always_inline
def _group_dot8(
    x: Pointer[UInt32, MutAnyOrigin],
    w: Pointer[UInt32, MutAnyOrigin],
    width: Int,
    i: Int,
    group: Int,
) -> SIMD[.float32, 8]:
    # Adjacent outputs share the input pair and use contiguous packed loads.
    # Keep low and high activation components in separate SIMD8 streams. This
    # avoids constructing SIMD16 repeated-activation and unpack temporaries,
    # while retaining the eight-output tile and four independent FP16 chains.
    var d0lo = SIMD[.float16, 8](0)
    var d0hi = SIMD[.float16, 8](0)
    var d1lo = SIMD[.float16, 8](0)
    var d1hi = SIMD[.float16, 8](0)
    var d2lo = SIMD[.float16, 8](0)
    var d2hi = SIMD[.float16, 8](0)
    var d3lo = SIMD[.float16, 8](0)
    var d3hi = SIMD[.float16, 8](0)
    for word in range(8):
        var codes = w.unsafe_load[width=8]((group * 8 + word) * width + i)
        comptime for pair in range(8):
            var q = bitcast[.float16, 2](
                x[unsafe_offset=group * 64 + word * 8 + pair]
            )
            var c = (codes >> UInt32(pair * 4)) & 15
            var lo_weight = (c & 3).cast[.float16]() - SIMD[.float16, 8](1)
            var hi_weight = (
                ((c >> 2) & 3).cast[.float16]() - SIMD[.float16, 8](1)
            )
            var qlo = SIMD[.float16, 8](q[0])
            var qhi = SIMD[.float16, 8](q[1])
            comptime if pair % 4 == 0:
                d0lo = qlo.fma[FastMathFlag.AFN](lo_weight, d0lo)
                d0hi = qhi.fma[FastMathFlag.AFN](hi_weight, d0hi)
            elif pair % 4 == 1:
                d1lo = qlo.fma[FastMathFlag.AFN](lo_weight, d1lo)
                d1hi = qhi.fma[FastMathFlag.AFN](hi_weight, d1hi)
            elif pair % 4 == 2:
                d2lo = qlo.fma[FastMathFlag.AFN](lo_weight, d2lo)
                d2hi = qhi.fma[FastMathFlag.AFN](hi_weight, d2hi)
            else:
                d3lo = qlo.fma[FastMathFlag.AFN](lo_weight, d3lo)
                d3hi = qhi.fma[FastMathFlag.AFN](hi_weight, d3hi)
    var result = SIMD[.float32, 8](0)
    comptime for output in range(8):
        result[output] = (
            (d0lo[output].cast[.float32]() + d0hi[output].cast[.float32]())
            + (d1lo[output].cast[.float32]() + d1hi[output].cast[.float32]())
        ) + (
            (d2lo[output].cast[.float32]() + d2hi[output].cast[.float32]())
            + (d3lo[output].cast[.float32]() + d3hi[output].cast[.float32]())
        )
    return result


def packed_linear(
    x: Pointer[UInt32, MutAnyOrigin],
    w: Pointer[UInt32, MutAnyOrigin],
    params: Pointer[Float16, MutAnyOrigin],
    partials: Pointer[Float32, MutAnyOrigin],
    width: Int32,
    reduction: Int32,
):
    """Join eight quantization groups inside each 64-lane subgroup.

    Eight lanes cover adjacent eight-output tiles; the other lane bits cover
    quantization groups. Only the leading group writes the FP32 partial.
    Inactive groups/outputs contribute zero and still participate in shuffles.
    """
    var lane = global_idx.x % 64
    var group = block_idx.y * 8 + lane // 8
    var i = (block_idx.x * 8 + lane % 8) * 8
    var dot = SIMD[.float32, 8](0)
    if group < Int(reduction) // 128:
        if i + 8 <= Int(width):
            var base = group * Int(width) + i
            var scales = params.unsafe_load[width=8](base).cast[.float32]()
            dot = _group_dot8(x, w, Int(width), i, group) * scales
        elif i < Int(width):
            for o in range(8):
                if i + o < Int(width):
                    dot[o] = _group_dot(
                        x, w, Int(width), i + o, group
                    ) * Float32(
                        params[unsafe_offset=group * Int(width) + i + o]
                    )
    comptime for stage in range(3):
        comptime for o in range(8):
            dot[o] += shuffle_xor(dot[o], UInt32(8 << stage))
    if lane < 8:
        if i + 8 <= Int(width):
            partials.unsafe_store[width=8](block_idx.y * Int(width) + i, dot)
        elif i < Int(width):
            for o in range(8):
                if i + o < Int(width):
                    partials[
                        unsafe_offset=block_idx.y * Int(width) + i + o
                    ] = dot[o]


# Experimental standalone benchmark candidate; model paths use packed_linear.
@always_inline
def _group_dot4_tiled(
    x: Pointer[UInt32, MutAnyOrigin],
    w: Pointer[UInt32, MutAnyOrigin],
    width: Int,
    groups: Int,
    i: Int,
    group: Int,
) -> SIMD[.float32, 4]:
    # Adjacent outputs share the input pair and use contiguous packed loads.
    # Keep low and high activation components in separate SIMD4 streams. This
    # avoids constructing SIMD16 repeated-activation and unpack temporaries,
    # while retaining the 4-output tile and four independent FP16 chains.
    var d0lo = SIMD[.float16, 4](0)
    var d0hi = SIMD[.float16, 4](0)
    var d1lo = SIMD[.float16, 4](0)
    var d1hi = SIMD[.float16, 4](0)
    var d2lo = SIMD[.float16, 4](0)
    var d2hi = SIMD[.float16, 4](0)
    var d3lo = SIMD[.float16, 4](0)
    var d3hi = SIMD[.float16, 4](0)
    for word in range(8):
        var codes = w.unsafe_load[width=4](Int(((UInt32(group) // 8 * (UInt32(width) // 32) + UInt32(i) // 32) * 64 + UInt32(word) * 8 + UInt32(group) % 8) * 32 + UInt32(i) % 32))
        comptime for pair in range(8):
            var q = bitcast[.float16, 2](
                x[unsafe_offset=group * 64 + word * 8 + pair]
            )
            var c = (codes >> UInt32(pair * 4)) & 15
            var lo_weight = (c & 3).cast[.float16]() - SIMD[.float16, 4](1)
            var hi_weight = (
                ((c >> 2) & 3).cast[.float16]() - SIMD[.float16, 4](1)
            )
            var qlo = SIMD[.float16, 4](q[0])
            var qhi = SIMD[.float16, 4](q[1])
            comptime if pair % 4 == 0:
                d0lo = qlo.fma[FastMathFlag.AFN](lo_weight, d0lo)
                d0hi = qhi.fma[FastMathFlag.AFN](hi_weight, d0hi)
            elif pair % 4 == 1:
                d1lo = qlo.fma[FastMathFlag.AFN](lo_weight, d1lo)
                d1hi = qhi.fma[FastMathFlag.AFN](hi_weight, d1hi)
            elif pair % 4 == 2:
                d2lo = qlo.fma[FastMathFlag.AFN](lo_weight, d2lo)
                d2hi = qhi.fma[FastMathFlag.AFN](hi_weight, d2hi)
            else:
                d3lo = qlo.fma[FastMathFlag.AFN](lo_weight, d3lo)
                d3hi = qhi.fma[FastMathFlag.AFN](hi_weight, d3hi)
    var result = SIMD[.float32, 4](0)
    comptime for output in range(4):
        result[output] = (
            (d0lo[output].cast[.float32]() + d0hi[output].cast[.float32]())
            + (d1lo[output].cast[.float32]() + d1hi[output].cast[.float32]())
        ) + (
            (d2lo[output].cast[.float32]() + d2hi[output].cast[.float32]())
            + (d3lo[output].cast[.float32]() + d3hi[output].cast[.float32]())
        )
    return result


def packed_linear_tiled(
    x: Pointer[UInt32, MutAnyOrigin],
    w: Pointer[UInt32, MutAnyOrigin],
    params: Pointer[Float16, MutAnyOrigin],
    partials: Pointer[Float32, MutAnyOrigin],
    width: Int32,
    reduction: Int32,
):
    """Project the 32-output tiled layout with eight-group subgroup joins.

    Requires width divisible by 64 and reduction divisible by 1024.
    The loader repacks only these complete non-head projection shapes.

    8 lanes cover adjacent 4-output tiles; the other lane bits cover
    quantization groups. Only the leading group writes the FP32 partial.
    Inactive groups/outputs contribute zero and still participate in shuffles.
    """
    var lane = global_idx.x % 64
    var group = Int(UInt32(block_idx.y) * 8 + UInt32(lane) // 8)
    var i = Int((UInt32(global_idx.x) // 64 * 8 + UInt32(lane) % 8) * 4)
    var dot = SIMD[.float32, 4](0)
    if group < Int(reduction) // 128:
        if i + 4 <= Int(width):
            var base = group * Int(width) + i
            var scales = params.unsafe_load[width=4](base).cast[.float32]()
            dot = _group_dot4_tiled(
                x, w, Int(width), Int(reduction) // 128, i, group
            ) * scales
        elif i < Int(width):
            for o in range(4):
                if i + o < Int(width):
                    dot[o] = _group_dot(
                        x, w, Int(width), i + o, group
                    ) * Float32(
                        params[unsafe_offset=group * Int(width) + i + o]
                    )
    comptime for stage in range(3):
        comptime for o in range(4):
            dot[o] += shuffle_xor(dot[o], UInt32(8 << stage))
    if lane < 8:
        if i + 4 <= Int(width):
            partials.unsafe_store[width=4](block_idx.y * Int(width) + i, dot)
        elif i < Int(width):
            for o in range(4):
                if i + o < Int(width):
                    partials[
                        unsafe_offset=block_idx.y * Int(width) + i + o
                    ] = dot[o]


def packed_reduce(
    partials: Pointer[Float32, MutAnyOrigin],
    y: Pointer[Float32, MutAnyOrigin],
    width: Int32,
    groups: Int32,
):
    var i = global_idx.x
    if i < Int(width):
        var value: Float32 = 0
        for group in range(Int(groups)):
            value += partials[unsafe_offset=group * Int(width) + i]
        y[unsafe_offset=i] = value


def packed_head(
    x: Pointer[UInt32, MutAnyOrigin],
    w: Pointer[UInt32, MutAnyOrigin],
    params: Pointer[Float16, MutAnyOrigin],
    y: Pointer[Float32, MutAnyOrigin],
    candidates: Pointer[Float32, MutAnyOrigin],
    width: Int32,
    reduction: Int32,
    select: Int32,
):
    var i = global_idx.x * 8
    var result = SIMD[.float32, 8](0)
    if i + 8 <= Int(width):
        for group in range(Int(reduction) // 128):
            var base = group * Int(width) + i
            var scales = SIMD[.float32, 8](
                Float32(params[unsafe_offset=base + 0]),
                Float32(params[unsafe_offset=base + 1]),
                Float32(params[unsafe_offset=base + 2]),
                Float32(params[unsafe_offset=base + 3]),
                Float32(params[unsafe_offset=base + 4]),
                Float32(params[unsafe_offset=base + 5]),
                Float32(params[unsafe_offset=base + 6]),
                Float32(params[unsafe_offset=base + 7]),
            )
            result += _group_dot8(x, w, Int(width), i, group) * scales
        y.unsafe_store[width=8](i, result)
    elif i < Int(width):
        for o in range(8):
            if i + o < Int(width):
                var result = Float32(0)
                for group in range(Int(reduction) // 128):
                    result += _group_dot(
                        x, w, Int(width), i + o, group
                    ) * Float32(
                        params[unsafe_offset=group * Int(width) + i + o]
                    )
                y[unsafe_offset=i + o] = result



def embedding(
    w: Pointer[UInt32, MutAnyOrigin],
    params: Pointer[Float16, MutAnyOrigin],
    y: Pointer[Float32, MutAnyOrigin],
    token: Int32,
):
    var i = global_idx.x
    if i < 2048:
        var code = (
            w[unsafe_offset=(i // 16) * 248320 + Int(token)]
            >> UInt32((i % 16) * 2)
        ) & 3
        var off = (i // 128) * 248320 + Int(token)
        y[unsafe_offset=i] = Float32(Int32(code) - 1) * Float32(
            params[unsafe_offset=off]
        )


def rms_partials(
    x: Pointer[Float32, MutAnyOrigin], partials: Pointer[Float32, MutAnyOrigin]
):
    """Read each input once, with adjacent work items reading adjacent elements.
    """
    var i = global_idx.x
    if i < 32:
        var ss: Float32 = 0
        for j in range(64):
            var v = x[unsafe_offset=j * 32 + i]
            ss += v * v
        partials[unsafe_offset=i] = ss


def rms_norm(
    x: Pointer[Float32, MutAnyOrigin],
    w: Pointer[Float32, MutAnyOrigin],
    y: Pointer[Float32, MutAnyOrigin],
    partials: Pointer[Float32, MutAnyOrigin],
):
    var i = global_idx.x
    if i < 2048:
        var ss: Float32 = 0
        for j in range(32):
            ss += partials[unsafe_offset=j]
        y[unsafe_offset=i] = (
            x[unsafe_offset=i]
            * rsqrt(ss / 2048 + Float32(1e-6))
            * w[unsafe_offset=i]
        )


def gate_partials(
    x: Pointer[Float32, MutAnyOrigin],
    a: Pointer[Float32, MutAnyOrigin],
    b: Pointer[Float32, MutAnyOrigin],
    partials: Pointer[Float32, MutAnyOrigin],
):
    """Split both 2048-to-16 gate projections into 32 parallel chunks."""
    var i = global_idx.x
    if i < 1024:
        var col = i % 32
        var chunk = i // 32
        var matrix = a
        if col >= 16:
            matrix = b
        var value: Float32 = 0
        for r in range(64):
            var j = chunk * 64 + r
            value += (
                x[unsafe_offset=j] * matrix[unsafe_offset=j * 16 + col % 16]
            )
        partials[unsafe_offset=i] = value


def convolve(
    x: Pointer[Float32, MutAnyOrigin],
    history: Pointer[Float32, MutAnyOrigin],
    w: Pointer[Float32, MutAnyOrigin],
    y: Pointer[Float32, MutAnyOrigin],
):
    var i = global_idx.x
    if i < 6144:
        var value = x[unsafe_offset=i] * w[unsafe_offset=3 * 6144 + i]
        for t in range(3):
            value += (
                history[unsafe_offset=t * 6144 + i]
                * w[unsafe_offset=t * 6144 + i]
            )
        history[unsafe_offset=i] = history[unsafe_offset=6144 + i]
        history[unsafe_offset=6144 + i] = history[unsafe_offset=12288 + i]
        history[unsafe_offset=12288 + i] = x[unsafe_offset=i]
        y[unsafe_offset=i] = value * sigmoid(value)


def normalize_qk(
    x: Pointer[Float32, MutAnyOrigin], y: Pointer[Float32, MutAnyOrigin]
):
    var i = global_idx.x
    if i < 6144:
        var value = x[unsafe_offset=i]
        if i < 4096:
            var base = (i // 128) * 128
            var ss: Float32 = 0
            var lane = i % 64
            comptime for chunk in range(2):
                var v = x[unsafe_offset=base + lane + chunk * 64]
                ss += v * v
            comptime for stage in range(6):
                ss += shuffle_xor(ss, UInt32(1 << stage))
            value *= rsqrt(ss + Float32(1e-6))
            if i < 2048:
                value *= Float32(0.08838834764831845)
        y[unsafe_offset=i] = value


def delta_gates(
    partials: Pointer[Float32, MutAnyOrigin],
    constants: Pointer[Float32, MutAnyOrigin],
    y: Pointer[Float32, MutAnyOrigin],
):
    """Finish both projections and fuse the gate activations into their reduction.
    """
    var i = global_idx.x
    if i < 16:
        var a: Float32 = 0
        var b: Float32 = 0
        for chunk in range(32):
            a += partials[unsafe_offset=chunk * 32 + i]
            b += partials[unsafe_offset=chunk * 32 + i + 16]
        var v = a + constants[unsafe_offset=16 + i]
        var softplus = max(v, Float32(0)) + log(1 + exp(-abs(v)))
        y[unsafe_offset=i * 2] = exp(
            -exp(constants[unsafe_offset=i]) * softplus
        )
        y[unsafe_offset=i * 2 + 1] = sigmoid(b)


def delta_update(
    qkv: Pointer[Float32, MutAnyOrigin],
    gates: Pointer[Float32, MutAnyOrigin],
    state: Pointer[Float32, MutAnyOrigin],
    y: Pointer[Float32, MutAnyOrigin],
):
    """Split each state column across eight lanes with disjoint K slices.

    Subgroup joins produce the prediction and output. Each lane updates only
    its sixteen rows, so no global intermediate, barrier or atomic is needed.
    """
    var lane = global_idx.x % 64
    var part = lane // 8
    var i = (global_idx.x // 64) * 8 + lane % 8
    if i < 2048:
        var h = i // 128
        var d = i % 128
        var decay = gates[unsafe_offset=h * 2]
        var beta = gates[unsafe_offset=h * 2 + 1]
        var predicted: Float32 = 0
        for step in range(16):
            var k = part * 16 + step
            predicted += (
                state[unsafe_offset=(h * 128 + k) * 128 + d]
                * decay
                * qkv[unsafe_offset=2048 + h * 128 + k]
            )
        predicted += shuffle_xor(predicted, UInt32(8))
        predicted += shuffle_xor(predicted, UInt32(16))
        predicted += shuffle_xor(predicted, UInt32(32))
        var delta = (qkv[unsafe_offset=4096 + i] - predicted) * beta
        var value: Float32 = 0
        for step in range(16):
            var k = part * 16 + step
            var off = (h * 128 + k) * 128 + d
            var updated = (
                state[unsafe_offset=off] * decay
                + qkv[unsafe_offset=2048 + h * 128 + k] * delta
            )
            state[unsafe_offset=off] = updated
            value += updated * qkv[unsafe_offset=h * 128 + k]
        value += shuffle_xor(value, UInt32(8))
        value += shuffle_xor(value, UInt32(16))
        value += shuffle_xor(value, UInt32(32))
        if part == 0:
            y[unsafe_offset=i] = value


def gated_norm(
    x: Pointer[Float32, MutAnyOrigin],
    z: Pointer[Float32, MutAnyOrigin],
    w: Pointer[Float32, MutAnyOrigin],
    y: Pointer[Float32, MutAnyOrigin],
):
    var i = global_idx.x
    if i < 2048:
        var ss: Float32 = 0
        var base = (i // 128) * 128
        var lane = i % 64
        comptime for chunk in range(2):
            var v = x[unsafe_offset=base + lane + chunk * 64]
            ss += v * v
        comptime for stage in range(6):
            ss += shuffle_xor(ss, UInt32(1 << stage))
        var gate = z[unsafe_offset=i]
        y[unsafe_offset=i] = (
            x[unsafe_offset=i]
            * rsqrt(ss / 128 + Float32(1e-6))
            * w[unsafe_offset=i % 128]
            * gate
            * sigmoid(gate)
        )


def head_norm(
    x: Pointer[Float32, MutAnyOrigin],
    w: Pointer[Float32, MutAnyOrigin],
    y: Pointer[Float32, MutAnyOrigin],
    mode: Int32,
):
    var i = global_idx.x
    var width = 2048 if mode == 0 else 512
    if i < width:
        var h = i // 256
        var base = h * (512 if mode == 0 else 256)
        var ss: Float32 = 0
        var lane = i % 64
        comptime for chunk in range(4):
            var v = x[unsafe_offset=base + lane + chunk * 64]
            ss += v * v
        comptime for stage in range(6):
            ss += shuffle_xor(ss, UInt32(1 << stage))
        y[unsafe_offset=i] = (
            x[unsafe_offset=base + i % 256]
            * rsqrt(ss / 256 + Float32(1e-6))
            * w[unsafe_offset=i % 256]
        )


def rope(
    x: Pointer[Float32, MutAnyOrigin],
    table: Pointer[Float32, MutAnyOrigin],
    width: Int32,
    position: Int32,
):
    var i = global_idx.x
    if i < Int(width) // 8:
        var head = i // 32
        var dim = i % 32
        var off = head * 256 + dim
        var a = x[unsafe_offset=off]
        var b = x[unsafe_offset=off + 32]
        var c = table[unsafe_offset=Int(position) * 64 + dim]
        var s = table[unsafe_offset=Int(position) * 64 + dim + 32]
        x[unsafe_offset=off] = a * c - b * s
        x[unsafe_offset=off + 32] = b * c + a * s


def cache_kv(
    k: Pointer[Float32, MutAnyOrigin],
    v: Pointer[Float32, MutAnyOrigin],
    kc: Pointer[Float32, MutAnyOrigin],
    vc: Pointer[Float32, MutAnyOrigin],
    position: Int32,
):
    var i = global_idx.x
    if i < 512:
        kc[unsafe_offset=Int(position) * 512 + i] = k[unsafe_offset=i]
        vc[unsafe_offset=Int(position) * 512 + i] = v[unsafe_offset=i]


def attention_scores(
    q: Pointer[Float32, MutAnyOrigin],
    kc: Pointer[Float32, MutAnyOrigin],
    y: Pointer[Float32, MutAnyOrigin],
    position: Int32,
    capacity: Int32,
):
    var i = global_idx.x
    var length = Int(position) + 1
    if i < 8 * length:
        var h = i // length
        var t = i % length
        var value: Float32 = 0
        for d in range(256):
            value += (
                q[unsafe_offset=h * 256 + d]
                * kc[unsafe_offset=t * 512 + (h // 4) * 256 + d]
            )
        y[unsafe_offset=h * Int(capacity) + t] = value * Float32(0.0625)


def attention_values(
    scores: Pointer[Float32, MutAnyOrigin],
    vc: Pointer[Float32, MutAnyOrigin],
    qg: Pointer[Float32, MutAnyOrigin],
    y: Pointer[Float32, MutAnyOrigin],
    position: Int32,
    capacity: Int32,
):
    var i = global_idx.x
    if i < 2048:
        var h = i // 256
        var d = i % 256
        var highest: Float32 = -3.402823e38
        for t in range(Int(position) + 1):
            highest = max(highest, scores[unsafe_offset=h * Int(capacity) + t])
        var total: Float32 = 0
        var value: Float32 = 0
        for t in range(Int(position) + 1):
            var p = exp(scores[unsafe_offset=h * Int(capacity) + t] - highest)
            total += p
            value += p * vc[unsafe_offset=t * 512 + (h // 4) * 256 + d]
        y[unsafe_offset=i] = (value / total) * sigmoid(
            qg[unsafe_offset=h * 512 + 256 + d]
        )


def residual(
    x: Pointer[Float32, MutAnyOrigin], y: Pointer[Float32, MutAnyOrigin]
):
    var i = global_idx.x
    if i < 2048:
        x[unsafe_offset=i] += y[unsafe_offset=i]


def swiglu(
    gate: Pointer[Float32, MutAnyOrigin], up: Pointer[Float32, MutAnyOrigin]
):
    var i = global_idx.x
    if i < 6144:
        var g = gate[unsafe_offset=i]
        gate[unsafe_offset=i] = g * sigmoid(g) * up[unsafe_offset=i]


def copy(x: Pointer[Float32, MutAnyOrigin], y: Pointer[Float32, MutAnyOrigin]):
    var i = global_idx.x
    if i < 2048:
        y[unsafe_offset=i] = x[unsafe_offset=i]

def hadamard_init(x: Pointer[Float32, MutAnyOrigin], signs: Pointer[Float32, MutAnyOrigin], y: Pointer[Float32, MutAnyOrigin], width: Int32, inverse: Int32):
    var i = global_idx.x
    if i < Int(width):
        y[unsafe_offset=i] = x[unsafe_offset=i] * (signs[unsafe_offset=i] if inverse == 0 else Float32(1))


def hadamard_stage(x: Pointer[Float32, MutAnyOrigin], y: Pointer[Float32, MutAnyOrigin], width: Int32, mask: Int32):
    var i = global_idx.x
    if i < Int(width):
        var other = x[unsafe_offset=i ^ Int(mask)]
        var value = x[unsafe_offset=i]
        y[unsafe_offset=i] = other - value if (i & Int(mask)) != 0 else value + other


def hadamard_final(x: Pointer[Float32, MutAnyOrigin], signs: Pointer[Float32, MutAnyOrigin], y: Pointer[Float32, MutAnyOrigin], q: Pointer[UInt32, MutAnyOrigin], width: Int32, inverse: Int32):
    var i = global_idx.x * 2
    if i < Int(width):
        var a = x[unsafe_offset=i] * Float32(0.04419417382415922)
        var b = x[unsafe_offset=i+1] * Float32(0.04419417382415922)
        if inverse != 0:
            a *= signs[unsafe_offset=i]
            b *= signs[unsafe_offset=i+1]
        y[unsafe_offset=i] = a
        y[unsafe_offset=i+1] = b
        if inverse == 0:
            q[unsafe_offset=i//2] = bitcast[.uint32, 1](SIMD[.float32, 2](a, b).cast[.float16]())


def portable_linear(x: Pointer[UInt32, MutAnyOrigin], w: Pointer[UInt32, MutAnyOrigin], params: Pointer[Float16, MutAnyOrigin], partials: Pointer[Float32, MutAnyOrigin], width: Int32, reduction: Int32):
    var i = global_idx.x * 8
    if i + 8 <= Int(width):
        var value = SIMD[.float32, 8](0)
        for n in range(8):
            var group = block_idx.y * 8 + n
            if group < Int(reduction)//128:
                var scales = params.unsafe_load[width=8](group * Int(width) + i).cast[.float32]()
                value += _group_dot8(x, w, Int(width), i, group) * scales
        partials.unsafe_store[width=8](block_idx.y * Int(width) + i, value)


def portable_normalize_qk(x: Pointer[Float32, MutAnyOrigin], y: Pointer[Float32, MutAnyOrigin]):
    var i = global_idx.x
    if i < 6144:
        var value = x[unsafe_offset=i]
        if i < 4096:
            var base = (i // 128) * 128
            var ss: Float32 = 0
            for j in range(128):
                var v = x[unsafe_offset=base+j]
                ss += v*v
            value *= rsqrt(ss + Float32(1e-6))
            if i < 2048:
                value *= Float32(0.08838834764831845)
        y[unsafe_offset=i] = value


def portable_delta_update(qkv: Pointer[Float32, MutAnyOrigin], gates: Pointer[Float32, MutAnyOrigin], state: Pointer[Float32, MutAnyOrigin], y: Pointer[Float32, MutAnyOrigin]):
    var i = global_idx.x
    if i < 2048:
        var h = i // 128
        var d = i % 128
        var decay = gates[unsafe_offset=h*2]
        var beta = gates[unsafe_offset=h*2+1]
        var predicted: Float32 = 0
        for k in range(128):
            predicted += state[unsafe_offset=(h*128+k)*128+d] * decay * qkv[unsafe_offset=2048+h*128+k]
        var delta = (qkv[unsafe_offset=4096+i] - predicted) * beta
        var value: Float32 = 0
        for k in range(128):
            var off = (h*128+k)*128+d
            var updated = state[unsafe_offset=off]*decay + qkv[unsafe_offset=2048+h*128+k]*delta
            state[unsafe_offset=off] = updated
            value += updated * qkv[unsafe_offset=h*128+k]
        y[unsafe_offset=i] = value


def portable_gated_norm(x: Pointer[Float32, MutAnyOrigin], z: Pointer[Float32, MutAnyOrigin], w: Pointer[Float32, MutAnyOrigin], y: Pointer[Float32, MutAnyOrigin]):
    var i = global_idx.x
    if i < 2048:
        var base = (i//128)*128
        var ss: Float32 = 0
        for j in range(128):
            var v = x[unsafe_offset=base+j]
            ss += v*v
        var gate = z[unsafe_offset=i]
        y[unsafe_offset=i] = x[unsafe_offset=i] * rsqrt(ss/128 + Float32(1e-6)) * w[unsafe_offset=i%128] * gate * sigmoid(gate)


def portable_head_norm(x: Pointer[Float32, MutAnyOrigin], w: Pointer[Float32, MutAnyOrigin], y: Pointer[Float32, MutAnyOrigin], mode: Int32):
    var i = global_idx.x
    var width = 2048 if mode == 0 else 512
    if i < width:
        var h = i//256
        var base = h*(512 if mode == 0 else 256)
        var ss: Float32 = 0
        for j in range(256):
            var v = x[unsafe_offset=base+j]
            ss += v*v
        y[unsafe_offset=i] = x[unsafe_offset=base+i%256] * rsqrt(ss/256 + Float32(1e-6)) * w[unsafe_offset=i%256]
