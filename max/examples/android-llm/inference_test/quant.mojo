# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the Apache License v2.0 with LLVM Exceptions: https://llvm.org/LICENSE.txt
"""GGUF Q4_K/Q5_K/Q6_K/Q8_0 blocks; byte planes preserve all source bits."""
from std.memory import Pointer, bitcast
from max.gpu import global_idx, block_idx


@always_inline
def load16(w: Pointer[UInt8, MutAnyOrigin], off: Int, stride: Int) -> Float32:
    var bits = UInt16(w[unsafe_offset=off]) | (
        UInt16(w[unsafe_offset=off + stride]) << 8
    )
    return Float32(bitcast[.float16](bits))


@always_inline
def value(
    w: Pointer[UInt8, MutAnyOrigin], row: Int, j: Int, width: Int
) -> Float32:
    var qt = Int(w[unsafe_offset=0])
    var block = 256
    var size = 144
    if qt == 13:
        size = 176
    elif qt == 14:
        size = 210
    elif qt == 8:
        block = 32
        size = 34
    elif qt == 0:
        block = 1
        size = 4
    var off = 16 + (j // block) * size * width + row
    var k = j % block
    if qt == 0:
        var bits: UInt32 = 0
        comptime for b in range(4):
            bits |= UInt32(w[unsafe_offset=off + b * width]) << UInt32(b * 8)
        return bitcast[.float32](bits)
    if qt == 8:
        return load16(w, off, width) * Float32(
            bitcast[.int8](w[unsafe_offset=off + (2 + k) * width])
        )
    if qt == 14:
        var half = k // 128
        var sub = (k % 128) // 32
        var lane = k % 32
        var lo = UInt32(
            w[unsafe_offset=off + (half * 64 + (sub % 2) * 32 + lane) * width]
        )
        lo = (lo >> UInt32((sub // 2) * 4)) & 15
        var hi = (
            UInt32(w[unsafe_offset=off + (128 + half * 32 + lane) * width])
            >> UInt32(sub * 2)
        ) & 3
        var sc = bitcast[.int8](w[unsafe_offset=off + (192 + k // 16) * width])
        return (
            load16(w, off + 208 * width, width)
            * Float32(sc)
            * Float32(Int32(lo | (hi << 4)) - 32)
        )
    var g = k // 32
    var sc = Int(w[unsafe_offset=off + (4 + g % 4) * width])
    var mn = Int(w[unsafe_offset=off + (8 + g % 4) * width])
    if g < 4:
        sc &= 63
        mn &= 63
    else:
        var both = Int(w[unsafe_offset=off + (12 + g % 4) * width])
        sc = (both & 15) | ((sc >> 2) & 48)
        mn = (both >> 4) | ((mn >> 2) & 48)
    var qoff = 16
    if qt == 13:
        qoff = 48
    var q = (
        UInt32(w[unsafe_offset=off + (qoff + (k // 64) * 32 + k % 32) * width])
        >> UInt32((g % 2) * 4)
    ) & 15
    if qt == 13:
        var hi = (
            UInt32(w[unsafe_offset=off + (16 + k % 32) * width]) >> UInt32(g)
        ) & 1
        q |= hi << 4
    return load16(w, off, width) * Float32(sc) * Float32(q) - load16(
        w, off + 2 * width, width
    ) * Float32(mn)


@always_inline
def load16x8(
    w: Pointer[UInt8, MutAnyOrigin], off: Int, stride: Int
) -> SIMD[.float32, 8]:
    var bits = w.unsafe_load[width=8](off).cast[.uint16]() | (
        w.unsafe_load[width=8](off + stride).cast[.uint16]() << 8
    )
    return bitcast[.float16, 8](bits).cast[.float32]()


@always_inline
def value8(
    w: Pointer[UInt8, MutAnyOrigin], row: Int, j: Int, width: Int
) -> SIMD[.float32, 8]:
    var qt = Int(w[unsafe_offset=0])
    var block = 256
    var size = 144
    if qt == 13:
        size = 176
    elif qt == 14:
        size = 210
    elif qt == 8:
        block = 32
        size = 34
    elif qt == 0:
        block = 1
        size = 4
    var off = 16 + (j // block) * size * width + row
    var k = j % block
    if qt == 0:
        var bits = SIMD[.uint32, 8](0)
        comptime for b in range(4):
            bits |= w.unsafe_load[width=8](off + b * width).cast[
                .uint32
            ]() << UInt32(b * 8)
        return bitcast[.float32, 8](bits)
    if qt == 8:
        return (
            load16x8(w, off, width)
            * bitcast[.int8, 8](
                w.unsafe_load[width=8](off + (2 + k) * width)
            ).cast[.float32]()
        )
    if qt == 14:
        var half = k // 128
        var sub = (k % 128) // 32
        var lane = k % 32
        var lo = w.unsafe_load[width=8](
            off + (half * 64 + (sub % 2) * 32 + lane) * width
        ).cast[.uint32]()
        lo = (lo >> UInt32((sub // 2) * 4)) & 15
        var hi = (
            w.unsafe_load[width=8](off + (128 + half * 32 + lane) * width).cast[
                .uint32
            ]()
            >> UInt32(sub * 2)
        ) & 3
        var sc = bitcast[.int8, 8](
            w.unsafe_load[width=8](off + (192 + k // 16) * width)
        )
        return (
            load16x8(w, off + 208 * width, width)
            * sc.cast[.float32]()
            * ((lo | (hi << 4)).cast[.int32]() - 32).cast[.float32]()
        )
    var g = k // 32
    var sc = w.unsafe_load[width=8](off + (4 + g % 4) * width).cast[.uint32]()
    var mn = w.unsafe_load[width=8](off + (8 + g % 4) * width).cast[.uint32]()
    if g < 4:
        sc &= 63
        mn &= 63
    else:
        var both = w.unsafe_load[width=8](off + (12 + g % 4) * width).cast[
            .uint32
        ]()
        sc = (both & 15) | ((sc >> 2) & 48)
        mn = (both >> 4) | ((mn >> 2) & 48)
    var qoff = 16
    if qt == 13:
        qoff = 48
    var q = (
        w.unsafe_load[width=8](
            off + (qoff + (k // 64) * 32 + k % 32) * width
        ).cast[.uint32]()
        >> UInt32((g % 2) * 4)
    ) & 15
    if qt == 13:
        var hi = (
            w.unsafe_load[width=8](off + (16 + k % 32) * width).cast[.uint32]()
            >> UInt32(g)
        ) & 1
        q |= hi << 4
    return (
        load16x8(w, off, width) * sc.cast[.float32]() * q.cast[.float32]()
        - load16x8(w, off + 2 * width, width) * mn.cast[.float32]()
    )


@always_inline
def bytes8(w: Pointer[UInt8, MutAnyOrigin], off: Int) -> SIMD[.uint32, 8]:
    return w.unsafe_load[width=8](off).cast[.uint32]()


@always_inline
def q4_pair(
    x: Pointer[Float32, MutAnyOrigin],
    w: Pointer[UInt8, MutAnyOrigin],
    d: SIMD[.float32, 8],
    dmin: SIMD[.float32, 8],
    sc_lo: SIMD[.uint32, 8],
    mn_lo: SIMD[.uint32, 8],
    sc_hi: SIMD[.uint32, 8],
    mn_hi: SIMD[.uint32, 8],
    off: Int,
    width: Int,
    kbase: Int,
    group: Int,
) -> SIMD[.float32, 8]:
    var scale_lo = d * sc_lo.cast[.float32]()
    var scale_hi = d * sc_hi.cast[.float32]()
    var min_lo = dmin * mn_lo.cast[.float32]()
    var min_hi = dmin * mn_hi.cast[.float32]()
    var base = off + (16 + group * 32) * width
    var even = SIMD[.float32, 8](0)
    var odd = SIMD[.float32, 8](0)
    for lane in range(16):
        var q0 = bytes8(w, base + (lane * 2) * width)
        var q1 = bytes8(w, base + (lane * 2 + 1) * width)
        var k0 = kbase + group * 64 + lane * 2
        even += x[unsafe_offset=k0] * (
            scale_lo * (q0 & 15).cast[.float32]() - min_lo
        )
        even += x[unsafe_offset=k0 + 32] * (
            scale_hi * ((q0 >> 4) & 15).cast[.float32]() - min_hi
        )
        odd += x[unsafe_offset=k0 + 1] * (
            scale_lo * (q1 & 15).cast[.float32]() - min_lo
        )
        odd += x[unsafe_offset=k0 + 33] * (
            scale_hi * ((q1 >> 4) & 15).cast[.float32]() - min_hi
        )
    return even + odd


@always_inline
def q4_block(
    x: Pointer[Float32, MutAnyOrigin],
    w: Pointer[UInt8, MutAnyOrigin],
    row: Int,
    width: Int,
    kbase: Int,
) -> SIMD[.float32, 8]:
    """One Q4_K block. Each byte plane is loaded once for these eight rows."""
    var off = 16 + (kbase // 256) * 144 * width + row
    var d = load16x8(w, off, width)
    var dmin = load16x8(w, off + 2 * width, width)
    var s0 = bytes8(w, off + 4 * width)
    var s1 = bytes8(w, off + 5 * width)
    var s2 = bytes8(w, off + 6 * width)
    var s3 = bytes8(w, off + 7 * width)
    var m0 = bytes8(w, off + 8 * width)
    var m1 = bytes8(w, off + 9 * width)
    var m2 = bytes8(w, off + 10 * width)
    var m3 = bytes8(w, off + 11 * width)
    var b0 = bytes8(w, off + 12 * width)
    var b1 = bytes8(w, off + 13 * width)
    var b2 = bytes8(w, off + 14 * width)
    var b3 = bytes8(w, off + 15 * width)
    return (
        q4_pair(
            x, w, d, dmin, s0 & 63, m0 & 63, s1 & 63, m1 & 63, off, width, kbase, 0
        ) + q4_pair(
            x, w, d, dmin, s2 & 63, m2 & 63, s3 & 63, m3 & 63, off, width, kbase, 1
        )
    ) + (
        q4_pair(
            x,
            w,
            d,
            dmin,
            (b0 & 15) | ((s0 >> 2) & 48),
            (b0 >> 4) | ((m0 >> 2) & 48),
            (b1 & 15) | ((s1 >> 2) & 48),
            (b1 >> 4) | ((m1 >> 2) & 48),
            off,
            width,
            kbase,
            2,
        ) + q4_pair(
            x,
            w,
            d,
            dmin,
            (b2 & 15) | ((s2 >> 2) & 48),
            (b2 >> 4) | ((m2 >> 2) & 48),
            (b3 & 15) | ((s3 >> 2) & 48),
            (b3 >> 4) | ((m3 >> 2) & 48),
            off,
            width,
            kbase,
            3,
        )
    )


@always_inline
def q6_block(
    x: Pointer[Float32, MutAnyOrigin],
    w: Pointer[UInt8, MutAnyOrigin],
    row: Int,
    width: Int,
    kbase: Int,
) -> SIMD[.float32, 8]:
    """One Q6_K block. ql/qh/scale planes are loaded once per nibble group."""
    var off = 16 + (kbase // 256) * 210 * width + row
    var d = load16x8(w, off + 208 * width, width)
    var acc = SIMD[.float32, 8](0)
    for half in range(2):
        for sub in range(4):
            var ql = off + (half * 64 + (sub % 2) * 32) * width
            var qh = off + (128 + half * 32) * width
            var shift = UInt32((sub // 2) * 4)
            var hshift = UInt32(sub * 2)
            var k0 = kbase + half * 128 + sub * 32
            for group in range(2):
                var scale = d * bitcast[.int8, 8](
                    w.unsafe_load[width=8](
                        off + (192 + half * 8 + sub * 2 + group) * width
                    )
                ).cast[.float32]()
                var even = SIMD[.float32, 8](0)
                var odd = SIMD[.float32, 8](0)
                var base = group * 16
                for lane in range(8):
                    var i0 = base + lane * 2
                    var i1 = i0 + 1
                    var p0 = bytes8(w, ql + i0 * width)
                    var p1 = bytes8(w, ql + i1 * width)
                    var h0 = bytes8(w, qh + i0 * width)
                    var h1 = bytes8(w, qh + i1 * width)
                    var n0 = ((p0 >> shift) & 15).cast[.int32]()
                    var n1 = ((p1 >> shift) & 15).cast[.int32]()
                    var z0 = ((h0 >> hshift) & 3).cast[.int32]()
                    var z1 = ((h1 >> hshift) & 3).cast[.int32]()
                    even += x[unsafe_offset=k0 + i0] * scale * (
                        (n0 | (z0 << 4)) - 32
                    ).cast[.float32]()
                    odd += x[unsafe_offset=k0 + i1] * scale * (
                        (n1 | (z1 << 4)) - 32
                    ).cast[.float32]()
                acc += even + odd
    return acc


def linear(
    x: Pointer[Float32, MutAnyOrigin],
    w: Pointer[UInt8, MutAnyOrigin],
    y: Pointer[Float32, MutAnyOrigin],
    width: Int32,
    reduction: Int32,
    parts: Int32,
):
    var row = global_idx.x * 8
    if row < Int(width):
        var begin = 0
        var end = Int(reduction)
        if Int(parts) > 1:
            begin = block_idx.y * 256
            end = min(begin + 256, Int(reduction))
        var acc = SIMD[.float32, 8](0)
        if begin < end:
            var qt = Int(w[unsafe_offset=0])
            var k = begin
            while k + 256 <= end:
                if qt == 14:
                    acc += q6_block(x, w, row, Int(width), k)
                elif qt == 12:
                    acc += q4_block(x, w, row, Int(width), k)
                else:
                    for j in range(k, k + 256):
                        acc += x[unsafe_offset=j] * value8(w, row, j, Int(width))
                k += 256
            for j in range(k, end):
                acc += x[unsafe_offset=j] * value8(w, row, j, Int(width))
        y.unsafe_store[width=8](block_idx.y * Int(width) + row, acc)


def reduce(
    x: Pointer[Float32, MutAnyOrigin],
    y: Pointer[Float32, MutAnyOrigin],
    width: Int32,
    reduction: Int32,
):
    var row = global_idx.x
    if row < Int(width):
        var acc: Float32 = 0
        for part in range((Int(reduction) + 255) // 256):
            acc += x[unsafe_offset=part * Int(width) + row]
        y[unsafe_offset=row] = acc


def embedding(
    w: Pointer[UInt8, MutAnyOrigin],
    y: Pointer[Float32, MutAnyOrigin],
    token: Int32,
    vocabulary: Int32,
):
    var i = global_idx.x
    if i < 2048:
        y[unsafe_offset=i] = value(w, Int(token), i, Int(vocabulary))
