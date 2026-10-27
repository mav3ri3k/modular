# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the Apache License v2.0 with LLVM Exceptions: https://llvm.org/LICENSE.txt
"""GGUF Q4_K/Q5_K/Q6_K/Q8_0 blocks; byte planes preserve all source bits."""
from std.memory import Pointer, bitcast
from max.gpu import global_idx, block_idx


@always_inline
def load16(w: Pointer[UInt8, MutAnyOrigin], off: Int, stride: Int) -> Float32:
    var bits = UInt16(w[unsafe_offset=off]) | (UInt16(w[unsafe_offset=off + stride]) << 8)
    return Float32(bitcast[.float16](bits))


@always_inline
def value(w: Pointer[UInt8, MutAnyOrigin], row: Int, j: Int, width: Int) -> Float32:
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
        return load16(w, off, width) * Float32(bitcast[.int8](w[unsafe_offset=off + (2 + k) * width]))
    if qt == 14:
        var half = k // 128
        var sub = (k % 128) // 32
        var lane = k % 32
        var lo = UInt32(w[unsafe_offset=off + (half * 64 + (sub % 2) * 32 + lane) * width])
        lo = (lo >> UInt32((sub // 2) * 4)) & 15
        var hi = (UInt32(w[unsafe_offset=off + (128 + half * 32 + lane) * width]) >> UInt32(sub * 2)) & 3
        var sc = bitcast[.int8](w[unsafe_offset=off + (192 + k // 16) * width])
        return load16(w, off + 208 * width, width) * Float32(sc) * Float32(Int32(lo | (hi << 4)) - 32)
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
    var q = (UInt32(w[unsafe_offset=off + (qoff + (k // 64) * 32 + k % 32) * width]) >> UInt32((g % 2) * 4)) & 15
    if qt == 13:
        var hi = (UInt32(w[unsafe_offset=off + (16 + k % 32) * width]) >> UInt32(g)) & 1
        q |= hi << 4
    return load16(w, off, width) * Float32(sc) * Float32(q) - load16(w, off + 2 * width, width) * Float32(mn)


@always_inline
def load16x8(w: Pointer[UInt8, MutAnyOrigin], off: Int, stride: Int) -> SIMD[.float32, 8]:
    var bits = w.unsafe_load[width=8](off).cast[.uint16]() | (w.unsafe_load[width=8](off + stride).cast[.uint16]() << 8)
    return bitcast[.float16, 8](bits).cast[.float32]()


@always_inline
def value8(w: Pointer[UInt8, MutAnyOrigin], row: Int, j: Int, width: Int) -> SIMD[.float32, 8]:
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
            bits |= w.unsafe_load[width=8](off + b * width).cast[.uint32]() << UInt32(b * 8)
        return bitcast[.float32, 8](bits)
    if qt == 8:
        return load16x8(w, off, width) * bitcast[.int8, 8](w.unsafe_load[width=8](off + (2 + k) * width)).cast[.float32]()
    if qt == 14:
        var half = k // 128
        var sub = (k % 128) // 32
        var lane = k % 32
        var lo = w.unsafe_load[width=8](off + (half * 64 + (sub % 2) * 32 + lane) * width).cast[.uint32]()
        lo = (lo >> UInt32((sub // 2) * 4)) & 15
        var hi = (w.unsafe_load[width=8](off + (128 + half * 32 + lane) * width).cast[.uint32]() >> UInt32(sub * 2)) & 3
        var sc = bitcast[.int8, 8](w.unsafe_load[width=8](off + (192 + k // 16) * width))
        return load16x8(w, off + 208 * width, width) * sc.cast[.float32]() * ((lo | (hi << 4)).cast[.int32]() - 32).cast[.float32]()
    var g = k // 32
    var sc = w.unsafe_load[width=8](off + (4 + g % 4) * width).cast[.uint32]()
    var mn = w.unsafe_load[width=8](off + (8 + g % 4) * width).cast[.uint32]()
    if g < 4:
        sc &= 63
        mn &= 63
    else:
        var both = w.unsafe_load[width=8](off + (12 + g % 4) * width).cast[.uint32]()
        sc = (both & 15) | ((sc >> 2) & 48)
        mn = (both >> 4) | ((mn >> 2) & 48)
    var qoff = 16
    if qt == 13:
        qoff = 48
    var q = (w.unsafe_load[width=8](off + (qoff + (k // 64) * 32 + k % 32) * width).cast[.uint32]() >> UInt32((g % 2) * 4)) & 15
    if qt == 13:
        var hi = (w.unsafe_load[width=8](off + (16 + k % 32) * width).cast[.uint32]() >> UInt32(g)) & 1
        q |= hi << 4
    return load16x8(w, off, width) * sc.cast[.float32]() * q.cast[.float32]() - load16x8(w, off + 2 * width, width) * mn.cast[.float32]()


def linear(x: Pointer[Float32, MutAnyOrigin], w: Pointer[UInt8, MutAnyOrigin], y: Pointer[Float32, MutAnyOrigin], width: Int32, reduction: Int32):
    var row = global_idx.x * 8
    var part = block_idx.y
    if row < Int(width):
        var acc = SIMD[.float32, 8](0)
        for j in range(part * 256, min((part + 1) * 256, Int(reduction))):
            acc += x[unsafe_offset=j] * value8(w, row, j, Int(width))
        y.unsafe_store[width=8](part * Int(width) + row, acc)


def reduce(x: Pointer[Float32, MutAnyOrigin], y: Pointer[Float32, MutAnyOrigin], width: Int32, reduction: Int32):
    var row = global_idx.x
    if row < Int(width):
        var acc: Float32 = 0
        for part in range((Int(reduction) + 255) // 256):
            acc += x[unsafe_offset=part * Int(width) + row]
        y[unsafe_offset=row] = acc


def embedding(w: Pointer[UInt8, MutAnyOrigin], y: Pointer[Float32, MutAnyOrigin], token: Int32, vocabulary: Int32):
    var i = global_idx.x
    if i < 2048:
        y[unsafe_offset=i] = value(w, Int(token), i, Int(vocabulary))
