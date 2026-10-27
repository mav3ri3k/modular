# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the Apache License v2.0 with LLVM Exceptions: https://llvm.org/LICENSE.txt
"""Scalar FP32 activations / BF16 weights for the Android SmolLM2 prototype."""
from std.memory import Pointer, bitcast
from std.math import exp, rsqrt
from max.gpu import global_idx


@always_inline
def bf16(value: UInt16) -> Float32:
    return bitcast[.float32](UInt32(value) << 16)


def embedding(
    w: Pointer[UInt16, MutAnyOrigin],
    x: Pointer[Float32, MutAnyOrigin],
    token: Int32,
):
    var i = global_idx.x
    if i < 576:
        x[unsafe_offset=i] = bf16(w[unsafe_offset=Int(token) * 576 + i])


def rms_norm(
    x: Pointer[Float32, MutAnyOrigin],
    w: Pointer[UInt16, MutAnyOrigin],
    y: Pointer[Float32, MutAnyOrigin],
):
    var i = global_idx.x
    if i < 576:
        var ss: Float32 = 0
        for j in range(576):
            var v = x[unsafe_offset=j]
            ss += v * v
        y[unsafe_offset=i] = (
            x[unsafe_offset=i]
            * rsqrt(ss / 576 + Float32(1e-5))
            * bf16(w[unsafe_offset=i])
        )


def linear(
    x: Pointer[Float32, MutAnyOrigin],
    w: Pointer[UInt16, MutAnyOrigin],
    y: Pointer[Float32, MutAnyOrigin],
    width: Int32,
    reduction: Int32,
):
    var i = global_idx.x
    if i < Int(width):
        var value: Float32 = 0
        for j in range(Int(reduction)):
            value += x[unsafe_offset=j] * bf16(
                w[unsafe_offset=j * Int(width) + i]
            )
        y[unsafe_offset=i] = value


def rope(
    x: Pointer[Float32, MutAnyOrigin],
    table: Pointer[Float32, MutAnyOrigin],
    width: Int32,
    position: Int32,
):
    # One thread owns both members of a half-split rotary pair.
    var i = global_idx.x
    if i < Int(width) // 2:
        var head = i // 32
        var dim = i % 32
        var a = x[unsafe_offset=head * 64 + dim]
        var b = x[unsafe_offset=head * 64 + dim + 32]
        var c = table[unsafe_offset=Int(position) * 64 + dim]
        var s = table[unsafe_offset=Int(position) * 64 + dim + 32]
        x[unsafe_offset=head * 64 + dim] = a * c - b * s
        x[unsafe_offset=head * 64 + dim + 32] = b * c + a * s


def cache_kv(
    k: Pointer[Float32, MutAnyOrigin],
    v: Pointer[Float32, MutAnyOrigin],
    kc: Pointer[Float32, MutAnyOrigin],
    vc: Pointer[Float32, MutAnyOrigin],
    layer: Int32,
    position: Int32,
    capacity: Int32,
):
    var i = global_idx.x
    if i < 192:
        var offset = (Int(layer) * Int(capacity) + Int(position)) * 192 + i
        kc[unsafe_offset=offset] = k[unsafe_offset=i]
        vc[unsafe_offset=offset] = v[unsafe_offset=i]


def attention_scores(
    q: Pointer[Float32, MutAnyOrigin],
    kc: Pointer[Float32, MutAnyOrigin],
    scores: Pointer[Float32, MutAnyOrigin],
    layer: Int32,
    position: Int32,
    capacity: Int32,
):
    var i = global_idx.x
    var length = Int(position) + 1
    if i < 9 * length:
        var h = i // length
        var t = i % length
        var value: Float32 = 0
        for d in range(64):
            value += (
                q[unsafe_offset=h * 64 + d]
                * kc[
                    unsafe_offset=(Int(layer) * Int(capacity) + t) * 192
                    + (h // 3) * 64
                    + d
                ]
            )
        scores[unsafe_offset=h * Int(capacity) + t] = value * Float32(0.125)


def attention_values(
    scores: Pointer[Float32, MutAnyOrigin],
    vc: Pointer[Float32, MutAnyOrigin],
    output: Pointer[Float32, MutAnyOrigin],
    layer: Int32,
    position: Int32,
    capacity: Int32,
):
    # Recomputing the softmax per output avoids an extra buffer and barrier.
    var i = global_idx.x
    if i < 576:
        var h = i // 64
        var d = i % 64
        var highest: Float32 = -3.402823e38
        for t in range(Int(position) + 1):
            highest = max(highest, scores[unsafe_offset=h * Int(capacity) + t])
        var total: Float32 = 0
        var value: Float32 = 0
        for t in range(Int(position) + 1):
            var probability = exp(
                scores[unsafe_offset=h * Int(capacity) + t] - highest
            )
            total += probability
            value += (
                probability
                * vc[
                    unsafe_offset=(Int(layer) * Int(capacity) + t) * 192
                    + (h // 3) * 64
                    + d
                ]
            )
        output[unsafe_offset=i] = value / total


def residual(
    x: Pointer[Float32, MutAnyOrigin], y: Pointer[Float32, MutAnyOrigin]
):
    var i = global_idx.x
    if i < 576:
        x[unsafe_offset=i] += y[unsafe_offset=i]


def swiglu(
    gate: Pointer[Float32, MutAnyOrigin], up: Pointer[Float32, MutAnyOrigin]
):
    var i = global_idx.x
    if i < 1536:
        var g = gate[unsafe_offset=i]
        gate[unsafe_offset=i] = (g / (1 + exp(-g))) * up[unsafe_offset=i]
