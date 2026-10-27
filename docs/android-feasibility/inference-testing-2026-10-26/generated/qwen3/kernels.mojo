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
    w: Pointer[Float32, MutAnyOrigin],
    x: Pointer[Float32, MutAnyOrigin],
    token: Int32,
):
    var i = global_idx.x
    if i < 2048:
        x[unsafe_offset=i] = bf16(w[unsafe_offset=Int(token) * 2048 + i])





def linear(
    x: Pointer[Float32, MutAnyOrigin],
    w: Pointer[Float32, MutAnyOrigin],
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


def rope(x: Pointer[Float32, MutAnyOrigin],table: Pointer[Float32, MutAnyOrigin],width: Int32,position: Int32):
    var i=global_idx.x
    if i<Int(width)//2:
        var h=i//64
        var d=i%64
        var off=h*128+d
        var c=table[unsafe_offset=Int(position)*128+d]
        var s=table[unsafe_offset=Int(position)*128+d+64]
        var a=x[unsafe_offset=off]
        var b=x[unsafe_offset=off+64]
        x[unsafe_offset=off]=a*c-b*s
        x[unsafe_offset=off+64]=b*c+a*s



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
    if i < 1024:
        var offset = (Int(layer) * Int(capacity) + Int(position)) * 1024 + i
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
    if i < 16 * length:
        var h = i // length
        var t = i % length
        var value: Float32 = 0
        for d in range(128):
            value += (
                q[unsafe_offset=h * 128 + d]
                * kc[
                    unsafe_offset=(Int(layer) * Int(capacity) + t) * 1024
                    + (h // 2) * 128
                    + d
                ]
            )
        scores[unsafe_offset=h * Int(capacity) + t] = value * Float32(0.08838834764831845)


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
    if i < 2048:
        var h = i // 128
        var d = i % 128
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
                    unsafe_offset=(Int(layer) * Int(capacity) + t) * 1024
                    + (h // 2) * 128
                    + d
                ]
            )
        output[unsafe_offset=i] = value / total


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
        gate[unsafe_offset=i] = (g / (1 + exp(-g))) * up[unsafe_offset=i]


def qk_norm(x: Pointer[Float32, MutAnyOrigin],w: Pointer[Float32, MutAnyOrigin],width: Int32):
    var i=global_idx.x
    if i<Int(width) and i%128==0:
        var base=(i//128)*128
        var ss:Float32=0
        for d in range(128):
            var v=x[unsafe_offset=base+d]
            ss+=v*v
        # Each work item owns one complete head to avoid in-place read/write races.
        # Only the first lane executes; this scalar baseline is intentionally simple.
        if i%128==0:
            var scale=rsqrt(ss/128+Float32(1e-6))
            for d in range(128):
                x[unsafe_offset=base+d]*=scale*w[unsafe_offset=d]


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
