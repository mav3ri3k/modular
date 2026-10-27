# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the Apache License v2.0 with LLVM Exceptions: https://llvm.org/LICENSE.txt
"""SmolLM2-135M forward orchestration, compiled to Android ARM64 host code.

C++ owns checkpoint I/O and sampling. Mojo owns typed DeviceBuffers and uses
ordinary DeviceContext launches on one in-order stream.
"""
from std.memory import Pointer, OptionalPointer
from std.sys import size_of
from max.gpu.host import DeviceBuffer, DeviceContext, Dim
import kernels


@always_inline
def launch_embedding(
    rt: DeviceContext,
    work: Int32,
    w: DeviceBuffer[.uint8],
    x: DeviceBuffer[.uint8],
    token: Int32,
) raises -> Int64:
    rt.enqueue_function[kernels.embedding](
        w.create_sub_buffer[.uint16](0, len(w) // size_of[UInt16]()),
        x.create_sub_buffer[.float32](0, len(x) // size_of[Float32]()),
        token,
        grid_dim=Dim((Int(work) + 63) // 64, 1, 1),
        block_dim=64,
    )
    return 0


@always_inline
def launch_rms_norm(
    rt: DeviceContext,
    work: Int32,
    x: DeviceBuffer[.uint8],
    w: DeviceBuffer[.uint8],
    y: DeviceBuffer[.uint8],
) raises -> Int64:
    rt.enqueue_function[kernels.rms_norm](
        x.create_sub_buffer[.float32](0, len(x) // size_of[Float32]()),
        w.create_sub_buffer[.uint16](0, len(w) // size_of[UInt16]()),
        y.create_sub_buffer[.float32](0, len(y) // size_of[Float32]()),
        grid_dim=Dim((Int(work) + 63) // 64, 1, 1),
        block_dim=64,
    )
    return 0


@always_inline
def launch_linear(
    rt: DeviceContext,
    work: Int32,
    x: DeviceBuffer[.uint8],
    w: DeviceBuffer[.uint8],
    y: DeviceBuffer[.uint8],
    width: Int32,
    reduction: Int32,
) raises -> Int64:
    rt.enqueue_function[kernels.linear](
        x.create_sub_buffer[.float32](0, len(x) // size_of[Float32]()),
        w.create_sub_buffer[.uint16](0, len(w) // size_of[UInt16]()),
        y.create_sub_buffer[.float32](0, len(y) // size_of[Float32]()),
        width,
        reduction,
        grid_dim=Dim((Int(work) + 63) // 64, 1, 1),
        block_dim=64,
    )
    return 0


@always_inline
def launch_rope(
    rt: DeviceContext,
    work: Int32,
    x: DeviceBuffer[.uint8],
    table: DeviceBuffer[.uint8],
    width: Int32,
    position: Int32,
) raises -> Int64:
    rt.enqueue_function[kernels.rope](
        x.create_sub_buffer[.float32](0, len(x) // size_of[Float32]()),
        table.create_sub_buffer[.float32](0, len(table) // size_of[Float32]()),
        width,
        position,
        grid_dim=Dim((Int(work) + 63) // 64, 1, 1),
        block_dim=64,
    )
    return 0


@always_inline
def launch_cache_kv(
    rt: DeviceContext,
    work: Int32,
    k: DeviceBuffer[.uint8],
    v: DeviceBuffer[.uint8],
    kc: DeviceBuffer[.uint8],
    vc: DeviceBuffer[.uint8],
    layer: Int32,
    position: Int32,
    capacity: Int32,
) raises -> Int64:
    rt.enqueue_function[kernels.cache_kv](
        k.create_sub_buffer[.float32](0, len(k) // size_of[Float32]()),
        v.create_sub_buffer[.float32](0, len(v) // size_of[Float32]()),
        kc.create_sub_buffer[.float32](0, len(kc) // size_of[Float32]()),
        vc.create_sub_buffer[.float32](0, len(vc) // size_of[Float32]()),
        layer,
        position,
        capacity,
        grid_dim=Dim((Int(work) + 63) // 64, 1, 1),
        block_dim=64,
    )
    return 0


@always_inline
def launch_attention_scores(
    rt: DeviceContext,
    work: Int32,
    q: DeviceBuffer[.uint8],
    kc: DeviceBuffer[.uint8],
    scores: DeviceBuffer[.uint8],
    layer: Int32,
    position: Int32,
    capacity: Int32,
) raises -> Int64:
    rt.enqueue_function[kernels.attention_scores](
        q.create_sub_buffer[.float32](0, len(q) // size_of[Float32]()),
        kc.create_sub_buffer[.float32](0, len(kc) // size_of[Float32]()),
        scores.create_sub_buffer[.float32](
            0, len(scores) // size_of[Float32]()
        ),
        layer,
        position,
        capacity,
        grid_dim=Dim((Int(work) + 63) // 64, 1, 1),
        block_dim=64,
    )
    return 0


@always_inline
def launch_attention_values(
    rt: DeviceContext,
    work: Int32,
    scores: DeviceBuffer[.uint8],
    vc: DeviceBuffer[.uint8],
    output: DeviceBuffer[.uint8],
    layer: Int32,
    position: Int32,
    capacity: Int32,
) raises -> Int64:
    rt.enqueue_function[kernels.attention_values](
        scores.create_sub_buffer[.float32](
            0, len(scores) // size_of[Float32]()
        ),
        vc.create_sub_buffer[.float32](0, len(vc) // size_of[Float32]()),
        output.create_sub_buffer[.float32](
            0, len(output) // size_of[Float32]()
        ),
        layer,
        position,
        capacity,
        grid_dim=Dim((Int(work) + 63) // 64, 1, 1),
        block_dim=64,
    )
    return 0


@always_inline
def launch_residual(
    rt: DeviceContext,
    work: Int32,
    x: DeviceBuffer[.uint8],
    y: DeviceBuffer[.uint8],
) raises -> Int64:
    rt.enqueue_function[kernels.residual](
        x.create_sub_buffer[.float32](0, len(x) // size_of[Float32]()),
        y.create_sub_buffer[.float32](0, len(y) // size_of[Float32]()),
        grid_dim=Dim((Int(work) + 63) // 64, 1, 1),
        block_dim=64,
    )
    return 0


@always_inline
def launch_swiglu(
    rt: DeviceContext,
    work: Int32,
    gate: DeviceBuffer[.uint8],
    up: DeviceBuffer[.uint8],
) raises -> Int64:
    rt.enqueue_function[kernels.swiglu](
        gate.create_sub_buffer[.float32](0, len(gate) // size_of[Float32]()),
        up.create_sub_buffer[.float32](0, len(up) // size_of[Float32]()),
        grid_dim=Dim((Int(work) + 63) // 64, 1, 1),
        block_dim=64,
    )
    return 0


def forward(
    rt: DeviceContext,
    w: List[DeviceBuffer[.uint8]],
    b: List[DeviceBuffer[.uint8]],
    token: Int32,
    position: Int32,
    capacity: Int32,
    head_gpu_columns: Int32 = -1,
    head_input: OptionalPointer[UInt8, MutUntrackedOrigin] = None,
) raises -> Int64:
    """Enqueue a token, updating all 30 layers' persistent key/value caches."""
    if case .Some(_) = head_input:
        return -1  # The optional Hexagon vocabulary backend is Bonsai-only.
    if token < 0 or token >= 49152 or position < 0 or position >= capacity:
        return -1
    # Buffers: x, norm, q, k, v, attention, scores, gate, up, logits,
    #          key cache, value cache, rotary table.
    var status = launch_embedding(rt, 576, w[0], b[0], token)
    if status != 0:
        return status
    for layer in range(30):
        var off = 3 + layer * 9
        status = launch_rms_norm(rt, 576, b[0], w[off], b[1])
        if status != 0:
            return status
        for p in range(3):
            var width: Int32 = 192
            if p == 0:
                width = 576
            status = launch_linear(
                rt, width, b[1], w[off + 1 + p], b[2 + p], width, 576
            )
            if status != 0:
                return status
        for p in range(2):
            var width: Int32 = 192
            if p == 0:
                width = 576
            status = launch_rope(
                rt, width // 2, b[2 + p], b[12], width, position
            )
            if status != 0:
                return status
        status = launch_cache_kv(
            rt, 192, b[3], b[4], b[10], b[11], Int32(layer), position, capacity
        )
        if status != 0:
            return status
        status = launch_attention_scores(
            rt,
            9 * (position + 1),
            b[2],
            b[10],
            b[6],
            Int32(layer),
            position,
            capacity,
        )
        if status != 0:
            return status
        status = launch_attention_values(
            rt, 576, b[6], b[11], b[5], Int32(layer), position, capacity
        )
        if status != 0:
            return status
        status = launch_linear(rt, 576, b[5], w[off + 4], b[1], 576, 576)
        if status != 0:
            return status
        status = launch_residual(rt, 576, b[0], b[1])
        if status != 0:
            return status
        status = launch_rms_norm(rt, 576, b[0], w[off + 5], b[1])
        if status != 0:
            return status
        for p in range(2):
            status = launch_linear(
                rt, 1536, b[1], w[off + 6 + p], b[7 + p], 1536, 576
            )
            if status != 0:
                return status
        status = launch_swiglu(rt, 1536, b[7], b[8])
        if status != 0:
            return status
        status = launch_linear(rt, 576, b[7], w[off + 8], b[1], 576, 1536)
        if status != 0:
            return status
        status = launch_residual(rt, 576, b[0], b[1])
        if status != 0:
            return status
    status = launch_rms_norm(rt, 576, b[0], w[2], b[1])
    if status != 0:
        return status
    return launch_linear(rt, 49152, b[1], w[1], b[9], 49152, 576)
