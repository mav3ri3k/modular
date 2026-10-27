# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the Apache License v2.0 with LLVM Exceptions: https://llvm.org/LICENSE.txt
"""Bonsai 2 2B forward orchestration, compiled to Android ARM64 host code.

C++ owns checkpoint I/O and sampling. Mojo owns typed DeviceBuffers and uses
ordinary DeviceContext launches on one in-order stream.
"""
from std.memory import Pointer, OptionalPointer
from std.sys import size_of
from max.gpu.host import DeviceBuffer, DeviceContext, Dim
import kernels


@always_inline
def launch_hadamard(
    rt: DeviceContext,
    work: Int32,
    x: DeviceBuffer[.uint8],
    signs: DeviceBuffer[.uint8],
    y: DeviceBuffer[.uint8],
    q: DeviceBuffer[.uint8],
    width: Int32,
    stride: Int32,
    inverse: Int32,
) raises -> Int64:
    rt.enqueue_function[kernels.hadamard](
        x.create_sub_buffer[.float32](0, len(x) // size_of[Float32]()),
        signs.create_sub_buffer[.float32](0, len(signs) // size_of[Float32]()),
        y.create_sub_buffer[.float32](0, len(y) // size_of[Float32]()),
        q.create_sub_buffer[.uint32](0, len(q) // size_of[UInt32]()),
        width,
        stride,
        inverse,
        grid_dim=Dim((Int(work) + 63) // 64, 1, 1),
        block_dim=64,
    )
    return 0


@always_inline
def launch_packed_linear(
    rt: DeviceContext,
    work: Int32,
    x: DeviceBuffer[.uint8],
    w: DeviceBuffer[.uint8],
    params: DeviceBuffer[.uint8],
    partials: DeviceBuffer[.uint8],
    width: Int32,
    reduction: Int32,
) raises -> Int64:
    rt.enqueue_function[kernels.packed_linear](
        x.create_sub_buffer[.uint32](0, len(x) // size_of[UInt32]()),
        w.create_sub_buffer[.uint32](0, len(w) // size_of[UInt32]()),
        params.create_sub_buffer[.float16](
            0, len(params) // size_of[Float16]()
        ),
        partials.create_sub_buffer[.float32](
            0, len(partials) // size_of[Float32]()
        ),
        width,
        reduction,
        grid_dim=Dim(
            (Int(work) + 63) // 64, (Int(reduction) + 1023) // 1024, 1
        ),
        block_dim=64,
    )
    return 0


@always_inline
def launch_embedding(
    rt: DeviceContext,
    work: Int32,
    w: DeviceBuffer[.uint8],
    params: DeviceBuffer[.uint8],
    y: DeviceBuffer[.uint8],
    token: Int32,
) raises -> Int64:
    rt.enqueue_function[kernels.embedding](
        w.create_sub_buffer[.uint32](0, len(w) // size_of[UInt32]()),
        params.create_sub_buffer[.float16](
            0, len(params) // size_of[Float16]()
        ),
        y.create_sub_buffer[.float32](0, len(y) // size_of[Float32]()),
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
    partials: DeviceBuffer[.uint8],
) raises -> Int64:
    rt.enqueue_function[kernels.rms_norm](
        x.create_sub_buffer[.float32](0, len(x) // size_of[Float32]()),
        w.create_sub_buffer[.float32](0, len(w) // size_of[Float32]()),
        y.create_sub_buffer[.float32](0, len(y) // size_of[Float32]()),
        partials.create_sub_buffer[.float32](
            0, len(partials) // size_of[Float32]()
        ),
        grid_dim=Dim((Int(work) + 63) // 64, 1, 1),
        block_dim=64,
    )
    return 0


@always_inline
def launch_gate_partials(
    rt: DeviceContext,
    work: Int32,
    x: DeviceBuffer[.uint8],
    a: DeviceBuffer[.uint8],
    b: DeviceBuffer[.uint8],
    partials: DeviceBuffer[.uint8],
) raises -> Int64:
    rt.enqueue_function[kernels.gate_partials](
        x.create_sub_buffer[.float32](0, len(x) // size_of[Float32]()),
        a.create_sub_buffer[.float32](0, len(a) // size_of[Float32]()),
        b.create_sub_buffer[.float32](0, len(b) // size_of[Float32]()),
        partials.create_sub_buffer[.float32](
            0, len(partials) // size_of[Float32]()
        ),
        grid_dim=Dim((Int(work) + 63) // 64, 1, 1),
        block_dim=64,
    )
    return 0


@always_inline
def launch_convolve(
    rt: DeviceContext,
    work: Int32,
    x: DeviceBuffer[.uint8],
    history: DeviceBuffer[.uint8],
    w: DeviceBuffer[.uint8],
    y: DeviceBuffer[.uint8],
) raises -> Int64:
    rt.enqueue_function[kernels.convolve](
        x.create_sub_buffer[.float32](0, len(x) // size_of[Float32]()),
        history.create_sub_buffer[.float32](
            0, len(history) // size_of[Float32]()
        ),
        w.create_sub_buffer[.float32](0, len(w) // size_of[Float32]()),
        y.create_sub_buffer[.float32](0, len(y) // size_of[Float32]()),
        grid_dim=Dim((Int(work) + 63) // 64, 1, 1),
        block_dim=64,
    )
    return 0


@always_inline
def launch_normalize_qk(
    rt: DeviceContext,
    work: Int32,
    x: DeviceBuffer[.uint8],
    y: DeviceBuffer[.uint8],
) raises -> Int64:
    rt.enqueue_function[kernels.normalize_qk](
        x.create_sub_buffer[.float32](0, len(x) // size_of[Float32]()),
        y.create_sub_buffer[.float32](0, len(y) // size_of[Float32]()),
        grid_dim=Dim((Int(work) + 63) // 64, 1, 1),
        block_dim=64,
    )
    return 0


@always_inline
def launch_delta_gates(
    rt: DeviceContext,
    work: Int32,
    partials: DeviceBuffer[.uint8],
    constants: DeviceBuffer[.uint8],
    y: DeviceBuffer[.uint8],
) raises -> Int64:
    rt.enqueue_function[kernels.delta_gates](
        partials.create_sub_buffer[.float32](
            0, len(partials) // size_of[Float32]()
        ),
        constants.create_sub_buffer[.float32](
            0, len(constants) // size_of[Float32]()
        ),
        y.create_sub_buffer[.float32](0, len(y) // size_of[Float32]()),
        grid_dim=Dim((Int(work) + 63) // 64, 1, 1),
        block_dim=64,
    )
    return 0


@always_inline
def launch_delta_update(
    rt: DeviceContext,
    work: Int32,
    qkv: DeviceBuffer[.uint8],
    gates: DeviceBuffer[.uint8],
    state: DeviceBuffer[.uint8],
    y: DeviceBuffer[.uint8],
) raises -> Int64:
    rt.enqueue_function[kernels.delta_update](
        qkv.create_sub_buffer[.float32](0, len(qkv) // size_of[Float32]()),
        gates.create_sub_buffer[.float32](0, len(gates) // size_of[Float32]()),
        state.create_sub_buffer[.float32](0, len(state) // size_of[Float32]()),
        y.create_sub_buffer[.float32](0, len(y) // size_of[Float32]()),
        grid_dim=Dim((Int(work) * 8 + 63) // 64, 1, 1),
        block_dim=64,
    )
    return 0


@always_inline
def launch_gated_norm(
    rt: DeviceContext,
    work: Int32,
    x: DeviceBuffer[.uint8],
    z: DeviceBuffer[.uint8],
    w: DeviceBuffer[.uint8],
    y: DeviceBuffer[.uint8],
) raises -> Int64:
    rt.enqueue_function[kernels.gated_norm](
        x.create_sub_buffer[.float32](0, len(x) // size_of[Float32]()),
        z.create_sub_buffer[.float32](0, len(z) // size_of[Float32]()),
        w.create_sub_buffer[.float32](0, len(w) // size_of[Float32]()),
        y.create_sub_buffer[.float32](0, len(y) // size_of[Float32]()),
        grid_dim=Dim((Int(work) + 63) // 64, 1, 1),
        block_dim=64,
    )
    return 0


@always_inline
def launch_head_norm(
    rt: DeviceContext,
    work: Int32,
    x: DeviceBuffer[.uint8],
    w: DeviceBuffer[.uint8],
    y: DeviceBuffer[.uint8],
    mode: Int32,
) raises -> Int64:
    rt.enqueue_function[kernels.head_norm](
        x.create_sub_buffer[.float32](0, len(x) // size_of[Float32]()),
        w.create_sub_buffer[.float32](0, len(w) // size_of[Float32]()),
        y.create_sub_buffer[.float32](0, len(y) // size_of[Float32]()),
        mode,
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
    position: Int32,
) raises -> Int64:
    rt.enqueue_function[kernels.cache_kv](
        k.create_sub_buffer[.float32](0, len(k) // size_of[Float32]()),
        v.create_sub_buffer[.float32](0, len(v) // size_of[Float32]()),
        kc.create_sub_buffer[.float32](0, len(kc) // size_of[Float32]()),
        vc.create_sub_buffer[.float32](0, len(vc) // size_of[Float32]()),
        position,
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
    y: DeviceBuffer[.uint8],
    position: Int32,
    capacity: Int32,
) raises -> Int64:
    rt.enqueue_function[kernels.attention_scores](
        q.create_sub_buffer[.float32](0, len(q) // size_of[Float32]()),
        kc.create_sub_buffer[.float32](0, len(kc) // size_of[Float32]()),
        y.create_sub_buffer[.float32](0, len(y) // size_of[Float32]()),
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
    qg: DeviceBuffer[.uint8],
    y: DeviceBuffer[.uint8],
    position: Int32,
    capacity: Int32,
) raises -> Int64:
    rt.enqueue_function[kernels.attention_values](
        scores.create_sub_buffer[.float32](
            0, len(scores) // size_of[Float32]()
        ),
        vc.create_sub_buffer[.float32](0, len(vc) // size_of[Float32]()),
        qg.create_sub_buffer[.float32](0, len(qg) // size_of[Float32]()),
        y.create_sub_buffer[.float32](0, len(y) // size_of[Float32]()),
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


@always_inline
def launch_copy(
    rt: DeviceContext,
    work: Int32,
    x: DeviceBuffer[.uint8],
    y: DeviceBuffer[.uint8],
) raises -> Int64:
    rt.enqueue_function[kernels.copy](
        x.create_sub_buffer[.float32](0, len(x) // size_of[Float32]()),
        y.create_sub_buffer[.float32](0, len(y) // size_of[Float32]()),
        grid_dim=Dim((Int(work) + 63) // 64, 1, 1),
        block_dim=64,
    )
    return 0


@always_inline
def launch_rms_partials(
    rt: DeviceContext,
    work: Int32,
    x: DeviceBuffer[.uint8],
    partials: DeviceBuffer[.uint8],
) raises -> Int64:
    rt.enqueue_function[kernels.rms_partials](
        x.create_sub_buffer[.float32](0, len(x) // size_of[Float32]()),
        partials.create_sub_buffer[.float32](
            0, len(partials) // size_of[Float32]()
        ),
        grid_dim=Dim((Int(work) + 63) // 64, 1, 1),
        block_dim=64,
    )
    return 0


@always_inline
def launch_packed_head(
    rt: DeviceContext,
    work: Int32,
    x: DeviceBuffer[.uint8],
    w: DeviceBuffer[.uint8],
    params: DeviceBuffer[.uint8],
    y: DeviceBuffer[.uint8],
    candidates: DeviceBuffer[.uint8],
    width: Int32,
    reduction: Int32,
    select: Bool,
) raises -> Int64:
    rt.enqueue_function[kernels.packed_head](
        x.create_sub_buffer[.uint32](0, len(x) // size_of[UInt32]()),
        w.create_sub_buffer[.uint32](0, len(w) // size_of[UInt32]()),
        params.create_sub_buffer[.float16](
            0, len(params) // size_of[Float16]()
        ),
        y.create_sub_buffer[.float32](0, len(y) // size_of[Float32]()),
        candidates.create_sub_buffer[.float32](
            0, len(candidates) // size_of[Float32]()
        ),
        width,
        reduction,
        Int32(select),
        grid_dim=Dim((Int(work) + 1023) // 1024, 1, 1),
        block_dim=128,
    )
    return 0


@always_inline
def launch_packed_reduce(
    rt: DeviceContext,
    work: Int32,
    partials: DeviceBuffer[.uint8],
    y: DeviceBuffer[.uint8],
    width: Int32,
    groups: Int32,
) raises -> Int64:
    rt.enqueue_function[kernels.packed_reduce](
        partials.create_sub_buffer[.float32](
            0, len(partials) // size_of[Float32]()
        ),
        y.create_sub_buffer[.float32](0, len(y) // size_of[Float32]()),
        width,
        groups,
        grid_dim=Dim((Int(work) + 63) // 64, 1, 1),
        block_dim=64,
    )
    return 0


@always_inline
def rotate(
    rt: DeviceContext,
    x: DeviceBuffer[.uint8],
    signs: DeviceBuffer[.uint8],
    b: List[DeviceBuffer[.uint8]],
    width: Int32,
    inverse: Int32 = 0,
) raises -> Int64:
    return launch_hadamard(
        rt, width // 8, x, signs, b[2], b[3], width, 0, inverse
    )


@always_inline
def project(
    rt: DeviceContext,
    w: List[DeviceBuffer[.uint8]],
    b: List[DeviceBuffer[.uint8]],
    slot: Int,
    x: DeviceBuffer[.uint8],
    y: DeviceBuffer[.uint8],
    width: Int32,
    reduction: Int32 = 2048,
    reuse_transform: Bool = False,
    head_gpu_columns: Int32 = -1,
    head_input: OptionalPointer[UInt8, MutUntrackedOrigin] = None,
    gpu_greedy: Bool = False,
) raises -> Int64:
    var status: Int64 = 0
    if not reuse_transform:
        status = rotate(rt, x, w[slot + 2], b, reduction)
        if status != 0:
            return status
    if width == 248320:
        if case .Some(ptr) = head_input:
            # Complete the small activation read before submitting the GPU head.
            # The host can then run Hexagon concurrently with that GPU work.
            rt.enqueue_copy(
                ptr, b[3].create_sub_buffer[.uint8](0, Int(reduction) * 2)
            )
            var ready = rt.create_event()
            rt.stream().record_event(ready)
            if head_gpu_columns != 0:
                status = launch_packed_head(
                    rt,
                    width if head_gpu_columns < 0 else head_gpu_columns,
                    b[3],
                    w[slot],
                    w[slot + 1],
                    y,
                    b[68],
                    width,
                    reduction,
                    gpu_greedy,
                )
            ready.synchronize()
            return status
        if head_gpu_columns == 0:
            return 0
        status = launch_packed_head(
            rt,
            width if head_gpu_columns < 0 else head_gpu_columns,
            b[3],
            w[slot],
            w[slot + 1],
            y,
            b[68],
            width,
            reduction,
            gpu_greedy,
        )
        return status
    status = launch_packed_linear(
        rt, width, b[3], w[slot], w[slot + 1], b[67], width, reduction
    )
    if status != 0:
        return status
    return launch_packed_reduce(
        rt, width, b[67], y, width, ((reduction + 1023) // 1024)
    )


@always_inline
def norm(
    rt: DeviceContext,
    x: DeviceBuffer[.uint8],
    weight: DeviceBuffer[.uint8],
    y: DeviceBuffer[.uint8],
    scratch: DeviceBuffer[.uint8],
) raises -> Int64:
    var status = launch_rms_partials(rt, 32, x, scratch)
    if status != 0:
        return status
    return launch_rms_norm(rt, 2048, x, weight, y, scratch)


def forward(
    rt: DeviceContext,
    w: List[DeviceBuffer[.uint8]],
    b: List[DeviceBuffer[.uint8]],
    token: Int32,
    position: Int32,
    capacity: Int32,
    head_gpu_columns: Int32 = -1,
    head_input: OptionalPointer[UInt8, MutUntrackedOrigin] = None,
    gpu_greedy: Bool = True,
) raises -> Int64:
    """Enqueue text-only Bonsai 2 2B with persistent convolution/delta/KV state.
    """
    if token < 0 or token >= 248320 or position < 0 or position >= capacity:
        return -1
    var status: Int64 = 0
    status = launch_embedding(rt, 2048, w[0], w[1], b[1], token)
    if status != 0:
        return status
    status = rotate(rt, b[1], w[2], b, 2048, 1)
    if status != 0:
        return status
    status = launch_copy(rt, 2048, b[2], b[0])
    if status != 0:
        return status
    for layer in range(24):
        var off = 4 + layer * 25
        var history = b[19 + layer * 2]
        var state = b[20 + layer * 2]
        status = norm(
            rt,
            b[0],
            w[off],
            b[1],
            b[3],
        )
        if status != 0:
            return status
        if (layer + 1) % 4 != 0:
            status = project(
                rt,
                w,
                b,
                off + 11,
                b[1],
                b[4],
                6144,
                2048,
            )
            if status != 0:
                return status
            status = project(
                rt,
                w,
                b,
                off + 14,
                b[1],
                b[5],
                2048,
                2048,
                reuse_transform=True,
            )
            if status != 0:
                return status
            # Reuse transform scratch: no live Hadamard input is needed here.
            status = launch_gate_partials(
                rt, 1024, b[1], w[off + 20], w[off + 21], b[2]
            )
            if status != 0:
                return status
            status = launch_convolve(rt, 6144, b[4], history, w[off + 23], b[9])
            if status != 0:
                return status
            status = launch_normalize_qk(rt, 6144, b[9], b[4])
            if status != 0:
                return status
            status = launch_delta_gates(rt, 16, b[2], w[off + 22], b[8])
            if status != 0:
                return status
            status = launch_delta_update(rt, 2048, b[4], b[8], state, b[10])
            if status != 0:
                return status
            status = launch_gated_norm(rt, 2048, b[10], b[5], w[off + 24], b[1])
            if status != 0:
                return status
            status = project(
                rt,
                w,
                b,
                off + 17,
                b[1],
                b[1],
                2048,
                2048,
            )
            if status != 0:
                return status
        else:
            status = project(
                rt,
                w,
                b,
                off + 11,
                b[1],
                b[5],
                4096,
                2048,
            )
            if status != 0:
                return status
            status = project(
                rt,
                w,
                b,
                off + 14,
                b[1],
                b[15],
                512,
                2048,
                reuse_transform=True,
            )
            if status != 0:
                return status
            status = project(
                rt,
                w,
                b,
                off + 17,
                b[1],
                b[16],
                512,
                2048,
                reuse_transform=True,
            )
            if status != 0:
                return status
            status = launch_head_norm(rt, 2048, b[5], w[off + 23], b[14], 0)
            if status != 0:
                return status
            status = launch_head_norm(rt, 512, b[15], w[off + 24], b[9], 1)
            if status != 0:
                return status
            status = launch_rope(rt, 256, b[14], b[18], 2048, position)
            if status != 0:
                return status
            status = launch_rope(rt, 64, b[9], b[18], 512, position)
            if status != 0:
                return status
            status = launch_cache_kv(
                rt, 512, b[9], b[16], history, state, position
            )
            if status != 0:
                return status
            status = launch_attention_scores(
                rt,
                8 * (position + 1),
                b[14],
                history,
                b[13],
                position,
                capacity,
            )
            if status != 0:
                return status
            status = launch_attention_values(
                rt, 2048, b[13], state, b[5], b[10], position, capacity
            )
            if status != 0:
                return status
            status = project(
                rt,
                w,
                b,
                off + 20,
                b[10],
                b[1],
                2048,
                2048,
            )
            if status != 0:
                return status
        status = launch_residual(rt, 2048, b[0], b[1])
        if status != 0:
            return status
        status = norm(
            rt,
            b[0],
            w[off + 1],
            b[1],
            b[3],
        )
        if status != 0:
            return status
        status = project(
            rt,
            w,
            b,
            off + 2,
            b[1],
            b[11],
            6144,
            2048,
        )
        if status != 0:
            return status
        status = project(
            rt,
            w,
            b,
            off + 5,
            b[1],
            b[12],
            6144,
            2048,
            reuse_transform=True,
        )
        if status != 0:
            return status
        status = launch_swiglu(rt, 6144, b[11], b[12])
        if status != 0:
            return status
        status = project(
            rt,
            w,
            b,
            off + 8,
            b[11],
            b[1],
            2048,
            6144,
        )
        if status != 0:
            return status
        status = launch_residual(rt, 2048, b[0], b[1])
        if status != 0:
            return status
    status = norm(
        rt,
        b[0],
        w[3],
        b[1],
        b[3],
    )
    if status != 0:
        return status
    status = project(
        rt,
        w,
        b,
        0,
        b[1],
        b[17],
        248320,
        2048,
        head_gpu_columns=head_gpu_columns,
        head_input=head_input,
        gpu_greedy=gpu_greedy,
    )
    if status != 0:
        return status
    return 0
