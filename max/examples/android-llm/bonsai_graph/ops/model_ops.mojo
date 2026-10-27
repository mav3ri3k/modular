# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the Apache License v2.0 with LLVM Exceptions: https://llvm.org/LICENSE.txt
"""MAX custom-op adapters for the shared Bonsai kernels.

Graph storage stays opaque on the host. Captured tiles resolve their pointers
only inside the device closure. Mutable recurrent state uses BufferType inputs
and MAX's inplace_custom effect tracking, never hidden global state.
"""
import bonsai_kernels as kernels
from extensibility import InputTensor, OutputTensor, register
from extensibility import _MutableInputTensor as MutableInputTensor
from max.gpu.host import DeviceContext
from std.utils.index import IndexList


@register("bonsai_embedding")
struct Embedding:
    @staticmethod
    def execute(
        y: OutputTensor[dtype=.float32, rank=1, ...],
        w: InputTensor[dtype=.uint32, rank=1, ...],
        params: InputTensor[dtype=.uint16, rank=1, ...],
        token_tensor: InputTensor[dtype=.int32, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        var y_tile = y.to_tile_tensor()
        var w_tile = w.to_tile_tensor()
        var params_tile = params.to_tile_tensor()
        var token = token_tensor.load[1](IndexList[1](0))

        def kernel() {var y_tile, var w_tile, var params_tile, var token}:
            kernels.embedding(
                w_tile.ptr.as_unsafe_any_origin(),
                params_tile.ptr.as_unsafe_any_origin().unsafe_bitcast[
                    Float16
                ](),
                y_tile.ptr.as_unsafe_any_origin(),
                token,
            )

        ctx.enqueue_function(kernel, grid_dim=32, block_dim=64)


@register("bonsai_hadamard")
struct Hadamard[width: Int, inverse: Int]:
    @staticmethod
    def execute(
        y: OutputTensor[dtype=.float32, rank=1, ...],
        q: OutputTensor[dtype=.uint32, rank=1, ...],
        x: InputTensor[dtype=.float32, rank=1, ...],
        signs: InputTensor[dtype=.float32, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        var y_tile = y.to_tile_tensor()
        var q_tile = q.to_tile_tensor()
        var x_tile = x.to_tile_tensor()
        var signs_tile = signs.to_tile_tensor()

        def kernel() {var y_tile, var q_tile, var x_tile, var signs_tile}:
            kernels.hadamard(
                x_tile.ptr.as_unsafe_any_origin(),
                signs_tile.ptr.as_unsafe_any_origin(),
                y_tile.ptr.as_unsafe_any_origin(),
                q_tile.ptr.as_unsafe_any_origin(),
                Int32(Self.width),
                Int32(0),
                Int32(Self.inverse),
            )

        ctx.enqueue_function(kernel, grid_dim=Self.width // 512, block_dim=64)


@register("bonsai_packed_linear")
struct PackedLinear[width: Int, reduction: Int]:
    @staticmethod
    def execute(
        partials: OutputTensor[dtype=.float32, rank=1, ...],
        x: InputTensor[dtype=.uint32, rank=1, ...],
        w: InputTensor[dtype=.uint32, rank=1, ...],
        params: InputTensor[dtype=.uint16, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        var partials_tile = partials.to_tile_tensor()
        var x_tile = x.to_tile_tensor()
        var w_tile = w.to_tile_tensor()
        var params_tile = params.to_tile_tensor()

        def kernel() {
            var partials_tile, var x_tile, var w_tile, var params_tile
        }:
            kernels.packed_linear(
                x_tile.ptr.as_unsafe_any_origin(),
                w_tile.ptr.as_unsafe_any_origin(),
                params_tile.ptr.as_unsafe_any_origin().unsafe_bitcast[
                    Float16
                ](),
                partials_tile.ptr.as_unsafe_any_origin(),
                Int32(Self.width),
                Int32(Self.reduction),
            )

        ctx.enqueue_function(
            kernel,
            grid_dim=(
                (Self.width + 63) // 64,
                (Self.reduction + 1023) // 1024,
                1,
            ),
            block_dim=64,
        )


@register("bonsai_packed_head")
struct PackedHead[width: Int, reduction: Int]:
    @staticmethod
    def execute(
        y: OutputTensor[dtype=.float32, rank=1, ...],
        x: InputTensor[dtype=.uint32, rank=1, ...],
        w: InputTensor[dtype=.uint32, rank=1, ...],
        params: InputTensor[dtype=.uint16, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        var y_tile = y.to_tile_tensor()
        var x_tile = x.to_tile_tensor()
        var w_tile = w.to_tile_tensor()
        var params_tile = params.to_tile_tensor()

        def kernel() {var y_tile, var x_tile, var w_tile, var params_tile}:
            kernels.packed_head(
                x_tile.ptr.as_unsafe_any_origin(),
                w_tile.ptr.as_unsafe_any_origin(),
                params_tile.ptr.as_unsafe_any_origin().unsafe_bitcast[
                    Float16
                ](),
                y_tile.ptr.as_unsafe_any_origin(),
                y_tile.ptr.as_unsafe_any_origin(),
                Int32(Self.width),
                Int32(Self.reduction),
                Int32(0),
            )

        ctx.enqueue_function(
            kernel, grid_dim=(Self.width + 1023) // 1024, block_dim=128
        )


@register("bonsai_gate_partials")
struct GatePartials:
    @staticmethod
    def execute(
        partials: OutputTensor[dtype=.float32, rank=1, ...],
        x: InputTensor[dtype=.float32, rank=1, ...],
        a: InputTensor[dtype=.float32, rank=1, ...],
        b: InputTensor[dtype=.float32, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        var partials_tile = partials.to_tile_tensor()
        var x_tile = x.to_tile_tensor()
        var a_tile = a.to_tile_tensor()
        var b_tile = b.to_tile_tensor()

        def kernel() {var partials_tile, var x_tile, var a_tile, var b_tile}:
            kernels.gate_partials(
                x_tile.ptr.as_unsafe_any_origin(),
                a_tile.ptr.as_unsafe_any_origin(),
                b_tile.ptr.as_unsafe_any_origin(),
                partials_tile.ptr.as_unsafe_any_origin(),
            )

        ctx.enqueue_function(kernel, grid_dim=16, block_dim=64)


@register("bonsai_convolve")
struct Convolve:
    @staticmethod
    def execute(
        y: OutputTensor[dtype=.float32, rank=1, ...],
        x: InputTensor[dtype=.float32, rank=1, ...],
        history: MutableInputTensor[dtype=.float32, rank=1, ...],
        w: InputTensor[dtype=.float32, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        var y_tile = y.to_tile_tensor()
        var x_tile = x.to_tile_tensor()
        var history_tile = history.to_tile_tensor()
        var w_tile = w.to_tile_tensor()

        def kernel() {var y_tile, var x_tile, var history_tile, var w_tile}:
            kernels.convolve(
                x_tile.ptr.as_unsafe_any_origin(),
                history_tile.ptr.as_unsafe_any_origin(),
                w_tile.ptr.as_unsafe_any_origin(),
                y_tile.ptr.as_unsafe_any_origin(),
            )

        ctx.enqueue_function(kernel, grid_dim=96, block_dim=64)


@register("bonsai_normalize_qk")
struct NormalizeQk:
    @staticmethod
    def execute(
        y: OutputTensor[dtype=.float32, rank=1, ...],
        x: InputTensor[dtype=.float32, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        var y_tile = y.to_tile_tensor()
        var x_tile = x.to_tile_tensor()

        def kernel() {var y_tile, var x_tile}:
            kernels.normalize_qk(
                x_tile.ptr.as_unsafe_any_origin(),
                y_tile.ptr.as_unsafe_any_origin(),
            )

        ctx.enqueue_function(kernel, grid_dim=96, block_dim=64)


@register("bonsai_delta_gates")
struct DeltaGates:
    @staticmethod
    def execute(
        y: OutputTensor[dtype=.float32, rank=1, ...],
        partials: InputTensor[dtype=.float32, rank=1, ...],
        constants: InputTensor[dtype=.float32, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        var y_tile = y.to_tile_tensor()
        var partials_tile = partials.to_tile_tensor()
        var constants_tile = constants.to_tile_tensor()

        def kernel() {var y_tile, var partials_tile, var constants_tile}:
            kernels.delta_gates(
                partials_tile.ptr.as_unsafe_any_origin(),
                constants_tile.ptr.as_unsafe_any_origin(),
                y_tile.ptr.as_unsafe_any_origin(),
            )

        ctx.enqueue_function(kernel, grid_dim=1, block_dim=64)


@register("bonsai_delta_update")
struct DeltaUpdate:
    @staticmethod
    def execute(
        y: OutputTensor[dtype=.float32, rank=1, ...],
        qkv: InputTensor[dtype=.float32, rank=1, ...],
        gates: InputTensor[dtype=.float32, rank=1, ...],
        state: MutableInputTensor[dtype=.float32, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        var y_tile = y.to_tile_tensor()
        var qkv_tile = qkv.to_tile_tensor()
        var gates_tile = gates.to_tile_tensor()
        var state_tile = state.to_tile_tensor()

        def kernel() {var y_tile, var qkv_tile, var gates_tile, var state_tile}:
            kernels.delta_update(
                qkv_tile.ptr.as_unsafe_any_origin(),
                gates_tile.ptr.as_unsafe_any_origin(),
                state_tile.ptr.as_unsafe_any_origin(),
                y_tile.ptr.as_unsafe_any_origin(),
            )

        ctx.enqueue_function(kernel, grid_dim=256, block_dim=64)


@register("bonsai_gated_norm")
struct GatedNorm:
    @staticmethod
    def execute(
        y: OutputTensor[dtype=.float32, rank=1, ...],
        x: InputTensor[dtype=.float32, rank=1, ...],
        z: InputTensor[dtype=.float32, rank=1, ...],
        w: InputTensor[dtype=.float32, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        var y_tile = y.to_tile_tensor()
        var x_tile = x.to_tile_tensor()
        var z_tile = z.to_tile_tensor()
        var w_tile = w.to_tile_tensor()

        def kernel() {var y_tile, var x_tile, var z_tile, var w_tile}:
            kernels.gated_norm(
                x_tile.ptr.as_unsafe_any_origin(),
                z_tile.ptr.as_unsafe_any_origin(),
                w_tile.ptr.as_unsafe_any_origin(),
                y_tile.ptr.as_unsafe_any_origin(),
            )

        ctx.enqueue_function(kernel, grid_dim=32, block_dim=64)


@register("bonsai_cache_kv")
struct CacheKv:
    @staticmethod
    def execute(
        k: InputTensor[dtype=.float32, rank=1, ...],
        v: InputTensor[dtype=.float32, rank=1, ...],
        kc: MutableInputTensor[dtype=.float32, rank=1, ...],
        vc: MutableInputTensor[dtype=.float32, rank=1, ...],
        position_tensor: InputTensor[dtype=.int32, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        var k_tile = k.to_tile_tensor()
        var v_tile = v.to_tile_tensor()
        var kc_tile = kc.to_tile_tensor()
        var vc_tile = vc.to_tile_tensor()
        var position = position_tensor.load[1](IndexList[1](0))

        def kernel() {
            var k_tile, var v_tile, var kc_tile, var vc_tile, var position
        }:
            kernels.cache_kv(
                k_tile.ptr.as_unsafe_any_origin(),
                v_tile.ptr.as_unsafe_any_origin(),
                kc_tile.ptr.as_unsafe_any_origin(),
                vc_tile.ptr.as_unsafe_any_origin(),
                position,
            )

        ctx.enqueue_function(kernel, grid_dim=8, block_dim=64)


@register("bonsai_attention_scores")
struct AttentionScores[capacity: Int]:
    @staticmethod
    def execute(
        y: OutputTensor[dtype=.float32, rank=1, ...],
        q: InputTensor[dtype=.float32, rank=1, ...],
        kc: InputTensor[dtype=.float32, rank=1, ...],
        position_tensor: InputTensor[dtype=.int32, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        var y_tile = y.to_tile_tensor()
        var q_tile = q.to_tile_tensor()
        var kc_tile = kc.to_tile_tensor()
        var position = position_tensor.load[1](IndexList[1](0))

        def kernel() {var y_tile, var q_tile, var kc_tile, var position}:
            kernels.attention_scores(
                q_tile.ptr.as_unsafe_any_origin(),
                kc_tile.ptr.as_unsafe_any_origin(),
                y_tile.ptr.as_unsafe_any_origin(),
                position,
                Int32(Self.capacity),
            )

        ctx.enqueue_function(
            kernel, grid_dim=(8 * (Int(position) + 1) + 63) // 64, block_dim=64
        )


@register("bonsai_attention_values")
struct AttentionValues[capacity: Int]:
    @staticmethod
    def execute(
        y: OutputTensor[dtype=.float32, rank=1, ...],
        scores: InputTensor[dtype=.float32, rank=1, ...],
        vc: InputTensor[dtype=.float32, rank=1, ...],
        qg: InputTensor[dtype=.float32, rank=1, ...],
        position_tensor: InputTensor[dtype=.int32, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        var y_tile = y.to_tile_tensor()
        var scores_tile = scores.to_tile_tensor()
        var vc_tile = vc.to_tile_tensor()
        var qg_tile = qg.to_tile_tensor()
        var position = position_tensor.load[1](IndexList[1](0))

        def kernel() {
            var y_tile, var scores_tile, var vc_tile, var qg_tile, var position
        }:
            kernels.attention_values(
                scores_tile.ptr.as_unsafe_any_origin(),
                vc_tile.ptr.as_unsafe_any_origin(),
                qg_tile.ptr.as_unsafe_any_origin(),
                y_tile.ptr.as_unsafe_any_origin(),
                position,
                Int32(Self.capacity),
            )

        ctx.enqueue_function(kernel, grid_dim=32, block_dim=64)


@register("bonsai_head_rope")
struct HeadRope[width: Int, mode: Int]:
    @staticmethod
    def execute(
        y: OutputTensor[dtype=.float32, rank=1, ...],
        x: InputTensor[dtype=.float32, rank=1, ...],
        w: InputTensor[dtype=.float32, rank=1, ...],
        table: InputTensor[dtype=.float32, rank=1, ...],
        position_tensor: InputTensor[dtype=.int32, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        var y_tile = y.to_tile_tensor()
        var x_tile = x.to_tile_tensor()
        var w_tile = w.to_tile_tensor()
        var table_tile = table.to_tile_tensor()
        var position = position_tensor.load[1](IndexList[1](0))

        def normalize() {var x_tile, var w_tile, var y_tile}:
            kernels.head_norm(
                x_tile.ptr.as_unsafe_any_origin(),
                w_tile.ptr.as_unsafe_any_origin(),
                y_tile.ptr.as_unsafe_any_origin(),
                Int32(Self.mode),
            )

        ctx.enqueue_function(normalize, grid_dim=Self.width // 64, block_dim=64)

        def rotate() {var y_tile, var table_tile, var position}:
            kernels.rope(
                y_tile.ptr.as_unsafe_any_origin(),
                table_tile.ptr.as_unsafe_any_origin(),
                Int32(Self.width),
                position,
            )

        ctx.enqueue_function(
            rotate, grid_dim=(Self.width // 8 + 63) // 64, block_dim=64
        )
