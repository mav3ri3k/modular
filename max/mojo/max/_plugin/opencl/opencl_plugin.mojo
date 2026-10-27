# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the Apache License v2.0 with LLVM Exceptions: https://llvm.org/LICENSE.txt
"""Portable OpenCL synchronization hooks."""

from std.collections import OptionalReg
from std.collections.string.string_span import _get_kgen_string
from std.ffi import external_call
from std.sys import CompilationTarget
from max.gpu.sync.sync import (
    _BarrierPluginHookFnType,
    _SyncwarpPluginHookFnType,
)
from .._trait import MaxPlugin, MaxPluginHooks
from .._shuffle import shuffle_words


@inline(.always)
def _barrier():
    external_call["__mojo_opencl_barrier", NoneType]()


@inline(.always)
def _syncwarp(mask: Int):
    # Like AMD/Apple, this backend synchronizes all active subgroup lanes.
    # Fence shared accesses as required by row-wise monoid reduction trees.
    external_call["__mojo_opencl_subgroup_barrier_w64", NoneType]()


@inline(.always)
def _shuffle_word(mask: UInt, word: UInt32, lane: UInt32) -> UInt32:
    # As on Metal, OpenCL shuffles require converged subgroup participation;
    # the CUDA-specific lane mask does not change that requirement.
    return external_call["__mojo_opencl_shuffle_idx_u32_w64", UInt32](
        word, lane
    )


struct OpenCLMaxPlugin(MaxPlugin):
    """Supplies workgroup synchronization without assuming a subgroup width."""

    comptime name: __mlir_type.`!kgen.string` = _get_kgen_string["opencl"]()
    comptime HookFor[target: CompilationTarget] = OpenCLMaxPluginHooks[target]


struct OpenCLMaxPluginHooks[target: CompilationTarget](MaxPluginHooks):
    comptime warp_size = 64 if Self.target._is_arch["adreno-840"]() else 1
    comptime max_threads_per_block_metadata: StaticString = (
        "pop.opencl.max_work_group_size"
    )
    comptime barrier_fn: OptionalReg[_BarrierPluginHookFnType] = _barrier
    comptime syncwarp_fn: OptionalReg[_SyncwarpPluginHookFnType] = (
        _syncwarp if Self.target._is_arch["adreno-840"]() else None
    )

    @staticmethod
    @inline(.always)
    def shuffle_idx_fn[
        dtype: DType, simd_width: SIMDLength, //
    ](mask: UInt, val: SIMD[dtype, simd_width], offset: UInt32) -> SIMD[
        dtype, simd_width
    ]:
        comptime assert Self.target._is_arch[
            "adreno-840"
        ](), "OpenCL shuffles require a target with a validated subgroup width"
        return shuffle_words[_shuffle_word](mask, val, offset)
