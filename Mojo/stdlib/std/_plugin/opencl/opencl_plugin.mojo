# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the Apache License v2.0 with LLVM Exceptions: https://llvm.org/LICENSE.txt
from std._gpu.primitives.id import _GPUIdPluginHookFnType
from std._plugin._trait import PluginHooks, StdPlugin, TargetDeviceKind
from std.collections import OptionalReg
from std.collections.string.string_span import (
    _get_kgen_string,
    get_static_string,
)
from std.ffi import external_call
from std.sys.info import CompilationTarget
from std.sys import llvm_intrinsic, bit_width_of
from std.math.math import _UnaryMathOp, _UnaryMathPluginHookFnType


def _opencl_gpu_id_fn[
    name: StaticString, dim: StaticString, target: CompilationTarget
]() -> OptionalReg[_GPUIdPluginHookFnType]:
    comptime if name == "lane_id" and target._is_arch["adreno-840"]():
        return comptime (OptionalReg[_GPUIdPluginHookFnType](_opencl_lane_id))
    comptime if (
        name == "thread_idx"
        or name == "block_idx"
        or name == "block_dim"
        or name == "grid_dim"
    ):
        return comptime (
            OptionalReg[_GPUIdPluginHookFnType](_opencl_gpu_id[name, dim])
        )
    else:
        return None


@inline(.always)
def _opencl_lane_id() -> Int:
    return external_call["__mojo_opencl_subgroup_local_id", Int]()


@inline(.always)
def _opencl_gpu_id[name: StaticString, dim: StaticString]() -> Int:
    comptime builtin = (
        "get_local_id" if name
        == "thread_idx" else "get_group_id" if name
        == "block_idx" else "get_local_size" if name
        == "block_dim" else "get_num_groups"
    )
    return external_call[
        get_static_string["__mojo_opencl_", builtin, "_", dim](), Int
    ]()


@inline(.always)
def _opencl_unary[
    dtype: DType, width: SIMDLength, //, name: StaticString
](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
    comptime assert (
        dtype == DType.float32 or dtype == DType.float16
    ), "OpenCL math requires FP16 or FP32"
    comptime suffix = String(
        "f", bit_width_of[dtype]()
    ) if width == 1 else String("v", Int(width), "f", bit_width_of[dtype]())
    return llvm_intrinsic[
        get_static_string[String(name).replace("f32", suffix)](),
        SIMD[dtype, width],
        has_side_effect=False,
    ](x)


@inline(.always)
def _opencl_rsqrt[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
    return 1 / _opencl_unary["llvm.sqrt.f32"](x)


@inline(.always)
def _opencl_recip[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
    return 1 / x


def _opencl_math_fn[
    op: _UnaryMathOp
]() -> OptionalReg[_UnaryMathPluginHookFnType]:
    comptime match op:
        case .exp:
            return comptime (
                OptionalReg[_UnaryMathPluginHookFnType](
                    _opencl_unary["llvm.exp.f32"]
                )
            )
        case .exp2:
            return comptime (
                OptionalReg[_UnaryMathPluginHookFnType](
                    _opencl_unary["llvm.exp2.f32"]
                )
            )
        case .log:
            return comptime (
                OptionalReg[_UnaryMathPluginHookFnType](
                    _opencl_unary["llvm.log.f32"]
                )
            )
        case .log2:
            return comptime (
                OptionalReg[_UnaryMathPluginHookFnType](
                    _opencl_unary["llvm.log2.f32"]
                )
            )
        case .sqrt:
            return comptime (
                OptionalReg[_UnaryMathPluginHookFnType](
                    _opencl_unary["llvm.sqrt.f32"]
                )
            )
        case .sin:
            return comptime (
                OptionalReg[_UnaryMathPluginHookFnType](
                    _opencl_unary["llvm.sin.f32"]
                )
            )
        case .cos:
            return comptime (
                OptionalReg[_UnaryMathPluginHookFnType](
                    _opencl_unary["llvm.cos.f32"]
                )
            )
        case .rsqrt:
            return comptime (
                OptionalReg[_UnaryMathPluginHookFnType](_opencl_rsqrt)
            )
        case .recip:
            return comptime (
                OptionalReg[_UnaryMathPluginHookFnType](_opencl_recip)
            )
        case .tanh:
            return None


struct OpenCLPlugin(StdPlugin):
    """Selects the portable scalar OpenCL GPU hooks."""

    comptime name: __mlir_type.`!kgen.string` = _get_kgen_string["opencl"]()
    comptime HookFor[target: CompilationTarget] = OpenCLPluginHooks[target]


struct OpenCLPluginHooks[target: CompilationTarget](PluginHooks):
    """Provides scalar GPU indexing for the OpenCL target."""

    comptime device_kind = TargetDeviceKind.gpu
    comptime lowers_subvector_intrinsics: Bool = False
    # Avoid instantiating the generic device printf fallback. Assertion failures
    # terminate the invocation and report through the HAL queue error flag.
    comptime _handles_debug_assert: Bool = True
    # Device failures have a fixed HAL status, without formatted text.
    comptime _debug_assert_needs_message: Bool = False
    comptime gpu_id_fn[name: StaticString, dim: StaticString]: OptionalReg[
        _GPUIdPluginHookFnType
    ] = _opencl_gpu_id_fn[name, dim, Self.target]()

    comptime unary_math_fn[
        op: _UnaryMathOp, dtype: DType, width: Int
    ]: OptionalReg[_UnaryMathPluginHookFnType] = _opencl_math_fn[op]()
