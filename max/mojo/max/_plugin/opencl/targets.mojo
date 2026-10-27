# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the Apache License v2.0 with LLVM Exceptions: https://llvm.org/LICENSE.txt
"""OpenCL target capabilities and scheduling bounds."""

from std.sys import CompilationTarget
from std._plugin.opencl import OpenCLPlugin
from .opencl_plugin import OpenCLMaxPlugin
from std._gpu.host.info import (
    GPUInfo,
    TargetAccelerator,
    TargetAcceleratorCollection,
    TargetAcceleratorType,
)

comptime _opencl_plugins = TypeList.of[OpenCLPlugin, OpenCLMaxPlugin]().values

comptime _opencl_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "aarch64-unknown-opencl", `,
        `stdlib_plugin = "opencl", opaque_plugin = `,
        _opencl_plugins,
        `, `,
        `arch = "opencl-generic", features = "", `,
        `data_layout = "e-m:e-p:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", `,
        `simd_bit_width = 32> : !kgen.target`,
    ]
]()

comptime OpenCLGeneric = TargetAccelerator[
    GPUInfo(
        name="OpenCLGeneric",
        api="opencl",
        arch_name="opencl-generic",
        compute=0,
        version="opencl_1_2",
        # A portable scheduling lower bound, not an assumed device CU count.
        sm_count=1,
        warp_size=1,
        threads_per_multiprocessor=1,
        shared_memory_per_multiprocessor=0,
        max_registers_per_block=0,
        max_thread_block_size=128,
    ),
    _opencl_target,
]()
"""OpenCL baseline; no subgroup width or optional hardware capability assumed."""

comptime _adreno_840_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "aarch64-unknown-opencl", `,
        `stdlib_plugin = "opencl", opaque_plugin = `,
        _opencl_plugins,
        `, `,
        `arch = "adreno-840", features = "", `,
        `data_layout = "e-m:e-p:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", `,
        `simd_bit_width = 128> : !kgen.target`,
    ]
]()

comptime Adreno840 = TargetAccelerator[
    GPUInfo(
        name="Adreno840",
        api="opencl",
        arch_name="adreno-840",
        compute=0,
        version="opencl_3_0",
        # CL_DEVICE_MAX_COMPUTE_UNITS / LOCAL_MEM_SIZE on Adreno 840.
        sm_count=12,
        warp_size=64,
        # At least one maximum-size workgroup can reside on a compute unit.
        # This is a scheduling lower bound, not an occupancy guarantee.
        threads_per_multiprocessor=1024,
        shared_memory_per_multiprocessor=32768,
        max_registers_per_block=0,
        max_thread_block_size=1024,
    ),
    _adreno_840_target,
]()
"""Adreno 840; subgroup kernels validate their native width at dispatch."""


struct OpenCLTargets(TargetAcceleratorCollection):
    """OpenCL targets selectable without probing a GPU on the build host."""

    comptime vendor_name: String = "OpenCL"
    comptime RAW_TARGETS = TypeList.of[
        Trait=TargetAcceleratorType,
        OpenCLGeneric._with_cli_values[["opencl-generic"]],
        Adreno840._with_cli_values[["adreno-840"]],
    ]().values

    @staticmethod
    def normalize_target_arch(arch: StaticString) -> String:
        return arch.replace("opencl:", "")

    @staticmethod
    def supported_archs_message() -> String:
        return String(
            "  OpenCL: opencl-generic (OpenCL 1.2 baseline), adreno-840"
            " (validated 64-lane subgroups)\n"
        )
