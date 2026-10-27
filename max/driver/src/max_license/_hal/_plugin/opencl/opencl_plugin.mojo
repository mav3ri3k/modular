# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the MAX License: https://modular.com/legal/max-license
from std.sys import CompilationTarget
from std.collections.string.string_span import _get_kgen_string
from .._trait import HalPlugin, HalPluginHooks
from ..._machine import DeviceRef
from ...execution_config import GPUExecutionConfiguration


struct OpenCLHalPlugin(HalPlugin):
    """Selects portable grid/block execution for OpenCL."""

    comptime name: __mlir_type.`!kgen.string` = _get_kgen_string["opencl"]()
    comptime HookFor[target: CompilationTarget] = OpenCLHalPluginHooks[target]


struct OpenCLHalPluginHooks[target: CompilationTarget](HalPluginHooks):
    comptime ExecutionConfigType[device: DeviceRef] = GPUExecutionConfiguration
