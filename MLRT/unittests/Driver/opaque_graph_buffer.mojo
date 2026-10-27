# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the MAX License: https://modular.com/legal/max-license
from builtin_primitives.primitives import (
    unpack_buffer_ref,
    mgp_buffer_slice,
    mgp_buffer_memset,
)
from max.runtime.async_value import AnyAsyncValueRef, _AsyncValuePtr
from max.gpu.host import DeviceContext
from std.memory import Pointer


@export("test_opaque_graph_slice")
def test_opaque_graph_slice(
    value: _AsyncValuePtr[mut=True],
    context: Pointer[DeviceContext, MutUntrackedOrigin],
) abi("C") -> _AsyncValuePtr[mut=True]:
    try:
        var owner = AnyAsyncValueRef(value)
        var buffer = unpack_buffer_ref(owner)
        var first = mgp_buffer_slice(buffer, 4, 32, context[])
        var nested = mgp_buffer_slice(first, 8, 16, context[])
        return nested.async_pack().take_handle()
    except e:
        print("FAIL graph slice:", e)
        return None


@export("test_opaque_graph_fill_read")
def test_opaque_graph_fill_read(
    value: _AsyncValuePtr[mut=True],
    context: Pointer[DeviceContext, MutUntrackedOrigin],
    output: Pointer[Float32, MutUntrackedOrigin],
) abi("C") -> Int32:
    try:
        var owner = AnyAsyncValueRef(value)
        var buffer = unpack_buffer_ref(owner)
        mgp_buffer_memset["gpu"](buffer, 0x40000000, 4, context[])
        var view = buffer.to_device_buffer(context[])
        context[].enqueue_copy(output.unsafe_bitcast[Int8](), view)
        context[].synchronize()
        return 0
    except e:
        print("FAIL graph fill/read:", e)
        return 1
