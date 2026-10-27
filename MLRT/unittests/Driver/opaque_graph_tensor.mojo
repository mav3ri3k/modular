# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the MAX License: https://modular.com/legal/max-license
from builtin_primitives.primitives import (
    unpack_tensor,
    mgp_tensor_extract_buffer,
    mgp_tensor_create,
    mogg_tensor_init,
    mgp_buffer_slice,
    mgp_buffer_memset,
    mogg_tensor_create_reshape,
    mogg_tensor_create_broadcast,
)
from max.runtime.async_value import AnyAsyncValueRef, _AsyncValuePtr
from max.gpu.host import DeviceContext
from std.memory import Pointer
from layout import Coord, IntTuple, row_major
from extensibility import ManagedTensorSlice, IOSpec, IO
from std.utils import IndexList
from std.testing import assert_equal, assert_true


@export("test_opaque_graph_tensor")
def test_opaque_graph_tensor(
    value: _AsyncValuePtr[mut=True],
    context: Pointer[DeviceContext, MutUntrackedOrigin],
) abi("C") -> _AsyncValuePtr[mut=True]:
    try:
        var owner = AnyAsyncValueRef(value)
        var tensor = unpack_tensor[2, 2, .float32](owner)
        assert_true(tensor.is_opaque())
        assert_equal(tensor.shape()[0], 2)
        assert_equal(tensor.shape()[1], 8)
        assert_equal(tensor.bytecount(), 64)
        var view = tensor.tensor_view()
        comptime MutableView = ManagedTensorSlice[
            io_spec=IOSpec.MutableInput,
            static_spec=type_of(view).static_spec,
        ]
        var input_view = MutableView(
            opaque_view=tensor._opaque_view.value(), shape=tensor.shape()
        )._as_input()
        var reshaped = mogg_tensor_create_reshape[
            output_static_shape=IntTuple(4, 4)
        ](input_view, IndexList[2](4, 4))
        var broadcast = mogg_tensor_create_broadcast[
            output_static_shape=IntTuple(3, 4, 4)
        ](reshaped)
        assert_equal(broadcast.shape()[0], 3)
        assert_equal(broadcast.strides()[0], 0)
        assert_equal(broadcast.strides()[1], 4)
        assert_equal(broadcast._ptr.allocation, view._ptr.allocation)
        assert_equal(broadcast._ptr.byte_offset, view._ptr.byte_offset)
        assert_equal(broadcast._ptr.byte_size, view._ptr.byte_size)
        var bytes = mgp_tensor_extract_buffer(tensor)
        var rebuilt = mogg_tensor_init[
            dtype=.float32, rank=2, mut=False, input=IO.Input, alignment=4
        ](bytes.tensor_view(), row_major(Coord(IndexList[2](2, 8))))
        assert_equal(rebuilt.shape()[0], 2)
        assert_equal(rebuilt.shape()[1], 8)
        assert_equal(rebuilt._ptr.allocation, view._ptr.allocation)
        assert_equal(rebuilt._ptr.byte_offset, view._ptr.byte_offset)
        assert_equal(rebuilt._ptr.byte_size, view._ptr.byte_size)
        var sliced = mgp_buffer_slice(bytes, 16, 32, context[])
        mgp_buffer_memset["gpu"](sliced, 0x40000000, 4, context[])
        var result = mgp_tensor_create[2, 2, .float32](
            sliced, IndexList[2](2, 4)
        )
        return result.async_pack().take_handle()
    except e:
        print("FAIL opaque tensor:", e)
        return None


@export("test_opaque_graph_tensor_read")
def test_opaque_graph_tensor_read(
    value: _AsyncValuePtr[mut=True],
    context: Pointer[DeviceContext, MutUntrackedOrigin],
    output: Pointer[Float32, MutUntrackedOrigin],
) abi("C") -> Int32:
    try:
        var owner = AnyAsyncValueRef(value)
        var tensor = unpack_tensor[2, 2, .float32](owner)
        var bytes = mgp_tensor_extract_buffer(tensor)
        var view = bytes.to_device_buffer(context[])
        context[].enqueue_copy(output.unsafe_bitcast[Int8](), view)
        context[].synchronize()
        return 0
    except e:
        print("FAIL opaque tensor read:", e)
        return -1
