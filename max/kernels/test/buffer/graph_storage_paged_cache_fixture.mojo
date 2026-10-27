# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the Apache License v2.0 with LLVM Exceptions: https://llvm.org/LICENSE.txt
"""Phone fixture for ordinary paged KV views backed by opaque graph storage."""

from extensibility import (
    IOSpec,
    ManagedTensorSlice,
    get_row_major_tensor_spec_static,
)
from extensibility.managed_tensor_slice import (
    StaticTensorSpec,
    _IndexListToTileLayout,
)
from kv_cache.types import KVCacheT
from nn.kv_cache import generic_get_paged_cache
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.testing import assert_equal
from std.utils import IndexList


def _write_cache[Cache: KVCacheT](cache: Cache, bias: Float32):
    var i = global_idx.x
    if i < 16:
        var token, d = divmod(i, 4)
        var storage = cache.block_paged_storage[1](0, token, 0, d)
        Cache.Engine.unsafe_ptr(storage)[] = (
            bias + Float32(token * 4 + d)
        ).cast[Cache.dtype]()


def _run[padded: Bool](ctx: DeviceContext) raises:
    comptime pitch = 40 if padded else 32
    comptime block_spec = StaticTensorSpec[
        .float32,
        6,
        static_layout=_IndexListToTileLayout[
            IndexList[6](4, 2, 2, 2, 1, 4),
            IndexList[6](-1, -1, -1, -1, -1, -1),
        ],
    ](4, .GENERIC)
    comptime scalar_u32 = get_row_major_tensor_spec_static[.uint32, 1, 1]()
    comptime scalar_i64 = get_row_major_tensor_spec_static[.int64, 1, 1]()
    comptime lookup_spec = get_row_major_tensor_spec_static[.uint32, 2, 1, 2]()
    var blocks = ctx.enqueue_create_buffer[.float32](4 * pitch)
    var lengths = ctx.enqueue_create_buffer[.uint32](1)
    var lookup = ctx.enqueue_create_buffer[.uint32](2)
    blocks.enqueue_fill(Float32(-77))
    lengths.enqueue_fill(UInt32(1))
    var table = ctx.enqueue_create_host_buffer[.uint32](2)
    ctx.synchronize()
    table[0] = 2
    table[1] = 0
    ctx.enqueue_copy(lookup, table)
    var block_view = ManagedTensorSlice[
        io_spec=IOSpec.MutableInput, static_spec=block_spec
    ](blocks.device_ptr(), (4, 2, 2, 2, 1, 4), (pitch, 16, 8, 4, 4, 1))
    var lengths_view = ManagedTensorSlice[
        io_spec=IOSpec.Input, static_spec=scalar_u32
    ](lengths.device_ptr(), (1,))
    var lookup_view = ManagedTensorSlice[
        io_spec=IOSpec.Input, static_spec=lookup_spec
    ](lookup.device_ptr(), (1, 2))
    # Launch metadata is host-resident; the cache contents, lengths and LUT
    # stay opaque until ordinary DevicePassable encoding at kernel launch.
    var stride: List[Int64] = [Int64(pitch if padded else -1)]
    var prompt: List[UInt32] = [3]
    var maximum: List[UInt32] = [4]
    var stride_view = ManagedTensorSlice[
        io_spec=IOSpec.Input, static_spec=scalar_i64
    ](stride.unsafe_ptr(), (1,))
    var prompt_view = ManagedTensorSlice[
        io_spec=IOSpec.Input, static_spec=scalar_u32
    ](prompt.unsafe_ptr(), (1,))
    var maximum_view = ManagedTensorSlice[
        io_spec=IOSpec.Input, static_spec=scalar_u32
    ](maximum.unsafe_ptr(), (1,))
    var collection = generic_get_paged_cache(
        block_view,
        stride_view,
        lengths_view,
        lookup_view,
        prompt_view,
        maximum_view,
    )
    for layer in range(2):
        var key = collection.get_key_cache(layer)
        var value = collection.get_value_cache(layer)
        ctx.enqueue_function[_write_cache[type_of(key)]](
            key, Float32(1 + layer * 100), grid_dim=1, block_dim=16
        )
        ctx.enqueue_function[_write_cache[type_of(value)]](
            value, Float32(21 + layer * 100), grid_dim=1, block_dim=16
        )
    var expected = List(length=4 * pitch, fill=Float32(-77))
    for layer in range(2):
        for token in range(4):
            var page = 2 if token < 2 else 0
            for d in range(4):
                var offset = page * pitch + layer * 8 + (token % 2) * 4 + d
                expected[offset] = Float32(1 + layer * 100 + token * 4 + d)
                expected[offset + 16] = expected[offset] + 20
    with blocks.map_to_host() as actual:
        for i in range(4 * pitch):
            assert_equal(actual[i], expected[i])
    ctx.synchronize()
    _ = blocks^
    _ = lengths^
    _ = lookup^
    _ = stride^
    _ = prompt^
    _ = maximum^


@export("run_opencl_device_context")
def check_device_context() abi("C") -> Int32:
    try:
        with DeviceContext() as ctx:
            _run[False](ctx)
            _run[True](ctx)
        print("PASS: graph-storage paged KV cache, reordered pages and padding")
        return 0
    except e:
        print("FAIL:", e)
        return 1
