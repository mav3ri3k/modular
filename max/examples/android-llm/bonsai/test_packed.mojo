# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the Apache License v2.0 with LLVM Exceptions: https://llvm.org/LICENSE.txt
"""Check vectorized packed projections and scalar tails through DeviceContext."""

import kernels
from max.gpu.host import DeviceContext
from std.testing import assert_equal


def check_width[width: Int, groups: Int](ctx: DeviceContext) raises:
    var inputs = ctx.enqueue_create_host_buffer[.uint32](64 * groups)
    var weights = ctx.enqueue_create_host_buffer[.uint32](8 * groups * width)
    var params = ctx.enqueue_create_host_buffer[.float16](groups * width)
    ctx.synchronize()
    for i in range(len(inputs)):
        inputs[i] = UInt32(0x3C003C00)  # Two exactly representable FP16 ones.
    for group in range(groups):
        for i in range(width):
            var sign = i % 4 - 1
            for word in range(8):
                weights[(group * 8 + word) * width + i] = UInt32(
                    sign + 1
                ) * UInt32(0x55555555)
            params[group * width + i] = Float16(group + 1) / 4
    var x = ctx.enqueue_create_buffer[.uint32](len(inputs))
    var w = ctx.enqueue_create_buffer[.uint32](len(weights))
    var p = ctx.enqueue_create_buffer[.float16](len(params))
    var partials = ctx.enqueue_create_buffer[.float32](groups * width)
    var output = ctx.enqueue_create_buffer[.float32](width)
    var candidates = ctx.enqueue_create_buffer[.float32](
        3 * ((width + 511) // 512)
    )
    ctx.enqueue_copy(x, inputs)
    ctx.enqueue_copy(w, weights)
    ctx.enqueue_copy(p, params)
    ctx.enqueue_function[kernels.packed_linear](
        x,
        w,
        p,
        partials,
        Int32(width),
        Int32(groups * 128),
        grid_dim=((width + 63) // 64, (groups + 7) // 8, 1),
        block_dim=64,
    )
    with partials.map_to_host() as actual:
        for tile in range((groups + 7) // 8):
            var expected: Float32 = 0
            for group in range(tile * 8, min(groups, (tile + 1) * 8)):
                expected += Float32(32 * (group + 1))
            for i in range(width):
                assert_equal(
                    actual[tile * width + i], expected * Float32(i % 4 - 1)
                )
    ctx.enqueue_function[kernels.packed_reduce](
        partials,
        output,
        Int32(width),
        Int32((groups + 7) // 8),
        grid_dim=(width + 127) // 128,
        block_dim=128,
    )
    with output.map_to_host() as actual:
        for i in range(width):
            assert_equal(
                actual[i], Float32(16 * groups * (groups + 1) * (i % 4 - 1))
            )
    output.enqueue_fill(Float32(12345))
    ctx.enqueue_function[kernels.packed_head](
        x,
        w,
        p,
        output,
        candidates,
        Int32(width),
        Int32(groups * 128),
        Int32(0),
        grid_dim=(width + 1023) // 1024,
        block_dim=128,
    )
    with output.map_to_host() as actual:
        for i in range(width):
            assert_equal(
                actual[i], Float32(16 * groups * (groups + 1) * (i % 4 - 1))
            )


def check_head_candidate_ties_and_nonfinite(ctx: DeviceContext) raises:
    # One subgroup covers 512 logits. Zero activations tie at every index;
    # infinities in the second launch must set the subgroup invalid flag.
    comptime width = 512
    comptime groups = 16
    var inputs = ctx.enqueue_create_host_buffer[.uint32](64 * groups)
    var weights = ctx.enqueue_create_host_buffer[.uint32](8 * groups * width)
    var params = ctx.enqueue_create_host_buffer[.float16](groups * width)
    var x = ctx.enqueue_create_buffer[.uint32](len(inputs))
    var w = ctx.enqueue_create_buffer[.uint32](len(weights))
    var p = ctx.enqueue_create_buffer[.float16](len(params))
    var output = ctx.enqueue_create_buffer[.float32](width)
    var candidates = ctx.enqueue_create_buffer[.float32](3)
    for i in range(len(inputs)):
        inputs[i] = UInt32(0)
    for i in range(len(weights)):
        weights[i] = UInt32(0x22222222)
    for i in range(len(params)):
        params[i] = Float16(0.25)
    ctx.enqueue_copy(x, inputs)
    ctx.enqueue_copy(w, weights)
    ctx.enqueue_copy(p, params)
    ctx.enqueue_function[kernels.packed_head](
        x,
        w,
        p,
        output,
        candidates,
        Int32(width),
        Int32(groups * 128),
        Int32(1),
        grid_dim=1,
        block_dim=128,
    )
    with candidates.map_to_host() as actual:
        assert_equal(actual[0], Float32(0))
        assert_equal(actual[1], Float32(0))
        assert_equal(actual[2], Float32(0))

    for i in range(len(inputs)):
        inputs[i] = UInt32(0x7C007C00)  # Two positive FP16 infinities.
    ctx.enqueue_copy(x, inputs)
    ctx.enqueue_function[kernels.packed_head](
        x,
        w,
        p,
        output,
        candidates,
        Int32(width),
        Int32(groups * 128),
        Int32(1),
        grid_dim=1,
        block_dim=128,
    )
    with candidates.map_to_host() as actual:
        assert_equal(actual[2], Float32(1))


@export("run_opencl_device_context")
def check_device_context() abi("C") -> Int32:
    try:
        var ctx = DeviceContext(api="opencl")
        check_width[1, 2](ctx)
        check_width[7, 5](ctx)
        check_width[8, 16](ctx)
        check_width[13, 17](ctx)
        check_width[16, 48](ctx)
        check_width[511, 5](ctx)
        check_width[513, 17](ctx)
        check_head_candidate_ties_and_nonfinite(ctx)

        print(
            "PASS: packed layer/head vectors, scalar tails and FP32 group"
            " scaling, greedy tie and nonfinite flags"
        )
        return 0
    except e:
        print("FAIL:", e)
        return 1
