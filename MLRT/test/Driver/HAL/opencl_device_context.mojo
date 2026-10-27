# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the MAX License: https://modular.com/legal/max-license
"""Android DeviceContext integration fixture; no example-specific launch ABI."""

from std.memory import Pointer, unsafe_stack_allocation
from std.builtin.device_passable import (
    DevicePassable,
    DevicePointerLike,
    DeviceTypeEncoder,
)
from std.collections import Span
from std.math import sqrt, fma
from std.memory import bitcast
from std.atomic import Atomic, Ordering
from layout import Coord, TileTensor, row_major, coord_to_index_list
from max.algorithm.functional import elementwise
from extensibility import DynamicTensor
from extensibility.managed_tensor_slice import (
    _TensorDataPointer,
    simd_load_from_tensor_pointer,
    simd_store_into_tensor_pointer,
)
from std.utils import IndexList
from std.testing import assert_equal, assert_true
from max.gpu import global_idx, thread_idx, block_idx
from max.gpu.sync import barrier
from max.gpu.memory import external_memory
from max.gpu.host import DeviceContext
from max.gpu.host.info import default_accelerator
from max.gpu.primitives.warp import shuffle_xor


def transform(
    source: Pointer[Float32, MutAnyOrigin],
    output: Pointer[Float32, MutAnyOrigin],
    count: Int32,
):
    var i = global_idx.x
    if i < Int(count):
        output[unsafe_offset=i] = source[unsafe_offset=i] * 2 + 1


@fieldwise_init
struct DevicePair(ImplicitlyCopyable, TrivialRegisterPassable):
    var count: Int32
    var output: Pointer[Float32, MutUntrackedOrigin]


@fieldwise_init
struct DeviceAggregate(ImplicitlyCopyable, TrivialRegisterPassable):
    var source: Pointer[Float32, MutUntrackedOrigin]
    var tail: DevicePair


comptime TestPointer = DevicePointerLike & DevicePassable & ImplicitlyCopyable & TrivialRegisterPassable


@fieldwise_init
struct HostPair[Out: TestPointer](
    DevicePassable, ImplicitlyCopyable, TrivialRegisterPassable
):
    comptime device_type: AnyType = DevicePair
    var count: Int32
    var output: Self.Out

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode_fields[Self.device_type](self, target)

    @staticmethod
    def get_type_name() -> String:
        return "HostPair"


@fieldwise_init
struct HostAggregate[Source: TestPointer, Out: TestPointer](
    DevicePassable, ImplicitlyCopyable, TrivialRegisterPassable
):
    comptime device_type: AnyType = DeviceAggregate
    var source: Self.Source
    var tail: HostPair[Self.Out]

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode_fields[Self.device_type](self, target)

    @staticmethod
    def get_type_name() -> String:
        return "HostAggregate"


def aggregate_transform(packed: DeviceAggregate, bias: Float32):
    var i = global_idx.x
    if i < Int(packed.tail.count):
        packed.tail.output[unsafe_offset=i] = (
            packed.source[unsafe_offset=i] * 3 + bias
        )


def vector_transform(
    source: Pointer[Float32, MutAnyOrigin],
    output: Pointer[Float32, MutAnyOrigin],
    bias: SIMD[.float32, 4],
):
    var i = global_idx.x * 4
    var v = source.unsafe_offset(i).unsafe_load[width=4]()
    var half = v.cast[.float16]()
    var restored = half.cast[.float32]().shuffle[1, 0, 3, 2]()
    var result = restored.gt(SIMD[.float32, 4](7)).select(
        restored * 2 + bias, -restored
    )
    output.unsafe_offset(i).unsafe_store(result)


def vector_sqrt(
    source: Pointer[Float32, MutAnyOrigin],
    output: Pointer[Float32, MutAnyOrigin],
):
    var i = global_idx.x * 4
    var value = source.unsafe_offset(i).unsafe_load[width=4]()
    output.unsafe_offset(i).unsafe_store(sqrt(value * value))


def vector_sum(
    source: Pointer[Float32, MutAnyOrigin],
    output: Pointer[Float32, MutAnyOrigin],
):
    var i = global_idx.x * 4
    output[unsafe_offset=global_idx.x] = (
        source.unsafe_offset(i).unsafe_load[width=4]().reduce_add()
    )


def wide_vector_argument(
    output: Pointer[Float32, MutAnyOrigin], value: SIMD[.float32, 16]
):
    output.unsafe_store(value)


def workgroup_reduce[
    dynamic: Bool
](
    source: Pointer[Float32, MutAnyOrigin],
    output: Pointer[Float32, MutAnyOrigin],
):
    var shared = external_memory[
        Float32, address_space=.SHARED, alignment=4
    ]() if dynamic else unsafe_stack_allocation[
        8, Float32, address_space=.SHARED
    ]()
    var tid = thread_idx.x
    shared[unsafe_offset=tid] = source[unsafe_offset=global_idx.x]
    barrier()
    var stride = 4
    while stride:
        if tid < stride:
            shared[unsafe_offset=tid] += shared[unsafe_offset=tid + stride]
        barrier()
        stride //= 2
    if tid == 0:
        output[unsafe_offset=block_idx.x] = shared[unsafe_offset=0]


def subgroup_shuffle(output: Pointer[Float32, MutAnyOrigin]):
    output[unsafe_offset=global_idx.x] = shuffle_xor(
        Float32(global_idx.x), UInt32(1)
    )


def atomic_tickets(
    counter: Pointer[UInt32, MutAnyOrigin],
    tickets: Pointer[UInt32, MutAnyOrigin],
):
    tickets[unsafe_offset=global_idx.x] = Atomic[UInt32].fetch_add(
        counter, UInt32(1)
    )


def atomic_load_store(
    counter: Pointer[UInt32, MutAnyOrigin],
    output: Pointer[UInt32, MutAnyOrigin],
):
    Atomic[UInt32].store[ordering=Ordering.RELEASE](counter, UInt32(42))
    output[] = Atomic[UInt32].load[ordering=Ordering.ACQUIRE](counter.as_imm())


def atomic_compare_and_float(
    counter: Pointer[UInt32, MutAnyOrigin],
    total: Pointer[Float32, MutAnyOrigin],
    result: Pointer[UInt32, MutAnyOrigin],
):
    var expected = UInt32(42)
    var success = Atomic[UInt32].compare_exchange(counter, expected, UInt32(43))
    result[] = UInt32(success)
    expected = 42
    success = Atomic[UInt32].compare_exchange(counter, expected, UInt32(99))
    result[unsafe_offset=1] = UInt32(success)
    result[unsafe_offset=2] = expected
    Atomic[Float32].store[ordering=Ordering.RELEASE](total, Float32(0))
    result[unsafe_offset=3] = UInt32(
        Atomic[Float32].load[ordering=Ordering.ACQUIRE](total.as_imm()) == 0
    )
    _ = Atomic[Float32].fetch_add(total, Float32(1.5))


def float_atomic_contention(total: Pointer[Float32, MutAnyOrigin]):
    _ = Atomic[Float32].fetch_add(total, Float32(0.5))


def wide_atomic(counter: Pointer[UInt64, MutAnyOrigin]):
    _ = Atomic[UInt64].fetch_add(counter, UInt64(1))


def standard_half_pair(output: Pointer[Float32, MutAnyOrigin]):
    var packed = UInt32(0x3C003800)
    var pair = bitcast[.float16, 2](packed)
    var weight = SIMD[.int32, 2](-1, 2).cast[.float16]()
    var acc = fma(pair, weight, SIMD[.float16, 2](0))
    output[unsafe_offset=0] = acc[0].cast[.float32]()
    output[unsafe_offset=1] = acc[1].cast[.float32]()
    output[unsafe_offset=2] = acc.cast[.float32]().reduce_add()
    var encoded = bitcast[.uint32, 1](
        SIMD[.float32, 2](0.5, 1).cast[.float16]()
    )
    output[unsafe_offset=3] = Float32(1) if encoded == packed else Float32(0)


comptime GraphTensorView = DynamicTensor[.float32, 2]


@fieldwise_init
struct GraphFusionCapture(TrivialRegisterPassable):
    var ptr: _TensorDataPointer[.float32]
    var shape: IndexList[2]
    var strides: IndexList[2]

    def load(self, index: IndexList[2]) -> Float32:
        return simd_load_from_tensor_pointer[
            static_spec=GraphTensorView.static_spec, simd_width=1
        ](self.ptr, self.shape, self.strides, index)

    def store(self, index: IndexList[2], value: Float32):
        simd_store_into_tensor_pointer[static_spec=GraphTensorView.static_spec](
            self.ptr, self.shape, self.strides, index, value
        )


def graph_tensor_projection(
    source: GraphTensorView.device_type, output: GraphTensorView.device_type
):
    var src = GraphTensorView(source)
    var dst = GraphTensorView(output)
    var i = global_idx.x
    dst[i // 4, i % 4] = src[i // 4, i % 4] * 3 + 2


def private_scratch(
    source: Pointer[Float32, MutAnyOrigin],
    output: Pointer[Float32, MutAnyOrigin],
):
    var scratch = unsafe_stack_allocation[8, Float32]()
    for j in range(8):
        scratch[unsafe_offset=j] = source[unsafe_offset=(global_idx.x + j) % 16]
    var value = Float32(0)
    for j in range(8):
        value += scratch[unsafe_offset=(j + global_idx.x) % 8]
    output[unsafe_offset=global_idx.x] = value


def check_device_context() raises:
    var ctx = DeviceContext()
    var source = ctx.enqueue_create_buffer[.float32](16)
    var output = ctx.enqueue_create_buffer[.float32](16)
    var empty = ctx.enqueue_create_buffer[.float32](0)
    var empty_values = List[Float32]()
    assert_equal(len(empty), 0)
    ctx.enqueue_copy(empty, Span(empty_values))
    ctx.enqueue_function[transform](
        empty, output, Int32(0), grid_dim=1, block_dim=1
    )
    ctx.synchronize()
    var values = List[Float32]()
    for i in range(16):
        values.append(Float32(i))
    ctx.enqueue_copy(source, Span(values))
    ctx.enqueue_copy(output, Span(values))
    ctx.enqueue_function[private_scratch](
        source, output, grid_dim=1, block_dim=16
    )
    var scratch_result = List[Float32](length=16, fill=-1)
    ctx.enqueue_copy(Span(scratch_result), output)
    ctx.synchronize()
    for i in range(16):
        var expected = Float32(0)
        for j in range(8):
            expected += Float32((i + j) % 16)
        assert_equal(scratch_result[i], expected)
    ctx.enqueue_copy(output, Span(values))
    print("PASS: private stack scratch with dynamic indexing")
    ctx.enqueue_function[standard_half_pair](output, grid_dim=1, block_dim=1)
    ctx.enqueue_copy(Span(scratch_result), output)
    ctx.synchronize()
    assert_equal(scratch_result[0], Float32(-0.5))
    assert_equal(scratch_result[1], Float32(2))
    assert_equal(scratch_result[2], Float32(1.5))
    assert_equal(scratch_result[3], Float32(1))
    ctx.enqueue_copy(output, Span(values))
    print("PASS: standard half2 casts, bitcasts, fma and reductions")
    var source_view = GraphTensorView(source.device_ptr(), IndexList[2](4, 4))
    var output_view = GraphTensorView(output.device_ptr(), IndexList[2](4, 4))
    ctx.enqueue_function[graph_tensor_projection](
        source_view, output_view, grid_dim=1, block_dim=16
    )
    ctx.enqueue_copy(Span(scratch_result), output)
    ctx.synchronize()
    for i in range(16):
        assert_equal(scratch_result[i], Float32(i * 3 + 2))
    ctx.enqueue_copy(output, Span(values))
    print(
        "PASS: graph tensor view projects through standard DevicePassable"
        " encoding"
    )

    def graph_tensor_closure() {var source_view, var output_view}:
        var i = Int(global_idx.x)
        if i < 16:
            output_view[i // 4, i % 4] = source_view[i // 4, i % 4] * 2 + 7

    ctx.enqueue_function(graph_tensor_closure, grid_dim=1, block_dim=16)
    ctx.enqueue_copy(Span(scratch_result), output)
    ctx.synchronize()
    for i in range(16):
        assert_equal(scratch_result[i], Float32(i * 2 + 7))
    ctx.enqueue_copy(output, Span(values))
    print("PASS: graph tensor views captured by a standard device closure")
    # A transposed input exercises non-unit strides and conservative alignment.
    var fusion_input = GraphFusionCapture(
        source_view._ptr, source_view.shape(), IndexList[2](1, 4)
    )
    var fusion_output = GraphFusionCapture(
        output_view._ptr, output_view.shape(), output_view.strides()
    )

    def graph_fusion_closure() {var fusion_input, var fusion_output}:
        var i = Int(global_idx.x)
        var index = IndexList[2](i // 4, i % 4)
        fusion_output.store(index, fusion_input.load(index) * 4 + 3)

    ctx.enqueue_function(graph_fusion_closure, grid_dim=1, block_dim=16)
    ctx.enqueue_copy(Span(scratch_result), output)
    ctx.synchronize()
    for i in range(16):
        assert_equal(scratch_result[i], Float32(((i % 4) * 4 + i // 4) * 4 + 3))
    ctx.enqueue_copy(output, Span(values))
    print("PASS: typed fusion captures resolve storage inside device kernels")
    var tail_source = GraphTensorView(source.device_ptr(), IndexList[2](2, 3))
    var tail_output = GraphTensorView(output.device_ptr(), IndexList[2](2, 3))

    def elementwise_body[
        width: Int, alignment: Int = 1
    ](index: Coord) {var tail_source, var tail_output,}:
        var idx = rebind[IndexList[2]](coord_to_index_list(index))
        var loaded = simd_load_from_tensor_pointer[
            static_spec=GraphTensorView.static_spec,
            simd_width=width,
            element_alignment=alignment,
        ](tail_source._ptr, tail_source.shape(), tail_source.strides(), idx)
        simd_store_into_tensor_pointer[
            static_spec=GraphTensorView.static_spec,
            element_alignment=alignment,
        ](
            tail_output._ptr,
            tail_output.shape(),
            tail_output.strides(),
            idx,
            loaded * 5 - 2,
        )

    elementwise[simd_width=4, target="gpu"](
        elementwise_body, Coord(IndexList[2](2, 3)), ctx
    )
    ctx.enqueue_copy(Span(scratch_result), output)
    ctx.synchronize()
    for i in range(16):
        assert_equal(scratch_result[i], Float32(i * 5 - 2 if i < 6 else i))
    ctx.enqueue_copy(output, Span(values))
    print(
        "PASS: standard elementwise dispatch with unknown occupancy and row"
        " tails"
    )
    var src = source.create_sub_buffer[.float32](1, 8)
    var dst = output.create_sub_buffer[.float32](2, 10).create_sub_buffer[
        .float32
    ](1, 5)
    assert_true(src.device_ptr() == source.device_ptr() + 1)
    assert_true(dst.device_ptr() == output.device_ptr() + 3)
    assert_true(src.device_ptr() < source.device_ptr() + 2)
    for offset, size in [
        (-1, 1),
        (0, -1),
        (16, 1),
        (4611686018427387904, 1),
        (9223372036854775807, 1),
    ]:
        var rejected = False
        try:
            _ = source.create_sub_buffer[.float32](offset, size)
        except:
            rejected = True
        assert_true(rejected)
    ctx.enqueue_function[transform](
        src.device_ptr(), dst, Int32(5), grid_dim=1, block_dim=8
    )
    ctx.synchronize()
    var actual = List[Float32](length=16, fill=0)
    ctx.enqueue_copy(Span(actual), output)
    ctx.synchronize()
    for i in range(16):
        assert_equal(
            actual[i], Float32((i - 2) * 2 + 1) if 3 <= i < 8 else Float32(i)
        )
    print(
        "PASS: ordinary DeviceContext launch, nested buffer views and host"
        " copies"
    )
    ctx.enqueue_function[aggregate_transform](
        HostAggregate(src.device_ptr(), HostPair(Int32(5), dst.device_ptr())),
        Float32(7),
        grid_dim=1,
        block_dim=8,
    )
    ctx.enqueue_copy(Span(actual), output)
    ctx.synchronize()
    for i in range(16):
        assert_equal(
            actual[i], Float32((i - 2) * 3 + 7) if 3 <= i < 8 else Float32(i)
        )
    print("PASS: nested aggregate pointers, scalar fields and offset views")
    var tile = TileTensor(dst.device_ptr(), row_major[5]())
    var closure_bias = Float32(9)

    def tensor_closure() {var tile, var closure_bias}:
        var i = global_idx.x
        if i < 5:
            tile[i] = Float32(i) + closure_bias

    ctx.enqueue_function(tensor_closure, grid_dim=1, block_dim=8)
    ctx.enqueue_copy(Span(actual), output)
    ctx.synchronize()
    for i in range(16):
        assert_equal(
            actual[i], Float32(i - 3 + 9) if 3 <= i < 8 else Float32(i)
        )
    print(
        "PASS: captured tensor and scalar through the standard closure encoder"
    )
    ctx.enqueue_function[vector_transform](
        source, output, SIMD[.float32, 4](1, 2, 3, 4), grid_dim=1, block_dim=4
    )
    ctx.enqueue_copy(Span(actual), output)
    ctx.synchronize()
    for i in range(16):
        var v = Float32(i ^ 1)
        assert_equal(actual[i], v * 2 + Float32(i % 4 + 1) if v > 7 else -v)
    print(
        "PASS: ordinary SIMD/FP16 conversion, shuffle, vector argument and"
        " masked select"
    )
    ctx.enqueue_function[vector_sqrt](source, output, grid_dim=1, block_dim=4)
    ctx.enqueue_copy(Span(actual), output)
    ctx.synchronize()
    for i in range(16):
        assert_equal(actual[i], Float32(i))
    print("PASS: standard vector math hooks")
    ctx.enqueue_function[vector_sum](source, output, grid_dim=1, block_dim=4)
    ctx.enqueue_copy(Span(actual), output)
    ctx.synchronize()
    for i in range(4):
        assert_equal(actual[i], Float32(16 * i + 6))
    print("PASS: standard SIMD reduction")
    ctx.enqueue_function[wide_vector_argument](
        output, SIMD[.float32, 16](7), grid_dim=1, block_dim=1
    )
    ctx.enqueue_copy(Span(actual), output)
    ctx.synchronize()
    for value in actual:
        assert_equal(value, Float32(7))
    print("PASS: aligned wide-vector argument after an opaque buffer view")
    ctx.enqueue_function[workgroup_reduce[False]](
        source, output, grid_dim=2, block_dim=8
    )
    ctx.enqueue_copy(Span(actual), output)
    ctx.synchronize()
    assert_equal(actual[0], Float32(28))
    assert_equal(actual[1], Float32(92))
    print(
        "PASS: standard shared allocation and workgroup barriers across two"
        " workgroups"
    )
    ctx.enqueue_function[workgroup_reduce[True]](
        source, output, grid_dim=2, block_dim=8, shared_mem_bytes=32
    )
    ctx.enqueue_copy(Span(actual), output)
    ctx.synchronize()
    assert_equal(actual[0], Float32(28))
    assert_equal(actual[1], Float32(92))
    print(
        "PASS: dynamic shared memory through the standard launch configuration"
    )
    var counter = ctx.enqueue_create_buffer[.uint32](1)
    var tickets = ctx.enqueue_create_buffer[.uint32](512)
    var count = List[UInt32](length=1, fill=0)
    ctx.enqueue_copy(counter, Span(count))
    ctx.enqueue_function[atomic_tickets](
        counter, tickets, grid_dim=4, block_dim=128
    )
    var ticket_values = List[UInt32](length=512, fill=0)
    ctx.enqueue_copy(Span(ticket_values), tickets)
    ctx.enqueue_copy(Span(count), counter)
    ctx.synchronize()
    assert_equal(count[0], UInt32(512))
    var seen = List[Bool](length=512, fill=False)
    for ticket in ticket_values:
        assert_true(ticket < 512)
        assert_true(not seen[Int(ticket)])
        seen[Int(ticket)] = True
    ctx.enqueue_function[atomic_load_store](
        counter, tickets, grid_dim=1, block_dim=1
    )
    ctx.enqueue_copy(Span(ticket_values), tickets)
    ctx.synchronize()
    assert_equal(ticket_values[0], UInt32(42))
    print(
        "PASS: standard 32-bit atomics, unique tickets across four workgroups"
        " and acquire/release load/store"
    )
    var total = ctx.enqueue_create_buffer[.float32](1)
    total.enqueue_fill(0)
    ctx.enqueue_function[atomic_compare_and_float](
        counter, total, tickets, grid_dim=1, block_dim=1
    )
    ctx.enqueue_copy(Span(ticket_values), tickets)
    ctx.synchronize()
    assert_equal(ticket_values[0], UInt32(1))
    assert_equal(ticket_values[1], UInt32(0))
    assert_equal(ticket_values[2], UInt32(43))
    assert_equal(ticket_values[3], UInt32(1))
    ctx.enqueue_function[float_atomic_contention](
        total, grid_dim=4, block_dim=128
    )
    var total_value = List[Float32](length=1, fill=0)
    ctx.enqueue_copy(Span(total_value), total)
    ctx.synchronize()
    assert_equal(total_value[0], Float32(257.5))
    print(
        "PASS: compare-exchange success/failure and floating CAS updates across"
        " four workgroups"
    )
    var wide = ctx.enqueue_create_buffer[.uint64](1)
    wide.enqueue_fill(0)
    var supported_wide = True
    try:
        ctx.enqueue_function[wide_atomic](wide, grid_dim=1, block_dim=1)
    except e:
        supported_wide = False
        assert_true("cl_khr_int64_" in String(e))
    if supported_wide:
        var wide_value = List[UInt64](length=1, fill=0)
        ctx.enqueue_copy(Span(wide_value), wide)
        ctx.synchronize()
        assert_equal(wide_value[0], UInt64(1))
        print("PASS: native 64-bit atomic capability")
    else:
        print(
            "PASS: rejects 64-bit atomics when required device extensions are"
            " absent"
        )
    comptime if default_accelerator()._is_arch["adreno-840"]():
        var shuffled = ctx.enqueue_create_buffer[.float32](128)
        ctx.enqueue_function[subgroup_shuffle](
            shuffled, grid_dim=1, block_dim=128
        )
        var result = List[Float32](length=128, fill=-1)
        ctx.enqueue_copy(Span(result), shuffled)
        ctx.synchronize()
        for i in range(128):
            assert_equal(result[i], Float32(i ^ 1))
        var rejected = False
        try:
            ctx.enqueue_function[subgroup_shuffle](
                shuffled, grid_dim=1, block_dim=32
            )
        except:
            rejected = True
        assert_true(rejected)
        print(
            "PASS: standard subgroup shuffle across two subgroups; rejects"
            " incompatible launch width"
        )
    var side = ctx.create_stream()
    var ready = ctx.create_event()
    var completed = ctx.create_event()
    ctx.enqueue_function[transform](
        source, output, Int32(16), grid_dim=1, block_dim=16
    )
    ctx.stream().record_event(ready)
    side.enqueue_wait_for(ready)
    side.enqueue_function[transform](
        output, source, Int32(16), grid_dim=1, block_dim=16
    )
    side.record_event(completed)
    ctx.stream().enqueue_wait_for(completed)
    ctx.enqueue_copy(Span(actual), source)
    ctx.synchronize()
    for i in range(16):
        assert_equal(actual[i], Float32(4 * i + 3))
    print("PASS: ordinary DeviceStream launches and cross-queue event ordering")


@export("run_opencl_device_context")
def run_opencl_device_context() abi("C") -> Int32:
    """Runs the fixture from a native embedding process."""
    try:
        check_device_context()
        return 0
    except e:
        print("FAIL:", e)
        return 1
