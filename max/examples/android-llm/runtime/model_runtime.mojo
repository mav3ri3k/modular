# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the Apache License v2.0 with LLVM Exceptions: https://llvm.org/LICENSE.txt
"""C embedding boundary for an owning, ordinary Mojo DeviceContext model.

Only host pointers cross this boundary. Device allocations and kernel argument
encoding stay in Mojo; C++ never sees a HAL memory handle or kernel bundle.
"""
from std.memory import (
    Pointer,
    OptionalPointer,
    alloc,
    dealloc,
    Layout,
    ThinAllocation,
)
from max.gpu.host import DeviceContext, DeviceBuffer
from model import forward


@fieldwise_init
struct ModelState(Movable):
    var ctx: DeviceContext
    var weights: List[DeviceBuffer[.uint8]]
    var buffers: List[DeviceBuffer[.uint8]]
    var head_gpu_columns: Int32


comptime StatePtr = Pointer[ModelState, MutUntrackedOrigin]
comptime Bytes = Pointer[UInt8, MutUntrackedOrigin]


@export
def android_model_create() abi("C") -> (
    OptionalPointer[ModelState, MutUntrackedOrigin]
):
    try:
        var state = ModelState(
            DeviceContext(api="opencl"),
            List[DeviceBuffer[.uint8]](),
            List[DeviceBuffer[.uint8]](),
            -1,
        )
        var box = alloc(Layout[ModelState].single())
        box.unsafe_ptr().unsafe_write(state^)
        return box^.unsafe_leak()
    except e:
        print("Model create:", e)
        return None


@export
def android_model_destroy(state: StatePtr) abi("C"):
    try:
        state[].ctx.synchronize()
    except e:
        print("Model synchronize during destroy:", e)
    state.unsafe_deinit_pointee()
    dealloc(
        ThinAllocation(unsafe_owned_ptr=state).unsafe_with_layout(
            Layout[ModelState].single()
        )
    )


@export
def android_model_allocate(
    state: StatePtr,
    weights: Int32,
    size: Int64,
    data: OptionalPointer[UInt8, MutUntrackedOrigin],
    zero: Int32,
) abi("C") -> Int64:
    try:
        if size <= 0:
            raise Error("Model buffer size must be positive")
        var buf = state[].ctx.enqueue_create_buffer[.uint8](Int(size))
        if case .Some(ptr) = data:
            state[].ctx.enqueue_copy(buf, ptr)
            # C++ may release the input storage immediately after return.
            state[].ctx.synchronize()
        elif zero != 0:
            buf.enqueue_fill(0)
        if weights != 0:
            state[].weights.append(buf^)
        else:
            state[].buffers.append(buf^)
        return 0
    except e:
        print("Model allocate:", e)
        return -1


@export
def android_model_alias_weight(state: StatePtr, index: Int32) abi("C") -> Int64:
    state[].buffers.append(state[].weights[Int(index)])
    return 0


@export
def android_model_synchronize(state: StatePtr) abi("C") -> Int64:
    try:
        state[].ctx.synchronize()
        return 0
    except e:
        print("Model synchronize:", e)
        return -1


@export
def android_model_zero(state: StatePtr, index: Int32) abi("C") -> Int64:
    try:
        state[].buffers[Int(index)].enqueue_fill(0)
        return 0
    except e:
        print("Model zero:", e)
        return -1


@export
def android_model_read(
    state: StatePtr, index: Int32, data: Bytes, size: Int64
) abi("C") -> Int64:
    try:
        var buf = (
            state[].buffers[Int(index)].create_sub_buffer[.uint8](0, Int(size))
        )
        state[].ctx.enqueue_copy(data, buf)
        state[].ctx.synchronize()
        return 0
    except e:
        print("Model read:", e)
        return -1


@export
def android_model_read_async(
    state: StatePtr, index: Int32, data: Bytes, size: Int64
) abi("C") -> Int64:
    """Enqueue a device-to-host copy; the destination remains live until sync."""
    try:
        var buf = (
            state[].buffers[Int(index)].create_sub_buffer[.uint8](0, Int(size))
        )
        state[].ctx.enqueue_copy(data, buf)
        return 0
    except e:
        print("Model async read:", e)
        return -1


@export
def android_model_name(
    state: StatePtr, data: Bytes, capacity: Int64
) abi("C") -> Int64:
    try:
        var name = state[].ctx.name()
        if Int(capacity) <= name.byte_length():
            return -1
        for i in range(name.byte_length()):
            data[unsafe_offset=i] = name.unsafe_ptr()[unsafe_offset=i]
        data[unsafe_offset=name.byte_length()] = 0
        return 0
    except e:
        print("Model name:", e)
        return -1


@export
def android_model_forward(
    state: StatePtr, token: Int32, position: Int32, capacity: Int32
) abi("C") -> Int64:
    try:
        return forward(
            state[].ctx,
            state[].weights,
            state[].buffers,
            token,
            position,
            capacity,
            state[].head_gpu_columns,
        )
    except e:
        print("Model forward:", e)
        return -1


@export
def android_model_forward_with_head_input(
    state: StatePtr,
    token: Int32,
    position: Int32,
    capacity: Int32,
    data: Bytes,
) abi("C") -> Int64:
    """Read Bonsai's transformed FP16 activation before submitting its GPU head.

    The caller supplies 4096 writable bytes, valid until this call returns.
    The GPU head remains asynchronous so the caller can overlap Hexagon work.
    """
    try:
        return forward(
            state[].ctx,
            state[].weights,
            state[].buffers,
            token,
            position,
            capacity,
            state[].head_gpu_columns,
            data,
            False,
        )
    except e:
        print("Model forward with head input:", e)
        return -1


@export
def android_model_set_head_columns(state: StatePtr, columns: Int32) abi("C"):
    state[].head_gpu_columns = columns
