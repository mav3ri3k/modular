# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Direct GPU launch coverage for GraphStorageEngine tiles from opaque storage.

Mirrors the lifecycle of test_managed_tensor_slice_device_encoding.mojo:
device buffers are created, filled and consumed without an intervening
`synchronize()` between view construction and launch.
"""

from extensibility import IOSpec, ManagedTensorSlice, StaticTensorSpec
from extensibility.managed_tensor_slice import (
    GraphStorageEngine,
    _IndexListToTileLayout,
)
from layout import Coord, TensorLayout, TileTensor
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.memory import AddressSpace
from std.testing import assert_equal
from std.utils import IndexList


comptime GraphTile[LayoutType: TensorLayout] = TileTensor[
    .float32,
    LayoutType,
    MutUntrackedOrigin,
    Engine=GraphStorageEngine[element_width=1],
    address_space=.GENERIC,
]
"""A `TileTensor` backed by `GraphStorageEngine`."""


@inline(.always)
def _shifted[
    LayoutType: TensorLayout
](tile: GraphTile[LayoutType], offset: Int) -> GraphTile[LayoutType]:
    """Advances a graph-storage tile by scalar elements on the host."""
    return {tile._offset_storage(offset), tile.layout}


def _tile_fill[LayoutType: TensorLayout](output: GraphTile[LayoutType]):
    var row = global_idx.y
    var col = global_idx.x
    if row >= Int(output.dim[0]()) or col >= Int(output.dim[1]()):
        return
    output.store(Coord(row, col), Float32(100 * row + col))


def _tile_combine[
    SourceLayout: TensorLayout, OutputLayout: TensorLayout
](
    source: GraphTile[SourceLayout],
    output: GraphTile[OutputLayout],
    scale: Float32,
):
    var row = global_idx.y
    var col = global_idx.x
    if row >= Int(source.dim[0]()) or col >= Int(source.dim[1]()):
        return
    var idx = Coord(row, col)
    output.store(idx, scale * source.load(idx) + Float32(100 * row + col))


def _tile_ops[
    SourceLayout: TensorLayout, OutputLayout: TensorLayout
](source: GraphTile[SourceLayout], target: GraphTile[OutputLayout],):
    var output = target
    output.copy_from(source)
    output += source
    output *= source
    output /= source
    output -= source
    output.abs()
    output.min(source)
    output.max(source)


def _pointer_mark(output: MutPointer[Float32, MutAnyOrigin]):
    output[0] = Float32(-99)


def _borrowed_pointer_launch[LayoutType: TensorLayout](
    ctx: DeviceContext, tile: GraphTile[LayoutType]
) raises:
    # The wrapper drops on return. It must not release the borrowed allocation.
    var borrowed = tile.to_device_buffer(ctx)
    ctx.enqueue_function[_pointer_mark](borrowed, grid_dim=1, block_dim=1)


def _tile_mark[LayoutType: TensorLayout](output: GraphTile[LayoutType]):
    if global_idx.y == 0 and global_idx.x == 0:
        output.store(Coord(0, 0), Float32(1234))
        output.store(Coord(0, 1), Float32(2345))


def _run_case[
    InputLayout: TensorLayout, OutputLayout: TensorLayout
](
    ctx: DeviceContext,
    rows: Int,
    cols: Int,
    input_strides: IndexList[2],
    output_strides: IndexList[2],
) raises:
    comptime input_spec = StaticTensorSpec[.float32, 2, InputLayout](
        1, AddressSpace.GENERIC
    )
    comptime output_spec = StaticTensorSpec[.float32, 2, OutputLayout](
        1, AddressSpace.GENERIC
    )
    comptime Input = ManagedTensorSlice[
        io_spec=IOSpec.Input, static_spec=input_spec
    ]
    comptime Output = ManagedTensorSlice[
        io_spec=IOSpec.Output, static_spec=output_spec
    ]
    var input_size = (
        (rows - 1) * input_strides[0] + (cols - 1) * input_strides[1] + 3
    )
    var output_size = (
        (rows - 1) * output_strides[0] + (cols - 1) * output_strides[1] + 3
    )
    var input_host = ctx.enqueue_create_host_buffer[.float32](input_size)
    var expected = ctx.enqueue_create_host_buffer[.float32](output_size)
    ctx.synchronize()
    for i in range(input_size):
        input_host[i] = Float32(i) + 0.25
    for i in range(output_size):
        expected[i] = Float32(-1)
    var input_device = ctx.enqueue_create_buffer[.float32](input_size)
    var output_device = ctx.enqueue_create_buffer[.float32](output_size)
    ctx.enqueue_copy(input_device, input_host)
    output_device.enqueue_fill(Float32(-1))

    var source = Input(
        input_device.device_ptr() + 1, (rows, cols), input_strides
    )
    var output = Output(
        output_device.device_ptr() + 1, (rows, cols), output_strides
    )

    # The headline: building tiles on the HOST from opaque storage must not
    # unwrap a raw address, and shape queries must survive.
    var source_tile = source.to_tile_tensor()
    var output_tile = output.to_tile_tensor()
    assert_equal(Int(source_tile.dim[0]()), rows)
    assert_equal(Int(source_tile.dim[1]()), cols)
    assert_equal(Int(output_tile.dim[0]()), rows)
    assert_equal(Int(output_tile.dim[1]()), cols)

    print("case: single graph-storage tile launch")
    for row in range(rows):
        for col in range(cols):
            expected[
                1 + row * output_strides[0] + col * output_strides[1]
            ] = Float32(100 * row + col)
    ctx.enqueue_function[_tile_fill[Output.RuntimeLayout]](
        output_tile, grid_dim=1, block_dim=(16, 8)
    )
    with output_device.map_to_host() as actual:
        for i in range(output_size):
            assert_equal(actual[i], expected[i])

    print("case: graph-storage tile pair + scalar")
    output_device.enqueue_fill(Float32(-1))
    for i in range(output_size):
        expected[i] = Float32(-1)
    for row in range(rows):
        for col in range(cols):
            expected[
                1 + row * output_strides[0] + col * output_strides[1]
            ] = 2 * input_host[
                1 + row * input_strides[0] + col * input_strides[1]
            ] + Float32(
                100 * row + col
            )
    ctx.enqueue_function[
        _tile_combine[Input.RuntimeLayout, Output.RuntimeLayout]
    ](source_tile, output_tile, Float32(2), grid_dim=1, block_dim=(16, 8))
    with output_device.map_to_host() as actual:
        for i in range(output_size):
            assert_equal(actual[i], expected[i])

    print("case: borrowed DeviceBuffer pointer launch")
    _borrowed_pointer_launch(ctx, output_tile)
    expected[1] = Float32(-99)
    with output_device.map_to_host() as actual:
        for i in range(output_size):
            assert_equal(actual[i], expected[i])

    print("case: host-offset shifted tile")
    var shifted_tile = _shifted(_shifted(output_tile, 8), -4)
    expected[1 + 4] = 1234
    expected[1 + 4 + output_strides[1]] = 2345
    ctx.enqueue_function[_tile_mark[Output.RuntimeLayout]](
        shifted_tile, grid_dim=1, block_dim=(1, 1)
    )
    with output_device.map_to_host() as actual:
        for i in range(output_size):
            assert_equal(actual[i], expected[i])

    comptime if Input.RuntimeLayout.all_dims_known and Output.RuntimeLayout.all_dims_known:
        ctx.enqueue_function[
            _tile_ops[Input.RuntimeLayout, Output.RuntimeLayout]
        ](source_tile, output_tile, grid_dim=1, block_dim=1)
        with output_device.map_to_host() as actual:
            for row in range(rows):
                for col in range(cols):
                    assert_equal(
                        actual[
                            1
                            + row * output_strides[0]
                            + col * output_strides[1]
                        ],
                        input_host[
                            1 + row * input_strides[0] + col * input_strides[1]
                        ],
                    )

    ctx.synchronize()
    _ = input_device^
    _ = output_device^


def main() raises:
    comptime StaticInput = _IndexListToTileLayout[
        IndexList[2](3, 5), IndexList[2](17, 2)
    ]
    comptime StaticOutput = _IndexListToTileLayout[
        IndexList[2](3, 5), IndexList[2](19, 3)
    ]
    comptime Dynamic = _IndexListToTileLayout[
        IndexList[2](-1, -1), IndexList[2](-1, -1)
    ]
    with DeviceContext() as ctx:
        _run_case[StaticInput, StaticOutput](ctx, 3, 5, (17, 2), (19, 3))
        _run_case[Dynamic, Dynamic](ctx, 7, 13, (31, 2), (41, 3))
