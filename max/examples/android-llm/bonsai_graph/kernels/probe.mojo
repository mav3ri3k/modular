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
"""Minimal custom op proving the Bonsai kernel-library pipeline end to end."""

from max.algorithm.functional import elementwise
from extensibility import InputTensor, OutputTensor, register
from max.gpu.host import DeviceContext
from layout import Coord


@register("bonsai_probe_scale")
struct ProbeScale:
    """Doubles a rank-1 FP32 tensor through a device kernel launch."""

    @staticmethod
    def execute(
        outp: OutputTensor[dtype=.float32, rank=1, ...],
        x: InputTensor[dtype=.float32, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        var source = x.to_tile_tensor()
        var dest = outp.to_tile_tensor()
        var count = Int(source.dim[0]())

        def scale[
            width: Int, alignment: Int = 1
        ](idx: Coord) {var source, var dest}:
            dest.store(idx, source.load(idx) * Float32(2))

        elementwise[1](scale, Coord(count), ctx)
