# Mojo inference projections on Snapdragon Hexagon

The experimental projection backend runs Bonsai's vocabulary matrix on either
**HVX (batch-one packed GEMV)** or **HMX (matrix tiles)**. Both use resident W2
weights and Mojo mathematical kernels. C/C++ handles transport, workers, power
and memory; no QNN or other inference engine executes the model.

Select the implementation at build time with `--backend hvx` or `--backend hmx`
(the historical default). HVX keeps weights packed, prepares a small activation
lookup table once per token, and computes 128 output rows per tile using vector
lookups and QFloat arithmetic. It avoids HMX Crouton weight expansion and
replicated activation rows. HVX widths must be multiples of 128.

The engine can split vocabulary rows between Adreno and Hexagon. It reads the
transformed activation before submitting the GPU head, then waits only for the
activation-copy event. Hexagon can execute while the GPU head runs; it no longer
waits for that head through the original queue-tail activation readback. Other
model operations stay on Adreno. This remains an operator backend, not a full
MAX Hexagon HAL plugin or native Mojo Hexagon target. The NPU route is opt-in in
the direct CLI; the ordinary MAX graph currently uses the GPU.

Current measurements and experiment records are in
[the decode experiment report](../../../docs/android-feasibility/hexagon-decode/README.md).
GPU-only remains the default pending repeatable benefit from the hybrid route.
The historical HMX results below are from an older, substantially slower GPU
build and must not be compared directly with current throughput.

## Historical HMX measurements on the S26

SM-S942U1 / SM8850, Android API 37; identical 21-token prompt, capacity 128,
96 requested output tokens / 95 decode forwards, fast math enabled:

| Route | Decode tok/s |
| --- | ---: |
| GPU-only | 41.45 |
| Entire vocabulary head on HMX | 29.09 |
| GPU + HMX head, 32,768 rows on HMX | 40.55 |

These are individual hill-climbing samples, not sustained medians. **There is no
demonstrated end-to-end NPU speedup yet.** The demo remains on the faster GPU
route; its latest packaged JNI France smoke test measured 41.21 tok/s.

Validation:

- Real packed checkpoint weights, 6,144×2,048 projector, synthetic half-exact
  activations: all 6,144 outputs match CPU reference, maximum error zero.
- Whole-head HMX generation: eight greedy IDs match the FP32 reference;
  maximum logit error 0.0100434, relative L2 0.000461039.
- Final concurrent GPU/HMX generation: eight greedy IDs match;
  maximum logit error 0.01056, relative L2 0.000510570.
- Earlier scalar Mojo ABI probe: 257 exact results; nonuniform FP16 HMX
  32×32×32 probe: all 1,024 outputs match.

The initial projector spent about 7.5 ms unpacking the 6,144-row test matrix.
Blocked packing and L2 prefetch brought that to about 1.13 ms; eight HVX workers
brought it to about 0.25 ms. HMX multiply itself was about 0.08 ms. Conversion,
memory access and cross-device synchronization remain important at batch one.
The original prototype used two queues. The current ordinary DeviceContext
implementation uses an activation-copy event to overlap the head portions.

## Build and run

Dependencies: source-built Mojo, matching precompiled stdlib, LLVM 22 with
Hexagon enabled (`opt` beside `llc`), Zig 0.15.2, NDK 29 and externally installed
Hexagon SDK Community REDACTED_IP headers. SDK headers/binaries are not vendored.

Build the projector and the normal Bonsai runner:

```bash
python Target/lib/Hexagon/build.py --backend hvx \
  --mojo bazel-bin/Mojo/tools/mojo/mojo \
  --stdlib-dir /path/containing/std.mojoc \
  --llc /path/to/llvm22/bin/llc --zig /path/to/zig \
  --ndk /path/to/android/ndk/29.0.14206865 \
  --sdk /path/to/hexagon-sdk/REDACTED_IP --output /tmp/modular-hexagon-build

python max/examples/android-llm/build.py --variant bonsai \
  --mojo bazel-bin/Mojo/tools/mojo/mojo \
  --ndk /path/to/android/ndk/29.0.14206865 \
  --opencl-headers /path/to/OpenCL-Headers --output /tmp/modular-bonsai-build
```

`--hvx-workers` selects 1/2/4/8 vector workers (default 8 on this phone).
The HMX backend additionally uses a persistent matrix worker that retains its
unit assignment for a projection and yields its critical-section flag between
tile batches. Its VTCM is acquired once per projection. Both backends retain
mapped packed weights across tokens.

Provision the regular runner, GPU bundle and exported model using the
[Bonsai instructions](../../../max/examples/android-llm/bonsai/README.md).
Also install the two NPU components:

```bash
adb shell mkdir -p /data/local/tmp/modular-hexagon
adb push /tmp/modular-hexagon-build/HexagonProjection.so \
  /tmp/modular-hexagon-build/libmodular_hexagon.so \
  /tmp/modular-hexagon-build/projection_bench \
  /data/local/tmp/modular-hexagon/
```

From the runner/model directory on the phone, enable the operator route:

```bash
export ADSP_LIBRARY_PATH='/data/local/tmp/modular-hexagon;/vendor/lib64/rfs/dsp;/vendor/lib/rfsa/adsp;/vendor/dsp'
export MODULAR_OPENCL_FAST_MATH=1
export MODULAR_HEXAGON_HEAD=/data/local/tmp/modular-hexagon/HexagonProjection.so
# Omit this to put the entire head on Hexagon; set a multiple of 128 for a split.
export MODULAR_HEXAGON_HEAD_COLUMNS=16384
# Optional process-local FastRPC PM QoS vote, removed at session destruction.
export MODULAR_HEXAGON_QOS=1
./runner ./OpenCLPlugin.so ./weights.bin ./prompt.txt benchmark 96
```

Unset `MODULAR_HEXAGON_HEAD` to restore GPU-only routing. An explicitly requested
NPU route fails on initialization/execution errors; it does not silently run the
selected Hexagon rows on the GPU. `MODULAR_HEXAGON_PROFILE=1` prints DSP timings
(HVX/lookup or HMX stages).
It makes an extra RPC and must be disabled for throughput comparisons.
`MODULAR_HEXAGON_TIMING=1` prints host forward/readback and RPC durations.
The projector benchmark accepts
`projection_bench HexagonProjection.so weights.bin 6144` for an independently
reference-checked matrix test.

## Runtime and compiler details

The C ABI in `Target/include/Target/HexagonProjection.h` supports one active
projection session per process. The existing engine currently registers the
vocabulary matrix (or its selected rows). Workers and shared mappings are
retained across tokens and released at destruction. Process-local performance
votes are removed at session shutdown. This prototype is not a concurrent
multi-model service.

**v81 resource detail:** the legacy `HAP_compute_res_attr_set_hmx_param` request
must be 0 for this autonomous allocation path. Setting it to 1 allows context
creation but causes subsequent lock3/lock4 calls to fail. The executed matrix
uses lock4 and the returned volatile `hmx_busy` flag.

Mojo lacks native Hexagon TargetTraits. The build emits 32-bit LLVM IR through
the riscv32 frontend, verifies its data layout, removes RISC-V CPU/features and
applies Hexagon's layout before LLVM optimization/code generation. Only the
explicitly tested pointer/scalar C ABI, integer SIMD and selected Hexagon
intrinsics are supported by this bridge. It rejects RISC-V-specific intrinsics.
It also translates LLVM 23's native-width FP32 literals for LLVM 22.

HVX uses the tested v73/QFloat compatibility instruction path on this v81 phone.
The current reduction flushes FP16 subnormal group partials to zero. Results are
approximate; the small reference fixtures do not establish general quality or
long-context equivalence.

Upstream LLVM 22 does not assemble HMX mnemonics. `hmx-encodings.s` records the
sequence verified by Qualcomm LLVM-MC 21 with
`-triple=hexagon -mcpu=hexagonv81 -mattr=+hmx -show-encoding`. Mojo emits those
words with fixed register constraints. Normal builds need no Qualcomm assembler
or Linux VM. The layout/instructions follow the
[Qualcomm v81 HMX reference](https://docs.qualcomm.com/doc/80-N2040-62/80-N2040-62_REV_AA_Qualcomm_Hexagon_V81_HMX_Programmers_Reference_Manual.pdf).

Further work: lower-cost packed conversion, activation memory sharing across
GPU/DSP, more overlap, and batched prefill kernels. Native target registration
and resident DSP state/attention are needed to move the remaining model onto
Hexagon without per-operator FastRPC transfers.
