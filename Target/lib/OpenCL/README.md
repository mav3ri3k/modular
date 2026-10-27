# Experimental Android OpenCL HAL backend

This is the first runtime slice of an Android backend for Modular. It implements
part of the **existing MAX vendor C HAL** in `SDK/include/HAL`, rather than
wrapping a separate inference engine. The initial target is Android ARM64 with a
vendor OpenCL 1.2 implementation. The library discovers GPU devices only.

## Verified on hardware

On 2026-10-24, an SM-S942U1 reported Qualcomm SM8850, Android API 37 and
`QUALCOMM Adreno(TM) 840`. Using NDK r29 (`29.0.14206865`), the release build:

- Resolved every symbol in `M_driver_hal_symbols.inc` through `dlopen`/`dlsym`.
- Created two contexts and two in-order queues.
- Allocated device buffers, copied data and compiled an OpenCL source bundle.
- Executed FP32 matrix multiplication at six shapes, including nonmultiples of
  the workgroup size and an empty reduction; compared against a CPU reference.
- Exercised cross-queue event waits, host event synchronization, re-recording,
  fills, device-to-device copies and offset host/device copies.
- Rejected invalid buffers, cross-context frees, incomplete argument lists,
  incompatible bundles, invalid fill sizes and unsupported execution modes.
- Returned the device compiler log for intentionally malformed source.

The Mojo compiler target emits ABI v3 bundles for normal DeviceContext launches. The
native compiler artifact also passed the matrix tests, cyclic SSA/UInt16 edge
cases and a tiny two-layer dense/ReLU model written in Mojo. See
[`Mojo/tools/mojo-opencl/README.md`](../../../Mojo/tools/mojo-opencl/README.md).
The model uses synthetic weights and GPU-resident intermediate activations.
Subsequent phone tests pass ordinary MAX transformer operators, wide reductions
and a full real-weight SmolLM2 prefill/decode through the normal MEF executor.
See [integration evidence](../../../docs/android-feasibility/INTEGRATION.md) and
[Bonsai performance](../../../max/examples/android-llm/bonsai/PERFORMANCE.md).

## Supported interface and ABI

Implemented: version negotiation; device enumeration/properties; context
creation; synchronous device and pinned host allocation/free; opaque device
handles and mapped host addresses; in-order queues;
blocking host transfers; intra-device copies; byte/pattern fills; source bundle
compilation; kernel loading/dispatch; and events.

Other SDK entry points are generated from the actual headers and return
`M_DRIVER_STATUS_UNIMPLEMENTED`. They never pretend to have completed work.
`generate_unsupported.py` keeps their signatures consistent with the current SDK.

A raw bundle must have file type `opencl-source` and carry OpenCL C 1.2 source bytes.
Zero optimization uses `-cl-opt-disable`; optimization levels 1–3 use the vendor
compiler defaults. Debug profiles are unsupported. GPU grid dimensions count
workgroups, so the global range is `grid * block` in each dimension.

Raw source bundles use **OpenCL's ABI**. Buffer argument bytes contain a
`cl_mem` handle obtained from the memory `address` property. This value is opaque:
it is neither a GPU virtual address nor a host pointer, and pointer arithmetic on
it is invalid. Copy views support byte offsets; compiled ABI v3 below also
supports offset kernel arguments. Scalar arguments use their OpenCL C widths.
Raw-source dynamic shared memory and launch attributes are unsupported. All kernel
arguments must be supplied on every launch; argument mutation/dispatch is locked
per function. Separate function objects may execute concurrently.

Normal compiled Mojo bundles use file type `mojo-opencl-v3`. Their checked
manifest records logical argument packets, typed leaf offsets, original/native
symbols and capability requirements. Pointer leaves carry SDK allocation views
expanded to native `cl_mem`/offset parameters; scalar leaves keep recorded
widths. Nested aggregates and captured tensor arguments preserve provenance.
The decoder rejects malformed/truncated manifests, invalid field bounds and
duplicate symbols. The device must have 64-bit, little-endian storage.
A canonical empty view `{nullptr, 0, 0}` binds an OpenCL null pointer for
optional arguments and type-only fusion views. Null handles with nonzero offsets
or extents are rejected. Compiler pointer leaves retain nullable semantics;
non-null views are still checked against live allocations and their bounds.

Callers must destroy child resources before their context and destroy contexts
before the driver. Normal in-order queues provide execution ordering; using a
separate queue or the synchronous context-copy functions requires explicit
synchronization. Host transfers currently block to keep staging-pointer lifetime
well defined. Event record flushes its producer queue for cross-queue progress.

## Build and run

The repository targets provide a host compile check:

```sh
./bazelw build //Target:OpenCLPlugin.so //Target:OpenCLPluginSmoke
```

The host build passed on the development Mac with
`--linkopt=-fuse-ld=/usr/bin/ld`. Its macOS 27 SDK contains TAPI target names
that the repository's bundled LLD cannot parse; using Apple's linker avoids
that local SDK/linker incompatibility. This flag is not used for Android.

The standalone CMake build follows the independently buildable vendor-plugin
model described in `Target/README.md`. It cross-compiles against Android's
Bionic libraries without first porting the entire repository toolchain.

Install NDK r29 using the Android CLI, and fetch the pinned Khronos headers:

```sh
android sdk install 'ndk/29.0.14206865'
git clone https://github.com/KhronosGroup/OpenCL-Headers.git /tmp/opencl-headers
git -C /tmp/opencl-headers checkout 82f60316cc142e7b21dc1d8983b26de18858fbd9
```

From the repository root, supply the NDK path and the explicit ADB device ID:

```sh
python3 Target/lib/OpenCL/run_android.py \
  --ndk /path/to/Android/sdk/ndk/29.0.14206865 \
  --opencl-headers /tmp/opencl-headers \
  --build-dir /tmp/modular-opencl-build \
  --serial YOUR_ADB_DEVICE_ID
```

Add `--view-bundle /tmp/buffer_views.mcl` to exercise checked ABI-v3 packets.
Compile `Mojo/test/mojo-opencl/buffer_views.mojo` using the compiler helper
described in the linked compiler README.

The runner builds with warnings as errors, pushes only the plugin and test
executable to a unique `/data/local/tmp/` directory, runs the tests, and removes
that directory. It does not install an APK or change system settings. OpenCL is
loaded dynamically from the device; no vendor library is copied into the repo.
The separate Mojo Chat APK also passes packaged JNI and streaming UI tests.
Its plugin packaging declares the required Android native-library visibility.
Android TCMalloc uses its existing non-RSEQ fallback for app seccomp compatibility.

## Remaining integration

Ordinary Mojo DeviceContext and normal MAX graphs use SDK plugin discovery,
opaque allocation/view ownership, standard tensor engines and the shared
compiler/runtime. Tested phone coverage includes transformer operators and
real-weight 30-layer SmolLM2 prefill/decode with phone-produced KV caches.
The [Bonsai MAX graph](../../../max/examples/android-llm/bonsai_graph/README.md)
also runs all 24 text layers with ordinary ops and registered custom kernels;
its first paired decode comparison is slightly slower than the Mojo runner.

Remaining work is broader operator/vendor coverage and optimization. Dynamic
stack allocation, arbitrary indirect/external calls, fp64/half atomics and safe
assertion termination across barriers remain unsupported. Device graph capture
is an optional HAL feature that is not implemented; broader tests requiring it
cannot run. Existing vendor-tuned kernels are not assumed portable unchanged.

## Ordinary Mojo DeviceContext

The SDK plugin can now be loaded by the existing Mojo HAL implementation through
`MODULAR_DRIVER_PLUGINS`. The `--config=hal-device-context` overlay and MAX alias
are wired in the public Bazel configuration. No `device_get_native` C++ bridge
or example-specific `android_launch` function is used by this path.

With a source-built compiler containing the OpenCL target, run:

```sh
python3 Target/lib/OpenCL/run_device_context.py \
  --mojo bazel-bin/Mojo/tools/mojo/mojo \
  --ndk /path/to/Android/sdk/ndk/29.0.14206865 \
  --build-dir /tmp/modular-hal-context \
  --serial YOUR_ADB_DEVICE_ID --target-accelerator adreno-840
```

Omit `--target-accelerator` for the conservative `opencl-generic` target. The
runner stages the existing HAL overlay with Bazel, packages matching std/MAX
sources, builds the Android host object with embedded GPU artifacts, and links
the normal `//Mojo:CompilerRT` runtime.
[`build_runtime.py`](build_runtime.py) uses `--config=android-arm64` to build
that runtime and both SDK plugins, and stages their libraries in
`android_artifacts.json`. Android API 33+ is required.
It tests normal allocation/copies, nested offset views and pointer comparisons,
SIMD/FP16/math, static/dynamic local memory and barriers, integer atomics and
(on Adreno) standard subgroup shuffles. Temporary device files are removed.

ABI v3 preserves allocation identity and byte offsets until dispatch. Kernel
symbols, typed field bounds, argument widths, optional extensions, dynamic local
memory and subgroup widths are validated. See the
[compiler ABI and limits](../../../Mojo/tools/mojo-opencl/README.md#ordinary-mojo-host-launches-abi-v3).
Only v3 artifacts are accepted; v1/v2 support has been removed.

The compiler's HAL discovery reads API/architecture properties rather than
assuming CUDA compute capability. C++ compile-only OpenCL devices also retain
the selected architecture without loading an Android plugin on the desktop;
vendor OpenCL compilation occurs on the target when the HAL loads the bundle.

Both model examples now use ordinary Mojo launches and owning typed buffers.
Graph buffer allocation, nested slicing and asynchronous ownership also carry
opaque metadata. Typed tensor arguments and closures, plus standard pinned
HostBuffer transfers, pass on the phone. Core AsyncRT tests and a real arithmetic
MAX graph pass on Android. The graph uses ordinary add/multiply operators,
outlined typed-storage stubs, explicit CPU/GPU transfers and the normal MEF
executor. Ordinary TileTensor construction uses GraphStorageEngine and keeps
opaque storage until device argument encoding. Paged-cache layouts, fused
residual/RMSNorm, BMM and public GPU tensor I/O also pass numeric phone tests.
Broader operator/model coverage remains incremental.

To reproduce graph execution, first compile the graph on the build machine:

```sh
./bazelw run -c opt --config=hal-device-context \
  //GraphCompiler/tools/mo-opt:mo-opt -- Target/test/OpenCLGraph.mlir \
  '--virtual-cpu-target=triple=aarch64-linux-android33;cpu=generic' \
  --virtual-device-count=1 --virtual-device-api=opencl \
  --virtual-device-arch=adreno-840 \
  '--gc-pipeline=constant-fold=false default-device-label=gpu default-device-target-triple=aarch64-unknown-opencl default-device-target-arch=adreno-840' \
  -o /tmp/opencl-graph-lowered.mlir
./bazelw run -c opt --config=hal-device-context \
  //GraphCompiler/tools/mt:mt -- --compile /tmp/opencl-graph-lowered.mlir \
  '--virtual-cpu-target=triple=aarch64-linux-android33;cpu=generic' \
  --virtual-device-count=1 --virtual-device-api=opencl \
  --virtual-device-arch=adreno-840 -o /tmp/opencl-graph.mef
python3 Target/lib/OpenCL/build_runtime.py \
  --ndk /path/to/Android/sdk/ndk/29.0.14206865 \
  --output /tmp/android-max --include-max-tool
python3 Target/lib/OpenCL/run_graph.py \
  --runtime /tmp/android-max --model /tmp/opencl-graph.mef \
  --serial YOUR_ADB_DEVICE_ID
```

The outputs are six `2`s and six `4`s. The deployment helper stages the normal
`mt` executable, graph/runtime libraries and SDK plugins, sets their standard
discovery paths, and removes its temporary device directory. It does not supply
a separate graph executor. Native and Android Bazel configurations share output
paths, so finish host compilation before staging the Android runtime.

The Snapdragon guide in
[llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/README.md)
is relevant reference material: it builds both Adreno/OpenCL and Hexagon backends.
Its [Hexagon build](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/CMakeLists.txt)
generates RPC interfaces and separate DSP libraries for several architecture
versions. No llama.cpp implementation code is incorporated here.

## Optional GPU profiling

`MODULAR_OPENCL_PROFILE=1` enables OpenCL command-queue profiling. Kernel event
START/END timestamps are aggregated by kernel name and printed to stderr when
the queue synchronizes or is destroyed. Reports include the sampled device-time
span, total kernel duration, missing timestamps and dropped samples. At most
100,000 events are retained per queue between reports. The default path does
not collect events; profiling runs are diagnostic and may affect throughput.
