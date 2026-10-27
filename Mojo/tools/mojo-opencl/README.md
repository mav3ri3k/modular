# Experimental Mojo OpenCL compiler target

The target compiles Mojo GPU kernels into a versioned artifact consumed by the
Android OpenCL HAL plugin. Kernel authors use Mojo, including
`max.gpu.global_idx`, `Pointer`, scalar arithmetic, control flow and standard
functions such as `max`. There is no user-authored OpenCL kernel in this path.
The backend generates OpenCL C internally because the tested Adreno driver
reports no supported intermediate-language format.

The target also runs [pretrained SmolLM2-135M and Bonsai 2 2B](../../../max/examples/android-llm/README.md)
on the S26, with model orchestration and mathematical kernels written in Mojo.
Scalar FP32 standard math hooks include exp/exp2/log/log2/sqrt/rsqrt/sin/cos.
The standard assertion hook retains checks and aborts but skips message
formatting, because the target has no diagnostic text sink. Failed checks set
a queue-owned device error flag and return; HAL events, blocking transfers and
queue synchronization report the failure. The compiler passes the hidden flag
through helpers without changing logical kernel arguments. Kernels combining
live failures with workgroup barriers are rejected until safe workgroup
termination is supported.

## Verified vertical slice

The source-built compiler accepts:

```sh
mojo build kernels.mojo --target-triple aarch64-unknown-opencl \
  --target-cpu generic --emit object -o kernels.mcl
```

It emits a `MOJOCL03` bundle directly through `TargetBackend` and
`ObjectCompiler`. `--emit asm` emits generated `.cl` for inspection; `--emit llvm`
exposes the target IR. Executable/shared-library emission is deliberately not
advertised for this device-only target.

The ordinary DeviceContext phone suite covers typed aggregates and closures,
SIMD/FP16, shared memory, barriers, atomics, subgroup operations and stream/event
ordering. Both model examples use embedded v3 artifacts and reference-checked
outputs. Compiler regression tests reject unsupported IR without writing an
artifact.

## Build and reproduce

Build the compiler and emitter from the repository root:

```sh
./bazelw build --compilation_mode=opt \
  //Mojo/tools/mojo:mojo //Mojo/tools/mojo-opencl:mojo-opencl-emit
./bazelw test --compilation_mode=opt \
  //Mojo/tools/mojo-opencl:emitter_rejections_test
```

The development Mac has an SDK 27 / bundled LLD incompatibility. The full source
build was verified using an isolated local SDK repository pointing at its
installed macOS 26.5 SDK, passed via
`--override_repository=sysroot-macos=/tmp/modular-macos26-sdk`. This is a local
build workaround, not an Android requirement. An Apple-linker override alone
cannot link the full compiler's `--start-lib` groups.

The helper packages the current stdlib with the selected compiler, ensuring the
new OpenCL hooks and serialized package version agree:

```sh
python3 Mojo/tools/mojo-opencl/compile.py \
  Mojo/test/mojo-opencl/kernels.mojo \
  --mojo bazel-bin/Mojo/tools/mojo/mojo \
  --import-path max/mojo \
  -o /tmp/kernels.mcl
```

For an end-to-end ordinary host launch, use
[run_device_context.py](../../../Target/lib/OpenCL/README.md#ordinary-mojo-devicecontext).
`run_android.py --view-bundle /tmp/kernels.mcl` checks the standalone buffer-view
fixture; its raw-source SDK smoke test is separate from Mojo compilation.

The standalone emitter is a test/debug front end to the same emitter library;
normal compilation uses this checkout's compiler. There is no bootstrap or
historical-ABI build path:

```sh
cmake -S Mojo/tools/mojo-opencl -B /tmp/mojo-opencl-compiler \
  -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/mojo-opencl-compiler
ctest --test-dir /tmp/mojo-opencl-compiler --output-on-failure
```

## Compiler structure

- `Mojo/lib/Target/OpenCL`: native target traits and `.mcl`/`.cl` emission kinds.
- `Mojo/lib/KGENToLLVM/Target/OpenCL`: SPIR kernel calling-convention marking.
- `Mojo/stdlib/std/_plugin/opencl`: work-item hooks used by the standard GPU API.
- `Mojo/lib/Compiler/ObjectCompiler/Target/OpenCL`: checked LLVM-to-OpenCL
  emission and the native `TargetBackend` implementation.
- `Target/include/Target/OpenCLBundle.h`: shared, versioned bundle/argument ABI.
- `Target/lib/OpenCL`: Android runtime plugin and ABI validation at dispatch.

The initial target reuses AArch64's 64-bit little-endian scalar data model. It is
classified as a GPU, is excluded from host target matching, and never emits ARM
machine instructions for device kernels. `is_cpu()` excludes accelerator plugins
using an otherwise recognized CPU data model. This target spelling does not
indicate that the GPU executes the AArch64 instruction set.

The generic compiler addition is an opt-in
`TargetBackend::usesModuleEmitterForArchive()` hook. Non-machine-code backends
can emit a standalone LLVM module through their normal artifact hooks without
routing it through MC assembly/linking. Existing backends default to the prior
route. This is distinct from the preexisting per-offload-kernel emission path.

SSA values are explicit private variables; branch edges perform parallel phi
copies before jumping to the destination. GEP offsets use 64-bit modular address
arithmetic, preserving signed indices and avoiding speculative out-of-bounds C
pointer arithmetic. Narrow integer operations avoid C's signed integer promotion
rules. FP32 constants preserve their bit patterns.

## Ordinary Mojo host launches (ABI v3)

Mojo HAL `DeviceContext` runs on Android through `MODULAR_DRIVER_PLUGINS`,
including normal `enqueue_function` compilation and embedded device artifacts.
Both Android LLM examples now use this path, including typed buffer ownership.
See the [host integration test](../../../Target/lib/OpenCL/README.md#ordinary-mojo-devicecontext).

The `opencl-generic` and `adreno-840` target collections select v3. Aggregate
arguments retain their typed pointer provenance before host C ABI coercion.
Each logical argument carries a scalar payload and a sparse sidecar of SDK
allocation/offset/extent descriptors. The manifest records typed leaf offsets;
OpenCL kernels receive flattened scalar and buffer/offset parameters. Nested
aggregates, borrowed aggregates and captured TileTensor arguments use the same
encoder. No opaque handle is presented as a raw device pointer.

Original Mojo symbols are distinct from native OpenCL names. Required
extensions, OpenCL C version, argument widths, dynamic local memory and subgroup
widths are recorded and checked. The format is experimental: rebuild compiler
and plugin together. Only v3 exists; historical formats and switches are removed.

## Current supported subset and limits

FP16/FP32 and integer vectors support OpenCL-native widths 2, 3, 4, 8 and 16,
arithmetic/conversions/comparisons/selects/shuffles, reductions, scalar-aligned
vector memory, math hooks, static/dynamic workgroup memory and barriers. FP16
requires `cl_khr_fp16`. Adreno subgroup launches validate native 64-lane width
and complete subgroups, including subgroup memory barriers; generic OpenCL
makes no fixed-width promise.

Integer32 atomics include RMW, load/store, fences and strong/weak compare-exchange
with old-value/success results. FP32 load/store and add/sub/exchange/min/max use
integer bit transport and CAS loops. Integer64 atomics require both Khronos
int64 atomic extensions. The attached Adreno840 lacks those extensions and
rejects such artifacts explicitly. Atomics require OpenCL C3.0 and the relevant
ordering/scope features. LLVM orderings are preserved. System scope maps to
device scope under the current non-SVM, non-peer contract.

Fixed entry-block private allocations retain private address-space provenance
through LLVM InferAddressSpaces, including dynamic indexing. Byte copy/move/set
intrinsics and additional rounding, bit-counting and math intrinsics are
supported. Numeric unaligned scalar/vector memory uses byte-vector transfers;
LLVM switches preserve control-flow edges and phi values. Direct defined
helpers remain supported subject to representable pointer address spaces. Numeric scalar-field SSA records preserve LLVM padding
and alignment, including helper returns, insert/extract operations and joins.
Unsupported operations fail compilation with no CPU fallback. Dynamic stack allocations, fp64/half atomics, packed boolean-vector
memory, arbitrary external/indirect calls and device sanitizers remain outside
the supported subset. Exact 64-bit pointer qualification round trips across
OpenCL global spaces are normalized; arbitrary integer/pointer casts are rejected.

MAX graph buffer/tensor ownership, outlined host stubs and tensor/fusion
captures preserve opaque views. An ordinary MAX arithmetic graph cross-compiles
through the graph pipeline into an Android/OpenCL MEF. Phone tests validate
normal MAX transformer operators and a full real-weight 30-layer SmolLM2
prefill/decode, including phone-produced KV caches. The standard MAX elementwise
algorithm passes phone tests with non-unit strides and SIMD row tails.
Existing vendor-tuned kernels cannot be assumed to compile unchanged.

The backend runs the shared LLVM optimization pipeline. It uses generic target
analysis rather than the borrowed AArch64 ISA cost model, and marks collective
calls convergent before optimization. LLVM's masked-memory legalization uses
work-item divergence information; SROA removes aggregate joins and target
legalization handles common vectorized pointer arithmetic without scalarizing
numeric SIMD. Functions containing values wider than OpenCL C's 16-lane limit
use LLVM's standard scalarizer; functions with native-width SIMD are left intact.
Workgroup-size metadata comes through the MAX plugin hook and
lowers to OpenCL's standard work_group_size_hint. Strict SIMD fma remains correctly rounded;
explicit `FastMathFlag.AFN` permits OpenCL mad lowering. Bonsai uses the latter
through the standard SIMD API, preserving its approximate packed FP16 contract.
