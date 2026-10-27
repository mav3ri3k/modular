# Android backend implementation ledger

Updated 2026-10-25. Working context for implementation continuity. User authorizes
implementation and attached-phone tests. Preserve the large preexisting staged
prototype; do not reset the index or commit unless requested.

Checkpoint commit created 2026-10-24 at ~80% integration: canonical stack fully
piped (aggregate ABI v3, opaque allocations, typed stub storage, android-arm64
platform, arithmetic graph E2E, model numerics). Remaining before profiling:
MTS→TileTensor raw-lane migration and full model-graph correctness.

## Priorities and decisions

- Finish storage integration and full MAX graph correctness first, then update
  /Users/REDACTED/projects/mojo-chat. User now authorizes Bonsai profiling and
  kernel optimization AFTER both finish, followed by a second app update.
  100 tok/s is an aspirational target, not a claimed result. Use shared compiler
  optimization. Final demos only require app packaging; profile via CLI.
- Bonsai model intrinsics use standard SIMD casts/bitcasts, fma, reductions and
  subgroup APIs. AFN is an explicit approximation contract, not a device call.
- ABI v1/v2 decoder/emitter/switches/bootstrap paths and compatibility tests are
  deleted. Both models use v3. Raw OpenCL-source SDK support is separate.
- Use normal Mojo DeviceContext, SDK HAL and normal CompilerRT/AsyncRT. The
  duplicate minimal Mojo/lib/CompilerRT/Android runtime has been deleted.
- MTS→TileTensor raw lane root cause (2026-10-24): the removed compact projection
  set MTS.device_type to TileTensor, but closure bodies are compiled against the
  DECLARED captured type, so projected 2-field fusion-less bytes mismatched the
  capture-slot layout and dropped fusion fields. Identity layout with
  pointer-leaf translation is mandatory. Fix direction: a graph-storage
  TensorEngine whose StorageType is _TensorDataPointer (already DevicePassable +
  DevicePointerLike, proven in on-phone fusion functor captures), following the
  DevicePointerEngine/StableAddrEngine precedents; NOT a sidecar ABI. Needs
  checked signed unsafe_offset for reverse/chained views. Engine-generic tile
  signatures already exist (linalg/lora.mojo).
- Complete real MAX graph execution. Allocator tests and scalar GraphRT MEF
  execution do not establish MAX operator/model support.
- No backend conditionals inside graph compiler passes; preserve MO buffer
  nonaliasing and MEF as the GraphRT boundary. Read GraphCompiler/AGENTS.md.

## Ordered audit status

1. DONE aggregate ABI v3: preserve pointers before host C ABI coercion; flatten
   typed nested/byval arguments. Tuple/capture/sliced-view launches pass phone.
2. DONE ordinary Mojo launches: MODULAR_DRIVER_PLUGINS, embedded artifacts,
   DeviceBuffer ownership, enqueue/copy/fill, streams/events, shared memory,
   barriers/subgroups and supported atomics. No android_launch/example bridge.
3. PARTIAL target integration: OpenCL/Adreno compiler targets, physical/virtual/
   compiler-only device discovery and HAL arch naming work. Audit graph target
   propagation and capability consumers during actual graph execution.
4. DONE initial emitter coverage: SIMD, shared/local memory, static private
   alloca, memory intrinsics, direct helpers, CAS, FP32 CAS-loop atomics, gated
   i64 atomics and math/bit intrinsics. Additional correctness coverage below.
5. DONE opaque graph allocation ownership: TensorBufferRef, OwnedByteBuffer and
   OwnedTensor preserve allocation/offset/extent/shape/owner through nested
   slices, pack/unpack, extraction and transfers. No fabricated raw addresses.
6. PARTIAL Android runtime/build: actual CompilerRT/AsyncRT and GraphRT
   execute on phone. Canonical Android runtime/plugins pass phone validation.
   Normal MAX mt arithmetic MEF execution passes; broader graph/model coverage
   and APK/JNI namespace validation remain.
7. PARTIAL MAX graphs: typed ManagedTensorSlice explicit args and closures work.
   Fusion capture and standard elementwise launches pass phone. Host TileTensor/raw consumers,
   variadics and some slice/trait wrappers still need integration.
8. DONE model migration to ordinary DeviceContext: Bonsai/Smol orchestration and
   kernels in Mojo; C++ owns checkpoint I/O/sampling. GPU and optional Hexagon
   numerics pass. These models are not yet MAX graphs.

## ABI and memory invariants

- cl_mem is opaque. SDK allocation handles, never GPU addresses, cross host code.
  DevicePointerLike exposes optional (allocation, byte_offset, remaining_size).
- ABI3 packet P = align_up(device payload size, 8), total size4P. Pointer field
  at original byte offset N uses a 24-byte SDK view descriptor at P+3N; scalar
  fields retain their original offsets. TargetLowering::usesHostCABI=false.
- DevicePassable encoding uses target data layout. ManagedTensorSlice and
  _TensorDataPointer have identity device_type: closure bodies retain the
  declared struct layout. A compact MTS→TileTensor projection failed closures
  and was removed. Actual typed pointer field is encoded; host-only metadata
  is zeroed on device. No fake integer pointer transport.
- _TensorDataPointer stores OptionalReg[Pointer] plus allocation/offset/extent.
  MTS accepts generic DevicePointerLike, _as_input preserves storage, and
  reshape/broadcast/transpose preserve typed metadata.
- Collective declarations AND calls are Convergent/NoDuplicate before shared
  optimization. Generic TTI avoids borrowing CPU ISA costs; SLP enabled.
  Pointer leaves are nullable; do not attach unconditional LLVM nonnull.
  Non-null allocation views remain validated at HAL binding.
- OpenCL HAL pinned HostBuffer uses CL_MEM_ALLOC_HOST_PTR + map/unmap.
  HOST_ADDRESS is a real mapped pointer; device ADDRESS remains opaque. Pinned
  handles reject device kernel bindings; host copies/fills preserve ordering.
- Logical kernel names are bounded by bundle size, not an arbitrary 4096 bytes;
  generic MTS signatures exceed that. Native identifiers retain separate bounds.
- Dynamic alloca, fp64/half atomics, packed bool-vector memory/arguments, arbitrary external/indirect calls and sanitizers remain unsupported.
  i64 atomics require extensions missing on Adreno840. Atomic system scope means
  device scope under non-SVM/non-peer contract. Do not silently CPU-fallback.

## Active standard-storage / full graph pass

User requested both complete storage integration and full-model graph validation,
not just another review. Profiling only after both finish. Work in progress;
do not claim complete. Artifacts/environment now live under
/Users/REDACTED/Library/Caches/modular-android-integration; previous /tmp
prototypes and logs were lost on interruption. Phone DEVICE_SERIAL connected.
Current rebuild log /tmp/modular-integration-build.log; native SDK override
recreated using /Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk.
- 2026-10-25: generalized tile adapters, attention dispatch/masks, collective
  arrays/vendor boundaries, optional GPTQ permutation, EP format fields and
  constructors. Inferred engine parameters need explicit binding when a kernel
  symbol becomes a launch function. Nested Array/Optional types cannot use `_`
  for an unbound engine: name a TensorEngine parameter. Engine defaults prevent
  constructor inference; pass actual engines explicitly for EP.
  Native mo-opt source build now succeeds (before the next bridge changes).
  Real graph compilation found and fixed stale PointerType construction in
  MAPOpsLowerableToMOGG::IterEpilogueOp (TensorGetPtrOp now returns StorageType).
- CheckLifetimes.cpp had inconsistent associated-type field evaluation (the
  tracker counted Engine.StorageType as one field; GER resolved an aggregate).
  Set evaluation context for field counting, field traversal, diagnostic paths,
  partial destructors and generated GERs. This preserves checks; no bypass.
  New ownership/associated_storage_fields.mojo regression fails with the old
  source compiler and passes with the fix; Bazel parser regression passed.
  Full std/MAX/builtin package build passed with the fix. Diagnostic field
  dump was removed from source. Broader lifetime/CPU tests still required.
- Current operators compile log /tmp/modular-operators-lower.log (Bazel run).
  Remaining discovered errors: GEMV uses TileTensor.to_device_buffer and only
  permits pointer engines; recursive closure encoding fails field reflection.
  New DeviceBuffer._borrow_pointer_like in both HAL/native implementations,
  and generic TileTensor.to_device_buffer, are IN PROGRESS, unvalidated. HAL
  preserves Buffer(handle,offset,extent,_is_view=True); it does not own the
  allocation. Generic type refinement needs downcast + rebind after conforms_to.
  Borrow path now builds, but phone borrowed-buffer fixture still pending.
  Reflection issue was an incorrect comptime guard in
  _contains_device_passable_field: expressions after a return still elaborate.
  Guard ALL reflection inside the is_struct / not _RegTuple branches. Removed
  the Bool name workaround. No reflection/compiler evaluator change remains;
  temporary KGENAttrsFolders.cpp diagnostic fully removed. Nested Int/UInt
  capture fields use encode (checks matching target width) rather than direct
  SIMD._to_device_type, which intentionally rejects direct Int kernel args.
  DevicePassable + reflection + managed-tensor-slice CPU suites PASS (log
  /tmp/modular-storage-tests.log); new index/shape-array encoding test covers it.
  Five parser ownership tests (associated storage + core/CF/EH/ops) also PASS.
  Do not run a Bazel build/test while mo-opt is elaborating: builds remove
  shared std.mojoc outputs and can cause a missing-std crash mid-run.
  New operators fixture: max/examples/android-llm/graphs/operators.mlir.
  No app changes/profiling yet.
- Paged KV cache constructor CPU suite PASS after making ptr_at_offset
  dispatch through Engine.offset/unsafe_ptr instead of requiring DefaultEngine.
  New graph_storage_paged_cache_fixture.mojo PASS on phone (packed and padded
  pages, reordered GPU LUT, both K/V and two layers; every untouched/padded
  element checked). Fixture compiled fresh from source; same staged runtime
  as storage fixture. Logs /tmp/modular-paged-cache-{build,phone,cpu}.log.
- Formatted OpenCL assertion regression emits LLVM with a condition, branch
  and llvm.trap, no _FixedWriteBuffer or printf (FileCheck PASS). Reduction
  elaboration now succeeds. Emitter next rejected <1 x float> from LLVM
  scalar masked-memory tails/broadcasts. Tried LLVM Scalarizer; it retains
  one-lane reconstruction for wider users. Removed that experiment and maps
  LLVM one-lane values to their identical scalar representation in the C
  emitter, including scalar component access and signed bool masks. Multi-lane
  arithmetic unchanged. Expanded regression includes scalar/vector bitcasts,
  loads/stores and a broadcast to four lanes. Emitter regression PASS;
  all 13 actual RMSNorm variant LLVM modules emit and pass Clang OpenCL C
  syntax checks (/tmp/modular-rmsnorm-saved-ir.log). Clang ignores the vendor
  subgroup shuffle pragma with -Wno-ignored-pragmas; phone compilation pending.
  Private storage of a global pointer also exposed C qualifier placement:
  emit `__global uchar * __private *`, keeping the two address spaces separate.
  Native source pipeline now recompile needs this emitter change.
- Updated GraphStorageEngine fixture PASS on phone DEVICE_SERIAL three runs,
  including new borrowed DeviceBuffer -> pointer launch, dropping the borrowed
  wrapper before synchronization, static/dynamic layouts, shifted host views
  and TensorOps. Built fresh fixture from source with native compiler snapshot;
  linked existing Oct24 Android CompilerRT/plugins (runtime unchanged).
  Logs: /tmp/modular-storage-fixture-build.log, /tmp/modular-storage-phone.log;
  durable host.o/compiler/packages under cache storage-fixture/ and native/.
- NEXT BLOCKER: checked gather kernels contain live llvm.trap (24 one-layer
  variants; no workgroup barriers in these modules). Native driver probe proves
  __builtin_trap is ignored on Adreno840: flag=0 AND flag=1 both return42 with
  clFinish success. DO NOT map llvm.trap to it, disable assertions, or fault an
  arbitrary address. Probe in cache trap_probe.cpp; logs /tmp/modular-trap-*.log.
  Standard failure support planned: ABI3 kernel flags bit1 (value2) marks an
  implicit queue-owned uint error buffer; bit0 remains dynamic local memory.
  Compiler propagates error-buffer argument through trapping helpers, records
  failure atomically and returns; callers propagate failure before proceeding.
  Reject kernels combining live failure and workgroup barriers until safe
  workgroup termination exists. HAL snapshots error flags asynchronously when
  recording events, retains readbacks through completion callbacks, checks at
  event/queue/blocking-copy completion, and propagates cross-queue dependencies.
  Use dedicated custom HAL status independent of thread-local error strings.
  Protocol now implemented in emitter/bundle/plugin. All 8 host OpenCL
  compiler/emitter tests + native plugin/SDK test builds PASS
  (/tmp/modular-device-error-tests.log). Target/test/OpenCLAssertions.cpp tests
  compiler-produced assertions.mcl: valid result, failures via event query/wait,
  queue sync/blocking reads, dependent queues, re-recording and early event
  destruction. Phone DEVICE_SERIAL PASS all four assertion cases, including failure query/wait,
  sticky queue status, cross-queue dependency snapshots, event re-recording and
  early destruction (/tmp/modular-assertions-phone.log). Test built with NDK33
  from Target/test/OpenCLAssertions.cpp; plugin is canonical Android Bazel build.
  Native tools rebuilt with error support. One-layer prefill graph now compiles
  through normal mo-opt and mt to block-s2.mef (123MiB).
  Snapshot native-graph/ now contains
  tools +22 packages + all dylibs; env.json permits direct mo-opt/mt independent
  of Bazel output switching. block-ir.*.post-opt.*.ll captures exact gather
  bounds-check failures. Native snapshot includes current sincos and device-error support.
- One-layer real-weight transformer elaborates all ordinary kernels, then
  exposed scalar bool leaves in captures. Scalar LLVM i1 has one-byte allocation
  layout and can use native uchar (already its C representation); removed the
  blanket bool-argument rejection, retain packed-bool vector rejection because
  its packed layout differs from ucharN. ABI3 fields preserve offset/size.
  Regression covers direct scalar bool and nested aggregate bool PASS.
  Standard optimizer also combines sin/cos to llvm.sincos (LLVM23), whose
  floating pair now maps to a C pair with a single OpenCL sincos call. Keep
  optimization; do not disable the combine. Generic standalone packed-math
  test now uses standard SIMD casts/fma(AFN)/reduce, no deleted explicit
  Bonsai intrinsics. Subgroup behavior is tested via Adreno offload, not the
  generic standalone target. All 8 OpenCL Bazel compiler/emitter tests PASS
  (/tmp/modular-opencl-regressions.log). mojo-build now preserves the underlying
  emitAssembly error text, instead of swallowing it as 'could not emit assembly'.
- Combined matmul/RMSNorm/softmax/slice+concat module now compiles through
  normal Android/OpenCL MAX lowering PASS (operators-lowered.mlir, 1.2 MiB).
  MEF serialization next. Std debug-assert suite + paged-cache CPU test: 11
  Bazel tests PASS. Fourteen GraphCompiler tests fail to build their common
  testing_kernels dependency because HAL DeviceGraphBuilder lacks add_function
  (optional capture API; not ordinary MAX execution). Do not suppress checks
  or fabricate capture support. Run applicable FileChecks directly or test
  native implementation without HAL overlay. New mojo-opencl BUILD wires
  compile-only tests; old C-symbol checks updated for ABI3's native names.
- Matmul dispatch now isolates NVIDIA/AMD/Apple vendor kernels from other
  targets at comptime and uses the existing engine-generic naive kernel for
  portable targets. Split-K kernel signatures now preserve input/output engines.
  Isolated ordinary MAX matmul and slicing/concat graphs compile successfully
  for Android/OpenCL. RMSNorm and softmax hit duplicate elaborator errors
  (`impl node already has an error`); investigating with single-worker mt.
  Suppressed error is `_printf` during formatted assertion elaboration:
  `_FixedWriteBuffer` overflow reporting instantiates device printf even if
  the backend assertion hook emits no text. Added generic PluginHooks
  _debug_assert_needs_message (default true, OpenCL false); _debug_assert_fail
  skips formatting for such hooks, retaining check/abort. Validation pending.
  Temporary Elaborator.cpp diagnostics restored completely.
  No phone operator/full-model execution or app update yet. Operator outputs
  have distinct names; separate graphs are cached as operator-NAME.mlir.
  One-layer sequence-1 decode graph exported to block-s1.mlirbc. Full 30-layer
  prefill graph and independent HF prefill/decode cache/logits references exist.
- Durable operators.mlir and block-s2.mlirbc regenerated in cache directory.
  New source export_graph.py builds normal MAX SmolLM2 graphs with real weights,
  explicit cache inputs/outputs and CPU/GPU transfers. graph_reference.py uses
  HF eager attention independently; one-layer reference prefill/decode succeeds
  (greedy IDs365/216). run_graph.py accepts binary inputs and pulls mt outputs.
  Graph execution/CPU regressions/phone revalidation remain pending.
- Standard MTS.to_tile_tensor now uses GraphStorageEngine; to_graph_tile_tensor
  removed. Host metadata gains byte_begin for checked signed/chained offsets;
  multiplication checked before performing it. GPU uses already-encoded raw
  field directly (metadata is host-only), avoiding unreachable host traps.
- GraphStorageEngine now conforms to TensorOps using existing shared loops.
  An attempt to put defaults on TensorOps hit dependent-Self compiler bugs and
  was removed; concrete engine methods delegate to shared helpers. TileTensor
  and shared helpers now pass erased origins explicitly.
- Standard engine fixture (owner lifetime fixed) passes phone; extended fixture
  with copy_from, +=, *=, /=, -=, abs/min/max and signed chained offsets also
  passes. Logs /tmp/modular-standard-graph-{engine,ops}-device.log.
- !mogg.storage now maps to _TensorDataPointer[int8], not MutByteBuffer. Rebind
  extracts tensor_view()._ptr. TensorGetPtrOp returns storage; fused epilogue
  accepts typed storage; variadics and trait/shape wrappers preserve handles.
  Generated code and graph regression validation still pending.
- PagedKVCacheCollection had fixed DefaultEngine fields; engine parameters are
  being threaded through its existing inner cache engine machinery. Current
  compile retry /tmp/modular-standard-operators-lower5.log (Bazel session42262).
  Some generic layout reconstruction needs explicit type rebinding; finish and
  test safely (do not leave unchecked cross-engine reinterpretation).
- Source-generated Python MAX operator graph /tmp/modular-standard-operators.mlir
  tests matmul, rmsnorm, softmax, concat/slice. Lower to Android/OpenCL with normal
  bazel run mo-opt cross CLI. At present build must finish KV-cache package;
  actual operator execution and full model graph not yet tested.
- Native tools current build target //GraphCompiler/tools/{mo-opt:mo-opt,mt:mt}
  with -c opt --config=hal-device-context and local macOS sysroot override.
  Android runtime snapshot /tmp/modular-android-max-repro is archived; avoid
  overwriting native tools when switching to Android builds.

## Current work and evidence

Canonical platform migration:
- --config=android-arm64 now selects @platforms//os:android + aarch64, rather than
  Linux plus Bionic. NDK compiler/resource/sysroot/libc++ selected as a unit via
  --repo_env=ANDROID_NDK_HOME. Minimum API33 supplies real backtrace().
- Normal rules_cc wrapper/header/layering machinery retained. Shared ELF flags
  use //:elf (Linux or Android), Crashpad uses shared Linux-kernel constraints.
- Focused upstream dependency fixes: rules_cc Android SONAME compatibility;
  rules_mojo optional target Python for pure Mojo (Python deps still require a
  target runtime); XZ Android preset disables nonexistent libintl; Curl uses
  portable -pthread; TCMalloc Android39-bit VA. Removed speculative deprecated-
  compiler-label patches and now-unnecessary XZ/Protobuf pthread patches.
- Bundled glibc-only NVIDIA compiler excluded from Android CompilationDevice
  data. Native compiler tools remain in exec config. No fake Android Python.
- Current Android MAX mt build log /tmp/modular-android-max-driver-build.log.
  Active canonical Android build includes //GraphCompiler/tools/mt:mt,
  //Mojo:CompilerRT, //Target:{CPUPlugin.so,OpenCLPlugin.so} and
  //AsyncRT/unittests:AsyncValueTest. MAX tool build is still running.
  Canonical Android runtime/plugins archived /tmp/modular-canonical-android-runtime;
  all42 AsyncValue tests and the full DeviceContext suite (including typed fusion
  captures and standard elementwise row tails) pass DEVICE_SERIAL. Evidence:
  /tmp/modular-canonical-android-device-tests.log.
  XZ gettext configuration is fixed. Late std precompile exposed ASAN compiler
  override selecting the target Android compiler instead of the build machine.
  Fixed by selecting execution configuration first, then applying instrumentation
  transition through a compiler alias. Full build PASS:
  /tmp/modular-android-max-driver-build-retry3.log. Native ASAN compiler selection
  cquery also PASS: std stays asan, override compiler stays production on host;
  /tmp/modular-native-asan-compiler-selection.log. This checkout uses
  --//:modular_config=asan (no --config=asan rc alias).
  Both models rebuilt using the source host compiler and canonical runtime: tokens
  and logits match. Bonsai max_abs .0127101 / rel_l2 .000557542; Smol max_abs
  9.41157e-05 / rel_l2 2.70013e-06. Logs:
  /tmp/modular-canonical-{bonsai,smollm2}-validation.log. No profiling run.

Current fusion / standard algorithm validation:
- Generated fusion functors preserve _TensorDataPointer on host; typed
  simd_*_tensor_pointer overloads unwrap inside device code. Encoder recursively
  encodes register-passable composites with nested DevicePassable fields.
- Full DeviceContext suite PASS on phone, including transposed fusion input and
  ordinary max.algorithm.functional.elementwise over 2x3, simd4, row tails,
  unknown occupancy and untouched buffer suffix:
  /tmp/modular-graph-fusion-device.log; build /tmp/modular-graph-fusion-compile.log.
- Standard elementwise single/dual scheduler no longer requires SM/register
  occupancy data: use declared workgroup limit and work-sized grid when absent.
  Known-occupancy behavior retained. Kernel scheduling fields use Int64 transport.
- MAX_THREADS_PER_BLOCK_METADATA now comes from MaxPluginHooks. CUDA/HIP/Metal
  retain their existing keys; OpenCL TargetLowering emits a standard OpenCL
  work_group_size_hint. No hardcoded backend selector in GPU globals.
- Emitter legalizes masked memory with LLVM ScalarizeMaskedMemIntrinPass and
  a minimal OpenCL TTI reporting work-item divergence. Aggregate PHIs use LLVM
  DemotePHIToStack + SROA; freezes split into stable leaves. Pointer vectors are
  legalized independently of numeric SIMD, with typed base+offset provenance
  recovery. Fixed stepvectors, unaligned numeric memory and switch/phi edges work.
- Standalone emitter CTest PASS, including new masked gather/scatter, aggregate
  join/freeze, vectorized pointer arithmetic, metadata and OpenCL C3 syntax:
  /tmp/modular-unaligned-emitter-tests.log. Dead rejection fixtures updated to
  use observable unsupported operations now that normal cleanup removes dead IR.
- BOTH temporary diagnostics RESTORED: Elaborator.cpp and OpenCLBackend.cpp.
  No debug dump or mutex remains in production code. Type errors retain useful
  LLVM type details. Native compiler build PASS:
  /tmp/modular-opencl-legalization-compiler-build.log.
- Generic CrossCompilationCLOptions compiled in mo-opt/mt exposes existing
  AsyncRT virtual CPU/API/arch/count globals. Native HAL tools build PASS.
- Real mo.add/mo.mul + explicit CPU↔GPU transfers graph now compiles through
  normal gc-pipeline into MEF IR with Android host code and OpenCL bundles:
  /tmp/modular-opencl-graph-lowered.mlir (888KiB),
  /tmp/modular-opencl-graph-lower.log, target aarch64-linux-android33/adreno-840.
  Normal mt --compile serialization PASS:
  /tmp/modular-opencl-graph-mef.log, /tmp/modular-opencl-graph.mef (444KiB).
- Native CPU fusion regression now passes (see typed-stub section). Use bazel run for the fixture's
  mo-opt/mt/FileCheck commands; bare binaries lack required package environment.
  Stopped lit CPU target because its generic tool list builds unrelated LLDB.

Previously verified baseline (before canonical Android-platform refactor):
- Existing AsyncValueTest all42 tests pass phone:
  /tmp/modular-android-full-runtime-tests.log.
- Normal CompilerRT + full DeviceContext suite pass:
  /tmp/modular-full-compiler-runtime-device.log,
  /tmp/modular-full-runtime-integration.log.
- Explicit + captured MTS identity encoding pass:
  /tmp/modular-graph-closure-device.log.
- Existing GPU MTS suite static/dynamic/mixed layouts, strides and aliasing pass:
  /tmp/modular-mts-device-encoding-device.log; fixture /tmp/mts_encoding_fixture.mojo.
- Native CPU MTS + OpaqueGraphBuffer/Tensor/TensorBufferRef tests pass:
  /tmp/modular-graph-tensor-native-final-tests.log.
- SDK pinned-memory and ABI3 tests pass:
  /tmp/modular-pinned-sdk-{smoke,v3-tests}.log.
- Scalar GraphRT basics.mlir all11 fixtures execute on Android, FileCheck pass:
  /tmp/modular-android-mef-basics.log, /tmp/modular-android-mef-basics-check.log.
- Declarative launch FileCheck pass /tmp/modular-graph-declarative-launch-tests.log.
  Run compiler-only tests WITHOUT hal-device-context: existing device-graph
  fixtures require missing DeviceGraphBuilder.add_function in HAL.
- Both models normal-runtime validation pass, unchanged errors:
  Bonsai24 max_abs.0127101 rel_l2.000557542; Smol16 max_abs9.41157e-05
  rel_l2 2.70013e-06. /tmp/modular-{bonsai,smol}-full-runtime-{build,validation}.log.
  No performance tests in this integration pass.

## Typed outlined-stub storage (current unverified change)

### GraphStorageEngine slice (2026-10-24, phone-verified core)

GraphStorageEngine implemented in extensibility/managed_tensor_slice.mojo
(modeled on DevicePointerEngine/StableAddrEngine; TensorEngine conformance,
origin-erasing StorageType=_TensorDataPointer needs explicit origin binding in
copy_from). ManagedTensorSlice.to_graph_tile_tensor() returns
{self._ptr, self._runtime_layout} with that engine: host-side construction
from opaque storage works (no raw unwrap, shape queries, bounds-checked view
offset via _offset_storage), and engine-backed TileTensor args launch and
verify on DEVICE_SERIAL (single tile; tile pair + scalar; static and dynamic
layouts). Emitted OpenCL C and MOJOCL03 manifests are byte-identical in shape
to MTS args (BufferView leaf at offset 0 + zeroed metadata), decoded from the
built objects. Fixture: max/kernels/test/buffer/{test_graph_storage_engine,
graph_storage_engine_fixture}.mojo via run_device_context.py.

### Review correction: fixture owner lifetime, not proven refcount underflow

The prior fixture showed FREE_SYNC before a later tile launch and BAD_MEMORY.
That observation does not establish an ArcPointer underflow. Mojo destroys at
last use, not lexical scope exit. Input.device_ptr() is converted into an MTS
whose _TensorDataPointer explicitly erases origin and does not retain an owner;
the new fixture never directly uses input_device after constructing source.
The stock MTS fixture does use input_device later (map_to_host and in-place
launches), so its passing result is consistent with this lifetime difference.
DeviceBuffer is ImplicitlyCopyable/Movable, not TrivialRegisterPassable;
ArcPointer has a retaining copy initializer. Do not add speculative add_strong.

Controlled review experiment: copied fixture to /tmp/modular-review-graph-engine,
kept both owning buffers live until completion via ctx.synchronize() followed by
explicit final consuming uses (_ = input_device^; _ = output_device^). No runtime,
engine or refcount changes. All three cases (single tile, tile pair/scalar,
shifted tile), both static/dynamic layouts, passed three phone runs on DEVICE_SERIAL.
Evidence /tmp/modular-review-graph-engine-{build,device}.log. Original repository
fixture remains unchanged and needs this owner-lifetime correction. This does
not prove the absence of every possible runtime lifetime bug.

Other review findings: both new .mojo files enter buffer/BUILD.bazel's generic
test glob (fixture has no main; GPU test needs explicit registration/constraints).
Debug forwarding currently always sets MODULAR_OPENCL_DEBUG, even when empty,
while the plugin checks getenv presence, so debug logging is effectively on by
default through that runner. Backtrace include is guarded for Android/Apple but
calls are unconditional, breaking Linux portability. These are cleanup items;
GraphStorageEngine itself uses the intended TensorEngine extension point.

Remaining GraphStorageEngine work: TensorOps (elementwise in-place), checked
signed offsets (reverse/chained views), and the callsite migration
(builtin kernels' to_tile_tensor → to_graph_tile_tensor).

Real graph compilation passed, but outlined host-stub RebindOp still called
OwnedByteBuffer.unsafe_ptr before tensor construction. Implemented a generic
!mogg.storage type (borrowed MutByteBuffer view); generated routine args use it
on ALL backends. Raw !mogg.pointer remains explicit interop/constant storage.
TensorCreate accepts either, with optional operand type annotation via custom
StorageOperandType parser/printer (old raw syntax stays valid). No backend gates.
Rebind storage borrows tensor_view; MTS pointer reinterpret_as preserves bytes
and owner borrowing, new mogg_tensor_init overload consumes the typed view.
Stable-slot overload explicitly extracts a raw pointer (unsupported opaque slots
remain a separate device-graph limitation). Added metadata reconstruction to
OpaqueGraphTensorTest fixture. Native tools build PASS:
/tmp/modular-typed-storage-native-build.log. Mojo fixture compile PASS /tmp/modular-opaque-stub-storage-compile.log.
Native OpaqueGraphTensorTest PASS /tmp/modular-opaque-stub-storage-native-test.log.
New graph lowering PASS /tmp/modular-opencl-graph-storage-lower.log, output
/tmp/modular-opencl-graph-storage-lowered.mlir (903KiB). Dumped source proves
__add_mul_async passes tensor_view() into typed MutByteBuffer stub args, with
no unsafe_ptr call in generated arithmetic module. Re-serialized current graph
MEF at /tmp/modular-opencl-graph.mef. Native CPU fusion regression PASS: existing kernels-in-out-fusion.mlir lowers,
executes all five graphs and passes FileCheck, including matmul and reduction
prologue/epilogue casts. Logs /tmp/modular-fusion-native-e2e-{lower,execute}.log.
Android mt/runtime/plugins/AsyncValueTest canonical build PASS,
/tmp/modular-android-max-driver-build.log. Runtime/plugins already pass phone
validation (see canonical-platform evidence above). Correct targets are
//GraphCompiler/tools/mt:mt //Mojo:CompilerRT //Target:CPUPlugin.so
//Target:OpenCLPlugin.so //AsyncRT/unittests:AsyncValueTest.
Current /tmp/modular-opencl-graph.mef has been rebuilt after typed migration;
its earlier raw-stub version was overwritten.

Parser roundtrip/FileCheck also PASS for explicit !mogg.storage tensor creation:
/tmp/modular-storage-parser-roundtrip.mlir. New syntax is covered in existing
GraphCompiler/test/mo-opt/MOGGDialect/Parsing/codegen-ops.mlir.

## Next steps / hard boundaries

1. DONE canonical Android mt cross-build, runtime tests, model numerical checks
   and real arithmetic MEF execution on phone.
   Native graph regression checks pass107, with1 preexisting unsupported fixture. Logs /tmp/modular-storage-lit-updated.log
   plus /tmp/modular-storage-lit-final.log and /tmp/modular-storage-lit-fused.log.
2. DONE arithmetic graph through normal mo-opt/mt to Android/OpenCL MEF, then
   normal Android mt execution with SDK plugins. Source fixture
   Target/test/OpenCLGraph.mlir. Outputs six2s and six4s. Evidence:
   /tmp/modular-android-max-repro-execution.log + fixture FileCheck PASS.
   Runtime staging build_runtime.py --include-max-tool and deployment
   run_graph.py tested end-to-end. Normal compilation/runtime discovery paths
   supplied via MODULAR_MOJO_MAX_{COMPILERRT,MGPRT}_PATH. No alternate executor.
   Build artifacts archived /tmp/modular-android-max-repro and
   /tmp/modular-canonical-android-max before native configuration switches.
   Full MAX model graphs remain unverified; don't equate this smoke with that.
3. Fix raw consumers encountered: MTS.to_tile_tensor unwraps on host; some normal
   operator/trait wrappers call it. mogg.tensor.create.slice uses slice_as_view
   on host; use shared layout/offset machinery with typed storage. Variadics
   erase pointers. MOGGOps _MoggTensorArg trait lane also uses raw storage.
   TensorGetPtrOp/epilogue outputPtr reconstruction also extracts raw storage,
   even though fusion capture fields themselves are now typed.
   Negative opaque unsafe_offset currently rejected; reverse/chained views need
   checked signed offset support. Do not fabricate addresses or add graph gates.
4. Establish small graph → transformer block → pretrained MAX model correctness.
5. Defer performance profiling. Optional HMX head currently synchronizes typed
   host transfers; restore stream/event overlap before hybrid comparisons later.

## Reproduction and paths

Phone DEVICE_SERIAL (SM-S942U1, Adreno840, Android37). SDK
/Users/REDACTED/Library/Android/sdk; NDK ndk/29.0.14206865; adb platform-tools.
Use ./bazelw --compilation_mode=opt. Local host override:
--override_repository=sysroot-macos=/tmp/modular-macos26-sdk.

CAUTION: Darwin/Android Bazel configurations share darwin_arm64-opt output paths.
Archive ELF artifacts before native rebuilds; bazel-bin symlink changes config.
Stable source compiler path bazel-out/darwin_arm64-opt/bin/Mojo/tools/mojo/mojo.

Target/lib/OpenCL/build_runtime.py builds normal CompilerRT + CPU/OpenCL plugins,
stages runtime global DSOs and NDK libc++_shared, emits android_artifacts.json.
run_device_context.py stages HAL source via Bazel and precompiles matching std/MAX.
Both accept --bazel-arg. --opencl-headers is only for standalone SDK CMake tests.
Model build.py/run.py deploy every manifest library; set remote LD_LIBRARY_PATH.
DestroyGlobals after all Mojo work/model destruction, before code unload.

Current snapshots /tmp/modular-full-runtime-integration,
/tmp/modular-android-full-compiler-runtime, /tmp/modular-android-full-runtime-phone,
/tmp/modular-bonsai-full-runtime, /tmp/modular-smol-full-runtime.
Model assets /tmp/modular-bonsai2-reference2, /tmp/modular-smollm2-export;
tokenizers /tmp/modular-bonsai2-hf, /tmp/modular-smollm2-hf. Python env
/tmp/modular-android-feasibility-20261023/bin/python has transformers5.20.2.

Standalone emitter /tmp/modular-opencl-compiler (cmake --build, ctest).
C++ formatter /opt/homebrew/opt/llvm@22/bin/clang-format; Mojo formatter
/tmp/modular-android-feasibility-20261023/bin/mblack --fast --preview -t mojo.
Historical profiling files remain outside repo; do not rerun during integration.

## Current phone graph follow-up (October 25)

- Canonical Android runtime + mt build PASS, staged in durable cache
  android-runtime/ (normal mt ~493MiB; logs modular-android-runtime-build.log).
- First combined ordinary operator run: matmul executes; RMSNorm rejects zero
  grid. Root cause compile-time Adreno840 GPUInfo.sm_count=0. Phone OpenCL
  query reports 12 compute units, max workgroup1024, local memory32768.
  Target table now records these and a conservative1024-thread/CU scheduling
  bound; genericOpenCL uses oneCU lower bound. Native rebuild pending.
- Real block first run traps on OwnedTensor.raw_view. Found missing opaque
  branch in mogg_async_pack_owned_tensor_value (consuming pack); borrowing
  async_pack already handled it. Added matching opaque SDK allocation/offset/
  extent + shape transfer. Phone rerun pending, not yet correctnessPASS.
- Full30-layer prefill lowers successfully. mt autosniff only scans first1MiB
  and misses mo dialect for this binary. Use documented --input-format=mgp
  after lowering; serializes successfully. compile_graph.py updated.
- Parser roundtrip found TileCreateEpilogueOp printer omitted optional
  !mogg.storage operand type (parser already accepted it). Fixed printer and
  FileCheck patterns; native rebuild then rerun regression.
- Current native build /tmp/modular-metadata-native-build.log, jobs4 to limit
  memory pressure. Refresh frozen native-graph snapshot after success, before
  Android config switches. Recompile operators/block/full with newmetadata
  and consuming-pack fix. Full decode initial compile was terminated to avoid
  running three heavycompiler jobs concurrently; still needs compile.

- Flat mgp.buffer.concat now uses ordinary ordered DeviceContext byte copies
  on accelerator targets and creates sub-buffer views without raw host pointers.
  CPU concat path unchanged. This follows the same opaque-aware copy path as
  graph device transfers; no new executor or backend-specific kernel.
- Native metadata/printer build PASS (312s). Final incremental build after
  consuming-pack/concat edits: /tmp/modular-storage-final-native-build.log.

- Corrected RMSNorm diagnosis: NOT a nested closure conversion defect. Dumped
  generated Mojo proves an intentional `unsafe_dangling()` type-only fusion
  MTS. Its data pointer=1, metadata=0, and loads use a real captured allocation.
  Source dump in cache/operators-source/; raw diagnostic phone log
  /tmp/modular-encoding-raw-phone.log. Temporary GRAPH_VIEW print removed.
- Implementing nullable kernel leaves as normal OpenCL null cl_mem arguments:
  canonical SDK view {nullptr,0,0} only; malformed offset/extent still rejected.
  Removed unconditional LLVM NonNull attrs (would erase Optional pointer guards),
  emitted C avoids null pointer arithmetic, `_TensorDataPointer.null()` backs
  type-only fusion/empty-buffer placeholders. No fabricated pointer1 and no
  dummy device allocations. Generic graph emitter change, no hardware gate.
  Added compiler optional-pointer check and SDK null/some/malformed regression
  to assertions fixture/test. Canonical native build in
  /tmp/modular-nullable-native-build.log; phone validation next.

- Nullable compiler/device test PASS on DEVICE_SERIAL:
  /tmp/modular-nullable-assertions-phone.log. Optional None->0 / Some->1,
  malformed null offset/extent rejected, all prior assertion/event cases PASS.
  Normal Mojo Pointer remains non-nullable; null is represented by OptionalReg
  in graph storage and must not escape through host unsafe_ptr(). The temporary
  attempt to materialize raw Pointer(0) was removed. `--emit llvm` is pre-LLVM-opt
  IR (even -O3), so nullable FileCheck matches ptrtoint+zero test; actual
  optimized generated device artifact proves the guard survives end-to-end.
  Frozen extensibility.mojoc re-precompiled directly after removing that bad
  temporary branch; full canonical final build still required before finishing.
  Canonical Android plugin and SDK test built; staged plugin updated.

## Latest verified state (resume here, October25 ~03:06)

- Ordinary MAX operators ALL numericPASS on phone: matmul exact8, RMSNorm
  maxerror6.83e-9, softmax exact0.125, slice+concat exact1.
  /tmp/modular-operators-fusion-pack-phone.log + modular-operators-numerics.log.
- Slice+concat initially returnedzeros: `_FusionPack` value param-pack was
  reflection-opaque and NOT DevicePassable, so nested prologue allocation views
  were byte-copied without resolving pointers. Implemented identity-layout
  DevicePassable, target size_of/align_of pack offsets, shared recursive encoder
  on each element. Standard pipeline now passes view/variadic/fusion numerical
  check. No temporary GRAPH_VIEW logging remains.
- Real block first post-nullable run exposed C++ TensorStateSlot get/set presence
  tests calling getBuffer() on opaque cached weights. Changed shared presence
  predicate to check opaque/owner before a raw-backed pointer, including retained
  zero-bytebuffers. Added native regression to OpaqueTensorBufferRefTest.
  Android MGPRT buildPASS /tmp/modular-cache-android-build.log. Canonical entire
  runtime+mt staging build now /tmp/modular-cache-full-runtime-build.log.
- Current block-s2 recompiling from frozen native-graph tools plus updated
  extensibility.mojoc directly: /tmp/modular-block-fusion-pack-compile.log.
  Then phone rerun tocache/block-phone/prefill, compare EVERYlogit/cache element.
  block-s1 and fullmodel MEFs still need freshcompile after FusionPack/null fixes.
- Frozen tools are current nullable/printer sourcecompiler. Frozen
  extensibility.mojoc was directly refreshed after removing bad rawPointer(0)
  and adding FusionPack conversion; other sourcepackages load it byimport.
  Nativecanonical final build/tests not yet run for cache/FusionPack changes.
- Remaining raw API boundary found during scan: GraphCompiler/lib/Driver/Device/
  DeviceExecution.cpp prepareInput() and constructDeviceMemoryFromTensor() still
  use raw pointer wrapping. Explicit CPUinput/output fixtures bypass that public
  GPUbufferboundary. Needs opaqueview-preserving adapter + test before claiming
  clean standard MAX GPUinputs/outputs. DeviceBuffer::hasOpaqueStorage(),
  getOpaqueMemoryView(), createSubBuffer(), MojoHALbuffer ownership APIs exist
  under HAL. Do not fabricate raw addresses.
- App untouched, performance work not started; user ordering unchanged.

## Active work after latest phone run (~03:18)

- One-layer run now gets past cached GPUweights after TensorStateSlot fix.
  Still NOT numericalPASS. Nextabort traced with call_location[inline_count=4]
  to nn/gather_scatter.mojo:1175 (scatter_nd host flatten via updates.ptr).
  Fixed flatten to updates.reshape, plus host alignment check replaced with
  TileTensor.is_storage_aligned. New DevicePointerLike._has_raw_address hook
  defaultTrue; HALDevicePointer and GraphTensorDataPointer override for opaque
  views. Unknown alignment takes scalar path; no fake numeric address.
  Stdtrait change requires full matching sourcepackages; native build active.
- `_TensorDataPointer.unsafe_ptr` now inlinealways; abort location currently
  call_location[inline_count=4] is a DIAGNOSTIC choice, should revert to a
  stable shallower callsite/default before final build if inaccurate in other
  paths. Do not leave temporary GRAPH_VIEW prints (already removed).
- Public MAX GPU input/output boundary IMPLEMENTED but UNVALIDATED:
  DeviceBuffer._borrow_allocation_view factors existing SDKviewborrow constructor;
  new MojoHAL_DeviceContext_borrowOpaqueMemoryView C export and C++
  DeviceContext::borrowOpaqueMemoryView reuse it. Graph Driver prepareInput
  preserves opaqueview+owner; constructDeviceMemoryFromTensor retains owner
  through an opaque borrowedbuffer destructor; typed borrowBuffer overload
  supports mt result readback without rawaddresses. Added const Tensor::
  getBufferRef overload. Runtimebuild hadconstgetter error, fixed via overload.
  No backendconditionals in passes; build-HAL boundary is driver/runtime only.
- Added FIFTH operators graph gpu_buffers with actual GPUinputs+outputs and
  expected2; previous four numericPASS stillvalid but newfivegraphnotcompiled/
  runtested. Cache/operators.mlir currently OLDfourgraphcopy, refresh fromrepo
  source before compile. Test prepareInput/readback paths using normalmt.
- Current native canonical build /tmp/modular-addressability-native-build.log
  session96116, jobs6, includes mo-opt/mt/mojo and OpaqueTensorBufferRefTest.
  Latest Android runtime staging /tmp/modular-graph-boundary-runtime-build.log
  failed onlyconstgetBufferRef; stillneedsrebuild AFTER native snapshot freeze.
- Cached android-runtime currently previousStateSlotfixed allruntime+mt.
  Immutable copy android-runtime-pre-boundary useddiagnostics; sourcecompiled
  block-s2.mef current with unsafe_ptr location4. Frozen tools/22packages still
  pre-new-stdhook and prepublicDriveradapter, don'tcompile newlyedited kernels
  against them. Refresh snapshot after nativebuildsuccess.
- New OpaqueTensorBufferRefTest StateSlot regression has not beenrun yet.
  Nativefinal testing shouldinclude stdDevicePassable + MTS CPU + updated
  compilerlit suite, parser tile roundtrip, then phoneoperator5 + block +full30.
- No app changes or new Bonsai profiling yet. Original task remains active.

## Latest status (~03:39): public GPU boundary PASS, BMM fix next

- Canonical Android runtime+mt updated with standard addressability hook,
  C++ publicGPUview adapters, StateSlotfix. BuildPASS
  /tmp/modular-addressability-android-build.log (366s).
- ALL FIVE ordinary graphs passphone+numerics with thisruntime: matmul, RMS,
  softmax, slice+concat AND GPUinputs+GPUoutputs (gpu_buffers exact2). Logs
  /tmp/modular-gpu-boundary-{phone,numerics}.log. This validates normal MAX
  prepareInput and mt GPUresult readback, not an alternateexecutor.
- Native stdDevicePassable, MTS CPU(includingheterogeneousFusionPack), all8
  compiler/emittertests PASS. C++StateSlot regression originallystackallocated
  ReferenceCountedobject, corrected to RCRef; canonicalnativebuild/test now
  PASS /tmp/modular-addressability-final-native.log (128s, test1PASS). No
  runtime refcountworkaround was added. New opaque alignment CPUregression
  also PASS /tmp/modular-storage-alignment-test.log.
- Block progressed after scatterfix, then anotherhostrawaccess. Diagnostic
  call_location4 pinpoints linalg/bmm.mojo:212: reshape-to3D helper rebuilt
  TileTensor from tensor.ptr and defaulted engine. Fixed result Engine=
  tensor.Engine and constructor storage=tensor._storage. Allnative/rawbackends
  preserve engine too. Diagnostic4 revertedagain tostablecall_location().
- Sourcepackages extensibility+linalg directly refreshing in frozen native-graph
  (session log /tmp/modular-bmm-packages.log). After successrecompileblock-s2,
  runphone against current android-runtime, comparealllogits/cache. block/full
  decode and full30model MEFs need newBMMcompile. Full30prefill currentlyoldBMM
  even though serializedsuccess /tmp/modular-model-integration-prefill-compile.log.
- CurrentBazelconfig native (CPUalignmenttestlast). Frozen tools22packages
  fromlatestnative sourcecompiler; C++runtime has newpublic adapters. Need
  canonicalnative build/test after BMM change + anylaterchanges atfinalverify.
- App untouched. Bonsai perf notstarted. No taskmilestonecomplete yet.

## Continuation: BMM storage signatures

- BMM reshape preserves Engine and _storage. Both GPU kernel signatures and
  all three launch specializations now carry actual C/A/B storage engines.
  Naive scalar contraction requests width=1 explicitly, avoiding an assumption
  that arbitrary TensorEngine element widths match. Frozen linalg package
  precompile PASS; block-s2 rebuild in /tmp/modular-block-engine-compile.log.
- Correctness gate remains: block and full-model prefill/decode numerics on
  phone, decode consuming phone-produced cache. Five ordinary operator graphs
  including public GPU input/output already pass. App and profiling pending.

## Continuation: real graph numerics and local pointer provenance

- Block-prefill completes after BMM signatures, cache maxabs1.91e-6 relL2~1e-7
  but logits FAIL. Intermediate graph shows attention probabilities/output and
  residual addition PASS; first divergence fused residual+RMS normalization.
- Isolated ordinary MAX fused residual+RMS576 reproduces FAIL; standalone
  RMS576 PASS. Raw OpenCL diagnostic (cache/fused_probe.cpp, fused-diagnostic/)
  verifies per-thread and per-warp sums PASS; cross-warp shared-memory reads
  uninitialized. Changing ONLY emitted GEP from integer roundtrip to direct
  byte-pointer arithmetic makes both row sums correct12.3479595/12.3075123.
  Volatile does NOT fix it. Preserve pointer provenance in source emitter.
- GEP emitter fixed for all address spaces, retaining modular64bit displacement
  and signed pointer addition. New emitter structural/signed-offset regression
  and sixth ordinary operator graph rms_residual (multiwarp fusion). Native
  canonical build /tmp/modular-pointer-provenance-native-build.log active.
  After success snapshot_native.py, compile/run block + full30 prefill/decode.
- exporter --intermediates optional first-layer tensor outputs helps diagnose
  real graph; cached independent torchhooks reference script. No app/perf yet.


## Integration and profiling pass ending 2026-10-25

# Android integration working ledger

Updated 2026-10-25. Active implementation; do not declare completion yet.
Earlier investigation and evidence: [history](INTEGRATION_HISTORY.md).

## User objective and required order

1. Finish standard tensor-storage integration (ordinary TileTensor/TensorEngine).
2. Validate normal transformer operators and a full real-weight MAX graph,
   including prefill/decode consuming actual phone-produced KV cache.
3. After both pass, update `/Users/REDACTED/projects/mojo-chat`, deploy and
   notify user that milestone is complete.
4. Profile Bonsai through CLI, optimize measured bottlenecks until no obvious
   low hanging fruit remains. Update app again. Report actual throughput;
   100 tok/s is the user's aspiration, not a guarantee. Prior measured ~41 tok/s.

Phone tests/app deployment authorized. No commits requested. Preserve all
collaborator changes and index. No subagents. Keep commentary updates flowing.
Read root/max/kernels/Mojo/GraphCompiler AGENTS for edits in those areas, app
AGENTS before app work, Android skill before its workflows.

## Landed in working tree

- ABI3 only, v1/v2 removed. Typed aggregate provenance survives lowering before
  host C ABI coercion; compiler `usesHostCABI=false`. SDK encodes opaque views.
- Normal MODULAR_DRIVER_PLUGINS discovery, Mojo DeviceContext, actual shared
  CompilerRT/AsyncRT/GraphRT, canonical Android NDK/Bazel cross toolchain.
- Opaque allocation/offset/extent/owner through graph TensorBufferRef,
  OwnedTensor/OwnedByteBuffer, views, transfers, public GPU input/output driver
  boundary and mt readback. No fabricated GPU addresses.
- MTS.to_tile_tensor uses GraphStorageEngine; old to_graph_tile_tensor removed.
  TensorOps, checked signed/chained offsets, pointer-like adapters and engine
  generic wrappers. Host storage alignment queries return unknown for opaque
  storage, safe scalar fallback. Device pointer resolution occurs after encoding.
- DevicePassable target-layout reflection, nested scalar identity handling,
  FusionPack per-member encoding/padding. Identity MTS/capture representation.
- Shared LLVM optimizations/SLP/generic TTI, convergent collective calls,
  SIMD, static private alloca, shared memory/barriers/subgroups, direct helpers,
  CAS/FP32 atomics and hardware-gated i64. Standard math contracts in Bonsai.
- Nullable BufferView `{null,0,0}` preserved; malformed null offsets rejected.
  Type-only fusion base uses explicit null storage, not dangling pointer1.
- Software device-error flag through kernel/helper/event/queue/copy completion.
  Adreno native trap experimentally ignored. Assertions+workgroup barriers
  fail closed pending safe group termination; don't disable assertions.
- StateSlot cached-weight presence uses ownership/opaque storage, not rawaddress.
- Scatter flatten and BMM reshape preserve engine/storage. BMM kernel launch
  signatures carry actual C/A/B engines; naive contraction explicitly width1.

No speculative ArcPointer retain fix: collaborator lifetime failure came from
nonowning views erasing origins and owners dying at last use. Tests retain/use
owners through completion; keep normal ownership semantics.

## Verified before current rebuild

Phone DEVICE_SERIAL (Galaxy S26/Adreno840):
- GraphStorageEngine static/dynamic tiles, TensorOps, signed shifted views,
  tile-pair/scalar args and borrowed wrappers: PASS (three runs).
- Paged cache packed/padded, reordered LUT, K/V, two layers/untouched checks PASS.
- Five normal MAX graphs: matmul, RMSNorm, softmax, slice+concat and actual
  public GPU input/output: numeric PASS. Logs `/tmp/modular-gpu-boundary-*`.
- Nullable and async assertion/error/event/crossqueue lifetime fixture PASS.
- Real-weight one-layer MAX Smol block completes after BMM fix. KV cache
  maxabs1.91e-6, relL2~1e-7 PASS; logits FAIL (not yet model correctness).
- Diagnostic intermediates: probabilities, attention output, residual add PASS;
  first divergence fused residual+RMSNorm. Isolated standalone RMS576 PASS;
  fused residual+RMS576 FAIL, reproducing independently of graph lifetimes.

Native stdDevicePassable, MTS CPU/FusionPack, opaque alignment, compiler/emitter,
parser/reflection and StateSlot tests passed before current emitter change.
Fourteen broader GraphCompiler tests cannot build common testing kernels due
HAL DeviceGraphBuilder lacking optional `add_function` capture support. Don't
fake capture or disable checks; ordinary graph execution does not need it.

## Current state: integration milestones PASS, first app update underway

Native canonical compiler build and all8 compiler/emitter tests PASS. Optional
LLVM22 clang syntax verification also PASS. GEP uses direct pointer arithmetic;
no integer roundtrip. Native stdDevicePassable + MTS CPU/FusionPack tests PASS.
Updated source GraphStorageEngine and paged-cache fixtures rebuilt and PASSphone.
ALL SIX ordinary graphs numericPASS including multiwarp fused rms_residual.

Real-weight normal MAX SmolLM2 (30 layers) validated every output value:

| Run | Max logit error | Max cache error | Greedy actual/reference |
| --- | --- | --- | --- |
| block prefill | 3.82e-5 | 1.91e-6 | 365/365 |
| block decode | 7.63e-5 | 1.91e-6 | 216/216 |
| full prefill | 1.86e-4 | 8.78e-5 | 198/198 |
| full decode | 5.73e-5 | 8.78e-5 | 1/1 |

Both decodes consumed actual phone prefill cache. Full relL2 errors <2.4e-6.
Reports cache/{block,model}-phone/{prefill,decode}/validation.json; reusable
verify_graph.py checks finite/all logits+cache/errors/greedy. Logs
/tmp/modular-{block,model}-provenance-*. Normal MAX→MEF→mt→SDKHAL, no customexec.
Optional exporter --intermediates emits first-layer stages +nameJSON fordiagnosis.

Integration1/2 is complete for tested normal transformer/storage paths. Generic
backend support is not universal operator/vendor coverage. Bonsai remains normal
Mojo DeviceContext orchestration, not a MAX Bonsai architecture graph.

Current app milestone (before profiling):
- cache/bonsai-integrated built canonically using updated frozencompiler,
  build.py --variant bonsai. EmbeddedABI3 model.o+runner+normalruntime manifest.
- mojo-chat prepare.py stages runtime libraries +HALplugins separately,
  --reuse-model retains weights/tokenizer. Old kernels.mcl assets removed.
- CMake imports canonical runtime instead of compiling alternateplugin; HALplugins
  packaged as jniLibs without linking them globally (avoid symbol interposition).
- JNI uses new engine(plugin,weights), removed global fastmath override (standard
  model mathcontracts apply), minSDK33. install.py provisions weights+tokenizer.
- App JDK21 build+lint/tests and AndroidTest APK running:
  /tmp/modular-app-integrated-build.log. Rusttokenizerparity running:
  /tmp/modular-app-tokenizer-test.log. After PASS, install/testphone via appscript,
  report milestone to user, THEN start Bonsai CLI profile/optimization.
- Initial integrated APK deployed; packaged JNI smoke PASS: Paris answer,
  35.36 tok/s (short app check, not matched tuning benchmark). Appsandbox
  initially killed TCMalloc rseq syscall293 at load. New externalpatch disables
  RSEQ support on Android, using existing thread-cache fallback. Runtime rebuild
  PASS, JNI smoke PASS. ChatDemoTest two UI regressions now running
  /tmp/modular-app-sandbox-ui-tests.log. Final app milestone notification after
  these pass. New source deps/MLX checkpoint in cache/bonsai-hf; 24token FP32
  reference generated cache/bonsai-reference-short.
- Plugin env
  MODULAR_OPENCL_PROFILE=1 logs perkernel GPUevents. Runner args now
  plugin weights prompt.txt reference-or-dash-or-benchmark generation_steps;
  obsolete docs --profile flag should be updated.

## Environment and commands

Repository `/Users/REDACTED/projects/modular`.
Cache `/Users/REDACTED/Library/Caches/modular-android-integration`:
- `native-graph/`: frozen mo-opt/mt/mojo +22 matched .mojoc packages/dylibs,
  `env.json`. Use this across Bazel native/Android configuration switches.
- `snapshot_native.py`: refresh ONLY after successful native full tools build.
- `android-runtime/`: canonical Android normal mt/runtime/plugins, current
  StateSlot/public GPU boundary/addressability fixes. `android_artifacts.json`.
- `compile_graph.py NAME`: source NAME.mlirbc (fallback .mlir) -> normal gc
  pipeline -> MGP -> MEF. Explicit mt `--input-format=mgp` mandatory forlarge IR.
- `block-{s2,s1}`, `model-{s2,s1}` graph sources. MEFs before provenance fix stale.
- `block-reference/{prefill,decode}`, `model-reference/{prefill,decode}` inputs0-2
  and expected0 logits/expected1 cache, generated via independent HF eagerf32.
- `fused_probe.cpp`, `fused-diagnostic/`, `extract_opencl.py`: temporary raw-source
  diagnostic tools only; not a replacement executor. Full stage/log traces here.
- `venv/bin/python`: MAX nightly, Torch/Transformers/numpy. `venv/bin/mblack`.

Native build `./bazelw build/test -c opt --config=hal-device-context --jobs=6
--override_repository=sysroot-macos=/tmp/modular-macos26-sdk ...`.
Android adds `--config=android-arm64 --repo_env=ANDROID_NDK_HOME=<NDK>`.
NDK `/Users/REDACTED/Library/Android/sdk/ndk/29.0.14206865`.
ADB `/Users/REDACTED/Library/Android/sdk/platform-tools/adb` serialDEVICE_SERIAL.
`Target/lib/OpenCL/run_graph.py --runtime <cache>/android-runtime --model ...
--serial DEVICE_SERIAL --adb <adb> --input ...` repeated inputorder0/1/2,
`--output-dir ...`; ordinary mt, no custom executor. Outputs output0/output1.
LLVM22 `/opt/homebrew/opt/llvm@22/bin` clang-format/FileCheck; source compiler
LLVM23/MLIR24 (old opt cannot parse all source-generated floats).

Compiler error logs may contain giant IR on one line; print truncated lines,
never indiscriminately grep/cat entire logs. tmp logs can disappear on restart.

## App build/deployment notes

`/Users/REDACTED/projects/mojo-chat`: preserve staged project/model/tokenizer
and collaborator benchmark. Read AGENTS. Old prepare.py insists MOJOCL01 and
separate model.o/kernels.mcl; new model build has embeddedABI3 and normal runtime.
Old JNI constructor `(plugin,kernels,weights)` vs new `(plugin,weights)`.
CMake old standalone plugin lacks normalCompilerRT. MinSDK31 ->33, JDK21,
normal runtime libs packaging/vendor OpenCL namespace. `.native/OpenCL-Headers`
symlink dangling; stable headers now in cache/OpenCL-Headers. Existing weights
~569MiB/context2048+tokenizer preserved; no redownload needed. App tests preserve
installed model with `scripts/test_device.py --serial ...`. Single JNI worker,
UTF8 parity and last tok/s remains visible after generation.

## Invariants and known unsupported features

MTS/_TensorDataPointer identity device_type; capture bodies compile declared
layout. ABI3 pointer leaves sidecar at alignedPayload+3*originalOffset, total4P.
Optional raw address never materialized from cl_mem. Nonowning borrowed buffers
must retain their real owners through completion. No backend gates in graph
passes; hardware choices stay kernels/driver. Preserve mo.buffer nonaliasing and
MEF GraphRT boundary. No silent CPU fallback. Dynamic alloca, packed bool vector
memory, arbitrary indirect/external calls, fp64/half atomics and safe assertion
termination across barriers unsupported. i64 atomics gated, missing onphone.

Git ignores new `Target/` sources on case-insensitive macOS via `target/` rule;
run_graph/build_runtime/source fixtures exist but may not show in git diff.
Ensure source files aren't lost at eventual commit; don't unignore binaries or
reset/stage collaborator index blindly. Base4d862e6, many preexisting edits.

### App sandbox follow-up

- tcmalloc-android-no-rseq.patch registered in common.MODULE/public-patches.
  Disables RSEQ platform flag Androidwide, existing thread-cache fallback.
  Canonical Android rebuild PASS14s. Stagefixedread-onlyBazeloutputs viaatomic
  replacement; plugins no longer linkedglobally. APK dependency tableverified.
- Packaged JNI smoke PASS, Paris response35.36tok/s. ChatDemoTest conversation/
  stop/clear PASS. Completion/rotationtest reachedcorrectanswer but missed
  fleetingpartialanswer in polling; isolatedrecheck running
  /tmp/modular-app-streaming-recheck.log. Don't weaken productionstreaming or
  addartificialdelays. Kernelprofilingnotstarted untilfirstappmilestonereported.
- Preserved cache/bonsai-baseline APFSclone of integratedbuild+fixedruntime.

### First app milestone COMPLETE; profiling begins

- Integrated APK installedDEVICE_SERIAL. JNIParis smokePASS35.36tok/s.
- Conversation/history/Stop/Clear UItestPASS. Streaming/completion/rotation/rate
  persistence UItestPASS /tmp/modular-app-stream-clock-check.log. Fixedtestclock
  autoadvance whileobservingstream (automaticComposeidling waitedwholeoutput);
  productionstreaming unchanged, noartificialdelay.
- Usernotifiedfirstappmilestonecomplete. Baselinepreservedcache/bonsai-baseline.
- CLI24token referencecheckedGPUprofile started
  /tmp/modular-bonsai-baseline-profile.log. ActualFP32refs/checkpoint pinned
  in cache/bonsai-reference-short and bonsai-hf. Next measure unprofiled baseline
  and identifydominantkernels+queuegaps, thenpreserveonlyvalidatedimprovements.

### Profiling/tuning active (do not stop)

- Baseline profile decode: packed_linear16.91ms, head4.56ms, other4.84ms;
  kernel26.31ms/span27.16ms/token. 24token logits/IDs PASS maxabs.0165186,
  relL2.000786457. Unprofiled validation baseline median33.05tok/s.
- All experiments live in cache only; no sourcewinnerapplied yet. Builder
  cache/build_bonsai_variant.py uses frozencompiler +baseline std/max packages,
  normal NDK C++runner +copied canonicalruntime. Onlymodel/kernel source differs.
- Workgroup128 retained: interleaved baseline32.20, wg64=28.40, wg256=32.33.
- Compact scale-only upload (validateexistingbias=-scale, keepfileformat) no
  speedwin32.53 vsbaseline32.50; do not keepjustforperfwithoutcombinedevidence.
- Standard SIMD2 adjacentoutputs:39.99 vsbaseline32.50 (~23%). SIMD4:45.62
  vsSIMD2=39.89 (~14% additional); SIMD4wg64 slower43.07. Identicalreference
  errors/IDs acrossall. Filescache/bonsai-{vec2,vec4,vec4wg64,...}.
- SIMD8wg64 built, benchmark active /tmp/modular-bonsai-vector-limit-benchmark.log
  vsvec4. Maxnativehalfvector16 means8outputs perthread.
- Further experiment: shared-memory16-entrylookup perinputpair, 4KB/group,
  precompute once perworkgroup then replace halfweightconvert/MAD withFP32
  LUTsums. Standard unsafe_stack_allocation(.SHARED)+barrier, no OpenCLcalls.
  Appliedonlylayers first, baselineSIMD4 head. cache/lookup_bonsai.py modifies
  cache/bonsai-vec4lookup. Initialcompileoriginmismatch fixed pointerparam to
  MutUntrackedOrigin. Buildactive /tmp/modular-bonsai-lookup-build.log.
- Currentbestcandidate SIMD4; need long96token validation/interleavedperformance
  (referencealreadycache/bonsai-reference-96) plusdifferentlongprompt. Then
  profilewinner, applyonlykeptcodeinrepo, canonicalrebuild, appseconddeploy/tests.
- FirstappupdateCOMPLETEalreadyreporteduser. APKstillbaseline35.36JNIshort.
  Profilingphaseongoing. Stillhavenotachieved100tok/s.

### Current resume: numeric records + standard GPU greedy candidate

- SIMD8wg64 won vsSIMD4:47.96 vs45.90, identicalerror/IDs. SharedLUTrejected
  33.4tok/s, though numericallycloser. Head64/256 marginal/noisy; chain2 no
  reliablewinandhighererror. Keep4chains, layers64threads, head128unlesslong
  benchmarkproveshead256gain. Rootmodel/kernelstillbaseline; winnernotapplied.
- Standard GPUgreedyprototype reuses builtin_kernels.rowwise_reductions
  reduce_argmax +TileTensor(DevicePointerEngine) +16B index/score readback.
  Nonfiniteinputs mapto+Inf forselection; originalwinnerbits checkedhost, so
  ANYNaN/Inf remainsanerror. C++CPUfallbackforoptionalHexagonhead maintained.
- Needed two realbackendextensions: OpenCLMaxPlugin.syncwarp_fn -> subgroup
  barrierw64 withlocal/globalfences +convergenceattrs +SG64validation, and
  emitter numericSSArecords (scalarfloat/int fields), explicitLLVMpadding/align,
  Csizeofcheck, inserts/extracts/constants/helperreturns. Packedrecordsnot
  misclassifiedasCAS/sincospairs. OnlyrequiredSGextensionsrequested.
- CanonicalnativebuildPASS /tmp/modular-record-layout-native-build.log.
  7compilerlit+emittertestspassedseparately. Originalnewtestwronglyassumed
  i64alignment8 whiletestDLdefault4; fixedtestexplicitDL. FinalemitterPASS
  /tmp/modular-record-layout-tests.log, plusLLVM22clangallpositivecasesPASS
  /tmp/modular-record-layout-clang-tests.log. Frozen snapshotrefreshedcurrent.
- cache/bonsai-vec8greedy compiledPASSusingfreshfrozen22packages (base max
  lacksnewsyncwarp; builderusesnative-graph importpathONLYgreedyvariants).
  Firstphonebenchmarknowrunning /tmp/modular-bonsai-greedy-benchmark.log.
- benchmark.py --generation-only nowchecksALLtokenIDs but excludes full-logit
  referencechecks, usesrunnerbenchmarkforcedlength. JSONmarksvalidationmode.
  Defaultstillfullindependentalllogitsvalidation. This distinguishesCPUscan
  savingsfromkernelperformancefairly. Bothdefaultandnewmode mustwork.
- Addedlegacycleanupmodel_spec.h removesunusedKernelSpec/kernels table,
  model_forwardmacros andkernelCount; actualweights/bufferspecskept.
- Long96referencefixture ready cache/bonsai-reference-long capacity256,
  longnumbersprompt. Short96 cache/bonsai-reference-96 capacity128alsoexists.
- Next: validateGPUgreedyinclNaN/Inf/ties/oddtails andmeasurematchedgeneration.
  Thenlong96validationbaseline/vec8(+head256ifworth),profilewinner, applyclean
  sourcecode(noallvariants!), canonicalbuild, finalappdeploy/native+UItests.
  Firstappupdatealreadydone/reported. Needfinishremainingperformancework.


### Latest resume (supersedes candidate status above)

- First integrated app update installed/tested/reported; now optimizing CLI.
- SIMD8wg64 long numbers prompt96 tokens all logits/IDs PASS, same error as
  baseline (.0147266 maxabs, .000397484 relL2). Interleaved3-repeat medians:
  baseline28.97 vsvec8=46.65 tok/s. ShortGPUprofile projection16.91→8.89ms,
  head4.56→2.60ms; kernel sum26.31→16.34ms, queue span~17.23ms/token.
- GPUgreedy standard argmax split-K slower (45.63 vsCPU sampling53.66 generation
  only). Edgecases tie/oddtail/NaN/Inf/recovery PASSphone. Root runtime/JNIstill
  baseline CPU sampling. Testing explicit standard rowwise supports_splitk=False
  hint to avoid2temporaryallocs+free_sync pertoken. Public scaffolder forwarded
  newdefaultTruehint; nativebuildactive /tmp/modular-wide-vector-native-build.log.
- Rowwise scratch now carries DevicePointerEngine TileTensor handles host→device;
  pointerunwrap deviceonly. Canonicalbuild+GPUargmaxedgecasesPASSbeforehint.
- Wide MAX65536-row fixture revealedLLVMwidevectors; standardLLVMScalarizer
  added onlyfunctions containing>16lane values, nativeSIMDfunctionspreserved.
  Newemittertest coversboth. Pendingnativebuild+wideMAXphonevalidation.
- Additionalprojection groups2/groups4 perthread candidates being benchmarked
  /tmp/modular-bonsai-groups-benchmark.log. AllfixedBonsaiwidthsmultiple8;
  experimental fallbacktailnotvalidforgroups variants, nottoshipas-is.
- Need selectwinner, applycleanroot kernels/model, canonicalbuild+longvalidation,
  finalapprebuild/deploy/smoke/UItests. Root Bonsai sourceSTILLbaselinecurrently.


### Final source/app build and validation

- Root Bonsai kernels/model now SIMD8 layer64/head128, fourchains, scalar tails.
  Genericunrollinghelper cleaned butsame reductionorder; legacyunusedpacked_dot
  wrapperdeleted. Canonical build.py ->cache/bonsai-final PASS, model.o SHA256
  6379b00a58c75b745c9dd3622d256b7cf0ab251af1114102796879edb95cf119.
- Long96full-logit canonicalmodel validationPASSsamebaseline errors/IDs. Timing
  uncontrolledpower variable; final medians30.87→35.67, genonly33.62→34.68 with
  outliersdown16/29. ObservedphoneDozingafterruns; do NOTdiscardtheseresults or
  callthemsteadyceiling. Rerunningawake3pairedgenonlyrepeats now, log
  /tmp/modular-bonsai-final-awake-generation.log.
- Saved originalstay_on_while_plugged_in=0 in cache/phone-power-restore.json,
  temporarily3 +wake/unlockfortests. MUSTrestore0whenfinished.
- GPUgreedywithoutsplitKstillSLOW42.98vsCPU53.47. Rejected. Removedallpublic
  supports_splitkexperimentchanges (algorithm/rowwise +builtin_reductionsclean).
  NoGPUgreedy/runtime/JNIchangesapplied. Keepusefulcompilerrecords/syncwarp/
  widelegalization andopaque rowwise scratch, validatedbynormalMAXwidegraphs.
- Native rowwisecpu splitaxis+largestate testsPASSaftercorrectCPUPluginenv.
  InitialfailuresmissingMODULAR_DRIVER_PLUGINSsetup; notcodebugs. Latestemitter
  test+7compilerlittestsPASS. Wide65536MAXsoftmaxallvaluesexact,argmax0PASSphone.
- New bonsai/test_packed.mojo verifieslayerpartials/reduce/headandwidths1/7/8/13/16
  with exactnumericchecks acrossbothgroups. PASSphone; canonicalfixturedriver.
- App prepare--reuse-model +JDK21build/lint/test/apkPASS. FinalAPKinstalled;
  packagedJNIFrance smokePASS53.04tok/s. Same model.o hashstagedinapp. Needfinal
  UItests +finishperfdocs, restorepower, conciseledger, finalreport. No100tok/s.


Final awake-screen paired generation median15.13→29.00tok/s (baseline vsSIMD8).
Selected29.04/28.95/29.00, baseline16.08/15.10/15.13. WarmupSIMD52.23.
Thermalstatus0andAwakeconfirmed; wakealone doesnotexplainoperatingratevariation.
DoNOTclaimsteadyrate53 orcausallyblameDozing. PERFORMANCE.mdkeepsallbatches.
FinalAPK JNI53.04shortPASS; UIclassrunning /tmp/modular-app-final-ui.log.
Originalstayawake0stillmustrestore. Nativealltests/model/tailsdone. Do not rerun
hardwarebenchmarkunnecessarily; finishUIandledger/docs, finalreportactualrange.
