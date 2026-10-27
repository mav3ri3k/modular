# Android integration working ledger

Follow-up: [batch-one HVX experiments](hexagon-decode/README.md) add packed HVX
GEMV, event-scoped head overlap, cooperative normalization, and reproducible
hybrid benchmarks. GPU-only remains default; all final samples and reference
checks are recorded. VTCM residency experiments are excluded from source.

Updated 2026-10-25. The integration/profiling pass and subsequent Bonsai MAX
graph comparison are complete. Earlier evidence and
investigation: [history](INTEGRATION_HISTORY.md). Performance protocols and all
measured batches: [Bonsai performance](../../max/examples/android-llm/bonsai/PERFORMANCE.md).

## Objective and outcome

User order: (1) finish standard tensor storage, (2) validate ordinary MAX
transformer/model execution, then update app; profile/improve Bonsai and update
app again. Both app updates were deployed/tested and reported. No commits made.
Preserve collaborator changes/index. No subagents. Phone tests authorized.

## Current hill climb (2026-10-25)

Agent memory: [completed experiments](HILLCLIMB_LOG.md), [ranked queue](HILLCLIMB_QUEUE.md).
Final selected-source medians59.11tok/s Mojo /56.43 MAX (3 alternating95-forward
runs after cooling), every96 IDs match; full-logit validation separately passes.
KEEP exact FP16 scales only (validate roundtrip; same centered4-chain arithmetic).
Weights596.1→508.0MB; 95-forward paired medians Mojo53.25→58.07 (+9.1%),
MAX49.23→53.11 (+7.9%). Full short/long logits and every ID pass. App unchanged.
MAX stores scale bits as uint16 external constants, ordinary hoisted transfers,
Float16 pointer view only inside custom GPU closure. Original checkpoint importer
still validates centered FP32 scale/bias then compacts; no new artifact version.

OpenCL now included in normal side-stream setup, but no automatic concurrency
speedup established. Vector comparison masks fixed (narrow lanes), generated-C
regression passes. Stdlib runtime getter reuses embedding options; unit passes.
Device-first C API still initializes CPU defaults before configured MAX runtime:
worker-count sweep blocked; do not claim fixed by stdlib change alone.

Rejected K-lane/SIMD4/8-chain/affine/int16/bit-logic variants (latter can stall).
Larger host sequence limits did not win reproducibly; compiler option reverted.
Final paired trace: Mojo14.987ms GPU/15.893span vs MAX15.170/16.092, both~.9ms gaps.
Earlier additional MAX2.55ms internal gaps did not reproduce. CPU sampling scans
cost~.8ms/output; combined greedy helper reduces stage to~.25–.30ms, tie/nonfinite
checks pass. ThermalStatus1 confounded later wall timing; no added speed claim.
Artifacts cache/bonsai-hill-{final,max-final}; compact math evidence in
bonsai_graph/results-hillclimb-2026-10-25.json. 100tok/s not reached; no hardware
counter claim or assertion kernels are close to device limits. Counter permission
follow-up RESOLVED: shell can temporarily enable KGSL perfcounter sysfs switch;
all29 READs work around95-forward Bonsai runs;7 selectors nonzero. Restore0 after
capture. Device-specific event/pipeline mapping still needs calibration. Wrapper:
max/examples/android-llm/with_gpu_counters.py; details in HILLCLIMB_LOG.md.

## Follow-up: Bonsai MAX graph now runs

Source: max/examples/android-llm/bonsai_graph/{build,export_graph,run}.py,
ops/model_ops.mojo, runner.cpp, README.md and results-2026-10-25.json.
Full24 layers, packed weights unchanged, persistent GPU state as explicit
BufferType inputs, inplace_custom effects. Standard MAX RMSNorm/residual/SwiGLU/
FP32 reductions; registered custom ops for packed/Hadamard/recurrent/attention
kernels. No whole-model custom forward, global state or custom graph executor.
MAX C API loads normal Android MEF; C++ does I/O/sampling only.

Phone all24 greedy IDs +every FP32-reference logit PASS. Maxabs .0197719,
relativeL2 .000812919. Three alternating repeats +warmup each, 24 outputs/23
forwards, capacity128: generation-only Mojo53.74 vsMAX49.90tok/s (-7.1% MAX);
fullvalidation47.76 vs42.90 (-10.2%). Every generation-only ID checked too.
Paired GPU profile: 721→599 dispatches, 16.45→16.50ms GPU sum, span17.35→19.89ms.
Fusion works; extra queue gaps offset it. Submission/scheduling/synchronization
is next to investigate, not a proven individual root cause or hardware ceiling.

Build/run CLI independently reproduced and phone-tested. Canonical runtime
stager has --include-max-c-api; Android libmax built successfully (~513MiB,
compiler included). MAX path is a benchmark bring-up; app retains prior engine.
Tool packages need separate native/Python-frontend builds from SAME source due
incompatible compiler checksums; no checks bypassed. Native gc pipeline imports
its own custom declarations via kernel-library-paths. Build stages existing
bonsai/kernels.mojo as package (no authored duplication) then adapters. C API
weights registry is HOST backed: constant_external must be CPU then .to(gpu),
which MAX hoists to initialization. Incorrect GPU declaration was rejected by
opaque encoder on embedding weight, corrected; no backend workaround added.
Generated packages/model/runtime outsidegit undercache/bonsai-max-repro.
Profiles/reports cache/bonsai-max-{comparison,generation-comparison,paired-profile-summary}.json.
Temporary phone directories cleaned; no phone settings changed in this pass.
Preexisting untracked probe files in bonsai_graph retained. No commits/staging.

## Diagnostic follow-up: async and bandwidth (no implementation changes)

See [full audit](ASYNC_AND_BANDWIDTH.md). Weights/state resident; steady MAX graph
has only final993280B logit D2H +sync, no intermediate CPU tensor transfers.
37 async compute sequences among45 host-runtime calls. Kernel/D2D enqueue async,
but host/device copies blocking; queues in order. Concrete gap: model setup
pre-creates side streams for CUDA/HIP only, skips OpenCL (MEFModel.cpp170–173).
Existing HAL stream/events do not imply automatic MAX overlap is fully wired.
Weights DRAM→register/cache streaming is kernel work, not per-layer H2D uploads.
Existing upstream clpeak3runs: float8~68.0–68.2GB/s, float4~67.5–67.7;
float16~42.4–42.7. Rough0.6GB weights/68GB/s ~8.8ms; actual model DRAM counters
unknown. Perfetto only GPU memory source; qprof shellpermissiondenied. CPU PMU
works. No GPU counters acquired, no claim of kernel stall/occupancy root cause.
Phone diagnostic binaries cleaned, power settings untouched. No backend/app edits.

## Integration implemented

- ABI3 only, old formats removed. Typed aggregate provenance survives lowering
  before host C ABI coercion (`usesHostCABI=false`); SDK encodes opaque leaves.
- Normal MODULAR_DRIVER_PLUGINS, DeviceContext, shared CompilerRT/AsyncRT/GraphRT,
  canonical Android NDK/Bazel cross toolchain. No example launch bridge.
- Opaque handles, checked signed offsets/extents and ownership through graph
  allocation, views, transfers, public GPU input/output and mt readback.
- MTS.to_tile_tensor uses GraphStorageEngine, including TensorOps and engine-
  generic wrappers. Pointer resolution happens on device after argument encoding.
  DevicePassable/FusionPack reflect target layout and encode each field.
- Associated storage-field lifetime tracking corrected. Do not add speculative
  ArcPointer retains: old fixture erased origins and lost owners at last use.
- StateSlot presence uses ownership; scatter/BMM preserve engines and storage.
- Shared LLVM optimization/generic TTI/SLP; collectives convergent before passes.
  SIMD, private alloca, shared memory, barriers/subgroups, CAS/FP32 atomics,
  gated i64, scalar-field numeric SSA records. Wide-vector functions use LLVM
  Scalarizer; native-width functions retain SIMD. GEP uses real pointer
  arithmetic (integer round trips broke Adreno cross-warp local aliasing).
- Rowwise split-K scratch is DevicePointerEngine TileTensor host transport;
  unwrap only inside device kernels. Nullable views and async software device
  errors work. Assertions with barriers still fail closed.
- Android TCMalloc uses existing non-RSEQ fallback: rseq is killed by app
  seccomp even though shell tests allow it. Registered public dependency patch.

## Validation

Phone DEVICE_SERIAL, Galaxy S26 SM-S942U1 / Adreno840 / Android37:
- Graph storage static/dynamic tiles, TensorOps, signed shifts, borrowed wrappers,
  pair/scalar arguments; paged KV packed/padded layouts, LUT/layers: PASS.
- Normal MAX matmul, RMSNorm, softmax, slice/concat, public GPU I/O and multiwarp
  fused residual/RMS576: PASS. Wide65536 softmax exact all values; argmax tie0 PASS.
- Normal MAX SmolLM2 real weights, 30 layers, every logit/cache checked:

| Run | Max logit error | Max cache error | Greedy actual/reference |
| --- | --- | --- | --- |
| block prefill | 3.82e-5 | 1.91e-6 | 365/365 |
| block decode | 7.63e-5 | 1.91e-6 | 216/216 |
| full prefill | 1.86e-4 | 8.78e-5 | 198/198 |
| full decode | 5.73e-5 | 8.78e-5 | 1/1 |

Both decodes consume actual phone prefill cache; full relative L2 <2.4e-6.
Reports cache/{block,model}-phone/{prefill,decode}/validation.json. Exporter
--intermediates + verify_graph.py support reproducible debugging/all-value checks.
- Native DevicePassable, MTS/FusionPack, opaque alignment, ownership/parser,
  StateSlot, rowwise CPU split-axis/large-state, 7 OpenCL lit + emitter tests PASS.
  CPU tests need real native CPUPlugin via MODULAR_DRIVER_PLUGINS. Initial missing
  env failures were corrected; do not mistake Android CPUPlugin for macOS one.

## Bonsai selected implementation and measurements

- Root kernels/model compute8 adjacent outputs/thread using standard SIMD/AFN,
  four FP16 chains, FP32 scale/reduction; layer64/head128 threads. No explicit
  model device calls or global fast-math override. Scalar tails retained.
- Canonical build.py --variant bonsai -> cache/bonsai-final. Embedded model.o
  SHA256 6379b00a58c75b745c9dd3622d256b7cf0ab251af1114102796879edb95cf119.
- Long numbers96 tokens all IDs/logits PASS, same errors as baseline:
  maxabs .0147266, relative L2 .000397484. SIMD8 also passes opposite24/96.
  New bonsai/test_packed.mojo checks exact partials/reduce/head at widths
  1/7/8/13/16 across two scaled groups: PASSphone.
- Initial paired long validated median28.97→46.65tok/s. GPU profile projection
  16.91→8.87ms; head4.56→2.60ms; total26.31→16.33ms, span17.22ms/token.
- Final canonical validation medians30.87→35.67; generation-only33.62→34.68.
  Rates varied significantly. Phone found Dozing; screen-awake repeat still
  15.13→29.00, thermal status0. Cause of operating-rate variation NOT established.
  Keep ALL samples; no claim that awake alone fixes it or 53 is sustained.
- Final packaged APK JNI France smoke PASS53.04tok/s (short). 100 NOT reached.
- Rejected scale compaction, shared LUT, two chains, alternate group/workgroup
  geometries. Longer finalist medians47.72/47.94/47.70 indistinguishable.
  Standard GPU argmax correct but slower (45.63 vsCPU53.66; no-splitK42.98 vs53.47).
  Prototype NOT shipped; public rowwise hint experiment fully removed.

## App and final verification

/Users/REDACTED/projects/mojo-chat now packages canonical runtime libs and
ABI3 model.o; HAL plugins in jniLibs, not globally linked. prepare --reuse-model
preserves existing context2048 weights/tokenizer; no separate kernels.mcl.
JDK21 Gradle build/lint/unit + tokenizer parity PASS. Final APK installed.
JNI smoke PASS. UI history, Stop/Clear, streaming, completion/rotation and stable
last-rate tests PASS (2 tests, /tmp/modular-app-final-ui-recheck.log). Both live-
stream tests explicitly step Compose frames to avoid auto-idling through the
whole generation; production streaming unchanged. Original phone stay-awake
setting0 RESTORED/verified; cache/phone-power-restore.json records restoration.

## Resume/tooling and remaining limits

Cache = ~/Library/Caches/modular-android-integration. native-graph has frozen
mojo/mo-opt/mt,22 matched packages,dylibs/env.json; snapshot_native.py refreshes
only after successful native full build, with no active readers. Use venv/bin/
python (UV3.13); Xcode Python strips inherited DYLD_LIBRARY_PATH. Native Bazel:
-c opt --config=hal-device-context --jobs=6
--override_repository=sysroot-macos=/tmp/modular-macos26-sdk.
Android adds --config=android-arm64 and --repo_env=ANDROID_NDK_HOME=NDK.
NDK ~/Library/Android/sdk/ndk/29.0.14206865; adb SDK/platform-tools/adb.
App JDK /opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home.
Never print gigantic MLIR error lines: match diagnostic prefixes and bound count
and line length. Use mt --input-format=mgp for large bytecode.

Tested integration is complete, not universal MAX operator coverage. Bonsai now
has both standalone Mojo and MAX graph paths; neither implies full serving/
batched-prefill architecture support. Optional HAL
DeviceGraphBuilder.add_function capture is absent (blocks14 broader tests).
Dynamic alloca, packed bool-vector memory, arbitrary external/indirect calls,
fp64/half atomics, safe assertion termination across barriers unsupported;
i64 atomics gated/missing onphone. Keep hardware gates out of graph passes,
preserve mo.buffer nonaliasing and MEF boundary. Nonowning erased views need
owners through completion. No silent CPU fallback or raw opaque unwraps.
Git may ignore new Target sources via case-insensitive target/ rule: preserve
source fixtures/scripts explicitly at eventual commit, never generated binaries.

### Adreno840 calibrated profiling

[Counter calibration](COUNTER_CALIBRATION.md) now validates GPU-interface byte counts and resident wave contexts. MAX decode moves about659MB/token. Next targets are projection live state/global partials and recurrent-update grid parallelism. ALU utilization, cache rates, spill counts and stall percentages remain unvalidated; do not infer them from responsive selectors alone. No production kernel changes in this calibration pass.

### Counter-guided kernel integration — selected

Eight-group projection partial joins and eight-way recurrent K splitting use
standard Mojo subgroup operations, existing DeviceContext launches and normal
MAX custom ops. MAX retains its ordinary sum for the remaining group join;
partial TensorTypes and direct scratch allocation both shrink8×. No backend,
raw-pointer transport, compiler target intrinsic or host tensor trip added.
Cooled MAX median64.35 vs56.84tok/s; GPU12.94 vs15.86ms/token; interface traffic
578.65 vs659.01MB/token. Correctness/source hashes: counter-tuning/summary.json.
Register pressure remains unresolved; next priority is packed arithmetic/live
state and exposed submission gaps, not another blind side-stream increase.
