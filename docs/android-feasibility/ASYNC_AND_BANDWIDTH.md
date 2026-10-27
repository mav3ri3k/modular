# Bonsai async, CPU traffic and bandwidth diagnostic

Historical diagnostic before the [hill climb](HILLCLIMB_LOG.md): normal OpenCL
side-stream setup has since been enabled; current weights are compact scales.

2026-10-25. Diagnostic-only pass: no compiler/backend/model/app implementation
changes, phone policy changes or new benchmark kernels. Inspected the existing
compiled MAX graph and source; built/ran upstream clpeak outside the repository.

## CPU transfers

Inspected native dump of cache/bonsai-max-repro/bonsai-lowered.mlirbc:

- `bonsai_decode$init`: 551 external constants and 551 sync summaries associated
  with initialization transfers. These are not repeated per token.
- `bonsai_decode$exec`: 45 generic host-runtime calls, including 37 async compute
  sequences, one GPU arena allocation and one CPU output allocation. One
  `mgp.buffer.device_to_host`, then one `mgp.sync`. No intermediate model tensor
  or weight host/device transfer in the steady-state graph.
- CPU token/position inputs are small control values. Recurrent/conv/KV state
  remains in the owning GPU input buffers across invocations.
- Remaining CPU round trip: all248320 FP32 logits (993280 bytes) return every
  step; C++ scans/selects the next token. The runner calls M_executeModelSync.
  The next token depends on this selection. Full reference-logit checks are
  benchmark-only; generation-only timing still has readback and CPU sampling.
- Artifact manifests contain no device-error-flag kernels (plain21 bundles,
  MAX31); the plugin's optional4-byte assertion-status readbacks therefore do
  not add per-kernel traffic for these artifacts.

This is a lowered-graph/source audit, not an instrumented runtime count of every
OpenCL API call. No CPU math fallback was found; CPU scheduling/stub execution
is distinct from transferring activations to CPU for computation.

## Async and overlap

- GPU launches call clEnqueueNDRangeKernel without clFinish. Pure device copies
  and fills enqueue normally. Cross-queue waits use OpenCL event/barrier wait
  lists without a host wait. Basic asynchronous submission exists.
- Queues are in order: creation flags request profiling only, never out-of-order
  execution. Ordinary serial queue ordering preserves dependent kernel work.
- H2D/D2H transfers use CL_TRUE, blocking the calling host thread. Copying via
  mapped pinned host storage is also blocking in this implementation. The
  scheduler cannot turn a blocking HAL transfer into a nonblocking one.
- Explicit synchronization maps to clFinish. The steady-state graph has its
  final sync after an already blocking output readback. It has no explicit
  per-kernel mgp.sync. memory_free_sync releases cl_mem; its name must not be
  mistaken for an unconditional clFinish.
- Concrete remaining integration gap: MEFModel.cpp's
  ensureStreamCountForDevices pre-creates side streams only for CUDA/HIP
  (lines170–173), skipping OpenCL. The HAL has explicit stream/event machinery,
  but this normal model-setup path does not activate the side-stream pool for
  OpenCL. Removing this gate alone would not establish useful concurrent work.
- Optional graph capture remains absent; external-stream import and some
  stream-level interfaces remain unsupported. These are advanced overlap/
  submission capabilities, not evidence that all async execution is broken.

Weights are already resident in GPU-accessible memory. Streaming weights from
DRAM into the GPU's registers/cache occurs inside the projection kernels; it
is not a CPU-to-GPU upload for the MAX scheduler to pipeline per layer. That
latency is hidden through kernel instruction scheduling, memory access, reuse
and occupancy. Independent graph branches could potentially overlap on multiple
queues, subject to device resources and shared bandwidth. Autoregressive tokens
still depend on the preceding token's result.

Earlier paired trace: Mojo/MAX GPU sums16.45/16.50ms, spans17.35/19.89ms. Extra
MAX gaps remain a measured symptom. This audit narrows candidates but does not
attribute those gaps to a particular wait or prove a speedup from multi-stream.

## Bandwidth measurement

Adreno840, current phone policy, no clock/thermal overrides. Upstream clpeak
1.1.2 commit a4d617d7d054450aa83965d0fd9d62ee2009957f; Khronos CLHPP headers
14f76ab0e180fe1b0e0591d52497ebc88f86ce5e. Existing global-memory bandwidth test,
50 iterations, best of two access patterns for each vector width:

| Width | Device-event run1 | Device-event run2 | Queued-batch wall timer |
| --- | ---: | ---: | ---: |
| float | 63.80 | 63.97 | 63.85 |
| float2 | 65.74 | 65.78 | 65.67 |
| float4 | 67.60 | 67.71 | 67.54 |
| float8 | 68.22 | 68.20 | 68.03 |
| float16 | 42.42 | 42.71 | 42.60 |

Units GB/s (decimal). Event mode finishes each iteration but measures device
START/END; wall mode enqueues/flushes the batch then finishes once. This is
streaming read throughput, not measured Bonsai DRAM traffic, a chip specification
or a long-duration thermal ceiling. Vector-width sensitivity is real; it does
not identify the packed kernels' register pressure without counters.

Current exported tensors total596106496 bytes: codes470024192,
scale/bias117506048, signs1630208, other6946048. A rough0.6GB one-pass weight
stream at68GB/s costs8.8ms, leaving little of a10ms token budget for other work.
This is a budget estimate; actual cache/DRAM traffic is unmeasured. Async alone
cannot establish100tok/s with the current format. Reduced traffic/representation
and GPU work remain plausible requirements.

## Counter access and artifacts

- Perfetto query exposes android.gpu.memory but no gpu.counters or GPU render
  stage source on this phone. GPU instruction/stall/cache counters were not
  obtained. This does not mean the hardware lacks them.
- Vendor qprof execution is permission-denied from adb shell. Snapdragon
  Profiler/Qualcomm profiling-library setup is a separate follow-up.
- CPU userspace PMU counters via simpleperf work; kernel samples are restricted.
  This pass did not capture a model CPU stack trace.
- Device thermal status reported0 afterward. No phone power settings changed;
  original stay_on_while_plugged_in remains0. Diagnostic binaries removed from
  /data/local/tmp; source/build/results remain outsidegit in the integration
  cache. Raw logs /tmp/modular-diagnostic-clpeak-run{1,2,3-wall}.log; parsed
  cache/diagnostic-bandwidth-results.json. Dump cache/diagnostic-bonsai-lowered.mlir.

References: [clpeak1.1.2](https://github.com/krrishnarraj/clpeak/tree/1.1.2),
[Perfetto GPU data sources](https://perfetto.dev/docs/data-sources/gpu),
[Qualcomm profiler](https://www.qualcomm.com/developer/software/snapdragon-profiler).
