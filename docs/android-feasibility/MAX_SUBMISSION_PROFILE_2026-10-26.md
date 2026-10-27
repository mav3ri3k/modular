# MAX submission profile — 2026-10-26

## Result

Correction: the first comparison used a cached MAX graph from 2026-10-25. That
graph still called the untiled packed projection kernel, while the direct
runner used the newer tiled layout. Its event-gap and host-phase results below
are preliminary and must not be attributed to the latest MAX graph.

I updated the MAX custom op to use `packed_linear_tiled`, changed its X grid to
the matching 32-output tiling, and repacked the selected packed matrices once
when loading the checkpoint. The current tiled MAX graph build has MEF SHA-256
`8e9c6e4345fef199436e9db823d4ad38a62caf10607a7e925537ecaed14b9f04`; its
kernel source SHA-256 is `ce86bb461289a51824e397b2983204a8b60646e8ed76be81ee625e09193535ad`.
The direct runner is the 78.10 tok/s artifact (SHA-256
`969bbb3fbff5f2bd7faac5f32b3ab7aa191a4a3e40214465df7be00507ba6c82`). Both
used the same instrumented HAL build (SHA-256
`8402682d5829251e7df62f140050cdd84d147be20f5e4efbd881dcd9beb0b24d`).

On the Galaxy S26 / Adreno 840, the rebuilt graph passed all expected token ID
and full-logit checks. In a 95-forward, generation-only paired run, the three
non-warmup rates were 67.20/67.88/69.08 tok/s direct and 70.61/70.91/71.21
tok/s MAX (medians 67.88 and 70.91). A separate three-run check including
full-logit readback/validation passed, but rates varied substantially and are
not used as a throughput comparison. The direct warmup reached 78.53 tok/s;
this short run did not sustain that peak.

With GPU event profiling enabled on both artifacts, median per-token results
for the last 95 decode forwards were:

| Artifact | Dispatches/token | GPU kernel sum | GPU event span | Span minus kernel sum |
| --- | ---: | ---: | ---: | ---: |
| Tiled direct runner | 721 | 11.83 ms | 13.60 ms | 1.82 ms |
| Rebuilt tiled MAX graph | 599 | 11.55 ms | 14.96 ms | 3.38 ms |

The matched tiled graph therefore reproduces an extra **1.55 ms/token** of
event span beyond kernel time. This run had one profiled sample per artifact;
profiling reduced throughput (67.75 tok/s direct, 57.38 MAX), so those rates
are not benchmark results. Per-event queued-to-submitted and submitted-to-
started totals overlap and are not wall-time buckets.

With host phase timing enabled and the same HAL in each runner, median times
per decode token were:

| HAL phase | Tiled direct | Tiled MAX | MAX minus direct |
| --- | ---: | ---: | ---: |
| `M_driver_queue_execute` host time | 10.02 ms | 9.33 ms | -0.69 ms |
| `clSetKernelArg` | 0.62 ms | 1.27 ms | +0.65 ms |
| `clGetKernelInfo` | 0.19 ms | 0.26 ms | +0.06 ms |
| Buffer retains | 1.43 ms | 0.75 ms | -0.68 ms |
| `clEnqueueNDRangeKernel` | 5.80 ms | 4.48 ms | -1.32 ms |
| Other launch preparation | 1.89 ms | 2.52 ms | +0.63 ms |
| Blocking device-to-host copy | 2.60 ms | 2.75 ms | +0.15 ms |
| Queue sync after readback | 0.002 ms | 0.002 ms | negligible |

MAX still issues 14,545 argument-setting calls per token versus 5,452 direct,
but the broader HAL queue-execute time is lower for MAX in this matched run.
That means the 1.55 ms additional event gap is **not explained by time inside
the measured HAL launch calls alone**. The leading remaining location is
between those calls in MAX graph task scheduling, worker wakeups, or dispatch
handoff. Those runtime intervals need a host scheduler trace to separate them.
Host-phase timing also adds measurement overhead, so treat its numbers as
diagnostic rather than wall-time components.

## Follow-up: where the extra event gap lands

The HAL event logger was extended to split each adjacent-event interval into
queued-to-submitted (`submit_delay`), submitted-to-started (`device_wait`), and
queued-to-first-API-enqueue (`late_queue`) time. In the matched tiled run of 95
decode forwards, median per-token totals were:

| Event interval component | Tiled direct | Tiled MAX | MAX minus direct |
| --- | ---: | ---: | ---: |
| Late queue arrival | 0.000 ms | 0.000 ms | 0.000 ms |
| Queued to submitted | 0.445 ms | 1.531 ms | +1.086 ms |
| Submitted to started | 1.448 ms | 1.767 ms | +0.319 ms |
| Total event gaps | 1.935 ms | 3.346 ms | +1.411 ms |

This pins the dominant measured component: MAX commands spend about **1.09
ms/token longer queued before the driver reports them submitted**. That argues
against the host simply arriving late with the next enqueue, and against a
single long GPU execution stall. The remaining ~0.32 ms/token is a smaller
increase in submitted-to-started wait. These are OpenCL event timestamp
intervals, which can overlap and are not end-to-end wall-time buckets.

The host-side interdispatch timer, measured from the previous HAL execute
return to the next execute arrival, was about 1.17 ms/token direct and 1.91
ms/token MAX. Meanwhile total HAL queue-execute time was 10.32 ms/token direct
and 9.36 ms/token MAX. This makes runtime task scheduling/handoff between
launches the best next source-level target, while showing that the broad HAL
launch function itself is not slower overall. MAX also has more argument
setting overhead (about 1.3 vs 0.69 ms/token) but fewer launches and lower
total queue-execute time, so reducing call counts or argument setup alone is
not yet justified as the main fix.

The largest per-transition medians reinforce a distributed handoff cost rather
than one pathological command: `Hadamard -> PackedLinear` was 0.525 ms MAX vs
0.311 ms direct; `CacheKv -> AttentionScores` 0.180 vs 0.080 ms;
`DeltaUpdate -> GatedNorm` 0.120 vs 0.032 ms; and
`AttentionScores -> AttentionValues` 0.101 vs 0.036 ms. Some transition labels
are not one-to-one between the runners, so use these as candidate boundaries,
not direct kernel comparisons. Raw logs:
`/tmp/bonsai-max-gap-components-profile-{mojo-0,max-1}.log`.

Two follow-up probes narrow the patch direction:

- `MODULAR_OPENCL_FLUSH_AFTER_ENQUEUE=1` (flush after every MAX kernel) cut
  unprofiled throughput to about 45 tok/s from about 71 tok/s. Under profiling,
  it increased MAX event gaps to about 12.94 ms/token, mostly submitted-to-
  started waiting. Per-kernel flushing is therefore rejected; it worsens the
  scheduling/device handoff and is not a candidate fix.
- The first AsyncRT/GraphRT trace attempt aborted during CPU device
  initialization: the runner creates a CPU HAL context before the MAX context,
  so `MODULAR_PROFILE_FILENAME` made the later context request profiling
  options that did not match the already-created process-wide CPUDevice. For
  diagnostic builds only, I propagated that filename into the CPU HAL context,
  rebuilt, and restored the source/config changes after packaging the binary.
  The retry completed and validated every generated token ID.

### AsyncRT/GraphRT trace

The level-3 AsyncRT / level-1 GraphRT trace covers one 95-forward decode sample
on the rebuilt tiled graph. Trace-instrumented throughput was 69.42 tok/s; do
not treat that as a benchmark. The JSON and text traces are at
`/tmp/bonsai-max-runtime-trace-level3.json` and
`/tmp/bonsai-max-runtime-trace-level3.time-events.txt`.

In the trace's 1.352 s steady decode window:

- The main `graph_runner` thread entered `grt.invoke` 95 times. The scopes sum
  to 1.69 ms total (16.9 us median), so the GraphRT invocation call itself is
  not where the per-token interval is spent.
- One AsyncRT worker (`Thread1`) ran all 3,666 `asyncrt.doWork` scopes: 38.6
  tasks/token, 1.029 s summed task-scope time, 290 us median and 346 us p95.
  The other seven worker threads were idle during the steady decode window.
- `asyncrt.waiter` appeared 3,760 times on `Thread1`, with 289 ms summed
  scope-time. These spans overlap task execution and must not be added to the
  `doWork` total or interpreted as 289 ms of pure waiting.
- Worker spin scopes totaled 18.9 ms and sleep scopes 13.4 ms for `Thread1`.
  The long intervals between tokens are therefore not explained by a worker
  sleeping for milliseconds; the repeated serialized task/continuation path is
  the more relevant scheduling lead.

This trace narrows the runtime hypothesis: graph invocation overhead is tiny,
while the dependency-driven decode runs through roughly 39 AsyncRT task scopes
per token on a single worker. It does not prove that using more workers would
help—the graph is autoregressive and many tasks may be dependent—and the task
scopes include useful host launch work that overlaps the GPU. Together with the
OpenCL timestamps, the actionable target is reducing the queued-to-submitted
delay across this repeated worker/continuation handoff path. The next useful
measurement should identify which task labels or event waits account for that
time before changing scheduling policy or batching tasks.

### Worker scheduling patch experiments

I tested two opt-in changes on the same tiled graph and phone. Both runs used
generation-only validation, which checks all expected token IDs.

- **Round-robin device tasks across all workers:** this directly tested whether
  waking the idle workers could improve handoff latency. With the flag off,
  three unprofiled rates were 69.61, 69.76, and 70.07 tok/s (median 69.76).
  With it on, rates were 27.57, 28.55, and 29.00 tok/s (median 28.55). This
  change is rejected. Migrating each dependent device task to a different
  worker adds substantial queue/wakeup or context handoff cost; the serialized
  decode chain does not gain useful parallelism from spreading those tasks.
- **Inline async device work when already on its assigned worker:** this
  removes the enqueue/pop handoff without changing worker affinity. The
  unprofiled median was 69.50 tok/s with the option off and 70.13 tok/s on
  across three runs each (off: 69.98/69.50/69.24; on: 70.80/70.13/68.11).
  The small median difference is within the run-to-run spread. A single event-
  profile sample also showed no reduction in the aggregate queued-to-submitted
  delay; that profile includes prefill and instrumentation overhead, so it is
  only a diagnostic check. This change is not retained.

Neither scheduling patch produced a reliable speedup, so both code changes and
their temporary runner flags were removed. The trace confirms that one worker
is active, but that is not itself a performance defect: the graph's dependent
tasks become a serial chain, and forcing them across workers is much slower.
Keep the next work focused on identifying which specific task or OpenCL driver
submission step holds commands in the queued state. Any future scheduling
change should preserve affinity for dependent device work and be accepted only
after repeated unprofiled decode runs improve.

## Earlier host CPU sample (cached untiled graph)

`simpleperf record -g -f 400 --delay 1700` collected 686 direct-runner and 926
MAX-graph user-space CPU-cycle samples during the run, with no lost samples.
Kernel samples were unavailable on the non-rooted device. In the inclusive
call-tree report, `M_driver_queue_execute` accounted for 38.9% of direct and
37.2% of MAX samples. The report also shows:

- `clEnqueueNDRangeKernel`: 25.1% direct, 18.0% MAX.
- `clSetKernelArg`: 2.4% direct, 3.5% MAX.
- HAL buffer retain/release helpers: roughly 1–2% each.
- `M_driver_bundle_load` and `clBuildProgram` still appear prominently in both
  samples, indicating lazy program materialization overlaps the sampled
  window. These inclusive percentages are not elapsed-time buckets and must
  not be added.

This identifies launch/enqueue as a major host activity, but does not assign a
millisecond cost to argument encoding, buffer validation, or scheduling. The
existing HAL API-boundary profile remains more useful for absolute call time:
17,969 enqueues took 140.1 ms total (7.8 us mean, 12 us p95), and 703
`clFinish` calls took 18.5 ms total. Those measurements are from the earlier
baseline profile, not the tiled direct runner.

## Earlier HAL phase timers (cached untiled graph)

I rebuilt the current HAL with `MODULAR_OPENCL_HOST_PROFILE=1` timing counters
from the same checkout (instrumented plugin SHA-256
`8402682d5829251e7df62f140050cdd84d147be20f5e4efbd881dcd9beb0b24d`) and ran
95 decode forwards without GPU event profiling. The diagnostic run measured
75.71 tok/s direct and 53.58 tok/s MAX; those rates include timer overhead and
are not benchmark results. Each direct-runner
token made 721 launches; each cached MAX-graph token made 599. Median phase
times per token were:

| HAL phase | Direct runner | Cached untiled MAX | MAX minus direct |
| --- | ---: | ---: | ---: |
| Total `M_driver_queue_execute` host time | 8.97 ms | 11.07 ms | +2.10 ms |
| `clGetKernelInfo(CL_KERNEL_NUM_ARGS)` | 0.16 ms | 0.29 ms | +0.13 ms |
| `clSetKernelArg` | 0.55 ms | 1.53 ms | +0.99 ms |
| Buffer retains | 1.27 ms | 1.02 ms | -0.25 ms |
| `clEnqueueNDRangeKernel` | 5.30 ms | 5.26 ms | -0.04 ms |
| Other launch work, including argument/view handling | 1.59 ms | 2.84 ms | +1.25 ms |
| Blocking device-to-host copy | 2.71 ms | 3.43 ms | +0.72 ms |
| Queue sync after the blocking copy | 0.001 ms | 0.003 ms | +0.002 ms |

MAX made **14,545 `clSetKernelArg` calls per token** (24.3 per launch), versus
**5,452** (7.6 per launch) for the direct runner. The measured extra
`clSetKernelArg` plus kernel-metadata time is **1.12 ms/token**, which matches
the scale of the original 1.1 ms question. The broader launch path is about
2.1 ms slower, with most of the remainder in non-OpenCL argument/view work.
Queue-lock wait is only about 0.04 ms/token in each path. `clFinish` itself
does not show a material wait after the blocking readback.

These phase totals are host elapsed time and can overlap GPU execution; they
are not additive with the GPU event sums. This untiled graph comparison is
superseded by the matched tiled comparison above.

The current Qualcomm recording path was also tried with capture enabled. The
device exposed the QCOM recording entry points, but the first capture failed
with OpenCL error `-50` and the HAL fell back to ordinary submission. No
steady-state replay timings were collected. So these results measure the
normal `clSetKernelArg` path, and the recording implementation needs a
separate capture-error investigation before it can test the expected saving.

## Earlier interpretation (superseded)

The earlier untiled comparison appeared to implicate repeated argument setup.
The matched tiled run above shows that conclusion was too strong: argument
setting is slower in MAX, but total measured HAL launch time is not.

The Qualcomm recording path in the current checkout was not enabled for these
runs. The profile run sets `MODULAR_OPENCL_PROFILE=1`, which disables capture
by design. This avoids mixing initial recording/capture work into the
measurements.

Matched tiled rerun records and raw logs are under `/tmp/bonsai-max-tiled-*`;
the original preliminary logs are under `/tmp/android-current-hal-*` and
`/tmp/android-hostprof-*` on the development machine. They are not committed.
