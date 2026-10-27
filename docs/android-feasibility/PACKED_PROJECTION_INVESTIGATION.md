# Packed projection and Adreno profiling investigation

Updated 2026-10-25. Device: Galaxy S26 (SM8850), Adreno 840. This report first
records five profiling experiments against the approximately 70 tok/s
baseline. A follow-up integrates the scalar-pair projection and fused
vocabulary-candidate selection; the direct runner now measures 76.75 tok/s
median in the awake paired run described below. The old full-model
bit-manipulation stall and hardware-counter calibration remain documented.

## Current decode budget

The values below are the 23-forward profiles in
[`hexagon-decode/final-profile-summary.json`](hexagon-decode/final-profile-summary.json).
The family totals are device-event durations summed across kernels; the span
also includes time between the first and last GPU event.

| Per-token GPU time | Direct Mojo runner | MAX graph |
| --- | ---: | ---: |
| Packed projections | 6.808 ms | 6.954 ms |
| Vocabulary head | 2.227 ms | 2.232 ms |
| Other GPU work | 2.909 ms | 2.812 ms |
| Total GPU execution | 11.945 ms | 11.998 ms |
| First-to-last GPU span | 13.403 ms | 14.571 ms |
| Time between GPU events inside span | 1.458 ms | 2.573 ms |

Packed projections are 57–58% of the GPU sum, so a 25% projection speedup
would save 1.70–1.74 ms. That would still leave about 10.25 ms of GPU execution,
slightly over budget before gaps and host work. It would leave an 11.7–12.8 ms
GPU span, so another 1.7–2.8 ms/token would still need to come from execution
or the measured gaps. Time before the first GPU event and host work after the
last event add further end-to-end cost. GPU sum alone is already over the 10 ms
budget.

## Memory and hardware-counter evidence

The calibrated post-tuning capture reports 578.65 MB/token of GPU-interface
traffic (546.19 MB read and 32.46 MB written). The calibrated external-memory
interface byte counters respond to known scans and copies and agree with
expected payloads to within 0.05%. The count includes traffic beyond the weight
payload; it is not a DRAM-only count, because requests served by a system cache
are not separated.

The phone's measured float8 streaming scan was about 68.2 GB/s. Dividing the
model traffic by that scan rate gives an idealized 8.5 ms transfer-time
comparison. It is not a strict lower bound: the scan is not a confirmed peak,
and the model's interface traffic and access pattern differ from the scan.
At 578.65 MB/token, a 10 ms budget requires an average 57.9 GB/s at that
interface. The measured decode profile averages about 48.4 GB/s over its
11.95 ms GPU execution sum, so it would need roughly 20% more effective
interface throughput to move the same traffic in 10 ms. The scan result makes
that rate plausible enough to investigate, but does not show that the packed
projection workload can sustain it or that memory bandwidth is its only
bottleneck. Non-transfer GPU work and submission gaps also consume time.

KGSL counter reads are enabled by the existing wrapper only for the duration
of a capture. Calibrated counters establish interface bytes, launched thread
counts, and relative resident wave contexts. They do **not** provide a
validated GPU busy percentage, ALU utilization, cache-hit rate, spill count,
or a wall-time memory-stall percentage. Some Adreno selector labels remain
unvalidated; zero values from those selectors are not evidence of no work.
Perfetto on this phone exposes GPU memory events, but not compute-stage or
hardware-counter tracks. `qprof` is unavailable to the shell. CPU userspace
PMU sampling with `simpleperf` works.

The selected eight-group/eight-way tuning reduced interface traffic from
659.01 to 578.65 MB/token and reduced the normal GPU sum to about 12 ms. It did
not establish a hardware ceiling. The existing counter calibration also found
that reducing private storage or raising occupancy alone is not a reliable
win: a lower-accumulator variant was slower, while the selected projection
kernel still reports 528 bytes of private storage per work item. That OpenCL
private-storage query is not a register count and does not prove spilling.

## Full-shape bit-manipulation stall

The earlier half-bit variant passed its small fixture but failed to finish a
full-model run. A temporary diagnostic plugin measured the OpenCL API
boundaries for this variant and the working baseline under the same runner and
device conditions:

| Measurement | Half-bit variant | Working baseline |
| --- | ---: | ---: |
| `clBuildProgram` calls | 21, total 0.982 s; max 318 ms | 21, total 0.506 s; max 79.6 ms |
| `clCreateKernel` calls | 21, total 0.11 ms; max 0.015 ms | 21, total 0.08 ms; max 0.009 ms |
| Packed projection-family enqueue calls | 939 before timeout | 3,775, completed |
| Total time inside those enqueues | 13.86 s | 60.3 ms |
| Enqueues over 10 ms | 49 | 0 |
| Slowest enqueue | 2.390 s (`grid=4,48,1`) | 2.638 ms |
| Explicit `clFinish` calls | 605, max 0.006 ms | 609, max 24.8 ms |
| Run result | timed out after 45 s, before decode | completed; 24-token prefill in 0.903 s |

The half-bit run's 2.390 s and 2.377 s enqueue calls occurred in
`clEnqueueNDRangeKernel`, not `clBuildProgram`, `clCreateKernel`, or
`clFinish`. The baseline completed with the same tracing plugin, which rules
out the plugin's logging as the cause. Thus the observed stall is a
variant-specific host-side delay inside the driver's enqueue path, not a long
queue synchronization wait. `clBuildProgram` succeeds; it does not include all
the expensive work.

A 15-second `simpleperf` sample of the uninstrumented half-bit variant showed
86.9% of samples under the model's `forward` call, and 96.8% of that sampled
child time under MAX `Context::get_or_load_function`. The hot stacks were in
`/vendor/lib64/libadreno_compiler_cl.so`, including Clang parsing,
SLP-vectorization, and Qualcomm QGPU register allocation (`QGPURAGreedy`).

A second 12-second sample was taken during a cold, instrumented half-bit run.
It recorded 6,256 CPU-cycle samples (3.92 billion cycles); 87.3% were under
`forward`, and 98.5% of the sampled child time was under
`Context::get_or_load_function`. Its hottest path passed through
`M_driver_bundle_load` and `clBuildProgram` into Qualcomm's compiler. The
instrumented trace captured just six packed-linear enqueues during that
window; one took 179.8 ms and the others took under 0.01 ms. On the separate
45-second API-boundary run, repeated long enqueues dominated elapsed time,
while aggregate `clBuildProgram` wall time was 0.982 s. These runs show that
compiler work can surface during program build/materialization and that the
variant also causes very long host delays at enqueue. They do not prove which
internal compiler phase accounts for each slow enqueue, because the CPU
sample was not synchronized to the 2.390-second enqueue. The trace does rule
out `clFinish` as the source of those long waits; neither this API trace nor
the CPU sample shows the GPU waiting on memory. The exact IR pattern or
allocator behavior behind the slow path remains unknown.

The half-bit generated OpenCL source was 46,977 bytes versus 29,125 bytes for
the selected baseline packed-linear program. Its source adds vector compares,
selects, and integer-to-half bit construction inside the already unrolled
multi-output dot. This size increase and the QGPU register-allocation stacks
make compiler complexity/live-range pressure the leading hypothesis, not a
measured register count. Other rejected bit-manipulation variants have not all
been re-profiled this way, so this diagnosis should not be generalized to
every arithmetic variant.

After the bounded profiling runs, ADB responded, no runner or `simpleperf`
process remained, GPU busy read 0%, thermal status was 0, and the KGSL
profiling switch was 0. There was no pre-run GPU reset-count snapshot, so the
observed reset count cannot be compared to a baseline. The user's earlier
freeze report is consistent with the very long host compiler/driver calls seen
in the half-bit run; it does not by itself establish memory exhaustion or a
GPU reset.

## Packed-kernel source findings and next work

The current [`kernels.mojo`](../../max/examples/android-llm/bonsai/kernels.mojo)
builds a 16-lane repeated activation vector and a 16-lane integer unpack vector
inside the eight-output dot path. It converts unpacked integers to FP16 before
feeding four independent FP16 accumulation chains. Those are source-level
temporaries; generated OpenCL is inspectable, but the Adreno machine code and
actual live ranges are not available through this setup.

## Five follow-up experiments

The device baseline was refreshed with five short runs: **70.22 tok/s median**
(70.22, 73.71, 70.73, 69.42, 68.13). The runner artifact's kernel-source hash
matched the current `kernels.mojo`. This is the appropriate baseline for the
comparisons below. Long-run API tracing and paired kernel comparisons add
instrumentation and run-to-run variation, so their throughput numbers are
reported separately rather than substituted for it.

1. **Baseline host/API profile and generated source.** Captured OpenCL for the
   current model: 21 programs, including `packed_linear` (40,447 source bytes)
   and `packed_head` (30,401 bytes). Both take six arguments; packed-linear
   width and reduction are runtime scalar arguments, so they are not
   compile-time constants. Across the traced run, 21 `clBuildProgram` calls
   took 559.6 ms total (median 17.1 ms, p95 79.9 ms, max 85.6 ms). Function
   loading took 0.245 ms total and kernel creation was at most 0.015 ms. These
   builds occur during startup/prefill, outside the decode timer. This baseline
   did not reproduce the multi-second lazy compiler stall.
2. **Fixed-shape specialization.** Built fixed entry points for the frequent
   6144x2048 and 2048x2048 projection shapes, retaining the dynamic fallback.
   The specialized ABI drops the two runtime dimension arguments, but generated
   source grows slightly because the specialized and fallback bodies coexist.
   Outputs matched the fixture. Three paired short-run comparisons measured
   69.08 tok/s median for baseline and 68.68 for specialization; this is no
   demonstrated win. Do not add the extra entry points on this evidence.
3. **Scalar-pair unpack.** Rewrote the unpack around two SIMD8 low/high weight
   vectors, retaining the eight-output tile, four accumulation chains, and
   summation order. This removes the explicit SIMD16 repeated-activation and
   unpack temporaries. Source shrank 8.6% for `packed_linear` (40,447 to
   36,978 bytes) and 11.3% for `packed_head` (30,401 to 26,977 bytes). All 95
   logit vectors passed comparison and selected token IDs were exact; max
   absolute logit error was 0.02017 and relative L2 error 0.000395. Short
   medians were 70.57 baseline and 71.37 candidate. Interleaved 95-token runs
   leaned toward the candidate but varied materially; a paired full-logit run
   was 63.86 versus 64.57 tok/s. That does not establish a reliable decode
   speedup. Device event timing shows why: packed-linear stayed effectively
   unchanged (6.837 vs 6.830 ms/token), packed-head improved about 4.2% (2.210
   vs 2.116 ms), and total GPU event time improved only about 0.9% (11.950 to
   11.846 ms). This is a promising source/compile-size reduction, not the
   needed packed-projection throughput improvement. It also reduced the 21
   `clBuildProgram` calls from 559.6 to 321.2 ms total (42.6%); function-load
   time remained negligible.
4. **Compiler and enqueue stall diagnosis.** In the current baseline's 95-token
   API trace, builds completed in 0.560 s total; `clCreateKernel` and function
   loading were negligible. The 17,969 enqueues took 140.1 ms total (7.8 us
   mean, 12 us p95, 2.215 ms max); 703 `clFinish` calls took 18.5 ms total,
   with 1 us p95 and one 18.1 ms maximum. Thus there is no repeated multi-second
   compile, enqueue, or finish stall in the baseline. The prior half-bit
   variant remains a separate regression: it stalled inside enqueue and its
   CPU samples pointed into Qualcomm compiler work. The scalar-pair CPU sample
   showed no dominant compiler or register-allocation path. No disassembly,
   validated register count, or reliable spill counter is available, so lower
   source size cannot be translated into a claim about machine-code live state.
5. **Warm decode host timeline and copy path.** A 96-output baseline trace
   recorded 96 blocking `OPENCL_READ_END` calls, each copying 993,280 bytes of
   logits to the host. They took 238.0 ms total (2.479 ms mean, 2.442 ms median,
   2.867 ms p95, 3.362 ms max), roughly 18% of the traced 1.292 s decode wall
   time. `runner.cpp` calls `readLogits()` each decode iteration and selects
   greedily on CPU; the MAX graph also transfers logits to CPU per token. The
   OpenCL plugin uses `clEnqueueReadBuffer(..., CL_TRUE, ...)`, so this wall
   time includes waiting for queued GPU work as well as the transfer. It cannot
   be added to GPU event time as independent overhead or treated as wholly
   removable copy cost. A warmed process CPU sample attributed 85.5% of samples
   to `forward`; nested within that, the sampled path ran through
   `Stream::execute`, `M_driver_queue_execute`, and Qualcomm's enqueue API.
   These are inclusive nested percentages, not additive buckets. The sample
   began at process launch, so its time outside `forward` includes setup and
   prefill and does not quantify steady-state per-token overhead.

## What the profiles say is actionable

At the time of the five profiling experiments, the refreshed baseline was
around 70 tok/s and consumed about 12 ms of GPU execution per token, before the
1.46–2.57 ms event gaps in the earlier direct and MAX profiles. It is not at a
demonstrated device speed limit: the 68.2 GB/s scan is a comparison point, not
a peak, and the available counters do not measure memory stalls, ALU busy time,
cache hit rate, or spills. But none of the
five experiments demonstrated the 25% packed-linear gain needed to save about
1.7 ms/token. Shape specialization did not help; scalar-pair reduced compile
source and modestly helped the head, while the dominant packed-linear event
remained ~6.83 ms.

## Follow-up: fused head candidates and compact readback

The scalar-pair `_group_dot8` variant is now in the tracked source. It keeps the
eight-output tile, four accumulation chains, and summation order while using
SIMD8 low/high streams instead of materializing repeated SIMD16 activations and
unpacked SIMD16 integers. A full 96-output reference run passed.

The direct runner's `packed_head` now computes one winning `(value, index,
nonfinite)` tuple per 64-lane subgroup while the eight logits are in registers.
The CPU reads 485 tuples (about 5.8 KB) and selects among them; the previous
standalone GPU argmax scanned the full vector and is a different, slower path.
The full 993,280-byte logits buffer remains available for validation. Candidate
selection checks the nonfinite flag from every subgroup and breaks equal-value
ties by the lowest vocabulary index.

Five interleaved, plugged-in, awake comparisons of 95 decode forwards measured
75.07 tok/s median with scalar-pair plus full-logit CPU selection and 76.75
tok/s median with fused candidates (+2.2%). Paired changes were +1.83%, +3.00%,
+2.07%, +2.12%, and +2.96%; the generated IDs matched on all five pairs. A
separate 96-output full-logit validation passed with max absolute logit error
0.0222683 and relative L2 error 0.000634382.

In a 23-forward instrumented profile, GPU execution was 11.852 ms/token for
scalar-pair with CPU selection and 11.907 ms/token with fused candidates; the
first-to-last spans were 13.516 and 13.489 ms. Packed-head event time rose by
about 0.13 ms/token, while the total event span stayed level. A separate
host-stage timing run attributed 277 ms / 23.5 ms to full-logit readback / CPU
selection across 96 outputs, and 253.5 ms / 0.18 ms to compact readback / CPU
candidate reduction. Allocation added 0.04 ms total. Those blocking read times
include queued-work waits, so they are not pure transfer costs.

The optional `MODULAR_ASYNC_GREEDY=1` runner path queues the compact read after
the next forward and completes it on the next iteration, overlapping token
printing with device work. Five paired runs measured 76.63 tok/s async median
versus 76.18 synchronous; the median paired change was 0.21%, with mixed
individual runs. It is not enabled by default because the speed change is not
reliable. The next worthwhile work is to profile and reduce MAX-graph's own
full-logit transfer and compare its queue gaps; strict autoregressive decoding
still needs each selected token before the next forward. The fused direct path
is a measurable small gain, but it does not establish a device ceiling or make
100 tok/s proven.
