# Standalone packed-projection diagnostics and layout experiments

2026-10-26. Connected Galaxy S26 SM-S942U1, Adreno 840. This pass isolates
`packed_linear`; it does not change model execution or claim an end-to-end
decode improvement. Scheduling/unroll tuning and autotuning are deferred.

## Reusable benchmark

[Source and reproduction](../../max/examples/android-llm/bonsai/projection_bench/README.md).
The benchmark extracts the current Mojo dot/projection, compiles it through the
production `adreno-840` target, and launches generated OpenCL directly. Its
baseline OpenCL is byte-for-byte identical to the selected production
scalar-pair kernel (36,978 bytes). Width/reduction remain runtime arguments.

The native host validates reversible packing, every GPU partial output against
the baseline, independent CPU checks, and per-bank diagnostic checksums before
measurement. Synthetic random codes exercise all four code values; dyadic
activations/scales make the sampled independent CPU projection oracle exact.
These are representative shapes and working sets, not captured model tensors.

`hot` repeats one matrix. `ring` cycles through at least 128 MiB of distinct
packed weight addresses plus separate scale storage (about 8 MiB). Activations
stay small and resident. All layouts retain two bits per weight without
padding, eight outputs per lane, eight groups per subgroup, four accumulation
chains, the 64-thread workgroup, and the original reduction/scaling order.
Only complete tiles are supported; the vocabulary head and final partial-output
reduction are outside the timed kernel.

After validation, every bank is warmed before 512 dispatches are timed with GPU
events, without a synchronization per dispatch. The primary metric is the median
across independent process runs of each run's mean event execution duration.
Host enqueue times, startup/build/kernel creation, first execution, event spans,
and queued/submitted/start/end statistics are retained separately. Queue wait
statistics include deliberately queued batch backlog, not model-token overhead.
No GPU clock or thermal-policy overrides are used.

## Baseline diagnostic results

Three runs per cell; microseconds per dispatch. Every sample is retained in the
[raw diagnostic results](packed-projection/diagnostics.json).

| Output × reduction | Working set | Load checksum | Unpack checksum | Full projection |
| --- | --- | ---: | ---: | ---: |
| 6144 × 2048 | hot | 44.237 | 67.405 | 46.557 |
| 6144 × 2048 | ring | 56.874 | 70.713 | 58.362 |
| 2048 × 2048 | hot | 13.924 | 25.986 | 16.732 |
| 2048 × 2048 | ring | 21.590 | 28.288 | 23.506 |
| 2048 × 6144 | hot | 44.168 | 67.358 | 46.730 |
| 2048 × 6144 | ring | 57.278 | 70.826 | 58.631 |

The large projections slow about 25% when switching from repeated weights to
the rotating working set. The ring load checksum is within approximately
1.5–1.9 microseconds of full execution. This supports prioritizing memory layout
and warns against benchmarking only one cache-resident matrix. It does not
establish DRAM bandwidth saturation or a hardware ceiling.

The unpack checksum intentionally consumes converted FP16 bits in a rolling
integer hash. Its additional dependency chain makes it slower than the full
projection. It validates the unpack path and provides an alternative stress
probe; **the difference is not an estimate of unpacking cost**. Removing or
replacing arithmetic changes compiler optimization and load/ALU overlap.

## Layout comparison

Two weight orders were compared with the baseline:

- `group_word`: `[output/64][group/8][group%8][word][output%64]`.
- `word_group`: `[output/64][group/8][word][group%8][output%64]`.

Each also has a bit-plane variant: the eight even two-bit codes occupy the low
16 bits and the eight odd codes the high 16 bits of each word. Group scales
retain their existing order. CPU packing is outside timing.

The tiled address expression initially exposed an existing emitter/driver
incompatibility: vectorized signed 64-bit division generated a signed vector
shift with an unsigned vector shift count that Adreno rejected. Using unsigned
arithmetic for known-nonnegative indices avoids that expression. No emitter or
driver source was changed by this pass.

A rotate-based diagnostic hash also produced `llvm.fshl.v8i32`, which the
OpenCL emitter does not support. The validated probe instead uses multiply/XOR
arithmetic. These build failures were resolved before the recorded comparisons;
they are not GPU throughput results.

Five alternating-order sweeps covered every non-head shape, hot and ring: 250
successful measured processes, each with validation outside timing. Ring medians
in microseconds per dispatch:

| Output × reduction | Baseline | Group then word | Word then group | Group/word + planes | Word/group + planes |
| --- | ---: | ---: | ---: | ---: | ---: |
| 6144 × 2048 | 58.385 | 61.176 | 56.581 | 61.093 | 56.642 |
| 2048 × 2048 | 23.616 | 25.754 | 23.554 | 25.787 | 23.552 |
| 2048 × 6144 | 58.720 | 61.700 | 57.738 | 61.685 | 57.670 |
| 4096 × 2048 | 40.438 | 43.954 | 39.911 | 43.893 | 39.921 |
| 512 × 2048 | 11.548 | 11.654 | 10.738 | 11.639 | 10.759 |

[All samples](packed-projection/layouts.json),
[cell summaries](packed-projection/layouts.summary.json).

For each sweep, multiplying kernel means by the model's call counts
(66/42/24/6/12) gives a **synthetic projection-family estimate**. Its median is
6.642 ms for baseline, 6.484 ms for `word_group`, and 6.480 ms for
`word_group_planes`. This is approximately 0.16 ms / 2.4% less projection time,
not a measured reduction in model token latency. The five paired reductions
for ordinary `word_group` were 2.29%, 2.28%, 2.36%, 3.13%, and 2.50%.

Grouping by word is a repeatable modest win; grouping by group is a regression.
Bit-plane packing gives no convincing additional gain: its tiny difference
from ordinary `word_group` is smaller than the sample variation. These results
do not approach the proposed 25% projection-latency reduction.

## Follow-up: group-major outer tile order

A sixth layout changes only the two outer tile axes of `word_group`:
`[group/8][output/64][word][group%8][output%64]`. This makes adjacent X
workgroups consume adjacent tiles rather than skipping over the tiles for other
Y workgroups. The computation and launch geometry remain unchanged.

Three new alternating-order ring sweeps compared it with refreshed baseline
and `word_group` runs. All 45 measured processes passed exact validation.

| Output × reduction | Baseline µs | Word/group µs | Group-major outer order µs |
| --- | ---: | ---: | ---: |
| 6144 × 2048 | 58.457 | 56.543 | 56.392 |
| 2048 × 2048 | 23.520 | 23.669 | 23.529 |
| 2048 × 6144 | 58.659 | 57.775 | 56.582 |
| 4096 × 2048 | 40.484 | 39.932 | 39.943 |
| 512 × 2048 | 11.544 | 10.738 | 10.885 |

The corresponding synthetic family estimates are **6.641 ms baseline,
6.486 ms word/group, and 6.442 ms group-major**. Group-major saves about
**0.20 ms / 3.0%** versus baseline. The three paired latency reductions were
2.84%, 3.16%, and 3.11%. Most of its incremental benefit over `word_group`
comes from the 2048×6144 down-projection. This is a smaller, exploratory sample
than the five-sweep primary comparison; it still needs real-model evaluation
before promotion. [Raw results](packed-projection/group-major.json).

## Layout load-probe confirmation

Three additional ring sweeps of the load-only probe (27 measured processes)
show the same directional benefit, in microseconds:

| Output × reduction | Baseline | Word/group | Group-major |
| --- | ---: | ---: | ---: |
| 6144 × 2048 | 57.153 | 55.068 | 54.976 |
| 2048 × 2048 | 21.641 | 21.100 | 21.019 |
| 2048 × 6144 | 57.474 | 56.533 | 54.936 |

[Raw results](packed-projection/layout-loads.json).
The improvement is present even without weight conversion/FMA, supporting a
memory-access explanation. This is sensitivity evidence, not calibrated DRAM
traffic or a proof that the full kernel is exclusively bandwidth-bound.

## Verification and disposition

- All recorded production-shape processes passed repacking and numerical checks:
  54 baseline diagnostic runs, 250 primary layout runs, 45 group-major follow-up
  runs, and 27 load-confirmation runs. No samples were filtered.
- Thermal status was 0 and the phone was awake before/after every recorded
  production-shape run. Power/thermal policy and GPU clocks were not overridden.
- In the 250-run primary layout comparison, the maximum candidate build plus
  kernel-creation time was 57.58 ms. The maximum first enqueue was 0.052 ms;
  the maximum warm enqueue was 0.408 ms. These variants did not reproduce the
  prior multi-second compiler/enqueue stall. This does not resolve every prior
  arithmetic variant's compiler behavior.
- The benchmark compiles through the source-built Mojo target; Python lint,
  syntax checks, C++ warnings-as-errors, and final small-tile device checks cover
  the new harness. [Compiler/package/source hashes](packed-projection/provenance.json)
  and [device details](packed-projection/device.txt) accompany the raw records.
- Keep the model's selected production kernel and loader unchanged. The new
  layouts are standalone candidates; model integration and full-logit validation
  remain necessary before claiming any end-to-end gain. No application update,
  staging, or commit was performed.

The useful result is a validated, reusable workload benchmark and a modest
layout candidate. The best measured synthetic family estimate improves about
3%, rather than the sought 25%. The current evidence favors layout/locality
work over expecting a large win solely from shorter unpack source. Scheduling
experiments and autotuning remain for a later pass.

## Iterative optimization and autotuning pass

The follow-up used two working notebooks, an experiment history and a priority
queue, and repeatedly generated Mojo candidates, checked outputs, and profiled
them before choosing the next experiment. All current queue items are closed
for this pass. This does not establish a global optimum or a hardware ceiling.

The selected standalone configuration is **four outputs per lane, eight groups
joined, 64 threads/workgroup, one-word loop, and 32-output packed weight tiles**.
It preserves four FP16 accumulation chains, centered code values, exact scales,
and two-bit storage. It requires a new weight order and twice the original
number of X subgroups; it is not a drop-in replacement for the existing loader
or launch geometry. [Machine-readable configuration](packed-projection/iteration/selected.json).

The final comparison used a fresh seed, non-dyadic FP16 activations, a rotating
128 MiB weight set plus scale storage, at least 256 warmup dispatches, 512 timed
projection/reduction pairs, and five alternating sweeps across all five shapes.
Each bank's partial outputs matched the original GPU baseline exactly. Timings
include the existing Mojo partial reduction; initialization, repacking, checks,
and readback are outside measurement.

| Synthetic family estimate, ms | Original kernel/order | Previous group-major winner | Selected candidate |
| --- | ---: | ---: | ---: |
| Projection + final partial reduction | 7.036884 | 6.836508 | **6.685752** |
| Projection component | 6.682326 | 6.483162 | **6.332352** |
| Reduction component | 0.354306 | 0.353718 | 0.353466 |

These are medians of per-sweep shape-weighted estimates, using the 150 real
non-head call counts. Component medians need not sum exactly to the total
median. The candidate saves approximately **0.35 ms / 5.0%** versus the original
pipeline and **0.15 ms / 2.2%** beyond the previous layout result. The projection
component alone improves about 5.2% versus original and 2.3% versus the previous
winner. This remains well below a 25% projection-latency reduction.

The five paired pipeline reductions versus original were 4.85%, 4.47%, 4.96%,
5.06%, and 5.21%; versus the previous winner they were 2.12%, 1.64%, 2.45%,
2.28%, and 2.27%. These are standalone results, not model token-rate gains.
[Final records](packed-projection/iteration/final.json),
[derived summary](packed-projection/iteration/summary.json).

| Output × reduction | Original pipeline µs | Previous winner µs | Selected µs |
| --- | ---: | ---: | ---: |
| 6144 × 2048 | 61.228 | 59.191 | 58.574 |
| 2048 × 2048 | 25.903 | 25.835 | 24.393 |
| 2048 × 6144 | 61.953 | 59.807 | 58.805 |
| 4096 × 2048 | 43.018 | 42.729 | 42.063 |
| 512 × 2048 | 13.569 | 12.777 | 10.653 |

### Experiments and decisions

- UInt32 matrix indexing and tiled scale storage: no material independent gain.
- Four-pair runtime loop: 3–6% slower despite smaller source.
- Straight-line half-bit construction: numerically correct, catastrophically
  slower on GPU; see the byte-counter evidence below.
- Narrow integer conversion: sub-percent effect, no reliable additional gain.
- One-word prefetch: about 7% slower.
- Two/four-word unroll: small-fixture wins failed to transfer to large shapes.
  The tuner retained unroll-two as an interaction, but confirmation favored one.
- Subgroup activation broadcast: about 14–21% slower.
- Eight-pair vector activation loads: correct but multi-millisecond execution;
  rejected after completed full-shape screens, with all results retained.
- Two-pair loads: compiler emitted exactly the already measured scalar kernel.
- Manual address hoisting: source inspection showed LLVM already does it.
- Matched 32-output packing: retained after an additional three-sweep, all-shape
  paired comparison. Pipeline estimates were 6.748932 ms for the SIMD4/p64 parent
  and 6.691416 ms for SIMD4/p32. The fresh paired benefit was about 0.8–1.1%.
  [Packing holdout](packed-projection/iteration/packing-holdout.json).

Autotuning screened 54 configurations: output tiles 4/8/16, groups 4/8/16,
workgroups 64/128/256, and word unroll 1/2. Every micro fixture passed numerical
checks. All 18 SIMD16 points exceeded the deliberately loose 10× execution-cliff
threshold and were retained as rejected screens. The other 36 were measured on
three major shapes with fresh controls every six candidates. Five finalists
then received five alternating all-shape sweeps (150 processes) plus 30
independent-seed, non-dyadic validation processes. The initial screen's best
configuration did not remain best in confirmation; this is why screen timings
were not treated as final wins. SIMD4, which lost under earlier packing, is a
modest winner with the new layout and geometry.

[Micro screen](packed-projection/iteration/autotune-micro.json),
[screen decisions](packed-projection/iteration/autotune-screen.json),
[full-shape screen](packed-projection/iteration/autotune-shape-screen.json),
[confirmation](packed-projection/iteration/autotune-confirm.json),
[accuracy](packed-projection/iteration/autotune-accuracy.json).

### Final calibrated hardware-counter profile

Twenty-five accepted windows used the previously calibrated GBIF read/write
selectors and SP thread/context selectors. Every window's compute-thread count
matched the expected launch geometry exactly. Profiling was enabled only for
this command, every acquired slot was released, and the switch was restored to
zero. These timings are from separate instrumented runs, not the ranking cohort.
[Raw counters](packed-projection/iteration/counters.json).

For 6144×2048, the second capture measured:

| Projection variant | Interface read MB/dispatch | Interface write MB/dispatch | GPU µs | Resident wave contexts |
| --- | ---: | ---: | ---: | ---: |
| Original | 3.352321 | 0.049730 | 58.156 | 9.35 |
| Previous group-major winner | 3.351829 | 0.049728 | 56.162 | 9.46 |
| Load-only control | 3.351060 | 0.049712 | 54.645 | 12.27 |
| Selected SIMD4/p32 | 3.352580 | 0.049713 | 55.695 | 13.78 |

Packed weights plus scales are 3.342336 MB. Normal reads are close to that
payload plus inputs/metadata. The winner moves essentially the same bytes;
its higher resident-context ratio is consistent with better latency hiding,
although these counters cannot establish causation or an occupancy percentage.
The load-only probe remains close to full execution. This supports a memory
access/throughput focus, without proving an absolute bandwidth limit. GBIF
includes requests served by system cache and is not a DRAM-only measure.

The rejected half-bit decoder is qualitatively different. A bounded single
6144×2048 projection passed numeric checks but recorded **1,249.33 MB read and
1,548.46 MB written**, with the expected 12,288 compute-thread invocations and
100.58 ms of GPU execution. Its explicit output is only the ordinary 49 KB of
partials. The excess traffic, together with a private-storage query of 3,224
bytes versus 320 for the control, is consistent with compiler-generated private
memory spilling/materialization. Disassembly and a validated spill count are
still unavailable. This particular regression is slow GPU execution, not a
multi-second compilation/enqueue hang; it does not diagnose every earlier
bit-manipulation variant or show that the underlying arithmetic idea is slow.

The reusable tools, selected source generator, configuration, raw samples, and
both working notebooks remain available. Production model kernels, loader,
application, clocks, and thermal policy were not changed. Actual model
integration, full-logit validation, and end-to-end throughput measurement remain
necessary before promoting the candidate.

## Quick direct-model integration check

The direct Mojo runner now repacks only the 150 non-head projection matrices
at load time, selects the four-output tiled kernel, and launches twice as many
X subgroups. The checkpoint format, scales, vocabulary head, and MAX graph's
legacy kernel/layout remain unchanged. Repacking stays outside decode timing.

A quick generation-only check used the existing 21-token prompt, capacity128,
and96 requested outputs /95 decode forwards. Each run checked all96 reference
token IDs; all matched. No full-logit validation or detailed profiling was run.

| Measured pair | Baseline tok/s | Integrated tiled tok/s |
| --- | ---: | ---: |
| First | 77.01 | 78.10 |
| Reversed order | 77.36 | 66.76 |
| Follow-up | 65.10 | 66.30 |

The follow-up had already completed when the user requested no further runs.
All samples are retained. The result is noisy and does not establish a reliable
model-level improvement or regression. Thermal status was0 at every sample.
No additional runs were started after the stop request.

[First comparison](packed-projection/model-integration/check.json),
[already-completed follow-up](packed-projection/model-integration/followup.json).
Builds and source snapshots: `/tmp/bonsai-projection-base` and
`/tmp/bonsai-projection-tiled`; both use the same compiler/packages/runtime.


## Demo rollback and backend check (2026-10-26)

Reverted tiled projection dispatch, launch geometry and loader repacking in both
Mojo direct and MAX graph paths. The experimental kernel and standalone records
remain available; production uses the original eight-output packed_linear.
Existing fused vocabulary-head selection remains enabled.

Fresh generation-only comparison, 96 reference IDs / 95 decode forwards,
three alternating measured runs per backend after warmup, same rebuilt HAL and
CompilerRT, no profiling:

| Path | Measured tok/s | Median |
| --- | --- | --- |
| Direct | 68.81 / 73.85 / 64.78 | 68.81 |
| MAX | 67.61 / 69.72 / 69.20 | 69.20 |

All reference IDs matched. The 0.6% median difference does not establish a
winner. Retain direct for the demo: existing 2048-token chat integration and
smaller runtime package. Do not claim sustained77 tok/s from the earlier
77.01/77.36 measurements.

The earlier77 tok/s cohort used different HAL/CompilerRT binaries. An additional
controlled comparison with the exact same restored direct runner tested that
confound: original runtime67.69/67.89/67.45 (median67.69), rebuilt runtime
69.42/67.71/68.12 (median68.12). Every ID matched; thermal status0. This does not
show a runtime rebuild regression or explain the earlier higher readings.

Raw records and artifact hashes: [demo/](packed-projection/demo/).
The chat app stages the restored direct model.o and canonical runtime, and uses
shared readGreedyToken rather than copying/scanning the full vocabulary logits.


Final packaged chat app: JNI France smoke check passed, answered Paris,79.12
decode tok/s on the short answer. This is not comparable to the fixed95-forward
CLI cohort and is not a sustained throughput claim. Both real-model UI tests,
Gradle build/lint and tokenizer parity passed. App left installed on DEVICE_SERIAL.
