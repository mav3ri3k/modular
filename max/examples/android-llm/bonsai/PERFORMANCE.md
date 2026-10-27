# Bonsai Android performance

## Scalar-pair unpack and fused vocabulary selection — 2026-10-25

The direct runner now uses two SIMD8 low/high activation streams in the packed
dot, retaining its eight-output tile, four FP16 accumulation chains, and
summation order. A full 96-output reference run passed (max absolute logit
error 0.02227, relative L2 0.0006344). This is the source variant selected
after the prior scalar-pair experiment reduced generated code and build time.

The vocabulary head now finds one local winner per 64-lane subgroup while its
eight logits are already in registers. It writes 485 `(value, index,
nonfinite)` records, about 5.8 KB per token. The CPU selects among those
records, preserving lowest-index ties and rejecting any nonfinite logit. The
full-logit buffer remains available and is still read during validation.

Five awake, interleaved 95-forward comparisons on the plugged-in S26 measured
**75.07 tok/s median** with scalar-pair plus full-logit CPU selection and
**76.75 tok/s median** with fused head candidates (+2.2%). The paired speedups
were +1.83%, +3.00%, +2.07%, +2.12%, and +2.96%; all generated IDs matched.
The phone remained in Android's ACTIVE device-idle state during these runs.
A separate 96-output full-logit validation passed every logit and token ID.
The Adreno fixture also exercised an all-equal 512-logit tile (winner index 0)
and infinities in its inputs (nonfinite flag set), alongside packed tails.

An instrumented 23-forward profile measured GPU execution at 11.852 ms/token
for the scalar-pair CPU-selection path and 11.907 ms/token for fused
candidates; first-to-last GPU spans were 13.516 and 13.489 ms. Packed-head
time rose by about 0.13 ms/token, but the overall event span stayed level.
In one host-stage timing run, the full-logit path spent 277 ms reading logits
and 23.5 ms selecting over them across 96 outputs. The compact path spent
253.5 ms in the blocking candidate read, 0.18 ms reducing candidates, and
0.04 ms allocating the temporary buffer. These read timings include waiting
for queued GPU work, so the approximately 47 ms difference is not a pure
transfer saving.

An opt-in `MODULAR_ASYNC_GREEDY=1` path enqueues candidate readback after the
next forward and completes it on the next loop iteration, allowing token
printing to overlap with queued GPU work. Five paired runs measured a 76.63
tok/s async median versus 76.18 tok/s synchronous, with a 0.21% median paired
change and mixed per-run results. Keep it experimental; it has no reliable
speedup yet. Strict autoregressive dependency still requires the token result
before starting the next forward, leaving little host work to hide.

This is a modest improvement, not a route to 100 tok/s by itself. The old
standalone GPU argmax results below remain applicable: rescanning the complete
logit vector in another GPU kernel was slower. The fused path avoids that extra
full-vector scan and reduction launch.

## Batch-one HVX and normalization pass — 2026-10-25

Added an opt-in packed-W2 HVX vocabulary GEMV and corrected activation-copy
ordering so Hexagon can overlap the GPU head. GPU-only remains the default:
the final five-repeat comparison measured medians of **71.18 tok/s starting
GPU, 71.31 updated GPU, and 71.77 GPU+HVX** (16,384 offloaded vocabulary rows).
The small hybrid difference is not consistent across batches and is below the
observed phone operating-rate variation; it is not a substantial NPU speedup.

Profiling exposed repeated serial sums in `normalize_qk`, `gated_norm` and
`head_norm`. Cooperative 64-lane reductions reduce their combined GPU time from
0.795 to 0.131 ms/token; total GPU work falls from 12.569 to 11.945 ms/token in
separate 23-forward profiles. The same kernels are used by the MAX graph.
All 96 IDs and full reference logits pass on two direct-runner prompts, with and
without HVX, and on the ordinary MAX graph for the first prompt. FP32 reduction
order changes; general quality and long-context equivalence are not established.

See the [full experiment report](../../../../docs/android-feasibility/hexagon-decode/README.md)
for all samples, cold/idle VTCM failures, staging experiments, numerical errors,
artifact hashes, and reproduction. Source builds and SDK dependencies remain in
`~/Library/Caches/modular-android-integration/`, under `hvx-final`,
`hvx-final-max`, `hexagon-sdk`, and `hvx-work`. No app update in this pass.

## Counter-guided subgroup pass — 2026-10-25

Selected eight-group projection reduction and eight-way recurrent K splitting.
Both use standard Mojo subgroup operations through the existing compiler,
DeviceContext and MAX custom-op path. MAX still performs the final partial
join with its ordinary `ops.sum`; no device intrinsic or runtime bypass was
added. Per-group FP16 dot accumulation is unchanged; FP32 join ordering changes.

Cooled MAX decode, 95 forwards, three interleaved measured cycles with 15-second
idle gaps and thermal status 0 throughout: **56.84 → 64.35 tok/s (+13.2%)**.
Selected samples: 64.35 / 63.08 / 64.40. Baseline: 56.85 / 56.67 / 56.84.
A final-source direct-runner smoke sample reaches 67.95 tok/s. These are decode
measurements including logit readback and sampling, excluding initialization,
prefill and CPU reference-logit verification; not sustained thermal ceilings.

Separate matched GPU profiles, mean over 23 decode forwards:

| Measurement | Before | Selected |
| --- | ---: | ---: |
| Packed layer projections | 8.61 ms/token | 7.15 ms/token |
| Rowwise kernels, including partial joins | 1.16 ms/token | 0.38 ms/token |
| Recurrent DeltaUpdate | 1.27 ms/token | 0.62 ms/token |
| Total GPU work | 15.86 ms/token | 12.94 ms/token |
| First-to-last GPU span | 16.60 ms/token | 14.59 ms/token |
| Dispatches | 599/token | 599/token |

Calibrated GPU-interface traffic falls **659.01 → 578.65 MB/token**:
546.19 MB read +32.46 MB written after the change. Serialized per-family
attribution measures packed-layer partial writes at 6.03 versus43.54 MB/token;
rowwise reads total7.70 versus45.54 MB/token. Instrumentation adds traffic, so
these separate captures are not exact conservation equations or DRAM readings.

Recurrent launches grow from2048 to16384 threads; measured resident contexts
rise2.67→7.48. Packed-projection residency falls6.36→5.24 and reported private
storage rises512→528 bytes/work item: this pass does **not** establish a register
pressure fix. Shorter dot loops and narrower unpack temporaries lost. Sixteen-
group projection reduction, larger/cached recurrent variants and late scale
loads did not beat the selected combination reliably.

Final-source full-logit and96-ID checks pass in both runners. MAX maxabs0.01787,
relativeL20.000620; direct maxabs0.02018, relativeL20.000631. A second, longer
numbers prompt passes96 IDs/all logits (maxabs0.01912, relativeL20.000418).
The packed fixture checks widths1/7/8/13/16/511/513, incomplete group tiles,
multiple group tiles, scalar tails and all four 2-bit codes. Projection scratch
storage shrinks8× in both the MAX graph and direct runner.

Raw counter/validation evidence, source/artifact hashes and selection runs:
[counter-tuning/summary.json](../../../../docs/android-feasibility/counter-tuning/summary.json).
Long uninterrupted experiment batches did throttle to~36 tok/s; their raw
records remain in the cache. No clocks, thermal policy or app package changed.


## Integrated compiler/backend pass — 2026-10-25

The current implementation uses ordinary Mojo DeviceContext, ABI-v3 opaque
views, SDK plugin discovery and the shared LLVM optimization pipeline. Packed
projections compute eight adjacent outputs per thread with standard Mojo SIMD
and the scoped `FastMathFlag.AFN` contract. Four FP16 accumulation chains and
FP32 group scaling/reduction are retained. Layer workgroups have 64 threads;
the vocabulary head has 128. No global fast-math override is required.

On the connected Galaxy S26 / Adreno 840, the final packaged JNI France smoke
test produced the correct answer at **53.04 tok/s**. This is a short sample,
not a sustained throughput guarantee. **100 tok/s has not been reached.**

All model comparisons use the pinned checkpoint in the model guide, unchanged
2-bit packed weights, greedy selection and independent FP32 MLX reference
logits. The long numbers fixture requests 96 tokens / 95 decode forwards at
capacity 256. Every generated ID matches; every logit is finite and checked.
Both baseline and selected kernels have maximum absolute error **0.0147266**
and relative L2 **0.000397484** on this fixture. A separate 96-token opposite
fixture and the short 24-token fixture also pass. Exact packed-kernel checks
cover widths 1, 7, 8, 13 and 16, including scalar tails and both group scales.

Interleaved measurements include one warmup per variant and alternating order:

| Measurement | Old scalar-output kernel | SIMD8 kernel |
| --- | ---: | ---: |
| Initial long fixture, full-logit validation, 3-repeat median | 28.97 tok/s | 46.65 tok/s |
| Final source build, full-logit validation, 3-repeat median | 30.87 tok/s | 35.67 tok/s |
| Final source build, generation only, 3-repeat median | 33.62 tok/s | 34.68 tok/s |
| Final source build, screen awake, generation only, 3-repeat median | 15.13 tok/s | 29.00 tok/s |

The first two final-build batches had substantial run variation: validation
samples were 35.67 / 45.38 / 30.15 tok/s, generation-only samples were
34.68 / 52.21 / 29.32. The phone was found Dozing afterward. These results are
retained; they are not filtered out or treated as a thermal steady state.
The subsequent awake-screen batch remained slower for both variants despite
Android reporting thermal status zero. Its selected samples were
29.04 / 28.95 / 29.00 tok/s; the baseline samples were 16.08 / 15.10 / 15.13.
Holding the screen awake did not remove the variation. CPU/GPU clocks and
thermal policy were not changed. The cause of the reduced operating rate is
not established; neither the 53.04 short JNI result nor the slower long batch
should be presented as a universal throughput figure.
Generation-only timing still checks **every output token ID**; it excludes
full-logit validation and does not stop early at EOS. Compilation, weight upload
and prefill are outside decode timing. No timings establish general quality
or sustained performance across contexts and phone power states.

GPU timestamp profiles of the short fixture localize the improvement:

| Kernel family | Before, ms/token | SIMD8, ms/token |
| --- | ---: | ---: |
| Packed layer projections | 16.91 | 8.87 |
| Vocabulary head | 4.56 | 2.60 |
| Packed partial reduction | 1.02 | 1.02 |
| DeltaNet state updates | 1.21 | 1.21 |
| All kernels | 26.31 | 16.33 |

The SIMD8 profile has 721 dispatches/token, mean queue span 17.22 ms and no
failed/dropped samples. Profiling adds overhead. Queue span excludes CPU greedy
selection and logit validation. Remaining work is dominated by projection data
movement/unpacking and recurrent updates; these measurements do not show
bandwidth saturation or a hardware ceiling. A 100-token/sec decode would need
less than 10 ms total per token, below the measured GPU execution alone.

Rejected experiments: 64/256-thread scalar workgroups, scale-only upload,
shared-memory ternary lookup, two accumulation chains, larger reduction groups,
and larger head workgroups. The latter two were indistinguishable in a longer
four-repeat comparison (47.72 / 47.94 / 47.70 tok/s), so the simpler original
geometry is retained. Standard GPU argmax passes tie, tail and nonfinite checks
but is slower than CPU selection, both with split-K scratch (45.63 versus
53.66 tok/s) and without it (42.98 versus 53.47). That standalone full-logit
scan remains rejected; the fused subgroup-candidate path above is a separate
implementation and is now used by the direct runner.

Reproduce paired full validation with `../benchmark.py --variant old=BUILD
--variant new=BUILD --model EXPORT --serial SERIAL --repeats 3 --output report.json`.
Add `--generation-only` for ID-checked timing without full-logit checks. Keep
screen/power state consistent between variants. JSON reports preserve all runs
and artifact hashes. Packed-tail checks can be compiled with
`Target/lib/OpenCL/run_device_context.py --fixture
max/examples/android-llm/bonsai/test_packed.mojo --target-accelerator adreno-840`
using the normal compiler/NDK/build/device arguments.

Development reports are retained in
`~/Library/Caches/modular-android-integration/bonsai-*-benchmark.json` and
`bonsai-final-*.json`; the exact final embedded object SHA256 is
`6379b00a58c75b745c9dd3622d256b7cf0ab251af1114102796879edb95cf119`.

## MAX graph comparison

Bonsai now also runs through an ordinary MAX graph with registered custom ops.
The first three-repeat generation-only comparison measures **53.74 tok/s Mojo
versus 49.90 tok/s MAX**, with every token checked. MAX reduces dispatches from
721 to 599 through fusion, but extra gaps between GPU work offset this gain.
See the [graph implementation and full results](../bonsai_graph/README.md).

## Historical optimization measurements

The sections below describe earlier implementations and protocols, including
the removed global fast-math path. They are not current build instructions.

The first optimization pass roughly doubled decode throughput while preserving
FP32 activations, the same packed weights, and reference-matching token IDs.
The measurements below distinguish the original baseline from the optimized
implementation; neither is a sustained-performance ceiling for the phone.

## Second optimization pass: FP16 and fused transforms

A matched fixed-length decode pair on the connected S26 measured **30.30 tok/s
(FP32) → 38.61 tok/s (FP16)**, a **27.4% improvement**. Each used the same
21-token prompt, cache capacity 128, 96 requested output tokens / 95 decode
forwards, model and fast-math setting. Decode includes full-vocabulary logit
readback and greedy selection. These are individual hill-climbing samples,
not repeated medians or a thermal steady-state ceiling. Earlier short tuning
samples were approximately 37–39 tok/s. Removing an unused full-vocabulary
reference argmax from non-validation runs then measured **41.10 tok/s** over
another 95-forward run. That final sample is separate from the matched pair;
these are short measurements and clock/thermal variation is not controlled.

Changes:

- Pack transformed inputs as FP16 pairs and use native half2 `mad` operations.
  Four independent packed accumulators limit reduction depth; final group
  scaling/reduction stays FP32. Recurrent state stays FP32.
- Fuse all nine block-512 Hadamard stages with subgroup shuffles and register
  butterflies. Pack the result in the same dispatch. Transform dispatches
  fall from 882 to 98/token; total dispatches from 1505 to 721/token.
- Use 128-thread workgroups for packed projections and the vocabulary head.
- Add opt-in `MODULAR_OPENCL_FAST_MATH=1` to the HAL's bundle compiler. This
  enables FMA contraction and relaxed math; the default remains strict.
- Skip the reference vocabulary argmax when reference validation is disabled.
- Add `benchmark` in the runner's reference-file argument to force a fixed
  generation length, including beyond EOS, without loading/checking reference
  logits during a tuning run.

The selected build passed the 24-output-token "opposite of hot" fixture:
all greedy IDs matched the FP32 reference, maximum absolute logit error
**0.0118408**, relative L2 error **0.000552528**. This is approximate arithmetic;
that fixture does not establish long-context or general quality equivalence.

Rejected experiments were retained only as external build artifacts:
W2A8 packed integer-dot math (~27 tok/s), fully unrolled W2A8 (~23), scalar
ternary selects (~23), cooperative eight-lane output reductions (~14), and
strict half2 `fma` (~4.5). Replacing strict `fma` with fast `mad` was essential.
Two packed accumulators did not improve over four. Half4 code hit an Adreno
program-build resource error and was dropped. No slower variant replaces the
selected source implementation.

Quick reproduction after building:

```bash
MODULAR_OPENCL_FAST_MATH=1 ./runner ./OpenCLPlugin.so \
  ./weights.bin ./prompt.txt benchmark 96
```

The source build is `/tmp/modular-bonsai-hill-best`; detailed measurements and
artifact hashes are in `/tmp/modular-bonsai-hill-results.json` on the development
machine. The demo was subsequently updated in the Hexagon work below.

### Hexagon work

The [Mojo Hexagon projection backend](../../../../Target/lib/Hexagon/README.md)
now executes the real vocabulary head, using resident packed weights, persistent
HVX/HMX workers and optional concurrent GPU/HMX row partitioning. A matched
95-forward sample measured GPU-only **41.45 tok/s**, whole-head HMX **29.09**,
and a 32,768-row HMX split **40.55**. NPU routing works but has not demonstrated
a speedup. The final split passes eight reference IDs and logit tolerances
(max absolute error 0.01056, relative L2 0.00051057).

The demo has now been updated with the faster GPU backend and measured
**41.21 tok/s** in the packaged JNI France smoke test. It retains GPU routing.
The NPU path currently runs through the native CLI/engine and is opt-in via
`MODULAR_HEXAGON_HEAD`, with `MODULAR_HEXAGON_HEAD_COLUMNS` selecting a row split.
See the backend README for reproduction and current limits.

## First optimization pass (October 23, 2026)

Warmup each variant once, then alternate baseline/optimized run order across
three measured repeats. Each run checks all vocabulary logits and generated IDs
against the same saved reference. Profiling is disabled for these throughput
comparisons; readback, greedy selection and reference validation are included.

| Fixture | Input tokens | Output tokens | Baseline median tok/s | Optimized median tok/s | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Opposite of hot | 21 | 24 | 15.45 | 29.12 | 1.88x |
| Numbers through 20 | 93 | 24 | 13.86 | 28.73 | 2.07x |

The first fixture's measured ranges were 15.23–15.80 baseline and 27.18–29.23
optimized. The longer fixture had more clock/thermal variability: 12.19–15.59
baseline and 22.96–29.49 optimized. No thermal or frequency controls were used;
these are short paired experiments, not a claim of sustained 29 tok/s.

The packaged Android JNI engine independently produced the France response at
**26.00 decode tok/s**, with its on-device tokenizer and a 2048-token allocation.
That smoke test validates packaging/integration; it is not a matched app-level
baseline benchmark. The chat UI was unchanged.

Kept changes:

1. Split the two 2048-to-16 gate projections across 32 chunks and finish their
   reductions together with sigmoid/decay computation. Measured GPU cost fell
   from about 17.44 to **0.52 ms/token** for the projections plus gate activations.
2. Reduce RMSNorm inputs once into 32 partial sums, then apply the scale. Its
   GPU cost fell from about 8.19 to **0.59 ms/token**, including both passes.
3. Use centered ternary dot products with four independent accumulation chains
   and compile-time lane unrolling. The loader verifies **bias == -scale** for
   every packed group before using this specialization.
4. Split non-head packed projections across quantization groups in a 2D launch,
   then reduce partial outputs. The vocabulary head remains a separate kernel.
   Scratch memory grows by only **384 KiB** (context-128 allocations ~592.28 MiB).
5. Reuse identical Hadamard inputs for Q/K/V or QKV/Z and gate/up projections.
   The loader verifies each shared sign-vector group. Hadamard stage launches
   fall from **1368 to 882/token**, and total dispatches from **1810 to 1505**.

All tested optimized IDs matched the saved FP32 reference. Maximum absolute
logit errors were 0.00001907 (France), 0.00003338 (opposite), and 0.00002670
(numbers). The existing SmolLM2 native reference regression also still passes.

Packed layer projections plus the vocabulary head remain the largest GPU
cost. The optimized France profile measured about 17.76 ms/token in layer
partial products, 1.07 ms/token in their reduction, and 5.67 ms/token in the
vocabulary head. Better vector/memory operations and Adreno-specific kernels
are the next substantial work. Head workgroups 64/128/256 were close in a short
test (29.12/29.07/29.47 tok/s); 64 was retained in the first pass. The FP16 second pass above uses 128.

### Reproduce an interleaved comparison

Preserve a prior build directory before rebuilding a candidate. The benchmark
uploads the model once, uses distinct native variants, warms each once and
reverses variant order on alternate measured repeats:

```bash
python max/examples/android-llm/benchmark.py \
  --variant baseline=/path/to/preserved-baseline-build \
  --variant optimized=/path/to/candidate-build \
  --model /path/to/exported-reference-fixture --serial YOUR_DEVICE_SERIAL \
  --repeats 3 --output /tmp/bonsai-comparison.json
```

The output JSON records every sample, medians, variant paths and reference
errors. Use the same fixture/checkpoint and leave GPU profiling disabled.

## Original baseline

### Device-timestamp profile

Collected on the connected Galaxy S26 / Adreno 840, with the France prompt
(19 input tokens) and eight reference-checked output tokens. Decode includes
seven new forward passes. OpenCL event START/END timestamps were collected
without adding per-kernel synchronization. These are GPU execution times in a
profiling run, not kernel enqueue wall times or a sustained benchmark.

| Kernel family | GPU ms per decode token | Share of summed kernel time |
| --- | ---: | ---: |
| Packed affine projections | 26.16 | 45.9% |
| Unquantized gate projections | 17.39 | 30.5% |
| RMS normalization | 8.19 | 14.4% |
| Hadamard transform stages | 2.54 | 4.4% |
| Recurrent DeltaNet updates | 1.26 | 2.2% |
| Other kernels | 1.50 | 2.6% |

Summed GPU kernel execution: 57.04 ms/token. Decode wall time: 62.14 ms/token
(16.10 tok/s), including readback, greedy selection and reference validation.
No samples were dropped or unavailable. Profiling itself adds overhead.

There were **1810 kernel dispatches per token**, including 1368 Hadamard stage
launches and 151 packed projections. Counting launches alone would incorrectly
identify Hadamard as the main arithmetic bottleneck: its summed GPU execution
is much smaller than projections and normalization. Queue/launch overhead and
gaps are not included in individual kernel durations.

### Original inefficiencies

- The original unquantized gate projection launched just **16 work items**, each doing
  a serial reduction over 2048 inputs. Most GPU lanes cannot participate.
- The original RMSNorm recomputed a 2048-element sum independently for each of 2048 output
  elements: about **4.2 million terms per normalization**, instead of reducing
  once and sharing the scale. There are 49 such normalizations per token.
- The original packed projection assigned one output to a work item, with serial unpacking
  and accumulation across the input dimension. It has no cooperative reduction,
  subgroup vector strategy or tuned Adreno tiling.
- The original Hadamard path used nine separate dispatches and intermediate global
  buffers. Identical input transforms are recomputed for separate projections.
- Activations and affine parameters are FP32 in this correctness baseline.
  Weights remain packed at 2 bits; we are not expanding the whole model to FP32.

The first pass above addresses these implementation problems. They do not establish that
memory bandwidth is currently saturated or that the phone has reached a
hardware limit. Cooperative reductions, subgroup/local-memory support, tuned
packed projections, and validated lower-precision activation paths are the
next optimization work. The initial compiler emitter supports a scalar subset;
using those GPU features also requires extending its supported lowering.

## Comparison with the 120 tok/s claim

[Prism's release announcement](https://prismml.com/news/bonsai-2-family)
reports **up to 120 tok/s for Bonsai 2 2B on iPhone 18 Pro Max**, with MLX on
Apple devices. Our short FP32-activation correctness runs are not a matched
benchmark of that implementation. General chip rankings do not establish
relative throughput for this particular packed/recurrent workload.

120 tok/s leaves **8.33 ms per token**. The original unquantized gate projections
alone took 17.39 ms. Reaching that target therefore requires substantial kernel
work; this profile supports pursuing improvements, not promising a particular
final speed. A comparison should match checkpoint/packing, prompt and context,
activation precision, generated length, warmup, sampling and thermal conditions.

## Reproduce

Use the normal build and exported reference fixture from [README.md](README.md),
then add `--profile` to the validation command:

```bash
python max/examples/android-llm/run.py \
  --build /tmp/modular-bonsai2-build --model /tmp/modular-bonsai2-model \
  --tokenizer /tmp/modular-bonsai2-hf --serial YOUR_DEVICE_SERIAL \
  --validate --profile
```

The flag sets `MODULAR_OPENCL_PROFILE=1` on the Android process. The HAL creates
profiling-enabled queues, retains kernel events, and reports/clears their
statistics after queue synchronization. At most 100,000 events are retained
between reports; longer runs explicitly report dropped samples. GPU timestamps
may perturb performance. Leave profiling disabled for throughput benchmarks.

## DeviceContext integration check (2026-10-24)

The v3 ordinary DeviceContext path passed all 24 reference steps on the attached
Adreno 840: maximum absolute error 0.0127101, relative L2 0.000557542, identical
IDs. Three sequential strict-math runs with the same exported checkpoint,
21-token prompt, capacity 128 and 95 decode forwards measured 40.42, 40.27 and
40.19 tok/s (median 40.27), including full logit readback and greedy selection.
These short samples establish that typed buffer ownership and standard launch
encoding remain practical; they are not a matched legacy/new speedup experiment
or a sustained thermal result. No kernel performance tuning was added here.

Optional Hexagon projection now receives inputs through typed DeviceContext
copies. Its transfer currently synchronizes the stream before host projection;
the previous bridge's overlapping transfer queue needs a typed stream/event
implementation before comparing hybrid performance.

### Exact compact scales — 2026-10-25 hill climb

The loader now checks that every centered scale roundtrips through FP16 exactly,
then stores one half scale per group, discarding the redundant bias. The GPU
still uses the same centered FP16 four-chain dot and FP32 scale/reduction.
Parameter storage falls117506048→29376512 bytes; total tensor weights596106496→
507976960 bytes. This changes resident storage without quantizing scales further.
The exported checkpoint format remains the input format, compacted once at load.

Three alternating95-forward generation-only measurements, every96 output IDs
checked, capacity128, no clock/power overrides:

| Path | Original parameters | Compact scales |
| --- | ---: | ---: |
| Mojo DeviceContext | 53.25tok/s | 58.07tok/s |
| MAX graph | 49.23tok/s | 53.11tok/s |

Independent short/long all-logit validation passes; Mojo errors remain exactly
.0165186/.000786457 and .0147266/.000397484 (maxabs/relativeL2).
[Raw comparison records](../bonsai_graph/results-hillclimb-2026-10-25.json).
GPU profiling of compact scales: Mojo14.99ms summed GPU,15.89ms span; MAX15.17/
16.09ms. Dispatches remain721/599. Projections8.09ms and head2.23ms dominate MAX.
These are event timings, not hardware traffic/occupancy counters.

A subsequent shared CPU greedy helper combines finiteness and selection with
independent reduction lanes, retaining first-index ties and rejecting NaN/Inf.
Its stage timing drops from about.8ms/output to.25–.30ms. Later whole-run results
were thermally noisy; the table above does not claim that additional gain.
Larger host batching, extra queues, K-lane reductions, smaller output vectors,
eight accumulators and affine bias hoisting did not win reproducibly. Some
bit-manipulation kernels passed tiny tests but stalled at model scale and were
rejected. See the concise [experiment log](../../../../docs/android-feasibility/HILLCLIMB_LOG.md).
100tok/s remains a goal; this pass does not establish a hardware ceiling.

Final selected-source comparison after cooling, three alternating95-forward runs:
**Mojo59.11tok/s, MAX56.43tok/s**; all96 IDs match. These include the combined CPU
sampler. Full-logit validation separately passes (Mojo.0204191/.00064232,
MAX.0200463/.000646198). The controlled compaction table isolates the weight
change; do not attribute the entire difference between batches to sampling.
Direct KGSL allocation/query worked, but all29 named counter reads returned
EPERM; no hardware counter values were obtained. No app update this pass.

Counter permission follow-up: EPERM was caused by KGSL's disabled profiling
switch. `../with_gpu_counters.py --serial SERIAL -- COMMAND ...` enables the
shell-writable `/sys/class/kgsl/kgsl-3d0/perfcounter` only around the profiling
command, then restores it. All29 reads now succeed in a Bonsai decode capture.
Only7 selectors produce nonzero deltas with the current Mesa A8xx labels;
Adreno840 event/pipeline mapping and normalization need calibration before
reporting utilization or DRAM GB/s. No root or lock-screen action was needed.

## Tiled projection model integration — 2026-10-26

The direct runner and MAX graph now both repack non-head matrices once at load
time to the selected 32-output layout and use four outputs/lane with
eight-group joins. The rebuilt MAX graph passed full-logit and token-ID
validation. In a matched 95-forward generation-only run, medians were 67.88
tok/s direct and 70.91 tok/s MAX; a separate event-profile run found 1.55 ms
more MAX event span beyond kernel time. Short-run throughput varies, so this
does not establish a stable MAX speedup. See the [integration record](../../../../docs/android-feasibility/PACKED_PROJECTION_BENCHMARK.md)
and [matched host profile](../../../../docs/android-feasibility/MAX_SUBMISSION_PROFILE_2026-10-26.md).


Demo decision: rolled back tiled model integration in both direct and MAX paths.
Fresh ID-validated comparison medians68.81 direct vs69.20 MAX tok/s are tied
within observed variation. Keep direct demo backend and compact fused token
readback. Earlier77 tok/s samples are not a repeatable sustained claim.
See docs/android-feasibility/PACKED_PROJECTION_BENCHMARK.md for raw comparisons.
