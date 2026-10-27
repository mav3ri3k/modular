# Standalone packed-projection investigation

This benchmark isolates Bonsai's `packed_linear` partial-product kernel on the
Adreno 840. GPU arithmetic is generated from the current `../kernels.mojo`
through the same Mojo target as the model. The native host supplies buffers,
validates outputs, and records OpenCL events. It bypasses model execution and
MAX submission overhead. It does not change the model's kernel or weight loader.

The baseline generated OpenCL was byte-for-byte identical to the selected
production scalar-pair kernel when this benchmark was introduced. Generated
sources, compiler/package hashes, native binary hash, and the production source
hash are recorded outside the repository in the build directory. Rebuild when
production changes; the generator intentionally fails if its expected source
structure no longer matches.

## Probes and layouts

All variants retain eight outputs per lane, eight groups per subgroup, the
64-thread workgroup, and the production subgroup reduction. Full projections
retain four FP16 accumulation chains and the FP32 group scaling/summation order.
Only complete tiles are supported: width must be divisible by 64 and reduction
width by 1024. Scalar tail code remains in generated sources to preserve the
production baseline's compiler input; tiled layouts are not valid for tails.
The vocabulary head is outside this benchmark.

- `full`: the production projection arithmetic.
- `load`: packed-word and activation-word checksum, followed by the same scaling,
  subgroup reduction, and partial-output stores.
- `unpack`: decode all codes, convert to centered FP16 weights, and consume their
  bits in a rolling checksum; retain activation loads and the same output path.

Checksums keep the loads/conversions observable. The unpack checksum has its own
integer dependency chain and can be slower than the full projection. These
probes test sensitivity to removing work; **do not subtract their times to
attribute exact unpack/FMA costs**, or call load-only a measured hardware peak.
The driver may optimize the diagnostic expressions differently from the full
projection. Generated OpenCL is available for inspection.

Weights remain exactly two bits per element, with no padding:

| Layout | Physical word order |
| --- | --- |
| `baseline` | `[group][word][output]` |
| `group_word` | `[output/64][group/8][group%8][word][output%64]` |
| `word_group` | `[output/64][group/8][word][group%8][output%64]` |
| `group_word_planes` | `group_word`, with even/odd codes in separate 16-bit halves |
| `word_group_planes` | `word_group`, with even/odd codes in separate 16-bit halves |
| `group_major` | `[group/8][output/64][word][group%8][output%64]` |

A quantization group contains eight words per output, with sixteen codes per
word. Bit-plane variants interleave the eight two-bit even codes in bits 0–15
and the eight odd codes in bits 16–31. Group scales retain their production
layout. Packing happens on CPU before timing and requires separate consideration
before adopting a layout in the model loader.

## Build

Use an existing Android HAL build's matching `std.mojoc` and `max.mojoc`, the
source-built Mojo compiler, Khronos OpenCL headers, an Android `libOpenCL.so`
(for linking), and the NDK. For example:

```sh
python3 max/examples/android-llm/bonsai/projection_bench/build.py \
  --mojo bazel-bin/Mojo/tools/mojo/mojo \
  --packages /tmp/bonsai-candidate-fixture \
  --ndk "$ANDROID_SDK_ROOT/ndk/29.0.14206865" \
  --opencl-headers /tmp/OpenCL-Headers \
  --opencl-library /tmp/libOpenCL-android.so \
  --output /tmp/packed-projection-bench
```

The temporary Mojo host wrapper exists only to instantiate the normal
`adreno-840` accelerator target and extract its generated OpenCL sidecar. Width
and reduction remain runtime kernel arguments. The actual benchmark executes
that sidecar directly with the generated ABI v3 `(buffer, byte offset)` pointer
arguments. OpenCL builds use `-cl-std=CL3.0` without global relaxed math; the
production scoped approximate multiply-add contract remains in the kernel.

## Run

Start with independent checks on small, complete tiles:

```sh
python3 max/examples/android-llm/bonsai/projection_bench/run.py \
  --adb "$ANDROID_SDK_ROOT/platform-tools/adb" --serial DEVICE_SERIAL \
  --build /tmp/packed-projection-bench --output /tmp/projection-smoke.json \
  --shapes 64x1024 --regimes hot --repeats 1 --iterations 128 --pause 0
```

Then run the diagnostics before comparing layouts:

```sh
python3 max/examples/android-llm/bonsai/projection_bench/run.py \
  --adb "$ANDROID_SDK_ROOT/platform-tools/adb" --serial DEVICE_SERIAL \
  --build /tmp/packed-projection-bench --output /tmp/projection-diagnostics.json \
  --layouts baseline --shapes 6144x2048 2048x2048 2048x6144 --repeats 3

python3 max/examples/android-llm/bonsai/projection_bench/run.py \
  --adb "$ANDROID_SDK_ROOT/platform-tools/adb" --serial DEVICE_SERIAL \
  --build /tmp/packed-projection-bench --output /tmp/projection-layouts.json \
  --modes full --repeats 5
```

The default shape set covers all 150 non-head packed projections per token:
66 calls of 6144×2048, 42 of 2048×2048, 24 of 2048×6144, six of 4096×2048,
and twelve of 512×2048 (output width × reduction width).

`hot` replays one matrix. `ring` rotates at least 128 MiB of distinct weight
addresses **and distinct scale buffers**; it reuses a small activation buffer.
This is a controlled synthetic working set, not a replay of captured model
activations, weights, or the model's complete dispatch sequence. Ring size is
configurable. Synthetic random codes exercise all four code values; dyadic
activation/scale values make the independent CPU projection oracle exact.

Every bank's host repack must be a bijection with an exact bitwise inverse.
Full projections compare every partial output in every bank exactly with the
production GPU baseline, plus independent CPU checks on sixteen columns of the
first bank. Load checksums are CPU-checked for every bank; unpack checksums are
CPU-checked for the first bank and compared with the baseline unpack probe for
remaining banks. All validation/readback occurs outside measurement.

Each process records build/kernel-creation time, first enqueue/completion, and
first device time separately. After validation it warms every bank, then queues
512 dispatches without per-dispatch synchronization. Event execution means,
medians, p95, span, queue/submission delays, and host enqueue durations are saved.
Queue delays include backlog from the batch; they are not per-token overhead.
The final synchronization and event queries are outside individual kernel
execution times. Profiling is enabled for every variant consistently.

Layout order reverses on alternate repeats. Results retain every sample, with
thermal/power state before and after; no clock or thermal-policy overrides are
used. The screen is awakened without dismissing the lock screen. Device-side
45-second timeouts bound each owned benchmark process. Compilation, data
initialization, and validation can dominate process wall time; only warmed
execution events rank kernel performance.

The first measurements and all raw samples are in the
[experiment report](../../../../../docs/android-feasibility/PACKED_PROJECTION_BENCHMARK.md).
The best tested layout, `group_major`, reduced the synthetic projection-family
estimate by about 3%; bit-plane packing did not add a convincing gain. These
results do not establish a model-level token-rate improvement.

## Iterative pass and selected candidate

`EXPERIMENTS.md` and `QUEUE.md` are working notebooks for the investigation.
`iterate.py` derives individual experiments from the production Mojo source;
`autotune.py` screens a bounded categorical grid and confirms finalists. Build
artifacts stay outside the repository. Local defaults can be overridden with
`PROJECTION_MOJO`, `PROJECTION_PACKAGES`, `PROJECTION_NDK`, `PROJECTION_ADB`,
`PROJECTION_OPENCL_HEADERS`, `PROJECTION_OPENCL_LIBRARY`, and `ANDROID_SERIAL`.

The later pass selected `t4_g8_w64_u1_p32`: four outputs per lane, eight groups
joined, 64 threads/workgroup, one-word loop, and 32-output physical weight tiles.
The four FP16 accumulation chains and numeric code mapping remain unchanged.
This is **not** compatible with the old weight order or old launch X dimension.
Its weight order is `[group/8][output/32][word][group%8][output%32]`; scales remain
`[group][output]`. For the supported complete tiles, launch global size
`(2*width, reduction/1024)` with local size `(64,1)`, then run the existing
partial reduction with `reduction/1024` rows. Storage remains exactly two bits
per weight. [Selected configuration](../../../../../docs/android-feasibility/packed-projection/iteration/selected.json).

```sh
python3 max/examples/android-llm/bonsai/projection_bench/iterate.py \
  --build --out /tmp/projection-selected \
  --names original control reduce t4_g8_w64_u1_p32

python3 max/examples/android-llm/bonsai/projection_bench/iterate.py \
  --run --pipeline --random-inputs --seed 314159265 \
  --out /tmp/projection-selected \
  --names original control t4_g8_w64_u1_p32 \
  --shapes 6144x2048 2048x2048 2048x6144 4096x2048 512x2048 \
  --repeats 5 --output /tmp/projection-final.json
```

The iterative host warms at least 256 dispatches as well as touching every bank.
`--pipeline` includes the actual Mojo partial-reduction kernel in event timing;
component timings are recorded separately. Changing groups joined changes the
partial shape, so validation compares complete outputs and independently checks
sampled partials. Group-eight candidates require exact numeric equality for
all partials; alternative groupings allow tight FP32 final-sum tolerance for
non-dyadic inputs. The selected candidate uses group eight and requires exact partial comparisons.

The bounded tuner covers output tiles 4/8/16, groups joined 4/8/16, workgroups
64/128/256, and word unroll 1/2. It retains every micro-screen result, rejects
only >10× execution cliffs, profiles survivors on major shapes, and confirms
five finalists on all shapes with five alternating sweeps. Its initial screen
is exploratory; the confirmation and independent-seed validation decide which
candidate to retain.

Optional `counters.h` instrumentation is enabled only with
`-DPROJECTION_KGSL_COUNTERS` and an external pinned KGSL UAPI header. The capture
helper additionally accepts `PROJECTION_KGSL_UAPI` for that header directory.
Use the existing counter wrapper to enable access temporarily and restore it:

```sh
python3 max/examples/android-llm/with_gpu_counters.py \
  --adb "$ANDROID_SDK_ROOT/platform-tools/adb" --serial DEVICE_SERIAL -- \
  python3 max/examples/android-llm/bonsai/projection_bench/profile_counters.py \
  --build /tmp/projection-selected --candidate t4_g8_w64_u1_p32
```

Counter capture uses the original diagnostic build's `group_major_load.cl` and
the arithmetic experiment build's `halfbits.cl` as controls. See the report for
those build artifacts. Every accepted window must match expected compute-thread
invocations exactly. GBIF bytes include system-cache-served requests; they are
not DRAM-only traffic. SP8/SP9 report relative resident wave contexts, not an
occupancy percentage. Counter timings are separate from performance ranking.
