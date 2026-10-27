# Adreno 840 counter calibration

These are the pre-tuning measurements. The subsequent selected subgroup pass
reduces traffic to578.65MB/token; see [tuning evidence](counter-tuning/summary.json)
and the Bonsai PERFORMANCE.md. Calibration rules below remain applicable.

Device DEVICE_SERIAL, Adreno840v2, 12 reported OpenCL compute units. This pass changes no production kernels. Raw evidence and artifact hashes are in [counter-calibration](counter-calibration/summary.json). Values below use decimal MB/GB.

## Validated interpretation

| KGSL group / selector | Interpretation | Control |
|---|---|---|
| GBIF 13 / 34,35 | Read bytes = 32 × sum of both channel beat counts | Known 32/64/128 MiB float4 scans, 4/8 repetitions; measured/expected 1.00009–1.00050 across early/late allocation orders |
| GBIF 13 / 46,47 | Write bytes = 32 × sum of both channels | Known float4 copy and scan outputs; matching payload plus small overhead |
| SP 10 / 100 | Compute thread invocations | Exactly global work size × launches across geometry controls and all 1,198 captured model dispatches |
| SP 10 / 8 divided by 9 | Busy-cycle-weighted resident wave contexts per active shader processor | One 64-thread workgroup: 1.000; one 256-thread workgroup: 3.999; larger grids increase ratio |

These GBIF counters measure traffic at the GPU external/system-memory interface. They do **not** distinguish DRAM accesses from requests satisfied by a system cache beyond that interface. Resident contexts are not an occupancy percentage: no validated maximum-residency denominator is available.

Selector names come from pinned [Mesa a8xx definitions](https://chromium.googlesource.com/external/gitlab.freedesktop.org/mesa/mesa/+/d2c4653ee953dba73f0be8e9eab872269813582b/src/freedreno/registers/adreno/a8xx_perfcntrs.xml). Qualcomm's [gen8 definitions](https://github.com/qualcomm-linux/kgsl/blob/2b2d762c0b843c560d2ae2203f6b3cfe1474fb3f/gen8_reg.h) independently identify GBIF selectors 34/35/46/47. The byte unit is established by measured payloads, rather than inferred solely from a bus-width declaration.

## Controls and rejected readings

- Float4 copies sustain approximately 65–69 GB/s combined read+write traffic; scans approximately 60–67 GB/s. This is a workload reference under current device conditions, not a universal bandwidth ceiling.
- Lane-varying arithmetic with 2/4/8/16/32 live float4 accumulators reports private storage of 48/80/144/272/528 bytes per work item. Resident contexts fall from 20.90 to 5.63. The two-accumulator variant is slower despite higher residency: preserving instruction parallelism matters.
- Initial uniform arithmetic could execute through scalar paths. Arithmetic conclusions use the later lane-varying controls. The earlier geometry controls remain useful for thread/context scaling.
- SP ALU-working selector 2 and full/half MAD selectors 109/110 stay zero even on lane-varying arithmetic. UCHE selectors scanned over 0–207 also stay zero. Do not interpret these as zero arithmetic/cache activity.
- SP non-execution and load/store non-execution selectors respond strongly to a serialized dependent-load control, but overlapping counter domains and cycle aggregation prevent a defensible wall-time stall percentage. GPR conflict counters are responsive but their units are insufficiently calibrated for conflict percentages.
- SP selector 97 does not count waves consistently across workgroup modes; reject it as a general wave counter.
- A long dependent-load window produced impossible timestamp/counter deltas and an incorrect invocation count. Idle windows also produced invalid deltas. These are excluded; no precise reset mechanism is established. Require bounded monotonic timestamps and expected invocation counts before accepting a capture.
- `CL_KERNEL_PRIVATE_MEM_SIZE` is private storage per work item, **not a hardware register count or proof of spilling**. See the [Khronos definition](https://registry.khronos.org/OpenCL/specs/unified/refpages/man/html/clGetKernelWorkGroupInfo.html).

## Real MAX decode

Normal counter-enabled decode: 24-token prompt, 95 forward steps, all 96 generated token IDs checked, two measured runs 56.07/56.64 tok/s. One representative run reads **589.12 MB/token**, writes **69.89 MB/token**, total **659.01 MB/token**. Read/write totals agree closely between runs. This exceeds the 507.98 MB exported weight payload because state and intermediates also move.

Separate per-dispatch captures cover all 24 layers, 599 launches per forward, two forwards. All generated IDs and reference logits pass (maximum absolute error 0.01711, relative L2 0.000754). Counted compute invocations exactly match 3,852,544 launched threads. Instrumentation serializes with `clFinish`; its 2.44 tok/s is **not** a model performance measurement. It measures attribution, with perturbation: 590.89 MB read / 73.07 MB written per token versus 589.12 / 69.89 in ordinary execution; decode positions also differ.

| Family | Read MB/token | Write MB/token | Resident contexts | Normal GPU time/token |
|---|---:|---:|---:|---:|
| Packed layer projections | 367.97 | 43.54 | 6.36 | 8.09 ms |
| Vocabulary projection | 135.15 | 1.00 | 5.74 | 2.23 ms |
| Recurrent DeltaUpdate | 23.72 | 19.10 | 2.67 | 1.23 ms |
| Rowwise kernels, combined | 45.54 | 2.98 | mostly 1–2 | 1.12 ms |

Timing comes from an earlier ordinary event-profile capture, not the serialized counter run. These are comparative measurements, not simultaneous bandwidth/clock-normalized estimates. Total normal GPU work is 15.17 ms/token, with 16.09 ms first-to-last GPU span.

## Ranked next experiments

1. **Packed projections: reduce live temporaries while preserving reuse and ILP.** Driver private-storage queries give 512 bytes/work item (head 560). Larger projection grids reach about 6.9 resident contexts; the live-state control shows a credible residency constraint worth testing, without proving spills. Smaller 1,024-thread projections reach 1.32 contexts, close to their grid ceiling of 1.33; those need more useful parallel work, not merely fewer registers. Previous SIMD4/fewer-chain attempts lost, so do not repeat blindly.
2. **Reduce global projection partials.** Packed layers write 43.54 MB/token; rowwise kernels read 45.54 MB/token including reduction inputs. Cooperative/tiled matvec reduction could remove much of the partial write/read traffic while retaining the standard kernel/graph path. Host sequence batching was tested previously and is a different optimization.
3. **Expose more recurrent-update parallelism.** DeltaUpdate launches only 2,048 threads: 32 workgroups on 12 CUs, or 2.67 wave contexts/CU. Measurement reaches that ceiling, with only 96 private bytes/work item. Each thread owns a state column and performs two serial 128-element loops. Tiling/cooperating across that dimension is a better hypothesis than reducing its registers.
4. **Vocabulary head is a lower priority.** Its 135 MB read payload is close to the expected packed weights plus scales, with no evidence of doubled weight reads. Its separate timing implies roughly 61 GB/s including writes, already near the scan reference. Bigger gains likely require reducing representation/traffic rather than launch knobs alone.

At current traffic, 659 MB divided by the 65–69 GB/s copy reference consumes roughly 9.5–10.1 ms. This is a rough budget, not a hard roof: workload mix and system-cache behavior differ. Reaching 100 tok/s likely requires both less intermediate/state traffic and better execution of packed projections; eliminating host overhead alone is insufficient.

## Reproduction and state

Host cache: `/Users/REDACTED/Library/Caches/modular-android-integration`. Controls: `calibrate_counters.cpp`, `calibrate_copy.cpp`, `calibrate_geometry.cpp`, `calibrate_registers.cpp`, `calibrate_latency.cpp`, and matching `calibration*_run.py` scripts. Whole-model builder: `calibrated_counter_build.py`; cache-only instrumented HAL builder: `calibration_kernel_capture.py`. Artifacts: `bonsai-calibrated-counters` and `bonsai-kernel-counter-capture`. The instrumented HAL is not installed into production sources.

Wrap commands with `max/examples/android-llm/with_gpu_counters.py --serial DEVICE_SERIAL --adb <adb-path> -- <command>`. The shell-writable KGSL perfcounter switch is temporarily enabled and restored in `finally`. It is confirmed restored to **0** after this pass; no root, clock override, or SELinux change was used. Counters are global GPU measurements, so unrelated GPU activity remains a possible contaminant. Retain known-work controls and invocation-count checks in future captures.
