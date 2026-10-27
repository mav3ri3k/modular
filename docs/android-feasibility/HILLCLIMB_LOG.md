# Bonsai hill climb — experiment log

Protocol: same weights/prompt/forced IDs; paired, interleaved warmed runs. GPU traces separate from throughput. Keep selected changes only after full-logit validation. No app updates during tuning.

- Starting point (2026-10-25): SIMD8/4 half accumulators; Mojo 53.74, MAX 49.90 tok/s (3 short ID-checked repeats). MAX 599 vs Mojo721 dispatches/token; GPU16.50 vs16.45ms, span19.89 vs17.35ms. Full logits validate. These are short measurements, not sustained ceilings.
- Prior rejected directions: shared ternary-pair LUT (33.4 vs48), standalone GPU argmax (45.6 vs53.7); scalar WG changes, scale-only weights, two accumulator chains and grouping/head-WG variants did not win reliably. Keep four chains.
- Bandwidth diagnostic: clpeak float8 ~68GB/s; float16 ~42.6GB/s. Not model DRAM counters. Weights resident; no intermediate CPU tensor roundtrips. Blocking final logits copy remains.
- E01: half-bit decoding exposed invalid short-vector comparison normalization (`shortN & int`). Fixed emitter to convert masks then normalize with typed ucharN one; focused generated-C regression + existing emitter suite pass. Native tools rebuilt/snapshotted.
- E02: branch/select ternary and half-bit decode variants failed phone execution (select process killed; half-bit stalled, terminated owned runner). No numeric/timing result; reject these variants for now, no model-source change.
- E03 ongoing: enable normal MEF side-stream pool for OpenCL. Initial MAX49.9 vs Mojo53.2tok/s resembles old result; queue creation alone not a speed claim. Compare stream-count1/2 and CPU worker counts next.
- Reference inspected: PrismML-Eng/mlx prism bbc151c6360ee79a7412fa23cfc5c5fc63ced1ca, quantized.h qmv_fast/qmv_quad/qdot. Uses cooperative K-lanes + SIMD reduction, reuses inputs across outputs, hoists powers-of-four scaling for affine2bit. Test cooperative K splitting first; affine subtraction may change cancellation/precision.
- E04: changing MAX runtime.num_threads aborts because CompilerRT stdlib getter re-requests defaults after MAX creates customized global CPUDevice. Reuse existing options in stdlib getter (same contract as CPU DeviceContext); rebuild/test before interpreting worker-count tuning.
- E05: cooperative K lanes2/4 warmups51.61/47.05 vs baseline53.71tok/s; packed integer-bit variant also stalls, terminated owned runner. Rerun cooperative-only pairs; no winner yet.
- E06 KEEP: one exact FP16 scale/group, discard centered bias on upload. 96 outputs/95 forwards, 3 alternating pairs: baseline53.25 vs compact58.07tok/s (+9.1%). All96 IDs match. Short all-logit errors identical .0165186/.000786457; roundtrip check added to reject scales not exactly representable. Parameters shrink117506048→29376512B (saves88.1MB). Applying shared kernels and MAX external-weight registry next.
- E05 complete REJECT: cooperative lanes2/4 medians51.37/47.12 vs54.27. Standard shuffle works after scalarizing shuffle calls at source, but no speed win.
- E04 refinement: C API M_newDevice(CPU) constructs default CPUDevice before M_newRuntimeContext reads requested options. Stdlib reuse change fixes embedding-first order only; worker-count experiments still abort in device-first order. Do not claim complete fix. Preserve as separate follow-up; no worker tuning selected.
- E07 ongoing: arbitrary16-structured-kernel host-sequence limit likely drives37 MAX submission sequences. Add backend-neutral pipeline/pass option (default16 unchanged); test64/0 limits on same graph. No named/hardware fusion patterns.
- E08 REJECT: compact SIMD4 median50.64 vs compact SIMD8 58.01tok/s (95 forwards,3 pairs). Keep8 outputs/thread.
- E07 initial: cap16/64/0 medians53.58/54.46/53.05tok/s (95forwards,3 alternating sweeps); all96 IDs and every logit validate. 64 removes27 host calls (45→18), but gain only1.6%, needs confirmation; unlimited does not win. Focused fusion-limit check passes, defaults unchanged.
- Compact scales: long numbers prompt96 all-logit maxabs .0147266/relL2 .000397484, identical to original. Existing packed width/tail fixture passes phone. MAX transports exact half-scale bits as uint16 external storage (ARM CPU rejects f16 ops); custom op reinterprets only inside device closure. No extra transfer/conversion kernel.
- Stdlib runtime reuse regression unit passes. Device-first C API initialization remains a separate limitation for configuring worker counts; no experimental worker setting shipped.
- E07 final REJECT batching knob: second3-sweep old/new comparison cap16/64=53.11/52.89, previous53.58/54.46 not reproducible. Removed experimental pass/pipeline option and limit test; retain default16. Fewer host tasks alone does not close MAX gap.
- MAX confirmed gain, same runtime and 95 forwards: old49.23→compact/default16 53.11tok/s (+7.9%); three alternating sweeps, all96 IDs and full logits pass. Compact64 not selected.
- E09: int16 unpack58.44 vs57.84 (+1.0%, needs confirmation); eight accumulators55.87 (reject; keep4). Testing affine dot bias-hoisting (Prism qdot idea) and packed sign/zero additions next.
- E10 REJECT: packed sign/zero-add passes small fixture but full-model run stalls (same class as E02; terminated owned runner). No valid timing. Could be large/head-kernel resource or driver compilation/execution issue; not diagnosed, do not blame a specific device/compiler defect without trace.
- E09/E11 final REJECT: second3-sweep compact/int16/affine medians57.84/56.79/55.71. Int16's earlier1% gain not reproducible. Affine bias-hoisting validates short logits (.0361881/.0011105 vs .0165186/.000786457) but loses performance; retain original centered4-chain math.
- Final paired GPU trace (23 decode forwards): Mojo721 dispatch,14.987ms GPU/15.893ms span; MAX599,15.170/16.092ms. Largest MAX: projections8.086ms,head2.231,Delta1.233,rowwise1.117. Gap inside GPU span ~.92ms in BOTH; earlier extra2.55ms MAX gaps do not reproduce here. Remaining wall gap must be attributed with host-stage/CPU profiling, not assumed to be queue waits.
- E12 host stages: 95 forwards1.749s; execute1.6658s, lookup46us, finite23.44ms, argmax52.93ms. ~0.8ms/output CPU scan. Combined8 independent greedy lanes keeps first-index ties and finite checks; stage times ~24–28ms vs old finite+argmax42–99ms across96 outputs. CPU helper retained; native tails/ties/NaN/Inf checks pass. Whole-run comparison confounded by phone ThermalStatus1 / GPU execute variation1.65→3.61s; do not quote wall speedup from that batch. Cool before final timing, no policy/clock changes.
- Final selected sources: both runners pass96-token all-logit validation after combined CPU sampler (Mojo.0204191/.00064232,MAX.0200463/.000646198). Unit/emitter/generated-C suite pass. No app changes.
- Hardware counter breakthrough: direct KGSL QUERY/GET/PUT works for CP(14 slots),UCHE(24),SP(24). Cache probe uses Mesa's pinned KGSL syscall header; every acquired slot released. Unlike qprof SELinux failure, this route is available. Now collecting29 named A8xx events only around steady decode (after prefill). Do not assume event units/aggregation or convert beats to bytes without validation.
- Counter limitation confirmed: all29 READ ioctls return EPERM even though QUERY/GET succeeds; no raw values obtained. Released all acquired counters and removed device probe. No root/security-policy changes attempted. Cache/hill-counter-{build script,headers} and hill-counters-max logs preserve the precise failure. Kernel syscall header and Mesa A8xx event definitions pinned; no claim of bandwidth/ALU utilization from these failed reads.
- FINAL unprofiled selected-source medians (3 alternating95-forward runs, phone ThermalStatus0 before): Mojo59.11,MAX56.43tok/s, every96 IDs pass. Full-logit validation separately already passes. Isolated compaction gains remain+9.1%/+7.9%; do not attribute entire final improvement to sampler or compare unmatched protocols. Phone tools/tempdirs cleaned, app unchanged, no commits/staging.
- Primary reference: https://github.com/PrismML-Eng/mlx/blob/bbc151c6360ee79a7412fa23cfc5c5fc63ced1ca/mlx/backend/metal/kernels/quantized.h . Counter definitions from Mesa d2c4653ee953dba73f0be8e9eab872269813582b src/freedreno/registers/adreno/a8xx_perfcntrs.xml; KGSL header from9a6269f65f43ed01cc0604829ee1aec8aab1da77. Counter READ permission remains unavailable.

- Counter permissions RESOLVED (2026-10-25): phone already unlocked; `/sys/class/kgsl/kgsl-3d0/perfcounter` was0, mode0660 root:shell, SELinux vendor_sysfs_kgsl_shell. Qualcomm kernel explicitly returns EPERM with perfcounter disabled. Temporarily set1 as adb shell: QUERY/GET/READ/PUT all work, no root/SELinux changes. Restored0 after probe/model runs. Two95-forward Bonsai runs: all29 event reads succeed,7 selectors have repeatable nonzero deltas; every96 token IDs match.22 zeros / CP-vs-SP rate scaling mean event mapping/normalization still unvalidated, no utilization or DRAM-bandwidth conclusions. Cache/counter-permission-resolution.json + counters-enabled-model*.log. Portable wrapper max/examples/android-llm/with_gpu_counters.py enables only for the child command and restores original state (success and exit7 tested).

## Counter calibration / attribution

Validated GBIF32B/beat with known32–128MiB scans/copies; SP100 equals launched threads, SP8/9 scales with known wave contexts. MAX95-forward traffic589MBread+70MBwrite/token. Per-family capture validates logits/IDs and3,852,544 thread count. Packed projections368/43.5MB read/write,6.36contexts; DeltaUpdate23.7/19.1MB,2.67contexts equals grid ceiling. Head135MBread,near stream reference. Zero ALU/UCHE selectors and one reset-like long window rejected. No production tuning in this pass. Evidence/priorities: COUNTER_CALIBRATION.md; raw logs+hashes: counter-calibration/.

## Calibrated tuning, in progress

- T01 REJECT shorter runtime pair loop: 57.38 vs59.16tok/s, 95 forwards,3 alternating pairs, full logits/IDs pass. Reducing unrolled code alone loses.
- T02 candidate KEEP cooperative four-group projection partial reduction:65.77 vs59.40tok/s direct Mojo,3 pairs;96 IDs/full logits pass. Same dot arithmetic, standard scalar subgroup shuffles,4×fewer partials. MAX and combined testing pending.
- T03 candidate KEEP four-way recurrent K split:61.28 vs59.40tok/s,3 pairs;96 IDs/full logits pass. 8192threads vs2048, disjoint state ownership and subgroup joins. Combined/MAX testing pending.
- Next running: narrower int8 unpack+late scale loads; tile4+delta4; tile8+delta8. Sources/build scripts/artifacts in cache tune3*. Production source not yet changed.

- T02/T03 MAX validated: tile4 alone62.48, tile4+delta4 62.88 vs55.39tok/s (95forwards,3paired cycles); all96 IDs/full logits pass. Warmed last cycle drops several%, so use medians/interleaving, not peak.
- T04 REJECT int8 unpack+late scales: numeric IDs pass but~2tok/s; terminated owned runner on next sample; no production change. Pure late-scale variant still available for separate test.
- T05 in progress: tile8/delta8 warmups~68, measured~66; cached-state variant not clearly better. Larger tile16 projection and delta16cached compiled, testing next.

- T05/T06 cooled direct selection (12s idle gaps, thermal0 throughout): tile8+delta8 67.60, clean tile16+delta8 66.23, tile16+delta16cached66.59, late-scales-only59.65tok/s (two measured cycles). All96 IDs/full logits pass. Eight-way is preferred direct; MAX8-vs16 comparison pending because singleton partial reductions may fold away.
- Thermal note: long uninterrupted round4 eventually entered status1 and dropped~66→36tok/s. Raw results retained; not used as sustained ceilings. Final comparisons add idle gaps and record thermal status.

## Selected counter-guided pass complete

- KEEP tile8+delta8: MAX64.35 vs56.84tok/s (+13.2%),3interleaved cooled cycles, all thermal0; clean tile16+delta8 63.62. Final source direct smoke67.95. Ordinary DeviceContext/subgroup ops; final join remains MAX sum. Scratch8×smaller.
- Matched normal GPU profiles: PackedLinear8.607→7.147ms; allrowwise1.157→.384; DeltaUpdate1.272→.623; total15.864→12.941, span16.604→14.592; dispatches599 unchanged.
- Normal95-forward counter traffic546.19MBread+32.46MBwrite=578.65MB/token vs659.01 before. Serialized attribution: packed writes43.54→6.03MB/token; rowwise reads45.54→7.70MB/token. Recurrent contexts2.67→7.48; packed6.36→5.24 (private512→528; no register-pressure fix claimed).
- Final source packed/tail/all-code fixture, MAX/direct96 IDs+alllogits, and direct long numbers96 IDs+alllogits PASS. Countercapture4,368,640 invocations exactly equals launched threads. Counter switch restored0, temp device files removed. App unchanged.
- REJECT: dot pair runtime loop, int8 unpack (~2tok/s), late scales only (no gain), cached state (no robust gain),16group/16way larger variants (no selected win). Do not blindly retry. Data+hashes: counter-tuning/summary.json; full cache: tune3*. Clear gains, not100tok/s or exhausted hardware headroom.

## Batch-one HVX follow-up — 2026-10-25

- Built direct packed-W2 HVX GEMV (activation lookup + QFloat), avoiding HMX
  replicated matrix rows and per-token FP16 weight tiles. Corrected overlap via
  activation-copy event in ordinary DeviceContext; queue-tail read serialized
  the recent integrated path. `--backend hvx` is opt-in; HMX remains available.
- GPU normalization profile exposed repeated 128/256-element sums. Kept
  cooperative reductions for NormalizeQk/GatedNorm/HeadNorm: 0.795→0.131 ms/token
  combined. Shared MAX kernels and direct runner pass reference checks.
- Final five-round medians: startingGPU71.18, updatedGPU71.31, hybrid71.77 tok/s.
  Variation exceeds the small NPU difference; retain GPU-only default. First
  selection batches and every final sample are retained, not filtered.
- VTCM experiments reveal unreliable retained contents across idle despite
  allocation/backup/power votes. Refreshing weights restores correctness but
  adds staging cost. Excluded from production sources; see report for results.
- Full evidence/reproduction: [hexagon-decode](hexagon-decode/README.md).
  No app package, global clocks, thermal policy, commits or staging changed.
