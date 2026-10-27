# Bonsai hill climb — ranked candidates

Read HILLCLIMB_LOG.md before repeating an experiment. Cache contains all artifacts and raw measurements.

Latest direct-runner follow-up: [HVX decode report](hexagon-decode/README.md).
Cooperative normalization is kept; HVX vocabulary routing stays opt-in because
its small throughput benefit does not consistently reproduce. Layer projections
remain dominant. Do not repeat retained-VTCM experiments without idle integrity
checks; allocation success did not guarantee retained contents.

Current selected MAX median64.35tok/s vs56.84 paired baseline (+13.2%); direct final-source smoke67.95. All96 IDs/full logits and second long-prompt direct validation pass. See counter-tuning/summary.json.

1. KEEP exact half scales: ~508MB weights vs596MB, Mojo+9.1%, MAX+7.9% in paired95-forward runs. Preserve per-group4-chain centered FP16 math.
2. KEEP tile8 partial joins + delta8 K split: packed7.15ms, rowwise.38ms, recurrent.62ms; totalGPU12.94ms, span14.59ms,599dispatches/token. Traffic578.65MB/token (546.19read+32.46write) vs659.01. Next packed arithmetic/live-state work must preserve reuse+ILP: private528B and contexts5.24 remain a limit hypothesis; shorter loops/int8/late scales lost. Recurrent now~39MB/interface traffic at.62ms: less obvious bandwidth headroom. Counter rules in COUNTER_CALIBRATION.md still apply; no spill/ALU/stall-percent claims.
3. Diagnose large bit-manipulation kernel stalls: small sign/zero-add fixture passes, full model stalls. Isolate head/group shape with submission/GPU tracing before drawing conclusions; do not rerun full-model stalled variants indefinitely.
4. MAX CPU submission profile (now more exposed: profile span minus GPU work1.65ms vs.74ms before): normal side-stream pool now enabled, but graph is serial and extra queues alone do not help. Larger sequence batches removed27 host calls without reproducible speed gain. Inspect per-dispatch argument encoding/stub costs and event handling with CPU traces.
5. Worker configuration: device-first C API creates default CPU runtime before MAX options apply. Stdlib now respects embedding-first runtime (unit-tested), but early device construction still blocks worker-count sweeps. Need proper initialization ordering, not ignored configuration mismatches.
6. Larger work: standard command capture if vendor supports mutable arguments; integer-dot activation quantization or a different packed representation. Require accuracy and bandwidth/ALU evidence, not assumed free speed.

Rejected in this pass: K-lanes2/4, SIMD4,8chains, affine bias hoist, int16 unpack (unreproducible), larger host sequence limits (unreproducible), bit-select/bitcast full-model stalls. Prior shared LUT and extra argmax also lost.

100tok/s remains a goal. Do not claim kernels close to hardware limit or all useful experiments exhausted. No app update during this pass.
