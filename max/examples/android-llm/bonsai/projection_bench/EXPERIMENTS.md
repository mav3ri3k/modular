# Private experiment notebook

Protocol: production-shaped Mojo kernels; independent numerical validation before
GPU event timing; rotating >=128 MiB weights and scales; alternating candidate
order; no clock overrides. Retain failures and all samples. Compare against a
fresh group_major control, not historical timings. Production model is unchanged.

Starting evidence: group_major ~6.442ms synthetic family vs original ~6.641ms.
Load-only is close to full; memory locality matters. Bit planes added no clear
benefit. Prior SIMD4 / fewer accumulators / full runtime pair loops lost with
older layouts; interactions with the new layout remain untested.

2026-10-26: iterative pass started. Queue is QUEUE.md. No new speedup claimed.

E01/E02: 27 production-shaped 128MiB ring processes, 3 repeats, 3 major shapes.
All exact checks passed. Median us (control/u32/tiled-scales):
6144x2048 56.003/55.971/55.787; 2048x2048 23.233/23.227/23.344;
2048x6144 56.175/55.974/55.978. No meaningful family win demonstrated;
scales regress square shape slightly. Do not promote either alone. E01 remains
available as a parent for arithmetic/scheduling experiments (sub-percent impact).
Raw: /tmp/projection-E01-E02.json. The new host warms at least 256 dispatches,
records private-storage queries, and can test non-dyadic activations.

E04 preliminary: exact small fixture passes, but 0.745ms vs ~0.0083ms.
Private-storage query 3224B vs control320B; not a validated spill/register count.
Generated source grows 37.3KB ->53.7KB. Full-shape bounded check pending.

E04 full-shape bounded diagnostic: valid exact 6144x2048 result. Build321ms,
first enqueue0.147ms, first device127.4ms, warm single device102.8ms. This
particular variant is GPU-execution slow, not stuck in compilation/enqueue.
Reject; do not generalize this result to every prior half-bit implementation.
Raw /tmp/projection-halfbits-full.log. Only 1 warmup/1 timing to bound cost.

Source profile: group_major/u32 already hoists invariant tiled addressing out of
the word loop. Generated word-loop address work is essentially shift/add/OR;
manual hoisting has no clear missing work to remove (u32.cl lines648-686).
Do not add a redundant rewrite. Small-fixture word2 looked faster despite
private query384B vs320B; follow up on full shapes and word4/input-vector reuse.

E03/E05/E06/E07/E08: 63 production-shaped ring runs, 3 major shapes, 3 repeats,
all exact numerical checks pass. Median us by shape (control/u32/compact4/
narrow16/prefetch/word2/broadcast):
6144x2048 56.25/56.15/59.27/55.90/60.02/56.44/64.80;
2048x2048 23.28/23.25/23.95/23.27/24.97/23.08/28.21;
2048x6144 56.22/55.92/59.24/56.01/59.98/56.70/64.68.
Reject partial-pair loop, prefetch, and activation shuffles. word2 small-fixture
win does not transfer to the big shapes; retain only as autotune interaction.
UInt16 path is at most sub-percent; no material independent win. Source sizes:
compact4 33KB, prefetch38KB, word2 50KB, broadcast49KB. Lower source size is
again insufficient. Raw /tmp/projection-E03-E08.json.

E09: 27 refreshed full-shape runs, exact checks pass. Median us {'6144x2048': {'control': 56.294, 'word2': 56.191, 'word4': 56.484}, '2048x2048': {'control': 23.425, 'word2': 23.152, 'word4': 24.971}, '2048x6144': {'control': 56.181, 'word2': 56.813, 'word4': 56.608}}.
No family win for word2 or word4. Small-fixture ranking was misleading.
Raw /tmp/projection-E09.json.
E10: input-vector loads passed exact checks but first6144x2048 measured13.67ms
(inputs8) and17.59ms (word2+inputs8), vs0.0568ms control. Private queries932/1684B.
Square-shape checks also slow3.33/6.98ms. Stopped remaining batch after these
large regressions, preserved every completed result in /tmp/projection-E09-E10.json.
No statistical filtering; these are rejected screens, not aggregate comparisons.
E12 queued: two-pair loads could avoid the full-vector live-state cliff.

E12 closed by compilation: explicit two-pair loads compile byte-for-byte to the
already tested u32 kernel (SHA b1054735...). LLVM scalarizes this case; no new
execution to measure. Larger input vectors are rejected from full-shape screens.
Manual queue exhausted. Next: bounded categorical autotune, with projection plus
final reduction timed together. Added generic wave indexing for multi-subgroup
workgroups, candidate partial geometry validation and independent final outputs.

Autotune: 54 categorical configurations (tile4/8/16, groups4/8/16,
WG64/128/256, word unroll1/2). All small-fixture checks pass; 18 SIMD16 points
rejected after >10x micro execution cliffs, with each result retained.
36 survivors screened on 3 major full shapes, 128MiB ring, controls every6
candidates. 5 finalists confirmed in5 alternating sweeps across all5 shapes:
150 confirmed processes +30 independent-seed/non-dyadic checks; all pass.
Pipeline means include the actual production partial-reduction kernel.
Synthetic family medians: prior group_major control6.850ms;
tile4/g8/WG128/u1 6.715ms; tile4/g8/WG64/u1 6.719ms.
Both improve ~2%, with individual paired reductions~1.5–2.6%.
The earlier single screen favored u2/WG128; confirmation favors simpler u1.
WG64 vs128 is near-tied globally. Do not attribute the earlier tile4 failure
under older packing to intrinsic inferiority; the interaction matters.
Raw artifacts /tmp/projection-autotune/{micro,screen,shape-screen,ranking,
confirm,accuracy}.json. One accuracy-control timing outlier is retained;
accuracy timings are not used for ranking. Numerical comparisons pass exactly
for all group8 partials, including non-dyadic activations.
New follow-ups E14 (physical32-output tiles) and E15 (narrow cast on tile4).

E14/E15: 54 paired ring processes, all numerical checks pass. Major-shape family
score (66/42/24 calls), median across3 sweeps, ms:
SIMD4/WG64/p64 6.358; SIMD4/WG128/p64 6.366;
SIMD4/WG64/p32 6.294; SIMD4/WG128/p32 6.459;
SIMD4/WG64/narrow16/p64 6.354; SIMD4/WG64/narrow16/p32 6.328.
Select p32 with WG64; ~1% major-shape improvement. Reject narrow-cast addition
and WG128+p32. Fresh final cohort uses original, prior group_major, selected
candidate; 5 repeats/all5 shapes, new random-input seed314159265,128MiB ring.
The native host now hoists packing-configuration lookup out of CPU per-word
repacking. This changes setup cost only; final controls share the same host.
Raw /tmp/projection-E14-E15.json; no whole-model speedup claimed.

Final held-out cohort:75 full-shape processes,5 sweeps, random/non-dyadic
activations, seed314159265,128MiB ring,512 measured launches, includes reduction.
Exact numeric partial equality to original baseline in every bank. Median
synthetic pipeline family: original7.036884ms, prior group_major6.836508ms,
selected SIMD4/g8/WG64/u1/p32 6.685752ms. Paired latency reductions vs original
4.85/4.47/4.96/5.06/5.21%; vs prior control2.12/1.64/2.45/2.28/2.27%.
Raw /tmp/projection-final.json. No real-model token-rate result.
E14 holdout:30 additional paired processes,3 sweeps/all5 shapes, same new seed.
p64 parent6.748932ms vs p32 6.691416ms median family. Paired p32 benefits
~0.8–1.1%, confirming the packing change is worth retaining. All checks pass.
Raw /tmp/projection-packing-holdout.json. E14 closed; selected descriptor pending
final hardware-counter sanity check. E13 is the only remaining queue item.

E13 completed:25 calibrated KGSL windows; all invocation counts exactly match
expected launches. Counter switch restored to0. At6144x2048, normal original,
prior group_major, load-only, and winner all read~3.35MB/dispatch and write~0.050MB.
Packed weights+scales are3.342336MB, so normal interface read traffic is close to
payload. Winner increases busy-cycle-weighted resident wave contexts from~9.46
(prior control) to~13.79 while preserving traffic; this is NOT occupancy percent.
Load-only ~54.65us, winner~55.70us, previouscontrol~56.16us in the second capture.
The specific rejected half-bit decoder reads1249.33MB and writes1548.46MB for
one3MB matrix projection, with correct12288-thread count and~100.6ms GPU time.
Its source has only the ordinary partial-output global stores; the extra traffic
is consistent with compiler-generated private-memory spills/materialization.
This is a traffic regression, not evidence of an intrinsic cost for all bit
manipulation. Disassembly/spill counts remain unavailable. Interface bytes are
not DRAM-only; counters do not prove a global hardware ceiling.
Raw /tmp/projection-final-counters.json; all windows validated; every slot released.
All current queue items are closed. No remaining material candidate identified
within the existing four-code2-bit workload contract. Selected config retained.

Closure: raw records archived under docs/android-feasibility/packed-projection/
iteration/. Selected.json records exact packing/launch requirements and CL hash.
Final clean rebuild in /tmp/projection-selected reproduces measured OpenCL hash
f06aaf9b...; current generator/native source fingerprints match build manifest.
Six additional non-dyadic sanity checks (new seed271828183, small+full shapes)
pass with the rebuilt host. Python lint/syntax and C++ formatting/warnings pass.
No production model/app changes or commits. Profiling switch0, no owned benchmark
process remains, thermal0. All queue items closed; no global-ceiling claim.


Model rollout decision (2026-10-26): rejected as the default after noisy paired
model results (baseline77.01/77.36, tiled78.10/66.76 tok/s). Restored both direct
and MAX graph dispatch and original checkpoint layout; retained experimental
kernel and standalone records. A fresh restored-path comparison determines the
chat demo backend. Prior statements of no production changes describe the
standalone pass, before the subsequent integration experiment.
