# Private prioritized experiment queue

1. [measured: no independent win] E01 UInt32 intra-matrix address arithmetic. Current tile indexing
   uses UInt64 despite each individual matrix being small; inspect emitted code
   and test whether less address arithmetic matters.
2. [measured: no independent win] E02 tile scale storage to match group-major weights. Scales retain
   strided row-major accesses even after weight repacking; preserve exact bits.
3. [rejected: 3–6% slower] E03 partial pair unroll (four pairs per runtime iteration), keeping
   four chains and their arithmetic order. Full runtime pair loop previously
   lost; this tests an intermediate code-size/live-state tradeoff.
4. [rejected: full-shape GPU execution ~103ms] E04 straight-line half-bit decoding without vector comparisons or
   selects. Four code values map exactly to -1/0/1/2; earlier compare/select
   variants had compiler stalls. Use bounded standalone processes to isolate it.
5. [measured: sub-percent/no material win] E05 UInt16 unpack/conversion. Determine whether narrower operations
   reduce work, rather than repeating the known-bad int8 path.
6. [rejected: ~7% slower] E06 one-word prefetch, preserving four chains. Test overlap against
   increased live state; inspect generated source first.
7. [measured: no family win; retain tuning interaction] E07 two-word unroll. Test instruction scheduling independently of
   pair-loop structure; watch build/enqueue latency.
8. [rejected: 14–21% slower] E08 subgroup activation broadcast. Shared input reads may already
   be served efficiently; explicitly test whether shuffles are worth their cost.
9. [complete: 54 screened, 36 full-shaped, 5 finalists confirmed] Bounded autotune over output tile, groups joined,
   workgroup size and surviving unroll configurations. Include final partial
   reduction when changing groups; validate complete outputs, not timings alone.

New findings can add or reorder items. Mark measured losses, invalid variants,
compiler failures and analytical rejections explicitly; do not silently delete.

10. [rejected: no full-shape family win] E09 four-word unroll: small word2 screen improved despite a larger
    private query; test whether more static scheduling helps or crosses a cliff.
11. [rejected: multi-ms execution regression] E10 explicit vector activation-pair loads, alone and with word2.
    Source still has scalar input loads; test reuse/address scheduling directly.
12. [closed by source inspection] E11 manual tiled-address hoisting: LLVM already
    hoists invariant address terms; loop contains only incremental shift/add/OR.

13. [closed: emitted OpenCL identical to tested u32] E12 two-pair activation loads instead of eight. Profile-driven test
    of whether a smaller explicit load avoids the observed private-state cliff.

14. [complete: 25 valid byte/thread/context captures] E13 calibrated GBIF/SP capture for control/load/finalist
    and one pathological decoder. Determine whether normal traffic is near the
    packed payload, and whether slow variants add interface traffic. Validate
    launched-thread counts; do not infer DRAM-only bandwidth or occupancy.

15. [complete: p32/WG64 wins heldout3 pairs/all5 shapes] E14 match physical output tile to winning SIMD4
    wave's 32 outputs, rather than the inherited 64-output packing. Existing
    accesses cover alternating half-tiles; test locality rather than assume it.
16. [rejected: no reliable extra gain] E15 UInt16 unpack on the winning smaller output
    tile. Earlier standalone gain was sub-percent; check this one interaction.

Disposition: all items closed for this pass. Keep SIMD4/g8/WG64/u1/p32.
Normal traffic is near packed payload; this does not establish a device ceiling.
Model integration and actual token-rate validation are outside this standalone pass.
