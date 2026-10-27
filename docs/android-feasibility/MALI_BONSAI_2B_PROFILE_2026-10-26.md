# Why the Mali Bonsai 2B baseline was slow

On the OnePlus CPH2717 / Mali-G615 MC6, the original portable Mojo model was
not inherently limited to the previously observed 1.75 tok/s. An opt-in
OpenCL queue flush after each kernel dispatch changed the measured `tg128`
rate to 10.07 tok/s and retained the 32-step logit agreement. A same-plugin
on/off control reproduced the effect. The evidence points to an interaction
between the large number of queued commands and this driver's submission or
scheduling behavior; it does not identify the driver's internal root cause.

The experiment uses the same model, device, workload and comparison boundary
as the [initial Mali report](MALI_BONSAI_2B_2026-10-26.md): 128 decode
forwards, a synchronization and full-vocabulary transfer per forward, no
sampling, and one-token warmup. Temperature and clocks were not controlled.

| Configuration | `tg128` tok/s | Repetitions | Notes |
| --- | ---: | ---: | --- |
| Portable Mojo, original staged plugin | 1.746 ± 0.001 | 3 | Earlier baseline |
| Same rebuilt diagnostic plugin, flush off | 1.746 | 1 | Control |
| Same rebuilt diagnostic plugin, flush on | 9.720 | 1 | Following control, 10 s pause |
| Rebuilt diagnostic plugin, flush on | 10.071 ± 0.009 | 3 | One process, shared warmup |
| Direct-Hadamard experimental model | 7.903 ± 0.068 | 3 | Original plugin, no forced flush |
| Direct-Hadamard plus 980 extra tiny launches | 1.723 | 1 | Diagnostic control |
| Prism llama-bench / Vulkan | 6.729 ± 0.005 | 3 | Fresh rerun after controls |

Three-repetition values are mean ± sample standard deviation. Runs are
sequential rather than randomized paired trials; the single-sample controls
show a large effect but do not quantify sustained thermally controlled speed.
Prism uses PTQ1_0, FP16 KV caches and Vulkan; Mojo uses the corresponding
packed values, FP32 KV state and OpenCL. The flushed path does not establish a
universal language or backend advantage.

## Evidence for the queue interaction

The baseline launches **1,701 kernels per forward**, including 98 Hadamard
transforms, each implemented as an initialization, nine butterfly stages and a
finalization: 1,078 Hadamard launches. A diagnostic version computed each
512-element transform in a single direct kernel. It reduced the count to 721
and measured 7.903 tok/s, despite using a more computationally expensive direct
transform. In a nearby one-sample A/B rerun, the baseline measured 1.743 and
that version 7.984 tok/s.

To distinguish launch count from the new arithmetic, a second diagnostic
added ten tiny scratch-only launches after each direct transform. It has the
same 1,701 launches as the baseline, the same output computation as the direct
version, and fell to 1.723 tok/s. Its eight-token logit validation passed. The
direct version passed the 32-step all-vocabulary fixture: 32/32 matching
argmax IDs, relative L2 0.000836567, maximum absolute error 0.0401869.

A short `tg8` sweep added three, four, five, six or eight tiny launches per
transform, for respectively 1,015, 1,113, 1,211, 1,309 and 1,505 launches per
token. Two-repetition means were 7.50, 7.38, 7.34, 2.17 and 1.74 tok/s. In a
separate 24-token timing run, the 1,211-launch variant stayed at 128–145 ms per
token. The 1,309-launch variant started near 134–140 ms for two measured
tokens, then remained around 533–547 ms. These short sequential diagnostics
show a nonlinear effect, not a proven precise hardware limit.

The current OpenCL plugin already supports the opt-in environment variable
`MODULAR_OPENCL_FLUSH_AFTER_ENQUEUE=1`; a diagnostic plugin was built from the
current working-tree source to test it. It is **not** automatically available
in the older staged plugin used in the initial report. The flush-on and
flush-off controls above used the exact same rebuilt plugin and model. The
flush-on 32-step check had exactly the same reported error and IDs as the
baseline: relative L2 0.000827509, maximum absolute error 0.0334730, 32/32
argmax agreement. No global default or existing S26 app was changed.

Host instrumentation on short runs measured roughly 17–20 ms spent in the
1,701 launch calls without flushing, followed by roughly 543–556 ms inside a
blocking device read. That read waits for earlier queued GPU work, so its time
cannot be interpreted as logits-copy bandwidth or pure transfer cost. With
flushing enabled, launches took roughly 67–70 ms including 40–45 ms in
`clFlush`, while the subsequent blocking read took roughly 20–25 ms. CPU and
GPU activities can overlap; these intervals are not an additive decomposition
of end-to-end latency.

## Profiling limits and Prism comparison

The initial OpenCL event summaries reported about 73–75 ms total kernel time
and a roughly 529–534 ms first-to-last span for a baseline forward. A detailed
trace revealed implausible differences between the driver's queued/submit and
start/end timestamps, and one approximately 450 ms discontinuity whose position
varied between forwards. Accordingly, those event timings do **not** establish
where the GPU was idle or prove which kernel caused the gap. The wall-clock
and controlled launch/flush experiments above support the queue interaction;
they do not distinguish driver batching, synchronization or other internal
mechanisms.

Prism's built-in Vulkan performance logger identified its PTQ1_0 matrix-vector
operations, output head and other graph operations. It recorded 3,707
operation/fusion entries across the warmup and four timed forwards. Its logger
inserts timestamps, extra synchronization and barriers, and its entries are not
a one-to-one count of Mojo kernel launches. Its profile timing must not be
compared directly with the OpenCL events or uninstrumented throughput. Prism
was validated against the 32-step fixture in the initial report, and its
uninstrumented fresh `tg128` rerun measured 6.729 tok/s as above.

All [raw profiles, thermal snapshots, validation records, source snapshots and
binary hashes](mali-bonsai2-profile/) are retained. The portable model and
experimental generators are under
[`bonsai_portable`](../../max/examples/android-llm/bonsai_portable/README.md).
The direct O(512²) transform and artificial-launch variants are diagnostic
experiments, not promoted as production defaults; the eight- and 32-step
fixtures do not establish long-context accuracy or broad quality.
