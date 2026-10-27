# Bonsai 2 2B on OnePlus / Mali — October 26, 2026

**The text model runs correctly on this phone using a portable Mojo/OpenCL
baseline.** The existing Adreno-specific model does not run unchanged. The
portable implementation is substantially slower than Prism's Vulkan path and
is not an optimized Mali implementation.

Device: OnePlus CPH2717, MediaTek MT6897, Mali-G615 MC6 r1p3, Android 16 / API
36. The vendor reports OpenCL 3.0 and FP16 support. A 64-thread workgroup in the
OpenCL subgroup probe returned four 16-lane subgroups. The Qualcomm model uses
64-lane subgroup operations; its existing staged build allocated the model but
failed during its first forward with HAL status -19.

OpenCL discovery also depends on Android's linker namespace on this image. The
ordinary `/data/local/tmp` executable saw `CL_PLATFORM_NOT_FOUND_KHR` (-1001).
Running the executable from the shell-writable `/data/local/tests/vendor` path
allowed the vendor OpenCL driver to load. No system files, root privileges,
driver, clocks or thermal controls were changed. This is a shell testing
arrangement, not a completed application integration.

## Decode benchmark

`tg128`, tokens/second, mean ± sample standard deviation, three repetitions in
one process with one warmup. The Prism run preceded the Mojo run, with a
10-second pause; runs were not paired or thermally controlled. Full-vocabulary
host transfer is included; loading, compilation, tokenization and sampling are
excluded.

| Implementation | Mean ± SD | Individual samples (tok/s) |
| --- | ---: | --- |
| Mojo / generic OpenCL | 1.746 ± 0.001 | 1.747, 1.746, 1.745 |
| Prism llama-bench / Vulkan | 6.655 ± 0.029 | 6.682, 6.659, 6.624 |

The Mojo build retains the original 2B packed values and FP16 dot chains, with
FP32 KV and recurrent state. It removes fixed-width subgroup operations: the
Hadamard transform uses nine separate butterfly passes, projection group joins
and reductions run within work items, and the runner selects greedy tokens on
the host. The timed workload transfers the whole logits vector but does not
sample. Prism uses PTQ1_0 and FP16 KV caches with backend-specific arithmetic.
Mojo uses cache capacity 128; llama-bench derives its context from the workload.
These differences and the unoptimized portable kernels preclude attributing the
speed gap to the programming language. This is a single device experiment,
not a comparison with a newly measured S26 run.

## Validation

The portable build passed eight generated-token checks after a 24-token prompt
against the existing MLX FP32-activation fixture, with relative full-logit L2
error 0.000849557 and maximum absolute error 0.0194926. A separate 32-step
teacher-forced all-vocabulary comparison against the previously captured 2B
phone reference passed: relative L2 0.000827509, maximum absolute error
0.0334730, and 32/32 matching argmax IDs. These fixtures do not establish
long-context accuracy or broader model quality.

Prism Vulkan, without the S26-specific workaround flags, passed the same
32-step fixture: relative L2 0.000604932, maximum absolute error 0.0264123,
and 32/32 matching argmax IDs. Only this fixture qualifies the native path;
no broader driver correctness claim is made. The 2B publisher GGUF and export
were previously compared by the 773-check procedure in the
[October 26 report](INFERENCE_TESTING_2026-10-26.md).

The generated portable model, build and deployment instructions are in
[`bonsai_portable`](../../max/examples/android-llm/bonsai_portable/README.md).
[Raw results, device/thermal information, generated source snapshot, validation
and provenance](mali-bonsai2-2026-10-26/) are retained. The reference is the
same unmodified Prism `llama.cpp` commit as the earlier report,
`c96baf4abe24d11a50f326e40b6223b687449d63`.
