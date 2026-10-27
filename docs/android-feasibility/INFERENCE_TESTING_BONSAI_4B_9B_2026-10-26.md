# Android inference comparison: Bonsai 2 4B and 9B

This extends the [October 26 2B comparison](INFERENCE_TESTING_2026-10-26.md)
on the same Galaxy S26 SM-S942U1, Snapdragon SM8850 / Adreno 840. The
experimental text-only Mojo DeviceContext/OpenCL ports and reproduction steps
are in [`bonsai_large`](../../max/examples/android-llm/bonsai_large/README.md).
No app replacement, vision processing, NPU inference or Python inference on
the phone is involved.

## Decode: `tg128`

Mean ± sample standard deviation, three alternating paired runs, one sample
per process. Full-vocabulary transfer is included; model loading, compilation,
tokenization and host sampling are excluded. Every sample is retained.

| Model | Mojo/OpenCL (tok/s) | Prism Vulkan (tok/s) | Mojo / Prism |
| --- | ---: | ---: | ---: |
| Bonsai 2 4B | 37.34 ± 0.27 | 27.20 ± 0.40 | 1.37× |
| Bonsai 2 9B | 21.89 ± 0.40 | 16.69 ± 0.43 | 1.31× |

[Raw samples and thermal snapshots](inference-testing-bonsai-large/tg128/summary.json)
are retained. The runs used fixed pauses, not controlled clocks or temperature;
these ratios describe this test, not a general hardware ceiling. Prism uses
FP16 KV cache; Mojo uses FP32 KV and recurrent state. Both Mojo models allocate
capacity 512, while llama-bench derives its context from the workload. This
study's Mojo OpenCL plugin has SHA-256 `d4d3e07c2480cae539a7d05b17aaa5197585e5b75f0dd0a73f10a0582ed4d14e`;
it is not the exact packaged plugin used for the earlier 2B result. The 2B
report's 69.74 and 51.74 tok/s are historical measurements, not a newly
paired three-model experiment.

The fresh repository rebuild used for the subsequent S26 rerun produced a
different plugin hash, `5a67266344ef0096f07387958f0013a21389f4d1a63bdf225fc9893505447d6c`.
Its fresh 4B and 9B model-object hashes are recorded in the [rebuild
provenance](inference-testing-s26-rebuild-2026-10-26/provenance.json); the
archived hashes above describe the earlier study.

## Serial prefill: `pp512`

The Mojo results below are a fresh repository rebuild rerun, with three
separate processes per model. The Prism values are from the earlier study and
were not remeasured alongside this rebuild, so the table is not a paired
comparison. Both implementations use batch size one. This measures serial
prefill, not batched GEMM throughput.

| Model | Mojo/OpenCL (tok/s) | Prism Vulkan (tok/s) |
| --- | ---: | ---: |
| Bonsai 2 4B | 29.86 ± 0.91 | 12.73 ± 0.93 |
| Bonsai 2 9B | 15.26 ± 0.96 | 6.98 ± 0.22 |

[Samples and thermal snapshots from the earlier paired study](inference-testing-bonsai-large/pp512-serial/summary.json)
remain available. Fresh Mojo samples were 30.88, 29.56 and 29.13 tok/s for
4B, and 16.28, 15.14 and 14.37 tok/s for 9B. Thermal status rose from 0 to 2
during 4B and from 2 to 3 during 9B; skin temperature reached 45°C. The older
4B/9B snapshots show thermal status 2 throughout those prefill runs, so the
snapshots do not show the older runs at a higher thermal status than the
fresh runs throughout; the measurements cannot isolate thermal effects from
the changed build artifacts. The fresh run's raw outputs, thermal snapshots, validation and
binary hashes are in the [S26 rebuild archive](inference-testing-s26-rebuild-2026-10-26/README.md).
The token-wise checks cover the reference path; there was no separate
all-logit check for the full 512-token prompt workload.

The optional 4B Vulkan batched-prefill control (`-b 2048 -ub 512`) terminated
with a segmentation fault, even with both workaround flags, and produced no
benchmark result. Its [failure output](inference-testing-bonsai-large/pp512/failure.stderr)
is preserved. The 9B batched control was not run. No batched-prefill speed is
claimed; the serial suite continued with `--skip-batched`.

## Correctness and architecture

The publisher's GGUF files are PTQ1_0. After accounting for the GGUF's recurrent
head permutation, the exported 4B and 9B weights pass 1,029 and 1,033 checks,
respectively, including packed codes, scales, signs and the 9B output head.
The only tolerance is the existing conversion between A_log and its negative
exponential. [Detailed checks](inference-testing-bonsai-large/validation/) are
archived.

Each Mojo port passes a 32-generated-token fixture following a 19-token chat
prompt: 32/32 greedy IDs and all vocabulary logits. Relative L2 errors against
MLX FP32-activation references are 0.00053193 (4B) and 0.00049299 (9B).
With both `GGML_VK_DISABLE_ASYNC=1` and `GGML_VK_DISABLE_FUSION=1`, Prism Vulkan
agrees with the same reference over those 32 positions: relative L2 0.00009974
and 0.00009202 respectively, 32/32 argmax IDs each. This is fixture coverage,
not a quality or long-context evaluation. The flags are necessary qualifications
from the earlier report; default Vulkan was not requalified here.

The 4B model reuses the 512-element Hadamard transform and packed projections,
with generalized attention and recurrent dimensions. Both larger models have
32 recurrent value heads mapped to 16 key heads. The 9B model has an untied
output head and uses a 4096-element Hadamard transform; it adds a cross-block
pass to the existing 512-element transform. At capacity 512 the reported Mojo
device allocations were 1,169.55 MiB and 2,382.42 MiB respectively. Both
loaded and ran on the phone.

The same unmodified Prism `llama.cpp` commit as the earlier report,
`c96baf4abe24d11a50f326e40b6223b687449d63`, is used. Pinned model revisions,
file and binary hashes are in [provenance](inference-testing-bonsai-large/provenance.json).
