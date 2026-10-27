# Android inference comparison — October 26, 2026

This report accompanies the [reproducible harness](../../max/examples/android-llm/inference_test/README.md)
and [raw records](inference-testing-2026-10-26/). Measurements below are from the
connected Galaxy S26 SM-S942U1, Snapdragon SM8850 / Adreno 840, Android API 37.

## Final measured results

**Bonsai retains a decode advantage; that advantage does not carry over to
these baseline Q4_K_M ports.** All rows below use completed, numerically checked
reference paths. The native reference is the strongest valid configuration
observed for that model, not the fastest unchecked timing.

### Decode: `tg128`, tokens/second

Mean ± sample standard deviation, three samples. Full-vocabulary host transfer
is included; tokenization and host sampling are excluded.

| Model | Mojo/OpenCL | Prism llama-bench | Native configuration | Mojo/native rate |
| --- | ---: | ---: | --- | ---: |
| Bonsai 2 2B | 69.74 ± 5.91 | 51.74 ± 0.65 | Vulkan, PTQ1_0, workaround flags | 1.35x |
| Qwen3-1.7B Q4_K_M | 8.56 ± 0.47 | 46.47 ± 1.58 | Vulkan, workaround flags | 0.18x |
| Qwen3.5-2B Q4_K_M | 8.54 ± 0.53 | 28.15 ± 3.52 | OpenCL | 0.30x |

The Vulkan rows were measured with **both** `GGML_VK_DISABLE_ASYNC=1` and
`GGML_VK_DISABLE_FUSION=1`. These are existing backend options; reference
inference source is unmodified. Stock Vulkan produced incorrect logits on
this phone and its unqualified timings are excluded. The workaround restored
Bonsai and Qwen3 fixture agreement, but did not restore Qwen3.5 agreement;
Qwen3.5 therefore uses the valid OpenCL reference. See the failure record below.

Bonsai's Mojo median in the reported set is 66.70 tok/s; the 69.74 mean includes
a 76.54 sample. No fast or slow sample is removed. The final qualified native
Vulkan runs use three repetitions in one process; the corresponding Mojo
observations came from separate processes in the paired phase. These are
measured workload rates with thermal variation, not isothermal speedups or
hardware limits. FP32 versus FP16 caches and arithmetic also differ as detailed
below. The report supersedes earlier provisional Vulkan comparisons.

### Prefill: `pp512`, tokens/second

The Mojo implementation is serial. The Bonsai Mojo value below is from a fresh
repository rebuild and three-process S26 rerun. The Prism values are retained
from the earlier reference run and were not remeasured alongside this rebuild;
do not treat this row as a paired comparison. OpenCL reference processes
completed; accuracy coverage here is the token-wise fixtures, not a separate
all-logit validation of batched 512-token execution.

| Model | Mojo, batch 1 | Prism OpenCL, batch 1 | Prism OpenCL, batch 2048 / microbatch 512 |
| --- | ---: | ---: | ---: |
| Bonsai 2 2B | 60.85 ± 1.60 | 11.17 ± 0.48 | 33.26 ± 0.89 |
| Qwen3-1.7B Q4_K_M | 5.26 ± 0.09 | 20.33 ± 1.68 | 514.42 ± 0.37 |
| Qwen3.5-2B Q4_K_M | 5.29 ± 0.02 | 18.15 ± 1.87 | 410.36 ± 0.54 |

Bonsai's OpenCL reference projections fall back to CPU. Vulkan prefill timings
from the default, numerically incorrect configuration are diagnostic only;
Qwen3 also segfaulted in that configuration. Correctness-qualified Vulkan
prefill was not re-benchmarked. Optional combined/deep-context workloads were
not completed; no results for them are claimed.

The fresh Bonsai `pp512` samples were 61.99, 61.54 and 59.03 tok/s. Thermal
status remained 0, with the skin sensor rising from 33.0°C to 37.5°C. The
fresh run used `runtime/model.o` SHA-256
`92c2ee5dcbed1034bcafeb994a642fa52e87b1e149ce1a3c9fa5bd8352933e13`; this
differs from the older app object's hash because the historical demo used a
separate `mojo-chat` checkout. Raw outputs, validation, thermal snapshots and
all fresh input and binary hashes are in the [fresh rebuild archive](inference-testing-s26-rebuild-2026-10-26/README.md).

## Scope and implementation

The original Bonsai comparison uses the exact
`mojo-chat/.native/engine/model.o` and staged runtime from the final demo on
`android-opencl-backend`: direct Mojo/OpenCL, original eight-output
projections, compact scales and fused GPU greedy candidates. The fresh
`pp512` rerun uses `runtime/model.o` built from this repository and is reported
separately above because its object hash differs. Neither run uses a tiled
integration, MAX graph path, NPU backend or app replacement. The app
configuration is preserved for the historical demo comparison.

Both Qwen models are new text-only Mojo/DeviceContext ports. They retain the
**actual Q4_K_M GGUF tensors**, including their mixed quantization types, in
losslessly transposed byte planes. They do not use BF16 substitutes or
re-quantized weights. Qwen3.5 excludes the extra MTP block and vision encoder,
as default text-only llama-bench does. They use the same native Android runtime
and Mojo OpenCL target as the Bonsai demo, rather than a Python MAX graph.

The bounded reuse pass applies the existing eight-output matvec pattern and
Bonsai's two-pass RMSNorm. Qwen3.5 also reuses Bonsai's full-attention and
recurrent DeltaNet operators. There was no per-model autotuning. These are
experimental baseline ports, not fully optimized K-quant kernels or general
MAX serving architecture registrations.

## Sources and reproducibility

Reference: [PrismML-Eng/llama.cpp, branch prism](https://github.com/PrismML-Eng/llama.cpp/tree/prism),
commit `c96baf4abe24d11a50f326e40b6223b687449d63`. The checkout was clean and
matched `origin/prism`. Both OpenCL and Vulkan are built from that unmodified
commit with the Android NDK 29.0.14206865, Release, ARM64, ARMv8.7-A CPU code,
OpenMP disabled. The OpenCL build enables the Adreno kernels and embeds their
sources. The Vulkan build uses the NDK host glslc, separately installed
SPIRV-Headers and Vulkan C/C++ headers; no inference source patch was needed.
Exact CMake configurations and dependency revisions are archived.

| Model | Publisher file and pinned revision |
| --- | --- |
| Bonsai 2 2B | [prism-ml/Ternary-Bonsai-2-2B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-2B-gguf/tree/683f7be2629342a50987627b40881e54761a2430), PQ2_0 and PTQ1_0 |
| Qwen3-1.7B | [bartowski/Qwen_Qwen3-1.7B-GGUF](https://huggingface.co/bartowski/Qwen_Qwen3-1.7B-GGUF/tree/dcb19155b962dbb6389f4691a982043a8e651022), Q4_K_M |
| Qwen3.5-2B | [bartowski/Qwen_Qwen3.5-2B-GGUF](https://huggingface.co/bartowski/Qwen_Qwen3.5-2B-GGUF/tree/7d26695454df6de5fbcce2e58681e62dae06ce43), Q4_K_M |

[Provenance](inference-testing-2026-10-26/provenance.json) records source and
binary hashes. [Generated sources](inference-testing-2026-10-26/generated/) and
the [existing source snapshot](inference-testing-2026-10-26/source-snapshot/)
preserve the tested code independently of the working branch's uncommitted
optimization work. The test directory contains no model weights or binary build artifacts.
Local build/model artifacts are under `/tmp/modular-inference-testing`.

## What the benchmark includes

The primary workloads are llama-bench's standard `pp512` and `tg128`.
`tg128` performs 128 forwards at positions 0–127 on an empty sequence, with a
synchronization after every token. It does not sample generated text or stop
at EOS. `pp512` processes 512 random input tokens. The port matches the Android
C library's random-token generator and default seed. Each cache reset and its
completion occur outside the measured interval. Warmup is the full prompt for
pp and one token for tg, as in the reference source.

**Full-vocabulary host transfer is included.** Although llama-bench does not
call a sampler, `llama_decode` itself enqueues the logits transfer in
`src/llama-context.cpp`. The Mojo harness therefore enqueues the same-size
transfer into persistent host storage after every forward. Sampling,
tokenization, weight loading, initialization and GPU compilation are excluded.
The exact Bonsai object still computes its fused greedy candidates; their
extra host transfer and selection are excluded, but the fused GPU work remains.

Plugin provenance was also checked against Gradle's `jniLibs` source set. The
packaged OpenCL plugin is `engine/plugins/arm64-v8a/libOpenCLPlugin.so`, SHA-256
`5a67266344ef0096f07387958f0013a21389f4d1a63bdf225fc9893505447d6c`.
An obsolete `engine/lib` copy has a different hash. Early Bonsai measurements
using that obsolete copy are retained as runtime diagnostics and excluded from
the final demo numbers. Qwen used the packaged-plugin hash throughout. Final
Bonsai Vulkan-paired decode, prefill and validation use the packaged plugin.

The first diagnostic harness omitted full-logit transfer. Its incomplete
[forward-only records](inference-testing-2026-10-26/forward-only-diagnostic/)
are retained and **excluded from the matched results**. They must not be used
as a llama-bench comparison or substituted for the corrected measurements.

Mojo currently implements serial prompt processing, so the equivalent prompt
comparison uses `llama-bench -b 1 -ub 1`. The reference's ordinary batched
`pp512` is also measured separately with batch 2048 / microbatch 512. A serial
Mojo prefill is not an implementation of batched GEMM prefill.

Decode comparisons reverse backend order across three paired runs. Prefill uses
three native repetitions per process after one shared warmup. Raw integer
nanosecond durations, mean of individual tokens/second rates, sample standard
deviation and median are retained. Runs are not filtered by throughput. Fixed
pauses and thermal snapshots accompany each comparison; clocks, governors,
charging settings and thermal controls are unchanged. Throttling occurred,
so the variation is part of these observations, not a controlled isothermal
or hardware-ceiling claim.

Precision is not identical: the unmodified reference CLI uses FP16 KV caches
and backend-specific arithmetic; the Mojo ports use FP32 KV/recurrent state.
Bonsai uses its existing FP16 dot chains and FP32 reductions. The Qwen ports
use FP32 accumulation. Allocated capacity is 2048 for the exact Bonsai demo and
1024 for Qwen; llama-bench derives capacity from the workload. Populated token
counts match, but allocation footprints and cache strides differ.

## Reference backend differences

The OpenCL `supports_op` implementation admits Q4_K/Q5_K/Q6_K/Q8_0 matmuls,
but not PQ2_0 or PTQ1_0. Bonsai's packed projections therefore use CPU fallback
in that reference build even when `-ngl 99` requests GPU offload. Its OpenCL
number is a valid available execution path, not an equivalent GPU ternary
kernel comparison. The Qwen K-quant kernels do have native Adreno support.

The Vulkan backend includes PQ2_0/PTQ1_0 handling and a recognized Hadamard
fast path. It is tested separately so the Bonsai comparison does not depend on
the OpenCL limitation. The more compact PTQ1_0 representation is also checked
against the exact demo values before its performance is considered.

## Correctness checks

The Bonsai exporter and both publisher encodings pass **773 parameter checks**
each: packed numerical codes, scales, biases, Hadamard signs, normalization,
convolution and gate parameters. Codes/scales/signs compare exactly; only
A_log versus its negative exponential uses a tolerance for the representation
conversion. PTQ1_0 is a lossless storage alternative for these ternary values.

Independent host FP32 transformers models load dequantized values from the
**same tested GGUF files**. The phone still executes packed kernels. Initial
16-step, all-vocabulary teacher-forced checks of the finalized Qwen ports:

| Model | Maximum absolute logit error | Relative L2 | Argmax agreement |
| --- | ---: | ---: | ---: |
| Qwen3-1.7B Q4_K_M | 0.000230789 | 0.00000308612 | 16 / 16 |
| Qwen3.5-2B Q4_K_M | 0.0000151396 | 0.000000762432 | 16 / 16 |

Comparisons with llama.cpp CPU also remain archived. Its quantized activation
arithmetic differs enough to change some near ties (about 4.5% relative logit
L2 in these fixtures), so that path is not treated as an FP32 mathematical
oracle. The OpenCL reference is much closer. Passing an FP32 reference does
not establish bit-identical outputs across native backend arithmetic.

## Code comparison and interpretation

Both Bonsai implementations execute the same 24-block hybrid transformer:
three recurrent DeltaNet blocks followed by one full-attention block. The
packed embedding/head, signs and learned parameters match. Prism builds a
GGML graph and dispatches backend-specific kernels, including Vulkan's
Hadamard recognition and packed-format handling. The demo uses direct Mojo
DeviceContext launches and dedicated ternary kernels: eight output columns per
work item, four independent FP16 accumulation chains, FP32 group reductions,
subgroup/register Hadamard transforms and fused vocabulary greedy candidates.
These algorithm, precision and backend differences preclude attributing any
observed gap simply to C++ versus Mojo or to a compiler advantage.

Qwen3.5 shares the hybrid architecture and dimensions, making its recurrent
operators reusable. Its K-quant projections need a different decoder and do
not become ternary matvecs by reusing the forward orchestration. Qwen3 is a
28-layer dense-attention model with Q/K normalization and full-head RoPE. Its
new port similarly uses the generic K-quant matvec rather than Bonsai's
specialized FP16 ternary dot product. The reference has existing specialized
Adreno K-quant kernels and batched prompt processing. This study measures
these baseline ports; it does not claim that they establish a K-quant
performance ceiling. No hardware-counter attribution is made here.

## Extended validation and generation

Both finalized Qwen ports also pass 128 teacher-forced steps containing a real
chat prompt followed by natural-language text. Every vocabulary logit is
checked at every position, including cached attention/recurrent updates:

| Model | Steps | Maximum absolute error | Relative L2 | Argmax agreement |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-1.7B Q4_K_M | 128 | 0.000310898 | 0.00000299356 | 128 / 128 |
| Qwen3.5-2B Q4_K_M | 128 | 0.0000256598 | 0.00000107125 | 128 / 128 |

The exact packaged Bonsai model/runtime passes the existing 96-output-token
all-vocabulary fixture: every generated ID matches, maximum absolute error
0.0222683 and relative L2 0.000634382. Its fused-candidate selection also agrees
with full-logit greedy selection. This is fixture coverage, not a perplexity
or broad quality evaluation.

Independent greedy-generation smokes use the prompt “Explain why the sky is
blue in three short sentences.” Qwen3 completes a coherent three-sentence
answer. Qwen3.5 produces a coherent explanation before the 64-token test cap.
[Decoded outputs and token/timing logs](inference-testing-2026-10-26/) are
archived. These generation rates include selection and use a different forward
count convention; they are not substituted for llama-bench measurements.

## Vulkan correctness gate and failures

The final native-output audit rejected stock Vulkan, including the normal
FP16-cache, context-128 configuration used for decode benchmarking. These are
large discrepancies, not near-tie rounding differences:

| Default Vulkan configuration | Relative logit L2 against the captured reference | Argmax agreement |
| --- | ---: | ---: |
| Bonsai PQ2_0 | 0.5750 | 0 / 32 |
| Bonsai PTQ1_0 | 0.5737 | 0 / 32 |
| Qwen3 Q4_K_M | 1.0854 | 0 / 16 |
| Qwen3.5 Q4_K_M | 1.3984 | 0 / 16 |

The stock `llama-completion` independently produced `Count@@@@@@@@@@@` for
Qwen3's “The capital of France is” prompt on Vulkan, while its OpenCL run
produced coherent text. Thus the failure is not inferred from timing alone or
solely from the custom logit dumper. Its exact root cause is not established;
no specific inference-source defect or compiler defect is claimed.

The existing async/fusion disable flags restore agreement for Qwen3
(relative L2 0.0002481, 16/16 argmax IDs) and Bonsai (PQ2_0 L2 0.0006043,
PTQ1_0 L2 0.0006006, 32/32 IDs each). Both flags were tested together, so this
does not isolate which optimization is responsible. These qualified paths
were measured again with real llama-bench, producing the final decode table.
Qwen3.5 still has L2 1.3690 and only 1/16 matching IDs with those flags, so its
Vulkan timings remain excluded. Disabling FP16 alone produced nonfinite values
in the Qwen3 diagnostic and is not an accepted workaround.

Default Vulkan Qwen3 batched `pp512` also terminated with a segmentation fault
and emitted no benchmark JSON. No speed is assigned to that failed run. All
failure logs and unsuccessful diagnostic timings remain archived, explicitly
separate from qualified performance results.

## Additional native controls

Bonsai PQ2_0 with `-ngl 0 -t 8` measured 36.40 ± 0.20 tok/s; four threads
measured 35.11 ± 7.79. The more compact PTQ1_0 CPU path was slower: 20.00 ±
0.80 with eight threads. PTQ1_0 OpenCL/offload-requested measurements were
11.30 ± 0.16 with eight threads and 8.17 ± 0.29 with four. These controls show
why neither one storage format nor a GPU-labelled backend can be assumed to
be fastest. The final valid Vulkan PTQ1_0 result, 51.74 ± 0.65 tok/s, exceeds
these CPU/fallback controls.

## Reproduction and artifact locations

All essential source, validation code, export/build/benchmark scripts, pinned
model revisions, source snapshots, raw accepted/rejected results, and hashes
are in this repository. The executable harness is
[`max/examples/android-llm/inference_test`](../../max/examples/android-llm/inference_test/README.md).
The exact qualified native commands are recorded in
[qualified-reference-commands.json](inference-testing-2026-10-26/qualified-reference-commands.json).

`/tmp/modular-inference-testing` contains downloaded GGUFs, exported weights,
compiled executables/libraries and temporary diagnostic orchestration. Those
large/generated files are not required to be committed: their sources, hashes,
revisions and build recipes are retained in the repo. Bonsai additionally uses
the exact staged `mojo-chat` object for this measurement; the corresponding
Mojo source and object hash are preserved, and the normal Bonsai build/export
instructions remain in the repository. Rebuild compiler/runtime dependencies
using the parent Android example instructions before using the Qwen builders.

Validation performed: native C++ builds with `-Wall -Wextra -Werror`, both Mojo
ports compiled for Adreno 840, complete-vocabulary 16/128-step Qwen checks,
96-step Bonsai check, both 773-check Bonsai format comparisons, real generation
smokes, native backend accuracy audits, exporter byte-for-byte reproduction,
and Ruff checks. No app replacement, firmware/driver change, benchmark-source
patch, commit or publication was performed.

The result supports the existing Bonsai implementation on this phone. It does
not establish a general Mojo/MAX advantage for Q4_K_M: the minimally ported
Qwen implementations are slower, and their generic quantized matvec/attention
paths need separate optimization to approach the native reference. Broad model
quality, long-context accuracy and sustained isothermal performance were not
established by these fixtures.
