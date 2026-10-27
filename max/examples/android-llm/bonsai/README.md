# Bonsai 2 2B on Android through Mojo

**The text-only Bonsai 2 2B model runs on the Galaxy S26's Adreno 840 through the
new Mojo OpenCL compiler target and experimental MAX SDK HAL backend.** There
is no llama.cpp, LiteRT, MLX or Python inference on the phone. The forward pass
and all mathematical GPU kernels are Mojo. A shared native runner handles
checkpoint I/O and greedy token selection. Mojo owns typed device buffers and
launches kernels through ordinary `DeviceContext`.

## What is ported

The checkpoint is
[`prism-ml/Ternary-Bonsai-2-2B-mlx-2bit`](https://huggingface.co/prism-ml/Ternary-Bonsai-2-2B-mlx-2bit/tree/1c6cbae21f8c38830ce337972b46fdae7b18eaaa),
revision `1c6cbae21f8c38830ce337972b46fdae7b18eaaa`. The port implements its
24 text blocks (hidden size 2048, MLP size 6144, vocabulary 248,320):

- Packed affine 2-bit projections, group size 128. Codes stay packed; group
  scales and biases are expanded to FP32. Packing is transposed for adjacent
  GPU outputs to read adjacent weight words.
- Nine-stage signed, normalized, block-512 Walsh-Hadamard transforms. Linear
  inputs use the forward transform; embeddings use the inverse sign order.
  The tied output head reuses the packed embedding matrix.
- Eighteen gated DeltaNet blocks: four-tap causal depthwise convolution, Q/K
  L2 normalization, learned decay and update gates, recurrent state updates,
  gated normalization and output projections.
- Six full-attention blocks: gated query projections, Q/K RMS normalization,
  partial rotary positions, causal grouped-query attention and KV caches.
- RMS normalization, residuals, SwiGLU, final normalization and output logits.

Packed projections now use **FP16 activations and standard SIMD multiply-adds**;
Hadamard math, group reductions, residuals and recurrent state remain FP32.
The fused block-512 Hadamard kernel targets the Adreno 840's 64-lane subgroups.
The FP16 vision tower is omitted. Image/video input, batching and stochastic
sampling remain outside this prototype. A separate
[MAX graph implementation](../bonsai_graph/README.md) now runs the full text
model using ordinary ops and registered custom kernels. The separate
Mojo Chat demo packages this same backend. See [performance](PERFORMANCE.md)
for measured throughput and approximation error. The model uses the scoped
standard `FastMathFlag.AFN` contract; a global fast-math switch is not required.

## Build

Start with the [compiler and NDK setup](../README.md#build). Build the compiler
from this checkout with `./bazelw`; a released compiler lacks the native target.
From the repository root:

```bash
python max/examples/android-llm/build.py --variant bonsai \
  --mojo bazel-bin/Mojo/tools/mojo/mojo \
  --ndk /path/to/android/ndk/29.0.14206865 \
  --output /tmp/modular-bonsai2-build
```

The Android ARM64 host object embeds ABI-v3 GPU artifacts. Standard Mojo
`DeviceContext` loads them through the SDK plugin discovered with
`MODULAR_DRIVER_PLUGINS`. The Adreno driver compiles the generated OpenCL C at
load time. The runner links the normal `//Mojo:CompilerRT`, cross-built with
Bazel for Android API 33+. `android_artifacts.json` lists all deployed runtime
and SDK plugin libraries. There is no example-specific kernel launch bridge or
separate `.mcl` file.

## Export the checkpoint

The exporter and numerical reference use MLX on the **development Mac**. Install
these dependencies in a separate Python 3.11+ virtual environment:

```bash
python -m pip install -r max/examples/android-llm/bonsai/requirements.txt
hf download prism-ml/Ternary-Bonsai-2-2B-mlx-2bit \
  --revision 1c6cbae21f8c38830ce337972b46fdae7b18eaaa \
  --local-dir /tmp/modular-bonsai2-hf

python max/examples/android-llm/bonsai/prepare.py \
  --checkpoint /tmp/modular-bonsai2-hf \
  --output /tmp/modular-bonsai2-model \
  --prompt 'What is the capital of France?' --tokens 8 --capacity 128
```

The download includes vision weights (~1.20 GB overall); only text weights are
exported. The exported `weights.bin` is **568 MiB** and contains packed codes,
FP32 affine parameters, sign vectors, unquantized auxiliary weights and a rotary
position table. It also records tensor shapes and a source checkpoint hash in
`manifest.json`. Keep these generated files outside git.

The exporter uses mlx-lm's published Hadamard-aware model and retains the
checkpoint's quantized values. It promotes floating parameters/activations to
FP32 and replaces the embedding's hardcoded FP16 output conversion with FP32.
The recorded logits serve as the **FP32 reference** for the Android port,
including the newer approximate FP16 kernels. They do not establish bitwise
identity with MLX's stock FP16 activation execution. For the France prompt,
the stock runtime independently produced the same eight greedy token IDs.

## Validate on the phone

```bash
python max/examples/android-llm/run.py \
  --build /tmp/modular-bonsai2-build \
  --model /tmp/modular-bonsai2-model \
  --tokenizer /tmp/modular-bonsai2-hf \
  --serial YOUR_DEVICE_SERIAL --validate
```

Use `--adb /path/to/adb` if needed. Each run uploads artifacts into a unique
temporary device directory and removes it afterward. Validation compares all
248,320 logits at every checked decode step, rejects non-finite values, requires
matching greedy IDs and checks absolute/relative logit tolerances. This also
exercises persistent convolution, recurrent and KV caches. Reference mode
continues for the requested number of tokens even after EOS.

## Enter an independent prompt

No MLX model or CPU reference is loaded by this command. Tokenization/display
run on the development host; all model inference runs on the phone:

```bash
python max/examples/android-llm/run.py \
  --build /tmp/modular-bonsai2-build \
  --model /tmp/modular-bonsai2-model \
  --tokenizer /tmp/modular-bonsai2-hf \
  --serial YOUR_DEVICE_SERIAL \
  --prompt 'Reply with only one word: hello.' --tokens 16
```

Thinking is disabled in the chat template. Generation is greedy and stops at
either `<|im_end|>` (248046) or `<|endoftext|>` (248044). Prompt plus requested output must fit the exported context capacity
(default 128; configurable up to 8192). The underlying model's longer advertised
context is not a claim about this prototype.

## Optimized engine

Packed layer and vocabulary projections compute eight adjacent outputs per
thread using standard Mojo SIMD, retaining four FP16 accumulation chains and
FP32 group scaling. Layers use 64-thread workgroups and the head uses 128.
The vocabulary head also emits one winning value, index and nonfinite flag per
64-lane subgroup. Generation reads these 485 candidates (about 5.8 KB) and
finishes greedy selection on the CPU; validation retains the full-logit path.
The selected source build is validated against independent FP32 logits over
96-token generations. See [measurements and limitations](PERFORMANCE.md) for
paired medians, GPU timings and rejected experiments.
An opt-in [Hexagon projection backend](../../../../Target/lib/Hexagon/README.md)
routes the vocabulary head, or selected rows concurrently, through HVX or HMX.
The batch-one HVX path avoids FP16 weight expansion. Its small incremental
throughput benefit is not consistent across measurement batches, so GPU-only
remains the default. See the [experiment report](../../../../docs/android-feasibility/hexagon-decode/README.md).

Gate reductions, RMSNorm, packed dot products and duplicate Hadamard transforms
were optimized. The loader enforces centered affine groups and shared sign
vectors required by those fast paths. Rebuild the embedded model object and native runner together.

## Initial correctness baseline on the connected S26

Galaxy S26 SM-S942U1, Snapdragon SM8850, Adreno 840, Android API 37, October 23,
2026. Three independent prompt/reference checks passed:

| Prompt | Checked output tokens | Maximum absolute logit error | Relative L2 error | Decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| Capital of France | 8 | 0.00005531 | 0.00000141 | 15.75 |
| Opposite of hot | 24 | 0.00006771 | 0.00000126 | 15.43 |
| List of numbers through 20 | 24 | 0.00004760 | 0.00000109 | 12.59 |

Example outputs begin **“The capital of France is \*\*Paris\*\*.”**, **“The
opposite of hot is \*\*cold\*\*.”**, and **“The last number in your list is
\*\*20\*\*.”** All checked greedy IDs matched the FP32 reference.

An independent generation check returned **“hello”** and stopped at the chat
EOS token after two output tokens, without loading a reference model.

Device buffer allocations are **591.90 MiB** at context capacity 128, including
weights, activations and caches. This excludes OpenCL driver/JIT/process
overhead and is not process RSS. These are short correctness runs, without
thermal control or a sustained-performance study. Generation timings include
the compact token-selection readback; validation also reads and checks full
logits. Uploads, checkpoint loading and GPU program compilation are excluded.
Prompt prefill currently processes one token at a time.

## Source map

- [model.mojo](model.mojo): full forward pass and checked kernel dispatch.
- [kernels.mojo](kernels.mojo): mathematical GPU kernels, including packed
  projections, transforms, recurrent state and attention.
- [prepare.py](prepare.py): text checkpoint exporter and numerical reference.
- [../runner.cpp](../runner.cpp), [../model_spec.h](../model_spec.h): shared
  Android transport/runner, parameterized for SmolLM2 or Bonsai.

The next product step is a persistent native library/JNI interface, tokenizer
on Android, and the prompt/output/tok-per-second app. The CLI already reports
prefill and decode counters, but the tokenizer currently remains on the host.

Detailed [GPU profiling and optimization priorities](PERFORMANCE.md) explain
why this initial implementation is slower than the published MLX demo. Add
`--profile` to the runner to collect per-kernel OpenCL device timestamps.
