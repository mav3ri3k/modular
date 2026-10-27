# Experimental Android LLM backend

**Bonsai 2 2B and SmolLM2-135M-Instruct generate text on a Galaxy S26 using
the native Mojo OpenCL target and experimental MAX SDK HAL plugin.** Both
model forward passes and their GPU kernels are written in Mojo. The host forward pass is
compiled to Android ARM64 machine code with embedded ABI-v3 GPU artifacts.
The runner uses ordinary Mojo HAL `DeviceContext` launches, typed owning
`DeviceBuffer`s, and `MODULAR_DRIVER_PLUGINS` discovery. C++ loads checkpoints
and performs greedy sampling; it does not marshal kernels or own HAL handles.
Normal MAX graph execution now passes transformer-operator tests and a full
real-weight 30-layer SmolLM2 prefill/decode, including actual phone-produced KV
caches. Graph tensors use storage engines that preserve opaque views on the
host. This is tested operator/model coverage, not support for every MAX kernel.
The examples below retain explicit Mojo forward orchestration. A separate
[Bonsai MAX graph](bonsai_graph/README.md) runs all 24 layers through ordinary
MAX ops and registered custom kernels, with a paired phone comparison.
The Android process performs all model inference. Python on the development
machine handles tokenization and display; no Python runtime runs on the phone.

For the larger model, see the [Bonsai 2 2B build, export and run instructions](bonsai/README.md).
It passes phone/reference checks with vectorized packed projections; see its
[performance measurements](bonsai/PERFORMANCE.md). The remaining instructions
here cover SmolLM2.

## SmolLM2 model

The pinned [SmolLM2 checkpoint](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/tree/12fd25f77366fa6b3b4b768ec3050bf629380bac)
has 30 Llama blocks, hidden size 576, intermediate size 1536, nine query heads,
three KV heads, head dimension 64, and vocabulary size 49,152. The exporter
checks those dimensions. It preserves checkpoint BF16 weights and transposes
linear matrices for adjacent GPU threads to read adjacent output weights.
Activations, normalization, softmax and KV caches use FP32. Kernels implement
embedding, RMSNorm, matrix-vector products, half-split RoPE, causal grouped-query
attention, residual addition and SwiGLU. Cached decoding includes every layer.

The output head is tied to the embedding in the source checkpoint. This first
implementation stores a transposed second copy for efficient linear access.
That costs about 54 MiB, included in the allocation figure below.

## Build

Use a compiler built from **this checkout**, including the new OpenCL target;
a released compiler will not produce native `.mcl` objects. See
[compiler build instructions](../../../Mojo/tools/mojo-opencl/README.md),
including the macOS SDK override if required. Build the compiler with `./bazelw`.
The following commands run from the repository root. Keep build/model files
outside the repository.

Install Python dependencies into a Python 3.11+ virtual environment:

```bash
python -m pip install -r max/examples/android-llm/requirements.txt
```

Supply an installed Android NDK. Bazel fetches the pinned OpenCL headers and
builds the normal runtime and SDK plugins. The runtime requires Android API 33
or newer:

```bash
python max/examples/android-llm/build.py \
  --mojo bazel-bin/Mojo/tools/mojo/mojo \
  --ndk /path/to/android/ndk/29.0.14206865 \
  --output /tmp/modular-smollm2-build
```

`build.py` stages the HAL MAX overlay with Bazel and precompiles matching std
and MAX packages. It emits an Android host object with embedded GPU kernels,
cross-builds `//Mojo:CompilerRT` and the CPU/OpenCL SDK plugins with
`--config=android-arm64`, and links the runner with NDK libc++. Deployment uses
the complete library list in `android_artifacts.json`, including the runtime
globals and shared libc++.
No separate `.mcl` file or example-specific launch bridge is needed. Select
`--target-accelerator opencl-generic` for portable kernels; Bonsai's subgroup
kernel requires the default `adreno-840` target.

## Export and validate

The exporter downloads the pinned checkpoint (~270 MB) unless `--checkpoint`
points to a local Hugging Face snapshot. The exported weights are about 311 MiB.
It also computes FP32 PyTorch reference logits on the development machine:

```bash
python max/examples/android-llm/prepare.py \
  --output /tmp/modular-smollm2-model \
  --prompt 'What is the capital of France?' --tokens 16 --capacity 128

python max/examples/android-llm/run.py \
  --build /tmp/modular-smollm2-build \
  --model /tmp/modular-smollm2-model \
  --tokenizer /path/to/downloaded/SmolLM2-135M-Instruct \
  --serial YOUR_DEVICE_SERIAL --validate
```

`--tokenizer` is the checkpoint directory containing `tokenizer.json` and its
configuration. `--adb /path/to/adb` selects an ADB binary. Each invocation uses a
unique device directory and removes its files afterward. Model weights are
uploaded again for each invocation, outside measured inference time.

Validation compares **every vocabulary logit at each decode step**, checks for
finite results, requires identical greedy token IDs, and enforces maximum
absolute logit error < 0.05 and relative L2 error < 0.005. Validation deliberately
continues after EOS to exercise more cached decoding steps. Ordinary generation
stops on EOS. Prompts plus requested output must fit the exported capacity;
cache capacity is configurable at export time, with a maximum of 8192.

## Enter a different prompt

After export, generation does not load PyTorch or a CPU reference model:

```bash
python max/examples/android-llm/run.py \
  --build /tmp/modular-smollm2-build \
  --model /tmp/modular-smollm2-model \
  --tokenizer /path/to/downloaded/SmolLM2-135M-Instruct \
  --serial YOUR_DEVICE_SERIAL \
  --prompt 'Complete this sentence: The opposite of hot is' --tokens 16
```

The native runner prints token IDs, prefill speed and decode speed. The host
script decodes the result. There is no Android app yet.

## Device results (October 23, 2026)

Galaxy S26 SM-S942U1, Snapdragon SM8850, Adreno 840, Android API 37:

- France prompt: 37 input tokens, 16 checked output tokens; all greedy IDs match
  the CPU reference. Maximum absolute logit error **0.00009412**; relative L2
  error **0.00000270**. Output begins **“The capital of France is Paris.”**
- A third reference run used **111 input tokens** and **16 output tokens**
  (within one token of the 128-token cache capacity); all IDs matched, with
  maximum absolute logit error **0.00008774**.
- First short run: **21.69 prefill tok/s**, **20.91 decode tok/s**.
- Independent generation prompt: **“The opposite of hot is cold.”**, stopping
  at EOS; **18.20 decode tok/s** in that run.
- Device buffer allocations **316.44 MiB** with context capacity 128. This
  excludes OpenCL driver/JIT/process overhead and is not Android process RSS.

These are short correctness runs with scalar kernels, sequential prompt
prefill, and no warmup/thermal control. Decode includes selection readback and,
when enabled, full-logit CPU-side validation. Initialization, uploads and
OpenCL program compilation are excluded. The figures establish feasibility,
not sustained or optimized throughput.

## Bonsai 2 2B

The [Bonsai port](bonsai/README.md) now implements packed 2-bit projections,
signed Hadamard transforms, recurrent gated DeltaNet and full attention. Its
forward pass and kernels are also Mojo, compiled through the same Android host
and OpenCL device targets. It reuses the native HAL transport and prompt runner.

An Android prompt/output/tok-per-second app remains a later integration step.
