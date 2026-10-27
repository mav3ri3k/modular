# Android backend for Modular

The objective is to add an Android backend to Modular, including a path toward
Mojo/MAX model execution. Existing Android inference engines are references,
not substitutes for this work.

The initial assessment used repository `f85b9f5` and Mojo
`1.3.0.dev2026102302` (`169a6d8c`). Development is on branch
`android-opencl-backend`.

## Implemented and tested

See the [OpenCL backend](../../Target/lib/OpenCL/README.md), its
[implementation](../../Target/lib/OpenCL/OpenCLPlugin.cpp), and the
[on-device test](../../Target/test/OpenCLPluginSmoke.cpp).

The connected S26 reports SM-S942U1, Qualcomm SM8850, Android API 37 and
Adreno 840. The experimental backend implements the existing MAX SDK vendor HAL
for device discovery, buffers, queues, source bundles, dispatch and events.
The Android build passed FP32 GPU matrix multiplication, cross-queue ordering,
resource lifecycle and negative tests on this phone. It uses the device's
`/vendor/lib64/libOpenCL.so`.

Separately, `matmul.mojo` was cross-compiled for `aarch64-linux-android31`, linked
as an ARM64 shared library and called successfully through the C ABI **on the
phone** using `probe.c`. At `-O0`, this restricted kernel links without unresolved
symbols or runtime dependencies. Its optimized object depends on libc `memset`.
This does not establish compatibility for the general Mojo standard library.

## Mojo compiler milestone

The source compiler now registers `aarch64-unknown-opencl` and emits `.mcl`
kernel bundles through `--emit object`. The new stdlib plugin supports ordinary
GPU indexing such as `max.gpu.global_idx`. The backend lowers a checked scalar and fixed-SIMD
LLVM subset to OpenCL C internally and packages an argument ABI manifest;
kernel authors supply Mojo source.

The native compiler artifact passed matrix multiplication, cyclic loop-carried
SSA values, wrapping UInt16 multiplication and a tiny two-layer dense/ReLU
model on the S26 GPU. The model uses synthetic weights. Both layers and the
intermediate activations stay on the GPU. CPU object emission still passed.
See the [compiler target documentation](../../Mojo/tools/mojo-opencl/README.md)
for reproduction, the supported subset and exact ABI.

## Pretrained model milestone

[Bonsai 2 2B and SmolLM2-135M now generate text on the S26](../../max/examples/android-llm/README.md).
Both model forward passes and GPU kernels are Mojo. The Bonsai text port keeps
2-bit weights packed and implements signed Hadamard transforms, 18 recurrent
gated DeltaNet layers and six full-attention layers. Three prompt/reference
checks passed. The first optimization pass improves short native decode
benchmarks from roughly 15 to 29 tok/s, using roughly 592 MiB of device buffers. The SmolLM2 baseline measured about 21 decode tok/s.

[Reproduction and numerical details](../../max/examples/android-llm/bonsai/README.md)
distinguish the FP32 activation reference from stock FP16 execution. The host
Mojo forward pass is native Android ARM64; C++ provides only I/O, HAL transport
and sampling. Tokenization remains on the development host.

## What remains

Ordinary Mojo HAL `DeviceContext` launches now work on Android, including
nested buffer views, SIMD/FP16, static/dynamic shared memory, barriers, integer
atomics and checked Adreno subgroup shuffles. See the
[integration ledger](INTEGRATION.md) for implementation state and verification.

MAX graph execution, aggregate tensor argument transport, graph memory planning,
the full Android runtime and app/JNI packaging remain. The existing model
forward passes still dispatch through their preserved C ABI bridge. The legacy
C++ plugin loader requires additional bridge objects if that alternative path
is used; ordinary Mojo HAL launches load the SDK plugin directly.

The [llama.cpp Snapdragon guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/README.md)
provides relevant architecture references for separate Adreno/OpenCL and
Hexagon/FastRPC backends. Hexagon is not reached through the OpenCL plugin.

## Reproduce the Mojo CPU probe

From the repository root, with `uv` installed:

```sh
uv venv /tmp/modular-android-probe
uv pip install --python /tmp/modular-android-probe/bin/python \
  --index https://whl.modular.com/nightly/simple/ mojo==1.3.0.dev2026102302
PROBE_ENV=/tmp/modular-android-probe
PROBE_SRC="$PWD/docs/android-feasibility"

"$PROBE_ENV/bin/mojo" build "$PROBE_SRC/matmul.mojo" --emit object \
  --target-triple aarch64-linux-android31 --target-cpu generic -O0 \
  -o "$PROBE_ENV/matmul.o"

PROBE_LLD=$("$PROBE_ENV/bin/python" -c \
  'import importlib.util; from pathlib import Path; print(Path(next(iter(importlib.util.find_spec("modular").submodule_search_locations))) / "bin/lld")')
"$PROBE_LLD" -flavor gnu -shared --no-undefined -z max-page-size=16384 \
  -soname libmojo_android_probe.so "$PROBE_ENV/matmul.o" \
  -o "$PROBE_ENV/libmojo_android_probe.so"
```

Build `probe.c` with the NDK's `aarch64-linux-android31-clang`, linking the
resulting `.so`. Push both into a dedicated directory under `/data/local/tmp`
and run the executable with `LD_LIBRARY_PATH` set to that directory.
The verified output is `PASS: Mojo matrix multiplication called through the C ABI`.
This CPU test is a platform smoke check; the OpenCL HAL tests establish GPU work.
