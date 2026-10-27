# Android inference comparison

This harness compares Prism's unmodified `llama-bench` with native Mojo/OpenCL
for Bonsai 2 2B, Qwen3-1.7B Q4_K_M and Qwen3.5-2B Q4_K_M. The Qwen ports are
experimental fixed-architecture text models, not architecture registrations for
the Python MAX serving stack. They use the same ordinary Mojo DeviceContext and
Android runtime as the Bonsai demo. No Python inference runs on the phone.

Measured results, exact revisions, hashes and limitations are recorded in
[the test report](../../../../docs/android-feasibility/INFERENCE_TESTING_2026-10-26.md).

## What is measured

`bench.cpp` ports `tools/llama-bench/llama-bench.cpp`'s `test_prompt` and
`test_gen` behavior. It runs random tokens, excludes tokenization, host sampling, loading and kernel compilation, resets model state before each
sample, enqueues a full-vocabulary device-to-host transfer after every forward
(as `llama_decode` does), and synchronizes after every decode forward. A `tg128` sample performs
**128 actual forwards**, rather than the 127 forwards used when generating 128
output IDs from a prefilled prompt. Its JSON records integer nanosecond samples,
arithmetic mean of per-sample tok/s and sample standard deviation, like
llama-bench. Random input is intentional; these tests measure execution, not
response quality. Token-by-token correctness is checked separately.

Prompt processing currently runs serially with batch size one, including the
output projection on every token. Match it with `llama-bench -b 1 -ub 1`.
The suite also measures llama.cpp's normal batched prefill separately. Do not
compare this serial port with batched prefill as an equivalent implementation.
`pg128-128` measures the total 256-token combined workload. `tg128-d512` fills
512 positions outside the timer, then times 128 decode forwards.

The warmup matches llama-bench: the entire prompt for pp and one token for tg.
Cache clearing and its completion are outside the measured interval. The suite
alternates Prism/Mojo and Mojo/Prism order, captures thermal state before/after
each sample, and preserves every sample, including slow ones. The ten-second
pause is a fixed pause, not a guarantee of equal starting temperature. Use `--samples-per-run 3 --repeats 1` for standard three-repetition native
benchmarks with one shared warmup per process.
There are no clock, governor, charging, thermal-service or GPU-driver overrides.

Numerical execution differs: the native reference CLI uses FP16 KV caches;
Mojo uses FP32 caches/state. Bonsai retains its validated FP16 packed dot chains.
The Qwen ports accumulate in FP32. The exact Bonsai object still computes fused
GPU greedy candidates inside its output kernel; fused-candidate readback and host selection
are excluded, but that fused GPU work remains timed. The full-vocabulary transfer
required by `llama_decode` is included separately. These are implementation comparisons on
identical quantized weights, not bit-identical arithmetic. Mojo cache capacity
is 2048 for the exact demo and 1024 for Qwen; llama-bench chooses context from
the workload. All attention loops still process only populated positions.

## Build the Vulkan reference

These commands reconstruct the recorded Bonsai Vulkan configuration on the measured Apple Silicon/macOS host. Start with `NDK`, `ADB`, `SERIAL`, and `WORK` from the [root prerequisites](../../../../README.md#reproduction-prerequisites), and the pinned Prism checkout from the [benchmark protocol](../../../../README.md#benchmark-protocol). All builds and downloads stay outside this repository. Use fresh directories when changing configurations.

The revisions below were checked against the original local checkouts. The NDK supplies both the Android C++ toolchain and a **host** `glslc`; do not substitute a shader compiler from another SDK when reproducing these measurements.

```sh
test "$(git -C "$WORK/prism" rev-parse HEAD)" = c96baf4abe24d11a50f326e40b6223b687449d63
git clone https://github.com/KhronosGroup/SPIRV-Headers.git "$WORK/SPIRV-Headers"
git -C "$WORK/SPIRV-Headers" checkout --detach fbf86836a9b1966586faf7be7bb8a6440e156d62
git clone https://github.com/KhronosGroup/Vulkan-Hpp.git "$WORK/Vulkan-Hpp"
git -C "$WORK/Vulkan-Hpp" checkout --detach e48d470db25ad598aff8aa80292667e66c45138e
git -C "$WORK/Vulkan-Hpp" submodule update --init Vulkan-Headers
test "$(git -C "$WORK/Vulkan-Hpp/Vulkan-Headers" rev-parse HEAD)" = 73814d006638aa00bd553215b054f0f02e457e22

cmake -S "$WORK/SPIRV-Headers" -B "$WORK/spirv-build" \
  -DSPIRV_HEADERS_ENABLE_TESTS=OFF -DSPIRV_HEADERS_ENABLE_INSTALL=ON \
  -DCMAKE_INSTALL_PREFIX="$WORK/spirv-install"
cmake --install "$WORK/spirv-build"

export NDK_HOST=darwin-x86_64
export GLSLC="$NDK/shader-tools/$NDK_HOST/glslc"
"$GLSLC" --version
cmake -S "$WORK/prism" -B "$WORK/prism-vulkan-build" \
  -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_C_FLAGS=-march=armv8.7a \
  -DCMAKE_CXX_FLAGS="-march=armv8.7a -I$WORK/spirv-install/include" \
  -DCMAKE_C_FLAGS_RELEASE= -DCMAKE_CXX_FLAGS_RELEASE= -DCMAKE_ASM_FLAGS_RELEASE= \
  -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF -DGGML_OPENCL=OFF -DGGML_VULKAN=ON \
  -DSPIRV-Headers_DIR="$WORK/spirv-install/share/cmake/SPIRV-Headers" \
  -DVulkan_INCLUDE_DIR="$WORK/Vulkan-Hpp/Vulkan-Headers/include" \
  -DVulkan_GLSLC_EXECUTABLE="$GLSLC"
cmake --build "$WORK/prism-vulkan-build" --target llama-bench -j 6
```

CMake 4.4.4 was used on the development host. Its original CMake cache explicitly has empty C/C++/ASM Release flags; those are reproduced above. Changing to an optimized reference is a useful additional experiment, but requires new measurements. On an x86-64 Linux host, the corresponding NDK directory is `linux-x86_64`; that host path has not been validated here. Host shader generation uses the host C/C++ toolchain; Android objects use NDK r29.

Stage the executable, every generated shared library, the NDK C++ runtime, and the pinned GGUF files. `REFERENCE_ROOT` is independent of the suite's default layout; set it explicitly for direct commands below.

```sh
export REFERENCE_ROOT="$DEVICE_ROOT/prism-vulkan"
"$ADB" -s "$SERIAL" shell "mkdir -p $REFERENCE_ROOT"
"$ADB" -s "$SERIAL" push "$WORK/prism-vulkan-build/bin/llama-bench" "$REFERENCE_ROOT/"
for library in "$WORK/prism-vulkan-build/bin/"*.so; do
  "$ADB" -s "$SERIAL" push "$library" "$REFERENCE_ROOT/"
done
"$ADB" -s "$SERIAL" push \
  "$NDK/toolchains/llvm/prebuilt/$NDK_HOST/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so" \
  "$REFERENCE_ROOT/"
for size in 2 4 9; do
  "$ADB" -s "$SERIAL" push "$WORK/models/Ternary-Bonsai-2-${size}B-PTQ1_0.gguf" "$REFERENCE_ROOT/"
done
"$ADB" -s "$SERIAL" shell "chmod 755 $REFERENCE_ROOT/llama-bench"
shasum -a 256 "$WORK/prism-vulkan-build/bin/llama-bench" "$WORK/prism-vulkan-build/bin/"*.so
```

### Validate the native reference

Weight parity alone does not establish correct GPU execution. Build the included teacher-forced logit dumper against the same reference libraries:

```sh
export NDK_CXX="$NDK/toolchains/llvm/prebuilt/$NDK_HOST/bin/aarch64-linux-android28-clang++"
"$NDK_CXX" -std=c++17 -march=armv8.7a \
  -I "$WORK/prism/include" -I "$WORK/prism/ggml/include" \
  max/examples/android-llm/inference_test/reference.cpp \
  -L "$WORK/prism-vulkan-build/bin" -lllama -lggml -lggml-base \
  '-Wl,-rpath,$ORIGIN' -o "$WORK/prism-vulkan-build/bin/reference"
"$ADB" -s "$SERIAL" push "$WORK/prism-vulkan-build/bin/reference" "$REFERENCE_ROOT/"
"$ADB" -s "$SERIAL" shell "chmod 755 $REFERENCE_ROOT/reference"
```

For each model, pass the **same teacher-forced token list** to Mojo's `validate` and Prism's `reference`, pull both dumps, and compare them with `compare.py`. Here is the S26 2B example using the exported prompt (use longer token lists for broader checks):

```sh
"$ADB" -s "$SERIAL" push "$WORK/model2/prompt.txt" "$REFERENCE_ROOT/tokens.txt"
"$ADB" -s "$SERIAL" shell "cd $DEVICE_ROOT/bonsai2 && LD_LIBRARY_PATH=. ./validate ./OpenCLPlugin.so ./weights.bin ./prompt.txt ./mojo.f32"
"$ADB" -s "$SERIAL" shell "cd $REFERENCE_ROOT && LD_LIBRARY_PATH=. GGML_VK_DISABLE_ASYNC=1 GGML_VK_DISABLE_FUSION=1 ./reference ./Ternary-Bonsai-2-2B-PTQ1_0.gguf ./tokens.txt ./prism.f32 99 f16 2048"
"$ADB" -s "$SERIAL" pull "$DEVICE_ROOT/bonsai2/mojo.f32" "$WORK/mojo-reference-check.f32"
"$ADB" -s "$SERIAL" pull "$REFERENCE_ROOT/prism.f32" "$WORK/prism-reference-check.f32"
python max/examples/android-llm/inference_test/compare.py \
  "$WORK/mojo-reference-check.f32" "$WORK/prism-reference-check.f32" \
  --vocabulary 248320 --output "$WORK/native-reference-check.json"
```

The numerical comparison must pass; inspect argmax agreement as well. Mojo must also pass its independent MLX reference checks described in the root guide. Repeat for 4B/9B using their exported prompts, paths, vocabulary sizes, and capacity 512. The vocabulary is recorded in each generated `qwen_spec.h`. The S26 and recorded Exynos references use both Vulkan workaround flags above; the recorded Mali reference uses default Vulkan without them. Validate the actual selected configuration on every device.

### Measure the native reference

For a direct 2B decode sample with three native repetitions:

```sh
"$ADB" -s "$SERIAL" shell "cd $REFERENCE_ROOT && LD_LIBRARY_PATH=. GGML_VK_DISABLE_ASYNC=1 GGML_VK_DISABLE_FUSION=1 ./llama-bench -m ./Ternary-Bonsai-2-2B-PTQ1_0.gguf -p 0 -n 128 -r 3 -t 4 -ngl 99 -b 1 -ub 1 -o json" > "$WORK/prism-2b-tg128.json" 2> "$WORK/prism-2b-tg128.stderr"
```

For serial prefill use `-p 512 -n 0`; change the GGUF filename for larger models. Retain stderr, thermal snapshots, and all samples. The archived 2B reference used three repetitions in one process; the 4B/9B paired protocol used three processes with one repetition each. A direct invocation does not reproduce that pairing/order; consult the root protocol for the archived suite invocation and its separate device layout.

### Optional OpenCL reference

The Qwen3.5 reference in the report uses Prism/OpenCL. Build it in a **separate** directory with the same Prism revision and NDK, `-DGGML_OPENCL=ON -DGGML_VULKAN=OFF -DGGML_OPENCL_USE_ADRENO_KERNELS=ON -DGGML_OPENCL_EMBED_KERNELS=ON`, and ARMv8.7-A/OpenMP-off settings. The Bonsai PQ2_0/PTQ1_0 projections fall back to CPU in this OpenCL backend; use Vulkan for the Bonsai comparison table.

## Build the Bonsai 2B benchmark

Build the 2B model and Android runtime with the [root S26 guide](../../../../README.md#1-build-and-export), then link the benchmark and validation runners from that build:

```sh
python max/examples/android-llm/inference_test/build_bonsai.py \
  --build /path/to/android-build --ndk "$NDK" --output /path/to/test/bonsai
```

The alternative `--native` option accepts the historical staged app layout for auditing old results. The generated `bench` accepts:

```sh
LD_LIBRARY_PATH=. ./bench ./OpenCLPlugin.so ./weights.bin 0 128 0 3 1
# positional arguments after weights: pp, tg, context depth, repeats, seed
```

`check_bonsai_weights.py` verifies all packed codes, scales, signs and remaining parameters against the publisher's GGUF. Its only approximate comparison is the conversion of stored A_log to GGUF's negative exponential decay.

## Export and build Qwen

Install the pinned reference checkout's `gguf-py` and NumPy in a virtual
environment. Both exports retain the original Q4_K/Q5_K/Q6_K/Q8_0 byte blocks,
transposed into byte planes for adjacent output columns. There is no weight
requantization or full-weight floating-point expansion on the phone. Small
normalization/convolution/gate tensors remain FP32. Qwen3.5's MTP block and
vision encoder are excluded, matching normal text-only llama-bench.

```sh
python max/examples/android-llm/inference_test/prepare_qwen.py \
  --gguf /path/to/model-Q4_K_M.gguf --output /path/to/test/qwen3 --capacity 1024
python max/examples/android-llm/inference_test/generate_qwen.py \
  --arch qwen3 --output /path/to/test/qwen3
python max/examples/android-llm/inference_test/build_qwen.py \
  --mojo /path/to/checkout-built/mojo --ndk "$NDK" \
  --packages /path/to/compatible-std-and-max-packages \
  --runtime-build /path/to/bonsai-build --output /path/to/test/qwen3
```

For Qwen3.5, use `--arch qwen35` and its own output directory. Each model has a
separate generated `qwen_spec.h`. The generator reuses the existing Bonsai
recurrent/attention kernels, two-pass RMSNorm, and eight-output matvec pattern;
Qwen3 uses the existing dense-transformer orchestration with its own dimensions,
Q/K normalization and full-head RoPE. `--env-json` optionally supplies compiler
library paths when using an isolated checkout-built compiler.

The output contains `bench`, `validate` and the normal `runner`. `validate`
accepts plugin, weights, whitespace-separated teacher-forced token IDs and a
binary output path; it writes every vocabulary logit after each input. `runner`
uses the existing generation CLI described in the parent README. Tokenization
and text decoding remain on the host. These are greedy-generation examples,
not a tokenizer integration or UI port for Qwen.

`qwen3_reference.py` and `qwen35_reference.py` dequantize the **same** GGUF values
into independent FP32 transformers models on the host. This is required to
separate a weight-decoding/architecture bug from llama.cpp's activation
quantization/rounding. `reference.cpp`, linked with the unmodified reference
libraries, also checks its CPU and GPU execution paths. `compare.py` rejects
nonfinite values and reports full-vocabulary errors and argmax agreement.
Its numerical pass requires max absolute error <0.05 and relative L2 <0.005;
argmax agreement is reported separately because near ties can flip.

## Run on the phone

Upload outside measured intervals. Keep all runtime shared libraries beside
the executables, including `libc++_shared.so`. The default device layout is:

```text
/data/local/tmp/modular-inference-testing/
  prism/             # llama-bench and its shared libraries
  bonsai/            # bench, validate, runner, staged demo libraries, weights.bin
  qwen3/             # bench, validate, runner, shared libraries, weights.bin
  qwen35/            # bench, validate, runner, shared libraries, weights.bin
  *.gguf             # the exact reference files
```

```sh
python max/examples/android-llm/inference_test/run_suite.py \
  --adb "$ANDROID_SDK_ROOT/platform-tools/adb" --serial DEVICE_SERIAL \
  --output /path/to/results --repeats 3
```

The suite writes a JSON journal after each sample, complete command lines, raw
stdout/stderr and thermal snapshots. A failed Android command aborts the suite
and retains its output. No benchmark number is accepted from a failed process.
Use `--reference-subdir prism-vulkan` for a staged Vulkan reference.
Use separate output directories for independent runs; an invocation starts a
new journal. See the report for the measured source/binary hashes and results.

## Qualified reference configuration from this report

Default Vulkan failed the captured logit checks on this phone. Do not use its
unchecked timings as a performance baseline. The existing environment options
below restore Bonsai and Qwen3 fixture agreement; Qwen3.5 still fails Vulkan
and uses the OpenCL reference instead:

```sh
LD_LIBRARY_PATH=. GGML_VK_DISABLE_ASYNC=1 GGML_VK_DISABLE_FUSION=1 \
  ./llama-bench -m ../Ternary-Bonsai-2-2B-PTQ1_0.gguf \
  -p 0 -n 128 -r 3 -t 4 -ngl 99 -b 1 -ub 1 -o json
```

The same flags can be passed through the repo harness with
`--reference-env GGML_VK_DISABLE_ASYNC=1 --reference-env GGML_VK_DISABLE_FUSION=1`.
Use `--reference-model bonsai=Ternary-Bonsai-2-2B-PTQ1_0.gguf` only after the
lossless parameter check. See the report for accepted versus diagnostic runs.

The model files can be fetched independently of `/tmp` using the pinned
[model source list](../../../../docs/android-feasibility/inference-testing-2026-10-26/model-sources.json).
For each `[repository, revision, filename]`, run:

```sh
hf download REPOSITORY FILENAME --revision REVISION --local-dir /path/to/models
```

Use the model SHA-256 values in the recorded provenance to verify downloads.
The four files are public publisher artifacts; no model binary is embedded in
this repository. Build/export outputs can be placed in any external directory.
The repo's `prepare_qwen.py`, `generate_qwen.py`, `build_qwen.py`,
`build_bonsai.py`, `bench.cpp`, validation programs and `run_suite.py` provide
all of the model-specific export/build/test logic; temporary orchestration
scripts are not prerequisites for reproduction.
