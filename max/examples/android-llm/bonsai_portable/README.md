# Portable OpenCL Bonsai 2 2B, 4B and 9B

These experimental, text-only variants generate generic OpenCL builds of the
existing direct DeviceContext models. They keep the packed weights, FP16 dot
chains and FP32 state, but avoids fixed-width subgroup shuffles. Hadamard uses
nine separate butterfly passes, packed projection joins groups in each work
item, and normalization and recurrent updates use scalar reductions. The
runner selects greedy tokens from full logits on the host. This is a
correctness-oriented baseline, not an optimized Mali kernel or a MAX graph.
The original Adreno model and app are unchanged.

With the compatible Android toolchain and the 2B model export described in the
parent example:

```sh
python max/examples/android-llm/bonsai_portable/generate.py --output /path/to/build
python max/examples/android-llm/bonsai_portable/build.py \
  --mojo /path/to/mojo --packages /path/to/packages --env-json /path/to/env.json \
  --ndk /path/to/ndk --runtime-build /path/to/runtime --output /path/to/build
```

The generator specializes the existing working-tree 2B source; keep that source
with the build when recording a result. `build.py` uses `opencl-generic` and
creates the existing `runner`, `validate` and llama-bench-compatible `bench`.
Pass `--size 4b` or `--size 9b` to `generate.py` for the larger variants;
the default is `2b`. Export their matching checkpoints with
[`bonsai_large/prepare.py`](../bonsai_large/README.md) and use the resulting
`weights.bin`, `prompt.txt` and `reference.f32` with the corresponding
generated build. The 4B and 9B variants have 32 value heads mapped to 16 key
heads; 9B also has a separate output head and a 4096-element Hadamard block,
implemented here with twelve subgroup-independent butterfly stages. The
compact centered scales and shared sign groups are checked by the common loader.

On the tested Mali phone, 2B has passed an eight-token generation fixture and
a separate 32-step teacher-forced fixture. The 4B and 9B variants have passed
short four-token fixtures and were benchmarked; their accuracy coverage remains
limited. Revalidate every new build on its target device. The root
[reproduction guide](../../../../README.md#oneplus-mali-build-and-reproduce)
documents the portable build and device-specific staging.

On the tested OnePlus CPH2717, the vendor's OpenCL library is discoverable from
Android's vendor linker namespace. An ordinary binary under `/data/local/tmp`
sees an ICD loader with no platforms. The phone exposes `/data/local/tests/vendor`
as a shell-writable vendor-namespace path: stage libraries, model and inputs
under a private directory in `/data/local/tmp`, copy the relevant executable
into `/data/local/tests/vendor`, then execute it with the staged directory as
working directory and `LD_LIBRARY_PATH=.`. For example:

```sh
adb shell 'cp /data/local/tmp/modular-bonsai-mali/portable/runner /data/local/tests/vendor/bonsai-portable-runner'
adb shell 'cd /data/local/tmp/modular-bonsai-mali/portable && LD_LIBRARY_PATH=. /data/local/tests/vendor/bonsai-portable-runner ./OpenCLPlugin.so ../weights.bin ../prompt.txt ../reference.f32 8'
```

This namespace arrangement is specific to the tested Android image and is not
a general app integration solution. A normal application needs its own
supported vendor-library access mechanism. The `subgroup_probe.cpp` diagnostic
queries the Mali vendor driver directly; it does not modify the system.

On the tested phone, a plugin built from the current `OpenCLPlugin.cpp` supports
the opt-in environment variable `MODULAR_OPENCL_FLUSH_AFTER_ENQUEUE=1`. Enabling
it greatly improved decode throughput while passing the 32-step fixture. It is
not a universal default: the older plugin staged by some runtime builds does
not implement the option, and its effect has not been established on other
devices or workloads. See the [profiling report](../../../../docs/android-feasibility/MALI_BONSAI_2B_PROFILE_2026-10-26.md).

For a diagnostic plugin, generate the unsupported HAL stubs and build against
the checkout's SDK and OpenCL headers with the Android NDK. The following
uses `NDK` and `OPENCL_HEADERS` set to local paths:

```sh
mkdir -p /tmp/bonsai-opencl-profile
python Target/lib/OpenCL/generate_unsupported.py SDK/include \
  Target/lib/OpenCL/OpenCLPlugin.cpp /tmp/bonsai-opencl-profile/Unsupported.cpp
"$NDK/toolchains/llvm/prebuilt/darwin-x86_64/bin/aarch64-linux-android33-clang++" \
  -shared -fPIC -fvisibility=hidden -O2 -std=c++17 \
  -ISDK/include -ITarget/include -I"$OPENCL_HEADERS" \
  Target/lib/OpenCL/OpenCLPlugin.cpp /tmp/bonsai-opencl-profile/Unsupported.cpp \
  -ldl -o /tmp/bonsai-opencl-profile/OpenCLPlugin.so
adb push /tmp/bonsai-opencl-profile/OpenCLPlugin.so \
  /data/local/tmp/modular-bonsai-mali/portable/ProfileOpenCLPlugin.so
adb shell 'cd /data/local/tmp/modular-bonsai-mali/portable && LD_LIBRARY_PATH=. MODULAR_OPENCL_FLUSH_AFTER_ENQUEUE=1 /data/local/tests/vendor/bonsai-portable-bench ./ProfileOpenCLPlugin.so ../weights.bin 0 128 0 3 1'
```

The toolchain path shown is for the tested Mac host. Validate outputs on the
target device before relying on the resulting model or timing.
