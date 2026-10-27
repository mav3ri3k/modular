# Modular on Android

This personal fork adds an experimental Android ARM64 backend for Mojo and MAX: an OpenCL compiler target, a runtime plugin, and native Mojo model implementations. Bonsai 2 runs on Adreno, Mali, and Samsung Xclipse GPUs.

Inference runs on the phone; model export and tokenization run on the host. The implementations are text-only, with serial prefill and greedy selection. See the [backend overview](docs/android-feasibility/README.md) for architecture and the [upstream README](README_original.md) for Modular's general getting-started guide.

## Benchmark results

Bonsai 2 decode throughput, in **tokens/s (mean ± sample standard deviation)**. These are the results used in the accompanying blog: `tg128`, 128 sequential decode forwards with full-logit readback, compared with PrismML's `llama.cpp` Vulkan backend.

| Device | SoC / GPU | Model | Mojo / OpenCL | Prism / Vulkan |
| --- | --- | --- | ---: | ---: |
| Galaxy S26 | Snapdragon SM8850 / Adreno 840 | 2B | 68.41 ± 5.79 | 51.74 ± 0.65 |
| Galaxy S26 | Snapdragon SM8850 / Adreno 840 | 4B | 37.41 ± 0.05 | 27.20 ± 0.40 |
| Galaxy S26 | Snapdragon SM8850 / Adreno 840 | 9B | 20.57 ± 1.33 | 16.69 ± 0.43 |
| OnePlus CPH2717 | MediaTek MT6897 / Mali-G615 | 2B | 10.07 ± 0.01 | 6.73 ± 0.01 |
| OnePlus CPH2717 | MediaTek MT6897 / Mali-G615 | 4B | 6.88 ± 0.04 | 3.01 ± 0.03 |
| OnePlus CPH2717 | MediaTek MT6897 / Mali-G615 | 9B | 4.98 ± 0.55 | 0.76 ± 0.02 |
| Galaxy S24 FE | Exynos s5e9945 / Xclipse 940 | 2B | 20.37 ± 0.62 | 21.55 ± 0.84 |
| Galaxy S24 FE | Exynos s5e9945 / Xclipse 940 | 4B | 11.79 ± 0.04 | 9.54 ± 0.07 |
| Galaxy S24 FE | Exynos s5e9945 / Xclipse 940 | 9B | 7.59 ± 0.11 | 7.23 ± 0.04 |

Mojo and Prism were measured in separate sessions; these are not paired comparisons. Temperature, clocks, drivers, and build differences affect throughput. S26 uses specialized Adreno kernels; the other devices use the portable path, with flush-after-enqueue enabled on Mali.

### Measurement sources

Use these sources for the table's samples, build hashes, validation logs, and prefill results. Older investigation reports remain in the archive.

| Device | Mojo measurements | Prism measurements |
| --- | --- | --- |
| Galaxy S26 | [October 26 rebuild](docs/android-feasibility/inference-testing-s26-rebuild-2026-10-26/README.md) | [2B reference](docs/android-feasibility/INFERENCE_TESTING_2026-10-26.md), [4B/9B reference](docs/android-feasibility/INFERENCE_TESTING_BONSAI_4B_9B_2026-10-26.md) |
| OnePlus CPH2717 | [2B flush-on run](docs/android-feasibility/MALI_BONSAI_2B_PROFILE_2026-10-26.md), [4B/9B rebuild](docs/android-feasibility/inference-testing-mali-rebuild-2026-10-26/README.md) | [2B reference](docs/android-feasibility/MALI_BONSAI_2B_PROFILE_2026-10-26.md), [4B/9B reference](docs/android-feasibility/MALI_BONSAI_4B_9B_LLAMA_BENCH_2026-10-26.md) |
| Galaxy S24 FE | [October 27 rebuild](docs/android-feasibility/inference-testing-exynos-rebuild-2026-10-27/README.md) | [October 26 reference](docs/android-feasibility/INFERENCE_TESTING_S24FE_BONSAI_2026-10-26.md) |

The blog's Mali 2B value is from the earlier flush-on run; the later rebuild measured **8.67 ± 1.69 tok/s**. The table therefore combines two Mali measurement sessions.

## Reproduce the results

Follow the prerequisites, then the instructions for your device. The steps cover export, build, numerical validation, and timing. A clean-checkout reproduction is still pending; see [build provenance](#build-provenance).

- [Prerequisites](#reproduction-prerequisites)
- [Galaxy S26](#galaxy-s26-exact-inputs-and-benchmark)
- [OnePlus Mali](#oneplus-mali-build-and-reproduce)
- [Samsung Exynos / Xclipse](#samsung-exynos--xclipse-portable-build)
- [Prism reference and benchmark protocol](#benchmark-protocol)

## Reproduction prerequisites

The measured development host was an Apple Silicon Mac. The Bonsai checkpoint exporter and FP32 reference use MLX on macOS; Android inference itself does not need Python. A Linux exporter path for these MLX references is not established. You need:

- This fork and its source-built Mojo compiler; released Mojo does not contain this OpenCL target. Use a revision that includes this guide and its benchmark builder; see [build provenance](#build-provenance) for historical artifact differences.
- Bazel through `./bazelw`, which uses the version in [.bazelversion](.bazelversion); a working C++ toolchain, CMake, Git, and sufficient disk space for compiler dependencies, checkpoints and builds. On the measured Mac a macOS SDK override was needed; the [compiler build notes](Mojo/tools/mojo-opencl/README.md#build-and-reproduce) describe it.
- Android SDK platform-tools (`adb`), Android NDK **29.0.14206865**, a connected ARM64 Android device with USB debugging, and an OpenCL-capable vendor driver. The runtime targets Android API 33+. GPU features and vendor library access differ across devices. Enable and authorize USB debugging and check `adb devices`.
- Python **3.13.11** as tested, a virtual environment and the exact direct dependencies in [Bonsai requirements](max/examples/android-llm/bonsai/requirements.txt). `hf` downloads checkpoints. The reference GGUF parity checks also require NumPy and `gguf-py` from the pinned Prism checkout. These direct pins are not a full transitive lockfile.
- Several GB of free space on both host and phone; the 9B device allocation alone was about 2.4 GiB, excluding driver and process overhead. Keep downloaded models and generated binaries outside Git. Follow the model publishers' licenses and access terms.

From the repository root, set paths for your machine (the commands below use Bash):

```sh
export NDK="$HOME/Library/Android/sdk/ndk/29.0.14206865"
export ADB="$HOME/Library/Android/sdk/platform-tools/adb"
export WORK=/tmp/modular-android-repro
mkdir -p "$WORK"
python3.13 -m venv "$WORK/venv"
. "$WORK/venv/bin/activate"
python -m pip install -r max/examples/android-llm/bonsai/requirements.txt
"$ADB" devices
./bazelw build --compilation_mode=opt \
  //Mojo/tools/mojo:mojo //Mojo/tools/mojo-opencl:mojo-opencl-emit
```

If a build fails on a newer macOS SDK with `ld64.lld` errors about an unknown `arm64e.x1` target, use an installed **macOS 26.5 SDK** as an isolated Bazel sysroot. On the measured Mac it was at `/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk`. Install that SDK separately if unavailable. From the repository root:

```sh
export MACOS_SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk
mkdir -p "$WORK/macos-sdk/sysroot"
printf 'module(name = "sysroot-macos")\n' > "$WORK/macos-sdk/MODULE.bazel"
cp max/examples/android-llm/macos_sdk.BUILD.bazel "$WORK/macos-sdk/sysroot/BUILD.bazel"
ln -s "$MACOS_SDK/usr" "$WORK/macos-sdk/sysroot/usr"
ln -s "$MACOS_SDK/System" "$WORK/macos-sdk/sysroot/System"
./bazelw build --compilation_mode=opt --override_repository="sysroot-macos=$WORK/macos-sdk" \
  //Mojo/tools/mojo:mojo //Mojo/tools/mojo-opencl:mojo-opencl-emit
```

For the `build.py` command below, add `--bazel-arg=--override_repository=sysroot-macos=$WORK/macos-sdk` to pass the same override to its Bazel builds. This workaround is host-specific. Record `git rev-parse HEAD`, `git status --short`, host and phone OS versions, GPU/driver details, NDK version, model hashes, and hashes of deployed binaries. Use the same device model, firmware and driver to make the closest comparison; even then thermals and clocks vary.

## Galaxy S26: exact inputs and benchmark

The recorded phone was a **Samsung Galaxy S26 SM-S942U1**, Qualcomm **SM8850 / Snapdragon**, **Adreno 840**, Android API **37**, tested October 26, 2026. The Adreno build targets `adreno-840` and uses 64-lane subgroup operations; it is not the portable build for other GPUs. Firmware and driver version were not fully pinned in the archive, so record them before your run. Connect only the intended device or set its serial explicitly:

```sh
export SERIAL=DEVICE_SERIAL
"$ADB" -s "$SERIAL" shell 'getprop ro.product.model; getprop ro.soc.model; getprop ro.build.version.sdk; getprop ro.build.fingerprint; dumpsys battery; dumpsys thermalservice' > "$WORK/device-before.txt"
```

The following uses **MLX checkpoints** for the Modular model and **GGUF files** for the Prism comparison. Each checkpoint revision pins all files; the SHA-256 below is specifically its `model.safetensors`. The GGUF is a separate publisher file, and the weight parity checks establish that it represents the same ternary values. The exported `weights.bin` is generated locally and is not a download.

| Model | MLX revision | `model.safetensors` SHA-256 | GGUF revision | PTQ1_0 GGUF SHA-256 |
| --- | --- | --- | --- | --- |
| 2B | `1c6cbae21f8c38830ce337972b46fdae7b18eaaa` | `da5095b9f34d7f0d5094090afeeadf124eef821c75136dc28957ce586ae7ac0c` | `683f7be2629342a50987627b40881e54761a2430` | `36061ad406df8a4903a6db005be350c50fabfaf717f9ef0acf4b76adcd9929fa` |
| 4B | `dff2caaa23fa8706164c44c10f1b58feb405f76a` | `432f48d0edcb8da2e6a4e0c96078588362397e2603e7fdea99957babfe968259` | `8d6557168168f91ea6697a4a49ed2ef622b53adb` | `56711263ec9a9ffeaa74a3a0e58dece8127cf9268404746c721113ca76994d60` |
| 9B | `f4e28a59531731c41b6dfac88cf271bd58f0a287` | `b8ebd13459dcd53504432a4a0a797287c3a4c00d80835466dbb499f05b7512bb` | `5b2abb02362efbed46bbe204014b02837778b2c2` | `3561df7ac7d26a71ffa991ac2db740b10841c3a350aa2676ccd3ff5d428b4e8b` |

Download all three checkpoints (approximately 1.2, 1.9 and 3.5 GB for the safetensors files alone) and the comparison files:

```sh
hf download prism-ml/Ternary-Bonsai-2-2B-mlx-2bit --revision 1c6cbae21f8c38830ce337972b46fdae7b18eaaa --local-dir "$WORK/ckpt2"
hf download prism-ml/Ternary-Bonsai-2-4B-mlx-2bit --revision dff2caaa23fa8706164c44c10f1b58feb405f76a --local-dir "$WORK/ckpt4"
hf download prism-ml/Ternary-Bonsai-2-9B-mlx-2bit --revision f4e28a59531731c41b6dfac88cf271bd58f0a287 --local-dir "$WORK/ckpt9"
hf download prism-ml/Ternary-Bonsai-2-2B-gguf Ternary-Bonsai-2-2B-PTQ1_0.gguf --revision 683f7be2629342a50987627b40881e54761a2430 --local-dir "$WORK/models"
hf download prism-ml/Ternary-Bonsai-2-4B-gguf Ternary-Bonsai-2-4B-PTQ1_0.gguf --revision 8d6557168168f91ea6697a4a49ed2ef622b53adb --local-dir "$WORK/models"
hf download prism-ml/Ternary-Bonsai-2-9B-gguf Ternary-Bonsai-2-9B-PTQ1_0.gguf --revision 5b2abb02362efbed46bbe204014b02837778b2c2 --local-dir "$WORK/models"
shasum -a 256 "$WORK"/ckpt{2,4,9}/model.safetensors "$WORK"/models/*PTQ1_0.gguf
```

Compare each hash with the table before proceeding. The original 2B `weights.bin` at capacity 2048 had SHA-256 `ac588b263b9d12cef29595efde850e771138990e74e026d654d505f173ecd657`; the 4B and 9B exports at capacity 512 had hashes `1c33e9b3521f14025c3c6dc47a40900a23665efaa381ad1a62461777c75e86de` and `f989b6bd549be758310855ed2f710dc44251159a9645c97aa5b757f933ba1c8d`. Compare exports after the steps below, but do not assume matching weight hashes imply matching executables.

### 1. Build and export

From the repository root, after completing the prerequisites above:

```sh
export MOJO="$PWD/bazel-bin/Mojo/tools/mojo/mojo"
python max/examples/android-llm/build.py --variant bonsai --mojo "$MOJO" --ndk "$NDK" --output "$WORK/runtime"
python max/examples/android-llm/bonsai/prepare.py --checkpoint "$WORK/ckpt2" --output "$WORK/model2" --prompt 'What is the capital of France?' --tokens 8 --capacity 2048
python max/examples/android-llm/inference_test/build_bonsai.py --build "$WORK/runtime" --ndk "$NDK" --output "$WORK/bonsai2"
cp "$WORK/model2/weights.bin" "$WORK/model2/prompt.txt" "$WORK/model2/reference.f32" "$WORK/bonsai2/"

python max/examples/android-llm/bonsai_large/prepare.py --checkpoint "$WORK/ckpt4" --output "$WORK/bonsai4" --capacity 512 --tokens 32
python max/examples/android-llm/bonsai_large/generate.py --size 4b --output "$WORK/bonsai4"
python max/examples/android-llm/bonsai_large/build.py --mojo "$MOJO" --ndk "$NDK" --packages "$WORK/runtime" --runtime-build "$WORK/runtime" --output "$WORK/bonsai4"

python max/examples/android-llm/bonsai_large/prepare.py --checkpoint "$WORK/ckpt9" --output "$WORK/bonsai9" --capacity 512 --tokens 32
python max/examples/android-llm/bonsai_large/generate.py --size 9b --output "$WORK/bonsai9"
python max/examples/android-llm/bonsai_large/build.py --mojo "$MOJO" --ndk "$NDK" --packages "$WORK/runtime" --runtime-build "$WORK/runtime" --output "$WORK/bonsai9"
shasum -a 256 "$WORK"/bonsai{2,4,9}/weights.bin "$WORK/runtime/model.o" "$WORK"/bonsai{4,9}/model.o
```

Record your plugin hashes with `shasum -a 256 "$WORK"/bonsai{2,4,9}/OpenCLPlugin.so`. The [S26 rebuild provenance](docs/android-feasibility/inference-testing-s26-rebuild-2026-10-26/provenance.json) contains the recorded input and binary hashes; historical builds used different model objects and, for 4B/9B, a different plugin.

The 2B model object is `runtime/model.o`. The runtime directory also contains compatible `std.mojoc`, `max.mojoc` and `android_artifacts.json`. If needed, pass the same SDK override to Bazel and `build.py` using `--bazel-arg=...`; the larger builders accept optional `--env-json` for compiler library paths. Check the export manifests and record their hashes as well.

### 2. Deploy and check correctness

Use a dedicated device directory and do not time uploads or loading. These commands leave the files on the phone for subsequent runs; make sure it has enough free space.

```sh
export DEVICE_ROOT=/data/local/tmp/modular-bonsai-repro
"$ADB" -s "$SERIAL" shell "mkdir -p $DEVICE_ROOT/bonsai2 $DEVICE_ROOT/bonsai4 $DEVICE_ROOT/bonsai9"
for size in 2 4 9; do
  "$ADB" -s "$SERIAL" push "$WORK/bonsai$size/." "$DEVICE_ROOT/bonsai$size/"
  "$ADB" -s "$SERIAL" shell "chmod 755 $DEVICE_ROOT/bonsai$size/runner $DEVICE_ROOT/bonsai$size/bench"
done
for size in 2 4 9; do
  steps=32; if [ "$size" = 2 ]; then steps=8; fi
  "$ADB" -s "$SERIAL" shell "cd $DEVICE_ROOT/bonsai$size && LD_LIBRARY_PATH=. ./runner ./OpenCLPlugin.so ./weights.bin ./prompt.txt ./reference.f32 $steps" > "$WORK/validate-$size.log" 2>&1 || { cat "$WORK/validate-$size.log"; break; }
  cat "$WORK/validate-$size.log"
  grep -q "PASS: all generated tokens and logits match reference" "$WORK/validate-$size.log" || break
done
```

Each run must report `PASS: all generated tokens and logits match reference`; investigate any error before timing. These are short fixtures, not broad quality or long-context validation. The original 2B report also used a longer 96-output-token fixture; the larger-model report used 32 output tokens. Export a longer reference if you want to repeat that coverage.

### 3. Measure Modular inference

The included [bench.cpp](max/examples/android-llm/inference_test/bench.cpp) is a llama-bench-compatible workload, not llama.cpp itself. `tg128` executes 128 decode forwards with a full-vocabulary transfer and synchronization per token; `pp512` uses serial batch-one prefill. Loading, GPU compilation, tokenization and sampling are outside the timer. Warmup and cache reset are described in the [harness documentation](max/examples/android-llm/inference_test/README.md).

Run three separate processes per model and workload, preserving every `RESULT` line. The 10-second pause matches the paired large-model protocol but does not control device temperature:

```sh
for size in 2 4 9; do
  for workload in tg128 pp512; do
    pp=0; tg=128; if [ "$workload" = pp512 ]; then pp=512; tg=0; fi
    for sample in 1 2 3; do
      sleep 10
      "$ADB" -s "$SERIAL" shell 'dumpsys battery; dumpsys thermalservice' > "$WORK/$size-$workload-$sample-thermal-before.txt"
      "$ADB" -s "$SERIAL" shell "cd $DEVICE_ROOT/bonsai$size && LD_LIBRARY_PATH=. ./bench ./OpenCLPlugin.so ./weights.bin $pp $tg 0 1 1" > "$WORK/$size-$workload-$sample.stdout" 2> "$WORK/$size-$workload-$sample.stderr" || { cat "$WORK/$size-$workload-$sample.stderr"; exit 1; }
      grep -q "^RESULT " "$WORK/$size-$workload-$sample.stdout" || { echo "Missing RESULT"; exit 1; }
      "$ADB" -s "$SERIAL" shell 'dumpsys battery; dumpsys thermalservice' > "$WORK/$size-$workload-$sample-thermal-after.txt"
    done
  done
done
```

Summarize the individual rates (the raw stdout, nanosecond samples, stderr and thermal snapshots remain in `WORK`):

```sh
python - "$WORK" <<'PYCODE'
import json, pathlib, statistics, sys
root = pathlib.Path(sys.argv[1])
for size in (2, 4, 9):
    for workload in ("tg128", "pp512"):
        results = []
        for sample in (1, 2, 3):
            lines = (root / f"{size}-{workload}-{sample}.stdout").read_text().splitlines()
            result = json.loads(next(line[7:] for line in lines if line.startswith("RESULT ")))
            results.append(result["avg_ts"])
        print(f"{size}B {workload}: {statistics.mean(results):.2f} ± {statistics.stdev(results):.2f} tok/s; samples: {results}")
PYCODE
```

Report the arithmetic mean and sample standard deviation of the three `avg_ts` values, retaining the raw nanoseconds and errors. Stop and investigate a failed process or missing `RESULT`; do not count it as a sample. Record the order, device battery and temperatures, build and plugin hashes. Runs above are grouped by model; the original 4B/9B comparison alternated Modular and Prism. The original 2B final reference used three repetitions in one process, whereas its Modular samples were separate processes. For comparisons with Prism, follow the reference section below and the exact archived reports, including correctness gates.

## OnePlus Mali: build and reproduce

The tested device is **OnePlus CPH2717** (`CPH2717IN`), MediaTek **MT6897**, **Mali-G615 MC6 r1p3**, Android 16 / API 36. Use the same pinned checkpoints and GGUFs as S26. The portable model avoids Adreno-specific subgroup operations; the Mali benchmark requires flush-after-enqueue, as shown below.

Start with the host prerequisites, checkpoint downloads, compiler and `"$WORK/runtime"` build above. Select the phone with `SERIAL`, and check `"$ADB" -s "$SERIAL" shell 'getprop ro.product.model; getprop ro.soc.model; getprop ro.build.version.sdk; getprop ro.build.fingerprint'`. The specific phone image's vendor OpenCL driver is visible to executables under `/data/local/tests/vendor`, but was not visible to ordinary executables in `/data/local/tmp`. The commands below use that shell-writable path; it is specific to this phone image, not a general Android app integration method.

Export the 2B model at capacity 128 to match the original Mali run. Export the 4B and 9B checkpoints as in the S26 instructions above (capacity 512). Generate and build portable models using the same compiler and runtime:

```sh
python max/examples/android-llm/bonsai/prepare.py --checkpoint "$WORK/ckpt2" --output "$WORK/mali-model2" --prompt 'What is the capital of France?' --tokens 8 --capacity 128
for size in 2 4 9; do
  python max/examples/android-llm/bonsai_portable/generate.py --size "${size}b" --output "$WORK/portable$size"
  python max/examples/android-llm/bonsai_portable/build.py --mojo "$MOJO" --ndk "$NDK" --packages "$WORK/runtime" --runtime-build "$WORK/runtime" --output "$WORK/portable$size"
done
cp "$WORK/mali-model2/weights.bin" "$WORK/mali-model2/prompt.txt" "$WORK/mali-model2/reference.f32" "$WORK/portable2/"
for size in 4 9; do
  cp "$WORK/bonsai$size/weights.bin" "$WORK/bonsai$size/prompt.txt" "$WORK/bonsai$size/reference.f32" "$WORK/portable$size/"
done
shasum -a 256 "$WORK"/portable{2,4,9}/weights.bin "$WORK"/portable{2,4,9}/model.o "$WORK"/portable{2,4,9}/OpenCLPlugin.so
```

The archived Mali 2B capacity-128 `weights.bin` SHA-256 is `391c052908906ee2a45aaf7ebc6183a71e4cd16176959fbd4034c3c885e002eb`. The 4B and 9B weight hashes are in the S26 table's export notes. The original flush-on runs used an OpenCL plugin with SHA-256 `3932cdec258d161bac574af81c8d0d0d777a39b0e586d1682c04fec69c05f771`; a rebuilt plugin can differ, so record its hash. Check that the selected plugin supports `MODULAR_OPENCL_FLUSH_AFTER_ENQUEUE` and validate it on the device. This setting improved performance on the tested phone; do not assume the same effect on another driver.

Stage the outputs under a separate directory and copy only the executables into the vendor namespace:

```sh
export DEVICE_ROOT=/data/local/tmp/modular-mali-repro
"$ADB" -s "$SERIAL" shell "mkdir -p $DEVICE_ROOT"
for size in 2 4 9; do
  "$ADB" -s "$SERIAL" shell "mkdir -p $DEVICE_ROOT/portable$size"
  "$ADB" -s "$SERIAL" push "$WORK/portable$size/." "$DEVICE_ROOT/portable$size/"
  "$ADB" -s "$SERIAL" shell "cp $DEVICE_ROOT/portable$size/runner /data/local/tests/vendor/modular-repro-$size-runner && cp $DEVICE_ROOT/portable$size/bench /data/local/tests/vendor/modular-repro-$size-bench && chmod 755 /data/local/tests/vendor/modular-repro-$size-runner /data/local/tests/vendor/modular-repro-$size-bench"
done
```

Check every model before measuring it. The 2B fixture below has eight generated tokens; if you exported the S26 4B/9B fixtures with `--tokens 32`, use 32 steps. Each command must print `PASS: all generated tokens and logits match reference`.

```sh
for size in 2 4 9; do
  steps=32; if [ "$size" = 2 ]; then steps=8; fi
  "$ADB" -s "$SERIAL" shell "cd $DEVICE_ROOT/portable$size && LD_LIBRARY_PATH=. MODULAR_OPENCL_FLUSH_AFTER_ENQUEUE=1 /data/local/tests/vendor/modular-repro-$size-runner ./OpenCLPlugin.so ./weights.bin ./prompt.txt ./reference.f32 $steps" > "$WORK/mali-$size-validation.log" 2>&1 || { cat "$WORK/mali-$size-validation.log"; break; }
  cat "$WORK/mali-$size-validation.log"
  grep -q 'PASS: all generated tokens and logits match reference' "$WORK/mali-$size-validation.log" || break
done
```

For the recorded flush-on `tg128` protocol, use three repetitions in a single process, one warmup, seed 1, and save the `RESULT` JSON and thermal data. Runs are sequential and not thermally controlled. The result's `avg_ts` is the mean of per-sample rates, and `stddev_ts` is their sample standard deviation.

```sh
for size in 2 4 9; do
  sleep 10
  "$ADB" -s "$SERIAL" shell 'dumpsys battery; dumpsys thermalservice' > "$WORK/mali-$size-thermal-before.txt"
  "$ADB" -s "$SERIAL" shell "cd $DEVICE_ROOT/portable$size && LD_LIBRARY_PATH=. MODULAR_OPENCL_FLUSH_AFTER_ENQUEUE=1 /data/local/tests/vendor/modular-repro-$size-bench ./OpenCLPlugin.so ./weights.bin 0 128 0 3 1" > "$WORK/mali-$size-tg128.stdout" 2> "$WORK/mali-$size-tg128.stderr" || { cat "$WORK/mali-$size-tg128.stderr"; exit 1; }
  grep '^RESULT ' "$WORK/mali-$size-tg128.stdout" || { echo 'Missing RESULT'; exit 1; }
  "$ADB" -s "$SERIAL" shell 'dumpsys battery; dumpsys thermalservice' > "$WORK/mali-$size-thermal-after.txt"
done
```

The [portable guide](max/examples/android-llm/bonsai_portable/README.md) includes the historical diagnostic plugin recipe and the Prism reference commands. For Exynos / Xclipse, use the device-specific instructions below and repeat numerical checks on your own build.

## Samsung Exynos / Xclipse: portable build

The tested device is the **Galaxy S24 FE SM-S721B**, Exynos **s5e9945**, Samsung **Xclipse 940**, Android 16 / API 36. Its fingerprint for the October 27 rerun was `samsung/r12sxins/r12s:16/BP4A.251205.006/S721BXXSEDZI1:user/release-keys`. Use the same pinned checkpoints, exports and host toolchain above. Unlike the tested OnePlus image, this device could load the vendor OpenCL driver from executables under `/data/local/tmp`; it did not require the OnePlus vendor-namespace workaround. The device-specific Adreno binaries cannot be used: build all three sizes with `opencl-generic` as in the OnePlus section.

For a comparison with the previous 2B experiment, export 2B with capacity 2048 as in the S26 steps (the larger models use capacity 512), then place the exported files with each portable build:

```sh
cp "$WORK/model2/weights.bin" "$WORK/model2/prompt.txt" "$WORK/model2/reference.f32" "$WORK/portable2/"
for size in 4 9; do
  cp "$WORK/bonsai$size/weights.bin" "$WORK/bonsai$size/prompt.txt" "$WORK/bonsai$size/reference.f32" "$WORK/portable$size/"
done
export DEVICE_ROOT=/data/local/tmp/modular-exynos-repro
"$ADB" -s "$SERIAL" shell "mkdir -p $DEVICE_ROOT"
for size in 2 4 9; do
  "$ADB" -s "$SERIAL" shell "mkdir -p $DEVICE_ROOT/portable$size"
  "$ADB" -s "$SERIAL" push "$WORK/portable$size/." "$DEVICE_ROOT/portable$size/"
done
```

Validate each model using its exported reference before benchmarking. The example assumes the S26 exports above: eight steps for 2B and 32 for the larger models. Check for the `PASS` line and stop on failure.

```sh
for size in 2 4 9; do
  steps=32; if [ "$size" = 2 ]; then steps=8; fi
  "$ADB" -s "$SERIAL" shell "cd $DEVICE_ROOT/portable$size && LD_LIBRARY_PATH=. ./runner ./OpenCLPlugin.so ./weights.bin ./prompt.txt ./reference.f32 $steps" > "$WORK/exynos-$size-validation.log" 2>&1 || { cat "$WORK/exynos-$size-validation.log"; break; }
  cat "$WORK/exynos-$size-validation.log"
  grep -q 'PASS: all generated tokens and logits match reference' "$WORK/exynos-$size-validation.log" || break
done
```

For `tg128` decode and serial `pp512`, use the root README's [S26 measurement loop](#3-measure-modular-inference), replacing `bonsai$size` in the device path with `portable$size`. Do not enable the Mali-specific flush setting when comparing with the October 27 Exynos rerun. Retain all three separate-process samples and thermal snapshots. The recorded rebuild used short correctness fixtures (four generated tokens for 4B/9B); use longer references for broader validation.

## Benchmark protocol

The benchmark harness and detailed native build instructions are in [inference_test](max/examples/android-llm/inference_test/README.md). The comparison uses Prism's `llama.cpp` at commit `c96baf4abe24d11a50f326e40b6223b687449d63`:

```sh
git clone https://github.com/PrismML-Eng/llama.cpp.git "$WORK/prism"
git -C "$WORK/prism" checkout --detach c96baf4abe24d11a50f326e40b6223b687449d63
```

The measured Vulkan reference used these pins, verified against the local source checkouts and archived CMake cache:

| Component | Revision / version |
| --- | --- |
| Prism `llama.cpp` | `c96baf4abe24d11a50f326e40b6223b687449d63` |
| SPIRV-Headers | `fbf86836a9b1966586faf7be7bb8a6440e156d62` |
| Vulkan-Hpp | `e48d470db25ad598aff8aa80292667e66c45138e` |
| Vulkan-Headers (Vulkan-Hpp submodule) | `73814d006638aa00bd553215b054f0f02e457e22` |
| Android NDK / host `glslc` | `29.0.14206865` / `shaderc v2022.3 ndk-r29-rc1` |
| CMake on the development host | `4.4.4` |

Follow the [pinned Vulkan build, staging, and correctness recipe](max/examples/android-llm/inference_test/README.md#build-the-vulkan-reference). It targets ARM64 Android API 28, ARMv8.7-A, shared libraries, OpenMP off, and Vulkan on. The historical cache has **empty C/C++ Release flags**, not CMake's usual `-O3 -DNDEBUG`; the recipe preserves that configuration. Mojo's runtime separately targets API 33. The [archived CMake cache](docs/android-feasibility/inference-testing-2026-10-26/prism-vulkan-CMakeCache.txt) preserves the remaining settings with personal paths redacted.

Install the GGUF reader from that pinned checkout, then verify the exported weights against the downloaded GGUFs before comparing backends:

```sh
python -m pip install -e "$WORK/prism/gguf-py"
python max/examples/android-llm/inference_test/check_bonsai_weights.py \
  --weights "$WORK/bonsai2/weights.bin" \
  --gguf "$WORK/models/Ternary-Bonsai-2-2B-PTQ1_0.gguf" --output "$WORK/parity2.json"
for size in 4 9; do
  python max/examples/android-llm/bonsai_large/check_weights.py \
    --weights "$WORK/bonsai$size/weights.bin" \
    --gguf "$WORK/models/Ternary-Bonsai-2-${size}B-PTQ1_0.gguf" --output "$WORK/parity$size.json"
done
```

Each checker must succeed with no failed comparisons. Checkpoint and GGUF revisions and SHA-256 values are pinned in the model table above; the host exporter's direct Python dependencies are pinned in `bonsai/requirements.txt`. A full transitive Python environment lock is not provided.

`tg128` measures 128 decode forwards and `pp512` 512 serial prompt forwards. Full-vocabulary host transfer is timed; model loading, compilation, tokenization and sampling are excluded. Reference and Mojo differ in cache precision and implementation. Save all samples, commands, thermal snapshots and failed runs. Use `--skip-batched` for the serial-only comparison. On S26, default Vulkan failed logit checks; the qualified Bonsai reference uses **both** `GGML_VK_DISABLE_ASYNC=1` and `GGML_VK_DISABLE_FUSION=1`. Validate the selected reference path on each device and model rather than assuming these flags work everywhere.

Once the documented device layout has been staged, the 4B/9B S26 paired protocol is:

```sh
python max/examples/android-llm/inference_test/run_suite.py \
  --adb "$ADB" --serial DEVICE_SERIAL --output "$WORK/results-large" \
  --reference-subdir prism-vulkan --models bonsai4 bonsai9 \
  --reference-env GGML_VK_DISABLE_ASYNC=1 --reference-env GGML_VK_DISABLE_FUSION=1 \
  --repeats 3 --samples-per-run 1 --skip-batched
```

The 2B final S26 reference used three repetitions in one process, while its Mojo samples came from separate processes. Exact commands and raw data are in the [2B report](docs/android-feasibility/INFERENCE_TESTING_2026-10-26.md); do not substitute the early forward-only or incorrect default-Vulkan diagnostics. The generic Mali path needs its vendor-namespace staging and flush setting, so the standard suite command above is not sufficient there. Its reports contain the executed commands.

### Build provenance

The table uses rebuilt Mojo binaries for S26, Mali 4B/9B, and Exynos. These differ from the original study's model objects and some plugins. The original S26 2B binary also depended on artifacts from a separate `mojo-chat` checkout. The measurement sources above preserve both historical and rebuilt hashes.

The instructions build from this fork, but a clean-checkout reproduction and reconstruction of the exact historical app artifacts remain unverified. Record your source revision, working-tree changes, model and binary hashes, firmware, and driver versions with each run.

## More documentation

- [Android LLM example](max/examples/android-llm/README.md), [Bonsai 2B](max/examples/android-llm/bonsai/README.md), [4B/9B](max/examples/android-llm/bonsai_large/README.md), and [portable variants](max/examples/android-llm/bonsai_portable/README.md).
- [Android feasibility and investigation archive](docs/android-feasibility/README.md); the [2B inference report](docs/android-feasibility/INFERENCE_TESTING_2026-10-26.md) also includes experimental Qwen results.
- [Original upstream README](README_original.md), [license](LICENSE), and [contribution guide](CONTRIBUTING.md).
