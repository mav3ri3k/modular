# Android inference comparison: Bonsai 2 on a Galaxy S24 FE

This repeats the [S26 Bonsai 4B/9B comparison](INFERENCE_TESTING_BONSAI_4B_9B_2026-10-26.md)
and the 2B row of the [earlier S26 report](INFERENCE_TESTING_2026-10-26.md)
on a different phone. The connected device is a Galaxy S24 FE, SM-S721B
(`r12s`), Exynos s5e9945, Samsung Xclipse 940, Android 16 / API 36, 7.1 GiB RAM.
It is not a Galaxy S24 (SM-S921) and not the S26 Snapdragon SM8850 / Adreno 840.

Prism is the same unmodified `llama.cpp` commit,
`c96baf4abe24d11a50f326e40b6223b687449d63`, Vulkan `llama-bench` binary
SHA-256 `1fe6f7ba2f942809cb44362919c6752076d2d51d230fdd45ba1f0e5fcc8feca1`.
Every Prism sample uses PTQ1_0, `GGML_VK_DISABLE_ASYNC=1`,
`GGML_VK_DISABLE_FUSION=1`, `-b 1 -ub 1`, `-t 4`, `-ngl 99`, and one repetition
per process. Full-vocabulary transfer is inside `llama_decode`. Loading,
compilation, tokenization and host sampling are outside the timer. Three
samples are kept. For 2B the two backends alternate, with a fixed 10-second
pause. There is no clock or thermal control. Battery started at 21% / 31.0°C
on USB power and reached 18% / 41.3°C before the last attempted run.

Raw samples and thermal snapshots are in
[inference-testing-s24fe-bonsai-2026-10-26](inference-testing-s24fe-bonsai-2026-10-26/).

## Decode: `tg128`

Mean ± sample standard deviation, tok/s.

| Model | This phone, Prism Vulkan | S26 Prism Vulkan | This phone, Mojo |
| --- | ---: | ---: | ---: |
| Bonsai 2 2B | 21.55 ± 0.84 | 51.74 ± 0.65 | 21.17 ± 0.52 portable OpenCL |
| Bonsai 2 4B | 9.54 ± 0.07 | 27.20 ± 0.40 | not run |
| Bonsai 2 9B | 7.23 ± 0.04 | 16.69 ± 0.43 | not run |

Prism decode keeps the same order, 2B then 4B then 9B. Absolute rates are
0.42×, 0.35× and 0.43× the S26 Vulkan numbers. The S26 Mojo/Prism decode
advantage (about 1.3×) does not appear. On this GPU the portable 2B bench is
0.98× the Prism rate.

## Serial prefill: `pp512`

Same batch-one protocol. Mean ± sample standard deviation, tok/s.

| Model | This phone, Prism Vulkan | S26 Prism Vulkan | This phone, Mojo |
| --- | ---: | ---: | ---: |
| Bonsai 2 2B | 20.66 ± 1.32 | not in the qualified Vulkan table | 20.13 ± 0.12 portable OpenCL |
| Bonsai 2 4B | 8.22 ± 0.55 | 12.73 ± 0.93 | not run |
| Bonsai 2 9B | no completed sample | 6.98 ± 0.22 | not run |

The 4B Prism samples fell from 8.84 to 7.77 tok/s as the battery temperature
rose from the mid-30s °C to 41°C. The 9B prefill process started, printed the
Xclipse device line, and produced no JSON. The USB device then disappeared
(`adb: device 'DEVICE_SERIAL' not found`) at 18% battery and 41.3°C. That run is
not a result.

## What did not run

The staged 2B Adreno bench (`./bench ./libOpenCLPlugin.so ./weights.bin 0 8 0 1 1`)
allocates the model and then fails:

```text
HALError(code=-65537): failed to load bundle: OpenCL device lacks required extension: cl_khr_subgroup_shuffle
```

Xclipse 940 reports a warp size of 64 and no matrix cores. The S26 4B and 9B
Mojo binaries are the Adreno 840 builds (`--target-accelerator adreno-840`)
and were not launched. There is no portable 4B or 9B bench in the staged
outputs. The 2B number above is the existing portable OpenCL bench from the
Mali experiment, which removes subgroup shuffles. It is not the S26 Adreno
model object. Its current bench hash is
`7ef1e860b61d55cab500ecca4004c99546fd4c562e31e0945bfa3a6be67ef5f3`, which also
differs from the 2B bench hash recorded for the S26 run. Weights match the
recorded 2B file `ac588b263b9d12cef29595efde850e771138990e74e026d654d505f173ecd657`.
No logit fixture was repeated on this phone.
