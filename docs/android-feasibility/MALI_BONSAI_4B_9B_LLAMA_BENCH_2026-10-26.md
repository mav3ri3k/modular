# Mali tg128 comparison: Bonsai 2 4B and 9B

Ran the official Prism `llama-bench` and the portable Mojo/OpenCL llama-bench
compatible harness on the OnePlus CPH2717 (MT6897, Mali-G615 MC6 r1p3). The
Prism build is commit `c96baf4abe24d11a50f326e40b6223b687449d63` (build
10827). Benchmarks used `n_prompt=0`, `n_gen=128`, `n_depth=0`, batch and
ubatch 1, four CPU threads, 99 GPU layers, and three repetitions. Prism used
the official PTQ1_0 GGUFs; their hashes match the pinned artifacts in the
[larger model provenance](inference-testing-bonsai-large/provenance.json).
Portable runs explicitly set `MODULAR_OPENCL_FLUSH_AFTER_ENQUEUE=1` and used
the same diagnostic OpenCL plugin as the previous Mali tests.

## Default Prism Vulkan versus portable OpenCL

| Model | Prism Vulkan mean ± SD | Prism samples | Portable OpenCL mean ± SD (latest run) | Portable samples |
| --- | ---: | --- | ---: | --- |
| 4B | 3.014 ± 0.027 tok/s | 3.035, 3.024, 2.983 | 6.876 ± 0.079 tok/s | 6.784, 6.920, 6.923 |
| 9B | 0.760 ± 0.021 tok/s | 0.745, 0.750, 0.783 | 4.680 ± 1.390 tok/s | 5.466, 5.498, 3.076 |

These are single three-repetition commands per implementation. Prism ran first
for each model; portable used seed 1. There was no thermal throttling reported
by Android during the runs. The 9B portable result has high sample variation,
so its mean should be treated as provisional.

## Repeatability and Vulkan flag control

The portable three-repetition means varied substantially across otherwise
identical commands, despite the same device, model files, plugin, and flush
environment setting:

| Model | Portable means from all runs (tok/s) |
| --- | --- |
| 4B | 6.462 ± 0.823 (seed 20261026); 3.242 ± 0.004 (seed 1); 3.548 ± 0.509 (seed 1); 6.876 ± 0.079 (seed 1, latest) |
| 9B | 5.160 ± 0.217 (seed 20261026); 2.665 ± 0.008 (seed 1); 4.680 ± 1.390 (seed 1, latest) |

I have not identified the cause of these shifts. The latest samples are
recorded above, but the portable-versus-Prism speed ratios need a more
controlled repeat before treating them as stable.

The larger-model suite had also used `GGML_VK_DISABLE_ASYNC=1` and
`GGML_VK_DISABLE_FUSION=1` on another device. On this Mali phone, those flags
changed Prism performance, especially for 9B:

| Model | Prism Vulkan with both flags (mean ± SD) | Samples (tok/s) |
| --- | ---: | --- |
| 4B | 2.898 ± 0.080 | 2.960, 2.808, 2.926 |
| 9B | 1.357 ± 0.330 | 1.055, 1.710, 1.308 |

The main comparison above uses Prism's default Vulkan settings to match the
2B Mali run. The flag-control runs are separate measurements, not combined
with the default results.

## Commands

Prism default Vulkan command (change model name for 9B):

```sh
cd /data/local/tmp/modular-bonsai-mali/prism-vulkan
LD_LIBRARY_PATH=. ./llama-bench \
  -m ../Ternary-Bonsai-2-4B-PTQ1_0.gguf \
  -p 0 -n 128 -d 0 -r 3 -t 4 -ngl 99 -o json -b 1 -ub 1
```

Portable OpenCL command (change directory and executable for 9B):

```sh
cd /data/local/tmp/modular-bonsai-mali/portable4
LD_LIBRARY_PATH=. MODULAR_OPENCL_FLUSH_AFTER_ENQUEUE=1 \
  /data/local/tests/vendor/bonsai-portable4-bench \
  ./OpenCLPlugin.so ./weights.bin 0 128 0 3 1
```

The phone retains both official GGUFs and portable exports. About 5.3 GB was
free after staging and running the benchmarks.
