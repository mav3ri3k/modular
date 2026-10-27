# Bonsai 2 4B and 9B on the OnePlus Mali phone

Both portable OpenCL builds loaded and ran on the OnePlus CPH2717 (MT6897,
Mali-G615 MC6 r1p3). Every model invocation used
`MODULAR_OPENCL_FLUSH_AFTER_ENQUEUE=1` with the current diagnostic OpenCL
plugin (`sha256 3932cdec258d161bac574af81c8d0d0d777a39b0e586d1682c04fec69c05f771`).

## Results

| Model | Device allocation | Fixture check | tg128 mean decode |
| --- | ---: | --- | ---: |
| Bonsai 2 4B | 1,169.55 MiB | 4/4 greedy IDs; max abs logit error 0.0112495; relative L2 0.00051623 | 6.462 tok/s (sample SD 0.823; 3 repetitions) |
| Bonsai 2 9B | 2,382.42 MiB | 4/4 greedy IDs; max abs logit error 0.00991082; relative L2 0.000413623 | 5.160 tok/s (sample SD 0.217; 3 repetitions) |

Both fixture runs generated token IDs `760, 6511, 314, 9338`. The fixture
contains four reference positions, so these results establish short-run
numerical agreement, not the 32-token portable validation previously done for
2B. The bench is the portable llama-bench-compatible harness with
`n_prompt=0`, `n_gen=128`, `n_depth=0`, `n_batch=1`, `n_ubatch=1`, seed
`20261026`, and three repetitions. Exact samples:

- 4B: 23.224, 18.432, and 18.469 seconds.
- 9B: 26.067, 24.287, and 24.151 seconds.

This records portable Mojo/OpenCL throughput on the Mali device. No Prism
Vulkan comparison was run for 4B or 9B in this extension.

## Reproduction

The runners and weights were staged under
`/data/local/tmp/modular-bonsai-mali/portable4` and `portable9`. The runner was
copied into `/data/local/tests/vendor` for access to the phone's vendor OpenCL
library namespace. For each model, the commands were:

```sh
LD_LIBRARY_PATH=. MODULAR_OPENCL_FLUSH_AFTER_ENQUEUE=1 \
  /data/local/tests/vendor/bonsai-portable4-runner \
  ./OpenCLPlugin.so ./weights.bin ./prompt.txt ./reference.f32 4
LD_LIBRARY_PATH=. MODULAR_OPENCL_FLUSH_AFTER_ENQUEUE=1 \
  /data/local/tests/vendor/bonsai-portable4-bench \
  ./OpenCLPlugin.so ./weights.bin 0 128 0 3 20261026
```

Replace `4` with `9` in the vendor runner names and use the `portable9`
directory for 9B.

The corresponding model objects are 4B `b989142d109c2ac30825e50ddd4c320b0b0917d4064c516b6bcb9b6df1f90a28`
and 9B `271d5b058cd8ab4ea070ca1d0fc73b79b75a100f3d8ee1005da3363ebb85702f`.
The exported weight hashes are 4B
`1c33e9b3521f14025c3c6dc47a40900a23665efaa381ad1a62461777c75e86de` and 9B
`f989b6bd549be758310855ed2f710dc44251159a9645c97aa5b757f933ba1c8d`.

The phone retains both staging directories and has about 5.9 GB free after the
runs. The existing 4B inputs were restored after copying 9B, keeping both models ready
to rerun; its runner binaries also remain installed in the vendor test directory.
