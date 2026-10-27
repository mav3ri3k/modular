# Galaxy S24 FE portable inference rebuild — October 27, 2026

Fresh portable Mojo/OpenCL builds previously prepared for the Mali rerun were
run on a Samsung Galaxy S24 FE SM-S721B, Exynos s5e9945 / Xclipse 940, Android
16 / API 36. The [provenance](provenance.json) records the firmware fingerprint,
source and binary hashes, and model-weight hashes. The source worktree was
not clean. The 2B weights were already staged on the phone; 4B and 9B were
transferred from the existing host exports. Their device-side SHA-256 hashes
match the archived exports. This is not a clean-download/export walkthrough.

The rebuilt plugin SHA-256 is
`5a67266344ef0096f07387958f0013a21389f4d1a63bdf225fc9893505447d6c`.
The Mali-specific `MODULAR_OPENCL_FLUSH_AFTER_ENQUEUE` option was not enabled.
The 2B model has context capacity 2048 and the 4B/9B models capacity 512.
The phone loads OpenCL from executables in `/data/local/tmp`; it does not need
the vendor-namespace workaround used by the tested OnePlus image.

## Numerical checks

Each check reads every vocabulary logit on each generated step and requires
matching greedy token IDs. The 2B runner passed 24 generated-token steps
(maximum absolute logit error `0.0210514`, relative L2 `0.000756475`). 4B and
9B each passed four generated-token steps: maximum absolute errors `0.0103179`
and `0.00964212`, and relative L2 errors `0.000509901` and `0.000433442`,
respectively. These are short fixtures and do not establish broader model
quality or long-context accuracy. Validation logs are included in this folder.

## Results

Three separate processes per model and workload, one repetition per process,
seed 1, a fixed ten-second pause before each process, one-token decode warmup
or full 512-token prefill warmup. `tg128` runs 128 decode forwards; `pp512`
is serial batch-one prefill. Both include full-vocabulary host transfers and
exclude model loading, GPU compilation, tokenization, and sampling. Values are
arithmetic mean ± sample standard deviation of per-process token rates. Every
sample is retained.

| Model | Decode `tg128`, tok/s | Serial prefill `pp512`, tok/s |
| --- | ---: | ---: |
| Bonsai 2 2B | 20.37 ± 0.62 | 19.98 ± 0.12 |
| Bonsai 2 4B | 11.79 ± 0.04 | 11.39 ± 0.16 |
| Bonsai 2 9B | 7.59 ± 0.11 | 6.84 ± 0.87 |

The 2B results are close to the [earlier portable 2B measurements](../INFERENCE_TESTING_S24FE_BONSAI_2026-10-26.md)
of 21.17 ± 0.52 decode and 20.13 ± 0.12 prefill tok/s, although the builds and
measurement sequences differ. These are the first 4B and 9B portable Mojo
results reported for this Exynos phone. Prism was not rerun and no paired
speedup is claimed.

The workloads ran sequentially, decode first and then prefill in 2B, 4B, 9B
order. There was no thermal or clock control. The 9B prefill samples were
7.47, 7.21 and 5.85 tok/s. Battery temperature rose from 39.2°C before the
first to 41.6°C after the last; Android's thermal status rose from 0 to 2.
The fall in the last sample coincided with this warming, but the measurements
do not isolate its cause. The [summary](summary.json), [sample records](samples.json),
raw stdout/stderr, and thermal snapshots contain all measurements.

The [root README](../../../README.md#samsung-exynos--xclipse-portable-build)
contains the independent build, export, validation and run instructions.
