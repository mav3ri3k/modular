# OnePlus Mali portable rebuild, October 26–27, 2026

A fresh portable Mojo model, runner and benchmark were built from the current
working tree for each size and run on OnePlus CPH2717, MT6897, Mali-G615 MC6
r1p3, Android 16 / API 36. The [provenance](provenance.json) records the device
fingerprint and source, compiler, model-object, plugin and binary hashes. The
working tree was not clean. Previously exported weights and reference fixtures
were reused from the phone, rather than downloaded and re-exported for this run;
the device-side weight hashes match the archived Mali 2B and 4B/9B exports.

All models used `MODULAR_OPENCL_FLUSH_AFTER_ENQUEUE=1` with the newly staged
plugin SHA-256 `5a67266344ef0096f07387958f0013a21389f4d1a63bdf225fc9893505447d6c`.
The earlier flush-on experiments used plugin hash
`3932cdec258d161bac574af81c8d0d0d777a39b0e586d1682c04fec69c05f771`;
the rebuilt model objects and runners also differ from the historical ones.
These are therefore new measurements of a rebuilt implementation, not a
bit-for-bit replay of the old binaries.

## Validation

The 2B runner passed the eight-output-token all-vocabulary reference fixture
(relative L2 `0.000849557`, maximum absolute error `0.0194926`). The 4B and 9B
runners passed their four-output-token all-vocabulary fixtures (relative L2
`0.00051623` and `0.000413623` respectively). These are short fixture checks;
they do not establish broad model quality or long-context correctness. Full
logs are `2-validate.log`, `4-validate.log`, and `9-validate.log`.

## Decode

`tg128`, three repetitions in one process with one warmup, seed 1, full-logit
host transfer and synchronization for every forward. Rates are the arithmetic
mean ± sample standard deviation of the individual sample rates. Loading,
compilation, tokenization and sampling are excluded.

| Model | Rebuilt Mojo, tok/s | Samples, tok/s | Previously reported flush-on Mojo, tok/s |
| --- | ---: | --- | ---: |
| 2B | 8.67 ± 1.69 | 6.72, 9.58, 9.72 | 10.07 ± 0.01 |
| 4B | 6.88 ± 0.04 | 6.90, 6.92, 6.84 | 6.88 ± 0.08 in the latest prior run |
| 9B | 4.98 ± 0.55 | 5.14, 4.37, 5.44 | 4.68 ± 1.39 in the latest prior run |

The 2B mean is lower, with a slow first sample; no sample was excluded. The 4B
and 9B rates are close to the selected prior runs, though the previous 4B/9B
studies also observed substantial run-to-run variation. The three models were
run sequentially, with a fixed ten-second pause between model processes, without
thermal or clock control. Android reported thermal status 0 in the captured
before/after snapshots, while battery temperature rose from 29.0°C to 30.6°C.
This does not prove fixed GPU clocks or identify the cause of any variation.

The [summary](summary.json) retains integer durations and rates; stdout, stderr,
validation and thermal snapshots are stored alongside it. Prism was not
rebenchmarked in this run. The setup and commands for an independent run are in
the [root README](../../../README.md#oneplus-mali-build-and-reproduce).
