# Batch-one Bonsai decode: HVX experiments

2026-10-25, Galaxy S26 SM-S942U1 / SM8850 / Adreno 840 / Hexagon v81,
serial DEVICE_SERIAL, Android API 37. This pass implements an opt-in packed-W2
HVX vocabulary backend and corrects GPU/DSP overlap. It does not establish a
substantial, repeatable NPU advantage. GPU-only remains the default.

## Matched final-source comparison

The source-built direct runner uses the same pinned checkpoint, 24-token prompt,
capacity 128 and 96 requested output tokens (95 decode forwards). One warmup per
variant, five measured rounds in alternating order, eight-second idle gaps.
Every output ID is checked against the independent FP32 reference. Decode
includes output readback and CPU greedy selection; initialization, compilation,
weight upload, prefill and CPU full-logit verification are excluded. Profiling is
disabled. Every reported Android thermal status is zero; no global clock or
thermal-policy settings are changed. Hexagon uses process-local power votes and
an optional FastRPC PM QoS vote, removed when the session closes.

| Variant | Five measured samples, tok/s | Median |
| --- | --- | ---: |
| Starting GPU build | 71.89 / 71.89 / 71.18 / 68.09 / 68.65 | 71.18 |
| GPU with cooperative normalization | 74.92 / 75.04 / 71.31 / 68.13 / 70.46 | 71.31 |
| Same GPU + 16,384 vocabulary rows on HVX | 74.51 / 74.79 / 70.17 / 71.77 / 70.96 | 71.77 |

The hybrid median is 0.8% above the starting build and 0.6% above the updated
GPU build. Phone operating-rate drift is much larger than this difference.
These are short repeated decode measurements, not thermal steady-state ceilings.
The 75.60 tok/s GPU warmup is not a measured median. No 100 tok/s result.

An earlier three-round comparison measured 67.58 starting GPU, 69.98 updated GPU,
and 71.05 hybrid medians. A separate row sweep measured 67.98 GPU-only versus
68.41 / 68.72 / 68.02 for 8,192 / 16,384 / 24,576 HVX rows. Retain both batches:
the small offload benefit does not consistently reproduce. The final-source
five-round comparison is the primary end-to-end result.

A final control batch holds the NPU transport/power path present while offloading
only 128 rows. Three measured alternating rounds give **74.55 GPU-only, 74.30
128-row control, and 73.87 tok/s for 12,288 HVX rows**. This does not reproduce
the preceding batch's 12,288-row advantage. No single batch supports changing
the default. See [control samples](power-control.json).

[Final comparison and artifact hashes](final-comparison.json),
[row sweep](row-sweep.json), [normalization selection](normalization-selection.json).

## What changed and what profiling found

- **HVX GEMV instead of HMX-expanded matrix tiles.** Resident weights remain W2,
  reordered once into blocks of 128 outputs. A per-token activation lookup table
  maps two packed codes to FP16 pair sums. Vector lookups and QFloat16
  accumulation compute each group; FP32 scales and group joins produce logits.
  No repeated FP16 expansion of the weight matrix or 32 identical activation
  rows. Existing HMX code remains selectable with `--backend hmx`.
- **Correct overlap through ordinary Mojo APIs.** Queue-tail activation readback
  in the recent DeviceContext integration waited for the GPU head before
  starting Hexagon. The new Bonsai entry point copies the transformed activation,
  records an event, submits the GPU head and waits only for that copy event.
  C++ can then execute Hexagon while the GPU head runs. Waiting on the whole
  context was substantially slower and was rejected.
- **Cooperative normalization.** `normalize_qk`, `gated_norm` and `head_norm`
  repeated serial 128/256-element sums for every output. The selected kernels
  split each sum across the existing 64-lane subgroup and reduce with shuffles.
  This changes FP32 reduction order, not the checkpoint or projection format.
  The ordinary MAX graph reuses these same kernels.

Separate event profiles, 23 decode forwards each; no dropped/failed samples:

| GPU family | Starting GPU, ms/token | Updated GPU | Updated GPU + HVX |
| --- | ---: | ---: | ---: |
| NormalizeQk | 0.239 | 0.047 | 0.048 |
| GatedNorm | 0.260 | 0.054 | 0.054 |
| HeadNorm | 0.296 | 0.030 | 0.030 |
| Packed layer projections | 6.791 | 6.808 | 6.797 |
| Vocabulary head | 2.212 | 2.227 | 2.280 |
| All GPU kernels | 12.569 | 11.945 | 11.986 |

The normalization families save about 0.66 ms of GPU work per token. The hybrid
GPU head does not get faster despite fewer rows. This is consistent with shared
memory contention during DSP work, but these separate captures do not isolate
bandwidth from clocks or prove a contention mechanism. GPU-interface counters
from prior passes must not be treated as total CPU/GPU/DSP DRAM traffic.

Most work remains in layer projections (~6.8 ms/token). A 16,384-row split removes
only 6.6% of the vocabulary rows, giving roughly 0.15 ms of head-time opportunity
under linear scaling before transport or contention. Larger HVX slices extend
the critical path. This explains why another execution unit has not produced a
large decode gain in this partition.

[Profile attribution](final-profile-summary.json) and [all evidence](summary.json).

## Validation and rejected variants

GPU-only and direct HVX hybrid both pass all 96 IDs and all reference-logit
checks on two prompts, including the longer numbers prompt at capacity 256.
The ordinary MAX graph also passes 96 IDs and full-logit checks on the first
prompt. Maximum absolute errors / relative L2:

| Route | First prompt | Longer prompt |
| --- | --- | --- |
| Updated GPU | 0.0222683 / 0.000634382 | 0.0201702 / 0.000395346 |
| GPU + HVX | 0.0222683 / 0.000642699 | 0.0201702 / 0.000402119 |
| MAX, GPU only | 0.0206356 / 0.000648266 | Not measured in this pass |

The 12,288-row direct-memory hybrid additionally passes both 96-token/full-logit
fixtures: relative L2 is 0.000640648 / 0.000400455, with the same maximum
absolute errors as the 16,384-row hybrid above.
[First prompt](validation-12288.json), [longer prompt](long-validation-12288.json).

The real 6,144×2,048 projector with half-exact synthetic activations matches every
CPU output exactly. Small fixtures do not establish general model quality or
long-context equivalence. HVX lookup sums use approximate FP16/QFloat arithmetic
and flush subnormal group sums, as documented in the backend.

Other experiments:

- IEEE HVX intrinsic variant returned zeros in this build. QFloat is the tested
  compatibility path; this is not a general claim that the silicon lacks IEEE
  support.
- Direct pair multiply/accumulate lost to lookup-based HVX GEMV. L2 prefetch
  improved the lookup path; 2/4-worker configurations and 256-output tiles did
  not justify replacing the 8-worker, 128-output implementation. The 256-output
  assembly also contained vector stack traffic.
- FastRPC polling QoS returned error 14. PM QoS with a 100 us latency request
  works. DSP profile mode makes an extra RPC, so its wall times are not throughput
  results.
- **VTCM residency was not reliable across idle.** The resource query reports
  8 MiB. Selected packed weights/scales fit, and short warm projection calls can
  run around 0.2–0.3 ms. However, vector integrity checks fail after a 2.5-second
  idle gap, including with process-local sleep-disable/latency votes. Some short
  full-model trials passed; others failed before generation. This variant is
  excluded. The exact cause of lost data/visibility is not established; do not
  assume an acquired resource guarantees retained contents across RPC idle.
- A refreshed VTCM staging variant passes CPU and full-model references after
  idle. The initial copy loop included a compiler-generated library alternative and was
  extremely slow
  (~55 ms/projection). Explicit vector assembly, prefetch and parallel staging
  reduced that to approximately 1–2 ms in short samples. End-to-end comparison
  and the final decision are recorded below; no VTCM experiment is in production
  sources.


A later same-row comparison (12,288 rows; three alternating measured rounds)
measured GPU-only **68.87**, direct-memory HVX **71.20**, and refreshed VTCM
**70.57 tok/s** medians. Each ID matches. Direct-memory samples were
70.37 / 71.63 / 71.20; VTCM samples were 70.22 / 71.11 / 70.57. The staging
variant is not retained: it adds resource, backup and copying complexity and
loses to direct-memory HVX. These values are a separate batch, not substitutes
for the five-round table above. See [VTCM comparison](vtcm-comparison.json).

Prototype sources, binaries and exploratory logs remain outside git in
`~/Library/Caches/modular-android-integration/hvx-work/`. No app was repackaged.
The working directory's pre-existing compiler, HAL and model changes are kept.

## Reproduce

Build the standard Bonsai runner and the Hexagon backend using
[the backend guide](../../../Target/lib/Hexagon/README.md); select `--backend hvx`.
Use a checkpoint export with independent reference logits and expected token IDs.
Then compare the same runner with and without offload:

```bash
python max/examples/android-llm/benchmark.py \
  --variant gpu=/path/to/bonsai-build \
  --variant hvx=/path/to/bonsai-build \
  --hexagon-build hvx=/path/to/hexagon-build --hexagon-rows hvx=16384 \
  --model /path/to/export --serial DEVICE_SERIAL --adb /path/to/adb \
  --repeats 5 --cooldown 8 --generation-only --output /tmp/hvx-comparison.json
```

Omit `--generation-only` for a separate full-logit check; do not mix those times
with generation-only timing. Reports retain all runs, artifact hashes, thermal
statuses and raw logs. An explicit Hexagon request fails on errors, rather than
silently falling back to GPU rows. The NPU route is currently the direct runner;
MAX graph execution remains on GPU. GPU-only is the recommended default for this
batch-one workload until a larger, repeatable benefit is demonstrated.
