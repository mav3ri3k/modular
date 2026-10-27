# Bonsai through the MAX graph runtime

The full text-only 24-layer Bonsai 2 2B model runs on the Adreno 840 through an
ordinary MAX graph, graph compiler, Android MEF and MAX C API. The C++ runner
loads weights and performs greedy selection; it never dispatches kernels.
The phone runs no Python, MLX or alternative inference engine.

## Current compact-scale improvement

Final selected-source medians: **56.43tok/s MAX /59.11 Mojo**, three alternating
95-forward runs after cooling, every96 IDs checked. Separate full-logit runs
pass. The controlled comparisons below isolate the weight change.

The shared kernels now read one exact FP16 scale per group. Both runners verify
scale roundtrips during loading, then discard the redundant centered bias.
MAX imports packed half bits as uint16 storage (ARM CPU graph ops reject f16),
hoists the normal transfers, and resolves Float16 views only inside GPU closures.
No extra conversion kernel or intermediate CPU tensor round trip is introduced.

Three paired95-forward sweeps: original MAX49.23→compact53.11tok/s (+7.9%);
Mojo53.25→58.07 (+9.1%). All96 IDs and full-logit checks pass. These gains precede
the subsequent combined CPU greedy scan; its stage timing improves, but later
thermal variability prevents a separate reliable wall-speed claim. The app is
unchanged. [Evidence](results-hillclimb-2026-10-25.json).

Normal side-stream setup now includes OpenCL. Creating extra queues and larger
host submission batches did not produce a repeatable model speedup. The final
trace shows about.9ms internal queue gaps in both paths; remaining wall overhead
needs host profiling, not an assumption of per-kernel synchronization.

## Initial comparison — 2026-10-25

Three interleaved repeats after one warmup per variant, same checkpoint, prompt,
24 forced output tokens / 23 decode forwards, capacity128, full-vocabulary logit
readback and greedy selection:

| Protocol | Mojo DeviceContext runner | MAX graph |
| --- | ---: | ---: |
| Generation only, every token ID checked | 53.74 tok/s | 49.90 tok/s |
| Every FP32-reference logit and ID checked | 47.76 tok/s | 42.90 tok/s |

MAX is **7.1% slower** in the generation-only comparison, **10.2% slower** with
full-logit validation. It does not provide an automatic end-to-end speedup in
this first graph version. These are short bring-up measurements, not a sustained
performance ceiling or long-context quality test. All raw repeats and artifact
hashes are in [the result record](results-2026-10-25.json).

All 24 IDs match the independent FP32 MLX reference. MAX maximum absolute logit
error is **0.0197719**, relative L2 **0.000812919**; the Mojo runner has
0.0165186 / 0.000786457. Both pass the same existing .05 / .005 tolerances.
Every output logit is finite. Recurrent/conv/KV state remains on the GPU across
calls and is checked indirectly by every subsequent reference-matching step.

A separate paired GPU trace shows real graph optimization:

| Decode metric | Mojo runner | MAX graph |
| --- | ---: | ---: |
| Dispatches/token | 721 | 599 |
| Summed GPU execution | 16.45 ms | 16.50 ms |
| Span covering GPU work | 17.35 ms | 19.89 ms |

MAX fuses residuals with RMSNorm and pointwise work with reductions, eliminating
122 dispatches/token (16.9%). The matrix kernels are shared. Extra gaps between
GPU work absorb that benefit. This is consistent with graph submission,
scheduling or synchronization overhead; the trace does not identify a specific
runtime call as the cause. Profiling affects wall throughput and those traced
wall rates are not used in the unprofiled table. The next useful investigation
is these gaps, before replacing the proven matrix arithmetic again.

## What uses which extension point

- Ordinary MAX ops: RMSNorm, residual add, SiLU/multiply and FP32 partial-product
  reduction. These remain visible to the structural fusion/compiler machinery.
- Registered Mojo custom ops: packed embedding/projections, signed Hadamard
  transforms, DeltaNet/conv updates, gated/head normalization, RoPE and the
  current cache/attention kernels. They reuse the same Bonsai kernel source.
  There is no whole-model custom forward op.
- Recurrent and KV state: explicit mutable `BufferType` graph inputs and
  `ops.inplace_custom` effects. No global model state or hidden allocator.
- Weights: host-backed `constant_external` values with explicit GPU transfers,
  hoisted by MAX to initialization. A GPU external constant promises an already
  resident device allocation; passing the C registry's host pointers under that
  declaration is invalid. The opaque-view encoder correctly rejects it.
- Transport: standard tensor engines preserve opaque storage on the host;
  custom closures resolve tile pointers only on the device after encoding.

This is a single-token graph; prefill loops over tokens, rather than using a
batched prefill kernel. The native C API runtime is the canonical source build,
including its compiler dependencies. This bring-up does not establish a compact
APK package or replace the current chat app's engine.

## Build and reproduce

Use a matching local native tool snapshot (`mojo`, `mo-opt`, `mt`, precompiled
packages and optionally `env.json`), as described in the
[integration ledger](../../../../docs/android-feasibility/INTEGRATION.md).
The installed Python MAX frontend has a different Mojo package checksum from
the local compiler. `build.py` compiles the **same source** with both compilers:
Python verifies ops using its matching package; the native pipeline imports and
verifies its own package. No compatibility checks or custom-op verification are
bypassed. Generated package staging is outside the source tree.

On the development machine, the existing tool snapshot/reference can be reused:

```bash
integration_cache="$HOME/Library/Caches/modular-android-integration"
"$integration_cache/venv/bin/python" max/examples/android-llm/bonsai_graph/build.py \
  --tools "$integration_cache/native-graph" \
  --analysis-mojo "$integration_cache/venv/bin/mojo" \
  --model "$integration_cache/bonsai-reference-short" \
  --ndk "$HOME/Library/Android/sdk/ndk/29.0.14206865" \
  --output /tmp/bonsai-max-build \
  --bazel-arg=--jobs=6 \
  --bazel-arg=--override_repository=sysroot-macos=/tmp/modular-macos26-sdk

"$integration_cache/venv/bin/python" max/examples/android-llm/bonsai_graph/run.py \
  --build /tmp/bonsai-max-build \
  --model "$integration_cache/bonsai-reference-short" \
  --compare-mojo "$integration_cache/bonsai-final" \
  --serial DEVICE_SERIAL --adb "$HOME/Library/Android/sdk/platform-tools/adb" \
  --repeats 3 --output /tmp/bonsai-max-comparison.json
```

Omit the machine-specific SDK override on hosts that do not need it. Add
`--generation-only` for ID-checked timing without reference-logit checking.
`--profile` runs one traced sample per variant, with no warmup/repeated timing
protocol. Reports retain all runs, artifact hashes and individual logs. The
runner cleans its unique temporary device directory and changes no phone power
or thermal settings.
