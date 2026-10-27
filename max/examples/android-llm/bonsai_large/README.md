# Bonsai 2 4B and 9B Android text ports

These experimental fixed-dimension ports reuse the 2B packed projections,
DeltaNet and attention operators. The generator specializes the 2B source to
the larger geometries. The 9B checkpoint additionally requires a block-4096
Hadamard transform and an untied output head. It uses a second transform pass
across eight 512-element blocks. Both ports use FP32 recurrent/KV state and
serial prefill. The generation and benchmark paths do not implement vision.

Use the compiler, Android NDK, compatible precompiled Mojo packages and
staged Android runtime described in the parent Android example and the
inference-test README. These files require the pinned MLX checkpoints:

- `prism-ml/Ternary-Bonsai-2-4B-mlx-2bit`, revision `dff2caaa23fa8706164c44c10f1b58feb405f76a`.
- `prism-ml/Ternary-Bonsai-2-9B-mlx-2bit`, revision `f4e28a59531731c41b6dfac88cf271bd58f0a287`.

For each size, export, generate, then build to a directory outside the repo:

```sh
python max/examples/android-llm/bonsai_large/prepare.py \
  --checkpoint /path/to/checkpoint --output /path/to/output --capacity 512 --tokens 32
python max/examples/android-llm/bonsai_large/generate.py --size 4b --output /path/to/output
python max/examples/android-llm/bonsai_large/build.py \
  --mojo /path/to/mojo --packages /path/to/packages --env-json /path/to/env.json \
  --ndk /path/to/ndk --runtime-build /path/to/runtime --output /path/to/output
```

Use `--size 9b` with the 9B checkpoint. The exporter accepts only the exact
supported geometries and block sizes. It produces packed weights, reference
logits and a prompt. The builder produces `runner`, `validate` and `bench`.
Do not enable the 2B-only Hexagon output-head environment option for these
ports. Stage the output and run `runner` with its prompt and reference to check
all logits and generated IDs before benchmarking. `check_weights.py` verifies
against publisher PTQ1_0 or PQ2_0 GGUF, including the reference's recurrent
head permutation; it requires the pinned Prism checkout's `gguf-py` and NumPy.

For benchmarking, stage the output directories as `bonsai4` and `bonsai9`
under the inference-test root and use the existing `run_suite.py --models
bonsai4 bonsai9`. Validate the reference backend on each model first; the
Vulkan workaround used for 2B must not be assumed correct for other models.
