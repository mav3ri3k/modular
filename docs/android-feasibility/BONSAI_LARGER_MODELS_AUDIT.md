# Bonsai 2 4B and 9B: initial port audit

This is the initial architecture audit. The subsequent ports and measurements
are in the [larger-model report](INFERENCE_TESTING_BONSAI_4B_9B_2026-10-26.md).
The existing 2B inference report remains unchanged.
Configuration values below come from
Prism's MLX checkpoints pinned at revisions
[`dff2caaa23fa8706164c44c10f1b58feb405f76a`](https://huggingface.co/prism-ml/Ternary-Bonsai-2-4B-mlx-2bit/blob/dff2caaa23fa8706164c44c10f1b58feb405f76a/config.json)
(4B) and
[`f4e28a59531731c41b6dfac88cf271bd58f0a287`](https://huggingface.co/prism-ml/Ternary-Bonsai-2-9B-mlx-2bit/blob/f4e28a59531731c41b6dfac88cf271bd58f0a287/config.json)
(9B). These are text-model properties; multimodal inference is out of scope.

| Property | 2B | 4B | 9B |
| --- | ---: | ---: | ---: |
| Hidden size | 2048 | 2560 | 4096 |
| Feed-forward size | 6144 | 9216 | 12288 |
| Layers | 24 | 32 | 32 |
| Attention query / KV heads | 8 / 2 | 16 / 4 | 16 / 4 |
| Recurrent key / value heads | 16 / 16 | 16 / 32 | 16 / 32 |
| Tied embeddings | yes | yes | no |
| Hadamard block size | 512 | 512 | 4096 |

All three retain 256-dimensional attention heads, 128-dimensional recurrent
heads, convolution width 4 and one full-attention layer per four layers.
The 4B and 9B recurrent value heads outnumber key heads by two, which requires
mapping each value head to its associated key and query head. Treating the
existing 2B delta-update indexing as dimension-independent would be incorrect.

The existing `bonsai/model.mojo`, `bonsai/kernels.mojo`, `model_spec.h` and
`bonsai/prepare.py` contain 2B-specific dimensions, state sizes and tensor
counts. In addition to generalized launch dimensions, the larger models need
larger recurrent state and attention cache allocations, gate projections with
32 value heads, and adjusted attention head/cache indexing. The 9B model needs
its separate output head exported and dispatched. The packed projection and
Hadamard approach can be reused for 4B. The 9B model also needs a block-4096
transform, implemented in the larger-model port as a second cross-block pass.

The pinned MLX checkpoints were inspected using
[`check_large_bonsai_contract.py`](../../max/examples/android-llm/inference_test/check_large_bonsai_contract.py).
All 201 packed tensors in 4B and 202 in 9B have biases equal to negative
scales. The signs also match within each gate/up and attention-input group
used by the 2B transform-sharing optimization. This checks a prerequisite
for reuse; it does not validate the forward pass or compare the GGUF weights.

For a comparable extension, check every exported tensor against the matching
publisher GGUF, validate full-vocabulary teacher-forced logits and state
updates, and independently qualify each native backend before reporting timings.
Use the existing llama-bench port's full-logit readback and the 2B report's
workload and thermal methodology. The previously qualified Vulkan flags are not
proof of correctness for either new model.
