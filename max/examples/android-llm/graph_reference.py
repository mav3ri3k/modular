#!/usr/bin/env python3
# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
"""Generate independent Hugging Face prefill/decode fixtures for export_graph.py."""

import argparse
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.cache_utils import Cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence", type=int, default=2)
    parser.add_argument("--capacity", type=int, default=16)
    parser.add_argument("--layers", type=int, default=30)
    args = parser.parse_args()
    config = AutoConfig.from_pretrained(args.checkpoint, local_files_only=True)
    if not 1 <= args.layers <= config.num_hidden_layers:
        parser.error("Invalid layer count")
    if not 1 <= args.sequence < args.capacity:
        parser.error("Prefill and decode must fit capacity")
    config.num_hidden_layers = args.layers
    config._attn_implementation = "eager"
    torch.set_num_threads(4)
    model = AutoModelForCausalLM.from_config(config).float().eval()
    missing, unexpected = model.load_state_dict(
        load_file(args.checkpoint / "model.safetensors"), strict=False
    )
    if set(missing) - {"lm_head.weight"}:
        raise ValueError(f"Missing weights: {missing}")
    if any(not name.startswith("model.layers.") for name in unexpected):
        raise ValueError(f"Unexpected weights: {unexpected}")
    model.tie_weights()
    shape = (
        args.layers,
        2,
        args.capacity,
        config.num_key_value_heads,
        config.hidden_size // config.num_attention_heads,
    )

    def cache_array(cache: Cache) -> np.ndarray:
        result = np.zeros(shape, dtype="<f4")
        for i, layer in enumerate(cache.layers):
            k, v = layer.keys, layer.values
            length = k.shape[2]
            result[i, 0, :length] = k[0].transpose(0, 1).numpy()
            result[i, 1, :length] = v[0].transpose(0, 1).numpy()
        return result

    ids = torch.arange(1, args.sequence + 1).reshape(1, -1)
    with torch.no_grad():
        prefill = model(ids, use_cache=True)
        cache_before_decode = cache_array(prefill.past_key_values)
        next_id = prefill.logits[:, -1].argmax(-1).reshape(1, 1)
        decode = model(
            next_id, past_key_values=prefill.past_key_values, use_cache=True
        )
        for name, inputs, positions, past, outputs in [
            (
                "prefill",
                ids.numpy(),
                np.arange(args.sequence, dtype="<i8"),
                np.zeros(shape, dtype="<f4"),
                (prefill.logits[:, -1].numpy(), cache_before_decode),
            ),
            (
                "decode",
                next_id.numpy(),
                np.array([args.sequence], dtype="<i8"),
                cache_before_decode,
                (
                    decode.logits[:, -1].numpy(),
                    cache_array(decode.past_key_values),
                ),
            ),
        ]:
            directory = args.output / name
            directory.mkdir(parents=True, exist_ok=True)
            inputs.astype("<i8").tofile(directory / "input0")
            positions.tofile(directory / "input1")
            past.tofile(directory / "input2")
            for i, output in enumerate(outputs):
                output.astype("<f4").tofile(directory / f"expected{i}")
            print(f"{name}: greedy ID {int(outputs[0].argmax())}")


if __name__ == "__main__":
    main()
