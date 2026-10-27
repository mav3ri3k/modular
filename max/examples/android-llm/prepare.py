#!/usr/bin/env python3
# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the Apache License v2.0 with LLVM Exceptions: https://llvm.org/LICENSE.txt
"""Export unmodified BF16 SmolLM2 weights and FP32 CPU reference logits.

Requires torch, transformers, safetensors, and huggingface_hub. All model
artifacts go to the explicitly selected output directory, outside the repository.
"""

import argparse
import hashlib
import json
import struct
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
# Pin the checkpoint; record this and the downloaded weight hash in the manifest.
REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--prompt", default="What is the capital of France?")
    p.add_argument("--tokens", type=int, default=16)
    p.add_argument("--capacity", type=int, default=128)
    args = p.parse_args()
    torch.set_num_threads(4)
    source = args.checkpoint or Path(
        snapshot_download(
            MODEL,
            revision=REVISION,
            allow_patterns=["*.json", "*.safetensors", "*.txt"],
        )
    )
    local_metadata = source / ".cache/huggingface/download/config.json.metadata"
    revision = REVISION if args.checkpoint is None else "local"
    if args.checkpoint is not None and local_metadata.exists():
        revision = local_metadata.read_text().splitlines()[0]
    config = json.loads((source / "config.json").read_text())
    expected = dict(
        hidden_size=576,
        intermediate_size=1536,
        num_hidden_layers=30,
        num_attention_heads=9,
        num_key_value_heads=3,
        vocab_size=49152,
        rope_theta=100000,
        rms_norm_eps=1e-5,
        tie_word_embeddings=True,
    )
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Unsupported {key}: {config.get(key)} != {value}")
    if not 1 <= args.capacity <= 8192 or args.tokens < 1:
        raise ValueError("Invalid capacity or token count")
    tokenizer = AutoTokenizer.from_pretrained(source)
    text = tokenizer.apply_chat_template(
        [dict(role="user", content=args.prompt)],
        tokenize=False,
        add_generation_prompt=True,
    )
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) + args.tokens > args.capacity:
        raise ValueError("Prompt plus generation exceeds context capacity")
    args.output.mkdir(parents=True, exist_ok=True)
    tensors = load_file(source / "model.safetensors")
    embed = tensors["model.embed_tokens.weight"]
    weights = [embed, embed.T.contiguous(), tensors["model.norm.weight"]]
    for layer in range(30):
        prefix = f"model.layers.{layer}."
        for suffix in [
            "input_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "post_attention_layernorm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ]:
            w = tensors[prefix + suffix]
            weights.append(w.T.contiguous() if w.ndim == 2 else w)
    inverse = 1 / (100000 ** (torch.arange(0, 64, 2).float() / 64))
    angles = torch.arange(args.capacity).float()[:, None] * inverse[None, :]
    rotary = torch.cat([angles.cos(), angles.sin()], dim=1).numpy().tobytes()
    with (args.output / "weights.bin").open("wb") as f:
        f.write(
            b"SMOLBF01" + struct.pack("<II", len(weights) + 1, args.capacity)
        )
        for w in weights:
            raw = (
                w.to(torch.bfloat16)
                .contiguous()
                .view(torch.uint16)
                .numpy()
                .tobytes()
            )
            f.write(struct.pack("<Q", len(raw)))
            f.write(raw)
        f.write(struct.pack("<Q", len(rotary)))
        f.write(rotary)
    del weights, tensors
    model = AutoModelForCausalLM.from_pretrained(
        source, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    generated = []
    cache = None
    inputs = torch.tensor([ids])
    # Store logits for the prompt and every generated token. The runner compares
    # every step and stops on a token mismatch rather than hiding divergence.
    with (
        torch.inference_mode(),
        (args.output / "reference.f32").open("wb") as f,
    ):
        for _ in range(args.tokens):
            result = model(inputs, past_key_values=cache, use_cache=True)
            logits = result.logits[0, -1].float().numpy()
            f.write(logits.astype("<f4").tobytes())
            token = int(logits.argmax())
            generated.append(token)
            cache = result.past_key_values
            inputs = torch.tensor([[token]])
    (args.output / "prompt.txt").write_text(" ".join(map(str, ids)) + "\n")
    manifest = dict(
        model=MODEL,
        revision=revision,
        config=expected,
        checkpoint_sha256=hashlib.file_digest(
            (source / "model.safetensors").open("rb"), "sha256"
        ).hexdigest(),
        prompt=args.prompt,
        prompt_ids=ids,
        expected_ids=generated,
        expected_text=tokenizer.decode(generated),
        capacity=args.capacity,
        generation_tokens=args.tokens,
        weights_dtype="bfloat16",
        activations_dtype="float32",
    )
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
