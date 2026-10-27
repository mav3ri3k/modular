#!/usr/bin/env python3
# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the Apache License v2.0 with LLVM Exceptions: https://llvm.org/LICENSE.txt
"""Losslessly transpose GGUF blocks for Mojo coalesced matvecs (no requantization).

Requires gguf-py from the pinned reference llama.cpp checkout and numpy.
Qwen3.5 MTP/vision tensors are excluded, matching default llama-bench text use.
"""

import argparse
import hashlib
import json
import struct
from collections.abc import Callable
from pathlib import Path

import numpy as np
from gguf import GGML_QUANT_SIZES, GGUFReader
from gguf import GGMLQuantizationType as QT


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gguf", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--capacity", type=int, default=1024)
    a = p.parse_args()
    if not 1 <= a.capacity <= 8192:
        raise ValueError("capacity must be in [1, 8192]")
    a.output.mkdir(parents=True, exist_ok=True)
    r = GGUFReader(a.gguf)
    ts = {t.name: t for t in r.tensors}
    arch = r.fields["general.architecture"].contents()
    assert arch in ("qwen3", "qwen35")
    n = 28 if arch == "qwen3" else 24
    vocab = 151936 if arch == "qwen3" else 248320
    expected = {
        "embedding_length": 2048,
        "feed_forward_length": 6144,
        "block_count": 28 if arch == "qwen3" else 25,
        "attention.head_count": 16 if arch == "qwen3" else 8,
        "attention.head_count_kv": 8 if arch == "qwen3" else 2,
        "attention.key_length": 128 if arch == "qwen3" else 256,
    }
    for key, value in expected.items():
        if r.fields[arch + "." + key].contents() != value:
            raise ValueError("unsupported " + key)
    eos = int(r.fields["tokenizer.ggml.eos_token_id"].contents())
    arrays = []
    names = []

    def add(name: str, data: np.ndarray) -> None:
        data = np.ascontiguousarray(data)
        arrays.append(data)
        names.append(dict(name=name, bytes=data.nbytes))

    def matrix(name: str) -> None:
        t = ts[name]
        qt = t.tensor_type
        assert qt in (QT.Q4_K, QT.Q5_K, QT.Q6_K, QT.Q8_0, QT.F32)
        reduction, width = map(int, t.shape)
        block, size = GGML_QUANT_SIZES[qt]
        raw = (
            t.data.view(np.uint8)
            .reshape(width, reduction // block, size)
            .transpose(1, 2, 0)
            .copy()
        )
        header = struct.pack("<IIII", int(qt), block, size, width)
        add(
            name,
            np.concatenate(
                [np.frombuffer(header, dtype=np.uint8), raw.reshape(-1)]
            ),
        )

    def f32(
        name: str, transform: Callable[[np.ndarray], np.ndarray] = lambda x: x
    ) -> None:
        t = ts[name]
        assert t.tensor_type == QT.F32
        add(name, np.asarray(transform(t.data), dtype="<f4"))

    matrix("token_embd.weight")
    if arch == "qwen3":
        matrix("output.weight")
        f32("output_norm.weight")
    else:
        add("unused", np.zeros(1, dtype="<f4"))
        add("unused", np.zeros(1, dtype="<f4"))
        f32("output_norm.weight", lambda x: x)

    def packed(name: str) -> None:
        matrix(name)
        add("unused", np.zeros(1, dtype="<f4"))
        add("unused", np.zeros(1, dtype="<f4"))

    for i in range(n):
        pre = f"blk.{i}."
        if arch == "qwen3":
            for name, mat in [
                ("attn_norm.weight", False),
                ("attn_q.weight", True),
                ("attn_k.weight", True),
                ("attn_v.weight", True),
                ("attn_output.weight", True),
                ("ffn_norm.weight", False),
                ("ffn_gate.weight", True),
                ("ffn_up.weight", True),
                ("ffn_down.weight", True),
                ("attn_q_norm.weight", False),
                ("attn_k_norm.weight", False),
            ]:
                (matrix if mat else f32)(pre + name)
        else:
            f32(pre + "attn_norm.weight", lambda x: x)
            f32(pre + "post_attention_norm.weight", lambda x: x)
            for name in ["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"]:
                packed(pre + name)
            if (i + 1) % 4:
                for name in [
                    "attn_qkv.weight",
                    "attn_gate.weight",
                    "ssm_out.weight",
                ]:
                    packed(pre + name)
                for name in ["ssm_alpha.weight", "ssm_beta.weight"]:
                    f32(pre + name, lambda x: x.T)
                # Keep GGUF negative decay constants; the reused gate kernel is adapted to accept them.
                add(
                    pre + "decay_constants",
                    np.concatenate(
                        [ts[pre + "ssm_a"].data, ts[pre + "ssm_dt.bias"].data]
                    ).astype("<f4"),
                )
                f32(pre + "ssm_conv1d.weight", lambda x: x.T)
                f32(pre + "ssm_norm.weight")
            else:
                for name in [
                    "attn_q.weight",
                    "attn_k.weight",
                    "attn_v.weight",
                    "attn_output.weight",
                ]:
                    packed(pre + name)
                f32(pre + "attn_q_norm.weight", lambda x: x)
                f32(pre + "attn_k_norm.weight", lambda x: x)
    dim = 128 if arch == "qwen3" else 64
    base = 1e6 if arch == "qwen3" else 1e7
    angles = (
        np.arange(a.capacity, dtype=np.float32)[:, None]
        * (1 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim)))[
            None, :
        ]
    )
    add(
        "rope",
        np.concatenate([np.cos(angles), np.sin(angles)], axis=1).astype("<f4"),
    )
    with (a.output / "weights.bin").open("wb") as f:
        f.write(b"QWENRAW1" + struct.pack("<II", len(arrays), a.capacity))
        for v in arrays:
            f.write(struct.pack("<Q", v.nbytes))
            f.write(v.tobytes())
    manifest = dict(
        architecture=arch,
        source=str(a.gguf),
        source_sha256=hashlib.file_digest(
            a.gguf.open("rb"), "sha256"
        ).hexdigest(),
        capacity=a.capacity,
        vocabulary=vocab,
        tensors=names,
        format="raw GGUF blocks, byte-plane transposed, 16-byte header; FP32 activations/state",
    )
    (a.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    # Compile-time sizes for the existing checked loader; independent Qwen builds.
    sizes = ",".join(str(v.nbytes) + "ull" for v in arrays)
    if arch == "qwen3":
        elements = [
            "2048",
            "2048",
            "2048",
            "1024",
            "1024",
            "2048",
            "16ull*capacity",
            "6144",
            "6144",
            "151936",
            "28ull*capacity*1024",
            "28ull*capacity*1024",
            "0",
            "151936ull*8",
        ]
        zeros = "false"
        rope = 12
        logits = 9
    else:
        elements = [
            "2048",
            "2048",
            "6144",
            "6144",
            "6144",
            "4096",
            "16",
            "16",
            "32",
            "6144",
            "2048",
            "6144",
            "6144",
            "8ull*capacity",
            "2048",
            "512",
            "512",
            "248320",
            "0",
        ]
        for i in range(24):
            elements += (
                ["512ull*capacity", "512ull*capacity"]
                if (i + 1) % 4 == 0
                else ["3ull*6144", "16ull*128*128"]
            )
        elements += ["248320ull*8"]
        zeros = "i>=19 && i<67 && (((i-19)/2+1)%4!=0)"
        rope = 18
        logits = 17
    (a.output / "qwen_spec.h").write_text(
        '#pragma once\nstatic constexpr char modelMagic[]="QWENRAW1";\n'
        + f"static constexpr int vocabulary={vocab},weightCount={len(arrays)},bufferCount={len(elements)},ropeBuffer={rope},logitsBuffer={logits},eosToken={eos},alternateEosToken={248044 if arch == 'qwen35' else 151643};\n"
        + f"static inline uint64_t weightBytes(int i,uint32_t) {{ static const uint64_t s[]={{ {sizes} }};return s[i];}}\n"
        + f"static inline uint64_t bufferElements(int i,uint32_t capacity) {{const uint64_t s[]={{ {','.join(elements)} }};return s[i];}}\n"
        + f"static inline bool zeroBuffer(int i){{(void)i;return {zeros};}}\n"
        + "static inline int sharedSignGroup(int){return -1;}\nstatic inline bool centeredParams(int){return false;}\n"
    )
    print(arch, len(arrays), sum(v.nbytes for v in arrays), flush=True)


if __name__ == "__main__":
    main()
