#!/usr/bin/env python3
# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the Apache License v2.0 with LLVM Exceptions: https://llvm.org/LICENSE.txt
"""Check demo Bonsai export against Prism PQ2_0 GGUF codes, scales and signs."""

import argparse
import json
import struct
from collections.abc import Callable
from pathlib import Path

import numpy as np
from gguf import GGUFReader

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--weights", type=Path, required=True)
p.add_argument("--gguf", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
r = GGUFReader(a.gguf)
ts = {t.name: t for t in r.tensors}
weights = []
with a.weights.open("rb") as f:
    assert f.read(8) == b"BONSAI01"
    n, capacity = struct.unpack("<II", f.read(8))
    assert n == 605
    for _ in range(n):
        weights.append(f.read(struct.unpack("<Q", f.read(8))[0]))
    assert not f.read(1)
widths = r.fields["prism.hadamard.sign_widths"].contents()
values = r.fields["prism.hadamard.sign_values"].contents()
signs = {}
off = 0
for width in widths:
    signs[width] = np.array(values[off : off + width], dtype=np.float32)
    off += width
records = []


def equal(name: str, x: np.ndarray, y: np.ndarray) -> None:
    x = np.asarray(x)
    y = np.asarray(y)
    assert x.shape == y.shape, (name, x.shape, y.shape)
    ok = bool(np.array_equal(x, y))
    record = dict(name=name, equal=ok)
    if not ok:
        record["max_abs"] = float(np.max(np.abs(x.astype(np.float64) - y)))
    records.append(record)


def packed(slot: int, name: str) -> None:
    t = ts[name]
    assert t.tensor_type.name in ("PQ2_0", "PTQ1_0")
    red, out = map(int, t.shape)
    if t.tensor_type.name == "PQ2_0":
        raw = t.data.view(np.uint8).reshape(out, red // 128, 34)
        source_codes = raw[:, :, 2:]
        source_scales = raw[:, :, :2]
    else:
        raw = t.data.view(np.uint8).reshape(out, red // 128, 28)
        pieces = []
        for offset, count in [(0, 16), (16, 8)]:
            for n in range(5):
                q = (
                    raw[:, :, offset : offset + count].astype(np.uint16)
                    * (3**n)
                ) & 255
                pieces.append((q * 3) >> 8)
        for n in range(4):
            q = (raw[:, :, 24:26].astype(np.uint16) * (3**n)) & 255
            pieces.append((q * 3) >> 8)
        trits = (
            np.concatenate(pieces, axis=-1)
            .astype(np.uint8)
            .reshape(out, red // 128, 32, 4)
        )
        source_codes = np.bitwise_or.reduce(
            trits << np.array([0, 2, 4, 6], dtype=np.uint8), axis=-1
        )
        source_scales = raw[:, :, 26:28]
    codes = (
        np.frombuffer(weights[slot], dtype="<u4")
        .reshape(red // 16, out)
        .T.copy()
        .view(np.uint8)
        .reshape(out, red // 128, 32)
    )
    equal(name + ".codes", codes, source_codes)
    params = (
        np.frombuffer(weights[slot + 1], dtype="<f4")
        .reshape(red // 128, out, 2)
        .transpose(1, 0, 2)
    )
    equal(
        name + ".scales",
        params[:, :, 0],
        source_scales.copy()
        .view("<f2")
        .reshape(out, red // 128)
        .astype(np.float32),
    )
    equal(name + ".bias", params[:, :, 1], -params[:, :, 0])
    equal(
        name + ".signs",
        np.frombuffer(weights[slot + 2], dtype="<f4"),
        signs[red],
    )


def f32(
    slot: int,
    name: str,
    transform: Callable[[np.ndarray], np.ndarray] = lambda x: x,
) -> None:
    equal(
        name,
        np.frombuffer(weights[slot], dtype="<f4").reshape(-1),
        np.asarray(transform(ts[name].data)).reshape(-1),
    )


packed(0, "token_embd.weight")
f32(3, "output_norm.weight")
for i in range(24):
    b = 4 + i * 25
    pre = f"blk.{i}."
    f32(b, pre + "attn_norm.weight")
    f32(b + 1, pre + "post_attention_norm.weight")
    for slot, name in [
        (2, "ffn_gate.weight"),
        (5, "ffn_up.weight"),
        (8, "ffn_down.weight"),
    ]:
        packed(b + slot, pre + name)
    if (i + 1) % 4:
        for slot, name in [
            (11, "attn_qkv.weight"),
            (14, "attn_gate.weight"),
            (17, "ssm_out.weight"),
        ]:
            packed(b + slot, pre + name)
        f32(b + 20, pre + "ssm_alpha.weight", lambda x: x.T)
        f32(b + 21, pre + "ssm_beta.weight", lambda x: x.T)
        old = np.frombuffer(weights[b + 22], dtype="<f4")
        current = ts[pre + "ssm_a"].data
        records.append(
            dict(
                name=pre + "ssm_a",
                equal=bool(
                    np.allclose(
                        -np.exp(old[:16]), current, rtol=1e-6, atol=1e-7
                    )
                ),
                max_abs=float(np.abs(-np.exp(old[:16]) - current).max()),
            )
        )
        equal(pre + "ssm_dt.bias", old[16:], ts[pre + "ssm_dt.bias"].data)
        f32(b + 23, pre + "ssm_conv1d.weight", lambda x: x.T)
        f32(b + 24, pre + "ssm_norm.weight")
    else:
        for slot, name in [
            (11, "attn_q.weight"),
            (14, "attn_k.weight"),
            (17, "attn_v.weight"),
            (20, "attn_output.weight"),
        ]:
            packed(b + slot, pre + name)
        f32(b + 23, pre + "attn_q_norm.weight")
        f32(b + 24, pre + "attn_k_norm.weight")
result = dict(
    pass_all=all(x["equal"] for x in records),
    checks=len(records),
    capacity=capacity,
    records=records,
)
a.output.write_text(json.dumps(result, indent=2) + "\n")
print(
    "checks", len(records), "failures", [x for x in records if not x["equal"]]
)
raise SystemExit(0 if result["pass_all"] else 1)
