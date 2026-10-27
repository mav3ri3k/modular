#!/usr/bin/env python3
# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the Apache License v2.0 with LLVM Exceptions: https://llvm.org/LICENSE.txt
"""Check centered packed parameters and shared signs in 32-layer checkpoints."""

import sys

import mlx.core as mx
import numpy as np

x = mx.load(sys.argv[1])
failures = []
count = 0
for k in x:
    if (
        k.endswith(".scales")
        and k.replace(".scales", ".biases") in x
        and k.startswith("language_model.")
    ):
        a = np.asarray(x[k])
        b = np.asarray(x[k.replace(".scales", ".biases")])
        count += 1
        if not np.array_equal(a, -b):
            failures.append(k)
for layer in range(32):
    p = f"language_model.model.layers.{layer}."
    groups = [["mlp.gate_proj", "mlp.up_proj"]]
    groups += [
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
        if layer % 4 == 3
        else ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z"]
    ]
    for group in groups:
        keys = [p + n + ".signs" for n in group]
        if not all(k in x for k in keys):
            failures.append("missing " + str(keys))
            continue
        if any(
            not np.array_equal(np.asarray(x[keys[0]]), np.asarray(x[k]))
            for k in keys[1:]
        ):
            failures.append("signs " + str(keys))
print("packed tensors checked:", count, "failures:", len(failures))
print("\n".join(failures))
if count not in (201, 202) or failures:
    raise SystemExit(1)
