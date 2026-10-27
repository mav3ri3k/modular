#!/usr/bin/env python3
# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the Apache License v2.0 with LLVM Exceptions: https://llvm.org/LICENSE.txt
"""Compare complete teacher-forced logits; fail on nonfinite or excessive error."""

import argparse
import json
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("reference", type=Path)
p.add_argument("actual", type=Path)
p.add_argument("--vocabulary", type=int, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--max-abs", type=float, default=0.05)
p.add_argument("--relative-l2", type=float, default=0.005)
a = p.parse_args()
x = (
    np.fromfile(a.reference, dtype="<f4")
    .reshape(-1, a.vocabulary)
    .astype(np.float64)
)
y = (
    np.fromfile(a.actual, dtype="<f4")
    .reshape(-1, a.vocabulary)
    .astype(np.float64)
)
assert (
    x.shape == y.shape
    and x.size
    and np.isfinite(x).all()
    and np.isfinite(y).all()
)
d = x - y
record = dict(
    steps=len(x),
    vocabulary=a.vocabulary,
    max_abs=float(np.abs(d).max()),
    relative_l2=float(np.linalg.norm(d) / np.linalg.norm(x)),
    reference_argmax=x.argmax(1).tolist(),
    actual_argmax=y.argmax(1).tolist(),
    argmax_agreement=float((x.argmax(1) == y.argmax(1)).mean()),
)
record["pass"] = (
    record["max_abs"] < a.max_abs and record["relative_l2"] < a.relative_l2
)
print(json.dumps(record, indent=2))
a.output.write_text(json.dumps(record, indent=2) + "\n")
raise SystemExit(0 if record["pass"] else 1)
