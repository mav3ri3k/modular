#!/usr/bin/env python3
# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the Apache License v2.0 with LLVM Exceptions: https://llvm.org/LICENSE.txt
"""Bounded, validation-first categorical projection tuning on the attached phone."""

import json
import statistics
from pathlib import Path

from iterate import build, run

OUT = Path("/tmp/projection-autotune")
NAMES = [
    f"t{t}_g{g}_w{w}_u{u}"
    for t in (4, 8, 16)
    for g in (4, 8, 16)
    for w in (64, 128, 256)
    for u in (1, 2)
]


def interleaved(names: list[str]) -> list[str]:
    result = ["control"]
    for i, name in enumerate(names):
        result.append(name)
        if (i + 1) % 6 == 0:
            result.append("control")
    if result[-1] != "control":
        result.append("control")
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    build(OUT, ["control", "reduce", *NAMES])
    # Small fixtures reject only order-of-magnitude execution cliffs, not small
    # differences or occupancy hypotheses. Every screened point is preserved.
    run(
        OUT,
        interleaved(NAMES),
        OUT / "micro.json",
        1,
        ["64x2048"],
        0,
        32,
        False,
        True,
    )
    micro = json.loads((OUT / "micro.json").read_text())["records"]
    baseline = statistics.median(
        r["result"]["device_mean_ms"]
        for r in micro
        if r["variant"] == "control"
    )
    rejected = {
        r["variant"]: r["result"]["device_mean_ms"]
        for r in micro
        if r["variant"] != "control"
        and r["result"]["device_mean_ms"] > 10 * baseline
    }
    survivors = [n for n in NAMES if n not in rejected]
    (OUT / "screen.json").write_text(
        json.dumps(
            {
                "baseline_ms": baseline,
                "threshold_ms": 10 * baseline,
                "rejected_execution_cliffs": rejected,
                "survivors": survivors,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        "MICRO_SCREEN",
        len(survivors),
        "survivors;",
        len(rejected),
        "execution cliffs",
        flush=True,
    )
    run(
        OUT,
        interleaved(survivors),
        OUT / "shape-screen.json",
        1,
        ["6144x2048", "2048x2048", "2048x6144"],
        128,
        256,
        False,
        True,
    )
    rows = json.loads((OUT / "shape-screen.json").read_text())["records"]
    frequency = {"6144x2048": 66, "2048x2048": 42, "2048x6144": 24}
    scores = []
    for name in ["control", *survivors]:
        score = sum(
            count
            * statistics.median(
                r["result"]["device_mean_ms"]
                for r in rows
                if r["variant"] == name and r["shape"] == shape
            )
            for shape, count in frequency.items()
        )
        scores.append({"variant": name, "weighted_major_shape_ms": score})
    scores.sort(key=lambda r: r["weighted_major_shape_ms"])
    (OUT / "ranking.json").write_text(json.dumps(scores, indent=2) + "\n")
    finalists = [
        "control",
        *[r["variant"] for r in scores if r["variant"] != "control"][:5],
    ]
    print("FINALISTS", finalists, flush=True)
    run(
        OUT,
        finalists,
        OUT / "confirm.json",
        5,
        ["6144x2048", "2048x2048", "2048x6144", "4096x2048", "512x2048"],
        128,
        512,
        False,
        True,
    )
    # Independent seed + non-dyadic FP16 activations: bit-exact partial checks
    # for group8; tight FP32 reduction tolerance for other groupings.
    run(
        OUT,
        finalists,
        OUT / "accuracy.json",
        1,
        ["6144x2048", "2048x2048", "2048x6144", "4096x2048", "512x2048"],
        16,
        128,
        True,
        True,
        987654321,
    )


if __name__ == "__main__":
    main()
