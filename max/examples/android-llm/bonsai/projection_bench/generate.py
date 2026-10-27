#!/usr/bin/env python3
# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the Apache License v2.0 with LLVM Exceptions: https://llvm.org/LICENSE.txt
"""Generate isolated Mojo projections from the current production dot/kernel.

Only layout addressing and bit extraction change. Scheduling and accumulation
order are inherited from kernels.mojo. Generated files belong outside git.
"""

import argparse
import hashlib
import json
from pathlib import Path

LAYOUTS = (
    "baseline",
    "group_word",
    "word_group",
    "group_word_planes",
    "word_group_planes",
    "group_major",
)


def generate(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    source = (Path(__file__).resolve().parents[1] / "kernels.mojo").read_text()
    dot = source[
        source.index("@always_inline\ndef _group_dot8(") : source.index(
            "\n\ndef packed_linear("
        )
    ]
    kernel = source[
        source.index("def packed_linear(") : source.index(
            "\n\ndef packed_reduce("
        )
    ]
    # Stop at a following helper too: integrated kernels may be defined
    # between the legacy projection and its reduction.
    next_helper = kernel.find("\n\n@always_inline")
    if next_helper >= 0:
        kernel = kernel[:next_helper]
    scalar = source[
        source.index("@always_inline\ndef _group_dot(") : source.index(
            "@always_inline\ndef _group_dot8("
        )
    ]
    expected = (
        (dot, "(group * 8 + word) * width + i"),
        (dot, "var c = (codes >> UInt32(pair * 4)) & 15"),
        (dot, "((c >> 2) & 3)"),
        (kernel, "_group_dot8(x, w, Int(width), i, group)"),
    )
    for body, pattern in expected:
        if body.count(pattern) != 1:
            raise ValueError(f"Production kernel structure changed: {pattern}")
    kernel = kernel.replace("def packed_linear(", "def projection(")
    header = """# Generated from production kernels.mojo; see generate.py.
from std.memory import Pointer, bitcast
from std.simd import FastMathFlag
from max.gpu import global_idx, block_idx
from max.gpu.primitives.warp import shuffle_xor

"""
    for layout in LAYOUTS:
        address = "(group * 8 + word) * width + i"
        if layout.startswith("group_word"):
            # groups is supplied separately: do not specialize dimensions.
            address = "Int(((UInt64(i) // 64 * (UInt64(groups) // 8) + UInt64(group) // 8) * 64 + UInt64(group) % 8 * 8 + UInt64(word)) * 64 + UInt64(i) % 64)"
        elif layout.startswith("word_group"):
            address = "Int(((UInt64(i) // 64 * (UInt64(groups) // 8) + UInt64(group) // 8) * 64 + UInt64(word) * 8 + UInt64(group) % 8) * 64 + UInt64(i) % 64)"
        elif layout == "group_major":
            address = "Int(((UInt64(group) // 8 * (UInt64(width) // 64) + UInt64(i) // 64) * 64 + UInt64(word) * 8 + UInt64(group) % 8) * 64 + UInt64(i) % 64)"
        for mode in ("full", "load", "unpack"):
            body = dot.replace(
                "    width: Int,", "    width: Int,\n    groups: Int,"
            )
            body = body.replace("(group * 8 + word) * width + i", address)
            if mode == "full":
                if layout.endswith("planes"):
                    body = body.replace(
                        "var c = (codes >> UInt32(pair * 4)) & 15",
                        "var c = codes >> UInt32(pair * 2)",
                    )
                    body = body.replace("((c >> 2) & 3)", "((c >> 16) & 3)")
            else:
                # Integer checksum keeps loads observable. The unpack checksum
                # additionally consumes every converted half's bits. Neither is
                # an additive estimate of a stage's cost in the full kernel.
                before = body[: body.index("    var d0lo")]
                extract = """
        comptime for pair in range(8):
            var c = (codes >> UInt32(pair * 4)) & 15
            var lo = (c & 3).cast[.float16]() - SIMD[.float16, 8](1)
            var hi = ((c >> 2) & 3).cast[.float16]() - SIMD[.float16, 8](1)
            checksum = (checksum * UInt32(16777619)) ^ bitcast[.uint16, 8](lo).cast[.uint32]()
            checksum = (checksum * UInt32(16777619)) ^ bitcast[.uint16, 8](hi).cast[.uint32]()
"""
                if layout.endswith("planes"):
                    extract = extract.replace(
                        "(codes >> UInt32(pair * 4)) & 15",
                        "codes >> UInt32(pair * 2)",
                    ).replace("((c >> 2) & 3)", "((c >> 16) & 3)")
                body = (
                    before
                    + "    var checksum = SIMD[.uint32, 8](0)\n    for word in range(8):\n"
                )
                body += (
                    f"        var codes = w.unsafe_load[width=8]({address})\n"
                )
                body += "        comptime for pair in range(8):\n            checksum ^= SIMD[.uint32, 8](x[unsafe_offset=group * 64 + word * 8 + pair])\n"
                body += (
                    "        checksum ^= codes\n" if mode == "load" else extract
                )
                body += "    return ((checksum ^ (checksum >> 16)) & 1023).cast[.float32]()\n"
            entry = kernel.replace(
                "_group_dot8(x, w, Int(width), i, group)",
                "_group_dot8(x, w, Int(width), Int(reduction) // 128, i, group)",
            )
            (output / f"{layout}_{mode}.mojo").write_text(
                header + scalar + body + "\n\n" + entry
            )
    (output / "provenance.json").write_text(
        json.dumps(
            {
                "production_sha256": hashlib.sha256(
                    source.encode()
                ).hexdigest(),
                "layouts": LAYOUTS,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    generate(parser.parse_args().output)
