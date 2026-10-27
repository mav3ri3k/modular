#!/usr/bin/env python3
# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the Apache License v2.0 with LLVM Exceptions: https://llvm.org/LICENSE.txt
"""Generate subgroup-independent Bonsai 2B, 4B and 9B Android models."""

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def replace_in_function(text: str, name: str, old: str, new: str) -> str:
    start = text.index("def " + name + "(")
    end = text.find("\n\n@always_inline", start + 10)
    plain = text.find("\n\ndef ", start + 10)
    if end < 0 or 0 <= plain < end:
        end = plain
    if end < 0:
        end = len(text)
    part = text[start:end]
    if old not in part:
        raise ValueError(f"{name}: expected source not found")
    return text[:start] + part.replace(old, new) + text[end:]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--size", choices=("2b", "4b", "9b"), default="2b")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    if a.size == "2b":
        m = (ROOT / "bonsai/model.mojo").read_text()
        k = (ROOT / "bonsai/kernels.mojo").read_text()
    else:
        subprocess.run(
            [
                "python3",
                str(ROOT / "bonsai_large/generate.py"),
                "--size",
                a.size,
                "--output",
                str(a.output),
            ],
            check=True,
        )
        m = (a.output / "model.mojo").read_text()
        k = (a.output / "kernels.mojo").read_text()
    for old, new in [
        ("kernels.packed_linear]", "kernels.portable_linear]"),
        ("kernels.normalize_qk]", "kernels.portable_normalize_qk]"),
        ("kernels.delta_update]", "kernels.portable_delta_update]"),
        ("kernels.gated_norm]", "kernels.portable_gated_norm]"),
        ("kernels.head_norm]", "kernels.portable_head_norm]"),
    ]:
        if old not in m:
            raise ValueError(old)
        m = m.replace(old, new)
    m = replace_in_function(
        m,
        "launch_packed_linear",
        "(Int(work) + 63) // 64",
        "(Int(work) + 511) // 512",
    )
    m = replace_in_function(
        m,
        "launch_delta_update",
        "(Int(work) * 8 + 63) // 64",
        "(Int(work) + 63) // 64",
    )
    scratch = 67 if a.size == "2b" else 83
    stages = 12 if a.size == "9b" else 9
    final_input = "src" if stages % 2 == 0 else "dst"
    new = """    var src = b[@SCRATCH@].create_sub_buffer[.float32](0, len(b[@SCRATCH@]) // 4)
    var dst = b[2].create_sub_buffer[.float32](0, len(b[2]) // 4)
    var sg = signs.create_sub_buffer[.float32](0, len(signs) // 4)
    var q = b[3].create_sub_buffer[.uint32](0, len(b[3]) // 4)
    rt.enqueue_function[kernels.hadamard_init](x.create_sub_buffer[.float32](0,len(x)//4), sg, src, width, inverse, grid_dim=Dim((Int(width)+63)//64,1,1), block_dim=64)
    for stage in range(@STAGES@):
        if stage % 2 == 0:
            rt.enqueue_function[kernels.hadamard_stage](src, dst, width, Int32(1 << stage), grid_dim=Dim((Int(width)+63)//64,1,1), block_dim=64)
        else:
            rt.enqueue_function[kernels.hadamard_stage](dst, src, width, Int32(1 << stage), grid_dim=Dim((Int(width)+63)//64,1,1), block_dim=64)
    rt.enqueue_function[kernels.hadamard_final](@FINAL_INPUT@, sg, dst, q, width, inverse, grid_dim=Dim((Int(width)+127)//128,1,1), block_dim=64)
    return 0"""
    new = (
        new.replace("@SCRATCH@", str(scratch))
        .replace("@STAGES@", str(stages))
        .replace("@FINAL_INPUT@", final_input)
    )
    rotate = m.index("def rotate(")
    body = m.index(") raises -> Int64:\n", rotate) + len(") raises -> Int64:\n")
    end = m.index("\n\n@always_inline\ndef project(", body)
    m = m[:body] + new + m[end:]
    # Fused candidate selection relies on 64-lane shuffles. The portable runner
    # selects from the complete logits on the host instead.
    head = k.index("def packed_head(")
    begin = k.index("    if select != 0:", head)
    end = k.index("\n\ndef embedding(", begin)
    k = k[:begin] + k[end:]
    portable = (Path(__file__).parent / "kernels.mojo.inc").read_text()
    if a.size != "2b":
        portable = replace_in_function(
            portable, "portable_normalize_qk", "i < 6144", "i < 8192"
        )
        portable = replace_in_function(
            portable, "portable_delta_update", "i < 2048", "i < 4096"
        )
        portable = replace_in_function(
            portable,
            "portable_delta_update",
            "2048+h*128+k",
            "2048+(h//2)*128+k",
        )
        portable = replace_in_function(
            portable,
            "portable_delta_update",
            "qkv[unsafe_offset=h*128+k]",
            "qkv[unsafe_offset=(h//2)*128+k]",
        )
        portable = replace_in_function(
            portable, "portable_gated_norm", "i < 2048", "i < 4096"
        )
        portable = replace_in_function(
            portable,
            "portable_head_norm",
            "2048 if mode == 0 else 512",
            "4096 if mode == 0 else 1024",
        )
    if a.size == "9b":
        portable = replace_in_function(
            portable, "hadamard_final", "0.04419417382415922", "0.015625"
        )
    k += portable
    (a.output / "model.mojo").write_text(m)
    (a.output / "kernels.mojo").write_text(k)
    if a.size == "2b":
        spec = (
            (ROOT / "model_spec.h")
            .read_text()
            .split("#elif defined(BONSAI_MODEL)\n", 1)[1]
            .split("\n#else\n", 1)[0]
        )
        (a.output / "qwen_spec.h").write_text(
            "#pragma once\n#include <cstdint>\n" + spec + "\n"
        )


if __name__ == "__main__":
    main()
