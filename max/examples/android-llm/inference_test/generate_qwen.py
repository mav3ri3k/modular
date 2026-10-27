#!/usr/bin/env python3
# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the Apache License v2.0 with LLVM Exceptions: https://llvm.org/LICENSE.txt
"""Generate the two fixed architecture ports, reusing existing Mojo operators."""

import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent


def section(s: str, name: str) -> tuple[int, int]:
    start = s.index("def " + name + "(")
    if s[max(0, start - 15) : start].endswith("@always_inline\n"):
        start -= len("@always_inline\n")
    end = s.find("\n\n@always_inline", start + 20)
    plain = s.find("\n\ndef ", start + 20)
    if end < 0 or 0 <= plain < end:
        end = plain
    return start, len(s) if end < 0 else end


def replace(s: str, name: str, body: str) -> str:
    a, b = section(s, name)
    return s[:a] + body + "\n" + s[b:]


RAW = """@always_inline
def raw_linear(rt: DeviceContext, x: DeviceBuffer[.uint8], w: DeviceBuffer[.uint8], y: DeviceBuffer[.uint8], scratch: DeviceBuffer[.uint8], width: Int32, reduction: Int32) raises -> Int64:
    # Vocabulary projections have enough rows to hide latency in one launch.
    # Narrow Q/K/V projections keep one 256-wide K tile per workgroup.
    var parts = (Int(reduction) + 255) // 256
    if (Int(width) + 7) // 8 >= 8192:
        parts = 1
    var dest = y
    if parts > 1:
        dest = scratch
    rt.enqueue_function[quant.linear](
        x.create_sub_buffer[.float32](0, len(x)//4), w,
        dest.create_sub_buffer[.float32](0, len(dest)//4), width, reduction, Int32(parts),
        grid_dim=Dim((Int(width)+511)//512, parts, 1), block_dim=64,
    )
    if parts > 1:
        rt.enqueue_function[quant.reduce](
            scratch.create_sub_buffer[.float32](0, len(scratch)//4),
            y.create_sub_buffer[.float32](0, len(y)//4), width, reduction,
            grid_dim=Dim((Int(width)+63)//64, 1, 1), block_dim=64,
        )
    return 0

@always_inline
def raw_embedding(rt: DeviceContext, w: DeviceBuffer[.uint8], y: DeviceBuffer[.uint8], token: Int32) raises -> Int64:
    rt.enqueue_function[quant.embedding](w, y.create_sub_buffer[.float32](0,len(y)//4),token,Int32(VOCAB),grid_dim=Dim(32,1,1),block_dim=64)
    return 0
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", choices=["qwen3", "qwen35"], required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    if a.arch == "qwen35":
        m = (SOURCE / "bonsai/model.mojo").read_text()
        k = (SOURCE / "bonsai/kernels.mojo").read_text()
        # Unused legacy launch wrappers stay compilable but are never called.
        m = replace(
            m,
            "project",
            """@always_inline
def project(rt: DeviceContext,w: List[DeviceBuffer[.uint8]],b: List[DeviceBuffer[.uint8]],slot: Int,x: DeviceBuffer[.uint8],y: DeviceBuffer[.uint8],width: Int32,reduction: Int32=2048,reuse_transform: Bool=False,head_gpu_columns: Int32=-1,head_input: OptionalPointer[UInt8, MutUntrackedOrigin]=None,gpu_greedy: Bool=False) raises -> Int64:
    return raw_linear(rt,x,w[slot],y,b[67],width,reduction)
""",
        )
        start = m.index(
            "    status = launch_embedding(", m.index("def forward(")
        )
        end = m.index("    for layer in range(24):", start)
        m = (
            m[:start]
            + """    status = raw_embedding(rt,w[0],b[0],token)
    if status != 0:
        return status
"""
            + m[end:]
        )
        k = k.replace(
            "-exp(constants[unsafe_offset=i]) * softplus",
            "constants[unsafe_offset=i] * softplus",
        )
        vocab = 248320
    else:
        m = (SOURCE / "model.mojo").read_text()
        k = (SOURCE / "kernels.mojo").read_text()
        # Dimensions in the SmolLM2 scaffolding, with Qwen3's full head RoPE.
        for x, y in [
            ("49152", "151936"),
            ("1536", "6144"),
            ("576", "2048"),
            ("192", "1024"),
        ]:
            m = m.replace(x, y)
            k = k.replace(x, y)
        m = (
            m.replace("range(30)", "range(28)")
            .replace("layer * 9", "layer * 11")
            .replace("9 * (position + 1)", "16 * (position + 1)")
        )
        k = (
            k.replace("9 * length", "16 * length")
            .replace("h // 3", "h // 2")
            .replace("range(64)", "range(128)")
            .replace("h * 64", "h * 128")
            .replace("d = i % 64", "d = i % 128")
            .replace("h = i // 64", "h = i // 128")
            .replace("(h // 2) * 64", "(h // 2) * 128")
            .replace("Float32(0.125)", "Float32(0.08838834764831845)")
        )
        # Rewrite rotary operator rather than change literal stride values globally.
        k = replace(
            k,
            "rope",
            """def rope(x: Pointer[Float32, MutAnyOrigin],table: Pointer[Float32, MutAnyOrigin],width: Int32,position: Int32):
    var i=global_idx.x
    if i<Int(width)//2:
        var h=i//64
        var d=i%64
        var off=h*128+d
        var c=table[unsafe_offset=Int(position)*128+d]
        var s=table[unsafe_offset=Int(position)*128+d+64]
        var a=x[unsafe_offset=off]
        var b=x[unsafe_offset=off+64]
        x[unsafe_offset=off]=a*c-b*s
        x[unsafe_offset=off+64]=b*c+a*s
""",
        )
        # Qwen3 F32 norm tensors; scalar RMSNorm is a correctness-first baseline.
        k = k.replace(
            "w: Pointer[UInt16, MutAnyOrigin]",
            "w: Pointer[Float32, MutAnyOrigin]",
        )
        k = k.replace("bf16(w[unsafe_offset=i])", "w[unsafe_offset=i]").replace(
            "Float32(1e-5)", "Float32(1e-6)"
        )
        # Unused embedding/linear wrappers removed; raw block operations used below.
        for name in ["launch_embedding", "launch_linear"]:
            m = replace(m, name, "")
        m = m.replace(
            "w.create_sub_buffer[.uint16](0, len(w) // size_of[UInt16]())",
            "w.create_sub_buffer[.float32](0, len(w) // 4)",
        )
        m = m.replace(
            "launch_embedding(rt, 2048, w[0], b[0], token)",
            "raw_embedding(rt, w[0], b[0], token)",
        )
        m = m.replace(
            "launch_linear(\n                rt, 6144, b[1], w[off + 6 + p], b[7 + p], 6144, 2048\n            )",
            "raw_linear(rt, b[1], w[off+6+p], b[7+p], b[13], 6144, 2048)",
        )
        m = m.replace(
            "launch_linear(\n                rt, width, b[1], w[off + 1 + p], b[2 + p], width, 2048\n            )",
            "raw_linear(rt,b[1],w[off+1+p],b[2+p],b[13],width,2048)",
        )
        for x, w, y, width, red in [
            ("b[5]", "w[off + 4]", "b[1]", 2048, 2048),
            ("b[7]", "w[off + 8]", "b[1]", 2048, 6144),
            ("b[1]", "w[1]", "b[9]", 151936, 2048),
        ]:
            m = m.replace(
                f"launch_linear(rt, {width}, {x}, {w}, {y}, {width}, {red})",
                f"raw_linear(rt,{x},{w},{y},b[13],{width},{red})",
            )
        insert = m.index("        for p in range(2):", m.index("def forward("))
        m = (
            m[:insert]
            + """        rt.enqueue_function[kernels.qk_norm](b[2].create_sub_buffer[.float32](0,len(b[2])//4), w[off+9].create_sub_buffer[.float32](0,len(w[off+9])//4),Int32(2048),grid_dim=Dim(32,1,1),block_dim=64)
        rt.enqueue_function[kernels.qk_norm](b[3].create_sub_buffer[.float32](0,len(b[3])//4), w[off+10].create_sub_buffer[.float32](0,len(w[off+10])//4),Int32(1024),grid_dim=Dim(16,1,1),block_dim=64)
"""
            + m[insert:]
        )
        k += """\n\ndef qk_norm(x: Pointer[Float32, MutAnyOrigin],w: Pointer[Float32, MutAnyOrigin],width: Int32):
    var i=global_idx.x
    if i<Int(width):
        var base=(i//128)*128
        var ss:Float32=0
        for d in range(128):
            var v=x[unsafe_offset=base+d]
            ss+=v*v
        # Each work item owns one complete head to avoid in-place read/write races.
        # Only the first lane executes; this scalar baseline is intentionally simple.
        if i%128==0:
            var scale=rsqrt(ss/128+Float32(1e-6))
            for d in range(128):
                x[unsafe_offset=base+d]*=scale*w[unsafe_offset=d]
"""
        # Avoid any other lane reading while lane zero writes.
        k = k.replace(
            "    if i<Int(width):\n        var base=(i//128)*128",
            "    if i<Int(width) and i%128==0:\n        var base=(i//128)*128",
        )
        # Reuse Bonsai's two-pass RMSNorm rather than the original scalar norm.
        bm = (SOURCE / "bonsai/model.mojo").read_text()
        bk = (SOURCE / "bonsai/kernels.mojo").read_text()
        m = replace(m, "launch_rms_norm", "")
        m = (
            m.replace(
                "launch_rms_norm(rt, 2048, b[0], w[off], b[1])",
                "norm(rt,b[0],w[off],b[1],b[13])",
            )
            .replace(
                "launch_rms_norm(rt, 2048, b[0], w[off + 5], b[1])",
                "norm(rt,b[0],w[off+5],b[1],b[13])",
            )
            .replace(
                "launch_rms_norm(rt, 2048, b[0], w[2], b[1])",
                "norm(rt,b[0],w[2],b[1],b[13])",
            )
        )
        for name in ["launch_rms_partials", "launch_rms_norm", "norm"]:
            start, end = section(bm, name)
            m = m.replace(
                "\n\ndef forward(", "\n\n" + bm[start:end] + "\n\ndef forward("
            )
        k = replace(k, "rms_norm", "")
        for name in ["rms_partials", "rms_norm"]:
            start, end = section(bk, name)
            k += "\n\n" + bk[start:end]
        vocab = 151936
    m = m.replace(
        "    head_input: OptionalPointer[UInt8, MutUntrackedOrigin] = None,\n) raises -> Int64:",
        "    head_input: OptionalPointer[UInt8, MutUntrackedOrigin] = None,\n    gpu_greedy: Bool = False,\n) raises -> Int64:",
    )
    m = m.replace("import kernels", "import kernels\nimport quant")
    m = m.replace(
        "\n\ndef forward(",
        "\n\n" + RAW.replace("VOCAB", str(vocab)) + "\n\ndef forward(",
    )
    (a.output / "model.mojo").write_text(m)
    (a.output / "kernels.mojo").write_text(k)
    (a.output / "quant.mojo").write_text((HERE / "quant.mojo").read_text())


if __name__ == "__main__":
    main()
