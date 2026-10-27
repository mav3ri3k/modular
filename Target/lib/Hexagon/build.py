#!/usr/bin/env python3
# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the MAX License: https://modular.com/legal/max-license
"""Build a freestanding Mojo -> Hexagon LLVM object and an Android FastRPC probe."""

import argparse
import os
import re
import struct
import subprocess
from pathlib import Path


def run(*args, **kwargs) -> None:
    subprocess.run([str(a) for a in args], check=True, **kwargs)


def retarget(ir: str) -> str:
    # Bootstrap only: source-generated 32-bit pointer/index IR has the same
    # scalar C ABI. SIMD, target intrinsics and aggregate layouts need native
    # Hexagon TargetTraits before general model compilation is supported.
    if 'target triple = "riscv32-unknown-linux-gnu"' not in ir:
        raise ValueError("Expected generic riscv32 bootstrap IR")
    if 'target datalayout = "e-m:e-p:32:32-i64:64-n32-S128"' not in ir:
        raise ValueError("Unexpected bootstrap data layout")
    layout = (
        "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-"
        "i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-"
        "v512:512:512-v1024:1024:1024-v2048:2048:2048"
    )
    ir = re.sub(
        r'target datalayout = "[^"]+"', f'target datalayout = "{layout}"', ir
    )
    ir = ir.replace("riscv32-unknown-linux-gnu", "hexagon-unknown-linux-musl")
    ir = re.sub(r' "target-(?:cpu|features)"="[^"]*"', "", ir)

    # LLVM 23 emits native-width f0x literals; LLVM 22 expects FP32 constants
    # represented by their exact double-precision hexadecimal encoding.
    def old_float(match: re.Match[str]) -> str:
        value = struct.unpack(">f", bytes.fromhex(match.group(1)))[0]
        return "0x" + struct.pack(">d", value).hex().upper()

    ir = re.sub(r"\bf0x([0-9A-Fa-f]{8})\b", old_float, ir)
    empty = re.findall(r"attributes #(\d+) = \{\s*\}", ir)
    for group in empty:
        ir = re.sub(r"attributes #" + group + r" = \{\s*\}", "", ir)
        ir = re.sub(r"\s+#" + group + r"\b", "", ir)
    if re.search(r"@llvm\.riscv\.", ir):
        raise ValueError("RISC-V-specific intrinsic in freestanding kernel")
    return ir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mojo", type=Path, required=True)
    p.add_argument("--stdlib-dir", type=Path, required=True)
    p.add_argument("--llc", type=Path, required=True)
    p.add_argument("--zig", type=Path, required=True)
    p.add_argument("--ndk", type=Path, required=True)
    p.add_argument("--sdk", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--hvx-workers", type=int, choices=[1, 2, 4, 8], default=8)
    p.add_argument("--backend", choices=["hmx", "hvx"], default="hmx")
    a = p.parse_args()
    src = Path(__file__).resolve().parent
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ, MODULAR_MOJO_MAX_IMPORT_PATH=str(a.stdlib_dir.resolve())
    )
    objects = []
    for name in [
        "kernels",
        "hvx_projection" if a.backend == "hvx" else "projection",
    ]:
        run(
            a.mojo.resolve(),
            "build",
            src / f"{name}.mojo",
            "--target-triple",
            "riscv32-unknown-linux-gnu",
            "--target-cpu",
            "generic-rv32",
            "--emit",
            "llvm",
            "-o",
            out / f"{name}-bootstrap.ll",
            env=env,
        )
        (out / f"{name}.ll").write_text(
            retarget((out / f"{name}-bootstrap.ll").read_text())
        )
        # The scalar smoke runs without an HVX lock; keep auto-vectorization off
        # there. The projector explicitly uses HVX on its registered workers.
        features = (
            ("-mattr=+hvxv73,+hvx-length128b,+hvx-qfloat",)
            if name != "kernels"
            else ()
        )
        run(
            a.llc.resolve().with_name("opt"),
            "-passes=default<O3>",
            "-S",
            "-mtriple=hexagon-unknown-linux-musl",
            "-mcpu=hexagonv73",
            *features,
            out / f"{name}.ll",
            "-o",
            out / f"{name}-optimized.ll",
        )
        run(
            a.llc.resolve(),
            "-mtriple=hexagon-unknown-linux-musl",
            "-mcpu=hexagonv73",
            *features,
            "-relocation-model=pic",
            "-filetype=obj",
            out / f"{name}-optimized.ll",
            "-o",
            out / f"{name}.o",
        )
        objects.append(out / f"{name}.o")
    sdk = a.sdk.resolve()
    run(
        a.zig.resolve(),
        "cc",
        "-target",
        "hexagon-linux-musl",
        "-mcpu=hexagonv73+hvx+hvxv73+hvx_length128b+hvx_qfloat",
        "-fPIC",
        "-nostdlib",
        "-O2",
        f"-DMODULAR_HEXAGON_HVX_WORKERS={a.hvx_workers}",
        "-I",
        src / "freestanding",
        "-I",
        sdk / "incs",
        "-I",
        sdk / "incs/stddef",
        "-I",
        sdk / "rtos/qurt/computev81/include/qurt",
        "-shared",
        f"-Wl,-T,{src}/probe.ld",
        src / "probe_skel.c",
        src
        / (
            "hvx_projection_skel.c"
            if a.backend == "hvx"
            else "projection_skel.c"
        ),
        *objects,
        "-o",
        out / "libmodular_hexagon.so",
    )
    cc = next(
        (a.ndk / "toolchains/llvm/prebuilt").glob(
            "*/bin/aarch64-linux-android31-clang"
        )
    )
    run(cc, "-O2", src / "probe_host.c", "-ldl", "-o", out / "probe_host")
    cxx = cc.with_name(cc.name + "++")
    run(
        cxx,
        "-O3",
        "-std=c++17",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-fPIC",
        "-shared",
        "-static-libstdc++",
        *(["-DMODULAR_HEXAGON_HVX"] if a.backend == "hvx" else []),
        "-I",
        src.parents[1] / "include",
        src / "Projection.cpp",
        "-ldl",
        "-o",
        out / "HexagonProjection.so",
    )
    run(
        cxx,
        "-O3",
        "-std=c++17",
        "-static-libstdc++",
        "-I",
        src.parents[1] / "include",
        src / "projection_bench.cpp",
        "-ldl",
        "-o",
        out / "projection_bench",
    )


if __name__ == "__main__":
    main()
