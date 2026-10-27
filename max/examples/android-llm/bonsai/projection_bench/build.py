#!/usr/bin/env python3
# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the Apache License v2.0 with LLVM Exceptions: https://llvm.org/LICENSE.txt
"""Compile standalone projection probes through the production Mojo target."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from generate import LAYOUTS, generate


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mojo", required=True, type=Path)
    p.add_argument(
        "--packages",
        required=True,
        type=Path,
        help="Directory with matching std.mojoc and HAL max.mojoc",
    )
    p.add_argument("--ndk", required=True, type=Path)
    p.add_argument("--opencl-headers", required=True, type=Path)
    p.add_argument(
        "--opencl-library",
        required=True,
        type=Path,
        help="Android libOpenCL.so used only for host linking",
    )
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    out = args.output.resolve()
    generate(out)
    env = dict(
        os.environ, MODULAR_MOJO_MAX_IMPORT_PATH=str(args.packages.resolve())
    )
    sources = [
        out / f"{layout}_{mode}.mojo"
        for layout in LAYOUTS
        for mode in ("full", "load", "unpack")
    ]
    for source in sources:
        host = out / "host.mojo"
        host.write_text(f"""import {source.stem} as probe
from max.gpu.host import DeviceContext
from std.memory import Pointer
@export
def launch(ctx: Pointer[DeviceContext, MutAnyOrigin]) abi("C"):
    try:
        var x = ctx[].enqueue_create_buffer[.uint32](1024)
        var w = ctx[].enqueue_create_buffer[.uint32](262144)
        var p = ctx[].enqueue_create_buffer[.float16](32768)
        var y = ctx[].enqueue_create_buffer[.float32](4096)
        ctx[].enqueue_function[probe.projection](x, w, p, y, Int32(2048), Int32(2048), grid_dim=(32, 2, 1), block_dim=64)
    except:
        pass
""")
        for old in out.glob("host_*.cl"):
            old.unlink()
        command = [
            str(args.mojo.resolve()),
            "build",
            str(host),
            "--target-triple",
            "aarch64-linux-android33",
            "--target-cpu",
            "generic",
            "--target-accelerator",
            "adreno-840",
            "-I",
            str(args.packages.resolve()),
            "--emit",
            "asm",
            "-O3",
            "-o",
            str(out / "host.s"),
        ]
        completed = subprocess.run(
            command, env=env, text=True, capture_output=True
        )
        if completed.returncode:
            raise RuntimeError(completed.stderr)
        emitted = list(out.glob("host_*.cl"))
        if len(emitted) != 1:
            raise RuntimeError(f"Expected one generated kernel: {emitted}")
        target = out / f"{source.stem}.cl"
        shutil.copyfile(emitted[0], target)
        print(f"{target.name}: {target.stat().st_size} bytes", flush=True)
    toolchains = list(
        (args.ndk / "toolchains/llvm/prebuilt").glob(
            "*/bin/aarch64-linux-android33-clang++"
        )
    )
    if len(toolchains) != 1:
        raise RuntimeError("Expected one NDK toolchain")
    subprocess.run(
        [
            str(toolchains[0]),
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(args.opencl_headers),
            str(Path(__file__).with_name("bench.cpp")),
            str(args.opencl_library),
            "-o",
            str(out / "bench"),
        ],
        check=True,
    )
    shutil.copyfile(args.opencl_library, out / "libOpenCL.so")
    cxx = list(
        (args.ndk / "toolchains/llvm/prebuilt").glob(
            "*/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"
        )
    )
    if len(cxx) != 1:
        raise RuntimeError("Expected one NDK libc++_shared.so")
    shutil.copyfile(cxx[0], out / "libc++_shared.so")
    metadata = json.loads((out / "provenance.json").read_text())
    metadata.update(
        {
            "generated_cl_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in out.glob("*.cl")
                if not p.name.startswith("host_")
            },
            "mojo_sha256": hashlib.sha256(
                args.mojo.resolve().read_bytes()
            ).hexdigest(),
        }
    )
    metadata["host_sha256"] = hashlib.sha256(
        (out / "bench").read_bytes()
    ).hexdigest()
    metadata["source_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path(__file__).parent.iterdir()
        if p.suffix in (".py", ".cpp")
    }
    metadata["package_sha256"] = {
        name: hashlib.sha256((args.packages / name).read_bytes()).hexdigest()
        for name in ("std.mojoc", "max.mojoc")
    }
    (out / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
