#!/usr/bin/env python3
# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the Apache License v2.0 with LLVM Exceptions: https://llvm.org/LICENSE.txt
"""Build and test ordinary Mojo HAL DeviceContext launches on Android ARM64."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


def run(*args: str | Path, **kwargs) -> None:
    subprocess.run([str(arg) for arg in args], check=True, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mojo", type=Path, required=True)
    parser.add_argument("--ndk", type=Path, required=True)
    parser.add_argument("--bazel-arg", action="append", default=[])
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--serial", required=True)
    parser.add_argument(
        "--target-accelerator",
        choices=["opencl-generic", "adreno-840"],
        default="opencl-generic",
    )
    parser.add_argument("--adb", type=Path)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--native-main", type=Path)
    parser.add_argument(
        "--optimization-level", type=int, choices=range(4), default=3
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    mojo, ndk, out = (
        args.mojo.resolve(),
        args.ndk.resolve(),
        args.build_dir.resolve(),
    )
    out.mkdir(parents=True, exist_ok=True)
    adb = args.adb or ndk.parents[1] / "platform-tools/adb"
    # Stage the existing HAL overlay with Bazel, rather than duplicating or
    # editing the MAX sources in a custom staging script.
    run(
        root / "bazelw",
        "build",
        "--compilation_mode=opt",
        "--config=hal-device-context",
        *args.bazel_arg,
        "//max/driver/src/max_license:max_srcs_hal",
        cwd=root,
    )
    staged = root / "bazel-bin/max/driver/src/max_license/max_srcs_hal/max"
    run(mojo, "precompile", root / "Mojo/stdlib/std", "-o", out / "std.mojoc")
    env = dict(os.environ, MODULAR_MOJO_MAX_IMPORT_PATH=str(out))
    run(mojo, "precompile", staged, "-o", out / "max.mojoc", env=env)
    run(
        mojo,
        "build",
        args.fixture.resolve()
        if args.fixture
        else root / "MLRT/test/Driver/HAL/opencl_device_context.mojo",
        "--target-triple",
        "aarch64-linux-android33",
        "--target-cpu",
        "generic",
        "--target-accelerator",
        args.target_accelerator,
        "-I",
        root / "max/kernels/src",
        "-I",
        root / "max/kernels/src/graph_compiler",
        "-I",
        root / "MLRT/lib/Driver/DeviceContext/HAL",
        f"-O{args.optimization_level}",
        "--emit",
        "object",
        "-o",
        out / "host.o",
        env=env,
    )
    run(
        sys.executable,
        root / "Target/lib/OpenCL/build_runtime.py",
        "--ndk",
        ndk,
        "--output",
        out,
        *[f"--bazel-arg={arg}" for arg in args.bazel_arg],
    )
    libraries = json.loads((out / "android_artifacts.json").read_text())[
        "libraries"
    ]
    compilers = list(
        (ndk / "toolchains/llvm/prebuilt").glob(
            "*/bin/aarch64-linux-android33-clang++"
        )
    )
    if len(compilers) != 1:
        raise RuntimeError("Expected one Android NDK host toolchain")
    run(
        compilers[0],
        "-O2",
        args.native_main.resolve()
        if args.native_main
        else root / "Target/test/OpenCLDeviceContextMain.cpp",
        out / "host.o",
        f"-L{out / 'runtime'}",
        "-lKGENCompilerRTShared",
        "-Wl,-rpath,$ORIGIN",
        "-o",
        out / "device_context_test",
    )
    adb_command = [str(adb), "-s", args.serial]
    remote = subprocess.check_output(
        [
            *adb_command,
            "shell",
            "mktemp",
            "-d",
            "/data/local/tmp/modular-hal-context.XXXXXX",
        ],
        text=True,
    ).strip()
    if not remote.startswith("/data/local/tmp/modular-hal-context.") or not all(
        c.isalnum() or c in "/.-" for c in remote
    ):
        raise RuntimeError(f"Unexpected remote directory: {remote!r}")
    try:
        run(
            *adb_command,
            "push",
            out / "device_context_test",
            *[out / name for name in libraries],
            remote + "/",
        )
        run(
            *adb_command,
            "shell",
            shlex.join(
                [
                    "env",
                    f"LD_LIBRARY_PATH={remote}",
                    f"MODULAR_DRIVER_PLUGINS=CPU@{remote}/CPUPlugin.so;OpenCL@{remote}/OpenCLPlugin.so",
                    *(
                        [
                            f"MODULAR_OPENCL_DEBUG={os.environ['MODULAR_OPENCL_DEBUG']}"
                        ]
                        if os.environ.get("MODULAR_OPENCL_DEBUG")
                        else []
                    ),
                    remote + "/device_context_test",
                ]
            ),
        )
    finally:
        run(*adb_command, "shell", "rm", "-rf", remote)


if __name__ == "__main__":
    main()
