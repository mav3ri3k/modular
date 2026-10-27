#!/usr/bin/env python3
# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the Apache License v2.0 with LLVM Exceptions: https://llvm.org/LICENSE.txt
"""Build a Mojo DeviceContext model with embedded GPU kernels for Android."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(*args, **kwargs) -> None:
    subprocess.run([str(a) for a in args], check=True, **kwargs)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--variant", choices=["smollm2", "bonsai"], default="smollm2"
    )
    p.add_argument("--mojo", type=Path, required=True)
    p.add_argument("--ndk", type=Path, required=True)
    p.add_argument("--bazel-arg", action="append", default=[])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--target-accelerator",
        choices=["opencl-generic", "adreno-840"],
        default="adreno-840",
    )
    args = p.parse_args()
    source = Path(__file__).resolve().parent
    root = source.parents[2]
    model_source = source / "bonsai" if args.variant == "bonsai" else source
    mojo = args.mojo.resolve()
    ndk = args.ndk.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
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
        source / "runtime/model_runtime.mojo",
        "--target-triple",
        "aarch64-linux-android33",
        "--target-cpu",
        "generic",
        "--target-accelerator",
        args.target_accelerator,
        "-I",
        model_source,
        "-O3",
        "--emit",
        "object",
        "-o",
        out / "model.o",
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
    toolchains = list(
        (ndk / "toolchains/llvm/prebuilt").glob(
            "*/bin/aarch64-linux-android33-clang++"
        )
    )
    if len(toolchains) != 1:
        raise RuntimeError("Expected one NDK host toolchain")
    run(
        toolchains[0],
        "-DBONSAI_MODEL" if args.variant == "bonsai" else "-DSMOLLM2_MODEL",
        "-O2",
        "-std=c++17",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-I",
        root / "SDK/include",
        "-I",
        root / "Target/include",
        source / "runner.cpp",
        source / "engine.cpp",
        out / "model.o",
        f"-L{out / 'runtime'}",
        "-lKGENCompilerRTShared",
        "-Wl,-rpath,$ORIGIN",
        "-ldl",
        "-o",
        out / "runner",
    )


if __name__ == "__main__":
    main()
