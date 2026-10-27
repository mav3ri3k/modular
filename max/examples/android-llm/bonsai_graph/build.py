#!/usr/bin/env python3
# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the Apache License v2.0 with LLVM Exceptions: https://llvm.org/LICENSE.txt
"""Cross-compile the Bonsai MAX graph and its normal C API runner.

--tools is a matching native compiler snapshot containing mojo, mo-opt, mt and
its precompiled packages. env.json may supply its runtime/import paths. The
Python frontend's Mojo compiler separately verifies the same custom-op source;
its package format need not match the local cross-compiler's format.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def run(*args, **kwargs) -> None:
    subprocess.run([str(arg) for arg in args], check=True, **kwargs)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tools", type=Path, required=True)
    p.add_argument("--analysis-mojo", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--ndk", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--layers", type=int, default=24)
    p.add_argument("--bazel-arg", action="append", default=[])
    args = p.parse_args()
    source = Path(__file__).resolve().parent
    root = source.parents[3]
    out, tools = args.output.resolve(), args.tools.resolve()
    out.mkdir(parents=True, exist_ok=True)
    native_env = dict(os.environ)
    if (tools / "env.json").exists():
        native_env.update(json.loads((tools / "env.json").read_text()))
    # precompile accepts packages. Stage the one shared source module as a
    # package without duplicating or modifying the authored kernel code.
    staged = out / "kernel_source"
    staged.mkdir(exist_ok=True)
    shutil.copyfile(
        source.parent / "bonsai/kernels.mojo", staged / "__init__.mojo"
    )
    for folder, compiler, env in (
        ("native_packages", tools / "mojo", native_env),
        ("analysis_packages", args.analysis_mojo.resolve(), dict(os.environ)),
    ):
        packages = out / folder
        packages.mkdir(exist_ok=True)
        for path, name in (
            (staged, "bonsai_kernels"),
            (source / "ops", "bonsai_ops"),
        ):
            run(
                compiler,
                "precompile",
                path,
                "-I",
                packages,
                "-o",
                packages / f"{name}.mojoc",
                env=env,
            )
    kernels = out / "native_packages/bonsai_ops.mojoc"
    run(
        sys.executable,
        source / "export_graph.py",
        "--model",
        args.model,
        "--kernels",
        kernels,
        "--analysis-kernels",
        out / "analysis_packages/bonsai_ops.mojoc",
        "--layers",
        args.layers,
        "--output",
        out / "bonsai.mlirbc",
    )
    cross = [
        "--virtual-cpu-target=triple=aarch64-linux-android33;cpu=generic",
        "--virtual-device-count=1",
        "--virtual-device-api=opencl",
        "--virtual-device-arch=adreno-840",
    ]
    run(
        tools / "mo-opt",
        out / "bonsai.mlirbc",
        "--mlir-disable-threading",
        *cross,
        (
            "--gc-pipeline=constant-fold=false default-device-label=gpu"
            " default-device-target-triple=aarch64-unknown-opencl"
            " default-device-target-arch=adreno-840"
            f" kernel-library-paths={kernels}"
        ),
        "-o",
        out / "bonsai-lowered.mlirbc",
        "--emit-bytecode",
        env=native_env,
    )
    run(
        tools / "mt",
        "--compile",
        out / "bonsai-lowered.mlirbc",
        "--input-format=mgp",
        *cross,
        "-o",
        out / "bonsai.mef",
        env=native_env,
    )
    run(
        sys.executable,
        root / "Target/lib/OpenCL/build_runtime.py",
        "--ndk",
        args.ndk,
        "--output",
        out,
        "--include-max-c-api",
        *[f"--bazel-arg={arg}" for arg in args.bazel_arg],
    )
    clang = list(
        (args.ndk.resolve() / "toolchains/llvm/prebuilt").glob(
            "*/bin/aarch64-linux-android33-clang++"
        )
    )
    if len(clang) != 1:
        raise RuntimeError("Expected one NDK host toolchain")
    run(
        clang[0],
        "-DBONSAI_MODEL",
        "-O2",
        "-std=c++17",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-I",
        root / "max/include",
        source / "runner.cpp",
        f"-L{out / 'runtime'}",
        "-lmax",
        "-lKGENCompilerRTShared",
        "-Wl,-rpath,$ORIGIN",
        "-o",
        out / "graph_runner",
    )
    (out / "build_info.json").write_text(
        json.dumps(
            {
                "graph": "bonsai.mef",
                "kernel_source_sha256": hashlib.sha256(
                    (staged / "__init__.mojo").read_bytes()
                ).hexdigest(),
                "graph_sha256": hashlib.sha256(
                    (out / "bonsai.mef").read_bytes()
                ).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
