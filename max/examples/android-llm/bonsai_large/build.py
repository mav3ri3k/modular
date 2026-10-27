#!/usr/bin/env python3
# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the Apache License v2.0 with LLVM Exceptions: https://llvm.org/LICENSE.txt
"""Build a generated Bonsai Mojo forward plus Android benchmark/validation runners.

--packages contains precompiled std.mojoc/max.mojoc compatible with --mojo.
--runtime-build uses the android_artifacts.json contract of ../build.py.
"""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ["mojo", "ndk", "packages", "runtime-build", "output"]:
        p.add_argument("--" + arg, type=Path, required=True)
    p.add_argument("--env-json", type=Path)
    a = p.parse_args()
    src = Path(__file__).resolve().parent.parent
    root = src.parents[2]
    env = dict(os.environ)
    if a.env_json:
        env.update(json.loads(a.env_json.read_text()))
    env["MODULAR_MOJO_MAX_IMPORT_PATH"] = str(a.packages.resolve())
    subprocess.run(
        [
            str(a.mojo),
            "build",
            str(src / "runtime/model_runtime.mojo"),
            "--target-triple",
            "aarch64-linux-android33",
            "--target-cpu",
            "generic",
            "--target-accelerator",
            "adreno-840",
            "-I",
            str(a.output),
            "-O3",
            "--emit",
            "object",
            "-o",
            str(a.output / "model.o"),
        ],
        env=env,
        check=True,
    )
    libs = json.loads((a.runtime_build / "android_artifacts.json").read_text())[
        "libraries"
    ]
    for lib in libs:
        dest = a.output / Path(lib).name
        if dest.exists():
            dest.unlink()
        shutil.copy2(a.runtime_build / lib, dest)
    clang = list(
        (a.ndk / "toolchains/llvm/prebuilt").glob(
            "*/bin/aarch64-linux-android33-clang++"
        )
    )[0]
    for main, name in [
        ("inference_test/bench.cpp", "bench"),
        ("inference_test/validate.cpp", "validate"),
        ("runner.cpp", "runner"),
    ]:
        subprocess.run(
            [
                str(clang),
                "-DQWEN_MODEL",
                "-DBONSAI_MODEL",
                "-O2",
                "-std=c++17",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-I",
                str(root / "Target/include"),
                "-I",
                str(src),
                "-I",
                str(a.output),
                str(src / main),
                str(src / "engine.cpp"),
                str(a.output / "model.o"),
                "-L" + str(a.output),
                "-lKGENCompilerRTShared",
                "-Wl,-rpath,$ORIGIN",
                "-ldl",
                "-o",
                str(a.output / name),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
