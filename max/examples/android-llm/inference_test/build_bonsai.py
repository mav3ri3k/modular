#!/usr/bin/env python3
# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the Apache License v2.0 with LLVM Exceptions: https://llvm.org/LICENSE.txt
"""Link 2B benchmark and validation runners against a staged Android build.

Use --build with the output of android-llm/build.py, or --native for the
historical app layout. Neither option rebuilds the Mojo model.
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--build", type=Path)
    source.add_argument("--native", type=Path)
    for name in ["ndk", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    src = Path(__file__).resolve().parent.parent
    root = src.parents[2]
    if a.build:
        runtime_names = json.loads(
            (a.build / "android_artifacts.json").read_text()
        )["libraries"]
        libraries = [a.build / name for name in runtime_names]
        model = a.build / "model.o"
    else:
        # Match the historical app packaging, ignoring obsolete plugin copies.
        runtime_names = json.loads(
            (a.native / "engine/runtime.json").read_text()
        )["libraries"]
        libraries = [a.native / "engine/lib" / name for name in runtime_names]
        libraries += list((a.native / "engine/plugins/arm64-v8a").glob("*.so"))
        model = a.native / "engine/model.o"
    for lib in libraries:
        dest = a.output / lib.name
        if dest.exists():
            dest.unlink()
        shutil.copy2(lib, dest)
    clang = list(
        (a.ndk / "toolchains/llvm/prebuilt").glob(
            "*/bin/aarch64-linux-android33-clang++"
        )
    )[0]
    cpp = (
        clang.parent.parent
        / "sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"
    )
    shutil.copy2(cpp, a.output / cpp.name)
    for source, name in [
        ("inference_test/bench.cpp", "bench"),
        ("inference_test/validate.cpp", "validate"),
        ("runner.cpp", "runner"),
    ]:
        subprocess.run(
            [
                str(clang),
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
                str(src / source),
                str(src / "engine.cpp"),
                str(model),
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
