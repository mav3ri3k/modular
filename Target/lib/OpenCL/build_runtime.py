#!/usr/bin/env python3
# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the MAX License: https://modular.com/legal/max-license
"""Build and stage the normal Mojo runtime and SDK plugins for Android."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ndk", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bazel-arg", action="append", default=[])
    parser.add_argument(
        "--include-max-tool",
        action="store_true",
        help="Also stage the normal MAX mt executable and graph runtime",
    )
    parser.add_argument(
        "--include-max-c-api",
        action="store_true",
        help="Also stage the normal MAX C API and graph runtime",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    ndk, output = args.ndk.resolve(), args.output.resolve()
    subprocess.run(
        [
            str(root / "bazelw"),
            "build",
            "--compilation_mode=opt",
            "--config=android-arm64",
            f"--repo_env=ANDROID_NDK_HOME={ndk}",
            *args.bazel_arg,
            "//Mojo:CompilerRT",
            "//Target:CPUPlugin.so",
            "//Target:OpenCLPlugin.so",
            *(["//GraphCompiler/tools/mt:mt"] if args.include_max_tool else []),
            *(["//max/lib/max_license:max"] if args.include_max_c_api else []),
        ],
        cwd=root,
        check=True,
    )
    libraries = {
        "runtime/libKGENCompilerRTShared.so": "Mojo/libKGENCompilerRTShared.so",
        "runtime/libAsyncRTRuntimeGlobals.so": (
            "AsyncRT/libAsyncRTRuntimeGlobals.so"
        ),
        "runtime/libMSupportGlobals.so": "Support/libMSupportGlobals.so",
        "hal/CPUPlugin.so": "Target/CPUPlugin.so",
        "hal/OpenCLPlugin.so": "Target/OpenCLPlugin.so",
    }
    tools = {}
    if args.include_max_tool or args.include_max_c_api:
        libraries.update(
            {
                "runtime/libHALDeviceContextMojo.so": (
                    "MLRT/libHALDeviceContextMojo.so"
                ),
                "runtime/libAsyncRTMojoBindings.so": (
                    "MLRT/libAsyncRTMojoBindings.so"
                ),
                "runtime/libMGPRT.so": "GraphCompiler/libMGPRT.so",
            }
        )
    if args.include_max_tool:
        tools["bin/mt"] = "GraphCompiler/tools/mt/mt"
    if args.include_max_c_api:
        libraries["runtime/libmax.so"] = "max/lib/max_license/libmax.so"
    for destination, source in (libraries | tools).items():
        path = output / destination
        path.parent.mkdir(parents=True, exist_ok=True)
        # Bazel outputs are read-only; replace a previous staged copy first.
        path.unlink(missing_ok=True)
        shutil.copy2(root / "bazel-bin" / source, path)
    cpp = list(
        (ndk / "toolchains/llvm/prebuilt").glob(
            "*/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"
        )
    )
    if len(cpp) != 1:
        raise RuntimeError("Expected one NDK libc++ runtime")
    cpp_destination = "runtime/libc++_shared.so"
    path = output / cpp_destination
    path.unlink(missing_ok=True)
    shutil.copy2(cpp[0], path)
    (output / "android_artifacts.json").write_text(
        json.dumps(
            {
                "android_api": 33,
                "libraries": [*libraries, cpp_destination],
                "tools": list(tools),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
