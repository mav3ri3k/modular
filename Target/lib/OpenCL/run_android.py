#!/usr/bin/env python3
"""Cross-build the OpenCL HAL plugin and run its tests on one explicit device."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def run(*args: str | Path) -> None:
    subprocess.run([str(arg) for arg in args], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ndk", type=Path, required=True)
    parser.add_argument("--opencl-headers", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--adb", type=Path)
    parser.add_argument(
        "--view-bundle",
        type=Path,
        help="Test the compiled ABI-v3 buffer_views fixture",
    )
    args = parser.parse_args()
    ndk = args.ndk.resolve()
    build = args.build_dir.resolve()
    adb = (
        args.adb or shutil.which("adb") or ndk.parents[1] / "platform-tools/adb"
    )
    source = Path(__file__).resolve().parent
    run(
        "cmake",
        "-S",
        source,
        "-B",
        build,
        f"-DCMAKE_TOOLCHAIN_FILE={ndk}/build/cmake/android.toolchain.cmake",
        "-DANDROID_ABI=arm64-v8a",
        "-DANDROID_PLATFORM=android-31",
        "-DANDROID_STL=c++_static",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DOPENCL_INCLUDE_DIR={args.opencl_headers.resolve()}",
    )
    run("cmake", "--build", build, "--parallel", "8")
    adb_command = [str(adb), "-s", args.serial]
    # Each run owns its directory and never removes another test's artifacts.
    remote = subprocess.check_output(
        [
            *adb_command,
            "shell",
            "mktemp",
            "-d",
            "/data/local/tmp/modular-opencl.XXXXXX",
        ],
        text=True,
    ).strip()
    if (
        not remote.startswith("/data/local/tmp/modular-opencl.")
        or not remote.replace("/", "")
        .replace("-", "")
        .replace(".", "")
        .isalnum()
    ):
        raise RuntimeError(f"Unexpected remote directory: {remote!r}")
    try:
        run(
            *adb_command,
            "push",
            build / "OpenCLPlugin.so",
            build / "opencl_hal_smoke",
            remote + "/",
        )
        run(
            *adb_command,
            "shell",
            remote + "/opencl_hal_smoke",
            remote + "/OpenCLPlugin.so",
        )
        if args.view_bundle:
            run(
                *adb_command,
                "push",
                build / "opencl_buffer_views",
                remote + "/",
            )
            run(
                *adb_command,
                "push",
                args.view_bundle.resolve(),
                remote + "/views.mcl",
            )
            run(
                *adb_command,
                "shell",
                remote + "/opencl_buffer_views",
                remote + "/OpenCLPlugin.so",
                remote + "/views.mcl",
            )
    finally:
        run(*adb_command, "shell", "rm", "-rf", remote)


if __name__ == "__main__":
    main()
