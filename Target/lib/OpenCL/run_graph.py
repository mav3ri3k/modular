#!/usr/bin/env python3
# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the MAX License: https://modular.com/legal/max-license
"""Deploy and execute a precompiled MEF with the normal Android MAX tool."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--adb", default="adb")
    parser.add_argument("--mt-arg", action="append", default=[])
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        default=[],
        help="Raw tensor file, repeated in graph input order",
    )
    parser.add_argument(
        "--output-dir", type=Path, help="Pull mt binary results here"
    )
    args = parser.parse_args()
    runtime = args.runtime.resolve()
    manifest = json.loads((runtime / "android_artifacts.json").read_text())
    if "bin/mt" not in manifest.get("tools", []):
        parser.error("Build the runtime with --include-max-tool first")
    adb = [args.adb, "-s", args.serial]
    remote = subprocess.check_output(
        [
            *adb,
            "shell",
            "mktemp",
            "-d",
            "/data/local/tmp/modular-max-graph.XXXXXX",
        ],
        text=True,
    ).strip()
    if not re.fullmatch(
        r"/data/local/tmp/modular-max-graph\.[A-Za-z0-9]+", remote
    ):
        raise RuntimeError(f"Unexpected remote directory: {remote!r}")
    try:
        subprocess.run(
            [
                *adb,
                "push",
                str(runtime / "bin/mt"),
                *[str(runtime / name) for name in manifest["libraries"]],
                remote + "/",
            ],
            check=True,
        )
        subprocess.run(
            [*adb, "push", str(args.model.resolve()), remote + "/model.mef"],
            check=True,
        )
        for index, path in enumerate(args.input):
            subprocess.run(
                [*adb, "push", str(path.resolve()), f"{remote}/input{index}"],
                check=True,
            )
        command = [
            "env",
            "MODULAR_CRASH_REPORTING_ENABLED=false",
            f"LD_LIBRARY_PATH={remote}",
            f"MODULAR_MOJO_MAX_COMPILERRT_PATH={remote}/libKGENCompilerRTShared.so",
            f"MODULAR_MOJO_MAX_MGPRT_PATH={remote}/libMGPRT.so",
            f"MODULAR_DRIVER_PLUGINS=CPU@{remote}/CPUPlugin.so;OpenCL@{remote}/OpenCLPlugin.so",
            remote + "/mt",
            "--execute",
            remote + "/model.mef",
            "--model-inputs=ones",
            *(
                [
                    "--result-output-style=binary",
                    f"--binary-output-dir={remote}/outputs",
                ]
                if args.output_dir
                else ["--result-output-style=full"]
            ),
            *[
                f"--input-data={i}::{remote}/input{i}"
                for i in range(len(args.input))
            ],
            *args.mt_arg,
        ]
        subprocess.run([*adb, "shell", shlex.join(command)], check=True)
        if args.output_dir:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    *adb,
                    "pull",
                    remote + "/outputs/.",
                    str(args.output_dir.resolve()),
                ],
                check=True,
            )
    finally:
        subprocess.run([*adb, "shell", "rm", "-rf", remote], check=True)


if __name__ == "__main__":
    main()
