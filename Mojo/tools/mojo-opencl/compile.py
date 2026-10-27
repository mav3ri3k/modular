#!/usr/bin/env python3
"""Compile a Mojo GPU module to a versioned OpenCL bundle."""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--mojo", type=Path, required=True)
    parser.add_argument("--import-path", type=Path, action="append", default=[])
    args = parser.parse_args()
    repo = Path(
        os.environ.get(
            "BUILD_WORKSPACE_DIRECTORY", Path(__file__).resolve().parents[3]
        )
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    mojo = str(args.mojo.resolve())
    with tempfile.TemporaryDirectory(
        prefix=".mojo-opencl-", dir=output.parent
    ) as temporary:
        work = Path(temporary)
        # Rebuild the stdlib with this target's hooks using exactly the selected
        # compiler. Serialized stdlib modules are compiler-version specific.
        subprocess.run(
            [
                mojo,
                "precompile",
                str(repo / "Mojo/stdlib/std"),
                "-o",
                str(work / "std.mojoc"),
            ],
            check=True,
        )
        env = os.environ.copy()
        previous = env.get("MODULAR_MOJO_MAX_IMPORT_PATH", "")
        env["MODULAR_MOJO_MAX_IMPORT_PATH"] = str(work) + (
            os.pathsep + previous if previous else ""
        )
        intermediate = work / "kernels.mcl"
        command = [
            mojo,
            "build",
            str(args.source.resolve()),
            "--target-triple",
            "aarch64-unknown-opencl",
            "--target-cpu",
            "generic",
            "--emit",
            "object",
            "-o",
            str(intermediate),
        ]
        for path in args.import_path:
            command.extend(["-I", str(path.resolve())])
        subprocess.run(command, env=env, check=True)
        artifact = intermediate
        if artifact.read_bytes()[:8] != b"MOJOCL03":
            raise RuntimeError(
                "Compiler did not produce an OpenCL ABI v3 bundle"
            )
        artifact.replace(output)
    print(f"Wrote OpenCL ABI v3 bundle: {output}")


if __name__ == "__main__":
    main()
