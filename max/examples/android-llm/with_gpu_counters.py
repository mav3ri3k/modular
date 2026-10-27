#!/usr/bin/env python3
# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the Apache License v2.0 with LLVM Exceptions: https://llvm.org/LICENSE.txt
"""Run a profiling command with KGSL counter reads temporarily enabled.

Example: with_gpu_counters.py --serial SERIAL -- python profiling_command.py
This enables access only; the command must allocate/read/release its counters.
"""

import argparse
import subprocess


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--adb", default="adb")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("Provide a profiling command after --")
    adb = [args.adb, "-s", args.serial]
    path = "/sys/class/kgsl/kgsl-3d0/perfcounter"

    def read() -> str:
        return subprocess.check_output(
            [*adb, "shell", "cat", path], text=True
        ).strip()

    original = read()
    if original not in ("0", "1"):
        raise RuntimeError(f"Unexpected KGSL counter state: {original!r}")
    if original == "1":
        return subprocess.run(command).returncode
    try:
        subprocess.run([*adb, "shell", f"printf '1\\n' > {path}"], check=True)
        if read() != "1":
            raise RuntimeError("KGSL profiling switch did not enable")
        print("KGSL counter reads enabled for this command", flush=True)
        return subprocess.run(command).returncode
    finally:
        subprocess.run([*adb, "shell", f"printf '0\\n' > {path}"], check=True)
        if read() != original:
            raise RuntimeError("Failed to restore the KGSL profiling switch")
        print("Restored KGSL profiling switch to 0", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
