#!/usr/bin/env python3
# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the Apache License v2.0 with LLVM Exceptions: https://llvm.org/LICENSE.txt
"""Capture calibrated interface bytes and thread counts in a bounded GPU window.

Invoke through ../../with_gpu_counters.py so the profiling switch is restored.
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

from iterate import ADB, HERE, NDK, OPENCL_HEADERS, OPENCL_LIBRARY, SERIAL


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build", type=Path, default=Path("/tmp/projection-followup")
    )
    parser.add_argument("--candidate", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/projection-final-counters.json"),
    )
    args = parser.parse_args()
    out = Path("/tmp/projection-counter-capture")
    out.mkdir(exist_ok=True)
    uapi = Path(
        os.environ.get(
            "PROJECTION_KGSL_UAPI",
            str(Path.home() / "Library/Caches/modular-android-integration"),
        )
    )
    compiler = next(
        (NDK / "toolchains/llvm/prebuilt").glob(
            "*/bin/aarch64-linux-android33-clang++"
        )
    )
    subprocess.run(
        [
            str(compiler),
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-DPROJECTION_KGSL_COUNTERS",
            "-I",
            str(OPENCL_HEADERS),
            "-I",
            str(uapi),
            str(HERE / "bench.cpp"),
            str(OPENCL_LIBRARY),
            "-o",
            str(out / "bench"),
        ],
        check=True,
    )
    adb = [str(ADB), "-s", SERIAL]
    remote = "/data/local/tmp/modular-projection-counters"

    def call(*cmd: str, **kw) -> str:
        return subprocess.run(
            [*adb, *cmd], text=True, capture_output=True, check=True, **kw
        ).stdout

    call("shell", f"mkdir -p {remote}; input keyevent KEYCODE_WAKEUP")
    sources = {
        "original": args.build / "original.cl",
        "control": args.build / "control.cl",
        "load": Path("/tmp/packed-projection-bench/group_major_load.cl"),
        "winner": args.build / f"{args.candidate}.cl",
        "halfbits": Path("/tmp/projection-iterate/halfbits.cl"),
    }
    for name, path in sources.items():
        call("push", str(path), remote + "/" + name + ".cl")
    for path in [out / "bench", *args.build.glob("*.so")]:
        call("push", str(path), remote + "/" + path.name)
    call("shell", f"chmod 755 {remote}/bench")
    records = []
    cases = [
        (rep, shape, name)
        for rep in range(2)
        for shape in ["6144x2048", "2048x2048", "2048x6144"]
        for name in (
            list(sources)[:4]
            if rep % 2 == 0
            else list(reversed(list(sources)[:4]))
        )
    ]
    cases.append((0, "6144x2048", "halfbits"))
    for repeat, shape, name in cases:
        width, k = map(int, shape.split("x"))
        bad = name == "halfbits"
        banks = 1 if bad else math.ceil(128 * 1024**2 / (width * k // 4))
        iterations = 1 if bad else max(256, banks)
        extra = "PROJECTION_COUNTERS=1 "
        if bad:
            extra += "PROJECTION_WARMUP=1 "
        if name == "winner":
            from iterate import tuning_config

            cfg = tuning_config(args.candidate)
            extra += f"PROJECTION_TILE={cfg['t']} PROJECTION_GROUPS={cfg['g']} PROJECTION_WORKGROUP={cfg['w']} "
            if cfg.get("p"):
                extra += f"PROJECTION_PACK_OUTPUT={cfg['p']} "
        layout = "baseline" if name == "original" else "group_major"
        mode = "load" if name == "load" else "full"
        before = call("shell", "dumpsys thermalservice")
        raw = call(
            "shell",
            f"cd {remote} && timeout -s KILL 45 env LD_LIBRARY_PATH={remote} {extra}./bench original.cl {name}.cl {layout} {mode} {width} {k} {banks} {iterations} 123456789",
            timeout=55,
        )
        stages = [
            json.loads(line)
            for line in raw.splitlines()
            if line.startswith("{")
        ]
        counter = next(s for s in stages if s["stage"] == "counters")
        result = next(s for s in stages if s["stage"] == "result")
        record = {
            "repeat": repeat,
            "shape": shape,
            "variant": name,
            "stages": stages,
            "before": before,
            "after": call("shell", "dumpsys thermalservice"),
        }
        records.append(record)
        args.output.write_text(
            json.dumps(
                {
                    "candidate": args.candidate,
                    "host_sha256": hashlib.sha256(
                        (out / "bench").read_bytes()
                    ).hexdigest(),
                    "kernel_sha256": {
                        n: hashlib.sha256(p.read_bytes()).hexdigest()
                        for n, p in sources.items()
                    },
                    "uapi_sha256": hashlib.sha256(
                        (uapi / "hill-msm_kgsl.h").read_bytes()
                    ).hexdigest(),
                    "records": records,
                },
                indent=2,
            )
            + "\n"
        )
        print(
            repeat,
            shape,
            name,
            "valid=",
            counter["valid_invocations"],
            "read_MB=",
            counter["read_bytes"] / iterations / 1e6,
            "write_MB=",
            counter["write_bytes"] / iterations / 1e6,
            "ms=",
            result["device_mean_ms"],
            flush=True,
        )
        if not counter["valid_invocations"]:
            raise RuntimeError(
                "Invalid counter window; preserved, not accepted"
            )


if __name__ == "__main__":
    main()
