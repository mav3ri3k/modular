#!/usr/bin/env python3
# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the Apache License v2.0 with LLVM Exceptions: https://llvm.org/LICENSE.txt
"""Run interleaved projection experiments with bounded device processes."""

import argparse
import json
import math
import statistics
import subprocess
import time
from pathlib import Path

from generate import LAYOUTS

# (output width, reduction width, calls per decode token), excluding head.
SHAPES = [
    (6144, 2048, 66),
    (2048, 2048, 42),
    (2048, 6144, 24),
    (4096, 2048, 6),
    (512, 2048, 12),
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adb", type=Path, required=True)
    p.add_argument("--serial", required=True)
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--ring-mib", type=int, default=128)
    p.add_argument(
        "--layouts", nargs="+", choices=LAYOUTS, default=list(LAYOUTS)
    )
    p.add_argument(
        "--modes",
        nargs="+",
        choices=["full", "load", "unpack"],
        default=["load", "unpack", "full"],
    )
    p.add_argument(
        "--shapes",
        nargs="+",
        default=[
            "6144x2048",
            "2048x2048",
            "2048x6144",
            "4096x2048",
            "512x2048",
        ],
    )
    p.add_argument(
        "--regimes", nargs="+", choices=["hot", "ring"], default=["hot", "ring"]
    )
    p.add_argument("--iterations", type=int, default=512)
    p.add_argument("--pause", type=float, default=3)
    args = p.parse_args()
    adb = [str(args.adb), "-s", args.serial]
    remote = "/data/local/tmp/modular-projection-bench"
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def command(*cmd, **kw) -> str:
        return subprocess.run(
            [*adb, *map(str, cmd)],
            check=True,
            text=True,
            capture_output=True,
            **kw,
        ).stdout

    def state() -> str:
        raw = command("shell", "dumpsys thermalservice; dumpsys power")
        return "\n".join(
            line
            for line in raw.splitlines()
            if any(
                key in line
                for key in (
                    "Thermal Status:",
                    "mWakefulness=",
                    "mIsPowered=",
                    "mBatteryLevel=",
                    "Temperature{",
                )
            )
        )

    command("shell", f"mkdir -p {remote}")
    for path in [
        args.build / "bench",
        *args.build.glob("*.cl"),
        *args.build.glob("*.so"),
    ]:
        if path.name.startswith("host_"):
            continue
        command("push", path, remote + "/" + path.name)
    command("shell", f"chmod 755 {remote}/bench")
    # Wake without changing power/thermal policy or dismissing the lock screen.
    command("shell", "input keyevent KEYCODE_WAKEUP")
    records = []
    provenance = json.loads((args.build / "provenance.json").read_text())
    for repeat in range(args.repeats):
        layouts = (
            args.layouts if repeat % 2 == 0 else list(reversed(args.layouts))
        )
        for shape in args.shapes:
            width, reduction = map(int, shape.split("x"))
            weight_bytes = width * reduction // 4
            for regime in args.regimes:
                banks = (
                    1
                    if regime == "hot"
                    else math.ceil(args.ring_mib * 1024**2 / weight_bytes)
                )
                for mode in args.modes:
                    for layout in layouts:
                        before = state()
                        cmd = f"cd {remote} && timeout -s KILL 45 env LD_LIBRARY_PATH={remote} ./bench baseline_{mode}.cl {layout}_{mode}.cl {layout} {mode} {width} {reduction} {banks} {max(banks, args.iterations)} 123456789"
                        begin = time.time()
                        try:
                            raw = command("shell", cmd, timeout=55)
                            stages = [
                                json.loads(line)
                                for line in raw.splitlines()
                                if line.startswith("{")
                            ]
                            result = next(
                                s for s in stages if s["stage"] == "result"
                            )
                            record = {
                                "repeat": repeat,
                                "regime": regime,
                                "shape": shape,
                                "layout": layout,
                                "mode": mode,
                                "stages": stages,
                                "result": result,
                                "before": before,
                                "after": state(),
                                "wall_seconds": time.time() - begin,
                            }
                            print(
                                f"{repeat} {shape} {regime} {mode} {layout}: {result['device_mean_ms']:.6f} ms",
                                flush=True,
                            )
                        except (
                            subprocess.SubprocessError,
                            StopIteration,
                            ValueError,
                        ) as e:
                            record = {
                                "repeat": repeat,
                                "regime": regime,
                                "shape": shape,
                                "layout": layout,
                                "mode": mode,
                                "error": str(e),
                                "stdout": getattr(e, "stdout", ""),
                                "stderr": getattr(e, "stderr", ""),
                                "before": before,
                                "after": state(),
                            }
                            print(
                                f"FAILED {shape} {mode} {layout}: {record['stderr']}",
                                flush=True,
                            )
                        records.append(record)
                        args.output.write_text(
                            json.dumps(
                                {
                                    "provenance": provenance,
                                    "config": vars(args)
                                    | {
                                        "adb": str(args.adb),
                                        "build": str(args.build),
                                        "output": str(args.output),
                                    },
                                    "records": records,
                                },
                                indent=2,
                            )
                            + "\n"
                        )
                        if "error" in record:
                            raise RuntimeError("Probe failed; see saved output")
                        time.sleep(args.pause)
    summary = []
    for shape in args.shapes:
        for regime in args.regimes:
            for mode in args.modes:
                for layout in args.layouts:
                    samples = [
                        r["result"]["device_mean_ms"]
                        for r in records
                        if (r["shape"], r["regime"], r["mode"], r["layout"])
                        == (shape, regime, mode, layout)
                    ]
                    summary.append(
                        {
                            "shape": shape,
                            "regime": regime,
                            "mode": mode,
                            "layout": layout,
                            "median_mean_ms": statistics.median(samples),
                            "samples_ms": samples,
                        }
                    )
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
