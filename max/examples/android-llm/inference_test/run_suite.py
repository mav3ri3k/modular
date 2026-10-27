#!/usr/bin/env python3
# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the Apache License v2.0 with LLVM Exceptions: https://llvm.org/LICENSE.txt
"""Interleave real llama-bench and the Mojo forward-only port on one phone.

Device layout: root/prism/{llama-bench, libraries}, root/<model>/{bench,
weights.bin, plugin and libraries}, root/<GGUF filename>. Upload outside timing.
"""

import argparse
import datetime
import json
import shlex
import statistics
import subprocess
import time
from pathlib import Path

MODELS = {
    "bonsai": "Ternary-Bonsai-2-2B-PQ2_0.gguf",
    "bonsai4": "Ternary-Bonsai-2-4B-PTQ1_0.gguf",
    "bonsai9": "Ternary-Bonsai-2-9B-PTQ1_0.gguf",
    "qwen3": "Qwen_Qwen3-1.7B-Q4_K_M.gguf",
    "qwen35": "Qwen_Qwen3.5-2B-Q4_K_M.gguf",
}
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--adb", default="adb")
p.add_argument("--serial", required=True)
p.add_argument(
    "--reference-subdir",
    default="prism",
    help="Reference backend directory beneath --root",
)
p.add_argument("--root", default="/data/local/tmp/modular-inference-testing")
p.add_argument("--output", type=Path, required=True)
p.add_argument(
    "--models",
    nargs="+",
    choices=list(MODELS),
    default=["bonsai", "qwen3", "qwen35"],
)
p.add_argument(
    "--reference-model",
    action="append",
    default=[],
    metavar="MODEL=FILENAME",
    help="Override a reference GGUF after verifying numerical weight parity",
)
p.add_argument(
    "--reference-env",
    action="append",
    default=[],
    metavar="KEY=VALUE",
    help="Explicit environment for the native reference only",
)
p.add_argument("--repeats", type=int, default=3)
p.add_argument(
    "--samples-per-run",
    type=int,
    default=1,
    help="Native repetitions per process, with one warmup",
)
p.add_argument("--cooldown", type=float, default=10)
p.add_argument(
    "--skip-batched",
    action="store_true",
    help="Skip the batched prompt control",
)
p.add_argument(
    "--tests",
    nargs="+",
    choices=["tg128", "pp512", "pg128-128", "tg128-d512"],
    default=["tg128", "pp512"],
)
a = p.parse_args()
for override in a.reference_model:
    name, filename = override.split("=", 1)
    if name not in MODELS:
        raise ValueError("unknown model: " + name)
    MODELS[name] = filename
a.output.mkdir(parents=True, exist_ok=True)
adb = [a.adb, "-s", a.serial]


def shell(command: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [*adb, "shell", command],
        text=True,
        capture_output=True,
    )
    if result.returncode:
        (a.output / "failure.stdout").write_text(result.stdout)
        (a.output / "failure.stderr").write_text(result.stderr)
        raise RuntimeError(
            f"Android command failed ({result.returncode}): {command}"
        )
    return result


def temperature() -> str:
    return shell(
        "getprop ro.product.model; getprop ro.soc.model; getprop ro.build.version.sdk; cat /proc/uptime; dumpsys battery; dumpsys thermalservice"
    ).stdout


records = []
# JSON journal is updated after every completed sample; failed runs remain logs.
for test in a.tests:
    pp, tg, depth = {
        "tg128": (0, 128, 0),
        "pp512": (512, 0, 0),
        "pg128-128": (128, 128, 0),
        "tg128-d512": (0, 128, 512),
    }[test]
    for model in a.models:
        for repeat in range(a.repeats):
            variants = (
                ["prism", "modular"]
                if (repeat + a.tests.index(test) + a.models.index(model)) % 2
                == 0
                else ["modular", "prism"]
            )
            # Practical batched prompt baseline in addition to matched batch-one tests.
            if pp and repeat == 0 and not a.skip_batched:
                variants = ["prism-batched"] + variants
            for variant in variants:
                tag = f"{model}-{test}-{repeat}-{variant}"
                time.sleep(a.cooldown)
                (a.output / (tag + "-thermal-before.txt")).write_text(
                    temperature()
                )
                if variant.startswith("prism"):
                    args = [
                        "./llama-bench",
                        "-m",
                        "../" + MODELS[model],
                        "-p",
                        str(pp),
                        "-n",
                        str(tg),
                        "-d",
                        str(depth),
                        "-r",
                        str(a.samples_per_run),
                        "-t",
                        "4",
                        "-ngl",
                        "99",
                        "-o",
                        "json",
                    ]
                    if pp and tg:
                        args = [
                            "./llama-bench",
                            "-m",
                            "../" + MODELS[model],
                            "-p",
                            "0",
                            "-n",
                            "0",
                            "-pg",
                            f"{pp},{tg}",
                            "-d",
                            str(depth),
                            "-r",
                            str(a.samples_per_run),
                            "-t",
                            "4",
                            "-ngl",
                            "99",
                            "-o",
                            "json",
                        ]
                    args += [
                        "-b",
                        "2048" if variant == "prism-batched" else "1",
                        "-ub",
                        "512" if variant == "prism-batched" else "1",
                    ]
                    # Keep reference defaults: this llama-bench CLI rejects f32 cache types.
                    # Mojo retains FP32 caches; record this numerical/storage difference.
                    cwd = a.root + "/" + a.reference_subdir
                else:
                    plugin = (
                        "./libOpenCLPlugin.so"
                        if model == "bonsai"
                        else "./OpenCLPlugin.so"
                    )
                    args = [
                        "./bench",
                        plugin,
                        "./weights.bin",
                        str(pp),
                        str(tg),
                        str(depth),
                        str(a.samples_per_run),
                        "1",
                    ]
                    cwd = a.root + "/" + model
                if variant.startswith("prism") and a.reference_env:
                    args = ["env", *a.reference_env, *args]
                command = (
                    "cd "
                    + shlex.quote(cwd)
                    + " && LD_LIBRARY_PATH=. "
                    + shlex.join(args)
                )
                print(datetime.datetime.now().isoformat(), tag, flush=True)
                result = shell(command)
                (a.output / (tag + ".stdout")).write_text(result.stdout)
                (a.output / (tag + ".stderr")).write_text(result.stderr)
                if variant.startswith("prism"):
                    metrics = json.loads(result.stdout)[0]
                else:
                    metrics = json.loads(
                        next(
                            line[7:]
                            for line in result.stdout.splitlines()
                            if line.startswith("RESULT ")
                        )
                    )
                records.append(
                    dict(
                        model=model,
                        test=test,
                        repeat=repeat,
                        variant=variant,
                        command=command,
                        metrics=metrics,
                    )
                )
                (a.output / "samples.json").write_text(
                    json.dumps(records, indent=2) + "\n"
                )
                (a.output / (tag + "-thermal-after.txt")).write_text(
                    temperature()
                )
                print("MEASURED", tag, metrics["avg_ts"], flush=True)
summary = []
for model in a.models:
    for test in a.tests:
        for variant in ["prism", "modular", "prism-batched"]:
            rates = [
                1e9 * (r["metrics"]["n_prompt"] + r["metrics"]["n_gen"]) / ns
                for r in records
                if r["model"] == model
                and r["test"] == test
                and r["variant"] == variant
                for ns in r["metrics"]["samples_ns"]
            ]
            if rates:
                summary.append(
                    dict(
                        model=model,
                        test=test,
                        variant=variant,
                        mean=statistics.mean(rates),
                        stddev=statistics.stdev(rates) if len(rates) > 1 else 0,
                        median=statistics.median(rates),
                        samples=rates,
                    )
                )
(a.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print("COMPLETE", flush=True)
