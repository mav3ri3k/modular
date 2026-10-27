#!/usr/bin/env python3
# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the Apache License v2.0 with LLVM Exceptions: https://llvm.org/LICENSE.txt
"""Interleave validated native variants on one device with one model upload.

All timed runs use the same prompt, forced decode length, reference logits and
sampling. Program compilation/uploads are outside runner timing. Optional
Hexagon variants use process-local backend power/latency votes; no global clock
or thermal settings are changed. --variant LABEL=BUILD may be repeated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import statistics
import subprocess
import time
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", action="append", required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--serial", required=True)
    p.add_argument("--adb", default="adb")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--hexagon-build", action="append", default=[], metavar="LABEL=DIR"
    )
    p.add_argument(
        "--hexagon-rows", action="append", default=[], metavar="LABEL=ROWS"
    )
    p.add_argument(
        "--cooldown", type=float, default=0, help="Idle seconds between runs"
    )
    p.add_argument(
        "--generation-only",
        action="store_true",
        help=(
            "Time fixed-length generation with token-ID checks, excluding"
            " full-logit validation"
        ),
    )
    args = p.parse_args()
    if not 1 <= args.repeats <= 50:
        p.error("Choose 1-50 measured repeats")
    if not 0 <= args.cooldown <= 60:
        p.error("Choose a cooldown between 0 and 60 seconds")
    variants = {}
    for variant in args.variant:
        label, directory = variant.split("=", 1)
        if not re.fullmatch(r"[A-Za-z0-9_-]+", label) or label in variants:
            p.error(
                "Variant labels must be unique alphanumeric/underscore/dash"
                " names"
            )
        variants[label] = Path(directory).resolve()
    hexagon = {}
    for value in args.hexagon_build:
        label, directory = value.split("=", 1)
        if label not in variants or label in hexagon:
            p.error("Hexagon build labels must be unique existing variants")
        hexagon[label] = Path(directory).resolve()
    rows = {}
    for value in args.hexagon_rows:
        label, count = value.split("=", 1)
        if label not in hexagon or label in rows:
            p.error("Hexagon row labels must be unique Hexagon variants")
        rows[label] = int(count)
        if not 128 <= rows[label] <= 248320 or rows[label] % 128:
            p.error("Hexagon rows must be a multiple of 128 in [128, 248320]")
    if set(rows) != set(hexagon):
        p.error("Supply --hexagon-rows for every --hexagon-build variant")
    variant_assets = {
        label: [
            "runner",
            *json.loads((directory / "android_artifacts.json").read_text())[
                "libraries"
            ],
        ]
        for label, directory in variants.items()
    }
    metadata = json.loads((args.model / "manifest.json").read_text())
    if metadata["generation_tokens"] < 2:
        p.error("A throughput fixture needs at least two output tokens")
    adb = [args.adb, "-s", args.serial]

    def command(*parts: str) -> str:
        try:
            return subprocess.check_output(
                [*adb, *parts], text=True, stderr=subprocess.STDOUT
            )
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                f"Android benchmark command failed:\n{error.output}"
            ) from error

    remote = command(
        "shell", "mktemp", "-d", "/data/local/tmp/modular-bench.XXXXXX"
    ).strip()
    if not re.fullmatch(r"/data/local/tmp/modular-bench\.[A-Za-z0-9]+", remote):
        raise RuntimeError(f"Unexpected directory {remote!r}")
    if (
        args.generation_only
        and len(metadata.get("expected_ids", []))
        != metadata["generation_tokens"]
    ):
        p.error("Generation-only timing requires all reference token IDs")
    records = []
    logs = args.output.with_suffix("")
    logs.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                *adb,
                "push",
                *[
                    str(args.model / name)
                    for name in (
                        [
                            "weights.bin",
                            "prompt.txt",
                        ]
                        if args.generation_only
                        else [
                            "weights.bin",
                            "prompt.txt",
                            "reference.f32",
                        ]
                    )
                ],
                remote + "/",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        for label, directory in variants.items():
            command("shell", "mkdir", remote + "/" + label)
            subprocess.run(
                [
                    *adb,
                    "push",
                    *[str(directory / name) for name in variant_assets[label]],
                    remote + "/" + label + "/",
                ],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            if label in hexagon:
                subprocess.run(
                    [
                        *adb,
                        "push",
                        *[
                            str(hexagon[label] / name)
                            for name in (
                                "HexagonProjection.so",
                                "libmodular_hexagon.so",
                            )
                        ],
                        remote + "/" + label + "/",
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                )

        def sample(label: str, warmup: bool) -> None:
            folder = remote + "/" + label
            if args.cooldown:
                time.sleep(args.cooldown)
            thermal = command("shell", "dumpsys", "thermalservice")
            thermal_status = re.search(r"Thermal Status: (\d+)", thermal)
            backend_env = []
            if label in hexagon:
                backend_env = [
                    f"ADSP_LIBRARY_PATH={folder};/vendor/lib64/rfs/dsp;/vendor/lib/rfsa/adsp;/vendor/dsp",
                    f"MODULAR_HEXAGON_HEAD={folder}/HexagonProjection.so",
                    f"MODULAR_HEXAGON_HEAD_COLUMNS={rows[label]}",
                    "MODULAR_HEXAGON_QOS=1",
                ]
            output = command(
                "shell",
                shlex.join(
                    [
                        "env",
                        f"LD_LIBRARY_PATH={folder}",
                        *backend_env,
                        folder + "/runner",
                        folder + "/OpenCLPlugin.so",
                        remote + "/weights.bin",
                        remote + "/prompt.txt",
                        "benchmark"
                        if args.generation_only
                        else remote + "/reference.f32",
                        str(metadata["generation_tokens"]),
                    ]
                ),
            )
            log = logs / f"{len(records):03d}-{label}.log"
            log.write_text(output)
            if label in hexagon and not re.search(
                rf"(?:Hexagon|HMX)={rows[label]}\b", output
            ):
                raise RuntimeError(f"Hexagon routing was not confirmed: {log}")
            match = re.search(
                r"Decode: (\d+) forwards, ([\d.]+) s, ([\d.]+) tok/s", output
            )
            errors = re.search(
                r"max_abs_error=([\deE.+-]+) relative_l2=([\deE.+-]+)", output
            )
            ids = [
                int(token)
                for token in re.findall(
                    r"^TOKEN \d+ (\d+)$", output, re.MULTILINE
                )
            ]
            if args.generation_only:
                valid = ids == metadata["expected_ids"]
            else:
                valid = (
                    errors is not None
                    and "PASS: all generated tokens" in output
                )
            if not match or not valid:
                raise RuntimeError(f"Unvalidated run for {label}:\n{output}")
            result = dict(
                variant=label,
                warmup=warmup,
                forwards=int(match[1]),
                seconds=float(match[2]),
                tokens_per_second=float(match[3]),
                max_abs_error=float(errors[1]) if errors else None,
                relative_l2=float(errors[2]) if errors else None,
                thermal_status=int(thermal_status[1])
                if thermal_status
                else None,
                log=str(log),
            )
            records.append(result)
            print(json.dumps(result), flush=True)

        labels = list(variants)
        for label in labels:
            sample(label, True)
        for repeat in range(args.repeats):
            for label in labels if repeat % 2 == 0 else list(reversed(labels)):
                sample(label, False)
        medians = {
            label: statistics.median(
                r["tokens_per_second"]
                for r in records
                if r["variant"] == label and not r["warmup"]
            )
            for label in labels
        }
        report = dict(
            serial=args.serial,
            model=metadata["model"],
            prompt=metadata["prompt"],
            prompt_tokens=len(metadata["prompt_ids"]),
            output_tokens=metadata["generation_tokens"],
            repeats=args.repeats,
            validation_included=not args.generation_only,
            token_ids_validated=True,
            profiling=False,
            cooldown_seconds=args.cooldown,
            hexagon_rows=rows,
            hexagon_artifact_sha256={
                label: {
                    name: hashlib.sha256(
                        (directory / name).read_bytes()
                    ).hexdigest()
                    for name in (
                        "HexagonProjection.so",
                        "libmodular_hexagon.so",
                    )
                }
                for label, directory in hexagon.items()
            },
            variants={
                label: str(directory) for label, directory in variants.items()
            },
            artifact_sha256={
                label: {
                    name: hashlib.sha256(
                        (directory / name).read_bytes()
                    ).hexdigest()
                    for name in variant_assets[label]
                }
                for label, directory in variants.items()
            },
            runs=records,
            median_tokens_per_second=medians,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print("Medians:", json.dumps(medians), flush=True)
    finally:
        command("shell", "rm", "-rf", remote)


if __name__ == "__main__":
    main()
