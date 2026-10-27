#!/usr/bin/env python3
# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the Apache License v2.0 with LLVM Exceptions: https://llvm.org/LICENSE.txt
"""Validate/time Bonsai MAX on USB, optionally paired with the Mojo runner.

Uploads once, excludes initialization/prefill from decode timing, checks every
reference ID and (by default) every logit, and cleans the temporary device files.
No phone clock, thermal or power policy is changed.
"""

import argparse
import hashlib
import json
import re
import shlex
import statistics
import subprocess
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--compare-mojo", type=Path)
    p.add_argument("--serial", required=True)
    p.add_argument("--adb", default="adb")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--generation-only", action="store_true")
    p.add_argument(
        "--profile",
        action="store_true",
        help="One traced sample per variant; not an unprofiled benchmark",
    )
    p.add_argument(
        "--host-profile",
        action="store_true",
        help="Collect OpenCL HAL host phase timings; diagnostic only",
    )
    p.add_argument(
        "--flush-max",
        action="store_true",
        help="Flush each MAX OpenCL kernel enqueue; diagnostic experiment",
    )
    p.add_argument(
        "--runtime-trace",
        type=Path,
        help="Write the MAX AsyncRT/GraphRT time trace to this local path",
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not 1 <= args.repeats <= 50:
        p.error("Choose 1-50 repeats")
    b, model = args.build.resolve(), args.model.resolve()
    graph = json.loads((b / "bonsai.json").read_text())
    meta = json.loads((model / "manifest.json").read_text())
    if (
        graph["layers"] != 24
        or graph["capacity"] != meta["capacity"]
        or graph["checkpoint_sha256"] != meta["checkpoint_sha256"]
    ):
        p.error("Graph must match this full-model reference and capacity")
    if (
        len(meta["expected_ids"]) != meta["generation_tokens"]
        or len(meta["expected_ids"]) < 2
    ):
        p.error("A fixed-length comparison needs all reference token IDs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    adb = [args.adb, "-s", args.serial]

    def command(*parts):
        try:
            return subprocess.check_output(
                [*adb, *parts], text=True, stderr=subprocess.STDOUT
            )
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                f"Device command failed:\n{error.output[-16000:]}"
            ) from error

    remote = command(
        "shell", "mktemp", "-d", "/data/local/tmp/bonsai-max.XXXXXX"
    ).strip()
    if not re.fullmatch(r"/data/local/tmp/bonsai-max\.[A-Za-z0-9]+", remote):
        raise RuntimeError(f"Unexpected temporary path: {remote!r}")
    records, hashes = [], {}
    variants = ["mojo", "max"] if args.compare_mojo else ["max"]
    try:
        assets = [
            model / n
            for n in ["weights.bin", "prompt.txt"]
            + ([] if args.generation_only else ["reference.f32"])
        ]
        manifest = json.loads((b / "android_artifacts.json").read_text())
        assets += [
            b / "graph_runner",
            b / "bonsai.mef",
            *[b / n for n in manifest["libraries"]],
        ]
        subprocess.run(
            [*adb, "push", *map(str, assets), remote + "/"],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        for path in assets:
            if path.parent != model:
                hashes["max/" + path.name] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
        if args.compare_mojo:
            plain = args.compare_mojo.resolve()
            libraries = json.loads(
                (plain / "android_artifacts.json").read_text()
            )["libraries"]
            command("shell", "mkdir", remote + "/mojo")
            assets = [plain / "runner", *[plain / n for n in libraries]]
            subprocess.run(
                [*adb, "push", *map(str, assets), remote + "/mojo/"],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            hashes["mojo/runner"] = hashlib.sha256(
                (plain / "runner").read_bytes()
            ).hexdigest()

        def sample(variant, warmup):
            reference = (
                "benchmark" if args.generation_only else remote
                + "/reference.f32"
            )
            env = [
                "env",
                *(
                    ["MODULAR_OPENCL_HOST_PROFILE=1"]
                    if args.host_profile
                    else []
                ),
                *(["MODULAR_OPENCL_PROFILE=1"] if args.profile else []),
            ]
            if variant == "max":
                env += [
                    f"LD_LIBRARY_PATH={remote}",
                    f"MODULAR_DRIVER_PLUGINS=CPU@{remote}/CPUPlugin.so;OpenCL@{remote}/OpenCLPlugin.so",
                    f"MODULAR_MOJO_MAX_COMPILERRT_PATH={remote}/libKGENCompilerRTShared.so",
                    f"MODULAR_MOJO_MAX_MGPRT_PATH={remote}/libMGPRT.so",
                ]
                if args.flush_max:
                    env.append("MODULAR_OPENCL_FLUSH_AFTER_ENQUEUE=1")
                if args.runtime_trace:
                    env.append(f"MODULAR_PROFILE_FILENAME={remote}/runtime")
                cmd = [
                    remote + "/graph_runner",
                    remote + "/bonsai.mef",
                    remote + "/weights.bin",
                    remote + "/prompt.txt",
                    reference,
                    str(meta["generation_tokens"]),
                    "24",
                    str(graph["capacity"]),
                ]
            else:
                folder = remote + "/mojo"
                env += [f"LD_LIBRARY_PATH={folder}"]
                cmd = [
                    folder + "/runner",
                    folder + "/OpenCLPlugin.so",
                    remote + "/weights.bin",
                    remote + "/prompt.txt",
                    reference,
                    str(meta["generation_tokens"]),
                ]
            output = command("shell", shlex.join(env + cmd))
            if variant == "max" and args.runtime_trace:
                args.runtime_trace.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(
                    [
                        *adb,
                        "pull",
                        remote + "/runtime.time-trace",
                        str(args.runtime_trace),
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                )
                subprocess.run(
                    [
                        *adb,
                        "pull",
                        remote + "/runtime.time-events.txt",
                        str(args.runtime_trace.with_suffix(".time-events.txt")),
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                )
            ids = [
                int(token)
                for token in re.findall(
                    r"^TOKEN \d+ (\d+)$", output, re.MULTILINE
                )
            ]
            if ids != meta["expected_ids"] or (
                not args.generation_only
                and "PASS: all generated tokens and logits match reference"
                not in output
            ):
                raise RuntimeError(
                    f"Unvalidated {variant} run:\n{output[-16000:]}"
                )
            timing = re.search(
                r"Decode: (\d+) forwards, ([\d.]+) s, ([\d.]+) tok/s", output
            )
            error = re.search(
                r"max_abs_error=([\deE.+-]+) relative_l2=([\deE.+-]+)", output
            )
            if timing is None:
                raise RuntimeError("Missing decode timing")
            log = args.output.with_name(
                args.output.stem + f"-{variant}-{len(records)}.log"
            )
            log.write_text(output)
            result = dict(
                variant=variant,
                warmup=warmup,
                forwards=int(timing[1]),
                seconds=float(timing[2]),
                tokens_per_second=float(timing[3]),
                log=str(log),
                max_abs_error=float(error[1]) if error else None,
                relative_l2=float(error[2]) if error else None,
            )
            records.append(result)
            print(json.dumps(result), flush=True)

        if not args.profile and not args.runtime_trace:
            for variant in variants:
                sample(variant, True)
        for repeat in range(
            1 if args.profile or args.runtime_trace else args.repeats
        ):
            for variant in variants if repeat % 2 == 0 else list(
                reversed(variants)
            ):
                sample(variant, False)
        medians = {
            v: statistics.median(
                r["tokens_per_second"]
                for r in records
                if r["variant"] == v and not r["warmup"]
            )
            for v in variants
        }
        report = dict(
            serial=args.serial,
            model=meta["model"],
            checkpoint_sha256=meta["checkpoint_sha256"],
            prompt_tokens=len(meta["prompt_ids"]),
            output_tokens=meta["generation_tokens"],
            validation_included=not args.generation_only,
            token_ids_validated=True,
            profiling=args.profile,
            flush_max=args.flush_max,
            runtime_trace=str(args.runtime_trace) if args.runtime_trace else None,
            runs=records,
            median_tokens_per_second=medians,
            artifact_sha256=hashes,
        )
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print("Medians:", medians)
    finally:
        command("shell", "rm", "-rf", remote)


if __name__ == "__main__":
    main()
