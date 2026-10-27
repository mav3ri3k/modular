#!/usr/bin/env python3
# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the Apache License v2.0 with LLVM Exceptions: https://llvm.org/LICENSE.txt
"""Run the Mojo model on an explicit USB device; tokenize/display on the host.

No CPU model is loaded in generation mode. --validate uses the exported test
prompt and reference logits instead. Temporary device files are cleaned up.
"""

import argparse
import json
import re
import shlex
import subprocess
import tempfile
from pathlib import Path

from transformers import AutoTokenizer


def run(*args) -> None:
    subprocess.run([str(a) for a in args], check=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--adb", default="adb")
    p.add_argument("--serial", required=True)
    p.add_argument("--prompt", default="What is the capital of France?")
    p.add_argument("--tokens", type=int, default=16)
    p.add_argument(
        "--profile",
        action="store_true",
        help="Collect OpenCL GPU timestamps (adds overhead)",
    )
    p.add_argument("--validate", action="store_true")
    args = p.parse_args()
    metadata = json.loads((args.model / "manifest.json").read_text())
    tokenizer_options = (
        {"fix_mistral_regex": True}
        if metadata["model"].startswith("prism-ml/")
        else {}
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, **tokenizer_options
    )
    count = metadata["generation_tokens"] if args.validate else args.tokens
    text = tokenizer.apply_chat_template(
        [dict(role="user", content=args.prompt)],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not args.validate and (
        count < 1 or len(ids) + count > metadata["capacity"]
    ):
        raise ValueError(
            "Prompt plus generation exceeds the exported context capacity"
        )
    adb = [args.adb, "-s", args.serial]
    remote = subprocess.check_output(
        [*adb, "shell", "mktemp", "-d", "/data/local/tmp/modular-llm.XXXXXX"],
        text=True,
    ).strip()
    if not re.fullmatch(r"/data/local/tmp/modular-llm\.[A-Za-z0-9]+", remote):
        raise RuntimeError(f"Unexpected remote directory {remote!r}")
    try:
        with tempfile.TemporaryDirectory() as temporary:
            prompt = Path(temporary) / "prompt.txt"
            if args.validate:
                prompt.write_bytes((args.model / "prompt.txt").read_bytes())
            else:
                prompt.write_text(" ".join(map(str, ids)) + "\n")
            run(
                *adb,
                "push",
                args.build / "runner",
                *[
                    args.build / name
                    for name in json.loads(
                        (args.build / "android_artifacts.json").read_text()
                    )["libraries"]
                ],
                args.model / "weights.bin",
                prompt,
                remote + "/",
            )
        reference = "-"
        if args.validate:
            run(*adb, "push", args.model / "reference.f32", remote + "/")
            reference = remote + "/reference.f32"
        command = [
            *adb,
            "shell",
            shlex.join(
                [
                    "env",
                    f"LD_LIBRARY_PATH={remote}",
                    *(["MODULAR_OPENCL_PROFILE=1"] if args.profile else []),
                    remote + "/runner",
                    remote + "/OpenCLPlugin.so",
                    remote + "/weights.bin",
                    remote + "/prompt.txt",
                    reference,
                    str(count),
                ]
            ),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        print(completed.stdout, end="")
        print(completed.stderr, end="")
        completed.check_returncode()
        generated = [
            int(m[1])
            for m in re.finditer(
                r"^TOKEN \d+ (\d+)$", completed.stdout, re.MULTILINE
            )
        ]
        print("Output:", tokenizer.decode(generated, skip_special_tokens=True))
    finally:
        run(*adb, "shell", "rm", "-rf", remote)


if __name__ == "__main__":
    main()
