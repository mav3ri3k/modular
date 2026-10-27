#!/usr/bin/env python3
# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the Apache License v2.0 with LLVM Exceptions: https://llvm.org/LICENSE.txt
"""Generate, compile and profile individually selected projection experiments."""

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

from generate import generate

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
MOJO = Path(
    os.environ.get(
        "PROJECTION_MOJO", str(ROOT / "bazel-bin/Mojo/tools/mojo/mojo")
    )
)
PACKAGES = Path(
    os.environ.get("PROJECTION_PACKAGES", "/tmp/bonsai-candidate-fixture")
)
NDK = Path(
    os.environ.get(
        "PROJECTION_NDK",
        str(Path.home() / "Library/Android/sdk/ndk/29.0.14206865"),
    )
)
ADB = Path(
    os.environ.get(
        "PROJECTION_ADB",
        str(Path.home() / "Library/Android/sdk/platform-tools/adb"),
    )
)
SERIAL = os.environ.get("ANDROID_SERIAL") or "DEVICE_SERIAL"
OPENCL_HEADERS = Path(
    os.environ.get("PROJECTION_OPENCL_HEADERS", "/tmp/OpenCL-Headers")
)
OPENCL_LIBRARY = Path(
    os.environ.get("PROJECTION_OPENCL_LIBRARY", "/tmp/libOpenCL-android.so")
)
REMOTE = "/data/local/tmp/modular-projection-iteration"


def variants(base: str) -> dict[str, str]:
    """Derive arithmetic/scheduling experiments without changing packing."""
    u32 = base.replace("UInt64(", "UInt32(")
    tiled = u32.replace(
        "var base = group * Int(width) + i",
        "var base = Int(((UInt32(group) // 8 * (UInt32(width) // 64) + UInt32(i) // 64) * 8 + UInt32(group) % 8) * 64 + UInt32(i) % 64)",
    )
    split = u32.index("@always_inline\ndef _group_dot8(")
    prefix = u32[:split]
    compact = u32[split:].replace(
        "        comptime for pair in range(8):",
        "        for pair_base in range(2):\n            comptime for chain in range(4):\n                var pair = pair_base * 4 + chain",
    )
    # Indent the pair body and preserve the compile-time accumulator choice.
    start = compact.index("            var q =", compact.index("for pair_base"))
    end = compact.index("    var result =", start)
    compact = (
        compact[:start]
        + "".join(
            "    " + line if line.strip() else line
            for line in compact[start:end].splitlines(True)
        )
        + compact[end:]
    )
    compact = prefix + compact.replace(
        "comptime if pair % 4 ==", "comptime if chain =="
    ).replace("elif pair % 4 ==", "elif chain ==")
    narrow = u32.replace(
        "(c & 3).cast[.float16]()", "(c & 3).cast[.uint16]().cast[.float16]()"
    ).replace(
        "((c >> 2) & 3).cast[.float16]()",
        "((c >> 2) & 3).cast[.uint16]().cast[.float16]()",
    )
    bits = u32.replace(
        "            var lo_weight = (c & 3).cast[.float16]() - SIMD[.float16, 8](1)\n            var hi_weight = (\n                ((c >> 2) & 3).cast[.float16]() - SIMD[.float16, 8](1)\n            )",
        """            var lo_code = c & 3
            var hi_code = (c >> 2) & 3
            var lo_bits = (((~lo_code) & 1) * UInt32(0x3c00)) | ((lo_code & (lo_code >> 1)) << 14) | (((lo_code - 1) >> 16) & 0x8000)
            var hi_bits = (((~hi_code) & 1) * UInt32(0x3c00)) | ((hi_code & (hi_code >> 1)) << 14) | (((hi_code - 1) >> 16) & 0x8000)
            var lo_weight = bitcast[.float16, 8](lo_bits.cast[.uint16]())
            var hi_weight = bitcast[.float16, 8](hi_bits.cast[.uint16]())""",
    )
    if bits == u32:
        raise ValueError("Half-bit source replacement failed")
    dot_start = u32.index("@always_inline\ndef _group_dot8(")
    pre = u32[:dot_start]
    body = u32[dot_start:]
    load = next(
        line
        for line in body.splitlines()
        if "var codes = w.unsafe_load" in line
    )
    init = (
        load.replace("var codes", "var next_codes")
        .replace("UInt32(word)", "UInt32(0)")
        .strip()
    )
    future = load.replace("var codes =", "next_codes =").replace(
        "UInt32(word)", "UInt32(word + 1)"
    )
    prefetched = body.replace(
        "    for word in range(8):",
        "    " + init + "\n    for word in range(8):",
        1,
    )
    prefetched = prefetched.replace(
        load,
        "        var codes = next_codes\n        if word < 7:\n    " + future,
    )
    words = body.replace(
        "    for word in range(8):",
        "    for word_base in range(4):\n        comptime for word_inner in range(2):\n            var word = word_base * 2 + word_inner",
        1,
    )
    start = words.index(load)
    end = words.index("    var result =", start)
    words = (
        words[:start]
        + "".join(
            "    " + line if line.strip() else line
            for line in words[start:end].splitlines(True)
        )
        + words[end:]
    )
    shared = body.replace(
        """            var q = bitcast[.float16, 2](
                x[unsafe_offset=group * 64 + word * 8 + pair]
            )""",
        """            var packed_q = UInt32(0)
            if global_idx.x % 8 == 0:
                packed_q = x[unsafe_offset=group * 64 + word * 8 + pair]
            packed_q = shuffle_idx(packed_q, UInt32((global_idx.x % 64) // 8 * 8))
            var q = bitcast[.float16, 2](packed_q)""",
    )
    if shared == body:
        raise ValueError("Activation replacement failed")
    broadcast = (pre + shared).replace(
        "import shuffle_xor", "import shuffle_xor, shuffle_idx"
    )
    words4 = body.replace(
        "    for word in range(8):",
        "    for word_base in range(2):\n        comptime for word_inner in range(4):\n            var word = word_base * 4 + word_inner",
        1,
    )
    start = words4.index(load)
    end = words4.index("    var result =", start)
    words4 = (
        words4[:start]
        + "".join(
            "    " + line if line.strip() else line
            for line in words4[start:end].splitlines(True)
        )
        + words4[end:]
    )

    def vector_inputs(text: str) -> str:
        anchor = next(
            line
            for line in text.splitlines()
            if "var codes = w.unsafe_load" in line
        )
        indent = anchor[: len(anchor) - len(anchor.lstrip())]
        text = text.replace(
            anchor,
            anchor
            + "\n"
            + indent
            + "var input_pairs = x.unsafe_load[width=8](group * 64 + word * 8)",
        )
        old = """var q = bitcast[.float16, 2](
                x[unsafe_offset=group * 64 + word * 8 + pair]
            )"""
        # The word-unrolled body has four extra spaces of nesting.
        if old not in text:
            old = old.replace("\n", "\n    ")
        if old not in text:
            raise ValueError("Vector input replacement failed")
        return text.replace(
            old, "var q = bitcast[.float16, 2](input_pairs[pair])"
        )

    inputs = pre + vector_inputs(body)
    word2_inputs = pre + vector_inputs(words)
    input2 = body.replace(
        "        comptime for pair in range(8):",
        "        comptime for pair_base in range(4):\n            var input_pairs = x.unsafe_load[width=2](group * 64 + word * 8 + pair_base * 2)\n            comptime for pair_inner in range(2):\n                comptime pair = pair_base * 2 + pair_inner",
        1,
    )
    start = input2.index(
        "            var q =", input2.index("comptime for pair_base")
    )
    end = input2.index("    var result =", start)
    input2 = (
        input2[:start]
        + "".join(
            "    " + line if line.strip() else line
            for line in input2[start:end].splitlines(True)
        )
        + input2[end:]
    )
    old = """var q = bitcast[.float16, 2](
                    x[unsafe_offset=group * 64 + word * 8 + pair]
                )"""
    if old not in input2:
        raise ValueError("Input2 replacement failed")
    input2 = input2.replace(
        old, "var q = bitcast[.float16, 2](input_pairs[pair_inner])"
    )
    return {
        "control": base,
        "u32": u32,
        "scales": tiled,
        "compact4": compact,
        "narrow16": narrow,
        "halfbits": bits,
        "prefetch": pre + prefetched,
        "word2": pre + words,
        "broadcast": broadcast,
        "word4": pre + words4,
        "inputs8": inputs,
        "word2_inputs": word2_inputs,
        "inputs2": pre + input2,
    }


def tuning_config(name: str) -> dict[str, int]:
    """Decode the explicit categorical configuration in a tuning name."""
    if not name.startswith("t"):
        return {}
    return {part[0]: int(part[1:]) for part in name.split("_")}


def tuned(
    source: str,
    tile: int,
    joined: int,
    packed_output: int = 64,
    narrow: bool = False,
) -> str:
    """Change lane geometry while keeping packing and summation explicit."""
    start = source.index("@always_inline\ndef _group_dot8(")
    prefix = source[:start]
    body = source[start:]
    for dtype in ("float16", "float32"):
        body = body.replace(f"SIMD[.{dtype}, 8]", f"SIMD[.{dtype}, {tile}]")
    body = body.replace(
        "unsafe_load[width=8]", f"unsafe_load[width={tile}]"
    ).replace("unsafe_store[width=8]", f"unsafe_store[width={tile}]")
    body = body.replace(
        "comptime for output in range(8)",
        f"comptime for output in range({tile})",
    )
    body = body.replace(
        "comptime for o in range(8)", f"comptime for o in range({tile})"
    ).replace("for o in range(8)", f"for o in range({tile})")
    body = body.replace("i + 8 <= Int(width)", f"i + {tile} <= Int(width)")
    lanes = 64 // joined
    body = body.replace(
        "var group = block_idx.y * 8 + lane // 8",
        f"var group = Int(UInt32(block_idx.y) * {joined} + UInt32(lane) // {lanes})",
    )
    body = body.replace(
        "var i = (block_idx.x * 8 + lane % 8) * 8",
        f"var i = Int((UInt32(global_idx.x) // 64 * {lanes} + UInt32(lane) % {lanes}) * {tile})",
    )
    body = body.replace(
        "comptime for stage in range(3)",
        f"comptime for stage in range({joined.bit_length() - 1})",
    )
    body = body.replace(
        "UInt32(8 << stage)", f"UInt32({lanes} << stage)"
    ).replace("if lane < 8:", f"if lane < {lanes}:")
    body = body.replace(
        "Join eight quantization groups inside each 64-lane subgroup.",
        f"Join {joined} quantization groups inside each 64-lane subgroup.",
    )
    body = body.replace(
        "Eight lanes cover adjacent eight-output tiles",
        f"{lanes} lanes cover adjacent {tile}-output tiles",
    )
    body = body.replace("SIMD8 streams", f"SIMD{tile} streams").replace(
        "eight-output tile", f"{tile}-output tile"
    )
    if packed_output != 64:
        load = next(
            line
            for line in body.splitlines()
            if "var codes = w.unsafe_load" in line
        )
        indent = load[: len(load) - len(load.lstrip())]
        address = f"Int(((UInt32(group) // 8 * (UInt32(width) // {packed_output}) + UInt32(i) // {packed_output}) * 64 + UInt32(word) * 8 + UInt32(group) % 8) * {packed_output} + UInt32(i) % {packed_output})"
        body = body.replace(
            load, indent + f"var codes = w.unsafe_load[width={tile}]({address})"
        )
    if narrow:
        body = body.replace(
            "(c & 3).cast[.float16]()",
            "(c & 3).cast[.uint16]().cast[.float16]()",
        ).replace(
            "((c >> 2) & 3).cast[.float16]()",
            "((c >> 2) & 3).cast[.uint16]().cast[.float16]()",
        )
    return prefix + body


def build(out: Path, names: list[str]) -> None:
    """Compile selected Mojo probes and a matching native validation host."""
    generated = out / "generated"
    generate(generated)
    sources = variants((generated / "group_major_full.mojo").read_text())
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, MODULAR_MOJO_MAX_IMPORT_PATH=str(PACKAGES))
    hashes = {}
    for name in dict.fromkeys(["original", *names]):
        source = out / f"{name}.mojo"
        cfg = tuning_config(name)
        if cfg:
            text = tuned(
                sources["u32" if cfg["u"] == 1 else "word2"],
                cfg["t"],
                cfg["g"],
                cfg.get("p", 64),
                "n" in cfg,
            )
        elif name == "original":
            text = (generated / "baseline_full.mojo").read_text()
        elif name == "reduce":
            production = (HERE.parent / "kernels.mojo").read_text()
            text = (
                "from std.memory import Pointer\nfrom max.gpu import global_idx\n\n"
                + production[
                    production.index("def packed_reduce(") : production.index(
                        "\n\ndef packed_head("
                    )
                ].replace("def packed_reduce(", "def projection(")
            )
        else:
            text = sources[name]
        source.write_text(text)
        host = out / "host.mojo"
        host.write_text(f"""import {name} as probe
from max.gpu.host import DeviceContext
from std.memory import Pointer
@export
def launch(ctx: Pointer[DeviceContext, MutAnyOrigin]) abi("C"):
    try:
        var x = ctx[].enqueue_create_buffer[.uint32](1024)
        var w = ctx[].enqueue_create_buffer[.uint32](262144)
        var p = ctx[].enqueue_create_buffer[.float16](32768)
        var y = ctx[].enqueue_create_buffer[.float32](4096)
        ctx[].enqueue_function[probe.projection](x, w, p, y, Int32(2048), Int32(2048), grid_dim=(32, 2, 1), block_dim=64)
    except:
        pass
""")
        if name == "reduce":
            host.write_text(
                host.read_text().replace(
                    "ctx[].enqueue_function[probe.projection](x, w, p, y, Int32(2048), Int32(2048), grid_dim=(32, 2, 1), block_dim=64)",
                    "ctx[].enqueue_function[probe.projection](y, y, Int32(2048), Int32(2), grid_dim=16, block_dim=128)",
                )
            )
        for old in out.glob("host_*.cl"):
            old.unlink()
        cmd = [
            str(MOJO),
            "build",
            str(host),
            "--target-triple",
            "aarch64-linux-android33",
            "--target-cpu",
            "generic",
            "--target-accelerator",
            "adreno-840",
            "-I",
            str(PACKAGES),
            "--emit",
            "asm",
            "-O3",
            "-o",
            str(out / "host.s"),
        ]
        begin = time.time()
        result = subprocess.run(
            cmd, env=env, text=True, capture_output=True, timeout=90
        )
        if result.returncode:
            raise RuntimeError(result.stderr)
        emitted = list(out.glob("host_*.cl"))
        assert len(emitted) == 1
        shutil.copyfile(emitted[0], out / f"{name}.cl")
        hashes[name] = {
            "sha256": hashlib.sha256(
                (out / f"{name}.cl").read_bytes()
            ).hexdigest(),
            "bytes": (out / f"{name}.cl").stat().st_size,
            "mojo_source_sha256": hashlib.sha256(
                source.read_bytes()
            ).hexdigest(),
            "mojo_build_seconds": time.time() - begin,
        }
        print(name, hashes[name], flush=True)
    toolchain = next(
        (NDK / "toolchains/llvm/prebuilt").glob(
            "*/bin/aarch64-linux-android33-clang++"
        )
    )
    subprocess.run(
        [
            str(toolchain),
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(OPENCL_HEADERS),
            str(HERE / "bench.cpp"),
            str(OPENCL_LIBRARY),
            "-o",
            str(out / "bench"),
        ],
        check=True,
    )
    shutil.copyfile(OPENCL_LIBRARY, out / "libOpenCL.so")
    cxx = list(
        (NDK / "toolchains/llvm/prebuilt").glob(
            "*/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"
        )
    )
    if len(cxx) != 1:
        raise RuntimeError("Expected one NDK libc++_shared.so")
    shutil.copyfile(cxx[0], out / "libc++_shared.so")
    shutil.copyfile(out / "original.cl", out / "reference.cl")
    (out / "build.json").write_text(
        json.dumps(
            {
                "kernels": hashes,
                "host_source_sha256": hashlib.sha256(
                    (HERE / "bench.cpp").read_bytes()
                ).hexdigest(),
                "generator_source_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "compiler_sha256": hashlib.sha256(
                    MOJO.resolve().read_bytes()
                ).hexdigest(),
                "package_sha256": {
                    n: hashlib.sha256((PACKAGES / n).read_bytes()).hexdigest()
                    for n in ("std.mojoc", "max.mojoc")
                },
                "host_sha256": hashlib.sha256(
                    (out / "bench").read_bytes()
                ).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )


def run(
    out: Path,
    names: list[str],
    output: Path,
    repeats: int,
    shapes: list[str],
    ring: int,
    iterations: int,
    random_inputs: bool,
    pipeline: bool,
    seed: int = 123456789,
) -> None:
    """Validate every bank and retain all timed processes and failure output."""
    adb = [str(ADB), "-s", SERIAL]

    def call(*cmd: str, **kw) -> str:
        return subprocess.run(
            [*adb, *cmd], text=True, capture_output=True, check=True, **kw
        ).stdout

    call("shell", f"mkdir -p {REMOTE}")
    for p in [
        out / "bench",
        *out.glob("*.so"),
        out / "reference.cl",
        *([out / "reduce.cl"] if pipeline else []),
        *[out / f"{n}.cl" for n in names],
    ]:
        call("push", str(p), REMOTE + "/" + p.name)
    call("shell", f"chmod 755 {REMOTE}/bench; input keyevent KEYCODE_WAKEUP")

    def state() -> str:
        raw = call("shell", "dumpsys thermalservice; dumpsys power")
        return "\n".join(
            line
            for line in raw.splitlines()
            if any(
                key in line
                for key in [
                    "Thermal Status:",
                    "Temperature{",
                    "mWakefulness=",
                    "mIsPowered=",
                ]
            )
        )

    records = []
    output.parent.mkdir(parents=True, exist_ok=True)
    for repeat in range(repeats):
        order = names if repeat % 2 == 0 else list(reversed(names))
        for shape in shapes:
            w, k = map(int, shape.split("x"))
            banks = math.ceil(ring * 1024**2 / (w * k // 4)) if ring else 1
            for name in order:
                before = state()
                extra = "PROJECTION_TILED_SCALES=1 " if name == "scales" else ""
                cfg = tuning_config(name)
                if cfg:
                    extra += f"PROJECTION_TILE={cfg['t']} PROJECTION_GROUPS={cfg['g']} PROJECTION_WORKGROUP={cfg['w']} "
                if cfg.get("p"):
                    extra += f"PROJECTION_PACK_OUTPUT={cfg['p']} "
                if pipeline:
                    extra += "PROJECTION_REDUCE_SOURCE=reduce.cl "
                if random_inputs:
                    extra += "PROJECTION_RANDOM_INPUTS=1 "
                layout = "baseline" if name == "original" else "group_major"
                cmd = f"cd {REMOTE} && timeout -s KILL 45 env LD_LIBRARY_PATH={REMOTE} {extra}./bench reference.cl {name}.cl {layout} full {w} {k} {banks} {max(iterations, banks)} {seed}"
                begin = time.time()
                try:
                    raw = call("shell", cmd, timeout=55)
                    stages = [
                        json.loads(line)
                        for line in raw.splitlines()
                        if line.startswith("{")
                    ]
                    result = next(s for s in stages if s["stage"] == "result")
                    record = {
                        "repeat": repeat,
                        "shape": shape,
                        "variant": name,
                        "stages": stages,
                        "result": result,
                        "before": before,
                        "after": state(),
                        "seconds": time.time() - begin,
                    }
                    print(
                        repeat,
                        shape,
                        name,
                        result["device_mean_ms"],
                        flush=True,
                    )
                except (
                    subprocess.SubprocessError,
                    StopIteration,
                    ValueError,
                ) as e:
                    record = {
                        "repeat": repeat,
                        "shape": shape,
                        "variant": name,
                        "error": str(e),
                        "stdout": getattr(e, "stdout", ""),
                        "stderr": getattr(e, "stderr", ""),
                        "before": before,
                        "after": state(),
                    }
                    print("FAILED", name, record["stderr"], flush=True)
                records.append(record)
                output.write_text(
                    json.dumps(
                        {
                            "build": json.loads(
                                (out / "build.json").read_text()
                            ),
                            "run_artifacts": {
                                n: hashlib.sha256(
                                    (out / f"{n}.cl").read_bytes()
                                ).hexdigest()
                                for n in names
                            },
                            "config": {
                                "names": names,
                                "repeats": repeats,
                                "shapes": shapes,
                                "ring_mib": ring,
                                "random_inputs": random_inputs,
                                "pipeline": pipeline,
                                "seed": seed,
                            },
                            "records": records,
                        },
                        indent=2,
                    )
                    + "\n"
                )
                if "error" in record:
                    raise RuntimeError("Probe failed; preserved in output")
                time.sleep(1 if ring else 0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build", action="store_true")
    p.add_argument("--run", action="store_true")
    p.add_argument("--out", type=Path, default=Path("/tmp/projection-iterate"))
    p.add_argument("--names", nargs="+", required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument(
        "--shapes", nargs="+", default=["6144x2048", "2048x2048", "2048x6144"]
    )
    p.add_argument("--ring", type=int, default=128)
    p.add_argument("--iterations", type=int, default=512)
    p.add_argument("--random-inputs", action="store_true")
    p.add_argument("--pipeline", action="store_true")
    p.add_argument("--seed", type=int, default=123456789)
    a = p.parse_args()
    out = a.out.resolve()
    if a.build:
        build(out, a.names)
    if a.run:
        run(
            out,
            a.names,
            a.output or out / "results.json",
            a.repeats,
            a.shapes,
            a.ring,
            a.iterations,
            a.random_inputs,
            a.pipeline,
            a.seed,
        )


if __name__ == "__main__":
    main()
