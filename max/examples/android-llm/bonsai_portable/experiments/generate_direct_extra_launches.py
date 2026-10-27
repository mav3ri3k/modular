#!/usr/bin/env python3
"""Diagnostic: add tiny launches after each direct Hadamard transform."""

import argparse
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--extra-count", type=int, default=10)
    a = p.parse_args()
    if a.extra_count < 0:
        p.error("--extra-count must be nonnegative")
    subprocess.run(
        [
            "python3",
            str(HERE / "generate_direct.py"),
            "--output",
            str(a.output),
        ],
        check=True,
    )
    path = a.output / "model.mojo"
    text = path.read_text()
    start = text.index("def rotate(")
    end = text.index("    return 0", start)
    text = (
        text[:end]
        + f"""    for _ in range({a.extra_count}):
        rt.enqueue_function[kernels.diagnostic_touch](
            b[67].create_sub_buffer[.float32](0, len(b[67])//4),
            grid_dim=Dim(1,1,1), block_dim=1,
        )
"""
        + text[end:]
    )
    path.write_text(text)
    path = a.output / "kernels.mojo"
    path.write_text(
        path.read_text()
        + """\n\ndef diagnostic_touch(x: Pointer[Float32, MutAnyOrigin]):
    x[unsafe_offset=0] = Float32(0)
"""
    )


if __name__ == "__main__":
    main()
