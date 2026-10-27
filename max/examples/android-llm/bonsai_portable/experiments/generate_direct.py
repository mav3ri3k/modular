#!/usr/bin/env python3
"""Diagnostic: replace nine-pass Hadamard with one direct transform launch."""

import argparse
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    subprocess.run(
        [
            "python3",
            str(HERE.parent / "generate.py"),
            "--output",
            str(a.output),
        ],
        check=True,
    )
    path = a.output / "model.mojo"
    source = path.read_text()
    start = source.index("    var src = b[67]", source.index("def rotate("))
    end = source.index("    return 0", start) + len("    return 0")
    body = """    rt.enqueue_function[kernels.direct_hadamard](
        x.create_sub_buffer[.float32](0, len(x)//4),
        signs.create_sub_buffer[.float32](0, len(signs)//4),
        b[2].create_sub_buffer[.float32](0, len(b[2])//4),
        b[3].create_sub_buffer[.uint32](0, len(b[3])//4),
        width, inverse, grid_dim=Dim((Int(width)+127)//128,1,1), block_dim=64,
    )
    return 0"""
    path.write_text(source[:start] + body + source[end:])
    path = a.output / "kernels.mojo"
    path.write_text(
        path.read_text() + (HERE / "direct_hadamard.mojo.inc").read_text()
    )


if __name__ == "__main__":
    main()
