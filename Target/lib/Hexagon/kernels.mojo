# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the Apache License v2.0 with LLVM Exceptions: https://llvm.org/LICENSE.txt
"""Freestanding Mojo DSP ABI and HMX matrix smoke tests (Hexagon v81)."""
from std.memory import Pointer
from std.sys import inlined_assembly


@export("modular_dsp_affine")
def modular_dsp_affine(
    x: Pointer[Float32, MutAnyOrigin],
    y: Pointer[Float32, MutAnyOrigin],
    n: Int32,
) abi("C"):
    for i in range(Int(n)):
        y[unsafe_offset=i] = x[unsafe_offset=i] * 2 + 1


@export("modular_dsp_hmx_clear")
def modular_dsp_hmx_clear() abi("C"):
    # mxclracc.hf, encoded by Qualcomm llvm-mc v21 for Hexagon v81.
    inlined_assembly[".word 0xa6e0c013", NoneType, constraints="~{memory}"]()


@export("modular_dsp_hmx_tile")
def modular_dsp_hmx_tile(
    a: Pointer[UInt16, MutAnyOrigin],
    w: Pointer[UInt16, MutAnyOrigin],
    scale: Pointer[UInt32, MutAnyOrigin],
    y: Pointer[UInt16, MutAnyOrigin],
) abi("C") -> Int32:
    for i in range(1024):
        var row = (i // 64) * 2 + i % 2
        var channel = (i % 64) // 2
        a[unsafe_offset=i] = 0x3C00 if row % 2 == 0 else 0x4000
        w[unsafe_offset=i] = (
            UInt16(0x3C00) if channel % 2 == 0 else UInt16(0x4000)
        ) if row == channel else UInt16(0)
        y[unsafe_offset=i] = 0
    for i in range(64):
        scale[unsafe_offset=i] = 0x3C00 if i < 32 else 0
    # Fixed register operands keep the encoded HMX instructions independent of allocation.
    # Clear; activation+weight load/multiply; bias load; convert/store FP16.
    inlined_assembly[
        (
            ".word 0xa6e0c013\n.word 0x920041e4\n.word 0x9202e3ef\n.word"
            " 0x9204c3fe\n.word 0xa6e5e604"
        ),
        NoneType,
        constraints="{r0},{r1},{r2},{r3},{r4},{r5},{r6},~{memory}",
    ](a, UInt32(2047), w, UInt32(2047), scale, y, UInt32(0))
    var failed: Int32 = 0
    for i in range(1024):
        var row = (i // 64) * 2 + i % 2
        var col = (i % 64) // 2
        var expected = UInt16(0x3C00) if (row % 2 + col % 2) == 0 else (
            UInt16(0x4000) if (row % 2 + col % 2) == 1 else UInt16(0x4400)
        )
        if y[unsafe_offset=i] != expected:
            failed += 1
    return failed
