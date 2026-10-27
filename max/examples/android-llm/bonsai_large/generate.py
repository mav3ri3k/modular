#!/usr/bin/env python3
# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the Apache License v2.0 with LLVM Exceptions: https://llvm.org/LICENSE.txt
"""Generate fixed-dimension 4B/9B ports from the validated 2B implementation."""

import argparse
from pathlib import Path

SOURCE = Path(__file__).resolve().parent.parent / "bonsai"


def change_function(
    source: str, name: str, changes: list[tuple[str, str]]
) -> str:
    start = source.index("def " + name + "(")
    end = source.find("\n\ndef ", start + 1)
    if end < 0:
        end = len(source)
    part = source[start:end]
    for old, new in changes:
        if old not in part:
            raise ValueError(f"{name}: missing {old!r}")
        part = part.replace(old, new)
    return source[:start] + part + source[end:]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--size", choices=("4b", "9b"), required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    h, ff = (2560, 9216) if a.size == "4b" else (4096, 12288)
    k = (SOURCE / "kernels.mojo").read_text()
    rules = {
        "embedding": [("i < 2048", f"i < {h}")],
        "rms_partials": [("range(64)", f"range({h // 32})")],
        "rms_norm": [("2048", str(h))],
        "gate_partials": [
            ("i < 1024", "i < 2048"),
            ("i % 32", "i % 64"),
            ("i // 32", "i // 64"),
            ("col >= 16", "col >= 32"),
            ("range(64)", f"range({h // 32})"),
            ("chunk * 64 + r", f"chunk * {h // 32} + r"),
            ("j * 16 + col % 16", "j * 32 + col % 32"),
        ],
        "convolve": [("12288", "16384"), ("6144", "8192")],
        "normalize_qk": [("i < 6144", "i < 8192")],
        "delta_gates": [
            ("i < 16", "i < 32"),
            ("chunk * 32", "chunk * 64"),
            ("+ i + 16", "+ i + 32"),
            ("16 + i", "32 + i"),
        ],
        "delta_update": [
            ("i < 2048", "i < 4096"),
            ("2048 + h * 128", "2048 + (h // 2) * 128"),
            ("qkv[unsafe_offset=h * 128", "qkv[unsafe_offset=(h // 2) * 128"),
        ],
        "gated_norm": [("i < 2048", "i < 4096")],
        "head_norm": [
            ("2048 if mode == 0 else 512", "4096 if mode == 0 else 1024")
        ],
        "cache_kv": [("512", "1024")],
        "attention_scores": [
            ("8 * length", "16 * length"),
            ("t * 512", "t * 1024"),
        ],
        "attention_values": [("i < 2048", "i < 4096"), ("t * 512", "t * 1024")],
        "residual": [("2048", str(h))],
        "swiglu": [("6144", str(ff))],
        "copy": [("2048", str(h))],
    }
    for name, changes in rules.items():
        k = change_function(k, name, changes)
    m = (
        (SOURCE / "model.mojo")
        .read_text()
        .replace("b[67]", "b[83]")
        .replace("b[68]", "b[84]")
    )
    if a.size == "9b":
        k = change_function(
            k, "hadamard", [("if inverse != 0:", "if inverse == 2:")]
        )
        k += (Path(__file__).parent / "hadamard4096.mojo.inc").read_text()
        old = """    return launch_hadamard(
        rt, width // 8, x, signs, b[2], b[3], width, 0, inverse
    )"""
        new = """    var status = launch_hadamard(rt, width // 8, x, signs, b[83], b[3], width, 0, inverse)
    if status != 0:
        return status
    rt.enqueue_function[kernels.hadamard_join_4096](
        b[83].create_sub_buffer[.float32](0, len(b[83]) // 4),
        signs.create_sub_buffer[.float32](0, len(signs) // 4),
        b[2].create_sub_buffer[.float32](0, len(b[2]) // 4),
        b[3].create_sub_buffer[.uint32](0, len(b[3]) // 4),
        width, inverse, grid_dim=Dim((Int(width) + 127) // 128, 1, 1), block_dim=64,
    )
    return 0"""
        if old not in m:
            raise ValueError("2B rotate implementation changed")
        m = m.replace(old, new)
    m = change_function(m, "norm", [("rt, 2048, x", f"rt, {h}, x")])
    # Replace the forward body explicitly by region so dimensions with different
    # meanings (hidden, attention, recurrent and MLP) are not conflated.
    start = m.index("    status = launch_embedding(", m.index("def forward("))
    end = m.index("    return 0", start)
    # The last return is the end of forward, not the first status check.
    end = m.rindex("    return 0") + len("    return 0")
    body = (Path(__file__).parent / "forward.mojo.inc").read_text()
    body = (
        body.replace("@H@", str(h))
        .replace("@FF@", str(ff))
        .replace("@HEAD@", "0" if a.size == "4b" else "804")
    )
    m = m[:start] + body.rstrip() + m[end:]
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "model.mojo").write_text(m)
    (a.output / "kernels.mojo").write_text(k)
    head = 804 if a.size == "9b" else -1
    count = 808 if head >= 0 else 805
    rope = count - 1
    scratch = max(
        ff * ((h + 1023) // 1024),
        h * ((ff + 1023) // 1024),
        8192 * ((h + 1023) // 1024),
    )
    spec = r"""// Generated fixed model geometry.
#pragma once
#include <cstdint>
static constexpr char modelMagic[] = "BONSAI01";
static constexpr int vocabulary=248320, weightCount=@COUNT@, bufferCount=85;
static constexpr int logitsBuffer=17, ropeBuffer=18, greedyBuffer=84, eosToken=248046, alternateEosToken=248044;
static constexpr int H=@H@, FF=@FF@, HEAD=@HEAD@, ROPE=@ROPE@;
static inline int sharedSignGroup(int i) {
  if (i < 4 || i >= 804) return -1;
  int layer=(i-4)/25, slot=(i-4)%25;
  if (slot==4 || slot==7) return layer*2;
  if (slot==13 || slot==16 || (slot==19 && (layer+1)%4==0)) return layer*2+1;
  return -1;
}
static inline bool centeredParams(int i) {
  if (i==1 || (HEAD >= 0 && i==HEAD+1)) return true;
  if (i < 4 || i >= 804) return false;
  int layer=(i-4)/25, slot=(i-4)%25;
  return slot==3 || slot==6 || slot==9 || slot==12 || slot==15 || slot==18 || ((layer+1)%4==0 && slot==21);
}
static inline uint64_t packedBytes(int part,uint64_t out,uint64_t in) {
  return part==0 ? out*in/16*4 : part==1 ? out*in/128*2*4 : in*4;
}
static inline uint64_t weightBytes(int i,uint32_t capacity) {
  if (i<3) return packedBytes(i, vocabulary,H);
  if (i==3) return H*4;
  if (i==ROPE) return capacity*64ull*4;
  if (HEAD>=0 && i>=HEAD) return packedBytes(i-HEAD,vocabulary,H);
  int layer=(i-4)/25,slot=(i-4)%25;
  if (slot<2) return H*4;
  if (slot<11) { int group=(slot-2)/3,part=(slot-2)%3; return packedBytes(part,group==2?H:FF,group==2?FF:H); }
  if ((layer+1)%4==0) {
    if (slot>=23) return 256*4;
    constexpr int width[]={8192,1024,1024,H};
    return packedBytes((slot-11)%3,width[(slot-11)/3],slot<20?H:4096);
  }
  if (slot<20) return packedBytes((slot-11)%3,slot<14?8192:slot<17?4096:H,slot<17?H:4096);
  if (slot<22) return H*32*4;
  if (slot==22) return 64*4;
  if (slot==23) return 8192*4*4;
  return 128*4;
}
static inline uint64_t bufferElements(int i,uint32_t capacity) {
  if (i==84) return 1455;
  if (i==83) return @SCRATCH@;
  const uint64_t sizes[]={H,H,FF,FF,8192,8192,32,32,64,8192,4096,FF,FF,0,4096,1024,1024,248320};
  if (i<18) return i==13 ? 16ull*capacity : sizes[i];
  int layer=(i-19)/2,part=(i-19)%2;
  return (layer+1)%4==0 ? 1024ull*capacity : part==0 ? 3ull*8192 : 32ull*128*128;
}
static inline bool zeroBuffer(int i) { return i>=19 && i<83 && (((i-19)/2+1)%4!=0); }
"""
    for old, new in {
        "@COUNT@": count,
        "@H@": h,
        "@FF@": ff,
        "@HEAD@": head,
        "@ROPE@": rope,
        "@SCRATCH@": scratch,
    }.items():
        spec = spec.replace(old, str(new))
    (a.output / "qwen_spec.h").write_text(spec)


if __name__ == "__main__":
    main()
