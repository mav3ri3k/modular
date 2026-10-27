
def direct_hadamard(
    x: Pointer[Float32, MutAnyOrigin], signs: Pointer[Float32, MutAnyOrigin],
    y: Pointer[Float32, MutAnyOrigin], q: Pointer[UInt32, MutAnyOrigin],
    width: Int32, inverse: Int32,
):
    var i = global_idx.x * 2
    if i < Int(width):
        var base = (i // 512) * 512
        var row0 = i % 512
        var row1 = row0 + 1
        var a: Float32 = 0
        var b: Float32 = 0
        for col in range(512):
            var value = x[unsafe_offset=base + col]
            if inverse == 0:
                value *= signs[unsafe_offset=base + col]
            var p0 = row0 & col
            var p1 = row1 & col
            p0 = p0 ^ (p0 >> 8)
            p0 = p0 ^ (p0 >> 4)
            p0 = p0 ^ (p0 >> 2)
            p0 = p0 ^ (p0 >> 1)
            p1 = p1 ^ (p1 >> 8)
            p1 = p1 ^ (p1 >> 4)
            p1 = p1 ^ (p1 >> 2)
            p1 = p1 ^ (p1 >> 1)
            a += value if (p0 & 1) == 0 else -value
            b += value if (p1 & 1) == 0 else -value
        a *= Float32(0.04419417382415922)
        b *= Float32(0.04419417382415922)
        if inverse != 0:
            a *= signs[unsafe_offset=i]
            b *= signs[unsafe_offset=i + 1]
        y[unsafe_offset=i] = a
        y[unsafe_offset=i + 1] = b
        if inverse == 0:
            q[unsafe_offset=i // 2] = bitcast[.uint32, 1](SIMD[.float32, 2](a, b).cast[.float16]())
