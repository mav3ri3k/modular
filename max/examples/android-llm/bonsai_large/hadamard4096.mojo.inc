

def hadamard_join_4096(
    x: Pointer[Float32, MutAnyOrigin],
    signs: Pointer[Float32, MutAnyOrigin],
    y: Pointer[Float32, MutAnyOrigin],
    q: Pointer[UInt32, MutAnyOrigin],
    width: Int32,
    inverse: Int32,
):
    var i = global_idx.x * 2
    if i < Int(width):
        var base = (i // 4096) * 4096
        var offset = i % 512
        var row = (i % 4096) // 512
        var v0: Float32 = 0
        var v1: Float32 = 0
        for col in range(8):
            var bits = row & col
            var parity = (bits ^ (bits >> 1) ^ (bits >> 2)) & 1
            var s = Float32(1) if parity == 0 else Float32(-1)
            v0 += s * x[unsafe_offset=base + col * 512 + offset]
            v1 += s * x[unsafe_offset=base + col * 512 + offset + 1]
        v0 *= Float32(0.3535533905932738)
        v1 *= Float32(0.3535533905932738)
        if inverse != 0:
            v0 *= signs[unsafe_offset=i]
            v1 *= signs[unsafe_offset=i + 1]
        y[unsafe_offset=i] = v0
        y[unsafe_offset=i + 1] = v1
        if inverse == 0:
            q[unsafe_offset=i // 2] = bitcast[.uint32, 1](
                SIMD[.float32, 2](v0, v1).cast[.float16]()
            )
