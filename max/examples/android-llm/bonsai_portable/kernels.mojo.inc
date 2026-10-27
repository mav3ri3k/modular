
def hadamard_init(x: Pointer[Float32, MutAnyOrigin], signs: Pointer[Float32, MutAnyOrigin], y: Pointer[Float32, MutAnyOrigin], width: Int32, inverse: Int32):
    var i = global_idx.x
    if i < Int(width):
        y[unsafe_offset=i] = x[unsafe_offset=i] * (signs[unsafe_offset=i] if inverse == 0 else Float32(1))


def hadamard_stage(x: Pointer[Float32, MutAnyOrigin], y: Pointer[Float32, MutAnyOrigin], width: Int32, mask: Int32):
    var i = global_idx.x
    if i < Int(width):
        var other = x[unsafe_offset=i ^ Int(mask)]
        var value = x[unsafe_offset=i]
        y[unsafe_offset=i] = other - value if (i & Int(mask)) != 0 else value + other


def hadamard_final(x: Pointer[Float32, MutAnyOrigin], signs: Pointer[Float32, MutAnyOrigin], y: Pointer[Float32, MutAnyOrigin], q: Pointer[UInt32, MutAnyOrigin], width: Int32, inverse: Int32):
    var i = global_idx.x * 2
    if i < Int(width):
        var a = x[unsafe_offset=i] * Float32(0.04419417382415922)
        var b = x[unsafe_offset=i+1] * Float32(0.04419417382415922)
        if inverse != 0:
            a *= signs[unsafe_offset=i]
            b *= signs[unsafe_offset=i+1]
        y[unsafe_offset=i] = a
        y[unsafe_offset=i+1] = b
        if inverse == 0:
            q[unsafe_offset=i//2] = bitcast[.uint32, 1](SIMD[.float32, 2](a, b).cast[.float16]())


def portable_linear(x: Pointer[UInt32, MutAnyOrigin], w: Pointer[UInt32, MutAnyOrigin], params: Pointer[Float16, MutAnyOrigin], partials: Pointer[Float32, MutAnyOrigin], width: Int32, reduction: Int32):
    var i = global_idx.x * 8
    if i + 8 <= Int(width):
        var value = SIMD[.float32, 8](0)
        for n in range(8):
            var group = block_idx.y * 8 + n
            if group < Int(reduction)//128:
                var scales = params.unsafe_load[width=8](group * Int(width) + i).cast[.float32]()
                value += _group_dot8(x, w, Int(width), i, group) * scales
        partials.unsafe_store[width=8](block_idx.y * Int(width) + i, value)


def portable_normalize_qk(x: Pointer[Float32, MutAnyOrigin], y: Pointer[Float32, MutAnyOrigin]):
    var i = global_idx.x
    if i < 6144:
        var value = x[unsafe_offset=i]
        if i < 4096:
            var base = (i // 128) * 128
            var ss: Float32 = 0
            for j in range(128):
                var v = x[unsafe_offset=base+j]
                ss += v*v
            value *= rsqrt(ss + Float32(1e-6))
            if i < 2048:
                value *= Float32(0.08838834764831845)
        y[unsafe_offset=i] = value


def portable_delta_update(qkv: Pointer[Float32, MutAnyOrigin], gates: Pointer[Float32, MutAnyOrigin], state: Pointer[Float32, MutAnyOrigin], y: Pointer[Float32, MutAnyOrigin]):
    var i = global_idx.x
    if i < 2048:
        var h = i // 128
        var d = i % 128
        var decay = gates[unsafe_offset=h*2]
        var beta = gates[unsafe_offset=h*2+1]
        var predicted: Float32 = 0
        for k in range(128):
            predicted += state[unsafe_offset=(h*128+k)*128+d] * decay * qkv[unsafe_offset=2048+h*128+k]
        var delta = (qkv[unsafe_offset=4096+i] - predicted) * beta
        var value: Float32 = 0
        for k in range(128):
            var off = (h*128+k)*128+d
            var updated = state[unsafe_offset=off]*decay + qkv[unsafe_offset=2048+h*128+k]*delta
            state[unsafe_offset=off] = updated
            value += updated * qkv[unsafe_offset=h*128+k]
        y[unsafe_offset=i] = value


def portable_gated_norm(x: Pointer[Float32, MutAnyOrigin], z: Pointer[Float32, MutAnyOrigin], w: Pointer[Float32, MutAnyOrigin], y: Pointer[Float32, MutAnyOrigin]):
    var i = global_idx.x
    if i < 2048:
        var base = (i//128)*128
        var ss: Float32 = 0
        for j in range(128):
            var v = x[unsafe_offset=base+j]
            ss += v*v
        var gate = z[unsafe_offset=i]
        y[unsafe_offset=i] = x[unsafe_offset=i] * rsqrt(ss/128 + Float32(1e-6)) * w[unsafe_offset=i%128] * gate * sigmoid(gate)


def portable_head_norm(x: Pointer[Float32, MutAnyOrigin], w: Pointer[Float32, MutAnyOrigin], y: Pointer[Float32, MutAnyOrigin], mode: Int32):
    var i = global_idx.x
    var width = 2048 if mode == 0 else 512
    if i < width:
        var h = i//256
        var base = h*(512 if mode == 0 else 256)
        var ss: Float32 = 0
        for j in range(256):
            var v = x[unsafe_offset=base+j]
            ss += v*v
        y[unsafe_offset=i] = x[unsafe_offset=base+i%256] * rsqrt(ss/256 + Float32(1e-6)) * w[unsafe_offset=i%256]
